#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""결정층 v1 백테스트: OOS 확률(calib_p2_walkforward.csv) → 3단계 상태기계 → 톤 → v0 와 같은 성적표 → results/ + docs/backtest_v1.html
(ARCHITECTURE_PHASE2.md §9·§13).

사용:
    python scripts/run_backtest_v1.py                       # --config default
    python scripts/run_backtest_v1.py --all-configs         # default + wide + symmetric_dwell + no_dwell (민감도는 보고만)
    python scripts/run_backtest_v1.py --results-dir results/tmp --docs-dir results/tmp/docs

흐름:
  results/calib_p2_walkforward.csv + summary_p2.json 을 읽어 spec_sha256·feature_rule 이 현재 코드와 같은지 확인(아니면 exit 1)
  → decision.run(**배포 단의 확률**, clim) → to_tone → evaluate.episode_eval / allocation_sim /
  directional_scorecard(방향은 참고만) → decision.kpis → results/backtest_v1.csv + results/summary_v1.json + docs/backtest_v1.html.

**헤드라인의 정의 (2026-09-08 정정)**
  backtest_v1.csv 의 p 열과 summary_v1.json 의 kpis/allocation/episodes/episodes_by_threshold/directional/flags,
  그리고 docs/backtest_v1.html 의 ②~⑤ 는 모두 **summary_p2.json.acceptance 가 배치한 단**(deploy_mode="tones" → tone_model;
  오늘은 M1)의 확률로 돌린 결정층이다. 그 전에는 배치와 무관하게 언제나 p_m3 를 헤드라인으로 실어, 배포되지 않은 모델의
  성적을 배포된 시스템의 성적처럼 읽게 만들었다(그 결함을 여기서 고친다).
  deploy_mode 가 "info_only" 면 배치된 단이 없으므로 생산 모델 M3 를 '정보 표시(배포 안 함)' 로 싣는다(톤·비중 주장 아님).
  배포되지 않은 단은 summary_v1.json["info_layers"][단] 과 리포트 ⑥ 에 '정보' 로만 남는다(m1_layer 키는 M1 이 배포되지
  않았을 때의 M1 정보층 = info_layers["M1"] 을 가리키는 옛 이름으로 유지).

원칙:
  * 홀드아웃 보호: SPY 종가·목표변수·에피소드는 OOS 마지막 세션(하드컷 끝)에서 자른다 — 결정층도 홀드아웃을 보지 않는다.
  * 결정층 상수는 config.DECISION_P2(사전 등록, 적합 파라미터 0). 민감도 3종은 보고만 하고 선택에 쓰지 않는다(§9).
  * deploy_mode/tone_model 은 summary_p2.json.acceptance 를 그대로 옮긴다 — 이 스크립트는 판정을 바꾸지 않는다.
  * 조용한 실패 금지: 산출물 불일치·열 누락은 예외(exit 1). 배치된 단의 확률 열이 없으면 exit 1(M3 로 조용히 대체하지 않는다).
    churn 경보·KPI 상한 위반은 summary_v1.json.flags 와 리포트에 표시.
종료 코드: 0 성공 · 1 실패(예외·spec 불일치·배치된 단의 확률 열 없음).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrl import decision as DEC                                           # noqa: E402
from mrl import evaluate as E                                             # noqa: E402
from mrl import features as F                                             # noqa: E402
from mrl import report as RPT                                             # noqa: E402
from mrl import targets as T                                              # noqa: E402
from mrl.config import (BLOCKS_24, DATA_DIR, DOCS_DIR, EPISODE_THRESHOLDS, HOLDOUT_START,   # noqa: E402
                        RESULTS_DIR, STATE_TO_TONE, TONE_EXPOSURE)
from mrl.data import load_cache                                           # noqa: E402

CALIB_CSV = "calib_p2_walkforward.csv"
SUMMARY_P2 = "summary_p2.json"
CSV_NAME = "backtest_v1.csv"
SUMMARY_NAME = "summary_v1.json"
HTML_NAME = "backtest_v1.html"
CSV_COLUMNS = ("p", "clim", "r", "state", "days_in_state", "tone", "changed")
CONFIGS = tuple(DEC.SENSITIVITY_NAMES)                                    # default, wide, symmetric_dwell, no_dwell
FLAG_CHURN = RPT.P2_CHURN_LABEL                                            # "결정층 잦은 전환"
INFO_RUNGS = ("M1", "M3")                                                 # 배포되지 않으면 정보층으로 함께 돌리는 단
INFO_RUNG = "M3"                                                          # 배치된 단이 없을 때(info_only) 정보로 싣는 생산 모델
DEPLOYED_LABEL = RPT.DEPLOYED_LABEL                                        # "배포"
INFO_LABEL = RPT.INFO_DISPLAY_LABEL                                        # "정보 표시(배포 안 함)"


def rung_column(rung: str) -> str:
    """사다리 단 이름 → calib_p2_walkforward.csv 의 확률 열 (M1 → p_m1)."""
    return f"p_{str(rung).lower()}"


def deployment_of(acc: dict) -> dict:
    """summary_p2.json.acceptance → {deploy_mode, tone_model, prob_rung, deployed, label, prob_column}.

    이 스크립트는 판정을 만들지 않고 옮기기만 한다: deploy_mode == "tones" 면 tone_model 의 확률이 헤드라인이고,
    아니면 배치된 단이 없으므로 생산 모델 M3 를 '정보 표시(배포 안 함)' 로 싣는다."""
    acc = acc if isinstance(acc, dict) else {}
    deploy = str(acc.get("deploy_mode") or "info_only")
    tone = acc.get("tone_model")
    tone = None if tone is None or str(tone).strip() == "" else str(tone)
    deployed = deploy == "tones" and bool(tone)
    if deploy == "tones" and not tone:
        warnings.warn("acceptance.deploy_mode 는 'tones' 인데 tone_model 이 비어 있음 → 배치된 단 없음(info_only)으로 다룬다")
        deploy = "info_only"
    rung = tone if deployed else INFO_RUNG
    return {"deploy_mode": deploy, "tone_model": tone, "prob_rung": rung, "deployed": deployed,
            "label": DEPLOYED_LABEL if deployed else INFO_LABEL, "prob_column": rung_column(rung)}


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _log(msg: str) -> None:
    print(msg, flush=True)


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dstr(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _messages(caught) -> list[str]:
    out: list[str] = []
    for w in caught:
        if issubclass(w.category, ResourceWarning):
            continue
        m = str(w.message)
        if m not in out:
            out.append(m)
    return out


def _pct(v) -> str:
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v * 100:.1f}%"


def _num(v, d=1) -> str:
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


# ------------------------------------------------------------------
# 입력
# ------------------------------------------------------------------
def load_calibration(results_dir: Path) -> tuple[pd.DataFrame, dict]:
    """calib_p2_walkforward.csv + summary_p2.json. spec_sha256·feature_rule 이 현재 코드와 다르면 SystemExit(1)."""
    csv, js = results_dir / CALIB_CSV, results_dir / SUMMARY_P2
    for p in (csv, js):
        if not p.exists():
            _log(f"::error::{p} 없음 — scripts/run_calibration.py 를 먼저 실행하라")
            raise SystemExit(1)
    summary = json.loads(js.read_text(encoding="utf-8"))
    run = summary.get("run") or {}
    sha_now, rule_now = F.spec_sha256(), F.FEATURE_RULE
    if run.get("spec_sha256") != sha_now or run.get("feature_rule") != rule_now:
        _log(f"::error::summary_p2.json 의 spec_sha256/feature_rule({str(run.get('spec_sha256'))[:12]!r}) 이 현재 코드({sha_now[:12]!r}) 와 다름 — "
             "run_calibration.py 를 다시 실행하라 (옛 확률로 결정층을 채점하지 않는다)")
        raise SystemExit(1)
    oos = pd.read_csv(csv, index_col="date", parse_dates=["date"], encoding="utf-8")
    need = {"y", "clim", "p_vix", "p_vix_bgk", "p_m1", "p_m3"}
    missing = sorted(need - set(oos.columns))
    if missing:
        raise RuntimeError(f"{csv.name} 에 열이 없음: {missing}")
    if oos.index.has_duplicates or not oos.index.is_monotonic_increasing:
        raise RuntimeError(f"{csv.name} 인덱스가 중복/비정렬")
    if (oos.index >= pd.Timestamp(HOLDOUT_START)).any() and not run.get("holdout_final"):
        raise RuntimeError(f"{csv.name} 에 홀드아웃({HOLDOUT_START}~) 행이 있음 — 하드컷 위반")
    return oos, summary


def load_v0(results_dir: Path) -> tuple[dict, pd.Series | None]:
    """summary_v0_completed.json (v0 줄·배분 비교) 와 backtest_v0_completed.csv 의 tone (누적수익 차트용)."""
    for d in (results_dir, RESULTS_DIR):
        js, csv = d / "summary_v0_completed.json", d / "backtest_v0_completed.csv"
        if js.exists():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = RPT.load_summary_v0(js)
            tone = None
            if csv.exists():
                rep = pd.read_csv(csv, index_col="date", parse_dates=["date"], encoding="utf-8")
                if "tone" in rep.columns:
                    tone = rep["tone"].astype(str)
            return s, tone
    warnings.warn("summary_v0_completed.json 없음 → v0 줄에 숫자 없음")
    return {}, None


# ------------------------------------------------------------------
# 결정층 한 구성 실행
# ------------------------------------------------------------------
def run_layer(p: pd.Series, clim: pd.Series, cfg: DEC.DecisionConfig, targets: pd.DataFrame, eps: dict, spy_close: pd.Series,
              label: str, warns: list[str], full: bool = True) -> dict:
    """decision.run → to_tone → kpis(+allocation) → episode_eval(5/10/20) → directional_scorecard. 반환 dict(states, replay, kpis, …)."""
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        st = DEC.run(p, clim, cfg)
    warns += [f"[{label}] 결정층: {m}" for m in _messages(caught)]
    warns += [f"[{label}] 결정층: {w}" for w in st.attrs.get("warnings", []) if w not in warns]
    tone = DEC.to_tone(st["state"])
    replay = pd.DataFrame({"tone": tone.to_numpy(), "state": st["state"].to_numpy(), "p": p.reindex(st.index).to_numpy(),
                           "clim": clim.reindex(st.index).to_numpy(), "r": st["r"].to_numpy(),
                           "days_in_state": st["days_in_state"].to_numpy(), "changed": st["changed"].to_numpy()}, index=st.index)
    y_dd = targets["y_dd5_20"].reindex(st.index)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kp = DEC.kpis(st, y_dd, spy_close)
    warns += [f"[{label}] KPI: {m}" for m in _messages(caught)]
    out = {"label": label, "config": cfg.as_dict(), "states": st, "replay": replay, "kpis": kp,
           "allocation": {k: v for k, v in (kp.get("allocation") or {}).items() if k != "series"},
           "runtime_sec": None}
    episodes = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for thr, ep_df in eps.items():
            if not full and thr != 0.10:
                continue
            tab, summ = E.episode_eval(replay[["tone"]], ep_df, targets)
            episodes[str(int(round(thr * 100)))] = {"threshold": thr, "table": tab, "summary": summ}
        directional = E.directional_scorecard(replay[["tone"]], targets) if full else None
    warns += [f"[{label}] 평가: {m}" for m in _messages(caught)]
    out["episodes"] = episodes
    out["directional"] = directional
    out["runtime_sec"] = round(time.perf_counter() - t0, 1)
    k = kp
    _log(f"[{label}] 전환 {k['switches_per_year']:.1f}회/년 · 경고 점유 {k['warn_share'] * 100:.1f}% "
         f"(caution {k['occupancy']['caution'] * 100:.1f}% · reduce {k['occupancy']['reduce'] * 100:.1f}%) · "
         f"경고 dd5 {_pct(k['dd5_rate_by_state'].get('caution'))}/{_pct(k['dd5_rate_by_state'].get('reduce'))} vs 정상 {_pct(k['dd5_rate_by_state'].get('normal'))} "
         f"(기저율 {_pct(k['base_rate'])}) · 진짜 경보 {_pct(k['true_alarm_share'])} · 5세션 최대 {k['max_changes_any_5_sessions']} · "
         f"churn 경보 {k['churn_alert_sessions']}세션 · CAGR {_pct(out['allocation'].get('cagr'))} vs 보유 {_pct(out['allocation'].get('bh_cagr'))} · "
         f"MaxDD {_pct(out['allocation'].get('max_dd'))} vs {_pct(out['allocation'].get('bh_max_dd'))} · {out['runtime_sec']}s")
    return out


def _layer_json(res: dict, full: bool = True) -> dict:
    """run_layer 결과 → JSON 직렬화 가능 dict (states/replay 제외)."""
    eps = {}
    for k, v in res["episodes"].items():
        eps[k] = {"threshold": v["threshold"], "table": E._jsonable(v["table"]), "summary": E._jsonable(v["summary"])}
    out = {"label": res["label"], "config": res["config"], "kpis": DEC.kpis_jsonable(res["kpis"]),
           "allocation": E._jsonable(res["allocation"]), "episodes_by_threshold": eps, "runtime_sec": res["runtime_sec"]}
    if full and res.get("directional") is not None:
        out["directional"] = E._jsonable(res["directional"])
    return out


def flags_of(kp: dict) -> list[str]:
    flags = []
    if kp.get("churn_alert_any"):
        flags.append(f"{FLAG_CHURN} — 직전 252세션 변경 > {kp.get('churn_alert_threshold')} 인 세션 {kp.get('churn_alert_sessions')}개 (최대 {kp.get('max_churn_252')})")
    if kp.get("kpi_ceiling_ok") is False:
        flags.append(f"KPI 상한 위반: 전환 {kp.get('switches_per_year'):.1f}회/년 > {kp.get('kpi_ceiling')}")
    if kp.get("structural_bound_ok") is False:
        flags.append(f"구조적 상한 위반: 5세션 변경 {kp.get('max_changes_any_5_sessions')} > {kp.get('structural_bound')}")
    return flags


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser(description="결정층 v1 백테스트 (OOS 확률 → 상태기계 → 톤 → 성적표)")
    ap.add_argument("--config", default="default", choices=list(CONFIGS), help="주 구성 (backtest_v1.csv·KPI 헤드라인)")
    ap.add_argument("--all-configs", action="store_true", help="민감도 3종(wide·symmetric_dwell·no_dwell)도 실행(보고만)")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--docs-dir", default=str(DOCS_DIR))
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args(argv)

    t_all = time.perf_counter()
    results_dir, docs_dir, data_dir = Path(args.results_dir), Path(args.docs_dir), Path(args.data_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    warns: list[str] = []

    oos, sp2 = load_calibration(results_dir)
    run_p2 = sp2.get("run") or {}
    acc = sp2.get("acceptance") or {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dep = deployment_of(acc)
    warns += [f"배치: {m}" for m in _messages(caught)]
    deploy_mode, tone_model = dep["deploy_mode"], dep["tone_model"]
    prob_rung, prob_col = dep["prob_rung"], dep["prob_column"]
    if prob_col not in oos.columns:
        _log(f"::error::배치된 단 {prob_rung} 의 확률 열 {prob_col} 이 {CALIB_CSV} 에 없음 — "
             "run_calibration.py 를 다시 실행하라 (배포되지 않은 단으로 조용히 대체하지 않는다)")
        raise SystemExit(1)
    _log(f"[run_backtest_v1] OOS {_dstr(oos.index[0])} ~ {_dstr(oos.index[-1])} ({len(oos):,}세션) · spec {str(run_p2.get('spec_sha256'))[:12]} · "
         f"deploy {deploy_mode} · tone_model {tone_model} (규칙 {acc.get('rule')}) · "
         f"헤드라인 = {prob_rung}({prob_col}) {dep['label']}")

    # SPY 종가·목표변수·에피소드 — OOS 끝에서 하드컷 (결정층도 홀드아웃을 보지 않는다)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle = load_cache(data_dir)
    for m in _messages(caught):
        _log(f"[캐시 경고] {m[:300]}")
    end = oos.index[-1]
    spy_close = bundle.spy_ohlc["Close"].astype(float).loc[:end].rename("SPY")
    if spy_close.index[-1] != end:
        raise RuntimeError(f"캐시 SPY 마지막 세션 {spy_close.index[-1]:%Y-%m-%d} ≠ OOS 끝 {end:%Y-%m-%d}")
    targets = T.make_targets(spy_close)
    eps = {thr: T.episodes(spy_close, thr, split=False) for thr in EPISODE_THRESHOLDS}
    # 라벨 정합성: CSV 의 y 와 캐시로 다시 만든 y_dd5_20 이 같아야 한다 (같은 하드컷·같은 종가)
    y_csv = oos["y"].astype(float)
    y_new = targets["y_dd5_20"].reindex(oos.index).astype(float)
    both = y_csv.notna() & y_new.notna()
    n_diff = int((y_csv[both] != y_new[both]).sum())
    if n_diff:
        raise RuntimeError(f"calib CSV 의 y 와 캐시 재계산 y_dd5_20 이 {n_diff}행 다름 — 캐시가 바뀌었다: run_calibration.py 를 다시 실행하라")
    _log(f"SPY {spy_close.index[0]:%Y-%m-%d} ~ {spy_close.index[-1]:%Y-%m-%d} ({len(spy_close):,}일, 하드컷) · 에피소드 "
         + " · ".join(f"≥{int(t * 100)}% {len(d)}회" for t, d in eps.items()))

    configs = list(CONFIGS) if args.all_configs else [args.config]
    if args.config not in configs:
        configs.insert(0, args.config)
    # 주 결정층 = 배치된 단의 확률(info_only 면 M3 정보). 민감도 3종도 같은 단으로 돌린다.
    results: dict[str, dict] = {}
    for name in configs:
        cfg = DEC.config_from_name(name)
        results[name] = run_layer(oos[prob_col], oos["clim"], cfg, targets, eps, spy_close, f"{prob_rung}·{name}", warns,
                                  full=(name == args.config))
    main_res = results[args.config]

    # 배포되지 않은 단 = 정보층(같은 구성). 톤·비중 주장이 아니라 비교용이다.
    info_layers: dict[str, dict] = {}
    for rung in INFO_RUNGS:
        col = rung_column(rung)
        if rung == prob_rung:
            continue
        if col not in oos.columns:
            warns.append(f"정보층 {rung}: {CALIB_CSV} 에 {col} 열이 없어 생략")
            continue
        info_layers[rung] = run_layer(oos[col], oos["clim"], DEC.config_from_name(args.config), targets, eps, spy_close,
                                      f"{rung}·{args.config} ({INFO_LABEL})", warns, full=True)

    # 산출: CSV
    rep = main_res["replay"]
    csv_out = rep[list(CSV_COLUMNS)].copy()
    csv_out["days_in_state"] = csv_out["days_in_state"].astype(int)
    csv_out["changed"] = csv_out["changed"].astype(bool)
    csv_path = results_dir / CSV_NAME
    csv_out.to_csv(csv_path, index_label="date", float_format="%.10g", lineterminator="\n", encoding="utf-8")

    v0_summary, v0_tone = load_v0(results_dir)
    v0h = RPT.v0_headline(v0_summary)
    kp = main_res["kpis"]
    flags = flags_of(kp)
    sens = {n: _layer_json(r, full=False) for n, r in results.items() if n != args.config}
    ep10 = main_res["episodes"].get("10", {})
    generated = _now_utc()
    summary = {
        "schema_version": 1,
        "headline_note": (f"kpis/allocation/episodes/directional/flags 와 backtest_v1.csv 의 p 는 배치된 단 {prob_rung}"
                          f"({prob_col}) 의 확률로 돌린 결정층 — {dep['label']}. "
                          "2026-09-08 이전에는 배치와 무관하게 p_m3 를 실었다(그 결함 정정). "
                          "배포되지 않은 단은 info_layers 에 정보로만 둔다."),
        "run": {"generated_at_utc": generated, "start": _dstr(rep.index[0]), "end": _dstr(rep.index[-1]), "n_sessions": int(len(rep)),
                "config": args.config, "configs_run": configs, "deploy_mode": deploy_mode, "tone_model": tone_model,
                "prob_rung": prob_rung, "prob_column": prob_col, "deployed": dep["deployed"],
                "acceptance_rule": acc.get("rule"), "spec_sha256": run_p2.get("spec_sha256"), "feature_rule": run_p2.get("feature_rule"),
                "calib_csv": str(results_dir / CALIB_CSV), "calib_generated_at_utc": run_p2.get("generated_at_utc"),
                "calib_oos_csv_sha256": run_p2.get("oos_csv_sha256"), "holdout_start": HOLDOUT_START, "hard_cut_end": _dstr(end),
                "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                "cache": {"fetched_at_utc": bundle.meta.get("fetched_at_utc"), "spy_last": bundle.meta.get("spy_last"),
                          "spy_last_used": _dstr(end)},
                "exposure_map": dict(TONE_EXPOSURE), "state_to_tone": dict(STATE_TO_TONE), "cost_bps": 5},
        "config_name": args.config, "config": main_res["config"], "deploy_mode": deploy_mode, "tone_model": tone_model,
        "deployed": {"prob_rung": prob_rung, "prob_column": prob_col, "deployed": dep["deployed"], "label": dep["label"],
                     "deploy_mode": deploy_mode, "tone_model": tone_model, "source": "summary_p2.json:acceptance"},
        "acceptance": acc, "v0_reference": E._jsonable(v0h),
        "kpis": DEC.kpis_jsonable(kp), "allocation": E._jsonable(main_res["allocation"]),
        "episodes": {"threshold": 0.10, "table": E._jsonable(ep10.get("table")), "summary": E._jsonable(ep10.get("summary"))},
        "episodes_by_threshold": _layer_json(main_res)["episodes_by_threshold"],
        "directional": E._jsonable(main_res["directional"]),
        "directional_note": "방향 적중률은 참고만 — 결정층은 방향을 예측하지 않는다(VALIDATION §0·§5)",
        "info_layers": {rung: {**_layer_json(r), "role": INFO_LABEL, "rung": rung, "prob_column": rung_column(rung)}
                        for rung, r in info_layers.items()},
        "info_layers_note": f"배포되지 않은 단을 같은 결정층 규칙으로 돌린 결과 — {INFO_LABEL}. 톤·비중 제안이 아니다.",
        "m1_layer": (_layer_json(info_layers["M1"]) if "M1" in info_layers else None),   # 옛 이름(M1 이 배포되면 헤드라인이 M1 이므로 null)
        "sensitivities": sens, "flags": flags, "warnings": warns,
        "artifacts": {"csv": str(csv_path), "summary_json": str(results_dir / SUMMARY_NAME), "report_html": str(docs_dir / HTML_NAME)},
        "total_runtime_sec": None,
    }

    charts: dict[str, bytes] = {}
    if not args.no_charts:
        cols = [c for c in ("y", "clim", "p_vix", "p_vix_bgk", "p_m1", "p_m2", "p_m3", prob_col) if c in oos.columns]
        frame = oos[list(dict.fromkeys(cols))].copy()
        frame["state"] = rep["state"]
        if v0_tone is not None:
            frame["v0_tone"] = v0_tone.reindex(frame.index)
        # 신뢰도·블록 표는 배치된 단의 것을 쓴다(다른 단의 표를 그 단의 이름으로 그리지 않는다)
        rel_key = {"M3": "reliability", "M1": "reliability_m1"}.get(prob_rung)
        rel = sp2.get(rel_key) if rel_key else None
        if rel is None:
            warns.append(f"차트: summary_p2.json 에 {prob_rung} 단의 신뢰도 표({rel_key or '없음'})가 없어 OOS 확률로 다시 계산")
        blocks24 = (sp2.get("blocks24_by_rung") or {}).get(prob_rung) or (sp2.get("blocks24") if prob_rung == INFO_RUNG else None)
        if blocks24 is None:
            warns.append(f"차트: summary_p2.json 에 {prob_rung} 단의 24개월 블록 표가 없어 OOS 확률로 다시 계산")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # 표는 단 이름을 붙여 넘긴다(charts_p2 가 배포 단의 표만 고르도록). rung_deployed 는 범례의 ', deployed' 여부.
            charts = RPT.charts_p2(frame, BLOCKS_24,
                                   {"blocks24_by_rung": {prob_rung: blocks24},
                                    ("reliability" if prob_rung == INFO_RUNG else f"reliability_{prob_rung.lower()}"): rel,
                                    "deployed_rung": prob_rung, "rung_deployed": dep["deployed"]}, spy_close)
        for m in _messages(caught):
            if any(k in m for k in ("coef_path", "ladder_ci", "era_auc")):
                continue                                          # 보정 리포트 전용 차트 — 여기서는 만들지 않는다
            warns.append(f"차트: {m}")
        for k in ("coef_path", "ladder_ci", "era_auc"):
            charts.pop(k, None)
    summary["total_runtime_sec"] = round(time.perf_counter() - t_all, 1)
    summary_json = E._jsonable(summary)
    with open(results_dir / SUMMARY_NAME, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=1, allow_nan=False)
        f.write("\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        RPT.render_backtest_v1(summary_json, v0_summary, docs_dir / HTML_NAME, charts)
    for m in _messages(caught):
        _log(f"[리포트 경고] {m[:300]}")

    a = main_res["allocation"]
    _log(f"[backtest_v1] {prob_rung}({dep['label']})·{args.config}: 전환 {kp['switches_per_year']:.1f}회/년 (v0 {_num(v0h.get('tone_switches_per_year'))}) · "
         f"경고 점유 {_pct(kp['warn_share'])} (v0 {_pct(v0h.get('warn_share'))}) · 진짜 경보 {_pct(kp['true_alarm_share'])} vs 기저율 "
         f"{_pct(kp['true_alarm_share_baseline'])} vs v0 {_pct(v0h.get('true_alarm_share'))} · 배분 CAGR {_pct(a.get('cagr'))} vs 보유 "
         f"{_pct(a.get('bh_cagr'))} (v0 {_pct(v0h.get('cagr_strategy'))} vs {_pct(v0h.get('cagr_bh'))}) · MaxDD {_pct(a.get('max_dd'))} vs "
         f"{_pct(a.get('bh_max_dd'))} (v0 {_pct(v0h.get('maxdd_strategy'))} vs {_pct(v0h.get('maxdd_bh'))}) · deploy {deploy_mode}"
         + (f" · 플래그 {flags}" if flags else ""))
    for rung, r in info_layers.items():
        ik, ia = r["kpis"], r["allocation"]
        _log(f"[backtest_v1] 정보층 {rung}({INFO_LABEL}): 전환 {ik['switches_per_year']:.1f}회/년 · 경고 점유 {_pct(ik['warn_share'])} · "
             f"진짜 경보 {_pct(ik['true_alarm_share'])} · CAGR {_pct(ia.get('cagr'))} · MaxDD {_pct(ia.get('max_dd'))} — 톤·비중 제안 아님")
    _log(f"[run_backtest_v1] 저장: {CSV_NAME}, {SUMMARY_NAME}, {HTML_NAME} · 경고 {len(warns)}건 · 총 {summary['total_runtime_sec']}s")
    for w in warns:
        _log(f"  - {w[:300]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
