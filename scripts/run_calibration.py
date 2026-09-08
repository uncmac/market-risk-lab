#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2 보정 모델 p2 — 특징 → walk-forward 사다리 → 벤치마크 → 검정 → HAR-RV → results/ + docs/calibration_p2.html
(ARCHITECTURE_PHASE2.md §13, VALIDATION.md §6 실험 #2).

사용:
    python scripts/run_calibration.py                                   # 기본: HOLDOUT_START 직전 세션까지 하드컷
    python scripts/run_calibration.py --end 2006-12-29 --first-refit 2003-01-02 --results-dir results/tmp --docs-dir results/tmp/docs
    python scripts/run_calibration.py --holdout-final                   # 홀드아웃 최종 검증 1회 (unlock 파일이 있으면 exit 2)

흐름 (계약 §13):
  load_cache → apply_guards → 완성 봉 자르기 → **하드컷(--end; 모든 입력 프레임)** → build_features → make_targets → episodes(0.20)
  → first_refit_ok(assert) → walk_forward(사다리 M0~M3 + 소거 M3-PK·M3-HAR96 + C 민감도 + 1999 시작 민감도)
  → block_scores(24·18·1999) → ladder_table → reliability/murphy/era_auc → acceptance(literal + #2a amended)
  → har_walk_forward + vol_scorecard → v0_reference(창 규칙·signals_v0 해시가 현재 코드와 같을 때만)
  → 라이브 계수 재적합 + 결정론 검사 → results/·docs/ 산출.

산출:
    results/calib_p2_walkforward.csv   date, y, clim, p_vix, p_vix_driftless, p_vix_bgk, p_m1, p_m2, p_m3, p_m3_pk, p_m3_har96,
                                       p_v0ref, har_fc_20, refit_year, block24, block18
    results/summary_p2.json            run, v0_reference, rungs, ladder, blocks24/18/from1999, reliability, murphy, era_auc,
                                       acceptance, params_by_refit, live_models, ablations, c_sensitivity, har, selftest,
                                       determinism, warnings, disclosure="#2 사전 관측 참조" (+ holdout: --holdout-final 뒤)
    results/model_p2.json              라이브 계수(4개) + deploy_mode/tone_model = acceptance 결과
    docs/calibration_p2.html           주간 리포트 ①~⑨ (report.render_calibration_report)

원칙:
  * 홀드아웃 하드컷: --holdout-final 없이는 bundle.close·spy_ohlc·cboe(·fg·eod)·targets 를 HOLDOUT_START 직전 세션에서 **데이터 자체**를
    자른다(라벨 마스크가 아니다). --end 가 HOLDOUT_START 이후를 가리키면 실행을 거부한다(exit 1).
  * 파라미터 예산: walk_forward 가 매 재적합에서 M3 의 n_params == PARAM_COUNT(4) 를 assert 한다.
  * 결정론: 난수는 전부 seed 0. 새로 적합한 라이브 계수와 기존 results/model_p2.json 의 계수 차가 1e-9 를 넘으면(같은 spec_sha256·
    같은 재적합일) exit 1 — 파일을 쓰지 않는다(주간 작업은 커밋 단계에 이르지 못한다). spec_sha256 이 다르면(코드 변경) 새 파일을 쓰되
    run.spec_changed=true 와 장부 기재 요구를 로그한다.
  * --holdout-final: results/holdout_unlock.json 이 있으면 즉시 exit 2. 없으면 하드컷 없이 2024-09-03~ 를 R_2024·R_2025·R_2026
    재적합(각각 자기 재적합일 전 퍼지 자료로만 학습)으로 채점해 summary_p2.json.holdout 에 기록하고 unlock 파일을 쓴다. --force 류 없음.
  * 조용한 실패 금지: v0 참조선만 조건 미충족 시 생략(+경고)하고, 그 밖의 단계는 예외로 죽는다. 모든 표에 n_blocks(=n//20) 병기.
  * --acceptance-rule 은 §13 #2a 의 소유자 선택 (a) literal / (b) amended 를 기록하는 스위치다(기본 literal = 사전 등록).
    amended 로 바꾸는 것은 post hoc 이며 장부 #2a 기재가 필요하다 — run.acceptance_rule 에 남는다.
종료 코드: 0 성공 · 1 실패(예외·결정론 검사 실패·홀드아웃 침범) · 2 홀드아웃 unlock 파일이 이미 존재(--holdout-final 재실행 거부).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrl import calendar_us as CAL                                        # noqa: E402
from mrl import calibrate as C                                            # noqa: E402
from mrl import evaluate as E                                             # noqa: E402
from mrl import features as F                                             # noqa: E402
from mrl import ledger as L                                               # noqa: E402
from mrl import model as M                                                # noqa: E402
from mrl import report as RPT                                             # noqa: E402
from mrl import signals_v0 as S                                           # noqa: E402
from mrl import targets as T                                              # noqa: E402
from mrl import vol as V                                                  # noqa: E402
from mrl.config import (BLOCKS_18, BLOCKS_24, BLOCKS_24_FROM_1999, DATA_DIR, DOCS_DIR,   # noqa: E402
                        HOLDOUT_START, HOLDOUT_UNLOCK_PATH, MODEL_P2_PATH, P2, RESULTS_DIR)
from mrl.data import load_cache                                           # noqa: E402

CSV_NAME = "calib_p2_walkforward.csv"
SUMMARY_NAME = "summary_p2.json"
MODEL_NAME = MODEL_P2_PATH.name                    # model_p2.json
UNLOCK_NAME = HOLDOUT_UNLOCK_PATH.name             # holdout_unlock.json
HTML_NAME = "calibration_p2.html"
OOS_CSV_COLUMNS = ("y", "clim", "p_vix", "p_vix_driftless", "p_vix_bgk", "p_m1", "p_m2", "p_m3", "p_m3_pk", "p_m3_har96",
                   "p_v0ref", "har_fc_20", "refit_year", "block24", "block18")
COEF_TOL = 1e-9                                    # 결정론 허용오차 (§2) — 같은 입력 지문일 때만 적용
COEF_DRIFT_MATERIAL = 1e-4                         # 입력 지문이 바뀐 경우: 이보다 큰 계수 이동은 '데이터 수정' 의심으로 경고(실패 아님)
# 근거(실측 2026-09-08): Yahoo 전량 재다운로드(daily/weekly 의 정상 경로)만으로 SPY 조정 OHLC 가 상대 ~1.5e-6 움직이고(배당 조정계수 반올림,
# 2025-12-15 이전 전 행) 라이브 계수는 최대 ~9e-6 이동한다. 6자리 CSV 반올림만의 이동은 ~2e-8. 한 행이라도 손으로 고친 자료는 계수를
# 1e-3 이상 움직이므로 1e-4 는 정상 재조정(≤1e-5)과 실제 수정(≥1e-3)을 두 자릿수 여유로 가른다. 1e-6 이면 매주 경고가 나 경보가 무뎌진다.
C_SENSITIVITY = (0.1, 10.0)                        # C 민감도 (기본 1.0 은 주 실행)
EXIT_UNLOCK_EXISTS = 2
DISCLOSURE = "#2 사전 관측 참조"
PRIMARY_RUNGS = ("M1", "M2", "M3")
ABLATION_LADDER = {"M3-PK": ("x_vix", "x_har_pk", "x_ma"), "M3-HAR96": tuple(P2["features"])}
SCORE_KEYS = ("n", "n_blocks", "n_pos", "base", "mean_p", "brier", "brier_clim", "brier_vix", "brier_vix_bgk", "brier_m1",
              "bss_clim", "bss_vix", "bss_vix_bgk", "bss_m1", "auc", "calib_in_large")


# ------------------------------------------------------------------
# 유틸
# ------------------------------------------------------------------
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
    """catch_warnings(record=True) 로 잡은 경고 → 문구 목록(순서 유지·중복 제거). ResourceWarning 만 제외."""
    out: list[str] = []
    for w in caught:
        if issubclass(w.category, ResourceWarning):
            continue
        m = str(w.message)
        if m not in out:
            out.append(m)
    return out


def _git_sha() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return None
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _signals_hash() -> str:
    """mrl/signals_v0.py 의 SHA-256 (CRLF→LF 정규화) — run_backtest_v0.py 와 같은 정의."""
    raw = Path(S.__file__).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _data_sha256(ohlc: pd.DataFrame, vix: pd.Series) -> str:
    """하드컷된 입력(SPY OHLC + 정렬된 VIX)의 지문 — 결정론 검사 실패 시 '데이터가 바뀐 것'과 '코드가 비결정적인 것'을 구분한다."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(ohlc[list(V.OHLC_COLUMNS)].to_numpy(dtype=float)).tobytes())
    h.update(np.ascontiguousarray(vix.to_numpy(dtype=float)).tobytes())
    h.update(ohlc.index[-1].strftime("%Y-%m-%d").encode())
    return h.hexdigest()


def _records(df) -> list[dict]:
    return E._jsonable(df) if isinstance(df, pd.DataFrame) else []


def _all_row(bs: pd.DataFrame) -> dict:
    """block_scores 의 'all' 행 → dict (SCORE_KEYS 만)."""
    row = bs[bs["block"].astype(str) == "all"]
    if len(row) == 0:
        return {k: float("nan") for k in SCORE_KEYS}
    r = row.iloc[0].to_dict()
    return {k: r.get(k, float("nan")) for k in SCORE_KEYS}


def _block_stats(bs: pd.DataFrame) -> dict:
    """블록별 요약: 평균 BSS(기후학·B1·BGK·M1), >0/≥0 블록 수, 실패 블록 목록 (n>0 블록만)."""
    blk = bs[(bs["block"].astype(str) != "all") & (bs["n"] > 0)]
    n_tab = int((bs["block"].astype(str) != "all").sum())

    def _mean(col):
        v = blk[col].astype(float).dropna()
        return float(v.mean()) if len(v) else float("nan")

    return {
        "n_blocks_table": n_tab, "n_blocks_scored": int(len(blk)),
        "mean_bss_clim": _mean("bss_clim"), "n_pos_clim": int((blk["bss_clim"] > 0).sum()),
        "failing_clim": [int(b) for b in blk.loc[~(blk["bss_clim"] > 0), "block"]],
        "mean_bss_vix": _mean("bss_vix"), "n_nonneg_vix": int((blk["bss_vix"] >= 0).sum()),
        "failing_vix": [int(b) for b in blk.loc[~(blk["bss_vix"] >= 0), "block"]],
        "mean_bss_vix_bgk": _mean("bss_vix_bgk"), "n_nonneg_bgk": int((blk["bss_vix_bgk"] >= 0).sum()),
        "mean_bss_m1": _mean("bss_m1"), "n_pos_m1": int((blk["bss_m1"] > 0).sum()),
        "min_bss_clim": float(blk["bss_clim"].min()) if len(blk) else float("nan"),
        "min_bss_vix": float(blk["bss_vix"].min()) if len(blk) else float("nan"),
    }


# ------------------------------------------------------------------
# 입력 준비 (완성 봉 · 하드컷)
# ------------------------------------------------------------------
def completed_last(bundle) -> pd.Timestamp:
    """캐시의 마지막 **완성** SPY 세션 (meta.spy_last_bar_complete=False 또는 지금 ET 기준 미완성이면 그 전 세션)."""
    spy_idx = bundle.spy_ohlc.index
    if len(spy_idx) < 2:
        raise RuntimeError("SPY 캐시에 세션이 2개 미만 — scripts/build_cache.py 로 재구축하라")
    last_ok = spy_idx[-1]
    if bundle.meta.get("spy_last_bar_complete") is False:
        last_ok = spy_idx[-2]
    completed = CAL.completed_daily(bundle.spy_ohlc)
    if len(completed) == 0:
        raise RuntimeError("완성된 SPY 일봉이 없음")
    return min(last_ok, completed.index[-1])


def resolve_end(bundle, end_arg: str | None, holdout_final: bool) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """(하드컷 끝 세션, 홀드아웃 직전 세션, 마지막 완성 세션). --end 가 HOLDOUT_START 이후면 SystemExit(1) — 하드컷은 협상 대상이 아니다."""
    spy_idx = bundle.spy_ohlc.index
    last_ok = completed_last(bundle)
    holdout_ts = pd.Timestamp(HOLDOUT_START)
    pre = spy_idx[spy_idx < holdout_ts]
    if len(pre) == 0:
        raise RuntimeError(f"HOLDOUT_START({HOLDOUT_START}) 이전 세션이 없음")
    pre_last = min(pre[-1], last_ok)
    if end_arg is None:
        return pre_last, pre_last, last_ok
    ts = pd.Timestamp(end_arg)
    if pd.isna(ts):
        raise ValueError(f"--end 해석 불가: {end_arg!r}")
    if ts >= holdout_ts and not holdout_final:
        _log(f"::error::--end {end_arg} 은 HOLDOUT_START({HOLDOUT_START}) 이후 — 홀드아웃은 --holdout-final(1회) 로만 접근한다")
        raise SystemExit(1)
    cand = spy_idx[spy_idx <= min(ts, last_ok)]
    if len(cand) == 0:
        raise ValueError(f"--end {end_arg} 이하에 세션이 없음")
    return cand[-1], pre_last, last_ok


def cut_bundle(bundle, end: pd.Timestamp):
    """모든 입력 프레임을 end(포함)에서 자른 새 Bundle — 라벨 마스크가 아니라 데이터 자체를 자른다."""
    e = pd.Timestamp(end)
    meta = dict(bundle.meta)
    meta["hard_cut_end"] = _dstr(e)
    return replace(bundle, close=bundle.close.loc[:e].copy(), spy_ohlc=bundle.spy_ohlc.loc[:e].copy(),
                   cboe=bundle.cboe.loc[:e].copy(), fg=bundle.fg.loc[:e].copy(), eod=bundle.eod.loc[:e].copy(), meta=meta)


def prepare_inputs(bundle_cut, end: pd.Timestamp) -> dict:
    """특징·목표변수·에피소드 (하드컷된 번들 기준)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        feats = F.build_features(bundle_cut, asof=end)
    fw = _messages(caught)
    spy_close = bundle_cut.spy_ohlc["Close"].astype(float).rename("SPY")
    targets = T.make_targets(spy_close)
    ep20 = T.episodes(spy_close, 0.20, split=False)
    if not feats.index.equals(spy_close.index):
        raise RuntimeError("특징 인덱스와 SPY 인덱스가 다름")
    return {"feats": feats, "targets": targets, "y": targets["y_dd5_20"], "spy_close": spy_close, "episodes20": ep20,
            "ohlc": bundle_cut.spy_ohlc, "feature_warnings": fw + list(feats.attrs.get("warnings", []) or [])}


# ------------------------------------------------------------------
# 사다리 · 소거 · 민감도
# ------------------------------------------------------------------
def parkinson_x_har(feats: pd.DataFrame, ohlc: pd.DataFrame) -> pd.Series:
    """소거 M3-PK: GK+OV 대신 Parkinson 분산으로 x_har (같은 HAR 창·가중·하한)."""
    var_pk = V.parkinson_variance(ohlc, floor=P2["var_floor"])
    comp = V.har_components(var_pk, lookbacks=P2["har_lookbacks"], floor=P2["var_floor"])
    ln_har = V.har_log_vol(comp, weights=P2["har_weights"])
    with np.errstate(divide="ignore", invalid="ignore"):
        x = ln_har - np.log(feats["vix"].astype(float) / 100.0)
    x.name = "x_har_pk"
    return x.reindex(feats.index)


def wf_end_of(end: pd.Timestamp) -> str:
    """walk_forward 의 end(배타) = min(HOLDOUT_START, end+1일)."""
    return _dstr(min(pd.Timestamp(HOLDOUT_START), pd.Timestamp(end) + pd.Timedelta(days=1)))


def run_ladder(inp: dict, first_refit: str, wf_end: str, sha: str, rule: str, warns: list[str]) -> dict:
    """주 사다리 + 소거(M3-PK·M3-HAR96) + C 민감도 + 1999 시작 민감도. 반환 dict(oos, params, ...)."""
    feats, y, ohlc = inp["feats"], inp["y"], inp["ohlc"]
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        oos, params = C.walk_forward(feats, y, ladder=M.LADDER, first_refit=first_refit, end=wf_end)
    warns += [f"walk_forward: {m}" for m in _messages(caught)]
    warns += [f"walk_forward: {w}" for w in oos.attrs.get("warnings", []) if not str(w).startswith("features:")]
    n_refit = oos.attrs.get("n_refits")
    _log(f"[ladder] 주 사다리 M1~M3 · 재적합 {n_refit}회 · OOS {len(oos):,}세션 · {time.perf_counter() - t0:.1f}s")

    # 소거 M3-PK
    feats_pk = feats.copy()
    feats_pk["x_har_pk"] = parkinson_x_har(feats, ohlc)
    feats_pk.attrs = dict(feats.attrs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        oos_pk, params_pk = C.walk_forward(feats_pk, y, ladder={"M3-PK": ABLATION_LADDER["M3-PK"]}, first_refit=first_refit, end=wf_end)
        oos_h96, params_h96 = C.walk_forward(feats, y, ladder={"M3-HAR96": ABLATION_LADDER["M3-HAR96"]}, first_refit=first_refit,
                                             end=wf_end, train_start=P2["har_train_start_sensitivity"])
    warns += [f"소거: {m}" for m in _messages(caught)]
    oos["p_m3_pk"] = oos_pk["p_m3_pk"].reindex(oos.index).to_numpy()
    oos["p_m3_har96"] = oos_h96["p_m3_har96"].reindex(oos.index).to_numpy()

    # C 민감도 (M3 만)
    c_runs = {}
    for c in C_SENSITIVITY:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            o_c, p_c = C.walk_forward(feats, y, ladder={"M3": M.LADDER["M3"]}, first_refit=first_refit, end=wf_end, C=c)
        warns += [f"C={c:g}: {m}" for m in _messages(caught)]
        o_c["p_m1"] = oos["p_m1"].reindex(o_c.index).to_numpy()
        c_runs[f"{c:g}"] = {"oos": o_c, "params": p_c}
    _log(f"[ladder] 소거 M3-PK·M3-HAR96 + C∈{C_SENSITIVITY} · {time.perf_counter() - t0:.1f}s 누적")

    # 1999 시작 민감도 (규칙 미충족 — 2000~02 약세장을 OOS 로)
    sens99 = None
    first_sens = P2["first_refit_sensitivity"]
    if first_refit != first_sens and pd.Timestamp(first_sens) in feats.index and pd.Timestamp(first_sens) < pd.Timestamp(wf_end):
        ok99, info99 = C.first_refit_ok(feats.index, y, inp["episodes20"], first_sens)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            oos99, params99 = C.walk_forward(feats, y, ladder=M.LADDER, first_refit=first_sens, end=wf_end)
        warns += [f"1999 민감도: {m}" for m in _messages(caught)]
        sens99 = {"first_refit": first_sens, "rule_ok": bool(ok99), "rule": info99, "oos": oos99, "params": params99}
        _log(f"[ladder] 1999-01-04 민감도 · 규칙 {'충족' if ok99 else '미충족(예상)'} · 학습 {info99['n_rows']:,}행 · OOS {len(oos99):,}세션")
    return {"oos": oos, "params": params, "oos_pk": oos_pk, "params_pk": params_pk, "oos_h96": oos_h96, "params_h96": params_h96,
            "c_runs": c_runs, "sens99": sens99, "ladder_runtime_sec": round(time.perf_counter() - t0, 1)}


# ------------------------------------------------------------------
# 채점
# ------------------------------------------------------------------
def score_ladder(lad: dict, primary_blocks, warns: list[str]) -> dict:
    """블록 표(24·18·1999) · 사다리 표 · 신뢰도 · Murphy · 소거/민감도 요약."""
    oos = lad["oos"]
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bs24 = {r: C.block_scores(oos, BLOCKS_24, C.rung_col(r)) for r in PRIMARY_RUNGS}
        bs18 = {r: C.block_scores(oos, BLOCKS_18, C.rung_col(r)) for r in PRIMARY_RUNGS}
        rungs = C.rung_table(oos, ("M0", "M1", "M2", "M3"))
        ladder = C.ladder_table(oos, primary_blocks)
        rel = C.reliability_table(oos["p_m3"], oos["y"])
        mur = C.murphy_decomposition(oos["p_m3"], oos["y"])
        rel_m1 = C.reliability_table(oos["p_m1"], oos["y"])
        mur_m1 = C.murphy_decomposition(oos["p_m1"], oos["y"])      # 배포 단이 M1 일 때 ④ 가 쓸 분해
    msgs = _messages(caught)
    warns += [f"채점: {m}" for m in msgs if "OOS 세션이 없습니다" not in m]
    n_empty = sum("OOS 세션이 없습니다" in m for m in msgs)
    if n_empty:
        warns.append(f"채점: 빈 블록 {n_empty}개(짧은 --end) — 표에 n=0·결측(—)으로 남김")
    # 소거·민감도 (블록 24 + 전체)
    abl, abl_detail = {}, {}
    for name, key, params in (("M3-PK", "oos_pk", "params_pk"), ("M3-HAR96", "oos_h96", "params_h96")):
        o = lad[key].copy()
        o["p_m1"] = oos["p_m1"].reindex(o.index).to_numpy()          # M1 대비 skill 은 주 사다리의 M1 과 같은 행에서
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bs = C.block_scores(o, BLOCKS_24, C.rung_col(name))
        # 리포트 ⑦ 이 표 한 행으로 렌더링하므로 스칼라만 — 블록 표·계수 경로는 *_detail 에 따로
        abl[name] = {"description": M.ABLATIONS.get(name, ""), **_flat_scores(bs)}
        abl_detail[name] = {"blocks": _block_stats(bs), "block_table": bs, "params": lad[params]}
    csens, csens_detail = {}, {}
    for c, run in lad["c_runs"].items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bs = C.block_scores(run["oos"], BLOCKS_24, "p_m3")
        csens[c] = _flat_scores(bs)
        csens_detail[c] = {"blocks": _block_stats(bs), "params": run["params"]}
    csens["1"] = _flat_scores(bs24["M3"])
    csens_detail["1"] = {"blocks": _block_stats(bs24["M3"]), "params": [p for p in lad["params"] if p["rung"] == "M3"]}
    csens = {k: csens[k] for k in sorted(csens, key=float)}
    csens_detail = {k: csens_detail[k] for k in sorted(csens_detail, key=float)}
    bs99 = None
    if lad["sens99"] is not None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bs99 = {r: C.block_scores(lad["sens99"]["oos"], BLOCKS_24_FROM_1999, C.rung_col(r)) for r in PRIMARY_RUNGS}
        warns += [f"1999 민감도 채점: {m}" for m in _messages(caught) if "OOS 세션이 없습니다" not in m]
    _log(f"[score] 블록 24/18/1999 · 사다리 {len(ladder)}행 · 신뢰도 · 소거 · {time.perf_counter() - t0:.1f}s")
    return {"bs24": bs24, "bs18": bs18, "bs99": bs99, "rungs": rungs, "ladder": ladder, "reliability": rel, "reliability_m1": rel_m1,
            "murphy": mur, "murphy_m1": mur_m1, "ablations": abl, "ablation_detail": abl_detail, "c_sensitivity": csens, "c_sensitivity_detail": csens_detail,
            "score_runtime_sec": round(time.perf_counter() - t0, 1)}


def _flat_scores(bs: pd.DataFrame) -> dict:
    """'all' 행 채점 + 블록 요약을 **스칼라만**으로 (리포트 표 한 행). 실패 블록 목록은 문자열."""
    st = _block_stats(bs)
    out = dict(_all_row(bs))
    out.update({"blocks_clim_pos": f"{st['n_pos_clim']}/{st['n_blocks_table']}", "blocks_vix_nonneg": f"{st['n_nonneg_vix']}/{st['n_blocks_table']}",
                "blocks_m1_pos": f"{st['n_pos_m1']}/{st['n_blocks_table']}", "mean_bss_clim_blocks": st["mean_bss_clim"],
                "mean_bss_vix_blocks": st["mean_bss_vix"], "failing_clim": ",".join(str(b) for b in st["failing_clim"]) or "-",
                "failing_vix": ",".join(str(b) for b in st["failing_vix"]) or "-"})
    return out


# ------------------------------------------------------------------
# HAR-RV 보조 출력
# ------------------------------------------------------------------
def har_block(inp: dict, first_refit: str, wf_end: str, end: pd.Timestamp, train_start: str, warns: list[str]) -> dict:
    feats, targets, ohlc, close = inp["feats"], inp["targets"], inp["ohlc"], inp["spy_close"]
    t0 = time.perf_counter()
    var = V.garman_klass_variance(ohlc, overnight=True, floor=P2["var_floor"])
    comp = V.har_components(var, lookbacks=P2["har_lookbacks"], floor=P2["var_floor"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        y_ln = V.har_target(close, h=P2["h"])
        rds = C.refit_dates(feats.index, first_refit, end=wf_end)
        ln_fc, tbl = V.har_walk_forward(comp, y_ln, rds, purge=P2["purge"], train_start=train_start)
        sc = V.vol_scorecard(ln_fc, y_ln, feats["vix"], comp["rv22"], BLOCKS_24, y_vol=targets["y_vol_20"])
        sens_start = P2["har_train_start_sensitivity"] if train_start != P2["har_train_start_sensitivity"] else "1993-03-03"
        ln_fc_s, tbl_s = V.har_walk_forward(comp, y_ln, rds, purge=P2["purge"], train_start=sens_start)
        sc_s = V.vol_scorecard(ln_fc_s, y_ln, feats["vix"], comp["rv22"], BLOCKS_24, y_vol=targets["y_vol_20"])
        _, tbl_live = V.har_walk_forward(comp, y_ln, [end], purge=P2["purge"], train_start=train_start)
    warns += [f"HAR: {m}" for m in _messages(caught)]
    live = tbl_live.iloc[-1]
    live_coef = {k: float(live[k]) for k in V.HAR_COEF_NAMES}
    oos_mask = (ln_fc.index >= pd.Timestamp(first_refit)) & np.isfinite(ln_fc.to_numpy()) & np.isfinite(y_ln.to_numpy())
    err = (ln_fc[oos_mask] - y_ln[oos_mask]).to_numpy()
    mae = float(np.mean(np.abs(err))) if len(err) else float("nan")
    _log(f"[har] 재적합 {len(tbl)}회 · OOS log-MAE {mae:.3f} · 라이브 계수 {live_coef} · {time.perf_counter() - t0:.1f}s")
    return {"ln_fc": ln_fc, "y_ln": y_ln, "scorecard": sc, "scorecard_sensitivity": sc_s, "sensitivity_train_start": sens_start,
            "coef_path": tbl, "coef_path_sensitivity": tbl_s, "live_coef": live_coef,
            "live_fit": {"refit_date": _dstr(end), "n": int(live["n"]), "train_start": str(live["train_start"]),
                         "train_end": str(live["train_end"])},
            "oos_log_mae": mae, "n_oos": int(oos_mask.sum()), "n_blocks": int(oos_mask.sum()) // P2["h"], "train_start": train_start,
            "definition": "ln RV20_fwd ~ 1 + ln rv1 + ln rv5 + ln rv22 (OLS 4, 예산 밖, 확률에 결합 금지); 연 1회 재적합·20일 퍼지",
            "warnings": []}


def har_chart(har: dict, vix: pd.Series, first_refit: str) -> bytes | None:
    try:
        df = pd.DataFrame({"rv": np.exp(har["y_ln"]), "fc": np.exp(har["ln_fc"]), "vix": vix.astype(float) / 100.0})
        df = df.loc[pd.Timestamp(first_refit):].dropna()
        if len(df) < 3:
            return None
        fig, ax = RPT._new_fig(9.0, 3.6)
        ax.plot(df.index, df["rv"] * 100, color=RPT.PALETTE["mut"], lw=0.8, label="realized RV20 (fwd)")
        ax.plot(df.index, df["fc"] * 100, color="#22c55e", lw=0.9, label="HAR forecast")
        ax.plot(df.index, df["vix"] * 100, color="#60a5fa", lw=0.7, alpha=0.8, label="VIX")
        ax.set_ylabel("annualized vol (%)")
        ax.set_title("HAR-RV forecast vs realized vs VIX (OOS, annual refit)")
        RPT._legend(ax, loc="upper left")
        return RPT._png(fig)
    except Exception as e:                                          # noqa: BLE001 — 차트 실패는 경고로만
        warnings.warn(f"HAR 차트 실패: {type(e).__name__}: {e}")
        return None


# ------------------------------------------------------------------
# v0 참조선 (예산 밖·모델 밖)
# ------------------------------------------------------------------
def _v0_paths(results_dir: Path) -> tuple[Path, Path] | None:
    for d in (results_dir, RESULTS_DIR):
        csv, js = d / "backtest_v0_completed.csv", d / "summary_v0_completed.json"
        if csv.exists() and js.exists():
            return csv, js
    return None


def v0_reference_block(results_dir: Path, oos: pd.DataFrame, y: pd.Series, rds: list, warns: list[str]) -> tuple[dict, pd.Series | None]:
    """v0 종합점수 Platt 참조선 — results/backtest_v0_completed.csv 의 run.window_rule·signals_v0_sha256 이 현재 코드와 같을 때만."""
    paths = _v0_paths(results_dir)
    if paths is None:
        warns.append("v0 참조선 생략: backtest_v0_completed.csv / summary_v0_completed.json 없음")
        return {"available": False, "reason": "v0 completed 산출물 없음"}, None
    csv, js = paths
    try:
        prev = json.loads(js.read_text(encoding="utf-8")).get("run", {}) or {}
    except (OSError, ValueError) as e:
        warns.append(f"v0 참조선 생략: {js.name} 읽기 실패({type(e).__name__})")
        return {"available": False, "reason": f"{js.name} 읽기 실패"}, None
    cur_hash = _signals_hash()
    if prev.get("window_rule") != S.WINDOW_RULE or prev.get("signals_v0_sha256") != cur_hash:
        warns.append(f"v0 참조선 생략: {js.name} 의 창 규칙/해시({str(prev.get('window_rule'))!r}, {str(prev.get('signals_v0_sha256'))[:12]!r}) "
                     f"≠ 현재 코드({S.WINDOW_RULE!r}, {cur_hash[:12]!r})")
        return {"available": False, "reason": "창 규칙·signals_v0 해시 불일치", "window_rule_prev": prev.get("window_rule"),
                "signals_v0_sha256_prev": prev.get("signals_v0_sha256")}, None
    replay = pd.read_csv(csv, index_col="date", parse_dates=["date"], encoding="utf-8")
    replay = replay.loc[:oos.index[-1]]                        # 홀드아웃 보호: OOS 끝에서 자른다
    if len(replay) == 0:
        warns.append("v0 참조선 생략: replay 가 OOS 구간과 겹치지 않음")
        return {"available": False, "reason": "replay 구간 불일치"}, None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        s = C.v0_reference(replay, y.reindex(replay.index), rds)
    warns += [f"v0 참조선: {m}" for m in _messages(caught)]
    warns += [f"v0 참조선: {w}" for w in s.attrs.get("warnings", [])]
    p = s.reindex(oos.index)
    fin = np.isfinite(p.to_numpy())
    out = {"available": bool(fin.any()), "first_refit": s.attrs.get("first"), "window_rule": S.WINDOW_RULE, "signals_v0_sha256": cur_hash,
           "params": s.attrs.get("params", []), "n": int(fin.sum()), "n_blocks": int(fin.sum()) // 20,
           "source_csv": str(csv), "replay_end_used": _dstr(replay.index[-1])}
    if fin.any():
        o = oos[fin].copy()
        o["p_v0ref"] = p[fin].to_numpy()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bs = C.block_scores(o, [(_dstr(o.index[0]), _dstr(o.index[-1] + pd.Timedelta(days=1)))], "p_v0ref")
            bs_m3 = C.block_scores(o, [(_dstr(o.index[0]), _dstr(o.index[-1] + pd.Timedelta(days=1)))], "p_m3")
            ci = C.loss_diff_ci(o["p_v0ref"], o["p_m3"], o["y"])
        out.update({k: v for k, v in _all_row(bs).items()})
        out["m3_same_rows"] = {k: v for k, v in _all_row(bs_m3).items() if k in ("brier", "bss_clim", "bss_vix", "auc")}
        out["v0ref_to_m3"] = ci
        out["note"] = ("v0 종합 s_t=-(score_d+score_w+score_m)/3 에 Platt(2) — 참조선(예산 밖, 모델 밖). "
                       "VALIDATION §8 '—' 행: OOS 정보 없음(설계 단계 AUC 0.526) → 모델 입력 후보에서 영구 제외")
    else:
        out["reason"] = "조건(학습 ≥2역년·≥40양성, 2017-01-03 재적합부터)을 만족하는 재적합이 없음"
    return out, p


# ------------------------------------------------------------------
# 라이브 모델 · 결정론 검사
# ------------------------------------------------------------------
def fit_live_models(feats: pd.DataFrame, y, refit_date, sha: str, rule: str) -> dict[str, M.LogitModel]:
    """라이브 계수: M1·M2·M3 를 같은 재적합일의 퍼지 학습창으로 적합 (M3 가 model_p2.json, M1·M2 는 카드의 사다리 오늘값용)."""
    out = {}
    for rung in PRIMARY_RUNGS:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m = C.fit_refit(feats, y, refit_date, M.LADDER[rung], rung=rung, spec_sha256=sha, feature_rule=rule)
        for w in _messages(caught):
            warnings.warn(f"live {rung}: {w}")
        if rung == "M3":
            assert m.n_params() == M.PARAM_COUNT, f"라이브 M3 n_params {m.n_params()} ≠ PARAM_COUNT {M.PARAM_COUNT}"
        out[rung] = m
    return out


def determinism_gate(m_new: M.LogitModel, model_path: Path, data_sha: str) -> dict:
    """새 라이브 계수 vs 기존 model_p2.json. 같은 spec·같은 재적합일인데 계수 차 > 1e-9 면 SystemExit(1) (파일을 쓰지 않는다)."""
    m2 = m_new
    # 같은 자료로 두 번 적합 → 비트 동일 확인 (플랫폼 안 결정론)
    info = {"tolerance": COEF_TOL, "model_path": str(model_path)}
    if not model_path.exists():
        info.update({"status": "new", "ok": True, "note": "기존 model_p2.json 없음 → 새로 씀"})
        return info
    try:
        old = M.load_model(model_path)
    except (OSError, ValueError) as e:
        info.update({"status": "unreadable", "ok": True, "note": f"기존 파일 읽기 실패({type(e).__name__}) → 새로 씀"})
        warnings.warn(info["note"])
        return info
    old_data = (old.extra or {}).get("data_sha256")
    info.update({"old_model_id": old.model_id, "old_refit_date": old.refit_date, "old_spec_sha256": old.spec_sha256,
                 "data_changed": (old_data is not None and old_data != data_sha), "old_data_sha256": old_data})
    if old.spec_sha256 != m2.spec_sha256:
        info.update({"status": "spec_changed", "ok": True,
                     "note": f"spec_sha256 변경({old.spec_sha256[:8]} → {m2.spec_sha256[:8]}): 코드가 바뀌었다 → 새 파일을 쓴다. "
                             "장부(VALIDATION §8)에 코드 변경 항목을 기재하라"})
        return info
    if old.refit_date != m2.refit_date:
        info.update({"status": "refit_date_changed", "ok": True,
                     "note": f"재적합일 변경({old.refit_date} → {m2.refit_date}; 홀드아웃 해제 후 연간 재적합 등) → 새 파일을 쓴다"})
        return info
    diffs = {"intercept": abs(old.intercept - m2.intercept), "clim": abs(old.clim - m2.clim)}
    for k in m2.features:
        diffs[k] = abs(float(old.coef.get(k, float("nan"))) - m2.coef[k])
    mx = max(diffs.values())
    info.update({"max_abs_diff": float(mx), "diffs": {k: float(v) for k, v in diffs.items()}})
    if info["data_changed"]:
        # 입력 지문이 바뀌었다(주간 build_cache/일간 update_daily 의 Yahoo 재다운로드: 배당 조정계수 반올림으로 SPY 가 상대 ~1.5e-6,
        # 계수가 ~9e-6 움직인다 — 실측 2026-09-08; 6자리 CSV 반올림만이면 ~2e-8).
        # 같은 코드·다른 자료의 비교는 결정론 검사가 아니므로 실패시키지 않고, 이동 폭을 기록·경고한다(조용히 넘기지 않는다).
        material = bool(math.isfinite(mx) and mx > COEF_DRIFT_MATERIAL)
        info.update({"status": "data_changed", "ok": True, "material_drift": material,
                     "note": (f"입력 데이터 지문 변경(캐시 재구축) → 계수 이동 최대 {mx:.3e} "
                              + (f"> {COEF_DRIFT_MATERIAL:g}: 단순 반올림이 아닌 데이터 수정 의심 — 장부(VALIDATION §8) 검토 대상"
                                 if material else f"≤ {COEF_DRIFT_MATERIAL:g}: 재조정 반올림 수준") + ". 새 파일을 쓴다")})
        warnings.warn(info["note"])
        return info
    info.update({"status": "compared", "ok": bool(math.isfinite(mx) and mx <= COEF_TOL)})
    if not info["ok"]:
        info["note"] = (f"결정론 검사 실패: 계수 차 최대 {mx:.3e} > {COEF_TOL:g} (입력 데이터 지문이 같다 — 코드/플랫폼 비결정성). "
                        "파일을 쓰지 않고 실패한다. 복구: 원인을 장부(VALIDATION §8)에 기재한 뒤 results/model_p2.json 을 지우고(=git 에 남음) 다시 실행")
    else:
        info["note"] = f"기존 계수와 일치 (최대 차 {mx:.2e} ≤ {COEF_TOL:g}, 입력 지문 동일)"
    return info


# ------------------------------------------------------------------
# 홀드아웃 최종 검증 (1회)
# ------------------------------------------------------------------
def _write_unlock(unlock: dict, unlock_path: Path) -> list[Path]:
    """unlock 기록을 --results-dir 위치와 정본(HOLDOUT_UNLOCK_PATH) 양쪽에 쓴다 — 배포 트리에 반드시 흔적을 남긴다."""
    targets = [unlock_path]
    if unlock_path.resolve() != HOLDOUT_UNLOCK_PATH.resolve():
        targets.append(HOLDOUT_UNLOCK_PATH)
    for p in targets:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            json.dump(E._jsonable(unlock), f, ensure_ascii=False, indent=1, allow_nan=False)
            f.write("\n")
        _log(f"[holdout] unlock 파일 기록 → {p}")
    return targets


def holdout_ledger_overlap(ledger_path: Path) -> dict:
    """장부가 이미 홀드아웃 구간을 채점했는지 (라이브 채점과 1회 검증의 중복 방지 — VALIDATION.md §6).
    반환 {n, start, end}; 겹치는 행이 없으면 n=0."""
    out = {"n": 0, "start": None, "end": None}
    try:
        if not Path(ledger_path).exists():
            return out
        df = pd.read_csv(ledger_path, encoding="utf-8")
    except (OSError, ValueError) as e:                             # 장부를 못 읽는 것은 경고로만 — 채점을 막지 않는다
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    if not {"asof", "prob_dd5_20", "y_dd5_20"} <= set(df.columns):
        return out
    asof = pd.to_datetime(df["asof"], errors="coerce")
    hit = (asof >= pd.Timestamp(HOLDOUT_START)) & pd.to_numeric(df["prob_dd5_20"], errors="coerce").notna() \
        & pd.to_numeric(df["y_dd5_20"], errors="coerce").notna()
    if bool(hit.any()):
        out.update({"n": int(hit.sum()), "start": _dstr(asof[hit].min()), "end": _dstr(asof[hit].max())})
    return out


def holdout_final_block(bundle, last_ok: pd.Timestamp, first_refit: str, sha: str, warns: list[str],
                        unlock_path: Path) -> dict:
    """하드컷 없이 2024-09-03~ 를 R_2024·R_2025·R_2026 재적합으로 채점. 반환 {holdout(dict), oos_full, params_full, feats_full, y_full}.

    unlock 기록은 **숫자를 찍기 전에** 쓴다: 뒤에서 무엇이 터져도 1회 보장(§6)이 깨지지 않는다."""
    t0 = time.perf_counter()
    ov = holdout_ledger_overlap(L.LEDGER)                      # 장부가 이미 같은 세션을 채점했으면 '오염되지 않은 1회' 가 아니다
    if ov.get("error"):
        warns.append(f"holdout: 장부({L.LEDGER.name}) 확인 실패 — {ov['error']}")
    elif ov["n"]:
        warns.append(f"holdout 중복 채점: 장부에 홀드아웃 구간 채점 행이 이미 {ov['n']}개 있다({ov['start']}~{ov['end']}) — "
                     "이 1회 검증은 '오염되지 않은 홀드아웃' 이 아니다. VALIDATION.md §8 에 기재하라")
        _log(f"::warning::[holdout] 장부에 이미 채점된 홀드아웃 행 {ov['n']}개 ({ov['start']}~{ov['end']}) — 오염되지 않은 1회가 아니다")
    b_full = cut_bundle(bundle, last_ok)                       # 완성 봉까지만 (홀드아웃은 자르지 않는다)
    inp = prepare_inputs(b_full, last_ok)
    feats, y = inp["feats"], inp["y"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        oos_full, params_full = C.walk_forward(feats, y, ladder=M.LADDER, first_refit=first_refit, end=None, holdout_final=True)
    warns += [f"holdout walk_forward: {m}" for m in _messages(caught)]
    ho = oos_full[oos_full.index >= pd.Timestamp(HOLDOUT_START)].copy()
    span = [(HOLDOUT_START, _dstr(ho.index[-1] + pd.Timedelta(days=1)))]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bs = {r: C.block_scores(ho, span, C.rung_col(r)) for r in PRIMARY_RUNGS}
        ci_clim = C.loss_diff_ci(ho["clim"], ho["p_m3"], ho["y"])
        ci_vix = C.loss_diff_ci(ho["p_vix"], ho["p_m3"], ho["y"])
        ci_m1 = C.loss_diff_ci(ho["p_m1"], ho["p_m3"], ho["y"])
        a, b, yy, _ = C._align(ho["clim"], ho["p_m3"], ho["y"])
        dm_clim = C.dm_test((a - yy) ** 2, (b - yy) ** 2)
        rel = C.reliability_table(ho["p_m3"], ho["y"])
    warns += [f"holdout 채점: {m}" for m in _messages(caught)]
    m3 = _all_row(bs["M3"])
    # BSS(기후학) 의 짝지은 블록 부트스트랩 구간 (ledger 의 라이브 채점과 같은 정의)
    from mrl.ledger import _bss_block_ci
    bss_ci = _bss_block_ci((b - yy) ** 2, (a - yy) ** 2) if len(yy) >= P2["boot_block"] else (float("nan"), float("nan"))
    refits = [p for p in params_full if p["rung"] == "M3" and int(p["refit_date"][:4]) >= pd.Timestamp(HOLDOUT_START).year]
    holdout = {
        "start": _dstr(ho.index[0]), "end": _dstr(ho.index[-1]), "n_sessions": int(len(ho)),
        "n": m3["n"], "n_blocks": m3["n_blocks"], "n_pos": m3["n_pos"], "base": m3["base"], "mean_p": m3["mean_p"],
        "brier": m3["brier"], "brier_clim": m3["brier_clim"], "brier_vix": m3["brier_vix"], "brier_m1": m3["brier_m1"],
        "bss_clim": m3["bss_clim"], "bss_vix": m3["bss_vix"], "bss_vix_bgk": m3["bss_vix_bgk"], "bss_m1": m3["bss_m1"], "auc": m3["auc"],
        "ci": [float(bss_ci[0]), float(bss_ci[1])], "ci_definition": "BSS vs 기후학, 짝지은 순환 블록 부트스트랩(40, 4000, seed 0) 95%",
        "loss_diff_ci": {"clim->M3": ci_clim, "B1->M3": ci_vix, "M1->M3": ci_m1}, "dm_clim_to_m3": dm_clim,
        "by_rung": {r: _all_row(bs[r]) for r in PRIMARY_RUNGS}, "reliability": _records(rel),
        "refits": refits, "rule": "R_2024·R_2025·R_2026 각각 자기 재적합일 전 퍼지 자료로만 학습 (§13 --holdout-final)",
        "ledger_entry": "2b", "runtime_sec": round(time.perf_counter() - t0, 1),
    }
    # ── 공개 전에 잠근다: unlock 기록이 먼저다(아래 _log 가 홀드아웃 숫자를 처음으로 드러낸다).
    #    이 뒤로 어떤 단계가 실패해도(결정론 게이트·to_csv·save_model·Ctrl-C) 재실행은 exit 2 로 막힌다.
    unlock = {"timestamp_utc": _now_utc(), "git_sha": _git_sha(), "spec_sha256": sha, "ledger_entry": "2b",
              "n": holdout["n"], "n_pos": holdout["n_pos"], "n_blocks": holdout["n_blocks"],
              "start": holdout["start"], "end": holdout["end"],
              "bss_clim": holdout["bss_clim"], "bss_vix": holdout["bss_vix"], "bss_m1": holdout["bss_m1"],
              "ci": holdout["ci"], "brier": holdout["brier"], "base": holdout["base"],
              "refits": [p["refit_date"] for p in holdout["refits"]],
              "note": "재실행 거부: 이 파일이 있으면 --holdout-final 은 exit 2. 지우는 행위는 git 에 남고 장부 기재 의무"}
    _write_unlock(unlock, unlock_path)
    _log(f"[holdout] {holdout['start']}~{holdout['end']} · n={holdout['n']} (창 {holdout['n_blocks']}) · 기저율 {holdout['base']:.3f} · "
         f"M3 Brier {holdout['brier']:.4f} · BSS clim {holdout['bss_clim']:+.3f} [{bss_ci[0]:+.3f}, {bss_ci[1]:+.3f}] · "
         f"vs B1 {holdout['bss_vix']:+.3f} · vs M1 {holdout['bss_m1']:+.3f} · {holdout['runtime_sec']}s")
    return {"holdout": holdout, "oos_full": oos_full, "params_full": params_full, "feats_full": feats, "y_full": y,
            "sha": sha, "inp_full": inp, "unlock": unlock}


# ------------------------------------------------------------------
# 요약 · 헤드라인
# ------------------------------------------------------------------
def _flatten_selftest(rc: dict) -> dict:
    g = rc.get("gkov_cc_ratio_250", {})
    pk = rc.get("pk_cc_rv20", {})
    q = rc.get("ohlc_quality_1993_95", {})
    late, early = g.get("since_1996", {}), g.get("early_1993_95", {})
    return {
        "ok": rc.get("ok"), "gk_ov_cc_ratio_1996plus_min": late.get("min"), "gk_ov_cc_ratio_1996plus_max": late.get("max"),
        "gk_ov_cc_ratio_1996plus_mean": late.get("mean"), "gk_ov_cc_ratio_1996plus_last": late.get("last"), "gk_ov_cc_bounds": g.get("bounds"),
        "gk_ov_cc_ratio_1993_95_mean": early.get("mean"), "gk_ov_cc_ratio_1993_95_min": early.get("min"), "gk_ov_cc_ratio_1993_95_max": early.get("max"),
        "pk_cc_rv20_min": pk.get("all", {}).get("min"), "pk_cc_rv20_max": pk.get("all", {}).get("max"), "pk_cc_bounds": pk.get("bounds"),
        "early_open_eq_hl_share": q.get("early_1993_95", {}).get("share_open_eq_high_or_low"),
        "early_median_log_range_pct": q.get("early_1993_95", {}).get("median_log_range_pct"),
        "late_open_eq_hl_share": q.get("since_1996", {}).get("share_open_eq_high_or_low"),
        "late_median_log_range_pct": q.get("since_1996", {}).get("median_log_range_pct"),
        "warnings": rc.get("warnings", []),
    }


def build_headline(oos: pd.DataFrame, sc: dict, acc: dict, first_refit: str) -> dict:
    rt = sc["rungs"]
    lad = sc["ladder"]
    steps = {}
    for step in C.STEP_LABELS:
        row = lad[(lad["step"] == step) & (lad["block"].astype(str) == "all")]
        if len(row):
            r = row.iloc[0]
            steps[step] = {"bss": float(r["bss"]), "mean": float(r["mean"]), "lo": float(r["lo"]), "hi": float(r["hi"]),
                           "dm_t": float(r["dm_t"]), "dm_p": float(r["dm_p"]), "phase_share_pos": float(r["phase_share_pos"]),
                           "n_blocks": int(r["n_blocks"])}
    y = oos["y"].astype(float)
    return {
        "oos_start": _dstr(oos.index[0]), "oos_end": _dstr(oos.index[-1]), "first_refit": first_refit,
        "n_sessions": int(len(oos)), "n_labeled": int(y.notna().sum()), "n_blocks20": int(y.notna().sum()) // 20,
        "base_rate": float(y.dropna().mean()) if y.notna().any() else float("nan"),
        "n_refits": int(oos["refit_year"].nunique()),
        "rungs": {r: {k: float(rt.loc[r, k]) for k in ("brier", "bss_clim", "bss_vix", "bss_vix_bgk", "bss_m1", "auc", "mean_p", "base")}
                  for r in rt.index},
        "blocks24": {r: _block_stats(sc["bs24"][r]) for r in PRIMARY_RUNGS},
        "blocks18": {r: _block_stats(sc["bs18"][r]) for r in PRIMARY_RUNGS},
        "blocks_from1999": ({r: _block_stats(sc["bs99"][r]) for r in PRIMARY_RUNGS} if sc["bs99"] else None),
        "ladder_all": steps,
        "acceptance": {"rule": acc["rule"], "literal_pass": acc["literal"]["M3"]["pass"], "literal_failing_blocks": acc["literal"]["M3"]["failing_blocks"],
                       "amended_pass_M3": acc["amended"]["M3"]["pass"], "amended_tone_model": acc["amended_tone_model"],
                       "deploy_mode": acc["deploy_mode"], "tone_model": acc["tone_model"],
                       "sensitivity_blocks": {k: v for k, v in (acc.get("sensitivity_blocks") or {}).items()
                                              if k not in ("literal", "amended")}},
    }


def _f3(v) -> str:
    return "—" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:+.3f}"


def headline_text(h: dict, sc: dict, acc: dict, det: dict) -> str:
    lines = [f"[calibration] OOS {h['oos_start']} ~ {h['oos_end']} · {h['n_sessions']:,}세션 · 라벨 {h['n_labeled']:,}행 (창 {h['n_blocks20']}) · "
             f"기저율 {h['base_rate'] * 100:.1f}% · 재적합 {h['n_refits']}회"]
    for r, d in h["rungs"].items():
        lines.append(f"  {r}: Brier {d['brier']:.4f} · BSS clim {_f3(d['bss_clim'])} · vs B1 {_f3(d['bss_vix'])} · vs BGK {_f3(d['bss_vix_bgk'])} · "
                     f"vs M1 {_f3(d['bss_m1'])} · AUC {d['auc']:.3f} · 평균 p {d['mean_p'] * 100:.1f}%")
    for r in PRIMARY_RUNGS:
        b = h["blocks24"][r]
        lines.append(f"  {r} 24개월 블록: BSS clim 평균 {_f3(b['mean_bss_clim'])} (>0 {b['n_pos_clim']}/{b['n_blocks_table']}, 실패 {b['failing_clim']}) · "
                     f"vs B1 평균 {_f3(b['mean_bss_vix'])} (≥0 {b['n_nonneg_vix']}/{b['n_blocks_table']}, 실패 {b['failing_vix']}) · "
                     f"vs M1 평균 {_f3(b['mean_bss_m1'])} (>0 {b['n_pos_m1']}/{b['n_blocks_table']})")
    for step in ("M0->M1", "M1->M2", "M2->M3", "M1->M3", "clim->M3", "B1->M3", "BGK->M3"):
        s = h["ladder_all"].get(step)
        if s:
            lines.append(f"  단 {step}: 손실차 {s['mean'] * 1e4:+.1f}×1e-4 [{s['lo'] * 1e4:+.1f}, {s['hi'] * 1e4:+.1f}] · DM t {s['dm_t']:+.2f} (p {s['dm_p']:.3f}) · "
                         f"위상 >0 {s['phase_share_pos'] * 100:.0f}%")
    a = h["acceptance"]
    lines.append(f"  §6 literal(M3): {'통과' if a['literal_pass'] else '실패'} (실패 블록 {a['literal_failing_blocks']}) · "
                 f"amended(#2a, post hoc) M3 {'통과' if a['amended_pass_M3'] else '실패'}, 톤 모델 후보 {a['amended_tone_model']} → "
                 f"적용 규칙 {a['rule']} · deploy {a['deploy_mode']} · tone_model {a['tone_model']}")
    sens = a.get("sensitivity_blocks") or {}
    if sens:
        lines.append(f"  §8.2 블록 민감도({sens.get('blocks')}개월): deploy {sens.get('deploy_mode')} · 톤 모델 {sens.get('tone_model') or '없음'} → "
                     + ("주 표와 일치" if sens.get("agrees") else "주 표와 불일치 — 배치 판정이 블록 정의에 의존한다"))
    lines.append(f"  결정론: {det.get('status')} — {det.get('note')}")
    return "\n".join(lines)


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser(description="Phase 2 보정 모델 p2 — walk-forward 사다리·검정·HAR-RV (실험 #2)")
    ap.add_argument("--end", default=None, help=f"하드컷 끝 세션 (기본: {HOLDOUT_START} 직전 세션). 홀드아웃 이후 값은 거부")
    ap.add_argument("--first-refit", default=P2["first_refit"], choices=[P2["first_refit"], P2["first_refit_sensitivity"]])
    ap.add_argument("--blocks", default="24", choices=["24", "18"], help="§6 판정에 쓰는 주 블록 표 (기본 24개월)")
    ap.add_argument("--har-train-start", default="1993-03-03", choices=["1993-03-03", P2["har_train_start_sensitivity"]])
    ap.add_argument("--holdout-final", action="store_true", help="홀드아웃 최종 검증 1회 (unlock 파일이 있으면 exit 2)")
    ap.add_argument("--acceptance-rule", default="literal", choices=["literal", "amended"],
                    help="§13 #2a 소유자 선택: (a) literal(기본·사전 등록) / (b) amended(post hoc, 장부 기재 필요)")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--docs-dir", default=str(DOCS_DIR))
    ap.add_argument("--no-charts", action="store_true", help="차트 생략 (테스트 속도)")
    args = ap.parse_args(argv)

    t_all = time.perf_counter()
    results_dir, docs_dir, data_dir = Path(args.results_dir), Path(args.docs_dir), Path(args.data_dir)
    model_path, unlock_path = results_dir / MODEL_NAME, results_dir / UNLOCK_NAME
    # 1회 보장은 --results-dir 로 옮길 수 없다: 정본 경로(HOLDOUT_UNLOCK_PATH)도 함께 본다.
    # (자료는 --data-dir 그대로라 결과 경로만 바꿔 재실행하면 같은 홀드아웃을 몇 번이든 볼 수 있었다.)
    unlock_hit = next((p for p in (HOLDOUT_UNLOCK_PATH, unlock_path) if p.exists()), None)
    if args.holdout_final and unlock_hit is not None:
        _log(f"::error::{unlock_hit} 이 이미 존재 — 홀드아웃 최종 검증은 1회뿐이다(§6). 재실행하려면 파일을 손으로 지우고(=git 에 남음) 장부에 기재하라")
        return EXIT_UNLOCK_EXISTS
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    warns: list[str] = []

    # 1) 캐시 · 완성 봉 · 하드컷
    _log(f"[run_calibration] 캐시 로드 {data_dir}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle = load_cache(data_dir)                          # apply_guards 포함(멱등)
    for m in _messages(caught):
        _log(f"[캐시 경고] {m[:300]}")
    cache_warns = list(bundle.meta.get("warnings", []) or [])
    end, pre_last, last_ok = resolve_end(bundle, args.end, args.holdout_final)
    if args.holdout_final and end != pre_last:
        end = pre_last                                          # 사전 등록 OOS 는 홀드아웃 모드에서도 하드컷 그대로
    b_cut = cut_bundle(bundle, end)
    _log(f"[run_calibration] 하드컷 end={_dstr(end)} (홀드아웃 직전 {_dstr(pre_last)} · 마지막 완성 봉 {_dstr(last_ok)}) · "
         f"입력 프레임 close {len(b_cut.close):,}행 · spy_ohlc {len(b_cut.spy_ohlc):,}행 · cboe {len(b_cut.cboe):,}행")
    inp = prepare_inputs(b_cut, end)
    feats, y, targets, spy_close = inp["feats"], inp["y"], inp["targets"], inp["spy_close"]
    warns += [f"features: {w}" for w in inp["feature_warnings"]]
    sha, rule = str(feats.attrs["spec_sha256"]), str(feats.attrs["feature_rule"])
    data_sha = _data_sha256(b_cut.spy_ohlc, feats["vix"])
    _log(f"[run_calibration] 특징 {len(feats):,}행 · spec {sha[:12]} · 라벨 {int(y.notna().sum()):,}행 · 기저율(전 이력) {y.dropna().mean() * 100:.1f}%")

    # 2) 첫 재적합 규칙
    ok, rule_info = C.first_refit_ok(feats.index, y, inp["episodes20"], args.first_refit)
    _log(f"[run_calibration] 첫 재적합 {args.first_refit}: 학습 {rule_info['n_rows']:,}행(≥{rule_info['min_rows']}) · "
         f"완결 ≥20% 에피소드 {rule_info['n_episodes']}개 → {'충족' if ok else '미충족'}")
    if args.first_refit == P2["first_refit"]:
        assert ok, f"첫 재적합 규칙 미충족: {rule_info}"
    elif not ok:
        warns.append(f"첫 재적합 {args.first_refit} 는 사전 등록 규칙 미충족(민감도 실행): {rule_info['n_rows']}행·에피소드 {rule_info['n_episodes']}")
    wf_end = wf_end_of(end)
    primary_blocks = BLOCKS_24 if args.blocks == "24" else BLOCKS_18

    # 3) 사다리·소거·민감도 → 채점
    lad = run_ladder(inp, args.first_refit, wf_end, sha, rule, warns)
    oos = lad["oos"]
    sc = score_ladder(lad, primary_blocks, warns)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        era = C.era_auc(feats, oos, P2["eras"], y=y)
        bs_key, alt_key = ("bs24", "bs18") if args.blocks == "24" else ("bs18", "bs24")
        acc = C.acceptance({r: sc[bs_key][r] for r in PRIMARY_RUNGS}, sc["ladder"], rule=args.acceptance_rule)
        # §8.2 블록 민감도: 같은 구간을 다르게 타일링한 표로 같은 규칙을 재채점한다.
        # 두 표는 [2003-01-01, HOLDOUT_START) 를 정확히 타일링하므로 사다리의 "all" 행(합집합)은 동일 — sc["ladder"] 를 그대로 쓴다.
        acc_alt = C.acceptance({r: sc[alt_key][r] for r in PRIMARY_RUNGS}, sc["ladder"], rule=args.acceptance_rule)
    warns += [f"시대별 AUC/판정: {m}" for m in _messages(caught)]
    acc["sensitivity_blocks"] = {
        "blocks": "18" if alt_key == "bs18" else "24",
        "deploy_mode": acc_alt["deploy_mode"], "tone_model": acc_alt["tone_model"],
        "literal": acc_alt["literal"], "amended": acc_alt["amended"],
        "agrees": bool(acc_alt["deploy_mode"] == acc["deploy_mode"] and acc_alt["tone_model"] == acc["tone_model"]),
        "note": "§8.2 블록 민감도 — 사다리 'all' 행은 두 표가 동일(합집합 같음)",
    }
    if not acc["sensitivity_blocks"]["agrees"]:
        warns.append(f"블록 민감도 불일치(§8.2): 주 {args.blocks}개월 → deploy={acc['deploy_mode']}/톤 모델 {acc['tone_model'] or '없음'} · "
                     f"{acc['sensitivity_blocks']['blocks']}개월 → deploy={acc_alt['deploy_mode']}/톤 모델 {acc_alt['tone_model'] or '없음'} "
                     "— 배치 판정이 블록 정의에 의존한다")
    if args.acceptance_rule != "literal":
        warns.append("acceptance_rule=amended 는 post hoc(#2a) — 장부 기재 없이는 채택 불가")

    # 4) HAR-RV · v0 참조선
    har = har_block(inp, args.first_refit, wf_end, end, args.har_train_start, warns)
    oos["har_fc_20"] = np.exp(har["ln_fc"]).reindex(oos.index).to_numpy()
    rds = C.refit_dates(feats.index, args.first_refit, end=wf_end)
    v0ref, p_v0 = v0_reference_block(results_dir, oos, y, rds, warns)
    oos["p_v0ref"] = (p_v0.to_numpy() if p_v0 is not None else np.nan)
    if v0ref.get("available"):
        _log(f"[v0ref] {v0ref['first_refit']}~ n={v0ref['n']} (창 {v0ref['n_blocks']}) · Brier {v0ref['brier']:.4f} · BSS clim {v0ref['bss_clim']:+.3f} · AUC {v0ref['auc']:.3f} · "
             f"v0ref→M3 손실차 {v0ref['v0ref_to_m3']['mean'] * 1e4:+.1f}×1e-4 [{v0ref['v0ref_to_m3']['lo'] * 1e4:+.1f}, {v0ref['v0ref_to_m3']['hi'] * 1e4:+.1f}]")
    else:
        _log(f"[v0ref] 생략: {v0ref.get('reason')}")

    # 5) 홀드아웃 최종 검증(선택) · 라이브 모델 · 결정론
    holdout = None
    holdout_run = None
    unlock_exists = unlock_path.exists() or HOLDOUT_UNLOCK_PATH.exists()   # 정본 해제도 해제다 (post_unlock_annual 분기)
    if args.holdout_final:
        holdout_run = holdout_final_block(bundle, last_ok, args.first_refit, sha, warns, unlock_path)
        holdout = holdout_run["holdout"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if holdout_run is not None or unlock_exists:
            # 홀드아웃 해제 뒤: 라이브 모델 = 전 자료(완성 봉)로 최근 1월 첫 거래일 재적합 (§8.1)
            if holdout_run is None:
                b_full = cut_bundle(bundle, last_ok)
                inp_full = prepare_inputs(b_full, last_ok)
                warns.append("holdout_unlock.json 존재 → 라이브 모델은 전 자료·최근 1월 재적합(사전 등록 OOS 채점은 하드컷 그대로)")
            else:
                inp_full = holdout_run["inp_full"]
            live_refit = C.refit_dates(inp_full["feats"].index, args.first_refit, end=None)[-1]
            live_args = (inp_full["feats"], inp_full["y"], live_refit)
            live_mode = "post_unlock_annual"
            # 라이브 지문은 라이브 적합에 실제로 들어간 구간만(특징 ≤ 학습 끝, 라벨 ≤ 재적합일) 해싱한다.
            # 하드컷 지문은 사전 등록 OOS 기록용이라 전 자료로 적합한 라이브 계수의 변화를 판별하지 못한다
            # (홀드아웃 이후 한 봉만 고쳐도 계수가 움직이는데 지문은 그대로 → '코드 비결정성' 오진).
            live_data_sha = _data_sha256(inp_full["ohlc"].loc[:live_refit], inp_full["feats"]["vix"].loc[:live_refit])
        else:
            live_args = (feats, y, end)
            live_mode = "pre_holdout_hard_cut"
            live_data_sha = data_sha
        live = fit_live_models(*live_args, sha, rule)
        live2 = fit_live_models(*live_args, sha, rule)      # 같은 입력 2회 적합(모드 무관) — 플랫폼 안 결정론 증거
    warns += [f"라이브 모델: {m}" for m in _messages(caught)]
    m_live = live["M3"]
    twice = max([abs(live2["M3"].intercept - m_live.intercept)] + [abs(live2["M3"].coef[k] - m_live.coef[k]) for k in m_live.features])
    det = determinism_gate(m_live, model_path, live_data_sha)
    det.update({"live_mode": live_mode, "refit_twice_max_abs_diff": float(twice), "data_sha256": live_data_sha,
                "hard_cut_data_sha256": data_sha, "spec_changed": det.get("status") == "spec_changed"})
    if twice > COEF_TOL:
        det.update({"ok": False, "note": f"같은 입력 2회 적합 차 {twice:.3e} > {COEF_TOL:g} — 플랫폼 안 비결정성. " + str(det.get("note") or "")})
    _log(f"[live] {m_live.model_id} · b0 {m_live.intercept:+.4f} · " + " · ".join(f"{k} {v:+.4f}" for k, v in m_live.coef.items())
         + f" · clim {m_live.clim:.4f} · 학습 {m_live.train_start}~{m_live.train_end} n={m_live.n_train:,} 양성 {m_live.n_pos:,} · 2회 적합 차 {twice:.1e}")
    _log(f"[determinism] {det['status']}: {det['note']}")
    if not det["ok"]:
        _log(f"::error::{det['note']}")
        return 1
    # 파라미터 밴드 (마지막 세션 x 에 최근 5회 재적합 + 라이브)
    last5 = [p for p in lad["params"] if p["rung"] == "M3"][-P2["param_band_refits"]:]
    band_models = [M.LogitModel(features=tuple(p["features"]), coef=p["coef"], intercept=p["intercept"], refit_date=p["refit_date"],
                                train_start=p["train_start"], train_end=p["train_end"], n_train=p["n_train"], n_pos=p["n_pos"], clim=p["clim"],
                                feature_rule=rule, spec_sha256=sha, model_id=p["model_id"]) for p in last5] + [m_live]
    x_last = feats.iloc[-1]
    band = M.parameter_band(band_models, x_last)
    p_last = m_live.predict_one(x_last)

    # 6) 자기점검 · 요약
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        selftest = _flatten_selftest(V.ratio_checks(inp["ohlc"]))
    warns += [f"자기점검: {m}" for m in _messages(caught)]
    warns += [f"자기점검: {w}" for w in selftest.get("warnings", [])]
    headline = build_headline(oos, sc, acc, args.first_refit)
    oos_out = oos.reindex(columns=list(OOS_CSV_COLUMNS))
    for c in ("refit_year", "block24", "block18"):
        oos_out[c] = oos_out[c].astype(int)
    csv_path = results_dir / CSV_NAME
    oos_out.to_csv(csv_path, index_label="date", float_format="%.10g", lineterminator="\n", encoding="utf-8")
    oos_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    generated = _now_utc()
    import sklearn
    import scipy
    run = {
        "generated_at_utc": generated, "end": _dstr(end), "wf_end_exclusive": wf_end, "holdout_start": HOLDOUT_START, "hard_cut": True,
        "holdout_final": bool(args.holdout_final), "unlock_exists_before_run": bool(unlock_exists), "first_refit": args.first_refit,
        "first_refit_rule_ok": bool(ok), "primary_blocks": args.blocks, "acceptance_rule": args.acceptance_rule,
        "har_train_start": args.har_train_start, "spec_sha256": sha, "feature_rule": rule, "data_sha256": data_sha,
        "live_data_sha256": live_data_sha,
        "spec_changed": bool(det.get("spec_changed")), "live_mode": live_mode, "git_sha": _git_sha(),
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__,
        "scipy": scipy.__version__, "platform": platform.platform(),
        "cache": {"fetched_at_utc": bundle.meta.get("fetched_at_utc"), "spy_last": bundle.meta.get("spy_last"),
                  "spy_last_bar_complete": bundle.meta.get("spy_last_bar_complete"), "spy_last_used": _dstr(end),
                  "last_completed_session": _dstr(last_ok), "yfinance_version": bundle.meta.get("yfinance_version"), "warnings": cache_warns},
        "n_sessions_features": int(len(feats)), "n_sessions_oos": int(len(oos)), "n_refits": int(oos.attrs.get("n_refits") or 0),
        "purge": P2["purge"], "C": P2["C"], "train_start": P2["train_start"], "boot_block": P2["boot_block"], "n_boot": P2["n_boot"],
        "hac_lag": P2["hac_lag"], "seed": 0, "param_count": M.PARAM_COUNT, "budget": M.BUDGET,
        "oos_csv_sha256": oos_sha, "ladder_runtime_sec": lad["ladder_runtime_sec"], "score_runtime_sec": sc["score_runtime_sec"],
    }
    summary = {
        "schema_version": 1, "disclosure": DISCLOSURE, "run": run, "headline": headline,
        "v0_reference": v0ref,
        "rungs": _records(sc["rungs"].reset_index()),
        "ladder": _records(sc["ladder"]),
        "blocks24": _records(sc["bs24"]["M3"]), "blocks24_by_rung": {r: _records(sc["bs24"][r]) for r in PRIMARY_RUNGS},
        "blocks18": _records(sc["bs18"]["M3"]), "blocks18_by_rung": {r: _records(sc["bs18"][r]) for r in PRIMARY_RUNGS},
        "blocks_from1999": (_records(sc["bs99"]["M3"]) if sc["bs99"] else []),
        "blocks_from1999_by_rung": ({r: _records(sc["bs99"][r]) for r in PRIMARY_RUNGS} if sc["bs99"] else {}),
        "first_refit_sensitivity": ({k: v for k, v in lad["sens99"].items() if k not in ("oos", "params")} | {"params": lad["sens99"]["params"]}
                                    if lad["sens99"] else None),
        "block_sessions": {"24": _records(C.block_sessions(feats.index, BLOCKS_24)), "18": _records(C.block_sessions(feats.index, BLOCKS_18))},
        "reliability": _records(sc["reliability"]), "reliability_m1": _records(sc["reliability_m1"]),
        "murphy": sc["murphy"], "murphy_m1": sc["murphy_m1"],
        # 배포 단(acceptance 가 배치한 단). 리포트·차트가 '어느 단을 그릴지' 추측하지 않도록 명시한다.
        "deployed_rung": (acc.get("tone_model") if acc.get("deploy_mode") == "tones" else "M3"),
        "rung_deployed": bool(acc.get("deploy_mode") == "tones" and acc.get("tone_model")),
        "era_auc": _records(era), "acceptance": acc, "first_refit_rule": rule_info,
        "params_by_refit": lad["params"],
        "live_models": {r: m.to_dict() for r, m in live.items()},
        "live_model": m_live.to_dict(), "param_band": [band[0], band[1]], "param_band_note": f"마지막 세션 {_dstr(end)} 의 x 에 최근 5회 재적합+라이브 적용 (p={p_last:.4f})",
        "ablations": sc["ablations"],
        "ablation_detail": {k: {kk: (_records(vv) if isinstance(vv, pd.DataFrame) else vv) for kk, vv in v.items()}
                            for k, v in sc["ablation_detail"].items()},
        "c_sensitivity": sc["c_sensitivity"], "c_sensitivity_detail": sc["c_sensitivity_detail"],
        "har": {k: (_records(v) if isinstance(v, pd.DataFrame) else v) for k, v in har.items() if k not in ("ln_fc", "y_ln")},
        "selftest": selftest, "determinism": det, "holdout": holdout,
        "warnings": warns, "artifacts": {"oos_csv": str(csv_path), "summary_json": str(results_dir / SUMMARY_NAME),
                                         "model_json": str(model_path), "report_html": str(docs_dir / HTML_NAME)},
        "total_runtime_sec": None,
    }
    if holdout is None and unlock_exists:
        src = next((p for p in (unlock_path, HOLDOUT_UNLOCK_PATH) if p.exists()), None)
        try:
            summary["holdout"] = json.loads(src.read_text(encoding="utf-8"))
            summary["holdout"]["source"] = f"{src} (해제 후 실행 — 재채점하지 않음)"
        except (OSError, ValueError) as e:
            warns.append(f"holdout_unlock.json 읽기 실패({type(e).__name__})")

    # 7) 모델 · unlock 파일 · 리포트
    m_live.extra.update({"data_sha256": live_data_sha, "hard_cut_data_sha256": data_sha,
                         "end": _dstr(end), "live_mode": live_mode, "acceptance_rule": args.acceptance_rule,
                         "git_sha": run["git_sha"], "first_refit": args.first_refit, "determinism_status": det["status"]})
    M.save_model(m_live, model_path, deploy_mode=acc["deploy_mode"], tone_model=acc["tone_model"])
    summary["live_model"] = m_live.to_dict()
    # unlock 기록은 holdout_final_block 이 채점 직후·공개 직전에 이미 썼다 (여기서 다시 쓰지 않는다)

    charts: dict[str, bytes] = {}
    if not args.no_charts:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            charts = RPT.charts_p2(oos, BLOCKS_24, summary, spy_close)
            png = har_chart(har, feats["vix"], args.first_refit)
            if png:
                charts["har"] = png
        for m in _messages(caught):
            warns.append(f"차트: {m}")
    v0_paths = _v0_paths(results_dir)
    v0src = RPT.load_summary_v0(v0_paths[1]) if v0_paths else {}
    summary["total_runtime_sec"] = round(time.perf_counter() - t_all, 1)
    summary_json = E._jsonable(summary)
    with open(results_dir / SUMMARY_NAME, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=1, allow_nan=False)
        f.write("\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        RPT.render_calibration_report({**summary_json, "v0_completed": v0src}, docs_dir / HTML_NAME, charts)
    for m in _messages(caught):
        _log(f"[리포트 경고] {m[:300]}")

    _log(headline_text(headline, sc, acc, det))
    _log(f"[run_calibration] 저장: {csv_path.name}, {SUMMARY_NAME}, {MODEL_NAME}, {HTML_NAME} · 경고 {len(warns)}건 · "
         f"총 {summary['total_runtime_sec']}s")
    for w in warns:
        _log(f"  - {w[:300]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
