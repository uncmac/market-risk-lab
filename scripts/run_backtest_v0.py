#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0 규칙 백테스트: replay → 목표변수·에피소드 → 평가 → results/ + docs/backtest_v0.html

사용:
    python scripts/run_backtest_v0.py                         # --variant both, 2015-01-02 ~ 캐시 마지막 날
    python scripts/run_backtest_v0.py --variant completed --start 2020-01-02 --end 2024-12-31
    python scripts/run_backtest_v0.py --results-dir results/tmp --docs-dir results/tmp/docs   # 실험용 출력 위치

산출물 (변형마다):
    results/backtest_v0_{variant}.csv     일별 재현 결과 (mrl.replay.replay_v0 열 계약)
    results/summary_v0_{variant}.json     evaluate.summarize_v0 dict + 실행 메타·125칸 점유표·≥20% 에피소드
                                          + 리포트 파생 키(directional·episode_summary·switches·data_range·honesty)
    docs/backtest_v0_{variant}.html       변형별 전체 리포트 (report.render_backtest_report ①~⑨)
공통:
    results/summary_v0.json               두 변형의 헤드라인만 모은 작은 요약
    docs/backtest_v0.html                 both 이면 두 변형을 나란히 비교하는 단일 페이지, 하나면 그 변형의 전체 리포트

원칙: 점(point-in-time) 원칙은 replay_v0 가 지킨다. 목표변수·에피소드는 SPY 전체 이력(1993~)으로 만든다.
      완성 봉 원칙: 캐시의 마지막 SPY 봉이 장중 부분 봉이면(meta.spy_last_bar_complete=False, 또는 지금 ET 시각 기준
      미완성) 재현 구간의 끝과 목표변수·에피소드용 종가를 마지막 완성 세션까지로 자른다 — 부분 봉을 종가로 채점하지 않는다.
      캐시 meta.warnings 는 리포트 ⑧ 에 '캐시 경고' 로 싣는다.
      에피소드 표는 기본으로 targets.episodes(split=False) — VALIDATION.md §2 의 감사 실측치(≥5% 36회 · ≥10% 12회 ·
      ≥20% 4회)를 재현하는 방식. --episode-split 로 분할 규칙(계약 기본값, 115·26·5회)을 쓸 수 있다.
      창 규칙 보호: summary JSON 의 run 블록에 재현이 쓴 창 규칙(signals_v0.WINDOW_RULE)과 mrl/signals_v0.py 의 SHA-256 을 기록한다.
      --reuse-replay 는 저장된 JSON 의 두 값이 현재 코드와 같을 때만 CSV 를 재사용하고, 다르거나 기록이 없으면 재현을 다시 돌린다 —
      창 규칙을 고친 코드가 옛 규칙의 벤치마크와 조용히 공존하지 못하게 (VALIDATION.md 실험 #1b).
종료 코드: 0 성공, 1 실패(예외). 검증(--verify N)이 실패하면 예외.
"""
from __future__ import annotations

import argparse
import hashlib
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

from mrl import calendar_us as CAL                                       # noqa: E402
from mrl import evaluate as E                                            # noqa: E402
from mrl import report as RPT                                            # noqa: E402
from mrl import targets as T                                             # noqa: E402
from mrl.config import (BACKTEST_START, DATA_DIR, DOCS_DIR, EPISODE_THRESHOLDS,   # noqa: E402
                        RESULTS_DIR, TONES, TONE_EXPOSURE)
from mrl.data import load_cache                                          # noqa: E402
from mrl import signals_v0 as S                                          # noqa: E402
from mrl.replay import cell_table, replay_v0, verify_replay             # noqa: E402
from mrl.signals_v0 import BASKETS, VARIANTS                             # noqa: E402

CHART_NAMES = ("cumret", "tone_bands", "fwd_box", "switches")
CHART_CAPTIONS = {
    "cumret": "누적 수익(로그): SPY 보유 vs v0 톤 배분 (100/100/100/50/25%, 전환 시 5bp).",
    "tone_bands": "SPY 종가(로그)와 v0 톤 밴드. 세로 점선 = ≥10% 에피소드 고점(빨강)·저점(파랑).",
    "fwd_box": "톤별 다음 20거래일 SPY 수익률 상자그림 (수염 5/95분위).",
    "switches": "월별 톤 전환 횟수.",
}


def _utf8_stdout() -> None:
    """Windows 콘솔(cp1252)에서 한국어 출력이 깨지지 않도록."""
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


def _messages(caught) -> list[str]:
    """catch_warnings(record=True) 로 잡은 경고 → 문구 목록(순서 유지·중복 제거).
    ResourceWarning('unclosed database …', yfinance sqlite 캐시 GC 소음)만 제외하고 나머지는 전부 남긴다."""
    out: list[str] = []
    for w in caught:
        if issubclass(w.category, ResourceWarning):
            continue
        m = str(w.message)
        if m not in out:
            out.append(m)
    return out


# ------------------------------------------------------------------
# 리포트용 dict 변환 (evaluate.summarize_v0 → report.render_backtest_report 가 읽는 키)
# ------------------------------------------------------------------
_SC_RENAME = {"baseline": "base", "hit_ci_lo": "ci_lo", "hit_ci_hi": "ci_hi", "fwd_ret_mean": "fwd_mean",
              "fwd_ret_median": "fwd_median", "fwd_ret_p10": "fwd_p10", "fwd_ret_p90": "fwd_p90",
              "dd5_20_rate": "dd5_rate"}
_ALLOC_STRAT = ("cagr", "max_dd", "max_dd_date", "total_return", "ann_vol", "worst_month", "worst_month_label",
                "n_switches", "switches_per_year", "avg_exposure", "cost_total")
_ALLOC_BH = {"cagr": "bh_cagr", "max_dd": "bh_max_dd", "max_dd_date": "bh_max_dd_date", "total_return": "bh_total_return",
             "ann_vol": "bh_ann_vol", "worst_month": "bh_worst_month", "worst_month_label": "bh_worst_month_label"}


def report_summary(summary: dict, variant: str, cells: pd.DataFrame, run: dict) -> dict:
    """summarize_v0 결과를 report 가 기대하는 키 이름으로 옮긴다 (원본 키는 그대로 두고 덧붙인다)."""
    meta = summary.get("meta", {})
    eps = summary.get("episodes", {})
    sc = []
    for r in summary.get("scorecard", []):
        row = {(_SC_RENAME.get(k, k)): v for k, v in r.items()}
        sc.append(row)
    ep10 = dict(eps.get("10", {}).get("summary", {}))
    ep10["switches_per_year"] = ep10.get("tone_switches_per_year")
    ep10["n_episodes_total_1993"] = ep10.get("n_episodes")
    ep10["n_episodes"] = ep10.get("n_evaluable")          # KPI "탐지 X / Y회" 의 Y = 재현 구간 안의 에피소드 수
    alloc = summary.get("allocation", {})
    alloc_rep = {k: v for k, v in alloc.items()
                 if k in ("start", "end", "n_days", "years", "cost_bps", "excess_cagr", "maxdd_improvement")}
    alloc_rep["allocation"] = {k: alloc.get(k) for k in _ALLOC_STRAT}
    alloc_rep["buy_hold"] = {k: alloc.get(src) for k, src in _ALLOC_BH.items()}
    alloc_rep["exposure_map"] = alloc.get("exposure_map", dict(TONE_EXPOSURE))
    switches = dict(summary.get("switching", {}))
    switches["switches_per_year"] = switches.get("tone_switches_per_year")
    data_range = {
        "variant": variant, "basket": run.get("basket"),
        "start": meta.get("replay_start"), "end": meta.get("replay_end"),
        "n_days": meta.get("n_days"), "years": meta.get("years"),
        "spy_first": meta.get("spy_start"), "spy_last": meta.get("spy_end"), "n_spy_days": meta.get("n_spy_days"),
        "fg_avail_first": meta.get("fg_avail_first"), "fg_avail_share": meta.get("fg_avail_share"),
        "eod_avail_first": meta.get("eod_avail_first"), "eod_avail_share": meta.get("eod_avail_share"),
        "n_watch_avail_min": meta.get("n_watch_avail_min"), "n_watch_avail_max": meta.get("n_watch_avail_max"),
        "episode_split": run.get("episode_split"),
        "replay_runtime_sec": run.get("replay_runtime_sec"), "ms_per_day": run.get("ms_per_day"),
        "cache_fetched_at_utc": run.get("cache", {}).get("fetched_at_utc"),
        "yfinance_version": run.get("cache", {}).get("yfinance_version"),
    }
    warns = list(summary.get("warnings", []))
    for k, v in (run.get("replay_warnings") or {}).items():
        warns.append(f"재현 경고 ×{v['n']} ({v['first']}~{v['last']}): {k}")
    for w in (run.get("cache") or {}).get("warnings") or []:          # 캐시 meta.warnings → ⑧ (부분 봉·유령 행·지연 티커 등)
        warns.append(f"캐시 경고: {w}")
    if (run.get("cache") or {}).get("spy_last_bar_complete") is False:
        warns.append(f"캐시 경고: 캐시의 마지막 SPY 봉 {(run.get('cache') or {}).get('spy_last')} 은 장중 부분 봉 → 재현·채점에서 제외됨")
    honesty = ("에피소드 표는 SPY 종가 ATH 기준 1구간 = 1에피소드(split=False; 1993~ ≥5% 36회 · ≥10% 12회 · ≥20% 4회) 방식. "
               if not run.get("episode_split") else
               "에피소드 표는 회복 전 재하락을 새 에피소드로 세는 분할 규칙(split=True) 방식. ")
    honesty += f"재현 구간 {data_range['start']} ~ {data_range['end']} 안의 에피소드만 채점된다(그 밖은 '—')."
    out = dict(summary)
    out.update({
        "variant": variant, "n_days": meta.get("n_days"), "data_range": data_range,
        "episode_summary": ep10, "directional": sc,
        "episodes5": eps.get("5", {}).get("table", []), "episodes10": eps.get("10", {}).get("table", []),
        "episodes20": eps.get("20", {}).get("table", []),
        "cells": [r for r in cells.to_dict("records") if r.get("n", 0) > 0],
        "switches": switches, "allocation": alloc_rep, "warnings": warns,
        "generated_at": run.get("generated_at_utc"), "honesty": honesty,
    })
    return out


# ------------------------------------------------------------------
# 변형 하나 실행
# ------------------------------------------------------------------
def _signals_hash() -> str:
    """mrl/signals_v0.py 내용의 SHA-256 (줄끝 CRLF→LF 정규화 — 체크아웃 방식과 무관하게 같은 값).
    summary JSON 의 run.signals_v0_sha256 에 기록해, 신호 코드가 바뀐 뒤 옛 재현 CSV 를 재사용하지 못하게 한다."""
    raw = Path(S.__file__).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _provenance_matches(prev: dict, json_name: str, variant: str) -> bool:
    """저장된 summary JSON 의 run 블록이 현재 코드와 같은 창 규칙·같은 signals_v0.py 로 만든 것인지. 아니면 사유를 로그하고 False."""
    rule_prev = prev.get("window_rule")
    if rule_prev != S.WINDOW_RULE:
        _log(f"[{variant}] {json_name} 의 창 규칙 {rule_prev!r} ≠ 현재 코드 {S.WINDOW_RULE!r} → 재현 실행 "
             "(다른 창 규칙으로 만든 CSV 는 재사용하지 않는다)")
        return False
    hash_prev = prev.get("signals_v0_sha256")
    hash_cur = _signals_hash()
    if hash_prev != hash_cur:
        _log(f"[{variant}] {json_name} 의 signals_v0.py 해시 {str(hash_prev)[:12]!r} ≠ 현재 {hash_cur[:12]!r} → 재현 실행 "
             "(신호 코드가 바뀐 뒤의 CSV 는 재사용하지 않는다)")
        return False
    return True


def _load_replay_csv(csv_path: Path, bundle, args, variant: str) -> pd.DataFrame | None:
    """--reuse-replay: 저장된 replay CSV 가 요청 구간과 정확히 맞고, 곁의 summary JSON 이 현재 코드와 같은 창 규칙·같은
    signals_v0.py 로 만든 것임을 증명하면 읽어서 쓴다(평가·리포트만 다시). 아니면 None(재현 실행). JSON 이 없거나 읽을 수
    없으면 출처를 증명할 수 없으므로 재사용하지 않는다."""
    if not csv_path.exists():
        _log(f"[{variant}] 재사용할 {csv_path.name} 없음 → 재현 실행")
        return None
    from mrl.replay import REPLAY_COLUMNS, resolve_range
    jp = csv_path.with_name(f"summary_v0_{variant}.json")
    prev: dict = {}
    if jp.exists():
        try:
            with open(jp, encoding="utf-8") as f:
                prev = json.load(f).get("run", {}) or {}
        except (OSError, ValueError) as e:
            _log(f"[{variant}] {jp.name} 을 읽을 수 없음({type(e).__name__}) → 출처 미상 CSV 는 재사용하지 않음 → 재현 실행")
            return None
    else:
        _log(f"[{variant}] {jp.name} 없음 → {csv_path.name} 의 창 규칙을 증명할 수 없어 재사용하지 않음 → 재현 실행")
        return None
    if not _provenance_matches(prev, jp.name, variant):
        return None
    df = pd.read_csv(csv_path, index_col="date", parse_dates=["date"], encoding="utf-8")
    want = resolve_range(bundle.spy_ohlc.index, args.start, args.end)
    if not df.index.equals(want) or any(c not in df.columns for c in REPLAY_COLUMNS):
        _log(f"[{variant}] {csv_path.name} 의 구간/열이 요청과 다름 → 재현 실행")
        return None
    # 같은 날짜라도 다른 종가(재조정·부분 봉)로 만든 replay 는 재사용하지 않는다 (CSV 6자리 반올림 → 상대 1e-6 허용)
    if "spy_close" in df.columns:
        cur = bundle.spy_ohlc["Close"].astype(float).reindex(df.index).to_numpy()
        old = df["spy_close"].astype(float).to_numpy()
        if not np.allclose(old, cur, rtol=1e-6, atol=0.0, equal_nan=False):
            n_bad = int((~np.isclose(old, cur, rtol=1e-6, atol=0.0)).sum())
            _log(f"[{variant}] {csv_path.name} 의 spy_close 가 현재 캐시 종가와 다름({n_bad}일; 재조정·부분 봉 의심) → 재현 실행")
            return None
    for c in ("fg_avail", "eod_avail"):
        df[c] = df[c].astype(bool)
    df.attrs.update({"variant": variant, "basket": args.basket, "start": want[0].strftime("%Y-%m-%d"),
                     "end": want[-1].strftime("%Y-%m-%d"), "n_days": int(len(df)), "reused": True,
                     "runtime_sec": prev.get("replay_runtime_sec"), "ms_per_day": prev.get("ms_per_day"),
                     "warnings": prev.get("replay_warnings", {}), "window_rule": prev.get("window_rule")})
    return df


def run_variant(bundle, variant: str, args, targets: pd.DataFrame, eps: dict, spy_close: pd.Series,
                results_dir: Path, docs_dir: Path) -> dict:
    _log(f"\n=== 변형 {variant} · 바스켓 {args.basket} · {args.start} ~ {args.end or '캐시 끝'} ===")
    t0 = time.perf_counter()
    csv_path = results_dir / f"backtest_v0_{variant}.csv"
    caught: list = []
    rep = _load_replay_csv(csv_path, bundle, args, variant) if args.reuse_replay else None
    if rep is None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rep = replay_v0(bundle, args.start, args.end, variant=variant, basket=args.basket,
                            progress=not args.no_progress)
        rep.to_csv(csv_path, index_label="date", float_format="%.6f", lineterminator="\n", encoding="utf-8")
    replay_runtime = rep.attrs.get("runtime_sec")
    _log(f"[{variant}] 재현 {len(rep):,}일 · {replay_runtime}s ({rep.attrs.get('ms_per_day')} ms/일)"
         + (" (저장된 CSV 재사용)" if rep.attrs.get("reused") else ""))

    verify = None
    if args.verify > 0:
        # 메모리 replay 는 비트 동일, CSV 재사용본은 소수 6자리 반올림 허용
        verify = verify_replay(bundle, rep, n_samples=args.verify, seed=0,
                               float_tol=(1e-6 if rep.attrs.get("reused") else 0.0))
        _log(f"[{variant}] 검증 {verify['n_checked']}일 전체 번들 재계산과 동일"
             f"{' (±1e-6, CSV 반올림)' if rep.attrs.get('reused') else ' (비트 동일)'} · 전체 번들 {verify['ms_per_day_full']} ms/일")

    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        summary = E.summarize_v0(rep, targets, eps[0.05], eps[0.10], spy_close)
        ep20_tab, ep20_sum = E.episode_eval(rep, eps[0.20], targets)
    summary["episodes"]["20"] = E._jsonable({"threshold": 0.20, "table": ep20_tab, "summary": ep20_sum})
    cells = cell_table(rep)
    extra_warns = sorted(set(_messages(caught2)) - set(summary.get("warnings", [])))
    summary["warnings"] = list(summary.get("warnings", [])) + extra_warns

    run = {
        "generated_at_utc": _now_utc(), "variant": variant, "basket": args.basket,
        "start": rep.attrs.get("start"), "end": rep.attrs.get("end"), "n_days": int(len(rep)),
        "replay_runtime_sec": replay_runtime, "ms_per_day": rep.attrs.get("ms_per_day"),
        "verify": verify, "episode_split": bool(args.episode_split),
        "n_episodes_1993": {str(int(k * 100)): int(len(v)) for k, v in eps.items()},
        "cache": {"fetched_at_utc": bundle.meta.get("fetched_at_utc"), "spy_last": bundle.meta.get("spy_last"),
                  "spy_last_bar_complete": bundle.meta.get("spy_last_bar_complete"),
                  "spy_last_used": spy_close.index[-1].strftime("%Y-%m-%d"),
                  "yfinance_version": bundle.meta.get("yfinance_version"),
                  "warnings": list(bundle.meta.get("warnings", []) or [])},
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "replay_warnings": rep.attrs.get("warnings", {}),
        # 출처 보호: 이 재현이 쓴 창 규칙과 신호 코드 해시. --reuse-replay 는 두 값이 현재 코드와 같을 때만 CSV 를 재사용한다.
        "window_rule": rep.attrs.get("window_rule") or S.WINDOW_RULE,
        "signals_v0_sha256": _signals_hash(),
    }
    if run["window_rule"] != S.WINDOW_RULE:       # 재사용 경로가 막았어야 할 상황 — 조용히 넘기지 않는다
        raise RuntimeError(f"[{variant}] replay 의 창 규칙 {run['window_rule']!r} 이 현재 코드 {S.WINDOW_RULE!r} 과 다름")
    # 리포트용 파생 키(directional·episode_summary·switches·data_range·honesty)도 JSON 에 함께 남긴다 —
    # 하류(Phase 2 비교·외부 검토)가 HTML 없이 같은 숫자를 읽을 수 있게. allocation 은 evaluate 의 평면 dict 그대로.
    rsum = report_summary(summary, variant, cells, run)
    summary_out = {"variant": variant, "basket": args.basket, "run": run, **summary,
                   "directional": E._jsonable(rsum["directional"]),
                   "episode_summary": E._jsonable(rsum["episode_summary"]),
                   "switches": E._jsonable(rsum["switches"]),
                   "data_range": E._jsonable(rsum["data_range"]),
                   "honesty": rsum["honesty"],
                   "cells": E._jsonable(cells)}
    json_path = results_dir / f"summary_v0_{variant}.json"
    with open(json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary_out, f, ensure_ascii=False, indent=1, default=str)
        f.write("\n")

    with warnings.catch_warnings(record=True) as caught3:
        warnings.simplefilter("always")
        charts = RPT.charts_for(rsum, rep, spy_close, targets)
    chart_warns = _messages(caught3)
    if chart_warns:
        rsum["warnings"] = list(rsum["warnings"]) + [f"차트: {m}" for m in chart_warns]
    missing_charts = [c for c in CHART_NAMES if c not in charts]
    if missing_charts:
        rsum["warnings"].append(f"생성되지 않은 차트: {missing_charts}")
    html_path = docs_dir / f"backtest_v0_{variant}.html"
    RPT.render_backtest_report(rsum, html_path, dict(charts))
    summary_out["artifacts"] = {"replay_csv": str(csv_path), "summary_json": str(json_path), "report_html": str(html_path)}
    summary_out["report_warnings"] = list(rsum["warnings"])       # 리포트 ⑧ 에 실린 경고(재현 요약·차트 포함)
    with open(json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary_out, f, ensure_ascii=False, indent=1, default=str)
        f.write("\n")

    elapsed = time.perf_counter() - t0
    _log(f"[{variant}] 저장: {csv_path.name}, {json_path.name}, {html_path.name} · 합계 {elapsed:.1f}s")
    for m in _messages(caught):
        _log(f"[{variant}] 경고: {m[:300]}")
    _log(headline_text(summary_out))
    return {"summary": summary_out, "report_summary": rsum, "charts": charts, "replay": rep, "elapsed_sec": elapsed}


# ------------------------------------------------------------------
# 헤드라인 (콘솔·요약 JSON)
# ------------------------------------------------------------------
def _p(v) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.1f}%"


def _f(v, d=1) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def headline_text(s: dict) -> str:
    h, run, m = s.get("headline", {}), s.get("run", {}), s.get("meta", {})
    tones = " · ".join(f"{t} {_p(h.get('hit_20_by_tone', {}).get(t))}(n={h.get('n_20_by_tone', {}).get(t)})" for t in TONES)
    share = " · ".join(f"{t} {_p((m.get('tone_share') or {}).get(t))}" for t in TONES)
    return "\n".join([
        f"[{s.get('variant')}] 헤드라인 — {run.get('start')} ~ {run.get('end')} ({run.get('n_days'):,}일, "
        f"재현 {run.get('replay_runtime_sec')}s)",
        f"  20일 방향 적중 {_p(h.get('hit_20_all'))} vs 항상-상승 기준선 {_p(h.get('baseline_20'))} · "
        f"60일 {_p(h.get('hit_60_all'))} vs {_p(h.get('baseline_60'))}",
        f"  톤별 20일 적중: {tones}",
        f"  톤 비중: {share}",
        f"  -5%/20일 낙폭 비율 (기저 {_p(h.get('dd5_20_base_rate'))}): "
        + " · ".join(f"{t} {_p(h.get('dd5_20_rate_by_tone', {}).get(t))}" for t in TONES),
        f"  ≥10% 에피소드 탐지 {h.get('n_detected_10')}/{h.get('n_evaluable_10')} ({_p(h.get('detection_rate_10'))}"
        f" · 무작위 이동 기준 {_p(h.get('null_detection_rate_10'))}), 중앙 리드 {_f(h.get('median_lead_10'), 0)}일"
        f" (탐색 창 최대 {h.get('lookback')}일 · 새로 켜진 경고만 {_f(h.get('median_lead_10_fresh'), 0)}일 · 무작위 {_f(h.get('null_median_lead_10'), 0)}일)",
        f"  ≥5% 탐지 {h.get('n_detected_5')}/{h.get('n_evaluable_5')} ({_p(h.get('detection_rate_5'))} · 무작위 {_p(h.get('null_detection_rate_5'))}), "
        f"중앙 리드 {_f(h.get('median_lead_5'), 0)}일 (새로 켜진 것만 {_f(h.get('median_lead_5_fresh'), 0)}일 · 무작위 {_f(h.get('null_median_lead_5'), 0)}일)",
        f"  오경보/년 {_f(h.get('false_alarms_per_year'))} · 오경보율 {_p(h.get('false_alarm_rate'))} vs 임의의 날 {_p(h.get('false_alarm_rate_baseline'))}"
        f" · 진짜 경보 비중 {_p(h.get('true_alarm_share'))} vs 기저율 {_p(h.get('true_alarm_share_baseline'))} · 톤 전환/년 {_f(h.get('tone_switches_per_year'))}",
        f"  배분 CAGR {_p(h.get('cagr_strategy'))} vs 보유 {_p(h.get('cagr_bh'))} · MaxDD {_p(h.get('maxdd_strategy'))} "
        f"vs {_p(h.get('maxdd_bh'))} (개선 {_p(s.get('allocation', {}).get('maxdd_improvement'))}) · 최악 월 {_p(h.get('worst_month_strategy'))} vs {_p(h.get('worst_month_bh'))}",
    ])


def _headline_json(s: dict) -> dict:
    return {"headline": s.get("headline"), "run": s.get("run"), "tone_share": s.get("meta", {}).get("tone_share"),
            "artifacts": s.get("artifacts")}


# ------------------------------------------------------------------
# 두 변형 비교 페이지 (docs/backtest_v0.html, --variant both)
# ------------------------------------------------------------------
def _fmt(v, kind: str) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    if kind == "pct":
        return f"{float(v) * 100:.1f}%"
    if kind == "int":
        return f"{int(v):,}"
    if kind == "num":
        return f"{float(v):.2f}"
    if kind == "days":
        return f"{float(v):.0f}일"
    return RPT._esc(str(v))


def _table(headers: list[str], rows: list[list[str]], first_col_left: bool = True) -> str:
    th = "".join(f'<th class="{"" if (i == 0 and first_col_left) else "num"}">{RPT._esc(h)}</th>' for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f'<td class="{"" if (i == 0 and first_col_left) else "num"}">{c}</td>'
                                    for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="tblwrap"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def _headline_rows(res: dict) -> list[tuple[str, str, str]]:
    """(라벨, 값 faithful, 값 completed) 헤드라인 표 행."""
    vs = list(res)

    def cell(v, path, kind):
        cur = res[v]["summary"]
        for k in path:
            cur = cur.get(k) if isinstance(cur, dict) else None
            if cur is None:
                break
        return _fmt(cur, kind)

    def pair(v, p1, p2, kind, sep=" / "):
        return f"{cell(v, p1, kind)}{sep}{cell(v, p2, kind)}"

    rows = [("거래일 수", *[cell(v, ("run", "n_days"), "int") for v in vs]),
            ("재현 실행 시간(초)", *[cell(v, ("run", "replay_runtime_sec"), "num") for v in vs]),
            ("20일 방향 적중 / 항상-상승 기준선", *[pair(v, ("headline", "hit_20_all"), ("headline", "baseline_20"), "pct") for v in vs]),
            ("60일 방향 적중 / 기준선", *[pair(v, ("headline", "hit_60_all"), ("headline", "baseline_60"), "pct") for v in vs])]
    for t in TONES:
        rows.append((f"20일 적중 · {RPT.TONE_KO.get(t, t)}({t}) [n]",
                     *[f"{cell(v, ('headline', 'hit_20_by_tone', t), 'pct')} [{cell(v, ('headline', 'n_20_by_tone', t), 'int')}]" for v in vs]))
    rows.append(("-5%/20일 낙폭 기저율", *[cell(v, ("headline", "dd5_20_base_rate"), "pct") for v in vs]))
    for t in TONES:
        rows.append((f"-5%/20일 낙폭 비율 · {RPT.TONE_KO.get(t, t)}({t})", *[cell(v, ("headline", "dd5_20_rate_by_tone", t), "pct") for v in vs]))
    rows += [("≥10% 에피소드 탐지 (탐지/평가가능)",
              *[f"{cell(v, ('headline', 'n_detected_10'), 'int')}/{cell(v, ('headline', 'n_evaluable_10'), 'int')} ({cell(v, ('headline', 'detection_rate_10'), 'pct')})" for v in vs]),
             ("≥10% 탐지율 · 무작위 순환 이동 기준선", *[cell(v, ("headline", "null_detection_rate_10"), "pct") for v in vs]),
             ("≥10% 중앙 리드타임 (+=고점 전; 탐색 창 최대 20일)", *[cell(v, ("headline", "median_lead_10"), "days") for v in vs]),
             ("≥10% 중앙 리드타임 · 새로 켜진 경고만 / 무작위 이동", *[pair(v, ("headline", "median_lead_10_fresh"), ("headline", "null_median_lead_10"), "days") for v in vs]),
             ("≥5% 에피소드 탐지 (탐지/평가가능)",
              *[f"{cell(v, ('headline', 'n_detected_5'), 'int')}/{cell(v, ('headline', 'n_evaluable_5'), 'int')} ({cell(v, ('headline', 'detection_rate_5'), 'pct')})" for v in vs]),
             ("≥5% 탐지율 · 무작위 순환 이동 기준선", *[cell(v, ("headline", "null_detection_rate_5"), "pct") for v in vs]),
             ("≥5% 중앙 리드타임", *[cell(v, ("headline", "median_lead_5"), "days") for v in vs]),
             ("≥5% 중앙 리드타임 · 새로 켜진 경고만 / 무작위 이동", *[pair(v, ("headline", "median_lead_5_fresh"), ("headline", "null_median_lead_5"), "days") for v in vs]),
             ("오경보 / 년 (경고 뒤 20일 내 -5% 없음)", *[cell(v, ("headline", "false_alarms_per_year"), "num") for v in vs]),
             ("오경보율 / 임의의 날 기준선", *[pair(v, ("headline", "false_alarm_rate"), ("headline", "false_alarm_rate_baseline"), "pct") for v in vs]),
             ("진짜 경보 비중 / -5%·20일 기저율", *[pair(v, ("headline", "true_alarm_share"), ("headline", "true_alarm_share_baseline"), "pct") for v in vs]),
             ("톤 전환 / 년", *[cell(v, ("headline", "tone_switches_per_year"), "num") for v in vs]),
             ("배분 CAGR / 보유 CAGR", *[pair(v, ("headline", "cagr_strategy"), ("headline", "cagr_bh"), "pct") for v in vs]),
             ("배분 MaxDD / 보유 MaxDD", *[pair(v, ("headline", "maxdd_strategy"), ("headline", "maxdd_bh"), "pct") for v in vs]),
             ("MaxDD 개선 (배분-보유, 양수=개선)", *[cell(v, ("allocation", "maxdd_improvement"), "pct") for v in vs]),
             ("배분 최악 월 / 보유 최악 월", *[pair(v, ("headline", "worst_month_strategy"), ("headline", "worst_month_bh"), "pct") for v in vs])]
    for t in TONES:
        rows.append((f"톤 비중 · {RPT.TONE_KO.get(t, t)}({t})", *[cell(v, ("meta", "tone_share", t), "pct") for v in vs]))
    return rows


def _agreement_section(res: dict) -> str:
    """두 변형의 톤·상태 일치율과 5×5 톤 혼동표 — 부분 봉 문제의 크기."""
    vs = list(res)
    if len(vs) < 2:
        return ""
    a, b = res[vs[0]]["replay"], res[vs[1]]["replay"]
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return '<section class="panel"><h2>② 두 변형의 일치도</h2><div class="note">공통 거래일 없음</div></section>'
    ta, tb = a.loc[common, "tone"].astype(str), b.loc[common, "tone"].astype(str)
    rows = [("톤(tone)", _fmt(float((ta == tb).mean()), "pct"))]
    for col, ko in (("overall_d", "일간 종합"), ("overall_w", "주간 종합"), ("overall_m", "월간 종합")):
        rows.append((ko, _fmt(float((a.loc[common, col] == b.loc[common, col]).mean()), "pct")))
    for k in ("state_fang", "state_macd", "state_vix", "state_btc"):
        rows.append((RPT.SIGNAL_KO.get(k[6:], k) + " (일간)", _fmt(float((a.loc[common, k] == b.loc[common, k]).mean()), "pct")))
    agree = _table(["항목", "일치율"], [[RPT._esc(k), v] for k, v in rows])
    conf = pd.crosstab(ta, tb).reindex(index=list(TONES), columns=list(TONES), fill_value=0)
    conf_rows = [[RPT._tone_pill(t)] + [f"{int(conf.loc[t, u]):,}" for u in TONES] for t in TONES]
    conf_html = _table([f"{vs[0]} ↓ / {vs[1]} →"] + [RPT.TONE_KO.get(u, u) for u in TONES], conf_rows)
    n_diff = int((ta != tb).sum())
    # 다른 날의 예 (최근 10개)
    diff_days = common[(ta != tb).to_numpy()]
    ex = ", ".join(f"{d:%Y-%m-%d} {ta[d]}→{tb[d]}" for d in diff_days[-10:])
    return ('<section class="panel" id="c2"><h2>② 두 변형의 일치도 — 부분 봉 문제의 크기</h2>'
            f'<div class="note">공통 거래일 {len(common):,}일 중 톤이 다른 날 {n_diff:,}일. faithful 은 라이브처럼 부분 주/월을 포함하고 '
            'completed 는 완성 봉만 쓰므로, 일간 상태(일봉은 두 변형 모두 asof 종가를 완성으로 간주)는 같고 주간·월간 종합이 달라진다.</div>'
            f'<div class="two"><div>{agree}</div><div><h3>톤 혼동표 (행 {RPT._esc(vs[0])}, 열 {RPT._esc(vs[1])})</h3>{conf_html}</div></div>'
            + (f'<div class="note">최근 다른 날 예: {RPT._esc(ex)}</div>' if ex else "") + "</section>")


def _scorecard_side_by_side(res: dict, h: int) -> str:
    vs = list(res)
    headers = ["톤"] + [f"{v} 적중" for v in vs] + [f"{v} 기준선" for v in vs] + [f"{v} 95% 구간" for v in vs] + [f"{v} 독립 창" for v in vs]
    rows = []
    for t in list(TONES) + ["all"]:
        cells = {v: next((r for r in res[v]["summary"].get("scorecard", []) if r.get("tone") == t and r.get("h") == h), {}) for v in vs}
        row = [RPT._tone_pill(t) if t in TONES else "전체"]
        row += [_fmt(cells[v].get("hit"), "pct") for v in vs]
        row += [_fmt(cells[v].get("baseline"), "pct") for v in vs]
        row += [f"[{_fmt(cells[v].get('hit_ci_lo'), 'pct')}, {_fmt(cells[v].get('hit_ci_hi'), 'pct')}]" for v in vs]
        row += [_fmt(cells[v].get("n_blocks"), "int") for v in vs]
        rows.append(row)
    return _table(headers, rows)


def _episodes_side_by_side(res: dict, key: str) -> str:
    vs = list(res)
    tabs = {v: res[v]["summary"].get("episodes", {}).get(key, {}).get("table", []) for v in vs}
    base = tabs[vs[0]]
    if not base:
        return '<div class="note">에피소드 없음</div>'
    headers = ["고점일", "저점일", "낙폭", "고점→저점(일)"] + [f"{v} 첫 경고일" for v in vs] + [f"{v} 리드(일)" for v in vs] + [f"{v} 놓침" for v in vs]
    rows = []
    for i, e in enumerate(base):
        row = [RPT._esc(str(e.get("peak_date"))), RPT._esc(str(e.get("trough_date"))), _fmt(e.get("depth"), "pct"), _fmt(e.get("days_to_trough"), "int")]
        others = {v: (tabs[v][i] if i < len(tabs[v]) else {}) for v in vs}
        row += [RPT._esc(str(others[v].get("warn_date") or "—")) if others[v].get("evaluable") else "—" for v in vs]
        row += [_fmt(others[v].get("lead_days"), "days") if others[v].get("evaluable") else "—" for v in vs]
        row += [("예" if others[v].get("missed") else "아니오") if others[v].get("evaluable") else "—" for v in vs]
        rows.append(row)
    return _table(headers, rows)


def render_comparison(res: dict, out_html: Path, start: str, end: str, generated_at: str) -> None:
    """두 변형을 나란히 보여주는 단일 페이지."""
    vs = list(res)
    tags = "".join(f'<span class="tag">변형 <b>{RPT._esc(v)}</b> {RPT._esc(str(res[v]["summary"]["run"].get("n_days")))}일 · '
                   f'{RPT._esc(str(res[v]["summary"]["run"].get("replay_runtime_sec")))}s</span>' for v in vs)
    head = ('<header><div class="eyebrow">market-risk-lab · Phase 1 · 실험 #0 · v0 동결 벤치마크</div>'
            '<h1>v0 규칙 백테스트 — faithful vs completed</h1>'
            '<div class="sub">같은 v0 규칙을 두 가지로 재현한 결과를 나란히 둡니다. faithful = 라이브와 동일(부분 주/월 포함), '
            'completed = 완성 봉만. 두 결과의 차이 자체가 부분 봉 문제의 크기입니다 (VALIDATION.md §4).</div>'
            f'<div class="tags"><span class="tag">구간 <b>{RPT._esc(start)} ~ {RPT._esc(end)}</b></span>{tags}'
            f'<span class="tag">생성 <b>{RPT._esc(generated_at)}</b></span></div></header>')
    nav = ('<nav class="nav"><a href="#c1">① 헤드라인</a><a href="#c2">② 일치도</a><a href="#c3">③ 차트</a>'
           '<a href="#c4">④ 방향 성적표</a><a href="#c5">⑤ 에피소드</a><a href="#c6">⑥ 전체 리포트·규약</a></nav>')
    s1 = ('<section class="panel" id="c1"><h2>① 헤드라인 — 나란히</h2>'
          '<div class="note">방향 적중률은 항상-상승 기준선 옆의 참고값입니다. 주 목표는 다음 한 달의 위험(-5%/20일 낙폭)입니다.</div>'
          + _table(["지표"] + vs, [[RPT._esc(r[0])] + list(r[1:]) for r in _headline_rows(res)])
          + '<div class="quote"><b>' + RPT._esc(RPT.HONESTY_TITLE) + "</b><ul>"
          + "".join(f"<li>{RPT._esc(b)}</li>" for b in RPT.HONESTY_BULLETS) + f"<li><b>{RPT._esc(RPT.HONESTY_BOLD)}</b></li></ul></div></section>")
    s2 = _agreement_section(res)
    charts_html = ""
    for name in CHART_NAMES:
        cols = "".join(f'<div><h3>{RPT._esc(v)}</h3>{RPT._img(res[v]["charts"].get(name), CHART_CAPTIONS[name], f"{name} {v}")}</div>' for v in vs)
        charts_html += f'<div class="two">{cols}</div>'
    s3 = f'<section class="panel" id="c3"><h2>③ 차트 — 왼쪽 {RPT._esc(vs[0])}, 오른쪽 {RPT._esc(vs[-1])}</h2>{charts_html}</section>'
    s4 = ('<section class="panel" id="c4"><h2>④ 방향 성적표 — 항상-상승 기준선 옆에</h2>'
          '<div class="note">buy/hold/neutral = 상승 예측, caution/reduce = 하락 예측. 95% 구간 = 블록 부트스트랩(블록 = 지평). 독립 창 = 겹치지 않는 앞창 수.</div>'
          "<h3>20거래일</h3>" + _scorecard_side_by_side(res, 20) + "<h3>60거래일</h3>" + _scorecard_side_by_side(res, 60) + "</section>")
    s5 = ('<section class="panel" id="c5"><h2>⑤ 에피소드 — 첫 경고일·리드타임·놓침</h2>'
          '<div class="note">리드타임 양수 = 고점 전 경고. 재현 구간 밖 에피소드는 "—". 경고 탐색 창 = 고점 20거래일 전 ~ 저점 — '
          '리드타임은 20일을 넘지 못하며(상한에 걸린 경고는 lead_capped), 탐지율·리드는 ① 의 무작위 이동 기준선과 나란히 읽는다.</div>'
          "<h3>≥10% 낙폭</h3>" + _episodes_side_by_side(res, "10") + "<h3>≥20% 낙폭</h3>" + _episodes_side_by_side(res, "20")
          + "<h3>≥5% 낙폭</h3>" + _episodes_side_by_side(res, "5") + "</section>")
    links = "".join(f'<li><a href="backtest_v0_{RPT._esc(v)}.html">{RPT._esc(VARIANT_LABEL(v))} — 전체 리포트 ①~⑨</a></li>' for v in vs)
    warns = []
    for v in vs:
        warns += [f"[{v}] {w}" for w in res[v]["report_summary"].get("warnings", [])]
    s6 = ('<section class="panel" id="c6"><h2>⑥ 전체 리포트 · 대체 규약 · 경고</h2><ul class="plain">' + links + "</ul>"
          '<h3>대체 규약 (VALIDATION.md §3)</h3><ul class="plain">' + "".join(f"<li>{RPT._esc(t)}</li>" for t in RPT.SUBSTITUTION_RULES) + "</ul>"
          + f'<div class="note">{RPT._esc(res[vs[0]]["report_summary"].get("honesty", ""))}</div>'
          "<h3>경고</h3>" + RPT._warn_list(warns)
          + f'<div class="foot">market-risk-lab · 생성 {RPT._esc(generated_at)} · <a href="index.html">오늘 판정으로</a></div></section>')
    RPT._write_html(out_html, "v0 백테스트 (faithful vs completed) — market-risk-lab", head + nav + s1 + s2 + s3 + s4 + s5 + s6)


def VARIANT_LABEL(v: str) -> str:
    return RPT.VARIANT_KO.get(v, v)


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser(description="v0 규칙 백테스트 (replay → targets → evaluate → report)")
    ap.add_argument("--variant", choices=list(VARIANTS) + ["both"], default="both")
    ap.add_argument("--start", default=BACKTEST_START, help=f"시작일 (기본 {BACKTEST_START})")
    ap.add_argument("--end", default=None, help="종료일 (기본: 캐시의 마지막 SPY 일자)")
    ap.add_argument("--basket", choices=list(BASKETS), default="v0")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--docs-dir", default=str(DOCS_DIR))
    ap.add_argument("--verify", type=int, default=30, help="표본 N일을 전체 번들로 재계산해 동일성 검증 (0 = 생략)")
    ap.add_argument("--episode-split", action="store_true", help="에피소드 분할 규칙(split=True) 사용 (기본 split=False)")
    ap.add_argument("--reuse-replay", action="store_true",
                    help="results/backtest_v0_{variant}.csv 가 요청 구간과 같으면 재현을 건너뛰고 평가·리포트만 다시 만든다")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args(argv)

    t_all = time.perf_counter()
    results_dir, docs_dir, data_dir = Path(args.results_dir), Path(args.docs_dir), Path(args.data_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    variants = list(VARIANTS) if args.variant == "both" else [args.variant]

    _log(f"[run_backtest_v0] 캐시 로드 {data_dir}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle = load_cache(data_dir)
    for m in _messages(caught):
        _log(f"[캐시 경고] {m[:300]}")
    for w in bundle.meta.get("warnings", []) or []:
        _log(f"[캐시 meta.warnings] {str(w)[:300]}")
    # 완성 봉 원칙: 마지막 SPY 봉이 장중 부분 봉이면(캐시 meta 또는 지금 ET 시각 기준) 마지막 완성 세션까지만 쓴다 —
    # 재현 구간(end)·목표변수·에피소드·배분 시뮬 모두 같은 절단선을 쓴다. 부분 봉을 close[t+h] 로 채점하지 않는다.
    spy_idx = bundle.spy_ohlc.index
    if len(spy_idx) < 2:
        raise RuntimeError("SPY 캐시에 세션이 2개 미만 — scripts/build_cache.py 로 재구축하라")
    last_ok = spy_idx[-1]
    if bundle.meta.get("spy_last_bar_complete") is False:            # 캐시가 스스로 '장중 부분 봉' 이라고 기록한 경우
        last_ok = spy_idx[-2]
        _log(f"[run_backtest_v0] 캐시 meta.spy_last_bar_complete=False → {spy_idx[-1]:%Y-%m-%d} 봉 제외")
    completed = CAL.completed_daily(bundle.spy_ohlc)                 # 지금(ET) 기준 16:05 를 지나지 않은 봉 제거
    if len(completed) == 0:
        raise RuntimeError("완성된 SPY 일봉이 없음")
    last_ok = min(last_ok, completed.index[-1])
    if args.end is None or pd.Timestamp(args.end) > last_ok:
        if args.end is not None or last_ok != spy_idx[-1]:
            _log(f"[run_backtest_v0] 미완성 SPY 봉 제외 → end={last_ok:%Y-%m-%d}")
        args.end = last_ok.strftime("%Y-%m-%d")
    spy_close = bundle.spy_ohlc["Close"].astype(float).loc[:last_ok].rename("SPY")   # 완성 종가만 (목표변수·에피소드·배분)
    _log(f"SPY {spy_close.index[0]:%Y-%m-%d} ~ {spy_close.index[-1]:%Y-%m-%d} ({len(spy_close):,}일, 완성 봉 기준) · "
         f"캐시 수집 {bundle.meta.get('fetched_at_utc')}")
    targets = T.make_targets(spy_close)
    eps = {thr: T.episodes(spy_close, thr, split=bool(args.episode_split)) for thr in EPISODE_THRESHOLDS}
    _log("에피소드(1993~, split=%s): " % bool(args.episode_split)
         + " · ".join(f"≥{int(thr * 100)}% {len(df)}회" for thr, df in eps.items()))

    res: dict[str, dict] = {}
    for v in variants:
        res[v] = run_variant(bundle, v, args, targets, eps, spy_close, results_dir, docs_dir)

    generated = _now_utc()
    start = res[variants[0]]["summary"]["run"]["start"]
    end = res[variants[0]]["summary"]["run"]["end"]
    out_html = docs_dir / "backtest_v0.html"
    if len(variants) >= 2:
        render_comparison(res, out_html, start, end, generated)
    else:
        v = variants[0]
        RPT.render_backtest_report(res[v]["report_summary"], out_html, dict(res[v]["charts"]))
    combined = {"generated_at_utc": generated, "start": start, "end": end, "basket": args.basket,
                "episode_split": bool(args.episode_split), "variants": {v: _headline_json(res[v]["summary"]) for v in variants},
                "artifacts": {"comparison_html": str(out_html)},
                "total_runtime_sec": round(time.perf_counter() - t_all, 1)}
    with open(results_dir / "summary_v0.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(combined, f, ensure_ascii=False, indent=1, default=str)
        f.write("\n")
    _log(f"\n[run_backtest_v0] 완료 {time.perf_counter() - t_all:.1f}s → {out_html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
