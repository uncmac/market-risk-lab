#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""캐시·계약 자기점검 (빈 열 · 유령 행 · 최근성 · 열 계약).

사용:
    python scripts/selftest.py [--data-dir DIR] [--max-age-days N]

네트워크를 쓰지 않는다. 결과는 [OK]/[WARN]/[FAIL] 줄로 출력하고, FAIL 이 하나라도 있으면 종료 코드 1.
  FAIL = 계약 위반(파일·열·인덱스·유령 행·SPY 오래됨) — 하류 모듈이 잘못된 답을 낼 수 있는 상태
  WARN = 주의(지연 티커·fg/eod/cboe 가 SPY 보다 며칠 뒤짐) — 기록하되 진행 가능
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrl import data as D                                        # noqa: E402
from mrl.config import ALL_TICKERS, DATA_DIR, TWENTY_FOUR_SEVEN  # noqa: E402


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.n_fail = 0
        self.n_warn = 0

    def ok(self, msg: str) -> None:
        self.lines.append(f"[OK]   {msg}")

    def warn(self, msg: str) -> None:
        self.n_warn += 1
        self.lines.append(f"[WARN] {msg}")

    def fail(self, msg: str) -> None:
        self.n_fail += 1
        self.lines.append(f"[FAIL] {msg}")

    def check(self, cond: bool, msg: str, level: str = "fail") -> bool:
        if cond:
            self.ok(msg)
        elif level == "warn":
            self.warn(msg)
        else:
            self.fail(msg)
        return bool(cond)


def _is_naive_daily(idx) -> bool:
    return (isinstance(idx, pd.DatetimeIndex) and idx.tz is None
            and idx.is_monotonic_increasing and idx.is_unique
            and bool((idx == idx.normalize()).all()))


def _lag_sessions(spy_idx: pd.DatetimeIndex, last) -> int:
    return int((spy_idx > last).sum())


def run(data_dir: Path, max_age_days: int) -> Report:
    r = Report()
    data_dir = Path(data_dir)

    # 0) 파일 존재 · meta
    for name in D.FILES.values():
        p = data_dir / name
        r.check(p.exists(), f"파일 존재: {name}" + (f" ({p.stat().st_size/1024:,.0f}KB)" if p.exists() else ""))
    if r.n_fail:
        return r
    with open(data_dir / D.FILES["meta"], encoding="utf-8") as f:
        meta = json.load(f)
    r.check(meta.get("schema_version") == D.SCHEMA_VERSION, f"meta.schema_version == {D.SCHEMA_VERSION}")
    r.check(bool(meta.get("fetched_at_utc")), f"meta.fetched_at_utc = {meta.get('fetched_at_utc')}")
    r.check("warnings" in meta and isinstance(meta["warnings"], list), "meta.warnings 목록 존재")
    r.check(bool(meta.get("yfinance_version")), f"meta.yfinance_version = {meta.get('yfinance_version')}")

    # 1) 로드 (가드 없이 → 파일 그대로 검사) 와 가드 적용본
    try:
        raw = D.load_cache(data_dir, guards=False)
    except Exception as e:      # noqa: BLE001
        r.fail(f"load_cache 실패: {e}")
        return r
    close, spy_ohlc, cboe, fg, eod = raw.close, raw.spy_ohlc, raw.cboe, raw.fg, raw.eod

    # 2) close 계약
    r.check(_is_naive_daily(close.index), "close 인덱스: tz-naive · 자정 · 단조증가 · 유일")
    r.check("SPY" in close.columns and close["SPY"].notna().any(), "close 에 SPY 존재")
    if "SPY" not in close.columns:
        return r
    spy = close["SPY"].dropna()
    spy_idx = spy.index
    spy_last = spy_idx[-1]
    r.check(spy_idx[0] <= pd.Timestamp("1993-02-01"), f"SPY 첫 일자 {spy_idx[0]:%Y-%m-%d} (1993-01-29 기대)")
    r.check(len(spy) >= 8000, f"SPY 행 수 {len(spy):,} (≥ 8,000)")
    r.check(spy_last.weekday() < 5, f"SPY 마지막 일자 {spy_last:%Y-%m-%d} 는 평일")
    expected = [t for t in ALL_TICKERS if t not in meta.get("missing_tickers", [])]
    absent = [t for t in expected if t not in close.columns]
    r.check(not absent, f"요청 티커 모두 존재 (누락 {absent})" if absent else f"요청 티커 {len(expected)}개 모두 존재")
    empty_cols = [c for c in close.columns if close[c].isna().all()]
    r.check(not empty_cols, f"빈 열 없음" if not empty_cols else f"빈 열(전부 NaN): {empty_cols}")
    # 선물(=F: 2020-04-20 WTI 음수)·환율(=X)·단기금리(^IRX) 는 0 이하가 실제 값일 수 있어 제외
    exempt = [c for c in close.columns if c.endswith("=F") or c.endswith("=X") or c == "^IRX"]
    neg = [c for c in close.columns if c not in exempt and (close[c].dropna() <= 0).any()]
    r.check(not neg, "0 이하 가격 없음 (선물·환율·^IRX 제외)" if not neg else f"0 이하 가격 존재: {neg}")
    # 유령 행: SPY 마지막 일자 이후에 값이 있는 비-24/7 티커
    after = close.index > spy_last
    ghosts = {c: int((after & close[c].notna().values).sum()) for c in close.columns if c not in TWENTY_FOUR_SEVEN}
    ghosts = {c: n for c, n in ghosts.items() if n}
    r.check(not ghosts, "유령 행 없음 (SPY 마지막 일자 이후 비-24/7 값)" if not ghosts else f"유령 행 존재: {ghosts}")
    # 최근성
    today = dt.date.today()
    age = (today - spy_last.date()).days
    r.check(age <= max_age_days, f"SPY 최근성: 마지막 {spy_last:%Y-%m-%d}, {age}일 전 (허용 {max_age_days}일)")
    # 지연 티커
    stale = {}
    for c in close.columns:
        s = close[c].dropna()
        if s.empty:
            continue
        lag = _lag_sessions(spy_idx, s.index[-1])
        if lag >= D.STALE_SESSIONS:
            stale[c] = f"{s.index[-1]:%Y-%m-%d} (-{lag})"
    r.check(not stale, "지연 티커 없음" if not stale else f"지연 티커(≥{D.STALE_SESSIONS}거래일): {stale}", level="warn")

    # 3) spy_ohlc
    r.check(_is_naive_daily(spy_ohlc.index), "spy_ohlc 인덱스: tz-naive · 자정 · 단조증가 · 유일")
    missing_cols = [c for c in D.SPY_OHLC_COLUMNS if c not in spy_ohlc.columns]
    r.check(not missing_cols, f"spy_ohlc 열 {D.SPY_OHLC_COLUMNS}" + (f" 누락 {missing_cols}" if missing_cols else ""))
    if not missing_cols:
        same_idx = spy_ohlc.index.equals(spy_idx)
        r.check(same_idx, "spy_ohlc 인덱스 == close.SPY 거래일" if same_idx
                else f"spy_ohlc 인덱스 불일치 (ohlc {len(spy_ohlc)} vs close {len(spy_idx)})")
        if same_idx:
            r.check(bool(np.allclose(spy_ohlc["Close"].values, spy.values, rtol=1e-6, atol=1e-6)),
                    "spy_ohlc.Close == close.SPY")
        bad = spy_ohlc[(spy_ohlc["High"] < spy_ohlc["Low"]) | (spy_ohlc["Close"] <= 0)]
        r.check(bad.empty, "spy_ohlc High ≥ Low, Close > 0" if bad.empty else f"spy_ohlc 이상 행 {len(bad)}개")

    # 4) cboe
    r.check(_is_naive_daily(cboe.index), "cboe 인덱스: tz-naive · 자정 · 단조증가 · 유일")
    cb_missing = [c for c in D.CBOE_COLUMNS if c not in cboe.columns]
    r.check(not cb_missing, f"cboe 열 {D.CBOE_COLUMNS}" + (f" 누락 {cb_missing}" if cb_missing else ""))
    cb_empty = [c for c in cboe.columns if cboe[c].isna().all()]
    r.check(not cb_empty, "cboe 빈 열 없음" if not cb_empty else f"cboe 빈 열: {cb_empty}", level="warn")
    if len(cboe):
        lag = _lag_sessions(spy_idx, cboe.index[-1])
        r.check(lag < D.STALE_SESSIONS, f"cboe 최근성: 마지막 {cboe.index[-1]:%Y-%m-%d} (SPY 대비 -{lag}거래일)", level="warn")
        r.check(cboe.index[-1] <= spy_last, "cboe 에 SPY 이후 행 없음")
    else:
        r.warn("cboe 비어 있음")

    # 5) fg
    r.check(_is_naive_daily(fg.index), "fg 인덱스: tz-naive · 자정 · 단조증가 · 유일")
    r.check("score" in fg.columns, "fg 에 score 열 존재")
    fg_missing = [c for c in D.FG_COLUMNS if c not in fg.columns]
    r.check(not fg_missing, "fg 구성요소 열 모두 존재" if not fg_missing else f"fg 열 누락: {fg_missing}", level="warn")
    if len(fg) and "score" in fg.columns:
        sc = fg["score"].dropna()
        r.check(bool(((sc >= 0) & (sc <= 100)).all()), "fg.score ∈ [0, 100]")
        r.check(fg.index[0] <= pd.Timestamp("2020-08-05"), f"fg 첫 일자 {fg.index[0]:%Y-%m-%d} (2020-08-03 기대)")
        lag = _lag_sessions(spy_idx, sc.index[-1])
        r.check(lag < D.STALE_SESSIONS, f"fg 최근성: 마지막 {sc.index[-1]:%Y-%m-%d} (SPY 대비 -{lag}거래일)", level="warn")
        r.check(fg.index[-1] <= spy_last, "fg 에 SPY 이후 행 없음")
        wk = fg.index[fg.index.weekday >= 5]
        r.check(len(wk) == 0, "fg 주말 행 없음" if len(wk) == 0 else f"fg 주말 행 {len(wk)}개: {list(wk[:3])}")
    else:
        r.warn("fg 비어 있음")

    # 6) eod
    r.check(_is_naive_daily(eod.index), "eod 인덱스: tz-naive · 자정 · 단조증가 · 유일")
    eod_missing = [c for c in D.EOD_COLUMNS if c not in eod.columns]
    r.check(not eod_missing, f"eod 열 {D.EOD_COLUMNS}" + (f" 누락 {eod_missing}" if eod_missing else ""))
    if len(eod) and not eod_missing:
        r.check(eod["half_day"].dtype == bool, f"eod.half_day dtype bool (실제 {eod['half_day'].dtype})")
        calc = eod["close"] / eod["open"] - 1                       # CSV 소수 6자리 반올림 → 1e-6 수준 오차 허용
        r.check(bool(np.allclose(calc.values, eod["ret"].values, atol=5e-6)), "eod.ret == close/open - 1 (±5e-6)")
        not_trading = eod.index.difference(spy_idx)
        r.check(len(not_trading) == 0, "eod 일자 ⊆ SPY 거래일" if len(not_trading) == 0
                else f"eod 에 SPY 거래일이 아닌 일자 {len(not_trading)}개: {list(not_trading[:3])}")
        bad_bar = eod[~eod["bar_start"].isin([D.FULL_DAY_LAST_BAR, D.HALF_DAY_LAST_BAR])]
        r.check(bad_bar.empty, "eod.bar_start ∈ {15:30, 11:30}" if bad_bar.empty else f"eod bar_start 이상 {len(bad_bar)}행")
        half_ok = bool((eod["half_day"] == (eod["bar_start"] == D.HALF_DAY_LAST_BAR)).all())
        r.check(half_ok, "eod.half_day ⇔ bar_start == 11:30")
        r.check(len(eod) >= 30, f"eod 행 수 {len(eod)} (P6 에 30세션 필요)", level="warn")
        lag = _lag_sessions(spy_idx, eod.index[-1])
        r.check(lag < D.STALE_SESSIONS, f"eod 최근성: 마지막 {eod.index[-1]:%Y-%m-%d} (SPY 대비 -{lag}거래일)", level="warn")
        r.check(eod.index[-1] <= spy_last, "eod 에 SPY 이후 행 없음")
    else:
        r.warn("eod 비어 있음")

    # 7) 가드 멱등성: 파일 그대로 vs 가드 적용본이 같아야 한다(저장 전에 가드를 적용했으므로)
    try:
        g = D.apply_guards(raw)
        same = (g.close.shape == close.shape and g.fg.shape == fg.shape
                and g.eod.shape == eod.shape and g.cboe.shape == cboe.shape)
        r.check(same, "apply_guards 재적용 시 변화 없음(저장본이 이미 가드 적용됨)")
    except Exception as e:      # noqa: BLE001
        r.fail(f"apply_guards 예외: {e}")

    # 8) meta 경고 노출
    for w in meta.get("warnings", []):
        r.warn(f"meta.warnings: {w}")
    return r


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
    ap = argparse.ArgumentParser(description="캐시 자기점검")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--max-age-days", type=int, default=6,
                    help="SPY 마지막 일자가 오늘로부터 이 일수보다 오래되면 FAIL (주말+휴장 고려, 기본 6)")
    args = ap.parse_args(argv)
    rep = run(Path(args.data_dir), args.max_age_days)
    print("\n".join(rep.lines))
    print(f"\n결과: FAIL {rep.n_fail} · WARN {rep.n_warn} · OK {len(rep.lines) - rep.n_fail - rep.n_warn}")
    return 1 if rep.n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
