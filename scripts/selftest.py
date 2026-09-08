#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""캐시·계약 자기점검 (빈 열 · 유령 행 · 최근성 · 열 계약) + Phase 2 (ARCHITECTURE_PHASE2.md §13) ·
Phase 3 (ARCHITECTURE_PHASE3.md §11) 자기점검.

사용:
    python scripts/selftest.py [--data-dir DIR] [--max-age-days N] [--results-dir DIR] [--ledger CSV] [--no-p2] [--no-p3]

네트워크를 쓰지 않는다. 결과는 [OK]/[WARN]/[FAIL] 줄로 출력하고, FAIL 이 하나라도 있으면 종료 코드 1.
  FAIL = 계약 위반(파일·열·인덱스·유령 행·SPY 오래됨·(GK+OV)/CC 비율 이탈·model_p2.json 의 spec 불일치·파라미터 수 ≠ 4)
         — 하류 모듈이 잘못된 답을 낼 수 있는 상태
  WARN = 주의(지연 티커·fg/eod/cboe 가 SPY 보다 며칠 뒤짐·마지막 세션 특징 결측·이벤트 표 잔여 <60일·장부 P2 열 결측일)
         — 기록하되 진행 가능

Phase 2 절(§13 selftest 추가):
  * `vol.ratio_checks` 경계 — rolling-250 (GK+OV)/CC ∈ [0.8, 1.3] (1996~), Parkinson/CC RV20 ∈ [0.3, 3] (이탈 = FAIL)
  * 특징 NaN 꼬리 — 마지막 세션에 x_vix·x_har·x_ma 결측이면 WARN (daily 는 '확률 계산 불가' 경로로 간다)
  * `model_p2.json.spec_sha256 == features.spec_sha256()` (불일치 = FAIL: 코드가 모델보다 새롭다 → run_calibration.py) · n_params == PARAM_COUNT
  * `events.table_horizon(asof).warn` — FOMC 손표 잔여 60일 미만이면 WARN
  * 장부 P2 열 결측일 — completed 행 가운데 prob_dd5_20 도 p2_input_missing 도 없는 날 (WARN)

Phase 3 절(ARCHITECTURE_PHASE3.md §11 selftest 추가):
  * `smoothed_probabilities` 가 regime.py 밖에서 호출되지 않음(grep; 평활은 미래를 읽는다 — FAIL)
  * `model_p3.json` 의 registry_sha·sizing_sha == 현재 코드 (불일치 = FAIL: daily 가 exit 1 할 상태)
  * D_max 가 사다리 위(≥ 0.21 = 격자 최소 6% × k_slow 3.5) · 비중 규칙 적합 파라미터 0개
  * `summary_p3.json.run.d_max == model_p3.json.sizing.d_max` · 주간 자기검사(PIT·파라미터 회계) 결과
  * 저장된 θ 를 실캐시 관측에 다시 대 본 guard 재검사 · θ 파라미터 수 12
  * `kill_record.json`/`kill_manual.json` 이 있으면 `model_p3.json.deploy_mode == "info_only"`(sticky)
  * `alarms.csv` 가 읽히고 열 계약을 지키는지 · 장부 P3 열 결측일(WARN) · 신선 블록 진행
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrl import data as D                                        # noqa: E402
from mrl.config import (ALARMS_PATH, ALL_TICKERS, DATA_DIR, ENSEMBLE_P3, HMM_P3_PATH,   # noqa: E402
                        KILL_MANUAL_PATH, KILL_RECORD_PATH, MODEL_P2_PATH, MODEL_P3_PATH, P3,
                        RESULTS_DIR, TWENTY_FOUR_SEVEN)

SUMMARY_P2_NAME = "summary_p2.json"


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


# ------------------------------------------------------------------
# Phase 2 자기점검 (§13) — 캐시 자기점검 뒤에 같은 Report 에 이어 쓴다. 네트워크 없음, 재적합 없음.
# ------------------------------------------------------------------
def _fmt3(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(f) else f"{f:.3f}"


def run_p2(r: Report, data_dir: Path, results_dir: Path, ledger_path: Path) -> Report:
    """Phase 2 절: ratio_checks 경계 · 특징 NaN 꼬리 · model_p2.json spec/파라미터 수 · 이벤트 표 잔여 · 장부 P2 열 결측일."""
    from mrl import events as EV
    from mrl import features as F
    from mrl import ledger as L
    from mrl import model as M
    from mrl import vol as V

    results_dir, ledger_path = Path(results_dir), Path(ledger_path)
    try:
        bundle = D.load_cache(data_dir)                      # apply_guards 포함(멱등) — daily.py 와 같은 입력
    except Exception as e:                                   # noqa: BLE001
        r.fail(f"P2: load_cache 실패: {e}")
        return r
    spy_last = bundle.spy_ohlc.index[-1]

    # (a) vol.ratio_checks 경계 — 실캐시 OHLC 전 구간
    try:
        rc = V.ratio_checks(bundle.spy_ohlc)
    except Exception as e:                                   # noqa: BLE001
        r.fail(f"P2: vol.ratio_checks 예외: {e}")
        rc = None
    if rc is not None:
        g, pk = rc["gkov_cc_ratio_250"], rc["pk_cc_rv20"]
        late, early = g.get("since_1996", {}), g.get("early_1993_95", {})
        r.check(bool(g.get("ok")),
                f"P2 (GK+OV)/CC rolling-250 비율 {g.get('check_start')}~ ∈ {g.get('bounds')}: "
                f"min {_fmt3(late.get('min'))} · max {_fmt3(late.get('max'))} · 마지막 {_fmt3(late.get('last'))} "
                f"(1993~95 보고만: 평균 {_fmt3(early.get('mean'))})")
        r.check(bool(pk.get("ok")),
                f"P2 Parkinson/CC RV20 비율 ∈ {pk.get('bounds')}: min {_fmt3(pk.get('all', {}).get('min'))} · "
                f"max {_fmt3(pk.get('all', {}).get('max'))}")
        q = rc.get("ohlc_quality_1993_95", {}).get("early_1993_95", {})
        r.ok(f"P2 1993~95 OHLC 품질(보고만): 시가==고/저가 비중 {_fmt3(q.get('share_open_eq_high_or_low'))} · "
             f"중앙 로그 범위 {_fmt3(q.get('median_log_range_pct'))}%")
        for w in rc.get("warnings", []):
            r.warn(f"P2 ratio_checks: {w}")

    # (b) 특징 NaN 꼬리 — 마지막 세션의 모델 입력 세 개
    try:
        import warnings as _w
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            feats = F.build_features(bundle, asof=spy_last)
        fwarns = [str(x.message) for x in caught if not issubclass(x.category, ResourceWarning)]
    except Exception as e:                                   # noqa: BLE001
        r.fail(f"P2: build_features 예외: {e}")
        feats = None
        fwarns = []
    if feats is not None:
        r.check(feats.index[-1] == spy_last, f"P2 특징 표 마지막 세션 == SPY 마지막 ({spy_last:%Y-%m-%d})")
        last_row = feats.iloc[-1]
        ok, why = F.input_status(last_row)
        vals = " · ".join(f"{k} {_fmt3(last_row.get(k))}" for k in F.MODEL_INPUTS)
        r.check(ok, f"P2 마지막 세션 모델 입력 유한: {vals}" if ok else f"P2 마지막 세션 특징 결측 → daily 는 '확률 계산 불가' 경로: {why}",
                level="warn")
        tail = feats.iloc[-5:][list(F.MODEL_INPUTS)]
        n_tail_nan = int(tail.isna().to_numpy().sum())
        r.check(n_tail_nan == 0, "P2 최근 5세션 모델 입력 결측 없음" if n_tail_nan == 0
                else f"P2 최근 5세션 모델 입력 결측 {n_tail_nan}칸: {tail[tail.isna().any(axis=1)].index.strftime('%Y-%m-%d').tolist()}",
                level="warn")
        for w in fwarns:
            r.warn(f"P2 build_features: {w}")

    # (c) model_p2.json — spec_sha256 == 현재 코드 · n_params == PARAM_COUNT (daily.py 의 exit 1 조건을 미리 잡는다)
    model_path = results_dir / MODEL_P2_PATH.name
    sha_now = F.spec_sha256()
    if not model_path.exists():
        r.warn(f"P2 {model_path.name} 없음 — scripts/run_calibration.py(주간)가 먼저 만들어야 daily 가 P2 확률을 낸다")
        m = None
    else:
        try:
            m = M.load_model(model_path)
        except Exception as e:                               # noqa: BLE001
            r.fail(f"P2 {model_path.name} 읽기 실패: {e}")
            m = None
    if m is not None:
        r.check(m.spec_sha256 == sha_now,
                f"P2 model_p2.json spec_sha256 {m.spec_sha256[:12]} == 현재 코드 {sha_now[:12]}" if m.spec_sha256 == sha_now
                else f"P2 model_p2.json spec_sha256 {m.spec_sha256[:12]} ≠ 현재 코드 {sha_now[:12]} — 코드가 모델보다 새롭다: run_calibration.py 를 다시 실행하라(daily 는 exit 1)")
        r.check(m.n_params() == M.PARAM_COUNT, f"P2 적합 파라미터 수 {m.n_params()} == PARAM_COUNT {M.PARAM_COUNT} "
                f"(계수 {list(m.coef)} + 절편)")
        r.check(tuple(m.features) == tuple(F.MODEL_INPUTS), f"P2 모델 특징 {list(m.features)} == {list(F.MODEL_INPUTS)}")
        r.check(m.deploy_mode in M.DEPLOY_MODES and (m.tone_model is None or m.tone_model in ("M1", "M3")),
                f"P2 model_id {m.model_id} · 재적합 {m.refit_date} · deploy {m.deploy_mode} · tone_model {m.tone_model} · clim {m.clim:.4f}")
        summ_path = results_dir / SUMMARY_P2_NAME
        if summ_path.exists():
            try:
                with open(summ_path, encoding="utf-8") as f:
                    s = json.load(f)
                run = s.get("run") or {}
                det = s.get("determinism") or {}
                r.check(run.get("spec_sha256") == m.spec_sha256, f"P2 summary_p2.json.run.spec_sha256 == model_p2.json ({str(run.get('spec_sha256'))[:12]})")
                r.check(det.get("ok") is True, f"P2 summary_p2.json.determinism: {det.get('status')} — {det.get('note')}", level="warn")
                acc = s.get("acceptance") or {}
                r.check(acc.get("deploy_mode") == m.deploy_mode, f"P2 acceptance.deploy_mode {acc.get('deploy_mode')} == model_p2.json {m.deploy_mode}")
            except (OSError, ValueError) as e:
                r.warn(f"P2 {summ_path.name} 읽기 실패: {e}")
        else:
            r.warn(f"P2 {summ_path.name} 없음 — 파라미터 밴드·신뢰도 구간·HAR 예측을 daily 가 채울 수 없다")

    # (d) 이벤트 표 잔여 (FOMC 손표)
    try:
        h = EV.table_horizon(spy_last)
        r.check(not h.get("warn"), f"P2 이벤트 표 잔여: 마지막 FOMC {h.get('last_fomc')} · {h.get('days_left')}일 남음"
                + (f" (< {h.get('warn_days')}일 — mrl/events.py FOMC_DECISION_DAYS 갱신 필요)" if h.get("warn") else "")
                + (" · 잠정 연도" if h.get("last_year_tentative") else ""), level="warn")
    except Exception as e:                                   # noqa: BLE001
        r.fail(f"P2 events.table_horizon 예외: {e}")

    # (e) 장부 P2 열 결측일 — completed 행 가운데 prob_dd5_20 도 p2_input_missing 도 없는 날
    if not ledger_path.exists():
        r.warn(f"P2 장부 {ledger_path.name} 없음 (아직 daily 기록 없음)")
    else:
        try:
            df = L._read(ledger_path)
        except Exception as e:                               # noqa: BLE001
            r.fail(f"P2 장부 읽기 실패: {e}")
            df = None
        if df is not None:
            missing_cols = [c for c in L.P2_COLUMNS + ["prob_dd5_20"] if c not in df.columns]
            r.check(not missing_cols, "P2 장부 P2 열 모두 존재" if not missing_cols else f"P2 장부 P2 열 누락: {missing_cols}")
            comp = df[df["variant"] == "completed"] if "variant" in df.columns else df
            prob = pd.to_numeric(comp["prob_dd5_20"], errors="coerce")
            miss_reason = comp["p2_input_missing"].map(lambda v: not L._is_missing(v) and str(v).strip() != "")
            gap = comp[prob.isna() & ~miss_reason]
            gap_days = sorted(str(a) for a in gap["asof"].tolist())
            r.check(len(gap_days) == 0,
                    f"P2 장부 completed {len(comp)}행: 확률·사유 모두 없는 날 없음" if not gap_days
                    else f"P2 장부 P2 열 결측일 {len(gap_days)}일 (확률도 사유도 없음 — Phase 1 시절 행이거나 daily 실패): "
                         f"{gap_days[:5]}{' …' if len(gap_days) > 5 else ''}", level="warn")
            n_unavail = int(miss_reason.sum())
            if n_unavail:
                r.warn(f"P2 장부 '확률 계산 불가' 기록 {n_unavail}일: {comp.loc[miss_reason, 'asof'].tolist()[:5]}")
            if len(comp) and m is not None:
                last = comp.sort_values("asof").iloc[-1]
                mid = last.get("p2_model_id")
                same_mid = L._is_missing(mid) or str(mid) == m.model_id
                # 통과·실패에서 문장이 뜻하는 바가 달라야 한다(같지 않은데 '==' 로 찍으면 경고가 거짓말이 된다)
                r.check(same_mid,
                        (f"P2 장부 마지막 행({last.get('asof')}) model_id {mid} == model_p2.json {m.model_id}"
                         if same_mid else
                         f"P2 장부 마지막 행({last.get('asof')}) model_id {mid} ≠ model_p2.json {m.model_id} "
                         f"— 그 행은 이전 모델·코드로 기록됐다(장부는 기록이므로 다시 쓰지 않는다)"),
                        level="warn")
    return r


# ------------------------------------------------------------------
# Phase 3 자기점검 (ARCHITECTURE_PHASE3.md §11) — 네트워크 없음, 재적합 없음
# ------------------------------------------------------------------
def _read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_SMOOTHED_CALL = re.compile(r"smoothed_probabilities\s*\(")


def _grep_smoothed(root: Path) -> list[str]:
    """`smoothed_probabilities` **호출**은 regime.py(정의)와 테스트(누수 카나리) 밖에 있으면 안 된다 (§2 점 원칙).

    호출만 본다(이름 뒤에 여는 괄호) — 이 검사기 자신처럼 문자열·주석으로 이름을 언급하는 줄은 위반이 아니다.
    반환: 위반 `파일:줄` 목록."""
    hits: list[str] = []
    for f in sorted(list((root / "mrl").glob("*.py")) + list((root / "scripts").glob("*.py"))):
        if f.name == "regime.py":
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if _SMOOTHED_CALL.search(line) and not line.lstrip().startswith("#"):
                hits.append(f"{f.relative_to(root).as_posix()}:{i}")
    return hits


def run_p3(r: Report, data_dir: Path, results_dir: Path, ledger_path: Path, root: Path = ROOT) -> Report:
    """Phase 3 절: registry_sha·sizing_sha 일치 · θ guard 재검사 · smoothed 호출 금지(grep) · 장부 P3 열 결측일 ·
    alarms.csv 읽힘 · kill_record 와 deploy_mode 정합 · D_max 가 사다리 위 · summary/model 의 d_max 일치 · 신선 블록."""
    from mrl import ensemble as EN
    from mrl import ledger as L
    from mrl import regime as RG
    from mrl import sizing as SZ
    from mrl import track as TR

    results_dir, ledger_path = Path(results_dir), Path(ledger_path)
    model_path = results_dir / MODEL_P3_PATH.name
    hmm_path = results_dir / HMM_P3_PATH.name

    # (a) smoothed_probabilities 호출 금지 — 필터가 아니라 평활을 쓰면 미래를 읽는다
    hits = _grep_smoothed(root)
    r.check(not hits, "P3 smoothed_probabilities 는 regime.py 밖에서 호출되지 않음(전방 필터만)" if not hits
            else f"P3 smoothed_probabilities 가 regime.py 밖에서 호출됨: {hits} — 평활은 미래를 읽는다(§2·§4.4)")

    # (b) 계약 이름 (동시 확장 중인 모듈)
    missing = [f"ledger.{n}" for n in ("P3_COLUMNS", "OUTCOME_COLUMNS_P3") if not hasattr(L, n)]
    r.check(not missing and int(getattr(L, "SCHEMA_VERSION", 0)) >= 3,
            f"P3 장부 계약: schema_version {getattr(L, 'SCHEMA_VERSION', None)} · P3 열 {len(getattr(L, 'P3_COLUMNS', []))}개"
            if not missing else f"P3 장부 계약 누락: {missing}")

    # (c) model_p3.json · hmm_p3.json
    if not model_path.exists():
        r.warn(f"P3 {model_path.name} 없음 — scripts/run_phase3.py(주간)가 먼저 만들어야 daily 가 P3 를 낸다")
        return r
    try:
        m3 = _read_json(model_path)
    except (OSError, ValueError) as e:
        r.fail(f"P3 {model_path.name} 읽기 실패: {e}")
        return r
    reg_now, siz_now = EN.registry_sha256(), SZ.sizing_sha256()
    r.check(m3.get("registry_sha") == reg_now,
            f"P3 registry_sha {str(m3.get('registry_sha'))[:12]} == 현재 코드 {reg_now[:12]}"
            if m3.get("registry_sha") == reg_now else
            f"P3 registry_sha {str(m3.get('registry_sha'))[:12]} ≠ 현재 코드 {reg_now[:12]} — 코드가 산출물보다 새롭다: "
            "run_phase3.py 를 다시 실행하라 (daily 는 exit 1)")
    r.check(m3.get("sizing_sha") == siz_now,
            f"P3 sizing_sha {str(m3.get('sizing_sha'))[:12]} == 현재 코드 {siz_now[:12]}"
            if m3.get("sizing_sha") == siz_now else
            f"P3 sizing_sha {str(m3.get('sizing_sha'))[:12]} ≠ 현재 코드 {siz_now[:12]} — run_phase3.py 를 다시 실행하라")
    sizing_cfg = m3.get("sizing") if isinstance(m3.get("sizing"), dict) else {}
    d_max = sizing_cfg.get("d_max")
    try:
        d_max_f = float(d_max)
    except (TypeError, ValueError):
        d_max_f = float("nan")
    floor = float(min(P3["vol_grid"])) * float(P3["k_slow"])          # 0.06 × 3.5 = 0.21
    r.check(np.isfinite(d_max_f) and d_max_f >= floor - 1e-12,
            f"P3 D_max {d_max_f:.2f} ≥ 사다리 최소 {floor:.2f} (σ_T {float(sizing_cfg.get('sigma_target', float('nan'))) * 100:.0f}%)"
            if np.isfinite(d_max_f) and d_max_f >= floor - 1e-12 else
            f"P3 D_max {d_max} 가 사다리 아래({floor:.2f} 미만) — §6.1.2 는 그 예산선을 제공하지 않는다")
    r.check(int(sizing_cfg.get("n_params", -1)) == 0,
            f"P3 비중 규칙 적합 파라미터 {sizing_cfg.get('n_params')} == 0 (§6-b)")

    # (d) summary_p3.json 의 run.d_max 와 model_p3.json.sizing.d_max 일치
    sp3_path = results_dir / "summary_p3.json"
    if sp3_path.exists():
        try:
            sp3 = _read_json(sp3_path)
            run3 = sp3.get("run") or {}
            same = (run3.get("d_max") is not None and np.isfinite(_num_or_nan(run3.get("d_max")))
                    and abs(_num_or_nan(run3.get("d_max")) - d_max_f) < 1e-12)
            r.check(bool(same), f"P3 summary_p3.run.d_max {run3.get('d_max')} == model_p3.sizing.d_max {d_max}")
            r.check(run3.get("registry_sha") == m3.get("registry_sha") and run3.get("sizing_sha") == m3.get("sizing_sha"),
                    "P3 summary_p3.run 의 registry_sha·sizing_sha == model_p3.json")
            det = sp3.get("determinism") if isinstance(sp3.get("determinism"), dict) else {}
            r.check(det.get("ok") is not False, f"P3 결정론: {det.get('status')} — {det.get('note')}", level="warn")
            st = sp3.get("selftest") if isinstance(sp3.get("selftest"), dict) else {}
            for k in ("hmm_param_count_ok", "p2_param_count_ok", "sizing_n_params_ok", "production_shadow_only",
                      "pit_bit_identical"):
                if k in st:
                    r.check(st[k] is not False, f"P3 주간 자기검사 {k} = {st[k]}")
            fb = ((sp3.get("reference") or {}).get("fresh_blocks")) or {}
            need = int(fb.get("need") or ENSEMBLE_P3["fresh_blocks_min"])
            r.ok(f"P3 신선 블록 {fb.get('complete', 0)}/{need}" + (f" (라이브 시작 {fb.get('live_start')})"
                                                                  if fb.get("live_start") else " (라이브 시작 전)"))
        except (OSError, ValueError) as e:
            r.warn(f"P3 {sp3_path.name} 읽기 실패: {e}")
    else:
        r.warn(f"P3 {sp3_path.name} 없음 — 시나리오·참조 분포 없이 daily 는 회색 'n/a' 로 간다")

    # (e) θ guard 재검사 — 저장된 θ 를 실캐시 관측에 다시 대 본다
    if not hmm_path.exists():
        r.warn(f"P3 {hmm_path.name} 없음 — daily 는 exit 1 이다")
    else:
        try:
            thetas, live = RG.load_thetas(hmm_path)
        except (OSError, ValueError, KeyError) as e:
            r.fail(f"P3 {hmm_path.name} 읽기 실패: {e}")
            thetas, live = [], None
        if thetas:
            r.check(all(RG.n_params(t) == RG.HMM_PARAM_COUNT for t in thetas),
                    f"P3 HMM θ 파라미터 수 {RG.HMM_PARAM_COUNT}개 × {len(thetas)}회 재적합")
            try:
                bundle = D.load_cache(data_dir)
                close = bundle.spy_ohlc["Close"].astype(float)
                obs = RG.observations(close)
                th = live or thetas[-1]
                occ = RG.occupancy(obs.dropna(), th)
                ok_g, why = RG.guard_check(th, occ, None)
                r.check(bool(ok_g), f"P3 θ guard 재검사({th.refit_date} · {th.theta_id}): 통과 · 점유 "
                        f"{float(occ[1]) * 100:.1f}%" if ok_g else f"P3 θ guard 재검사 실패: {why}")
            except Exception as e:                                # noqa: BLE001 - 조용한 실패 금지
                r.fail(f"P3 θ guard 재검사 예외: {type(e).__name__}: {e}")

    # (f) kill_record.json 과 deploy_mode 정합 (킬은 sticky — 주간 재적합이 풀 수 없다)
    kr, km = results_dir / KILL_RECORD_PATH.name, results_dir / KILL_MANUAL_PATH.name
    killed = kr.exists() or km.exists()
    mode = str(m3.get("deploy_mode") or "")
    r.check((not killed) or mode == "info_only",
            (f"P3 킬 기록 {'있음' if killed else '없음'} · model_p3.deploy_mode = {mode}" if (not killed) or mode == "info_only"
             else f"P3 kill_record.json 이 있는데 model_p3.deploy_mode = {mode} — 킬은 sticky 여야 한다(§8.3)"))
    if killed:
        for p in (kr, km):
            if p.exists():
                try:
                    rec = _read_json(p)
                    r.ok(f"P3 {p.name}: {rec.get('asof')} · {rec.get('ci_label') or rec.get('state')} · "
                         f"장부 {rec.get('ledger_entry')}")
                except (OSError, ValueError) as e:
                    r.fail(f"P3 {p.name} 읽기 실패: {e}")

    # (g) alarms.csv 읽힘
    ap_ = results_dir / ALARMS_PATH.name
    if ap_.exists():
        try:
            al = pd.read_csv(ap_, encoding="utf-8")
            missing_cols = [c for c in TR.ALARM_CSV_COLUMNS if c not in al.columns]
            r.check(not missing_cols, f"P3 {ap_.name} {len(al)}행 · 열 {list(TR.ALARM_CSV_COLUMNS)}"
                    if not missing_cols else f"P3 {ap_.name} 열 누락: {missing_cols}")
            if len(al):
                last = al.sort_values("asof").iloc[-1]
                r.ok(f"P3 마지막 경보 {last.get('asof')} {last.get('code')} (값 {last.get('value')})")
        except (OSError, ValueError) as e:
            r.fail(f"P3 {ap_.name} 읽기 실패: {e}")
    else:
        r.ok(f"P3 {ap_.name} 없음 — 아직 경보가 없다(정상)")

    # (h) 장부 P3 열 결측일 — P2 확률이 있는데 P3 열도 사유도 없는 날
    if not ledger_path.exists():
        r.warn(f"P3 장부 {ledger_path.name} 없음 (아직 daily 기록 없음)")
        return r
    try:
        df = L._read(ledger_path)
    except Exception as e:                                        # noqa: BLE001
        r.fail(f"P3 장부 읽기 실패: {e}")
        return r
    miss_cols = [c for c in L.P3_COLUMNS if c not in df.columns]
    r.check(not miss_cols, "P3 장부 P3 열 모두 존재" if not miss_cols else f"P3 장부 P3 열 누락: {miss_cols}")
    comp = df[df["variant"] == "completed"] if "variant" in df.columns else df
    if len(comp) and not miss_cols:
        has_p2 = pd.to_numeric(comp["prob_dd5_20"], errors="coerce").notna()
        has_p3 = pd.to_numeric(comp["p3_sigma_ewma"], errors="coerce").notna()
        reason = comp["p3_w_reason"].map(lambda v: not L._is_missing(v) and str(v).strip() != "")
        gap = comp[has_p2 & ~has_p3 & ~reason]
        days = sorted(str(a) for a in gap["asof"].tolist())
        r.check(not days, f"P3 장부 completed {len(comp)}행: P3 열 결측일 없음" if not days
                else f"P3 장부 P3 열 결측일 {len(days)}일 (Phase 2 시절 행이거나 daily 실패): "
                     f"{days[:5]}{' …' if len(days) > 5 else ''}", level="warn")
        last = comp.sort_values("asof").iloc[-1]
        for col, want, label in (("p3_registry_sha", reg_now, "registry_sha"), ("p3_sizing_sha", siz_now, "sizing_sha")):
            v = last.get(col)
            r.check(L._is_missing(v) or str(v) == want,
                    f"P3 장부 마지막 행({last.get('asof')}) {label} == 현재 코드", level="warn")
        eff = last.get("p3_effective_mode")
        # 결측(P3 이전에 쓰인 행)은 'nan' 이 아니라 '기록 없음' 으로 읽는다 — 페이지·로그에 불량 토큰을 흘리지 않는다
        r.check(L._is_missing(eff) or str(eff) in L.P3_DEPLOY_MODES,
                f"P3 장부 마지막 행({last.get('asof')}) 유효 모드 = "
                + ("기록 없음 (P3 배선 전 행)" if L._is_missing(eff) else str(eff)))
    return r


def _num_or_nan(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
    ap = argparse.ArgumentParser(description="캐시 자기점검 + Phase 2·Phase 3 자기점검")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--max-age-days", type=int, default=6,
                    help="SPY 마지막 일자가 오늘로부터 이 일수보다 오래되면 FAIL (주말+휴장 고려, 기본 6)")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR), help="model_p2.json · summary_p2.json 위치 (읽기 전용)")
    ap.add_argument("--ledger", default=str(RESULTS_DIR / "track_record.csv"), help="장부 CSV 경로")
    ap.add_argument("--no-p2", action="store_true", help="Phase 2 절 생략 (캐시 자기점검만)")
    ap.add_argument("--no-p3", action="store_true", help="Phase 3 절 생략")
    args = ap.parse_args(argv)
    rep = run(Path(args.data_dir), args.max_age_days)
    if not args.no_p2 and rep.n_fail == 0:
        rep.lines.append("")
        rep.lines.append("--- Phase 2 자기점검 (ARCHITECTURE_PHASE2.md §13) ---")
        run_p2(rep, Path(args.data_dir), Path(args.results_dir), Path(args.ledger))
    elif not args.no_p2:
        rep.warn("캐시 자기점검 FAIL → Phase 2 절 생략")
    if not args.no_p3 and rep.n_fail == 0:
        rep.lines.append("")
        rep.lines.append("--- Phase 3 자기점검 (ARCHITECTURE_PHASE3.md §11) ---")
        run_p3(rep, Path(args.data_dir), Path(args.results_dir), Path(args.ledger))
    elif not args.no_p3:
        rep.warn("앞 절 FAIL → Phase 3 절 생략")
    print("\n".join(rep.lines))
    n_ok = sum(1 for ln in rep.lines if ln.startswith("[OK]"))
    print(f"\n결과: FAIL {rep.n_fail} · WARN {rep.n_warn} · OK {n_ok}")
    return 1 if rep.n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
