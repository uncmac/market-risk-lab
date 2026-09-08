#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""캐시·계약 자기점검 (빈 열 · 유령 행 · 최근성 · 열 계약) + Phase 2 자기점검 (ARCHITECTURE_PHASE2.md §13).

사용:
    python scripts/selftest.py [--data-dir DIR] [--max-age-days N] [--results-dir DIR] [--ledger CSV] [--no-p2]

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
from mrl.config import (ALL_TICKERS, DATA_DIR, MODEL_P2_PATH, RESULTS_DIR,   # noqa: E402
                        TWENTY_FOUR_SEVEN)

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
                r.check(L._is_missing(mid) or str(mid) == m.model_id,
                        f"P2 장부 마지막 행({last.get('asof')}) model_id {mid} == model_p2.json {m.model_id}", level="warn")
    return r


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
    ap = argparse.ArgumentParser(description="캐시 자기점검 + Phase 2 자기점검")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--max-age-days", type=int, default=6,
                    help="SPY 마지막 일자가 오늘로부터 이 일수보다 오래되면 FAIL (주말+휴장 고려, 기본 6)")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR), help="model_p2.json · summary_p2.json 위치 (읽기 전용)")
    ap.add_argument("--ledger", default=str(RESULTS_DIR / "track_record.csv"), help="장부 CSV 경로")
    ap.add_argument("--no-p2", action="store_true", help="Phase 2 절 생략 (캐시 자기점검만)")
    args = ap.parse_args(argv)
    rep = run(Path(args.data_dir), args.max_age_days)
    if not args.no_p2 and rep.n_fail == 0:
        rep.lines.append("")
        rep.lines.append("--- Phase 2 자기점검 (ARCHITECTURE_PHASE2.md §13) ---")
        run_p2(rep, Path(args.data_dir), Path(args.results_dir), Path(args.ledger))
    elif not args.no_p2:
        rep.warn("캐시 자기점검 FAIL → Phase 2 절 생략")
    print("\n".join(rep.lines))
    n_ok = sum(1 for ln in rep.lines if ln.startswith("[OK]"))
    print(f"\n결과: FAIL {rep.n_fail} · WARN {rep.n_warn} · OK {n_ok}")
    return 1 if rep.n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
