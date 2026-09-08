# -*- coding: utf-8 -*-
"""mrl.signals_v0 — v0(reference/market_dashboard_v0.py) 신호의 충실한 포팅.

ARCHITECTURE.md 「v0 신호 포팅 — mrl/signals_v0.py」 계약 구현.

원칙
  * reference 의 resample_close / macd_phase / ret / build_metrics / assess / composite / overall /
    combo_advice 를 **그대로**(같은 임계값·같은 연산 순서·같은 float 산술) 옮긴다. 데이터 입력만
    `Bundle + asof` 로 바꾸고, 라이브가 보던 창(window)을 명시적으로 에뮬레이션한다.
  * 점(point-in-time) 원칙: `window()` 는 asof 이후의 행을 절대 포함하지 않는다.
  * 실패는 조용히 넘기지 않는다: 잘못된 입력은 예외, 데이터 한계는 `warnings.warn` 또는
    반환 dict 의 `meta["warnings"]` 에 기록한다.

창(window) 에뮬레이션 — 라이브 `fetch_real` 이 보던 데이터
  * spy / vix / fang : `period="2y"` 는 Yahoo 의 **달력 창** (마지막 봉 - 2y, 마지막 봉] 이다 (시작 경계 배타). 재현은 asof 를
                       앵커로 같은 창을 자른다: (asof - 2y, asof]. 거래일 수는 500~507 로 흔들리며(504 로 고정하면 P1 바스켓의
                       첫날 정규화와 월간 MACD 의 부분 월 봉 때문에 ~0.5% 의 날에 상태가 뒤집힌다), 2026-09-07 실측
                       SPY 2y = 502행 2024-09-05~2026-09-04 (tests/fixtures/fetch_real_*.json). 라이브는 티커 자기 마지막 봉이
                       앵커라 휴장일의 유령 봉(예: ^VIX 2026-09-07) 이 있으면 한두 세션 어긋난다 — 캐시 가드가 유령 행을 지우므로
                       재현할 수 없는 라이브 쪽 결함이다.
  * watch            : `period="1y"` → (asof - 1y, asof] (252행 안팎). 창 안에 봉이 하나도 없는 종목(상장 전)은 라이브의
                       "조회 실패 → 제외" 와 같이 열에서 뺀다 (meta 에 기록).
  * btc              : `period="2y"` 는 BTC 가 매일 거래되므로 [마지막 BTC 봉 - 2y, 마지막 BTC 봉] **포함** 경계 731행 (실측).
                       BTC 지표는 끝 기준(모멘텀·연속 상승일)이라 창 시작은 결과에 영향이 없다.
  * eod              : 라이브는 30m 봉 30일(period="30d")의 세션 마지막 봉. 재현은 `bundle.eod`(1h 봉의 마지막 봉,
                       Open/Close 는 30m 마지막 봉과 비트 동일 — 감사 확인) 에서 반일장을 제외한 asof 이하 마지막
                       30세션. `ret` 는 CSV 의 6자리 반올림 값 대신 close/open-1 을 다시 계산한다(임계 0.001 비교의
                       반올림 오염 방지). asof 세션이 없으면 eod_missing → 가중치 제외.
  * fg               : `bundle.fg` 의 asof 점수. d_daily/d_weekly/d_monthly = SPY 거래일 기준 1/5/21세션 전 점수와의
                       차이(그 날짜에 점수가 없으면 그 이전 마지막 점수 = as-of 조회). 라이브와 같이 `round()` 적용.
                       asof 점수가 없으면 None → fg_missing → 가중치 제외.
  * variant="faithful": asof 일봉 포함, 주/월 부분 기간 그대로 (v0 라이브 재현).
    variant="completed": 일봉은 asof 종가를 완성으로 간주(동일), 주/월은 calendar_us.resample_close(completed_only=True).

주의: 반환 dict 의 계약 키(spy, vix, btc, fang, watch, eod, eod_vals, fg) 외에 asof / variant / meta 를 덧붙인다
(추가 키는 계약을 깨지 않는다; v0_metrics 가 variant 를 읽는다).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from mrl import calendar_us
from mrl.config import V0, V0_DAILY_ONLY, V0_SC, V0_SIGNALS, V0_WEIGHTS

__all__ = [
    "TFS", "VARIANTS", "BASKETS", "WINDOW_SPAN", "EOD_SESSIONS", "BTC_MOM_K", "FG_HIST_N", "MIN_WATCH_BARS",
    "resample_close", "macd_phase", "ret",
    "window", "v0_metrics", "v0_assess", "v0_composite", "v0_overall", "v0_combo", "v0_day",
]

TFS = ("daily", "weekly", "monthly")
VARIANTS = ("faithful", "completed")
BASKETS = ("v0", "equal")
# 라이브 조회 창 — Yahoo 의 period="2y"/"1y" 는 마지막 봉 기준 달력 창 (last - span, last] (시작 경계 배타; 모듈 docstring).
# 재현은 asof 를 앵커로 같은 달력 창을 자른다. BTC 는 자기 마지막 봉 기준 [last - 2y, last] 포함 경계(731행, 실측).
WINDOW_SPAN = {"spy": pd.DateOffset(years=2), "vix": pd.DateOffset(years=2), "fang": pd.DateOffset(years=2),
               "watch": pd.DateOffset(years=1), "btc": pd.DateOffset(years=2)}
# 라이브 30m 봉 period="30d" 는 인트라데이에선 30 세션이다 (2026-09-07 실측: 2026-07-27~09-04 = 30세션)
EOD_SESSIONS = 30
# reference build_metrics 에 인라인된 P9 모멘텀 봉수
BTC_MOM_K = {"daily": 7, "weekly": 4, "monthly": 3}
# reference 의 F&G hist 길이 (스파크라인용; 채점에는 쓰이지 않음)
FG_HIST_N = 180
# reference: `if len(s) < 65: continue` (상장 초기 종목 제외)
MIN_WATCH_BARS = 65
# P3 델타의 거래일 오프셋 (CNN previous_close / previous_1_week / previous_1_month 의 근사)
FG_DELTA_SESSIONS = {"d_daily": 1, "d_weekly": 5, "d_monthly": 21}
_CORE = ("SPY", "^VIX", "BTC-USD")
_RESAMPLE_RULE = {"weekly": "W-FRI", "monthly": "ME"}   # reference resample_close 와 동일


# ============================================================
# reference 의 순수 함수 — 그대로 이식
# ============================================================
def resample_close(s: pd.Series, freq: str, completed_only: bool = False) -> pd.Series:
    """일간 종가 → 주간(금요일 마감)/월간 종가. reference 와 동일한 식('W-FRI'/'ME' last().dropna()).

    completed_only=True 면 calendar_us.resample_close 로 마지막 부분 기간을 제거한다(variant="completed").
    freq 는 daily/weekly/monthly (daily 는 그대로 반환).
    """
    if freq == "daily":
        return s
    if freq not in _RESAMPLE_RULE:
        raise ValueError(f"지원하지 않는 freq: {freq!r} (허용: {TFS})")
    if completed_only:
        return calendar_us.resample_close(s, freq, completed_only=True)
    return s.resample(_RESAMPLE_RULE[freq]).last().dropna()


def macd_phase(close: pd.Series, turn_k: int):
    """MACD(12,26,9) 히스토그램 국면 분류 (P2). reference 와 동일. 반환 (phase, streak, hist)."""
    e12 = close.ewm(span=12, adjust=False).mean()
    e26 = close.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    sig = macd.ewm(span=9, adjust=False).mean()
    hist = (macd - sig).dropna()
    if hist.empty:
        raise ValueError("macd_phase: 입력 시계열이 비어 있음")
    d = hist.diff().dropna()
    up = dn = 0
    for x in d.iloc[::-1]:
        if x > 0 and dn == 0:
            up += 1
        elif x < 0 and up == 0:
            dn += 1
        else:
            break
    h0 = hist.iloc[-1]
    if h0 < 0 and up >= turn_k:
        return "turn", up, hist
    if h0 > 0 and dn >= turn_k:
        return "roll", dn, hist
    if h0 >= 0:
        return "up", up, hist
    return "down", dn, hist


def ret(s: pd.Series, k: int) -> float:
    """마지막 값의 k봉 수익률(%). reference 와 동일 (봉 부족이면 0.0)."""
    if len(s) <= k:
        return 0.0
    return float((s.iloc[-1] / s.iloc[-1 - k] - 1) * 100)


# ============================================================
# 창(window) 에뮬레이션
# ============================================================
def _asof_ts(asof) -> pd.Timestamp:
    """asof 입력(str/date/datetime/Timestamp) → tz-naive 자정 Timestamp."""
    ts = pd.Timestamp(asof)
    if pd.isna(ts):
        raise ValueError(f"asof 를 날짜로 해석할 수 없음: {asof!r}")
    if ts.tzinfo is not None:
        raise ValueError("계약 위반: asof 는 tz-naive 여야 함")
    return ts.normalize()


def _check_frame(df, name: str) -> None:
    if not isinstance(df, (pd.DataFrame, pd.Series)):
        raise TypeError(f"bundle.{name} 은 DataFrame/Series 여야 함: {type(df).__name__}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"bundle.{name} 인덱스는 DatetimeIndex 여야 함")
    if df.index.tz is not None:
        raise ValueError(f"계약 위반: bundle.{name} 인덱스는 tz-naive 여야 함")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"bundle.{name} 인덱스가 오름차순이 아님")


def _span(sub: pd.DataFrame, ticker: str, span: pd.DateOffset, anchor: pd.Timestamp) -> pd.Series:
    """asof 이하로 잘린 close 프레임에서 티커의 달력 창 (anchor - span, anchor] (자기 봉 기준, NaN 제거).
    Yahoo period 창과 같은 규칙 — 시작 경계는 배타(>)."""
    s = sub[ticker].dropna()
    return s[s.index > anchor - span]


def _eod_window(bundle, asof: pd.Timestamp, notes: list) -> tuple[pd.Series, pd.Series, bool]:
    """P6: 반일장을 제외한 asof 이하 마지막 30세션 → (eod_bool, eod_vals[%], eod_avail)."""
    eod = bundle.eod
    if eod is None or len(eod) == 0:
        notes.append("EOD 캐시가 비어 있음 → P6 가중치 제외")
        empty_idx = pd.DatetimeIndex([], name="date")
        return (pd.Series([], index=empty_idx, dtype=bool), pd.Series([], index=empty_idx, dtype=float), False)
    _check_frame(eod, "eod")
    for c in ("open", "close", "half_day"):
        if c not in eod.columns:
            raise ValueError(f"bundle.eod 에 {c} 열이 없음 (열: {list(eod.columns)})")
    sub = eod.loc[:asof]
    sub = sub[~sub["half_day"].astype(bool)]
    avail = asof in sub.index
    sub = sub.iloc[-EOD_SESSIONS:]
    o = sub["open"].astype(float)
    c = sub["close"].astype(float)
    if len(sub) and (not np.isfinite(o.values).all() or (o.values <= 0).any() or not np.isfinite(c.values).all()):
        raise ValueError(f"bundle.eod 에 비정상 open/close 값이 있음 (asof {asof:%Y-%m-%d} 이하 30세션)")
    ret30 = c / o - 1                      # 라이브: last_bar["Close"] / last_bar["Open"] - 1
    idx30 = pd.DatetimeIndex(sub.index)
    eod_bool = pd.Series((ret30 > V0["eod_ret"]).values, index=idx30)
    eod_vals = pd.Series((ret30 * 100).values, index=idx30)   # 마감 30분 수익률(%)
    if not avail:
        notes.append(f"EOD {asof:%Y-%m-%d} 세션 없음(이력 이전·반일장·결손) → P6 가중치 제외")
    return eod_bool, eod_vals, avail


def _fg_window(bundle, asof: pd.Timestamp, spy_idx: pd.DatetimeIndex, notes: list):
    """P3: asof 점수와 1/5/21거래일 전 대비 변화 (라이브 CNN 값 정의의 근사). 없으면 None."""
    fg = bundle.fg
    if fg is None or len(fg) == 0 or "score" not in fg.columns:
        notes.append("F&G 캐시가 비어 있음 → P3 가중치 제외")
        return None
    _check_frame(fg, "fg")
    score_s = fg["score"].astype(float)
    if asof not in score_s.index or not np.isfinite(score_s.at[asof]):
        notes.append(f"F&G {asof:%Y-%m-%d} 점수 없음 → P3 가중치 제외")
        return None
    score = float(score_s.at[asof])
    past = score_s.loc[:asof].dropna()
    pos = spy_idx.get_loc(asof)
    out = {"score": round(score)}          # 라이브: round(j["score"]) → int (파이썬 반올림)
    for name, k in FG_DELTA_SESSIONS.items():
        prev = None
        if pos - k >= 0:
            d = spy_idx[pos - k]
            hit = past.loc[:d]
            if len(hit):
                prev = float(hit.iloc[-1])
        if prev is None:
            notes.append(f"F&G {name}: {asof:%Y-%m-%d} 기준 {k}거래일 전 점수 없음 → 변화 0 으로 처리")
            out[name] = 0
        else:
            out[name] = round(score - prev)  # 라이브: round(score - previous_*)
    out["hist"] = [float(x) for x in past.iloc[-FG_HIST_N:]]
    return out


def window(bundle, asof, variant: str = "faithful") -> dict:
    """v0 라이브가 asof 시점에 보던 데이터 dict 를 재현한다 (모듈 docstring 참고).

    반환 키: spy(DataFrame OHLC), vix(Series), btc(Series), fang(DataFrame), watch(DataFrame),
            eod(Series[bool]), eod_vals(Series[%]), fg(dict|None)
            + asof(Timestamp), variant(str), meta(dict: warnings, fg_avail, eod_avail, n_watch_avail,
              watch_avail, excluded, window_rows)
    asof 는 SPY 거래일이어야 한다(아니면 ValueError — 조용히 전일로 옮기지 않는다).
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant 는 {VARIANTS} 중 하나여야 함: {variant!r}")
    ts = _asof_ts(asof)
    close = bundle.close
    _check_frame(close, "close")
    spy_ohlc = bundle.spy_ohlc
    _check_frame(spy_ohlc, "spy_ohlc")
    if "Close" not in spy_ohlc.columns:
        raise ValueError("bundle.spy_ohlc 에 Close 열이 없음")
    missing_core = [t for t in _CORE if t not in close.columns]
    if missing_core:
        raise ValueError(f"bundle.close 에 핵심 티커가 없음: {missing_core}")
    spy_idx = spy_ohlc.index
    if ts not in spy_idx:
        raise ValueError(f"asof {ts:%Y-%m-%d} 는 SPY 거래일이 아님 (캐시 범위 {spy_idx[0]:%Y-%m-%d}~{spy_idx[-1]:%Y-%m-%d})")

    notes: list[str] = []
    excluded: dict[str, list[str]] = {"fang": [], "watch": []}
    sub = close.loc[:ts]                         # 점 원칙: 이 이후로는 asof 이하만 본다

    spy = spy_ohlc.loc[:ts]
    spy = spy[spy.index > ts - WINDOW_SPAN["spy"]]          # (asof - 2y, asof] — Yahoo period="2y" 달력 창
    vix = _span(sub, "^VIX", WINDOW_SPAN["vix"], ts)
    if vix.empty or vix.index[-1] != ts:
        raise ValueError(f"^VIX 에 {ts:%Y-%m-%d} 종가가 없음")
    btc = sub["BTC-USD"].dropna()
    if btc.empty:
        raise ValueError(f"BTC-USD 에 {ts:%Y-%m-%d} 이하 데이터가 없음")
    btc = btc[btc.index >= btc.index[-1] - WINDOW_SPAN["btc"]]   # [마지막 BTC 봉 - 2y, 마지막 BTC 봉] 포함 경계 (실측 731행)
    if btc.index[-1] != ts:
        msg = f"BTC-USD 마지막 일자 {btc.index[-1]:%Y-%m-%d} < asof {ts:%Y-%m-%d} (데이터 결손)"
        notes.append(msg)
        warnings.warn(msg, stacklevel=2)

    fang = {}
    for t in V0["fang"]:
        s = _span(sub, t, WINDOW_SPAN["fang"], ts) if t in sub.columns else pd.Series(dtype=float)
        if s.empty:
            excluded["fang"].append(t)
            continue
        fang[t] = s
    if not fang:
        raise RuntimeError(f"FANG 전 종목에 {ts:%Y-%m-%d} 이하 데이터가 없음")
    if excluded["fang"]:
        msg = f"FANG {excluded['fang']} 은 {ts:%Y-%m-%d} 창에 데이터 없음 → 바스켓에서 제외 (라이브 동일)"
        notes.append(msg)
        warnings.warn(msg, stacklevel=2)
    fang = pd.DataFrame(fang).dropna()          # 라이브: pd.DataFrame(fang).dropna()
    if fang.empty:
        raise RuntimeError(f"FANG 바스켓이 비어 있음 (asof {ts:%Y-%m-%d})")

    watch = {}
    for t in V0["watchlist"]:
        s = _span(sub, t, WINDOW_SPAN["watch"], ts) if t in sub.columns else pd.Series(dtype=float)
        if s.empty:
            excluded["watch"].append(t)           # 상장 전 → 라이브의 "조회 실패 → 제외" 와 동일
            continue
        watch[t] = s
    if excluded["watch"]:
        notes.append(f"워치리스트 {excluded['watch']} 은 {ts:%Y-%m-%d} 창에 데이터 없음 → 제외")
    watch = pd.DataFrame(watch)                  # 라이브: pd.DataFrame(watch) (dropna 없음)
    watch_avail = [t for t in watch.columns if len(watch[t].dropna()) >= MIN_WATCH_BARS]
    if not watch_avail:
        notes.append(f"워치리스트 중 {MIN_WATCH_BARS}봉 이상 종목이 없음 → P7 가중치 제외")

    eod_bool, eod_vals, eod_avail = _eod_window(bundle, ts, notes)
    fg = _fg_window(bundle, ts, spy_idx, notes)

    win = {
        "spy": spy, "vix": vix, "btc": btc, "fang": fang, "watch": watch,
        "eod": eod_bool, "eod_vals": eod_vals, "fg": fg,
        "asof": ts, "variant": variant,
        "meta": {
            "warnings": notes,
            "fg_avail": fg is not None,
            "eod_avail": bool(eod_avail),
            "n_watch_avail": len(watch_avail),
            "watch_avail": watch_avail,
            "excluded": excluded,
            "window_rows": {"spy": len(spy), "vix": len(vix), "btc": len(btc), "fang": len(fang),
                            "watch": len(watch), "eod": len(eod_bool)},
        },
    }
    # 점 원칙 자기점검 — 어떤 프레임도 asof 이후 행을 가지면 안 된다
    for k in ("spy", "vix", "btc", "fang", "watch", "eod", "eod_vals"):
        obj = win[k]
        if len(obj) and obj.index.max() > ts:
            raise AssertionError(f"점 원칙 위반: window[{k!r}] 에 asof {ts:%Y-%m-%d} 이후 행이 있음")
    return win


# ============================================================
# 메트릭 — reference build_metrics 이식 (data → win, CONFIG → V0)
# ============================================================
def _equal_weight_basket(fang_d: pd.DataFrame) -> pd.Series:
    """basket="equal": 일별 동일가중 리밸런스 수익률 누적 (첫날 = 1.0). Phase 2 실험용."""
    r = fang_d.pct_change(fill_method=None)
    r.iloc[0] = 0.0
    return (1.0 + r.mean(axis=1)).cumprod()


def v0_metrics(win: dict, tf: str, cut: int = 0, basket: str = "v0") -> dict:
    """reference build_metrics 와 동일 키의 메트릭 dict.

    cut > 0 이면 SPY 마지막 cut 일을 제외한 시점(asof') 으로 되감아 재현한다(추세 비교용).
    win["variant"]=="completed" 면 주/월 재표집에 completed_only 를 적용한다.
    """
    if tf not in TFS:
        raise ValueError(f"tf 는 {TFS} 중 하나여야 함: {tf!r}")
    if basket not in BASKETS:
        raise ValueError(f"basket 은 {BASKETS} 중 하나여야 함: {basket!r}")
    if not isinstance(cut, (int, np.integer)) or isinstance(cut, bool) or cut < 0:
        raise ValueError(f"cut 은 0 이상의 정수여야 함: {cut!r}")
    cut = int(cut)
    variant = win.get("variant", "faithful")
    if variant not in VARIANTS:
        raise ValueError(f"win['variant'] 가 잘못됨: {variant!r}")
    completed = variant == "completed"

    lb = V0["lookback"][tf]
    spy_d = win["spy"]["Close"]
    if cut:
        spy_d = spy_d.iloc[:-cut]
    if spy_d.empty:
        raise ValueError(f"SPY 창이 비어 있음 (cut={cut})")
    asof = spy_d.index[-1]

    fang_d = win["fang"].loc[:asof]
    vix_d = win["vix"].loc[:asof]
    btc_d = win["btc"]                        # BTC는 주말 포함 최신 유지 (reference 동일)
    if cut:
        btc_d = btc_d.loc[:asof]
    watch_d = win["watch"].loc[:asof]
    if fang_d.empty or vix_d.empty or btc_d.empty:
        raise ValueError(f"{asof:%Y-%m-%d} 기준 FANG/VIX/BTC 창 중 빈 것이 있음")

    # 타임프레임 변환
    spy = resample_close(spy_d, tf, completed)
    if basket == "v0":
        basket_d = fang_d.div(fang_d.iloc[0]).mean(axis=1)      # 창 첫날 정규화 평균 (reference 동일)
    else:
        basket_d = _equal_weight_basket(fang_d)
    basket_s = resample_close(basket_d, tf, completed)
    vix = resample_close(vix_d, tf, completed)
    btc = resample_close(btc_d, tf, completed)
    if spy.empty or vix.empty or btc.empty:
        raise ValueError(f"{asof:%Y-%m-%d} {tf} 재표집 결과가 비어 있음 (variant={variant})")

    # P5: 180일선 (일간 기준, 전 타임프레임 공통)
    ma_ser = spy_d.rolling(V0["ma_window"]).mean().dropna()
    if ma_ser.empty:
        raise ValueError(f"SPY 창({len(spy_d)}봉)이 180일선 계산에 부족함 (asof {asof:%Y-%m-%d})")
    ma_dist = float((spy_d.iloc[-1] / ma_ser.iloc[-1] - 1) * 100)
    ma_slope = float((ma_ser.iloc[-1] / ma_ser.iloc[-21] - 1) * 100) if len(ma_ser) > 21 else 0.0
    below = ma_dist <= 0

    # P1
    fang_rs = ret(basket_s, lb) - ret(spy, lb)
    # P2
    phase, streak, hist = macd_phase(spy, V0["turn_k"][tf])
    # P4
    vix_now = float(vix.iloc[-1])
    vix_d_ = float(vix.iloc[-1] - vix.iloc[-1 - lb]) if len(vix) > lb else 0.0
    # P6 (일간 고유 신호 — 전 타임프레임 공통 표기)
    eod = win["eod"]
    if cut:
        eod = eod[eod.index <= asof]
    streak_eod = 0
    for x in eod.iloc[::-1]:
        if x:
            streak_eod += 1
        else:
            break
    eod20 = int(eod.iloc[-20:].sum())
    # P7 (일간 기준)
    leaders = []
    spy_hi = spy_d.rolling(60).max().iloc[-1]
    spy_dd = float(spy_d.iloc[-1] / spy_hi - 1)
    for t in watch_d.columns:
        s = watch_d[t].dropna()
        if len(s) < MIN_WATCH_BARS:      # 상장 초기 등 데이터 부족 종목 제외
            continue
        if below:
            dd = float(s.iloc[-1] / s.rolling(60).max().iloc[-1] - 1)
            if dd >= 0.6 * spy_dd and ret(s, 5) > 0:
                leaders.append(t)
        else:
            if ret(s, 20) - ret(spy_d, 20) > 3:
                leaders.append(t)
    # P9
    btc_mom = ret(btc, BTC_MOM_K[tf])
    up_days = 0
    for i in range(len(btc_d) - 1, 0, -1):
        if btc_d.iloc[i] > btc_d.iloc[i - 1]:
            up_days += 1
        else:
            break
    btc_turned = btc_mom > 2 and up_days >= 2
    # P3
    fg = win["fg"]
    fg_score = fg["score"] if fg else None
    fg_delta = fg[f"d_{tf}"] if fg else 0

    return {
        "belowMA": below, "maDist": ma_dist, "maSlope": ma_slope,
        "fangRS": fang_rs, "macd": phase, "histStreak": streak,
        "fg": fg_score, "fgD": fg_delta,
        "vix": vix_now, "vixD": vix_d_,
        "eod": streak_eod, "eod20": eod20, "leaders": leaders,
        "btcMom": btc_mom, "btcTurned": btc_turned, "btcUpDays": up_days,
        "lb": lb,
    }


# ============================================================
# 신호 평가 — reference assess 이식 (P1~P9)
# ============================================================
def v0_assess(m: dict) -> dict:
    s = {}
    # P1 FANG 상대강도
    if m["belowMA"]:
        s["fang"] = "R2G" if m["fangRS"] >= 1.5 else "RED" if m["fangRS"] <= -1.5 else "AMBER"
    else:
        s["fang"] = "G2R" if m["fangRS"] <= -2 else "GREEN" if m["fangRS"] >= 0 else "AMBER"
    # P2 MACD
    s["macd"] = {"down": "RED", "turn": "R2G", "up": "GREEN", "roll": "G2R"}[m["macd"]]
    # P3 F&G
    if m["fg"] is None:
        s["fg"] = "AMBER"
    elif m["fg"] <= 10:
        s["fg"] = "R2G"
    elif m["fg"] <= 25:
        s["fg"] = "R2G" if m["fgD"] > 0 else "RED"
    elif m["fg"] >= 75:
        s["fg"] = "G2R" if m["fgD"] < 0 else "GREEN"
    elif m["fg"] >= 55:
        s["fg"] = "GREEN"
    else:
        s["fg"] = "AMBER"
    # P4 VIX
    if m["vix"] >= 40:
        s["vix"] = "R2G" if m["vixD"] <= 0 else "RED"
    elif m["vix"] >= 28:
        s["vix"] = "RED"
    elif m["vix"] <= 17:
        s["vix"] = "GREEN"
    elif m["vixD"] >= 4 and not m["belowMA"]:
        s["vix"] = "G2R"
    else:
        s["vix"] = "AMBER"
    # P5 180일선
    s["ma"] = "R2G" if m["maDist"] <= 0 else "AMBER" if m["maDist"] <= 3 else "GREEN"
    # P6 장마감 매수
    if m["belowMA"]:
        s["eod"] = "R2G" if m["eod"] >= 3 else "RED" if m["eod"] == 0 else "AMBER"
    else:
        s["eod"] = "GREEN" if m["eod"] >= 3 else "AMBER"
    # P7 주도주
    n = len(m["leaders"])
    if n >= 5:
        s["lead"] = "R2G" if m["belowMA"] else "GREEN"
    elif n <= 2:
        s["lead"] = "RED" if m["belowMA"] else "AMBER"
    else:
        s["lead"] = "AMBER"
    # P9 BTC
    if m["belowMA"]:
        s["btc"] = "R2G" if m["btcTurned"] else "RED"
    elif m["btcMom"] < -3:
        s["btc"] = "G2R"
    else:
        s["btc"] = "GREEN" if m["btcMom"] > 0 else "AMBER"
    return s


# ============================================================
# 종합 — reference composite / overall / combo_advice 이식
# ============================================================
def v0_composite(states: dict, fg_missing: bool, eod_missing: bool = False, lead_missing: bool = False) -> float:
    """가중 평균 점수. reference 와 같은 누적 순서(states 의 삽입 순서)·같은 float 산술.

    v0: fg_missing 이면 fg 가중치 제외. 재현 확장: 데이터가 없는 P6(eod_missing)·P7(lead_missing) 도
    같은 방식으로 제외 — VALIDATION.md 의 '대체 규약'. 남는 가중치가 0 이면 ValueError.
    """
    skip = set()
    if fg_missing:
        skip.add("fg")
    if eod_missing:
        skip.add("eod")
    if lead_missing:
        skip.add("lead")
    tot = w = 0.0
    for k, st in states.items():
        if k in skip:
            continue
        if k not in V0_WEIGHTS:
            raise ValueError(f"알 수 없는 신호 키: {k!r} (허용: {V0_SIGNALS})")
        if st not in V0_SC:
            raise ValueError(f"알 수 없는 상태 {st!r} (신호 {k}; 허용: {tuple(V0_SC)})")
        tot += V0_SC[st] * V0_WEIGHTS[k]
        w += V0_WEIGHTS[k]
    if w == 0.0:
        raise ValueError("composite: 가중치가 남지 않음 (모든 신호가 제외됨)")
    return tot / w


def v0_overall(score: float, trend: int) -> str:
    b = V0["band"]
    if score >= b:
        return "G2R" if trend < 0 else "GREEN"
    if score <= -b:
        return "R2G" if trend > 0 else "RED"
    return "R2G" if trend > 0 else "G2R" if trend < 0 else "AMBER"


def v0_combo(mo: str, wk: str, dy: str, lang: str = "ko") -> tuple:
    """월간/주간/일간 종합 상태 조합 → (국면명, 권장 행동, 톤). reference combo_advice 와 동일 규칙·문구.
    위에서부터 첫 일치 규칙 적용."""
    for v in (mo, wk, dy):
        if v not in V0_SC:
            raise ValueError(f"알 수 없는 상태: {v!r} (허용: {tuple(V0_SC)})")
    if lang not in ("ko", "en"):
        raise ValueError(f"lang 은 ko/en: {lang!r}")
    UP, DN = ("GREEN", "R2G"), ("RED", "G2R")
    ko = lang == "ko"
    rules = [
        (mo in DN and wk in DN and dy in DN,
         ("시장 전체가 내리막", "주식은 줄이고 현금을 확보하세요. 잘 버티는 종목만 지켜보기") if ko else
         ("Broad market in decline", "Reduce stock exposure and raise cash. Just watch the names holding up"), "reduce"),
        (dy == "R2G" and wk in DN and mo in DN,
         ("바닥 신호가 살짝 보임", "아직 확실하지 않습니다. 며칠 더 지켜보고, 사더라도 아주 조금만") if ko else
         ("Faint bottoming signal", "Not confirmed yet. Watch a few more days; if you buy, buy only a little"), "caution"),
        (wk == "R2G" and dy in UP and mo != "GREEN",
         ("바닥을 다지는 중", "관심 우량주를 소액으로 나눠서 사기 시작 (1단계)") if ko else
         ("Building a bottom", "Start buying quality watchlist names in small installments (stage 1)"), "buy"),
        (mo in ("RED", "AMBER", "G2R") and wk in UP and dy == "GREEN",
         ("반등이 확인됨", "사는 양을 한 단계 늘리기 (2단계)") if ko else
         ("Rebound confirmed", "Step up the buying one notch (stage 2)"), "buy"),
        (mo == "R2G" and (wk in DN or dy in DN),
         ("신호가 엇갈림", "큰 흐름은 좋아지는데 단기가 흔들립니다. 추가 매수는 잠시 멈추고 기다리기") if ko else
         ("Mixed signals", "The big trend is improving but the short term is shaky. Pause new buying and wait"), "caution"),
        (mo == "R2G",
         ("큰 흐름이 좋아지는 중", "목표한 비중까지 조금씩 늘려 가기") if ko else
         ("Big trend improving", "Gradually build toward your target allocation"), "buy"),
        (mo == "GREEN" and wk not in DN and dy in ("RED", "R2G"),
         ("상승장 속 잠깐 쉬어가는 구간", "좋은 종목을 싸게 살 기회입니다. 나눠서 매수") if ko else
         ("Brief pause within an uptrend", "A chance to buy good names cheaply. Buy in installments"), "buy"),
        (mo == "GREEN" and wk == "GREEN" and dy == "G2R",
         ("단기 과열 뒤 주춤", "새로 사는 건 잠시 쉬고, 가진 것은 그대로 유지") if ko else
         ("Stalling after short-term overheating", "Hold off on new buys; keep what you own"), "caution"),
        (mo == "GREEN" and wk in DN,
         ("조정이 올 수 있음", "이익 난 것 일부는 팔고, 새 매수는 중단") if ko else
         ("A correction may be coming", "Take some profits and stop new buying"), "reduce"),
        (mo == "GREEN" and wk in ("GREEN", "AMBER") and dy in ("GREEN", "AMBER"),
         ("꾸준한 상승 흐름", "그대로 보유. 쉬어가는 구간이 오면 추가 매수 고려") if ko else
         ("Steady uptrend", "Stay invested. Consider adding on pullbacks"), "hold"),
    ]
    for cond, (name, act), tone in rules:
        if cond:
            return name, act, tone
    avg = (V0_SC[mo] + V0_SC[wk] + V0_SC[dy]) / 3
    if avg >= 0.35:
        return (("좋은 쪽에 가까움", "그대로 보유") if ko else ("Leaning positive", "Stay invested")) + ("hold",)
    if avg <= -0.35:
        return (("나쁜 쪽에 가까움", "방어 위주로. 현금 확보") if ko else ("Leaning negative", "Play defense. Raise cash")) + ("reduce",)
    return (("방향이 뚜렷하지 않음", "일단 관망하며 가진 것만 유지") if ko else
            ("No clear direction", "Wait and see; just hold what you own")) + ("neutral",)


# ============================================================
# 하루 전체 — reference build_payload 의 계산 부분과 동일 흐름
# ============================================================
def v0_day(bundle, asof, variant: str = "faithful", basket: str = "v0") -> dict:
    """asof 하루의 v0 판정: 3개 타임프레임 × (now, prev=cut) → states, composite, trend, overall, combo.

    반환 키(계약): asof('YYYY-MM-DD'), states_d/w/m, score_d/w/m, trend_d/w/m, overall_d/w/m, tone, verdict_ko,
                   n_watch_avail, fg_avail, eod_avail, leaders
    부가 키: action_ko, variant, basket, spy_close, vix_close, metrics_d/w/m(원시 메트릭), warnings(list)
    """
    if basket not in BASKETS:
        raise ValueError(f"basket 은 {BASKETS} 중 하나여야 함: {basket!r}")
    win = window(bundle, asof, variant)
    meta = win["meta"]
    eod_missing = not meta["eod_avail"]
    lead_missing = meta["n_watch_avail"] == 0

    tfs, mets, states = {}, {}, {}
    for tf in TFS:
        m_now = v0_metrics(win, tf, cut=0, basket=basket)
        m_prev = v0_metrics(win, tf, cut=V0["trend_cut"][tf], basket=basket)
        st_now, st_prev = v0_assess(m_now), v0_assess(m_prev)
        if tf != "daily":
            for k in V0_DAILY_ONLY:
                st_now.pop(k, None)
                st_prev.pop(k, None)
        fg_missing = m_now["fg"] is None
        sc_now = v0_composite(st_now, fg_missing, eod_missing, lead_missing)
        sc_prev = v0_composite(st_prev, fg_missing, eod_missing, lead_missing)
        dsc = sc_now - sc_prev
        trend = 1 if dsc > V0["trend_eps"] else -1 if dsc < -V0["trend_eps"] else 0
        tfs[tf] = {"score": sc_now, "trend": trend, "overall": v0_overall(sc_now, trend)}
        mets[tf], states[tf] = m_now, st_now

    mo, wk, dy = (tfs[t]["overall"] for t in ("monthly", "weekly", "daily"))
    name, action, tone = v0_combo(mo, wk, dy, "ko")
    sfx = {"daily": "d", "weekly": "w", "monthly": "m"}
    out = {
        "asof": win["asof"].strftime("%Y-%m-%d"),
        "variant": variant, "basket": basket,
        "tone": tone, "verdict_ko": name, "action_ko": action,
        "n_watch_avail": meta["n_watch_avail"],
        "fg_avail": bool(meta["fg_avail"]), "eod_avail": bool(meta["eod_avail"]),
        "leaders": list(mets["daily"]["leaders"]),
        "spy_close": float(win["spy"]["Close"].iloc[-1]),
        "vix_close": float(win["vix"].iloc[-1]),
        "warnings": list(meta["warnings"]),
    }
    for tf in TFS:
        x = sfx[tf]
        out[f"states_{x}"] = states[tf]
        out[f"score_{x}"] = tfs[tf]["score"]
        out[f"trend_{x}"] = tfs[tf]["trend"]
        out[f"overall_{x}"] = tfs[tf]["overall"]
        out[f"metrics_{x}"] = mets[tf]
    return out
