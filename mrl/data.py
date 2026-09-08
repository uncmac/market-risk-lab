# -*- coding: utf-8 -*-
"""mrl.data — 데이터 캐시 계층 (ARCHITECTURE.md 「데이터 계약 — mrl/data.py」 구현).

역할
  * yfinance 일괄 다운로드(조정 종가 · SPY OHLCV), CBOE 변동성 기간구조 CSV, CNN Fear & Greed 이력,
    SPY 마지막 1시간봉(15:30~16:00 ET)을 수집해 `data/` 에 CSV 로 저장한다.
  * 가격(close/spy_ohlc)은 매번 전량 재구축한다 — 자동조정(배당·분할) 재계산 때문에 증분 append 금지.
  * fg/eod/cboe 는 기존 파일에 새 날짜를 append 한다(fg/cboe 는 셀 단위: 새 값 우선·없는 셀은 기존 유지,
    eod 는 행 단위: 같은 날짜는 새 행으로 대체).
  * `apply_guards` 가 (a) SPY 마지막 일자 이후의 유령 행을 제거(24/7 자산 제외)하고
    (b) SPY 대비 3거래일 이상 뒤처진 티커를 meta["warnings"] 에 기록한다.

원칙
  * 모든 인덱스는 tz-naive `DatetimeIndex`(자정), 이름 "date". CSV 는 `index_label="date"`, 소수 6자리, LF.
  * 실패는 조용히 넘기지 않는다: 원천 조회 실패는 예외, 부분 실패는 `warnings.warn` + meta["warnings"].
  * 점(point-in-time) 원칙: 미완성 세션(오늘 장중)의 EOD 봉은 저장하지 않고, SPY 마지막 봉이
    장중 부분 봉일 가능성은 meta["spy_last_bar_complete"] 로 알린다.
  * 퇴행 거부: yfinance 는 티커별 실패(레이트리밋 429 등)를 예외 없이 전부-NaN 열로 돌려준다. 기존 close.csv 에 있던
    티커가 사라지거나 행 수가 크게 줄거나 SPY 마지막 일자가 뒤로 가면 `_check_against_previous` 가 RuntimeError 로
    저장을 거부한다 (`force=True` 로만 우회). 30년 이력이 조용히 지워진 캐시가 커밋되는 일을 막는다.
  * 원자적 저장: 여섯 파일을 모두 `.tmp` 에 쓰고 fsync 한 뒤에야 `os.replace` 로 바꿔 넣는다. 중간에 죽어도 기존
    캐시는 그대로다. `load_cache` 는 meta.spy_last 와 close/spy_ohlc 의 마지막 일자가 어긋나면 예외를 낸다.

외부 원천 형식 (2026-09-07 검증)
  * yf.download(…, period="max"): 열 MultiIndex(Price, Ticker), 일자 인덱스 tz-naive. 존재하지 않는
    티커는 예외 없이 전부-NaN 열로 온다 → meta.warnings 에 기록하고 캐시에서 제외.
  * CBOE: VIX3M/VIX9D 는 `DATE,OPEN,HIGH,LOW,CLOSE`, VVIX/SKEW 는 `DATE,<이름>` (단일 값 열).
  * CNN graphdata/{date}: `fear_and_greed_historical.data` 가 점수, 9개 구성요소 키에 각각 `data` 목록.
    포인트 x 는 거래일 자정 UTC(epoch ms). 마지막 포인트만 자정이 아닌 "라이브" 시각이라 날짜가 중복된다.
  * SPY 1h 봉: 세션 마지막 봉은 15:30 ET(= 마감 30분 봉), 반일장은 11:30 ET.
"""
from __future__ import annotations

import copy
import datetime as dt
import io
import json
import os
import platform
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from mrl.config import (ALL_TICKERS, CBOE_SERIES, CNN_FG_SEED_DATE, CNN_FG_URL,
                        CNN_HEADERS, DATA_DIR, ET, TWENTY_FOUR_SEVEN)

__all__ = [
    "Bundle", "fetch_prices", "split_prices", "fetch_cboe", "parse_cboe_csv",
    "fetch_fg_history", "parse_fg_json", "fetch_fg_today", "fetch_spy_eod",
    "eod_from_intraday", "build_cache", "update_daily", "load_cache", "save_cache",
    "apply_guards", "FILES", "SPY_OHLC_COLUMNS", "CBOE_COLUMNS", "FG_COMPONENTS",
    "FG_COLUMNS", "EOD_COLUMNS", "SCHEMA_VERSION", "STALE_SESSIONS",
]

# ------------------------------------------------------------------
# 상수
# ------------------------------------------------------------------
SCHEMA_VERSION = 1
FILES = {
    "close": "close.csv",
    "spy_ohlc": "spy_ohlc.csv",
    "cboe": "cboe.csv",
    "fg": "fg_history.csv",
    "eod": "spy_eod.csv",
    "meta": "meta.json",
}
SPY_OHLC_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
CBOE_COLUMNS = list(CBOE_SERIES)                       # VIX3M, VIX9D, VVIX, SKEW
FG_COMPONENTS = [
    "market_momentum_sp500", "market_momentum_sp125", "stock_price_strength",
    "stock_price_breadth", "put_call_options", "market_volatility_vix",
    "market_volatility_vix_50", "junk_bond_demand", "safe_haven_demand",
]
FG_COLUMNS = ["score"] + FG_COMPONENTS
EOD_COLUMNS = ["open", "close", "ret", "half_day", "bar_start"]
FULL_DAY_LAST_BAR = "15:30"        # 정규장 마지막 1h 봉 시작 시각 (ET)
HALF_DAY_LAST_BAR = "11:30"        # 반일장(13:00 마감) 마지막 1h 봉 시작 시각 (ET)
SESSION_CLOSE_FULL = dt.time(16, 5)    # 이 시각(ET) 이후에만 오늘 정규장 마지막 봉을 확정으로 본다
SESSION_CLOSE_HALF = dt.time(13, 5)    # 반일장 동일
STALE_SESSIONS = 3                 # SPY 대비 이만큼 뒤처지면 경고
MAX_ROW_SHRINK = 0.90              # 티커의 비-NaN 행 수가 이전 캐시의 90% 미만이면 불량 다운로드로 보고 저장 거부
CSV_FLOAT_FORMAT = "%.6f"
HTTP_TIMEOUT = 30
RETRIES = 3
RETRY_SLEEP = 5                    # 초 × 시도 횟수
_ET = ZoneInfo(ET)


# ------------------------------------------------------------------
# 번들
# ------------------------------------------------------------------
@dataclass
class Bundle:
    """캐시 한 벌. 모든 프레임 인덱스는 tz-naive DatetimeIndex(이름 "date")."""
    close: pd.DataFrame          # 조정 종가. 열 = ALL_TICKERS 중 존재하는 것. 인덱스 = 거래일(BTC 주말 포함 union)
    spy_ohlc: pd.DataFrame       # SPY Open/High/Low/Close/Volume (조정)
    cboe: pd.DataFrame           # 열 VIX3M, VIX9D, VVIX, SKEW
    fg: pd.DataFrame             # 열 score + CNN 구성요소. 2020-08-03~
    eod: pd.DataFrame            # SPY 세션 마지막 1시간봉: open, close, ret(=close/open-1), half_day, bar_start
    meta: dict = field(default_factory=dict)   # fetched_at_utc, tickers{first,last,n}, yfinance_version, warnings[...]


# ------------------------------------------------------------------
# 공용 헬퍼
# ------------------------------------------------------------------
def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _now_et() -> dt.datetime:
    return dt.datetime.now(_ET)


def _iso_utc(ts: dt.datetime | pd.Timestamp) -> str:
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _dstr(ts) -> str | None:
    """Timestamp → 'YYYY-MM-DD' (None/NaT 는 None)."""
    if ts is None or pd.isna(ts):
        return None
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _record(warn_list: list, msg: str) -> None:
    """경고를 목록에 기록(중복 제거)하고 warnings.warn 으로도 알린다 — 조용한 실패 금지."""
    if msg not in warn_list:
        warn_list.append(msg)
    warnings.warn(msg, stacklevel=3)


def _naive_daily_index(idx) -> pd.DatetimeIndex:
    """tz-aware 든 naive 든 tz-naive 자정 DatetimeIndex 로 (yfinance 일봉은 거래소 현지 자정)."""
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    idx = idx.normalize()
    idx.name = "date"
    return idx


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    """열만 있는 빈 프레임 (float, 인덱스 이름 date)."""
    df = pd.DataFrame({c: pd.Series(dtype=float) for c in columns})
    df.index = pd.DatetimeIndex([], name="date")
    return df


def _to_bool(s: pd.Series) -> pd.Series:
    """CSV 왕복 후의 'True'/'False'/1/0/NaN 을 bool 로."""
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes"])


def _coerce_eod(df: pd.DataFrame) -> pd.DataFrame:
    """eod 프레임의 열·dtype 을 계약대로 고정한다."""
    if df is None or df.empty:
        out = _empty_frame(["open", "close", "ret"])
        out["half_day"] = pd.Series(dtype=bool)
        out["bar_start"] = pd.Series(dtype=str)
        return out[EOD_COLUMNS]
    out = df.reindex(columns=EOD_COLUMNS).copy()
    for c in ("open", "close", "ret"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    out["half_day"] = _to_bool(out["half_day"])
    out["bar_start"] = out["bar_start"].fillna("").astype(str)
    out.index = _naive_daily_index(out.index)
    return out.sort_index()


def _frame_stats(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"rows": 0, "first": None, "last": None, "columns": list(df.columns) if df is not None else []}
    return {"rows": int(len(df)), "first": _dstr(df.index[0]), "last": _dstr(df.index[-1]),
            "columns": [str(c) for c in df.columns]}


def _ticker_stats(close: pd.DataFrame) -> dict:
    out = {}
    for t in close.columns:
        s = close[t].dropna()
        out[str(t)] = ({"first": _dstr(s.index[0]), "last": _dstr(s.index[-1]), "n": int(len(s))}
                       if not s.empty else {"first": None, "last": None, "n": 0})
    return out


# ------------------------------------------------------------------
# HTTP / yfinance (재시도 포함)
# ------------------------------------------------------------------
def _http_get(url: str, headers: dict | None = None) -> requests.Response:
    last_err: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
            if r.status_code == 200:
                return r
            last_err = RuntimeError(f"HTTP {r.status_code}, 응답 앞부분 {r.text[:60]!r}")
            if 400 <= r.status_code < 500 and r.status_code != 429:
                break                     # 4xx 는 재시도해도 같다
        except requests.RequestException as e:
            last_err = e
        if attempt < RETRIES:
            time.sleep(RETRY_SLEEP * attempt)
    raise RuntimeError(f"GET {url} 실패: {last_err}")


def _http_get_json(url: str, headers: dict | None = None) -> dict:
    r = _http_get(url, headers)
    if not r.text.lstrip().startswith("{"):
        raise RuntimeError(f"GET {url}: JSON 이 아님, 응답 앞부분 {r.text[:60]!r}")
    return r.json()


def _yf_download(tickers, **kwargs) -> pd.DataFrame:
    """yf.download 래퍼 — 일시적 실패(레이트리밋 등)에 대비해 최대 RETRIES 회 시도."""
    import yfinance as yf   # 지연 임포트: 테스트는 네트워크 없이 파서만 검사한다
    last_err: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            df = yf.download(tickers, auto_adjust=True, threads=True, progress=False, **kwargs)
            if df is not None and not df.empty:
                return df
            last_err = RuntimeError("빈 응답")
        except Exception as e:      # noqa: BLE001 — 재시도 후 마지막 오류를 예외로 올린다
            last_err = e
        if attempt < RETRIES:
            time.sleep(RETRY_SLEEP * attempt)
    label = tickers if isinstance(tickers, str) else f"{len(tickers)}개 티커"
    raise RuntimeError(f"yfinance 다운로드 실패 ({RETRIES}회 시도, {label}, {kwargs}): {last_err}")


# ------------------------------------------------------------------
# 가격 (yfinance 일괄)
# ------------------------------------------------------------------
def fetch_prices(tickers: list[str], period: str = "max") -> pd.DataFrame:
    """yf.download 일괄 조회 (auto_adjust=True, threads=True, progress=False).

    반환: 열 = MultiIndex(Price ∈ Open/High/Low/Close/Volume, Ticker), 인덱스 = tz-naive 일자.
    존재하지 않거나 전부 NaN 인 티커는 예외 대신 `df.attrs["missing"]` 에 담는다
    (호출자가 meta.warnings 에 기록). 종가·SPY OHLC 분리는 `split_prices()`.
    """
    tickers = list(dict.fromkeys(tickers))      # 순서 보존 중복 제거
    if not tickers:
        raise ValueError("fetch_prices: tickers 가 비어 있음")
    raw = _yf_download(tickers, period=period)
    if not isinstance(raw.columns, pd.MultiIndex):
        # 단일 티커 구형 반환(평평한 열) → MultiIndex 로 승격해 형식을 통일
        raw.columns = pd.MultiIndex.from_product([list(raw.columns), [tickers[0]]],
                                                 names=["Price", "Ticker"])
    if "Close" not in raw.columns.get_level_values(0):
        raise RuntimeError("fetch_prices: 응답에 Close 열이 없음")
    raw.index = _naive_daily_index(raw.index)
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    close = raw["Close"]
    raw.attrs["missing"] = [t for t in tickers if t not in close.columns or close[t].isna().all()]
    raw.attrs["period"] = period
    return raw


def split_prices(raw: pd.DataFrame, tickers: list[str] | None = None
                 ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """fetch_prices 결과 → (close, spy_ohlc, missing).

    close   : 조정 종가. 열 = 요청 티커 중 데이터가 있는 것(정렬). 인덱스 = 각 티커 일자의 합집합 중
              SPY 첫 거래일(1993-01-29) 이후 — SPY 가 진실 인덱스이므로 그 이전(^GSPC 1927~ 등)은 버린다.
    spy_ohlc: SPY Open/High/Low/Close/Volume(조정). 인덱스 = SPY 거래일.
    missing : 응답에 없거나 전부 NaN 인 티커. SPY 가 없으면 캐시 자체가 무의미하므로 RuntimeError.
    """
    if not isinstance(raw.columns, pd.MultiIndex):
        raise ValueError("split_prices: fetch_prices 의 MultiIndex 프레임이 필요")
    close_all = raw["Close"]
    if tickers is None:
        tickers = list(close_all.columns)
    missing = [t for t in tickers if t not in close_all.columns or close_all[t].isna().all()]
    if "SPY" in missing or "SPY" not in tickers:
        raise RuntimeError("split_prices: SPY 종가가 응답에 없음 — 캐시를 만들 수 없다")
    keep = sorted(t for t in tickers if t not in missing)
    close = close_all[keep].astype(float)
    spy = close["SPY"].dropna()
    close = close.loc[close.index >= spy.index[0]].dropna(how="all")
    close.index = _naive_daily_index(close.index)
    close.columns.name = None

    spy_ohlc = raw.xs("SPY", axis=1, level=1)
    cols = [c for c in SPY_OHLC_COLUMNS if c in spy_ohlc.columns]
    if "Close" not in cols:
        raise RuntimeError("split_prices: SPY OHLC 에 Close 가 없음")
    spy_ohlc = spy_ohlc[cols].dropna(subset=["Close"]).astype(float)
    spy_ohlc.index = _naive_daily_index(spy_ohlc.index)
    spy_ohlc.columns.name = None
    return close, spy_ohlc, missing


# ------------------------------------------------------------------
# CBOE 기간구조
# ------------------------------------------------------------------
def parse_cboe_csv(text: str, name: str) -> pd.Series:
    """CBOE 일별 CSV 텍스트 → float Series(이름 name, 인덱스 date).
    `DATE,OPEN,HIGH,LOW,CLOSE` 면 CLOSE, `DATE,<이름>` 이면 마지막 값 열을 쓴다."""
    df = pd.read_csv(io.StringIO(text))
    df.columns = [str(c).strip().upper() for c in df.columns]
    if "DATE" not in df.columns or df.shape[1] < 2:
        raise ValueError(f"CBOE {name}: 예상 밖 헤더 {list(df.columns)}")
    val_col = "CLOSE" if "CLOSE" in df.columns else [c for c in df.columns if c != "DATE"][-1]
    dates = pd.to_datetime(df["DATE"].astype(str).str.strip(), format="%m/%d/%Y", errors="coerce")
    if dates.isna().all():                       # 형식이 바뀌면 일반 파서로 재시도
        dates = pd.to_datetime(df["DATE"].astype(str).str.strip(), errors="coerce")
    vals = pd.to_numeric(df[val_col], errors="coerce")
    s = pd.Series(vals.values, index=pd.DatetimeIndex(dates), name=name, dtype=float)
    s = s[s.index.notna() & s.notna()]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if s.empty:
        raise ValueError(f"CBOE {name}: 파싱 결과가 비어 있음")
    s.index = _naive_daily_index(s.index)
    return s


def fetch_cboe(series: dict[str, str] | None = None) -> pd.DataFrame:
    """CBOE 무료 CSV(VIX3M·VIX9D·VVIX·SKEW) → 열별 outer join 프레임.
    일부 계열 실패는 `df.attrs["warnings"]` + warnings.warn, 전부 실패면 RuntimeError."""
    series = dict(CBOE_SERIES if series is None else series)
    cols, notes = {}, []
    for name, url in series.items():
        try:
            cols[name] = parse_cboe_csv(_http_get(url).text, name)
        except Exception as e:      # noqa: BLE001 — 계열별 실패를 모아 보고
            notes.append(f"CBOE {name} 조회 실패: {e}")
    if not cols:
        raise RuntimeError("CBOE 전 계열 조회 실패: " + "; ".join(notes))
    df = pd.concat(list(cols.values()), axis=1).sort_index()
    df = df.reindex(columns=[c for c in series if c in df.columns]).astype(float)
    df.index = _naive_daily_index(df.index)
    df.columns.name = None
    df.attrs["warnings"] = notes
    for n in notes:
        warnings.warn(n, stacklevel=2)
    return df


# ------------------------------------------------------------------
# CNN Fear & Greed
# ------------------------------------------------------------------
def _fg_point_date(x_ms) -> pd.Timestamp:
    """포인트 x(epoch ms) → 날짜. 자정 UTC 포인트는 그 UTC 날짜(=거래일), 라이브 포인트는 ET 날짜."""
    ts = pd.Timestamp(int(x_ms), unit="ms", tz="UTC")
    if ts == ts.normalize():
        return ts.tz_localize(None)
    return ts.tz_convert(ET).tz_localize(None).normalize()


def _fg_points_to_series(points, name: str) -> pd.Series:
    rows: dict = {}
    for p in points or []:
        x, y = p.get("x"), p.get("y")
        if x is None or y is None:
            continue
        try:
            rows[_fg_point_date(x)] = float(y)     # 같은 날짜면 나중 포인트(라이브)가 이긴다
        except (TypeError, ValueError, OverflowError):
            continue
    s = pd.Series(rows, dtype=float, name=name)
    s.index = pd.DatetimeIndex(s.index, name="date")
    return s.sort_index()


def parse_fg_json(j: dict) -> pd.DataFrame:
    """graphdata JSON → 열 score + 구성요소(FG_COLUMNS 순서), 날짜별 1행(outer join; 구성요소 결측 허용).
    attrs: warnings, last_point_utc, last_point_is_live."""
    hist = (j.get("fear_and_greed_historical") or {}).get("data")
    if not hist:
        raise ValueError("CNN 응답에 fear_and_greed_historical.data 가 없음")
    parts = [_fg_points_to_series(hist, "score")]
    notes: list[str] = []
    for k in FG_COMPONENTS:
        pts = (j.get(k) or {}).get("data") if isinstance(j.get(k), dict) else None
        if not pts:
            notes.append(f"CNN 구성요소 {k} 가 응답에 없음")
            continue
        parts.append(_fg_points_to_series(pts, k))
    extra = sorted(k for k, v in j.items()
                   if isinstance(v, dict) and isinstance(v.get("data"), list)
                   and k not in FG_COMPONENTS and k != "fear_and_greed_historical")
    if extra:
        notes.append(f"CNN 응답의 미등록 구성요소 무시(열 고정): {extra}")
    df = pd.concat(parts, axis=1).reindex(columns=FG_COLUMNS).astype(float).sort_index()
    df.index = _naive_daily_index(df.index)
    df.columns.name = None
    xs = [p["x"] for p in hist if p.get("x") is not None]
    last_ts = pd.Timestamp(int(max(xs)), unit="ms", tz="UTC")
    df.attrs["warnings"] = notes
    df.attrs["last_point_utc"] = _iso_utc(last_ts)
    df.attrs["last_point_is_live"] = bool(last_ts != last_ts.normalize())
    return df


def fetch_fg_history(seed_date: str = CNN_FG_SEED_DATE) -> pd.DataFrame:
    """graphdata/{seed_date} → seed_date 이후 일별 점수+구성요소. 2020-08-03 이전 이력은 없다."""
    url = CNN_FG_URL.format(date=seed_date)
    df = parse_fg_json(_http_get_json(url, CNN_HEADERS))
    df.attrs["seed_date"] = str(seed_date)
    for n in df.attrs.get("warnings", []):
        warnings.warn(n, stacklevel=2)
    return df


def fetch_fg_today() -> dict:
    """graphdata(날짜 없음) → 오늘 점수·구성요소 dict (JSON 직렬화 가능).
    d_daily/d_weekly/d_monthly 는 v0 와 같이 round(score - previous_*)."""
    base = CNN_FG_URL.split("{date}")[0].rstrip("/")
    j = _http_get_json(base, CNN_HEADERS)
    cur = j.get("fear_and_greed") or {}
    if "score" not in cur:
        raise RuntimeError("CNN 응답에 fear_and_greed.score 가 없음")
    score = float(cur["score"])

    def _f(k):
        v = cur.get(k)
        return None if v is None else float(v)

    ts = pd.Timestamp(cur["timestamp"]) if cur.get("timestamp") else pd.Timestamp.now(tz="UTC")
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    prev_c, prev_w, prev_m = _f("previous_close"), _f("previous_1_week"), _f("previous_1_month")
    comps = {}
    for k in FG_COMPONENTS:
        c = j.get(k)
        if not isinstance(c, dict):
            continue
        pts = c.get("data") or []
        comps[k] = {"score": (None if c.get("score") is None else float(c["score"])),
                    "rating": c.get("rating"),
                    "value": (float(pts[-1]["y"]) if pts and pts[-1].get("y") is not None else None)}
    return {
        "score": score, "rating": cur.get("rating"),
        "timestamp_utc": _iso_utc(ts), "asof_date": ts.tz_convert(ET).strftime("%Y-%m-%d"),
        "previous_close": prev_c, "previous_1_week": prev_w, "previous_1_month": prev_m,
        "previous_1_year": _f("previous_1_year"),
        "d_daily": (None if prev_c is None else int(round(score - prev_c))),
        "d_weekly": (None if prev_w is None else int(round(score - prev_w))),
        "d_monthly": (None if prev_m is None else int(round(score - prev_m))),
        "components": comps,
    }


# ------------------------------------------------------------------
# SPY 마감 봉 (P6)
# ------------------------------------------------------------------
def eod_from_intraday(bars: pd.DataFrame, now_et: dt.datetime) -> pd.DataFrame:
    """1시간봉 프레임(Open/Close, tz-aware 인덱스) → 세션별 마지막 봉 (open, close, ret, half_day, bar_start).

    * 정규장: 마지막 봉 15:30 ET (15:30~16:00 = 마감 30분 봉). 반일장: 11:30 ET → half_day=True.
    * 오늘(now_et 기준) 세션은 16:05 ET(반일장 13:05) 이후에만 확정으로 본다 — 장중 부분 봉 금지.
    * 마지막 봉이 15:30/11:30 이 아닌 세션(데이터 결손·장중 조회)은 제외하고 attrs["warnings"] 에 남긴다.
    """
    df = bars.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if not {"Open", "Close"} <= set(df.columns):
        raise ValueError(f"eod_from_intraday: Open/Close 열 필요, 실제 {list(df.columns)}")
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")          # yfinance 인트라데이 인덱스는 UTC-aware
    df.index = idx.tz_convert(_ET)
    df = df[["Open", "Close"]].apply(pd.to_numeric, errors="coerce").dropna().sort_index()

    if now_et.tzinfo is None:
        now_et = now_et.replace(tzinfo=_ET)
    now_et = now_et.astimezone(_ET)
    today, now_t = now_et.date(), now_et.time()

    notes: list[str] = []
    rows = []
    if not df.empty:
        session_key = df.index.tz_localize(None).normalize()
        for session, grp in df.groupby(session_key):
            bar_ts = grp.index[-1]
            bs = bar_ts.strftime("%H:%M")
            d = session.date()
            if bs == FULL_DAY_LAST_BAR:
                half, complete = False, (d < today or (d == today and now_t >= SESSION_CLOSE_FULL))
            elif bs == HALF_DAY_LAST_BAR:
                half, complete = True, (d < today or (d == today and now_t >= SESSION_CLOSE_HALF))
            else:
                half, complete = False, False
                notes.append(f"EOD {d}: 마지막 봉 시각 {bs} (15:30/11:30 아님) → 세션 제외")
            if not complete:
                if d >= today and bs in (FULL_DAY_LAST_BAR, HALF_DAY_LAST_BAR):
                    notes.append(f"EOD {d}: 세션 미완성(현재 {now_et:%H:%M} ET) → 오늘 행 보류")
                continue
            o, c = float(grp["Open"].iloc[-1]), float(grp["Close"].iloc[-1])
            if o <= 0:
                notes.append(f"EOD {d}: 시가 {o} 비정상 → 제외")
                continue
            rows.append((session, o, c, c / o - 1.0, half, bs))
    out = pd.DataFrame(rows, columns=["date"] + EOD_COLUMNS).set_index("date") if rows else None
    out = _coerce_eod(out)
    out.attrs["warnings"] = notes
    return out


def fetch_spy_eod(period: str = "730d", now_et: dt.datetime | None = None) -> pd.DataFrame:
    """SPY interval="1h" 다운로드 → 세션별 마지막 봉. 반일장은 half_day=True."""
    raw = _yf_download("SPY", interval="1h", period=period)
    out = eod_from_intraday(raw, now_et or _now_et())
    out.attrs["period"] = period
    notes = out.attrs.get("warnings", [])
    if notes:
        warnings.warn("SPY EOD: " + " | ".join(notes[:10]) + (f" … 외 {len(notes)-10}건" if len(notes) > 10 else ""),
                      stacklevel=2)
    return out


# ------------------------------------------------------------------
# 병합 (append 규약)
# ------------------------------------------------------------------
def _merge_cells(existing: pd.DataFrame | None, fetched: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """셀 단위 병합: 새 값 우선, 새 응답에 없는 셀은 기존 유지 (fg/cboe)."""
    fetched = fetched.copy()
    if existing is None or existing.empty:
        out = fetched
    else:
        out = fetched.combine_first(existing)
    cols = list(columns) + [c for c in out.columns if c not in columns]
    out = out.reindex(columns=cols).astype(float)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index = _naive_daily_index(out.index)
    out.columns.name = None
    return out


def _merge_rows(existing: pd.DataFrame | None, fetched: pd.DataFrame) -> pd.DataFrame:
    """행 단위 병합: 같은 날짜는 새 행이 통째로 대체 (eod)."""
    parts = [x for x in (existing, fetched) if x is not None and not x.empty]
    if not parts:
        return _coerce_eod(None)
    out = pd.concat(parts)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return _coerce_eod(out)


# ------------------------------------------------------------------
# 가드
# ------------------------------------------------------------------
def apply_guards(bundle: Bundle) -> Bundle:
    """(a) SPY 마지막 일자 이후의 유령 행 제거 — close 의 24/7 자산(BTC-USD) 제외, spy_ohlc/cboe/fg/eod 도 동일 기준으로 절단
       (b) 마지막 일자가 SPY 보다 STALE_SESSIONS 거래일 이상 뒤처진 티커 → meta["warnings"] 기록
    멱등: 같은 번들에 두 번 적용해도 결과·경고가 늘지 않는다. 원본은 수정하지 않는다."""
    close = bundle.close.copy()
    meta = copy.deepcopy(bundle.meta) if bundle.meta else {}
    warn_list = meta.setdefault("warnings", [])
    if "SPY" not in close.columns or close["SPY"].dropna().empty:
        raise RuntimeError("apply_guards: close 에 SPY 가 없다 — 캐시가 손상됐거나 SPY 다운로드 실패")
    close.index = _naive_daily_index(close.index)
    spy_idx = close["SPY"].dropna().index
    spy_last = spy_idx[-1]

    # (a) 유령 행: SPY 마지막 일자 이후에 값이 있는 비-24/7 티커 셀 → NaN, 전부 NaN 이 된 행은 삭제
    phantom: dict[str, int] = {}
    after = np.asarray(close.index > spy_last)
    for t in close.columns:
        if t in TWENTY_FOUR_SEVEN:
            continue
        mask = after & close[t].notna().values
        n = int(mask.sum())
        if n:
            phantom[str(t)] = n
            close.loc[mask, t] = np.nan
    close = close.dropna(how="all")
    if phantom:
        _record(warn_list, f"유령 행 제거(SPY 마지막 {spy_last:%Y-%m-%d} 이후): "
                + ", ".join(f"{t}×{n}" for t, n in phantom.items()))

    dropped: dict[str, int] = {}

    def _trim(df: pd.DataFrame | None, name: str) -> pd.DataFrame | None:
        if df is None or df.empty:
            return df
        m = pd.DatetimeIndex(df.index) > spy_last
        n = int(m.sum())
        if n:
            dropped[name] = n
            return df.loc[~m]
        return df

    spy_ohlc = _trim(bundle.spy_ohlc, "spy_ohlc")
    cboe = _trim(bundle.cboe, "cboe")
    fg = _trim(bundle.fg, "fg")
    eod = _trim(bundle.eod, "eod")
    if dropped:
        _record(warn_list, f"SPY 마지막 {spy_last:%Y-%m-%d} 이후 행 절단: "
                + ", ".join(f"{k}×{n}" for k, n in dropped.items()))

    # (b) 지연 티커
    stale: dict[str, dict] = {}
    empty_cols: list[str] = []
    for t in close.columns:
        s = close[t].dropna()
        if s.empty:
            empty_cols.append(str(t))
            continue
        lag = int((spy_idx > s.index[-1]).sum())
        if lag >= STALE_SESSIONS:
            stale[str(t)] = {"last": _dstr(s.index[-1]), "lag_sessions": lag}
    if empty_cols:
        _record(warn_list, f"빈 열(전부 NaN): {empty_cols}")
    if stale:
        _record(warn_list, f"지연 티커(SPY {spy_last:%Y-%m-%d} 대비 {STALE_SESSIONS}거래일 이상 뒤처짐): "
                + ", ".join(f"{t}({v['last']}, -{v['lag_sessions']}일)" for t, v in stale.items()))

    meta["guards"] = {
        "checked_at_utc": _iso_utc(_utcnow()),
        "spy_last": _dstr(spy_last),
        "phantom_cells": phantom,
        "dropped_rows_after_spy_last": dropped,
        "stale_tickers": stale,
        "empty_columns": empty_cols,
        "twenty_four_seven": sorted(TWENTY_FOUR_SEVEN),
    }
    return Bundle(close=close, spy_ohlc=spy_ohlc, cboe=cboe, fg=fg, eod=eod, meta=meta)


# ------------------------------------------------------------------
# 저장 / 로드
# ------------------------------------------------------------------
def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    """path 옆의 `<이름>.tmp` 에 쓰고 fsync 한다. 반환: 임시 경로 (호출자가 os.replace 로 바꿔 넣는다)."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        df.to_csv(f, index_label="date", float_format=CSV_FLOAT_FORMAT, lineterminator="\n")
        f.flush()
        os.fsync(f.fileno())
    return tmp


def _read_csv(path: Path, str_cols: tuple[str, ...] = ()) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음 — scripts/build_cache.py 를 먼저 실행하라")
    dtype = {c: str for c in str_cols} or None
    df = pd.read_csv(path, index_col="date", encoding="utf-8", dtype=dtype)
    df.index = _naive_daily_index(pd.to_datetime(df.index))
    df.columns.name = None
    return df.sort_index()


def save_cache(bundle: Bundle, data_dir: Path = DATA_DIR) -> dict[str, Path]:
    """번들을 data_dir 에 저장 (CSV 5개 + meta.json). 반환: 이름 → 경로.

    원자적: 여섯 파일을 전부 `.tmp` 에 쓰고 fsync 한 뒤에야 os.replace 로 바꿔 넣는다. 직렬화 도중 예외가 나면
    임시 파일을 지우고 기존 캐시는 손대지 않는다(예외 구간이 rename 6회로 줄어든다).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {k: data_dir / v for k, v in FILES.items()}
    frames = {
        "close": bundle.close,
        "spy_ohlc": bundle.spy_ohlc,
        "cboe": bundle.cboe if bundle.cboe is not None else _empty_frame(CBOE_COLUMNS),
        "fg": bundle.fg if bundle.fg is not None else _empty_frame(FG_COLUMNS),
        "eod": _coerce_eod(bundle.eod),
    }
    meta = dict(bundle.meta or {})
    meta.setdefault("schema_version", SCHEMA_VERSION)
    staged: list[tuple[Path, Path]] = []
    try:
        for k, df in frames.items():
            staged.append((_write_csv(df, paths[k]), paths[k]))
        tmp_meta = paths["meta"].with_name(paths["meta"].name + ".tmp")
        with open(tmp_meta, "w", encoding="utf-8", newline="\n") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        staged.append((tmp_meta, paths["meta"]))
        for tmp, final in staged:          # 모든 직렬화가 끝난 뒤에만 교체
            os.replace(tmp, final)
    except BaseException:
        for tmp, _final in staged:
            tmp.unlink(missing_ok=True)
        raise
    return paths


def _load_existing(data_dir: Path) -> dict[str, pd.DataFrame | None]:
    """append 대상(fg/eod/cboe)의 기존 파일을 가드 없이 읽는다. 없으면 None (최초 구축)."""
    out: dict[str, pd.DataFrame | None] = {}
    for key, loader in (("cboe", lambda p: _read_csv(p).astype(float)),
                        ("fg", lambda p: _read_csv(p).astype(float)),
                        ("eod", lambda p: _coerce_eod(_read_csv(p, str_cols=("bar_start",))))):
        p = data_dir / FILES[key]
        out[key] = loader(p) if p.exists() else None
    return out


def load_cache(data_dir: Path = DATA_DIR, guards: bool = True) -> Bundle:
    """data_dir 의 캐시를 읽어 Bundle 로. 기본으로 apply_guards 를 적용한다(멱등).
    파일이 하나라도 없으면 FileNotFoundError."""
    data_dir = Path(data_dir)
    close = _read_csv(data_dir / FILES["close"]).astype(float)
    spy_ohlc = _read_csv(data_dir / FILES["spy_ohlc"]).astype(float)
    cboe = _read_csv(data_dir / FILES["cboe"]).astype(float)
    fg = _read_csv(data_dir / FILES["fg"]).astype(float)
    eod = _coerce_eod(_read_csv(data_dir / FILES["eod"], str_cols=("bar_start",)))
    meta_path = data_dir / FILES["meta"]
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} 없음 — scripts/build_cache.py 를 먼저 실행하라")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    # 일관성: 저장이 중간에 끊긴 캐시(close 는 새 날짜, meta 는 옛 날짜)를 조용히 쓰지 않는다.
    # meta.spy_last 가 없는 번들(테스트 픽스처·구버전)은 검사하지 않는다.
    spy_last = (_dstr(close["SPY"].dropna().index[-1])
                if "SPY" in close.columns and not close["SPY"].dropna().empty else None)
    ohlc_last = _dstr(spy_ohlc.index[-1]) if len(spy_ohlc) else None
    if meta.get("spy_last") and (meta["spy_last"] != spy_last or meta["spy_last"] != ohlc_last):
        raise RuntimeError(f"캐시 불일치: meta.spy_last={meta['spy_last']} close.SPY={spy_last} spy_ohlc={ohlc_last} — "
                           "저장이 중간에 끊겼을 가능성; scripts/build_cache.py 로 재구축하라")
    bundle = Bundle(close=close, spy_ohlc=spy_ohlc, cboe=cboe, fg=fg, eod=eod, meta=meta)
    return apply_guards(bundle) if guards else bundle


def _check_against_previous(close: pd.DataFrame, spy_ohlc: pd.DataFrame, data_dir: Path) -> None:
    """퇴행한 다운로드(부분 응답·레이트리밋)로 기존 close.csv 를 덮어쓰지 않는다 — 셋 중 하나라도 걸리면 RuntimeError.

    (a) 이전 캐시에 있던 티커(ALL_TICKERS 에 아직 있는 것)가 이번 다운로드에 없음 — config 에서 뺀 티커는 허용
    (b) 어떤 티커의 비-NaN 행 수가 이전의 MAX_ROW_SHRINK 배 미만 (전량 재다운로드는 행이 줄 이유가 없다)
    (c) SPY 마지막 일자가 이전보다 이전
    최초 구축(close.csv 없음)은 비교 대상이 없으므로 통과.
    """
    p = Path(data_dir) / FILES["close"]
    if not p.exists():
        return
    prev = _read_csv(p).astype(float)
    lost = sorted(t for t in prev.columns if t in ALL_TICKERS and t not in close.columns)
    if lost:
        raise RuntimeError(f"이전 캐시에 있던 티커가 이번 다운로드에 없음(부분 응답·레이트리밋 의심) → 저장 거부: {lost}")
    shrunk = {}
    for t in close.columns:
        if t not in prev.columns:
            continue
        n_prev, n_new = int(prev[t].notna().sum()), int(close[t].notna().sum())
        if n_new < MAX_ROW_SHRINK * n_prev:
            shrunk[str(t)] = (n_prev, n_new)
    if shrunk:
        raise RuntimeError(f"티커 행 수가 크게 줄어듦 → 저장 거부 {{ticker: (prev, new)}}: {shrunk}")
    if "SPY" in prev.columns and not prev["SPY"].dropna().empty:
        prev_spy_last = prev["SPY"].dropna().index[-1]
        if spy_ohlc.index[-1] < prev_spy_last:
            raise RuntimeError(f"SPY 마지막 일자 후퇴 {prev_spy_last:%Y-%m-%d} → {spy_ohlc.index[-1]:%Y-%m-%d} → 저장 거부")


# ------------------------------------------------------------------
# 구축 / 갱신
# ------------------------------------------------------------------
def _build(data_dir: Path, mode: str, force: bool = False) -> Bundle:
    """공통 구축 흐름. mode="build": fg 전체 이력 + eod 730d, mode="update": fg 최근 10일 + eod 60d.
    force=True 면 퇴행 검사(_check_against_previous)를 건너뛴다 — 원천이 정당하게 이력을 줄인 경우의 수동 재구축용."""
    assert mode in ("build", "update")
    data_dir = Path(data_dir)
    t0 = time.time()
    fetched_at = _utcnow()
    now_et = fetched_at.astimezone(_ET)
    warn_list: list[str] = []
    prev = _load_existing(data_dir)

    # 1) 가격: 전량 재다운로드 (증분 append 금지). 퇴행한 응답이면 다른 원천을 부르기 전에 여기서 실패한다.
    raw = fetch_prices(ALL_TICKERS, period="max")
    close, spy_ohlc, missing = split_prices(raw, ALL_TICKERS)
    if force:
        _record(warn_list, "force=True: 이전 캐시 대비 퇴행 검사를 건너뜀")
    else:
        _check_against_previous(close, spy_ohlc, data_dir)
    if missing:
        _record(warn_list, f"다운로드에 없는 티커(캐시에서 제외): {missing}")
    spy_last = spy_ohlc.index[-1]

    # 2) CBOE: 전체 CSV 를 받아 셀 단위 병합. 실패 시 기존 유지 + 경고
    try:
        new = fetch_cboe()
        for n in new.attrs.get("warnings", []):
            _record(warn_list, n)
        cboe = _merge_cells(prev["cboe"], new, CBOE_COLUMNS)
    except Exception as e:          # noqa: BLE001 — 기록 후 기존 캐시로 진행
        _record(warn_list, f"CBOE 조회 실패 → 기존 캐시 유지: {e}")
        cboe = prev["cboe"] if prev["cboe"] is not None else _empty_frame(CBOE_COLUMNS)

    # 3) CNN F&G: build 는 seed 부터 전체, update 는 마지막 캐시 일자 -10일부터
    seed = CNN_FG_SEED_DATE
    if mode == "update" and prev["fg"] is not None and not prev["fg"].empty:
        seed = (prev["fg"].index.max() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    fg_info: dict = {"seed_date": seed}
    try:
        new = fetch_fg_history(seed)
        for n in new.attrs.get("warnings", []):
            _record(warn_list, n)
        fg_info.update({"last_point_utc": new.attrs.get("last_point_utc"),
                        "last_point_is_live": new.attrs.get("last_point_is_live"),
                        "n_fetched": int(len(new))})
        fg = _merge_cells(prev["fg"], new, FG_COLUMNS)
    except Exception as e:          # noqa: BLE001
        _record(warn_list, f"CNN F&G 조회 실패 → 기존 캐시 유지: {e}")
        fg = prev["fg"] if prev["fg"] is not None else _empty_frame(FG_COLUMNS)

    # 4) SPY EOD: 세션 마지막 1h 봉, 행 단위 병합
    period = "730d" if mode == "build" else "60d"
    try:
        new = fetch_spy_eod(period, now_et=now_et)
        for n in new.attrs.get("warnings", []):
            _record(warn_list, n)
        eod = _merge_rows(prev["eod"], new)
    except Exception as e:          # noqa: BLE001
        _record(warn_list, f"SPY EOD 조회 실패 → 기존 캐시 유지: {e}")
        eod = _coerce_eod(prev["eod"])

    # 5) 점 원칙: SPY 마지막 봉이 오늘 장중 부분 봉일 가능성
    complete = not (spy_last.date() == now_et.date() and now_et.time() < SESSION_CLOSE_FULL)
    if not complete:
        _record(warn_list, f"SPY 마지막 봉 {spy_last:%Y-%m-%d} 은 장중 부분 봉일 수 있음 (현재 {now_et:%H:%M} ET)")

    import yfinance
    meta = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "fetched_at_utc": _iso_utc(fetched_at),
        "fetched_at_et": now_et.strftime("%Y-%m-%d %H:%M ET"),
        "python_version": platform.python_version(),
        "yfinance_version": getattr(yfinance, "__version__", "?"),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "price_period": "max",
        "requested_tickers": list(ALL_TICKERS),
        "missing_tickers": missing,
        "spy_last": _dstr(spy_last),
        "spy_last_bar_complete": complete,
        "fg": fg_info,
        "eod_period": period,
        "warnings": warn_list,
    }
    bundle = apply_guards(Bundle(close=close, spy_ohlc=spy_ohlc, cboe=cboe, fg=fg, eod=eod, meta=meta))
    bundle.meta["tickers"] = _ticker_stats(bundle.close)
    bundle.meta["frames"] = {k: _frame_stats(getattr(bundle, k)) for k in ("close", "spy_ohlc", "cboe", "fg", "eod")}
    bundle.meta["elapsed_sec"] = round(time.time() - t0, 1)
    save_cache(bundle, data_dir)
    return bundle


def build_cache(data_dir: Path = DATA_DIR, *, force: bool = False) -> Bundle:
    """전량 재구축(가격) + append(fg/eod/cboe) + meta.json 저장. 반환은 가드 적용 번들.
    force=True 는 이전 캐시 대비 퇴행 검사를 건너뛴다(수동 재구축 전용)."""
    return _build(data_dir, "build", force=force)


def update_daily(data_dir: Path = DATA_DIR, *, force: bool = False) -> Bundle:
    """가격 전량 재다운로드(~4초), fg/eod/cboe 는 새 날짜만 append. 반환은 가드 적용 번들.
    퇴행한 다운로드(티커 소실·행 수 급감·SPY 일자 후퇴)는 RuntimeError 로 거부한다(force=True 로만 우회)."""
    return _build(data_dir, "update", force=force)
