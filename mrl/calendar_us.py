# -*- coding: utf-8 -*-
"""NYSE 거래 캘린더와 완성 봉·기간말 판정 (ARCHITECTURE.md 「거래 캘린더」 계약 구현).

규칙 요약
* 정기 휴장 (모두 규칙 기반으로 생성):
  - 신정 1/1: 일요일이면 다음 월요일(1/2) 관측. **토요일이면 NYSE 는 전날 금요일(12/31)에 휴장하지 않는다.**
  - 마틴 루터 킹 데이: 1월 셋째 월요일, 1998년부터.
  - 대통령의 날: 2월 셋째 월요일.
  - 성금요일: 부활절(그레고리력, 익명 알고리즘) 이틀 전.
  - 현충일: 5월 마지막 월요일.
  - 준틴스 6/19: 2022년부터, 관측일 규칙.
  - 독립기념일 7/4, 성탄절 12/25: 관측일 규칙.
  - 노동절: 9월 첫째 월요일. 추수감사절: 11월 넷째 목요일.
  - 관측일 규칙: 토요일 → 전날 금요일, 일요일 → 다음 월요일.
* 특별 휴장 (SPECIAL_CLOSURES): 1994-04-27, 2001-09-11~14, 2004-06-11, 2007-01-02,
  2012-10-29~30, 2018-12-05, 2025-01-09.
* 검증 범위는 1990~2030년(SUPPORTED_YEARS). 범위 밖 연도는 정기 규칙만 적용하고
  CalendarRangeWarning 을 낸다 — 특별 휴장을 알 수 없으므로 조용히 넘기지 않는다.
* 기간 정의는 v0(reference/market_dashboard_v0.py) 의 resample 과 동일:
  주 = 'W-FRI' (토~금, 금요일 라벨), 월 = 'ME' (달력 월, 월말 라벨).
* 일봉 완성 기준: 장 마감 후 5분 여유 — now_et > bar_date 16:05 ET (SESSION_CLOSE_GRACE).
  반일장(13:00 마감)도 같은 기준을 써서 보수적으로 판정한다.
* 모든 날짜 입력은 date / datetime / pd.Timestamp / 'YYYY-MM-DD' 문자열을 받는다.
  인덱스는 tz-naive DatetimeIndex 만 허용 (계약: 모든 인덱스는 tz-naive 거래일).
"""
from __future__ import annotations

import warnings
from calendar import monthrange
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from mrl.config import ET

__all__ = [
    "SUPPORTED_YEARS", "SPECIAL_CLOSURES", "SESSION_CLOSE_GRACE", "CalendarRangeWarning",
    "easter_sunday", "nyse_holiday_names", "nyse_holidays",
    "is_trading_day", "next_trading_day", "prev_trading_day", "trading_days",
    "is_period_end", "period_end_from_index",
    "session_complete", "completed_daily", "resample_close",
]

# 규칙·특별휴장을 검증한 연도 범위 (포함)
SUPPORTED_YEARS = (1990, 2030)
# 절대 한계 — 이 밖은 그레고리력 규칙조차 의미가 없으므로 예외
_HARD_YEAR_LIMITS = (1900, 2200)
# 일봉 확정 시각 (ET): 16:00 마감 + 5분 여유. 이 시각을 *지나야* 완성으로 본다.
SESSION_CLOSE_GRACE = time(16, 5)

# 특별 휴장일 (주말·정기휴장 외의 임시 휴장). 값은 사유(리포트 표시용).
SPECIAL_CLOSURES: dict[date, str] = {
    date(1994, 4, 27): "닉슨 전 대통령 국장",
    date(2001, 9, 11): "9·11 테러",
    date(2001, 9, 12): "9·11 테러",
    date(2001, 9, 13): "9·11 테러",
    date(2001, 9, 14): "9·11 테러",
    date(2004, 6, 11): "레이건 전 대통령 국장",
    date(2007, 1, 2): "포드 전 대통령 국장",
    date(2012, 10, 29): "허리케인 샌디",
    date(2012, 10, 30): "허리케인 샌디",
    date(2018, 12, 5): "부시(41대) 전 대통령 국장",
    date(2025, 1, 9): "카터 전 대통령 국장",
}

# resample_close / is_period_end 가 받는 freq 표기 (v0 의 weekly/monthly 표기도 허용)
_FREQ_ALIASES = {
    "W": "W", "WEEKLY": "W", "W-FRI": "W",
    "M": "M", "MONTHLY": "M", "ME": "M",
    "D": "D", "DAILY": "D",
}
_RESAMPLE_RULE = {"W": "W-FRI", "M": "ME"}   # v0 와 동일한 pandas 규칙


class CalendarRangeWarning(UserWarning):
    """검증 범위(SUPPORTED_YEARS) 밖 연도의 캘린더를 요청했을 때 내는 경고."""


# ------------------------------------------------------------------
# 입력 정규화
# ------------------------------------------------------------------
def _to_date(d) -> date:
    """date-like 입력을 datetime.date 로. (Timestamp/datetime 은 date 의 서브클래스이므로 순서 주의)"""
    if d is None or d is pd.NaT:
        raise ValueError("날짜가 비어 있음(None/NaT)")
    if isinstance(d, pd.Timestamp):
        return d.date()
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, (str, np.datetime64)):
        ts = pd.Timestamp(d)
        if pd.isna(ts):
            raise ValueError(f"날짜로 해석 불가: {d!r}")
        return ts.date()
    raise TypeError(f"날짜형이 아님: {type(d).__name__}")


def _norm_freq(freq) -> str:
    """W|weekly|W-FRI → W, M|monthly|ME → M, D|daily → D. 그 밖은 ValueError."""
    key = str(freq).upper()
    if key not in _FREQ_ALIASES:
        raise ValueError(f"지원하지 않는 freq: {freq!r} (허용: W/weekly, M/monthly, D/daily)")
    return _FREQ_ALIASES[key]


def _to_et_naive(now) -> datetime:
    """datetime/Timestamp 를 ET 기준 tz-naive datetime 으로. naive 입력은 이미 ET 라고 간주."""
    if isinstance(now, pd.Timestamp):
        now = now.to_pydatetime()
    if not isinstance(now, datetime):
        raise TypeError(f"now_et 는 datetime 이어야 함: {type(now).__name__}")
    if now.tzinfo is not None:
        now = now.astimezone(ZoneInfo(ET)).replace(tzinfo=None)
    return now


def _check_index(obj, *, require_sorted: bool = False) -> pd.DatetimeIndex:
    """tz-naive DatetimeIndex 인지 검사하고 인덱스를 돌려준다 (Series/DataFrame 도 받음)."""
    idx = obj.index if isinstance(obj, (pd.Series, pd.DataFrame)) else obj
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError(f"DatetimeIndex 가 필요함: {type(idx).__name__}")
    if idx.tz is not None:
        raise ValueError("계약 위반: 인덱스는 tz-naive 여야 함 (tz_localize(None) 처리 필요)")
    if require_sorted:
        if not idx.is_monotonic_increasing:
            raise ValueError("인덱스가 오름차순이 아님")
        if not idx.is_unique:
            raise ValueError("인덱스에 중복 날짜가 있음")
    return idx


# ------------------------------------------------------------------
# 휴장일 생성
# ------------------------------------------------------------------
def easter_sunday(year: int) -> date:
    """그레고리력 부활절 (익명 그레고리 알고리즘, Meeus/Jones/Butcher)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """그 달의 n번째 weekday (월=0 … 일=6)."""
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """그 달의 마지막 weekday."""
    last = date(year, month, monthrange(year, month)[1])
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(d: date) -> date:
    """주말 휴일의 관측일: 토요일 → 전날 금요일, 일요일 → 다음 월요일."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _check_year(year: int) -> None:
    if not isinstance(year, (int, np.integer)) or isinstance(year, bool):
        raise TypeError(f"연도는 int 여야 함: {year!r}")
    if not (_HARD_YEAR_LIMITS[0] <= year <= _HARD_YEAR_LIMITS[1]):
        raise ValueError(f"연도 {year} 는 지원 한계 {_HARD_YEAR_LIMITS} 밖")
    if not (SUPPORTED_YEARS[0] <= year <= SUPPORTED_YEARS[1]):
        warnings.warn(
            f"{year}년은 캘린더 검증 범위 {SUPPORTED_YEARS} 밖 — 정기 휴장 규칙만 적용하며 "
            f"특별 휴장은 반영되지 않음",
            CalendarRangeWarning, stacklevel=4,
        )


@lru_cache(maxsize=None)
def _holidays_for_year(year: int) -> tuple[tuple[date, str], ...]:
    """한 해의 NYSE 휴장일 (평일만; 관측되지 않는 주말 휴일은 제외). 캐시되므로 범위 경고는 연도당 1회."""
    _check_year(year)
    items: dict[date, str] = {}

    ny = date(year, 1, 1)
    if ny.weekday() == 6:                       # 일요일 → 월요일 1/2 관측
        items[date(year, 1, 2)] = "신정(관측)"
    elif ny.weekday() < 5:                      # 평일
        items[ny] = "신정"
    # 토요일: NYSE 규칙상 회계연도 말인 12/31(금)에는 휴장하지 않음 → 해당 연도에 신정 휴장 없음

    if year >= 1998:
        items[_nth_weekday(year, 1, 0, 3)] = "마틴 루터 킹 데이"
    items[_nth_weekday(year, 2, 0, 3)] = "대통령의 날"
    items[easter_sunday(year) - timedelta(days=2)] = "성금요일"
    items[_last_weekday(year, 5, 0)] = "현충일"
    if year >= 2022:
        items[_observed(date(year, 6, 19))] = "준틴스"
    items[_observed(date(year, 7, 4))] = "독립기념일"
    items[_nth_weekday(year, 9, 0, 1)] = "노동절"
    items[_nth_weekday(year, 11, 3, 4)] = "추수감사절"
    items[_observed(date(year, 12, 25))] = "성탄절"

    for d, reason in SPECIAL_CLOSURES.items():
        if d.year == year:
            items[d] = f"특별휴장: {reason}"

    for d in items:
        # 생성 규칙의 자기점검: 휴장일은 반드시 평일
        if d.weekday() >= 5:
            raise AssertionError(f"휴장일 생성 오류(주말): {d} {items[d]}")
    return tuple(sorted(items.items()))


@lru_cache(maxsize=None)
def _holiday_set_for_year(year: int) -> frozenset[date]:
    return frozenset(d for d, _ in _holidays_for_year(year))


def _check_year_range(start_year: int, end_year: int) -> None:
    for y in (start_year, end_year):
        if not isinstance(y, (int, np.integer)) or isinstance(y, bool):
            raise TypeError(f"연도는 int 여야 함: {y!r}")
    if start_year > end_year:
        raise ValueError(f"start_year({start_year}) > end_year({end_year})")


def nyse_holiday_names(start_year: int, end_year: int) -> dict[date, str]:
    """start_year~end_year(포함) NYSE 휴장일 → 사유 이름. 날짜 오름차순."""
    _check_year_range(start_year, end_year)
    out: dict[date, str] = {}
    for y in range(start_year, end_year + 1):
        out.update(_holidays_for_year(y))
    return out


def nyse_holidays(start_year: int, end_year: int) -> set[date]:
    """start_year~end_year(포함) NYSE 휴장일 집합 (평일에 실제로 휴장한 날짜만)."""
    return set(nyse_holiday_names(start_year, end_year))


# ------------------------------------------------------------------
# 거래일 판정·이동
# ------------------------------------------------------------------
def is_trading_day(d) -> bool:
    """d 가 NYSE 거래일인가 (평일이고 휴장일이 아님)."""
    d = _to_date(d)
    if d.weekday() >= 5:
        return False
    return d not in _holiday_set_for_year(d.year)


def next_trading_day(d) -> date:
    """d **이후**(d 제외) 첫 거래일."""
    d = _to_date(d)
    for _ in range(30):                          # 최장 공백(9·11 주간 + 주말)도 10일 미만
        d += timedelta(days=1)
        if is_trading_day(d):
            return d
    raise RuntimeError(f"{d} 이후 30일 내 거래일을 찾지 못함 — 캘린더 규칙 오류")


def prev_trading_day(d) -> date:
    """d **이전**(d 제외) 마지막 거래일."""
    d = _to_date(d)
    for _ in range(30):
        d -= timedelta(days=1)
        if is_trading_day(d):
            return d
    raise RuntimeError(f"{d} 이전 30일 내 거래일을 찾지 못함 — 캘린더 규칙 오류")


def trading_days(start, end) -> pd.DatetimeIndex:
    """start~end(포함) 거래일의 tz-naive DatetimeIndex (name='date')."""
    s, e = _to_date(start), _to_date(end)
    if s > e:
        raise ValueError(f"start({s}) > end({e})")
    days = pd.bdate_range(s, e)
    hol = nyse_holidays(s.year, e.year)
    keep = [ts for ts in days if ts.date() not in hol]
    return pd.DatetimeIndex(keep, name="date")


# ------------------------------------------------------------------
# 기간말 판정 (주 = W-FRI 토~금, 월 = 달력 월)
# ------------------------------------------------------------------
def _week_friday(d: date) -> date:
    """d 가 속한 W-FRI 구간(토~금)의 금요일 라벨."""
    return d + timedelta(days=(4 - d.weekday()) % 7)


def _month_last_day(d: date) -> date:
    return date(d.year, d.month, monthrange(d.year, d.month)[1])


def is_period_end(d, freq) -> bool:
    """d 가 그 주(금요일 마감)/그 달의 **마지막 거래일**인가. 거래일이 아니면 False.

    다음 거래일이 다른 주(W-FRI 구간)/다른 달에 속하면 기간말이다. 예) 성금요일 전 목요일은 주말(週末).
    freq='D' 는 거래일이면 항상 True.
    """
    d = _to_date(d)
    f = _norm_freq(freq)
    if not is_trading_day(d):
        return False
    if f == "D":
        return True
    nd = next_trading_day(d)
    if f == "W":
        return nd > _week_friday(d)
    return (nd.year, nd.month) != (d.year, d.month)


def _period_complete_at(d: date, f: str) -> bool:
    """관측이 존재하는 날짜 d 로 기간이 완성되었는가.

    규칙 기반 is_period_end 에 더해, d 가 기간의 마지막 *달력일*(금요일 / 월말)이면 어떤 캘린더로도
    더 이상 관측이 올 수 없으므로 완성으로 본다 (BTC 처럼 주말·휴일에도 봉이 있는 자산 대비).
    NYSE 거래 자산에서는 두 조건이 항상 같은 결과를 낸다.
    """
    if f == "D":
        return True
    if is_period_end(d, f):
        return True
    if f == "W":
        return d.weekday() == 4
    return d == _month_last_day(d)


def _period_keys(idx: pd.DatetimeIndex, f: str) -> np.ndarray:
    """각 관측이 속한 기간의 정수 키 (W: 금요일 라벨의 ns, M: 연*12+월)."""
    if f == "W":
        fri = idx.normalize() + pd.to_timedelta((4 - idx.weekday) % 7, unit="D")
        return fri.asi8
    if f == "M":
        return (idx.year * 12 + idx.month).to_numpy()
    return idx.normalize().asi8                    # D: 날짜 자체


def _period_label(ts: pd.Timestamp, f: str) -> pd.Timestamp:
    """관측 시각 ts 가 속한 기간의 resample 라벨 (W-FRI 금요일 / ME 월말)."""
    ts = pd.Timestamp(ts).normalize()
    if f == "W":
        return ts + pd.Timedelta(days=(4 - ts.weekday()) % 7)
    if f == "M":
        return ts + pd.offsets.MonthEnd(0)
    return ts


def period_end_from_index(idx, freq=None, check_rules: bool = False):
    """인덱스(실제 관측일)를 진실로 삼아 각 관측이 기간의 마지막 관측인지 판정.

    * 중간 관측: 다음 관측이 다른 주(W-FRI)/다른 달이면 True — resample('W-FRI'/'ME').last() 가 고르는 행과 동일.
    * 마지막 관측(살아있는 날짜): 다음 관측이 없으므로 규칙 기반(is_period_end + 달력 마지막 날) 으로 판정.
    * check_rules=True 면 중간 관측에 대해 규칙 기반 판정과 교차 확인하고 불일치를 경고한다
      (SPY 처럼 NYSE 거래일에만 봉이 있는 인덱스에서만 의미 있음; 데이터 결측·미반영 휴장 탐지용).

    freq=None → 열 'W','M' 의 bool DataFrame. freq='W'|'M' → bool Series (name='period_end_W' 등).
    """
    idx = _check_index(idx, require_sorted=True)
    freqs = ("W", "M") if freq is None else (_norm_freq(freq),)
    n = len(idx)
    out: dict[str, pd.Series] = {}
    for f in freqs:
        flags = np.zeros(n, dtype=bool)
        if n:
            keys = _period_keys(idx, f)
            flags[:-1] = keys[1:] != keys[:-1]
            flags[-1] = _period_complete_at(idx[-1].date(), f)
        if check_rules and n > 1:
            rule = np.fromiter((is_period_end(d, f) for d in idx[:-1].date), dtype=bool, count=n - 1)
            mism = idx[:-1][rule != flags[:-1]]
            if len(mism):
                head = ", ".join(str(t.date()) for t in mism[:5])
                warnings.warn(
                    f"기간말({f}) 규칙 판정과 인덱스 판정 불일치 {len(mism)}건 (예: {head}) — "
                    f"데이터 결측 또는 캘린더에 없는 휴장 가능성",
                    UserWarning, stacklevel=2,
                )
        out[f] = pd.Series(flags, index=idx, name=f"period_end_{f}")
    if freq is None:
        return pd.DataFrame(out)
    return out[freqs[0]]


# ------------------------------------------------------------------
# 완성 봉
# ------------------------------------------------------------------
def session_complete(now_et, bar_date) -> bool:
    """bar_date 의 일봉이 확정됐는가: now_et > bar_date 16:05 ET.

    now_et 가 tz-aware 면 ET 로 변환, naive 면 ET 로 간주.
    """
    now = _to_et_naive(now_et)
    bd = _to_date(bar_date)
    return now > datetime.combine(bd, SESSION_CLOSE_GRACE)


def completed_daily(df, now_et=None):
    """미완성 마지막 일봉(및 now_et 이후 날짜의 유령 행) 제거. Series/DataFrame 모두 가능.

    now_et=None 이면 현재 ET 시각. session_complete 와 같은 규칙: 오늘 봉은 16:05 ET 를 지나야 남긴다.
    """
    idx = _check_index(df)
    if now_et is None:
        now_et = datetime.now(ZoneInfo(ET))
    now = _to_et_naive(now_et)
    last_ok = now.date() if now.time() > SESSION_CLOSE_GRACE else now.date() - timedelta(days=1)
    mask = np.asarray(idx.normalize() <= pd.Timestamp(last_ok), dtype=bool)
    return df.loc[mask].copy()


def resample_close(s, freq, completed_only: bool = False):
    """일간 종가 → 주간(금요일 마감)/월간 종가. v0 의 resample_close 와 동일:
    'W-FRI'/'ME' .last().dropna(). freq 가 일간('D'/'daily')이면 v0 처럼 그대로 돌려준다.

    completed_only=True 면 마지막 관측일(s.index 최대값)이 기간을 완성하지 못했을 때
    (is_period_end 가 아니고 기간의 마지막 달력일도 아닐 때) 그 부분 기간의 값을 제거한다.
    일봉의 완성 여부는 시각에 달려 있으므로 completed_daily() 로 따로 처리한다.
    """
    f = _norm_freq(freq)
    if f == "D":
        return s
    idx = _check_index(s)
    out = s.resample(_RESAMPLE_RULE[f]).last().dropna()          # v0 와 동일한 식
    if completed_only and len(out) and len(idx):
        last_obs = idx.max()
        if not _period_complete_at(last_obs.date(), f) and out.index[-1] == _period_label(last_obs, f):
            out = out.iloc[:-1]
    return out
