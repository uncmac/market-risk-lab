# -*- coding: utf-8 -*-
"""mrl/calendar_us.py 테스트 — 알려진 휴장일, 기간말 판정, 완성 봉 제거, v0 resample_close 와의 동일성."""
from __future__ import annotations

import importlib.util
import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrl import calendar_us as cal  # noqa: E402
from mrl.config import ET  # noqa: E402

D = date


@pytest.fixture(scope="module")
def ref():
    """reference/market_dashboard_v0.py 를 파일 경로로 import (최상위 실행 코드는 __main__ 가드 안에 있음)."""
    path = ROOT / "reference" / "market_dashboard_v0.py"
    spec = importlib.util.spec_from_file_location("market_dashboard_v0", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------
# 부활절·정기 휴장
# ------------------------------------------------------------------
EASTER = {
    1990: D(1990, 4, 15), 1991: D(1991, 3, 31), 1992: D(1992, 4, 19), 1993: D(1993, 4, 11),
    1994: D(1994, 4, 3), 1995: D(1995, 4, 16), 1996: D(1996, 4, 7), 1997: D(1997, 3, 30),
    1998: D(1998, 4, 12), 1999: D(1999, 4, 4), 2000: D(2000, 4, 23), 2001: D(2001, 4, 15),
    2002: D(2002, 3, 31), 2003: D(2003, 4, 20), 2004: D(2004, 4, 11), 2005: D(2005, 3, 27),
    2006: D(2006, 4, 16), 2007: D(2007, 4, 8), 2008: D(2008, 3, 23), 2009: D(2009, 4, 12),
    2010: D(2010, 4, 4), 2011: D(2011, 4, 24), 2012: D(2012, 4, 8), 2013: D(2013, 3, 31),
    2014: D(2014, 4, 20), 2015: D(2015, 4, 5), 2016: D(2016, 3, 27), 2017: D(2017, 4, 16),
    2018: D(2018, 4, 1), 2019: D(2019, 4, 21), 2020: D(2020, 4, 12), 2021: D(2021, 4, 4),
    2022: D(2022, 4, 17), 2023: D(2023, 4, 9), 2024: D(2024, 3, 31), 2025: D(2025, 4, 20),
    2026: D(2026, 4, 5), 2027: D(2027, 3, 28), 2028: D(2028, 4, 16), 2029: D(2029, 4, 1),
    2030: D(2030, 4, 21),
}


def test_easter_table_1990_2030():
    for y, e in EASTER.items():
        assert cal.easter_sunday(y) == e, y
        assert e.weekday() == 6
        # 성금요일은 매년 휴장
        assert (e - timedelta(days=2)) in cal.nyse_holidays(y, y)


# NYSE 공식 휴장일 목록과 대조 (특별휴장 포함)
PUBLISHED = {
    1994: ["02-21", "04-01", "04-27", "05-30", "07-04", "09-05", "11-24", "12-26"],       # 신정(토) 미관측, 성탄절(일)→월
    1997: ["01-01", "02-17", "03-28", "05-26", "07-04", "09-01", "11-27", "12-25"],       # MLK 이전
    2001: ["01-01", "01-15", "02-19", "04-13", "05-28", "07-04", "09-03",
           "09-11", "09-12", "09-13", "09-14", "11-22", "12-25"],
    2022: ["01-17", "02-21", "04-15", "05-30", "06-20", "07-04", "09-05", "11-24", "12-26"],
    2023: ["01-02", "01-16", "02-20", "04-07", "05-29", "06-19", "07-04", "09-04", "11-23", "12-25"],
    2024: ["01-01", "01-15", "02-19", "03-29", "05-27", "06-19", "07-04", "09-02", "11-28", "12-25"],
    2025: ["01-01", "01-09", "01-20", "02-17", "04-18", "05-26", "06-19", "07-04", "09-01", "11-27", "12-25"],
    2026: ["01-01", "01-19", "02-16", "04-03", "05-25", "06-19", "07-03", "09-07", "11-26", "12-25"],
    2027: ["01-01", "01-18", "02-15", "03-26", "05-31", "06-18", "07-05", "09-06", "11-25", "12-24"],
}


@pytest.mark.parametrize("year", sorted(PUBLISHED))
def test_published_holiday_lists(year):
    got = sorted(f"{d:%m-%d}" for d in cal.nyse_holidays(year, year))
    assert got == PUBLISHED[year]


def test_known_holidays_from_assignment():
    assert not cal.is_trading_day(D(2026, 4, 3))       # 성금요일
    assert not cal.is_trading_day(D(2026, 7, 3))       # 독립기념일(토) → 금요일 관측
    assert not cal.is_trading_day(D(2027, 12, 24))     # 성탄절(토) → 금요일 관측
    assert not cal.is_trading_day(D(2022, 6, 20))      # 준틴스(일) → 월요일 관측
    assert cal.is_trading_day(D(2021, 6, 18))          # 2021 년 준틴스는 NYSE 미관측
    assert cal.is_trading_day(D(2021, 6, 21))
    assert cal.is_trading_day(D(2020, 6, 19))


def test_new_years_saturday_not_moved_to_friday():
    # 1/1 이 토요일인 해: 전날 12/31(금)에 휴장하지 않고, 다음 월요일도 정상 거래
    for y in (1994, 2000, 2005, 2011, 2022, 2028):
        assert D(y, 1, 1).weekday() == 5
        assert cal.is_trading_day(D(y - 1, 12, 31))
        assert cal.is_trading_day(D(y, 1, 3))
    # 1/1 이 일요일인 해: 월요일 1/2 휴장
    for y in (1995, 2006, 2012, 2017, 2023):
        assert D(y, 1, 1).weekday() == 6
        assert not cal.is_trading_day(D(y, 1, 2))


def test_weekend_observed_rules():
    assert not cal.is_trading_day(D(2021, 12, 24))     # 성탄절(토) → 금
    assert not cal.is_trading_day(D(2022, 12, 26))     # 성탄절(일) → 월
    assert not cal.is_trading_day(D(2020, 7, 3))       # 독립기념일(토) → 금
    assert not cal.is_trading_day(D(2021, 7, 5))       # 독립기념일(일) → 월
    assert not cal.is_trading_day(D(2027, 6, 18))      # 준틴스(토) → 금
    assert cal.is_trading_day(D(2021, 7, 2))
    assert cal.is_trading_day(D(2022, 12, 23))


def test_mlk_starts_1998():
    assert not cal.is_trading_day(D(1998, 1, 19))
    assert cal.is_trading_day(D(1997, 1, 20))
    assert cal.is_trading_day(D(1990, 1, 15))


def test_special_closures_present():
    hol = cal.nyse_holidays(1990, 2030)
    for d in cal.SPECIAL_CLOSURES:
        assert d in hol, d
    assert D(1994, 4, 27) in hol
    for d in (D(2001, 9, 11), D(2001, 9, 14), D(2012, 10, 29), D(2012, 10, 30),
              D(2004, 6, 11), D(2007, 1, 2), D(2018, 12, 5), D(2025, 1, 9)):
        assert not cal.is_trading_day(d)
    assert cal.is_trading_day(D(2001, 9, 17))
    assert cal.is_trading_day(D(2012, 10, 31))
    names = cal.nyse_holiday_names(2025, 2025)
    assert names[D(2025, 1, 9)].startswith("특별휴장")


def test_holidays_are_weekdays_and_no_warnings_in_supported_range():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hol = cal.nyse_holidays(*cal.SUPPORTED_YEARS)
    assert all(d.weekday() < 5 for d in hol)
    assert len(hol) > 41 * 8


def test_trading_day_counts():
    assert len(cal.trading_days("2024-01-01", "2024-12-31")) == 252
    assert len(cal.trading_days("2025-01-01", "2025-12-31")) == 250
    assert len(cal.trading_days("2026-01-01", "2026-12-31")) == 251
    td = cal.trading_days("2026-04-01", "2026-04-07")
    assert list(td.strftime("%Y-%m-%d")) == ["2026-04-01", "2026-04-02", "2026-04-06", "2026-04-07"]
    assert td.tz is None and td.name == "date"


def test_out_of_range_year_warns_and_hard_limit_raises():
    cal._holidays_for_year.cache_clear()
    cal._holiday_set_for_year.cache_clear()
    with pytest.warns(cal.CalendarRangeWarning):
        assert cal.is_trading_day(D(1985, 1, 2))
    with pytest.raises(ValueError):
        cal.nyse_holidays(1800, 1800)
    with pytest.raises(ValueError):
        cal.nyse_holidays(2001, 2000)


# ------------------------------------------------------------------
# 거래일 이동·입력 형식
# ------------------------------------------------------------------
def test_next_prev_trading_day():
    assert cal.next_trading_day(D(2026, 4, 2)) == D(2026, 4, 6)       # 성금요일 + 주말 건너뜀
    assert cal.prev_trading_day(D(2026, 4, 6)) == D(2026, 4, 2)
    assert cal.next_trading_day(D(2001, 9, 10)) == D(2001, 9, 17)
    assert cal.next_trading_day(D(2025, 1, 8)) == D(2025, 1, 10)
    assert cal.next_trading_day(D(2026, 12, 31)) == D(2027, 1, 4)
    assert cal.next_trading_day(D(2026, 9, 4)) == D(2026, 9, 8)       # 노동절
    # 입력 형식: Timestamp / datetime / 문자열 모두 date 로 돌려준다
    assert cal.next_trading_day(pd.Timestamp("2026-04-02")) == D(2026, 4, 6)
    assert cal.next_trading_day(datetime(2026, 4, 2, 15, 30)) == D(2026, 4, 6)
    assert cal.next_trading_day("2026-04-02") == D(2026, 4, 6)
    assert isinstance(cal.next_trading_day("2026-04-02"), date)
    with pytest.raises(TypeError):
        cal.is_trading_day(20260402)
    with pytest.raises(ValueError):
        cal.is_trading_day(pd.NaT)


# ------------------------------------------------------------------
# 기간말 판정
# ------------------------------------------------------------------
def test_is_period_end_weekly_around_holidays():
    assert cal.is_period_end(D(2026, 4, 2), "W")          # 성금요일 전 목요일
    assert not cal.is_period_end(D(2026, 4, 3), "W")      # 휴장일 자체는 기간말이 아님
    assert not cal.is_period_end(D(2026, 4, 1), "W")
    assert cal.is_period_end(D(2026, 4, 10), "W")
    assert not cal.is_period_end(D(2026, 4, 11), "W")     # 토요일
    assert not cal.is_period_end(D(2026, 11, 25), "W")    # 추수감사절 전 수요일 — 금요일(반일장) 거래 있음
    assert cal.is_period_end(D(2026, 11, 27), "W")
    assert cal.is_period_end(D(2026, 12, 24), "W")        # 성탄절(금) 전 목요일
    assert cal.is_period_end(D(2026, 12, 31), "W")        # 1/1(금) 휴장
    assert cal.is_period_end(D(2001, 9, 10), "W")         # 9·11 주간: 월요일이 그 주의 마지막 거래일
    assert cal.is_period_end(D(2025, 1, 8), "W") is False  # 1/9 휴장이지만 1/10(금) 거래
    # v0 표기도 허용
    assert cal.is_period_end(D(2026, 4, 2), "weekly")


def test_is_period_end_monthly():
    assert cal.is_period_end(D(2026, 4, 30), "M")
    assert cal.is_period_end(D(2026, 5, 29), "M")         # 5/30·31 주말
    assert cal.is_period_end(D(2026, 1, 30), "M")         # 1/31 토
    assert not cal.is_period_end(D(2026, 1, 29), "M")
    assert cal.is_period_end(D(2026, 12, 31), "M")
    assert cal.is_period_end(D(2027, 12, 31), "M")        # 2028-01-01 토 → 12/31 정상 거래
    assert cal.is_period_end(D(2021, 12, 31), "M") and cal.is_period_end(D(2021, 12, 31), "W")
    assert not cal.is_period_end(D(2026, 5, 30), "M")     # 토요일
    assert cal.is_period_end(D(2026, 4, 30), "monthly")
    with pytest.raises(ValueError):
        cal.is_period_end(D(2026, 4, 30), "Q")


def test_period_end_from_index_matches_resample_rows():
    idx = cal.trading_days("2026-03-02", "2026-04-15")
    pe = cal.period_end_from_index(idx)
    assert list(pe.columns) == ["W", "M"] and pe.index.equals(idx)
    assert pe.loc["2026-03-27", "W"] and pe.loc["2026-04-02", "W"] and not pe.loc["2026-04-01", "W"]
    assert pe.loc["2026-03-31", "M"] and not pe.loc["2026-03-30", "M"]
    # 마지막 관측(2026-04-15 수)은 규칙 기반: 주·월 모두 미완성
    assert not pe.iloc[-1]["W"] and not pe.iloc[-1]["M"]
    # Series 반환 형태
    w = cal.period_end_from_index(idx, "W")
    assert isinstance(w, pd.Series) and w.name == "period_end_W" and w.equals(pe["W"])
    # 플래그가 고르는 행 = resample('W-FRI').last() 가 고르는 행 (금요일로 끝나는 구간)
    idx2 = cal.trading_days("2026-01-02", "2026-04-10")
    s = pd.Series(np.arange(len(idx2), dtype=float), index=idx2)
    flags = cal.period_end_from_index(idx2, "W")
    assert flags.iloc[-1]                                  # 4/10 금요일 → 완성
    rs = cal.resample_close(s, "W", False)
    assert np.array_equal(s[flags].to_numpy(), rs.to_numpy())
    assert [cal._period_label(t, "W") for t in s[flags].index] == list(rs.index)
    # 월간: 4/10 은 미완성 월 → 플래그는 4월을 제외하고, completed_only=True 결과와 정확히 일치
    m_flags = cal.period_end_from_index(idx2, "M")
    assert not m_flags.iloc[-1]
    rm_completed = cal.resample_close(s, "M", True)
    rm_full = cal.resample_close(s, "M", False)
    assert np.array_equal(s[m_flags].to_numpy(), rm_completed.to_numpy())
    assert [cal._period_label(t, "M") for t in s[m_flags].index] == list(rm_completed.index)
    assert len(rm_full) == len(rm_completed) + 1 and rm_full.index[-1] == pd.Timestamp("2026-04-30")


def test_period_end_from_index_24_7_index_and_last_calendar_day():
    # BTC 처럼 주말·휴일에도 관측이 있으면 인덱스가 진실: 성금요일(4/3) 이 그 주의 마지막 관측
    idx = pd.date_range("2026-03-28", "2026-04-05", freq="D")
    w = cal.period_end_from_index(idx, "W")
    assert w.loc["2026-04-03"] and not w.loc["2026-04-02"]
    assert not w.iloc[-1]                                  # 4/5(일) — 다음 주 부분 구간
    # 마지막 관측이 월의 마지막 달력일(일요일)이면 완성
    idx_m = pd.date_range("2026-05-01", "2026-05-31", freq="D")
    assert cal.period_end_from_index(idx_m, "M").iloc[-1]
    assert not cal.period_end_from_index(idx_m, "W").iloc[-1]     # 5/31(일) 은 주 기준으로는 부분 구간


def test_period_end_from_index_validation_and_check_rules():
    with pytest.raises(ValueError):
        cal.period_end_from_index(pd.DatetimeIndex(["2026-04-02", "2026-04-01"]))
    with pytest.raises(ValueError):
        cal.period_end_from_index(pd.DatetimeIndex(["2026-04-01", "2026-04-01"]))
    with pytest.raises(TypeError):
        cal.period_end_from_index(pd.Index([1, 2, 3]))
    with pytest.raises(ValueError):
        cal.period_end_from_index(pd.date_range("2026-04-01", periods=3, tz="UTC"))
    full = cal.trading_days("2026-03-02", "2026-04-15")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cal.period_end_from_index(full, check_rules=True)  # 결측 없음 → 경고 없음
    gapped = full.drop(pd.Timestamp("2026-03-27"))         # 실제 거래일(금) 결측 → 목요일이 인덱스상 주말이 됨
    with pytest.warns(UserWarning, match="불일치"):
        cal.period_end_from_index(gapped, "W", check_rules=True)
    assert len(cal.period_end_from_index(pd.DatetimeIndex([]))) == 0


# ------------------------------------------------------------------
# 완성 봉
# ------------------------------------------------------------------
def test_session_complete():
    bar = D(2026, 9, 4)
    assert not cal.session_complete(datetime(2026, 9, 4, 16, 4), bar)
    assert not cal.session_complete(datetime(2026, 9, 4, 16, 5), bar)       # 정확히 16:05 는 아직
    assert cal.session_complete(datetime(2026, 9, 4, 16, 5, 1), bar)
    assert cal.session_complete(datetime(2026, 9, 4, 16, 6), bar)
    assert cal.session_complete(datetime(2026, 9, 5, 9, 0), bar)
    assert not cal.session_complete(datetime(2026, 9, 3, 23, 59), bar)
    # tz-aware: 2026-09-04 20:06 UTC = 16:06 EDT
    assert cal.session_complete(datetime(2026, 9, 4, 20, 6, tzinfo=ZoneInfo("UTC")), bar)
    assert not cal.session_complete(datetime(2026, 9, 4, 20, 4, tzinfo=ZoneInfo("UTC")), bar)
    assert cal.session_complete(pd.Timestamp("2026-09-04 16:06", tz=ET), pd.Timestamp("2026-09-04"))
    assert cal.session_complete(pd.Timestamp("2026-09-04 16:06"), "2026-09-04")
    with pytest.raises(TypeError):
        cal.session_complete("2026-09-04 16:06", bar)


def test_completed_daily_drops_incomplete_and_ghost_rows():
    idx = cal.trading_days("2026-08-24", "2026-09-04")
    df = pd.DataFrame({"Close": np.arange(len(idx), dtype=float)}, index=idx)
    out = cal.completed_daily(df, datetime(2026, 9, 4, 15, 0))
    assert out.index[-1] == pd.Timestamp("2026-09-03") and len(out) == len(df) - 1
    out = cal.completed_daily(df, datetime(2026, 9, 4, 16, 10))
    assert out.index[-1] == pd.Timestamp("2026-09-04") and len(out) == len(df)
    assert cal.completed_daily(df, datetime(2026, 9, 6, 12, 0)).equals(df)
    # tz-aware now (UTC) 도 ET 로 변환
    assert len(cal.completed_daily(df, datetime(2026, 9, 4, 20, 0, tzinfo=ZoneInfo("UTC")))) == len(df) - 1
    # 유령 행(now 이후 날짜) 제거, Series 입력 지원
    ghost = pd.concat([df, pd.DataFrame({"Close": [99.0]}, index=pd.DatetimeIndex([pd.Timestamp("2026-09-08")]))])
    out = cal.completed_daily(ghost, datetime(2026, 9, 7, 18, 0))
    assert out.index[-1] == pd.Timestamp("2026-09-04")
    s_out = cal.completed_daily(df["Close"], datetime(2026, 9, 4, 15, 0))
    assert isinstance(s_out, pd.Series) and len(s_out) == len(df) - 1
    # now_et=None 은 현재 시각 사용 — 예외 없이 길이만 줄거나 같음
    assert len(cal.completed_daily(df, None)) <= len(df)
    # 반환값은 복사본 (원본 불변)
    out = cal.completed_daily(df, datetime(2026, 9, 4, 15, 0))
    out.iloc[0, 0] = -1.0
    assert df.iloc[0, 0] == 0.0
    with pytest.raises(ValueError):
        cal.completed_daily(df.tz_localize("UTC"), datetime(2026, 9, 4, 15, 0))
    with pytest.raises(TypeError):
        cal.completed_daily(df.reset_index(), datetime(2026, 9, 4, 15, 0))


# ------------------------------------------------------------------
# resample_close
# ------------------------------------------------------------------
def _walk(idx, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx, name="Close")


def test_resample_close_equals_reference_when_not_completed_only(ref):
    idx = cal.trading_days("2023-01-03", "2026-09-04")
    s = _walk(idx)
    s.iloc[[5, 6, 100, len(s) - 1]] = np.nan                      # 결측이 있어도 동일
    for mine, theirs in (("W", "weekly"), ("M", "monthly"), ("D", "daily")):
        pd.testing.assert_series_equal(cal.resample_close(s, mine, False), ref.resample_close(s, theirs))
    pd.testing.assert_series_equal(cal.resample_close(s, "weekly", completed_only=False), ref.resample_close(s, "weekly"))
    # BTC 처럼 주말 포함 인덱스
    b = _walk(pd.date_range("2024-01-01", "2026-09-06", freq="D"), seed=1)
    pd.testing.assert_series_equal(cal.resample_close(b, "W", False), ref.resample_close(b, "weekly"))
    pd.testing.assert_series_equal(cal.resample_close(b, "M", False), ref.resample_close(b, "monthly"))
    # DataFrame 입력
    df = pd.DataFrame({"a": _walk(idx, 2), "b": _walk(idx, 3)})
    pd.testing.assert_frame_equal(cal.resample_close(df, "M", False), ref.resample_close(df, "monthly"))
    # 일간은 그대로 (v0 동일)
    assert cal.resample_close(s, "D", True) is s


def test_resample_close_completed_only_drops_partial_period():
    # 수요일에 끝나는 시계열: 마지막 주(9/4 금 마감)는 부분 → 제거
    s = _walk(cal.trading_days("2026-06-01", "2026-09-02"))
    full = cal.resample_close(s, "W", False)
    comp = cal.resample_close(s, "W", True)
    assert full.index[-1] == pd.Timestamp("2026-09-04")
    assert comp.index[-1] == pd.Timestamp("2026-08-28")
    pd.testing.assert_series_equal(comp, full.iloc[:-1])
    # 금요일에 끝나면 동일
    s2 = _walk(cal.trading_days("2026-06-01", "2026-09-04"))
    pd.testing.assert_series_equal(cal.resample_close(s2, "W", True), cal.resample_close(s2, "W", False))
    # 월간: 8/28(금)에 끝나면 8월은 부분(8/31 월요일 거래) → 제거, 8/31 에 끝나면 유지
    s3 = _walk(cal.trading_days("2026-03-02", "2026-08-28"))
    assert cal.resample_close(s3, "M", True).index[-1] == pd.Timestamp("2026-07-31")
    assert cal.resample_close(s3, "M", False).index[-1] == pd.Timestamp("2026-08-31")
    s4 = _walk(cal.trading_days("2026-03-02", "2026-08-31"))
    assert cal.resample_close(s4, "M", True).index[-1] == pd.Timestamp("2026-08-31")
    # 5/29(금)에 끝나면 5월 완성 (5/30·31 주말)
    s5 = _walk(cal.trading_days("2026-03-02", "2026-05-29"))
    assert cal.resample_close(s5, "M", True).index[-1] == pd.Timestamp("2026-05-31")


def test_resample_close_completed_only_around_good_friday():
    # 2026-04-02(목)에 끝나는 주: 4/3 성금요일 휴장 → 주 완성으로 유지
    s = _walk(cal.trading_days("2026-01-02", "2026-04-02"))
    comp = cal.resample_close(s, "W", True)
    assert comp.index[-1] == pd.Timestamp("2026-04-03") and comp.iloc[-1] == s.iloc[-1]
    # 4/1(수)에 끝나면 부분 주 → 제거
    s2 = _walk(cal.trading_days("2026-01-02", "2026-04-01"))
    assert cal.resample_close(s2, "W", True).index[-1] == pd.Timestamp("2026-03-27")
    # 9·11 주간: 2001-09-10(월) 이 그 주의 마지막 거래일
    s3 = _walk(cal.trading_days("2001-06-01", "2001-09-10"))
    assert cal.resample_close(s3, "W", True).index[-1] == pd.Timestamp("2001-09-14")


def test_resample_close_completed_only_when_last_bin_already_dropped_by_nan():
    # 마지막 주가 전부 NaN 이면 dropna 가 이미 제거 → 그 앞의 완성된 주를 잘못 지우면 안 됨
    s = _walk(cal.trading_days("2026-06-01", "2026-09-02"))
    s.loc["2026-08-31":] = np.nan
    comp = cal.resample_close(s, "W", True)
    assert comp.index[-1] == pd.Timestamp("2026-08-28")
    # 24/7 인덱스: 토요일에 끝나면 부분 주 제거, 금요일(휴장일이어도)에 끝나면 유지
    b = _walk(pd.date_range("2026-01-01", "2026-04-04", freq="D"))
    assert cal.resample_close(b, "W", True).index[-1] == pd.Timestamp("2026-04-03")
    b2 = _walk(pd.date_range("2026-01-01", "2026-04-03", freq="D"))
    assert cal.resample_close(b2, "W", True).index[-1] == pd.Timestamp("2026-04-03")
    assert cal.resample_close(b2, "W", True).iloc[-1] == b2.iloc[-1]


def test_resample_close_rejects_bad_input():
    s = _walk(cal.trading_days("2026-06-01", "2026-09-02"))
    with pytest.raises(ValueError):
        cal.resample_close(s, "Q", False)
    with pytest.raises(ValueError):
        cal.resample_close(s.tz_localize("UTC"), "W", False)
    with pytest.raises(TypeError):
        cal.resample_close(s.reset_index(drop=True), "W", False)
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    assert len(cal.resample_close(empty, "W", True)) == 0
