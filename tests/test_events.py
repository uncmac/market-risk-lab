# -*- coding: utf-8 -*-
"""mrl/events.py 테스트 — OPEX/쿼드위칭 규칙 parity, FOMC 손표, NFP 규칙·override, upcoming, 표 잔여 경고 (ARCHITECTURE_PHASE2.md §10·§15).

실행: 프로젝트 루트에서  python -m pytest tests/test_events.py -q
"""
from __future__ import annotations

import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrl import calendar_us as cal  # noqa: E402
from mrl import events as EV  # noqa: E402

D = date
YEARS = (2025, 2026, 2027)


def _iso(ds) -> list[str]:
    return [d.isoformat() for d in ds]


# ------------------------------------------------------------------
# OPEX · 쿼드위칭
# ------------------------------------------------------------------
def test_opex_rule_matches_expected_table_2025_2027():
    all_dates = []
    for y in YEARS:
        got = EV.opex_dates(y)
        assert _iso(got) == EV.OPEX_EXPECTED[y], y
        all_dates += got
    assert len(all_dates) == 36
    # 휴장으로 이동한 세 날짜 (성금요일 2025-04-18, 준틴스 2026-06-19, 2027-06-18 관측 휴장)
    for moved in ("2025-04-17", "2026-06-18", "2027-06-17"):
        assert D.fromisoformat(moved) in all_dates
        assert D.fromisoformat(moved).weekday() == 3        # 목요일로 당겨짐
    for d in all_dates:
        assert cal.is_trading_day(d)
        assert d.weekday() in (3, 4)
        assert 15 <= d.day <= 21                             # 셋째 주 안
    assert D(2026, 9, 18) in EV.opex_dates(2026)


def test_quad_witching_twelve_dates():
    quads = [d for y in YEARS for d in EV.quad_witching(y)]
    assert len(quads) == 12
    assert all(d.month in (3, 6, 9, 12) for d in quads)
    for y in YEARS:
        assert set(EV.quad_witching(y)) <= set(EV.opex_dates(y))
        assert [d.month for d in EV.quad_witching(y)] == [3, 6, 9, 12]
    assert D(2026, 12, 18) in EV.quad_witching(2026)
    assert D(2026, 9, 18) in EV.quad_witching(2026)


# ------------------------------------------------------------------
# FOMC 표
# ------------------------------------------------------------------
def test_fomc_table_24_trading_wednesdays():
    all_dates = []
    for y in YEARS:
        got = EV.fomc_dates(y)
        assert len(got) == 8 and got == sorted(got), y
        assert all(d.year == y for d in got)
        all_dates += got
    assert len(all_dates) == 24 and len(set(all_dates)) == 24
    for d in all_dates:
        assert cal.is_trading_day(d), d
        assert d.weekday() == 2, d                            # 수요일(2일 회의 둘째 날)
    assert D(2026, 9, 16) in EV.fomc_dates(2026)
    assert D(2025, 9, 17) in EV.fomc_dates(2025)
    assert EV.fomc_table_years() == [2025, 2026, 2027]
    assert 2027 in EV.FOMC_TENTATIVE_YEARS and 2026 not in EV.FOMC_TENTATIVE_YEARS


def test_fomc_missing_year_warns_and_returns_empty():
    with pytest.warns(UserWarning):
        assert EV.fomc_dates(2031) == []
    with pytest.raises(TypeError):
        EV.fomc_dates("2026")


# ------------------------------------------------------------------
# NFP 규칙 · 연방 휴일 · override
# ------------------------------------------------------------------
NFP_EXPECTED = {
    # 1월: 첫 금요일 1/3 → 1/10 ; 7/4(금) 연방 휴일 → 7/3(목)
    2025: ["2025-01-10", "2025-02-07", "2025-03-07", "2025-04-04", "2025-05-02", "2025-06-06",
           "2025-07-03", "2025-08-01", "2025-09-05", "2025-10-03", "2025-11-07", "2025-12-05"],
    # 1월: 첫 금요일 1/2 → 1/9 ; 7/3(금) 은 7/4(토) 의 관측 휴일 → 7/2(목) ; 4/3 은 성금요일이지만 연방 휴일 아님 → 발표
    2026: ["2026-01-09", "2026-02-06", "2026-03-06", "2026-04-03", "2026-05-01", "2026-06-05",
           "2026-07-02", "2026-08-07", "2026-09-04", "2026-10-02", "2026-11-06", "2026-12-04"],
    # 1월: 첫 금요일 1/1(휴일) → 1/8
    2027: ["2027-01-08", "2027-02-05", "2027-03-05", "2027-04-02", "2027-05-07", "2027-06-04",
           "2027-07-02", "2027-08-06", "2027-09-03", "2027-10-01", "2027-11-05", "2027-12-03"],
}


def test_nfp_rule_first_friday_with_january_and_holiday_exceptions():
    for y in YEARS:
        got = EV.nfp_dates(y)
        assert _iso(got) == NFP_EXPECTED[y], y
        assert len(got) == 12 and all(d.year == y and d.month == m for m, d in enumerate(got, 1))
        for d in got:
            assert d.weekday() in (3, 4)                       # 금요일 또는 (휴일 회피) 목요일
            assert not EV.is_federal_holiday(d)
    # 1월 규칙 경계: 첫 금요일이 1/4 이면 그대로 (2019, 2030), 1/3 이면 다음 주 (2020, 2025)
    assert EV.nfp_dates(2019)[0] == D(2019, 1, 4)
    assert EV.nfp_dates(2020)[0] == D(2020, 1, 10)
    assert EV.nfp_dates(2021)[0] == D(2021, 1, 8)


def test_nfp_override(monkeypatch):
    monkeypatch.setitem(EV.NFP_OVERRIDES, "2025-12", "2025-12-16")        # 2025 셧다운 지연 실제 발표일
    got = EV.nfp_dates(2025)
    assert got[11] == D(2025, 12, 16) and D(2025, 12, 5) not in got
    assert got[:11] == [D.fromisoformat(s) for s in NFP_EXPECTED[2025][:11]]


def test_federal_holiday_helper():
    assert EV.is_federal_holiday(D(2025, 7, 4))
    assert EV.is_federal_holiday(D(2026, 7, 3))              # 7/4 토 → 금 관측
    assert EV.is_federal_holiday(D(2027, 1, 1))
    assert EV.is_federal_holiday(D(2025, 10, 13))            # 콜럼버스 데이(NYSE 는 개장)
    assert EV.is_federal_holiday(D(2025, 11, 11))            # 재향군인의 날(NYSE 는 개장)
    assert not EV.is_federal_holiday(D(2026, 4, 3))          # 성금요일: NYSE 휴장이지만 연방 휴일 아님
    assert not cal.is_trading_day(D(2026, 4, 3))
    assert EV.is_federal_holiday(D(2021, 12, 31))            # 2022-01-01 토 → 12/31 금 관측
    assert not EV.is_federal_holiday(D(2026, 9, 4))


# ------------------------------------------------------------------
# upcoming
# ------------------------------------------------------------------
def test_upcoming_only_trading_days_after_asof():
    asof = D(2026, 9, 4)                                     # 금요일; 9/7 노동절 휴장
    up = EV.upcoming(asof, 20)
    assert list(up.columns) == list(EV.UPCOMING_COLUMNS)
    assert (up["date"] > pd.Timestamp(asof)).all()
    assert up["sessions_ahead"].between(1, 20).all()
    assert list(up["sessions_ahead"]) == sorted(up["sessions_ahead"])
    rows = {(r.kind, r.date.date()): int(r.sessions_ahead) for r in up.itertuples()}
    assert rows[("FOMC", D(2026, 9, 16))] == 7                # 9/8,9,10,11,14,15,16
    assert rows[("QUAD", D(2026, 9, 18))] == 9
    assert rows[("NFP", D(2026, 10, 2))] == 19
    assert ("OPEX", D(2026, 9, 18)) not in rows              # 쿼드위칭 달의 OPEX 는 QUAD 한 줄로
    assert len(up) == 3                                      # CPI 미등록, 10/16 OPEX 는 지평 밖(> 20세션)
    assert up.attrs["horizon_end"] == "2026-10-05" and up.attrs["n_sessions"] == 20
    assert up.attrs["cpi_registered"] is False and "CPI 일정 미등록" in up.attrs["warnings"]
    assert up.attrs["fomc_years_missing"] == []
    assert not up["tentative"].any()
    # asof 당일 이벤트는 제외 (이미 지남)
    up2 = EV.upcoming(D(2026, 9, 16), 20)
    assert D(2026, 9, 16) not in set(up2["date"].dt.date)
    assert D(2026, 9, 18) in set(up2["date"].dt.date)
    # 주말 asof 도 다음 거래일부터 센다
    up3 = EV.upcoming(D(2026, 9, 5), 20)
    assert rows[("FOMC", D(2026, 9, 16))] == int(up3.set_index("kind").loc["FOMC", "sessions_ahead"])
    # 짧은 지평
    up4 = EV.upcoming(asof, 5)
    assert len(up4) == 0 and list(up4.columns) == list(EV.UPCOMING_COLUMNS)
    with pytest.raises(ValueError):
        EV.upcoming(asof, 0)


def test_upcoming_release_on_market_holiday_maps_to_next_session():
    up = EV.upcoming(D(2026, 3, 25), 20)                     # NFP 2026-04-03 = 성금요일(휴장)
    nfp = up[up["kind"] == "NFP"].iloc[0]
    assert nfp["date"] == pd.Timestamp("2026-04-03")
    assert int(nfp["sessions_ahead"]) == 7                   # 3/26,27,30,31,4/1,4/2,(4/3 휴장),4/6
    assert "휴장일 발표" in nfp["label_ko"]


def test_upcoming_tentative_flag_and_cpi_table(monkeypatch):
    up = EV.upcoming(D(2027, 9, 1), 20)
    fomc = up[up["kind"] == "FOMC"]
    assert len(fomc) == 1 and fomc.iloc[0]["date"] == pd.Timestamp("2027-09-15")
    assert bool(fomc.iloc[0]["tentative"]) is True and "잠정" in fomc.iloc[0]["label_ko"]
    # CPI 표를 등록하면 표시된다
    monkeypatch.setitem(EV.CPI_RELEASE_DAYS, 2026, ["2026-09-11", "2026-10-14"])
    assert EV.cpi_dates(2026) == [D(2026, 9, 11), D(2026, 10, 14)]
    up = EV.upcoming(D(2026, 9, 4), 20)
    cpi = up[up["kind"] == "CPI"]
    assert list(cpi["date"].dt.date) == [D(2026, 9, 11)] and int(cpi.iloc[0]["sessions_ahead"]) == 4
    assert up.attrs["cpi_registered"] is True and "CPI 일정 미등록" not in up.attrs["warnings"]
    assert EV.cpi_dates(2025) == []


def test_upcoming_past_table_end_flags_missing_year():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        up = EV.upcoming(D(2027, 12, 20), 20)
    assert 2028 in up.attrs["fomc_years_missing"]
    assert any("FOMC 표" in w for w in up.attrs["warnings"])
    assert (up["kind"] != "FOMC").all()


# ------------------------------------------------------------------
# 표 잔여 경고
# ------------------------------------------------------------------
def test_table_horizon_60_day_boundary():
    last = D(2027, 12, 8)
    assert EV.table_horizon(D(2026, 9, 4)) == {
        "asof": "2026-09-04", "last_fomc": "2027-12-08", "days_left": 460, "warn": False,
        "last_year": 2027, "last_year_tentative": True, "warn_days": 60,
    }
    h60 = EV.table_horizon(last - timedelta(days=60))
    assert h60["days_left"] == 60 and h60["warn"] is False
    h59 = EV.table_horizon(last - timedelta(days=59))
    assert h59["days_left"] == 59 and h59["warn"] is True
    late = EV.table_horizon(last + timedelta(days=1))
    assert late["days_left"] == -1 and late["warn"] is True
    assert EV.table_horizon(pd.Timestamp("2026-09-04"))["days_left"] == 460
    assert EV.table_horizon("2026-09-04")["days_left"] == 460
