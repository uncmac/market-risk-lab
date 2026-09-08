# -*- coding: utf-8 -*-
"""이벤트 달력 (ARCHITECTURE_PHASE2.md §10) — 표시 전용, 적합 파라미터 0, 모델 입력 아님, 네트워크 없음.

* FOMC: 2일 회의의 둘째 날(성명 14:00 ET) 손표 2025~2027. 출처 federalreserve.gov 회의 달력·보도자료(2024-08-09, 2025-09-05).
  2027 은 잠정(직전 회의에서 확정) → ``tentative=True``. 매년 12월 손으로 갱신하고, 표 잔여가 60일 미만이면 selftest 가 경고.
* OPEX: 셋째 금요일 → calendar_us.is_trading_day 가 아니면 직전 거래일 (성금요일·준틴스 관측일 등).
* 쿼드위칭: 3·6·9·12월 OPEX.
* NFP(고용보고서, 08:30 ET): 기본 규칙(계약 §10) = 그 달 첫 금요일. 예외 (1) 1월: 첫 금요일이 1/1~1/3 이면 다음 금요일
  (BLS 관행: 2015-01-09, 2016-01-08, 2020-01-10, 2021-01-08, 2025-01-10, 2026-01-09). (2) 연방 휴일이면 직전 목요일
  (2025-07-03, 2026-07-02). (3) ``NFP_OVERRIDES["YYYY-MM"]`` 가 있으면 그 날짜(BLS 공표 예외·셧다운 지연 등).
  BLS 의 실제 편성 규칙은 "12일이 든 참조 주(일~토) 다음의 셋째 금요일" 이라, 참조 달의 12일이 일요일이고 그 달이
  30일이면(또는 2월) 첫 금요일보다 한 주 늦다 (실측: 2020-05-08, 2021-10-08, 2023-12-08, 2024-03-08; 표 범위: 2026-05-08,
  2027-10-08). 그런 달은 ``NFP_OVERRIDES`` 에 넣고, ``nfp_bls_rule_dates`` / ``nfp_rule_discrepancies`` 가 두 규칙의
  불일치를 드러낸다 — 규칙을 조용히 바꾸지 않는다. BLS 발표 일정과 대조한 해는 ``NFP_CONFIRMED_YEARS``, 그 밖의 해는
  ``upcoming`` 에서 ``tentative=True``. BLS 는 성금요일에도 발표한다(연방 휴일이 아님): 발표일이 휴장일이면 다음 세션에
  반영되는 것으로 sessions_ahead 를 센다. 2025-10~12 는 연방정부 셧다운으로 실제 발표가 지연·병합(9월분 11/20, 10월분
  별도 발표 없음, 11월분 12/16)됐으나 이 모듈은 앞으로의 일정만 표시하므로 지난 달의 실제값은 되채우지 않는다.
* CPI: 선택 표(``CPI_RELEASE_DAYS``). 비어 있으면 카드에 "CPI 일정 미등록".

계약
    opex_dates(year) / quad_witching(year) / nfp_dates(year) / fomc_dates(year) / cpi_dates(year) -> list[date]
    nfp_bls_rule_dates(year) -> list[date] ; nfp_rule_discrepancies(year) -> list[dict]   # 자기점검(첫 금요일+override vs BLS 규칙)
    upcoming(asof, n_sessions=20) -> DataFrame[date, kind, sessions_ahead, label_ko, tentative]   # asof 이후 거래일만
    table_horizon(asof) -> {last_fomc, days_left, warn: days_left < 60}
"""
from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta
from numbers import Integral

import pandas as pd

from mrl import calendar_us as cal

__all__ = [
    "FOMC_DECISION_DAYS", "FOMC_TENTATIVE_YEARS", "OPEX_EXPECTED", "CPI_RELEASE_DAYS", "NFP_OVERRIDES",
    "NFP_OVERRIDE_NOTES", "NFP_CONFIRMED_YEARS",
    "KINDS", "LABEL_KO", "UPCOMING_COLUMNS", "QUAD_MONTHS", "DEFAULT_HORIZON_SESSIONS", "TABLE_WARN_DAYS",
    "opex_dates", "quad_witching", "nfp_dates", "nfp_bls_rule_dates", "nfp_rule_discrepancies",
    "fomc_dates", "cpi_dates", "upcoming", "table_horizon", "fomc_table_years", "is_federal_holiday",
]

# 2일 회의의 둘째 날(성명 14:00 ET). 출처: federalreserve.gov 회의 달력·보도자료(2024-08-09, 2025-09-05). 매년 12월 손으로 갱신
FOMC_DECISION_DAYS: dict[int, list[str]] = {
    2025: ["2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10"],
    2026: ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"],
    2027: ["2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09", "2027-07-28", "2027-09-15", "2027-10-27", "2027-12-08"],   # 잠정(직전 회의에서 확정)
}
FOMC_TENTATIVE_YEARS: frozenset[int] = frozenset({2027})

# 규칙 산출과의 parity 표(테스트): 셋째 금요일, 휴장이면 직전 거래일
OPEX_EXPECTED: dict[int, list[str]] = {
    2025: ["2025-01-17", "2025-02-21", "2025-03-21", "2025-04-17", "2025-05-16", "2025-06-20", "2025-07-18", "2025-08-15", "2025-09-19", "2025-10-17", "2025-11-21", "2025-12-19"],
    2026: ["2026-01-16", "2026-02-20", "2026-03-20", "2026-04-17", "2026-05-15", "2026-06-18", "2026-07-17", "2026-08-21", "2026-09-18", "2026-10-16", "2026-11-20", "2026-12-18"],
    2027: ["2027-01-15", "2027-02-19", "2027-03-19", "2027-04-16", "2027-05-21", "2027-06-17", "2027-07-16", "2027-08-20", "2027-09-17", "2027-10-15", "2027-11-19", "2027-12-17"],
}   # 2025-04-17(성금요일 4/18), 2026-06-18(준틴스 6/19), 2027-06-17(준틴스 6/19 토→금 6/18 관측 휴장)

CPI_RELEASE_DAYS: dict[int, list[str]] = {}       # 선택: BLS 표를 넣으면 표시, 비어 있으면 카드에 "CPI 일정 미등록"
# "YYYY-MM" → 날짜 (BLS 예외; 기본 규칙: 그 달 첫 금요일, 연방 휴일이면 직전 목요일).
# 참조 달의 12일이 일요일이고 그 달이 30일이면 BLS 규칙(참조주 뒤 셋째 금요일)은 첫 금요일보다 한 주 늦다 — 그 달을 여기에 둔다.
NFP_OVERRIDES: dict[str, str] = {
    "2026-05": "2026-05-08",   # 4월 12일(일)·30일 달 → BLS 2026 발표 일정 5/8 (2020-05-08 과 같은 구조)
    "2027-10": "2027-10-08",   # 9월 12일(일)·30일 달 → BLS 규칙 산출 10/8 (2027 일정 미공표 → 잠정)
}
NFP_OVERRIDE_NOTES: dict[str, str] = {
    "2026-05": "BLS 2026 Employment Situation 일정(참조주 규칙; 첫 금요일 5/1 아님)",
    "2027-10": "BLS 참조주 규칙 산출 — 2027 일정 공표 전 잠정",
}
NFP_CONFIRMED_YEARS: frozenset[int] = frozenset({2025, 2026})   # 규칙+override 를 BLS 공표 일정과 대조한 해. 그 밖은 tentative

QUAD_MONTHS = (3, 6, 9, 12)
DEFAULT_HORIZON_SESSIONS = 20                     # 라벨 지평과 동일
TABLE_WARN_DAYS = 60                              # FOMC 표 잔여가 이보다 짧으면 경고
KINDS = ("FOMC", "OPEX", "QUAD", "NFP", "CPI")
_KIND_ORDER = {k: i for i, k in enumerate(KINDS)}
LABEL_KO = {
    "FOMC": "FOMC 금리 결정(성명 14:00 ET)",
    "OPEX": "월간 옵션 만기(OPEX)",
    "QUAD": "쿼드위칭(분기 옵션·선물 동시 만기, OPEX 겸)",
    "NFP": "고용보고서(NFP, 08:30 ET)",
    "CPI": "소비자물가(CPI, 08:30 ET)",
}
UPCOMING_COLUMNS = ("date", "kind", "sessions_ahead", "label_ko", "tentative")
_NONTRADING_NOTE = "휴장일 발표 → 다음 세션 반영"
_TENTATIVE_NOTE = "잠정"


# ------------------------------------------------------------------
# 날짜 헬퍼
# ------------------------------------------------------------------
def _to_date(d) -> date:
    if d is None or d is pd.NaT:
        raise ValueError("날짜가 비어 있음(None/NaT)")
    if isinstance(d, pd.Timestamp):
        return d.date()
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    ts = pd.Timestamp(d)
    if pd.isna(ts):
        raise ValueError(f"날짜로 해석 불가: {d!r}")
    return ts.date()


def _check_year(year) -> int:
    if isinstance(year, bool) or not isinstance(year, Integral):
        raise TypeError(f"연도는 int 여야 함: {year!r}")
    year = int(year)
    if not (1900 <= year <= 2200):
        raise ValueError(f"연도 범위 밖: {year}")
    return year


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """그 달의 n번째 weekday (월=0 … 금=4)."""
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    nxt = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(d: date) -> date:
    """연방 휴일 관측일: 토 → 금, 일 → 월."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def is_federal_holiday(d) -> bool:
    """미 연방 공휴일(관측일 기준)인가 — BLS 발표일 이동 판정용. NYSE 휴장일과 다르다(성금요일 ×, 콜럼버스·재향군인의 날 ○)."""
    d = _to_date(d)
    y = d.year
    fixed = [date(y, 1, 1), date(y, 7, 4), date(y, 11, 11), date(y, 12, 25)]
    if y >= 2021:
        fixed.append(date(y, 6, 19))
    hol = {_observed(x) for x in fixed}
    # 12/31(금) 이 다음 해 신정(토) 관측일인 경우
    if date(y + 1, 1, 1).weekday() == 5:
        hol.add(date(y, 12, 31))
    hol.add(_nth_weekday(y, 1, 0, 3))          # 마틴 루터 킹
    hol.add(_nth_weekday(y, 2, 0, 3))          # 대통령의 날
    hol.add(_last_weekday(y, 5, 0))            # 현충일
    hol.add(_nth_weekday(y, 9, 0, 1))          # 노동절
    hol.add(_nth_weekday(y, 10, 0, 2))         # 콜럼버스 데이
    hol.add(_nth_weekday(y, 11, 3, 4))         # 추수감사절
    return d in hol


# ------------------------------------------------------------------
# 규칙·표
# ------------------------------------------------------------------
def opex_dates(year: int) -> list[date]:
    """월간 옵션 만기: 셋째 금요일 → 휴장이면 직전 거래일 (calendar_us 규칙)."""
    year = _check_year(year)
    out = []
    for m in range(1, 13):
        d = _nth_weekday(year, m, 4, 3)
        if not cal.is_trading_day(d):
            d = cal.prev_trading_day(d)
        out.append(d)
    return out


def quad_witching(year: int) -> list[date]:
    """쿼드위칭: 3·6·9·12월 OPEX."""
    return [d for d in opex_dates(year) if d.month in QUAD_MONTHS]


def nfp_dates(year: int) -> list[date]:
    """고용보고서 발표일(월별 12개). 규칙은 모듈 docstring — 첫 금요일 · 1월 예외 · 연방 휴일 → 직전 목요일 · NFP_OVERRIDES."""
    year = _check_year(year)
    out = []
    for m in range(1, 13):
        key = f"{year:04d}-{m:02d}"
        if key in NFP_OVERRIDES:
            # 셧다운 지연처럼 다음 달로 넘어간 날짜도 허용 (해당 월 항목으로 둔다)
            out.append(_to_date(NFP_OVERRIDES[key]))
            continue
        out.append(_nfp_adjust(_nth_weekday(year, m, 4, 1), m))
    return out


def _nfp_adjust(d: date, month: int) -> date:
    """NFP 공통 예외: 1월은 1/1~1/3 이면 다음 금요일(BLS 관행), 연방 휴일·주말이면 직전 평일(보통 목요일)."""
    if month == 1 and d.day <= 3:
        d += timedelta(days=7)
    while is_federal_holiday(d) or d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def nfp_bls_rule_dates(year: int) -> list[date]:
    """자기점검용: BLS 편성 규칙 — 참조 달(전달)의 12일이 든 주(일~토)의 토요일 뒤 **셋째 금요일** + 1월·휴일 예외.
    override 를 보지 않는다. ``nfp_dates`` 와의 차이는 ``nfp_rule_discrepancies`` 가 보고한다."""
    year = _check_year(year)
    out = []
    for m in range(1, 13):
        ry, rm = (year - 1, 12) if m == 1 else (year, m - 1)
        d12 = date(ry, rm, 12)
        saturday = d12 + timedelta(days=(5 - d12.weekday()) % 7)
        out.append(_nfp_adjust(saturday + timedelta(days=6 + 14), m))
    return out


def nfp_rule_discrepancies(year: int) -> list[dict]:
    """``nfp_dates``(첫 금요일 + 예외 + override) 와 ``nfp_bls_rule_dates`` 가 다른 달의 목록
    [{month, first_friday_rule, bls_rule, override}]. 비어 있어야 정상 — selftest·테스트가 감시(조용히 넘기지 않는다)."""
    year = _check_year(year)
    a, b = nfp_dates(year), nfp_bls_rule_dates(year)
    return [{"month": f"{year:04d}-{m:02d}", "first_friday_rule": x.isoformat(), "bls_rule": y.isoformat(),
             "override": NFP_OVERRIDES.get(f"{year:04d}-{m:02d}")}
            for m, (x, y) in enumerate(zip(a, b), 1) if x != y]


def fomc_table_years() -> list[int]:
    return sorted(FOMC_DECISION_DAYS)


def fomc_dates(year: int) -> list[date]:
    """FOMC 결정일(손표). 표에 없는 해는 빈 목록 + 경고(조용히 넘기지 않는다; table_horizon 이 잔여를 감시)."""
    year = _check_year(year)
    if year not in FOMC_DECISION_DAYS:
        warnings.warn(f"FOMC 표에 {year}년이 없음 (표 범위 {fomc_table_years()[0]}~{fomc_table_years()[-1]}) — 12월 갱신 필요",
                      stacklevel=2)
        return []
    return sorted(_to_date(s) for s in FOMC_DECISION_DAYS[year])


def cpi_dates(year: int) -> list[date]:
    """CPI 발표일(선택 표). 등록되지 않은 해는 빈 목록 (카드 문구 'CPI 일정 미등록')."""
    year = _check_year(year)
    if year not in CPI_RELEASE_DAYS:
        return []
    return sorted(_to_date(s) for s in CPI_RELEASE_DAYS[year])


def _events_for_year(year: int) -> list[tuple[date, str, bool]]:
    """(날짜, 종류, 잠정) 목록. 쿼드위칭 달의 OPEX 는 QUAD 한 줄로만 낸다."""
    ev: list[tuple[date, str, bool]] = []
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        fomc = fomc_dates(year)
    ev += [(d, "FOMC", year in FOMC_TENTATIVE_YEARS) for d in fomc]
    for d in opex_dates(year):
        ev.append((d, "QUAD" if d.month in QUAD_MONTHS else "OPEX", False))
    ev += [(d, "NFP", year not in NFP_CONFIRMED_YEARS) for d in nfp_dates(year)]
    ev += [(d, "CPI", False) for d in cpi_dates(year)]
    return ev


# ------------------------------------------------------------------
# 조회
# ------------------------------------------------------------------
def _empty_upcoming() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.Series(dtype="datetime64[ns]"), "kind": pd.Series(dtype=str),
        "sessions_ahead": pd.Series(dtype=int), "label_ko": pd.Series(dtype=str), "tentative": pd.Series(dtype=bool),
    })[list(UPCOMING_COLUMNS)]


def upcoming(asof, n_sessions: int = DEFAULT_HORIZON_SESSIONS) -> pd.DataFrame:
    """asof **이후** n_sessions 거래일 안의 이벤트. 열: date, kind, sessions_ahead, label_ko, tentative.

    sessions_ahead = asof 다음 거래일이 1. 발표일이 휴장일(예: 성금요일 NFP)이면 다음 세션 기준으로 세고 라벨에 표시.
    tentative: FOMC 잠정 연도(FOMC_TENTATIVE_YEARS) 또는 BLS 일정과 대조하지 않은 해의 NFP(NFP_CONFIRMED_YEARS 밖).
    asof 당일 이벤트는 포함하지 않는다(이미 지난 것). .attrs: asof, n_sessions, horizon_end, cpi_registered, fomc_years_missing, warnings.
    """
    if isinstance(n_sessions, bool) or not isinstance(n_sessions, int) or n_sessions < 1:
        raise ValueError(f"n_sessions 는 1 이상의 정수여야 함: {n_sessions!r}")
    a = _to_date(asof)
    sessions: list[date] = []
    d = a
    for _ in range(n_sessions):
        d = cal.next_trading_day(d)
        sessions.append(d)
    pos = {s: i + 1 for i, s in enumerate(sessions)}
    years = sorted({sessions[0].year, sessions[-1].year})
    warn_list: list[str] = []
    missing_years = [y for y in years if y not in FOMC_DECISION_DAYS]
    if missing_years:
        warn_list.append(f"FOMC 표에 없는 해: {missing_years} — 표 갱신 필요")
    cpi_registered = any(y in CPI_RELEASE_DAYS for y in years)
    if not cpi_registered:
        warn_list.append("CPI 일정 미등록")

    rows = []
    for y in years:
        for ev_date, kind, tentative in _events_for_year(y):
            if ev_date <= a:
                continue
            eff = ev_date if cal.is_trading_day(ev_date) else cal.next_trading_day(ev_date)
            if eff not in pos:
                continue
            notes = []
            if eff != ev_date:
                notes.append(_NONTRADING_NOTE)
            if tentative:
                notes.append(_TENTATIVE_NOTE)
            label = LABEL_KO[kind] + (f" [{' · '.join(notes)}]" if notes else "")
            rows.append({"date": pd.Timestamp(ev_date), "kind": kind, "sessions_ahead": int(pos[eff]),
                         "label_ko": label, "tentative": bool(tentative)})
    if rows:
        out = pd.DataFrame(rows, columns=list(UPCOMING_COLUMNS))
        out = out.sort_values(["sessions_ahead", "kind"], key=lambda s: s.map(_KIND_ORDER) if s.name == "kind" else s,
                              kind="mergesort").reset_index(drop=True)
        out["sessions_ahead"] = out["sessions_ahead"].astype(int)
        out["tentative"] = out["tentative"].astype(bool)
    else:
        out = _empty_upcoming()
    out.attrs = {"asof": a.isoformat(), "n_sessions": int(n_sessions), "horizon_end": sessions[-1].isoformat(),
                 "cpi_registered": bool(cpi_registered), "fomc_years_missing": missing_years, "warnings": warn_list}
    return out


def table_horizon(asof) -> dict:
    """FOMC 손표의 잔여: {last_fomc, days_left(달력일), warn(days_left < 60), last_year, last_year_tentative, asof}.
    selftest 가 warn 을 경고하고 카드에 표시한다."""
    a = _to_date(asof)
    if not FOMC_DECISION_DAYS:
        return {"asof": a.isoformat(), "last_fomc": None, "days_left": None, "warn": True,
                "last_year": None, "last_year_tentative": None, "warn_days": TABLE_WARN_DAYS}
    last = max(_to_date(s) for v in FOMC_DECISION_DAYS.values() for s in v)
    days_left = (last - a).days
    return {
        "asof": a.isoformat(),
        "last_fomc": last.isoformat(),
        "days_left": int(days_left),
        "warn": bool(days_left < TABLE_WARN_DAYS),
        "last_year": last.year,
        "last_year_tentative": bool(last.year in FOMC_TENTATIVE_YEARS),
        "warn_days": TABLE_WARN_DAYS,
    }
