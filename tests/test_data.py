# -*- coding: utf-8 -*-
"""mrl.data 단위 테스트 — 네트워크 없이 합성 프레임으로 가드·파서·CSV 왕복·구축 흐름을 검사한다."""
from __future__ import annotations

import datetime as dt
import json
import sys
import warnings
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrl import data as D                       # noqa: E402
from mrl.config import TWENTY_FOUR_SEVEN        # noqa: E402

ET = ZoneInfo("America/New_York")


# ------------------------------------------------------------------
# 합성 데이터
# ------------------------------------------------------------------
def _bdays(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end, name="date")


def make_bundle(with_phantoms: bool = True) -> D.Bundle:
    """SPY 는 2024-01-01~2024-03-29(평일), BTC 는 주말 포함 + SPY 이후 3일, ^VIX/AAPL 은 SPY 이후 유령 행,
    LAGGY 는 SPY 보다 5거래일 일찍 끝난다."""
    spy_days = _bdays("2024-01-01", "2024-03-29")
    spy_last = spy_days[-1]
    rng = np.random.default_rng(0)
    close = pd.DataFrame(index=spy_days)
    close["SPY"] = 400 + np.cumsum(rng.normal(0, 1, len(spy_days)))
    close["^VIX"] = 15 + rng.normal(0, 1, len(spy_days))
    close["AAPL"] = 180 + np.cumsum(rng.normal(0, 1, len(spy_days)))
    close["LAGGY"] = 50 + np.cumsum(rng.normal(0, 1, len(spy_days)))
    close.loc[spy_days[-5:], "LAGGY"] = np.nan                      # 5거래일 지연
    btc_days = pd.date_range("2024-01-01", spy_last + pd.Timedelta(days=3), freq="D", name="date")
    btc = pd.Series(40000 + np.cumsum(rng.normal(0, 100, len(btc_days))), index=btc_days, name="BTC-USD")
    close = close.join(btc, how="outer")
    if with_phantoms:
        ph = pd.bdate_range(spy_last + pd.Timedelta(days=1), periods=2)
        for d in ph:
            close.loc[d, "^VIX"] = 16.0                              # SPY 이후 유령 행
        close.loc[ph[0], "AAPL"] = 181.0
    close = close.sort_index()
    close.index.name = "date"
    close = close[sorted(close.columns)]

    spy_ohlc = pd.DataFrame({"Open": close["SPY"].dropna() - 1, "High": close["SPY"].dropna() + 2,
                             "Low": close["SPY"].dropna() - 2, "Close": close["SPY"].dropna(),
                             "Volume": 1_000_000.0}, index=spy_days)
    cboe_days = spy_days.append(pd.bdate_range(spy_last + pd.Timedelta(days=1), periods=1) if with_phantoms else pd.DatetimeIndex([]))
    cboe = pd.DataFrame({"VIX3M": 17.0, "VIX9D": 14.0, "VVIX": 90.0, "SKEW": 140.0}, index=cboe_days)
    cboe.index.name = "date"
    fg = pd.DataFrame({"score": np.linspace(20, 80, len(spy_days))}, index=spy_days)
    for c in D.FG_COMPONENTS:
        fg[c] = 1.0
    fg.loc[spy_days[3], "junk_bond_demand"] = np.nan                # 구성요소 결측 허용
    if with_phantoms:
        fg.loc[spy_last + pd.Timedelta(days=1), "score"] = 55.0     # 주말/미래 라이브 포인트
    fg = fg.sort_index()
    fg.index.name = "date"
    eod_days = spy_days[-40:]
    eod = pd.DataFrame({"open": 400.0, "close": 400.5}, index=eod_days)
    eod["ret"] = eod["close"] / eod["open"] - 1
    eod["half_day"] = False
    eod.loc[eod_days[5], "half_day"] = True
    eod["bar_start"] = np.where(eod["half_day"], "11:30", "15:30")
    eod.index.name = "date"
    meta = {"fetched_at_utc": "2024-03-30T00:00:00Z", "yfinance_version": "test", "warnings": []}
    return D.Bundle(close=close, spy_ohlc=spy_ohlc, cboe=cboe, fg=fg, eod=eod, meta=meta)


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


# ------------------------------------------------------------------
# apply_guards
# ------------------------------------------------------------------
def test_apply_guards_removes_phantom_rows_for_non_247():
    b = make_bundle()
    spy_last = b.close["SPY"].dropna().index[-1]
    assert b.close.loc[b.close.index > spy_last, "^VIX"].notna().sum() == 2   # 전제: 유령 행 존재
    g = _quiet(D.apply_guards, b)
    after = g.close.index > spy_last
    for t in g.close.columns:
        if t not in TWENTY_FOUR_SEVEN:
            assert g.close.loc[after, t].isna().all(), t
    assert g.meta["guards"]["phantom_cells"] == {"^VIX": 2, "AAPL": 1}
    assert any("유령 행" in w for w in g.meta["warnings"])
    # 원본은 그대로
    assert b.close.loc[b.close.index > spy_last, "^VIX"].notna().sum() == 2


def test_apply_guards_keeps_247_rows_after_spy_last():
    b = make_bundle()
    spy_last = b.close["SPY"].dropna().index[-1]
    g = _quiet(D.apply_guards, b)
    btc_after = g.close.loc[g.close.index > spy_last, "BTC-USD"].dropna()
    assert len(btc_after) == 3                       # 주말·이후 3일 유지
    # BTC 만 남은 행은 유지, 전부 NaN 행은 삭제
    assert not g.close.isna().all(axis=1).any()


def test_apply_guards_trims_fg_cboe_eod_after_spy_last():
    b = make_bundle()
    spy_last = b.close["SPY"].dropna().index[-1]
    assert (b.fg.index > spy_last).sum() == 1 and (b.cboe.index > spy_last).sum() == 1
    g = _quiet(D.apply_guards, b)
    assert g.fg.index.max() <= spy_last
    assert g.cboe.index.max() <= spy_last
    assert g.eod.index.max() <= spy_last
    assert g.spy_ohlc.index.max() <= spy_last
    assert g.meta["guards"]["dropped_rows_after_spy_last"] == {"cboe": 1, "fg": 1}


def test_apply_guards_flags_stale_ticker():
    b = make_bundle()
    with pytest.warns(UserWarning, match="지연 티커"):
        g = D.apply_guards(b)
    stale = g.meta["guards"]["stale_tickers"]
    assert "LAGGY" in stale and stale["LAGGY"]["lag_sessions"] == 5
    assert "AAPL" not in stale and "BTC-USD" not in stale
    assert any("LAGGY" in w for w in g.meta["warnings"])


def test_apply_guards_idempotent():
    b = make_bundle()
    g1 = _quiet(D.apply_guards, b)
    g2 = _quiet(D.apply_guards, g1)
    pd.testing.assert_frame_equal(g1.close, g2.close)
    pd.testing.assert_frame_equal(g1.fg, g2.fg)
    pd.testing.assert_frame_equal(g1.eod, g2.eod)
    assert g1.meta["warnings"] == g2.meta["warnings"]          # 경고 중복 누적 없음
    assert g2.meta["guards"]["phantom_cells"] == {}


def test_apply_guards_requires_spy():
    b = make_bundle()
    b.close = b.close.drop(columns=["SPY"])
    with pytest.raises(RuntimeError, match="SPY"):
        D.apply_guards(b)


def test_apply_guards_reports_empty_column():
    b = make_bundle(with_phantoms=False)
    b.close["EMPTY"] = np.nan
    g = _quiet(D.apply_guards, b)
    assert g.meta["guards"]["empty_columns"] == ["EMPTY"]
    assert any("빈 열" in w for w in g.meta["warnings"])


# ------------------------------------------------------------------
# CSV 왕복
# ------------------------------------------------------------------
def test_csv_round_trip(tmp_path):
    b = make_bundle(with_phantoms=False)
    paths = D.save_cache(b, tmp_path)
    for name in D.FILES.values():
        assert (tmp_path / name).exists(), name
    # LF 줄바꿈 · index_label=date · 소수 6자리
    raw = (tmp_path / "close.csv").read_bytes()
    assert b"\r\n" not in raw
    header = raw.split(b"\n", 1)[0].decode()
    assert header.startswith("date,")
    first_row = raw.split(b"\n", 2)[1].decode()
    assert any(len(x.split(".")[1]) == 6 for x in first_row.split(",")[1:] if "." in x)

    lb = _quiet(D.load_cache, tmp_path, guards=False)
    for key in ("close", "spy_ohlc", "cboe", "fg"):
        a, c = getattr(b, key), getattr(lb, key)
        assert isinstance(c.index, pd.DatetimeIndex) and c.index.tz is None and c.index.name == "date"
        assert list(a.columns) == list(c.columns), key
        pd.testing.assert_index_equal(a.index, c.index, check_names=False)
        np.testing.assert_allclose(a.values.astype(float), c.values.astype(float), atol=1e-6, equal_nan=True)
    e = lb.eod
    assert list(e.columns) == D.EOD_COLUMNS
    assert e["half_day"].dtype == bool and int(e["half_day"].sum()) == 1
    assert set(e["bar_start"].unique()) == {"15:30", "11:30"}
    np.testing.assert_allclose(e["ret"].values, b.eod["ret"].values, atol=1e-6)
    assert lb.meta["fetched_at_utc"] == "2024-03-30T00:00:00Z"
    assert lb.meta["schema_version"] == D.SCHEMA_VERSION
    assert paths["meta"].read_text(encoding="utf-8").endswith("}\n")


def test_load_cache_applies_guards_by_default(tmp_path):
    b = make_bundle(with_phantoms=True)
    D.save_cache(b, tmp_path)                  # 저장은 가드 없이(테스트) → 로드 시 가드가 잡아야 한다
    lb = _quiet(D.load_cache, tmp_path)
    spy_last = lb.close["SPY"].dropna().index[-1]
    assert lb.close.loc[lb.close.index > spy_last, "^VIX"].isna().all()
    assert lb.fg.index.max() <= spy_last
    assert lb.meta["guards"]["phantom_cells"] == {"^VIX": 2, "AAPL": 1}


def test_load_cache_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        D.load_cache(tmp_path)


def test_empty_csv_round_trip(tmp_path):
    b = make_bundle(with_phantoms=False)
    b.cboe = D._empty_frame(D.CBOE_COLUMNS)
    b.eod = D._coerce_eod(None)
    D.save_cache(b, tmp_path)
    lb = _quiet(D.load_cache, tmp_path)
    assert list(lb.cboe.columns) == D.CBOE_COLUMNS and lb.cboe.empty
    assert isinstance(lb.cboe.index, pd.DatetimeIndex)
    assert list(lb.eod.columns) == D.EOD_COLUMNS and lb.eod.empty


# ------------------------------------------------------------------
# 파서
# ------------------------------------------------------------------
def test_parse_cboe_csv_ohlc_and_single_value():
    ohlc = "DATE,OPEN,HIGH,LOW,CLOSE\n09/18/2009,25.91,26.66,25.91,26.54\n09/21/2009,25.0,25.5,24.9,25.20\n"
    s = D.parse_cboe_csv(ohlc, "VIX3M")
    assert s.name == "VIX3M" and s.index.tz is None
    assert s.loc["2009-09-18"] == pytest.approx(26.54)
    single = "DATE,VVIX\n03/07/2006,72.10\n03/06/2006,71.73\n03/07/2006,72.20\n"
    v = D.parse_cboe_csv(single, "VVIX")
    assert list(v.index.strftime("%Y-%m-%d")) == ["2006-03-06", "2006-03-07"]   # 정렬 + 중복은 마지막
    assert v.iloc[-1] == pytest.approx(72.20)
    with pytest.raises(ValueError):
        D.parse_cboe_csv("FOO,BAR\n1,2\n", "X")


def _ms(date: str, hour_utc: int = 0, minute: int = 0) -> float:
    ts = pd.Timestamp(date, tz="UTC") + pd.Timedelta(hours=hour_utc, minutes=minute)
    return float(ts.value // 1_000_000)


def test_parse_fg_json_dedupes_live_point_and_outer_joins_components():
    j = {
        "fear_and_greed": {"score": 41.0},
        "fear_and_greed_historical": {"data": [
            {"x": _ms("2020-08-03"), "y": 50.0}, {"x": _ms("2020-08-04"), "y": 52.0},
            {"x": _ms("2020-08-05"), "y": 40.0},
            {"x": _ms("2020-08-05", 20, 33), "y": 41.0},            # 같은 날짜 라이브 포인트 → 이긴다
        ]},
        "junk_bond_demand": {"data": [{"x": _ms("2020-08-03"), "y": 1.9}]},   # 구성요소 결측 허용
        "put_call_options": {"data": [{"x": _ms("2020-08-04"), "y": 0.6}, {"x": _ms("2020-08-05"), "y": 0.7}]},
        "new_unknown_component": {"data": [{"x": _ms("2020-08-04"), "y": 1.0}]},
    }
    df = D.parse_fg_json(j)
    assert list(df.columns) == D.FG_COLUMNS
    assert list(df.index.strftime("%Y-%m-%d")) == ["2020-08-03", "2020-08-04", "2020-08-05"]
    assert df.loc["2020-08-05", "score"] == pytest.approx(41.0)
    assert df.loc["2020-08-03", "junk_bond_demand"] == pytest.approx(1.9)
    assert np.isnan(df.loc["2020-08-04", "junk_bond_demand"])
    assert df.loc["2020-08-05", "put_call_options"] == pytest.approx(0.7)
    assert np.isnan(df["market_momentum_sp500"]).all()
    assert df.attrs["last_point_is_live"] is True
    assert df.attrs["last_point_utc"] == "2020-08-05T20:33:00Z"
    assert any("market_momentum_sp500" in w for w in df.attrs["warnings"])
    assert any("new_unknown_component" in w for w in df.attrs["warnings"])
    with pytest.raises(ValueError):
        D.parse_fg_json({"fear_and_greed_historical": {"data": []}})


def test_fg_live_point_uses_et_date():
    # 2020-08-06 01:00 UTC = 2020-08-05 21:00 ET → 라이브 포인트는 ET 날짜(08-05)
    assert D._fg_point_date(_ms("2020-08-06", 1, 0)) == pd.Timestamp("2020-08-05")
    assert D._fg_point_date(_ms("2020-08-06")) == pd.Timestamp("2020-08-06")


def _bars(sessions: dict[str, list[str]]) -> pd.DataFrame:
    """{날짜: [봉 시작 시각 ...]} → Open/Close 1h 봉 프레임 (ET-aware 인덱스, yfinance 형식과 동일하게 UTC 로 변환)."""
    idx, opens, closes = [], [], []
    for d, times in sessions.items():
        for i, t in enumerate(times):
            idx.append(pd.Timestamp(f"{d} {t}", tz=ET).tz_convert("UTC"))
            opens.append(100.0 + i)
            closes.append(100.0 + i + 0.5)
    df = pd.DataFrame({"Open": opens, "Close": closes}, index=pd.DatetimeIndex(idx, name="Datetime"))
    df.columns = pd.MultiIndex.from_product([df.columns, ["SPY"]], names=["Price", "Ticker"])
    return df


def test_eod_from_intraday_full_half_incomplete_and_glitch():
    full = ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]
    bars = _bars({
        "2025-11-26": full,                                  # 정규장
        "2025-11-28": ["09:30", "10:30", "11:30"],           # 반일장
        "2026-01-30": ["09:30"],                             # 데이터 결손 세션 → 제외
        "2026-02-02": ["09:30", "10:30", "11:30", "12:30"],  # 오늘, 장중(13:00) → 보류
    })
    now = dt.datetime(2026, 2, 2, 13, 0, tzinfo=ET)
    out = D.eod_from_intraday(bars, now)
    assert list(out.columns) == D.EOD_COLUMNS
    assert list(out.index.strftime("%Y-%m-%d")) == ["2025-11-26", "2025-11-28"]
    assert out.index.tz is None
    r = out.loc["2025-11-26"]
    assert r["bar_start"] == "15:30" and not bool(r["half_day"])
    assert r["open"] == pytest.approx(106.0) and r["close"] == pytest.approx(106.5)
    assert r["ret"] == pytest.approx(106.5 / 106.0 - 1)
    h = out.loc["2025-11-28"]
    assert h["bar_start"] == "11:30" and bool(h["half_day"])
    assert out["half_day"].dtype == bool
    assert any("2026-01-30" in w for w in out.attrs["warnings"])
    # 오늘 세션: 15:30 봉이 있어도 16:05 전이면 보류, 이후면 확정
    today_bars = _bars({"2026-02-02": full})
    early = D.eod_from_intraday(today_bars, dt.datetime(2026, 2, 2, 15, 50, tzinfo=ET))
    assert early.empty and any("미완성" in w for w in early.attrs["warnings"])
    late = D.eod_from_intraday(today_bars, dt.datetime(2026, 2, 2, 16, 10, tzinfo=ET))
    assert len(late) == 1 and late.index[0] == pd.Timestamp("2026-02-02")
    # 반일장 오늘: 13:05 이후 확정
    half_today = _bars({"2026-02-02": ["09:30", "10:30", "11:30"]})
    assert D.eod_from_intraday(half_today, dt.datetime(2026, 2, 2, 12, 40, tzinfo=ET)).empty
    assert len(D.eod_from_intraday(half_today, dt.datetime(2026, 2, 2, 13, 10, tzinfo=ET))) == 1


def test_split_prices_missing_ticker_and_spy_trim():
    days = pd.bdate_range("2024-01-01", "2024-01-10")
    old = pd.bdate_range("2023-12-20", "2023-12-29")           # SPY 이전 (^GSPC 만 존재) → 잘림
    idx = old.append(days)
    cols = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], ["SPY", "^GSPC", "NOPE"]],
                                      names=["Price", "Ticker"])
    raw = pd.DataFrame(np.nan, index=idx, columns=cols)
    for f in ("Open", "High", "Low", "Close"):
        raw.loc[days, (f, "SPY")] = 400.0
        raw.loc[idx, (f, "^GSPC")] = 4000.0
    raw.loc[days, ("Volume", "SPY")] = 1e6
    raw.index.name = "Date"
    close, spy_ohlc, missing = D.split_prices(raw, ["SPY", "^GSPC", "NOPE"])
    assert missing == ["NOPE"]
    assert list(close.columns) == ["SPY", "^GSPC"]
    assert close.index[0] == days[0] and len(close) == len(days)
    assert list(spy_ohlc.columns) == D.SPY_OHLC_COLUMNS
    assert spy_ohlc.index.equals(pd.DatetimeIndex(days))
    assert close.index.name == "date" and close.index.tz is None
    with pytest.raises(RuntimeError, match="SPY"):
        D.split_prices(raw.drop(columns=["SPY"], level=1), ["^GSPC"])


def test_merge_helpers():
    a = pd.DataFrame({"score": [1.0, 2.0], "put_call_options": [np.nan, 0.5]},
                     index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="date"))
    b = pd.DataFrame({"score": [2.5, 3.0], "put_call_options": [np.nan, 0.6]},
                     index=pd.DatetimeIndex(["2024-01-03", "2024-01-04"], name="date"))
    m = D._merge_cells(a, b, D.FG_COLUMNS)
    assert list(m.columns) == D.FG_COLUMNS and len(m) == 3
    assert m.loc["2024-01-03", "score"] == 2.5                       # 새 값 우선
    assert m.loc["2024-01-03", "put_call_options"] == 0.5            # 새 응답에 없는 셀은 기존 유지
    assert m.loc["2024-01-02", "score"] == 1.0
    e1 = D._coerce_eod(pd.DataFrame({"open": [1.0], "close": [1.1], "ret": [0.1], "half_day": [False], "bar_start": ["15:30"]},
                                    index=pd.DatetimeIndex(["2024-01-02"])))
    e2 = D._coerce_eod(pd.DataFrame({"open": [2.0, 3.0], "close": [2.2, 3.3], "ret": [0.1, 0.1],
                                     "half_day": [True, False], "bar_start": ["11:30", "15:30"]},
                                    index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"])))
    r = D._merge_rows(e1, e2)
    assert len(r) == 2 and r.loc["2024-01-02", "open"] == 2.0 and bool(r.loc["2024-01-02", "half_day"]) is True
    assert D._merge_rows(None, None).empty and list(D._merge_rows(None, None).columns) == D.EOD_COLUMNS


# ------------------------------------------------------------------
# build_cache 흐름 (원천을 몽키패치 → 네트워크 없음)
# ------------------------------------------------------------------
def _fake_raw(days: pd.DatetimeIndex, tickers: list[str], extra_days: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    idx = days if extra_days is None else days.append(extra_days).sort_values()
    cols = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], tickers], names=["Price", "Ticker"])
    raw = pd.DataFrame(np.nan, index=idx, columns=cols)
    rng = np.random.default_rng(1)
    for t in tickers:
        if t == "MISSING":
            continue
        d = idx if t in TWENTY_FOUR_SEVEN else days
        base = 100 + np.cumsum(rng.normal(0, 1, len(d)))
        for f in ("Open", "High", "Low", "Close"):
            raw.loc[d, (f, t)] = base
        raw.loc[d, ("Volume", t)] = 1e6
    # 유령 행: ^VIX 가 SPY 이후 날짜에 값
    if extra_days is not None and "^VIX" in tickers:
        raw.loc[extra_days[0], ("Close", "^VIX")] = 20.0
    raw.attrs["missing"] = [t for t in tickers if t == "MISSING"]
    return raw


def test_build_cache_offline_and_append(tmp_path, monkeypatch):
    days = pd.bdate_range("2024-01-01", "2024-02-29")
    spy_last = days[-1]
    tickers = ["SPY", "^VIX", "BTC-USD", "MISSING"]
    extra = pd.date_range(spy_last + pd.Timedelta(days=1), periods=2, freq="D")
    monkeypatch.setattr(D, "ALL_TICKERS", tickers)
    monkeypatch.setattr(D, "fetch_prices", lambda tk, period="max": _fake_raw(days, tk, extra))
    cboe1 = pd.DataFrame({"VIX3M": 17.0, "VIX9D": 14.0, "VVIX": 90.0, "SKEW": 140.0}, index=days[:-2])
    monkeypatch.setattr(D, "fetch_cboe", lambda series=None: cboe1)
    fg1 = pd.DataFrame({"score": np.linspace(30, 60, len(days))}, index=days).reindex(columns=D.FG_COLUMNS)
    fg1.attrs.update({"last_point_utc": "x", "last_point_is_live": False})
    monkeypatch.setattr(D, "fetch_fg_history", lambda seed_date=None: fg1)
    eod1 = D._coerce_eod(pd.DataFrame({"open": 100.0, "close": 100.2, "ret": 0.002, "half_day": False, "bar_start": "15:30"},
                                      index=days[-30:]))
    monkeypatch.setattr(D, "fetch_spy_eod", lambda period="730d", now_et=None: eod1)

    b = _quiet(D.build_cache, tmp_path)
    assert (tmp_path / "close.csv").exists() and (tmp_path / "meta.json").exists()
    assert "MISSING" not in b.close.columns and "MISSING" in b.meta["missing_tickers"]
    assert any("MISSING" in w for w in b.meta["warnings"])
    assert b.meta["mode"] == "build" and b.meta["fetched_at_utc"].endswith("Z")
    assert b.meta["tickers"]["SPY"]["last"] == spy_last.strftime("%Y-%m-%d")
    # 유령 행 제거됨, BTC 는 유지
    assert b.close.loc[b.close.index > spy_last, "^VIX"].isna().all()
    assert b.close.loc[b.close.index > spy_last, "BTC-USD"].notna().sum() == 2
    assert b.meta["guards"]["phantom_cells"] == {"^VIX": 1}
    assert len(b.eod) == 30 and len(b.fg) == len(days)
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["tickers"]["BTC-USD"]["n"] == len(days) + 2
    assert meta["frames"]["eod"]["rows"] == 30

    # 두 번째 실행(update_daily): 원천이 최근 며칠만 줘도 기존 행이 유지되고 새 날짜가 append 된다
    days2 = pd.bdate_range("2024-01-01", "2024-03-05")
    new_days = days2[len(days):]
    monkeypatch.setattr(D, "fetch_prices", lambda tk, period="max": _fake_raw(days2, tk))
    fg2 = pd.DataFrame({"score": [70.0, 71.0, 72.0]}, index=days2[-3:]).reindex(columns=D.FG_COLUMNS)
    fg2.loc[days2[-3], "put_call_options"] = 0.9
    seeds = []

    def fake_fg(seed_date=None):
        seeds.append(seed_date)
        return fg2

    monkeypatch.setattr(D, "fetch_fg_history", fake_fg)
    eod2 = D._coerce_eod(pd.DataFrame({"open": 100.0, "close": 99.0, "ret": -0.01, "half_day": True, "bar_start": "11:30"},
                                      index=days2[-5:]))
    monkeypatch.setattr(D, "fetch_spy_eod", lambda period="730d", now_et=None: eod2)

    def fail_cboe(series=None):
        raise RuntimeError("cdn down")

    monkeypatch.setattr(D, "fetch_cboe", fail_cboe)
    u = _quiet(D.update_daily, tmp_path)
    assert u.meta["mode"] == "update"
    assert seeds and seeds[0] == (spy_last - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    assert len(u.fg) == len(days2)                                   # 기존 + 새 날짜
    assert u.fg.loc[days2[-3], "score"] == 70.0                      # 새 값 우선
    assert u.fg.loc[days[0], "score"] == pytest.approx(30.0)         # 기존 유지
    assert u.fg.loc[days2[-3], "put_call_options"] == 0.9
    assert len(u.eod) == 30 + 5 - 2                                 # eod2 5일 중 겹치는 2일은 대체, 3일 append
    assert bool(u.eod.loc[days2[-1], "half_day"]) is True
    assert bool(u.eod.loc[days[-2], "half_day"]) is True            # 겹친 날짜는 새 행으로 대체
    assert bool(u.eod.loc[days[-6], "half_day"]) is False           # 겹치지 않은 기존 행 그대로
    pd.testing.assert_frame_equal(u.cboe, b.cboe)                   # CBOE 실패 → 기존 유지
    assert any("CBOE 조회 실패" in w for w in u.meta["warnings"])
    # 재로드 == 반환값
    lb = _quiet(D.load_cache, tmp_path)
    pd.testing.assert_frame_equal(lb.close, u.close, check_exact=False, atol=1e-6)
    pd.testing.assert_frame_equal(lb.eod, u.eod, check_exact=False, atol=1e-6)
    assert lb.meta["warnings"] == u.meta["warnings"]


def test_build_cache_fails_loudly_without_spy(tmp_path, monkeypatch):
    days = pd.bdate_range("2024-01-01", "2024-01-31")
    monkeypatch.setattr(D, "ALL_TICKERS", ["^VIX", "BTC-USD"])
    monkeypatch.setattr(D, "fetch_prices", lambda tk, period="max": _fake_raw(days, tk))
    with pytest.raises(RuntimeError, match="SPY"):
        _quiet(D.build_cache, tmp_path)
    assert not (tmp_path / "close.csv").exists()


# ------------------------------------------------------------------
# 퇴행한 다운로드 거부 (부분 yfinance 응답 · 레이트리밋) — 기존 close.csv 를 지키고, 기록되지 않은 채 커밋되지 않게
# ------------------------------------------------------------------
def _build_small_cache(tmp_path, monkeypatch, tickers, days):
    monkeypatch.setattr(D, "ALL_TICKERS", tickers)
    monkeypatch.setattr(D, "fetch_prices", lambda tk, period="max": _fake_raw(days, tk))
    monkeypatch.setattr(D, "fetch_cboe", lambda series=None: pd.DataFrame(
        {"VIX3M": 17.0, "VIX9D": 14.0, "VVIX": 90.0, "SKEW": 140.0}, index=days))
    fg = pd.DataFrame({"score": np.linspace(30, 60, len(days))}, index=days).reindex(columns=D.FG_COLUMNS)
    monkeypatch.setattr(D, "fetch_fg_history", lambda seed_date=None: fg)
    eod = D._coerce_eod(pd.DataFrame({"open": 100.0, "close": 100.2, "ret": 0.002, "half_day": False, "bar_start": "15:30"},
                                     index=days[-30:]))
    monkeypatch.setattr(D, "fetch_spy_eod", lambda period="730d", now_et=None: eod)
    return _quiet(D.build_cache, tmp_path)


def _snapshot(tmp_path) -> dict[str, bytes]:
    return {n: (tmp_path / n).read_bytes() for n in D.FILES.values()}


def test_update_daily_refuses_partial_download(tmp_path, monkeypatch):
    """이전에 있던 NFLX 가 전부-NaN 으로 오면(yfinance 는 예외를 내지 않는다) 저장을 거부하고 캐시는 바이트 단위로 그대로."""
    days = pd.bdate_range("2024-01-01", "2024-03-29")
    tickers = ["SPY", "^VIX", "BTC-USD", "NFLX"]
    b = _build_small_cache(tmp_path, monkeypatch, tickers, days)
    assert "NFLX" in b.close.columns
    before = _snapshot(tmp_path)
    days2 = pd.bdate_range("2024-01-01", "2024-04-05")

    def partial(tk, period="max"):
        raw = _fake_raw(days2, tk)
        for f in ("Open", "High", "Low", "Close", "Volume"):
            raw[(f, "NFLX")] = np.nan                       # 레이트리밋으로 한 티커만 빈 열
        return raw

    monkeypatch.setattr(D, "fetch_prices", partial)
    with pytest.raises(RuntimeError, match="NFLX"):
        _quiet(D.update_daily, tmp_path)
    assert _snapshot(tmp_path) == before                    # close.csv 등 여섯 파일 모두 그대로
    assert not list(tmp_path.glob("*.tmp"))
    # 행 수가 크게 줄어든 티커 (^VIX 가 마지막 10행만 옴) 도 거부
    def shrunk(tk, period="max"):
        raw = _fake_raw(days2, tk)
        raw.loc[days2[:-10], ("Close", "^VIX")] = np.nan
        return raw

    monkeypatch.setattr(D, "fetch_prices", shrunk)
    with pytest.raises(RuntimeError, match="VIX"):
        _quiet(D.update_daily, tmp_path)
    assert _snapshot(tmp_path) == before
    # SPY 마지막 일자가 뒤로 가도 거부
    monkeypatch.setattr(D, "fetch_prices", lambda tk, period="max": _fake_raw(days[:-5], tk))
    with pytest.raises(RuntimeError, match="SPY 마지막 일자 후퇴"):
        _quiet(D.update_daily, tmp_path)
    assert _snapshot(tmp_path) == before
    # config 에서 뺀 티커는 허용, 정상 응답은 통과, force=True 는 퇴행도 통과(경고 기록)
    monkeypatch.setattr(D, "ALL_TICKERS", ["SPY", "^VIX", "BTC-USD"])
    monkeypatch.setattr(D, "fetch_prices", lambda tk, period="max": _fake_raw(days2, tk))
    u = _quiet(D.update_daily, tmp_path)
    assert "NFLX" not in u.close.columns and u.spy_ohlc.index[-1] == days2[-1]
    monkeypatch.setattr(D, "ALL_TICKERS", tickers)
    monkeypatch.setattr(D, "fetch_prices", partial)
    f = _quiet(D.update_daily, tmp_path, force=True)
    assert "NFLX" not in f.close.columns and any("force=True" in w for w in f.meta["warnings"])


def test_save_cache_is_atomic_and_load_cache_checks_consistency(tmp_path, monkeypatch):
    """네 번째 파일 쓰기에서 예외가 나면 여섯 파일 모두 이전 상태 그대로이고 *.tmp 가 남지 않는다;
    meta.spy_last 와 close/spy_ohlc 마지막 일자가 어긋나면 load_cache 가 예외를 낸다."""
    b = make_bundle(with_phantoms=False)
    b.meta["spy_last"] = b.spy_ohlc.index[-1].strftime("%Y-%m-%d")
    D.save_cache(b, tmp_path)
    before = _snapshot(tmp_path)
    assert not list(tmp_path.glob("*.tmp"))
    calls = {"n": 0}
    real = D._write_csv

    def flaky(df, path):
        calls["n"] += 1
        if calls["n"] == 4:
            raise OSError("disk full")
        return real(df, path)

    monkeypatch.setattr(D, "_write_csv", flaky)
    b2 = D.Bundle(close=b.close.iloc[:-3], spy_ohlc=b.spy_ohlc.iloc[:-3], cboe=b.cboe, fg=b.fg, eod=b.eod,
                  meta={**b.meta, "spy_last": b.spy_ohlc.index[-4].strftime("%Y-%m-%d")})
    with pytest.raises(OSError, match="disk full"):
        D.save_cache(b2, tmp_path)
    assert _snapshot(tmp_path) == before and not list(tmp_path.glob("*.tmp"))
    monkeypatch.setattr(D, "_write_csv", real)
    lb = _quiet(D.load_cache, tmp_path)
    assert lb.spy_ohlc.index[-1] == b.spy_ohlc.index[-1]
    # 어긋난 meta (close 는 새 날짜, meta 는 옛 날짜 — 예전 방식의 반쯤 쓰인 캐시)
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    meta["spy_last"] = b.spy_ohlc.index[-4].strftime("%Y-%m-%d")
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(RuntimeError, match="캐시 불일치"):
        _quiet(D.load_cache, tmp_path)
    # meta.spy_last 가 없는 번들(픽스처)은 검사하지 않는다
    meta.pop("spy_last")
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert _quiet(D.load_cache, tmp_path).spy_ohlc.index[-1] == b.spy_ohlc.index[-1]


# ------------------------------------------------------------------
# 계약 표면
# ------------------------------------------------------------------
def test_contract_surface():
    import inspect
    for name in ("fetch_prices", "fetch_cboe", "fetch_fg_history", "fetch_fg_today", "fetch_spy_eod",
                 "build_cache", "update_daily", "load_cache", "apply_guards"):
        assert callable(getattr(D, name)), name
    assert [f.name for f in D.Bundle.__dataclass_fields__.values()] == ["close", "spy_ohlc", "cboe", "fg", "eod", "meta"]
    assert inspect.signature(D.fetch_prices).parameters["period"].default == "max"
    assert inspect.signature(D.fetch_spy_eod).parameters["period"].default == "730d"
    assert D.FILES == {"close": "close.csv", "spy_ohlc": "spy_ohlc.csv", "cboe": "cboe.csv",
                       "fg": "fg_history.csv", "eod": "spy_eod.csv", "meta": "meta.json"}
