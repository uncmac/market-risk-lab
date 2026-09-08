# -*- coding: utf-8 -*-
"""mrl/targets.py 검증 — 합성 시계열로 목표변수·에피소드 정의를 확인한다.

실행: 프로젝트 루트에서  python -m pytest tests/test_targets.py -q
(data/close.csv 가 있으면 실제 SPY 에피소드 수도 출력한다; 없으면 그 테스트는 skip)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mrl import targets as T
from mrl.config import DATA_DIR, DD_TARGETS, HORIZONS

# ------------------------------------------------------------------
# 합성 데이터
# ------------------------------------------------------------------
def _random_walk(n: int = 400, seed: int = 0, mu: float = 0.0003, sigma: float = 0.012) -> pd.Series:
    idx = pd.bdate_range("2018-01-02", periods=n)
    rng = np.random.default_rng(seed)
    px = 100.0 * np.exp(np.cumsum(rng.normal(mu, sigma, n)))
    return pd.Series(px, index=idx, name="SPY")


# 손으로 만든 에피소드 시나리오 (일 단위 종가). 기대값은 아래 테스트에 주석으로.
#  A : 102 고점 → 96.5 (-5.4%) → 103 회복                     [5%: 1개, 10%: 0개]
#  B : 104 고점 → 91 저점, 97 까지 반등(+6.6%) → 92 (-5.2%) 재하락 → 90 → 105 회복
#      [5%: 2개(회복일 공유), 10%: 1개(반등 <10% 이라 분할 없음, 저점 90)]
#  C : 106 고점 → 100 → 100.4(0.4% 미세 반등) → 95 → 108 회복   [5%: 1개(분할 없음), 10%: 1개]
#  D : 109 고점 → 103 → 108.5(+5.3% 확정 반등) → 97 (하루 급락) → 110 회복
#      [5%: 2개(급락일에 분할 우선), 10%: 1개]
#  E : 111 고점 → 99 → 100 (끝, 미회복)                        [5%: 1개, 10%: 1개, recovery NaT]
_SCENARIO = [100, 101, 102, 100, 97, 96.5, 98, 100, 103, 104, 100, 95, 91, 93, 96, 97, 94, 92, 90, 95, 100,
             105, 106, 101, 100, 100.4, 95, 108, 109, 103, 108.5, 97, 110, 111, 105, 99, 100]


def _scenario() -> pd.Series:
    idx = pd.bdate_range("2021-01-04", periods=len(_SCENARIO))
    return pd.Series(_SCENARIO, index=idx, dtype=float, name="SPY")


# ------------------------------------------------------------------
# make_targets
# ------------------------------------------------------------------
def test_columns_and_nan_tails():
    close = _random_walk(400)
    tg = T.make_targets(close)
    assert list(tg.columns) == list(T.TARGET_COLUMNS)
    assert tg.index.equals(close.index)
    for h in HORIZONS:
        for col in (f"fwd_ret_{h}", f"y_sign_{h}"):
            assert tg[col].iloc[-h:].isna().all(), col
            assert tg[col].iloc[:-h].notna().all(), col
    for col, (_thr, w) in DD_TARGETS.items():
        for c in (col, f"fwd_maxdd_{w}"):
            assert tg[c].iloc[-w:].isna().all(), c
            assert tg[c].iloc[:-w].notna().all(), c
    # y_vol_20: 기준 중앙값에 252개의 20일-RV 가 필요 → 처음 271행 NaN, 272번째부터 값, 마지막 20행 NaN
    v = tg["y_vol_20"]
    assert v.iloc[:271].isna().all()
    assert v.iloc[271:-20].notna().all()
    assert v.iloc[-20:].isna().all()


def test_binary_columns_are_float_zero_one():
    tg = T.make_targets(_random_walk(300))
    for col in T.Y_COLUMNS:
        assert tg[col].dtype == float
        vals = set(tg[col].dropna().unique())
        assert vals <= {0.0, 1.0}, col


def test_matches_bruteforce_definition():
    """벡터화 결과가 정의를 그대로 옮긴 O(n·h) 루프와 일치하는가."""
    close = _random_walk(420, seed=3)
    tg = T.make_targets(close)
    arr = close.to_numpy()
    n = len(arr)
    logret = np.diff(np.log(arr))                      # logret[i] = log(arr[i+1]/arr[i]) → 날짜 i+1 의 수익률

    def rv20_at(s):                                    # 날짜 s 까지의 20일 RV (r[s-19..s])
        if s < 20:
            return np.nan
        r = logret[s - 20: s]                          # logret 인덱스 s-20..s-1 ↔ 날짜 s-19..s
        return np.std(r, ddof=1) * math.sqrt(252)

    for t in range(n):
        for h in HORIZONS:
            if t + h < n:
                fr = arr[t + h] / arr[t] - 1
                assert math.isclose(tg[f"fwd_ret_{h}"].iloc[t], fr, rel_tol=1e-12)
                assert tg[f"y_sign_{h}"].iloc[t] == (1.0 if fr > 0 else 0.0)
            else:
                assert math.isnan(tg[f"fwd_ret_{h}"].iloc[t])
        for col, (thr, w) in DD_TARGETS.items():
            if t + w < n:
                mdd = arr[t + 1: t + w + 1].min() / arr[t] - 1
                assert math.isclose(tg[f"fwd_maxdd_{w}"].iloc[t], mdd, rel_tol=1e-12)
                assert tg[col].iloc[t] == (1.0 if mdd <= -thr else 0.0)
            else:
                assert math.isnan(tg[col].iloc[t])
        # y_vol_20
        if t + 20 < n and t >= 271:
            fwd = rv20_at(t + 20)
            ref = np.median([rv20_at(s) for s in range(t - 251, t + 1)])
            assert tg["y_vol_20"].iloc[t] == (1.0 if fwd > ref else 0.0), t
        else:
            assert math.isnan(tg["y_vol_20"].iloc[t]), t


def test_point_in_time_no_lookahead():
    """t 행의 값은 t+창 이후 데이터에 의존하지 않고, 시계열을 잘라도 실현된 행은 같아야 한다."""
    close = _random_walk(400, seed=5)
    full = T.make_targets(close)
    # (1) t+61 이후를 크게 바꿔도 0..t 행은 그대로
    t = 320
    tweaked = close.copy()
    tweaked.iloc[t + 61:] *= 1.5
    tw = T.make_targets(tweaked)
    pd.testing.assert_frame_equal(full.iloc[: t + 1], tw.iloc[: t + 1])
    # (2) 잘라낸 시계열에서 실현된 값은 전체 시계열의 값과 동일
    cut = T.make_targets(close.iloc[:350])
    for col in T.TARGET_COLUMNS:
        m = cut[col].notna()
        pd.testing.assert_series_equal(cut.loc[m, col], full.loc[cut.index[m], col], check_names=False)


def test_input_validation_and_nan_rows_dropped():
    close = _random_walk(50)
    dup = pd.concat([close, close.iloc[[10]]]).sort_index()
    with pytest.raises(ValueError):
        T.make_targets(dup)
    with pytest.raises(ValueError):
        T.make_targets(close.iloc[::-1])
    bad = close.copy()
    bad.iloc[3] = 0.0
    with pytest.raises(ValueError):
        T.make_targets(bad)
    with pytest.raises(TypeError):
        T.make_targets(close.to_frame())
    # BTC 주말 union 인덱스처럼 NaN 행이 섞여 있으면 제거하고 계산
    union_idx = pd.date_range(close.index[0], close.index[-1], freq="D")
    with_nan = close.reindex(union_idx)
    tg = T.make_targets(with_nan)
    assert tg.index.equals(close.index)
    pd.testing.assert_frame_equal(tg, T.make_targets(close), check_freq=False)


# ------------------------------------------------------------------
# base_rates
# ------------------------------------------------------------------
def test_base_rates_slice_and_validation():
    tg = T.make_targets(_random_walk(600, seed=7))
    br = T.base_rates(tg)
    assert list(br.index) == list(T.Y_COLUMNS)
    assert br.name == "base_rate"
    assert br.notna().all()
    for col in T.Y_COLUMNS:
        assert math.isclose(br[col], tg[col].mean())
    start, end = tg.index[300], tg.index[500]          # y_vol_20 이 실현된 구간
    br2 = T.base_rates(tg, start, end)
    for col in T.Y_COLUMNS:
        assert math.isclose(br2[col], tg.loc[start:end, col].mean())
    # 실현된 행이 하나도 없는 구간은 NaN (조용히 0 으로 만들지 않는다)
    assert math.isnan(T.base_rates(tg, tg.index[0], tg.index[100])["y_vol_20"])
    with pytest.raises(ValueError):
        T.base_rates(tg.drop(columns=["y_dd5_20"]))


# ------------------------------------------------------------------
# episodes
# ------------------------------------------------------------------
def _d(s: pd.Series, i: int) -> pd.Timestamp:
    return s.index[i]


def test_episodes_5pct_scenario():
    s = _scenario()
    ep = T.episodes(s, 0.05)
    assert list(ep.columns) == list(T.EPISODE_COLUMNS)
    assert len(ep) == 7
    # (peak_i, trough_i, recovery_i or None)
    expected = [(2, 5, 8), (9, 12, 21), (15, 18, 21), (22, 26, 27), (28, 29, 32), (30, 31, 32), (33, 35, None)]
    for row, (p, tr, rec) in zip(ep.itertuples(index=False), expected):
        assert row.peak_date == _d(s, p)
        assert row.trough_date == _d(s, tr)
        assert math.isclose(row.depth, s.iloc[tr] / s.iloc[p] - 1)
        assert row.depth <= -0.05
        assert row.days_to_trough == tr - p
        if rec is None:
            assert pd.isna(row.recovery_date) and math.isnan(row.days_to_recover)
        else:
            assert row.recovery_date == _d(s, rec)
            assert row.days_to_recover == rec - tr
    # B 구간: 분할된 두 에피소드가 회복일을 공유하고, 두 번째의 고점은 반등 고점(97)
    assert ep.loc[1, "recovery_date"] == ep.loc[2, "recovery_date"] == _d(s, 21)
    assert s.loc[ep.loc[2, "peak_date"]] == 97
    # D 구간: 확정 반등 뒤 하루 급락 → 연장이 아니라 분할 (첫 조각의 저점은 103 그대로)
    assert s.loc[ep.loc[4, "trough_date"]] == 103
    assert s.loc[ep.loc[5, "peak_date"]] == 108.5


def test_episodes_5pct_scenario_without_split_rule():
    """split=False: ATH 기준 한 underwater 구간 = 한 에피소드 (B·D 구간이 합쳐지고 저점은 구간 최저)."""
    s = _scenario()
    ep = T.episodes(s, 0.05, split=False)
    assert len(ep) == 5
    expected = [(2, 5, 8), (9, 18, 21), (22, 26, 27), (28, 31, 32), (33, 35, None)]
    for row, (p, tr, rec) in zip(ep.itertuples(index=False), expected):
        assert row.peak_date == _d(s, p) and row.trough_date == _d(s, tr)
        assert math.isclose(row.depth, s.iloc[tr] / s.iloc[p] - 1)
        assert row.days_to_trough == tr - p
        if rec is None:
            assert pd.isna(row.recovery_date)
        else:
            assert row.recovery_date == _d(s, rec) and row.days_to_recover == rec - tr
    # 10% 에서는 분할이 일어나지 않았으므로 두 방식이 같다
    pd.testing.assert_frame_equal(T.episodes(s, 0.10, split=False), T.episodes(s, 0.10))


def test_episodes_10pct_scenario_no_split_without_threshold_rebound():
    s = _scenario()
    ep = T.episodes(s, 0.10)
    assert len(ep) == 4
    expected = [(9, 18, 21), (22, 26, 27), (28, 31, 32), (33, 35, None)]
    for row, (p, tr, rec) in zip(ep.itertuples(index=False), expected):
        assert row.peak_date == _d(s, p) and row.trough_date == _d(s, tr)
        assert row.depth <= -0.10
        if rec is None:
            assert pd.isna(row.recovery_date)
        else:
            assert row.recovery_date == _d(s, rec)
    # 20% 낙폭은 없음 → 빈 표(열은 유지)
    ep20 = T.episodes(s, 0.20)
    assert len(ep20) == 0 and list(ep20.columns) == list(T.EPISODE_COLUMNS)


def test_episodes_flat_or_rising_series_is_empty():
    idx = pd.bdate_range("2020-01-01", periods=30)
    rising = pd.Series(np.linspace(100, 130, 30), index=idx)
    assert len(T.episodes(rising, 0.05)) == 0
    with pytest.raises(ValueError):
        T.episodes(rising, 0.0)
    with pytest.raises(ValueError):
        T.episodes(rising, 1.0)


def test_episodes_invariants_on_random_walk():
    s = _random_walk(2000, seed=11, mu=0.0, sigma=0.015)
    n_prev = None
    for thr in (0.05, 0.10, 0.20):
        ep = T.episodes(s, thr)
        assert (ep["depth"] <= -thr).all()
        assert (ep["trough_date"] >= ep["peak_date"]).all()
        rec = ep["recovery_date"].dropna()
        assert (rec >= ep.loc[rec.index, "trough_date"]).all()
        assert (ep["days_to_trough"] >= 0).all()
        assert ep["peak_date"].is_monotonic_increasing
        # 미회복은 마지막 에피소드(들)에만 가능
        unrec = ep["recovery_date"].isna().to_numpy()
        if unrec.any():
            assert unrec[np.argmax(unrec):].all()
        # 회복일은 원래 고점 이후 첫 신고점이어야 한다
        for row in ep.itertuples(index=False):
            if pd.notna(row.recovery_date):
                seg = s.loc[row.peak_date: row.recovery_date]
                assert seg.iloc[-1] > seg.iloc[0] * (1 - 1e-12)
                assert (seg.iloc[1:-1] <= s.loc[: row.peak_date].max()).all()
        if n_prev is not None:
            assert len(ep) <= n_prev
        n_prev = len(ep)


def test_independent_blocks():
    assert T.independent_blocks(100, 20) == 5
    assert T.independent_blocks(19, 20) == 0
    assert T.independent_blocks(8458, 20) == 422       # VALIDATION.md §0 의 숫자
    assert T.independent_blocks(8458, 60) == 140
    with pytest.raises(ValueError):
        T.independent_blocks(100, 0)
    with pytest.raises(ValueError):
        T.independent_blocks(-1, 5)


# ------------------------------------------------------------------
# 실제 SPY (캐시가 있을 때만)
# ------------------------------------------------------------------
_CLOSE_CSV = Path(DATA_DIR) / "close.csv"


@pytest.mark.skipif(not _CLOSE_CSV.exists(), reason="data/close.csv 없음 — 캐시 구축 후 실행")
def test_episodes_real_spy_counts():
    close = pd.read_csv(_CLOSE_CSV, index_col=0, parse_dates=True)
    assert "SPY" in close.columns
    spy = close["SPY"].dropna()
    counts, counts_nosplit = {}, {}
    for thr in (0.05, 0.10, 0.20):
        ep = T.episodes(spy, thr)
        counts[thr] = len(ep)
        assert (ep["depth"] <= -thr).all()
        ep_ns = T.episodes(spy, thr, split=False)
        counts_nosplit[thr] = len(ep_ns)
        assert (ep_ns["depth"] <= -thr).all()
        # 분할 없는 표는 분할 표의 부분집합 구조: 에피소드 수는 적거나 같고 회복일 집합은 같다
        assert len(ep_ns) <= len(ep)
        assert set(ep_ns["recovery_date"].dropna()) == set(ep["recovery_date"].dropna())
    print(f"\n[real SPY {spy.index[0].date()}~{spy.index[-1].date()}, {len(spy)} days] "
          f"split rule (contract default) >=5%: {counts[0.05]}, >=10%: {counts[0.10]}, >=20%: {counts[0.20]} | "
          f"no-split (audit rule) >=5%: {counts_nosplit[0.05]}, >=10%: {counts_nosplit[0.10]}, "
          f">=20%: {counts_nosplit[0.20]} (audit reference: 36 / 12 / 4)")
    assert counts[0.20] <= counts[0.10] <= counts[0.05]
    # VALIDATION.md §2 의 감사 실측치는 분할 없는 규칙으로 정확히 재현된다 (1993-01-29 ~ 2026-09-04 기준)
    if spy.index[0] == pd.Timestamp("1993-01-29") and spy.index[-1] >= pd.Timestamp("2026-09-04"):
        assert (counts_nosplit[0.05], counts_nosplit[0.10], counts_nosplit[0.20]) == (36, 12, 4)
    ep20 = T.episodes(spy, 0.20)
    # 2020 코로나 급락(-30% 이상)과 2008 금융위기(-50% 이상)는 반드시 잡혀야 한다
    assert ((ep20["trough_date"].dt.year == 2020) & (ep20["depth"] <= -0.30)).any()
    assert ((ep20["trough_date"].dt.year.isin([2008, 2009])) & (ep20["depth"] <= -0.40)).any()
    tg = T.make_targets(spy)
    br = T.base_rates(tg)
    print("base rates:", br.round(3).to_dict())
    # VALIDATION.md §0: 일 54.1% · 20일 65.4% · 60일 72.0% (±2%p 허용)
    assert abs(br["y_sign_20"] - 0.654) < 0.02
    assert abs(br["y_sign_60"] - 0.720) < 0.02
