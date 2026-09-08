# -*- coding: utf-8 -*-
"""mrl/scenarios.py 검증 (ARCHITECTURE_PHASE3.md §7·§13).

실행: 프로젝트 루트에서  python -m pytest tests/test_scenarios.py -q

* 규칙(풀링·회색·구간 산술·모든 행의 n/n_eff)은 **합성 자료**로 확인한다 — 규칙이 자료에 기대지 않는다는 뜻.
* 실캐시 확인(에피소드 33/11/4, 시장 내재 범위 포함률, 설계 단계 구간 표)은 전부 **하드컷 2024-08-30**
  (`HOLDOUT_START` 미만) 입력으로만 돌린다. 홀드아웃은 이 파일에서 한 번도 채점되지 않는다
  (`test_holdout_is_never_scored` 가 프레임 최댓값을 직접 검사한다).
* 블록 부트스트랩은 seed 0 결정론. 테스트에서는 속도를 위해 n_boot 을 줄여 부르되(규칙과 무관),
  결정론·재현성은 같은 인자로 두 번 불러 비트 동일성으로 확인한다.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
import pytest

from mrl import scenarios as S
from mrl.calibrate import wilson
from mrl.config import HOLDOUT_START, P2, SCENARIO_P3
from mrl.data import load_cache
from mrl.targets import episodes as targets_episodes
from mrl.targets import make_targets

HOLDOUT = pd.Timestamp(HOLDOUT_START)
BAD_TOKEN = re.compile(r"\b(undefined|NaN|nan|None|null|NaT)\b")
FAST = dict(n_boot=200)                     # 규칙 검증용(구간 폭은 seed 로 결정론)

# 설계 단계 실측 (ARCHITECTURE_PHASE3.md §7; 하드컷 2024-08-30)
SPEC_BIN_N = (1422, 1137, 869, 620, 453, 316, 353, 101, 162)
SPEC_COVERAGE_VIX = {"1s": 0.838, "80": 0.936, "90": 0.973, "tail": 0.024}


# ------------------------------------------------------------------
# 합성 자료
# ------------------------------------------------------------------
def _synth_bins(counts, *, seed: int = 0, rates=None):
    """구간별 원하는 개수를 정확히 갖는 (p, y, fwd_ret, fwd_maxdd) — 확률은 각 구간의 중점."""
    bins = list(SCENARIO_P3["bins"])
    mids = [(bins[k] + bins[k + 1]) / 2.0 for k in range(len(bins) - 1)]
    rng = np.random.default_rng(seed)
    p, y = [], []
    for k, c in enumerate(counts):
        rate = mids[k] if rates is None else rates[k]
        p.extend([mids[k]] * c)
        n_pos = int(round(rate * c))
        lab = np.zeros(c)
        lab[:n_pos] = 1.0
        rng.shuffle(lab)
        y.extend(lab.tolist())
    n = len(p)
    idx = pd.bdate_range("2003-01-02", periods=n)
    ret = pd.Series(rng.normal(0.01, 0.05, n), index=idx)
    mdd = pd.Series(-np.abs(rng.normal(0.03, 0.03, n)), index=idx)
    return pd.Series(p, index=idx), pd.Series(y, index=idx), ret, mdd


def _synth_close(segments, start="2000-01-03"):
    """구간별 (세션 수, 종료 배율) 목록 → 로그선형 종가 (에피소드 검사용, 결정적)."""
    vals = [100.0]
    for n_sess, mult in segments:
        step = mult ** (1.0 / n_sess)
        for _ in range(n_sess):
            vals.append(vals[-1] * step)
    idx = pd.bdate_range(start, periods=len(vals))
    return pd.Series(vals, index=idx, name="close")


# ------------------------------------------------------------------
# 실캐시 픽스처 (전부 하드컷)
# ------------------------------------------------------------------
@pytest.fixture(scope="module")
def real():
    bundle = load_cache()
    spy = bundle.close["SPY"].dropna()
    close = spy.loc[spy.index < HOLDOUT]                       # 하드컷
    vix = bundle.close["^VIX"].reindex(close.index).ffill(limit=P2["vix_ffill_limit"])
    tg = make_targets(close)
    oos_path = "results/calib_p2_walkforward.csv"
    try:
        oos = pd.read_csv(oos_path, index_col=0, parse_dates=True)
    except FileNotFoundError:                                   # 아직 Phase 2 산출물이 없으면 확률 표는 건너뛴다
        oos = None
    if oos is not None:
        oos = oos.loc[oos.index < HOLDOUT]
    return {"close": close, "vix": vix, "targets": tg, "oos": oos}


# ------------------------------------------------------------------
# 1. 풀링 규칙
# ------------------------------------------------------------------
def test_pool_bins_matches_prereg_rule_on_design_stage_counts():
    """설계 단계 실측 개수에서 [.25,.30)(n_eff 15.8)이 위 구간들과 합쳐져 [.25, 1] 한 행이 된다."""
    groups = S.pool_bins(SPEC_BIN_N)
    assert groups == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 8)]
    pooled_n = sum(SPEC_BIN_N[5:])
    assert pooled_n == 932 and pooled_n / 20 == pytest.approx(46.6)


def test_pooling_merges_thin_bin_upward_until_n_eff_20():
    counts = [400, 400, 400, 400, 400, 320, 360, 100, 160]      # n_eff 20,20,20,20,20,16,18,5,8
    p, y, ret, mdd = _synth_bins(counts)
    tbl = S.bin_table(p, y, ret, mdd, **FAST)
    assert len(tbl) == 6
    assert list(tbl["n"]) == [400, 400, 400, 400, 400, 940]
    top = tbl.iloc[-1]
    assert bool(top["pooled"]) and int(top["n_bins"]) == 4
    assert top["bin_lo"] == pytest.approx(0.25) and top["bin_hi"] == pytest.approx(1.0)
    assert top["n_eff"] == pytest.approx(47.0)
    # 표시되는 행 중 n_eff < 20 은 없다 (풀링 전 16/18/5/8 은 단독으로 살아남지 못한다)
    assert (tbl["n_eff"] >= SCENARIO_P3["min_n_eff"]).all()
    assert not tbl["grey"].any()


def test_pooling_absorbs_unfinished_tail_into_previous_group():
    counts = [400, 400, 100]                                    # 마지막 꼬리 n_eff 5 → 위 그룹에 흡수
    p, y, ret, mdd = _synth_bins(counts + [0] * 6)
    tbl = S.bin_table(p, y, ret, mdd, **FAST)
    assert len(tbl) == 2
    assert list(tbl["n"]) == [400, 500]
    assert tbl.iloc[-1]["bin_hi"] == pytest.approx(1.0)         # 빈 구간까지 흡수해 경계가 닫힌다


def test_pooling_whole_table_when_everything_is_thin_and_marks_it_grey():
    counts = [40, 40, 40] + [0] * 6                             # 전체 n_eff 6
    p, y, ret, mdd = _synth_bins(counts)
    tbl = S.bin_table(p, y, ret, mdd, **FAST)
    assert len(tbl) == 1
    assert bool(tbl.iloc[0]["grey"]) and tbl.iloc[0]["n_eff"] == pytest.approx(6.0)


def test_pool_bins_is_monotone_in_min_n_eff():
    counts = [400, 400, 400, 400, 400, 320, 360, 100, 160]
    assert len(S.pool_bins(counts, min_n_eff=5)) >= len(S.pool_bins(counts, min_n_eff=20))
    assert len(S.pool_bins(counts, min_n_eff=100)) == 1


def test_bin_assignment_matches_calibrate_reliability_table():
    """구간 배정 규약([lo,hi), 마지막만 닫힘)이 Phase 2 신뢰도 표와 같아야 라이브 열이 붙는다."""
    from mrl.calibrate import reliability_table
    counts = [400, 400, 400, 400, 400, 320, 360, 100, 160]
    p, y, ret, mdd = _synth_bins(counts)
    rel = reliability_table(p, y)
    assert list(rel["n"]) == counts
    tbl = S.bin_table(p, y, ret, mdd, **FAST)
    assert int(tbl["n"].sum()) == int(rel["n"].sum())


# ------------------------------------------------------------------
# 2. 회색 규칙 · 모든 행의 n/n_eff/구간
# ------------------------------------------------------------------
def test_state_table_greys_thin_cell_and_keeps_thick_ones():
    idx = pd.bdate_range("2003-01-02", periods=1000)
    rng = np.random.default_rng(0)
    state = pd.Series(["normal"] * 700 + ["caution"] * 200 + ["reduce"] * 100, index=idx)
    y = pd.Series(rng.integers(0, 2, 1000).astype(float), index=idx)
    ret = pd.Series(rng.normal(0.01, 0.05, 1000), index=idx)
    mdd = pd.Series(-np.abs(rng.normal(0.03, 0.03, 1000)), index=idx)
    tbl = S.state_table(state, y, ret, mdd, **FAST)
    assert list(tbl["state"]) == ["normal", "caution", "reduce"]
    assert list(tbl["n"]) == [700, 200, 100]
    assert list(tbl["n_eff"]) == [35.0, 10.0, 5.0]
    assert list(tbl["grey"]) == [False, True, True]             # n_eff 10·5 < 20 → 단독 표시 금지


def test_every_row_has_n_n_eff_and_interval():
    counts = [400, 400, 400, 400, 400, 320, 360, 100, 160]
    p, y, ret, mdd = _synth_bins(counts)
    tbl = S.bin_table(p, y, ret, mdd, **FAST)
    for col in ("n", "n_eff", "obs", "wilson_lo", "wilson_hi", "ret_p50_lo", "ret_p50_hi", "grey"):
        assert col in tbl.columns
        assert tbl[col].notna().all(), col
    assert (tbl["n"] > 0).all()
    assert np.allclose(tbl["n_eff"].to_numpy(), tbl["n"].to_numpy() / SCENARIO_P3["n_eff_div"])
    S.assert_displayable(tbl, name="bin_table")

    idx = pd.bdate_range("2003-01-02", periods=600)
    rng = np.random.default_rng(1)
    st = S.state_table(pd.Series(["normal"] * 400 + ["caution"] * 150 + ["reduce"] * 50, index=idx),
                       pd.Series(rng.integers(0, 2, 600).astype(float), index=idx),
                       pd.Series(rng.normal(0, 0.04, 600), index=idx),
                       pd.Series(-np.abs(rng.normal(0.03, 0.02, 600)), index=idx), **FAST)
    for col in ("n", "n_eff", "wilson_lo", "wilson_hi"):
        assert st[col].notna().all(), col
    S.assert_displayable(st, name="state_table")


def test_bin_table_accepts_positional_arrays_and_rejects_ragged_input():
    counts = [400, 400, 400, 400, 400, 320, 360, 100, 160]
    p, y, ret, mdd = _synth_bins(counts)
    tbl = S.bin_table(p.to_numpy(), y.to_numpy(), ret.to_numpy(), mdd.to_numpy(), **FAST)
    assert list(tbl["n"]) == [400, 400, 400, 400, 400, 940]
    assert not np.isfinite(tbl["share_ep10_start"]).any()       # 날짜 인덱스가 없으면 계산하지 않는다
    with pytest.raises(ValueError):
        S.bin_table(p.to_numpy()[:-1], y.to_numpy(), ret.to_numpy(), mdd.to_numpy(), **FAST)
    with pytest.raises(ValueError):
        S.bin_table(p, y * np.nan, ret, mdd, **FAST)            # 라벨이 하나도 없으면 조용히 빈 표를 주지 않는다


def test_live_column_keeps_backtest_rows_and_flags_calibration_drift():
    counts = [400, 400, 400, 400, 400, 320, 360, 100, 160]
    p, y, ret, mdd = _synth_bins(counts)
    tbl = S.bin_table(p, y, ret, mdd, **FAST)
    live_idx = pd.bdate_range("2026-01-05", periods=200)
    live_p = pd.Series([0.04] * 200, index=live_idx)            # 전부 첫 구간
    live_y = pd.Series([1.0] * 200, index=live_idx)             # 관측 100% → 백테스트 구간 밖
    out = S.live_column(tbl, live_p, live_y)
    assert len(out) == len(tbl) and list(out["bin_lo"]) == list(tbl["bin_lo"])
    assert int(out.iloc[0]["live_n"]) == 200 and out.iloc[0]["live_n_eff"] == pytest.approx(10.0)
    assert out.iloc[0]["live_obs"] == pytest.approx(1.0)
    assert bool(out.iloc[0]["live_grey"]) and bool(out.iloc[0]["calib_drift"])
    assert int(out.iloc[-1]["live_n"]) == 0 and not bool(out.iloc[-1]["calib_drift"])
    # 라이브가 백테스트 구간 안이면 플래그가 서지 않는다
    calm_y = pd.Series(([1.0] * 10 + [0.0] * 90) * 2, index=live_idx)
    out2 = S.live_column(tbl, live_p, calm_y)
    assert not bool(out2.iloc[0]["calib_drift"])


def test_assert_displayable_rejects_a_table_without_counts():
    counts = [400, 400, 400, 400, 400, 320, 360, 100, 160]
    p, y, ret, mdd = _synth_bins(counts)
    tbl = S.bin_table(p, y, ret, mdd, **FAST)
    with pytest.raises(ValueError):
        S.assert_displayable(tbl.drop(columns=["n_eff"]), name="bin_table")
    broken = tbl.copy()
    broken.loc[broken.index[0], "wilson_lo"] = np.nan
    with pytest.raises(ValueError):
        S.assert_displayable(broken, name="bin_table")


# ------------------------------------------------------------------
# 3. 구간 산술 (Wilson 은 n/20, 중앙값은 블록 부트스트랩)
# ------------------------------------------------------------------
def test_wilson_uses_n_eff_not_n():
    counts = [400] + [0] * 8
    p, y, ret, mdd = _synth_bins(counts, rates=[0.25] + [0] * 8)
    tbl = S.bin_table(p, y, ret, mdd, **FAST)
    row = tbl.iloc[0]
    assert int(row["n"]) == 400 and row["n_eff"] == pytest.approx(20.0)
    obs = float(row["obs"])
    lo_eff, hi_eff = wilson(obs, 20.0)
    lo_raw, hi_raw = wilson(obs, 400.0)
    assert (row["wilson_lo"], row["wilson_hi"]) == pytest.approx((lo_eff, hi_eff))
    assert not math.isclose(row["wilson_lo"], lo_raw, abs_tol=1e-6)     # 겹치는 창을 독립으로 세지 않는다
    assert hi_eff - lo_eff > hi_raw - lo_raw


def test_episode_probability_intervals_are_wilson_on_episode_counts():
    lo, hi = wilson(11 / 33, 33.0)
    assert (lo, hi) == pytest.approx((0.1975, 0.5039), abs=5e-4)        # §7 C '11/33 = 0.33 (0.20~0.50)'
    lo2, hi2 = wilson(4 / 11, 11.0)
    assert (lo2, hi2) == pytest.approx((0.1517, 0.6462), abs=5e-4)      # '4/11 (0.15~0.65)'


def test_block_bootstrap_quantile_is_deterministic_and_brackets_the_median():
    rng = np.random.default_rng(7)
    v = pd.Series(rng.normal(0.01, 0.05, 2000))
    a = S.block_bootstrap_quantile(v, 0.5, n_boot=500)
    b = S.block_bootstrap_quantile(v, 0.5, n_boot=500)
    assert a == b                                                       # seed 0 결정론(비트 동일)
    lo, hi = a
    assert lo < float(v.median()) < hi
    wide = S.block_bootstrap_quantile(v.iloc[:200], 0.5, n_boot=500)
    assert (wide[1] - wide[0]) > (hi - lo)                              # 표본이 작으면 구간이 넓다
    assert S.block_bootstrap_quantile(pd.Series([np.nan, np.nan]), 0.5, n_boot=10) == pytest.approx(
        (float("nan"), float("nan")), nan_ok=True)
    with pytest.raises(ValueError):
        S.block_bootstrap_quantile(v, 0.5, block=0)


def test_bin_table_is_deterministic():
    counts = [400, 400, 400, 400, 400, 320, 360, 100, 160]
    p, y, ret, mdd = _synth_bins(counts)
    a = S.bin_table(p, y, ret, mdd, **FAST)
    b = S.bin_table(p, y, ret, mdd, **FAST)
    pd.testing.assert_frame_equal(a, b)


# ------------------------------------------------------------------
# 4. 에피소드 조건부
# ------------------------------------------------------------------
def test_episode_table_counts_split_false_and_includes_in_progress_episode():
    """진행 중(미회복) 에피소드도 세고, split=False 는 한 underwater 구간을 한 번만 센다."""
    close = _synth_close([(60, 1.20), (30, 0.90), (40, 1.35),      # −10% 에피소드 → 회복
                          (20, 0.97), (20, 1.10),                  # 얕은 조정(−3%): 에피소드 아님
                          (40, 0.80), (10, 1.05), (30, 0.85)])     # 큰 하락 안에서 반등 후 재하락 → split=False 는 1건(미회복)
    tbl = S.episode_table(close, 0.05)
    assert len(tbl) == 2
    assert bool(tbl["recovery_date"].isna().iloc[-1])              # 마지막은 진행 중
    ref = targets_episodes(close, 0.05, split=False)
    assert len(tbl) == len(ref)
    split_true = targets_episodes(close, 0.05, split=True)
    assert len(split_true) > len(ref)                              # split=True 는 조각으로 센다(규약 차이)
    assert (tbl["breach_date"] >= tbl["peak_date"]).all()
    assert (tbl["breach_date"] <= tbl["trough_date"]).all()
    assert (tbl["depth"] < 0).all()


def test_episode_conditionals_shape_and_extra_loss_definition():
    close = _synth_close([(60, 1.20), (30, 0.90), (40, 1.35), (40, 0.80), (30, 1.40)])
    out = S.episode_conditionals(close)
    assert out["n"] == 2
    for key in ("p_ge10_given5", "p_ge20_given5", "p_ge15_given10", "p_ge20_given10"):
        c = out[key]
        assert set(("k", "n", "p", "lo", "hi", "n_eff")) <= set(c)
        assert c["n_eff"] == float(c["n"])                          # 에피소드 수가 곧 n_eff
        if c["n"]:
            assert 0.0 <= c["lo"] <= c["p"] <= c["hi"] <= 1.0
    # 추가 손실 = 저점/(고점×(1−5%)) − 1 (돌파 '수준' 기준)
    depth = out["table"]["depth"].to_numpy(dtype=float)
    expect = (1.0 + depth) / 0.95 - 1.0
    assert out["table"]["extra_loss"].to_numpy(dtype=float) == pytest.approx(expect)
    assert out["per_year"] > 0 and sum(out["by_year"].values()) == out["n"]


def test_episode_conditionals_end_excludes_later_peaks():
    """`end` 하드컷은 홀드아웃 고점을 표에서 제외한다 (§7 C '고점 < 2024-09-01 만')."""
    n_before = 1000
    close = _synth_close([(n_before, 2.0), (30, 0.85), (40, 1.30)], start="2018-01-01")
    end = close.index[n_before]                                    # 이 뒤의 하락은 잘려야 한다
    full = S.episode_conditionals(close)
    cut = S.episode_conditionals(close, end=end)
    assert full["n"] == 1 and cut["n"] == 0
    assert pd.Timestamp(cut["end"]) <= pd.Timestamp(end)


def test_episode_starts_returns_breach_dates_and_feeds_bin_table():
    close = _synth_close([(60, 1.20), (30, 0.85), (60, 1.40)])
    starts = S.episode_starts(close, 0.10)
    assert isinstance(starts, pd.DatetimeIndex) and len(starts) == 1
    tbl10 = S.episode_table(close, 0.10)
    assert starts[0] == tbl10["breach_date"].iloc[0]
    # (t, t+h] 창 규칙: 돌파 20세션 전 행은 True, 21세션 전 행은 False
    idx = close.index
    pos = int(idx.get_loc(starts[0]))
    flags = S._start_flags(idx, starts, 20)
    assert flags[pos - 1] and flags[pos - 20] and not flags[pos - 21] and not flags[pos]


def test_current_drawdown_branches():
    close = _synth_close([(60, 1.30), (20, 0.88), (10, 1.02)])
    dd = S.current_drawdown(close, close.index[-1])
    assert dd["dd_from_ath"] < 0 and dd["min_dd_from_ath"] <= dd["dd_from_ath"]
    assert dd["breached_5"] and dd["breached_10"]
    assert dd["sessions_since_breach"] is not None and dd["sessions_since_breach"] >= 0
    top = S.current_drawdown(close, close.index[59])
    assert top["dd_from_ath"] == pytest.approx(0.0, abs=1e-12) and not top["breached_5"]
    assert top["sessions_since_breach"] is None


# ------------------------------------------------------------------
# 5. 시장 내재 / HAR 20세션 범위
# ------------------------------------------------------------------
def test_implied_range_matches_spec_examples():
    """§7 D: VIX 15 → 1σ ±4.2%, 80% ±5.4% ; VIX 14.53 → ±4.1 / ±5.2 / ±6.7%."""
    r = S.implied_range(100.0, 15.0)
    assert r["s"] == pytest.approx(0.15 * math.sqrt(20 / 252))
    assert 100 * r["pct"]["1s"] == pytest.approx(4.2, abs=0.05)
    assert 100 * r["pct"]["80"] == pytest.approx(5.4, abs=0.05)
    r2 = S.implied_range(100.0, 14.53)
    assert [round(100 * r2["pct"][k], 1) for k in ("1s", "80", "90")] == [4.1, 5.2, 6.7]
    lo, hi = r2["1s"]
    assert lo == pytest.approx(100.0 * math.exp(-r2["s"])) and hi == pytest.approx(100.0 * math.exp(r2["s"]))
    assert lo < 100.0 < hi
    for k in ("1s", "80", "90"):
        assert r2[k][0] < r2[k][1]
    assert r2["90"][1] - r2["90"][0] > r2["80"][1] - r2["80"][0] > r2["1s"][1] - r2["1s"][0]


def test_har_range_uses_the_same_geometry():
    r = S.har_range(100.0, 0.20)
    assert r["s"] == pytest.approx(0.20 * math.sqrt(20 / 252))
    assert r["src"] == "har" and r["ok"]
    assert S.implied_range(100.0, 20.0)["s"] == pytest.approx(r["s"])   # VIX 20 == HAR 0.20


def test_ranges_report_missing_input_instead_of_failing_silently():
    with pytest.warns(UserWarning):
        r = S.implied_range(100.0, float("nan"))
    assert r["ok"] is False and not np.isfinite(r["s"]) and not np.isfinite(r["1s"][0])
    with pytest.raises(ValueError):
        S.implied_range(-1.0, 15.0)
    with pytest.raises(ValueError):
        S.implied_range(100.0, 15.0, h=0)


# ------------------------------------------------------------------
# 6. 포함률 (계수 + 실캐시)
# ------------------------------------------------------------------
def test_coverage_counts_are_exact_on_constructed_z():
    """s 를 상수로 두고 실현 수익을 z 값에 맞춰 만들면 포함 개수를 손으로 셀 수 있다."""
    idx = pd.bdate_range("2003-01-02", periods=8)
    s = 0.05                                    # ln 수익 표준편차(20세션)
    vix = pd.Series([100.0 * s / math.sqrt(20 / 252)] * 8, index=idx)
    zs = np.array([0.0, 0.9, -0.9, 1.1, -1.1, 1.5, -1.5, -2.0])
    fwd = pd.Series(np.expm1(zs * s), index=idx)
    cov = S.coverage(fwd, vix, vix * np.nan)
    v = cov["vix"]
    assert v["n"] == 8 and v["n_eff"] == pytest.approx(0.4)
    assert v["hit_1s"] == pytest.approx(3 / 8)          # |z| <= 1     → 0, ±0.9
    assert v["hit_80"] == pytest.approx(5 / 8)          # |z| <= 1.2816 → + ±1.1
    assert v["hit_90"] == pytest.approx(7 / 8)          # |z| <= 1.645  → + ±1.5
    assert v["tail"] == pytest.approx(1 / 8)            # z < -1.645    → -2.0
    assert v["grey"] is True                             # n_eff 0.4 < 20
    assert not np.isfinite(cov["har"]["hit_1s"])         # 전부 결측이면 계산하지 않는다
    assert cov["har"]["n"] == 0


def test_coverage_labels_use_wilson_against_the_nominal_band():
    idx = pd.bdate_range("2003-01-02", periods=4000)
    rng = np.random.default_rng(0)
    s = 0.05
    vix_conservative = pd.Series([100.0 * (2 * s) / math.sqrt(20 / 252)] * 4000, index=idx)  # 두 배 넓은 밴드
    fwd = pd.Series(np.expm1(rng.normal(0, s, 4000)), index=idx)
    cov = S.coverage(fwd, vix_conservative, vix_conservative * np.nan)
    assert cov["vix"]["label_1s"] == "보수적"
    assert cov["vix"]["hit_1s"] > S.NOMINAL["1s"]
    assert S.range_label(0.68, 271.65, "1s") == "중심"
    assert S.range_label(0.30, 271.65, "1s") == "낙관적"
    assert S.range_label(float("nan"), 10.0, "1s") == S.MISSING


def test_coverage_on_real_cache_matches_spec_84_94_97(real, capsys):
    """실캐시(하드컷 2024-08-30, 2003~) 시장 내재 밴드 포함률 = 설계 단계 83.8 / 93.6 / 97.3%."""
    if real["oos"] is None:
        pytest.skip("results/calib_p2_walkforward.csv 없음")
    tg, close = real["targets"], real["close"]
    fwd = tg["fwd_ret_20"].reindex(real["oos"].index).dropna()
    assert fwd.index.max() < HOLDOUT
    lret = np.log(close).diff()
    rv20 = lret.rolling(20).std().shift(-20) * math.sqrt(252)
    cov = S.coverage(fwd, real["vix"], real["oos"]["har_fc_20"], rv20=rv20)
    v, h = cov["vix"], cov["har"]
    print(f"\n[coverage 2003-01-02~{fwd.index.max().date()} n={cov['n']} n_eff={cov['n_eff']:.1f}]")
    print(f"  VIX : 1s {v['hit_1s']:.4f} (명목 {S.NOMINAL['1s']:.4f}) · 80% {v['hit_80']:.4f} · "
          f"90% {v['hit_90']:.4f} · 왼꼬리 {v['tail']:.4f} · 라벨 {v['label_1s']}")
    print(f"  HAR : 1s {h['hit_1s']:.4f} · 80% {h['hit_80']:.4f} · 90% {h['hit_90']:.4f} · "
          f"왼꼬리 {h['tail']:.4f} · 라벨 {h['label_1s']}")
    print(f"  RV20/VIX 평균 {cov['rv_ratio']['rv_over_vix_mean']:.3f} · "
          f"중앙 {cov['rv_ratio']['rv_over_vix_median']:.3f}")
    assert cov["n"] == 5433 and cov["n_eff"] == pytest.approx(271.65)
    assert v["hit_1s"] == pytest.approx(SPEC_COVERAGE_VIX["1s"], abs=0.002)
    assert v["hit_80"] == pytest.approx(SPEC_COVERAGE_VIX["80"], abs=0.002)
    assert v["hit_90"] == pytest.approx(SPEC_COVERAGE_VIX["90"], abs=0.002)
    assert v["tail"] == pytest.approx(SPEC_COVERAGE_VIX["tail"], abs=0.002)
    assert v["label_1s"] == "보수적" and v["hit_1s"] > S.NOMINAL["1s"]          # '보수적: 과거 84% 포함'
    assert h["hit_1s"] < v["hit_1s"] and h["tail"] > v["tail"]                  # HAR 은 중심, VIX 는 보수적
    assert cov["rv_ratio"]["rv_over_vix_mean"] == pytest.approx(0.81, abs=0.02)  # §7 D 'RV20/VIX 0.81'
    assert not v["grey"] and not h["grey"]


# ------------------------------------------------------------------
# 7. 실캐시 — 에피소드·구간 표 (전부 하드컷)
# ------------------------------------------------------------------
def test_real_cache_episode_conditionals_match_spec(real, capsys):
    ec = S.episode_conditionals(real["close"], end=HOLDOUT_START)
    print(f"\n[episodes ≤2024-08-30] n={ec['n']} per_year={ec['per_year']:.3f} "
          f"P(≥10%|−5%)={ec['p_ge10_given5']['k']}/{ec['p_ge10_given5']['n']} "
          f"({ec['p_ge10_given5']['lo']:.3f}~{ec['p_ge10_given5']['hi']:.3f})")
    assert ec["n"] == 33                                            # §7 C '33건'
    assert (ec["p_ge10_given5"]["k"], ec["p_ge10_given5"]["n"]) == (11, 33)
    assert (ec["p_ge20_given5"]["k"], ec["p_ge20_given5"]["n"]) == (4, 33)
    assert (ec["p_ge15_given10"]["k"], ec["p_ge15_given10"]["n"]) == (6, 11)
    assert (ec["p_ge20_given10"]["k"], ec["p_ge20_given10"]["n"]) == (4, 11)
    assert (ec["p_ge10_given5"]["lo"], ec["p_ge10_given5"]["hi"]) == pytest.approx((0.20, 0.50), abs=0.005)
    assert (ec["p_ge20_given5"]["lo"], ec["p_ge20_given5"]["hi"]) == pytest.approx((0.05, 0.27), abs=0.005)
    assert (ec["p_ge20_given10"]["lo"], ec["p_ge20_given10"]["hi"]) == pytest.approx((0.15, 0.65), abs=0.005)
    q = ec["depth_q"]
    assert [round(100 * q[k], 1) for k in ("p90", "p75", "p50", "p25", "p10")] == [-5.5, -6.0, -7.6, -11.2, -23.5]
    x = ec["extra_loss_q"]
    assert [round(100 * x[k], 1) for k in ("p10", "p25", "p50")] == [-19.4, -6.5, -2.8]
    assert ec["n_extra_worse_than_base"] == 11                      # '11/33 이 −5% 보다 더'
    b2t, t2r = ec["breach_to_trough_q"], ec["trough_to_recovery_q"]
    assert (round(b2t["p10"]), round(b2t["p50"]), round(b2t["p90"])) == (0, 3, 106)
    assert round(ec["peak_to_trough_q"]["p50"]) == 22
    assert (round(t2r["p10"]), round(t2r["p50"]), round(t2r["p90"])) == (12, 32, 121)
    assert ec["per_year"] == pytest.approx(1.04, abs=0.01)
    assert ec["by_year"].get(1997) == 4 and sum(ec["by_year"].get(y, 0) for y in range(2001, 2007)) == 0
    assert pd.Timestamp(ec["end"]) < HOLDOUT


def test_real_cache_bin_table_matches_design_stage_table(real, capsys):
    if real["oos"] is None:
        pytest.skip("results/calib_p2_walkforward.csv 없음")
    oos, tg = real["oos"], real["targets"]
    starts = S.episode_starts(real["close"], 0.10, end=HOLDOUT_START)
    tbl = S.bin_table(oos["p_m3"], oos["y"], tg["fwd_ret_20"], tg["fwd_maxdd_20"], episodes10=starts, **FAST)
    print("\n[bin_table p_m3 2003~2024-08]\n" + tbl[["bin", "n", "n_eff", "obs", "wilson_lo", "wilson_hi",
                                                     "ret_p10", "ret_p50", "ret_p90", "mdd_p10",
                                                     "share_ep10_start", "grey"]].round(4).to_string())
    # 공식 p_m3 실측. 설계 단계 근사(SPEC_BIN_N)와는 경계 부근 ~9행이 다르게 배정될 뿐
    # 총 행수와 풀링 그룹은 동일하다 — §2/§6.2 '근사와 공식을 섞지 않는다'.
    assert sum(tbl.attrs["n_per_bin"]) == sum(SPEC_BIN_N) == 5433
    assert S.pool_bins(tbl.attrs["n_per_bin"]) == S.pool_bins(SPEC_BIN_N) \
        == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 8)]
    assert list(tbl["n"]) == [1431, 1131, 867, 620, 454, 930]          # 공식 실측
    assert [float(v) for v in tbl["n_eff"]] == pytest.approx([71.1, 56.9, 43.5, 31.0, 22.6, 46.6], abs=1.0)
    assert [float(v) for v in tbl["obs"]] == pytest.approx([0.076, 0.102, 0.116, 0.161, 0.236, 0.329], abs=0.005)
    assert [round(100 * float(v), 1) for v in tbl["ret_p10"][:5]] == [-3.0, -3.6, -3.9, -5.3, -6.1]
    assert [round(100 * float(v), 1) for v in tbl["ret_p50"][:5]] == [1.0, 1.2, 1.7, 2.1, 2.7]
    assert [round(100 * float(v), 1) for v in tbl["ret_p90"][:5]] == [3.3, 4.0, 4.9, 5.5, 6.6]
    assert [round(100 * float(v), 1) for v in tbl["mdd_p10"][:5]] == [-4.5, -5.0, -5.3, -6.7, -7.9]
    assert bool(tbl.iloc[-1]["pooled"]) and not tbl["grey"].any()
    assert tbl["share_ep10_start"].notna().all() and (tbl["share_ep10_start"] >= 0).all()
    S.assert_displayable(tbl, name="bin_table")


def test_real_cache_state_table_matches_design_stage_numbers(real, capsys):
    if real["oos"] is None:
        pytest.skip("results/calib_p2_walkforward.csv 없음")
    from mrl import decision
    oos, tg = real["oos"], real["targets"]
    states = decision.run(oos["p_m3"], oos["clim"])["state"]
    tbl = S.state_table(states, oos["y"], tg["fwd_ret_20"], tg["fwd_maxdd_20"], **FAST)
    print("\n[state_table M3 2003~2024-08]\n" + tbl.round(4).to_string())
    assert [round(float(v)) for v in tbl["n_eff"]] == [216, 43, 13]          # §7 B: 215 / 43 / 12
    # §7 B 설계 단계 근사 0.113/0.262/0.482 — 공식 실측 0.113/0.263/0.480
    assert [float(v) for v in tbl["obs"]] == pytest.approx([0.113, 0.262, 0.482], abs=0.005)
    assert [round(100 * float(v), 1) for v in tbl["ret_p10"]] == [-3.7, -5.2, -13.8]
    assert [round(100 * float(v), 1) for v in tbl["ret_p50"]] == [1.4, 2.4, 2.5]
    assert [round(100 * float(v), 1) for v in tbl["ret_p90"]] == [4.5, 7.3, 12.4]
    assert (tbl["ret_p50"] > 0).all()                                        # §7 B '중앙값은 모두 양수'
    assert list(tbl["grey"]) == [False, False, True]                         # reduce n_eff 12.75 < 20
    assert float(tbl.iloc[-1]["ret_p90"]) - float(tbl.iloc[-1]["ret_p10"]) > \
           float(tbl.iloc[0]["ret_p90"]) - float(tbl.iloc[0]["ret_p10"])     # 높은 위험 구간에서 분포가 넓어진다


def test_holdout_is_never_scored(real):
    assert real["close"].index.max() < HOLDOUT
    assert real["targets"].index.max() < HOLDOUT
    if real["oos"] is not None:
        assert real["oos"].index.max() < HOLDOUT
    ec = S.episode_conditionals(real["close"], end=HOLDOUT_START)
    assert pd.Timestamp(ec["table"]["peak_date"].max()) < HOLDOUT
    assert pd.Timestamp(ec["table"]["trough_date"].max()) < HOLDOUT


# ------------------------------------------------------------------
# 8. 오늘 문맥 (3줄 고정 순서 · 회색 행 단독 표시 금지)
# ------------------------------------------------------------------
def _tables(real):
    from mrl import decision
    oos, tg = real["oos"], real["targets"]
    starts = S.episode_starts(real["close"], 0.10, end=HOLDOUT_START)
    bins = S.bin_table(oos["p_m3"], oos["y"], tg["fwd_ret_20"], tg["fwd_maxdd_20"], episodes10=starts, **FAST)
    states = S.state_table(decision.run(oos["p_m3"], oos["clim"])["state"], oos["y"],
                           tg["fwd_ret_20"], tg["fwd_maxdd_20"], **FAST)
    fwd = tg["fwd_ret_20"].reindex(oos.index).dropna()
    cov = S.coverage(fwd, real["vix"], oos["har_fc_20"])
    ep = S.episode_conditionals(real["close"], end=HOLDOUT_START)
    return {"bins": bins, "states": states, "episodes": ep, "coverage": cov}


def test_today_context_three_lines_in_fixed_order(real, capsys):
    if real["oos"] is None:
        pytest.skip("results/calib_p2_walkforward.csv 없음")
    tables = _tables(real)
    dd = S.current_drawdown(real["close"], "2024-08-30")
    ctx = S.today_context(0.222, "normal", 14.53, 0.11, 560.0, tables, dd)
    print("\n[today_context]\n" + "\n".join(" " + line for line in ctx["lines"]))
    assert ctx["line_keys"] == list(S.LINE_ORDER) == ["market_range", "today_like", "if_episode"]
    l1, l2, l3 = ctx["lines"]
    assert l1.startswith("시장이 보는 20일 범위") and "80%" in l1 and "예상 변동성 기준" in l1
    assert l2.startswith("오늘 같은 날(확률대 20~25%)") and "과거 독립 23창 중 5창(11~44%)" in l2
    assert "20일 수익 10~90%: −6.1%~+6.6%" in l2 and "최대낙폭 나쁜 10% −7.9%" in l2
    assert l3.startswith("만약 −5% 에피소드가 시작되면") and "33번 중 11번(20~50%)" in l3
    assert "4번(5~27%)" in l3 and "절반은 −7.6% 안에서 멈춘다" in l3
    assert ctx["branch"] == "normal"
    assert ctx["range"]["headline"] in ("vix", "har")
    assert ctx["spread_sentence"] == S.SPREAD_SENTENCE and "넓어진다" in ctx["spread_sentence"]
    for line in ctx["lines"]:
        assert not BAD_TOKEN.search(line), line


def test_today_context_picks_a_different_line_per_drawdown_state(real):
    if real["oos"] is None:
        pytest.skip("results/calib_p2_walkforward.csv 없음")
    tables = _tables(real)
    seen = {}
    for asof in ("2024-08-30", "2018-10-29", "2020-03-12"):
        dd = S.current_drawdown(real["close"], asof)
        ctx = S.today_context(0.222, "normal", 20.0, 0.20, 300.0, tables, dd)
        seen[ctx["branch"]] = ctx["lines"][2]
    assert set(seen) == {"normal", "breached_5", "breached_10"}
    assert seen["normal"].startswith("만약")
    assert seen["breached_5"].startswith("지금 −5%")
    assert seen["breached_10"].startswith("지금 −10%")
    assert len({v for v in seen.values()}) == 3


def test_today_context_never_shows_a_thin_row_alone():
    """n_eff < 20 인 행은 카드에서 단독 표시 금지 — 문장이 그 행의 숫자를 쓰지 않는다."""
    thin = [{"bin": "[0.20, 0.25)", "bin_lo": 0.20, "bin_hi": 0.25, "pooled": False, "n_bins": 1,
             "n": 120, "n_eff": 6.0, "mean_p": 0.22, "obs": 0.5, "wilson_lo": 0.24, "wilson_hi": 0.76,
             "ret_p10": -0.2, "ret_p50": 0.01, "ret_p90": 0.2, "ret_p50_lo": -0.01, "ret_p50_hi": 0.03,
             "mdd_p10": -0.25, "mdd_p50": -0.03, "share_ep10_start": 0.1, "ep10_lo": 0.0, "ep10_hi": 0.4,
             "grey": True},
            {"bin": "[0.00, 0.20)", "bin_lo": 0.0, "bin_hi": 0.20, "pooled": True, "n_bins": 4,
             "n": 4000, "n_eff": 200.0, "mean_p": 0.10, "obs": 0.10, "wilson_lo": 0.06, "wilson_hi": 0.15,
             "ret_p10": -0.03, "ret_p50": 0.01, "ret_p90": 0.04, "ret_p50_lo": 0.005, "ret_p50_hi": 0.015,
             "mdd_p10": -0.05, "mdd_p50": -0.01, "share_ep10_start": 0.01, "ep10_lo": 0.0, "ep10_hi": 0.05,
             "grey": False}]
    ep = {"n": 33, "p_ge10_given5": {"k": 11, "n": 33, "p": 1 / 3, "lo": 0.2, "hi": 0.5},
          "p_ge20_given5": {"k": 4, "n": 33, "p": 0.12, "lo": 0.05, "hi": 0.27},
          "p_ge15_given10": {"k": 6, "n": 11, "p": 0.55, "lo": 0.28, "hi": 0.79},
          "p_ge20_given10": {"k": 4, "n": 11, "p": 0.36, "lo": 0.15, "hi": 0.65},
          "depth_q": {"p50": -0.076}, "extra_loss_q": {"p10": -0.194, "p50": -0.028},
          "breach_to_trough_q": {"p50": 3.0}, "trough_to_recovery_q": {"p50": 32.0}}
    dd = {"dd_from_ath": -0.01, "breached_5": False, "breached_10": False}
    ctx = S.today_context(0.22, "reduce", 15.0, 0.12, 100.0, {"bins": thin, "episodes": ep}, dd)
    line = ctx["lines"][1]
    assert ctx["bin_grey"] is True
    assert "표본이 얇아" in line and "6 < 20" in line
    assert "과거 독립" not in line and "50%" not in line          # 그 행의 관측 비율·구간을 쓰지 않는다
    assert any("단독 표시 금지" in n for n in ctx["notes"])
    for text in ctx["lines"]:
        assert not BAD_TOKEN.search(text), text
    # 두꺼운 행이면 정상 문장
    ctx2 = S.today_context(0.10, "normal", 15.0, 0.12, 100.0, {"bins": thin, "episodes": ep}, dd)
    assert ctx2["bin_grey"] is False and "과거 독립 200창" in ctx2["lines"][1]


def test_today_context_survives_missing_inputs_without_bad_tokens():
    dd = {"dd_from_ath": float("nan"), "breached_5": False, "breached_10": False}
    with pytest.warns(UserWarning):
        ctx = S.today_context(float("nan"), None, float("nan"), float("nan"), float("nan"), {}, dd)
    assert len(ctx["lines"]) == 3
    for line in ctx["lines"]:
        assert not BAD_TOKEN.search(line), line
        assert S.MISSING in line or "표본" in line or "쓸 수 없다" in line
    assert ctx["notes"]                                            # 조용히 넘어가지 않는다


def test_today_context_accepts_json_round_tripped_tables(real):
    """daily.py 는 summary_p3.json 에서 읽으므로 records(list[dict])도 받아야 한다."""
    if real["oos"] is None:
        pytest.skip("results/calib_p2_walkforward.csv 없음")
    tables = _tables(real)
    dd = S.current_drawdown(real["close"], "2024-08-30")
    a = S.today_context(0.222, "normal", 14.53, 0.11, 560.0, tables, dd)
    records = {"bins": tables["bins"].to_dict("records"), "states": tables["states"].to_dict("records"),
               "episodes": tables["episodes"], "coverage": tables["coverage"]}
    b = S.today_context(0.222, "normal", 14.53, 0.11, 560.0, records, dd)
    assert a["lines"] == b["lines"]
    assert a["branch"] == b["branch"]
