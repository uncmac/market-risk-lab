# -*- coding: utf-8 -*-
"""mrl/evaluate.py 검증 — 합성 replay(계약 열 전부) + 합성 SPY 로 채점 규약을 확인한다.

실행: 프로젝트 루트에서  python -m pytest tests/test_evaluate.py -q
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from mrl import evaluate as E
from mrl import targets as T
from mrl.config import HORIZONS, TONES, TONE_EXPOSURE, V0_SIGNALS


# ------------------------------------------------------------------
# 합성 데이터
# ------------------------------------------------------------------
def _close(n: int = 900, seed: int = 0, mu: float = 0.0003, sigma: float = 0.011) -> pd.Series:
    idx = pd.bdate_range("2014-01-02", periods=n)
    rng = np.random.default_rng(seed)
    px = 100.0 * np.exp(np.cumsum(rng.normal(mu, sigma, n)))
    return pd.Series(px, index=idx, name="SPY")


def make_replay(index: pd.DatetimeIndex, tones) -> pd.DataFrame:
    """ARCHITECTURE.md 의 replay 열 계약을 모두 갖춘 합성 replay."""
    tone = pd.Series(tones, index=index) if not isinstance(tones, pd.Series) else tones.reindex(index)
    rep = pd.DataFrame(index=index)
    for k in V0_SIGNALS:
        rep[f"state_{k}"] = "AMBER"
    for s in ("d", "w", "m"):
        rep[f"score_{s}"] = 0.0
        rep[f"trend_{s}"] = 0
        rep[f"overall_{s}"] = "AMBER"
    rep["tone"] = tone.astype(str)
    rep["verdict_ko"] = "방향이 뚜렷하지 않음"
    rep["n_watch_avail"] = 25
    rep["fg_avail"] = False
    rep["eod_avail"] = False
    rep["n_leaders"] = 2
    return rep


def _random_tones(index: pd.DatetimeIndex, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    # 며칠씩 이어지는 런으로 생성 (실제 판정처럼)
    out, i = [], 0
    while i < len(index):
        L = int(rng.integers(1, 15))
        out.extend([str(rng.choice(TONES))] * L)
        i += L
    return pd.Series(out[: len(index)], index=index)


@pytest.fixture(scope="module")
def synth():
    close = _close()
    targets = T.make_targets(close)
    ridx = close.index[260:820]                     # 대상 창이 실현된 구간
    replay = make_replay(ridx, _random_tones(ridx))
    return close, targets, replay


# ------------------------------------------------------------------
# directional_scorecard
# ------------------------------------------------------------------
def test_scorecard_shape_and_columns(synth):
    close, targets, replay = synth
    sc = E.directional_scorecard(replay, targets)
    assert list(sc.columns) == list(E.SCORECARD_COLUMNS)
    assert len(sc) == (len(TONES) + 1) * len(HORIZONS)
    assert list(sc["tone"].unique()) == list(TONES) + ["all"]
    assert sorted(sc["h"].unique()) == sorted(HORIZONS)


def test_scorecard_hit_semantics_bruteforce(synth):
    """hit: buy/hold/neutral → y_sign=1 예측, caution/reduce → 0 예측. baseline = 같은 행의 mean(y_sign)."""
    close, targets, replay = synth
    sc = E.directional_scorecard(replay, targets).set_index(["tone", "h"])
    joined = targets.loc[replay.index].copy()
    joined["tone"] = replay["tone"]
    for tone in list(TONES) + ["all"]:
        sub_t = joined if tone == "all" else joined[joined["tone"] == tone]
        for h in HORIZONS:
            sub = sub_t[sub_t[f"y_sign_{h}"].notna()]
            row = sc.loc[(tone, h)]
            assert row["n"] == len(sub)
            if len(sub) == 0:
                assert math.isnan(row["hit"])
                continue
            pred = sub["tone"].isin(E.UP_TONES).astype(float)
            assert math.isclose(row["hit"], (pred == sub[f"y_sign_{h}"]).mean())
            assert math.isclose(row["baseline"], sub[f"y_sign_{h}"].mean())
            assert math.isclose(row["edge"], row["hit"] - row["baseline"])
            r = sub[f"fwd_ret_{h}"]
            assert math.isclose(row["fwd_ret_mean"], r.mean())
            assert math.isclose(row["fwd_ret_median"], r.median())
            assert math.isclose(row["fwd_ret_p10"], r.quantile(0.10))
            assert math.isclose(row["fwd_ret_p90"], r.quantile(0.90))
            assert math.isclose(row["dd5_20_rate"], sub["y_dd5_20"].mean())
            assert row["hit_ci_lo"] <= row["hit"] + 1e-12 and row["hit"] - 1e-12 <= row["hit_ci_hi"]
            assert 0 < row["n_blocks"] <= row["n"]
            # 상승 예측 톤은 hit == baseline, 경고 톤은 hit == 1 - baseline
            if tone in E.UP_TONES:
                assert math.isclose(row["hit"], row["baseline"])
            elif tone in E.WARN_TONES:
                assert math.isclose(row["hit"], 1.0 - row["baseline"])


def test_scorecard_constant_tone_and_blocks(synth):
    close, targets, replay = synth
    idx = replay.index
    all_buy = make_replay(idx, ["buy"] * len(idx))
    sc = E.directional_scorecard(all_buy, targets).set_index(["tone", "h"])
    for h in HORIZONS:
        n = int(targets.loc[idx, f"y_sign_{h}"].notna().sum())
        assert sc.loc[("buy", h), "n"] == n
        assert math.isclose(sc.loc[("buy", h), "hit"], sc.loc[("buy", h), "baseline"])
        # 연속 구간이면 겹치지 않는 창 수 = ceil(n/h) (탐욕 선택)
        assert sc.loc[("buy", h), "n_blocks"] == math.ceil(n / h)
        assert sc.loc[("all", h), "n_blocks"] == math.ceil(n / h)
        assert sc.loc[("reduce", h), "n"] == 0 and math.isnan(sc.loc[("reduce", h), "hit"])
    all_reduce = make_replay(idx, ["reduce"] * len(idx))
    sc2 = E.directional_scorecard(all_reduce, targets).set_index(["tone", "h"])
    for h in HORIZONS:
        assert math.isclose(sc2.loc[("reduce", h), "hit"], 1.0 - sc2.loc[("reduce", h), "baseline"])


def test_scorecard_validation(synth):
    close, targets, replay = synth
    bad = replay.copy()
    bad.iloc[3, bad.columns.get_loc("tone")] = "panic"
    with pytest.raises(ValueError):
        E.directional_scorecard(bad, targets)
    with pytest.raises(ValueError):
        E.directional_scorecard(replay.drop(columns=["tone"]), targets)
    with pytest.raises(ValueError):
        E.directional_scorecard(replay, targets.drop(columns=["y_sign_20"]))
    # targets 에 없는 날짜가 섞이면 경고 후 제외
    ext = make_replay(replay.index.append(pd.bdate_range("2030-01-01", periods=3)), "hold")
    with pytest.warns(UserWarning):
        sc = E.directional_scorecard(ext, targets)
    assert sc.set_index(["tone", "h"]).loc[("all", 5), "n"] == int(targets.loc[replay.index, "y_sign_5"].notna().sum())


# ------------------------------------------------------------------
# episode_eval
# ------------------------------------------------------------------
def _planted_close() -> pd.Series:
    """완만한 상승 + 두 번의 -12% 낙폭(회복 포함) + 마지막에 낙폭 없이 끝나는 합성 종가."""
    idx = pd.bdate_range("2015-01-02", periods=700)
    px = np.full(700, 100.0)
    px = 100.0 * (1.0002 ** np.arange(700))              # 완만한 상승
    # 낙폭 1: 200 고점 → 10일간 -12% → 30일간 회복
    for k in range(1, 11):
        px[200 + k] = px[200] * (1 - 0.012 * k)
    for k in range(1, 31):
        px[210 + k] = px[210] * (1 + (px[200] * 1.01 / px[210] - 1) * k / 30)
    px[241:] = px[240] * (1.0002 ** np.arange(700 - 241))
    # 낙폭 2: 450 고점 → 10일간 -12% → 30일간 회복
    for k in range(1, 11):
        px[450 + k] = px[450] * (1 - 0.012 * k)
    for k in range(1, 31):
        px[460 + k] = px[460] * (1 + (px[450] * 1.01 / px[460] - 1) * k / 30)
    px[491:] = px[490] * (1.0002 ** np.arange(700 - 491))
    return pd.Series(px, index=idx, name="SPY")


def test_episode_eval_lead_and_miss():
    close = _planted_close()
    targets = T.make_targets(close)
    ep = T.episodes(close, 0.10)
    assert len(ep) == 2 and ep["days_to_trough"].tolist() == [10, 10]
    idx = close.index
    tones = pd.Series("buy", index=idx)
    p1 = idx.get_loc(ep.loc[0, "peak_date"])
    tr1 = idx.get_loc(ep.loc[0, "trough_date"])
    tones.iloc[p1 - 3: tr1 + 1] = "caution"                # 첫 에피소드: 고점 3일 전부터 저점까지 경고
    replay = make_replay(idx[100:], tones.iloc[100:])       # 두 번째 에피소드는 경고 없음
    table, summ = E.episode_eval(replay, ep, targets)
    assert list(table.columns) == list(T.EPISODE_COLUMNS) + [c for c in E.EPISODE_EVAL_COLUMNS
                                                             if c not in T.EPISODE_COLUMNS]
    r0, r1 = table.iloc[0], table.iloc[1]
    assert bool(r0["evaluable"]) and not bool(r0["missed"])
    assert r0["warn_date"] == idx[p1 - 3] and r0["lead_days"] == 3
    assert r0["warn_run_start"] == idx[p1 - 3] and r0["warn_run_age_at_peak"] == 3 and not bool(r0["lead_capped"])
    assert bool(r0["held_to_trough"]) and math.isclose(r0["warn_frac_to_trough"], 1.0)
    assert r0["tone_at_peak"] == "caution" and r0["tone_at_trough"] == "caution"
    assert bool(r1["evaluable"]) and bool(r1["missed"]) and pd.isna(r1["warn_date"])
    assert summ["n_episodes"] == 2 and summ["n_evaluable"] == 2
    assert summ["n_detected"] == 1 and summ["n_missed"] == 1 and math.isclose(summ["detection_rate"], 0.5)
    assert summ["median_lead_days"] == 3 and summ["n_lead_positive"] == 1 and summ["n_held_to_trough"] == 1
    assert summ["n_lead_capped"] == 0 and summ["median_lead_days_fresh"] == 3
    assert summ["lookback"] == 20
    # 고점 60일 전부터 계속 켜져 있던 경고: lead 는 lookback(20)에 걸리고 lead_capped 로 표시된다
    tones2 = pd.Series("buy", index=idx)
    tones2.iloc[p1 - 60: tr1 + 1] = "reduce"
    table2, summ2 = E.episode_eval(make_replay(idx[100:], tones2.iloc[100:]), ep, targets)
    q = table2.iloc[0]
    assert q["lead_days"] == 20 and bool(q["lead_capped"]) and q["warn_run_age_at_peak"] == 60
    assert q["warn_run_start"] == idx[p1 - 60] and q["warn_date"] == idx[p1 - 20]
    assert summ2["n_lead_capped"] == 1 and math.isnan(summ2["median_lead_days_fresh"])
    assert summ2["median_lead_days"] == 20
    # 경고 런 1개: 시작일(고점 3일 전) 기준 20일 내 -5% 있음 → 진짜 경보, 오경보 0
    assert summ["n_warn_runs"] == 1 and summ["n_true_alarms"] == 1 and summ["n_false_alarms"] == 0
    assert summ["n_tone_switches"] == 2 and math.isclose(summ["tone_switches_per_year"], 2 / (600 / 252))
    assert summ["median_warn_run_len"] == 14


def test_episode_eval_outside_replay_and_late_warning():
    close = _planted_close()
    targets = T.make_targets(close)
    ep = T.episodes(close, 0.10)
    idx = close.index
    p2 = idx.get_loc(ep.loc[1, "peak_date"])
    tones = pd.Series("hold", index=idx)
    tones.iloc[p2 + 4: p2 + 7] = "reduce"                   # 고점 4일 뒤 경고 (3일만)
    tones.iloc[300:305] = "caution"                         # 조용한 구간의 경고 → 오경보
    replay = make_replay(idx[300:], tones.iloc[300:])       # 첫 에피소드(고점 200)는 replay 밖
    table, summ = E.episode_eval(replay, ep, targets, lookback=10)
    r0, r1 = table.iloc[0], table.iloc[1]
    assert not bool(r0["evaluable"]) and pd.isna(r0["missed"]) and r0["tone_at_peak"] is None
    assert bool(r1["evaluable"]) and r1["lead_days"] == -4 and not bool(r1["held_to_trough"])
    assert math.isclose(r1["warn_frac_to_trough"], 3 / 7)
    assert r1["warn_run_start"] == idx[p2 + 4] and math.isnan(r1["warn_run_age_at_peak"]) and not bool(r1["lead_capped"])
    assert summ["n_evaluable"] == 1 and summ["n_detected"] == 1 and summ["n_lead_positive"] == 0
    assert summ["lookback"] == 10
    assert summ["n_warn_runs"] == 2 and summ["n_false_alarms"] == 1 and summ["n_true_alarms"] == 1
    assert math.isclose(summ["false_alarm_rate"], 0.5)
    assert math.isclose(summ["false_alarms_per_year"], 1 / (400 / 252))
    assert summ["median_run_len"] == 3.0 or summ["median_run_len"] >= 3.0


def test_episode_eval_null_baseline_and_lead_cap_edge():
    """무작위 순환 이동 기준선(null)·기저율 열이 요약에 있고 결정적이며, 창 첫날부터 켜진 경고는 lead_capped 로 표시된다."""
    close = _planted_close()
    targets = T.make_targets(close)
    ep = T.episodes(close, 0.10)
    idx = close.index
    p1 = idx.get_loc(ep.loc[0, "peak_date"])
    tr1 = idx.get_loc(ep.loc[0, "trough_date"])
    tones = pd.Series("buy", index=idx)
    tones.iloc[p1 - 20: tr1 + 1] = "caution"                # 정확히 탐색 창 첫날(peak-20)부터 켜진 경고 → 상한에 걸림
    replay = make_replay(idx[100:], tones.iloc[100:])
    table, summ = E.episode_eval(replay, ep, targets)
    r0 = table.iloc[0]
    assert r0["lead_days"] == 20 and bool(r0["lead_capped"]) and summ["n_lead_capped"] == 1
    assert math.isnan(summ["median_lead_days_fresh"])         # 새로 켜진 경고가 없다
    # 기저율 열: 임의의 날 뒤 20일 내 -5% 확률과 그 여집합
    p_dd = targets.loc[replay.index, "y_dd5_20"].dropna().mean()
    assert math.isclose(summ["true_alarm_share_baseline"], p_dd)
    assert math.isclose(summ["false_alarm_rate_baseline"], 1.0 - p_dd)
    assert math.isclose(summ["true_alarm_share"], 1.0 - summ["false_alarm_rate"])
    # null: 키·범위·결정성
    null = summ["null"]
    for k in ("n_shift", "detection_rate_mean", "detection_100_share", "median_lead_days_p5",
              "median_lead_days_p50", "median_lead_days_p95", "true_alarm_share_mean"):
        assert k in null, k
    assert null["n_shift"] == E.NULL_N_SHIFT
    assert 0.0 <= null["detection_rate_mean"] <= 1.0 and 0.0 <= null["detection_100_share"] <= 1.0
    assert null["median_lead_days_p5"] <= null["median_lead_days_p50"] <= null["median_lead_days_p95"] <= 20
    assert 0.0 <= null["true_alarm_share_mean"] <= 1.0
    _t2, summ2 = E.episode_eval(replay, ep, targets)
    assert summ2["null"] == null                            # 같은 seed → 같은 값
    # 경고 비중이 4%(21/600) 인 무작위 이동은 -20~+10일 창(31일)에서 대개 탐지에 실패한다 → 실제(100%) 와 구별된다
    assert null["detection_rate_mean"] < 0.9
    # 정보 없는 신호(경고 없음): 런이 없으면 진짜 경보 비중은 NaN, 탐지율은 0
    _t3, summ3 = E.episode_eval(make_replay(idx[100:], ["hold"] * len(idx[100:])), ep, targets)
    assert summ3["null"]["detection_rate_mean"] == 0.0 and math.isnan(summ3["null"]["true_alarm_share_mean"])
    assert math.isnan(summ3["true_alarm_share"]) and math.isclose(summ3["false_alarm_rate_baseline"], 1.0 - p_dd)


def test_episode_eval_empty_episodes(synth):
    close, targets, replay = synth
    table, summ = E.episode_eval(replay, T.episodes(close, 0.9), targets)
    assert len(table) == 0 and summ["n_episodes"] == 0 and math.isnan(summ["detection_rate"])
    with pytest.raises(ValueError):
        E.episode_eval(replay, pd.DataFrame({"peak_date": []}), targets)


# ------------------------------------------------------------------
# allocation_sim
# ------------------------------------------------------------------
def test_allocation_all_buy_equals_buy_and_hold(synth):
    close, targets, replay = synth
    idx = replay.index
    al = E.allocation_sim(make_replay(idx, ["buy"] * len(idx)), close)
    for k in ("cagr", "max_dd", "worst_month", "total_return", "ann_vol"):
        assert math.isclose(al[k], al[f"bh_{k}"]), k
    assert al["n_switches"] == 0 and al["switches_per_year"] == 0 and al["cost_total"] == 0
    assert al["avg_exposure"] == 1.0 and al["n_days"] == len(idx) - 1
    assert al["start"] == idx[0] and al["end"] == idx[-1]
    bh_total = close.loc[idx[-1]] / close.loc[idx[0]] - 1
    assert math.isclose(al["bh_total_return"], bh_total)
    assert math.isclose(al["bh_cagr"], (1 + bh_total) ** (252 / (len(idx) - 1)) - 1)
    s = al["series"]
    assert isinstance(s["equity"], pd.Series) and len(s["equity"]) == len(idx) - 1


def test_allocation_constant_reduce_scales_returns(synth):
    close, targets, replay = synth
    idx = replay.index
    al = E.allocation_sim(make_replay(idx, ["reduce"] * len(idx)), close, cost_bps=0)
    r = close.loc[idx].pct_change().iloc[1:]
    pd.testing.assert_series_equal(al["series"]["daily_ret"], 0.25 * r, check_names=False)
    assert math.isclose(al["avg_exposure"], 0.25)


def test_allocation_switch_cost_and_point_in_time():
    idx = pd.bdate_range("2020-01-01", periods=8)
    close = pd.Series([100, 102, 101, 103, 104, 102, 105, 106], index=idx, dtype=float)
    tones = ["buy", "buy", "buy", "reduce", "reduce", "reduce", "reduce", "hold"]
    al = E.allocation_sim(make_replay(idx, tones), close, cost_bps=5)
    r = close.pct_change()
    d = al["series"]["daily_ret"]
    # 3일차(인덱스 3) 종가에 reduce 확정: 그날 수익은 전날 비중 1.0 로, 비용 |1-0.25|·5bp 는 그날 부과
    assert math.isclose(d.iloc[2], (1 + 1.0 * r.iloc[3]) * (1 - 0.75 * 5e-4) - 1)
    # 4일차: 비중 0.25
    assert math.isclose(d.iloc[3], 0.25 * r.iloc[4])
    # 마지막 날 hold 로 바뀜: 그날 수익은 0.25 비중, 비용 부과
    assert math.isclose(d.iloc[-1], (1 + 0.25 * r.iloc[7]) * (1 - 0.75 * 5e-4) - 1)
    assert al["n_switches"] == 2 and math.isclose(al["cost_total"], 2 * 0.75 * 5e-4)
    assert math.isclose(al["switches_per_year"], 2 / (7 / 252))
    # 점 원칙: 마지막 날 톤만 바꾸면(비용 0) 수익률 시계열은 그대로
    t2 = tones[:-1] + ["reduce"]
    al2 = E.allocation_sim(make_replay(idx, t2), close, cost_bps=0)
    al1 = E.allocation_sim(make_replay(idx, tones), close, cost_bps=0)
    pd.testing.assert_series_equal(al1["series"]["daily_ret"], al2["series"]["daily_ret"])
    # 첫날 톤은 둘째 날 수익률에 적용된다
    t3 = ["reduce"] + tones[1:]
    al3 = E.allocation_sim(make_replay(idx, t3), close, cost_bps=0)
    assert math.isclose(al3["series"]["daily_ret"].iloc[0], 0.25 * r.iloc[1])
    assert math.isclose(al1["series"]["daily_ret"].iloc[0], 1.0 * r.iloc[1])


def test_allocation_maxdd_improvement_sign():
    """maxdd_improvement 는 '양수면 개선': 낙폭이 있는 종가에서 항상-reduce(25%) 배분은 보유보다 얕은 MaxDD 를 갖는다."""
    close = _planted_close()                                # -12% 낙폭 두 번 포함
    idx = close.index[100:]
    al = E.allocation_sim(make_replay(idx, ["reduce"] * len(idx)), close)
    assert al["max_dd"] < 0 and al["bh_max_dd"] < 0
    assert al["max_dd"] > al["bh_max_dd"]                    # 배분 쪽 낙폭이 얕다 (음수가 덜 음수)
    assert al["maxdd_improvement"] > 0
    assert math.isclose(al["maxdd_improvement"], al["max_dd"] - al["bh_max_dd"])
    assert math.isclose(al["maxdd_improvement"], abs(al["bh_max_dd"]) - abs(al["max_dd"]))
    # 항상-buy 는 보유와 같으므로 개선 0
    al_b = E.allocation_sim(make_replay(idx, ["buy"] * len(idx)), close)
    assert math.isclose(al_b["maxdd_improvement"], 0.0, abs_tol=1e-12)


def test_allocation_validation(synth):
    close, targets, replay = synth
    with pytest.raises(ValueError):
        E.allocation_sim(replay, close, exposure={"buy": 1.0})
    with pytest.raises(ValueError):
        E.allocation_sim(replay, close, cost_bps=-1)
    with pytest.raises(ValueError):
        E.allocation_sim(replay.iloc[:1], close)
    assert E.allocation_sim(replay, close)["exposure_map"] == TONE_EXPOSURE


# ------------------------------------------------------------------
# brier · block bootstrap
# ------------------------------------------------------------------
def test_brier_and_skill():
    y = pd.Series([1.0, 0.0, 1.0, 1.0, 0.0, np.nan], index=pd.bdate_range("2020-01-01", periods=6))
    assert E.brier(y, y) == 0.0
    assert math.isclose(E.brier(0.5, y), 0.25)
    p = pd.Series([0.8, 0.2, 0.6, 0.9, 0.1, 0.5], index=y.index)
    expect = np.mean([(0.8 - 1) ** 2, 0.2 ** 2, 0.4 ** 2, 0.1 ** 2, 0.1 ** 2])
    assert math.isclose(E.brier(p, y), expect)
    assert math.isclose(E.brier_skill(y, y, 0.5), 1.0)
    assert math.isclose(E.brier_skill(0.5, y, 0.5), 0.0)
    assert math.isclose(E.brier_skill(p, y, 0.5), 1 - expect / 0.25)
    with pytest.raises(ValueError):
        E.brier(pd.Series([1.2] * 6, index=y.index), y)
    with pytest.raises(ValueError):
        E.brier_skill(p, y, y)                         # 기준이 완벽하면 정의 불가


def test_block_bootstrap_ci():
    idx = pd.bdate_range("2020-01-01", periods=500)
    const = pd.Series(0.3, index=idx)
    assert E.block_bootstrap_ci(const, 20) == pytest.approx((0.3, 0.3))
    # 순환 블록: 부트스트랩 평균의 기대값 = 표본 평균 (양끝 관측도 똑같이 뽑힌다)
    ramp = pd.Series(np.arange(101, dtype=float), index=idx[:101])
    lo_r, hi_r = E.block_bootstrap_ci(ramp, 60, n_boot=4000)
    assert lo_r < ramp.mean() < hi_r
    rng = np.random.default_rng(3)
    v = pd.Series(rng.binomial(1, 0.6, 500).astype(float), index=idx)
    v.iloc[[5, 77]] = np.nan
    lo, hi = E.block_bootstrap_ci(v, 20, n_boot=500)
    m = v.mean()
    assert lo < m < hi and 0 < hi - lo < 0.2
    assert E.block_bootstrap_ci(v, 20, n_boot=500) == (lo, hi)                # 재현성
    assert E.block_bootstrap_ci(v, 20, n_boot=500, seed=1) != (lo, hi)
    lo2, hi2 = E.block_bootstrap_ci(v, 20, n_boot=500, ci=0.5)
    assert lo <= lo2 <= hi2 <= hi
    # 블록이 표본보다 길면 퇴화(평균 하나), 빈 입력은 NaN
    assert E.block_bootstrap_ci(v.iloc[:10], 20, n_boot=50) == pytest.approx((v.iloc[:10].mean(),) * 2)
    assert all(math.isnan(x) for x in E.block_bootstrap_ci(pd.Series(dtype=float), 20))
    with pytest.raises(ValueError):
        E.block_bootstrap_ci(v, 0)
    with pytest.raises(ValueError):
        E.block_bootstrap_ci(v, 5, ci=1.0)


# ------------------------------------------------------------------
# summarize_v0
# ------------------------------------------------------------------
def test_summarize_v0_is_json_serializable(synth):
    close, targets, replay = synth
    ep5, ep10 = T.episodes(close, 0.05), T.episodes(close, 0.10)
    s = E.summarize_v0(replay, targets, ep5, ep10, close)
    text = json.dumps(s, ensure_ascii=False)                 # numpy/Timestamp/NaN 이 남아 있으면 여기서 실패
    back = json.loads(text)
    assert set(back) >= {"meta", "base_rates", "scorecard", "episodes", "allocation", "switching",
                         "brier_reference", "headline", "warnings"}
    assert back["meta"]["replay_start"] == replay.index[0].strftime("%Y-%m-%d")
    assert back["meta"]["n_days"] == len(replay)
    assert back["meta"]["n_blocks"]["20"] == len(replay) // 20
    assert sum(back["meta"]["tone_counts"].values()) == len(replay)
    assert len(back["scorecard"]) == (len(TONES) + 1) * len(HORIZONS)
    sc = pd.DataFrame(back["scorecard"])
    assert list(sc.columns) == list(E.SCORECARD_COLUMNS)
    assert back["episodes"]["5"]["summary"]["n_episodes"] == len(ep5)
    assert back["episodes"]["10"]["summary"]["n_episodes"] == len(ep10)
    assert "series" not in back["allocation"]
    for k in ("cagr", "bh_cagr", "max_dd", "bh_max_dd", "worst_month", "switches_per_year"):
        assert isinstance(back["allocation"][k], float)
    assert isinstance(back["allocation"]["start"], str)
    # 미회복 에피소드의 recovery_date 는 None 으로, 날짜는 문자열로
    tab = back["episodes"]["5"]["table"]
    if tab:
        assert isinstance(tab[0]["peak_date"], str)
        unrec = [r for r in tab if r["recovery_date"] is None]
        assert len(unrec) == int(ep5["recovery_date"].isna().sum())
    assert set(back["headline"]["hit_20_by_tone"]) == set(TONES)
    assert back["headline"]["baseline_20"] == sc.set_index(["tone", "h"]).loc[("all", 20), "baseline"]
    # 기준선 키(VALIDATION.md §5): 무작위 이동 탐지율·리드, 새로 켜진 경고만의 리드, 상한 걸린 수, 오경보율 기저선
    for k in ("median_lead_10_fresh", "n_lead_capped_10", "lookback", "null_detection_rate_10", "null_median_lead_10",
              "null_detection_rate_5", "false_alarm_rate", "false_alarm_rate_baseline", "true_alarm_share_baseline"):
        assert k in back["headline"], k
    assert back["headline"]["lookback"] == E.DEFAULT_LOOKBACK
    assert set(back["episodes"]["10"]["summary"]["null"]) >= {"n_shift", "detection_rate_mean", "median_lead_days_p50"}
    assert "false_alarm_rate_baseline" in back["switching"] and "null_true_alarm_share" in back["switching"]
    assert back["allocation"]["maxdd_improvement"] == pytest.approx(back["allocation"]["max_dd"] - back["allocation"]["bh_max_dd"])
    assert math.isclose(back["brier_reference"]["y_dd5_20"]["base_rate"],
                        targets.loc[replay.index, "y_dd5_20"].mean())
    assert back["base_rates"]["replay_period"]["y_sign_20"] == pytest.approx(
        targets.loc[replay.index, "y_sign_20"].mean())


def test_jsonable_edge_cases():
    obj = {"a": np.int64(3), "b": np.float64(np.nan), "c": pd.NaT, "d": pd.Timestamp("2020-01-02"),
           "e": np.bool_(True), "f": np.array([1.5, np.inf]), "g": pd.Series([1, 2], index=["x", "y"]),
           "h": (1, 2), "i": np.datetime64("NaT")}
    out = E._jsonable(obj)
    assert out == {"a": 3, "b": None, "c": None, "d": "2020-01-02", "e": True, "f": [1.5, None],
                   "g": {"x": 1, "y": 2}, "h": [1, 2], "i": None}
    json.dumps(out)
