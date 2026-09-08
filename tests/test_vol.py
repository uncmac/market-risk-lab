# -*- coding: utf-8 -*-
"""mrl/vol.py 검증 (ARCHITECTURE_PHASE2.md §5·§15).

* 합성 OHLC: GK·OV·Parkinson 손계산 대조, 하한, 비음수, HAR 성분·고정 동일가중, 미래 행 불변.
* 적합 log-HAR(보조 출력): 합성 회복(계수 4개, 오차 < 0.05), walk-forward 퍼지·일정, 누수 카나리, 성적표.
* 실캐시(홀드아웃 제외 하드컷): rolling-250 (GK+OV)/CC ∈ [0.8,1.3](1996~), Parkinson/CC RV20 ∈ [0.3,3], 1993~95 OHLC 품질.
실행: python -m pytest tests/test_vol.py -q -s
"""
from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrl import vol as V  # noqa: E402
from mrl.config import DATA_DIR, HOLDOUT_START, P2  # noqa: E402

HAS_CACHE = (DATA_DIR / "spy_ohlc.csv").exists()
needs_cache = pytest.mark.skipif(not HAS_CACHE, reason="data/ 캐시 없음")
PRE_HOLDOUT_END = (pd.Timestamp(HOLDOUT_START) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
FLOOR = P2["var_floor"]


# ------------------------------------------------------------------
# 합성 자료
# ------------------------------------------------------------------
def _ohlc(n: int = 400, seed: int = 0, start: str = "2012-01-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n, name="date")
    r = rng.normal(0.0002, 0.011, n)
    close = 100.0 * np.exp(np.cumsum(r))
    prev = np.r_[close[0] / np.exp(r[0]), close[:-1]]
    open_ = prev * np.exp(rng.normal(0.0, 0.004, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0.0, 0.005, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0.0, 0.005, n)))
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1e6}, index=idx)


def _hand_ohlc() -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-06", periods=4, name="date")
    return pd.DataFrame({"Open": [100.0, 101.0, 99.0, 102.0], "High": [102.0, 103.0, 101.0, 104.0],
                         "Low": [99.0, 100.0, 97.5, 101.0], "Close": [101.5, 100.5, 100.0, 103.0]}, index=idx)


@pytest.fixture(scope="module")
def ohlc_pre_holdout() -> pd.DataFrame:
    if not HAS_CACHE:
        pytest.skip("data/ 캐시 없음")
    from mrl.data import load_cache
    b = load_cache()
    o = b.spy_ohlc.loc[:PRE_HOLDOUT_END]
    assert o.index[-1] < pd.Timestamp(HOLDOUT_START)
    return o


# ------------------------------------------------------------------
# 일간 분산 추정량
# ------------------------------------------------------------------
def test_gk_ov_hand_calculation():
    o = _hand_ohlc()
    v = V.garman_klass_variance(o, overnight=True, floor=FLOOR)
    g = V.garman_klass_variance(o, overnight=False, floor=FLOOR)
    k = 2 * math.log(2) - 1
    for t in range(4):
        gk = 0.5 * math.log(o.High.iloc[t] / o.Low.iloc[t]) ** 2 - k * math.log(o.Close.iloc[t] / o.Open.iloc[t]) ** 2
        assert abs(g.iloc[t] - max(gk, FLOOR)) < 1e-18, t
        if t == 0:
            assert np.isnan(v.iloc[0]), "첫 행은 전일 종가가 없어 NaN"
        else:
            ov = math.log(o.Open.iloc[t] / o.Close.iloc[t - 1]) ** 2
            assert abs(v.iloc[t] - max(gk + ov, FLOOR)) < 1e-18, t
    assert v.name == "var_gkov" and g.name == "var_gk"
    assert abs(V.GK_COEF - 0.3862943611198906) < 1e-15
    # OV 항은 필수: 야간갭이 있으면 GK+OV > GK
    assert (v.iloc[1:] > g.iloc[1:]).all()


def test_parkinson_hand_calculation():
    o = _hand_ohlc()
    p = V.parkinson_variance(o)
    for t in range(4):
        assert abs(p.iloc[t] - math.log(o.High.iloc[t] / o.Low.iloc[t]) ** 2 / (4 * math.log(2))) < 1e-18
    assert p.name == "var_pk"


def test_variance_floor_and_nonnegativity():
    idx = pd.bdate_range("2020-01-06", periods=5, name="date")
    flat = pd.DataFrame({"Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0}, index=idx)
    v = V.garman_klass_variance(flat)
    assert (v.iloc[1:] == FLOOR).all() and np.isnan(v.iloc[0])
    assert (V.garman_klass_variance(flat, overnight=False) == FLOOR).all()
    assert (V.parkinson_variance(flat) == FLOOR).all()
    v2 = V.garman_klass_variance(flat, floor=1e-6)
    assert (v2.iloc[1:] == 1e-6).all()
    # 무작위 OHLC: 전부 ≥ floor, 유한
    o = _ohlc(1000, seed=1)
    for s in (V.garman_klass_variance(o), V.garman_klass_variance(o, overnight=False), V.parkinson_variance(o)):
        d = s.dropna()
        assert (d >= FLOOR).all() and np.isfinite(d).all()
    # NaN 행은 NaN 으로 전파(지어내지 않음)
    o2 = o.copy()
    o2.loc[o2.index[10], "High"] = np.nan
    v3 = V.garman_klass_variance(o2)
    assert np.isnan(v3.iloc[10]) and np.isfinite(v3.iloc[11])


def test_input_validation():
    o = _ohlc(50)
    with pytest.raises(ValueError):
        V.garman_klass_variance(o.drop(columns="High"))
    bad = o.copy()
    bad.loc[bad.index[3], "Low"] = -1.0
    with pytest.raises(ValueError):
        V.garman_klass_variance(bad)
    dup = pd.concat([o.iloc[:5], o.iloc[4:6]])
    with pytest.raises(ValueError):
        V.garman_klass_variance(dup)
    with pytest.raises(ValueError):
        V.garman_klass_variance(o, floor=0.0)
    with pytest.raises(TypeError):
        V.garman_klass_variance(o["Close"])
    with pytest.raises(ValueError):
        V.har_components(pd.Series([-1.0, 1.0], index=o.index[:2]))
    with pytest.raises(ValueError):
        V.har_components(V.garman_klass_variance(o), lookbacks=(1, 1, 5))


# ------------------------------------------------------------------
# HAR 성분 · 고정가중
# ------------------------------------------------------------------
def test_har_components_definition_and_floor():
    o = _ohlc(300, seed=2)
    v = V.garman_klass_variance(o)
    comp = V.har_components(v, lookbacks=(1, 5, 22), floor=FLOOR)
    assert list(comp.columns) == ["rv1", "rv5", "rv22"]
    for k in (1, 5, 22):
        ref = np.sqrt(252 * v.rolling(k, min_periods=k).mean().clip(lower=FLOOR))
        assert np.allclose(comp[f"rv{k}"].dropna(), ref.dropna(), rtol=1e-14, atol=0)
        assert comp[f"rv{k}"].iloc[:k].isna().all()          # 첫 행 NaN(OV) + 창 미충족
        assert comp[f"rv{k}"].iloc[k:].notna().all()
    # 하한: 분산 0 → rv = √(252·1e-8)
    z = pd.Series(0.0, index=o.index)
    cz = V.har_components(z)
    assert np.allclose(cz.dropna(), math.sqrt(252 * FLOOR), rtol=1e-14)


def test_har_log_vol_equal_weights():
    idx = pd.bdate_range("2020-01-06", periods=3)
    comp = pd.DataFrame({"rv1": [0.1, 0.2, 0.3], "rv5": [0.2, 0.2, 0.3], "rv22": [0.4, 0.2, 0.3]}, index=idx)
    ln = V.har_log_vol(comp)                                   # 기본 P2["har_weights"] = (1/3,1/3,1/3)
    assert abs(math.exp(ln.iloc[0]) - (0.1 * 0.2 * 0.4) ** (1 / 3)) < 1e-12
    assert abs(math.exp(ln.iloc[1]) - 0.2) < 1e-12 and abs(math.exp(ln.iloc[2]) - 0.3) < 1e-12
    assert np.allclose(ln, (np.log(comp["rv1"]) + np.log(comp["rv5"]) + np.log(comp["rv22"])) / 3, rtol=0, atol=1e-15)
    assert P2["har_weights"] == (1 / 3, 1 / 3, 1 / 3)
    # 단조: 성분이 커지면 ln σ_har 도 커진다 (갭 부호 규약의 근거)
    assert V.har_log_vol(comp * 2).iloc[0] > ln.iloc[0]
    with pytest.raises(ValueError):
        V.har_log_vol(comp, weights=(0.5, 0.5, 0.5))
    with pytest.raises(ValueError):
        V.har_log_vol(comp, weights=(0.5, 0.5))
    with pytest.raises(ValueError):
        V.har_log_vol(comp.assign(rv1=[0.0, 0.2, 0.3]))


def test_future_rows_do_not_change_past():
    """t 이후 행을 덧붙여도 t 까지의 var/rv/ln σ_har 는 비트 동일 (뒤를 보는 창)."""
    full = _ohlc(500, seed=4)
    t = 300
    part = full.iloc[:t]
    v_full, v_part = V.garman_klass_variance(full), V.garman_klass_variance(part)
    c_full, c_part = V.har_components(v_full), V.har_components(v_part)
    h_full, h_part = V.har_log_vol(c_full), V.har_log_vol(c_part)
    p_full, p_part = V.parkinson_variance(full), V.parkinson_variance(part)
    assert np.array_equal(v_full.iloc[:t].to_numpy(), v_part.to_numpy(), equal_nan=True)
    assert np.array_equal(p_full.iloc[:t].to_numpy(), p_part.to_numpy(), equal_nan=True)
    for k in c_part.columns:
        assert np.array_equal(c_full[k].iloc[:t].to_numpy(), c_part[k].to_numpy(), equal_nan=True)
    assert np.array_equal(h_full.iloc[:t].to_numpy(), h_part.to_numpy(), equal_nan=True)


# ------------------------------------------------------------------
# 보조 출력: 적합 log-HAR
# ------------------------------------------------------------------
def test_har_target_matches_targets_convention():
    o = _ohlc(300, seed=5)
    c = o["Close"]
    y = V.har_target(c, h=20)
    logret = np.log(c).diff()
    rv20 = logret.rolling(20, min_periods=20).std() * math.sqrt(252)
    ref = np.log(rv20.shift(-20))
    assert y.iloc[-20:].isna().all()
    assert np.allclose(y.dropna(), ref.dropna(), rtol=0, atol=1e-15)
    assert y.name == "ln_rv_fwd"
    # 가격 불변 → 경고 + NaN (지어내지 않음)
    flat = pd.Series(100.0, index=o.index[:60])
    with pytest.warns(UserWarning):
        yf = V.har_target(flat)
    assert yf.isna().all()


def _synthetic_har(n: int = 5000, seed: int = 0, noise: float = 0.05, beta=(0.1, 0.3, 0.4, 0.2)):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1995-01-02", periods=n, name="date")
    ln1 = np.log(0.15) + 0.3 * rng.standard_normal(n)
    ln5 = pd.Series(ln1).rolling(5, min_periods=1).mean().to_numpy()
    ln22 = pd.Series(ln1).rolling(22, min_periods=1).mean().to_numpy()
    comp = pd.DataFrame({"rv1": np.exp(ln1), "rv5": np.exp(ln5), "rv22": np.exp(ln22)}, index=idx)
    y = beta[0] + beta[1] * ln1 + beta[2] * ln5 + beta[3] * ln22 + noise * rng.standard_normal(n)
    return comp, pd.Series(y, index=idx, name="ln_rv_fwd"), beta


def test_har_fit_synthetic_recovery():
    comp, y, beta = _synthetic_har()
    fit = V.har_fit(comp, y, refit_date="2015-01-02")
    assert list(fit["coef"]) == list(V.HAR_COEF_NAMES) and len(fit["coef"]) == V.HAR_PARAM_COUNT == 4
    for name, b in zip(V.HAR_COEF_NAMES, beta):
        assert abs(fit["coef"][name] - b) < 0.05, (name, fit["coef"][name], b)
    assert fit["n"] == len(comp) and fit["refit_date"] == "2015-01-02"
    assert fit["train_start"] == "1995-01-02" and abs(fit["resid_std"] - 0.05) < 0.01
    # 예측 왕복
    pred = V.har_predict(fit["coef"], comp)
    A = np.column_stack([np.ones(len(comp)), np.log(comp.to_numpy())])
    assert np.allclose(pred, A @ np.array(list(fit["coef"].values())), rtol=0, atol=1e-12)
    # NaN 성분 → NaN 예측 ; 표본 부족 → ValueError
    comp2 = comp.copy()
    comp2.iloc[3, 0] = np.nan
    assert np.isnan(V.har_predict(fit["coef"], comp2).iloc[3])
    with pytest.raises(ValueError):
        V.har_fit(comp.iloc[:50], y.iloc[:50])
    with pytest.raises(ValueError):
        V.har_predict({"const": 0.0}, comp)


def test_har_walk_forward_schedule_purge_and_leakage_canary():
    comp, y, _ = _synthetic_har(n=3000, seed=1)
    idx = comp.index
    refits = [idx[idx.year == yr][0] for yr in range(1998, 2007)]      # 매년 첫 세션
    fc, table = V.har_walk_forward(comp, y, refits, purge=20, train_start="1995-01-02")
    assert fc.index.equals(idx) and fc.name == "ln_har_fc"
    assert len(table) == len(refits) and list(table.columns[:5]) == ["refit_date", *V.HAR_COEF_NAMES]
    first = idx.get_loc(refits[0])
    assert fc.iloc[:first].isna().all() and fc.iloc[first:].notna().all()
    A = np.column_stack([np.ones(len(comp)), np.log(comp.to_numpy())])
    yv = y.to_numpy()
    for i, r in enumerate(refits):
        pos = idx.get_loc(r)
        end = idx.get_loc(refits[i + 1]) if i + 1 < len(refits) else len(idx)
        mask = np.arange(len(idx)) <= pos - 21                        # 퍼지 20: max(학습 pos) + 20 < pos(R)
        beta = np.linalg.lstsq(A[mask], yv[mask], rcond=None)[0]
        row = table.iloc[i]
        assert row["refit_date"] == r.strftime("%Y-%m-%d") and row["n"] == int(mask.sum())
        assert idx.get_loc(pd.Timestamp(row["train_end"])) == pos - 21
        assert np.allclose(row[list(V.HAR_COEF_NAMES)].to_numpy(dtype=float), beta, rtol=0, atol=1e-10)
        assert np.allclose(fc.iloc[pos:end].to_numpy(), A[pos:end] @ beta, rtol=0, atol=1e-10)
    # 누수 카나리: 퍼지 창(pos(R)-20..pos(R)-1) 과 R 이후의 라벨을 망가뜨려도 R 의 계수는 그대로
    y2 = y.copy()
    pos1 = idx.get_loc(refits[3])
    y2.iloc[pos1 - 20:] = 99.0
    fc2, table2 = V.har_walk_forward(comp, y2, refits[:4], purge=20, train_start="1995-01-02")
    assert np.allclose(table2.iloc[3][list(V.HAR_COEF_NAMES)].to_numpy(dtype=float),
                       table.iloc[3][list(V.HAR_COEF_NAMES)].to_numpy(dtype=float), rtol=0, atol=0)
    # 퍼지 0 이면 pos(R)-1 까지 학습 (다른 계수)
    fc0, table0 = V.har_walk_forward(comp, y, refits[:2], purge=0)
    assert idx.get_loc(pd.Timestamp(table0.iloc[0]["train_end"])) == idx.get_loc(refits[0]) - 1
    # train_start 민감도: 학습 시작 이전 행 제외
    _, t96 = V.har_walk_forward(comp, y, refits[:1], train_start="1996-01-02")
    assert t96.iloc[0]["train_start"] == "1996-01-02"
    # 세션이 아닌 재적합일 → 다음 세션 사용 + 경고 ; 표본 부족 → ValueError
    with pytest.warns(UserWarning):
        V.har_walk_forward(comp, y, [refits[0] + pd.Timedelta(days=5 - refits[0].weekday())])
    with pytest.raises(ValueError):
        V.har_walk_forward(comp, y, [idx[50]])


def test_vol_scorecard_columns_and_perfect_forecast():
    comp, y, _ = _synthetic_har(n=1500, seed=2)
    idx = comp.index
    vix = pd.Series(np.exp(y) * 100 * 1.1, index=idx)          # VIX = 실현 × 1.1 (프리미엄)
    rv22 = comp["rv22"]
    blocks = [("1995-01-01", "1997-01-01"), ("1997-01-01", "1999-01-01"), ("1999-01-01", "2001-01-01")]
    sc = V.vol_scorecard(y, y, vix, rv22, blocks)                # 완벽 예측
    assert list(sc["block"])[-1] == "all" and len(sc) == len(blocks) + 1
    for c in ("n", "n_blocks", "mse_log", "qlike", "r2_log", "auc_vol", "mse_log_vix", "qlike_vix", "r2_log_vix", "auc_vol_vix",
              "mse_log_rv22", "qlike_rv22", "r2_log_rv22", "auc_vol_rv22", "n_pos_vol"):
        assert c in sc.columns, c
    fin = sc[sc["n"] > 0]
    assert (fin["mse_log"] < 1e-24).all() and (fin["qlike"].abs() < 1e-12).all() and (fin["r2_log"] > 1 - 1e-12).all()
    assert (sc["n_blocks"] == sc["n"] // 20).all()
    # VIX 예측(편향 +10%): mse = ln(1.1)^2, QLIKE ≥ 0
    assert np.allclose(fin["mse_log_vix"], math.log(1.1) ** 2, rtol=1e-9)
    assert (fin["qlike_vix"] > 0).all() and (fin["qlike_rv22"] >= 0).all()
    # 무작위 예측은 QLIKE ≥ 0, y_vol 을 직접 주면 그 라벨로 AUC
    rng = np.random.default_rng(0)
    noisy = y + rng.normal(0, 0.3, len(y))
    y_vol = pd.Series((rng.random(len(y)) < 0.5).astype(float), index=idx)
    sc2 = V.vol_scorecard(noisy, y, vix, rv22, blocks, y_vol=y_vol)
    assert (sc2["qlike"].dropna() > 0).all() and sc2["auc_vol"].notna().all()
    with pytest.raises(ValueError):
        V.vol_scorecard(y, y, vix, rv22, [])


def test_auc_helper():
    assert V.auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == 0.75
    assert V.auc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert V.auc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5
    assert V.auc([1, 2, 3, 4], [1, 1, 0, 0]) == 0.0
    assert np.isnan(V.auc([1, 2], [1, 1]))
    assert V.auc([1, np.nan, 3, 4], [0, 1, 1, 0]) == 0.5
    with pytest.raises(ValueError):
        V.auc([1, 2], [0, 2])


# ------------------------------------------------------------------
# 실캐시 자기점검 (홀드아웃 제외)
# ------------------------------------------------------------------
@needs_cache
def test_ratio_checks_real_cache(ohlc_pre_holdout):
    rc = V.ratio_checks(ohlc_pre_holdout)
    a, b, q = rc["gkov_cc_ratio_250"], rc["pk_cc_rv20"], rc["ohlc_quality_1993_95"]
    print(f"\n(GK+OV)/CC rolling-250 1996+: min {a['since_1996']['min']:.3f} max {a['since_1996']['max']:.3f} "
          f"mean {a['since_1996']['mean']:.3f} | 1993-95 mean {a['early_1993_95']['mean']:.3f} "
          f"[{a['early_1993_95']['min']:.3f}, {a['early_1993_95']['max']:.3f}]")
    print(f"Parkinson/CC RV20: min {b['all']['min']:.3f} max {b['all']['max']:.3f} mean {b['all']['mean']:.3f}")
    print(f"1993-95 Open==H|L share {q['early_1993_95']['share_open_eq_high_or_low']:.3%}, median log range "
          f"{q['early_1993_95']['median_log_range_pct']:.2f}% vs 1996+ {q['since_1996']['median_log_range_pct']:.2f}% "
          f"(share {q['since_1996']['share_open_eq_high_or_low']:.2%})")
    assert rc["ok"] and a["ok"] and b["ok"] and rc["warnings"] == []
    assert 0.8 <= a["since_1996"]["min"] and a["since_1996"]["max"] <= 1.3
    assert 0.3 <= b["all"]["min"] and b["all"]["max"] <= 3.0
    assert abs(q["early_1993_95"]["share_open_eq_high_or_low"] - 0.299) < 0.01
    assert abs(q["early_1993_95"]["median_log_range_pct"] - 0.62) < 0.05
    assert abs(q["since_1996"]["median_log_range_pct"] - 1.09) < 0.05
    assert q["early_1993_95"]["share_open_eq_high_or_low"] > 5 * q["since_1996"]["share_open_eq_high_or_low"]


@needs_cache
def test_gkov_cc_ratio_real_cache_direct(ohlc_pre_holdout):
    """ratio_checks 와 독립적으로 다시 계산: rolling-250 √(mean v_gkov)/std(logret) 이 1996년 이후 [0.8, 1.3]."""
    o = ohlc_pre_holdout
    v = V.garman_klass_variance(o)
    lr = np.log(o["Close"]).diff()
    ratio = (np.sqrt(v.rolling(250).mean()) / lr.rolling(250).std()).loc["1996-01-01":].dropna()
    assert len(ratio) > 7000 and ratio.min() >= 0.8 and ratio.max() <= 1.3
    pk = V.parkinson_variance(o)
    pk_cc = (np.sqrt(pk.rolling(20).mean()) / lr.rolling(20).std()).dropna()
    assert pk_cc.min() >= 0.3 and pk_cc.max() <= 3.0
    # 실캐시 var_gkov: 전부 유한·≥ floor, 첫 행만 NaN, 하한이 실제로 걸린 행은 없다
    assert np.isnan(v.iloc[0]) and (v.iloc[1:] > FLOOR).all()
