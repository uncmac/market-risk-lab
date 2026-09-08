# -*- coding: utf-8 -*-
"""mrl/calibrate.py 검증 (ARCHITECTURE_PHASE2.md §8·§15).

실행: 프로젝트 루트에서  python -m pytest tests/test_calibrate.py -q

* 일정·퍼지·첫 재적합 규칙·블록은 실캐시(SPY 인덱스·targets)로 확인한다.
* walk-forward 역학(열 계약·결정론·예산·홀드아웃 하드컷·누수 카나리)은 합성 특징으로 확인한다.
* 실캐시 '지연 특징' 검사는 mrl.features 가 있으면 build_features, 없으면 같은 열 계약의 프록시 특징(종가 HAR)을 쓴다.
* walk-forward vs 오프라인 parity 는 build_features 가 필요하므로 mrl.features 가 없으면 skip.
"""
from __future__ import annotations

import importlib
import math

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, ndtr

from mrl import calibrate as C
from mrl import model as M
from mrl.config import BLOCKS_18, BLOCKS_24, BLOCKS_24_FROM_1999, HOLDOUT_START, P2
from mrl.data import load_cache
from mrl.evaluate import brier
from mrl.targets import episodes, make_targets

HOLDOUT = pd.Timestamp(HOLDOUT_START)
SESSIONS_24 = [504, 503, 504, 504, 502, 504, 504, 502, 505, 503, 418]                      # §8.2 표
SESSIONS_18 = [376, 380, 375, 380, 376, 380, 374, 380, 377, 378, 419, 419, 421, 418]


# ------------------------------------------------------------------
# 합성·프록시 특징
# ------------------------------------------------------------------
def _b1(vix, dd=0.05, h=20, drift=True, bgk=False):
    """§6 B1 식(테스트용 독립 구현). VIX 20 → 0.372."""
    V = np.asarray(vix, dtype=float) / 100.0
    T = h / 252.0
    s = V * np.sqrt(T)
    b = np.log(1.0 - dd)
    if bgk:
        b = b - 0.5826 * V * np.sqrt(1.0 / 252.0)
    m = -s * s / 2.0 if drift else 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        return ndtr((b - m) / s) + np.exp(2.0 * m * b / (s * s)) * ndtr((b + m) / s)


def _ar1(n, phi, sd, rng, mean=0.0):
    out = np.empty(n)
    out[0] = mean
    eps = rng.normal(0.0, sd, n)
    for t in range(1, n):
        out[t] = mean + phi * (out[t - 1] - mean) + eps[t]
    return out


def synth_feats(idx: pd.DatetimeIndex, seed: int = 0, b=(-1.0, 0.9, 0.8, -4.0)) -> tuple[pd.DataFrame, pd.Series]:
    """열 계약(§4)을 갖춘 합성 특징 + 로지스틱 라벨(마지막 20행 NaN)."""
    rng = np.random.default_rng(seed)
    n = len(idx)
    vix = np.exp(_ar1(n, 0.98, 0.06, rng, mean=2.9))
    f = pd.DataFrame(index=idx)
    f["vix"] = vix
    f["p_vix"] = _b1(vix)
    f["p_vix_driftless"] = _b1(vix, drift=False)
    f["p_vix_bgk"] = _b1(vix, bgk=True)
    f["x_vix"] = np.log(f["p_vix"] / (1.0 - f["p_vix"]))
    f["x_har"] = _ar1(n, 0.9, 0.15, rng, mean=-0.3)
    f["x_ma"] = _ar1(n, 0.99, 0.01, rng, mean=0.0)
    z = b[0] + b[1] * f["x_vix"].to_numpy() + b[2] * f["x_har"].to_numpy() + b[3] * f["x_ma"].to_numpy()
    y = pd.Series((rng.uniform(size=n) < expit(z)).astype(float), index=idx, name="y_dd5_20")
    y.iloc[-20:] = np.nan
    f.attrs = {"feature_rule": "synthetic", "spec_sha256": "s" * 64, "warnings": []}
    return f, y


def proxy_features(bundle, end=None) -> pd.DataFrame:
    """mrl.features 부재 시의 프록시: B1 x_vix · 종가 HAR x_har · MA180 x_ma (열 계약 동일)."""
    spy, close = bundle.spy_ohlc, bundle.close
    if end is not None:
        spy = spy[spy.index < pd.Timestamp(end)]
        close = close[close.index < pd.Timestamp(end)]
    idx = spy.index
    vix = close["^VIX"].reindex(idx).ffill(limit=3)
    f = pd.DataFrame(index=idx)
    f["vix"] = vix
    f["p_vix"] = _b1(vix)
    f["p_vix_driftless"] = _b1(vix, drift=False)
    f["p_vix_bgk"] = _b1(vix, bgk=True)
    f["x_vix"] = np.log(f["p_vix"] / (1.0 - f["p_vix"]))
    r2 = np.log(spy["Close"]).diff() ** 2
    rvs = {k: np.sqrt(252.0 * r2.rolling(k, min_periods=k).mean().clip(lower=1e-8)) for k in (1, 5, 22)}
    har = np.exp((np.log(rvs[1]) + np.log(rvs[5]) + np.log(rvs[22])) / 3.0)
    f["x_har"] = np.log(har) - np.log(vix / 100.0)
    sma = spy["Close"].rolling(180, min_periods=180).mean()
    f["x_ma"] = spy["Close"] / sma - 1.0
    f.attrs = {"feature_rule": "proxy(cc-har)", "spec_sha256": M.spec_sha256(), "warnings": []}
    return f


def _features_module():
    try:
        return importlib.import_module("mrl.features")
    except ImportError:
        return None


def real_or_proxy_features(bundle, end=HOLDOUT_START) -> tuple[pd.DataFrame, str]:
    F = _features_module()
    if F is not None and hasattr(F, "build_features"):
        last = bundle.spy_ohlc.index[bundle.spy_ohlc.index < pd.Timestamp(end)][-1]
        return F.build_features(bundle, asof=last), "build_features"
    return proxy_features(bundle, end=end), "proxy"


# ------------------------------------------------------------------
# 픽스처 (실캐시)
# ------------------------------------------------------------------
@pytest.fixture(scope="module")
def bundle():
    return load_cache()


@pytest.fixture(scope="module")
def full_idx(bundle) -> pd.DatetimeIndex:
    return bundle.spy_ohlc.index


@pytest.fixture(scope="module")
def cut(bundle):
    """홀드아웃 하드컷(2024-08-30 까지)의 (idx, targets, episodes20)."""
    close = bundle.spy_ohlc["Close"]
    close = close[close.index < HOLDOUT]
    tg = make_targets(close)
    return close.index, tg, episodes(close, 0.20)


@pytest.fixture(scope="module")
def synth_cut(cut):
    idx, _, _ = cut
    return synth_feats(idx)


@pytest.fixture(scope="module")
def wf(synth_cut):
    feats, y = synth_cut
    return C.walk_forward(feats, y)


# ------------------------------------------------------------------
# 8.1 일정·퍼지·기후학·첫 재적합 규칙
# ------------------------------------------------------------------
def test_refit_dates_are_first_january_sessions(cut):
    idx, _, _ = cut
    rd = C.refit_dates(idx, first="1999-01-04")
    s = {d.strftime("%Y-%m-%d") for d in rd}
    for d in ("1999-01-04", "2003-01-02", "2007-01-03", "2024-01-02"):
        assert d in s, d
    assert len(rd) == 26 and rd == sorted(rd)
    for d in rd:
        jan = idx[(idx.year == d.year) & (idx.month == 1)]
        assert d == jan[0]
    rd2003 = C.refit_dates(idx)
    assert len(rd2003) == 22 and rd2003[0] == pd.Timestamp("2003-01-02") and rd2003[-1] == pd.Timestamp("2024-01-02")
    assert all(d < HOLDOUT for d in rd2003)
    with pytest.raises(ValueError):
        C.refit_dates(idx, first="2003-01-03")        # 1월 첫 거래일이 아님
    with pytest.raises(ValueError):
        C.refit_dates(idx, first="2003-01-01")        # 인덱스에 없음


def test_purge_every_refit(cut):
    idx, tg, _ = cut
    pos = np.arange(len(idx))
    for R in C.refit_dates(idx, first="1999-01-04"):
        mask = C.training_mask(idx, R)
        pos_R = idx.get_loc(R)
        assert mask.any()
        assert pos[mask].max() + 20 < pos_R
        assert pos[mask].max() == pos_R - 21
        assert idx[mask][0] == pd.Timestamp("1993-10-14")
        # 학습 라벨 창이 R 전에 완전히 실현: 마지막 학습 행의 20일 뒤 세션 < R
        assert idx[pos[mask].max() + 20] < R
    with pytest.raises(ValueError):
        C.training_mask(idx, "2003-01-01")


def test_holdout_rows_never_in_training_mask(full_idx):
    hold = full_idx >= HOLDOUT
    assert hold.sum() > 400
    for R in C.refit_dates(full_idx, end=HOLDOUT_START):
        mask = C.training_mask(full_idx, R)
        assert not (mask & hold).any()
    # 라이브 모델(refit_date=2024-08-30)의 학습창도 홀드아웃 전(≈2024-08-01 까지)
    live = C.training_mask(full_idx, "2024-08-30")
    assert not (live & hold).any()
    assert full_idx[live][-1] == pd.Timestamp("2024-08-01")


def test_walk_forward_hard_cuts_inputs_before_holdout(full_idx):
    feats, y = synth_feats(full_idx, seed=1)
    y = y.copy()
    y.iloc[:] = y.fillna(0.0)                                   # 홀드아웃 안까지 라벨을 다 채워도
    oos, params = C.walk_forward(feats, y)                      # 입력 하드컷이 먼저 일어난다
    assert oos.index.max() < HOLDOUT
    assert oos.index.min() == pd.Timestamp("2003-01-02")
    assert oos.attrs["end"] == HOLDOUT_START and oos.attrs["holdout_final"] is False
    for p in params:
        assert pd.Timestamp(p["train_end"]) < HOLDOUT
        assert pd.Timestamp(p["refit_date"]) < HOLDOUT
    assert max(int(p["refit_date"][:4]) for p in params) == 2024
    assert oos.attrs["n_refits"] == 22


def test_holdout_final_extends_to_last_session(full_idx):
    feats, y = synth_feats(full_idx, seed=2)
    oos, params = C.walk_forward(feats, y, holdout_final=True)
    assert oos.index.max() == full_idx[-1]
    years = sorted({int(p["refit_date"][:4]) for p in params})
    assert years[-1] == full_idx[-1].year and 2025 in years
    r2025 = [p for p in params if p["refit_date"].startswith("2025") and p["rung"] == "M3"][0]
    assert pd.Timestamp(r2025["train_end"]) > HOLDOUT               # 홀드아웃 자료가 R_2025 학습에 들어간다(최종 검증 모드)
    assert (oos.loc[oos.index >= HOLDOUT, "refit_year"] >= 2024).all()


def test_climatology_equals_purged_training_mean(cut):
    idx, tg, _ = cut
    y = tg["y_dd5_20"]
    for R in ("2003-01-02", "2010-01-04", "2024-01-02"):
        mask = C.training_mask(idx, R) & y.notna().to_numpy()
        clim = C.pit_climatology(y, mask)
        assert clim == pytest.approx(float(y[mask].mean()), abs=0)
        assert 0.10 < clim < 0.25
    with pytest.raises(ValueError):
        C.pit_climatology(y, np.zeros(len(idx), dtype=bool))


def test_first_refit_rule(cut):
    idx, tg, ep = cut
    ok, info = C.first_refit_ok(idx, tg, ep, "2003-01-02")
    assert ok is True and info["n_rows"] == 2302 and info["n_episodes"] >= 1
    assert info["episodes"][0]["peak_date"] == "2000-03-24" and info["train_end"] == "2002-12-02"
    ok99, info99 = C.first_refit_ok(idx, tg, ep, "1999-01-04")
    assert ok99 is False and info99["n_rows"] == 1298 and info99["n_episodes"] == 0


# ------------------------------------------------------------------
# 8.2 블록
# ------------------------------------------------------------------
def test_blocks_tile_and_session_counts(cut):
    idx, _, _ = cut
    r24 = C.check_blocks(BLOCKS_24, "2003-01-01", HOLDOUT_START)
    r18 = C.check_blocks(BLOCKS_18, "2003-01-01", HOLDOUT_START)
    assert r24["n_blocks"] == 11 and r18["n_blocks"] == 14
    assert all(18 <= m <= 24 for m in r24["months"] + r18["months"])
    assert r24["months"] == [24] * 10 + [20]
    assert r18["months"] == [18] * 10 + [20] * 4
    C.check_blocks(BLOCKS_24_FROM_1999, "1999-01-01", HOLDOUT_START)
    assert C.block_sessions(idx, BLOCKS_24)["n"].tolist() == SESSIONS_24
    assert C.block_sessions(idx, BLOCKS_18)["n"].tolist() == SESSIONS_18
    b24 = C.block_sessions(idx, BLOCKS_24)
    assert b24["first_session"].iloc[0] == "2003-01-02" and b24["last_session"].iloc[-1] == "2024-08-30"
    assert b24["first_session"].iloc[2] == "2007-01-03"
    with pytest.raises(ValueError):
        C.check_blocks(BLOCKS_24[:-1], "2003-01-01", HOLDOUT_START)
    with pytest.raises(ValueError):
        C.check_blocks([("2003-01-01", "2006-01-01")] + BLOCKS_24[2:], "2003-01-01", HOLDOUT_START)   # 36개월


def test_assign_blocks(cut):
    idx, _, _ = cut
    a = C.assign_blocks(idx, BLOCKS_24)
    assert a[idx < pd.Timestamp("2003-01-01")].max() == 0
    assert a[idx == pd.Timestamp("2003-01-02")][0] == 1 and a[idx == pd.Timestamp("2024-08-30")][0] == 11
    assert np.bincount(a)[1:].tolist() == SESSIONS_24


# ------------------------------------------------------------------
# walk-forward 역학 (합성 특징, 실 인덱스)
# ------------------------------------------------------------------
def test_walk_forward_columns_and_m0(wf, synth_cut):
    feats, y = synth_cut
    oos, params = wf
    for c in ("y", "clim", "p_vix", "p_vix_driftless", "p_vix_bgk", "p_m1", "p_m2", "p_m3", "refit_year", "block24", "block18"):
        assert c in oos.columns, c
    assert oos.index.name == "date"
    assert len(oos) == 5453 and oos.index[0] == pd.Timestamp("2003-01-02") and oos.index[-1] == pd.Timestamp("2024-08-30")
    # M0 == p_vix (적합 없음)
    assert C.rung_col("M0") == "p_vix" and C.rung_col("B1") == "p_vix" and C.rung_col("BGK") == "p_vix_bgk"
    assert C.rung_col("M3") == "p_m3" and C.rung_col("M3-PK") == "p_m3_pk" and C.rung_col("clim") == "clim"
    np.testing.assert_array_equal(oos["p_vix"].to_numpy(), feats.loc[oos.index, "p_vix"].to_numpy())
    # 확률·기후학 범위, 연중 상수 기후학
    for c in ("p_m1", "p_m2", "p_m3"):
        assert oos[c].notna().all() and ((oos[c] > 0) & (oos[c] < 1)).all()
    assert oos.groupby("refit_year")["clim"].nunique().eq(1).all()
    assert sorted(oos["refit_year"].unique()) == list(range(2003, 2025))
    assert oos["block24"].max() == 11 and oos["block18"].max() == 14 and (oos["block24"] > 0).all()
    assert oos["y"].isna().sum() == 20                       # 하드컷 꼬리(라벨 미실현) — 지어내지 않는다
    # params 계약
    assert len(params) == 22 * 3
    for p in params:
        assert set(p) >= {"refit_date", "rung", "coef", "intercept", "n_train", "n_pos", "clim", "train_start", "train_end", "model_id"}
        assert p["n_params"] <= M.BUDGET
        if p["rung"] == "M3":
            assert p["n_params"] == M.PARAM_COUNT == 4 and list(p["coef"]) == ["x_vix", "x_har", "x_ma"]
    # 기후학 == params 의 clim == 퍼지 학습창 평균
    p2003 = [p for p in params if p["refit_date"] == "2003-01-02"][0]
    mask = C.training_mask(feats.index, "2003-01-02") & y.notna().to_numpy()
    assert p2003["clim"] == pytest.approx(float(y[mask].mean()), abs=0)
    assert oos.loc["2003-06-02", "clim"] == p2003["clim"]
    # 합성 계수 회복(넉넉한 오차) — 사다리가 실제로 신호를 배운다
    m3 = [p for p in params if p["refit_date"] == "2024-01-02" and p["rung"] == "M3"][0]
    assert abs(m3["coef"]["x_vix"] - 0.9) < 0.2 and abs(m3["coef"]["x_har"] - 0.8) < 0.3


def test_walk_forward_rejects_bad_inputs(synth_cut):
    feats, y = synth_cut
    with pytest.raises(ValueError):
        C.walk_forward(feats.drop(columns=["p_vix_bgk"]), y)
    with pytest.raises(ValueError):
        C.walk_forward(feats, y, ladder={"M3": ("x_vix", "x_har")})           # M3 는 사전 등록 특징 집합
    with pytest.raises(ValueError):
        C.walk_forward(feats, y, ladder={"M9": ("x_vix", "x_har", "x_ma", "vix", "p_vix")})   # 예산 초과
    with pytest.raises(ValueError):
        C.walk_forward(feats, y, end=None)


def test_walk_forward_determinism(synth_cut):
    feats, y = synth_cut
    r = C.determinism_check(feats, y)
    assert r["csv_equal"] and r["coef_ok"] and r["max_abs_coef_diff"] == 0.0
    o1, _ = C.walk_forward(feats, y)
    o2, _ = C.walk_forward(feats.copy(), y.copy())
    assert C.oos_sha256(o1) == C.oos_sha256(o2)
    assert o1.to_csv() == o2.to_csv()


def test_walk_forward_nan_inputs_give_nan_probabilities(synth_cut):
    feats, y = synth_cut
    f2 = feats.copy()
    f2.loc["2010-03-01":"2010-03-05", "x_har"] = np.nan
    oos, _ = C.walk_forward(f2, y)
    assert oos.loc["2010-03-01":"2010-03-05", "p_m2"].isna().all()
    assert oos.loc["2010-03-01":"2010-03-05", "p_m3"].isna().all()
    assert oos.loc["2010-03-01":"2010-03-05", "p_m1"].notna().all()
    assert any("입력 결측" in w for w in oos.attrs["warnings"])


def test_live_model_fit_refit(synth_cut):
    feats, y = synth_cut
    m = C.fit_refit(feats, y, "2024-08-30")
    assert m.refit_date == "2024-08-30" and m.train_end == "2024-08-01" and m.n_params() == 4
    assert m.model_id.startswith("p2m3-") and m.spec_sha256 == feats.attrs["spec_sha256"]
    mask = C.training_mask(feats.index, "2024-08-30") & y.notna().to_numpy()
    assert m.clim == pytest.approx(float(y[mask].mean()), abs=0) and m.n_train == int(mask.sum())


# ------------------------------------------------------------------
# 누수 카나리 (합성): 정직 파이프라인 BSS≈0, 고의로 망가뜨린 변형 BSS>0.5
# ------------------------------------------------------------------
def test_leak_canary(cut):
    idx, _, _ = cut
    rng = np.random.default_rng(7)
    n = len(idx)
    x = rng.normal(size=n)
    y = pd.Series(np.nan, index=idx)
    y.iloc[:-1] = (x[1:] > 0).astype(float)                  # y_t = 1[x_{t+1} > 0]

    def _frame(xcol):
        f = pd.DataFrame(index=idx)
        f["x_vix"] = xcol
        f["p_vix"] = expit(xcol)
        f["p_vix_driftless"] = f["p_vix"]
        f["p_vix_bgk"] = f["p_vix"]
        return f

    honest, _ = C.walk_forward(_frame(x), y, ladder={"M1": ("x_vix",)}, purge=20)
    bss = 1.0 - brier(honest["p_m1"], honest["y"]) / brier(honest["clim"], honest["y"])
    assert -0.05 <= bss <= 0.05, bss
    x_leak = np.append(x[1:], np.nan)                        # 특징이 라벨 창(t+1)을 읽는다
    broken, _ = C.walk_forward(_frame(x_leak), y, ladder={"M1": ("x_vix",)}, purge=0)
    bss_leak = 1.0 - brier(broken["p_m1"], broken["y"]) / brier(broken["clim"], broken["y"])
    assert bss_leak > 0.5, bss_leak


# ------------------------------------------------------------------
# 실캐시: 지연 특징 → skill 감소 ; walk-forward vs 오프라인 parity
# ------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_feats(bundle, cut):
    feats, src = real_or_proxy_features(bundle, end=HOLDOUT_START)
    idx, tg, _ = cut
    assert feats.index.equals(idx), src
    return feats, src


def _pooled_bss_clim(feats, tg, lag: int) -> float:
    f = feats.copy()
    cols = ["x_vix", "x_har", "x_ma", "p_vix", "p_vix_driftless", "p_vix_bgk"]
    f[cols] = f[cols].shift(lag)
    oos, _ = C.walk_forward(f, tg)
    return 1.0 - brier(oos["p_m3"], oos["y"]) / brier(oos["clim"], oos["y"])


def test_lagged_features_lose_skill_real_cache(real_feats, cut):
    feats, src = real_feats
    _, tg, _ = cut
    s0 = _pooled_bss_clim(feats, tg, 0)
    s20 = _pooled_bss_clim(feats, tg, 20)
    s250 = _pooled_bss_clim(feats, tg, 250)
    assert s0 > 0.03, (src, s0)
    assert s20 < s0 and s250 < s0, (src, s0, s20, s250)
    assert s250 < 0.25 * s0, (src, s0, s250)


def test_parity_walk_forward_vs_offline(bundle, cut):
    F = pytest.importorskip("mrl.features")
    idx, tg, _ = cut
    feats = F.build_features(bundle, asof=idx[-1])
    oos, params = C.walk_forward(feats, tg)
    m3 = {p["refit_date"]: p for p in params if p["rung"] == "M3"}
    valid = oos.index[oos["p_m3"].notna()]
    rng = np.random.default_rng(0)
    for t in rng.choice(valid, size=10, replace=False):
        t = pd.Timestamp(t)
        R = [r for r in m3 if pd.Timestamp(r) <= t][-1]
        p = m3[R]
        m = M.LogitModel(features=tuple(p["features"]), coef=p["coef"], intercept=p["intercept"], refit_date=R,
                         train_start=p["train_start"], train_end=p["train_end"], n_train=p["n_train"], n_pos=p["n_pos"],
                         clim=p["clim"], feature_rule="", spec_sha256="0" * 64, model_id=p["model_id"])
        row = F.build_features(bundle, asof=t).loc[t]
        assert abs(m.predict_one(row) - oos.loc[t, "p_m3"]) < 1e-12, t


# ------------------------------------------------------------------
# 채점·검정
# ------------------------------------------------------------------
def test_block_scores(wf):
    oos, _ = wf
    bs = C.block_scores(oos, BLOCKS_24, "p_m3")
    assert list(bs.columns) == list(C.BLOCK_SCORE_COLUMNS)
    assert len(bs) == 12 and bs["block"].iloc[-1] == "all"
    blk = bs.iloc[:-1]
    assert blk["n"].tolist() == SESSIONS_24[:-1] + [398]         # 마지막 블록: 하드컷 꼬리 20행 라벨 없음
    assert (blk["n_blocks"] == blk["n"] // 20).all()
    assert bs["n"].iloc[-1] == 5433 and bs["n_blocks"].iloc[-1] == 271
    assert (bs["brier"] > 0).all() and bs["auc"].between(0.5, 1.0).all()
    assert bs["bss_m1"].iloc[-1] > 0 and bs["bss_clim"].iloc[-1] > 0     # 합성: M3 가 진짜 모델
    assert bs["calib_in_large"].iloc[-1] == pytest.approx(bs["mean_p"].iloc[-1] - bs["base"].iloc[-1])
    with pytest.warns(RuntimeWarning):
        bs99 = C.block_scores(oos, BLOCKS_24_FROM_1999, "p_m3")
    assert bs99["n"].iloc[0] == 0 and math.isnan(bs99["brier"].iloc[0]) and len(bs99) == 14
    rt = C.rung_table(oos)
    assert list(rt.index) == ["M0", "M1", "M2", "M3"] and rt.loc["M1", "bss_m1"] == 0.0
    with pytest.raises(ValueError):
        C.block_scores(oos, BLOCKS_24, "p_m9")


def test_loss_diff_ci_sign():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2003-01-02", periods=2000)
    y = pd.Series((rng.uniform(size=2000) < 0.2).astype(float), index=idx)
    p_good = pd.Series(np.clip(0.2 + 0.5 * (y - 0.2) + rng.normal(0, 0.05, 2000), 0.01, 0.99), index=idx)
    p_bad = pd.Series(0.5, index=idx)
    r = C.loss_diff_ci(p_bad, p_good, y)
    assert r["mean"] > 0 and r["lo"] > 0 and r["lo"] <= r["mean"] <= r["hi"]
    assert r["n"] == 2000 and r["n_blocks"] == 100 and r["block"] == 40 and r["n_boot"] == 4000
    r2 = C.loss_diff_ci(p_good, p_bad, y)
    assert r2["mean"] == pytest.approx(-r["mean"]) and r2["hi"] < 0
    assert C.loss_diff_ci(p_good, p_bad, y) == r2                         # seed 0 결정론
    e = C.loss_diff_ci(p_good, p_bad, y * np.nan)
    assert e["n"] == 0 and math.isnan(e["mean"])


def test_dm_test_hac_variance_positive():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2003-01-02", periods=1500)
    la = pd.Series(rng.normal(0.15, 0.05, 1500) ** 2, index=idx)
    lb = la * 0.9 + rng.normal(0, 0.002, 1500)
    r = C.dm_test(la, lb, lag=19)
    assert r["var_hac"] > 0 and math.isfinite(r["t"]) and 0 <= r["p"] <= 1 and r["lag"] == 19 and r["n"] == 1500
    assert r["t"] > 0 and r["mean"] > 0
    assert C.dm_test(la.to_numpy(), lb.to_numpy())["t"] == pytest.approx(r["t"])
    with pytest.warns(RuntimeWarning):
        z = C.dm_test(la, la)                                              # 손실 차 상수 0 → 분산 0
    assert math.isnan(z["t"])
    with pytest.warns(RuntimeWarning):
        assert math.isnan(C.dm_test(la.iloc[:10], lb.iloc[:10])["t"])


def test_phase_offset_skill_has_20_rows(wf):
    oos, _ = wf
    df = C.phase_offset_skill(oos["p_m3"], oos["y"], oos["clim"])
    assert len(df) == 20 and df["offset"].tolist() == list(range(20))
    assert df["n"].sum() == 5433 and (df["n"].between(271, 272)).all()
    s = df.attrs["summary"]
    assert s["min"] <= s["median"] <= s["max"] and 0 <= s["share_pos"] <= 1 and s["n_offsets"] == 20
    assert C.phase_summary(df) == s


def test_reliability_table_wilson_uses_n_over_20(wf):
    oos, _ = wf
    rt = C.reliability_table(oos["p_m3"], oos["y"])
    assert len(rt) == 9 and rt["lo"].tolist() == list(P2["reliability_bins"][:-1])
    assert rt["n"].sum() == 5433
    for r in rt.itertuples():
        if r.n == 0:
            assert math.isnan(r.obs)
            continue
        assert r.n_eff == pytest.approx(r.n / 20.0)
        lo, hi = C.wilson(r.obs, r.n / 20.0)
        assert r.wilson_lo == pytest.approx(lo) and r.wilson_hi == pytest.approx(hi)
        lo_full, hi_full = C.wilson(r.obs, float(r.n))
        assert (hi - lo) > (hi_full - lo_full)                             # n/20 이 더 넓다
        assert r.lo <= r.mean_p < r.hi or (r.hi == 1.0 and r.mean_p <= 1.0)
    assert C.wilson(0.5, 0.0) == (pytest.approx(math.nan, nan_ok=True), pytest.approx(math.nan, nan_ok=True))
    lo, hi = C.wilson(0.0, 5.0)
    assert lo == 0.0 and 0 < hi < 0.5


def test_murphy_decomposition_identity(wf):
    oos, _ = wf
    d = C.murphy_decomposition(oos["p_m3"], oos["y"])
    assert d["n"] == 5433 and d["n_blocks"] == 271
    assert d["brier"] == pytest.approx(brier(oos["p_m3"], oos["y"]))
    assert d["brier_binned"] == pytest.approx(d["reliability"] - d["resolution"] + d["uncertainty"], abs=1e-12)   # 구간 평균 예보: 정확
    assert d["brier"] == pytest.approx(d["brier_binned"] + d["residual"], abs=1e-12)
    assert abs(d["residual"]) < 0.01
    assert d["reliability"] >= 0 and d["resolution"] >= 0 and 0 < d["uncertainty"] <= 0.25


def test_era_auc(wf, synth_cut):
    feats, y = synth_cut
    oos, _ = wf
    df = C.era_auc(feats, oos, y=y)
    assert len(df) == 5 and df["era"].iloc[0].startswith("1993-01-29")
    assert df["n_raw"].iloc[0] > 1000 and df["n_oos"].iloc[0] == 0 and math.isnan(df["auc_p_m3"].iloc[0])
    assert df["auc_x_vix"].iloc[1:].between(0.5, 1.0).all() and df["auc_p_m3"].iloc[1:].between(0.5, 1.0).all()
    df2 = C.era_auc(feats, oos)                                            # y 없으면 oos.y 만
    assert df2["n_raw"].iloc[0] == 0


def test_ladder_table_and_m0_step(wf):
    oos, _ = wf
    lad = C.ladder_table(oos, BLOCKS_24, n_boot=400)
    assert list(lad.columns) == list(C.LADDER_COLUMNS)
    assert len(lad) == len(C.STEPS) * 12
    assert set(lad["step"]) == set(C.STEP_LABELS) and "M0->M1" in set(lad["step"])
    allrows = lad[lad["block"] == "all"].set_index("step")
    assert allrows.loc["M0->M1", "n"] == 5433
    assert allrows.loc["M0->M1", "brier_from"] == pytest.approx(brier(oos["p_vix"], oos["y"]))      # M0 == p_vix
    assert allrows.loc["B1->M3", "brier_from"] == allrows.loc["M0->M1", "brier_from"]
    assert allrows.loc["clim->M3", "brier_from"] == pytest.approx(brier(oos["clim"], oos["y"]))
    assert (allrows["lo"] <= allrows["mean"]).all() and (allrows["mean"] <= allrows["hi"]).all()
    assert allrows.loc["clim->M3", "lo"] > 0 and allrows.loc["M1->M3", "lo"] > 0                   # 합성: 정보 이득 실재
    assert allrows["dm_t"].notna().all() and allrows["phase_share_pos"].between(0, 1).all()
    with pytest.warns(RuntimeWarning):
        lad2 = C.ladder_table(oos.drop(columns=["p_m2"]), BLOCKS_24, n_boot=100)
    assert set(lad2["step"]) == set(C.STEP_LABELS) - {"M1->M2", "M2->M3", "clim->M2", "B1->M2"}


# ------------------------------------------------------------------
# 수용 판정 (합성 블록 표)
# ------------------------------------------------------------------
def _bs_frame(bss_clim, bss_vix, pooled_clim, pooled_vix):
    rows = []
    for k, (a, b) in enumerate(BLOCKS_24, start=1):
        rows.append({"block": k, "start": a, "end": b, "n": 500, "n_blocks": 25, "n_pos": 80, "base": 0.16, "mean_p": 0.17,
                     "brier": 0.12, "bss_clim": bss_clim[k - 1], "bss_vix": bss_vix[k - 1], "bss_vix_bgk": 0.1, "bss_m1": 0.0, "auc": 0.65})
    rows.append({"block": "all", "start": BLOCKS_24[0][0], "end": BLOCKS_24[-1][1], "n": 5433, "n_blocks": 271, "n_pos": 839,
                 "base": 0.154, "mean_p": 0.16, "brier": 0.12, "bss_clim": pooled_clim, "bss_vix": pooled_vix, "bss_vix_bgk": 0.1,
                 "bss_m1": 0.0, "auc": 0.68})
    return pd.DataFrame(rows)


def _ladder_frame(lo: dict):
    rows = []
    for (a, b) in C.STEPS:
        v = lo.get(f"{a}->{b}", 0.001)
        rows.append({"step": f"{a}->{b}", "from": a, "to": b, "block": "all", "n": 5433, "n_blocks": 271, "mean": v + 0.001,
                     "lo": v, "hi": v + 0.01, "dm_t": 2.0, "dm_p": 0.05})
    return pd.DataFrame(rows)


def test_acceptance_literal_pass_selects_m3():
    good = _bs_frame([0.05] * 11, [0.1] * 11, 0.08, 0.2)
    lad = _ladder_frame({})
    acc = C.acceptance({"M1": good, "M2": good, "M3": good}, lad, None, rule="literal")
    assert acc["literal"]["M3"]["pass"] and acc["literal"]["M3"]["failing_blocks"] == []
    assert acc["tone_model"] == "M3" and acc["deploy_mode"] == "tones" and acc["rule"] == "literal"
    assert acc["n_blocks"] == 11 and acc["min_pass_blocks"] == 8
    # DataFrame 하나만 주면 후보(M3)로 해석
    acc1 = C.acceptance(good, lad, None)
    assert acc1["tone_model"] == "M3"


def test_acceptance_literal_fail_amended_picks_highest_passing_rung():
    clim3 = [0.05] * 11
    clim3[8] = -0.038                                  # 2019-20 실패
    vix3 = [0.1] * 11
    vix3[2] = -0.08                                    # 2007-08 실패
    m3 = _bs_frame(clim3, vix3, 0.087, 0.20)
    m1 = _bs_frame([0.04] * 10 + [-0.01], [0.1] * 11, 0.08, 0.19)
    m2 = _bs_frame([0.04] * 11, [0.1] * 11, 0.08, 0.19)
    lad = _ladder_frame({"M2->M3": -0.0003, "M1->M2": -0.0002, "M0->M1": 0.02, "clim->M1": 0.005, "clim->M2": 0.005,
                         "clim->M3": 0.005, "B1->M1": 0.02, "B1->M2": 0.02, "B1->M3": 0.02})
    lit = C.acceptance({"M1": m1, "M2": m2, "M3": m3}, lad, None, rule="literal")
    assert not lit["literal"]["M3"]["pass"] and lit["literal"]["M3"]["failing_blocks"] == [3, 9]
    assert lit["literal"]["M3"]["failing_clim"] == [9] and lit["literal"]["M3"]["failing_vix"] == [3]
    assert lit["tone_model"] is None and lit["deploy_mode"] == "info_only"
    assert lit["literal_tone_model"] is None and lit["amended_tone_model"] == "M1"
    amd = C.acceptance({"M1": m1, "M2": m2, "M3": m3}, lad, None, rule="amended")
    assert amd["amended"]["M3"]["A"] and amd["amended"]["M3"]["B"] and not amd["amended"]["M3"]["C"]
    assert amd["amended"]["M2"]["A"] and not amd["amended"]["M2"]["C"]
    assert amd["amended"]["M1"]["pass"] and amd["tone_model"] == "M1" and amd["deploy_mode"] == "tones"
    assert amd["amended"]["M3"]["n_blocks_clim_pos"] == 10 and amd["amended"]["M3"]["n_blocks_vix_nonneg"] == 10
    assert "post hoc" in amd["rationale"] or "post hoc" in amd["post_hoc_note"]


def test_acceptance_all_fail_is_info_only():
    bad = _bs_frame([-0.1] * 11, [-0.1] * 11, -0.05, -0.1)
    lad = _ladder_frame({k: -0.01 for k in C.STEP_LABELS})
    for rule in ("literal", "amended"):
        acc = C.acceptance({"M1": bad, "M2": bad, "M3": bad}, lad, None, rule=rule)
        assert acc["tone_model"] is None and acc["deploy_mode"] == "info_only"
    # 완화안 A: 한 블록이 −0.05 미만이면 실패
    m = _bs_frame([0.05] * 10 + [-0.06], [0.1] * 11, 0.08, 0.2)
    acc = C.acceptance({"M3": m}, _ladder_frame({}), None, rule="amended")
    assert not acc["amended"]["M3"]["A"] and acc["tone_model"] is None
    with pytest.raises(ValueError):
        C.acceptance({"M3": m}, _ladder_frame({}), None, rule="whatever")


# ------------------------------------------------------------------
# v0 참조선 (예산 밖)
# ------------------------------------------------------------------
def test_v0_reference(cut):
    idx, tg, _ = cut
    sub = idx[idx >= pd.Timestamp("2015-01-02")]
    rng = np.random.default_rng(3)
    y = tg.loc[sub, "y_dd5_20"]
    sig = -(y.fillna(0.0) - 0.15) * 0.8 + rng.normal(0, 0.5, len(sub))       # 약한 신호
    rep = pd.DataFrame({"score_d": sig, "score_w": sig * 0.5, "score_m": sig * 0.2}, index=sub)
    rd = C.refit_dates(sub, first="2015-01-02")
    p = C.v0_reference(rep, tg, rd)
    assert p.name == "p_v0ref" and p.index.equals(sub)
    assert p[p.index < pd.Timestamp("2017-01-03")].isna().all()
    assert p[p.index >= pd.Timestamp("2017-01-03")].notna().all()
    assert len(p.attrs["params"]) == 8 and all(q["n_params"] == 2 for q in p.attrs["params"])
    assert p.attrs["params"][0]["refit_date"] == "2017-01-03"
    with pytest.raises(ValueError):
        C.v0_reference(rep.drop(columns=["score_m"]), tg, rd)
