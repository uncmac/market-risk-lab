# -*- coding: utf-8 -*-
"""mrl/features.py 검증 (ARCHITECTURE_PHASE2.md §4·§6·§15).

* 합성 번들: 식 단위 검사(VIX 내재 확률, 반사원리, ffill 한도, HAR 갭 부호, 중첩 σ(x_vix)==p_vix).
* 실캐시(data/): 최초 유효일, **점 원칙**(무작위 T 20개 비트 동일), 특징 AUC(홀드아웃 제외).
  실캐시 테스트는 모든 입력 프레임을 2024-08-30(HOLDOUT_START 직전 세션)에서 **하드컷**한 번들만 쓴다 — 홀드아웃 보호.
실행: python -m pytest tests/test_features.py -q -s   (AUC 수치는 -s 로 출력)
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

from mrl import features as F  # noqa: E402
from mrl import vol as V  # noqa: E402
from mrl.config import DATA_DIR, HOLDOUT_START, P2  # noqa: E402
from mrl.data import Bundle  # noqa: E402

HAS_CACHE = (DATA_DIR / "close.csv").exists() and (DATA_DIR / "spy_ohlc.csv").exists()
needs_cache = pytest.mark.skipif(not HAS_CACHE, reason="data/ 캐시 없음")
PRE_HOLDOUT_END = (pd.Timestamp(HOLDOUT_START) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")   # 2024-08-31 → .loc 는 08-30 까지

CONTRACT_FEATURE_RULE = ("p2|x_vix=logit(refl(VIX,dd=0.05,h=20,drift=-s2/2))"
                         "|x_har=mean_log(rv1,rv5,rv22;GK+OV;floor=1e-8)-ln(VIX/100)|x_ma=C/SMA180-1"
                         "|vix_ffill<=3|purge=20|C=1.0|train_start=1993-10-14")
CONTRACT_COLUMNS = ("vix", "p_vix", "p_vix_driftless", "p_vix_bgk", "x_vix", "var_gkov", "rv1", "rv5", "rv22",
                    "har_vol_20", "x_har", "sma180", "x_ma", "rv20_cc", "ts_diag")
# §6 값 (VIX → B1 / 무드리프트 / BGK)
B1_VALUES = {12: 0.133, 16: 0.262, 20: 0.372, 30: 0.558, 45: 0.703}
DRIFTLESS_VALUES = {12: 0.129, 15: 0.225, 20: 0.363, 30: 0.544}
BGK_VALUES = {12: 0.102, 16: 0.211, 20: 0.307, 30: 0.475, 45: 0.613}


# ------------------------------------------------------------------
# 합성 번들
# ------------------------------------------------------------------
def _synthetic_ohlc(n: int = 400, seed: int = 0, start: str = "2010-01-04") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n, name="date")
    r = rng.normal(0.0002, 0.011, n)
    close = 100.0 * np.exp(np.cumsum(r))
    prev = np.r_[close[0] / np.exp(r[0]), close[:-1]]
    gap = rng.normal(0.0, 0.004, n)
    open_ = prev * np.exp(gap)
    hi_ext = np.abs(rng.normal(0.0, 0.005, n))
    lo_ext = np.abs(rng.normal(0.0, 0.005, n))
    high = np.maximum(open_, close) * np.exp(hi_ext)
    low = np.minimum(open_, close) * np.exp(-lo_ext)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1e6}, index=idx)


def _make_bundle(ohlc: pd.DataFrame, vix: pd.Series, vix3m: pd.Series | None = None, extra_close_rows=None) -> Bundle:
    close = pd.DataFrame({"SPY": ohlc["Close"], "^VIX": vix.reindex(ohlc.index)})
    if extra_close_rows is not None:
        close = pd.concat([close, extra_close_rows]).sort_index()
    close.index.name = "date"
    cboe = pd.DataFrame({"VIX3M": vix3m if vix3m is not None else pd.Series(np.nan, index=ohlc.index),
                         "VIX9D": np.nan, "VVIX": np.nan, "SKEW": np.nan}, index=ohlc.index)
    empty = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    return Bundle(close=close, spy_ohlc=ohlc, cboe=cboe, fg=empty, eod=empty, meta={"warnings": []})


def _synthetic_bundle(n: int = 400, seed: int = 0) -> Bundle:
    ohlc = _synthetic_ohlc(n, seed)
    rng = np.random.default_rng(seed + 1)
    vix = pd.Series(np.clip(18.0 + np.cumsum(rng.normal(0, 0.6, n)), 9.0, 80.0), index=ohlc.index)
    vix3m = vix * 1.05
    return _make_bundle(ohlc, vix, vix3m)


def _cut_bundle(b: Bundle, end) -> Bundle:
    """모든 입력 프레임 하드컷 (라벨 마스크가 아니라 데이터 자체를 자른다)."""
    end = pd.Timestamp(end)
    return Bundle(close=b.close.loc[:end], spy_ohlc=b.spy_ohlc.loc[:end], cboe=b.cboe.loc[:end],
                  fg=b.fg.loc[:end] if len(b.fg) else b.fg, eod=b.eod.loc[:end] if len(b.eod) else b.eod,
                  meta=dict(b.meta))


def _frames_bit_identical(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if not a.index.equals(b.index) or list(a.columns) != list(b.columns):
        return False
    return all(np.array_equal(a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float), equal_nan=True) for c in a.columns)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@pytest.fixture(scope="module")
def bundle_pre_holdout() -> Bundle:
    if not HAS_CACHE:
        pytest.skip("data/ 캐시 없음")
    from mrl.data import load_cache
    b = load_cache()
    cut = _cut_bundle(b, PRE_HOLDOUT_END)
    assert cut.spy_ohlc.index[-1] < pd.Timestamp(HOLDOUT_START)
    return cut


@pytest.fixture(scope="module")
def feats_real(bundle_pre_holdout) -> pd.DataFrame:
    return F.build_features(bundle_pre_holdout)


# ------------------------------------------------------------------
# 상수·사양
# ------------------------------------------------------------------
def test_constants_match_contract():
    assert F.FEATURE_RULE == CONTRACT_FEATURE_RULE
    assert F.FEATURE_COLUMNS == CONTRACT_COLUMNS
    assert F.MODEL_INPUTS == ("x_vix", "x_har", "x_ma") == tuple(P2["features"])
    for col, spec in F.FEATURE_SPEC.items():
        for k in ("source", "columns", "lookback", "first_date", "formula", "label_ko", "model_input"):
            assert k in spec, (col, k)
        pd.Timestamp(spec["first_date"])
    assert [c for c, s in F.FEATURE_SPEC.items() if s["model_input"]] == list(F.MODEL_INPUTS)
    assert F.FEATURE_SPEC["x_vix"]["first_date"] == "1993-01-29"
    assert F.FEATURE_SPEC["x_har"]["first_date"] == "1993-03-03"
    assert F.FEATURE_SPEC["x_ma"]["first_date"] == "1993-10-14"


def test_spec_sha256_is_hex_and_deterministic(tmp_path):
    h1 = F.spec_sha256()
    h2 = F.spec_sha256()
    assert h1 == h2 and len(h1) == 64 and all(c in "0123456789abcdef" for c in h1)
    # 내용이 바뀌면 해시가 바뀌고, CRLF/LF 는 같은 해시
    p = tmp_path / "a.py"
    p.write_bytes(b"x = 1\ny = 2\n")
    d1, m1 = F._hash_sources([p])
    p.write_bytes(b"x = 1\r\ny = 2\r\n")
    d2, _ = F._hash_sources([p])
    p.write_bytes(b"x = 1\ny = 3\n")
    d3, _ = F._hash_sources([p])
    assert d1 == d2 and d1 != d3 and m1 == []
    # 없는 파일은 표식 + 목록
    d4, m4 = F._hash_sources([p, tmp_path / "missing.py"])
    assert m4 == ["missing.py"] and d4 != d3


# ------------------------------------------------------------------
# VIX 내재 확률 식 (§6)
# ------------------------------------------------------------------
@pytest.mark.parametrize("vix,expected", list(B1_VALUES.items()))
def test_vix_implied_prob_b1_known_values(vix, expected):
    assert abs(F.vix_implied_prob(vix) - expected) < 1e-3


@pytest.mark.parametrize("vix,expected", list(DRIFTLESS_VALUES.items()))
def test_vix_implied_prob_driftless_known_values(vix, expected):
    p = F.vix_implied_prob(vix, drift="none")
    assert abs(p - expected) < 1e-3
    # m=0 이면 식이 정확히 2Φ(b/s) 로 준다
    s = vix / 100 * math.sqrt(20 / 252)
    from scipy.special import ndtr
    assert abs(p - 2 * ndtr(math.log(0.95) / s)) < 1e-12


@pytest.mark.parametrize("vix,expected", list(BGK_VALUES.items()))
def test_vix_implied_prob_bgk_known_values(vix, expected):
    assert abs(F.vix_implied_prob(vix, monitoring="bgk") - expected) < 1e-3


def test_vix_implied_prob_explicit_formula():
    """식을 손으로 옮겨 적은 값과 1e-12 안에서 같다 (VIX 20)."""
    from scipy.special import ndtr
    V, T = 0.20, 20 / 252
    s = V * math.sqrt(T)
    b = math.log(0.95)
    m = -s * s / 2
    p = ndtr((b - m) / s) + math.exp(2 * m * b / s ** 2) * ndtr((b + m) / s)
    assert abs(F.vix_implied_prob(20.0) - p) < 1e-12
    b2 = b - 0.5826 * V * math.sqrt(1 / 252)
    p2 = ndtr((b2 - m) / s) + math.exp(2 * m * b2 / s ** 2) * ndtr((b2 + m) / s)
    assert abs(F.vix_implied_prob(20.0, monitoring="bgk") - p2) < 1e-12


def test_vix_implied_prob_monotone_and_barrier_ordering():
    grid = np.linspace(5.0, 120.0, 400)
    p_b1 = F.vix_implied_prob(grid)
    p_none = F.vix_implied_prob(grid, drift="none")
    p_bgk = F.vix_implied_prob(grid, monitoring="bgk")
    for p in (p_b1, p_none, p_bgk):
        assert np.all(np.diff(p) > 0), "VIX 에 대해 단조 증가"
        assert np.all((p > 0) & (p < 1))
    # 장벽 관계: 이산 관측 보정(장벽을 더 멀리)은 연속 관측보다 작다. 드리프트 관계: m=−s²/2 (음수) 는 하향 도달을 쉽게 하므로
    # B1(martingale) > 무드리프트 — §6 의 값(20: 0.372 > 0.363)과 일치. (§15 의 문구 "bgk < martingale < driftless" 는 §6 값과 모순이라
    # 값 쪽을 따른다: bgk < driftless < martingale.)
    assert np.all(p_bgk < p_none) and np.all(p_none < p_b1), "bgk < driftless < martingale(B1)"
    assert np.all(p_bgk < p_b1)
    assert np.all(p_b1 - p_none < 0.03), "B1 과 무드리프트 차이는 작다 (VIX 12~30 에서 <1.5pp, §6)"
    assert np.all((p_b1 - p_none)[grid <= 20.0] < 0.01)


def test_reflection_principle_min_ge_terminal():
    """반사원리 sanity: P(min ≤ −5%) ≥ P(terminal ≤ −5%) (드리프트 유무 모두), 차이는 반사 항."""
    from scipy.special import ndtr
    for vix in (8.0, 12.0, 20.0, 30.0, 45.0, 80.0):
        for drift in ("martingale", "none"):
            V = vix / 100
            s = V * math.sqrt(20 / 252)
            b = math.log(0.95)
            m = -s * s / 2 if drift == "martingale" else 0.0
            p_terminal = float(ndtr((b - m) / s))
            p_min = F.vix_implied_prob(vix, drift=drift)
            assert p_min >= p_terminal
            assert abs((p_min - p_terminal) - math.exp(2 * m * b / s ** 2) * ndtr((b + m) / s)) < 1e-12
    # 무드리프트에선 정확히 2배
    assert abs(F.vix_implied_prob(20.0, drift="none") / float(ndtr(math.log(0.95) / (0.2 * math.sqrt(20 / 252)))) - 2.0) < 1e-12


def test_vix_implied_prob_types_and_errors():
    idx = pd.bdate_range("2020-01-01", periods=4)
    s = pd.Series([12.0, np.nan, 20.0, 45.0], index=idx)
    out = F.vix_implied_prob(s)
    assert isinstance(out, pd.Series) and out.index.equals(idx)
    assert np.isnan(out.iloc[1]) and abs(out.iloc[0] - B1_VALUES[12]) < 1e-3
    arr = F.vix_implied_prob(np.array([12.0, 20.0]))
    assert isinstance(arr, np.ndarray) and arr.shape == (2,)
    assert isinstance(F.vix_implied_prob(20), float)
    with pytest.raises(ValueError):
        F.vix_implied_prob(0.0)
    with pytest.raises(ValueError):
        F.vix_implied_prob(pd.Series([12.0, -1.0]))
    with pytest.raises(ValueError):
        F.vix_implied_prob(np.inf)
    with pytest.raises(ValueError):
        F.vix_implied_prob(20.0, drift="risk-neutral")
    with pytest.raises(ValueError):
        F.vix_implied_prob(20.0, monitoring="weekly")
    with pytest.raises(ValueError):
        F.vix_implied_prob(20.0, dd=1.5)
    with pytest.raises(ValueError):
        F.vix_implied_prob(20.0, h=0)


def test_bgk_matches_daily_monte_carlo():
    """(slow) 200k 경로 × 20일 일별 GBM(seed 0)의 20종가 최저값 확률이 BGK 와 ±0.005 (VIX 12/20/30)."""
    rng = np.random.default_rng(0)
    n, h = 200_000, 20
    for vix in (12, 20, 30):
        sd = vix / 100 / math.sqrt(252)
        z = rng.standard_normal((n, h))
        lp = np.cumsum(-0.5 * sd ** 2 + sd * z, axis=1)
        mc = float((lp.min(axis=1) <= math.log(0.95)).mean())
        p_bgk = F.vix_implied_prob(vix, monitoring="bgk")
        p_b1 = F.vix_implied_prob(vix)
        assert abs(mc - p_bgk) < 0.005, (vix, mc, p_bgk)
        assert p_b1 > mc, "연속 관측 식은 이산 관측 확률보다 크다"


# ------------------------------------------------------------------
# VIX 정렬 (ffill 한도·유령 행)
# ------------------------------------------------------------------
def test_align_vix_ffill_limit_and_phantom_rows():
    ohlc = _synthetic_ohlc(60, seed=2)
    idx = ohlc.index
    vix = pd.Series(20.0 + np.arange(60) * 0.1, index=idx)
    # 결측: 2세션(10,11) → 채움 ; 5세션(30..34) → 앞 3개 채움, 뒤 2개 NaN
    vix.iloc[[10, 11]] = np.nan
    vix.iloc[30:35] = np.nan
    # 유령 행: 토요일에 VIX 값
    sat = idx[20] + pd.Timedelta(days=5 - idx[20].weekday())
    assert sat.weekday() == 5
    ghost = pd.DataFrame({"SPY": np.nan, "^VIX": 99.0}, index=pd.DatetimeIndex([sat]))
    b = _make_bundle(ohlc, vix, extra_close_rows=ghost)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v, notes = F.align_vix(b.close, idx, limit=3)
    assert v.index.equals(idx) and v.name == "vix"
    assert v.iloc[10] == v.iloc[9] and v.iloc[11] == v.iloc[9]
    assert v.iloc[30] == v.iloc[29] and v.iloc[32] == v.iloc[29]
    assert np.isnan(v.iloc[33]) and np.isnan(v.iloc[34])
    assert v.iloc[35] == vix.iloc[35]
    assert not (v == 99.0).any(), "유령 행은 무시"
    assert any("유령" in n or "SPY 세션이 아닌" in n for n in notes)
    assert any("채움" in n for n in notes) and any("초과" in n for n in notes)
    assert len(w) >= 2
    # limit=0 → 채우지 않음
    v0, _ = F.align_vix(b.close, idx, limit=0)
    assert v0.isna().sum() == 7
    # 최초 관측 이전은 채우지 않는다
    vix2 = vix.copy()
    vix2.iloc[:5] = np.nan
    v2, notes2 = F.align_vix(_make_bundle(ohlc, vix2).close, idx)
    assert v2.iloc[:5].isna().all() and any("최초 관측" in n for n in notes2)
    with pytest.raises(ValueError):
        F.align_vix(b.close.drop(columns="^VIX"), idx)


def test_build_features_propagates_vix_gap_as_nan_not_fabricated():
    ohlc = _synthetic_ohlc(300, seed=3)
    vix = pd.Series(20.0, index=ohlc.index)
    vix.iloc[250:256] = np.nan          # 6세션 결측 → 3 채움 + 3 NaN
    b = _make_bundle(ohlc, vix)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feats = F.build_features(b)
    assert feats["x_vix"].iloc[250:253].notna().all()
    assert feats["x_vix"].iloc[253:256].isna().all()
    assert feats["x_har"].iloc[253:256].isna().all()
    ok, why = F.input_status(feats.iloc[254])
    assert ok is False and "x_vix" in why and "확률 계산 불가" in why
    assert any("초과" in n for n in feats.attrs["warnings"])


# ------------------------------------------------------------------
# build_features (합성)
# ------------------------------------------------------------------
def test_build_features_columns_attrs_and_nesting_synthetic():
    b = _synthetic_bundle(400)
    feats = F.build_features(b)
    assert tuple(feats.columns) == CONTRACT_COLUMNS
    assert feats.index.equals(b.spy_ohlc.index)
    for k in ("feature_rule", "spec_sha256", "warnings"):
        assert k in feats.attrs
    assert feats.attrs["feature_rule"] == F.FEATURE_RULE
    # 중첩: σ(x_vix) == p_vix
    m = feats["p_vix"].notna()
    assert np.max(np.abs(_sigmoid(feats.loc[m, "x_vix"]) - feats.loc[m, "p_vix"])) < 1e-12
    # 정의 대조
    assert np.allclose(feats["x_ma"].dropna(), (feats["sma180"].pow(-1) * b.spy_ohlc["Close"] - 1).dropna(), rtol=0, atol=1e-12)
    assert np.allclose(feats["har_vol_20"].dropna(), np.exp((np.log(feats["rv1"]) + np.log(feats["rv5"]) + np.log(feats["rv22"])) / 3).dropna(),
                       rtol=1e-12, atol=0)
    assert np.allclose(feats["x_har"].dropna(), (np.log(feats["har_vol_20"]) - np.log(feats["vix"] / 100)).dropna(), rtol=0, atol=1e-12)
    assert np.allclose(feats["ts_diag"].dropna(), (feats["vix"] / b.cboe["VIX3M"]).dropna(), rtol=0, atol=1e-12)
    # NaN 꼬리: 첫 179행 x_ma NaN, 180행부터 유효
    assert feats["x_ma"].iloc[:179].isna().all() and feats["x_ma"].iloc[179:].notna().all()
    assert feats["x_har"].iloc[:22].isna().all() and feats["x_har"].iloc[22:].notna().all()
    assert feats["x_vix"].notna().all()
    # 백분위 열 없음
    assert not any(c.startswith("pct_") for c in feats.columns)


def test_har_gap_sign_convention():
    """일간 분산이 상수 (0.2)^2/252 이면 σ_har = 0.2: VIX 20 → x_har = 0, VIX 15 → 양(+), VIX 30 → 음(−)."""
    n = 60
    idx = pd.bdate_range("2015-01-05", periods=n, name="date")
    a = 0.2 / math.sqrt(504)                     # 2a² = σ²/252
    c = 100.0
    ohlc = pd.DataFrame({"Open": c, "High": c * math.exp(a), "Low": c * math.exp(-a), "Close": c, "Volume": 1.0}, index=idx)
    for vix, sign in ((20.0, 0), (15.0, +1), (30.0, -1)):
        b = _make_bundle(ohlc, pd.Series(vix, index=idx))
        feats = F.build_features(b)
        xh = feats["x_har"].iloc[-1]
        assert abs(feats["har_vol_20"].iloc[-1] - 0.2) < 1e-9
        if sign == 0:
            assert abs(xh) < 1e-9
        else:
            assert np.sign(xh) == sign
            assert abs(xh - (math.log(0.2) - math.log(vix / 100))) < 1e-9


def test_asof_semantics_and_hard_cut_equivalence_synthetic():
    b = _synthetic_bundle(400, seed=5)
    full = F.build_features(b)
    T = b.spy_ohlc.index[250]
    cut = F.build_features(b, asof=T)
    assert cut.index[-1] == T and _frames_bit_identical(cut, full.loc[:T])
    assert cut.attrs["asof"] == T.strftime("%Y-%m-%d")
    # 세션이 아닌 asof(토요일) → 그 이전 마지막 세션까지
    sat = T + pd.Timedelta(days=5 - T.weekday())
    cut2 = F.build_features(b, asof=sat)
    assert cut2.index[-1] == full.loc[:sat].index[-1] and _frames_bit_identical(cut2, full.loc[:sat])
    # 하드컷 번들 == asof 절단
    hard = F.build_features(_cut_bundle(b, T))
    assert _frames_bit_identical(hard, cut)
    with pytest.raises(ValueError):
        F.build_features(b, asof="1990-01-01")
    with pytest.raises(ValueError):
        F.build_features(b, asof=pd.Timestamp("2010-06-01", tz="UTC"))


def test_factor_percentiles_display_only():
    b = _synthetic_bundle(700, seed=6)
    feats = F.build_features(b)
    pct = F.factor_percentiles(feats, window=2520, min_periods=252)
    assert list(pct.columns) == [f"pct_{c}" for c in F.PERCENTILE_COLUMNS]
    assert pct.index.equals(feats.index)
    v = pct["pct_x_vix"].dropna()
    assert len(v) > 0 and ((v > 0) & (v <= 1)).all()
    assert pct["pct_x_vix"].iloc[:251].isna().all()
    assert not set(pct.columns) & set(feats.columns)
    with pytest.raises(ValueError):
        F.factor_percentiles(feats[["vix"]].rename(columns={"vix": "zzz"}))


def test_input_status():
    row = pd.Series({"x_vix": 0.1, "x_har": -0.2, "x_ma": 0.03})
    assert F.input_status(row) == (True, "")
    ok, why = F.input_status(pd.Series({"x_vix": np.nan, "x_har": -0.2, "x_ma": 0.03}))
    assert not ok and "x_vix" in why and "x_har" not in why
    ok, why = F.input_status(pd.Series({"x_vix": 0.1, "x_har": np.inf, "x_ma": np.nan}))
    assert not ok and "x_har" in why and "x_ma" in why
    ok, why = F.input_status(pd.Series({"vix": 20.0}))
    assert not ok and all(k in why for k in F.MODEL_INPUTS)
    with pytest.raises(TypeError):
        F.input_status({"x_vix": 0.1})


def test_ma_distance():
    idx = pd.bdate_range("2020-01-01", periods=200)
    c = pd.Series(np.linspace(100, 120, 200), index=idx)
    d = F.ma_distance(c, window=180)
    assert d.iloc[:179].isna().all() and d.iloc[179:].notna().all()
    assert abs(d.iloc[179] - (c.iloc[179] / c.iloc[:180].mean() - 1)) < 1e-12
    with pytest.raises(ValueError):
        F.ma_distance(pd.Series([1.0, -1.0], index=idx[:2]))


# ------------------------------------------------------------------
# 실캐시 (홀드아웃 제외 하드컷)
# ------------------------------------------------------------------
@needs_cache
def test_first_available_dates_real_cache(feats_real):
    first = {c: feats_real[c].first_valid_index().strftime("%Y-%m-%d") for c in feats_real.columns}
    assert first["x_vix"] == "1993-01-29"
    assert first["x_har"] == "1993-03-03"
    assert first["x_ma"] == "1993-10-14"
    for c, spec in F.FEATURE_SPEC.items():
        assert first[c] == spec["first_date"], (c, first[c], spec["first_date"])
        before = feats_real.loc[: pd.Timestamp(spec["first_date"]) - pd.Timedelta(days=1), c]
        assert before.isna().all(), c
    # 최초 유효일 이후 모델 입력은 결측이 없다 (VIX 결측 3세션 초과가 없음)
    assert feats_real.loc["1993-10-14":, list(F.MODEL_INPUTS)].notna().all().all()
    assert feats_real.index[-1] == pd.Timestamp("2024-08-30")
    assert np.max(np.abs(_sigmoid(feats_real["x_vix"]) - feats_real["p_vix"])) < 1e-12


@needs_cache
def test_point_in_time_truncation(bundle_pre_holdout, feats_real):
    """무작위 T 20개: build_features(bundle, asof=T) 와 하드컷 번들 계산이 전체 계산의 [:T] 와 NaN 포함 비트 동일."""
    rng = np.random.default_rng(0)
    idx = feats_real.index
    picks = idx[np.sort(rng.choice(len(idx), 20, replace=False))]
    for T in picks:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            by_asof = F.build_features(bundle_pre_holdout, asof=T)
            by_cut = F.build_features(_cut_bundle(bundle_pre_holdout, T))
        ref = feats_real.loc[:T]
        assert by_asof.index[-1] == T
        assert _frames_bit_identical(by_asof, ref), f"asof 절단 불일치 T={T:%Y-%m-%d}"
        assert _frames_bit_identical(by_cut, ref), f"하드컷 번들 불일치 T={T:%Y-%m-%d}"


@needs_cache
def test_feature_auc_pre_holdout(bundle_pre_holdout, feats_real):
    """특징별 y_dd5_20 AUC (1993-10-14 ~ 2024-08-30, 라벨은 하드컷 자료 안에서만 실현). x_vix ≈ 0.70."""
    from mrl.targets import make_targets
    tg = make_targets(bundle_pre_holdout.close["SPY"])
    y = tg["y_dd5_20"].reindex(feats_real.index)
    assert tg.index[-1] <= pd.Timestamp("2024-08-30")
    out = {}
    for start in ("1993-01-29", "1993-10-14", "1993-11-01"):
        sl = slice(start, "2024-08-30")
        yy = y.loc[sl]
        res = {"n": int((yy.notna()).sum()), "base": float(yy.mean())}
        for name, sc in (("x_vix", feats_real["x_vix"]), ("x_har", feats_real["x_har"]), ("-x_ma", -feats_real["x_ma"]),
                         ("p_vix_bgk", feats_real["p_vix_bgk"]), ("rv20_cc", feats_real["rv20_cc"])):
            res[name] = V.auc(sc.loc[sl].to_numpy(), yy.to_numpy())
        out[start] = res
        print(f"\nAUC vs y_dd5_20 [{start} ~ 2024-08-30] n={res['n']} base={res['base']:.3f}: "
              + " ".join(f"{k}={v:.4f}" for k, v in res.items() if k not in ('n', 'base')))
    r = out["1993-10-14"]
    assert 0.67 <= r["x_vix"] <= 0.73, r
    assert abs(r["x_vix"] - r["p_vix_bgk"]) < 1e-9, "단조 변환이라 AUC 동일"
    assert r["x_har"] > 0.58 and r["-x_ma"] > 0.60
    assert 0.68 <= out["1993-01-29"]["x_vix"] <= 0.72   # VALIDATION §17 #1c: 0.700
