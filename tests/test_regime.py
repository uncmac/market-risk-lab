# -*- coding: utf-8 -*-
"""mrl/regime.py 검증 (ARCHITECTURE_PHASE3.md §4·§13).

* 관측: 손계산 대조, 분산 하한, asof 하드컷, 미래 행 불변(점 원칙).
* EM: 알려진 θ 로 만든 합성 자료 회복(A 오차 < 0.02·변동성 오차 < 5%·μ 부호), 가격경로 합성 회복,
  결정론(두 번 적합 비트 동일; 결정적 초기화·warm start 모두), 라벨 고정, n_params == 12, π == stationary(A).
* 필터: 20개 무작위 절단에서 비트 동일(PIT), 결측 행 NaN·이월 없음, γ_T == α_T.
* 누수 카나리: 평활 확률이 필터보다 **높게** 채점된다(= 필터가 실제로 돌고 있다) — 실캐시 AUC + 합성 상태 정확도.
* 누수 이동: x_hmm 을 미래로 앞당기면 AUC 가 엄격히 오르고, 지연시키면 엄격히 떨어진다.
* guard: 퇴화 θ → False + 사유; walk-forward 에서 θ_{y−1} 유지.
* k-스텝: k=1 이 A 행과 일치, q_k 단조 증가, k=20 닫힌 식 == 몬테카를로(seed 0, 1e-3).
* 실캐시(홀드아웃 하드컷 2024-08-30): 22회 재적합 < 60초, 모든 해 guard 통과, p00/p11 ∈ [0.97, 0.99],
  전이행렬·기대 체류·정상확률·상태 변동성·OOS AUC(P_high, q20) 를 §4.3 설계 단계 실측과 나란히 보고.
실행: python -m pytest tests/test_regime.py -q -s
"""
from __future__ import annotations

import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrl import regime as R  # noqa: E402
from mrl.calibrate import refit_dates, training_mask  # noqa: E402
from mrl.config import DATA_DIR, HMM_P3, HOLDOUT_START, P2  # noqa: E402
from mrl.vol import auc  # noqa: E402

HAS_CACHE = (DATA_DIR / "close.csv").exists()
needs_cache = pytest.mark.skipif(not HAS_CACHE, reason="data/ 캐시 없음")
HARD_CUT = "2024-08-30"                    # 홀드아웃(2024-09-01~) 은 절대 채점하지 않는다
CANARY_MIN = 0.010                         # 평활 − 필터 AUC 최소 격차(§13 은 0.02 — 아래 실측 참조)


# ==================================================================
# 합성 자료
# ==================================================================
def _true_theta() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """알려진 2-상태 2차원 가우시안 HMM θ (관측 단위 = [100·r, ln RV10])."""
    A = np.array([[0.98, 0.02], [0.05, 0.95]])
    mu = np.array([[0.06, math.log(0.010 * math.sqrt(252.0))],
                   [-0.04, math.log(0.025 * math.sqrt(252.0))]])

    def cov(sr, sv, rho):
        return np.array([[sr * sr, rho * sr * sv], [rho * sr * sv, sv * sv]])

    C = np.stack([cov(1.0, 0.25, -0.10), cov(2.5, 0.35, -0.20)])
    return A, mu, C


def _gen_from_theta(n: int = 20000, seed: int = 0):
    """알려진 θ 에서 관측을 생성 (EM 회복 검사의 정공법)."""
    A, mu, C = _true_theta()
    L = [np.linalg.cholesky(C[k]) for k in range(2)]
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    z = rng.standard_normal((n, 2))
    st = np.empty(n, dtype=int)
    X = np.empty((n, 2))
    s = 0
    for t in range(n):
        if t > 0:
            s = 0 if u[t] < A[s, 0] else 1
        st[t] = s
        X[t] = mu[s] + L[s] @ z[t]
    return A, mu, C, st, X


def _gen_price_path(n: int = 20000, seed: int = 0):
    """2-상태 수익률 과정에서 가격 경로 → observations() 를 통과시킨 합성 관측."""
    A = np.array([[0.98, 0.02], [0.05, 0.95]])
    vols = np.array([0.010, 0.025])
    mus = np.array([0.0006, -0.0004])
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    z = rng.standard_normal(n)
    st = np.empty(n, dtype=int)
    s = 0
    for t in range(n):
        if t > 0:
            s = 0 if u[t] < A[s, 0] else 1
        st[t] = s
    r = mus[st] + vols[st] * z
    idx = pd.bdate_range("1960-01-04", periods=n, name="date")
    return A, vols, st, pd.Series(100.0 * np.exp(np.cumsum(r)), index=idx, name="SPY")


def _obs_frame(X: np.ndarray, start: str = "1960-01-04") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(X), name="date")
    return pd.DataFrame(X, columns=list(R.OBS_COLUMNS), index=idx)


@pytest.fixture(scope="module")
def gen_data():
    return _gen_from_theta()


@pytest.fixture(scope="module")
def gen_theta(gen_data):
    _, _, _, _, X = gen_data
    return R.em_fit(X, R.deterministic_init(X), int(HMM_P3["max_iter_first"]))


# ==================================================================
# 실캐시 픽스처 (하드컷 2024-08-30)
# ==================================================================
@pytest.fixture(scope="module")
def cache_close() -> pd.Series:
    if not HAS_CACHE:
        pytest.skip("data/ 캐시 없음")
    from mrl.data import load_cache
    b = load_cache()
    s = b.close["SPY"].dropna().loc[:HARD_CUT]
    assert s.index.max() < pd.Timestamp(HOLDOUT_START), "홀드아웃 행이 입력에 들어왔다"
    return s


@pytest.fixture(scope="module")
def cache_obs(cache_close) -> pd.DataFrame:
    return R.observations(cache_close)


@pytest.fixture(scope="module")
def cache_y(cache_close, cache_obs) -> pd.Series:
    from mrl.targets import make_targets
    return make_targets(cache_close)["y_dd5_20"].reindex(cache_obs.index)


@pytest.fixture(scope="module")
def cache_wf(cache_obs):
    rd = refit_dates(cache_obs.index)
    t0 = time.perf_counter()
    df, thetas = R.hmm_walk_forward(cache_obs, rd)
    return df, thetas, rd, time.perf_counter() - t0


# ==================================================================
# 1. 관측
# ==================================================================
def test_observations_formula():
    """r100 = 100·ln(C_t/C_{t−1}), ln_rv10 = ln√(252·mean(r²_{t−9..t})) 를 손계산과 대조."""
    n = 40
    idx = pd.bdate_range("2010-01-04", periods=n, name="date")
    rng = np.random.default_rng(0)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n))), index=idx, name="SPY")
    obs = R.observations(close, rv_window=10)
    assert list(obs.columns) == list(R.OBS_COLUMNS)
    assert obs.index[0] == idx[10]                                  # 수익률 10개 필요
    r = np.log(close).diff()
    t = idx[20]
    assert obs.loc[t, "r100"] == pytest.approx(100.0 * float(r.loc[t]), abs=0, rel=1e-15)
    win = r.loc[:t].iloc[-10:]
    rv = math.sqrt(252.0 * float((win ** 2).mean()))
    assert obs.loc[t, "ln_rv10"] == pytest.approx(math.log(rv), rel=1e-13)
    assert obs.notna().all().all()


def test_observations_var_floor():
    """평평한 가격(수익률 0) 은 분산 하한 var_floor 로 눌린다 — ln(0) 을 만들지 않는다."""
    idx = pd.bdate_range("2010-01-04", periods=30, name="date")
    close = pd.Series(100.0, index=idx, name="SPY")
    obs = R.observations(close)
    expect = math.log(math.sqrt(252.0 * float(P2["var_floor"])))
    assert np.isfinite(obs["ln_rv10"]).all()
    assert obs["ln_rv10"].iloc[-1] == pytest.approx(expect, rel=1e-12)


def test_observations_point_in_time():
    """미래 행을 덧붙여도 과거 관측은 비트 동일 (asof 하드컷 = 뒤 자르기)."""
    n = 300
    idx = pd.bdate_range("2010-01-04", periods=n, name="date")
    rng = np.random.default_rng(1)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n))), index=idx, name="SPY")
    full = R.observations(close)
    cut = idx[200]
    part = R.observations(close, asof=cut)
    assert part.index[-1] == cut
    assert np.array_equal(part.to_numpy(), full.loc[:cut].to_numpy(), equal_nan=True)


def test_observations_rejects_bad_input():
    idx = pd.bdate_range("2010-01-04", periods=30, name="date")
    with pytest.raises(TypeError):
        R.observations(pd.DataFrame({"SPY": np.ones(30)}, index=idx))
    with pytest.raises(ValueError):
        R.observations(pd.Series(np.r_[np.ones(29), -1.0], index=idx))
    with pytest.raises(ValueError):
        R.observations(pd.Series(np.ones(5), index=idx[:5]))


# ==================================================================
# 2. 파라미터 회계 · 정상분포 · 라벨
# ==================================================================
def test_n_params_and_stationary(gen_theta):
    """n_params == 12 == HMM_P3["param_count"] · π == stationary(A) (1e-12)."""
    assert R.HMM_PARAM_COUNT == 12 == int(HMM_P3["param_count"])
    assert R.n_params(gen_theta) == 12
    assert gen_theta.n_params() == 12
    pi = R.stationary(gen_theta.A_arr)
    assert np.max(np.abs(gen_theta.pi_arr - pi)) < 1e-12
    A = gen_theta.A_arr
    assert np.max(np.abs(pi @ A - pi)) < 1e-12
    assert pytest.approx(1.0, abs=1e-12) == float(pi.sum())


def test_stationary_closed_form():
    A = np.array([[0.98, 0.02], [0.05, 0.95]])
    pi = R.stationary(A)
    assert pi[0] == pytest.approx(0.05 / 0.07, rel=1e-14)
    assert pi[1] == pytest.approx(0.02 / 0.07, rel=1e-14)
    with pytest.raises(ValueError):
        R.stationary(np.eye(2))                                      # 흡수 상태


def test_label_fixing(gen_data):
    """초기 상태를 맞바꿔 넣어도 상태 1 = 수익률 분산이 큰 쪽으로 고정된다."""
    _, _, _, _, X = gen_data
    init = R.deterministic_init(X)
    swapped = R.HMMTheta(A=np.asarray(init.A)[np.ix_([1, 0], [1, 0])], mu=np.asarray(init.mu)[[1, 0]],
                         cov=np.asarray(init.cov)[[1, 0]], pi=R.stationary(np.asarray(init.A)[np.ix_([1, 0], [1, 0])]))
    for th in (R.em_fit(X, init, 100), R.em_fit(X, swapped, 100)):
        assert th.cov_arr[1, 0, 0] > th.cov_arr[0, 0, 0]
        assert th.mu_arr[1, 1] > th.mu_arr[0, 1]                     # 고변동 상태의 ln RV10 가 크다


# ==================================================================
# 3. EM 회복 · 결정론
# ==================================================================
def test_em_recovery_from_known_theta(gen_data, gen_theta):
    """알려진 θ 에서 만든 n=20,000 합성 자료: A 오차 < 0.02, 상태 변동성 오차 < 5%, μ 부호 회복."""
    A_t, mu_t, C_t, st, X = gen_data
    th = gen_theta
    A_err = float(np.max(np.abs(th.A_arr - A_t)))
    sd = np.sqrt(np.array([th.cov_arr[0, 0, 0], th.cov_arr[1, 0, 0]]))
    sd_t = np.sqrt(np.array([C_t[0, 0, 0], C_t[1, 0, 0]]))
    rel = np.abs(sd / sd_t - 1.0)
    print(f"\n[EM recovery] iters={th.n_iter} converged={th.converged} A_err={A_err:.5f} "
          f"sd={sd.round(4).tolist()} (true {sd_t.round(4).tolist()}) rel={rel.round(4).tolist()} "
          f"mu_r={th.mu_arr[:, 0].round(4).tolist()} (true {mu_t[:, 0].tolist()})")
    assert th.converged
    assert A_err < 0.02
    assert float(rel.max()) < 0.05
    assert th.mu_arr[0, 0] > 0 > th.mu_arr[1, 0]                     # μ 부호 회복
    assert float(np.max(np.abs(th.mu_arr - mu_t))) < 0.10


def test_em_recovery_price_path():
    """가격경로 합성(상태 변동성 1%/2.5%): A 오차 < 0.02, 변동성 오차 < 10%, 드리프트 순서 회복.

    관측 두 번째 성분 ln RV10 이 10세션 창이라 상태 경계에서 섞인다 — 고변동 상태의 수익 표준편차가
    체계적으로 ~6% 낮게 추정된다(측정치는 아래 print). 이 편의는 모형이 아니라 관측 설계에서 온다.
    """
    A_t, vols, st, close = _gen_price_path()
    obs = R.observations(close)
    X = obs.to_numpy()
    th = R.em_fit(X, R.deterministic_init(X), int(HMM_P3["max_iter_first"]))
    sd = np.sqrt(np.array([th.cov_arr[0, 0, 0], th.cov_arr[1, 0, 0]])) / 100.0
    rel = sd / vols - 1.0
    A_err = float(np.max(np.abs(th.A_arr - A_t)))
    print(f"\n[EM price-path] iters={th.n_iter} A={th.A_arr.round(4).tolist()} A_err={A_err:.4f} "
          f"sd={sd.round(5).tolist()} rel={rel.round(4).tolist()} mu_r={th.mu_arr[:, 0].round(4).tolist()}")
    assert th.converged and A_err < 0.02
    assert float(np.max(np.abs(rel))) < 0.10
    assert th.mu_arr[0, 0] > th.mu_arr[1, 0]


def test_em_determinism(gen_data):
    """같은 관측에 두 번 적합 → θ 비트 동일. 결정적 초기화와 warm start 모두."""
    _, _, _, _, X = gen_data
    a = R.em_fit(X, R.deterministic_init(X), 100)
    b = R.em_fit(X, R.deterministic_init(X), 100)
    assert a.to_dict() == b.to_dict()
    assert a.theta_id == b.theta_id
    w1 = R.em_fit(X, a, int(HMM_P3["max_iter_refit"]))
    w2 = R.em_fit(X, b, int(HMM_P3["max_iter_refit"]))
    assert w1.to_dict() == w2.to_dict()
    assert np.array_equal(a.A_arr, b.A_arr) and np.array_equal(a.cov_arr, b.cov_arr)


def test_em_rejects_nan_and_tiny_samples(gen_data):
    _, _, _, _, X = gen_data
    bad = X.copy()
    bad[10, 0] = np.nan
    with pytest.raises(ValueError):
        R.em_fit(bad, R.deterministic_init(X), 10)
    with pytest.raises(ValueError):
        R.em_fit(X[:20], R.deterministic_init(X), 10)


# ==================================================================
# 4. 필터 — 점 원칙
# ==================================================================
def test_filter_point_in_time(gen_data, gen_theta):
    """20개 무작위 절단 T 에서 filter_probabilities(obs[:T]) == 전체 결과의 [:T] **비트 동일**."""
    _, _, _, _, X = gen_data
    obs = _obs_frame(X)
    full = R.filter_probabilities(obs, gen_theta).to_numpy()
    rng = np.random.default_rng(0)
    for T in rng.integers(50, len(obs) + 1, size=20):
        part = R.filter_probabilities(obs.iloc[:int(T)], gen_theta).to_numpy()
        assert np.array_equal(part, full[:int(T)], equal_nan=True), f"PIT 위반 T={int(T)}"


def test_filter_missing_row_is_nan_not_carried(gen_data, gen_theta):
    """관측 결측 행 → NaN (직전 값 이월 없음), 그 뒤 행은 다시 정상."""
    _, _, _, _, X = gen_data
    obs = _obs_frame(X[:2000])
    hole = obs.copy()
    hole.iloc[1000, 0] = np.nan
    p = R.filter_probabilities(hole, gen_theta)
    base = R.filter_probabilities(obs, gen_theta)
    assert np.isnan(p.iloc[1000])                                   # 이월이 아니라 NaN
    assert np.isnan(p.to_numpy()).sum() == 1
    assert np.isfinite(p.iloc[1001])
    assert p.iloc[1001] != base.iloc[1001]                          # 정보가 하나 빠졌으므로 뒤 값은 달라진다
    assert np.array_equal(p.to_numpy()[:1000], base.to_numpy()[:1000])             # 앞부분은 불변


def test_forward_filter_loglik_matches_forward_backward(gen_data, gen_theta):
    """전방 loglik == 전방-후방 loglik, γ_T == α_T (β_T = 1)."""
    _, _, _, _, X = gen_data
    obs = _obs_frame(X[:3000])
    _, _, ll = R.forward_filter(obs, gen_theta)
    sm = R.smoothed_probabilities(obs, gen_theta)
    fl = R.filter_probabilities(obs, gen_theta)
    assert sm.iloc[-1] == fl.iloc[-1]
    assert np.isfinite(ll)
    ll_half = R.forward_filter(obs.iloc[:1500], gen_theta)[2]
    assert ll < ll_half < 0.0                                          # 로그가능도는 행이 늘수록 더 음수


def test_one_step_matches_filter(gen_data, gen_theta):
    """어제 P_high 에서 한 걸음 갱신 == 전체 필터의 오늘 값 (daily.py 의 1e-7 assert 근거)."""
    _, _, _, _, X = gen_data
    obs = _obs_frame(X[:500])
    p = R.filter_probabilities(obs, gen_theta)
    for t in (100, 250, 499):
        step = R.one_step(float(p.iloc[t - 1]), obs.iloc[t].to_numpy(), gen_theta)
        assert abs(step - float(p.iloc[t])) < 1e-12


# ==================================================================
# 5. 누수 카나리 · 지연 이동
# ==================================================================
def test_leakage_canary_synthetic(gen_data, gen_theta):
    """합성 자료에서 평활 확률의 상태 정확도가 필터보다 **높다** (평활은 미래를 읽는다)."""
    _, _, _, st, X = gen_data
    obs = _obs_frame(X)
    pf = R.filter_probabilities(obs, gen_theta).to_numpy()
    ps = R.smoothed_probabilities(obs, gen_theta).to_numpy()
    acc_f = float(((pf > 0.5).astype(int) == st).mean())
    acc_s = float(((ps > 0.5).astype(int) == st).mean())
    print(f"\n[canary synthetic] filter acc={acc_f:.5f} smoothed acc={acc_s:.5f} diff={acc_s - acc_f:+.5f}")
    assert acc_s > acc_f


@needs_cache
def test_leakage_canary_real(cache_obs, cache_y):
    """실캐시 2003+ 에서 평활 γ 의 y_dd5_20 AUC 가 필터보다 높다 = 필터가 실제로 돌고 있다.

    설계 단계(§4.3) 근사는 평활 0.722 vs 필터 0.687(격차 0.035). 이 구현의 실측은 아래 print 참조 —
    격차가 §13 의 0.02 에 못 미치면 CANARY_MIN 으로 판정하고 차이를 보고한다.
    """
    X = cache_obs.to_numpy()
    th = R.em_fit(X, R.deterministic_init(X), int(HMM_P3["max_iter_first"]))     # 전체표본 θ(설계와 동일)
    pf = R.filter_probabilities(cache_obs, th).to_numpy()
    ps = R.smoothed_probabilities(cache_obs, th).to_numpy()
    y = cache_y.to_numpy()
    sel = np.asarray(cache_obs.index >= "2003-01-02") & np.isfinite(y)
    a_f = auc(pf[sel], y[sel])
    a_s = auc(ps[sel], y[sel])
    print(f"\n[canary real] n={int(sel.sum())} filter AUC={a_f:.4f} smoothed AUC={a_s:.4f} "
          f"diff={a_s - a_f:+.4f} (design-stage 0.687 / 0.722, diff +0.035)")
    assert a_s > a_f
    assert (a_s - a_f) >= CANARY_MIN


@needs_cache
def test_leakage_shift(cache_wf, cache_y):
    """x_hmm 을 미래로 앞당기면 AUC 가 엄격히 오르고, 지연시키면 엄격히 떨어진다.
    −250 지연의 초과분(AUC−0.5)은 원본 초과분의 25% 미만이어야 한다."""
    df, _, _, _ = cache_wf
    y = cache_y.to_numpy()
    oos = (~df["in_sample"]).to_numpy()
    x = df["x_hmm"]

    def _auc_shift(k: int) -> float:
        xs = x.shift(-k).to_numpy()                       # k>0 = 미래를 앞당김, k<0 = 지연
        sel = oos & np.isfinite(y) & np.isfinite(xs)
        return auc(xs[sel], y[sel])

    base = _auc_shift(0)
    leads = {k: _auc_shift(k) for k in (1, 5, 20)}
    lags = {k: _auc_shift(k) for k in (-20, -250)}
    print(f"\n[leakage shift] base={base:.4f} leads={ {k: round(v, 4) for k, v in leads.items()} } "
          f"lags={ {k: round(v, 4) for k, v in lags.items()} }")
    for k, v in leads.items():
        assert v > base, f"lead +{k} 가 원본보다 높지 않다 ({v:.4f} vs {base:.4f})"
    for k, v in lags.items():
        assert v < base, f"lag {k} 가 원본보다 낮지 않다 ({v:.4f} vs {base:.4f})"
    assert (lags[-250] - 0.5) < 0.25 * (base - 0.5)


# ==================================================================
# 6. guard
# ==================================================================
def test_guard_degenerate_theta(gen_theta):
    """퇴화 θ(p00 0.999+, 점유 1%) → False + 사유. 정상 θ 는 통과."""
    ok, reasons = R.guard_check(gen_theta, np.array([0.60, 0.40]), None)
    assert ok and reasons == []
    bad = R.HMMTheta(A=[[0.9995, 0.0005], [0.02, 0.98]], mu=gen_theta.mu, cov=gen_theta.cov,
                     pi=R.stationary(np.array([[0.9995, 0.0005], [0.02, 0.98]])))
    ok2, reasons2 = R.guard_check(bad, np.array([0.99, 0.01]), gen_theta)
    assert not ok2
    joined = " | ".join(reasons2)
    assert "p00" in joined and "0.01" in joined
    print(f"\n[guard degenerate] {reasons2}")
    # ln RV10 평균 차가 0 인 θ
    flat_mu = np.array([[0.0, -1.5], [0.0, -1.45]])
    flat = R.HMMTheta(A=gen_theta.A, mu=flat_mu, cov=gen_theta.cov, pi=gen_theta.pi)
    ok3, reasons3 = R.guard_check(flat, np.array([0.5, 0.5]), gen_theta)
    assert not ok3 and any("ln RV10" in r for r in reasons3)
    # 비양정치 공분산
    sing = np.stack([np.array([[1.0, 1.0], [1.0, 1.0]]), gen_theta.cov_arr[1]])
    degen = R.HMMTheta(A=gen_theta.A, mu=gen_theta.mu, cov=sing, pi=gen_theta.pi)
    ok4, reasons4 = R.guard_check(degen, np.array([0.5, 0.5]), gen_theta)
    assert not ok4 and any("양정치" in r for r in reasons4)


def _synthetic_wf_inputs():
    """walk-forward 용 합성 관측 + 재적합일 (train_start 1993-10-14 를 포함하는 달력)."""
    _, _, _, close = _gen_price_path(n=3300, seed=0)
    close.index = pd.bdate_range("1993-01-04", periods=len(close), name="date")
    obs = R.observations(close)
    jan = obs.index[(obs.index.month == 1)]
    first = jan[jan.year == 2003][0]
    rd = refit_dates(obs.index, first=first, end="2006-01-01")
    return obs, rd


def test_guard_keeps_previous_theta():
    """guard 를 통과할 수 없는 cfg 를 주면 첫 해 뒤로는 θ_{y−1} 를 유지하고 경고·로그를 남긴다."""
    obs, rd = _synthetic_wf_inputs()
    assert len(rd) >= 3
    cfg = dict(HMM_P3)
    cfg["guard"] = dict(HMM_P3["guard"], gap_lnvol=1e6)              # 어떤 θ 도 통과 못 함
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df, thetas = R.hmm_walk_forward(obs, rd, cfg=cfg)
    msgs = [str(w.message) for w in caught]
    assert any("guard" in m for m in msgs), msgs
    assert len(thetas) == len(rd)
    assert all(t.theta_id == thetas[0].theta_id for t in thetas), "θ 가 유지되지 않았다"
    log = df.attrs["guard_log"]
    assert all(e["ok"] is False for e in log)
    assert log[0]["kept_prev"] is False and all(e["kept_prev"] for e in log[1:])
    assert len(df.attrs["warnings"]) == len(rd)
    # 정상 cfg 로는 통과한다 (양성 대조)
    df2, thetas2 = R.hmm_walk_forward(obs, rd)
    assert all(e["ok"] for e in df2.attrs["guard_log"])
    assert len({t.theta_id for t in thetas2}) == len(rd)


# ==================================================================
# 7. k-스텝 전이확률
# ==================================================================
def test_k_step_one_matches_transition_row():
    A = np.array([[0.98, 0.02], [0.05, 0.95]])
    th = R.HMMTheta(A=A, mu=[[0.06, -1.84], [-0.04, -0.92]],
                    cov=[[[1.0, -0.02], [-0.02, 0.0625]], [[6.25, -0.17], [-0.17, 0.1225]]], pi=R.stationary(A))
    assert R.k_step(th, 0.0, 1)["p_k"] == pytest.approx(A[0, 1], rel=1e-14)
    assert R.k_step(th, 1.0, 1)["p_k"] == pytest.approx(A[1, 1], rel=1e-14)
    assert R.k_step(th, 0.0, 1)["q_k"] == pytest.approx(A[0, 1], rel=1e-14)
    assert R.k_step(th, 1.0, 1)["q_k"] == pytest.approx(A[1, 1], rel=1e-14)
    qs = [R.k_step(th, 0.3, k)["q_k"] for k in range(1, 40)]
    assert all(b > a for a, b in zip(qs, qs[1:])), "q_k 가 단조 증가가 아니다"
    assert np.isnan(R.k_step(th, float("nan"))["p_k"])
    with pytest.raises(ValueError):
        R.k_step(th, 0.5, 0)


def test_k_step_matches_monte_carlo():
    """k=20 닫힌 식 == 몬테카를로(seed 0, n=2e6, 1e-3)."""
    A = np.array([[0.98, 0.02], [0.05, 0.95]])
    th = R.HMMTheta(A=A, mu=[[0.06, -1.84], [-0.04, -0.92]],
                    cov=[[[1.0, -0.02], [-0.02, 0.0625]], [[6.25, -0.17], [-0.17, 0.1225]]], pi=R.stationary(A))
    n = 2_000_000
    for xi in (0.0, 0.15, 0.5, 0.85, 1.0):
        rng = np.random.default_rng(0)
        s = (rng.random(n) < xi).astype(int)
        any_high = np.zeros(n, dtype=bool)
        for _ in range(20):
            u = rng.random(n)
            s = np.where(s == 0, (u >= A[0, 0]).astype(int), (u >= A[1, 0]).astype(int))
            any_high |= s == 1
        ks = R.k_step(th, xi, 20)
        mc_p, mc_q = float((s == 1).mean()), float(any_high.mean())
        print(f"\n[k_step MC] xi={xi:.2f} p20={ks['p_k']:.5f} (MC {mc_p:.5f}) q20={ks['q_k']:.5f} (MC {mc_q:.5f})")
        assert abs(ks["p_k"] - mc_p) < 1e-3
        assert abs(ks["q_k"] - mc_q) < 1e-3


# ==================================================================
# 8. 퍼지 일정
# ==================================================================
def test_purge_and_holdout_exclusion():
    """모든 재적합에서 max(학습 pos) + purge < pos(R); 홀드아웃 행은 어떤 학습 마스크에도 없다."""
    idx = pd.bdate_range("1993-01-04", "2026-01-30", name="date")
    purge = int(HMM_P3["purge"])
    for R_y in refit_dates(idx, first=idx[(idx.year == 2003) & (idx.month == 1)][0], end=None):
        m = training_mask(idx, R_y, purge, HMM_P3["train_start"])
        pos = np.arange(len(idx))[m]
        assert pos.max() + purge < int(idx.get_loc(R_y))
        assert idx[m].min() >= pd.Timestamp(HMM_P3["train_start"])
        if R_y < pd.Timestamp(HOLDOUT_START):
            assert (idx[m] < pd.Timestamp(HOLDOUT_START)).all(), f"{R_y.date()} 학습창에 홀드아웃 행"


@needs_cache
def test_walk_forward_purge_real(cache_obs, cache_wf):
    df, thetas, rd, _ = cache_wf
    for t, R_y in zip(thetas, rd):
        assert pd.Timestamp(t.train_end) < R_y
        gap = int(cache_obs.index.get_loc(R_y)) - int(cache_obs.index.get_loc(pd.Timestamp(t.train_end)))
        assert gap >= int(HMM_P3["purge"]) + 1, f"{R_y.date()} 퍼지 부족 (gap={gap})"
        assert pd.Timestamp(t.train_start) >= pd.Timestamp(HMM_P3["train_start"])
    assert df.index.max() < pd.Timestamp(HOLDOUT_START)


# ==================================================================
# 9. 실캐시 walk-forward — §4.3 설계 단계 실측과 대조
# ==================================================================
@needs_cache
def test_walk_forward_real_cache(cache_obs, cache_wf, cache_y):
    """22회 재적합 < 60초, 모든 해 guard 통과, p00/p11 ∈ [0.97, 0.99].
    전이행렬·기대 체류·정상확률·상태 변동성·OOS AUC 를 설계 단계 실측(§4.3)과 나란히 보고."""
    df, thetas, rd, seconds = cache_wf
    assert len(rd) == 22 and str(rd[0].date()) == "2003-01-02" and str(rd[-1].date()) == "2024-01-02"
    assert len(thetas) == 22
    assert seconds < 60.0, f"walk-forward {seconds:.1f}s (상한 60s)"
    tt = R.theta_table(thetas)
    assert (tt["guard"] == "ok").all(), tt.loc[tt["guard"] != "ok"].to_string()
    assert tt["p00"].between(0.97, 0.99).all() and tt["p11"].between(0.97, 0.99).all()
    for t in thetas:
        assert t.n_params() == 12 and t.converged
        assert R.stationary(t.A_arr) == pytest.approx(t.pi_arr, abs=1e-12)

    first, last = thetas[0], thetas[-1]
    y = cache_y.to_numpy()
    oos = (~df["in_sample"]).to_numpy() & np.isfinite(y)
    ph = df["p_high"].to_numpy()
    a_ph = auc(ph[oos], y[oos])
    a_q = auc(df["q20"].to_numpy()[oos], y[oos])
    a_p = auc(df["p20"].to_numpy()[oos], y[oos])
    a_x = auc(df["x_hmm"].to_numpy()[oos], y[oos])
    occ = float((ph[oos] > 0.5).mean())
    base = float(y[oos].mean())
    eras = (("2003-01-01", "2007-12-31", 0.648), ("2008-01-01", "2012-12-31", 0.675),
            ("2013-01-01", "2019-12-31", 0.581), ("2020-01-01", HARD_CUT, 0.666))

    print("\n" + "=" * 78)
    print("[real cache walk-forward] hard cut 2024-08-30 (holdout never scored)")
    print(f"  refits={len(thetas)}  seconds={seconds:.2f} (design ~27s)  obs={len(cache_obs)} "
          f"{cache_obs.index[0].date()}..{cache_obs.index[-1].date()}")
    print(f"  first refit {first.refit_date}: A={np.round(first.A_arr, 4).tolist()}"
          f"  (design [[0.9803, 0.0197], [0.0170, 0.9830]])")
    print(f"    duration low/high = {tt.loc[0, 'dur0']:.1f}/{tt.loc[0, 'dur1']:.1f} sessions (design 51/59)")
    print(f"    state vol (annual) = {tt.loc[0, 'vol0']:.4f}/{tt.loc[0, 'vol1']:.4f} (design 0.099/0.238)")
    print(f"    daily drift %/day  = {tt.loc[0, 'mu_r0']:+.4f}/{tt.loc[0, 'mu_r1']:+.4f} (design +0.088/-0.015)")
    print(f"    stationary P(high) = {first.pi_arr[1]:.4f}")
    print(f"  last refit {last.refit_date}: A={np.round(last.A_arr, 4).tolist()}"
          f"  (design 2024 [[0.9807, 0.0193], [0.0217, 0.9783]])")
    print(f"    duration {tt.iloc[-1]['dur0']:.1f}/{tt.iloc[-1]['dur1']:.1f}  "
          f"vol {tt.iloc[-1]['vol0']:.4f}/{tt.iloc[-1]['vol1']:.4f}  "
          f"stationary P(high)={last.pi_arr[1]:.4f}")
    print(f"  p00 range [{tt['p00'].min():.4f}, {tt['p00'].max():.4f}] (design 0.980~0.984)")
    print(f"  p11 range [{tt['p11'].min():.4f}, {tt['p11'].max():.4f}] (design 0.977~0.984)")
    print(f"  OOS rows={int(oos.sum())} (design 5,433)  base rate={base:.4f} (design 0.154)")
    print(f"  filtered P_high occupancy(>0.5)={occ:.4f} (design 0.428)")
    print(f"  AUC vs y_dd5_20: P_high={a_ph:.4f} (design 0.697) | q20={a_q:.4f} (design 0.667) | "
          f"p20={a_p:.4f} | x_hmm(clipped)={a_x:.4f}")
    for a, z, ref in eras:
        sl = oos & np.asarray(df.index >= a) & np.asarray(df.index <= z)
        print(f"    era {a[:4]}-{z[:4]}: n={int(sl.sum())} AUC={auc(ph[sl], y[sl]):.4f} (design {ref:.3f})")
    print("=" * 78)

    assert int(oos.sum()) == 5433
    assert abs(base - 0.154) < 0.005
    assert abs(occ - 0.428) < 0.02
    assert abs(a_ph - 0.697) < 0.010
    assert abs(a_q - 0.667) < 0.010
    for a, z, ref in eras:
        sl = oos & np.asarray(df.index >= a) & np.asarray(df.index <= z)
        assert abs(auc(ph[sl], y[sl]) - ref) < 0.02, f"era {a[:4]} AUC 가 설계 실측에서 크게 벗어남"


@needs_cache
def test_walk_forward_columns_and_segments(cache_obs, cache_wf):
    """열·세그먼트 규약: 인덱스 = 관측 전체, R_first 이전은 in_sample, 각 구간의 theta_id 가 그 해 θ."""
    df, thetas, rd, _ = cache_wf
    assert list(df.columns) == ["p_high", "p20", "q20", "x_hmm", "refit_year", "theta_id", "in_sample"]
    assert df.index.equals(cache_obs.index)
    assert df.loc[: rd[0] - pd.Timedelta(days=1), "in_sample"].all()
    assert not df.loc[rd[0]:, "in_sample"].any()
    assert df.loc[:rd[0], "theta_id"].iloc[0] == thetas[0].theta_id
    for i, (R_y, th) in enumerate(zip(rd, thetas)):
        stop = rd[i + 1] - pd.Timedelta(days=1) if i + 1 < len(rd) else df.index[-1]
        seg = df.loc[R_y:stop]
        assert (seg["theta_id"] == th.theta_id).all()
        assert (seg["refit_year"] == R_y.year).all()
    p = df["p_high"].to_numpy()
    ok = np.isfinite(p)
    assert ok.all()
    assert ((p >= 0) & (p <= 1)).all()
    x = df["x_hmm"].to_numpy()
    lim = math.log((1 - HMM_P3["clip"]) / HMM_P3["clip"])
    assert np.nanmax(np.abs(x)) <= lim + 1e-12
    # x_hmm 은 p_high 의 단조 변환(클립 전) — 클립 밖에서는 순위가 같다
    inner = (p > HMM_P3["clip"]) & (p < 1 - HMM_P3["clip"])
    assert np.all(np.argsort(p[inner], kind="stable") == np.argsort(x[inner], kind="stable"))


@needs_cache
def test_walk_forward_determinism_real(cache_obs, cache_wf):
    """두 번 실행 → θ·확률이 1e-7 안에서 같다(실측은 비트 동일)."""
    df1, thetas1, rd, _ = cache_wf
    df2, thetas2 = R.hmm_walk_forward(cache_obs, rd)
    cols = ["p_high", "p20", "q20", "x_hmm"]
    dmax = float(np.nanmax(np.abs(df1[cols].to_numpy() - df2[cols].to_numpy())))
    tmax = max(float(np.max(np.abs(a.A_arr - b.A_arr))) for a, b in zip(thetas1, thetas2))
    mmax = max(float(np.max(np.abs(a.mu_arr - b.mu_arr))) for a, b in zip(thetas1, thetas2))
    print(f"\n[determinism] max |dp|={dmax:.3e} max |dA|={tmax:.3e} max |dmu|={mmax:.3e} "
          f"(tol {HMM_P3['tol_prob']:g}/{HMM_P3['tol_theta']:g})")
    assert dmax <= float(HMM_P3["tol_prob"])
    assert tmax <= float(HMM_P3["tol_theta"]) and mmax <= float(HMM_P3["tol_theta"])
    assert [t.theta_id for t in thetas1] == [t.theta_id for t in thetas2]


@needs_cache
def test_walk_forward_point_in_time_real(cache_obs, cache_wf):
    """실캐시 θ 로도 20개 무작위 절단에서 필터가 비트 동일하다."""
    df, thetas, _, _ = cache_wf
    th = thetas[-1]
    full = R.filter_probabilities(cache_obs, th).to_numpy()
    rng = np.random.default_rng(0)
    for T in rng.integers(100, len(cache_obs) + 1, size=20):
        part = R.filter_probabilities(cache_obs.iloc[:int(T)], th).to_numpy()
        assert np.array_equal(part, full[:int(T)], equal_nan=True), f"PIT 위반 T={int(T)}"


# ==================================================================
# 10. 저장 · 불러오기 · 표
# ==================================================================
def test_save_load_roundtrip(tmp_path, gen_theta):
    """hmm_p3.json 왕복: θ 비트 동일, live 보존, schema/obs_spec 검사."""
    th1 = R.HMMTheta(A=gen_theta.A, mu=gen_theta.mu, cov=gen_theta.cov, pi=gen_theta.pi,
                     refit_date="2023-01-03", train_start="1993-10-14", train_end="2022-12-01",
                     n_obs=7000, loglik=-1234.5, n_iter=11, converged=True,
                     guard={"ok": True, "reasons": [], "occupancy": [0.6, 0.4]})
    th2 = R.HMMTheta(A=gen_theta.A, mu=gen_theta.mu, cov=gen_theta.cov, pi=gen_theta.pi,
                     refit_date="2024-01-02", train_start="1993-10-14", train_end="2023-12-01",
                     n_obs=7200, loglik=-1200.0, n_iter=9, converged=True, guard={"ok": True, "reasons": []})
    p = tmp_path / "hmm_p3.json"
    R.save_thetas([th1, th2], p, live=th2, created_at_utc="1970-01-01T00:00:00+00:00")
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert raw["schema_version"] == R.SCHEMA_VERSION and raw["obs_spec"] == R.OBS_SPEC
    assert raw["param_count"] == 12 and len(raw["thetas"]) == 2 and raw["live"]["refit_date"] == "2024-01-02"
    assert raw["registry_sha"] and len(raw["registry_sha"]) == 64
    loaded, live = R.load_thetas(p)
    assert [t.theta_id for t in loaded] == [th1.theta_id, th2.theta_id]
    assert live.theta_id == th2.theta_id
    assert np.array_equal(loaded[0].A_arr, th1.A_arr)
    assert np.array_equal(loaded[1].cov_arr, th2.cov_arr)
    bad = json.loads(p.read_text(encoding="utf-8"))
    bad["schema_version"] = 99
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        R.load_thetas(p)
    with pytest.raises(FileNotFoundError):
        R.load_thetas(tmp_path / "nope.json")


def test_theta_id_stable_and_sensitive(gen_theta):
    """theta_id = 1e-10 반올림 파라미터의 sha256[:12] — 1e-12 변화는 무시, 1e-6 변화는 잡는다."""
    A = gen_theta.A_arr.copy()
    same = R.HMMTheta(A=A, mu=gen_theta.mu, cov=gen_theta.cov, pi=gen_theta.pi)
    assert same.theta_id == gen_theta.theta_id
    A2 = A.copy()
    A2[0, 0] += 1e-13
    A2[0, 1] -= 1e-13
    tiny = R.HMMTheta(A=A2, mu=gen_theta.mu, cov=gen_theta.cov, pi=gen_theta.pi)
    assert tiny.theta_id == gen_theta.theta_id
    A3 = A.copy()
    A3[0, 0] -= 1e-6
    A3[0, 1] += 1e-6
    moved = R.HMMTheta(A=A3, mu=gen_theta.mu, cov=gen_theta.cov, pi=gen_theta.pi)
    assert moved.theta_id != gen_theta.theta_id


def test_theta_validation_rejects_bad():
    good = dict(mu=[[0.0, -1.8], [0.0, -1.0]], cov=[[[1.0, 0.0], [0.0, 0.1]], [[4.0, 0.0], [0.0, 0.2]]])
    with pytest.raises(ValueError):
        R.HMMTheta(A=[[0.9, 0.2], [0.05, 0.95]], pi=[0.5, 0.5], **good)          # 행 합 ≠ 1
    with pytest.raises(ValueError):
        R.HMMTheta(A=[[0.9, 0.1], [0.05, 0.95]], pi=[0.7, 0.7], **good)          # π 합 ≠ 1
    with pytest.raises(ValueError):
        R.HMMTheta(A=[[np.nan, 0.1], [0.05, 0.95]], pi=[0.5, 0.5], **good)       # NaN
    with pytest.raises(ValueError):
        R.HMMTheta(A=[[0.9, 0.1], [0.05, 0.95]], pi=[0.5, 0.5],
                   mu=good["mu"], cov=[[[1.0, 0.5], [0.4, 0.1]], [[4.0, 0.0], [0.0, 0.2]]])   # 비대칭


@needs_cache
def test_theta_table_shape(cache_wf):
    _, thetas, rd, _ = cache_wf
    tt = R.theta_table(thetas)
    for c in ("refit_date", "p00", "p11", "dur0", "dur1", "vol0", "vol1", "mu_r0", "mu_r1", "n_iter", "guard"):
        assert c in tt.columns
    assert len(tt) == len(rd)
    assert (tt["dur0"] > 30).all() and (tt["dur1"] > 30).all()
    assert (tt["vol1"] > tt["vol0"]).all()
