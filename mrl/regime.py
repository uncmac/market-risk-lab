# -*- coding: utf-8 -*-
"""mrl.regime — 2-상태 가우시안 HMM 국면 엔진 (ARCHITECTURE_PHASE3.md §4 계약).

역할
  * 관측 `o_t = [100·r_t, ln RV10_t]` (조정 종가만 사용 — 1993~95 OHLC 품질 문제에 면역).
  * K=2, 상태별 **완전** 2×2 공분산 가우시안. 전이행렬 A(2×2), 초기분포 π = A 의 정상분포(적합 아님).
  * 직접 구현한 numpy EM(스케일드 전방-후방, einsum ξ 누적), **결정적 초기화·난수 재시작 없음**, warm start.
  * 연 1회(각 해 1월 첫 거래일) 퍼지(20세션) 재적합 — 일정은 `mrl.calibrate.refit_dates` 와 동일.
  * **필터(전방)만** 생산 경로에 쓴다. 평활 γ 는 누수 카나리 테스트 전용(`smoothed_probabilities`).
  * k-스텝 전이확률 p_k·q_k 는 A 의 닫힌 식(표시·소거 전용), 특징은 `x_hmm = logit(clip(P_high))` 하나.

파라미터 회계 (VALIDATION §6)
  * 이 모듈이 만드는 적합값은 **전부 생산 확률 예산 밖**이다. 생산 확률(Phase 2 M3/배치 단)은 4개 그대로이고
    Phase 3 는 거기에 0개를 더한다. HMM θ 는 라벨을 보지 않는 **비지도** 12개(K_u)로 별도 공개한다:
    μ 4 + Σ 6(대칭 2×3) + A 2 = 12 = `HMM_P3["param_count"]`.

원칙
  * 점(point-in-time): `filter_probabilities` 는 전방 재귀만 돌리므로 t 이후 행을 덧붙여도 t 까지의 값이
    **비트 동일**하다(tests/test_regime.py::test_filter_point_in_time 이 20개 무작위 절단에서 검사).
    이를 보장하려고 방출 로그밀도는 2차원 닫힌 식의 **원소별** 연산으로만 계산한다(블록 분할 의존 없음).
  * 결정론: 난수 없음(seed 무관), 결정적 초기화 + warm start. 전방·후방 재귀는 K=2 스칼라 루프(BLAS 무관).
    러너·로컬의 BLAS 차이는 M-스텝(einsum)에만 남으므로 허용오차 θ·확률 1e-7(`HMM_P3["tol_theta"]`).
  * 조용한 실패 금지: 형식 오류·표본 부족·수치 붕괴는 예외, 퇴화 θ 는 guard 가 잡아 θ_{y−1} 유지 + 경고 + 로그.
    관측 결측 행의 P_high 는 NaN 이며 **직전 값을 이월하지 않는다**.
  * 한국어 주석, 영어 식별자.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from mrl.calibrate import training_mask
from mrl.config import HMM_P3, HMM_P3_PATH, P2

__all__ = [
    "OBS_COLUMNS", "OBS_SPEC", "HMM_PARAM_COUNT", "TRADING_DAYS", "SCHEMA_VERSION",
    "HMMTheta", "observations", "deterministic_init", "stationary", "log_emissions",
    "forward_filter", "em_fit", "guard_check", "occupancy", "filter_probabilities",
    "smoothed_probabilities", "k_step", "hmm_walk_forward", "n_params",
    "save_thetas", "load_thetas", "theta_table", "regime_sha256", "logit", "one_step",
]

OBS_COLUMNS = ("r100", "ln_rv10")                     # = HMM_P3["obs"]
OBS_SPEC = "r100,ln_rv10;rv_window=10;cov=full;K=2;pi=stationary"
HMM_PARAM_COUNT = 12                                  # μ 4 + Σ 6 + A 2 (π 는 A 의 정상분포 — 자유도 0)
TRADING_DAYS = 252
SCHEMA_VERSION = 1
_LOG_2PI = math.log(2.0 * math.pi)
_ROUND_DIGITS = 10                                    # theta_id 해시용 반올림(1e-10)


# ------------------------------------------------------------------
# 입력 검증 헬퍼
# ------------------------------------------------------------------
def _clean_close(close: pd.Series, name: str = "close") -> pd.Series:
    """종가 시계열 검증·정규화: NaN 제거, DatetimeIndex(tz-naive), 오름차순, 중복 없음, 양수."""
    if not isinstance(close, pd.Series):
        raise TypeError(f"{name} 는 pd.Series 여야 합니다 (받은 형: {type(close).__name__})")
    s = close.dropna()
    if len(s) == 0:
        raise ValueError(f"{name} 에 유효한(비-NaN) 값이 없습니다")
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError(f"{name} 인덱스는 DatetimeIndex 여야 합니다 (받은 형: {type(s.index).__name__})")
    if s.index.tz is not None:
        raise ValueError(f"계약 위반: {name} 인덱스는 tz-naive 여야 합니다")
    if s.index.has_duplicates:
        dup = s.index[s.index.duplicated()][:3].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"{name} 인덱스에 중복 날짜가 있습니다 (예: {dup})")
    if not s.index.is_monotonic_increasing:
        raise ValueError(f"{name} 인덱스가 오름차순이 아닙니다")
    s = s.astype(float)
    if (s <= 0).any():
        bad = s.index[s <= 0][:3].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"{name} 에 0 이하 가격이 있습니다 (예: {bad})")
    return s


def _obs_matrix(obs) -> np.ndarray:
    """DataFrame(열 OBS_COLUMNS) 또는 T×2 ndarray → float ndarray (복사본)."""
    if isinstance(obs, pd.DataFrame):
        missing = [c for c in OBS_COLUMNS if c not in obs.columns]
        if missing:
            raise ValueError(f"관측 프레임에 열이 없습니다: {missing} (필요: {list(OBS_COLUMNS)})")
        arr = obs[list(OBS_COLUMNS)].to_numpy(dtype=float)
    else:
        arr = np.asarray(obs, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"관측은 T×d 2차원이어야 합니다 (받은 shape={arr.shape})")
    if arr.shape[0] == 0:
        raise ValueError("관측이 비어 있습니다")
    return np.array(arr, dtype=float, copy=True)


def _dstr(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _round(x) -> float:
    v = round(float(x), _ROUND_DIGITS)
    return 0.0 if v == 0 else v


def _round_nested(obj):
    if isinstance(obj, (list, tuple)):
        return [_round_nested(o) for o in obj]
    return _round(obj)


def _nested(obj):
    """중첩 튜플 → 중첩 리스트 (반올림 없음 — JSON 왕복이 비트 동일해야 한다)."""
    if isinstance(obj, (list, tuple)):
        return [_nested(o) for o in obj]
    return float(obj)


def logit(p):
    """logit(p). 스칼라·배열 모두. p 는 (0,1) 안이어야 하며 호출자가 먼저 clip 한다."""
    a = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(a) - np.log1p(-a)
    return float(out) if np.ndim(p) == 0 else out


# ------------------------------------------------------------------
# θ (동결 dataclass)
# ------------------------------------------------------------------
@dataclass(frozen=True)
class HMMTheta:
    """한 번의 적합 결과. A/mu/cov/pi 는 JSON 직렬화용 중첩 튜플이며 계산은 np.asarray 로 한다.

    A   : 2×2 전이행렬(행 확률)          mu  : 2×2 (상태별 평균 [100·r, ln RV10])
    cov : 2×2×2 (상태별 완전 공분산)     pi  : 2 (= stationary(A); 적합 아님)
    라벨 규약: **상태 1 = 수익률 분산이 큰 쪽**('고변동 상태'; '위기'가 아니다).
    """
    A: tuple
    mu: tuple
    cov: tuple
    pi: tuple
    refit_date: str = ""
    train_start: str = ""
    train_end: str = ""
    n_obs: int = 0
    loglik: float = float("nan")
    n_iter: int = 0
    converged: bool = False
    obs_spec: str = OBS_SPEC
    theta_id: str = ""
    guard: dict = field(default_factory=dict)

    def __post_init__(self):
        A = np.asarray(self.A, dtype=float)
        mu = np.asarray(self.mu, dtype=float)
        cov = np.asarray(self.cov, dtype=float)
        pi = np.asarray(self.pi, dtype=float)
        K = A.shape[0]
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A 는 K×K 여야 합니다 (shape={A.shape})")
        if mu.shape[0] != K or mu.ndim != 2:
            raise ValueError(f"mu 는 K×d 여야 합니다 (shape={mu.shape}, K={K})")
        d = mu.shape[1]
        if cov.shape != (K, d, d):
            raise ValueError(f"cov 는 K×d×d 여야 합니다 (shape={cov.shape}, 기대 {(K, d, d)})")
        if pi.shape != (K,):
            raise ValueError(f"pi 는 길이 {K} 여야 합니다 (shape={pi.shape})")
        if not np.isfinite(A).all() or not np.isfinite(mu).all() or not np.isfinite(cov).all() or not np.isfinite(pi).all():
            raise ValueError("θ 에 비유한값(NaN/inf)이 있습니다")
        if (A < 0).any():
            raise ValueError("A 에 음수 확률이 있습니다")
        rs = A.sum(axis=1)
        if not np.allclose(rs, 1.0, atol=1e-8):
            raise ValueError(f"A 의 행 합이 1 이 아닙니다: {rs.tolist()}")
        if not np.allclose(cov, np.transpose(cov, (0, 2, 1)), atol=1e-12):
            raise ValueError("cov 가 대칭이 아닙니다")
        if abs(float(pi.sum()) - 1.0) > 1e-8 or (pi < 0).any():
            raise ValueError(f"pi 가 확률벡터가 아닙니다: {pi.tolist()}")
        # 양정치는 여기서 검사하지 않는다 — 퇴화 θ 를 만들 수 있어야 guard_check 가 사유를 보고한다.
        object.__setattr__(self, "A", tuple(tuple(float(v) for v in row) for row in A))
        object.__setattr__(self, "mu", tuple(tuple(float(v) for v in row) for row in mu))
        object.__setattr__(self, "cov", tuple(tuple(tuple(float(v) for v in r) for r in m) for m in cov))
        object.__setattr__(self, "pi", tuple(float(v) for v in pi))
        object.__setattr__(self, "n_obs", int(self.n_obs))
        object.__setattr__(self, "n_iter", int(self.n_iter))
        object.__setattr__(self, "converged", bool(self.converged))
        object.__setattr__(self, "loglik", float(self.loglik))
        if not self.theta_id:
            object.__setattr__(self, "theta_id", self._make_id())

    # ---- 배열 접근 ----
    @property
    def A_arr(self) -> np.ndarray:
        return np.asarray(self.A, dtype=float)

    @property
    def mu_arr(self) -> np.ndarray:
        return np.asarray(self.mu, dtype=float)

    @property
    def cov_arr(self) -> np.ndarray:
        return np.asarray(self.cov, dtype=float)

    @property
    def pi_arr(self) -> np.ndarray:
        return np.asarray(self.pi, dtype=float)

    @property
    def n_states(self) -> int:
        return len(self.A)

    def _make_id(self) -> str:
        payload = json.dumps({"A": _round_nested(self.A), "mu": _round_nested(self.mu),
                              "cov": _round_nested(self.cov), "pi": _round_nested(self.pi)},
                             sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def n_params(self) -> int:
        return n_params(self)

    # ---- 직렬화 ----
    def to_dict(self) -> dict:
        # 저장은 **완전 정밀도**(왕복 비트 동일). 1e-10 반올림은 theta_id 해시에만 쓴다.
        return {"A": _nested(self.A), "mu": _nested(self.mu), "cov": _nested(self.cov),
                "pi": _nested(self.pi), "refit_date": self.refit_date, "train_start": self.train_start,
                "train_end": self.train_end, "n_obs": int(self.n_obs), "loglik": float(self.loglik),
                "n_iter": int(self.n_iter), "converged": bool(self.converged), "obs_spec": self.obs_spec,
                "theta_id": self.theta_id, "n_params": self.n_params(), "guard": _jsonable(self.guard)}

    @classmethod
    def from_dict(cls, d: dict) -> "HMMTheta":
        required = ("A", "mu", "cov", "pi")
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(f"θ dict 에 키가 없습니다: {missing}")
        return cls(A=d["A"], mu=d["mu"], cov=d["cov"], pi=d["pi"],
                   refit_date=str(d.get("refit_date", "")), train_start=str(d.get("train_start", "")),
                   train_end=str(d.get("train_end", "")), n_obs=int(d.get("n_obs", 0)),
                   loglik=float(d.get("loglik", float("nan"))), n_iter=int(d.get("n_iter", 0)),
                   converged=bool(d.get("converged", False)), obs_spec=str(d.get("obs_spec", OBS_SPEC)),
                   theta_id=str(d.get("theta_id", "")), guard=dict(d.get("guard") or {}))


def _jsonable(obj):
    """JSON 저장용 변환 (numpy 스칼라·배열·NaN 처리)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def n_params(theta: HMMTheta) -> int:
    """적합 파라미터 수: μ K·d + Σ K·d(d+1)/2 + A K(K−1). K=2, d=2 → 12 (π 는 A 의 정상분포이므로 0)."""
    A = np.asarray(theta.A, dtype=float)
    mu = np.asarray(theta.mu, dtype=float)
    K, d = mu.shape
    return int(K * d + K * d * (d + 1) // 2 + K * (K - 1)) if A.shape[0] == K else -1


# ------------------------------------------------------------------
# 4.1 관측
# ------------------------------------------------------------------
def observations(close: pd.Series, asof=None, rv_window: int = HMM_P3["rv_window"]) -> pd.DataFrame:
    """조정 종가 → 관측 프레임 (열 `r100`, `ln_rv10`).

    r_t = ln(C_t / C_{t−1}) (비율이라 조정 불변) · r100 = 100·r_t
    RV10_t = sqrt(252 · max(mean(r²_{t−9..t}), var_floor)) · ln_rv10 = ln RV10_t
    * `asof` 는 **계산 전에** 자른다(계약 §4.2). 모든 창이 뒤를 보는 창이라 값은 같지만 계약대로 자른다.
    * 반환 인덱스는 첫 유효 관측(수익률 rv_window 개; 실캐시 1993-02-12)부터. NaN 종가 행은 제거된다.
    """
    if int(rv_window) < 2:
        raise ValueError(f"rv_window 는 2 이상이어야 합니다: {rv_window}")
    s = _clean_close(close)
    if asof is not None:
        s = s.loc[:pd.Timestamp(asof)]
    if len(s) < int(rv_window) + 1:
        raise ValueError(f"관측을 만들기에 종가가 부족합니다 (n={len(s)}, 필요 ≥ {int(rv_window) + 1})")
    r = np.log(s).diff()
    mean_r2 = (r ** 2).rolling(int(rv_window), min_periods=int(rv_window)).mean()
    rv = np.sqrt(TRADING_DAYS * np.maximum(mean_r2, float(P2["var_floor"])))
    out = pd.DataFrame({"r100": 100.0 * r, "ln_rv10": np.log(rv)}, index=s.index)
    valid = out.notna().all(axis=1).to_numpy()
    if not valid.any():
        raise ValueError("유효한 관측이 하나도 없습니다")
    out = out.iloc[int(np.argmax(valid)):]
    out.index.name = s.index.name or "date"
    out.attrs = {"obs_spec": OBS_SPEC, "rv_window": int(rv_window), "var_floor": float(P2["var_floor"]),
                 "first_obs": _dstr(out.index[0]), "asof": _dstr(asof) if asof is not None else None}
    return out


# ------------------------------------------------------------------
# 4.1 정상분포 · 결정적 초기화
# ------------------------------------------------------------------
def stationary(A: np.ndarray) -> np.ndarray:
    """A 의 정상분포 π (πA = π, Σπ = 1). K=2 는 닫힌 식 — 적합 파라미터가 아니다."""
    M = np.asarray(A, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"A 는 정사각 행렬이어야 합니다 (shape={M.shape})")
    K = M.shape[0]
    if K == 2:
        a01, a10 = float(M[0, 1]), float(M[1, 0])
        s = a01 + a10
        if not (s > 0.0):
            raise ValueError(f"흡수 상태(A 의 비대각 합 0) — 정상분포가 유일하지 않습니다: A={M.tolist()}")
        return np.array([a10 / s, a01 / s], dtype=float)
    L = np.vstack([M.T - np.eye(K), np.ones(K)])
    b = np.zeros(K + 1)
    b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(L, b, rcond=None)
    if (pi < -1e-10).any():
        raise ValueError(f"정상분포에 음수 성분이 있습니다: {pi.tolist()}")
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum()


def deterministic_init(obs, init_A=HMM_P3["init_A"]) -> HMMTheta:
    """결정적 초기화(난수 재시작 없음): ln RV10 중앙값으로 관측을 둘로 갈라
    상태 0 = 하위 절반의 평균·공분산, 상태 1 = 상위 절반. A0 = init_A, π0 = stationary(A0)."""
    X = _obs_matrix(obs)
    if not np.isfinite(X).all():
        raise ValueError("deterministic_init: 관측에 NaN/inf 가 있습니다 (호출자가 먼저 걸러야 합니다)")
    if X.shape[1] != 2:
        raise ValueError(f"관측은 2차원(r100, ln_rv10)이어야 합니다 (d={X.shape[1]})")
    med = float(np.median(X[:, 1]))
    lo = X[X[:, 1] <= med]
    hi = X[X[:, 1] > med]
    if len(lo) < 2 or len(hi) < 2:
        raise ValueError(f"중앙값 분할이 퇴화했습니다 (lo={len(lo)}, hi={len(hi)}) — 표본이 너무 작습니다")
    A0 = np.asarray(init_A, dtype=float)
    mu = np.vstack([lo.mean(axis=0), hi.mean(axis=0)])
    jit = float(HMM_P3["cov_jitter"]) * np.eye(2)
    cov = np.stack([_cov_of(lo) + jit, _cov_of(hi) + jit])
    # 라벨 규약(상태 1 = 수익률 분산이 큰 쪽)을 초기화 단계에서도 지킨다.
    A0, mu, cov = _relabel(A0, mu, cov)
    return HMMTheta(A=A0, mu=mu, cov=cov, pi=stationary(A0), n_obs=int(len(X)), n_iter=0, converged=False,
                    guard={"source": "deterministic_init"})


def _cov_of(X: np.ndarray) -> np.ndarray:
    d = X - X.mean(axis=0)
    return (d.T @ d) / float(len(X))


def _relabel(A: np.ndarray, mu: np.ndarray, cov: np.ndarray):
    """라벨 고정: 상태 1 = 수익률(첫 성분) 분산이 큰 쪽. 필요하면 두 상태를 맞바꾼다."""
    if float(cov[0, 0, 0]) > float(cov[1, 0, 0]):
        order = np.array([1, 0])
        A = A[np.ix_(order, order)]
        mu = mu[order]
        cov = cov[order]
    return np.asarray(A, dtype=float), np.asarray(mu, dtype=float), np.asarray(cov, dtype=float)


# ------------------------------------------------------------------
# 4.1 방출 로그밀도
# ------------------------------------------------------------------
def log_emissions(obs, theta: HMMTheta) -> np.ndarray:
    """logB[t,k] = −ln(2π) − ½ln|Σ_k| − ½(o_t−μ_k)ᵀΣ_k⁻¹(o_t−μ_k) (T×K).

    d=2 는 2×2 역행렬의 **닫힌 식 + 원소별 연산**으로만 계산한다 — 행마다 독립이라 프레임을 어디서 잘라도
    비트 동일(점 원칙 테스트가 요구). NaN 관측 행은 NaN 을 그대로 전파한다.
    """
    X = _obs_matrix(obs)
    mu = theta.mu_arr
    cov = theta.cov_arr
    K, d = mu.shape
    if X.shape[1] != d:
        raise ValueError(f"관측 차원 {X.shape[1]} ≠ θ 차원 {d}")
    out = np.empty((X.shape[0], K), dtype=float)
    for k in range(K):
        S = cov[k]
        if d == 2:
            s11, s12, s22 = float(S[0, 0]), float(S[0, 1]), float(S[1, 1])
            det = s11 * s22 - s12 * s12
            if not (det > 0.0) or not math.isfinite(det):
                raise ValueError(f"상태 {k} 의 공분산이 양정치가 아닙니다 (det={det})")
            d0 = X[:, 0] - float(mu[k, 0])
            d1 = X[:, 1] - float(mu[k, 1])
            q = (s22 * d0 * d0 - 2.0 * s12 * d0 * d1 + s11 * d1 * d1) / det
            out[:, k] = -_LOG_2PI - 0.5 * math.log(det) - 0.5 * q
        else:                                            # 일반 d (계약 밖 — 비트 동일 보장 없음)
            L = np.linalg.cholesky(S)
            z = np.linalg.solve(L, (X - mu[k]).T).T
            logdet = 2.0 * float(np.log(np.diag(L)).sum())
            out[:, k] = -0.5 * d * _LOG_2PI - 0.5 * logdet - 0.5 * (z ** 2).sum(axis=1)
    return out


def _scaled_emissions(logB: np.ndarray):
    """행별 최대값 이동(언더플로 방지): B = exp(logB − m), m = 행 최대값. 반환 (B, m, valid)."""
    valid = np.isfinite(logB).all(axis=1)
    m = np.zeros(len(logB), dtype=float)
    B = np.zeros_like(logB)
    if valid.any():
        mv = logB[valid].max(axis=1)
        m[valid] = mv
        B[valid] = np.exp(logB[valid] - mv[:, None])
    return B, m, valid


# ------------------------------------------------------------------
# 4.1 스케일드 전방 / 후방
# ------------------------------------------------------------------
def _forward_core(B: np.ndarray, A: np.ndarray, pi: np.ndarray, valid: np.ndarray):
    """전방 재귀. K=2 는 스칼라 루프(BLAS 무관·비트 재현). 관측 결측 행은 예측만 하고 c=NaN."""
    T, K = B.shape
    alpha = np.empty((T, K), dtype=float)
    c = np.empty(T, dtype=float)
    ok = valid.tolist()
    if K == 2:
        a00, a01 = float(A[0, 0]), float(A[0, 1])
        a10, a11 = float(A[1, 0]), float(A[1, 1])
        b0 = B[:, 0].tolist()
        b1 = B[:, 1].tolist()
        o0 = [0.0] * T
        o1 = [0.0] * T
        cs = [0.0] * T
        cur0 = float(pi[0])
        cur1 = float(pi[1])
        for t in range(T):
            if t == 0:
                pr0, pr1 = float(pi[0]), float(pi[1])
            else:
                pr0 = cur0 * a00 + cur1 * a10
                pr1 = cur0 * a01 + cur1 * a11
            if ok[t]:
                u0 = pr0 * b0[t]
                u1 = pr1 * b1[t]
                s = u0 + u1
                if not (s > 0.0) or not math.isfinite(s):
                    raise ValueError(f"전방 재귀가 붕괴했습니다 (t={t}, c={s}) — θ 또는 관측을 확인하라")
                cur0 = u0 / s
                cur1 = u1 / s
                cs[t] = s
            else:                                        # 관측 결측: 정보 없음 → 예측분포로 한 걸음
                cur0, cur1 = pr0, pr1
                cs[t] = float("nan")
            o0[t] = cur0
            o1[t] = cur1
        alpha[:, 0] = o0
        alpha[:, 1] = o1
        c = np.asarray(cs, dtype=float)
    else:                                                # 일반 K (계약 밖)
        cur = np.asarray(pi, dtype=float)
        for t in range(T):
            pr = np.asarray(pi, dtype=float) if t == 0 else cur @ A
            if ok[t]:
                u = pr * B[t]
                s = float(u.sum())
                if not (s > 0.0) or not math.isfinite(s):
                    raise ValueError(f"전방 재귀가 붕괴했습니다 (t={t}, c={s})")
                cur = u / s
                c[t] = s
            else:
                cur = pr
                c[t] = float("nan")
            alpha[t] = cur
    return alpha, c


def forward_filter(obs, theta: HMMTheta) -> tuple[np.ndarray, np.ndarray, float]:
    """스케일드 전방 필터. 반환 (α T×K, c T, loglik). **전방만** — 미래를 읽지 않는다.

    c_t 는 행별 최대값 이동 뒤의 정규화 상수이며 loglik = Σ_valid (ln c_t + m_t) 로 실제 로그가능도가 된다.
    관측 결측 행은 α 를 한 걸음 예측만 하고 c=NaN·가능도 기여 0 (호출자가 그 행의 확률을 NaN 으로 만든다).
    """
    logB = log_emissions(obs, theta)
    B, m, valid = _scaled_emissions(logB)
    A = theta.A_arr
    pi = theta.pi_arr
    alpha, c = _forward_core(B, A, pi, valid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ll_terms = np.log(c[valid]) + m[valid]
    loglik = float(ll_terms.sum()) if valid.any() else float("nan")
    return alpha, c, loglik


def _forward_backward(X: np.ndarray, theta: HMMTheta):
    """EM·카나리 전용 전방-후방. X 는 유한 관측만. 반환 (alpha, beta, gamma, xi, c, loglik)."""
    if not np.isfinite(X).all():
        raise ValueError("_forward_backward: 관측에 NaN/inf 가 있습니다")
    logB = log_emissions(X, theta)
    B, m, valid = _scaled_emissions(logB)
    A = theta.A_arr
    alpha, c = _forward_core(B, A, theta.pi_arr, valid)
    T, K = B.shape
    beta = np.ones((T, K), dtype=float)
    if K == 2:
        a00, a01 = float(A[0, 0]), float(A[0, 1])
        a10, a11 = float(A[1, 0]), float(A[1, 1])
        b0 = B[:, 0].tolist()
        b1 = B[:, 1].tolist()
        cl = c.tolist()
        e0 = [1.0] * T
        e1 = [1.0] * T
        for t in range(T - 2, -1, -1):
            v0 = b0[t + 1] * e0[t + 1]
            v1 = b1[t + 1] * e1[t + 1]
            ct = cl[t + 1]
            e0[t] = (a00 * v0 + a01 * v1) / ct
            e1[t] = (a10 * v0 + a11 * v1) / ct
        beta[:, 0] = e0
        beta[:, 1] = e1
    else:
        for t in range(T - 2, -1, -1):
            beta[t] = A @ (B[t + 1] * beta[t + 1]) / c[t + 1]
    gamma = alpha * beta
    gsum = gamma.sum(axis=1, keepdims=True)
    if not np.isfinite(gsum).all() or (gsum <= 0).any():
        raise ValueError("후방 재귀가 붕괴했습니다 (γ 정규화 불가)")
    gamma = gamma / gsum
    # ξ 누적: t 루프 없이 einsum 으로 (계약 §4.1)
    W = (B[1:] * beta[1:]) / c[1:, None]
    xi = np.einsum("ti,ij,tj->ij", alpha[:-1], A, W)
    with np.errstate(divide="ignore", invalid="ignore"):
        loglik = float((np.log(c) + m).sum())
    return alpha, beta, gamma, xi, c, loglik


def occupancy(obs, theta: HMMTheta) -> np.ndarray:
    """상태 점유 Σγ/T (적합 표본 위의 사후 평균; guard 입력). 유한 관측만 받는다."""
    X = _obs_matrix(obs)
    _, _, gamma, _, _, _ = _forward_backward(X, theta)
    return np.asarray(gamma.mean(axis=0), dtype=float)


# ------------------------------------------------------------------
# 4.1 EM
# ------------------------------------------------------------------
def em_fit(obs, init: HMMTheta, max_iter: int, tol: float = HMM_P3["tol"],
           jitter: float = HMM_P3["cov_jitter"]) -> HMMTheta:
    """스케일드 전방-후방 EM. 결정적(난수 없음). 종료: Δloglik < tol 또는 max_iter.

    M-스텝: A_ij = Σξ_ij / Σγ_i · μ_k = Σγ_tk o_t / Σγ_tk · Σ_k = Σγ_tk (o−μ)(o−μ)ᵀ / Σγ_tk + jitter·I
    π 는 적합하지 않고 매 반복 stationary(A) 로 둔다(퇴화 추정치 하나 제거).
    종료 뒤 라벨 고정(상태 1 = 수익률 분산이 큰 쪽) → π = stationary(A).
    theta.guard 에 {"occupancy": [...], "source": "em_fit"} 를 넣어 guard_check 가 바로 쓸 수 있게 한다.
    """
    X = _obs_matrix(obs)
    if not np.isfinite(X).all():
        raise ValueError("em_fit: 관측에 NaN/inf 가 있습니다 (호출자가 먼저 걸러야 합니다)")
    T, d = X.shape
    if T < 50:
        raise ValueError(f"em_fit: 학습 관측이 너무 적습니다 (T={T}, 최소 50)")
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError(f"max_iter 는 1 이상이어야 합니다: {max_iter}")
    tol = float(tol)
    jitter = float(jitter)

    A = init.A_arr.copy()
    mu = init.mu_arr.copy()
    cov = init.cov_arr.copy()
    pi = stationary(A)
    K = A.shape[0]
    eye = np.eye(d)
    prev_ll = -np.inf
    n_iter = 0
    converged = False
    ll = float("nan")
    for it in range(max_iter):
        cur = HMMTheta(A=A, mu=mu, cov=cov, pi=pi)
        _, _, gamma, xi, _, ll = _forward_backward(X, cur)
        if it > 0 and (ll - prev_ll) < tol:
            converged = True
            break
        prev_ll = ll
        # --- M-스텝 ---
        denom_A = gamma[:-1].sum(axis=0)
        if (denom_A <= 0).any():
            raise ValueError(f"M-스텝 붕괴: 상태 점유 0 (Σγ={denom_A.tolist()})")
        A_new = xi / denom_A[:, None]
        rs = A_new.sum(axis=1)
        if not np.allclose(rs, 1.0, atol=1e-8):
            raise ValueError(f"M-스텝 A 의 행 합이 1 이 아닙니다: {rs.tolist()}")
        gk = gamma.sum(axis=0)
        if (gk <= 0).any():
            raise ValueError(f"M-스텝 붕괴: Σγ_k = {gk.tolist()}")
        mu_new = (gamma.T @ X) / gk[:, None]
        cov_new = np.empty((K, d, d), dtype=float)
        for k in range(K):
            dev = X - mu_new[k]
            cov_new[k] = (dev * gamma[:, k, None]).T @ dev / gk[k] + jitter * eye
        A, mu, cov = A_new, mu_new, cov_new
        pi = stationary(A)
        n_iter = it + 1
    A, mu, cov = _relabel(A, mu, cov)
    pi = stationary(A)
    theta = HMMTheta(A=A, mu=mu, cov=cov, pi=pi, n_obs=int(T), loglik=float("nan"),
                     n_iter=int(n_iter), converged=bool(converged), obs_spec=init.obs_spec)
    _, _, gamma, _, _, ll_final = _forward_backward(X, theta)
    occ = gamma.mean(axis=0)
    if not converged and n_iter >= max_iter:
        warnings.warn(f"EM 이 max_iter={max_iter} 안에 수렴하지 않았습니다 (마지막 Δloglik 미달 확인 필요)",
                      RuntimeWarning)
    return HMMTheta(A=A, mu=mu, cov=cov, pi=pi, n_obs=int(T), loglik=float(ll_final),
                    n_iter=int(n_iter), converged=bool(converged), obs_spec=init.obs_spec,
                    guard={"source": "em_fit", "occupancy": [float(v) for v in occ]})


# ------------------------------------------------------------------
# 4.1 퇴화 guard
# ------------------------------------------------------------------
def guard_check(theta: HMMTheta, occupancy, prev: HMMTheta | None,
                cfg=HMM_P3["guard"]) -> tuple[bool, list[str]]:
    """퇴화 guard. 위반 시 (False, 사유들) — 호출자는 θ_{y−1} 를 유지하고 경고·로그를 남긴다.

    검사: p00, p11 ∈ [p_min, p_max] · 두 상태의 ln RV10 평균 차 ≥ gap_lnvol · 두 상태 점유 ≥ occ_min · Σ 양정치.
    """
    reasons: list[str] = []
    A = theta.A_arr
    mu = theta.mu_arr
    cov = theta.cov_arr
    p_min, p_max = float(cfg["p_min"]), float(cfg["p_max"])
    for k in range(A.shape[0]):
        pkk = float(A[k, k])
        if not (p_min <= pkk <= p_max):
            reasons.append(f"p{k}{k}={pkk:.6f} ∉ [{p_min}, {p_max}]")
    gap = float(mu[1, 1] - mu[0, 1])
    if not (gap >= float(cfg["gap_lnvol"])):
        reasons.append(f"ln RV10 평균 차 {gap:.4f} < {cfg['gap_lnvol']}")
    occ = np.asarray(occupancy, dtype=float).ravel()
    if occ.shape[0] != A.shape[0]:
        raise ValueError(f"occupancy 길이 {occ.shape[0]} ≠ 상태 수 {A.shape[0]}")
    if not np.isfinite(occ).all():
        reasons.append(f"점유에 비유한값: {occ.tolist()}")
    else:
        for k, o in enumerate(occ):
            if float(o) < float(cfg["occ_min"]):
                reasons.append(f"상태 {k} 점유 {float(o):.4f} < {cfg['occ_min']}")
    for k in range(cov.shape[0]):
        S = cov[k]
        det = float(S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]) if S.shape == (2, 2) else float(np.linalg.det(S))
        try:
            np.linalg.cholesky(S)
            pd_ok = det > 0.0
        except np.linalg.LinAlgError:
            pd_ok = False
        if not pd_ok:
            reasons.append(f"상태 {k} 공분산이 양정치가 아님 (det={det:.3e})")
    ok = not reasons
    if not ok and prev is None:
        reasons.append("유지할 이전 θ 가 없음")
    return ok, reasons


# ------------------------------------------------------------------
# 4.1 필터 · 평활(카나리 전용)
# ------------------------------------------------------------------
def filter_probabilities(obs_df: pd.DataFrame, theta: HMMTheta) -> pd.Series:
    """P_high_t = α_t[1] = P(state_t = high | o_1..o_t) — **전방만**. 점 원칙의 생산 경로.

    관측 결측 행은 NaN 이며 직전 값을 **이월하지 않는다**(조용한 실패 금지).
    프레임을 t 에서 잘라도 t 까지의 값은 비트 동일하다.
    """
    if not isinstance(obs_df, pd.DataFrame):
        raise TypeError(f"obs_df 는 DataFrame 이어야 합니다 (받은 형: {type(obs_df).__name__})")
    X = _obs_matrix(obs_df)
    alpha, _, _ = forward_filter(X, theta)
    p = np.asarray(alpha[:, 1], dtype=float).copy()
    bad = ~np.isfinite(X).all(axis=1)
    p[bad] = np.nan
    return pd.Series(p, index=obs_df.index, name="p_high")


def smoothed_probabilities(obs_df: pd.DataFrame, theta: HMMTheta) -> pd.Series:
    """γ_t[1] = P(state_t = high | o_1..o_T) — **미래를 읽는다**.

    누수 카나리 테스트 전용이다. regime.py 밖에서 호출하지 말 것(scripts/selftest.py 가 grep 으로 검사한다).
    """
    if not isinstance(obs_df, pd.DataFrame):
        raise TypeError(f"obs_df 는 DataFrame 이어야 합니다 (받은 형: {type(obs_df).__name__})")
    X = _obs_matrix(obs_df)
    good = np.isfinite(X).all(axis=1)
    out = np.full(len(X), np.nan)
    if good.any():
        _, _, gamma, _, _, _ = _forward_backward(X[good], theta)
        out[good] = gamma[:, 1]
    return pd.Series(out, index=obs_df.index, name="p_high_smoothed")


def one_step(prev_p_high: float, obs_row, theta: HMMTheta) -> float:
    """어제의 P_high 에서 한 걸음 갱신 (daily.py 의 1e-7 대조용). obs_row = [r100, ln_rv10]."""
    if prev_p_high is None or not np.isfinite(prev_p_high):
        return float("nan")
    x = np.asarray(obs_row, dtype=float).reshape(1, -1)
    if not np.isfinite(x).all():
        return float("nan")
    A = theta.A_arr
    prev = np.array([1.0 - float(prev_p_high), float(prev_p_high)], dtype=float)
    logB = log_emissions(x, theta)[0]
    B = np.exp(logB - logB.max())
    u = (prev @ A) * B
    s = float(u.sum())
    if not (s > 0.0):
        raise ValueError("one_step 재귀가 붕괴했습니다")
    return float(u[1] / s)


# ------------------------------------------------------------------
# 4.1 k-스텝 전이확률 (A 의 닫힌 식 — 표시·소거 전용)
# ------------------------------------------------------------------
def k_step(theta: HMMTheta, p_high_t: float, k: int = HMM_P3["turn_h"]) -> dict:
    """{"p_k", "q_k", "k"}.

    p_k = ([1−ξ, ξ] · A^k)[high]                              # k 세션 뒤 고변동일 확률
    q_k = 1 − [ξ·p10·p00^{k−1} + (1−ξ)·p00^k]                 # t+1..t+k 에 고변동 세션이 하나라도 있을 확률
    ξ = p_high_t (오늘 고변동 상태일 필터 확률). NaN 입력 → NaN 출력.
    """
    k = int(k)
    if k < 1:
        raise ValueError(f"k 는 1 이상이어야 합니다: {k}")
    xi = float(p_high_t) if p_high_t is not None else float("nan")
    if not np.isfinite(xi):
        return {"p_k": float("nan"), "q_k": float("nan"), "k": k}
    if not (-1e-9 <= xi <= 1.0 + 1e-9):
        raise ValueError(f"p_high_t 는 [0,1] 이어야 합니다: {xi}")
    xi = min(max(xi, 0.0), 1.0)
    A = theta.A_arr
    v = np.array([1.0 - xi, xi], dtype=float) @ np.linalg.matrix_power(A, k)
    p00 = float(A[0, 0])
    p10 = float(A[1, 0])
    stay_low = xi * p10 * (p00 ** (k - 1)) + (1.0 - xi) * (p00 ** k)
    return {"p_k": float(v[1]), "q_k": float(1.0 - stay_low), "k": k}


def _k_step_vec(theta: HMMTheta, p_high: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """k_step 의 벡터판 (walk-forward 열 채우기). 값은 k_step 과 동일한 식."""
    k = int(k)
    A = theta.A_arr
    Ak = np.linalg.matrix_power(A, k)
    xi = np.asarray(p_high, dtype=float)
    ok = np.isfinite(xi)
    p_k = np.full(xi.shape, np.nan)
    q_k = np.full(xi.shape, np.nan)
    if ok.any():
        x = np.clip(xi[ok], 0.0, 1.0)
        p_k[ok] = (1.0 - x) * Ak[0, 1] + x * Ak[1, 1]
        p00 = float(A[0, 0])
        p10 = float(A[1, 0])
        stay_low = x * p10 * (p00 ** (k - 1)) + (1.0 - x) * (p00 ** k)
        q_k[ok] = 1.0 - stay_low
    return p_k, q_k


# ------------------------------------------------------------------
# 4.1 퍼지 walk-forward
# ------------------------------------------------------------------
def hmm_walk_forward(obs_df: pd.DataFrame, refit_dates, *, purge: int = HMM_P3["purge"],
                     train_start=HMM_P3["train_start"], cfg=HMM_P3,
                     prev_thetas: list[HMMTheta] | None = None) -> tuple[pd.DataFrame, list[HMMTheta]]:
    """연 1회 퍼지 재적합 walk-forward (θ_y 가 [R_y, R_{y+1}) 을 채점).

    학습 행 = `calibrate.training_mask(idx, R_y, purge, train_start)` ∧ 관측 유한
    (pos(t) ≤ pos(R_y) − purge − 1; EM 은 라벨을 보지 않지만 모든 적합 객체의 학습 끝을 동일하게 둔다).
    init = 직전 재적합 θ(warm start; max_iter_refit), 없으면 `prev_thetas` 의 가장 최근 것, 그것도 없으면
    `deterministic_init`(max_iter_first).
    guard 위반 → θ_{y−1} 유지 + 경고 + guard_log (첫 재적합은 유지할 것이 없어 후보를 쓰고 경고만).

    반환 df: 인덱스 = 관측 세션 전체. 열
      p_high(전방 필터), p20, q20, x_hmm = logit(clip(p_high)), refit_year, theta_id, in_sample
      (R_first 이전 행은 θ_first 로 채우고 in_sample=True — Platt 학습 전용, OOS 채점에서 제외한다)
    .attrs = {guard_log, warnings, obs_spec, n_refits, seconds, purge, train_start, first_refit}
    """
    t0 = time.perf_counter()
    if not isinstance(obs_df, pd.DataFrame):
        raise TypeError(f"obs_df 는 DataFrame 이어야 합니다 (받은 형: {type(obs_df).__name__})")
    idx = obs_df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("obs_df 인덱스는 DatetimeIndex 여야 합니다")
    if not idx.is_monotonic_increasing or idx.has_duplicates:
        raise ValueError("obs_df 인덱스는 중복 없는 오름차순이어야 합니다")
    R_list = [pd.Timestamp(d) for d in refit_dates]
    if not R_list:
        raise ValueError("refit_dates 가 비어 있습니다")
    if any(R not in idx for R in R_list):
        miss = [_dstr(R) for R in R_list if R not in idx][:3]
        raise ValueError(f"재적합일이 관측 인덱스에 없습니다: {miss}")
    if list(R_list) != sorted(R_list):
        raise ValueError("refit_dates 는 오름차순이어야 합니다")

    X = _obs_matrix(obs_df)
    finite = np.isfinite(X).all(axis=1)
    turn_h = int(cfg["turn_h"])
    clip = float(cfg["clip"])
    warn_list: list[str] = []
    guard_log: list[dict] = []
    thetas: list[HMMTheta] = []

    prior = sorted([t for t in (prev_thetas or []) if t.refit_date], key=lambda t: t.refit_date)
    prev: HMMTheta | None = None
    for R in R_list:
        mask = training_mask(idx, R, int(purge), train_start) & finite
        n_train = int(mask.sum())
        if n_train < 50:
            raise ValueError(f"재적합 {_dstr(R)} 의 학습 관측이 부족합니다 (n={n_train})")
        Xtr = X[mask]
        train_idx = idx[mask]
        if prev is not None:
            init, max_iter, src = prev, int(cfg["max_iter_refit"]), f"warm:{prev.refit_date or 'prev'}"
        else:
            warm = [t for t in prior if t.refit_date < _dstr(R)]
            if warm:
                init, max_iter, src = warm[-1], int(cfg["max_iter_refit"]), f"warm_saved:{warm[-1].refit_date}"
            else:
                init, max_iter, src = deterministic_init(Xtr), int(cfg["max_iter_first"]), "deterministic_init"
        cand = em_fit(Xtr, init, max_iter, tol=float(cfg["tol"]), jitter=float(cfg["cov_jitter"]))
        if cand.n_params() != int(cfg["param_count"]):
            raise AssertionError(f"θ 파라미터 수 {cand.n_params()} ≠ {cfg['param_count']}")
        occ = np.asarray(cand.guard.get("occupancy", []), dtype=float)
        if occ.size != cand.n_states:
            occ = occupancy(Xtr, cand)
        ok, reasons = guard_check(cand, occ, prev, cfg=cfg["guard"])
        entry = {"refit_date": _dstr(R), "ok": bool(ok), "reasons": list(reasons),
                 "occupancy": [float(v) for v in occ], "n_train": n_train,
                 "train_start": _dstr(train_idx[0]), "train_end": _dstr(train_idx[-1]),
                 "n_iter": int(cand.n_iter), "converged": bool(cand.converged), "init": src,
                 "kept_prev": False, "theta_id": cand.theta_id}
        if ok:
            theta = HMMTheta(A=cand.A, mu=cand.mu, cov=cand.cov, pi=cand.pi, refit_date=_dstr(R),
                             train_start=_dstr(train_idx[0]), train_end=_dstr(train_idx[-1]), n_obs=n_train,
                             loglik=cand.loglik, n_iter=cand.n_iter, converged=cand.converged,
                             obs_spec=cand.obs_spec, guard=dict(entry))
        elif prev is not None:
            msg = f"guard 위반({_dstr(R)}): {'; '.join(reasons)} → 이전 θ({prev.refit_date}) 유지"
            warnings.warn(msg, RuntimeWarning)
            warn_list.append(msg)
            entry["kept_prev"] = True
            entry["kept_theta_id"] = prev.theta_id
            theta = HMMTheta(A=prev.A, mu=prev.mu, cov=prev.cov, pi=prev.pi, refit_date=_dstr(R),
                             train_start=prev.train_start, train_end=_dstr(train_idx[-1]), n_obs=prev.n_obs,
                             loglik=prev.loglik, n_iter=0, converged=prev.converged,
                             obs_spec=prev.obs_spec, guard=dict(entry))
        else:
            msg = (f"guard 위반({_dstr(R)}): {'; '.join(reasons)} — 유지할 이전 θ 가 없어 후보를 씁니다"
                   " (게이지는 탈락 후보로 표시할 것)")
            warnings.warn(msg, RuntimeWarning)
            warn_list.append(msg)
            theta = HMMTheta(A=cand.A, mu=cand.mu, cov=cand.cov, pi=cand.pi, refit_date=_dstr(R),
                             train_start=_dstr(train_idx[0]), train_end=_dstr(train_idx[-1]), n_obs=n_train,
                             loglik=cand.loglik, n_iter=cand.n_iter, converged=cand.converged,
                             obs_spec=cand.obs_spec, guard=dict(entry))
        guard_log.append(entry)
        thetas.append(theta)
        prev = theta

    # ---- 세그먼트별 필터 채우기 (θ_y 로 관측 처음부터 전방 재귀) ----
    n = len(idx)
    p_high = np.full(n, np.nan)
    refit_year = np.full(n, -1, dtype=int)
    theta_id = np.empty(n, dtype=object)
    in_sample = np.zeros(n, dtype=bool)
    p_k = np.full(n, np.nan)
    q_k = np.full(n, np.nan)
    pos_R = [int(idx.get_loc(R)) for R in R_list]
    for i, (R, theta) in enumerate(zip(R_list, thetas)):
        alpha, _, _ = forward_filter(X, theta)
        full = np.where(finite, alpha[:, 1], np.nan)
        start = 0 if i == 0 else pos_R[i]
        stop = pos_R[i + 1] if i + 1 < len(pos_R) else n
        sl = slice(start, stop)
        p_high[sl] = full[sl]
        refit_year[sl] = R.year
        theta_id[sl] = theta.theta_id
        in_sample[sl] = np.arange(start, stop) < pos_R[0]
        pk, qk = _k_step_vec(theta, p_high[sl], turn_h)
        p_k[sl] = pk
        q_k[sl] = qk
    x_hmm = np.full(n, np.nan)
    ok = np.isfinite(p_high)
    x_hmm[ok] = logit(np.clip(p_high[ok], clip, 1.0 - clip))

    out = pd.DataFrame({"p_high": p_high, f"p{turn_h}": p_k, f"q{turn_h}": q_k, "x_hmm": x_hmm,
                        "refit_year": refit_year, "theta_id": theta_id, "in_sample": in_sample}, index=idx)
    out.index.name = idx.name or "date"
    out.attrs = {"guard_log": guard_log, "warnings": warn_list, "obs_spec": OBS_SPEC,
                 "n_refits": len(thetas), "seconds": float(time.perf_counter() - t0),
                 "purge": int(purge), "train_start": _dstr(train_start), "first_refit": _dstr(R_list[0]),
                 "turn_h": turn_h, "clip": clip}
    return out, thetas


# ------------------------------------------------------------------
# 저장 · 불러오기 · 표
# ------------------------------------------------------------------
def regime_sha256() -> str:
    """이 파일 소스의 sha256 (registry_sha256 의 구성 요소)."""
    h = hashlib.sha256()
    h.update(Path(__file__).resolve().read_bytes())
    return h.hexdigest()


def _registry_sha() -> str:
    """`ensemble.registry_sha256()` 가 있으면 그것, 없으면 regime.py 소스 해시(단독 실행용 대체)."""
    try:
        from mrl import ensemble as _E              # noqa: WPS433 - 선택적 결합
    except Exception:                               # noqa: BLE001 - 아직 없는 모듈이면 대체 경로
        return regime_sha256()
    fn = getattr(_E, "registry_sha256", None)
    return str(fn()) if callable(fn) else regime_sha256()


def save_thetas(thetas: list[HMMTheta], path=HMM_P3_PATH, *, live: HMMTheta | None = None,
                guard_log: list | None = None, registry_sha: str | None = None,
                created_at_utc: str | None = None) -> None:
    """`results/hmm_p3.json` 저장. created_at_utc 는 메타데이터이며 결정론 비교에서 제외한다."""
    if not thetas:
        raise ValueError("저장할 θ 가 없습니다")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if guard_log is None:
        guard_log = [t.guard for t in thetas if t.guard]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "obs_spec": OBS_SPEC,
        "param_count": int(HMM_P3["param_count"]),
        "thetas": [t.to_dict() for t in thetas],
        "live": live.to_dict() if live is not None else None,
        "guard_log": _jsonable(guard_log),
        "registry_sha": registry_sha if registry_sha is not None else _registry_sha(),
        "created_at_utc": created_at_utc or dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
    }
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.write("\n")


def load_thetas(path=HMM_P3_PATH) -> tuple[list[HMMTheta], HMMTheta | None]:
    """`results/hmm_p3.json` → (재적합별 θ 목록, 라이브 θ 또는 None). schema_version 불일치는 예외."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} 없음 — scripts/run_phase3.py 를 먼저 실행하라")
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    sv = d.get("schema_version")
    if sv != SCHEMA_VERSION:
        raise ValueError(f"hmm_p3.json schema_version={sv!r} (기대 {SCHEMA_VERSION})")
    if d.get("obs_spec") != OBS_SPEC:
        raise ValueError(f"hmm_p3.json obs_spec 불일치: {d.get('obs_spec')!r} (기대 {OBS_SPEC!r})")
    thetas = [HMMTheta.from_dict(t) for t in d.get("thetas", [])]
    if not thetas:
        raise ValueError("hmm_p3.json 에 θ 가 없습니다")
    live_d = d.get("live")
    live = HMMTheta.from_dict(live_d) if live_d else None
    if live is None:
        warnings.warn("hmm_p3.json 에 live θ 가 없습니다 — daily.py 는 마지막 재적합 θ 를 써야 합니다",
                      RuntimeWarning)
    return thetas, live


def theta_table(thetas) -> pd.DataFrame:
    """θ 경로 표: refit_date, p00, p11, dur0, dur1, vol0, vol1, mu_r0, mu_r1, n_iter, guard.

    dur_k = 1/(1−p_kk) (기대 체류 세션) · vol_k = 연율 변동성(소수; 관측이 100·r 이므로 √Σ[0,0]/100·√252)
    mu_r0/1 = 상태별 일 수익 드리프트(%; 관측 단위 그대로).
    """
    rows = []
    for t in thetas:
        A = t.A_arr
        cov = t.cov_arr
        p00, p11 = float(A[0, 0]), float(A[1, 1])
        sd0 = math.sqrt(float(cov[0, 0, 0])) / 100.0
        sd1 = math.sqrt(float(cov[1, 0, 0])) / 100.0
        g = t.guard or {}
        if g.get("kept_prev"):
            guard_txt = "kept_prev: " + "; ".join(g.get("reasons", []))
        elif g.get("ok") is False:
            guard_txt = "FAIL: " + "; ".join(g.get("reasons", []))
        elif g.get("ok") is True:
            guard_txt = "ok"
        else:
            guard_txt = ""
        rows.append({"refit_date": t.refit_date, "p00": p00, "p11": p11,
                     "dur0": (1.0 / (1.0 - p00)) if p00 < 1.0 else float("inf"),
                     "dur1": (1.0 / (1.0 - p11)) if p11 < 1.0 else float("inf"),
                     "vol0": sd0 * math.sqrt(TRADING_DAYS), "vol1": sd1 * math.sqrt(TRADING_DAYS),
                     "mu_r0": float(t.mu_arr[0, 0]), "mu_r1": float(t.mu_arr[1, 0]),
                     "pi_high": float(t.pi_arr[1]), "n_iter": int(t.n_iter),
                     "theta_id": t.theta_id, "guard": guard_txt})
    return pd.DataFrame(rows)
