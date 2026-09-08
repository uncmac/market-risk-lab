# -*- coding: utf-8 -*-
"""Phase 2 보정 모델 — 4-파라미터 중첩 로지스틱 (ARCHITECTURE_PHASE2.md §7).

    logit p_t = b0 + b1·x_vix_t + b2·x_har_t + b3·x_ma_t ,  p_t = 1/(1+exp(−logit))

계약:
    PARAM_COUNT = 4 ; BUDGET = 5 ; LADDER ; ABLATIONS
    LogitModel(features, coef, intercept, refit_date, train_start, train_end, n_train, n_pos, clim,
               feature_rule, spec_sha256, model_id).predict_proba(X) / .n_params()
    n_params("M3") == 4                     # 단(rung) 이름 또는 모델을 받는 모듈 함수
    fit_logit(X, y, features, C=1.0, meta) -> LogitModel   # sklearn LogisticRegression(l2, C=1.0, lbfgs, tol 1e-8)
    save_model(m, path) / load_model(path)  # results/model_p2.json (schema_version 1)
    contributions(m, x) -> DataFrame        # 수준 귀속: 행 intercept·x_vix·x_har·x_ma, 열 logit·pp (정확 합산)
    day_over_day(m_now, x_now, m_prev, x_prev) -> dict   # 일간 귀속(할선 기울기, 재적합 항, 잔차 0)
    parameter_band(models_last5, x) -> (lo, hi)          # 최근 5회 재적합 계수를 오늘 x 에 적용한 min/max p

설계 원칙
* 적합 파라미터는 정확히 4개(절편 + 계수 3). 5번째 슬롯은 비어 있고 장부 예약(Phase 3 #3~#5).
  두 번째 Platt·isotonic 없음 — 로지스틱 자체가 보정 사상이며 (b0,b1)=(0,1),(b2,b3)=(0,0) 이면 정확히 p_vix 를 재현한다.
* 적합 사양은 사전 등록·고정: penalty="l2", C=1.0(sklearn 기본값, 조정 대상 아님), solver="lbfgs", fit_intercept=True,
  max_iter=1000, tol=1e-8, 절편 비벌칙, 클래스·표본 가중치 없음, 표준화 없음.
* 조용한 실패 금지: 입력이 NaN 인 행의 확률·귀속은 NaN (채우지 않는다). 양성 < 20 이면 ValueError.
* 결정론: lbfgs 는 같은 자료에 같은 계수를 낸다. JSON 은 float repr 로 저장해 왕복이 비트 동일.
* 이 모듈은 mrl.features / mrl.vol 을 import 하지 않는다 (features.spec_sha256 이 이 파일의 소스를 해시하므로
  느슨하게 결합). spec_sha256 은 필요할 때 지연 import 하고, features 모듈이 없으면 같은 정의로 직접 해시한다.
"""
from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from mrl.config import MODEL_P2_PATH, P2, ROOT

__all__ = [
    "PARAM_COUNT", "BUDGET", "LADDER", "ABLATIONS", "SCHEMA_VERSION", "MIN_POS", "FIT_KWARGS",
    "LogitModel", "n_params", "rung_of", "fit_logit", "save_model", "load_model",
    "contributions", "day_over_day", "parameter_band", "spec_sha256", "sigmoid", "logit",
]

# ------------------------------------------------------------------
# 사전 등록 상수
# ------------------------------------------------------------------
PARAM_COUNT = 4          # 확률 모델의 적합 파라미터 수 (b0, b1, b2, b3) — walk_forward 가 매 재적합마다 assert
BUDGET = 5               # VALIDATION §6 예산. 5번째 슬롯은 비어 있다(Phase 3 #3~#5 예약)
LADDER = {"M0": (), "M1": ("x_vix",), "M2": ("x_vix", "x_har"), "M3": ("x_vix", "x_har", "x_ma")}   # 채택 = M3
ABLATIONS = {"M3-PK": "GK+OV 대신 Parkinson 분산으로 x_har", "M3-HAR96": "학습 시작 1996-01-02",
             "R-v0": "v0 종합 s_t=-(score_d+score_w+score_m)/3 에 Platt(2) — 참조선, 2017-01-03 재적합부터(학습 ≥2역년·≥40양성)"}
SCHEMA_VERSION = 1
MIN_POS = 20             # 학습 양성 수 하한 — 이보다 적으면 적합 거부(ValueError)
FIT_KWARGS = dict(penalty="l2", C=float(P2["C"]), solver="lbfgs", fit_intercept=True, max_iter=1000, tol=1e-8)
ATTRIBUTION_ORDER = tuple(P2["features"])          # 수준 귀속의 순차 대입 순서: vix → har → ma (고정)
DEPLOY_MODES = ("info_only", "tones")
ACCEPTANCE_REF = "summary_p2.json:acceptance"
_SPEC_FILES = ("features.py", "vol.py", "model.py")   # features.spec_sha256 과 같은 정의(대체 경로)
_EPS_DLOGIT = 1e-12      # 일간 귀속에서 |Δlogit| 이 이보다 작으면 할선 대신 접선 p(1−p)


# ------------------------------------------------------------------
# 수치 유틸
# ------------------------------------------------------------------
def sigmoid(z):
    """σ(z) = 1/(1+exp(−z)) — scipy.special.expit (오버플로 없음)."""
    return expit(z)


def logit(p):
    """ln(p/(1−p)). p ∈ (0,1) 밖이면 ±inf/NaN 을 그대로 돌려준다(조용히 자르지 않는다)."""
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(p) - np.log1p(-p)


def _dstr(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _spec_sha256_fallback() -> str:
    """features 모듈이 없을 때의 대체 — mrl.features._hash_sources 와 **같은 정의**(이름 + CRLF→LF 정규화 내용,
    없는 파일은 "MISSING:<name>" 표식). 두 경로의 해시가 달라지면 model_p2.json 의 spec 검사가 어긋나므로 정의를 맞춘다."""
    h = hashlib.sha256()
    for name in _SPEC_FILES:
        p = ROOT / "mrl" / name
        h.update(p.name.encode("utf-8") + b"\n")
        if p.exists():
            h.update(p.read_bytes().replace(b"\r\n", b"\n"))
        else:
            h.update(b"MISSING:" + p.name.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def spec_sha256() -> str:
    """mrl.features.spec_sha256() 를 지연 import 로 호출. features 모듈이 아직 없으면 같은 정의로 직접 계산."""
    try:
        from mrl import features as _F          # noqa: WPS433 - 지연 import(순환 방지)
    except ImportError:
        return _spec_sha256_fallback()
    return str(_F.spec_sha256())


def feature_rule() -> str:
    """mrl.features.FEATURE_RULE (없으면 빈 문자열 — 호출자가 meta 로 넘기는 것이 원칙)."""
    try:
        from mrl import features as _F          # noqa: WPS433
    except ImportError:
        return ""
    return str(getattr(_F, "FEATURE_RULE", ""))


def rung_of(features) -> str | None:
    """특징 튜플이 사다리의 어느 단인지 (없으면 None)."""
    fs = tuple(features)
    for name, spec in LADDER.items():
        if tuple(spec) == fs:
            return name
    return None


def n_params(rung_or_model) -> int:
    """단 이름("M0".."M3") 또는 LogitModel 의 적합 파라미터 수. M0 = 0 (적합 없음), M3 = 4."""
    if isinstance(rung_or_model, LogitModel):
        return rung_or_model.n_params()
    key = str(rung_or_model)
    if key not in LADDER:
        raise KeyError(f"알 수 없는 사다리 단: {key!r} (가능: {list(LADDER)})")
    fs = LADDER[key]
    return len(fs) + 1 if fs else 0


# ------------------------------------------------------------------
# 모델
# ------------------------------------------------------------------
@dataclass
class LogitModel:
    """한 번의 재적합 결과. coef 는 features 순서의 dict, intercept 는 b0."""
    features: tuple[str, ...]
    coef: dict[str, float]
    intercept: float
    refit_date: str
    train_start: str
    train_end: str
    n_train: int
    n_pos: int
    clim: float
    feature_rule: str
    spec_sha256: str
    model_id: str                          # f"p2m3-{spec_sha256[:8]}-{refit_date}"
    deploy_mode: str = "info_only"         # "info_only" | "tones" (acceptance 결과; 저장 파일에만 의미)
    tone_model: str | None = None          # "M1" | "M3" | None
    acceptance_ref: str = ACCEPTANCE_REF
    extra: dict = field(default_factory=dict)   # 저장 파일의 부가 정보(rung 등). 확률 계산에 쓰지 않는다

    def __post_init__(self):
        self.features = tuple(self.features)
        missing = [f for f in self.features if f not in self.coef]
        if missing:
            raise ValueError(f"coef 에 특징 계수가 없습니다: {missing}")
        extra = [k for k in self.coef if k not in self.features]
        if extra:
            raise ValueError(f"coef 에 features 밖의 키가 있습니다: {extra}")
        self.coef = {f: float(self.coef[f]) for f in self.features}
        self.intercept = float(self.intercept)
        if self.deploy_mode not in DEPLOY_MODES:
            raise ValueError(f"deploy_mode 는 {DEPLOY_MODES} 중 하나여야 합니다: {self.deploy_mode!r}")

    # ---- 파라미터 ----
    def n_params(self) -> int:
        """len(coef)+1 (절편 포함). M3 == PARAM_COUNT == 4."""
        return len(self.coef) + 1

    def coef_vector(self) -> np.ndarray:
        return np.array([self.coef[f] for f in self.features], dtype=float)

    # ---- 예측 ----
    def _matrix(self, X) -> tuple[np.ndarray, pd.Index]:
        if isinstance(X, pd.Series):
            X = X.to_frame().T                                  # 한 행(특징 이름이 인덱스)
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X 는 DataFrame(또는 한 행 Series)이어야 합니다: {type(X).__name__}")
        missing = [f for f in self.features if f not in X.columns]
        if missing:
            raise ValueError(f"X 에 특징 열이 없습니다: {missing}")
        A = X[list(self.features)].to_numpy(dtype=float) if self.features else np.zeros((len(X), 0))
        return A, X.index

    def linear(self, X) -> pd.Series:
        """선형 예측치 logit p (입력 NaN 행 → NaN)."""
        A, idx = self._matrix(X)
        z = np.full(len(A), np.nan)
        fin = np.isfinite(A).all(axis=1) if A.shape[1] else np.ones(len(A), dtype=bool)
        z[fin] = self.intercept + A[fin] @ self.coef_vector()
        return pd.Series(z, index=idx, name="logit")

    def predict_proba(self, X) -> pd.Series:
        """p = σ(b0 + Σ b_k x_k). 입력 NaN 행 → NaN (조용히 채우지 않음)."""
        z = self.linear(X)
        p = pd.Series(np.where(np.isfinite(z.to_numpy()), expit(z.to_numpy()), np.nan), index=z.index, name="p")
        return p

    def predict_one(self, x: pd.Series) -> float:
        """한 행(특징 이름이 인덱스인 Series) → float p (NaN 가능)."""
        return float(self.predict_proba(x).iloc[0])

    # ---- 직렬화 ----
    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_id": self.model_id,
            "features": list(self.features),
            "coef": {f: self.coef[f] for f in self.features},
            "intercept": self.intercept,
            "clim": float(self.clim),
            "refit_date": self.refit_date,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "n_train": int(self.n_train),
            "n_pos": int(self.n_pos),
            "n_params": self.n_params(),
            "feature_rule": self.feature_rule,
            "spec_sha256": self.spec_sha256,
            "deploy_mode": self.deploy_mode,
            "tone_model": self.tone_model,
            "acceptance_ref": self.acceptance_ref,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LogitModel":
        sv = d.get("schema_version")
        if sv != SCHEMA_VERSION:
            raise ValueError(f"model_p2.json schema_version={sv!r} (기대 {SCHEMA_VERSION})")
        required = ("model_id", "features", "coef", "intercept", "clim", "refit_date", "train_start", "train_end",
                    "n_train", "n_pos", "feature_rule", "spec_sha256")
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(f"model_p2.json 에 키가 없습니다: {missing}")
        return cls(
            features=tuple(d["features"]), coef=dict(d["coef"]), intercept=float(d["intercept"]),
            refit_date=str(d["refit_date"]), train_start=str(d["train_start"]), train_end=str(d["train_end"]),
            n_train=int(d["n_train"]), n_pos=int(d["n_pos"]), clim=float(d["clim"]),
            feature_rule=str(d["feature_rule"]), spec_sha256=str(d["spec_sha256"]), model_id=str(d["model_id"]),
            deploy_mode=str(d.get("deploy_mode", "info_only")), tone_model=d.get("tone_model"),
            acceptance_ref=str(d.get("acceptance_ref", ACCEPTANCE_REF)), extra=dict(d.get("extra") or {}),
        )


# ------------------------------------------------------------------
# 적합
# ------------------------------------------------------------------
def fit_logit(X: pd.DataFrame, y: pd.Series, features, C: float = P2["C"], meta: dict | None = None) -> LogitModel:
    """L2 로지스틱 적합 (사전 등록 사양). y ∈ {0,1}; 특징·라벨이 모두 유한한 행만 사용; n_pos < 20 이면 ValueError.

    meta (선택): refit_date(없으면 학습 마지막 행 날짜), clim(없으면 학습 라벨 평균 — walk_forward 는 퍼지 학습창
    평균을 명시적으로 넘긴다), feature_rule, spec_sha256, rung(모델 id 접두), 그 밖의 키는 extra 에 보존.
    """
    feats = tuple(features)
    if not feats:
        raise ValueError("적합할 특징이 없습니다 — M0 는 적합하지 않는 p_vix 자체입니다")
    if len(set(feats)) != len(feats):
        raise ValueError(f"특징이 중복됩니다: {feats}")
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X 는 DataFrame 이어야 합니다")
    missing = [f for f in feats if f not in X.columns]
    if missing:
        raise ValueError(f"X 에 특징 열이 없습니다: {missing}")
    if len(feats) + 1 > BUDGET:
        raise ValueError(f"적합 파라미터 {len(feats) + 1} > 예산 {BUDGET}")
    Xf = X[list(feats)].astype(float)
    yy = pd.Series(y).astype(float)
    if not yy.index.equals(Xf.index):
        yy = yy.reindex(Xf.index)
    A = Xf.to_numpy(dtype=float)
    yv = yy.to_numpy(dtype=float)
    ok = np.isfinite(A).all(axis=1) & np.isfinite(yv)
    A, yv = A[ok], yv[ok]
    n = int(len(yv))
    if n == 0:
        raise ValueError("유한한 학습 행이 없습니다")
    if not np.isin(np.unique(yv), [0.0, 1.0]).all():
        raise ValueError("y 는 {0,1} 이어야 합니다")
    n_pos = int(yv.sum())
    if n_pos < MIN_POS:
        raise ValueError(f"학습 양성 {n_pos} < {MIN_POS} — 적합 거부")
    if n_pos == n:
        raise ValueError("학습 라벨이 전부 양성이라 적합할 수 없습니다")

    kw = dict(FIT_KWARGS)
    kw["C"] = float(C)
    lr = LogisticRegression(**kw)
    lr.fit(A, yv)
    n_iter = int(np.asarray(lr.n_iter_).max())
    if n_iter >= kw["max_iter"]:
        warnings.warn(f"로지스틱 적합이 max_iter={kw['max_iter']} 에 도달했습니다 (수렴 의심)", RuntimeWarning)
    coef = {f: float(c) for f, c in zip(feats, lr.coef_[0])}
    intercept = float(lr.intercept_[0])

    meta = dict(meta or {})
    train_idx = Xf.index[ok]
    train_start = _dstr(train_idx[0]) if isinstance(train_idx, pd.DatetimeIndex) else str(train_idx[0])
    train_end = _dstr(train_idx[-1]) if isinstance(train_idx, pd.DatetimeIndex) else str(train_idx[-1])
    refit_date = meta.pop("refit_date", None)
    refit_date = train_end if refit_date is None else _dstr(refit_date)
    clim = meta.pop("clim", None)
    clim = float(yv.mean()) if clim is None else float(clim)
    if not math.isfinite(clim):
        raise ValueError("clim 이 유한하지 않습니다")
    sha = meta.pop("spec_sha256", None)
    sha = spec_sha256() if sha is None else str(sha)
    rule = meta.pop("feature_rule", None)
    rule = feature_rule() if rule is None else str(rule)
    rung = meta.pop("rung", None) or rung_of(feats) or "x"
    model_id = f"p2{str(rung).lower()}-{sha[:8]}-{refit_date}"
    extra = {"rung": str(rung), "C": float(C), "n_iter": n_iter, **meta}
    return LogitModel(features=feats, coef=coef, intercept=intercept, refit_date=refit_date,
                      train_start=train_start, train_end=train_end, n_train=n, n_pos=n_pos, clim=clim,
                      feature_rule=rule, spec_sha256=sha, model_id=model_id, extra=extra)


# ------------------------------------------------------------------
# 저장 / 적재
# ------------------------------------------------------------------
def save_model(m: LogitModel, path: Path = MODEL_P2_PATH, *, deploy_mode: str | None = None,
               tone_model: str | None = None) -> None:
    """model_p2.json 저장 (UTF-8, LF, float repr 그대로 → 왕복 비트 동일). NaN 이 있으면 저장 거부."""
    if deploy_mode is not None:
        if deploy_mode not in DEPLOY_MODES:
            raise ValueError(f"deploy_mode 는 {DEPLOY_MODES} 중 하나: {deploy_mode!r}")
        m.deploy_mode = deploy_mode
    if tone_model is not None or deploy_mode is not None:
        m.tone_model = tone_model
    d = m.to_dict()
    d["created_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    text = json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")


def load_model(path: Path = MODEL_P2_PATH) -> LogitModel:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return LogitModel.from_dict(d)


# ------------------------------------------------------------------
# 귀속 (정확 합산)
# ------------------------------------------------------------------
def _x_vector(m: LogitModel, x: pd.Series) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
        if len(x) != 1:
            raise ValueError("x 는 한 행이어야 합니다")
        x = x.iloc[0]
    x = pd.Series(x)
    missing = [f for f in m.features if f not in x.index]
    if missing:
        raise ValueError(f"x 에 특징이 없습니다: {missing}")
    return x.reindex(list(m.features)).to_numpy(dtype=float)


def contributions(m: LogitModel, x: pd.Series) -> pd.DataFrame:
    """수준 귀속. 행: intercept, 그 뒤 m.features 순서(vix→har→ma 고정). 열:
       logit — b_k·x_k (절편 행은 b0). Σ logit == logit p.
       pp    — 순차 대입 p_k = σ(b0 + Σ_{j≤k} b_j x_j), pp_k = p_k − p_{k−1}; 절편 행은 σ(b0).
               따라서 σ(b0) + Σ_{특징} pp == p (1e-12), 그리고 pp 열 전체 합 == p.
    입력 NaN → 특징 행 전부 NaN (절편 행만 유효). .attrs = {"p", "logit"}."""
    xv = _x_vector(m, x)
    b = m.coef_vector()
    rows = ["intercept"] + list(m.features)
    terms = np.concatenate([[m.intercept], b * xv])
    if not np.isfinite(terms).all():
        out = pd.DataFrame({"logit": terms, "pp": np.full(len(rows), np.nan)}, index=rows)
        out.loc["intercept", "pp"] = float(expit(m.intercept))
        out.attrs = {"p": float("nan"), "logit": float("nan")}
        return out
    cum = np.cumsum(terms)
    p_k = expit(cum)
    pp = np.concatenate([[p_k[0]], np.diff(p_k)])
    out = pd.DataFrame({"logit": terms, "pp": pp}, index=rows)
    out.index.name = "term"
    out.attrs = {"p": float(p_k[-1]), "logit": float(cum[-1])}
    return out


def _gap_sessions(x_prev: pd.Series, x_now: pd.Series) -> int | None:
    """두 행의 이름(날짜)으로 (prev, now] 거래일 수. 이름이 날짜가 아니면 None (지어내지 않는다)."""
    try:
        a = pd.Timestamp(x_prev.name)
        b = pd.Timestamp(x_now.name)
    except (TypeError, ValueError):
        return None
    if pd.isna(a) or pd.isna(b) or b <= a:
        return None
    from mrl.calendar_us import trading_days           # noqa: WPS433 - 지연 import
    days = trading_days(a + pd.Timedelta(days=1), b)
    return int(len(days))


def day_over_day(m_now: LogitModel, x_now: pd.Series, m_prev: LogitModel, x_prev: pd.Series,
                 gap_sessions: int | None = None) -> dict:
    """일간 귀속.
       Δlogit 항: 특징 k: b_k^now·(x_k^now − x_k^prev) ; refit: (b0^now − b0^prev) + Σ_k (b_k^now − b_k^prev)·x_k^prev
                  (재적합 없으면 정확히 0). 항의 합 == logit_now − logit_prev.
       pp 항: pp_k = m·Δlogit_k, m = (p_now − p_prev)/(logit_now − logit_prev) (할선 기울기; |Δlogit| < 1e-12 면 p(1−p)).
              → Σ pp == Δp 정확(잔차 0), 순서 무관.
       반환 {"d_logit": {...}, "d_pp": {...}, "d_p", "p_now", "p_prev", "logit_now", "logit_prev", "refit", "gap_sessions"}.
       입력 NaN 이면 수치는 NaN (refit·gap 은 그대로)."""
    if tuple(m_now.features) != tuple(m_prev.features):
        raise ValueError(f"두 모델의 특징이 다릅니다: {m_now.features} vs {m_prev.features}")
    feats = list(m_now.features)
    xn, xp = _x_vector(m_now, x_now), _x_vector(m_prev, x_prev)
    bn, bp = m_now.coef_vector(), m_prev.coef_vector()
    b0n, b0p = m_now.intercept, m_prev.intercept
    refit = (m_now.model_id != m_prev.model_id) or (b0n != b0p) or not np.array_equal(bn, bp)
    gap = gap_sessions if gap_sessions is not None else _gap_sessions(x_prev, x_now)

    keys = feats + ["refit"]
    if not (np.isfinite(xn).all() and np.isfinite(xp).all()):
        nan = {k: float("nan") for k in keys}
        return {"d_logit": nan, "d_pp": dict(nan), "d_p": float("nan"), "p_now": float("nan"), "p_prev": float("nan"),
                "logit_now": float("nan"), "logit_prev": float("nan"), "refit": bool(refit), "gap_sessions": gap}

    logit_now = float(b0n + bn @ xn)
    logit_prev = float(b0p + bp @ xp)
    p_now, p_prev = float(expit(logit_now)), float(expit(logit_prev))
    d_logit = {f: float(bn[i] * (xn[i] - xp[i])) for i, f in enumerate(feats)}
    d_logit["refit"] = float((b0n - b0p) + (bn - bp) @ xp) if refit else 0.0
    dl = logit_now - logit_prev
    if abs(dl) < _EPS_DLOGIT:
        slope = p_now * (1.0 - p_now)
    else:
        slope = (p_now - p_prev) / dl
    d_pp = {k: float(slope * v) for k, v in d_logit.items()}
    return {"d_logit": d_logit, "d_pp": d_pp, "d_p": p_now - p_prev, "p_now": p_now, "p_prev": p_prev,
            "logit_now": logit_now, "logit_prev": logit_prev, "refit": bool(refit), "gap_sessions": gap}


def parameter_band(models_last5: list[LogitModel], x: pd.Series) -> tuple[float, float]:
    """최근(≤5회) 연간 재적합 계수 + 라이브 모델을 오늘 x 에 적용한 확률의 (min, max).
    목록에 현재 모델을 포함시켜야 현재 p 가 밴드 안에 든다(계약 ARCHITECTURE_PHASE2.md §17·§387).
    상한은 '재적합 5개 + 라이브 1개' = 6개다 — 5 로 자르면 호출자가 넘긴 가장 오래된 재적합이 조용히 사라져
    밴드가 라벨('최근 5회 재적합+라이브')보다 좁아진다. 빈 목록이면 ValueError, 어느 하나라도 NaN 이면 (nan, nan)."""
    models = list(models_last5)
    if not models:
        raise ValueError("parameter_band: 모델 목록이 비어 있습니다")
    cap = int(P2["param_band_refits"]) + 1                     # 재적합 5회 + 라이브 1개
    if len(models) > cap:
        models = models[-cap:]
    ps = np.array([m.predict_one(x) for m in models], dtype=float)
    if not np.isfinite(ps).all():
        return (float("nan"), float("nan"))
    return (float(ps.min()), float(ps.max()))
