# -*- coding: utf-8 -*-
"""mrl.vol — 실현변동성 (ARCHITECTURE_PHASE2.md §5 계약).

역할
  * Garman–Klass + 야간갭(GK+OV) 일간 분산 `v_t` — 분산 하한 `var_floor`=1e-8 (=(1bp 일변동)^2).
  * HAR 성분 rv1/rv5/rv22 = √(252·mean(v_{t-k+1..t})) (연율화 소수) 와 고정 동일가중 log-HAR
    (ln σ_har = Σ w_k ln rv_k, w=(1/3,1/3,1/3), 적합 아님) — 모델 특징 `x_har` 전용.
  * 보조 출력: 적합 log-HAR (OLS 4개; 확률 예산 밖 — 확률 모델에 절대 결합하지 않는다) 와 그 walk-forward·성적표.
  * 자기점검: (GK+OV)/CC 비율, Parkinson/CC 비율, 1993~95 OHLC 품질(시가==고/저가 비중·중앙 로그 범위).
  * Parkinson 분산은 소거실험 M3-PK 와 자기점검 전용이다.

식 (사전 등록 상수 — 적합 아님)
  GK_t = 0.5·ln(H_t/L_t)^2 − (2ln2 − 1)·ln(C_t/O_t)^2
  OV_t = ln(O_t / C_{t−1})^2
  v_t  = max(GK_t + OV_t, floor)                         # overnight=False 면 GK_t 만
  PK_t = ln(H_t/L_t)^2 / (4 ln 2)
  rv_k[t] = √(252 · max(mean(v_{t−k+1..t}), floor)),  k ∈ (1, 5, 22), min_periods = k
  ln σ_har = Σ_k w_k · ln rv_k                            # w = (1/3, 1/3, 1/3)
  야간갭이 필수인 이유: SPY 야간 분산 비중이 1990년대 ~20% → 2020년대 ~45% 로 올라 GK 단독은 종가 변동성 대비 표류한다.

원칙
  * 점(point-in-time): 모든 창은 뒤를 보는(trailing) 창이다. t 이후 행을 덧붙여도 t 까지의 값은 비트 동일
    (tests/test_vol.py 가 검사). 적합 HAR 은 재적합일 R 에 대해 pos(t) ≤ pos(R) − (purge+1) 행만 학습한다
    (라벨 창 t+1..t+h 가 R 전에 완전히 실현).
  * 조용한 실패 금지: 형식 오류·0 이하 가격·표본 부족은 예외, 데이터 한계는 warnings.warn. NaN 입력은 NaN 으로
    전파하고 절대 값을 지어내지 않는다.
  * 한국어 주석, 영어 식별자.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from mrl.config import P2

__all__ = [
    "GK_COEF", "OHLC_COLUMNS", "TRADING_DAYS", "HAR_PARAM_COUNT", "HAR_COEF_NAMES",
    "RATIO_250_WINDOW", "RATIO_250_BOUNDS", "RATIO_CHECK_START", "EARLY_OHLC_END", "PK_CC_WINDOW", "PK_CC_BOUNDS",
    "garman_klass_variance", "parkinson_variance", "close_to_close_variance",
    "har_components", "har_log_vol",
    "har_target", "har_fit", "har_predict", "har_walk_forward", "vol_scorecard", "auc",
    "ratio_checks",
]

TRADING_DAYS = 252
GK_COEF = 2.0 * math.log(2.0) - 1.0            # 0.386294… (Garman–Klass 의 ln(C/O)^2 계수)
PK_DENOM = 4.0 * math.log(2.0)                 # Parkinson 분모
OHLC_COLUMNS = ("Open", "High", "Low", "Close")
HAR_PARAM_COUNT = 4                            # 보조 출력 OLS 계수 수 (절편 + 3) — 확률 모델 PARAM_COUNT 와 무관
HAR_COEF_NAMES = ("const", "ln_rv1", "ln_rv5", "ln_rv22")
HAR_MIN_TRAIN_ROWS = 100                       # 이보다 적으면 적합을 거부한다 (조용한 실패 금지)
# 자기점검 상수 (§5 ratio_checks)
RATIO_250_WINDOW = 250
RATIO_250_BOUNDS = (0.8, 1.3)                  # rolling-250 (GK+OV)/CC 변동성 비율 허용 범위 (1996년 이후)
RATIO_CHECK_START = "1996-01-01"               # 1993~95 는 OHLC 품질이 달라 보고만 한다
EARLY_OHLC_END = "1995-12-31"
PK_CC_WINDOW = 20
PK_CC_BOUNDS = (0.3, 3.0)                      # Parkinson RV20 / CC RV20 허용 범위


# ------------------------------------------------------------------
# 입력 검증
# ------------------------------------------------------------------
def _check_index(obj, name: str) -> None:
    if not isinstance(obj.index, pd.DatetimeIndex):
        raise TypeError(f"{name} 인덱스는 DatetimeIndex 여야 함 (받은 형: {type(obj.index).__name__})")
    if obj.index.tz is not None:
        raise ValueError(f"계약 위반: {name} 인덱스는 tz-naive 여야 함")
    if not obj.index.is_monotonic_increasing:
        raise ValueError(f"{name} 인덱스가 오름차순이 아님")
    if obj.index.has_duplicates:
        dup = obj.index[obj.index.duplicated()][:3].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"{name} 인덱스에 중복 날짜가 있음 (예: {dup})")


def _check_ohlc(ohlc: pd.DataFrame) -> pd.DataFrame:
    """OHLC 프레임 검증: 열 존재, DatetimeIndex, 오름차순·중복 없음, 0 이하 가격 없음(NaN 은 허용 → NaN 전파)."""
    if not isinstance(ohlc, pd.DataFrame):
        raise TypeError(f"ohlc 는 DataFrame 이어야 함 (받은 형: {type(ohlc).__name__})")
    missing = [c for c in OHLC_COLUMNS if c not in ohlc.columns]
    if missing:
        raise ValueError(f"ohlc 에 열이 없음: {missing} (열: {list(ohlc.columns)})")
    _check_index(ohlc, "ohlc")
    px = ohlc[list(OHLC_COLUMNS)].astype(float)
    bad = (px <= 0).any(axis=1)
    if bad.any():
        ex = px.index[bad][:3].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"ohlc 에 0 이하 가격이 있음 (예: {ex})")
    return px


def _check_positive_series(s: pd.Series, name: str) -> pd.Series:
    if not isinstance(s, pd.Series):
        raise TypeError(f"{name} 는 Series 여야 함 (받은 형: {type(s).__name__})")
    _check_index(s, name)
    v = s.astype(float)
    if (v <= 0).any():
        ex = v.index[v <= 0][:3].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"{name} 에 0 이하 값이 있음 (예: {ex})")
    return v


def _check_floor(floor: float) -> float:
    floor = float(floor)
    if not (floor > 0) or not math.isfinite(floor):
        raise ValueError(f"floor 는 양의 유한값이어야 함: {floor!r}")
    return floor


# ------------------------------------------------------------------
# 일간 분산 추정량
# ------------------------------------------------------------------
def garman_klass_variance(ohlc: pd.DataFrame, overnight: bool = True, floor: float = P2["var_floor"]) -> pd.Series:
    """GK_t = 0.5·ln(H/L)^2 − (2ln2−1)·ln(C/O)^2 ; OV_t = ln(O_t/C_{t−1})^2 ; v_t = max(GK_t + OV_t, floor).

    overnight=False 면 GK_t 만(하한은 동일 적용). 첫 행은 C_{t−1} 이 없어 overnight=True 일 때 NaN.
    NaN 이 있는 행은 NaN (채우지 않는다). 반환 이름: "var_gkov" / "var_gk".
    """
    px = _check_ohlc(ohlc)
    floor = _check_floor(floor)
    o, h, l, c = (px[k] for k in OHLC_COLUMNS)
    gk = 0.5 * np.log(h / l) ** 2 - GK_COEF * np.log(c / o) ** 2
    if overnight:
        ov = np.log(o / c.shift(1)) ** 2
        v = gk + ov
        name = "var_gkov"
    else:
        v = gk
        name = "var_gk"
    v = v.clip(lower=floor)            # NaN 은 NaN 그대로
    v.name = name
    return v


def parkinson_variance(ohlc: pd.DataFrame, floor: float = P2["var_floor"]) -> pd.Series:
    """Parkinson 분산 ln(H/L)^2 / (4 ln 2), 하한 floor. 소거실험 M3-PK 와 자기점검 전용 (모델 특징 아님)."""
    px = _check_ohlc(ohlc)
    floor = _check_floor(floor)
    v = (np.log(px["High"] / px["Low"]) ** 2 / PK_DENOM).clip(lower=floor)
    v.name = "var_pk"
    return v


def close_to_close_variance(close: pd.Series, floor: float = P2["var_floor"]) -> pd.Series:
    """종가 로그수익률 제곱 (자기점검 비교용). 첫 행 NaN."""
    c = _check_positive_series(close, "close")
    floor = _check_floor(floor)
    v = (np.log(c).diff() ** 2).clip(lower=floor)
    v.name = "var_cc"
    return v


# ------------------------------------------------------------------
# HAR 성분 · 고정가중 log-HAR (모델 특징 전용)
# ------------------------------------------------------------------
def har_components(var: pd.Series, lookbacks: tuple = P2["har_lookbacks"], floor: float = P2["var_floor"]) -> pd.DataFrame:
    """rv_k[t] = √(252 · max(mean(var[t−k+1..t]), floor)), min_periods = k. 열 rv{k} (연율화 %가 아니라 소수).

    var 는 일간 분산(NaN 허용 → 창 안에 NaN 이 있으면 그 행 NaN). 뒤를 보는 창이라 미래 행에 영향받지 않는다.
    """
    if not isinstance(var, pd.Series):
        raise TypeError(f"var 는 Series 여야 함 (받은 형: {type(var).__name__})")
    _check_index(var, "var")
    floor = _check_floor(floor)
    lbs = tuple(int(k) for k in lookbacks)
    if not lbs or any(k <= 0 for k in lbs) or len(set(lbs)) != len(lbs):
        raise ValueError(f"lookbacks 는 서로 다른 양의 정수여야 함: {lookbacks!r}")
    v = var.astype(float)
    if (v.dropna() < 0).any():
        raise ValueError("var 에 음수 분산이 있음 — garman_klass_variance 의 출력을 넘겨라")
    out = pd.DataFrame(index=v.index)
    for k in lbs:
        m = v.rolling(k, min_periods=k).mean().clip(lower=floor)
        out[f"rv{k}"] = np.sqrt(TRADING_DAYS * m)
    return out


def har_log_vol(comp: pd.DataFrame, weights: tuple = P2["har_weights"]) -> pd.Series:
    """ln σ_har = Σ_k w_k · ln rv_k — 고정가중(적합 아님), 모델 특징 x_har 전용. 가중치는 합 1 이어야 한다.

    comp 의 열 순서(rv1, rv5, rv22)와 weights 순서가 대응한다. 어느 성분이든 NaN 이면 NaN.
    """
    if not isinstance(comp, pd.DataFrame):
        raise TypeError(f"comp 는 DataFrame 이어야 함 (받은 형: {type(comp).__name__})")
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) != comp.shape[1]:
        raise ValueError(f"weights 길이 {len(w)} ≠ 성분 수 {comp.shape[1]}")
    if not np.isfinite(w).all() or (w < 0).any() or abs(w.sum() - 1.0) > 1e-12:
        raise ValueError(f"weights 는 음이 아니고 합이 1 이어야 함: {weights!r}")
    vals = comp.to_numpy(dtype=float)
    if (vals[np.isfinite(vals)] <= 0).any():
        raise ValueError("comp 에 0 이하 rv 가 있음 — har_components 의 출력을 넘겨라 (하한 적용)")
    ln = np.log(vals)
    out = pd.Series(ln @ w, index=comp.index, name="ln_har_vol")
    return out


# ------------------------------------------------------------------
# 보조 출력: 적합 log-HAR (OLS 4개 — 확률 예산 밖, 확률에 결합 금지)
# ------------------------------------------------------------------
def har_target(close: pd.Series, h: int = P2["h"]) -> pd.Series:
    """ln( std(logret_{t+1..t+h}) · √252 ) — targets 의 종가 RV 규약(20일 로그수익률 std·√252)과 동일. 마지막 h 행 NaN.

    실현변동성이 0 이면(가격 불변) 로그가 정의되지 않아 NaN 으로 두고 경고한다.
    """
    c = _check_positive_series(close, "close")
    h = int(h)
    if h <= 1:
        raise ValueError(f"h 는 2 이상이어야 함: {h}")
    logret = np.log(c).diff()
    rv = logret.rolling(h, min_periods=h).std() * math.sqrt(TRADING_DAYS)
    fwd = rv.shift(-h)                                   # t+1..t+h 의 std → t 행
    nonpos = fwd.notna() & (fwd <= 0)
    if nonpos.any():
        warnings.warn(f"har_target: 실현변동성 0 인 창 {int(nonpos.sum())}행 → NaN (가격 불변)")
        fwd = fwd.where(~nonpos)
    out = np.log(fwd)
    out.name = "ln_rv_fwd"
    return out


def _har_design(comp: pd.DataFrame) -> np.ndarray:
    """[1, ln rv1, ln rv5, ln rv22] 설계행렬 (NaN 은 그대로 → 호출자가 유한 행만 고른다)."""
    cols = [f"rv{k}" for k in P2["har_lookbacks"]]
    missing = [c for c in cols if c not in comp.columns]
    if missing:
        raise ValueError(f"comp 에 열이 없음: {missing} (열: {list(comp.columns)})")
    vals = comp[cols].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ln = np.log(vals)
    return np.column_stack([np.ones(len(comp)), ln])


def har_fit(X: pd.DataFrame, y_ln: pd.Series, refit_date=None) -> dict:
    """OLS  ln RV20_fwd ~ 1 + ln rv1 + ln rv5 + ln rv22  →  {coef(4), n, refit_date, train_start, train_end, resid_std}.

    X 는 har_components 의 출력(rv 수준; 로그는 안에서 취한다), y_ln 은 har_target 의 출력. 유한 행만 쓴다.
    표본이 HAR_MIN_TRAIN_ROWS 미만이면 ValueError. 계수는 정확히 HAR_PARAM_COUNT(4)개 — 확률 모델 예산과는 별개.
    """
    if not isinstance(y_ln, pd.Series):
        raise TypeError("y_ln 은 Series 여야 함")
    if not X.index.equals(y_ln.index):
        y_ln = y_ln.reindex(X.index)
    A = _har_design(X)
    y = y_ln.to_numpy(dtype=float)
    ok = np.isfinite(A).all(axis=1) & np.isfinite(y)
    n = int(ok.sum())
    if n < HAR_MIN_TRAIN_ROWS:
        raise ValueError(f"har_fit: 유한 학습 행 {n} < {HAR_MIN_TRAIN_ROWS}")
    beta, _res, rank, _sv = np.linalg.lstsq(A[ok], y[ok], rcond=None)
    if rank < A.shape[1]:
        raise ValueError(f"har_fit: 설계행렬 계수 부족(rank {rank} < {A.shape[1]}) — 성분이 공선적")
    resid = y[ok] - A[ok] @ beta
    coef = {k: float(b) for k, b in zip(HAR_COEF_NAMES, beta)}
    assert len(coef) == HAR_PARAM_COUNT
    idx_ok = X.index[ok]
    return {
        "coef": coef, "n": n,
        "refit_date": None if refit_date is None else pd.Timestamp(refit_date).strftime("%Y-%m-%d"),
        "train_start": idx_ok[0].strftime("%Y-%m-%d"), "train_end": idx_ok[-1].strftime("%Y-%m-%d"),
        "resid_std": float(np.std(resid, ddof=A.shape[1])),
    }


def har_predict(coef: dict, comp: pd.DataFrame) -> pd.Series:
    """계수 dict(const, ln_rv1, ln_rv5, ln_rv22) 로 ln σ̂ 예측. 성분 NaN 행은 NaN."""
    missing = [k for k in HAR_COEF_NAMES if k not in coef]
    if missing:
        raise ValueError(f"coef 에 키가 없음: {missing}")
    beta = np.array([float(coef[k]) for k in HAR_COEF_NAMES])
    A = _har_design(comp)
    pred = A @ beta
    pred[~np.isfinite(A).all(axis=1)] = np.nan
    return pd.Series(pred, index=comp.index, name="ln_har_fc")


def har_walk_forward(comp: pd.DataFrame, y_ln: pd.Series, refit_dates, purge: int = P2["purge"],
                     train_start: str = "1993-03-03") -> tuple[pd.Series, pd.DataFrame]:
    """확률 모델과 같은 일정의 연 1회 재적합 + 퍼지. 반환 (ln 예측 Series, 계수 표).

    재적합일 R 의 학습 행: index ≥ train_start, 성분·라벨 유한, pos(t) ≤ pos(R) − (purge+1)
    (라벨 창 t+1..t+h 가 R 전에 완전히 실현). R 의 계수가 [R, 다음 R) 을 예측하고, 마지막 R 은 끝까지.
    첫 R 이전 행은 NaN. R 이 인덱스에 없으면 R 이후 첫 세션을 쓰고 경고한다. 민감도: train_start=P2["har_train_start_sensitivity"].
    """
    if not isinstance(comp, pd.DataFrame):
        raise TypeError("comp 는 DataFrame 이어야 함")
    _check_index(comp, "comp")
    if not comp.index.equals(y_ln.index):
        y_ln = y_ln.reindex(comp.index)
    purge = int(purge)
    if purge < 0:
        raise ValueError(f"purge 는 0 이상이어야 함: {purge}")
    idx = comp.index
    rds = sorted({pd.Timestamp(r).normalize() for r in refit_dates})
    if not rds:
        raise ValueError("refit_dates 가 비어 있음")
    start_ts = pd.Timestamp(train_start).normalize()
    positions = []
    for r in rds:
        pos = int(idx.searchsorted(r, side="left"))
        if pos >= len(idx):
            warnings.warn(f"har_walk_forward: 재적합일 {r:%Y-%m-%d} 이 인덱스 끝 이후 → 건너뜀")
            continue
        if idx[pos] != r:
            warnings.warn(f"har_walk_forward: 재적합일 {r:%Y-%m-%d} 이 세션이 아님 → {idx[pos]:%Y-%m-%d} 사용")
        positions.append(pos)
    if not positions:
        raise ValueError("유효한 재적합일이 없음")
    positions = sorted(set(positions))
    A = _har_design(comp)
    y = y_ln.to_numpy(dtype=float)
    finite = np.isfinite(A).all(axis=1) & np.isfinite(y) & (idx >= start_ts)
    pos_arr = np.arange(len(idx))
    fc = np.full(len(idx), np.nan)
    rows = []
    for i, pos in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(idx)
        mask = finite & (pos_arr <= pos - (purge + 1))
        n = int(mask.sum())
        if n < HAR_MIN_TRAIN_ROWS:
            raise ValueError(f"har_walk_forward: {idx[pos]:%Y-%m-%d} 학습 행 {n} < {HAR_MIN_TRAIN_ROWS}")
        beta, _r, rank, _s = np.linalg.lstsq(A[mask], y[mask], rcond=None)
        if rank < A.shape[1]:
            raise ValueError(f"har_walk_forward: {idx[pos]:%Y-%m-%d} 설계행렬 계수 부족")
        pred = A[pos:end] @ beta
        pred[~np.isfinite(A[pos:end]).all(axis=1)] = np.nan
        fc[pos:end] = pred
        tr_idx = idx[mask]
        rows.append({"refit_date": idx[pos].strftime("%Y-%m-%d"),
                     **{k: float(b) for k, b in zip(HAR_COEF_NAMES, beta)},
                     "n": n, "train_start": tr_idx[0].strftime("%Y-%m-%d"), "train_end": tr_idx[-1].strftime("%Y-%m-%d"),
                     "purge": purge})
    table = pd.DataFrame(rows, columns=["refit_date", *HAR_COEF_NAMES, "n", "train_start", "train_end", "purge"])
    return pd.Series(fc, index=idx, name="ln_har_fc"), table


# ------------------------------------------------------------------
# 성적표
# ------------------------------------------------------------------
def auc(score, y) -> float:
    """ROC AUC (Mann–Whitney, 동순위 평균). 유한 쌍만 사용. 양성/음성이 하나라도 없으면 NaN."""
    s = np.asarray(score, dtype=float)
    yy = np.asarray(y, dtype=float)
    if s.shape != yy.shape:
        raise ValueError(f"auc: 길이 불일치 {s.shape} vs {yy.shape}")
    ok = np.isfinite(s) & np.isfinite(yy)
    s, yy = s[ok], yy[ok]
    n1 = int((yy == 1).sum())
    n0 = int((yy == 0).sum())
    if n1 + n0 != len(yy):
        raise ValueError("auc: y 는 0/1 이어야 함")
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _log_scores(ln_fc: np.ndarray, ln_rv: np.ndarray) -> dict:
    """MSE(log) · QLIKE = RV²/σ̂² − ln(RV²/σ̂²) − 1 · R²(log). 유한 쌍만."""
    ok = np.isfinite(ln_fc) & np.isfinite(ln_rv)
    if ok.sum() < 2:
        return {"mse_log": np.nan, "qlike": np.nan, "r2_log": np.nan}
    f, r = ln_fc[ok], ln_rv[ok]
    e = r - f
    ratio = np.exp(2.0 * e)                      # RV² / σ̂²
    qlike = ratio - 2.0 * e - 1.0
    sst = float(np.sum((r - r.mean()) ** 2))
    return {"mse_log": float(np.mean(e ** 2)), "qlike": float(np.mean(qlike)),
            "r2_log": float(1.0 - np.sum(e ** 2) / sst) if sst > 0 else np.nan}


def _reconstruct_y_vol(ln_rv_fwd: pd.Series, h: int = P2["h"], ref_window: int = 252) -> pd.Series:
    """targets.y_vol_20 재구성: RV_fwd[t] > median(RV_trailing[t−251..t]), RV_trailing[t] = RV_fwd[t−h]."""
    rv_fwd = np.exp(ln_rv_fwd)
    rv_trail = rv_fwd.shift(h)
    ref = rv_trail.rolling(ref_window, min_periods=ref_window).median()
    valid = rv_fwd.notna() & ref.notna()
    return (rv_fwd > ref).astype(float).where(valid)


def vol_scorecard(ln_fc: pd.Series, ln_rv_fwd: pd.Series, vix: pd.Series, rv22: pd.Series, blocks,
                  y_vol: pd.Series | None = None) -> pd.DataFrame:
    """블록별·전체 HAR 예측 성적표.

    각 블록 [a, b) 과 'all'(블록 합집합): n, n_blocks(=n//20), 그리고 세 예측 — model(ln_fc), vix(σ̂=VIX/100), rv22(지속) —
    각각의 mse_log, qlike, r2_log, auc_vol(y_vol_20 판별). y_vol 을 주지 않으면 ln_rv_fwd 에서 targets 규약대로 재구성한다.
    """
    idx = ln_fc.index
    _check_index(ln_fc, "ln_fc")
    rv = ln_rv_fwd.reindex(idx).to_numpy(dtype=float)
    fc = ln_fc.to_numpy(dtype=float)
    v = vix.reindex(idx).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ln_vix = np.log(v / 100.0)
        ln_rv22 = np.log(rv22.reindex(idx).to_numpy(dtype=float))
    if y_vol is None:
        yv = _reconstruct_y_vol(ln_rv_fwd.reindex(idx)).to_numpy(dtype=float)
    else:
        yv = y_vol.reindex(idx).to_numpy(dtype=float)
    blocks = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in blocks]
    if not blocks:
        raise ValueError("blocks 가 비어 있음")
    union = np.zeros(len(idx), dtype=bool)
    rows = []

    def _row(label, a, b, mask):
        m = mask & np.isfinite(fc) & np.isfinite(rv)
        n = int(m.sum())
        rec = {"block": label, "start": a, "end": b, "n": n, "n_blocks": n // P2["h"],
               "n_pos_vol": int(np.nansum(yv[m] == 1)) if n else 0}
        for name, arr in (("model", fc), ("vix", ln_vix), ("rv22", ln_rv22)):
            sc = _log_scores(arr[m], rv[m]) if n else {"mse_log": np.nan, "qlike": np.nan, "r2_log": np.nan}
            suffix = "" if name == "model" else f"_{name}"
            rec[f"mse_log{suffix}"] = sc["mse_log"]
            rec[f"qlike{suffix}"] = sc["qlike"]
            rec[f"r2_log{suffix}"] = sc["r2_log"]
            rec[f"auc_vol{suffix}"] = auc(arr[m], yv[m]) if n else np.nan
        return rec

    for a, b in blocks:
        mask = (idx >= a) & (idx < b)
        union |= mask
        rows.append(_row(f"{a:%Y-%m-%d}~{b:%Y-%m-%d}", a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d"), mask))
    rows.append(_row("all", blocks[0][0].strftime("%Y-%m-%d"), blocks[-1][1].strftime("%Y-%m-%d"), union))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 자기점검
# ------------------------------------------------------------------
def _summ(s: pd.Series) -> dict:
    s = s.dropna()
    if s.empty:
        return {"n": 0, "mean": np.nan, "min": np.nan, "max": np.nan, "last": np.nan}
    return {"n": int(len(s)), "mean": float(s.mean()), "min": float(s.min()), "max": float(s.max()), "last": float(s.iloc[-1])}


def ratio_checks(ohlc: pd.DataFrame) -> dict:
    """자기점검 (selftest.py · 리포트 ⑨).

    (a) rolling-250 (GK+OV)/CC **변동성** 비율 = √(mean_250 v_gkov) / std_250(logret): 1996년 이후 [0.8, 1.3] 안이어야 한다.
        1993~95 값은 보고만 한다(OHLC 품질이 다르다). 변동성(√분산) 비율을 쓰는 이유: 분산 비율은 같은 자료에서 0.75 까지
        내려가 범위 [0.8,1.3] 이 성립하지 않는다 — 실측(2026-09) 변동성 비율 1996+ [0.867, 1.165].
    (b) Parkinson RV20 / CC RV20 = √(mean_20 v_pk) / std_20(logret) ∈ [0.3, 3] (전 구간).
    (c) 1993~95 시가==고가|저가 비중(실측 29.9%)·중앙 로그 범위(0.62% vs 1996+ 1.09%) 보고.
    반환 dict: gkov_cc_ratio_250, pk_cc_rv20, ohlc_quality_1993_95, ok(bool), warnings(list[str]).
    """
    px = _check_ohlc(ohlc)
    warn: list[str] = []
    v = garman_klass_variance(px, overnight=True)
    pk = parkinson_variance(px)
    logret = np.log(px["Close"]).diff()

    # (a)
    ratio250 = np.sqrt(v.rolling(RATIO_250_WINDOW, min_periods=RATIO_250_WINDOW).mean()) \
        / logret.rolling(RATIO_250_WINDOW, min_periods=RATIO_250_WINDOW).std()
    ratio250.name = "gkov_cc_ratio_250"
    late = ratio250.loc[RATIO_CHECK_START:]
    early = ratio250.loc[:EARLY_OHLC_END]
    lo, hi = RATIO_250_BOUNDS
    a_ok = bool(late.dropna().empty or ((late.dropna() >= lo) & (late.dropna() <= hi)).all())
    if not a_ok:
        out_of = late[(late < lo) | (late > hi)]
        warn.append(f"(GK+OV)/CC rolling-250 비율이 {RATIO_CHECK_START} 이후 [{lo}, {hi}] 를 벗어남: "
                    f"{int(out_of.notna().sum())}행 (min {late.min():.3f}, max {late.max():.3f}, "
                    f"예: {out_of.index[:3].strftime('%Y-%m-%d').tolist()})")
    if late.dropna().empty:
        warn.append(f"(GK+OV)/CC 비율 검사 구간({RATIO_CHECK_START}~)에 자료가 없음")

    # (b)
    pk_cc = np.sqrt(pk.rolling(PK_CC_WINDOW, min_periods=PK_CC_WINDOW).mean()) \
        / logret.rolling(PK_CC_WINDOW, min_periods=PK_CC_WINDOW).std()
    pk_cc.name = "pk_cc_rv20"
    plo, phi = PK_CC_BOUNDS
    pkd = pk_cc.dropna()
    b_ok = bool(pkd.empty or ((pkd >= plo) & (pkd <= phi)).all())
    if not b_ok:
        out_of = pkd[(pkd < plo) | (pkd > phi)]
        warn.append(f"Parkinson/CC RV20 비율이 [{plo}, {phi}] 를 벗어남: {len(out_of)}행 "
                    f"(min {pkd.min():.3f}, max {pkd.max():.3f}, 예: {out_of.index[:3].strftime('%Y-%m-%d').tolist()})")

    # (c)
    e = px.loc[:EARLY_OHLC_END]
    l = px.loc[RATIO_CHECK_START:]

    def _quality(df: pd.DataFrame) -> dict:
        if df.empty:
            return {"n": 0, "share_open_eq_high_or_low": np.nan, "median_log_range_pct": np.nan}
        eq = (df["Open"] == df["High"]) | (df["Open"] == df["Low"])
        return {"n": int(len(df)), "share_open_eq_high_or_low": float(eq.mean()),
                "median_log_range_pct": float(np.median(np.log(df["High"] / df["Low"])) * 100.0)}

    q_early, q_late = _quality(e), _quality(l)
    return {
        "gkov_cc_ratio_250": {"bounds": list(RATIO_250_BOUNDS), "check_start": RATIO_CHECK_START, "ok": a_ok,
                              "since_1996": _summ(late), "early_1993_95": _summ(early), "definition": "sqrt(mean250 v_gkov)/std250(logret)"},
        "pk_cc_rv20": {"bounds": list(PK_CC_BOUNDS), "ok": b_ok, "all": _summ(pk_cc),
                       "definition": "sqrt(mean20 v_pk)/std20(logret)"},
        "ohlc_quality_1993_95": {"early_1993_95": q_early, "since_1996": q_late},
        "ok": bool(a_ok and b_ok),
        "warnings": warn,
    }
