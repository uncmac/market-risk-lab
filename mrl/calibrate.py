# -*- coding: utf-8 -*-
"""Phase 2 보정·평가 — 재적합 일정 · 퍼지 · 기후학 · walk-forward 사다리 · 블록 · 검정 · 신뢰도 · 수용 판정
(ARCHITECTURE_PHASE2.md §8, VALIDATION.md §6).

계약:
    refit_dates(idx, first="2003-01-02", end=HOLDOUT_START) -> list[Timestamp]      # 각 해 1월 첫 거래일
    first_refit_ok(idx, y, episodes20, refit_date, rule) -> (bool, dict)            # ≥2,000행 AND 완결 ≥20% 에피소드
    training_mask(idx, refit_date, purge=20, train_start="1993-10-14") -> ndarray[bool]   # pos ≤ pos(R)−21
    pit_climatology(y, mask) -> float                                                # 퍼지된 학습창 라벨 평균
    walk_forward(feats, y, ladder, first_refit, end, C, purge, ..., holdout_final=False) -> (oos, params)
    block_scores(oos, blocks, model_col="p_m3") -> DataFrame
    loss_diff_ci(p_a, p_b, y, block=40, n_boot=4000, seed=0) -> dict
    dm_test(loss_a, loss_b, lag=19) -> dict
    phase_offset_skill(p, y, ref, h=20) -> DataFrame
    reliability_table(p, y, bins, n_eff_div=20) -> DataFrame
    murphy_decomposition(p, y, bins) -> dict
    era_auc(feats, oos, eras) -> DataFrame
    ladder_table(oos, blocks) -> DataFrame
    acceptance(block_scores24, ladder, pooled, rule="literal") -> dict
    v0_reference(replay_completed, y, refit_dates) -> Series
    determinism_check(feats, y, ...) -> dict

설계 원칙
* 홀드아웃 보호: walk_forward 는 holdout_final=False 이면 feats·y 를 end(=HOLDOUT_START) 에서 **먼저 자른다**
  (라벨 마스크가 아니라 입력 프레임). 스크립트의 하드컷과 이중 보호.
* 퍼지: 재적합 R 의 학습 행은 pos(t) ≤ pos(R) − purge − 1 (라벨 창 t+1..t+20 이 R 전에 완전히 실현).
* 기후학은 재적합 연도의 퍼지된 학습창 라벨 평균(연중 상수) — 결정층 r = p/clim 이 결정적이도록.
* 파라미터 예산: 매 재적합에서 M3 의 n_params == PARAM_COUNT(4) 를 assert, 모든 단 ≤ BUDGET.
* 결정론: 난수는 전부 seed=0. 같은 입력의 walk_forward 두 번은 CSV 해시가 같다(determinism_check).
* 조용한 실패 금지: 빈 블록·단일 클래스 AUC·비양 HAC 분산은 NaN + warnings.warn (지어내지 않는다).
* 겹치는 라벨: 모든 표에 n_blocks(= n//20) 를 병기한다.
"""
from __future__ import annotations

import hashlib
import math
import warnings
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from mrl.config import BLOCKS_18, BLOCKS_24, HOLDOUT_START, P2
from mrl.evaluate import block_bootstrap_ci, brier
from mrl.model import BUDGET, LADDER, PARAM_COUNT, LogitModel, fit_logit
from mrl.model import spec_sha256 as _spec_sha256

__all__ = [
    "refit_dates", "first_refit_ok", "training_mask", "pit_climatology", "fit_refit", "walk_forward",
    "rung_col", "assign_blocks", "block_sessions", "check_blocks", "block_scores", "rung_table",
    "loss_diff_ci", "dm_test", "phase_offset_skill", "phase_summary", "reliability_table", "wilson",
    "murphy_decomposition", "era_auc", "ladder_table", "acceptance", "v0_reference",
    "oos_sha256", "determinism_check", "STEPS", "STEP_LABELS", "OOS_BASE_COLUMNS",
]

Y_COL = "y_dd5_20"
OOS_BASE_COLUMNS = ("y", "clim", "p_vix", "p_vix_driftless", "p_vix_bgk")
OOS_TAIL_COLUMNS = ("refit_year", "block24", "block18")
BENCH_COLUMNS = ("p_vix", "p_vix_driftless", "p_vix_bgk")
V0_REF_FIRST = "2017-01-03"          # v0 참조선 첫 재적합
V0_REF_MIN_YEARS = 2                 # 학습 ≥ 2 역년
V0_REF_MIN_POS = 40                  # 학습 양성 ≥ 40
Z95 = 1.959963984540054
AMENDED_BLOCK_FRAC = 8.0 / 11.0      # 부호검정 "≥ 8/11 블록"
AMENDED_MIN_BLOCK_BSS = -0.05        # 완화안 A: 모든 블록 bss_clim ≥ −0.05

# 사다리 단·벤치마크 → 열
_FIXED_COLS = {"M0": "p_vix", "B1": "p_vix", "BGK": "p_vix_bgk", "DRIFTLESS": "p_vix_driftless", "clim": "clim", "CLIM": "clim"}
# 사다리 표의 단 (from, to). 앞 7개가 계약의 단, 뒤 4개는 완화안(A/B)이 M1·M2 에도 필요해 추가.
STEPS = (("M0", "M1"), ("M1", "M2"), ("M2", "M3"), ("M1", "M3"), ("clim", "M3"), ("B1", "M3"), ("BGK", "M3"),
         ("clim", "M1"), ("clim", "M2"), ("B1", "M1"), ("B1", "M2"))
STEP_LABELS = tuple(f"{a}->{b}" for a, b in STEPS)


# ------------------------------------------------------------------
# 유틸
# ------------------------------------------------------------------
def _idx(obj) -> pd.DatetimeIndex:
    idx = obj if isinstance(obj, pd.Index) else getattr(obj, "index", None)
    if not isinstance(idx, pd.DatetimeIndex):
        try:
            idx = pd.DatetimeIndex(pd.to_datetime(idx))
        except Exception as e:                      # noqa: BLE001
            raise TypeError(f"DatetimeIndex 가 필요합니다: {e}") from e
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    if idx.has_duplicates:
        raise ValueError("인덱스에 중복 날짜가 있습니다")
    if not idx.is_monotonic_increasing:
        raise ValueError("인덱스가 오름차순이 아닙니다")
    return idx


def _y_series(y, idx: pd.DatetimeIndex | None = None) -> pd.Series:
    """y 가 targets DataFrame 이면 y_dd5_20 열, Series 면 그대로. float(0/1/NaN)로 정규화, idx 로 reindex."""
    if isinstance(y, pd.DataFrame):
        if Y_COL not in y.columns:
            raise ValueError(f"targets 에 {Y_COL} 열이 없습니다")
        y = y[Y_COL]
    if not isinstance(y, pd.Series):
        raise TypeError("y 는 Series(y_dd5_20) 또는 targets DataFrame 이어야 합니다")
    s = y.astype(float)
    vals = s.dropna().unique()
    if not np.isin(vals, [0.0, 1.0]).all():
        raise ValueError("y 는 {0,1,NaN} 이어야 합니다")
    if idx is not None:
        s = s.reindex(idx)
    s.name = "y"
    return s


def _dstr(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _align(*series) -> tuple[np.ndarray, ...]:
    """공통 인덱스 · 전부 유한한 행만 (시간순). Series/스칼라 혼용 가능(스칼라는 브로드캐스트)."""
    base = None
    for s in series:
        if isinstance(s, pd.Series):
            base = s.index if base is None else base.intersection(s.index)
    if base is None:
        raise TypeError("_align: 최소 하나는 Series 여야 합니다")
    base = base.sort_values()
    arrs = []
    for s in series:
        if isinstance(s, pd.Series):
            arrs.append(s.reindex(base).to_numpy(dtype=float))
        else:
            arrs.append(np.full(len(base), float(s)))
    ok = np.ones(len(base), dtype=bool)
    for a in arrs:
        ok &= np.isfinite(a)
    return tuple(a[ok] for a in arrs) + (base[ok],)


def rung_col(rung: str) -> str:
    """단 이름 → oos 열. M0/B1 → p_vix, BGK → p_vix_bgk, clim → clim, 그 밖은 p_<rung> (M3-PK → p_m3_pk)."""
    if rung in _FIXED_COLS:
        return _FIXED_COLS[rung]
    return "p_" + str(rung).lower().replace("-", "_")


def _auc(y: np.ndarray, s: np.ndarray, what: str = "") -> float:
    if len(y) == 0 or y.min() == y.max():
        warnings.warn(f"AUC 계산 불가({what}): 클래스가 하나뿐이거나 표본이 없습니다 (n={len(y)})", RuntimeWarning)
        return float("nan")
    return float(roc_auc_score(y, s))


# ------------------------------------------------------------------
# 8.1 walk-forward 일정
# ------------------------------------------------------------------
def refit_dates(idx, first=P2["first_refit"], end=HOLDOUT_START) -> list[pd.Timestamp]:
    """각 해 1월 첫 거래일(idx 기준) 중 [first, end) 안의 것. first 는 idx 에 있는 1월 첫 거래일이어야 한다.
    end=None 이면 idx 끝까지(홀드아웃 최종 검증: R_2024·R_2025·R_2026)."""
    idx = _idx(idx)
    first_ts = pd.Timestamp(first)
    if first_ts not in idx:
        raise ValueError(f"first={_dstr(first_ts)} 가 인덱스에 없습니다")
    end_ts = pd.Timestamp(end) if end is not None else idx[-1] + pd.Timedelta(days=1)
    jan = idx[idx.month == 1]
    firsts = jan.to_series().groupby(jan.year).first().tolist()
    out = [pd.Timestamp(d) for d in firsts if first_ts <= d < end_ts]
    if not out or out[0] != first_ts:
        raise ValueError(f"first={_dstr(first_ts)} 는 그 해 1월 첫 거래일이 아닙니다")
    return out


def training_mask(idx, refit_date, purge: int = P2["purge"], train_start=P2["train_start"]) -> np.ndarray:
    """학습 행 마스크: date ≥ train_start AND pos(t) ≤ pos(R) − purge − 1 (라벨 창 t+1..t+purge 가 R 전에 완전히 실현).
    라벨 NaN 여부는 포함하지 않는다(호출자가 y.notna() 와 결합)."""
    idx = _idx(idx)
    R = pd.Timestamp(refit_date)
    if R not in idx:
        raise ValueError(f"refit_date={_dstr(R)} 가 인덱스에 없습니다")
    if purge < 0:
        raise ValueError("purge 는 0 이상이어야 합니다")
    pos_R = int(idx.get_loc(R))
    pos = np.arange(len(idx))
    return np.asarray((idx >= pd.Timestamp(train_start)) & (pos <= pos_R - int(purge) - 1))


def pit_climatology(y, mask) -> float:
    """퍼지된 학습창의 y 평균(연중 상수). 유효 행이 없으면 ValueError."""
    yv = _y_series(y).to_numpy(dtype=float)
    m = np.asarray(mask, dtype=bool)
    if m.shape != yv.shape:
        raise ValueError(f"mask 길이 {m.shape} ≠ y 길이 {yv.shape}")
    vals = yv[m]
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        raise ValueError("기후학을 계산할 학습 라벨이 없습니다")
    return float(vals.mean())


def first_refit_ok(idx, y, episodes20: pd.DataFrame, refit_date, rule: dict = P2["first_refit_rule"],
                   purge: int = P2["purge"], train_start=P2["train_start"]) -> tuple[bool, dict]:
    """첫 재적합 규칙(사전 등록): 학습 행(퍼지·라벨 유효) ≥ min_rows AND 학습창 안에 저점이 실현된 ≥ min_episode_depth
    에피소드 ≥ 1 (episodes20 = targets.episodes(close, 0.20); trough_date ≤ 학습 마지막 행)."""
    idx = _idx(idx)
    ys = _y_series(y, idx)
    mask = training_mask(idx, refit_date, purge, train_start) & ys.notna().to_numpy()
    n_rows = int(mask.sum())
    train_end = idx[mask][-1] if n_rows else None
    depth = float(rule["min_episode_depth"])
    if episodes20 is None or len(episodes20) == 0 or train_end is None:
        deep = pd.DataFrame(columns=["peak_date", "trough_date", "depth"])
    else:
        ep = episodes20
        deep = ep[(ep["depth"].astype(float) <= -depth) & (pd.to_datetime(ep["trough_date"]) <= train_end)]
    n_ep = int(len(deep))
    ok = bool(n_rows >= int(rule["min_rows"]) and n_ep >= 1)
    info = {"refit_date": _dstr(refit_date), "n_rows": n_rows, "min_rows": int(rule["min_rows"]),
            "n_episodes": n_ep, "min_episode_depth": depth, "train_end": _dstr(train_end) if train_end is not None else None,
            "episodes": [{"peak_date": _dstr(r.peak_date), "trough_date": _dstr(r.trough_date), "depth": float(r.depth)}
                         for r in deep.itertuples()],
            "rows_ok": bool(n_rows >= int(rule["min_rows"])), "episode_ok": bool(n_ep >= 1)}
    return ok, info


# ------------------------------------------------------------------
# 적합 (한 재적합일)
# ------------------------------------------------------------------
def fit_refit(feats: pd.DataFrame, y, refit_date, features=LADDER["M3"], *, C: float = P2["C"], purge: int = P2["purge"],
              train_start=P2["train_start"], rung: str | None = None, clim: float | None = None,
              spec_sha256: str | None = None, feature_rule: str | None = None) -> LogitModel:
    """재적합일 R 의 퍼지된 학습창(train_start ≤ t, pos ≤ pos(R)−purge−1, 라벨·특징 유한)으로 한 번 적합.
    라이브 모델(refit_date="2024-08-30")도 이 함수로 만든다. clim 을 주지 않으면 퍼지 학습창(라벨 유효 행 전체) 평균."""
    idx = _idx(feats.index)
    ys = _y_series(y, idx)
    base = training_mask(idx, refit_date, purge, train_start) & ys.notna().to_numpy()
    if clim is None:
        clim = pit_climatology(ys, base)
    fs = tuple(features)
    rows = base & np.isfinite(feats[list(fs)].to_numpy(dtype=float)).all(axis=1)
    meta = {"refit_date": _dstr(refit_date), "clim": float(clim), "rung": rung or None,
            "spec_sha256": spec_sha256 if spec_sha256 is not None else feats.attrs.get("spec_sha256"),
            "feature_rule": feature_rule if feature_rule is not None else feats.attrs.get("feature_rule"),
            "purge": int(purge), "train_start_rule": _dstr(train_start)}
    meta = {k: v for k, v in meta.items() if v is not None}
    return fit_logit(feats.loc[rows, list(fs)], ys[rows], fs, C=C, meta=meta)


# ------------------------------------------------------------------
# walk-forward 사다리
# ------------------------------------------------------------------
def walk_forward(feats: pd.DataFrame, y, ladder: dict = LADDER, first_refit=P2["first_refit"], end=HOLDOUT_START,
                 C: float = P2["C"], purge: int = P2["purge"], train_start=P2["train_start"], *,
                 holdout_final: bool = False, blocks24=BLOCKS_24, blocks18=BLOCKS_18) -> tuple[pd.DataFrame, list[dict]]:
    """확장창 walk-forward. 반환 oos: 인덱스 = [first_refit, end) 의 세션. 열: y, clim, p_vix, p_vix_driftless, p_vix_bgk,
    p_m1, p_m2, p_m3(사다리 단마다 rung_col), refit_year, block24, block18. params: 재적합별
    {refit_date, rung, coef, intercept, n_train, n_pos, clim, train_start, train_end, model_id, n_params}.
    * holdout_final=False: feats·y 를 end 에서 **먼저** 자른다(입력 하드컷). True: 자르지 않고 마지막 세션까지,
      재적합도 매년 계속(R_2024·R_2025·…; 각각 자기 재적합일 전 퍼지 자료로만 학습).
    * M0(특징 없음) 는 적합하지 않고 p_vix 자체(열 p_vix). 매 재적합에서 M3 의 n_params == PARAM_COUNT 를 assert.
    * 특징 NaN 인 OOS 행의 확률은 NaN. .attrs = {warnings, first_refit, end, holdout_final, spec_sha256, feature_rule, n_refits}."""
    if not isinstance(feats, pd.DataFrame):
        raise TypeError("feats 는 build_features 의 DataFrame 이어야 합니다")
    idx = _idx(feats.index)
    feats = feats.copy()
    feats.index = idx
    ys = _y_series(y, idx)
    warn_list: list[str] = []

    if holdout_final:
        end_ts = None
    else:
        if end is None:
            raise ValueError("holdout_final=False 이면 end 가 필요합니다 (기본 HOLDOUT_START)")
        end_ts = pd.Timestamp(end)
        keep = idx < end_ts
        feats, ys, idx = feats[keep], ys[keep], idx[keep]      # 입력 하드컷 (라벨 마스크가 아니라 데이터 자체)
    if len(idx) == 0:
        raise ValueError("하드컷 뒤 남은 행이 없습니다")

    ladder = {str(k): tuple(v) for k, v in ladder.items()}
    need = set(BENCH_COLUMNS) | {f for fs in ladder.values() for f in fs}
    missing = sorted(c for c in need if c not in feats.columns)
    if missing:
        raise ValueError(f"feats 에 열이 없습니다: {missing}")
    for rung, fs in ladder.items():
        if len(fs) + 1 > BUDGET and fs:
            raise ValueError(f"{rung}: 파라미터 {len(fs) + 1} > 예산 {BUDGET}")
        if rung == "M3" and fs != tuple(P2["features"]):
            raise ValueError(f"M3 의 특징은 {tuple(P2['features'])} 이어야 합니다: {fs}")

    rds = refit_dates(idx, first_refit, end=None if holdout_final else end_ts)
    sha = feats.attrs.get("spec_sha256") or _spec_sha256()
    rule = feats.attrs.get("feature_rule") or ""
    for w in feats.attrs.get("warnings", []) or []:
        warn_list.append(f"features: {w}")

    oos_idx = idx[idx >= rds[0]]
    oos = pd.DataFrame(index=oos_idx)
    oos.index.name = "date"
    oos["y"] = ys.reindex(oos_idx).to_numpy()
    oos["clim"] = np.nan
    for c in BENCH_COLUMNS:
        oos[c] = feats[c].reindex(oos_idx).to_numpy(dtype=float)
    fitted_rungs = [r for r, fs in ladder.items() if fs]
    for rung in fitted_rungs:
        oos[rung_col(rung)] = np.nan
    oos["refit_year"] = 0

    params: list[dict] = []
    for i, R in enumerate(rds):
        R_next = rds[i + 1] if i + 1 < len(rds) else None
        sel = (oos_idx >= R) if R_next is None else ((oos_idx >= R) & (oos_idx < R_next))
        score_idx = oos_idx[sel]
        base = training_mask(idx, R, purge, train_start) & ys.notna().to_numpy()
        clim = pit_climatology(ys, base)
        oos.loc[score_idx, "clim"] = clim
        oos.loc[score_idx, "refit_year"] = int(R.year)
        for rung in fitted_rungs:
            fs = ladder[rung]
            m = fit_refit(feats, ys, R, fs, C=C, purge=purge, train_start=train_start, rung=rung, clim=clim,
                          spec_sha256=sha, feature_rule=rule)
            if rung == "M3" or fs == tuple(P2["features"]):
                assert m.n_params() == PARAM_COUNT, f"{rung}@{_dstr(R)}: n_params {m.n_params()} ≠ PARAM_COUNT {PARAM_COUNT}"
            assert m.n_params() <= BUDGET, f"{rung}@{_dstr(R)}: n_params {m.n_params()} > BUDGET {BUDGET}"
            p = m.predict_proba(feats.loc[score_idx, list(fs)])
            oos.loc[score_idx, rung_col(rung)] = p.to_numpy()
            n_nan = int(p.isna().sum())
            if n_nan:
                warn_list.append(f"{rung}@{_dstr(R)}: OOS {n_nan}행의 입력 결측 → 확률 NaN")
            params.append({"refit_date": _dstr(R), "rung": rung, "features": list(fs), "coef": dict(m.coef),
                           "intercept": m.intercept, "n_train": m.n_train, "n_pos": m.n_pos, "clim": clim,
                           "train_start": m.train_start, "train_end": m.train_end, "model_id": m.model_id,
                           "n_params": m.n_params(), "C": float(C)})

    oos["block24"] = assign_blocks(oos_idx, blocks24)
    oos["block18"] = assign_blocks(oos_idx, blocks18)
    oos.attrs = {"warnings": warn_list, "first_refit": _dstr(rds[0]), "end": _dstr(end_ts) if end_ts is not None else None,
                 "holdout_final": bool(holdout_final), "spec_sha256": sha, "feature_rule": rule, "n_refits": len(rds),
                 "purge": int(purge), "C": float(C), "train_start": _dstr(train_start), "ladder": {k: list(v) for k, v in ladder.items()}}
    return oos, params


# ------------------------------------------------------------------
# 8.2 블록
# ------------------------------------------------------------------
def _months_between(a, b) -> int:
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    return (b.year - a.year) * 12 + (b.month - a.month) - (1 if b.day < a.day else 0)


def assign_blocks(idx, blocks) -> np.ndarray:
    """각 세션이 속한 블록의 1-기반 순번(반개구간 [a,b)), 어디에도 안 들면 0."""
    idx = _idx(idx)
    out = np.zeros(len(idx), dtype=int)
    for k, (a, b) in enumerate(blocks, start=1):
        m = (idx >= pd.Timestamp(a)) & (idx < pd.Timestamp(b))
        if (out[m] != 0).any():
            raise ValueError(f"블록이 겹칩니다: #{k} {a}~{b}")
        out[m] = k
    return out


def block_sessions(idx, blocks) -> pd.DataFrame:
    """블록 표: block, start, end, months, first_session, last_session, n."""
    idx = _idx(idx)
    rows = []
    for k, (a, b) in enumerate(blocks, start=1):
        sub = idx[(idx >= pd.Timestamp(a)) & (idx < pd.Timestamp(b))]
        rows.append({"block": k, "start": _dstr(a), "end": _dstr(b), "months": _months_between(a, b),
                     "first_session": _dstr(sub[0]) if len(sub) else None,
                     "last_session": _dstr(sub[-1]) if len(sub) else None, "n": int(len(sub))})
    return pd.DataFrame(rows)


def check_blocks(blocks, start="2003-01-01", end=HOLDOUT_START, min_months: int = 18, max_months: int = 24) -> dict:
    """블록 표가 [start, end) 를 빈틈·겹침 없이 타일링하고 길이 ∈ [min, max] 개월인지. 위반이면 ValueError."""
    if not blocks:
        raise ValueError("블록이 비어 있습니다")
    prev_end = pd.Timestamp(start)
    months = []
    for a, b in blocks:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if a != prev_end:
            raise ValueError(f"블록 경계가 이어지지 않습니다: {_dstr(prev_end)} → {_dstr(a)}")
        if b <= a:
            raise ValueError(f"블록 길이가 0 이하: {_dstr(a)}~{_dstr(b)}")
        mth = _months_between(a, b)
        if not (min_months <= mth <= max_months):
            raise ValueError(f"블록 {_dstr(a)}~{_dstr(b)} 길이 {mth}개월 ∉ [{min_months},{max_months}]")
        months.append(mth)
        prev_end = b
    if prev_end != pd.Timestamp(end):
        raise ValueError(f"마지막 블록 끝 {_dstr(prev_end)} ≠ end {_dstr(end)}")
    return {"n_blocks": len(blocks), "months": months, "start": _dstr(start), "end": _dstr(end)}


# ------------------------------------------------------------------
# 채점
# ------------------------------------------------------------------
def _score_subset(sub: pd.DataFrame, model_col: str) -> dict:
    """한 구간의 채점. 라벨·모델·벤치마크가 전부 유한한 행만(같은 행에서 비교)."""
    cols = ["y", "clim", "p_vix", "p_vix_bgk", model_col] + (["p_m1"] if "p_m1" in sub.columns else [])
    cols = list(dict.fromkeys(cols))
    d = sub[cols].astype(float)
    ok = np.isfinite(d.to_numpy()).all(axis=1)
    n_lab = int(np.isfinite(d[["y", model_col]].to_numpy()).all(axis=1).sum())
    d = d[ok]
    n = int(len(d))
    out = {"n": n, "n_blocks": n // 20, "n_pos": int(d["y"].sum()) if n else 0}
    if n == 0:
        out.update({k: float("nan") for k in ("base", "mean_p", "brier", "brier_clim", "brier_vix", "brier_vix_bgk", "brier_m1",
                                             "bss_clim", "bss_vix", "bss_vix_bgk", "bss_m1", "auc", "calib_in_large")})
        return out
    if n_lab != n:
        warnings.warn(f"{model_col}: 벤치마크 결측으로 {n_lab - n}행 제외", RuntimeWarning)
    yv = d["y"].to_numpy()
    p = d[model_col].to_numpy()
    bs = brier(d[model_col], d["y"])
    bs_clim = brier(d["clim"], d["y"])
    bs_vix = brier(d["p_vix"], d["y"])
    bs_bgk = brier(d["p_vix_bgk"], d["y"])
    bs_m1 = brier(d["p_m1"], d["y"]) if "p_m1" in d.columns else float("nan")

    def _skill(ref):
        return float(1.0 - bs / ref) if (ref is not None and math.isfinite(ref) and ref > 0) else float("nan")

    out.update({"base": float(yv.mean()), "mean_p": float(p.mean()), "brier": bs, "brier_clim": bs_clim,
                "brier_vix": bs_vix, "brier_vix_bgk": bs_bgk, "brier_m1": bs_m1,
                "bss_clim": _skill(bs_clim), "bss_vix": _skill(bs_vix), "bss_vix_bgk": _skill(bs_bgk), "bss_m1": _skill(bs_m1),
                "auc": _auc(yv, p, model_col), "calib_in_large": float(p.mean() - yv.mean())})
    return out


BLOCK_SCORE_COLUMNS = ("block", "start", "end", "first_session", "last_session", "n", "n_blocks", "n_pos", "base", "mean_p",
                       "brier", "brier_clim", "brier_vix", "brier_vix_bgk", "brier_m1",
                       "bss_clim", "bss_vix", "bss_vix_bgk", "bss_m1", "auc", "calib_in_large")


def block_scores(oos: pd.DataFrame, blocks, model_col: str = "p_m3") -> pd.DataFrame:
    """블록별(+ 마지막 행 "all" = 블록 합집합) 채점: n, n_blocks(=n//20), n_pos, base, mean_p, brier, bss_clim, bss_vix(B1),
    bss_vix_bgk, bss_m1, auc, calib_in_large(mean_p − base). 빈 블록은 n=0·NaN + 경고."""
    if model_col not in oos.columns:
        raise ValueError(f"oos 에 {model_col} 열이 없습니다")
    idx = _idx(oos.index)
    rows = []
    union = np.zeros(len(idx), dtype=bool)
    for k, (a, b) in enumerate(blocks, start=1):
        m = np.asarray((idx >= pd.Timestamp(a)) & (idx < pd.Timestamp(b)))
        union |= m
        sub = oos[m]
        rec = {"block": k, "start": _dstr(a), "end": _dstr(b),
               "first_session": _dstr(sub.index[0]) if len(sub) else None,
               "last_session": _dstr(sub.index[-1]) if len(sub) else None}
        if len(sub) == 0:
            warnings.warn(f"블록 #{k} {a}~{b} 에 OOS 세션이 없습니다", RuntimeWarning)
        rec.update(_score_subset(sub, model_col))
        rows.append(rec)
    sub = oos[union]
    rec = {"block": "all", "start": _dstr(blocks[0][0]), "end": _dstr(blocks[-1][1]),
           "first_session": _dstr(sub.index[0]) if len(sub) else None,
           "last_session": _dstr(sub.index[-1]) if len(sub) else None}
    rec.update(_score_subset(sub, model_col))
    rows.append(rec)
    df = pd.DataFrame(rows, columns=list(BLOCK_SCORE_COLUMNS))
    df.attrs = {"model_col": model_col, "n_blocks_table": len(blocks)}
    return df


def rung_table(oos: pd.DataFrame, rungs: Iterable[str] = ("M0", "M1", "M2", "M3")) -> pd.DataFrame:
    """단별 전체(OOS 전 구간) 채점 표 — 리포트 사다리 표의 머리. 인덱스 = 단, 열 = _score_subset 의 키."""
    rows = {}
    for r in rungs:
        col = rung_col(r)
        if col not in oos.columns:
            warnings.warn(f"rung_table: {r}({col}) 열이 없어 건너뜀", RuntimeWarning)
            continue
        rows[r] = _score_subset(oos, col)
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "rung"
    return df


# ------------------------------------------------------------------
# 검정
# ------------------------------------------------------------------
def loss_diff_ci(p_a, p_b, y, block: int = P2["boot_block"], n_boot: int = P2["n_boot"], seed: int = 0) -> dict:
    """d_t = (p_a − y)² − (p_b − y)² (b 가 좋으면 양수): mean 과 40일 순환 블록 부트스트랩 95% 구간(evaluate.block_bootstrap_ci)."""
    a, b, yy, idx = _align(p_a, p_b, y)
    n = int(len(yy))
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0, "n_blocks": 0,
                "block": int(block), "n_boot": int(n_boot), "seed": int(seed)}
    d = (a - yy) ** 2 - (b - yy) ** 2
    lo, hi = block_bootstrap_ci(pd.Series(d, index=idx), block=block, n_boot=n_boot, ci=0.95, seed=seed)
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": n, "n_blocks": n // 20,
            "block": int(block), "n_boot": int(n_boot), "seed": int(seed)}


def dm_test(loss_a, loss_b, lag: int = P2["hac_lag"]) -> dict:
    """Diebold–Mariano: d = loss_a − loss_b, Newey–West(Bartlett) HAC 분산 lag=19 (20일 라벨 겹침).
    t = d̄ / sqrt(LRV/n), p = 양측 정규. LRV ≤ 0 이거나 n ≤ lag+1 이면 NaN + 경고."""
    if isinstance(loss_a, pd.Series) or isinstance(loss_b, pd.Series):
        a, b, _ = _align(loss_a, loss_b)
    else:
        a = np.asarray(loss_a, dtype=float)
        b = np.asarray(loss_b, dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
    d = a - b
    n = int(len(d))
    out = {"t": float("nan"), "p": float("nan"), "mean": float("nan"), "n": n, "var_hac": float("nan"), "lag": int(lag)}
    if n <= lag + 1:
        warnings.warn(f"dm_test: n={n} ≤ lag+1={lag + 1} — 검정 불가", RuntimeWarning)
        return out
    dbar = float(d.mean())
    e = d - dbar
    lrv = float(np.mean(e * e))
    for k in range(1, int(lag) + 1):
        w = 1.0 - k / (lag + 1.0)
        lrv += 2.0 * w * float(np.sum(e[k:] * e[:-k]) / n)
    out["mean"] = dbar
    out["var_hac"] = lrv
    if not (lrv > 0):
        warnings.warn("dm_test: HAC 분산이 양수가 아닙니다 (손실 차가 상수?)", RuntimeWarning)
        return out
    t = dbar / math.sqrt(lrv / n)
    out["t"] = float(t)
    out["p"] = float(math.erfc(abs(t) / math.sqrt(2.0)))
    return out


def phase_offset_skill(p, y, ref, h: int = P2["h"]) -> pd.DataFrame:
    """20개 위상 오프셋 부분표본(매 h 번째 세션, 겹치지 않는 라벨) 각각의 BSS(ref 대비).
    열: offset, n, n_pos, brier, brier_ref, bss. .attrs["summary"] = {min, median, max, share_pos, n_offsets}."""
    pv, yv, rv, _ = _align(p, y, ref)
    rows = []
    for k in range(int(h)):
        sl = slice(k, None, int(h))
        pk, yk, rk = pv[sl], yv[sl], rv[sl]
        n = int(len(yk))
        if n == 0:
            rows.append({"offset": k, "n": 0, "n_pos": 0, "brier": float("nan"), "brier_ref": float("nan"), "bss": float("nan")})
            continue
        bs = float(np.mean((pk - yk) ** 2))
        bref = float(np.mean((rk - yk) ** 2))
        rows.append({"offset": k, "n": n, "n_pos": int(yk.sum()), "brier": bs, "brier_ref": bref,
                     "bss": float(1.0 - bs / bref) if bref > 0 else float("nan")})
    df = pd.DataFrame(rows)
    df.attrs = {"summary": phase_summary(df), "h": int(h)}
    return df


def phase_summary(df: pd.DataFrame) -> dict:
    b = df["bss"].dropna()
    if len(b) == 0:
        return {"min": float("nan"), "median": float("nan"), "max": float("nan"), "share_pos": float("nan"), "n_offsets": 0}
    return {"min": float(b.min()), "median": float(b.median()), "max": float(b.max()),
            "share_pos": float((b > 0).mean()), "n_offsets": int(len(b))}


# ------------------------------------------------------------------
# 신뢰도
# ------------------------------------------------------------------
def wilson(p_hat: float, n_eff: float, z: float = Z95) -> tuple[float, float]:
    """Wilson 구간 (n_eff 는 실수 허용 = n/20). n_eff ≤ 0 이면 (nan, nan)."""
    if not (n_eff > 0) or not math.isfinite(p_hat):
        return (float("nan"), float("nan"))
    denom = 1.0 + z * z / n_eff
    centre = (p_hat + z * z / (2.0 * n_eff)) / denom
    half = z * math.sqrt(p_hat * (1.0 - p_hat) / n_eff + z * z / (4.0 * n_eff * n_eff)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _bin_ids(p: np.ndarray, bins) -> np.ndarray:
    edges = np.asarray(bins, dtype=float)
    if len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("bins 는 오름차순 경계 ≥ 2개여야 합니다")
    ids = np.searchsorted(edges, p, side="right") - 1          # [lo, hi)
    ids = np.where(p >= edges[-1], len(edges) - 2, ids)         # 마지막 구간은 닫힘 [lo, hi]
    if (ids < 0).any() or (ids > len(edges) - 2).any():
        raise ValueError("확률이 bins 범위 밖에 있습니다")
    return ids


def reliability_table(p, y, bins=P2["reliability_bins"], n_eff_div: int = P2["n_eff_div"]) -> pd.DataFrame:
    """고정 구간 신뢰도 표: bin, lo, hi, n, n_eff(=n/n_eff_div), mean_p, obs, wilson_lo/hi(n_eff 기준). 빈 구간은 n=0·NaN."""
    pv, yv, _ = _align(p, y)
    ids = _bin_ids(pv, bins)
    edges = list(bins)
    rows = []
    for k in range(len(edges) - 1):
        m = ids == k
        n = int(m.sum())
        n_eff = n / float(n_eff_div)
        if n == 0:
            rows.append({"bin": f"[{edges[k]:.2f}, {edges[k + 1]:.2f})", "lo": edges[k], "hi": edges[k + 1], "n": 0, "n_eff": 0.0,
                         "mean_p": float("nan"), "obs": float("nan"), "wilson_lo": float("nan"), "wilson_hi": float("nan")})
            continue
        obs = float(yv[m].mean())
        wl, wh = wilson(obs, n_eff)
        rows.append({"bin": f"[{edges[k]:.2f}, {edges[k + 1]:.2f})", "lo": edges[k], "hi": edges[k + 1], "n": n, "n_eff": n_eff,
                     "mean_p": float(pv[m].mean()), "obs": obs, "wilson_lo": wl, "wilson_hi": wh})
    df = pd.DataFrame(rows)
    df.attrs = {"n": int(len(pv)), "n_eff_div": int(n_eff_div), "base": float(yv.mean()) if len(yv) else float("nan")}
    return df


def murphy_decomposition(p, y, bins=P2["reliability_bins"]) -> dict:
    """Murphy 분해(고정 구간, 구간 평균 확률 p̄_k 기준): reliability = Σ n_k (p̄_k − ō_k)²/n, resolution = Σ n_k (ō_k − ō)²/n,
    uncertainty = ō(1−ō). brier_binned = reliability − resolution + uncertainty (구간 평균 예보의 Brier, 정확).
    실제 brier = brier_binned + residual (residual = 구간 안 확률 산포 − 2·구간 안 공분산; 숨기지 않고 명시)."""
    pv, yv, _ = _align(p, y)
    n = len(pv)
    nan = float("nan")
    if n == 0:
        return {"reliability": nan, "resolution": nan, "uncertainty": nan, "brier": nan, "brier_binned": nan, "residual": nan,
                "n": 0, "n_blocks": 0}
    ids = _bin_ids(pv, bins)
    obar = float(yv.mean())
    rel = res = 0.0
    p_binned = np.empty(n)
    for k in np.unique(ids):
        m = ids == k
        nk = int(m.sum())
        pk, ok = float(pv[m].mean()), float(yv[m].mean())
        rel += nk * (pk - ok) ** 2
        res += nk * (ok - obar) ** 2
        p_binned[m] = pk
    bs = float(np.mean((pv - yv) ** 2))
    bs_binned = float(np.mean((p_binned - yv) ** 2))
    return {"reliability": rel / n, "resolution": res / n, "uncertainty": obar * (1.0 - obar), "brier": bs,
            "brier_binned": bs_binned, "residual": bs - bs_binned, "n": int(n), "n_blocks": int(n) // 20}


# ------------------------------------------------------------------
# 시대별 AUC
# ------------------------------------------------------------------
def era_auc(feats: pd.DataFrame, oos: pd.DataFrame, eras=P2["eras"], y=None) -> pd.DataFrame:
    """시대별 AUC: x_vix, x_har, −x_ma(원시 특징; y 가 주어지면 전 이력, 아니면 oos.y), p_m1, p_m3(OOS 행만).
    열: era, start, end, n_raw, n_pos_raw, auc_x_vix, auc_x_har, auc_neg_x_ma, n_oos, n_pos_oos, auc_p_m1, auc_p_m3."""
    fi = _idx(feats.index)
    y_raw = _y_series(y, fi) if y is not None else _y_series(oos["y"], fi)
    rows = []
    for a, b in eras:
        a_ts, b_ts = pd.Timestamp(a), pd.Timestamp(b)
        m_raw = (fi >= a_ts) & (fi <= b_ts)
        sub_f = feats[m_raw]
        y_sub = y_raw[m_raw]
        rec = {"era": f"{_dstr(a)}..{_dstr(b)}", "start": _dstr(a), "end": _dstr(b)}
        for name, col, sign in (("auc_x_vix", "x_vix", 1.0), ("auc_x_har", "x_har", 1.0), ("auc_neg_x_ma", "x_ma", -1.0)):
            if col not in sub_f.columns:
                rec[name] = float("nan")
                continue
            s = sub_f[col].astype(float) * sign
            sv, yv, _ = _align(s, y_sub)
            rec[name] = _auc(yv, sv, f"{col} {rec['era']}") if len(yv) else float("nan")
            if col == "x_vix":
                rec["n_raw"], rec["n_pos_raw"] = int(len(yv)), int(yv.sum()) if len(yv) else 0
        oi = _idx(oos.index)
        m_oos = (oi >= a_ts) & (oi <= b_ts)
        sub_o = oos[m_oos]
        for name, col in (("auc_p_m1", "p_m1"), ("auc_p_m3", "p_m3")):
            if col not in sub_o.columns or len(sub_o) == 0:
                rec[name] = float("nan")
                if col == "p_m1":
                    rec["n_oos"], rec["n_pos_oos"] = 0, 0
                continue
            sv, yv, _ = _align(sub_o[col].astype(float), sub_o["y"].astype(float))
            rec[name] = _auc(yv, sv, f"{col} {rec['era']}") if len(yv) else float("nan")
            if col == "p_m1":
                rec["n_oos"], rec["n_pos_oos"] = int(len(yv)), int(yv.sum()) if len(yv) else 0
        rows.append(rec)
    cols = ["era", "start", "end", "n_raw", "n_pos_raw", "auc_x_vix", "auc_x_har", "auc_neg_x_ma",
            "n_oos", "n_pos_oos", "auc_p_m1", "auc_p_m3"]
    return pd.DataFrame(rows).reindex(columns=cols)


# ------------------------------------------------------------------
# 사다리 표
# ------------------------------------------------------------------
LADDER_COLUMNS = ("step", "from", "to", "block", "start", "end", "n", "n_blocks", "brier_from", "brier_to", "bss",
                  "mean", "lo", "hi", "dm_t", "dm_p", "phase_min", "phase_median", "phase_max", "phase_share_pos")


def ladder_table(oos: pd.DataFrame, blocks, steps=STEPS, *, block: int = P2["boot_block"], n_boot: int = P2["n_boot"],
                 seed: int = 0, lag: int = P2["hac_lag"], h: int = P2["h"]) -> pd.DataFrame:
    """단(M0→M1, M1→M2, M2→M3, M1→M3, clim→M3, B1→M3, BGK→M3 (+ clim/B1→M1·M2)) × (전체 "all" + 블록별):
    loss_diff_ci(mean/lo/hi; to 가 좋으면 양수) + dm_test(t/p) + 위상 오프셋 요약(min/median/max/share>0).
    열이 없는 단(예: p_m2 없음)은 경고 후 건너뜀. 단 이름은 ASCII "A->B"."""
    idx = _idx(oos.index)
    union = np.zeros(len(idx), dtype=bool)
    spans = []
    for k, (a, b) in enumerate(blocks, start=1):
        m = np.asarray((idx >= pd.Timestamp(a)) & (idx < pd.Timestamp(b)))
        union |= m
        spans.append((k, _dstr(a), _dstr(b), m))
    spans = [("all", _dstr(blocks[0][0]), _dstr(blocks[-1][1]), union)] + spans
    rows = []
    for frm, to in steps:
        ca, cb = rung_col(frm), rung_col(to)
        if ca not in oos.columns or cb not in oos.columns:
            warnings.warn(f"ladder_table: {frm}->{to} 열({ca},{cb})이 없어 건너뜀", RuntimeWarning)
            continue
        for label, a, b, m in spans:
            sub = oos[m]
            pa, pb, yv, sidx = _align(sub[ca].astype(float), sub[cb].astype(float), sub["y"].astype(float))
            rec = {"step": f"{frm}->{to}", "from": frm, "to": to, "block": label, "start": a, "end": b,
                   "n": int(len(yv)), "n_blocks": int(len(yv)) // 20}
            if len(yv) == 0:
                rec.update({k: float("nan") for k in LADDER_COLUMNS if k not in rec})
                rows.append(rec)
                continue
            la, lb = (pa - yv) ** 2, (pb - yv) ** 2
            bf, bt = float(la.mean()), float(lb.mean())
            rec["brier_from"], rec["brier_to"] = bf, bt
            rec["bss"] = float(1.0 - bt / bf) if bf > 0 else float("nan")
            ci = loss_diff_ci(pd.Series(pa, index=sidx), pd.Series(pb, index=sidx), pd.Series(yv, index=sidx),
                              block=block, n_boot=n_boot, seed=seed)
            rec.update({"mean": ci["mean"], "lo": ci["lo"], "hi": ci["hi"]})
            dm = dm_test(la, lb, lag=lag)
            rec.update({"dm_t": dm["t"], "dm_p": dm["p"]})
            ph = phase_summary(phase_offset_skill(pd.Series(pb, index=sidx), pd.Series(yv, index=sidx),
                                                  pd.Series(pa, index=sidx), h=h))
            rec.update({"phase_min": ph["min"], "phase_median": ph["median"], "phase_max": ph["max"],
                        "phase_share_pos": ph["share_pos"]})
            rows.append(rec)
    df = pd.DataFrame(rows, columns=list(LADDER_COLUMNS))
    df.attrs = {"block": int(block), "n_boot": int(n_boot), "seed": int(seed), "lag": int(lag), "h": int(h),
                "n_blocks_table": len(blocks)}
    return df


# ------------------------------------------------------------------
# 수용 판정 (VALIDATION §6 문자 그대로 + #2a 완화안(사후))
# ------------------------------------------------------------------
def _step_row(ladder: pd.DataFrame, frm: str, to: str, block="all") -> dict | None:
    if ladder is None or len(ladder) == 0:
        return None
    m = (ladder["from"] == frm) & (ladder["to"] == to) & (ladder["block"].astype(str) == str(block))
    if not m.any():
        return None
    return ladder[m].iloc[0].to_dict()


def _literal(bs: pd.DataFrame) -> dict:
    blk = bs[bs["block"].astype(str) != "all"]
    empty = blk[blk["n"] == 0]
    scored = blk[blk["n"] > 0]
    fail_clim = scored[~(scored["bss_clim"] > 0)]
    fail_vix = scored[~(scored["bss_vix"] >= 0)]
    n_blocks = int(len(blk))
    pass_clim = bool(len(scored) == n_blocks and len(fail_clim) == 0)
    pass_vix = bool(len(scored) == n_blocks and len(fail_vix) == 0)
    return {"pass_clim": pass_clim, "pass_vix": pass_vix, "pass": bool(pass_clim and pass_vix), "n_blocks": n_blocks,
            "n_blocks_empty": int(len(empty)),
            "failing_blocks": sorted(set(int(b) for b in fail_clim["block"]) | set(int(b) for b in fail_vix["block"])),
            "failing_clim": [int(b) for b in fail_clim["block"]], "failing_vix": [int(b) for b in fail_vix["block"]],
            "min_bss_clim": float(scored["bss_clim"].min()) if len(scored) else float("nan"),
            "min_bss_vix": float(scored["bss_vix"].min()) if len(scored) else float("nan"),
            "rule_text": "홀드아웃 제외 모든 블록에서 BSS(vs 기저율) > 0 이고 VIX 내재 확률(B1) 대비 BSS ≥ 0"}


def _amended(rung: str, bs: pd.DataFrame, ladder: pd.DataFrame, pooled: dict, min_pass: int) -> dict:
    blk = bs[(bs["block"].astype(str) != "all") & (bs["n"] > 0)]
    n_blocks = int(len(bs[bs["block"].astype(str) != "all"]))
    clim_step = _step_row(ladder, "clim", rung)
    vix_step = _step_row(ladder, "B1", rung)
    prev = {"M1": "M0", "M2": "M1", "M3": "M2"}.get(rung)
    info_step = _step_row(ladder, prev, rung) if prev else None
    pooled_clim = float(pooled.get("bss_clim", float("nan")))
    pooled_vix = float(pooled.get("bss_vix", float("nan")))
    a_lo = float(clim_step["lo"]) if clim_step else float("nan")
    b_lo = float(vix_step["lo"]) if vix_step else float("nan")
    c_lo = float(info_step["lo"]) if info_step else float("nan")
    n_clim_pos = int((blk["bss_clim"] > 0).sum())
    n_vix_nonneg = int((blk["bss_vix"] >= 0).sum())
    A = bool(pooled_clim > 0 and a_lo > 0 and (len(blk) == n_blocks) and bool((blk["bss_clim"] >= AMENDED_MIN_BLOCK_BSS).all())
             and n_clim_pos >= min_pass)
    B = bool(pooled_vix >= 0 and b_lo >= 0 and (len(blk) == n_blocks) and n_vix_nonneg >= min_pass)
    C = bool(c_lo > 0)
    return {"A": A, "B": B, "C": C, "pass": bool(A and B and C), "pooled_bss_clim": pooled_clim, "pooled_bss_vix": pooled_vix,
            "ci_lo_clim": a_lo, "ci_lo_vix": b_lo, "ci_lo_info": c_lo, "info_step": f"{prev}->{rung}" if prev else None,
            "n_blocks": n_blocks, "n_blocks_clim_pos": n_clim_pos, "n_blocks_vix_nonneg": n_vix_nonneg, "min_pass_blocks": int(min_pass),
            "min_block_bss_clim": float(blk["bss_clim"].min()) if len(blk) else float("nan"),
            "rule_text": ("A) 전체 bss_clim>0 ∧ loss_diff_ci(clim→M).lo>0 ∧ 모든 블록 bss_clim≥−0.05 ∧ ≥8/11 블록>0 ; "
                          "B) 전체 bss_vix(B1)≥0 ∧ loss_diff_ci(B1→M).lo≥0 ∧ ≥8/11 블록≥0 ; C) 정보 단(M_{k−1}→M_k).lo>0 — post hoc(#2a)")}


def acceptance(block_scores24, ladder: pd.DataFrame, pooled: dict | None = None, rule: str = "literal",
               candidate: str = "M3") -> dict:
    """수용 판정.
    block_scores24: {rung: block_scores(oos, BLOCKS_24, rung_col(rung))} (DataFrame 하나면 {candidate: df}).
    ladder: ladder_table(oos, BLOCKS_24). pooled: {rung: {bss_clim, bss_vix, ...}} (None 이면 block_scores 의 "all" 행).
    literal(VALIDATION §6 그대로): pass_clim = all(bss_clim > 0) ; pass_vix = all(bss_vix ≥ 0) ; failing_blocks.
    amended(#2a, 사후): A ∧ B ∧ 자기 단의 C 를 만족하는 최상위 단. deploy: rule 에 따라 tone_model / deploy_mode."""
    if rule not in ("literal", "amended"):
        raise ValueError("rule 은 'literal' 또는 'amended'")
    if isinstance(block_scores24, pd.DataFrame):
        block_scores24 = {candidate: block_scores24}
    if not block_scores24 or candidate not in block_scores24:
        raise ValueError(f"block_scores24 에 후보 {candidate} 가 없습니다")
    pooled = dict(pooled or {})
    for r, bs in block_scores24.items():
        if r not in pooled:
            allrow = bs[bs["block"].astype(str) == "all"]
            pooled[r] = allrow.iloc[0].to_dict() if len(allrow) else {}
    n_blocks = int((block_scores24[candidate]["block"].astype(str) != "all").sum())
    min_pass = int(math.ceil(n_blocks * AMENDED_BLOCK_FRAC - 1e-9))

    literal = {r: _literal(bs) for r, bs in block_scores24.items()}
    amended = {r: _amended(r, bs, ladder, pooled[r], min_pass) for r, bs in block_scores24.items()}
    order = [r for r in ("M3", "M2", "M1") if r in block_scores24]

    lit_tone = candidate if literal[candidate]["pass"] else None
    amd_tone = next((r for r in order if amended[r]["pass"]), None)
    tone_model = lit_tone if rule == "literal" else amd_tone
    deploy_mode = "tones" if tone_model else "info_only"
    if rule == "literal":
        rationale = (f"§6 문자 그대로: {candidate} {'통과' if lit_tone else '실패'}"
                     + (f" (실패 블록 {literal[candidate]['failing_blocks']})" if not lit_tone else ""))
    else:
        rationale = (f"#2a 완화안(post hoc): A∧B∧C 를 만족하는 최상위 단 = {amd_tone or '없음'} "
                     + "; ".join(f"{r}:A={amended[r]['A']},B={amended[r]['B']},C={amended[r]['C']}" for r in order))
    return {"rule": rule, "candidate": candidate, "literal": literal, "amended": amended,
            "literal_tone_model": lit_tone, "amended_tone_model": amd_tone,
            "deploy_mode": deploy_mode, "tone_model": tone_model, "rationale": rationale,
            "n_blocks": n_blocks, "min_pass_blocks": min_pass, "post_hoc_note": "amended 는 #2 사전 관측 이후의 규칙이므로 post hoc"}


# ------------------------------------------------------------------
# v0 참조선 (예산 밖, 모델 밖)
# ------------------------------------------------------------------
def v0_reference(replay_completed: pd.DataFrame, y, refit_dates_: Iterable, *, purge: int = P2["purge"],
                 first=V0_REF_FIRST, min_years: int = V0_REF_MIN_YEARS, min_pos: int = V0_REF_MIN_POS) -> pd.Series:
    """v0 종합 s_t = −(score_d+score_w+score_m)/3 에 Platt(2) — 2017-01-03 재적합부터, 학습 ≥2역년·≥40양성 아니면 그 해 NaN + 경고.
    반환 Series "p_v0ref" (replay 인덱스). .attrs["params"] = 재적합별 계수, .attrs["warnings"]."""
    need = ("score_d", "score_w", "score_m")
    missing = [c for c in need if c not in replay_completed.columns]
    if missing:
        raise ValueError(f"replay 에 열이 없습니다: {missing}")
    idx = _idx(replay_completed.index)
    s = -(replay_completed["score_d"].astype(float) + replay_completed["score_w"].astype(float)
          + replay_completed["score_m"].astype(float)) / 3.0
    s.index = idx
    X = pd.DataFrame({"s_v0": s})
    ys = _y_series(y, idx)
    out = pd.Series(np.nan, index=idx, name="p_v0ref")
    params, warn_list = [], []
    rds = [pd.Timestamp(d) for d in refit_dates_]
    rds = [d for d in rds if d >= pd.Timestamp(first)]
    for i, R in enumerate(rds):
        if R not in idx:
            warn_list.append(f"v0 참조선: 재적합일 {_dstr(R)} 이 replay 인덱스에 없어 건너뜀")
            continue
        R_next = rds[i + 1] if i + 1 < len(rds) else None
        mask = training_mask(idx, R, purge, train_start=idx[0]) & ys.notna().to_numpy() & np.isfinite(s.to_numpy())
        n_pos = int(np.nansum(ys.to_numpy()[mask]))
        years = (R - idx[0]).days / 365.25
        if years < min_years or n_pos < min_pos:
            warn_list.append(f"v0 참조선 {_dstr(R)}: 학습 {years:.2f}년·양성 {n_pos} — 조건(≥{min_years}년·≥{min_pos}) 미충족, NaN")
            continue
        m = fit_logit(X[mask], ys[mask], ("s_v0",), meta={"refit_date": _dstr(R), "rung": "R-v0", "feature_rule": "v0-platt",
                                                            "spec_sha256": "v0"})
        sel = (idx >= R) if R_next is None else ((idx >= R) & (idx < R_next))
        out[sel] = m.predict_proba(X[sel]).to_numpy()
        params.append({"refit_date": _dstr(R), "coef": dict(m.coef), "intercept": m.intercept, "n_train": m.n_train,
                       "n_pos": m.n_pos, "n_params": m.n_params()})
    out.attrs = {"params": params, "warnings": warn_list, "first": _dstr(first)}
    return out


# ------------------------------------------------------------------
# 결정론
# ------------------------------------------------------------------
def oos_sha256(oos: pd.DataFrame) -> str:
    """walk_forward 산출 CSV(고정 표현)의 sha256."""
    text = oos.to_csv(index=True, lineterminator="\n", float_format="%.17g")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def determinism_check(feats: pd.DataFrame, y, **kwargs) -> dict:
    """walk_forward 를 두 번 돌려 CSV 해시·최대 계수 차를 비교. 허용오차 1e-9(계수)·해시 동일(CSV)."""
    o1, p1 = walk_forward(feats, y, **kwargs)
    o2, p2 = walk_forward(feats, y, **kwargs)
    h1, h2 = oos_sha256(o1), oos_sha256(o2)
    diffs = [abs(a["intercept"] - b["intercept"]) for a, b in zip(p1, p2)]
    diffs += [abs(a["coef"][k] - b["coef"][k]) for a, b in zip(p1, p2) for k in a["coef"]]
    mx = float(max(diffs)) if diffs else 0.0
    return {"csv_equal": h1 == h2, "sha256_a": h1, "sha256_b": h2, "max_abs_coef_diff": mx, "coef_ok": mx <= 1e-9,
            "n_refits": len(p1) // max(1, len({p["rung"] for p in p1}))}
