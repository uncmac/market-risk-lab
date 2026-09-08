# -*- coding: utf-8 -*-
"""가족 위험예산 → 변동성 목표 × 상태 배수 비중 규칙 (ARCHITECTURE_PHASE3.md §6).

적합 파라미터 **0개**. 이 모듈이 쓰는 숫자는 전부 `mrl.config` 의 사전 등록 상수
(`P3`, `TONE_EXPOSURE`, `STATE_TO_TONE`)이며 생산 확률(p2, 4-파라미터)에 단 하나도 더하지 않는다(§2 예산).
여기서 적합(fit)되는 것은 아무것도 없다 — `PARAM_COUNT == 0`, `n_params() == 0` 이 테스트로 고정된다.

규칙 요약 (`SIZING_RULE`):
    σ̂_t = sqrt(252·v_t), v_t = λ v_{t−1} + (1−λ) r_t², λ=0.94, v seed = 첫 60세션 표본분산
    σ_T  = 격자 내림(D_max / k_slow),  k_slow = 3.5 (헤드라인)  ·  k_fast = 2.0 (낙관선, 병기만)
    w_vol = clip(σ_T / σ̂, 0.25, 1.00)
    m     = TONE_EXPOSURE[STATE_TO_TONE[p2_state]]  = 1.00 / 0.50 / 0.25   (새 숫자 없음)
    w_target = clip(w_vol × m, 0.25, 1.00)                                  ← 곱 뒤 재클립
    실행: cand = round(w_target/0.05)·0.05 ; 격상 즉시(하향만) · 주 마지막 세션 + |Δ| ≥ 0.10 · 그 외 유지
    점 원칙: w_exec_t 는 t 종가에 확정되어 t+1 수익률에 적용. 비용 5bp × |Δw|, 현금 0%.

조용한 실패 금지: σ̂ 또는 상태가 결측이면 비중은 **직전 값 유지**(reason `input_missing`), 절대 1.0 으로
돌아가지 않는다. 알 수 없는 상태·잘못된 예산은 ValueError.

사전 관측 공개(§2·§14): 이 상수들은 1993~2024-08 기록으로 ~40 변형을 본 뒤 골랐으므로 그 기록에 대해
**post hoc** 이다. 지금 동결하고 라이브 장부로 재조정하지 않는다.

계약(§6.6):
    sizing_sha256 · ewma_vol · target_vol · budget_ladder · state_multiplier · exposure_target · step ·
    execute · next_thresholds · run · backtest_table · sensitivities · retention · episode_pnl · rolling_relative
§6.6 의 evaluate 추가분(`allocation_from_weights`, `window_distribution`)은 계약대로 `mrl/evaluate.py` 에
**하나만** 구현되어 있고 여기서는 재수출한다 — 이 모듈의 공개 API 는 그대로이며(`sizing.allocation_from_weights`
는 `evaluate.allocation_from_weights` 와 같은 객체), 백테스트 표·장부의 `rule_ret_20` 이 같은 규약을 쓴다.
"""
from __future__ import annotations

import hashlib
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from mrl import calendar_us
from mrl.config import P3, STATE_TO_TONE, TONE_EXPOSURE
from mrl.evaluate import allocation_from_weights, window_distribution      # noqa: F401 — §6.6 재수출(구현은 evaluate 하나)

__all__ = [
    "SIZING_RULE", "PARAM_COUNT", "TRADING_DAYS", "SIZING_CONSTANTS", "REASONS",
    "SENSITIVITY_NAMES", "MILD_MULTIPLIER", "GRID_SENSITIVITY",
    "ROW_BUY_HOLD", "ROW_P2_DECISION", "ROW_ADOPTED", "ROW_VOL_ONLY", "ROW_VT14",
    "WINDOW_EVAL", "WINDOW_VOL_ONLY", "WINDOW_V0", "BACKTEST_COLUMNS", "BACKTEST_COLUMNS_KO",
    "LADDER_COLUMNS", "RETENTION_CONDITIONS", "NARROW_MARGIN", "HONEST_READING",
    "n_params", "sizing_sha256", "sizing_constants", "ewma_vol", "target_vol", "budget_ladder",
    "state_multiplier", "exposure_target", "step", "execute", "next_thresholds", "run",
    "allocation_from_weights", "window_distribution", "backtest_table", "sensitivities",
    "retention", "episode_pnl", "rolling_relative", "calendar_year_relative",
    "honest_reading", "budget_sentence",
]

TRADING_DAYS = 252
PARAM_COUNT = 0                      # 비중 규칙의 적합 파라미터 수 (§6: 반드시 0)

# 규칙 식별자 — 모든 산출물·장부 행에 기록 (FEATURE_RULE 과 같은 보호)
SIZING_RULE = ("p3|sigma=ewma(lambda=0.94,seed=60)|sT=floor_grid(Dmax/3.5)|w_vol=clip(sT/sigma,.25,1)"
               "|m=TONE_EXPOSURE[STATE_TO_TONE[p2_state]]|w=clip(w_vol*m,.25,1)|grid=.05|band=.10|weekly+escalate|cost=5bp|cash=0")

_SIZING_EPS = 1e-9                   # 격자 내림·밴드 비교의 부동소수 여유 (0.35/3.5 = 0.09999999999999999)
_ROUND_DP = 10                       # 격자 반올림 뒤 부동소수 먼지 제거

REASONS = ("init", "escalation", "weekly", "hold", "input_missing", "info_only")

# 민감도 전용 상수 (생산 규칙 아님 — §6.3 민감도 행에만 쓴다)
MILD_MULTIPLIER = {"normal": 1.0, "caution": 0.75, "reduce": 0.5}      # C 안의 완만한 배수
GRID_SENSITIVITY = {"grid": 0.25, "band": 0.175, "cadence": "daily"}   # A 안의 0.25 격자 + 히스테리시스
SENSITIVITY_NAMES = ("S-nofloor", "S-volonly", "S-monthly", "S-mild", "S-grid", "S-grid-min",
                     "S-VT14", "S-noesc", "S-lag", "S-10bp", "S-HAR", "S-daily",
                     "S-VT7.5", "S-VT12.5", "S-VT15")

# 표 행·창 이름 (retention 이 이 이름으로 행을 찾는다)
ROW_BUY_HOLD = "buy_hold"
ROW_P2_DECISION = "p2_decision"
ROW_ADOPTED = "adopted"
ROW_VOL_ONLY = "S-volonly"
ROW_VT14 = "S-VT14"
WINDOW_EVAL = "2003+"                # 결정층 OOS (2003-01-02 ~ 2024-08-30)
WINDOW_VOL_ONLY = "1993+"            # 2003 이전 변동성 단독 (1993-10-14 ~)
WINDOW_V0 = "2015+"                  # v0 창

# §6.3 고정 형식 (English identifiers; 리포트 한글 라벨은 BACKTEST_COLUMNS_KO)
BACKTEST_COLUMNS = ("row", "window", "start", "end", "years", "cagr", "max_dd", "max_dd_date",
                    "worst_month", "worst_month_label", "ann_vol", "switches_per_year",
                    "avg_exposure", "cost_total", "d_cagr_vs_bh", "d_maxdd_vs_bh",
                    "vol_ratio", "maxdd_over_sigma_t", "n_switches", "cost_bps")
BACKTEST_COLUMNS_KO = {
    "row": "행", "window": "기간", "cagr": "CAGR", "max_dd": "MaxDD", "max_dd_date": "MaxDD 일자",
    "worst_month": "최악 월", "worst_month_label": "최악 월(월)", "ann_vol": "연변동성",
    "switches_per_year": "전환/년", "avg_exposure": "평균 비중", "cost_total": "누적 비용",
    "d_cagr_vs_bh": "ΔCAGR vs 보유", "d_maxdd_vs_bh": "ΔMaxDD vs 보유",
    "vol_ratio": "실현변동성/σ_T", "maxdd_over_sigma_t": "MaxDD/σ_T",
}
LADDER_COLUMNS = ("sigma_target", "d_slow", "d_fast", "maxdd_vol_only", "maxdd_vol_only_date",
                  "maxdd_decision", "maxdd_decision_date", "worst_month", "worst_month_label",
                  "cagr", "avg_exposure", "switches_per_year", "d_cagr_vs_bh", "share_behind_years")

# §6.4 사전 등록 유지 조건 (공식 산출로 판정, 근사로 판정 금지)
RETENTION_CONDITIONS = {
    "a": "MaxDD_2003~24(채택 규칙) ≥ −2.5·σ_T",
    "b": "MaxDD_1993~24(변동성 단독) ≥ −k_slow·σ_T",
    "c": "p2 결정층 대비 MaxDD 개선 ≥ 5pp",
    "d": "전환 ≤ 24회/년 ∧ 누적 비용 ≤ 5% (2003~24)",
    "e": "최악 월이 p2 결정층보다 2pp 이상 나쁘지 않음",
    "f": "실현변동성/σ_T ∈ [0.75, 1.15]",
}
NARROW_MARGIN = 0.01                 # 이보다 적은 여유로 통과하면 경고 (§6.4: 근소한 (a)·(b) 는 뒤집힐 수 있다)
_RETENTION_BOUNDS = {"a_k": 2.5, "c_pp": 0.05, "d_switches": 24.0, "d_cost": 0.05,
                     "e_pp": 0.02, "f_lo": 0.75, "f_hi": 1.15}

# §6.5 정직한 읽기 — 표 위에 그대로 인쇄한다 (설계 단계 근사; 공식 수치가 있으면 honest_reading 이 갈아끼운다)
HONEST_READING = (
    "(a) 변동성을 맞추면 파라미터 0개의 변동성 목표(VT14)가 p2 결정층과 같다"
    " ({vt14_cagr}/{vt14_dd} vs {p2_cagr}/{p2_dd}, {vt14_sw} vs {p2_sw}회) — 비중은 확률 모델 위에 서 있지 않다.",
    "(b) 확률층이 MaxDD 를 낮추는 것은 바닥 없는 곱일 때뿐({volonly_dd} → {nofloor_dd};"
    " 2008-09~2009-06 평균 비중 0.12, 2020-03~05 0.10 = 사실상 퇴장)이며, 바닥을 두면 {adopted_dd}"
    "(변동성 단독 {volonly_dd} 와 같다). 상태 배수의 가치는 낙폭이 아니라 가족용 상태·킬룰 시험 대상이라는 점이다.",
    "(c) 변동성 목표는 완만한 약세장(2000~02: {mild_bear_dd})을 막지 못한다 — k_slow 3.5 의 이유.",
    "(d) 모든 규칙 행이 보유 CAGR 에 뒤지고, 채택 규칙은 달력연도의 약 {share_behind} 에서 뒤진다"
    " — 상품은 낙폭 통제이며 그 말을 그대로 쓴다. 현금 0%·5bp 는 분기별로 늦게 실행하는 가족의 실제 마찰"
    "(중개·세금)을 과소평가한다.",
)
_HONEST_DEFAULTS = {                 # 설계 단계 근사(심사 재현, M3 상태 기준) — 공식 수치로 대체할 것
    "vt14_cagr": "10.14%", "vt14_dd": "−30.3%", "p2_cagr": "10.19%", "p2_dd": "−31.0%",
    "vt14_sw": "3.6", "p2_sw": "3.9", "volonly_dd": "−24.1%", "nofloor_dd": "−17.9%",
    "adopted_dd": "−23.5%", "mild_bear_dd": "−30.7~−34.6%", "share_behind": "90%",
}


# ------------------------------------------------------------------
# 0. 사양 해시 · 파라미터 회계
# ------------------------------------------------------------------
def n_params() -> int:
    """비중 규칙의 적합 파라미터 수 = 0 (§6-b). 상수는 파라미터가 아니다."""
    return PARAM_COUNT


def sizing_constants() -> dict:
    """카드 정직 스트립·산출물에 그대로 기록하는 동결 상수 (전부 config 에서 온다 — 여기서 새로 만들지 않는다)."""
    return {
        "sigma_model": P3["sigma_model"], "ewma_lambda": P3["ewma_lambda"],
        "ewma_seed_sessions": P3["ewma_seed_sessions"], "k_slow": P3["k_slow"], "k_fast": P3["k_fast"],
        "vol_grid": tuple(P3["vol_grid"]), "d_max_default": P3["d_max_default"],
        "w_min": P3["w_min"], "w_max": P3["w_max"], "grid": P3["grid"], "band": P3["band"],
        "floor_after_multiplier": P3["floor_after_multiplier"], "cadence": P3["cadence"],
        "escalate_immediately": P3["escalate_immediately"], "cost_bps": P3["cost_bps"],
        "cash_return": P3["cash_return"], "churn_ceiling": P3["churn_ceiling"],
        "multiplier_map": {s: TONE_EXPOSURE[t] for s, t in STATE_TO_TONE.items()},
        "param_count": PARAM_COUNT, "sizing_rule": SIZING_RULE,
    }


SIZING_CONSTANTS = sizing_constants()


def sizing_sha256() -> str:
    """`mrl/sizing.py` 소스 + SIZING_RULE 의 SHA-256 — 모든 P3 산출물·장부 행에 기록한다.

    CRLF→LF 정규화(체크아웃 방식 무관). 소스가 없으면(설치본에서 .py 제거 등) 경고 + 표식으로 대신한다."""
    h = hashlib.sha256()
    src = Path(__file__).resolve()
    h.update(src.name.encode("utf-8") + b"\n")
    if src.exists():
        h.update(src.read_bytes().replace(b"\r\n", b"\n"))
    else:                                                    # pragma: no cover - 설치 형태에 따른 방어
        warnings.warn("sizing_sha256: 소스 없음 — 표식으로 대신함(파일이 생기면 해시가 바뀐다)")
        h.update(b"MISSING:sizing.py")
    h.update(b"\n" + SIZING_RULE.encode("utf-8"))
    return h.hexdigest()


# ------------------------------------------------------------------
# 1. 공용 헬퍼
# ------------------------------------------------------------------
def _check_index(obj, name: str) -> pd.DatetimeIndex:
    idx = obj.index if hasattr(obj, "index") else obj
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError(f"{name} 인덱스는 DatetimeIndex 여야 함 (받은 형: {type(idx).__name__})")
    if not idx.is_monotonic_increasing:
        raise ValueError(f"{name} 인덱스는 오름차순이어야 함")
    if idx.has_duplicates:
        raise ValueError(f"{name} 인덱스에 중복 날짜가 있음")
    return idx


def _asof_ts(asof) -> pd.Timestamp | None:
    if asof is None:
        return None
    ts = pd.Timestamp(asof)
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _clean_close(close: pd.Series, name: str = "spy_close", asof=None) -> pd.Series:
    """점 원칙: asof 로 **먼저** 자른 뒤 결측 제거·양수 검사."""
    if not isinstance(close, pd.Series):
        raise TypeError(f"{name} 는 pandas Series 여야 함 (받은 형: {type(close).__name__})")
    _check_index(close, name)
    s = close
    ts = _asof_ts(asof)
    if ts is not None:
        s = s.loc[s.index <= ts]
    s = s.dropna().astype(float)
    if len(s) == 0:
        raise ValueError(f"{name}: asof {ts} 이하에 유효 종가가 없음")
    if (s <= 0).any():
        bad = s.index[s <= 0]
        raise ValueError(f"{name}: 0 이하 종가 {len(bad)}행 (예: {bad[0].date()})")
    return s


def _finite(x) -> bool:
    """None·NaN·비수치를 한 번에 거른다."""
    if x is None:
        return False
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def _round_grid(w: float, grid: float) -> float:
    """cand = round(w/grid)·grid — 부동소수 먼지를 없애 장부·해시가 안정되게 한다."""
    return float(round(round(float(w) / float(grid)) * float(grid), _ROUND_DP))


def _week_end_flags(idx: pd.DatetimeIndex, cadence: str) -> np.ndarray:
    """점검일 플래그. weekly = 그 ISO 주의 마지막 거래일, monthly = 그 달의 마지막 거래일, daily = 매일.

    월간은 '월말 종가에 확정 → 다음 달 첫 세션 수익률에 적용' 이므로 §6.3 의 '월초 리밸런스' 와 같은 경로다."""
    c = str(cadence).lower()
    if c == "daily":
        return np.ones(len(idx), dtype=bool)
    if c not in ("weekly", "monthly"):
        raise ValueError(f"cadence 는 weekly/monthly/daily 중 하나여야 함: {cadence!r}")
    freq = "W" if c == "weekly" else "M"
    return calendar_us.period_end_from_index(idx, freq).to_numpy()


# ------------------------------------------------------------------
# 2. 변동성 예측 (EWMA λ=0.94 — RiskMetrics 상수, 적합 아님)
# ------------------------------------------------------------------
def ewma_vol(close: pd.Series, lam: float = P3["ewma_lambda"],
             seed_sessions: int = P3["ewma_seed_sessions"], asof=None) -> pd.Series:
    """연율화 EWMA 변동성 σ̂_t = sqrt(252·v_t), v_t = λ v_{t−1} + (1−λ) r_t², r_t = ln(C_t/C_{t−1}).

    v 는 첫 `seed_sessions` 개 수익률의 **표본분산(ddof=1)** 으로 seed 되며 그 세션에서 처음 유효해진다
    (그 앞은 NaN — 이월하지 않는다). 재귀는 인과적이라 미래 행을 덧붙여도 과거 값이 변하지 않는다(점 원칙).
    `asof` 를 주면 **먼저** 자른 뒤 계산한다."""
    lam = float(lam)
    if not (0.0 < lam < 1.0):
        raise ValueError(f"lam 은 (0,1) 이어야 함: {lam}")
    seed_sessions = int(seed_sessions)
    if seed_sessions < 2:
        raise ValueError(f"seed_sessions 는 2 이상이어야 함: {seed_sessions}")
    s = _clean_close(close, "close", asof=asof)
    px = s.to_numpy(dtype=float)
    n_r = len(px) - 1
    out = pd.Series(np.nan, index=s.index, name="sigma_ewma")
    if n_r < seed_sessions:
        warnings.warn(f"ewma_vol: 수익률 {n_r}개 < seed {seed_sessions} — 전부 NaN", stacklevel=2)
        return out
    r = np.log(px[1:] / px[:-1])
    v = np.full(n_r, np.nan, dtype=float)
    v[seed_sessions - 1] = float(np.var(r[:seed_sessions], ddof=1))
    one_m = 1.0 - lam
    for t in range(seed_sessions, n_r):                      # 인과적 재귀 — 벡터화하면 λ^t 언더플로가 생긴다
        v[t] = lam * v[t - 1] + one_m * r[t] * r[t]
    out.iloc[1:] = np.sqrt(TRADING_DAYS * v)
    out.attrs = {"lam": lam, "seed_sessions": seed_sessions, "annualized": True,
                 "first_valid": str(out.first_valid_index().date()) if out.notna().any() else None}
    return out


# ------------------------------------------------------------------
# 3. 예산 → 목표 변동성 · 사다리
# ------------------------------------------------------------------
def target_vol(d_max: float, k: float = P3["k_slow"], grid=P3["vol_grid"]) -> float:
    """가족 위험예산 D_max → 목표 변동성 σ_T = max{g ∈ grid : g ≤ D_max/k} (격자 **내림**).

    D_max/k 가 격자 최소값(6%) 미만이면 ValueError — 바닥 0.25 만으로도 55% 급락에서 ≥14% 를 잃으므로
    그보다 낮은 예산선은 제공하지 않는다(§6.1.2)."""
    d = float(d_max)
    k = float(k)
    if not (0.0 < d < 1.0):
        raise ValueError(f"d_max 는 (0,1) 이어야 함: {d_max}")
    if k <= 0.0:
        raise ValueError(f"k 는 양수여야 함: {k}")
    g = sorted(float(x) for x in grid)
    if not g:
        raise ValueError("vol_grid 가 비었음")
    ratio = d / k
    ok = [x for x in g if x <= ratio + _SIZING_EPS]
    if not ok:
        raise ValueError(f"D_max {d:.4f} / k {k} = {ratio:.4f} 가 격자 최소값 {g[0]:.4f} 미만 — "
                         f"이 예산선은 제공하지 않는다 (§6.1.2: D_max ≥ {g[0] * k:.2f})")
    return float(ok[-1])


def budget_ladder(grid=P3["vol_grid"], k_slow: float = P3["k_slow"], k_fast: float = P3["k_fast"],
                  results: dict | None = None) -> pd.DataFrame:
    """§6.2 예산 사다리: 같은 σ_T 를 두 예산선으로 읽는다.

    D_slow = k_slow·σ_T (완만한 약세장선; 헤드라인) · D_fast = k_fast·σ_T (급락형 낙관선, 병기만).
    `results` = {σ_T: {LADDER_COLUMNS 의 결과 열}} 가 있으면 채운다(없으면 NaN — 근사와 공식을 섞지 않는다)."""
    k_slow = float(k_slow)
    k_fast = float(k_fast)
    if k_fast >= k_slow:
        raise ValueError(f"k_fast({k_fast}) 는 k_slow({k_slow}) 보다 작아야 함 (낙관선)")
    rows = []
    for g in sorted(float(x) for x in grid):
        row = {"sigma_target": g, "d_slow": round(k_slow * g, _ROUND_DP), "d_fast": round(k_fast * g, _ROUND_DP)}
        for c in LADDER_COLUMNS[3:]:
            row[c] = np.nan
        if results:
            fill = results.get(g) or results.get(round(g, 4)) or {}
            unknown = sorted(set(fill) - set(LADDER_COLUMNS))
            if unknown:
                raise ValueError(f"budget_ladder: 알 수 없는 결과 열 {unknown}")
            row.update(fill)
        rows.append(row)
    out = pd.DataFrame(rows, columns=list(LADDER_COLUMNS))
    out.attrs = {"k_slow": k_slow, "k_fast": k_fast, "d_max_default": P3["d_max_default"],
                 "sigma_target_default": target_vol(P3["d_max_default"], k_slow, grid)}
    return out


def budget_sentence(d_max: float, sigma_target: float, maxdd_decision=None, maxdd_vol_only=None) -> str:
    """§6.2 카드 문장. 공식 수치가 없으면 설계 단계 근사를 라벨과 함께 쓴다(섞지 않는다)."""
    approx = maxdd_decision is None or maxdd_vol_only is None
    dd_dec = -0.235 if maxdd_decision is None else float(maxdd_decision)
    dd_vol = -0.307 if maxdd_vol_only is None else float(maxdd_vol_only)
    tail = " (설계 단계 근사)" if approx else ""
    return (f"예산 −{d_max * 100:.0f}%(완만한 약세장 기준) ≈ 급락형 −{d_max * P3['k_fast'] / P3['k_slow'] * 100:.0f}% "
            f"→ 목표 변동성 {sigma_target * 100:.0f}%. 2003년 이후 기록에서는 −{abs(dd_dec) * 100:.1f}% 였고, "
            f"2000~02년형 완만한 약세장에서는 −{abs(dd_vol) * 100:.1f}% 까지 갔다 — 예산은 보장이 아니다.{tail}")


def honest_reading(results: dict | None = None) -> list[str]:
    """§6.5 정직한 읽기 4줄. `results` 로 공식 수치를 넣으면 갈아끼우고, 없으면 설계 단계 근사 라벨을 붙인다."""
    vals = dict(_HONEST_DEFAULTS)
    unknown = sorted(set(results or {}) - set(_HONEST_DEFAULTS))
    if unknown:
        raise ValueError(f"honest_reading: 알 수 없는 키 {unknown}")
    vals.update(results or {})
    tail = "" if results else "  (설계 단계 근사 — 공식 산출로 대체)"
    return [line.format(**vals) + tail for line in HONEST_READING]


# ------------------------------------------------------------------
# 4. 상태 배수 · 목표 비중
# ------------------------------------------------------------------
def state_multiplier(states):
    """p2 결정 상태 → 노출 배수 = TONE_EXPOSURE[STATE_TO_TONE[state]] (1.00 / 0.50 / 0.25; 새 숫자 없음).

    Series 를 주면 Series, 스칼라를 주면 float 를 돌려준다(daily.py 는 오늘 한 값만 쓴다).
    결측(None·NaN)은 NaN, 등록되지 않은 상태 문자열은 ValueError(조용한 실패 금지)."""
    table = {s: float(TONE_EXPOSURE[t]) for s, t in STATE_TO_TONE.items()}
    if isinstance(states, pd.Series):
        s = states
        known = set(table)
        bad = sorted({v for v in s.dropna().unique() if v not in known})
        if bad:
            raise ValueError(f"등록되지 않은 결정 상태: {bad} (허용: {sorted(known)})")
        out = s.map(table).astype(float)
        out.name = "mult"
        return out
    if states is None or (isinstance(states, float) and math.isnan(states)):
        return float("nan")
    if states not in table:
        raise ValueError(f"등록되지 않은 결정 상태: {states!r} (허용: {sorted(table)})")
    return float(table[states])


def exposure_target(sigma, mult, sigma_target: float, w_min: float = P3["w_min"], w_max: float = P3["w_max"],
                    floor_after: bool = P3["floor_after_multiplier"]) -> pd.DataFrame:
    """w_vol = clip(σ_T/σ̂, w_min, w_max) → w_target = clip(w_vol × m, w_min, w_max).

    `floor_after=False` 면 곱 뒤 재클립을 하지 않는다(민감도 S-nofloor: 비중 0.0625~0.125 가 나온다).
    스칼라를 주면 1행 DataFrame 을 돌려준다(daily.py 는 `.iloc[0]` 로 읽는다). 열: w_vol, mult, w_target."""
    sigma_target = float(sigma_target)
    if not (0.0 < sigma_target < 5.0):
        raise ValueError(f"sigma_target 은 (0,5) 이어야 함: {sigma_target}")
    if not (0.0 <= w_min <= w_max <= 1.0):
        raise ValueError(f"[w_min, w_max] 가 [0,1] 안의 구간이 아님: [{w_min}, {w_max}]")
    scalar = not isinstance(sigma, pd.Series)
    if scalar:
        idx = pd.RangeIndex(1)
        sig = pd.Series([np.nan if not _finite(sigma) else float(sigma)], index=idx, dtype=float)
        m = pd.Series([np.nan if not _finite(mult) else float(mult)], index=idx, dtype=float)
    else:
        _check_index(sigma, "sigma")
        sig = sigma.astype(float)
        if isinstance(mult, pd.Series):
            m = mult.reindex(sig.index).astype(float)
        else:
            m = pd.Series(np.nan if not _finite(mult) else float(mult), index=sig.index, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        w_vol = (sigma_target / sig).clip(lower=w_min, upper=w_max)
    w_vol = w_vol.where(sig > 0.0)                           # σ̂ ≤ 0·NaN → NaN (이월 금지)
    prod = w_vol * m
    w_target = prod.clip(lower=w_min, upper=w_max) if floor_after else prod
    return pd.DataFrame({"w_vol": w_vol, "mult": m, "w_target": w_target})


# ------------------------------------------------------------------
# 5. 실행 (격자 · 무거래 밴드 · 주간 리듬 · 즉시 격상)
# ------------------------------------------------------------------
def step(w_target_t, mult_t, mult_prev, w_exec_prev, week_end: bool, cfg=P3,
         *, deploy: bool = True) -> tuple[float, str]:
    """한 세션의 실행 비중 (daily.py 용; 재개 안전 — 장부의 마지막 유효 행만 있으면 이어 계산된다).

    우선순위(§6.1.5): 배치 아님 → `info_only` / 입력 결측 → `input_missing`(직전 값 **유지**) /
    직전 값 없음 → `init` / 격상(m 하락) → 즉시 하향 min(cand, prev) `escalation` /
    주 마지막 세션 ∧ |w_target − prev| ≥ band → `weekly` / 그 외 `hold`.
    반환 (w_exec, reason). 절대 1.0 으로 복귀하지 않는다.

    밴드 비교는 §6.1.5 를 **문자 그대로** IEEE-754 배정밀도로 한다(여유 epsilon 없음): 상한에 걸린
    w_target = 1.0 과 prev = 0.90 의 차이는 부동소수에서 0.09999999999999998 이라 재조정이 아니다.
    이 경계 처리는 설계 단계(1993~2024-08) 백테스트를 만든 구현과 동일하며 IEEE-754 는 플랫폼 간 결정적이다.
    `tests/test_sizing.py::test_band_boundary_is_literal` 이 이 동작을 고정한다."""
    grid = float(cfg["grid"])
    band = float(cfg["band"])
    escalate = bool(cfg.get("escalate_immediately", True))
    prev = float(w_exec_prev) if _finite(w_exec_prev) else None
    if not deploy:
        return (prev if prev is not None else float("nan")), "info_only"
    if not (_finite(w_target_t) and _finite(mult_t)):
        return (prev if prev is not None else float("nan")), "input_missing"
    cand = _round_grid(w_target_t, grid)                      # §6.1.5 그대로 — 바닥/상한은 exposure_target 의 몫
    if prev is None:
        return cand, "init"
    if escalate and _finite(mult_prev) and float(mult_t) < float(mult_prev):
        return min(cand, prev), "escalation"                 # 즉시, 밴드 무시, 하향만
    if bool(week_end) and abs(float(w_target_t) - prev) >= band:
        return cand, "weekly"
    return prev, "hold"


def execute(w_target: pd.Series, mult: pd.Series, *, grid: float = P3["grid"], band: float = P3["band"],
            cadence: str = P3["cadence"], escalate: bool = P3["escalate_immediately"],
            init_w=None, init_mult=None, check=None) -> pd.DataFrame:
    """비중 경로 전체를 `step` 과 **같은 규칙**으로 굴린다. 열: w_exec, reason, cand, week_end.

    `init_w`/`init_mult` 를 주면 그 상태에서 이어간다(재개; 장부의 마지막 유효 행). 점 원칙: 어떤 행도
    미래를 읽지 않으므로 뒤에 행을 덧붙여도 앞의 w_exec 는 변하지 않는다.
    `check` (bool Series)를 주면 cadence 대신 그 점검일 플래그를 쓴다 — 민감도 S-monthly/S-mild 의
    'C 안: 월말 ∪ 상태 변경' 리듬 전용이며 생산 규칙(`run`)은 절대 쓰지 않는다."""
    _check_index(w_target, "w_target")
    idx = w_target.index
    if isinstance(mult, pd.Series):
        m_ser = mult.reindex(idx)
    else:
        m_ser = pd.Series(np.nan if not _finite(mult) else float(mult), index=idx, dtype=float)
    wt = w_target.to_numpy(dtype=float)
    mu = m_ser.to_numpy(dtype=float)
    if check is None:
        week_end = _week_end_flags(idx, cadence)
    else:
        week_end = pd.Series(check).reindex(idx).fillna(False).to_numpy(dtype=bool)
    cfg = dict(P3, grid=float(grid), band=float(band), escalate_immediately=bool(escalate))
    n = len(idx)
    w_exec = np.full(n, np.nan)
    cand = np.full(n, np.nan)
    reasons: list[str] = []
    prev = float(init_w) if _finite(init_w) else None
    mprev = float(init_mult) if _finite(init_mult) else np.nan
    for t in range(n):
        if _finite(wt[t]):
            cand[t] = _round_grid(wt[t], grid)
        w, why = step(wt[t], mu[t], mprev, prev, bool(week_end[t]), cfg)
        w_exec[t] = w
        reasons.append(why)
        if _finite(w):
            prev = float(w)
        if _finite(mu[t]):
            mprev = float(mu[t])
    out = pd.DataFrame({"w_exec": w_exec, "reason": reasons, "cand": cand, "week_end": week_end}, index=idx)
    n_chg = int((out["w_exec"].diff().abs() > _SIZING_EPS).sum())
    out.attrs = {"grid": float(grid), "band": float(band), "cadence": str(cadence), "escalate": bool(escalate),
                 "n_changes": n_chg, "sizing_rule": SIZING_RULE}
    return out


def next_thresholds(w_exec: float, sigma_target: float, mult: float, asof, band: float = P3["band"]) -> dict:
    """가족용 파생값(표시 전용): 한 단계 하향/상향이 일어나는 예상 변동성과 다음 점검일.

    하향은 w_target ≤ w_exec − band 에서 일어나므로 σ_down = σ_T·m/(w_exec − band);
    상향은 w_target ≥ w_exec + band 에서 σ_up = σ_T·m/(w_exec + band).
    격자 바닥·상태 배수 상한 때문에 도달 불가능한 쪽은 None 과 사유를 돌려준다(조용한 0 금지).
    상향은 w_target = clip(w_vol·m, w_min, w_max) ≤ max(w_min, w_max·m) 이므로 상태 배수 m 이 1 보다
    작으면 상한이 함께 내려간다 — 그 위를 광고하면 도달 불가능한 임계를 카드에 찍게 된다."""
    w_min, w_max = float(P3["w_min"]), float(P3["w_max"])
    out: dict = {"w_exec": float(w_exec) if _finite(w_exec) else None,
                 "mult": float(mult) if _finite(mult) else None,
                 "sigma_target": float(sigma_target), "band": float(band),
                 "sigma_down": None, "sigma_up": None, "down_note": None, "up_note": None}
    nc = _next_week_end(asof)
    out["next_check"] = str(nc)
    if not (_finite(w_exec) and _finite(mult)):
        out["down_note"] = out["up_note"] = "입력 결측 — 임계 계산 불가"
        return out
    w = float(w_exec)
    m = float(mult)
    lo, hi = w - float(band), w + float(band)
    if lo < w_min - _SIZING_EPS:
        out["down_note"] = f"바닥 {w_min:.2f} 때문에 한 단계 하향이 불가능(현재 {w:.2f})"
    else:
        out["sigma_down"] = float(sigma_target) * m / lo
    cap_up = max(w_min, w_max * m)          # 곱 뒤 재클립까지 반영한 실제 도달 상한: w_target ≤ max(w_min, w_max·m)
    if hi > cap_up + _SIZING_EPS:
        out["up_note"] = (f"상한 {w_max:.2f} 때문에 한 단계 상향이 불가능(현재 {w:.2f})"
                          if cap_up >= w_max - _SIZING_EPS else
                          f"상태 배수 {m:.2f} 의 상한 {cap_up:.2f} 때문에 한 단계 상향이 불가능(현재 {w:.2f})")
    else:
        out["sigma_up"] = float(sigma_target) * m / hi
    return out


def _next_week_end(asof):
    """asof **이후** 첫 주간 점검일(그 주의 마지막 거래일)."""
    d = calendar_us.next_trading_day(asof)
    for _ in range(30):
        if calendar_us.is_period_end(d, "W"):
            return d
        d = calendar_us.next_trading_day(d)
    raise RuntimeError(f"다음 주간 점검일을 찾지 못함: {asof}")     # pragma: no cover - 캘린더가 깨진 경우


def run(sigma: pd.Series, states, sigma_target: float, cfg=P3, deploy: bool = True) -> pd.DataFrame:
    """σ̂ + p2 상태 → 비중 경로. 열: sigma, w_vol, mult, w_target, cand, w_exec, reason, week_end.

    `deploy=False`(유효 배치 모드가 tones 가 아님) → 상태 배수를 적용하지 않고 w_exec 는 NaN,
    reason 은 전부 `info_only`; w_vol 은 장부에 회색 '가정치' 로만 남는다(§6.1.4)."""
    _check_index(sigma, "sigma")
    idx = sigma.index
    mult = state_multiplier(states) if isinstance(states, pd.Series) else states
    if isinstance(mult, pd.Series):
        mult = mult.reindex(idx)
    et = exposure_target(sigma, mult if deploy else np.nan, sigma_target,
                         w_min=cfg["w_min"], w_max=cfg["w_max"], floor_after=cfg["floor_after_multiplier"])
    week_end = _week_end_flags(idx, cfg["cadence"])
    if not deploy:
        out = pd.DataFrame({"sigma": sigma.astype(float), "w_vol": et["w_vol"], "mult": np.nan,
                            "w_target": et["w_vol"], "cand": np.nan, "w_exec": np.nan,
                            "reason": "info_only", "week_end": week_end}, index=idx)
    else:
        ex = execute(et["w_target"], et["mult"], grid=cfg["grid"], band=cfg["band"],
                     cadence=cfg["cadence"], escalate=cfg["escalate_immediately"])
        out = pd.DataFrame({"sigma": sigma.astype(float), "w_vol": et["w_vol"], "mult": et["mult"],
                            "w_target": et["w_target"], "cand": ex["cand"], "w_exec": ex["w_exec"],
                            "reason": ex["reason"], "week_end": ex["week_end"]}, index=idx)
    n_missing = int((out["reason"] == "input_missing").sum())
    notes = []
    if n_missing:
        notes.append(f"입력 결측 {n_missing}행 — 비중 직전 값 유지(input_missing)")
    out.attrs = {"sigma_target": float(sigma_target), "deploy": bool(deploy), "sizing_rule": SIZING_RULE,
                 "sizing_sha256": sizing_sha256(), "param_count": PARAM_COUNT,
                 "constants": sizing_constants(), "warnings": notes}
    return out


# ------------------------------------------------------------------
# 6. 배분 시뮬 (§6.6) — 구현은 `mrl/evaluate.py` 에 하나만 둔다
#    ARCHITECTURE_PHASE3.md §6.6 은 이 두 함수를 evaluate 의 계약으로 적었고, §6.3 백테스트 표와 장부의
#    rule_ret_20 이 같은 규약·같은 숫자를 써야 하므로 여기서는 **재수출만** 한다
#    (공개 API 불변: sizing.allocation_from_weights is evaluate.allocation_from_weights).
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# 7. 백테스트 표 (§6.3 고정 형식) · 민감도 · 유지 조건
# ------------------------------------------------------------------
def _slice(s: pd.Series, span) -> pd.Series:
    a, b = span
    out = s
    if a is not None:
        out = out.loc[out.index >= pd.Timestamp(a)]
    if b is not None:
        out = out.loc[out.index <= pd.Timestamp(b)]
    return out


def backtest_table(paths: dict, spy_close: pd.Series, windows: dict, cost_bps=P3["cost_bps"],
                   references: dict | None = None, sigma_target: float | None = None) -> pd.DataFrame:
    """§6.3 백테스트 표. `paths` = {행 이름: 비중 Series}, `windows` = {기간 라벨: (start, end)}.

    보유(w≡1) 행은 창마다 **자동으로** 맨 앞에 들어간다. `references` 의 행(v0 completed·p2 결정층·VT14)은
    그 다음, `paths` 는 마지막. `cost_bps` 는 스칼라 또는 {행 이름: bp} (S-10bp 처럼 같은 경로·다른 비용).
    `sigma_target` 을 주면 실현변동성/σ_T·MaxDD/σ_T 열을 채운다(없으면 NaN — 근사와 공식을 섞지 않는다)."""
    close = spy_close.dropna().astype(float)
    _check_index(close, "spy_close")
    refs = dict(references or {})
    dup = sorted(set(refs) & set(paths))
    if dup:
        raise ValueError(f"references 와 paths 에 같은 행 이름: {dup}")
    ordered = [(ROW_BUY_HOLD, None)] + [(k, v) for k, v in refs.items()] + [(k, v) for k, v in paths.items()]
    rows = []
    for label, span in windows.items():
        c_win = _slice(close, span)
        if len(c_win) < 2:
            raise ValueError(f"창 {label}: 거래일이 2개 미만")
        for name, w in ordered:
            ws = pd.Series(1.0, index=c_win.index) if w is None else _slice(pd.Series(w).astype(float), span)
            ws = ws.dropna()
            if len(ws) < 2:
                warnings.warn(f"{name} / {label}: 유효 비중 {len(ws)}행 — 행을 건너뜁니다", stacklevel=2)
                continue
            if ws.index[0] > c_win.index[min(10, len(c_win) - 1)]:       # 라벨과 실제 구간이 다르면 알린다
                warnings.warn(f"{name} / {label}: 비중이 {ws.index[0].date()} 부터라 창 시작"
                              f"({c_win.index[0].date()})보다 늦습니다 — 행의 start/end 를 읽을 것", stacklevel=2)
            if name == ROW_BUY_HOLD:
                bps = 0.0                                    # 보유는 거래가 없다
            elif isinstance(cost_bps, dict):
                if name not in cost_bps:
                    raise ValueError(f"cost_bps 사전에 행 {name!r} 이 없음 (조용한 기본값 금지)")
                bps = float(cost_bps[name])
            else:
                bps = float(cost_bps)
            a = allocation_from_weights(ws, c_win, cost_bps=bps)
            vr = (a["ann_vol"] / sigma_target) if sigma_target else np.nan
            dr = (abs(a["max_dd"]) / sigma_target) if sigma_target else np.nan
            rows.append({
                "row": name, "window": label, "start": a["start"], "end": a["end"], "years": a["years"],
                "cagr": a["cagr"], "max_dd": a["max_dd"],
                "max_dd_date": None if pd.isna(a["max_dd_date"]) else str(pd.Timestamp(a["max_dd_date"]).date()),
                "worst_month": a["worst_month"], "worst_month_label": a["worst_month_label"],
                "ann_vol": a["ann_vol"], "switches_per_year": a["switches_per_year"],
                "avg_exposure": a["avg_exposure"], "cost_total": a["cost_total"],
                "d_cagr_vs_bh": a["excess_cagr"], "d_maxdd_vs_bh": a["maxdd_improvement"],
                "vol_ratio": vr, "maxdd_over_sigma_t": dr,
                "n_switches": a["n_switches"], "cost_bps": a["cost_bps"],
            })
    out = pd.DataFrame(rows, columns=list(BACKTEST_COLUMNS))
    out.attrs = {"sigma_target": sigma_target, "sizing_rule": SIZING_RULE, "sizing_sha256": sizing_sha256(),
                 "columns_ko": dict(BACKTEST_COLUMNS_KO), "cash_return": P3["cash_return"]}
    return out


def sensitivities(feats, states, spy_close: pd.Series, cfg=P3, sigma_target: float | None = None) -> dict:
    """§6.3 민감도 비중 경로 (전부 표시·소거용; 생산 규칙은 `run` 하나뿐).

    `feats` 는 `har_fc_20`(또는 `p2_har_fc_20`) 열을 가진 프레임이거나 None — 없으면 S-HAR 을 경고와 함께 건너뛴다.
    반환 {이름: w_exec Series}.

    §6.3 표의 정의를 그대로 옮긴다:
      * S-monthly · S-mild = C 안의 리듬 '월말 ∪ 상태 변경'(밴드 0.10 유지, 격상 즉시), 바닥 0.25
      * S-noesc · S-lag · S-10bp · S-HAR · S-daily 와 목표 격자(7.5/12.5/15%) = **바닥 없는 곱** 위에서 잰다
        (§6.3 의 그 묶음 MaxDD 가 −12~−18% 인 이유). S-daily 는 '일간 5pp' = cadence 일간 + 밴드 = 격자 0.05
      * S-10bp 는 S-nofloor 와 **같은 경로**이며 비용만 다르다 — `backtest_table(cost_bps={"S-10bp": 10, ...})`"""
    sT = float(sigma_target) if sigma_target is not None else target_vol(cfg["d_max_default"], cfg["k_slow"], cfg["vol_grid"])
    sigma = ewma_vol(spy_close, cfg["ewma_lambda"], cfg["ewma_seed_sessions"])
    st = states if isinstance(states, pd.Series) else pd.Series(states)
    idx = st.index
    sig = sigma.reindex(idx)
    mult = state_multiplier(st)
    mild = st.map({k: float(v) for k, v in MILD_MULTIPLIER.items()}).astype(float)
    w_vol = exposure_target(sig, 1.0, sT, cfg["w_min"], cfg["w_max"], floor_after=True)["w_vol"]
    out: dict[str, pd.Series] = {}

    def _exec(w_target, m, **kw):
        kw.setdefault("grid", cfg["grid"])
        kw.setdefault("band", cfg["band"])
        kw.setdefault("cadence", cfg["cadence"])
        kw.setdefault("escalate", cfg["escalate_immediately"])
        return execute(w_target, m, **kw)["w_exec"]

    base_target = (w_vol * mult).clip(lower=cfg["w_min"], upper=cfg["w_max"])
    nofloor_target = w_vol * mult                             # 바닥 없는 곱 (B 원안)
    month_or_change = pd.Series(_week_end_flags(idx, "monthly"), index=idx) | (mult != mult.shift(1))
    mild_check = pd.Series(_week_end_flags(idx, "monthly"), index=idx) | (mild != mild.shift(1))

    out["S-nofloor"] = _exec(nofloor_target, mult)
    out["S-volonly"] = _exec(w_vol, pd.Series(1.0, index=idx))
    out["S-monthly"] = _exec(base_target, mult, check=month_or_change)        # C: 월말 ∪ 상태 변경
    out["S-mild"] = _exec((w_vol * mild).clip(lower=cfg["w_min"], upper=cfg["w_max"]), mild, check=mild_check)
    out["S-grid"] = _exec(w_vol, pd.Series(1.0, index=idx), grid=GRID_SENSITIVITY["grid"],
                          band=GRID_SENSITIVITY["band"], cadence=GRID_SENSITIVITY["cadence"])
    out["S-grid-min"] = _exec(np.minimum(w_vol, mult), mult, grid=GRID_SENSITIVITY["grid"],
                              band=GRID_SENSITIVITY["band"], cadence=GRID_SENSITIVITY["cadence"])
    w_vt14 = exposure_target(sig, 1.0, 0.14, cfg["w_min"], cfg["w_max"])["w_vol"]
    out["S-VT14"] = _exec(w_vt14, pd.Series(1.0, index=idx), grid=GRID_SENSITIVITY["grid"],
                          band=GRID_SENSITIVITY["band"], cadence=GRID_SENSITIVITY["cadence"])
    # 아래 네 행은 §6.3 의 '바닥 없음' 묶음 — 바닥 있는 채택 규칙이 아니라 S-nofloor 위에서 잰다
    nofloor_base = _exec(nofloor_target, mult)
    out["S-noesc"] = _exec(nofloor_target, mult, escalate=False)
    out["S-lag"] = nofloor_base.shift(1)
    out["S-10bp"] = nofloor_base                              # 같은 경로, 비용만 10bp
    out["S-daily"] = _exec(nofloor_target, mult, cadence="daily", band=cfg["grid"])   # 일간 5pp
    for g in (0.075, 0.125, 0.15):
        wv = exposure_target(sig, 1.0, g, cfg["w_min"], cfg["w_max"])["w_vol"]
        out[f"S-VT{g * 100:g}"] = _exec(wv * mult, mult)      # 목표 격자 행도 바닥 없음(§6.3)

    har = None
    if feats is not None:
        for col in ("har_fc_20", "p2_har_fc_20"):
            if col in getattr(feats, "columns", []):
                har = feats[col].reindex(idx).astype(float)
                break
    if har is None:
        warnings.warn("sensitivities: har_fc_20 열이 없어 S-HAR 을 건너뜁니다", stacklevel=2)
    else:
        w_har = exposure_target(har, 1.0, sT, cfg["w_min"], cfg["w_max"])["w_vol"]
        out["S-HAR"] = _exec(w_har * mult, mult)
    return out


def _row(table: pd.DataFrame, row: str, window: str) -> pd.Series | None:
    m = table[(table["row"] == row) & (table["window"] == window)]
    return None if m.empty else m.iloc[0]


def retention(table: pd.DataFrame, sigma_target: float, cfg=P3, *, adopted: str = ROW_ADOPTED,
              vol_only: str = ROW_VOL_ONLY, reference: str = ROW_P2_DECISION,
              eval_window: str = WINDOW_EVAL, vol_only_window: str = WINDOW_VOL_ONLY) -> dict:
    """§6.4 사전 등록 유지 조건 (a)~(f) 판정 → `deploy_sizing`.

    하나라도 위반하면 `deploy_sizing=False` — 카드의 비중 블록을 숨기고 장부는 계속 기록한다.
    필요한 행이 표에 없으면 그 조건은 `None`(미판정)이고 `deploy_sizing` 은 False + `missing` 에 기록한다
    (조용히 통과시키지 않는다)."""
    sT = float(sigma_target)
    a_row = _row(table, adopted, eval_window)
    v_row = _row(table, vol_only, vol_only_window)
    r_row = _row(table, reference, eval_window)
    b = _RETENTION_BOUNDS
    res: dict[str, dict] = {}
    missing: list[str] = []

    def _put(key, value, threshold, ok, note, margin=None):
        res[key] = {"name": RETENTION_CONDITIONS[key], "value": value, "threshold": threshold,
                    "pass": ok, "note_ko": note, "margin": margin}

    if a_row is None:
        missing.append(f"{adopted}/{eval_window}")
        for k in ("a", "d", "f"):
            _put(k, None, None, None, "행 없음 — 미판정")
    else:
        thr_a = -b["a_k"] * sT
        _put("a", float(a_row["max_dd"]), thr_a, bool(a_row["max_dd"] >= thr_a),
             f"MaxDD {a_row['max_dd'] * 100:.2f}% vs 한계 {thr_a * 100:.2f}%", float(a_row["max_dd"]) - thr_a)
        ok_d = bool(a_row["switches_per_year"] <= b["d_switches"] and a_row["cost_total"] <= b["d_cost"])
        _put("d", {"switches_per_year": float(a_row["switches_per_year"]), "cost_total": float(a_row["cost_total"])},
             {"switches_per_year": b["d_switches"], "cost_total": b["d_cost"]}, ok_d,
             f"전환 {a_row['switches_per_year']:.1f}회/년 · 누적 비용 {a_row['cost_total'] * 100:.1f}%")
        ratio = float(a_row["ann_vol"]) / sT
        _put("f", ratio, [b["f_lo"], b["f_hi"]], bool(b["f_lo"] <= ratio <= b["f_hi"]),
             f"실현변동성 {a_row['ann_vol'] * 100:.1f}% / σ_T {sT * 100:.0f}% = {ratio:.2f}",
             min(ratio - b["f_lo"], b["f_hi"] - ratio))
    if v_row is None:
        missing.append(f"{vol_only}/{vol_only_window}")
        _put("b", None, None, None, "행 없음 — 미판정")
    else:
        thr_b = -float(cfg["k_slow"]) * sT
        _put("b", float(v_row["max_dd"]), thr_b, bool(v_row["max_dd"] >= thr_b),
             f"변동성 단독 MaxDD {v_row['max_dd'] * 100:.2f}% vs 한계 {thr_b * 100:.2f}%", float(v_row["max_dd"]) - thr_b)
    if a_row is None or r_row is None:
        missing.append(f"{reference}/{eval_window}")
        for k in ("c", "e"):
            if k not in res:
                _put(k, None, None, None, "행 없음 — 미판정")
    else:
        impr = float(a_row["max_dd"]) - float(r_row["max_dd"])          # 둘 다 음수: 얕을수록 양수
        _put("c", impr, b["c_pp"], bool(impr >= b["c_pp"]),
             f"MaxDD 개선 {impr * 100:+.1f}pp (채택 {a_row['max_dd'] * 100:.1f}% vs 결정층 {r_row['max_dd'] * 100:.1f}%)",
             impr - b["c_pp"])
        gap = float(a_row["worst_month"]) - float(r_row["worst_month"])  # 음수면 채택이 더 나쁨
        _put("e", gap, -b["e_pp"], bool(gap >= -b["e_pp"]),
             f"최악 월 {a_row['worst_month'] * 100:.1f}% vs 결정층 {r_row['worst_month'] * 100:.1f}% ({gap * 100:+.1f}pp)",
             gap + b["e_pp"])

    failed = sorted(k for k, v in res.items() if v["pass"] is False)
    undecided = sorted(k for k, v in res.items() if v["pass"] is None)
    # 근소 통과를 boolean 뒤에 숨기지 않는다 — §6.4 가 (a)·(b) 는 공식 수치에서 뒤집힐 수 있다고 미리 적었다
    narrow = sorted(k for k, v in res.items()
                    if v["pass"] is True and v["margin"] is not None and v["margin"] < NARROW_MARGIN)
    if narrow:
        warnings.warn("유지 조건 " + ", ".join(f"({k}) 여유 {res[k]['margin'] * 100:+.2f}pp" for k in narrow)
                      + f" — {NARROW_MARGIN * 100:.0f}pp 미만으로 통과(자료가 조금만 바뀌어도 뒤집힌다)", stacklevel=2)
    return {"conditions": res, "failed": failed, "undecided": undecided, "missing": sorted(set(missing)),
            "narrow": narrow, "narrow_margin": NARROW_MARGIN,
            "sigma_target": sT, "k_slow": float(cfg["k_slow"]),
            "deploy_sizing": bool(not failed and not undecided),
            "note_ko": ("모든 유지 조건 통과 — 비중 블록 표시" + (f" (근소 통과: {narrow})" if narrow else "")
                        if not failed and not undecided
                        else f"유지 조건 위반/미판정 {failed + undecided} — 비중 블록 숨김(장부는 계속 기록)")}


# ------------------------------------------------------------------
# 8. 에피소드 손익 · 상대성과
# ------------------------------------------------------------------
def episode_pnl(w_exec: pd.Series, spy_close: pd.Series, episodes5: pd.DataFrame) -> pd.DataFrame:
    """에피소드(고점→저점)별 규칙 손익 vs 보유 손익 + 그 구간 평균 비중.

    점 원칙 그대로 w[t] → t+1 수익률. 구간 = (peak_date, trough_date]. 비중 경로 밖의 에피소드는 건너뛴다."""
    _check_index(w_exec, "w_exec")
    close = spy_close.dropna().astype(float)
    _check_index(close, "spy_close")
    common = w_exec.index.intersection(close.index)
    if len(common) < 2:
        raise ValueError("에피소드 손익에 최소 2 거래일이 필요합니다")
    c = close.loc[common]
    w = w_exec.loc[common].astype(float)
    r = c.pct_change()
    w_prev = w.shift(1)
    ret_rule = (1.0 + w_prev * r) - 1.0                       # 비용 제외(에피소드 손익 비교는 총액이 아니라 형태)
    rows = []
    for _, ep in episodes5.iterrows():
        p, t = pd.Timestamp(ep["peak_date"]), pd.Timestamp(ep["trough_date"])
        seg = (c.index > p) & (c.index <= t)
        if seg.sum() < 1:
            continue
        rr = float(np.prod(1.0 + ret_rule[seg].dropna()) - 1.0)
        rb = float(np.prod(1.0 + r[seg].dropna()) - 1.0)
        rows.append({"peak_date": str(p.date()), "trough_date": str(t.date()),
                     "depth": float(ep["depth"]), "ret_rule": rr, "ret_bh": rb,
                     "avg_exposure": float(w_prev[seg].mean()), "n_sessions": int(seg.sum()),
                     "protection_pp": rr - rb})
    return pd.DataFrame(rows, columns=["peak_date", "trough_date", "depth", "ret_rule", "ret_bh",
                                       "avg_exposure", "n_sessions", "protection_pp"])


def rolling_relative(ret_rule: pd.Series, ret_bh: pd.Series, L: int) -> dict:
    """길이 L 의 **비겹침** 창에서 규칙 − 보유 누적수익 차: p10/p50/p90 · 뒤진 비율 · 최악 창."""
    L = int(L)
    if L < 2:
        raise ValueError(f"L 은 2 이상이어야 함: {L}")
    a = pd.Series(ret_rule).astype(float)
    b = pd.Series(ret_bh).astype(float)
    common = a.index.intersection(b.index)
    a, b = a.loc[common].dropna(), b.loc[common].dropna()
    common = a.index.intersection(b.index)
    a, b = a.loc[common].to_numpy(), b.loc[common].to_numpy()
    diffs = [float(np.prod(1.0 + a[i:i + L]) - np.prod(1.0 + b[i:i + L]))
             for i in range(0, len(a) - L + 1, L)]
    if not diffs:
        return {"L": L, "n": 0, "p10": np.nan, "p50": np.nan, "p90": np.nan,
                "share_behind": np.nan, "worst": np.nan, "best": np.nan}
    d = np.asarray(diffs, dtype=float)
    q = np.percentile(d, [10, 50, 90])
    return {"L": L, "n": int(len(d)), "p10": float(q[0]), "p50": float(q[1]), "p90": float(q[2]),
            "share_behind": float((d < 0).mean()), "worst": float(d.min()), "best": float(d.max())}


def calendar_year_relative(ret_rule: pd.Series, ret_bh: pd.Series) -> dict:
    """달력연도별 규칙 − 보유 수익 차: 뒤진 해의 비율·중앙값·10/90 분위 (§6.2 사다리 열)."""
    a = pd.Series(ret_rule).astype(float).dropna()
    b = pd.Series(ret_bh).astype(float).dropna()
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    ya = (1.0 + a).groupby(a.index.year).prod() - 1.0
    yb = (1.0 + b).groupby(b.index.year).prod() - 1.0
    d = (ya - yb).dropna()
    if d.empty:
        return {"n_years": 0, "n_behind": 0, "share_behind": np.nan, "median": np.nan,
                "p10": np.nan, "p90": np.nan, "by_year": {}}
    q = np.percentile(d.to_numpy(), [10, 90])
    return {"n_years": int(len(d)), "n_behind": int((d < 0).sum()), "share_behind": float((d < 0).mean()),
            "median": float(d.median()), "p10": float(q[0]), "p90": float(q[1]),
            "by_year": {int(k): float(v) for k, v in d.items()}}
