# -*- coding: utf-8 -*-
"""결정층 v1 — 3단계 상태기계 (ARCHITECTURE_PHASE2.md §9).

상태 3개 ``normal / caution / reduce``. 신호는 ``r_t = p_t / clim_y`` (확률 ÷ 그 해 기후학) — 기저율 배수라
기저율이 표류해도 점유율이 유지된다 (clim≈0.16 이면 절대 확률로 caution≈0.24, reduce≈0.40).

규칙(사전 등록, 적합 파라미터 0 — 상수는 config.DECISION_P2)
* 격상(즉시, dwell 없음): ``normal→caution`` r ≥ enter_caution(1.5) ; 어느 상태에서든 ``→reduce`` r ≥ enter_reduce(2.5).
* 격하(현 상태 체류 ≥ dwell(5)세션일 때만): ``reduce→caution`` r < exit_reduce(2.0) ; ``reduce→normal`` r < exit_caution(1.2) ;
  ``caution→normal`` r < exit_caution(1.2).
* [exit_caution, enter_caution) 안에서 진동해도 caution 은 그대로(히스테리시스). 상태가 바뀌면 days_in_state = 1 부터.
* 체류(dwell) 규약: 어제 기록된 days_in_state(어제까지 그 상태로 보낸 세션 수)가 dwell 이상이면 오늘 격하 가능.
  즉 k 에 진입(days=1)한 상태는 k+4(days=5)까지 반드시 유지되고 k+5 부터 격하 가능 — 최소 5세션 체류.
* 최대 변경 횟수: 하드캡 없음(숨은 파라미터가 된다). 구조적 상한 = 어떤 5세션 창에서도 ≤ 3회(격하 1 + 격상 2).
  KPI 상한 12회/년(사전 등록), churn 경보 = 직전 252세션 변경 > 12 (자동 재조정 없음, 장부 §8 검토 대상).
* 입력 결측(r=None/NaN): 상태 유지, days_in_state 계속 증가, r=NaN 기록, 사유 "확률 계산 불가". 조용히 채우지 않는다.
* 재적합일: 상태 이월(재적합 자체로 격하 불가; dwell 은 그대로 센다) — 이 모듈은 재적합을 모르므로 자동으로 만족.
* 상태 → 톤: config.STATE_TO_TONE → config.TONE_EXPOSURE(100/50/25%) 그대로, 비용 5bp → evaluate.allocation_sim 재사용.
* 민감도(선택에 쓰지 않음, 보고만): wide(1.75/1.25, 3.0/2.25) · symmetric_dwell(격상도 5세션) · no_dwell(dwell 0).

계약
    DecisionConfig(enter_caution, exit_caution, enter_reduce, exit_reduce, dwell, churn_alert[, dwell_escalate])
    step(r, prev_state, days_in_state, cfg) -> (state, days, reason_ko)
    run(p, clim, cfg) -> DataFrame[r, state, days_in_state, changed, reason_ko, churn_252, churn_alert]
    to_tone(states) -> Series 'tone'
    kpis(states_df, y_dd, spy_close) -> dict
"""
from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from mrl import evaluate as E
from mrl.config import (DECISION_P2, DECISION_P2_SENSITIVITY, P2_STATES, STATE_TO_TONE,
                        TONE_EXPOSURE)

__all__ = [
    "DecisionConfig", "RUN_COLUMNS", "CHURN_WINDOW", "STRUCTURAL_MAX_CHANGES_5", "KPI_MAX_SWITCHES_PER_YEAR",
    "SENSITIVITY_NAMES", "WARN_STATES", "STATE_KO", "REASON_MISSING", "V0_TRUE_ALARM_SHARE_REF",
    "config_from_name", "sensitivity_configs", "step", "run", "to_tone", "next_thresholds", "kpis", "kpis_jsonable",
]

# run() 반환 열 (순서 고정 — 스크립트·리포트가 의존)
RUN_COLUMNS = ("r", "state", "days_in_state", "changed", "reason_ko", "churn_252", "churn_alert")
CHURN_WINDOW = 252                       # churn 경보 창(세션)
STRUCTURAL_MAX_CHANGES_5 = 3             # dwell ≥ 1 이면 어떤 5세션 창에서도 변경 ≤ 3 (격하 1 + 격상 2)
KPI_MAX_SWITCHES_PER_YEAR = DECISION_P2["kpi_max_switches_per_year"]
SENSITIVITY_NAMES = ("default",) + tuple(DECISION_P2_SENSITIVITY)      # default, wide, symmetric_dwell, no_dwell
WARN_STATES = ("caution", "reduce")      # 경고 상태 (evaluate.WARN_TONES 와 같은 뜻)
STATE_KO = {"normal": "정상", "caution": "주의", "reduce": "축소"}
REASON_MISSING = "확률 계산 불가 — 상태 유지"
# v0 참조값 (VALIDATION.md §8 #1: v0 경고 런 진짜 경보 비중 10.9~13.2% < 기저율 14.7%). 재계산이 아니라 장부 기록값.
V0_TRUE_ALARM_SHARE_REF = {"lo": 0.109, "hi": 0.132, "source": "VALIDATION.md §8 #1 (v0 faithful/completed 2015~2026)"}
_KPI_STATE_KEYS = tuple(P2_STATES)


# ------------------------------------------------------------------
# 설정
# ------------------------------------------------------------------
@dataclass(frozen=True)
class DecisionConfig:
    """결정층 상수. 기본값 = config.DECISION_P2 (사전 등록). ``dwell_escalate`` 는 민감도 symmetric_dwell 전용(기본 0 = 격상 즉시)."""
    enter_caution: float = DECISION_P2["enter_caution"]
    exit_caution: float = DECISION_P2["exit_caution"]
    enter_reduce: float = DECISION_P2["enter_reduce"]
    exit_reduce: float = DECISION_P2["exit_reduce"]
    dwell: int = DECISION_P2["dwell"]
    churn_alert: int = DECISION_P2["churn_alert"]
    dwell_escalate: int = 0

    def __post_init__(self) -> None:
        for name in ("enter_caution", "exit_caution", "enter_reduce", "exit_reduce"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"{name} 는 양의 유한값이어야 함: {v!r}")
        for name in ("dwell", "churn_alert", "dwell_escalate"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 0:
                raise ValueError(f"{name} 는 0 이상의 정수여야 함: {v!r}")
        if not (self.exit_caution < self.enter_caution):
            raise ValueError(f"히스테리시스 위반: exit_caution({self.exit_caution}) < enter_caution({self.enter_caution}) 이어야 함")
        if not (self.exit_reduce < self.enter_reduce):
            raise ValueError(f"히스테리시스 위반: exit_reduce({self.exit_reduce}) < enter_reduce({self.enter_reduce}) 이어야 함")
        if self.enter_caution > self.enter_reduce:
            raise ValueError("enter_caution 은 enter_reduce 이하여야 함")
        if self.exit_caution > self.exit_reduce:
            raise ValueError("exit_caution 은 exit_reduce 이하여야 함")

    def as_dict(self) -> dict:
        return asdict(self)


def config_from_name(name: str = "default") -> DecisionConfig:
    """'default' 또는 config.DECISION_P2_SENSITIVITY 의 이름(wide / symmetric_dwell / no_dwell) → DecisionConfig."""
    if name == "default":
        return DecisionConfig()
    if name not in DECISION_P2_SENSITIVITY:
        raise ValueError(f"알 수 없는 결정층 구성: {name!r} (허용: {SENSITIVITY_NAMES})")
    return replace(DecisionConfig(), **DECISION_P2_SENSITIVITY[name])


def sensitivity_configs() -> dict[str, DecisionConfig]:
    """이름 → DecisionConfig (default 포함, SENSITIVITY_NAMES 순서)."""
    return {n: config_from_name(n) for n in SENSITIVITY_NAMES}


# ------------------------------------------------------------------
# 한 걸음
# ------------------------------------------------------------------
def _r_value(r) -> float | None:
    """r → float, 결측(None/NaN)은 None. 음수·무한은 ValueError (r = p/clim 은 0 이상이어야 한다)."""
    if r is None:
        return None
    if isinstance(r, bool):
        raise TypeError("r 는 bool 일 수 없음")
    try:
        f = float(r)
    except (TypeError, ValueError) as e:
        raise TypeError(f"r 를 float 로 해석 불가: {r!r}") from e
    if math.isnan(f):
        return None
    if math.isinf(f) or f < 0:
        raise ValueError(f"r 는 0 이상의 유한값이어야 함: {r!r}")
    return f


def step(r, prev_state: str, days_in_state: int, cfg: DecisionConfig = DecisionConfig()) -> tuple[str, int, str]:
    """상태기계 한 걸음. 반환 (state, days_in_state, reason_ko).

    r: 오늘의 p/clim (None/NaN = 확률 계산 불가 → 상태 유지·days 증가).
    prev_state / days_in_state: 어제 기록(장부)의 상태와 체류 세션 수. 최초는 ("normal", 0).
    격상은 즉시(cfg.dwell_escalate=0), 격하는 days_in_state ≥ cfg.dwell 일 때만. 상태가 바뀌면 days=1.
    """
    if prev_state not in P2_STATES:
        raise ValueError(f"알 수 없는 상태: {prev_state!r} (허용: {P2_STATES})")
    if isinstance(days_in_state, bool) or not isinstance(days_in_state, (int, np.integer)):
        raise TypeError(f"days_in_state 는 정수여야 함: {days_in_state!r}")
    days = int(days_in_state)
    if days < 0:
        raise ValueError(f"days_in_state 는 0 이상이어야 함: {days}")
    rv = _r_value(r)
    if rv is None:
        return prev_state, days + 1, REASON_MISSING
    ko = STATE_KO[prev_state]

    # ---- 격상 (기본 즉시; symmetric_dwell 민감도에서만 체류 조건) ----
    esc_target = None
    if prev_state != "reduce" and rv >= cfg.enter_reduce:
        esc_target, esc_thr = "reduce", cfg.enter_reduce
    elif prev_state == "normal" and rv >= cfg.enter_caution:
        esc_target, esc_thr = "caution", cfg.enter_caution
    if esc_target is not None:
        if days >= cfg.dwell_escalate:
            return esc_target, 1, f"격상 {ko}→{STATE_KO[esc_target]}: r={rv:.2f} ≥ {esc_thr:g}"
        return prev_state, days + 1, (f"유지 {ko}: 격상 조건(r={rv:.2f} ≥ {esc_thr:g})이나 "
                                      f"격상 체류 미달({days}/{cfg.dwell_escalate}, 민감도)")

    # ---- 격하 (체류 ≥ dwell 일 때만) ----
    de_target = None
    if prev_state == "reduce":
        if rv < cfg.exit_caution:
            de_target, de_thr = "normal", cfg.exit_caution
        elif rv < cfg.exit_reduce:
            de_target, de_thr = "caution", cfg.exit_reduce
    elif prev_state == "caution":
        if rv < cfg.exit_caution:
            de_target, de_thr = "normal", cfg.exit_caution
    if de_target is not None:
        if days >= cfg.dwell:
            return de_target, 1, f"격하 {ko}→{STATE_KO[de_target]}: r={rv:.2f} < {de_thr:g} (체류 {days}세션 ≥ {cfg.dwell})"
        return prev_state, days + 1, (f"유지 {ko}: 격하 조건(r={rv:.2f} < {de_thr:g})이나 "
                                      f"체류 미달({days}/{cfg.dwell})")

    # ---- 유지 ----
    if prev_state == "normal":
        why = f"r={rv:.2f} < {cfg.enter_caution:g}"
    elif prev_state == "caution":
        why = f"r={rv:.2f} ∈ [{cfg.exit_caution:g}, {cfg.enter_reduce:g}) 밴드 안(히스테리시스)"
    else:
        why = f"r={rv:.2f} ≥ {cfg.exit_reduce:g}"
    return prev_state, days + 1, f"유지 {ko}: {why}"


def next_thresholds(state: str, days_in_state: int, clim: float, cfg: DecisionConfig = DecisionConfig()) -> dict:
    """카드용: 현 상태에서 다음 격상·격하 임계를 r 과 확률(=r·clim)로 환산. 격하는 남은 체류 세션도 함께.

    반환 {escalate_to, r_escalate, p_escalate, deescalate_to, r_deescalate, p_deescalate, dwell_remaining}
    (해당 없음은 None). clim 이 결측이면 p_* 는 NaN.
    """
    if state not in P2_STATES:
        raise ValueError(f"알 수 없는 상태: {state!r}")
    c = float(clim) if clim is not None and not (isinstance(clim, float) and math.isnan(clim)) else np.nan
    if not math.isnan(c) and c <= 0:
        raise ValueError(f"clim 은 양수여야 함: {clim!r}")
    esc = {"normal": ("caution", cfg.enter_caution), "caution": ("reduce", cfg.enter_reduce), "reduce": (None, None)}[state]
    de = {"normal": (None, None), "caution": ("normal", cfg.exit_caution), "reduce": ("caution", cfg.exit_reduce)}[state]
    remaining = max(0, cfg.dwell - int(days_in_state)) if de[0] is not None else None
    return {
        "state": state,
        "escalate_to": esc[0], "r_escalate": esc[1], "p_escalate": (esc[1] * c) if esc[1] is not None else None,
        "deescalate_to": de[0], "r_deescalate": de[1], "p_deescalate": (de[1] * c) if de[1] is not None else None,
        "dwell_remaining": remaining,
        "clim": c,
    }


# ------------------------------------------------------------------
# 시계열 실행
# ------------------------------------------------------------------
def _check_index(idx: pd.Index) -> None:
    if idx.has_duplicates:
        dup = list(idx[idx.duplicated()][:3])
        raise ValueError(f"인덱스에 중복이 있음 (예: {dup})")
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is not None:
            raise ValueError("계약 위반: 인덱스는 tz-naive 여야 함")
        if not idx.is_monotonic_increasing:
            raise ValueError("인덱스가 오름차순이 아님")


def run(p, clim, cfg: DecisionConfig = DecisionConfig(), init_state: str = "normal", init_days: int = 0) -> pd.DataFrame:
    """확률 시계열 → 상태 시계열.

    p: 확률 Series(인덱스 = 세션; NaN = 계산 불가). clim: 같은 인덱스의 Series(연중 상수, 재적합마다 갱신) 또는 스칼라.
    init_state/init_days: 장부에서 재개할 때의 직전 상태(기본 normal/0). 이어서 계산한 결과는 처음부터 계산한 것과 같다
    (churn_252 만 창 안의 이력에 의존하므로 재개 구간 앞 252세션은 전체 CSV 로 다시 센다).
    반환 열: r, state, days_in_state, changed, reason_ko, churn_252(직전 252세션 변경 수, 오늘 포함), churn_alert(> cfg.churn_alert).
    .attrs: config, init_state, init_days, n_missing, warnings.
    """
    if init_state not in P2_STATES:
        raise ValueError(f"알 수 없는 초기 상태: {init_state!r}")
    if isinstance(init_days, bool) or not isinstance(init_days, (int, np.integer)) or init_days < 0:
        raise ValueError(f"init_days 는 0 이상의 정수여야 함: {init_days!r}")
    ps = pd.Series(p).astype(float)
    _check_index(ps.index)
    warn_list: list[str] = []
    finite = ps.notna()
    if ((ps[finite] < 0) | (ps[finite] > 1)).any():
        bad = list(ps.index[finite & ((ps < 0) | (ps > 1))][:3])
        raise ValueError(f"p 는 [0,1] 안이어야 함 (예: {bad})")

    if np.isscalar(clim):
        cs = pd.Series(float(clim), index=ps.index)
    else:
        cs = pd.Series(clim).astype(float)
        if not cs.index.equals(ps.index):
            missing = ps.index.difference(cs.index)
            if len(missing) and finite.loc[missing].any():
                warn_list.append(f"clim 이 없는 세션 {int(finite.loc[missing].sum())}행 → r=NaN(확률 계산 불가로 처리)")
            cs = cs.reindex(ps.index)
    bad_clim = cs.isna() | ~np.isfinite(cs) | (cs <= 0)
    n_bad_clim = int((bad_clim & finite).sum())
    if n_bad_clim:
        msg = f"clim 결측/비양수 {n_bad_clim}행 → r=NaN(확률 계산 불가로 처리)"
        if msg not in warn_list:
            warn_list.append(msg)
        warnings.warn(msg, stacklevel=2)
    r = (ps / cs).where(~bad_clim)
    r_arr = r.to_numpy(dtype=float)

    n = len(ps)
    states = np.empty(n, dtype=object)
    days_arr = np.zeros(n, dtype=int)
    changed = np.zeros(n, dtype=bool)
    reasons = np.empty(n, dtype=object)
    st, d = init_state, int(init_days)
    for i in range(n):
        rv = r_arr[i]
        new_st, new_d, why = step(None if np.isnan(rv) else float(rv), st, d, cfg)
        changed[i] = new_st != st
        st, d = new_st, new_d
        states[i], days_arr[i], reasons[i] = st, d, why

    out = pd.DataFrame(index=ps.index)
    out["r"] = r_arr
    out["state"] = states.astype(str) if n else pd.Series(dtype=str)
    out["days_in_state"] = days_arr
    out["changed"] = changed
    out["reason_ko"] = reasons.astype(str) if n else pd.Series(dtype=str)
    churn = pd.Series(changed.astype(int), index=ps.index).rolling(CHURN_WINDOW, min_periods=1).sum()
    out["churn_252"] = churn.fillna(0).astype(int)
    out["churn_alert"] = out["churn_252"] > cfg.churn_alert
    out = out[list(RUN_COLUMNS)]
    n_missing = int(np.isnan(r_arr).sum())
    if n_missing:
        warn_list.append(f"확률 계산 불가 {n_missing}세션 (상태 유지·days 증가)")
    if int(out["churn_alert"].sum()):
        warn_list.append(f"결정층 잦은 전환(churn 경보) {int(out['churn_alert'].sum())}세션 — 직전 {CHURN_WINDOW}세션 변경 > {cfg.churn_alert}")
    out.attrs = {"config": cfg.as_dict(), "init_state": init_state, "init_days": int(init_days),
                 "n_missing": n_missing, "warnings": warn_list}
    return out


def to_tone(states) -> pd.Series:
    """상태 → 톤(config.STATE_TO_TONE). evaluate.* 가 받는 'tone' 열. 알 수 없는 상태는 ValueError."""
    s = pd.Series(states)
    if s.isna().any():
        raise ValueError("상태에 결측이 있음 (결측일에도 상태는 유지되어야 한다)")
    s = s.astype(str)
    unknown = sorted(set(s.unique()) - set(STATE_TO_TONE))
    if unknown:
        raise ValueError(f"알 수 없는 상태: {unknown} (허용: {P2_STATES})")
    out = s.map(STATE_TO_TONE)
    out.name = "tone"
    return out


# ------------------------------------------------------------------
# KPI
# ------------------------------------------------------------------
def _median_or_nan(vals) -> float:
    vals = list(vals)
    return float(np.median(vals)) if vals else np.nan


def kpis(states_df: pd.DataFrame, y_dd: pd.Series, spy_close, v0_ref: dict | None = None) -> dict:
    """결정층 KPI (사전 등록 상한과 함께).

    states_df: run() 의 반환(열 state, changed 필수; r·churn_252 있으면 사용). y_dd: y_dd5_20 (0/1/NaN, 인덱스로 정렬).
    spy_close: 배분 시뮬용 종가(None 이면 allocation=None + 경고).
    반환 키: n_sessions, years(세션/252), n_changes, switches_per_year, kpi_ceiling(12), kpi_ceiling_ok,
      occupancy{normal,caution,reduce}, n_by_state, dd5_rate_by_state, base_rate,
      n_warn_runs, median_warn_run(경고 런 중앙 길이), median_run_by_state, warn_share,
      true_alarm_share(경고 런 시작일의 y_dd5_20 비율 = evaluate._true_alarm_share), true_alarm_share_baseline(=base_rate),
      true_alarm_edge, true_alarm_share_v0_ref(10.9~13.2%, VALIDATION §8 #1),
      max_changes_any_5_sessions, structural_bound(3 | None), structural_bound_ok,
      churn_alert_sessions, churn_alert_any, max_churn_252, n_missing, missing_share,
      allocation(evaluate.allocation_sim 전체; 'series' 는 pandas — JSON 은 kpis_jsonable), warnings.
    """
    for c in ("state", "changed"):
        if c not in states_df.columns:
            raise ValueError(f"states_df 에 '{c}' 열이 없음")
    df = states_df
    _check_index(df.index)
    n = int(len(df))
    warn_list: list[str] = []
    state = df["state"].astype(str)
    unknown = sorted(set(state.unique()) - set(P2_STATES))
    if unknown:
        raise ValueError(f"알 수 없는 상태: {unknown}")
    changed = df["changed"].astype(bool)
    n_changes = int(changed.sum())
    years = n / E.TRADING_DAYS
    y = pd.Series(y_dd).astype(float)
    y = y.reindex(df.index) if not y.index.equals(df.index) else y
    yv = y.to_numpy(dtype=float)
    if not np.isin(yv[~np.isnan(yv)], [0.0, 1.0]).all():
        raise ValueError("y_dd 는 0/1/NaN 이어야 함")
    if n and y.isna().all():
        warn_list.append("y_dd 가 전부 결측 — 상태별 dd5 비율·진짜 경보 비중은 NaN")

    occupancy = {s: (float((state == s).mean()) if n else np.nan) for s in _KPI_STATE_KEYS}
    n_by_state = {s: int((state == s).sum()) for s in _KPI_STATE_KEYS}
    dd5_by_state = {}
    for s in _KPI_STATE_KEYS:
        sub = y[(state == s).to_numpy()].dropna()
        dd5_by_state[s] = float(sub.mean()) if len(sub) else np.nan
    base_rate = float(y.dropna().mean()) if y.notna().any() else np.nan

    is_warn = state.isin(WARN_STATES).to_numpy()
    warn_runs = E._warning_runs(pd.Series(is_warn))
    runs_all = E._runs(state)
    run_by_state = {s: _median_or_nan(b - a + 1 for a, b, v in runs_all if v == s) for s in _KPI_STATE_KEYS}
    true_alarm = E._true_alarm_share(is_warn, yv) if n else np.nan
    max5 = int(changed.astype(int).rolling(5, min_periods=1).sum().max()) if n else 0
    cfg = dict(df.attrs.get("config", {}))
    dwell = cfg.get("dwell", DECISION_P2["dwell"])
    structural_bound = STRUCTURAL_MAX_CHANGES_5 if dwell >= 1 else None
    if "churn_252" in df.columns:
        churn = df["churn_252"].astype(int)
    else:
        churn = changed.astype(int).rolling(CHURN_WINDOW, min_periods=1).sum().fillna(0).astype(int)
    churn_thr = cfg.get("churn_alert", DECISION_P2["churn_alert"])
    churn_alert = (churn > churn_thr) if "churn_alert" not in df.columns else df["churn_alert"].astype(bool)
    n_missing = int(df["r"].isna().sum()) if "r" in df.columns else 0

    allocation = None
    if spy_close is None:
        warn_list.append("spy_close 없음 — 배분 시뮬 생략")
    elif n >= 2:
        tone_df = pd.DataFrame({"tone": to_tone(state)}, index=df.index)
        allocation = E.allocation_sim(tone_df, spy_close, exposure=TONE_EXPOSURE, cost_bps=5)
    else:
        warn_list.append("세션 2개 미만 — 배분 시뮬 생략")

    switches_per_year = float(n_changes / years) if years > 0 else np.nan
    ref = dict(V0_TRUE_ALARM_SHARE_REF) if v0_ref is None else dict(v0_ref)
    return {
        "n_sessions": n,
        "years": float(years),
        "start": df.index[0] if n else None,
        "end": df.index[-1] if n else None,
        "n_changes": n_changes,
        "switches_per_year": switches_per_year,
        "kpi_ceiling": int(KPI_MAX_SWITCHES_PER_YEAR),
        "kpi_ceiling_ok": (bool(switches_per_year <= KPI_MAX_SWITCHES_PER_YEAR) if not math.isnan(switches_per_year) else None),
        "occupancy": occupancy,
        "n_by_state": n_by_state,
        "dd5_rate_by_state": dd5_by_state,
        "base_rate": base_rate,
        "warn_share": float(is_warn.mean()) if n else np.nan,
        "n_warn_runs": int(len(warn_runs)),
        "median_warn_run": _median_or_nan(b - a + 1 for a, b in warn_runs),
        "median_run_by_state": run_by_state,
        "true_alarm_share": true_alarm,
        "true_alarm_share_baseline": base_rate,
        "true_alarm_edge": (true_alarm - base_rate) if not (math.isnan(true_alarm) or math.isnan(base_rate)) else np.nan,
        "true_alarm_share_v0_ref": ref,
        "max_changes_any_5_sessions": max5,
        "structural_bound": structural_bound,
        "structural_bound_ok": (bool(max5 <= structural_bound) if structural_bound is not None else None),
        "churn_window": CHURN_WINDOW,
        "churn_alert_threshold": int(churn_thr),
        "churn_alert_sessions": int(churn_alert.sum()),
        "churn_alert_any": bool(churn_alert.any()),
        "max_churn_252": int(churn.max()) if n else 0,
        "n_missing": n_missing,
        "missing_share": float(n_missing / n) if n else np.nan,
        "config": cfg,
        "allocation": allocation,
        "warnings": warn_list,
    }


def kpis_jsonable(k: dict) -> dict:
    """kpis() → JSON 직렬화 가능 dict (allocation.series 제거, NaN → None)."""
    out = dict(k)
    alloc = out.get("allocation")
    if isinstance(alloc, dict):
        out["allocation"] = {kk: vv for kk, vv in alloc.items() if kk != "series"}
    return E._jsonable(out)
