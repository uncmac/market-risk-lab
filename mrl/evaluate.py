# -*- coding: utf-8 -*-
"""v0 재현 결과의 정직한 평가 (VALIDATION.md §5, ARCHITECTURE.md 계약).

계약:
    directional_scorecard(replay, targets) -> DataFrame
    episode_eval(replay, episodes_df, targets) -> (DataFrame, dict)
    allocation_sim(replay, spy_close, exposure=TONE_EXPOSURE, cost_bps=5) -> dict
    brier(prob, y) -> float ; brier_skill(prob, y, ref_prob) -> float
    block_bootstrap_ci(values, block, n_boot=2000, ci=0.95, seed=0) -> (lo, hi)
    summarize_v0(replay, targets, episodes5, episodes10, spy_close) -> dict (JSON 직렬화 가능)

설계 원칙
* replay 는 거래일 인덱스 + 'tone' 열만 있으면 평가 가능(다른 열은 meta 용으로만 읽는다).
* 방향 적중 규약: buy/hold/neutral = 상승(y_sign=1) 예측, caution/reduce = 비상승(0) 예측.
  기준선 = 같은 행들의 mean(y_sign_h) (항상-상승 적중률).
* 점(point-in-time) 원칙: 배분 시뮬은 t 종가에 정해진 톤을 t+1 수익률에 적용한다.
* 조용한 실패 금지: 알 수 없는 톤·정렬 불가 인덱스는 ValueError, 부분 결측은 warnings.warn.
"""
from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd

from mrl.config import HORIZONS, TONES, TONE_EXPOSURE
from mrl.targets import Y_COLUMNS, base_rates, independent_blocks

UP_TONES = ("buy", "hold", "neutral")          # 상승 예측으로 간주하는 톤
WARN_TONES = ("caution", "reduce")             # 경고(비상승 예측) 톤
TRADING_DAYS = 252
DEFAULT_LOOKBACK = 20                          # 에피소드 고점 이전 경고 탐색 창(거래일)
NULL_N_SHIFT = 200                             # 무작위 순환 이동(null) 표본 수 — 경고 비중·런 길이를 보존한 채 정보만 없앤다
NULL_SEED = 0

SCORECARD_COLUMNS = ("tone", "h", "n", "n_blocks", "hit", "hit_ci_lo", "hit_ci_hi", "baseline", "edge",
                     "fwd_ret_mean", "fwd_ret_median", "fwd_ret_p10", "fwd_ret_p90", "dd5_20_rate")
EPISODE_EVAL_COLUMNS = ("peak_date", "trough_date", "recovery_date", "depth", "days_to_trough", "days_to_recover",
                        "evaluable", "warn_date", "lead_days", "warn_run_start", "warn_run_age_at_peak",
                        "lead_capped", "held_to_trough", "warn_frac_to_trough",
                        "missed", "tone_at_peak", "tone_at_trough")


# ------------------------------------------------------------------
# 공통 유틸
# ------------------------------------------------------------------
def _check_tones(tone: pd.Series) -> pd.Series:
    """톤 열 검증: 문자열, TONES 안의 값만. 결측/미지 톤은 ValueError."""
    if tone.isna().any():
        bad = tone.index[tone.isna()][:3]
        raise ValueError(f"replay.tone 에 결측이 있습니다 (예: {list(bad)})")
    t = tone.astype(str)
    unknown = sorted(set(t.unique()) - set(TONES))
    if unknown:
        raise ValueError(f"알 수 없는 톤: {unknown} (허용: {TONES})")
    return t


def _replay_tone(replay: pd.DataFrame) -> pd.Series:
    if "tone" not in replay.columns:
        raise ValueError("replay 에 'tone' 열이 없습니다")
    if not isinstance(replay.index, pd.DatetimeIndex):
        raise TypeError("replay 인덱스는 DatetimeIndex(거래일)여야 합니다")
    if replay.index.has_duplicates or not replay.index.is_monotonic_increasing:
        raise ValueError("replay 인덱스는 중복 없는 오름차순이어야 합니다")
    return _check_tones(replay["tone"])


def _greedy_blocks(positions: np.ndarray, h: int) -> int:
    """정렬된 정수 위치(거래일 순번)에서 서로 겹치지 않는 h 일 앞창을 탐욕적으로 고른 개수.
    연속된 n 일이면 ceil(n/h), 흩어져 있으면 그만큼 더 많다(겹침이 적으므로)."""
    if len(positions) == 0:
        return 0
    count, next_free = 0, -1
    for p in positions:
        if p >= next_free:
            count += 1
            next_free = p + h
    return int(count)


def _runs(values: pd.Series) -> list[tuple[int, int, Any]]:
    """연속 동일값 런 → [(start_pos, end_pos(포함), value)]."""
    out: list[tuple[int, int, Any]] = []
    if len(values) == 0:
        return out
    arr = values.to_numpy()
    start = 0
    for i in range(1, len(arr) + 1):
        if i == len(arr) or arr[i] != arr[start]:
            out.append((start, i - 1, arr[start]))
            start = i
    return out


def _warning_runs(is_warn: pd.Series) -> list[tuple[int, int]]:
    """경고(True) 런의 (start_pos, end_pos) 목록."""
    return [(a, b) for a, b, v in _runs(is_warn) if bool(v)]


def _nan_to_none(x):
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def _episode_hits(is_warn: np.ndarray, windows: list[tuple[int, int, int]]) -> tuple[int, list[int]]:
    """경고 bool 배열과 에피소드 창 [(w_start, p_peak, p_trough)] → (탐지 수, 탐지된 에피소드의 lead_days 목록).
    episode_eval 의 '창 안 첫 경고' 규칙과 동일 — 실제 경고에도, 무작위 이동한 경고에도 같은 함수를 쓴다."""
    n_det, leads = 0, []
    for w_start, p_peak, p_trough in windows:
        hits = np.flatnonzero(is_warn[w_start: p_trough + 1])
        if len(hits):
            n_det += 1
            leads.append(int(p_peak - (w_start + int(hits[0]))))
    return n_det, leads


def _true_alarm_share(is_warn: np.ndarray, y_dd: np.ndarray) -> float:
    """경고 런 시작일 기준 y_dd5_20 의 비율(진짜 경보 비중). 판정 불가(NaN) 런은 제외. 런이 없으면 NaN."""
    runs = _warning_runs(pd.Series(is_warn))
    vals = [y_dd[a] for a, _b in runs if not np.isnan(y_dd[a])]
    return float(np.mean(vals)) if vals else np.nan


def _null_shift(is_warn: np.ndarray, windows: list[tuple[int, int, int]], y_dd: np.ndarray,
                n_shift: int = NULL_N_SHIFT, seed: int = NULL_SEED) -> dict:
    """정보 없는 기준선: 경고 시계열을 무작위 오프셋으로 **순환 이동**(np.roll)해 같은 채점을 반복한다.

    경고 비중과 런 길이 분포는 그대로 두고 시점 정보만 없앤 것이므로, 실제 탐지율·리드타임·진짜 경보 비중이
    이 분포와 구별되지 않으면 "lookback 창 × 경고 비중" 의 산물일 뿐이다 (VALIDATION.md §5: 항상 기준선 옆에).
    반환: n_shift, detection_rate_mean, detection_100_share(탐지율 100% 인 이동의 비율), median_lead_days_p5/p50/p95
          (이동별 중앙 리드의 분위), true_alarm_share_mean. 평가 가능 에피소드가 없으면 탐지 관련은 NaN.
    """
    n = len(is_warn)
    n_ep = len(windows)
    if n < 2 or n_shift <= 0:
        return {"n_shift": 0, "detection_rate_mean": np.nan, "detection_100_share": np.nan,
                "median_lead_days_p5": np.nan, "median_lead_days_p50": np.nan, "median_lead_days_p95": np.nan,
                "true_alarm_share_mean": np.nan}
    rng = np.random.default_rng(seed)
    offsets = rng.integers(1, n, size=int(n_shift))
    det_rates, med_leads, ta_shares = [], [], []
    for k in offsets:
        rolled = np.roll(is_warn, int(k))
        if n_ep:
            n_det, leads = _episode_hits(rolled, windows)
            det_rates.append(n_det / n_ep)
            if leads:
                med_leads.append(float(np.median(leads)))
        ta = _true_alarm_share(rolled, y_dd)
        if not np.isnan(ta):
            ta_shares.append(ta)
    q = (lambda p: float(np.quantile(med_leads, p))) if med_leads else (lambda p: np.nan)
    return {
        "n_shift": int(n_shift),
        "detection_rate_mean": float(np.mean(det_rates)) if det_rates else np.nan,
        "detection_100_share": float(np.mean([r >= 1.0 for r in det_rates])) if det_rates else np.nan,
        "median_lead_days_p5": q(0.05), "median_lead_days_p50": q(0.50), "median_lead_days_p95": q(0.95),
        "true_alarm_share_mean": float(np.mean(ta_shares)) if ta_shares else np.nan,
    }


def _jsonable(obj: Any) -> Any:
    """numpy/pandas 형을 JSON 직렬화 가능한 파이썬 기본형으로 재귀 변환. NaN/NaT → None."""
    if obj is None:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        ts = pd.Timestamp(obj)
        return None if pd.isna(ts) else ts.strftime("%Y-%m-%d")
    if isinstance(obj, pd.Timedelta):
        return obj.days
    if obj is pd.NaT:
        return None
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, pd.Series):
        return {str(_jsonable(k)): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return [_jsonable(r) for r in obj.to_dict(orient="records")]
    if isinstance(obj, (list, tuple, set, np.ndarray, pd.Index)):
        return [_jsonable(v) for v in list(obj)]
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return str(obj)


# ------------------------------------------------------------------
# 1. 방향 성적표
# ------------------------------------------------------------------
def directional_scorecard(replay: pd.DataFrame, targets: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    """톤별 × 지평(5/20/60) 방향 성적표.

    행: TONES 순서의 각 톤 + 'all'(전체) × h ∈ HORIZONS. 톤이 한 번도 나오지 않아도 n=0 행을 둔다.
    열:
      n           : y_sign_h 가 실현된 행 수
      n_blocks    : 그 행들 가운데 서로 겹치지 않는 h 일 앞창의 수(탐욕 선택; 독립 표본 수의 근사)
      hit         : 적중률 — 톤 ∈ {buy,hold,neutral} 이면 y_sign_h=1 예측, {caution,reduce} 이면 0 예측
      hit_ci_lo/hi: 적중 지표의 블록 부트스트랩 95% 구간(블록 = h)
      baseline    : 같은 행들의 mean(y_sign_h) = 항상-상승 예측의 적중률
      edge        : hit - baseline
      fwd_ret_*   : fwd_ret_h 의 평균·중앙·10/90 분위
      dd5_20_rate : 같은 행들의 mean(y_dd5_20) (NaN 제외)
    """
    tone = _replay_tone(replay)
    need = [f"y_sign_{h}" for h in HORIZONS] + [f"fwd_ret_{h}" for h in HORIZONS] + ["y_dd5_20"]
    missing = [c for c in need if c not in targets.columns]
    if missing:
        raise ValueError(f"targets 에 열이 없습니다: {missing}")

    common = tone.index.intersection(targets.index)
    n_unmatched = len(tone.index) - len(common)
    if n_unmatched:
        warnings.warn(f"replay {n_unmatched}행이 targets 인덱스에 없어 성적표에서 제외됩니다", stacklevel=2)
    if len(common) == 0:
        raise ValueError("replay 와 targets 의 공통 거래일이 없습니다")

    df = targets.loc[common, need].copy()
    df["tone"] = tone.loc[common]
    df["pos"] = np.arange(len(df))              # 거래일 순번 (블록 계산용)
    df["pred_up"] = df["tone"].isin(UP_TONES).astype(float)

    rows = []
    for tn in list(TONES) + ["all"]:
        sub_t = df if tn == "all" else df[df["tone"] == tn]
        for h in HORIZONS:
            ycol, rcol = f"y_sign_{h}", f"fwd_ret_{h}"
            sub = sub_t[sub_t[ycol].notna()]
            n = int(len(sub))
            row: dict[str, Any] = {"tone": tn, "h": int(h), "n": n,
                                   "n_blocks": _greedy_blocks(sub["pos"].to_numpy(), h)}
            if n == 0:
                row.update({k: np.nan for k in SCORECARD_COLUMNS if k not in row})
                rows.append(row)
                continue
            hit_ind = (sub["pred_up"] == sub[ycol]).astype(float)
            lo, hi = block_bootstrap_ci(hit_ind, block=h, n_boot=n_boot, ci=0.95, seed=0)
            r = sub[rcol].dropna()
            row.update({
                "hit": float(hit_ind.mean()),
                "hit_ci_lo": lo, "hit_ci_hi": hi,
                "baseline": float(sub[ycol].mean()),
                "fwd_ret_mean": float(r.mean()) if len(r) else np.nan,
                "fwd_ret_median": float(r.median()) if len(r) else np.nan,
                "fwd_ret_p10": float(r.quantile(0.10)) if len(r) else np.nan,
                "fwd_ret_p90": float(r.quantile(0.90)) if len(r) else np.nan,
                "dd5_20_rate": float(sub["y_dd5_20"].mean()) if sub["y_dd5_20"].notna().any() else np.nan,
            })
            row["edge"] = row["hit"] - row["baseline"]
            rows.append(row)
    out = pd.DataFrame(rows, columns=list(SCORECARD_COLUMNS))
    out["h"] = out["h"].astype(int)
    out["n"] = out["n"].astype(int)
    out["n_blocks"] = out["n_blocks"].astype(int)
    return out


# ------------------------------------------------------------------
# 2. 에피소드 평가
# ------------------------------------------------------------------
def episode_eval(replay: pd.DataFrame, episodes_df: pd.DataFrame, targets: pd.DataFrame,
                 lookback: int = DEFAULT_LOOKBACK) -> tuple[pd.DataFrame, dict]:
    """에피소드별 v0 경고 성적 + 경고 행태 요약.

    에피소드 표 열 (episodes_df 의 열을 그대로 두고 추가):
      evaluable          : 고점일이 replay 구간 안에 있어 평가 가능한가
      warn_date          : [peak - lookback, trough] 안에서 첫 caution/reduce 톤 날짜 (없으면 NaT)
      lead_days          : pos(peak) - pos(warn) ; 양수 = 고점 전 경고, 음수 = 고점 후 경고 (최대 lookback)
      warn_run_start     : 그 경고가 속한 런의 실제 시작일(창 밖까지 거슬러) — "이미 켜져 있던 경고"를 드러낸다
      warn_run_age_at_peak: 고점일 기준 경고가 켜져 있던 거래일 수(런이 고점 뒤에 시작했으면 NaN)
      lead_capped        : 경고 런이 탐색 창 시작(peak - lookback) 이전 또는 정확히 그 날부터 켜져 있어 lead_days 가
                           lookback 상한에 걸린 경우 True (탐색 창은 고점 전 최대 lookback 거래일까지만 본다)
      held_to_trough     : warn_date 부터 trough 까지 매일 경고 톤이 유지됐는가
      warn_frac_to_trough: [warn_date, trough] 중 경고 톤 비율
      missed             : 평가 가능한데 창 안에 경고가 없음
      tone_at_peak/trough: 고점일·저점일의 톤
    요약 dict:
      n_episodes, n_evaluable, n_detected, n_missed, detection_rate, median_lead_days, mean_lead_days,
      n_lead_positive(고점 전 경고 수), n_lead_capped(이미 켜져 있던 경고 수),
      median_lead_days_fresh(창 안에서 새로 켜진 경고만의 중앙 리드), n_held_to_trough,
      n_warn_runs, n_true_alarms, n_false_alarms, false_alarm_rate, false_alarms_per_year
        (오경보 = 경고 런 시작일 기준 20일 내 -5% 낙폭 없음, 즉 y_dd5_20[start]=0; 실현 전 런은 미판정),
      true_alarm_share(=1-false_alarm_rate), true_alarm_share_baseline(=구간의 y_dd5_20 기저율: 임의의 날 뒤 20일 내
        -5% 가 올 확률), false_alarm_rate_baseline(=1-기저율: 임의의 날을 경고라 불렀을 때의 오경보율),
      null{n_shift, detection_rate_mean, detection_100_share, median_lead_days_p5/p50/p95, true_alarm_share_mean}
        (경고 시계열을 무작위 순환 이동한 정보 없는 기준선 — _null_shift),
      n_tone_switches, tone_switches_per_year, median_run_len(모든 톤 런), median_warn_run_len,
      years(replay 길이/252), lookback
    """
    tone = _replay_tone(replay)
    if lookback < 0:
        raise ValueError("lookback 은 0 이상이어야 합니다")
    for c in ("peak_date", "trough_date"):
        if c not in episodes_df.columns:
            raise ValueError(f"episodes_df 에 '{c}' 열이 없습니다")
    if "y_dd5_20" not in targets.columns:
        raise ValueError("targets 에 'y_dd5_20' 열이 없습니다")

    idx = tone.index
    pos_of = pd.Series(np.arange(len(idx)), index=idx)
    is_warn = tone.isin(WARN_TONES)
    r_start, r_end = idx[0], idx[-1]

    recs = []
    windows: list[tuple[int, int, int]] = []                # 평가 가능 에피소드의 (w_start, p_peak, p_trough) — null 용
    for _, ep in episodes_df.iterrows():
        peak, trough = pd.Timestamp(ep["peak_date"]), pd.Timestamp(ep["trough_date"])
        rec: dict[str, Any] = {c: ep[c] for c in episodes_df.columns}
        rec.update({"evaluable": False, "warn_date": pd.NaT, "lead_days": np.nan, "warn_run_start": pd.NaT,
                    "warn_run_age_at_peak": np.nan, "lead_capped": np.nan, "held_to_trough": np.nan,
                    "warn_frac_to_trough": np.nan, "missed": np.nan, "tone_at_peak": None, "tone_at_trough": None})
        if not (r_start <= peak <= r_end):
            recs.append(rec)
            continue
        rec["evaluable"] = True
        p_peak = int(idx.searchsorted(peak))                 # peak 는 인덱스에 있어야 정상 (없으면 다음 거래일)
        if idx[p_peak] != peak:
            warnings.warn(f"에피소드 고점일 {peak.date()} 이 replay 인덱스에 없습니다 — 다음 거래일로 대체", stacklevel=2)
        p_trough = int(min(idx.searchsorted(trough, side="right") - 1, len(idx) - 1))
        p_trough = max(p_trough, p_peak)
        w_start = max(p_peak - lookback, 0)
        windows.append((w_start, p_peak, p_trough))
        rec["tone_at_peak"] = str(tone.iloc[p_peak])
        rec["tone_at_trough"] = str(tone.iloc[p_trough]) if idx[p_trough] <= trough else None
        window = is_warn.iloc[w_start: p_trough + 1]
        hits = np.flatnonzero(window.to_numpy())
        if len(hits) == 0:
            rec["missed"] = True
            recs.append(rec)
            continue
        p_warn = w_start + int(hits[0])
        rec["missed"] = False
        rec["warn_date"] = idx[p_warn]
        rec["lead_days"] = int(p_peak - p_warn)
        # 경고 런의 실제 시작(창 밖까지 거슬러): 고점 훨씬 전부터 켜져 있던 경고를 리드타임으로 오해하지 않도록
        p_run_start = p_warn
        while p_run_start > 0 and bool(is_warn.iloc[p_run_start - 1]):
            p_run_start -= 1
        rec["warn_run_start"] = idx[p_run_start]
        rec["warn_run_age_at_peak"] = int(p_peak - p_run_start) if p_run_start <= p_peak else np.nan
        rec["lead_capped"] = bool(p_run_start <= w_start)   # 창 첫날에 이미 켜져 있던 경고도 상한(lookback)에 걸린 것
        seg = is_warn.iloc[p_warn: p_trough + 1]
        rec["held_to_trough"] = bool(seg.all())
        rec["warn_frac_to_trough"] = float(seg.mean())
        recs.append(rec)

    cols = list(episodes_df.columns) + [c for c in EPISODE_EVAL_COLUMNS if c not in episodes_df.columns]
    table = pd.DataFrame(recs, columns=cols) if recs else pd.DataFrame(columns=cols)

    # 요약
    n_days = len(idx)
    years = n_days / TRADING_DAYS
    ev = table[table["evaluable"] == True] if len(table) else table        # noqa: E712
    det = ev[ev["missed"] == False] if len(ev) else ev                     # noqa: E712
    leads = det["lead_days"].dropna().astype(float) if len(det) else pd.Series(dtype=float)
    fresh = det[det["lead_capped"] == False] if len(det) else det                   # noqa: E712
    leads_fresh = fresh["lead_days"].dropna().astype(float) if len(fresh) else pd.Series(dtype=float)

    # 경고 런과 오경보 (런 시작일 기준 y_dd5_20) — 기저율(임의의 날) 과 무작위 이동 기준선을 옆에 둔다
    y_dd = targets["y_dd5_20"].reindex(idx)
    p_dd = float(y_dd.dropna().mean()) if y_dd.notna().any() else np.nan
    warn_arr = is_warn.to_numpy(dtype=bool)
    null = _null_shift(warn_arr, windows, y_dd.to_numpy(dtype=float))
    warn_runs = _warning_runs(is_warn)
    n_true = n_false = n_unresolved = 0
    for a, _b in warn_runs:
        y = y_dd.iloc[a]
        if pd.isna(y):
            n_unresolved += 1
        elif y >= 0.5:
            n_true += 1
        else:
            n_false += 1
    n_scored = n_true + n_false

    all_runs = _runs(tone)
    run_lens = [b - a + 1 for a, b, _v in all_runs]
    warn_lens = [b - a + 1 for a, b in warn_runs]
    n_switch = max(len(all_runs) - 1, 0)

    summary = {
        "n_episodes": int(len(table)),
        "n_evaluable": int(len(ev)),
        "n_detected": int(len(det)),
        "n_missed": int(len(ev) - len(det)),
        "detection_rate": float(len(det) / len(ev)) if len(ev) else np.nan,
        "median_lead_days": float(leads.median()) if len(leads) else np.nan,
        "mean_lead_days": float(leads.mean()) if len(leads) else np.nan,
        "n_lead_positive": int((leads > 0).sum()) if len(leads) else 0,
        "n_lead_capped": int(det["lead_capped"].eq(True).sum()) if len(det) else 0,
        "median_lead_days_fresh": float(leads_fresh.median()) if len(leads_fresh) else np.nan,
        "n_held_to_trough": int(det["held_to_trough"].eq(True).sum()) if len(det) else 0,
        "n_warn_runs": int(len(warn_runs)),
        "n_true_alarms": int(n_true),
        "n_false_alarms": int(n_false),
        "n_unresolved_alarms": int(n_unresolved),
        "false_alarm_rate": float(n_false / n_scored) if n_scored else np.nan,
        "false_alarms_per_year": float(n_false / years) if years > 0 else np.nan,
        "true_alarm_share": float(n_true / n_scored) if n_scored else np.nan,
        "true_alarm_share_baseline": p_dd,
        "false_alarm_rate_baseline": (1.0 - p_dd) if not np.isnan(p_dd) else np.nan,
        "null": null,
        "n_tone_switches": int(n_switch),
        "tone_switches_per_year": float(n_switch / years) if years > 0 else np.nan,
        "median_run_len": float(np.median(run_lens)) if run_lens else np.nan,
        "median_warn_run_len": float(np.median(warn_lens)) if warn_lens else np.nan,
        "warn_share": float(is_warn.mean()) if n_days else np.nan,
        "years": float(years),
        "lookback": int(lookback),
    }
    return table, summary


# ------------------------------------------------------------------
# 3. 배분 시뮬레이션
# ------------------------------------------------------------------
def _perf_stats(daily_ret: pd.Series) -> dict:
    """일별 수익률 → CAGR·MaxDD·최악 월·변동성 등."""
    eq = (1.0 + daily_ret).cumprod()
    n = len(daily_ret)
    years = n / TRADING_DAYS
    final = float(eq.iloc[-1])
    dd = eq / eq.cummax() - 1.0
    monthly = (1.0 + daily_ret).resample("ME").prod() - 1.0
    worst_m = monthly.idxmin() if len(monthly) else pd.NaT
    return {
        "total_return": final - 1.0,
        "cagr": final ** (1.0 / years) - 1.0 if years > 0 and final > 0 else np.nan,
        "ann_vol": float(daily_ret.std(ddof=1) * np.sqrt(TRADING_DAYS)) if n > 1 else np.nan,
        "max_dd": float(dd.min()),
        "max_dd_date": dd.idxmin() if n else pd.NaT,
        "worst_month": float(monthly.min()) if len(monthly) else np.nan,
        "worst_month_label": worst_m.strftime("%Y-%m") if not pd.isna(worst_m) else None,
        "best_month": float(monthly.max()) if len(monthly) else np.nan,
        "n_days": int(n),
        "years": float(years),
        "_equity": eq,
        "_monthly": monthly,
    }


def allocation_sim(replay: pd.DataFrame, spy_close: pd.Series, exposure: dict = TONE_EXPOSURE,
                   cost_bps: float = 5) -> dict:
    """톤 → 주식 비중 배분 시뮬레이션 vs 보유.

    규칙(사전 등록): t 종가에 확정된 톤의 비중 w[t] 를 t+1 수익률에 적용(점 원칙). 매일 목표 비중으로
    리밸런스하되 비용은 목표 비중이 바뀔 때만 |Δw|×cost_bps 로 부과(첫날 진입 비용은 보유·배분 모두 0).
    나머지(1-w)는 현금(수익률 0 가정 — 무위험수익을 더하지 않으므로 배분 쪽에 보수적).
    반환 dict: cagr, max_dd, worst_month, switches_per_year, n_switches, total_return, ann_vol, avg_exposure,
      cost_total, bh_* (보유 동일 지표), excess_cagr(=cagr-bh_cagr), maxdd_improvement(=max_dd-bh_max_dd, 양수면 개선;
      max_dd 는 음수라 배분의 낙폭이 얕을수록 max_dd 가 크고 차이가 양수가 된다),
      start/end/n_days/years, exposure_map, cost_bps,
      series: {'equity','equity_bh','daily_ret','daily_ret_bh','exposure'} (pandas; JSON 요약에선 제거)
    """
    tone = _replay_tone(replay)
    if cost_bps < 0:
        raise ValueError("cost_bps 는 0 이상이어야 합니다")
    unknown = sorted(set(tone.unique()) - set(exposure))
    if unknown:
        raise ValueError(f"exposure 맵에 없는 톤: {unknown}")
    close = spy_close.dropna()
    common = tone.index.intersection(close.index)
    n_unmatched = len(tone.index) - len(common)
    if n_unmatched:
        warnings.warn(f"replay {n_unmatched}행이 spy_close 에 없어 배분 시뮬에서 제외됩니다", stacklevel=2)
    if len(common) < 2:
        raise ValueError("배분 시뮬에 최소 2 거래일이 필요합니다")
    close = close.loc[common].astype(float)
    w = tone.loc[common].map(exposure).astype(float)

    r = close.pct_change()                         # r[t] = close[t]/close[t-1]-1
    w_prev = w.shift(1)                            # t 의 수익률에 적용되는 비중 = t-1 종가의 톤
    dw = (w - w_prev).abs()
    cost = dw * (cost_bps / 1e4)
    ret_strat = ((1.0 + w_prev * r) * (1.0 - cost) - 1.0).iloc[1:]
    ret_bh = r.iloc[1:]
    switched = (dw.iloc[1:] > 0)

    st = _perf_stats(ret_strat)
    bh = _perf_stats(ret_bh)
    years = st["years"]
    n_switch = int(switched.sum())
    out = {
        "start": common[0], "end": common[-1], "n_days": int(len(ret_strat)), "years": float(years),
        "cagr": st["cagr"], "max_dd": st["max_dd"], "max_dd_date": st["max_dd_date"],
        "worst_month": st["worst_month"], "worst_month_label": st["worst_month_label"],
        "total_return": st["total_return"], "ann_vol": st["ann_vol"],
        "n_switches": n_switch, "switches_per_year": float(n_switch / years) if years > 0 else np.nan,
        "avg_exposure": float(w_prev.iloc[1:].mean()),
        "cost_total": float(cost.iloc[1:].sum()),
        "bh_cagr": bh["cagr"], "bh_max_dd": bh["max_dd"], "bh_max_dd_date": bh["max_dd_date"],
        "bh_worst_month": bh["worst_month"], "bh_worst_month_label": bh["worst_month_label"],
        "bh_total_return": bh["total_return"], "bh_ann_vol": bh["ann_vol"],
        "excess_cagr": st["cagr"] - bh["cagr"],
        "maxdd_improvement": st["max_dd"] - bh["max_dd"],      # 둘 다 음수: 배분 -18.9% vs 보유 -33.7% → +14.8pp 개선
        "exposure_map": dict(exposure), "cost_bps": float(cost_bps),
        "series": {"equity": st["_equity"], "equity_bh": bh["_equity"],
                   "daily_ret": ret_strat, "daily_ret_bh": ret_bh, "exposure": w_prev.iloc[1:]},
    }
    return out


# ------------------------------------------------------------------
# 4. Brier · 부트스트랩
# ------------------------------------------------------------------
def _align_prob(prob, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    y = pd.Series(y).astype(float)
    if np.isscalar(prob):
        p = pd.Series(float(prob), index=y.index)
    else:
        p = pd.Series(prob).astype(float)
        p = p.reindex(y.index) if not p.index.equals(y.index) else p
    m = p.notna() & y.notna()
    if m.sum() == 0:
        raise ValueError("prob 와 y 의 공통 유효 행이 없습니다")
    pa, ya = p[m].to_numpy(), y[m].to_numpy()
    if ((pa < 0) | (pa > 1)).any():
        raise ValueError("prob 는 [0,1] 안이어야 합니다")
    if not np.isin(ya, [0.0, 1.0]).all():
        raise ValueError("y 는 0/1 이어야 합니다")
    return pa, ya


def brier(prob, y: pd.Series) -> float:
    """Brier 점수 = mean((p - y)^2). prob 는 Series 또는 스칼라(기후학 확률)."""
    p, yy = _align_prob(prob, y)
    return float(np.mean((p - yy) ** 2))


def brier_skill(prob, y: pd.Series, ref_prob) -> float:
    """Brier skill = 1 - BS(prob)/BS(ref). ref 가 완벽(BS=0)이면 NaN 대신 ValueError."""
    bs_ref = brier(ref_prob, y)
    if bs_ref == 0:
        raise ValueError("기준 확률의 Brier 점수가 0 이라 skill 을 정의할 수 없습니다")
    return float(1.0 - brier(prob, y) / bs_ref)


def block_bootstrap_ci(values: pd.Series, block: int, n_boot: int = 2000, ci: float = 0.95,
                       seed: int = 0) -> tuple[float, float]:
    """순환 블록 부트스트랩(circular block bootstrap)으로 평균의 (lo, hi) 백분위 신뢰구간.

    values 의 순서를 시간순으로 보고 길이 block 의 연속 블록을 복원추출해 이어 붙인다(길이 n 으로 절단).
    블록 시작점은 0..n-1 전체에서 뽑고 끝에서는 앞으로 감아(순환) 모든 관측이 정확히 block 번씩 블록에
    들어가게 한다 — 그래야 부트스트랩 평균의 기대값이 표본 평균과 같다(이동 블록은 양끝을 덜 뽑아 치우친다).
    NaN 은 제거. n == 0 이면 (nan, nan). block > n 이면 block = n (구간이 퇴화한다).
    """
    if block <= 0:
        raise ValueError("block 은 양수여야 합니다")
    if not (0 < ci < 1):
        raise ValueError("ci 는 (0,1) 사이여야 합니다")
    v = pd.Series(values).dropna().to_numpy(dtype=float)
    n = len(v)
    if n == 0:
        return (np.nan, np.nan)
    b = min(int(block), n)
    k = int(math.ceil(n / b))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    chunk = max(1, min(n_boot, int(2_000_000 // max(k * b, 1))))
    offs = np.arange(b)
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        starts = rng.integers(0, n, size=(m, k))
        sel = ((starts[:, :, None] + offs[None, None, :]) % n).reshape(m, k * b)[:, :n]
        means[done: done + m] = v[sel].mean(axis=1)
        done += m
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return (float(lo), float(hi))


# ------------------------------------------------------------------
# 5. 리포트용 단일 요약
# ------------------------------------------------------------------
def summarize_v0(replay: pd.DataFrame, targets: pd.DataFrame, episodes5: pd.DataFrame,
                 episodes10: pd.DataFrame, spy_close: pd.Series) -> dict:
    """리포트가 쓰는 단일 dict (JSON 직렬화 가능; numpy/pandas 형은 기본형으로, NaN 은 None 으로).

    키: meta, base_rates{replay_period, full_history}, scorecard[list], episodes{"5","10"}{table, summary},
        allocation, switching, brier_reference, headline, warnings[list]
    """
    tone = _replay_tone(replay)
    warns: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sc = directional_scorecard(replay, targets)
        ep5_tab, ep5_sum = episode_eval(replay, episodes5, targets)
        ep10_tab, ep10_sum = episode_eval(replay, episodes10, targets)
        alloc = allocation_sim(replay, spy_close)
    warns.extend(str(w.message) for w in caught)

    idx = tone.index
    n_days = len(idx)
    tone_counts = tone.value_counts().reindex(list(TONES), fill_value=0)
    br_replay = base_rates(targets, idx[0], idx[-1])
    br_full = base_rates(targets)

    meta: dict[str, Any] = {
        "replay_start": idx[0], "replay_end": idx[-1], "n_days": n_days, "years": n_days / TRADING_DAYS,
        "n_blocks": {str(h): independent_blocks(n_days, h) for h in HORIZONS},
        "tone_counts": tone_counts, "tone_share": tone_counts / max(n_days, 1),
        "spy_start": spy_close.dropna().index[0], "spy_end": spy_close.dropna().index[-1],
        "n_spy_days": int(spy_close.dropna().shape[0]),
        "targets_start": targets.index[0], "targets_end": targets.index[-1],
        "hit_rule": "buy/hold/neutral → 상승 예측, caution/reduce → 비상승 예측; 기준선 = 항상-상승",
        "exposure_map": dict(TONE_EXPOSURE), "cost_bps": 5,
    }
    # replay 의 부가 열이 있으면 데이터 가용성도 기록
    for col in ("n_watch_avail", "n_leaders"):
        if col in replay.columns and replay[col].notna().any():
            meta[f"{col}_min"] = replay[col].min()
            meta[f"{col}_max"] = replay[col].max()
    for col in ("fg_avail", "eod_avail"):
        if col in replay.columns and replay[col].notna().any():
            meta[f"{col}_share"] = float(replay[col].astype(bool).mean())
            avail_idx = replay.index[replay[col].astype(bool)]
            meta[f"{col}_first"] = avail_idx[0] if len(avail_idx) else None

    # 기후학(기저율) Brier — Phase 2 skill 의 기준값
    brier_ref = {}
    for col in ("y_dd5_20", "y_dd10_60", "y_sign_20", "y_sign_60"):
        y = targets[col].reindex(idx).dropna()
        if len(y):
            p0 = float(y.mean())
            brier_ref[col] = {"base_rate": p0, "brier_climatology": brier(p0, y), "n": int(len(y))}

    def _sc_val(tn, h, col):
        m = sc[(sc["tone"] == tn) & (sc["h"] == h)]
        return m[col].iloc[0] if len(m) else np.nan

    headline = {
        "hit_20_by_tone": {tn: _sc_val(tn, 20, "hit") for tn in TONES},
        "n_20_by_tone": {tn: _sc_val(tn, 20, "n") for tn in TONES},
        "baseline_20": _sc_val("all", 20, "baseline"),
        "hit_20_all": _sc_val("all", 20, "hit"),
        "hit_60_all": _sc_val("all", 60, "hit"),
        "baseline_60": _sc_val("all", 60, "baseline"),
        "dd5_20_rate_by_tone": {tn: _sc_val(tn, 20, "dd5_20_rate") for tn in TONES},
        "dd5_20_base_rate": br_replay.get("y_dd5_20", np.nan),
        "detection_rate_10": ep10_sum["detection_rate"], "median_lead_10": ep10_sum["median_lead_days"],
        "n_evaluable_10": ep10_sum["n_evaluable"], "n_detected_10": ep10_sum["n_detected"],
        "median_lead_10_fresh": ep10_sum["median_lead_days_fresh"], "n_lead_capped_10": ep10_sum["n_lead_capped"],
        "lookback": ep10_sum["lookback"],
        "null_detection_rate_10": ep10_sum["null"]["detection_rate_mean"],
        "null_detection_100_share_10": ep10_sum["null"]["detection_100_share"],
        "null_median_lead_10": ep10_sum["null"]["median_lead_days_p50"],
        "detection_rate_5": ep5_sum["detection_rate"], "median_lead_5": ep5_sum["median_lead_days"],
        "n_evaluable_5": ep5_sum["n_evaluable"], "n_detected_5": ep5_sum["n_detected"],
        "median_lead_5_fresh": ep5_sum["median_lead_days_fresh"], "n_lead_capped_5": ep5_sum["n_lead_capped"],
        "null_detection_rate_5": ep5_sum["null"]["detection_rate_mean"],
        "null_median_lead_5": ep5_sum["null"]["median_lead_days_p50"],
        "false_alarms_per_year": ep5_sum["false_alarms_per_year"],
        "false_alarm_rate": ep5_sum["false_alarm_rate"],
        "false_alarm_rate_baseline": ep5_sum["false_alarm_rate_baseline"],
        "true_alarm_share": ep5_sum["true_alarm_share"],
        "true_alarm_share_baseline": ep5_sum["true_alarm_share_baseline"],
        "null_true_alarm_share": ep5_sum["null"]["true_alarm_share_mean"],
        "tone_switches_per_year": ep5_sum["tone_switches_per_year"],
        "cagr_strategy": alloc["cagr"], "cagr_bh": alloc["bh_cagr"],
        "maxdd_strategy": alloc["max_dd"], "maxdd_bh": alloc["bh_max_dd"],
        "worst_month_strategy": alloc["worst_month"], "worst_month_bh": alloc["bh_worst_month"],
    }
    switching = {k: ep5_sum[k] for k in ("n_tone_switches", "tone_switches_per_year", "median_run_len",
                                          "median_warn_run_len", "warn_share", "n_warn_runs",
                                          "n_true_alarms", "n_false_alarms", "n_unresolved_alarms",
                                          "false_alarm_rate", "false_alarm_rate_baseline",
                                          "true_alarm_share", "true_alarm_share_baseline", "false_alarms_per_year")}
    switching["null_true_alarm_share"] = ep5_sum["null"]["true_alarm_share_mean"]
    alloc_json = {k: v for k, v in alloc.items() if k != "series"}

    summary = {
        "meta": meta,
        "base_rates": {"replay_period": br_replay, "full_history": br_full},
        "scorecard": sc,
        "episodes": {"5": {"threshold": 0.05, "table": ep5_tab, "summary": ep5_sum},
                     "10": {"threshold": 0.10, "table": ep10_tab, "summary": ep10_sum}},
        "allocation": alloc_json,
        "switching": switching,
        "brier_reference": brier_ref,
        "headline": headline,
        "warnings": warns,
    }
    return _jsonable(summary)
