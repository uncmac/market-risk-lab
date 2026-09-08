# -*- coding: utf-8 -*-
"""매일 판정 기록 장부 (track record).

ARCHITECTURE.md 계약:
    LEDGER = RESULTS_DIR / "track_record.csv"
    append_today(row, path=LEDGER) -> bool      # asof 중복이면 False
    backfill(path, spy_close) -> DataFrame      # 20/60일 지난 행의 결과 열 채움
    summary(path) -> dict                       # 표본 수·적중·기준선·마지막 기록일·결측일

ARCHITECTURE_PHASE2.md §12 (schema_version=2; 기존 열 순서 불변):
    P2_COLUMNS 를 LEDGER_COLUMNS_V1 뒤에 고정 순서로 추가(구 행은 NaN). append_today 는 P2 열을 받아 정규화
    (숫자 NaN 허용, 문자열은 object; 평면 키 p2_* · 중첩 dict row["p2"] · 별칭(p2_prob→prob_dd5_20 등) 허용).
    backfill 은 그대로 — y_dd5_20 이 채워지면 summary()["p2"] 의 라이브 Brier(vs 기후학·M1·B1·BGK·M3)가 자동 누적된다.
    summary(path, trading_days=None, spy_close=None)["p2"] = {n_scored, brier, brier_clim, brier_m1, brier_vix, brier_m3,
        bss_clim, bss_m1, bss_vix, bss_m3, ci_bss_clim(블록 40 순환 블록 부트스트랩·4,000회·seed 0), episodes5_observed,
        months_elapsed, kill_rule_due, ...} (VALIDATION.md §7 킬 규칙 카운트다운).

열 의미 (2026-09-08 정정 — 배포 모델과 리포트 헤드라인의 불일치 수정):
    prob_dd5_20   = **배포 확률**. summary_p2.json.acceptance 가 배치한 단(tone_model)의 확률이며, 상태(p2_state)·
                    r(p2_r)·톤·비중이 모두 이 값에서 나온다. 예전 계약 문구는 "= p_m3" 였으나(§12), acceptance 가
                    M1 을 배치한 뒤에도 M3 를 기록하면 배포되지 않은 모델을 채점하게 되므로 배포 단으로 바꾼다.
                    deploy_mode == "info_only"(배포 단 없음)이면 생산 모델 M3 의 확률을 **정보로** 기록한다
                    (그때 p2_deploy_mode 가 'info_only' 이므로 톤·비중 주장 아님).
    p2_p_m1/m2/m3 = 사다리 각 단의 그날 확률(정보). 배포 단의 열은 prob_dd5_20 과 같은 값이 된다
                    (그래서 그 단 대비 BSS 는 정의상 0 — summary()["p2"].notes 에 사유를 남긴다).
    p2_prob_model_id = prob_dd5_20 을 실제로 만든 모델의 id (tones: 배포 단, info_only: M3).
    p2_model_id      = 생산 모델(results/model_p2.json = M3)의 id — 재적합 신선도·spec 추적용(의미 불변).
    p2_tone_model / p2_deploy_mode = 그 행에 적용된 acceptance 결과. 이 둘이 prob_dd5_20 의 출처 단을 결정한다
                    (tones → tone_model 단, info_only → M3 정보).
    p2_d_vix/har/ma/refit = **배포 모델**의 일간 귀속(합 == Δ prob_dd5_20). 배포 모델에 없는 항(M1 의 har·ma)은
                    구조적으로 0 이며(지어낸 값이 아니라 그 모델에 그 항이 없다는 뜻), 확률을 못 구한 날은 NaN.
ARCHITECTURE_PHASE3.md §9 (schema_version=3; 기존 열 순서 불변):
    P3_COLUMNS 를 P2_COLUMNS 뒤에 계약 순서 그대로, 그 뒤에 OUTCOME_COLUMNS_P3. **기존 열은 재배열하지도
    다시 계산하지도 않는다** — schema 2 로 기록된 파일은 읽는 순간 없는 열이 NaN 으로 채워져 제자리에서
    승격되며(`_read`), 이미 있는 값은 backfill 도 summary 도 덮어쓰지 않는다. 그래서 '기록의 역사'가 안 바뀐다.
    append_today 는 평면 p3_* 키·중첩 dict row["p3"]·별칭을 같은 열로 정규화하고, 확률 열 [0,1]·
    p3_w_reason·p3_kill_state 열거값을 검사한다(§9). p3_members 는 등록부 순서로 직렬화한 JSON 문자열이다.
    backfill 은 20세션 뒤 fwd_maxdd_20·rv20_realized·ret20_in_vix80/har80·bh_ret_20·rule_ret_20 과
    매일의 p3_rule_dd 를 채운다(장부 자신의 p3_w_exec 경로 · 5bp · evaluate.allocation_from_weights 와 같은 규약).
    summary()["p3"] 는 track.summary_p3 위임 — 킬룰·경보의 단일 원천은 track 이고 장부는 자료를 댄다.
    schema_version 2 → 3 의 유일한 의미는 '뒤에 열이 더 있다' 이다. 파일을 다시 쓰는 이관 절차는 없다.

설계 원칙
* CSV 한 파일이 진실. 매 호출마다 전체를 읽고 통째로 다시 쓴다(행 수가 작아 충분).
* 점(point-in-time) 원칙: 결과 열은 asof 이후 h 거래일이 실제로 지난 행에만 채운다.
* 조용한 실패 금지: 형식 오류는 예외, 데이터 부족은 warnings.warn + 반환 dict 의 warnings.
* 중복 판정 키는 (asof, variant). 같은 날 faithful·completed 두 변형을 나란히 기록하기 위함
  (계약 문구 "asof 중복"의 실질적 의미: 같은 변형의 같은 날은 한 번만).
"""
from __future__ import annotations

import json
import math
import warnings
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from mrl.config import (ENSEMBLE_P3, HOLDOUT_START, HOLDOUT_UNLOCK_PATH, P2, P2_STATES, P3, RESULTS_DIR,
                        TONES, V0_SIGNALS)

LEDGER = RESULTS_DIR / "track_record.csv"

# 열 순서(계약): asof, recorded_at_utc, variant, states(8), score_d/w/m, overall_d/w/m, tone,
#                spy_close, vix_close, prob_dd5_20(None; Phase 2), run_id  + 결과 열
STATE_COLUMNS = [f"state_{k}" for k in V0_SIGNALS]
CORE_COLUMNS = (["asof", "recorded_at_utc", "variant"] + STATE_COLUMNS
                + ["score_d", "score_w", "score_m", "overall_d", "overall_w", "overall_m", "tone",
                   "spy_close", "vix_close", "prob_dd5_20", "run_id"])
OUTCOME_COLUMNS = ["y_sign_20", "y_sign_60", "y_dd5_20", "fwd_ret_20", "fwd_ret_60"]
LEDGER_COLUMNS_V1 = CORE_COLUMNS + OUTCOME_COLUMNS          # Phase 1 장부(schema_version=1)의 전체 열 — 순서 불변

# ------------------------------------------------------------------
# Phase 2 열 (ARCHITECTURE_PHASE2.md §12) — LEDGER_COLUMNS_V1 뒤에 이 순서 그대로. 구 행은 NaN.
#   prob_dd5_20(기존 예약 열) = 배포 확률(acceptance 가 배치한 단; info_only 면 M3 정보). 나머지는 p2_ 접두사.
#   맨 뒤 두 열(p2_p_m3, p2_prob_model_id)은 2026-09-08 에 덧붙였다 — 앞 열의 순서·이름은 그대로다.
# ------------------------------------------------------------------
SCHEMA_VERSION = 3          # Phase 3 (§9). 2 → 3: 열을 **뒤에 덧붙이기만** 했고 기존 열의 이름·순서·의미는 그대로다.
P2_COLUMNS = ["p2_p_m1", "p2_p_m2", "p2_p_vix", "p2_p_vix_bgk", "p2_clim", "p2_lo", "p2_hi", "p2_band_src",
              "p2_x_vix", "p2_x_har", "p2_x_ma", "p2_har_vol_20", "p2_har_fc_20", "p2_r", "p2_state",
              "p2_days_in_state", "p2_tone_model", "p2_deploy_mode", "p2_d_vix", "p2_d_har", "p2_d_ma", "p2_d_refit",
              "p2_input_missing", "p2_model_id", "p2_p_m3", "p2_prob_model_id"]
P2_STRING_COLUMNS = ("p2_band_src", "p2_state", "p2_tone_model", "p2_deploy_mode", "p2_input_missing", "p2_model_id",
                     "p2_prob_model_id")
P2_NUMERIC_COLUMNS = tuple(c for c in P2_COLUMNS if c not in P2_STRING_COLUMNS)
P2_PROB_COLUMNS = ("prob_dd5_20", "p2_p_m1", "p2_p_m2", "p2_p_m3", "p2_p_vix", "p2_p_vix_bgk", "p2_clim", "p2_lo", "p2_hi")   # [0,1] 검사
# 입력 별칭(daily.py 가 어떤 이름을 쓰든 같은 열에 닿게) → 계약 열 이름.
#   'p_m3'/'p2_p_m3' 은 더 이상 prob_dd5_20 의 별칭이 아니다(각자 열이 있다): prob_dd5_20 은 배포 확률이다.
P2_ALIASES = {"p2_prob": "prob_dd5_20", "p2_p": "prob_dd5_20",
              "prob": "prob_dd5_20", "p": "prob_dd5_20", "har_vol_fcst": "p2_har_fc_20", "p2_har_fc": "p2_har_fc_20", "har_fc_20": "p2_har_fc_20",
              "p2_har_vol": "p2_har_vol_20", "p2_days": "p2_days_in_state", "p2_missing": "p2_input_missing"}
P2_BAND_SRC = ("param", "calib")
P2_DEPLOY_MODES = ("info_only", "tones")
P2_TONE_MODELS = ("M1", "M3")
LEDGER_COLUMNS_V2 = LEDGER_COLUMNS_V1 + P2_COLUMNS          # schema_version=2 의 전체 열 — 순서 불변(구 파일이 이 순서로 있다)

# ------------------------------------------------------------------
# Phase 3 열 (ARCHITECTURE_PHASE3.md §9) — P2_COLUMNS 뒤에 계약 순서 그대로. 구 행(schema 2)은 NaN.
#   기존 열은 하나도 옮기지 않고 다시 평균내지도 않는다: 스키마 2 파일은 **제자리에서 승격**되며
#   (`_read` 가 없는 열을 NaN 으로 보강) 기록된 값은 backfill 도 summary 도 덮어쓰지 않는다.
#   결과 열(OUTCOME_COLUMNS_P3)은 P3 입력 열 뒤에 온다 — V1 이 CORE + OUTCOME 이었던 것과 같은 배치.
# ------------------------------------------------------------------
P3_COLUMNS = [
    # 비중
    "p3_sigma_ewma", "p3_sigma_har_fc", "p3_sigma_target", "p3_d_max", "p3_w_vol", "p3_state_mult", "p3_w_target",
    "p3_w_exec", "p3_w_reason", "p3_next_check", "p3_sigma_down", "p3_sigma_up", "p3_deploy_sizing",
    # 국면·등록부·구간
    "p3_hmm_p_high", "p3_hmm_p20", "p3_hmm_q20", "p3_x_hmm", "p3_p_h", "p3_hmm_theta_id", "p3_hmm_gauge",
    "p3_members", "p3_lo", "p3_hi", "p3_band_src", "p3_disagree_flag",
    # 시나리오
    "p3_range_vix_lo", "p3_range_vix_hi", "p3_range_har_lo", "p3_range_har_hi", "p3_scen_bin", "p3_dd_from_ath",
    # 킬룰·경보·정합
    "p3_kill_state", "p3_kill_n_ep", "p3_kill_months", "p3_deploy_mode", "p3_effective_mode", "p3_alarms",
    "p3_registry_sha", "p3_sizing_sha", "p3_input_missing", "p3_run_id",
]
P3_STRING_COLUMNS = ("p3_w_reason", "p3_next_check", "p3_hmm_theta_id", "p3_hmm_gauge", "p3_members", "p3_band_src",
                     "p3_scen_bin", "p3_kill_state", "p3_deploy_mode", "p3_effective_mode", "p3_alarms",
                     "p3_registry_sha", "p3_sizing_sha", "p3_input_missing", "p3_run_id")
P3_PROB_COLUMNS = ("p3_hmm_p_high", "p3_hmm_p20", "p3_hmm_q20", "p3_p_h", "p3_lo", "p3_hi")          # [0,1] 검사
P3_W_REASONS = ("init", "weekly", "escalation", "hold", "input_missing", "info_only")
P3_KILL_STATES = ("not_started", "not_due", "provisional", "validated", "info_only", "manual_kill")
OUTCOME_COLUMNS_P3 = ["fwd_maxdd_20", "rv20_realized", "ret20_in_vix80", "ret20_in_har80", "rule_ret_20",
                      "bh_ret_20", "p3_rule_dd"]

P3_NUMERIC_COLUMNS = tuple(c for c in P3_COLUMNS if c not in P3_STRING_COLUMNS)
P3_DEPLOY_MODES = P2_DEPLOY_MODES                  # ("info_only", "tones") — 유효 모드도 같은 두 값(track.effective_mode)
P3_BAND_SRC = ("members", "p2")                    # ensemble.disagreement 의 src
P3_INT_COLUMNS = ("p3_kill_n_ep", "p3_kill_months")            # 0 이상 정수
P3_BOOL_COLUMNS = ("p3_deploy_sizing",)                        # True/False → 1.0/0.0 (0 도 정보다: 카드 숨김)
# 사건만 기록하는 플래그 열: 플래그가 **선 세션에만** 1.0 이고 그 밖은 NaN(= 플래그 없음).
#   p2_input_missing 과 같은 규약이며, track.live_panel 의 flag_sessions(빈 값이 아닌 행 수)와 정확히 맞는다.
P3_EVENT_FLAG_COLUMNS = ("p3_disagree_flag",)
# 멤버 확률 JSON(`p3_members`)의 고정 키 순서 — 두 실행이 같은 문자열을 내도록(결정론) 여기서 정렬한다.
P3_MEMBER_ORDER = tuple(ENSEMBLE_P3["members"])                # ("p2", "M1", "H")
# 입력 별칭: 평면 키는 반드시 p3_ 접두사, 중첩 dict row["p3"] 의 키는 접두사를 생략해도 된다
#   (`sizing.run()` 의 열 이름 sigma/mult/reason/w_exec 이 그대로 닿게 한다).
P3_ALIASES = {"p3_sigma": "p3_sigma_ewma", "p3_sigma_har": "p3_sigma_har_fc", "p3_har_fc": "p3_sigma_har_fc",
              "p3_mult": "p3_state_mult", "p3_reason": "p3_w_reason", "p3_w": "p3_w_exec",
              "p3_missing": "p3_input_missing", "p3_dd": "p3_dd_from_ath", "p3_theta_id": "p3_hmm_theta_id",
              "p3_gauge": "p3_hmm_gauge", "p3_p_high": "p3_hmm_p_high"}

LEDGER_COLUMNS = LEDGER_COLUMNS_V2 + P3_COLUMNS + OUTCOME_COLUMNS_P3   # schema 3 전체 열(알 수 없는 추가 열은 그 뒤에 보존)
ALL_OUTCOME_COLUMNS = OUTCOME_COLUMNS + OUTCOME_COLUMNS_P3             # backfill 이 채우는 열 전부

# 결과 열 정의 (mrl/targets.py · VALIDATION.md 와 동일)
_DD5_THRESHOLD = 0.05
_DD5_WINDOW = 20
_HORIZONS = (20, 60)

# 방향 예측 규약 (evaluate 계약): buy/hold/neutral → 상승 예측, caution/reduce → 하락 예측
_UP_TONES = ("buy", "hold", "neutral")


# ------------------------------------------------------------------
# 유틸
# ------------------------------------------------------------------
def _asof_str(v) -> str:
    """asof 를 'YYYY-MM-DD' 문자열로 정규화. 해석 불가면 ValueError."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        raise ValueError("asof 가 비어 있습니다")
    if isinstance(v, (datetime, pd.Timestamp)):
        return pd.Timestamp(v).strftime("%Y-%m-%d")
    if isinstance(v, date):
        return v.strftime("%Y-%m-%d")
    ts = pd.Timestamp(str(v))
    if pd.isna(ts):
        raise ValueError(f"asof 해석 불가: {v!r}")
    return ts.strftime("%Y-%m-%d")


def _is_missing(v) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _num(v):
    """숫자 열 값 정규화: None/NaN → NaN, 그 외 float."""
    if _is_missing(v):
        return np.nan
    return float(v)


def _read(path: Path) -> pd.DataFrame:
    """장부 CSV 읽기. 없으면 빈 DataFrame(계약 열). 열이 부족하면 NaN 으로 보강.

    **제자리 승격(schema 2 → 3)**: 스키마 2 로 기록된 파일은 P3 열이 없다. 그 열들을 NaN 으로 덧붙이기만 하고
    기존 열의 순서·이름·값은 하나도 건드리지 않는다(기록을 다시 쓰지 않는다). 알 수 없는 추가 열은 맨 뒤에 보존."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    df = pd.read_csv(path, dtype={"asof": str, "variant": str, "tone": str, "run_id": str,
                                  **{c: str for c in STATE_COLUMNS},
                                  **{c: str for c in P2_STRING_COLUMNS},
                                  **{c: str for c in P3_STRING_COLUMNS},
                                  "overall_d": str, "overall_w": str, "overall_m": str})
    for c in LEDGER_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    # 알 수 없는 추가 열은 보존(뒤에 둠)
    extra = [c for c in df.columns if c not in LEDGER_COLUMNS]
    return df[LEDGER_COLUMNS + extra]


def _write(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f", lineterminator="\n", encoding="utf-8")


def _p2_colname(key) -> str | None:
    """입력 키 → P2 계약 열 이름. 모르는 키면 None. 허용: 계약 열 그대로, 별칭, 접두사 없는 이름(중첩 dict 용)."""
    k = str(key)
    if k in P2_COLUMNS or k == "prob_dd5_20":
        return k
    if k in P2_ALIASES:
        return P2_ALIASES[k]
    if f"p2_{k}" in P2_COLUMNS:
        return f"p2_{k}"
    return None


def _collect_p2(row: dict) -> tuple[dict, list[str]]:
    """row 의 P2 값을 {계약 열: 값} 으로 모은다. 평면 키(p2_*/별칭)가 중첩 dict row["p2"] 보다 우선.
    반환 (found, unknown_keys) — 알 수 없는 p2_* 키는 호출자가 경고한다(조용히 버리지 않음)."""
    found: dict = {}
    unknown: list[str] = []
    nested = row.get("p2") if isinstance(row.get("p2"), dict) else {}
    for k, v in nested.items():
        col = _p2_colname(k)
        if col is None:
            unknown.append(f"p2.{k}")
        else:
            found[col] = v
    for k, v in row.items():
        if k == "p2" or k == "prob_dd5_20":
            continue
        ks = str(k)
        if ks in P2_COLUMNS or ks in P2_ALIASES:
            found[_p2_colname(ks)] = v
        elif ks.startswith("p2_"):
            unknown.append(ks)
    return found, unknown


def _normalize_p2(row: dict, out: dict) -> None:
    """P2 열 정규화: 숫자 열은 float(NaN 허용, 숫자 아니면 ValueError), 문자열 열은 str(빈 문자열은 NaN).
    검사: p2_state ∈ P2_STATES(아니면 ValueError), 확률 열 ∈ [0,1](아니면 ValueError), lo ≤ p ≤ hi(아니면 경고),
    확률 NaN 인데 p2_input_missing 사유가 없으면 경고, deploy_mode/band_src/tone_model 의 알 수 없는 값은 경고."""
    found, unknown = _collect_p2(row)
    if unknown:
        warnings.warn(f"알 수 없는 P2 키 {sorted(unknown)} 은 장부 열이 아니라 기록하지 않습니다 (계약 §12 열: {P2_COLUMNS})")
    if not found:
        return
    for col, v in found.items():
        if col == "prob_dd5_20":
            if _is_missing(out.get("prob_dd5_20")):       # 평면 prob_dd5_20 이 이미 있으면 그것이 우선
                out["prob_dd5_20"] = _num(v)
            continue
        if col in P2_STRING_COLUMNS:
            out[col] = np.nan if _is_missing(v) or str(v).strip() == "" else str(v).strip()
        else:
            try:
                out[col] = _num(v)
            except (TypeError, ValueError) as e:
                raise ValueError(f"P2 숫자 열 {col} 의 값이 숫자가 아닙니다: {v!r}") from e
    # 값 검사
    st = out.get("p2_state")
    if not _is_missing(st) and st not in P2_STATES:
        raise ValueError(f"알 수 없는 p2_state: {st!r} (허용: {P2_STATES})")
    for c in P2_PROB_COLUMNS:
        v = out.get(c)
        if not _is_missing(v) and not (0.0 <= float(v) <= 1.0):
            raise ValueError(f"{c} 는 [0,1] 확률이어야 합니다: {v}")
    dm = out.get("p2_deploy_mode")
    if not _is_missing(dm) and dm not in P2_DEPLOY_MODES:
        warnings.warn(f"알 수 없는 p2_deploy_mode {dm!r} (허용: {P2_DEPLOY_MODES}) — 그대로 기록")
    bs = out.get("p2_band_src")
    if not _is_missing(bs) and bs not in P2_BAND_SRC:
        warnings.warn(f"알 수 없는 p2_band_src {bs!r} (허용: {P2_BAND_SRC}) — 그대로 기록")
    tm = out.get("p2_tone_model")
    if not _is_missing(tm) and tm not in P2_TONE_MODELS:
        warnings.warn(f"알 수 없는 p2_tone_model {tm!r} (허용: {P2_TONE_MODELS}) — 그대로 기록")
    p, lo, hi = out.get("prob_dd5_20"), out.get("p2_lo"), out.get("p2_hi")
    if not _is_missing(p):
        if not _is_missing(lo) and float(lo) > float(p) + 1e-12:
            warnings.warn(f"p2_lo({lo}) > prob_dd5_20({p}) — 구간이 확률을 포함하지 않음")
        if not _is_missing(hi) and float(hi) < float(p) - 1e-12:
            warnings.warn(f"p2_hi({hi}) < prob_dd5_20({p}) — 구간이 확률을 포함하지 않음")
    elif _is_missing(out.get("p2_input_missing")):
        warnings.warn("prob_dd5_20 이 NaN 인데 p2_input_missing 사유가 없습니다 (조용한 실패 금지: 사유를 기록하세요)")
    d = out.get("p2_days_in_state")
    if not _is_missing(d) and (float(d) < 0 or float(d) != int(float(d))):
        raise ValueError(f"p2_days_in_state 는 0 이상 정수여야 합니다: {d}")


# ------------------------------------------------------------------
# Phase 3 열 정규화 (ARCHITECTURE_PHASE3.md §9)
# ------------------------------------------------------------------
def _p3_colname(key) -> str | None:
    """입력 키 → P3 계약 열 이름. 모르는 키면 None.
    허용: 계약 열 그대로, p3_ 별칭, 그리고 (중첩 dict 용) 접두사를 뗀 이름."""
    k = str(key)
    if k in P3_COLUMNS:
        return k
    if k in P3_ALIASES:
        return P3_ALIASES[k]
    pk = f"p3_{k}"
    if pk in P3_COLUMNS:
        return pk
    if pk in P3_ALIASES:
        return P3_ALIASES[pk]
    return None


def _collect_p3(row: dict) -> tuple[dict, list[str]]:
    """row 의 P3 값을 {계약 열: 값} 으로 모은다. 평면 키(p3_*/별칭)가 중첩 dict row["p3"] 보다 우선.
    평면 키는 반드시 `p3_` 로 시작해야 한다(접두사 없는 이름을 최상위에서 낚아채면 v0/P2 키와 충돌한다).
    반환 (found, unknown_keys) — 알 수 없는 p3_* 키는 호출자가 경고한다(조용히 버리지 않음)."""
    found: dict = {}
    unknown: list[str] = []
    nested = row.get("p3") if isinstance(row.get("p3"), dict) else {}
    for k, v in nested.items():
        col = _p3_colname(k)
        if col is None:
            unknown.append(f"p3.{k}")
        else:
            found[col] = v
    for k, v in row.items():
        if k == "p3":
            continue
        ks = str(k)
        if ks in P3_COLUMNS or ks in P3_ALIASES:
            found[_p3_colname(ks)] = v
        elif ks in OUTCOME_COLUMNS_P3:
            continue                       # 결과 열 — _normalize_row 가 따로 경고한다(backfill 소관)
        elif ks.startswith("p3_"):
            unknown.append(ks)
    return found, unknown


def _members_json(v) -> str:
    """`p3_members` 정규화 → JSON 문자열 `{"p2": p, "M1": p, "H": p}`.

    dict 를 주면 등록부 순서(P3_MEMBER_ORDER, 그 뒤 나머지는 이름순)로 직렬화한다 — 호출자의 dict 순서와
    무관하게 **같은 입력이면 같은 문자열**이 나와야 하기 때문(결정론). 문자열을 주면 파싱해 같은 규약으로
    다시 쓴다(형식이 아니면 ValueError). 값은 [0,1] 확률 또는 결측(→ null)."""
    if isinstance(v, str):
        try:
            obj = json.loads(v)
        except (TypeError, ValueError) as e:
            raise ValueError(f"p3_members 는 JSON 객체 문자열이어야 합니다: {v!r}") from e
    else:
        obj = v
    if not isinstance(obj, dict):
        raise ValueError(f"p3_members 는 {{멤버: 확률}} dict(또는 그 JSON 문자열)이어야 합니다: {v!r}")
    clean: dict = {}
    for k, val in obj.items():
        name = str(k)
        if _is_missing(val):
            clean[name] = None
            continue
        try:
            f = float(val)
        except (TypeError, ValueError) as e:
            raise ValueError(f"p3_members[{name!r}] 이 숫자가 아닙니다: {val!r}") from e
        if not (0.0 <= f <= 1.0):
            raise ValueError(f"p3_members[{name!r}] 는 [0,1] 확률이어야 합니다: {f}")
        clean[name] = f
    order = [m for m in P3_MEMBER_ORDER if m in clean] + sorted(k for k in clean if k not in P3_MEMBER_ORDER)
    return json.dumps({k: clean[k] for k in order}, ensure_ascii=False)


def _alarms_str(v) -> str:
    """`p3_alarms` 정규화 → 콤마로 이은 코드 문자열. 리스트/집합을 주면 **이름순**으로 잇는다(결정론).
    경보는 순서 없는 집합이므로 정렬이 곧 규약이다. 빈 목록은 호출자가 NaN 으로 처리한다."""
    if isinstance(v, (list, tuple, set, frozenset)):
        codes = sorted({str(x).strip() for x in v if str(x).strip()})
    else:
        codes = sorted({c.strip() for c in str(v).split(",") if c.strip()})
    return ",".join(codes)


def _flag_true(v) -> bool:
    """불리언 열 해석. CSV 왕복(문자열 'True'/'0')과 numpy bool 을 모두 같은 뜻으로 읽는다 —
    `bool("False") is True` 같은 파이썬 함정으로 플래그가 뒤집히지 않게."""
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "t", "yes", "y", "1", "1.0"):
            return True
        if s in ("false", "f", "no", "n", "0", "0.0", ""):
            return False
        raise ValueError(f"불리언으로 읽을 수 없는 값: {v!r}")
    try:
        return float(v) != 0.0
    except (TypeError, ValueError):
        return bool(v)


def _normalize_p3(row: dict, out: dict) -> None:
    """P3 열 정규화·검사 (§9).

    * 숫자 열은 float(NaN 허용), 문자열 열은 str(빈 문자열 → NaN), `p3_members` 는 JSON 문자열.
    * 열거값: `p3_w_reason` ∈ P3_W_REASONS · `p3_kill_state` ∈ P3_KILL_STATES 는 **ValueError**
      (이 모듈이 정의한 닫힌 집합이다). `p3_deploy_mode`/`p3_effective_mode`/`p3_band_src` 는 경고만
      (P2 의 deploy_mode·band_src 와 같은 취급 — 다른 모듈이 값을 넓힐 수 있다).
    * 확률 열 ∈ [0,1] 은 ValueError. `p3_lo > p3_hi` 는 ValueError, 밴드가 prob_dd5_20 을 품지 않으면 경고.
    * `p3_deploy_sizing` 은 True/False → 1.0/0.0 (0 은 '유지 조건 위반 → 카드 숨김' 이라는 정보다).
    * `p3_disagree_flag` 는 **사건만 기록**: 참이면 1.0, 거짓/결측이면 NaN.
    * `p3_w_exec` 가 NaN 인데 사유(`p3_w_reason`/`p3_input_missing`)가 없으면 경고(조용한 실패 금지).
    """
    found, unknown = _collect_p3(row)
    if unknown:
        warnings.warn(f"알 수 없는 P3 키 {sorted(unknown)} 은 장부 열이 아니라 기록하지 않습니다 (계약 §9 열: {P3_COLUMNS})")
    if not found:
        return
    for col, v in found.items():
        if col == "p3_members":
            out[col] = np.nan if _is_missing(v) else _members_json(v)
        elif col == "p3_alarms":
            s = "" if _is_missing(v) else _alarms_str(v)
            out[col] = s if s else np.nan
        elif col == "p3_next_check":
            if _is_missing(v):
                out[col] = np.nan
            elif isinstance(v, (date, datetime, pd.Timestamp)):
                out[col] = _asof_str(v)
            else:
                out[col] = np.nan if str(v).strip() == "" else str(v).strip()
        elif col in P3_STRING_COLUMNS:
            out[col] = np.nan if _is_missing(v) or str(v).strip() == "" else str(v).strip()
        elif col in P3_EVENT_FLAG_COLUMNS:
            out[col] = 1.0 if (not _is_missing(v) and _flag_true(v)) else np.nan
        elif col in P3_BOOL_COLUMNS:
            out[col] = np.nan if _is_missing(v) else (1.0 if _flag_true(v) else 0.0)
        else:
            try:
                out[col] = _num(v)
            except (TypeError, ValueError) as e:
                raise ValueError(f"P3 숫자 열 {col} 의 값이 숫자가 아닙니다: {v!r}") from e
    # 값 검사 — 닫힌 열거값은 예외, 열린 것은 경고
    for col, allowed in (("p3_w_reason", P3_W_REASONS), ("p3_kill_state", P3_KILL_STATES)):
        v = out.get(col)
        if not _is_missing(v) and v not in allowed:
            raise ValueError(f"알 수 없는 {col}: {v!r} (허용: {allowed})")
    for col, allowed in (("p3_deploy_mode", P3_DEPLOY_MODES), ("p3_effective_mode", P3_DEPLOY_MODES),
                         ("p3_band_src", P3_BAND_SRC)):
        v = out.get(col)
        if not _is_missing(v) and v not in allowed:
            warnings.warn(f"알 수 없는 {col} {v!r} (허용: {allowed}) — 그대로 기록")
    for c in P3_PROB_COLUMNS:
        v = out.get(c)
        if not _is_missing(v) and not (0.0 <= float(v) <= 1.0):
            raise ValueError(f"{c} 는 [0,1] 확률이어야 합니다: {v}")
    for c in P3_INT_COLUMNS:
        v = out.get(c)
        if not _is_missing(v) and (float(v) < 0 or float(v) != int(float(v))):
            raise ValueError(f"{c} 는 0 이상 정수여야 합니다: {v}")
    lo, hi = out.get("p3_lo"), out.get("p3_hi")
    if not _is_missing(lo) and not _is_missing(hi) and float(lo) > float(hi) + 1e-12:
        raise ValueError(f"p3_lo({lo}) > p3_hi({hi}) — 뒤집힌 구간")
    p = out.get("prob_dd5_20")
    if not _is_missing(p):
        if not _is_missing(lo) and float(lo) > float(p) + 1e-12:
            warnings.warn(f"p3_lo({lo}) > prob_dd5_20({p}) — 등록부 구간이 배포 확률을 포함하지 않음")
        if not _is_missing(hi) and float(hi) < float(p) - 1e-12:
            warnings.warn(f"p3_hi({hi}) < prob_dd5_20({p}) — 등록부 구간이 배포 확률을 포함하지 않음")
    w = out.get("p3_w_exec")
    reason = out.get("p3_w_reason")
    if _is_missing(w) and _is_missing(reason) and _is_missing(out.get("p3_input_missing")):
        warnings.warn("p3_w_exec 이 NaN 인데 p3_w_reason·p3_input_missing 사유가 없습니다 "
                      "(조용한 실패 금지: info_only/input_missing 같은 사유를 기록하세요)")
    if not _is_missing(w) and reason == "info_only":
        warnings.warn(f"p3_w_reason='info_only' 인데 p3_w_exec({w}) 이 기록됐습니다 — "
                      "정보 제공 전용에서는 비중을 제안하지 않습니다(§6.1-4)")


def _normalize_row(row: dict) -> dict:
    """입력 row(v0_day 반환값 + 부가 정보)를 장부 열로 정규화.

    허용 입력:
      * 상태: row["state_fang"]... 또는 row["states_d"] / row["states"] (dict) 중 하나
      * asof: str/date/datetime/Timestamp
      * variant: 없으면 'faithful' (v0 라이브와 동일) — 경고 없이 기본값 적용은 하지 않고 warnings.warn
    """
    if not isinstance(row, dict):
        raise TypeError("row 는 dict 여야 합니다")
    if "asof" not in row:
        raise ValueError("row 에 asof 가 없습니다")
    out: dict = {c: np.nan for c in LEDGER_COLUMNS}
    out["asof"] = _asof_str(row["asof"])

    variant = row.get("variant")
    if _is_missing(variant) or str(variant).strip() == "":
        warnings.warn("row 에 variant 가 없어 'faithful' 로 기록합니다")
        variant = "faithful"
    out["variant"] = str(variant)

    rec = row.get("recorded_at_utc")
    if _is_missing(rec):
        rec = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out["recorded_at_utc"] = str(rec)

    # 상태 8개
    states = row.get("states_d")
    if not isinstance(states, dict):
        states = row.get("states") if isinstance(row.get("states"), dict) else None
    missing_states = []
    for k in V0_SIGNALS:
        col = f"state_{k}"
        v = row.get(col)
        if _is_missing(v) and states is not None:
            v = states.get(k)
        if _is_missing(v):
            missing_states.append(k)
            out[col] = np.nan
        else:
            out[col] = str(v)
    if missing_states:
        warnings.warn(f"상태 누락 신호: {missing_states} (NaN 으로 기록)")

    for c in ("score_d", "score_w", "score_m", "spy_close", "vix_close", "prob_dd5_20"):
        out[c] = _num(row.get(c))
    for c in ("overall_d", "overall_w", "overall_m"):
        v = row.get(c)
        out[c] = np.nan if _is_missing(v) else str(v)

    tone = row.get("tone")
    if _is_missing(tone):
        raise ValueError("row 에 tone 이 없습니다")
    tone = str(tone)
    if tone not in TONES:
        raise ValueError(f"알 수 없는 tone: {tone!r} (허용: {TONES})")
    out["tone"] = tone

    rid = row.get("run_id")
    out["run_id"] = np.nan if _is_missing(rid) else str(rid)

    # Phase 2 열 (없으면 전부 NaN — Phase 1 호출과 호환)
    _normalize_p2(row, out)
    # Phase 3 열 (없으면 전부 NaN — Phase 1·2 호출과 호환)
    _normalize_p3(row, out)

    # 결과 열은 append 시점엔 항상 비어 있음(backfill 이 채움). 입력에 있어도 무시하고 경고.
    given = [c for c in ALL_OUTCOME_COLUMNS if c in row and not _is_missing(row[c])]
    if given:
        warnings.warn(f"결과 열 {given} 은 append 시 기록하지 않습니다 (backfill 이 채움)")
    return out


# ------------------------------------------------------------------
# 계약 함수
# ------------------------------------------------------------------
def append_today(row: dict, path=LEDGER) -> bool:
    """오늘 판정 한 행을 장부에 추가. (asof, variant) 가 이미 있으면 False."""
    path = Path(path)
    new = _normalize_row(row)
    df = _read(path)
    if len(df):
        dup = (df["asof"] == new["asof"]) & (df["variant"] == new["variant"])
        if bool(dup.any()):
            return False
    new_df = pd.DataFrame([new], columns=LEDGER_COLUMNS)
    if len(df):
        df = pd.concat([df, new_df], ignore_index=True)
    else:
        df = new_df
    df = df.sort_values(["asof", "variant"], kind="stable").reset_index(drop=True)
    _write(df, path)
    return True


def _prep_close(spy_close) -> pd.Series:
    if isinstance(spy_close, pd.DataFrame):
        if "Close" in spy_close.columns:
            spy_close = spy_close["Close"]
        elif "SPY" in spy_close.columns:
            spy_close = spy_close["SPY"]
        elif spy_close.shape[1] == 1:
            spy_close = spy_close.iloc[:, 0]
        else:
            raise ValueError("spy_close DataFrame 에서 종가 열을 찾을 수 없습니다 (Close/SPY)")
    s = pd.Series(spy_close).dropna().astype(float)
    if len(s) == 0:
        raise ValueError("spy_close 가 비어 있습니다")
    idx = pd.DatetimeIndex(pd.to_datetime(s.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _rule_daily_returns(w_on_sessions: pd.Series, close: pd.Series, cost_bps: float) -> pd.Series:
    """장부의 w_exec 경로 → 세션별 규칙 수익률. `evaluate.allocation_from_weights` 와 **같은 규약**:
    r[k] = (1 + w[k-1]·(C[k]/C[k-1] − 1))·(1 − cost_bps·|w[k] − w[k-1]|) − 1. 첫 세션은 정의되지 않는다."""
    c = close.loc[w_on_sessions.index].astype(float)
    r = c.pct_change()
    w_prev = w_on_sessions.shift(1)
    dw = (w_on_sessions - w_prev).abs()
    cost = dw * (float(cost_bps) / 1e4)
    return ((1.0 + w_prev * r) * (1.0 - cost) - 1.0).iloc[1:]


def _backfill_p3(df: pd.DataFrame, s: pd.Series, warns: list[str]) -> None:
    """§9 의 P3 결과 열을 제자리에서 채운다 (이미 채워진 값은 다시 쓰지 않는다).

    * `fwd_maxdd_20` = min(close[t+1..t+20])/close[t] − 1 (targets.py 와 같은 정의)
    * `rv20_realized` = t+1..t+20 종가 로그수익 20개의 표본표준편차 × √252
    * `ret20_in_vix80` / `ret20_in_har80` = 실현 close[t+20] 이 그날 기록한 80% 밴드
      (`p3_range_vix_lo/hi`, `p3_range_har_lo/hi`) 안인가 (밴드가 없으면 NaN — 0 으로 채우지 않는다)
    * `bh_ret_20` = close[t+20]/close[t] − 1 (보유 20세션 수익; fwd_ret_20 과 같은 수, 규칙과 나란히 두려고 별도 열)
    * `rule_ret_20` = **장부 자신의** `p3_w_exec` 경로를 SPY 에 적용해 20세션 복리(5bp; 점 원칙 w[t]→r[t+1])
    * `p3_rule_dd` = 규칙 라이브 자본곡선의 그날 낙폭(규칙 라이브 시작 = p3_w_exec 이 처음 기록된 세션부터).
      미래 자료가 필요 없으므로 20세션을 기다리지 않고 **매일** 채운다.
    변형(variant)마다 독립으로 계산한다 — faithful 과 completed 의 비중 경로를 섞지 않는다.
    """
    idx, vals, n = s.index, s.to_numpy(), len(s)
    ln = np.log(vals)
    for c in OUTCOME_COLUMNS_P3:
        if c not in df.columns:
            df[c] = np.nan
    pos_of = {}
    for i in df.index:
        ts = pd.Timestamp(df.at[i, "asof"])
        if ts in idx:
            pos_of[i] = int(idx.get_loc(ts))

    # (1) 20세션 뒤에 확정되는 열
    for i, pos in pos_of.items():
        if pos + _DD5_WINDOW >= n:
            continue
        c0 = vals[pos]
        win = vals[pos + 1: pos + _DD5_WINDOW + 1]
        if _is_missing(df.at[i, "fwd_maxdd_20"]):
            df.at[i, "fwd_maxdd_20"] = round(float(win.min() / c0 - 1.0), 6)
        if _is_missing(df.at[i, "rv20_realized"]):
            lr = np.diff(ln[pos: pos + _DD5_WINDOW + 1])
            df.at[i, "rv20_realized"] = round(float(lr.std(ddof=1) * math.sqrt(252)), 6)
        if _is_missing(df.at[i, "bh_ret_20"]):
            df.at[i, "bh_ret_20"] = round(float(vals[pos + _DD5_WINDOW] / c0 - 1.0), 6)
        c_end = vals[pos + _DD5_WINDOW]
        for col, lo_c, hi_c in (("ret20_in_vix80", "p3_range_vix_lo", "p3_range_vix_hi"),
                                ("ret20_in_har80", "p3_range_har_lo", "p3_range_har_hi")):
            if not _is_missing(df.at[i, col]):
                continue
            lo, hi = df.at[i, lo_c] if lo_c in df.columns else np.nan, df.at[i, hi_c] if hi_c in df.columns else np.nan
            if _is_missing(lo) or _is_missing(hi):
                continue                                  # 밴드를 기록하지 않은 날은 채점하지 않는다
            df.at[i, col] = float(float(lo) <= c_end <= float(hi))

    # (2) 규칙 경로 — 변형마다 따로
    if "p3_w_exec" not in df.columns:
        return
    w_all = pd.to_numeric(df["p3_w_exec"], errors="coerce")
    for variant in sorted(str(v) for v in df["variant"].dropna().unique()):
        rows = [i for i in df.index if str(df.at[i, "variant"]) == variant and i in pos_of and not _is_missing(w_all.at[i])]
        if len(rows) < 2:
            continue
        rows.sort(key=lambda i: pos_of[i])
        first_pos, last_pos = pos_of[rows[0]], pos_of[rows[-1]]
        sess = idx[first_pos: last_pos + 1]
        w_ses = pd.Series(np.nan, index=sess, dtype=float)
        for i in rows:
            w_ses.iloc[pos_of[i] - first_pos] = float(w_all.at[i])
        n_gap = int(w_ses.isna().sum())
        if n_gap:
            warns.append(f"[{variant}] 규칙 경로: 장부에 없는 거래일 {n_gap}일은 직전 w_exec 을 유지(규칙의 'hold')로 "
                         f"메워 rule_ret_20·p3_rule_dd 를 계산했습니다")
        w_ses = w_ses.ffill()
        ret = _rule_daily_returns(w_ses, s, float(P3["cost_bps"]))          # 첫 세션 제외
        growth = (1.0 + ret).to_numpy(dtype=float)
        cum = np.concatenate([[1.0], np.cumprod(growth)])                   # cum[k] = 자본(첫 세션 = 1.0)
        eq = pd.Series(cum, index=sess)
        dd = (eq / eq.cummax() - 1.0)
        for i in rows:
            k = pos_of[i] - first_pos
            if _is_missing(df.at[i, "p3_rule_dd"]):
                df.at[i, "p3_rule_dd"] = round(float(dd.iloc[k]), 6)
            if k + _DD5_WINDOW < len(sess) and _is_missing(df.at[i, "rule_ret_20"]):
                df.at[i, "rule_ret_20"] = round(float(cum[k + _DD5_WINDOW] / cum[k] - 1.0), 6)


def backfill(path, spy_close) -> pd.DataFrame:
    """asof 로부터 20/60 거래일이 지난 행의 y_sign_20/60, y_dd5_20, fwd_ret_20/60 (+ §9 의 P3 결과 열)을 채운다.

    * 정의는 mrl/targets.py 와 동일: fwd_ret_h = close[t+h]/close[t]-1, y_sign_h = fwd_ret_h > 0,
      y_dd5_20 = min(close[t+1..t+20])/close[t]-1 <= -0.05. 위치(거래일 수) 기준.
    * 기준 종가는 장부의 spy_close 가 아니라 전달된 spy_close 시계열의 asof 값(같은 조정 기준 유지).
    * 이미 채워진 값은 다시 쓰지 않는다(기록 안정성). asof 가 spy_close 인덱스에 없으면 경고 후 건너뜀.
    * P3 결과 열은 `_backfill_p3` 참조. 호출자는 **완성 세션까지의 종가만** 넘겨야 한다(점 원칙).
    """
    path = Path(path)
    df = _read(path)
    if len(df) == 0:
        return df
    s = _prep_close(spy_close)
    idx = s.index
    vals = s.to_numpy()
    n = len(vals)
    missing_asof = []
    for i in df.index:
        ts = pd.Timestamp(df.at[i, "asof"])
        if ts not in idx:
            missing_asof.append(df.at[i, "asof"])
            continue
        pos = int(idx.get_loc(ts))
        c0 = vals[pos]
        for h in _HORIZONS:
            if pos + h >= n:           # 아직 h 거래일이 지나지 않음(미래 행 없음) → 채우지 않음
                continue
            fr = vals[pos + h] / c0 - 1.0
            if _is_missing(df.at[i, f"fwd_ret_{h}"]):
                df.at[i, f"fwd_ret_{h}"] = round(float(fr), 6)   # CSV 저장 정밀도(6자리)와 동일하게 — 반환값 = 파일 내용
            if _is_missing(df.at[i, f"y_sign_{h}"]):
                df.at[i, f"y_sign_{h}"] = float(fr > 0)
        if pos + _DD5_WINDOW < n and _is_missing(df.at[i, "y_dd5_20"]):
            mn = vals[pos + 1: pos + _DD5_WINDOW + 1].min() / c0 - 1.0
            df.at[i, "y_dd5_20"] = float(mn <= -_DD5_THRESHOLD)
    if missing_asof:
        warnings.warn(f"spy_close 에 없는 asof {len(missing_asof)}건 → backfill 건너뜀: "
                      f"{missing_asof[:5]}{' ...' if len(missing_asof) > 5 else ''}")
    p3_warns: list[str] = []
    _backfill_p3(df, s, p3_warns)
    for w in p3_warns:
        warnings.warn(w)
    for c in ALL_OUTCOME_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    _write(df, path)
    return df


def _trading_days_between(start: pd.Timestamp, end: pd.Timestamp, trading_days=None) -> tuple[pd.DatetimeIndex, str]:
    """[start, end] 구간의 거래일 목록과 판정 방법 문자열.
    우선순위: 인자 trading_days → mrl.calendar_us.is_trading_day → 평일(주말만 제외, 휴장일 미반영)."""
    if trading_days is not None:
        td = pd.DatetimeIndex(pd.to_datetime(pd.Index(trading_days))).normalize()
        return td[(td >= start) & (td <= end)], "trading_days 인자"
    try:
        from mrl.calendar_us import is_trading_day  # 다른 모듈(동시 작성) — 있으면 사용
        days = pd.date_range(start, end, freq="D")
        return pd.DatetimeIndex([d for d in days if is_trading_day(d.date())]), "mrl.calendar_us"
    except Exception as e:  # ImportError 또는 함수 오류 → 평일 폴백(경고)
        warnings.warn(f"calendar_us 사용 불가({type(e).__name__}: {e}) → 평일 기준으로 결측일 계산(휴장일 포함될 수 있음)")
        return pd.bdate_range(start, end), "평일(휴장일 미반영)"


def _tone_block(sub: pd.DataFrame) -> dict:
    """톤별(또는 전체) 적중 통계. 결과가 있는 행만 사용."""
    out = {"n": int(len(sub))}
    for h in _HORIZONS:
        y = pd.to_numeric(sub[f"y_sign_{h}"], errors="coerce")
        ok = y.notna()
        out[f"n_outcome_{h}"] = int(ok.sum())
        if ok.any():
            pred_up = sub.loc[ok, "tone"].isin(_UP_TONES).astype(float)
            out[f"hit_{h}"] = float((pred_up.to_numpy() == y[ok].to_numpy()).mean())
            out[f"base_{h}"] = float(y[ok].mean())          # 항상-상승 기준선(이 표본의 상승 비율)
            fr = pd.to_numeric(sub.loc[ok, f"fwd_ret_{h}"], errors="coerce")
            out[f"fwd_ret_{h}_mean"] = float(fr.mean()) if fr.notna().any() else None
        else:
            out[f"hit_{h}"] = None
            out[f"base_{h}"] = None
            out[f"fwd_ret_{h}_mean"] = None
    dd = pd.to_numeric(sub["y_dd5_20"], errors="coerce")
    out["n_outcome_dd5"] = int(dd.notna().sum())
    out["dd5_rate"] = float(dd.dropna().mean()) if dd.notna().any() else None
    return out


def summary(path=LEDGER, trading_days=None, spy_close=None, *, reference=None, prev_kill=None,
            deploy_sizing=None, recomputed=None) -> dict:
    """장부 요약 (JSON 직렬화 가능).

    키: exists, path, schema_version, n, n_with_outcome, n_with_outcome_60, first_asof, last_asof, last_recorded_at_utc,
        variants(list), by_tone{tone: {...}}, by_variant{variant: {...}}, overall{...}, base_rate_20/60,
        missing_days(list[str]), n_missing_days, missing_method, warnings(list[str]),
        p2{...} (Phase 2 라이브 Brier·킬룰 카운트다운 — _p2_block 참조; P2 행이 없으면 n_rows=0 인 빈 블록),
        p3{...} (§9: track.summary_p3 위임 — 비중·등록부·불일치·킬룰·경보·구간 포함률)
    trading_days: 결측일 판정에 쓸 거래일 인덱스(선택). 없으면 calendar_us → 평일 순으로 폴백.
    spy_close: (선택) SPY 종가 시계열 — 있으면 p2.episodes5_observed 를 targets.episodes(0.05, split=False) 로 센다.
               없으면 장부의 y_dd5_20 런 수로 근사(p2.episodes5_method 에 표기). p3 의 실현 에피소드도 이것으로 센다.
    reference / prev_kill / deploy_sizing / recomputed: p3 블록에 그대로 넘긴다(각각 백테스트 참조분포
        `summary_p3.json.reference`, 직전 킬 상태 `model_p3.json.kill`, `sizing.retention` 의 판정, 주간 재현 프레임).

    주의(§9): `summary()["p2"]["kill_rule"]` 은 VALIDATION §7 의 **문자 그대로**('둘 다') 판독이라 남겨 두지만,
    카드·페이지는 2단계 킬룰인 `summary()["p3"]["kill"]` 만 읽는다.
    """
    path = Path(path)
    warns: list[str] = []
    out = {"exists": path.exists(), "path": str(path), "schema_version": SCHEMA_VERSION,
           "n": 0, "n_with_outcome": 0, "n_with_outcome_60": 0,
           "first_asof": None, "last_asof": None, "last_recorded_at_utc": None, "variants": [],
           "by_tone": {}, "by_variant": {}, "overall": None, "base_rate_20": None, "base_rate_60": None,
           "missing_days": [], "n_missing_days": 0, "missing_method": None, "warnings": warns,
           "p2": _p2_empty(), "p3": _p3_empty()}
    if not path.exists():
        warns.append("장부 파일이 없습니다 (아직 기록 없음)")
        return out
    df = _read(path)
    out["n"] = int(len(df))
    if len(df) == 0:
        warns.append("장부에 행이 없습니다")
        return out
    for c in ALL_OUTCOME_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    out["n_with_outcome"] = int(df["y_sign_20"].notna().sum())
    out["n_with_outcome_60"] = int(df["y_sign_60"].notna().sum())
    out["first_asof"] = str(df["asof"].min())
    out["last_asof"] = str(df["asof"].max())
    last = df.sort_values("asof").iloc[-1]
    out["last_recorded_at_utc"] = None if _is_missing(last["recorded_at_utc"]) else str(last["recorded_at_utc"])
    out["variants"] = sorted(str(v) for v in df["variant"].dropna().unique())
    out["overall"] = _tone_block(df)
    out["base_rate_20"] = out["overall"]["base_20"]
    out["base_rate_60"] = out["overall"]["base_60"]
    for t in TONES:
        sub = df[df["tone"] == t]
        if len(sub):
            out["by_tone"][t] = _tone_block(sub)
    unknown = sorted(set(df["tone"].dropna().unique()) - set(TONES))
    if unknown:
        warns.append(f"알 수 없는 tone 값이 장부에 있습니다: {unknown}")
    for v in out["variants"]:
        sub = df[df["variant"] == v]
        blk = _tone_block(sub)
        blk["by_tone"] = {t: _tone_block(sub[sub["tone"] == t]) for t in TONES if (sub["tone"] == t).any()}
        blk["last_asof"] = str(sub["asof"].max())
        out["by_variant"][v] = blk
    # 결측일: 첫 기록일~마지막 기록일 사이 거래일인데 행이 하나도 없는 날
    have = set(pd.to_datetime(df["asof"]).dt.normalize())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        td, method = _trading_days_between(pd.Timestamp(out["first_asof"]), pd.Timestamp(out["last_asof"]), trading_days)
    for w in caught:
        warns.append(str(w.message))
    missing = [d.strftime("%Y-%m-%d") for d in td if d not in have]
    out["missing_days"] = missing
    out["n_missing_days"] = len(missing)
    out["missing_method"] = method
    if missing:
        warns.append(f"결측 거래일 {len(missing)}일 ({method})")
    if out["n_with_outcome"] == 0:
        warns.append("아직 결과(20일)가 확정된 행이 없습니다")
    # 해제 파일은 장부와 같은 디렉터리(=results/)에서 찾는다 — --results-dir 로 옮긴 실행도 그 트리의 상태를 본다
    unlock = Path(path).parent / HOLDOUT_UNLOCK_PATH.name
    # Phase 2 라이브 채점 (P2 행이 없으면 빈 블록)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out["p2"] = _p2_block(df, spy_close, unlock_path=unlock)
    for w in caught:
        warns.append(f"p2: {w.message}")
    # Phase 3 (§9): track.summary_p3 에 위임한다 — 킬룰·경보·패널의 단일 원천이 두 군데가 되지 않게.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out["p3"] = _p3_block(df, spy_close, unlock_path=unlock, reference=reference, prev_kill=prev_kill,
                              deploy_sizing=deploy_sizing, recomputed=recomputed)
    for w in caught:
        warns.append(f"p3: {w.message}")
    return out


# ------------------------------------------------------------------
# Phase 2 라이브 채점 (summary()["p2"])
# ------------------------------------------------------------------
_P2_BOOT_BLOCK = int(P2["boot_block"])      # 40 — ARCHITECTURE_PHASE2.md §3 (사전 등록 상수)
_P2_N_BOOT = int(P2["n_boot"])              # 4000
# 아래 두 상수는 VALIDATION.md §7 **본문**(문자 그대로의 카운트다운)이다. Phase 3 의 2단계 판정은
# 같은 절의 실행 해석 각주(장부 #7)이며 `mrl/track.py`(KILL_P3)가 맡는다 — 여기서는 세기만 한다.
_P2_KILL_EPISODES = 8       # VALIDATION.md §7 본문: 실현 ≥5% 에피소드 8회 이상
_P2_KILL_MONTHS = 36        # VALIDATION.md §7 본문: 3년 경과 — 둘 중 늦은 쪽에 평가
_P2_SCORE_REFS = (("clim", "p2_clim"), ("m1", "p2_p_m1"), ("vix", "p2_p_vix"), ("vix_bgk", "p2_p_vix_bgk"),
                  ("m3", "p2_p_m3"))
# 배포 확률이 그 기준과 같은 열에서 나오면(예: M1 배포 → prob_dd5_20 == p2_p_m1) BSS 는 정의상 0 이다 — 숨기지 않고 사유를 남긴다
_P2_REF_RUNG = {"m1": "M1", "m3": "M3"}


def _jsonable_float(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


def _p2_empty() -> dict:
    d = {"schema_version": SCHEMA_VERSION, "n_rows": 0, "n_scored": 0, "n_blocks": 0, "first_asof": None, "last_asof": None,
         "variant_used": None, "holdout_locked": False, "n_input_missing": 0, "last_state": None, "last_days_in_state": None, "last_deploy_mode": None,
         "last_tone_model": None, "last_model_id": None, "last_prob_model_id": None, "last_prob": None, "prob_sources": {},
         "base_rate": None, "mean_p": None, "brier": None, "n_ref": {}, "ci_bss_clim": None,
         "ci_method": f"순환 블록 부트스트랩(block={_P2_BOOT_BLOCK}, n_boot={_P2_N_BOOT}, seed=0) — n_scored ≥ {_P2_BOOT_BLOCK} 일 때만",
         "episodes5_observed": 0, "episodes5_method": None, "months_elapsed": None,
         "kill_rule": {"episodes_required": _P2_KILL_EPISODES, "months_required": _P2_KILL_MONTHS,
                       "episodes_ok": False, "months_ok": False, "due": False, "verdict": None,
                       "rule": "실현 ≥5% 에피소드 8회 이상 그리고 3년 경과(늦은 쪽)에 bss_clim(블록 부트스트랩 95% 구간) "
                               "평가 — VALIDATION.md §7 본문의 문자 그대로의 카운트다운. 실제 판정(2단계 36/60개월·"
                               "12개월 재평가·점추정 트리거)은 같은 절의 실행 해석 각주(장부 #7)이며 track.kill_status 가 낸다"},
         "kill_rule_due": False, "notes": []}
    for k, _ in _P2_SCORE_REFS:
        d[f"brier_{k}"] = None
        d[f"bss_{k}"] = None
    return d


def _bss_block_ci(loss_p: np.ndarray, loss_ref: np.ndarray, block: int = _P2_BOOT_BLOCK, n_boot: int = _P2_N_BOOT,
                  ci: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """짝지은(paired) 순환 블록 부트스트랩으로 BSS = 1 − mean(loss_p)/mean(loss_ref) 의 백분위 구간.
    evaluate.block_bootstrap_ci 와 같은 블록 추출(시작점 0..n-1, 끝에서 감아 붙임)을 두 손실에 동시에 적용한다
    (비율이라 평균의 구간을 그대로 못 쓴다). 재표본에서 mean(loss_ref)==0 인 경우는 제외."""
    lp = np.asarray(loss_p, dtype=float)
    lr = np.asarray(loss_ref, dtype=float)
    n = len(lp)
    if n == 0 or len(lr) != n:
        return (np.nan, np.nan)
    b = min(int(block), n)
    k = int(math.ceil(n / b))
    rng = np.random.default_rng(seed)
    offs = np.arange(b)
    bss = np.empty(n_boot)
    chunk = max(1, min(n_boot, int(2_000_000 // max(k * b, 1))))
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        starts = rng.integers(0, n, size=(m, k))
        sel = ((starts[:, :, None] + offs[None, None, :]) % n).reshape(m, k * b)[:, :n]
        mp = lp[sel].mean(axis=1)
        mr = lr[sel].mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            bss[done: done + m] = np.where(mr > 0, 1.0 - mp / mr, np.nan)
        done += m
    bss = bss[np.isfinite(bss)]
    if len(bss) == 0:
        return (np.nan, np.nan)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(bss, [alpha, 1.0 - alpha])
    return (float(lo), float(hi))


def _prob_source(deploy_mode, tone_model) -> str:
    """그 행의 prob_dd5_20 이 어느 단에서 나왔는지: tones → 배포 단(tone_model), info_only → M3(정보), 그 밖 → 'unknown'."""
    dm = None if _is_missing(deploy_mode) else str(deploy_mode)
    tm = None if _is_missing(tone_model) else str(tone_model)
    if dm == "tones" and tm:
        return f"{tm}(배포)"
    if dm == "info_only":
        return "M3(정보 표시)"
    if dm is None and tm is None:
        return "unknown"
    return f"{dm or '—'}/{tm or '—'}"


def _count_runs(flags: np.ndarray) -> int:
    """1 로 이어진 런의 수 (y_dd5_20 런 = 낙폭 에피소드 근사)."""
    f = np.asarray(flags, dtype=float)
    f = np.nan_to_num(f, nan=0.0) > 0.5
    if len(f) == 0:
        return 0
    return int(f[0]) + int(np.sum(f[1:] & ~f[:-1]))


def _p2_block(df: pd.DataFrame, spy_close=None, unlock_path=HOLDOUT_UNLOCK_PATH) -> dict:
    """summary()["p2"]: 장부의 P2 행으로 라이브 Brier·skill(기후학·M1·B1·BGK 대비)·CI·킬룰 카운트다운을 계산한다.

    * 홀드아웃 보호(VALIDATION.md §6): unlock_path 가 없으면 HOLDOUT_START 이후 행은 **채점하지 않는다**.
      라이브 세션은 전부 홀드아웃 구간 안이라(HOLDOUT_START=2024-09-01, 끝 없음) 여기서 채점·공표하면
      run_calibration --holdout-final 이 "오염되지 않은 1회" 로 채점할 세션을 미리 쓰는 셈이 된다.
      unlock_path=None 을 주면 검사를 끈다(테스트·사후 분석용).
    * P2 행 = P2 열(또는 prob_dd5_20) 중 하나라도 값이 있는 행. 'completed' 변형 행이 있으면 그것만 쓴다(정식 기록).
    * 채점 행 = prob_dd5_20 과 y_dd5_20 이 모두 있는 행(20거래일 뒤 backfill 이 채운 뒤에만 누적).
    * 각 기준(ref)의 skill 은 ref 도 있는 행에서 같은 행끼리 계산: bss = 1 − mean((p−y)²)/mean((ref−y)²).
    * ci_bss_clim 은 n_scored ≥ 40 일 때만(블록 하나도 못 채우면 순환 재표본이 원본과 같아 구간이 퇴화한다).
    * 킬룰(VALIDATION.md §7): 실현 ≥5% 에피소드 ≥8 그리고 36개월 경과(늦은 쪽) → due. verdict 는 due 일 때
      bss_clim > 0 이면 'pass', 아니면 'info_only'(톤·비중 제안 숨김) — 최종 판단은 장부 §8 에 소유자가 기재.
    """
    out = _p2_empty()
    notes: list[str] = out["notes"]
    cols = P2_COLUMNS + ["prob_dd5_20"]
    d = df.copy()
    for c in P2_NUMERIC_COLUMNS + ("prob_dd5_20", "y_dd5_20"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    is_p2 = d[cols].notna().any(axis=1)
    p2 = d[is_p2]
    if len(p2) == 0:
        notes.append("장부에 P2 행이 없습니다")
        return out
    if (p2["variant"] == "completed").any():
        p2 = p2[p2["variant"] == "completed"]
        out["variant_used"] = "completed"
    else:
        out["variant_used"] = "all"
    p2 = p2.sort_values(["asof", "variant"], kind="stable").drop_duplicates("asof", keep="first")
    out["n_rows"] = int(len(p2))
    out["first_asof"] = str(p2["asof"].iloc[0])
    out["last_asof"] = str(p2["asof"].iloc[-1])
    last = p2.iloc[-1]
    out["last_state"] = None if _is_missing(last["p2_state"]) else str(last["p2_state"])
    out["last_days_in_state"] = None if _is_missing(last["p2_days_in_state"]) else int(float(last["p2_days_in_state"]))
    out["last_deploy_mode"] = None if _is_missing(last["p2_deploy_mode"]) else str(last["p2_deploy_mode"])
    out["last_tone_model"] = None if _is_missing(last["p2_tone_model"]) else str(last["p2_tone_model"])
    out["last_model_id"] = None if _is_missing(last["p2_model_id"]) else str(last["p2_model_id"])
    out["last_prob_model_id"] = None if _is_missing(last["p2_prob_model_id"]) else str(last["p2_prob_model_id"])
    out["last_prob"] = _jsonable_float(last["prob_dd5_20"])
    out["n_input_missing"] = int(p2["p2_input_missing"].notna().sum())
    # prob_dd5_20 의 출처(배포 단) 분포 — 장부가 여러 단을 섞고 있으면 드러낸다(조용한 실패 금지)
    src = p2.apply(lambda r: _prob_source(r.get("p2_deploy_mode"), r.get("p2_tone_model")), axis=1)
    out["prob_sources"] = {str(k): int(v) for k, v in src.value_counts().items()}
    if len(out["prob_sources"]) > 1:
        notes.append(f"장부의 prob_dd5_20 이 여러 출처에서 나왔습니다: {out['prob_sources']} — "
                     "배포 단이 바뀐 날 이전·이후를 한 계열로 읽지 마세요(행마다 p2_deploy_mode·p2_tone_model 참조)")

    # 홀드아웃 보호(VALIDATION.md §6): 해제 파일이 없으면 HOLDOUT_START 이후 세션은 채점하지 않는다.
    p2_scorable = p2
    out["holdout_locked"] = False
    if unlock_path is not None and not Path(unlock_path).exists():
        in_ho = pd.to_datetime(p2["asof"], errors="coerce") >= pd.Timestamp(HOLDOUT_START)
        if bool(in_ho.any()):
            out["holdout_locked"] = True
            notes.append(f"홀드아웃 미해제({Path(unlock_path).name} 없음) → {HOLDOUT_START} 이후 {int(in_ho.sum())}행은 "
                         "라이브 채점·킬룰에서 제외 (최종 검증 1회 전 접근 금지 — VALIDATION.md §6)")
        p2_scorable = p2[~in_ho]

    scored = p2_scorable[p2_scorable["prob_dd5_20"].notna() & p2_scorable["y_dd5_20"].notna()]
    out["n_scored"] = int(len(scored))
    out["n_blocks"] = int(len(scored) // _DD5_WINDOW)
    if len(scored) == 0:
        notes.append("홀드아웃 미해제 → 라이브 채점 보류(해제 후 시작)" if out["holdout_locked"]
                     else "아직 채점된 P2 행이 없습니다 (y_dd5_20 확정 전)")
    else:
        p = scored["prob_dd5_20"].to_numpy(dtype=float)
        y = scored["y_dd5_20"].to_numpy(dtype=float)
        loss_p = (p - y) ** 2
        out["brier"] = float(loss_p.mean())
        out["base_rate"] = float(y.mean())
        out["mean_p"] = float(p.mean())
        for name, col in _P2_SCORE_REFS:
            ok = scored[col].notna().to_numpy()
            n_ref = int(ok.sum())
            out["n_ref"][name] = n_ref
            if n_ref == 0:
                notes.append(f"{col} 이 없어 {name} 대비 skill 을 계산할 수 없습니다")
                continue
            ref = scored[col].to_numpy(dtype=float)[ok]
            loss_ref = (ref - y[ok]) ** 2
            b_ref = float(loss_ref.mean())
            out[f"brier_{name}"] = b_ref
            if b_ref <= 0:
                notes.append(f"{name} 기준 Brier 가 0 이라 skill 을 정의할 수 없습니다")
                continue
            out[f"bss_{name}"] = float(1.0 - loss_p[ok].mean() / b_ref)
            rung = _P2_REF_RUNG.get(name)
            if rung and np.allclose(ref, p[ok], atol=1e-12, equal_nan=True):
                notes.append(f"{name} 대비 skill 은 정의상 0 입니다 — 배포 확률(prob_dd5_20)이 {rung} 그 자체이기 때문"
                             f"(출처 {out['prob_sources']})")
            if name == "clim":
                if n_ref >= _P2_BOOT_BLOCK:
                    lo, hi = _bss_block_ci(loss_p[ok], loss_ref)
                    out["ci_bss_clim"] = None if (math.isnan(lo) or math.isnan(hi)) else [lo, hi]
                else:
                    notes.append(f"ci_bss_clim: 채점 행 {n_ref} < 블록 {_P2_BOOT_BLOCK} → 구간 생략")
    # 킬룰 카운트다운 — §7 시계는 채점이 정당하게 시작되는 시점부터 (홀드아웃 잠금 중이면 아직 시작하지 않는다)
    if len(p2_scorable) == 0:
        notes.append("킬룰 카운트다운 보류: 채점 가능한 P2 행이 없습니다(홀드아웃 미해제)")
        out["kill_rule_due"] = False
        for k in (("brier", "base_rate", "mean_p", "months_elapsed", "last_prob")
                  + tuple(f"brier_{n}" for n, _ in _P2_SCORE_REFS) + tuple(f"bss_{n}" for n, _ in _P2_SCORE_REFS)):
            out[k] = _jsonable_float(out[k])
        return out
    first_ts = pd.Timestamp(str(p2_scorable["asof"].iloc[0]))
    last_ts = pd.Timestamp(str(p2_scorable["asof"].iloc[-1] if out["holdout_locked"] else df["asof"].max()))
    out["months_elapsed"] = round(float((last_ts - first_ts).days) / 30.4375, 2)
    n_ep = None
    if spy_close is not None:
        try:
            from mrl.targets import episodes as _episodes
            s = _prep_close(spy_close)
            ep = _episodes(s, 0.05, split=False)
            n_ep = int((pd.to_datetime(ep["trough_date"]) >= first_ts).sum()) if len(ep) else 0
            out["episodes5_method"] = "targets.episodes(spy_close, 0.05, split=False) 중 trough_date ≥ P2 첫 기록일"
        except Exception as e:   # 계산 실패는 숨기지 않고 근사로 내려가며 사유를 남긴다
            notes.append(f"episodes5: spy_close 로 계산 실패({type(e).__name__}: {e}) → y_dd5_20 런 수로 근사")
            n_ep = None
    if n_ep is None:
        n_ep = _count_runs(p2_scorable["y_dd5_20"].to_numpy(dtype=float))
        out["episodes5_method"] = "장부 y_dd5_20 런 수(근사; spy_close 를 주면 에피소드 표로 센다)"
    out["episodes5_observed"] = int(n_ep)
    kr = out["kill_rule"]
    kr["episodes_ok"] = bool(n_ep >= _P2_KILL_EPISODES)
    kr["months_ok"] = bool(out["months_elapsed"] >= _P2_KILL_MONTHS)
    kr["due"] = bool(kr["episodes_ok"] and kr["months_ok"])
    kr["bss_clim"] = out["bss_clim"]
    kr["ci_bss_clim"] = out["ci_bss_clim"]
    if kr["due"]:
        kr["verdict"] = "pass" if (out["bss_clim"] is not None and out["bss_clim"] > 0) else "info_only"
    out["kill_rule_due"] = kr["due"]
    # JSON 안전(NaN 없음)
    for k in (("brier", "base_rate", "mean_p", "months_elapsed", "last_prob")
              + tuple(f"brier_{n}" for n, _ in _P2_SCORE_REFS) + tuple(f"bss_{n}" for n, _ in _P2_SCORE_REFS)):
        out[k] = _jsonable_float(out[k])
    return out


# ------------------------------------------------------------------
# Phase 3 블록 (summary()["p3"]) — ARCHITECTURE_PHASE3.md §9
#   계산은 하지 않는다: `mrl/track.py::summary_p3` 가 킬룰·경보·패널의 단일 원천이고, 여기서는 장부를
#   그 계약대로 넘겨 주고(홀드아웃 게이트·asof·배치 모드) 결과를 그대로 싣는다.
#   §15 단계 0: p2 가 info_only 면 유효 모드도 info_only 이고 비중·상태 카드는 뜨지 않는다 —
#   그 판단은 여기서 하지 않고 effective_mode(= track) 가 낸 값을 그대로 전달한다.
# ------------------------------------------------------------------
def _p3_empty(note: str | None = None) -> dict:
    d = {"schema_version": SCHEMA_VERSION, "n_rows": 0, "n_scored": 0, "n_eff": 0.0, "live_start": None,
         "months_elapsed": 0, "effective_mode": None, "deploy_sizing": None,
         "sizing": {"avg_w": None, "n_changes_252": 0, "realized_vol_ratio": None, "rule_dd": None,
                    "rule_vs_bh_20": None},
         "members": {}, "disagreement": {"median_width": None, "p90_width": None, "flag_sessions": 0},
         "kill": {}, "alarms": [], "replay_mismatch_count": None, "coverage": {},
         "holdout_locked": False, "notes": []}
    if note:
        d["notes"].append(note)
    return d


def _last_str(df: pd.DataFrame, col: str):
    """마지막(asof 순) 행의 문자열 값. 열이 없거나 전부 결측이면 None."""
    if col not in df.columns:
        return None
    s = df.sort_values("asof", kind="stable")[col]
    s = s[s.notna()]
    return None if len(s) == 0 else str(s.iloc[-1])


def _p3_block(df: pd.DataFrame, spy_close=None, *, unlock_path=HOLDOUT_UNLOCK_PATH, reference=None,
              prev_kill=None, deploy_sizing=None, recomputed=None) -> dict:
    """summary()["p3"] — `track.summary_p3` 위임. 실패해도 summary() 전체를 무너뜨리지 않되 **조용하지 않다**
    (warnings.warn + notes 에 사유; 호출자는 summary()["warnings"] 에서 'p3: ...' 로 본다)."""
    if df is None or len(df) == 0:
        return _p3_empty("장부에 행이 없습니다")
    try:
        from mrl import track
    except Exception as e:                                    # 배선 전(모듈 부재)에도 장부 요약은 나와야 한다
        warnings.warn(f"mrl.track 을 불러올 수 없어 p3 블록을 비웁니다 ({type(e).__name__}: {e})")
        return _p3_empty(f"mrl.track 불러오기 실패({type(e).__name__}: {e})")

    d = df.copy()
    # 정식 기록은 'completed' 변형 — _p2_block 과 같은 규칙으로 하나만 고른다(변형을 섞어 채점하지 않는다).
    if "variant" in d.columns and (d["variant"].astype("string") == "completed").any():
        d = d[d["variant"].astype("string") == "completed"]
    d = d.sort_values(["asof", "variant"], kind="stable").drop_duplicates("asof", keep="first")
    asof = str(d["asof"].max())
    try:
        out = track.summary_p3(d, spy_close, reference, asof, prev_kill=prev_kill, unlock_path=unlock_path,
                               recomputed=recomputed, p2_deploy_mode=_last_str(d, "p2_deploy_mode"),
                               p3_deploy_mode=_last_str(d, "p3_deploy_mode"), deploy_sizing=deploy_sizing)
    except Exception as e:                                    # 조용한 실패 금지: 사유를 크게 남기고 빈 블록
        warnings.warn(f"track.summary_p3 실패 ({type(e).__name__}: {e}) → p3 블록을 비웁니다")
        return _p3_empty(f"track.summary_p3 실패({type(e).__name__}: {e})")
    out.setdefault("notes", [])
    n_p3 = int(d[[c for c in P3_COLUMNS if c in d.columns]].notna().any(axis=1).sum()) if len(d) else 0
    out["n_p3_rows"] = n_p3
    if n_p3 == 0:
        out["notes"].append("장부에 P3 행이 없습니다 (schema 2 기록만 있음 — P3 열은 빈 값으로 승격되어 있습니다)")
    return out
