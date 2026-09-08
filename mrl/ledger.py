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
    schema_version 은 2 유지: 새 열은 뒤에 덧붙였을 뿐 기존 열의 순서·이름을 바꾸지 않는다(3 은 Phase 3 예약).

설계 원칙
* CSV 한 파일이 진실. 매 호출마다 전체를 읽고 통째로 다시 쓴다(행 수가 작아 충분).
* 점(point-in-time) 원칙: 결과 열은 asof 이후 h 거래일이 실제로 지난 행에만 채운다.
* 조용한 실패 금지: 형식 오류는 예외, 데이터 부족은 warnings.warn + 반환 dict 의 warnings.
* 중복 판정 키는 (asof, variant). 같은 날 faithful·completed 두 변형을 나란히 기록하기 위함
  (계약 문구 "asof 중복"의 실질적 의미: 같은 변형의 같은 날은 한 번만).
"""
from __future__ import annotations

import math
import warnings
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from mrl.config import HOLDOUT_START, HOLDOUT_UNLOCK_PATH, P2, P2_STATES, RESULTS_DIR, TONES, V0_SIGNALS

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
SCHEMA_VERSION = 2
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
LEDGER_COLUMNS = LEDGER_COLUMNS_V1 + P2_COLUMNS             # schema_version=2 의 전체 열(알 수 없는 추가 열은 그 뒤에 보존)

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
    """장부 CSV 읽기. 없으면 빈 DataFrame(계약 열). 열이 부족하면 NaN 으로 보강."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    df = pd.read_csv(path, dtype={"asof": str, "variant": str, "tone": str, "run_id": str,
                                  **{c: str for c in STATE_COLUMNS},
                                  **{c: str for c in P2_STRING_COLUMNS},
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

    # 결과 열은 append 시점엔 항상 비어 있음(backfill 이 채움). 입력에 있어도 무시하고 경고.
    given = [c for c in OUTCOME_COLUMNS if c in row and not _is_missing(row[c])]
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


def backfill(path, spy_close) -> pd.DataFrame:
    """asof 로부터 20/60 거래일이 지난 행의 y_sign_20/60, y_dd5_20, fwd_ret_20/60 을 채운다.

    * 정의는 mrl/targets.py 와 동일: fwd_ret_h = close[t+h]/close[t]-1, y_sign_h = fwd_ret_h > 0,
      y_dd5_20 = min(close[t+1..t+20])/close[t]-1 <= -0.05. 위치(거래일 수) 기준.
    * 기준 종가는 장부의 spy_close 가 아니라 전달된 spy_close 시계열의 asof 값(같은 조정 기준 유지).
    * 이미 채워진 값은 다시 쓰지 않는다(기록 안정성). asof 가 spy_close 인덱스에 없으면 경고 후 건너뜀.
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
    for c in OUTCOME_COLUMNS:
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


def summary(path=LEDGER, trading_days=None, spy_close=None) -> dict:
    """장부 요약 (JSON 직렬화 가능).

    키: exists, path, schema_version, n, n_with_outcome, n_with_outcome_60, first_asof, last_asof, last_recorded_at_utc,
        variants(list), by_tone{tone: {...}}, by_variant{variant: {...}}, overall{...}, base_rate_20/60,
        missing_days(list[str]), n_missing_days, missing_method, warnings(list[str]),
        p2{...} (Phase 2 라이브 Brier·킬룰 카운트다운 — _p2_block 참조; P2 행이 없으면 n_rows=0 인 빈 블록)
    trading_days: 결측일 판정에 쓸 거래일 인덱스(선택). 없으면 calendar_us → 평일 순으로 폴백.
    spy_close: (선택) SPY 종가 시계열 — 있으면 p2.episodes5_observed 를 targets.episodes(0.05, split=False) 로 센다.
               없으면 장부의 y_dd5_20 런 수로 근사(p2.episodes5_method 에 표기).
    """
    path = Path(path)
    warns: list[str] = []
    out = {"exists": path.exists(), "path": str(path), "schema_version": SCHEMA_VERSION,
           "n": 0, "n_with_outcome": 0, "n_with_outcome_60": 0,
           "first_asof": None, "last_asof": None, "last_recorded_at_utc": None, "variants": [],
           "by_tone": {}, "by_variant": {}, "overall": None, "base_rate_20": None, "base_rate_60": None,
           "missing_days": [], "n_missing_days": 0, "missing_method": None, "warnings": warns,
           "p2": _p2_empty()}
    if not path.exists():
        warns.append("장부 파일이 없습니다 (아직 기록 없음)")
        return out
    df = _read(path)
    out["n"] = int(len(df))
    if len(df) == 0:
        warns.append("장부에 행이 없습니다")
        return out
    for c in OUTCOME_COLUMNS:
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
    # Phase 2 라이브 채점 (P2 행이 없으면 빈 블록)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # 해제 파일은 장부와 같은 디렉터리(=results/)에서 찾는다 — --results-dir 로 옮긴 실행도 그 트리의 상태를 본다
        out["p2"] = _p2_block(df, spy_close, unlock_path=Path(path).parent / HOLDOUT_UNLOCK_PATH.name)
    for w in caught:
        warns.append(f"p2: {w.message}")
    return out


# ------------------------------------------------------------------
# Phase 2 라이브 채점 (summary()["p2"])
# ------------------------------------------------------------------
_P2_BOOT_BLOCK = int(P2["boot_block"])      # 40 — ARCHITECTURE_PHASE2.md §3 (사전 등록 상수)
_P2_N_BOOT = int(P2["n_boot"])              # 4000
_P2_KILL_EPISODES = 8       # VALIDATION.md §7: 실현 ≥5% 에피소드 8회 이상
_P2_KILL_MONTHS = 36        # VALIDATION.md §7: 3년 경과 — 둘 중 늦은 쪽에 평가
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
                       "rule": "실현 ≥5% 에피소드 8회 이상 그리고 3년 경과(늦은 쪽)에 bss_clim(블록 부트스트랩 95% 구간) 평가 — VALIDATION.md §7"},
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
