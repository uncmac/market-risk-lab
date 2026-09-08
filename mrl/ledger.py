# -*- coding: utf-8 -*-
"""매일 판정 기록 장부 (track record).

ARCHITECTURE.md 계약:
    LEDGER = RESULTS_DIR / "track_record.csv"
    append_today(row, path=LEDGER) -> bool      # asof 중복이면 False
    backfill(path, spy_close) -> DataFrame      # 20/60일 지난 행의 결과 열 채움
    summary(path) -> dict                       # 표본 수·적중·기준선·마지막 기록일·결측일

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

from mrl.config import RESULTS_DIR, TONES, V0_SIGNALS

LEDGER = RESULTS_DIR / "track_record.csv"

# 열 순서(계약): asof, recorded_at_utc, variant, states(8), score_d/w/m, overall_d/w/m, tone,
#                spy_close, vix_close, prob_dd5_20(None; Phase 2), run_id  + 결과 열
STATE_COLUMNS = [f"state_{k}" for k in V0_SIGNALS]
CORE_COLUMNS = (["asof", "recorded_at_utc", "variant"] + STATE_COLUMNS
                + ["score_d", "score_w", "score_m", "overall_d", "overall_w", "overall_m", "tone",
                   "spy_close", "vix_close", "prob_dd5_20", "run_id"])
OUTCOME_COLUMNS = ["y_sign_20", "y_sign_60", "y_dd5_20", "fwd_ret_20", "fwd_ret_60"]
LEDGER_COLUMNS = CORE_COLUMNS + OUTCOME_COLUMNS

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


def summary(path=LEDGER, trading_days=None) -> dict:
    """장부 요약 (JSON 직렬화 가능).

    키: exists, path, n, n_with_outcome, n_with_outcome_60, first_asof, last_asof, last_recorded_at_utc,
        variants(list), by_tone{tone: {...}}, by_variant{variant: {...}}, overall{...}, base_rate_20/60,
        missing_days(list[str]), n_missing_days, missing_method, warnings(list[str])
    trading_days: 결측일 판정에 쓸 거래일 인덱스(선택). 없으면 calendar_us → 평일 순으로 폴백.
    """
    path = Path(path)
    warns: list[str] = []
    out = {"exists": path.exists(), "path": str(path), "n": 0, "n_with_outcome": 0, "n_with_outcome_60": 0,
           "first_asof": None, "last_asof": None, "last_recorded_at_utc": None, "variants": [],
           "by_tone": {}, "by_variant": {}, "overall": None, "base_rate_20": None, "base_rate_60": None,
           "missing_days": [], "n_missing_days": 0, "missing_method": None, "warnings": warns}
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
    return out
