# -*- coding: utf-8 -*-
"""목표변수·에피소드 표 (VALIDATION.md §1·§2 와 동일, ARCHITECTURE.md 계약).

계약:
    make_targets(spy_close) -> DataFrame
    base_rates(targets, start=None, end=None) -> Series
    episodes(spy_close, threshold) -> DataFrame
    independent_blocks(n_days, h) -> int

설계 원칙
* 점(point-in-time) 원칙: 날짜 t 의 목표변수는 t 이후 h 거래일의 실현값이므로, 시계열 끝에서
  h 일이 아직 지나지 않은 행은 NaN 으로 남긴다(부분 창으로 채우지 않는다). y_vol_20 의 기준값
  (직전 1년 20일-RV 중앙값)은 t 까지의 데이터만 쓴다.
* 이진 목표변수(y_*)는 float 0.0/1.0 + NaN 으로 저장한다. bool 은 NaN 을 담을 수 없고,
  float 이면 .mean() 이 곧 기저율이 된다.
* 조용한 실패 금지: 입력 형식 오류(중복 인덱스·비단조·0 이하 가격)는 ValueError.
  단, SPY 열이 BTC 주말 union 인덱스 때문에 갖는 NaN 행은 구조적인 것이므로 제거한다(문서화된 동작).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from mrl.config import DD_TARGETS, HORIZONS

# 열 이름 계약 (다른 모듈이 의존)
SIGN_COLUMNS = tuple(f"y_sign_{h}" for h in HORIZONS)          # y_sign_5, y_sign_20, y_sign_60
DD_COLUMNS = tuple(DD_TARGETS.keys())                          # y_dd5_20, y_dd10_60
VOL_COLUMNS = ("y_vol_20",)
Y_COLUMNS = SIGN_COLUMNS + DD_COLUMNS + VOL_COLUMNS
FWD_RET_COLUMNS = tuple(f"fwd_ret_{h}" for h in HORIZONS)      # fwd_ret_5, fwd_ret_20, fwd_ret_60
FWD_MAXDD_COLUMNS = tuple(f"fwd_maxdd_{w}" for _, w in DD_TARGETS.values())   # fwd_maxdd_20, fwd_maxdd_60
TARGET_COLUMNS = Y_COLUMNS + FWD_RET_COLUMNS + FWD_MAXDD_COLUMNS

EPISODE_COLUMNS = ("peak_date", "trough_date", "depth", "days_to_trough",
                   "recovery_date", "days_to_recover")

_RV_WINDOW = 20          # 실현변동성 창(거래일)
_RV_REF_WINDOW = 252     # 기준 중앙값을 구하는 직전 창(거래일, ≈1년)
_ANNUALIZE = np.sqrt(252.0)


# ------------------------------------------------------------------
# 입력 정규화
# ------------------------------------------------------------------
def _clean_close(spy_close: pd.Series, name: str = "spy_close") -> pd.Series:
    """종가 시계열을 검증·정규화한다: NaN 행 제거, DatetimeIndex, 오름차순, 중복 없음, 양수."""
    if not isinstance(spy_close, pd.Series):
        raise TypeError(f"{name} 는 pd.Series 여야 합니다 (받은 형: {type(spy_close).__name__})")
    s = spy_close.dropna()
    if len(s) == 0:
        raise ValueError(f"{name} 에 유효한(비-NaN) 값이 없습니다")
    if not isinstance(s.index, pd.DatetimeIndex):
        try:
            s.index = pd.DatetimeIndex(pd.to_datetime(s.index))
        except Exception as e:      # noqa: BLE001 - 형식 오류를 명시적으로 알린다
            raise TypeError(f"{name} 인덱스를 DatetimeIndex 로 바꿀 수 없습니다: {e}") from e
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    if s.index.has_duplicates:
        dup = s.index[s.index.duplicated()][:3].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"{name} 인덱스에 중복 날짜가 있습니다 (예: {dup})")
    if not s.index.is_monotonic_increasing:
        raise ValueError(f"{name} 인덱스가 오름차순이 아닙니다")
    s = s.astype(float)
    if (s <= 0).any():
        bad = s.index[s <= 0][:3].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"{name} 에 0 이하 가격이 있습니다 (예: {bad})")
    s.name = spy_close.name if spy_close.name is not None else name
    return s


# ------------------------------------------------------------------
# 목표변수
# ------------------------------------------------------------------
def _forward_min(arr: np.ndarray, h: int) -> np.ndarray:
    """out[t] = min(arr[t+1 .. t+h]). 마지막 h 개 행은 NaN (창이 완전히 실현된 행만)."""
    n = len(arr)
    out = np.full(n, np.nan)
    if n > h:
        win = sliding_window_view(arr[1:], h)      # 행 j 가 arr[j+1 .. j+h]
        out[: n - h] = win.min(axis=1)
    return out


def make_targets(spy_close: pd.Series) -> pd.DataFrame:
    """SPY 종가 → 목표변수 표 (인덱스 = spy_close 의 거래일).

    열 (모두 float; 실현되지 않은 창은 NaN):
      y_sign_5/20/60   : close[t+h]/close[t]-1 > 0  →  1.0/0.0
      y_dd5_20         : min(close[t+1..t+20])/close[t]-1 <= -0.05
      y_dd10_60        : min(close[t+1..t+60])/close[t]-1 <= -0.10
      y_vol_20         : 다음 20일 실현변동성(로그수익률 std·√252) > 직전 252일의 20일-RV 중앙값
                         (기준 중앙값은 t 까지의 값만 사용; 초기 271행은 NaN)
      fwd_ret_5/20/60  : close[t+h]/close[t]-1 (연속값)
      fwd_maxdd_20/60  : min(close[t+1..t+w])/close[t]-1 (연속값, 보통 음수·양수 가능)
    마지막 h(창 길이) 행은 NaN — 부분 창으로 채우지 않는다.
    """
    s = _clean_close(spy_close)
    arr = s.to_numpy(dtype=float)
    out = pd.DataFrame(index=s.index)

    # 방향·선행수익
    for h in HORIZONS:
        fwd = s.shift(-h) / s - 1.0
        out[f"fwd_ret_{h}"] = fwd
        out[f"y_sign_{h}"] = (fwd > 0).astype(float).where(fwd.notna())

    # 낙폭
    for col, (thr, w) in DD_TARGETS.items():
        fmin = _forward_min(arr, w)
        maxdd = pd.Series(fmin / arr - 1.0, index=s.index)
        out[f"fwd_maxdd_{w}"] = maxdd
        out[col] = (maxdd <= -thr).astype(float).where(maxdd.notna())

    # 변동성: 20일 RV(로그수익률 std·√252). rv20[t] 는 r[t-19..t] 를 쓰므로 t 까지만 사용.
    logret = np.log(s).diff()
    rv20 = logret.rolling(_RV_WINDOW, min_periods=_RV_WINDOW).std() * _ANNUALIZE
    fwd_rv20 = rv20.shift(-_RV_WINDOW)                              # r[t+1..t+20] 의 std
    ref = rv20.rolling(_RV_REF_WINDOW, min_periods=_RV_REF_WINDOW).median()   # 직전 252개 20일-RV 중앙값
    valid = fwd_rv20.notna() & ref.notna()
    out["y_vol_20"] = (fwd_rv20 > ref).astype(float).where(valid)

    # 열 순서 고정
    out = out[list(TARGET_COLUMNS)]
    out.index.name = spy_close.index.name or "date"
    return out


def base_rates(targets: pd.DataFrame, start=None, end=None) -> pd.Series:
    """구간 [start, end] 의 이진 목표변수 기저율(각 열의 비-NaN 평균). 인덱스 = y_* 열 이름."""
    missing = [c for c in Y_COLUMNS if c not in targets.columns]
    if missing:
        raise ValueError(f"targets 에 열이 없습니다: {missing}")
    sub = targets.loc[start:end, list(Y_COLUMNS)]
    rates = sub.mean(axis=0, skipna=True)
    rates.name = "base_rate"
    return rates


# ------------------------------------------------------------------
# 에피소드
# ------------------------------------------------------------------
def episodes(spy_close: pd.Series, threshold: float, split: bool = True) -> pd.DataFrame:
    """고점→저점 낙폭이 threshold 이상인 구간 표.

    열: peak_date, trough_date, depth(= trough/peak - 1, 음수), days_to_trough(고점→저점 거래일 수),
        recovery_date(직전 고점(ATH)을 처음 넘어선 날; 미회복이면 NaT), days_to_recover(저점→회복 거래일 수; 미회복 NaN)

    정의(사전 등록, VALIDATION.md §2):
      * 기준 고점 = 종가 신고점(ATH). 종가가 ATH 대비 -threshold 이하로 처음 내려가면 에피소드 시작,
        종가가 ATH 를 넘어서면(엄격 초과) 회복·종료.
      * 병합 규칙(split=True, 계약 기본값): 저점 이후 신고점 전에 "다시 threshold 하락"하면 새 에피소드로
        세되 recovery_date 는 공유. 구현상 조작적 정의 —
          - 저점은 그 후 threshold 이상 반등(반등 고점 >= 저점×(1+threshold))이 있어야 확정된다.
            (반등 없이 이어지는 하락은 같은 에피소드의 연장이며 저점만 갱신된다.)
          - 확정된 저점 이후의 반등 고점에서 종가가 -threshold 이하로 재하락하면 새 에피소드
            (peak = 반등 고점, 그날부터 저점 추적). 이전 에피소드의 저점은 확정 시점의 저점.
          - 이 규칙은 회복(ATH 경신) 전에 반복 적용된다. 모든 조각의 recovery_date 는 ATH 경신일.
      * split=False: 분할 없이 ATH 기준 한 underwater 구간 = 한 에피소드(저점 = 구간 전체 최저 종가).
        VALIDATION.md §2 의 감사 실측치(1993~2026 SPY: ≥5% 36회 · ≥10% 12회 · ≥20% 4회)는 이 방식으로만
        정확히 재현된다(split=True 는 2000-02·2007-09 약세장을 여러 조각으로 세어 115 · 26 · 5 회).
        두 사전 등록 문장이 서로 맞지 않으므로 둘 다 제공하고, 어느 쪽을 쓰는지는 리포트에 명시할 것.
      * 시계열 끝까지 회복하지 못한 에피소드도 포함한다(recovery_date=NaT).
      * 동률 고점은 처음 도달한 날을 peak_date 로 쓴다.
    """
    if not (0 < threshold < 1):
        raise ValueError(f"threshold 는 (0,1) 사이여야 합니다: {threshold}")
    s = _clean_close(spy_close)
    arr = s.to_numpy(dtype=float)
    idx = s.index
    n = len(arr)

    rows: list[dict] = []          # 확정된 에피소드 (recovery 는 나중에 채움)
    pending: list[int] = []        # 같은 낙폭 구간 안에서 recovery_date 를 공유할 rows 의 위치

    ath_i = 0                      # 신고점 위치
    in_ep = False
    peak_i = trough_i = reb_i = 0  # 현재 에피소드의 고점·저점·(저점 이후) 반등 고점 위치

    def _close_current(recovery_i):
        rows.append({"peak_i": peak_i, "trough_i": trough_i})
        pending.append(len(rows) - 1)
        if recovery_i is not None:
            for k in pending:
                rows[k]["recovery_i"] = recovery_i
            pending.clear()

    for t in range(1, n):
        c = arr[t]
        if not in_ep:
            if c > arr[ath_i]:
                ath_i = t
            elif c / arr[ath_i] - 1.0 <= -threshold:
                in_ep = True
                peak_i, trough_i, reb_i = ath_i, t, t
            continue

        # 에피소드 진행 중
        if c > arr[ath_i]:
            # 회복: 현재 조각을 닫고 같은 구간의 모든 조각에 회복일 부여
            _close_current(t)
            in_ep = False
            ath_i = t
            continue

        confirmed = split and arr[reb_i] >= arr[trough_i] * (1.0 + threshold)
        if confirmed and c / arr[reb_i] - 1.0 <= -threshold:
            # 확정 저점 이후 반등 고점에서 threshold 재하락 → 새 에피소드 (회복일 공유)
            _close_current(None)
            peak_i, trough_i, reb_i = reb_i, t, t
        elif c < arr[trough_i]:
            trough_i, reb_i = t, t          # 연장: 저점 갱신, 반등 상태 초기화
        elif c > arr[reb_i]:
            reb_i = t

    if in_ep:
        _close_current(None)                # 시계열 끝까지 미회복

    if not rows:
        return _empty_episodes()

    recs = []
    for r in rows:
        p, tr = r["peak_i"], r["trough_i"]
        rec_i = r.get("recovery_i")
        recs.append({
            "peak_date": idx[p],
            "trough_date": idx[tr],
            "depth": arr[tr] / arr[p] - 1.0,
            "days_to_trough": int(tr - p),
            "recovery_date": idx[rec_i] if rec_i is not None else pd.NaT,
            "days_to_recover": float(rec_i - tr) if rec_i is not None else np.nan,
        })
    df = pd.DataFrame(recs, columns=list(EPISODE_COLUMNS))
    df["recovery_date"] = pd.to_datetime(df["recovery_date"])
    return df


def _empty_episodes() -> pd.DataFrame:
    return pd.DataFrame({
        "peak_date": pd.Series(dtype="datetime64[ns]"),
        "trough_date": pd.Series(dtype="datetime64[ns]"),
        "depth": pd.Series(dtype=float),
        "days_to_trough": pd.Series(dtype=int),
        "recovery_date": pd.Series(dtype="datetime64[ns]"),
        "days_to_recover": pd.Series(dtype=float),
    })


def independent_blocks(n_days: int, h: int) -> int:
    """n_days 안에 들어가는 겹치지 않는 h 일 창의 수 (n // h)."""
    if h <= 0:
        raise ValueError(f"h 는 양수여야 합니다: {h}")
    if n_days < 0:
        raise ValueError(f"n_days 는 0 이상이어야 합니다: {n_days}")
    return int(n_days) // int(h)
