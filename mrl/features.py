# -*- coding: utf-8 -*-
"""mrl.features — Phase 2 보정 모델의 특징 (ARCHITECTURE_PHASE2.md §4·§6 계약).

세 모델 입력은 모두 **척도 불변 비율/로짓**이라 표준화(z-score·백분위)를 하지 않는다 (백분위는 표시 전용).
  x_vix = logit(p_vix),  p_vix = B1 사전 등록 벤치마크(§6, 드리프트 m=−s²/2 반사원리)
  x_har = ln σ_har − ln(VIX/100),  σ_har = exp(mean(ln rv1, ln rv5, ln rv22)) (GK+OV, 고정 동일가중)
  x_ma  = Close/SMA180 − 1

원칙
  * 점(point-in-time): `build_features(bundle, asof)` 는 asof 이후 행을 **먼저** 버리고 계산한다. 중심 창·미래 행 참조 금지.
    tests/test_features.py::test_point_in_time_truncation 이 무작위 T 20개에서 비트 동일성을 검사한다.
  * VIX 정렬: close.csv 의 ^VIX 를 SPY 세션(spy_ohlc 인덱스)에 reindex, 결측은 ≤3세션 ffill(초과 NaN + 경고).
    SPY 세션이 아닌 ^VIX 행(휴장일 유령 행)은 무시한다.
  * 조용한 실패 금지: 입력 결측이면 NaN + attrs["warnings"]; 값을 지어내지 않는다. 형식 오류는 예외.
  * 한국어 주석, 영어 식별자.
"""
from __future__ import annotations

import hashlib
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from mrl import vol
from mrl.config import P2

__all__ = [
    "FEATURE_SPEC", "FEATURE_COLUMNS", "MODEL_INPUTS", "FEATURE_RULE", "SPEC_SOURCES", "BGK_BETA", "DRIFTS", "MONITORINGS",
    "PERCENTILE_COLUMNS",
    "spec_sha256", "align_vix", "vix_implied_prob", "ma_distance", "build_features", "factor_percentiles", "input_status",
]

MODEL_INPUTS = tuple(P2["features"])                     # ("x_vix", "x_har", "x_ma")
DRIFTS = ("martingale", "none")
MONITORINGS = ("continuous", "bgk")
BGK_BETA = 0.5826                                        # Broadie–Glasserman–Kou 이산관측 장벽 보정 상수
TRADING_DAYS = 252
_LOGIT_EPS = 1e-15                                       # p∈{0,1} 의 ±inf 방지 (실제 VIX 범위에선 절대 닿지 않음)
_VIX3M_COL = "VIX3M"
_VIX_COL = "^VIX"

# 특징 규칙 식별자 — 모든 산출물에 기록 (WINDOW_RULE 과 같은 보호)
FEATURE_RULE = ("p2|x_vix=logit(refl(VIX,dd=0.05,h=20,drift=-s2/2))"
                "|x_har=mean_log(rv1,rv5,rv22;GK+OV;floor=1e-8)-ln(VIX/100)|x_ma=C/SMA180-1"
                "|vix_ffill<=3|purge=20|C=1.0|train_start=1993-10-14")

# spec_sha256 이 해시하는 소스 (mrl/ 기준). model.py 는 다른 모듈이 쓰며, 없으면 경고 + 표식으로 대신한다.
SPEC_SOURCES = ("features.py", "vol.py", "model.py")

# 열 → {source, columns, lookback, first_date, formula, label_ko, model_input}. 순서 = build_features 열 순서.
FEATURE_SPEC: dict[str, dict] = {
    "vix": {"source": "close.csv", "columns": ["^VIX"], "lookback": 0, "first_date": "1993-01-29",
            "formula": "reindex(SPY sessions), ffill<=3", "label_ko": "VIX 종가 (SPY 세션 정렬)", "model_input": False},
    "p_vix": {"source": "vix", "columns": ["vix"], "lookback": 0, "first_date": "1993-01-29",
              "formula": "Phi((b-m)/s)+exp(2mb/s^2)Phi((b+m)/s); s=V*sqrt(20/252), m=-s^2/2, b=ln(0.95)",
              "label_ko": "VIX 내재 확률 (B1, 드리프트 반사원리)", "model_input": False},
    "p_vix_driftless": {"source": "vix", "columns": ["vix"], "lookback": 0, "first_date": "1993-01-29",
                        "formula": "2*Phi(b/s)", "label_ko": "VIX 내재 확률 (무드리프트, 대조용)", "model_input": False},
    "p_vix_bgk": {"source": "vix", "columns": ["vix"], "lookback": 0, "first_date": "1993-01-29",
                  "formula": "B1 with b' = b - 0.5826*V*sqrt(1/252)", "label_ko": "VIX 내재 확률 (BGK 이산관측 보정)", "model_input": False},
    "x_vix": {"source": "vix", "columns": ["p_vix"], "lookback": 0, "first_date": "1993-01-29",
              "formula": "ln(p_vix/(1-p_vix))", "label_ko": "VIX 내재 확률 로짓", "model_input": True},
    "var_gkov": {"source": "spy_ohlc.csv", "columns": ["Open", "High", "Low", "Close"], "lookback": 1, "first_date": "1993-02-01",
                 "formula": "max(0.5 ln(H/L)^2 - (2ln2-1) ln(C/O)^2 + ln(O/C_prev)^2, 1e-8)", "label_ko": "GK+야간갭 일간 분산",
                 "model_input": False},
    "rv1": {"source": "var_gkov", "columns": ["var_gkov"], "lookback": 1, "first_date": "1993-02-01",
            "formula": "sqrt(252*max(mean_1(v), 1e-8))", "label_ko": "실현변동성 1일", "model_input": False},
    "rv5": {"source": "var_gkov", "columns": ["var_gkov"], "lookback": 5, "first_date": "1993-02-05",
            "formula": "sqrt(252*max(mean_5(v), 1e-8))", "label_ko": "실현변동성 5일", "model_input": False},
    "rv22": {"source": "var_gkov", "columns": ["var_gkov"], "lookback": 22, "first_date": "1993-03-03",
             "formula": "sqrt(252*max(mean_22(v), 1e-8))", "label_ko": "실현변동성 22일", "model_input": False},
    "har_vol_20": {"source": "rv1/rv5/rv22", "columns": ["rv1", "rv5", "rv22"], "lookback": 22, "first_date": "1993-03-03",
                   "formula": "exp((ln rv1 + ln rv5 + ln rv22)/3)", "label_ko": "고정가중 HAR 변동성(적합 아님)", "model_input": False},
    "x_har": {"source": "har_vol_20, vix", "columns": ["har_vol_20", "vix"], "lookback": 22, "first_date": "1993-03-03",
              "formula": "ln(har_vol_20) - ln(vix/100)", "label_ko": "실현-내재 변동성 갭(로그)", "model_input": True},
    "sma180": {"source": "spy_ohlc.csv", "columns": ["Close"], "lookback": 180, "first_date": "1993-10-14",
               "formula": "mean(Close_{t-179..t}), min_periods=180", "label_ko": "180일 단순이동평균", "model_input": False},
    "x_ma": {"source": "spy_ohlc.csv", "columns": ["Close", "sma180"], "lookback": 180, "first_date": "1993-10-14",
             "formula": "Close/sma180 - 1", "label_ko": "180일선 이격률", "model_input": True},
    # targets 의 RV 규약(로그수익률 20개 std·√252)이라 21번째 세션(1993-03-01)이 최초 유효일 (계약표의 02-26 은 20가격·19수익 기준)
    "rv20_cc": {"source": "spy_ohlc.csv", "columns": ["Close"], "lookback": 20, "first_date": "1993-03-01",
                "formula": "std(logret_{t-19..t})*sqrt(252)", "label_ko": "종가 실현변동성 20일 (표시·자기점검)", "model_input": False},
    "ts_diag": {"source": "vix, cboe.csv", "columns": ["vix", "VIX3M"], "lookback": 0, "first_date": "2009-09-18",
                "formula": "vix / VIX3M", "label_ko": "VIX 기간구조 (표시 전용)", "model_input": False},
}
FEATURE_COLUMNS = tuple(FEATURE_SPEC)
assert tuple(c for c, s in FEATURE_SPEC.items() if s["model_input"]) == MODEL_INPUTS
# 표시 전용 백분위 대상 (모델 입력 금지 — build_features 열에 넣지 않는다)
PERCENTILE_COLUMNS = ("vix", "x_vix", "har_vol_20", "x_har", "x_ma", "rv20_cc", "ts_diag")


# ------------------------------------------------------------------
# 사양 해시
# ------------------------------------------------------------------
def _hash_sources(paths) -> tuple[str, list[str]]:
    """파일 목록의 SHA-256 (이름 + 내용, CRLF→LF 정규화 — 체크아웃 방식과 무관). 없는 파일은 표식으로 대신하고 목록에 담는다."""
    h = hashlib.sha256()
    missing: list[str] = []
    for p in paths:
        p = Path(p)
        h.update(p.name.encode("utf-8") + b"\n")
        if p.exists():
            h.update(p.read_bytes().replace(b"\r\n", b"\n"))
        else:
            missing.append(p.name)
            h.update(b"MISSING:" + p.name.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest(), missing


def spec_sha256() -> str:
    """mrl/features.py + mrl/vol.py + mrl/model.py 소스의 SHA-256 — 모든 산출물에 기록한다.
    소스가 없으면(예: model.py 미작성) warnings.warn 하고 표식을 해시에 넣는다 — 파일이 생기면 해시가 바뀐다(= 사양 변경)."""
    base = Path(__file__).resolve().parent
    digest, missing = _hash_sources(base / n for n in SPEC_SOURCES)
    if missing:
        warnings.warn(f"spec_sha256: 소스 없음 {missing} — 표식으로 대신함(파일이 생기면 해시가 바뀐다)")
    return digest


# ------------------------------------------------------------------
# VIX 정렬
# ------------------------------------------------------------------
def _check_index(idx, name: str) -> pd.DatetimeIndex:
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError(f"{name} 는 DatetimeIndex 여야 함 (받은 형: {type(idx).__name__})")
    if idx.tz is not None:
        raise ValueError(f"계약 위반: {name} 는 tz-naive 여야 함")
    if not idx.is_monotonic_increasing:
        raise ValueError(f"{name} 가 오름차순이 아님")
    if idx.has_duplicates:
        dup = idx[idx.duplicated()][:3].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"{name} 에 중복 날짜가 있음 (예: {dup})")
    return idx


def _fmt_dates(idx, k: int = 5) -> str:
    idx = pd.DatetimeIndex(idx)
    s = ", ".join(idx[:k].strftime("%Y-%m-%d"))
    return s + (f" 외 {len(idx) - k}개" if len(idx) > k else "")


def align_vix(close: pd.DataFrame, spy_index: pd.DatetimeIndex, limit: int = P2["vix_ffill_limit"]) -> tuple[pd.Series, list[str]]:
    """close['^VIX'] 를 SPY 세션에 정렬. 결측 세션은 ≤ limit 연속까지 직전 값으로 채우고(경고), 초과분은 NaN(경고).

    SPY 세션이 아닌 ^VIX 행(휴장일 유령 행)은 무시하고 목록에 기록한다. VIX 최초 관측 이전의 SPY 세션은 NaN(채우지 않음).
    반환 (vix Series[name='vix', index=spy_index], warnings list[str]).
    """
    if not isinstance(close, pd.DataFrame):
        raise TypeError(f"close 는 DataFrame 이어야 함 (받은 형: {type(close).__name__})")
    if _VIX_COL not in close.columns:
        raise ValueError(f"close 에 {_VIX_COL} 열이 없음 (열 {len(close.columns)}개)")
    spy_index = _check_index(pd.DatetimeIndex(spy_index), "spy_index")
    _check_index(pd.DatetimeIndex(close.index), "close.index")
    limit = int(limit)
    if limit < 0:
        raise ValueError(f"limit 는 0 이상이어야 함: {limit}")
    notes: list[str] = []
    raw = close[_VIX_COL].astype(float).dropna()
    if (raw <= 0).any():
        ex = raw.index[raw <= 0][:3].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"^VIX 에 0 이하 값이 있음 (예: {ex})")
    extra = raw.index.difference(spy_index)
    if len(extra):
        notes.append(f"SPY 세션이 아닌 ^VIX 행 {len(extra)}개 무시 (휴장일 유령 행 등): {_fmt_dates(extra)}")
    v = raw.reindex(spy_index)
    v.name = "vix"
    if raw.empty:
        notes.append("^VIX 관측이 없음 → vix 전부 NaN")
        return v, notes
    after_first = spy_index >= raw.index[0]
    missing = v.isna().to_numpy() & after_first
    if missing.any():
        filled = v.ffill(limit=limit) if limit > 0 else v
        got = missing & filled.notna().to_numpy()
        lost = missing & filled.isna().to_numpy()
        if got.any():
            msg = f"VIX 결측 {int(got.sum())}세션을 직전 값으로 채움(≤{limit}세션): {_fmt_dates(spy_index[got])}"
            notes.append(msg)
            warnings.warn(msg)
        if lost.any():
            how = f"{limit}세션 초과" if limit > 0 else "(ffill 없음)"
            msg = f"VIX 결측 {how} → NaN {int(lost.sum())}세션: {_fmt_dates(spy_index[lost])}"
            notes.append(msg)
            warnings.warn(msg)
        v = filled
    lead = (~after_first).sum()
    if lead:
        notes.append(f"VIX 최초 관측({raw.index[0]:%Y-%m-%d}) 이전 SPY 세션 {int(lead)}개는 NaN")
    return v, notes


# ------------------------------------------------------------------
# VIX 내재 확률 (§6 사전 등록 식)
# ------------------------------------------------------------------
def _refl_prob(V: np.ndarray, dd: float, h: int, drift: str, monitoring: str) -> np.ndarray:
    """V = VIX/100 (배열). 브라운 운동 로그가격의 running-minimum 반사원리.
    p = Φ((b−m)/s) + exp(2mb/s²)·Φ((b+m)/s),  s = V·√(h/252), b = ln(1−dd), m = −s²/2 (martingale) 또는 0 (none).
    monitoring='bgk' 면 장벽을 b' = b − 0.5826·V·√(1/252) 로 옮긴다 (일 1회 종가 관측 보정)."""
    T = h / TRADING_DAYS
    s = V * math.sqrt(T)
    b = math.log(1.0 - dd)
    if monitoring == "bgk":
        b = b - BGK_BETA * V * math.sqrt(1.0 / TRADING_DAYS)
    if drift == "martingale":
        m = -0.5 * s ** 2
    else:
        m = np.zeros_like(s)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        p = ndtr((b - m) / s) + np.exp(2.0 * m * b / s ** 2) * ndtr((b + m) / s)
    return p


def vix_implied_prob(vix, dd: float = P2["dd"], h: int = P2["h"], drift: str = "martingale",
                     monitoring: str = "continuous"):
    """§6 식. VIX(지수 수준, 예: 20.0) → 다음 h 거래일 안에 종가가 −dd 이하로 내려갈 확률.

    drift: "martingale"(B1 사전 등록, m=−s²/2) | "none"(2Φ(b/s), 대조용). monitoring: "continuous" | "bgk"(이산 장벽 보정).
    vix ≤ 0 이나 ±inf 면 ValueError; NaN 은 NaN 으로 전파. Series → Series(같은 인덱스), 스칼라 → float, 그 외 → ndarray.
    값(B1): VIX 12/16/20/30/45 → 0.133/0.262/0.372/0.558/0.703. BGK: 0.102/0.211/0.307/0.475/0.613. 무드리프트 20 → 0.363.
    """
    if drift not in DRIFTS:
        raise ValueError(f"drift 는 {DRIFTS} 중 하나: {drift!r}")
    if monitoring not in MONITORINGS:
        raise ValueError(f"monitoring 은 {MONITORINGS} 중 하나: {monitoring!r}")
    dd = float(dd)
    h = int(h)
    if not (0.0 < dd < 1.0):
        raise ValueError(f"dd 는 (0,1) 사이여야 함: {dd}")
    if h <= 0:
        raise ValueError(f"h 는 양의 정수여야 함: {h}")
    is_series = isinstance(vix, pd.Series)
    arr = np.asarray(vix.to_numpy(dtype=float) if is_series else vix, dtype=float)
    finite = np.isfinite(arr)
    if (arr[finite] <= 0).any():
        raise ValueError("vix 는 양수여야 함 (0 이하 값 존재)")
    if np.isinf(arr).any():
        raise ValueError("vix 에 무한값이 있음")
    p = _refl_prob(arr / 100.0, dd, h, drift, monitoring)
    p = np.where(finite, p, np.nan)
    if is_series:
        return pd.Series(p, index=vix.index, name=f"p_vix_{drift}_{monitoring}")
    if p.ndim == 0:
        return float(p)
    return p


def _logit(p):
    """ln(p/(1−p)); p 를 [eps, 1−eps] 로 자른다 (실제 VIX 범위에선 절대 닿지 않음 — ±inf 방지)."""
    q = np.clip(p, _LOGIT_EPS, 1.0 - _LOGIT_EPS)
    return np.log(q / (1.0 - q))


# ------------------------------------------------------------------
# 추세
# ------------------------------------------------------------------
def ma_distance(close: pd.Series, window: int = P2["ma_window"]) -> pd.Series:
    """Close_t / mean(Close_{t−window+1..t}) − 1, min_periods=window. 비율이라 조정 방식에 불변."""
    if not isinstance(close, pd.Series):
        raise TypeError(f"close 는 Series 여야 함 (받은 형: {type(close).__name__})")
    _check_index(pd.DatetimeIndex(close.index), "close.index")
    window = int(window)
    if window <= 0:
        raise ValueError(f"window 는 양의 정수여야 함: {window}")
    c = close.astype(float)
    if (c.dropna() <= 0).any():
        raise ValueError("close 에 0 이하 가격이 있음")
    sma = c.rolling(window, min_periods=window).mean()
    out = c / sma - 1.0
    out.name = "x_ma"
    return out


# ------------------------------------------------------------------
# 특징 표
# ------------------------------------------------------------------
def _asof_ts(asof) -> pd.Timestamp | None:
    if asof is None:
        return None
    ts = pd.Timestamp(asof)
    if pd.isna(ts):
        raise ValueError(f"asof 를 날짜로 해석할 수 없음: {asof!r}")
    if ts.tzinfo is not None:
        raise ValueError("계약 위반: asof 는 tz-naive 여야 함")
    return ts.normalize()


def _cut(df, asof: pd.Timestamp | None):
    """asof 이후 행 제거 (asof=None 이면 그대로). 계산 **전에** 호출한다 — 점 원칙."""
    if df is None or asof is None:
        return df
    return df.loc[:asof]


def build_features(bundle, asof=None) -> pd.DataFrame:
    """특징 표. 인덱스 = SPY 세션(asof 까지 자른 뒤 계산). 열 = FEATURE_COLUMNS 순서, 각 열은 first_date 전 NaN.

    반환 .attrs = {"feature_rule", "spec_sha256", "warnings"(list[str]), "asof"(str|None), "last_session"(str)}.
    백분위 열은 넣지 않는다(모델 입력 금지; factor_percentiles 로 별도 계산).
    """
    for name in ("close", "spy_ohlc"):
        if not hasattr(bundle, name):
            raise TypeError(f"bundle 에 {name} 가 없음")
    ts = _asof_ts(asof)
    # ---- 1. 점 원칙: 모든 입력 프레임을 먼저 자른다 ----
    ohlc = _cut(bundle.spy_ohlc, ts)
    close = _cut(bundle.close, ts)
    cboe = _cut(getattr(bundle, "cboe", None), ts)
    if ohlc is None or len(ohlc) == 0:
        raise ValueError(f"asof {ts} 이하에 SPY 세션이 없음")
    idx = _check_index(pd.DatetimeIndex(ohlc.index), "spy_ohlc.index")
    notes: list[str] = []
    nan_rows = ohlc[list(vol.OHLC_COLUMNS)].isna().any(axis=1)
    if nan_rows.any():
        notes.append(f"SPY OHLC 결측 {int(nan_rows.sum())}행 → 해당 행 변동성 NaN: {_fmt_dates(idx[nan_rows.to_numpy()])}")

    # ---- 2. VIX 와 내재 확률 ----
    vix, vnotes = align_vix(close, idx)
    notes.extend(vnotes)
    p_vix = vix_implied_prob(vix, drift="martingale", monitoring="continuous")
    p_drl = vix_implied_prob(vix, drift="none", monitoring="continuous")
    p_bgk = vix_implied_prob(vix, drift="martingale", monitoring="bgk")
    x_vix = pd.Series(_logit(p_vix.to_numpy()), index=idx)

    # ---- 3. 실현변동성 (GK+OV) · 고정가중 HAR ----
    var_gkov = vol.garman_klass_variance(ohlc, overnight=True, floor=P2["var_floor"])
    comp = vol.har_components(var_gkov, lookbacks=P2["har_lookbacks"], floor=P2["var_floor"])
    ln_har = vol.har_log_vol(comp, weights=P2["har_weights"])
    har_vol_20 = np.exp(ln_har)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_har = ln_har - np.log(vix / 100.0)
    floored = (var_gkov.notna() & (var_gkov <= P2["var_floor"]))
    if floored.any():
        notes.append(f"var_gkov 하한({P2['var_floor']:g}) 적용 {int(floored.sum())}행: {_fmt_dates(idx[floored.to_numpy()])}")

    # ---- 4. 추세 ----
    c = ohlc["Close"].astype(float)
    sma = c.rolling(P2["ma_window"], min_periods=P2["ma_window"]).mean()
    x_ma = c / sma - 1.0

    # ---- 5. 표시·자기점검 ----
    rv20_cc = np.log(c).diff().rolling(20, min_periods=20).std() * math.sqrt(TRADING_DAYS)
    if cboe is not None and len(cboe) and _VIX3M_COL in cboe.columns:
        v3 = cboe[_VIX3M_COL].astype(float).reindex(idx)
        v3 = v3.where(v3 > 0)
        ts_diag = vix / v3
    else:
        ts_diag = pd.Series(np.nan, index=idx)
        notes.append("cboe.VIX3M 없음 → ts_diag 전부 NaN (표시 전용)")

    out = pd.DataFrame({
        "vix": vix.to_numpy(), "p_vix": p_vix.to_numpy(), "p_vix_driftless": p_drl.to_numpy(), "p_vix_bgk": p_bgk.to_numpy(),
        "x_vix": x_vix.to_numpy(), "var_gkov": var_gkov.to_numpy(),
        "rv1": comp["rv1"].to_numpy(), "rv5": comp["rv5"].to_numpy(), "rv22": comp["rv22"].to_numpy(),
        "har_vol_20": np.asarray(har_vol_20, dtype=float), "x_har": np.asarray(x_har, dtype=float),
        "sma180": sma.to_numpy(), "x_ma": x_ma.to_numpy(), "rv20_cc": rv20_cc.to_numpy(), "ts_diag": ts_diag.to_numpy(),
    }, index=idx)
    out.index.name = "date"
    out = out[list(FEATURE_COLUMNS)]
    last = idx[-1]
    tail_missing = [k for k in MODEL_INPUTS if not np.isfinite(out[k].iloc[-1])]
    if tail_missing and len(out) >= P2["ma_window"]:
        notes.append(f"마지막 세션 {last:%Y-%m-%d} 모델 입력 결측: {tail_missing}")
    out.attrs = {"feature_rule": FEATURE_RULE, "spec_sha256": spec_sha256(), "warnings": notes,
                 "asof": None if ts is None else ts.strftime("%Y-%m-%d"), "last_session": last.strftime("%Y-%m-%d")}
    return out


def factor_percentiles(feats: pd.DataFrame, window: int = 2520, min_periods: int = 252) -> pd.DataFrame:
    """표시 전용 10년(2520세션) 이동 백분위 (0~1, 창 안 순위/창 길이). 모델 입력 금지 — build_features 열에 넣지 않는다.
    열 pct_<name>, 대상 = PERCENTILE_COLUMNS 중 feats 에 있는 열. min_periods 미만이면 NaN."""
    window = int(window)
    min_periods = int(min_periods)
    if window <= 0 or min_periods <= 0 or min_periods > window:
        raise ValueError(f"window/min_periods 가 잘못됨: {window}/{min_periods}")
    cols = [c for c in PERCENTILE_COLUMNS if c in feats.columns]
    if not cols:
        raise ValueError(f"feats 에 백분위 대상 열이 없음 (기대: {PERCENTILE_COLUMNS})")
    out = pd.DataFrame(index=feats.index)
    for ccol in cols:
        out[f"pct_{ccol}"] = feats[ccol].astype(float).rolling(window, min_periods=min_periods).rank(pct=True)
    return out


_INPUT_REASON = {
    "x_vix": "VIX 결측(정렬 후 3세션 초과 결측 또는 VIX 없음)",
    "x_har": "실현변동성(OHLC 22세션) 또는 VIX 결측",
    "x_ma": "SMA180 미충족(180세션 필요) 또는 종가 결측",
}


def input_status(feats_row: pd.Series) -> tuple[bool, str]:
    """세 모델 입력(x_vix, x_har, x_ma)이 모두 유한한가. 아니면 (False, 사유 문자열) — 카드의 '확률 계산 불가' 경로."""
    if not isinstance(feats_row, pd.Series):
        raise TypeError("feats_row 는 Series(특징 표의 한 행) 여야 함")
    bad = []
    for k in MODEL_INPUTS:
        v = feats_row.get(k, np.nan)
        try:
            ok = np.isfinite(float(v))
        except (TypeError, ValueError):
            ok = False
        if not ok:
            bad.append(f"{k}({_INPUT_REASON[k]})")
    if not bad:
        return True, ""
    return False, "확률 계산 불가: " + "; ".join(bad)
