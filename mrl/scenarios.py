# -*- coding: utf-8 -*-
"""시나리오 — 세기만 한다 (ARCHITECTURE_PHASE3.md §7). 적합 파라미터 **0개**.

이 모듈은 모형을 적합하지 않는다. OOS 기록(2003-01-02~2024-08-30; 홀드아웃 해제 후에는 + 홀드아웃,
라이브는 라벨이 닫히는 대로 별도 호출)에서 **조건부 빈도와 분위수를 세기만** 한다. 상수는 전부
`config.SCENARIO_P3`·`P2` 에서 오고, 이 모듈이 만드는 적합값은 없다(§14 '시나리오·경보·킬룰' 줄).

계약 (ARCHITECTURE_PHASE3.md §7)
    bin_table(p, y, fwd_ret_20, fwd_maxdd_20, bins, min_n_eff, n_eff_div, episodes10) -> DataFrame
    state_table(states, y, fwd_ret_20, fwd_maxdd_20) -> DataFrame
    episode_conditionals(spy_close, *, end, levels, split) -> dict
    current_drawdown(spy_close, asof) -> dict
    implied_range(spot, vix, h, z) -> dict          # 시장 내재 20세션 범위
    har_range(spot, har_fc_20, h, z) -> dict        # 적합 HAR 예측 기준 같은 범위(병기)
    coverage(fwd_ret_20, vix, har_fc) -> dict       # 과거·라이브 포함률
    today_context(p_today, state_today, vix, har_fc, spot, tables, dd) -> dict   # 카드 3줄(고정 순서)

사전 등록 규칙 (§7·§2 '고정 상수')
* **표본 수**: 겹치는 20세션 창이므로 `n_eff = n / n_eff_div`(= n/20). 에피소드 표는 에피소드 수가 곧 n_eff.
  모든 행에 n·n_eff·구간이 있다 — 하나라도 없으면 표시 금지(코드로 강제, `assert_displayable`).
* **구간**: 비율은 `calibrate.wilson(p_hat, n_eff)`(z=1.96), 중앙값은 40세션 순환 블록 부트스트랩
  (`block_bootstrap_quantile`, seed 0 — `evaluate.block_bootstrap_ci` 는 평균 전용이라 같은 재표본
  방식으로 분위수판을 여기 둔다).
* **풀링(bin_table)**: 표 **위에서부터**(낮은 확률 구간부터) 인접 구간을 합쳐, 표시되는 모든 행이
  n_eff ≥ 20 이 되게 한다. 남은 꼬리가 20 에 못 미치면 바로 위 그룹에 흡수한다. 설계 단계 기록에서
  [0.30, 0.40) 이 n_eff 18 에서 관측 0.229 로 꺼지는 비단조가 이 규칙 하나로 사라진다(손으로 고치지 않는다).
* **회색(그레이)**: 풀링 뒤에도(또는 풀링이 불가능한 상태·에피소드 표에서) n_eff < min_n_eff 인 셀은
  `grey=True` — 카드에서 단독 표시 금지. `today_context` 는 회색 행을 절대 단독으로 문장에 쓰지 않는다.
* **점(point-in-time)**: 이 모듈은 과거를 세는 도구다. 입력 프레임의 하드컷(홀드아웃 2024-09-01)은
  **호출자**(`run_phase3.py`)의 책임이며, 여기서는 잘린 프레임을 그대로 센다. `end` 를 주면 종가를
  그 날짜까지(포함) 자른 뒤 에피소드를 만든다 — 홀드아웃 고점이 표에 들어가지 않게 하는 경로.
* **조용한 실패 금지**: 결측 입력은 NaN·`ok=False`·경고로 남기고 문장에서는 '미상'으로 표기한다.
  구조적 오류(음수 가격, h ≤ 0, 정렬 불가 인덱스)는 ValueError.

측정 규약(설계 단계 재현: ARCHITECTURE_PHASE3 §7 D)
* 시장 내재 범위: `s = VIX_t/100 · sqrt(h/252)`, 밴드 = spot·exp(±z·s). 표시 ±% 는 로그 폭 z·s.
* 포함률: 실현 `ln(1 + fwd_ret_h) / s` 의 |z| ≤ 1 / 1.2816 / 1.645 비율과 −1.645s 아래 왼꼬리 비율.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

from mrl.calibrate import Z95, wilson
from mrl.config import P2, P2_STATES, SCENARIO_P3
from mrl.targets import episodes as _episodes

__all__ = [
    "BINS", "MIN_N_EFF", "N_EFF_DIV", "H", "Z", "DD_LEVELS", "EPISODE_SPLIT", "NOMINAL", "TRADING_DAYS",
    "BOOT_BLOCK", "N_BOOT", "BIN_COLUMNS", "STATE_COLUMNS", "EPISODE_TABLE_COLUMNS", "LINE_ORDER",
    "SPREAD_SENTENCE", "MISSING",
    "block_bootstrap_quantile", "pool_bins", "bin_table", "state_table", "assert_displayable",
    "episode_starts", "episode_table", "episode_conditionals", "current_drawdown", "live_column",
    "implied_range", "har_range", "coverage", "range_label", "today_context",
]

# ------------------------------------------------------------------
# 상수 (전부 config 에서; 이 모듈은 새 숫자를 만들지 않는다)
# ------------------------------------------------------------------
BINS = tuple(SCENARIO_P3["bins"])
MIN_N_EFF = float(SCENARIO_P3["min_n_eff"])
N_EFF_DIV = int(SCENARIO_P3["n_eff_div"])
H = int(SCENARIO_P3["h"])
Z = dict(SCENARIO_P3["z"])                 # {"1s": 1.0, "80": 1.2816, "90": 1.645}
DD_LEVELS = tuple(SCENARIO_P3["dd_levels"])
EPISODE_SPLIT = bool(SCENARIO_P3["episode_split"])
BOOT_BLOCK = int(P2["boot_block"])         # 40
N_BOOT = int(P2["n_boot"])                 # 4,000
TRADING_DAYS = 252
BAND_KEYS = ("1s", "80", "90")
# 명목 포함률 = 정규분포의 양측 확률(z 에서 유도 — 새 상수 아님)
NOMINAL = {k: float(math.erf(float(Z[k]) / math.sqrt(2.0))) for k in BAND_KEYS}
MISSING = "미상"                            # 결측 표기(BAD_TOKEN 회피: nan/None/null 금지)

BIN_COLUMNS = ("bin", "bin_lo", "bin_hi", "pooled", "n_bins", "n", "n_eff", "mean_p", "obs",
               "wilson_lo", "wilson_hi", "ret_p10", "ret_p50", "ret_p90", "ret_p50_lo", "ret_p50_hi",
               "mdd_p10", "mdd_p50", "share_ep10_start", "ep10_lo", "ep10_hi", "grey")
STATE_COLUMNS = ("state", "n", "n_eff", "obs", "wilson_lo", "wilson_hi", "ret_p10", "ret_p50", "ret_p90",
                 "ret_p50_lo", "ret_p50_hi", "mdd_p10", "mdd_p50", "grey")
EPISODE_TABLE_COLUMNS = ("peak_date", "breach_date", "trough_date", "recovery_date", "depth", "extra_loss",
                         "extra_loss_close", "peak_to_trough", "breach_to_trough", "trough_to_recovery", "year")
LINE_ORDER = ("market_range", "today_like", "if_episode")     # §7 카드 3줄 고정 순서
SPREAD_SENTENCE = ("모든 상태·구간에서 20일 수익의 중앙값은 양수다 — 모델은 방향이 아니라 결과의 폭을 예측한다"
                   "(VALIDATION §0). 높은 위험 구간일수록 수익 분포는 좁아지지 않고 넓어진다(급락과 반등이 같이 산다).")

_MINUS = "−"                          # 카드 표기용 뺄셈 기호


# ------------------------------------------------------------------
# 공용 헬퍼
# ------------------------------------------------------------------
def _num(x) -> float:
    """스칼라를 float 으로 (None·빈 문자열·비수치는 NaN)."""
    if x is None:
        return float("nan")
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _pct(x, digits: int = 1, signed: bool = False) -> str:
    """비율(0.061) → '6.1%' / '+6.1%' / '−6.1%'. 비유한이면 MISSING."""
    v = _num(x)
    if not np.isfinite(v):
        return MISSING
    s = f"{100.0 * v:+.{digits}f}" if signed else f"{100.0 * v:.{digits}f}"
    return s.replace("-", _MINUS) + "%"


def _pct0(x) -> str:
    return _pct(x, digits=0)


def _span(lo, hi, digits: int = 0) -> str:
    """구간 표기 '20~50%' (한쪽이 결측이면 MISSING)."""
    a, b = _num(lo), _num(hi)
    if not (np.isfinite(a) and np.isfinite(b)):
        return MISSING
    return (f"{100.0 * a:.{digits}f}".replace("-", _MINUS) + "~"
            + f"{100.0 * b:.{digits}f}".replace("-", _MINUS) + "%")


def _cnt(x) -> str:
    v = _num(x)
    return MISSING if not np.isfinite(v) else f"{v:.0f}"


def _quantile(a: np.ndarray, q: float) -> float:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.quantile(a, q)) if a.size else float("nan")


def _as_frame(obj) -> pd.DataFrame:
    """DataFrame · records(list[dict]) · dict-of-lists 를 DataFrame 으로 (JSON 왕복 대응)."""
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, (list, tuple)):
        return pd.DataFrame(list(obj))
    if isinstance(obj, dict):
        return pd.DataFrame(obj)
    raise TypeError(f"표는 DataFrame·records·dict 여야 합니다: {type(obj).__name__}")


def _pair(x) -> tuple[float, float]:
    """(lo, hi) 또는 [lo, hi] → (float, float). JSON 왕복 후 리스트가 되는 것을 흡수."""
    if x is None:
        return (float("nan"), float("nan"))
    try:
        a, b = x[0], x[1]
    except (TypeError, IndexError, KeyError):
        return (float("nan"), float("nan"))
    return (_num(a), _num(b))


def _clean_close(spy_close, name: str = "spy_close") -> pd.Series:
    """종가 Series 검증(양수·DatetimeIndex·오름차순·중복 없음). targets._clean_close 와 같은 규약."""
    if not isinstance(spy_close, pd.Series):
        raise TypeError(f"{name} 는 pd.Series 여야 합니다 (받은 형: {type(spy_close).__name__})")
    s = spy_close.dropna()
    if len(s) == 0:
        raise ValueError(f"{name} 에 유효한 값이 없습니다")
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.DatetimeIndex(pd.to_datetime(s.index))
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    if s.index.has_duplicates:
        raise ValueError(f"{name} 인덱스에 중복 날짜가 있습니다")
    if not s.index.is_monotonic_increasing:
        s = s.sort_index()
    if (s.to_numpy(dtype=float) <= 0).any():
        raise ValueError(f"{name} 에 0 이하 가격이 있습니다")
    return s.astype(float)


def _cut(s: pd.Series, end) -> pd.Series:
    """end(포함)까지 자른다. end=None 이면 그대로. 하드컷은 호출자의 책임이며 여기서는 그 도구만 제공."""
    if end is None:
        return s
    ts = pd.Timestamp(end)
    return s.loc[s.index <= ts]


def _bin_ids(p: np.ndarray, bins) -> np.ndarray:
    """calibrate.reliability_table 과 같은 구간 배정: [lo, hi), 마지막 구간만 닫힘."""
    edges = np.asarray(bins, dtype=float)
    if len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("bins 는 오름차순 경계 ≥ 2개여야 합니다")
    ids = np.searchsorted(edges, p, side="right") - 1
    ids = np.where(p >= edges[-1], len(edges) - 2, ids)
    if (ids < 0).any() or (ids > len(edges) - 2).any():
        raise ValueError("확률이 bins 범위 밖에 있습니다")
    return ids


# ------------------------------------------------------------------
# 블록 부트스트랩 분위수 구간 (evaluate.block_bootstrap_ci 와 같은 순환 블록 재표본, 통계량만 분위수)
# ------------------------------------------------------------------
def block_bootstrap_quantile(values, q: float = 0.5, *, block: int = BOOT_BLOCK, n_boot: int = N_BOOT,
                             ci: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """순환 블록 부트스트랩으로 분위수 q 의 (lo, hi) 백분위 신뢰구간. seed 고정(결정론).

    `evaluate.block_bootstrap_ci` 와 같은 재표본 방식(길이 block 의 순환 블록을 복원추출해 길이 n 으로 절단)이며
    통계량만 평균 → 분위수로 바꾼 것이다(§7 '중앙값 구간은 40세션 블록 부트스트랩'). NaN 은 제거.
    """
    if block <= 0:
        raise ValueError("block 은 양수여야 합니다")
    if not (0 < ci < 1):
        raise ValueError("ci 는 (0,1) 사이여야 합니다")
    if not (0.0 <= q <= 1.0):
        raise ValueError("q 는 [0,1] 이어야 합니다")
    v = pd.Series(values).astype(float).to_numpy()
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return (float("nan"), float("nan"))
    b = min(int(block), n)
    k = int(math.ceil(n / b))
    rng = np.random.default_rng(int(seed))
    stats = np.empty(int(n_boot))
    offs = np.arange(b)
    chunk = max(1, min(int(n_boot), int(2_000_000 // max(k * b, 1))))
    done = 0
    while done < n_boot:
        m = min(chunk, int(n_boot) - done)
        starts = rng.integers(0, n, size=(m, k))
        sel = ((starts[:, :, None] + offs[None, None, :]) % n).reshape(m, k * b)[:, :n]
        stats[done: done + m] = np.quantile(v[sel], q, axis=1)
        done += m
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha])
    return (float(lo), float(hi))


# ------------------------------------------------------------------
# A. 확률 구간 표 (풀링)
# ------------------------------------------------------------------
def pool_bins(n_per_bin, min_n_eff: float = MIN_N_EFF, n_eff_div: int = N_EFF_DIV) -> list[tuple[int, int]]:
    """사전 등록 풀링 규칙 → 인접 구간 그룹 [(첫 구간, 마지막 구간), ...] (양끝 포함).

    표 **위에서부터** 누적하다가 n_eff ≥ min_n_eff 가 되는 순간 그룹을 닫는다. 마지막에 남은 꼬리가
    min_n_eff 에 못 미치면 바로 위 그룹에 흡수한다(그룹이 하나도 없으면 전체가 한 그룹).
    """
    counts = [int(c) for c in n_per_bin]
    if not counts:
        return []
    if n_eff_div <= 0:
        raise ValueError("n_eff_div 는 양수여야 합니다")
    groups: list[tuple[int, int]] = []
    start, acc = 0, 0
    for k, c in enumerate(counts):
        acc += c
        if acc / float(n_eff_div) >= min_n_eff:
            groups.append((start, k))
            start, acc = k + 1, 0
    if start < len(counts):                       # 남은 꼬리
        if groups:
            groups[-1] = (groups[-1][0], len(counts) - 1)
        else:
            groups.append((0, len(counts) - 1))
    return groups


def _start_dates(episodes10) -> pd.DatetimeIndex | None:
    """episodes10 → 에피소드 '시작'(돌파) 날짜 인덱스. DataFrame(breach_date/peak_date) 또는 날짜 시퀀스."""
    if episodes10 is None:
        return None
    if isinstance(episodes10, pd.DataFrame):
        if len(episodes10) == 0:
            return pd.DatetimeIndex([])
        for col in ("breach_date", "start_date"):
            if col in episodes10.columns:
                return pd.DatetimeIndex(pd.to_datetime(episodes10[col].dropna()))
        if "peak_date" in episodes10.columns:
            warnings.warn("episodes10 에 breach_date 가 없어 peak_date 를 시작일로 씁니다 "
                          "(scenarios.episode_starts 를 쓰면 돌파일이 옵니다)")
            return pd.DatetimeIndex(pd.to_datetime(episodes10["peak_date"].dropna()))
        raise ValueError("episodes10 에 breach_date·start_date·peak_date 중 하나가 있어야 합니다")
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(list(episodes10))))
    return idx


def _start_flags(idx: pd.Index, starts: pd.DatetimeIndex | None, h: int) -> np.ndarray | None:
    """세션 t 의 다음 h 세션 (t, t+h] 안에 ≥10% 에피소드 돌파일이 있는가 (idx 위치 기준)."""
    if starts is None or len(idx) == 0:
        return None
    if not isinstance(idx, pd.DatetimeIndex):
        return None
    n = len(idx)
    is_start = np.zeros(n, dtype=bool)
    if len(starts):
        pos = idx.get_indexer(starts)
        is_start[pos[pos >= 0]] = True
    c = np.concatenate([[0], np.cumsum(is_start)])
    lo = np.arange(1, n + 1)
    hi = np.minimum(lo + int(h), n)
    return (c[hi] - c[lo]) > 0


def _dist_stats(ret: np.ndarray, mdd: np.ndarray, *, block: int, n_boot: int, seed: int) -> dict:
    """수익·최대낙폭 분위수 + 중앙값 블록 부트스트랩 구간."""
    lo, hi = block_bootstrap_quantile(ret, 0.5, block=block, n_boot=n_boot, seed=seed)
    return {"ret_p10": _quantile(ret, 0.10), "ret_p50": _quantile(ret, 0.50), "ret_p90": _quantile(ret, 0.90),
            "ret_p50_lo": lo, "ret_p50_hi": hi,
            "mdd_p10": _quantile(mdd, 0.10), "mdd_p50": _quantile(mdd, 0.50)}


def bin_table(p, y, fwd_ret_20, fwd_maxdd_20, bins=BINS, min_n_eff: float = MIN_N_EFF,
              n_eff_div: int = N_EFF_DIV, episodes10=None, *, h: int = H, z: float = Z95,
              block: int = BOOT_BLOCK, n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """확률 구간 표(풀링). 행 = 표시 구간, 모든 행에 n·n_eff·Wilson·중앙값 구간이 있다.

    열: bin, bin_lo, bin_hi, pooled, n_bins, n, n_eff, mean_p, obs, wilson_lo, wilson_hi,
        ret_p10/p50/p90, ret_p50_lo/hi(40세션 블록 부트스트랩), mdd_p10, mdd_p50,
        share_ep10_start(+ep10_lo/hi), grey(n_eff < min_n_eff — 단독 표시 금지)

    `share_ep10_start` = 그 구간의 세션 중 **다음 h 세션 (t, t+h] 안에 ≥10% 에피소드가 시작(돌파)된**
    비율. `episodes10` 은 `episode_starts(close, 0.10)` 의 날짜들(또는 breach_date 열이 있는 표).
    """
    ps, ys = pd.Series(p), pd.Series(y)
    common_index = isinstance(ps.index, pd.DatetimeIndex) and isinstance(ys.index, pd.DatetimeIndex)
    if common_index:
        idx = ps.index.intersection(ys.index)
        for s in (fwd_ret_20, fwd_maxdd_20):
            if isinstance(s, pd.Series) and isinstance(s.index, pd.DatetimeIndex):
                idx = idx.intersection(s.index)
        idx = pd.DatetimeIndex(idx).sort_values()
        df = pd.DataFrame({
            "p": pd.to_numeric(ps.reindex(idx), errors="coerce"),
            "y": pd.to_numeric(ys.reindex(idx), errors="coerce"),
            "ret": pd.to_numeric(pd.Series(fwd_ret_20).reindex(idx), errors="coerce"),
            "mdd": pd.to_numeric(pd.Series(fwd_maxdd_20).reindex(idx), errors="coerce"),
        }, index=idx)
    else:
        arrs = [np.asarray(v, dtype=float).ravel() for v in (p, y, fwd_ret_20, fwd_maxdd_20)]
        if len({len(a) for a in arrs}) != 1:
            raise ValueError("p·y·fwd_ret_20·fwd_maxdd_20 의 길이가 다릅니다 (Series 로 주면 인덱스로 맞춥니다)")
        df = pd.DataFrame({"p": arrs[0], "y": arrs[1], "ret": arrs[2], "mdd": arrs[3]})
    ok = np.isfinite(df["p"].to_numpy()) & np.isfinite(df["y"].to_numpy())
    df = df.loc[ok]
    if len(df) == 0:
        raise ValueError("p·y 가 모두 유한한 행이 없습니다")

    ids = _bin_ids(df["p"].to_numpy(dtype=float), bins)
    edges = [float(e) for e in bins]
    n_per_bin = [int((ids == k).sum()) for k in range(len(edges) - 1)]
    groups = pool_bins(n_per_bin, min_n_eff=min_n_eff, n_eff_div=n_eff_div)

    starts = _start_dates(episodes10)
    flags = _start_flags(df.index, starts, h) if starts is not None else None
    if starts is not None and flags is None:
        warnings.warn("episodes10 이 주어졌으나 인덱스가 DatetimeIndex 가 아니라 share_ep10_start 를 계산하지 않습니다")

    rows = []
    for (a, b) in groups:
        m = (ids >= a) & (ids <= b)
        n = int(m.sum())
        n_eff = n / float(n_eff_div)
        obs = float(df["y"].to_numpy()[m].mean()) if n else float("nan")
        wl, wh = wilson(obs, n_eff, z=z) if n else (float("nan"), float("nan"))
        st = _dist_stats(df["ret"].to_numpy()[m], df["mdd"].to_numpy()[m], block=block, n_boot=n_boot, seed=seed)
        if flags is None:
            share, el, eh = float("nan"), float("nan"), float("nan")
        else:
            share = float(flags[m].mean()) if n else float("nan")
            el, eh = wilson(share, n_eff, z=z) if n else (float("nan"), float("nan"))
        rows.append({
            "bin": f"[{edges[a]:.2f}, {edges[b + 1]:.2f}{']' if b == len(edges) - 2 else ')'}",
            "bin_lo": edges[a], "bin_hi": edges[b + 1], "pooled": bool(b > a), "n_bins": int(b - a + 1),
            "n": n, "n_eff": n_eff, "mean_p": float(df["p"].to_numpy()[m].mean()) if n else float("nan"),
            "obs": obs, "wilson_lo": wl, "wilson_hi": wh, **st,
            "share_ep10_start": share, "ep10_lo": el, "ep10_hi": eh,
            "grey": bool(n_eff < min_n_eff),
        })
    out = pd.DataFrame(rows, columns=list(BIN_COLUMNS))
    attrs = {"n": int(len(df)), "n_eff": len(df) / float(n_eff_div), "base": float(df["y"].mean()),
             "min_n_eff": float(min_n_eff), "n_eff_div": int(n_eff_div), "bins": tuple(edges),
             "n_per_bin": tuple(n_per_bin), "groups": tuple(groups), "h": int(h),
             "block": int(block), "n_boot": int(n_boot), "seed": int(seed),
             "ep10_rule": f"(t, t+{int(h)}] 안에 ≥10% 에피소드 돌파" if flags is not None else "미계산"}
    if isinstance(df.index, pd.DatetimeIndex):
        attrs["start"], attrs["end"] = df.index[0], df.index[-1]
    out.attrs = attrs
    return out


# ------------------------------------------------------------------
# B. 결정 상태 표
# ------------------------------------------------------------------
def state_table(states, y, fwd_ret_20, fwd_maxdd_20, *, order=P2_STATES, min_n_eff: float = MIN_N_EFF,
                n_eff_div: int = N_EFF_DIV, z: float = Z95, block: int = BOOT_BLOCK,
                n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """결정 상태(normal/caution/reduce)별 표. 상태는 범주라 풀링하지 않는다 — 얇은 셀은 grey=True.

    열: state, n, n_eff, obs, wilson_lo, wilson_hi, ret_p10/p50/p90, ret_p50_lo/hi, mdd_p10, mdd_p50, grey
    """
    ss = pd.Series(states)
    ys = pd.Series(y)
    if isinstance(ss.index, pd.DatetimeIndex) and isinstance(ys.index, pd.DatetimeIndex):
        idx = ss.index.intersection(ys.index)
        for s in (fwd_ret_20, fwd_maxdd_20):
            if isinstance(s, pd.Series) and isinstance(s.index, pd.DatetimeIndex):
                idx = idx.intersection(s.index)
        idx = pd.DatetimeIndex(idx).sort_values()
        df = pd.DataFrame({
            "state": ss.reindex(idx).astype(object),
            "y": pd.to_numeric(ys.reindex(idx), errors="coerce"),
            "ret": pd.to_numeric(pd.Series(fwd_ret_20).reindex(idx), errors="coerce"),
            "mdd": pd.to_numeric(pd.Series(fwd_maxdd_20).reindex(idx), errors="coerce"),
        }, index=idx)
    else:
        arrs = [np.asarray(v, dtype=float).ravel() for v in (y, fwd_ret_20, fwd_maxdd_20)]
        if len({len(a) for a in arrs} | {len(ss)}) != 1:
            raise ValueError("states·y·fwd_ret_20·fwd_maxdd_20 의 길이가 다릅니다")
        df = pd.DataFrame({"state": np.asarray(ss, dtype=object), "y": arrs[0], "ret": arrs[1], "mdd": arrs[2]})
    df = df.loc[np.isfinite(df["y"].to_numpy()) & df["state"].notna()]
    if len(df) == 0:
        raise ValueError("state·y 가 모두 유효한 행이 없습니다")

    seen = list(dict.fromkeys(df["state"].astype(str).tolist()))
    unknown = [s for s in seen if s not in tuple(order)]
    if unknown:
        warnings.warn(f"등록되지 않은 상태값을 표 끝에 붙입니다: {unknown}")
    rows = []
    for st_name in list(order) + unknown:
        m = (df["state"].astype(str) == st_name).to_numpy()
        n = int(m.sum())
        n_eff = n / float(n_eff_div)
        obs = float(df["y"].to_numpy()[m].mean()) if n else float("nan")
        wl, wh = wilson(obs, n_eff, z=z) if n else (float("nan"), float("nan"))
        stats = _dist_stats(df["ret"].to_numpy()[m], df["mdd"].to_numpy()[m], block=block, n_boot=n_boot, seed=seed)
        rows.append({"state": st_name, "n": n, "n_eff": n_eff, "obs": obs, "wilson_lo": wl, "wilson_hi": wh,
                     **stats, "grey": bool(n_eff < min_n_eff)})
    out = pd.DataFrame(rows, columns=list(STATE_COLUMNS))
    out.attrs = {"n": int(len(df)), "n_eff": len(df) / float(n_eff_div), "base": float(df["y"].mean()),
                 "min_n_eff": float(min_n_eff), "n_eff_div": int(n_eff_div),
                 "block": int(block), "n_boot": int(n_boot), "seed": int(seed)}
    return out


def live_column(bins_tbl: pd.DataFrame, p_live, y_live, *, n_eff_div: int = N_EFF_DIV,
                z: float = Z95, min_n_eff: float | None = None) -> pd.DataFrame:
    """백테스트 구간 표의 **표시 행(풀링 결과)** 에 라이브 열을 붙인다 (§7 '주간 페이지엔 전체 + 라이브 열').

    추가 열: live_n, live_n_eff, live_obs, live_wilson_lo/hi, live_grey,
             calib_drift(라이브 비율이 **백테스트** Wilson 구간 밖 — '보정 드리프트', D8 과 별개 표시)
    라이브 행은 백테스트 표의 경계를 그대로 쓴다(라이브 표본으로 다시 풀링하지 않는다 — 행이 흔들리면 비교가 안 된다).
    """
    tbl = _as_frame(bins_tbl).copy()
    if not {"bin_lo", "bin_hi"} <= set(tbl.columns):
        raise ValueError("bins_tbl 에 bin_lo·bin_hi 가 필요합니다")
    if min_n_eff is None:
        min_n_eff = float((tbl.attrs or {}).get("min_n_eff", MIN_N_EFF))
    pv = pd.Series(p_live).astype(float)
    yv = pd.Series(y_live).astype(float)
    if isinstance(pv.index, pd.DatetimeIndex) and isinstance(yv.index, pd.DatetimeIndex):
        idx = pv.index.intersection(yv.index)
        pv, yv = pv.reindex(idx), yv.reindex(idx)
    p_arr, y_arr = pv.to_numpy(dtype=float), yv.to_numpy(dtype=float)
    ok = np.isfinite(p_arr) & np.isfinite(y_arr)
    p_arr, y_arr = p_arr[ok], y_arr[ok]
    lo_e = tbl["bin_lo"].to_numpy(dtype=float)
    hi_e = tbl["bin_hi"].to_numpy(dtype=float)
    rows = []
    for k in range(len(tbl)):
        last = k == len(tbl) - 1
        m = (p_arr >= lo_e[k]) & ((p_arr <= hi_e[k]) if last else (p_arr < hi_e[k]))
        n = int(m.sum())
        n_eff = n / float(n_eff_div)
        obs = float(y_arr[m].mean()) if n else float("nan")
        wl, wh = wilson(obs, n_eff, z=z) if n else (float("nan"), float("nan"))
        bl, bh = _num(tbl.iloc[k].get("wilson_lo")), _num(tbl.iloc[k].get("wilson_hi"))
        drift = bool(n and np.isfinite(obs) and np.isfinite(bl) and np.isfinite(bh) and not (bl <= obs <= bh))
        rows.append({"live_n": n, "live_n_eff": n_eff, "live_obs": obs, "live_wilson_lo": wl,
                     "live_wilson_hi": wh, "live_grey": bool(n_eff < min_n_eff), "calib_drift": drift})
    for col, vals in pd.DataFrame(rows, index=tbl.index).items():
        tbl[col] = vals
    return tbl


def assert_displayable(table: pd.DataFrame, *, name: str = "표") -> None:
    """표시 전 강제 검사: 모든 행에 n·n_eff·구간이 있어야 한다(§7 '모든 행에 n·n_eff·구간')."""
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"{name}: DataFrame 이어야 합니다")
    need = ("n", "n_eff", "wilson_lo", "wilson_hi", "grey")
    missing = [c for c in need if c not in table.columns]
    if missing:
        raise ValueError(f"{name}: 표시 금지 — 열이 없습니다 {missing}")
    if table["n"].isna().any() or table["n_eff"].isna().any():
        raise ValueError(f"{name}: 표시 금지 — n·n_eff 가 비어 있는 행이 있습니다")
    bad = table.loc[(table["n"] > 0) & (~np.isfinite(table["wilson_lo"].to_numpy(dtype=float)))]
    if len(bad):
        raise ValueError(f"{name}: 표시 금지 — 구간이 없는 행이 있습니다 ({len(bad)}행)")


# ------------------------------------------------------------------
# C. 에피소드 조건부
# ------------------------------------------------------------------
def episode_table(spy_close, threshold: float = DD_LEVELS[0], *, end=None, split: bool = EPISODE_SPLIT) -> pd.DataFrame:
    """에피소드별 원표(돌파일·추가손실·세션 수 포함). `end` 를 주면 종가를 그때까지 자른 뒤 센다.

    열: peak_date, breach_date(고점 대비 −threshold 를 처음 깨는 종가일), trough_date, recovery_date,
        depth, extra_loss(= 저점/(고점×(1−threshold)) − 1 — '−5% 선에서부터'의 추가 손실),
        extra_loss_close(= 저점/돌파종가 − 1 — 갭다운이 섞이는 판), peak_to_trough, breach_to_trough,
        trough_to_recovery, year(돌파 연도)
    """
    s = _cut(_clean_close(spy_close), end)
    ep = _episodes(s, threshold, split=split)
    idx, arr = s.index, s.to_numpy(dtype=float)
    rows = []
    for _, e in ep.iterrows():
        pi = int(idx.get_loc(e["peak_date"]))
        ti = int(idx.get_loc(e["trough_date"]))
        seg = arr[pi: ti + 1] / arr[pi] - 1.0
        below = np.flatnonzero(seg <= -threshold)
        bi = pi + int(below[0]) if below.size else ti
        ri = int(idx.get_loc(e["recovery_date"])) if pd.notna(e["recovery_date"]) else None
        rows.append({
            "peak_date": idx[pi], "breach_date": idx[bi], "trough_date": idx[ti],
            "recovery_date": idx[ri] if ri is not None else pd.NaT,
            "depth": float(e["depth"]),
            "extra_loss": (1.0 + float(e["depth"])) / (1.0 - threshold) - 1.0,
            "extra_loss_close": float(arr[ti] / arr[bi] - 1.0),
            "peak_to_trough": int(ti - pi), "breach_to_trough": int(ti - bi),
            "trough_to_recovery": float(ri - ti) if ri is not None else float("nan"),
            "year": int(idx[bi].year),
        })
    out = pd.DataFrame(rows, columns=list(EPISODE_TABLE_COLUMNS))
    if len(out) == 0:
        out = out.astype({"depth": float, "extra_loss": float, "extra_loss_close": float,
                          "peak_to_trough": int, "breach_to_trough": int, "trough_to_recovery": float, "year": int})
    out.attrs = {"threshold": float(threshold), "split": bool(split),
                 "start": idx[0], "end": idx[-1], "n_sessions": int(len(idx))}
    return out


def episode_starts(spy_close, threshold: float = 0.10, *, end=None, split: bool = EPISODE_SPLIT) -> pd.DatetimeIndex:
    """≥threshold 에피소드의 **시작(돌파)일** — `bin_table(..., episodes10=...)` 입력."""
    tbl = episode_table(spy_close, threshold, end=end, split=split)
    return pd.DatetimeIndex(tbl["breach_date"]) if len(tbl) else pd.DatetimeIndex([])


def _kn(k: int, n: int, z: float = Z95) -> dict:
    """(k, n, lo, hi) — 에피소드 표본은 에피소드 수가 곧 n_eff."""
    p_hat = (k / n) if n else float("nan")
    lo, hi = wilson(p_hat, float(n), z=z) if n else (float("nan"), float("nan"))
    return {"k": int(k), "n": int(n), "p": p_hat, "lo": lo, "hi": hi, "n_eff": float(n),
            "grey": bool(n < 1)}


def episode_conditionals(spy_close, *, end=None, levels=DD_LEVELS, split: bool = EPISODE_SPLIT,
                         states=None, z: float = Z95) -> dict:
    """−5% 에피소드가 시작됐을 때의 조건부 분포 (세기만 한다; 모든 값에 k·n·구간).

    반환: {n, base_level, levels, p_ge10_given5, p_ge20_given5, p_ge15_given10, p_ge20_given10,
           depth_q, extra_loss_q, extra_loss_close_q, breach_to_trough_q, peak_to_trough_q,
           trough_to_recovery_q, per_year, by_year, breach_state_counts, table, cond, span_years, start, end}

    각 확률은 `{"k","n","p","lo","hi","n_eff","grey"}` (Wilson, n_eff = 에피소드 수).
    `states` 를 주면 돌파일의 결정 상태 분포도 센다(표시용; 계산에는 쓰이지 않는다).
    """
    lv = tuple(float(x) for x in levels)
    if len(lv) < 2 or not all(0 < x < 1 for x in lv) or list(lv) != sorted(lv):
        raise ValueError(f"levels 는 (0,1) 오름차순 2개 이상이어야 합니다: {levels}")
    base = lv[0]
    tbl = episode_table(spy_close, base, end=end, split=split)
    n = int(len(tbl))
    depth = tbl["depth"].to_numpy(dtype=float) if n else np.array([])

    cond: dict[str, dict] = {}
    for L in lv[1:]:
        k = int((depth <= -L).sum())
        cond[f"p_ge{int(round(L * 100))}_given{int(round(base * 100))}"] = _kn(k, n, z=z)
    # ≥10% 에 조건부인 더 깊은 단계
    ref = lv[1] if len(lv) > 1 else base
    m_ref = depth <= -ref
    n_ref = int(m_ref.sum())
    for L in lv[2:]:
        k = int((depth[m_ref] <= -L).sum())
        cond[f"p_ge{int(round(L * 100))}_given{int(round(ref * 100))}"] = _kn(k, n_ref, z=z)

    def _q(col, qs):
        a = tbl[col].to_numpy(dtype=float) if n else np.array([])
        return {f"p{int(round(q * 100))}": _quantile(a, q) for q in qs}

    s0, s1 = tbl.attrs.get("start"), tbl.attrs.get("end")
    span_years = (float((pd.Timestamp(s1) - pd.Timestamp(s0)).days / 365.25)
                  if (s0 is not None and s1 is not None) else float("nan"))

    by_year = {int(k): int(v) for k, v in tbl["year"].value_counts().sort_index().items()} if n else {}
    breach_states = {}
    if states is not None and n:
        ss = pd.Series(states)
        got = ss.reindex(pd.DatetimeIndex(tbl["breach_date"]))
        breach_states = {str(k): int(v) for k, v in got.dropna().astype(str).value_counts().items()}
        n_missing = int(got.isna().sum())
        if n_missing:
            breach_states["미기록"] = n_missing

    out = {
        "n": n, "base_level": base, "levels": lv, "split": bool(split),
        "start": str(pd.Timestamp(s0).date()), "end": str(pd.Timestamp(s1).date()),
        "span_years": span_years,
        "per_year": (n / span_years) if (span_years and np.isfinite(span_years) and span_years > 0) else float("nan"),
        "by_year": by_year,
        "depth_q": _q("depth", (0.90, 0.75, 0.50, 0.25, 0.10)),
        "extra_loss_q": _q("extra_loss", (0.10, 0.25, 0.50)),
        "extra_loss_close_q": _q("extra_loss_close", (0.10, 0.25, 0.50)),
        "breach_to_trough_q": _q("breach_to_trough", (0.10, 0.50, 0.90)),
        "peak_to_trough_q": _q("peak_to_trough", (0.10, 0.50, 0.90)),
        "trough_to_recovery_q": _q("trough_to_recovery", (0.10, 0.50, 0.90)),
        "n_extra_worse_than_base": int((tbl["extra_loss"].to_numpy(dtype=float) <= -base).sum()) if n else 0,
        "n_recovered": int(tbl["recovery_date"].notna().sum()) if n else 0,
        "breach_state_counts": breach_states,
        "cond": cond,
        "table": tbl,
    }
    out.update(cond)                                        # 계약 키(p_ge10_given5 등)를 최상위에도
    for key in ("p_ge10_given5", "p_ge20_given5", "p_ge15_given10", "p_ge20_given10"):
        out.setdefault(key, _kn(0, 0, z=z))
    return out


def current_drawdown(spy_close, asof) -> dict:
    """asof 기준 ATH 대비 낙폭 상태 (표시 분기용).

    반환: {asof, close, ath, ath_date, dd_from_ath(오늘), min_dd_from_ath(ATH 이후 최악),
           breached_5, breached_10, breached{level: bool}, breach_date, sessions_since_breach, n_sessions}
    """
    s = _cut(_clean_close(spy_close), asof)
    if len(s) == 0:
        raise ValueError("asof 이전 종가가 없습니다")
    arr = s.to_numpy(dtype=float)
    idx = s.index
    run_max = np.maximum.accumulate(arr)
    ath = float(run_max[-1])
    ath_i = int(np.flatnonzero(arr >= ath)[0])              # 현재 ATH 에 처음 도달한 날
    seg = arr[ath_i:] / ath - 1.0
    min_dd = float(seg.min())
    dd = float(arr[-1] / ath - 1.0)
    breached = {}
    breach_i = None
    for L in DD_LEVELS:
        hit = np.flatnonzero(seg <= -L)
        breached[f"{int(round(L * 100))}"] = bool(hit.size > 0)
        if L == DD_LEVELS[0] and hit.size:
            breach_i = ath_i + int(hit[0])
    return {
        "asof": str(idx[-1].date()), "close": float(arr[-1]), "ath": ath, "ath_date": str(idx[ath_i].date()),
        "dd_from_ath": dd, "min_dd_from_ath": min_dd,
        "breached_5": bool(breached.get("5", False)), "breached_10": bool(breached.get("10", False)),
        "breached": breached,
        "breach_date": str(idx[breach_i].date()) if breach_i is not None else None,
        "sessions_since_breach": int(len(idx) - 1 - breach_i) if breach_i is not None else None,
        "n_sessions": int(len(idx)),
    }


# ------------------------------------------------------------------
# D. 시장 내재 / 예상 변동성 20세션 범위와 포함률
# ------------------------------------------------------------------
def _range_from_sigma(spot, sigma_ann, *, h: int, z: dict, src: str, what: str) -> dict:
    if h <= 0:
        raise ValueError(f"h 는 양수여야 합니다: {h}")
    sp, sg = _num(spot), _num(sigma_ann)
    if np.isfinite(sp) and sp <= 0:
        raise ValueError(f"spot 은 양수여야 합니다: {spot}")
    if np.isfinite(sg) and sg < 0:
        raise ValueError(f"{what} 은 음수일 수 없습니다: {sigma_ann}")
    ok = bool(np.isfinite(sp) and np.isfinite(sg))
    if not ok:
        warnings.warn(f"{src} 범위: 입력 결측(spot={spot}, {what}={sigma_ann}) → NaN 범위")
    s = sg * math.sqrt(h / float(TRADING_DAYS)) if ok else float("nan")
    out = {"s": s, "spot": sp, "sigma_ann": sg, "h": int(h), "src": src, "ok": ok, "pct": {}}
    for k in BAND_KEYS:
        zz = float(z[k])
        w = zz * s
        out[k] = (sp * math.exp(-w), sp * math.exp(w)) if ok else (float("nan"), float("nan"))
        out["pct"][k] = w if ok else float("nan")
    return out


def implied_range(spot: float, vix: float, h: int = H, z=None) -> dict:
    """시장 내재 h 세션 범위: s = VIX/100·sqrt(h/252), 밴드 = spot·exp(±z·s). 표시 ±% = z·s(로그 폭).

    반환 {"1s": (lo, hi), "80": (lo, hi), "90": (lo, hi), "s", "pct", "spot", "sigma_ann", "h", "src", "ok"}
    """
    z = Z if z is None else z
    v = _num(vix)
    if np.isfinite(v) and v < 0:
        raise ValueError(f"vix 는 음수일 수 없습니다: {vix}")
    if np.isfinite(v) and 0 < v < 1.0:
        warnings.warn(f"vix={vix} — 포인트(예: 14.53)가 아니라 비율로 보입니다")
    return _range_from_sigma(spot, v / 100.0 if np.isfinite(v) else float("nan"), h=h, z=z, src="vix", what="vix")


def har_range(spot: float, har_fc_20: float, h: int = H, z=None) -> dict:
    """적합 HAR 예측(연율 변동성) 기준 같은 범위 — 병기 전용(예산 밖 OLS 4개; 확률에 결합 금지)."""
    z = Z if z is None else z
    return _range_from_sigma(spot, _num(har_fc_20), h=h, z=z, src="har", what="har_fc_20")


def range_label(hit: float, n_eff: float, band: str = "1s", *, z: float = Z95) -> str:
    """포함률 라벨: Wilson 구간이 명목 위 → '보수적', 아래 → '낙관적', 걸치면 '중심'."""
    hv, ne = _num(hit), _num(n_eff)
    if not (np.isfinite(hv) and np.isfinite(ne) and ne > 0):
        return MISSING
    lo, hi = wilson(hv, ne, z=z)
    nom = NOMINAL[band]
    if lo > nom:
        return "보수적"
    if hi < nom:
        return "낙관적"
    return "중심"


def _coverage_one(lr: pd.Series, sigma_ann: pd.Series, *, h: int, z: dict, n_eff_div: int,
                  min_n_eff: float, zq: float, src: str) -> dict:
    s = pd.to_numeric(sigma_ann, errors="coerce") * math.sqrt(h / float(TRADING_DAYS))
    zz = (lr / s).replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(zz))
    n_eff = n / float(n_eff_div)
    out = {"src": src, "n": n, "n_eff": n_eff, "grey": bool(n_eff < min_n_eff)}
    a = zz.to_numpy(dtype=float)
    for k in BAND_KEYS:
        hit = float((np.abs(a) <= float(z[k])).mean()) if n else float("nan")
        lo, hi = wilson(hit, n_eff, z=zq) if n else (float("nan"), float("nan"))
        out[f"hit_{k}"] = hit
        out[f"hit_{k}_lo"], out[f"hit_{k}_hi"] = lo, hi
        out[f"label_{k}"] = range_label(hit, n_eff, k, z=zq)
    tail = float((a < -float(z["90"])).mean()) if n else float("nan")
    tlo, thi = wilson(tail, n_eff, z=zq) if n else (float("nan"), float("nan"))
    out["tail"] = tail
    out["tail_lo"], out["tail_hi"] = tlo, thi
    out["tail_nominal"] = (1.0 - NOMINAL["90"]) / 2.0
    out["label"] = out["label_1s"]
    return out


def coverage(fwd_ret_20, vix, har_fc, *, h: int = H, z=None, n_eff_div: int = N_EFF_DIV,
             min_n_eff: float = MIN_N_EFF, zq: float = Z95, rv20=None) -> dict:
    """실현 fwd_ret_h 가 각 밴드 안에 들어온 비율(과거·라이브 공통).

    z 통계 = ln(1 + fwd_ret_h) / s, s = σ_ann·sqrt(h/252). 반환 {"h","z","nominal","vix"{...},"har"{...},"rv_ratio"}.
    각 소스별로 hit_1s/80/90(+Wilson lo/hi·라벨), 왼꼬리(< −1.645s) 비율, n, n_eff(=n/20), grey 를 준다.
    """
    z = Z if z is None else z
    fr = pd.Series(fwd_ret_20).astype(float)
    if not isinstance(fr.index, pd.DatetimeIndex):
        raise TypeError("fwd_ret_20 은 DatetimeIndex Series 여야 합니다")
    fr = fr.dropna()
    if (fr <= -1.0).any():
        raise ValueError("fwd_ret_20 에 −100% 이하 수익률이 있습니다")
    lr = np.log1p(fr)
    vs = pd.to_numeric(pd.Series(vix), errors="coerce").reindex(fr.index)
    if np.isfinite(vs.to_numpy(dtype=float)).any() and float(np.nanmedian(vs.to_numpy(dtype=float))) < 1.0:
        warnings.warn("vix 중앙값 < 1 — 포인트가 아니라 비율로 보입니다")
    hs = pd.to_numeric(pd.Series(har_fc), errors="coerce").reindex(fr.index)
    out = {"h": int(h), "z": dict(z), "nominal": dict(NOMINAL), "n": int(len(fr)),
           "n_eff": len(fr) / float(n_eff_div), "n_eff_div": int(n_eff_div), "min_n_eff": float(min_n_eff)}
    out["vix"] = _coverage_one(lr, vs / 100.0, h=h, z=z, n_eff_div=n_eff_div, min_n_eff=min_n_eff, zq=zq, src="vix")
    out["har"] = _coverage_one(lr, hs, h=h, z=z, n_eff_div=n_eff_div, min_n_eff=min_n_eff, zq=zq, src="har")
    if rv20 is not None:
        rv = pd.to_numeric(pd.Series(rv20), errors="coerce").reindex(fr.index)
        rr = (rv / (vs / 100.0)).replace([np.inf, -np.inf], np.nan).dropna()
        rh = (rv / hs).replace([np.inf, -np.inf], np.nan).dropna()
        out["rv_ratio"] = {"rv_over_vix_mean": float(rr.mean()) if len(rr) else float("nan"),
                           "rv_over_vix_median": float(rr.median()) if len(rr) else float("nan"),
                           "rv_over_har_median": float(rh.median()) if len(rh) else float("nan"),
                           "n": int(len(rr))}
    else:
        out["rv_ratio"] = None
    return out


# ------------------------------------------------------------------
# 오늘 문맥 — 카드 3줄 (고정 순서: 시장 범위 → 오늘 같은 날 → 에피소드가 시작되면)
# ------------------------------------------------------------------
def _pick_bin_row(bins_tbl: pd.DataFrame, p_today: float) -> dict | None:
    if len(bins_tbl) == 0 or not np.isfinite(_num(p_today)):
        return None
    p = _num(p_today)
    lo = bins_tbl["bin_lo"].to_numpy(dtype=float)
    hi = bins_tbl["bin_hi"].to_numpy(dtype=float)
    m = (p >= lo) & (p < hi)
    if not m.any():
        m = (p >= lo) & (p <= hi)                 # 마지막 구간은 닫힘
    if not m.any():
        return None
    return bins_tbl.loc[m].iloc[0].to_dict()


def _line_market(vix, har_fc, spot, cov: dict, rng_vix: dict, rng_har: dict, notes: list) -> tuple[str, tuple[str, str]]:
    """카드 1줄: (원문, (쉬운 한국어, English)). 원문은 한 글자도 바꾸지 않는다 — 숫자는 셋 다 같은 값."""
    cv = (cov or {}).get("vix") or {}
    ch = (cov or {}).get("har") or {}
    v = _num(vix)
    head = f"시장이 보는 20일 범위(VIX {v:.1f})" if np.isfinite(v) else "시장이 보는 20일 범위(VIX 미상)"
    vix_txt = f"VIX {v:.1f}" if np.isfinite(v) else "VIX 미상"
    vix_txt_en = f"VIX {v:.1f}" if np.isfinite(v) else "VIX not available"
    if not rng_vix.get("ok"):
        notes.append("시장 내재 범위 계산 불가(VIX 또는 종가 결측)")
        left = f"{head}: {MISSING}"
        left_ko = (f"옵션 시장이 보는 앞으로 20일 범위는 구할 수 없습니다({vix_txt}). "
                   "VIX 나 종가가 없으면 범위를 지어내지 않습니다.")
        left_en = ("The 20-day range the options market expects cannot be computed "
                   f"({vix_txt_en}). Without VIX or a close we do not invent a range.")
    else:
        lab = cv.get("label_80") or range_label(cv.get("hit_80"), cv.get("n_eff"), "80")
        hit = _pct0(cv.get("hit_80"))
        left = (f"{head}: ±{_pct(rng_vix['pct']['80'])} (80%; 과거 {hit} 포함"
                + (f", {lab}" if lab != MISSING else "") + ")")
        lab_ko, lab_en = _LABEL_PLAIN.get(lab, ("", ""))
        left_ko = (f"옵션 시장이 보는 앞으로 20일 범위입니다({vix_txt}). 지금 값에서 위아래 "
                   f"±{_pct(rng_vix['pct']['80'])} 안에 들어올 가능성을 10번 중 8번(80%)으로 봅니다. "
                   f"과거에 실제로 그 범위 안에 들어온 날은 {hit} 였습니다"
                   + (f" — {lab_ko}." if lab_ko else "."))
        left_en = (f"This is the 20-day range the options market expects ({vix_txt_en}). It puts the chance "
                   f"of staying within ±{_pct(rng_vix['pct']['80'])} of today's level at 8 times out of 10 "
                   f"(80%). In the past {hit} of days actually stayed inside that range"
                   + (f" — {lab_en}." if lab_en else "."))
    if not rng_har.get("ok"):
        notes.append("예상 변동성(HAR) 범위 계산 불가")
        right = f"예상 변동성 기준 {MISSING}"
        right_ko = "예상 변동성으로 계산한 범위는 구할 수 없습니다."
        right_en = "The range from the expected swing cannot be computed."
    else:
        right = f"예상 변동성 기준 ±{_pct(rng_har['pct']['80'])} (과거 ~{_pct0(ch.get('hit_80'))})"
        right_ko = (f"예상 변동성으로 계산하면 범위는 ±{_pct(rng_har['pct']['80'])} 이고, 과거에 그 안에 "
                    f"들어온 날은 약 {_pct0(ch.get('hit_80'))} 였습니다.")
        right_en = (f"Using the expected swing instead, the range is ±{_pct(rng_har['pct']['80'])}, and in the "
                    f"past about {_pct0(ch.get('hit_80'))} of days stayed inside it.")
    return left + " · " + right, (left_ko + " " + right_ko, left_en + " " + right_en)


# 포함률 라벨(range_label 의 세 값)의 쉬운 말 — 원문 라벨은 그대로 두고 화면 문장에만 쓴다
_LABEL_PLAIN = {"보수적": ("범위를 넉넉하게 잡은 편입니다", "the range is on the generous side"),
                "낙관적": ("범위를 좁게 잡은 편입니다", "the range is on the tight side"),
                "중심": ("대체로 들어맞는 편입니다", "the range is about right")}


def _line_today(p_today, row: dict | None, base: float, min_n_eff: float,
                notes: list) -> tuple[str, tuple[str, str]]:
    """카드 2줄: (원문, (쉬운 한국어, English)). 표본 수·구간·평소 비율을 어느 쪽에서도 빼지 않는다."""
    if row is None:
        notes.append("오늘 확률에 해당하는 구간 행이 없다(확률 결측 또는 표 없음)")
        raw = (f"오늘 같은 날: 구간 표를 쓸 수 없다 — 전체 기저율 {_pct0(base)} "
               f"(20일 안에 {_MINUS}5%)")
        ko = (f"오늘 확률에 해당하는 줄이 표에 없어 오늘만의 숫자는 말하지 않습니다. 대신 평소 비율을 "
              f"씁니다: 20일 안에 5% 넘게 떨어진 날은 전체의 {_pct0(base)} 였습니다.")
        en = (f"Today's probability does not land on any row of the table, so we give no number just for "
              f"today. Here is the normal rate instead: {_pct0(base)} of all days fell more than 5% within "
              f"20 days.")
        return raw, (ko, en)
    n_eff = _num(row.get("n_eff"))
    if bool(row.get("grey")) or not (n_eff >= min_n_eff):
        notes.append(f"오늘 구간 n_eff {n_eff:.1f} < {min_n_eff:.0f} — 단독 표시 금지(풀링 규칙)")
        raw = (f"오늘 같은 날: 표본이 얇아(독립 창 {n_eff:.0f} < {min_n_eff:.0f}) 이 구간만으로는 말하지 않는다 — "
               f"전체 기저율 {_pct0(base)}")
        ko = (f"오늘 확률대는 표본이 얇습니다. 서로 겹치지 않는 창이 {n_eff:.0f}개뿐이라 기준인 "
              f"{min_n_eff:.0f}개에 못 미칩니다. 그래서 이 구간만으로는 말하지 않고 평소 비율을 씁니다: "
              f"{_pct0(base)}.")
        en = (f"The probability band for today has a thin sample: only {n_eff:.0f} non-overlapping windows, "
              f"below the {min_n_eff:.0f} we require. So we do not speak from that band alone and use the "
              f"normal rate instead: {_pct0(base)}.")
        return raw, (ko, en)
    obs = _num(row.get("obs"))
    k = obs * n_eff
    span = _span(row.get("bin_lo"), row.get("bin_hi"))
    wil = _span(row.get("wilson_lo"), row.get("wilson_hi"))
    p10 = _pct(row.get("ret_p10"), signed=True)
    p90 = _pct(row.get("ret_p90"), signed=True)
    mdd = _pct(row.get("mdd_p10"), signed=True)
    raw = (f"오늘 같은 날(확률대 {span}): "
           f"과거 독립 {n_eff:.0f}창 중 {k:.0f}창({wil})이 "
           f"20일 안에 {_MINUS}5%; 그때 20일 수익 10~90%: {p10}~{p90}, 최대낙폭 나쁜 10% {mdd}")
    # 분위수는 그 구간의 **모든 거래일**에서 계산한 값이다(bin_table 의 _dist_stats) — n_eff 개 창의
    # 값이 아니므로 "그 N번의 수익" 이라고 쓰지 않는다.
    ko = (f"오늘과 확률이 비슷했던 날(확률대 {span})은 과거에 서로 겹치지 않게 세어 {n_eff:.0f}번 "
          f"있었습니다. 그중 {k:.0f}번(95% 범위 {wil})이 20일 안에 5% 넘게 떨어졌습니다. "
          f"그 확률대의 날들을 보면 20일 뒤 수익은 10번 중 8번이 {p10}~{p90} 사이였고(10~90분위), "
          f"나빴던 10%는 고점 대비 하락폭이 {mdd} 보다 컸습니다.")
    en = (f"Days with a probability like today's (band {span}) happened {n_eff:.0f} times in the past, "
          f"counted in windows that do not overlap. {k:.0f} of them (95% range {wil}) fell more than 5% "
          f"within 20 days. Across the days in that band the 20-day return was between {p10} and {p90} "
          f"8 times out of 10 (10th-90th percentile), and on the worst 10% the drop from the peak was "
          f"bigger than {mdd}.")
    return raw, (ko, en)


def _cond(ep: dict, key: str) -> dict:
    c = (ep or {}).get(key) or ((ep or {}).get("cond") or {}).get(key) or {}
    if isinstance(c, (list, tuple)):                      # (k, n, lo, hi) 형태 허용
        k, n = (int(c[0]), int(c[1])) if len(c) >= 2 else (0, 0)
        lo, hi = _pair(c[2:4]) if len(c) >= 4 else (float("nan"), float("nan"))
        return {"k": k, "n": n, "p": (k / n) if n else float("nan"), "lo": lo, "hi": hi}
    return c


def _line_episode(ep: dict, dd: dict, notes: list) -> tuple[str, str, tuple[str, str]]:
    """카드 3줄: (분기, 원문, (쉬운 한국어, English)). 조건 프레이밍('만약 ~ 시작되면')을 반드시 유지한다."""
    ep = ep or {}
    dd = dd or {}
    q = ep.get("depth_q") or {}
    xq = ep.get("extra_loss_q") or {}
    b2t = ep.get("breach_to_trough_q") or {}
    t2r = ep.get("trough_to_recovery_q") or {}
    g10 = _cond(ep, "p_ge10_given5")
    g20 = _cond(ep, "p_ge20_given5")
    g15_10 = _cond(ep, "p_ge15_given10")
    g20_10 = _cond(ep, "p_ge20_given10")
    n = int(_num(ep.get("n")) if np.isfinite(_num(ep.get("n"))) else 0)
    if n == 0:
        notes.append("에피소드 표가 비어 있다")
        return ("empty", f"에피소드 조건부: {MISSING} (표본 없음)",
                ("하락 사건 표가 비어 있어 앞으로 벌어질 수 있는 일을 말하지 않습니다(표본 없음).",
                 "The table of decline episodes is empty, so we say nothing about what can happen next "
                 "(no sample)."))
    # 분기 = **오늘의** ATH 대비 낙폭(§7 '표시는 ATH 대비 현재 낙폭에 조건부').
    # dd_from_ath 가 없으면 이 underwater 구간의 돌파 여부로 물러선다.
    cur = _num(dd.get("dd_from_ath"))
    if np.isfinite(cur):
        breached_10 = cur <= -DD_LEVELS[1]
        breached_5 = cur <= -DD_LEVELS[0]
    else:
        notes.append("현재 낙폭 미상 — 돌파 플래그로 분기")
        breached_10 = bool(dd.get("breached_10"))
        breached_5 = bool(dd.get("breached_5"))
    sp10, sp20 = _span(g10.get("lo"), g10.get("hi")), _span(g20.get("lo"), g20.get("hi"))
    sp20_10 = _span(g20_10.get("lo"), g20_10.get("hi"))
    xq50, xq10 = _pct(xq.get("p50"), signed=True), _pct(xq.get("p10"), signed=True)
    if breached_10:
        branch = "breached_10"
        line = (f"지금 {_MINUS}10% 아래다: 과거 {_cnt(g15_10.get('n'))}번의 {_MINUS}10% 중 "
                f"{_cnt(g15_10.get('k'))}번은 {_MINUS}15%, {_cnt(g20_10.get('k'))}번"
                f"({sp20_10})은 {_MINUS}20% 까지 갔다; "
                f"저점까지 중앙 {_cnt(b2t.get('p50'))}세션, 저점에서 회복까지 중앙 {_cnt(t2r.get('p50'))}세션")
        ko = (f"지금은 고점보다 10% 넘게 내려온 상태입니다. 과거에 10% 넘게 떨어진 일은 "
              f"{_cnt(g15_10.get('n'))}번 있었습니다. 그중 {_cnt(g15_10.get('k'))}번은 15% 까지, "
              f"{_cnt(g20_10.get('k'))}번(95% 범위 {sp20_10})은 20% 까지 갔습니다. 바닥까지 걸린 기간은 "
              f"가운데가 {_cnt(b2t.get('p50'))}거래일이었고, 바닥에서 원래 값으로 돌아오기까지는 가운데가 "
              f"{_cnt(t2r.get('p50'))}거래일이었습니다.")
        en = (f"Right now the market is more than 10% below its peak. In the past a fall of 10% or more "
              f"happened {_cnt(g15_10.get('n'))} times. {_cnt(g15_10.get('k'))} of them went on to 15%, and "
              f"{_cnt(g20_10.get('k'))} (95% range {sp20_10}) went to 20%. The median time to the bottom was "
              f"{_cnt(b2t.get('p50'))} trading days, and the median time from the bottom back to the old "
              f"level was {_cnt(t2r.get('p50'))} trading days.")
    elif breached_5:
        branch = "breached_5"
        line = (f"지금 {_MINUS}5% 를 뚫었다: 과거 {_cnt(g10.get('n'))}번 중 {_cnt(g10.get('k'))}번"
                f"({sp10})이 {_MINUS}10% 까지, "
                f"{_cnt(g20.get('k'))}번({sp20})이 {_MINUS}20% 까지; "
                f"여기서 추가 손실 중앙 {xq50}, 나쁜 10% {xq10}; "
                f"저점까지 중앙 {_cnt(b2t.get('p50'))}세션")
        ko = (f"지금은 고점보다 5% 넘게 내려온 상태입니다. 과거에 그런 일은 {_cnt(g10.get('n'))}번 "
              f"있었습니다. 그중 {_cnt(g10.get('k'))}번(95% 범위 {sp10})은 10% 까지, "
              f"{_cnt(g20.get('k'))}번(95% 범위 {sp20})은 20% 까지 갔습니다. 여기서 더 떨어진 폭은 "
              f"가운데가 {xq50}, 나빴던 10%는 {xq10} 였습니다. 바닥까지는 가운데가 "
              f"{_cnt(b2t.get('p50'))}거래일 걸렸습니다.")
        en = (f"Right now the market is more than 5% below its peak. In the past that happened "
              f"{_cnt(g10.get('n'))} times. {_cnt(g10.get('k'))} of them (95% range {sp10}) went on to 10%, "
              f"and {_cnt(g20.get('k'))} (95% range {sp20}) went to 20%. From here the further fall was "
              f"{xq50} in the middle and {xq10} for the worst 10%. The median time to the bottom was "
              f"{_cnt(b2t.get('p50'))} trading days.")
    else:
        branch = "normal"
        line = (f"만약 {_MINUS}5% 에피소드가 시작되면: {_cnt(g10.get('n'))}번 중 {_cnt(g10.get('k'))}번"
                f"({sp10})은 {_MINUS}10% 까지, "
                f"{_cnt(g20.get('k'))}번({sp20})은 {_MINUS}20% 까지; "
                f"절반은 {_pct(q.get('p50'), signed=True)} 안에서 멈춘다; "
                f"추가 손실 중앙 {xq50}, 나쁜 10% {xq10}")
        ko = (f"5% 넘게 떨어지는 일이 시작된다면: 과거에 그런 일은 {_cnt(g10.get('n'))}번 있었습니다. "
              f"그중 {_cnt(g10.get('k'))}번(95% 범위 {sp10})은 10% 까지 갔고, "
              f"{_cnt(g20.get('k'))}번(95% 범위 {sp20})은 20% 까지 갔습니다. 절반은 "
              f"{_pct(q.get('p50'), signed=True)} 안에서 멈췄습니다. 시작된 뒤 더 떨어진 폭은 가운데가 "
              f"{xq50}, 나빴던 10%는 {xq10} 였습니다.")
        en = (f"If a fall of more than 5% starts: in the past that happened {_cnt(g10.get('n'))} times. "
              f"{_cnt(g10.get('k'))} of them (95% range {sp10}) went on to 10%, and {_cnt(g20.get('k'))} "
              f"(95% range {sp20}) went to 20%. Half of them stopped within "
              f"{_pct(q.get('p50'), signed=True)}. After the start the further fall was {xq50} in the middle "
              f"and {xq10} for the worst 10%.")
    return branch, line, (ko, en)


def today_context(p_today, state_today, vix, har_fc, spot, tables: dict, dd: dict) -> dict:
    """카드 '시나리오' 블록: 고정 3줄(시장 범위 → 오늘 같은 날 → 에피소드가 시작되면) + 오늘 행 + 라벨.

    `tables` = summary_p3.scenarios = {"bins": ..., "states": ..., "episodes": ..., "coverage": ...}
    (DataFrame 이든 JSON 왕복한 records 든 받는다). `dd` = current_drawdown(...) 결과.
    회색(n_eff < min_n_eff) 행은 절대 단독으로 문장에 쓰지 않는다.
    """
    tables = tables or {}
    notes: list[str] = []
    bins_tbl = _as_frame(tables.get("bins"))
    states_tbl = _as_frame(tables.get("states"))
    cov = tables.get("coverage") or {}
    ep = tables.get("episodes") or {}
    min_n_eff = float((bins_tbl.attrs or {}).get("min_n_eff", MIN_N_EFF)) if len(bins_tbl) else MIN_N_EFF
    base = float((bins_tbl.attrs or {}).get("base", float("nan"))) if len(bins_tbl) else float("nan")
    if not np.isfinite(base) and len(bins_tbl) and {"n", "obs"} <= set(bins_tbl.columns):
        w = bins_tbl["n"].to_numpy(dtype=float)
        o = bins_tbl["obs"].to_numpy(dtype=float)
        base = float(np.nansum(w * o) / np.nansum(w)) if np.nansum(w) > 0 else float("nan")

    rng_vix = implied_range(spot, vix)
    rng_har = har_range(spot, har_fc)
    l1, b1 = _line_market(vix, har_fc, spot, cov, rng_vix, rng_har, notes)

    row = _pick_bin_row(bins_tbl, p_today) if len(bins_tbl) else None
    l2, b2 = _line_today(p_today, row, base, min_n_eff, notes)

    branch, l3, b3 = _line_episode(ep, dd, notes)

    state_row = None
    if len(states_tbl) and state_today is not None and "state" in states_tbl.columns:
        m = states_tbl["state"].astype(str) == str(state_today)
        if m.any():
            state_row = states_tbl.loc[m].iloc[0].to_dict()
            if bool(state_row.get("grey")):
                notes.append(f"상태 '{state_today}' 는 n_eff {_num(state_row.get('n_eff')):.1f} < "
                             f"{min_n_eff:.0f} — 회색(단독 표시 금지)")
    headline = "vix"
    if rng_vix.get("ok") and rng_har.get("ok"):
        headline = "har" if _num(rng_har["pct"]["80"]) > _num(rng_vix["pct"]["80"]) else "vix"
    elif rng_har.get("ok"):
        headline = "har"
    return {
        # lines 는 원문(계산층 문자열) — 글자가 바뀌지 않았음을 test_scenarios.py 가 지킨다.
        # lines_bi 는 **화면용** 쉬운 한국어/영어 한 쌍이다(메모리 전용 · 산출물 JSON 에는 들어가지 않는다).
        "lines": [l1, l2, l3], "lines_bi": [b1, b2, b3], "line_keys": list(LINE_ORDER), "branch": branch,
        "range": {"vix": rng_vix, "har": rng_har, "headline": headline},
        "bin_row": row, "bin_grey": bool(row.get("grey")) if row else True,
        "state_row": state_row, "state_grey": bool(state_row.get("grey")) if state_row else True,
        "dd": dd, "p_today": _num(p_today), "state_today": None if state_today is None else str(state_today),
        "spread_sentence": SPREAD_SENTENCE, "base_rate": base, "min_n_eff": min_n_eff, "notes": notes,
    }
