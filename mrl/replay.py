# -*- coding: utf-8 -*-
"""mrl.replay — v0 판정의 일별 재현(replay) 과 125칸 점유표 (ARCHITECTURE.md 「재현 — mrl/replay.py」 계약).

계약
    replay_v0(bundle, start=BACKTEST_START, end=None, variant="faithful", basket="v0", progress=True) -> DataFrame
        인덱스 = 거래일(name="date"). 열: state_fang..state_btc(일간 상태 8개), score_d/w/m, trend_d/w/m,
        overall_d/w/m, tone, verdict_ko, n_watch_avail, fg_avail, eod_avail, n_leaders
        (+ 부가 열 spy_close, vix_close — 계약 열 뒤에 둔다).
    cell_table(replay) -> DataFrame
        (overall_m, overall_w, overall_d) 125칸 점유표: n(일수), share, tone, rule(규칙 번호), verdict_ko.

설계
  * 하루의 판정은 **signals_v0.v0_day 를 그대로 호출**한다 — 재현 로직을 여기서 다시 쓰지 않는다.
  * 속도: v0_day 의 window() 는 `close.loc[:asof]` 뒤 티커별 달력 창(S.WINDOW_SPAN: spy/vix/fang 2y, watch 1y,
    btc 는 자기 마지막 봉 기준 2y 포함 경계)만 읽는다. 그래서 asof 마다 "window 가 실제로 읽는 행"만 담은 축소
    Bundle 을 만들어 넘긴다(_WindowSlicer). 시작 행을 날짜로(하한 포함) 정해 축소 번들이 항상 달력 창의 상위집합이
    되게 하고, window() 가 그 안에서 다시 날짜로 자르므로 결과는 전체 번들과 **비트 단위로 동일**하다(verify_replay 로 검증).
    이것으로 하루 ~100ms → ~70ms. MACD(ewm) 는 창 시작값에 의존해 전 구간 벡터화가 불가능하므로
    (adjust=False 초기화), 결과 동일성을 지키는 선에서 여기까지만 최적화한다.
  * 점(point-in-time) 원칙: 축소 번들은 asof 이후 행을 담지 않는다. v0_day 자체도 asof 이후를 읽지 않는다.
  * 조용한 실패 금지: 하루라도 v0_day 가 실패하면 날짜를 붙여 RuntimeError. 일별 경고(F&G/EOD 결측 등)는
    날짜를 지운 범주별로 집계해 `df.attrs["warnings"]` 에 담고, 끝에 한 번 warnings.warn 으로 알린다.
"""
from __future__ import annotations

import itertools
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd

from mrl import signals_v0 as S
from mrl.config import BACKTEST_START, TONES, V0, V0_SC, V0_SIGNALS
from mrl.data import Bundle

__all__ = [
    "REPLAY_COLUMNS", "EXTRA_COLUMNS", "CELL_COLUMNS", "STATES", "RULE_NAMES_KO", "FALLBACK_NAMES_KO",
    "replay_v0", "cell_table", "verify_replay", "resolve_range", "day_bundle",
]

STATE_COLUMNS = [f"state_{k}" for k in V0_SIGNALS]
REPLAY_COLUMNS = (STATE_COLUMNS
                  + ["score_d", "score_w", "score_m", "trend_d", "trend_w", "trend_m",
                     "overall_d", "overall_w", "overall_m", "tone", "verdict_ko",
                     "n_watch_avail", "fg_avail", "eod_avail", "n_leaders"])
EXTRA_COLUMNS = ["spy_close", "vix_close"]
CELL_COLUMNS = ["overall_m", "overall_w", "overall_d", "n", "share", "tone", "rule", "verdict_ko"]
STATES = tuple(V0_SC)                      # GREEN, R2G, AMBER, G2R, RED (reference SC 순서)
# v0_combo(reference combo_advice) 규칙 1~10 의 국면명 — 순서가 곧 규칙 번호. 규칙 밖(평균 폴백)은 0.
RULE_NAMES_KO = (
    "시장 전체가 내리막",            # 1
    "바닥 신호가 살짝 보임",         # 2
    "바닥을 다지는 중",              # 3
    "반등이 확인됨",                 # 4
    "신호가 엇갈림",                 # 5
    "큰 흐름이 좋아지는 중",         # 6
    "상승장 속 잠깐 쉬어가는 구간",  # 7
    "단기 과열 뒤 주춤",             # 8
    "조정이 올 수 있음",             # 9
    "꾸준한 상승 흐름",              # 10
)
FALLBACK_NAMES_KO = ("좋은 쪽에 가까움", "나쁜 쪽에 가까움", "방향이 뚜렷하지 않음")   # 규칙 0 (평균 폴백)
PROGRESS_EVERY = 250
_CORE = ("SPY", "^VIX", "BTC-USD")          # signals_v0.window 가 반드시 요구하는 핵심 티커
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_SFX = {"daily": "d", "weekly": "w", "monthly": "m"}


# ------------------------------------------------------------------
# 구간 해석
# ------------------------------------------------------------------
def _ts(x, label: str) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    if pd.isna(ts):
        raise ValueError(f"{label} 를 날짜로 해석할 수 없음: {x!r}")
    if ts.tzinfo is not None:
        raise ValueError(f"계약 위반: {label} 는 tz-naive 여야 함")
    return ts.normalize()


def resolve_range(spy_idx: pd.DatetimeIndex, start=BACKTEST_START, end=None) -> pd.DatetimeIndex:
    """[start, end] 안의 SPY 거래일. start 가 거래일이 아니면 다음 거래일, end 가 아니면 이전 거래일.
    end=None 이면 캐시의 마지막 SPY 일자. 구간이 비거나 캐시 범위 밖이면 ValueError."""
    if not isinstance(spy_idx, pd.DatetimeIndex) or len(spy_idx) == 0:
        raise ValueError("spy_idx 가 비어 있거나 DatetimeIndex 가 아님")
    s = _ts(start, "start")
    e = spy_idx[-1] if end is None else _ts(end, "end")
    if s > e:
        raise ValueError(f"start({s:%Y-%m-%d}) > end({e:%Y-%m-%d})")
    if s < spy_idx[0] or e > spy_idx[-1]:
        raise ValueError(f"구간 {s:%Y-%m-%d}~{e:%Y-%m-%d} 가 캐시 범위 {spy_idx[0]:%Y-%m-%d}~{spy_idx[-1]:%Y-%m-%d} 밖")
    days = spy_idx[(spy_idx >= s) & (spy_idx <= e)]
    if len(days) == 0:
        raise ValueError(f"{s:%Y-%m-%d}~{e:%Y-%m-%d} 안에 SPY 거래일이 없음")
    return pd.DatetimeIndex(days, name="date")


# ------------------------------------------------------------------
# 축소 번들 (window 가 읽는 행만)
# ------------------------------------------------------------------
class _WindowSlicer:
    """asof 별로 signals_v0.window 가 실제로 읽는 행만 담은 Bundle 을 만든다.

    window() 의 달력 창 (S.WINDOW_SPAN):
      * spy/vix/fang = (asof - 2y, asof], watch = (asof - 1y, asof]           (asof 앵커, 시작 경계 배타)
      * btc = [마지막 BTC 봉(≤ asof) - 2y, 마지막 BTC 봉]                     (자기 봉 앵커, 포함 경계)
    시작 행은 두 하한 중 이른 쪽을 날짜로 찾아(searchsorted, 하한 포함) 정하므로 축소 번들은 항상 달력 창의
    상위집합이고, window() 가 그 안에서 다시 날짜로 자르므로 결과는 전체 번들과 비트 동일하다(verify_replay).
    `t in sub.columns` 로 상장 전 티커를 판정 → 열 집합을 바꾸지 않고 행만 줄인다(필요 열만 남김).
    eod/fg/cboe 는 작아서 그대로 넘긴다 (`.loc[:asof]` 비용이 미미).
    """

    def __init__(self, bundle: Bundle) -> None:
        close = bundle.close
        if not isinstance(close, pd.DataFrame) or not isinstance(close.index, pd.DatetimeIndex):
            raise TypeError("bundle.close 는 DatetimeIndex 를 가진 DataFrame 이어야 함")
        if not close.index.is_monotonic_increasing or not close.index.is_unique:
            raise ValueError("bundle.close 인덱스는 중복 없는 오름차순이어야 함")
        need = set(_CORE) | set(V0["fang"]) | set(V0["watchlist"])
        cols = [c for c in close.columns if c in need]        # close 의 열 순서 유지
        self.close = close[cols]
        self.cidx = close.index
        # BTC 는 자기 마지막 봉(≤ asof)이 앵커라 비-NaN 위치가 필요하다
        self.btc_pos = (np.flatnonzero(self.close["BTC-USD"].notna().to_numpy())
                        if "BTC-USD" in cols else np.array([], dtype=int))
        self.spy_ohlc = bundle.spy_ohlc
        self.bundle = bundle

    def start_row(self, ci: int) -> int:
        """close 위치 ci(asof) 기준, 모든 달력 창을 포함하는 시작 행 (하한 날짜 포함)."""
        ts = self.cidx[ci]
        lo = min(ts - S.WINDOW_SPAN[k] for k in ("spy", "vix", "fang", "watch"))
        c = int(np.searchsorted(self.btc_pos, ci, side="right"))     # asof 이하 BTC 봉 수
        if c > 0:
            lo = min(lo, self.cidx[self.btc_pos[c - 1]] - S.WINDOW_SPAN["btc"])
        return int(self.cidx.searchsorted(lo, side="left"))

    def bundle_for(self, ts: pd.Timestamp) -> Bundle:
        ci = self.cidx.get_loc(ts)
        if not isinstance(ci, (int, np.integer)):
            raise ValueError(f"close 인덱스에서 {ts:%Y-%m-%d} 위치가 유일하지 않음")
        ci = int(ci)
        close_sub = self.close.iloc[self.start_row(ci): ci + 1]
        si = int(self.spy_ohlc.index.get_loc(ts))
        lo_spy = ts - S.WINDOW_SPAN["spy"]
        spy_sub = self.spy_ohlc.iloc[int(self.spy_ohlc.index.searchsorted(lo_spy, side="left")): si + 1]
        return Bundle(close=close_sub, spy_ohlc=spy_sub, cboe=self.bundle.cboe,
                      fg=self.bundle.fg, eod=self.bundle.eod, meta=self.bundle.meta)


def day_bundle(bundle: Bundle, asof) -> Bundle:
    """asof 하루에 v0_day 가 읽는 행만 담은 축소 Bundle (테스트·검증용 공개 헬퍼)."""
    return _WindowSlicer(bundle).bundle_for(_ts(asof, "asof"))


# ------------------------------------------------------------------
# 행 변환 · 검증
# ------------------------------------------------------------------
def _row_from_day(out: dict) -> dict:
    row: dict = {}
    states = out["states_d"]
    for k in V0_SIGNALS:
        if k not in states:
            raise RuntimeError(f"{out['asof']}: states_d 에 {k} 가 없음")
        row[f"state_{k}"] = states[k]
    for x in ("d", "w", "m"):
        row[f"score_{x}"] = float(out[f"score_{x}"])
        row[f"trend_{x}"] = int(out[f"trend_{x}"])
        row[f"overall_{x}"] = out[f"overall_{x}"]
    row["tone"] = out["tone"]
    row["verdict_ko"] = out["verdict_ko"]
    row["n_watch_avail"] = int(out["n_watch_avail"])
    row["fg_avail"] = bool(out["fg_avail"])
    row["eod_avail"] = bool(out["eod_avail"])
    row["n_leaders"] = int(len(out["leaders"]))
    row["spy_close"] = float(out["spy_close"])
    row["vix_close"] = float(out["vix_close"])
    return row


def _validate_replay(df: pd.DataFrame) -> None:
    missing = [c for c in REPLAY_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"replay 열 누락: {missing}")
    if not df.index.is_unique or not df.index.is_monotonic_increasing:
        raise RuntimeError("replay 인덱스가 중복 없는 오름차순이 아님")
    na = df[REPLAY_COLUMNS].isna().sum()
    na = na[na > 0]
    if len(na):
        raise RuntimeError(f"replay 에 결측이 있음: {na.to_dict()}")
    bad_tone = sorted(set(df["tone"].unique()) - set(TONES))
    if bad_tone:
        raise RuntimeError(f"알 수 없는 톤: {bad_tone}")
    for c in STATE_COLUMNS + ["overall_d", "overall_w", "overall_m"]:
        bad = sorted(set(df[c].unique()) - set(STATES))
        if bad:
            raise RuntimeError(f"{c} 에 알 수 없는 상태: {bad}")


def _warn_key(msg: str) -> str:
    """일별 경고 문구에서 날짜를 지워 범주 키로 만든다."""
    return _DATE_RE.sub("<date>", str(msg)).strip()


def _tally(counts: dict, msg: str, ts: pd.Timestamp) -> None:
    key = _warn_key(msg)
    d = ts.strftime("%Y-%m-%d")
    rec = counts.get(key)
    if rec is None:
        counts[key] = {"n": 1, "first": d, "last": d}
    else:
        rec["n"] += 1
        rec["last"] = d


def _progress(label: str, i: int, n: int, ts: pd.Timestamp, t0: float) -> None:
    el = time.perf_counter() - t0
    rate = el / max(i, 1)
    eta = rate * (n - i)
    sys.stdout.write(f"[replay {label}] {i}/{n} ({i / n * 100:5.1f}%) {ts:%Y-%m-%d} · "
                     f"{el:6.1f}s 경과 · {rate * 1000:5.1f}ms/일 · 남은 ~{eta / 60:4.1f}분\n")
    sys.stdout.flush()


# ------------------------------------------------------------------
# 계약 함수
# ------------------------------------------------------------------
def replay_v0(bundle: Bundle, start=BACKTEST_START, end=None, variant: str = "faithful",
              basket: str = "v0", progress: bool = True) -> pd.DataFrame:
    """거래일마다 signals_v0.v0_day 를 호출해 v0 판정을 재현한다.

    반환 DataFrame: 인덱스 = 거래일(name="date"), 열 = REPLAY_COLUMNS + EXTRA_COLUMNS.
    attrs: variant, basket, start, end, n_days, runtime_sec, ms_per_day, warnings(dict: 범주 → n/first/last),
           window_rule(signals_v0.WINDOW_RULE — 재현에 쓴 창 규칙).
    실패(하루라도 v0_day 예외)는 날짜를 붙여 RuntimeError 로 올린다.
    """
    if variant not in S.VARIANTS:
        raise ValueError(f"variant 는 {S.VARIANTS} 중 하나여야 함: {variant!r}")
    if basket not in S.BASKETS:
        raise ValueError(f"basket 은 {S.BASKETS} 중 하나여야 함: {basket!r}")
    if not isinstance(bundle, Bundle):
        raise TypeError(f"bundle 은 mrl.data.Bundle 이어야 함: {type(bundle).__name__}")
    days = resolve_range(bundle.spy_ohlc.index, start, end)
    slicer = _WindowSlicer(bundle)
    n = len(days)
    label = f"{variant}/{basket}"
    rows: list[dict] = []
    warn_counts: dict[str, dict] = {}
    t0 = time.perf_counter()
    for i, ts in enumerate(days, start=1):
        sub = slicer.bundle_for(ts)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                out = S.v0_day(sub, ts, variant=variant, basket=basket)
            except Exception as e:      # noqa: BLE001 — 날짜를 붙여 다시 올린다 (조용히 건너뛰지 않음)
                raise RuntimeError(f"replay_v0 {ts:%Y-%m-%d} ({label}) v0_day 실패: {type(e).__name__}: {e}") from e
        seen = set()
        for w in caught:
            m = str(w.message)
            if m not in seen:
                seen.add(m)
                _tally(warn_counts, m, ts)
        for m in out.get("warnings", []):
            if m not in seen:
                seen.add(m)
                _tally(warn_counts, m, ts)
        rows.append(_row_from_day(out))
        if progress and (i % PROGRESS_EVERY == 0 or i == n):
            _progress(label, i, n, ts, t0)
    runtime = time.perf_counter() - t0

    df = pd.DataFrame(rows, index=days)
    df.index.name = "date"
    df = df[REPLAY_COLUMNS + EXTRA_COLUMNS]
    for c in ("score_d", "score_w", "score_m", "spy_close", "vix_close"):
        df[c] = df[c].astype(float)
    for c in ("trend_d", "trend_w", "trend_m", "n_watch_avail", "n_leaders"):
        df[c] = df[c].astype(int)
    for c in ("fg_avail", "eod_avail"):
        df[c] = df[c].astype(bool)
    _validate_replay(df)
    df.attrs.update({
        "variant": variant, "basket": basket,
        "start": days[0].strftime("%Y-%m-%d"), "end": days[-1].strftime("%Y-%m-%d"),
        "n_days": int(n), "runtime_sec": round(runtime, 2), "ms_per_day": round(runtime / n * 1000, 2),
        "warnings": warn_counts,
        "window_rule": S.WINDOW_RULE,        # 이 재현이 쓴 창 규칙 — 산출물에 기록해 다른 규칙의 CSV 재사용을 막는다
    })
    if warn_counts:
        summary = "; ".join(f"{k} ×{v['n']} ({v['first']}~{v['last']})" for k, v in warn_counts.items())
        if progress:
            sys.stdout.write(f"[replay {label}] 일별 경고 범주 {len(warn_counts)}건: {summary}\n")
            sys.stdout.flush()
        warnings.warn(f"replay_v0({label}) 일별 경고 {len(warn_counts)}범주 — attrs['warnings'] 참고: {summary[:400]}",
                      stacklevel=2)
    return df


def cell_table(replay: pd.DataFrame) -> pd.DataFrame:
    """(overall_m, overall_w, overall_d) 125칸 점유표.

    열: overall_m, overall_w, overall_d, n(일수), share(비율), tone, rule(v0_combo 규칙 번호 1~10, 0=평균 폴백), verdict_ko.
    관측되지 않은 칸도 n=0 으로 포함한다(총 125행). 정렬: n 내림차순, 같으면 상태 순서.
    replay 에 기록된 tone 이 그 칸의 v0_combo 결과와 다르면 ValueError (재현 결과 손상 탐지).
    """
    for c in ("overall_m", "overall_w", "overall_d", "tone"):
        if c not in replay.columns:
            raise ValueError(f"replay 에 '{c}' 열이 없음")
    n_total = int(len(replay))
    if n_total:
        grp = replay.groupby(["overall_m", "overall_w", "overall_d"], sort=False)
        counts = grp.size().to_dict()
        tones_obs = grp["tone"].agg(lambda s: tuple(sorted(set(map(str, s))))).to_dict()
    else:
        counts, tones_obs = {}, {}
    rows = []
    for order, (mo, wk, dy) in enumerate(itertools.product(STATES, repeat=3)):
        name, _action, tone = S.v0_combo(mo, wk, dy, "ko")
        if name in RULE_NAMES_KO:
            rule = RULE_NAMES_KO.index(name) + 1
        elif name in FALLBACK_NAMES_KO:
            rule = 0
        else:
            raise RuntimeError(f"v0_combo 국면명 {name!r} 이 RULE_NAMES_KO/FALLBACK_NAMES_KO 에 없음 — 규칙 목록 갱신 필요")
        key = (mo, wk, dy)
        n = int(counts.get(key, 0))
        if n and tones_obs.get(key) != (tone,):
            raise ValueError(f"칸 {key} 의 replay tone {tones_obs.get(key)} 이 v0_combo 결과 {tone!r} 과 다름")
        rows.append({"overall_m": mo, "overall_w": wk, "overall_d": dy, "n": n,
                     "share": (n / n_total) if n_total else 0.0, "tone": tone, "rule": int(rule),
                     "verdict_ko": name, "_order": order})
    df = pd.DataFrame(rows)
    df = df.sort_values(["n", "_order"], ascending=[False, True], kind="stable").drop(columns="_order")
    df = df[CELL_COLUMNS].reset_index(drop=True)
    if len(df) != 125 or int(df["n"].sum()) != n_total:
        raise RuntimeError("점유표 자기점검 실패 (125행 / 합계 불일치)")
    return df


def verify_replay(bundle: Bundle, replay: pd.DataFrame, n_samples: int = 30, seed: int = 0,
                  variant: str | None = None, basket: str | None = None, float_tol: float = 0.0) -> dict:
    """replay 의 표본 날짜를 **전체 번들**로 v0_day 를 다시 돌려 행이 동일한지 확인한다.

    축소 번들(_WindowSlicer)이 결과를 바꾸지 않았음을 증명하는 장치. 불일치가 하나라도 있으면 RuntimeError.
    variant/basket 은 replay.attrs 에서 읽되 인자가 있으면 그것을 쓴다.
    float_tol=0 이면 실수도 비트 동일을 요구한다(메모리 replay). CSV 에서 다시 읽은 replay(소수 6자리)는
    float_tol=1e-6 처럼 허용 오차를 주어 검증한다. 허용 오차는 |a-b| <= float_tol * max(1, |b|) —
    점수(|x|<=1)에는 절대, 가격(spy_close·vix_close)에는 상대 오차로 작동한다. 가격 캐시를 전량 재다운로드하면
    yfinance 조정 종가가 ~1e-7 상대 수준으로 흔들리므로(daily.py 뒤 --reuse-replay), 절대 1e-6 만으로는 깨진다.
    반환: {n_checked, dates, ms_per_day_full, variant, basket}.
    """
    variant = variant or replay.attrs.get("variant", "faithful")
    basket = basket or replay.attrs.get("basket", "v0")
    if float_tol < 0:
        raise ValueError("float_tol 은 0 이상이어야 함")
    if len(replay) == 0:
        raise ValueError("빈 replay 는 검증할 수 없음")
    rng = np.random.default_rng(seed)
    k = min(int(n_samples), len(replay))
    picks = sorted(rng.choice(len(replay), size=k, replace=False).tolist())
    mismatches = []
    t0 = time.perf_counter()
    for p in picks:
        ts = replay.index[p]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full = _row_from_day(S.v0_day(bundle, ts, variant=variant, basket=basket))
        got = replay.iloc[p]
        for c in REPLAY_COLUMNS + EXTRA_COLUMNS:
            a, b = full[c], got[c]
            if isinstance(a, float):
                fb = float(b)
                same = (a == fb) if float_tol == 0.0 else (abs(a - fb) <= float_tol * max(1.0, abs(fb)))
            else:
                same = (a == b)
            if not same:
                mismatches.append((ts.strftime("%Y-%m-%d"), c, a, b))
    el = time.perf_counter() - t0
    if mismatches:
        head = "; ".join(f"{d} {c}: 전체={a!r} vs replay={b!r}" for d, c, a, b in mismatches[:8])
        raise RuntimeError(f"replay 검증 실패 {len(mismatches)}건 ({variant}/{basket}): {head}")
    return {"n_checked": k, "dates": [replay.index[p].strftime("%Y-%m-%d") for p in picks],
            "ms_per_day_full": round(el / k * 1000, 1), "variant": variant, "basket": basket}
