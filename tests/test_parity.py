# -*- coding: utf-8 -*-
"""v0 동결(parity) 테스트 — reference/market_dashboard_v0.py 와 mrl.signals_v0 의 비트 단위 동일성.

검사 내용
  * mrl.config.V0* 상수 == reference CONFIG/WEIGHTS/SC/DAILY_ONLY (공유 키)
  * 실캐시(data/)로 만든 asof 별 2y 창에 대해
      - window() 가 fetch_real 이 보던 data dict 와 프레임 단위로 동일한가 (테스트가 독립적으로 재구성)
      - macd_phase(phase, streak, hist) / ret / build_metrics 의 모든 키 / assess / composite / overall / combo_advice
        결과가 비트 단위로 동일한가
      - v0_day(variant="faithful") 가 reference build_payload 의 계산 흐름과 같은 상태·점수·판정을 내는가
  * 대체 규약(fg/eod/lead 결측 시 가중치 제외), 점 원칙(미래 행 무관), completed 변형, 오류 경로

reference 모듈은 파일 경로로 import 한다 (최상위는 함수 정의뿐이고 실행 코드는 __main__ 가드 안에 있다;
import 시 warnings.filterwarnings("ignore") 를 호출하므로 catch_warnings 안에서 실행해 전역 필터 오염을 막는다).
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrl import config as C                      # noqa: E402
from mrl import signals_v0 as S                  # noqa: E402
from mrl.data import Bundle, load_cache          # noqa: E402

REF_PATH = ROOT / "reference" / "market_dashboard_v0.py"
# 파리티 검사 기준일: 상승장(2024-03), 급락장(2025-04 관세 충격), 최근(2026-08), 캐시 마지막 날(2026-09-04)
ASOF_DATES = ["2024-03-15", "2025-04-08", "2026-08-28", "2026-09-04"]
TFS = ("daily", "weekly", "monthly")
STATES = ("GREEN", "R2G", "AMBER", "G2R", "RED")


# ------------------------------------------------------------------
# 픽스처
# ------------------------------------------------------------------
@pytest.fixture(scope="module")
def ref():
    """reference 를 파일 경로로 import. __main__ 블록은 실행되지 않는다."""
    spec = importlib.util.spec_from_file_location("market_dashboard_v0_parity", REF_PATH)
    mod = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():              # reference 의 filterwarnings("ignore") 가 새지 않게
        spec.loader.exec_module(mod)
    assert mod.__name__ != "__main__"
    for fn in ("resample_close", "macd_phase", "ret", "build_metrics", "assess", "composite", "overall", "combo_advice"):
        assert callable(getattr(mod, fn)), fn
    return mod


@pytest.fixture(scope="module")
def bundle():
    """실캐시. 없으면 조용히 skip 하지 않고 실패한다 — parity 통과가 '동결 성공' 의 정의이기 때문."""
    missing = [n for n in ("close.csv", "spy_ohlc.csv", "fg_history.csv", "spy_eod.csv", "meta.json")
               if not (C.DATA_DIR / n).exists()]
    if missing:
        pytest.fail(f"캐시 파일 없음 {missing} — scripts/build_cache.py 를 먼저 실행하라")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # 가드 경고(유령 행 등)는 여기서 관심 밖
        return load_cache(C.DATA_DIR)


# ------------------------------------------------------------------
# 도우미
# ------------------------------------------------------------------
def reference_data(bundle: Bundle, asof: str, ref) -> dict:
    """fetch_real 이 asof 시점에 만들었을 data dict 를 캐시에서 **독립적으로** 재구성 (S.window 를 쓰지 않는다).

    Yahoo period 창의 실제 규칙(tests/fixtures/fetch_real_*.json 으로 검증): 마지막 봉 기준 달력 창, 시작 경계 배타.
    spy OHLC · vix · fang = (asof-2y, asof] → fang 은 DataFrame().dropna() · watch = (asof-1y, asof] → DataFrame() ·
    btc = [마지막 BTC 봉-2y, 마지막 BTC 봉] 포함 경계 · eod = 반일장 제외 마지막 30세션의 close/open-1 > eod_ret ·
    fg = asof 점수와 1/5/21거래일 전 대비. (이 재구성은 규칙의 **독립 구현**일 뿐이고, 규칙이 라이브와 같은지는
    test_window_reproduces_recorded_fetch_real 이 기록된 실제 페이로드로 검사한다.)
    """
    ts = pd.Timestamp(asof)
    spy_idx = bundle.spy_ohlc.index
    assert ts in spy_idx
    sub = bundle.close.loc[:ts]
    two_y, one_y = pd.DateOffset(years=2), pd.DateOffset(years=1)

    def span(t, off):
        s = sub[t].dropna()
        return s[s.index > ts - off]

    spy = bundle.spy_ohlc.loc[:ts]
    spy = spy[spy.index > ts - two_y]
    vix = span("^VIX", two_y)
    b = sub["BTC-USD"].dropna()
    btc = b[b.index >= b.index[-1] - two_y]
    fang = pd.DataFrame({t: span(t, two_y) for t in ref.CONFIG["fang"] if len(span(t, two_y))}).dropna()
    watch = pd.DataFrame({t: span(t, one_y) for t in ref.CONFIG["watchlist"] if len(span(t, one_y))})

    e = bundle.eod.loc[:ts]
    e = e[~e["half_day"].astype(bool)].iloc[-30:]
    ret30 = e["close"].astype(float) / e["open"].astype(float) - 1
    idx30 = pd.DatetimeIndex(e.index)
    eod_bool = pd.Series((ret30 > ref.CONFIG["eod_ret"]).values, index=idx30)
    eod_vals = pd.Series((ret30 * 100).values, index=idx30)

    fg = None
    sc = bundle.fg["score"].astype(float)
    if ts in sc.index and np.isfinite(sc.at[ts]):
        score = float(sc.at[ts])
        pos = spy_idx.get_loc(ts)

        def delta(k):
            hit = sc.loc[:spy_idx[pos - k]].dropna()
            return round(score - float(hit.iloc[-1])) if len(hit) else 0

        fg = {"score": round(score), "d_daily": delta(1), "d_weekly": delta(5), "d_monthly": delta(21),
              "hist": [float(x) for x in sc.loc[:ts].dropna().iloc[-180:]]}
    return {"spy": spy, "vix": vix, "btc": btc, "fang": fang, "watch": watch,
            "eod": eod_bool, "eod_vals": eod_vals, "fg": fg}


def reference_day(ref, data: dict):
    """reference build_payload 의 계산 부분(신호 → 종합 → combo)을 reference 함수로만 재실행."""
    tfs, states, mets = {}, {}, {}
    for tf in TFS:
        m_now = ref.build_metrics(data, tf, cut=0)
        m_prev = ref.build_metrics(data, tf, cut=ref.CONFIG["trend_cut"][tf])
        st_now, st_prev = ref.assess(m_now), ref.assess(m_prev)
        if tf != "daily":
            for k in ref.DAILY_ONLY:
                st_now.pop(k, None)
                st_prev.pop(k, None)
        fg_missing = m_now["fg"] is None
        sc_now = ref.composite(st_now, fg_missing)
        sc_prev = ref.composite(st_prev, fg_missing)
        dsc = sc_now - sc_prev
        trend = 1 if dsc > ref.CONFIG["trend_eps"] else -1 if dsc < -ref.CONFIG["trend_eps"] else 0
        tfs[tf] = {"score": sc_now, "trend": trend, "overall": ref.overall(sc_now, trend)}
        states[tf], mets[tf] = st_now, m_now
    mo, wk, dy = (tfs[t]["overall"] for t in ("monthly", "weekly", "daily"))
    return tfs, states, mets, ref.combo_advice(mo, wk, dy, "ko")


def same_scalar(a, b) -> bool:
    """비트 단위 동일(타입까지). float NaN 끼리는 동일로 본다."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    return a == b


def assert_metrics_identical(m_ref: dict, m_mrl: dict, label: str) -> None:
    assert list(m_ref) == list(m_mrl), f"{label}: 메트릭 키/순서 다름"
    for k in m_ref:
        a, b = m_ref[k], m_mrl[k]
        if isinstance(a, list):
            assert a == b, f"{label}: {k} {a} != {b}"
        else:
            assert same_scalar(a, b), f"{label}: {k} {a!r}({type(a).__name__}) != {b!r}({type(b).__name__})"


def deep_equal(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return list(a) == list(b) and all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    return same_scalar(a, b)


# ------------------------------------------------------------------
# 상수 동결
# ------------------------------------------------------------------
def test_constants_match_reference(ref):
    shared = set(C.V0) & set(ref.CONFIG)
    for k in shared:
        assert C.V0[k] == ref.CONFIG[k], k
    # V0 는 재현용 watch_period 만 추가로 가진다; 포팅이 쓰는 CONFIG 키는 모두 V0 에 있어야 한다
    assert set(C.V0) - set(ref.CONFIG) == {"watch_period"}
    for k in ("fang", "watchlist", "period", "ma_window", "lookback", "turn_k", "eod_ret", "trend_cut", "trend_eps", "band"):
        assert k in C.V0 and k in ref.CONFIG, k
    assert C.V0_WEIGHTS == ref.WEIGHTS and list(C.V0_WEIGHTS) == list(ref.WEIGHTS)
    assert C.V0_SC == ref.SC and list(C.V0_SC) == list(ref.SC)
    assert tuple(C.V0_DAILY_ONLY) == tuple(ref.DAILY_ONLY)
    assert tuple(C.V0_SIGNALS) == tuple(ref.WEIGHTS)
    assert S.BTC_MOM_K == {"daily": 7, "weekly": 4, "monthly": 3}   # build_metrics 인라인 상수
    assert S.MIN_WATCH_BARS == 65 and S.FG_HIST_N == 180


# ------------------------------------------------------------------
# 창 재현 — window() vs 독립 재구성
# ------------------------------------------------------------------
@pytest.mark.parametrize("asof", ASOF_DATES)
def test_window_matches_fetch_real_emulation(bundle, ref, asof):
    data = reference_data(bundle, asof, ref)
    win = S.window(bundle, asof, "faithful")
    ts = pd.Timestamp(asof)
    assert win["asof"] == ts and win["variant"] == "faithful"
    pd.testing.assert_frame_equal(win["spy"], data["spy"], check_exact=True)
    pd.testing.assert_series_equal(win["vix"], data["vix"], check_exact=True)
    pd.testing.assert_series_equal(win["btc"], data["btc"], check_exact=True)
    pd.testing.assert_frame_equal(win["fang"], data["fang"], check_exact=True)
    pd.testing.assert_frame_equal(win["watch"], data["watch"], check_exact=True)
    pd.testing.assert_series_equal(win["eod"], data["eod"], check_exact=True)
    pd.testing.assert_series_equal(win["eod_vals"], data["eod_vals"], check_exact=True)
    assert win["fg"] == data["fg"]
    # 창 규칙(달력 창, 시작 경계 배타)과 점 원칙 — 행 수는 고정값이 아니라 규칙에서 나온다
    two_y = pd.DateOffset(years=2)
    for k in ("spy", "vix", "fang"):
        first = win[k].index[0]
        assert first > ts - two_y and 500 <= len(win[k]) <= 507, k
        prev_bar = bundle.close["^VIX" if k == "vix" else "SPY"].dropna().loc[:first].index[-2]   # 창 직전 봉은 경계 밖
        assert prev_bar <= ts - two_y, k
    assert win["btc"].index[0] >= win["btc"].index[-1] - two_y and 725 <= len(win["btc"]) <= 732
    assert list(win["fang"].columns) == ref.CONFIG["fang"]
    assert len(win["eod"]) == S.EOD_SESSIONS and win["eod"].dtype == bool
    for k in ("spy", "vix", "btc", "fang", "watch", "eod", "eod_vals"):
        assert win[k].index.max() <= ts, k
    assert win["spy"].index[-1] == ts and win["vix"].index[-1] == ts and win["btc"].index[-1] == ts
    assert win["meta"]["fg_avail"] and win["meta"]["eod_avail"]
    assert win["meta"]["n_watch_avail"] == sum(len(data["watch"][t].dropna()) >= 65 for t in data["watch"].columns)
    assert win["meta"]["n_watch_avail"] >= 26


# ------------------------------------------------------------------
# 기록된 실제 fetch_real 페이로드와 대조 — 포팅 상수가 아니라 라이브 Yahoo/CNN 응답의 모양이 기준
# ------------------------------------------------------------------
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
# 라이브 CNN 의 previous_close/1_week/1_month 는 SPY 1/5/21세션 전 점수가 아니다 (기록: 라이브 +7/-10/-18 vs 포팅 -2/·/·).
# 포팅은 이 근사를 의도적으로 유지한다(VALIDATION.md §3 대체 규약의 연장) — 여기서는 라이브 값과 포팅 값을 둘 다 고정해
# 어느 쪽이 바뀌어도 드러나게 한다.
FG_DELTA_PINNED_PORT = {"2026-09-04": {"d_daily": -2, "d_weekly": -10, "d_monthly": -17}}


def _fixtures() -> list[Path]:
    return sorted(FIXTURE_DIR.glob("fetch_real_*.json"))


@pytest.mark.parametrize("fixture_path", _fixtures() or [None], ids=lambda p: p.name if p else "missing")
def test_window_reproduces_recorded_fetch_real(bundle, fixture_path):
    """window(bundle, spy_last) 가 기록된 실제 fetch_real() 페이로드의 창 모양을 재현하는가.

    * 기록 자체가 Yahoo 규칙 '(마지막 봉 - 2y, 마지막 봉] 배타 시작' 을 캐시 달력으로 만족하는지 먼저 검사한다.
    * 라이브 앵커가 SPY 마지막 봉과 같은 프레임(spy·fang·watch)은 행 수·첫/마지막 일자가 정확히 같아야 한다.
    * 휴장일 유령 봉(예: ^VIX 2026-09-07)으로 라이브 앵커가 뒤로 밀린 프레임은 재현할 수 없는 라이브 결함이므로,
      포팅의 창이 asof 앵커 규칙과 정확히 같은지만 확인하고 어긋남을 명시한다.
    * btc 는 자기 마지막 봉 앵커 포함 경계(731행) 규칙을 양쪽에 검사한다. eod 는 세션 일자·bool 값이 겹치는 구간에서 같아야 한다.
    * fg 는 점수는 같아야 하고, d_* 는 라이브 값과 포팅 값을 각각 고정한다(위 주석).
    """
    if fixture_path is None:
        pytest.fail("tests/fixtures/fetch_real_*.json 이 없음 — python tests/fixtures/record_fetch_real.py 로 기록하라")
    with open(fixture_path, encoding="utf-8") as f:
        fx = json.load(f)
    spy_last = pd.Timestamp(fx["spy"]["last"])
    two_y, one_y = pd.DateOffset(years=2), pd.DateOffset(years=1)
    close = bundle.close

    def rule_rows(series: pd.Series, anchor: pd.Timestamp, span, inclusive: bool = False) -> pd.Series:
        s = series.dropna()
        s = s[s.index <= max(anchor, spy_last)]
        return s[(s.index >= anchor - span) if inclusive else (s.index > anchor - span)]

    # 1) 기록이 Yahoo 규칙을 만족하는가 (캐시 달력 기준; 유령 봉은 캐시에 없으므로 spy_last 이후 행은 무시)
    for k, col, span in (("spy", "SPY", two_y), ("vix", "^VIX", two_y), ("fang", "META", two_y)):
        rec = fx[k]
        anchor = pd.Timestamp(rec["last"])
        exp = rule_rows(close[col], anchor, span)
        assert pd.Timestamp(rec["first"]) == exp.index[0], (k, "라이브 창 시작이 규칙과 다름")
        # 유령 봉(휴장일 앵커)은 기록 당시 캐시엔 없었지만 캐시가 전진하면 포함된다
        # (apply_guards 가 'SPY 마지막 일자 이후'만 잘라내므로) — 재구성에 이미 들어왔으면 더하지 않는다.
        phantom = 1 if (anchor > spy_last and anchor not in close[col].dropna().index) else 0
        assert rec["rows"] == len(exp) + phantom, (k, "라이브 행 수가 규칙과 다름")
    b_rec = fx["btc"]
    assert pd.Timestamp(b_rec["first"]) == pd.Timestamp(b_rec["last"]) - two_y and b_rec["rows"] in (731, 732)
    for t, rec in fx["watch"]["per_column"].items():
        exp = rule_rows(close[t], pd.Timestamp(rec["last"]), one_y)
        assert rec["rows"] == len(exp) and pd.Timestamp(rec["first"]) == exp.index[0], t

    # 2) 포팅의 창
    win = S.window(bundle, spy_last, "faithful")
    for k in ("spy", "fang"):                                # 라이브 앵커 == SPY 마지막 봉 → 정확히 일치
        assert fx[k]["last"] == fx["spy"]["last"], k
        assert (len(win[k]), win[k].index[0], win[k].index[-1]) == \
               (fx[k]["rows"], pd.Timestamp(fx[k]["first"]), pd.Timestamp(fx[k]["last"])), k
    assert list(win["fang"].columns) == fx["fang"]["columns"]
    vix_anchor = pd.Timestamp(fx["vix"]["last"])
    exp_vix = rule_rows(close["^VIX"], spy_last, two_y)
    assert (len(win["vix"]), win["vix"].index[0], win["vix"].index[-1]) == (len(exp_vix), exp_vix.index[0], spy_last)
    if vix_anchor == spy_last:
        assert (len(win["vix"]), win["vix"].index[0]) == (fx["vix"]["rows"], pd.Timestamp(fx["vix"]["first"]))
    else:                                                    # 유령 봉 앵커: 라이브 창이 규칙대로 뒤로 밀린 것을 명시
        assert vix_anchor > spy_last and pd.Timestamp(fx["vix"]["first"]) > win["vix"].index[0]
    assert win["btc"].index[0] == win["btc"].index[-1] - two_y and len(win["btc"]) in (731, 732)
    assert list(win["watch"].columns) == fx["watch"]["columns"]
    for t, rec in fx["watch"]["per_column"].items():
        s = win["watch"][t].dropna()
        if pd.Timestamp(rec["last"]) == spy_last:
            assert (len(s), s.index[0], s.index[-1]) == (rec["rows"], pd.Timestamp(rec["first"]), spy_last), t
    # 3) eod: 30 세션(반일장 제외) — 겹치는 세션의 bool 값은 같고, 라이브에만 있는 세션은 반일장뿐
    fx_eod = pd.Series(fx["eod"]["values"], index=pd.DatetimeIndex(fx["eod"]["dates"]))
    fx_eod = fx_eod[fx_eod.index <= spy_last]
    half_days = set(bundle.eod.index[bundle.eod["half_day"].astype(bool)])
    only_live = set(fx_eod.index) - set(win["eod"].index)
    assert only_live <= half_days, f"라이브에만 있는 EOD 세션이 반일장이 아님: {sorted(only_live)}"
    common = fx_eod.index.intersection(win["eod"].index)
    assert len(common) >= 20 and (win["eod"].loc[common].astype(bool) == fx_eod.loc[common].astype(bool)).all()
    assert len(win["eod"]) == S.EOD_SESSIONS == len(fx["eod"]["dates"])
    # 4) fg: 점수 동일, 델타는 라이브/포팅 값을 각각 고정
    if fx["fg"] is not None:
        assert win["fg"] is not None and win["fg"]["score"] == fx["fg"]["score"]
        pinned = FG_DELTA_PINNED_PORT.get(fx["spy"]["last"])
        if pinned is not None:
            assert {k: win["fg"][k] for k in pinned} == pinned, "포팅 F&G 델타(1/5/21세션 근사)가 고정값과 다름"
        # 라이브 d_* (fx["fg"]) 는 기록 그대로 두어 근사의 크기를 남긴다 — 2026-09-04: 라이브 +7/-10/-18 vs 포팅 -2/-10/-17


# ------------------------------------------------------------------
# 순수 함수 파리티
# ------------------------------------------------------------------
@pytest.mark.parametrize("asof", ASOF_DATES)
def test_resample_macd_ret_bit_identical(bundle, ref, asof):
    data = reference_data(bundle, asof, ref)
    for tf in TFS:
        for name in ("spy", "vix", "btc"):
            s = data[name]["Close"] if name == "spy" else data[name]
            r_ref, r_mrl = ref.resample_close(s, tf), S.resample_close(s, tf)
            pd.testing.assert_series_equal(r_ref, r_mrl, check_exact=True)
            # macd_phase: (phase, streak, hist) 모두 동일
            p_ref, k_ref, h_ref = ref.macd_phase(r_ref, ref.CONFIG["turn_k"][tf])
            p_mrl, k_mrl, h_mrl = S.macd_phase(r_mrl, C.V0["turn_k"][tf])
            assert p_ref == p_mrl and same_scalar(k_ref, k_mrl), (asof, tf, name)
            pd.testing.assert_series_equal(h_ref, h_mrl, check_exact=True)
            for k in (0, 1, 3, 5, 7, 20, len(r_ref) - 1, len(r_ref), len(r_ref) + 5):
                assert same_scalar(ref.ret(r_ref, k), S.ret(r_mrl, k)), (asof, tf, name, k)


@pytest.mark.parametrize("asof", ASOF_DATES)
def test_metrics_assess_composite_overall_parity(bundle, ref, asof):
    data = reference_data(bundle, asof, ref)
    win = S.window(bundle, asof, "faithful")
    for tf in TFS:
        for cut in (0, ref.CONFIG["trend_cut"][tf]):
            m_ref = ref.build_metrics(data, tf, cut=cut)
            m_mrl = S.v0_metrics(win, tf, cut=cut, basket="v0")
            assert_metrics_identical(m_ref, m_mrl, f"{asof} {tf} cut={cut}")
            st_ref, st_mrl = ref.assess(m_ref), S.v0_assess(m_mrl)
            assert st_ref == st_mrl and list(st_ref) == list(st_mrl)
            assert list(st_ref) == list(ref.WEIGHTS)
            if tf != "daily":
                for k in ref.DAILY_ONLY:
                    st_ref.pop(k, None)
                    st_mrl.pop(k, None)
            for fg_missing in (False, True):
                a, b = ref.composite(st_ref, fg_missing), S.v0_composite(st_mrl, fg_missing)
                assert same_scalar(a, b), (asof, tf, cut, fg_missing, a, b)
                for trend in (-1, 0, 1):
                    assert ref.overall(a, trend) == S.v0_overall(b, trend)


@pytest.mark.parametrize("asof", ASOF_DATES)
def test_v0_day_matches_reference_flow(bundle, ref, asof):
    data = reference_data(bundle, asof, ref)
    tfs, states, mets, (name, action, tone) = reference_day(ref, data)
    out = S.v0_day(bundle, asof, variant="faithful", basket="v0")
    assert out["asof"] == asof and out["variant"] == "faithful" and out["basket"] == "v0"
    for tf, x in (("daily", "d"), ("weekly", "w"), ("monthly", "m")):
        assert out[f"states_{x}"] == states[tf] and list(out[f"states_{x}"]) == list(states[tf]), (asof, tf)
        assert same_scalar(out[f"score_{x}"], tfs[tf]["score"]), (asof, tf)
        assert out[f"trend_{x}"] == tfs[tf]["trend"], (asof, tf)
        assert out[f"overall_{x}"] == tfs[tf]["overall"], (asof, tf)
        assert_metrics_identical(mets[tf], out[f"metrics_{x}"], f"{asof} {tf} v0_day")
    assert (out["verdict_ko"], out["action_ko"], out["tone"]) == (name, action, tone)
    assert out["tone"] in C.TONES
    assert out["leaders"] == mets["daily"]["leaders"]
    assert set(out["states_d"]) == set(ref.WEIGHTS)
    assert set(out["states_w"]) == set(out["states_m"]) == set(ref.WEIGHTS) - set(ref.DAILY_ONLY)
    assert out["fg_avail"] is True and out["eod_avail"] is True
    assert out["spy_close"] == float(data["spy"]["Close"].iloc[-1])
    assert out["vix_close"] == float(data["vix"].iloc[-1])
    for k in ("n_watch_avail", "leaders", "warnings", "spy_close", "vix_close"):
        assert k in out


def test_combo_all_125_cells(ref):
    for mo, wk, dy in itertools.product(STATES, STATES, STATES):
        for lang in ("ko", "en"):
            assert S.v0_combo(mo, wk, dy, lang) == ref.combo_advice(mo, wk, dy, lang), (mo, wk, dy, lang)
    assert S.v0_combo("GREEN", "GREEN", "GREEN") == ref.combo_advice("GREEN", "GREEN", "GREEN")  # 기본 lang=ko
    tones = {S.v0_combo(*c)[2] for c in itertools.product(STATES, repeat=3)}
    assert tones <= set(C.TONES)


def test_overall_grid(ref):
    band = ref.CONFIG["band"]
    scores = list(np.linspace(-1, 1, 81)) + [band, -band, band - 1e-12, -band + 1e-12, 0.0]
    for sc in scores:
        for trend in (-1, 0, 1):
            assert S.v0_overall(float(sc), trend) == ref.overall(float(sc), trend), (sc, trend)


def test_composite_missing_flags(ref):
    """eod/lead 결측 제외는 reference composite 에서 그 키를 뺀 것과 비트 동일해야 한다 (같은 누적 순서)."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        states = {k: STATES[rng.integers(5)] for k in ref.WEIGHTS}
        for fg_m, eod_m, lead_m in itertools.product((False, True), repeat=3):
            drop = {k for k, f in (("eod", eod_m), ("lead", lead_m)) if f}
            expect = ref.composite({k: v for k, v in states.items() if k not in drop}, fg_m)
            got = S.v0_composite(states, fg_m, eod_missing=eod_m, lead_missing=lead_m)
            assert same_scalar(expect, got)
        weekly = {k: v for k, v in states.items() if k not in ref.DAILY_ONLY}
        assert same_scalar(ref.composite(weekly, True), S.v0_composite(weekly, True, True, True))
    with pytest.raises(ValueError):
        S.v0_composite({"fg": "GREEN", "eod": "RED", "lead": "AMBER"}, True, True, True)
    with pytest.raises(ValueError):
        S.v0_composite({"fang": "PURPLE"}, False)
    with pytest.raises(ValueError):
        S.v0_composite({"nope": "GREEN"}, False)


# ------------------------------------------------------------------
# 대체 규약 · 점 원칙 · 변형
# ------------------------------------------------------------------
def test_substitution_rules_before_fg_and_eod_history(bundle, ref):
    """2019-06-14: F&G(2020-08~)·EOD(2023-10~) 모두 없음 → 상태는 8개 그대로, 가중치만 제외."""
    asof = "2019-06-14"
    win = S.window(bundle, asof)
    assert win["fg"] is None and not win["meta"]["fg_avail"] and not win["meta"]["eod_avail"]
    assert len(win["eod"]) == 0
    out = S.v0_day(bundle, asof)
    assert out["fg_avail"] is False and out["eod_avail"] is False
    assert list(out["states_d"]) == list(ref.WEIGHTS)
    assert out["states_d"]["fg"] == "AMBER"
    m = out["metrics_d"]
    assert m["fg"] is None and m["fgD"] == 0 and m["eod"] == 0 and m["eod20"] == 0
    kept = {k: v for k, v in out["states_d"].items() if k not in ("fg", "eod")}
    assert same_scalar(out["score_d"], ref.composite(kept, False))
    assert any("F&G" in w for w in out["warnings"]) and any("EOD" in w for w in out["warnings"])
    assert 20 <= out["n_watch_avail"] < 28


def test_half_day_asof_excludes_eod(bundle):
    """2024-07-03 반일장: EOD 세션 제외 → eod_avail False (F&G 는 있음)."""
    out = S.v0_day(bundle, "2024-07-03")
    assert out["eod_avail"] is False and out["fg_avail"] is True
    assert pd.Timestamp("2024-07-03") not in S.window(bundle, "2024-07-03")["eod"].index


@pytest.mark.parametrize("asof", ["2025-04-08", "2026-08-28"])
def test_point_in_time_future_rows_irrelevant(bundle, asof):
    """asof 이후 데이터를 잘라낸 번들로도 결과가 완전히 같아야 한다 (미래 행 미사용)."""
    cut = pd.Timestamp(asof) + pd.Timedelta(days=10)
    trimmed = Bundle(close=bundle.close.loc[:cut], spy_ohlc=bundle.spy_ohlc.loc[:cut], cboe=bundle.cboe.loc[:cut],
                     fg=bundle.fg.loc[:cut], eod=bundle.eod.loc[:cut], meta=dict(bundle.meta))
    full, part = S.v0_day(bundle, asof), S.v0_day(trimmed, asof)
    assert deep_equal(full, part)
    # 더 공격적으로: asof 당일까지만 (BTC 주말 행도 없음)
    exact = Bundle(close=bundle.close.loc[:asof], spy_ohlc=bundle.spy_ohlc.loc[:asof], cboe=bundle.cboe.loc[:asof],
                   fg=bundle.fg.loc[:asof], eod=bundle.eod.loc[:asof], meta=dict(bundle.meta))
    assert deep_equal(full, S.v0_day(exact, asof))


def test_completed_variant_period_end_equals_faithful(bundle):
    """2024-03-28(목): 3월 마지막 거래일이자 주말(성금요일 휴장) → 완성 봉 기준 now 상태·점수가 faithful 과 같다.
    (prev=cut 시점은 기간말이 아니라 추세·overall 은 다를 수 있다.)"""
    asof = "2024-03-28"
    f, c = S.v0_day(bundle, asof, "faithful"), S.v0_day(bundle, asof, "completed")
    assert c["variant"] == "completed"
    for x in ("d", "w", "m"):
        assert f[f"states_{x}"] == c[f"states_{x}"], x
        assert same_scalar(f[f"score_{x}"], c[f"score_{x}"]), x
        assert_metrics_identical(f[f"metrics_{x}"], c[f"metrics_{x}"], f"completed now {x}")


def test_completed_variant_drops_partial_period(bundle):
    """2025-04-08(화): 부분 주·부분 월 → completed 는 마지막 부분 기간을 뺀다; 일간은 동일."""
    asof = "2025-04-08"
    win_f, win_c = S.window(bundle, asof, "faithful"), S.window(bundle, asof, "completed")
    spy = win_f["spy"]["Close"]
    for tf in ("weekly", "monthly"):
        r_f = S.resample_close(spy, tf, completed_only=False)
        r_c = S.resample_close(spy, tf, completed_only=True)
        assert len(r_c) == len(r_f) - 1 and r_c.index[-1] < r_f.index[-1], tf
        pd.testing.assert_series_equal(r_c, r_f.iloc[:-1], check_exact=True)
    assert_metrics_identical(S.v0_metrics(win_f, "daily"), S.v0_metrics(win_c, "daily"), "daily same")
    out_c = S.v0_day(bundle, asof, "completed")
    assert set(out_c) == set(S.v0_day(bundle, asof, "faithful"))
    assert out_c["tone"] in C.TONES


def test_equal_basket_runs(bundle, ref):
    asof = "2026-08-28"
    win = S.window(bundle, asof)
    m_eq, m_v0 = S.v0_metrics(win, "daily", basket="equal"), S.v0_metrics(win, "daily", basket="v0")
    assert list(m_eq) == list(m_v0) and math.isfinite(m_eq["fangRS"])
    # 바스켓 외 메트릭은 동일
    for k in m_v0:
        if k != "fangRS":
            assert same_scalar(m_v0[k], m_eq[k]) if not isinstance(m_v0[k], list) else m_v0[k] == m_eq[k], k
    out = S.v0_day(bundle, asof, basket="equal")
    assert out["basket"] == "equal" and out["tone"] in C.TONES


def test_error_paths(bundle):
    with pytest.raises(ValueError):
        S.window(bundle, "2026-08-29")           # 토요일
    with pytest.raises(ValueError):
        S.window(bundle, "2026-09-07")           # 노동절 휴장 (BTC 행만 있음)
    with pytest.raises(ValueError):
        S.window(bundle, "2026-08-28", "partial")
    with pytest.raises(ValueError):
        S.v0_day(bundle, "2026-08-28", basket="cap")
    win = S.window(bundle, "2026-08-28")
    with pytest.raises(ValueError):
        S.v0_metrics(win, "hourly")
    with pytest.raises(ValueError):
        S.v0_metrics(win, "daily", cut=-1)
    with pytest.raises(ValueError):
        S.v0_combo("GREEN", "GREEN", "BLUE")
    with pytest.raises(ValueError):
        S.resample_close(win["vix"], "quarterly")
