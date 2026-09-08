# -*- coding: utf-8 -*-
"""evaluate 의 Phase 3 추가분 (ARCHITECTURE_PHASE3.md §6.6) 테스트.

계약:
    allocation_from_weights(w, spy_close, cost_bps=5) -> dict   # allocation_sim 과 같은 규약·키
    window_distribution(series, L, step=21) -> dict             # p5/p25/p50/p75/p95·min·max·share<0

여기서 고정하는 것:
* **구현은 하나뿐**: `sizing.allocation_from_weights is evaluate.allocation_from_weights` (재수출).
  같은 규약이 두 군데 살아 있으면 §6.3 백테스트 표와 장부의 rule_ret_20 이 갈라진다.
* 톤 경로(`tone.map(TONE_EXPOSURE)`)로 만든 비중에서 `allocation_sim` 과 **비트 동일**.
* 점 원칙(w[t] → r[t+1]) · 비용은 변경일에만 · 현금 0% · 적합 파라미터 0.
* 조용한 실패 금지: 결측 비중·뒤섞인 인덱스·음수 비용은 예외, 짝 없는 날짜는 경고.
실캐시를 읽지 않는다(합성 자료 + 손계산).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrl import evaluate, sizing                              # noqa: E402
from mrl.config import P3, TONE_EXPOSURE                      # noqa: E402

BPS = 1e-4


def _close(n=260, seed=7) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=n)
    ret = rng.normal(0.0005, 0.011, n)
    ret[60:64] = -0.03                                        # 급락 구간(낙폭·최악 월이 실제로 생기게)
    return pd.Series(100.0 * np.cumprod(1.0 + ret), index=idx)


def _tone_frame(close: pd.Series, seed=1) -> pd.DataFrame:
    """톤 경로 replay 프레임 — 상태가 실제로 여러 번 바뀌게 만든다(전환 비용이 0 이 아니도록)."""
    rng = np.random.default_rng(seed)
    tones = np.array(["hold", "caution", "reduce"])
    t = tones[rng.integers(0, 3, len(close))]
    return pd.DataFrame({"tone": t}, index=close.index)


# ------------------------------------------------------------------
# 1. 구현이 하나인가 (§6.6 의 핵심 — 규약이 갈라지지 않게)
# ------------------------------------------------------------------
def test_single_implementation_sizing_reexports_evaluate():
    assert sizing.allocation_from_weights is evaluate.allocation_from_weights
    assert sizing.window_distribution is evaluate.window_distribution
    assert evaluate.allocation_from_weights.__module__ == "mrl.evaluate"
    assert evaluate.window_distribution.__module__ == "mrl.evaluate"
    # sizing 의 공개 API 는 그대로여야 한다(§6.6 목록)
    assert "allocation_from_weights" in sizing.__all__ and "window_distribution" in sizing.__all__
    # 계약의 기본 비용 5bp = P3["cost_bps"] (두 이름이 같은 수를 가리킨다)
    assert float(P3["cost_bps"]) == 5.0
    assert evaluate.allocation_from_weights.__defaults__[0] == float(P3["cost_bps"])
    assert evaluate.window_distribution.__defaults__ == (21,)


# ------------------------------------------------------------------
# 2. allocation_sim 과 비트 동일 (톤 경로에서)
# ------------------------------------------------------------------
def test_allocation_from_weights_bitwise_equals_allocation_sim_on_tone_path():
    close = _close()
    replay = _tone_frame(close)
    sim = evaluate.allocation_sim(replay, close, cost_bps=5)
    got = evaluate.allocation_from_weights(replay["tone"].map(TONE_EXPOSURE).astype(float), close, cost_bps=5)
    # exposure_map 만 다르다(비중 경로엔 톤→비중 사전이 없다). 그 밖의 스칼라 키는 전부 비트 동일.
    assert set(sim) == set(got)
    assert sim["exposure_map"] == dict(TONE_EXPOSURE) and got["exposure_map"] is None
    for k, v in sim.items():
        if k in ("series", "exposure_map"):
            continue
        if isinstance(v, float):
            assert got[k] == v or (np.isnan(v) and np.isnan(got[k])), k
        else:
            assert got[k] == v, k
    for k in ("equity", "equity_bh", "daily_ret", "daily_ret_bh", "exposure"):
        pd.testing.assert_series_equal(sim["series"][k], got["series"][k], check_exact=True)


def test_perf_stats_keys_and_deterministic_repeat():
    close = _close()
    w = pd.Series(np.where(np.arange(len(close)) % 40 < 20, 1.0, 0.5), index=close.index)
    a = evaluate.allocation_from_weights(w, close)
    b = evaluate.allocation_from_weights(w, close)
    for k, v in a.items():
        if k == "series":
            continue
        assert (b[k] == v) or (isinstance(v, float) and np.isnan(v) and np.isnan(b[k])), k
    # allocation_sim 과 같은 _perf_stats 를 쓴다: 내부 키(_equity/_monthly)는 새지 않는다
    assert "_equity" not in a and "_monthly" not in a
    assert a["cost_bps"] == 5.0 and a["n_days"] == len(close) - 1


# ------------------------------------------------------------------
# 3. 규약: 점 원칙 · 비용은 변경일에만 · 현금 0%
# ------------------------------------------------------------------
def test_point_in_time_and_cost_only_on_change_days():
    idx = pd.bdate_range("2026-01-05", periods=6)
    close = pd.Series([100.0, 110.0, 121.0, 121.0, 121.0, 133.1], index=idx)   # +10%, +10%, 0, 0, +10%
    w = pd.Series([0.5, 0.5, 1.0, 1.0, 1.0, 1.0], index=idx)                   # 3번째 세션에 0.5 → 1.0
    out = evaluate.allocation_from_weights(w, close, cost_bps=5)
    r = out["series"]["daily_ret"].to_numpy()
    # t=1: w[0]=0.5 가 r[1]=+10% 에 걸린다(= 오늘의 비중이 아니라 어제 종가의 비중)
    assert r[0] == pytest.approx(0.5 * 0.10)
    # t=2: w[1]=0.5 × r[2]=+10%, 그리고 이 날 |Δw| = 0.5 만큼 비용
    assert r[1] == pytest.approx((1 + 0.5 * 0.10) * (1 - 0.5 * 5 * BPS) - 1)
    # t=3,4: 무변경·무수익 → 정확히 0 (비용이 매일 붙지 않는다)
    assert r[2] == 0.0 and r[3] == 0.0
    # t=5: w[4]=1.0 × +10%, 변경 없음
    assert r[4] == pytest.approx(0.10)
    assert out["n_switches"] == 1
    assert out["cost_total"] == pytest.approx(0.5 * 5 * BPS)
    assert out["avg_exposure"] == pytest.approx(np.mean([0.5, 0.5, 1.0, 1.0, 1.0]))


def test_flat_full_weight_equals_buy_and_hold_and_cash_earns_zero():
    close = _close()
    full = evaluate.allocation_from_weights(pd.Series(1.0, index=close.index), close)
    assert full["cagr"] == pytest.approx(full["bh_cagr"])
    assert full["max_dd"] == pytest.approx(full["bh_max_dd"])
    assert full["cost_total"] == 0.0 and full["n_switches"] == 0
    # 절반 현금은 수익률 0 을 받는다(무위험수익을 더하지 않는다 — 배분 쪽에 보수적)
    half = evaluate.allocation_from_weights(pd.Series(0.5, index=close.index), close)
    assert np.allclose(half["series"]["daily_ret"].to_numpy(),
                       0.5 * full["series"]["daily_ret"].to_numpy(), rtol=1e-12, atol=1e-15)


def test_maxdd_improvement_sign_and_zero_cost_option():
    close = _close()
    w = pd.Series(np.where(np.arange(len(close)) >= 55, 0.25, 1.0), index=close.index)   # 급락 직전에 줄인다
    out = evaluate.allocation_from_weights(w, close, cost_bps=0)
    assert out["cost_total"] == 0.0
    assert out["max_dd"] > out["bh_max_dd"]                 # 둘 다 음수 — 얕은 낙폭이 더 크다
    assert out["maxdd_improvement"] == pytest.approx(out["max_dd"] - out["bh_max_dd"])
    assert out["maxdd_improvement"] > 0


# ------------------------------------------------------------------
# 4. 조용한 실패 금지
# ------------------------------------------------------------------
def test_input_validation_raises_and_warns():
    close = _close(60)
    w = pd.Series(0.5, index=close.index)
    with pytest.raises(TypeError):
        evaluate.allocation_from_weights(w.to_numpy(), close)                      # Series 가 아님
    with pytest.raises(TypeError):
        evaluate.allocation_from_weights(pd.Series(0.5, index=range(len(close))), close)   # DatetimeIndex 아님
    with pytest.raises(ValueError):
        evaluate.allocation_from_weights(w.iloc[::-1], close)                      # 내림차순
    dup = pd.concat([w, w.iloc[[0]]]).sort_index()
    with pytest.raises(ValueError):
        evaluate.allocation_from_weights(dup, close)                               # 중복 날짜
    with pytest.raises(ValueError):
        evaluate.allocation_from_weights(w, close, cost_bps=-1)
    with pytest.raises(ValueError, match="결측"):
        evaluate.allocation_from_weights(w.mask(w.index == w.index[3]), close)      # 결측 비중은 거부(0 으로 채우지 않는다)
    with pytest.raises(ValueError, match="2 거래일"):
        evaluate.allocation_from_weights(w.iloc[:1], close)
    # spy_close 에 없는 날짜는 조용히 버리지 않고 경고한다
    extra = pd.concat([w, pd.Series([0.5], index=[w.index[-1] + pd.Timedelta(days=30)])])
    with pytest.warns(UserWarning, match="spy_close 에 없어"):
        evaluate.allocation_from_weights(extra, close)


# ------------------------------------------------------------------
# 5. window_distribution — 참조 분포(손계산)
# ------------------------------------------------------------------
def test_window_distribution_hand_calculation_and_step():
    r = pd.Series([0.10, -0.10, 0.10, -0.10, 0.10, -0.10],
                  index=pd.bdate_range("2026-01-05", periods=6))
    wd = evaluate.window_distribution(r, 2, 1)                 # 창 5개: 시작 0..4
    assert wd["L"] == 2 and wd["step"] == 1 and wd["n"] == 5
    up_down = 1.10 * 0.90 - 1.0                                # -0.01
    down_up = 0.90 * 1.10 - 1.0                                # -0.01 (같다)
    assert wd["min"] == pytest.approx(min(up_down, down_up))
    assert wd["max"] == pytest.approx(max(up_down, down_up))
    assert wd["p50"] == pytest.approx(-0.01)
    assert wd["share_negative"] == 1.0
    # step 2 → 창 3개(시작 0, 2, 4), step 이 커지면 창 수만 줄고 정의는 같다
    assert evaluate.window_distribution(r, 2, 2)["n"] == 3
    # 분위는 np.percentile(선형 보간)과 같은 정의
    r2 = pd.Series(np.arange(1, 21) / 1000.0, index=pd.bdate_range("2026-01-05", periods=20))
    got = evaluate.window_distribution(r2, 5, 1)
    vals = [np.prod(1.0 + r2.to_numpy()[a:a + 5]) - 1.0 for a in range(16)]
    for key, q in (("p5", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95)):
        assert got[key] == pytest.approx(float(np.percentile(vals, q)))
    assert got["share_negative"] == 0.0


def test_window_distribution_edges():
    idx = pd.bdate_range("2026-01-05", periods=5)
    r = pd.Series([0.01, np.nan, 0.02, -0.01, 0.03], index=idx)
    # NaN 은 계산 전에 떨어진다 — 남은 4개로 창을 센다(조용히 0 으로 채우지 않는다)
    assert evaluate.window_distribution(r, 4, 1)["n"] == 1
    # 창이 하나도 안 나오면 예외가 아니라 빈 분포(n=0, 분위 NaN)
    empty = evaluate.window_distribution(r, 40, 1)
    assert empty["n"] == 0 and np.isnan(empty["p50"]) and np.isnan(empty["share_negative"])
    with pytest.raises(ValueError):
        evaluate.window_distribution(r, 1, 1)
    with pytest.raises(ValueError):
        evaluate.window_distribution(r, 3, 0)


def test_window_distribution_is_not_track_window_distribution():
    """이름이 같은 track.window_distribution 은 **다른 계약**(OOS 프레임 → 롤링 BSS)이다 — 섞지 않는다."""
    from mrl import track
    assert track.window_distribution is not evaluate.window_distribution
    with pytest.raises((TypeError, ValueError, KeyError, AttributeError)):
        track.window_distribution(pd.Series([0.01, 0.02, 0.03]), 2, 1)


# ------------------------------------------------------------------
# 6. 적합 파라미터 0 (§14 파라미터 회계)
# ------------------------------------------------------------------
def test_no_fitted_parameters_added():
    assert sizing.PARAM_COUNT == 0 and sizing.n_params() == 0
    src = (ROOT / "mrl" / "evaluate.py").read_text(encoding="utf-8")
    start = src.index("def allocation_from_weights")
    end = src.index("# 4. Brier")
    body = src[start:end]
    for forbidden in ("LogisticRegression", ".fit(", "np.polyfit", "curve_fit", "lstsq"):
        assert forbidden not in body, forbidden


def test_no_warnings_on_clean_inputs():
    close = _close(80)
    w = pd.Series(0.6, index=close.index)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        evaluate.allocation_from_weights(w, close)
        evaluate.window_distribution(close.pct_change(), 20, 21)
    assert [str(c.message) for c in caught] == []
