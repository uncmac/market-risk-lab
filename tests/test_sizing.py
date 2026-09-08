# -*- coding: utf-8 -*-
"""mrl/sizing.py 테스트 — 예산→목표 변동성·EWMA·격자/밴드/주간 리듬·재개·점 원칙·백테스트 (ARCHITECTURE_PHASE3.md §6·§13).

실행: 프로젝트 루트에서  python -m pytest tests/test_sizing.py -q

실캐시 테스트는 홀드아웃(HOLDOUT_START=2024-09-01)을 **입력 프레임 단계에서** 2024-08-30 으로 하드컷한다.
"""
from __future__ import annotations

import hashlib
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

from mrl import calendar_us as CAL  # noqa: E402
from mrl import decision as D  # noqa: E402
from mrl import evaluate as E  # noqa: E402
from mrl import model as M  # noqa: E402
from mrl import sizing as S  # noqa: E402
from mrl.config import HOLDOUT_START, P3, STATE_TO_TONE, TONE_EXPOSURE  # noqa: E402

HARD_CUT = "2024-08-30"                     # 홀드아웃 직전 세션 (HOLDOUT_START = 2024-09-01)
CLOSE_CSV = ROOT / "data" / "close.csv"
BT_V1_CSV = ROOT / "results" / "backtest_v1.csv"
WF_CSV = ROOT / "results" / "calib_p2_walkforward.csv"


# ------------------------------------------------------------------
# 헬퍼
# ------------------------------------------------------------------
def _bdays(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _close_from_returns(r, start: str = "2020-01-01", p0: float = 100.0) -> pd.Series:
    r = np.asarray(r, dtype=float)
    px = p0 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    return pd.Series(px, index=_bdays(len(px), start), name="SPY")


def _rand_close(n: int, seed: int = 0, sigma: float = 0.011, mu: float = 0.0003) -> pd.Series:
    rng = np.random.default_rng(seed)
    return _close_from_returns(rng.normal(mu, sigma, n - 1))


def _frame_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(float_format="%.12g").encode("utf-8")).hexdigest()


def _real_close() -> pd.Series:
    if not CLOSE_CSV.exists():
        pytest.skip("data/close.csv 없음 — 실캐시 테스트 생략")
    s = pd.read_csv(CLOSE_CSV, index_col=0, parse_dates=True)["SPY"].dropna()
    s = s.loc[s.index <= pd.Timestamp(HARD_CUT)]                 # 하드컷: 홀드아웃 입력 자체를 자른다
    assert s.index.max() < pd.Timestamp(HOLDOUT_START)
    return s


def _real_states() -> pd.Series:
    if not BT_V1_CSV.exists():
        pytest.skip("results/backtest_v1.csv 없음 — 실캐시 테스트 생략")
    st = pd.read_csv(BT_V1_CSV, index_col=0, parse_dates=True)["state"]
    st = st.loc[st.index <= pd.Timestamp(HARD_CUT)]
    assert st.index.max() < pd.Timestamp(HOLDOUT_START)
    return st


# ------------------------------------------------------------------
# 1. 적합 파라미터 0 · 사양 해시
# ------------------------------------------------------------------
def test_zero_fitted_parameters():
    """§6-b: 비중 규칙의 적합 파라미터는 0개이고 생산 확률(4개)에 아무것도 더하지 않는다."""
    assert S.PARAM_COUNT == 0
    assert S.n_params() == 0
    assert M.PARAM_COUNT == 4                                    # Phase 3 는 여기에 0 을 더한다
    src = (ROOT / "mrl" / "sizing.py").read_text(encoding="utf-8")
    banned = ("sklearn", "LogisticRegression", "polyfit", "lstsq", "curve_fit", "minimize(",
              "scipy.optimize", ".fit(")
    hit = [b for b in banned if b in src]
    assert not hit, f"sizing.py 에 적합 흔적: {hit}"
    assert "np.random" not in src and "default_rng" not in src, "비중 규칙에 난수 없음"


def test_all_constants_come_from_config():
    """새 숫자를 만들지 않는다 — 규칙 상수는 전부 P3·TONE_EXPOSURE 에서 온다."""
    c = S.sizing_constants()
    for k in ("ewma_lambda", "ewma_seed_sessions", "k_slow", "k_fast", "d_max_default",
              "w_min", "w_max", "grid", "band", "cost_bps", "cash_return", "churn_ceiling"):
        assert c[k] == P3[k], k
    assert c["vol_grid"] == tuple(P3["vol_grid"])
    assert c["multiplier_map"] == {"normal": 1.0, "caution": 0.5, "reduce": 0.25}
    assert set(c["multiplier_map"]) == set(STATE_TO_TONE)
    assert S.SIZING_RULE.startswith("p3|sigma=ewma(lambda=0.94,seed=60)")


def test_sizing_sha256_stable_and_rule_bound():
    a, b = S.sizing_sha256(), S.sizing_sha256()
    assert a == b and len(a) == 64 and int(a, 16) >= 0


# ------------------------------------------------------------------
# 2. EWMA 변동성
# ------------------------------------------------------------------
def test_ewma_hand_calculation_five_returns():
    """합성 수익률 5개 손계산 (seed 2)."""
    r = [0.01, -0.02, 0.015, -0.005, 0.03]
    close = _close_from_returns(r)
    out = S.ewma_vol(close, lam=0.94, seed_sessions=2)
    # 손계산: v[1] = 표본분산(r0, r1); 그 뒤 v[t] = .94 v[t-1] + .06 r_t²
    m = (r[0] + r[1]) / 2.0
    v = [math.nan, ((r[0] - m) ** 2 + (r[1] - m) ** 2) / 1.0]
    for t in (2, 3, 4):
        v.append(0.94 * v[t - 1] + 0.06 * r[t] ** 2)
    exp = [math.nan] + [math.nan if math.isnan(x) else math.sqrt(252.0 * x) for x in v]
    got = out.to_numpy()
    assert math.isnan(got[0]) and math.isnan(got[1])
    for i in (2, 3, 4, 5):
        assert got[i] == pytest.approx(exp[i], rel=1e-12), i


def test_ewma_seed_60_first_valid_and_annualized():
    close = _rand_close(400, seed=0)
    out = S.ewma_vol(close)                                      # 기본 seed 60
    assert out.attrs["seed_sessions"] == 60 == P3["ewma_seed_sessions"]
    assert out.iloc[:60].isna().all()                            # 첫 60세션은 NaN — 이월 없음
    assert np.isfinite(out.iloc[60])
    r = np.log(close / close.shift(1)).dropna()
    assert out.iloc[60] == pytest.approx(math.sqrt(252.0 * r.iloc[:60].var(ddof=1)), rel=1e-12)
    assert 0.05 < out.dropna().median() < 0.60                   # 연율화 스케일


def test_ewma_point_in_time_and_asof():
    """인과적 재귀 — 미래 행을 덧붙여도 과거 σ̂ 가 변하지 않는다."""
    close = _rand_close(300, seed=1)
    full = S.ewma_vol(close)
    for t in (100, 180, 250):
        cut = S.ewma_vol(close.iloc[:t])
        pd.testing.assert_series_equal(cut, full.iloc[:t], check_names=False)
        asof = S.ewma_vol(close, asof=close.index[t - 1])
        pd.testing.assert_series_equal(asof, full.iloc[:t], check_names=False)


def test_ewma_rejects_bad_inputs():
    close = _rand_close(100, seed=2)
    with pytest.raises(ValueError):
        S.ewma_vol(close, lam=1.0)
    with pytest.raises(ValueError):
        S.ewma_vol(close, seed_sessions=1)
    bad = close.copy()
    bad.iloc[5] = -1.0
    with pytest.raises(ValueError):
        S.ewma_vol(bad)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = S.ewma_vol(close.iloc[:30])                        # 수익률 29개 < seed 60
    assert out.isna().all() and any("seed" in str(x.message) for x in w)


# ------------------------------------------------------------------
# 3. 예산 → 목표 변동성 · 사다리
# ------------------------------------------------------------------
def test_target_vol_floor_to_grid():
    assert S.target_vol(0.35) == 0.10                             # 기본 예산 → 10% (0.35/3.5 부동소수 주의)
    assert S.target_vol(0.28) == 0.08
    assert S.target_vol(0.30) == 0.08                             # 0.0857 → 내림
    assert S.target_vol(0.525) == 0.15
    assert S.target_vol(0.21) == 0.06
    with pytest.raises(ValueError):
        S.target_vol(0.20)                                        # 0.0571 < 격자 최소 6%
    with pytest.raises(ValueError):
        S.target_vol(1.5)
    with pytest.raises(ValueError):
        S.target_vol(0.35, k=0.0)


def test_budget_ladder_two_budget_lines():
    lad = S.budget_ladder()
    assert list(lad["sigma_target"]) == [0.06, 0.08, 0.10, 0.12, 0.15]
    assert [round(x * 100, 1) for x in lad["d_slow"]] == [21.0, 28.0, 35.0, 42.0, 52.5]
    assert [round(x * 100, 1) for x in lad["d_fast"]] == [12.0, 16.0, 20.0, 24.0, 30.0]
    assert lad.attrs["sigma_target_default"] == 0.10
    assert lad[list(S.LADDER_COLUMNS[3:])].isna().all().all()      # 결과 열은 채우기 전 NaN
    filled = S.budget_ladder(results={0.10: {"cagr": 0.0722, "maxdd_decision": -0.235}})
    assert filled.loc[filled["sigma_target"] == 0.10, "cagr"].iloc[0] == 0.0722
    with pytest.raises(ValueError):
        S.budget_ladder(results={0.10: {"nope": 1}})
    with pytest.raises(ValueError):
        S.budget_ladder(k_fast=4.0)                               # 낙관선이 헤드라인보다 클 수 없다


def test_budget_sentence_and_honest_reading():
    s = S.budget_sentence(0.35, 0.10)
    assert "−35%" in s and "−20%" in s and "10%" in s and "예산은 보장이 아니다" in s
    assert "설계 단계 근사" in s
    assert "설계 단계 근사" not in S.budget_sentence(0.35, 0.10, -0.250, -0.307)
    lines = S.honest_reading()
    assert len(lines) == 4 and all(isinstance(x, str) and x for x in lines)
    assert "VT14" in lines[0] and "k_slow 3.5" in lines[2]
    off = S.honest_reading({"adopted_dd": "−25.0%"})
    assert "−25.0%" in off[1] and "설계 단계 근사" not in off[1]
    with pytest.raises(ValueError):
        S.honest_reading({"nope": "x"})


# ------------------------------------------------------------------
# 4. 상태 배수 · 클립 · 곱 · 재클립
# ------------------------------------------------------------------
def test_state_multiplier_uses_registered_map_only():
    st = pd.Series(["normal", "caution", "reduce", None], index=_bdays(4))
    m = S.state_multiplier(st)
    assert list(m.iloc[:3]) == [1.0, 0.5, 0.25]
    assert math.isnan(m.iloc[3])
    assert S.state_multiplier("reduce") == TONE_EXPOSURE[STATE_TO_TONE["reduce"]] == 0.25
    assert math.isnan(S.state_multiplier(None))
    with pytest.raises(ValueError):
        S.state_multiplier(pd.Series(["panic"], index=_bdays(1)))
    with pytest.raises(ValueError):
        S.state_multiplier("panic")


def test_exposure_target_clip_product_reclip():
    idx = _bdays(5)
    sigma = pd.Series([0.05, 0.10, 0.20, 0.60, np.nan], index=idx)     # σ_T/σ̂ = 2.0, 1.0, 0.5, 0.167, NaN
    mult = pd.Series([1.0, 0.5, 0.5, 1.0, 1.0], index=idx)
    et = S.exposure_target(sigma, mult, 0.10)
    assert list(et["w_vol"].iloc[:4]) == [1.0, 1.0, 0.5, 0.25]         # 상한 1.0 · 바닥 0.25
    assert list(et["w_target"].iloc[:4]) == [1.0, 0.5, 0.25, 0.25]     # 곱 뒤 재클립 → 0.25 아래로 안 감
    assert math.isnan(et["w_vol"].iloc[4]) and math.isnan(et["w_target"].iloc[4])
    nf = S.exposure_target(sigma, mult, 0.10, floor_after=False)
    assert nf["w_target"].iloc[2] == 0.25 and nf["w_target"].iloc[1] == 0.5
    assert nf["w_target"].iloc[3] == pytest.approx(0.25)               # 1.0 배수라 같음
    # 바닥 없는 곱이 실제로 0.25 아래로 내려간다 (등록된 배분표에 없는 비중)
    nf2 = S.exposure_target(pd.Series([1.0], index=_bdays(1)), pd.Series([0.25], index=_bdays(1)),
                            0.10, floor_after=False)
    assert nf2["w_target"].iloc[0] == pytest.approx(0.0625)
    assert S.exposure_target(pd.Series([1.0], index=_bdays(1)), pd.Series([0.25], index=_bdays(1)),
                             0.10)["w_target"].iloc[0] == 0.25
    one = S.exposure_target(0.20, 0.5, 0.10)                           # 스칼라 (daily.py 경로)
    assert len(one) == 1 and one["w_target"].iloc[0] == 0.25
    with pytest.raises(ValueError):
        S.exposure_target(sigma, mult, 0.10, w_min=0.6, w_max=0.3)


def test_reclip_range_is_always_in_bounds():
    rng = np.random.default_rng(0)
    idx = _bdays(500)
    sigma = pd.Series(np.exp(rng.normal(np.log(0.15), 0.6, 500)), index=idx)
    mult = pd.Series(rng.choice([1.0, 0.5, 0.25], 500), index=idx)
    w = S.exposure_target(sigma, mult, 0.10)["w_target"]
    assert w.min() >= P3["w_min"] - 1e-15 and w.max() <= P3["w_max"] + 1e-15


# ------------------------------------------------------------------
# 5. 실행 의미론 — 격자 · 밴드 · 주간 · 즉시 격상
# ------------------------------------------------------------------
def _weekday_index(n: int, start: str = "2021-01-04") -> pd.DatetimeIndex:
    """휴일이 없는 구간의 거래일 (월~금)."""
    return CAL.trading_days(start, pd.Timestamp(start) + pd.Timedelta(days=int(n * 2 + 20)))[:n]


def test_step_reason_priority_and_grid():
    cfg = P3
    assert S.step(0.63, 1.0, np.nan, None, False, cfg) == (0.65, "init")          # 격자 반올림
    assert S.step(0.63, 1.0, 1.0, 0.65, False, cfg) == (0.65, "hold")
    assert S.step(0.40, 1.0, 1.0, 0.65, False, cfg) == (0.65, "hold")             # 주말 아님 → 유지
    assert S.step(0.40, 1.0, 1.0, 0.65, True, cfg) == (0.40, "weekly")            # |Δ| = .25 ≥ .10
    assert S.step(0.60, 1.0, 1.0, 0.65, True, cfg) == (0.65, "hold")              # |Δ| = .05 < .10 → 무거래
    assert S.step(0.90, 0.5, 1.0, 0.65, False, cfg) == (0.65, "escalation")       # 격상: 하향만 (min)
    assert S.step(0.30, 0.5, 1.0, 0.65, False, cfg) == (0.30, "escalation")       # 격상: 즉시 하향
    assert S.step(np.nan, 1.0, 1.0, 0.65, True, cfg) == (0.65, "input_missing")   # σ̂ 결측 → 유지
    assert S.step(0.9, np.nan, 1.0, 0.65, True, cfg) == (0.65, "input_missing")   # 상태 결측 → 유지
    w, why = S.step(np.nan, np.nan, np.nan, None, True, cfg)
    assert math.isnan(w) and why == "input_missing"
    assert S.step(0.9, 1.0, 1.0, 0.65, True, cfg, deploy=False) == (0.65, "info_only")


def test_input_missing_never_returns_to_one():
    """조용한 실패 금지: 결측이 이어져도 비중은 1.0 으로 복귀하지 않는다."""
    idx = _weekday_index(15)
    wt = pd.Series([0.5] * 3 + [np.nan] * 9 + [0.5] * 3, index=idx)
    mult = pd.Series(1.0, index=idx)
    ex = S.execute(wt, mult)
    assert (ex["w_exec"] == 0.5).all()
    assert list(ex["reason"])[3:12] == ["input_missing"] * 9
    assert not (ex["w_exec"] == 1.0).any()


def test_band_no_trade_oscillation():
    """밴드 안 진동(|Δ| < 0.10)은 주말이어도 무변경."""
    idx = _weekday_index(30)
    rng = np.random.default_rng(0)
    wt = pd.Series(0.60 + rng.uniform(-0.045, 0.045, 30), index=idx)
    ex = S.execute(wt, pd.Series(1.0, index=idx))
    assert ex["w_exec"].nunique() == 1                      # init 한 번 뒤 변화 없음
    assert set(ex["reason"].iloc[1:]) == {"hold"}


def test_band_boundary_is_literal():
    """밴드 비교는 §6.1.5 문자 그대로 IEEE-754 로 한다 (설계 단계 백테스트와 동일한 경계)."""
    assert (1.0 - 0.9) < 0.10                                # 부동소수 사실
    assert S.step(1.0, 1.0, 1.0, 0.90, True, P3) == (0.90, "hold")
    assert S.step(1.0, 1.0, 1.0, 0.85, True, P3) == (1.00, "weekly")
    assert S.step(0.30, 1.0, 1.0, 0.40, True, P3) == (0.30, "weekly")     # 0.40-0.30 = .1000000000000000055 ≥ .1


def test_escalation_immediate_downgrade_de_escalation_waits():
    """화요일 격상 → 즉시 하향(min 규칙); 격하는 주말까지 대기."""
    idx = _weekday_index(12, "2021-01-04")                  # 월~금 2주 + 이틀
    week_end = CAL.period_end_from_index(idx, "W")
    tue = idx[1]
    assert not week_end.loc[tue]
    mult = pd.Series(1.0, index=idx)
    mult.loc[tue:] = 0.5                                    # 화요일 caution 격상
    mult.loc[idx[6]:] = 1.0                                 # 다음 주 월요일 격하
    wt = (pd.Series(0.90, index=idx) * mult).clip(0.25, 1.0)
    ex = S.execute(wt, mult)
    assert ex["reason"].iloc[1] == "escalation"
    assert ex["w_exec"].iloc[1] == 0.45                     # 즉시 하향
    assert ex["reason"].iloc[6] == "hold"                   # 격하는 즉시가 아니다
    assert ex["w_exec"].iloc[6] == 0.45
    fri = [i for i, f in enumerate(week_end) if f and i > 6][0]
    assert ex["reason"].iloc[fri] == "weekly" and ex["w_exec"].iloc[fri] == 0.90
    # 격상이 상향을 만들지 않는다: cand 가 더 커도 min 이 걸린다
    assert S.step(0.95, 0.5, 1.0, 0.30, False, P3) == (0.30, "escalation")


def test_structural_ceiling_three_changes_per_five_sessions():
    """어떤 5세션 창에서도 변경 ≤ 3회 (주 1회 + 격상; 결정층 dwell 5 와 같은 경계)."""
    n = 800
    idx = CAL.trading_days("2015-01-02", "2018-06-01")[:n]
    rng = np.random.default_rng(0)
    # 적대적 확률 경로: 결정층을 최대한 자주 흔든다
    clim = 0.16
    p = pd.Series(clim * (1.6 + 1.4 * np.sin(np.arange(n) / 3.0) + rng.normal(0, 0.35, n)), index=idx).clip(0.001, 0.95)
    states = D.run(p, clim)["state"]
    mult = S.state_multiplier(states)
    sigma = pd.Series(np.exp(np.log(0.12) + 0.9 * np.sin(np.arange(n) / 5.0) + rng.normal(0, 0.25, n)), index=idx)
    w_target = S.exposure_target(sigma, mult, 0.10)["w_target"]
    ex = S.execute(w_target, mult)
    changed = (ex["w_exec"].diff().abs() > 1e-12).to_numpy()
    worst = max(int(changed[i:i + 5].sum()) for i in range(len(changed) - 4))
    assert worst <= 3, f"5세션 창 최대 변경 {worst}회"
    assert changed.sum() > 20                                        # 적대적 경로가 실제로 흔들렸다


def test_cadence_flags_and_custom_check():
    idx = _weekday_index(25)
    wt = pd.Series(np.linspace(0.30, 1.00, 25), index=idx)
    mult = pd.Series(1.0, index=idx)
    weekly = S.execute(wt, mult)
    daily = S.execute(wt, mult, cadence="daily")
    monthly = S.execute(wt, mult, cadence="monthly")
    n = lambda ex: int((ex["w_exec"].diff().abs() > 1e-12).sum())
    assert n(daily) >= n(weekly) >= n(monthly)
    assert weekly["week_end"].sum() == int(CAL.period_end_from_index(idx, "W").sum())
    custom = S.execute(wt, mult, check=pd.Series(True, index=idx))
    assert n(custom) == n(daily)
    with pytest.raises(ValueError):
        S.execute(wt, mult, cadence="quarterly")


def test_next_thresholds_up_respects_multiplier_ceiling():
    """§10 카드 2줄: 상태 배수 상한 때문에 도달 불가능한 σ_up 을 광고하지 않는다(조용한 0 금지)."""
    sT = 0.10
    for w, m in ((0.25, 0.25), (0.45, 0.50), (0.50, 0.50), (0.95, 1.00)):
        t = S.next_thresholds(w, sT, m, "2024-08-30")
        assert t["sigma_up"] is None and t["up_note"]
    for w, m in ((0.40, 0.50), (0.30, 0.50), (0.90, 1.00), (0.65, 1.00), (0.25, 1.00)):
        t = S.next_thresholds(w, sT, m, "2024-08-30")
        assert t["up_note"] is None and t["sigma_up"] is not None
        wt = S.exposure_target(t["sigma_up"], m, sT).iloc[0]["w_target"]
        assert abs(float(wt) - (w + P3["band"])) < 1e-9      # 광고한 σ̂ 를 되먹이면 광고한 비중이 나온다


def test_next_thresholds_down_and_missing_inputs():
    t = S.next_thresholds(P3["w_min"], 0.10, 1.0, "2024-08-30")
    assert t["sigma_down"] is None and "바닥" in t["down_note"]
    t2 = S.next_thresholds(0.70, 0.10, 1.0, "2024-08-30")
    assert t2["sigma_down"] == pytest.approx(0.10 / 0.60)
    wt = S.exposure_target(t2["sigma_down"], 1.0, 0.10).iloc[0]["w_target"]
    assert float(wt) == pytest.approx(0.60)
    t3 = S.next_thresholds(float("nan"), 0.10, 1.0, "2024-08-30")
    assert t3["sigma_down"] is None and t3["sigma_up"] is None and t3["down_note"] == t3["up_note"]


# ------------------------------------------------------------------
# 6. 재개 · 점 원칙 · 결정론
# ------------------------------------------------------------------
def test_resume_from_ledger_state_matches_full_run():
    """장부 상태에서 `step` 으로 이어 계산 == `execute` 로 처음부터 계산."""
    n = 300
    idx = CAL.trading_days("2018-01-02", "2019-06-01")[:n]
    rng = np.random.default_rng(3)
    sigma = pd.Series(np.exp(np.log(0.14) + rng.normal(0, 0.35, n).cumsum() * 0.05), index=idx)
    mult = pd.Series(rng.choice([1.0, 0.5, 0.25], n, p=[0.8, 0.15, 0.05]), index=idx)
    wt = S.exposure_target(sigma, mult, 0.10)["w_target"]
    full = S.execute(wt, mult)
    week_end = CAL.period_end_from_index(idx, "W")
    cut = 137
    head = S.execute(wt.iloc[:cut], mult.iloc[:cut])
    prev_w = float(head["w_exec"].iloc[-1])
    prev_m = float(mult.iloc[cut - 1])
    got = []
    for t in range(cut, n):
        w, why = S.step(wt.iloc[t], mult.iloc[t], prev_m, prev_w, bool(week_end.iloc[t]))
        got.append((w, why))
        prev_w, prev_m = w, float(mult.iloc[t])
    assert [g[0] for g in got] == list(full["w_exec"].iloc[cut:])
    assert [g[1] for g in got] == list(full["reason"].iloc[cut:])
    # execute 의 init_w/init_mult 재개도 같은 경로
    tail = S.execute(wt.iloc[cut:], mult.iloc[cut:], init_w=prev_w if False else float(head["w_exec"].iloc[-1]),
                     init_mult=float(mult.iloc[cut - 1]))
    assert list(tail["w_exec"]) == list(full["w_exec"].iloc[cut:])


def test_point_in_time_future_rows_do_not_change_past():
    n = 260
    idx = CAL.trading_days("2019-01-02", "2020-03-01")[:n]
    rng = np.random.default_rng(4)
    sigma = pd.Series(np.exp(np.log(0.13) + 0.4 * np.sin(np.arange(n) / 7.0) + rng.normal(0, 0.2, n)), index=idx)
    states = pd.Series(rng.choice(["normal", "caution", "reduce"], n, p=[0.8, 0.15, 0.05]), index=idx)
    full = S.run(sigma, states, 0.10)
    for t in (60, 150, 230):
        part = S.run(sigma.iloc[:t], states.iloc[:t], 0.10)
        pd.testing.assert_frame_equal(part[["w_vol", "mult", "w_target", "cand", "w_exec", "reason"]],
                                      full[["w_vol", "mult", "w_target", "cand", "w_exec", "reason"]].iloc[:t])


def test_run_is_deterministic_and_info_only_mode():
    n = 200
    idx = CAL.trading_days("2019-01-02", "2019-12-31")[:n]
    rng = np.random.default_rng(5)
    sigma = pd.Series(np.exp(np.log(0.12) + rng.normal(0, 0.25, n)), index=idx)
    states = pd.Series("normal", index=idx)
    a, b = S.run(sigma, states, 0.10), S.run(sigma, states, 0.10)
    assert _frame_hash(a) == _frame_hash(b)
    assert a.attrs["param_count"] == 0 and a.attrs["sizing_rule"] == S.SIZING_RULE
    info = S.run(sigma, states, 0.10, deploy=False)
    assert info["w_exec"].isna().all() and set(info["reason"]) == {"info_only"}
    assert info["mult"].isna().all()
    assert info["w_vol"].notna().sum() > 0                      # 회색 '가정치' 는 남는다
    pd.testing.assert_series_equal(info["w_vol"], a["w_vol"])


# ------------------------------------------------------------------
# 7. 배분 시뮬 — allocation_sim 과 비트 동일 · 비용은 변경일에만
# ------------------------------------------------------------------
def test_allocation_from_weights_matches_allocation_sim_bitwise():
    n = 400
    close = _rand_close(n, seed=6)
    rng = np.random.default_rng(6)
    tone = pd.Series(rng.choice(["hold", "caution", "reduce"], n, p=[0.7, 0.2, 0.1]), index=close.index)
    replay = pd.DataFrame({"tone": tone})
    ref = E.allocation_sim(replay, close, cost_bps=5)
    got = S.allocation_from_weights(tone.map(TONE_EXPOSURE).astype(float), close, cost_bps=5)
    for k in ("cagr", "max_dd", "worst_month", "ann_vol", "total_return", "avg_exposure",
              "cost_total", "excess_cagr", "maxdd_improvement", "bh_cagr", "bh_max_dd", "years"):
        assert got[k] == ref[k], k                                     # 비트 동일
    assert got["n_switches"] == ref["n_switches"] and got["start"] == ref["start"]
    assert str(got["max_dd_date"]) == str(ref["max_dd_date"])
    pd.testing.assert_series_equal(got["series"]["daily_ret"], ref["series"]["daily_ret"])


def test_costs_charged_only_on_change_days():
    close = _rand_close(60, seed=7)
    w = pd.Series(1.0, index=close.index)
    w.iloc[30:] = 0.60
    out = S.allocation_from_weights(w, close, cost_bps=5)
    assert out["n_switches"] == 1
    assert out["cost_total"] == pytest.approx(0.40 * 5e-4, rel=1e-12)
    flat = S.allocation_from_weights(pd.Series(1.0, index=close.index), close)
    assert flat["cost_total"] == 0.0 and flat["cagr"] == pytest.approx(flat["bh_cagr"], rel=1e-12)
    with pytest.raises(ValueError):
        S.allocation_from_weights(w.mask(w.index == w.index[3]), close)   # 결측 비중은 거부


def test_window_distribution_and_rolling_relative():
    r = pd.Series(np.full(100, 0.001), index=_bdays(100))
    wd = S.window_distribution(r, 20, 20)
    assert wd["n"] == 5 and wd["p50"] == pytest.approx(1.001 ** 20 - 1, rel=1e-12)
    assert wd["share_negative"] == 0.0
    rr = S.rolling_relative(r, r * 0.5, 20)
    assert rr["n"] == 5 and rr["share_behind"] == 0.0 and rr["p50"] > 0
    behind = S.rolling_relative(r * 0.5, r, 20)
    assert behind["share_behind"] == 1.0
    assert S.rolling_relative(r, r, 500)["n"] == 0
    cy = S.calendar_year_relative(r, r * 2)
    assert cy["share_behind"] == 1.0 and cy["n_years"] >= 1


# ------------------------------------------------------------------
# 8. 유지 조건 (a)~(f)
# ------------------------------------------------------------------
def _table(**over) -> pd.DataFrame:
    base = {"cagr": 0.072, "max_dd": -0.235, "max_dd_date": "2009-03-09", "worst_month": -0.078,
            "worst_month_label": "2018-10", "ann_vol": 0.093, "switches_per_year": 10.9,
            "avg_exposure": 0.69, "cost_total": 0.022, "d_cagr_vs_bh": -0.036,
            "d_maxdd_vs_bh": 0.317, "vol_ratio": 0.93, "maxdd_over_sigma_t": 2.35,
            "start": pd.NaT, "end": pd.NaT, "years": 21.7, "n_switches": 236, "cost_bps": 5.0}
    rows = [
        dict(base, row=S.ROW_ADOPTED, window=S.WINDOW_EVAL),
        dict(base, row=S.ROW_P2_DECISION, window=S.WINDOW_EVAL, max_dd=-0.310, worst_month=-0.086),
        dict(base, row=S.ROW_VOL_ONLY, window=S.WINDOW_VOL_ONLY, max_dd=-0.307),
    ]
    df = pd.DataFrame(rows, columns=list(S.BACKTEST_COLUMNS))
    for key, val in over.items():
        row, col = key.split("__")
        name = {"adopted": S.ROW_ADOPTED, "p2": S.ROW_P2_DECISION, "vol": S.ROW_VOL_ONLY}[row]
        df.loc[df["row"] == name, col] = val
    return df


def test_retention_all_pass():
    r = S.retention(_table(), 0.10)
    assert r["deploy_sizing"] is True and not r["failed"] and not r["undecided"]
    assert set(r["conditions"]) == set("abcdef")
    assert all(v["pass"] for v in r["conditions"].values())


@pytest.mark.parametrize("over,failed", [
    ({"adopted__max_dd": -0.260}, "a"),                        # ≥ -2.5·σ_T = -25% 위반
    ({"vol__max_dd": -0.360}, "b"),                            # ≥ -k_slow·σ_T = -35% 위반
    ({"adopted__max_dd": -0.270, "p2__max_dd": -0.300}, "c"),  # 개선 3pp < 5pp (a 도 같이 깨짐)
    ({"adopted__switches_per_year": 30.0}, "d"),
    ({"adopted__cost_total": 0.06}, "d"),
    ({"adopted__worst_month": -0.115}, "e"),                   # 결정층 -8.6% 보다 2.9pp 나쁨
    ({"adopted__ann_vol": 0.160}, "f"),                        # 비율 1.6 > 1.15
])
def test_retention_each_condition_can_fail(over, failed):
    r = S.retention(_table(**over), 0.10)
    assert failed in r["failed"], (over, r["failed"])
    assert r["deploy_sizing"] is False


def test_retention_missing_row_is_not_silent_pass():
    t = _table()
    t = t[t["row"] != S.ROW_VOL_ONLY]
    r = S.retention(t, 0.10)
    assert "b" in r["undecided"] and r["deploy_sizing"] is False
    assert r["missing"] and "b" not in r["failed"]


def test_retention_narrow_pass_is_flagged_not_hidden():
    """§6.4 는 근소한 (a)·(b) 가 공식 수치에서 뒤집힐 수 있다고 미리 적었다 — boolean 뒤에 숨기지 않는다."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        r = S.retention(_table(adopted__max_dd=-0.2499), 0.10)          # 한계 -25.00% 를 0.01pp 차로 통과
    assert r["deploy_sizing"] is True and r["narrow"] == ["a"]
    assert r["conditions"]["a"]["margin"] == pytest.approx(0.0001, abs=1e-12)
    assert any("근소" in str(x.message) or "뒤집힌다" in str(x.message) for x in rec)
    assert "근소 통과" in r["note_ko"]
    with warnings.catch_warnings(record=True) as rec2:
        warnings.simplefilter("always")
        wide = S.retention(_table(), 0.10)
    assert wide["narrow"] == [] and not [x for x in rec2 if "유지 조건" in str(x.message)]
    assert wide["conditions"]["a"]["margin"] == pytest.approx(0.015, abs=1e-12)
    assert wide["conditions"]["d"]["margin"] is None                    # 단위가 섞인 조건은 여유를 재지 않는다


def test_retention_conditions_text_is_pre_registered():
    assert set(S.RETENTION_CONDITIONS) == set("abcdef")
    assert "−2.5·σ_T" in S.RETENTION_CONDITIONS["a"] and "5pp" in S.RETENTION_CONDITIONS["c"]


# ------------------------------------------------------------------
# 9. 백테스트 표 · 에피소드
# ------------------------------------------------------------------
def test_backtest_table_shape_and_buy_hold_row():
    close = _rand_close(600, seed=8)
    idx = close.index
    w = pd.Series(0.7, index=idx)
    tbl = S.backtest_table({S.ROW_ADOPTED: w}, close, {"all": (None, None)}, sigma_target=0.10)
    assert list(tbl.columns) == list(S.BACKTEST_COLUMNS)
    assert list(tbl["row"]) == [S.ROW_BUY_HOLD, S.ROW_ADOPTED]
    bh = tbl.iloc[0]
    assert bh["avg_exposure"] == 1.0 and bh["cost_total"] == 0.0 and bh["switches_per_year"] == 0.0
    assert bh["d_cagr_vs_bh"] == pytest.approx(0.0, abs=1e-15)
    assert tbl.iloc[1]["vol_ratio"] == pytest.approx(tbl.iloc[1]["ann_vol"] / 0.10)
    assert tbl.iloc[1]["maxdd_over_sigma_t"] == pytest.approx(abs(tbl.iloc[1]["max_dd"]) / 0.10)
    per_cost = S.backtest_table({"a": w, "b": w}, close, {"all": (None, None)}, cost_bps={"a": 5, "b": 10})
    assert list(per_cost["cost_bps"]) == [0.0, 5.0, 10.0]
    with pytest.raises(ValueError):
        S.backtest_table({"x": w}, close, {"all": (None, None)}, references={"x": w})


def test_episode_pnl_counts_and_protection():
    from mrl import targets as T
    close = _rand_close(1500, seed=9, sigma=0.014, mu=0.0)
    sigma = S.ewma_vol(close)
    states = pd.Series("normal", index=close.index)
    res = S.run(sigma.dropna(), states.reindex(sigma.dropna().index), 0.10)
    eps = T.episodes(close, 0.05, split=False)
    pn = S.episode_pnl(res["w_exec"], close, eps)
    assert len(pn) >= 1 and set(pn.columns) >= {"ret_rule", "ret_bh", "avg_exposure", "protection_pp"}
    assert (pn["ret_rule"] >= pn["ret_bh"] - 1e-9).all()        # 비중 ≤ 1 이므로 규칙이 더 얕다
    assert (pn["avg_exposure"] <= 1.0).all() and (pn["avg_exposure"] >= 0.25 - 1e-9).all()


# ------------------------------------------------------------------
# 10. 실캐시 백테스트 (하드컷 2024-08-30) — 설계 단계 수치 재현
# ------------------------------------------------------------------
@pytest.fixture(scope="module")
def real():
    close = _real_close()
    states = _real_states()
    sigma = S.ewma_vol(close)
    idx = states.index
    return {"close": close, "states": states, "sigma": sigma, "idx": idx}


def test_real_cache_hard_cut_and_sigma(real):
    assert real["close"].index.max() == pd.Timestamp(HARD_CUT)
    assert real["states"].index.max() == pd.Timestamp(HARD_CUT)
    assert str(real["sigma"].first_valid_index().date()) == "1993-04-27"     # 1993-02-01 + 60세션
    assert 0.05 < real["sigma"].dropna().median() < 0.25


def test_real_cache_adopted_rule_kpis(real):
    """채택 규칙 2003~24: 전환/년 ≤ 24 · 평균 비중 ∈ [0.5, 0.9] · 실현변동성/σ_T ∈ [0.75, 1.15] (§13)."""
    idx = real["idx"]
    res = S.run(real["sigma"].reindex(idx), real["states"], 0.10)
    a = S.allocation_from_weights(res["w_exec"], real["close"].reindex(idx))
    assert a["switches_per_year"] <= P3["churn_ceiling"]
    assert 0.50 <= a["avg_exposure"] <= 0.90
    assert 0.75 <= a["ann_vol"] / 0.10 <= 1.15
    assert res["w_exec"].min() >= P3["w_min"] - 1e-12 and res["w_exec"].max() <= P3["w_max"] + 1e-12
    assert set(res["reason"]) <= set(S.REASONS)
    assert a["max_dd"] > real["close"].pct_change().pipe(lambda r: ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min())


def test_real_cache_reproduces_design_stage_vt14_and_vol_only(real):
    """§6.3 의 파라미터 0개 참조 행을 소수 둘째 자리까지 재현한다 (상태 무관 = 모델 무관).

    S-VT14  : CAGR 10.14% · MaxDD −30.3% (2009-03-09) · 최악 월 −7.2% (2008-09) · 변동성 12.5% · 3.6회 · 87% · 1.0%
    변동성 단독 10% 주간 : 7.84% · −24.1% · −6.8% (2020-02) · 10.2% · 9.6회 · 72% · 1.5%
    """
    idx = real["idx"]
    sig = real["sigma"].reindex(idx)
    ones = pd.Series(1.0, index=idx)
    w_vt14 = S.exposure_target(sig, 1.0, 0.14)["w_vol"]
    vt14 = S.execute(w_vt14, ones, grid=S.GRID_SENSITIVITY["grid"], band=S.GRID_SENSITIVITY["band"],
                     cadence=S.GRID_SENSITIVITY["cadence"])["w_exec"]
    a = S.allocation_from_weights(vt14, real["close"].reindex(idx))
    assert round(a["cagr"] * 100, 2) == 10.14
    assert round(a["max_dd"] * 100, 1) == -30.3
    assert str(pd.Timestamp(a["max_dd_date"]).date()) == "2009-03-09"
    assert round(a["worst_month"] * 100, 1) == -7.2 and a["worst_month_label"] == "2008-09"
    assert round(a["ann_vol"] * 100, 1) == 12.5
    assert round(a["switches_per_year"], 1) == 3.6
    assert round(a["avg_exposure"] * 100) == 87
    assert round(a["cost_total"] * 100, 1) == 1.0

    w_vol = S.exposure_target(sig, 1.0, 0.10)["w_vol"]
    vo = S.execute(w_vol, ones)["w_exec"]
    b = S.allocation_from_weights(vo, real["close"].reindex(idx))
    assert round(b["cagr"] * 100, 2) == 7.84
    assert round(b["max_dd"] * 100, 1) == -24.1
    assert round(b["worst_month"] * 100, 1) == -6.8 and b["worst_month_label"] == "2020-02"
    assert round(b["ann_vol"] * 100, 1) == 10.2
    assert round(b["switches_per_year"], 1) == 9.6
    assert round(b["avg_exposure"] * 100) == 72
    assert round(b["cost_total"] * 100, 1) == 1.5


def test_real_cache_vol_only_ladder_ratio_backs_k_slow(real):
    """§6.2: 1993~ 변동성 단독 A격자 사다리의 MaxDD/σ_T 최악값 3.46 이 k_slow 3.5 의 근거."""
    close = real["close"]
    idx = close.index[close.index >= P3["vol_only_start"]]
    sig = real["sigma"].reindex(idx)
    ones = pd.Series(1.0, index=idx)
    ratios, cagrs, dds = [], [], []
    for g in P3["vol_grid"]:
        wv = S.exposure_target(sig, 1.0, g)["w_vol"]
        w = S.execute(wv, ones, grid=S.GRID_SENSITIVITY["grid"], band=S.GRID_SENSITIVITY["band"],
                      cadence=S.GRID_SENSITIVITY["cadence"])["w_exec"]
        a = S.allocation_from_weights(w, close.reindex(idx))
        ratios.append(abs(a["max_dd"]) / g)
        cagrs.append(round(a["cagr"] * 100, 2))
        dds.append(round(a["max_dd"] * 100, 1))
    assert cagrs == [4.54, 6.25, 7.03, 7.88, 9.25]                       # 설계 단계 실측 그대로
    assert dds == [-19.2, -24.8, -34.6, -35.2, -41.6]
    assert round(max(ratios), 2) == 3.46
    assert max(ratios) <= P3["k_slow"], "k_slow 3.5 가 실측 최악 비율을 덮어야 한다"


def test_real_cache_backtest_table_and_retention(real):
    """§6.3 표 + §6.4 유지 조건을 공식 상태(backtest_v1.csv)로 한 번 굴린다."""
    idx = real["idx"]
    close = real["close"]
    sig = real["sigma"]
    sT = S.target_vol(P3["d_max_default"])
    assert sT == 0.10
    res = S.run(sig.reindex(idx), real["states"], sT)
    idx93 = close.index[close.index >= P3["vol_only_start"]]
    wv93 = S.exposure_target(sig.reindex(idx93), 1.0, sT)["w_vol"]
    vol_only_93 = S.execute(wv93, pd.Series(1.0, index=idx93))["w_exec"]
    refs = {S.ROW_P2_DECISION: S.state_multiplier(real["states"]), S.ROW_VOL_ONLY: vol_only_93}
    tbl = S.backtest_table({S.ROW_ADOPTED: res["w_exec"]}, close,
                           {S.WINDOW_EVAL: (P3["eval_start"], HARD_CUT),
                            S.WINDOW_VOL_ONLY: (P3["vol_only_start"], HARD_CUT)},
                           references=refs, sigma_target=sT)
    assert len(tbl) == 8                                                  # 4행 × 2창 (p2/adopted 는 2003+ 만 유효해도 잘린 채 계산)
    ret = S.retention(tbl, sT)
    assert set(ret["conditions"]) == set("abcdef")
    assert not ret["undecided"] and not ret["missing"]
    assert isinstance(ret["deploy_sizing"], bool)
    a = tbl[(tbl["row"] == S.ROW_ADOPTED) & (tbl["window"] == S.WINDOW_EVAL)].iloc[0]
    assert -0.30 < a["max_dd"] < -0.20 and 0.05 < a["cagr"] < 0.09       # 설계 단계 이웃
    assert ret["conditions"]["d"]["pass"] and ret["conditions"]["f"]["pass"]


def test_real_cache_determinism_two_runs(real):
    idx = real["idx"]
    one = S.run(real["sigma"].reindex(idx), real["states"], 0.10)
    two = S.run(real["sigma"].reindex(idx), real["states"], 0.10)
    assert _frame_hash(one) == _frame_hash(two)


def test_real_cache_sensitivities_cover_contract(real):
    if not WF_CSV.exists():
        pytest.skip("results/calib_p2_walkforward.csv 없음")
    wf = pd.read_csv(WF_CSV, index_col=0, parse_dates=True)
    wf = wf.loc[wf.index <= pd.Timestamp(HARD_CUT)]
    sens = S.sensitivities(wf, real["states"], real["close"], sigma_target=0.10)
    assert set(sens) == set(S.SENSITIVITY_NAMES)
    for name, w in sens.items():
        assert isinstance(w, pd.Series) and w.notna().sum() > 100, name
    assert sens["S-nofloor"].min() < P3["w_min"]                          # 바닥 없는 곱은 0.25 아래로 간다
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        no_har = S.sensitivities(None, real["states"], real["close"], sigma_target=0.10)
    assert "S-HAR" not in no_har and any("S-HAR" in str(x.message) for x in rec)
