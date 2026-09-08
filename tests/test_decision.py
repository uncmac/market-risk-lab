# -*- coding: utf-8 -*-
"""mrl/decision.py 테스트 — 3단계 상태기계(히스테리시스·비대칭 dwell)·재개·결측·churn·KPI (ARCHITECTURE_PHASE2.md §9·§15).

실행: 프로젝트 루트에서  python -m pytest tests/test_decision.py -q
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrl import decision as D  # noqa: E402
from mrl import evaluate as E  # noqa: E402
from mrl import targets as T  # noqa: E402
from mrl.config import (DECISION_P2, DECISION_P2_SENSITIVITY, P2_STATES, STATE_TO_TONE,  # noqa: E402
                        TONE_EXPOSURE)

CLIM = 0.16


# ------------------------------------------------------------------
# 헬퍼
# ------------------------------------------------------------------
def _series(r_vals, start="2020-01-01") -> pd.Series:
    """r 경로 → p = r·clim (clim 상수) Series (거래일 인덱스)."""
    idx = pd.bdate_range(start, periods=len(r_vals))
    return pd.Series(np.asarray(r_vals, dtype=float) * CLIM, index=idx, name="p")


def _run(r_vals, cfg=D.DecisionConfig(), **kw) -> pd.DataFrame:
    return D.run(_series(r_vals), CLIM, cfg, **kw)


def _close(n: int, seed: int = 0, mu: float = 0.0003, sigma: float = 0.011) -> pd.Series:
    idx = pd.bdate_range("2014-01-02", periods=n)
    rng = np.random.default_rng(seed)
    px = 100.0 * np.exp(np.cumsum(rng.normal(mu, sigma, n)))
    return pd.Series(px, index=idx, name="SPY")


# ------------------------------------------------------------------
# 설정
# ------------------------------------------------------------------
def test_config_defaults_match_decision_p2():
    cfg = D.DecisionConfig()
    for k in ("enter_caution", "exit_caution", "enter_reduce", "exit_reduce", "dwell", "churn_alert"):
        assert getattr(cfg, k) == DECISION_P2[k], k
    assert cfg.dwell_escalate == 0                      # 격상은 즉시(기본)
    assert D.KPI_MAX_SWITCHES_PER_YEAR == DECISION_P2["kpi_max_switches_per_year"] == 12
    assert cfg.as_dict()["dwell"] == 5


def test_sensitivity_configs():
    cfgs = D.sensitivity_configs()
    assert list(cfgs) == ["default"] + list(DECISION_P2_SENSITIVITY)
    w = cfgs["wide"]
    assert (w.enter_caution, w.exit_caution, w.enter_reduce, w.exit_reduce) == (1.75, 1.25, 3.0, 2.25)
    assert w.dwell == 5
    assert cfgs["symmetric_dwell"].dwell_escalate == 5 and cfgs["symmetric_dwell"].dwell == 5
    assert cfgs["no_dwell"].dwell == 0
    with pytest.raises(ValueError):
        D.config_from_name("nonexistent")


def test_config_validation_hysteresis():
    with pytest.raises(ValueError):
        D.DecisionConfig(enter_caution=1.2, exit_caution=1.5)       # exit ≥ enter → 히스테리시스 없음
    with pytest.raises(ValueError):
        D.DecisionConfig(exit_reduce=2.5, enter_reduce=2.5)
    with pytest.raises(ValueError):
        D.DecisionConfig(dwell=-1)
    with pytest.raises(ValueError):
        D.DecisionConfig(dwell=2.5)                                  # 정수만


# ------------------------------------------------------------------
# step: 격상 즉시 · 히스테리시스 · 격하 dwell
# ------------------------------------------------------------------
def test_escalation_is_immediate():
    # normal → caution 첫 r ≥ 1.5 에서 즉시 (days_in_state 와 무관)
    for days in (0, 1, 3, 100):
        st, d, why = D.step(1.5, "normal", days)
        assert (st, d) == ("caution", 1) and "격상" in why
    # caution → reduce 즉시 (체류 1일이어도)
    st, d, why = D.step(2.5, "caution", 1)
    assert (st, d) == ("reduce", 1)
    # normal → reduce 직행
    st, d, _ = D.step(2.5, "normal", 0)
    assert (st, d) == ("reduce", 1)
    # 경계 바로 아래는 격상 아님
    assert D.step(1.4999, "normal", 0)[0] == "normal"
    assert D.step(2.4999, "caution", 1)[0] == "caution"
    # reduce 에서는 더 올라갈 곳 없음
    st, d, _ = D.step(9.0, "reduce", 2)
    assert (st, d) == ("reduce", 3)


def test_oscillation_inside_band_does_not_change_state():
    """[1.2, 1.5) 안에서 진동해도 caution 유지(히스테리시스)."""
    rng = np.random.default_rng(0)
    band = rng.uniform(1.2, 1.4999, size=200)
    out = _run([1.6] + list(band))
    assert out["state"].iloc[0] == "caution"
    assert (out["state"] == "caution").all()
    assert int(out["changed"].sum()) == 1
    assert list(out["days_in_state"]) == list(range(1, 202))
    # normal 에서도 [1.2,1.5) 는 normal 유지
    out2 = _run(list(band))
    assert (out2["state"] == "normal").all() and int(out2["changed"].sum()) == 0


def test_deescalation_requires_five_session_dwell():
    """t0 진입(days=1) → t4(days=5) 까지 유지 → t5 격하. 최소 체류 5세션."""
    out = _run([1.6] + [1.0] * 10)
    assert list(out["state"].iloc[:5]) == ["caution"] * 5
    assert out["state"].iloc[5] == "normal"
    assert list(out["days_in_state"].iloc[:7]) == [1, 2, 3, 4, 5, 1, 2]
    assert list(out["changed"].iloc[:7]) == [True, False, False, False, False, True, False]
    assert "체류 미달" in out["reason_ko"].iloc[1] and "격하" in out["reason_ko"].iloc[5]
    # step 수준: days=4 면 아직, days=5 면 격하
    assert D.step(1.0, "caution", 4)[:2] == ("caution", 5)
    assert D.step(1.0, "caution", 5)[:2] == ("normal", 1)
    # r 이 exit 경계(1.2) 정확히면 격하 아님 (r < 1.2 만)
    assert D.step(1.2, "caution", 10)[0] == "caution"
    assert D.step(1.1999, "caution", 10)[0] == "normal"


def test_reduce_to_normal_and_reduce_to_caution_paths():
    # reduce → normal 직행 (r < 1.2, 체류 ≥ 5), caution 을 거치지 않음
    out = _run([3.0] + [0.5] * 10)
    assert list(out["state"].iloc[:5]) == ["reduce"] * 5
    assert out["state"].iloc[5] == "normal"
    assert "caution" not in set(out["state"])
    # reduce → caution (1.2 ≤ r < 2.0, 체류 ≥ 5) → 그 뒤 caution → normal 은 다시 5세션 (비대칭: 격하마다 dwell)
    out = _run([3.0] + [1.5] * 6 + [1.0] * 10)
    assert list(out["state"].iloc[:5]) == ["reduce"] * 5
    assert out["state"].iloc[5] == "caution" and out["days_in_state"].iloc[5] == 1
    assert list(out["state"].iloc[5:10]) == ["caution"] * 5          # 7~10 은 r=1.0 이지만 체류 미달
    assert out["state"].iloc[10] == "normal"
    # reduce 에서 2.0 ≤ r 은 유지, r=2.0 정확히도 유지 (r < 2.0 만 격하)
    assert D.step(2.0, "reduce", 10)[0] == "reduce"
    assert D.step(1.9999, "reduce", 10)[:2] == ("caution", 1)
    assert D.step(1.1999, "reduce", 10)[:2] == ("normal", 1)


def test_initial_state_is_honored():
    out = _run([0.5, 0.5], init_state="reduce", init_days=5)
    assert out["state"].iloc[0] == "normal" and out["changed"].iloc[0]
    out = _run([0.5, 0.5], init_state="reduce", init_days=2)
    assert list(out["state"]) == ["reduce", "reduce"] and list(out["days_in_state"]) == [3, 4]
    with pytest.raises(ValueError):
        _run([0.5], init_state="panic")


# ------------------------------------------------------------------
# 재개 · 결측
# ------------------------------------------------------------------
def _random_r(n: int, seed: int, nan_share: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = rng.choice([0.5, 1.0, 1.3, 1.6, 2.2, 3.0], size=n, p=[0.3, 0.3, 0.15, 0.1, 0.1, 0.05])
    r[rng.uniform(size=n) < nan_share] = np.nan
    return r


def test_resume_from_ledger_state_equals_full_run():
    """장부의 (state, days) 에서 이어서 계산한 결과 == 처음부터 계산."""
    r = _random_r(600, seed=1)
    p = _series(r)
    full = D.run(p, CLIM)
    cols = ["r", "state", "days_in_state", "changed", "reason_ko"]
    for k in (1, 7, 100, 333, 599):
        head = D.run(p.iloc[:k], CLIM)
        last = head.iloc[-1]
        tail = D.run(p.iloc[k:], CLIM, init_state=str(last["state"]), init_days=int(last["days_in_state"]))
        joined = pd.concat([head, tail])
        pd.testing.assert_frame_equal(joined[cols], full[cols])


def test_missing_input_holds_state_and_counts_days():
    st, d, why = D.step(None, "caution", 3)
    assert (st, d, why) == ("caution", 4, D.REASON_MISSING)
    assert D.step(float("nan"), "reduce", 0) == ("reduce", 1, D.REASON_MISSING)
    assert D.step(np.nan, "normal", 9)[:2] == ("normal", 10)
    # 시계열: NaN 세션은 상태 유지·days 증가·changed False·r NaN, 그리고 dwell 에 산입
    out = _run([1.6, np.nan, np.nan, np.nan, np.nan, 1.0, 1.0])
    assert list(out["state"].iloc[:5]) == ["caution"] * 5
    assert out["state"].iloc[5] == "normal"                    # NaN 4세션도 체류로 센다
    assert out["r"].iloc[1:5].isna().all()
    assert not out["changed"].iloc[1:5].any()
    assert (out["reason_ko"].iloc[1:5] == D.REASON_MISSING).all()
    assert out.attrs["n_missing"] == 4
    assert any("확률 계산 불가" in w for w in out.attrs["warnings"])


def test_clim_missing_gives_nan_r_with_warning():
    p = _series([1.6, 1.6, 1.6])
    clim = pd.Series([CLIM, np.nan, 0.0], index=p.index)
    with pytest.warns(UserWarning):
        out = D.run(p, clim)
    assert out["r"].iloc[0] == pytest.approx(1.6)
    assert out["r"].iloc[1:].isna().all()
    assert list(out["state"]) == ["caution"] * 3
    assert any("clim" in w for w in out.attrs["warnings"])
    # clim 인덱스가 p 를 덮지 않으면 그 세션은 결측 처리 + 경고
    with pytest.warns(UserWarning):
        out2 = D.run(p, clim.iloc[:1])
    assert out2["r"].iloc[1:].isna().all()


def test_input_validation():
    with pytest.raises(ValueError):
        D.step(1.0, "panic", 0)
    with pytest.raises(ValueError):
        D.step(1.0, "normal", -1)
    with pytest.raises(ValueError):
        D.step(-0.1, "normal", 0)
    with pytest.raises(ValueError):
        D.step(float("inf"), "normal", 0)
    with pytest.raises(TypeError):
        D.step(1.0, "normal", 1.5)
    with pytest.raises(ValueError):
        D.run(pd.Series([1.2, 0.5], index=pd.bdate_range("2020-01-01", periods=2)), CLIM)   # p > 1
    dup = pd.Series([0.1, 0.1], index=pd.DatetimeIndex(["2020-01-02", "2020-01-02"]))
    with pytest.raises(ValueError):
        D.run(dup, CLIM)


# ------------------------------------------------------------------
# churn · 구조적 상한
# ------------------------------------------------------------------
ADVERSARIAL_CYCLE = [3.0, 3.0, 3.0, 3.0, 3.0, 0.5, 1.6]    # reduce 5세션 → normal → caution → reduce (7세션마다 3회)


def test_churn_alert_on_adversarial_path():
    r = (ADVERSARIAL_CYCLE * 60)[:400]
    out = _run(r)
    assert out["churn_alert"].any()
    first = out.index[out["churn_alert"]][0]
    assert out.loc[first, "churn_252"] > DECISION_P2["churn_alert"]
    assert out["churn_252"].max() <= 252
    assert any("churn" in w for w in out.attrs["warnings"])
    # churn_252 는 직전 252세션(오늘 포함)의 변경 수와 같다
    manual = out["changed"].astype(int).rolling(252, min_periods=1).sum().astype(int)
    assert (out["churn_252"] == manual).all()
    # 조용한 경로에서는 경보 없음
    quiet = _run([1.0] * 400 + [1.6] + [1.3] * 100)
    assert not quiet["churn_alert"].any()


def test_structural_bound_max_three_changes_in_any_five_sessions():
    """dwell=5 이면 어떤 5세션 창에서도 변경 ≤ 3 (격하 1 + 격상 2). 적대적·무작위 경로 모두."""
    rng = np.random.default_rng(0)
    for seed in range(5):
        r = _random_r(3000, seed=seed, nan_share=0.02)
        out = _run(r)
        max5 = int(out["changed"].astype(int).rolling(5, min_periods=1).sum().max())
        assert max5 <= D.STRUCTURAL_MAX_CHANGES_5
    # 극단 경로: 0/3 를 매 세션 번갈아도 ≤ 3
    alt = [3.0, 0.0] * 500
    out = _run(alt)
    assert int(out["changed"].astype(int).rolling(5, min_periods=1).sum().max()) <= 3
    # 정확히 3회를 달성하는 경로 (상한이 tight)
    out = _run(ADVERSARIAL_CYCLE * 10)
    assert int(out["changed"].astype(int).rolling(5, min_periods=1).sum().max()) == 3
    # dwell 이 없으면 상한이 깨진다 — 상한이 dwell 에서 나옴을 증명
    out0 = _run(alt, cfg=D.config_from_name("no_dwell"))
    assert int(out0["changed"].astype(int).rolling(5, min_periods=1).sum().max()) > 3
    del rng


def test_symmetric_dwell_and_no_dwell_variants():
    sym = D.config_from_name("symmetric_dwell")
    # 격상도 체류 5 필요: normal(days 0) 에서 r=3 이어도 유지, days=5 부터 격상
    assert D.step(3.0, "normal", 0, sym)[:2] == ("normal", 1)
    assert D.step(3.0, "normal", 4, sym)[:2] == ("normal", 5)
    assert D.step(3.0, "normal", 5, sym)[:2] == ("reduce", 1)
    out = _run([3.0] * 8, cfg=sym)
    assert list(out["state"]) == ["normal"] * 5 + ["reduce"] * 3
    # no_dwell: 격하도 즉시
    nd = D.config_from_name("no_dwell")
    out = _run([1.6, 1.0, 1.6, 1.0], cfg=nd)
    assert list(out["state"]) == ["caution", "normal", "caution", "normal"]
    # wide: 1.6 은 격상 아님(1.75), 2.9 도 reduce 아님(3.0)
    wd = D.config_from_name("wide")
    assert D.step(1.6, "normal", 0, wd)[0] == "normal"
    assert D.step(1.75, "normal", 0, wd)[0] == "caution"
    assert D.step(2.9, "caution", 1, wd)[0] == "caution"
    assert D.step(1.24, "caution", 5, wd)[0] == "normal"


# ------------------------------------------------------------------
# 톤 · 임계 환산 · 출력 형식
# ------------------------------------------------------------------
def test_to_tone_mapping():
    s = pd.Series(["normal", "caution", "reduce"])
    t = D.to_tone(s)
    assert list(t) == ["hold", "caution", "reduce"] and t.name == "tone"
    assert set(t) <= set(TONE_EXPOSURE)                           # allocation_sim 이 그대로 받는다
    assert {STATE_TO_TONE[s] for s in P2_STATES} == {"hold", "caution", "reduce"}
    with pytest.raises(ValueError):
        D.to_tone(pd.Series(["normal", "panic"]))
    with pytest.raises(ValueError):
        D.to_tone(pd.Series(["normal", None]))


def test_next_thresholds():
    t = D.next_thresholds("normal", 0, 0.16)
    assert t["escalate_to"] == "caution" and t["p_escalate"] == pytest.approx(0.24)
    assert t["deescalate_to"] is None and t["dwell_remaining"] is None
    t = D.next_thresholds("caution", 2, 0.16)
    assert t["escalate_to"] == "reduce" and t["p_escalate"] == pytest.approx(0.40)
    assert t["deescalate_to"] == "normal" and t["p_deescalate"] == pytest.approx(0.192)
    assert t["dwell_remaining"] == 3
    t = D.next_thresholds("reduce", 9, 0.16)
    assert t["escalate_to"] is None and t["deescalate_to"] == "caution" and t["dwell_remaining"] == 0
    assert math.isnan(D.next_thresholds("caution", 1, float("nan"))["p_escalate"])


def test_run_output_contract():
    out = _run(_random_r(50, seed=3))
    assert list(out.columns) == list(D.RUN_COLUMNS)
    assert out["state"].isin(P2_STATES).all()
    assert out["days_in_state"].dtype.kind == "i" and out["churn_252"].dtype.kind == "i"
    assert out["changed"].dtype == bool and out["churn_alert"].dtype == bool
    assert (out["reason_ko"].str.len() > 0).all()
    assert out.index.equals(_series(_random_r(50, seed=3)).index)
    assert out.attrs["config"] == D.DecisionConfig().as_dict()
    # changed 인 행은 days_in_state == 1
    assert (out.loc[out["changed"], "days_in_state"] == 1).all()
    # 빈 입력
    empty = D.run(pd.Series([], dtype=float, index=pd.DatetimeIndex([])), CLIM)
    assert len(empty) == 0 and list(empty.columns) == list(D.RUN_COLUMNS)


# ------------------------------------------------------------------
# KPI
# ------------------------------------------------------------------
@pytest.fixture(scope="module")
def synth_kpi():
    close = _close(1500, seed=0)
    targets = T.make_targets(close)
    idx = close.index[300:1300]
    rng = np.random.default_rng(0)
    p = pd.Series(np.clip(rng.beta(2, 8, size=len(idx)) * 2.0, 0, 1), index=idx)   # 평균 ≈ 0.4 → 자주 caution
    clim = pd.Series(0.16, index=idx)
    states = D.run(p, clim)
    return close, targets, states


def test_kpis_keys_and_consistency(synth_kpi):
    close, targets, states = synth_kpi
    y = targets["y_dd5_20"]
    k = D.kpis(states, y, close)
    for key in ("n_sessions", "years", "n_changes", "switches_per_year", "kpi_ceiling", "kpi_ceiling_ok",
                "occupancy", "n_by_state", "dd5_rate_by_state", "base_rate", "warn_share",
                "n_warn_runs", "median_warn_run", "median_run_by_state",
                "true_alarm_share", "true_alarm_share_baseline", "true_alarm_edge", "true_alarm_share_v0_ref",
                "max_changes_any_5_sessions", "structural_bound", "structural_bound_ok",
                "churn_alert_sessions", "churn_alert_any", "max_churn_252", "n_missing", "missing_share",
                "allocation", "warnings", "config"):
        assert key in k, key
    n = len(states)
    assert k["n_sessions"] == n and k["years"] == pytest.approx(n / 252)
    assert k["n_changes"] == int(states["changed"].sum())
    assert k["switches_per_year"] == pytest.approx(k["n_changes"] / (n / 252))
    assert k["kpi_ceiling"] == 12 and k["kpi_ceiling_ok"] == (k["switches_per_year"] <= 12)
    assert set(k["occupancy"]) == set(P2_STATES) and sum(k["occupancy"].values()) == pytest.approx(1.0)
    assert sum(k["n_by_state"].values()) == n
    assert k["base_rate"] == pytest.approx(y.reindex(states.index).mean())
    # 진짜 경보 비중: evaluate._true_alarm_share 와 동일 (경고 런 시작일 기준)
    is_warn = states["state"].isin(D.WARN_STATES).to_numpy()
    ref = E._true_alarm_share(is_warn, y.reindex(states.index).to_numpy(dtype=float))
    assert k["true_alarm_share"] == pytest.approx(ref)
    assert k["true_alarm_edge"] == pytest.approx(ref - k["base_rate"])
    assert k["true_alarm_share_v0_ref"]["lo"] < k["true_alarm_share_v0_ref"]["hi"]
    assert k["n_warn_runs"] == len(E._warning_runs(pd.Series(is_warn)))
    assert k["max_changes_any_5_sessions"] <= 3 and k["structural_bound"] == 3 and k["structural_bound_ok"] is True
    for s in P2_STATES:
        v = k["dd5_rate_by_state"][s]
        assert math.isnan(v) or 0.0 <= v <= 1.0
    # 배분 시뮬은 evaluate.allocation_sim 전체(톤 = STATE_TO_TONE, 비용 5bp)
    alloc = k["allocation"]
    assert alloc["cost_bps"] == 5.0 and alloc["exposure_map"] == TONE_EXPOSURE
    for key in ("cagr", "max_dd", "switches_per_year", "bh_cagr", "excess_cagr", "series"):
        assert key in alloc
    tone_df = pd.DataFrame({"tone": D.to_tone(states["state"])}, index=states.index)
    direct = E.allocation_sim(tone_df, close)
    assert alloc["cagr"] == pytest.approx(direct["cagr"]) and alloc["n_switches"] == direct["n_switches"]
    # 톤 전환 = 상태 전환. 단 allocation_sim 은 첫 세션(직전 비중 없음)의 진입을 전환으로 세지 않는다
    assert alloc["n_switches"] == int(states["changed"].iloc[1:].sum())
    # JSON 직렬화
    js = json.dumps(D.kpis_jsonable(k), ensure_ascii=False)
    assert "series" not in json.loads(js)["allocation"]


def test_kpis_without_close_and_missing_y(synth_kpi):
    close, targets, states = synth_kpi
    k = D.kpis(states, targets["y_dd5_20"], None)
    assert k["allocation"] is None and any("배분" in w for w in k["warnings"])
    y_none = pd.Series(np.nan, index=states.index)
    k2 = D.kpis(states, y_none, close)
    assert math.isnan(k2["true_alarm_share"]) and math.isnan(k2["base_rate"])
    assert all(math.isnan(v) for v in k2["dd5_rate_by_state"].values())
    with pytest.raises(ValueError):
        D.kpis(states, pd.Series(0.5, index=states.index), close)     # y 는 0/1 만


def test_kpis_switches_per_year_ceiling_flag():
    r = (ADVERSARIAL_CYCLE * 60)[:400]
    states = _run(r)
    y = pd.Series(0.0, index=states.index)
    k = D.kpis(states, y, None)
    assert k["switches_per_year"] > 12 and k["kpi_ceiling_ok"] is False
    assert k["churn_alert_any"] is True and k["churn_alert_sessions"] > 0
    assert k["median_warn_run"] >= 5                                # 경고 런은 최소 5세션(dwell)
    assert k["median_run_by_state"]["reduce"] >= 5
