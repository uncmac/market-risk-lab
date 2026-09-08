# -*- coding: utf-8 -*-
"""mrl/ensemble.py 검증 (ARCHITECTURE_PHASE3.md §5·§13).

실행: 프로젝트 루트에서  python -m pytest tests/test_ensemble.py -q

* `mrl/regime.py` 는 동시 작성 중이므로 여기서는 **합성 x_hmm** 만 쓴다(계약: logit P_high 하나의 열).
  regime 이 들어오면 `member_h_walk_forward` 로 같은 경로를 실캐시에서 확인한다.
* 채택 검정의 '신선 블록'은 라이브 장부(2026-10~)를 모사한 합성 세션 인덱스로 만든다 —
  관측 기록(2003~2024-08)과 홀드아웃(2024-09-01~)은 **기각만** 할 수 있다는 규칙 자체를 검사한다.
"""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from mrl import calibrate as C
from mrl import ensemble as E
from mrl import model as M
from mrl.calendar_us import trading_days
from mrl.config import BLOCKS_24, ENSEMBLE_P3, HMM_P3, HOLDOUT_START, ROOT

HOLDOUT = pd.Timestamp(HOLDOUT_START)

# REGISTRY 튜플의 동결 해시 — 등록부를 바꾸면 이 값이 바뀌고 테스트가 실패한다.
# 그때는 코드가 아니라 **장부에 번호 붙인 항목**을 먼저 쓰고 여기 값을 갱신한다 (§5.1).
REGISTRY_TUPLE_SHA256 = "05c99a91952b384e3ac6c1addb50e858139dbd1d973c21ccbbd6641851731312"


# ==================================================================
# 5.1 등록부 동결
# ==================================================================
def test_registry_is_frozen_tuple():
    assert isinstance(E.REGISTRY, tuple)
    assert E.REGISTRY_NAMES == ("p2", "M1", "H") == tuple(ENSEMBLE_P3["members"])
    with pytest.raises(dataclasses.FrozenInstanceError):
        E.REGISTRY[0].name = "x"                       # 멤버는 frozen dataclass
    with pytest.raises(TypeError):
        E.REGISTRY[0] = E.REGISTRY[1]                  # 등록부는 튜플(불변)


def test_registry_rows_match_spec_table():
    """§5.1 표 그대로: 종류·K_s·K_u·1일차 상태·장부 번호."""
    p2, m1, h = E.REGISTRY
    assert (p2.name, p2.kind, p2.k_s, p2.k_u, p2.day1_status, p2.ledger_no) == ("p2", "seed", 4, 0, "seed", "#2")
    assert (m1.name, m1.kind, m1.k_s, m1.k_u, m1.day1_status, m1.ledger_no) == ("M1", "ladder", 2, 0, "shadow", "#8")
    assert (h.name, h.kind, h.k_s, h.k_u, h.day1_status, h.ledger_no) == ("H", "hmm_platt", 2, 12, "shadow", "#3a")
    assert h.k_u == HMM_P3["param_count"] == 12
    assert p2.k_s == M.PARAM_COUNT == 4
    # (미등록) 폭/신용 멤버 B 는 등록부에 없다 — 수치만 기록
    assert E.UNREGISTERED[0].name == "B" and "B" not in E.REGISTRY_NAMES
    assert E.UNREGISTERED[0].day1_status == "candidate_rejected"


def test_registry_tuple_sha256_is_frozen():
    assert E.registry_tuple_sha256() == REGISTRY_TUPLE_SHA256
    assert E.registry_repr().splitlines()[0] == f"ensemble-registry-v{E.SCHEMA_VERSION}"


def test_registry_tuple_sha256_changes_when_registry_changes(monkeypatch):
    extra = E.REGISTRY + (E.Member("M1x", "ladder", "p_m1x", 2, 0, "shadow", "#9"),)
    monkeypatch.setattr(E, "REGISTRY", extra)
    assert E.registry_tuple_sha256() != REGISTRY_TUPLE_SHA256


def test_registry_sha256_covers_sources():
    s = E.registry_sha256()
    assert re.fullmatch(r"[0-9a-f]{64}", s)
    assert s != E.registry_tuple_sha256()               # 소스까지 들어가므로 다르다
    assert E.registry_sha256() == s                     # 결정적


def test_parameter_accounting_day1():
    st = E.day1_statuses()
    assert st == {"p2": "seed", "M1": "shadow", "H": "shadow"}
    assert E.production_members(st) == ("p2",)
    assert E.production_param_count(st) == M.PARAM_COUNT == 4
    acc = E.shadow_param_counts(st)
    assert acc == {"K_s_production": 4, "K_u_production": 0, "K_s_shadow": 4, "K_u_shadow": 12,
                   "phase3_added_to_production": 0}
    assert acc["K_s_production"] + acc["K_s_shadow"] <= ENSEMBLE_P3["K_s_cap"]
    assert acc["K_u_production"] + acc["K_u_shadow"] <= ENSEMBLE_P3["K_u_cap"]


def test_parameter_accounting_conserves_k_u_after_admission():
    """채택되면 HMM θ 의 비지도 12개는 **생산** 줄로 옮겨 공개한다 — 회계에서 사라지면 안 된다 (§14)."""
    st = E.day1_statuses()
    adm = E.shadow_param_counts({**st, "H": "admitted"})
    assert adm["K_s_production"] == 6 and adm["K_u_production"] == 12   # §14: M3+H = K_s 6 · K_u 12
    assert adm["K_u_shadow"] == 0 and adm["phase3_added_to_production"] == 2
    for s in ({}, {"H": "admitted"}, {"H": "admitted", "M1": "admitted"}, {"H": "killed"},
              {"H": "candidate_rejected"}, {"M1": "admitted"}):
        a = E.shadow_param_counts({**st, **s})
        assert a["K_u_production"] + a["K_u_shadow"] == 12              # K_u 총계 보존
        assert a["K_s_production"] + a["K_s_shadow"] == 8
        assert a["K_u_production"] + a["K_u_shadow"] <= ENSEMBLE_P3["K_u_cap"]


def test_check_statuses_rejects_unknown():
    with pytest.raises(ValueError, match="등록부에 없는"):
        E.check_statuses({"ZZ": "shadow"})
    with pytest.raises(ValueError, match="알 수 없는 상태"):
        E.check_statuses({"H": "great"})
    with pytest.raises(ValueError, match="지정되지 않은"):
        E.check_statuses({"p2": "seed"}, ("p2", "H"))


# ==================================================================
# 5.2 그림자 멤버 H 의 Platt — Phase 2 M1/M3 와 같은 레시피
# ==================================================================
def _platt_fixture(seed: int = 0):
    """합성 x_hmm(= logit P_high) + 라벨. 실제 인덱스(거래일)로 재적합 일정을 그대로 쓴다."""
    idx = trading_days("1994-01-03", "2006-12-29")
    rng = np.random.default_rng(seed)
    n = len(idx)
    x = np.empty(n)                                     # AR(1) — P_high 의 지속성 모사
    x[0] = 0.0
    eps = rng.normal(0.0, 0.45, n)
    for t in range(1, n):
        x[t] = 0.985 * x[t - 1] + eps[t]
    p = expit(-1.7 + 0.8 * x)
    y = (rng.random(n) < p).astype(float)
    ys = pd.Series(y, index=idx, name="y")
    ys.iloc[-20:] = np.nan                              # 라벨 미실현 (20일 지평)
    return pd.Series(x, index=idx, name="x_hmm"), ys


def test_platt_walk_forward_recipe_parity_with_phase2():
    """H 의 Platt 계수 == Phase 2 의 `calibrate.fit_refit`(같은 sklearn 사양) 결과와 비트 동일."""
    x, y = _platt_fixture()
    rds = C.refit_dates(x.index, first="2003-01-02", end="2007-01-01")
    assert [d.year for d in rds] == [2003, 2004, 2005, 2006]
    df, models = E.platt_walk_forward(x, y, rds)
    feats = x.to_frame(E.PLATT_FEATURE)
    for R, m in zip(rds, models):
        ref = C.fit_refit(feats, y, R, (E.PLATT_FEATURE,), C=1.0, purge=HMM_P3["purge"],
                          train_start=HMM_P3["train_start"], rung="H")
        assert m.coef[E.PLATT_FEATURE] == ref.coef[E.PLATT_FEATURE]
        assert m.intercept == ref.intercept
        assert (m.n_train, m.n_pos, m.clim, m.train_start, m.train_end) == \
               (ref.n_train, ref.n_pos, ref.clim, ref.train_start, ref.train_end)
        assert m.n_params() == E.member("H").k_s == 2   # K_s = 2 (예산 밖 공개값)


def test_platt_uses_phase2_sklearn_kwargs():
    """같은 학습 행에 `model.FIT_KWARGS` 로 직접 적합해도 계수가 같다 (레시피가 갈라지지 않았다)."""
    x, y = _platt_fixture()
    rds = C.refit_dates(x.index, first="2003-01-02", end="2007-01-01")
    _, models = E.platt_walk_forward(x, y, rds)
    R, m = rds[1], models[1]
    mask = (C.training_mask(x.index, R, HMM_P3["purge"], HMM_P3["train_start"])
            & y.notna().to_numpy() & np.isfinite(x.to_numpy()))
    lr = LogisticRegression(**M.FIT_KWARGS)
    lr.fit(x[mask].to_numpy().reshape(-1, 1), y[mask].to_numpy())
    assert float(lr.coef_[0][0]) == m.coef[E.PLATT_FEATURE]
    assert float(lr.intercept_[0]) == m.intercept
    assert M.FIT_KWARGS["C"] == HMM_P3["platt_C"] == 1.0
    assert M.FIT_KWARGS["tol"] == 1e-8 and M.FIT_KWARGS["solver"] == "lbfgs" and M.FIT_KWARGS["penalty"] == "l2"


def test_platt_walk_forward_windows_purge_and_nan():
    x, y = _platt_fixture()
    rds = C.refit_dates(x.index, first="2003-01-02", end="2007-01-01")
    x2 = x.copy()
    x2.iloc[-5:] = np.nan                               # 입력 결측 → 확률 NaN (채우지 않는다)
    df, models = E.platt_walk_forward(x2, y, rds)
    assert list(df.columns) == ["p_h", "refit_year", "in_sample"]
    assert df["p_h"].iloc[-5:].isna().all()
    # OOS 창: [R_y, R_{y+1}) 가 그 해 모델로 채점된다
    for i, R in enumerate(rds):
        nxt = rds[i + 1] if i + 1 < len(rds) else None
        sel = (df.index >= R) if nxt is None else ((df.index >= R) & (df.index < nxt))
        assert (df.loc[sel, "refit_year"] == R.year).all()
        assert not df.loc[sel, "in_sample"].any()
    assert df.loc[df.index < rds[0], "in_sample"].all()  # 첫 재적합 이전은 in_sample 표시
    # 퍼지: 학습 마지막 행 + 20세션 < 재적합일
    for R, m in zip(rds, models):
        pos_R = int(x.index.get_loc(R))
        pos_end = int(x.index.get_loc(pd.Timestamp(m.train_end)))
        assert pos_end + HMM_P3["purge"] < pos_R


def test_platt_point_in_time_future_rows_do_not_change_past():
    """미래 행을 덧붙여도 과거 p_h 가 바뀌지 않는다(점 원칙)."""
    x, y = _platt_fixture()
    cut = pd.Timestamp("2005-06-30")
    rds = C.refit_dates(x.index, first="2003-01-02", end="2005-06-30")
    full, _ = E.platt_walk_forward(x, y, rds)
    part, _ = E.platt_walk_forward(x[x.index <= cut], y[y.index <= cut], rds)
    a = full.loc[full.index <= cut, "p_h"].to_numpy()
    b = part["p_h"].to_numpy()
    assert np.array_equal(a, b, equal_nan=True)


# ==================================================================
# 5.3 동일가중 결합
# ==================================================================
def _probs(n: int = 8, seed: int = 3) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.bdate_range("2027-01-04", periods=n), name="date")
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"p2": rng.uniform(0.05, 0.4, n), "M1": rng.uniform(0.05, 0.4, n),
                         "H": rng.uniform(0.05, 0.4, n)}, index=idx)


def test_combine_day1_is_bit_identical_to_p2():
    probs = _probs()
    out = E.combine(probs, E.day1_statuses())
    assert np.array_equal(out.to_numpy(), probs["p2"].to_numpy(), equal_nan=True)
    assert out.attrs["members"] == ("p2",) and out.attrs["weight"] == 1.0


def test_combine_equal_weight_math_and_no_reweighting():
    probs = _probs()
    st = {**E.day1_statuses(), "H": "admitted"}
    out = E.combine(probs, st)
    exp = (probs["p2"].to_numpy() + probs["H"].to_numpy()) / 2.0
    assert np.allclose(out.to_numpy(), exp, rtol=0, atol=0)
    assert out.attrs["members"] == ("p2", "H") and out.attrs["weight"] == 0.5
    st3 = {"p2": "seed", "M1": "admitted", "H": "admitted"}
    out3 = E.combine(probs, st3)
    assert np.allclose(out3.to_numpy(), probs[["p2", "M1", "H"]].to_numpy().mean(axis=1), rtol=0, atol=0)


def test_combine_nan_propagates_for_admitted_members_only():
    probs = _probs()
    probs.iloc[2, probs.columns.get_loc("H")] = np.nan
    probs.iloc[3, probs.columns.get_loc("M1")] = np.nan
    st = {**E.day1_statuses(), "H": "admitted"}          # M1 은 shadow
    out = E.combine(probs, st)
    assert np.isnan(out.iloc[2])                          # admitted 멤버 NaN → 행 NaN (조용히 빼지 않는다)
    assert np.isfinite(out.iloc[3])                       # shadow 멤버 NaN 은 평균과 무관
    assert out.iloc[3] == pytest.approx((probs["p2"].iloc[3] + probs["H"].iloc[3]) / 2.0)


def test_combine_excludes_killed_and_shadow_and_validates_input():
    probs = _probs()
    out = E.combine(probs, {"p2": "seed", "M1": "killed", "H": "candidate_rejected"})
    assert np.array_equal(out.to_numpy(), probs["p2"].to_numpy(), equal_nan=True)
    with pytest.warns(RuntimeWarning, match="평균에 들어갈 멤버가 없습니다"):
        empty = E.combine(probs, {"p2": "killed", "M1": "shadow", "H": "shadow"})
    assert empty.isna().all()
    with pytest.raises(ValueError, match="등록부 밖 열"):
        E.combine(probs.rename(columns={"H": "HH"}), {**E.day1_statuses(), "M1": "shadow"})
    bad = probs.copy()
    bad.iloc[0, 0] = 1.5
    with pytest.raises(ValueError, match=r"\[0,1\] 밖"):
        E.combine(bad, E.day1_statuses())


def test_member_probs_resolves_ledger_and_oos_columns():
    idx = pd.DatetimeIndex(pd.bdate_range("2027-01-04", periods=4), name="date")
    frame = pd.DataFrame({"prob_dd5_20": 0.2, "p2_p_m1": 0.3, "p3_p_h": 0.4}, index=idx)
    out = E.member_probs(frame)
    assert list(out.columns) == ["p2", "M1", "H"]
    assert out["M1"].eq(0.3).all() and out["H"].eq(0.4).all()
    with pytest.raises(ValueError, match="확률 열을 찾지 못했습니다"):
        E.member_probs(frame.drop(columns=["p3_p_h"]))
    with pytest.warns(RuntimeWarning):
        part = E.member_probs(frame.drop(columns=["p3_p_h"]), required=False)
    assert list(part.columns) == ["p2", "M1"]


# ==================================================================
# 5.3 불일치 구간
# ==================================================================
def _band_fixture(n: int = 20):
    idx = pd.DatetimeIndex(pd.bdate_range("2027-01-04", periods=n), name="date")
    probs = pd.DataFrame({"p2": 0.12, "M1": 0.12, "H": 0.12}, index=idx)
    probs.iloc[3:8, probs.columns.get_loc("H")] = 0.50    # 5세션 연속 넓음
    probs.iloc[12:16, probs.columns.get_loc("H")] = 0.50  # 4세션만 넓음
    lo = pd.Series(0.10, index=idx)
    hi = pd.Series(0.15, index=idx)
    return probs, lo, hi


def test_disagreement_contains_p2_band_and_flags_runs():
    probs, lo, hi = _band_fixture()
    out = E.disagreement(probs, E.day1_statuses(), lo, hi)
    assert list(out.columns) == ["lo", "hi", "width", "src", "flag"]
    assert (out["lo"] <= lo).all() and (out["hi"] >= hi).all()          # p2 밴드를 언제나 포함
    assert out["lo"].eq(0.10).all()                                     # 멤버가 아래로는 넓히지 않았다
    assert out["hi"].iloc[3:8].eq(0.50).all() and out["width"].iloc[3:8].eq(0.40).all()
    assert out["src"].iloc[3:8].eq("members").all()
    assert out["src"].iloc[0] == "p2" and out["width"].iloc[0] == pytest.approx(0.05)
    assert out["flag"].iloc[3:8].all()                                  # 5세션 연속 → 플래그
    assert not out["flag"].iloc[12:16].any()                            # 4세션은 아님
    assert int(out["flag"].sum()) == 5
    assert ENSEMBLE_P3["disagree_flag"] == {"width": 0.20, "sessions": 5}


def test_disagreement_drops_killed_member():
    probs, lo, hi = _band_fixture()
    out = E.disagreement(probs, {**E.day1_statuses(), "H": "killed"}, lo, hi)
    assert out["hi"].eq(0.15).all() and np.allclose(out["width"].to_numpy(), 0.05)
    assert out["src"].eq("p2").all() and not out["flag"].any()
    assert out.attrs["excluded"] == ("H",)
    keep = E.disagreement(probs, {**E.day1_statuses(), "H": "candidate_rejected"}, lo, hi)
    assert keep["hi"].iloc[3:8].eq(0.50).all()                          # killed 만 빠진다 (§5.1)


def test_disagreement_handles_missing_rows():
    probs, lo, hi = _band_fixture(6)
    probs.iloc[0, :] = np.nan
    lo2, hi2 = lo.copy(), hi.copy()
    lo2.iloc[1] = np.nan
    hi2.iloc[1] = np.nan
    out = E.disagreement(probs, E.day1_statuses(), lo2, hi2)
    assert out["lo"].iloc[0] == 0.10 and out["hi"].iloc[0] == 0.15      # 멤버 전부 NaN → p2 밴드만
    assert out["lo"].iloc[1] == pytest.approx(0.12) and out["src"].iloc[1] == "members"
    with pytest.raises(ValueError, match="뒤집힌"):
        E.disagreement(probs, E.day1_statuses(), hi, lo)


# ==================================================================
# 5.4 채택 규칙 — 신선 블록만, 관측 기록·홀드아웃은 기각만
# ==================================================================
FRESH_START = "2026-10-01"


def _fresh_index() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.bdate_range(FRESH_START, "2029-08-31"), name="date")


def _frame(idx, blocks, eps_by_block, seed: int = 0) -> pd.DataFrame:
    """p2 = 0.5 고정, H = 0.5 + eps·(2y−1). eps > 0 이면 그 블록은 반드시 개선, eps < 0 이면 악화."""
    rng = np.random.default_rng(seed)
    y = (rng.random(len(idx)) < 0.15).astype(float)
    bid = C.assign_blocks(idx, blocks)
    h = np.full(len(idx), 0.5)
    for k, eps in eps_by_block.items():
        m = bid == k
        h[m] = 0.5 + float(eps) * (2.0 * y[m] - 1.0)
    return pd.DataFrame({"y": y, "p2": 0.5, "H": h}, index=idx)


def test_fresh_blocks_tile_from_live_start():
    idx = _fresh_index()
    blocks = E.fresh_blocks(idx, FRESH_START)
    assert len(blocks) == ENSEMBLE_P3["fresh_blocks_min"] == 11
    assert blocks[0] == (pd.Timestamp("2026-10-01"), pd.Timestamp("2027-01-01"))
    assert blocks[-1] == (pd.Timestamp("2029-04-01"), pd.Timestamp("2029-07-01"))
    tbl = E.fresh_block_table(idx, FRESH_START)
    assert len(tbl) == 12 and bool(tbl["complete"].iloc[-1]) is False   # 마지막 타일은 라벨 미실현
    assert tbl["complete"].iloc[:11].all()
    assert ENSEMBLE_P3["fresh_block_months"] == 3


def test_fresh_blocks_incomplete_when_label_not_realized():
    idx = pd.DatetimeIndex(pd.bdate_range(FRESH_START, "2027-01-08"), name="date")
    assert E.fresh_blocks(idx, FRESH_START) == []       # 20세션 라벨 미실현 → 미완결
    tbl = E.fresh_block_table(idx, FRESH_START)
    assert tbl["n"].iloc[0] > 0 and tbl["sessions_after"].iloc[0] < 20


def test_fresh_blocks_refuses_holdout_start():
    idx = _fresh_index()
    with pytest.raises(ValueError, match="홀드아웃"):
        E.fresh_blocks(idx, HOLDOUT_START)              # 2024-09-01 → 금지
    with pytest.raises(ValueError, match="홀드아웃"):
        E.fresh_blocks(idx, "2010-01-04")
    with pytest.raises(ValueError, match="홀드아웃"):
        E.fresh_block_table(idx, HOLDOUT_START)


def test_admission_test_admits_8_of_11_on_fresh_blocks():
    idx = _fresh_index()
    blocks = E.fresh_blocks(idx, FRESH_START)
    eps = {k: (0.30 if k <= 8 else -0.001) for k in range(1, 12)}
    res = E.admission_test(_frame(idx, blocks, eps), "H", ("p2",), blocks,
                           live_start=FRESH_START, n_boot=500)
    assert res["evidence"] == "fresh" and res["admissible"] is True
    assert (res["wins"], res["n_blocks"], res["need"]) == (8, 11, 8)
    assert res["no_harm"] is True and res["ci"]["hi"] > 0
    assert res["verdict"] == res["verdict_raw"] == "ADMIT"
    assert sum(1 for v in res["per_block"] if v > 0) == 8
    assert res["frac"] == ENSEMBLE_P3["admit_frac"] == 8 / 11


def test_admission_test_shadow_at_7_of_11():
    idx = _fresh_index()
    blocks = E.fresh_blocks(idx, FRESH_START)
    eps = {k: (0.30 if k <= 7 else -0.001) for k in range(1, 12)}
    res = E.admission_test(_frame(idx, blocks, eps), "H", ("p2",), blocks,
                           live_start=FRESH_START, n_boot=500)
    assert (res["wins"], res["need"]) == (7, 8)
    assert res["verdict"] == "SHADOW"


def test_admission_test_ci_veto_even_at_8_of_11():
    """블록은 8개 이겨도 풀링 손실차 CI 상한 ≤ 0 이면 거부권이 작동한다."""
    idx = _fresh_index()
    blocks = E.fresh_blocks(idx, FRESH_START)
    eps = {k: (0.001 if k <= 8 else -0.45) for k in range(1, 12)}
    res = E.admission_test(_frame(idx, blocks, eps), "H", ("p2",), blocks,
                           live_start=FRESH_START, n_boot=500)
    assert res["wins"] == 8 >= res["need"]
    assert res["pooled"] < 0 and res["ci"]["hi"] <= 0 and res["no_harm"] is False
    assert res["verdict"] == "SHADOW"


def test_admission_test_observed_record_can_never_admit():
    """2003~2024-08 관측 기록에서 11/11 을 이겨도 채택은 불가 — 기각만 할 수 있다 (§5.4)."""
    idx = trading_days("2003-01-02", "2024-08-29")
    eps = {k: 0.30 for k in range(1, len(BLOCKS_24) + 1)}
    frame = _frame(idx, BLOCKS_24, eps)
    with pytest.warns(RuntimeWarning, match="채택은 신선 자료로만"):
        res = E.admission_test(frame, "H", ("p2",), BLOCKS_24, n_boot=500)
    assert res["evidence"] == "observed" and res["admissible"] is False
    assert res["wins"] == 11 >= res["need"] and res["no_harm"] is True
    assert res["verdict_raw"] == "ADMIT" and res["verdict"] == "SHADOW"
    assert res["reason"] == "observed_record_cannot_admit"
    assert pd.Timestamp(res["last_session"]) < HOLDOUT


def test_admission_test_refuses_fewer_than_11_fresh_blocks():
    """§5.4 사전 등록 최소치: 완결 신선 블록 < 11 이면 통계가 통과해도 채택 불가(SHADOW + 사유)."""
    idx = pd.DatetimeIndex(pd.bdate_range(FRESH_START, "2027-12-31"), name="date")
    blocks = E.fresh_blocks(idx, FRESH_START)
    n = len(blocks)
    assert 0 < n < ENSEMBLE_P3["fresh_blocks_min"]
    frame = _frame(idx, blocks, {k: 0.30 for k in range(1, n + 1)})
    with pytest.warns(RuntimeWarning, match="완결 신선 블록"):
        res = E.admission_test(frame, "H", ("p2",), blocks, live_start=FRESH_START, n_boot=500)
    assert res["evidence"] == "fresh" and res["n_blocks"] == n
    assert res["wins"] == n >= res["need"] and res["no_harm"] is True
    assert res["verdict_raw"] == "ADMIT" and res["verdict"] == "SHADOW"
    assert res["admissible"] is False
    assert res["reason"] == f"insufficient_fresh_blocks_{n}/{ENSEMBLE_P3['fresh_blocks_min']}"
    assert res["min_blocks"] == ENSEMBLE_P3["fresh_blocks_min"]
    assert E.admission_due(n)["due"] is False                  # 채택 재검 시점 함수와 같은 답
    with pytest.raises(ValueError, match="admissible=False"):  # 자연 경로: 채택 반영도 막힌다
        E.apply_admission({}, res, "#8a", "2030-01-02")
    with pytest.raises(ValueError, match="완결 신선 블록"):     # 손으로 verdict 를 고쳐도 막힌다
        E.apply_admission({}, {**res, "verdict": "ADMIT", "admissible": True}, "#8a", "2030-01-02")


def test_admission_test_refuses_holdout_rows():
    idx = trading_days("2003-01-02", "2024-12-31")
    blocks = list(BLOCKS_24) + [("2024-09-01", "2025-01-01")]
    eps = {k: 0.30 for k in range(1, len(blocks) + 1)}
    frame = _frame(idx, blocks, eps)
    with pytest.raises(ValueError, match="홀드아웃 행"):
        E.admission_test(frame, "H", ("p2",), blocks, n_boot=200)


def test_admission_test_refuses_pre_live_blocks_and_bad_live_start():
    idx = _fresh_index()
    blocks = E.fresh_blocks(idx, FRESH_START)
    frame = _frame(idx, blocks, {k: 0.30 for k in range(1, 12)})
    with pytest.raises(ValueError, match="홀드아웃 시작"):
        E.admission_test(frame, "H", ("p2",), blocks, live_start=HOLDOUT_START, n_boot=200)
    stale = [("2024-01-01", "2024-04-01")] + blocks
    with pytest.raises(ValueError, match="이전에서 시작하는 블록"):
        E.admission_test(frame, "H", ("p2",), stale, live_start=FRESH_START, n_boot=200)


def test_admission_test_input_guards():
    idx = _fresh_index()
    blocks = E.fresh_blocks(idx, FRESH_START)
    frame = _frame(idx, blocks, {k: 0.30 for k in range(1, 12)})
    with pytest.raises(ValueError, match="등록부에 없는 후보"):
        E.admission_test(frame, "ZZ", ("p2",), blocks, live_start=FRESH_START, n_boot=200)
    with pytest.raises(ValueError, match="이미 admitted"):
        E.admission_test(frame, "p2", ("p2",), blocks, live_start=FRESH_START, n_boot=200)
    with pytest.raises(ValueError, match="라벨 열"):
        E.admission_test(frame.drop(columns=["y"]), "H", ("p2",), blocks, live_start=FRESH_START, n_boot=200)


def test_admission_sequence_is_fixed_order_greedy():
    idx = _fresh_index()
    blocks = E.fresh_blocks(idx, FRESH_START)
    frame = _frame(idx, blocks, {k: (0.30 if k <= 8 else -0.001) for k in range(1, 12)}, seed=2)
    frame["M1"] = 0.5                                   # M1 은 p2 와 동일 → 개선 없음
    seq = E.admission_sequence(frame, blocks, live_start=FRESH_START, n_boot=500)
    assert seq["order"] == ENSEMBLE_P3["member_order"] == ("H", "M1")
    assert seq["admitted"] == ("p2", "H")               # H 먼저 통과, M1 은 통과 못 함
    assert seq["results"]["H"]["verdict"] == "ADMIT"
    assert seq["results"]["M1"]["verdict"] == "SHADOW"
    assert seq["results"]["M1"]["admitted"] == ("p2", "H")   # 탐욕적 전진: H 가 들어간 뒤에 검정
    with pytest.raises(ValueError, match="재배열 금지"):
        E.admission_sequence(frame, blocks, order=("M1", "H"), live_start=FRESH_START, n_boot=200)


def test_admission_sequence_skips_rejected_members():
    idx = _fresh_index()
    blocks = E.fresh_blocks(idx, FRESH_START)
    frame = _frame(idx, blocks, {k: 0.30 for k in range(1, 12)})
    frame["M1"] = 0.5
    st = {**E.day1_statuses(), "H": "candidate_rejected"}
    seq = E.admission_sequence(frame, blocks, statuses=st, live_start=FRESH_START, n_boot=300)
    assert seq["skipped"]["H"] == "candidate_rejected" and "H" not in seq["results"]
    assert seq["admitted"] == ("p2",)


def test_admission_due_cadence():
    assert E.admission_due(10)["due"] is False and E.admission_due(10)["next_at"] == 11
    first = E.admission_due(11)
    assert first["due"] is True and first["next_at"] == 15 and first["reason"] == "최초 검정"
    assert E.admission_due(14, 11)["due"] is False      # 4블록(12개월) 전에는 재검 없음
    assert E.admission_due(15, 11)["due"] is True
    assert ENSEMBLE_P3["reeval_every_blocks"] == 4
    with pytest.raises(ValueError):
        E.admission_due(15, 3)


def test_admission_power_table_matches_prereg_values():
    tbl = E.admission_power_table()
    assert list(tbl["q"]) == [0.5, 0.6, 0.7, 0.8, 0.9]
    got = dict(zip(tbl["q"], tbl["p_at_least"]))
    assert got[0.5] == pytest.approx(0.113, abs=5e-4)
    assert got[0.6] == pytest.approx(0.296, abs=5e-4)
    assert got[0.7] == pytest.approx(0.570, abs=5e-4)
    assert got[0.8] == pytest.approx(0.839, abs=5e-4)
    assert (tbl["need"] == 8).all() and (tbl["n_blocks"] == 11).all()
    assert E.admission_power_table(n_blocks=4, need=4, qs=(0.5,))["p_at_least"].iloc[0] == pytest.approx(0.0625)


# ==================================================================
# 5.4 채택 반영 — 발효는 다음 1월 재적합
# ==================================================================
def _admit_result():
    idx = _fresh_index()
    blocks = E.fresh_blocks(idx, FRESH_START)
    eps = {k: (0.30 if k <= 8 else -0.001) for k in range(1, 12)}
    return E.admission_test(_frame(idx, blocks, eps), "H", ("p2",), blocks, live_start=FRESH_START, n_boot=500)


def test_apply_admission_sets_status_with_january_effective_date():
    res = _admit_result()
    assert res["verdict"] == "ADMIT"
    eff = E.next_january(res["last_session"])
    assert eff == "2030-01-02"
    m3 = E.apply_admission({"schema_version": 3}, res, "#8a", eff)
    assert m3["members"]["H"]["status"] == "admitted"
    assert m3["members"]["H"]["effective_refit"] == eff
    assert m3["members"]["H"]["ledger_no"] == "#8a"
    assert m3["members"]["p2"]["status"] == "seed" and m3["members"]["M1"]["status"] == "shadow"
    assert m3["mode_history"][-1]["event"] == "admission"
    assert m3["registry_sha"] == E.registry_sha256()


def test_apply_admission_effective_date_gates_the_average():
    """상태는 바뀌지만 다음 1월 전까지는 평균에 들어가지 않는다(연중 변경 금지)."""
    res = _admit_result()
    m3 = E.apply_admission({}, res, "#8a", "2030-01-02")
    before = E.effective_statuses(m3, "2029-12-31")
    after = E.effective_statuses(m3, "2030-01-02")
    assert before["H"] == "shadow" and after["H"] == "admitted"
    probs = _probs()
    assert np.array_equal(E.combine(probs, before).to_numpy(), probs["p2"].to_numpy(), equal_nan=True)
    assert E.production_members(after) == ("p2", "H")


def test_apply_admission_refuses_observed_evidence():
    idx = trading_days("2003-01-02", "2024-08-29")
    eps = {k: 0.30 for k in range(1, len(BLOCKS_24) + 1)}
    with pytest.warns(RuntimeWarning):
        res = E.admission_test(_frame(idx, BLOCKS_24, eps), "H", ("p2",), BLOCKS_24, n_boot=500)
    with pytest.raises(ValueError, match="신선 자료가 아닙니다"):
        E.apply_admission({}, res, "#8a", "2025-01-02")


def test_apply_admission_rechecks_fresh_block_minimum():
    """손으로 만든 result 로도 최소치를 우회할 수 없다(이중 방어)."""
    res = _admit_result()
    assert res["n_blocks"] == ENSEMBLE_P3["fresh_blocks_min"] == 11
    with pytest.raises(ValueError, match="완결 신선 블록 2/11"):
        E.apply_admission({}, {**res, "n_blocks": 2}, "#8a", "2030-01-02")
    m3 = E.apply_admission({}, res, "#8a", "2030-01-02")        # 11개면 통과
    assert m3["members"]["H"]["admission"]["n_blocks"] == 11


def test_apply_admission_validates_effective_refit():
    res = _admit_result()
    with pytest.raises(ValueError, match="1월이 아닙니다"):
        E.apply_admission({}, res, "#8a", "2030-03-01")
    with pytest.raises(ValueError, match="이후여야"):
        E.apply_admission({}, res, "#8a", "2029-01-02")
    with pytest.raises(ValueError, match="장부 항목"):
        E.apply_admission({}, res, "  ", "2030-01-02")


def test_effective_statuses_requires_effective_refit():
    with pytest.raises(ValueError, match="effective_refit"):
        E.effective_statuses({"members": {"H": {"status": "admitted"}}}, "2030-01-02")
    with pytest.raises(ValueError, match="등록부 밖"):
        E.effective_statuses({"members": {"ZZ": {"status": "shadow"}}}, "2030-01-02")
    assert E.effective_statuses(None, "2027-01-04") == E.day1_statuses()


def test_next_january_with_and_without_index():
    idx = trading_days("2029-01-01", "2030-12-31")           # 캘린더 검증 범위(≤2030) 안
    assert E.next_january("2029-06-29", idx) == "2030-01-02"
    assert E.next_january("2029-06-29") == "2030-01-02"      # 인덱스 없이 달력으로도 같은 날


# ==================================================================
# 생산 확률 보호 가드 — 앙상블 산출은 어디로도 새지 않는다
# ==================================================================
PRODUCTION_MODULES = ("model.py", "features.py", "vol.py", "decision.py")


def test_production_modules_do_not_import_ensemble():
    """생산 확률 경로(특징 → 4-파라미터 로지스틱 → 결정층)는 ensemble 을 모른다."""
    for name in PRODUCTION_MODULES:
        src = (ROOT / "mrl" / name).read_text(encoding="utf-8")
        assert "ensemble" not in src, f"{name} 이 ensemble 을 참조합니다 — 그림자가 생산 경로로 샜습니다"


def test_ensemble_cannot_write_any_artifact():
    """ensemble.py 는 생산 모델 파일 경로·저장 함수를 모르고, 어떤 파일에도 쓰지 않는다."""
    src = (ROOT / "mrl" / "ensemble.py").read_text(encoding="utf-8")
    body = "\n".join(src.split('"""', 2)[2].splitlines())          # 모듈 docstring 제외한 본문
    for token in ("MODEL_P2_PATH", "save_model", "model_p2.json", "json.dump", "to_csv", "write_text",
                  "open(", "mkdir"):
        assert token not in body, f"ensemble.py 본문에 쓰기 수단 {token!r} 이 있습니다"


def test_shadow_only_guard_catches_unapproved_admission():
    st = E.day1_statuses()
    E.assert_shadow_only(st)                                        # 1일차는 통과
    with pytest.raises(ValueError, match="그림자 전용 위반"):
        E.assert_shadow_only({**st, "H": "admitted"}, where="daily")
    with pytest.raises(ValueError, match="그림자 전용 위반"):
        E.assert_shadow_only({**st, "M1": "admitted"})
    assert E.production_param_count({**st, "H": "admitted"}) == 6    # 채택되면 K_s 4 → 6 (§5.4)


def test_phase3_adds_zero_fitted_parameters_to_production():
    assert M.PARAM_COUNT == 4 and M.BUDGET == 5
    assert E.production_param_count(E.day1_statuses()) == M.PARAM_COUNT
    assert E.shadow_param_counts()["phase3_added_to_production"] == 0


def test_registry_table_marks_shadow_members():
    idx = pd.DatetimeIndex(pd.bdate_range("2027-01-04", periods=200), name="date")
    rng = np.random.default_rng(1)
    y = (rng.random(len(idx)) < 0.15).astype(float)
    oos = pd.DataFrame({"y": y, "clim": 0.15, "p2": 0.15, "M1": 0.15,
                        "H": np.clip(0.15 + 0.1 * (2 * y - 1), 0, 1)}, index=idx)
    rows = E.registry_table(oos, E.day1_statuses())
    assert [r["member"] for r in rows] == ["p2", "M1", "H"]
    assert rows[0]["in_average"] is True and rows[1]["in_average"] is False and rows[2]["in_average"] is False
    assert rows[2]["bss_clim"] > 0 and rows[2]["auc"] > 0.5
    assert [r["K_s"] for r in rows] == [4, 2, 2] and [r["K_u"] for r in rows] == [0, 0, 12]


def test_member_h_walk_forward_requires_regime_module():
    """regime 은 지연 import — 없으면 등록부·결합·채택은 살고 H 경로만 명확히 실패한다."""
    if (ROOT / "mrl" / "regime.py").exists():
        pytest.skip("mrl/regime.py 가 있으므로 지연 import 실패 경로는 검사하지 않는다")
    with pytest.raises(ImportError, match="regime"):
        E.hmm_x_walk_forward(pd.Series([1.0, 2.0], index=pd.bdate_range("2027-01-04", periods=2)), [])


# ==================================================================
# walk-forward vs 오프라인 parity (regime + 실캐시가 있을 때만)
# ==================================================================
def _real_close(end: str):
    from mrl.data import load_cache                          # noqa: WPS433 - 캐시 없는 환경에서 skip
    try:
        b = load_cache()
    except FileNotFoundError:                                 # pragma: no cover - 캐시 없는 환경
        pytest.skip("데이터 캐시가 없습니다 (scripts/build_cache.py)")
    s = b.close["SPY"].dropna()
    return s[s.index < pd.Timestamp(end)]


def test_platt_training_feature_is_theta_y_refiltered():
    """§5.2/§4.1: R_y 의 Platt 학습 특징 = θ_y 를 학습 구간에 전방 필터한 값(walk-forward 모자이크가 아니다)."""
    regime = pytest.importorskip("mrl.regime", reason="mrl/regime.py 동시 작성 중")
    close = _real_close("2009-01-01")
    assert close.index[-1] < HOLDOUT
    from mrl.targets import make_targets                      # noqa: WPS433
    y = make_targets(close)["y_dd5_20"]
    rds = C.refit_dates(close.index, first="2003-01-02", end="2009-01-01")
    df, models, thetas = E.member_h_walk_forward(close, y, rds)
    assert df.attrs["train_feats"] is True and len(models) == len(thetas) == len(rds)

    obs = regime.observations(close)
    clip = float(HMM_P3["clip"])
    for m, R, th in zip(models, rds, thetas):
        assert th.refit_date == str(pd.Timestamp(R).date())
        v = regime.filter_probabilities(obs, th).to_numpy(dtype=float)
        ok = np.isfinite(v)
        xv = np.full(len(v), np.nan)
        xv[ok] = regime.logit(np.clip(v[ok], clip, 1.0 - clip))
        x = pd.Series(xv, index=obs.index)
        ref = C.fit_refit(x.to_frame(E.PLATT_FEATURE), y.reindex(obs.index), R, (E.PLATT_FEATURE,),
                          C=float(HMM_P3["platt_C"]), purge=int(HMM_P3["purge"]),
                          train_start=HMM_P3["train_start"], rung="H")
        assert m.intercept == pytest.approx(ref.intercept, abs=1e-10)
        assert m.coef[E.PLATT_FEATURE] == pytest.approx(ref.coef[E.PLATT_FEATURE], abs=1e-10)

    # 이 테스트가 이빨을 가지는지: 모자이크로 적합하면 2003 이후 재적합에서 계수가 달라진다
    wf, _ = E.hmm_x_walk_forward(close, rds)
    mosaic_df, mosaic = E.platt_walk_forward(wf["x_hmm"], y, rds)     # train_feats 미지정 = 이탈 경로
    assert mosaic_df.attrs["train_feats"] is False
    assert any("§5.2 이탈" in w for w in mosaic_df.attrs["warnings"])
    assert any(abs(a.intercept - b.intercept) > 1e-4 for a, b in zip(models[1:], mosaic[1:]))


def test_walk_forward_matches_offline_recompute():
    """OOS 무작위 10일 t: 그 해 θ·Platt 로 `observations(close[:t])` 를 필터·예측한 값과 1e-12 안에서 같다.
    (§13 test_ensemble parity; 홀드아웃은 입력 하드컷으로 접근하지 않는다)"""
    regime = pytest.importorskip("mrl.regime", reason="mrl/regime.py 동시 작성 중")
    close = _real_close("2009-01-01")
    assert close.index[-1] < HOLDOUT
    from mrl.targets import make_targets                      # noqa: WPS433
    y = make_targets(close)["y_dd5_20"]
    rds = C.refit_dates(close.index, first="2003-01-02", end="2009-01-01")
    df, models, thetas = E.member_h_walk_forward(close, y, rds)
    by_year = {pd.Timestamp(t.refit_date).year: t for t in thetas}
    platt = {pd.Timestamp(m.refit_date).year: m for m in models}

    oos = df.index[(df.index >= rds[0]) & df["p_h"].notna() & ~df["in_sample"].to_numpy()]
    picks = pd.DatetimeIndex(np.random.default_rng(0).choice(oos.to_numpy(), size=10, replace=False)).sort_values()
    for t in picks:
        year = int(df.loc[t, "refit_year"])
        obs_t = regime.observations(close[close.index <= t])
        p_high = float(regime.filter_probabilities(obs_t, by_year[year]).iloc[-1])
        assert p_high == pytest.approx(float(df.loc[t, "p_high"]), abs=1e-12)
        clip = float(HMM_P3["clip"])
        x = float(np.log(np.clip(p_high, clip, 1 - clip)) - np.log1p(-np.clip(p_high, clip, 1 - clip)))
        p_h = platt[year].predict_one(pd.Series({E.PLATT_FEATURE: x}, name=t))
        assert p_h == pytest.approx(float(df.loc[t, "p_h"]), abs=1e-12)
