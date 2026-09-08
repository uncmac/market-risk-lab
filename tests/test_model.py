# -*- coding: utf-8 -*-
"""mrl/model.py 검증 (ARCHITECTURE_PHASE2.md §7·§15).

실행: 프로젝트 루트에서  python -m pytest tests/test_model.py -q
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from mrl import model as M
from mrl.config import P2


# ------------------------------------------------------------------
# 합성 자료
# ------------------------------------------------------------------
TRUE_B0 = -1.0
TRUE_B = {"x_vix": 0.9, "x_har": 0.6, "x_ma": -1.2}


def _synth(n: int = 20_000, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1990-01-02", periods=n)
    X = pd.DataFrame({"x_vix": rng.normal(-1.0, 1.0, n), "x_har": rng.normal(0.0, 1.0, n), "x_ma": rng.normal(0.0, 1.0, n)},
                     index=idx)
    z = TRUE_B0 + sum(TRUE_B[k] * X[k].to_numpy() for k in TRUE_B)
    p = 1.0 / (1.0 + np.exp(-z))
    y = pd.Series((rng.uniform(size=n) < p).astype(float), index=idx)
    return X, y


def _model(coef: dict, b0: float, refit_date: str = "2024-01-02", model_id: str | None = None) -> M.LogitModel:
    feats = tuple(coef)
    return M.LogitModel(features=feats, coef=dict(coef), intercept=b0, refit_date=refit_date, train_start="1993-10-14",
                        train_end="2023-12-01", n_train=7500, n_pos=1200, clim=0.17, feature_rule="test", spec_sha256="0" * 64,
                        model_id=model_id or f"p2m3-00000000-{refit_date}")


@pytest.fixture(scope="module")
def synth():
    return _synth()


@pytest.fixture(scope="module")
def fitted(synth) -> M.LogitModel:
    X, y = synth
    return M.fit_logit(X, y, M.LADDER["M3"], meta={"refit_date": "2003-01-02", "clim": float(y.mean())})


# ------------------------------------------------------------------
# 상수·예산
# ------------------------------------------------------------------
def test_param_count_and_ladder():
    assert M.PARAM_COUNT == 4 == P2["param_count"]
    assert M.BUDGET == 5 == P2["budget"]
    assert M.n_params("M3") == 4
    assert M.n_params("M0") == 0 and M.n_params("M1") == 2 and M.n_params("M2") == 3
    assert M.LADDER["M3"] == tuple(P2["features"]) == ("x_vix", "x_har", "x_ma")
    assert list(M.LADDER) == ["M0", "M1", "M2", "M3"]
    assert set(M.ABLATIONS) == {"M3-PK", "M3-HAR96", "R-v0"}
    assert M.FIT_KWARGS["C"] == 1.0 and M.FIT_KWARGS["penalty"] == "l2" and M.FIT_KWARGS["solver"] == "lbfgs"
    assert M.FIT_KWARGS["max_iter"] == 1000 and M.FIT_KWARGS["tol"] == 1e-8
    with pytest.raises(KeyError):
        M.n_params("M9")


def test_fitted_model_has_exactly_four_params(fitted):
    assert fitted.n_params() == M.PARAM_COUNT == 4
    assert M.n_params(fitted) == 4
    assert M.rung_of(fitted.features) == "M3"
    assert fitted.model_id.startswith("p2m3-") and fitted.model_id.endswith("-2003-01-02")
    assert len(fitted.model_id.split("-")[1]) == 8
    assert fitted.n_train == 20_000 and 0 < fitted.n_pos < fitted.n_train
    assert fitted.train_start == "1990-01-02"


# ------------------------------------------------------------------
# 적합
# ------------------------------------------------------------------
def test_synthetic_coefficient_recovery(fitted):
    assert abs(fitted.intercept - TRUE_B0) < 0.05
    for k, v in TRUE_B.items():
        assert abs(fitted.coef[k] - v) < 0.05, (k, fitted.coef[k], v)


def test_fit_uses_only_finite_rows_and_rejects_few_positives(synth):
    X, y = synth
    X2 = X.copy()
    X2.iloc[:100, 0] = np.nan          # 특징 NaN 100행 제거
    y2 = y.copy()
    y2.iloc[100:150] = np.nan          # 라벨 NaN 50행 제거
    m = M.fit_logit(X2, y2, M.LADDER["M3"], meta={"refit_date": "2003-01-02"})
    assert m.n_train == len(X) - 150
    # 양성 < 20 → ValueError
    y3 = y.copy()
    y3[:] = 0.0
    y3.iloc[:19] = 1.0
    with pytest.raises(ValueError):
        M.fit_logit(X, y3, M.LADDER["M3"], meta={"refit_date": "2003-01-02"})
    # 특징 없음(M0)·예산 초과·열 누락
    with pytest.raises(ValueError):
        M.fit_logit(X, y, (), meta={})
    with pytest.raises(ValueError):
        M.fit_logit(X, y, ("x_vix", "x_har", "x_ma", "x_vix"), meta={})
    with pytest.raises(ValueError):
        M.fit_logit(X, y, ("x_vix", "nope"), meta={})
    # y 가 {0,1} 이 아니면 거부
    with pytest.raises(ValueError):
        M.fit_logit(X, y * 2.0, M.LADDER["M3"], meta={})


def test_fit_ladder_rungs_param_counts(synth):
    X, y = synth
    for rung, fs in M.LADDER.items():
        if not fs:
            continue
        m = M.fit_logit(X, y, fs, meta={"refit_date": "2003-01-02"})
        assert m.n_params() == M.n_params(rung) == len(fs) + 1
        assert m.model_id.startswith(f"p2{rung.lower()}-")


def test_fit_is_deterministic(synth):
    X, y = synth
    a = M.fit_logit(X, y, M.LADDER["M3"], meta={"refit_date": "2003-01-02"})
    b = M.fit_logit(X, y, M.LADDER["M3"], meta={"refit_date": "2003-01-02"})
    assert a.coef == b.coef and a.intercept == b.intercept


# ------------------------------------------------------------------
# 중첩: (b0,b1)=(0,1), (b2,b3)=(0,0) → p_vix 재현
# ------------------------------------------------------------------
def test_identity_model_reproduces_p_vix():
    m = _model({"x_vix": 1.0, "x_har": 0.0, "x_ma": 0.0}, 0.0)
    rng = np.random.default_rng(1)
    p_vix = rng.uniform(0.05, 0.8, 500)
    X = pd.DataFrame({"x_vix": np.log(p_vix / (1 - p_vix)), "x_har": rng.normal(size=500), "x_ma": rng.normal(size=500)})
    p = m.predict_proba(X)
    np.testing.assert_allclose(p.to_numpy(), p_vix, rtol=0, atol=1e-12)
    # 한 행 Series 도 지원
    assert abs(m.predict_one(X.iloc[0]) - p_vix[0]) < 1e-12


def test_predict_nan_rows_stay_nan(fitted):
    X = pd.DataFrame({"x_vix": [0.0, np.nan, 1.0], "x_har": [0.0, 0.0, np.nan], "x_ma": [0.0, 0.0, 0.0]})
    p = fitted.predict_proba(X)
    assert math.isfinite(p.iloc[0]) and math.isnan(p.iloc[1]) and math.isnan(p.iloc[2])
    with pytest.raises(ValueError):
        fitted.predict_proba(X[["x_vix", "x_har"]])


# ------------------------------------------------------------------
# 저장 / 적재
# ------------------------------------------------------------------
def test_save_load_roundtrip(tmp_path, fitted):
    path = tmp_path / "model_p2.json"
    M.save_model(fitted, path, deploy_mode="tones", tone_model="M1")
    raw = json.loads(path.read_text(encoding="utf-8"))
    for k in ("schema_version", "model_id", "features", "coef", "intercept", "clim", "refit_date", "train_start", "train_end",
              "n_train", "n_pos", "feature_rule", "spec_sha256", "deploy_mode", "tone_model", "acceptance_ref", "created_at_utc"):
        assert k in raw, k
    assert raw["schema_version"] == 1 and raw["deploy_mode"] == "tones" and raw["tone_model"] == "M1"
    assert raw["acceptance_ref"] == "summary_p2.json:acceptance"
    assert path.read_bytes().count(b"\r\n") == 0                       # LF
    m2 = M.load_model(path)
    assert m2.features == fitted.features and m2.coef == fitted.coef and m2.intercept == fitted.intercept
    assert m2.clim == fitted.clim and m2.model_id == fitted.model_id and m2.spec_sha256 == fitted.spec_sha256
    assert m2.n_train == fitted.n_train and m2.n_pos == fitted.n_pos and m2.refit_date == fitted.refit_date
    assert m2.deploy_mode == "tones" and m2.tone_model == "M1"
    # 계수는 비트 동일(허용오차 1e-9 결정론 검사의 전제)
    X = pd.DataFrame({"x_vix": [-1.2], "x_har": [0.3], "x_ma": [0.02]})
    assert fitted.predict_proba(X).iloc[0] == m2.predict_proba(X).iloc[0]
    # 잘못된 schema_version 거부
    raw["schema_version"] = 2
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        M.load_model(path)
    with pytest.raises(FileNotFoundError):
        M.load_model(tmp_path / "missing.json")


def test_save_refuses_nan():
    m = _model({"x_vix": 1.0, "x_har": 0.0, "x_ma": 0.0}, 0.0)
    m.clim = float("nan")
    with pytest.raises(ValueError):
        M.save_model(m, "C:/nonexistent-should-not-be-written/model.json")


# ------------------------------------------------------------------
# 귀속
# ------------------------------------------------------------------
def test_contributions_exact_sums(fitted):
    rng = np.random.default_rng(3)
    for _ in range(50):
        x = pd.Series({"x_vix": rng.normal(-1, 1), "x_har": rng.normal(0, 0.5), "x_ma": rng.normal(0, 0.1)})
        c = M.contributions(fitted, x)
        assert list(c.index) == ["intercept", "x_vix", "x_har", "x_ma"]
        p = fitted.predict_one(x)
        lg = fitted.intercept + sum(fitted.coef[k] * x[k] for k in fitted.features)
        assert abs(c["logit"].sum() - lg) < 1e-12
        sigma_b0 = 1.0 / (1.0 + math.exp(-fitted.intercept))
        assert abs(c.loc["intercept", "pp"] - sigma_b0) < 1e-12
        assert abs(sigma_b0 + c.loc[["x_vix", "x_har", "x_ma"], "pp"].sum() - p) < 1e-12
        assert abs(c["pp"].sum() - p) < 1e-12
        assert abs(c.attrs["p"] - p) < 1e-12
    # 입력 NaN → 특징 행 NaN, 지어내지 않음
    c = M.contributions(fitted, pd.Series({"x_vix": np.nan, "x_har": 0.0, "x_ma": 0.0}))
    assert c.loc[["x_vix", "x_har", "x_ma"], "pp"].isna().all() and math.isnan(c.attrs["p"])


def test_day_over_day_exact_sum_no_refit(fitted):
    rng = np.random.default_rng(4)
    for _ in range(50):
        xp = pd.Series({"x_vix": rng.normal(-1, 1), "x_har": rng.normal(0, 0.5), "x_ma": rng.normal(0, 0.1)},
                       name=pd.Timestamp("2024-03-01"))
        xn = xp + pd.Series({"x_vix": rng.normal(0, 0.3), "x_har": rng.normal(0, 0.2), "x_ma": rng.normal(0, 0.02)})
        xn.name = pd.Timestamp("2024-03-04")
        d = M.day_over_day(fitted, xn, fitted, xp)
        assert d["refit"] is False and d["d_logit"]["refit"] == 0.0 and d["d_pp"]["refit"] == 0.0
        dp = fitted.predict_one(xn) - fitted.predict_one(xp)
        assert abs(sum(d["d_pp"].values()) - dp) < 1e-12
        assert abs(sum(d["d_logit"].values()) - (d["logit_now"] - d["logit_prev"])) < 1e-12
        assert d["gap_sessions"] == 1


def test_day_over_day_with_refit_and_gap(fitted):
    prev = _model({"x_vix": 0.8, "x_har": 0.5, "x_ma": -2.0}, -1.1, refit_date="2023-01-03", model_id="p2m3-0-2023")
    xp = pd.Series({"x_vix": -0.9, "x_har": 0.2, "x_ma": 0.03}, name=pd.Timestamp("2023-12-29"))
    xn = pd.Series({"x_vix": -0.7, "x_har": 0.1, "x_ma": 0.02}, name=pd.Timestamp("2024-01-02"))
    d = M.day_over_day(fitted, xn, prev, xp)
    assert d["refit"] is True and d["d_logit"]["refit"] != 0.0
    dp = fitted.predict_one(xn) - prev.predict_one(xp)
    assert abs(d["d_p"] - dp) < 1e-15
    assert abs(sum(d["d_pp"].values()) - dp) < 1e-12
    assert abs(sum(d["d_logit"].values()) - (d["logit_now"] - d["logit_prev"])) < 1e-12
    # 재적합 항 정의: (b0n−b0p) + Σ (bkn−bkp)·xk_prev
    expect = (fitted.intercept - prev.intercept) + sum((fitted.coef[k] - prev.coef[k]) * xp[k] for k in fitted.features)
    assert abs(d["d_logit"]["refit"] - expect) < 1e-12
    assert d["gap_sessions"] == 1                     # 2023-12-29(금) → 2024-01-02(화): 1/1 휴장
    # 5세션 간격: (12/29, 1/8] = 1/2·1/3·1/4·1/5·1/8
    xn5 = xn.copy()
    xn5.name = pd.Timestamp("2024-01-08")
    assert M.day_over_day(fitted, xn5, prev, xp)["gap_sessions"] == 5
    assert M.day_over_day(fitted, xn5, prev, xp, gap_sessions=7)["gap_sessions"] == 7


def test_day_over_day_zero_delta_logit_boundary(fitted):
    x = pd.Series({"x_vix": -1.0, "x_har": 0.1, "x_ma": 0.0})
    d = M.day_over_day(fitted, x, fitted, x.copy())
    assert all(v == 0.0 for v in d["d_logit"].values()) and all(v == 0.0 for v in d["d_pp"].values())
    assert d["d_p"] == 0.0
    # Δlogit 이 1e-12 보다 작지만 0 이 아닌 경우도 잔차 0 (합 == Δp)
    x2 = x.copy()
    x2["x_har"] += 1e-13
    d2 = M.day_over_day(fitted, x2, fitted, x)
    assert abs(sum(d2["d_pp"].values()) - d2["d_p"]) < 1e-15
    # 재적합으로 Δlogit 이 정확히 상쇄되는 경우: refit 항 == −Σ 특징 항, pp 합 == Δp
    prev = _model(dict(fitted.coef), fitted.intercept + 0.05, refit_date="2023-01-03", model_id="prev")
    xp = x.copy()
    xn = x.copy()
    xn["x_vix"] += 0.05 / fitted.coef["x_vix"]       # b1·Δx_vix = +0.05 가 절편 −0.05 를 상쇄
    d3 = M.day_over_day(fitted, xn, prev, xp)
    assert abs(d3["logit_now"] - d3["logit_prev"]) < 1e-12
    assert abs(sum(d3["d_pp"].values()) - d3["d_p"]) < 1e-15
    assert abs(sum(d3["d_logit"].values())) < 1e-12
    # 입력 NaN → 수치 NaN (refit 플래그는 유지)
    xnan = x.copy()
    xnan["x_ma"] = np.nan
    d4 = M.day_over_day(fitted, xnan, prev, xp)
    assert math.isnan(d4["d_p"]) and d4["refit"] is True
    # 특징 집합이 다르면 거부
    with pytest.raises(ValueError):
        M.day_over_day(fitted, x, _model({"x_vix": 1.0}, 0.0), x)


# ------------------------------------------------------------------
# 파라미터 밴드
# ------------------------------------------------------------------
def test_parameter_band_contains_current_p(fitted):
    rng = np.random.default_rng(5)
    models = [fitted]
    for k in range(4):
        coef = {f: fitted.coef[f] + rng.normal(0, 0.05) for f in fitted.features}
        models.append(_model(coef, fitted.intercept + rng.normal(0, 0.05), refit_date=f"20{20 + k}-01-02"))
    for _ in range(20):
        x = pd.Series({"x_vix": rng.normal(-1, 1), "x_har": rng.normal(0, 0.5), "x_ma": rng.normal(0, 0.1)})
        lo, hi = M.parameter_band(models, x)
        p = fitted.predict_one(x)
        assert lo <= p <= hi and lo <= hi
    # 상한은 '재적합 5 + 라이브 1' = 6개: 6개는 전부 쓴다(맨 앞의 극단 모델도 밴드에 들어간다)
    extreme = _model({f: 0.0 for f in fitted.features}, 5.0, refit_date="2018-01-02")
    x_ext = pd.Series({"x_vix": -1, "x_har": 0, "x_ma": 0})
    lo6a, hi6a = M.parameter_band([extreme] + models, x_ext)
    assert hi6a > 0.99
    lo6, hi6 = M.parameter_band(models + [extreme], x_ext)
    assert hi6 > 0.99
    # 7개를 주면 최근 6개만 (맨 앞의 극단 모델은 잘려 나간다)
    lo7, hi7 = M.parameter_band([extreme] + models + [models[0]], x_ext)
    assert hi7 < 0.99
    with pytest.raises(ValueError):
        M.parameter_band([], pd.Series({"x_vix": -1, "x_har": 0, "x_ma": 0}))
    lo, hi = M.parameter_band(models, pd.Series({"x_vix": np.nan, "x_har": 0, "x_ma": 0}))
    assert math.isnan(lo) and math.isnan(hi)


def test_spec_sha256_available_without_features_module():
    s = M.spec_sha256()
    assert isinstance(s, str) and len(s) == 64
    assert M.spec_sha256() == s
