# -*- coding: utf-8 -*-
"""장부 Phase 2 확장(mrl/ledger.py — ARCHITECTURE_PHASE2.md §12, schema_version=2) 테스트.

* P2 열이 LEDGER_COLUMNS_V1 뒤에 계약 순서대로 붙고, 기존 열 순서는 불변.
* 2026-09-08 정정: prob_dd5_20 = **배포 확률**(acceptance 가 배치한 단; info_only 면 M3 정보)이고, 사다리 M3 는 새 열
  p2_p_m3, 그 확률을 만든 모델은 p2_prob_model_id 에 기록한다(둘 다 맨 뒤에 덧붙임 — 앞 열 순서 불변, schema_version 2 유지).
* append_today 가 평면 p2_* 키·중첩 dict row["p2"]·별칭(p2_prob→prob_dd5_20, har_vol_fcst→p2_har_fc_20)을 같은 열로 정규화.
* 구 행(Phase 1 append)·구 파일(V1 열만)의 P2 열은 NaN 으로 보존되고 알 수 없는 추가 열은 뒤에 남는다.
* 값 검사: 상태·확률 범위는 ValueError, 확률 NaN 인데 사유 없음·알 수 없는 p2_* 키는 경고(조용한 실패 금지).
* backfill 뒤 summary()["p2"] 가 라이브 Brier·skill(기후학·M1·B1·BGK)·n·CI·킬룰 카운트다운을 낸다 — 손계산과 대조.
임시 디렉터리만 사용한다(results/ 의 실제 장부는 건드리지 않음).
"""
from __future__ import annotations

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

from mrl import ledger                                            # noqa: E402
from mrl.config import P2, P2_STATES, V0_SIGNALS                  # noqa: E402

CONTRACT_P2 = ["p2_p_m1", "p2_p_m2", "p2_p_vix", "p2_p_vix_bgk", "p2_clim", "p2_lo", "p2_hi", "p2_band_src",
               "p2_x_vix", "p2_x_har", "p2_x_ma", "p2_har_vol_20", "p2_har_fc_20", "p2_r", "p2_state",
               "p2_days_in_state", "p2_tone_model", "p2_deploy_mode", "p2_d_vix", "p2_d_har", "p2_d_ma", "p2_d_refit",
               "p2_input_missing", "p2_model_id"]
ADDED_P2 = ["p2_p_m3", "p2_prob_model_id"]      # 2026-09-08 덧붙임 — 반드시 뒤에, 순서 이 그대로
V1_COLUMNS = (["asof", "recorded_at_utc", "variant"] + [f"state_{k}" for k in V0_SIGNALS]
              + ["score_d", "score_w", "score_m", "overall_d", "overall_w", "overall_m", "tone", "spy_close", "vix_close", "prob_dd5_20", "run_id",
                 "y_sign_20", "y_sign_60", "y_dd5_20", "fwd_ret_20", "fwd_ret_60"])


def _row(asof, tone="hold", variant="completed", **kw) -> dict:
    r = {"asof": asof, "variant": variant, "tone": tone, "states_d": {k: "GREEN" for k in V0_SIGNALS},
         "score_d": 0.3, "score_w": 0.1, "score_m": 0.5, "overall_d": "GREEN", "overall_w": "AMBER", "overall_m": "GREEN",
         "spy_close": 512.34, "vix_close": 15.2, "run_id": "test"}
    r.update(kw)
    return r


def _unlock(tmp_path) -> Path:
    """장부 옆에 홀드아웃 해제 스텁을 둔다 — 이 파일이 없으면 HOLDOUT_START 이후 행은 채점되지 않는다(VALIDATION.md §6).
    이 파일의 픽스처 날짜(2026-01-05~)는 전부 홀드아웃 구간 안이라, 채점을 검사하는 테스트는 해제 상태를 가정한다."""
    f = Path(tmp_path) / "holdout_unlock.json"
    f.write_text('{"ledger_entry": "2b", "note": "test stub"}', encoding="utf-8")
    return f


def _p2_flat(p=0.21, state="caution", **kw) -> dict:
    d = {"prob_dd5_20": p, "p2_p_m1": 0.19, "p2_p_m2": 0.205, "p2_p_vix": 0.262, "p2_p_vix_bgk": 0.211, "p2_clim": 0.158,
         "p2_lo": 0.12, "p2_hi": 0.47, "p2_band_src": "calib", "p2_x_vix": -1.035, "p2_x_har": -0.31, "p2_x_ma": 0.042,
         "p2_har_vol_20": 0.118, "p2_har_fc_20": 0.131, "p2_r": 1.33, "p2_state": state, "p2_days_in_state": 3,
         "p2_tone_model": "M3", "p2_deploy_mode": "tones", "p2_d_vix": 0.0087, "p2_d_har": 0.0061, "p2_d_ma": -0.0021, "p2_d_refit": 0.0,
         "p2_input_missing": None, "p2_model_id": "p2m3-abcd1234-2024-08-30"}
    d.update(kw)
    return d


# ------------------------------------------------------------------
# 스키마
# ------------------------------------------------------------------
def test_schema_v2_columns_append_after_v1_in_contract_order():
    assert ledger.SCHEMA_VERSION == 2                                  # 새 열은 덧붙이기만 — 3 은 Phase 3 예약
    assert ledger.LEDGER_COLUMNS_V1 == V1_COLUMNS                      # 기존 열 순서 불변
    assert ledger.P2_COLUMNS == CONTRACT_P2 + ADDED_P2                 # §12 고정 순서 + 뒤에 덧붙인 두 열
    assert ledger.LEDGER_COLUMNS == V1_COLUMNS + CONTRACT_P2 + ADDED_P2
    assert "prob_dd5_20" in ledger.LEDGER_COLUMNS_V1 and "prob_dd5_20" not in ledger.P2_COLUMNS   # 기존 예약 열 = 배포 확률
    assert set(ledger.P2_STRING_COLUMNS) <= set(CONTRACT_P2 + ADDED_P2)
    assert "p2_prob_model_id" in ledger.P2_STRING_COLUMNS and "p2_p_m3" in ledger.P2_PROB_COLUMNS
    assert ledger.P2_ALIASES["p2_prob"] == "prob_dd5_20" and ledger.P2_ALIASES["har_vol_fcst"] == "p2_har_fc_20"
    # p_m3 는 더 이상 prob_dd5_20 의 별칭이 아니다: 사다리 M3 는 제 열로 간다(배포 확률과 섞이지 않는다)
    assert "p_m3" not in ledger.P2_ALIASES and "p2_p_m3" not in ledger.P2_ALIASES
    assert ledger._p2_colname("p_m3") == "p2_p_m3" and ledger._p2_colname("p2_p_m3") == "p2_p_m3"


def test_append_flat_p2_keys_round_trip_and_types(tmp_path):
    path = tmp_path / "t.csv"
    assert ledger.append_today(_row("2026-09-04", **_p2_flat()), path) is True
    df = pd.read_csv(path)
    assert list(df.columns) == ledger.LEDGER_COLUMNS
    r = df.iloc[0]
    assert r["prob_dd5_20"] == pytest.approx(0.21) and r["p2_p_m1"] == pytest.approx(0.19) and r["p2_clim"] == pytest.approx(0.158)
    assert r["p2_state"] == "caution" and r["p2_band_src"] == "calib" and r["p2_model_id"] == "p2m3-abcd1234-2024-08-30"
    assert r["p2_days_in_state"] == 3 and r["p2_deploy_mode"] == "tones" and r["p2_tone_model"] == "M3"
    assert r["p2_d_vix"] == pytest.approx(0.0087) and r["p2_d_refit"] == 0.0
    assert pd.isna(r["p2_input_missing"]) and pd.isna(r["y_dd5_20"])
    # 다시 읽으면 문자열 열은 object, 숫자 열은 float
    back = ledger._read(path)
    for c in ledger.P2_STRING_COLUMNS:
        assert back[c].dtype == object
    for c in ledger.P2_NUMERIC_COLUMNS:
        assert pd.api.types.is_float_dtype(back[c]) or pd.api.types.is_integer_dtype(back[c]), c
    assert b"\r\n" not in path.read_bytes()


def test_append_nested_dict_and_aliases_map_to_same_columns(tmp_path):
    path = tmp_path / "t.csv"
    nested = {"p": 0.3, "p_m1": 0.25, "clim": 0.16, "state": "normal", "days_in_state": 12, "deploy_mode": "info_only",
              "x_vix": -0.9, "har_fc_20": 0.14, "input_missing": "", "model_id": "p2m3-x-2024-08-30", "r": 1.875}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ledger.append_today(_row("2026-09-04", p2=nested), path) is True
    with pytest.warns(UserWarning, match="알 수 없는 P2 키"):     # 모르는 p2_* 키는 버리되 경고(조용히 버리지 않음)
        assert ledger.append_today(_row("2026-09-05", p2_prob=0.33, har_vol_fcst=0.15, p2_bogus=1.0, p2_state="reduce",
                                        p2_days_in_state=1, p2_input_missing=None), path) is True
    df = ledger._read(path).set_index("asof")
    a = df.loc["2026-09-04"]
    assert a["prob_dd5_20"] == pytest.approx(0.3) and a["p2_p_m1"] == pytest.approx(0.25) and a["p2_state"] == "normal"
    assert a["p2_days_in_state"] == 12 and a["p2_har_fc_20"] == pytest.approx(0.14) and pd.isna(a["p2_input_missing"])
    assert a["p2_r"] == pytest.approx(1.875) and a["p2_model_id"] == "p2m3-x-2024-08-30"
    b = df.loc["2026-09-05"]
    assert b["prob_dd5_20"] == pytest.approx(0.33) and b["p2_har_fc_20"] == pytest.approx(0.15) and b["p2_state"] == "reduce"
    assert "p2_bogus" not in df.columns


def test_phase1_rows_and_legacy_v1_file_keep_nan_p2_and_extra_columns(tmp_path):
    path = tmp_path / "t.csv"
    # Phase 1 형식의 파일(V1 열 + 추가 열)을 손으로 만든다 — daily.py 가 남긴 tone_faithful·market_status 도 보존되어야 한다
    legacy = pd.DataFrame([{c: np.nan for c in V1_COLUMNS} | {"asof": "2026-09-03", "recorded_at_utc": "2026-09-03T21:00:00Z", "variant": "completed",
                            **{f"state_{k}": "GREEN" for k in V0_SIGNALS}, "score_d": 0.1, "score_w": 0.2, "score_m": 0.3,
                            "overall_d": "GREEN", "overall_w": "GREEN", "overall_m": "GREEN", "tone": "hold", "spy_close": 700.0, "vix_close": 14.0,
                            "run_id": "legacy", "tone_faithful": "hold", "market_status": "current"}])
    legacy.to_csv(path, index=False, lineterminator="\n")
    df = ledger._read(path)
    assert list(df.columns) == ledger.LEDGER_COLUMNS + ["tone_faithful", "market_status"]
    assert df[ledger.P2_COLUMNS].isna().all().all()
    # 새 P2 행을 붙이면 구 행의 P2 열은 NaN 그대로, 추가 열도 유지
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ledger.append_today(_row("2026-09-04", **_p2_flat()), path) is True
    df = ledger._read(path)
    assert list(df.columns) == ledger.LEDGER_COLUMNS + ["tone_faithful", "market_status"]
    old, new = df.iloc[0], df.iloc[1]
    assert old["asof"] == "2026-09-03" and pd.isna(old["p2_state"]) and pd.isna(old["prob_dd5_20"]) and old["tone_faithful"] == "hold"
    assert new["p2_state"] == "caution"
    # Phase 1 방식(P2 없이) append 도 여전히 동작하고 P2 열은 NaN
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ledger.append_today(_row("2026-09-05"), path) is True
    df = ledger._read(path)
    assert df[df["asof"] == "2026-09-05"][ledger.P2_COLUMNS].isna().all().all()


def test_p2_validation_errors_and_warnings(tmp_path):
    path = tmp_path / "t.csv"
    with pytest.raises(ValueError, match="p2_state"):
        ledger.append_today(_row("2026-09-04", **_p2_flat(state="panic")), path)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        ledger.append_today(_row("2026-09-04", **_p2_flat(p=1.2)), path)
    with pytest.raises(ValueError, match="숫자가 아닙니다"):
        ledger.append_today(_row("2026-09-04", **_p2_flat(p2_r="abc")), path)
    with pytest.raises(ValueError, match="p2_days_in_state"):
        ledger.append_today(_row("2026-09-04", **_p2_flat(p2_days_in_state=-1)), path)
    with pytest.warns(UserWarning, match="사유가 없습니다"):     # 확률 NaN 인데 사유 없음 → 경고
        assert ledger.append_today(_row("2026-09-04", **_p2_flat(p=float("nan"))), path) is True
    with warnings.catch_warnings():                              # 사유가 있으면 경고 없음(정상 '확률 계산 불가' 경로)
        warnings.simplefilter("error")
        assert ledger.append_today(_row("2026-09-05", **_p2_flat(p=None, p2_r=None, p2_input_missing="확률 계산 불가: x_vix(VIX 결측)")), path) is True
    with pytest.warns(UserWarning, match="p2_lo"):
        ledger.append_today(_row("2026-09-08", **_p2_flat(p=0.1, p2_lo=0.2, p2_hi=0.3)), path)
    with pytest.warns(UserWarning, match="p2_deploy_mode"):
        ledger.append_today(_row("2026-09-09", **_p2_flat(p2_deploy_mode="maybe")), path)
    df = ledger._read(path).set_index("asof")
    assert df.loc["2026-09-05", "p2_input_missing"].startswith("확률 계산 불가") and pd.isna(df.loc["2026-09-05", "prob_dd5_20"])
    assert df.loc["2026-09-05", "p2_state"] == "caution"          # 입력 결측일에도 상태는 기록(유지)
    assert set(df["p2_state"].dropna()) <= set(P2_STATES)


# ------------------------------------------------------------------
# backfill → summary()["p2"] 라이브 Brier
# ------------------------------------------------------------------
def _spy_with_crashes(n=140, seed=3) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2026-01-05", periods=n)
    ret = rng.normal(0.0004, 0.006, n)
    ret[40:44] = -0.02          # 위치 40~43 에 -8% 급락 → 그 앞 20일의 y_dd5_20 = 1
    ret[95:98] = -0.025         # 두 번째 에피소드
    return pd.Series(100 * np.cumprod(1 + ret), index=idx)


def test_summary_p2_live_brier_matches_hand_calculation(tmp_path):
    path = tmp_path / "t.csv"
    _unlock(tmp_path)
    spy = _spy_with_crashes()
    idx = spy.index
    rng = np.random.default_rng(0)
    n_rows = 60                                             # 위치 0..59 → 20일 뒤 결과가 모두 확정(140 > 80)
    p_list = []
    for i in range(n_rows):
        p = float(np.clip(0.15 + 0.3 * (30 <= i < 45) + rng.normal(0, 0.02), 0.01, 0.99))
        p_list.append(p)
        ledger.append_today(_row(idx[i], **_p2_flat(p=p, p2_p_m1=0.17, p2_p_vix=0.26, p2_p_vix_bgk=0.21, p2_clim=0.16,
                                                    p2_lo=min(p, 0.1), p2_hi=max(p, 0.5), state="normal")), path)
    ledger.append_today(_row(idx[130], **_p2_flat(p=0.2, p2_lo=0.1, p2_hi=0.5, state="normal")), path)  # 아직 미확정(130+20 ≥ 140)
    ledger.append_today(_row(idx[131], variant="faithful", **_p2_flat(p=0.9, p2_lo=0.1, p2_hi=0.95)), path)  # faithful 은 채점에서 제외
    s0 = ledger.summary(path, trading_days=idx)          # backfill 전: 채점 0, 사유는 반환 dict 의 warnings/notes 에 남는다
    assert s0["p2"]["n_scored"] == 0 and s0["p2"]["brier"] is None and s0["p2"]["n_rows"] == 61
    assert any("확정된 행이 없습니다" in w for w in s0["warnings"]) and any("채점된 P2 행이 없습니다" in n for n in s0["p2"]["notes"])
    df = ledger.backfill(path, spy).set_index("asof")
    y = df.loc[[d.strftime("%Y-%m-%d") for d in idx[:n_rows]], "y_dd5_20"].to_numpy(dtype=float)
    assert not np.isnan(y).any() and y.sum() >= 5                      # 급락 앞 20일이 양성
    s = ledger.summary(path, trading_days=idx, spy_close=spy)
    p2 = s["p2"]
    assert p2["schema_version"] == 2 and p2["variant_used"] == "completed" and p2["n_rows"] == 61
    assert p2["n_scored"] == n_rows and p2["n_blocks"] == n_rows // 20
    p = np.array(p_list)
    brier = float(np.mean((p - y) ** 2))
    b_clim = float(np.mean((0.16 - y) ** 2))
    b_m1 = float(np.mean((0.17 - y) ** 2))
    b_vix = float(np.mean((0.26 - y) ** 2))
    b_bgk = float(np.mean((0.21 - y) ** 2))
    assert p2["brier"] == pytest.approx(brier, abs=1e-6)
    assert p2["brier_clim"] == pytest.approx(b_clim, abs=1e-6) and p2["brier_m1"] == pytest.approx(b_m1, abs=1e-6)
    assert p2["brier_vix"] == pytest.approx(b_vix, abs=1e-6) and p2["brier_vix_bgk"] == pytest.approx(b_bgk, abs=1e-6)
    assert p2["bss_clim"] == pytest.approx(1 - brier / b_clim, abs=1e-6)
    assert p2["bss_m1"] == pytest.approx(1 - brier / b_m1, abs=1e-6)
    assert p2["bss_vix"] == pytest.approx(1 - brier / b_vix, abs=1e-6)
    assert p2["bss_vix_bgk"] == pytest.approx(1 - brier / b_bgk, abs=1e-6)
    assert p2["n_ref"] == {"clim": n_rows, "m1": n_rows, "vix": n_rows, "vix_bgk": n_rows, "m3": 0}
    assert p2["bss_m3"] is None and any("p2_p_m3" in n for n in p2["notes"])   # M3 열이 없으면 사유를 남긴다
    assert p2["base_rate"] == pytest.approx(y.mean()) and p2["mean_p"] == pytest.approx(p.mean(), abs=1e-6)
    # 블록 40 부트스트랩 CI (n_scored=60 ≥ 40): 점추정을 포함하는 유한 구간, seed 0 → 결정론
    lo, hi = p2["ci_bss_clim"]
    assert lo <= p2["bss_clim"] <= hi and hi - lo > 0
    assert ledger.summary(path, trading_days=idx, spy_close=spy)["p2"]["ci_bss_clim"] == [lo, hi]
    # 킬룰 카운트다운: 에피소드 표 기준(spy_close 있음), 36개월·8회 미달 → due False
    assert p2["episodes5_observed"] >= 1 and "targets.episodes" in p2["episodes5_method"]
    assert p2["months_elapsed"] == pytest.approx((idx[131] - idx[0]).days / 30.4375, abs=0.01)
    assert p2["kill_rule_due"] is False and p2["kill_rule"]["due"] is False and p2["kill_rule"]["verdict"] is None
    assert p2["kill_rule"]["episodes_required"] == 8 and p2["kill_rule"]["months_required"] == 36
    assert p2["last_state"] == "normal" and p2["last_prob"] == pytest.approx(0.2) and p2["last_asof"] == idx[130].strftime("%Y-%m-%d")
    # spy_close 없이도 동작(런 수 근사) + JSON 직렬화 가능(NaN 없음)
    s2 = ledger.summary(path, trading_days=idx)
    assert "런 수" in s2["p2"]["episodes5_method"] and s2["p2"]["episodes5_observed"] >= 1
    txt = json.dumps(s)
    assert "NaN" not in txt and "Infinity" not in txt


def test_summary_p2_small_sample_skips_ci_and_partial_refs(tmp_path):
    path = tmp_path / "t.csv"
    _unlock(tmp_path)
    spy = _spy_with_crashes()
    idx = spy.index
    for i in range(10):                                   # 채점 행 10 < 블록 40 → CI 생략 + 노트
        ledger.append_today(_row(idx[i], **_p2_flat(p=0.2, p2_p_m1=None, p2_lo=0.1, p2_hi=0.5)), path)
    ledger.backfill(path, spy)
    p2 = ledger.summary(path, trading_days=idx)["p2"]
    assert p2["n_scored"] == 10 and p2["n_blocks"] == 0 and p2["ci_bss_clim"] is None
    assert any("구간 생략" in n for n in p2["notes"])
    assert p2["bss_m1"] is None and p2["n_ref"]["m1"] == 0 and any("m1" in n for n in p2["notes"])   # M1 없으면 skill 없음 + 사유
    assert p2["bss_clim"] is not None and p2["n_ref"]["clim"] == 10


def test_deployed_probability_columns_and_degenerate_skill_note(tmp_path):
    """배포 단이 M1 이면 prob_dd5_20 == p2_p_m1 이고, M3 는 p2_p_m3 에 정보로 남는다.
    같은 값이라 M1 대비 skill 은 정의상 0 — 숨기지 말고 사유를 남긴다."""
    path = tmp_path / "t.csv"
    _unlock(tmp_path)
    spy = _spy_with_crashes()
    idx = spy.index
    for i in range(45):
        p = 0.12 + 0.004 * i
        ledger.append_today(_row(idx[i], **_p2_flat(p=p, p2_p_m1=p, p2_p_m2=p + 0.005, p2_p_m3=p + 0.01, p2_clim=0.16,
                                                    p2_lo=0.05, p2_hi=0.6, state="normal", p2_tone_model="M1",
                                                    p2_deploy_mode="tones", p2_prob_model_id="p2m1-abcd1234-2024-08-30")), path)
    ledger.backfill(path, spy)
    df = ledger._read(path)
    assert (df["prob_dd5_20"] == df["p2_p_m1"]).all() and (df["p2_p_m3"] > df["prob_dd5_20"]).all()
    assert (df["p2_prob_model_id"] == "p2m1-abcd1234-2024-08-30").all()
    p2 = ledger.summary(path, trading_days=idx, spy_close=spy)["p2"]
    assert p2["last_prob_model_id"] == "p2m1-abcd1234-2024-08-30" and p2["last_tone_model"] == "M1"
    assert p2["prob_sources"] == {"M1(배포)": 45}
    assert p2["bss_m1"] == pytest.approx(0.0, abs=1e-12) and any("정의상 0" in n for n in p2["notes"])
    assert p2["bss_m3"] is not None and p2["n_ref"]["m3"] == 45          # 배포 단 vs 정보 단은 실제로 채점된다
    json.dumps(p2)


def test_summary_flags_mixed_prob_sources(tmp_path):
    """배포 단이 바뀐 장부(info_only 시절 M3 행 + tones 시절 M1 행)는 한 계열이 아님을 드러낸다."""
    path = tmp_path / "t.csv"
    spy = _spy_with_crashes()
    idx = spy.index
    for i in range(6):
        ledger.append_today(_row(idx[i], **_p2_flat(p=0.2, p2_lo=0.1, p2_hi=0.5, state="normal",
                                                    p2_deploy_mode="info_only", p2_tone_model=None)), path)
    for i in range(6, 12):
        ledger.append_today(_row(idx[i], **_p2_flat(p=0.21, p2_p_m1=0.21, p2_lo=0.1, p2_hi=0.5, state="normal",
                                                    p2_deploy_mode="tones", p2_tone_model="M1")), path)
    p2 = ledger.summary(path, trading_days=idx)["p2"]
    assert p2["prob_sources"] == {"M3(정보 표시)": 6, "M1(배포)": 6}
    assert any("여러 출처" in n for n in p2["notes"])


def test_holdout_rows_are_not_scored_until_unlock(tmp_path):
    """해제 파일이 없으면 HOLDOUT_START 이후 행은 채점·킬룰에서 빠지고 사유가 남는다(VALIDATION.md §6).
    같은 장부에 해제 파일을 두면 그때부터 채점된다 — 잠금은 '자료 없음' 이 아니라 '아직 열지 않았다' 이다."""
    from mrl.config import HOLDOUT_START
    path = tmp_path / "t.csv"
    spy = _spy_with_crashes()
    idx = spy.index
    assert (idx[:45] >= pd.Timestamp(HOLDOUT_START)).all()          # 픽스처는 전부 홀드아웃 구간 안
    for i in range(45):
        ledger.append_today(_row(idx[i], **_p2_flat(p=0.2, p2_lo=0.1, p2_hi=0.5, state="normal")), path)
    ledger.backfill(path, spy)
    locked = ledger.summary(path, trading_days=idx, spy_close=spy)["p2"]
    assert locked["holdout_locked"] is True and locked["n_scored"] == 0 and locked["n_rows"] == 45
    assert locked["brier"] is None and locked["bss_clim"] is None and locked["kill_rule_due"] is False
    assert locked["months_elapsed"] is None and locked["episodes5_observed"] == 0
    assert any("홀드아웃 미해제" in n for n in locked["notes"])
    json.dumps(locked)
    _unlock(tmp_path)
    opened = ledger.summary(path, trading_days=idx, spy_close=spy)["p2"]
    assert opened["holdout_locked"] is False and opened["n_scored"] == 45 and opened["brier"] is not None
    # unlock_path=None 이면 검사를 끈다(사후 분석용)
    assert ledger._p2_block(ledger._read(path), spy, unlock_path=None)["n_scored"] == 45


def test_summary_without_p2_rows_has_empty_block(tmp_path):
    path = tmp_path / "t.csv"
    ledger.append_today(_row("2026-09-03"), path)
    s = ledger.summary(path, trading_days=pd.bdate_range("2026-09-01", "2026-09-04"))
    p2 = s["p2"]
    assert p2["n_rows"] == 0 and p2["n_scored"] == 0 and p2["brier"] is None and p2["kill_rule_due"] is False
    assert any("P2 행이 없습니다" in n for n in p2["notes"])
    assert ledger.summary(tmp_path / "nope.csv")["p2"]["n_rows"] == 0
    json.dumps(s)


def test_bss_block_ci_properties():
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.2).astype(float)
    p = np.clip(0.2 + 0.3 * y + rng.normal(0, 0.05, 200), 0.01, 0.99)      # 정보 있는 확률
    loss_p, loss_ref = (p - y) ** 2, (0.2 - y) ** 2
    lo, hi = ledger._bss_block_ci(loss_p, loss_ref, block=P2["boot_block"], n_boot=500, seed=0)
    bss = 1 - loss_p.mean() / loss_ref.mean()
    assert lo < bss < hi and lo > 0                                    # 참 skill 이 크면 하한 > 0
    assert ledger._bss_block_ci(loss_p, loss_ref, n_boot=500, seed=0) == (lo, hi)   # 결정론
    assert all(math.isnan(v) for v in ledger._bss_block_ci(np.array([]), np.array([])))
