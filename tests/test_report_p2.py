# -*- coding: utf-8 -*-
"""리포트 Phase 2 확장(mrl/report.py — ARCHITECTURE_PHASE2.md §11) 테스트.

* p2_card: BAD_TOKEN 없이 렌더, 자연빈도 헤드라인·기저율·구간(넓은 쪽), info_only 에서 톤 문구 숨김·'시험 운용' 라벨 노출,
  tones 에서 톤 pill·비중 노출, 입력 결측 → '확률 계산 불가'(상태 유지), 일간 변화 pp 정확 합산, 이벤트 표·60일 경고, 정직 스트립.
* render_calibration_report / render_backtest_v1: 섹션 ①~⑨ / ①~⑦ 존재, v0 completed 숫자(영구 줄), 빈 summary 허용, BAD_TOKEN 없음.
* charts_p2: 합성 oos 로 PNG 8종, 열이 없으면 경고 후 생략(예외 없음).
* render_index: today["p2"] 가 있으면 카드·링크가 끼워지고, 없으면 Phase 1 과 동일.
실제 산출물(results/·docs/)은 건드리지 않는다(tmp_path 만).
"""
from __future__ import annotations

import json
import math
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrl import ledger, report                                   # noqa: E402
from mrl.config import BLOCKS_24, RESULTS_DIR, TONE_EXPOSURE, V0_SIGNALS   # noqa: E402

BAD_TOKEN = re.compile(r"\b(undefined|NaN|nan|None|null|NaT)\b")


def _visible(html: str) -> str:
    t = re.sub(r"<style.*?</style>", "", html, flags=re.S)
    return re.sub(r"<[^>]+>", " ", t)


# ------------------------------------------------------------------
# 합성 자료
# ------------------------------------------------------------------
def _today_p2(**kw) -> dict:
    d = {
        "asof": "2026-09-04", "p": 0.213, "p_m1": 0.19, "p_m2": 0.205, "p_vix": 0.262, "p_vix_bgk": 0.211, "clim": 0.158,
        "param_band": [0.19, 0.24], "calib_band": [0.12, 0.47],
        "calib_bin": {"bin": "[0.20, 0.25)", "lo": 0.20, "hi": 0.25, "n": 440, "n_eff": 22.0, "mean_p": 0.222, "obs": 0.26,
                      "wilson_lo": 0.12, "wilson_hi": 0.47},
        "x_vix": -1.035, "x_har": -0.31, "x_ma": 0.042, "vix": 16.0, "har_vol_20": 0.118, "har_fc_20": 0.131, "rv20_cc": 0.109, "ts_diag": 0.93,
        "percentiles": {"vix": 0.41, "x_har": 0.33, "x_ma": 0.71}, "context": {"VIX3M": 17.2, "SKEW": 141.0, "VVIX": 88.0, "fg": 55},
        "r": 1.35, "state": "caution", "days_in_state": 3, "deploy_mode": "tones", "tone_model": "M3",
        "thresholds": {"state": "caution", "escalate_to": "reduce", "r_escalate": 2.5, "p_escalate": 0.395, "deescalate_to": "normal",
                       "r_deescalate": 1.2, "p_deescalate": 0.1896, "dwell_remaining": 2, "clim": 0.158},
        "churn_252": 4, "churn_alert": False, "reason_ko": "유지 주의: r=1.35 ∈ [1.2, 1.5)",
        "contributions": pd.DataFrame({"logit": [-1.65, -0.92, 0.28, -0.05], "pp": [0.161, -0.089, 0.155, -0.014]},
                                      index=pd.Index(["intercept", "x_vix", "x_har", "x_ma"], name="term")),
        "dod": {"d_logit": {"x_vix": 0.05, "x_har": 0.035, "x_ma": -0.012, "refit": 0.0},
                "d_pp": {"x_vix": 0.0087, "x_har": 0.0061, "x_ma": -0.0021, "refit": 0.0}, "d_p": 0.0127, "refit": False, "gap_sessions": 1},
        "dod_5d": {"d_pp": {"x_vix": 0.021, "x_har": -0.004, "x_ma": 0.001, "refit": 0.0}, "d_p": 0.018, "refit": False, "gap_sessions": 5},
        "events": pd.DataFrame({"date": pd.to_datetime(["2026-09-16", "2026-09-18", "2026-10-02"]), "kind": ["FOMC", "QUAD", "NFP"],
                                "sessions_ahead": [8, 10, 20], "label_ko": ["FOMC 금리 결정(성명 14:00 ET)", "쿼드위칭", "고용보고서(NFP)"],
                                "tentative": [False, False, False]}),
        "events_horizon": {"last_fomc": "2027-12-08", "days_left": 460, "warn": False, "warn_days": 60},
        "model": {"model_id": "p2m3-abcd1234-2024-08-30", "coef": {"x_vix": 0.89, "x_har": 0.9, "x_ma": -1.2}, "intercept": -1.65, "clim": 0.158,
                  "refit_date": "2024-08-30", "spec_sha256": "abcd1234ef567890", "deploy_mode": "tones", "tone_model": "M3"},
        "acceptance": {"rule": "literal", "candidate": "M3", "deploy_mode": "tones", "tone_model": "M3",
                       "literal": {"M3": {"pass": True, "pass_clim": True, "pass_vix": True, "failing_blocks": []}},
                       "amended": {"M3": {"A": True, "B": True, "C": False, "pass": False}}, "amended_tone_model": "M1"},
        "live": {"n_scored": 45, "n_blocks": 2, "brier": 0.1188, "brier_clim": 0.131, "bss_clim": 0.093, "bss_m1": 0.012, "bss_vix": 0.201,
                 "ci_bss_clim": [-0.05, 0.21], "episodes5_observed": 1, "months_elapsed": 2.5, "kill_rule_due": False,
                 "kill_rule": {"episodes_required": 8, "months_required": 36}},
        "har_oos_log_mae": 0.27, "warnings": [],
    }
    d.update(kw)
    return d


def _synthetic_oos(n=1200, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2003-01-02", periods=n)
    x = rng.normal(0, 1, n)
    p3 = 1 / (1 + np.exp(-(-1.6 + 0.9 * x)))
    y = (rng.random(n) < p3).astype(float)
    df = pd.DataFrame({"y": y, "clim": 0.16, "p_vix": np.clip(p3 + 0.1, 0.01, 0.99), "p_vix_driftless": np.clip(p3 + 0.11, 0.01, 0.99),
                       "p_vix_bgk": np.clip(p3 + 0.05, 0.01, 0.99), "p_m1": np.clip(p3 - 0.01, 0.01, 0.99), "p_m2": p3, "p_m3": p3,
                       "refit_year": idx.year, "block24": 1, "block18": 1}, index=idx)
    df.index.name = "date"
    st = np.where(p3 > 0.35, "reduce", np.where(p3 > 0.24, "caution", "normal"))
    df["state"] = st
    df["v0_tone"] = np.where(x > 0.5, "caution", "hold")
    return df


def _summary_p2() -> dict:
    ladder = []
    for step, (a, b) in {"M0->M1": ("M0", "M1"), "M1->M2": ("M1", "M2"), "M2->M3": ("M2", "M3"), "clim->M3": ("clim", "M3"), "B1->M3": ("B1", "M3")}.items():
        for blk in ["all", 1, 2]:
            ladder.append({"step": step, "from": a, "to": b, "block": blk, "start": "2003-01-01", "end": "2024-09-01", "n": 5453 if blk == "all" else 504,
                           "n_blocks": 272 if blk == "all" else 25, "brier_from": 0.135, "brier_to": 0.1205, "bss": 0.107,
                           "mean": 13.6e-4, "lo": -2.8e-4, "hi": 31.4e-4, "dm_t": 1.7, "dm_p": 0.089,
                           "phase_min": -0.01, "phase_median": 0.09, "phase_max": 0.15, "phase_share_pos": 0.9})
    blocks = [{"block": k, "start": a, "end": b, "first_session": a, "last_session": b, "n": 504, "n_blocks": 25, "n_pos": 60, "base": 0.12,
               "mean_p": 0.15, "brier": 0.1, "brier_clim": 0.105, "brier_vix": 0.13, "brier_vix_bgk": 0.11, "brier_m1": 0.101,
               "bss_clim": 0.05 * (1 if k % 3 else -1), "bss_vix": 0.2, "bss_vix_bgk": 0.1, "bss_m1": 0.01, "auc": 0.68, "calib_in_large": 0.03}
              for k, (a, b) in enumerate(BLOCKS_24, start=1)]
    blocks.append({**blocks[0], "block": "all", "n": 5453, "n_blocks": 272})
    return {
        "run": {"generated_at_utc": "2026-09-07T00:00:00Z", "end": "2024-08-30", "first_refit": "2003-01-02", "spec_sha256": "deadbeef" * 8,
                "feature_rule": "p2|x_vix=logit(...)", "python": "3.13", "cache": {"spy_last": "2026-09-04"}},
        "v0_reference": {"brier": 0.14, "bss_clim": -0.027, "auc": 0.526, "first_refit": "2017-01-03"},
        "rungs": {"M0": {"n": 5453, "n_blocks": 272, "brier": 0.155, "bss_clim": -0.11, "bss_vix": 0.0, "auc": 0.70},
                  "M1": {"n": 5453, "n_blocks": 272, "brier": 0.1219, "bss_clim": 0.077, "bss_vix": 0.19, "auc": 0.70},
                  "M3": {"n": 5453, "n_blocks": 272, "brier": 0.1205, "bss_clim": 0.087, "bss_vix": 0.202, "bss_m1": 0.011, "auc": 0.70}},
        "ladder": ladder, "blocks24": blocks, "blocks18": blocks[:3], "blocks_from1999": blocks[:2],
        "reliability": [{"bin": "[0.20, 0.25)", "lo": 0.2, "hi": 0.25, "n": 440, "n_eff": 22.0, "mean_p": 0.222, "obs": 0.26, "wilson_lo": 0.12, "wilson_hi": 0.47},
                        {"bin": "[0.12, 0.16)", "lo": 0.12, "hi": 0.16, "n": 1200, "n_eff": 60.0, "mean_p": 0.14, "obs": 0.13, "wilson_lo": 0.07, "wilson_hi": 0.24},
                        {"bin": "[0.50, 1.00)", "lo": 0.5, "hi": 1.0, "n": 0, "n_eff": 0.0, "mean_p": float("nan"), "obs": float("nan"),
                         "wilson_lo": float("nan"), "wilson_hi": float("nan")}],
        "murphy": {"reliability": 0.002, "resolution": 0.013, "uncertainty": 0.13, "brier": 0.1205, "n": 5453, "n_blocks": 272},
        "era_auc": [{"era": "2000-01-01..2007-12-31", "start": "2000-01-01", "end": "2007-12-31", "n_raw": 2000, "n_pos_raw": 300,
                     "auc_x_vix": 0.72, "auc_x_har": 0.6, "auc_neg_x_ma": 0.58, "n_oos": 1200, "n_pos_oos": 200, "auc_p_m1": 0.71, "auc_p_m3": 0.72},
                    {"era": "2013-01-01..2019-12-31", "start": "2013-01-01", "end": "2019-12-31", "n_raw": 1700, "n_pos_raw": 200,
                     "auc_x_vix": 0.57, "auc_x_har": 0.55, "auc_neg_x_ma": 0.6, "n_oos": 1700, "n_pos_oos": 200, "auc_p_m1": 0.6, "auc_p_m3": 0.63}],
        "acceptance": {"rule": "literal", "candidate": "M3", "deploy_mode": "info_only", "tone_model": None,
                       "literal": {"M3": {"pass": False, "pass_clim": False, "pass_vix": False, "failing_blocks": [3, 8, 9, 11], "n_blocks": 11,
                                          "rule_text": "모든 블록 BSS>0 이고 VIX 대비 ≥0"}},
                       "amended": {"M3": {"A": True, "B": True, "C": False, "pass": False, "rule_text": "A∧B∧C"},
                                   "M1": {"A": True, "B": True, "C": True, "pass": True}},
                       "literal_tone_model": None, "amended_tone_model": "M1", "rationale": "§6 문자 그대로: M3 실패 (실패 블록 [3, 8, 9, 11])",
                       "post_hoc_note": "amended 는 post hoc"},
        "params_by_refit": [{"refit_date": f"{y}-01-02", "rung": "M3", "coef": {"x_vix": 0.8 + 0.01 * (y - 2003), "x_har": 0.9, "x_ma": -1.1},
                             "intercept": -1.6, "n_train": 2000 + 250 * (y - 2003), "n_pos": 300, "clim": 0.16, "train_start": "1993-10-14",
                             "train_end": f"{y - 1}-12-01", "model_id": f"p2m3-deadbeef-{y}-01-02"} for y in range(2003, 2025)]
                            + [{"refit_date": "2010-01-04", "rung": "M1", "coef": {"x_vix": 0.7}, "intercept": -1.0, "n_train": 3000, "n_pos": 400, "clim": 0.17}],
        "param_band": [0.19, 0.24],
        "ablations": {"M3-PK": {"brier": 0.121, "bss_clim": 0.084, "bss_vix": 0.2}, "M3-HAR96": {"brier": 0.1206, "bss_clim": 0.086, "bss_vix": 0.201}},
        "c_sensitivity": {"0.1": {"brier": 0.1206}, "1": {"brier": 0.1205}, "10": {"brier": 0.1205}},
        "har": {"scorecard": [{"block": "all", "start": "2003-01-01", "end": "2024-09-01", "n": 5400, "n_blocks": 270, "mse_log": 0.12, "qlike": 0.08,
                               "r2_log": 0.55, "auc_vol": 0.85, "mse_log_vix": 0.15, "qlike_vix": 0.1, "r2_log_vix": 0.5, "auc_vol_vix": 0.87}],
                "live_coef": {"const": -0.3, "ln_rv1": 0.1, "ln_rv5": 0.3, "ln_rv22": 0.5}, "warnings": ["합성"]},
        "selftest": {"gk_ov_cc_ratio_1996plus_min": 0.85, "gk_ov_cc_ratio_1996plus_max": 1.2, "early_open_eq_hl_share": 0.299},
        "warnings": ["합성 자료 — 테스트용"], "disclosure": "#2 사전 관측 참조",
    }


def _summary_v1() -> dict:
    return {
        "run": {"generated_at_utc": "2026-09-07T00:00:00Z", "start": "2003-01-02", "end": "2024-08-30", "config": "default"},
        "deploy_mode": "info_only", "tone_model": None,
        "kpis": {"n_sessions": 5453, "years": 21.6, "start": "2003-01-02", "end": "2024-08-30", "n_changes": 95, "switches_per_year": 4.4,
                 "kpi_ceiling": 12, "kpi_ceiling_ok": True, "occupancy": {"normal": 0.783, "caution": 0.19, "reduce": 0.027},
                 "n_by_state": {"normal": 4270, "caution": 1036, "reduce": 147}, "dd5_rate_by_state": {"normal": 0.11, "caution": 0.30, "reduce": 0.42},
                 "base_rate": 0.154, "warn_share": 0.217, "n_warn_runs": 48, "median_warn_run": 17.0,
                 "median_run_by_state": {"normal": 60.0, "caution": 15.0, "reduce": 8.0}, "true_alarm_share": 0.312,
                 "true_alarm_share_baseline": 0.154, "true_alarm_edge": 0.158, "true_alarm_share_v0_ref": {"lo": 0.109, "hi": 0.132},
                 "max_changes_any_5_sessions": 2, "structural_bound": 3, "structural_bound_ok": True, "churn_window": 252,
                 "churn_alert_threshold": 12, "churn_alert_sessions": 0, "churn_alert_any": False, "max_churn_252": 9, "n_missing": 0,
                 "missing_share": 0.0, "config": {"enter_caution": 1.5, "exit_caution": 1.2, "enter_reduce": 2.5, "exit_reduce": 2.0, "dwell": 5},
                 "warnings": []},
        "allocation": {"cagr": 0.085, "max_dd": -0.28, "worst_month": -0.09, "total_return": 4.1, "ann_vol": 0.14, "n_switches": 95,
                       "switches_per_year": 4.4, "avg_exposure": 0.87, "cost_total": 0.02, "bh_cagr": 0.095, "bh_max_dd": -0.55,
                       "bh_worst_month": -0.165, "bh_total_return": 6.2, "bh_ann_vol": 0.19, "n_days": 5452, "years": 21.6,
                       "start": "2003-01-02", "end": "2024-08-30"},
        "episodes": {"table": [{"peak_date": "2007-10-09", "trough_date": "2009-03-09", "depth": -0.55, "days_to_trough": 355, "warn_date": "2007-10-19",
                                "lead_days": -8, "lead_capped": False, "held_to_trough": False, "missed": False}],
                     "summary": {"n_episodes": 12, "n_detected": 10, "detection_rate": 0.83, "median_lead_days": 3.0, "false_alarms_per_year": 1.9}},
        "sensitivities": {"wide": {"kpis": {"switches_per_year": 3.1, "warn_share": 0.15, "true_alarm_share": 0.33, "max_changes_any_5_sessions": 2,
                                            "churn_alert_sessions": 0, "kpi_ceiling_ok": True, "occupancy": {"normal": 0.85, "caution": 0.13, "reduce": 0.02},
                                            "allocation": {"cagr": 0.088, "max_dd": -0.3}}},
                          "no_dwell": {"switches_per_year": 9.8, "warn_share": 0.21, "true_alarm_share": 0.25, "max_changes_any_5_sessions": 4,
                                       "churn_alert_sessions": 40, "kpi_ceiling_ok": True, "occupancy": {"normal": 0.79, "caution": 0.18, "reduce": 0.03}}},
        "flags": [], "warnings": ["합성 자료 — 테스트용"],
    }


def _summary_v0() -> dict:
    p = RESULTS_DIR / "summary_v0_completed.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"variant": "completed", "run": {"start": "2015-01-02", "end": "2026-09-04", "n_days": 2936},
            "headline": {"tone_switches_per_year": 82.57, "true_alarm_share": 0.128, "true_alarm_share_baseline": 0.147,
                         "dd5_20_rate_by_tone": {"buy": 0.123, "hold": 0.108, "neutral": 0.109, "caution": 0.181, "reduce": 0.19},
                         "cagr_strategy": 0.099, "cagr_bh": 0.139, "maxdd_strategy": -0.161, "maxdd_bh": -0.337},
            "switching": {"warn_share": 0.465}, "allocation": {"cagr": 0.099, "max_dd": -0.161, "bh_cagr": 0.139, "bh_max_dd": -0.337},
            "brier_reference": {"y_dd5_20": {"base_rate": 0.147, "brier_climatology": 0.1255, "n": 2916}}}


# ------------------------------------------------------------------
# p2_card
# ------------------------------------------------------------------
def test_p2_card_tones_mode_renders_all_sections_without_bad_tokens():
    html = report.p2_card(_today_p2())
    vis = _visible(html)
    assert not BAD_TOKEN.search(vis), BAD_TOKEN.search(vis)
    assert 'id="p2"' in html
    # 1 헤드라인: 자연빈도 + 기저율 + VIX 공식 + VIX 보정 + 구간(넓은 쪽 = 보정 구간 12~47)
    assert "100일 중 약 21일" in html and "10번 중 2번" in html
    assert "평소에는" in html and "100일 중 약 16일" in html and "100일 중 약 26일" in html and "100일 중 약 19일" in html
    assert "범위 12%~47%" in html and "넓은 쪽을 씁니다(calib)" in html and "흔들어 본 범위 19%~24%" in html
    assert ("과거에 이 모델이 20~25% 라고 말한 날들을 모아 보면, 실제로는 100번 중 26번 일어났습니다 "
            "(서로 겹치지 않는 사례 22개, 95% 범위 12%~47%)") in html
    # 2 사다리 오늘값 + 추가 요인
    for nm in (report.LADDER_BI["M0"][0], report.LADDER_BI["BGK"][0], report.LADDER_BI["M1"][0],
               report.LADDER_BI["M2"][0], report.LADDER_BI["M3"][0]):
        assert nm in html
    assert "실제로 움직인 폭과 추세를 더했을 때 확률이 바뀐 정도: +2.3pp" in html
    # 3 상태·톤 (tones): 상태 pill + 톤 pill + 비중 + 다음 임계(확률 환산)
    assert 'class="pill"' in html and "주의" in html and "주식 비중 50%" in html
    assert "caution 에서 내려오려면 확률이 19.0% 아래" in html
    assert "reduce 로 올라가려면 확률이 39.5% 이상" in html and "최소 2거래일은 더 유지" in html
    assert report.INFO_ONLY_BI[0] not in html
    # 4 귀속 3막대 + 일간 변화(정확 합산) + 5일 누적
    assert html.count('class="p2bar"') == 3 and "아무 정보도 안 쓴 출발점이 16.1%" in html
    assert "어제 대비 +1.3pp" in html and "VIX +0.9 · 실현-내재 갭 +0.6 · 추세 -0.2 · (재적합 +0.0)" in html
    assert "5일 누적 +1.8pp" in html
    # 5 요인 문맥 6 HAR 7 이벤트 8 정직 스트립
    assert "10년 백분위" in html and "VIX3M 17.2" in html
    assert "예상 13.1%" in html and "VIX 16.0%" in html and "실제로 움직인 폭 10.9%" in html and "둘의 차이 +2.9pp" in html
    assert "빗나간 정도 0.270" in html
    assert "2026-09-16" in html and "FOMC" in html and "이벤트 표 잔여" not in html
    assert "숨기지 않는 것들" in html and "p2m3-abcd1234-2024-08-30" in html and "b0 -1.650 · x_vix +0.890" in html
    assert "미리 정해둔 규칙 그대로(literal, M3): 통과" in html
    assert "결과를 본 뒤 완화한 규칙(amended #2a, post hoc, M3): 실패" in html
    assert "실제로 5% 넘게 떨어진 사건 1/8 · 지난 기간 2.5/36개월" in html
    assert "채점 45행 (겹치지 않는 창 2)" in html
    assert "평소 평균보다 나아진 정도 +0.093 (95% [-0.050, +0.210])" in html
    assert report.P2_FOOTNOTE in html


def test_p2_card_headline_is_the_deployed_rung_and_labels_the_rest_as_information():
    """acceptance 가 M1 을 배치하면 헤드라인·상태·구간은 M1 의 확률이고, M3 는 사다리에 '정보 표시(배포 안 함)' 로만 남는다."""
    d = _today_p2(p=0.19, p_m3=0.213, tone_model="M1", deploy_mode="tones", prob_rung="M1", r=1.20,
                  model={"model_id": "p2m1-abcd1234-2024-08-30", "coef": {"x_vix": 0.92}, "intercept": -1.0, "clim": 0.158,
                         "refit_date": "2024-08-30", "spec_sha256": "abcd1234ef567890", "deploy_mode": "tones", "tone_model": "M1"},
                  prod_model={"model_id": "p2m3-abcd1234-2024-08-30", "coef": {"x_vix": 0.89, "x_har": 0.9, "x_ma": -1.2},
                              "intercept": -1.65, "clim": 0.158, "refit_date": "2024-08-30"},
                  contributions=pd.DataFrame({"logit": [-1.0, -0.45], "pp": [0.269, -0.079]},
                                             index=pd.Index(["intercept", "x_vix"], name="term")),
                  dod={"d_logit": {"x_vix": 0.05, "refit": 0.0}, "d_pp": {"x_vix": 0.0087, "refit": 0.0}, "d_p": 0.0087,
                       "refit": False, "gap_sessions": 1},
                  dod_5d={"d_pp": {"x_vix": 0.021, "refit": 0.0}, "d_p": 0.021, "refit": False, "gap_sessions": 5})
    html = report.p2_card(d)
    vis = _visible(html)
    assert not BAD_TOKEN.search(vis), BAD_TOKEN.search(vis)
    assert "100일 중 약 19일" in html                         # 헤드라인 = M1 (M3 의 21일이 아니다)
    assert "실제로 쓰는 확률 = M1 (VIX 확률을 실제와 맞춤, 조정 숫자 2개)" in html and f"M3 는 {report.INFO_DISPLAY_BI[0]}" in html
    assert "이 판정을 실제로 씁니다 — M1" in html and "주식 비중 50%" in html   # 배포됐으므로 판정·비중은 그대로
    assert "지금 확률은 평소의 1.20배입니다" in html
    assert html.count(report.INFO_DISPLAY_BI[0]) >= 2                 # 사다리 M2·M3 칸 + 눈썹줄
    assert "이 확률이 어떻게 만들어졌나 — M1 기준" in html and html.count('class="p2bar"') == 1     # M1 은 VIX 항 하나뿐
    assert f"만들어 둔 모델 M3 ({report.INFO_DISPLAY_BI[0]})" in vis and "p2m3-abcd1234-2024-08-30" in html
    assert "b0 -1.000 · x_vix +0.920" in html                        # 정직 스트립의 계수 = 배포 모델의 것
    # 사다리에 세 단이 모두 있고 M3 는 정보값(21.3%)으로 남는다
    assert "21.3%" in html and "19.0%" in html


def test_p2_card_info_only_hides_tone_and_shows_trial_label():
    html = report.p2_card(_today_p2(deploy_mode="info_only", tone_model=None))
    assert report.INFO_ONLY_BI[0] in html
    # 톤 문구 숨김: 톤 pill 도, 실제 비중 **값**(예 "주식 비중 50%")도 나오지 않아야 한다.
    # ("신호등 판정이나 주식 비중을 주장하지 않습니다" 처럼 비중을 **부정**하는 문장은 나와야 하므로 낱말만 찾지 않는다.)
    assert 'class="pill"' not in html
    assert not re.search(r"주식 비중\s*\d+\s*%", html)
    assert "caution" in html and "주의" in html                            # 상태는 회색으로 정보만(식별자 + 뜻풀이)
    assert "비중은 예전 v0 규칙을 그대로 따릅니다" in html
    assert not BAD_TOKEN.search(_visible(html))


def test_p2_card_input_missing_keeps_state_and_says_unavailable():
    d = _today_p2(p=float("nan"), p_m2=None, r=float("nan"), input_missing="확률 계산 불가: x_vix(VIX 결측)", contributions=None,
                  dod={"d_pp": {"x_vix": float("nan"), "x_har": float("nan"), "x_ma": float("nan"), "refit": 0.0}, "d_p": float("nan"), "refit": False})
    html = report.p2_card(d)
    assert report.P2_PROB_UNAVAILABLE in html and "x_vix(VIX 결측)" in html
    assert "100일 중 약" not in html.split("단계별 모델이 오늘 내놓은 값")[0]   # 헤드라인 자연빈도 없음
    assert "caution" in html and "이 상태로 3거래일째" in html   # 상태 유지
    assert "평소 대비 몇 배인지 계산할 수 없습니다" in html
    assert "변화 계산 불가" in html
    assert not BAD_TOKEN.search(_visible(html))
    # 확률 NaN 인데 사유가 없으면 카드가 사유 미기록을 드러낸다(지어내지 않음)
    html2 = report.p2_card({"p": None, "state": "normal", "deploy_mode": "info_only"})
    assert "사유 미기록" in html2 and not BAD_TOKEN.search(_visible(html2))


def test_p2_card_empty_and_odd_inputs():
    html = report.p2_card({})
    assert 'id="p2"' in html and not BAD_TOKEN.search(_visible(html))
    assert "상태 자료가 없습니다" in html and "귀속 자료 없음" in html
    assert "손대지 않고 남겨둔 최근 구간(2024-09-01~): 아직 열지 않았습니다" in html
    html = report.p2_card(None)
    assert 'id="p2"' in html
    # 별칭(장부 열 이름)·churn 경보·이벤트 표 잔여 경고·홀드아웃
    d = {"prob_dd5_20": 0.3, "p2_clim": 0.15, "p2_state": "reduce", "p2_days_in_state": 9, "p2_deploy_mode": "tones", "p2_r": 2.0,
         "p2_lo": 0.2, "p2_hi": 0.4, "p2_band_src": "param", "churn_alert": True, "churn_252": 14,
         "events_horizon": {"last_fomc": "2026-10-28", "days_left": 40, "warn": True, "warn_days": 60},
         "holdout": {"n": 484, "bss_clim": 0.04, "bss_vix": 0.15, "bss_m1": -0.01, "ci": [-0.1, 0.18]}}
    html = report.p2_card(d)
    assert "100일 중 약 30일" in html and "범위 20%~40%" in html and report.P2_CHURN_LABEL_BI[0] in html
    assert "이벤트 표 잔여 40일" in html
    assert "손대지 않고 남겨둔 최근 구간(2024-09-03~, 한 번만 씁니다): n=484 (겹치지 않는 창 24)" in html
    assert "reduce" in html and "축소" in html and "주식 비중 25%" in html
    assert not BAD_TOKEN.search(_visible(html))


def test_exact_pp_parts_sum_matches_total_after_rounding():
    parts = {"x_vix": 0.00449, "x_har": 0.00449, "x_ma": 0.00449, "refit": 0.0}
    r, tot = report._exact_pp_parts(parts, sum(parts.values()))
    assert tot == pytest.approx(1.3)                        # 1.347 → 1.3
    assert sum(r.values()) == pytest.approx(tot, abs=1e-9)  # 0.4+0.4+0.4=1.2 → 잔차 0.1 을 최대 항에
    r2, tot2 = report._exact_pp_parts({"a": 0.0123, "b": -0.0051}, 0.0072)
    assert sum(r2.values()) == pytest.approx(tot2, abs=1e-9) and tot2 == pytest.approx(0.7)
    r3, tot3 = report._exact_pp_parts({"a": float("nan"), "b": 0.01}, float("nan"))
    assert math.isnan(r3["a"]) and math.isnan(tot3)          # 결측 항이 있으면 합계를 지어내지 않는다
    r3b, tot3b = report._exact_pp_parts({"a": float("nan"), "b": 0.01}, 0.01)
    assert math.isnan(r3b["a"]) and r3b["b"] == pytest.approx(1.0) and tot3b == pytest.approx(1.0)
    r4, tot4 = report._exact_pp_parts({}, float("nan"))
    assert math.isnan(tot4) and r4 == {}


# ------------------------------------------------------------------
# calibration report
# ------------------------------------------------------------------
def test_render_calibration_report_sections_and_v0_line(tmp_path):
    s = _summary_p2()
    s["v0_completed"] = _summary_v0()
    out = tmp_path / "docs" / "calibration_p2.html"
    with warnings.catch_warnings():
        warnings.simplefilter("error")           # 완전한 summary 면 경고 없이 렌더
        report.render_calibration_report(s, out, {})
    html = out.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    for i in range(1, 10):
        assert f'id="s{i}"' in html
    vis = _visible(html)
    assert not BAD_TOKEN.search(vis), BAD_TOKEN.search(vis)
    # ① v0 줄 (영구): summary_v0_completed 의 숫자
    v0 = report.v0_headline(_summary_v0())
    assert "v0 (동결 벤치마크" in html and "영구 표기" in html
    assert f"{v0['tone_switches_per_year']:,.1f}회/년" in html and f"{v0['cagr_strategy'] * 100:.1f}%" in html
    # ② 사다리: 단별·단 간(전체·블록별), ×1e-4 손실차
    assert "M0-&gt;M1" in html or "M0->M1" in html
    assert "+13.6" in html and "-2.8" in html and "+31.4" in html and "DM t(HAC 19)" in html
    # ③ 블록·판정: info_only 상자 + 실패 블록 + amended 표
    assert "정보 제공 전용(info_only)" in html and "[3, 8, 9, 11]" in html and "#2a 완화안 (post hoc)" in html
    assert "24개월 블록(주, M3 생산 모델)" in html and "18개월 블록(민감도, M3)" in html and "1999 시작" in html
    # ④ 신뢰도 표 (빈 구간은 —) ⑤ 계수 경로 (M3 만 22행) ⑥ 시대 ⑦ 소거 ⑧ HAR ⑨ spec
    assert "[0.20, 0.25)" in html and "Murphy 분해" in html
    assert html.count("p2m3-deadbeef-") == 22 and "M1" in html
    assert "2013-2019" in html or "2013-01-01" in html
    assert "M3-PK" in html and "M3-HAR96" in html and "C 민감도" in html
    assert "QLIKE" in html and "ln_rv22" in html
    assert "deadbeef" * 8 in html and "p2|x_vix=logit(...)" in html and "합성 자료 — 테스트용" in html
    import html as _h
    for t in report.P2_HONESTY_ITEMS:
        assert t[:30] in _h.unescape(vis)
    assert "#0b1220" in html and "fonts.googleapis" not in html and b"\r\n" not in out.read_bytes()


def test_render_calibration_report_tolerates_empty_summary(tmp_path):
    out = tmp_path / "c.html"
    report.render_calibration_report({"v0_completed": {}}, out, {"weird": b""})
    html = out.read_text(encoding="utf-8")
    assert "v0 completed 요약을 찾지 못해" in html          # v0 줄은 항상 있고, 숫자가 없으면 경고 목록에 남는다
    assert "v0 (동결 벤치마크" in html
    # 파일 폴백: 없는 경로면 경고 + {}
    with pytest.warns(UserWarning):
        assert report.load_summary_v0(tmp_path / "nope.json") == {}
    for i in range(1, 10):
        assert f'id="s{i}"' in html
    assert "자료 없음" in html or "없음" in html
    assert "차트 없음 — 추가 차트: weird" in html
    assert not BAD_TOKEN.search(_visible(html))
    with pytest.raises(TypeError):
        report.render_calibration_report([], out, {})


# ------------------------------------------------------------------
# backtest_v1 report
# ------------------------------------------------------------------
def test_render_backtest_v1_side_by_side_with_v0(tmp_path):
    out = tmp_path / "docs" / "backtest_v1.html"
    v0 = _summary_v0()
    report.render_backtest_v1(_summary_v1(), v0, out, {})
    html = out.read_text(encoding="utf-8")
    for i in range(1, 8):
        assert f'id="s{i}"' in html
    vis = _visible(html)
    assert not BAD_TOKEN.search(vis), BAD_TOKEN.search(vis)
    v0h = report.v0_headline(v0)
    assert "v0 (동결 벤치마크" in html                                       # ① 먼저
    assert html.index('id="s1"') < html.index('id="s2"')
    assert "4.4" in html and f"v0 {v0h['tone_switches_per_year']:,.1f}" in html   # 전환/년 나란히
    assert "30.0~42.0%" in html and "정상 11.0%" in html                      # 상태별 dd5
    assert "기저율 15.4%" in html and f"v0 {v0h['true_alarm_share'] * 100:.1f}%" in html
    assert "v1 결정층" in html and "v0 톤" in html and "보유(B&amp;H)" in html   # 배분 비교표
    assert "2007-10-09" in html                                               # 에피소드
    assert "wide" in html and "no_dwell" in html                              # 민감도
    assert "사실상 2단계" in html and report.P2_FOOTNOTE in html
    assert "KPI 상한 12 충족" in html


def test_render_backtest_v1_tolerates_empty(tmp_path):
    out = tmp_path / "b.html"
    report.render_backtest_v1({}, {}, out, {})
    html = out.read_text(encoding="utf-8")
    for i in range(1, 8):
        assert f'id="s{i}"' in html
    assert "allocation 결과 없음" in html and "sensitivities 없음" in html
    assert not BAD_TOKEN.search(_visible(html))
    with pytest.raises(TypeError):
        report.render_backtest_v1(None, {}, out, {})


# ------------------------------------------------------------------
# charts_p2
# ------------------------------------------------------------------
def test_charts_p2_returns_pngs_from_oos_and_summary():
    oos = _synthetic_oos()
    spy = pd.Series(300 * np.cumprod(1 + np.random.default_rng(1).normal(0.0003, 0.01, len(oos))), index=oos.index)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        charts = report.charts_p2(oos, BLOCKS_24[:3], _summary_p2(), spy)
    assert set(charts) == {"reliability", "block_skill", "coef_path", "ladder_ci", "era_auc", "prob_path", "state_bands", "cumret_v1"}
    for png in charts.values():
        assert isinstance(png, bytes) and png[:8] == b"\x89PNG\r\n\x1a\n"
    # summary 표가 없으면 oos 로 계산(reliability·block_skill), 상태 열 없으면 밴드·누적 생략(경고)
    with pytest.warns(UserWarning):
        c2 = report.charts_p2(oos.drop(columns=["state", "v0_tone"]), BLOCKS_24[:3], {}, spy)
    assert {"reliability", "block_skill", "prob_path"} <= set(c2) and "state_bands" not in c2 and "coef_path" not in c2
    # 아무것도 없으면 빈 dict + 경고, 예외 없음
    with pytest.warns(UserWarning):
        assert report.charts_p2(None, [], {}, None) == {}
    # backtest_v1 모양(p, clim, state) 도 허용
    v1 = pd.DataFrame({"p": oos["p_m3"], "clim": 0.16, "state": oos["state"]}, index=oos.index)
    with pytest.warns(UserWarning):
        c3 = report.charts_p2(v1, [], {}, spy)
    assert {"prob_path", "state_bands", "cumret_v1"} <= set(c3)


def test_charts_p2_prob_path_legend_claims_deployment_only_when_deployed(monkeypatch):
    """범례의 ', deployed' 는 명시 불리언(rung_deployed)에만 붙는다 — 단 이름이 있다고 배포된 것은 아니다."""
    labels: list[str] = []
    real_png = report._png

    def _cap(fig):
        for ax in fig.axes:
            labels.extend(ax.get_legend_handles_labels()[1])
        return real_png(fig)

    monkeypatch.setattr(report, "_png", _cap)
    oos = _synthetic_oos()
    spy = pd.Series(300.0, index=oos.index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report.charts_p2(oos, [], {"deployed_rung": "M3"}, spy)
        assert "p (M3)" in labels and "p (M3, deployed)" not in labels
        labels.clear()
        report.charts_p2(oos, [], {"deployed_rung": "M3", "rung_deployed": True}, spy)
        assert "p (M3, deployed)" in labels


def test_charts_p2_uses_the_deployed_rungs_precomputed_tables(monkeypatch):
    """배포 단이 M1 이면 reliability_m1 · blocks24_by_rung['M1'] 을 쓴다(M3 표를 M1 이름으로 그리지 않는다)."""
    oos = _synthetic_oos()
    spy = pd.Series(300.0, index=oos.index)
    s = _summary_p2()
    s["deployed_rung"] = "M1"
    s["reliability_m1"] = [{"bin": "[0.10, 0.20)", "n": 400, "n_eff": 20, "mean_p": 0.15, "obs": 0.11,
                            "wilson_lo": 0.02, "wilson_hi": 0.33}]
    s["blocks24_by_rung"] = {"M1": [{"block": 1, "start": "2003-01-01", "end": "2005-01-01", "n": 400, "n_blocks": 20,
                                     "bss_clim": 0.111, "bss_vix": 0.05, "bss_vix_bgk": 0.04, "bss_m1": 0.0}]}
    used = {}
    real_records = report._to_records
    monkeypatch.setattr(report, "_to_records", lambda v: (used.setdefault("seen", []).append(v), real_records(v))[-1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report.charts_p2(oos, BLOCKS_24[:3], s, spy)
    seen = used["seen"]
    assert s["reliability_m1"] in seen and s["blocks24_by_rung"]["M1"] in seen
    assert s["reliability"] not in seen and s["blocks24"] not in seen


def test_honesty_strip_reports_the_rung_amended_actually_selected():
    """amended(#2a)가 M1 을 골라 tones 를 배포했으면, 후보 M3 의 '실패' 가 아니라 M1 의 '통과' 를 적는다."""
    d = _today_p2()
    d["deploy_mode"], d["tone_model"] = "tones", "M1"
    d["acceptance"] = {"rule": "amended", "candidate": "M3", "deploy_mode": "tones", "tone_model": "M1",
                       "amended_tone_model": "M1",
                       "literal": {"M3": {"pass": False, "failing_blocks": [3, 8, 9]}},
                       "amended": {"M3": {"pass": False, "A": True, "B": True, "C": False},
                                   "M1": {"pass": True, "A": True, "B": True, "C": True}}}
    html = report.p2_card(d)
    assert "결과를 본 뒤 완화한 규칙(amended #2a, post hoc, M1): 통과 (미리 정해둔 후보 M3 는 실패)" in html
    assert "amended(#2a, post hoc): 실패" not in html
    assert not BAD_TOKEN.search(_visible(html))


def test_honesty_strip_does_not_claim_zero_skill_when_rows_came_from_another_rung():
    """'배포 단이 M1 이라 vs M1 은 정의상 0' 은 장부가 실제로 확인한 동일성에만 붙인다."""
    d = _today_p2()
    d["deploy_mode"], d["tone_model"] = "tones", "M1"
    d["live"] = {"n_scored": 2, "n_blocks": 0, "brier": 0.19, "bss_clim": 0.02, "bss_m1": -0.038,
                 "bss_vix": 0.01, "prob_sources": {"M3(정보 표시)": 1, "M1(배포)": 1}, "notes": [],
                 "kill_rule": {"episodes_required": 8, "months_required": 36}}
    html = report.p2_card(d)
    assert "정의상 0" not in html and "여러 단계에서 나왔습니다" in html
    # 실제로 M1 계열만 채점됐고 장부가 동일성을 확인했으면 문구가 돌아온다
    d["live"] = {**d["live"], "bss_m1": 0.0, "prob_sources": {"M1(배포)": 45},
                 "notes": ["m1 대비 skill 은 정의상 0 입니다 — 배포 확률(prob_dd5_20)이 M1 그 자체이기 때문"]}
    html2 = report.p2_card(d)
    assert "채점한 확률이 M1 에서 나왔으므로 M1 대비는 정의상 0 입니다" in html2
    assert "여러 단계에서 나왔습니다" not in html2


# ------------------------------------------------------------------
# render_index 통합
# ------------------------------------------------------------------
def _today(with_p2: bool) -> dict:
    day = {"asof": "2026-09-04", "tone": "hold", "verdict_ko": "꾸준한 상승 흐름", "overall_d": "GREEN", "overall_w": "GREEN", "overall_m": "GREEN",
           "score_d": 0.5, "score_w": 0.6, "score_m": 0.7, "states_d": {k: "GREEN" for k in V0_SIGNALS}, "n_watch_avail": 28,
           "fg_avail": True, "eod_avail": True}
    t = {"asof": "2026-09-04", "market_status": "current", "generated_at": "2026-09-04 16:25 ET", "spy_close": 770.19, "vix_close": 14.53,
         "warnings": [], "faithful": day, "completed": day}
    if with_p2:
        t["p2"] = _today_p2()
        t["p2"].pop("live")
    return t


def test_render_index_with_p2_card_and_links(tmp_path):
    path = tmp_path / "track_record.csv"
    ledger.append_today({"asof": "2026-09-03", "variant": "completed", "tone": "hold", "states_d": {k: "GREEN" for k in V0_SIGNALS},
                         "p2_prob": 0.2, "p2_clim": 0.15, "p2_state": "normal", "p2_days_in_state": 40, "p2_deploy_mode": "info_only"}, path)
    ls = ledger.summary(path, trading_days=pd.bdate_range("2026-09-01", "2026-09-04"))
    out = tmp_path / "index.html"
    report.render_index(_today(True), ls, out)
    html = out.read_text(encoding="utf-8")
    assert 'id="p2"' in html and "100일 중 약 21일" in html
    assert 'href="calibration_p2.html"' in html and 'href="backtest_v1.html"' in html and 'href="backtest_v0.html"' in html
    # 장부 행(2026-09-03)은 홀드아웃 구간 안이고 해제 파일이 없다 → 채점 보류를 사유와 함께 밝힌다(§6)
    assert "실제 기록으로 매긴 점수: 아직 매기지 않습니다 — 남겨둔 구간을 열지 않았기 때문입니다" in html
    assert "P2: 홀드아웃 미해제" in html                          # 장부 P2 주석이 경고 목록에 실린다
    assert html.index("꾸준한 상승 흐름") < html.index('id="p2"')     # v0 판정 블록 아래
    assert not BAD_TOKEN.search(_visible(html))
    # p2 없으면 Phase 1 과 동일(카드·링크 없음)
    report.render_index(_today(False), ls, out)
    html = out.read_text(encoding="utf-8")
    assert 'id="p2"' not in html and 'href="calibration_p2.html"' not in html


def test_fmt_p2_rules():
    f = report._fmt_p2
    assert f(0.1205, "brier") == "0.1205" and f(0.087, "bss_clim") == "+0.087" and f(0.7, "auc") == "0.700"
    assert f(13.6e-4, "diff_mean") == "+13.6" and f(-2.8e-4, "diff_lo") == "-2.8"
    assert f(0.158, "clim") == "15.8%" and f(0.26, "obs") == "26.0%" and f(0.217, "warn_share") == "21.7%"
    assert f(1.35, "r") == "1.35" and f(4.4, "switches_per_year") == "4.4" and f(5453.0, "n") == "5,453"
    assert f(-1.65, "intercept") == "-1.650" and f(0.89, "b_x_vix") == "+0.890"
    assert f(float("nan"), "brier") == "—" and f(True, "pass") == "예"
    assert 'class="st"' in f("caution", "state") and f("hold", "tone").startswith('<span class="pill"')
    assert f([0.19, 0.24], "p") == "[19.0%, 24.0%]"
