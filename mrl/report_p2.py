# -*- coding: utf-8 -*-
"""Phase 2 페이지 — 확률 카드 · docs/calibration_p2.html · docs/backtest_v1.html · 차트.

mrl/report.py 에서 그대로 옮겨온 코드다(계산·문구·임계값 변경 없음). ARCHITECTURE_PHASE2.md §11.
"""
from __future__ import annotations

import base64
import html as _h
import io
import math
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from mrl import sizing as _sizing
from mrl import track as _track
from mrl.config import (TONES, TONE_EXPOSURE, V0_SIGNALS, SITE_URL, BACKTEST_START, DECISION_P2, P2, P2_STATES, STATE_TO_TONE,
                        HOLDOUT_START, RESULTS_DIR, P3, HMM_P3, ENSEMBLE_P3, KILL_P3, P3_DRIFT, SCENARIO_P3,
                        MODEL_P2_PATH, MODEL_P3_PATH, KILL_RECORD_PATH, KILL_MANUAL_PATH)

from mrl.report_common import (LABELS, PALETTE, _esc, _fmt, _img, _is_nan, _is_num, _kpi, _kv_table, _label, _legend, _new_fig,
                               _now_str, _plain_log_axis, _png, _records_table, _scalar_ok, _series_close, _to_records,
                               _tone_pill, _tone_runs, _warn_list, _write_html, bi, bi_attr, bi_html)


# ==================================================================
# Phase 2 — 보정 확률 카드 · 보정 리포트 · 결정층 v1 리포트 · 차트 (ARCHITECTURE_PHASE2.md §11)
# ==================================================================
# 설계 원칙(Phase 1 과 동일): 어떤 키가 있든 있는 것만 렌더링한다. 없는 값은 "—", 비어 있으면 '자료 없음'.
# 확률은 자연빈도("100일 중 약 N일")로, 겹치는 라벨의 표본 수는 항상 n_blocks(=n/20)를 병기한다(§16 6).
# 차트 안 글자는 ASCII 만. 한글은 HTML 캡션에.
P2_STATE_COLORS = {"normal": "#22c55e", "caution": "#eab308", "reduce": "#ef4444"}


P2_STATE_KO = {"normal": "정상", "caution": "주의", "reduce": "축소"}


# 상태 이름의 영어 뜻 — 식별자(normal·caution·reduce)는 그대로 두고 뜻풀이만 언어를 따른다
P2_STATE_EN = {"normal": "normal", "caution": "be careful", "reduce": "cut back"}


FEATURE_KO = {"x_vix": "VIX", "x_har": "실현-내재 갭", "x_ma": "추세", "refit": "재적합", "intercept": "절편"}


# 막대·변화 문장이 쓰는 **짧은** 요인 이름 한 쌍 (열 제목용 BI_LABELS 의 긴 표현과 달리 좁은 칸에 맞춘다).
# FEATURE_KO 는 장부·테스트가 글자 그대로 참조하므로 손대지 않고, 화면에 쓸 짝만 새로 둔다.
FEATURE_BI = {"x_vix": ("VIX", "VIX"), "x_har": ("실현-내재 갭", "realised-implied gap"),
              "x_ma": ("추세", "trend"), "refit": ("재적합", "refit"), "intercept": ("절편", "intercept")}


def _feat_bi(key: str) -> tuple[str, str]:
    """요인 이름 한 쌍 — 막대와 변화 문장이 **같은 이름**을 쓰게 한 곳에서 준다."""
    return FEATURE_BI.get(key, (FEATURE_KO.get(key, key), key))


FEATURE_COLORS = {"x_vix": "#60a5fa", "x_har": "#f472b6", "x_ma": "#34d399", "refit": "#8a94a8", "intercept": "#5c6a82"}


RUNG_KO = {"M0": "VIX 공식(보정 전)", "BGK": "VIX 공식(일별 관측 보정)", "B1": "VIX 공식(B1)", "DRIFTLESS": "VIX 공식(무드리프트)",
           "M1": "VIX 보정만(M1)", "M2": "+실현변동성(M2)", "M3": "+추세(M3)", "clim": "기후학(기저율)", "CLIM": "기후학(기저율)"}


INFO_ONLY_LABEL = "시험 운용 — 비중 제안 아님"                      # VALIDATION §7 문구


INFO_DISPLAY_LABEL = "정보 표시(배포 안 함)"                        # 배포되지 않은 단(사다리·비교층)에 붙이는 라벨


# 위 한국어 상수의 **유일한** 영어 판 — 페이지마다 다른 말을 쓰지 않게 한 곳에 둔다.
# '배포 안 함' = 실제로 쓰지 않음이므로 STYLE_I18N.md §3 용어표대로 "in use" 를 쓴다("deployed" 아님).
INFO_DISPLAY_EN = "shown for information (not in use)"


DEPLOYED_LABEL = "배포"                                            # acceptance 가 배치한 단 — 상태·톤·비중이 여기서 나온다


RUNG_PARAMS = {"M0": 0, "M1": 2, "M2": 3, "M3": 4}                 # 적합 파라미터 수(절편 포함)


P2_PROB_UNAVAILABLE = "확률 계산 불가"


P2_CHURN_LABEL = "결정층 잦은 전환"


# ── 카드용 쉬운 한국어 + 영어 짝 (STYLE_I18N.md §2·§3) ────────────────
# 위 상수들은 장부·테스트가 글자 그대로 참조하므로 손대지 않고, 화면에 쓸 짝만 새로 둔다.
P2_PROB_UNAVAILABLE_BI = ("확률을 계산할 수 없습니다", "the probability could not be computed")
P2_CHURN_LABEL_BI = ("판정이 너무 자주 바뀝니다", "the call is switching too often")
INFO_ONLY_BI = ("시험 운용 — 주식을 얼마나 들고 갈지는 제안하지 않습니다",
                "trial run — this does not suggest how much to hold in stocks")
INFO_DISPLAY_BI = ("참고용만 (실제로 쓰지 않음)", "information only (not in use)")
DEPLOYED_BI = ("실제 사용", "in use")

# 사다리 칸 이름 — (쉬운 한국어, English). 괄호 안 식별자(M1·M2·M3)는 번역하지 않는다.
# (아래 _rung_bi() 는 '조정 숫자 몇 개' 까지 밝히는 더 자세한 판이고, 이 표는 사다리 칸 제목용이다.)
LADDER_BI = {
    "M0": ("VIX 만 쓴 공식 (맞추기 전)", "the VIX-only formula (before correction)"),
    "BGK": ("VIX 공식 (하루하루 관측으로 맞춤)", "the VIX formula, matched to daily observations"),
    "B1": ("VIX 공식 (B1)", "the VIX formula (B1)"),
    "DRIFTLESS": ("VIX 공식 (흐름 보정 없음)", "the VIX formula (no drift)"),
    "M1": ("VIX 를 실제와 맞춘 것 (M1)", "the VIX probability matched to reality (M1)"),
    "M2": ("+ 실제로 움직인 폭 (M2)", "+ how much it actually moved (M2)"),
    "M3": ("+ 추세 (M3)", "+ the trend (M3)"),
    "clim": ("평소 비율", "the base rate"), "CLIM": ("평소 비율", "the base rate"),
}


# 요인 문맥 해설 — (쉬운 한국어, English). 이름(vix·x_har…)은 열 값이라 번역하지 않는다.
_CTX_NOTES_BI = {
    "vix": ("옵션 시장이 보는 앞으로의 출렁임 — 높을수록 확률이 올라갑니다",
            "how much swing the options market expects — the higher it is, the higher the probability"),
    "x_vix": ("위 값을 모델이 쓰는 형태로 바꾼 것 (모델 입력)",
              "the same value reshaped for the model to use (a model input)"),
    "har_vol_20": ("최근에 실제로 움직인 폭을 기간별로 섞어 만든 평균",
                   "an average of how much it actually moved, blended over several look-back lengths"),
    "x_har": ("실제로 움직인 폭 − 옵션 시장이 본 폭: 음수면 옵션이 더 비싸게 보고 있다는 뜻",
              "how much it actually moved minus what the options market expected: negative means options are dearer"),
    "x_ma": ("종가가 180일 평균보다 얼마나 위인지: 음수면 추세 아래",
             "how far the close sits above its 180-day average: negative means below the trend"),
    "rv20_cc": ("최근 20일 동안 종가 기준으로 실제로 움직인 폭 (보여 주기만)",
                "how much it actually moved over the last 20 days, close to close (display only)"),
    "ts_diag": ("가까운 만기와 3개월 만기의 비교 (보여 주기만)",
                "near-term versus three-month expectations (display only)"),
}


def _nat_freq_bi(p) -> str:
    """자연빈도 두 벌: '100일 중 약 9일' / 'about 9 days in 100' (STYLE_I18N.md §2.2)."""
    f = _fnum(p)
    if math.isnan(f):
        return "—"
    n = int(round(f * 100))
    return bi(f"100일 중 약 {n}일", f"about {n} days in 100")


def _ten_freq_bi(p) -> str:
    f = _fnum(p)
    if math.isnan(f):
        return "—"
    n = int(round(f * 10))
    return bi(f"10번 중 {n}번", f"{n} time in 10" if n == 1 else f"{n} times in 10")


# 쉬운 한국어로 다시 쓴 각주 (STYLE_I18N.md §2) — 숫자·조건은 그대로 두고 말만 바꾼다.
P2_FOOTNOTE = ("AUC 0.63~0.70: 위험이 커질지는 어느 정도 맞히지만 오를지 내릴지는 못 맞힙니다 · "
               "VIX 공식보다 나아진 부분은 대부분 치우침을 바로잡은 것입니다 · "
               "확률은 매일 참고만 하고, 주식을 얼마나 들고 갈지는 상태 규칙으로만 바꿉니다")


P2_FOOTNOTE_EN = ("AUC 0.63-0.70: how risky the next month is can be predicted in part, the direction cannot · "
                  "most of the gain over the VIX formula comes from fixing a bias · "
                  "read the probability every day, but change how much you hold in stocks only through the state rule")


P2_FAMILY_LINE = "VIX 를 실제와 맞춘 값에 조금 더"                   # §16 2 가족용 문구


P2_FAMILY_LINE_EN = "the VIX probability matched to reality, plus a little"


# §16 정직 문구·리스크 (리포트 ⑨ · 카드 정직 스트립에 상시) — 숫자·한계는 하나도 빼지 않는다.
P2_HONESTY_ITEMS = [
    "미리 정해둔 §6 규칙(모든 구간에서 이겨야 함)은 설계 때 이미 본 자료에서 후보가 전부 떨어졌습니다 "
    "(M3: 평소 평균 대비 2019-20 −0.038, 2023-24 −0.000; VIX 공식 B1 대비 2007-08 −0.082, 2017-18 −0.029; "
    "VIX 확률만 실제와 맞춘 것도 2019-20 에 떨어짐). 한 구간에 겹치지 않는 사례가 25개쯤뿐이라 점수가 ±0.1 정도 흔들립니다. "
    "그래서 '모든 구간에서 0보다 큼' 은 실력이 정말 +0.05~0.09 있어도 잡아내기 어렵습니다 — 그래도 규칙은 그대로 채점하고, "
    "완화안은 결과를 본 뒤 정한 것(#2a)이라 따로 표시합니다.",
    "VIX 공식보다 나아진 부분은 대부분 치우침을 바로잡은 것입니다 (M1 이 +0.19, M3 이 +0.20). "
    "실제로 움직인 폭과 추세를 더해 얻은 이득은 +13.6×1e-4 이고 95% 범위가 −2.8~31.4 라 0 을 품고 있습니다 — "
    "그래도 단계별 모델을 숨기지 않고 다 보여 줍니다. 가족에게 할 말은 'VIX 를 실제와 맞춘 값에 조금 더' 입니다.",
    "구조적으로 못 맞히는 자리: 조용하던 시장이 갑자기 무너질 때(2018-02, 2020-02)는 확률을 낮게 부르고, "
    "급락 뒤 되튀는 구간(2020-04~05, 2003)은 높게 부릅니다. '100일 중 며칠' 로 적고 평소 비율을 함께 적는 것은 "
    "덜 속게 해 줄 뿐, 이 약점을 없애지는 못합니다.",
    "손대지 않고 남겨둔 최근 구간은 겹치지 않는 사례가 24개쯤(2025-04 관세 급락 포함)이라 큰 실패만 걸러낼 수 있고 "
    "+1~2% 정도의 우위는 인증하지 못합니다. 진짜 시험은 Phase 3 실사용 기록입니다 (성적이 나쁘면 끄는 규칙 §7).",
    "자료: 1993~95 SPY 는 시가가 고가·저가와 같은 날이 30%이고 하루 움직임 폭도 절반 수준입니다"
    "(자기점검, 그리고 1996년부터 다시 맞춰 본 민감도로 확인). Yahoo 조정 종가를 다시 받아도 시가·고가·저가·종가의 비율과 "
    "추세 입력은 바뀌지 않지만, 어제 저장한 확률과 오늘 다시 계산한 확률의 차이가 0.01 을 넘으면 경고를 냅니다. "
    "^VIX 는 휴장일에 유령 행이 생깁니다.",
    "겹치는 라벨: 20거래일 앞을 보는 라벨이라 이웃한 날끼리 서로 겹칩니다. 그래서 모든 구간·검정·신뢰도의 표본 수는 "
    "구간 방법이나 n÷20 으로 셉니다 — 5,453 을 독립 관측 5,453개로 읽지 않도록 표마다 겹치지 않는 구간 수를 함께 적습니다.",
    "'reduce'(축소)는 VIX 가 50을 넘는 시장 분위기 밖에서는 거의 켜지지 않습니다 → 사실상 2단계로 움직인다는 뜻이라 그대로 적습니다.",
    "설계 단계에서 판정 규칙 구성 4종을 돌려 봤습니다 — 고른 구성은 지금 얼려 두고 같은 자료로 다시 손보지 않습니다.",
]


P2_HONESTY_ITEMS_EN = [
    "The rule fixed in advance (§6: win in every block) rejected every candidate on data we had already seen while "
    "designing it (M3: −0.038 in 2019-20 and −0.000 in 2023-24 against the long-run average; −0.082 in 2007-08 and "
    "−0.029 in 2017-18 against the VIX formula B1; even the VIX probability matched to reality on its own fails 2019-20). "
    "With only about 25 non-overlapping cases per block the score wobbles by about ±0.1, so 'above zero in every block' "
    "is unlikely to detect a real edge of +0.05 to +0.09 — we still score the rule exactly as written and mark the "
    "relaxed version (#2a) as something decided after seeing the results.",
    "Most of the gain over the VIX formula is a bias correction (M1 gives +0.19 of the +0.20 that M3 gives). "
    "Adding how much the market actually moved and the trend is worth +13.6×1e-4, and its 95% range of −2.8 to 31.4 "
    "still contains zero — we show every step anyway. What we tell the family: the VIX probability matched to reality, plus a little.",
    "Where it structurally fails: when a calm market breaks suddenly (2018-02, 2020-02) it calls the probability too low, "
    "and in the bounce right after a crash (2020-04 to 05, 2003) too high. Writing it as 'so many days out of 100' and "
    "showing the base rate next to it makes the number harder to misread, but it does not fix this weakness.",
    "The recent stretch we left untouched holds only about 24 non-overlapping cases (including the tariff drop of 2025-04), "
    "so it can reject a big failure but cannot certify an edge of +1 to +2%. The real test is the live record in Phase 3 "
    "(the rule that switches it off, §7).",
    "Data: in 1993-95 the SPY open equals the high or the low on 30% of days and the daily range is about half of what it "
    "later is (checked by the self-test and by re-fitting from 1996). Re-downloading the adjusted close does not change the "
    "open/high/low/close ratios or the trend input, but if today's recomputed probability differs from yesterday's stored "
    "one by more than 0.01 we raise a warning. ^VIX also has ghost rows on market holidays.",
    "Overlapping labels: the label looks 20 trading days ahead, so neighbouring days overlap. Every sample size for blocks, "
    "tests and reliability is therefore counted by the block method or as n/20 — each table also shows the number of "
    "non-overlapping windows so that 5,453 is never read as 5,453 independent observations.",
    "'reduce' almost never switches on outside a market mood with VIX above 50, so in practice this is a two-step system, "
    "and we say so.",
    "Four versions of the decision rule were simulated while designing it — the version we picked is frozen now and will "
    "not be re-tuned on the same data.",
]


P2_KILL_EPISODES = 8


P2_KILL_MONTHS = 36


_LADDER_ORDER = ["step", "from", "to", "block", "n", "n_blocks", "brier_from", "brier_to", "bss", "diff_mean", "diff_lo", "diff_hi",
                 "dm_t", "dm_p", "phase_min", "phase_median", "phase_max", "phase_share_pos"]


_BLOCK_ORDER = ["block", "start", "end", "first_session", "last_session", "n", "n_blocks", "n_pos", "base", "mean_p", "brier",
                "brier_clim", "brier_vix", "brier_vix_bgk", "brier_m1", "bss_clim", "bss_vix", "bss_vix_bgk", "bss_m1", "auc", "calib_in_large"]


_RUNG_ORDER = ["rung", "key", "n", "n_blocks", "n_pos", "base", "mean_p", "brier", "bss_clim", "bss_vix", "bss_vix_bgk", "bss_m1", "auc", "calib_in_large"]


_RELIAB_ORDER = ["bin", "lo", "hi", "n", "n_eff", "mean_p", "obs", "wilson_lo", "wilson_hi"]


_ERA_ORDER = ["era", "start", "end", "n_raw", "n_pos_raw", "auc_x_vix", "auc_x_har", "auc_neg_x_ma", "n_oos", "n_pos_oos", "auc_p_m1", "auc_p_m3"]


_PARAM_ORDER = ["refit_date", "rung", "intercept", "b_x_vix", "b_x_har", "b_x_ma", "clim", "n_train", "n_pos", "train_start", "train_end", "model_id"]


_HAR_ORDER = ["block", "start", "end", "n", "n_blocks", "mse_log", "qlike", "r2_log", "auc_vol", "mse_log_vix", "qlike_vix", "r2_log_vix",
              "auc_vol_vix", "mse_log_rv22", "qlike_rv22", "r2_log_rv22", "auc_vol_rv22"]


_ALLOC_KEYS = ["cagr", "max_dd", "worst_month", "total_return", "ann_vol", "n_switches", "switches_per_year", "avg_exposure", "cost_total",
               "n_days", "years", "start", "end"]


_DEC4_KEYS = ("brier", "mse", "qlike", "r2", "residual", "reliability", "resolution", "uncertainty")


_PCT_KEYS_P2 = ("base", "mean_p", "obs", "clim", "share", "occupancy", "cagr", "max_dd", "maxdd", "worst_month", "total_return",
                "ann_vol", "avg_exposure", "cost_total", "calib_in_large", "wilson", "rate", "dd5", "p_escalate", "p_deescalate",
                "excess_cagr", "maxdd_improvement", "missing_share", "true_alarm", "warn_share", "exposure", "detection", "depth")


LABELS.update({
    "bss_clim": "평소 평균보다 얼마나 정확", "bss_vix": "BSS vs VIX(B1)", "bss_vix_bgk": "BSS vs BGK", "bss_m1": "BSS vs M1", "bss": "BSS(단)",
    "brier": "Brier", "brier_clim": "평소 평균의 빗나간 정도", "brier_vix": "Brier VIX(B1)", "brier_vix_bgk": "Brier BGK", "brier_m1": "Brier M1",
    "brier_from": "Brier(기준)", "brier_to": "Brier(모델)", "auc": "AUC", "calib_in_large": "모델이 말한 평균 − 실제 비율", "mean_p": "평균 p",
    "n_pos": "양성 수", "block": "블록", "first_session": "첫 세션", "last_session": "마지막 세션", "step": "단", "from": "기준", "to": "모델",
    "diff_mean": "손실차 평균(×1e-4)", "diff_lo": "95% 하한(×1e-4)", "diff_hi": "95% 상한(×1e-4)", "dm_t": "DM t(HAC 19)", "dm_p": "DM p",
    "phase_min": "위상 BSS min", "phase_median": "위상 BSS median", "phase_max": "위상 BSS max", "phase_share_pos": "위상 >0 비율",
    "bin": "확률 구간", "lo": "하한", "hi": "상한", "n_eff": "n_eff(=n/20)", "obs": "실현 비율", "wilson_lo": "Wilson 하한", "wilson_hi": "Wilson 상한",
    "reliability": "reliability", "resolution": "resolution", "uncertainty": "uncertainty", "brier_binned": "Brier(구간 평균)", "residual": "잔차",
    "era": "시대", "n_raw": "행(전 이력)", "n_pos_raw": "양성(전 이력)", "auc_x_vix": "AUC x_vix", "auc_x_har": "AUC x_har", "auc_neg_x_ma": "AUC −x_ma",
    "n_oos": "행(OOS)", "n_pos_oos": "양성(OOS)", "auc_p_m1": "AUC p_m1", "auc_p_m3": "AUC p_m3",
    "refit_date": "재적합일", "rung": "단", "intercept": "절편 b0", "b_x_vix": "b1 x_vix", "b_x_har": "b2 x_har", "b_x_ma": "b3 x_ma",
    "clim": "기후학", "n_train": "학습 행", "train_start": "학습 시작", "train_end": "학습 끝", "model_id": "모델 id", "n_params": "파라미터 수",
    "mse_log": "MSE(log)", "qlike": "QLIKE", "r2_log": "R²(log)", "auc_vol": "AUC y_vol_20", "mse_log_vix": "MSE(log) VIX", "qlike_vix": "QLIKE VIX",
    "r2_log_vix": "R²(log) VIX", "auc_vol_vix": "AUC VIX", "mse_log_rv22": "MSE(log) RV22", "qlike_rv22": "QLIKE RV22", "r2_log_rv22": "R²(log) RV22",
    "auc_vol_rv22": "AUC RV22", "n_pos_vol": "고변동 수",
    "state": "상태", "days_in_state": "체류(세션)", "r": "r = p/clim", "churn_252": "252세션 변경 수", "churn_alert": "churn 경보",
    "occupancy": "점유", "n_by_state": "상태별 일수", "dd5_rate_by_state": "상태별 -5% 비율", "median_run_by_state": "상태별 중앙 런",
    "n_changes": "변경 수", "kpi_ceiling": "KPI 상한(회/년)", "kpi_ceiling_ok": "KPI 상한 충족", "max_changes_any_5_sessions": "5세션 최대 변경",
    "structural_bound": "구조적 상한", "structural_bound_ok": "구조적 상한 충족", "churn_alert_sessions": "churn 경보 세션", "churn_alert_any": "churn 경보 발생",
    "max_churn_252": "최대 252세션 변경", "n_missing": "결측 세션", "missing_share": "결측 비중", "true_alarm_edge": "진짜 경보 − 기저율",
    "true_alarm_share_v0_ref": "v0 진짜 경보 비중(참조)", "median_warn_run": "중앙 경고 런(세션)", "n_sessions": "세션 수",
    "deploy_mode": "배포 모드", "tone_model": "톤 모델", "rule": "규칙", "pass": "통과", "pass_clim": "기후학 통과", "pass_vix": "VIX 통과",
    "failing_blocks": "실패 블록", "failing_clim": "실패(기후학)", "failing_vix": "실패(VIX)", "min_bss_clim": "최소 BSS 기후학", "min_bss_vix": "최소 BSS VIX",
    "A": "A", "B": "B", "C": "C", "pooled_bss_clim": "전체 BSS 기후학", "pooled_bss_vix": "전체 BSS VIX",
    "ci_lo_clim": "CI 하한(clim→M, ×1e-4)", "ci_lo_vix": "CI 하한(B1→M, ×1e-4)", "ci_lo_info": "CI 하한(정보 단, ×1e-4)", "info_step": "정보 단", "n_blocks_clim_pos": "블록 >0 (기후학)",
    "n_blocks_vix_nonneg": "블록 ≥0 (VIX)", "min_pass_blocks": "필요 블록 수", "min_block_bss_clim": "최소 블록 BSS 기후학", "n_blocks_empty": "빈 블록",
    "min_block_clim": "최소 블록(기후학)", "table": "블록 표", "required": "판정에 사용", "passing_rungs": "통과 단",
    "require_tables": "요구 표", "primary_table": "주 표", "blocking_tables": "막은 표", "verdict_line": "배치 판정",
    "alone_tone": "이 표 단독 톤 모델", "alone_deploy": "이 표 단독 결론", "candidate": "사전 후보 단",
    "literal_tone_model": "literal 톤 모델 후보(주 표)", "amended_tone_model": "amended 톤 모델 후보(주 표)",
    "avg_exposure": "평균 비중", "cost_total": "누적 비용", "ann_vol": "변동성(연)", "max_dd_date": "MaxDD 일자", "worst_month_label": "최악 월",
    "bh_cagr": "보유 CAGR", "bh_max_dd": "보유 MaxDD", "bh_worst_month": "보유 최악 월", "bh_total_return": "보유 총수익", "bh_ann_vol": "보유 변동성",
    "n_scored": "채점 행", "ci_bss_clim": "BSS 기후학 95% 구간", "episodes5_observed": "실현 ≥5% 에피소드",
    "months_elapsed": "경과 개월", "kill_rule_due": "킬룰 평가 시점", "p_m1": "p M1", "p_m2": "p M2", "p_m3": "p M3", "p_vix": "p VIX(B1)",
    "p_vix_bgk": "p BGK", "p_vix_driftless": "p 무드리프트", "x_vix": "x_vix", "x_har": "x_har", "x_ma": "x_ma", "har_vol_20": "HAR 고정가중 σ",
    "har_fc_20": "HAR 예측 σ(20일)", "rv20_cc": "실현변동성 20일", "vix": "VIX", "ts_diag": "VIX/VIX3M", "pct": "10년 백분위", "note": "해설",
    "kind": "종류", "date": "날짜", "sessions_ahead": "남은 세션", "label_ko": "이벤트", "tentative": "잠정",
    # (열 제목의 영어 짝은 BI_LABELS 에 둔다)
    "sensitivity": "민감도", "config": "구성", "name": "이름", "verdict": "판정", "flags": "플래그", "value": "값",
    "role": "역할", "occupancy_caution": "점유 caution", "occupancy_reduce": "점유 reduce", "prob_rung": "확률 단",
    "true_alarm_share_baseline": "진짜 경보 기저율", "deployed": "배포됨", "prob_column": "확률 열",
})


# ------------------------------------------------------------------
# P2 포매팅
# ------------------------------------------------------------------
def _fnum(v) -> float:
    """숫자로 해석(불가·결측이면 NaN). 지어내지 않는다."""
    if v is None or isinstance(v, (bool, np.bool_)):
        return math.nan
    try:
        f = float(v)
    except (TypeError, ValueError):
        return math.nan
    return f


def _first(d: dict, *keys, default=None):
    """dict 에서 후보 키 중 처음으로 결측이 아닌 스칼라를 돌려준다."""
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and _scalar_ok(d[k]):
            return d[k]
    return default


def _pct1(v, digits: int = 1) -> str:
    f = _fnum(v)
    return "—" if math.isnan(f) else f"{f * 100:.{digits}f}%"


def _pp1(v, digits: int = 1) -> str:
    """확률 차이 → '+1.3pp' (입력은 확률 단위)."""
    f = _fnum(v)
    return "—" if math.isnan(f) else f"{f * 100:+.{digits}f}pp"


def _nat_freq(p) -> str:
    """자연빈도: p → '100일 중 약 N일'."""
    f = _fnum(p)
    return "—" if math.isnan(f) else f"100일 중 약 {int(round(f * 100))}일"


def _ten_freq(p) -> str:
    f = _fnum(p)
    return "—" if math.isnan(f) else f"10번 중 {int(round(f * 10))}번"


def _ten_freq_en(p) -> str:
    """자연빈도(영문 평문): p → 'about 2 in 10'. 평문 리스트·부속 설명 전용(스팬은 호출자가 만든다)."""
    f = _fnum(p)
    return "—" if math.isnan(f) else f"about {int(round(f * 10))} in 10"


def _exact_pp_parts(parts: dict, total, nd: int = 1) -> tuple[dict, float]:
    """pp 항(확률 단위 dict)을 소수 nd 자리 pp 로 반올림하되, 표시된 항의 합이 표시된 합계와 **정확히** 같도록
    반올림 잔차를 |값| 이 가장 큰 항에 얹는다. 반환 ({항: pp}, 합계 pp). 결측 항은 NaN 그대로(합에서 제외)."""
    tot = _fnum(total)
    vals = {k: _fnum(v) for k, v in (parts or {}).items()}
    fin = {k: v for k, v in vals.items() if not math.isnan(v)}
    if math.isnan(tot):
        # 합계가 없으면 모든 항이 유한할 때만 항의 합으로 복원한다 — 결측 항이 있으면 합계를 지어내지 않는다
        tot = sum(fin.values()) if (fin and len(fin) == len(vals)) else math.nan
    if math.isnan(tot):
        return {k: math.nan for k in vals}, math.nan
    tot_pp = round(tot * 100.0, nd)
    r = {k: round(v * 100.0, nd) for k, v in fin.items()}
    resid = round(tot_pp - sum(r.values()), nd)
    if r and abs(resid) >= 10 ** (-nd) / 2.0:
        k = max(r, key=lambda kk: abs(fin[kk]))
        r[k] = round(r[k] + resid, nd)
    out = {k: (r[k] if k in r else math.nan) for k in vals}
    return out, tot_pp


def _p2_state_pill(s, grey: bool = False, bi_gloss: bool = True) -> str:
    """상태 알약. 식별자(normal·caution·reduce)는 그대로, 뜻풀이만 화면 언어를 따른다.

    bi_gloss=False 는 표의 셀 값으로 쓸 때 (STYLE_I18N.md §1: 셀 값은 두 벌로 심지 않는다).
    """
    s = "" if s is None else str(s)
    c = PALETTE["mut"] if grey else P2_STATE_COLORS.get(s, PALETTE["mut"])
    ko, en = P2_STATE_KO.get(s, ""), P2_STATE_EN.get(s, "")
    gloss = (bi(ko, en) if bi_gloss else _esc(ko)) if (ko or en) else ""
    return f'<span class="st" style="background:{c}">{_esc(s)} {gloss}</span>'


def _fmt_p2(v, key: str = "") -> str:
    """P2 표 값 포맷: Brier/MSE 4자리, BSS/AUC 3자리, 손실차(diff_*) ×1e-4, 계수 ±3자리, 확률·비율 백분율, 상태 pill."""
    k = str(key or "").lower()
    if _is_nan(v):
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return "예" if v else "아니오"
    if isinstance(v, str):
        if k in ("state", "p2_state", "last_state", "escalate_to", "deescalate_to") and v in P2_STATE_COLORS:
            return _p2_state_pill(v, bi_gloss=False)   # 표 셀 값 — 두 벌로 심지 않는다
        return _fmt(v, key)
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(_is_num(x) for x in v):
        return f"[{_fmt_p2(v[0], key)}, {_fmt_p2(v[1], key)}]"
    if _is_num(v) and not isinstance(v, (int, np.integer)):
        f = float(v)
        if math.isinf(f):
            return "∞" if f > 0 else "-∞"
        if k == "n" or k.startswith("n_") or k.endswith("_n") or "days" in k or "sessions" in k:
            return f"{int(round(f)):,}" if abs(f - round(f)) < 1e-9 else f"{f:,.1f}"
        if any(k.startswith(s) for s in _DEC4_KEYS):
            return f"{f:.4f}"
        # 손실차와 그 신뢰구간 한계는 같은 단위(×1e-4)로 — ci_lo_clim/ci_lo_vix/ci_lo_info 는 배치를 가르는 수라
        # 백분율 1자리로 찍으면 -1.25e-4 가 "-0.0%" 가 되어 0 과 구별되지 않는다(사다리 표의 -1.3 과도 대조 불가).
        if k.startswith("diff_") or k.startswith("ci_lo_") or k.startswith("ci_hi_"):
            return f"{f * 1e4:+.1f}"
        # BSS 는 어디에 있든 BSS: min_/pooled_/min_block_/mean_ 접두 aggregates 도 같은 단위로 찍는다
        if "bss" in k or k.startswith("edge") or (k.startswith("phase_") and "share" not in k):
            return f"{f:+.3f}"
        if k.startswith("auc"):
            return f"{f:.3f}"
        if k.startswith("dm_t"):
            return f"{f:+.2f}"
        if k.startswith("dm_p") or k == "p_value":
            return f"{f:.3f}"
        if k.startswith("coef") or k in ("intercept", "b0", "b1", "b2", "b3") or k.startswith("b_") or k.startswith("x_") or "logit" in k:
            return f"{f:+.3f}"
        if k in ("r", "r_escalate", "r_deescalate") or k.startswith("churn") or k.startswith("max_churn"):
            return f"{f:.2f}"
        if "months" in k or "years" in k or "switches" in k or k.startswith("median_run") or k.startswith("median_warn"):
            return f"{f:,.1f}"
        if any(s in k for s in _PCT_KEYS_P2) or k.startswith("p_") or k in ("p", "lo", "hi", "prob", "prob_dd5_20") or k.startswith("p2_p"):
            return f"{f * 100:.1f}%"
    return _fmt(v, key)


def _p2_table(records, order=None, empty_msg="자료 없음") -> str:
    return _records_table(records, order, empty_msg, fmt=_fmt_p2)


def _p2_kv(d, empty_msg="자료 없음") -> str:
    return _kv_table(d, empty_msg, fmt=_fmt_p2)


# ==================================================================
# 한/영 표현 도구 (STYLE_I18N.md §1·§2) — 이 파일의 두 주간 페이지 전용
# ==================================================================
# 왜 여기에 또 만드나: _p2_table·_p2_kv·_v0_line_html 은 report_p3 와 index 카드가 함께 쓴다.
# 그 함수들을 이중 언어로 바꾸면 남의 페이지까지 바뀌므로, **표현만 바꾼 짝**을 새로 둔다.
# 규칙: 열 제목·문장·라벨만 두 언어로 심고, 셀 값(숫자·날짜·티커·식별자)은 건드리지 않는다.
#       계산이 만든 한국어 문장(경고·판정 원문)은 _raw() 로 **그대로** 인용하고, 옆에 쉬운 말/영어 설명을 붙인다.
_HANGUL_TXT = re.compile(r"[가-힣]")


class _Html(str):
    """이미 HTML 인 셀 값 — 표가 다시 포맷하지 않고 그대로 내보낸다."""


def _raw(text) -> str:
    """계산이 남긴 원문(한국어)을 손대지 않고 인용한다. 언어 중립이라 두 화면에 모두 보인다."""
    return f'<span class="raw mono">{_esc(text)}</span>'


# 열 제목·항목 이름: 원문 라벨 → (쉬운 한국어, English). 여기 없는 키는 LABELS 의 한국어와 키 이름을 쓴다.
BI_LABELS: dict[str, tuple[str, str]] = {
    # ── 표본·구간 ────────────────────────────────────────────────
    "n": ("관측 일수", "days observed"),
    # ── 이벤트 표(일간 카드) ────────────────────────────────────
    "date": ("날짜", "date"), "kind": ("종류", "kind"),
    "sessions_ahead": ("남은 거래일", "trading days ahead"),
    "label_ko": ("무슨 일정인지 (한국어 원문)", "what is scheduled (original Korean)"),
    "tentative": ("아직 확정 아님", "not confirmed yet"),
    "n_days": ("관측 일수", "days observed"),
    "n_sessions": ("거래일 수", "trading days"),
    "n_blocks": ("겹치지 않는 구간 수 (n÷20)", "independent windows (n/20)"),
    "n_eff": ("독립 사례 수 (n÷20)", "independent cases (n/20)"),
    "n_pos": ("실제로 일어난 횟수", "times it actually happened"),
    "n_pos_raw": ("실제로 일어난 횟수 (전 기간)", "times it happened (full history)"),
    "n_pos_oos": ("실제로 일어난 횟수 (시험 구간)", "times it happened (test rows)"),
    "n_pos_vol": ("크게 움직인 날 수", "days that moved a lot"),
    "n_raw": ("행 수 (전 기간)", "rows (full history)"),
    "n_oos": ("행 수 (시험 구간)", "rows (test only)"),
    "n_train": ("학습에 쓴 행 수", "rows used to fit"),
    "n_scored": ("채점한 행 수", "rows scored"),
    "n_params": ("조정한 숫자 개수", "numbers we tuned"),
    "n_shift": ("무작위로 밀어 본 횟수", "random shifts tried"),
    "start": ("시작", "start"), "end": ("끝", "end"),
    "first_session": ("첫 거래일", "first session"), "last_session": ("마지막 거래일", "last session"),
    "train_start": ("학습 시작", "training starts"), "train_end": ("학습 끝", "training ends"),
    "block": ("구간", "block"), "table": ("구간 표", "block table"), "era": ("시대", "period"),
    "years": ("햇수", "years"), "variant": ("자료 방식", "data variant"), "key": ("구분", "item"),
    "name": ("이름", "name"), "note": ("설명", "note"), "value": ("값", "value"), "role": ("역할", "role"),
    "rule": ("규칙", "rule"), "verdict": ("판정", "verdict"), "flags": ("표시", "flags"),
    "config": ("설정", "settings"), "sensitivity": ("설정을 바꿔 본 결과", "settings changed"),
    # ── 확률·정확도 ─────────────────────────────────────────────
    "base": ("실제로 일어난 비율", "how often it actually happened"),
    "base_rate": ("평소 비율", "base rate (how often it normally happens)"),
    "clim": ("평소 평균", "the long-run average"),
    "mean_p": ("모델이 말한 평균 확률", "average probability the model stated"),
    "obs": ("실제로 일어난 비율", "how often it actually happened"),
    "bin": ("확률 구간", "probability band"),
    "lo": ("아래끝", "low"), "hi": ("위끝", "high"),
    "wilson_lo": ("95% 아래끝 (Wilson)", "95% low (Wilson)"), "wilson_hi": ("95% 위끝 (Wilson)", "95% high (Wilson)"),
    "brier": ("빗나간 정도 점수 (Brier)", "error score (Brier, lower is better)"),
    "brier_binned": ("빗나간 정도 (구간 평균)", "error score (bin average)"),
    "brier_clim": ("평소 평균의 빗나간 정도", "error score of the long-run average"),
    "brier_vix": ("VIX 공식의 빗나간 정도 (B1)", "error score of the VIX formula (B1)"),
    "brier_vix_bgk": ("BGK 공식의 빗나간 정도", "error score of the BGK formula"),
    "brier_m1": ("M1 의 빗나간 정도", "error score of M1"),
    "brier_from": ("기준 쪽 빗나간 정도", "error score of the baseline"),
    "brier_to": ("모델 쪽 빗나간 정도", "error score of the model"),
    "bss": ("이 단계에서 좋아진 정도", "improvement at this step"),
    "bss_clim": ("평소 평균보다 얼마나 정확", "accuracy gain vs the long-run average"),
    "bss_vix": ("VIX 공식보다 얼마나 정확 (B1)", "accuracy gain vs the VIX formula (B1)"),
    "bss_vix_bgk": ("BGK 공식보다 얼마나 정확", "accuracy gain vs the BGK formula"),
    "bss_m1": ("M1 보다 얼마나 정확", "accuracy gain vs M1"),
    "ci_bss_clim": ("평소 평균 대비 정확도 95% 범위", "95% range of the gain vs the long-run average"),
    "auc": ("위험한 날을 앞줄에 세우는 힘 (AUC)", "power to rank risky days first (AUC)"),
    "auc_x_vix": ("VIX 입력의 줄 세우기 힘", "ranking power of the VIX input"),
    "auc_x_har": ("실제-예상 갭 입력의 줄 세우기 힘", "ranking power of the realised-minus-implied input"),
    "auc_neg_x_ma": ("추세 입력의 줄 세우기 힘", "ranking power of the trend input"),
    "auc_p_m1": ("M1 확률의 줄 세우기 힘", "ranking power of the M1 probability"),
    "auc_p_m3": ("M3 확률의 줄 세우기 힘", "ranking power of the M3 probability"),
    "auc_vol": ("크게 움직일 날 가려내는 힘", "power to spot high-movement days"),
    "auc_vol_vix": ("같은 힘 — VIX 만 썼을 때", "same power — VIX alone"),
    "auc_vol_rv22": ("같은 힘 — 지난 22일만 썼을 때", "same power — last 22 days alone"),
    "calib_in_large": ("모델이 말한 평균 − 실제 비율", "average stated % minus actual %"),
    "residual": ("남은 차이", "left-over gap"),
    "reliability": ("말한 확률과 실제의 어긋남", "gap between the stated % and reality"),
    "resolution": ("상황을 갈라내는 힘", "how much it separates situations"),
    "uncertainty": ("사건 자체의 불확실함", "the event's own uncertainty"),
    # ── 단계별 모델(사다리)·검정 ────────────────────────────────
    "step": ("비교 단계", "step"), "from": ("기준", "baseline"), "to": ("모델", "model"),
    "rung": ("단계", "step"), "candidate": ("미리 정해둔 후보 단계", "the step chosen in advance"),
    "info_step": ("정보가 늘어난 단계", "the step that adds information"),
    "diff_mean": ("빗나간 정도가 줄어든 평균 (×1e-4)", "average error reduction (×1e-4)"),
    "diff_lo": ("95% 범위 아래끝 (×1e-4)", "95% range low (×1e-4)"),
    "diff_hi": ("95% 범위 위끝 (×1e-4)", "95% range high (×1e-4)"),
    "ci_lo_clim": ("95% 아래끝 — 평소 평균 대비 (×1e-4)", "95% low vs the long-run average (×1e-4)"),
    "ci_lo_vix": ("95% 아래끝 — VIX 공식 대비 (×1e-4)", "95% low vs the VIX formula (×1e-4)"),
    "ci_lo_info": ("95% 아래끝 — 바로 아래 단계 대비 (×1e-4)", "95% low vs the step below (×1e-4)"),
    "dm_t": ("우연 아닌지 보는 값 (DM t(HAC 19))", "test statistic that the gap is not luck (DM t, HAC 19)"),
    "dm_p": ("그 값이 우연일 확률 (DM p)", "chance of seeing that by luck (DM p)"),
    "phase_min": ("시작일을 20가지로 바꿔 본 최솟값", "worst of 20 start-day offsets"),
    "phase_median": ("시작일 20가지의 가운데값", "median of 20 start-day offsets"),
    "phase_max": ("시작일 20가지의 최댓값", "best of 20 start-day offsets"),
    "phase_share_pos": ("시작일 20가지 중 0보다 큰 비율", "share of the 20 offsets above zero"),
    "mean_bss_clim_blocks": ("구간별 정확도 평균 (평소 평균 대비)", "average gain across blocks (vs the long-run average)"),
    "mean_bss_vix_blocks": ("구간별 정확도 평균 (VIX 공식 대비)", "average gain across blocks (vs the VIX formula)"),
    "pooled_bss_clim": ("전체를 합친 정확도 (평소 평균 대비)", "gain over all rows (vs the long-run average)"),
    "pooled_bss_vix": ("전체를 합친 정확도 (VIX 공식 대비)", "gain over all rows (vs the VIX formula)"),
    "min_bss_clim": ("가장 나쁜 구간 점수 (평소 평균 대비)", "worst block score (vs the long-run average)"),
    "min_bss_vix": ("가장 나쁜 구간 점수 (VIX 공식 대비)", "worst block score (vs the VIX formula)"),
    "min_block_bss_clim": ("가장 나쁜 구간 점수 (평소 평균 대비)", "worst block score (vs the long-run average)"),
    "min_block_clim": ("가장 나쁜 구간", "the worst block"),
    "n_blocks_clim_pos": ("0보다 큰 구간 수 (평소 평균 대비)", "blocks above zero (vs the long-run average)"),
    "n_blocks_vix_nonneg": ("0 이상인 구간 수 (VIX 공식 대비)", "blocks at or above zero (vs the VIX formula)"),
    "n_blocks_empty": ("채점할 수 없던 구간 수", "blocks that could not be scored"),
    "min_pass_blocks": ("통과에 필요한 구간 수", "blocks required to pass"),
    "failing_blocks": ("떨어진 구간", "blocks that failed"),
    "failing_clim": ("떨어진 구간 (평소 평균 대비)", "blocks that failed (vs the long-run average)"),
    "failing_vix": ("떨어진 구간 (VIX 공식 대비)", "blocks that failed (vs the VIX formula)"),
    "pass": ("통과", "passed"),
    "pass_clim": ("평소 평균 기준 통과", "passed vs the long-run average"),
    "pass_vix": ("VIX 공식 기준 통과", "passed vs the VIX formula"),
    "passing_rungs": ("통과한 단계", "steps that passed"),
    "required": ("판정에 썼는지", "used for the verdict?"),
    "primary_table": ("첫 번째 표", "first table"),
    "require_tables": ("통과해야 하는 표", "tables that must pass"),
    "blocking_tables": ("막은 표", "tables that blocked it"),
    "verdict_line": ("실제 사용 판정", "deployment verdict"),
    "alone_tone": ("이 표 하나만 봤을 때 고를 단계", "step this table alone would pick"),
    "alone_deploy": ("이 표 하나만 봤을 때 결론", "conclusion from this table alone"),
    "literal_tone_model": ("미리 정한 규칙이 고른 단계", "step picked by the pre-registered rule"),
    "amended_tone_model": ("완화안이 고른 단계", "step picked by the relaxed rule"),
    "tone_model": ("신호등 판정을 만드는 단계", "the step that drives the traffic-light call"),
    "deploy_mode": ("실제로 쓰는지", "whether it is in use"),
    "deployed": ("실제 사용 중", "in use"),
    "prob_rung": ("확률을 만든 단계", "the step the probability comes from"),
    "prob_column": ("쓴 확률 열", "probability column used"),
    "A": ("A 기준", "test A"), "B": ("B 기준", "test B"), "C": ("C 기준", "test C"),
    # ── 계수·모델 ───────────────────────────────────────────────
    "refit_date": ("다시 맞춘 날", "refit date"),
    "intercept": ("기본값 b0", "intercept b0"),
    "b_x_vix": ("VIX 계수 b1", "VIX coefficient b1"),
    "b_x_har": ("실제-예상 갭 계수 b2", "realised-minus-implied coefficient b2"),
    "b_x_ma": ("추세 계수 b3", "trend coefficient b3"),
    "x_vix": ("VIX 입력", "VIX input"),
    "x_har": ("실제-예상 갭 입력", "realised-minus-implied input"),
    "x_ma": ("추세 입력", "trend input"),
    "model_id": ("모델 번호", "model id"),
    # ── 변동성 보조(HAR) ────────────────────────────────────────
    "mse_log": ("빗나간 정도 (log MSE)", "error (log MSE)"),
    "mse_log_vix": ("빗나간 정도 — VIX (log MSE)", "error — VIX (log MSE)"),
    "mse_log_rv22": ("빗나간 정도 — 지난 22일 (log MSE)", "error — last 22 days (log MSE)"),
    "qlike": ("또 다른 빗나감 점수 (QLIKE)", "another error score (QLIKE)"),
    "qlike_vix": ("같은 점수 — VIX", "same score — VIX"),
    "qlike_rv22": ("같은 점수 — 지난 22일", "same score — last 22 days"),
    "r2_log": ("설명한 몫 (R²(log))", "share explained (R², log)"),
    "r2_log_vix": ("설명한 몫 — VIX", "share explained — VIX"),
    "r2_log_rv22": ("설명한 몫 — 지난 22일", "share explained — last 22 days"),
    "purge": ("겹치는 구간 제거 (일)", "overlapping days removed"),
    # ── 판정 규칙(상태 기계)·성적 ───────────────────────────────
    "state": ("상태", "state"),
    "occupancy": ("머문 날 비율", "share of days spent here"),
    "occupancy_caution": ("caution 에 머문 비율", "share of days in caution"),
    "occupancy_reduce": ("reduce 에 머문 비율", "share of days in reduce"),
    "n_by_state": ("상태별 날 수", "days in each state"),
    "dd5_rate": ("그 상태에서 20일 안에 5% 넘게 떨어진 비율", "share that fell more than 5% within 20 days"),
    "dd5_rate_by_state": ("상태별 5% 하락 비율", "5% fall rate by state"),
    "median_run": ("한 번 켜지면 가운데 며칠", "median days per run"),
    "median_run_len": ("한 번 켜지면 가운데 며칠", "median days per run"),
    "median_run_by_state": ("상태별 가운데 유지 일수", "median run length by state"),
    "median_warn_run": ("경고가 켜지면 가운데 며칠", "median days per warning run"),
    "median_warn_run_len": ("경고가 켜지면 가운데 며칠", "median days per warning run"),
    "n_warn_runs": ("경고가 켜진 횟수", "number of warning runs"),
    "warn_share": ("경고가 켜져 있던 날 비율", "share of days with a warning on"),
    "n_changes": ("상태가 바뀐 횟수", "state changes"),
    "n_switches": ("상태가 바뀐 횟수", "state changes"),
    "n_tone_switches": ("판정이 바뀐 횟수", "call changes"),
    "switches_per_year": ("1년에 상태가 바뀐 횟수", "state changes per year"),
    "tone_switches_per_year": ("1년에 판정이 바뀐 횟수", "call changes per year"),
    "kpi_ceiling": ("정해 둔 상한 (회/년)", "the ceiling we set (per year)"),
    "kpi_ceiling_ok": ("상한을 지켰는지", "ceiling respected"),
    "max_changes_any_5_sessions": ("5거래일 안 최대 변경 수", "most changes within any 5 sessions"),
    "structural_bound": ("규칙상 넘을 수 없는 수", "the most the rule allows"),
    "structural_bound_ok": ("그 한도를 지켰는지", "within that limit"),
    "structural_window": ("그 한도를 세는 창 (거래일)", "window for that limit (sessions)"),
    "churn_alert": ("잦은 전환 경보", "churn alert"),
    "churn_alert_any": ("잦은 전환 경보가 켜진 적 있는지", "churn alert ever on"),
    "churn_alert_sessions": ("잦은 전환 경보가 켜진 날 수", "days with the churn alert on"),
    "churn_alert_threshold": ("잦은 전환 경보 기준", "churn alert threshold"),
    "churn_window": ("잦은 전환을 세는 창 (거래일)", "churn window (sessions)"),
    "max_churn_252": ("최근 252거래일 최대 변경 수", "most changes in any 252 sessions"),
    "n_missing": ("빠진 거래일 수", "missing sessions"),
    "missing_share": ("빠진 날 비율", "share of missing days"),
    "enter_caution": ("caution 으로 올라가는 문턱", "threshold to enter caution"),
    "exit_caution": ("caution 에서 내려오는 문턱", "threshold to leave caution"),
    "enter_reduce": ("reduce 로 올라가는 문턱", "threshold to enter reduce"),
    "exit_reduce": ("reduce 에서 내려오는 문턱", "threshold to leave reduce"),
    "dwell": ("내려오기 전 최소 유지 일수", "minimum days before switching back"),
    "dwell_escalate": ("올라갈 때의 최소 유지 일수", "minimum days before switching up"),
    "true_alarm_share": ("경고가 진짜였던 비율", "share of warnings that were real"),
    "true_alarm_share_baseline": ("아무 날이나 골라도 나오는 비율", "what picking any day would give"),
    "true_alarm_share_mean": ("진짜 경보 비율 평균", "average share of real warnings"),
    "true_alarm_edge": ("진짜 경보 비율 − 평소 비율", "real-warning share minus the base rate"),
    "true_alarm_share_v0_ref": ("v0 의 진짜 경보 비율 (참고)", "v0 real-warning share (reference)"),
    "n_true_alarms": ("진짜 경보 수", "real warnings"),
    "n_false_alarms": ("헛경보 수", "false alarms"),
    "n_unresolved_alarms": ("아직 판정 못 한 경보 수", "warnings not yet resolved"),
    "false_alarm_rate": ("헛경보 비율", "false-alarm rate"),
    "false_alarm_rate_baseline": ("아무 날이나 경고라 했을 때의 헛경보 비율", "false-alarm rate if any day were called a warning"),
    "false_alarms_per_year": ("1년에 헛경보 횟수", "false alarms per year"),
    "null": ("무작위로 밀어 본 기준선", "randomly shifted baseline"),
    "detection_rate": ("미리 잡아낸 비율", "share caught in advance"),
    "detection_rate_mean": ("미리 잡아낸 비율 평균", "average share caught in advance"),
    "detection_100_share": ("전부 잡아낸 경우의 비율", "share of shifts that caught them all"),
    "n_detected": ("잡아낸 수", "caught"), "n_missed": ("놓친 수", "missed"),
    "n_episodes": ("하락 사건 수", "decline episodes"),
    "n_evaluable": ("채점할 수 있는 하락 사건 수", "episodes we can score"),
    "evaluable": ("채점 가능", "can be scored"),
    "episodes5_observed": ("5% 넘게 떨어진 사건 수", "declines deeper than 5% so far"),
    "peak_date": ("고점 날짜", "peak date"), "trough_date": ("바닥 날짜", "trough date"),
    "depth": ("고점에서 떨어진 폭", "drop from the peak"),
    "days_to_trough": ("고점에서 바닥까지 (일)", "days from peak to trough"),
    "recovery_date": ("회복한 날", "recovery date"),
    "days_to_recover": ("회복까지 (일)", "days to recover"),
    "warn_date": ("첫 경고일", "first warning"),
    "warn_run_start": ("그 경고가 켜진 날", "day that warning turned on"),
    "warn_run_age_at_peak": ("고점 때 경고가 켜진 지 며칠", "days the warning had been on at the peak"),
    "warn_frac_to_trough": ("바닥까지 경고가 켜져 있던 몫", "share of the fall with the warning on"),
    "lead_days": ("고점보다 며칠 먼저 경고 (+ = 먼저)", "days of warning before the peak (+ = earlier)"),
    "lead_capped": ("탐색 창 끝에 걸렸는지", "hit the edge of the search window"),
    "mean_lead_days": ("평균 며칠 먼저", "average days of warning"),
    "median_lead_days": ("가운데값 며칠 먼저", "median days of warning"),
    "median_lead_days_fresh": ("새로 켜진 경고만 며칠 먼저", "median days, freshly turned-on warnings only"),
    "median_lead_days_p5": ("며칠 먼저 — 아래쪽 5%", "days of warning — 5th percentile"),
    "median_lead_days_p50": ("며칠 먼저 — 가운데", "days of warning — median"),
    "median_lead_days_p95": ("며칠 먼저 — 위쪽 95%", "days of warning — 95th percentile"),
    "n_lead_positive": ("고점 전에 켜진 경고 수", "warnings that came before the peak"),
    "n_lead_capped": ("탐색 창 끝에 걸린 수", "warnings that hit the window edge"),
    "n_held_to_trough": ("바닥까지 경고를 유지한 수", "warnings held to the trough"),
    "held_to_trough": ("바닥까지 경고 유지", "warning held to the trough"),
    "missed": ("놓침", "missed"),
    "lookback": ("고점 앞을 몇 거래일까지 볼지", "how many sessions before the peak we look"),
    "tone_at_peak": ("고점에서의 신호등 판정", "traffic-light call at the peak"),
    "tone_at_trough": ("바닥에서의 신호등 판정", "traffic-light call at the trough"),
    "kill_rule_due": ("성적이 나쁘면 끄는 규칙 — 평가 시점", "the switch-off rule — time to check"),
    "months_elapsed": ("지난 개월", "months elapsed"),
    # ── 배분 성적 ───────────────────────────────────────────────
    "cagr": ("1년 평균 수익률 (CAGR)", "return per year (CAGR)"),
    "max_dd": ("고점 대비 가장 큰 하락 (MaxDD)", "biggest drop from the peak (max drawdown)"),
    "max_dd_date": ("그 하락이 바닥친 날", "date of that biggest drop"),
    "worst_month": ("가장 나빴던 달", "worst month"),
    "worst_month_label": ("가장 나빴던 달", "worst month"),
    "total_return": ("전체 기간 수익", "total return"),
    "ann_vol": ("1년 기준 흔들린 폭", "yearly swing size"),
    "avg_exposure": ("평균적으로 주식을 얼마나 들고 있었나", "average share held in stocks"),
    "cost_total": ("매매 비용 합계", "total trading cost"),
    "bh_cagr": ("그냥 들고 있었을 때 1년 수익률", "buy-and-hold return per year"),
    "bh_max_dd": ("그냥 들고 있었을 때 최대 하락", "buy-and-hold biggest drop"),
    "bh_worst_month": ("그냥 들고 있었을 때 가장 나쁜 달", "buy-and-hold worst month"),
    "bh_total_return": ("그냥 들고 있었을 때 전체 수익", "buy-and-hold total return"),
    "bh_ann_vol": ("그냥 들고 있었을 때 흔들린 폭", "buy-and-hold swing size"),
    # ── ⑨ 실행·자기점검·캐시 (식별자만 있던 칸에 사람 말로 이름을 붙인다) ──
    "python": ("파이썬 버전", "Python version"), "numpy": ("numpy 버전", "numpy version"),
    "pandas": ("pandas 버전", "pandas version"), "scipy": ("scipy 버전", "scipy version"),
    "sklearn": ("scikit-learn 버전", "scikit-learn version"), "platform": ("실행한 컴퓨터", "machine it ran on"),
    "yfinance_version": ("yfinance 버전", "yfinance version"), "git_sha": ("코드 버전 지문", "code version fingerprint"),
    "seed": ("난수 씨앗", "random seed"), "schema_version": ("산출물 형식 번호", "artifact format version"),
    "generated_at_utc": ("만든 시각 (UTC)", "generated at (UTC)"), "fetched_at_utc": ("자료 받은 시각 (UTC)", "data fetched at (UTC)"),
    "spec_sha256": ("설계 지문", "spec fingerprint"), "data_sha256": ("자료 지문", "data fingerprint"),
    "oos_csv_sha256": ("채점표 파일 지문", "scored-rows file fingerprint"),
    "live_data_sha256": ("지금 쓰는 자료의 지문", "fingerprint of the data in use"),
    "signals_v0_sha256": ("예전 v0 신호 코드의 지문", "fingerprint of the old v0 signal code"),
    "spec_changed": ("설계가 바뀌었는지", "did the spec change"),
    "feature_rule": ("입력을 만드는 규칙", "the rule that builds the inputs"),
    "window_rule": ("창을 정하는 규칙", "the rule that sets the window"),
    "source_csv": ("불러온 파일", "source file"),
    "first_refit": ("처음 다시 맞춘 날", "first refit"),
    "first_refit_rule_ok": ("처음 다시 맞춘 날이 규칙에 맞는지", "first refit follows the rule"),
    "hard_cut": ("여기까지만 학습에 씀", "training stops here"),
    "holdout_start": ("남겨둔 구간이 시작되는 날", "the untouched stretch starts"),
    "holdout_final": ("남겨둔 구간을 열어 본 적 있는지", "has the untouched stretch been opened"),
    "unlock_exists_before_run": ("실행 전에 해제 파일이 있었는지", "was an unlock file present before the run"),
    "live_mode": ("실제 운용 모드인지", "running in live mode"),
    "replay_end_used": ("다시 돌려 본 마지막 날", "last day of the replay"),
    "wf_end_exclusive": ("채점을 멈춘 날 (이 날은 뺌)", "scoring stops before this day"),
    "har_train_start": ("보조 모델 학습 시작", "side model training starts"),
    "purge": ("겹치는 구간 제거 (거래일)", "overlapping days removed"),
    "n_boot": ("다시 뽑은 횟수", "resamples drawn"),
    "boot_block": ("한 번에 잘라 뽑는 길이 (일)", "chunk length resampled (days)"),
    "hac_lag": ("이웃한 날의 겹침을 감안한 지연 수", "lag used for overlapping days"),
    "budget": ("조정해도 되는 숫자의 한도", "how many numbers we may tune"),
    "param_count": ("실제로 조정한 숫자", "numbers actually tuned"),
    "n_refits": ("다시 맞춘 횟수", "number of refits"),
    "n_sessions_features": ("입력을 만든 거래일 수", "sessions with inputs built"),
    "n_sessions_oos": ("채점한 거래일 수", "sessions scored"),
    "ladder_blocks": ("단계 비교에 쓴 구간", "blocks used to compare steps"),
    "ladder_runtime_sec": ("단계 비교에 걸린 시간 (초)", "seconds spent comparing steps"),
    "score_runtime_sec": ("채점에 걸린 시간 (초)", "seconds spent scoring"),
    "spy_last": ("SPY 마지막 자료일", "last SPY day in the cache"),
    "spy_last_used": ("실제로 쓴 SPY 마지막 날", "last SPY day actually used"),
    "spy_last_bar_complete": ("그 날 봉이 완성됐는지", "was that day's bar complete"),
    "last_completed_session": ("마지막으로 끝난 거래일", "last completed session"),
    "m3_same_rows": ("M3 와 같은 행으로 채점했는지", "scored on the same rows as M3"),
    "v0ref_to_m3": ("v0 참조선과 M3 를 같은 행에서 견줬는지", "v0 reference compared with M3 on the same rows"),
    "acceptance_rule": ("판정에 쓴 규칙", "rule used for the verdict"),
    "acceptance_blocks": ("판정에 쓴 구간 표", "block tables used for the verdict"),
    "acceptance_ref": ("판정 근거 문서", "where the verdict rule is written down"),
    "blocks_clim_pos": ("평소 평균보다 나았던 구간", "blocks better than the long-run average"),
    "blocks_m1_pos": ("M1 보다 나았던 구간", "blocks better than M1"),
    "blocks_vix_nonneg": ("VIX 공식에 뒤지지 않은 구간", "blocks not worse than the VIX formula"),
    "gk_ov_cc_bounds": ("하루 폭 계산 비교의 허용 범위", "allowed range when the two range measures are compared"),
    "gk_ov_cc_ratio_1993_95_min": ("하루 폭 두 계산의 비 — 1993~95 최소", "range-measure ratio, 1993-95 minimum"),
    "gk_ov_cc_ratio_1993_95_mean": ("하루 폭 두 계산의 비 — 1993~95 평균", "range-measure ratio, 1993-95 average"),
    "gk_ov_cc_ratio_1993_95_max": ("하루 폭 두 계산의 비 — 1993~95 최대", "range-measure ratio, 1993-95 maximum"),
    "gk_ov_cc_ratio_1996plus_min": ("하루 폭 두 계산의 비 — 1996년 이후 최소", "range-measure ratio, 1996 onward minimum"),
    "gk_ov_cc_ratio_1996plus_mean": ("하루 폭 두 계산의 비 — 1996년 이후 평균", "range-measure ratio, 1996 onward average"),
    "gk_ov_cc_ratio_1996plus_max": ("하루 폭 두 계산의 비 — 1996년 이후 최대", "range-measure ratio, 1996 onward maximum"),
    "gk_ov_cc_ratio_1996plus_last": ("하루 폭 두 계산의 비 — 가장 최근", "range-measure ratio, most recent"),
    "early_open_eq_hl_share": ("초기 자료에서 시가가 고가·저가와 같은 날 비율", "early data: share of days where the open equals the high or low"),
    "late_open_eq_hl_share": ("최근 자료에서 시가가 고가·저가와 같은 날 비율", "recent data: share of days where the open equals the high or low"),
    "early_median_log_range_pct": ("초기 자료의 하루 폭 가운데값", "early data: median daily range"),
    "late_median_log_range_pct": ("최근 자료의 하루 폭 가운데값", "recent data: median daily range"),
    "pk_cc_bounds": ("또 다른 폭 계산 비교의 허용 범위", "allowed range for the other range measure"),
    "pk_cc_rv20_min": ("그 비교의 최소", "that comparison, minimum"),
    "pk_cc_rv20_max": ("그 비교의 최대", "that comparison, maximum"),
    "available": ("쓸 수 있는지", "available"),
    "description": ("설명", "description"),
    "ok": ("이상 없음", "all good"),
    "coef": ("계수", "coefficients"), "features": ("입력 이름", "input names"),
    "const": ("기본값", "constant"),
    "ln_rv1": ("어제 움직임 (ln_rv1, 로그)", "yesterday (ln_rv1, log)"),
    "ln_rv5": ("최근 5일 움직임 (ln_rv5, 로그)", "last 5 days (ln_rv5, log)"),
    "ln_rv22": ("최근 22일 움직임 (ln_rv22, 로그)", "last 22 days (ln_rv22, log)"),
    "warnings": ("경고", "warnings"),
}


# 셀에 들어오는 짧은 상태값 → (쉬운 한국어, English). 숫자·날짜·식별자는 건드리지 않는다.
_VALUE_BI: dict[str, tuple[str, str]] = {
    "info_only": ("참고용만 (info_only)", "information only (not in use)"),
    "tones": ("신호등 판정에 사용 (tones)", "traffic-light calls in use"),
    INFO_DISPLAY_LABEL: ("참고 표시 (정보 표시(배포 안 함))", INFO_DISPLAY_EN),
    DEPLOYED_LABEL: ("실제 사용 (배포)", "in use"),
}


def _bi_label(key) -> str:
    """열 제목 한 칸의 두 언어. 대응이 없으면 한국어 라벨 + 키 이름(영어 자리)."""
    k = str(key)
    pair = BI_LABELS.get(k) or BI_LABELS.get(k.lower())
    if pair:
        return bi(pair[0], pair[1])
    ko = LABELS.get(k, LABELS.get(k.lower(), k))
    return bi(ko, k)


_MACHINE_KEYS = ("feature_rule", "window_rule", "rule_text", "spec", "note")


def _quote_machine(d) -> dict:
    """기계가 만든 규칙 문자열은 번역하지 않고 원문 그대로 인용한다 — 한 글자만 달라도 다른 규칙이다."""
    if not isinstance(d, dict):
        return {}
    return {k: (_Html(_raw(v)) if k in _MACHINE_KEYS and isinstance(v, str) else v) for k, v in d.items()}


def _val_bi(v, key: str = "") -> tuple[str, str]:
    """값 한 개를 (한국어 표시, 영어 표시) 로. 숫자·날짜·식별자는 같고 예/아니오만 갈린다."""
    if isinstance(v, (bool, np.bool_)):
        return ("예", "yes") if v else ("아니오", "no")
    t = _fmt_p2(v, key)
    return (t, t)


def _bi_cell(v, key: str = "") -> str:
    """셀 값. 숫자·날짜·식별자는 그대로, 예/아니오와 몇 안 되는 상태말만 두 언어로."""
    if isinstance(v, _Html):
        return str(v)
    if isinstance(v, (bool, np.bool_)):
        return bi("예", "yes") if v else bi("아니오", "no")
    if isinstance(v, str):
        pair = _VALUE_BI.get(v)
        if pair:
            return bi(pair[0], pair[1])
    return _fmt_p2(v, key)


def _bi_table(records, order=None, empty=("자료 없음", "no data")) -> str:
    """표 하나 — 열 제목만 두 언어로 심고 셀 값은 그대로 둔다 (STYLE_I18N.md §1)."""
    recs = _to_records(records)
    if not recs:
        return f'<div class="note">{bi(empty[0], empty[1])}</div>'
    cols: list[str] = []
    for r in recs:
        for k in r.keys():
            if k not in cols:
                cols.append(k)
    if order:
        cols = [c for c in order if c in cols] + [c for c in cols if c not in order]
    numeric = {c: all(_is_num(r.get(c)) or _is_nan(r.get(c)) for r in recs) for c in cols}
    head = "".join(f'<th class="{"num" if numeric[c] else ""}">{_bi_label(c)}</th>' for c in cols)
    body = []
    for r in recs:
        cells = "".join(f'<td class="{"num" if numeric[c] else ""}">{_bi_cell(r.get(c), c)}</td>' for c in cols)
        body.append(f"<tr>{cells}</tr>")
    return f'<div class="tblwrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _bi_kv(d, empty=("자료 없음", "no data"), cols: dict | None = None) -> str:
    """항목/값 표. 값이 dict 이면 열로 편다(cols 로 열 제목의 두 언어를 준다)."""
    if not isinstance(d, dict) or not d:
        return f'<div class="note">{bi(empty[0], empty[1])}</div>'
    sub = {k: v for k, v in d.items() if isinstance(v, dict)}
    flat = {k: v for k, v in d.items() if not isinstance(v, (dict, list, tuple, pd.DataFrame, pd.Series))}
    lists = {k: v for k, v in d.items() if isinstance(v, (list, tuple, pd.DataFrame, pd.Series))}
    parts = []
    if flat:
        rows = "".join(f"<tr><td>{_bi_label(k)}</td><td class=\"num\">{_bi_cell(v, k)}</td></tr>" for k, v in flat.items())
        parts.append('<div class="tblwrap"><table><thead><tr>'
                     f'<th>{bi("항목", "item")}</th><th class="num">{bi("값", "value")}</th>'
                     f"</tr></thead><tbody>{rows}</tbody></table></div>")
    if sub:
        metrics: list[str] = []
        for v in sub.values():
            for k in v.keys():
                if k not in metrics:
                    metrics.append(k)
        def _col_head(k):
            pair = (cols or {}).get(k)
            return f'<th class="num">{bi(pair[0], pair[1]) if pair else _bi_label(k)}</th>'

        head = "".join(_col_head(k) for k in sub.keys())
        rows = []
        for m in metrics:
            cells = "".join(f'<td class="num">{_bi_cell(v.get(m), m)}</td>' for v in sub.values())
            rows.append(f"<tr><td>{_bi_label(m)}</td>{cells}</tr>")
        parts.append(f'<div class="tblwrap"><table><thead><tr><th>{bi("지표", "measure")}</th>{head}</tr>'
                     f"</thead><tbody>{''.join(rows)}</tbody></table></div>")
    for k, v in lists.items():
        recs = _to_records(v)
        if recs:
            parts.append(f"<h3>{_bi_label(k)}</h3>" + _bi_table(recs))
        else:
            # 표로 그릴 수 없는 목록(경고문 등)은 계산이 남긴 한국어 원문이다 — 인용 표시를 달아
            # 영어 화면에도 한국어가 남는다는 사실을 드러낸다(고지는 _bi_label 이 두 벌로 준다).
            cell = _bi_cell(v, k)
            if _HANGUL_TXT.search(cell) and 'class="lg ' not in cell:
                cell = _raw(_h.unescape(cell))
            parts.append(f'<div class="note">{_bi_label(k)}: {cell}</div>')
    return "".join(parts) if parts else f'<div class="note">{bi(empty[0], empty[1])}</div>'


def _bi_kpi(ko: str, en: str, value: str, sub_ko: str = "", sub_en: str = "") -> str:
    """KPI 카드 한 장 — 제목·아래 설명만 두 언어로, 값은 그대로."""
    return (f'<div class="kpi"><div class="lb">{bi(ko, en)}</div><div class="v">{value}</div>'
            f'<div class="s">{bi(sub_ko, sub_en) if (sub_ko or sub_en) else ""}</div></div>')


def _bi_h2(ko: str, en: str) -> str:
    return f"<h2>{bi(ko, en)}</h2>"


def _bi_h3(ko: str, en: str) -> str:
    return f"<h3>{bi(ko, en)}</h3>"


def _bi_note(ko: str, en: str, cls: str = "note", style: str = "") -> str:
    st = f' style="{style}"' if style else ""
    return f'<div class="{cls}"{st}>{bi(ko, en)}</div>'


def _bi_note_html(ko: str, en: str, cls: str = "note", style: str = "") -> str:
    """안에 <b>·원문 인용이 들어가는 문장 — 호출자가 이스케이프 책임을 진다."""
    st = f' style="{style}"' if style else ""
    return f'<div class="{cls}"{st}>{bi_html(ko, en)}</div>'


def _bi_tag(ko: str, en: str, cls: str = "tag") -> str:
    return f'<span class="{cls}">{bi(ko, en)}</span>'


def _bi_tag_html(ko: str, en: str, cls: str = "tag") -> str:
    return f'<span class="{cls}">{bi_html(ko, en)}</span>'


def _bi_img(png: bytes | None, ko_cap: str, en_cap: str, alt_ko: str, alt_en: str) -> str:
    """차트 한 장 — 캡션은 두 언어, 그림 안 글자는 영어 그대로(차트는 이중화하지 않는다)."""
    if not png:
        return f'<div class="note">{bi(f"차트 없음 — {ko_cap}", f"no chart — {en_cap}")}</div>'
    b64 = base64.b64encode(png).decode("ascii")
    return (f'<div class="chart"><img src="data:image/png;base64,{b64}" alt="{bi_attr(alt_ko, alt_en)}">'
            f'<div class="cap">{bi(ko_cap, en_cap)}</div></div>')


def _bi_warn_list(items, ko_empty: str = "경고 없음", en_empty: str = "no warnings") -> str:
    """계산이 남긴 경고 — 원문 그대로 인용한다(번역하면 무엇이 걸렸는지 흐려진다)."""
    items = [str(w) for w in (items or []) if w is not None and str(w).strip()]
    if not items:
        return f'<div class="note">{bi(ko_empty, en_empty)}</div>'
    return ('<ul class="warnlist">' + "".join(f"<li>{_raw(w)}</li>" for w in items) + "</ul>"
            + _bi_note("위 줄은 이 페이지를 만든 계산이 남긴 경고 원문입니다 — 번역하지 않고 그대로 싣습니다.",
                       "The lines above are the warning text left by the run that produced this page, quoted as-is "
                       "(Korean) rather than translated."))


def _ladder_records(obj) -> list[dict]:
    """ladder_table 기록의 mean/lo/hi 를 diff_* 로 이름 바꿔(손실차 ×1e-4 포맷) 돌려준다."""
    out = []
    for r in _to_records(obj):
        rr = dict(r)
        for a, b in (("mean", "diff_mean"), ("lo", "diff_lo"), ("hi", "diff_hi")):
            if a in rr and b not in rr:
                rr[b] = rr.pop(a)
        out.append(rr)
    return out


def _param_records(obj, rung: str | None = "M3") -> list[dict]:
    """params_by_refit → 평면 기록(b_x_vix …). rung 이 주어지면 그 단만."""
    out = []
    for r in _to_records(obj):
        if rung is not None and r.get("rung") not in (None, rung):
            continue
        rr = {k: v for k, v in r.items() if k not in ("coef", "features")}
        coef = r.get("coef") if isinstance(r.get("coef"), dict) else {}
        for f, b in coef.items():
            rr[f"b_{f}"] = b
        out.append(rr)
    return out


# ------------------------------------------------------------------
# v0 영구 표기 줄 (VALIDATION §4)
# ------------------------------------------------------------------
def v0_headline(summary_v0: dict) -> dict:
    """results/summary_v0_completed.json → 영구 표기용 v0 숫자 dict (없는 키는 None). 리포트 첫 줄과 v1 비교표가 쓴다."""
    s = summary_v0 if isinstance(summary_v0, dict) else {}
    h = s.get("headline") if isinstance(s.get("headline"), dict) else {}
    a = s.get("allocation") if isinstance(s.get("allocation"), dict) else {}
    sw = s.get("switching") if isinstance(s.get("switching"), dict) else (s.get("switches") if isinstance(s.get("switches"), dict) else {})
    run = s.get("run") if isinstance(s.get("run"), dict) else {}
    meta = s.get("meta") if isinstance(s.get("meta"), dict) else {}
    dr = s.get("data_range") if isinstance(s.get("data_range"), dict) else {}
    br = s.get("brier_reference") if isinstance(s.get("brier_reference"), dict) else {}
    br_dd = br.get("y_dd5_20") if isinstance(br.get("y_dd5_20"), dict) else {}
    dd_by_tone = h.get("dd5_20_rate_by_tone") if isinstance(h.get("dd5_20_rate_by_tone"), dict) else {}

    def _pick(*cands):
        for v in cands:
            if v is not None and not _is_nan(v):
                return v
        return None

    return {
        "variant": _pick(s.get("variant"), run.get("variant"), dr.get("variant")),
        "start": _pick(run.get("start"), dr.get("start"), a.get("start")), "end": _pick(run.get("end"), dr.get("end"), a.get("end")),
        "n_days": _pick(run.get("n_days"), dr.get("n_days"), meta.get("n_days")),
        "tone_switches_per_year": _pick(_first(h, "tone_switches_per_year"), _first(sw, "tone_switches_per_year", "switches_per_year")),
        "warn_share": _pick(_first(sw, "warn_share"), _first(h, "warn_share")),
        "true_alarm_share": _pick(_first(h, "true_alarm_share"), _first(sw, "true_alarm_share")),
        "true_alarm_share_baseline": _pick(_first(h, "true_alarm_share_baseline"), _first(sw, "true_alarm_share_baseline")),
        "null_true_alarm_share": _pick(_first(h, "null_true_alarm_share"), _first(sw, "null_true_alarm_share")),
        "false_alarms_per_year": _pick(_first(h, "false_alarms_per_year"), _first(sw, "false_alarms_per_year")),
        "dd5_20_rate_by_tone": dict(dd_by_tone), "dd5_20_base_rate": _first(h, "dd5_20_base_rate"),
        "hit_20_all": _first(h, "hit_20_all"), "baseline_20": _first(h, "baseline_20"),
        "detection_rate_10": _first(h, "detection_rate_10"), "median_lead_10": _first(h, "median_lead_10"),
        "cagr_strategy": _pick(_first(h, "cagr_strategy"), _first(a, "cagr")), "cagr_bh": _pick(_first(h, "cagr_bh"), _first(a, "bh_cagr")),
        "maxdd_strategy": _pick(_first(h, "maxdd_strategy"), _first(a, "max_dd")), "maxdd_bh": _pick(_first(h, "maxdd_bh"), _first(a, "bh_max_dd")),
        "worst_month_strategy": _pick(_first(h, "worst_month_strategy"), _first(a, "worst_month")),
        "worst_month_bh": _pick(_first(h, "worst_month_bh"), _first(a, "bh_worst_month")),
        "brier_climatology_dd5": _first(br_dd, "brier_climatology"), "base_rate_dd5": _first(br_dd, "base_rate"),
        "signals_v0_sha256": run.get("signals_v0_sha256"), "window_rule": run.get("window_rule"),
        "generated_at_utc": run.get("generated_at_utc"), "allocation": {k: a.get(k) for k in a if k != "series"},
    }


def load_summary_v0(path=None) -> dict:
    """results/summary_v0_completed.json 을 읽는다(없으면 {} + 경고). 렌더 함수가 v0 줄을 영구 표기하기 위한 폴백."""
    import json
    p = Path(path) if path is not None else (RESULTS_DIR / "summary_v0_completed.json")
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError) as e:
        warnings.warn(f"summary_v0_completed.json 을 읽지 못함({type(e).__name__}) → v0 줄에 숫자 없음")
        return {}


def _v0_line_html(v0h: dict, prefix: str = "v0 (동결 벤치마크", *, bi_mode: bool = False) -> str:
    """v0 결과 영구 표기 한 줄. 숫자가 없으면 '—' 로 남기되 줄 자체는 항상 있다.

    bi_mode=True 면 같은 숫자를 쉬운 한국어와 영어 두 벌로 심는다(주간 두 페이지 전용).
    기본값은 예전 그대로 — index 카드와 Phase 3 페이지가 이 줄을 그대로 쓴다.
    """
    v = v0h if isinstance(v0h, dict) else {}
    dd = v.get("dd5_20_rate_by_tone") if isinstance(v.get("dd5_20_rate_by_tone"), dict) else {}
    dd_warn = [dd.get(t) for t in ("caution", "reduce") if not _is_nan(dd.get(t))]
    dd_up = [dd.get(t) for t in ("buy", "hold", "neutral") if not _is_nan(dd.get(t))]
    dd_txt = ""
    if dd_warn and dd_up:
        dd_txt = f" · 경고 톤 -5%/20일 {min(dd_warn) * 100:.0f}~{max(dd_warn) * 100:.0f}% vs 상승 톤 {min(dd_up) * 100:.0f}~{max(dd_up) * 100:.0f}%"
    span = f", {v.get('variant') or 'completed'}, {v.get('start') or '?'}~{v.get('end') or '?'}, {_fmt(v.get('n_days'), 'n_days')}거래일)"
    parts = [f"톤 전환 {_fmt_p2(v.get('tone_switches_per_year'), 'switches_per_year')}회/년",
             f"경고 비중 {_pct1(v.get('warn_share'))}",
             f"진짜 경보 {_pct1(v.get('true_alarm_share'))} (기저율 {_pct1(v.get('true_alarm_share_baseline'))})",
             f"배분 CAGR {_pct1(v.get('cagr_strategy'))} vs 보유 {_pct1(v.get('cagr_bh'))}",
             f"MaxDD {_pct1(v.get('maxdd_strategy'))} vs {_pct1(v.get('maxdd_bh'))}",
             f"기후학 Brier(y_dd5_20) {_fmt_p2(v.get('brier_climatology_dd5'), 'brier')}"]
    if not bi_mode:
        return (f'<div class="v0line"><b>{_esc(prefix)}{_esc(span)}</b>: ' + " · ".join(_esc(x) for x in parts) + _esc(dd_txt)
                + ' <span class="note" style="display:inline">— VALIDATION.md §4: v0 결과는 모든 Phase 2 리포트 첫 줄에 영구 표기</span></div>')
    n_days_txt = _fmt(v.get("n_days"), "n_days")
    variant = v.get("variant") or "completed"
    head_ko = f"{prefix}{span}"
    head_en = (f"v0 (frozen benchmark, {variant}, {v.get('start') or '?'} to {v.get('end') or '?'}, "
               f"{n_days_txt} trading days)")
    ko = [f"신호등 판정 바뀜 {_fmt_p2(v.get('tone_switches_per_year'), 'switches_per_year')}회/년",
          f"경고가 켜져 있던 날 {_pct1(v.get('warn_share'))}",
          f"경고가 진짜였던 비율 {_pct1(v.get('true_alarm_share'))} — {_ten_freq(v.get('true_alarm_share'))}쯤 "
          f"(평소 비율, 즉 기저율 {_pct1(v.get('true_alarm_share_baseline'))} — "
          f"{_ten_freq(v.get('true_alarm_share_baseline'))}쯤)",
          f"1년 평균 수익률 {_pct1(v.get('cagr_strategy'))} (그냥 들고 있었으면 {_pct1(v.get('cagr_bh'))})",
          f"고점 대비 최대 하락 {_pct1(v.get('maxdd_strategy'))} (그냥 들고 있었으면 {_pct1(v.get('maxdd_bh'))})",
          f"평소 평균이 빗나간 정도 {_fmt_p2(v.get('brier_climatology_dd5'), 'brier')}"]
    en = [f"traffic-light call changed {_fmt_p2(v.get('tone_switches_per_year'), 'switches_per_year')} times a year",
          f"a warning was on {_pct1(v.get('warn_share'))} of days",
          f"{_pct1(v.get('true_alarm_share'))} of warnings were real — {_ten_freq_en(v.get('true_alarm_share'))} "
          f"(any day would give {_pct1(v.get('true_alarm_share_baseline'))} — "
          f"{_ten_freq_en(v.get('true_alarm_share_baseline'))})",
          f"return per year {_pct1(v.get('cagr_strategy'))} (buy and hold {_pct1(v.get('cagr_bh'))})",
          f"biggest drop from the peak {_pct1(v.get('maxdd_strategy'))} (buy and hold {_pct1(v.get('maxdd_bh'))})",
          f"error score of the long-run average {_fmt_p2(v.get('brier_climatology_dd5'), 'brier')}"]
    if dd_warn and dd_up:
        ko.append(f"경고가 켜진 날은 20일 안에 5% 넘게 떨어진 비율이 {min(dd_warn) * 100:.0f}~{max(dd_warn) * 100:.0f}%, "
                  f"평온한 날은 {min(dd_up) * 100:.0f}~{max(dd_up) * 100:.0f}%")
        en.append(f"on warning days {min(dd_warn) * 100:.0f}-{max(dd_warn) * 100:.0f}% fell more than 5% within 20 days, "
                  f"on calm days {min(dd_up) * 100:.0f}-{max(dd_up) * 100:.0f}%")
    body = bi_html(f"<b>{_esc(head_ko)}</b>: " + _esc(" · ".join(ko)),
                   f"<b>{_esc(head_en)}</b>: " + _esc(" · ".join(en)))
    tail = bi("— VALIDATION.md §4: v0 결과는 모든 Phase 2 리포트 첫 줄에 영구 표기합니다.",
              "— VALIDATION.md §4: the v0 numbers stay on the first line of every Phase 2 page, permanently.")
    return f'<div class="v0line">{body} <span class="note" style="display:inline">{tail}</span></div>'


# ------------------------------------------------------------------
# 결정층 임계 (카드용 폴백 — decision.next_thresholds 와 같은 값)
# ------------------------------------------------------------------
def _thresholds_fallback(state: str, days_in_state, clim) -> dict:
    cfg = DECISION_P2
    c = _fnum(clim)
    esc = {"normal": ("caution", cfg["enter_caution"]), "caution": ("reduce", cfg["enter_reduce"]), "reduce": (None, None)}.get(state, (None, None))
    de = {"normal": (None, None), "caution": ("normal", cfg["exit_caution"]), "reduce": ("caution", cfg["exit_reduce"])}.get(state, (None, None))
    d = int(_fnum(days_in_state)) if not math.isnan(_fnum(days_in_state)) else 0
    return {"state": state, "escalate_to": esc[0], "r_escalate": esc[1], "p_escalate": (esc[1] * c if esc[1] is not None else None),
            "deescalate_to": de[0], "r_deescalate": de[1], "p_deescalate": (de[1] * c if de[1] is not None else None),
            "dwell_remaining": (max(0, int(cfg["dwell"]) - d) if de[0] is not None else None), "clim": c}


# ------------------------------------------------------------------
# 일간 카드 (index.html 의 Phase 2 조각)
# ------------------------------------------------------------------
def _band_of(d: dict) -> tuple[float, float, str, tuple, tuple]:
    """(lo, hi, src, param_band, calib_band). lo/hi 가 주어지면 그대로, 아니면 두 밴드 중 넓은 쪽(§0: 헤드라인은 넓은 쪽)."""
    def _pair(v):
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return (_fnum(v[0]), _fnum(v[1]))
        if isinstance(v, dict):
            return (_fnum(_first(v, "lo", "wilson_lo", "min")), _fnum(_first(v, "hi", "wilson_hi", "max")))
        return (math.nan, math.nan)
    pb = _pair(d.get("param_band"))
    cb = _pair(d.get("calib_band"))
    if math.isnan(cb[0]) and isinstance(d.get("calib_bin"), dict):
        cb = (_fnum(d["calib_bin"].get("wilson_lo")), _fnum(d["calib_bin"].get("wilson_hi")))
    lo, hi = _fnum(_first(d, "lo", "p2_lo")), _fnum(_first(d, "hi", "p2_hi"))
    src = _first(d, "band_src", "p2_band_src")
    if math.isnan(lo) or math.isnan(hi):
        wp = pb[1] - pb[0] if not (math.isnan(pb[0]) or math.isnan(pb[1])) else -1.0
        wc = cb[1] - cb[0] if not (math.isnan(cb[0]) or math.isnan(cb[1])) else -1.0
        if wp < 0 and wc < 0:
            return (math.nan, math.nan, str(src or ""), pb, cb)
        if wc > wp:
            lo, hi, src = cb[0], cb[1], "calib"
        else:
            lo, hi, src = pb[0], pb[1], "param"
    return (lo, hi, str(src or ""), pb, cb)


def _contrib_rows(d: dict, key: str = "contributions") -> list[dict]:
    """contributions(DataFrame/records/dict) → [{term, logit, pp}] (절편 포함, 순서 intercept→x_vix→x_har→x_ma).
    key 로 다른 귀속 표(예: contributions_m3 = 배포되지 않은 M3 의 정보 표시)를 읽을 수 있다."""
    c = d.get(key) if isinstance(d, dict) else None
    rows: list[dict] = []
    if isinstance(c, pd.DataFrame):
        df = c.reset_index()
        col = "term" if "term" in df.columns else df.columns[0]
        for _, r in df.iterrows():
            rows.append({"term": str(r[col]), "logit": _fnum(r.get("logit")), "pp": _fnum(r.get("pp"))})
    elif isinstance(c, dict) and c and all(isinstance(v, dict) for v in c.values()):
        for k, v in c.items():
            rows.append({"term": str(k), "logit": _fnum(v.get("logit")), "pp": _fnum(v.get("pp"))})
    else:
        for r in _to_records(c):
            rows.append({"term": str(r.get("term", r.get("index", r.get("key", "")))), "logit": _fnum(r.get("logit")), "pp": _fnum(r.get("pp"))})
    order = ["intercept", "x_vix", "x_har", "x_ma"]
    rows.sort(key=lambda r: order.index(r["term"]) if r["term"] in order else 99)
    return rows


def _dod_line(dod: dict, label_ko: str, label_en: str) -> str:
    """일간(또는 5일 누적) 변화 문장 — 표시된 항의 합 == 표시된 합계. 두 언어로 함께 심는다.

    보존하는 것(정직성): 각 항의 pp, 합계, (재적합) 괄호, 'N세션 전 대비' 단서, 재적합 배지,
    입력이 없을 때의 실패 문구. 어느 것도 한쪽 언어에서만 보이지 않는다.
    """
    if not isinstance(dod, dict):
        return ""
    d_pp = dod.get("d_pp") if isinstance(dod.get("d_pp"), dict) else {}
    d_p = _fnum(_first(dod, "d_p"))
    if not d_pp and math.isnan(d_p):
        return ""
    parts, tot = _exact_pp_parts(d_pp, d_p)
    if math.isnan(tot):
        return _bi_note(f"{label_ko}: 변화 계산 불가(입력 결측)",
                        f"{label_en}: cannot compute the change (an input is missing)")
    order = ["x_vix", "x_har", "x_ma", "refit"]
    keys = [k for k in order if k in parts] + [k for k in parts if k not in order]
    seg_ko, seg_en = [], []
    for k in keys:
        v = parts[k]
        txt = "—" if math.isnan(v) else f"{v:+.1f}"
        nm_ko, nm_en = _feat_bi(k)
        seg_ko.append(f"({nm_ko} {txt})" if k == "refit" else f"{nm_ko} {txt}")
        seg_en.append(f"({nm_en} {txt})" if k == "refit" else f"{nm_en} {txt}")
    gap = dod.get("gap_sessions")
    show_gap = isinstance(gap, (int, np.integer)) and not isinstance(gap, bool) and int(gap) > 1
    gap_ko = f" · {int(gap)}세션 전 대비" if show_gap else ""
    gap_en = f" · compared with {int(gap)} sessions ago" if show_gap else ""
    body_ko = f"<b>{_esc(label_ko)} {tot:+.1f}pp</b>: " + _esc(" · ".join(seg_ko)) + _esc(gap_ko)
    body_en = f"<b>{_esc(label_en)} {tot:+.1f}pp</b>: " + _esc(" · ".join(seg_en)) + _esc(gap_en)
    tag = (" " + _bi_tag("재적합 반영", "includes a refit", cls="tag warn")) if dod.get("refit") else ""
    return f'<div class="dod">{bi_html(body_ko, body_en)}{tag}</div>'


def _bars_html(rows: list[dict]) -> str:
    """수준 귀속 3막대(VIX / 실현-내재 갭 / 추세): pp 비례 가로 막대 + logit·pp 숫자."""
    feats = [r for r in rows if r["term"] != "intercept"]
    if not feats:
        return _bi_note("귀속 자료 없음", "no attribution data")
    mx = max((abs(r["pp"]) for r in feats if not math.isnan(r["pp"])), default=0.0)
    html = []
    for r in feats:
        pp, lg = r["pp"], r["logit"]
        w = 0.0 if (mx <= 0 or math.isnan(pp)) else min(100.0, abs(pp) / mx * 100.0)
        color = FEATURE_COLORS.get(r["term"], PALETTE["mut"])
        pp_txt = "—" if math.isnan(pp) else f"{pp * 100:+.1f}pp"
        lg_txt = "—" if math.isnan(lg) else f"logit {lg:+.2f}"
        html.append(f'<div class="p2bar"><div class="lb">{bi(*_feat_bi(r["term"]))}</div>'
                    f'<div class="tr"><div class="fl" style="width:{w:.1f}%;background:{color}"></div></div>'
                    f'<div class="vv">{_esc(pp_txt)} <span class="mono" style="color:var(--dim)">{_esc(lg_txt)}</span></div></div>')
    return "".join(html)


def _events_html(d: dict) -> str:
    ev = d.get("events")
    recs = []
    for r in _to_records(ev):
        dt = r.get("date")
        recs.append({"date": (pd.Timestamp(dt).strftime("%Y-%m-%d") if not _is_nan(dt) else "—"), "kind": r.get("kind"),
                     "sessions_ahead": r.get("sessions_ahead"), "label_ko": r.get("label_ko"), "tentative": r.get("tentative")})
    hz = d.get("events_horizon") if isinstance(d.get("events_horizon"), dict) else {}
    warn = ""
    if hz.get("warn"):
        warn = _bi_tag_html(f'이벤트 표 잔여 {_fmt(hz.get("days_left"), "days")}일 &lt; '
                            f'{_fmt(hz.get("warn_days", 60), "days")}일 — FOMC 표 갱신 필요'
                            f'(마지막 {_esc(hz.get("last_fomc") or "?")})',
                            f'the event table only reaches {_fmt(hz.get("days_left"), "days")} days ahead, '
                            f'under {_fmt(hz.get("warn_days", 60), "days")} — the FOMC table needs updating '
                            f'(last {_esc(hz.get("last_fomc") or "?")})', cls="tag warn")
    attrs = ev.attrs if isinstance(ev, pd.DataFrame) else {}
    notes = list(attrs.get("warnings", []) or []) + list(d.get("events_warnings") or [])
    cpi = attrs.get("cpi_registered") if attrs else d.get("cpi_registered")
    if cpi is False and "CPI 일정 미등록" not in notes:
        notes.append("CPI 일정 미등록")
    body = _bi_table(recs, ["date", "kind", "sessions_ahead", "label_ko", "tentative"],
                     ("다음 20거래일 안에 등록된 이벤트 없음", "no scheduled event in the next 20 trading days"))
    note_html = ""
    if notes:
        # 계산이 남긴 한국어 원문 — 번역하지 않고 인용 표시하되, 그 사실을 두 언어로 밝힌다.
        note_html = ('<div class="note">'
                     + bi("계산이 남긴 한국어 원문", "raw Korean written by the pipeline") + ": "
                     + _raw(" · ".join(str(n) for n in notes)) + "</div>")
    return body + warn + note_html


def _coef_txt(model: dict) -> str:
    """'b0 -1.650 · x_vix +0.890 …' — 그 모델이 실제로 가진 계수만(없는 항을 0 으로 지어내지 않는다)."""
    coef = model.get("coef") if isinstance(model.get("coef"), dict) else {}
    b0 = _fnum(_first(model, "intercept", "b0"))
    txt = f"b0 {b0:+.3f}" if not math.isnan(b0) else "b0 —"
    for f in [k for k in ("x_vix", "x_har", "x_ma") if k in coef] + [k for k in coef if k not in ("x_vix", "x_har", "x_ma")]:
        v = _fnum(coef.get(f))
        txt += f" · {f} {v:+.3f}" if not math.isnan(v) else f" · {f} —"
    return txt


TABLE_KO_P2 = {"24": "24개월", "18": "18개월", "1999": "1999 시작"}     # 블록 표 이름 → 사람이 읽는 이름


TABLE_EN_P2 = {"24": "24-month", "18": "18-month", "1999": "from 1999"}


def _table_ko_p2(name) -> str:
    return TABLE_KO_P2.get(str(name), f"{name} 블록")


def _table_bi_p2(name) -> tuple[str, str]:
    """구간 표 이름 (쉬운 한국어, English)."""
    n = str(name)
    return (TABLE_KO_P2.get(n, f"{n} 구간"), TABLE_EN_P2.get(n, f"{n} block"))


def acceptance_verdict_line(acc: dict) -> str:
    """배치 판정 한 줄 — calibrate.acceptance 의 verdict_line 을 **그대로** 쓴다(리포트가 판정을 다시 쓰지 않는다).
    예: "24개월 표: M1 통과 · 18개월 표: 실패(최소 블록 BSS_clim −0.0996) → 두 표 모두 통과 요구(소유자 결정 2026-09-08)
         → 배치 없음(정보 제공 전용)". verdict_line 이 없는 예전 산출물이면 rationale 로 내려간다."""
    if not isinstance(acc, dict) or not acc:
        return ""
    txt = acc.get("verdict_line") or acc.get("rationale")
    return str(txt) if txt else ""


def _acc_named_rung(acc: dict) -> str:
    """어느 단계가 '거의 될 뻔했는지' — calibrate.acceptance 와 같은 순서(M3→M2→M1→M0)로 고른다."""
    per = acc.get("per_table") if isinstance(acc.get("per_table"), dict) else {}
    req = [str(t) for t in (acc.get("require_tables") or list(per))]
    union: set[str] = set()
    for t in req:
        union |= {str(r) for r in ((per.get(t) or {}).get("passing_rungs") or [])}
    for r in ("M3", "M2", "M1", "M0"):
        if r in union:
            return r
    return str(acc.get("candidate") or "M3")


def _acc_fail_bits(acc: dict, table, rung: str) -> tuple[str, str]:
    """그 표에서 왜 떨어졌는지 — 판정 dict 의 숫자 칸에서 직접 읽는다(원문 문장을 파싱하지 않는다)."""
    tbl = (acc.get("tables") or {}).get(str(table)) if isinstance(acc.get("tables"), dict) else None
    tbl = tbl if isinstance(tbl, dict) else {}
    rule = str(acc.get("rule") or "literal")
    blk = tbl.get("amended" if rule == "amended" else "literal")
    if not isinstance(blk, dict):
        blk = acc.get("amended" if rule == "amended" else "literal") if str(acc.get("primary_table") or "") == str(table) else {}
    e = blk.get(rung) if isinstance(blk, dict) and isinstance(blk.get(rung), dict) else {}
    if rule == "amended":
        mn = _fnum(e.get("min_block_bss_clim"))
        if not math.isnan(mn):
            return (f"가장 나쁜 구간 점수 {mn:+.4f}", f"worst block score {mn:+.4f}")
        bad = [k for k in ("A", "B", "C") if e.get(k) is False]
        if bad:
            return (f"{'·'.join(bad)} 기준 미달", f"test {' and '.join(bad)} not met")
    else:
        fb = e.get("failing_blocks")
        if fb:
            return (f"떨어진 구간 {fb}", f"blocks that failed: {fb}")
        ne = e.get("n_blocks_empty")
        if ne:
            return (f"채점할 수 없는 구간 {_fmt(ne, 'n')}개", f"{_fmt(ne, 'n')} blocks could not be scored")
    return (f"{rung} 은 통과하지 못함", f"{rung} did not pass")


def acceptance_verdict_bi(acc: dict) -> tuple[str, str]:
    """배치 판정을 **같은 사실 그대로** 쉬운 한국어와 영어로 다시 적는다.

    계산이 만든 한 줄(acceptance_verdict_line)은 옆에 원문 그대로 함께 싣는다 — 이 함수는 그 줄을 대체하지 않는다.
    표별 통과 여부·통과한 단계·막은 표는 모두 acceptance dict 의 구조 필드에서 읽는다(문장 번역이 아니다).
    """
    if not isinstance(acc, dict) or not acc:
        return ("", "")
    per = acc.get("per_table") if isinstance(acc.get("per_table"), dict) else {}
    req = [str(t) for t in (acc.get("require_tables") or [])] or ([str(acc.get("primary_table"))] if acc.get("primary_table") else [])
    named = _acc_named_rung(acc)
    ko_parts, en_parts = [], []
    for t in req:
        ko_t, en_t = _table_bi_p2(t)
        tone = (per.get(t) or {}).get("tone_model")
        if tone:
            ko_parts.append(f"{ko_t} 표에서는 {tone} 이 통과")
            en_parts.append(f"on the {en_t} table {tone} passes")
        else:
            f_ko, f_en = _acc_fail_bits(acc, t, named)
            ko_parts.append(f"{ko_t} 표에서는 떨어짐 ({f_ko})")
            en_parts.append(f"on the {en_t} table it fails ({f_en})")
    m = re.search(r"\d{4}-\d{2}-\d{2}", str(acc.get("conjunction_note") or acc.get("verdict_line") or ""))
    when = f", 소유자 결정 {m.group(0)}" if m else ""
    when_en = f", owner decision {m.group(0)}" if m else ""
    if len(req) > 1:
        _n_ko = "두" if len(req) == 2 else f"{len(req)}개"
        _n_en = "both tables" if len(req) == 2 else f"all {len(req)} tables"
        ko_parts.append(f"{_n_ko} 표가 모두 통과해야 실제로 씁니다{when}")
        en_parts.append(f"{_n_en} must pass before anything is used{when_en}")
    tone_model = acc.get("tone_model")
    if tone_model:
        ko_parts.append(f"그래서 {tone_model} 단계의 확률로 신호등 판정을 만듭니다 (실제 사용, tones)")
        en_parts.append(f"so the {tone_model} step drives the traffic-light call (in use, tones)")
    else:
        ko_parts.append("그래서 어느 단계도 실제로 쓰지 않고, 참고용으로만 보여 줍니다 (info_only)")
        en_parts.append("so nothing is put in use and everything here is information only (info_only)")
    blocking = [str(t) for t in (acc.get("blocking_tables") or [])]
    if blocking:
        ko_parts.append("막은 표: " + ", ".join(_table_bi_p2(t)[0] for t in blocking))
        en_parts.append("blocked by: " + ", ".join(_table_bi_p2(t)[1] for t in blocking) + " table")
    return (" → ".join(ko_parts), " → ".join(en_parts))


def _acceptance_tables(acc: dict) -> dict:
    """{표: {"literal": ..., "amended": ..., "required": bool, ...}} — 새 산출물은 acc["tables"], 예전 산출물은 평면 형식."""
    tabs = acc.get("tables") if isinstance(acc.get("tables"), dict) else None
    req = [str(t) for t in (acc.get("require_tables") or [])]
    if tabs and all(isinstance(v, dict) for v in tabs.values()):
        return {str(t): {**v, "required": bool(str(t) in req) if req else True} for t, v in tabs.items()}
    primary = str(acc.get("primary_table") or (req[0] if req else "24"))
    return {primary: {"literal": acc.get("literal"), "amended": acc.get("amended"), "required": True,
                      "tone_model": acc.get("tone_model"), "deploy_mode": acc.get("deploy_mode")}}


def _acceptance_tags(acc: dict) -> list[str]:
    """머리 태그: 요구 표별 판정 + 교집합 결론. 배치가 없으면 어느 표도 '통과' 로 보이지 않게 한다."""
    if not isinstance(acc, dict) or not acc:
        return []
    req = [str(t) for t in (acc.get("require_tables") or [])]
    per = acc.get("per_table") if isinstance(acc.get("per_table"), dict) else {}
    tags = []
    for t in req:
        tm = (per.get(t) or {}).get("tone_model")
        ko_t, en_t = _table_bi_p2(t)
        tags.append(_bi_tag_html(f"{_esc(ko_t)} 표 <b>{_esc(str(tm) + ' 통과' if tm else '떨어짐')}</b>",
                                 f"{_esc(en_t)} table <b>{_esc(str(tm) + ' passes' if tm else 'fails')}</b>",
                                 cls="tag" if tm else "tag warn"))
    if len(req) > 1:
        tags.append(_bi_tag(f"{len(req)}개 표가 모두 통과해야 함", f"all {len(req)} tables must pass", cls="tag warn"))
    return tags


def _honesty_strip(d: dict, model: dict, lo: float, hi: float, src: str) -> str:
    """정직 스트립: 배포 모델의 model_id·계수·clim·밴드, 생산 모델 M3(정보), #2 §6 판정(literal/amended), 홀드아웃,
    Phase 3 킬룰 카운트다운, 라이브 Brier, 각주.

    표현(STYLE_I18N.md): 줄마다 (쉬운 한국어, English) 두 벌을 심는다. **숫자·식별자·판정 원문은 한 글자도
    바꾸지 않는다** — 바꾸는 것은 그 앞에 붙는 이름표뿐이다(예: '홀드아웃' → '손대지 않고 남겨둔 최근 구간').
    """
    dep = deployment_of(d)
    clim = _fnum(_first(d, "clim", "p2_clim", default=model.get("clim")))
    rung = dep["prob_rung"]
    dep_ko, dep_en = (DEPLOYED_BI if dep["deployed"] else INFO_DISPLAY_BI)
    mid = model.get("model_id") or d.get("model_id") or "—"
    refit = model.get("refit_date") or "—"
    spec = str(model.get("spec_sha256") or "")[:8] or "—"
    # items = [(쉬운 한국어, English)] — 값·식별자는 두 벌에 똑같이 들어간다
    items: list[tuple[str, str]] = [
        (f"확률을 만든 모델: {rung} ({dep_ko}) · model_id {mid} (다시 맞춘 날 {refit} · spec {spec})",
         f"the model behind the probability: {rung} ({dep_en}) · model_id {mid} (refit {refit} · spec {spec})"),
        (f"계수 {_coef_txt(model)} · 평소 비율 {_pct1(clim)} · 범위 {_pct1(lo)}~{_pct1(hi)} ({src or '—'})",
         f"coefficients {_coef_txt(model)} · base rate {_pct1(clim)} · range {_pct1(lo)}-{_pct1(hi)} ({src or '—'})"),
    ]
    prod = d.get("prod_model") if isinstance(d.get("prod_model"), dict) else {}
    if prod and prod.get("model_id") != model.get("model_id"):
        items.append((f"만들어 둔 모델 M3 ({INFO_DISPLAY_BI[0]}) · model_id {prod.get('model_id') or '—'} · "
                      f"계수 {_coef_txt(prod)} — 상태도 신호등 판정도 주식 비중도 이 모델에서 나오지 않습니다",
                      f"the built model M3 ({INFO_DISPLAY_BI[1]}) · model_id {prod.get('model_id') or '—'} · "
                      f"coefficients {_coef_txt(prod)} — the state, the call and the stock weight do not come from it"))
    # 실험 #2 §6 판정
    acc = d.get("acceptance") if isinstance(d.get("acceptance"), dict) else {}
    if acc:
        def _pass_of(block, rung_):
            if not isinstance(block, dict):
                return None
            if rung_ in block and isinstance(block[rung_], dict):
                return block[rung_].get("pass")
            return block.get("pass")
        cand = acc.get("candidate") or "M3"
        amd_tone = acc.get("amended_tone_model")
        amd_blk = acc.get("amended") if isinstance(acc.get("amended"), dict) else {}
        # amended(#2a)는 후보 한 단이 아니라 사다리를 훑어 A∧B∧C 를 만족하는 최상위 단을 고른다(calibrate.acceptance).
        # 배치 근거가 된 그 단의 판정을 적는다 — 후보(M3)의 판정만 적으면 "실패인데 tones 배포" 로 읽힌다.
        amd_rung = amd_tone if (amd_tone and isinstance(amd_blk.get(amd_tone), dict)) else cand
        lit = _pass_of(acc.get("literal"), cand)
        amd = _pass_of(amd_blk, amd_rung)
        lit_txt = "통과" if lit else ("실패" if lit is not None else "—")
        lit_en = "passed" if lit else ("failed" if lit is not None else "—")
        amd_txt = "통과" if amd else ("실패" if amd is not None else "—")
        amd_en = "passed" if amd else ("failed" if amd is not None else "—")
        lit_blk = acc.get("literal", {}).get(cand) if isinstance(acc.get("literal"), dict) else None
        fb = lit_blk.get("failing_blocks") if isinstance(lit_blk, dict) else None
        prim = str(acc.get("primary_table") or "24")
        prim_ko, prim_en = _table_bi_p2(prim)
        fb_ko = f" (떨어진 구간 {fb})" if fb else ""
        fb_en = f" (failing blocks {fb})" if fb else ""
        also_ko = (f" (미리 정해둔 후보 {cand} 는 실패)"
                   if amd_rung != cand and _pass_of(amd_blk, cand) is False else "")
        also_en = (f" (the candidate fixed in advance, {cand}, failed)"
                   if amd_rung != cand and _pass_of(amd_blk, cand) is False else "")
        tone_ko = f", 이 표만 보면 신호등 판정을 맡길 후보는 {amd_tone}" if amd_tone else ""
        tone_en = f", on this table alone the step that would drive the call is {amd_tone}" if amd_tone else ""
        items.append((f"실험 #2 §6 시험({prim_ko} 표) — 미리 정해둔 규칙 그대로(literal, {cand}): {lit_txt}{fb_ko}"
                      f" · 결과를 본 뒤 완화한 규칙(amended #2a, post hoc, {amd_rung}): {amd_txt}{also_ko}{tone_ko}"
                      f" → 적용한 규칙 {acc.get('rule') or '—'}",
                      f"experiment #2 §6 test ({prim_en} table) — the rule fixed in advance (literal, {cand}): "
                      f"{lit_en}{fb_en} · the rule relaxed after seeing the results (amended #2a, post hoc, "
                      f"{amd_rung}): {amd_en}{also_en}{tone_en} → rule applied: {acc.get('rule') or '—'}"))
        verdict = acceptance_verdict_line(acc)
        deploy_txt = acc.get("deploy_mode") or d.get("deploy_mode") or "—"
        req_txt = "+".join(str(t) for t in (acc.get("require_tables") or []))
        req_ko = f" (통과해야 하는 표 {req_txt})" if req_txt else ""
        req_en = f" (tables that must pass: {req_txt})" if req_txt else ""
        if verdict:
            # 판정 원문은 계산층이 쓴 한국어라 그대로 인용한다(스팬 밖) — 이름표만 두 벌
            items.append((f"실제로 쓸지 정한 결과{req_ko} — @RAW@", f"the decision on whether to use it{req_en} — @RAW@"))
        else:
            items.append((f"실제로 쓸지 정한 결과{req_ko} — 실제 사용(deploy) {deploy_txt}",
                          f"the decision on whether to use it{req_en} — in use (deploy) {deploy_txt}"))
        none_ko = " · 실제로 쓰는 단계 없음 — 신호등 판정도 주식 비중도 주장하지 않음"
        none_en = " · no step in use — no traffic-light call and no stock weight is claimed"
        # conjunction_note 는 계산층이 쓴 한국어다 → 두 언어 칸에 복사하지 말고 스팬 밖 인용으로 뺀다
        cj = " · @RAW2@" if acc.get("conjunction_note") else ""
        items.append((f"실제 사용(deploy) {deploy_txt}"
                      + (f" · tone_model {acc.get('tone_model')}" if acc.get("tone_model") else none_ko) + cj,
                      f"in use (deploy) {deploy_txt}"
                      + (f" · tone_model {acc.get('tone_model')}" if acc.get("tone_model") else none_en) + cj))
        sens_all = acc.get("sensitivity_blocks") if isinstance(acc.get("sensitivity_blocks"), dict) else {}
        for t, sens in sens_all.items():
            if isinstance(sens, dict) and not sens.get("agrees", True):
                t_ko, t_en = _table_bi_p2(t)
                items.append((f"구간을 어떻게 자르느냐에 결론이 달라집니다 (§8.2): 같은 규칙을 {t_ko} 표로 다시 채점하면 "
                              f"실제 사용(deploy) {sens.get('deploy_mode') or '—'} · 신호등 판정을 맡을 단계 "
                              f"{sens.get('tone_model') or '없음'} — 이 표는 판정에 쓰지 않습니다",
                              f"the conclusion depends on where the blocks are cut (§8.2): scoring the same rule on the "
                              f"{t_en} table gives: in use (deploy) {sens.get('deploy_mode') or '—'} · the step driving the call "
                              f"would be {sens.get('tone_model') or 'none'} — this table is not used for the decision"))
    else:
        dm = d.get("deploy_mode") or model.get("deploy_mode") or "—"
        items.append((f"실험 #2 §6 시험 — 실제 사용(deploy) {dm} (자세한 것은 summary_p2.json 의 acceptance)",
                      f"experiment #2 §6 test — in use (deploy) {dm} (details in summary_p2.json, acceptance)"))
    ho = d.get("holdout")
    if isinstance(ho, dict) and ho:
        ci = ho.get("ci") or ho.get("ci_bss_clim")
        n_ho = _fnum(ho.get("n"))
        nb = ho.get("n_blocks", int(n_ho // 20) if not math.isnan(n_ho) else None)
        ci_txt = f" · 95% {_fmt_p2(ci, 'bss')}" if isinstance(ci, (list, tuple)) else ""
        items.append((f"손대지 않고 남겨둔 최근 구간(2024-09-03~, 한 번만 씁니다): n={_fmt(ho.get('n'), 'n')} "
                      f"(겹치지 않는 창 {_fmt(nb, 'n')}) · 평소 평균보다 나아진 정도 {_fmt_p2(ho.get('bss_clim'), 'bss')} · "
                      f"B1 대비 {_fmt_p2(ho.get('bss_vix'), 'bss')} · M1 대비 {_fmt_p2(ho.get('bss_m1'), 'bss')}{ci_txt}",
                      f"untouched recent data (from 2024-09-03, used once): n={_fmt(ho.get('n'), 'n')} "
                      f"(non-overlapping windows {_fmt(nb, 'n')}) · better than the long-run average by "
                      f"{_fmt_p2(ho.get('bss_clim'), 'bss')} · vs B1 {_fmt_p2(ho.get('bss_vix'), 'bss')} · "
                      f"vs M1 {_fmt_p2(ho.get('bss_m1'), 'bss')}{ci_txt}"))
    else:
        items.append(("손대지 않고 남겨둔 최근 구간(2024-09-01~): 아직 열지 않았습니다 — 입력에서 잘라 두었고, "
                      "마지막에 딱 한 번 채점한 뒤 여기에 적습니다",
                      "untouched recent data (from 2024-09-01): not opened yet — it is cut out of the inputs, and will be "
                      "scored exactly once at the end and written here"))
    live = d.get("live") if isinstance(d.get("live"), dict) else {}
    kr = live.get("kill_rule") if isinstance(live.get("kill_rule"), dict) else {}
    ep = live.get("episodes5_observed")
    mo = _fnum(live.get("months_elapsed"))
    ep_txt = _fmt(ep, "n") if ep is not None else "—"
    mo_txt = f"{mo:.1f}" if not math.isnan(mo) else "—"
    due_ko = " · 평가할 때가 됐습니다" if live.get("kill_rule_due") else ""
    due_en = " · it is now time to evaluate" if live.get("kill_rule_due") else ""
    items.append((f"성적이 나쁘면 끄는 규칙까지 (Phase 3): 실제로 5% 넘게 떨어진 사건 "
                  f"{ep_txt}/{kr.get('episodes_required', P2_KILL_EPISODES)} · "
                  f"지난 기간 {mo_txt}/{kr.get('months_required', P2_KILL_MONTHS)}개월{due_ko}",
                  f"countdown to the rule that switches it off (Phase 3): declines of 5% or more actually seen "
                  f"{ep_txt}/{kr.get('episodes_required', P2_KILL_EPISODES)} · months elapsed "
                  f"{mo_txt}/{kr.get('months_required', P2_KILL_MONTHS)}{due_en}"))
    n_sc = live.get("n_scored")
    srcs = live.get("prob_sources") if isinstance(live.get("prob_sources"), dict) else {}
    if n_sc:
        ci = live.get("ci_bss_clim")
        refs_ko = [f"M1 대비 {_fmt_p2(live.get('bss_m1'), 'bss')}", f"B1 대비 {_fmt_p2(live.get('bss_vix'), 'bss')}"]
        refs_en = [f"vs M1 {_fmt_p2(live.get('bss_m1'), 'bss')}", f"vs B1 {_fmt_p2(live.get('bss_vix'), 'bss')}"]
        if live.get("bss_m3") is not None:
            refs_ko.append(f"M3 대비 {_fmt_p2(live.get('bss_m3'), 'bss')}")
            refs_en.append(f"vs M3 {_fmt_p2(live.get('bss_m3'), 'bss')}")
        # '정의상 0' 은 오늘의 배포 단 이름이 아니라 **장부가 실제로 확인한 동일성**(_p2_block: np.allclose(ref, p))에만 붙인다.
        # 채점된 행이 다른 단에서 나왔으면 vs {rung} 은 0 이 아니고, 그 문장은 독자를 속인다.
        ident = (len(srcs) <= 1) and any(f"{rung.lower()} 대비 skill 은 정의상 0" in str(n) for n in (live.get("notes") or []))
        id_ko = f" — 채점한 확률이 {rung} 에서 나왔으므로 {rung} 대비는 정의상 0 입니다" if ident else ""
        id_en = f" — the scored probability came from {rung}, so vs {rung} is 0 by definition" if ident else ""
        ci_txt = f" (95% {_fmt_p2(ci, 'bss')})" if isinstance(ci, (list, tuple)) else ""
        items.append((f"실제 기록으로 매긴 점수({rung} {dep_ko}): 채점 {_fmt(n_sc, 'n')}행 "
                      f"(겹치지 않는 창 {_fmt(live.get('n_blocks'), 'n')}) · Brier {_fmt_p2(live.get('brier'), 'brier')} · "
                      f"평소 평균보다 나아진 정도 {_fmt_p2(live.get('bss_clim'), 'bss')}{ci_txt} · "
                      + " · ".join(refs_ko) + id_ko,
                      f"score from the live record ({rung} {dep_en}): {_fmt(n_sc, 'n')} rows scored "
                      f"(non-overlapping windows {_fmt(live.get('n_blocks'), 'n')}) · Brier "
                      f"{_fmt_p2(live.get('brier'), 'brier')} · better than the long-run average by "
                      f"{_fmt_p2(live.get('bss_clim'), 'bss')}{ci_txt} · " + " · ".join(refs_en) + id_en))
    elif live.get("holdout_locked"):
        items.append((f"실제 기록으로 매긴 점수: 아직 매기지 않습니다 — 남겨둔 구간을 열지 않았기 때문입니다 "
                      f"({HOLDOUT_START} 이후 행은 마지막 한 번의 검증 전까지 채점하지 않습니다 — VALIDATION.md §6)",
                      f"score from the live record: not scored yet — the untouched data has not been opened "
                      f"(rows after {HOLDOUT_START} are not scored until the single final check — VALIDATION.md §6)"))
    else:
        items.append(("실제 기록으로 매긴 점수: 아직 채점된 날이 없습니다 (결과는 20거래일이 지나야 확정됩니다)",
                      "score from the live record: no day has been scored yet (an outcome is only settled after "
                      "20 trading days)"))
    if len(srcs) > 1:
        items.append((f"주의 — 이 기록의 확률이 여러 단계에서 나왔습니다 {srcs}. 한 모델의 성적표가 아니므로, "
                      "쓰는 단계가 바뀐 경계의 앞뒤를 하나로 이어 읽지 마세요",
                      f"caution — the probabilities in this record came from more than one step {srcs}. It is not one "
                      "model's scorecard, so do not read across the boundary where the step in use changed"))
    verdict_raw = acceptance_verdict_line(acc) if acc else ""
    cj_raw = str(acc.get("conjunction_note") or "") if acc else ""
    lis = []
    for ko, en in items:
        # 계산층이 쓴 한국어 원문(@RAW@ 판정 · @RAW2@ 합산 사유)은 번역하지 않고 스팬 **밖**에 인용한다 —
        # 그래야 영어 화면에서도 같은 원문이 한 번만, 그대로 보인다.
        for mark, raw_txt in (("@RAW@", verdict_raw), ("@RAW2@", cj_raw)):
            if mark in ko:
                a_ko, b_ko = ko.split(mark, 1)
                a_en, b_en = en.split(mark, 1)
                ko, en = None, None
                lis.append(f"<li>{bi(a_ko, a_en)}{_raw(raw_txt)}{bi(b_ko, b_en)}</li>")
                break
        if ko is not None:
            lis.append(f"<li>{bi(ko, en)}</li>")
    return ('<div class="strip"><b>' + bi("숨기지 않는 것들", "What we do not hide") + '</b><ul class="plain">'
            + "".join(lis) + f'</ul><div class="note">{bi(P2_FOOTNOTE, P2_FOOTNOTE_EN)}</div></div>')

def deployment_of(d: dict) -> dict:
    """어느 단이 배포됐는지(= 헤드라인 확률의 출처)를 dict 에서 읽는다.

    반환 {deploy_mode, tone_model, prob_rung, deployed, label}. 규칙(계약: acceptance 가 진실):
      deploy_mode == "tones" → prob_rung = tone_model (그 단이 상태·톤·비중을 만든다),
      그 밖(info_only 등)     → 배포된 단이 없다 → prob_rung = M3(생산 모델)을 '정보 표시(배포 안 함)' 로만 보여준다.
    """
    d = d if isinstance(d, dict) else {}
    model = d.get("model") if isinstance(d.get("model"), dict) else {}
    deploy = str(_first(d, "deploy_mode", "p2_deploy_mode", default=model.get("deploy_mode") or "info_only"))
    tone = _first(d, "tone_model", "p2_tone_model", default=model.get("tone_model"))
    tone = None if tone is None or str(tone).strip() in ("", "None") else str(tone)
    deployed = deploy == "tones"
    rung = _first(d, "prob_rung", default=None)
    rung = str(rung) if rung else ((tone if deployed else None) or "M3")
    return {"deploy_mode": deploy, "tone_model": tone, "prob_rung": rung, "deployed": deployed,
            "label": DEPLOYED_LABEL if deployed else INFO_DISPLAY_LABEL}


def _rung_txt(rung: str | None) -> str:
    """'M1(VIX 보정만, 2 파라미터)' 형태의 짧은 설명."""
    if not rung:
        return "—"
    nm = RUNG_KO.get(rung, rung).replace(f"({rung})", "").strip()
    n = RUNG_PARAMS.get(rung)
    return f"{rung}({nm}" + (f", {n} 파라미터)" if n else ")")


def p2_card(today_p2: dict) -> str:
    """index.html 의 Phase 2 카드 조각(v0 판정 블록 아래; v0 는 그대로). 위에서 아래로 §11 1~8.

    헤드라인 확률 p 는 **배포된 단**(acceptance.tone_model)의 확률이다. 배포되지 않은 단은 사다리에 '정보 표시(배포 안 함)'
    로만 남고 상태·톤·비중을 만들지 않는다. deploy_mode 가 info_only 면 배포된 단이 없으므로 생산 모델 M3 를 정보로 보여준다.

    표현: 이 카드는 가장 많이 읽는 화면이라 문장을 전부 쉬운 한국어 + English 두 벌로 심는다 (STYLE_I18N.md §1·§4).
    숫자·구간·표본 수·post hoc·info_only·"투자 조언 아님" 은 쉬운 말로 바꾸되 **하나도 빼지 않는다**.

    today_p2 (전부 선택; 없는 값은 '—'):
      p(=배포 확률)|prob|prob_dd5_20, p_m1, p_m2, p_m3, p_vix, p_vix_bgk, p_vix_driftless, clim, lo, hi, band_src, param_band[lo,hi], calib_band[lo,hi],
      calib_bin{bin|lo,hi,n,n_eff,obs,wilson_lo,wilson_hi}, x_vix, x_har, x_ma, vix, har_vol_20, har_fc_20, rv20_cc, ts_diag,
      percentiles{열: 백분위}, context{VIX3M, SKEW, VVIX, fg …}, r, state, days_in_state, deploy_mode, tone_model, prob_rung, thresholds(decision.next_thresholds),
      churn_252, churn_alert, reason_ko, contributions(model.contributions — 배포 모델), contributions_m3(정보), dod(model.day_over_day),
      dod_5d(5일 누적 같은 형식),
      input_missing(str|None), warnings[list], events(events.upcoming), events_horizon(events.table_horizon), model{model_id, coef, intercept, clim,
      refit_date, spec_sha256, deploy_mode, tone_model} (= 배포 모델), prod_model(생산 모델 M3, 배포 단이 M3 가 아닐 때만),
      acceptance(summary_p2.acceptance), holdout, live(ledger.summary()['p2']), har_oos_log_mae, asof.
    """
    d = today_p2 if isinstance(today_p2, dict) else {}
    model = d.get("model") if isinstance(d.get("model"), dict) else {}
    p = _fnum(_first(d, "p", "prob", "prob_dd5_20"))
    clim = _fnum(_first(d, "clim", "p2_clim", default=model.get("clim")))
    p_vix, p_bgk, p_m1, p_m2 = (_fnum(_first(d, k, f"p2_{k}")) for k in ("p_vix", "p_vix_bgk", "p_m1", "p_m2"))
    dep = deployment_of(d)
    rung = dep["prob_rung"]
    p_m3 = _fnum(_first(d, "p_m3", "p2_p_m3"))
    rung_p = {"M1": p_m1, "M2": p_m2, "M3": p_m3}
    if math.isnan(rung_p.get(rung, math.nan)) and not math.isnan(p):
        rung_p[rung] = p                       # 배포 단의 값이 따로 오지 않으면 헤드라인이 곧 그 단의 값이다
        p_m1, p_m2, p_m3 = rung_p["M1"], rung_p["M2"], rung_p["M3"]
    missing = _first(d, "input_missing", "p2_input_missing")
    missing = None if _is_nan(missing) or str(missing).strip() == "" else str(missing)
    if math.isnan(p) and not missing:
        missing = f"{P2_PROB_UNAVAILABLE}: 확률 값이 없습니다(사유 미기록 — 조용한 실패 금지, 사유를 기록하세요)"
    deploy = dep["deploy_mode"]
    tones_on = dep["deployed"]
    state = _first(d, "state", "p2_state")
    state = None if state is None else str(state)
    days = _first(d, "days_in_state", "p2_days_in_state")
    r = _fnum(_first(d, "r", "p2_r"))
    lo, hi, src, pb, cb = _band_of(d)
    asof = d.get("asof")
    warns = [str(w) for w in (d.get("warnings") or []) if w is not None and str(w).strip()]
    rung_ko, rung_en = _rung_bi(rung)

    tags = ['<span class="tag">' + bi_html(f"모델 <b>{_esc(model.get('model_id') or d.get('model_id') or '—')}</b>",
                                           f"model <b>{_esc(model.get('model_id') or d.get('model_id') or '—')}</b>") + "</span>",
            (_bi_tag(f"이 판정을 실제로 씁니다 — {rung}", f"this call is in use — {rung}")
             if tones_on else _bi_tag(*INFO_ONLY_BI, cls="tag warn"))]
    if asof:
        tags.insert(0, '<span class="tag">' + bi_html(f"기준일 <b>{_esc(pd.Timestamp(asof).strftime('%Y-%m-%d'))}</b>",
                                                      f"as of <b>{_esc(pd.Timestamp(asof).strftime('%Y-%m-%d'))}</b>") + "</span>")
    if d.get("churn_alert"):
        n_churn = _fmt(d.get("churn_252"), "n")
        tags.append(_bi_tag(f"{P2_CHURN_LABEL_BI[0]} (최근 252거래일에 {n_churn}번)",
                            f"{P2_CHURN_LABEL_BI[1]} ({n_churn} times in the last 252 trading days)", cls="tag bad"))
    if tones_on:
        eye_ko = f"Phase 2 · 실제로 쓰는 확률 = {rung_ko} · 실험 #2"
        eye_en = f"Phase 2 · the probability in use = {rung_en} · experiment #2"
        if rung != "M3":
            eye_ko += f" · M3 는 {INFO_DISPLAY_BI[0]}"
            eye_en += f" · M3 is {INFO_DISPLAY_BI[1]}"
    else:
        eye_ko = f"Phase 2 · 만들어 둔 모델 {rung_ko} — {INFO_DISPLAY_BI[0]} · 실험 #2"
        eye_en = f"Phase 2 · the built model {rung_en} — {INFO_DISPLAY_BI[1]} · experiment #2"
    head = (f'<div class="eyebrow">{bi(eye_ko, eye_en)}</div>'
            + "<h2>" + bi("앞으로 20거래일(약 한 달) 안에 5% 넘게 떨어질 확률",
                          "The chance of a fall of more than 5% within the next 20 trading days (about a month)") + "</h2>"
            + f'<div class="tags">{"".join(tags)}</div>')

    # 1. 헤드라인(자연빈도) — 입력 결측이면 '확률 계산 불가'
    if missing:
        s1 = (f'<div class="verdict" style="color:#eab308">{bi(*P2_PROB_UNAVAILABLE_BI)}</div>'
              + _bi_note_html(f"{_esc(missing)} — 상태는 그대로 두고(며칠째인지도 계속 셉니다) 확률과 요인 분해만 비워 둡니다. "
                              "이유를 적지 않고 조용히 넘어가지 않습니다.",
                              f"{_esc(missing)} — the state is kept (and the day count keeps running); only the probability "
                              "and the breakdown are left blank. We never fail silently without recording why."))
    else:
        interval_txt = ""
        if not (math.isnan(lo) or math.isnan(hi)):
            both_ko, both_en = [], []
            if not (math.isnan(pb[0]) or math.isnan(pb[1])):
                both_ko.append(f"모델 계수를 흔들어 본 범위 {_pct1(pb[0], 0)}~{_pct1(pb[1], 0)}")
                both_en.append(f"range from shaking the model's coefficients {_pct1(pb[0], 0)}-{_pct1(pb[1], 0)}")
            if not (math.isnan(cb[0]) or math.isnan(cb[1])):
                both_ko.append(f"과거 같은 확률대에서 실제로 일어난 범위 {_pct1(cb[0], 0)}~{_pct1(cb[1], 0)}")
                both_en.append(f"range actually observed at this probability in the past {_pct1(cb[0], 0)}-{_pct1(cb[1], 0)}")
            interval_txt = ('<span class="mut">' + bi(f"범위 {_pct1(lo, 0)}~{_pct1(hi, 0)}",
                                                      f"range {_pct1(lo, 0)}-{_pct1(hi, 0)}") + "</span>"
                            + (_bi_note(" · ".join(both_ko) + f" — 둘 중 넓은 쪽을 씁니다({src or '—'})",
                                        " · ".join(both_en) + f" — we show the wider of the two ({src or '—'})")
                               if both_ko else ""))
        color = P2_STATE_COLORS.get(state or "", PALETTE["tx"]) if tones_on else PALETTE["tx"]
        s1 = (f'<div class="verdict" style="color:{color}">{_nat_freq_bi(p)}</div>'
              + f'<div class="v-act">= {bi("약", "about")} {_pct1(p)} ({_ten_freq_bi(p)}) {interval_txt}</div>'
              + '<div class="note">'
              + bi("평소에는", "normally it is") + f" {_nat_freq_bi(clim)} · "
              + bi("VIX 만 쓴 공식", "the VIX-only formula") + f" {_nat_freq_bi(p_vix)} · "
              + bi("VIX 를 실제와 맞춘 값", "the VIX probability matched to reality") + f" {_nat_freq_bi(p_m1)}</div>")
        # 평소와 견주기 (STYLE_I18N.md §4) — 새 계산이 아니라 위에 이미 보인 두 숫자의 비교다.
        if not (math.isnan(p) or math.isnan(clim)):
            if p > clim * 1.05:
                cmp_ko, cmp_en = "평소보다 높은 편입니다", "that is higher than usual"
            elif p < clim * 0.95:
                cmp_ko, cmp_en = "평소보다 낮은 편입니다", "that is lower than usual"
            else:
                cmp_ko, cmp_en = "평소와 비슷합니다", "that is about the same as usual"
            # 범위가 평소 값을 품으면 "뚜렷한 차이" 라고 말할 수 없다 — 그 한계를 같이 적는다
            if not (math.isnan(lo) or math.isnan(hi)) and lo <= clim <= hi:
                cmp_ko += f". 다만 범위({_pct1(lo, 0)}~{_pct1(hi, 0)})가 평소 값을 품고 있어 뚜렷한 차이라고 보기는 어렵습니다"
                cmp_en += (f", but the range ({_pct1(lo, 0)}-{_pct1(hi, 0)}) still covers the usual level, so it is not a "
                           "clear difference")
            s1 += _bi_note(cmp_ko + ". 아직 시험 운용 중이라 투자 조언이 아닙니다.",
                           cmp_en + ". This is still a trial run and is not investment advice.")
        cbin = d.get("calib_bin") if isinstance(d.get("calib_bin"), dict) else {}
        if cbin and not _is_nan(cbin.get("obs")):
            blo, bhi = _fnum(cbin.get("lo")), _fnum(cbin.get("hi"))
            has_rng = not (math.isnan(blo) or math.isnan(bhi))
            rng = f"{blo * 100:.0f}~{bhi * 100:.0f}%" if has_rng else str(cbin.get("bin") or "이 구간")
            rng_en = f"{blo * 100:.0f}-{bhi * 100:.0f}%" if has_rng else str(cbin.get("bin") or "this band")
            n_eff = _fnum(cbin.get("n_eff"))
            obs_n = int(round(_fnum(cbin.get("obs")) * 100))
            n_eff_txt = int(round(n_eff)) if not math.isnan(n_eff) else "—"
            s1 += _bi_note(f"과거에 이 모델이 {rng} 라고 말한 날들을 모아 보면, 실제로는 100번 중 {obs_n}번 일어났습니다 "
                           f"(서로 겹치지 않는 사례 {n_eff_txt}개, 95% 범위 {_pct1(cbin.get('wilson_lo'), 0)}~{_pct1(cbin.get('wilson_hi'), 0)}).",
                           f"Gathering the past days when this model said {rng_en}, it actually happened {obs_n} times in 100 "
                           f"({n_eff_txt} non-overlapping cases, 95% range {_pct1(cbin.get('wilson_lo'), 0)}-{_pct1(cbin.get('wilson_hi'), 0)}).")

    # 2. 사다리 오늘값 — 배포 단에 표시를 달고, 나머지는 정보
    ladder = [(None, "M0", p_vix), (None, "BGK", p_bgk), ("M1", "M1", p_m1), ("M2", "M2", p_m2), ("M3", "M3", p_m3)]
    cells = ""
    for rk, nm, v in ladder:
        mark = ""
        if rk and rk == rung:
            mark = f'<div class="note">{bi(*(DEPLOYED_BI if dep["deployed"] else INFO_DISPLAY_BI))}</div>'
        elif rk:
            mark = f'<div class="note">{bi(*INFO_DISPLAY_BI)}</div>'
        cells += f'<div class="tf"><div class="lb">{bi(*(LADDER_BI.get(nm) or _rung_bi(nm)))}</div><div class="mono">{_pct1(v)}</div>{mark}</div>'
    add = p_m3 - p_m1 if not (math.isnan(p_m3) or math.isnan(p_m1)) else math.nan
    ladder_note = ""
    if not math.isnan(add):
        extra_ko = (f" (M3 는 {INFO_DISPLAY_BI[0]} — 상태와 판정은 {rung} 에서 나옵니다)" if tones_on and rung != "M3" else "")
        extra_en = (f" (M3 is {INFO_DISPLAY_BI[1]} — the state and the call come from {rung})" if tones_on and rung != "M3" else "")
        ladder_note = _bi_note(f"실제로 움직인 폭과 추세를 더했을 때 확률이 바뀐 정도: {_pp1(add)} — {P2_FAMILY_LINE}{extra_ko}",
                               f"How much adding the actual movement and the trend changed the probability: {_pp1(add)} — "
                               f"{P2_FAMILY_LINE_EN}{extra_en}")
    s2 = _bi_h3("단계별 모델이 오늘 내놓은 값", "What each model step says today") + f'<div class="tfrow">{cells}</div>{ladder_note}'

    # 3. 상태
    thr = d.get("thresholds") if isinstance(d.get("thresholds"), dict) else (_thresholds_fallback(state, days, clim) if state in P2_STATE_COLORS else {})
    thr_ko, thr_en = [], []
    if thr.get("escalate_to"):
        thr_ko.append(f"{thr['escalate_to']} 로 올라가려면 확률이 {_pct1(thr.get('p_escalate'))} 이상 (평소의 {_fnum(thr.get('r_escalate')):g}배)")
        thr_en.append(f"to move up to {thr['escalate_to']}, the probability must reach {_pct1(thr.get('p_escalate'))} "
                      f"({_fnum(thr.get('r_escalate')):g}x the normal rate)")
    if thr.get("deescalate_to"):
        rem = thr.get("dwell_remaining")
        tail_ko = (f", 그리고 최소 {int(rem)}거래일은 더 유지" if isinstance(rem, (int, np.integer)) and not isinstance(rem, bool) and int(rem) > 0 else "")
        tail_en = (f", and it must stay for at least {int(rem)} more trading days" if isinstance(rem, (int, np.integer)) and not isinstance(rem, bool) and int(rem) > 0 else "")
        thr_ko.append(f"{state} 에서 내려오려면 확률이 {_pct1(thr.get('p_deescalate'))} 아래 (평소의 {_fnum(thr.get('r_deescalate')):g}배 아래){tail_ko}")
        thr_en.append(f"to come down from {state}, the probability must fall below {_pct1(thr.get('p_deescalate'))} "
                      f"(under {_fnum(thr.get('r_deescalate')):g}x the normal rate){tail_en}")
    tone = STATE_TO_TONE.get(state or "", None)
    r_ko = (f"지금 확률은 평소의 {r:.2f}배입니다" if not math.isnan(r) else "평소 대비 몇 배인지 계산할 수 없습니다")
    r_en = (f"the probability is {r:.2f}x the normal rate" if not math.isnan(r) else "cannot compute the ratio to the normal rate")
    day_ko = (f"이 상태로 {_fmt(days, 'days')}거래일째" if days is not None else "이 상태가 며칠째인지 기록이 없습니다")
    day_en = (f"{_fmt(days, 'days')} trading days in this state" if days is not None else "no record of how long this state has lasted")
    reason = d.get("reason_ko")
    # 배치 판정 한 줄(교집합) — 카드가 배포 여부를 스스로 말한다. 출처는 summary_p2.json.acceptance 하나뿐.
    acc_line = acceptance_verdict_line(d.get("acceptance") if isinstance(d.get("acceptance"), dict) else {})
    # 계산층이 쓴 한국어 원문은 번역하지 않고 그대로 인용한다. 라벨만 두 벌로 심고 원문은 스팬 **밖**에 두어
    # 영어 화면에서도 같은 원문이 보이게 한다(원문을 영어 칸 안에 넣으면 한 벌이 사라진 것처럼 보인다).
    acc_html = (f'<div class="note">{bi("판정 근거(계산이 남긴 한국어 원문)", "how that was decided (raw Korean from the pipeline)")}'
                f": {_raw(acc_line)}</div>" if acc_line else "")
    reason_html = (f'<div class="note">{bi("덧붙인 사유(계산이 남긴 한국어 원문)", "reason recorded by the pipeline (raw Korean)")}'
                   f": {_raw(reason)}</div>" if reason else "")
    if state is None:
        s3 = _bi_h3("지금 상태", "Where we are now") + _bi_note("상태 자료가 없습니다", "no state data")
    elif tones_on:
        exp_ = TONE_EXPOSURE.get(tone, None)
        s3 = (_bi_h3("지금 상태 · 신호등 판정", "Where we are now · the traffic-light call")
              + f'<div>{_p2_state_pill(state)} {_tone_pill(tone) if tone else ""}'
              + (f' <span class="mut">{bi(f"주식 비중 {int(round(exp_ * 100))}%", f"hold {int(round(exp_ * 100))}% in stocks")}</span>'
                 if exp_ is not None else "")
              + "</div>" + _bi_note(f"{r_ko} · {day_ko}", f"{r_en} · {day_en}") + reason_html
              + (_bi_note(" · ".join(thr_ko), " · ".join(thr_en)) if thr_ko else "") + acc_html)
    else:
        s3 = (_bi_h3("지금 상태 (시험 운용)", "Where we are now (trial run)")
              + f'<div class="info">{_p2_state_pill(state, grey=True)} '
              + _bi_tag(*INFO_ONLY_BI, cls="tag warn") + " " + _bi_tag(*INFO_DISPLAY_BI, cls="tag warn")
              + _bi_note(f"{r_ko} · {day_ko}", f"{r_en} · {day_en}") + reason_html
              + (_bi_note(" · ".join(thr_ko), " · ".join(thr_en)) if thr_ko else "")
              + _bi_note(f"아직 §6 채택 시험을 통과한 단계가 없어서, 실제로 쓰는 모델이 없습니다. {rung_ko} 의 확률과 상태는 "
                         "참고로만 보여 주고, 신호등 판정이나 주식 비중을 주장하지 않습니다 "
                         "(비중은 예전 v0 규칙을 그대로 따릅니다).",
                         f"No step has passed the §6 admission test yet, so no model is in use. The probability and state "
                         f"from {rung_en} are shown for information only; they do not claim a traffic-light call or a "
                         "stock weight (the weight still follows the older v0 rule).")
              + acc_html + "</div>")
    if d.get("churn_alert"):
        s3 += ('<div class="tag bad">'
               + bi(f"{P2_CHURN_LABEL_BI[0]} — 최근 252거래일 동안 {DECISION_P2['churn_alert']}번보다 많이 바뀌었습니다. "
                    "장부 §8 에서 살펴볼 일이며, 저절로 다시 맞추지는 않습니다.",
                    f"{P2_CHURN_LABEL_BI[1]} — it changed more than {DECISION_P2['churn_alert']} times in the last 252 "
                    "trading days. That is reviewed in ledger §8; nothing is re-tuned automatically.")
               + "</div>")

    # 4. 귀속 — 배포 모델 기준(헤드라인 확률을 만든 그 모델)
    rows = _contrib_rows(d)
    b0_p = next((r_["pp"] for r_ in rows if r_["term"] == "intercept"), math.nan)
    n_terms = len([r_ for r_ in rows if r_["term"] != "intercept"])
    s4 = (_bi_h3(f"이 확률이 어떻게 만들어졌나 — {rung} 기준", f"How this probability was built — from {rung}")
          + (_bi_note(f"아무 정보도 안 쓴 출발점이 {_pct1(b0_p)} 이고, 거기에 요인 {n_terms}개를 차례로 더해 {_pct1(p)} 가 됐습니다.",
                      f"The starting point with no information is {_pct1(b0_p)}; adding the {n_terms} factors one by one "
                      f"gives {_pct1(p)}.") if not math.isnan(b0_p) else "")
          + _bars_html(rows) + _dod_line(d.get("dod"), "어제 대비", "vs yesterday")
          + _dod_line(d.get("dod_5d"), "5일 누적", "5-day total"))
    m3_rows = [r_ for r_ in _contrib_rows(d, key="contributions_m3") if r_["term"] != "intercept"]
    if m3_rows:
        parts_html = " · ".join(f'{_bi_label(r_["term"])} {_pp1(r_["pp"])}' for r_ in m3_rows)
        s4 += _bi_note_html(f"참고 — 같은 날 만들어 둔 모델 M3({INFO_DISPLAY_BI[0]})의 요인 분해: {parts_html}",
                            f"For reference — the same day's breakdown from the built model M3 ({INFO_DISPLAY_BI[1]}): {parts_html}")

    # 5. 요인 문맥 (표시 전용)
    pct = d.get("percentiles") if isinstance(d.get("percentiles"), dict) else {}
    frows = []
    for k in ("vix", "x_vix", "har_vol_20", "x_har", "x_ma", "rv20_cc", "ts_diag"):
        v = _first(d, k, f"p2_{k}")
        if v is None and k not in pct:
            continue
        note = _CTX_NOTES_BI.get(k)
        frows.append({"name": k, "value": v, "pct": pct.get(k),
                      "note": _Html(bi(note[0], note[1])) if note else ""})
    ctx = d.get("context") if isinstance(d.get("context"), dict) else {}
    ctx_txt = " · ".join(f"{k} {_fmt_p2(v, k)}" for k, v in ctx.items() if _scalar_ok(v))
    s5 = (_bi_h3("요인들이 지금 어디쯤 있나 (보여 주기만 — 최근 10년 안에서의 위치, 모델 입력 아님)",
                 "Where each factor stands right now (display only — its place within the last 10 years, not a model input)")
          + _bi_table(frows, ["name", "value", "pct", "note"], ("요인 자료가 없습니다", "no factor data"))
          + (f'<div class="note">{bi("변동성 주변 값", "volatility context")}: {_raw(ctx_txt)}</div>' if ctx else ""))

    # 6. 보조 A — 변동성
    har = _fnum(_first(d, "har_fc_20", "p2_har_fc_20"))
    vix = _fnum(_first(d, "vix", "vix_close"))
    rv20 = _fnum(_first(d, "rv20_cc"))
    prem = (vix / 100.0 - har) if not (math.isnan(vix) or math.isnan(har)) else math.nan
    mae = _fnum(_first(d, "har_oos_log_mae", "har_log_mae"))
    vix_txt = f"{vix:.1f}%" if not math.isnan(vix) else "—"
    s6 = (_bi_h3("곁들여 보는 것 A — 앞으로 얼마나 출렁일지 (확률 모델 밖의 별도 계산)",
                 "Alongside A — how much it is expected to swing (a separate calculation, outside the probability model)")
          + _bi_note(f"예상 {_pct1(har)} · VIX {vix_txt} · 최근 20일 동안 실제로 움직인 폭 {_pct1(rv20)}"
                     + (f" · 둘의 차이 {_pp1(prem)} (옵션 시장이 더 비싸게 보는 정도)" if not math.isnan(prem) else "")
                     + (f" · 이 예상이 과거에 빗나간 정도 {mae:.3f}" if not math.isnan(mae) else ""),
                     f"expected {_pct1(har)} · VIX {vix_txt} · how much it actually moved over the last 20 days {_pct1(rv20)}"
                     + (f" · the gap between them {_pp1(prem)} (how much dearer the options market prices it)" if not math.isnan(prem) else "")
                     + (f" · how far this forecast missed in the past {mae:.3f}" if not math.isnan(mae) else "")))

    # 7. 보조 B — 이벤트
    s7 = _bi_h3("곁들여 보는 것 B — 앞으로 20거래일 안의 일정 (보여 주기만)",
                "Alongside B — what is scheduled in the next 20 trading days (display only)") + _events_html(d)

    # 8. 정직 스트립
    s8 = _honesty_strip(d, model, lo, hi, src)
    wsec = ((_bi_h3("이 카드가 남긴 경고", "Warnings recorded for this card")
             + _bi_note("아래 경고문은 계산 과정이 남긴 기록이라 한국어 원문 그대로 둡니다.",
                        "The warnings below are raw diagnostics written by the pipeline and are kept in the original Korean.")
             + '<ul class="warnlist">'
             + "".join(f'<li><span class="raw mono">{_esc(w)}</span></li>' for w in warns)
             + "</ul>") if warns else "")
    return f'<section class="panel" id="p2">{head}{s1}{s2}{s3}{s4}{s5}{s6}{s7}{wsec}{s8}</section>'

# ------------------------------------------------------------------
# 주간 리포트 — 보정 (docs/calibration_p2.html)
# ------------------------------------------------------------------
_RULE_BI = {"literal": ("미리 정해둔 규칙 그대로 (literal)", "the rule fixed in advance (literal)"),
            "amended": ("결과를 본 뒤 완화한 규칙 (amended, post hoc)", "the relaxed rule, decided after seeing the results (amended, post hoc)")}


def _acceptance_box(acc: dict) -> str:
    """§6 literal / #2a amended 판정 상자 — **요구 표 전부의 교집합**(장부 #2d)을 표별로 펼쳐 보인다.

    판정 한 줄은 두 가지를 함께 싣는다: (1) 계산이 만든 원문 그대로, (2) 같은 사실을 쉬운 한국어와 영어로.
    어느 쪽도 상대를 대신하지 않는다 — 원문을 지우면 무엇으로 판정했는지 사라지고, 원문만 두면 읽히지 않는다.
    """
    if not isinstance(acc, dict) or not acc:
        return _bi_note("판정 자료 없음", "no verdict data")
    dm = acc.get("deploy_mode") or "—"
    ok = dm == "tones"
    color = "#22c55e" if ok else "#eab308"
    req = [str(t) for t in (acc.get("require_tables") or [])]
    per = acc.get("per_table") if isinstance(acc.get("per_table"), dict) else {}
    rule_ko, rule_en = _RULE_BI.get(str(acc.get("rule") or ""), (str(acc.get("rule") or "—"), str(acc.get("rule") or "—")))
    tables_ko = " + ".join(_table_bi_p2(t)[0] for t in req)
    tables_en = " + ".join(_table_bi_p2(t)[1] for t in req)
    head_ko = ("신호등 판정에 사용(tones)" if ok else "정보 제공 전용(info_only)") + f" — 적용 규칙: {rule_ko}"
    head_en = ("used for the traffic-light call (tones)" if ok else "information only, nothing in use (info_only)") + f" — rule applied: {rule_en}"
    if req:
        head_ko += f" · 판정에 쓴 표 {tables_ko}"
        head_en += f" · tables used for the verdict: {tables_en}"
    if acc.get("tone_model"):
        head_ko += f" · 신호등 판정을 만드는 단계 {acc.get('tone_model')}"
        head_en += f" · the step that drives the call: {acc.get('tone_model')}"
    else:
        head_ko += " · 실제로 쓰는 단계 없음"
        head_en += " · no step is in use"
    head = (f'<div class="acc" style="border-color:{color}">'
            f'<div class="verdict" style="color:{color};font-size:18px">{bi(head_ko, head_en)}</div>')
    verdict = acceptance_verdict_line(acc)
    v_ko, v_en = acceptance_verdict_bi(acc)
    if v_ko or v_en:
        head += _bi_note_html(f"<b>실제 사용 판정</b>: {_esc(v_ko)}", f"<b>Deployment verdict</b>: {_esc(v_en)}",
                              style=f"color:{color}")
    if verdict:
        head += ('<div class="note">' + bi("판정 원문(계산이 그대로 남긴 줄):", "The verdict line exactly as the run wrote it (Korean):")
                 + " " + _raw(verdict) + "</div>")
    if req:
        rows = [{"table": _Html(bi(*_table_bi_p2(t))), "required": bool((per.get(t) or {}).get("required", True)),
                 "alone_tone": (per.get(t) or {}).get("tone_model"), "alone_deploy": (per.get(t) or {}).get("deploy_mode"),
                 "passing_rungs": ", ".join((per.get(t) or {}).get("passing_rungs") or []) or _Html(bi("없음", "none of them")),
                 "n_blocks": (per.get(t) or {}).get("n_blocks"), "min_pass_blocks": (per.get(t) or {}).get("min_pass_blocks")}
                for t in req]
        head += (_bi_h3("표별 판정 — 실제 사용은 이 표들이 모두 통과할 때만",
                        "Verdict per table — it is used only when every table passes")
                 + _bi_note_html("아래 두 열은 <b>그 표 하나만 봤을 때</b>의 결론입니다 — 실제 결론은 위의 한 줄입니다.",
                                 "The two columns below say what <b>each table alone</b> would conclude — the real conclusion is the line above.")
                 + _bi_table(rows))
    if acc.get("blocking_tables"):
        head += _bi_note_html("<b>실제 사용을 막은 표</b>: " + _esc(", ".join(_table_bi_p2(t)[0] for t in acc["blocking_tables"])),
                              "<b>Tables that blocked it</b>: " + _esc(", ".join(_table_bi_p2(t)[1] for t in acc["blocking_tables"])),
                              style="color:#eab308")
    note_bi = {"conjunction_note": ("표를 합쳐서 보는 이유", "why the tables are combined"),
               "rationale": ("판정의 자세한 근거", "the full reasoning behind the verdict"),
               "post_hoc_note": ("결과를 본 뒤 정한 부분(post hoc) 고지", "what was decided after seeing the results (post hoc)")}
    for key in ("conjunction_note", "rationale", "post_hoc_note"):
        if acc.get(key):
            ko, en = note_bi[key]
            head += f'<div class="note">{bi(ko + " (원문 그대로):", en + " (quoted as-is, Korean):")} {_raw(acc.get(key))}</div>'
    parts = []
    for tname, tbl in _acceptance_tables(acc).items():
        ko_t, en_t = _table_bi_p2(tname)
        extra_ko = "" if tbl.get("required", True) else " (참고용 — 판정에 쓰지 않음)"
        extra_en = "" if tbl.get("required", True) else " (reference only — not used for the verdict)"
        for name, title_ko, title_en in (("literal", "§6 문자 그대로 (사전 등록)", "§6 exactly as written (fixed in advance)"),
                                         ("amended", "#2a 완화안 (post hoc)", "#2a relaxed version (decided after seeing the results)")):
            blk = tbl.get(name)
            if isinstance(blk, dict) and blk and all(isinstance(v, dict) for v in blk.values()):
                recs = [{"rung": k, **{kk: vv for kk, vv in v.items() if kk != "rule_text"}} for k, v in blk.items()]
                rule_text = next((v.get("rule_text") for v in blk.values() if isinstance(v, dict) and v.get("rule_text")), None)
                parts.append(_bi_h3(f"{ko_t} 구간 표{extra_ko} — {title_ko}", f"{en_t} block table{extra_en} — {title_en}")
                             + (f'<div class="note">{bi("통과 조건(원문 그대로):", "Pass condition, quoted as-is (Korean):")} '
                                f"{_raw(rule_text)}</div>" if rule_text else "")
                             + _bi_table(recs))
            elif isinstance(blk, dict) and blk:
                parts.append(_bi_h3(f"{ko_t} 구간 표{extra_ko} — {title_ko}", f"{en_t} block table{extra_en} — {title_en}")
                             + _bi_kv({k: v for k, v in blk.items() if k != "rule_text"})
                             + (f'<div class="note">{bi("통과 조건(원문 그대로):", "Pass condition, quoted as-is (Korean):")} '
                                f"{_raw(blk.get('rule_text'))}</div>" if blk.get("rule_text") else ""))
    sens_all = acc.get("sensitivity_blocks") if isinstance(acc.get("sensitivity_blocks"), dict) else {}
    for t, sens in sens_all.items():
        if isinstance(sens, dict) and not sens.get("agrees", True):
            ko_t, en_t = _table_bi_p2(t)
            head += _bi_note_html(
                f"<b>구간을 어떻게 자르냐에 따라 결론이 달라집니다 (§8.2 블록 민감도)</b>: 같은 규칙을 {_esc(ko_t)} 구간 표로 다시 채점하면 "
                f"결론이 <b>{_esc(str(sens.get('deploy_mode') or '—'))}</b> 이고 신호등 판정을 만드는 단계는 "
                f"{_esc(str(sens.get('tone_model') or '없음'))} 입니다 — 판정은 구간을 어떻게 나누느냐에 기댑니다"
                "(이 표는 판정에 쓰지 않습니다).",
                f"<b>The conclusion depends on how the blocks are cut (§8.2 block sensitivity)</b>: scoring the same rule on the "
                f"{_esc(en_t)} block table gives <b>{_esc(str(sens.get('deploy_mode') or '—'))}</b> and the step driving the call "
                f"would be {_esc(str(sens.get('tone_model') or 'none'))} — the verdict depends on where the blocks are cut "
                "(this table is not used for the verdict).",
                style="color:#eab308")
    extra = {k: v for k, v in acc.items() if k in ("candidate", "primary_table", "literal_tone_model", "amended_tone_model",
                                                   "n_blocks", "min_pass_blocks")}
    return head + "".join(parts) + (_bi_kv(extra) if extra else "") + "</div>"


def _dep_label_bi(label) -> tuple[str, str]:
    """배포/정보 라벨 → (쉬운 한국어, English). 라벨 문자열 자체는 산출물(summary_v1.json)이 정하므로 바꾸지 않는다."""
    return _VALUE_BI.get(str(label), (str(label), str(label)))


def _rung_bi(rung: str | None) -> tuple[str, str]:
    """'M1 (VIX 확률만 실제와 맞춤, 조정 숫자 2개)' 처럼 단계 하나를 짧게 설명한다."""
    if not rung:
        return ("—", "—")
    ko_map = {"M0": "VIX 공식 그대로", "M1": "VIX 확률을 실제와 맞춤", "M2": "+ 실제로 움직인 폭", "M3": "+ 추세",
              "clim": "평소 평균", "CLIM": "평소 평균"}
    en_map = {"M0": "the VIX formula as is", "M1": "the VIX probability matched to reality", "M2": "plus how much it actually moved",
              "M3": "plus the trend", "clim": "the long-run average", "CLIM": "the long-run average"}
    n = RUNG_PARAMS.get(rung)
    ko = f"{rung} ({ko_map.get(rung, rung)}" + (f", 조정 숫자 {n}개)" if n else ")")
    en = f"{rung} ({en_map.get(rung, rung)}" + (f", {n} tuned numbers)" if n else ")")
    return (ko, en)


def render_calibration_report(summary_p2: dict, out_html: Path, charts: dict[str, bytes]) -> None:
    """docs/calibration_p2.html — ① v0 줄(영구) ② 사다리 표 ③ 블록 표 24/18/1999 + §6 판정 ④ 신뢰도·Murphy ⑤ 계수 경로·파라미터 밴드
    ⑥ 시대별 AUC ⑦ 소거·v0 참조선 ⑧ HAR-RV 성적표 ⑨ 데이터 범위·자기점검·경고·spec_sha256·FEATURE_RULE·정직 문구. 없는 키는 '자료 없음'.

    문구는 전부 bi(쉬운 한국어, English) 로 심는다 (STYLE_I18N.md). 숫자·임계값·표 스키마는 건드리지 않는다 —
    바뀐 것은 말뿐이고, 계산이 남긴 한국어(판정 원문·경고)는 _raw() 로 그대로 인용한다.
    """
    if not isinstance(summary_p2, dict):
        raise TypeError("summary_p2 는 dict 여야 합니다")
    s = summary_p2
    charts = dict(charts or {})
    run = s.get("run") if isinstance(s.get("run"), dict) else {}
    warns = [str(w) for w in (s.get("warnings") or [])]
    v0src = s.get("v0_completed") if isinstance(s.get("v0_completed"), dict) else None
    if v0src is None:
        v0src = load_summary_v0()
    v0h = v0_headline(v0src)
    if not v0src:
        warns.append("v0 completed 요약을 찾지 못해 첫 줄 숫자가 비어 있음")
    acc = s.get("acceptance") if isinstance(s.get("acceptance"), dict) else {}
    end = run.get("end") or s.get("end") or "?"
    first_refit = run.get("first_refit") or s.get("first_refit") or "?"
    sha = run.get("spec_sha256") or s.get("spec_sha256") or "—"
    rule = run.get("feature_rule") or s.get("feature_rule") or "—"
    holdout = s.get("holdout") if isinstance(s.get("holdout"), dict) else None

    dep_mode = acc.get("deploy_mode") or "—"
    dep_ko, dep_en = _VALUE_BI.get(str(dep_mode), (str(dep_mode), str(dep_mode)))
    tags = [_bi_tag_html(f"과거만 보고 매긴 구간 <b>{_esc(first_refit)} ~ {_esc(end)}</b>",
                         f"scored only on unseen data, <b>{_esc(first_refit)} to {_esc(end)}</b>"),
            _bi_tag_html(f'설계 지문 <b class="mono">{_esc(str(sha)[:12])}</b>',
                         f'spec fingerprint <b class="mono">{_esc(str(sha)[:12])}</b>'),
            _bi_tag_html(f'만든 시각 <b>{_esc(run.get("generated_at_utc") or s.get("generated_at") or _now_str())}</b>',
                         f'generated <b>{_esc(run.get("generated_at_utc") or s.get("generated_at") or _now_str())}</b>'),
            _bi_tag_html(f"<b>{_esc(dep_ko)}</b>" + (f" · 신호등 판정을 만드는 단계 {_esc(acc.get('tone_model'))}"
                                                    if acc.get("tone_model") else " · 실제로 쓰는 단계 없음"),
                         f"<b>{_esc(dep_en)}</b>" + (f" · call comes from {_esc(acc.get('tone_model'))}"
                                                    if acc.get("tone_model") else " · no step is in use"),
                         cls="tag" if dep_mode == "tones" else "tag warn")]
    tags += _acceptance_tags(acc)
    _sens_acc = acc.get("sensitivity_blocks") if isinstance(acc.get("sensitivity_blocks"), dict) else {}
    for _t, _s in _sens_acc.items():
        if isinstance(_s, dict) and not _s.get("agrees", True):
            _ko_t, _en_t = _table_bi_p2(_t)
            tags.append(_bi_tag(f"{_ko_t} 구간 표는 결론이 다름", f"the {_en_t} block table disagrees", cls="tag warn"))
    if holdout:
        tags.append(_bi_tag("남겨둔 최근 구간을 한 번 열어 봤음", "the untouched recent data has been opened once", cls="tag bad"))
    else:
        tags.append(_bi_tag_html(f"남겨둔 최근 구간 <b>{_esc(HOLDOUT_START)} 이후는 아직 안 봄</b>",
                                 f"untouched recent data <b>{_esc(HOLDOUT_START)} onward, still sealed</b>"))
    if warns:
        tags.append(_bi_tag(f"경고 {len(warns)}건", f"{len(warns)} warnings", cls="tag warn"))
    disclosure = s.get("disclosure") or "#2 사전 관측 참조"
    head = ('<header><div class="eyebrow">'
            + bi("market-risk-lab · Phase 2 · 실험 #2 (확률을 실제와 맞춘 모델 p2)",
                 "market-risk-lab · Phase 2 · experiment #2 (the model whose probabilities are matched to reality, p2)")
            + "</div><h1>"
            + bi("확률이 얼마나 맞았나 — 과거만 보고 매긴 성적표. 단계별 모델 M0→M3 를 평소 평균·VIX 공식과 견줍니다",
                 "How good the probabilities were — scored only on data the model had not seen, comparing model steps "
                 "M0 to M3 with the long-run average and the VIX formula")
            + "</h1>"
            + _bi_note_html(
                "조정하는 숫자가 4개뿐인 식(VIX 입력·실제와 예상의 갭·추세)을 매년 1월 첫 거래일에 한 번만 다시 맞추고, "
                "결과가 겹치는 20거래일은 학습에서 뺍니다. VALIDATION §6 이 구간 길이를 '18~24개월' 로 미리 정해 두었으므로, "
                "실제로 쓰려면 24개월 표와 18개월 표 <b>모두</b>에서 통과해야 합니다(소유자 결정 2026-09-08, 장부 #2d). "
                "표본 수 옆에는 언제나 겹치지 않는 구간 수(n÷20)를 함께 적습니다.",
                "A formula with only four tuned numbers (the VIX input, the gap between what actually moved and what was "
                "expected, and the trend) is re-fitted once a year on the first trading day of January, and the 20 trading "
                "days whose outcomes overlap are removed from training. VALIDATION §6 fixed the block length at 18-24 months "
                "in advance, so nothing may be used unless it passes on <b>both</b> the 24-month and the 18-month table "
                "(owner decision 2026-09-08, ledger #2d). Every sample size is shown together with the number of "
                "non-overlapping windows (n/20).", cls="sub")
            + f'<div class="tags">{"".join(tags)}</div>'
            + '<div class="note">'
            + bi("공개: 2003~2024-08 성적은 설계 단계에서 이미 본 것입니다(§13 #2). 그 뒤에 규칙을 바꾼 부분은 결과를 본 뒤 정한 것입니다. 산출물의 공개 문구:",
                 "Disclosure: the 2003 to 2024-08 record was already seen while the model was being designed (§13 #2). "
                 "Anything changed in the rule after that was decided after seeing the results. "
                 "The run's own disclosure text (Korean, quoted as-is):")
            + " " + _raw(disclosure) + "</div></header>")
    sec_titles = [("① v0 요약(영구)", "① v0 summary (always shown)"),
                  ("② 단계별 모델", "② model steps"),
                  ("③ 구간 표·§6 판정", "③ block tables and the §6 verdict"),
                  ("④ 확률이 맞았나·나눠 보기", "④ were the stated % right, and why"),
                  ("⑤ 계수 경로", "⑤ coefficient path"),
                  ("⑥ 시대별 판별력", "⑥ ranking power by era"),
                  ("⑦ 빼고 돌려보기·참조선", "⑦ leave-one-out runs and the reference line"),
                  ("⑧ 변동성 보조(HAR-RV)", "⑧ volatility side-model (HAR-RV)"),
                  ("⑨ 자료·경고·정직", "⑨ data, warnings, honesty")]
    nav = '<nav class="nav">' + "".join(f'<a href="#s{i + 1}">{bi(t[0], t[1])}</a>' for i, t in enumerate(sec_titles)) + "</nav>"

    # ① v0 completed 요약 줄(영구)
    v0ref = s.get("v0_reference") if isinstance(s.get("v0_reference"), dict) else {}
    v0ref_flat = {k: v for k, v in v0ref.items() if _scalar_ok(v)}
    s1 = ('<section class="panel" id="s1">'
          + _bi_h2("① 예전 규칙 v0 의 성적 (영구 표기)", "① the older v0 rule's record (always shown)")
          + _v0_line_html(v0h, bi_mode=True)
          + ((_bi_note("v0 참조선 — v0 점수를 모델 밖에서 확률로 바꾼 것(Platt, 숫자 2개)입니다.",
                       "The v0 reference line — the old v0 score turned into a probability outside the model "
                       "(Platt, 2 numbers).")
              + _bi_kv(_quote_machine(v0ref_flat))) if v0ref_flat else "")
          + "</section>")

    # ② 사다리
    ladder = _ladder_records(s.get("ladder"))
    lad_all = [r for r in ladder if str(r.get("block")) == "all"]
    lad_blk = [r for r in ladder if str(r.get("block")) != "all"]
    rungs = _to_records(s.get("rungs") or s.get("rung_table"))
    for r in rungs:
        if "rung" not in r and "key" in r:
            r["rung"] = r.pop("key")
    pooled = s.get("pooled") if isinstance(s.get("pooled"), dict) else {}
    if not rungs and pooled:
        rungs = [{"rung": k, **v} for k, v in pooled.items() if isinstance(v, dict)]
    s2 = ('<section class="panel" id="s2">'
          + _bi_h2("② 단계별 모델 — M0(VIX 공식) → M1(VIX 확률을 실제와 맞춤) → M2(+실제로 움직인 폭) → M3(+추세)",
                   "② the model steps — M0 (the VIX formula) → M1 (the VIX probability matched to reality) → "
                   "M2 (plus how much the market actually moved) → M3 (plus the trend)")
          + _bi_note(
              "단계마다 시험 구간 전체를 채점합니다: 빗나간 정도(Brier), 평소 평균·VIX 공식(B1)·BGK·M1 대비 정확도, "
              "위험한 날을 앞줄에 세우는 힘(AUC). 그 아래 표는 단계 사이의 차이입니다 — (기준 확률−실제)² − (모델 확률−실제)² 의 "
              "평균(×1e-4, 양수면 뒤 단계가 낫다), 40일씩 잘라 4,000번 다시 뽑아 만든 95% 범위, 우연인지 보는 값 DM t(HAC lag 19), "
              "그리고 시작일을 20가지로 바꿔 본 점수의 최소·가운데·최대.",
              "Every step is scored on the whole test stretch: the error score (Brier), the accuracy gain over the long-run "
              "average, the VIX formula (B1), BGK and M1, and the power to rank risky days first (AUC). The table below it "
              "compares neighbouring steps: the average of (baseline−outcome)² − (model−outcome)² (×1e-4, positive means the "
              "later step is better), a 95% range from 4,000 resamples of 40-day chunks, the DM t statistic (HAC lag 19) for "
              "whether the gap is luck, and the smallest, middle and largest score across 20 different start days.")
          + _bi_note(P2_HONESTY_ITEMS[1], P2_HONESTY_ITEMS_EN[1])
          + _bi_h3("단계별 채점 (전체 구간)", "score of each step (whole stretch)")
          + _bi_table(rungs, _RUNG_ORDER, ("단계별 채점 자료 없음", "no per-step scores"))
          + _bi_h3("이웃한 단계 비교 (전체 구간)", "neighbouring steps compared (whole stretch)")
          + _bi_table(lad_all, _LADDER_ORDER, ("단계 비교 자료 없음", "no step comparison"))
          + _bi_img(charts.pop("ladder_ci", None),
                    "단계 사이의 차이(×1e-4)와 40일씩 잘라 다시 뽑은 95% 범위 — 막대가 0 오른쪽에 있으면 뒤 단계가 낫다는 뜻입니다.",
                    "Gap between neighbouring steps (×1e-4) with a 95% range from resampled 40-day chunks — a bar to the "
                    "right of zero means the later step is better.",
                    "단계 비교 그림", "Ladder loss-diff CI")
          + (_bi_h3("이웃한 단계 비교 (구간별)", "neighbouring steps compared (block by block)")
             + _bi_table(lad_blk, _LADDER_ORDER) if lad_blk else "")
          + "</section>")

    # ③ 블록 표 + 판정
    b24 = _to_records(s.get("blocks24"))
    # 민감도 표는 배포 단으로 — 판정이 블록 정의에 의존하는지 보려면 그 단의 18개월 블록을 봐야 한다
    dep3 = deployment_of(acc)["prob_rung"]
    b18 = _to_records((s.get("blocks18_by_rung") or {}).get(dep3)) or _to_records(s.get("blocks18"))
    b18_rung = dep3 if _to_records((s.get("blocks18_by_rung") or {}).get(dep3)) else "M3"
    b99 = _to_records(s.get("blocks_from1999") or s.get("blocks24_from_1999") or s.get("blocks1999"))
    _req3 = [str(t) for t in (acc.get("require_tables") or [])]
    _b24_role = "판정" if "24" in _req3 else ("민감도" if _req3 else "주")     # require_tables 가 없는 예전 산출물은 옛 라벨 그대로
    _b18_role = "판정" if "18" in _req3 else "민감도"
    _role_en = {"판정": "used for the verdict", "민감도": "reference only", "주": "main"}
    s3 = ('<section class="panel" id="s3">'
          + _bi_h2("③ 구간을 잘라 본 표 (24개월 · 18개월 · 1999 시작) 와 §6 판정",
                   "③ the block tables (24-month · 18-month · from 1999) and the §6 verdict")
          + _bi_note(P2_HONESTY_ITEMS[0], P2_HONESTY_ITEMS_EN[0])
          + _acceptance_box(acc)
          + _bi_img(charts.pop("block_skill", None),
                    "구간마다 얼마나 더 정확했나 — 견주는 상대 4가지(평소 평균·VIX 공식 B1·BGK·M1). 0 아래면 그 구간에서 진 것입니다.",
                    "How much more accurate in each block, against four yardsticks (the long-run average, the VIX formula B1, "
                    "BGK and M1). Below zero means it lost in that block.",
                    "구간별 정확도 막대", "Block skill bars")
          + _bi_h3(f"24개월씩 자른 구간 — 24개월 블록({_b24_role}, M3 생산 모델)",
                   f"cut into 24-month blocks ({_role_en.get(_b24_role, _b24_role)}, production model M3)")
          + _bi_table(b24, _BLOCK_ORDER, ("24개월 구간 자료 없음", "no 24-month blocks"))
          + _bi_h3(f"18개월씩 자른 구간 — 18개월 블록({_b18_role}, {b18_rung})",
                   f"cut into 18-month blocks ({_role_en.get(_b18_role, _b18_role)}, {b18_rung})")
          + _bi_table(b18, _BLOCK_ORDER, ("18개월 구간 자료 없음", "no 18-month blocks"))
          + _bi_h3("1999 시작 — 2000~02 약세장까지 시험 구간에 넣어 본 것(참고용, M3)",
                   "starting from 1999 — the 2000-02 bear market included in the test stretch (reference only, M3)")
          + _bi_table(b99, _BLOCK_ORDER, ("1999 시작 구간 자료 없음", "no blocks from 1999"))
          + ((_bi_h3("손대지 않고 남겨둔 최근 구간 (2024-09-03~, 단 한 번만 여는 검증)",
                     "the untouched recent stretch (2024-09-03 onward, opened once and only once)")
              + _bi_kv(holdout)) if holdout else "")
          + "</section>")

    # ④ 신뢰도·Murphy — 배포 단(acceptance 가 진실)의 표를 싣는다. 다른 단의 표를 대신 쓰지 않는다.
    dep4 = deployment_of(acc)
    rung4 = dep4["prob_rung"]
    rel = _to_records(s.get("reliability" if rung4 == "M3" else f"reliability_{rung4.lower()}"))
    if not rel:
        warns.append(f"신뢰도: summary_p2.json 에 {rung4} 단의 표가 없어 ④ 를 비움(다른 단의 표를 대신 쓰지 않는다)")
    mur_key = "murphy" if rung4 == "M3" else f"murphy_{rung4.lower()}"
    mur = s.get(mur_key) if isinstance(s.get(mur_key), dict) else {}
    _lab4_ko, _lab4_en = _dep_label_bi(dep4["label"])
    _bins_txt = ", ".join(f"{b:g}" for b in P2["reliability_bins"])
    s4 = ('<section class="panel" id="s4">'
          + _bi_h2(f"④ 말한 확률이 실제와 맞았나 — {rung4} ({_lab4_ko}) · 확률 구간은 미리 정해 고정 · 나눠 보기 (Murphy 분해)",
                   f"④ did the stated % match reality — {rung4} ({_lab4_en}) · the probability bands were fixed in advance · "
                   f"broken into parts (Murphy decomposition)")
          + _bi_note(f"확률 구간 {_bins_txt} 은 미리 정해 두고 바꾸지 않습니다. 겹치는 날을 빼고 센 독립 사례 수(n÷20)로 "
                     "Wilson 95% 범위를 그립니다.",
                     f"The probability bands {_bins_txt} were fixed in advance and are not changed. The 95% Wilson range uses "
                     "the number of independent cases (n/20), not the raw day count.")
          + _bi_note(P2_HONESTY_ITEMS[2], P2_HONESTY_ITEMS_EN[2])
          + _bi_img(charts.pop("reliability", None),
                    "구간마다 모델이 말한 평균 확률과 실제로 일어난 비율을 견줍니다(세로 막대는 Wilson 95% 범위). "
                    "점이 대각선 위에 있으면 말한 확률이 실제와 딱 맞았다는 뜻입니다.",
                    "For each band, the average probability the model stated against how often it actually happened "
                    "(the bars are the 95% Wilson range). A point on the diagonal means the stated % matched reality exactly.",
                    "확률이 맞았나 그림", "Reliability diagram")
          + _bi_table(rel, _RELIAB_ORDER, (f"{rung4} 단계의 표가 없음", f"no table for the {rung4} step"))
          + (_bi_h3(f"빗나간 정도를 셋으로 나눠 보기 — Murphy 분해 ({rung4})",
                    f"the error split into three parts — Murphy decomposition ({rung4})") + _bi_kv(mur) if mur else "")
          + "</section>")

    # ⑤ 계수 경로·파라미터 밴드
    params = _param_records(s.get("params_by_refit"), "M3") or _param_records(s.get("params_by_refit"), None)
    band = s.get("param_band") if isinstance(s.get("param_band"), (list, tuple)) else None
    live = s.get("live_model") if isinstance(s.get("live_model"), dict) else (s.get("model") if isinstance(s.get("model"), dict) else {})
    s5 = ('<section class="panel" id="s5">'
          + _bi_h2("⑤ 계수가 해마다 어떻게 움직였나 (1년에 한 번만 다시 맞춤, M3 생산 모델) · 오늘 값에 적용했을 때의 범위",
                   "⑤ how the coefficients moved year by year (re-fitted once a year, production model M3) and the range "
                   "they give on today's inputs")
          + _bi_note("조정하는 숫자는 정확히 4개입니다 (기본값 b0, VIX 계수 b1, 실제-예상 갭 계수 b2, 추세 계수 b3). "
                     "다섯 번째 자리는 비워 두고 장부에 예약해 두었습니다(#3~#5). "
                     "아래 범위는 최근 5번의 다시 맞춘 계수와 지금 쓰는 모델을 오늘 입력에 넣었을 때의 최소~최대입니다.",
                     "Exactly four numbers are tuned (the intercept b0, the VIX coefficient b1, the realised-minus-implied "
                     "coefficient b2 and the trend coefficient b3). A fifth slot is deliberately left empty and reserved in "
                     "the ledger (#3 to #5). The range below is the smallest and largest probability the last five refits and "
                     "the live model give on today's inputs.")
          + _bi_img(charts.pop("coef_path", None), "다시 맞출 때마다 계수가 어떻게 움직였는지(M3).",
                    "How each coefficient moved at every refit (M3).", "계수 경로 그림", "Coefficient path")
          + _bi_table(params, _PARAM_ORDER, ("다시 맞춘 기록 없음", "no refit history"))
          + (_bi_note(f'오늘 값에 적용했을 때의 범위: {_fmt_p2(list(band), "p")}',
                      f'range on today\'s inputs: {_fmt_p2(list(band), "p")}') if band is not None else "")
          + ((_bi_h3("지금 실제로 쓰는 모델 (model_p2.json)", "the model actually in use (model_p2.json)")
              + _bi_kv(_quote_machine({k: v for k, v in live.items() if k != "extra"}))) if live else "")
          + "</section>")

    # ⑥ 시대별 AUC
    era = _to_records(s.get("era_auc"))
    s6 = ('<section class="panel" id="s6">'
          + _bi_h2("⑥ 시대가 바뀌면 판별력도 바뀐다 — VIX 의 힘이 0.72 에서 0.57 로 약해진 것을 늘 함께 보여 줍니다",
                   "⑥ ranking power changes with the era — the VIX input weakened from 0.72 to 0.57, and we always show it")
          + _bi_note("원래 입력(VIX 입력, 실제-예상 갭, 추세를 뒤집은 값)은 전 기간으로, M1·M3 확률은 시험 구간 행만 셉니다. "
                     "이 점수는 순서만 보기 때문에 확률을 실제와 맞췄는지와는 상관이 없습니다.",
                     "The raw inputs (the VIX input, the realised-minus-implied gap and the trend with its sign flipped) are "
                     "measured over the full history, while the M1 and M3 probabilities use test rows only. This score looks "
                     "only at the ordering, so it is unaffected by whether the stated % matches reality.")
          + _bi_img(charts.pop("era_auc", None), "시대별 줄 세우기 힘 (입력과 모델).", "Ranking power by era (inputs and models).",
                    "시대별 판별력 그림", "Era AUC")
          + _bi_table(era, _ERA_ORDER, ("시대별 자료 없음", "no per-era scores")) + "</section>")

    # ⑦ 소거·민감도·v0 참조선
    abl = s.get("ablations")
    if isinstance(abl, dict) and abl and all(isinstance(v, dict) for v in abl.values()):
        abl_recs = [{"name": k, **v} for k, v in abl.items()]
    else:
        abl_recs = _to_records(abl)
    csens = s.get("c_sensitivity") or s.get("sensitivity_C")
    if isinstance(csens, dict) and csens and all(isinstance(v, dict) for v in csens.values()):
        csens_recs = [{"C": k, **v} for k, v in csens.items()]
    else:
        csens_recs = _to_records(csens)
    s7 = ('<section class="panel" id="s7">'
          + _bi_h2("⑦ 일부러 빼고 돌려 본 실험 (M3-PK · M3-HAR96 · C 민감도) · v0 참조선",
                   "⑦ runs with one piece deliberately removed (M3-PK · M3-HAR96 · C sensitivity) and the v0 reference line")
          + _bi_note("빼고 돌려 본 결과는 이름을 붙여 기록만 하고, 어떤 모델을 쓸지 고르는 데는 쓰지 않습니다. "
                     "v0 참조선은 v0 종합점수를 모델 밖에서 확률로 바꾼 것(Platt 2)이며 2017-01-03 다시 맞춤부터입니다 — "
                     "조정 숫자 한도 밖이고 모델 밖이라 주간 페이지에서 견주는 선으로만 씁니다.",
                     "These runs are recorded under a name and never used to choose a model. The v0 reference line turns the "
                     "old v0 score into a probability outside the model (Platt, 2 numbers) from the 2017-01-03 refit onward — "
                     "it sits outside the tuned-number budget and outside the model, and serves only as a line to compare "
                     "against on this weekly page.")
          + _bi_h3("하나씩 빼고 돌린 결과", "runs with one piece removed")
          + _bi_table(abl_recs, None, ("빼고 돌린 결과 없음", "no such runs"))
          + (_bi_h3("C 값을 바꿔 본 결과", "changing the C setting") + _bi_table(csens_recs) if csens_recs else "")
          + _bi_h3("v0 참조선", "the v0 reference line")
          + (_bi_kv(_quote_machine({k: v for k, v in v0ref.items() if k not in ("params",)})) if v0ref else
             _bi_note("v0 참조선 없음 — 창 규칙이나 지문이 맞지 않으면 싣지 않습니다.",
                      "No v0 reference line — it is left out when the window rule or the fingerprint does not match."))
          + "</section>")

    # ⑧ HAR-RV
    har = s.get("har") if isinstance(s.get("har"), dict) else {}
    har_sc = _to_records(har.get("scorecard"))
    har_sens = _to_records(har.get("scorecard_sensitivity") or har.get("sensitivity"))
    har_coef = har.get("live_coef") if isinstance(har.get("live_coef"), dict) else {}
    har_path = _to_records(har.get("coef_path") or har.get("params_by_refit"))
    s8 = ('<section class="panel" id="s8">'
          + _bi_h2("⑧ 앞으로 얼마나 움직일까 — 보조 모델 성적표 (log-HAR, 조정 숫자 4개. 한도 밖이라 확률과 절대 합치지 않습니다)",
                   "⑧ how much will it move — the side model's record (log-HAR, 4 tuned numbers; outside the budget and never "
                   "merged into the probability)")
          + _bi_note("앞으로 20일 움직임의 로그값을 어제·최근 5일·최근 22일 움직임으로 설명합니다. 1년에 한 번만 다시 맞추고 "
                     "겹치는 20일은 뺍니다. 같은 잣대로 VIX(σ̂=VIX/100)와 '최근 22일이 그대로 이어진다' 가정도 함께 채점하고, "
                     "학습을 1996년부터 시작했을 때도 따로 보여 줍니다.",
                     "It explains the log of the next 20 days' movement using yesterday, the last 5 days and the last 22 days. "
                     "It is re-fitted once a year with the overlapping 20 days removed. The same yardstick also scores VIX "
                     "(sigma = VIX/100) and simply assuming the last 22 days continue, and we also show what happens when "
                     "training starts in 1996.")
          + _bi_img(charts.pop("har", None), "예측한 움직임 폭과 실제로 움직인 폭(20일).",
                    "Forecast movement against what actually happened over 20 days.", "변동성 예측 그림", "HAR forecast")
          + _bi_h3("성적표", "the record")
          + _bi_table(har_sc, _HAR_ORDER, ("보조 모델 성적 없음", "no side-model score"))
          + (_bi_h3("학습을 1996-01-02 부터 시작했을 때", "when training starts on 1996-01-02")
             + _bi_table(har_sens, _HAR_ORDER) if har_sens else "")
          + (_bi_h3("지금 쓰는 계수", "the coefficients in use") + _bi_kv(har_coef) if har_coef else "")
          + (_bi_h3("다시 맞출 때마다의 계수", "coefficients at each refit") + _bi_table(har_path) if har_path else "")
          + (('<div class="note">' + bi("보조 모델 경고(원문 그대로):", "side-model warnings, quoted as-is (Korean):")
              + " " + _raw(" · ".join(str(w) for w in har.get("warnings", []))) + "</div>") if har.get("warnings") else "")
          + "</section>")

    # ⑨ 데이터 범위·자기점검·경고·spec·정직
    selftest = s.get("selftest") if isinstance(s.get("selftest"), dict) else {}
    cache = run.get("cache") if isinstance(run.get("cache"), dict) else {}
    # 기계가 만든 긴 규칙 문자열(입력 규칙·창 규칙)은 번역하지 않고 원문 그대로 인용한다 — 한 글자만 달라도 다른 규칙이다
    run_flat = _quote_machine({k: v for k, v in run.items() if _scalar_ok(v)})
    s9 = ('<section class="panel" id="s9">'
          + _bi_h2("⑨ 쓴 자료 · 자기점검 · 경고 · 정직 문구", "⑨ the data used, self-checks, warnings and the honesty notes")
          + _bi_h3("이번 실행", "this run") + _bi_kv(run_flat, ("실행 정보 없음", "no run information"))
          + (_bi_h3("받아 둔 자료(캐시)", "cached data") + _bi_kv(cache) if cache else "")
          + _bi_h3("자기점검 (하루 움직임 폭 계산끼리 견주기 · 1993~95 자료 품질)",
                   "self-check (the two ways of measuring a day's range against each other, and 1993-95 data quality)")
          + (_bi_kv(selftest) if selftest else _bi_note("자기점검 결과 없음", "no self-check results"))
          + f'<div class="note mono">{bi("설계 지문 spec_sha256", "spec fingerprint spec_sha256")} {_raw(sha)}<br>'
          + f'{bi("입력을 만드는 규칙 FEATURE_RULE", "the rule that builds the inputs, FEATURE_RULE")} {_raw(rule)}</div>'
          + _bi_h3("경고", "warnings") + _bi_warn_list(warns)
          + _bi_h3("정직 문구와 남는 위험 (§16)", "honesty notes and the risks that remain (§16)")
          + '<ol class="method">'
          + "".join(f"<li>{bi(t, P2_HONESTY_ITEMS_EN[i] if i < len(P2_HONESTY_ITEMS_EN) else t)}</li>"
                    for i, t in enumerate(P2_HONESTY_ITEMS)) + "</ol>"
          + _bi_note(P2_FOOTNOTE, P2_FOOTNOTE_EN))
    for k, png in list(charts.items()):
        s9 += _bi_img(png, f"추가 차트: {k}", f"extra chart: {k}", f"추가 차트 {k}", f"extra chart {k}")
    s9 += ('<div class="foot">'
           + bi(f'market-risk-lab · 실험 #2 · 만든 시각 {run.get("generated_at_utc") or _now_str()}',
                f'market-risk-lab · experiment #2 · generated {run.get("generated_at_utc") or _now_str()}')
           + f' · <a href="index.html">{bi("오늘 판정 보기", "the call for today")}</a>'
           + f' · <a href="backtest_v1.html">{bi("판정 규칙 v1", "decision rule v1")}</a>'
           + f' · <a href="backtest_v0.html">{bi("예전 규칙 v0 시험", "the older v0 backtest")}</a></div></section>')
    _write_html(out_html, "Probability check p2 — market-risk-lab", head + nav + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9)


# ------------------------------------------------------------------
# 주간 리포트 — 결정층 v1 (docs/backtest_v1.html)
# ------------------------------------------------------------------
# 비교표의 열 이름 — 예전 이름(v1 결정층 · v0 톤 · 보유(B&H))은 다른 페이지·문서가 이 이름으로 가리키므로 괄호로 남긴다
_ALLOC_COLS_BI = {"v1 결정층": ("v1 판정 규칙 (v1 결정층)", "the v1 decision rule"),
                  "v0 톤": ("v0 신호등 판정 (v0 톤)", "the v0 traffic-light calls"),
                  "보유(B&H)": ("그냥 들고 있기 (보유(B&H))", "buy and hold (B&H)")}


def _alloc_matrix(v1_alloc: dict, v0_alloc: dict) -> str:
    """배분 시뮬 비교표: v1 / v0 / 보유 (열) × 지표(행)."""
    a1 = v1_alloc if isinstance(v1_alloc, dict) else {}
    a0 = v0_alloc if isinstance(v0_alloc, dict) else {}
    if not a1 and not a0:
        return _bi_note("배분 결과 없음 (allocation 결과 없음)", "no allocation results")
    bh_src = a1 if a1 else a0
    bh = {k: bh_src.get(f"bh_{k}") for k in ("cagr", "max_dd", "worst_month", "total_return", "ann_vol")}
    bh.update({"n_switches": 0, "switches_per_year": 0.0, "avg_exposure": 1.0, "cost_total": 0.0,
               "n_days": bh_src.get("n_days"), "years": bh_src.get("years"), "start": bh_src.get("start"), "end": bh_src.get("end")})
    cols = {"v1 결정층": {k: a1.get(k) for k in _ALLOC_KEYS}, "v0 톤": {k: a0.get(k) for k in _ALLOC_KEYS}, "보유(B&H)": bh}
    return _bi_kv(cols, cols=_ALLOC_COLS_BI)


def _info_layer_rows(s: dict) -> list[dict]:
    """summary_v1.info_layers(배포되지 않은 단) → 비교 표 행. 헤드라인 행(배포 단)을 맨 위에 둔다."""
    kp = s.get("kpis") if isinstance(s.get("kpis"), dict) else {}
    alloc = s.get("allocation") if isinstance(s.get("allocation"), dict) else {}
    dep = s.get("deployed") if isinstance(s.get("deployed"), dict) else {}
    keys = ("switches_per_year", "warn_share", "true_alarm_share", "true_alarm_share_baseline", "max_changes_any_5_sessions", "churn_alert_sessions")

    def _row(name: str, role: str, k: dict, a: dict, ep: dict | None) -> dict:
        r = {"name": name, "role": role}
        r.update({q: k.get(q) for q in keys})
        occ = k.get("occupancy") if isinstance(k.get("occupancy"), dict) else {}
        r["occupancy_caution"], r["occupancy_reduce"] = occ.get("caution"), occ.get("reduce")
        r["cagr"], r["max_dd"] = a.get("cagr"), a.get("max_dd")
        r["detection_rate"] = (ep or {}).get("detection_rate")
        r["median_lead_days"] = (ep or {}).get("median_lead_days")
        return r

    ep_head = (s.get("episodes") or {}).get("summary") if isinstance(s.get("episodes"), dict) else None
    head_name = f"{dep.get('prob_rung') or s.get('tone_model') or 'M3'}·{s.get('config_name') or 'default'}"
    rows = [_row(head_name, dep.get("label") or (DEPLOYED_LABEL if s.get("deploy_mode") == "tones" else INFO_DISPLAY_LABEL),
                 kp, alloc, ep_head if isinstance(ep_head, dict) else None)]
    info = s.get("info_layers") if isinstance(s.get("info_layers"), dict) else {}
    for rung, v in info.items():
        if not isinstance(v, dict):
            continue
        k = v.get("kpis") if isinstance(v.get("kpis"), dict) else {}
        a = v.get("allocation") if isinstance(v.get("allocation"), dict) else {}
        eps = v.get("episodes_by_threshold") if isinstance(v.get("episodes_by_threshold"), dict) else {}
        ep = (eps.get("10") or {}).get("summary") if isinstance(eps.get("10"), dict) else None
        rows.append(_row(str(v.get("label") or rung), INFO_DISPLAY_LABEL, k, a, ep if isinstance(ep, dict) else None))
    return rows


def render_backtest_v1(summary_v1: dict, summary_v0: dict, out_html: Path, charts: dict[str, bytes]) -> None:
    """docs/backtest_v1.html — v0 줄 먼저 → **배포 단**의 결정층 KPI(전환/년, 점유, 상태별 dd5, 진짜 경보 vs 기저율 vs v0,
    5세션 최대 변경) → episode_eval · allocation_sim(v1 vs v0 vs 보유) → 민감도 3종 → 배포되지 않은 단(정보) →
    상태 밴드·확률 경로·누적수익 차트 → 정직 문구.

    헤드라인(kpis/allocation/episodes/flags)은 summary_p2.json.acceptance 가 배치한 단의 확률로 돌린 결정층이다
    (2026-09-08 정정 — 그 전에는 배치와 무관하게 M3 를 헤드라인으로 실었다). 배포되지 않은 단은 ⑥ 에 '정보 표시(배포 안 함)'
    로 나란히 둔다."""
    if not isinstance(summary_v1, dict):
        raise TypeError("summary_v1 는 dict 여야 합니다")
    s = summary_v1
    charts = dict(charts or {})
    v0h = v0_headline(summary_v0 if isinstance(summary_v0, dict) else {})
    if isinstance(s.get("v0_reference"), dict):
        for k, v in s["v0_reference"].items():
            if v0h.get(k) is None and _scalar_ok(v):
                v0h[k] = v
    kp = s.get("kpis") if isinstance(s.get("kpis"), dict) else {}
    alloc = s.get("allocation") if isinstance(s.get("allocation"), dict) else (kp.get("allocation") if isinstance(kp.get("allocation"), dict) else {})
    v0_alloc = v0h.get("allocation") if isinstance(v0h.get("allocation"), dict) else {}
    run = s.get("run") if isinstance(s.get("run"), dict) else {}
    cfg = s.get("config") if isinstance(s.get("config"), dict) else (kp.get("config") if isinstance(kp.get("config"), dict) else {})
    flags = [str(f) for f in (s.get("flags") or [])]
    warns = [str(w) for w in (s.get("warnings") or [])] + [str(w) for w in (kp.get("warnings") or [])]
    deploy = s.get("deploy_mode") or run.get("deploy_mode") or "—"
    tone_model = s.get("tone_model") or run.get("tone_model")
    dep = s.get("deployed") if isinstance(s.get("deployed"), dict) else {}
    rung = dep.get("prob_rung") or (tone_model if deploy == "tones" else None) or "M3"
    deployed = bool(dep.get("deployed")) if "deployed" in dep else (deploy == "tones")
    dep_label = DEPLOYED_LABEL if deployed else INFO_DISPLAY_LABEL
    info_names = [str(k) for k in (s.get("info_layers") or {})]

    dep_ko, dep_en = _dep_label_bi(dep_label)
    rung_ko, rung_en = _rung_bi(rung)
    tags = [_bi_tag_html(f'구간 <b>{_esc(_fmt(kp.get("start") or run.get("start"), "start"))} ~ {_esc(_fmt(kp.get("end") or run.get("end"), "end"))}</b>',
                         f'period <b>{_esc(_fmt(kp.get("start") or run.get("start"), "start"))} to {_esc(_fmt(kp.get("end") or run.get("end"), "end"))}</b>'),
            _bi_tag_html(f'거래일 <b>{_fmt(kp.get("n_sessions"), "n")}</b>', f'trading days <b>{_fmt(kp.get("n_sessions"), "n")}</b>'),
            _bi_tag_html(f'설정 <b>{_esc(s.get("config_name") or run.get("config") or "default")}</b>',
                         f'settings <b>{_esc(s.get("config_name") or run.get("config") or "default")}</b>'),
            _bi_tag_html(("실제 사용 중" if deployed else "실제로는 쓰지 않음") + f' (deploy {_esc(deploy)}'
                         + ((" · " + _esc(tone_model)) if tone_model else "") + ")",
                         ("in use" if deployed else "not in use") + f' (deploy {_esc(deploy)}'
                         + ((" · " + _esc(tone_model)) if tone_model else "") + ")",
                         cls="tag" if deployed else "tag warn"),
            _bi_tag_html(f'이 페이지가 쓴 확률 <b>{_esc(rung)}</b> (헤드라인 <b>{_esc(rung)}</b> {_esc(dep_label)})',
                         f'this page uses the <b>{_esc(rung)}</b> probability ({_esc(dep_en)})',
                         cls="tag" if deployed else "tag warn"),
            _bi_tag_html(f'만든 시각 <b>{_esc(run.get("generated_at_utc") or s.get("generated_at") or _now_str())}</b>',
                         f'generated <b>{_esc(run.get("generated_at_utc") or s.get("generated_at") or _now_str())}</b>')]
    for f in flags:
        tags.append(f'<span class="tag bad">{_raw(f)}</span>')
    if warns:
        tags.append(_bi_tag(f"경고 {len(warns)}건", f"{len(warns)} warnings", cls="tag warn"))
    head = ('<header><div class="eyebrow">'
            + bi("market-risk-lab · Phase 2 · 판정 규칙 v1 (상태 3단계)",
                 "market-risk-lab · Phase 2 · decision rule v1 (three states)")
            + "</div><h1>"
            + bi("판정 규칙 v1 을 과거에 돌려 본 결과 — 오늘 확률 ÷ 평소 평균으로 상태를 정하고, 올릴 땐 바로 · 내릴 땐 5거래일 기다립니다",
                 "The v1 decision rule run over the past — the state comes from today's probability divided by the long-run "
                 "average; it steps up at once but waits 5 trading days before stepping down")
            + "</h1>"
            + _bi_note(f'문턱: caution 은 {DECISION_P2["enter_caution"]} 배에서 켜고 {DECISION_P2["exit_caution"]} 배에서 끄며, '
                       f'reduce 는 {DECISION_P2["enter_reduce"]} 배에서 켜고 {DECISION_P2["exit_reduce"]} 배에서 끕니다. '
                       f'내리기 전 최소 유지 {DECISION_P2["dwell"]}거래일, 1년에 바뀌는 횟수는 {DECISION_P2["kpi_max_switches_per_year"]}회까지, '
                       f'최근 252거래일에 {DECISION_P2["churn_alert"]}회를 넘으면 잦은 전환 경보. '
                       "상태 → 신호등 판정 → 주식을 얼마나 들고 갈지(100/50/25%) 로 이어지고 매매비용 5bp 를 붙여 v0 와 같은 잣대로 잽니다.",
                       f'Thresholds: caution turns on at {DECISION_P2["enter_caution"]}x the long-run average and off at '
                       f'{DECISION_P2["exit_caution"]}x; reduce turns on at {DECISION_P2["enter_reduce"]}x and off at '
                       f'{DECISION_P2["exit_reduce"]}x. It must stay put {DECISION_P2["dwell"]} trading days before stepping '
                       f'down (dwell), may change at most {DECISION_P2["kpi_max_switches_per_year"]} times a year, and raises a '
                       f'churn alert above {DECISION_P2["churn_alert"]} changes in any 252 trading days. State becomes a '
                       "traffic-light call and then how much to hold in stocks (100/50/25%), with a 5bp trading cost, scored on "
                       "the same yardstick as v0.", cls="sub")
            + _bi_note_html(
                f'아래 ②~⑤ 의 숫자는 모두 <b>{_esc(rung_ko)} 확률로 돌린 판정 규칙</b> 입니다'
                + (f' — summary_p2.json 의 판정이 실제로 쓰기로 한 단계({_esc(dep_ko)})입니다.' if deployed else
                   f' — 실제로 쓰는 단계가 없어서 생산 모델을 참고로만 싣습니다 '
                   f'(배치된 단이 없어(deploy {_esc(str(deploy))}) 생산 모델을 {_esc(INFO_DISPLAY_LABEL)} 로 싣습니다: 톤·비중 주장이 아닙니다).')
                + (f' 실제로 쓰지 않는 단계({_esc(", ".join(info_names))})는 ⑥ 에 참고로만 둡니다.' if info_names else ""),
                f'Every number in ② to ⑤ comes from the decision rule run on the <b>{_esc(rung_en)}</b> probability'
                + (f' — the step that summary_p2.json says is actually in use ({_esc(dep_en)}).' if deployed else
                   f' — no step is actually in use (deploy {_esc(str(deploy))}), so the production model is shown for '
                   f'information only: this is not a claim about calls or about how much to hold.')
                + (f' The steps that are not in use ({_esc(", ".join(info_names))}) sit in ⑥ for comparison only.'
                   if info_names else ""), cls="sub")
            + f'<div class="tags">{"".join(tags)}</div></header>')
    sec_titles = [("① v0 요약(영구)", "① v0 summary (always shown)"),
                  ("② 판정 규칙 성적 vs v0", "② how the rule did, next to v0"),
                  ("③ 하락 사건", "③ decline episodes"),
                  ("④ 주식 얼마나 들고 갈지 — v1·v0·그냥 보유", "④ how much to hold — v1, v0, buy and hold"),
                  ("⑤ 설정을 바꿔 보면", "⑤ if the settings change"),
                  ("⑥ 쓰지 않는 단계(참고)", "⑥ steps not in use (for reference)"),
                  ("⑦ 그림", "⑦ charts"),
                  ("⑧ 정직 문구·경고", "⑧ honesty notes and warnings")]
    nav = '<nav class="nav">' + "".join(f'<a href="#s{i + 1}">{bi(t[0], t[1])}</a>' for i, t in enumerate(sec_titles)) + "</nav>"

    s1 = ('<section class="panel" id="s1">'
          + _bi_h2("① 예전 규칙 v0 의 성적 (영구 표기)", "① the older v0 rule's record (always shown)")
          + _v0_line_html(v0h, bi_mode=True) + "</section>")

    # ② KPI
    occ = kp.get("occupancy") if isinstance(kp.get("occupancy"), dict) else {}
    dd_state = kp.get("dd5_rate_by_state") if isinstance(kp.get("dd5_rate_by_state"), dict) else {}
    v0_dd = v0h.get("dd5_20_rate_by_tone") if isinstance(v0h.get("dd5_20_rate_by_tone"), dict) else {}
    v0_ref = kp.get("true_alarm_share_v0_ref") if isinstance(kp.get("true_alarm_share_v0_ref"), dict) else {}
    v0_ta = v0h.get("true_alarm_share")
    v0_ta_txt = _pct1(v0_ta) if v0_ta is not None else (f"{_pct1(v0_ref.get('lo'))}~{_pct1(v0_ref.get('hi'))}" if v0_ref else "—")
    warn_dd = [dd_state.get(k) for k in ("caution", "reduce") if not _is_nan(dd_state.get(k))]
    v0_warn_dd = [v0_dd.get(k) for k in ("caution", "reduce") if not _is_nan(v0_dd.get(k))]
    v0_up_dd = [v0_dd.get(k) for k in ("buy", "hold", "neutral") if not _is_nan(v0_dd.get(k))]
    ceiling_ok = kp.get("kpi_ceiling_ok")
    _ceiling = kp.get("kpi_ceiling", DECISION_P2["kpi_max_switches_per_year"])
    _ceil_ko = "충족" if ceiling_ok else ("위반" if ceiling_ok is False else "—")
    _ceil_en = "respected" if ceiling_ok else ("broken" if ceiling_ok is False else "—")
    _bh_cagr = alloc.get("bh_cagr") if alloc.get("bh_cagr") is not None else v0h.get("cagr_bh")
    _bh_dd = alloc.get("bh_max_dd") if alloc.get("bh_max_dd") is not None else v0h.get("maxdd_bh")
    kpis = [
        _bi_kpi("1년에 상태가 바뀐 횟수", "state changes per year",
                _fmt_p2(kp.get("switches_per_year"), "switches_per_year"),
                f"v0 {_fmt_p2(v0h.get('tone_switches_per_year'), 'switches_per_year')}회 · KPI 상한 {_ceiling} {_ceil_ko}",
                f"v0 changed {_fmt_p2(v0h.get('tone_switches_per_year'), 'switches_per_year')} times · "
                f"our ceiling {_ceiling} a year, {_ceil_en}"),
        _bi_kpi("경고가 켜져 있던 날 (caution+reduce)", "days with a warning on (caution+reduce)",
                _pct1(kp.get("warn_share")),
                f"v0 는 {_pct1(v0h.get('warn_share'))}", f"v0: {_pct1(v0h.get('warn_share'))}"),
        _bi_kpi("경고가 켜진 날, 20일 안에 5% 넘게 떨어진 비율",
                "on warning days, how often it fell more than 5% within 20 days",
                (f"{min(warn_dd) * 100:.1f}~{max(warn_dd) * 100:.1f}%" if warn_dd else "—"),
                f"정상 {_pct1(dd_state.get('normal'))} · 평소 비율 (기저율 {_pct1(kp.get('base_rate'))} — "
                f"{_ten_freq(kp.get('base_rate'))}쯤) · v0 는 경고일 "
                f"{(f'{min(v0_warn_dd) * 100:.0f}~{max(v0_warn_dd) * 100:.0f}%' if v0_warn_dd else '—')} vs 평온한 날 "
                f"{(f'{min(v0_up_dd) * 100:.0f}~{max(v0_up_dd) * 100:.0f}%' if v0_up_dd else '—')}",
                f"in the normal state {_pct1(dd_state.get('normal'))} · any day at all {_pct1(kp.get('base_rate'))} — "
                f"{_ten_freq_en(kp.get('base_rate'))} · "
                f"v0 warning days {(f'{min(v0_warn_dd) * 100:.0f}-{max(v0_warn_dd) * 100:.0f}%' if v0_warn_dd else '—')} "
                f"vs calm days {(f'{min(v0_up_dd) * 100:.0f}-{max(v0_up_dd) * 100:.0f}%' if v0_up_dd else '—')}"),
        _bi_kpi("경고가 진짜였던 비율 (경고가 켜진 첫날 기준)", "share of warnings that were real (counted on the day it turned on)",
                _pct1(kp.get("true_alarm_share")),
                f"아무 날이나 골랐을 때 (기저율 {_pct1(kp.get('true_alarm_share_baseline'))} — "
                f"{_ten_freq(kp.get('true_alarm_share_baseline'))}쯤) · v0 {v0_ta_txt}",
                f"picking any day would give {_pct1(kp.get('true_alarm_share_baseline'))} — "
                f"{_ten_freq_en(kp.get('true_alarm_share_baseline'))} · v0 {v0_ta_txt}"),
        _bi_kpi("5거래일 안에 상태가 바뀐 최대 횟수", "most state changes within any 5 trading days",
                _fmt(kp.get("max_changes_any_5_sessions"), "n"),
                f"규칙상 넘을 수 없는 수 {kp.get('structural_bound') if kp.get('structural_bound') is not None else '—'} "
                f"{'충족' if kp.get('structural_bound_ok') else ''}",
                f"the rule allows at most {kp.get('structural_bound') if kp.get('structural_bound') is not None else '—'} "
                f"{'and that held' if kp.get('structural_bound_ok') else ''}"),
        _bi_kpi("잦은 전환 경보가 켜진 날", "days with the churn alert on",
                _fmt(kp.get("churn_alert_sessions"), "n"),
                f"최근 252거래일 최대 변경 {_fmt(kp.get('max_churn_252'), 'n')}회",
                f"most changes in any 252 trading days: {_fmt(kp.get('max_churn_252'), 'n')}"),
        _bi_kpi("1년 평균 수익률", "return per year", _pct1(alloc.get("cagr")),
                f"v0 {_pct1(v0h.get('cagr_strategy'))} · 그냥 들고 있으면 {_pct1(_bh_cagr)}",
                f"v0 {_pct1(v0h.get('cagr_strategy'))} · buy and hold {_pct1(_bh_cagr)}"),
        _bi_kpi("고점 대비 가장 큰 하락", "biggest drop from the peak", _pct1(alloc.get("max_dd")),
                f"v0 {_pct1(v0h.get('maxdd_strategy'))} · 그냥 들고 있으면 {_pct1(_bh_dd)}",
                f"v0 {_pct1(v0h.get('maxdd_strategy'))} · buy and hold {_pct1(_bh_dd)}"),
    ]
    side = {"v1 결정층": {"switches_per_year": kp.get("switches_per_year"), "warn_share": kp.get("warn_share"),
                        "true_alarm_share": kp.get("true_alarm_share"), "true_alarm_share_baseline": kp.get("true_alarm_share_baseline"),
                        "median_warn_run": kp.get("median_warn_run"), "n_warn_runs": kp.get("n_warn_runs"),
                        "cagr": alloc.get("cagr"), "max_dd": alloc.get("max_dd"), "worst_month": alloc.get("worst_month")},
            "v0 톤": {"switches_per_year": v0h.get("tone_switches_per_year"), "warn_share": v0h.get("warn_share"),
                     "true_alarm_share": v0h.get("true_alarm_share"), "true_alarm_share_baseline": v0h.get("true_alarm_share_baseline"),
                     "median_warn_run": None, "n_warn_runs": None,
                     "cagr": v0h.get("cagr_strategy"), "max_dd": v0h.get("maxdd_strategy"), "worst_month": v0h.get("worst_month_strategy")}}
    n_by = kp.get("n_by_state") if isinstance(kp.get("n_by_state"), dict) else {}
    run_by = kp.get("median_run_by_state") if isinstance(kp.get("median_run_by_state"), dict) else {}
    st_rows = [{"state": st, "occupancy": occ.get(st), "n": n_by.get(st), "dd5_rate": dd_state.get(st), "median_run": run_by.get(st)} for st in P2_STATES]
    kp_rest = {k: v for k, v in kp.items() if _scalar_ok(v) and k not in ("start", "end")}
    s2 = ('<section class="panel" id="s2">'
          + _bi_h2(f"② 판정 규칙 성적 — {rung} 확률로 돌린 결과 (② 결정층 KPI — {rung} {dep_label}) · v0 와 나란히",
                   f"② how the rule did, run on the {rung} probability ({dep_en}), side by side with v0")
          + _bi_note_html(f'이 절의 성적·배분·하락 사건·표시는 모두 <b>{_esc(rung)}</b> 확률로 돌린 판정 규칙입니다 '
                          f'({_esc(dep_ko)}; summary_v1.json 의 같은 이름 칸과 같은 뜻입니다).',
                          f'The scores, the allocation, the episodes and the flags in this section all come from the decision '
                          f'rule run on the <b>{_esc(rung)}</b> probability ({_esc(dep_en)}; the same thing as the keys of the '
                          f'same name in summary_v1.json).')
          + f'<div class="grid">{"".join(kpis)}</div>'
          + _bi_note(f"{P2_HONESTY_ITEMS[6]} {P2_HONESTY_ITEMS[7]}",
                     f"{P2_HONESTY_ITEMS_EN[6]} {P2_HONESTY_ITEMS_EN[7]}")
          + _bi_h3("v1 과 v0 를 나란히", "v1 next to v0") + _bi_kv(side, cols=_ALLOC_COLS_BI)
          + _bi_h3("상태마다 — 머문 날 비율 · 20일 안에 5% 하락한 비율 · 한 번 켜지면 며칠",
                   "for each state — share of days, how often a 5% fall followed within 20 days, and how long a run lasts")
          + _bi_table(st_rows, ["state", "occupancy", "n", "dd5_rate", "median_run"])
          + _bi_h3("성적 전체", "every measure") + _bi_kv(kp_rest, ("성적 자료 없음", "no measures"))
          + (_bi_h3("이 실행에 쓴 설정", "the settings used for this run") + _bi_kv(cfg) if cfg else "") + "</section>")

    # ③ 에피소드
    ep = s.get("episodes")
    if isinstance(ep, dict):
        ep_tab = _to_records(ep.get("table"))
        ep_sum = ep.get("summary") if isinstance(ep.get("summary"), dict) else {k: v for k, v in ep.items() if _scalar_ok(v)}
    else:
        ep_tab = _to_records(ep)
        ep_sum = s.get("episode_summary") if isinstance(s.get("episode_summary"), dict) else {}
    order3 = ["peak_date", "trough_date", "depth", "days_to_trough", "recovery_date", "days_to_recover", "warn_date", "lead_days",
              "lead_capped", "held_to_trough", "missed", "tone_at_peak", "tone_at_trough"]
    s3 = ('<section class="panel" id="s3">'
          + _bi_h2(f"③ 큰 하락이 있었던 때, 경고가 먼저 켜졌나 — {rung} ({dep_ko}), v0 와 같은 규칙",
                   f"③ when a big fall came, did the warning arrive first — {rung} ({dep_en}), same rule as v0")
          + _bi_note("'며칠 먼저 켜졌나'는 고점 앞 20거래일까지만 찾습니다. 그 한도에 걸리는 경우가 있으므로, "
                     "날짜를 무작위로 밀어서 만든 기준선과 반드시 나란히 읽어야 합니다(실험 #1).",
                     "We look at most 20 trading days before the peak for a warning, so the number can hit that limit. "
                     "Always read it next to the baseline built by shifting the dates at random (experiment #1).")
          + (_bi_h3("요약", "summary") + _bi_kv(ep_sum) if ep_sum else "")
          + _bi_table(ep_tab, order3, ("하락 사건 표 없음", "no episode table")) + "</section>")

    # ④ 배분
    s4 = ('<section class="panel" id="s4">'
          + _bi_h2(f"④ 주식을 얼마나 들고 갔다면 — v1({rung}, {dep_ko}) · v0 · 그냥 들고 있기",
                   f"④ how much to hold — v1 ({rung}, {dep_en}), v0, and buy and hold")
          + _bi_note("상태를 신호등 판정으로 옮기고(normal→hold, caution→caution, reduce→reduce) 주식을 100/50/25% 들고 갑니다. "
                     "오늘 종가로 정한 판정은 내일 수익에 적용하고, 바뀔 때마다 매매비용 5bp 를 뺍니다. "
                     "v0 열은 summary_v0_completed.json 의 값 그대로입니다.",
                     "The state becomes a traffic-light call (normal to hold, caution to caution, reduce to reduce) and you "
                     "hold 100/50/25% in stocks. A call made at today's close applies to tomorrow's return, and every change "
                     "costs 5bp. The v0 column is taken as-is from summary_v0_completed.json.")
          + _alloc_matrix(alloc, v0_alloc)
          + _bi_img(charts.pop("cumret_v1", None),
                    "돈이 어떻게 불었는지(로그 눈금): 그냥 들고 있기 · v1 판정 규칙 · v0 신호등 판정.",
                    "How the money grew (log scale): buy and hold, the v1 decision rule, and the v0 traffic-light calls.",
                    "누적 수익 그림", "Cumulative return v1 vs v0 vs hold") + "</section>")

    # ⑤ 민감도
    sens = s.get("sensitivities")
    sens_recs = []
    if isinstance(sens, dict):
        for name, v in sens.items():
            if not isinstance(v, dict):
                continue
            row = {"name": _Html(_raw(name))}          # 설정 이름은 기계 이름 그대로 인용한다(번역 대상 아님)
            src = v.get("kpis") if isinstance(v.get("kpis"), dict) else v
            for k in ("switches_per_year", "warn_share", "true_alarm_share", "max_changes_any_5_sessions", "churn_alert_sessions", "kpi_ceiling_ok"):
                row[k] = src.get(k)
            occ_ = src.get("occupancy") if isinstance(src.get("occupancy"), dict) else {}
            row["occupancy_caution"] = occ_.get("caution")
            row["occupancy_reduce"] = occ_.get("reduce")
            al = src.get("allocation") if isinstance(src.get("allocation"), dict) else (v.get("allocation") if isinstance(v.get("allocation"), dict) else {})
            row["cagr"], row["max_dd"] = al.get("cagr"), al.get("max_dd")
            sens_recs.append(row)
    else:
        sens_recs = _to_records(sens)
    s5 = ('<section class="panel" id="s5">'
          + _bi_h2("⑤ 문턱을 바꿔 보면 (wide · symmetric_dwell · no_dwell) — 보여 주기만 하고 고르는 데는 쓰지 않습니다",
                   "⑤ what happens with other thresholds (wide · symmetric_dwell · no_dwell) — shown, never used to choose")
          + _bi_table(sens_recs, ["name", "switches_per_year", "warn_share", "occupancy_caution", "occupancy_reduce", "true_alarm_share",
                                  "max_changes_any_5_sessions", "churn_alert_sessions", "kpi_ceiling_ok", "cagr", "max_dd"],
                      ("바꿔 본 결과 없음 (sensitivities 없음)", "no alternative settings were run"))
          + "</section>")

    # ⑥ 비배포 단(정보) — 배포 단과 나란히, 톤·비중 주장이 아님
    info_rows = _info_layer_rows(s)
    s6 = ('<section class="panel" id="s6">'
          + _bi_h2(f"⑥ 실제로 쓰지 않는 단계 — 참고 표시 ({INFO_DISPLAY_LABEL})",
                   "⑥ the steps that are not in use — shown for reference only")
          + _bi_note_html(
              "같은 판정 규칙을 다른 단계의 확률로 돌려 본 결과입니다. "
              + (f'실제로 쓰는 단계는 <b>{_esc(rung)}</b> 하나뿐이고, ' if deployed
                 else f'실제로 쓰는 단계가 없어서 맨 윗줄 <b>{_esc(rung)}</b> 도 참고용입니다 '
                      f'(배포된 단이 없어(deploy {_esc(str(deploy))}) 맨 윗줄 <b>{_esc(rung)}</b> 도 '
                      f'{_esc(INFO_DISPLAY_LABEL)} 이며), ')
              + "나머지 줄은 견주어 보라고 둔 것입니다 — 신호등 판정이나 주식을 얼마나 들지에 대한 제안이 아니고, "
                "②~⑤ 의 숫자에도 들어가지 않습니다. (어느 단계를 쓸지는 summary_p2.json 의 판정이 정하고, 이 페이지는 그 판정을 바꾸지 않습니다.)",
              "The same decision rule run on the probability of each other step. "
              + (f'Only <b>{_esc(rung)}</b> is actually in use, ' if deployed
                 else f'No step is actually in use, so even the top row <b>{_esc(rung)}</b> is there for reference only, ')
              + "and the other rows are there to compare against — they are not a suggestion about calls or about how much to "
                "hold, and they do not enter the numbers in ② to ⑤. (Which step is used is decided by the verdict in "
                "summary_p2.json; this page does not change it.)")
          + _bi_table(info_rows, ["name", "role", "switches_per_year", "warn_share", "occupancy_caution", "occupancy_reduce",
                                  "true_alarm_share", "true_alarm_share_baseline", "detection_rate", "median_lead_days",
                                  "max_changes_any_5_sessions", "churn_alert_sessions", "cagr", "max_dd"],
                      ("쓰지 않는 단계 자료 없음", "no data for the steps not in use"))
          + "</section>")

    # ⑦ 차트
    s7 = ('<section class="panel" id="s7">' + _bi_h2("⑦ 그림", "⑦ charts")
          + _bi_img(charts.pop("prob_path", None),
                    f"확률이 지나온 길 — 굵은 선이 {'실제로 쓰는' if deployed else '참고용'} {rung} 확률, 회색 점선이 평소 평균, "
                    "색 띠가 그날의 상태입니다."
                    + (" 얇은 선은 실제로 쓰지 않는 M3 입니다." if rung != "M3" else ""),
                    f"The path of the probability — the thick line is the {rung} probability "
                    f"({'in use' if deployed else 'for reference only'}), the grey dotted line is the long-run average, "
                    "and the coloured bands are the state on that day."
                    + (" The thin line is M3, which is not in use." if rung != "M3" else ""),
                    "확률 경로 그림", "OOS probability path with decision states")
          + _bi_img(charts.pop("state_bands", None), "SPY 종가(로그 눈금)와 그날의 상태 띠.",
                    "The SPY close (log scale) with the state bands.", "상태 띠 그림", "SPY with state bands"))
    for k, png in list(charts.items()):
        s7 += _bi_img(png, f"추가 차트: {k}", f"extra chart: {k}", f"추가 차트 {k}", f"extra chart {k}")
    s7 += "</section>"

    # ⑧ 정직·경고
    s8 = ('<section class="panel" id="s8">'
          + _bi_h2("⑧ 정직 문구 · 눈에 띈 것 · 경고", "⑧ honesty notes, flags and warnings")
          + (_bi_h3("눈에 띈 것", "flags") + _bi_warn_list(flags, "눈에 띈 것 없음", "nothing flagged") if flags else "")
          + _bi_h3("경고", "warnings") + _bi_warn_list(warns)
          + _bi_h3("정직 문구와 남는 위험 (§16)", "honesty notes and the risks that remain (§16)")
          + '<ol class="method">'
          + "".join(f"<li>{bi(t, P2_HONESTY_ITEMS_EN[i] if i < len(P2_HONESTY_ITEMS_EN) else t)}</li>"
                    for i, t in enumerate(P2_HONESTY_ITEMS)) + "</ol>"
          + _bi_note(P2_FOOTNOTE, P2_FOOTNOTE_EN)
          + '<div class="foot">'
          + bi(f'market-risk-lab · 판정 규칙 v1 · 만든 시각 {run.get("generated_at_utc") or _now_str()}',
               f'market-risk-lab · decision rule v1 · generated {run.get("generated_at_utc") or _now_str()}')
          + f' · <a href="index.html">{bi("오늘 판정 보기", "the call for today")}</a>'
          + f' · <a href="calibration_p2.html">{bi("확률이 맞았나 보기", "how good the probabilities were")}</a>'
          + f' · <a href="backtest_v0.html">{bi("예전 규칙 v0 시험", "the older v0 backtest")}</a></div></section>')
    _write_html(out_html, "Decision rule v1 backtest — market-risk-lab", head + nav + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8)


# ------------------------------------------------------------------
# 차트 (matplotlib Agg → PNG bytes)
# ------------------------------------------------------------------
def _p2_frame(oos) -> pd.DataFrame | None:
    if not isinstance(oos, pd.DataFrame) or len(oos) == 0:
        return None
    df = oos.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if "p_m3" not in df.columns:
        for alt in ("p", "prob", "prob_dd5_20"):
            if alt in df.columns:
                df["p_m3"] = pd.to_numeric(df[alt], errors="coerce")
                break
    if "state" not in df.columns and "p2_state" in df.columns:
        df["state"] = df["p2_state"]
    return df


def charts_p2(oos, blocks, summary_p2, spy_close) -> dict[str, bytes]:
    """차트 → PNG bytes: reliability(고정 구간·Wilson), block_skill(4 벤치마크 막대), coef_path, ladder_ci, era_auc, prob_path(확률 경로+상태),
    state_bands(상태 밴드+SPY), cumret_v1(누적수익 v1 vs 보유 vs v0). 자료가 없으면 그 차트만 경고 후 생략(예외로 죽지 않음).
    oos: calib_p2_walkforward(y, clim, p_m3 …) 또는 backtest_v1(p, clim, state …) — 있는 열만 쓴다. summary_p2 의 표(blocks24, ladder,
    params_by_refit, era_auc, reliability)가 있으면 그것을 우선한다. v0 톤은 oos['v0_tone'|'tone_v0'] 또는 summary_p2['v0_tone'] (Series/dict)."""
    out: dict[str, bytes] = {}
    s = summary_p2 if isinstance(summary_p2, dict) else {}
    df = _p2_frame(oos)
    spy = _series_close(spy_close)
    # 배포 단(있으면). summary_p2['deployed_rung'] 이 없으면 예전대로 M3 를 그린다.
    dep_rung = str(s.get("deployed_rung") or "M3").upper()
    dep_col = f"p_{dep_rung.lower()}"
    if df is not None and dep_col not in df.columns and dep_col != "p_m3":
        warnings.warn(f"charts_p2: 배포 단 열 {dep_col} 이 oos 에 없어 p_m3 로 그린다(범례 라벨 확인)")
        dep_rung, dep_col = "M3", "p_m3"

    # 1) 신뢰도
    try:
        # 미리 계산된 표는 배포 단의 것만 쓴다 — M3 표를 'M1' 범례 아래 그리면 더 나쁘다
        rel = _to_records(s.get("reliability" if dep_rung == "M3" else f"reliability_{dep_rung.lower()}"))
        if not rel and df is not None and {dep_col, "y"} <= set(df.columns):
            from mrl.calibrate import reliability_table
            sub = df[[dep_col, "y"]].astype(float).dropna()
            if len(sub):
                rel = reliability_table(sub[dep_col], sub["y"]).to_dict("records")
        rel = [r for r in rel if not _is_nan(r.get("obs")) and not _is_nan(r.get("mean_p"))]
        if rel:
            fig, ax = _new_fig(5.6, 5.0)
            xs = np.array([_fnum(r.get("mean_p")) for r in rel])
            ys = np.array([_fnum(r.get("obs")) for r in rel])
            lo = np.array([_fnum(r.get("wilson_lo")) for r in rel])
            hi = np.array([_fnum(r.get("wilson_hi")) for r in rel])
            err = np.vstack([np.where(np.isnan(lo), 0, ys - lo), np.where(np.isnan(hi), 0, hi - ys)])
            ax.plot([0, 1], [0, 1], color=PALETTE["mut"], lw=0.9, ls="--", label="perfect")
            ax.errorbar(xs, ys, yerr=err, fmt="o", color="#60a5fa", ecolor="#60a5fa", capsize=3, ms=5,
                        label=f"{dep_rung} (Wilson, n/20)")
            for r, x, y_ in zip(rel, xs, ys):
                ne = _fnum(r.get("n_eff"))
                if not math.isnan(ne):
                    ax.annotate(f"{ne:.0f}", (x, y_), textcoords="offset points", xytext=(5, 4), fontsize=7, color=PALETTE["mut"])
            top = np.concatenate([xs, ys, hi[~np.isnan(hi)]])
            lim = min(1.0, max(0.5, float(np.nanmax(top)) + 0.05)) if len(top) else 1.0
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_xlabel("mean forecast p (fixed bins)")
            ax.set_ylabel("observed frequency")
            ax.set_title("Reliability diagram (labels = n_eff)")
            _legend(ax, loc="upper left")
            out["reliability"] = _png(fig)
        else:
            warnings.warn("reliability: 자료 없음(oos 에 p_m3/y 없음, summary 에 reliability 없음) → 생략")
    except Exception as e:
        warnings.warn(f"reliability 차트 실패: {type(e).__name__}: {e}")

    # 2) 블록 skill 막대
    try:
        bs = _to_records((s.get("blocks24_by_rung") or {}).get(dep_rung)
                         or (s.get("blocks24") if dep_rung == "M3" else None))
        if not bs and df is not None and blocks and {dep_col, "y", "clim", "p_vix", "p_vix_bgk"} <= set(df.columns):
            from mrl.calibrate import block_scores
            bs = block_scores(df, blocks, dep_col).to_dict("records")
        bs = [r for r in bs if str(r.get("block")) != "all" and not _is_nan(r.get("bss_clim"))]
        if bs:
            fig, ax = _new_fig(9.0, 4.0)
            names = [f"#{r.get('block')}\n{str(r.get('start', ''))[:4]}-{str(r.get('end', ''))[2:4]}" for r in bs]
            series = [("bss_clim", "vs climatology", "#22c55e"), ("bss_vix", "vs VIX (B1)", "#60a5fa"),
                      ("bss_vix_bgk", "vs BGK", "#a78bfa"), ("bss_m1", "vs M1", "#f472b6")]
            x = np.arange(len(bs))
            w = 0.2
            for i, (k, lb, c) in enumerate(series):
                ax.bar(x + (i - 1.5) * w, [_fnum(r.get(k)) for r in bs], width=w, color=c, label=lb)
            ax.axhline(0, color=PALETTE["tx"], lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(names, fontsize=7)
            ax.set_ylabel("Brier skill")
            ax.set_title(f"Block Brier skill of {dep_rung} (24-month blocks; last = 20 months)")
            _legend(ax, loc="lower left", ncol=4)
            out["block_skill"] = _png(fig)
        else:
            warnings.warn("block_skill: 블록 자료 없음 → 생략")
    except Exception as e:
        warnings.warn(f"block_skill 차트 실패: {type(e).__name__}: {e}")

    # 3) 계수 경로
    try:
        pr = [r for r in _param_records(s.get("params_by_refit"), "M3") if r.get("refit_date")]
        if pr:
            fig, ax = _new_fig(9.0, 3.6)
            xs = [pd.Timestamp(r["refit_date"]) for r in pr]
            for k, lb, c in (("intercept", "b0 intercept", PALETTE["mut"]), ("b_x_vix", "b1 x_vix", FEATURE_COLORS["x_vix"]),
                             ("b_x_har", "b2 x_har", FEATURE_COLORS["x_har"]), ("b_x_ma", "b3 x_ma", FEATURE_COLORS["x_ma"])):
                ys = [_fnum(r.get(k)) for r in pr]
                if any(not math.isnan(v) for v in ys):
                    ax.plot(xs, ys, marker="o", ms=3, lw=1.2, color=c, label=lb)
            ax.axhline(0, color=PALETTE["line"], lw=0.8)
            ax.set_title("Coefficient path by annual refit (M3, exactly 4 fitted parameters)")
            _legend(ax, loc="best", ncol=4)
            out["coef_path"] = _png(fig)
        else:
            warnings.warn("coef_path: params_by_refit 없음 → 생략")
    except Exception as e:
        warnings.warn(f"coef_path 차트 실패: {type(e).__name__}: {e}")

    # 4) 사다리 loss-diff CI
    try:
        lad = [r for r in _ladder_records(s.get("ladder")) if str(r.get("block")) == "all" and not _is_nan(r.get("diff_mean"))]
        if lad:
            fig, ax = _new_fig(7.5, 0.55 * len(lad) + 1.6)
            ys = np.arange(len(lad))[::-1]
            for y_, r in zip(ys, lad):
                m, lo, hi = (_fnum(r.get(k)) * 1e4 for k in ("diff_mean", "diff_lo", "diff_hi"))
                c = "#22c55e" if (not math.isnan(lo) and lo > 0) else ("#ef4444" if (not math.isnan(hi) and hi < 0) else "#eab308")
                if not (math.isnan(lo) or math.isnan(hi)):
                    ax.plot([lo, hi], [y_, y_], color=c, lw=2.2)
                ax.plot([m], [y_], marker="o", color=c, ms=5)
            ax.axvline(0, color=PALETTE["tx"], lw=0.8)
            ax.set_yticks(ys)
            ax.set_yticklabels([str(r.get("step")) for r in lad], fontsize=8)
            ax.set_xlabel("mean loss difference x 1e-4 (>0: higher rung better), 95% block bootstrap")
            ax.set_title("Ladder: paired Brier loss differences")
            out["ladder_ci"] = _png(fig)
        else:
            warnings.warn("ladder_ci: ladder 'all' 행 없음 → 생략")
    except Exception as e:
        warnings.warn(f"ladder_ci 차트 실패: {type(e).__name__}: {e}")

    # 5) 시대별 AUC
    try:
        era = [r for r in _to_records(s.get("era_auc")) if any(not _is_nan(r.get(k)) for k in ("auc_x_vix", "auc_p_m3", "auc_p_m1"))]
        if era:
            fig, ax = _new_fig(9.0, 3.8)
            x = np.arange(len(era))
            series = [("auc_x_vix", "x_vix", FEATURE_COLORS["x_vix"]), ("auc_x_har", "x_har", FEATURE_COLORS["x_har"]),
                      ("auc_neg_x_ma", "-x_ma", FEATURE_COLORS["x_ma"]), ("auc_p_m1", "p_m1", "#a78bfa"), ("auc_p_m3", "p_m3", "#f59e0b")]
            w = 0.16
            for i, (k, lb, c) in enumerate(series):
                ax.bar(x + (i - 2) * w, [_fnum(r.get(k)) for r in era], width=w, color=c, label=lb)
            ax.axhline(0.5, color=PALETTE["tx"], lw=0.8, ls="--")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{str(r.get('start', ''))[:4]}-{str(r.get('end', ''))[:4]}" for r in era], fontsize=8)
            ax.set_ylim(0.4, 0.85)
            ax.set_ylabel("AUC")
            ax.set_title("AUC by era (VIX discrimination decays)")
            _legend(ax, loc="upper right", ncol=5)
            out["era_auc"] = _png(fig)
        else:
            warnings.warn("era_auc: 자료 없음 → 생략")
    except Exception as e:
        warnings.warn(f"era_auc 차트 실패: {type(e).__name__}: {e}")

    # 6) 확률 경로 + 상태 — 굵은 선은 배포 단(summary_p2['deployed_rung'], 없으면 M3), 배포되지 않은 M3 는 얇은 정보선
    try:
        if df is not None and dep_col in df.columns:
            p = pd.to_numeric(df[dep_col], errors="coerce")
            fig, ax = _new_fig(9.0, 3.8)
            if "state" in df.columns:
                for a, b, st in _tone_runs(df["state"].fillna("normal").astype(str)):
                    if st in ("caution", "reduce"):
                        ax.axvspan(a, b + pd.Timedelta(days=1), color=P2_STATE_COLORS.get(st, PALETTE["mut"]), alpha=0.25, lw=0)
            if dep_col != "p_m3" and "p_m3" in df.columns:      # 배포되지 않은 생산 모델은 정보로만
                ax.plot(df.index, pd.to_numeric(df["p_m3"], errors="coerce").values, color="#f59e0b", lw=0.7, alpha=0.75,
                        label="p (M3, information only)")
            ax.plot(p.index, p.values, color="#60a5fa", lw=0.9,
                    label=f"p ({dep_rung}{', deployed' if s.get('rung_deployed') else ''})")   # 배포 여부는 명시 불리언으로
            if "clim" in df.columns:
                ax.plot(df.index, pd.to_numeric(df["clim"], errors="coerce").values, color=PALETTE["mut"], lw=1.0, ls="--", label="climatology")
            if "y" in df.columns:
                yv = pd.to_numeric(df["y"], errors="coerce")
                hits = yv.index[yv == 1]
                if len(hits):
                    ax.scatter(hits, np.full(len(hits), 0.01), s=2, color="#ef4444", label="y_dd5_20 = 1", zorder=3)
            ax.set_ylim(0, 1)
            ax.set_ylabel("P(-5% within 20 sessions)")
            ax.set_title("OOS probability path with decision states (yellow=caution, red=reduce)")
            _legend(ax, loc="upper left")
            out["prob_path"] = _png(fig)
        else:
            warnings.warn("prob_path: oos 에 p_m3/p 열 없음 → 생략")
    except Exception as e:
        warnings.warn(f"prob_path 차트 실패: {type(e).__name__}: {e}")

    # 7) 상태 밴드 + SPY, 8) 누적수익 v1 vs 보유 vs v0
    if df is not None and "state" in df.columns and spy is not None:
        st = df["state"].fillna("normal").astype(str)
        try:
            sp = spy[(spy.index >= st.index[0]) & (spy.index <= st.index[-1])]
            if len(sp) >= 3:
                fig, ax = _new_fig(9.0, 4.0)
                for a, b, s_ in _tone_runs(st):
                    if s_ in ("caution", "reduce"):
                        ax.axvspan(a, b + pd.Timedelta(days=1), color=P2_STATE_COLORS.get(s_, PALETTE["mut"]), alpha=0.28, lw=0)
                ax.plot(sp.index, sp.values, color=PALETTE["tx"], lw=1.0)
                ax.set_yscale("log")
                _plain_log_axis(ax)
                from matplotlib.patches import Patch
                ax.legend(handles=[Patch(color=P2_STATE_COLORS[k], alpha=0.6, label=k) for k in ("caution", "reduce")], loc="upper left",
                          fontsize=8, facecolor=PALETTE["card"], edgecolor=PALETTE["line"], labelcolor=PALETTE["tx"])
                ax.set_title("SPY close (log) with v1 decision states")
                out["state_bands"] = _png(fig)
            else:
                warnings.warn("state_bands: SPY 구간 자료 부족 → 생략")
        except Exception as e:
            warnings.warn(f"state_bands 차트 실패: {type(e).__name__}: {e}")
        try:
            tone = st.map(STATE_TO_TONE).fillna("hold")
            sp = spy.reindex(tone.index).dropna()
            if len(sp) >= 3:
                cost = 5 / 1e4
                ret = sp.pct_change().fillna(0.0)

                def _equity(t: pd.Series):
                    ex = t.reindex(sp.index).map(TONE_EXPOSURE).fillna(1.0).astype(float)
                    pos = ex.shift(1).fillna(ex.iloc[0])
                    turn = ex.diff().abs().fillna(0.0).shift(1).fillna(0.0)
                    return (1 + pos * ret - turn * cost).cumprod()

                fig, ax = _new_fig()
                ax.plot(sp.index, (1 + ret).cumprod().values, color=PALETTE["mut"], lw=1.2, label="SPY buy & hold")
                ax.plot(sp.index, _equity(tone).values, color="#22c55e", lw=1.4, label="v1 decision layer (5bp/switch)")
                v0t = None
                for c in ("v0_tone", "tone_v0"):
                    if c in df.columns:
                        v0t = df[c]
                        break
                if v0t is None and s.get("v0_tone") is not None:
                    v0t = pd.Series(s["v0_tone"])
                    v0t.index = pd.DatetimeIndex(pd.to_datetime(v0t.index)).normalize()
                if v0t is not None:
                    v0t = v0t.dropna().astype(str)
                    common = v0t.index.intersection(sp.index)
                    if len(common) >= 3:
                        ax.plot(common, _equity(v0t.reindex(common)).reindex(common).values, color="#eab308", lw=1.1, label="v0 tone allocation")
                ax.set_yscale("log")
                _plain_log_axis(ax)
                ax.set_title("Cumulative growth of 1.0 (log): hold vs v1 vs v0")
                _legend(ax, loc="upper left")
                out["cumret_v1"] = _png(fig)
            else:
                warnings.warn("cumret_v1: 상태와 SPY 의 공통 구간 3일 미만 → 생략")
        except Exception as e:
            warnings.warn(f"cumret_v1 차트 실패: {type(e).__name__}: {e}")
    elif df is not None and "state" in df.columns:
        warnings.warn("spy_close 없음 → state_bands/cumret_v1 생략")
    return out
