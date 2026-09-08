# -*- coding: utf-8 -*-
"""리포트 렌더링 — 자체 완결 한국어 HTML (다크 팔레트, 시스템 폰트, 모바일 대응).

ARCHITECTURE.md 계약:
    render_backtest_report(summary, out_html, charts)   # 섹션 ①~⑨, 차트는 base64 PNG 인라인
    render_index(today, ledger_summary, out_html)       # 오늘 판정(두 변형)·마지막 갱신·백테스트 링크·장부 요약·정직 문구
    charts_for(summary, replay, spy_close, targets)     # matplotlib(Agg) → {이름: PNG bytes}

ARCHITECTURE_PHASE2.md §11 (확장):
    p2_card(today_p2)                                          # index.html 의 Phase 2 카드 조각 (v0 판정 블록 아래, v0 는 그대로)
    render_calibration_report(summary_p2, out_html, charts)    # docs/calibration_p2.html ①~⑨ (v0 줄 영구, 사다리, 블록·§6 판정, 신뢰도, 계수, 시대 AUC, 소거, HAR, 정직)
    render_backtest_v1(summary_v1, summary_v0, out_html, charts)   # docs/backtest_v1.html (v0 줄 먼저, 결정층 KPI vs v0, 배분 v1·v0·보유, 민감도, 차트)
    charts_p2(oos, blocks, summary_p2, spy_close)              # 신뢰도·블록 skill·계수 경로·사다리 CI·시대 AUC·확률 경로·상태 밴드·누적수익 → PNG bytes
    render_index 는 today["p2"] 가 있으면 p2_card 를 v0 판정 아래에 끼워 넣고 calibration_p2/backtest_v1 링크를 단다.

설계 원칙
* summary 는 evaluate.summarize_v0 가 만드는 dict. 이 모듈은 어떤 키가 있든 있는 것만 렌더링한다
  (.get + 폴백). 알 수 없는 열 이름은 원문 그대로 표 머리글로 쓴다 — 숨기지 않는다.
* 차트는 열이 없으면 warnings.warn 하고 그 차트만 건너뛴다(반환 dict 에서 빠짐). 예외로 죽지 않는다.
* 차트 안 글자는 ASCII(톤 이름·날짜)만 사용 — CI 환경에 한글 폰트가 없어도 깨지지 않게. 한글 설명은 HTML 캡션에.
* 외부 폰트·스크립트 없음. 표는 overflow-x:auto 컨테이너 안.
"""
from __future__ import annotations

import base64
import html as _h
import io
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from mrl.config import (TONES, TONE_EXPOSURE, V0_SIGNALS, SITE_URL, BACKTEST_START, DECISION_P2, P2, P2_STATES, STATE_TO_TONE,
                        HOLDOUT_START, RESULTS_DIR)

# ------------------------------------------------------------------
# 팔레트·라벨 (reference 대시보드와 동일)
# ------------------------------------------------------------------
PALETTE = {"bg": "#0b1220", "card": "#111a2b", "line": "#1e2a40", "tx": "#e6eaf2", "mut": "#8a94a8", "dim": "#5c6a82"}
TONE_COLORS = {"buy": "#22c55e", "hold": "#2dd4bf", "neutral": "#8a94a8", "caution": "#eab308", "reduce": "#ef4444"}
TONE_KO = {"buy": "매수", "hold": "보유", "neutral": "관망", "caution": "주의", "reduce": "축소"}
STATE_COLORS = {"GREEN": "#22c55e", "R2G": "#2dd4bf", "AMBER": "#eab308", "G2R": "#fb923c", "RED": "#ef4444"}
STATE_LABEL = {"GREEN": "GREEN", "R2G": "RED→GREEN", "AMBER": "AMBER", "G2R": "GREEN→RED", "RED": "RED"}
SIGNAL_KO = {"fang": "P1 FANG 상대강도", "macd": "P2 V-바닥/MACD", "fg": "P3 CNN 공포탐욕", "vix": "P4 VIX 정점",
             "ma": "P5 SPY vs 180일선", "eod": "P6 장마감 매수", "lead": "P7 주도주 후보", "btc": "P9 BTC 선행"}
VARIANT_KO = {"faithful": "faithful (라이브와 동일, 부분 봉 포함)", "completed": "completed (완성 봉만)"}

# VALIDATION.md §0 — 원문 그대로 (수정 금지)
HONESTY_TITLE = "0. 정직한 전제 (측정값, SPY 1993-01-29 ~ 2026-09-04, 8,458거래일)"
HONESTY_BULLETS = [
    "상승 비율: 일 54.1% · 20거래일 창 65.4%(겹치지 않는 창 422개 중 64.0%) · 60거래일 창 72.0%(140개 중 70.0%). 2013년 이후는 더 강세: 55.3% / 68~69% / 77~79%.",
    "방향(오를지 내릴지)은 사실상 예측 불가: 학계 최고 모형의 월간 out-of-sample R²는 0.5~1%이며 이는 항상-상승 대비 적중률 +0.3%p 수준. 5%p 우위를 α=5%, 검정력 80%로 확인하려면 20일 창 717개(≈57년)가 필요.",
    "위험(변동성·낙폭)은 예측 가능: VIX 수준만으로 다음 20일 실현변동성 상위여부 AUC 0.87, 20일 내 -5% 낙폭 AUC 0.76, 다음 20일 상승 여부 AUC 0.52.",
]
HONESTY_BOLD = "따라서 이 플랫폼의 주 목표는 \"다음 한 달의 위험\"이며, 방향 적중률은 항상 기준선 옆에 참고로만 보고한다."
HONESTY_LINE = "방향 적중률은 항상-상승 기준선 옆의 참고값일 뿐이며, 이 플랫폼의 주 목표는 \"다음 한 달의 위험\"입니다. 모든 성적표는 \"12회 중 X회\" 수준의 넓은 구간을 벗어날 수 없습니다."

# VALIDATION.md §3 대체 규약 (섹션 ⑧ 고정 문구)
SUBSTITUTION_RULES = [
    "P3(CNN F&G): 2020-08-03부터만 실이력. 그 이전 구간은 v0 규칙대로 \"조회 실패 시 가중치 제외\"와 동일하게 제외.",
    "P6(마감 30분 매수): 1시간봉 730세션(2023-10~)부터만. 그 이전은 제외.",
    "P7(주도주): 워치리스트 28종목 중 그 시점 65봉 이상 존재하는 종목만. 2013년 22개 → 2026년 27~28개. 생존편향(2026년에 고른 종목)은 제거 불가 — 결과 해석 시 명시. n_watch_avail 을 리포트에 표시.",
    f"6개 신호(P1·P2·P4·P5·P7·P9) 공통 재현 가능 구간: {BACKTEST_START} ~ (BTC-USD 2014-09 시작, 2y 달력 창(500~507거래일) 확보 후). 이것이 BACKTEST_START.",
]

# 표 머리글 한국어 라벨 (알 수 없는 키는 원문 표시)
LABELS = {
    "tone": "톤", "h": "지평(일)", "horizon": "지평(일)", "n": "표본(일)", "n_days": "표본(일)", "n_blocks": "독립 창 수",
    "hit": "적중률", "hit_rate": "적중률", "base": "기준선(항상-상승)", "base_rate": "기준선(항상-상승)", "baseline": "기준선(항상-상승)",
    "always_up": "기준선(항상-상승)", "edge": "우위(적중-기준선)", "ci_lo": "95% 하한", "ci_hi": "95% 상한", "ci_low": "95% 하한",
    "ci_high": "95% 상한", "ci": "95% 구간", "pred": "예측 방향", "pred_up": "상승 예측",
    "fwd_mean": "선행수익 평균", "fwd_median": "선행수익 중앙", "fwd_p10": "10분위", "fwd_p90": "90분위",
    "mean": "평균", "median": "중앙", "p10": "10분위", "p90": "90분위", "std": "표준편차",
    "dd5_rate": "-5% 낙폭 비율(20일)", "y_dd5_20": "-5% 낙폭 비율(20일)", "y_dd10_60": "-10% 낙폭 비율(60일)",
    "y_sign_5": "5일 상승", "y_sign_20": "20일 상승", "y_sign_60": "60일 상승", "y_vol_20": "고변동성(20일)",
    "fwd_ret_5": "5일 선행수익", "fwd_ret_20": "20일 선행수익", "fwd_ret_60": "60일 선행수익",
    "fwd_maxdd_20": "20일 최대낙폭", "fwd_maxdd_60": "60일 최대낙폭",
    "peak_date": "고점일", "trough_date": "저점일", "depth": "낙폭", "days_to_trough": "고점→저점(일)",
    "recovery_date": "회복일", "days_to_recover": "회복 소요(일)", "first_warn_date": "첫 경고일", "first_warning": "첫 경고일",
    "first_warn": "첫 경고일", "warn_date": "첫 경고일", "warn_tone": "경고 톤", "lead_days": "리드타임(일, +=고점 전)",
    "held_to_trough": "저점까지 경고 유지", "held": "저점까지 경고 유지", "missed": "놓침", "detected": "탐지",
    "detection_rate": "탐지율", "detect_rate": "탐지율", "n_episodes": "에피소드 수", "n_detected": "탐지 수", "n_missed": "놓침 수",
    "median_lead_days": "중앙 리드타임(일)", "median_lead": "중앙 리드타임(일)", "false_alarms_per_year": "오경보/년",
    "mean_lead_days": "평균 리드타임(일)", "median_lead_days_fresh": "중앙 리드타임(새로 켜진 경고만, 일)",
    "lookback": "탐색 창(고점 전 최대 거래일)", "n_lead_capped": "상한(탐색 창)에 걸린 리드 수",
    "n_lead_positive": "고점 전 경고 수", "n_held_to_trough": "저점까지 경고 유지 수", "n_evaluable": "평가 가능 에피소드",
    "n_episodes_total_1993": "에피소드 수(1993~)", "warn_share": "경고 톤 비중", "n_warn_runs": "경고 런 수",
    "n_true_alarms": "진짜 경보 수", "n_false_alarms": "오경보 수", "n_unresolved_alarms": "미판정 경보 수",
    "false_alarm_rate": "오경보율", "false_alarm_rate_baseline": "오경보율 기준선(임의의 날을 경고라 했을 때)",
    "true_alarm_share": "진짜 경보 비중", "true_alarm_share_baseline": "진짜 경보 기준선(-5%/20일 기저율)",
    "null_true_alarm_share": "진짜 경보 비중(무작위 이동 기준선)", "null": "무작위 순환 이동 기준선",
    "n_shift": "이동 표본 수", "detection_rate_mean": "탐지율 평균", "detection_100_share": "탐지율 100% 인 이동 비율",
    "median_lead_days_p5": "중앙 리드타임 5분위(일)", "median_lead_days_p50": "중앙 리드타임 중앙값(일)",
    "median_lead_days_p95": "중앙 리드타임 95분위(일)", "true_alarm_share_mean": "진짜 경보 비중 평균",
    "n_tone_switches": "톤 전환 수", "tone_switches_per_year": "톤 전환/년", "median_run_len": "중앙 런 길이(일)",
    "median_warn_run_len": "중앙 경고 런 길이(일)", "maxdd_improvement": "MaxDD 개선(배분-보유, 양수=개선)",
    "excess_cagr": "초과 CAGR(배분-보유)",
    "false_alarm_per_year": "오경보/년", "false_alarms": "오경보 수", "switches_per_year": "톤 전환/년", "n_switches": "톤 전환 수",
    "median_run_days": "중앙 런 길이(일)", "median_run": "중앙 런 길이(일)", "mean_run_days": "평균 런 길이(일)",
    "cagr": "CAGR", "maxdd": "MaxDD", "max_dd": "MaxDD", "worst_month": "최악 월", "worst_month_date": "최악 월(일자)",
    "total_return": "총수익", "vol": "변동성(연)", "sharpe": "샤프", "cost_bps": "비용(bp)", "exposure_mean": "평균 비중",
    "buy_hold": "보유(B&H)", "allocation": "배분", "strategy": "배분", "bh": "보유(B&H)", "diff": "차이",
    "overall_m": "월간", "overall_w": "주간", "overall_d": "일간", "rule": "규칙", "share": "비중", "count": "일수",
    "start": "시작", "end": "끝", "first": "첫 관측", "last": "마지막 관측", "variant": "변형", "basket": "바스켓",
    "n_watch_avail": "워치리스트 가용 종목", "fg_avail": "F&G 가용", "eod_avail": "마감봉 가용",
    "per_year": "연간", "years": "연수",
    # 장부(ledger.summary) 키
    "n_outcome_20": "결과 확정(20일)", "n_outcome_60": "결과 확정(60일)", "hit_20": "적중(20일)", "base_20": "기준선(20일)",
    "hit_60": "적중(60일)", "base_60": "기준선(60일)", "fwd_ret_20_mean": "20일 선행수익 평균", "fwd_ret_60_mean": "60일 선행수익 평균",
    "n_outcome_dd5": "결과 확정(낙폭)", "key": "구분",
}
# 백분율로 표시할 키 힌트 (값이 |v| <= 1.5 인 float 일 때)
_PCT_HINT = ("hit", "base", "rate", "cagr", "maxdd", "max_dd", "depth", "share", "worst", "ret", "mean", "median",
             "p10", "p90", "ci", "dd5", "dd10", "frac", "pct", "prob", "edge", "total_return", "vol", "std",
             "exposure", "always_up", "y_sign", "y_dd", "y_vol", "brier", "skill")
_NO_PCT = ("days", "count", "sharpe", "bps", "score", "years", "switches", "per_year")

_CSS = r"""
:root{--bg:#0b1220;--card:#111a2b;--line:#1e2a40;--tx:#e6eaf2;--mut:#8a94a8;--dim:#5c6a82;}
*{box-sizing:border-box;margin:0;padding:0}
html{-webkit-text-size-adjust:100%}
body{background:var(--bg);color:var(--tx);font-family:system-ui,-apple-system,"Segoe UI","Malgun Gothic","Apple SD Gothic Neo","Noto Sans KR",sans-serif;font-size:14px;line-height:1.55}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.wrap{max-width:1000px;margin:0 auto;padding:20px 16px 48px;display:flex;flex-direction:column;gap:18px}
.eyebrow{font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--dim)}
h1{font-size:24px;margin-top:4px;line-height:1.25}
h2{font-size:17px;margin-bottom:10px}
h3{font-size:14px;color:var(--mut);margin:14px 0 6px}
.sub{color:var(--mut);margin-top:4px}
.tags{margin-top:10px;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.tag{border:1px solid var(--line);border-radius:999px;padding:4px 12px;font-size:12px;color:var(--mut)}
.tag b{color:var(--tx)}
.tag.warn{border-color:#eab30855;color:#eab308;background:#eab30814}
.tag.bad{border-color:#ef444466;color:#ef4444;background:#ef444414}
.panel{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px}
.nav{display:flex;flex-wrap:wrap;gap:6px}
.nav a{color:var(--mut);text-decoration:none;border:1px solid var(--line);border-radius:999px;padding:3px 10px;font-size:12px}
.nav a:hover{color:var(--tx)}
a{color:#7dd3fc}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.kpi{background:#0e1626;border:1px solid var(--line);border-radius:12px;padding:12px}
.kpi .lb{font-size:11px;color:var(--dim)}
.kpi .v{font-size:20px;font-weight:600;margin-top:2px}
.kpi .s{font-size:11px;color:var(--mut);margin-top:2px}
.tblwrap{overflow-x:auto;margin-top:8px;-webkit-overflow-scrolling:touch}
table{width:100%;border-collapse:collapse;font-size:13px}
th{color:var(--dim);font-weight:500;text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);font-size:12px;white-space:nowrap}
td{padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap;vertical-align:top}
tr:last-child td{border-bottom:0}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.pill{display:inline-block;border-radius:999px;padding:1px 9px;font-size:12px;font-weight:600;color:#0b1220}
.st{display:inline-block;border-radius:6px;padding:0 6px;font-size:12px;font-weight:600;color:#0b1220}
.quote{border-left:3px solid #eab308;padding:10px 14px;background:#0e1626;border-radius:8px;margin-top:10px;font-size:13px;line-height:1.7}
.quote ul{padding-left:18px}
.note{font-size:12px;color:var(--dim);margin-top:8px;line-height:1.6}
.warnlist{padding-left:18px;font-size:13px;line-height:1.7;color:#eab308}
.chart{margin-top:12px}
.chart img{width:100%;height:auto;border-radius:10px;border:1px solid var(--line);display:block}
.cap{font-size:12px;color:var(--dim);margin-top:4px}
.verdict{font-size:22px;font-weight:700;margin-top:8px;line-height:1.3}
.v-act{margin-top:4px;font-size:13px;color:var(--mut)}
.tfrow{display:flex;flex-wrap:wrap;gap:14px;margin-top:12px}
.tf{display:flex;flex-direction:column;gap:2px;min-width:90px}
.tf .lb{font-size:11px;color:var(--dim)}
.tf .sc{font-size:12px;color:var(--mut)}
.sigs{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}
.sig{border:1px solid var(--line);border-radius:8px;padding:4px 8px;font-size:12px;color:var(--mut);display:flex;gap:6px;align-items:center}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
.foot{font-size:12px;color:var(--dim);line-height:1.7}
ol.method{padding-left:20px;font-size:13px;line-height:1.75}
ul.plain{padding-left:20px;font-size:13px;line-height:1.75}
.mut{color:var(--mut);font-size:13px}
.p2bar{display:flex;align-items:center;gap:8px;font-size:12px;margin-top:6px}
.p2bar .lb{width:110px;color:var(--mut);flex:none}
.p2bar .tr{flex:1;height:10px;background:#0e1626;border-radius:5px;position:relative;overflow:hidden}
.p2bar .fl{position:absolute;top:0;left:0;height:10px;border-radius:5px}
.p2bar .vv{width:170px;text-align:right;flex:none;font-variant-numeric:tabular-nums}
.dod{margin-top:8px;font-size:13px;line-height:1.6}
.info{color:var(--mut);border:1px dashed var(--line);border-radius:12px;padding:12px;margin-top:6px}
.acc{border-radius:12px;padding:12px;border:1px solid var(--line);margin-top:8px}
.strip{border-top:1px solid var(--line);margin-top:14px;padding-top:10px;font-size:12px;color:var(--mut);line-height:1.7}
.strip ul{padding-left:18px}
.v0line{font-size:13px;line-height:1.7;border-left:3px solid #7dd3fc;padding:8px 12px;background:#0e1626;border-radius:8px}
@media(max-width:520px){h1{font-size:20px}.verdict{font-size:19px}.kpi .v{font-size:18px}.p2bar .vv{width:120px}}
"""


# ------------------------------------------------------------------
# 값 포매팅
# ------------------------------------------------------------------
def _esc(x) -> str:
    return _h.escape("" if x is None else str(x), quote=True)


def _is_nan(v) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v)) if np.isscalar(v) or isinstance(v, (pd.Timestamp, type(pd.NaT))) else False
    except (TypeError, ValueError):
        return False


def _pct_key(key: str) -> bool:
    """이 키의 float 값을 백분율로 표시할지. 개수(n, n_*, *_n)·일수·점수·bp 는 제외."""
    k = (key or "").lower()
    if k == "n" or k.startswith("n_") or k.endswith("_n"):
        return False
    if any(s in k for s in _NO_PCT):
        return False
    return any(s in k for s in _PCT_HINT)


def _plain_float(f: float) -> str:
    """백분율이 아닌 실수: 3자리(|f|<10) 또는 2자리, 뒤따르는 0 제거 (2.500 → 2.5, 14.000 → 14)."""
    s = f"{f:,.3f}" if abs(f) < 10 else f"{f:,.2f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def _fmt(v, key: str = "") -> str:
    """값 → 표시 문자열(HTML escape 완료). 톤/상태는 색 pill."""
    if _is_nan(v):
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return "예" if v else "아니오"
    if isinstance(v, (pd.Timestamp, datetime)):
        return _esc(pd.Timestamp(v).strftime("%Y-%m-%d"))
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if math.isinf(f):
            return "∞" if f > 0 else "-∞"
        if _pct_key(key) and abs(f) <= 1.5:
            return f"{f * 100:.1f}%"
        if f.is_integer() and abs(f) < 1e6 and ("days" in key.lower() or key.lower().startswith("n")):
            return f"{int(f):,}"
        return _plain_float(f)
    if isinstance(v, str):
        if v in TONE_COLORS:
            return _tone_pill(v)
        if v in STATE_COLORS:
            return _state_pill(v)
        return _esc(v)
    if isinstance(v, (list, tuple)):
        if len(v) == 2 and all(isinstance(x, (int, float, np.integer, np.floating)) for x in v):
            return f"[{_fmt(v[0], key)}, {_fmt(v[1], key)}]"
        return _esc(", ".join(str(x) for x in v))
    if isinstance(v, dict):
        return _esc(", ".join(f"{k}: {val}" for k, val in v.items()))
    return _esc(v)


def _tone_pill(t: str) -> str:
    c = TONE_COLORS.get(t, PALETTE["mut"])
    return f'<span class="pill" style="background:{c}">{_esc(t)} {_esc(TONE_KO.get(t, ""))}</span>'


def _state_pill(s: str) -> str:
    c = STATE_COLORS.get(s, PALETTE["mut"])
    return f'<span class="st" style="background:{c}">{_esc(STATE_LABEL.get(s, s))}</span>'


def _label(key) -> str:
    k = str(key)
    return _esc(LABELS.get(k, LABELS.get(k.lower(), k)))


def _is_num(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


# ------------------------------------------------------------------
# 표 렌더링 (records / dict 를 방어적으로)
# ------------------------------------------------------------------
def _to_records(obj) -> list[dict]:
    """list[dict] / DataFrame / dict-of-lists / dict-of-dicts → list[dict]. 못 바꾸면 []."""
    if obj is None:
        return []
    if isinstance(obj, pd.DataFrame):
        df = obj
        if not isinstance(df.index, pd.RangeIndex):
            df = df.reset_index()
        return df.to_dict("records")
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    if isinstance(obj, dict):
        if obj and all(isinstance(v, dict) for v in obj.values()):
            return [{"key": k, **v} for k, v in obj.items()]
        if obj and all(isinstance(v, (list, tuple)) for v in obj.values()):
            try:
                return pd.DataFrame(obj).to_dict("records")
            except ValueError:
                return []
    return []


def _records_table(records, order: list[str] | None = None, empty_msg="자료 없음", fmt=None) -> str:
    fmt = fmt or _fmt
    recs = _to_records(records)
    if not recs:
        return f'<div class="note">{_esc(empty_msg)}</div>'
    cols: list[str] = []
    for r in recs:
        for k in r.keys():
            if k not in cols:
                cols.append(k)
    if order:
        cols = [c for c in order if c in cols] + [c for c in cols if c not in order]
    numeric = {c: all(_is_num(r.get(c)) or _is_nan(r.get(c)) for r in recs) for c in cols}
    head = "".join(f'<th class="{"num" if numeric[c] else ""}">{_label(c)}</th>' for c in cols)
    body = []
    for r in recs:
        cells = "".join(f'<td class="{"num" if numeric[c] else ""}">{fmt(r.get(c), c)}</td>' for c in cols)
        body.append(f"<tr>{cells}</tr>")
    return f'<div class="tblwrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _kv_table(d, empty_msg="자료 없음", fmt=None) -> str:
    """dict → 키/값 표. 값이 dict 이면(예: allocation vs buy_hold) 행렬 표로."""
    fmt = fmt or _fmt
    if not isinstance(d, dict) or not d:
        return f'<div class="note">{_esc(empty_msg)}</div>'
    sub = {k: v for k, v in d.items() if isinstance(v, dict)}
    flat = {k: v for k, v in d.items() if not isinstance(v, (dict, list, tuple, pd.DataFrame, pd.Series))}
    lists = {k: v for k, v in d.items() if isinstance(v, (list, tuple, pd.DataFrame, pd.Series))}
    parts = []
    if flat:
        rows = "".join(f"<tr><td>{_label(k)}</td><td class=\"num\">{fmt(v, k)}</td></tr>" for k, v in flat.items())
        parts.append(f'<div class="tblwrap"><table><thead><tr><th>항목</th><th class="num">값</th></tr></thead><tbody>{rows}</tbody></table></div>')
    if sub:
        metrics: list[str] = []
        for v in sub.values():
            for k in v.keys():
                if k not in metrics:
                    metrics.append(k)
        head = "".join(f'<th class="num">{_label(k)}</th>' for k in sub.keys())
        rows = []
        for m in metrics:
            cells = "".join(f'<td class="num">{fmt(v.get(m), m)}</td>' for v in sub.values())
            rows.append(f"<tr><td>{_label(m)}</td>{cells}</tr>")
        parts.append(f'<div class="tblwrap"><table><thead><tr><th>지표</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    for k, v in lists.items():
        recs = _to_records(v)
        if recs:
            parts.append(f"<h3>{_label(k)}</h3>" + _records_table(recs, fmt=fmt))
        else:
            parts.append(f'<div class="note">{_label(k)}: {fmt(v, k)}</div>')
    return "".join(parts) if parts else f'<div class="note">{_esc(empty_msg)}</div>'


def _scalar_ok(v) -> bool:
    return not isinstance(v, (dict, list, tuple)) and not _is_nan(v)


def _find(d, keys, default=None):
    """dict 에서 후보 키를 순서대로 찾는다(스칼라만; 1단계 중첩 dict 까지)."""
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and _scalar_ok(d[k]):
            return d[k]
    for v in d.values():
        if isinstance(v, dict):
            for k in keys:
                if k in v and _scalar_ok(v[k]):
                    return v[k]
    return default


def _img(png: bytes | None, caption: str, alt: str) -> str:
    if not png:
        return f'<div class="note">차트 없음 — {_esc(caption)}</div>'
    b64 = base64.b64encode(png).decode("ascii")
    return (f'<div class="chart"><img src="data:image/png;base64,{b64}" alt="{_esc(alt)}">'
            f'<div class="cap">{_esc(caption)}</div></div>')


def _kpi(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="kpi"><div class="lb">{_esc(label)}</div><div class="v">{value}</div>'
            f'<div class="s">{_esc(sub)}</div></div>')


def _warn_list(items) -> str:
    items = [str(w) for w in (items or []) if w is not None and str(w).strip()]
    if not items:
        return '<div class="note">경고 없음</div>'
    return '<ul class="warnlist">' + "".join(f"<li>{_esc(w)}</li>" for w in items) + "</ul>"


def _write_html(out_html: Path, title: str, body: str) -> None:
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    doc = ('<!doctype html><html lang="ko"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width,initial-scale=1">'
           f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body><div class=\"wrap\">{body}</div></body></html>")
    with open(out_html, "w", encoding="utf-8", newline="\n") as f:
        f.write(doc)


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ------------------------------------------------------------------
# 백테스트 리포트
# ------------------------------------------------------------------
def _directional_records(summary: dict) -> list[dict]:
    return _to_records(summary.get("directional"))


def _h_of(r: dict):
    for k in ("h", "horizon", "H"):
        if k in r:
            try:
                return int(r[k])
            except (TypeError, ValueError):
                return r[k]
    return None


def render_backtest_report(summary: dict, out_html: Path, charts: dict[str, bytes]) -> None:
    """섹션 ①~⑨ 자체 완결 HTML. summary 에 없는 키는 '자료 없음'으로 표시(예외 없음)."""
    if not isinstance(summary, dict):
        raise TypeError("summary 는 dict 여야 합니다")
    charts = dict(charts or {})
    variant = str(summary.get("variant") or "faithful")
    n_days = summary.get("n_days")
    dr = summary.get("data_range") if isinstance(summary.get("data_range"), dict) else {}
    start = dr.get("start") or dr.get("first") or summary.get("start") or "?"
    end = dr.get("end") or dr.get("last") or summary.get("end") or "?"
    if n_days is None:
        n_days = dr.get("n_days")
    warns = list(summary.get("warnings") or [])
    ep_sum = summary.get("episode_summary") if isinstance(summary.get("episode_summary"), dict) else {}
    alloc = summary.get("allocation") if isinstance(summary.get("allocation"), dict) else {}
    switches = summary.get("switches") if isinstance(summary.get("switches"), dict) else {}
    base_rates = summary.get("base_rates") if isinstance(summary.get("base_rates"), dict) else {}
    directional = _directional_records(summary)

    # ---- 헤더 ----
    tags = [f'<span class="tag">변형 <b>{_esc(variant)}</b></span>',
            f'<span class="tag">구간 <b>{_esc(start)} ~ {_esc(end)}</b></span>',
            f'<span class="tag">거래일 <b>{_fmt(n_days, "n_days")}</b></span>',
            f'<span class="tag">생성 <b>{_esc(summary.get("generated_at") or _now_str())}</b></span>']
    if warns:
        tags.append(f'<span class="tag warn">경고 {len(warns)}건</span>')
    head = ('<header><div class="eyebrow">market-risk-lab · Phase 1 · 실험 #0</div>'
            f'<h1>v0 규칙 백테스트 — {_esc(VARIANT_KO.get(variant, variant))}</h1>'
            '<div class="sub">기존 대시보드(v0) 규칙을 한 줄도 바꾸지 않고, 과거 매일 그날까지의 데이터만으로 판정을 재현해 채점한 결과입니다.</div>'
            f'<div class="tags">{"".join(tags)}</div></header>')
    sec_titles = ["① 정직한 요약", "② 방향 성적표", "③ 에피소드 표", "④ 톤별 선행수익", "⑤ 배분 vs 보유",
                  "⑥ 판정 전환 빈도", "⑦ 125칸 점유표", "⑧ 데이터·규약·경고", "⑨ 방법"]
    nav = '<nav class="nav">' + "".join(f'<a href="#s{i + 1}">{_esc(t)}</a>' for i, t in enumerate(sec_titles)) + "</nav>"

    # ---- ① 정직한 요약 ----
    det = _find(ep_sum, ["detection_rate", "detect_rate", "hit_rate"])
    n_ep = _find(ep_sum, ["n_episodes", "n"])
    n_det = _find(ep_sum, ["n_detected", "detected"])
    lead = _find(ep_sum, ["median_lead_days", "median_lead", "lead_median"])
    fa = _find(ep_sum, ["false_alarms_per_year", "false_alarm_per_year", "false_alarms_py"])
    spy_ = _find(ep_sum, ["switches_per_year"])
    if spy_ is None:
        spy_ = _find(switches, ["switches_per_year", "per_year"])
    a_cagr = _find(alloc, ["cagr", "cagr_alloc", "cagr_strategy"])
    b_cagr = _find(alloc, ["cagr_bh", "cagr_buy_hold", "bh_cagr"])
    a_dd = _find(alloc, ["maxdd", "max_dd", "maxdd_alloc"])
    b_dd = _find(alloc, ["maxdd_bh", "max_dd_bh", "bh_maxdd"])
    # 중첩 형태 {"allocation": {...}, "buy_hold": {...}} 도 지원
    for k_a, k_b in (("allocation", "buy_hold"), ("strategy", "buy_hold"), ("alloc", "bh")):
        if isinstance(alloc.get(k_a), dict) and isinstance(alloc.get(k_b), dict):
            a_cagr = a_cagr if a_cagr is not None else alloc[k_a].get("cagr")
            b_cagr = b_cagr if b_cagr is not None else alloc[k_b].get("cagr")
            a_dd = a_dd if a_dd is not None else (alloc[k_a].get("maxdd") if "maxdd" in alloc[k_a] else alloc[k_a].get("max_dd"))
            b_dd = b_dd if b_dd is not None else (alloc[k_b].get("maxdd") if "maxdd" in alloc[k_b] else alloc[k_b].get("max_dd"))
    # 20일 방향: caution/reduce 톤의 적중 vs 기준선 (있으면)
    dir20 = [r for r in directional if _h_of(r) == 20]
    hit_kpis = []
    for r in dir20:
        t = r.get("tone")
        hit = r.get("hit", r.get("hit_rate"))
        base = r.get("base", r.get("base_rate", r.get("baseline")))
        if t in TONE_COLORS and not _is_nan(hit):
            hit_kpis.append(_kpi(f"20일 적중 · {TONE_KO.get(t, t)}", _fmt(hit, "hit"),
                                 f"기준선 {_fmt(base, 'base')} · n={_fmt(r.get('n'), 'n')} · 창 {_fmt(r.get('n_blocks'), 'n_blocks')}"))
    # 기준선(VALIDATION.md §5 "항상 기준선 옆에"): 무작위 순환 이동(null) 탐지율·리드, 임의의 날 오경보율, 탐색 창 상한
    lookback = _find(ep_sum, ["lookback"])
    fresh = _find(ep_sum, ["median_lead_days_fresh"])
    null = ep_sum.get("null") if isinstance(ep_sum.get("null"), dict) else {}
    null_det, null_lead = null.get("detection_rate_mean"), null.get("median_lead_days_p50")
    far = _find(ep_sum, ["false_alarm_rate"])
    far_base = _find(ep_sum, ["false_alarm_rate_baseline"])
    if far is None:
        far = _find(switches, ["false_alarm_rate"])
    if far_base is None:
        far_base = _find(switches, ["false_alarm_rate_baseline"])
    det_sub = (f"{_fmt(n_det, 'n')} / {_fmt(n_ep, 'n')}회" if n_ep is not None else "episode_summary 기준")
    if not _is_nan(null_det):
        det_sub += f" · 무작위 이동 기준 {_fmt(null_det, 'detection_rate')}"
    lead_sub = "양수 = 고점 전 경고"
    if lookback is not None:
        lead_sub += f" · 탐색 창 최대 {_fmt(lookback, 'lookback')}일"
    if not _is_nan(fresh):
        lead_sub += f" · 새로 켜진 경고만 {_fmt(fresh, 'median_lead_days_fresh')}일"
    if not _is_nan(null_lead):
        lead_sub += f" · 무작위 {_fmt(null_lead, 'median_lead_days_p50')}일"
    fa_sub = "경고 뒤 20일 내 -5% 없음"
    if not _is_nan(far):
        fa_sub += f" · 오경보율 {_fmt(far, 'false_alarm_rate')}"
    if not _is_nan(far_base):
        fa_sub += f" · 임의의 날 기준 {_fmt(far_base, 'false_alarm_rate_baseline')}"
    kpis = [
        _kpi("≥10% 에피소드 탐지율", _fmt(det, "detection_rate"), det_sub),
        _kpi("중앙 리드타임", (f"{_fmt(lead, 'lead_days')}일" if lead is not None else "—"), lead_sub),
        _kpi("오경보 / 년", _fmt(fa, "false_alarms_per_year"), fa_sub),
        _kpi("톤 전환 / 년", _fmt(spy_, "switches_per_year"), ""),
        _kpi("배분 CAGR", _fmt(a_cagr, "cagr"), f"보유 {_fmt(b_cagr, 'cagr')}"),
        _kpi("배분 MaxDD", _fmt(a_dd, "maxdd"), f"보유 {_fmt(b_dd, 'maxdd')}"),
    ] + hit_kpis
    honesty_extra = summary.get("honesty")
    s1 = ('<section class="panel" id="s1"><h2>① 정직한 요약</h2>'
          f'<div class="grid">{"".join(kpis)}</div>'
          '<div class="quote"><b>' + _esc(HONESTY_TITLE) + "</b><ul>"
          + "".join(f"<li>{_esc(b)}</li>" for b in HONESTY_BULLETS)
          + f"<li><b>{_esc(HONESTY_BOLD)}</b></li></ul></div>"
          + (f'<div class="note">{_esc(honesty_extra)}</div>' if honesty_extra else "")
          + '<div class="note">위 문단은 VALIDATION.md §0 의 사전 등록 문구를 그대로 옮긴 것으로, 결과와 무관하게 고정됩니다. '
            '숫자 카드는 evaluate 요약에 해당 항목이 있을 때만 채워집니다("—" 는 자료 없음).</div></section>')

    # ---- ② 방향 성적표 ----
    order2 = ["tone", "h", "horizon", "n", "n_blocks", "hit", "hit_rate", "base", "base_rate", "baseline", "edge",
              "ci_lo", "ci_low", "ci_hi", "ci_high", "ci", "fwd_mean", "fwd_median", "fwd_p10", "fwd_p90", "dd5_rate", "y_dd5_20"]
    s2 = ('<section class="panel" id="s2"><h2>② 방향 성적표 — 항상-상승 기준선 옆에</h2>'
          '<div class="note">buy/hold/neutral 톤은 "h일 뒤 상승"을, caution/reduce 톤은 "하락"을 예측한 것으로 채점. '
          '기준선 = 같은 표본에서 무조건 상승이라고 말했을 때의 적중률. 독립 창 수(n_blocks)와 95% 블록 부트스트랩 구간이 있으면 함께 표시.</div>'
          + _records_table(directional, order2, "directional 기록 없음")
          + (f"<h3>기저율(base_rates)</h3>{_kv_table(base_rates)}" if base_rates else "")
          + "</section>")

    # ---- ③ 에피소드 표 ----
    order3 = ["peak_date", "trough_date", "depth", "days_to_trough", "recovery_date", "days_to_recover",
              "first_warn_date", "first_warning", "first_warn", "warn_date", "warn_tone", "lead_days", "held_to_trough", "held", "missed", "detected"]
    ep10 = summary.get("episodes10")
    ep5 = summary.get("episodes5")
    ep20 = summary.get("episodes20")
    s3 = ('<section class="panel" id="s3"><h2>③ 에피소드 표 — SPY 고점 대비 낙폭 (1993~)</h2>'
          '<div class="note">저점 이후 직전 고점을 회복하면 종료. 회복 전 다시 임계 이상 재하락은 새 에피소드로 세되 회복일은 공유. '
          '리드타임 = 첫 caution/reduce 톤 등장일과 고점일의 거래일 차(양수 = 고점 전 경고). 재현 구간 밖의 에피소드는 v0 판정이 없어 "—". '
          f'경고 탐색은 고점 전 최대 lookback={_fmt(lookback, "lookback") if lookback is not None else 20}거래일까지만 하므로 리드타임은 그 값을 넘지 못하고, '
          '그 전부터 켜져 있던 경고는 lead_capped 로 표시된다(새로 켜진 경고만의 중앙 리드는 median_lead_days_fresh). '
          '탐지율·리드타임은 경고 시계열을 무작위로 순환 이동한 기준선과 나란히 읽는다 — 구별되지 않으면 탐색 창 × 경고 비중의 산물이다.</div>'
          + (f"<h3>요약</h3>{_kv_table(ep_sum)}" if ep_sum else "")
          + "<h3>≥10% 낙폭</h3>" + _records_table(ep10, order3, "episodes10 없음")
          + "<h3>≥5% 낙폭</h3>" + _records_table(ep5, order3, "episodes5 없음")
          + (("<h3>≥20% 낙폭</h3>" + _records_table(ep20, order3)) if ep20 is not None else "")
          + _img(charts.pop("tone_bands", None), "SPY 종가(로그)와 v0 톤 밴드. 세로 점선 = ≥10% 에피소드 고점(빨강)·저점(파랑).", "SPY price with tone bands")
          + "</section>")

    # ---- ④ 톤별 선행수익 분포 ----
    dist_cols = ["tone", "h", "horizon", "n", "fwd_mean", "fwd_median", "fwd_p10", "fwd_p90", "dd5_rate", "y_dd5_20"]
    dist_recs = []
    for r in directional:
        sub = {k: r[k] for k in dist_cols if k in r}
        if any(k in sub for k in ("fwd_mean", "fwd_median", "fwd_p10", "fwd_p90", "dd5_rate", "y_dd5_20")):
            dist_recs.append(sub)
    s4 = ('<section class="panel" id="s4"><h2>④ 톤별 선행수익 분포</h2>'
          '<div class="note">각 톤이 나온 날로부터 h거래일 뒤 SPY 수익률의 분포(평균·중앙·10/90분위)와 20일 내 -5% 낙폭 비율. 겹치는 창이므로 표본 수는 과대평가.</div>'
          + _img(charts.pop("fwd_box", None), "톤별 다음 20거래일 SPY 수익률 상자그림(가운데 선 = 중앙값, 수염 = 5/95분위).", "Forward 20-day return by tone")
          + (_records_table(dist_recs, dist_cols) if dist_recs else '<div class="note">분포 열(fwd_mean 등)이 directional 에 없음</div>')
          + "</section>")

    # ---- ⑤ 배분 시뮬 vs 보유 ----
    exp_txt = " · ".join(f"{TONE_KO.get(t, t)} {int(round(e * 100))}%" for t, e in TONE_EXPOSURE.items())
    s5 = ('<section class="panel" id="s5"><h2>⑤ 배분 시뮬레이션 vs 보유</h2>'
          f'<div class="note">톤→주식 비중: {_esc(exp_txt)} (VALIDATION.md 사전 등록). t일 종가 판정을 t+1일 수익에 적용, 전환 시 비용 반영(기본 5bp). 나머지는 현금(무이자).</div>'
          + _kv_table(alloc, "allocation 결과 없음")
          + _img(charts.pop("cumret", None), "누적 수익(로그): SPY 보유 vs v0 톤 배분.", "Cumulative return: buy&hold vs allocation")
          + "</section>")

    # ---- ⑥ 판정 전환 빈도 ----
    s6 = ('<section class="panel" id="s6"><h2>⑥ 판정 전환 빈도</h2>'
          '<div class="note">톤이 바뀐 날의 수. 잦은 전환은 비용과 "말 바꾸기"를 뜻하고, 너무 드문 전환은 경고가 늦다는 뜻입니다.</div>'
          + _kv_table(switches, "switches 결과 없음")
          + _img(charts.pop("switches", None), "월별 톤 전환 횟수.", "Monthly tone switches")
          + "</section>")

    # ---- ⑦ 125칸 점유표 ----
    cells = _to_records(summary.get("cells"))
    if cells:
        tot = sum(float(r.get("n", 0) or 0) for r in cells)
        for r in cells:
            if "share" not in r and tot > 0 and r.get("n") is not None:
                r["share"] = float(r["n"]) / tot
        cells = sorted(cells, key=lambda r: -(float(r.get("n", 0) or 0)))
    s7 = ('<section class="panel" id="s7"><h2>⑦ 125칸 점유표 — (월간, 주간, 일간) 상태 조합</h2>'
          f'<div class="note">5상태³ = 125칸 중 실제로 나타난 칸만 표시(점유일 내림차순, 총 {len(cells)}칸). 규칙 번호 = combo_advice 의 첫 일치 규칙(0 = 평균 폴백).</div>'
          + _records_table(cells, ["overall_m", "overall_w", "overall_d", "n", "share", "tone", "rule"], "cells 결과 없음")
          + "</section>")

    # ---- ⑧ 데이터 범위·대체 규약·경고 ----
    s8 = ('<section class="panel" id="s8"><h2>⑧ 데이터 범위 · 대체 규약 · 경고</h2>'
          "<h3>데이터 범위</h3>" + _kv_table(dr, "data_range 없음")
          + "<h3>대체 규약 (VALIDATION.md §3)</h3><ul class=\"plain\">"
          + "".join(f"<li>{_esc(t)}</li>" for t in SUBSTITUTION_RULES) + "</ul>"
          + "<h3>경고</h3>" + _warn_list(warns))
    # 남은 차트(알 수 없는 이름)는 여기서 모두 보여준다 — 숨기지 않음
    for k, png in list(charts.items()):
        s8 += _img(png, f"추가 차트: {k}", k)
    s8 += "</section>"

    # ---- ⑨ 방법 설명 ----
    s9 = ('<section class="panel" id="s9"><h2>⑨ 방법 설명</h2><ol class="method">'
          '<li><b>재현</b>: 매 거래일 t에 대해 v0 라이브가 보는 창(SPY·VIX·FANG = Yahoo 2y 달력 창 (t-2y, t] 500~507거래일, 워치리스트 1y 창 ≈252거래일, BTC 2y 731행)을 t까지의 캐시로 되살려 '
          'v0 의 build_metrics → assess → composite → overall → combo_advice 를 그대로 실행. 미래 행은 쓰지 않는다.</li>'
          '<li><b>두 변형</b>: faithful 은 라이브처럼 t일 부분 봉·부분 주/월을 포함, completed 는 완성 봉만. 둘의 차이가 부분 봉 문제의 크기.</li>'
          '<li><b>목표변수</b>: y_dd5_20(주 목표: 다음 20거래일 내 -5%), y_dd10_60, y_sign_5/20/60(참고), y_vol_20. 정의는 VALIDATION.md §1.</li>'
          '<li><b>방향 채점</b>: buy/hold/neutral → 상승 예측, caution/reduce → 하락 예측. 항상-상승 기준선과 겹치지 않는 창 수(n//h)를 병기. 95% 구간은 블록 부트스트랩.</li>'
          '<li><b>에피소드</b>: SPY 종가 고점 대비 5/10/20% 낙폭 구간. 첫 경고일·리드타임·놓침·오경보(경고 런 뒤 20일 내 -5% 없음).</li>'
          '<li><b>배분</b>: 톤→비중(100/100/100/50/25%), t 종가 판정을 t+1 수익에 적용, 전환 시 5bp.</li>'
          '<li><b>한계</b>: P3·P6 는 이력이 짧아 그 이전엔 가중치 제외(대체 규약), P7 은 생존편향 제거 불가, 표본은 에피소드 12회 수준 — 모든 숫자는 넓은 구간을 가진다.</li>'
          "</ol>"
          f'<div class="foot">market-risk-lab · v0 동결 벤치마크(실험 #0) · 생성 {_esc(summary.get("generated_at") or _now_str())} · '
          f'<a href="index.html">오늘 판정으로</a></div></section>')

    _write_html(out_html, f"v0 백테스트 ({variant}) — market-risk-lab", head + nav + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9)


# ------------------------------------------------------------------
# 차트
# ------------------------------------------------------------------
def _new_fig(w=9.0, h=4.0):
    from matplotlib.figure import Figure   # pyplot 미사용(전역 상태·GUI 백엔드 회피) — Figure.savefig 가 Agg 사용
    fig = Figure(figsize=(w, h), dpi=110, facecolor=PALETTE["card"])
    ax = fig.add_subplot(111)
    _style_ax(ax)
    return fig, ax


def _style_ax(ax):
    ax.set_facecolor(PALETTE["card"])
    for sp in ax.spines.values():
        sp.set_color(PALETTE["line"])
    ax.tick_params(colors=PALETTE["mut"], labelsize=8)
    ax.xaxis.label.set_color(PALETTE["mut"])
    ax.yaxis.label.set_color(PALETTE["mut"])
    ax.title.set_color(PALETTE["tx"])
    ax.grid(True, color=PALETTE["line"], linewidth=0.6, alpha=0.8)


def _plain_log_axis(ax):
    """로그축 눈금을 5×10² 대신 500 처럼 표시."""
    from matplotlib.ticker import FuncFormatter, NullFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    return buf.getvalue()


def _series_close(spy_close) -> pd.Series | None:
    if spy_close is None:
        return None
    if isinstance(spy_close, pd.DataFrame):
        col = "Close" if "Close" in spy_close.columns else ("SPY" if "SPY" in spy_close.columns else spy_close.columns[0])
        spy_close = spy_close[col]
    s = pd.Series(spy_close).dropna().astype(float)
    if s.empty:
        return None
    idx = pd.DatetimeIndex(pd.to_datetime(s.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    return s[~s.index.duplicated(keep="last")].sort_index()


def _tone_runs(tone: pd.Series):
    """연속 같은 톤 구간 → [(start, end, tone)]."""
    runs = []
    if tone.empty:
        return runs
    cur, start = tone.iloc[0], tone.index[0]
    prev = start
    for d, t in tone.iloc[1:].items():
        if t != cur:
            runs.append((start, prev, cur))
            cur, start = t, d
        prev = d
    runs.append((start, prev, cur))
    return runs


def charts_for(summary, replay, spy_close, targets) -> dict[str, bytes]:
    """차트 4종 → PNG bytes. 열이 없으면 해당 차트만 경고 후 생략."""
    out: dict[str, bytes] = {}
    summary = summary if isinstance(summary, dict) else {}
    tone = None
    if isinstance(replay, pd.DataFrame) and "tone" in replay.columns and len(replay):
        tone = replay["tone"].astype(str)
        idx = pd.DatetimeIndex(pd.to_datetime(tone.index))
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        tone.index = idx.normalize()
        tone = tone.sort_index()
        unknown = sorted(set(tone.unique()) - set(TONE_EXPOSURE))
        if unknown:
            warnings.warn(f"replay.tone 에 알 수 없는 톤 {unknown} → 노출 1.0 으로 처리(차트용)")
    else:
        warnings.warn("replay 에 tone 열이 없거나 비어 있음 → cumret/tone_bands/fwd_box/switches 차트 생략")
    spy = _series_close(spy_close)
    if spy is None:
        warnings.warn("spy_close 가 비어 있음 → cumret/tone_bands 차트 생략")

    # 1) 누적수익: 보유 vs 배분 (t 종가 판정 → t+1 수익; 전환 시 비용)
    if tone is not None and spy is not None:
        try:
            s = spy.reindex(tone.index).dropna()
            if len(s) >= 3:
                alloc = summary.get("allocation") if isinstance(summary.get("allocation"), dict) else {}
                cost = float(alloc.get("cost_bps", 5)) / 1e4
                exp = tone.reindex(s.index).map(TONE_EXPOSURE).fillna(1.0).astype(float)
                ret = s.pct_change().fillna(0.0)
                pos = exp.shift(1).fillna(exp.iloc[0])
                turn = exp.diff().abs().fillna(0.0).shift(1).fillna(0.0)
                strat = pos * ret - turn * cost
                bh = (1 + ret).cumprod()
                al = (1 + strat).cumprod()
                fig, ax = _new_fig()
                ax.plot(bh.index, bh.values, color="#8a94a8", lw=1.2, label="SPY buy & hold")
                ax.plot(al.index, al.values, color="#22c55e", lw=1.4, label=f"v0 tone allocation ({int(cost * 1e4)}bp/switch)")
                ax.set_yscale("log")
                _plain_log_axis(ax)
                ax.set_title("Cumulative growth of 1.0 (log scale)")
                ax.legend(loc="upper left", fontsize=8, facecolor=PALETTE["card"], edgecolor=PALETTE["line"], labelcolor=PALETTE["tx"])
                out["cumret"] = _png(fig)
            else:
                warnings.warn("replay 와 spy_close 의 공통 구간이 3일 미만 → cumret 생략")
        except Exception as e:  # 차트 하나가 전체를 막지 않도록 — 원인은 경고로 남김
            warnings.warn(f"cumret 차트 실패: {type(e).__name__}: {e}")

    # 2) SPY 가격 + 톤 밴드 (+ ≥10% 에피소드 고점/저점 표시)
    if tone is not None and spy is not None:
        try:
            s = spy[(spy.index >= tone.index[0]) & (spy.index <= tone.index[-1])]
            if len(s) >= 3:
                fig, ax = _new_fig(9.0, 4.2)
                for a, b, t in _tone_runs(tone):
                    ax.axvspan(a, b + pd.Timedelta(days=1), color=TONE_COLORS.get(t, PALETTE["mut"]), alpha=0.22, lw=0)
                ax.plot(s.index, s.values, color=PALETTE["tx"], lw=1.0)
                ax.set_yscale("log")
                _plain_log_axis(ax)
                for r in _to_records(summary.get("episodes10")):
                    pk, tr = r.get("peak_date"), r.get("trough_date")
                    for d, c in ((pk, "#ef4444"), (tr, "#60a5fa")):
                        if d is None or _is_nan(d):
                            continue
                        ts = pd.Timestamp(d)
                        if tone.index[0] <= ts <= tone.index[-1]:
                            ax.axvline(ts, color=c, lw=0.8, ls=":", alpha=0.9)
                from matplotlib.patches import Patch
                handles = [Patch(color=TONE_COLORS[t], alpha=0.5, label=t) for t in TONES if t in set(tone.unique())]
                if handles:
                    ax.legend(handles=handles, loc="upper left", fontsize=8, ncol=len(handles), facecolor=PALETTE["card"],
                              edgecolor=PALETTE["line"], labelcolor=PALETTE["tx"])
                ax.set_title("SPY close (log) with v0 tone bands")
                out["tone_bands"] = _png(fig)
            else:
                warnings.warn("tone_bands: SPY 구간 자료 부족 → 생략")
        except Exception as e:
            warnings.warn(f"tone_bands 차트 실패: {type(e).__name__}: {e}")

    # 3) 톤별 fwd_ret_20 상자그림
    if tone is not None and isinstance(targets, pd.DataFrame) and "fwd_ret_20" in targets.columns:
        try:
            fr = targets["fwd_ret_20"]
            fidx = pd.DatetimeIndex(pd.to_datetime(fr.index))
            if fidx.tz is not None:
                fidx = fidx.tz_localize(None)
            fr = pd.Series(fr.to_numpy(), index=fidx.normalize()).astype(float)
            fr = fr[~fr.index.duplicated(keep="last")]
            joined = pd.DataFrame({"tone": tone, "fr": fr.reindex(tone.index)}).dropna()
            groups = [(t, joined.loc[joined["tone"] == t, "fr"].to_numpy() * 100) for t in TONES if (joined["tone"] == t).any()]
            if groups:
                fig, ax = _new_fig(8.0, 4.0)
                bp = ax.boxplot([g for _, g in groups], tick_labels=[f"{t}\n(n={len(g)})" for t, g in groups],
                                whis=(5, 95), showfliers=False, patch_artist=True, widths=0.55,
                                medianprops={"color": PALETTE["tx"], "lw": 1.4})
                for patch, (t, _) in zip(bp["boxes"], groups):
                    patch.set_facecolor(TONE_COLORS.get(t, PALETTE["mut"]))
                    patch.set_alpha(0.75)
                    patch.set_edgecolor(PALETTE["line"])
                for k in ("whiskers", "caps"):
                    for ln in bp[k]:
                        ln.set_color(PALETTE["mut"])
                ax.axhline(0, color=PALETTE["mut"], lw=0.8, ls="--")
                ax.set_ylabel("forward 20-day SPY return (%)")
                ax.set_title("Forward 20-day return by tone (whiskers 5/95 pct)")
                out["fwd_box"] = _png(fig)
            else:
                warnings.warn("fwd_box: replay 와 targets 의 공통 자료 없음 → 생략")
        except Exception as e:
            warnings.warn(f"fwd_box 차트 실패: {type(e).__name__}: {e}")
    elif tone is not None:
        warnings.warn("targets 에 fwd_ret_20 열이 없음 → fwd_box 차트 생략")

    # 4) 월별 톤 전환 횟수
    if tone is not None:
        try:
            sw = (tone != tone.shift(1)).astype(int)
            sw.iloc[0] = 0
            monthly = sw.resample("ME").sum()
            fig, ax = _new_fig(9.0, 3.2)
            ax.bar(monthly.index, monthly.values, width=20, color="#eab308", alpha=0.85)
            ax.set_title(f"Monthly tone switches (total {int(sw.sum())}, {sw.sum() / max(len(sw), 1) * 252:.1f}/yr)")
            ax.set_ylabel("switches")
            out["switches"] = _png(fig)
        except Exception as e:
            warnings.warn(f"switches 차트 실패: {type(e).__name__}: {e}")
    return out


# ------------------------------------------------------------------
# index.html — 오늘 판정
# ------------------------------------------------------------------
_STATUS_KO = {"current": "확정 종가", "intraday": "장중 — 오늘 봉 미완성", "pre_open": "개장 전 — 전일 종가 기준",
              "weekend": "주말 휴장 — 직전 영업일 종가", "holiday": "휴장일 — 직전 영업일 종가", "stale": "데이터 지연 — 직전 영업일 종가",
              "incomplete": "미완성 봉 — 전일 기준"}


def _variants_of(today: dict) -> dict[str, dict]:
    """today 에서 변형별 판정 dict 를 꺼낸다. {'variants': {...}} / {'faithful':..,'completed':..} / 단일 v0_day dict 모두 허용."""
    if isinstance(today.get("variants"), dict) and today["variants"]:
        return {str(k): v for k, v in today["variants"].items() if isinstance(v, dict)}
    vs = {k: today[k] for k in ("faithful", "completed") if isinstance(today.get(k), dict)}
    if vs:
        return vs
    if "tone" in today:
        return {str(today.get("variant") or "faithful"): today}
    return {}


def _verdict_block(name: str, d: dict) -> str:
    tone = str(d.get("tone") or "neutral")
    color = TONE_COLORS.get(tone, PALETTE["mut"])
    verdict = d.get("verdict_ko") or d.get("verdict") or d.get("name") or "판정 없음"
    action = d.get("action_ko") or d.get("action") or ""
    tf = ""
    for lb, ko, sk in (("overall_m", "월간", "score_m"), ("overall_w", "주간", "score_w"), ("overall_d", "일간", "score_d")):
        st = d.get(lb)
        sc = d.get(sk)
        sc_txt = "" if _is_nan(sc) else f"{float(sc) * 100:+.1f}"
        tf += (f'<div class="tf"><div class="lb">{ko}</div><div>{_state_pill(str(st)) if not _is_nan(st) else "—"}</div>'
               f'<div class="sc">{_esc(sc_txt)}</div></div>')
    states = d.get("states_d") if isinstance(d.get("states_d"), dict) else {}
    if not states:
        states = {k: d.get(f"state_{k}") for k in V0_SIGNALS if not _is_nan(d.get(f"state_{k}"))}
    sigs = ""
    for k in V0_SIGNALS:
        v = states.get(k)
        if v is None or _is_nan(v):
            continue
        sigs += f'<div class="sig">{_esc(SIGNAL_KO.get(k, k))} {_state_pill(str(v))}</div>'
    extras = []
    if d.get("fg_avail") is False:
        extras.append('<span class="tag warn">F&G 없음 → 가중치 제외</span>')
    if d.get("eod_avail") is False:
        extras.append('<span class="tag warn">마감봉 없음 → 가중치 제외</span>')
    if d.get("n_watch_avail") is not None:
        extras.append(f'<span class="tag">워치리스트 <b>{_fmt(d.get("n_watch_avail"), "n")}</b>종목</span>')
    if d.get("asof"):
        extras.append(f'<span class="tag">기준일 <b>{_esc(pd.Timestamp(d["asof"]).strftime("%Y-%m-%d"))}</b></span>')
    return (f'<div class="panel"><div class="eyebrow">{_esc(VARIANT_KO.get(name, name))}</div>'
            f'<div class="verdict" style="color:{color}">{_esc(verdict)}</div>'
            f'<div class="v-act">{_esc(action)}</div><div style="margin-top:8px">{_tone_pill(tone)}</div>'
            f'<div class="tfrow">{tf}</div>' + (f'<div class="sigs">{sigs}</div>' if sigs else "")
            + (f'<div class="tags">{"".join(extras)}</div>' if extras else "") + "</div>")


def _ledger_block(ls: dict) -> str:
    if not isinstance(ls, dict) or not ls:
        return '<div class="note">장부 요약 없음</div>'
    n = ls.get("n", 0)
    n_out = ls.get("n_with_outcome", 0)
    kpis = [_kpi("기록 일수", _fmt(n, "n"), f"첫 {ls.get('first_asof') or '—'} ~ 마지막 {ls.get('last_asof') or '—'}"),
            _kpi("결과 확정(20일)", _fmt(n_out, "n"), f"60일 {_fmt(ls.get('n_with_outcome_60'), 'n')}"),
            _kpi("항상-상승 기준선(20일)", _fmt(ls.get("base_rate_20"), "base"), "이 표본의 상승 비율")]
    ov = ls.get("overall") if isinstance(ls.get("overall"), dict) else {}
    if ov.get("hit_20") is not None:
        kpis.append(_kpi("전체 적중(20일)", _fmt(ov.get("hit_20"), "hit"), f"기준선 {_fmt(ov.get('base_20'), 'base')}"))
    rows = []
    by_tone = ls.get("by_tone") if isinstance(ls.get("by_tone"), dict) else {}
    for t in TONES:
        b = by_tone.get(t)
        if not isinstance(b, dict):
            continue
        rows.append({"tone": t, "n": b.get("n"), "n_outcome_20": b.get("n_outcome_20"), "hit_20": b.get("hit_20"),
                     "base_20": b.get("base_20"), "hit_60": b.get("hit_60"), "base_60": b.get("base_60"), "dd5_rate": b.get("dd5_rate")})
    missing = ls.get("missing_days") or []
    miss_txt = ""
    if missing:
        show = ", ".join(missing[:8]) + (" …" if len(missing) > 8 else "")
        miss_txt = f'<div class="note">결측 거래일 {len(missing)}일 ({_esc(ls.get("missing_method") or "")}): {_esc(show)}</div>'
    # 장부 P2 블록의 주석(출처 혼재·구간 생략·정의상 0 사유·홀드아웃 잠금)은 계산만 하고 아무도 안 읽던 값이었다 — 경고 목록에 싣는다
    p2s = ls.get("p2") if isinstance(ls.get("p2"), dict) else {}
    warns = list(ls.get("warnings") or []) + [f"P2: {n}" for n in (p2s.get("notes") or [])]
    return (f'<div class="grid">{"".join(kpis)}</div>'
            + ("<h3>톤별 적중 (기준선 옆에)</h3>" + _records_table(rows) if rows else '<div class="note">아직 톤별 결과 없음</div>')
            + miss_txt + ("<h3>장부 경고</h3>" + _warn_list(warns) if warns else ""))


def render_index(today: dict, ledger_summary: dict, out_html: Path) -> None:
    """docs/index.html — 오늘 판정(faithful·completed), 마지막 갱신, 백테스트 링크, 장부 요약, 정직 문구."""
    if not isinstance(today, dict):
        raise TypeError("today 는 dict 여야 합니다")
    variants = _variants_of(today)
    asof = today.get("asof") or next((v.get("asof") for v in variants.values() if v.get("asof")), None)
    asof_txt = pd.Timestamp(asof).strftime("%Y-%m-%d (%a)") if asof else "—"
    status = str(today.get("market_status") or today.get("status") or ("incomplete" if today.get("incomplete") else "current"))
    status_ko = _STATUS_KO.get(status, status)
    updated = today.get("generated_at") or today.get("updated_at") or today.get("updated_at_utc") or _now_str()
    tags = [f'<span class="tag">기준일 <b>{_esc(asof_txt)}</b></span>',
            f'<span class="tag{" warn" if status != "current" else ""}">{_esc(status_ko)}</span>',
            f'<span class="tag">마지막 갱신 <b>{_esc(updated)}</b></span>']
    if not _is_nan(today.get("spy_close")):
        tags.append(f'<span class="tag">SPY <b>{float(today["spy_close"]):,.2f}</b></span>')
    if not _is_nan(today.get("vix_close")):
        tags.append(f'<span class="tag">VIX <b>{float(today["vix_close"]):,.2f}</b></span>')
    note = today.get("note")
    head = ('<header><div class="eyebrow">market-risk-lab · 오늘의 v0 판정 (완성 봉 기준)</div>'
            '<h1>시장 위험 실험실 — 오늘 판정</h1>'
            '<div class="sub">기존 대시보드와 같은 v0 규칙을, 장 마감 후 완성된 봉으로만 다시 계산해 매일 장부에 기록합니다. 판정은 두 변형(faithful·completed)을 나란히 보여줍니다.</div>'
            f'<div class="tags">{"".join(tags)}</div>' + (f'<div class="note">{_esc(note)}</div>' if note else "") + "</header>")
    if variants:
        blocks = "".join(_verdict_block(k, v) for k, v in variants.items())
        vsec = f'<div class="two">{blocks}</div>'
    else:
        vsec = '<div class="panel"><div class="verdict" style="color:#eab308">오늘 판정 없음</div><div class="note">today 에 변형별 판정(faithful/completed)이 없습니다.</div></div>'
    diff_note = ""
    if len(variants) >= 2:
        tones = {k: str(v.get("tone")) for k, v in variants.items()}
        if len(set(tones.values())) > 1:
            diff_note = ('<div class="panel"><b style="color:#eab308">두 변형의 톤이 다릅니다</b> — '
                         + " · ".join(f"{_esc(k)}: {_esc(t)}" for k, t in tones.items())
                         + '<div class="note">차이는 부분 봉(주/월 미완성 기간)의 영향입니다. 완성 봉 기준(completed)이 재현 가능한 판정입니다.</div></div>')
    # Phase 2 카드 (v0 판정 블록 아래; today["p2"] 가 없으면 Phase 1 페이지와 동일)
    p2sec = ""
    p2 = today.get("p2")
    if isinstance(p2, dict):
        if "live" not in p2 and isinstance(ledger_summary, dict) and isinstance(ledger_summary.get("p2"), dict):
            p2 = {**p2, "live": ledger_summary["p2"]}
        if "asof" not in p2 and asof:
            p2 = {**p2, "asof": asof}
        p2sec = p2_card(p2)
    warns = list(today.get("warnings") or [])
    wsec = f'<section class="panel"><h2>오늘 경고</h2>{_warn_list(warns)}</section>' if warns else ""
    lsec = ('<section class="panel"><h2>장부 요약 — 매일 기록, 20/60거래일 뒤 채점</h2>'
            '<div class="note">기록 시점의 판정을 바꾸지 않고, 시간이 지나 결과가 확정되면 y_sign_20/60·y_dd5_20 을 채웁니다. 적중률은 항상 "항상-상승" 기준선 옆에 둡니다.</div>'
            + _ledger_block(ledger_summary) + "</section>")
    honest = (f'<section class="panel"><h2>정직 문구</h2><div class="quote">{_esc(HONESTY_LINE)}<br>'
              f'<b>{_esc(HONESTY_BOLD)}</b></div>'
              '<div class="note">사전 등록 문서(VALIDATION.md)에 따라 킬 규칙이 발동하면 톤·비중 제안은 숨기고 정보 제공 전용으로 전환합니다.</div></section>')
    links = ('<section class="panel"><h2>자료</h2><ul class="plain">'
             '<li><a href="backtest_v0.html">v0 백테스트 리포트 (정직한 성적표 · 두 변형)</a></li>'
             + ('<li><a href="calibration_p2.html">Phase 2 보정 리포트 (실험 #2 · 사다리 · 블록 · §6 판정)</a></li>'
                '<li><a href="backtest_v1.html">결정층 v1 백테스트 (v0 와 나란히)</a></li>' if p2sec else "")
             + f'<li><a href="{_esc(SITE_URL)}">{_esc(SITE_URL)}</a></li></ul>'
             f'<div class="foot">생성 {_esc(updated)} · 이 페이지는 투자 조언이 아닙니다.</div></section>')
    _write_html(out_html, "오늘 판정 — market-risk-lab", head + vsec + diff_note + p2sec + wsec + lsec + honest + links)


# ==================================================================
# Phase 2 — 보정 확률 카드 · 보정 리포트 · 결정층 v1 리포트 · 차트 (ARCHITECTURE_PHASE2.md §11)
# ==================================================================
# 설계 원칙(Phase 1 과 동일): 어떤 키가 있든 있는 것만 렌더링한다. 없는 값은 "—", 비어 있으면 '자료 없음'.
# 확률은 자연빈도("100일 중 약 N일")로, 겹치는 라벨의 표본 수는 항상 n_blocks(=n/20)를 병기한다(§16 6).
# 차트 안 글자는 ASCII 만. 한글은 HTML 캡션에.
P2_STATE_COLORS = {"normal": "#22c55e", "caution": "#eab308", "reduce": "#ef4444"}
P2_STATE_KO = {"normal": "정상", "caution": "주의", "reduce": "축소"}
FEATURE_KO = {"x_vix": "VIX", "x_har": "실현-내재 갭", "x_ma": "추세", "refit": "재적합", "intercept": "절편"}
FEATURE_COLORS = {"x_vix": "#60a5fa", "x_har": "#f472b6", "x_ma": "#34d399", "refit": "#8a94a8", "intercept": "#5c6a82"}
RUNG_KO = {"M0": "VIX 공식(보정 전)", "BGK": "VIX 공식(일별 관측 보정)", "B1": "VIX 공식(B1)", "DRIFTLESS": "VIX 공식(무드리프트)",
           "M1": "VIX 보정만(M1)", "M2": "+실현변동성(M2)", "M3": "+추세(M3)", "clim": "기후학(기저율)", "CLIM": "기후학(기저율)"}
INFO_ONLY_LABEL = "시험 운용 — 비중 제안 아님"                      # VALIDATION §7 문구
INFO_DISPLAY_LABEL = "정보 표시(배포 안 함)"                        # 배포되지 않은 단(사다리·비교층)에 붙이는 라벨
DEPLOYED_LABEL = "배포"                                            # acceptance 가 배치한 단 — 상태·톤·비중이 여기서 나온다
RUNG_PARAMS = {"M0": 0, "M1": 2, "M2": 3, "M3": 4}                 # 적합 파라미터 수(절편 포함)
P2_PROB_UNAVAILABLE = "확률 계산 불가"
P2_CHURN_LABEL = "결정층 잦은 전환"
P2_FOOTNOTE = ("AUC 0.63~0.70: 위험 확률은 예측 가능하지만 방향은 아니다 · VIX 대비 skill 의 대부분은 편향 보정 · "
               "확률은 매일 참고, 배분은 상태 기계로만")
P2_FAMILY_LINE = "보정된 VIX 에 조금 더"                            # §16 2 가족용 문구
# §16 정직 문구·리스크 (리포트 ⑨ · 카드 정직 스트립에 상시)
P2_HONESTY_ITEMS = [
    "§6 문자 그대로의 규칙은 설계 단계 관측에서 모든 후보가 실패(M3: 기후학 대비 2019-20 −0.038, 2023-24 −0.000; B1 대비 2007-08 −0.082, "
    "2017-18 −0.029; 보정 VIX 단독도 2019-20 실패). 블록당 ~25개 독립 창에서 BSS 의 표준편차는 ~0.1 이라 '모든 블록 > 0' 은 "
    "참 skill +0.05~0.09 에 대해 검정력이 낮다 — 그래도 규칙은 그대로 채점하고, 완화는 #2a 로 사후 표시한다.",
    "VIX 대비 skill 의 대부분은 편향 보정(M1 +0.19 of M3 +0.20); HAR·MA 의 정보 이득 +13.6×1e-4 (CI −2.8..31.4) — 사다리를 숨기지 않는다. "
    "가족용 문구는 \"보정된 VIX 에 조금 더\".",
    "구조적 실패 모드: 저변동 상태에서의 급락(2018-02, 2020-02)은 과소, 급락 뒤 반등기(2020-04~05, 2003)는 과대. "
    "자연빈도+기저율 표시가 완화책이지 해결책이 아니다.",
    "홀드아웃(~24 독립 창, 2025-04 관세 급락 포함)은 큰 실패만 기각할 수 있고 +1~2% 우위를 인증할 수 없다. 진짜 시험은 Phase 3 라이브 장부(킬룰 §7).",
    "데이터: 1993~95 SPY OHLC 는 시가==고/저가 30%·범위 절반(자기점검·HAR96 민감도); Yahoo 조정 종가 재다운로드는 O/H/L/C 비율·x_ma 를 바꾸지 않지만 "
    "어제 저장값과 오늘 재계산값의 |Δp| > 0.01 이면 경고; ^VIX 휴장일 유령 행.",
    "겹치는 라벨: 모든 구간·검정·신뢰도 표본 수는 블록 방법 또는 n/20 — 5,453 을 '관측치'로 읽는 검토자를 막기 위해 표마다 n_blocks 병기.",
    "'reduce' 는 VIX>50 국면 밖에서 거의 켜지지 않는다 → 사실상 2단계임을 표시.",
    "설계 단계에서 결정층 구성 4종을 시뮬레이션했다 — 채택 구성은 지금 동결하고 같은 자료로 재조정하지 않는다.",
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
    "bss_clim": "BSS vs 기후학", "bss_vix": "BSS vs VIX(B1)", "bss_vix_bgk": "BSS vs BGK", "bss_m1": "BSS vs M1", "bss": "BSS(단)",
    "brier": "Brier", "brier_clim": "Brier 기후학", "brier_vix": "Brier VIX(B1)", "brier_vix_bgk": "Brier BGK", "brier_m1": "Brier M1",
    "brier_from": "Brier(기준)", "brier_to": "Brier(모델)", "auc": "AUC", "calib_in_large": "평균 p − 기저율", "mean_p": "평균 p",
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


def _p2_state_pill(s, grey: bool = False) -> str:
    s = "" if s is None else str(s)
    c = PALETTE["mut"] if grey else P2_STATE_COLORS.get(s, PALETTE["mut"])
    return f'<span class="st" style="background:{c}">{_esc(s)} {_esc(P2_STATE_KO.get(s, ""))}</span>'


def _fmt_p2(v, key: str = "") -> str:
    """P2 표 값 포맷: Brier/MSE 4자리, BSS/AUC 3자리, 손실차(diff_*) ×1e-4, 계수 ±3자리, 확률·비율 백분율, 상태 pill."""
    k = str(key or "").lower()
    if _is_nan(v):
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return "예" if v else "아니오"
    if isinstance(v, str):
        if k in ("state", "p2_state", "last_state", "escalate_to", "deescalate_to") and v in P2_STATE_COLORS:
            return _p2_state_pill(v)
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


def _v0_line_html(v0h: dict, prefix: str = "v0 (동결 벤치마크") -> str:
    """v0 결과 영구 표기 한 줄. 숫자가 없으면 '—' 로 남기되 줄 자체는 항상 있다."""
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
    return (f'<div class="v0line"><b>{_esc(prefix)}{_esc(span)}</b>: ' + " · ".join(_esc(x) for x in parts) + _esc(dd_txt)
            + ' <span class="note" style="display:inline">— VALIDATION.md §4: v0 결과는 모든 Phase 2 리포트 첫 줄에 영구 표기</span></div>')


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


def _dod_line(dod: dict, label: str) -> str:
    """일간(또는 5일 누적) 변화 문장: '어제 대비 +1.3pp: VIX +0.9 · 실현변동성 +0.6 · 추세 −0.2 (· 재적합 0.0)' — 표시된 항의 합 == 표시된 합계."""
    if not isinstance(dod, dict):
        return ""
    d_pp = dod.get("d_pp") if isinstance(dod.get("d_pp"), dict) else {}
    d_p = _fnum(_first(dod, "d_p"))
    if not d_pp and math.isnan(d_p):
        return ""
    parts, tot = _exact_pp_parts(d_pp, d_p)
    if math.isnan(tot):
        return f'<div class="note">{_esc(label)}: 변화 계산 불가(입력 결측)</div>'
    order = ["x_vix", "x_har", "x_ma", "refit"]
    keys = [k for k in order if k in parts] + [k for k in parts if k not in order]
    seg = []
    for k in keys:
        v = parts[k]
        txt = "—" if math.isnan(v) else f"{v:+.1f}"
        nm = FEATURE_KO.get(k, k)
        seg.append(f"({nm} {txt})" if k == "refit" else f"{nm} {txt}")
    gap = dod.get("gap_sessions")
    gap_txt = f" · {int(gap)}세션 전 대비" if isinstance(gap, (int, np.integer)) and not isinstance(gap, bool) and int(gap) > 1 else ""
    return (f'<div class="dod"><b>{_esc(label)} {tot:+.1f}pp</b>: ' + _esc(" · ".join(seg)) + _esc(gap_txt)
            + (' <span class="tag warn">재적합 반영</span>' if dod.get("refit") else "") + "</div>")


def _bars_html(rows: list[dict]) -> str:
    """수준 귀속 3막대(VIX / 실현-내재 갭 / 추세): pp 비례 가로 막대 + logit·pp 숫자."""
    feats = [r for r in rows if r["term"] != "intercept"]
    if not feats:
        return '<div class="note">귀속 자료 없음</div>'
    mx = max((abs(r["pp"]) for r in feats if not math.isnan(r["pp"])), default=0.0)
    html = []
    for r in feats:
        pp, lg = r["pp"], r["logit"]
        w = 0.0 if (mx <= 0 or math.isnan(pp)) else min(100.0, abs(pp) / mx * 100.0)
        color = FEATURE_COLORS.get(r["term"], PALETTE["mut"])
        pp_txt = "—" if math.isnan(pp) else f"{pp * 100:+.1f}pp"
        lg_txt = "—" if math.isnan(lg) else f"logit {lg:+.2f}"
        html.append(f'<div class="p2bar"><div class="lb">{_esc(FEATURE_KO.get(r["term"], r["term"]))}</div>'
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
        warn = (f'<div class="tag warn">이벤트 표 잔여 {_fmt(hz.get("days_left"), "days")}일 &lt; {_fmt(hz.get("warn_days", 60), "days")}일 — '
                f'FOMC 표 갱신 필요(마지막 {_esc(hz.get("last_fomc") or "?")})</div>')
    attrs = ev.attrs if isinstance(ev, pd.DataFrame) else {}
    notes = list(attrs.get("warnings", []) or []) + list(d.get("events_warnings") or [])
    cpi = attrs.get("cpi_registered") if attrs else d.get("cpi_registered")
    if cpi is False and "CPI 일정 미등록" not in notes:
        notes.append("CPI 일정 미등록")
    body = _p2_table(recs, ["date", "kind", "sessions_ahead", "label_ko", "tentative"], "다음 20거래일 안에 등록된 이벤트 없음")
    return body + warn + (f'<div class="note">{_esc(" · ".join(str(n) for n in notes))}</div>' if notes else "")


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


def _table_ko_p2(name) -> str:
    return TABLE_KO_P2.get(str(name), f"{name} 블록")


def acceptance_verdict_line(acc: dict) -> str:
    """배치 판정 한 줄 — calibrate.acceptance 의 verdict_line 을 **그대로** 쓴다(리포트가 판정을 다시 쓰지 않는다).
    예: "24개월 표: M1 통과 · 18개월 표: 실패(최소 블록 BSS_clim −0.0996) → 두 표 모두 통과 요구(소유자 결정 2026-09-08)
         → 배치 없음(정보 제공 전용)". verdict_line 이 없는 예전 산출물이면 rationale 로 내려간다."""
    if not isinstance(acc, dict) or not acc:
        return ""
    txt = acc.get("verdict_line") or acc.get("rationale")
    return str(txt) if txt else ""


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
        tags.append(f'<span class="tag{"" if tm else " warn"}">{_esc(_table_ko_p2(t))} 표 '
                    f'<b>{_esc(str(tm) + " 통과" if tm else "실패")}</b></span>')
    if len(req) > 1:
        tags.append(f'<span class="tag warn">{_esc(f"{len(req)}개 표 모두 통과 요구")}</span>')
    return tags


def _honesty_strip(d: dict, model: dict, lo: float, hi: float, src: str) -> str:
    """정직 스트립: 배포 모델의 model_id·계수·clim·밴드, 생산 모델 M3(정보), #2 §6 판정(literal/amended), 홀드아웃,
    Phase 3 킬룰 카운트다운, 라이브 Brier, 각주."""
    dep = deployment_of(d)
    clim = _fnum(_first(d, "clim", "p2_clim", default=model.get("clim")))
    who = f"{dep['prob_rung']} {dep['label']}"
    items = [f"{who} · model_id {model.get('model_id') or d.get('model_id') or '—'} "
             f"(재적합 {model.get('refit_date') or '—'} · spec {str(model.get('spec_sha256') or '')[:8] or '—'})",
             f"계수 {_coef_txt(model)} · clim {_pct1(clim)} · 구간 {_pct1(lo)}~{_pct1(hi)} ({src or '—'})"]
    prod = d.get("prod_model") if isinstance(d.get("prod_model"), dict) else {}
    if prod and prod.get("model_id") != model.get("model_id"):
        items.append(f"생산 모델 M3 ({INFO_DISPLAY_LABEL}) · model_id {prod.get('model_id') or '—'} · 계수 {_coef_txt(prod)} — "
                     "상태·톤·비중은 이 모델에서 나오지 않습니다")
    # 실험 #2 §6 판정
    acc = d.get("acceptance") if isinstance(d.get("acceptance"), dict) else {}
    if acc:
        def _pass_of(block, rung):
            if not isinstance(block, dict):
                return None
            if rung in block and isinstance(block[rung], dict):
                return block[rung].get("pass")
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
        lit_blk = acc.get("literal", {}).get(cand) if isinstance(acc.get("literal"), dict) else None
        fb = lit_blk.get("failing_blocks") if isinstance(lit_blk, dict) else None
        prim = str(acc.get("primary_table") or "24")
        items.append(f"실험 #2 §6 판정({_table_ko_p2(prim)} 표) — literal({cand}): {lit_txt}" + (f" (실패 블록 {fb})" if fb else "")
                     + f" · amended(#2a, post hoc, {amd_rung}): {'통과' if amd else ('실패' if amd is not None else '—')}"
                     + (f" (사전 후보 {cand} 는 실패)" if amd_rung != cand and _pass_of(amd_blk, cand) is False else "")
                     + (f", 이 표만 보면 톤 모델 후보 {amd_tone}" if amd_tone else "")
                     + f" → 적용 규칙 {acc.get('rule') or '—'}")
        verdict = acceptance_verdict_line(acc)
        deploy_txt = acc.get("deploy_mode") or d.get("deploy_mode") or "—"
        req_txt = "+".join(str(t) for t in (acc.get("require_tables") or []))
        head_txt = "배치 판정" + (f"(요구 표 {req_txt})" if req_txt else "")
        items.append(f"{head_txt} — {verdict}" if verdict else f"{head_txt} — deploy {deploy_txt}")
        items.append(f"deploy {deploy_txt}"
                     + (f" · tone_model {acc.get('tone_model')}" if acc.get("tone_model") else " · 배포된 단 없음 — 톤·비중 주장 없음")
                     + (f" · {acc.get('conjunction_note')}" if acc.get("conjunction_note") else ""))
        sens_all = acc.get("sensitivity_blocks") if isinstance(acc.get("sensitivity_blocks"), dict) else {}
        for t, sens in sens_all.items():
            if isinstance(sens, dict) and not sens.get("agrees", True):
                items.append(f"블록 민감도(§8.2): 같은 규칙을 {_table_ko_p2(t)} 블록 표로 재채점하면 "
                             f"deploy {sens.get('deploy_mode') or '—'} · 톤 모델 {sens.get('tone_model') or '없음'}"
                             " — 배치 판정은 블록 정의에 의존한다(이 표는 배치 판정에 쓰지 않는다)")
    else:
        items.append(f"실험 #2 §6 판정 — deploy {d.get('deploy_mode') or model.get('deploy_mode') or '—'} (summary_p2.json:acceptance 참조)")
    ho = d.get("holdout")
    if isinstance(ho, dict) and ho:
        ci = ho.get("ci") or ho.get("ci_bss_clim")
        n_ho = _fnum(ho.get("n"))
        nb = ho.get("n_blocks", int(n_ho // 20) if not math.isnan(n_ho) else None)
        items.append(f"홀드아웃(2024-09-03~, 1회): n={_fmt(ho.get('n'), 'n')} (창 {_fmt(nb, 'n')}) · "
                     f"BSS vs 기후학 {_fmt_p2(ho.get('bss_clim'), 'bss')} · vs B1 {_fmt_p2(ho.get('bss_vix'), 'bss')} · vs M1 {_fmt_p2(ho.get('bss_m1'), 'bss')}"
                     + (f" · 95% {_fmt_p2(ci, 'bss')}" if isinstance(ci, (list, tuple)) else ""))
    else:
        items.append("홀드아웃(2024-09-01~): 미해제 — 입력 프레임 하드컷, 최종 검증 1회 후 여기 표기")
    live = d.get("live") if isinstance(d.get("live"), dict) else {}
    kr = live.get("kill_rule") if isinstance(live.get("kill_rule"), dict) else {}
    ep = live.get("episodes5_observed")
    mo = _fnum(live.get("months_elapsed"))
    items.append(f"Phase 3 킬룰 카운트다운: 실현 ≥5% 에피소드 {_fmt(ep, 'n') if ep is not None else '—'}/{kr.get('episodes_required', P2_KILL_EPISODES)} · "
                 f"경과 {(f'{mo:.1f}' if not math.isnan(mo) else '—')}/{kr.get('months_required', P2_KILL_MONTHS)}개월"
                 + (" · 평가 시점 도달" if live.get("kill_rule_due") else ""))
    n_sc = live.get("n_scored")
    srcs = live.get("prob_sources") if isinstance(live.get("prob_sources"), dict) else {}
    if n_sc:
        ci = live.get("ci_bss_clim")
        rung = dep["prob_rung"]
        refs = [f"vs M1 {_fmt_p2(live.get('bss_m1'), 'bss')}", f"vs B1 {_fmt_p2(live.get('bss_vix'), 'bss')}"]
        if live.get("bss_m3") is not None:
            refs.append(f"vs M3 {_fmt_p2(live.get('bss_m3'), 'bss')}")
        # '정의상 0' 은 오늘의 배포 단 이름이 아니라 **장부가 실제로 확인한 동일성**(_p2_block: np.allclose(ref, p))에만 붙인다.
        # 채점된 행이 다른 단에서 나왔으면 vs {rung} 은 0 이 아니고, 그 문장은 독자를 속인다.
        ident = (len(srcs) <= 1) and any(f"{rung.lower()} 대비 skill 은 정의상 0" in str(n) for n in (live.get("notes") or []))
        items.append(f"장부 라이브 Brier({who}): 채점 {_fmt(n_sc, 'n')}행 (창 {_fmt(live.get('n_blocks'), 'n')}) · "
                     f"Brier {_fmt_p2(live.get('brier'), 'brier')} · skill vs 기후학 {_fmt_p2(live.get('bss_clim'), 'bss')}"
                     + (f" (95% {_fmt_p2(ci, 'bss')})" if isinstance(ci, (list, tuple)) else "")
                     + " · " + " · ".join(refs)
                     + (f" — 배포 단이 {rung} 이라 vs {rung} 은 정의상 0" if ident else ""))
    elif live.get("holdout_locked"):
        items.append(f"장부 라이브 Brier: 홀드아웃 미해제 → 라이브 채점 보류({HOLDOUT_START} 이후 행은 최종 검증 1회 전까지 채점하지 않는다 — VALIDATION.md §6)")
    else:
        items.append("장부 라이브 Brier: 아직 채점된 행 없음(y_dd5_20 은 20거래일 뒤 확정)")
    if len(srcs) > 1:
        items.append(f"주의 — 이 라이브 계열의 prob_dd5_20 은 여러 단에서 나왔습니다 {srcs} — "
                     "한 모델의 성적표가 아닙니다(배포 단이 바뀐 경계 이전·이후를 한 계열로 읽지 마세요)")
    return ('<div class="strip"><b>정직 스트립</b><ul class="plain">' + "".join(f"<li>{_esc(x)}</li>" for x in items)
            + f'</ul><div class="note">{_esc(P2_FOOTNOTE)}</div></div>')


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

    tags = [f'<span class="tag">모델 <b>{_esc(model.get("model_id") or d.get("model_id") or "—")}</b></span>',
            f'<span class="tag{"" if tones_on else " warn"}">{_esc(f"톤 적용 — {rung} {DEPLOYED_LABEL}" if tones_on else INFO_ONLY_LABEL)}</span>']
    if asof:
        tags.insert(0, f'<span class="tag">기준일 <b>{_esc(pd.Timestamp(asof).strftime("%Y-%m-%d"))}</b></span>')
    if d.get("churn_alert"):
        tags.append(f'<span class="tag bad">{_esc(P2_CHURN_LABEL)} (직전 252세션 {_fmt(d.get("churn_252"), "n")}회)</span>')
    if tones_on:
        eyebrow = f"Phase 2 · 배포 확률 = {_rung_txt(rung)} · 실험 #2"
        if rung != "M3":
            eyebrow += f" · M3 는 {INFO_DISPLAY_LABEL}"
    else:
        eyebrow = f"Phase 2 · 생산 모델 {_rung_txt(rung)} — {INFO_DISPLAY_LABEL} · 실험 #2"
    head = (f'<div class="eyebrow">{_esc(eyebrow)}</div>'
            '<h2>다음 20거래일 안에 -5% 하락할 확률</h2>'
            f'<div class="tags">{"".join(tags)}</div>')

    # 1. 헤드라인(자연빈도) — 입력 결측이면 '확률 계산 불가'
    if missing:
        s1 = (f'<div class="verdict" style="color:#eab308">{_esc(P2_PROB_UNAVAILABLE)}</div>'
              f'<div class="note">{_esc(missing)} — 상태는 유지되고(체류 일수는 계속 셈) 확률·귀속은 비워 둡니다(조용한 실패 금지).</div>')
    else:
        interval_txt = ""
        if not (math.isnan(lo) or math.isnan(hi)):
            both = []
            if not (math.isnan(pb[0]) or math.isnan(pb[1])):
                both.append(f"파라미터 밴드 {_pct1(pb[0], 0)}~{_pct1(pb[1], 0)}")
            if not (math.isnan(cb[0]) or math.isnan(cb[1])):
                both.append(f"보정 구간 {_pct1(cb[0], 0)}~{_pct1(cb[1], 0)}")
            interval_txt = (f' <span class="mut">구간 {_pct1(lo, 0)}~{_pct1(hi, 0)}</span>'
                            + (f'<div class="note">{_esc(" · ".join(both))} — 헤드라인은 넓은 쪽({_esc(src)})</div>' if both else ""))
        color = P2_STATE_COLORS.get(state or "", PALETTE["tx"]) if tones_on else PALETTE["tx"]
        s1 = (f'<div class="verdict" style="color:{color}">이런 날 {_esc(_nat_freq(p))}</div>'
              f'<div class="v-act">= 약 {_pct1(p)} ({_esc(_ten_freq(p))}){interval_txt}</div>'
              f'<div class="note">기저율 {_esc(_nat_freq(clim))} · VIX 공식 {_esc(_nat_freq(p_vix))} · VIX 보정 {_esc(_nat_freq(p_m1))}</div>')
        cbin = d.get("calib_bin") if isinstance(d.get("calib_bin"), dict) else {}
        if cbin and not _is_nan(cbin.get("obs")):
            blo, bhi = _fnum(cbin.get("lo")), _fnum(cbin.get("hi"))
            rng = f"{blo * 100:.0f}~{bhi * 100:.0f}%" if not (math.isnan(blo) or math.isnan(bhi)) else str(cbin.get("bin") or "이 구간")
            n_eff = _fnum(cbin.get("n_eff"))
            s1 += (f'<div class="note">모델이 과거에 {_esc(rng)} 라고 말했을 때 100번 중 {int(round(_fnum(cbin.get("obs")) * 100))}번 일어났습니다 '
                   f'(독립 사례 {int(round(n_eff)) if not math.isnan(n_eff) else "—"}개, 95% 구간 {_pct1(cbin.get("wilson_lo"), 0)}~{_pct1(cbin.get("wilson_hi"), 0)})</div>')

    # 2. 사다리 오늘값 — 배포 단에 표시를 달고, 나머지는 정보
    ladder = [(None, "VIX 공식(보정 전)", p_vix), (None, "VIX 공식(일별 관측 보정)", p_bgk), ("M1", "VIX 보정만(M1)", p_m1),
              ("M2", "+실현변동성(M2)", p_m2), ("M3", "+추세(M3)", p_m3)]
    cells = ""
    for rk, nm, v in ladder:
        mark = ""
        if rk and rk == rung:
            mark = f'<div class="note">{_esc(dep["label"])}</div>'
        elif rk:
            mark = f'<div class="note">{_esc(INFO_DISPLAY_LABEL)}</div>'
        cells += f'<div class="tf"><div class="lb">{_esc(nm)}</div><div class="mono">{_pct1(v)}</div>{mark}</div>'
    add = p_m3 - p_m1 if not (math.isnan(p_m3) or math.isnan(p_m1)) else math.nan
    ladder_note = (f'<div class="note">오늘 추가 요인(실현변동성·추세)이 더한 것: {_esc(_pp1(add))} — {_esc(P2_FAMILY_LINE)}'
                   + (f' (M3 는 {_esc(INFO_DISPLAY_LABEL)} — 상태·톤은 {_esc(rung)} 에서 나옵니다)' if tones_on and rung != "M3" else "")
                   + "</div>") if not math.isnan(add) else ""
    s2 = f'<h3>사다리 — 오늘값</h3><div class="tfrow">{cells}</div>{ladder_note}'

    # 3. 상태
    thr = d.get("thresholds") if isinstance(d.get("thresholds"), dict) else (_thresholds_fallback(state, days, clim) if state in P2_STATE_COLORS else {})
    thr_txt = []
    if thr.get("escalate_to"):
        thr_txt.append(f"{thr['escalate_to']} 격상: p ≥ {_pct1(thr.get('p_escalate'))} (r ≥ {_fnum(thr.get('r_escalate')):g})")
    if thr.get("deescalate_to"):
        rem = thr.get("dwell_remaining")
        thr_txt.append(f"{state} 해제까지 p < {_pct1(thr.get('p_deescalate'))} (r < {_fnum(thr.get('r_deescalate')):g})"
                       + (f", 체류 잔여 {int(rem)}세션" if isinstance(rem, (int, np.integer)) and not isinstance(rem, bool) and int(rem) > 0 else ""))
    tone = STATE_TO_TONE.get(state or "", None)
    state_bits = [f"r = {r:.2f} (= {rung} 확률 ÷ 기저율)" if not math.isnan(r) else "r = — (확률 계산 불가)",
                  f"체류 {_fmt(days, 'days')}세션" if days is not None else "체류 —"]
    reason = d.get("reason_ko")
    # 배치 판정 한 줄(교집합) — 카드가 배포 여부를 스스로 말한다. 출처는 summary_p2.json.acceptance 하나뿐.
    acc_line = acceptance_verdict_line(d.get("acceptance") if isinstance(d.get("acceptance"), dict) else {})
    if state is None:
        s3 = '<h3>상태</h3><div class="note">상태 자료 없음</div>'
    elif tones_on:
        exp_ = TONE_EXPOSURE.get(tone, None)
        s3 = (f'<h3>상태 · 톤</h3><div>{_p2_state_pill(state)} {_tone_pill(tone) if tone else ""}'
              + (f' <span class="mut">주식 비중 {int(round(exp_ * 100))}%</span>' if exp_ is not None else "")
              + f'</div><div class="note">{_esc(" · ".join(state_bits))}' + (f" · {_esc(reason)}" if reason else "") + "</div>"
              + (f'<div class="note">{_esc(" · ".join(thr_txt))}</div>' if thr_txt else "")
              + (f'<div class="note">{_esc(acc_line)}</div>' if acc_line else ""))
    else:
        s3 = (f'<h3>상태 (시험 운용)</h3><div class="info">{_p2_state_pill(state, grey=True)} '
              f'<span class="tag warn">{_esc(INFO_ONLY_LABEL)}</span> <span class="tag warn">{_esc(INFO_DISPLAY_LABEL)}</span>'
              f'<div class="note">{_esc(" · ".join(state_bits))}' + (f" · {_esc(reason)}" if reason else "") + "</div>"
              + (f'<div class="note">{_esc(" · ".join(thr_txt))}</div>' if thr_txt else "")
              + f'<div class="note">§6 채택 규칙을 통과하기 전이라 배포된 단이 없습니다 — {_esc(_rung_txt(rung))} 확률과 상태는 '
                '정보로만 보여주고 톤·비중은 주장하지 않습니다(비중은 v0 판정을 따릅니다).</div>'
              + (f'<div class="note">{_esc(acc_line)}</div>' if acc_line else "") + "</div>")
    if d.get("churn_alert"):
        s3 += f'<div class="tag bad">{_esc(P2_CHURN_LABEL)} — 직전 252세션 변경 &gt; {DECISION_P2["churn_alert"]} (장부 §8 검토 대상, 자동 재조정 없음)</div>'

    # 4. 귀속 — 배포 모델 기준(헤드라인 확률을 만든 그 모델)
    rows = _contrib_rows(d)
    b0_p = next((r_["pp"] for r_ in rows if r_["term"] == "intercept"), math.nan)
    n_terms = len([r_ for r_ in rows if r_["term"] != "intercept"])
    s4 = (f"<h3>수준 귀속 — {_esc(rung)} 기준 (절편에서 순차 대입, pp 정확 합산)</h3>"
          + (f'<div class="note">절편만의 확률 σ(b0) = {_pct1(b0_p)} → {n_terms}개 요인을 차례로 더해 {_pct1(p)}</div>' if not math.isnan(b0_p) else "")
          + _bars_html(rows) + _dod_line(d.get("dod"), "어제 대비") + _dod_line(d.get("dod_5d"), "5일 누적"))
    m3_rows = [r_ for r_ in _contrib_rows(d, key="contributions_m3") if r_["term"] != "intercept"]
    if m3_rows:
        s4 += (f'<div class="note">참고 — 생산 모델 M3({_esc(INFO_DISPLAY_LABEL)})의 같은 날 수준 귀속: '
               + " · ".join(f'{_esc(_label(r_["term"]))} {_pp1(r_["pp"])}' for r_ in m3_rows) + "</div>")

    # 5. 요인 문맥 (표시 전용)
    pct = d.get("percentiles") if isinstance(d.get("percentiles"), dict) else {}
    ctx_notes = {"vix": "내재 변동성 수준 — 높을수록 VIX 공식 확률↑", "x_vix": "VIX 내재 확률의 로짓(모델 입력)",
                 "har_vol_20": "GK+OV 실현변동성의 고정가중 HAR 평균", "x_har": "실현 − 내재(log): 음수면 내재가 실현보다 높음(프리미엄)",
                 "x_ma": "종가/180일선 − 1: 음수면 추세 아래", "rv20_cc": "종가 기준 20일 실현변동성(표시 전용)", "ts_diag": "VIX/VIX3M 기간구조(표시 전용)"}
    frows = []
    for k in ("vix", "x_vix", "har_vol_20", "x_har", "x_ma", "rv20_cc", "ts_diag"):
        v = _first(d, k, f"p2_{k}")
        if v is None and k not in pct:
            continue
        frows.append({"name": k, "value": v, "pct": pct.get(k), "note": ctx_notes.get(k, "")})
    ctx = d.get("context") if isinstance(d.get("context"), dict) else {}
    s5 = ("<h3>요인 문맥 (표시 전용 — 10년 이동 백분위, 모델 입력 아님)</h3>"
          + _p2_table(frows, ["name", "value", "pct", "note"], "요인 자료 없음")
          + (f'<div class="note">변동성 문맥: {_esc(" · ".join(f"{k} {_fmt_p2(v, k)}" for k, v in ctx.items() if _scalar_ok(v)))}</div>' if ctx else ""))

    # 6. 보조 A — 변동성
    har = _fnum(_first(d, "har_fc_20", "p2_har_fc_20"))
    vix = _fnum(_first(d, "vix", "vix_close"))
    rv20 = _fnum(_first(d, "rv20_cc"))
    prem = (vix / 100.0 - har) if not (math.isnan(vix) or math.isnan(har)) else math.nan
    mae = _fnum(_first(d, "har_oos_log_mae", "har_log_mae"))
    s6 = ("<h3>보조 A — 예상 변동성 (적합 log-HAR, 확률 모델 밖)</h3>"
          f'<div class="note">예상 변동성(HAR) {_pct1(har)} · VIX {(f"{vix:.1f}%" if not math.isnan(vix) else "—")} · 최근 20일 실현 {_pct1(rv20)}'
          + (f" · 차이(현재 변동성 프리미엄) {_pp1(prem)}" if not math.isnan(prem) else "")
          + (f" · OOS log-MAE {mae:.3f}" if not math.isnan(mae) else "") + "</div>")

    # 7. 보조 B — 이벤트
    s7 = "<h3>보조 B — 다음 20거래일 이벤트 (표시 전용)</h3>" + _events_html(d)

    # 8. 정직 스트립
    s8 = _honesty_strip(d, model, lo, hi, src)
    wsec = ("<h3>P2 경고</h3>" + _warn_list(warns)) if warns else ""
    return f'<section class="panel" id="p2">{head}{s1}{s2}{s3}{s4}{s5}{s6}{s7}{wsec}{s8}</section>'


# ------------------------------------------------------------------
# 주간 리포트 — 보정 (docs/calibration_p2.html)
# ------------------------------------------------------------------
def _acceptance_box(acc: dict) -> str:
    """§6 literal / #2a amended 판정 상자 — **요구 표 전부의 교집합**(장부 #2d)을 표별로 펼쳐 보인다."""
    if not isinstance(acc, dict) or not acc:
        return '<div class="note">acceptance 결과 없음</div>'
    dm = acc.get("deploy_mode") or "—"
    ok = dm == "tones"
    color = "#22c55e" if ok else "#eab308"
    req = [str(t) for t in (acc.get("require_tables") or [])]
    per = acc.get("per_table") if isinstance(acc.get("per_table"), dict) else {}
    head = (f'<div class="acc" style="border-color:{color}"><div class="verdict" style="color:{color};font-size:18px">'
            f'{"톤 적용(tones)" if ok else "정보 제공 전용(info_only)"} — 규칙 {_esc(acc.get("rule") or "—")}'
            + (f' · 판정 표 {_esc(" + ".join(_table_ko_p2(t) for t in req))}' if req else "")
            + (f' · 톤 모델 {_esc(acc.get("tone_model"))}' if acc.get("tone_model") else " · 배포된 단 없음") + "</div>")
    verdict = acceptance_verdict_line(acc)
    if verdict:
        head += f'<div class="note" style="color:{color}"><b>배치 판정</b>: {_esc(verdict)}</div>'
    if req:
        rows = [{"table": _table_ko_p2(t), "required": bool((per.get(t) or {}).get("required", True)),
                 "alone_tone": (per.get(t) or {}).get("tone_model"), "alone_deploy": (per.get(t) or {}).get("deploy_mode"),
                 "passing_rungs": ", ".join((per.get(t) or {}).get("passing_rungs") or []) or "없음",
                 "n_blocks": (per.get(t) or {}).get("n_blocks"), "min_pass_blocks": (per.get(t) or {}).get("min_pass_blocks")}
                for t in req]
        head += ("<h3>표별 판정 (배치는 이 표들의 교집합)</h3>"
                 '<div class="note">아래 두 열은 <b>그 표 하나만 봤을 때</b>의 결론이다 — 실제 배치는 위의 배치 판정 한 줄이다.</div>'
                 + _p2_table(rows))
    if acc.get("blocking_tables"):
        head += ('<div class="note" style="color:#eab308"><b>배치를 막은 표</b>: '
                 + _esc(", ".join(_table_ko_p2(t) for t in acc["blocking_tables"])) + "</div>")
    for key in ("conjunction_note", "rationale", "post_hoc_note"):
        if acc.get(key):
            head += f'<div class="note">{_esc(acc.get(key))}</div>'
    parts = []
    for tname, tbl in _acceptance_tables(acc).items():
        label = f"{_table_ko_p2(tname)} 블록 표" + ("" if tbl.get("required", True) else " (민감도 — 배치 판정에 쓰지 않음)")
        for name, title in (("literal", "§6 문자 그대로 (사전 등록)"), ("amended", "#2a 완화안 (post hoc)")):
            blk = tbl.get(name)
            if isinstance(blk, dict) and blk and all(isinstance(v, dict) for v in blk.values()):
                recs = [{"rung": k, **{kk: vv for kk, vv in v.items() if kk != "rule_text"}} for k, v in blk.items()]
                rule_text = next((v.get("rule_text") for v in blk.values() if isinstance(v, dict) and v.get("rule_text")), None)
                parts.append(f"<h3>{_esc(label)} — {_esc(title)}</h3>"
                             + (f'<div class="note">{_esc(rule_text)}</div>' if rule_text else "") + _p2_table(recs))
            elif isinstance(blk, dict) and blk:
                parts.append(f"<h3>{_esc(label)} — {_esc(title)}</h3>" + _p2_kv({k: v for k, v in blk.items() if k != "rule_text"})
                             + (f'<div class="note">{_esc(blk.get("rule_text"))}</div>' if blk.get("rule_text") else ""))
    sens_all = acc.get("sensitivity_blocks") if isinstance(acc.get("sensitivity_blocks"), dict) else {}
    for t, sens in sens_all.items():
        if isinstance(sens, dict) and not sens.get("agrees", True):
            head += ('<div class="note" style="color:#eab308"><b>블록 민감도(§8.2)</b>: 같은 규칙을 '
                     f'{_esc(_table_ko_p2(t))} 블록 표로 재채점하면 deploy <b>{_esc(str(sens.get("deploy_mode") or "—"))}</b>'
                     f' · 톤 모델 {_esc(str(sens.get("tone_model") or "없음"))} — 배치 판정은 블록 정의에 의존한다'
                     "(이 표는 배치 판정에 쓰지 않는다).</div>")
    extra = {k: v for k, v in acc.items() if k in ("candidate", "primary_table", "literal_tone_model", "amended_tone_model",
                                                   "n_blocks", "min_pass_blocks")}
    return head + "".join(parts) + (_p2_kv(extra) if extra else "") + "</div>"


def render_calibration_report(summary_p2: dict, out_html: Path, charts: dict[str, bytes]) -> None:
    """docs/calibration_p2.html — ① v0 줄(영구) ② 사다리 표 ③ 블록 표 24/18/1999 + §6 판정 ④ 신뢰도·Murphy ⑤ 계수 경로·파라미터 밴드
    ⑥ 시대별 AUC ⑦ 소거·v0 참조선 ⑧ HAR-RV 성적표 ⑨ 데이터 범위·자기점검·경고·spec_sha256·FEATURE_RULE·정직 문구. 없는 키는 '자료 없음'."""
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

    tags = [f'<span class="tag">OOS <b>{_esc(first_refit)} ~ {_esc(end)}</b></span>',
            f'<span class="tag">spec <b class="mono">{_esc(str(sha)[:12])}</b></span>',
            f'<span class="tag">생성 <b>{_esc(run.get("generated_at_utc") or s.get("generated_at") or _now_str())}</b></span>',
            f'<span class="tag{"" if acc.get("deploy_mode") == "tones" else " warn"}">deploy <b>{_esc(acc.get("deploy_mode") or "—")}</b>'
            + (f' · 톤 모델 {_esc(acc.get("tone_model"))}' if acc.get("tone_model") else " · 배포된 단 없음") + "</span>"]
    tags += _acceptance_tags(acc)
    _sens_acc = acc.get("sensitivity_blocks") if isinstance(acc.get("sensitivity_blocks"), dict) else {}
    for _t, _s in _sens_acc.items():
        if isinstance(_s, dict) and not _s.get("agrees", True):
            tags.append(f'<span class="tag warn">{_esc(_table_ko_p2(_t))} 블록 민감도 불일치</span>')
    if holdout:
        tags.append('<span class="tag bad">홀드아웃 해제됨(1회)</span>')
    else:
        tags.append(f'<span class="tag">홀드아웃 <b>{_esc(HOLDOUT_START)}~ 하드컷</b></span>')
    if warns:
        tags.append(f'<span class="tag warn">경고 {len(warns)}건</span>')
    disclosure = s.get("disclosure") or "#2 사전 관측 참조"
    head = ('<header><div class="eyebrow">market-risk-lab · Phase 2 · 실험 #2 (보정 모델 p2)</div>'
            '<h1>보정 확률 walk-forward 채점 — 사다리 M0→M3, 기후학·VIX 대비 Brier skill</h1>'
            '<div class="sub">4-파라미터 로지스틱(x_vix·x_har·x_ma), 1월 첫 거래일 연 1회 재적합, 20거래일 퍼지. '
            'VALIDATION §6 이 블록을 "18~24개월" 로 사전 등록했으므로 배치 판정은 24개월 표와 18개월 표 <b>모두</b>에서 통과해야 합니다'
            '(소유자 결정 2026-09-08, 장부 #2d). 모든 표본 수는 n/20 독립 창을 병기합니다.</div>'
            f'<div class="tags">{"".join(tags)}</div><div class="note">공개: {_esc(disclosure)} — 2003~2024-08 OOS 기록은 설계 단계에서 이미 관측됨(§13 #2). 이후 규칙 변경은 post hoc.</div></header>')
    sec_titles = ["① v0 요약(영구)", "② 사다리", "③ 블록·§6 판정", "④ 신뢰도·Murphy", "⑤ 계수 경로", "⑥ 시대별 AUC", "⑦ 소거·참조선", "⑧ HAR-RV", "⑨ 데이터·경고·정직"]
    nav = '<nav class="nav">' + "".join(f'<a href="#s{i + 1}">{_esc(t)}</a>' for i, t in enumerate(sec_titles)) + "</nav>"

    # ① v0 completed 요약 줄(영구)
    v0ref = s.get("v0_reference") if isinstance(s.get("v0_reference"), dict) else {}
    v0ref_flat = {k: v for k, v in v0ref.items() if _scalar_ok(v)}
    s1 = ('<section class="panel" id="s1"><h2>① v0 completed 요약 (영구 표기)</h2>' + _v0_line_html(v0h)
          + (f'<div class="note">v0 참조선(Platt 2, 모델 밖): {_esc(", ".join(f"{k} {_fmt_p2(v, k)}" for k, v in v0ref_flat.items()))}</div>' if v0ref_flat else "")
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
    s2 = ('<section class="panel" id="s2"><h2>② 사다리 — M0(VIX 공식) → M1(보정) → M2(+실현변동성) → M3(+추세)</h2>'
          '<div class="note">단별 전체 OOS 채점: Brier, BSS vs 기후학·VIX(B1)·BGK·M1, AUC. 그 아래 단 간 손실차 = (p_from−y)² − (p_to−y)² 의 평균(×1e-4, 양수 = 모델이 좋음), '
          '40일 블록 부트스트랩 4,000회 95% 구간, Diebold–Mariano t(HAC lag 19), 20 위상 오프셋 부분표본 BSS 의 min/median/max. '
          f'{_esc(P2_HONESTY_ITEMS[1])}</div>'
          + "<h3>단별 채점(전체)</h3>" + _p2_table(rungs, _RUNG_ORDER, "rungs 없음")
          + "<h3>단 간 검정(전체)</h3>" + _p2_table(lad_all, _LADDER_ORDER, "ladder 없음")
          + _img(charts.pop("ladder_ci", None), "단 간 손실차(×1e-4)와 95% 블록 부트스트랩 구간 — 0 오른쪽이면 위 단이 좋음.", "Ladder loss-diff CI")
          + ("<h3>단 간 검정(블록별)</h3>" + _p2_table(lad_blk, _LADDER_ORDER) if lad_blk else "")
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
    s3 = ('<section class="panel" id="s3"><h2>③ 블록 표 (24개월 · 18개월 · 1999 시작) 와 §6 판정</h2>'
          f'<div class="note">{_esc(P2_HONESTY_ITEMS[0])}</div>'
          + _acceptance_box(acc)
          + _img(charts.pop("block_skill", None), "블록별 Brier skill — 4 벤치마크(기후학·VIX B1·BGK·M1). 0 아래가 실패 블록.", "Block skill bars")
          + f"<h3>24개월 블록({_esc(_b24_role)}, M3 생산 모델)</h3>" + _p2_table(b24, _BLOCK_ORDER, "blocks24 없음")
          + f"<h3>18개월 블록({_esc(_b18_role)}, {_esc(b18_rung)})</h3>" + _p2_table(b18, _BLOCK_ORDER, "blocks18 없음")
          + "<h3>1999 시작(민감도, 2000~02 약세장을 OOS 로 · M3)</h3>" + _p2_table(b99, _BLOCK_ORDER, "blocks_from1999 없음")
          + (("<h3>홀드아웃(2024-09-03~, 1회 검증)</h3>" + _p2_kv(holdout)) if holdout else "")
          + "</section>")

    # ④ 신뢰도·Murphy — 배포 단(acceptance 가 진실)의 표를 싣는다. 다른 단의 표를 대신 쓰지 않는다.
    dep4 = deployment_of(acc)
    rung4 = dep4["prob_rung"]
    rel = _to_records(s.get("reliability" if rung4 == "M3" else f"reliability_{rung4.lower()}"))
    if not rel:
        warns.append(f"신뢰도: summary_p2.json 에 {rung4} 단의 표가 없어 ④ 를 비움(다른 단의 표를 대신 쓰지 않는다)")
    mur_key = "murphy" if rung4 == "M3" else f"murphy_{rung4.lower()}"
    mur = s.get(mur_key) if isinstance(s.get(mur_key), dict) else {}
    s4 = ('<section class="panel" id="s4"><h2>④ 신뢰도 — ' + _esc(rung4) + '(' + _esc(dep4["label"])
          + ') (고정 구간 · Wilson n/20) · Murphy 분해</h2>'
          f'<div class="note">구간 {_esc(", ".join(f"{b:g}" for b in P2["reliability_bins"]))} 고정(사전 등록). n_eff = n/20 으로 Wilson 95% 구간. '
          f'{_esc(P2_HONESTY_ITEMS[2])}</div>'
          + _img(charts.pop("reliability", None), "신뢰도 다이어그램: 구간 평균 확률 vs 실현 비율(Wilson n/20 구간). 대각선 = 완전 보정.", "Reliability diagram")
          + _p2_table(rel, _RELIAB_ORDER, f"{rung4} 신뢰도 표 없음")
          + (f"<h3>Murphy 분해 ({_esc(rung4)})</h3>" + _p2_kv(mur) if mur else "")
          + "</section>")

    # ⑤ 계수 경로·파라미터 밴드
    params = _param_records(s.get("params_by_refit"), "M3") or _param_records(s.get("params_by_refit"), None)
    band = s.get("param_band") if isinstance(s.get("param_band"), (list, tuple)) else None
    live = s.get("live_model") if isinstance(s.get("live_model"), dict) else (s.get("model") if isinstance(s.get("model"), dict) else {})
    s5 = ('<section class="panel" id="s5"><h2>⑤ 계수 경로 (연 1회 재적합, M3 생산 모델) · 파라미터 밴드</h2>'
          '<div class="note">정확히 4개의 적합 파라미터(b0, b1 x_vix, b2 x_har, b3 x_ma). 5번째 슬롯은 비어 있고 장부에 예약(#3~#5). '
          '파라미터 밴드 = 최근 5회 재적합 계수 + 라이브 모델을 오늘 x 에 적용한 min/max.</div>'
          + _img(charts.pop("coef_path", None), "재적합별 계수 경로(M3).", "Coefficient path")
          + _p2_table(params, _PARAM_ORDER, "params_by_refit 없음")
          + (f'<div class="note">라이브 파라미터 밴드: {_fmt_p2(list(band), "p")}</div>' if band is not None else "")
          + (("<h3>라이브 모델(model_p2.json)</h3>" + _p2_kv({k: v for k, v in live.items() if k != "extra"})) if live else "")
          + "</section>")

    # ⑥ 시대별 AUC
    era = _to_records(s.get("era_auc"))
    s6 = ('<section class="panel" id="s6"><h2>⑥ 시대별 AUC — VIX 판별력 감쇠(0.72→0.57)를 상시 표시</h2>'
          '<div class="note">원시 특징(x_vix, x_har, −x_ma)은 전 이력, p_m1·p_m3 는 OOS 행만. AUC 는 순위만 보므로 보정과 무관.</div>'
          + _img(charts.pop("era_auc", None), "시대별 AUC (특징·모델).", "Era AUC")
          + _p2_table(era, _ERA_ORDER, "era_auc 없음") + "</section>")

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
    s7 = ('<section class="panel" id="s7"><h2>⑦ 소거 실험 (M3-PK · M3-HAR96 · C 민감도) · v0 참조선</h2>'
          '<div class="note">소거는 이름 붙인 실험이며 선택에 쓰지 않는다. v0 참조선 = v0 종합점수에 Platt(2), 2017-01-03 재적합부터 — 예산 밖·모델 밖(주간 페이지 참조선 전용).</div>'
          + "<h3>소거</h3>" + _p2_table(abl_recs, None, "ablations 없음")
          + ("<h3>C 민감도</h3>" + _p2_table(csens_recs) if csens_recs else "")
          + "<h3>v0 참조선</h3>" + (_p2_kv({k: v for k, v in v0ref.items() if k not in ("params",)}) if v0ref else '<div class="note">v0 참조선 없음(창 규칙·해시 불일치 시 생략)</div>')
          + "</section>")

    # ⑧ HAR-RV
    har = s.get("har") if isinstance(s.get("har"), dict) else {}
    har_sc = _to_records(har.get("scorecard"))
    har_sens = _to_records(har.get("scorecard_sensitivity") or har.get("sensitivity"))
    har_coef = har.get("live_coef") if isinstance(har.get("live_coef"), dict) else {}
    har_path = _to_records(har.get("coef_path") or har.get("params_by_refit"))
    s8 = ('<section class="panel" id="s8"><h2>⑧ HAR-RV 보조 출력 성적표 (적합 log-HAR, OLS 4 — 예산 밖, 확률에 결합 금지)</h2>'
          '<div class="note">ln RV20_fwd ~ 1 + ln rv1 + ln rv5 + ln rv22, 연 1회 재적합·20일 퍼지. 같은 지표를 VIX(σ̂=VIX/100)·RV22 지속에 대해 병기. 학습 시작 1996 민감도.</div>'
          + _img(charts.pop("har", None), "HAR 예측 vs 실현 20일 변동성.", "HAR forecast")
          + "<h3>성적표</h3>" + _p2_table(har_sc, _HAR_ORDER, "har.scorecard 없음")
          + ("<h3>민감도(학습 시작 1996-01-02)</h3>" + _p2_table(har_sens, _HAR_ORDER) if har_sens else "")
          + ("<h3>라이브 계수</h3>" + _p2_kv(har_coef) if har_coef else "")
          + ("<h3>재적합별 계수</h3>" + _p2_table(har_path) if har_path else "")
          + (f'<div class="note">{_esc(" · ".join(str(w) for w in har.get("warnings", [])))}</div>' if har.get("warnings") else "")
          + "</section>")

    # ⑨ 데이터 범위·자기점검·경고·spec·정직
    selftest = s.get("selftest") if isinstance(s.get("selftest"), dict) else {}
    cache = run.get("cache") if isinstance(run.get("cache"), dict) else {}
    run_flat = {k: v for k, v in run.items() if _scalar_ok(v)}
    s9 = ('<section class="panel" id="s9"><h2>⑨ 데이터 범위 · 자기점검 · 경고 · 정직 문구</h2>'
          "<h3>실행</h3>" + _p2_kv(run_flat, "run 없음")
          + ("<h3>캐시</h3>" + _p2_kv(cache) if cache else "")
          + "<h3>자기점검 (GK+OV/CC · 1993~95 품질)</h3>" + (_p2_kv(selftest) if selftest else '<div class="note">selftest 없음</div>')
          + f'<div class="note mono">spec_sha256 {_esc(sha)}<br>FEATURE_RULE {_esc(rule)}</div>'
          + "<h3>경고</h3>" + _warn_list(warns)
          + "<h3>정직 문구와 리스크 (§16)</h3><ol class=\"method\">" + "".join(f"<li>{_esc(t)}</li>" for t in P2_HONESTY_ITEMS) + "</ol>"
          + f'<div class="note">{_esc(P2_FOOTNOTE)}</div>')
    for k, png in list(charts.items()):
        s9 += _img(png, f"추가 차트: {k}", k)
    s9 += (f'<div class="foot">market-risk-lab · 실험 #2 · 생성 {_esc(run.get("generated_at_utc") or _now_str())} · '
           '<a href="index.html">오늘 판정으로</a> · <a href="backtest_v1.html">결정층 v1</a> · <a href="backtest_v0.html">v0 백테스트</a></div></section>')
    _write_html(out_html, "보정 모델 p2 — 실험 #2 — market-risk-lab", head + nav + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9)


# ------------------------------------------------------------------
# 주간 리포트 — 결정층 v1 (docs/backtest_v1.html)
# ------------------------------------------------------------------
def _alloc_matrix(v1_alloc: dict, v0_alloc: dict) -> str:
    """배분 시뮬 비교표: v1 / v0 / 보유 (열) × 지표(행)."""
    a1 = v1_alloc if isinstance(v1_alloc, dict) else {}
    a0 = v0_alloc if isinstance(v0_alloc, dict) else {}
    if not a1 and not a0:
        return '<div class="note">allocation 결과 없음</div>'
    bh_src = a1 if a1 else a0
    bh = {k: bh_src.get(f"bh_{k}") for k in ("cagr", "max_dd", "worst_month", "total_return", "ann_vol")}
    bh.update({"n_switches": 0, "switches_per_year": 0.0, "avg_exposure": 1.0, "cost_total": 0.0,
               "n_days": bh_src.get("n_days"), "years": bh_src.get("years"), "start": bh_src.get("start"), "end": bh_src.get("end")})
    cols = {"v1 결정층": {k: a1.get(k) for k in _ALLOC_KEYS}, "v0 톤": {k: a0.get(k) for k in _ALLOC_KEYS}, "보유(B&H)": bh}
    return _p2_kv(cols)


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

    tags = [f'<span class="tag">구간 <b>{_esc(_fmt(kp.get("start") or run.get("start"), "start"))} ~ {_esc(_fmt(kp.get("end") or run.get("end"), "end"))}</b></span>',
            f'<span class="tag">세션 <b>{_fmt(kp.get("n_sessions"), "n")}</b></span>',
            f'<span class="tag">구성 <b>{_esc(s.get("config_name") or run.get("config") or "default")}</b></span>',
            f'<span class="tag{"" if deployed else " warn"}">deploy <b>{_esc(deploy)}</b>{(" · " + _esc(tone_model)) if tone_model else ""}</span>',
            f'<span class="tag{"" if deployed else " warn"}">헤드라인 <b>{_esc(rung)}</b> {_esc(dep_label)}</span>',
            f'<span class="tag">생성 <b>{_esc(run.get("generated_at_utc") or s.get("generated_at") or _now_str())}</b></span>']
    for f in flags:
        tags.append(f'<span class="tag bad">{_esc(f)}</span>')
    if warns:
        tags.append(f'<span class="tag warn">경고 {len(warns)}건</span>')
    head = ('<header><div class="eyebrow">market-risk-lab · Phase 2 · 결정층 v1 (3단계 상태기계)</div>'
            '<h1>결정층 v1 백테스트 — r = p/clim, 격상 즉시 · 격하 5세션 체류</h1>'
            f'<div class="sub">밴드 caution {DECISION_P2["enter_caution"]}/{DECISION_P2["exit_caution"]}, reduce {DECISION_P2["enter_reduce"]}/{DECISION_P2["exit_reduce"]}, '
            f'dwell {DECISION_P2["dwell"]}, KPI 상한 {DECISION_P2["kpi_max_switches_per_year"]}회/년, churn 경보 252세션 &gt; {DECISION_P2["churn_alert"]}. '
            '상태→톤→비중(100/50/25%)·비용 5bp 로 v0 와 같은 성적표.</div>'
            f'<div class="sub">아래 ②~⑤ 의 모든 숫자는 <b>{_esc(_rung_txt(rung))} 확률로 돌린 결정층</b>'
            + (f' — summary_p2.json.acceptance 가 배치한 단({_esc(dep_label)})입니다.'
               if deployed else
               f' — 배치된 단이 없어(deploy {_esc(str(deploy))}) 생산 모델을 {_esc(INFO_DISPLAY_LABEL)} 로 싣습니다: 톤·비중 주장이 아닙니다.')
            + (f' 배포되지 않은 단({_esc(", ".join(info_names))})은 ⑥ 에 정보로만 둡니다.' if info_names else "")
            + "</div>"
            f'<div class="tags">{"".join(tags)}</div></header>')
    sec_titles = ["① v0 요약(영구)", "② 결정층 KPI vs v0", "③ 에피소드", "④ 배분 v1·v0·보유", "⑤ 민감도",
                  "⑥ 비배포 단(정보)", "⑦ 차트", "⑧ 정직·경고"]
    nav = '<nav class="nav">' + "".join(f'<a href="#s{i + 1}">{_esc(t)}</a>' for i, t in enumerate(sec_titles)) + "</nav>"

    s1 = '<section class="panel" id="s1"><h2>① v0 completed 요약 (영구 표기)</h2>' + _v0_line_html(v0h) + "</section>"

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
    kpis = [
        _kpi("상태 전환 / 년", _fmt_p2(kp.get("switches_per_year"), "switches_per_year"),
             f"v0 {_fmt_p2(v0h.get('tone_switches_per_year'), 'switches_per_year')} · KPI 상한 {kp.get('kpi_ceiling', DECISION_P2['kpi_max_switches_per_year'])} "
             f"{'충족' if ceiling_ok else ('위반' if ceiling_ok is False else '—')}"),
        _kpi("경고 점유(caution+reduce)", _pct1(kp.get("warn_share")), f"v0 경고 비중 {_pct1(v0h.get('warn_share'))}"),
        _kpi("경고 상태 -5%/20일 비율", (f"{min(warn_dd) * 100:.1f}~{max(warn_dd) * 100:.1f}%" if warn_dd else "—"),
             f"정상 {_pct1(dd_state.get('normal'))} · 기저율 {_pct1(kp.get('base_rate'))} · v0 경고 "
             f"{(f'{min(v0_warn_dd) * 100:.0f}~{max(v0_warn_dd) * 100:.0f}%' if v0_warn_dd else '—')} vs 상승 {(f'{min(v0_up_dd) * 100:.0f}~{max(v0_up_dd) * 100:.0f}%' if v0_up_dd else '—')}"),
        _kpi("진짜 경보 비중(경고 런 시작일)", _pct1(kp.get("true_alarm_share")),
             f"기저율 {_pct1(kp.get('true_alarm_share_baseline'))} · v0 {v0_ta_txt}"),
        _kpi("5세션 최대 변경", _fmt(kp.get("max_changes_any_5_sessions"), "n"),
             f"구조적 상한 {kp.get('structural_bound') if kp.get('structural_bound') is not None else '—'} {'충족' if kp.get('structural_bound_ok') else ''}"),
        _kpi("churn 경보 세션", _fmt(kp.get("churn_alert_sessions"), "n"), f"최대 252세션 변경 {_fmt(kp.get('max_churn_252'), 'n')}"),
        _kpi("배분 CAGR", _pct1(alloc.get("cagr")), f"v0 {_pct1(v0h.get('cagr_strategy'))} · 보유 {_pct1(alloc.get('bh_cagr') if alloc.get('bh_cagr') is not None else v0h.get('cagr_bh'))}"),
        _kpi("배분 MaxDD", _pct1(alloc.get("max_dd")), f"v0 {_pct1(v0h.get('maxdd_strategy'))} · 보유 {_pct1(alloc.get('bh_max_dd') if alloc.get('bh_max_dd') is not None else v0h.get('maxdd_bh'))}"),
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
    s2 = (f'<section class="panel" id="s2"><h2>② 결정층 KPI — {_esc(rung)} {_esc(dep_label)} · v0 와 나란히</h2>'
          f'<div class="note">이 절의 kpis/allocation/episodes/flags 는 <b>{_esc(rung)}</b> 확률로 돌린 결정층입니다 '
          f'({_esc(dep_label)}; summary_v1.json 의 같은 키와 같은 뜻).</div>'
          f'<div class="grid">{"".join(kpis)}</div>'
          f'<div class="note">{_esc(P2_HONESTY_ITEMS[6])} {_esc(P2_HONESTY_ITEMS[7])}</div>'
          + "<h3>v1 vs v0</h3>" + _p2_kv(side)
          + "<h3>상태별 점유 · -5%/20일 비율 · 중앙 런</h3>" + _p2_table(st_rows, ["state", "occupancy", "n", "dd5_rate", "median_run"])
          + "<h3>KPI 전체</h3>" + _p2_kv(kp_rest, "kpis 없음")
          + ("<h3>구성</h3>" + _p2_kv(cfg) if cfg else "") + "</section>")

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
    s3 = (f'<section class="panel" id="s3"><h2>③ 에피소드 평가 — {_esc(rung)} {_esc(dep_label)} (evaluate.episode_eval, v0 와 같은 규칙)</h2>'
          '<div class="note">리드타임은 고점 전 최대 20거래일 탐색 창에 걸리므로 무작위 순환 이동 기준선과 나란히 읽는다(실험 #1).</div>'
          + ("<h3>요약</h3>" + _p2_kv(ep_sum) if ep_sum else "")
          + _p2_table(ep_tab, order3, "에피소드 표 없음") + "</section>")

    # ④ 배분
    s4 = (f'<section class="panel" id="s4"><h2>④ 배분 시뮬레이션 — v1({_esc(rung)} {_esc(dep_label)}) vs v0 vs 보유</h2>'
          '<div class="note">상태→톤(normal→hold, caution→caution, reduce→reduce) → 비중 100/50/25%, t 종가 판정을 t+1 수익에, 전환 시 5bp. v0 열은 summary_v0_completed.json 의 allocation.</div>'
          + _alloc_matrix(alloc, v0_alloc)
          + _img(charts.pop("cumret_v1", None), "누적 수익(로그): 보유 vs v1 결정층 vs v0 톤 배분.", "Cumulative return v1 vs v0 vs hold") + "</section>")

    # ⑤ 민감도
    sens = s.get("sensitivities")
    sens_recs = []
    if isinstance(sens, dict):
        for name, v in sens.items():
            if not isinstance(v, dict):
                continue
            row = {"name": name}
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
    s5 = ('<section class="panel" id="s5"><h2>⑤ 민감도 (wide · symmetric_dwell · no_dwell) — 보고만, 선택에 쓰지 않음</h2>'
          + _p2_table(sens_recs, ["name", "switches_per_year", "warn_share", "occupancy_caution", "occupancy_reduce", "true_alarm_share",
                                  "max_changes_any_5_sessions", "churn_alert_sessions", "kpi_ceiling_ok", "cagr", "max_dd"], "sensitivities 없음")
          + "</section>")

    # ⑥ 비배포 단(정보) — 배포 단과 나란히, 톤·비중 주장이 아님
    info_rows = _info_layer_rows(s)
    s6 = ('<section class="panel" id="s6">'
          f'<h2>⑥ 배포되지 않은 단 — {_esc(INFO_DISPLAY_LABEL)}</h2>'
          '<div class="note">같은 결정층 규칙을 다른 사다리 단의 확률로 돌린 결과입니다. '
          + (f'배포된 단은 <b>{_esc(rung)}</b> 하나뿐이고, ' if deployed
             else f'배포된 단이 없어(deploy {_esc(str(deploy))}) 맨 윗줄 <b>{_esc(rung)}</b> 도 {_esc(INFO_DISPLAY_LABEL)} 이며, ')
          + '나머지 줄은 비교용 정보입니다 — 톤·비중 제안이 아니며 ②~⑤ 의 헤드라인 숫자에 들어가지 않습니다. '
          '(단 선택은 summary_p2.json.acceptance 가 하고, 이 페이지는 판정을 바꾸지 않습니다.)</div>'
          + _p2_table(info_rows, ["name", "role", "switches_per_year", "warn_share", "occupancy_caution", "occupancy_reduce",
                                  "true_alarm_share", "true_alarm_share_baseline", "detection_rate", "median_lead_days",
                                  "max_changes_any_5_sessions", "churn_alert_sessions", "cagr", "max_dd"], "비배포 단 자료 없음")
          + "</section>")

    # ⑦ 차트
    s7 = ('<section class="panel" id="s7"><h2>⑦ 차트</h2>'
          + _img(charts.pop("prob_path", None),
                 f"OOS 확률 경로 — 굵은 선이 {'배포 단' if deployed else INFO_DISPLAY_LABEL}({rung}) 확률, 회색 점선이 기후학(clim), 밴드가 결정층 상태."
                 + (" 얇은 선은 배포되지 않은 M3(정보)." if rung != "M3" else ""),
                 "OOS probability path with decision states")
          + _img(charts.pop("state_bands", None), "SPY 종가(로그)와 결정층 상태 밴드.", "SPY with state bands"))
    for k, png in list(charts.items()):
        s7 += _img(png, f"추가 차트: {k}", k)
    s7 += "</section>"

    # ⑧ 정직·경고
    s8 = ('<section class="panel" id="s8"><h2>⑧ 정직 문구 · 플래그 · 경고</h2>'
          + ("<h3>플래그</h3>" + _warn_list(flags) if flags else "")
          + "<h3>경고</h3>" + _warn_list(warns)
          + "<h3>정직 문구와 리스크 (§16)</h3><ol class=\"method\">" + "".join(f"<li>{_esc(t)}</li>" for t in P2_HONESTY_ITEMS) + "</ol>"
          + f'<div class="note">{_esc(P2_FOOTNOTE)}</div>'
          + f'<div class="foot">market-risk-lab · 결정층 v1 · 생성 {_esc(run.get("generated_at_utc") or _now_str())} · '
            '<a href="index.html">오늘 판정으로</a> · <a href="calibration_p2.html">보정 리포트</a> · <a href="backtest_v0.html">v0 백테스트</a></div></section>')
    _write_html(out_html, "결정층 v1 백테스트 — market-risk-lab", head + nav + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8)


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


def _legend(ax, **kw):
    ax.legend(fontsize=8, facecolor=PALETTE["card"], edgecolor=PALETTE["line"], labelcolor=PALETTE["tx"], **kw)


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
