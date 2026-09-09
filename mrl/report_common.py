# -*- coding: utf-8 -*-
"""리포트 공용 조각 — 페이지 껍데기·CSS·값 포매팅·표/섹션 helper·차트 helper·한영 토글 배선.

mrl/report.py 에서 그대로 옮겨온 코드다(계산·문구·임계값 변경 없음). 페이지 모듈
(report_v0 · report_p2 · report_p3)이 모두 이 모듈만 바라보므로, 한 파일을 건드리지 않고
페이지별로 나누어 작업할 수 있다.

한/영 토글은 여기 _write_html 한 곳에서만 배선한다 → 일곱 페이지가 한꺼번에 토글을 얻는다
(STYLE_I18N.md §1).
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
from mrl.i18n import (LANG_CSS, LANG_CSS_EN, LANG_TOGGLE_HTML, bi, bi_attr, bi_html, bi_th, bi_ths,
                      lang_toggle_html)



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
    "n_tone_switches": "신호등 판정이 바뀐 횟수", "tone_switches_per_year": "1년에 판정이 바뀐 횟수", "median_run_len": "중앙 런 길이(일)",
    "median_warn_run_len": "중앙 경고 런 길이(일)", "maxdd_improvement": "MaxDD 개선(배분-보유, 양수=개선)",
    "excess_cagr": "초과 CAGR(배분-보유)",
    "false_alarm_per_year": "오경보/년", "false_alarms": "오경보 수", "switches_per_year": "1년에 판정이 바뀐 횟수", "n_switches": "신호등 판정이 바뀐 횟수",
    "median_run_days": "중앙 런 길이(일)", "median_run": "중앙 런 길이(일)", "mean_run_days": "평균 런 길이(일)",
    "cagr": "CAGR", "maxdd": "고점 대비 가장 큰 하락(MaxDD)", "max_dd": "고점 대비 가장 큰 하락(MaxDD)",
    "worst_month": "최악 월", "worst_month_date": "최악 월(일자)",
    "total_return": "총수익", "vol": "변동성(연)", "sharpe": "샤프", "cost_bps": "비용(bp)", "exposure_mean": "평균 비중",
    "buy_hold": "보유(B&H)", "allocation": "배분", "strategy": "배분", "bh": "보유(B&H)", "diff": "차이",
    "overall_m": "월간", "overall_w": "주간", "overall_d": "일간", "rule": "규칙", "share": "비율", "count": "일수",
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
            return _tone_pill(v, bi_gloss=False)   # 표 셀 값 — 두 벌로 심지 않는다
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


# 신호등 판정 이름의 영어 뜻 — 식별자(buy·hold…)는 그대로 두고 옆의 뜻풀이만 언어를 따른다
TONE_EN = {"buy": "buy", "hold": "hold", "neutral": "wait and see", "caution": "be careful", "reduce": "cut back"}


def _tone_pill(t: str, bi_gloss: bool = True) -> str:
    """신호등 알약. 식별자(buy·reduce…)는 그대로 두고 옆의 뜻풀이만 화면 언어를 따른다.

    bi_gloss=False 는 **표의 셀 값**으로 쓸 때다 — STYLE_I18N.md §1 은 셀 값을 두 벌로 심지 말라고
    한다(열 제목과 각주만 이중화). 그 자리에서는 한국어 뜻풀이만 남긴다.
    """
    c = TONE_COLORS.get(t, PALETTE["mut"])
    ko, en = TONE_KO.get(t, ""), TONE_EN.get(t, "")
    gloss = (bi(ko, en) if bi_gloss else _esc(ko)) if (ko or en) else ""
    return f'<span class="pill" style="background:{c}">{_esc(t)} {gloss}</span>'


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


def page_html(title: str, body: str, stem: str = "index", lang: str = "ko") -> str:
    """페이지 껍데기 한 곳 — 여기서만 한/영 토글을 단다(일곱 페이지가 함께 얻는다).

    구조는 market-brief 대시보드와 같다: 숨긴 체크박스와 알약 라벨을 <body> 바로 안,
    본문 래퍼(.wrap) **앞**에 두어 CSS 형제 결합자(~)로 본문 언어를 바꾼다. 자바스크립트 0줄.

    `lang="en"` 이면 기본 표시를 뒤집은 자매 문서(`<stem>.en.html`)를 만든다 — 링크를 타고
    다른 페이지로 가도 언어가 유지된다(여러 페이지로 나뉜 뒤 생긴 문제; market-brief 는 한 장이라 없었다).
    두 판은 **같은 bi(ko, en) 쌍**에서 나오므로 숫자·구간·표본 수·post hoc·info_only·"투자 조언 아님" 이
    양쪽에 똑같이 실린다.
    """
    en = lang == "en"
    css = _CSS + LANG_CSS + (LANG_CSS_EN if en else "")
    return (f'<!doctype html><html lang="{"en" if en else "ko"}"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{_esc(title)}</title><style>{css}</style></head><body>"
            f"{lang_toggle_html(stem)}<div class=\"wrap\">{body}</div></body></html>")


_REL_LINK = re.compile(r'href="([A-Za-z0-9_]+)\.html"')


def _write_html(out_html: Path, title: str, body: str) -> None:
    """한국어판(`<stem>.html`)과 영어판(`<stem>.en.html`)을 함께 쓴다.

    영어판은 **같은 본문**에 껍데기만 다르고, 페이지 사이 상대 링크만 `.en.html` 로 바꾼다 —
    영어를 고른 독자가 링크를 눌러도 영어로 남는다. 페이지 안 앵커(#s1…)와 절대 URL 은 건드리지 않는다.
    """
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    stem = out_html.stem
    with open(out_html, "w", encoding="utf-8", newline="\n") as f:
        f.write(page_html(title, body, stem, "ko"))
    # 본문(.wrap 안)의 상대 링크만 바꾼다 — 알약 자신의 두 링크(한국어판/영어판)는 그대로 두어야
    # 영어판에서 '한국어' 를 눌러 돌아올 수 있다.
    doc_en = page_html(title, body, stem, "en")
    cut = doc_en.find('<div class="wrap">')
    en_doc = doc_en[:cut] + _REL_LINK.sub(r'href="\1.en.html"', doc_en[cut:])
    with open(out_html.with_suffix(".en.html"), "w", encoding="utf-8", newline="\n") as f:
        f.write(en_doc)


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


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


def _legend(ax, **kw):
    ax.legend(fontsize=8, facecolor=PALETTE["card"], edgecolor=PALETTE["line"], labelcolor=PALETTE["tx"], **kw)
