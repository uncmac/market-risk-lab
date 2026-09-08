# -*- coding: utf-8 -*-
"""리포트 렌더링 — 자체 완결 한국어 HTML (다크 팔레트, 시스템 폰트, 모바일 대응).

ARCHITECTURE.md 계약:
    render_backtest_report(summary, out_html, charts)   # 섹션 ①~⑨, 차트는 base64 PNG 인라인
    render_index(today, ledger_summary, out_html)       # 오늘 판정(두 변형)·마지막 갱신·백테스트 링크·장부 요약·정직 문구
    charts_for(summary, replay, spy_close, targets)     # matplotlib(Agg) → {이름: PNG bytes}

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

from mrl.config import TONES, TONE_EXPOSURE, V0_SIGNALS, SITE_URL, BACKTEST_START

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
@media(max-width:520px){h1{font-size:20px}.verdict{font-size:19px}.kpi .v{font-size:18px}}
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


def _records_table(records, order: list[str] | None = None, empty_msg="자료 없음") -> str:
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
        cells = "".join(f'<td class="{"num" if numeric[c] else ""}">{_fmt(r.get(c), c)}</td>' for c in cols)
        body.append(f"<tr>{cells}</tr>")
    return f'<div class="tblwrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _kv_table(d, empty_msg="자료 없음") -> str:
    """dict → 키/값 표. 값이 dict 이면(예: allocation vs buy_hold) 행렬 표로."""
    if not isinstance(d, dict) or not d:
        return f'<div class="note">{_esc(empty_msg)}</div>'
    sub = {k: v for k, v in d.items() if isinstance(v, dict)}
    flat = {k: v for k, v in d.items() if not isinstance(v, (dict, list, tuple, pd.DataFrame, pd.Series))}
    lists = {k: v for k, v in d.items() if isinstance(v, (list, tuple, pd.DataFrame, pd.Series))}
    parts = []
    if flat:
        rows = "".join(f"<tr><td>{_label(k)}</td><td class=\"num\">{_fmt(v, k)}</td></tr>" for k, v in flat.items())
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
            cells = "".join(f'<td class="num">{_fmt(v.get(m), m)}</td>' for v in sub.values())
            rows.append(f"<tr><td>{_label(m)}</td>{cells}</tr>")
        parts.append(f'<div class="tblwrap"><table><thead><tr><th>지표</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    for k, v in lists.items():
        recs = _to_records(v)
        if recs:
            parts.append(f"<h3>{_label(k)}</h3>" + _records_table(recs))
        else:
            parts.append(f'<div class="note">{_label(k)}: {_fmt(v, k)}</div>')
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
    warns = ls.get("warnings") or []
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
             f'<li><a href="{_esc(SITE_URL)}">{_esc(SITE_URL)}</a></li></ul>'
             f'<div class="foot">생성 {_esc(updated)} · 이 페이지는 투자 조언이 아닙니다.</div></section>')
    _write_html(out_html, "오늘 판정 — market-risk-lab", head + vsec + diff_note + wsec + lsec + honest + links)
