# -*- coding: utf-8 -*-
"""v0 페이지 — docs/backtest_v0.html · docs/index.html · v0 차트.

계산·임계값·산출물 스키마는 mrl/report.py 에서 옮겨온 그대로다. 이 파일에서 바뀐 것은 **보이는 글자뿐**이다:
고등학생·대학생이 처음 봐도 읽히는 한국어로 고쳐 쓰고, 같은 문장의 영어판을 함께 심어 오른쪽 위 토글로 바꿔 본다
(STYLE_I18N.md §1~§4). 토글 자체는 껍데기(report_common.page_html)가 한 곳에서 달아 준다.

지키는 선
    * 숫자·티커·날짜·구간·표본 수는 하나도 지우지 않는다. 쉬운 말로 바꾸되 조건과 한계는 그대로 남긴다.
    * "투자 조언 아님", 사전 등록 문구, 무작위 이동 기준선, v0 동결 표기는 두 언어 모두에 남는다.
    * 숫자·티커·날짜는 스팬 밖에 한 번만 둔다(두 벌로 심지 않는다 — STYLE_I18N.md §1).
    * 표는 열 제목과 각주만 두 언어로, 셀 값은 건드리지 않는다.
render_index 는 today 에 p2/p3 가 있으면 각 카드를 끼워 넣으므로 report_p2 · report_p3 를 쓴다.
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

from mrl import sizing as _sizing
from mrl import track as _track
from mrl.config import (TONES, TONE_EXPOSURE, V0_SIGNALS, SITE_URL, BACKTEST_START, DECISION_P2, P2, P2_STATES, STATE_TO_TONE,
                        HOLDOUT_START, RESULTS_DIR, P3, HMM_P3, ENSEMBLE_P3, KILL_P3, P3_DRIFT, SCENARIO_P3,
                        MODEL_P2_PATH, MODEL_P3_PATH, KILL_RECORD_PATH, KILL_MANUAL_PATH)

from mrl.i18n import bi, bi_attr, bi_html
# _kpi·_img·_kv_table·_records_table·_warn_list 은 아래 두 언어판(_bi_kpi·_bi_img·_bi_kv·_bi_tbl·_bi_warns)으로 갈아탔다
from mrl.report_common import (HONESTY_BOLD, HONESTY_BULLETS, HONESTY_LINE, HONESTY_TITLE, PALETTE, SIGNAL_KO,
                               SUBSTITUTION_RULES, TONE_COLORS, TONE_KO, VARIANT_KO, _esc, _find, _fmt, _is_nan, _is_num,
                               _label, _new_fig, _now_str, _plain_log_axis, _png, _series_close, _state_pill, _to_records,
                               _tone_pill, _tone_runs, _write_html)
from mrl.report_p2 import (_first, p2_card)
from mrl.report_p3 import (_sub, p3_card)
from mrl.signals_v0 import v0_combo as _v0_combo      # 판정 문구의 영어판을 v0 규칙 자신에게서 꺼내 온다(읽기만 함)


# ==================================================================
# 0. 쉬운 한국어 + 한/영 두 벌 (STYLE_I18N.md) — 표현 전용 도구
# ==================================================================
def _bi_note(ko: str, en: str) -> str:
    """작은 설명 한 줄."""
    return f'<div class="note">{bi(ko, en)}</div>'


def _bi_note_html(ko: str, en: str) -> str:
    """숫자·<b> 를 품은 설명 한 줄 — 호출자가 이스케이프를 책임진다(우리가 만든 마크업만)."""
    return f'<div class="note">{bi_html(ko, en)}</div>'


def _bi_h2(num: str, ko: str, en: str) -> str:
    """섹션 제목. 번호(①…)는 번역하지 않으므로 스팬 밖에 둔다."""
    return f"<h2>{num} {bi(ko, en)}</h2>"


def _bi_h3(ko: str, en: str) -> str:
    return f"<h3>{bi(ko, en)}</h3>"


def _bi_tag(ko: str, en: str, value_html: str = "", warn: bool = False) -> str:
    """알약 태그 — 라벨만 두 벌, 값(날짜·개수)은 한 번만."""
    cls = "tag warn" if warn else "tag"
    return f'<span class="{cls}">{bi(ko, en)}{(" " + value_html) if value_html else ""}</span>'


def _bi_kpi(ko: str, en: str, value: str, sub: str = "") -> str:
    """숫자 카드 한 장. 제목·설명만 두 벌이고 값은 그대로 둔다(report_common._kpi 와 같은 모양)."""
    return (f'<div class="kpi"><div class="lb">{bi(ko, en)}</div><div class="v">{value}</div>'
            f'<div class="s">{sub}</div></div>')


def _bi_bits(*parts) -> str:
    """카드 아래 설명 = (한국어, English, 값) 묶음들을 ' · ' 로 잇는다. 값은 스팬 밖에 한 번만 나온다."""
    out = []
    for ko, en, val in parts:
        out.append(bi(ko, en) + (f" {val}" if val not in (None, "") else ""))
    return " · ".join(out)


def _bi_img(png: bytes | None, ko: str, en: str, alt: str) -> str:
    """차트 — 설명만 두 벌. 차트 안 글자는 영어로 통일한다(STYLE_I18N.md §1: 그림은 이중화하지 않는다)."""
    if not png:
        return f'<div class="note">{bi("차트 없음 — " + ko, "no chart — " + en)}</div>'
    b64 = base64.b64encode(png).decode("ascii")
    return (f'<div class="chart"><img src="data:image/png;base64,{b64}" alt="{_esc(alt)}">'
            f'<div class="cap">{bi(ko, en)}</div></div>')


# 표 열 제목 — 원문 키 → (쉬운 한국어, English). 없는 키는 report_common._label 로 떨어진다(원문 표시).
_LB: dict[str, tuple[str, str]] = {
    # 방향 성적표
    "tone": ("신호등 판정", "traffic-light call"),
    "h": ("며칠 뒤(거래일)", "days ahead"), "horizon": ("며칠 뒤(거래일)", "days ahead"),
    "n": ("표본(일)", "sample (days)"), "n_days": ("표본(일)", "sample (days)"),
    "n_blocks": ("겹치지 않는 창 수", "non-overlapping windows"),
    "hit": ("적중률", "hit rate"), "hit_rate": ("적중률", "hit rate"),
    "base": ("기준선(무조건 오른다고 할 때)", "baseline (always saying up)"),
    "base_rate": ("기준선(무조건 오른다고 할 때)", "baseline (always saying up)"),
    "baseline": ("기준선(무조건 오른다고 할 때)", "baseline (always saying up)"),
    "always_up": ("기준선(무조건 오른다고 할 때)", "baseline (always saying up)"),
    "edge": ("차이(적중률 − 기준선)", "edge (hit rate − baseline)"),
    "ci_lo": ("95% 범위 아래", "95% range, low"), "ci_low": ("95% 범위 아래", "95% range, low"),
    "ci_hi": ("95% 범위 위", "95% range, high"), "ci_high": ("95% 범위 위", "95% range, high"),
    "ci": ("95% 범위", "95% range"),
    "fwd_mean": ("그 뒤 수익률 평균", "average return after"), "fwd_median": ("그 뒤 수익률 중앙값", "median return after"),
    "fwd_p10": ("나쁜 쪽 10%", "10th percentile"), "fwd_p90": ("좋은 쪽 10%", "90th percentile"),
    "mean": ("평균", "average"), "median": ("중앙값", "median"), "std": ("퍼진 정도", "standard deviation"),
    "p10": ("나쁜 쪽 10%", "10th percentile"), "p90": ("좋은 쪽 10%", "90th percentile"),
    "dd5_rate": ("20일 안에 5% 넘게 떨어진 비율", "share that fell 5% or more within 20 days"),
    "y_dd5_20": ("20일 안에 5% 넘게 떨어진 비율", "share that fell 5% or more within 20 days"),
    "y_dd10_60": ("60일 안에 10% 넘게 떨어진 비율", "share that fell 10% or more within 60 days"),
    "y_sign_5": ("5일 뒤 상승", "up after 5 days"), "y_sign_20": ("20일 뒤 상승", "up after 20 days"),
    "y_sign_60": ("60일 뒤 상승", "up after 60 days"), "y_vol_20": ("많이 움직인 20일", "high-movement 20 days"),
    "fwd_ret_5": ("5일 뒤 수익률", "return after 5 days"), "fwd_ret_20": ("20일 뒤 수익률", "return after 20 days"),
    "fwd_ret_60": ("60일 뒤 수익률", "return after 60 days"),
    "fwd_maxdd_20": ("20일 안 최대 하락폭", "worst drop within 20 days"),
    "fwd_maxdd_60": ("60일 안 최대 하락폭", "worst drop within 60 days"),
    # 하락 사건 표
    "peak_date": ("고점 날", "peak date"), "trough_date": ("바닥 날", "trough date"),
    "depth": ("고점 대비 하락폭", "drop from the peak"), "days_to_trough": ("고점→바닥(일)", "days peak → trough"),
    "recovery_date": ("고점 되찾은 날", "date it got back to the peak"),
    "days_to_recover": ("되찾기까지(일)", "days to get back"),
    "first_warn_date": ("첫 경고 날", "first warning date"), "first_warning": ("첫 경고 날", "first warning date"),
    "first_warn": ("첫 경고 날", "first warning date"), "warn_date": ("첫 경고 날", "first warning date"),
    "warn_tone": ("경고 판정", "warning call"),
    "lead_days": ("며칠 먼저 경고(+ = 고점 전)", "days of warning (+ = before the peak)"),
    "held_to_trough": ("바닥까지 경고 유지", "warning held to the trough"),
    "held": ("바닥까지 경고 유지", "warning held to the trough"),
    "missed": ("놓침", "missed"), "detected": ("미리 잡음", "caught in advance"),
    # 사건 요약 · 경보
    "detection_rate": ("미리 잡은 비율", "share caught in advance"),
    "detect_rate": ("미리 잡은 비율", "share caught in advance"),
    "n_episodes": ("하락 사건 수", "number of declines"), "n_detected": ("미리 잡은 수", "caught in advance"),
    "n_missed": ("놓친 수", "missed"),
    "median_lead_days": ("며칠 먼저 경고했나(중앙값)", "median days of warning"),
    "median_lead": ("며칠 먼저 경고했나(중앙값)", "median days of warning"),
    "mean_lead_days": ("며칠 먼저 경고했나(평균)", "mean days of warning"),
    "median_lead_days_fresh": ("새로 켜진 경고만(중앙값, 일)", "median for freshly turned-on warnings (days)"),
    "lookback": ("경고를 찾아보는 창(고점 앞 최대 거래일)", "search window before the peak (trading days)"),
    "n_lead_capped": ("창 끝에 걸린 수", "hit the search-window cap"),
    "n_lead_positive": ("고점 전에 경고한 수", "warned before the peak"),
    "n_held_to_trough": ("바닥까지 유지된 수", "held to the trough"),
    "n_evaluable": ("채점할 수 있는 사건 수", "episodes that can be scored"),
    "n_episodes_total_1993": ("1993년부터의 사건 수", "declines since 1993"),
    "warn_share": ("경고가 켜져 있던 날 비율", "share of days under warning"),
    "n_warn_runs": ("경고가 이어진 구간 수", "warning runs"),
    "n_true_alarms": ("진짜였던 경보 수", "true alarms"), "n_false_alarms": ("헛경보 수", "false alarms"),
    "n_unresolved_alarms": ("아직 판정 못 한 경보 수", "alarms not yet resolved"),
    "false_alarms_per_year": ("헛경보(1년에 몇 번)", "false alarms per year"),
    "false_alarm_per_year": ("헛경보(1년에 몇 번)", "false alarms per year"),
    "false_alarms": ("헛경보 수", "false alarms"),
    "false_alarm_rate": ("헛경보 비율", "false-alarm rate"),
    "false_alarm_rate_baseline": ("헛경보 비율 기준선(아무 날이나 경고라 했을 때)",
                                  "false-alarm baseline (if any day were called a warning)"),
    "true_alarm_share": ("진짜였던 경보 비중", "share of alarms that were real"),
    "true_alarm_share_baseline": ("그 기준선(평소에 20일 안 −5% 가 나는 비율)",
                                  "baseline (how often a −5% drop happens anyway)"),
    "null_true_alarm_share": ("진짜 경보 비중(경고를 무작위로 돌렸을 때)",
                              "share of real alarms after randomly rotating the warnings"),
    "null": ("경고를 무작위로 돌려놓은 기준선", "baseline after randomly rotating the warnings"),
    "n_shift": ("돌려본 횟수", "number of random shifts"),
    "detection_rate_mean": ("미리 잡은 비율 평균", "mean share caught"),
    "detection_100_share": ("전부 잡아낸 경우의 비율", "share of shifts that caught every decline"),
    "median_lead_days_p5": ("먼저 경고한 날수 5분위", "days of warning, 5th percentile"),
    "median_lead_days_p50": ("먼저 경고한 날수 중앙값", "days of warning, median"),
    "median_lead_days_p95": ("먼저 경고한 날수 95분위", "days of warning, 95th percentile"),
    "true_alarm_share_mean": ("진짜 경보 비중 평균", "mean share of real alarms"),
    # 판정 전환
    "n_tone_switches": ("판정이 바뀐 수", "number of switches"),
    "tone_switches_per_year": ("판정 바뀜(1년에)", "switches per year"),
    "switches_per_year": ("판정 바뀜(1년에)", "switches per year"), "n_switches": ("판정이 바뀐 수", "number of switches"),
    "median_run_len": ("한 판정이 이어진 기간(중앙값, 일)", "median days a call lasted"),
    "median_run_days": ("한 판정이 이어진 기간(중앙값, 일)", "median days a call lasted"),
    "median_run": ("한 판정이 이어진 기간(중앙값, 일)", "median days a call lasted"),
    "mean_run_days": ("한 판정이 이어진 기간(평균, 일)", "mean days a call lasted"),
    "median_warn_run_len": ("경고가 이어진 기간(중앙값, 일)", "median days a warning lasted"),
    # 배분 시뮬레이션
    "cagr": ("연평균 수익(CAGR)", "annual return (CAGR)"),
    "cagr_alloc": ("규칙대로 했을 때 연평균 수익", "annual return following the rule"),
    "cagr_strategy": ("규칙대로 했을 때 연평균 수익", "annual return following the rule"),
    "cagr_bh": ("그냥 보유했을 때 연평균 수익", "annual return just holding"),
    "cagr_buy_hold": ("그냥 보유했을 때 연평균 수익", "annual return just holding"),
    "bh_cagr": ("그냥 보유했을 때 연평균 수익", "annual return just holding"),
    "maxdd": ("가장 크게 깎였을 때", "biggest drop from the peak"),
    "max_dd": ("가장 크게 깎였을 때", "biggest drop from the peak"),
    "maxdd_alloc": ("규칙대로 했을 때 가장 크게 깎인 폭", "biggest drop from the peak, following the rule"),
    "maxdd_strategy": ("규칙대로 했을 때 가장 크게 깎인 폭", "biggest drop from the peak, following the rule"),
    "maxdd_bh": ("그냥 보유했을 때 가장 크게 깎인 폭", "biggest drop from the peak, just holding"),
    "max_dd_bh": ("그냥 보유했을 때 가장 크게 깎인 폭", "biggest drop from the peak, just holding"),
    "bh_maxdd": ("그냥 보유했을 때 가장 크게 깎인 폭", "biggest drop from the peak, just holding"),
    "bh_max_dd": ("그냥 보유했을 때 가장 크게 깎인 폭", "biggest drop from the peak, just holding"),
    "maxdd_improvement": ("깎임이 줄어든 정도(규칙−보유, +면 나아짐)", "reduction in the worst drop (rule − holding, + is better)"),
    "excess_cagr": ("연수익 차이(규칙−보유)", "excess annual return (rule − holding)"),
    "total_return": ("전체 수익", "total return"), "vol": ("움직인 폭(연)", "how much it moved (annual)"),
    "sharpe": ("샤프 비율", "Sharpe ratio"), "cost_bps": ("바꿀 때 드는 비용(bp)", "switching cost (bp)"),
    "exposure_mean": ("평균 주식 비중", "average share held in stocks"),
    "buy_hold": ("그냥 계속 보유", "just holding"), "bh": ("그냥 계속 보유", "just holding"),
    "allocation": ("규칙대로 넣고 빼기", "following the rule"), "strategy": ("규칙대로 넣고 빼기", "following the rule"),
    "diff": ("차이", "difference"), "worst_month": ("가장 나빴던 달", "worst month"),
    "worst_month_date": ("가장 나빴던 달(날짜)", "worst month (date)"),
    # 125칸 · 자료 범위
    "overall_m": ("한 달 흐름", "monthly"), "overall_w": ("한 주 흐름", "weekly"), "overall_d": ("하루 흐름", "daily"),
    "rule": ("걸린 규칙 번호", "rule that matched"), "share": ("전체 대비 비율", "share of all days"),
    "count": ("일수", "days"),
    "start": ("시작", "start"), "end": ("끝", "end"), "first": ("첫 관측", "first"), "last": ("마지막 관측", "last"),
    "variant": ("계산 방식", "variant"), "basket": ("바스켓", "basket"),
    "n_watch_avail": ("그날 쓸 수 있던 워치리스트 종목", "watchlist names available that day"),
    "fg_avail": ("공포·탐욕 지수 있었나", "Fear & Greed available"), "eod_avail": ("마감 30분 자료 있었나", "closing-bar data available"),
    "per_year": ("1년에", "per year"), "years": ("햇수", "years"), "key": ("구분", "item"),
    # 장부
    "n_outcome_20": ("결과가 나온 날 수(20일)", "days already scored (20 days)"),
    "n_outcome_60": ("결과가 나온 날 수(60일)", "days already scored (60 days)"),
    "n_outcome_dd5": ("결과가 나온 날 수(하락 여부)", "days already scored (drop check)"),
    "hit_20": ("적중률(20일)", "hit rate (20 days)"), "base_20": ("기준선(20일)", "baseline (20 days)"),
    "hit_60": ("적중률(60일)", "hit rate (60 days)"), "base_60": ("기준선(60일)", "baseline (60 days)"),
    "fwd_ret_20_mean": ("20일 뒤 수익률 평균", "average return after 20 days"),
    "fwd_ret_60_mean": ("60일 뒤 수익률 평균", "average return after 60 days"),
}


def _bi_label(key) -> str:
    """표 머리글 한 칸. 아는 키는 두 언어로, 모르는 키는 원문 그대로 둔다(자료 열 이름은 번역하지 않는다)."""
    pair = _LB.get(str(key))
    return bi(pair[0], pair[1]) if pair else _label(key)


def _bi_tbl(records, order: list[str] | None = None, empty=("자료 없음", "no data yet"), fmt=None) -> str:
    """records → 표. report_common._records_table 과 같은 마크업·같은 셀 값이고, **열 제목만** 두 언어다."""
    fmt = fmt or _fmt
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
        cells = "".join(f'<td class="{"num" if numeric[c] else ""}">{fmt(r.get(c), c)}</td>' for c in cols)
        body.append(f"<tr>{cells}</tr>")
    return f'<div class="tblwrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _bi_kv(d, empty=("자료 없음", "no data yet"), fmt=None) -> str:
    """dict → 항목/값 표(report_common._kv_table 과 같은 모양). 항목 이름과 머리글만 두 언어."""
    fmt = fmt or _fmt
    if not isinstance(d, dict) or not d:
        return f'<div class="note">{bi(empty[0], empty[1])}</div>'
    sub = {k: v for k, v in d.items() if isinstance(v, dict)}
    flat = {k: v for k, v in d.items() if not isinstance(v, (dict, list, tuple, pd.DataFrame, pd.Series))}
    lists = {k: v for k, v in d.items() if isinstance(v, (list, tuple, pd.DataFrame, pd.Series))}
    parts = []
    if flat:
        rows = "".join(f'<tr><td>{_bi_label(k)}</td><td class="num">{fmt(v, k)}</td></tr>' for k, v in flat.items())
        parts.append('<div class="tblwrap"><table><thead><tr><th>' + bi("항목", "item")
                     + '</th><th class="num">' + bi("값", "value") + f"</th></tr></thead><tbody>{rows}</tbody></table></div>")
    if sub:
        metrics: list[str] = []
        for v in sub.values():
            for k in v.keys():
                if k not in metrics:
                    metrics.append(k)
        head = "".join(f'<th class="num">{_bi_label(k)}</th>' for k in sub.keys())
        rows = []
        for m in metrics:
            cells = "".join(f'<td class="num">{fmt(v.get(m), m)}</td>' for v in sub.values())
            rows.append(f"<tr><td>{_bi_label(m)}</td>{cells}</tr>")
        parts.append('<div class="tblwrap"><table><thead><tr><th>' + bi("지표", "measure")
                     + f'</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    for k, v in lists.items():
        recs = _to_records(v)
        if recs:
            parts.append(f"<h3>{_bi_label(k)}</h3>" + _bi_tbl(recs, fmt=fmt))
        else:
            parts.append(f'<div class="note">{_bi_label(k)}: {fmt(v, k)}</div>')
    return "".join(parts) if parts else f'<div class="note">{bi(empty[0], empty[1])}</div>'


def _bi_warns(items, empty=("경고 없음", "no warnings")) -> str:
    """경고 목록 — 문장은 계산층이 만든 **자료**라 번역하지 않고 그대로 보여준다(원문 보존).

    대신 목록 앞에 그 사실을 두 언어로 밝힌다. 경고를 요약하거나 빼지 않는다.
    """
    items = [str(w) for w in (items or []) if w is not None and str(w).strip()]
    if not items:
        return f'<div class="note">{bi(empty[0], empty[1])}</div>'
    body = ('<ul class="warnlist">'
            + "".join(f'<li><span class="raw mono">{_esc(w)}</span></li>' for w in items) + "</ul>")
    return _bi_note("아래 경고문은 계산 과정이 남긴 기록이라 한국어 원문 그대로 둡니다.",
                    "The warnings below are raw diagnostics written by the pipeline and are kept in the original Korean.") + body


# 계산 방식 두 가지 — 이름(faithful·completed)은 식별자라 번역하지 않는다
_VARIANT_BI = {
    "faithful": ("faithful — 실제 대시보드와 똑같이, 아직 안 끝난 오늘 봉까지 넣고 계산",
                 "faithful — exactly like the live dashboard, including today's unfinished bar"),
    "completed": ("completed — 끝난 봉만 넣고 계산(다시 돌려도 같은 값)",
                  "completed — only finished bars, so it can be reproduced exactly"),
}


def _variant_bi(name: str) -> tuple[str, str]:
    """모르는 이름이면 예전 라벨(VARIANT_KO)로 떨어진다 — 설명을 잃지 않는다."""
    if name in _VARIANT_BI:
        return _VARIANT_BI[name]
    return (VARIANT_KO.get(name, str(name)), str(name))


def _unit(value: str, ko: str, en: str) -> str:
    """숫자 + 단위. 숫자는 한 번만, 단위만 두 벌 심는다 ('20일' / '20 days')."""
    return f"{value}{bi(ko, ' ' + en)}"


def _ten_of(x) -> str:
    """비율을 자연빈도로 (STYLE_I18N.md §2.2): 0.63 → '10번 중 약 6번 (63.0%)'. 못 읽는 값이면 빈 문자열."""
    if _is_nan(x):
        return ""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return ""
    if not 0.0 <= v <= 1.0:
        return ""
    n = int(round(v * 10))
    return bi(f"10번 중 약 {n}번", f"about {n} times in 10") + f" ({v * 100:.1f}%)"


# P1~P9 신호 — 이름 + 무엇을 보는지 한 줄. 번호는 기존 대시보드(v0)와 같게 두고, P8(뉴스)은 v0 점수에 없다.
_SIGNAL_BI = {
    "fang": ("P1 · 큰 기술주(FANG)가 시장보다 센가", "P1 · are the big tech names (FANG) outrunning the market"),
    "macd": ("P2 · 흐름이 바닥에서 위로 도는가 (MACD)", "P2 · is the trend turning up off a bottom (MACD)"),
    "fg": ("P3 · 사람들이 얼마나 겁먹었나 (CNN 공포·탐욕)", "P3 · how scared the market feels (CNN Fear & Greed)"),
    "vix": ("P4 · 겁이 꼭대기를 찍고 꺾였나 (VIX)", "P4 · has fear peaked and turned down (VIX)"),
    "ma": ("P5 · SPY 가 180일 평균선 위인가", "P5 · is SPY above its 180-day average"),
    "eod": ("P6 · 장 마지막 30분에 사는 힘이 있나", "P6 · is there buying in the last 30 minutes"),
    "lead": ("P7 · 앞서 달리는 종목이 있나", "P7 · are there stocks clearly leading"),
    "btc": ("P9 · 비트코인이 먼저 움직였나 (BTC)", "P9 · has bitcoin moved first (BTC)"),
}


# VALIDATION.md §0 사전 등록 문단을 쉬운 말로 옮긴 것 — 숫자·조건은 한 글자도 빼지 않았다(원문은 VALIDATION.md §0).
_HONESTY_PLAIN = [
    ("주식은 원래 오르는 날이 더 많습니다: 하루 54.1% · 20거래일 65.4%(서로 겹치지 않는 창 422개로만 세면 64.0%) · "
     "60거래일 72.0%(창 140개 중 70.0%). 2013년 이후는 더 높습니다: 55.3% / 68~69% / 77~79%.",
     "Stocks rise more often than not: 54.1% of days, 65.4% over 20 trading days (64.0% counting only the 422 "
     "non-overlapping windows), 72.0% over 60 trading days (70.0% of 140 windows). Since 2013 it is higher still: "
     "55.3% / 68-69% / 77-79%."),
    ("오를지 내릴지는 사실상 못 맞힙니다: 논문에 나온 가장 좋은 모형도 한 달 예측력(out-of-sample R²)이 0.5~1%뿐이고, "
     "이는 '무조건 오른다'고 말할 때보다 적중률이 0.3%p 높은 정도입니다. 5%p 더 맞힌다는 것을 통계로 확인하려면 "
     "20일 창이 717개(약 57년) 필요합니다(기준 α=5%, 검정력 80%).",
     "Direction is close to unpredictable: the best models in the literature reach only 0.5-1% monthly out-of-sample "
     "R², which is about +0.3 percentage points of hit rate over always saying 'up'. To confirm a 5-point edge you "
     "would need 717 twenty-day windows (about 57 years) at the usual 5% significance and 80% power."),
    ("반대로 위험은 어느 정도 맞힙니다: VIX 하나만 봐도 다음 20일에 실제로 움직인 폭(실현변동성)이 큰 쪽인지 "
     "맞히는 정확도(AUC)가 0.87, 20일 안에 5% 넘게 떨어지는지는 0.76입니다. 반면 20일 뒤 오를지 내릴지는 0.52로 "
     "동전 던지기와 거의 같습니다(AUC 는 0.5 가 찍기, 1.0 이 완벽).",
     "Risk, on the other hand, is partly predictable: the VIX level alone tells whether the next 20 days will be a "
     "high-movement stretch with 0.87 accuracy (AUC), and whether the market drops 5% within 20 days with 0.76. "
     "Whether it ends up or down after 20 days scores 0.52 — a coin flip (AUC 0.5 is guessing, 1.0 is perfect)."),
]


def _substitution_en(i: int) -> str:
    """VALIDATION.md §3 대체 규약의 영어판. 목록 길이가 달라지면 원문을 가리킨다(빼지 않는다)."""
    en = [
        "P3 (CNN Fear & Greed): real history only from 2020-08-03. Before that the signal is left out of the average, "
        "exactly as the v0 rule does when the lookup fails.",
        "P6 (buying in the last 30 minutes): only from the 730 hourly sessions available since 2023-10. Earlier days "
        "are left out.",
        "P7 (leading stocks): of the 28 watchlist names, only those with at least 65 bars at that date — 22 names in "
        "2013, 27-28 in 2026. Survivor bias (the list was chosen in 2026) cannot be removed, so it is stated openly; "
        "n_watch_avail is shown in the report.",
        f"The six signals P1, P2, P4, P5, P7 and P9 can be reproduced together only from {BACKTEST_START} onward "
        "(BTC-USD starts 2014-09 and the rules need a full 2-year calendar window of 500-507 trading days). "
        "That date is BACKTEST_START.",
    ]
    return en[i] if i < len(en) else "See VALIDATION.md §3 for this rule (Korean original shown on the Korean side)."


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
    v_ko, v_en = _variant_bi(variant)
    tags = [_bi_tag("계산 방식", "variant", f"<b>{_esc(variant)}</b>"),
            _bi_tag("기간", "period", f"<b>{_esc(start)} ~ {_esc(end)}</b>"),
            _bi_tag("거래일", "trading days", f'<b>{_fmt(n_days, "n_days")}</b>'),
            _bi_tag("만든 시각", "generated", f'<b>{_esc(summary.get("generated_at") or _now_str())}</b>')]
    if warns:
        tags.append(_bi_tag("경고", "warnings", f"<b>{len(warns)}</b>", warn=True))
    head = ('<header><div class="eyebrow">market-risk-lab · '
            + bi("1단계 · 실험 #0", "Phase 1 · experiment #0") + "</div>"
            + "<h1>" + bi("v0 규칙을 과거에 그대로 돌려 본 성적표", "How the v0 rules would have scored in the past")
            + f' — <span class="mono">{_esc(variant)}</span></h1>'
            + '<div class="sub">'
            + bi("기존 대시보드(v0)의 규칙을 한 줄도 바꾸지 않고, 과거의 하루하루를 그날까지 알 수 있던 자료만으로 다시 판정해 채점했습니다.",
                 "We replayed the old dashboard's rules (v0) unchanged, day by day, using only the data that was "
                 "available on each day, and then scored the calls.")
            + "</div>" + _bi_note(v_ko, v_en)
            + f'<div class="tags">{"".join(tags)}</div></header>')
    sec_titles = [("①", "정직한 요약", "An honest summary"),
                  ("②", "방향 성적표", "Direction scorecard"),
                  ("③", "하락 사건 표", "Table of declines"),
                  ("④", "판정별 그 뒤 수익", "Returns after each call"),
                  ("⑤", "규칙대로 넣고 빼기 vs 그냥 보유", "Rule vs just holding"),
                  ("⑥", "판정이 바뀐 횟수", "How often the call changed"),
                  ("⑦", "125칸 표", "The 125 combinations"),
                  ("⑧", "자료·약속·경고", "Data, rules, warnings"),
                  ("⑨", "방법", "Method")]
    nav = ('<nav class="nav">'
           + "".join(f'<a href="#s{i + 1}">{num} {bi(ko, en)}</a>' for i, (num, ko, en) in enumerate(sec_titles))
           + "</nav>")

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
            hit_kpis.append(_bi_kpi(f"20거래일 뒤 방향 적중 · {TONE_KO.get(t, t)}",
                                    f"direction called right after 20 trading days · {t}", _fmt(hit, "hit"),
                                    _bi_bits(("기준선", "baseline", _fmt(base, "base")),
                                             ("표본", "sample", _unit(_fmt(r.get("n"), "n"), "일", "days")),
                                             ("겹치지 않는 창", "non-overlapping windows", _fmt(r.get("n_blocks"), "n_blocks")))))
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
    det_sub = (_bi_bits(("미리 잡은 사건", "caught in advance", f"{_fmt(n_det, 'n')} / {_fmt(n_ep, 'n')}"))
               if n_ep is not None else bi("episode_summary 기준", "from episode_summary"))
    if not _is_nan(null_det):
        det_sub += " · " + bi("경고를 무작위로 돌려놨을 때", "after randomly rotating the warnings") + f" {_fmt(null_det, 'detection_rate')}"
    lead_sub = bi("+ 면 고점 전에 경고한 것", "+ means the warning came before the peak")
    if lookback is not None:
        lead_sub += " · " + bi("경고를 찾아본 창은 최대", "we look back at most") + " " + _unit(_fmt(lookback, "lookback"), "거래일", "trading days")
    if not _is_nan(fresh):
        lead_sub += " · " + bi("새로 켜진 경고만", "freshly turned-on warnings only") + " " + _unit(_fmt(fresh, "median_lead_days_fresh"), "일", "days")
    if not _is_nan(null_lead):
        lead_sub += " · " + bi("무작위로 돌려놨을 때", "after randomly rotating") + " " + _unit(_fmt(null_lead, "median_lead_days_p50"), "일", "days")
    fa_sub = bi("경고가 났는데 20일 안에 −5% 가 없던 경우", "a warning with no −5% drop in the next 20 days")
    if not _is_nan(far):
        fa_sub += " · " + bi("헛경보 비율", "false-alarm rate") + f" {_fmt(far, 'false_alarm_rate')}"
    if not _is_nan(far_base):
        fa_sub += " · " + bi("아무 날이나 경고라 했을 때", "if any day were called a warning") + f" {_fmt(far_base, 'false_alarm_rate_baseline')}"
    kpis = [
        _bi_kpi("10% 넘게 떨어진 사건을 미리 잡은 비율", "share of ≥10% declines caught in advance",
                _fmt(det, "detection_rate"), det_sub),
        _bi_kpi("고점보다 며칠 먼저 경고했나 (중앙값)", "days of warning before the peak (median)",
                (_unit(_fmt(lead, "lead_days"), "일", "days") if lead is not None else "—"), lead_sub),
        _bi_kpi("헛경보 (1년에 몇 번)", "false alarms per year", _fmt(fa, "false_alarms_per_year"), fa_sub),
        _bi_kpi("판정이 바뀐 횟수 (1년에)", "how often the call changed (per year)", _fmt(spy_, "switches_per_year"), ""),
        _bi_kpi("규칙대로 넣고 뺐을 때 연평균 수익", "annual return if you had followed the rule", _fmt(a_cagr, "cagr"),
                _bi_bits(("그냥 계속 들고 있었다면", "just holding", _fmt(b_cagr, "cagr")))),
        _bi_kpi("가장 크게 깎였을 때 (고점 대비)", "biggest drop from the peak", _fmt(a_dd, "maxdd"),
                _bi_bits(("그냥 계속 들고 있었다면", "just holding", _fmt(b_dd, "maxdd")))),
    ] + hit_kpis
    honesty_extra = summary.get("honesty")
    # 사전 등록 문단: 화면에는 쉬운 말로 옮긴 판을 보여주고, 원문(VALIDATION.md §0 그대로)은 tooltip 으로 함께 싣는다.
    honesty_src = _esc(HONESTY_TITLE + " — " + " ".join(HONESTY_BULLETS) + " " + HONESTY_BOLD)
    s1 = ('<section class="panel" id="s1">' + _bi_h2("①", "정직한 요약", "An honest summary")
          + f'<div class="grid">{"".join(kpis)}</div>'
          + f'<div class="quote" title="{honesty_src}"><b>'
          + bi(HONESTY_TITLE, "0. Honest premises (measured on SPY, 1993-01-29 to 2026-09-04, 8,458 trading days)")
          + "</b><ul>"
          + "".join(f"<li>{bi(ko, en)}</li>" for ko, en in _HONESTY_PLAIN)
          + "<li><b>"
          + bi(HONESTY_BOLD,
               'So the main target of this platform is "the risk over the next month", and direction hit rates are '
               "reported only as a reference, always next to the baseline.")
          + "</b></li></ul></div>"
          + (_bi_note("하락 사건을 세는 규칙은 계산 과정이 남긴 기록이라 한국어 원문 그대로 둡니다.",
                      "The rule for counting decline episodes is written by the pipeline and is kept in "
                      "the original Korean.")
             + f'<div class="note"><span class="raw mono">{_esc(honesty_extra)}</span></div>'
             if honesty_extra else "")
          + _bi_note("위 문단은 VALIDATION.md §0 에 미리 적어 둔 글을 쉬운 말로 옮긴 것입니다. 숫자와 조건은 하나도 바꾸지 않았고, "
                     "성적이 좋든 나쁘든 이 문단은 그대로 둡니다(사전 등록 원문은 VALIDATION.md §0 에 그대로 있습니다). "
                     "숫자 카드는 채점 결과에 그 항목이 있을 때만 채워집니다 — “—” 는 자료가 없다는 뜻입니다.",
                     "This paragraph restates, in plain words, the text pre-registered in VALIDATION.md §0. Not one "
                     "number or condition was changed, and it stays fixed whether the results look good or bad (the "
                     "pre-registered original is kept as it is in VALIDATION.md §0). A number card is filled in only "
                     "when the scoring summary contains that item — “—” means no data.")
          + "</section>")

    # ---- ② 방향 성적표 ----
    order2 = ["tone", "h", "horizon", "n", "n_blocks", "hit", "hit_rate", "base", "base_rate", "baseline", "edge",
              "ci_lo", "ci_low", "ci_hi", "ci_high", "ci", "fwd_mean", "fwd_median", "fwd_p10", "fwd_p90", "dd5_rate", "y_dd5_20"]
    s2 = ('<section class="panel" id="s2">' + _bi_h2("②", "방향 성적표 — 언제나 기준선과 나란히",
                                                     "Direction scorecard — always next to the baseline")
          + _bi_note("buy·hold·neutral 판정은 “h거래일 뒤에 오른다”고 말한 것으로, caution·reduce 판정은 “내린다”고 말한 것으로 "
                     "셉니다. 기준선은 같은 표본에서 아무 생각 없이 “오른다”고만 답했을 때의 적중률입니다. 서로 겹치지 않는 창 수"
                     "(n_blocks)와, 구간을 통째로 잘라 다시 계산한 95% 범위(블록 부트스트랩)가 있으면 함께 적습니다.",
                     "buy/hold/neutral calls are scored as saying “up in h trading days”, caution/reduce as saying "
                     "“down”. The baseline is what you get by always answering “up” over the same sample. Where "
                     "available we also print the number of non-overlapping windows (n_blocks) and a 95% range "
                     "obtained by resampling whole blocks of days.")
          + _bi_tbl(directional, order2, ("방향 채점 기록이 없습니다 (directional)", "no direction records yet (directional)"))
          + ((_bi_h3("평소에는 얼마나 자주 일어나나 (base_rates)", "how often it normally happens (base_rates)")
              + _bi_kv(base_rates)) if base_rates else "")
          + "</section>")

    # ---- ③ 에피소드 표 ----
    order3 = ["peak_date", "trough_date", "depth", "days_to_trough", "recovery_date", "days_to_recover",
              "first_warn_date", "first_warning", "first_warn", "warn_date", "warn_tone", "lead_days", "held_to_trough", "held", "missed", "detected"]
    ep10 = summary.get("episodes10")
    ep5 = summary.get("episodes5")
    ep20 = summary.get("episodes20")
    lb_txt = f'{_fmt(lookback, "lookback") if lookback is not None else 20}'
    s3 = ('<section class="panel" id="s3">' + _bi_h2("③", "하락 사건 표 — SPY 가 고점에서 얼마나 떨어졌나 (1993년~)",
                                                     "Table of declines — how far SPY fell from its peak (since 1993)")
          + _bi_note_html(
              "바닥을 찍고 직전 고점을 되찾으면 한 사건이 끝난 것으로 봅니다. 되찾기 전에 다시 기준만큼 떨어지면 새 사건으로 세되 "
              "되찾은 날은 함께 씁니다. “며칠 먼저 경고”는 첫 경고(caution·reduce 판정)가 나온 날과 고점 날의 거래일 차이입니다"
              "(+ 면 고점 전에 경고). 되살린 구간 밖의 사건은 v0 판정이 아예 없어 “—” 로 둡니다. 경고는 고점 앞 최대 "
              f"{lb_txt}거래일(lookback)까지만 찾으므로 이 값을 넘는 리드는 나올 수 없고, 그 전부터 켜져 있던 경고는 lead_capped "
              "로 표시합니다(새로 켜진 경고만의 중앙값은 median_lead_days_fresh). 미리 잡은 비율과 며칠 먼저 경고했는지는, "
              "경고 기록을 통째로 아무 데나 무작위로 돌려놓고 잰 값과 반드시 나란히 읽어야 합니다 — 두 값이 구별되지 않으면 그 "
              "숫자는 “찾아보는 창의 길이 × 경고가 켜져 있던 날 비율”이 만들어낸 것일 뿐입니다.",
              "A decline ends when the market gets back to the peak it fell from. If it drops again by the threshold "
              "before recovering, that counts as a new decline but shares the recovery date. “Days of warning” is the "
              "number of trading days between the first warning (a caution/reduce call) and the peak (+ means the "
              "warning came first). Declines outside the replayed window have no v0 call at all and are shown as “—”. "
              f"We search for a warning at most {lb_txt} trading days before the peak (lookback), so no lead can "
              "exceed that; a warning that was already on before the window is marked lead_capped (the median over "
              "freshly turned-on warnings only is median_lead_days_fresh). The share caught and the days of warning "
              "must be read next to the same measurement taken after rotating the whole warning series to a random "
              "starting point — if the two are indistinguishable, the number is only an artifact of the search-window "
              "length times the share of days under warning.")
          + ((_bi_h3("요약", "summary") + _bi_kv(ep_sum)) if ep_sum else "")
          + _bi_h3("10% 넘게 떨어진 사건", "declines of 10% or more")
          + _bi_tbl(ep10, order3, ("사건 표가 없습니다 (episodes10)", "no table yet (episodes10)"))
          + _bi_h3("5% 넘게 떨어진 사건", "declines of 5% or more")
          + _bi_tbl(ep5, order3, ("사건 표가 없습니다 (episodes5)", "no table yet (episodes5)"))
          + ((_bi_h3("20% 넘게 떨어진 사건", "declines of 20% or more") + _bi_tbl(ep20, order3)) if ep20 is not None else "")
          + _bi_img(charts.pop("tone_bands", None),
                    "SPY 종가(로그 눈금)와 그날의 신호등 판정 색. 세로 점선 = 10% 넘게 떨어진 사건의 고점(빨강)·바닥(파랑).",
                    "SPY close on a log scale, shaded by that day's traffic-light call. Dotted lines mark the peak "
                    "(red) and trough (blue) of each ≥10% decline.",
                    "SPY price with tone bands")
          + "</section>")

    # ---- ④ 톤별 선행수익 분포 ----
    dist_cols = ["tone", "h", "horizon", "n", "fwd_mean", "fwd_median", "fwd_p10", "fwd_p90", "dd5_rate", "y_dd5_20"]
    dist_recs = []
    for r in directional:
        sub = {k: r[k] for k in dist_cols if k in r}
        if any(k in sub for k in ("fwd_mean", "fwd_median", "fwd_p10", "fwd_p90", "dd5_rate", "y_dd5_20")):
            dist_recs.append(sub)
    s4 = ('<section class="panel" id="s4">' + _bi_h2("④", "판정별로 그 뒤 수익이 어떻게 퍼졌나",
                                                     "How returns were spread out after each call")
          + _bi_note("각 판정이 나온 날부터 h거래일 뒤 SPY 수익률이 어떻게 퍼져 있는지(평균·중앙값·나쁜 쪽 10%·좋은 쪽 10%)와, "
                     "20일 안에 5% 넘게 떨어진 비율입니다. 창이 서로 겹치기 때문에 표본 수는 실제보다 크게 보입니다.",
                     "For each call, how the SPY return h trading days later was spread out (average, median, worst "
                     "10%, best 10%), plus how often the market fell 5% or more within 20 days. The windows overlap, "
                     "so the sample count looks larger than it really is.")
          + _bi_img(charts.pop("fwd_box", None),
                    "판정별 다음 20거래일 SPY 수익률 상자그림(가운데 선 = 중앙값, 수염 = 나쁜 쪽·좋은 쪽 5%).",
                    "Box plot of the SPY return over the next 20 trading days for each call (middle line = median, "
                    "whiskers = 5th and 95th percentile).",
                    "Forward 20-day return by tone")
          + (_bi_tbl(dist_recs, dist_cols) if dist_recs
             else _bi_note("퍼짐을 보여 주는 열(fwd_mean 등)이 directional 자료에 없습니다.",
                           "The spread columns (fwd_mean and friends) are not in the directional data."))
          + "</section>")

    # ---- ⑤ 배분 시뮬 vs 보유 ----
    exp_txt = " · ".join(f"{TONE_KO.get(t, t)} {int(round(e * 100))}%" for t, e in TONE_EXPOSURE.items())
    exp_en = " · ".join(f"{t} {int(round(e * 100))}%" for t, e in TONE_EXPOSURE.items())
    s5 = ('<section class="panel" id="s5">' + _bi_h2("⑤", "규칙대로 넣고 뺐을 때 vs 그냥 들고 있었을 때",
                                                     "Following the rule vs just holding")
          + _bi_note_html(
              f"신호등 판정 → 주식 비중: {_esc(exp_txt)} (VALIDATION.md 에 미리 정해 둔 값). t일 종가로 내린 판정을 다음 날"
              "(t+1)의 수익에 적용하고, 주식 비중이 바뀌는 날에는 비용 5bp(0.05%)를 뺍니다. 주식에 넣지 않은 나머지는 현금이고 "
              "이자는 0으로 봅니다.",
              f"Traffic-light call → how much to hold in stocks: {_esc(exp_en)} (fixed in advance in VALIDATION.md). "
              "The call made at day t's close is applied to day t+1's return, and on days when the weight changes we "
              "subtract a cost of 5bp (0.05%). Whatever is not in stocks sits in cash and earns nothing.")
          + _bi_kv(alloc, ("배분 결과가 없습니다 (allocation)", "no allocation result yet"))
          + _bi_img(charts.pop("cumret", None),
                    "1.0 을 넣었을 때 불어난 배수(로그 눈금): 그냥 SPY 보유 vs 판정대로 넣고 빼기.",
                    "Growth of 1.0 on a log scale: just holding SPY vs following the calls.",
                    "Cumulative return: buy&hold vs allocation")
          + "</section>")

    # ---- ⑥ 판정 전환 빈도 ----
    s6 = ('<section class="panel" id="s6">' + _bi_h2("⑥", "판정이 얼마나 자주 바뀌었나", "How often the call changed")
          + _bi_note("신호등 판정이 바뀐 날의 수입니다. 너무 자주 바뀌면 비용이 들고 말이 자꾸 바뀐다는 뜻이고, 너무 드물면 "
                     "경고가 늦다는 뜻입니다.",
                     "The number of days on which the traffic-light call changed. Changing too often costs money and "
                     "means the page keeps contradicting itself; changing too rarely means the warnings arrive late.")
          + _bi_kv(switches, ("판정 전환 결과가 없습니다 (switches)", "no switch counts yet"))
          + _bi_img(charts.pop("switches", None), "달마다 판정이 바뀐 횟수.", "Number of times the call changed each month.",
                    "Monthly tone switches")
          + "</section>")

    # ---- ⑦ 125칸 점유표 ----
    cells = _to_records(summary.get("cells"))
    if cells:
        tot = sum(float(r.get("n", 0) or 0) for r in cells)
        for r in cells:
            if "share" not in r and tot > 0 and r.get("n") is not None:
                r["share"] = float(r["n"]) / tot
        cells = sorted(cells, key=lambda r: -(float(r.get("n", 0) or 0)))
    s7 = ('<section class="panel" id="s7">' + _bi_h2("⑦", "125칸 표 — 한 달·한 주·하루 상태의 조합",
                                                     "The 125 combinations — monthly, weekly and daily state")
          + _bi_note_html(
              f"상태가 5가지이고 기간이 셋(한 달·한 주·하루)이라 5×5×5 = 125칸이 나옵니다. 그중 실제로 나온 칸만 많이 머문 "
              f"순서로 보여 줍니다(총 {len(cells)}칸). 규칙 번호는 combo_advice 에서 처음 걸린 규칙이고, 0 은 걸린 규칙이 "
              "없어 평균으로 정했다는 뜻입니다.",
              "There are 5 states and 3 time frames (monthly, weekly, daily), so 5×5×5 = 125 boxes. Only the boxes "
              f"that actually occurred are listed, most-visited first ({len(cells)} of them). The rule number is the "
              "first matching rule in combo_advice; 0 means no rule matched and the average was used instead.")
          + _bi_tbl(cells, ["overall_m", "overall_w", "overall_d", "n", "share", "tone", "rule"],
                    ("칸 표가 없습니다 (cells)", "no combination table yet (cells)"))
          + "</section>")

    # ---- ⑧ 데이터 범위·대체 규약·경고 ----
    s8 = ('<section class="panel" id="s8">' + _bi_h2("⑧", "자료 기간 · 빠진 자료를 다루는 약속 · 경고",
                                                     "Data range, missing-data rules and warnings")
          + _bi_h3("자료 기간", "data range") + _bi_kv(dr, ("자료 기간이 없습니다 (data_range)", "no data range recorded"))
          + _bi_h3("자료가 빠졌을 때의 약속 (VALIDATION.md §3)", "what we do when data is missing (VALIDATION.md §3)")
          + '<ul class="plain">'
          + "".join(f"<li>{bi(t, _substitution_en(i))}</li>" for i, t in enumerate(SUBSTITUTION_RULES)) + "</ul>"
          + _bi_h3("경고", "warnings") + _bi_warns(warns))
    # 남은 차트(알 수 없는 이름)는 여기서 모두 보여준다 — 숨기지 않음
    for k, png in list(charts.items()):
        s8 += _bi_img(png, f"추가 차트: {k}", f"extra chart: {k}", k)
    s8 += "</section>"

    # ---- ⑨ 방법 설명 ----
    method = [
        ("<b>되살리기</b>: 거래일 t 마다, 그날 v0 가 실제로 볼 수 있던 창(SPY·VIX·FANG = 2년 달력 창 (t-2y, t] 로 "
         "500~507거래일, 워치리스트 1년 창 ≈252거래일, BTC 2년 731행)을 그날까지의 저장 자료로 되살려 v0 의 "
         "build_metrics → assess → composite → overall → combo_advice 를 그대로 돌립니다. 미래의 값은 한 줄도 쓰지 않습니다.",
         "<b>Replay</b>: for every trading day t we rebuild the exact window v0 could see that day (SPY, VIX and FANG "
         "over a 2-year calendar window (t-2y, t], i.e. 500-507 trading days; the watchlist over a 1-year window of "
         "about 252 trading days; BTC over 2 years, 731 rows) from the cache as of that day, and run v0's "
         "build_metrics → assess → composite → overall → combo_advice unchanged. Not one future row is used."),
        ("<b>계산 방식 두 가지</b>: faithful 은 실제 화면처럼 아직 안 끝난 t일 봉과 주·월을 넣고, completed 는 끝난 봉만 "
         "넣습니다. 둘의 차이가 곧 “안 끝난 봉” 때문에 생기는 오차의 크기입니다.",
         "<b>Two variants</b>: faithful includes day t's unfinished daily bar and the unfinished week and month, just "
         "like the live page; completed uses finished bars only. The gap between them is the size of the "
         "unfinished-bar problem."),
        ("<b>맞히려는 것</b>: y_dd5_20(주 목표 — 다음 20거래일 안에 −5%), y_dd10_60, y_sign_5/20/60(참고용), y_vol_20. "
         "정의는 VALIDATION.md §1 에 적혀 있습니다.",
         "<b>What we try to call</b>: y_dd5_20 (the main target — a −5% drop within the next 20 trading days), "
         "y_dd10_60, y_sign_5/20/60 (reference only) and y_vol_20. The definitions are in VALIDATION.md §1."),
        ("<b>방향 채점</b>: buy·hold·neutral 은 오른다고 말한 것, caution·reduce 는 내린다고 말한 것으로 셉니다. 무조건 "
         "오른다고 답하는 기준선과 서로 겹치지 않는 창 수(n//h)를 항상 함께 적고, 95% 범위는 구간을 통째로 잘라 다시 "
         "계산해(블록 부트스트랩) 구합니다.",
         "<b>Scoring direction</b>: buy/hold/neutral count as calling 'up', caution/reduce as calling 'down'. We always "
         "print the always-up baseline and the number of non-overlapping windows (n//h); the 95% range comes from "
         "resampling whole blocks of days."),
        ("<b>하락 사건</b>: SPY 종가가 고점 대비 5·10·20% 떨어진 구간입니다. 첫 경고 날, 며칠 먼저 경고했는지, 놓쳤는지, "
         "헛경보(경고가 이어진 뒤 20일 안에 −5% 가 없던 경우)를 셉니다.",
         "<b>Declines</b>: stretches where the SPY close fell 5%, 10% or 20% below its peak. We count the first warning "
         "day, how many days of warning there were, misses, and false alarms (a run of warnings with no −5% drop in "
         "the next 20 days)."),
        ("<b>넣고 빼기</b>: 신호등 판정 → 주식 비중(100/100/100/50/25%). t일 종가 판정을 다음 날(t+1) 수익에 적용하고, "
         "주식 비중이 바뀌는 날에는 5bp(0.05%)를 뺍니다.",
         "<b>Allocation</b>: the traffic-light call sets how much to hold in stocks (100/100/100/50/25%). The call from "
         "day t's close is applied to day t+1's return, minus 5bp (0.05%) on days when the weight changes."),
        ("<b>한계</b>: P3·P6 는 자료 자체가 짧아 그 이전 구간에서는 평균에서 빼고 셉니다(위의 약속). P7 은 지금 살아남은 "
         "종목으로 만든 목록이라 그 치우침(생존편향)을 없앨 수 없습니다. 하락 사건 표본이 12회 수준이라 여기 있는 모든 "
         "숫자는 범위가 넓습니다.",
         "<b>Limits</b>: P3 and P6 simply do not have long histories, so before those dates they are left out of the "
         "average (see the rules above). P7 uses a watchlist picked in 2026, so survivor bias cannot be removed. With "
         "only about 12 declines in the sample, every number on this page carries a wide range."),
    ]
    s9 = ('<section class="panel" id="s9">' + _bi_h2("⑨", "어떻게 쟀는지", "How this was measured")
          + '<ol class="method">' + "".join(f"<li>{bi_html(ko, en)}</li>" for ko, en in method) + "</ol>"
          + '<div class="foot">market-risk-lab · '
          + bi("v0 는 손대지 않고 얼려 둔 기준입니다 (실험 #0)", "v0 is a frozen benchmark and is never edited (experiment #0)")
          + " · " + bi("만든 시각", "generated") + f' {_esc(summary.get("generated_at") or _now_str())} · '
          + f'<a href="index.html">{bi("오늘 판정 보기", "see today\'s call")}</a><br>'
          + bi("이 페이지는 과거를 채점한 성적표이지 투자 조언이 아닙니다.",
               "This page is a scorecard of the past, not investment advice.")
          + "</div></section>")

    _write_html(out_html, f"v0 backtest ({variant}) — market-risk-lab", head + nav + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9)


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
# 오늘 자료가 어떤 상태인지 — (쉬운 한국어, English)
_STATUS_BI = {
    "current": ("장이 끝난 뒤 확정된 종가", "final close, after the market closed"),
    "intraday": ("장중 — 오늘 봉이 아직 안 끝났습니다", "market open — today's bar is not finished yet"),
    "pre_open": ("개장 전 — 어제 종가로 계산", "before the open — based on yesterday's close"),
    "weekend": ("주말 — 직전 거래일 종가로 계산", "weekend — based on the last trading day's close"),
    "holiday": ("휴장일 — 직전 거래일 종가로 계산", "market holiday — based on the last trading day's close"),
    "stale": ("자료가 늦게 들어옴 — 직전 거래일 종가로 계산", "data arrived late — based on the last trading day's close"),
    "incomplete": ("오늘 봉이 아직 안 끝나 어제 기준으로 계산", "today's bar is unfinished — based on yesterday"),
}


_STATUS_KO = {k: v[0] for k, v in _STATUS_BI.items()}      # 예전 이름(파사드가 내보낸다) — 한국어 쪽만


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


def _verdict_en(d: dict) -> tuple[str, str] | None:
    """오늘 판정 문구의 영어판 — v0 규칙이 이미 들고 있는 영어 문장을 꺼내 온다(새로 번역하지 않는다).

    같은 상태 조합으로 v0 규칙을 한국어로 다시 돌려 화면의 문구·톤과 **정확히 같을 때만** 영어를 쓴다.
    조금이라도 어긋나면 None 을 돌려주고, 화면에는 한국어 원문 하나만 남긴다(모르면 지어내지 않는다).
    """
    ko = d.get("verdict_ko") or d.get("verdict") or d.get("name")
    if not ko:
        return None
    try:
        name_ko, act_ko, tone_ko = _v0_combo(str(d.get("overall_m")), str(d.get("overall_w")), str(d.get("overall_d")), "ko")
        name_en, act_en, tone_en = _v0_combo(str(d.get("overall_m")), str(d.get("overall_w")), str(d.get("overall_d")), "en")
    except (ValueError, TypeError, KeyError):
        return None
    if name_ko != str(ko) or tone_ko != tone_en or tone_ko != str(d.get("tone") or ""):
        return None
    stored_act = d.get("action_ko") or d.get("action") or ""
    return (name_en, act_en if (not stored_act or stored_act == act_ko) else "")


def _verdict_block(name: str, d: dict) -> str:
    """오늘의 판정 카드 한 장 — STYLE_I18N.md §4 차례: ① 무엇을 말하는 칸인지 ② 오늘 값 ③ 읽는 법 ④ 범위·한계."""
    tone = str(d.get("tone") or "neutral")
    color = TONE_COLORS.get(tone, PALETTE["mut"])
    verdict = d.get("verdict_ko") or d.get("verdict") or d.get("name")
    action = d.get("action_ko") or d.get("action") or ""
    en_pair = _verdict_en(d)
    if not verdict:                                    # 판정 자체가 없을 때만 우리가 쓴 문구 → 두 언어로
        verdict_html = bi("판정 없음", "no call")
    else:
        verdict_html = bi(str(verdict), en_pair[0]) if en_pair else _esc(verdict)
    action_html = (bi(str(action), en_pair[1]) if (en_pair and action and en_pair[1]) else _esc(action))
    tf = ""
    for lb, ko, en, sk in (("overall_m", "한 달 흐름", "monthly", "score_m"),
                           ("overall_w", "한 주 흐름", "weekly", "score_w"),
                           ("overall_d", "하루 흐름", "daily", "score_d")):
        st = d.get(lb)
        sc = d.get(sk)
        sc_txt = "" if _is_nan(sc) else f"{float(sc) * 100:+.1f}"
        tf += (f'<div class="tf"><div class="lb">{bi(ko, en)}</div><div>{_state_pill(str(st)) if not _is_nan(st) else "—"}</div>'
               f'<div class="sc">{_esc(sc_txt)}</div></div>')
    states = d.get("states_d") if isinstance(d.get("states_d"), dict) else {}
    if not states:
        states = {k: d.get(f"state_{k}") for k in V0_SIGNALS if not _is_nan(d.get(f"state_{k}"))}
    sigs = ""
    for k in V0_SIGNALS:
        v = states.get(k)
        if v is None or _is_nan(v):
            continue
        s_ko, s_en = _SIGNAL_BI.get(k, (SIGNAL_KO.get(k, k), str(k)))
        sigs += f'<div class="sig">{bi(s_ko, s_en)} {_state_pill(str(v))}</div>'
    extras = []
    if d.get("fg_avail") is False:
        extras.append(_bi_tag("F&G 없음 → 이 신호는 빼고 계산", "Fear & Greed missing → this signal is left out", warn=True))
    if d.get("eod_avail") is False:
        extras.append(_bi_tag("마감봉 없음 → 이 신호는 빼고 계산", "closing bar missing → this signal is left out", warn=True))
    if d.get("n_watch_avail") is not None:
        extras.append(_bi_tag("그날 쓸 수 있던 워치리스트 종목", "watchlist names available",
                              f'<b>{_fmt(d.get("n_watch_avail"), "n")}</b>'))
    if d.get("asof"):
        extras.append(_bi_tag("기준 날짜", "as of", f'<b>{_esc(pd.Timestamp(d["asof"]).strftime("%Y-%m-%d"))}</b>'))
    v_ko, v_en = _variant_bi(name)
    return ('<div class="panel">'
            + f'<div class="eyebrow">{bi(v_ko, v_en)}</div>'
            + _bi_note("오늘 v0 규칙이 내린 한 줄 판정", "What the v0 rules call today, in one line")
            + f'<div class="verdict" style="color:{color}">{verdict_html}</div>'
            + f'<div class="v-act">{action_html}</div>'
            + f'<div style="margin-top:8px">{bi("오늘 신호등", "today's traffic light")} {_tone_pill(tone)}</div>'
            + f'<div class="tfrow">{tf}</div>'
            + _bi_note("색은 그 기간의 상태이고, 옆 숫자는 v0 규칙이 매긴 종합 점수를 100배 한 값입니다(클수록 좋다는 뜻).",
                       "The colour is the state over that period; the number next to it is v0's overall score "
                       "multiplied by 100 (higher means better).")
            + (f'<div class="sigs">{sigs}</div>' if sigs else "")
            + (_bi_note("신호 번호(P1~P9)는 기존 대시보드와 같습니다. P8 은 뉴스 카드라 점수에 들어가지 않아 여기에 없습니다.",
                        "The signal numbers (P1-P9) are the same as on the old dashboard. P8 is a news card and is not "
                        "scored, so it does not appear here.") if sigs else "")
            + (f'<div class="tags">{"".join(extras)}</div>' if extras else "") + "</div>")


def _track_line(variants: dict, ls: dict) -> str:
    """STYLE_I18N.md §4 셋째 줄 — “평소와 비교”. 장부에 채점된 결과가 있을 때만 쓰고,
    표본 수·기준선·겹치지 않는 창 수를 반드시 함께 적는다(없으면 아무 말도 하지 않는다)."""
    if not isinstance(ls, dict) or not variants:
        return ""
    name = "completed" if "completed" in variants else next(iter(variants))
    tone = str((variants.get(name) or {}).get("tone") or "")
    by_tone = ls.get("by_tone") if isinstance(ls.get("by_tone"), dict) else {}
    b = by_tone.get(tone)
    if not isinstance(b, dict) or _is_nan(b.get("hit_20")) or _is_nan(b.get("n_outcome_20")):
        return ""
    n_out = int(float(b["n_outcome_20"]))
    if n_out < 1:
        return ""
    n_days = _fmt(b.get("n"), "n")
    hit, base = _fmt(b.get("hit_20"), "hit"), _fmt(b.get("base_20"), "base")
    tone_ko = TONE_KO.get(tone, tone)
    return _bi_note(
        f"장부에 적히는 것은 {name} 쪽 판정입니다. 지금까지 ‘{tone_ko}’ 판정이 나온 날은 {n_days}일이고, 그중 결과가 "
        f"확정된 {n_out}일에서 {hit} 맞혔습니다. 같은 표본에서 ‘무조건 오른다’고만 답했으면 {base} 였습니다. "
        f"창이 서로 겹치므로 실제로 독립된 관측은 약 {n_out // 20}개뿐이라, 이 숫자는 아직 크게 흔들립니다.",
        f"The call written into the log is the {name} one. So far the ‘{tone}’ call has appeared on {n_days} days; of "
        f"the {n_out} that have been scored, {hit} were right. Always answering ‘up’ over the same sample would have "
        f"given {base}. The windows overlap, so this rests on only about {n_out // 20} independent observations — "
        "these numbers still move a lot.")


def _ledger_block(ls: dict) -> str:
    if not isinstance(ls, dict) or not ls:
        return _bi_note("장부 요약이 아직 없습니다.", "The log has nothing to summarise yet.")
    n = ls.get("n", 0)
    n_out = ls.get("n_with_outcome", 0)
    kpis = [_bi_kpi("기록 일수", "days logged", _fmt(n, "n"),
                    _bi_bits(("첫 기록", "first", _esc(ls.get("first_asof") or "—")),
                             ("마지막 기록", "last", _esc(ls.get("last_asof") or "—")))),
            _bi_kpi("결과가 나온 날 수 (20거래일 뒤)", "days already scored (20 trading days later)", _fmt(n_out, "n"),
                    _bi_bits(("60거래일 뒤", "after 60 trading days", _fmt(ls.get("n_with_outcome_60"), "n")))),
            _bi_kpi("비교 기준: 무조건 오른다고 했을 때 (20거래일)", "baseline: always saying up (20 trading days)",
                    _fmt(ls.get("base_rate_20"), "base"),
                    " · ".join(x for x in (bi("이 표본에서 실제로 오른 날의 비율", "how often this sample actually rose"),
                                           _ten_of(ls.get("base_rate_20"))) if x))]
    ov = ls.get("overall") if isinstance(ls.get("overall"), dict) else {}
    if ov.get("hit_20") is not None:
        kpis.append(_bi_kpi("전체 적중률 (20거래일)", "overall hit rate (20 trading days)", _fmt(ov.get("hit_20"), "hit"),
                            " · ".join(x for x in (_ten_of(ov.get("hit_20")),
                                                   _bi_bits(("기준선", "baseline", _fmt(ov.get("base_20"), "base")))) if x)))
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
        miss_txt = ('<div class="note">' + bi("기록이 빠진 거래일", "trading days missing from the log")
                    + f' {len(missing)}' + bi("일", " days") + f' ({_esc(ls.get("missing_method") or "")}): {_esc(show)}</div>')
    # 장부 P2 블록의 주석(출처 혼재·구간 생략·정의상 0 사유·홀드아웃 잠금)은 계산만 하고 아무도 안 읽던 값이었다 — 경고 목록에 싣는다
    p2s = ls.get("p2") if isinstance(ls.get("p2"), dict) else {}
    warns = list(ls.get("warnings") or []) + [f"P2: {n}" for n in (p2s.get("notes") or [])]
    return (f'<div class="grid">{"".join(kpis)}</div>'
            + ((_bi_h3("신호등 판정별 적중률 — 언제나 기준선과 나란히", "hit rate by traffic-light call — always next to the baseline")
                + _bi_tbl(rows)) if rows
               else _bi_note("아직 판정별로 채점된 결과가 없습니다.", "No results by call have been scored yet."))
            + miss_txt
            + ((_bi_h3("장부 경고", "log warnings") + _bi_warns(warns)) if warns else ""))


def render_index(today: dict, ledger_summary: dict, out_html: Path) -> None:
    """docs/index.html — 오늘 판정(faithful·completed), 마지막 갱신, 백테스트 링크, 장부 요약, 정직 문구."""
    if not isinstance(today, dict):
        raise TypeError("today 는 dict 여야 합니다")
    variants = _variants_of(today)
    asof = today.get("asof") or next((v.get("asof") for v in variants.values() if v.get("asof")), None)
    asof_txt = pd.Timestamp(asof).strftime("%Y-%m-%d (%a)") if asof else "—"
    status = str(today.get("market_status") or today.get("status") or ("incomplete" if today.get("incomplete") else "current"))
    status_ko = _STATUS_KO.get(status, status)
    status_en = _STATUS_BI.get(status, (status_ko, status))[1]
    updated = today.get("generated_at") or today.get("updated_at") or today.get("updated_at_utc") or _now_str()
    tags = [_bi_tag("기준 날짜", "as of", f"<b>{_esc(asof_txt)}</b>"),
            _bi_tag(status_ko, status_en, warn=(status != "current")),
            _bi_tag("마지막 갱신", "last updated", f"<b>{_esc(updated)}</b>")]
    if not _is_nan(today.get("spy_close")):
        tags.append(f'<span class="tag">SPY <b>{float(today["spy_close"]):,.2f}</b></span>')
    if not _is_nan(today.get("vix_close")):
        tags.append(f'<span class="tag">VIX <b>{float(today["vix_close"]):,.2f}</b></span>')
    note = today.get("note")
    head = ('<header><div class="eyebrow">market-risk-lab · '
            + bi("오늘의 v0 판정 (끝난 봉 기준)", "today's v0 call (finished bars only)") + "</div>"
            + "<h1>" + bi("시장 위험 실험실 — 오늘 판정", "Market Risk Lab — today's call") + "</h1>"
            + '<div class="sub">'
            + bi("기존 대시보드와 똑같은 v0 규칙을, 장이 끝난 뒤 끝난 봉만으로 다시 계산해 매일 장부에 적습니다. "
                 "계산 방식 두 가지(faithful·completed)를 나란히 보여 줍니다.",
                 "We run the same v0 rules as the old dashboard, recomputed after the close using finished bars only, "
                 "and write the result into the log every day. The two ways of computing it (faithful and completed) "
                 "are shown side by side.")
            + "</div>"
            + f'<div class="tags">{"".join(tags)}</div>' + (f'<div class="note">{_esc(note)}</div>' if note else "") + "</header>")
    if variants:
        blocks = "".join(_verdict_block(k, v) for k, v in variants.items())
        vsec = (f'<div class="two">{blocks}</div>'
                # STYLE_I18N.md §4 의 셋째·넷째 줄 — 평소와 비교, 그리고 읽는 법·한계를 첫 화면에서 한 번 밝힌다
                + _track_line(variants, ledger_summary)
                + _bi_note("이 판정이 맞았는지는 20거래일·60거래일 뒤에 채점해 아래 장부에 그대로 남깁니다. 방향 적중률은 "
                           "언제나 “무조건 오른다”는 기준선 옆에서 읽어야 합니다. 아직 시험 운용 중이라, 이 페이지는 "
                           "투자 조언이 아닙니다.",
                           "Whether today's call was right is scored 20 and 60 trading days later and written into the "
                           "log below, unchanged. A direction hit rate only means something next to the “always up” "
                           "baseline. This is still a trial run, so this page is not investment advice."))
    else:
        vsec = ('<div class="panel"><div class="verdict" style="color:#eab308">'
                + bi("오늘 판정 없음", "no call today") + "</div>"
                + _bi_note("today 자료에 계산 방식별 판정(faithful·completed)이 들어 있지 않습니다.",
                           "The today data has no per-variant call (faithful / completed) in it."))
    diff_note = ""
    if len(variants) >= 2:
        tones = {k: str(v.get("tone")) for k, v in variants.items()}
        if len(set(tones.values())) > 1:
            # title 속성에는 예전 대시보드 표현("톤")을 그대로 남긴다 — 기존 독자와 기존 테스트가 이 문구를 찾는다
            diff_note = ('<div class="panel" title="'
                         + bi_attr("두 변형의 톤이 다릅니다", "the two variants disagree on the tone")
                         + '"><b style="color:#eab308">'
                         + bi("두 계산 방식의 신호등 판정이 다릅니다", "The two ways of computing it disagree")
                         + "</b> — " + " · ".join(f"{_esc(k)}: {_esc(t)}" for k, t in tones.items())
                         + _bi_note("차이는 아직 안 끝난 봉(주·월이 진행 중인 기간) 때문입니다. 다시 돌려도 같은 값이 나오는 "
                                    "쪽은 끝난 봉만 쓰는 completed 입니다.",
                                    "The difference comes from bars that have not finished yet (a week or month still "
                                    "in progress). The reproducible one is completed, which uses finished bars only.")
                         + "</div>")
    # Phase 2 카드 (v0 판정 블록 아래; today["p2"] 가 없으면 Phase 1 페이지와 동일)
    p2sec = ""
    p2 = today.get("p2")
    if isinstance(p2, dict):
        if "live" not in p2 and isinstance(ledger_summary, dict) and isinstance(ledger_summary.get("p2"), dict):
            p2 = {**p2, "live": ledger_summary["p2"]}
        if "asof" not in p2 and asof:
            p2 = {**p2, "asof": asof}
        p2sec = p2_card(p2)
    # Phase 3 카드 (P2 카드 아래; v0·P2 카드는 그대로 — ARCHITECTURE_PHASE3.md §10)
    p3sec = ""
    p3 = today.get("p3")
    if isinstance(p3, dict):
        if "asof" not in p3 and asof:
            p3 = {**p3, "asof": asof}
        if "track" not in p3 and isinstance(ledger_summary, dict) and isinstance(ledger_summary.get("p3"), dict):
            p3 = {**p3, "track": ledger_summary["p3"]}
        if "acceptance" not in p3 and isinstance(p2, dict) and isinstance(p2.get("acceptance"), dict):
            p3 = {**p3, "acceptance": p2["acceptance"]}
        if "p2_deploy_mode" not in p3 and isinstance(p2, dict):
            dep2 = _first(p2, "deploy_mode", "p2_deploy_mode", default=_sub(p2, "model").get("deploy_mode"))
            if dep2 is not None:
                p3 = {**p3, "p2_deploy_mode": dep2}
        p3sec = p3_card(p3, p3.get("effective_mode"))
    warns = list(today.get("warnings") or [])
    wsec = (('<section class="panel"><h2>' + bi("오늘 경고", "Today's warnings") + "</h2>"
             + _bi_warns(warns) + "</section>") if warns else "")
    lsec = ('<section class="panel"><h2>'
            + bi("장부 요약 — 매일 적고, 20·60거래일 뒤에 채점합니다",
                 "The log — written every day, scored 20 and 60 trading days later") + "</h2>"
            + _bi_note("적어 둔 판정은 나중에 고치지 않습니다. 시간이 지나 결과가 확정되면 y_sign_20/60·y_dd5_20 칸을 채울 "
                       "뿐입니다. 적중률은 언제나 “무조건 오른다”는 기준선 옆에 나란히 둡니다.",
                       "A call, once written, is never edited. When the outcome is known we only fill in the "
                       "y_sign_20/60 and y_dd5_20 columns. Hit rates are always shown next to the “always up” baseline.")
            + _ledger_block(ledger_summary) + "</section>")
    honest = ('<section class="panel"><h2>' + bi("정직 문구", "The honest line") + "</h2>"
              + '<div class="quote">'
              + bi(HONESTY_LINE,
                   "A direction hit rate is only a reference value read next to the always-up baseline; the main "
                   "target of this platform is \"the risk over the next month\". No scorecard here can escape a wide "
                   "range — the \"X out of 12\" kind.")
              + "<br><b>"
              + bi(HONESTY_BOLD,
                   'So the main target of this platform is "the risk over the next month", and direction hit rates are '
                   "reported only as a reference, always next to the baseline.")
              + "</b></div>"
              + _bi_note("미리 정해 둔 문서(VALIDATION.md)대로, 성적이 나쁘면 끄는 규칙이 걸리는 순간 신호등 판정과 주식 비중 "
                         "제안을 감추고 참고용 정보만 보여 줍니다.",
                         "As written in advance in VALIDATION.md, the moment the switch-it-off rule fires we hide the "
                         "traffic-light call and any suggestion about how much to hold, and show information only.")
              + "</section>")
    links = ('<section class="panel"><h2>' + bi("함께 볼 자료", "More to read") + '</h2><ul class="plain">'
             + '<li><a href="backtest_v0.html">'
             + bi("v0 규칙 과거 성적표 (계산 방식 두 가지)", "How the v0 rules scored in the past (both variants)") + "</a></li>"
             + ('<li><a href="calibration_p2.html">'
                + bi("2단계 확률 맞추기 리포트 (실험 #2 · 단계별 모델 · 구간별 표 · §6 판정)",
                     "Phase 2 probability report (experiment #2, model steps, block tables, the §6 verdict)")
                + "</a></li><li><a href=\"backtest_v1.html\">"
                + bi("판정 규칙 v1 의 과거 성적표 (v0 와 나란히)", "Decision layer v1 scored on the past, side by side with v0")
                + "</a></li>" if p2sec else "")
             + ('<li><a href="sizing_p3.html">'
                + bi("3단계 주식 비중 규칙 (과거 표 · 단계 사다리 · 유지 조건)",
                     "Phase 3 rule for how much to hold (past tables, the ladder, the holding conditions)")
                + "</a></li><li><a href=\"regime_p3.html\">"
                + bi("3단계 시장 분위기 엔진·등록부 (HMM · 후보 모형 · 채택 검정)",
                     "Phase 3 market-mood engine and registry (HMM, candidate models, the adoption test)")
                + "</a></li><li><a href=\"track_record.html\">"
                + bi("3단계 성적 기록 (실제 vs 과거 재현 · 경보 · 끄는 규칙)",
                     "Phase 3 track record (live vs replayed, alarms, the switch-it-off rule)")
                + "</a></li>" if p3sec else "")
             + f'<li><a href="{_esc(SITE_URL)}">{_esc(SITE_URL)}</a></li></ul>'
             + '<div class="foot">' + bi("만든 시각", "generated") + f" {_esc(updated)} · "
             + bi("이 페이지는 투자 조언이 아닙니다.", "This page is not investment advice.") + "</div></section>")
    _write_html(out_html, "Today's call — market-risk-lab",
                head + vsec + diff_note + p2sec + p3sec + wsec + lsec + honest + links)
