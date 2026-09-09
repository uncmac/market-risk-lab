# -*- coding: utf-8 -*-
"""v0 두 페이지(docs/index.html · docs/backtest_v0.html)의 쉬운 한국어 + 한/영 두 벌 검사.

계약: STYLE_I18N.md
  §1 토글·bi()  라벨과 문장은 두 벌, 숫자·티커·날짜는 한 벌. 표는 열 제목만 이중화.
  §2 쉬운 한국어 어려운 말을 화면에 그대로 두지 않는다(괄호 안은 허용 — "먼저 뜻, 그다음 용어").
  §3 용어표     대응표의 '원문' 표현이 우리가 쓴 문장에 남아 있으면 실패.
  §4 카드 형식  첫 화면에 ① 무엇을 말하는지 ② 오늘 값 ③ 읽는 법 ④ 범위·한계·"투자 조언 아님".
  §5 검사       한/영 스팬 개수 일치 · 영어 칸에 한글 없음 · 한국어 칸에 영어 문장 없음.

정직성(타협 없음): 쉬운 말로 바꾸는 과정에서 숫자·표본 수·구간·사전 등록 문구·무작위 이동 기준선·
v0 동결 표기·"투자 조언 아님" 이 하나라도 사라지면 아래 테스트가 실패한다.

실제 산출물(results/·docs/)은 건드리지 않는다 — tmp_path 에만 그린다.
"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrl import i18n, report, report_v0                                  # noqa: E402
from mrl.config import TONE_EXPOSURE, V0_SIGNALS                         # noqa: E402
from mrl.report_common import HONESTY_BOLD, HONESTY_BULLETS, HONESTY_LINE, HONESTY_TITLE, SUBSTITUTION_RULES  # noqa: E402


# ------------------------------------------------------------------
# 화이트리스트 — STYLE_I18N.md §5 "티커·숫자·고유명사 제외"
# ------------------------------------------------------------------
# 한국어 문장 안에 남아도 되는 식별자: 자료 열 이름·판정 이름·함수 이름은 번역하면 오히려 못 찾는다.
V0_IDENTIFIERS = {
    "buy", "hold", "neutral", "caution", "reduce",                      # 판정 이름(장부·CSV 값 그대로)
    "faithful", "completed", "today",                                   # 계산 방식·입력 dict 이름
    "build_metrics", "assess", "composite", "overall", "combo_advice",  # v0 함수 이름
    "directional", "episodes", "episode_summary", "base_rates", "cells", "switches", "allocation", "data_range",
    "lookback", "lead_capped", "median_lead_days_fresh", "n_blocks", "n_watch_avail",
    "fwd_mean", "fwd_median", "y_dd", "y_sign_", "y_vol_",              # 목표변수·열 이름(숫자 앞에서 잘린다)
}


class _identifiers_allowed:
    """검사 동안만 식별자를 화이트리스트에 얹는다(STYLE_I18N.md §5 의 '티커·고유명사 제외' 를 v0 열 이름으로 확장)."""

    def __enter__(self):
        self._old = set(i18n.ALLOWED_LATIN)
        i18n.ALLOWED_LATIN.update(w.lower() for w in V0_IDENTIFIERS)
        return self

    def __exit__(self, *exc):
        i18n.ALLOWED_LATIN.clear()
        i18n.ALLOWED_LATIN.update(self._old)
        return False


def _check_pairs(html: str):
    with _identifiers_allowed():
        return i18n.check_pairs(html)


def _ko_text(html: str) -> str:
    """우리가 쓴 한국어 문장만 모은다(= 한국어 스팬 안쪽). 자료 값·표 셀은 여기 들어오지 않는다."""
    return " ".join(i18n.visible_text(inner) for lang, inner in i18n._spans(html) if lang == "ko")


def _en_text(html: str) -> str:
    return " ".join(i18n.visible_text(inner) for lang, inner in i18n._spans(html) if lang == "en")


def _en_view(html: str) -> str:
    """영어 화면에 실제로 보이는 글자 — 한국어 스팬만 지우고 나머지는 그대로 둔다(자료 값 포함)."""
    doc = re.sub(r'<span class="lg ko">.*?</span>', " ", html, flags=re.S)
    return i18n.visible_text(doc)


_RAW_SPAN = re.compile(r'<span class="raw[^"]*">.*?</span>', re.S)
_TD = re.compile(r"<td[^>]*>.*?</td>", re.S)


def _ko_screen(html: str) -> str:
    """한국어를 고른 독자가 실제로 읽는 화면 전체 (STYLE_I18N.md §5).

    빼는 것은 계약이 정한 둘뿐이다: 원문 인용(`span.raw` — 옆에 고지가 붙는다)과 표의 셀 값(§1).
    `bi()` 밖에 그대로 꽂힌 글자도 검사에 들어온다 — 그래야 안 고친 문장이 걸린다.
    """
    doc = re.sub(r'<span class="lg en">.*?</span>', " ", html or "", flags=re.S)
    doc = _RAW_SPAN.sub(" ", doc)
    doc = _TD.sub(" ", doc)
    return i18n.visible_text(doc)


def _hard_terms(html: str) -> dict:
    """한국어 화면에 남은 어려운 용어. 쉬운 말 자체가 원문을 품는 경우(비중 ⊂ '주식 비중')는 뺀다."""
    found = i18n.find_hard_terms(_ko_screen(html))
    out = {}
    for pat, n in found.items():
        plain = i18n.TERMS[i18n.HARD_PATTERNS[pat]][0]
        if pat not in plain:                        # 쉬운 말에도 그 글자가 들어가면 피할 수 없다
            out[pat] = n
    return out


# ------------------------------------------------------------------
# 자료 — 실제 daily.py / run_backtest_v0.py 가 넘기는 모양
# ------------------------------------------------------------------
DAY = {"asof": "2026-09-04", "tone": "caution", "verdict_ko": "단기 과열 뒤 주춤",
       "action_ko": "새로 사는 건 잠시 쉬고, 가진 것은 그대로 유지",
       "overall_m": "GREEN", "overall_w": "GREEN", "overall_d": "G2R",
       "score_m": 0.70, "score_w": 0.62, "score_d": 0.51,
       "states_d": {k: "GREEN" for k in V0_SIGNALS}, "n_watch_avail": 27, "fg_avail": True, "eod_avail": False}

TODAY = {"asof": "2026-09-04", "market_status": "incomplete", "generated_at": "2026-09-08 14:51 ET",
         "spy_close": 770.19, "vix_close": 14.53, "warnings": ["SPY 마지막 봉은 장중 부분 봉일 수 있음"],
         "faithful": DAY,
         "completed": {**DAY, "tone": "hold", "verdict_ko": "꾸준한 상승 흐름",
                       "action_ko": "그대로 보유. 쉬어가는 구간이 오면 추가 매수 고려", "overall_d": "GREEN"}}

LEDGER = {"n": 240, "n_with_outcome": 180, "n_with_outcome_60": 120, "first_asof": "2025-09-05",
          "last_asof": "2026-09-04", "base_rate_20": 0.66, "overall": {"hit_20": 0.63, "base_20": 0.66},
          "by_tone": {"hold": {"n": 150, "n_outcome_20": 120, "hit_20": 0.68, "base_20": 0.66, "dd5_rate": 0.12},
                      "caution": {"n": 40, "n_outcome_20": 30, "hit_20": 0.30, "base_20": 0.66, "dd5_rate": 0.23}},
          "missing_days": ["2026-01-02"], "missing_method": "거래일 달력 대조",
          "warnings": ["아직 결과(20일)가 확정된 행이 없습니다"], "p2": {"notes": ["홀드아웃 미해제 → 라이브 채점 보류"]}}

SUMMARY = {
    "variant": "completed", "n_days": 3204, "generated_at": "2026-09-08 14:51 ET",
    "data_range": {"start": "2014-09-17", "end": "2026-09-04", "n_days": 3204},
    "episode_summary": {"n_episodes": 12, "n_detected": 9, "detection_rate": 0.75, "median_lead_days": 3.0,
                        "median_lead_days_fresh": 2.0, "false_alarms_per_year": 1.8, "lookback": 20,
                        "false_alarm_rate": 0.42, "false_alarm_rate_baseline": 0.55, "warn_share": 0.217,
                        "null": {"detection_rate_mean": 0.68, "median_lead_days_p50": 2.0, "n_shift": 500}},
    "switches": {"n_switches": 61, "switches_per_year": 4.8, "median_run_days": 9.0},
    "allocation": {"cagr": 0.121, "cagr_bh": 0.134, "maxdd": -0.246, "maxdd_bh": -0.337, "cost_bps": 5},
    "base_rates": {"y_dd5_20": 0.147, "y_sign_20": 0.654},
    "directional": [{"tone": "caution", "h": 20, "n": 452, "n_blocks": 22, "hit": 0.39, "base": 0.66, "edge": -0.27,
                     "ci_lo": 0.28, "ci_hi": 0.51, "fwd_mean": 0.004, "fwd_median": 0.009, "fwd_p10": -0.06,
                     "fwd_p90": 0.06, "dd5_rate": 0.21}],
    "episodes10": [{"peak_date": "2020-02-19", "trough_date": "2020-03-23", "depth": -0.339, "days_to_trough": 23,
                    "recovery_date": "2020-08-18", "days_to_recover": 103, "first_warn_date": "2020-02-21",
                    "warn_tone": "caution", "lead_days": 2, "held_to_trough": True, "detected": True}],
    "episodes5": [{"peak_date": "2025-02-19", "trough_date": "2025-04-08", "depth": -0.187, "lead_days": -8}],
    "cells": [{"overall_m": "GREEN", "overall_w": "GREEN", "overall_d": "GREEN", "n": 820, "tone": "buy", "rule": 1}],
    "warnings": ["캐시 경고: 마지막 SPY 봉은 장중 부분 봉 → 재현·채점에서 제외됨"],
}


@pytest.fixture()
def index_html(tmp_path) -> str:
    out = tmp_path / "index.html"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report.render_index(TODAY, LEDGER, out)
    return out.read_text(encoding="utf-8")


@pytest.fixture()
def backtest_html(tmp_path) -> str:
    out = tmp_path / "backtest_v0.html"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report.render_backtest_report(SUMMARY, out, {})
    return out.read_text(encoding="utf-8")


@pytest.fixture()
def pages(index_html, backtest_html) -> dict:
    return {"index": index_html, "backtest_v0": backtest_html}


# ==================================================================
# 1. 두 벌 심기 (STYLE_I18N.md §1 · §5)
# ==================================================================
def test_both_pages_have_matching_korean_and_english_spans(pages):
    """한 문장을 한쪽 언어로만 심으면 반대 언어 화면에 구멍이 난다."""
    for name, html in pages.items():
        n_ko, n_en, problems = _check_pairs(html)
        assert problems == [], f"{name}: {problems[:3]}"
        assert n_ko == n_en > 30, f"{name}: ko={n_ko}, en={n_en}"


def test_no_span_is_empty(pages):
    """빈 스팬은 개수만 맞추고 뜻은 비어 있는 상태 — 짝 검사를 속이지 못하게 막는다."""
    for name, html in pages.items():
        empty = [inner for _, inner in i18n._spans(html) if not i18n.visible_text(inner)]
        assert empty == [], f"{name}: 빈 스팬 {len(empty)}개"


def test_numbers_and_cells_are_not_planted_twice(pages):
    """숫자·티커·날짜는 번역하지 않는다 — 값만 든 스팬도, 표 셀 안의 스팬도 없어야 한다(§1)."""
    for name, html in pages.items():
        bare = [t for t in (i18n.visible_text(inner) for _, inner in i18n._spans(html))
                if t and not re.search(r"[가-힣]|[A-Za-z]{2,}", t)]      # 단위 낱말('일')은 라벨이라 허용
        assert bare == [], f"{name}: 값만 두 벌로 심은 스팬 {bare[:3]}"
        # 값 칸은 class 를 달고 나온다(<td class="num"> / <td class="">); class 없는 <td> 는 항목 이름 칸이다
        for cell in re.findall(r'<td class="[^"]*">(.*?)</td>', html, flags=re.S):
            assert 'class="lg' not in cell, f"{name}: 표 셀 값을 두 벌로 심었습니다 — {cell[:40]}"


def test_table_headers_are_bilingual_but_cells_are_not(backtest_html):
    """표는 열 제목만 이중화한다 — 셀 값(티커·숫자·날짜)은 그대로 둔다."""
    assert i18n.bi("적중률", "hit rate") in backtest_html
    assert i18n.bi("기준선(무조건 오른다고 할 때)", "baseline (always saying up)") in backtest_html
    assert i18n.bi("고점 대비 하락폭", "drop from the peak") in backtest_html
    assert backtest_html.count("2020-03-23") == 1                     # 셀 값은 한 번뿐


def test_every_column_of_the_fixed_tables_has_a_plain_label():
    """자주 쓰는 열은 모두 쉬운 이름을 갖는다 — 하나라도 빠지면 원문 키가 화면에 튀어나온다."""
    needed = ["tone", "h", "n", "n_blocks", "hit", "base", "edge", "ci_lo", "ci_hi", "fwd_mean", "fwd_median",
              "fwd_p10", "fwd_p90", "dd5_rate", "peak_date", "trough_date", "depth", "days_to_trough",
              "recovery_date", "days_to_recover", "first_warn_date", "warn_tone", "lead_days", "held_to_trough",
              "detected", "missed", "overall_m", "overall_w", "overall_d", "share", "rule",
              "n_outcome_20", "hit_20", "base_20", "hit_60", "base_60", "detection_rate", "n_episodes",
              "median_lead_days", "false_alarms_per_year", "lookback", "false_alarm_rate",
              "false_alarm_rate_baseline", "null", "cagr", "maxdd", "cagr_bh", "maxdd_bh", "cost_bps"]
    missing = [k for k in needed if k not in report_v0._LB]
    assert missing == [], f"쉬운 열 이름이 없는 키: {missing}"


# ==================================================================
# 2. 토글이 두 페이지에 닿는다 (STYLE_I18N.md §1)
# ==================================================================
def test_toggle_and_narrow_screen_rule_reach_both_pages(pages):
    """자바스크립트 0줄 · 페이지당 체크박스 하나 · 좁은 화면에서 헤더와 겹치지 않는다."""
    for name, html in pages.items():
        assert html.count('id="lang-sw"') == 1, name
        assert "<script" not in html, name
        assert "@media(max-width:640px)" in html and "position:static" in html, name
        assert html.index('id="lang-sw"') < html.index('class="wrap"'), name


def test_report_v0_does_not_wire_the_toggle_itself():
    """토글 배선은 껍데기 한 곳뿐이다 — 페이지 모듈이 또 심으면 개수가 어긋난다."""
    src = (ROOT / "mrl" / "report_v0.py").read_text(encoding="utf-8")
    assert "LANG_TOGGLE_HTML" not in src and "lang-sw" not in src


# ==================================================================
# 3. 쉬운 한국어 (STYLE_I18N.md §2 · §3)
# ==================================================================
def test_no_hard_term_is_left_in_our_own_sentences(pages):
    """대응표의 '원문' 표현이 우리가 쓴 문장에 남아 있으면 실패(괄호 안에 남긴 용어는 허용)."""
    for name, html in pages.items():
        assert _hard_terms(html) == {}, f"{name}: 어려운 말이 그대로 남았습니다 → {_hard_terms(html)}"


def test_the_glossary_wording_actually_appears(pages):
    """바꾸라고 정해 둔 쉬운 말을 실제로 쓰는지 — '안 쓰기'만 하고 '쓰기'를 안 하면 뜻이 사라진다."""
    ko = _ko_text(pages["index"]) + " " + _ko_text(pages["backtest_v0"])
    for plain in ("신호등 판정", "하락 사건", "평소", "고점 대비 하락폭", "무작위로 돌려"):
        assert plain in ko, f"쉬운 표현 '{plain}' 이 안 보입니다"


# 120자를 넘겨도 좋은 문장 — 하나하나 이유가 있고 **줄어들기만 한다**(§2.3).
# 예전에는 '식별자가 들었으면 통과 · 숫자가 3개 이상이면 통과' 라는 조건부 면제였는데, 그 두 조건이
# 이 프로젝트의 거의 모든 긴 문장을 덮어 버려 검사가 사실상 꺼져 있었다. 이제는 이름으로만 면제한다.
# 길이를 줄여야 하면 숫자를 지우지 말고 **문장을 나눈다**(정직성 > 길이).
V0_LONG_SENTENCES_ALLOWED = (
    ("주식은 원래 오르는 날이 더 많습니다", "사전 등록 전제 — 세 지평의 비율과 겹치지 않는 창 수를 한 줄에 둔다."),
    ("숫자와 조건은 하나도 바꾸지 않았고", "VALIDATION §0 원문을 그대로 옮긴 약속 문장."),
    ("경고는 고점 앞 최대 20거래일", "리드타임의 상한과 그 상한이 만드는 편향을 한 호흡에 밝힌다."),
    ("신호등 판정 → 주식 비중", "다섯 판정의 비중 값과 체결·비용 규약 — 쪼개면 규약이 흩어진다."),
    ("P7(주도주)", "워치리스트 종목 수의 연도별 변화와 생존편향 고지."),
    ("되살리기", "재현 규약 — 창 정의와 세션 수를 원문 그대로 남긴다."),
)


def test_sentences_stay_short_enough_to_read(pages):
    """한 문장 = 한 가지(§2.3). 목록에 없는 긴 문장은 실패시킨다."""
    heads = tuple(h for h, _ in V0_LONG_SENTENCES_ALLOWED)
    long_ones = []
    for name, html in pages.items():
        for lang, inner in i18n._spans(html):
            if lang != "ko":
                continue
            for sent in re.split(r"(?<=다)\.\s|(?<=다)\.$|[!?]\s", i18n.visible_text(inner)):
                body = sent.strip()
                if body and len(body) > 120 and not any(h in body for h in heads):
                    long_ones.append((name, len(body), body[:70]))
    assert long_ones == [], f"목록에 없는 긴 문장 {len(long_ones)}개: {long_ones[:3]}"


def test_signal_names_are_plain_and_bilingual(index_html):
    """P1~P9 는 이름만이 아니라 '무엇을 보는지' 한 줄이 함께 있어야 한다."""
    for key in V0_SIGNALS:
        ko, en = report_v0._SIGNAL_BI[key]
        assert ko.startswith("P") and len(ko) > 8 and en.startswith("P")
        assert i18n.bi(ko, en) in index_html, key
    assert "P8" in i18n.visible_text(index_html)                      # 번호가 비어 있는 이유를 밝힌다


# ==================================================================
# 4. 카드 첫 화면 형식 (STYLE_I18N.md §4)
# ==================================================================
def test_daily_card_follows_the_four_line_template(index_html):
    """① 무엇을 말하는 칸인지 ② 오늘 값 ③ 읽는 법 ④ 범위·한계 + 투자 조언 아님."""
    txt = i18n.visible_text(index_html)
    assert "오늘 v0 규칙이 내린 한 줄 판정" in txt                    # ①
    assert "단기 과열 뒤 주춤" in txt                                  # ② (자료 그대로)
    assert "종합 점수를 100배 한 값" in txt                            # ③ 읽는 법
    assert "20거래일·60거래일 뒤에 채점" in txt                        # ④ 언제 채점되는지
    assert "무조건 오른다” 는 기준선".replace(" ", "") in txt.replace(" ", "")
    assert "시험 운용" in txt and "투자 조언이 아닙니다" in txt
    en = _en_text(index_html)
    assert "not investment advice" in en and "trial run" in en and "baseline" in en


def test_the_card_compares_today_with_the_track_record_and_keeps_the_sample_size(index_html):
    """§4 셋째 줄(평소와 비교) — 적중률·기준선·표본 수·독립 관측 수를 한꺼번에 밝힌다."""
    ko, en = _ko_text(index_html), _en_text(index_html)
    assert "장부에 적히는 것은 completed 쪽 판정입니다" in ko
    assert "150일" in ko and "120일" in ko and "68.0%" in ko and "66.0%" in ko    # 표본·적중·기준선
    assert "독립된 관측은 약 6개뿐" in ko and "크게 흔들립니다" in ko              # 겹치는 창 경고
    assert "about 6 independent observations" in en and "still move a lot" in en


def test_the_card_says_nothing_when_the_log_has_not_scored_anything(tmp_path):
    """채점된 결과가 없으면 비교 문장을 아예 쓰지 않는다(없는 성적을 말하지 않는다)."""
    assert report_v0._track_line({"completed": {"tone": "hold"}}, {}) == ""
    assert report_v0._track_line({"completed": {"tone": "hold"}},
                                 {"by_tone": {"hold": {"n": 2, "n_outcome_20": 0, "hit_20": None}}}) == ""
    out = tmp_path / "index.html"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report.render_index(TODAY, {"n": 2, "by_tone": {"hold": {"n": 2, "n_outcome_20": 0}}}, out)
    assert "장부에 적히는 것은" not in out.read_text(encoding="utf-8")


def test_rates_are_also_given_as_natural_frequencies(index_html):
    """§2.2 비율은 자연빈도로 — '10번 중 약 N번' 이 백분율과 함께 나온다."""
    assert report_v0._ten_of(0.63).startswith('<span class="lg ko">10번 중 약 6번')
    assert "(63.0%)" in report_v0._ten_of(0.63)
    assert report_v0._ten_of(None) == "" and report_v0._ten_of(1.7) == ""
    assert "10번 중 약 7번" in _ko_text(index_html) and "about 7 times in 10" in _en_text(index_html)


def test_verdict_english_comes_from_the_v0_rule_itself(index_html):
    """판정 문구의 영어는 v0 규칙이 들고 있는 문장 그대로 — 새로 지어내지 않는다."""
    assert i18n.bi("단기 과열 뒤 주춤", "Stalling after short-term overheating") in index_html
    assert i18n.bi("꾸준한 상승 흐름", "Steady uptrend") in index_html


def test_verdict_falls_back_to_korean_when_the_rule_disagrees(tmp_path):
    """상태 조합과 문구가 어긋나면 영어를 붙이지 않는다(모르면 지어내지 않는다) — 한국어 한 벌만 남는다."""
    odd = {**DAY, "verdict_ko": "직접 적어 넣은 문구"}
    assert report_v0._verdict_en(odd) is None
    out = tmp_path / "index.html"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report.render_index({"asof": "2026-09-04", "faithful": odd}, {}, out)
    html = out.read_text(encoding="utf-8")
    assert "직접 적어 넣은 문구" in html
    assert _check_pairs(html)[2] == []                                 # 짝은 여전히 맞는다


# ==================================================================
# 5. 정직성 — 쉬운 말로 바꾸면서 무엇도 잃지 않았다
# ==================================================================
def test_pre_registered_honesty_paragraph_keeps_every_number(backtest_html):
    """VALIDATION.md §0 문단의 숫자는 하나도 빠지지 않는다(한국어·영어 양쪽)."""
    txt = i18n.visible_text(backtest_html)
    ko, en = _ko_text(backtest_html), _en_text(backtest_html)
    numbers = sorted({n for b in HONESTY_BULLETS for n in re.findall(r"\d+\.\d+|\d+", b)}, key=len, reverse=True)
    assert len(numbers) > 15
    for n in numbers:
        assert n in ko, f"쉬운 한국어에서 숫자 {n} 이 사라졌습니다"
        assert n in en, f"영어판에서 숫자 {n} 이 사라졌습니다"
    assert "8,458거래일" in txt and "정직한 전제" in txt
    assert HONESTY_TITLE in txt
    assert HONESTY_BOLD in backtest_html.replace("&quot;", '"')         # 굵은 결론은 원문 그대로
    for word in ("out-of-sample", "AUC", "α=5%"):
        assert word in ko, word                                        # 근거 용어는 괄호로 남긴다


def test_the_original_pre_registered_wording_is_still_carried_by_the_page(backtest_html):
    """쉬운 말로 옮겼다는 사실과 원문의 소재를 밝히고, 원문 자체도 페이지 안에 남긴다."""
    assert "VALIDATION.md §0" in i18n.visible_text(backtest_html)
    assert "쉬운 말로 옮긴 것" in i18n.visible_text(backtest_html)
    for bullet in HONESTY_BULLETS:                                     # tooltip 에 사전 등록 원문 그대로
        assert bullet.replace('"', "&quot;") in backtest_html


def test_the_random_shift_baseline_disclosure_survives_in_both_languages(backtest_html):
    """탐지율·리드타임을 무작위 이동 기준선과 나란히 읽으라는 경고는 두 언어에 모두 남는다."""
    ko, en = _ko_text(backtest_html), _en_text(backtest_html)
    assert "무작위로 돌려" in ko and "구별되지 않으면" in ko
    assert "randomly rotating" in en or "random starting point" in en
    assert "indistinguishable" in en
    assert "경고를 무작위로 돌려놨을 때" in ko                          # ① 카드에도 기준선 값이 붙는다
    assert "68.0%" in i18n.visible_text(backtest_html)                 # 그 값 자체


def test_sample_sizes_intervals_and_caps_survive(backtest_html):
    """표본 수·구간·탐색 창 상한·겹치는 창 경고 — 쉬운 말로 바뀌되 사라지지 않는다."""
    ko, en = _ko_text(backtest_html), _en_text(backtest_html)
    assert "겹치지 않는 창" in ko and "non-overlapping windows" in en
    assert "95% 범위" in ko and "95% range" in en
    assert "블록 부트스트랩" in ko                                      # 용어는 괄호 안에 남긴다
    assert "lookback" in ko and "lead_capped" in ko and "median_lead_days_fresh" in ko
    assert "창이 서로 겹치기 때문에 표본 수는 실제보다 크게 보입니다" in ko
    assert "12회 수준" in ko or "12회" in ko
    assert "about 12 declines" in en


def test_substitution_rules_are_kept_in_full(backtest_html):
    """VALIDATION.md §3 대체 규약 4줄은 한 줄도 빠지지 않고, 영어판도 함께 실린다."""
    txt = i18n.visible_text(backtest_html)
    for rule in SUBSTITUTION_RULES:
        assert rule.replace('"', "“").replace('"', "”") in txt or rule in txt, rule[:24]
    en = _en_text(backtest_html)
    assert "Survivor bias" in en and "BACKTEST_START" in en and "2020-08-03" in en


def test_tone_exposure_percentages_and_cost_survive(backtest_html):
    """톤→비중(100/100/100/50/25%)과 전환 비용 5bp 는 두 언어 모두에 남는다."""
    ko, en = _ko_text(backtest_html), _en_text(backtest_html)
    for _, e in TONE_EXPOSURE.items():
        assert f"{int(round(e * 100))}%" in ko and f"{int(round(e * 100))}%" in en
    assert "5bp" in ko and "5bp" in en


def test_v0_frozen_line_and_not_investment_advice_are_on_both_pages(pages):
    """v0 영구 표기와 '투자 조언 아님' 은 두 페이지·두 언어에 모두 있다."""
    for name, html in pages.items():
        ko, en = _ko_text(html), _en_text(html)
        assert "투자 조언이 아닙니다" in ko, name
        assert "not investment advice" in en, name
    assert "얼려 둔 기준" in _ko_text(pages["backtest_v0"])
    assert "frozen benchmark" in _en_text(pages["backtest_v0"])


def test_the_honest_line_and_kill_rule_note_survive_on_the_index(index_html):
    """정직 문구(사전 등록)와 킬 규칙 안내는 index 에서도 원문 그대로 + 영어판."""
    assert HONESTY_LINE in index_html.replace("&quot;", '"')
    assert HONESTY_BOLD in index_html.replace("&quot;", '"')
    ko, en = _ko_text(index_html), _en_text(index_html)
    assert "성적이 나쁘면 끄는 규칙" in ko and "VALIDATION.md" in ko
    assert "switch-it-off rule" in en and "information only" in en


def test_warnings_are_shown_verbatim_and_said_to_be_verbatim(pages):
    """계산층이 남긴 경고문은 요약하거나 지우지 않는다 — 원문 그대로 싣고 그 사실을 밝힌다."""
    assert "SPY 마지막 봉은 장중 부분 봉일 수 있음" in pages["index"]
    assert "아직 결과(20일)가 확정된 행이 없습니다" in pages["index"]
    assert "P2: 홀드아웃 미해제 → 라이브 채점 보류" in pages["index"]
    assert "캐시 경고: 마지막 SPY 봉은 장중 부분 봉 → 재현·채점에서 제외됨" in pages["backtest_v0"]
    assert "한국어 원문 그대로" in _ko_text(pages["index"])
    assert "kept in the original Korean" in _en_text(pages["index"])


def test_ledger_keeps_the_baseline_next_to_every_hit_rate(index_html):
    """적중률 옆에는 언제나 기준선이 있다(장부 카드·표 모두)."""
    ko = _ko_text(index_html)
    assert "비교 기준: 무조건 오른다고 했을 때 (20거래일)" in ko
    assert "언제나 기준선과 나란히" in ko
    txt = i18n.visible_text(index_html)
    assert "63.0%" in txt and "66.0%" in txt                           # 적중률과 기준선이 나란히
    assert "기록이 빠진 거래일" in ko                                   # 결측일 공개도 유지


# ==================================================================
# 6. 영어 화면에 남는 한국어는 '자료'뿐 (라벨은 전부 번역됐다)
# ==================================================================
def test_english_view_keeps_korean_only_where_the_data_itself_is_korean(index_html):
    """영어로 바꿨을 때 남는 한글은 계산층이 만든 자료뿐이어야 한다 — 라벨·문장이 남으면 실패."""
    view = _en_view(index_html)
    allowed = [TODAY["warnings"][0], LEDGER["warnings"][0], LEDGER["missing_method"],
               "P2: " + LEDGER["p2"]["notes"][0],
               "매수", "보유", "관망", "주의", "축소",                  # 톤 알약(report_common._tone_pill)
               "예", "아니오", "한국어", "English"]
    for s in allowed:
        view = view.replace(s, " ")
    left = re.findall(r"[가-힣][가-힣 ]{2,}", view)
    assert left == [], f"영어 화면에 한국어 라벨이 남았습니다: {left[:5]}"


# ==================================================================
# 7. 다른 테스트가 기대는 문구 (표현을 바꿔도 계약은 지킨다)
# ==================================================================
def test_strings_other_tests_depend_on_are_still_present(pages, tmp_path):
    """다른 테스트·스크립트가 찾는 문구를 문서로 남긴다 — 표현을 다듬을 때 함께 확인하라는 뜻."""
    idx, bt = pages["index"], pages["backtest_v0"]
    for s in ("오늘 판정", "장부 요약", "정직 문구", "자료", "기록 일수", "마감봉 없음", "faithful", "completed"):
        assert s in i18n.visible_text(idx), s
    assert "두 변형의 톤이 다릅니다" in idx                            # 예전 표현은 tooltip 으로 보존
    assert "정직한 전제" in bt and "8,458거래일" in bt
    for i in range(1, 10):
        assert f'id="s{i}"' in bt


def test_empty_inputs_still_render_balanced_pages(tmp_path):
    """자료가 하나도 없어도 페이지는 그려지고 짝은 맞는다(예외·구멍 없음)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report.render_index({}, {}, tmp_path / "i.html")
        report.render_backtest_report({}, tmp_path / "b.html", {})
    for name in ("i", "b"):
        html = (tmp_path / f"{name}.html").read_text(encoding="utf-8")
        n_ko, n_en, problems = _check_pairs(html)
        assert problems == [] and n_ko == n_en > 10, f"{name}: {problems[:3]}"
        assert _hard_terms(html) == {}, f"{name}: {_hard_terms(html)}"


def test_status_labels_have_both_languages_and_keep_the_old_keys():
    """장 상태 라벨은 두 언어를 갖고, 예전 이름(_STATUS_KO)도 그대로 남는다(파사드 호환)."""
    assert set(report_v0._STATUS_KO) == set(report_v0._STATUS_BI)
    assert set(report_v0._STATUS_BI) >= {"current", "intraday", "pre_open", "weekend", "holiday", "stale", "incomplete"}
    for key, (ko, en) in report_v0._STATUS_BI.items():
        assert re.search(r"[가-힣]", ko) and not re.search(r"[가-힣]", en), key
