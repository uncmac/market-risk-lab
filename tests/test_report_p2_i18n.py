# -*- coding: utf-8 -*-
"""주간 두 페이지의 쉬운 한국어 + 한/영 토글 검사 (STYLE_I18N.md §5).

대상: docs/calibration_p2.html (render_calibration_report) · docs/backtest_v1.html (render_backtest_v1).

검사하는 것
  1. 한/영 스팬 개수가 같다 — 한쪽만 심으면 언어를 바꿨을 때 문장이 사라진다.
  2. 영어 칸에 한글이, 한국어 칸에 영어 문장이 남아 있지 않다(상태 이름·설정 이름 같은 고유 식별자는 뺀다).
  3. 토글 CSS 와 좁은 화면(≤640px) 규칙이 두 페이지에 모두 닿는다.
  4. 용어 대응표(STYLE_I18N.md §3)의 어려운 말이 **우리가 쓴 문장**에 남아 있지 않다.
     계산이 만든 한국어(판정 원문·경고·규칙 문자열)는 <span class="raw"> 로 **그대로 인용**하므로 이 검사에서 뺀다 —
     번역하거나 요약하면 무엇으로 판정했는지가 사라지기 때문이다. 대신 그 옆에는 언제나 쉬운 말 설명이 붙는다(아래 6·7).
  5. 정직성: 숫자·표본 수·구간·post hoc 고지·#2 사전 관측·info_only·블록 민감도 불일치가 **두 언어 모두**에 남아 있다.
  6. 다른 페이지가 함께 쓰는 helper(_p2_table·_p2_kv·_v0_line_html 기본값)는 건드리지 않았다.

실제 산출물(results/·docs/)은 건드리지 않는다(tmp_path 만).
"""
from __future__ import annotations

import importlib.util
import re
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrl import i18n, report, report_p2                              # noqa: E402


def _load_fixtures():
    """tests/test_report_p2.py 의 합성 산출물을 그대로 쓴다(같은 자료로 검사해야 비교가 된다)."""
    spec = importlib.util.spec_from_file_location("_p2_fixtures", ROOT / "tests" / "test_report_p2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FIX = _load_fixtures()

# 화면에 그대로 두는 라틴 낱말 — 상태 이름·설정 이름·산출물 파일 이름·식별자 (STYLE_I18N.md §5 화이트리스트)
PAGE_LATIN = {
    "normal", "caution", "reduce", "hold", "buy", "neutral", "tones", "info_only", "deploy",
    "wide", "symmetric_dwell", "no_dwell", "default", "dwell", "clim", "literal", "amended",
    "allocation", "sensitivities", "summary_p", "summary_v", "json", "csv", "acceptance",
    "reliability", "resolution", "uncertainty", "murphy", "qlike", "mse", "log", "rv", "ln_rv",
    "platt", "wilson", "spec_sha", "feature_rule", "window_rule", "prob", "rung", "step",
}

RAW = re.compile(r'<span class="raw[^"]*">.*?</span>', re.S)


@pytest.fixture()
def latin_ok(monkeypatch):
    """고유 식별자를 한국어 칸의 예외로 넣는다(계약 §5 의 '티커·고유명사 화이트리스트')."""
    monkeypatch.setattr(i18n, "ALLOWED_LATIN", set(i18n.ALLOWED_LATIN) | PAGE_LATIN)
    return PAGE_LATIN


def _calibration(tmp_path: Path, **over) -> str:
    s = FIX._summary_p2()
    s["v0_completed"] = FIX._summary_v0()
    s.update(over)
    out = tmp_path / "calibration_p2.html"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report.render_calibration_report(s, out, {})
    return out.read_text(encoding="utf-8")


def _backtest_v1(tmp_path: Path, **over) -> str:
    s = FIX._summary_v1()
    s.update(over)
    out = tmp_path / "backtest_v1.html"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report.render_backtest_v1(s, FIX._summary_v0(), out, {})
    return out.read_text(encoding="utf-8")


def _both(tmp_path: Path) -> dict[str, str]:
    return {"calibration_p2": _calibration(tmp_path), "backtest_v1": _backtest_v1(tmp_path)}


def _ko_text(html: str) -> str:
    """한국어 화면에서 읽히는 글자만(영어 스팬은 뺀다)."""
    return i18n.visible_text(re.sub(r'<span class="lg en">.*?</span>', " ", html, flags=re.S))


def _en_text(html: str) -> str:
    return i18n.visible_text(re.sub(r'<span class="lg ko">.*?</span>', " ", html, flags=re.S))


# ==================================================================
# 1. 두 언어가 짝을 이룬다
# ==================================================================
@pytest.mark.parametrize("name", ["calibration_p2", "backtest_v1"])
def test_both_languages_are_planted_in_equal_numbers(name, tmp_path):
    html = _both(tmp_path)[name]
    n_ko, n_en = i18n.count_pairs(html)
    assert n_ko == n_en and n_ko > 150, f"{name}: ko={n_ko}, en={n_en} (bi() 로 두 벌을 함께 심어야 합니다)"


@pytest.mark.parametrize("name", ["calibration_p2", "backtest_v1"])
def test_no_language_leaks_into_the_other_column(name, tmp_path, latin_ok):
    html = _both(tmp_path)[name]
    n_ko, n_en, problems = i18n.check_pairs(html)
    assert problems == [], f"{name}: " + " | ".join(problems[:5])


@pytest.mark.parametrize("name", ["calibration_p2", "backtest_v1"])
def test_empty_summaries_still_render_both_languages(name, tmp_path, latin_ok):
    """자료가 하나도 없어도 '자료 없음' 까지 두 언어로 나온다(빈 페이지가 한국어만 남지 않는다)."""
    out = tmp_path / f"{name}.html"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "calibration_p2":
            report.render_calibration_report({"v0_completed": {}}, out, {})
        else:
            report.render_backtest_v1({}, {}, out, {})
    html = out.read_text(encoding="utf-8")
    n_ko, n_en, problems = i18n.check_pairs(html)
    assert (n_ko == n_en) and n_ko > 50 and problems == [], f"{name}: ko={n_ko} en={n_en} {problems[:3]}"
    ko, en = _ko_text(html), _en_text(html)
    assert ko.count("없음") >= 3, "빈 자리를 한국어로 알리지 않았습니다"
    assert len(re.findall(r"\bno\b", en)) >= 3, "빈 자리를 영어로 알리지 않았습니다"


# ==================================================================
# 2. 토글이 두 페이지에 닿는다 (STYLE_I18N.md §1)
# ==================================================================
@pytest.mark.parametrize("name", ["calibration_p2", "backtest_v1"])
def test_toggle_css_and_narrow_screen_rule_reach_the_page(name, tmp_path):
    html = _both(tmp_path)[name]
    assert html.count('id="lang-sw"') == 1 and "<script" not in html
    for rule in ("#lang-sw:checked ~ * .lg.ko{display:none}", "#lang-sw:checked ~ * .lg.en{display:inline}"):
        assert rule in html, f"{name}: 없는 규칙 {rule}"
    m = re.search(r"@media\(max-width:640px\)\{[^}]*\{[^}]*\}", html)
    assert m and "position:static" in m.group(0), f"{name}: 좁은 화면에서 토글이 헤더와 겹칩니다"


@pytest.mark.parametrize("name", ["calibration_p2", "backtest_v1"])
def test_page_title_is_english_only(name, tmp_path):
    """<title> 은 영어 한 벌 (STYLE_I18N.md §1 — 스팬을 넣을 수 없는 자리)."""
    html = _both(tmp_path)[name]
    title = re.search(r"<title>(.*?)</title>", html).group(1)
    assert title and not re.search(r"[가-힣]", title), f"{name}: {title!r}"
    assert "market-risk-lab" in title


# ==================================================================
# 3. 어려운 용어가 남지 않았다 (STYLE_I18N.md §3)
# ==================================================================
@pytest.mark.parametrize("name", ["calibration_p2", "backtest_v1"])
def test_no_glossary_jargon_is_left_in_our_own_prose(name, tmp_path):
    """우리가 쓴 문장에는 대응표의 '원문' 표현이 남지 않는다(괄호 안에 한 번 남기는 것은 허용)."""
    html = _both(tmp_path)[name]
    found = i18n.find_hard_terms(RAW.sub(" ", html))
    assert found == {}, f"{name}: 쉬운 말로 바꾸지 않은 표현 {found}"


@pytest.mark.parametrize("name", ["calibration_p2", "backtest_v1"])
def test_hard_terms_that_stay_are_only_inside_parentheses(name, tmp_path):
    """열 이름·예전 표기처럼 남겨야 하는 용어는 괄호 안에만 있다 — 괄호를 무시하면 잡혀야 한다."""
    html = _both(tmp_path)[name]
    strict = i18n.find_hard_terms(RAW.sub(" ", html), allow_in_parens=False)
    assert strict, f"{name}: 괄호 안에도 용어가 하나도 없다면 대조가 사라진 것입니다"


def test_machine_written_korean_is_quoted_verbatim_not_translated(tmp_path):
    """계산이 만든 한국어(판정 원문·경고·규칙 문자열)는 손대지 않고 그대로 인용한다."""
    html = _calibration(tmp_path)
    acc = FIX._summary_p2()["acceptance"]
    raw_blocks = " ".join(RAW.findall(html))
    for text in (acc["rationale"], acc["post_hoc_note"], acc["literal"]["M3"]["rule_text"], "합성 자료 — 테스트용"):
        assert i18n.esc(text) in raw_blocks, f"원문 인용이 사라졌습니다: {text[:40]!r}"


# ==================================================================
# 4. 정직성 — 쉬운 말로 바꾸면서 무엇도 빠지지 않았다
# ==================================================================
def test_acceptance_verdict_survives_verbatim_and_in_both_languages(tmp_path):
    """§6 판정: (1) 계산이 만든 한 줄 그대로 (2) 같은 사실을 쉬운 한국어로 (3) 영어로 — 셋 다 있어야 한다."""
    acc = dict(FIX._summary_p2()["acceptance"])
    acc.update({"require_tables": ["24", "18"], "primary_table": "24",
                "per_table": {"24": {"tone_model": "M1", "deploy_mode": "tones", "passing_rungs": ["M1"], "required": True},
                              "18": {"tone_model": None, "deploy_mode": "info_only", "passing_rungs": [], "required": True}},
                "tables": {"24": {"literal": acc["literal"], "amended": acc["amended"]},
                           "18": {"literal": acc["literal"], "amended": {"M1": {"A": False, "B": True, "C": True, "pass": False,
                                                                               "min_block_bss_clim": -0.0996}}}},
                "rule": "amended", "tone_model": None, "deploy_mode": "info_only", "blocking_tables": ["18"],
                "conjunction_note": "배치는 require_tables ['24', '18'] 전부에서 통과한 단만 — 소유자 결정 2026-09-08(장부 #2d)",
                "verdict_line": "24개월 표: M1 통과 · 18개월 표: 실패(최소 블록 BSS_clim −0.0996) → 두 표 모두 통과 요구"
                                "(소유자 결정 2026-09-08) → 배치 없음(정보 제공 전용)"})
    html = _calibration(tmp_path, acceptance=acc)
    ko, en = _ko_text(html), _en_text(html)

    assert i18n.esc(acc["verdict_line"]) in html                      # (1) 원문 한 글자도 바꾸지 않는다
    assert acc["verdict_line"] in " ".join(RAW.findall(html)).replace("&#x27;", "'").replace("&amp;", "&")

    for text in (ko, en):                                             # (2)(3) 두 화면 모두 같은 사실을 말한다
        assert "24" in text and "18" in text and "M1" in text
        assert "-0.0996" in text or "−0.0996" in text
        assert "2026-09-08" in text
    assert "떨어짐" in ko and "가장 나쁜 구간 점수" in ko
    assert "fails" in en and "worst block score" in en
    assert "정보 제공 전용(info_only)" in ko and "information only" in en
    assert "막은 표" in ko and "blocked by" in en


def test_block_sensitivity_disagreement_survives_in_both_languages(tmp_path):
    """§8.2 '구간을 다르게 자르면 결론이 달라진다' 는 두 언어 모두에 그대로 남는다 — 판정을 좋게 보이게 하려고 지우지 않는다."""
    acc = dict(FIX._summary_p2()["acceptance"])
    acc["sensitivity_blocks"] = {"1999": {"agrees": False, "deploy_mode": "tones", "tone_model": "M1",
                                          "note": "§8.2 블록 민감도 — 1999 시작 표는 배치 판정에 쓰지 않는다"}}
    html = _calibration(tmp_path, acceptance=acc)
    ko, en = _ko_text(html), _en_text(html)
    assert "구간을 어떻게 자르냐에 따라 결론이 달라집니다" in ko and "1999 시작" in ko and "tones" in ko and "M1" in ko
    assert "판정에 쓰지 않습니다" in ko
    assert "depends on how the blocks are cut" in en and "from 1999" in en and "not used for the verdict" in en
    assert "결론이 다름" in ko and "disagrees" in en                    # 머리 태그에도 경고가 붙는다


def test_post_hoc_and_pre_observation_disclosures_survive(tmp_path):
    """'결과를 본 뒤에 정했다'(post hoc)와 '#2 사전 관측' 고지는 두 언어 모두에 남는다."""
    html = _calibration(tmp_path)
    ko, en = _ko_text(html), _en_text(html)
    assert "#2" in ko and "#2" in en
    assert "설계 단계에서 이미 본" in ko or "설계 단계에서 이미 본 것입니다" in ko
    assert "already seen while the model was being designed" in en
    assert "결과를 본 뒤" in ko and "decided after seeing the results" in en
    assert "#2a 완화안 (post hoc)" in ko                                # 표 제목은 예전 이름 그대로 남긴다
    assert "#2a relaxed version" in en
    assert i18n.esc("#2 사전 관측 참조") in html                        # 산출물의 공개 문구 그대로


def test_info_only_status_is_never_softened(tmp_path):
    """배치되지 않았다는 사실은 두 언어 모두에서 분명해야 한다 — 쉬운 말로 바꾸며 '쓰고 있다' 로 들리면 안 된다."""
    html = _calibration(tmp_path)
    ko, en = _ko_text(html), _en_text(html)
    assert "정보 제공 전용(info_only)" in ko and "실제로 쓰는 단계 없음" in ko
    assert "information only, nothing in use (info_only)" in en and "no step is in use" in en
    assert "신호등 판정에 사용(tones)" not in ko and "used for the traffic-light call (tones)" not in en

    v1 = _backtest_v1(tmp_path)
    ko1, en1 = _ko_text(v1), _en_text(v1)
    assert report_p2.INFO_DISPLAY_LABEL in ko1
    assert "실제로 쓰는 단계가 없어서" in ko1 and "톤·비중 주장이 아닙니다" in ko1
    assert "no step is actually in use" in en1 and "not a claim about calls or about how much to hold" in en1


def test_sample_sizes_intervals_and_caveats_are_not_dropped(tmp_path):
    """표본 수(겹치지 않는 구간 수)·95% 범위·흔들림 폭 같은 한계 표시는 쉬운 말로 바뀌어도 사라지지 않는다."""
    html = _calibration(tmp_path)
    ko, en = _ko_text(html), _en_text(html)
    assert "겹치지 않는 구간 수 (n÷20)" in ko and "independent windows (n/20)" in en
    assert "독립 사례 수 (n÷20)" in ko and "independent cases (n/20)" in en
    assert "95%" in ko and "95%" in en
    assert "±0.1" in ko and "±0.1" in en                              # 구간당 사례가 적어 점수가 흔들린다는 경고
    assert "+0.05~0.09" in ko and "+0.05 to +0.09" in en
    assert "+13.6" in ko and "+13.6" in en and "−2.8" in ko and "−2.8" in en
    assert "5,453" in ko and "5,453" in en                            # 겹치는 라벨 경고
    for i, t in enumerate(report_p2.P2_HONESTY_ITEMS):                # 여덟 항목 모두 두 언어로
        assert t[:24] in ko, t[:24]
        assert report_p2.P2_HONESTY_ITEMS_EN[i][:24] in en, report_p2.P2_HONESTY_ITEMS_EN[i][:24]


def test_backtest_v1_headline_numbers_and_comparisons_survive(tmp_path):
    """②의 성적 카드는 숫자·비교 대상(v0·평소 비율·상한)을 하나도 잃지 않는다."""
    html = _backtest_v1(tmp_path)
    ko, en = _ko_text(html), _en_text(html)
    assert "4.4" in ko and "4.4" in en                                 # 1년에 상태가 바뀐 횟수
    assert "KPI 상한 12 충족" in ko and "our ceiling 12 a year, respected" in en
    assert "30.0~42.0%" in ko and "30.0~42.0%" in en
    assert "정상 11.0%" in ko and "in the normal state 11.0%" in en
    assert "기저율 15.4%" in ko and "picking any day would give 15.4%" in en
    assert "v0 12.8%" in ko and "v0 12.8%" in en
    assert "2007-10-09" in ko and "2007-10-09" in en                   # 하락 사건 표는 두 화면 모두에
    assert "v1 결정층" in ko and "v0 톤" in ko                          # 다른 문서가 쓰는 예전 열 이름은 괄호로 남는다
    assert "the v1 decision rule" in en and "the v0 traffic-light calls" in en


def test_v0_permanent_line_keeps_every_number_in_both_languages(tmp_path):
    """VALIDATION §4 — v0 줄은 두 페이지 첫 줄에 남고, 숫자가 한 언어에만 있으면 안 된다."""
    v0 = report.v0_headline(FIX._summary_v0())
    for html in _both(tmp_path).values():
        ko, en = _ko_text(html), _en_text(html)
        assert "v0 (동결 벤치마크" in ko and "v0 (frozen benchmark" in en
        for num in (f"{v0['tone_switches_per_year']:,.1f}", f"{v0['cagr_strategy'] * 100:.1f}%",
                    f"{v0['maxdd_strategy'] * 100:.1f}%", f"{v0['true_alarm_share'] * 100:.1f}%"):
            assert num in ko and num in en, num
        assert "VALIDATION.md §4" in ko and "VALIDATION.md §4" in en


def test_warnings_are_shown_not_swallowed(tmp_path):
    """경고는 원문 그대로 싣고, 몇 건인지도 두 언어로 밝힌다."""
    html = _calibration(tmp_path)
    assert i18n.esc("합성 자료 — 테스트용") in html
    ko, en = _ko_text(html), _en_text(html)
    assert "경고 1건" in ko and "1 warnings" in en
    assert "번역하지 않고 그대로 싣습니다" in ko and "quoted as-is" in en


def test_deployed_case_keeps_the_old_column_names_but_stays_plain(tmp_path, latin_ok):
    """실제로 쓰는 단계가 있을 때도 (1) 다른 문서가 인용하는 예전 표기는 괄호로 남고 (2) 어려운 말은 남지 않는다."""
    html = _backtest_v1(tmp_path, deploy_mode="tones", tone_model="M1",
                        deployed={"prob_rung": "M1", "deployed": True, "label": report_p2.DEPLOYED_LABEL},
                        info_layers={"M3": {"kpis": {"switches_per_year": 4.9}, "allocation": {"cagr": 0.08}}})
    ko, en = _ko_text(html), _en_text(html)
    assert "헤드라인 <b>M1</b> 배포" in html                          # 다른 스크립트·문서가 이 표기를 인용한다
    assert "② 결정층 KPI — M1 배포" in ko
    assert "실제 사용 (배포)" in ko or "실제 사용 중" in ko
    assert "in use" in en and "M1" in en
    assert report_p2.INFO_DISPLAY_LABEL in ko                         # ⑥ 은 여전히 '쓰지 않는 단계' 라고 말한다
    n_ko, n_en, problems = i18n.check_pairs(html)
    assert n_ko == n_en and problems == []
    assert i18n.find_hard_terms(RAW.sub(" ", html)) == {}


def test_tones_case_of_the_calibration_page_stays_plain(tmp_path, latin_ok):
    acc = dict(FIX._summary_p2()["acceptance"])
    acc.update({"deploy_mode": "tones", "tone_model": "M1", "require_tables": ["24"],
                "per_table": {"24": {"tone_model": "M1", "deploy_mode": "tones", "passing_rungs": ["M1"], "required": True}}})
    html = _calibration(tmp_path, acceptance=acc)
    ko, en = _ko_text(html), _en_text(html)
    assert "신호등 판정에 사용(tones)" in ko and "used for the traffic-light call (tones)" in en
    assert "24개월 표 M1 통과" in ko and "24-month table M1 passes" in en
    n_ko, n_en, problems = i18n.check_pairs(html)
    assert n_ko == n_en and problems == []
    assert i18n.find_hard_terms(RAW.sub(" ", html)) == {}


# ==================================================================
# 5. 다른 페이지의 helper 는 건드리지 않았다
# ==================================================================
def test_shared_helpers_stay_korean_only_for_other_pages():
    """_p2_table·_p2_kv·_v0_line_html(기본값)은 index 카드와 Phase 3 페이지가 함께 쓴다 — 스팬을 심으면 그 페이지의 짝이 깨진다."""
    t = report_p2._p2_table([{"n": 20, "bss_clim": 0.1}])
    k = report_p2._p2_kv({"n": 20})
    v = report_p2._v0_line_html(report.v0_headline(FIX._summary_v0()))
    for html in (t, k, v):
        assert i18n.count_pairs(html) == (0, 0)
    assert "v0 (동결 벤치마크" in v and "톤 전환" in v                    # 예전 문구 그대로


def test_bilingual_helpers_do_not_touch_cell_values():
    """표는 열 제목만 두 언어로 — 셀 값(숫자·날짜·식별자)은 그대로 둔다 (STYLE_I18N.md §1)."""
    html = report_p2._bi_table([{"n_blocks": 25, "block": "2018-01-01"}])
    assert i18n.count_pairs(html) == (2, 2)                            # 열 제목 두 개만
    assert ">25<" in html and "2018-01-01" in html


def test_render_is_deterministic(tmp_path):
    """같은 입력이면 같은 HTML — 표현 계층이 무작위로 흔들리지 않는다."""
    a, b = _calibration(tmp_path / "a"), _calibration(tmp_path / "b")
    assert a == b
    c, d = _backtest_v1(tmp_path / "c"), _backtest_v1(tmp_path / "d")
    assert c == d


@pytest.mark.parametrize("name", ["calibration_p2", "backtest_v1"])
def test_real_artifacts_pass_the_same_checks(name, tmp_path, latin_ok):
    """실제 results/*.json 으로 그려도 같은 검사를 통과한다(합성 자료에만 맞춘 문구가 아니다)."""
    import json
    paths = {p: ROOT / "results" / p for p in ("summary_p2.json", "summary_v1.json", "summary_v0_completed.json")}
    if not all(p.exists() for p in paths.values()):
        pytest.skip("results/*.json 없음 — 실제 산출물 검사 생략")
    load = {k: json.loads(v.read_text(encoding="utf-8")) for k, v in paths.items()}
    out = tmp_path / f"{name}.html"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "calibration_p2":
            report.render_calibration_report({**load["summary_p2.json"], "v0_completed": load["summary_v0_completed.json"]}, out, {})
        else:
            report.render_backtest_v1(load["summary_v1.json"], load["summary_v0_completed.json"], out, {})
    html = out.read_text(encoding="utf-8")
    n_ko, n_en, problems = i18n.check_pairs(html)
    assert n_ko == n_en and problems == [], f"{name}: ko={n_ko} en={n_en} {problems[:3]}"
    assert i18n.find_hard_terms(RAW.sub(" ", html)) == {}


# ==================================================================
# 8. 일간 카드의 확률 귀속 블록 — 막대 이름·변화 문장이 두 언어로 나온다
#    (이 블록이 한국어만 남아 있던 것이 index.html 의 영어 화면 결함이었다)
# ==================================================================
def _p2_card_html() -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return report.p2_card(FIX._today_p2())


def test_attribution_block_is_bilingual_in_the_daily_card():
    """'이 확률이 어떻게 만들어졌나' 막대와 변화 문장에 한국어만 남아 있으면 안 된다."""
    html = _p2_card_html()
    assert "p2bar" in html, "귀속 막대가 그려지지 않았습니다"
    en_doc = re.sub(r'<span class="lg ko">.*?</span>', " ", html, flags=re.S)   # 영어 화면
    en_doc = RAW.sub(" ", en_doc)                                # 원문 인용은 고지와 함께 그대로 둔다
    en_doc = re.sub(r"<td[^>]*>.*?</td>", " ", en_doc, flags=re.S)   # 셀 값은 번역하지 않는다(§1)
    i = en_doc.find("p2bar")
    en_only = i18n.visible_text(en_doc[i:])
    assert not re.search(r"[가-힣]", en_only), f"영어 화면에 한국어가 남았습니다: {en_only[:120]}"


def test_attribution_block_keeps_every_qualifier_in_korean():
    """쉬운 말로 바꾸면서 (재적합)·'N세션 전 대비'·재적합 배지를 잃지 않았다."""
    ko = _ko_text(_p2_card_html())
    assert "어제 대비" in ko and "5일 누적" in ko
    assert "(재적합 +0.0)" in ko                       # 괄호 안 항이 그대로 남는다
    assert "5세션 전 대비" in ko                       # 5일 누적 문장의 단서
    assert "실현-내재 갭" in ko and "추세" in ko        # 막대 이름


def test_attribution_block_keeps_every_qualifier_in_english():
    en = _en_text(_p2_card_html())
    assert "vs yesterday" in en and "5-day total" in en
    assert "(refit +0.0)" in en
    assert "compared with 5 sessions ago" in en
    assert "realised-implied gap" in en and "trend" in en


def test_attribution_block_says_it_out_loud_when_the_change_cannot_be_computed():
    """입력이 없으면 두 언어 모두에서 '계산 불가' 라고 말한다 — 조용히 지우지 않는다."""
    d = FIX._today_p2(dod={"d_pp": {"x_vix": float("nan")}, "d_p": float("nan"), "refit": False})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        html = report.p2_card(d)
    assert "변화 계산 불가(입력 결측)" in _ko_text(html)
    assert "cannot compute the change" in _en_text(html)
