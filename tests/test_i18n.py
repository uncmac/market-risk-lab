# -*- coding: utf-8 -*-
"""한/영 토글·쉬운 한국어 도구(mrl/i18n.py)와 리포트 모듈 분리(mrl/report_*.py) 테스트.

계약: STYLE_I18N.md
  §1 토글      숨긴 체크박스 #lang-sw 하나 + 알약 라벨, 자바스크립트 0줄, 좁은 화면(≤640px)에서 헤더와 겹치지 않음
  §1 bi()      한/영 두 벌을 함께 심고 화면엔 고른 언어만 보인다 · 속성 자리는 bi_attr 로 한국어를 고르고 기록
  §3 용어표    어려운 용어가 화면에 그대로 남으면 find_hard_terms 가 잡는다(괄호 안은 허용 — "먼저 뜻, 그다음 용어")
  §5 검사      한/영 스팬 개수 일치 · 영어 칸에 한글 없음 · 한국어 칸에 영어 문장 없음

분리 계약: mrl/report.py 는 얇은 파사드이고, 예전(HEAD, 4,273 줄) 이 내보내던 이름을 하나도 잃지 않는다.
정직성: 이 단계는 **표현 배선만** 한다. 문구를 아직 번역하지 않았으므로 페이지 본문의 용어 린트는
        강제하지 않고(다음 단계), 도구가 제대로 잡는지만 검사한다.
실제 산출물(results/·docs/)은 건드리지 않는다(tmp_path 만).
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrl import i18n, report                                        # noqa: E402
from mrl import report_common, report_p2, report_p3, report_v0      # noqa: E402

# HEAD(분리 직전 커밋 8def34c, mrl/report.py 4,273 줄) 가 모듈 수준에서 묶던 이름 219개 — 파사드는 이걸 다 가져야 한다.
HEAD_NAMES = (
    'BACKTEST_START', 'DECISION_P2', 'DEPLOYED_LABEL', 'ENSEMBLE_P3', 'FEATURE_COLORS', 'FEATURE_KO', 'HMM_P3',
    'HOLDOUT_START', 'HONESTY_BOLD', 'HONESTY_BULLETS', 'HONESTY_LINE', 'HONESTY_TITLE', 'INFO_DISPLAY_LABEL',
    'INFO_ONLY_LABEL', 'KILL_MANUAL_PATH', 'KILL_P3', 'KILL_RECORD_PATH', 'LABELS', 'MODEL_P2_PATH',
    'MODEL_P3_PATH', 'P2', 'P2_CHURN_LABEL', 'P2_FAMILY_LINE', 'P2_FOOTNOTE', 'P2_HONESTY_ITEMS',
    'P2_KILL_EPISODES', 'P2_KILL_MONTHS', 'P2_PROB_UNAVAILABLE', 'P2_STATES', 'P2_STATE_COLORS', 'P2_STATE_KO',
    'P3', 'P3_CARD_FOOTNOTE', 'P3_DRIFT', 'P3_FIRST_LINE', 'P3_HIGH_STATE_KO', 'P3_HONESTY_ITEMS',
    'P3_INFO_NOT_SIZING', 'P3_INFO_ONLY_LABEL', 'P3_LADDER_HIDDEN', 'P3_MEMBER_KO', 'P3_PARAM_LINE_FMT',
    'P3_PARAM_TABLE', 'P3_PARAM_TABLE_NOTE', 'P3_REASON_KO', 'P3_REGIME_CUTS', 'P3_REGIME_LEVELS',
    'P3_SHADOW_LABEL', 'P3_SIZING_SHORT', 'PALETTE', 'Path', 'RESULTS_DIR', 'RUNG_KO', 'RUNG_PARAMS',
    'SCENARIO_P3', 'SIGNAL_KO', 'SITE_URL', 'STATE_COLORS', 'STATE_LABEL', 'STATE_TO_TONE', 'SUBSTITUTION_RULES',
    'TABLE_KO_P2', 'TONES', 'TONE_COLORS', 'TONE_EXPOSURE', 'TONE_KO', 'V0_SIGNALS', 'VARIANT_KO', '_ALLOC_KEYS',
    '_BLOCK_ORDER', '_CSS', '_DEC4_KEYS', '_ERA_ORDER', '_HAR_ORDER', '_KO_DOW', '_LADDER_ORDER', '_NO_PCT',
    '_P2_ACC_CACHE', '_P3_ALARM_ORDER', '_P3_BIN_ORDER', '_P3_BT_ORDER', '_P3_LADDER_ORDER', '_P3_PCT',
    '_P3_PLAIN', '_P3_REGISTRY_ORDER', '_P3_STATE_ORDER', '_PARAM_ORDER', '_PCT_HINT', '_PCT_KEYS_P2',
    '_REGIME_SECTIONS', '_RELIAB_ORDER', '_RUNG_ORDER', '_SIZING_SECTIONS', '_STATUS_KO', '_TRACK_PANEL_TITLES',
    '_TRACK_SECTIONS', '_acceptance_box', '_acceptance_tables', '_acceptance_tags', '_alloc_matrix', '_band_of',
    '_bars_html', '_coef_txt', '_contrib_rows', '_cum', '_dd_of', '_directional_records', '_dod_line', '_dow_ko',
    '_esc', '_events_html', '_exact_pp_parts', '_find', '_first', '_fmt', '_fmt_p2', '_fmt_p3', '_fnum', '_h',
    '_h_of', '_honesty_strip', '_hz_block', '_hz_key', '_hz_reason', '_hz_redact', '_hz_stage', '_hz_val', '_img',
    '_info_layer_rows', '_is_nan', '_is_num', '_kpi', '_kv_table', '_label', '_ladder_records', '_ledger_block',
    '_legend', '_model_deploy_mode', '_nat_freq', '_new_fig', '_now_str', '_num_of', '_p2_frame', '_p2_kv',
    '_p2_state_pill', '_p2_table', '_p3_col', '_p3_frame', '_p3_head', '_p3_honesty_section', '_p3_kill_bits',
    '_p3_kv', '_p3_mode_banner', '_p3_mode_source', '_p3_nav', '_p3_table', '_p3_window_tables', '_panel_kv',
    '_param_records', '_pct1', '_pct_key', '_plain_float', '_plain_log_axis', '_png', '_pp1', '_records_table',
    '_rung_txt', '_scalar_ok', '_scenarios_spread_sentence', '_series_close', '_sizing', '_state_pill',
    '_style_ax', '_sub', '_table_ko_p2', '_ten_freq', '_thresholds_fallback', '_to_records', '_tone_pill',
    '_tone_runs', '_track', '_v0_line_html', '_variants_of', '_verdict_block', '_warn_list', '_write_html',
    'acceptance_verdict_line', 'annotations', 'base64', 'charts_for', 'charts_p2', 'charts_p3', 'datetime',
    'deployment_of', 'horizon_may_show', 'horizon_view', 'io', 'kill_power_line', 'load_acceptance_p2',
    'load_summary_v0', 'math', 'np', 'p2_acceptance_of', 'p2_card', 'p3_card', 'p3_effective_mode',
    'p3_param_line', 'p3_verdict_block', 'pd', 'render_backtest_report', 'render_backtest_v1',
    'render_calibration_report', 'render_index', 'render_regime_report', 'render_sizing_report',
    'render_track_record', 'timezone', 'v0_headline', 'warnings',
)

# 렌더 진입점 → 빈 입력으로 페이지 하나를 만드는 호출 (있는 키만 렌더링하는 설계라 {} 로도 그려진다)
PAGES = {
    "backtest_v0": lambda p: report.render_backtest_report({}, p, {}),
    "index": lambda p: report.render_index({}, {}, p),
    "calibration_p2": lambda p: report.render_calibration_report({}, p, {}),
    "backtest_v1": lambda p: report.render_backtest_v1({}, {}, p, {}),
    "sizing_p3": lambda p: report.render_sizing_report({}, {}, {}, p, {}),
    "regime_p3": lambda p: report.render_regime_report({}, p, {}),
    "track_record": lambda p: report.render_track_record({}, {}, p, {}),
}


@pytest.fixture()
def no_kill(monkeypatch, tmp_path):
    """킬 파일이 없는 상태를 고정(실제 results/ 상태에 흔들리지 않게)."""
    monkeypatch.setattr(report, "KILL_RECORD_PATH", tmp_path / "kill_record.json")
    monkeypatch.setattr(report, "KILL_MANUAL_PATH", tmp_path / "kill_manual.json")
    return tmp_path


def _render(name: str, tmp_path: Path) -> str:
    out = tmp_path / f"{name}.html"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        PAGES[name](out)
    return out.read_text(encoding="utf-8")


# ==================================================================
# 1. bi() 계열 — 두 언어를 함께 심는다
# ==================================================================
def test_bi_plants_both_languages_and_escapes():
    """bi() 는 한/영 두 벌을 심고, 둘 다 이스케이프한다(값이 마크업을 깨지 않는다)."""
    out = i18n.bi("평소 비율", "base rate")
    assert out == '<span class="lg ko">평소 비율</span><span class="lg en">base rate</span>'
    assert "&lt;b&gt;" in i18n.bi("<b>위험</b>", "<b>risk</b>")


def test_bi_html_keeps_markup_but_bi_does_not():
    """문장 안에 <b> 를 넣어야 할 때만 bi_html — 개수 검사는 둘 다 똑같이 통과한다."""
    out = i18n.bi_html("<b>8%</b> 확률", "<b>8%</b> chance")
    assert "<b>8%</b> 확률" in out
    assert i18n.count_pairs(out) == (1, 1)


def test_bi_attr_picks_korean_and_records_the_english():
    """속성 자리에는 스팬을 못 넣는다 → 한국어를 고르고, 버린 영어를 기록해 둔다(감사용)."""
    txt = i18n.bi_attr("확률 경로", "probability path")
    assert txt == "확률 경로"
    assert ("확률 경로", "probability path") in i18n.ATTR_KO_ONLY


def test_bilingual_table_headers():
    """표는 열 제목만 이중화한다 — 셀 값은 건드리지 않는다."""
    row = i18n.bi_ths([("확률 구간", "probability bin"), ("표본 수", "sample size")], num_from=1)
    assert row.startswith("<tr><th>") and row.endswith("</tr>")
    assert '<th class="num">' in row
    assert i18n.count_pairs(row) == (2, 2)


# ==================================================================
# 2. 토글 (STYLE_I18N.md §1)
# ==================================================================
def test_toggle_markup_is_checkbox_plus_pill_and_has_no_javascript():
    """자바스크립트 0줄 — 숨긴 체크박스와 라벨뿐이다(주피터 뷰어·오프라인에서도 동작)."""
    html = i18n.LANG_TOGGLE_HTML
    assert 'type="checkbox"' in html and 'id="lang-sw"' in html and 'for="lang-sw"' in html
    assert "한국어" in html and "English" in html
    assert "<script" not in html and "onclick" not in html


@pytest.mark.parametrize("name", sorted(PAGES))
def test_every_page_carries_exactly_one_toggle(name, tmp_path, no_kill):
    """일곱 페이지 모두 토글을 얻고, 페이지당 #lang-sw 는 정확히 하나다(중복이면 CSS 가 어긋난다)."""
    html = _render(name, tmp_path)
    n = html.count('id="lang-sw"')
    assert n == 1, f"{name}: #lang-sw 가 {n} 개 (페이지당 정확히 하나여야 CSS 가 맞는다)"
    assert html.count('for="lang-sw"') == 1
    assert "<script" not in html


@pytest.mark.parametrize("name", sorted(PAGES))
def test_toggle_sits_before_the_page_wrapper(name, tmp_path, no_kill):
    """체크박스는 본문 래퍼의 앞 형제여야 한다 — CSS ~ 결합자가 본문 언어를 바꾸는 근거."""
    html = _render(name, tmp_path)
    assert html.index('id="lang-sw"') < html.index('class="wrap"')


@pytest.mark.parametrize("name", sorted(PAGES))
def test_pages_emit_the_toggle_css(name, tmp_path, no_kill):
    """껍데기가 LANG_CSS 를 함께 내보낸다(가시성 규칙과 알약 스타일)."""
    html = _render(name, tmp_path)
    for rule in ("#lang-sw:checked ~ * .lg.ko{display:none}",
                 "#lang-sw:checked ~ * .lg.en{display:inline}",
                 ".lg.en{display:none}"):
        assert rule in html, f"{name}: 없는 규칙 {rule}"


def test_korean_is_the_default_language():
    """기본은 한국어: 체크 안 한 상태에서 영어 칸만 숨는다."""
    assert ".lg.en{display:none}" in i18n.LANG_CSS
    assert ".lg.ko{display:none}" not in i18n.LANG_CSS.split("#lang-sw:checked")[0]
    assert i18n.LANG_DEFAULT == "ko"


def test_narrow_screen_moves_the_pill_off_the_header():
    """≤640px 에서는 절대위치를 풀고 흐름 배치로 내린다 — market-brief 에서 겪은 겹침 문제."""
    m = re.search(r"@media\(max-width:640px\)\{([^}]*\{[^}]*\})*[^}]*\}", i18n.LANG_CSS)
    assert m, "좁은 화면 규칙이 없다"
    rule = m.group(0)
    assert "position:static" in rule, "절대위치를 풀지 않으면 헤더와 겹친다"
    assert "auto" in rule, "오른쪽 정렬(margin-left:auto)이 있어야 한다"


@pytest.mark.parametrize("name", sorted(PAGES))
def test_narrow_screen_rule_reaches_every_page(name, tmp_path, no_kill):
    html = _render(name, tmp_path)
    assert "@media(max-width:640px)" in html and "position:static" in html


# ==================================================================
# 3. 용어 린트 (STYLE_I18N.md §3)
# ==================================================================
def test_terms_table_matches_the_contract():
    """용어표는 STYLE_I18N.md §3 그대로 — 원문마다 쉬운 한국어와 영어가 짝으로 있다."""
    assert len(i18n.TERMS) == 29
    for src, pair in i18n.TERMS.items():
        assert isinstance(pair, tuple) and len(pair) == 2
        ko, en = pair
        assert ko and en and ko != src
    assert i18n.TERMS["기저율"][0] == "평소 비율"
    assert i18n.TERMS["info_only"][0].startswith("참고용만")


def test_find_hard_terms_catches_raw_jargon_shown_to_the_reader():
    """어려운 말이 화면에 그대로 있으면 잡는다(대응표의 '원문' 칸)."""
    found = i18n.find_hard_terms("<p>기저율보다 낮고 낙폭이 작다</p>")
    assert found.get("기저율") == 1 and found.get("낙폭") == 1


def test_find_hard_terms_allows_the_term_inside_parentheses():
    """§2.1 '먼저 뜻, 그다음 용어' — 괄호 안에 남긴 용어는 통과시킨다."""
    ok = "<p>기준선보다 9.1% 더 정확 (Brier skill +0.091)</p>"
    assert i18n.find_hard_terms(ok) == {}
    assert i18n.find_hard_terms(ok, allow_in_parens=False).get("Brier skill") == 1


def test_find_hard_terms_ignores_css_and_attributes():
    """<style> 안이나 태그 속성에 있는 글자는 독자가 읽는 글자가 아니다."""
    html = '<style>.tone{color:red}</style><div class="국면" title="앙상블">평온합니다</div>'
    assert i18n.find_hard_terms(html) == {}


def test_hard_patterns_split_compound_entries():
    """'낙폭(drawdown)' · 'dwell / 체류' 같은 칸은 실제로 화면에 나올 표현으로 쪼개 둔다."""
    for pat in ("낙폭", "drawdown", "dwell", "체류", "walk-forward", "post hoc"):
        assert pat in i18n.HARD_PATTERNS


# ==================================================================
# 4. 짝 검사 (STYLE_I18N.md §5)
# ==================================================================
def test_check_pairs_passes_on_balanced_markup():
    html = "<div>" + i18n.bi("평소 비율", "base rate") + i18n.bi("표본 12개", "12 samples") + "</div>"
    n_ko, n_en, problems = i18n.check_pairs(html)
    assert (n_ko, n_en) == (2, 2) and problems == []


def test_check_pairs_fails_when_one_language_is_missing():
    n_ko, n_en, problems = i18n.check_pairs('<span class="lg ko">평소 비율</span>')
    assert (n_ko, n_en) == (1, 0)
    assert any("개수가 다릅니다" in p for p in problems)


def test_check_pairs_catches_korean_left_in_the_english_span():
    _, _, problems = i18n.check_pairs(i18n.bi_html("평소 비율", "평소 비율"))
    assert any("영어 칸에 한글" in p for p in problems)


def test_check_pairs_catches_english_sentence_left_in_the_korean_span():
    _, _, problems = i18n.check_pairs(i18n.bi_html("the options market said so", "the options market said so"))
    assert any("한국어 칸에 영어 문장" in p for p in problems)


def test_check_pairs_allows_tickers_and_numbers_in_the_korean_span():
    """티커·숫자·날짜는 번역하지 않는다 — 화이트리스트가 이를 흘려보낸다."""
    html = i18n.bi("SPY 가 8% 내렸습니다 (VIX 19.4, 2026-09-04)", "SPY fell 8% (VIX 19.4, 2026-09-04)")
    assert i18n.check_pairs(html)[2] == []


def test_check_pairs_handles_nested_spans():
    """스팬 안에 스팬이 있어도 깊이를 세어 올바르게 끊는다."""
    html = '<span class="lg ko">평소 <span class="pill">비율</span></span><span class="lg en">base <span class="pill">rate</span></span>'
    assert i18n.check_pairs(html) == (1, 1, [])


def test_visible_text_strips_markup_and_restores_entities():
    assert i18n.visible_text("<p>1 &lt; 2</p><style>x{}</style>") == "1 < 2"


# ==================================================================
# 5. 모듈 분리 — 파사드가 예전 이름을 하나도 잃지 않는다
# ==================================================================
def test_facade_reexports_every_name_the_old_module_had():
    """4,273 줄짜리 mrl/report.py 가 내보내던 이름 219개가 그대로 있어야 한다(호출자 무수정)."""
    missing = sorted(n for n in HEAD_NAMES if not hasattr(report, n))
    assert missing == [], f"파사드에서 사라진 이름: {missing}"
    assert len(HEAD_NAMES) == 219


def test_head_names_still_match_git_head():
    """git HEAD 의 mrl/report.py 를 직접 읽어 위 목록이 낡지 않았는지 확인한다."""
    try:
        src = subprocess.run(["git", "show", "HEAD:mrl/report.py"], cwd=ROOT, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:            # git 없는 환경(배포 tarball 등)
        pytest.skip(f"git 사용 불가: {e}")
    if src.returncode != 0:
        pytest.skip("git HEAD 에 mrl/report.py 없음")
    text = src.stdout.decode("utf-8")
    names = set()
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {a.asname or a.name.split(".")[0] for a in node.names}
    names -= {"annotations", "_sys", "_types", "_MODULES", "_Facade",
              "_report_common", "_report_p2", "_report_p3", "_report_v0"}
    missing = sorted(n for n in names if not hasattr(report, n))
    assert missing == [], f"git HEAD 에는 있으나 파사드에 없는 이름: {missing}"


def test_facade_is_thin():
    """파사드는 얇아야 한다 — 렌더링 코드가 다시 흘러들어오면 분리가 무의미해진다."""
    text = (ROOT / "mrl" / "report.py").read_text(encoding="utf-8")
    assert len(text.splitlines()) < 120
    assert "def render_" not in text and "<div" not in text


def test_render_entry_points_live_in_the_page_modules():
    """진입점이 실제로 페이지 모듈에 있고, 파사드는 같은 함수 객체를 가리킨다."""
    assert report.render_backtest_report is report_v0.render_backtest_report
    assert report.render_index is report_v0.render_index
    assert report.charts_for is report_v0.charts_for
    assert report.render_calibration_report is report_p2.render_calibration_report
    assert report.render_backtest_v1 is report_p2.render_backtest_v1
    assert report.charts_p2 is report_p2.charts_p2
    assert report.render_sizing_report is report_p3.render_sizing_report
    assert report.render_regime_report is report_p3.render_regime_report
    assert report.render_track_record is report_p3.render_track_record
    assert report.charts_p3 is report_p3.charts_p3
    assert report._write_html is report_common._write_html


def test_patching_the_facade_reaches_the_page_modules(monkeypatch):
    """기존 테스트가 monkeypatch.setattr(report, ...) 로 동작을 바꾸므로, 파사드 수정이 실제 모듈까지 닿아야 한다."""
    sentinel = object()
    monkeypatch.setattr(report, "_png", sentinel)
    assert report_common._png is sentinel and report_p2._png is sentinel and report_p3._png is sentinel
    monkeypatch.undo()
    assert report_common._png is not sentinel and report_p2._png is not sentinel


def test_page_modules_do_not_import_the_facade():
    """순환 참조 금지 — 페이지 모듈은 report_common(과 앞 단계 모듈)만 본다."""
    for mod in ("report_common", "report_v0", "report_p2", "report_p3"):
        text = (ROOT / "mrl" / f"{mod}.py").read_text(encoding="utf-8")
        assert "from mrl import report\n" not in text and "from mrl.report import" not in text


def test_the_shell_is_the_only_place_that_wires_the_toggle():
    """토글 배선은 껍데기 한 곳뿐 — 페이지 모듈이 각자 심으면 개수가 어긋난다."""
    for mod in ("report_v0", "report_p2", "report_p3"):
        text = (ROOT / "mrl" / f"{mod}.py").read_text(encoding="utf-8")
        assert "LANG_TOGGLE_HTML" not in text and "lang-sw" not in text
    assert "LANG_TOGGLE_HTML" in (ROOT / "mrl" / "report_common.py").read_text(encoding="utf-8")


# ==================================================================
# 6. 정직성 — 배선 단계가 문구를 지우지 않았는지
# ==================================================================
@pytest.mark.parametrize("name", sorted(PAGES))
def test_pages_are_still_written_and_nonempty(name, tmp_path, no_kill):
    """토글을 달았다고 본문이 사라지면 안 된다."""
    html = _render(name, tmp_path)
    assert html.startswith("<!doctype html>")
    assert len(i18n.visible_text(html)) > 200


# ==================================================================
# 8. 언어가 페이지를 넘어가도 유지된다 (여러 페이지로 나뉜 뒤 생긴 문제)
# ==================================================================
@pytest.mark.parametrize("name", sorted(PAGES))
def test_each_page_also_writes_an_english_sibling(name, tmp_path, no_kill):
    """`<stem>.html` 옆에 `<stem>.en.html` 이 함께 나온다 — 언어가 파일 이름에 담긴다."""
    _render(name, tmp_path)
    en = tmp_path / f"{name}.en.html"
    assert en.exists(), f"{name}: 영어판 자매 문서가 없습니다"
    html = en.read_text(encoding="utf-8")
    assert '<html lang="en"' in html
    assert ".lg.ko{display:none}.lg.en{display:inline}" in html      # 기본 표시가 뒤집혔다
    assert "<script" not in html and "onclick" not in html           # 자바스크립트 0줄은 그대로
    assert html.count('id="lang-sw"') == 1 and html.count('for="lang-sw"') == 1


@pytest.mark.parametrize("name", sorted(PAGES))
def test_both_variants_carry_the_same_bilingual_pairs(name, tmp_path, no_kill):
    """두 판은 같은 bi(ko, en) 쌍에서 나온다 — 어느 판에서도 숫자·한계가 빠지지 않는다."""
    ko_html = _render(name, tmp_path)
    en_html = (tmp_path / f"{name}.en.html").read_text(encoding="utf-8")
    assert i18n.count_pairs(ko_html) == i18n.count_pairs(en_html)
    body = lambda h: h.split('<div class="wrap">', 1)[1]
    assert body(ko_html) == body(en_html).replace(".en.html", ".html")


@pytest.mark.parametrize("name", sorted(PAGES))
def test_the_pill_links_to_the_sibling_document(name, tmp_path, no_kill):
    """알약 두 칸이 자매 문서로 간다 — 링크를 타고 가도 언어가 유지된다(자바스크립트 없이)."""
    ko_html = _render(name, tmp_path)
    en_html = (tmp_path / f"{name}.en.html").read_text(encoding="utf-8")
    for html in (ko_html, en_html):
        pill = re.search(r'<label for="lang-sw".*?</label>', html, re.S).group(0)
        assert f'href="{name}.html"' in pill and f'href="{name}.en.html"' in pill


@pytest.mark.parametrize("name", sorted(PAGES))
def test_in_page_anchors_are_never_rewritten(name, tmp_path, no_kill):
    """차례의 #s1…#c6 앵커와 절대 URL 은 영어판에서도 그대로다(링크가 깨지면 목차가 죽는다)."""
    _render(name, tmp_path)
    en_html = (tmp_path / f"{name}.en.html").read_text(encoding="utf-8")
    assert ".en.html#" not in en_html
    assert "https://uncmac.github.io/market-risk-lab/.en.html" not in en_html
    for href in re.findall(r'href="([^"]+)"', en_html):
        assert not href.startswith("#") or href.count(".") == 0


def test_wide_screen_reserves_room_for_the_pill():
    """넓은 화면에서 알약은 절대위치다 — 헤더가 자리를 비워 두지 않으면 제목 첫 줄을 덮는다.

    market-brief 에서 겪은 겹침이 ≤640px 에서만 고쳐져 있었다: 641px~1000px 사이에서는 불투명 알약이
    긴 제목의 오른쪽 끝 글자를 그대로 덮었다(실측: regime_p3 921px 에서 '후보 모형' 이 3.55px 잘림).
    """
    m = re.search(r"\.wrap>header\{padding-right:(\d+)px\}", i18n.LANG_CSS)
    assert m, "넓은 화면에서 헤더가 알약 자리를 비워 두지 않는다"
    assert int(m.group(1)) >= 140, "알약 실측 폭 135px + 여백보다 좁게 잡으면 여전히 겹친다"
    narrow = re.search(r"@media\(max-width:640px\)\{.*?\}\}", i18n.LANG_CSS, re.S)
    assert narrow, "좁은 화면 규칙이 없다"
    assert "header{padding-right:0}" in narrow.group(0).replace(" ", ""), \
        "좁은 화면에서는 알약이 흐름 배치라 자리를 비워 둘 필요가 없다(폰 가로폭 낭비)"


@pytest.mark.parametrize("name", sorted(PAGES))
def test_wide_screen_reservation_reaches_every_page(name, tmp_path, no_kill):
    html = _render(name, tmp_path)
    assert ".wrap>header{padding-right:150px}" in html
    assert '<div class="wrap"><header' in html.replace("\n", ""), f"{name}: .wrap>header 선택자가 닿지 않는다"
