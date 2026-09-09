# -*- coding: utf-8 -*-
"""Phase 3 페이지(mrl/report_p3.py)의 표현 계약 — 쉬운 한국어 + 한/영 두 벌 (STYLE_I18N.md).

무엇을 강제하는가
  §1 두 벌      비중 카드·sizing_p3·regime_p3·track_record 의 문장이 `bi()` 로 심겨 한/영 개수가 맞는다.
                표는 **열 제목만** 두 벌이고 셀 값은 건드리지 않는다.
  §2 쉬운 말    한국어 화면에 남은 어려운 용어를 센다(괄호 안은 허용 — "먼저 뜻, 그다음 용어").
  §5 언어 섞임  영어 칸에 한글이 없고, 한국어 칸에 영어 **문장**이 없다(식별자는 예외로 명시).
  정직성        쉬운 말로 바꾸면서 숫자·표본 수·구간·post hoc·info_only·"투자 조언 아님" 이 사라지지 않았다.

검사의 범위 — 무엇을 린트하고 무엇을 린트하지 않는가 (STYLE_I18N.md §5)
  * 린트 대상 = **한국어를 고른 독자가 실제로 읽는 화면 전체** = 영어 스팬만 지운 문서.
    `bi()` 밖에 그대로 꽂힌 글자(시나리오 줄·킬룰 문장·인용 블록)도 여기 들어온다 —
    예전처럼 한국어 스팬 안만 보면 안 고친 문장이 구조적으로 걸리지 않는다.
  * 빼는 것은 딱 두 가지뿐이고 둘 다 계약이 정한 것이다:
      1) `<span class="raw …">` — 계산이 남긴 한국어 **원문 인용**. 바로 옆에 그 사실을 밝히는
         두 벌 고지가 붙어 있다(test_raw_korean_is_always_disclosed 가 강제).
      2) `<td>` 안의 **셀 값** — §1 "표 안의 셀 값은 그대로 두고 열 제목과 각주만 이중화한다".
         열 제목(`<th>`)은 검사 대상으로 남는다.
  * 그래도 남는 예외는 아래 PINNED_* 에 이유와 함께 하나씩 적는다 — 목록에 없는 새 위반은 실패한다.
    이 목록들은 **줄어들기만 한다**(늘리려면 그 자리에 쉬운 말을 붙이는 것이 먼저다).

실제 산출물(results/·docs/)은 건드리지 않는다(tmp_path 만).
"""
from __future__ import annotations

import html as _html
import re
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrl import i18n, report, report_p3, sizing, track                # noqa: E402
from mrl.config import ENSEMBLE_P3, KILL_P3, P3, SCENARIO_P3          # noqa: E402

import tests.test_report_p3 as P3T                                    # noqa: E402  (같은 합성 자료를 쓴다)

_PAREN = re.compile(r"[(（][^()（）]*[)）]")
_HANGUL = re.compile(r"[가-힣]")
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z_'-]*")


# ------------------------------------------------------------------
# 예외 목록 — 하나하나 이유가 있고, 목록에 없는 위반은 실패한다
# ------------------------------------------------------------------
# (1) 차례(nav)의 항목 이름: 기존 테스트가 글자 그대로 찾는다(test_report_p3.py
#     ::test_sizing_report_has_every_section / ::test_regime_report_has_every_section_and_param_table).
#     본문 제목은 쉬운 말로 바꾸고 용어는 괄호에 넣었지만, 차례 항목은 그 검사 때문에 원문을 유지한다.
# 차례 항목도 이제 쉬운 말로 바꿨으므로(그리고 그 검사도 함께 고쳤으므로) 예외가 필요 없다.
PINNED_NAV = ()

# (2) 계산층(mrl.track · mrl.sizing)이 정한 원문 — 규칙의 단일 원천이라 문구를 이 모듈에서 고치지 않는다.
#     영어 판(HORIZON_FOOTNOTE_EN · HORIZON_SENTENCES_EN · HONEST_READING_EN)은 **같은 자리표**로
#     리포트 층에 붙였으므로 영어 독자도 같은 숫자·같은 한계를 읽는다. 남는 것은 한국어 화면의 원문
#     용어(에피소드·낙폭·MaxDD·킬룰 …)뿐이고, 그 앞에는 쉬운 요약이 두 벌로 함께 붙는다.
def _honest_lines() -> tuple[str, ...]:
    """§6.5 네 줄 — 이 페이지가 실제로 인용하는 문자열(숫자만 갈아끼운 mrl.sizing 원문)."""
    sz = (P3T._summary_p3().get("sizing") or {})
    hr = sz.get("honest_reading")
    if not hr:
        res = sz.get("honest_results") if isinstance(sz.get("honest_results"), dict) else None
        hr = sizing.honest_reading(res)
    return tuple(str(x) for x in hr) + tuple(sizing.honest_reading())


PINNED_QUOTES = ((track.HORIZON_FOOTNOTE,) + tuple(track.HORIZON_SENTENCES.values())
                 + _honest_lines())

# (2b) 화면에 그대로 나와야 뜻이 통하는 **식별자**(산출물 키·태그 이름). 번역하면 파일·열을 못 찾는다.
#      용어표의 원문 글자를 품고 있어서(deploy_sizing ⊃ deploy) 세기 전에 지운다.
PINNED_IDENTIFIERS = ("deploy_sizing", "deploy_mode", "effective_mode", "info_only",
                      "model_p2.json", "model_p3.json", "kill_replay")

# (3) 한국어 문장 안에 남는 라틴 낱말 — 코드·산출물·장부 열에 그대로 나오는 **식별자**다.
#     번역하면 오히려 파일·열을 못 찾으므로 그대로 둔다(STYLE_I18N.md §1: 식별자는 번역하지 않는다).
IDENTIFIERS = {
    "skill", "ens", "bss", "brier", "clip", "verdict", "reason", "guard", "sticky", "exit",
    "smoothed", "filter", "w_vol", "w_exec", "n_eff", "spec_sha", "registry_sha", "sizing_sha",
    "spec_sha256", "deploy_mode", "effective_mode", "info_only", "tones", "kill", "mrl", "track",
    "model_p2", "model_p3", "results", "summary_p3", "summary_p2", "sizing", "scenarios", "hmm",
    "theta_table", "ladder", "registry", "admission", "determinism", "episode_pnl", "sensitivities",
    "relative", "retention", "table", "bins", "states", "kill_replay", "numpy",
}
ALLOWED_LATIN = {w.lower() for w in i18n.ALLOWED_LATIN} | IDENTIFIERS


# ------------------------------------------------------------------
# 렌더 (합성 자료 — 실제 산출물은 읽지 않는다)
# ------------------------------------------------------------------
@pytest.fixture()
def no_kill(monkeypatch, tmp_path):
    monkeypatch.setattr(report, "KILL_RECORD_PATH", tmp_path / "kill_record.json")
    monkeypatch.setattr(report, "KILL_MANUAL_PATH", tmp_path / "kill_manual.json")
    return tmp_path


def _track_input() -> dict:
    """§8.5 패널이 채워진 트랙레코드 입력(합성)."""
    return {"n_scored": 300, "n_eff": 15.0, "months_elapsed": 14, "live_start": "2025-07-01",
            "kill": {"state": "not_due", "months": 14, "episodes5": 2},
            "panel": {"i_p_level": {"mean_p": 0.16, "share_gt_030": 0.1, "backtest": {"mean_p": 0.15}},
                      "ii_state_occupancy": {"normal": 0.8},
                      "iii_switching": {"switches_per_year": 3.9, "w_changes_per_year": 12.0},
                      "iv_exposure": {"avg_w": 0.69, "share_floor": 0.1},
                      "v_vol_forecast": {"har_log_mae": 0.28},
                      "vi_coverage": {"coverage": 0.81},
                      "vii_skill": {"brier": 0.12, "bss_clim": 0.05, "hist_pct": {"252": 0.4}},
                      "viii_relative": {"share_behind": 0.8},
                      "ix_kill": {"months": 14, "episodes5": 2},
                      "x_alarms": {"n": 1},
                      "xi_registry": {"members": {"H": {"n": 300, "n_eff": 15.0, "brier": 0.12}},
                                      "disagreement": {"width_pp": 8.0}, "fresh_blocks": "3/12"}},
            "alarms": [{"asof": "2026-09-01", "code": "D1_p_level", "value": 0.038, "threshold": 0.04,
                        "action": "display"}],
            "replay_mismatch_count": 0, "acceptance": dict(P3T.ACC), "deploy_sizing": True,
            "notes": ["장부 기록"]}


@pytest.fixture()
def pages(tmp_path, no_kill) -> dict[str, str]:
    """세 페이지 + 두 모드의 카드 → {이름: HTML}."""
    s = P3T._summary_p3()
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p1, p2, p3 = (tmp_path / "sizing_p3.html", tmp_path / "regime_p3.html",
                      tmp_path / "track_record.html")
        report.render_sizing_report(s, {"allocation": {"cagr": 0.1019, "max_dd": -0.310}}, {}, p1, {})
        report.render_regime_report(s, p2, {})
        report.render_track_record(_track_input(), s, p3, {})
        for p in (p1, p2, p3):
            out[p.stem] = p.read_text(encoding="utf-8")
        out["card_tones"] = report.p3_card(P3T._today_p3(), "tones")
        out["card_info_only"] = report.p3_card(P3T._today_p3(p2_deploy_mode="info_only"), "info_only")
        out["card_empty"] = report.p3_card({}, None)
    return out


# ------------------------------------------------------------------
# 도구
# ------------------------------------------------------------------
def _visible(html: str) -> str:
    t = re.sub(r"<style.*?</style>", " ", html, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", _html.unescape(t)).strip()


def _spans(html: str, lang: str) -> list[str]:
    """그 언어 스팬 안의 글자만(겹친 스팬도 깊이를 세어 끊는다 — i18n 의 파서를 그대로 쓴다)."""
    return [i18n.visible_text(inner) for lg, inner in i18n._spans(html) if lg == lang]


_SPAN_OPEN_LANG = re.compile(r'<span\b[^>]*class="lg (ko|en)"[^>]*>', re.I)
_RAW_SPAN = re.compile(r'<span class="raw[^"]*">.*?</span>', re.S)
_TD = re.compile(r"<td[^>]*>.*?</td>", re.S)


def _drop_lang(html: str, lang: str) -> str:
    """그 언어 스팬을 통째로(안쪽 내용까지) 지운다 — 겹친 <span> 도 깊이를 세어 올바르게 끊는다."""
    out, i, src = [], 0, html or ""
    while True:
        m = _SPAN_OPEN_LANG.search(src, i)
        while m and m.group(1).lower() != lang:
            m = _SPAN_OPEN_LANG.search(src, m.end())
        if not m:
            out.append(src[i:])
            return "".join(out)
        out.append(src[i:m.start()])
        j, depth = m.end(), 1
        while depth and j < len(src):
            nxt, end = src.find("<span", j), src.find("</span>", j)
            if end < 0:
                break
            if 0 <= nxt < end:
                depth += 1
                j = nxt + 5
            else:
                depth -= 1
                j = end + 7
        i = j


def _korean_screen(html: str) -> str:
    """한국어를 고른 독자가 실제로 읽는 글자 = **영어 스팬만** 지운 화면 전체.

    빼는 것은 계약이 정한 두 가지뿐이다: 원문 인용(`span.raw`)과 표의 셀 값(`<td>`, §1).
    """
    doc = _drop_lang(html, "en")
    doc = _RAW_SPAN.sub(" ", doc)
    doc = _TD.sub(" ", doc)
    return i18n.visible_text(doc)


def _english_screen(html: str) -> str:
    """영어를 고른 독자가 읽는 글자 — _korean_screen 과 **같은 범위**(원문 인용·셀 값 제외)."""
    doc = _drop_lang(html, "ko")
    doc = _RAW_SPAN.sub(" ", doc)
    doc = _TD.sub(" ", doc)
    return i18n.visible_text(doc)


def _english_view(html: str) -> str:
    """영어 화면에 실제로 보이는 글자 전부 — 한국어 스팬만 지운다(값·표 셀까지 포함)."""
    return i18n.visible_text(_drop_lang(html, "ko"))


def _strip_parens(text: str) -> str:
    prev = None
    while prev != text:                       # 중첩 괄호를 안쪽부터 걷어낸다
        prev = text
        text = _PAREN.sub(" ", text)
    return text


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def _hard_terms_in_korean(html: str) -> dict[str, int]:
    """한국어 화면에 그대로 남은 어려운 용어 → {용어: 횟수}. 괄호 안·예외 목록은 뺀다."""
    text = _korean_screen(html)
    for quoted in PINNED_QUOTES + PINNED_NAV:
        text = text.replace(_squash(quoted), " ")
    for ident in PINNED_IDENTIFIERS:
        text = text.replace(ident, " ")
    for _ko, _ in i18n.TERMS.values():             # 승인된 쉬운 말이 원문 글자를 품는 경우(주식 비중 ⊃ 비중)
        text = text.replace(_ko, " ")
    text = _strip_parens(text)
    return {p: n for p in i18n.HARD_PATTERNS if (n := text.count(p))}


def _english_sentence_in_korean(html: str) -> list[str]:
    """한국어 칸에 남은 영어 '문장'(식별자 두 개까지는 문장으로 보지 않는다)."""
    bad = []
    for text in _spans(html, "ko"):
        words = [w for w in _LATIN_WORD.findall(text)
                 if w.islower() and len(w) >= 3 and w.lower() not in ALLOWED_LATIN]
        if len(words) >= 2:
            bad.append(f"{text[:70]!r} ({', '.join(words[:4])})")
    return bad


ALL = ("sizing_p3", "regime_p3", "track_record", "card_tones", "card_info_only", "card_empty")
PAGES_ONLY = ("sizing_p3", "regime_p3", "track_record")


# ==================================================================
# 1. 두 벌 심기 (STYLE_I18N.md §1)
# ==================================================================
@pytest.mark.parametrize("name", ALL)
def test_every_korean_sentence_has_an_english_twin(name, pages):
    """한/영 스팬 개수가 같다 — 한쪽만 심으면 그 언어 화면에서 문장이 사라진다."""
    n_ko, n_en, problems = i18n.check_pairs(pages[name])
    assert n_ko == n_en, f"{name}: ko={n_ko} en={n_en}"
    assert n_ko > 20, f"{name}: 두 벌로 심은 문장이 {n_ko}개뿐 — 대부분이 아직 한 언어입니다"
    assert not [p for p in problems if "영어 칸에 한글" in p], [p for p in problems if "영어 칸에 한글" in p]


@pytest.mark.parametrize("name", ALL)
def test_no_english_sentence_left_in_the_korean_span(name, pages):
    """한국어 칸에 영어 문장이 남지 않았다(코드·산출물 식별자는 IDENTIFIERS 로 명시)."""
    assert _english_sentence_in_korean(pages[name]) == []


@pytest.mark.parametrize("name", ALL)
def test_no_korean_left_in_the_english_span(name, pages):
    """영어 칸에 한글이 남지 않았다 — 영어를 고른 독자가 읽지 못하는 글이 없어야 한다."""
    left = [t[:70] for t in _spans(pages[name], "en") if _HANGUL.search(t)]
    assert left == []


# regime_p3 의 표도 이 모듈이 책임진다 — 사다리·블록·시대별 AUC·소거·신뢰도는 _p3_table 로 그린다.
@pytest.mark.parametrize("name", PAGES_ONLY)
def test_table_headers_are_bilingual(name, pages):
    """표는 열 제목만 두 벌 (STYLE_I18N.md §1) — 한글 머리글에는 반드시 영어 짝이 있다."""
    html = pages[name]
    ths = re.findall(r"<th[^>]*>(.*?)</th>", html, re.S)
    assert ths, f"{name}: 표 머리글이 없다"
    lonely = [t for t in ths if _HANGUL.search(_visible(t)) and 'class="lg en"' not in t]
    assert lonely == [], f"{name}: 영어 짝이 없는 열 제목 {lonely[:3]}"


@pytest.mark.parametrize("name", PAGES_ONLY)
def test_table_values_are_left_untouched(name, pages):
    """셀 **값**(<td class="num">)에는 언어 스팬을 넣지 않는다 — 값은 번역하지 않는다(§1).

    항목 이름을 담는 왼쪽 칸은 라벨이므로 두 벌이어도 된다."""
    tds = re.findall(r'<td class="num"[^>]*>(.*?)</td>', html := pages[name], re.S)
    assert tds, f"{name}: 숫자 칸이 없다"
    assert not [t for t in tds if 'class="lg ' in t], f"{name}: 값 칸에 언어 스팬이 들어갔다"
    assert html


# ==================================================================
# 2. 쉬운 한국어 (STYLE_I18N.md §2·§3)
# ==================================================================
@pytest.mark.parametrize("name", ALL)
def test_korean_screen_has_no_raw_jargon(name, pages):
    """한국어 화면에 어려운 용어가 맨몸으로 남지 않았다(괄호 안은 허용 — 먼저 뜻, 그다음 용어)."""
    assert _hard_terms_in_korean(pages[name]) == {}


def test_glossary_terms_are_actually_replaced_by_the_plain_words(pages):
    """대응표의 쉬운 말이 실제로 화면에 있다 — 용어만 지우고 뜻을 안 쓴 것이 아니다."""
    ko = " ".join(_korean_screen(pages[n]) for n in ALL)
    for plain in ("시장 분위기", "주식 비중", "판정 규칙", "고점 대비 하락폭", "겹치지 않는 창",
                  "결과를 본 뒤", "참고용", "후보 모형", "성적이 나쁘면 끄는 규칙"):
        assert plain in ko, plain


# §2.3 "한 문장 = 한 가지" 의 상한(120자)을 넘겨도 좋은 문장 — 하나하나 이유가 있고 **줄어들기만 한다**.
# 길이를 줄이려고 숫자를 지우는 것은 금지다(정직성 > 길이). 쪼갤 수 있는 문장은 쪼개서 목록에서 뺀다.
LONG_SENTENCES_ALLOWED = (
    ("v0 (동결 벤치마크", "v0 영구 표기 한 줄 — 여섯 지표를 한 줄에 모으는 것이 계약이다(VALIDATION §4)."),
    ("규칙 원문 그대로:", "비중 규칙의 상수·수식 원문. 쪼개면 '한 규칙'이라는 사실이 흐려진다."),
    ("이 규칙에는 값이 붙는다:", "§16 정직 문구 — 세 가지 비용(수익·현금 이자·거래비용)을 한 항목에 묶어 둔다."),
    ("대신 변동성만 보는 규칙", "info_only 배너 — 무엇을 숨기고 무엇을 남기는지 한 번에 적어야 뜻이 통한다."),
    ("(b) 확률층이", "§6.5 정직한 읽기 (b) — mrl.sizing 원문 인용(글자를 바꾸지 않는다)."),
    ("라이브에서는 D4", "경보 D4·D4b·D6 의 조건 세 개와 그 결과 — 쪼개면 '하나만 걸려도' 라는 조건이 깨진다."),
    ("AUC 0.697", "⑥ 시대별 AUC 해설 — 숫자와 그 숫자를 믿으면 안 되는 이유를 한 호흡에 둔다."),
    ("유효 모드(p2 ∧ p3 ∧ ¬kill)", "정직 스트립의 근거 줄 — 값 세 개와 사유를 붙여 두어야 판정을 되짚을 수 있다."),
)


def test_plain_korean_sentences_are_short_enough(pages):
    """한 문장 = 한 가지(§2.3). 상한을 넘는 문장은 이름 붙인 예외 목록에 있어야 한다."""
    heads = tuple(h for h, _ in LONG_SENTENCES_ALLOWED)
    long_ones = []
    for name in ALL:
        for text in _spans(pages[name], "ko"):
            for sent in re.split(r"(?<=[.。!?])\s+", text):
                if len(sent) > 120 and not any(h in sent for h in heads):
                    long_ones.append((name, len(sent), sent[:80]))
    assert long_ones == [], f"목록에 없는 긴 문장 {len(long_ones)}개: {long_ones[:3]}"


# ==================================================================
# 3. 정직성 — 쉬운 말로 바꾸면서 무엇도 잃지 않았다
# ==================================================================
def test_info_only_gating_text_survives_in_both_languages(pages):
    """§7 게이팅 문구: 한국어는 원문 그대로, 영어에도 같은 뜻이 있다."""
    for name in ("card_info_only", "sizing_p3", "regime_p3", "track_record"):
        html = pages[name]
        assert report.P3_INFO_ONLY_LABEL in _visible(html), f"{name}: VALIDATION §7 문구가 사라졌다"
        en = " ".join(_spans(html, "en")).lower()
        assert "information only" in en, f"{name}: 영어 화면에 '정보 제공 전용' 이 없다"
    card = pages["card_info_only"]
    assert report.P3_INFO_NOT_SIZING in card                       # '정보(비중 제안 아님)' 회색 라벨
    assert "not a suggested amount to hold" in " ".join(_spans(card, "en"))
    # 비중·상태·톤을 제안하지 않는다는 사실이 두 언어에 모두 있다
    assert "제안을 하지 않습니다" in _korean_screen(card)
    assert "suggests no amount to hold" in " ".join(_spans(card, "en"))


@pytest.mark.parametrize("name", ALL)
def test_no_undisclosed_korean_in_the_english_view(name, pages):
    """영어 화면에 남는 한글은 반드시 `class="raw"` 안이고, 그 사실을 밝히는 고지가 함께 있어야 한다.

    한국어를 못 읽는 독자가 '이 페이지를 다 읽었다' 고 착각하면 그 자리의 한계(표본 수·구간·경고)가
    조용히 사라진 것과 같다 — 그래서 남기더라도 **남겼다고 말한다**.
    """
    html = pages[name]
    body = _RAW_SPAN.sub(" ", _drop_lang(html, "ko"))
    body = re.sub(r"<title>.*?</title>", " ", body, flags=re.S)     # <title> 은 영어로 통일(별도 검사)
    body = re.sub(r'<label[^>]*class="lang-btn[^"]*".*?</label>', " ", body, flags=re.S)   # 알약의 '한국어' 칸
    body = _TD.sub(" ", body)                                        # 셀 값은 번역하지 않는다(§1)
    left = [t for t in re.findall(r"[가-힣][^<>]{0,40}", i18n.visible_text(body)) if t.strip()]
    assert left == [], f"{name}: 영어 화면에 고지 없는 한국어가 남았습니다 {left[:5]}"
    if _HANGUL.search(_english_view(html)):                          # raw 가 있으면 고지도 있어야 한다
        assert ("raw Korean" in _english_view(html)
                or "original Korean" in _english_view(html)
                or "quoted verbatim" in _english_view(html)), f"{name}: 원문 인용에 고지가 없습니다"


def _scenario_tables():
    """모든 분기를 돌려 볼 수 있는 합성 시나리오 표 (숫자는 임의 — 문장 모양만 본다)."""
    bins = [{"bin": "[0.08, 0.12)", "bin_lo": 0.08, "bin_hi": 0.12, "pooled": False, "n": 1140,
             "n_eff": 57.0, "mean_p": 0.10, "obs": 0.105, "wilson_lo": 0.05, "wilson_hi": 0.21,
             "ret_p10": -0.036, "ret_p90": 0.040, "mdd_p10": -0.050, "grey": False},
            {"bin": "[0.12, 0.20)", "bin_lo": 0.12, "bin_hi": 0.20, "pooled": False, "n": 120,
             "n_eff": 6.0, "mean_p": 0.15, "obs": 0.30, "wilson_lo": 0.10, "wilson_hi": 0.60,
             "ret_p10": -0.05, "ret_p90": 0.03, "mdd_p10": -0.07, "grey": True}]
    ep = {"n": 33,
          "p_ge10_given5": {"k": 11, "n": 33, "lo": 0.20, "hi": 0.50},
          "p_ge20_given5": {"k": 4, "n": 33, "lo": 0.05, "hi": 0.27},
          "p_ge15_given10": {"k": 5, "n": 11, "lo": 0.20, "hi": 0.73},
          "p_ge20_given10": {"k": 4, "n": 11, "lo": 0.15, "hi": 0.65},
          "depth_q": {"p50": -0.076}, "extra_loss_q": {"p50": -0.028, "p10": -0.194},
          "breach_to_trough_q": {"p50": 22}, "trough_to_recovery_q": {"p50": 48}}
    cov = {"vix": {"hit_80": 0.94, "n_eff": 57.0, "label_80": "보수적"}, "har": {"hit_80": 0.90}}
    return {"bins": bins, "episodes": ep, "coverage": cov, "states": []}


@pytest.mark.parametrize("p_today,dd,branch", [
    (0.10, {"dd_from_ath": -0.02}, "normal"),
    (0.10, {"dd_from_ath": -0.07}, "breached_5"),
    (0.10, {"dd_from_ath": -0.14}, "breached_10"),
    (0.15, {"dd_from_ath": -0.02}, "normal"),          # 얇은(회색) 구간 → _line_today 대체 문장
    (0.90, {"dd_from_ath": -0.02}, "normal"),          # 해당 구간 없음 → _line_today 대체 문장
])
def test_card_scenario_lines_are_plain_and_bilingual(p_today, dd, branch, no_kill):
    """시나리오 세 줄은 모든 분기에서 쉬운 한국어 + 영어 두 벌로 나온다(§1·§2).

    계산층 원문(에피소드·최대낙폭·10~90%·나쁜 10%·창)은 화면에서 사라지고, 숫자·구간·표본 수는 남는다.
    """
    from mrl import scenarios as S
    ctx = S.today_context(p_today, "normal", 15.3, 0.11, 640.0, _scenario_tables(), dd)
    assert ctx["branch"] == branch
    assert len(ctx["lines_bi"]) == 3 and all(len(x) == 2 for x in ctx["lines_bi"])
    card = report.p3_card(P3T._today_p3(scenarios=ctx), "tones")
    ul = re.search(r'<ul class="plain">(.*?)</ul>', card, re.S)
    lis = re.findall(r"<li>(.*?)</li>", ul.group(1), re.S)
    assert len(lis) == 3, lis
    for li in lis:
        assert li.count('class="lg ko"') == 1 and li.count('class="lg en"') == 1, li[:80]
    ko, en = _korean_screen(card), _english_screen(card)
    assert ko != en
    raw_terms = ("에피소드", "최대낙폭", "낙폭 나쁜", "10~90%", "나쁜 10%", "창 중", "독립 창 6")
    li_ko = " ".join(_spans(ul.group(1), "ko"))
    for raw in raw_terms:                                   # 세 줄 안에는 괄호 안에도 없어야 한다
        assert raw not in li_ko, f"{raw} 가 시나리오 세 줄에 남았습니다"
    rest = ko
    for quoted in PINNED_QUOTES:                            # 계산층 원문은 §5 예외 목록이 이미 맡는다
        rest = rest.replace(_squash(quoted), " ")
    for raw in raw_terms:                                   # 페이지 전체 — 괄호 안 용어는 §2.1 이 허용
        assert raw not in _strip_parens(rest), f"{raw} 가 한국어 화면에 맨몸으로 남았습니다"
    assert not _HANGUL.search(" ".join(_spans(card, "en")))
    # 숫자·구간·표본 수는 두 언어 어디에서도 사라지지 않는다
    assert "95% 범위" in ko or "표본이 얇" in ko or "줄이 표에 없어" in ko
    assert "95% range" in en or "thin sample" in en or "does not land on any row" in en


def test_kill_rule_power_disclosure_survives_in_both_languages(tmp_path, no_kill):
    """§16.7 킬룰 검정력: 산출된 문장은 원문 그대로, 없으면 '재생성하지 않았다' 를 두 언어로 말한다."""
    s = P3T._summary_p3()
    s["kill_power"] = {"p2": {"p_col": "p_p2", "n_windows": 223,
                              "sentence_ko": "이 규칙은 과거 36개월 창에서 이 모델을 12% 오기각한다."}}
    out = tmp_path / "s.html"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report.render_sizing_report(s, {}, {}, out, {})
    html = out.read_text(encoding="utf-8")
    vis = _visible(html)
    assert "이 규칙은 과거 36개월 창에서 이 모델을 12% 오기각한다." in vis     # 산출값 원문
    assert "12%" in vis and "창 223" in vis                                    # ~79% 계열 숫자와 창 수
    en = " ".join(_spans(html, "en"))
    assert "regenerated by the weekly self-test" in en and "223 windows" in en
    # 재생성되지 않았으면 숫자를 지어내지 않는다
    s.pop("kill_power")
    out2 = tmp_path / "s2.html"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report.render_sizing_report(s, {}, {}, out2, {})
    h2 = out2.read_text(encoding="utf-8")
    assert "재생성하지 않아 표시하지 않는다" in _visible(h2)
    assert "did not regenerate the figure" in " ".join(_spans(h2, "en"))
    assert "오기각" not in _visible(h2).replace("재생성하지 않아 표시하지 않는다", "")


def test_grey_row_rule_for_thin_samples_survives_in_both_languages(tmp_path, no_kill):
    """얇은 칸 회색 규칙 — 회색 배지와 n_eff 한계선이 두 언어에 모두 남는다."""
    s = P3T._summary_p3()
    s["scenarios"] = {
        "bins": [{"bin": "[0.25, 1.00)", "bin_lo": 0.25, "bin_hi": 1.0, "pooled": True, "n": 380,
                  "n_eff": 19.0, "mean_p": 0.35, "obs": 0.42, "wilson_lo": 0.25, "wilson_hi": 0.61,
                  "grey": True}],
        "states": [{"state": "reduce", "n": 255, "n_eff": 12.75, "obs": 0.482, "wilson_lo": 0.246,
                    "wilson_hi": 0.727, "grey": True}]}
    out = tmp_path / "t.html"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report.render_track_record(_track_input(), s, out, {})
    html = out.read_text(encoding="utf-8")
    vis = _visible(html)
    assert "회색(단독 표시 금지)" in vis                                   # 열 제목
    assert f'n_eff &lt; {int(SCENARIO_P3["min_n_eff"])}' in html or "12.8" in vis
    assert "12.8" in vis and "48.2%" in vis and "24.6%" in vis and "72.7%" in vis   # 얇은 행의 숫자·구간
    en = " ".join(_spans(html, "en"))
    assert "grey, never shown alone" in en and "independent windows" in en
    assert "merged upward" in en                                          # 풀링 규칙


@pytest.mark.parametrize("name", ALL)
def test_sample_sizes_and_intervals_are_never_dropped(name, pages):
    """n·n_eff·구간이 한국어와 영어 어느 쪽에서도 사라지지 않았다."""
    ko, en = _korean_screen(pages[name]), " ".join(_spans(pages[name], "en"))
    assert "n_eff" in ko or "겹치지 않는 창" in ko, f"{name}: 한국어 화면에 표본 단위가 없다"
    assert "n_eff" in en or "independent window" in en, f"{name}: 영어 화면에 표본 단위가 없다"


def test_post_hoc_and_not_advice_lines_survive(pages):
    """'결과를 본 뒤 정한 것(post hoc)' 과 '투자 조언 아님' 이 두 언어에 모두 있다."""
    for name in PAGES_ONLY:                       # §16 정직 문구를 싣는 곳
        ko, en = _korean_screen(pages[name]), " ".join(_spans(pages[name], "en"))
        assert "post hoc" in ko and "결과를 본 뒤" in ko, f"{name}: post hoc 공개가 사라졌다"
        assert "post hoc" in en, f"{name}: 영어 화면에 post hoc 공개가 없다"
    for name in ALL:                              # 카드 각주까지 모든 화면
        ko, en = _korean_screen(pages[name]), " ".join(_spans(pages[name], "en"))
        assert "투자 조언이 아닙니다" in ko, f"{name}: '투자 조언 아님' 이 사라졌다"
        assert "not investment advice" in en, f"{name}: 영어 화면에 'not investment advice' 가 없다"


def test_headline_numbers_are_identical_in_both_languages(pages):
    """숫자는 번역하지 않는다 — 한국어 화면의 모든 수가 영어 화면에도 그대로 있다.

    부호·% 는 표기 관습이 달라(−3.6%p vs −3.6pp) 자릿수만 비교한다."""
    num = re.compile(r"\d+(?:[.,]\d+)?")
    for name in ALL:
        ko = set(num.findall(_korean_screen(pages[name])))
        en = set(num.findall(_english_screen(pages[name])))
        missing = sorted(ko - en)
        assert not missing, f"{name}: 영어 화면에 없는 숫자 {missing[:8]}"


def test_card_keeps_the_one_number_one_sentence_shape_in_both_languages(pages):
    """카드 첫 화면: 숫자 하나 + 문장 하나 (§10 1) — 두 언어 모두."""
    card = pages["card_tones"]
    assert "오늘 주식 비중 0.65" in card
    assert "Today's share in stocks: 0.65" in _visible(card)
    ko = _korean_screen(card)
    assert "목표 10%" in ko and "예상 15.4%" in ko and "상태 배수 1.00" in ko
    en = " ".join(_spans(card, "en"))
    assert "target 10%" in en and "expected 15.4%" in en and "state multiplier 1.00" in en


def test_kill_countdown_keeps_every_count_in_both_languages(pages):
    """킬 카운트다운의 분자/분모와 상한이 두 언어에서 모두 보인다."""
    vis = _visible(pages["card_tones"])
    assert f"실현 ≥5% 에피소드 0/{KILL_P3['min_episodes']}" in vis
    assert f"2/{KILL_P3['min_months']}개월" in vis and f"상한 {KILL_P3['max_months']}" in vis
    en = " ".join(_spans(pages["card_tones"], "en"))
    assert "How far the switch-off rule still has to go" in en


# ==================================================================
# 4. 토글 배선 (STYLE_I18N.md §1) — 세 페이지가 실제로 언어를 바꿀 수 있다
# ==================================================================
@pytest.mark.parametrize("name", PAGES_ONLY)
def test_pages_carry_the_toggle_and_the_narrow_screen_rule(name, pages):
    """토글은 페이지당 하나, 자바스크립트 0줄, 좁은 화면에서 헤더와 겹치지 않는다."""
    html = pages[name]
    assert html.count('id="lang-sw"') == 1 and html.count('for="lang-sw"') == 1
    assert "<script" not in html and "onclick" not in html
    assert "@media(max-width:640px)" in html and "position:static" in html
    assert html.index('id="lang-sw"') < html.index('class="wrap"')


@pytest.mark.parametrize("name", PAGES_ONLY)
def test_page_title_and_chart_text_stay_english(name, pages):
    """<title> 과 차트 안 글자는 영어로 통일한다(차트는 이중화하지 않는다)."""
    title = re.search(r"<title>(.*?)</title>", pages[name], re.S).group(1)
    assert not _HANGUL.search(title), f"{name}: <title> 에 한글 {title!r}"
    for alt in re.findall(r'<img[^>]*alt="([^"]*)"', pages[name]):
        assert not _HANGUL.search(alt), f"{name}: 차트 alt 에 한글 {alt!r}"


def test_module_does_not_wire_the_toggle_itself():
    """토글 배선은 껍데기(report_common) 한 곳뿐 — 여기서 또 심으면 개수가 어긋난다."""
    src = (ROOT / "mrl" / "report_p3.py").read_text(encoding="utf-8")
    assert "LANG_TOGGLE_HTML" not in src and "lang-sw" not in src


# ==================================================================
# 5. 계산·산출물은 건드리지 않았다 (표현 전용 작업)
# ==================================================================
def test_this_is_a_presentation_only_change():
    """상수·임계값은 config·mrl.sizing 에서만 온다 — 리포트가 숫자를 새로 정하지 않는다."""
    assert report_p3.P3_REGIME_CUTS == (1.0 / 3.0, 2.0 / 3.0)
    assert sizing.n_params() == 0
    assert report.p3_param_line() == ("생산 확률 적합 4/5 (Phase 2) · Phase 3 추가 0 · "
                                      "그림자 H: Platt 2 (K_s) + HMM θ 12 (K_u) · 예산 밖 HAR OLS 4 · v0 Platt 2")
    assert report_p3.P3_PARAM_LINE_FMT_EN.count("{") == 5      # 같은 숫자를 같은 자리에 넣는다
    assert len(report_p3.P3_HONESTY_ITEMS) == len(report_p3.P3_HONESTY_ITEMS_EN)
    assert len(report_p3.P3_REGIME_LEVELS) == len(report_p3.P3_REGIME_LEVELS_EN)
    assert set(report_p3.P3_MEMBER_KO) == set(report_p3.P3_MEMBER_EN)
    assert set(report_p3.P3_REASON_KO) == set(report_p3.P3_REASON_EN)
    assert set(report_p3._TRACK_PANEL_TITLES) == set(report_p3._TRACK_PANEL_TITLES_EN)


def test_page_sections_and_ids_are_unchanged(pages):
    """섹션 id 는 그대로 — 링크와 스크립트 검사(§10)가 깨지지 않는다."""
    for name in PAGES_ONLY:
        for i in range(1, 10):
            assert f'id="s{i}"' in pages[name], f"{name}: s{i}"


def test_effective_mode_dict_gained_only_a_translation_key(no_kill):
    """p3_effective_mode 의 결과에 영어 사유가 더해졌을 뿐, 판정 키는 그대로다."""
    eff = report.p3_effective_mode({"p2_deploy_mode": "info_only", "p3_deploy_mode": "tones"}, "tones")
    assert eff["mode"] == "info_only" and eff["tones"] is False
    assert set(eff) == {"mode", "tones", "p2", "p3", "killed", "kill_state", "caller", "sources",
                        "reasons", "reasons_en"}
    assert len(eff["reasons"]) == len(eff["reasons_en"])
    assert not any(_HANGUL.search(r) for r in eff["reasons_en"])


def test_empty_inputs_still_say_it_out_loud_in_both_languages(tmp_path, no_kill):
    """빈 자료여도 '…없습니다' 를 두 언어로 말한다 — 조용히 비우지 않는다."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p1, p2, p3 = (tmp_path / "a.html", tmp_path / "b.html", tmp_path / "c.html")
        report.render_sizing_report({}, {}, {}, p1, {})
        report.render_regime_report({}, p2, {})
        report.render_track_record({}, {}, p3, {})
    for p in (p1, p2, p3):
        html = p.read_text(encoding="utf-8")
        assert "없습니다" in _korean_screen(html) or "없음" in _korean_screen(html)
        assert re.search(r"\bno\b", " ".join(_spans(html, "en")), re.I)
        n_ko, n_en, _ = i18n.check_pairs(html)
        assert n_ko == n_en


def test_charts_and_constants_are_untouched():
    """차트 키와 규칙 상수(P3)는 표현 작업에서 건드리지 않는다."""
    assert P3["w_min"] == 0.25 and P3["w_max"] == 1.0
    assert ENSEMBLE_P3["fresh_blocks_min"] >= 1
    src = (ROOT / "mrl" / "report_p3.py").read_text(encoding="utf-8")
    for key in ("w_path", "cum_returns", "drawdown", "ladder_frontier", "members_band", "hmm_phigh",
                "era_auc", "bins_live", "bss_vs_hist", "alarms_timeline", "kill_countdown"):
        assert f'"{key}"' in src, key
