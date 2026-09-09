# -*- coding: utf-8 -*-
"""한/영 토글과 쉬운 한국어 도구 — STYLE_I18N.md 계약의 구현.

이 모듈은 표현(presentation) 전용이다. 계산·임계값·산출물 스키마에는 손대지 않는다.

담는 것
    bi(ko, en)          한 문장을 두 언어로 심는다 → 화면엔 고른 언어만 보인다 (STYLE_I18N.md §1)
    bi_html(ko, en)     같은 일을 하되 안에 태그(<b> 등)를 그대로 넣을 때 쓴다 (이스케이프 안 함)
    bi_attr(ko, en)     title/aria/alt 처럼 <span> 을 넣을 수 없는 자리 — 한국어를 고르고, 그 사실을 기록한다
    bi_th(ko, en)       표 머리글 <th> 한 칸
    bi_ths(pairs)       표 머리글 한 줄
    LANG_CSS            토글 CSS (숨긴 체크박스 #lang-sw + 알약 라벨 + 좁은 화면 규칙)
    LANG_TOGGLE_HTML    <body> 바로 안, 페이지 래퍼 앞에 넣는 체크박스+라벨 마크업
    TERMS               용어 대응표 (STYLE_I18N.md §3): 원문 → (쉬운 한국어, English)
    find_hard_terms()   독자에게 그대로 보이는 어려운 용어를 찾는 린트 (테스트가 실패시키는 용도)
    check_pairs()       한/영 스팬 짝·언어 섞임 검사 (STYLE_I18N.md §5)

원칙
    * 자바스크립트 0줄. CSS 체크박스만 쓴다 — 주피터 뷰어·오프라인·메일 미리보기에서도 동작한다.
    * 기본은 한국어. 체크되면 영어.
    * 숫자·티커·날짜·수식 기호는 번역하지 않는다. 라벨과 문장만 감싼다.
    * 쉬운 말로 바꾸되 숫자·조건·한계(표본 수·구간·post hoc·info_only·"투자 조언 아님")는 절대 빼지 않는다.
"""
from __future__ import annotations

import html as _h
import re

__all__ = [
    "esc", "bi", "bi_html", "bi_attr", "bi_th", "bi_ths", "ATTR_KO_ONLY",
    "LANG_CSS", "LANG_CSS_EN", "LANG_TOGGLE_HTML", "lang_toggle_html", "LANG_DEFAULT",
    "TERMS", "HARD_PATTERNS", "ALLOWED_LATIN",
    "visible_text", "find_hard_terms", "check_pairs", "count_pairs", "ENGLISH_FUNCTION_WORDS",
]

LANG_DEFAULT = "ko"          # 체크박스가 꺼진 상태 = 한국어


# ==================================================================
# 1. 두 언어 심기
# ==================================================================
def esc(x) -> str:
    """HTML 이스케이프. None 은 빈 문자열."""
    return _h.escape("" if x is None else str(x), quote=True)


def bi(ko: str, en: str) -> str:
    """한/영 두 벌을 함께 심는다. 화면엔 선택된 언어만 보인다 (STYLE_I18N.md §1)."""
    return f'<span class="lg ko">{esc(ko)}</span><span class="lg en">{esc(en)}</span>'


def bi_html(ko: str, en: str) -> str:
    """bi() 와 같지만 안쪽 HTML 을 그대로 둔다 — <b>·<a> 를 품은 문장에만 쓴다.

    호출자가 이스케이프 책임을 진다(사용자 입력 금지, 우리가 만든 마크업만).
    """
    return f'<span class="lg ko">{ko}</span><span class="lg en">{en}</span>'


ATTR_KO_ONLY: list[tuple[str, str]] = []      # bi_attr 로 한국어만 나간 자리 기록(감사용)


def bi_attr(ko: str, en: str) -> str:
    """속성값(title·aria-label·alt)처럼 <span> 두 개를 넣을 수 없는 자리.

    한계를 명시한다: **한국어를 고른다.** 영어 화면에서도 이 속성만은 한국어로 남는다.
    영어 쪽 문구는 버리지 않고 ATTR_KO_ONLY 에 기록해 나중에 감사할 수 있게 한다.
    (보이는 본문은 언제나 bi() 로 두 언어를 함께 심으므로, 이 자리는 보조 설명에만 쓴다.)
    """
    pair = (str(ko), str(en))
    if pair not in ATTR_KO_ONLY:
        ATTR_KO_ONLY.append(pair)
    return esc(ko)


def bi_th(ko: str, en: str, num: bool = False) -> str:
    """표 머리글 한 칸. 셀 값은 그대로 두고 열 제목만 이중화한다 (STYLE_I18N.md §1)."""
    cls = ' class="num"' if num else ""
    return f"<th{cls}>{bi(ko, en)}</th>"


def bi_ths(pairs, num_from: int | None = None) -> str:
    """표 머리글 한 줄. pairs = [(ko, en), ...]. num_from 부터는 숫자 열(오른쪽 정렬)."""
    out = []
    for i, p in enumerate(pairs):
        ko, en = (p[0], p[1]) if isinstance(p, (tuple, list)) else (p, p)
        out.append(bi_th(ko, en, num=(num_from is not None and i >= num_from)))
    return "<tr>" + "".join(out) + "</tr>"


# ==================================================================
# 2. 토글 (market-brief 대시보드와 같은 구조 — 자바스크립트 0줄)
# ==================================================================
# 구조:  <body> <input id="lang-sw"> <label class="lang-btn"> <div class="wrap"> ... </div> </body>
#        체크박스가 래퍼의 앞 형제라서 ~ 결합자로 본문 전체의 언어를 바꿀 수 있다.
LANG_CSS = r"""
/* 언어 토글: 숨긴 체크박스 + :checked (JS 없이 동작 — 주피터 뷰어·오프라인 호환) */
body{position:relative}
#lang-sw{position:absolute;left:-9999px;width:1px;height:1px;opacity:0}
.lg.en{display:none}
#lang-sw:checked ~ * .lg.ko{display:none}
#lang-sw:checked ~ * .lg.en{display:inline}
.lang-btn{position:absolute;top:18px;right:16px;right:calc(50% - min(500px, 50%) + 16px);z-index:5;display:inline-flex;border:1px solid var(--line);border-radius:999px;overflow:hidden;cursor:pointer;font-size:12px;user-select:none;background:var(--card)}
.lang-btn .opt{padding:5px 12px;color:var(--mut);text-decoration:none}
.lang-btn a.opt{color:inherit}
/* 넓은 화면에선 알약이 절대위치로 뜬다 — 헤더가 그만큼(알약 실측 135px) 오른쪽을 비워 두지 않으면 제목 첫 줄을 불투명 알약이 덮는다 */
.wrap>header{padding-right:150px}
.lang-btn .opt-ko{background:#1e2a40;color:var(--tx);font-weight:600}
#lang-sw:checked ~ .lang-btn .opt-ko{background:transparent;color:var(--mut);font-weight:400}
#lang-sw:checked ~ .lang-btn .opt-en{background:#1e2a40;color:var(--tx);font-weight:600}
/* 좁은 화면: 절대위치 알약이 헤더 첫 줄과 겹치므로 흐름 배치(헤더 위, 오른쪽 정렬)로 내린다 — market-brief 에서 겪은 문제 */
@media(max-width:640px){.lang-btn{position:static;display:flex;width:max-content;margin:16px 16px 0 auto}.wrap>header{padding-right:0}}
"""

LANG_TOGGLE_HTML = (
    '<input type="checkbox" id="lang-sw" class="lang-sw" aria-label="한국어 / English">'
    '<label for="lang-sw" class="lang-btn mono">'
    '<span class="opt opt-ko">한국어</span><span class="opt opt-en">English</span></label>'
)

# 영어판 껍데기: 기본 표시를 뒤집고 체크박스의 뜻도 함께 뒤집는다(알약이 그대로 작동하도록).
# 자매 문서(`<stem>.en.html`)가 없거나 file:// 로 열어도 알약을 눌러 그 자리에서 언어를 바꿀 수 있다.
LANG_CSS_EN = (".lg.ko{display:none}.lg.en{display:inline}"
               "#lang-sw:checked ~ * .lg.en{display:none}"
               "#lang-sw:checked ~ * .lg.ko{display:inline}"
               ".lang-btn .opt-ko{background:transparent;color:var(--mut);font-weight:400}"
               ".lang-btn .opt-en{background:#1e2a40;color:var(--tx);font-weight:600}"
               "#lang-sw:checked ~ .lang-btn .opt-ko{background:#1e2a40;color:var(--tx);font-weight:600}"
               "#lang-sw:checked ~ .lang-btn .opt-en{background:transparent;color:var(--mut);font-weight:400}")


def lang_toggle_html(stem: str = "index") -> str:
    """알약 두 칸을 **자매 문서로 가는 링크**로 만든다 — 링크를 타고 가도 고른 언어가 유지된다.

    여러 페이지로 나뉜 뒤 생긴 문제(문서마다 체크박스가 처음부터 다시 시작한다)를 자바스크립트 0줄로
    푼다: 언어를 파일 이름에 담는다(`sizing_p3.html` / `sizing_p3.en.html`). 숨긴 체크박스와
    `label[for=lang-sw]` 는 그대로 두어, 링크가 없는 환경에서도 그 자리에서 언어를 바꿀 수 있다.
    """
    s = esc(stem)
    return ('<input type="checkbox" id="lang-sw" class="lang-sw" aria-label="한국어 / English">'
            '<label for="lang-sw" class="lang-btn mono">'
            f'<a class="opt opt-ko" href="{s}.html">한국어</a>'
            f'<a class="opt opt-en" href="{s}.en.html">English</a></label>')


# ==================================================================
# 3. 용어 대응표 (STYLE_I18N.md §3) — 원문 → (쉬운 한국어, English)
# ==================================================================
TERMS: dict[str, tuple[str, str]] = {
    "기저율": ("평소 비율", "base rate (how often it normally happens)"),
    "Brier skill": ("정확도 점수(기준선 대비)", "accuracy score vs a naive guess"),
    "기후학(climatology)": ("평소 평균", "the long-run average"),
    "walk-forward": ("과거만 보고 미래를 맞혀 본 방식", "trained only on the past, tested on the future"),
    "홀드아웃": ("손대지 않고 남겨둔 최근 구간", "untouched recent data"),
    "퍼지(purge)": ("겹치는 구간 제거", "removing overlapping days"),
    "블록 부트스트랩": ("구간을 잘라 다시 계산한 범위", "resampled range"),
    "히스테리시스": ("한 번 바뀌면 쉽게 되돌리지 않기", "requires a clear move to switch back"),
    "dwell / 체류": ("최소 유지 일수", "minimum days before switching back"),
    "결정층": ("판정 규칙", "the decision rule"),
    "사다리(M0~M3)": ("단계별 모델", "model steps"),
    "보정(calibration)": ("확률을 실제와 맞추기", "making the stated % match reality"),
    "배포/배치(deploy)": ("실제 사용", "in use"),
    "info_only": ("참고용만 (실제 사용 안 함)", "information only (not in use)"),
    "톤": ("신호등 판정", "traffic-light call"),
    "노출/비중": ("주식 비중", "how much to hold in stocks"),
    "낙폭(drawdown)": ("고점 대비 하락폭", "drop from the peak"),
    "실현변동성": ("실제로 움직인 폭", "how much it actually moved"),
    "VIX 내재 확률": ("옵션 시장이 보는 확률", "the options market's own estimate"),
    "국면(regime)": ("시장 분위기", "market mood (calm / turbulent)"),
    "앙상블": ("여러 모형의 평균", "average of several models"),
    "그림자 멤버": ("후보 모형 (아직 사용 안 함)", "candidate model (not in use yet)"),
    "킬룰": ("성적이 나쁘면 끄는 규칙", "the rule that switches it off"),
    "드리프트 경보": ("예전과 달라졌다는 경고", "drift warning"),
    "오경보": ("헛경보", "false alarm"),
    "에피소드": ("하락 사건", "a decline episode"),
    "파라미터 예산": ("조정 가능한 숫자의 한도", "how many numbers we may tune"),
    "post hoc": ("결과를 본 뒤에 정한 것", "decided after seeing the results"),
    "사전 등록": ("미리 정해둔 것", "decided in advance"),
}


def _patterns_of(key: str) -> list[str]:
    """대응표의 '원문' 칸에서 실제로 화면에 나타날 표현들을 뽑는다.

    '낙폭(drawdown)' → ['낙폭', 'drawdown'] · 'dwell / 체류' → ['dwell', '체류']
    """
    out: list[str] = []
    for part in key.split("/"):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(.*?)\s*\((.+)\)$", part)
        if m:
            head, inner = m.group(1).strip(), m.group(2).strip()
            if head:
                out.append(head)
            if inner and not inner.startswith("M0"):     # 'M0~M3' 는 표현이 아니라 이름이라 검사 대상이 아니다
                out.append(inner)
        else:
            out.append(part)
    return [p for p in out if p]


HARD_PATTERNS: dict[str, str] = {}                  # 화면에 보이면 안 되는 표현 → 대응표의 원문 키
for _k in TERMS:
    for _p in _patterns_of(_k):
        HARD_PATTERNS.setdefault(_p, _k)


# ==================================================================
# 4. 린트 — 독자에게 보이는 글자만 검사한다
# ==================================================================
_TAG = re.compile(r"<[^>]*>")
_STYLE = re.compile(r"<(style|script)\b.*?</\1>", re.S | re.I)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_PAREN = re.compile(r"[(（][^()（）]*[)）]")
_WS = re.compile(r"\s+")


def visible_text(html: str) -> str:
    """독자가 실제로 읽는 글자만 남긴다: <style>·주석 제거 → 태그를 공백으로 → 엔티티 복원 → 공백 정리."""
    s = _STYLE.sub(" ", html or "")
    s = _COMMENT.sub(" ", s)
    s = _TAG.sub(" ", s)
    s = _h.unescape(s)
    return _WS.sub(" ", s).strip()


def find_hard_terms(html: str, *, allow_in_parens: bool = True) -> dict[str, int]:
    """화면에 그대로 남은 어려운 용어를 센다 → {보인 표현: 횟수}. 비어 있으면 통과.

    STYLE_I18N.md §2.1 "먼저 뜻, 그다음 용어" 를 그대로 따른다:
    괄호 안의 용어는 허용이므로(예: `기준선보다 9.1% 더 정확 (Brier skill +0.091)`)
    기본값에서는 괄호 구간을 먼저 지우고 센다.
    """
    text = visible_text(html)
    if allow_in_parens:
        prev = None
        while prev != text:                       # 중첩 괄호를 안쪽부터 걷어낸다
            prev = text
            text = _PAREN.sub(" ", text)
    # 승인된 '쉬운 한국어' 가 원문 표현을 품는 경우가 있다 (예: '비중' → '주식 비중').
    # 대체어를 먼저 지워야 "이미 고친 문구" 를 위반으로 세지 않는다 — 남는 것만 진짜 날것이다.
    for _ko, _ in TERMS.values():
        if _ko in text:
            text = text.replace(_ko, " ")
    found: dict[str, int] = {}
    for pat in HARD_PATTERNS:
        n = text.count(pat)
        if n:
            found[pat] = n
    return found


# 한국어 스팬 안에 남아도 되는 라틴 문자 — 티커·약어·고유명사·식별자 (STYLE_I18N.md §5 화이트리스트)
#
# 여기 있는 낱말은 "번역하지 않는 이름" 이다: 신호등 판정 이름·JSON 키·열 이름·라이브러리 이름처럼
# 화면에 그대로 나와야 뜻이 통하는 것들. 영어 '문장' 이 한국어 칸에 남는 것은 아래 _looks_english()
# 가 따로 잡는다 — 그래서 이 목록을 늘려도 진짜 번역 누락은 계속 걸린다.
ALLOWED_LATIN = {
    "spy", "vix", "vxx", "qqq", "cnn", "fang", "macd", "har", "hmm", "auc", "bgk", "ewma",
    "brier", "wilson", "phase", "sharpe", "cagr", "github", "gmail", "market", "risk", "lab",
    "info_only", "post", "hoc", "walk", "forward", "dwell", "utc", "csv", "json", "html",
    # 신호등 판정·상태 이름 (계산층의 식별자 — 표·문장에 그대로 나온다)
    "buy", "hold", "neutral", "caution", "reduce", "normal", "tones", "calm", "turbulent",
    # 두 가지 계산 방식 이름
    "faithful", "completed",
    # 라이브러리·프로젝트 이름
    "numpy", "pandas", "scipy", "yfinance", "scikit-learn", "matplotlib", "mrl", "market-risk-lab",
    # 산출물 키·설정값 이름 (밑줄 없는 것만 — 밑줄 있는 것은 _is_identifier 가 알아서 통과시킨다)
    "deploy", "stage", "skill", "bss", "acceptance", "spec", "exit", "verdict", "guard", "done",
    "default", "wide", "lag", "amended", "clip", "log", "smoothed", "filter", "reason", "track",
    "countdown", "reliability", "lookback", "assess", "composite", "overall", "ens", "sizing",
    "regime", "ensemble", "scenarios", "kill", "drift", "alarms", "ledger", "registry", "admission",
}

# 영어 '문장' 을 알려 주는 기능어 — 식별자 목록에는 거의 나오지 않고 산문에는 반드시 나온다.
ENGLISH_FUNCTION_WORDS = {
    "the", "and", "of", "is", "are", "was", "were", "to", "for", "with", "that", "this", "from",
    "not", "but", "than", "then", "only", "when", "which", "would", "have", "has", "had", "been",
    "they", "their", "its", "it", "in", "on", "at", "by", "as", "or", "if", "so", "we", "you",
    "per", "into", "over", "under", "about", "after", "before", "how", "what", "why", "all",
    "any", "each", "more", "most", "less", "least", "same", "other", "such", "does", "did",
    "can", "could", "should", "must", "may", "might", "will", "shall", "being", "there", "here",
    "also", "both", "very", "much", "many", "few", "some", "still", "every", "because", "while",
}


def _is_identifier(w: str) -> bool:
    """식별자처럼 생겼으면 True — 밑줄·숫자·하이픈이 섞인 낱말은 번역 대상이 아니다."""
    return ("_" in w) or any(ch.isdigit() for ch in w) or w.lower() in ALLOWED_LATIN


def _looks_english(text: str) -> list[str]:
    """한국어 칸에 영어 '문장' 이 남았으면 근거 낱말을 돌려준다. 식별자 나열은 잡지 않는다.

    두 가지 신호 중 하나면 문장으로 본다:
      1) 기능어(the·of·is…)가 2개 이상 — 산문의 확실한 표시
      2) 식별자가 아닌 소문자 낱말이 4개 이상 이어짐
    """
    words = [w for w in _LATIN_WORD.findall(text) if w.islower() and len(w) >= 2]
    funcs = [w for w in words if w in ENGLISH_FUNCTION_WORDS]
    if len(funcs) >= 2:
        return funcs[:4]
    plain = [w for w in words if len(w) >= 3 and not _is_identifier(w)]
    return plain[:4] if len(plain) >= 4 else []
_HANGUL = re.compile(r"[가-힣]")
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z_'-]*")
_SPAN_OPEN = re.compile(r'<span\b[^>]*class="lg (ko|en)"[^>]*>', re.I)


def _spans(html: str) -> list[tuple[str, str]]:
    """(언어, 안쪽 HTML) 목록. 안에 <span> 이 겹쳐 있어도 깊이를 세어 올바르게 끊는다."""
    out: list[tuple[str, str]] = []
    src = html or ""
    for m in _SPAN_OPEN.finditer(src):
        lang = m.group(1).lower()
        i, depth = m.end(), 1
        while depth and i < len(src):
            nxt = src.find("<span", i)
            end = src.find("</span>", i)
            if end < 0:
                break
            if 0 <= nxt < end:
                depth += 1
                i = nxt + 5
            else:
                depth -= 1
                i = end + 7
        out.append((lang, src[m.end():max(m.end(), i - 7)]))
    return out


def count_pairs(html: str) -> tuple[int, int]:
    """(한국어 스팬 수, 영어 스팬 수)."""
    spans = _spans(html)
    return sum(1 for lang, _ in spans if lang == "ko"), sum(1 for lang, _ in spans if lang == "en")


def check_pairs(html: str) -> tuple[int, int, list[str]]:
    """STYLE_I18N.md §5 검사 → (한국어 스팬 수, 영어 스팬 수, 문제 목록).

    문제로 잡는 것
      1) 두 언어 스팬 개수가 다르다 (한쪽만 심었다)
      2) 영어 스팬 안에 한글이 남았다
      3) 한국어 스팬 안에 영어 문장이 남았다 — 소문자 라틴 낱말 기준,
         티커·약어·고유명사(대문자/첫 글자 대문자)와 ALLOWED_LATIN 은 뺀다
    """
    spans = _spans(html)
    n_ko = sum(1 for lang, _ in spans if lang == "ko")
    n_en = sum(1 for lang, _ in spans if lang == "en")
    problems: list[str] = []
    if n_ko != n_en:
        problems.append(f"한/영 스팬 개수가 다릅니다: ko={n_ko}, en={n_en} (bi() 로 두 벌을 함께 심어야 합니다)")
    for lang, inner in spans:
        text = visible_text(inner)
        if not text:
            continue
        if lang == "en" and _HANGUL.search(text):
            problems.append(f"영어 칸에 한글이 남았습니다: {text[:60]!r}")
        elif lang == "ko":
            bad = _looks_english(text)
            if bad:
                problems.append(f"한국어 칸에 영어 문장이 남았습니다: {text[:60]!r} ({', '.join(bad)})")
    return n_ko, n_en, problems
