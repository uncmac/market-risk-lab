# -*- coding: utf-8 -*-
"""Phase 3 페이지 — 비중 카드 · docs/sizing_p3.html · docs/regime_p3.html · docs/track_record.html · 차트.

계산·임계값·산출물 스키마는 mrl/report.py 에서 옮겨온 그대로다. **바뀐 것은 표현뿐이다**
(ARCHITECTURE_PHASE3.md §10 · STYLE_I18N.md):

    * 독자에게 보이는 문구를 고등학생·대학생이 바로 읽히는 한국어로 다시 썼다.
    * 같은 문구의 영어 판을 `i18n.bi(ko, en)` 로 함께 심는다 → 오른쪽 위 토글이 언어를 바꾼다.
    * **정직성은 낮추지 않았다**: 숫자·표본 수(n·n_eff)·구간·"결과를 본 뒤 정한 것(post hoc)"·
      참고용만(info_only)·"투자 조언 아님" 은 두 언어에서 모두 그대로 남는다. 쉬운 말로 바꾸면
      뜻이 흐려지는 사양 문구(VALIDATION §7 등)는 괄호 안에 원문을 그대로 붙여 둔다.

표현 규칙(STYLE_I18N.md 요약)
    * 먼저 뜻, 그다음 용어 — 어려운 말은 괄호 안에만 남긴다.
    * 숫자·티커·날짜·해시·식별자(deploy_mode, sizing_sha …)는 번역하지 않는다 → 스팬 밖에 둔다.
      그래야 두 언어에서 **같은 값**이 보이고, 값이 라벨과 붙어 있는 문장은 두 벌을 각각 만든다.
    * 표는 열 제목만 두 벌로 만들고 셀 값은 건드리지 않는다.
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

# _img·_records_table·_kv_table·_label 은 이 모듈에서 한/영 판(_img_bi·_p3_table·_p3_kv·_label_bi)으로 대체했지만,
# 기존 테스트가 monkeypatch.setattr(report, "...") 로 값을 심을 때 이 이름 공간에도 닿아야 하므로 import 는 남긴다.
from mrl.report_common import (LABELS, PALETTE, _esc, _fmt, _img, _is_nan, _is_num, _kv_table, _label, _legend, _new_fig,
                               _now_str, _plain_log_axis, _png, _records_table, _series_close, _to_records, _warn_list,
                               _write_html)
from mrl.report_p2 import (INFO_DISPLAY_EN, INFO_DISPLAY_LABEL, _ALLOC_KEYS, _BLOCK_ORDER, _ERA_ORDER, _LADDER_ORDER,
                           _RELIAB_ORDER, _first, _fmt_p2, _fnum, _ladder_records, _pct1, _pp1, _v0_line_html,
                           acceptance_verdict_line, deployment_of, load_summary_v0, v0_headline)
from mrl.i18n import bi, bi_html, bi_th


# ------------------------------------------------------------------
# 한/영 조각 helper (STYLE_I18N.md §1) — 라벨·문장만 감싸고 값은 스팬 밖에 둔다
# ------------------------------------------------------------------
def _n(ko: str, en: str) -> str:
    """작은 설명 줄 한 개."""
    return f'<div class="note">{bi(ko, en)}</div>'


def _n_html(ko: str, en: str) -> str:
    """안에 <b>·<a> 를 품은 설명 줄(호출자가 이스케이프 책임)."""
    return f'<div class="note">{bi_html(ko, en)}</div>'


def _n_raw(txt: str) -> str:
    """번역하지 않는 줄(식별자·해시·경로·다른 모듈이 만든 원문).

    영어 화면에도 한국어 원문이 그대로 남으므로 **그 사실을 밝힌다** — 영어 독자가 '이 페이지는 다 읽었다'
    고 착각하지 않게 한다(report_p2.py:429 규약과 같은 모양: `_raw()` 로 인용 표시 + 옆에 고지).
    """
    return (f'<div class="note">{bi("계산이 남긴 한국어 원문", "raw Korean written by the pipeline")}: '
            f'<span class="raw mono">{_esc(txt)}</span></div>')


def _raw(txt) -> str:
    """계산이 남긴 원문(한국어)을 손대지 않고 인용 표시만 한다 (report_p2.py:436 과 같은 규약)."""
    return f'<span class="raw mono">{_esc(txt)}</span>'


def _warns_bi(items) -> str:
    """경고·기록 목록 — 계산이 남긴 한국어 원문이라 번역하지 않되, **그 사실을 밝히고** 인용 표시한다.

    (report_v0.py 의 경고 목록과 같은 규칙. 영어 독자가 읽을 수 없는 줄이 남는다는 것을 숨기지 않는다.)
    """
    rows = [str(w) for w in (items or []) if w is not None and str(w).strip()]
    if not rows:
        return f'<div class="note">{bi("경고 없음", "no warnings")}</div>'
    return (_n("아래 줄은 계산 과정이 남긴 기록이라 한국어 원문 그대로 둡니다.",
               "The lines below are raw diagnostics written by the pipeline and are kept in the original "
               "Korean.")
            + '<ul class="warnlist">' + "".join(f"<li>{_raw(w)}</li>" for w in rows) + "</ul>")


def _yn_bi(v) -> str:
    """예/아니오 같은 참·거짓 값 한 쌍 — 태그처럼 표 밖에 쓰는 자리에서 쓴다(셀 값은 그대로 둔다)."""
    if v is None:
        return "—"
    if isinstance(v, str):
        s = v.strip()
        if s in ("예", "true", "True", "yes"):
            return bi("예", "yes")
        if s in ("아니오", "false", "False", "no"):
            return bi("아니오", "no")
        return _esc(s)
    return bi("예", "yes") if bool(v) else bi("아니오", "no")


def _bi_or_raw(ko: str, en: str | None) -> str:
    """계산이 남긴 한국어 원문 + 영어 판(_hz_footnote 와 같은 방식).

    짝을 확인할 수 없으면 **영어를 지어내지 않고** 원문만 인용 표시한다 — 지어낸 영어보다 읽지 못하는
    한국어가 정직하다(HARD RULE 1: 한계를 잃지 않는다).
    """
    return bi(ko, en) if en else _raw(ko)


def _fresh_blocks_line(fb) -> str:
    """'손대지 않은 새 자료' 진행도를 문장으로 — 파이썬 dict 를 그대로 찍지 않는다.

    숫자(모은 것·필요한 것)와 계산층이 붙인 note 를 하나도 빼지 않는다. note 는 파이프라인이
    쓴 한국어 원문이라 번역하지 않고 그대로 두고, 그 사실을 밝힌다(_bi_warns 와 같은 규칙).
    """
    if not isinstance(fb, dict):
        return _n_raw(str(fb))
    done, need = fb.get("complete"), fb.get("need")
    note = str(fb.get("note") or "").strip()
    if done is None or need is None:                     # 모양이 다르면 원문을 그대로 보여 준다(숨기지 않는다)
        return _n_raw(str(fb))
    ko = f"손대지 않은 새 자료: {need}개 가운데 {done}개 모았습니다."
    en = f"Untouched new data: {done} of the {need} blocks needed have been collected."
    out = _n(ko, en)
    if note:
        # 원문은 스팬 밖에 둔다 — 영어 화면에서도 같은 한국어 원문이 그대로 보여야 한다
        out += (f'<div class="note">{bi("덧붙인 기록(계산이 남긴 한국어 원문)", "note recorded by the pipeline (raw Korean)")}'
                f": {_raw(note)}</div>")
    return out


def _q(ko: str, en: str) -> str:
    return f'<div class="quote">{bi(ko, en)}</div>'


def _h2(ko: str, en: str) -> str:
    return f"<h2>{bi(ko, en)}</h2>"


def _h3(ko: str, en: str) -> str:
    return f"<h3>{bi(ko, en)}</h3>"


def _tag(ko: str, en: str, cls: str = "") -> str:
    return f'<span class="tag{cls}">{bi(ko, en)}</span>'


def _tag_html(ko: str, en: str, cls: str = "") -> str:
    return f'<span class="tag{cls}">{bi_html(ko, en)}</span>'


def _img_bi(png: bytes | None, cap_ko: str, cap_en: str, alt: str) -> str:
    """차트 + 한/영 설명. 차트 **안**의 글자는 영어로 통일한다(STYLE_I18N.md §1) — alt 도 영어."""
    if not png:
        return ('<div class="note">' + bi(f"차트 없음 — {cap_ko}", f"chart missing — {cap_en}") + "</div>")
    b64 = base64.b64encode(png).decode("ascii")
    return (f'<div class="chart"><img src="data:image/png;base64,{b64}" alt="{_esc(alt)}">'
            f'<div class="cap">{bi(cap_ko, cap_en)}</div></div>')


# ==================================================================
# Phase 3 — 비중 카드 · 비중/국면/트랙레코드 페이지 · 차트 (ARCHITECTURE_PHASE3.md §10)
# ==================================================================
# 설계 원칙(Phase 1·2 와 같다): 있는 키만 렌더링하고 없는 값은 "—", 비어 있으면 '자료 없음'.
# 추가 규칙(Phase 3 고유):
#   * 유효 배치 모드 = p2.deploy_mode == "tones" ∧ p3.deploy_mode == "tones" ∧ ¬kill (§8.3).
#     리포트는 이 값을 **스스로 계산**하고, 호출자가 준 값과 AND 한다 — 절대 올려 잡지 않는다.
#   * info_only 에서는 상태·톤·비중·D_max 사다리를 숨기고(§8.3·§15 stage 0), 변동성 단독 규칙·게이지·
#     시나리오·구간만 회색 '정보(비중 제안 아님)' 로 보인다.
#   * 지평별 표시 규칙(§8.4)의 단일 원천은 mrl.track 이다(horizon_stage/may_show/horizon_copy).
#     리포트는 규칙을 새로 쓰지 않고 그 함수를 통과시킨다.
#   * 모든 Phase 3 페이지는 summary_p2.json 의 acceptance.verdict_line 을 **그대로** 싣는다.

# VALIDATION §7 이 못 박은 문구 — 원문 그대로 둔다(§8.3 카드). 쉬운 설명은 _PLAIN 쪽이 맡는다.
P3_INFO_ONLY_LABEL = "정보 제공 전용 — 톤·비중 제안 숨김"


P3_INFO_ONLY_PLAIN = ("참고용으로만 보여 줍니다 — 지금은 얼마나 담으라는 제안을 하지 않습니다 "
                      "(정보 제공 전용 — 톤·비중 제안 숨김)")


P3_INFO_ONLY_EN = ("Information only — right now this card suggests neither a traffic-light call "
                   "nor how much to hold in stocks")


P3_INFO_NOT_SIZING = "정보(비중 제안 아님)"                        # §15 stage 0 — 회색 라벨(원문 유지)


P3_INFO_NOT_SIZING_PLAIN = "참고용 · 정보(비중 제안 아님)"


P3_INFO_NOT_SIZING_EN = "reference only — not a suggested amount to hold"


# §10 5 국면 게이지 라벨 — 뜻을 먼저 쓰고 사양 문구는 괄호에 남긴다
P3_SHADOW_LABEL = ("아직 쓰지 않는 후보 모형입니다 — 오늘의 확률에도, 주식을 얼마나 담을지에도, 판정 규칙에도, "
                   "성적이 나쁘면 끄는 규칙에도 들어가지 않습니다 (그림자 — 확률·비중·결정층·킬룰에 들어가지 않음)")


P3_SHADOW_LABEL_EN = ("a candidate model that is not in use yet — it feeds none of today's probability, "
                      "none of how much to hold, none of the decision rule and none of the switch-off rule")


P3_HIGH_STATE_KO = "고변동 상태"                                   # §16 9 — '위기' 가 아니다


P3_HIGH_STATE_EN = "high-swing state"


P3_HIGH_STATE_GLOSS_KO = "많이 출렁이는 시기"                       # 처음 나올 때 붙이는 쉬운 말


P3_LADDER_HIDDEN = ("지금은 단계별 예산 표를 보여 주지 않습니다 — 실제로 쓰는 모드일 때만 나옵니다. 그동안에도 "
                    "기록은 계속 남깁니다 (예산 사다리는 유효 배치 모드에서만 표시합니다 — §8.3: info_only 에서 "
                    "카드는 상태·톤·비중·D_max 사다리를 숨긴다; 장부는 계속 기록).")


P3_LADDER_HIDDEN_EN = ("The budget ladder is hidden — it appears only when the platform is actually in use "
                       "(§8.3). The ledger keeps recording either way.")


P3_CARD_FOOTNOTE = ("이 숫자는 미리 정해 둔 규칙을 SPY 한 종목 몫에만 적용한 것입니다. 집에 있는 현금과 채권은 "
                    "계산에 넣지 않았고, 투자 조언이 아닙니다.")


P3_CARD_FOOTNOTE_EN = ("These numbers apply a rule fixed in advance to the SPY sleeve only. The cash and bonds "
                       "the family holds sit outside the model, and this is not investment advice.")


# §16 1 — 모든 페이지 첫 줄
P3_FIRST_LINE = ("3단계(Phase 3)는 맞히는 실력을 더하지 못했습니다. 어떤 후보 모형도 확률을 실제와 맞춘 VIX "
                 "모형보다 정확하지 않았습니다. 기준선 대비 정확도 점수 차이는 M3→p3 −0.0002, 여러 모형의 "
                 "평균(M3·H) +0.0000, M4a 는 나아진 것이 없었습니다. 3단계가 새로 더한 것은 다섯 가지입니다: "
                 "얼마나 담을지 정하는 규칙, 앞으로 벌어질 수 있는 일의 표, 그 범위, 성적 기록, 성적이 나쁘면 "
                 "끄는 장치. 이 페이지는 그 다섯 가지만 주장합니다.")


P3_FIRST_LINE_EN = ("Phase 3 adds no skill — it does not guess better than before. No candidate beat the VIX "
                    "model whose probabilities were matched to reality (M3→p3 −0.0002, ens(M3,H) +0.0000, "
                    "M4a no gain). What it does add is the rule for how much to hold, the scenario tables, "
                    "the ranges, the live record and the switch-off machinery — and that is all this page claims.")


# §6.5 (a)~(d) 의 카드용 축약 (§10 2)
P3_SIZING_SHORT = ("변동성을 맞춰 주면 조정할 숫자가 0개인 규칙도 p2 판정 규칙과 성적이 같다 · 하락폭을 줄이는 "
                   "대가는 그냥 들고 있을 때보다 해마다 −3.6%p (달력 해의 약 90% 에서 뒤진다)")


P3_SIZING_SHORT_EN = ("Once volatility is matched, a rule with zero fitted numbers performs like the p2 "
                      "decision rule · controlling the drop costs about −3.6pp a year versus simply holding "
                      "(behind in about 90% of calendar years)")


# mrl.track.HORIZON_FOOTNOTE 의 영어 판 (§8.4 영구 각주 — 규칙의 단일 원천은 mrl.track 이다)
HORIZON_FOOTNOTE_EN = ("A live record gains only about 12 independent windows and about 1 decline episode a "
                       "year. The judgement dates are 36 months (first) and 8 episodes or 60 months (second); "
                       "until then this page is a diary, not a verdict.")


# ── 계산층이 쓴 한국어 원문의 영어 판 (HORIZON_FOOTNOTE_EN 과 같은 방식) ──────────────
# 왜 여기 두나: 원문의 단일 원천은 mrl.track·mrl.sizing·mrl.scenarios 이고 거기 문자열은 한 글자도
# 바꾸지 않는다. 영어 독자가 **같은 한계**를 읽을 수 있도록, 같은 자리표({...})를 쓰는 영어 판만
# 리포트 층에 둔다 → 숫자는 구조적으로 같을 수밖에 없다(정직성: 표본 수·구간·한계를 잃지 않는다).
HORIZON_SENTENCES_EN = {
    "0-3m": ("Three independent windows — no conclusion at all. All that can be seen so far is that the "
             "pipeline runs every day and the record piles up correctly. In the past the accuracy score "
             "over a 3-month window ran from −0.17 to +0.85 (10th-90th percentile)."),
    "6m": ("Six windows: the interval is wider than the accuracy score itself (36% of past 6-month windows "
           "were negative; 10th-90th percentile −0.12 to +0.81)."),
    "12m": ("One year cannot say whether a model lives or dies: for the same model the past 1-year accuracy "
            "score (BSS) ran from −0.07 (bottom 10%) to +0.50, and the low end of the interval was above 0 "
            "in only 26-29% of them. The switch-off rule waits 36 months."),
    "36m": ("With 8 or more decline episodes only a big failure can be caught; an edge of +1 to 2% cannot be "
            "certified (P2 §16.4)."),
}


# mrl.scenarios.SPREAD_SENTENCE 의 쉬운 한국어 판 (원문 상수는 mrl.scenarios 에서 한 글자도 바뀌지 않는다).
# 고정 문장이라 매 실행마다 숫자가 달라지지 않는다 → 뜻·숫자를 그대로 옮길 수 있다.
SPREAD_SENTENCE_PLAIN = ("어떤 시장 분위기에서도, 어떤 확률대에서도 20일 뒤 수익의 가운데값은 플러스였습니다. "
                         "이 모델은 오를지 내릴지를 맞히지 않고, 결과가 얼마나 넓게 벌어질지를 말합니다"
                         "(VALIDATION §0). 위험이 큰 구간일수록 결과의 폭은 좁아지지 않고 오히려 넓어집니다 — "
                         "크게 떨어지는 날과 크게 튀어 오르는 날이 함께 늘기 때문입니다.")


# 같은 문장의 영어 판 (§7 카드에 반드시 인쇄하는 한계 문장)
SPREAD_SENTENCE_EN = ("In every state and every probability band the median 20-day return is positive — the "
                      "model predicts how wide the outcomes are, not which direction they go (VALIDATION §0). "
                      "The higher the risk band, the wider the spread of returns, not the narrower (sharp "
                      "falls and sharp rebounds live together).")


# mrl.sizing.HONEST_READING (§6.5) 의 영어 판 — **같은 자리표**를 쓴다.
# 같은 자리표를 쓰므로 두 언어의 숫자가 어긋날 수 없다(원문을 감싸지 않고 남겨 둔 이유였던 위험이 사라진다).
HONEST_READING_EN = (
    "(a) Once volatility is matched, a volatility-target rule with 0 fitted parameters (VT14) does as well as "
    "the p2 decision rule ({vt14_cagr}/{vt14_dd} vs {p2_cagr}/{p2_dd}, {vt14_sw} vs {p2_sw} switches a year) "
    "— how much to hold does not stand on the probability model.",
    "(b) The probability layer lowers the biggest drop from the peak only when the multiply has no floor "
    "({volonly_dd} → {nofloor_dd}; average share 0.12 from 2008-09 to 2009-06 and 0.10 from 2020-03 to 05, "
    "which is effectively leaving the market). With the floor it is {adopted_dd}, the same as volatility "
    "alone ({volonly_dd}). The state multiplier is worth keeping not for the drop but because it gives the "
    "family a state and gives the switch-off rule something to test.",
    "(c) A volatility target does not stop a slow bear market (2000-02: {mild_bear_dd}) — that is why "
    "k_slow is 3.5.",
    "(d) Every rule row earns less per year (CAGR) than simply holding, and the adopted rule is behind in "
    "about {share_behind} of calendar years — what this product sells is control of the drop from the peak, "
    "and we say exactly that. Cash at 0% and a 5bp cost understate the real friction (brokerage, tax) for a "
    "family that rebalances late, once a quarter.",
)


_HONEST_TAIL_EN = "  (design-stage approximation — to be replaced by the official run)"


# 멤버 이름 — 카드에서는 언제나 괄호 안에 들어간다(먼저 뜻, 그다음 용어)
P3_MEMBER_KO = {"p2": "p2", "M1": "보정 VIX", "H": "HMM", "ens": "앙상블"}


P3_MEMBER_EN = {"p2": "p2", "M1": "VIX matched to reality", "H": "HMM", "ens": "average of the models"}


P3_REASON_KO = {"init": "첫 기록", "escalation": "위험 단계가 올라가 곧바로 낮춤", "weekly": "주간 점검",
                "hold": "그대로 둠", "input_missing": "입력값이 없어 그대로 둠 (1.0 으로 되돌리지 않음)",
                "info_only": "참고용만 (정보 제공 전용)"}


P3_REASON_EN = {"init": "first record", "escalation": "risk step rose — lowered immediately",
                "weekly": "weekly check", "hold": "left unchanged",
                "input_missing": "an input was missing — left unchanged (never snapped back to 1.0)",
                "info_only": "information only"}


P3_REGIME_CUTS = (1.0 / 3.0, 2.0 / 3.0)                    # 표시 전용 3구간 (적합 아님)


P3_REGIME_LEVELS = ("낮음", "중간", "높음")


P3_REGIME_LEVELS_EN = ("low", "medium", "high")


_KO_DOW = ("월", "화", "수", "목", "금", "토", "일")


_EN_DOW = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


# 유효 모드 사유에 붙는 '출처' 의 영어 판 (파일 경로는 번역하지 않는다)
_SOURCE_EN = {"입력": "from the input", "확인 불가": "could not be determined"}


# §14 파라미터 회계 — 카드 정직 스트립의 한 줄과 등록부 페이지의 표(문서 그대로)
P3_PARAM_LINE_FMT = ("생산 확률 적합 {k_prod}/{budget} (Phase 2) · Phase 3 추가 {k_p3} · "
                     "그림자 H: Platt {h_ks} (K_s) + HMM θ {h_ku} (K_u) · 예산 밖 HAR OLS 4 · v0 Platt 2")


P3_PARAM_LINE_FMT_EN = ("numbers fitted into today's probability {k_prod}/{budget} (Phase 2) · added in "
                        "Phase 3: {k_p3} · candidate H: Platt {h_ks} (K_s) + HMM θ {h_ku} (K_u) · "
                        "outside the budget: HAR OLS 4 · v0 Platt 2")


P3_PARAM_TABLE = (
    {"줄": "생산 확률 p2", "객체": "b0, b1(x_vix), b2(x_har), b3(x_ma)", "K_s": 4, "K_u": 0, "생산 확률에?": "예",
     "예산 줄": "VALIDATION §6 (상한 5; 1 슬롯 예약) — Phase 3 변경 없음, PARAM_COUNT==4 assert 유지"},
    {"줄": "판정 규칙", "객체": "밴드 1.5/1.2·2.5/2.0, dwell 5", "K_s": 0, "K_u": 0, "생산 확률에?": "—",
     "예산 줄": "Phase 2 상수 동결"},
    {"줄": "주식 비중 규칙(§6)", "객체": "λ .94, seed 60, 바닥 .25/상한 1, k_slow 3.5·k_fast 2.0, 격자 {6..15}%, "
                                  "D_max .35, 격자 .05, 밴드 .10, 주간+격상, 배수 = TONE_EXPOSURE, 비용 5bp·현금 0",
     "K_s": 0, "K_u": 0, "생산 확률에?": "—", "예산 줄": "§6-b: 적합 0 필수. ~40 변형을 1993~2024 로 본 뒤 "
                                                      "선택 → post hoc, 재조정 금지"},
    {"줄": "시나리오·경보·끄는 규칙", "객체": "구간·풀링 20·z·D1~D11 임계·36/8/60/12·블록 40·4,000·seed 0",
     "K_s": 0, "K_u": 0, "생산 확률에?": "—", "예산 줄": "상수(설계 단계 관측 후 고정; 경보 임계는 p5/p95 에서 → post hoc)"},
    {"줄": "그림자 M1", "객체": "b0, b1", "K_s": 2, "K_u": 0, "생산 확률에?": "아니오(#8)",
     "예산 줄": "등록 시험(Phase 2 사다리 재사용)"},
    {"줄": "그림자 H", "객체": "Platt a, b / HMM μ 4·Σ 6·A 2", "K_s": 2, "K_u": 12, "생산 확률에?": "아니오(#3a)",
     "예산 줄": "등록 시험; K_u 는 라벨을 보지 않지만 파라미터로 공개"},
    {"줄": "여러 모형을 섞는 가중치", "객체": "동일", "K_s": 0, "K_u": 0, "생산 확률에?": "—", "예산 줄": "상수"},
    {"줄": "예산 밖(Phase 2, 불변)", "객체": "HAR OLS 4 · v0 Platt 2", "K_s": "표시 전용", "K_u": 0,
     "생산 확률에?": "아니오", "예산 줄": "—"},
)


P3_PARAM_TABLE_NOTE = ("오늘의 확률에 닿는 조정값은 4개(예산 안) + 0개(Phase 3) 입니다. 가족이 보는 어떤 숫자에든 "
                       "닿는 조정값은 모두 26개 (4 + 0 + 변동성 표시 4 + 2 + 2 + 후보 모형 12 + v0 2) 이고, "
                       "그중 예산 안은 4개입니다. 조정값이 아니라 '고른 것' 도 있습니다: HMM 관측형 · Platt 특징 · "
                       "바닥에서 한 번 더 자르기 · 주간 리듬 · k_slow 3.5 · 20개씩 묶기 · 2단계 해석 · 경보 한계선. "
                       "고른 것도 결과에 영향을 주므로 함께 적어 둡니다.")


P3_PARAM_TABLE_NOTE_EN = ("Numbers fitted into today's probability: 4 (inside the budget) + 0 (Phase 3). "
                          "Numbers fitted into anything the family sees: 26 in total (4 + 0 + 4 for the "
                          "volatility display + 2 + 2 + 12 for the candidate model + 2 for v0), of which 4 "
                          "are inside the budget. Some things are choices rather than fitted numbers: the HMM "
                          "observation set, the Platt features, re-clipping at the floor, the weekly rhythm, "
                          "k_slow 3.5, pooling by 20, the two-stage reading, and the alarm thresholds. "
                          "Choices move the results too, so they are listed here as well.")


# §14 표의 열 제목 (한/영). 셀 값은 사양 문서의 표현 그대로 둔다 (STYLE_I18N.md §1: 셀은 번역하지 않는다).
P3_PARAM_TABLE_COLS = (("줄", "row"), ("객체", "what is fitted"), ("K_s", "K_s"), ("K_u", "K_u"),
                       ("생산 확률에?", "in today's probability?"), ("예산 줄", "budget note"))


# §16 정직 문구·리스크 — 모든 Phase 3 페이지의 마지막 섹션.
# 쉬운 말로 다시 썼지만 숫자·한계·"결과를 본 뒤 정했다" 는 하나도 빼지 않았다.
P3_HONESTY_ITEMS = [
    "3단계(Phase 3)는 맞히는 실력을 더하지 못했습니다. 어떤 후보 모형도 확률을 실제와 맞춘 VIX 모형보다 "
    "정확해지지 않았습니다. 기준선 대비 정확도 점수 차이는 M3→p3_A −0.0002, 여러 모형의 평균(M3·H) +0.0000, "
    "M4a 는 나아진 것 없음, 폭·신용을 더한 것은 −0.055 였습니다. 3단계의 값어치는 다섯 가지입니다: 얼마나 "
    "담을지 정하는 규칙, 앞으로 벌어질 수 있는 일의 표, 그 범위, 성적 기록, 성적이 나쁘면 끄는 장치.",
    "Phase 2 의 공식 결과가 Phase 3 의 전제다. 모델 파일(model_p2.json)이 참고용만(info_only)이면 전체가 "
    "참고용이 되므로 주식 비중·상태 카드가 뜨지 않는다. 이 결정은 Phase 3 코드가 아니라 장부에서 내려진다.",
    "이 규칙에는 값이 붙는다: 목표 변동성 10% 에서 그냥 들고 있는 것보다 해마다 약 3.6%p 덜 벌고 달력 해의 "
    "약 90% 에서 뒤진다; 현금에 이자 0% 를 가정했고(넣으면 해마다 +0.6%p); 거래비용 5bp 는 분기마다 늦게 "
    "실행하는 실제 마찰(수수료·세금)을 낮게 잡은 값이다.",
    "천천히 내려가는 약세장에서 진다: 2000~02년, 변동성만 보는 규칙은 −30.7~−34.6% 까지 갔다(목표 변동성 10%). "
    "k_slow 3.5 는 그 한 사건을 보고 맞춘 상수다 — 예산은 보장이 아니다. 조용하다가 갑자기 떨어지는 날"
    "(2018-02, 2020-02 첫 주)은 규칙이 반응하기 전에 그대로 맞는다.",
    "바닥을 둔 대가: 곱한 뒤 0.25 로 한 번 더 자르면 고점 대비 최대 하락폭이 −17.9% 에서 −23.5% 로 커진다"
    "(변동성만 보는 규칙과 같다). 상태 배수가 하락폭을 줄이는 효과는 바닥이 없을 때만 나타났고, 그 모습은 "
    "2008~09·2020 에 사실상 시장에서 빠져나가는 것이었다(0.05~0.10 만 담음).",
    "얼마나 담을지 정하는 상수는 전부 결과를 본 뒤에 정했다(post hoc — λ, 격자, 밴드, 리듬, k_slow, 바닥 처리; "
    "약 40가지 변형을 본 뒤 골랐다). 유지 조건 (a)·(b) 는 아슬아슬하게 통과해서 공식 수치에서 뒤집힐 수 있다 — "
    "그러면 카드가 숨겨지고 기록만 쌓인다.",
    "표본이 적은 것은 영원한 조건이다: 해마다 겹치지 않는 창이 약 12개, 하락 사건이 약 1개 쌓인다. 3·6개월 "
    "성적은 잡음이라(10/90분위 ±0.8) 보여 주지 않는다 — 창이 6개(n_eff < 6)보다 적으면 점수를 내지 않고, "
    "판정일 전에는 판정하는 말을 쓰지 않는다. 이건 코드가 막는다(§8.4).",
    "HMM 의 한계: 두 상태의 이름이 뒤바뀔 수 있고, 계산이 엉뚱한 답에 눌러앉을 수 있고, 한 상태가 이어질 확률이 "
    f"0.98 이라 급락을 늦게 본다. 점유 43% 는 '위기' 가 아니라 '{P3_HIGH_STATE_KO}' 다. 라벨을 보지 않고 정한 "
    "숫자 12개도 조정값으로 공개하며 오늘의 확률에는 들어가지 않는다.",
    "자료가 고쳐지는 일: Yahoo 조정 종가를 다시 내려받으면 예상 변동성·고변동 확률·주식 비중 이력이 조금 바뀐다 "
    "→ D10 이 차이를 세고 D7 은 그날 작업을 실패시킨다; 기록은 기록이라 다시 쓰지 않는다.",
    "이어 붙이는 위험: 열 이름과 spec_sha256·registry_sha·sizing_sha 가 계약이다. 어긋나면 그냥 멈춘다"
    "(exit 1) — 대신 채워 넣지 않는다. 끄는 규칙은 model_p2.json 을 건드리지 않고 세 조건이 모두 맞을 때만 "
    "'실제로 쓴다' 로 본다.",
    "손대지 않고 남겨 둔 최근 구간(2024-09-01~)은 잠금 해제 파일 없이 채점하지 않는다. 후보 모형의 그 구간 "
    "점수는 참고일 뿐 채택 근거로 쓰지 않으며, 새 자료 블록은 라이브 시작일부터만 센다.",
    "제품 위험: 확률·상태 기계·변동성 규칙 세 겹은 v0 신호등보다 설명하기 어렵다 → 카드는 숫자 하나와 문장 "
    "하나로 시작한다. 앞으로 벌어질 수 있는 일의 표에는 얇은 칸이 있다(줄임 상태 n_eff 12, 하락 사건 33/11/4) — "
    "얇은 칸을 회색으로 두고 위에서부터 묶는 규칙은 코드가 강제한다.",
    "계산 비용: 주간 작업 +4분 미만, 매일 작업 +5초 미만; 새로 설치할 것은 없다(numpy 로 짠 EM).",
]


P3_HONESTY_ITEMS_EN = [
    "Phase 3 adds no skill — it does not guess better than before. No candidate beat the VIX model whose "
    "probabilities were matched to reality (M3→p3_A −0.0002, ens(M3,H) +0.0000, M4a no gain, breadth/credit "
    "−0.055). The value is the rule for how much to hold, the scenario tables, the ranges, the live record "
    "and the switch-off machinery.",
    "Phase 2's official result is the premise for Phase 3. If the model file (model_p2.json) says information "
    "only, the whole platform is information only, so the stock-share and state cards do not appear. That "
    "decision is made in the ledger, not in Phase 3 code.",
    "The rule has a price: at a 10% volatility target it earns about 3.6 percentage points a year less than "
    "simply holding, and it lags in about 90% of calendar years; cash is assumed to earn 0% (counting it "
    "would add about +0.6pp a year); and the 5bp trading cost understates the real friction (fees, taxes) "
    "for a family that rebalances late, once a quarter.",
    "It loses in a slow bear market: through 2000-02 the volatility-only rule fell −30.7% to −34.6% "
    "(10% volatility target). k_slow 3.5 was tuned on that single episode — the budget is not a guarantee. "
    "Sudden drops out of a calm market (2018-02, the first week of 2020-02) land before the rule can react.",
    "The price of the floor: re-clipping to 0.25 after the multiply raises the biggest drop from the peak from "
    "−17.9% to −23.5% (the same as the volatility-only rule). The state multiplier only reduced the drop when "
    "there was no floor, and that shape meant leaving the market almost entirely in 2008-09 and 2020 "
    "(holding only 0.05-0.10).",
    "Every constant that sets how much to hold was decided after seeing the results (post hoc: λ, the grid, "
    "the no-trade band, the rhythm, k_slow, how the floor is applied; chosen after looking at about 40 "
    "variants). Retention conditions (a) and (b) pass only narrowly and can flip on the official numbers — "
    "then the card is hidden and only the record keeps growing.",
    "Small samples are a permanent condition: about 12 non-overlapping windows and about 1 decline episode "
    "accumulate per year. Three- and six-month scores are noise (10th/90th percentile ±0.8) so they are not "
    "shown — below six effective windows (n_eff < 6) no score is printed, and before the judgement date no "
    "judging word is used. The code enforces this (§8.4).",
    "Limits of the HMM: the two states can swap names, the fitting can settle on a wrong answer, and with a "
    "0.98 chance of staying put it sees a crash late. Occupancy of 43% is not a 'crisis' but a high-swing "
    "state. Its 12 numbers, fitted without ever seeing the labels, are disclosed as parameters and do not "
    "enter today's probability.",
    "Data gets revised: re-downloading Yahoo adjusted closes shifts the volatility forecast, the high-swing "
    "probability and the stock-share history slightly → D10 counts the mismatches and D7 fails that day's "
    "run; the record stays as recorded and is never rewritten.",
    "Integration risk: the column names and spec_sha256 / registry_sha / sizing_sha are the contract. A "
    "mismatch simply stops the run (exit 1) — nothing is substituted. The switch-off rule never touches "
    "model_p2.json and counts as 'in use' only when all three conditions hold at once.",
    "The untouched recent data (from 2024-09-01) is not scored without an unlock file. A candidate model's "
    "score there is information only, never grounds for adoption, and fresh blocks are counted only from the "
    "live start date.",
    "Product risk: three layers (probability, state machine, volatility rule) are harder to explain than the "
    "v0 traffic light → the card starts with one number and one sentence. Some cells of the scenario table "
    "are thin (the lower-holding state has n_eff 12; 33/11/4 episodes) — greying thin cells and pooling from "
    "the top is enforced by code.",
    "Compute: under +4 minutes a week and under +5 seconds a day; no new dependencies (EM written with numpy).",
]


LABELS.update({
    # 비중(§6.3 백테스트 표 · §6.2 사다리)
    "row": "행", "window": "기간", "max_dd_date": "최대 하락일", "worst_month_label": "최악 월(월)",
    "d_cagr_vs_bh": "연수익 차이(규칙−보유)", "d_maxdd_vs_bh": "최대 하락폭 차이(규칙−보유)",
    "vol_ratio": "실제로 움직인 폭 ÷ 목표(σ_T)",
    "maxdd_over_sigma_t": "최대 하락폭 ÷ σ_T", "sigma_target": "σ_T", "d_slow": "D_slow(3.5×)", "d_fast": "D_fast(2.0×)",
    "maxdd_vol_only": "최대 하락폭 — 변동성만(1993~)", "maxdd_vol_only_date": "일자(변동성만)",
    "maxdd_decision": "최대 하락폭 — 판정 규칙 포함(2003~)", "maxdd_decision_date": "일자(판정 규칙 포함)",
    "share_behind_years": "그냥 보유에 뒤진 해의 비율", "protection_pp": "덜 맞은 정도(규칙−보유)",
    "ret_rule": "규칙 수익", "ret_bh": "보유 수익", "avg_exposure": "평균 주식 비중", "w_exec": "실제 주식 비중",
    "w_vol": "변동성만 본 주식 비중", "w_target": "목표 주식 비중", "mult": "상태 배수", "cand": "격자 후보",
    "reason": "사유",
    "week_end": "주간 점검일", "sigma": "예상 변동성(EWMA)", "sigma_ewma": "예상 변동성(EWMA)",
    "sigma_har_fc": "예상 변동성(HAR, 표시 전용)", "sigma_down": "낮출 기준선", "sigma_up": "올릴 기준선",
    "next_check": "다음 점검일", "d_max": "감당할 최대 하락폭 D_max", "deploy_sizing": "주식 비중 블록 표시",
    # 유지 조건·상대성과
    "condition": "조건", "threshold": "한계", "margin": "여유", "note_ko": "판정", "failed": "위반",
    "undecided": "미판정", "narrow": "근소 통과", "share_behind": "뒤진 비율", "worst": "최악 창", "best": "최선 창",
    "p50": "중앙(50분위)",
    "L": "창 길이(세션)", "n_years": "연수", "n_behind": "뒤진 해", "by_year": "연도별",
    # 등록부·HMM
    "member": "멤버", "K_s": "라벨 보고 정한 수(K_s)", "K_u": "라벨 없이 정한 수(K_u)",
    "in_average": "오늘 확률에 들어감?", "status": "상태",
    "ledger_no": "장부 번호", "source": "출처", "admission": "채택 검정", "theta_id": "θ id",
    "p_high": "P(고변동)", "q20": "20세션 안 고변동 확률", "p_k": "k세션 뒤 고변동", "q_k": "k세션 안 고변동",
    "dwell": "한 상태가 이어지는 기간(세션)", "p00": "p00", "p11": "p11", "occupancy_high": "고변동 점유",
    "refit_date": "재적합일", "guard": "guard", "guard_ok": "guard 통과", "n_iter": "EM 반복",
    # 시나리오
    "bin_lo": "구간 하한", "bin_hi": "구간 상한", "pooled": "위 칸과 합침", "ret_p10": "20일 수익 p10",
    "ret_p50": "20일 수익 p50", "ret_p90": "20일 수익 p90", "mdd_p10": "20일 최대 하락폭 p10",
    "mdd_p50": "20일 최대 하락폭 p50", "share_ep10_start": "10% 넘게 떨어지기 시작한 비율", "grey": "회색(단독 표시 금지)",
    "obs_live": "라이브 실현 비율", "n_live": "라이브 n", "n_eff_live": "라이브 n_eff",
    # 트랙레코드·경보·킬
    "code": "코드", "action": "효과", "asof": "기준일", "stage": "지평 단계", "live_start": "라이브 시작",
    "months": "경과(개월)", "months_elapsed": "경과(개월)", "episodes5": "5% 넘게 떨어진 사건",
    "stage1_due": "1차 평가 도달", "stage2_due": "2차 평가 도달", "ci_label": "구간 라벨", "killed": "껐는지",
    "gross_failure_badge": "중대 실패 배지", "n_scored": "채점 행", "replay_mismatch_count": "D10 불일치 수",
    "hist_pct": "과거 같은 길이 창에서의 위치(백분위)", "backtest": "백테스트 참조", "fresh_blocks": "새 자료 블록",
    "switches_per_year": "전환/년", "w_changes_per_year": "주식 비중 변경/년", "n_changes_252": "252세션 변경 수",
    "share_gt_030": "p > 0.30 비율", "share_floor": "바닥(0.25) 비율", "share_full": "100% 비율",
    "avg_w": "평균 주식 비중", "har_log_mae": "HAR log-MAE", "ewma_log_bias": "ln(실현/EWMA) 편향",
    "hit": "적중률", "verdict_state": "판정", "hidden": "숨김", "hidden_reason": "숨김 사유",
})


# 열 제목의 영어 판 (STYLE_I18N.md §1: 표는 **열 제목만** 두 벌로 만든다).
# 없는 키는 열 이름을 그대로 쓴다 — 열 이름은 원래 영어 식별자라 번역이 필요 없다.
LABELS_EN: dict[str, str] = {
    # 공통
    "n": "sample (days)", "n_days": "sample (days)", "n_eff": "independent windows (n_eff)",
    "n_blocks": "independent windows", "key": "item", "start": "start", "end": "end", "years": "years",
    "cagr": "CAGR", "max_dd": "biggest drop from the peak", "maxdd": "biggest drop from the peak",
    "worst_month": "worst month", "ann_vol": "yearly swing", "vol": "yearly swing", "sharpe": "Sharpe",
    "total_return": "total return", "cost_bps": "cost (bp)", "cost_total": "total cost",
    "n_switches": "switches", "buy_hold": "buy & hold", "bh": "buy & hold", "allocation": "rule",
    "strategy": "rule", "diff": "difference", "share": "share", "count": "days", "note": "note",
    "value": "value", "pass": "passes?", "rule": "rule", "variant": "variant", "block": "block",
    "n_episodes": "decline episodes", "episodes": "decline episodes",
    # 비중 표·사다리
    "row": "row", "window": "period", "max_dd_date": "date of that biggest drop", "worst_month_label": "worst month (date)",
    "d_cagr_vs_bh": "yearly return vs holding", "d_maxdd_vs_bh": "biggest drop vs holding",
    "vol_ratio": "actual swing ÷ target (σ_T)", "maxdd_over_sigma_t": "biggest drop ÷ σ_T",
    "sigma_target": "σ_T (volatility target)", "d_slow": "D_slow (3.5×)", "d_fast": "D_fast (2.0×)",
    "maxdd_vol_only": "biggest drop — volatility only (1993-)", "maxdd_vol_only_date": "date (volatility only)",
    "maxdd_decision": "biggest drop — with decision rule (2003-)", "maxdd_decision_date": "date (with decision rule)",
    "share_behind_years": "share of years behind holding", "protection_pp": "how much less it fell (rule - holding)",
    "ret_rule": "rule return", "ret_bh": "holding return", "avg_exposure": "average share in stocks",
    "w_exec": "share actually held", "w_vol": "share from volatility only", "w_target": "target share",
    "mult": "state multiplier", "cand": "grid candidate", "reason": "why",
    "week_end": "weekly check date", "sigma": "expected swing (EWMA)", "sigma_ewma": "expected swing (EWMA)",
    "sigma_har_fc": "expected swing (HAR, display only)", "sigma_down": "cut-back trigger",
    "sigma_up": "top-up trigger", "next_check": "next check", "d_max": "budget D_max (biggest drop we accept)",
    "deploy_sizing": "show the sizing block",
    # 유지 조건·상대성과
    "condition": "condition", "threshold": "limit", "margin": "margin", "note_ko": "verdict",
    "failed": "failed", "undecided": "not decided", "narrow": "passed narrowly", "share_behind": "share behind",
    "worst": "worst window", "best": "best window", "p10": "10th percentile", "p50": "median (50th)",
    "p90": "90th percentile", "median": "median", "mean": "average",
    "L": "window length (sessions)", "n_years": "years", "n_behind": "years behind", "by_year": "by year",
    # 등록부·HMM
    "member": "model", "kind": "kind", "K_s": "numbers fitted with labels (K_s)",
    "K_u": "numbers fitted without labels (K_u)", "in_average": "in today's probability?", "status": "status",
    "ledger_no": "ledger entry", "source": "source", "admission": "admission test", "theta_id": "θ id",
    "p_high": "P(high-swing)", "q20": "chance of a high-swing day within 20 sessions",
    "p_k": "high-swing in k sessions", "q_k": "high-swing within k sessions",
    "dwell": "how long a state lasts (sessions)", "p00": "p00", "p11": "p11",
    "occupancy_high": "time spent high-swing", "refit_date": "refit date", "guard": "guard",
    "guard_ok": "guard passed", "n_iter": "EM iterations", "brier": "Brier", "bss_clim": "score vs the average",
    "auc": "AUC", "train_start": "training start", "train_end": "training end",
    # 시나리오
    "bin": "probability band", "bin_lo": "band from", "bin_hi": "band to", "pooled": "merged upward",
    "mean_p": "average stated %", "obs": "what actually happened", "wilson_lo": "Wilson low",
    "wilson_hi": "Wilson high", "ret_p10": "20-day return p10", "ret_p50": "20-day return p50",
    "ret_p90": "20-day return p90", "mdd_p10": "20-day biggest drop p10", "mdd_p50": "20-day biggest drop p50",
    "share_ep10_start": "share that began a ≥10% fall", "grey": "grey (never shown alone)",
    "obs_live": "live observed", "n_live": "live n", "n_eff_live": "live n_eff", "state": "state",
    # 트랙레코드·경보·킬
    "code": "code", "action": "effect", "asof": "as of", "stage": "horizon stage", "live_start": "live start",
    "months": "months elapsed", "months_elapsed": "months elapsed",
    "episodes5": "falls of 5% or more", "stage1_due": "first review reached", "stage2_due": "second review reached",
    "ci_label": "interval label", "killed": "switched off", "gross_failure_badge": "serious-failure badge",
    "n_scored": "scored rows", "replay_mismatch_count": "D10 mismatches", "evaluated_now": "evaluated now",
    "stage1_done": "first review done", "stage2_done": "second review done", "last_eval_month": "last review month",
    "hist_pct": "where it sits among past windows of the same length", "backtest": "backtest reference",
    "fresh_blocks": "fresh blocks", "switches_per_year": "switches per year",
    "w_changes_per_year": "share changes per year", "n_changes_252": "changes in 252 sessions",
    "share_gt_030": "share of days with p > 0.30", "share_floor": "share of days at the floor (0.25)",
    "share_full": "share of days fully invested", "avg_w": "average share in stocks",
    "har_log_mae": "HAR log-MAE", "ewma_log_bias": "ln(actual/EWMA) bias", "hit": "hit rate",
    "verdict_state": "verdict", "hidden": "hidden", "hidden_reason": "why it is hidden",
    "coverage": "band hit rate", "n_watch_avail": "watchlist names available",
}


# 공용 LABELS 의 한국어가 아직 전문용어인 열 — **이 모듈이 그리는 표에서만** 쉬운 말로 바꾼다.
# (공용 dict 를 고치면 v0·p2 페이지까지 흔들리므로 여기서만 덮어쓴다.)
_LABEL_KO_P3 = {
    "bss_clim": "정확도 점수(평소 평균 대비)", "bss": "정확도 점수(기준선 대비)",
    "depth": "고점 대비 하락폭", "n_eff": "겹치지 않는 창 수(n_eff)",
    "n_episodes": "하락 사건 수", "episodes": "하락 사건 수",
}


def _label_ko(key) -> str:
    """열 제목의 한국어(이스케이프 전 원문) — bi() 가 이스케이프한다."""
    k = str(key)
    if k in _LABEL_KO_P3:
        return _LABEL_KO_P3[k]
    return str(LABELS.get(k, LABELS.get(k.lower(), k)))


def _label_en(key) -> str:
    """열 제목의 영어. 없으면 열 이름 그대로(원래 영어 식별자다)."""
    k = str(key)
    return str(LABELS_EN.get(k, LABELS_EN.get(k.lower(), k)))


def _label_bi(key) -> str:
    """항목 이름 한 벌. 두 언어가 같으면(= 번역 없는 식별자) 스팬 없이 한 번만 쓴다."""
    ko, en = _label_ko(key), _label_en(key)
    return _esc(ko) if ko == en else bi(ko, en)


def _th_bi(ko: str, en: str, num: bool = False) -> str:
    """표 머리글 한 칸. 두 언어가 같으면 스팬 없이(식별자는 번역하지 않는다)."""
    if ko == en:
        return f'<th class="num">{_esc(ko)}</th>' if num else f"<th>{_esc(ko)}</th>"
    return bi_th(ko, en, num=num)


_P3_BT_ORDER = ["row", "window", "start", "end", "years", "cagr", "max_dd", "max_dd_date", "worst_month",
                "worst_month_label", "ann_vol", "switches_per_year", "avg_exposure", "cost_total",
                "d_cagr_vs_bh", "d_maxdd_vs_bh", "vol_ratio", "maxdd_over_sigma_t", "n_switches", "cost_bps"]


_P3_LADDER_ORDER = ["sigma_target", "d_slow", "d_fast", "maxdd_vol_only", "maxdd_vol_only_date",
                    "maxdd_decision", "maxdd_decision_date", "worst_month", "worst_month_label", "cagr",
                    "avg_exposure", "switches_per_year", "d_cagr_vs_bh", "share_behind_years"]


_P3_BIN_ORDER = ["bin", "bin_lo", "bin_hi", "pooled", "n", "n_eff", "mean_p", "obs", "wilson_lo", "wilson_hi",
                 "ret_p10", "ret_p50", "ret_p90", "mdd_p10", "mdd_p50", "share_ep10_start", "grey",
                 "n_live", "n_eff_live", "obs_live"]


# 상태는 범주라 풀링하지 않는다(§7 B) — 얇은 칸은 회색으로만 표시한다
_P3_STATE_ORDER = ["state", "n", "n_eff", "obs", "wilson_lo", "wilson_hi", "ret_p10", "ret_p50",
                   "ret_p90", "mdd_p10", "mdd_p50", "grey"]


_P3_REGISTRY_ORDER = ["member", "kind", "source", "K_s", "K_u", "status", "in_average", "ledger_no",
                      "n", "brier", "bss_clim", "auc"]


_P3_ALARM_ORDER = ["asof", "code", "value", "threshold", "action", "window", "note"]


# 백분율로 찍으면 안 되는 P3 키(비율·배수·비중은 0.65 처럼 그대로 읽는다)
_P3_PLAIN = ("vol_ratio", "maxdd_over_sigma_t", "mult", "w_exec", "w_vol", "w_target", "cand", "prev_w_exec",
             "k_slow", "k_fast", "p_k", "q_k", "hist_pct", "width_pp", "n_eff_live", "avg_w_backtest")


_P3_PCT = ("sigma", "sigma_ewma", "sigma_har_fc", "sigma_target", "sigma_down", "sigma_up", "d_slow", "d_fast",
           "d_max", "p_high", "q20", "obs_live", "share_behind_years", "share_gt_030", "share_floor",
           "share_full", "avg_w", "hit", "ret_p10", "ret_p50", "ret_p90", "mdd_p10", "mdd_p50",
           "share_ep10_start", "protection_pp", "margin", "p50", "best")


def _fmt_p3(v, key: str = "") -> str:
    """P3 표 값 포맷. 비율·배수·비중은 그대로(0.65), 변동성·낙폭·점유는 백분율. 나머지는 _fmt_p2 규약."""
    k = str(key or "").lower()
    if _is_nan(v):
        return "—"
    if _is_num(v) and not isinstance(v, (bool, np.bool_)):
        f = float(v)
        if math.isinf(f):
            return "∞" if f > 0 else "-∞"
        if k in _P3_PLAIN:
            return f"{f:,.2f}" if abs(f) < 100 else f"{f:,.1f}"
        if k in _P3_PCT:
            return f"{f * 100:.1f}%"
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(_is_num(x) for x in v):
        return f"[{_fmt_p3(v[0], key)}, {_fmt_p3(v[1], key)}]"
    return _fmt_p2(v, key)


_EMPTY_DEFAULT = ("자료 없음", "no data")


def _empty_note(empty_msg) -> str:
    """'…없음' 줄 — 조용히 비우지 않는다.

    empty_msg 는 (한국어, English) 또는 (한국어, English, 산출물 키). 키는 번역하지 않고 스팬 밖에 둔다.
    옛 호출처를 위해 한국어 한 줄(str)도 받는다.
    """
    key = ""
    if isinstance(empty_msg, (tuple, list)) and len(empty_msg) >= 2:
        ko, en = str(empty_msg[0]), str(empty_msg[1])
        key = str(empty_msg[2]) if len(empty_msg) > 2 else ""
    elif empty_msg is None or str(empty_msg) == _EMPTY_DEFAULT[0]:
        ko, en = _EMPTY_DEFAULT
    else:                                     # 한국어만 준 옛 호출 — 뜻은 언제나 '기록이 없다' 다
        ko, en = str(empty_msg), _EMPTY_DEFAULT[1]
    tail = f' <span class="mono">({_esc(key)})</span>' if key else ""
    return f'<div class="note">{bi(ko, en)}{tail}</div>'


def _p3_cols(recs: list[dict], order=None) -> list[str]:
    """표에 실을 열 순서 (mrl.report_common._records_table 과 같은 규칙)."""
    cols: list[str] = []
    for r in recs:
        for k in r.keys():
            if k not in cols:
                cols.append(k)
    if order:
        cols = [c for c in order if c in cols] + [c for c in cols if c not in order]
    return cols


def _p3_table(records, order=None, empty_msg=_EMPTY_DEFAULT, heads=None) -> str:
    """P3 표 — **열 제목만** 한/영 두 벌로 만들고 셀 값은 그대로 둔다 (STYLE_I18N.md §1).

    heads 로 (한국어, English) 쌍을 열 이름에 직접 줄 수 있다(사양 표처럼 LABELS 에 없는 열).
    """
    recs = _to_records(records)
    if not recs:
        return _empty_note(empty_msg)
    heads = dict(heads or {})
    cols = _p3_cols(recs, order)
    numeric = {c: all(_is_num(r.get(c)) or _is_nan(r.get(c)) for r in recs) for c in cols}

    def _pair(c):
        h = heads.get(c)
        if isinstance(h, (tuple, list)) and len(h) == 2:
            return str(h[0]), str(h[1])
        if isinstance(h, str):                   # 영어만 준 경우 — 한국어는 LABELS 에서
            return _label_ko(c), h
        return _label_ko(c), _label_en(c)

    head = "".join(_th_bi(*_pair(c), num=numeric[c]) for c in cols)
    body = []
    for r in recs:
        cells = "".join(f'<td class="{"num" if numeric[c] else ""}">{_fmt_p3(r.get(c), c)}</td>' for c in cols)
        body.append(f"<tr>{cells}</tr>")
    return f'<div class="tblwrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _p3_kv(d, empty_msg=_EMPTY_DEFAULT) -> str:
    """항목/값 표 — 항목 이름과 표 머리글만 두 벌 (mrl.report_common._kv_table 의 한/영 판)."""
    if not isinstance(d, dict) or not d:
        return _empty_note(empty_msg)
    sub = {k: v for k, v in d.items() if isinstance(v, dict)}
    flat = {k: v for k, v in d.items() if not isinstance(v, (dict, list, tuple, pd.DataFrame, pd.Series))}
    lists = {k: v for k, v in d.items() if isinstance(v, (list, tuple, pd.DataFrame, pd.Series))}
    parts = []
    if flat:
        rows = "".join(f"<tr><td>{_label_bi(k)}</td>"
                       f'<td class="num">{_fmt_p3(v, k)}</td></tr>' for k, v in flat.items())
        parts.append('<div class="tblwrap"><table><thead><tr>'
                     + _th_bi("항목", "item") + _th_bi("값", "value", num=True)
                     + f"</tr></thead><tbody>{rows}</tbody></table></div>")
    if sub:
        metrics: list[str] = []
        for v in sub.values():
            for k in v.keys():
                if k not in metrics:
                    metrics.append(k)
        head = "".join(_th_bi(_label_ko(k), _label_en(k), num=True) for k in sub.keys())
        rows = []
        for m in metrics:
            cells = "".join(f'<td class="num">{_fmt_p3(v.get(m), m)}</td>' for v in sub.values())
            rows.append(f"<tr><td>{_label_bi(m)}</td>{cells}</tr>")
        parts.append('<div class="tblwrap"><table><thead><tr>' + _th_bi("지표", "measure") + head
                     + f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")
    for k, v in lists.items():
        recs = _to_records(v)
        if recs:
            parts.append(_h3(_label_ko(k), _label_en(k)) + _p3_table(recs))
        else:
            parts.append(f'<div class="note">{_label_bi(k)}: {_fmt_p3(v, k)}</div>')
    return "".join(parts) if parts else _empty_note(empty_msg)


def _sub(d, *names) -> dict:
    """중첩 dict 를 방어적으로 꺼낸다(없으면 {})."""
    cur = d if isinstance(d, dict) else {}
    for n in names:
        nxt = cur.get(n) if isinstance(cur, dict) else None
        cur = nxt if isinstance(nxt, dict) else {}
    return cur


def _num_of(*sources, keys) -> float:
    """여러 dict 를 순서대로 뒤져 첫 유한 숫자를 돌려준다(없으면 NaN)."""
    for src in sources:
        if not isinstance(src, dict):
            continue
        v = _first(src, *keys)
        f = _fnum(v)
        if not math.isnan(f):
            return f
    return math.nan


def _dow_ko(x) -> str:
    try:
        ts = pd.Timestamp(x)
    except (TypeError, ValueError):
        return ""
    if pd.isna(ts):
        return ""
    return f"{_KO_DOW[ts.weekday()]}요일"


def _dow_en(x) -> str:
    try:
        ts = pd.Timestamp(x)
    except (TypeError, ValueError):
        return ""
    if pd.isna(ts):
        return ""
    return _EN_DOW[ts.weekday()]


# ------------------------------------------------------------------
# 유효 배치 모드 (§8.3: p2 ∧ p3 ∧ ¬kill — 리포트가 스스로 계산한다)
# ------------------------------------------------------------------
def _model_deploy_mode(path) -> str | None:
    """model_p2.json / model_p3.json 의 deploy_mode. 파일이 없거나 못 읽으면 None(지어내지 않는다)."""
    import json
    try:
        with open(Path(path), encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, ValueError):
        return None
    v = m.get("deploy_mode") if isinstance(m, dict) else None
    return None if v is None else str(v)


def p3_effective_mode(today_p3: dict, effective_mode: str | None = None) -> dict:
    """유효 배치 모드를 **자료에서 직접** 계산하고 호출자가 준 값과 AND 한다 (§8.3).

    유효 모드 = p2.deploy_mode == "tones" ∧ p3.deploy_mode == "tones" ∧ ¬kill_record ∧ ¬kill_manual.
    AND 자체는 `track.effective_mode` 하나만 쓴다(같은 규칙을 두 번 쓰지 않는다). dict 에 모드가 없으면
    results/model_p{2,3}.json 을 읽고, 그래도 없으면 **tones 로 올려 잡지 않는다**(사유를 남긴다).

    반환 {mode, tones, p2, p3, killed, kill_state, caller, sources, reasons}.
    """
    d = today_p3 if isinstance(today_p3, dict) else {}
    acc = _sub(d, "acceptance")
    p2m = _first(d, "p2_deploy_mode") or _first(_sub(d, "p2"), "deploy_mode") or _first(acc, "deploy_mode")
    p3m = _first(d, "p3_deploy_mode", "deploy_mode") or _first(_sub(d, "model_p3"), "deploy_mode")
    sources = {"p2": "입력" if p2m is not None else None, "p3": "입력" if p3m is not None else None}
    if p2m is None:
        p2m = _model_deploy_mode(MODEL_P2_PATH)
        sources["p2"] = "results/model_p2.json" if p2m is not None else "확인 불가"
    if p3m is None:
        p3m = _model_deploy_mode(MODEL_P3_PATH)
        sources["p3"] = "results/model_p3.json" if p3m is not None else "확인 불가"
    p2m = None if p2m is None else str(p2m)
    p3m = None if p3m is None else str(p3m)

    kill = _sub(d, "kill") or _sub(d, "track", "kill")
    kill_state = kill.get("state")
    kill_file = Path(KILL_RECORD_PATH).exists() or Path(KILL_MANUAL_PATH).exists()
    killed = (bool(kill.get("killed")) or str(kill_state or "") in ("info_only", "manual_kill")
              or bool(d.get("kill_record")) or bool(d.get("kill_manual")) or kill_file)
    caller = None if effective_mode is None else str(effective_mode)

    ok = (_track.effective_mode(p2m, p3m, kill_record_path=KILL_RECORD_PATH,
                                kill_manual_path=KILL_MANUAL_PATH) == "tones") and not killed
    reasons: list[str] = []
    reasons_en: list[str] = []                     # 같은 사유의 영어 판(표현 전용 — 판정에는 쓰지 않는다)
    if p2m != "tones":
        reasons.append(f"p2 deploy_mode = {p2m or '확인 불가'} ({sources['p2']})")
        reasons_en.append(f"p2 deploy_mode = {p2m or 'unknown'} ({_SOURCE_EN.get(sources['p2'], sources['p2'])})")
    if p3m != "tones":
        reasons.append(f"p3 deploy_mode = {p3m or '확인 불가'} ({sources['p3']})")
        reasons_en.append(f"p3 deploy_mode = {p3m or 'unknown'} ({_SOURCE_EN.get(sources['p3'], sources['p3'])})")
    if killed:
        reasons.append(f"킬룰 발동 — kill.state = {kill_state or ('kill 파일 존재' if kill_file else '기록')} (sticky)")
        reasons_en.append("the switch-off rule fired — kill.state = "
                          + str(kill_state or ("a kill file exists" if kill_file else "recorded")) + " (sticky)")
    if caller is not None and caller != "tones":
        if ok:
            reasons.append(f"호출자 effective_mode = {caller}")
            reasons_en.append(f"the caller passed effective_mode = {caller}")
        ok = False
    elif caller == "tones" and not ok:
        reasons.append("호출자는 tones 라 했지만 자료가 info_only 를 가리킨다 — 낮은 쪽을 따른다")
        reasons_en.append("the caller said tones but the data says info_only — we follow the lower of the two")
    return {"mode": "tones" if ok else "info_only", "tones": ok, "p2": p2m, "p3": p3m,
            "killed": killed, "kill_state": kill_state, "caller": caller, "sources": sources,
            "reasons": reasons, "reasons_en": reasons_en}


# ------------------------------------------------------------------
# 지평별 표시 규칙 (§8.4) — 단일 원천은 mrl.track
# ------------------------------------------------------------------
def horizon_view(src) -> dict:
    """§8.4 지평 dict. src 는 track.summary_p3()/live_panel() 산출, 그 안의 'horizon', 또는
    {'n_eff':…, 'months_elapsed':…} 무엇이든 받는다. 규칙은 track.horizon_copy 가 정한다(리포트는 다시 쓰지 않는다).
    자료가 없으면 가장 보수적인 단계('0-3m': skill 숫자·판정어 금지)로 떨어진다."""
    d = src if isinstance(src, dict) else {}
    for cand in (d.get("horizon"), _sub(d, "panel").get("horizon"), _sub(d, "track").get("horizon")):
        if isinstance(cand, dict) and cand.get("stage") in _track.HORIZON_STAGES:
            out = dict(cand)
            out.setdefault("footnote_ko", _track.HORIZON_FOOTNOTE)
            return out
    base = d if ("n_eff" in d or "n_scored" in d or "months_elapsed" in d) else _sub(d, "track")
    return _track.horizon_copy({"n_eff": base.get("n_eff"), "n_scored": base.get("n_scored"),
                                "months_elapsed": base.get("months_elapsed") or base.get("months"),
                                "kill": base.get("kill") if isinstance(base.get("kill"), dict) else {}})


def _hz_stage(hz) -> str:
    st = (hz or {}).get("stage")
    return st if st in _track.HORIZON_STAGES else _track.HORIZON_STAGES[0]


def horizon_may_show(hz, key: str) -> bool:
    """이 지평에서 key 를 보여도 되는가 — track.may_show 그대로."""
    return _track.may_show(_hz_stage(hz), key)


def _hz_reason(hz, key: str) -> str:
    stage = _hz_stage(hz)
    r = (hz or {}).get("hidden_reason") or _track.horizon_copy({"n_eff": 0, "months_elapsed": 0}).get("hidden_reason") \
        if stage == _track.HORIZON_STAGES[0] else (hz or {}).get("hidden_reason")
    if not r:
        r = "판정일(36개월) 전 — §8.4: 판정어 금지" if key in ("verdict", "ci_label") else None
    return f"§8.4 {stage} — {key} 숨김" + (f": {r}" if r else "")


# mrl.track 이 정한 사유(§8.4)의 영어 판. 규칙은 track 이 단일 원천이고 여기서는 같은 뜻을 영어로만 옮긴다.
_HZ_REASON_EN = {
    "0-3m": ("fewer than six independent windows (n_eff < 6) — §8.4: a three-month score is noise "
             "(10th/90th percentile ±0.8), so it is not shown even in grey"),
    "6m": "before the judgement date (36 months) — §8.4: no judging word is used",
    "12m": "before the judgement date (36 months) — §8.4: no judging word is used",
}


_HZ_REASON_FALLBACK_EN = "the code blocks this at the current stage"


def _reason_en(ko) -> str:
    """mrl.track 이 남긴 한국어 사유를 같은 뜻의 영어로. 모르면 일반 문장으로 떨어진다(지어내지 않는다)."""
    s = str(ko or "")
    if "n_eff < 6" in s:
        return _HZ_REASON_EN["0-3m"]
    if "판정일" in s:
        return _HZ_REASON_EN["6m"]
    return _HZ_REASON_FALLBACK_EN


def _hz_reason_en(hz, key: str) -> str:
    stage = _hz_stage(hz)
    r = _HZ_REASON_EN.get(stage)
    if not r and key in ("verdict", "ci_label"):
        r = "before the judgement date (36 months) — §8.4: no judging word is used"
    return f"§8.4 {stage} — {key} hidden" + (f": {r}" if r else "")


def _hz_val(hz, key: str, txt: str) -> str:
    """지평이 막는 값은 회색 '—' 와 **사유**로 바꾼다(조용히 지우지 않는다)."""
    if horizon_may_show(hz, key):
        return txt
    return ('<span class="mut">— <span class="note" style="display:inline">('
            + bi(_hz_reason(hz, key), _hz_reason_en(hz, key)) + ")</span></span>")


def _hz_sentence_en(ko: str) -> str | None:
    """§8.4 지평 문장(mrl.track 원문) → 영어 판. 원문이 바뀌었으면 None(영어를 지어내지 않는다)."""
    for st, txt in _track.HORIZON_SENTENCES.items():
        if str(ko) == str(txt):
            return HORIZON_SENTENCES_EN.get(st)
    return None


def _hz_footnote(hz) -> str:
    """§8.4 영구 각주 — mrl.track 원문(한국어)과 같은 뜻의 영어를 함께 심는다."""
    ko = str((hz or {}).get("footnote_ko") or _track.HORIZON_FOOTNOTE)
    en = (HORIZON_FOOTNOTE_EN if ko == _track.HORIZON_FOOTNOTE else
          "A live record grows only a dozen independent windows a year; until the judgement dates this page "
          "is a diary, not a verdict.")
    return _n(ko, en)


def _hz_key(col) -> str | None:
    """표의 열 이름 → §8.4 지평 키(막을 수 있는 것만). 모르는 열은 None(막지 않는다)."""
    c = str(col or "").lower()
    if c in ("verdict", "state", "ci_state", "verdict_state") or c.endswith("_verdict"):
        return "verdict"
    if c == "ci_label" or c.endswith("_ci_label"):
        return "ci_label"
    if c.startswith("bss") or "_bss" in c:
        return "bss"
    if c.startswith("brier"):
        return "brier"
    if c.startswith("ci"):
        return "ci"
    if c.startswith("hist_pct"):
        return "hist_pct"
    if c.startswith("reliability"):
        return "reliability_pooled" if "pool" in c else "reliability_2bin"
    return None


def _hz_redact(block, hz):
    """§8.4 가 막는 열을 None 으로 바꾸고 사유를 붙인다 — **리포트 층의 마지막 관문**.

    track.live_panel 이 이미 지운 것과 중복돼도 상관없다(둘 다 같은 track.may_show 를 쓴다). 여기서 한 번 더
    거르는 이유는 페이지가 패널 밖의 dict(예전 산출물·부분 패널)도 렌더링하기 때문이다."""
    if not isinstance(block, dict):
        return block
    out: dict = {}
    hidden: list[str] = [str(x) for x in (block.get("hidden") or [])]
    for k, v in block.items():
        key = _hz_key(k)
        if key and not horizon_may_show(hz, key):
            out[k] = None
            if key not in hidden:
                hidden.append(key)
        else:
            out[k] = v
    if hidden:
        out["hidden"] = sorted(set(hidden))
        if not out.get("hidden_reason"):
            out["hidden_reason"] = _hz_reason(hz, hidden[0]).split("숨김: ")[-1]
    return out


def _hz_block(hz) -> str:
    """지평 문장 + 영구 각주 + 이 단계에서 보이는/막히는 항목."""
    hz = hz or {}
    sents = hz.get("sentences_ko") or ([hz["sentence_ko"]] if hz.get("sentence_ko") else [])
    show = [str(x) for x in (hz.get("show") or [])]
    hide = [str(x) for x in (hz.get("hide") or [])]
    n_eff = _fnum(hz.get("n_eff"))
    stage, months = _hz_stage(hz), _fmt(hz.get("months"), "months")
    neff = "—" if math.isnan(n_eff) else f"{n_eff:.1f}"
    parts = [_n_html(f"지평 단계 <b>{_esc(stage)}</b> · 경과 {months}개월 · n_eff {neff}",
                     f"horizon stage <b>{_esc(stage)}</b> · {months} months elapsed · n_eff {neff}")]
    if sents:
        parts.append(_n("이 단계에서 할 수 있는 말은 여기까지입니다 — 원문 그대로:",
                        "This is as much as can be said at this stage — quoted as written:")
                     + '<div class="quote">'
                     + "<br>".join(_bi_or_raw(str(s), _hz_sentence_en(str(s))) for s in sents) + "</div>")
    if show:
        parts.append(f'<div class="note">{bi("지금 보이는 것", "shown now")}: {_esc(" · ".join(show))}</div>')
    if hide:
        parts.append(f'<div class="note">{bi("코드가 막아 둔 것", "blocked by the code")}: '
                     f'{_esc(" · ".join(hide))}'
                     + (" — " + bi(str(hz.get("hidden_reason")),
                                   _HZ_REASON_EN.get(stage) or _reason_en(hz.get("hidden_reason")))
                        if hz.get("hidden_reason") else "") + "</div>")
    parts.append(_hz_footnote(hz))
    return "".join(parts)


# ------------------------------------------------------------------
# 파라미터 회계 줄 · 배치 판정 줄 (모든 Phase 3 페이지 공통)
# ------------------------------------------------------------------
def _p3_param_nums(d: dict | None = None) -> dict:
    """파라미터 줄에 들어가는 숫자 — config·사양 상수에서만 온다(지어내지 않는다)."""
    src = d if isinstance(d, dict) else {}
    return {"k_prod": int(_first(src, "k_prod", "param_count", default=P2["param_count"]) or P2["param_count"]),
            "budget": int(_first(src, "budget", default=P2["budget"]) or P2["budget"]),
            "k_p3": int(_first(src, "k_p3", "p3_param_count", default=0) or 0),
            "h_ks": int(_first(src, "h_k_s", default=2) or 2),
            "h_ku": int(_first(src, "h_k_u", default=HMM_P3["param_count"]) or HMM_P3["param_count"])}


def p3_param_line(d: dict | None = None) -> str:
    """§10 6 정직 스트립의 파라미터 줄. 숫자는 config·사양 상수에서 오고, 문구는 §14 표 그대로."""
    return P3_PARAM_LINE_FMT.format(**_p3_param_nums(d))


def p3_param_line_en(d: dict | None = None) -> str:
    """p3_param_line 의 영어 판 — 같은 숫자, 같은 순서."""
    return P3_PARAM_LINE_FMT_EN.format(**_p3_param_nums(d))


def p3_param_bi(d: dict | None = None) -> str:
    """파라미터 줄 한/영 한 벌."""
    return bi(p3_param_line(d), p3_param_line_en(d))


_P2_ACC_CACHE: dict[str, dict] = {}


def load_acceptance_p2(path=None) -> dict:
    """results/summary_p2.json 의 acceptance 블록(없으면 {} + 경고). Phase 3 페이지가 배치 판정 줄을
    **원문 그대로** 싣기 위한 폴백 — 리포트는 판정을 다시 쓰지 않는다."""
    import json
    p = Path(path) if path is not None else (RESULTS_DIR / "summary_p2.json")
    key = str(p)
    if key in _P2_ACC_CACHE:
        return _P2_ACC_CACHE[key]
    try:
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        acc = s.get("acceptance") if isinstance(s, dict) and isinstance(s.get("acceptance"), dict) else {}
    except (OSError, ValueError) as e:
        warnings.warn(f"summary_p2.json 을 읽지 못함({type(e).__name__}) → 배치 판정 줄 없음")
        acc = {}
    _P2_ACC_CACHE[key] = acc
    return acc


def p2_acceptance_of(*sources, load: bool = True) -> dict:
    """summary_p3/track_p3/today_p3 에서 Phase 2 acceptance 를 찾고, 없으면 results/summary_p2.json 을 읽는다."""
    for s in sources:
        if not isinstance(s, dict):
            continue
        for cand in (s.get("acceptance"), _sub(s, "p2").get("acceptance"), _sub(s, "run").get("acceptance"),
                     _sub(s, "run").get("p2_acceptance"), _sub(s, "summary_p2").get("acceptance")):
            if isinstance(cand, dict) and cand:
                return cand
    return load_acceptance_p2() if load else {}


def p3_verdict_block(acc: dict, *, tag: str = "실제로 쓰는지에 대한 판정 (배치 판정(Phase 2 acceptance) — 원문 그대로)",
                     tag_en: str = "Is it actually in use? (Phase 2 acceptance verdict, quoted verbatim)") -> str:
    """acceptance.verdict_line 을 **그대로** 싣는 블록. 없으면 조용히 넘기지 않고 그 사실을 적는다.

    판정 줄 자체는 Phase 2 산출물의 원문이라 번역하지 않는다 — 스팬 밖에 두어 두 언어에서 같은 문장이 보인다.
    """
    line = acceptance_verdict_line(acc if isinstance(acc, dict) else {})
    if not line:
        return ('<div class="quote"><b>'
                + bi("배치 판정 줄 없음", "No deployment verdict line") + "</b><br>"
                + bi("실제로 쓸지 말지를 적은 한 줄을 찾지 못했습니다. 리포트가 그 판단을 대신 하지 않습니다"
                     "(배치 판정 줄 없음 — summary_p2.json:acceptance.verdict_line 을 찾지 못했습니다. "
                     "조용한 실패 금지).",
                     "The one line that says whether this is actually in use was not found "
                     "(summary_p2.json: acceptance.verdict_line). The report does not decide that on its own, "
                     "and it does not fail quietly.")
                + "</div>")
    dep = (acc or {}).get("deploy_mode")
    tone = (acc or {}).get("tone_model")
    tail_ko = (f"쓰임새 (deploy {dep or '—'})"
               + (f" · 신호등 판정에 쓰는 모델 {tone}" if tone else
                  " · 실제로 쓰는 모델 없음 — 신호등 판정도, 주식을 얼마나 담을지도 주장하지 않음"))
    tail_en = (f"deploy {dep or '—'}"
               + (f" · tone model {tone}" if tone else
                  " · no model is deployed — no traffic-light call and no suggested stock share"))
    return (f'<div class="quote"><b>{bi(tag, tag_en)}</b><br>{_raw(line)}'
            f'<div class="note">{bi(tail_ko, tail_en)}</div></div>')


def _p3_mode_banner(eff: dict, deploy_sizing=None) -> str:
    """유효 모드 배너 — info_only 면 무엇이 숨겨지는지 정확히 적는다.

    사유 줄(`deploy_mode = …`, 파일 경로)은 산출물의 이름이라 번역하지 않고 스팬 밖에 둔다 —
    두 언어에서 **같은 값**이 보여야 서로 다른 말을 하지 않는다.
    """
    eff = eff or {}
    tones = bool(eff.get("tones"))
    if tones:
        body = ("<b>" + bi("지금은 실제로 쓰는 모드입니다", "The platform is actually in use right now")
                + "</b> " + _esc(f'(mode = tones · p2 {eff.get("p2") or "—"} ∧ p3 {eff.get("p3") or "—"} ∧ '
                                 "kill 없음/none)"))
        extra = ""
    else:
        why_ko = " · ".join(eff.get("reasons") or ["사유 미기록"])
        why_en = " · ".join(eff.get("reasons_en") or []) or "no reason recorded"
        body = ("<b>" + bi(P3_INFO_ONLY_PLAIN, P3_INFO_ONLY_EN) + "</b> (VALIDATION §7)<br>"
                + bi(f"실제로 쓰지 않는 이유 ({why_ko})", f"Why it is not in use ({why_en})"))
        extra = ('<div class="note">'
                 + bi("실제로 쓰는 모드가 아니라서 상태·신호등 판정·주식 비중·단계별 예산 표를 감춥니다. 대신 "
                      "변동성만 보는 규칙·시장 분위기 게이지·앞으로 벌어질 수 있는 일의 표·구간만 회색 "
                      f"'{P3_INFO_NOT_SIZING}' 로 보여 줍니다 "
                      "(§15 stage 0 · §8.3: 상태·톤·비중·D_max 사다리를 숨기고, 변동성 단독 규칙·국면 게이지·"
                      "시나리오·구간만 회색으로 보여줍니다). 그동안에도 장부는 확률·상태·변동성만 본 주식 비중을 "
                      "사유를 '참고용만(info_only)' 로 계속 기록합니다 — 실제로 그렇게 했다는 뜻이 아니라 '이렇게 됐을 것' "
                      "이라는 가정치입니다.",
                      "Because the platform is not in use, the state, the traffic-light call, the stock share "
                      "and the budget ladder are hidden (§8.3, §15 stage 0). What stays is the volatility-only "
                      "rule, the market-mood gauge, the scenario tables and the ranges, all greyed out as "
                      "reference only. The ledger keeps recording the probability, the state and the "
                      "volatility-only share with reason info_only — these are what-if numbers, not something "
                      "the family did.")
                 + "</div>")
    if deploy_sizing is False:
        extra += ('<div class="note">'
                  + bi("유지 조건(§6.4)을 하나라도 어겼습니다 → 주식 비중 블록을 감춥니다. 기록은 계속 남깁니다.",
                       "A retention condition (§6.4) was broken → the sizing block is hidden. The ledger "
                       "keeps recording.")
                  + ' <span class="mono">(deploy_sizing=false)</span></div>')
    style = "" if tones else "border-color:#eab30855;background:#eab30814"
    return f'<div class="acc" style="{style}">{body}{extra}</div>'


# ------------------------------------------------------------------
# 일간 카드 (docs/index.html, P2 카드 아래) — §10 1~6
# ------------------------------------------------------------------
def _p3_kill_bits(d: dict, hz: dict) -> list[str]:
    """킬 카운트다운 조각. skill 숫자·판정어는 §8.4 지평 규칙을 통과해야 나온다."""
    ks = _sub(d, "kill") or _sub(d, "track", "kill")
    cd = _sub(ks, "countdown")
    ep = cd.get("episodes") or (f"{_fmt(ks.get('episodes5'), 'n')}/{KILL_P3['min_episodes']}"
                                if ks.get("episodes5") is not None else None)
    mo = cd.get("months") or (f"{_fmt(ks.get('months'), 'n')}/{KILL_P3['min_months']}"
                              if ks.get("months") is not None else None)
    cap = cd.get("cap") or f"상한 {KILL_P3['max_months']}"
    cap_en = cap.replace("상한", "cap")          # 산출물이 '2/60' 이면 그대로, '상한 60' 이면 'cap 60'
    # 분자/분모는 산출물 표기 그대로 두고(두 언어에서 같은 숫자) 그것을 **설명하는 말**만 두 벌로 심는다.
    bits = [bi("성적이 나쁘면 끄는 규칙까지 남은 조건", "How far the switch-off rule still has to go")
            + " " + bi(f"(킬 카운트다운: 실현 ≥5% 에피소드 {ep or '—'} · 경과 {mo or '—'}개월({cap}))",
                       f"(countdown: falls of 5% or more {ep or '—'} · {mo or '—'} months elapsed ({cap_en}))")]
    sc = _sub(ks, "score")
    bss = _fnum(sc.get("bss_clim") if sc else ks.get("bss_clim"))
    ci = (sc.get("ci_bss_clim") if sc else ks.get("ci")) or ks.get("ci_bss_clim")
    if math.isnan(bss):
        skill = bi("지금까지의 정확도 점수 — (채점된 행 없음)", "accuracy score so far — (no scored rows yet)")
    else:
        ci_txt = f" (95% {_fmt_p2(ci, 'bss')})" if isinstance(ci, (list, tuple)) and len(ci) == 2 else ""
        skill = bi_html(f"지금까지의 정확도 점수(기준선 대비) {_fmt_p2(bss, 'bss')}{ci_txt}",
                        f"current accuracy score vs a naive guess {_fmt_p2(bss, 'bss')}{ci_txt}")
    bits.append(_hz_val(hz, "bss", skill))
    hp = _sub(d, "track").get("hist_pct") or _sub(_sub(d, "track", "panel"), "vii_skill").get("hist_pct")
    if isinstance(hp, dict) and hp:
        ko_txt = " · ".join(f"{_esc(k)}세션 {_fmt_p3(v, 'hist_pct')}" for k, v in hp.items() if v is not None)
        en_txt = " · ".join(f"{_esc(k)} sessions {_fmt_p3(v, 'hist_pct')}" for k, v in hp.items() if v is not None)
        if ko_txt:
            bits.append(_hz_val(hz, "hist_pct",
                                bi_html(f"과거 같은 길이 창들 사이에서 우리 성적의 자리 — {ko_txt}",
                                        f"where our score sits among past windows of the same length — {en_txt}")))
    st = ks.get("state")
    if st:
        bits.append(_hz_val(hz, "verdict",
                            bi(f"성적이 나쁘면 끄는 규칙의 지금 단계 (킬룰 상태: {st})",
                               f"current stage of the switch-off rule: {st}")))
    if ks.get("gross_failure_badge"):
        bits.append(bi("중대 실패 배지가 붙었습니다 — 점수가 −0.05 이하이고 95% 구간 위쪽 끝도 0 이하라는 뜻입니다. "
                       "중간 점검 표시일 뿐, 아직 끈 것이 아닙니다 (BSS ≤ −0.05 ∧ CI 상한 ≤ 0).",
                       "A serious-failure badge is showing — the score is −0.05 or below and even the top of "
                       "the 95% interval is at or below 0. It is a mid-term flag, not a switch-off."))
    bits.append(kill_power_bi(d))                         # 카운트다운을 정직하게 만드는 줄(§8.3)
    return bits


def p3_card(today_p3: dict, effective_mode: str | None = None) -> str:
    """index.html 의 Phase 3 카드 조각(P2 카드 아래; v0·P2 카드는 그대로). 위에서 아래로 §10 1~6.

    **유효 배치 모드가 info_only 면**(p2 info_only · p3 info_only · 킬 중 하나라도) 상태·톤·비중·D_max 사다리를
    숨기고(§8.3 · §15 stage 0), 변동성 단독 규칙 값만 회색 '정보(비중 제안 아님)' 로 보여준다. 카드는 이 모드를
    자료에서 **스스로** 계산해 호출자 값과 AND 한다(올려 잡지 않는다).

    today_p3 (전부 선택; 없는 값은 '—'):
      asof, p2_deploy_mode, p3_deploy_mode, kill_record(bool), kill_manual(bool), deploy_sizing,
      sizing{w_exec, w_vol, w_target, mult, cand, sigma, sigma_target, d_max, state, reason, changed,
             prev_w_exec, thresholds(sizing.next_thresholds)},
      budget{d_max, sigma_target, sentence(sizing.budget_sentence), maxdd_decision, maxdd_vol_only, short},
      members{p2, M1, H}, disagreement{lo, hi, width, src, flag}, rung(p2 배포/생산 단 이름),
      scenarios(scenarios.today_context 산출), regime{p_high, q20, k_step{p_k,q_k,k}, dwell, theta_id, hidden},
      kill(track.kill_status) 또는 track(track.summary_p3 산출: kill·horizon·alarms·replay_mismatch_count·fresh_blocks),
      acceptance(summary_p2.acceptance), registry_sha, sizing_sha, model_id, theta_id, spec_sha256, warnings[].
    """
    d = today_p3 if isinstance(today_p3, dict) else {}
    eff = p3_effective_mode(d, effective_mode)
    tones = eff["tones"]
    trk = _sub(d, "track")
    hz = horizon_view(d if d.get("horizon") else (trk or d))
    sz = _sub(d, "sizing")
    thr = _sub(sz, "thresholds") or _sub(d, "thresholds")
    bd = _sub(d, "budget")
    asof = _first(d, "asof")
    warns = [str(w) for w in (d.get("warnings") or []) if w is not None and str(w).strip()]

    w_exec = _num_of(sz, d, keys=("w_exec", "p3_w_exec"))
    w_vol = _num_of(sz, d, keys=("w_vol", "p3_w_vol"))
    mult = _num_of(sz, d, keys=("mult", "p3_mult"))
    sigma = _num_of(sz, d, keys=("sigma", "sigma_ewma", "p3_sigma_ewma"))
    sT = _num_of(sz, bd, d, keys=("sigma_target", "p3_sigma_target"))
    d_max = _num_of(bd, sz, d, keys=("d_max", "p3_d_max"))
    state = _first(sz, "state") or _first(d, "state", "p2_state")
    state = None if state is None else str(state)
    reason = _first(sz, "reason", "p3_reason")
    reason = None if reason is None else str(reason)
    band = float(P3["band"])

    # 머리
    tags = [_tag("주식 비중 적용 중", "sizing in use") if tones
            else _tag(P3_INFO_NOT_SIZING_PLAIN, P3_INFO_NOT_SIZING_EN, cls=" warn")]
    if asof:
        tags.insert(0, '<span class="tag">' + bi("기준일", "as of")
                    + f' <b>{_esc(pd.Timestamp(asof).strftime("%Y-%m-%d"))}</b></span>')
    if not math.isnan(sT):
        tags.append('<span class="tag">' + bi("목표 변동성", "volatility target")
                    + f" <b>{_pct1(sT, 0)}</b></span>")
    if d.get("deploy_sizing") is False:
        tags.append(_tag("유지 조건 위반 — 주식 비중 블록 숨김", "a retention condition failed — sizing hidden",
                         cls=" bad"))
    al = [a for a in (trk.get("alarms") or d.get("alarms") or []) if isinstance(a, dict)]
    halt = [a for a in al if a.get("action") in ("halt_sizing_card", "require_ledger_entry")]
    if halt:
        tags.append('<span class="tag bad">' + bi("경보로 정지", "halted by an alarm") + " "
                    + _esc(", ".join(sorted({str(a.get("code")) for a in halt}))) + "</span>")
    head = ('<div class="eyebrow">'
            + bi("Phase 3 · 얼마나 담을지 정하는 규칙(조정 값 0개) · 후보 모형 명단 · 앞으로 벌어질 수 있는 일",
                 "Phase 3 · the rule for how much to hold (zero fitted numbers) · candidate models · scenarios")
            + "</div><h2>" + bi("오늘 주식을 얼마나 담을지, 그리고 그 이유 (오늘의 주식 비중과 그 근거)",
                                "How much to hold in stocks today, and why") + "</h2>"
            f'<div class="tags">{"".join(tags)}</div>'
            + _n(P3_FIRST_LINE, P3_FIRST_LINE_EN))

    # 1. 숫자 하나 + 문장 하나 (A·B) — info_only 면 회색 정보 블록
    w_vol_txt = "—" if math.isnan(w_vol) else f"{w_vol:.2f}"
    # 규칙 원문(식별자·수식)은 번역하지 않는다 — 괄호 안에 그대로 두면 두 언어가 같은 규칙을 가리킨다.
    vol_rule_src = (f"변동성 단독 규칙 w_vol = clip(목표 {_pct1(sT, 0)} ÷ 예상 {_pct1(sigma)}, "
                    f"{P3['w_min']:.2f}, {P3['w_max']:.2f}) = {w_vol_txt}")
    if tones:
        mult_txt = ("—" if math.isnan(mult) else f"{mult:.2f}") + (f" ({_esc(state)})" if state else "")
        w_txt = "—" if math.isnan(w_exec) else f"{w_exec:.2f}"
        s1 = ('<div class="verdict">'
              + bi_html(f"오늘 주식 비중 {w_txt}", f"Today's share in stocks: {w_txt}")
              + '</div><div class="v-act">'
              + bi_html(f"— 변동성 규칙 {w_vol_txt} (목표 {_pct1(sT, 0)} ÷ 예상 {_pct1(sigma)}) "
                        f"× 상태 배수 {mult_txt}",
                        f"— volatility rule {w_vol_txt} (target {_pct1(sT, 0)} ÷ expected {_pct1(sigma)}) "
                        f"× state multiplier {mult_txt}")
              + "</div>")
        line_ko, line_en, line_raw = [], [], []
        nc = thr.get("next_check")
        if nc:
            line_ko.append(f"다음 점검 {_dow_ko(nc)}({_esc(str(nc))})")
            line_en.append(f"next check {_dow_en(nc)} ({_esc(str(nc))})")
        sd, su = _fnum(thr.get("sigma_down")), _fnum(thr.get("sigma_up"))
        if not math.isnan(w_exec) and not math.isnan(sd):
            down = f"{max(w_exec - band, float(P3['w_min'])):.2f}"
            line_ko.append(f"예상 변동성이 {_pct1(sd)} 를 넘으면 {down}")
            line_en.append(f"if the expected swing goes above {_pct1(sd)}, {down}")
        elif thr.get("down_note"):
            line_raw.append(str(thr["down_note"]))
        if not math.isnan(w_exec) and not math.isnan(su):
            up = f"{min(w_exec + band, float(P3['w_max'])):.2f}"
            line_ko.append(f"{_pct1(su)} 아래면 {up}")
            line_en.append(f"if it drops below {_pct1(su)}, {up}")
        elif thr.get("up_note"):
            line_raw.append(str(thr["up_note"]))
        line_ko.append("판정 규칙이 위험 단계를 올리면 곧바로 낮춥니다 (결정층이 격상되면 즉시 하향)")
        line_en.append("if the decision rule raises the risk step, the share is cut the same day")
        s1 += ('<div class="note">' + bi_html(" · ".join(line_ko), " · ".join(line_en))
               + ("" if not line_raw else " · " + _esc(" · ".join(line_raw))) + "</div>")
        if reason:
            changed = d.get("changed") if d.get("changed") is not None else sz.get("changed")
            prev = _fmt_p3(sz.get("prev_w_exec"), "prev_w_exec")
            s1 += ('<div class="note">'
                   + bi_html(f"사유 {_esc(P3_REASON_KO.get(reason, reason))} ({_esc(reason)})"
                             + (f" — 직전 {prev} 에서 바뀜" if changed else ""),
                             f"why: {_esc(P3_REASON_EN.get(reason, reason))} ({_esc(reason)})"
                             + (f" — changed from {prev}" if changed else ""))
                   + "</div>")
    else:
        recorded = [bi_html(f"변동성만 보고 계산하면 {w_vol_txt} 가 나옵니다. 상태 배수는 곱하지 않았습니다 — "
                            f"상태 배수 미적용(장부 reason info_only, 가정치). ({_esc(vol_rule_src)})",
                            f"Volatility alone would give {w_vol_txt}. The state multiplier was not applied; "
                            f"the ledger stores it as a what-if with reason info_only. "
                            f"(rule: w_vol = clip(target {_pct1(sT, 0)} ÷ expected {_pct1(sigma)}, "
                            f"{P3['w_min']:.2f}, {P3['w_max']:.2f}) = {w_vol_txt})")]
        if not math.isnan(sigma):
            recorded.append(bi_html(f"앞으로 얼마나 출렁일지 예상한 값 {_pct1(sigma)} (EWMA λ=0.94)",
                                    f"expected swing from here {_pct1(sigma)} (EWMA λ=0.94)"))
        ks = _sub(d, "kill") or _sub(trk, "kill")
        sc = _sub(ks, "score")
        if sc.get("n"):
            nums = bi_html(f"채점한 날 {_fmt(sc.get('n'), 'n')}일 (겹치지 않는 창으로는 "
                           f"{_fmt_p3(sc.get('n_eff'), 'n_eff')}개)",
                           f"{_fmt(sc.get('n'), 'n')} scored days ("
                           f"{_fmt_p3(sc.get('n_eff'), 'n_eff')} independent windows)") + " · "
            nums += _hz_val(hz, "bss", bi_html(f"BSS {_fmt_p2(sc.get('bss_clim'), 'bss')} "
                                               f"(95% {_fmt_p2(sc.get('ci_bss_clim'), 'bss')})",
                                               f"accuracy score vs a naive guess "
                                               f"{_fmt_p2(sc.get('bss_clim'), 'bss')} "
                                               f"(95% {_fmt_p2(sc.get('ci_bss_clim'), 'bss')})"))
            recorded.append(nums)
        if ks.get("episodes5") is not None:
            recorded.append(bi_html(f"5% 넘게 떨어진 사건 {_fmt(ks.get('episodes5'), 'n')}회 · 시작한 지 "
                                    f"{_fmt(ks.get('months'), 'n')}개월",
                                    f"{_fmt(ks.get('episodes5'), 'n')} falls of 5% or more · "
                                    f"{_fmt(ks.get('months'), 'n')} months since the start"))
        why_ko = " · ".join(eff["reasons"] or ["사유 미기록"])
        why_en = " · ".join(eff.get("reasons_en") or []) or "no reason recorded"
        s1 = ('<div class="info"><div class="verdict" style="color:' + PALETTE["mut"] + '">'
              + bi(P3_INFO_ONLY_PLAIN, P3_INFO_ONLY_EN) + "</div>"
              + '<div class="tags">' + _tag(P3_INFO_NOT_SIZING_PLAIN, P3_INFO_NOT_SIZING_EN, cls=" warn")
              + _tag(INFO_DISPLAY_LABEL, INFO_DISPLAY_EN, cls=" warn") + "</div>"
              + "".join(f'<div class="note">{x}</div>' for x in recorded)
              + '<div class="note">'
              + bi(f"실제로 쓰지 않는 이유 ({why_ko})", f"Why it is not in use ({why_en})") + "</div>"
              + _n("주식을 얼마나 담으라거나, 지금이 어떤 상태라거나, 사라/팔라는 제안을 하지 않습니다. 위 숫자는 "
                   "규칙이 무엇을 하고 있는지 보여 주는 참고값이고, 실제 배분은 v0 판정을 따릅니다.",
                   "This card suggests no amount to hold, no state and no traffic-light call. The numbers "
                   "above only show what the rule is doing; the actual allocation follows the v0 call.")
              + "</div>")

    # 2. 예산 줄 — info_only 에서는 사다리를 숨긴다(§8.3)
    if tones:
        sent = bd.get("sentence") or _first(d, "budget_sentence")
        if not sent and not (math.isnan(d_max) or math.isnan(sT)):
            dd_dec, dd_vol = _fnum(bd.get("maxdd_decision")), _fnum(bd.get("maxdd_vol_only"))
            sent = _sizing.budget_sentence(d_max, sT,
                                           None if math.isnan(dd_dec) else dd_dec,
                                           None if math.isnan(dd_vol) else dd_vol)
        short = bd.get("short")
        s2 = (_h3("감당할 수 있는 최대 하락폭 — 한 줄 요약", "The most we are willing to fall — in one line")
              + (_n("아래 문장은 계산 결과 원문입니다:", "The sentence below is quoted from the run:")
                 + f'<div class="quote">{_raw(str(sent))}</div>' if sent else
                 _n("예산 문장을 만들 수 없습니다 — 감당할 최대 하락폭(D_max)이나 목표 변동성(σ_T)이 없습니다.",
                    "No budget sentence — D_max or the volatility target σ_T is missing."))
              + '<div class="note">'
              + (_esc(str(short)) if short else bi(P3_SIZING_SHORT, P3_SIZING_SHORT_EN))
              + ' · <a href="sizing_p3.html">'
              + bi("단계별 예산 표와 성적표 보기", "see the budget ladder and the backtest table") + "</a></div>")
    else:
        s2 = (_h3("감당할 수 있는 최대 하락폭 — 한 줄 요약", "The most we are willing to fall — in one line")
              + _n(P3_LADDER_HIDDEN, P3_LADDER_HIDDEN_EN))

    # 3. 확률 헤드라인 — P2 카드의 숫자를 반복하지 않고 멤버 간 폭만 말한다
    dis = _sub(d, "disagreement")
    mem = _sub(d, "members") or _sub(dis, "members")
    lo, hi = _fnum(dis.get("lo")), _fnum(dis.get("hi"))
    dep3 = deployment_of(_sub(d, "p2") or d)                  # 배포 여부는 p2 acceptance 가 진실
    rung = str(_first(d, "rung", "prob_rung") or dep3["prob_rung"])
    parts, parts_en = [], []
    for name in ("p2", "M1", "H"):
        v = _fnum(mem.get(name))
        if math.isnan(v):
            continue
        pct = int(round(v * 100))
        lab = f"p2 {rung}" if name == "p2" else P3_MEMBER_KO.get(name, name)
        lab_en = f"p2 {rung}" if name == "p2" else P3_MEMBER_EN.get(name, name)
        parts.append(f"{lab} {pct}")
        parts_en.append(f"{lab_en} {pct}")
    head3 = _h3("모형마다 확률이 얼마나 다른가", "How far the models disagree")
    if math.isnan(lo) or math.isnan(hi):
        s3 = head3 + _n("모형별 확률이 기록되지 않아 폭을 말할 수 없습니다 (불일치 구간 없음) — 모르는 것을 "
                        "좁혀서 말하지 않습니다.",
                        "The per-model probabilities were not recorded, so no spread is shown — we do not "
                        "narrow what we do not know.")
    else:
        wpp = (hi - lo) * 100.0
        lo_d, hi_d = int(round(lo * 100)), int(round(hi * 100))
        s3 = (head3 + '<div class="v-act">'
              + bi_html(f"모델들은 100일 중 {lo_d}~{hi_d}일로 갈린다"
                        + (f' ({_esc(" · ".join(parts))}; 불일치 {wpp:.1f}pp)' if parts
                           else f" (불일치 {wpp:.1f}pp)"),
                        f"The models say between {lo_d} and {hi_d} days out of 100"
                        + (f' ({_esc(" · ".join(parts_en))}; spread {wpp:.1f}pp)' if parts_en
                           else f" (spread {wpp:.1f}pp)"))
              + "</div>")
        if dis.get("flag"):
            s3 += ('<div class="tag bad">'
                   + bi_html(f"모형들이 오래 크게 갈리고 있습니다 (불일치 플래그 — 폭 &gt; "
                             f'{ENSEMBLE_P3["disagree_flag"]["width"] * 100:.0f}pp 가 '
                             f'{ENSEMBLE_P3["disagree_flag"]["sessions"]}세션 이상 이어짐)',
                             "the models have disagreed widely for a while (gap &gt; "
                             f'{ENSEMBLE_P3["disagree_flag"]["width"] * 100:.0f}pp for '
                             f'{ENSEMBLE_P3["disagree_flag"]["sessions"]} sessions or more)')
                   + "</div>")
        # 배포된 단이 없으면 '배포 단' 이라 쓰지 않는다 — 같은 페이지가 '실제로 쓰는 모델 없음' 이라 말하고 있다
        src_txt = _esc(str(dis.get("src") or "—"))
        s3 += ('<div class="note">'
               + bi_html("이 폭은 보여 주기만 합니다. 아직 쓰지 않는 후보 모형은 오늘 확률에 들어가지 않습니다"
                         + (f"(오늘 배포 단 {_esc(rung)})" if dep3["deployed"]
                            else f"(오늘 생산 확률 단 {_esc(rung)} — {_esc(INFO_DISPLAY_LABEL)})")
                         + f", 폭의 출처 {src_txt}.",
                         "The spread is shown for information only. Candidate models are not in today's "
                         "probability"
                         + (f" (today's model in use: {_esc(rung)})" if dep3["deployed"]
                            else f" (today's probability model: {_esc(rung)} — shown for information, "
                                 "not in use)")
                         + f", spread source {src_txt}.")
               + "</div>")

    # 4. 시나리오 3줄 + ATH 대비 현재 낙폭
    scen = _sub(d, "scenarios") or _sub(d, "scenario")
    lines = [str(x) for x in (scen.get("lines") or []) if x]
    dd = _sub(scen, "dd") or _sub(d, "drawdown")
    s4 = _h3("앞으로 벌어질 수 있는 일 (시나리오 — 과거를 세기만 합니다, 맞춘 숫자 0개)",
             "What can happen next (scenarios — we only count the past, zero fitted numbers)")
    if lines:
        # 계산층이 같은 숫자로 만들어 준 쉬운 한국어/영어 쌍이 있으면 그것을 심는다(§1 두 벌).
        # 없으면(옛 산출물·합성 입력) 영어를 지어내지 않고 원문을 인용 표시하고 그 사실을 밝힌다.
        pairs = scen.get("lines_bi")
        ok = (isinstance(pairs, (list, tuple)) and len(pairs) == len(lines)
              and all(isinstance(p_, (list, tuple)) and len(p_) == 2 for p_ in pairs))
        if ok:
            s4 += ('<ul class="plain">'
                   + "".join(f"<li>{bi(str(k_), str(e_))}</li>" for k_, e_ in pairs) + "</ul>")
        else:
            s4 += _n("아래 줄은 계산 과정이 남긴 기록이라 한국어 원문 그대로 둡니다.",
                     "The lines below are raw sentences written by the pipeline and are kept in the "
                     "original Korean.")
            s4 += '<ul class="plain">' + "".join(f"<li>{_raw(x)}</li>" for x in lines) + "</ul>"
    else:
        s4 += _n("표가 없어 문장을 만들지 않습니다 — 표본 수(n)·겹치지 않는 창 수(n_eff)·구간이 없으면 "
                 "아무 말도 하지 않습니다 (시나리오 표 없음).",
                 "No table, so no sentence — without the sample size (n), the number of independent windows "
                 "(n_eff) and an interval we say nothing.")
    if dd:
        ddv = _fnum(dd.get("dd_from_ath"))
        s4 += ('<div class="note">'
               + bi_html(f"ATH 대비 현재 {_pct1(ddv)}"
                         + (f' (고점 {_esc(str(dd.get("ath_date")))})' if dd.get("ath_date") else "")
                         + (f" · −5% 를 지난 뒤 {_fmt(dd.get('sessions_since_breach'), 'n')}세션"
                            if dd.get("breached_5") and dd.get("sessions_since_breach") is not None else ""),
                         f"currently {_pct1(ddv)} below the all-time high"
                         + (f' (peak {_esc(str(dd.get("ath_date")))})' if dd.get("ath_date") else "")
                         + (f" · {_fmt(dd.get('sessions_since_breach'), 'n')} sessions since it passed −5%"
                            if dd.get("breached_5") and dd.get("sessions_since_breach") is not None else ""))
               + "</div>")
    sp_ko = str(scen.get("spread_sentence") or _scenarios_spread_sentence())
    # 고정 상수라 숫자가 매번 달라지지 않는다 → 쉬운 한국어/영어 판을 바로 심는다(원문과 뜻·숫자가 같다).
    s4 += (_n(SPREAD_SENTENCE_PLAIN, SPREAD_SENTENCE_EN) if sp_ko == _scenarios_spread_sentence()
           else _n_raw(sp_ko))
    for n in (scen.get("notes") or []):
        s4 += _n_raw(str(n))

    # 5. 국면 게이지 (그림자)
    rg = _sub(d, "regime")
    hide_gauge = bool(rg.get("hidden")) or any(str(a.get("code")) == "D9_hmm" for a in al)
    p_high = _fnum(rg.get("p_high"))
    ks20 = _sub(rg, "k_step")
    q20 = _fnum(rg.get("q20") if rg.get("q20") is not None else ks20.get("q_k"))
    head5 = _h3("시장 분위기 게이지 (국면 게이지 · 아직 쓰지 않는 후보)",
                "Market-mood gauge (candidate model, not in use)")
    if hide_gauge:
        s5 = head5 + _n("게이지 숨김 — 최근 756세션에서 이 신호의 판별력이 동전 던지기보다 못했거나(D9) 안전장치가 "
                        "걸렸습니다. 이 후보 모형은 기각으로 기록됩니다.",
                        "The gauge is hidden — over the last 756 sessions this signal did no better than a "
                        "coin toss (D9), or a guard failed. The candidate model is recorded as turned down.")
    elif math.isnan(p_high):
        s5 = head5 + _n("P(고변동) 값이 없습니다 — HMM 숫자가 기록되지 않았습니다.",
                        "No P(high-swing) value — the HMM number was not recorded.")
    else:
        i_lvl = 0 if p_high < P3_REGIME_CUTS[0] else (1 if p_high < P3_REGIME_CUTS[1] else 2)
        lvl, lvl_en = P3_REGIME_LEVELS[i_lvl], P3_REGIME_LEVELS_EN[i_lvl]
        k_step = int(ks20.get("k") or P2["h"])
        dwell = _fnum(rg.get("dwell") if rg.get("dwell") is not None else rg.get("expected_dwell"))
        s5 = (head5 + '<div class="v-act">'
              + bi_html(f"{_esc(lvl)} — P({_esc(P3_HIGH_STATE_KO)}) {_pct1(p_high)}"
                        + (f" · {k_step}세션 안에 {_esc(P3_HIGH_STATE_KO)}가 될 확률 {_pct1(q20)}"
                           if not math.isnan(q20) else "")
                        + (f" · 한 상태가 보통 {dwell:.0f}세션쯤 이어짐" if not math.isnan(dwell) else ""),
                        f"{_esc(lvl_en)} — P({_esc(P3_HIGH_STATE_EN)}) {_pct1(p_high)}"
                        + (f" · chance of a {_esc(P3_HIGH_STATE_EN)} within {k_step} sessions {_pct1(q20)}"
                           if not math.isnan(q20) else "")
                        + (f" · a state usually lasts about {dwell:.0f} sessions" if not math.isnan(dwell)
                           else ""))
              + "</div>"
              + '<div class="note">' + bi(P3_SHADOW_LABEL, P3_SHADOW_LABEL_EN) + " · "
              + bi(f"{P3_HIGH_STATE_KO}로 지낸 시간이 길다는 것은 '위기' 라는 뜻이 아니라 "
                   f"'{P3_HIGH_STATE_GLOSS_KO}({P3_HIGH_STATE_KO})' 라는 뜻입니다 (§16 9)",
                   "Spending a lot of time in the high-swing state does not mean 'crisis' — it means the "
                   "market has been swinging a lot (§16 9)")
              + "</div>")

    # 6. 정직 스트립 — items 는 이미 HTML. 해시·식별자·산출물 원문은 번역하지 않고 스팬 밖에 둔다.
    items = [p3_param_bi(d)]
    shas = [f"registry_sha {str(_first(d, 'registry_sha') or '—')[:12]}",
            f"sizing_sha {str(_first(d, 'sizing_sha') or '—')[:12]}",
            f"model_id {_first(d, 'model_id') or '—'}",
            f"theta_id {_first(d, 'theta_id') or _first(rg, 'theta_id') or '—'}"]
    items.append(bi("이 페이지를 만든 코드·모델의 지문", "fingerprints of the code and model behind this page")
                 + " " + _esc(" · ".join(shas)))
    # 값(mode·p2·p3)은 산출물 이름이라 번역하지 않고, 그 값을 **설명하는 말**만 두 벌로 심는다.
    p2_src, p3_src = eff.get("p2"), eff.get("p3")
    mode_ko = (f"유효 모드(p2 ∧ p3 ∧ ¬kill) = {eff['mode']} — p2 {p2_src or '확인 불가'} · "
               f"p3 {p3_src or '확인 불가'} · 끄는 규칙 {'걸림' if eff['killed'] else '안 걸림'}"
               + (f" ({'; '.join(eff['reasons'])})" if eff["reasons"] else ""))
    mode_en = (f"mode in force (p2 AND p3 AND no switch-off) = {eff['mode']} — p2 "
               f"{p2_src or 'could not be determined'} · p3 {p3_src or 'could not be determined'} · "
               f"switch-off rule {'fired' if eff['killed'] else 'not fired'}"
               + (f" ({'; '.join(eff.get('reasons_en') or eff['reasons'])})" if eff["reasons"] else ""))
    items.append(bi("지금 실제로 쓰는 모드 — p2 와 p3 가 모두 켜져 있고 끄는 규칙이 걸리지 않아야 켜집니다",
                    "The mode actually in force — it is on only when p2 and p3 are both on and the "
                    "switch-off rule has not fired")
                 + " " + bi(f"({mode_ko})", f"({mode_en})"))
    items.extend(_p3_kill_bits(d, hz))
    hz_sent = str(hz.get("sentence_ko") or "")
    items.append(bi(f"지평({_hz_stage(hz)})", f"horizon stage ({_hz_stage(hz)})")
                 + ": " + (_bi_or_raw(hz_sent, _hz_sentence_en(hz_sent)) if hz_sent else ""))
    if al:
        items.append(bi("드리프트 경보", "drift alarms") + " "
                     + _esc(" · ".join(f"{a.get('code')}({a.get('action') or '표시/display'})" for a in al)))
    else:
        items.append(bi("예전과 달라졌다는 경고 없음 (경보 없음, D1~D11)",
                        "no drift warnings (D1-D11 all quiet)"))
    rmc = trk.get("replay_mismatch_count", d.get("replay_mismatch_count"))
    items.append(bi_html(f"D10 재현 불일치 {'—' if rmc is None else _fmt(rmc, 'n')}건",
                         f"D10 replay mismatches: {'—' if rmc is None else _fmt(rmc, 'n')}"))
    fb = trk.get("fresh_blocks") or d.get("fresh_blocks")
    if isinstance(fb, dict):
        need = fb.get("required") or ENSEMBLE_P3["fresh_blocks_min"]
        items.append(bi_html(f"손대지 않은 새 자료가 얼마나 쌓였는지 "
                             f"(신선 블록 {_fmt(fb.get('n_complete'), 'n')}/{_esc(need)})",
                             f"how much untouched new data has piled up "
                             f"(fresh blocks {_fmt(fb.get('n_complete'), 'n')}/{_esc(need)})"))
    else:
        items.append(bi(f"손대지 않은 새 자료가 얼마나 쌓였는지 (신선 블록 —/{ENSEMBLE_P3['fresh_blocks_min']} — "
                        "라이브 시작 전에는 세지 않는다)",
                        f"how much untouched new data has piled up (fresh blocks —/"
                        f"{ENSEMBLE_P3['fresh_blocks_min']}; not counted before the live start)"))
    acc_line = acceptance_verdict_line(p2_acceptance_of(d))
    if acc_line:
        items.append(bi("실제로 쓸지에 대한 Phase 2 판정 (계산이 남긴 한국어 원문 그대로)",
                        "The Phase 2 verdict on whether to use it (raw Korean from the pipeline)")
                     + " — " + _raw(acc_line))
    strip = ('<div class="strip"><b>' + bi("정직 스트립 — 숨기지 않는 것들",
                                           "Honesty strip — what we do not hide") + "</b>"
             '<ul class="plain">' + "".join(f"<li>{x}</li>" for x in items)
             + "</ul>" + _n(P3_CARD_FOOTNOTE, P3_CARD_FOOTNOTE_EN) + _hz_footnote(hz) + "</div>")
    wsec = (_h3("P3 경고", "P3 warnings") + _warns_bi(warns)) if warns else ""
    return f'<section class="panel" id="p3">{head}{s1}{s2}{s3}{s4}{s5}{wsec}{strip}</section>'


def _honest_reading_bi(sz: dict, hr) -> list[str]:
    """§6.5 네 줄: 한국어 원문(글자 그대로) + 같은 자리표로 만든 영어 판.

    영어는 **재현할 수 있을 때만** 붙인다 — `mrl.sizing.honest_reading()` 을 같은 숫자로 다시 만들어
    지금 그리는 줄과 글자까지 같을 때에만 짝으로 인정한다. 다르면 영어를 지어내지 않고 원문만 인용한다
    (HARD RULE 1: 한계를 잃느니 읽지 못하는 한국어가 낫다).
    """
    ko_lines = [str(x) for x in (hr or [])]
    res = sz.get("honest_results") if isinstance(sz, dict) else None
    res = res if isinstance(res, dict) else None
    try:
        regen = _sizing.honest_reading(res)
    except (ValueError, TypeError):                                # pragma: no cover - 방어
        regen = None
    if regen is None or list(regen) != ko_lines or len(ko_lines) != len(HONEST_READING_EN):
        return [_raw(x) for x in ko_lines]
    vals = dict(_sizing._HONEST_DEFAULTS)
    vals.update(res or {})
    tail_en = "" if res else _HONEST_TAIL_EN
    out = []
    for ko, tpl in zip(ko_lines, HONEST_READING_EN):
        try:
            out.append(bi(ko, tpl.format(**vals) + tail_en))
        except (KeyError, IndexError):                              # pragma: no cover - 방어
            out.append(_raw(ko))
    return out


def _scenarios_spread_sentence() -> str:
    """§7 카드에 반드시 인쇄하는 문장(단일 원천: mrl.scenarios). 임포트 실패 시에도 문장은 남는다."""
    try:
        from mrl.scenarios import SPREAD_SENTENCE
        return str(SPREAD_SENTENCE)
    except ImportError:                                            # pragma: no cover - 방어
        return ("모든 상태·구간에서 20일 수익의 중앙값은 양수다 — 모델은 방향이 아니라 결과의 폭을 예측한다"
                "(VALIDATION §0). 높은 위험 구간일수록 수익 분포는 좁아지지 않고 넓어진다"
                "(급락과 반등이 같이 산다).")


# ------------------------------------------------------------------
# Phase 3 페이지 공통 조각
# ------------------------------------------------------------------
def _p3_head(eyebrow, title, sub, tags: list[str], acc: dict, eff: dict,
             deploy_sizing=None) -> str:
    """모든 Phase 3 페이지의 머리: §16 1 첫 줄 → 실제 사용 여부 판정(원문) → 유효 모드 배너.

    eyebrow·title·sub 은 (한국어, English) 쌍이다.
    """
    def _pair(x):
        return (x[0], x[1]) if isinstance(x, (tuple, list)) else (x, x)

    e_ko, e_en = _pair(eyebrow)
    t_ko, t_en = _pair(title)
    s_ko, s_en = _pair(sub)
    return ("<header>"
            f'<div class="eyebrow">{bi(e_ko, e_en)}</div><h1>{bi(t_ko, t_en)}</h1>'
            f'<div class="sub">{bi(s_ko, s_en)}</div>'
            + (f'<div class="tags">{"".join(tags)}</div>' if tags else "")
            + _q(P3_FIRST_LINE, P3_FIRST_LINE_EN)
            + p3_verdict_block(acc)
            + _p3_mode_banner(eff, deploy_sizing)
            + "</header>")


def _p3_nav(titles) -> str:
    """차례. titles 는 (한국어, English) 쌍의 목록."""
    out = []
    for i, t in enumerate(titles):
        ko, en = (t[0], t[1]) if isinstance(t, (tuple, list)) else (t, t)
        out.append(f'<a href="#s{i + 1}">{bi(ko, en)}</a>')
    return '<nav class="nav">' + "".join(out) + "</nav>"


def kill_power_line(src, member: str = "p2") -> str:
    """§8.3·§16.7 — 검정력 문장은 kill_replay 산출로만 만든다.

    재생성되지 않았으면 숫자를 지어내지 않고 그 사실을 쓴다("재생성되지 않으면 페이지에 쓰지 않는다").
    산문에 숫자를 박아 두면 코드가 다른 값을 내도 페이지가 영원히 옛 숫자를 말한다."""
    kp = _sub(src, "kill_power").get(member) if isinstance(src, dict) else None
    if not isinstance(kp, dict) or not kp.get("sentence_ko"):
        return ("성적이 나쁘면 끄는 규칙이 얼마나 잘 가려내는지: 이번 주간 자기검사(kill_replay)가 이 수치를 "
                "재생성하지 않아 표시하지 않는다(§16.7 — 재생성되지 않으면 페이지에 쓰지 않는다).")
    return (f"성적이 나쁘면 끄는 규칙이 얼마나 잘 가려내는지({kp.get('p_col') or member}, 주간 자기검사 재생성 · "
            f"창 {_fmt(kp.get('n_windows'), 'n')}): {kp['sentence_ko']} 2단계로 나눠 읽는 방식은 문자 그대로 "
            "읽는 것보다 엄격하다(#7).")


def _kill_power_nums(kp: dict):
    """표시용 숫자만 뽑는다 — 문장은 화면에서 만들고 산출물 JSON 은 손대지 않는다.

    §16.7 은 "재생성되지 않으면 쓰지 않는다" 이므로, 다섯 값이 모두 이번 재생성분에 있을 때만 만든다.
    0.0 이 유효값이라 `is not None` 으로 검사한다(거짓값 함정 회피).
    """
    need = ("window_months", "false_kill_share", "validated_share",
            "power_vs_block_shuffle", "null_noise_pass")
    if not all(kp.get(k) is not None for k in need):
        return None
    try:
        return (int(kp["window_months"]),
                round(100 * float(kp["false_kill_share"])),
                round(100 * float(kp["validated_share"])),
                round(100 * float(kp["power_vs_block_shuffle"])),
                round(100 * (1 - float(kp["null_noise_pass"]))))
    except (TypeError, ValueError):                                # pragma: no cover - 방어
        return None


def kill_power_bi(src, member: str = "p2") -> str:
    """kill_power_line 의 한/영 판.

    주간 자기검사가 만든 문장(sentence_ko)은 **산출물 원문**이라 번역하지 않는다 — 스팬 밖에 두어
    두 언어에서 같은 문장이 보이게 한다(숫자가 언어에 따라 달라지면 안 된다)."""
    kp = _sub(src, "kill_power").get(member) if isinstance(src, dict) else None
    if not isinstance(kp, dict) or not kp.get("sentence_ko"):
        return bi(kill_power_line(src, member),
                  "How well the switch-off rule can tell a good model from a bad one: this week's self-test "
                  "(kill_replay) did not regenerate the figure, so it is not printed here (§16.7 — what is "
                  "not regenerated is not written on the page).")
    head_ko = (f"성적이 나쁘면 끄는 규칙이 얼마나 잘 가려내는지({kp.get('p_col') or member}, 주간 자기검사 "
               f"재생성 · 창 {_fmt(kp.get('n_windows'), 'n')})")
    head_en = (f"How well the switch-off rule can tell a good model from a bad one ({kp.get('p_col') or member}, "
               f"regenerated by the weekly self-test · {_fmt(kp.get('n_windows'), 'n')} windows)")
    tail = bi("2단계로 나눠 읽는 방식은 문자 그대로 읽는 것보다 더 엄격합니다(#7).",
              "Reading it in two stages is stricter than reading it literally (#7).")
    nums = _kill_power_nums(kp)
    if nums is None:                       # 숫자가 없으면 산출물 원문을 그대로 인용한다(§16.7)
        body = (bi("계산이 남긴 한국어 원문", "raw Korean written by the pipeline") + " "
                + _raw(str(kp["sentence_ko"])))
    else:
        m, fk, vs, pw, nz = nums
        # '기후학' → '평소 평균'(§3 대응표), '오기각/기각/검정력' → 누가 무엇을 하는지 그대로 풀어 쓴다.
        # 79% 짜리 귀무모형은 '실력 없는 모형' 이 아니라 **날짜 순서만 섞은** 모형이다 — 그 한정을 쉬운 말로 남긴다.
        body = bi(
            f"실력이 진짜 있는 모델인데도 {m}개월 기록만 보고 잘못 끄는 경우가 100번 중 {fk}번이었습니다. 반대로 '잘한다' 고 인정해 주는 경우는 100번 중 "
            f"{vs}번뿐이었습니다. 숫자가 움직인 크기는 그대로 두고 날짜 순서만 뒤섞어 정보를 없앤 모델은 100번 중 "
            f"{pw}번, 평소 평균에 잡음만 섞은 모델은 100번 중 {nz}번 껐습니다.",
            f"Even for a model that really has skill, {m} months of record wrongly switched it off "
            f"{fk} times out of 100, and "
            f"it was called 'passed' only {vs} times out of 100. A model whose day order was shuffled — same "
            f"size of moves, no timing information — was switched off {pw} times out of 100, and one that is "
            f"just the long-run average plus noise {nz} times out of 100.")
    return bi_html(head_ko, head_en) + ": " + body + " " + tail


def _p3_honesty_section(sec_id: str, extra=None, *, kill_power_text: str | None = None) -> str:
    """§16 정직 문구 — 한/영 두 벌.

    extra 는 (한국어, English) 쌍의 목록, kill_power_text 는 이미 두 벌로 만들어진 HTML(kill_power_bi).
    """
    items = [bi(ko, en) for ko, en in zip(P3_HONESTY_ITEMS, P3_HONESTY_ITEMS_EN)]
    if kill_power_text:                                   # 지운 정적 문장이 있던 자리에 산출값을 넣는다
        items.insert(6, str(kill_power_text))
    for x in (extra or []):
        items.append(bi(x[0], x[1]) if isinstance(x, (tuple, list)) else _esc(str(x)))
    return (f'<section class="panel" id="{_esc(sec_id)}"><h2>'
            + bi("정직 문구와 리스크 (§16)", "Honesty notes and the risks that remain (§16)") + "</h2>"
            '<ul class="plain">' + "".join(f"<li>{x}</li>" for x in items) + "</ul>"
            + _n(P3_CARD_FOOTNOTE, P3_CARD_FOOTNOTE_EN) + "</section>")


def _panel_kv(block, live_label=("라이브", "live"), ref_label=("백테스트 참조", "backtest reference")) -> str:
    """라이브 값과 백테스트 참조를 **같은 행**에 두는 표 (§8.5). 숨겨진 항목은 사유를 남긴다.

    live_label·ref_label 은 (한국어, English) 쌍이다. 셀 값은 손대지 않고 열 제목만 두 벌로 만든다.
    """
    live_ko, live_en = (live_label if isinstance(live_label, (tuple, list)) else (live_label, live_label))
    ref_ko, ref_en = (ref_label if isinstance(ref_label, (tuple, list)) else (ref_label, ref_label))
    b = dict(block) if isinstance(block, dict) else {}
    bt = b.pop("backtest", None)
    hidden = b.pop("hidden", None)
    reason = b.pop("hidden_reason", None)
    btd = bt if isinstance(bt, dict) else {}
    flat = {k: v for k, v in b.items() if not isinstance(v, (dict, list, tuple))}
    nested = {k: v for k, v in b.items() if isinstance(v, (dict, list, tuple))}
    rows = "".join(f'<tr><td>{_label_bi(k)}</td><td class="num">{_fmt_p3(v, k)}</td>'
                   f"<td class=\"num\">{_fmt_p3(btd.get(k), k) if k in btd else '—'}</td></tr>"
                   for k, v in flat.items())
    out = ""
    if rows:
        out += ('<div class="tblwrap"><table><thead><tr>' + _th_bi("항목", "item")
                + _th_bi(live_ko, live_en, num=True) + _th_bi(ref_ko, ref_en, num=True)
                + f"</tr></thead><tbody>{rows}</tbody></table></div>")
    if not isinstance(bt, dict) and bt is not None:
        out += f'<div class="note">{bi(ref_ko, ref_en)}: {_fmt_p3(bt, "backtest")}</div>'
    for k, v in nested.items():
        recs = _to_records(v)
        if recs:
            out += _h3(_label_ko(k), _label_en(k)) + _p3_table(recs)
        elif isinstance(v, dict) and v:
            out += _h3(_label_ko(k), _label_en(k)) + _p3_kv(v)
    if hidden:
        out += ('<div class="note">' + bi("숨긴 항목", "hidden here") + ": "
                + _esc(", ".join(str(x) for x in hidden))
                + (" — " + bi(str(reason), _reason_en(reason)) if reason else "") + "</div>")
    return out or _empty_note(_EMPTY_DEFAULT)


def _p3_window_tables(records: list[dict], order: list[str], empty) -> str:
    """행에 'window' 가 있으면 창별로 나눠 §6.3 형식을 유지한다. 기간 이름(2003+ 등)은 번역하지 않는다."""
    if not records:
        return _empty_note(empty)
    wins: list[str] = []
    for r in records:
        w = str(r.get("window") or "")
        if w not in wins:
            wins.append(w)
    if len(wins) <= 1:
        return _p3_table(records, order, empty)
    out = []
    for w in wins:
        rows = [r for r in records if str(r.get("window") or "") == w]
        head = f"<h3>{_esc(w)}</h3>" if w else _h3("기간 미상", "period unknown")
        out.append(head + _p3_table(rows, order, empty))
    return "".join(out)


def _p3_mode_source(s: dict, acc: dict) -> dict:
    """summary_p3/track_p3 에서 유효 모드 계산 입력을 모은다."""
    run = _sub(s, "run")
    m3 = _sub(s, "model_p3")
    return {"p2_deploy_mode": _first(run, "p2_deploy_mode") or _first(acc, "deploy_mode")
                              or _first(_sub(s, "p2"), "deploy_mode"),
            "p3_deploy_mode": _first(s, "deploy_mode") or _first(m3, "deploy_mode") or _first(run, "deploy_mode"),
            "kill": _sub(s, "kill") or _sub(_sub(s, "track"), "kill"),
            "kill_record": s.get("kill_record"), "kill_manual": s.get("kill_manual"),
            "acceptance": acc, "deploy_sizing": _first(_sub(s, "sizing"), "deploy_sizing",
                                                       default=s.get("deploy_sizing"))}


# ------------------------------------------------------------------
# 주간 리포트 — 비중 (docs/sizing_p3.html)  §10 주간 ①~⑩
# ------------------------------------------------------------------
_SIZING_SECTIONS = [("① v0 요약(영구)", "① v0 summary (always)"),
                    ("② p2 판정 규칙", "② the p2 decision rule"),
                    ("③ 정직한 읽기", "③ reading it honestly"),
                    ("④ 백테스트 표", "④ backtest table"),
                    ("⑤ 담을 수 있는 최대치 단계", "⑤ budget ladder"),
                    ("⑥ 민감도", "⑥ sensitivity"),
                    ("⑦ 하락 사건별 손익", "⑦ episode by episode"),
                    ("⑧ 유지 조건", "⑧ conditions to keep using it"),
                    ("⑨ 창별 상대성과", "⑨ rule vs holding, by window"),
                    ("⑩ 규약·해시", "⑩ conventions and hashes"),
                    ("⑪ 정직 문구", "⑪ honesty notes")]


def render_sizing_report(summary_p3: dict, summary_v1: dict, summary_v0: dict, out_html: Path,
                         charts: dict[str, bytes]) -> None:
    """docs/sizing_p3.html — ① v0 completed 줄(영구) ② p2 결정층 줄 ③ 정직한 읽기 (a)~(d) ④ §6.3 백테스트 표(세 창)
    ⑤ D_max 사다리(두 창·프론티어) ⑥ 민감도(선택에 쓰지 않음) ⑦ 에피소드 손익 ⑧ 유지 조건 (a)~(f) ⑨ 63/126/252 창
    상대성과 ⑩ 규약(5bp·현금 0%·점 원칙)·sizing_sha ⑪ §16 정직 문구. 없는 키는 '자료 없음'."""
    if not isinstance(summary_p3, dict):
        raise TypeError("summary_p3 는 dict 여야 합니다")
    s = summary_p3
    charts = dict(charts or {})
    run = _sub(s, "run")
    sz = _sub(s, "sizing")
    acc = p2_acceptance_of(s, summary_v1)
    eff = p3_effective_mode(_p3_mode_source(s, acc))
    deploy_sizing = _first(sz, "deploy_sizing", default=s.get("deploy_sizing"))
    warns = [str(w) for w in (s.get("warnings") or [])]

    sT = _fnum(_first(run, "sigma_target", default=sz.get("sigma_target")))
    d_max = _fnum(_first(run, "d_max", default=sz.get("d_max")))
    tags = [f'<span class="tag">σ_T <b>{_pct1(sT, 0)}</b></span>',
            '<span class="tag">' + bi("감당할 최대 하락폭", "worst drop we accept")
            + f' D_max <b>{_pct1(d_max, 0)}</b></span>',
            f'<span class="tag">sizing_sha <b class="mono">{_esc(str(run.get("sizing_sha") or "—")[:12])}</b></span>',
            '<span class="tag">' + bi("생성", "generated")
            + f' <b>{_esc(run.get("generated_at_utc") or _now_str())}</b></span>',
            f'<span class="tag{"" if deploy_sizing else " warn"}">deploy_sizing <b>'
            f'{_yn_bi(deploy_sizing)}</b></span>']
    if warns:
        tags.append(_tag_html(f"경고 {len(warns)}건", f"{len(warns)} warnings", cls=" warn"))
    head = _p3_head(("market-risk-lab · Phase 3 · 주식을 얼마나 담을지 — 맞춘 숫자 0개",
                     "market-risk-lab · Phase 3 · how much to hold — zero fitted numbers"),
                    ("감당할 수 있는 하락폭에서 출발해, 목표 변동성과 상태 배수로 주식 비중을 정합니다",
                     "Start from the drop the family can accept, then set the share from a volatility "
                     "target and a state multiplier"),
                    # 부제는 **뜻만** 말한다. 수식·상수는 바로 아래 .note 에 원문 그대로 남긴다
                    # (§2.1 먼저 뜻, 그다음 용어 / §2.3 한 문장 = 한 가지).
                    ("시장이 많이 출렁이는 때에는 주식을 덜 담고, 조용해지면 다시 늘립니다. 목표로 삼는 "
                     "출렁임 폭(σ_T, 위 태그의 값)을 정해 두고 거기에 맞춥니다. 주식 비중은 최소 25%, 최대 "
                     "100% 사이에서만 움직입니다. 5%p 씩 계단으로 바꾸고, 10%p 보다 작은 차이는 그냥 둡니다"
                     "(거래비용 5bp · 현금 이자 0%). 맞춘 숫자는 0개지만 상수는 1993~2024 기록을 보고 "
                     "정했으므로, 그 기록에 대해서는 결과를 본 뒤 정한 것(post hoc)입니다.",
                     "When the market swings a lot we hold less in stocks, and when it calms down we hold "
                     "more. We pick a target swing size (σ_T, the value in the tag above) and aim at it. The "
                     "share in stocks only moves between 25% and 100%. It changes in steps of 5pp, and gaps "
                     "smaller than 10pp are left alone (5bp trading cost · cash earns 0%). Zero numbers are "
                     "fitted, but the constants were chosen after looking at the 1993-2024 record, so with "
                     "respect to that record they are post hoc."),
                    tags, acc, eff, deploy_sizing)
    # 수식과 상수는 하나도 빼지 않고 여기에 그대로 남긴다(부제에서 내린 것뿐이다 — 정직성 손실 0).
    head += _n("규칙 원문 그대로: σ̂ = EWMA(λ=0.94) · σ_T = D_max/3.5 를 격자로 내림 · "
               "w = clip(clip(σ_T/σ̂, .25, 1) × 상태 배수, .25, 1) · 5%p 격자 · 10%p 무거래 밴드 · "
               "주 마지막 날 갱신 · 판정 규칙이 위험을 올리면 즉시 하향 · 거래비용 5bp · 현금 이자 0% · "
               "맞춘 숫자 0개.",
               "The rule as written: σ̂ = EWMA(λ=0.94) · σ_T = D_max/3.5 rounded down to the grid · "
               "w = clip(clip(σ_T/σ̂, .25, 1) × state multiplier, .25, 1) · 5pp grid · 10pp no-trade band · "
               "updated on the last session of the week · cut immediately when the decision rule raises the "
               "risk step · 5bp trading cost · cash earns 0% · 0 fitted numbers.")
    nav = _p3_nav(_SIZING_SECTIONS)

    # ① v0 completed 줄(영구)
    v0src = summary_v0 if isinstance(summary_v0, dict) and summary_v0 else load_summary_v0()
    s1 = ('<section class="panel" id="s1">'
          + _h2("① v0 completed 요약 (영구 표기)", "① v0 completed summary (always shown)")
          + _v0_line_html(v0_headline(v0src), bi_mode=True) + "</section>")

    # ② p2 결정층 줄
    tbl = _to_records(sz.get("table"))
    p2row = [r for r in tbl if str(r.get("row")) in ("p2_decision", "p2 결정층")]
    v1 = summary_v1 if isinstance(summary_v1, dict) else {}
    s2 = ('<section class="panel" id="s2">'
          + _h2("② 판정 규칙만 쓰는 가벼운 대안 (p2 결정층)", "② the lighter alternative: the p2 decision rule")
          + _n("판정 규칙은 평균적으로 주식을 약 89% 담아 그냥 보유에 가깝습니다. 여기 규칙은 약 69% 만 담아 "
               "하락폭을 더 줄이는 대신 수익을 더 내놓습니다. 가족은 단계별 예산 표에서 더 높은 행을 고를 수도 "
               "있습니다(§16 3).",
               "The decision rule holds about 89% in stocks on average, close to simply holding. This rule "
               "holds about 69%: it cuts the fall further but gives up more return. The family can pick a "
               "higher row on the budget ladder (§16 3).")
          + (_p3_table(p2row, _P3_BT_ORDER) if p2row else
             _p3_kv({k: _sub(v1, "allocation").get(k) for k in _ALLOC_KEYS} if _sub(v1, "allocation") else {},
                    ("판정 규칙 행이 없습니다 — 이 표를 그리려면 아래 자료가 있어야 합니다",
                     "no decision-rule row — this table needs one of the following",
                     "summary_p3.sizing.table / summary_v1.allocation")))
          + "</section>")

    # ③ 정직한 읽기 (a)~(d) — 표 위에 그대로
    hr = sz.get("honest_reading")
    if not isinstance(hr, (list, tuple)) or not hr:
        try:
            hr = _sizing.honest_reading(sz.get("honest_results") if isinstance(sz.get("honest_results"), dict) else None)
        except (ValueError, TypeError) as e:                       # pragma: no cover - 방어
            warnings.warn(f"honest_reading 실패({type(e).__name__}) → 기본 문구")
            hr = list(_sizing.HONEST_READING)
    s3 = ('<section class="panel" id="s3">'
          + _h2("③ 정직한 읽기 (§6.5) — 표 위에 그대로", "③ Reading it honestly (§6.5) — printed above the tables")
          # 쉬운 요약 먼저(§2.1) — 숫자는 다시 쓰지 않는다. 숫자는 아래 원문에만 두어야 원문과 어긋날 수 없다.
          + _n("쉬운 요약 — (a) 확률 모델 없이 출렁임만 보고 정해도 판정 규칙과 성적이 거의 같았습니다. "
               "(b) 확률 모델을 곱해도 고점 대비 하락폭은 거의 줄지 않았습니다(바닥을 없애야만 줄고, 그건 "
               "2008~09·2020 에 주식을 거의 다 판 것과 같습니다). (c) 천천히 흘러내리는 약세장은 막지 "
               "못합니다. (d) 모든 규칙이 그냥 들고 있는 것보다 1년 평균 수익률(CAGR)이 낮고 대부분의 해에 "
               "뒤집니다 — 이 규칙이 파는 것은 수익이 아니라 고점 대비 하락폭을 줄이는 일입니다. "
               "아래 네 줄이 계산이 남긴 원문이고, 숫자는 전부 그 원문에 있습니다.",
               "In plain words — (a) even with no probability model, going by the size of the swings alone "
               "does about as well as the decision rule. (b) Multiplying by the probability model barely "
               "reduces the drop from the peak (it only helps if the floor is removed, which means selling "
               "almost all the stock in 2008-09 and 2020). (c) It does not stop a slow, grinding bear "
               "market. (d) Every rule earns less per year (CAGR) than simply holding and is behind in most "
               "years — what this rule sells is a smaller drop from the peak, not a higher return. The four "
               "lines below are the pipeline's own text, and every number lives there.")
          + _n("아래 네 줄은 계산 결과 원문입니다. 표를 보기 전에 먼저 읽어 주세요.",
               "The four lines below are quoted from the run. Read them before the tables.")
          + '<div class="quote">' + "<br><br>".join(_honest_reading_bi(sz, hr)) + "</div></section>")

    # ④ 백테스트 표 (세 창)
    s4 = ('<section class="panel" id="s4">'
          + _h2("④ 백테스트 성적표 (§6.3 형식 고정)", "④ Backtest scorecard (fixed §6.3 layout)")
          + _n("과거에 이 규칙을 그대로 돌렸으면 어땠을지입니다. 오늘 정한 주식 비중은 다음 날 수익에 적용하고"
               "(w[t] → r[t+1]), 주식 비중을 바꾼 날에만 바뀐 만큼의 5bp 를 비용으로 빼며, 현금은 이자를 0% 로 "
               "봅니다. 늘 보유·v0 completed(2015+)·p2 판정 규칙·변동성만 맞춘 참조(VT14) 줄을 함께 싣습니다.",
               "What this rule would have done in the past. Today's share is applied to tomorrow's return "
               "(w[t] → r[t+1]); on days the share changes we subtract 5bp of the change as cost; cash earns "
               "0%. Rows for always holding, v0 completed (2015+), the p2 decision rule and a volatility-only "
               "reference (VT14) are shown alongside.")
          + _p3_window_tables(tbl, _P3_BT_ORDER,
                              ("성적표가 없습니다", "no backtest table", "summary_p3.sizing.table"))
          + _img_bi(charts.pop("cum_returns", None),
                    "돈이 불어난 모습(로그 눈금) — 규칙 vs 그냥 보유 vs p2 판정 규칙 vs v0. 규칙은 대부분의 "
                    "해에 그냥 보유에 뒤집니다.",
                    "Growth of 1.0 (log scale) — rule vs holding vs the p2 decision rule vs v0. The rule "
                    "lags plain holding in most years.",
                    "Cumulative growth: rule vs hold vs p2 vs v0")
          + _img_bi(charts.pop("drawdown", None),
                    "고점 대비 하락폭 비교 — 규칙 vs 그냥 보유. 이 상품이 파는 것은 하락폭 통제입니다.",
                    "Drop from the peak — rule vs holding. What this product sells is a smaller fall.",
                    "Drawdown comparison")
          + _img_bi(charts.pop("w_path", None),
                    "주식 비중이 지나온 길과 SPY — 바닥 0.25, 최대 1.00.",
                    "The path of the stock share against SPY — floor 0.25, cap 1.00.",
                    "Exposure path vs SPY")
          + "</section>")

    # ⑤ D_max 사다리
    lad = _to_records(sz.get("ladder"))
    s5 = ('<section class="panel" id="s5">'
          + _h2("⑤ 단계별 예산 표 — 같은 목표 변동성을 두 예산선으로 (D_max 사다리)",
                "⑤ The budget ladder — one volatility target, two budget lines")
          + _n_html("감당할 하락폭을 얼마로 잡느냐에 따라 목표 변동성이 달라집니다. D_slow = 3.5×σ_T 는 천천히 "
                    "내려가는 약세장 기준(대표값)이고, D_fast = 2.0×σ_T 는 급락형 낙관 기준(같이 적어 둡니다). "
                    "고점 대비 최대 하락폭은 1993년부터 변동성만 본 경우와 2003년부터 판정 규칙까지 넣은 경우를 "
                    "나란히 둡니다. <b>예산은 보장이 아닙니다</b> — k_slow 3.5 는 2000~02년 한 사건을 보고 맞춘 "
                    "상수입니다(§16 4).",
                    "The drop you are willing to accept sets the volatility target. D_slow = 3.5×σ_T is the "
                    "slow-bear line (the headline); D_fast = 2.0×σ_T is the optimistic crash line (shown "
                    "beside it). The worst drop is given twice: volatility only from 1993, and with the "
                    "decision rule from 2003. <b>A budget is not a guarantee</b> — k_slow 3.5 was tuned on a "
                    "single 2000-02 episode (§16 4).")
          + _p3_table(lad, _P3_LADDER_ORDER,
                      ("단계별 예산 표가 없습니다", "no ladder rows", "summary_p3.sizing.ladder"))
          + _img_bi(charts.pop("ladder_frontier", None),
                    "목표 변동성별 최대 하락폭 곡선(두 기간)과 두 예산선 — 예산선 아래로 내려가면 예산을 넘긴 "
                    "것입니다.",
                    "Worst drop against the volatility target (two windows) with the two budget lines — "
                    "below a line means the budget was exceeded.",
                    "MaxDD vs sigma_T frontier")
          + _n_html(f"기본 예산 {_pct1(d_max, 0)} → 목표 변동성 {_pct1(sT, 0)}. 소유자가 가구 기본값을 장부에 "
                    "등록하고, 다른 가족은 자기 행을 읽으면 됩니다.",
                    f"Default budget {_pct1(d_max, 0)} → volatility target {_pct1(sT, 0)}. The owner records "
                    "the household default in the ledger; everyone else reads their own row.")
          + "</section>")

    # ⑥ 민감도 (선택에 쓰지 않음)
    sens = _to_records(sz.get("sensitivities"))
    s6 = ('<section class="panel" id="s6">'
          + _h2("⑥ 조금씩 바꿔 보기 (민감도 — 보여 주기만 하고 선택에 쓰지 않습니다)",
                "⑥ What if we tweaked it (sensitivity — shown, never used to choose)")
          + _n("바닥 없음(B 원안)·월간 갱신·완만한 배수·0.25 격자·VT14·즉시 하향 없음·1세션 지연·10bp·S-HAR·"
               "일간 5%p·목표 격자로 하나씩 바꿔 본 결과입니다. 그럴듯한 변형들 사이의 차이가 작아서 한 '규칙의 "
               "가족' 으로 읽고, 성적이 좋아 보이는 쪽으로 다시 고르지 않습니다(§16 6, 장부 #6a).",
               "Each row changes one thing: no floor (the original B), monthly updates, gentler multipliers, "
               "a 0.25 grid, VT14, no immediate cut, a one-session delay, 10bp costs, S-HAR, a daily 5pp "
               "grid, a target grid. The differences between reasonable variants are small, so we read them "
               "as one family of rules and never re-pick whichever looks best (§16 6, ledger #6a).")
          + _p3_window_tables(sens, _P3_BT_ORDER,
                              ("변형 결과가 없습니다", "no sensitivity rows", "summary_p3.sizing.sensitivities"))
          + "</section>")

    # ⑦ 에피소드 손익
    ep = _to_records(sz.get("episode_pnl"))
    s7 = ('<section class="panel" id="s7">'
          + _h2("⑦ 하락 사건 하나하나의 손익 (에피소드 손익 — 규칙 vs 그냥 보유, 그 기간 평균 주식 비중)",
                "⑦ Episode by episode (rule vs holding, and the average share held)")
          + _n("−5% 를 처음 지나는 날의 판정 상태는 대부분 normal 입니다 — 모델이 첫날을 맞힐 것이라 기대하지 "
               "마세요. 조용하다가 갑자기 떨어진 경우(2018-02, 2020-02 첫 주)에는 보호가 거의 없었습니다.",
               "On the day a fall first passes −5% the decision state is usually normal — do not expect the "
               "model to catch day one. When the drop came out of a calm market (2018-02, the first week of "
               "2020-02) there was almost no protection.")
          + _p3_table(ep, ["peak_date", "trough_date", "depth", "ret_rule", "ret_bh", "protection_pp",
                           "avg_exposure", "n_sessions"],
                      ("하락 사건 표가 없습니다", "no episode rows", "summary_p3.sizing.episode_pnl"))
          + "</section>")

    # ⑧ 유지 조건 (a)~(f)
    ret = _sub(sz, "retention")
    conds = _sub(ret, "conditions")
    crows = []
    for k in sorted(conds):
        c = conds[k] if isinstance(conds[k], dict) else {}
        crows.append({"condition": f"({k}) {_sizing.RETENTION_CONDITIONS.get(k, '')}",
                      "value": c.get("value"), "threshold": c.get("threshold"),
                      "pass": c.get("pass"), "margin": c.get("margin"), "note": c.get("note")})
    s8 = ('<section class="panel" id="s8">'
          + _h2("⑧ 계속 쓰기 위한 조건 (유지 조건 §6.4 — 공식 산출로만 판정)",
                "⑧ Conditions for keeping this rule (§6.4 — judged only on official numbers)")
          + _n("미리 정해 둔 여섯 가지 조건입니다. 하나라도 어기면 주식 비중 블록을 감춥니다.",
               "Six conditions fixed in advance. Break any one and the sizing block is hidden.")
          + _p3_table(crows, ["condition", "value", "threshold", "pass", "margin", "note"],
                      ("유지 조건 기록이 없습니다", "no retention rows", "summary_p3.sizing.retention"))
          + (_n_raw(str(ret.get("note_ko"))) if ret.get("note_ko") else "")
          + ('<div class="tag bad">'
             + bi_html(f'근소 통과 {_esc(", ".join(str(x) for x in ret.get("narrow") or []))} — 여유가 '
                       f'{_pct1(ret.get("narrow_margin") or _sizing.NARROW_MARGIN, 0)} 도 안 되게 아슬아슬해서, '
                       "공식 수치가 조금만 달라져도 뒤집힐 수 있습니다",
                       f'passed narrowly: {_esc(", ".join(str(x) for x in ret.get("narrow") or []))} — with '
                       f'less than {_pct1(ret.get("narrow_margin") or _sizing.NARROW_MARGIN, 0)} to spare, a '
                       "small change in the official numbers could flip it")
             + "</div>" if ret.get("narrow") else "")
          + _n_html("라이브에서는 D4(최근 60세션 실제 움직임 ÷ σ_T &gt; 1.5) · D4b(라이브 최대 하락폭이 "
                    "−D_max 보다 깊음) · D6(1년에 24번 넘게 바꿈) 중 하나만 걸려도, 소유자가 장부에 항목을 쓸 "
                    "때까지 주식 비중 카드를 멈춥니다. 성적이 나쁘면 끄는 규칙(§7)은 이 성적표와 상관없이 "
                    "상태 배수와 주식 비중 제안을 감춥니다.",
                    "Live, any one of D4 (last 60 sessions actual swing ÷ σ_T &gt; 1.5), D4b (live worst drop "
                    "deeper than −D_max) or D6 (more than 24 changes a year) halts the sizing card until the "
                    "owner writes a ledger entry. The switch-off rule (§7) hides the state multiplier and the "
                    "suggested share regardless of this scorecard.")
          + "</section>")

    # ⑨ 창별 상대성과
    rel = sz.get("relative") if isinstance(sz.get("relative"), dict) else {}
    rel_rows = []
    for k, v in rel.items():
        if isinstance(v, dict):
            rel_rows.append({"L": v.get("L", k), **{kk: v.get(kk) for kk in ("n", "p10", "p50", "p90",
                                                                             "share_behind", "worst", "best")}})
    cy = rel.get("calendar_year") if isinstance(rel.get("calendar_year"), dict) else {}
    # §15 단계 0: 설계 단계 근사(79/88/90%)를 공식 표에 섞지 않는다 — 문장을 바로 아래 표와 같은 값으로 만든다
    _sb = "/".join(_pct1(rel[k].get("share_behind")) if isinstance(rel.get(k), dict) else "—"
                   for k in ("63", "126", "252"))
    _cyb = _pct1(cy.get("share_behind"), 0) if cy else "—"
    s9 = ('<section class="panel" id="s9">'
          + _h2("⑨ 63/126/252 세션 창의 규칙 − 보유 (겹치지 않는 창)",
                "⑨ Rule minus holding over 63/126/252-session windows (non-overlapping)")
          + _n_html(f"이 상품이 파는 것은 하락폭 통제이지 더 높은 수익이 아닙니다. 63/126/252 세션 창에서 그냥 "
                    f"보유에 뒤질 확률은 각각 {_sb} 이고, 달력 해 기준으로는 {_cyb} 에서 뒤집니다 — 이 줄은 "
                    "카드에서도 지우지 않습니다(§16 3).",
                    f"What this sells is a smaller fall, not a higher return. Over 63/126/252-session "
                    f"windows it lags plain holding {_sb} of the time, and in {_cyb} of calendar years — "
                    "this line is never removed from the card either (§16 3).")
          + _p3_table(rel_rows, ["L", "n", "p10", "p50", "p90", "share_behind", "worst", "best"],
                      ("창별 비교 기록이 없습니다", "no window comparison", "summary_p3.sizing.relative"))
          + (_n_html(f'달력 해로 보면 {_fmt(cy.get("n_behind"), "n")}/{_fmt(cy.get("n_years"), "n")}년 뒤졌습니다 '
                     f'(중앙 {_pp1(cy.get("median"))} · 10/90분위 {_pp1(cy.get("p10"))}/{_pp1(cy.get("p90"))})',
                     f'By calendar year it lagged in {_fmt(cy.get("n_behind"), "n")} of '
                     f'{_fmt(cy.get("n_years"), "n")} years (median {_pp1(cy.get("median"))} · 10th/90th '
                     f'{_pp1(cy.get("p10"))}/{_pp1(cy.get("p90"))})') if cy else "")
          + "</section>")

    # ⑩ 규약·해시
    conv = {"cost_bps": run.get("cost_bps", P3["cost_bps"]), "cash_return": P3["cash_return"],
            "grid": P3["grid"], "band": P3["band"], "w_min": P3["w_min"], "w_max": P3["w_max"],
            "k_slow": P3["k_slow"], "k_fast": P3["k_fast"], "ewma_lambda": P3["ewma_lambda"],
            "ewma_seed_sessions": P3["ewma_seed_sessions"], "sizing_sha": run.get("sizing_sha"),
            "registry_sha": run.get("registry_sha"), "spec_sha256": run.get("spec_sha256"),
            "end": run.get("end"), "d_max": d_max, "sigma_target": sT}
    s10 = ('<section class="panel" id="s10">'
           + _h2("⑩ 규약과 해시", "⑩ Conventions and hashes")
           + _n("점 원칙: 오늘 종가에 정한 주식 비중을 내일 수익률에 적용합니다(\"내일부터 주식 비중 X%\"). "
                "주식 비중을 바꾼 날에만 바뀐 만큼의 5bp 를 비용으로 빼고, 현금 이자는 0% 로 봅니다 — 보수적인 "
                "가정입니다(평균 현금 약 31% 에 단기금리 2% 를 주면 해마다 +0.6%p).",
                "Timing rule: the share fixed at today's close is applied to tomorrow's return (\"from "
                "tomorrow, X% in stocks\"). On days the share changes we subtract 5bp of the change, and cash "
                "earns 0% — a conservative assumption (about 31% average cash at a 2% short rate would add "
                "roughly +0.6pp a year).")
           + _p3_kv(conv)
           + f'<div class="note">{bi("규칙 문자열", "rule string")}: '
             f'<span class="mono">{_esc(_sizing.SIZING_RULE)}</span></div>'
           + '<div class="note">'
           + bi_html(f"맞춘(적합) 파라미터 수 = {_sizing.n_params()} — §6-b 는 반드시 0 이어야 합니다",
                     f"fitted parameters = {_sizing.n_params()} — §6-b requires exactly 0")
           + " · " + p3_param_bi(s) + "</div>"
           + (_h3("경고", "warnings") + _warns_bi(warns) if warns else "")
           + "</section>")

    body = (head + nav + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10
            + _p3_honesty_section("s11", kill_power_text=kill_power_bi(s)))
    _write_html(out_html, "Sizing rule (Phase 3) - market-risk-lab", body)


# ------------------------------------------------------------------
# 주간 리포트 — 국면·등록부 (docs/regime_p3.html)  §10 주간 ①~⑨
# ------------------------------------------------------------------
_REGIME_SECTIONS = [("① 등록부·파라미터 회계", "① candidate list & parameter count"),
                    ("② 단계 늘려 보기", "② extending the model steps"),
                    ("③ 블록 표", "③ block tables"),
                    ("④ θ 경로·guard", "④ θ path & guard"),
                    ("⑤ P_high", "⑤ P(high-swing)"),
                    ("⑥ 시대별 AUC", "⑥ AUC by era"),
                    ("⑦ 소거", "⑦ ablations"),
                    ("⑧ 말한 확률이 실제와 맞나", "⑧ did the stated % match reality"),
                    ("⑨ 결정론·PIT·spec", "⑨ determinism, PIT, spec"),
                    ("⑩ 정직 문구", "⑩ honesty notes")]


def render_regime_report(summary_p3: dict, out_html: Path, charts: dict[str, bytes]) -> None:
    """docs/regime_p3.html — ① 등록부 표(멤버·K_s/K_u·상태·장부 번호·블록 기록·채택 판정·검정력·신선 블록)
    ② 사다리 확장 ③ 블록 표(24·18) ④ θ 경로·guard 로그·체류 ⑤ P_high 밴드 ⑥ 시대별 AUC ⑦ 소거 ⑧ 신뢰도
    ⑨ 결정론·PIT·카나리·spec ⑩ §16 정직 문구."""
    if not isinstance(summary_p3, dict):
        raise TypeError("summary_p3 는 dict 여야 합니다")
    s = summary_p3
    charts = dict(charts or {})
    run = _sub(s, "run")
    acc = p2_acceptance_of(s)
    eff = p3_effective_mode(_p3_mode_source(s, acc))
    warns = [str(w) for w in (s.get("warnings") or [])]
    hmm = _sub(s, "hmm")

    tags = [f'<span class="tag">registry_sha <b class="mono">{_esc(str(run.get("registry_sha") or "—")[:12])}</b></span>',
            f'<span class="tag">OOS <b>{_esc(str(run.get("first_refit") or P2["first_refit"]))} ~ '
            f'{_esc(str(run.get("end") or "—"))}</b></span>',
            '<span class="tag">' + bi("손대지 않고 남겨 둔 구간", "untouched recent data")
            + f' <b>{_esc(HOLDOUT_START)}~</b></span>',
            '<span class="tag">' + bi("생성", "generated")
            + f' <b>{_esc(run.get("generated_at_utc") or _now_str())}</b></span>']
    if warns:
        tags.append(_tag_html(f"경고 {len(warns)}건", f"{len(warns)} warnings", cls=" warn"))
    head = _p3_head(("market-risk-lab · Phase 3 · 시장 분위기 엔진(HMM) · 후보 모형 명단",
                     "market-risk-lab · Phase 3 · market-mood engine (HMM) · candidate models"),
                    ("시장을 '평온'과 '많이 출렁임' 두 분위기로 보는 모형과, 아직 쓰지 않는 후보 모형 명단",
                     "A model that sees the market as calm or turbulent, and the list of candidates not "
                     "yet in use"),
                    ("이 모형은 정답 라벨을 보지 않고 종가만으로 스스로 두 분위기를 찾아냅니다(라벨 없이 정한 "
                     "숫자 12개). 거기서 나온 '많이 출렁일 확률' 을 확률로 바꿔 준 것이 후보 H 입니다. 오늘 "
                     "쓰는 확률은 Phase 2 그대로이고 명단에 오른 모형은 전부 아직 쓰지 않는 후보입니다 — "
                     "승격은 손대지 않은 새 자료로만 이뤄집니다(§6-c(d)).",
                     "The model finds the two moods on its own from closing prices, never seeing the answer "
                     "labels (12 numbers fitted without labels). Turning its 'chance of a turbulent day' "
                     "into a probability gives candidate H. Today's probability is still the Phase 2 one, "
                     "and every listed model is a candidate not in use — promotion happens only on untouched "
                     "new data (§6-c(d))."),
                    tags, acc, eff)
    nav = _p3_nav(_REGIME_SECTIONS)

    # ① 등록부 + §14 파라미터 회계
    reg = _to_records(s.get("registry"))
    adm = s.get("admission")
    pw = _to_records(_sub(s, "kill_power").get("admission_power") or s.get("admission_power"))
    fb = s.get("fresh_blocks") or _sub(s, "reference").get("fresh_blocks")
    s1 = ('<section class="panel" id="s1">'
          + _h2("① 후보 모형 명단 (등록부 — 한 번 적으면 고정, 바꾸려면 번호 붙인 장부 항목이 필요합니다)",
                "① The candidate list (frozen once written; changing it needs a numbered ledger entry)")
          + _p3_table(reg, _P3_REGISTRY_ORDER, ("명단이 없습니다", "no registry rows", "summary_p3.registry"))
          + _h3("채택 검정 (이미 본 자료로는 떨어뜨릴 수만 있고, 뽑는 것은 손대지 않은 새 자료로만)",
                "Admission test (data we have already seen can only reject; only fresh data can admit)")
          + (_p3_table(_to_records(adm), None,
                       ("채택 검정 기록이 없습니다", "no admission rows", "summary_p3.admission"))
             if _to_records(adm) else
             _p3_kv(adm, ("채택 검정 기록이 없습니다", "no admission rows", "summary_p3.admission")))
          + (_h3("검정력 표", "Power table") + _p3_table(pw) if pw else "")
          + (_fresh_blocks_line(fb) if fb else
             _n(f"손대지 않은 새 자료는 라이브 시작 전에는 세지 않습니다 "
                f"(필요 {ENSEMBLE_P3['fresh_blocks_min']}개, {ENSEMBLE_P3['fresh_block_months']}달력월 단위).",
                f"Untouched new data is not counted before the live start "
                f"({ENSEMBLE_P3['fresh_blocks_min']} blocks needed, {ENSEMBLE_P3['fresh_block_months']} "
                "calendar months each)."))
          + _h3("조정한 숫자를 남김없이 세기 (파라미터 회계 §14 — 이 표를 그대로)",
                "Counting every fitted number (§14 — this table as written)")
          + _p3_table(list(P3_PARAM_TABLE), ["줄", "객체", "K_s", "K_u", "생산 확률에?", "예산 줄"],
                      heads=dict(P3_PARAM_TABLE_COLS))
          + _n(P3_PARAM_TABLE_NOTE, P3_PARAM_TABLE_NOTE_EN)
          + f'<div class="note">{p3_param_bi(s)}</div>'
          + "</section>")

    # ② 사다리 확장
    lad = _ladder_records(s.get("ladder"))
    s2 = ('<section class="panel" id="s2">'
          + _h2("② 단계별 모델을 한 칸 더 올려 보기 (M1→H · M3→H · M3→ens · M3→M4a · B1→H)",
                "② Trying one more step up the model ladder (M1→H · M3→H · M3→ens · M3→M4a · B1→H)")
          + _n("두 모델이 빗나간 정도를 날마다 빼서 평균한 값입니다(×1e-4, 양수면 위 단계가 낫다는 뜻). "
               "구간은 40일씩 잘라 4,000번 다시 계산한 95% 범위이고, 여기에 DM t(HAC 19)와 위상을 어긋나게 한 "
               "부분표본까지 봤습니다. 어떤 후보도 잴 수 있을 만한 이득을 내지 못했습니다.",
               "For each day we subtract how far each model missed, then average (×1e-4; positive means the "
               "upper step is better). The 95% interval comes from cutting the record into 40-day blocks and "
               "recomputing 4,000 times, plus a DM t-test (HAC 19) and phase-shifted subsamples. No candidate "
               "produced a gain we could measure.")
          + (_p3_table(lad, _LADDER_ORDER) if lad else
             _empty_note(("단계별 비교 표가 없습니다", "no ladder rows", "summary_p3.ladder")))
          + "</section>")

    # ③ 블록 표
    blk = _sub(s, "blocks")
    b24 = _to_records(blk.get("24") or s.get("blocks24"))
    b18 = _to_records(blk.get("18") or s.get("blocks18"))
    s3 = ('<section class="panel" id="s3">'
          + _h2("③ 기간을 나눠 본 표 (24개월 · 18개월)", "③ Split by period (24 months · 18 months)")
          + _n("표본 수 옆에는 겹치지 않는 창의 수(n_blocks = n/20)를 항상 같이 적습니다 — 5,453 을 서로 다른 "
               "5,453개의 관측으로 읽으면 안 됩니다. 하루하루가 서로 겹쳐 있기 때문입니다.",
               "Next to every sample size we also print the number of independent windows (n_blocks = n/20) — "
               "5,453 must not be read as 5,453 separate observations, because the days overlap.")
          + _h3("24개월", "24 months")
          + (_p3_table(b24, _BLOCK_ORDER) if b24 else
             _empty_note(("24개월 표가 없습니다", "no 24-month blocks", "summary_p3.blocks.24")))
          + _h3("18개월", "18 months")
          + (_p3_table(b18, _BLOCK_ORDER) if b18 else
             _empty_note(("18개월 표가 없습니다", "no 18-month blocks", "summary_p3.blocks.18")))
          + "</section>")

    # ④ θ 경로·guard·체류
    th = _to_records(hmm.get("theta_table"))
    gl = _to_records(hmm.get("guard_log"))
    tm = hmm.get("timing") if isinstance(hmm.get("timing"), dict) else {}
    s4 = ('<section class="panel" id="s4">'
          + _h2("④ θ 가 해마다 어떻게 바뀌었나 · 안전장치 기록 · 한 분위기가 이어지는 기간",
                "④ How θ moved year by year · guard log · how long a mood lasts")
          + _n("해마다 1월 첫 거래일에 한 번만 다시 맞추고, 겹치는 20세션은 지우고, 지난해 값에서 이어서 "
               "시작합니다(같은 입력이면 언제나 같은 결과). 상태 1 은 수익률이 크게 흔들리는 쪽으로 이름을 "
               "고정했습니다. 한쪽으로 쏠린 이상한 결과(한 상태가 이어질 확률 0.999 이상이거나 점유가 1% 미만)는 "
               "안전장치가 막고 지난해 값을 그대로 씁니다.",
               "We refit once a year, on the first trading day of January, drop the 20 overlapping sessions, "
               "and start from last year's values (same input, same result, always). State 1 is pinned to the "
               "side with the bigger swings. Degenerate fits (a state persisting with probability 0.999+, or "
               "occupying under 1% of the time) are blocked by a guard and last year's values are kept.")
          + _p3_table(th, ["refit_date", "theta_id", "p00", "p11", "occupancy_high", "n_iter", "guard_ok",
                           "train_start", "train_end"],
                      ("θ 기록이 없습니다", "no θ rows", "summary_p3.hmm.theta_table"))
          + (_h3("안전장치 기록 (guard)", "Guard log") + _p3_table(gl) if gl
             else _n("안전장치가 걸린 적이 없습니다 (guard 로그 없음).", "The guard never fired (no guard log)."))
          + (_n_html(f'계산 시간: {_esc(", ".join(f"{k} {_fmt_p3(v, k)}" for k, v in tm.items()))}',
                     f'compute time: {_esc(", ".join(f"{k} {_fmt_p3(v, k)}" for k, v in tm.items()))}')
             if tm else "")
          + "</section>")

    # ⑤ P_high 밴드
    s5 = ('<section class="panel" id="s5">'
          + _h2("⑤ P(고변동) 밴드와 모형별 확률", "⑤ The P(high-swing) band and each model's probability")
          + '<div class="note">' + bi(P3_SHADOW_LABEL, P3_SHADOW_LABEL_EN) + " · "
          + bi(f"{P3_HIGH_STATE_KO}로 지낸 시간이 약 43% 라는 것은 '위기' 가 아니라 "
               f"'{P3_HIGH_STATE_GLOSS_KO}({P3_HIGH_STATE_KO})' 라는 뜻입니다(§16 9). 한 분위기가 이어질 확률이 "
               "0.98 이라 급락을 늦게 알아챕니다.",
               "Spending about 43% of the time in the high-swing state does not mean 'crisis' — it means the "
               "market was swinging a lot (§16 9). Because a mood persists with probability 0.98, the model "
               "notices a crash late.")
          + "</div>"
          + _img_bi(charts.pop("hmm_phigh", None),
                    "그날까지의 자료만 써서 계산한 '많이 출렁일 확률' 과 해마다 다시 맞춘 지점 — 나중 자료를 "
                    "끌어다 쓴 매끈한 값이 아닙니다.",
                    "The chance of a turbulent day, computed only from data up to that day, with the yearly "
                    "refit marks — not a smoothed line that peeks at later data.",
                    "HMM filtered P(high) path")
          + _img_bi(charts.pop("members_band", None),
                    "모형별 확률과 서로 갈리는 폭, SPY 와 함께 — 이 폭은 보여 주기만 합니다.",
                    "Each model's probability and the spread between them against SPY — the band is shown "
                    "for information only.",
                    "Member probabilities and disagreement band")
          + "</section>")

    # ⑥ 시대별 AUC
    era = _to_records(s.get("era_auc"))
    s6 = ('<section class="panel" id="s6">'
          + _h2("⑥ 시기별로 얼마나 잘 갈라냈나 (AUC — P_high vs p_vix vs M3)",
                "⑥ How well each signal ranked risky days first, by era (AUC)")
          + _n("AUC 0.697 이 x_vix 보다 높아 보여도, 이건 겹치지 않는 창 약 270개에서 나온 숫자 하나이고 "
               "결국 같은 정보(실제로 움직인 폭이 이어진다는 사실)를 보고 있습니다 — 정확도로 따지면 이득이 "
               "0 입니다.",
               "An AUC of 0.697 may look higher than x_vix, but it is one number from about 270 independent "
               "windows and it reads the same information (recent swings persist) — in accuracy terms the "
               "gain is zero.")
          + (_p3_table(era, _ERA_ORDER) if era else
             _empty_note(("시기별 표가 없습니다", "no era rows", "summary_p3.era_auc")))
          + _img_bi(charts.pop("era_auc", None),
                    "시기별 판별력 — 2020~24 에 VIX 의 힘이 약해진 자리를 HMM 이 조금 메웁니다.",
                    "Ranking power by era — as VIX weakened in 2020-24 the HMM filled part of the gap.",
                    "Era AUC bars")
          + "</section>")

    # ⑦ 소거
    abl = s.get("ablations")
    s7 = ('<section class="panel" id="s7">'
          + _h2("⑦ 부품을 하나씩 빼 보기 (소거 — q20 Platt · A 관측형 · B 동결형 · 완전/대각 공분산)",
                "⑦ Removing one part at a time (ablations)")
          + _n("모형에서 한 조각씩 빼거나 바꿔 보고 성적이 얼마나 달라지는지 봅니다. 성적이 좋은 쪽을 고르려는 "
               "것이 아니라, 결과가 그 조각 하나에 매달려 있지는 않은지 확인하려는 것입니다.",
               "We drop or swap one piece at a time and see how much the score moves. The point is not to "
               "pick the best variant but to check that the result does not hang on any single piece.")
          + (_p3_table(_to_records(abl), None) if _to_records(abl) else
             (_p3_kv(abl) if abl else
              _empty_note(("부품을 뺀 결과가 없습니다", "no ablation rows", "summary_p3.ablations"))))
          + "</section>")

    # ⑧ 신뢰도
    rel_h = _to_records(_sub(s, "reliability").get("H") or s.get("reliability_h"))
    rel_e = _to_records(_sub(s, "reliability").get("ens") or s.get("reliability_ens"))
    s8 = ('<section class="panel" id="s8">'
          + _h2("⑧ 말한 확률이 실제와 맞았나 (신뢰도 — 후보 H · 여러 모형의 평균)",
                "⑧ Did the stated % match reality? (candidate H and the average of the models)")
          + _n("\"20%\" 라고 말한 날들을 모아 보면 실제로 100번 중 20번쯤 일어났는지 확인하는 표입니다. "
               "칸마다 표본 수와 구간을 함께 싣습니다.",
               "We gather the days where the model said \"20%\" and check whether it actually happened about "
               "20 times in 100. Every cell carries its sample size and interval.")
          + _h3("후보 H", "candidate H")
          + (_p3_table(rel_h, _RELIAB_ORDER) if rel_h else
             _empty_note(("후보 H 의 표가 없습니다", "no rows for candidate H", "summary_p3.reliability.H")))
          + _h3("여러 모형의 평균", "average of the models")
          + (_p3_table(rel_e, _RELIAB_ORDER) if rel_e else
             _empty_note(("평균 모형의 표가 없습니다", "no rows for the averaged model",
                          "summary_p3.reliability.ens")))
          + "</section>")

    # ⑨ 결정론·PIT·카나리·spec
    det = s.get("determinism") if isinstance(s.get("determinism"), dict) else {}
    st = s.get("selftest") if isinstance(s.get("selftest"), dict) else {}
    spec = {"registry_sha": run.get("registry_sha"), "sizing_sha": run.get("sizing_sha"),
            "spec_sha256": run.get("spec_sha256"), "obs_spec": run.get("obs_spec"),
            "python": run.get("python"), "numpy": run.get("numpy"), "pandas": run.get("pandas"),
            "sklearn": run.get("sklearn"), "end": run.get("end"), "disclosure": s.get("disclosure")}
    s9 = ('<section class="panel" id="s9">'
          + _h2("⑨ 결정론 · PIT · 누수 카나리 · spec",
                "⑨ Determinism · PIT · leak canary · spec")
          + _n("같은 자료로 두 번 계산하면 결과가 비트 단위로 똑같아야 합니다. 자료를 뒤에서 잘라 내도 앞부분 "
               "값은 그대로여야 합니다 — 그래야 미래를 훔쳐보지 않았다는 뜻입니다(PIT). 미래를 본 값(smoothed)이 "
               "그날까지만 본 값(filter)보다 잘 맞아야 정상이며, 그렇지 않으면 어딘가 새고 있다는 신호입니다.",
               "Computing twice on the same data must give bit-identical results. Cutting the data short must "
               "leave the earlier values unchanged — that is what proves nothing peeked ahead (PIT). The "
               "version that sees the future (smoothed) should score better than the one that stops at each "
               "day (filter); if it does not, something is leaking.")
          + _p3_kv(det, ("결정론 기록이 없습니다", "no determinism record", "summary_p3.determinism"))
          + (_h3("자기검사", "self-test") + _p3_kv(st) if st else "")
          + _h3("spec", "spec") + _p3_kv(spec)
          + (_h3("경고", "warnings") + _warns_bi(warns) if warns else "")
          + "</section>")

    body = (head + nav + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9
            + _p3_honesty_section("s10", kill_power_text=kill_power_bi(s)))
    _write_html(out_html, "Market-mood engine and candidate list (Phase 3) - market-risk-lab", body)


# ------------------------------------------------------------------
# 주간 리포트 — 트랙레코드 (docs/track_record.html)  §8.4 지평 규칙 × §8.5 패널
# ------------------------------------------------------------------
_TRACK_SECTIONS = [("① 지평 규칙", "① what we may say yet"),
                   ("② 확률·상태", "② probability & state"),
                   ("③ 전환·주식 비중", "③ switches & stock share"),
                   ("④ 변동성·밴드", "④ swings & bands"),
                   ("⑤ 성적", "⑤ score"),
                   ("⑥ 규칙 vs 보유", "⑥ rule vs holding"),
                   ("⑦ 끄는 규칙까지", "⑦ countdown to switch-off"),
                   ("⑧ 경보·D10", "⑧ alarms & D10"),
                   ("⑨ 후보 명단", "⑨ candidate list"),
                   ("⑩ 정직 문구", "⑩ honesty notes")]


_TRACK_PANEL_TITLES = {
    "i_p_level": "(i) 평균 확률 · 확률이 0.30 을 넘은 날의 비율",
    "ii_state_occupancy": "(ii) 각 상태로 지낸 시간",
    "iii_switching": "(iii) 판정이 바뀐 횟수 · 주식 비중을 바꾼 횟수",
    "iv_exposure": "(iv) 주식 비중 분포",
    "v_vol_forecast": "(v) 얼마나 출렁일지 맞혔나",
    "vi_coverage": "(vi) 말한 범위 안에 들어온 비율",
    "vii_skill": "(vii) 정확도 점수 + 같은 길이의 과거 창들과 비교",
    "viii_relative": "(viii) 규칙 vs 그냥 보유",
    "ix_kill": "(ix) 끄는 규칙까지 남은 조건 · 중간 점검",
    "x_alarms": "(x) 경보 · D10 불일치",
    "xi_registry": "(xi) 후보 모형 명단 — 모형별 라이브 기록 · 갈리는 폭 · 새 자료 블록",
}


_TRACK_PANEL_TITLES_EN = {
    "i_p_level": "(i) average probability · share of days above 0.30",
    "ii_state_occupancy": "(ii) time spent in each state",
    "iii_switching": "(iii) decision switches · changes to the stock share",
    "iv_exposure": "(iv) distribution of the stock share",
    "v_vol_forecast": "(v) how well we called the swings",
    "vi_coverage": "(vi) how often reality landed inside our range",
    "vii_skill": "(vii) accuracy score, next to past windows of the same length",
    "viii_relative": "(viii) rule vs simply holding",
    "ix_kill": "(ix) countdown to the switch-off rule · mid-term checks",
    "x_alarms": "(x) alarms · D10 mismatches",
    "xi_registry": "(xi) candidate models — live record, spread, fresh blocks",
}


def _panel_title(key) -> str:
    """패널 제목 한/영 한 벌."""
    k = str(key)
    return bi(_TRACK_PANEL_TITLES.get(k, k), _TRACK_PANEL_TITLES_EN.get(k, k))


def render_track_record(track_p3: dict, summary_p3: dict, out_html: Path, charts: dict[str, bytes]) -> None:
    """docs/track_record.html — §8.4 지평 규칙에 따른 §8.5 패널 (i)~(xi) + 경보 이력 + 킬 기록.

    지평 규칙은 mrl.track 이 정하고(horizon_stage/may_show), 이 함수는 그 판정을 통과한 것만 렌더링하며
    막힌 항목은 **사유와 함께** 회색으로 남긴다(조용히 지우지 않는다)."""
    if not isinstance(track_p3, dict):
        raise TypeError("track_p3 는 dict 여야 합니다")
    t = track_p3
    s = summary_p3 if isinstance(summary_p3, dict) else {}
    charts = dict(charts or {})
    acc = p2_acceptance_of(t, s)
    eff = p3_effective_mode({**_p3_mode_source(s, acc),
                             "p2_deploy_mode": _first(t, "p2_deploy_mode") or _first(acc, "deploy_mode"),
                             "p3_deploy_mode": _first(t, "p3_deploy_mode") or _first(t, "effective_mode")
                                               or _p3_mode_source(s, acc)["p3_deploy_mode"],
                             "kill": _sub(t, "kill"), "deploy_sizing": t.get("deploy_sizing")})
    hz = horizon_view(t)
    panel = _sub(t, "panel")
    kill = _sub(t, "kill")
    notes = [str(x) for x in (t.get("notes") or [])]

    live_start = str(t.get("live_start") or "")
    tags = ['<span class="tag">' + bi("지평 단계", "horizon stage") + f' <b>{_esc(_hz_stage(hz))}</b></span>',
            '<span class="tag">' + bi("라이브 시작", "live since")
            + (f" <b>{_esc(live_start)}</b></span>" if live_start
               else " <b>" + bi("아직 시작 전", "not started yet") + "</b></span>"),
            '<span class="tag">' + bi_html(f'채점한 날 <b>{_fmt(t.get("n_scored"), "n")}일 (겹치지 않는 창 '
                                           f'{_fmt_p3(t.get("n_eff"), "n_eff")}개)</b>',
                                           f'<b>{_fmt(t.get("n_scored"), "n")}</b> scored days '
                                           f'(<b>{_fmt_p3(t.get("n_eff"), "n_eff")}</b> independent windows)')
            + "</span>",
            '<span class="tag">' + bi_html(f'시작 후 <b>{_fmt(t.get("months_elapsed"), "n")}개월</b>',
                                           f'<b>{_fmt(t.get("months_elapsed"), "n")}</b> months in') + "</span>"]
    if t.get("holdout_locked"):
        tags.append('<span class="tag">' + bi("손대지 않고 잠가 둔 구간", "locked, untouched data")
                    + f' <b>{_esc(HOLDOUT_START)}~</b></span>')
    al = [a for a in (t.get("alarms") or []) if isinstance(a, dict)]
    if al:
        tags.append(_tag_html(f"경보 {len(al)}건", f"{len(al)} alarms", cls=" warn"))
    head = _p3_head(("market-risk-lab · Phase 3 · 성적 기록 · 달라짐 경고 · 끄는 규칙",
                     "market-risk-lab · Phase 3 · live record · drift warnings · switch-off rule"),
                    ("실제로 기록한 것과 과거 시험 결과를 나란히 — 이건 일기이지 판결이 아닙니다",
                     "The live record next to the backtest — this is a diary, not a verdict"),
                    ("라이브 숫자는 전부 장부에서만 다시 계산합니다(새로 돌린 백테스트에서 가져오지 않습니다). "
                     "판정하는 날은 36개월이 지난 때(1차), 그리고 하락 사건 8회 또는 60개월 중 먼저 오는 때"
                     "(2차)입니다. 그 전에는 잘한다·못한다 같은 판정하는 말을 쓰지 않습니다.",
                     "Every live number is recomputed from the ledger only — never taken from a fresh "
                     "backtest. The judgement dates are 36 months (first) and 8 decline episodes or 60 "
                     "months, whichever comes first (second). Before then we use no judging words."),
                    tags, acc, eff, t.get("deploy_sizing"))
    nav = _p3_nav(_TRACK_SECTIONS)

    # ① 지평 규칙 (§8.4) — 이 페이지의 나머지가 따르는 규칙
    s1 = ('<section class="panel" id="s1">'
          + _h2("① 지평별 표시 규칙 (§8.4 — 코드가 강제합니다)",
                "① What we are allowed to say yet (§8.4 — enforced by code)")
          + _hz_block(hz)
          + _n("겹치지 않는 창이 6개보다 적으면(n_eff &lt; 6) 정확도 숫자를 회색으로도 내보내지 않고, 판정일"
               "(36개월) 전에는 판정하는 말을 쓰지 않습니다. 이 규칙의 단일 원천은 mrl/track.py 입니다.",
               "With fewer than six independent windows (n_eff &lt; 6) no accuracy number is printed, not "
               "even in grey, and before the judgement date (36 months) no judging word is used. The single "
               "source for this rule is mrl/track.py.")
          + "</section>")

    def _sec(sec_id: str, title_ko: str, title_en: str, keys: list[str], extra: str = "") -> str:
        body = ""
        for k in keys:
            blk = panel.get(k)
            body += f"<h3>{_panel_title(k)}</h3>"
            body += (_panel_kv(_hz_redact(blk, hz)) if isinstance(blk, dict)
                     else _n("이 칸의 자료가 아직 없습니다.", "no data for this panel yet"))
        return (f'<section class="panel" id="{_esc(sec_id)}">' + _h2(title_ko, title_en)
                + body + extra + "</section>")

    s2 = _sec("s2", "② 확률은 어느 수준이었고, 어떤 상태로 얼마나 지냈나",
              "② What the probability looked like, and how long each state lasted",
              ["i_p_level", "ii_state_occupancy"])
    s3 = _sec("s3", "③ 판정이 바뀐 횟수 · 주식 비중을 바꾼 횟수 · 주식 비중 분포",
              "③ Decision switches, changes to the stock share, and its distribution",
              ["iii_switching", "iv_exposure"])
    # §7 A·B 표를 실제로 렌더한다 — 회색·풀링 규칙이 코드로 강제된다는 주장을 표에서 확인할 수 있어야 한다
    scen = _sub(s, "scenarios")
    bin_rows = _to_records(scen.get("bins"))
    state_rows = _to_records(scen.get("states"))
    grey_rows = [str(r.get("bin") or r.get("state") or "?") for r in (bin_rows + state_rows) if r.get("grey")]
    s4 = _sec("s4", "④ 얼마나 출렁일지 맞혔나, 말한 범위 안에 들어왔나",
              "④ Did we call the swings, and did reality land inside our range?",
              ["v_vol_forecast", "vi_coverage"],
              _img_bi(charts.pop("bins_live", None),
                      "확률대별로 실제로 몇 번 일어났는지 — 과거 시험 결과와 라이브를 나란히(구간은 Wilson, "
                      "겹치지 않는 창 수 기준).",
                      "How often it actually happened in each probability band — backtest next to live "
                      "(Wilson intervals on independent windows).",
                      "Bins: backtest vs live")
              + _h3("확률대별 표 (§7 A — 전체; 라이브 열은 라이브 시작 뒤부터 찹니다)",
                    "Probability-band table (§7 A — the live column fills in after the live start)")
              + _p3_table(bin_rows, _P3_BIN_ORDER,
                          ("확률대 표가 없습니다", "no probability bands", "summary_p3.scenarios.bins"))
              + _h3("판정 상태별 표 (§7 B — 상태는 종류라서 위 칸과 합치지 않습니다; 표본이 얇은 칸은 회색)",
                    "By decision state (§7 B — states are categories, never merged; thin cells are greyed)")
              + _p3_table(state_rows, _P3_STATE_ORDER,
                          ("상태별 표가 없습니다", "no state rows", "summary_p3.scenarios.states"))
              + ('<div class="tag bad">'
                 + bi_html(f'회색(단독 표시 금지): {_esc(", ".join(grey_rows))} — 겹치지 않는 창이 '
                           f'{SCENARIO_P3["min_n_eff"]:.0f}개도 안 됩니다',
                           f'grey, never shown alone: {_esc(", ".join(grey_rows))} — fewer than '
                           f'{SCENARIO_P3["min_n_eff"]:.0f} independent windows')
                 + "</div>" if grey_rows else "")
              + _n_html(f"이 표는 표본 수 n 과 겹치지 않는 창 수(n_eff = n/{SCENARIO_P3['n_eff_div']}), 그리고 "
                        "구간 없이는 절대 표시하지 않습니다. 창이 "
                        f"{SCENARIO_P3['min_n_eff']:.0f}개보다 적은 칸은 위 칸과 합칩니다(코드가 강제). "
                        "라이브 비율이 과거 구간 밖으로 나가면 '말한 확률이 어긋나기 시작했다' 로 표시합니다"
                        "(D8 과는 별개입니다).",
                        f"This table is never shown without the sample size n, the number of independent "
                        f"windows (n_eff = n/{SCENARIO_P3['n_eff_div']}) and an interval. Cells with fewer "
                        f"than {SCENARIO_P3['min_n_eff']:.0f} windows are merged upward — enforced by code. "
                        "If the live rate falls outside the backtest interval we flag it as the stated "
                        "percentages drifting (separate from D8)."))

    # ⑤ skill — 지평이 막으면 숫자 대신 사유
    vii = _hz_redact(_sub(panel, "vii_skill"), hz)
    skill_head = f"<h3>{_panel_title('vii_skill')}</h3>"
    if horizon_may_show(hz, "bss"):
        skill_body = skill_head + _panel_kv(vii) + _img_bi(
            charts.pop("bss_vs_hist", None),
            "라이브 성적 한 점과, 같은 길이의 과거 창들이 만든 분포 — 그 점이 분포 안 어디쯤인지 봅니다.",
            "One live score against the spread of past windows of the same length — we look at where the "
            "single point sits.",
            "Live BSS vs backtest window distribution")
    else:
        skill_body = (skill_head
                      + '<div class="note">' + bi(_hz_reason(hz, "bss"), _hz_reason_en(hz, "bss")) + "</div>"
                      + _n("지금 말할 수 있는 것은 파이프라인이 매일 돌고 기록이 제대로 쌓인다는 것뿐입니다 — "
                           "계수·경보·주식 비중이 지나온 길·하락 사건 수·범위 적중·변동성 오차는 위 칸에 "
                           "있습니다.",
                           "All we can say yet is that the pipeline runs daily and the record is piling up "
                           "correctly — the coefficients, alarms, the path of the stock share, the episode "
                           "count, the range hit rate and the volatility error are in the sections above."))
    s5 = ('<section class="panel" id="s5">'
          + _h2("⑤ 정확도 점수 (Brier · 기준선 대비)", "⑤ Accuracy score (Brier, and versus a naive guess)")
          + skill_body + "</section>")

    s6 = _sec("s6", "⑥ 규칙 vs 그냥 보유 (같은 길이의 과거 분포 옆에)",
              "⑥ Rule versus simply holding (next to past windows of the same length)",
              ["viii_relative"])

    # ⑦ 킬 카운트다운 — 판정어는 due 전에 없다
    ix = _hz_redact(_sub(panel, "ix_kill"), hz)
    kill_rows = {k: kill.get(k) for k in ("live_start", "months", "episodes5", "stage1_due", "stage2_due",
                                          "evaluated_now", "killed", "stage1_done", "stage2_done",
                                          "last_eval_month", "gross_failure_badge") if k in kill}
    verdict_txt = _hz_val(hz, "verdict", _esc(str(kill.get("state") or "—")))
    s7 = ('<section class="panel" id="s7">'
          + _h2("⑦ 성적이 나쁘면 끄는 규칙까지 · 중간 점검",
                "⑦ Countdown to the switch-off rule · mid-term checks")
          + f'<div class="v-act">{bi("판정", "verdict")}: {verdict_txt}</div>'
          + f"<h3>{_panel_title('ix_kill')}</h3>" + _panel_kv(ix)
          + (_h3("끄는 규칙 기록", "switch-off record") + _p3_kv(kill_rows) if kill_rows else "")
          + ('<div class="note">' + bi("−5% 를 지난 날", "days it passed −5%") + ": "
             + _esc(", ".join(str(x) for x in (kill.get("breach_dates") or []))) + "</div>"
             if kill.get("breach_dates") else "")
          + _img_bi(charts.pop("kill_countdown", None),
                    "끄는 규칙까지 남은 조건 — 하락 사건 X/8 · 시작 후 Y/36개월(상한 60).",
                    "Countdown to the switch-off rule — X/8 decline episodes · Y/36 months in (60-month cap).",
                    "Kill rule countdown")
          + _n("12개월·24개월 중간 점검은 숫자를 보여 줄 뿐, 그때는 끌 수 없습니다(§7 이 3년을 하한으로 "
               "정해 두었습니다). 한 번 끄면 그대로 유지되고, 되돌리려면 새 장부 항목 + 끈 뒤 12개월 이상 "
               "경과 + 전체 라이브 구간의 95% 하한이 0 보다 커야 합니다. 자동으로 되살아나지 않습니다.",
               "The 12- and 24-month checks only show numbers; they cannot switch anything off (§7 sets a "
               "three-year floor). Once switched off it stays off, and coming back requires a new ledger "
               "entry, at least 12 months since the switch-off, and the lower end of the 95% interval over "
               "the whole live window above 0. Nothing comes back automatically.")
          + f'<div class="note">{kill_power_bi(s)}</div>'
          + "</section>")

    # ⑧ 경보·D10
    s8 = ('<section class="panel" id="s8">'
          + _h2("⑧ 예전과 달라졌다는 경고 (D1~D11) · D10 재현 불일치",
                "⑧ Warnings that things changed (D1-D11) · D10 replay mismatches")
          + f"<h3>{_panel_title('x_alarms')}</h3>"
          + _p3_table(al, _P3_ALARM_ORDER, ("발화한 경보가 없습니다", "no alarms fired"))
          + _n_html(f'D10 재현 불일치 {_fmt(t.get("replay_mismatch_count"), "n")}건 · 최근 20세션에서 확률이 '
                    "0.01 넘게 어긋나면(D7) 그날 작업은 아무것도 커밋하지 않고 멈춥니다.",
                    f'D10 replay mismatches: {_fmt(t.get("replay_mismatch_count"), "n")} · if the probability '
                    "moves by more than 0.01 over the last 20 sessions (D7) the daily run stops and commits "
                    "nothing.")
          + _img_bi(charts.pop("alarms_timeline", None),
                    "경보가 언제 울렸는지 — 코드별로 찍었습니다.",
                    "When each alarm fired, by code.",
                    "Alarm timeline")
          + _h3("미리 등록해 둔 한계선 (§8.2 · P3_DRIFT — 바꾸려면 번호 붙인 장부 항목이 필요합니다)",
                "The thresholds registered in advance (§8.2 — changing one needs a numbered ledger entry)")
          + _n("이 한계선들은 2003~24년 기록의 5분위·95분위에서 가져왔습니다. 그 기록에 대해서는 결과를 본 뒤 "
               "정한 것(post hoc)입니다. 라이브 자료를 보고 다시 조정하지 않으며, 경보는 표시하고 기록할 뿐 "
               "스스로 무언가를 바꾸지 않습니다.",
               "These thresholds come from the 5th and 95th percentiles of the 2003-24 record, so with "
               "respect to that record they are post hoc. We never re-tune them on live data, and an alarm "
               "only flags and records — it changes nothing by itself.")
          + _p3_table([{"code": c, "threshold": P3_DRIFT.get(c), "action": _track.ALARM_ACTIONS.get(c)}
                       for c in _track.ALARM_CODES], ["code", "threshold", "action"])
          + (_h3("기록", "notes") + _warns_bi(notes) if notes else "")
          + "</section>")

    # ⑨ 등록부
    xi = _sub(panel, "xi_registry")
    mem_rows = [_hz_redact({"member": k, **(v if isinstance(v, dict) else {})}, hz)
                for k, v in (_sub(xi, "members") or _sub(t, "members")).items()]
    for r in mem_rows:                                     # 막힌 열은 표에서 아예 뺀다(사유는 아래 줄에)
        for k in [kk for kk, vv in list(r.items()) if vv is None and _hz_key(kk)]:
            r.pop(k, None)
        r.pop("hidden", None)
        r.pop("hidden_reason", None)
    fresh = xi.get("fresh_blocks") or t.get("fresh_blocks")
    s9 = ('<section class="panel" id="s9">'
          + _h2("⑨ 후보 모형 명단 — 모형별 라이브 기록 · 갈리는 폭 · 새 자료 블록",
                "⑨ Candidate models — live record, how far they disagree, fresh data blocks")
          + f"<h3>{_panel_title('xi_registry')}</h3>"
          + _p3_table(mem_rows, ["member", "live_start", "n", "n_eff", "brier", "bss_clim", "verdict"],
                      ("모형별 라이브 기록이 아직 없습니다", "no live record per model yet"))
          + ('<div class="note">' + bi(_hz_reason(hz, "bss"), _hz_reason_en(hz, "bss")) + "</div>"
             if not horizon_may_show(hz, "bss") else "")
          + _h3("모형들이 갈리는 폭", "How far the models disagree")
          + _p3_kv(_sub(xi, "disagreement"), ("갈리는 폭 기록이 없습니다", "no spread recorded"))
          + '<div class="note">'
          + bi(f"손대지 않은 새 자료: 필요한 만큼은 {ENSEMBLE_P3['fresh_blocks_min']}개입니다",
               f"Untouched new data: {ENSEMBLE_P3['fresh_blocks_min']} blocks are required")
          + " — " + (_raw(str(fresh)) if fresh else bi("라이브 시작 전", "before the live start"))
          + "</div>"
          + "</section>")

    body = (head + nav + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9
            + _p3_honesty_section("s10", kill_power_text=kill_power_bi(s)))
    _write_html(out_html, "Live track record (Phase 3) - market-risk-lab", body)


# ------------------------------------------------------------------
# Phase 3 차트 (matplotlib Agg → PNG bytes) — §10 charts_p3
# ------------------------------------------------------------------
def _p3_frame(df) -> pd.DataFrame | None:
    """DatetimeIndex 로 정규화된 프레임(또는 None). 'date' 열이 있으면 인덱스로 올린다."""
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return None
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        for c in ("date", "asof", "Date"):
            if c in out.columns:
                out = out.set_index(c)
                break
    try:
        idx = pd.DatetimeIndex(pd.to_datetime(out.index))
    except (TypeError, ValueError):
        return None
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    return out[~out.index.duplicated(keep="last")].sort_index()


def _p3_col(df: pd.DataFrame | None, *names) -> pd.Series | None:
    if df is None:
        return None
    for n in names:
        if n in df.columns:
            s = pd.to_numeric(df[n], errors="coerce")
            if s.notna().any():
                return s
    return None


def _cum(ret: pd.Series) -> pd.Series:
    return (1.0 + ret.fillna(0.0)).cumprod()


def _dd_of(cum: pd.Series) -> pd.Series:
    return cum / cum.cummax() - 1.0


def charts_p3(oos_p3, backtest_p3, spy_close, summary_p3, ledger_df) -> dict[str, bytes]:
    """Phase 3 차트 → PNG bytes. 열이 없으면 그 차트만 경고 후 생략한다(예외로 죽지 않는다).

    반환 키: w_path · cum_returns · drawdown · ladder_frontier · members_band · hmm_phigh · era_auc ·
    bins_live · bss_vs_hist · alarms_timeline · kill_countdown. 차트 안 글자는 ASCII 만(CI 폰트 대비).
    """
    out: dict[str, bytes] = {}
    s = summary_p3 if isinstance(summary_p3, dict) else {}
    bt = _p3_frame(backtest_p3)
    oos = _p3_frame(oos_p3)
    led = _p3_frame(ledger_df)
    spy = _series_close(spy_close)

    # 1) 비중 경로 + SPY
    w = _p3_col(bt, "w_exec", "p3_w_exec")
    if w is None and led is not None:
        w = _p3_col(led, "p3_w_exec")
    if w is not None:
        try:
            fig, ax = _new_fig(9.0, 3.6)
            ww = w.dropna()
            ax.plot(ww.index, ww.values, color="#60a5fa", lw=1.2, label="w_exec")
            wv = _p3_col(bt, "w_vol")
            if wv is not None:
                wv = wv.dropna()
                ax.plot(wv.index, wv.values, color=PALETTE["dim"], lw=0.8, alpha=0.8, label="w_vol (vol only)")
            ax.axhline(float(P3["w_min"]), color="#ef4444", lw=0.8, ls="--", label=f"floor {P3['w_min']:.2f}")
            ax.axhline(float(P3["w_max"]), color=PALETTE["line"], lw=0.8, ls="--")
            ax.set_ylim(0.0, 1.08)
            ax.set_ylabel("exposure")
            if spy is not None:
                common = ww.index.intersection(spy.index)
                if len(common) >= 3:
                    ax2 = ax.twinx()
                    ax2.plot(common, spy.reindex(common).values, color=PALETTE["mut"], lw=0.8, alpha=0.7)
                    ax2.set_yscale("log")
                    _plain_log_axis(ax2)
                    ax2.tick_params(colors=PALETTE["dim"], labelsize=7)
                    ax2.set_ylabel("SPY (log)", color=PALETTE["dim"], fontsize=8)
                    for sp in ax2.spines.values():
                        sp.set_color(PALETTE["line"])
            ax.set_title("Exposure path (w_exec) vs SPY")
            _legend(ax, loc="lower left")
            out["w_path"] = _png(fig)
        except Exception as e:
            warnings.warn(f"w_path 차트 실패: {type(e).__name__}: {e}")
    else:
        warnings.warn("w_path: w_exec 열 없음 → 생략")

    # 2) 누적수익 · 3) 낙폭
    rr = _p3_col(bt, "ret_rule")
    rb = _p3_col(bt, "ret_bh")
    if rr is not None and rb is not None:
        try:
            c_rule, c_bh = _cum(rr.dropna()), _cum(rb.dropna())
            fig, ax = _new_fig()
            ax.plot(c_bh.index, c_bh.values, color=PALETTE["mut"], lw=1.2, label="buy & hold")
            ax.plot(c_rule.index, c_rule.values, color="#22c55e", lw=1.4, label="p3 sizing rule")
            for col, color, lab in (("ret_p2", "#eab308", "p2 decision layer"), ("ret_v0", "#f472b6", "v0 tone")):
                extra = _p3_col(bt, col)
                if extra is not None:
                    ce = _cum(extra.dropna())
                    ax.plot(ce.index, ce.values, color=color, lw=1.0, label=lab)
            ax.set_yscale("log")
            _plain_log_axis(ax)
            ax.set_title("Cumulative growth of 1.0 (log): rule vs hold vs p2 vs v0")
            _legend(ax, loc="upper left")
            out["cum_returns"] = _png(fig)

            fig, ax = _new_fig(9.0, 3.4)
            dd_rule = _p3_col(bt, "dd_rule")
            dr = dd_rule.dropna() if dd_rule is not None else _dd_of(c_rule)
            ax.fill_between(_dd_of(c_bh).index, _dd_of(c_bh).values * 100.0, 0, color=PALETTE["mut"],
                            alpha=0.35, label="buy & hold")
            ax.plot(dr.index, dr.values * 100.0, color="#22c55e", lw=1.1, label="p3 sizing rule")
            ax.set_ylabel("drawdown (%)")
            ax.set_title("Drawdown: rule vs buy & hold")
            _legend(ax, loc="lower left")
            out["drawdown"] = _png(fig)
        except Exception as e:
            warnings.warn(f"cum_returns/drawdown 차트 실패: {type(e).__name__}: {e}")
    else:
        warnings.warn("cum_returns/drawdown: ret_rule·ret_bh 열 없음 → 생략")

    # 4) D_max 사다리 프론티어 (두 창)
    lad = _to_records(_sub(s, "sizing").get("ladder"))
    if lad:
        try:
            g = sorted((_fnum(r.get("sigma_target")) for r in lad if not math.isnan(_fnum(r.get("sigma_target")))))
            by = {_fnum(r.get("sigma_target")): r for r in lad}
            xs = [x * 100.0 for x in g]
            fig, ax = _new_fig(8.0, 4.0)
            for key, color, lab in (("maxdd_vol_only", "#60a5fa", "MaxDD 1993+ (vol only)"),
                                    ("maxdd_decision", "#22c55e", "MaxDD 2003+ (with decision layer)")):
                ys = [_fnum(by[x].get(key)) * 100.0 for x in g]
                if any(not math.isnan(y) for y in ys):
                    ax.plot(xs, ys, marker="o", color=color, lw=1.3, label=lab)
            ax.plot(xs, [-float(P3["k_slow"]) * x for x in xs], color="#ef4444", lw=1.0, ls="--",
                    label=f"budget line -{P3['k_slow']}x sigma_T")
            ax.plot(xs, [-float(P3["k_fast"]) * x for x in xs], color="#eab308", lw=1.0, ls=":",
                    label=f"optimistic -{P3['k_fast']}x sigma_T")
            ax.set_xlabel("sigma_T (%)")
            ax.set_ylabel("MaxDD (%)")
            ax.set_title("MaxDD vs sigma_T frontier (two windows) and budget lines")
            _legend(ax, loc="lower left")
            out["ladder_frontier"] = _png(fig)
        except Exception as e:
            warnings.warn(f"ladder_frontier 차트 실패: {type(e).__name__}: {e}")
    else:
        warnings.warn("ladder_frontier: sizing.ladder 없음 → 생략")

    # 5) 멤버 확률 + 불일치 밴드 vs SPY
    if oos is not None:
        try:
            lo, hi = _p3_col(oos, "lo", "p3_lo"), _p3_col(oos, "hi", "p3_hi")
            series = [(("p_p2", "prob_dd5_20", "p_m3"), "#e6eaf2", "p2 (production)"),
                      (("p_m1", "p2_p_m1"), "#60a5fa", "M1 (VIX matched to reality)"),
                      (("p_h", "p3_p_h"), "#f472b6", "H (HMM shadow)")]
            drawn = 0
            fig, ax = _new_fig()
            if lo is not None and hi is not None:
                idx = lo.index
                ax.fill_between(idx, lo.values * 100.0, hi.values * 100.0, color="#1e2a40", alpha=0.85,
                                label="disagreement band")
            for names, color, lab in series:
                col = _p3_col(oos, *names)
                if col is None:
                    continue
                cc = col.dropna()
                ax.plot(cc.index, cc.values * 100.0, color=color, lw=0.9, label=lab)
                drawn += 1
            if drawn:
                ax.set_ylabel("P(-5% in 20 sessions) (%)")
                ax.set_title("Member probabilities and disagreement band")
                _legend(ax, loc="upper left")
                out["members_band"] = _png(fig)
            else:
                warnings.warn("members_band: 멤버 확률 열 없음 → 생략")
        except Exception as e:
            warnings.warn(f"members_band 차트 실패: {type(e).__name__}: {e}")

        # 6) HMM P_high
        try:
            ph = _p3_col(oos, "p_hmm_high", "p3_hmm_p_high", "p20")
            if ph is None:
                warnings.warn("hmm_phigh: P_high 열 없음 → 생략")
            else:
                pp = ph.dropna()
                fig, ax = _new_fig(9.0, 3.4)
                ax.plot(pp.index, pp.values, color="#f472b6", lw=0.9, label="filtered P(high vol)")
                q = _p3_col(oos, "q20")
                if q is not None:
                    qq = q.dropna()
                    ax.plot(qq.index, qq.values, color="#60a5fa", lw=0.7, alpha=0.8, label="q20 (any high in 20)")
                if "refit_year" in oos.columns:
                    yrs = pd.to_numeric(oos["refit_year"], errors="coerce")
                    edges = yrs.ne(yrs.shift(1)) & yrs.notna()
                    for d0 in oos.index[edges.fillna(False)][1:]:
                        ax.axvline(d0, color=PALETTE["line"], lw=0.6, alpha=0.7)
                ax.set_ylim(0, 1.02)
                ax.set_title("HMM filtered P(high-vol state); vertical lines = January refits")
                _legend(ax, loc="upper left")
                out["hmm_phigh"] = _png(fig)
        except Exception as e:
            warnings.warn(f"hmm_phigh 차트 실패: {type(e).__name__}: {e}")
    else:
        warnings.warn("members_band/hmm_phigh: oos_p3 없음 → 생략")

    # 7) 시대별 AUC
    era = _to_records(s.get("era_auc"))
    if era:
        try:
            labs = [str(r.get("era") or r.get("start") or i) for i, r in enumerate(era)]
            keys = [("auc_p_hmm", "HMM P_high"), ("auc_x_vix", "x_vix"), ("auc_p_m3", "p_m3"), ("auc_p_h", "member H")]
            keys = [(k, lab) for k, lab in keys if any(not math.isnan(_fnum(r.get(k))) for r in era)]
            if not keys:
                warnings.warn("era_auc: AUC 열 없음 → 생략")
            else:
                x = np.arange(len(era), dtype=float)
                width = 0.8 / len(keys)
                fig, ax = _new_fig(9.0, 3.6)
                for j, (k, lab) in enumerate(keys):
                    ax.bar(x + j * width, [_fnum(r.get(k)) for r in era], width=width, label=lab)
                ax.axhline(0.5, color=PALETTE["mut"], lw=0.8, ls="--")
                ax.set_xticks(x + 0.4 - width / 2)
                ax.set_xticklabels([lab[:10] for lab in labs], rotation=30, ha="right", fontsize=7)
                ax.set_ylabel("AUC")
                ax.set_title("AUC by era (0.5 = no discrimination)")
                _legend(ax, loc="lower left")
                out["era_auc"] = _png(fig)
        except Exception as e:
            warnings.warn(f"era_auc 차트 실패: {type(e).__name__}: {e}")

    # 8) 구간 표 (백테스트 vs 라이브)
    bins = _to_records(_sub(s, "scenarios").get("bins"))
    if bins:
        try:
            labs = [str(r.get("bin") or f"{_fnum(r.get('bin_lo')):.2f}-{_fnum(r.get('bin_hi')):.2f}") for r in bins]
            obs = [_fnum(r.get("obs")) for r in bins]
            lo = [max(_fnum(r.get("obs")) - _fnum(r.get("wilson_lo")), 0.0) for r in bins]
            hi = [max(_fnum(r.get("wilson_hi")) - _fnum(r.get("obs")), 0.0) for r in bins]
            x = np.arange(len(bins), dtype=float)
            fig, ax = _new_fig(9.0, 3.8)
            ax.bar(x, obs, width=0.55, color="#60a5fa", label="backtest observed")
            ax.errorbar(x, obs, yerr=[lo, hi], fmt="none", ecolor=PALETTE["mut"], elinewidth=0.9, capsize=3)
            live = [_fnum(r.get("obs_live")) for r in bins]
            if any(not math.isnan(v) for v in live):
                ax.plot(x, live, "o", color="#22c55e", label="live observed")
            mp = [_fnum(r.get("mean_p")) for r in bins]
            ax.plot(x, mp, "s--", color=PALETTE["tx"], lw=0.9, ms=4, label="mean p (perfect calibration)")
            ax.set_xticks(x)
            ax.set_xticklabels(labs, rotation=30, ha="right", fontsize=7)
            ax.set_ylabel("P(-5% in 20 sessions)")
            ax.set_title("Probability bins: mean p vs observed (Wilson on n_eff = n/20)")
            _legend(ax, loc="upper left")
            out["bins_live"] = _png(fig)
        except Exception as e:
            warnings.warn(f"bins_live 차트 실패: {type(e).__name__}: {e}")
    else:
        warnings.warn("bins_live: scenarios.bins 없음 → 생략")

    # 9) 라이브 BSS vs 동일 길이 과거 분포
    roll = _sub(s, "reference").get("rolling_bss")
    live_bss = _fnum(_first(s, "live_bss", default=_sub(s, "track").get("bss_clim")))
    if isinstance(roll, dict) and roll:
        try:
            keys = [k for k in roll if isinstance(roll[k], (list, tuple)) and len(roll[k]) >= 5]
            if not keys:
                warnings.warn("bss_vs_hist: rolling_bss 표본 부족 → 생략")
            else:
                fig, ax = _new_fig(8.0, 3.6)
                for k in sorted(keys, key=lambda z: _fnum(z) if not math.isnan(_fnum(z)) else 0.0):
                    vals = np.asarray([_fnum(v) for v in roll[k]], dtype=float)
                    vals = vals[np.isfinite(vals)]
                    if vals.size:
                        ax.hist(vals, bins=24, histtype="step", lw=1.2, label=f"backtest windows L={k}")
                if not math.isnan(live_bss):
                    ax.axvline(live_bss, color="#22c55e", lw=1.6, label=f"live BSS {live_bss:+.3f}")
                ax.axvline(0.0, color=PALETTE["mut"], lw=0.8, ls="--")
                ax.set_xlabel("BSS vs climatology")
                ax.set_title("Live BSS against the backtest distribution of same-length windows")
                _legend(ax, loc="upper left")
                out["bss_vs_hist"] = _png(fig)
        except Exception as e:
            warnings.warn(f"bss_vs_hist 차트 실패: {type(e).__name__}: {e}")
    else:
        warnings.warn("bss_vs_hist: reference.rolling_bss 없음 → 생략")

    # 10) 경보 타임라인
    al = [a for a in (s.get("alarms") or _sub(s, "track").get("alarms") or []) if isinstance(a, dict)]
    if al:
        try:
            codes = sorted({str(a.get("code")) for a in al})
            ypos = {c: i for i, c in enumerate(codes)}
            xs, ys = [], []
            for a in al:
                d0 = pd.to_datetime(a.get("asof"), errors="coerce")
                if pd.isna(d0):
                    continue
                xs.append(d0)
                ys.append(ypos[str(a.get("code"))])
            if not xs:
                warnings.warn("alarms_timeline: asof 없는 경보뿐 → 생략")
            else:
                fig, ax = _new_fig(9.0, max(2.4, 0.35 * len(codes) + 1.2))
                ax.scatter(xs, ys, s=26, color="#eab308")
                ax.set_yticks(range(len(codes)))
                ax.set_yticklabels(codes, fontsize=8)
                ax.set_title("Drift alarm timeline (flags only; no automatic re-tuning)")
                out["alarms_timeline"] = _png(fig)
        except Exception as e:
            warnings.warn(f"alarms_timeline 차트 실패: {type(e).__name__}: {e}")
    else:
        warnings.warn("alarms_timeline: 경보 기록 없음 → 생략")

    # 11) 킬 카운트다운
    kill = _sub(s, "kill") or _sub(s, "track", "kill")
    if kill:
        try:
            ep = _fnum(kill.get("episodes5"))
            mo = _fnum(kill.get("months"))
            need_ep, need_mo, cap_mo = (float(KILL_P3["min_episodes"]), float(KILL_P3["min_months"]),
                                        float(KILL_P3["max_months"]))
            fig, ax = _new_fig(7.5, 2.4)
            labels = ["episodes >=5%", "months elapsed"]
            done = [0.0 if math.isnan(ep) else ep, 0.0 if math.isnan(mo) else mo]
            need = [need_ep, need_mo]
            y = np.arange(2, dtype=float)
            ax.barh(y, need, color=PALETTE["line"], height=0.5, label="required (stage 1)")
            ax.barh(y, done, color="#22c55e", height=0.5, label="recorded")
            ax.axvline(cap_mo, color="#eab308", lw=1.0, ls="--", label=f"stage-2 cap {cap_mo:.0f} months")
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=8)
            for i, (dv, nv) in enumerate(zip(done, need)):
                ax.text(max(dv, 0.2), i, f" {dv:.0f}/{nv:.0f}", va="center", color=PALETTE["tx"], fontsize=8)
            ax.set_title("Kill-rule countdown (stage 1 = 36 months; stage 2 = 8 episodes or 60 months)")
            _legend(ax, loc="lower right")
            out["kill_countdown"] = _png(fig)
        except Exception as e:
            warnings.warn(f"kill_countdown 차트 실패: {type(e).__name__}: {e}")
    else:
        warnings.warn("kill_countdown: kill 상태 없음 → 생략")
    return out
