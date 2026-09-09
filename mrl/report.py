# -*- coding: utf-8 -*-
"""리포트 파사드 — 페이지별 모듈을 하나의 이름 공간으로 다시 내보낸다.

예전에는 이 파일 하나에 4,273 줄이 들어 있었다. 페이지마다 따로 작업하려면 한 파일을
동시에 고쳐야 해서, 실제 코드는 아래 네 모듈로 옮겼다(옮기기만 했고 계산·문구·임계값은
그대로다):

    mrl/report_common.py   페이지 껍데기·CSS·값 포매팅·표/차트 helper·한/영 토글 배선
    mrl/report_v0.py       docs/backtest_v0.html · docs/index.html · v0 차트
    mrl/report_p2.py       docs/calibration_p2.html · docs/backtest_v1.html · 확률 카드 · 차트
    mrl/report_p3.py       docs/sizing_p3.html · docs/regime_p3.html · docs/track_record.html · 비중 카드 · 차트

`from mrl import report` 를 쓰던 모든 호출자(scripts/*.py, tests/*.py)는 한 줄도 고치지
않는다 — HEAD 가 내보내던 이름이 여기에 전부 다시 있다.
"""
from __future__ import annotations

import sys as _sys
import types as _types

from mrl import report_common as _report_common
from mrl import report_p2 as _report_p2
from mrl import report_p3 as _report_p3
from mrl import report_v0 as _report_v0


from mrl.report_common import (BACKTEST_START, DECISION_P2, ENSEMBLE_P3, HMM_P3, HOLDOUT_START, HONESTY_BOLD,
                               HONESTY_BULLETS, HONESTY_LINE, HONESTY_TITLE, KILL_MANUAL_PATH, KILL_P3, KILL_RECORD_PATH,
                               LABELS, MODEL_P2_PATH, MODEL_P3_PATH, P2, P2_STATES, P3, P3_DRIFT, PALETTE, Path, RESULTS_DIR,
                               SCENARIO_P3, SIGNAL_KO, SITE_URL, STATE_COLORS, STATE_LABEL, STATE_TO_TONE,
                               SUBSTITUTION_RULES, TONES, TONE_COLORS, TONE_EXPOSURE, TONE_KO, V0_SIGNALS, VARIANT_KO, _CSS,
                               _NO_PCT, _PCT_HINT, _esc, _find, _fmt, _h, _img, _is_nan, _is_num, _kpi, _kv_table, _label,
                               _legend, _new_fig, _now_str, _pct_key, _plain_float, _plain_log_axis, _png, _records_table,
                               _scalar_ok, _series_close, _sizing, _state_pill, _style_ax, _to_records, _tone_pill,
                               _tone_runs, _track, _warn_list, _write_html, base64, datetime, io, math, np, pd, timezone,
                               warnings)
# 한/영 토글 (STYLE_I18N.md §1) — 페이지 껍데기가 쓰는 도구도 같은 이름 공간에서 꺼낼 수 있게 한다
from mrl.report_common import (LANG_CSS, LANG_TOGGLE_HTML, bi, bi_attr, bi_html, bi_th, bi_ths, page_html)

from mrl.report_p2 import (DEPLOYED_BI, INFO_DISPLAY_BI, INFO_ONLY_BI, LADDER_BI, P2_CHURN_LABEL_BI, P2_PROB_UNAVAILABLE_BI)
from mrl.report_p2 import (DEPLOYED_LABEL, FEATURE_COLORS, FEATURE_KO, INFO_DISPLAY_LABEL, INFO_ONLY_LABEL, P2_CHURN_LABEL,
                           P2_FAMILY_LINE, P2_FOOTNOTE, P2_HONESTY_ITEMS, P2_KILL_EPISODES, P2_KILL_MONTHS,
                           P2_PROB_UNAVAILABLE, P2_STATE_COLORS, P2_STATE_KO, RUNG_KO, RUNG_PARAMS, TABLE_KO_P2, _ALLOC_KEYS,
                           _BLOCK_ORDER, _DEC4_KEYS, _ERA_ORDER, _HAR_ORDER, _LADDER_ORDER, _PARAM_ORDER, _PCT_KEYS_P2,
                           _RELIAB_ORDER, _RUNG_ORDER, _acceptance_box, _acceptance_tables, _acceptance_tags, _alloc_matrix,
                           _band_of, _bars_html, _coef_txt, _contrib_rows, _dod_line, _events_html, _exact_pp_parts, _first,
                           _fmt_p2, _fnum, _honesty_strip, _info_layer_rows, _ladder_records, _nat_freq, _p2_frame, _p2_kv,
                           _p2_state_pill, _p2_table, _param_records, _pct1, _pp1, _rung_txt, _table_ko_p2, _ten_freq,
                           _thresholds_fallback, _v0_line_html, acceptance_verdict_line, charts_p2, deployment_of,
                           load_summary_v0, p2_card, render_backtest_v1, render_calibration_report, v0_headline)

from mrl.report_p3 import (P3_CARD_FOOTNOTE, P3_FIRST_LINE, P3_HIGH_STATE_KO, P3_HONESTY_ITEMS, P3_INFO_NOT_SIZING,
                           P3_INFO_ONLY_LABEL, P3_LADDER_HIDDEN, P3_MEMBER_KO, P3_PARAM_LINE_FMT, P3_PARAM_TABLE,
                           P3_PARAM_TABLE_NOTE, P3_REASON_KO, P3_REGIME_CUTS, P3_REGIME_LEVELS, P3_SHADOW_LABEL,
                           P3_SIZING_SHORT, _KO_DOW, _P2_ACC_CACHE, _P3_ALARM_ORDER, _P3_BIN_ORDER, _P3_BT_ORDER,
                           _P3_LADDER_ORDER, _P3_PCT, _P3_PLAIN, _P3_REGISTRY_ORDER, _P3_STATE_ORDER, _REGIME_SECTIONS,
                           _SIZING_SECTIONS, _TRACK_PANEL_TITLES, _TRACK_SECTIONS, _cum, _dd_of, _dow_ko, _fmt_p3, _hz_block,
                           _hz_key, _hz_reason, _hz_redact, _hz_stage, _hz_val, _model_deploy_mode, _num_of, _p3_col,
                           _p3_frame, _p3_head, _p3_honesty_section, _p3_kill_bits, _p3_kv, _p3_mode_banner, _p3_mode_source,
                           _p3_nav, _p3_table, _p3_window_tables, _panel_kv, _scenarios_spread_sentence, _sub, charts_p3,
                           horizon_may_show, horizon_view, kill_power_line, load_acceptance_p2, p2_acceptance_of, p3_card,
                           p3_effective_mode, p3_param_line, p3_verdict_block, render_regime_report, render_sizing_report,
                           render_track_record)

from mrl.report_v0 import (_STATUS_KO, _directional_records, _h_of, _ledger_block, _variants_of, _verdict_block, charts_for,
                           render_backtest_report, render_index)


# 이름을 정의한 모듈들 — 아래 _Facade 가 값을 함께 심을 곳
_MODULES = (_report_common, _report_p2, _report_p3, _report_v0)


class _Facade(_types.ModuleType):
    """`mrl.report.X = ...` 를 실제 정의 모듈에도 그대로 전달한다.

    파사드는 이름을 복사해 온 것뿐이라, 여기서만 바꾸면 렌더링 코드가 보는 값은 옛 값 그대로다.
    기존 테스트는 monkeypatch.setattr(report, "_png", ...) · setattr(report, "KILL_RECORD_PATH", ...)
    처럼 파사드에 값을 심어 동작을 바꾸므로, 같은 이름을 가진 모듈 모두에 함께 심는다.
    (monkeypatch 가 되돌릴 때도 같은 경로를 지나므로 원래 값으로 정확히 돌아온다.)
    """

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        for _mod in _MODULES:
            if name in vars(_mod):
                setattr(_mod, name, value)

    def __delattr__(self, name):
        super().__delattr__(name)
        for _mod in _MODULES:
            if name in vars(_mod):
                delattr(_mod, name)


_sys.modules[__name__].__class__ = _Facade          # 반드시 파일 맨 끝 — 모듈 본문의 대입은 __setattr__ 를 거치지 않는다
