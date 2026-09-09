# -*- coding: utf-8 -*-
"""리포트 Phase 3 확장(mrl/report.py — ARCHITECTURE_PHASE3.md §10) 테스트.

* p3_card: BAD_TOKEN 없이 두 모드로 렌더 · tones 에서 한 숫자·한 문장(A·B)·예산 줄·불일치 구간·시나리오 3줄·
  게이지·정직 스트립 · **info_only 에서 상태·톤·비중·D_max 사다리 숨김**과 VALIDATION §7 문구·
  '정보(비중 제안 아님)' 회색 라벨·변동성 단독 규칙 노출(§15 stage 0).
* 유효 배치 모드 = p2 ∧ p3 ∧ ¬kill (§8.3): 리포트가 스스로 계산하고 호출자 값과 AND 한다 — 절대 올려 잡지 않는다.
* §8.4 지평 규칙: 합성 장부(3/6/12/36개월)로 n_eff < 6 에 BSS 숫자 없음·판정일 전 판정어 없음을 검사.
* render_sizing_report / render_regime_report / render_track_record: 섹션 전부 존재, 빈 summary 허용,
  모든 페이지가 summary_p2.json 의 acceptance.verdict_line 을 **원문 그대로** 싣는다.
* charts_p3: 합성 프레임으로 PNG, 열이 없으면 경고 후 생략(예외 없음).
* render_index: today["p3"] 가 있으면 카드·링크가 끼워지고, 없으면 Phase 1/2 와 동일.
실제 산출물(results/·docs/)은 건드리지 않는다(tmp_path 만).
"""
from __future__ import annotations

import html as _html
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrl import report, track                                       # noqa: E402
from mrl.config import ENSEMBLE_P3, HMM_P3, KILL_P3, P2, P3         # noqa: E402

BAD_TOKEN = re.compile(r"\b(undefined|NaN|nan|None|null|NaT)\b")
# 판정어(§8.4: 판정일 전에는 쓰지 않는다)
VERDICT_WORDS = ("validated", "provisional", "manual_kill")
ACC = {"verdict_line": "24개월 표: M1 통과 · 18개월 표: 실패(최소 블록 BSS_clim −0.0996) → 두 표 모두 통과 요구"
                       "(소유자 결정 2026-09-08) → 배치 없음(정보 제공 전용)",
       "deploy_mode": "info_only", "tone_model": None, "rule": "amended", "require_tables": ["24", "18"]}


def _visible(html: str) -> str:
    """보이는 텍스트: <style> 제거 → 태그를 공백으로 → 엔티티 복원 → 공백 정규화."""
    t = re.sub(r"<style.*?</style>", "", html, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", _html.unescape(t))


def _no_bad(html: str) -> None:
    found = sorted(set(BAD_TOKEN.findall(_visible(html))))
    assert not found, f"보이는 텍스트에 금지 토큰: {found}"


@pytest.fixture()
def no_kill(monkeypatch, tmp_path):
    """킬 파일이 없는 상태를 고정(실제 results/ 에 kill_record 가 생겨도 테스트가 흔들리지 않게)."""
    monkeypatch.setattr(report, "KILL_RECORD_PATH", tmp_path / "kill_record.json")
    monkeypatch.setattr(report, "KILL_MANUAL_PATH", tmp_path / "kill_manual.json")
    return tmp_path


# ------------------------------------------------------------------
# 합성 자료
# ------------------------------------------------------------------
def _today_p3(**kw) -> dict:
    d = {
        "asof": "2026-09-04",
        "p2_deploy_mode": "tones", "p3_deploy_mode": "tones", "deploy_sizing": True,
        "rung": "M3", "model_id": "p2m3-81c2a6bf-2024-08-30", "registry_sha": "abcdef0123456789",
        "sizing_sha": "0123456789abcdef", "theta_id": "th-2024-0a1b",
        "sizing": {"w_exec": 0.65, "w_vol": 0.65, "w_target": 0.65, "mult": 1.0, "cand": 0.65,
                   "sigma": 0.154, "sigma_target": 0.10, "state": "normal", "reason": "weekly",
                   "changed": True, "prev_w_exec": 0.70,
                   "thresholds": {"sigma_down": 0.182, "sigma_up": 0.133, "next_check": "2026-09-11"}},
        "budget": {"d_max": 0.35, "sigma_target": 0.10, "maxdd_decision": -0.235, "maxdd_vol_only": -0.307},
        "members": {"p2": 0.213, "M1": 0.19, "H": 0.24},
        "disagreement": {"lo": 0.17, "hi": 0.27, "width": 0.10, "src": "members", "flag": False},
        "scenarios": {"lines": ["시장이 보는 20일 범위(VIX 14.5): ±5.2% (80%; 과거 94% 포함, 보수적)",
                                "오늘 같은 날(확률대 20~25%): 과거 독립 23창 중 5창(11~44%)이 20일 안에 −5%",
                                "만약 −5% 에피소드가 시작되면: 33번 중 11번(20~50%)은 −10% 까지"],
                      "dd": {"dd_from_ath": -0.032, "ath_date": "2026-08-14", "breached_5": False},
                      "spread_sentence": "모든 상태·구간에서 20일 수익의 중앙값은 양수다.", "notes": []},
        "regime": {"p_high": 0.42, "q20": 0.71, "k_step": {"p_k": 0.38, "q_k": 0.71, "k": 20}, "dwell": 45},
        "kill": {"state": "not_due", "months": 2, "episodes5": 0,
                 "countdown": {"episodes": "0/8", "months": "2/36", "cap": "상한 60"},
                 "score": {"n": 30, "n_eff": 1.5, "bss_clim": 0.05, "ci_bss_clim": [-0.2, 0.3]}},
        "track": {"n_eff": 1.5, "months_elapsed": 2, "replay_mismatch_count": 0, "alarms": []},
        "acceptance": dict(ACC), "warnings": [],
    }
    d.update(kw)
    return d


def _summary_p3() -> dict:
    rng = np.random.default_rng(0)
    return {
        "run": {"generated_at_utc": "2026-09-08T00:00:00Z", "end": "2024-08-30", "first_refit": "2003-01-02",
                "sigma_target": 0.10, "d_max": 0.35, "sizing_sha": "0123456789abcdef",
                "registry_sha": "abcdef0123456789", "spec_sha256": "deadbeef" * 8, "p2_deploy_mode": "info_only"},
        "deploy_mode": "info_only",
        "sizing": {
            "table": [{"row": "buy_hold", "window": "2003+", "start": "2003-01-02", "end": "2024-08-30",
                       "years": 21.7, "cagr": 0.1085, "max_dd": -0.552, "max_dd_date": "2009-03-09",
                       "worst_month": -0.165, "ann_vol": 0.187, "switches_per_year": 0.0, "avg_exposure": 1.0,
                       "cost_total": 0.0, "vol_ratio": 1.87, "maxdd_over_sigma_t": 5.52},
                      {"row": "adopted", "window": "2003+", "start": "2003-01-02", "end": "2024-08-30",
                       "years": 21.7, "cagr": 0.0722, "max_dd": -0.235, "max_dd_date": "2009-03-09",
                       "worst_month": -0.078, "ann_vol": 0.093, "switches_per_year": 10.9,
                       "avg_exposure": 0.69, "cost_total": 0.022, "vol_ratio": 0.93,
                       "maxdd_over_sigma_t": 2.35},
                      {"row": "p2_decision", "window": "2003+", "cagr": 0.1019, "max_dd": -0.310},
                      {"row": "adopted", "window": "1993+", "cagr": 0.065, "max_dd": -0.307}],
            "ladder": [{"sigma_target": g, "d_slow": 3.5 * g, "d_fast": 2.0 * g, "maxdd_vol_only": -3.1 * g,
                        "maxdd_decision": -2.35 * g, "cagr": 0.07, "avg_exposure": 0.69,
                        "switches_per_year": 10.9, "share_behind_years": 0.91} for g in P3["vol_grid"]],
            "sensitivities": [{"row": "S-nofloor", "window": "2003+", "cagr": 0.0694, "max_dd": -0.179}],
            "episode_pnl": [{"peak_date": "2007-10-09", "trough_date": "2009-03-09", "depth": -0.552,
                             "ret_rule": -0.135, "ret_bh": -0.552, "avg_exposure": 0.25, "n_sessions": 356,
                             "protection_pp": 0.417}],
            "retention": {"conditions": {k: {"value": -0.235, "threshold": -0.25, "pass": True,
                                             "margin": 0.015, "note": f"조건 {k}"} for k in "abcdef"},
                          "failed": [], "undecided": [], "narrow": ["a"], "narrow_margin": 0.01,
                          "note_ko": "모든 유지 조건 통과 — 비중 블록 표시", "deploy_sizing": True},
            "relative": {"63": {"L": 63, "n": 86, "p10": -0.057, "p50": -0.010, "p90": 0.036,
                                "share_behind": 0.79},
                         "calendar_year": {"n_years": 22, "n_behind": 20, "share_behind": 0.91,
                                           "median": -0.058, "p10": -0.141, "p90": -0.012}},
            "deploy_sizing": True},
        "registry": [{"member": "p2", "kind": "seed", "K_s": 4, "K_u": 0, "status": "seed", "in_average": True,
                      "ledger_no": "2", "n": 5453, "brier": 0.1205, "bss_clim": 0.087, "auc": 0.70},
                     {"member": "H", "kind": "shadow", "K_s": 2, "K_u": 12, "status": "shadow",
                      "in_average": False, "ledger_no": "3a", "n": 5453, "brier": 0.1243,
                      "bss_clim": 0.066, "auc": 0.658}],
        "admission": [{"member": "H", "n_blocks": 11, "n_improved": 6, "verdict": "SHADOW"}],
        "ladder": [{"step": "M3->H", "from": "M3", "to": "H", "block": "all", "n": 5453, "n_blocks": 272,
                    "brier_from": 0.1205, "brier_to": 0.1243, "bss": -0.02, "mean": -3.8e-4,
                    "lo": -20e-4, "hi": 12e-4}],
        "blocks": {"24": [{"block": 1, "n": 504, "n_blocks": 25, "bss_clim": 0.151}],
                   "18": [{"block": 1, "n": 378, "n_blocks": 19, "bss_clim": -0.0996}]},
        "hmm": {"theta_table": [{"refit_date": "2024-01-02", "theta_id": "th-2024-0a1b", "p00": 0.981,
                                 "p11": 0.976, "occupancy_high": 0.43, "n_iter": 22, "guard_ok": True}],
                "guard_log": [], "timing": {"total_sec": 27.1}},
        "era_auc": [{"era": "2020-2024", "start": "2020-01-01", "end": "2024-08-30", "auc_x_vix": 0.594,
                     "auc_p_m3": 0.62, "auc_p_hmm": 0.666}],
        "reliability": {"H": [{"bin": "[0.20, 0.25)", "n": 453, "n_eff": 22.6, "mean_p": 0.22, "obs": 0.236,
                               "wilson_lo": 0.11, "wilson_hi": 0.44}]},
        "ablations": {"q20_platt": 0.0902, "diag_cov": 0.0898},
        "scenarios": {"bins": [{"bin": "[0.00, 0.08)", "bin_lo": 0.0, "bin_hi": 0.08, "pooled": False,
                                "n": 1422, "n_eff": 71.1, "mean_p": 0.06, "obs": 0.076, "wilson_lo": 0.03,
                                "wilson_hi": 0.16, "ret_p10": -0.030, "ret_p50": 0.010, "ret_p90": 0.033,
                                "mdd_p10": -0.045, "grey": False},
                               {"bin": "[0.25, 1.00]", "bin_lo": 0.25, "bin_hi": 1.0, "pooled": True,
                                "n": 932, "n_eff": 46.6, "mean_p": 0.32, "obs": 0.33, "wilson_lo": 0.20,
                                "wilson_hi": 0.48, "ret_p10": -0.09, "ret_p50": 0.02, "ret_p90": 0.08,
                                "mdd_p10": -0.15, "grey": False}]},
        "reference": {"rolling_bss": {"252": [float(x) for x in rng.normal(0.06, 0.15, 200)],
                                      "756": [float(x) for x in rng.normal(0.065, 0.05, 120)]}},
        "kill": {"state": "not_due", "months": 2, "episodes5": 0},
        "kill_power": {"admission_power": [{"n_improved": 8, "p": 0.113}]},
        "determinism": {"theta_bit_identical": True},
        "selftest": {"smoothed_not_called_outside_regime": True},
        "alarms": [{"asof": "2026-09-01", "code": "D1_p_level", "value": 0.038, "threshold": 0.04,
                    "action": "display"}],
        "acceptance": dict(ACC),
        "warnings": ["테스트 경고"],
        "disclosure": "#2 사전 관측 + 설계 단계 비중 관측 참조",
    }


def _frames(n: int = 400, seed: int = 0):
    idx = pd.bdate_range("2003-01-02", periods=n)
    rng = np.random.default_rng(seed)
    bt = pd.DataFrame({"w_exec": np.clip(rng.random(n), 0.25, 1.0),
                       "w_vol": np.clip(rng.random(n), 0.25, 1.0),
                       "ret_rule": rng.normal(0.0003, 0.006, n),
                       "ret_bh": rng.normal(0.0004, 0.011, n),
                       "dd_rule": -np.abs(rng.normal(0.03, 0.02, n))}, index=idx)
    oos = pd.DataFrame({"p_p2": rng.random(n) * 0.4, "p_m1": rng.random(n) * 0.4, "p_h": rng.random(n) * 0.4,
                        "lo": rng.random(n) * 0.2, "hi": 0.2 + rng.random(n) * 0.3,
                        "p_hmm_high": rng.random(n), "q20": rng.random(n), "refit_year": idx.year}, index=idx)
    spy = pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.011, n)), index=idx)
    return oos, bt, spy


def _sessions(n: int, start: str = "2016-01-04") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _synthetic_ledger(dates, *, good: bool = True) -> pd.DataFrame:
    """킬룰·지평 검사용 최소 장부(계약 열 이름 그대로; test_track.py 와 같은 규약)."""
    n = len(dates)
    y = np.zeros(n)
    y[::6] = 1.0
    p = (0.05 + 0.50 * y) if good else (0.55 - 0.50 * y)
    return pd.DataFrame({"asof": [str(d.date()) for d in dates], "prob_dd5_20": p, "p2_clim": 0.16,
                         "y_dd5_20": y, "p2_deploy_mode": "tones", "p2_state": "normal",
                         "p3_w_exec": 0.65, "p3_sigma_ewma": 0.15})


def _rising(dates) -> pd.Series:
    return pd.Series(100.0 * np.exp(np.arange(len(dates)) * 0.0008), index=dates)


# ==================================================================
# 1. p3_card — 두 모드
# ==================================================================
def test_p3_card_renders_both_modes_without_bad_token(no_kill):
    _no_bad(report.p3_card(_today_p3(), "tones"))
    _no_bad(report.p3_card(_today_p3(p2_deploy_mode="info_only"), "info_only"))
    _no_bad(report.p3_card({}, None))                      # 빈 dict 도 예외 없이, 금지 토큰 없이


def test_p3_card_tones_shows_one_number_and_one_sentence(no_kill):
    h = report.p3_card(_today_p3(), "tones")
    # (A) 숫자 하나 + 문장 하나
    assert "오늘 주식 비중 0.65" in h
    assert "변동성 규칙 0.65" in h and "목표 10%" in h and "예상 15.4%" in h
    assert "상태 배수 1.00" in h and "normal" in h
    # (B) 둘째 줄: 다음 점검일·임계·즉시 격상
    assert "다음 점검" in h and "2026-09-11" in h
    assert "18.2%" in h and "0.55" in h and "13.3%" in h and "0.75" in h
    assert "결정층이 격상되면 즉시 하향" in h
    assert "주간 점검" in h and "weekly" in h              # 변경 사유
    # 예산 줄 + 사다리 링크
    assert "예산 −35%" in h and "목표 변동성 10%" in h
    assert 'href="sizing_p3.html"' in h
    _no_bad(h)


def test_p3_card_info_only_hides_state_tone_exposure_and_ladder(no_kill):
    """§15 stage 0 · §8.3 — p2 가 info_only 면 상태·톤·비중·D_max 사다리를 숨기고
    변동성 단독 규칙만 회색 '정보(비중 제안 아님)' 로 보여준다."""
    h = report.p3_card(_today_p3(p2_deploy_mode="info_only"), "info_only")
    vis = _visible(h)
    # 비중·톤·상태 제안이 없어야 한다
    assert "오늘 주식 비중" not in h
    assert "× 상태 배수" not in vis, "상태 배수를 적용한 비중을 제안하면 안 된다"
    assert "주식 비중 65%" not in vis and "주식 비중 0.65" not in vis
    assert '<span class="pill"' not in h, "톤 pill 이 보이면 안 된다"
    assert 'class="st"' not in h, "상태 pill 이 보이면 안 된다"
    # 대신 보여야 하는 것
    assert report.P3_INFO_ONLY_LABEL in h                  # VALIDATION §7 문구
    assert report.P3_INFO_NOT_SIZING in h                  # '정보(비중 제안 아님)'
    assert "변동성 단독 규칙" in h and "상태 배수 미적용" in h
    assert "0.65" in h                                     # w_vol 은 회색 가정치로 남는다
    # D_max 사다리는 숨긴다
    assert "예산 −35%" not in h
    assert "사다리" in h and "유효 배치 모드에서만" in h
    # 게이지·시나리오·구간은 남는다(§8.3)
    assert "국면 게이지" in h and report.P3_SHADOW_LABEL in h
    assert "시나리오" in h and "모델들은 100일 중" in h
    _no_bad(h)


def test_p3_card_info_only_when_deploy_sizing_false(no_kill):
    h = report.p3_card(_today_p3(deploy_sizing=False), "tones")
    assert "유지 조건 위반" in h                            # §6.4 — 비중 블록 숨김 배지
    _no_bad(h)


# ==================================================================
# 2. 유효 배치 모드 (§8.3: p2 ∧ p3 ∧ ¬kill; 올려 잡지 않는다)
# ==================================================================
def test_effective_mode_never_upgrades_beyond_the_data(no_kill):
    eff = report.p3_effective_mode({"p2_deploy_mode": "info_only", "p3_deploy_mode": "tones"}, "tones")
    assert eff["mode"] == "info_only" and eff["tones"] is False
    assert any("p2 deploy_mode = info_only" in r for r in eff["reasons"])
    assert any("낮은 쪽" in r for r in eff["reasons"])


def test_effective_mode_caller_info_only_wins_over_data(no_kill):
    eff = report.p3_effective_mode({"p2_deploy_mode": "tones", "p3_deploy_mode": "tones"}, "info_only")
    assert eff["mode"] == "info_only"
    assert any("호출자 effective_mode = info_only" in r for r in eff["reasons"])


def test_effective_mode_kill_state_forces_info_only(no_kill):
    for kill in ({"state": "info_only"}, {"state": "manual_kill"}, {"killed": True, "state": "provisional"}):
        eff = report.p3_effective_mode({"p2_deploy_mode": "tones", "p3_deploy_mode": "tones", "kill": kill},
                                       "tones")
        assert eff["mode"] == "info_only", kill
        assert eff["killed"] is True


def test_effective_mode_kill_record_file_forces_info_only(monkeypatch, tmp_path):
    rec = tmp_path / "kill_record.json"
    rec.write_text('{"ledger_entry": "7x"}', encoding="utf-8")
    monkeypatch.setattr(report, "KILL_RECORD_PATH", rec)
    monkeypatch.setattr(report, "KILL_MANUAL_PATH", tmp_path / "kill_manual.json")
    h = report.p3_card(_today_p3(), "tones")
    assert "오늘 주식 비중" not in h
    assert report.P3_INFO_ONLY_LABEL in h
    _no_bad(h)


def test_effective_mode_reads_model_files_when_dict_is_silent(monkeypatch, tmp_path):
    """dict 에 모드가 없으면 results/model_p{2,3}.json 을 읽는다(§8.3: 리포트가 스스로 계산)."""
    (tmp_path / "m2.json").write_text(json.dumps({"deploy_mode": "tones"}), encoding="utf-8")
    (tmp_path / "m3.json").write_text(json.dumps({"deploy_mode": "tones"}), encoding="utf-8")
    monkeypatch.setattr(report, "MODEL_P2_PATH", tmp_path / "m2.json")
    monkeypatch.setattr(report, "MODEL_P3_PATH", tmp_path / "m3.json")
    monkeypatch.setattr(report, "KILL_RECORD_PATH", tmp_path / "kill_record.json")
    monkeypatch.setattr(report, "KILL_MANUAL_PATH", tmp_path / "kill_manual.json")
    eff = report.p3_effective_mode({}, "tones")
    assert eff["mode"] == "tones"
    assert "model_p2.json" in eff["sources"]["p2"] and "model_p3.json" in eff["sources"]["p3"]

    # 파일이 없으면 '확인 불가' — tones 로 올려 잡지 않는다
    monkeypatch.setattr(report, "MODEL_P2_PATH", tmp_path / "missing.json")
    eff2 = report.p3_effective_mode({}, "tones")
    assert eff2["mode"] == "info_only"
    assert eff2["sources"]["p2"] == "확인 불가"


def test_effective_mode_matches_track_effective_mode(no_kill, tmp_path):
    """AND 규칙의 단일 원천은 track.effective_mode 다 — 두 곳이 달라지면 실패한다."""
    for p2 in ("tones", "info_only"):
        for p3 in ("tones", "info_only"):
            mine = report.p3_effective_mode({"p2_deploy_mode": p2, "p3_deploy_mode": p3})["mode"]
            theirs = track.effective_mode(p2, p3, kill_record_path=tmp_path / "a.json",
                                          kill_manual_path=tmp_path / "b.json")
            assert mine == theirs, (p2, p3)


# ==================================================================
# 3. 카드의 나머지 블록 (§10 3~6)
# ==================================================================
def test_p3_card_probability_headline_and_disagreement_flag(no_kill):
    h = report.p3_card(_today_p3(), "tones")
    assert "모델들은 100일 중 17~27일로 갈린다" in h
    assert "p2 M3 21" in h and "보정 VIX 19" in h and "HMM 24" in h
    assert "불일치 10.0pp" in h
    flagged = report.p3_card(_today_p3(disagreement={"lo": 0.10, "hi": 0.45, "src": "members", "flag": True}),
                             "tones")
    assert "불일치 플래그" in flagged
    # 구간이 없으면 조용히 좁히지 않는다
    none_band = report.p3_card(_today_p3(disagreement={}, members={}), "tones")
    assert "불일치 구간 없음" in none_band
    _no_bad(none_band)


def test_p3_card_scenario_lines_and_spread_sentence(no_kill):
    h = report.p3_card(_today_p3(), "tones")
    for line in _today_p3()["scenarios"]["lines"]:
        assert line in _visible(h)
    assert "ATH 대비 현재" in h and "2026-08-14" in h
    assert "20일 수익의 중앙값은 양수" in h                  # §7 필수 문장
    empty = report.p3_card(_today_p3(scenarios={}), "tones")
    assert "시나리오 표 없음" in empty                       # n·n_eff 없이 문장을 만들지 않는다
    _no_bad(empty)


def test_p3_card_regime_gauge_is_shadow_and_d9_hides_it(no_kill):
    h = report.p3_card(_today_p3(), "tones")
    assert report.P3_SHADOW_LABEL in h
    assert report.P3_HIGH_STATE_KO in h and "위기" in h      # '위기가 아니라 고변동 상태' (§16 9)
    assert "중간" in _visible(h)                             # P_high 0.42 → 중간
    hidden = report.p3_card(_today_p3(track={"alarms": [{"code": "D9_hmm", "action": "hide_gauge_reject_member"}]}),
                            "tones")
    assert "게이지 숨김" in hidden
    _no_bad(hidden)


def test_p3_param_line_is_the_contract_text():
    line = report.p3_param_line()
    assert line == ("생산 확률 적합 4/5 (Phase 2) · Phase 3 추가 0 · 그림자 H: Platt 2 (K_s) + HMM θ 12 (K_u)"
                    " · 예산 밖 HAR OLS 4 · v0 Platt 2")
    assert f"{P2['param_count']}/{P2['budget']}" in line
    assert f"HMM θ {HMM_P3['param_count']}" in line


def test_p3_card_honesty_strip_has_shas_mode_countdown_and_footnote(no_kill):
    h = report.p3_card(_today_p3(), "tones")
    assert report.p3_param_line() in _visible(h)
    assert "registry_sha abcdef012345" in h and "sizing_sha 0123456789ab" in h
    assert "theta_id th-2024-0a1b" in h
    assert "유효 모드(p2 ∧ p3 ∧ ¬kill) = tones" in h
    assert f"실현 ≥5% 에피소드 0/{KILL_P3['min_episodes']}" in h
    assert f"2/{KILL_P3['min_months']}개월" in h and f"상한 {KILL_P3['max_months']}" in h
    assert "D10 재현 불일치 0건" in h
    assert f"신선 블록 —/{ENSEMBLE_P3['fresh_blocks_min']}" in h
    assert report.P3_CARD_FOOTNOTE in _visible(h)
    assert track.HORIZON_FOOTNOTE in _visible(h)


def test_p3_card_prints_acceptance_verdict_line_verbatim(no_kill):
    h = report.p3_card(_today_p3(), "tones")
    assert ACC["verdict_line"] in _visible(h)


# ==================================================================
# 4. §8.4 지평 규칙 — 카드
# ==================================================================
def test_p3_card_hides_skill_numbers_before_six_windows(no_kill):
    """n_eff < 6 이면 BSS 숫자를 회색으로도 내지 않고 사유를 남긴다(§8.4)."""
    h = report.p3_card(_today_p3(), "tones")
    vis = _visible(h)
    assert re.search(r"정확도 점수\(기준선 대비\)\s*[+-]?\d", vis) is None   # BSS 점추정이 그대로 새면 안 된다
    assert re.search(r"BSS\s*[+-]\d", vis) is None
    assert "§8.4" in vis and "bss 숨김" in vis
    # 36개월·n_eff 40 이면 숫자가 나온다
    grown = _today_p3(track={"n_eff": 40.0, "months_elapsed": 37, "replay_mismatch_count": 0, "alarms": []},
                      kill={"state": "provisional", "months": 37, "episodes5": 3,
                            "countdown": {"episodes": "3/8", "months": "37/36", "cap": "상한 60"},
                            "score": {"n": 800, "n_eff": 40.0, "bss_clim": 0.062,
                                      "ci_bss_clim": [0.01, 0.12]}})
    h2 = report.p3_card(grown, "tones")
    assert "지금까지의 정확도 점수(기준선 대비)" in _visible(h2) and "+0.062" in _visible(h2)
    assert "킬룰 상태: provisional" in _visible(h2)
    _no_bad(h2)


def test_horizon_view_defaults_to_the_most_conservative_stage():
    hz = report.horizon_view({})
    assert hz["stage"] == "0-3m"
    assert report.horizon_may_show(hz, "bss") is False
    assert report.horizon_may_show(hz, "verdict") is False
    assert report.horizon_may_show(hz, "alarms") is True
    # track 이 준 dict 는 그대로 쓴다
    hz2 = report.horizon_view({"horizon": track.horizon_copy({"n_eff": 40.0, "months_elapsed": 40})})
    assert hz2["stage"] == "36m" and report.horizon_may_show(hz2, "verdict") is True


# ==================================================================
# 5. 주간 페이지
# ==================================================================
def _pages(tmp_path, s=None, t=None, charts=None):
    s = _summary_p3() if s is None else s
    charts = {} if charts is None else charts
    p1, p2_, p3_ = (tmp_path / "sizing_p3.html", tmp_path / "regime_p3.html", tmp_path / "track_record.html")
    report.render_sizing_report(s, {"allocation": {"cagr": 0.1019, "max_dd": -0.310}}, {}, p1, dict(charts))
    report.render_regime_report(s, p2_, dict(charts))
    tt = t if t is not None else {"n_scored": 0, "n_eff": 0.0, "months_elapsed": 0, "kill": {"state": "not_started"},
                                  "panel": {}, "alarms": [], "acceptance": dict(ACC)}
    report.render_track_record(tt, s, p3_, dict(charts))
    return {p.name: p.read_text(encoding="utf-8") for p in (p1, p2_, p3_)}


def test_sizing_report_has_every_section(tmp_path, no_kill):
    html = _pages(tmp_path)["sizing_p3.html"]
    for sec in ("① v0 completed 요약", "② p2 판정 규칙", "③ 정직한 읽기", "④ 백테스트 성적표",
                "⑤ 담을 수 있는 최대치 단계", "⑥ 민감도", "⑦ 하락 사건별 손익", "⑧ 유지 조건", "⑨ 63/126/252",
                "⑩ 규약과 해시", "정직 문구와 리스크"):
        assert sec in html, sec
    assert "2003+" in html and "1993+" in html               # 창별 표
    assert "예산은 보장이 아닙니다" in html
    assert "근소 통과" in html                                # (a) 근소 통과 경고
    assert "w[t] → r[t+1]" in html or "w[t] → t+1" in html or "점 원칙" in html
    assert "0123456789ab" in html                             # sizing_sha
    _no_bad(html)


def test_regime_report_has_every_section_and_param_table(tmp_path, no_kill):
    html = _pages(tmp_path)["regime_p3.html"]
    for sec in ("① 등록부", "② 단계 늘려 보기", "③ 블록 표", "④ θ 경로", "⑤ P(고변동) 밴드",
                "⑥ 시대별 AUC", "⑦ 소거", "⑧ 말한 확률이 실제와 맞나", "⑨ 결정론", "정직 문구와 리스크"):
        assert sec in html, sec
    # §14 파라미터 회계 표를 그대로
    assert "파라미터 회계" in html
    for row in report.P3_PARAM_TABLE:
        assert row["줄"] in html, row["줄"]
    assert report.p3_param_line() in _visible(html)
    assert report.P3_HIGH_STATE_KO in html
    _no_bad(html)


def test_track_record_has_every_panel(tmp_path, no_kill):
    dates = _sessions(300)
    t = track.summary_p3(_synthetic_ledger(dates), _rising(dates), None, dates[-1],
                         p2_deploy_mode="tones", p3_deploy_mode="info_only")
    vis = _visible(_pages(tmp_path, t=t)["track_record.html"])
    for title in report._TRACK_PANEL_TITLES.values():
        assert title in vis, title
    html = _pages(tmp_path, t=t)["track_record.html"]
    assert "① 지평별 표시 규칙" in html
    assert track.HORIZON_FOOTNOTE in _visible(html)
    _no_bad(html)


def test_every_page_carries_the_acceptance_verdict_line_verbatim(tmp_path, no_kill):
    pages = _pages(tmp_path)
    for name, html in pages.items():
        assert ACC["verdict_line"] in _visible(html), name
        assert "배치 판정" in html, name


def test_every_page_opens_with_no_added_skill(tmp_path, no_kill):
    for name, html in _pages(tmp_path).items():
        assert "맞히는 실력을 더하지 못했습니다" in html, name


def test_pages_show_info_only_banner_when_p2_is_info_only(tmp_path, no_kill):
    for name, html in _pages(tmp_path).items():
        assert report.P3_INFO_ONLY_LABEL in html, name
        assert "상태·톤·비중·D_max 사다리를 숨기고" in html, name


def test_pages_tolerate_empty_summary(tmp_path, no_kill):
    p1, p2_, p3_ = (tmp_path / "a.html", tmp_path / "b.html", tmp_path / "c.html")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report.render_sizing_report({}, {}, {}, p1, {})
        report.render_regime_report({}, p2_, {})
        report.render_track_record({}, {}, p3_, {})
    for p in (p1, p2_, p3_):
        html = p.read_text(encoding="utf-8")
        assert "없음" in html, "빈 summary 는 '…없음' 으로 말해야 한다(조용히 비우지 않는다)"
        _no_bad(html)
    for bad in (None, [], "x"):
        with pytest.raises(TypeError):
            report.render_sizing_report(bad, {}, {}, p1, {})
        with pytest.raises(TypeError):
            report.render_regime_report(bad, p2_, {})
        with pytest.raises(TypeError):
            report.render_track_record(bad, {}, p3_, {})


def test_verdict_line_missing_is_said_out_loud(tmp_path, monkeypatch, no_kill):
    """acceptance 를 못 찾으면 리포트가 판정을 대신 쓰지 않고 그 사실을 적는다."""
    monkeypatch.setattr(report, "load_acceptance_p2", lambda path=None: {})
    s = _summary_p3()
    s.pop("acceptance")
    out = tmp_path / "s.html"
    report.render_sizing_report(s, {}, {}, out, {})
    html = out.read_text(encoding="utf-8")
    assert "배치 판정 줄 없음" in html and "조용한 실패 금지" in html
    _no_bad(html)


# ==================================================================
# 6. §8.4 지평 규칙 — track_record 페이지 (합성 장부)
# ==================================================================
@pytest.mark.parametrize("n,stage", [(63, "0-3m"), (200, "6m"), (300, "12m"), (800, "36m")])
def test_track_record_applies_horizon_rules(tmp_path, no_kill, n, stage):
    dates = _sessions(n)
    t = track.summary_p3(_synthetic_ledger(dates), _rising(dates), None, dates[-1],
                         p2_deploy_mode="tones", p3_deploy_mode="info_only")
    assert t["horizon"]["stage"] == stage, (n, t["horizon"]["stage"])
    out = tmp_path / f"track_{n}.html"
    report.render_track_record(t, _summary_p3(), out, {})
    html = out.read_text(encoding="utf-8")
    vis = _visible(html)
    _no_bad(html)
    assert f"지평 단계 {stage}" in vis

    if stage == "0-3m":
        # n_eff < 6 — BSS·Brier 숫자가 회색으로도 나오면 안 된다
        assert "§8.4" in vis and "bss 숨김" in vis
        assert "+0.7" not in vis and "0.742" not in vis
        assert re.search(r"BSS vs 기후학\s*[+-]\d", vis) is None
    else:
        assert "Brier" in vis                                # 6개월부터 숫자가 나온다
    # §16 정직 문구는 킬룰 검정력을 설명하며 판정어를 인용한다 — 판정 검사는 그 앞부분만 본다
    body = vis.split("정직 문구와 리스크")[0]
    if stage != "36m":
        for w in VERDICT_WORDS:
            assert w not in body, (stage, w)
        assert "verdict 숨김" in body or "판정: —" in body
    else:
        assert "provisional" in body                         # 36개월·BSS > 0 → 잠정


def test_track_record_hides_member_verdicts_before_six_windows(tmp_path, no_kill):
    dates = _sessions(63)
    led = _synthetic_ledger(dates)
    led["p3_p_h"] = led["prob_dd5_20"] + 0.01
    t = track.summary_p3(led, _rising(dates), None, dates[-1],
                         p2_deploy_mode="tones", p3_deploy_mode="info_only")
    out = tmp_path / "t.html"
    report.render_track_record(t, {}, out, {})
    vis = _visible(out.read_text(encoding="utf-8"))
    assert "멤버" in vis
    assert "undecided" not in vis and "rejected" not in vis


# ==================================================================
# 7. charts_p3
# ==================================================================
def test_charts_p3_makes_pngs_from_synthetic_frames():
    oos, bt, spy = _frames()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = report.charts_p3(oos, bt, spy, _summary_p3(), None)
    for key in ("w_path", "cum_returns", "drawdown", "ladder_frontier", "members_band", "hmm_phigh",
                "era_auc", "bins_live", "bss_vs_hist", "alarms_timeline", "kill_countdown"):
        assert key in out, key
        assert out[key][:4] == b"\x89PNG", key


def test_charts_p3_skips_missing_inputs_with_warnings_not_exceptions():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = report.charts_p3(None, None, None, {}, None)
    assert out == {}
    msgs = " ".join(str(x.message) for x in w)
    for key in ("w_path", "cum_returns", "ladder_frontier", "bins_live", "bss_vs_hist"):
        assert key in msgs, key
    # 열이 일부만 있어도 그 차트만 빠진다
    _, bt, _ = _frames()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out2 = report.charts_p3(None, bt[["w_exec"]], None, {}, None)
    assert set(out2) == {"w_path"}


# ==================================================================
# 8. render_index 배선
# ==================================================================
def _today_index(**kw) -> dict:
    d = {"asof": "2026-09-04", "market_status": "current",
         "completed": {"tone": "hold", "score": 0.4, "state": "GREEN", "asof": "2026-09-04"}}
    d.update(kw)
    return d


def test_render_index_inserts_p3_card_and_links(tmp_path, no_kill):
    out = tmp_path / "index.html"
    report.render_index(_today_index(p3=_today_p3()), {}, out)
    html = out.read_text(encoding="utf-8")
    assert 'id="p3"' in html
    assert "오늘 주식 비중 0.65" in html
    for link in ("sizing_p3.html", "regime_p3.html", "track_record.html"):
        assert f'href="{link}"' in html, link
    _no_bad(html)


def test_render_index_inherits_p2_deploy_mode_into_the_p3_card(tmp_path, no_kill):
    """p2 카드가 info_only 면 p3 카드도 info_only 여야 한다(한 페이지가 두 말을 하지 않는다)."""
    p3 = _today_p3()
    p3.pop("p2_deploy_mode")
    out = tmp_path / "index.html"
    report.render_index(_today_index(p2={"p": 0.21, "clim": 0.16, "deploy_mode": "info_only",
                                         "state": "normal", "acceptance": dict(ACC)}, p3=p3), {}, out)
    html = out.read_text(encoding="utf-8")
    assert 'id="p3"' in html
    assert "오늘 주식 비중" not in html
    assert report.P3_INFO_ONLY_LABEL in html
    _no_bad(html)


def test_render_index_without_p3_is_unchanged(tmp_path):
    out = tmp_path / "index.html"
    report.render_index(_today_index(), {}, out)
    html = out.read_text(encoding="utf-8")
    assert 'id="p3"' not in html
    for link in ("sizing_p3.html", "regime_p3.html", "track_record.html"):
        assert link not in html
    _no_bad(html)


def test_pages_and_card_are_deterministic(tmp_path, no_kill):
    """같은 입력 → 같은 바이트(차트 PNG 포함). 계약의 결정론(seed=0) 을 리포트 층에서도 지킨다."""
    s = _summary_p3()
    oos, bt, spy = _frames()
    digests = []
    # 파일 **이름**이 같아야 한다 — 알약이 자매 문서(<stem>.en.html)를 가리키므로 이름이 내용에 들어간다.
    for i in (1, 2):
        run = tmp_path / f"run{i}"
        run.mkdir()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ch = report.charts_p3(oos, bt, spy, s, None)
            report.render_sizing_report(s, {}, {}, run / "sizing_p3.html", dict(ch))
            report.render_regime_report(s, run / "regime_p3.html", dict(ch))
        digests.append(tuple((run / f"{x}.html").read_bytes() for x in ("sizing_p3", "regime_p3")))
    assert digests[0] == digests[1]
    assert report.p3_card(_today_p3(), "tones") == report.p3_card(_today_p3(), "tones")


def test_info_only_card_never_claims_a_tone_even_with_a_state(no_kill):
    """상태가 들어와도 info_only 면 톤·비중을 주장하지 않는다(§15 stage 0 — 코드가 강제)."""
    d = _today_p3(p2_deploy_mode="info_only")
    d["sizing"]["state"] = "reduce"
    d["sizing"]["mult"] = 0.25
    h = report.p3_card(d, "info_only")
    vis = _visible(h)
    assert "축소" not in vis and "reduce" not in vis
    assert "× 상태 배수" not in vis
    assert report.P3_INFO_NOT_SIZING in h
    _no_bad(h)


# ==================================================================
# 회귀 — 검토에서 확인된 결함
# ==================================================================
def test_p3_card_does_not_claim_a_deployed_rung_when_nothing_is_deployed(no_kill):
    """§15 stage 0 — 같은 페이지가 '배포된 단 없음' 이라 말하는데 카드가 '오늘 배포 단 M3' 를 주장했다."""
    h = report.p3_card(_today_p3(p2_deploy_mode="info_only"), "info_only")
    vis = _visible(h)
    assert "배포 단" not in vis, "배치된 단이 없는 카드가 '배포 단' 을 주장하면 안 된다"
    assert "오늘 생산 확률 단" in vis and report.INFO_DISPLAY_LABEL in vis
    _no_bad(h)


def test_p3_card_says_deployed_rung_when_a_rung_is_deployed(no_kill):
    h = _visible(report.p3_card(_today_p3(p2_deploy_mode="tones", p2_tone_model="M1"), "tones"))
    assert "오늘 배포 단" in h


def test_sizing_relative_sentence_matches_its_own_table(tmp_path, no_kill):
    """§15 stage 0: 설계 단계 근사(79/88/90%)를 공식 표 옆에 섞지 않는다."""
    s = _summary_p3()
    rel = s["sizing"]["relative"]
    rel["63"]["share_behind"] = 0.7558
    rel["126"] = {"L": 126, "n": 43, "share_behind": 0.8372}
    rel["252"] = {"L": 252, "n": 21, "share_behind": 0.9048, "p50": -0.047, "best": 0.226}
    out = tmp_path / "s.html"
    report.render_sizing_report(s, {}, {}, out, {})
    html = out.read_text(encoding="utf-8")
    assert "75.6%/83.7%/90.5%" in html, "문장이 바로 아래 표와 다른 값을 말한다"
    assert "79/88/90" not in html
    assert "중앙(50분위)" in html                              # p50 에 한글 머리말
    assert "-4.7%" in html and "0.226" not in html             # p50·best 는 백분율로
    _no_bad(html)


def test_kill_power_sentence_comes_from_the_run_not_from_prose(tmp_path, no_kill):
    """§16.7 — 재생성되지 않으면 페이지에 쓰지 않는다. 쓸 때는 산출값 그대로."""
    s = _summary_p3()
    s["kill_power"] = {"p2": {"p_col": "p_p2", "n_windows": 223,
                              "sentence_ko": "이 규칙은 과거 36개월 창에서 이 모델을 12% 오기각한다."}}
    out = tmp_path / "s.html"
    report.render_sizing_report(s, {}, {}, out, {})
    html = out.read_text(encoding="utf-8")
    assert "이 규칙은 과거 36개월 창에서 이 모델을 12% 오기각한다." in html
    assert "창 223" in html
    # 없으면 숫자를 지어내지 않고 그 사실을 쓴다
    s.pop("kill_power")
    out2 = tmp_path / "s2.html"
    report.render_sizing_report(s, {}, {}, out2, {})
    h2 = out2.read_text(encoding="utf-8")
    assert "재생성하지 않아 표시하지 않는다" in h2
    assert "오기각" not in _visible(h2).replace("재생성하지 않아 표시하지 않는다", "")
    _no_bad(h2)


def test_track_record_renders_scenario_bin_and_state_tables_with_grey_rows(tmp_path, no_kill):
    """§7 A·B + VALIDATION §5 6번: n·n_eff·Wilson 없이 시나리오 표를 보이지 않으며,
    n_eff < 20 인 칸은 회색으로 **보여야** 한다(계산만 하고 감추면 규칙이 검증 불가능하다)."""
    dates = _sessions(300)
    t = track.summary_p3(_synthetic_ledger(dates), _rising(dates), None, dates[-1],
                         p2_deploy_mode="tones", p3_deploy_mode="info_only")
    s = _summary_p3()
    s["scenarios"] = {
        "bins": [{"bin": "[0.00, 0.08)", "bin_lo": 0.0, "bin_hi": 0.08, "pooled": False, "n": 1422,
                  "n_eff": 71.1, "mean_p": 0.054, "obs": 0.0759, "wilson_lo": 0.034, "wilson_hi": 0.161,
                  "share_ep10_start": 0.0176, "grey": False},
                 {"bin": "[0.25, 1.00)", "bin_lo": 0.25, "bin_hi": 1.0, "pooled": True, "n": 380,
                  "n_eff": 19.0, "mean_p": 0.35, "obs": 0.42, "wilson_lo": 0.25, "wilson_hi": 0.61,
                  "share_ep10_start": 0.09, "grey": True}],
        "states": [{"state": "normal", "n": 4313, "n_eff": 215.65, "obs": 0.1134,
                    "wilson_lo": 0.0777, "wilson_hi": 0.1626, "grey": False},
                   {"state": "reduce", "n": 255, "n_eff": 12.75, "obs": 0.482,
                    "wilson_lo": 0.246, "wilson_hi": 0.727, "grey": True}],
    }
    out = tmp_path / "track.html"
    report.render_track_record(t, s, out, {})
    html = out.read_text(encoding="utf-8")
    vis = _visible(html)
    for head in ("구간 하한", "10% 넘게 떨어지기 시작한 비율", "회색(단독 표시 금지)", "Wilson 하한"):
        assert head in vis, head
    assert "12.8" in vis and "48.2%" in vis and "24.6%" in vis and "72.7%" in vis   # 얇은 reduce 행
    assert "회색(단독 표시 금지): [0.25, 1.00), reduce" in vis                       # 회색 배지
    _no_bad(html)
