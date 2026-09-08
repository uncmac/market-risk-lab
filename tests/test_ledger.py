# -*- coding: utf-8 -*-
"""장부(mrl/ledger.py) 테스트 + 리포트(mrl/report.py) 스모크 테스트.

임시 디렉터리만 사용한다(results/ 의 실제 장부는 건드리지 않음).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # tests/ 만 sys.path 에 오르는 경우 대비
    sys.path.insert(0, str(ROOT))

from mrl import ledger, report                                    # noqa: E402
from mrl.config import TONES, V0_SIGNALS, TONE_EXPOSURE           # noqa: E402


# ------------------------------------------------------------------
# 합성 자료
# ------------------------------------------------------------------
def _spy(n=200, start="2025-01-02", seed=0) -> pd.Series:
    """평일 인덱스 합성 SPY 종가(양의 드리프트 + 잡음)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    px = 500 * np.cumprod(1 + rng.normal(0.0004, 0.01, n))
    return pd.Series(px, index=idx, name="SPY")


def _row(asof, tone="hold", variant="completed", **kw) -> dict:
    """v0_day 반환값 모양(states_d dict) + 부가 정보."""
    r = {"asof": asof, "variant": variant, "tone": tone,
         "states_d": {k: "GREEN" for k in V0_SIGNALS},
         "score_d": 0.31, "score_w": 0.12, "score_m": 0.55,
         "overall_d": "GREEN", "overall_w": "AMBER", "overall_m": "GREEN",
         "spy_close": 512.34, "vix_close": 15.2, "run_id": "test"}
    r.update(kw)
    return r


# ------------------------------------------------------------------
# append_today
# ------------------------------------------------------------------
def test_append_creates_file_with_contract_columns(tmp_path):
    path = tmp_path / "track_record.csv"
    assert ledger.append_today(_row("2025-03-03"), path) is True
    assert path.exists()
    df = pd.read_csv(path)
    assert list(df.columns) == ledger.LEDGER_COLUMNS
    assert len(df) == 1
    assert df.loc[0, "asof"] == "2025-03-03"
    assert df.loc[0, "variant"] == "completed"
    assert df.loc[0, "tone"] == "hold"
    for k in V0_SIGNALS:
        assert df.loc[0, f"state_{k}"] == "GREEN"
    assert df.loc[0, "score_m"] == pytest.approx(0.55)
    assert pd.isna(df.loc[0, "prob_dd5_20"])          # Phase 2 전까지 비어 있음
    assert pd.isna(df.loc[0, "y_sign_20"])            # 결과 열은 backfill 이 채움
    # LF 줄바꿈
    assert b"\r\n" not in path.read_bytes()


def test_append_dedupes_on_asof_and_variant(tmp_path):
    path = tmp_path / "track_record.csv"
    assert ledger.append_today(_row("2025-03-03", variant="faithful"), path) is True
    assert ledger.append_today(_row("2025-03-03", variant="faithful", tone="buy"), path) is False   # 같은 날·같은 변형
    assert ledger.append_today(_row("2025-03-03", variant="completed"), path) is True             # 같은 날·다른 변형은 허용
    assert ledger.append_today(_row(pd.Timestamp("2025-03-04"), variant="faithful"), path) is True
    df = pd.read_csv(path)
    assert len(df) == 3
    assert df["asof"].tolist() == ["2025-03-03", "2025-03-03", "2025-03-04"]   # 정렬 유지


def test_append_accepts_flat_state_columns_and_rejects_bad_input(tmp_path):
    path = tmp_path / "t.csv"
    flat = {"asof": "2025-03-05", "variant": "completed", "tone": "caution",
            **{f"state_{k}": "RED" for k in V0_SIGNALS}, "score_d": -0.4}
    assert ledger.append_today(flat, path) is True
    df = pd.read_csv(path)
    assert df.loc[0, "state_btc"] == "RED"
    with pytest.raises(ValueError):
        ledger.append_today({"asof": "2025-03-06", "variant": "completed", "tone": "moon"}, path)  # 알 수 없는 톤
    with pytest.raises(ValueError):
        ledger.append_today({"variant": "completed", "tone": "buy"}, path)                      # asof 없음
    with pytest.warns(UserWarning):
        ledger.append_today({"asof": "2025-03-07", "tone": "buy"}, path)                        # variant·상태 누락 → 경고
    df = pd.read_csv(path)
    assert df[df["asof"] == "2025-03-07"]["variant"].iloc[0] == "faithful"


# ------------------------------------------------------------------
# backfill
# ------------------------------------------------------------------
def test_backfill_fills_only_matured_rows_with_correct_definitions(tmp_path):
    path = tmp_path / "t.csv"
    spy = _spy(200)
    idx = spy.index
    # 위치 10(60일 지남), 위치 150(20일만 지남), 위치 190(둘 다 안 지남), 인덱스에 없는 날
    for pos, tone in ((10, "hold"), (150, "caution"), (190, "buy")):
        ledger.append_today(_row(idx[pos], tone=tone), path)
    ledger.append_today(_row("2025-01-04", tone="buy"), path)   # 토요일 — spy 인덱스에 없음
    with pytest.warns(UserWarning, match="spy_close 에 없는 asof"):
        df = ledger.backfill(path, spy)
    df = df.set_index("asof")

    r = df.loc[idx[10].strftime("%Y-%m-%d")]
    c0 = spy.iloc[10]
    # 장부 CSV 는 소수 6자리로 저장되고 반환값도 같은 정밀도
    assert r["fwd_ret_20"] == pytest.approx(spy.iloc[30] / c0 - 1, abs=1e-6)
    assert r["fwd_ret_60"] == pytest.approx(spy.iloc[70] / c0 - 1, abs=1e-6)
    assert r["y_sign_20"] == float(spy.iloc[30] > c0)
    assert r["y_sign_60"] == float(spy.iloc[70] > c0)
    exp_dd = float(spy.iloc[11:31].min() / c0 - 1 <= -0.05)
    assert r["y_dd5_20"] == exp_dd

    r = df.loc[idx[150].strftime("%Y-%m-%d")]
    assert not pd.isna(r["fwd_ret_20"]) and not pd.isna(r["y_dd5_20"])
    assert pd.isna(r["fwd_ret_60"]) and pd.isna(r["y_sign_60"])     # 60일은 아직

    r = df.loc[idx[190].strftime("%Y-%m-%d")]
    assert pd.isna(r["fwd_ret_20"]) and pd.isna(r["y_dd5_20"]) and pd.isna(r["y_sign_20"])   # 미래 행 없음

    r = df.loc["2025-01-04"]
    assert pd.isna(r["fwd_ret_20"])

    # 파일에도 반영됐는지 + 재실행해도 값이 바뀌지 않는지(멱등)
    again = ledger.backfill(path, spy).set_index("asof")
    pd.testing.assert_frame_equal(again[ledger.OUTCOME_COLUMNS].astype(float), df[ledger.OUTCOME_COLUMNS].astype(float))


def test_backfill_dd5_true_on_crash(tmp_path):
    path = tmp_path / "t.csv"
    idx = pd.bdate_range("2025-01-02", periods=60)
    px = np.full(60, 100.0)
    px[15] = 94.0          # 위치 0 기준 15일 뒤 -6%
    spy = pd.Series(px, index=idx)
    ledger.append_today(_row(idx[0], tone="reduce"), path)
    ledger.append_today(_row(idx[20], tone="hold"), path)   # 이후 낙폭 없음
    df = ledger.backfill(path, spy).set_index("asof")
    assert df.loc[idx[0].strftime("%Y-%m-%d"), "y_dd5_20"] == 1.0
    assert df.loc[idx[20].strftime("%Y-%m-%d"), "y_dd5_20"] == 0.0
    assert df.loc[idx[20].strftime("%Y-%m-%d"), "y_sign_20"] == 0.0     # 100 → 100: 상승 아님


# ------------------------------------------------------------------
# summary
# ------------------------------------------------------------------
def test_summary_empty_and_missing_file(tmp_path):
    s = ledger.summary(tmp_path / "nope.csv")
    assert s["exists"] is False and s["n"] == 0
    assert any("없" in w for w in s["warnings"])


def test_summary_hits_baseline_and_missing_days(tmp_path):
    path = tmp_path / "t.csv"
    idx = pd.bdate_range("2025-01-06", periods=80)   # 월요일 시작
    px = np.linspace(100, 120, 80)                      # 단조 상승 → 항상 y_sign=1
    spy = pd.Series(px, index=idx)
    plan = {0: "buy", 1: "hold", 2: "caution", 4: "reduce", 5: "neutral"}   # 위치 3 은 결측(거래일인데 기록 없음)
    for pos, tone in plan.items():
        ledger.append_today(_row(idx[pos], tone=tone), path)
    ledger.backfill(path, spy)
    s = ledger.summary(path, trading_days=idx)   # 결측일 판정에 합성 거래일 사용
    assert s["n"] == 5 and s["n_with_outcome"] == 5 and s["n_with_outcome_60"] == 5
    assert s["first_asof"] == idx[0].strftime("%Y-%m-%d") and s["last_asof"] == idx[5].strftime("%Y-%m-%d")
    assert s["base_rate_20"] == pytest.approx(1.0)
    assert s["by_tone"]["buy"]["hit_20"] == pytest.approx(1.0)
    assert s["by_tone"]["caution"]["hit_20"] == pytest.approx(0.0)
    assert s["by_tone"]["reduce"]["hit_60"] == pytest.approx(0.0)
    assert s["overall"]["hit_20"] == pytest.approx(3 / 5)
    assert s["missing_days"] == [idx[3].strftime("%Y-%m-%d")]
    assert s["n_missing_days"] == 1 and s["missing_method"] == "trading_days 인자"
    assert s["variants"] == ["completed"]
    assert s["by_variant"]["completed"]["n"] == 5
    assert s["overall"]["dd5_rate"] == pytest.approx(0.0)
    # JSON 직렬화 가능
    import json
    json.dumps(s)


def test_summary_missing_days_fallback_without_calendar(tmp_path):
    """trading_days 인자 없이도 동작(calendar_us 가 없으면 평일 폴백 + 경고 기록)."""
    path = tmp_path / "t.csv"
    ledger.append_today(_row("2025-01-06", tone="buy"), path)
    ledger.append_today(_row("2025-01-08", tone="buy"), path)
    s = ledger.summary(path)
    assert "2025-01-07" in s["missing_days"]
    assert s["missing_method"] in ("mrl.calendar_us", "평일(휴장일 미반영)")


# ------------------------------------------------------------------
# report 스모크
# ------------------------------------------------------------------
def _synthetic_replay(n=300, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    tones = []
    cur = "hold"
    for _ in range(n):
        if rng.random() < 0.08:
            cur = TONES[rng.integers(len(TONES))]
        tones.append(cur)
    rep = pd.DataFrame({"tone": tones, "overall_d": "GREEN", "overall_w": "AMBER", "overall_m": "GREEN",
                        "score_d": rng.normal(0, 0.3, n)}, index=idx)
    spy = pd.Series(500 * np.cumprod(1 + rng.normal(0.0003, 0.011, n)), index=idx)
    fwd20 = spy.shift(-20) / spy - 1
    targets = pd.DataFrame({"fwd_ret_20": fwd20, "y_sign_20": (fwd20 > 0).astype(float).where(fwd20.notna())}, index=idx)
    return rep, spy, targets


def _minimal_summary():
    return {
        "variant": "faithful", "n_days": 300, "generated_at": "2026-09-07 00:00 UTC",
        "data_range": {"start": "2024-01-02", "end": "2025-03-14", "n_days": 300, "n_watch_avail": 27},
        "directional": [
            {"tone": "hold", "h": 20, "n": 150, "n_blocks": 7, "hit": 0.66, "base": 0.65, "ci_lo": 0.5, "ci_hi": 0.8,
             "fwd_mean": 0.012, "fwd_median": 0.015, "fwd_p10": -0.04, "fwd_p90": 0.06, "dd5_rate": 0.12},
            {"tone": "reduce", "h": 20, "n": 40, "n_blocks": 2, "hit": 0.4, "base": 0.6, "ci_lo": np.nan, "ci_hi": None,
             "fwd_mean": 0.004, "fwd_median": 0.01, "fwd_p10": -0.06, "fwd_p90": 0.05, "dd5_rate": 0.3},
            {"tone": "hold", "h": 60, "n": 150, "n_blocks": 2, "hit": 0.7, "base": 0.72},
        ],
        "episodes10": [{"peak_date": "2024-07-16", "trough_date": "2024-08-05", "depth": -0.085, "days_to_trough": 14,
                        "recovery_date": "2024-09-19", "days_to_recover": 46, "first_warn_date": "2024-07-24",
                        "lead_days": -6, "held_to_trough": True, "missed": False}],
        "episodes5": [],
        "episode_summary": {"n_episodes": 1, "n_detected": 1, "detection_rate": 1.0, "median_lead_days": -6.0,
                            "false_alarms_per_year": 2.5, "switches_per_year": 14.0, "median_run_days": 9},
        "allocation": {"allocation": {"cagr": 0.11, "maxdd": -0.09, "worst_month": -0.05, "switches_per_year": 14.0},
                       "buy_hold": {"cagr": 0.14, "maxdd": -0.12, "worst_month": -0.06, "switches_per_year": 0.0},
                       "cost_bps": 5},
        "switches": {"n_switches": 17, "switches_per_year": 14.0, "median_run_days": 9},
        "cells": [{"overall_m": "GREEN", "overall_w": "AMBER", "overall_d": "GREEN", "n": 120, "tone": "hold", "rule": 10},
                  {"overall_m": "RED", "overall_w": "RED", "overall_d": "RED", "n": 12, "tone": "reduce", "rule": 1}],
        "base_rates": {"y_sign_20": 0.65, "y_dd5_20": 0.14},
        "warnings": ["합성 자료 — 테스트용"],
        "honesty": "테스트 문구",
    }


def test_charts_for_returns_pngs_and_is_defensive():
    rep, spy, targets = _synthetic_replay()
    charts = report.charts_for(_minimal_summary(), rep, spy, targets)
    assert set(charts) == {"cumret", "tone_bands", "fwd_box", "switches"}
    for png in charts.values():
        assert isinstance(png, bytes) and png[:8] == b"\x89PNG\r\n\x1a\n"
    # tone 열 없음 → 예외 없이 빈 dict + 경고
    with pytest.warns(UserWarning):
        assert report.charts_for({}, pd.DataFrame(index=rep.index), spy, targets) == {}
    # targets 에 fwd_ret_20 없음 → fwd_box 만 빠짐
    with pytest.warns(UserWarning, match="fwd_ret_20"):
        c2 = report.charts_for({}, rep, spy, pd.DataFrame(index=rep.index))
    assert "fwd_box" not in c2 and "cumret" in c2
    # spy 없음 → cumret/tone_bands 빠짐, switches 는 남음
    with pytest.warns(UserWarning):
        c3 = report.charts_for({}, rep, None, targets)
    assert "switches" in c3 and "cumret" not in c3


def test_render_backtest_report_writes_all_sections(tmp_path):
    rep, spy, targets = _synthetic_replay()
    summary = _minimal_summary()
    charts = report.charts_for(summary, rep, spy, targets)
    out = tmp_path / "docs" / "backtest_v0.html"
    report.render_backtest_report(summary, out, charts)
    html = out.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    for i in range(1, 10):
        assert f'id="s{i}"' in html
    assert "정직한 전제" in html and "8,458거래일" in html
    assert report.HONESTY_BOLD in html.replace("&quot;", '"')
    assert "#0b1220" in html and "@import" not in html and "fonts.googleapis" not in html
    assert "data:image/png;base64," in html and html.count("<img") == 4
    assert "overflow-x:auto" in html and "viewport" in html
    assert "합성 자료 — 테스트용" in html          # 경고 목록
    assert "2024-07-16" in html                      # 에피소드 표
    assert "66.0%" in html and "65.0%" in html      # 적중률·기준선(백분율 포맷)
    assert "테스트 문구" in html
    assert b"\r\n" not in out.read_bytes()


def test_render_backtest_report_tolerates_empty_summary(tmp_path):
    out = tmp_path / "empty.html"
    report.render_backtest_report({}, out, {})
    html = out.read_text(encoding="utf-8")
    assert 'id="s9"' in html and "자료 없음" in html or "없음" in html
    # DataFrame 형태의 records 도 허용 + 알 수 없는 차트 이름은 ⑧에 표시
    df = pd.DataFrame([{"tone": "buy", "h": 20, "n": 5, "hit": 0.6, "base": 0.5, "weird_col": 1.5}])
    png = report.charts_for({}, _synthetic_replay(60)[0], None, None).get("switches")
    report.render_backtest_report({"directional": df, "episodes10": pd.DataFrame(), "cells": {"a": {"n": 3}}}, out, {"extra_chart": png})
    html = out.read_text(encoding="utf-8")
    assert "weird_col" in html and "추가 차트: extra_chart" in html


def test_render_index_both_variants(tmp_path):
    day = {"asof": "2026-09-04", "tone": "caution", "verdict_ko": "단기 과열 뒤 주춤", "action_ko": "새로 사는 건 잠시 쉬기",
           "overall_d": "G2R", "overall_w": "GREEN", "overall_m": "GREEN", "score_d": 0.51, "score_w": 0.62, "score_m": 0.7,
           "states_d": {k: "GREEN" for k in V0_SIGNALS}, "n_watch_avail": 28, "fg_avail": True, "eod_avail": False}
    today = {"asof": "2026-09-04", "market_status": "current", "generated_at": "2026-09-04 16:25 ET",
             "spy_close": 640.12, "vix_close": 15.3, "warnings": ["테스트 경고"],
             "faithful": day, "completed": {**day, "tone": "hold", "verdict_ko": "꾸준한 상승 흐름"}}
    path = tmp_path / "track_record.csv"
    ledger.append_today(_row("2026-09-03", tone="hold"), path)
    ls = ledger.summary(path, trading_days=pd.bdate_range("2026-09-01", "2026-09-04"))
    out = tmp_path / "index.html"
    report.render_index(today, ls, out)
    html = out.read_text(encoding="utf-8")
    assert "faithful" in html and "completed" in html
    assert "단기 과열 뒤 주춤" in html and "꾸준한 상승 흐름" in html
    assert "두 변형의 톤이 다릅니다" in html
    assert 'href="backtest_v0.html"' in html
    assert "2026-09-04 16:25 ET" in html
    assert "테스트 경고" in html and "마감봉 없음" in html
    assert report.HONESTY_LINE.split(".")[0][:10] in html
    assert "기록 일수" in html
    # 단일 v0_day dict 도 허용, 장부 요약이 비어도 동작
    report.render_index({**day, "variant": "completed"}, {}, out)
    assert "completed" in out.read_text(encoding="utf-8")


def test_tone_exposure_used_in_report_text(tmp_path):
    out = tmp_path / "r.html"
    report.render_backtest_report({"variant": "completed"}, out, {})
    html = out.read_text(encoding="utf-8")
    for t, e in TONE_EXPOSURE.items():
        assert f"{int(round(e * 100))}%" in html
    assert "completed" in html
