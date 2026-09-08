# -*- coding: utf-8 -*-
"""스크립트 통합 테스트 — run_backtest_v0.py · daily.py 를 짧은 구간/오프라인으로 끝까지 돌려 산출물 계약을 확인한다.

* 실제 캐시(data/)를 읽기만 하고, 산출물은 tmp_path 에만 쓴다 (results/ · docs/ · track_record.csv 는 건드리지 않음).
* 검사: 변형별 summary JSON 이 엄격한 JSON(NaN/Infinity 없음)이고 base_rates·directional·episode_summary·allocation·switches
  키를 갖는지, HTML 이 'undefined'/'nan'/'None' 토큰 없이 섹션을 모두 갖는지, 장부에 기준일 행이 정확히 하나 생기는지.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrl.config import DATA_DIR  # noqa: E402

pytestmark = pytest.mark.skipif(not (DATA_DIR / "close.csv").exists(), reason="data/ 캐시 없음")

BAD_TOKEN = re.compile(r"\b(undefined|NaN|nan|None|null|NaT)\b")


def _strict_json(path: Path) -> dict:
    def _no_const(c):
        raise ValueError(f"엄격하지 않은 JSON 상수: {c}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_no_const)


def _walk_nan(o, path="") -> list[str]:
    bad: list[str] = []
    if isinstance(o, dict):
        for k, v in o.items():
            bad += _walk_nan(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            bad += _walk_nan(v, f"{path}[{i}]")
    elif isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        bad.append(path)
    elif isinstance(o, str) and o.strip().lower() in ("nan", "none", "inf", "-inf", "nat"):
        bad.append(f"{path}={o}")
    return bad


def _visible_text(html: str) -> str:
    t = re.sub(r"<style.*?</style>", "", html, flags=re.S)
    t = re.sub(r"<script.*?</script>", "", t, flags=re.S)
    return re.sub(r"<[^>]+>", " ", t)


def test_run_backtest_v0_short_window(tmp_path):
    """completed 변형 · 약 60거래일 · 검증 생략 → JSON 키·직렬화·HTML 섹션·CSV 열."""
    from scripts import run_backtest_v0 as RB
    results, docs = tmp_path / "results", tmp_path / "docs"
    rc = RB.main(["--variant", "completed", "--start", "2024-01-02", "--end", "2024-03-28",
                  "--results-dir", str(results), "--docs-dir", str(docs), "--verify", "0", "--no-progress"])
    assert rc == 0
    js = _strict_json(results / "summary_v0_completed.json")
    for k in ("base_rates", "directional", "episode_summary", "allocation", "switches",
              "scorecard", "episodes", "headline", "meta", "cells", "run", "data_range"):
        assert k in js, f"summary JSON 에 {k} 없음"
    assert _walk_nan(js) == []
    spy = pd.read_csv(DATA_DIR / "spy_ohlc.csv", index_col="date", parse_dates=["date"])
    n_expected = int(((spy.index >= "2024-01-02") & (spy.index <= "2024-03-28")).sum())
    assert 55 <= n_expected <= 65 and js["run"]["n_days"] == n_expected
    assert js["base_rates"]["full_history"]["y_sign_20"] == pytest.approx(0.654, abs=0.01)
    assert isinstance(js["directional"], list) and js["directional"]
    assert {"tone", "h", "n", "n_blocks", "hit", "base", "ci_lo", "ci_hi"} <= set(js["directional"][0])
    for k in ("detection_rate", "median_lead_days", "n_evaluable", "n_detected", "false_alarms_per_year", "switches_per_year"):
        assert k in js["episode_summary"]
    assert "switches_per_year" in js["switches"]
    assert {"cagr", "bh_cagr", "max_dd", "bh_max_dd"} <= set(js["allocation"])
    combined = _strict_json(results / "summary_v0.json")
    assert "completed" in combined["variants"]
    rep = pd.read_csv(results / "backtest_v0_completed.csv", index_col="date", parse_dates=["date"])
    assert len(rep) == n_expected and "tone" in rep.columns and rep["tone"].notna().all()
    for name in ("backtest_v0_completed.html", "backtest_v0.html"):
        html = (docs / name).read_text(encoding="utf-8")
        txt = _visible_text(html)
        assert not BAD_TOKEN.search(txt), f"{name}: 불량 토큰 {BAD_TOKEN.findall(txt)[:5]}"
        for s in ("①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨"):
            assert s in txt, f"{name}: 섹션 {s} 없음"
        assert html.count("data:image/png;base64,") >= 4


def test_run_backtest_v0_excludes_incomplete_last_bar(tmp_path):
    """캐시 meta.spy_last_bar_complete=False (장중에 갱신된 캐시) → 마지막 SPY 봉을 재현·목표변수·에피소드에서 제외하고
    리포트 ⑧ 에 캐시 경고를 싣는다. 명시적 --end 가 그 봉을 가리켜도 마지막 완성 세션으로 내린다."""
    import shutil
    from scripts import run_backtest_v0 as RB
    data = tmp_path / "data"
    shutil.copytree(DATA_DIR, data)
    meta = json.loads((data / "meta.json").read_text(encoding="utf-8"))
    meta["spy_last_bar_complete"] = False
    (data / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    spy = pd.read_csv(data / "spy_ohlc.csv", index_col="date", parse_dates=["date"])
    last, prev = spy.index[-1], spy.index[-2]
    start = spy.index[-6]
    results, docs = tmp_path / "results", tmp_path / "docs"
    rc = RB.main(["--variant", "faithful", "--start", f"{start:%Y-%m-%d}", "--end", f"{last:%Y-%m-%d}", "--data-dir", str(data),
                  "--results-dir", str(results), "--docs-dir", str(docs), "--verify", "0", "--no-progress"])
    assert rc == 0
    rep = pd.read_csv(results / "backtest_v0_faithful.csv", index_col="date", parse_dates=["date"])
    assert rep.index[-1] == prev and last not in rep.index and len(rep) == 5
    js = _strict_json(results / "summary_v0_faithful.json")
    assert js["run"]["end"] == prev.strftime("%Y-%m-%d")
    assert js["run"]["cache"]["spy_last_bar_complete"] is False
    assert js["run"]["cache"]["spy_last_used"] == prev.strftime("%Y-%m-%d")
    assert js["meta"]["spy_end"] == prev.strftime("%Y-%m-%d")            # 목표변수·에피소드용 종가도 같은 절단선
    assert any("장중 부분 봉" in w for w in js["report_warnings"])
    html = (docs / "backtest_v0_faithful.html").read_text(encoding="utf-8")
    assert "장중 부분 봉" in html and "캐시 경고" in html
    # --reuse-replay: 같은 날짜라도 종가가 다르면(재조정·부분 봉) 재사용하지 않는다
    csv = results / "backtest_v0_faithful.csv"
    rep.loc[rep.index[0], "spy_close"] = float(rep["spy_close"].iloc[0]) * 1.01
    rep.to_csv(csv, index_label="date", float_format="%.6f", lineterminator="\n", encoding="utf-8")
    import io as _io
    import contextlib
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = RB.main(["--variant", "faithful", "--start", f"{start:%Y-%m-%d}", "--data-dir", str(data), "--results-dir", str(results),
                      "--docs-dir", str(docs), "--verify", "0", "--no-progress", "--reuse-replay"])
    assert rc == 0 and "spy_close 가 현재 캐시 종가와 다름" in buf.getvalue()
    rep2 = pd.read_csv(csv, index_col="date", parse_dates=["date"])
    assert abs(float(rep2["spy_close"].iloc[0]) - float(spy.loc[start, "Close"])) < 1e-3   # 재현으로 되돌아옴


def test_daily_offline_writes_one_ledger_row_and_index(tmp_path):
    """--no-update · 금요일 장 마감 후 시각 고정 → 장부 1행(completed) + tone_faithful 열 + index.html."""
    from scripts import daily as DY
    ledger, docs = tmp_path / "track_record.csv", tmp_path / "docs"
    spy = pd.read_csv(DATA_DIR / "spy_ohlc.csv", index_col="date", parse_dates=["date"])
    last = spy.index[-1]
    now = f"{last:%Y-%m-%d} 17:00"
    rc = DY.main(["--no-update", "--now", now, "--ledger", str(ledger), "--docs-dir", str(docs)])
    assert rc == 0
    df = pd.read_csv(ledger, dtype=str)
    assert len(df) == 1 and df.loc[0, "asof"] == last.strftime("%Y-%m-%d") and df.loc[0, "variant"] == "completed"
    assert df.loc[0, "market_status"] == "current" and df.loc[0, "tone_faithful"] in ("buy", "hold", "neutral", "caution", "reduce")
    # 같은 날 재실행 → 행이 늘지 않는다
    assert DY.main(["--no-update", "--now", now, "--ledger", str(ledger), "--docs-dir", str(docs)]) == 0
    assert len(pd.read_csv(ledger)) == 1
    txt = _visible_text((docs / "index.html").read_text(encoding="utf-8"))
    assert not BAD_TOKEN.search(txt)
    for s in ("오늘 판정", "faithful", "completed", "장부 요약", "정직 문구", "자료"):
        assert s in txt
    assert 'href="backtest_v0.html"' in (docs / "index.html").read_text(encoding="utf-8")
