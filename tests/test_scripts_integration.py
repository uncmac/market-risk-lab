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
from datetime import timedelta
from pathlib import Path

import numpy as np
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


def test_reuse_replay_refuses_other_window_rule_or_signals_hash(tmp_path):
    """창 규칙 보호: summary JSON 의 run.window_rule / run.signals_v0_sha256 가 현재 코드와 같을 때만 --reuse-replay 가 CSV 를
    재사용한다. 다르거나(옛 504행 규칙으로 만든 벤치마크) JSON 이 없으면 재현을 다시 돌리고 현재 규칙을 다시 기록한다."""
    import contextlib
    import io
    from mrl import signals_v0 as S
    from scripts import run_backtest_v0 as RB
    spy = pd.read_csv(DATA_DIR / "spy_ohlc.csv", index_col="date", parse_dates=["date"])
    start, end = spy.index[-5], spy.index[-1]
    results, docs = tmp_path / "results", tmp_path / "docs"
    common = ["--variant", "faithful", "--start", f"{start:%Y-%m-%d}", "--end", f"{end:%Y-%m-%d}",
              "--results-dir", str(results), "--docs-dir", str(docs), "--verify", "0", "--no-progress"]
    assert RB.main(common) == 0
    jp = results / "summary_v0_faithful.json"
    js = _strict_json(jp)
    assert js["run"]["window_rule"] == S.WINDOW_RULE
    assert S.WINDOW_RULE.startswith("calendar|") and "spy=2y" in S.WINDOW_RULE and "watch=1y" in S.WINDOW_RULE
    assert re.fullmatch(r"[0-9a-f]{64}", js["run"]["signals_v0_sha256"]) and js["run"]["signals_v0_sha256"] == RB._signals_hash()
    csv_mtime = (results / "backtest_v0_faithful.csv").stat().st_mtime_ns

    def run_reuse() -> str:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert RB.main(common + ["--reuse-replay"]) == 0
        return buf.getvalue()

    # 1) 출처가 같으면 재사용 (CSV 를 다시 쓰지 않는다)
    out = run_reuse()
    assert "저장된 CSV 재사용" in out and (results / "backtest_v0_faithful.csv").stat().st_mtime_ns == csv_mtime
    # 2) 창 규칙이 다르면(옛 504행 고정 규칙) 재현 실행 → 현재 규칙을 다시 기록
    js["run"]["window_rule"] = "rows|spy=504,vix=504,fang=504,watch=252,btc=731"
    jp.write_text(json.dumps(js, ensure_ascii=False), encoding="utf-8")
    out = run_reuse()
    assert "창 규칙" in out and "저장된 CSV 재사용" not in out
    assert _strict_json(jp)["run"]["window_rule"] == S.WINDOW_RULE
    # 3) signals_v0.py 해시가 다르면 재현 실행
    js = _strict_json(jp)
    js["run"]["signals_v0_sha256"] = "0" * 64
    jp.write_text(json.dumps(js, ensure_ascii=False), encoding="utf-8")
    out = run_reuse()
    assert "해시" in out and "저장된 CSV 재사용" not in out
    assert _strict_json(jp)["run"]["signals_v0_sha256"] == RB._signals_hash()
    # 4) JSON 이 없으면 출처를 증명할 수 없다 → 재현 실행
    jp.unlink()
    out = run_reuse()
    assert "증명할 수 없어" in out and "저장된 CSV 재사용" not in out and jp.exists()


# ==================================================================
# Phase 2 — run_calibration.py · run_backtest_v1.py · daily.py (ARCHITECTURE_PHASE2.md §15)
# ==================================================================
P2_LEDGER_COLUMNS = ["prob_dd5_20", "p2_p_m1", "p2_p_m2", "p2_p_vix", "p2_p_vix_bgk", "p2_clim", "p2_lo", "p2_hi", "p2_band_src",
                     "p2_x_vix", "p2_x_har", "p2_x_ma", "p2_har_vol_20", "p2_har_fc_20", "p2_r", "p2_state", "p2_days_in_state",
                     "p2_tone_model", "p2_deploy_mode", "p2_d_vix", "p2_d_har", "p2_d_ma", "p2_d_refit", "p2_input_missing", "p2_model_id"]


@pytest.fixture(scope="module")
def calib_short(tmp_path_factory):
    """run_calibration.py --end 2006-12-29 --first-refit 2003-01-02 (tmp results/docs) — 모듈 안에서 한 번만."""
    from scripts import run_calibration as RC
    base = tmp_path_factory.mktemp("p2")
    results, docs = base / "results", base / "docs"
    rc = RC.main(["--end", "2006-12-29", "--first-refit", "2003-01-02", "--results-dir", str(results), "--docs-dir", str(docs)])
    assert rc == 0
    return {"results": results, "docs": docs}


def test_run_calibration_short_window_writes_all_artifacts(calib_short):
    """산출물 4개 + 엄격 JSON + 키 계약 + CSV 열 + 하드컷 + 파라미터 4개 + HTML ①~⑨."""
    from mrl import features as F
    from mrl.config import HOLDOUT_START
    results, docs = calib_short["results"], calib_short["docs"]
    for name in ("calib_p2_walkforward.csv", "summary_p2.json", "model_p2.json"):
        assert (results / name).exists(), name
    assert (docs / "calibration_p2.html").exists()
    js = _strict_json(results / "summary_p2.json")
    for k in ("run", "v0_reference", "ladder", "blocks24", "blocks18", "blocks_from1999", "reliability", "murphy", "era_auc",
              "acceptance", "params_by_refit", "har", "selftest", "warnings", "disclosure", "rungs", "live_model", "live_models",
              "ablations", "c_sensitivity", "determinism", "headline"):
        assert k in js, f"summary_p2.json 에 {k} 없음"
    assert _walk_nan(js) == []
    assert js["disclosure"] == "#2 사전 관측 참조"
    run = js["run"]
    assert run["end"] == "2006-12-29" and run["first_refit"] == "2003-01-02" and run["hard_cut"] is True
    assert run["spec_sha256"] == F.spec_sha256() and run["feature_rule"] == F.FEATURE_RULE
    assert run["param_count"] == 4 and run["n_refits"] == 4 and run["first_refit_rule_ok"] is True
    for k in ("python", "pandas", "numpy", "sklearn", "cache"):
        assert k in run
    oos = pd.read_csv(results / "calib_p2_walkforward.csv", index_col="date", parse_dates=["date"])
    assert list(oos.columns) == ["y", "clim", "p_vix", "p_vix_driftless", "p_vix_bgk", "p_m1", "p_m2", "p_m3", "p_m3_pk", "p_m3_har96",
                                 "p_v0ref", "har_fc_20", "refit_year", "block24", "block18"]
    assert oos.index[0] == pd.Timestamp("2003-01-02") and oos.index[-1] == pd.Timestamp("2006-12-29")
    assert (oos.index < pd.Timestamp(HOLDOUT_START)).all()
    assert oos["y"].iloc[-20:].isna().all() and oos["y"].iloc[:-20].notna().all()     # 라벨은 하드컷된 종가로만(마지막 20행 NaN)
    assert oos[["p_m1", "p_m2", "p_m3", "p_m3_pk", "p_m3_har96"]].notna().all().all()
    assert oos["p_v0ref"].isna().all()                                                # 2017 이전 → 참조선 없음
    assert set(oos["refit_year"].unique()) == {2003, 2004, 2005, 2006}
    # 모델: 정확히 4개 파라미터, spec 일치, 재적합일 = 하드컷 끝, deploy = acceptance 결과
    m = _strict_json(results / "model_p2.json")
    assert m["n_params"] == 4 and set(m["coef"]) == {"x_vix", "x_har", "x_ma"} and m["refit_date"] == "2006-12-29"
    assert m["spec_sha256"] == F.spec_sha256() and m["deploy_mode"] == js["acceptance"]["deploy_mode"]
    assert m["deploy_mode"] == "info_only" and m["tone_model"] is None            # 빈 블록이 있어 literal 은 통과할 수 없다
    assert js["acceptance"]["rule"] == "literal" and js["acceptance"]["literal"]["M3"]["n_blocks_empty"] == 9
    assert all(p["n_params"] == 4 for p in js["params_by_refit"] if p["rung"] == "M3")
    assert js["determinism"]["refit_twice_max_abs_diff"] == 0.0
    # HTML
    html = (docs / "calibration_p2.html").read_text(encoding="utf-8")
    txt = _visible_text(html)
    assert not BAD_TOKEN.search(txt), BAD_TOKEN.findall(txt)[:5]
    for i in range(1, 10):
        assert f'id="s{i}"' in html
    assert "v0 (동결 벤치마크" in html and run["spec_sha256"] in html
    assert b"\r\n" not in (docs / "calibration_p2.html").read_bytes()


def test_run_calibration_is_deterministic_and_gate_passes(calib_short):
    """같은 입력으로 두 번 → CSV 바이트 동일, 결정론 게이트 'compared' 통과. 계수를 틀어 심은 model_p2.json 은 exit 1(파일 불변)."""
    from scripts import run_calibration as RC
    results, docs = calib_short["results"], calib_short["docs"]
    before = (results / "calib_p2_walkforward.csv").read_bytes()
    model_before = (results / "model_p2.json").read_bytes()
    rc = RC.main(["--end", "2006-12-29", "--results-dir", str(results), "--docs-dir", str(docs), "--no-charts"])
    assert rc == 0
    assert (results / "calib_p2_walkforward.csv").read_bytes() == before
    js = _strict_json(results / "summary_p2.json")
    assert js["determinism"]["status"] == "compared" and js["determinism"]["ok"] is True and js["determinism"]["max_abs_diff"] == 0.0
    # 계수를 1e-6 틀어 저장 → 같은 입력 지문이므로 결정론 검사 실패 → exit 1, 파일은 그대로
    m = json.loads((results / "model_p2.json").read_text(encoding="utf-8"))
    m["coef"]["x_vix"] += 1e-6
    tampered = json.dumps(m, ensure_ascii=False, indent=2) + "\n"
    (results / "model_p2.json").write_text(tampered, encoding="utf-8")
    rc = RC.main(["--end", "2006-12-29", "--results-dir", str(results), "--docs-dir", str(docs), "--no-charts"])
    assert rc == 1
    assert (results / "model_p2.json").read_text(encoding="utf-8") == tampered
    (results / "model_p2.json").write_bytes(model_before)                       # 복구 (다른 테스트가 이 fixture 를 쓴다)


def test_run_calibration_refuses_holdout_without_final_flag_and_exit2_when_unlocked(tmp_path):
    """--end 가 HOLDOUT_START 이후면 SystemExit(1); --holdout-final 은 unlock 파일이 있으면 exit 2 (아무것도 쓰지 않음)."""
    from scripts import run_calibration as RC
    from mrl.config import HOLDOUT_START
    results, docs = tmp_path / "results", tmp_path / "docs"
    with pytest.raises(SystemExit) as ex:
        RC.main(["--end", HOLDOUT_START, "--results-dir", str(results), "--docs-dir", str(docs), "--no-charts"])
    assert ex.value.code == 1
    results.mkdir(parents=True, exist_ok=True)
    (results / "holdout_unlock.json").write_text('{"ledger_entry": "2b"}', encoding="utf-8")
    rc = RC.main(["--holdout-final", "--results-dir", str(results), "--docs-dir", str(docs)])
    assert rc == 2
    assert not (results / "summary_p2.json").exists() and not (results / "model_p2.json").exists()


def _fake_holdout_oos(n: int = 140, seed: int = 0) -> pd.DataFrame:
    """holdout_final_block 채점에 필요한 최소 OOS 프레임(합성) — 실제 홀드아웃을 쓰지 않고 순서를 검사하기 위한 것."""
    idx = pd.bdate_range("2024-09-03", periods=n)
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(float)
    def _p(scale):
        return np.clip(0.16 + scale * y + rng.normal(0, 0.02, n), 0.01, 0.99)
    return pd.DataFrame({"y": y, "clim": np.full(n, 0.16), "p_vix": _p(0.05), "p_vix_bgk": _p(0.05),
                         "p_m1": _p(0.08), "p_m2": _p(0.10), "p_m3": _p(0.12)}, index=idx)


def test_holdout_final_locks_before_disclosing_numbers(tmp_path, monkeypatch):
    """§6 1회 보장: unlock 기록은 홀드아웃 숫자를 로그에 찍기 **전에** 남는다.
    (실제 홀드아웃은 쓰지 않는다 — 채점 입력을 합성 프레임으로 갈아 끼운다.)"""
    from scripts import run_calibration as RC
    canonical = tmp_path / "canon" / "holdout_unlock.json"
    relocated = tmp_path / "results" / "holdout_unlock.json"
    monkeypatch.setattr(RC, "HOLDOUT_UNLOCK_PATH", canonical)
    monkeypatch.setattr(RC.L, "LEDGER", tmp_path / "no_ledger.csv")
    oos = _fake_holdout_oos()
    params = [{"rung": "M3", "refit_date": "2024-01-02"}, {"rung": "M3", "refit_date": "2025-01-02"},
              {"rung": "M1", "refit_date": "2025-01-02"}]
    monkeypatch.setattr(RC, "cut_bundle", lambda b, e: b)
    monkeypatch.setattr(RC, "prepare_inputs", lambda b, e: {"feats": oos, "y": oos["y"], "ohlc": None})
    monkeypatch.setattr(RC.C, "walk_forward", lambda *a, **k: (oos, params))
    locked = {}
    real_log = RC._log
    monkeypatch.setattr(RC, "_log", lambda m: (locked.setdefault("at_disclosure", relocated.exists() and canonical.exists())
                                               if m.startswith("[holdout] 2024-09-03~") else None, real_log(m))[-1])
    warns: list[str] = []
    out = RC.holdout_final_block(object(), pd.Timestamp("2026-09-04"), "2003-01-02", "s" * 64, warns, relocated)
    assert locked.get("at_disclosure") is True          # 숫자를 찍는 순간 이미 두 경로 모두 잠겨 있었다
    assert relocated.exists() and canonical.exists()    # 정본 경로에도 반드시 흔적이 남는다(--results-dir 로 숨길 수 없다)
    rec = json.loads(canonical.read_text(encoding="utf-8"))
    assert rec["ledger_entry"] == "2b" and rec["start"] == "2024-09-03" and rec["refits"] == ["2024-01-02", "2025-01-02"]
    assert rec["timestamp_utc"] and rec == json.loads(relocated.read_text(encoding="utf-8"))
    assert out["unlock"]["timestamp_utc"] == rec["timestamp_utc"]


def test_holdout_final_guard_uses_canonical_path_even_with_results_dir(tmp_path, monkeypatch):
    """--results-dir 로 옮겨도 1회 보장은 정본 경로가 지킨다 — 자료는 --data-dir 그대로이므로 우회가 되면 안 된다."""
    from scripts import run_calibration as RC
    canonical = tmp_path / "canon" / "holdout_unlock.json"
    canonical.parent.mkdir(parents=True)
    canonical.write_text('{"ledger_entry": "2b"}', encoding="utf-8")
    monkeypatch.setattr(RC, "HOLDOUT_UNLOCK_PATH", canonical)
    results, docs = tmp_path / "elsewhere", tmp_path / "elsewhere_docs"
    assert RC.main(["--holdout-final", "--results-dir", str(results), "--docs-dir", str(docs)]) == 2
    assert not results.exists() and not (results / "holdout_unlock.json").exists()


def test_live_fingerprint_tracks_post_holdout_bars(tmp_path):
    """결정론 게이트의 지문은 라이브 적합에 실제로 들어간 구간을 덮어야 한다.
    하드컷 지문만 쓰면 홀드아웃 이후 자료 수정이 '코드 비결정성' 으로 오진된다."""
    from scripts import run_calibration as RC
    ohlc = pd.read_csv(DATA_DIR / "spy_ohlc.csv", index_col="date", parse_dates=["date"])
    vix = pd.read_csv(DATA_DIR / "close.csv", index_col="date", parse_dates=["date"])["^VIX"].reindex(ohlc.index).ffill()
    cut, live_refit = pd.Timestamp("2024-08-30"), pd.Timestamp("2026-01-02")
    bumped = ohlc.copy()
    row = bumped.index[(bumped.index > cut) & (bumped.index < live_refit)][-40]      # 홀드아웃 이후·라이브 학습창 안
    bumped.loc[row, ["Open", "High", "Low", "Close"]] *= 1.003
    assert RC._data_sha256(ohlc.loc[:cut], vix.loc[:cut]) == RC._data_sha256(bumped.loc[:cut], vix.loc[:cut])
    assert RC._data_sha256(ohlc.loc[:live_refit], vix.loc[:live_refit]) != RC._data_sha256(bumped.loc[:live_refit], vix.loc[:live_refit])


def test_run_calibration_post_unlock_refits_live_on_full_data_and_fits_twice(calib_short, tmp_path):
    """holdout_unlock.json 이 있으면 라이브 모델은 전 자료·최근 1월 재적합이고(사전 등록 OOS 는 하드컷 그대로),
    지문은 그 라이브 구간의 것이며 '2회 적합' 은 실제로 두 번 적합한 결과다."""
    import shutil
    from scripts import run_calibration as RC
    results, docs = tmp_path / "results", tmp_path / "docs"
    shutil.copytree(calib_short["results"], results)
    (results / "model_p2.json").unlink()                       # 재적합일이 바뀌므로 새로 쓴다
    (results / "holdout_unlock.json").write_text('{"ledger_entry": "2b", "note": "test stub"}', encoding="utf-8")
    assert RC.main(["--end", "2006-12-29", "--results-dir", str(results), "--docs-dir", str(docs), "--no-charts"]) == 0
    js = _strict_json(results / "summary_p2.json")
    det, run = js["determinism"], js["run"]
    assert det["live_mode"] == "post_unlock_annual" and run["live_mode"] == "post_unlock_annual"
    assert det["refit_twice_max_abs_diff"] == 0.0 and det["ok"] is True
    assert det["data_sha256"] == run["live_data_sha256"] != run["data_sha256"] == det["hard_cut_data_sha256"]
    m = _strict_json(results / "model_p2.json")
    assert m["extra"]["hard_cut_data_sha256"] == run["data_sha256"] and m["extra"]["data_sha256"] == run["live_data_sha256"]
    assert m["refit_date"] > "2006-12-29"                      # 하드컷 끝이 아니라 최근 1월 첫 거래일
    assert js["run"]["end"] == "2006-12-29" and js["headline"]["oos_end"] == "2006-12-29"   # OOS 채점은 하드컷 그대로


def test_weekly_refit_anchor_is_the_first_sunday_after_the_years_first_session():
    """홀드아웃 해제 후 가드의 기준점은 '1월 첫 거래일 이후의 첫 일요일' 이다.
    달력 첫 일요일을 쓰면 1월 1일이 금·토·일인 해(2027·2028·2033·2034 …)에 주간 작업이 정상 실행됐는데도
    daily 가 그 주 내내 죽는다."""
    from datetime import date
    from scripts import daily as DY
    idx = pd.DatetimeIndex(sorted(set(pd.bdate_range("2026-12-01", "2040-12-31")) - {pd.Timestamp("2027-01-01")}))
    # 2027-01-01 은 금요일 휴장 → 첫 거래일 월 01-04, 그 뒤 첫 일요일 01-10 (달력 첫 일요일은 01-03)
    assert DY._weekly_refit_anchor(idx, 2027) == date(2027, 1, 10)
    for d in pd.bdate_range("2027-01-04", "2027-01-08"):        # 주간 작업이 아직 새해 세션을 못 본 주 → 가드 미발동
        assert not (d.date() > DY._weekly_refit_anchor(idx, 2027))
    assert date(2027, 1, 11) > DY._weekly_refit_anchor(idx, 2027)   # 그 다음 주 월요일부터는 진짜 누락
    # 1월 1일이 평일(월~목)인 해는 예전 규칙과 같다
    for year in (2029, 2030, 2031, 2032):
        first = idx[(idx.year == year) & (idx.month == 1)][0].date()
        assert DY._weekly_refit_anchor(idx, year) == first + timedelta(days=(6 - first.weekday()) % 7)
    assert DY._weekly_refit_anchor(idx, 2050) is None            # 그 해 1월 세션이 없으면 가드를 적용하지 않는다


def test_run_backtest_v1_completes_on_short_calibration(calib_short):
    """run_backtest_v1.py --all-configs 가 짧은 보정 산출물로 완주: CSV 열·JSON 키·민감도 3종·HTML ①~⑦."""
    from scripts import run_backtest_v1 as RB1
    results, docs = calib_short["results"], calib_short["docs"]
    rc = RB1.main(["--all-configs", "--results-dir", str(results), "--docs-dir", str(docs)])
    assert rc == 0
    df = pd.read_csv(results / "backtest_v1.csv", index_col="date", parse_dates=["date"])
    assert list(df.columns) == ["p", "clim", "r", "state", "days_in_state", "tone", "changed"]
    assert len(df) == 1007 and set(df["state"].unique()) <= {"normal", "caution", "reduce"}
    assert set(df["tone"].unique()) <= {"hold", "caution", "reduce"} and (df["days_in_state"] >= 1).all()
    js = _strict_json(results / "summary_v1.json")
    for k in ("run", "kpis", "allocation", "episodes", "sensitivities", "flags", "warnings", "v0_reference", "deploy_mode", "directional"):
        assert k in js
    assert _walk_nan(js) == []
    assert js["deploy_mode"] == "info_only" and js["run"]["config"] == "default"
    assert set(js["sensitivities"]) == {"wide", "symmetric_dwell", "no_dwell"}
    kp = js["kpis"]
    for k in ("switches_per_year", "occupancy", "dd5_rate_by_state", "true_alarm_share", "max_changes_any_5_sessions", "kpi_ceiling_ok"):
        assert k in kp
    assert kp["max_changes_any_5_sessions"] <= 3 and {"cagr", "bh_cagr", "max_dd", "bh_max_dd"} <= set(js["allocation"])
    html = (docs / "backtest_v1.html").read_text(encoding="utf-8")
    txt = _visible_text(html)
    assert not BAD_TOKEN.search(txt), BAD_TOKEN.findall(txt)[:5]
    for i in range(1, 8):
        assert f'id="s{i}"' in html
    assert "v0 (동결 벤치마크" in html and "wide" in html and "no_dwell" in html
    # spec 불일치 → exit 1
    sp = json.loads((results / "summary_p2.json").read_text(encoding="utf-8"))
    sp["run"]["spec_sha256"] = "0" * 64
    (results / "summary_p2.json").write_text(json.dumps(sp, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SystemExit) as ex:
        RB1.main(["--results-dir", str(results), "--docs-dir", str(docs), "--no-charts"])
    assert ex.value.code == 1


def _calib_copy(calib_short, tmp_path, acceptance: dict) -> Path:
    """calib_short 산출물을 tmp 로 복사하고 acceptance 만 갈아 끼운다(spec 은 현재 코드에 맞춰 되돌린다 —
    앞선 테스트가 spec 을 훼손해도 이 테스트는 그것과 무관하게 배치 규칙만 본다)."""
    import shutil
    from mrl import features as F
    dst = tmp_path / "results"
    shutil.copytree(calib_short["results"], dst)
    sp = json.loads((dst / "summary_p2.json").read_text(encoding="utf-8"))
    sp["run"]["spec_sha256"], sp["run"]["feature_rule"] = F.spec_sha256(), F.FEATURE_RULE
    sp["acceptance"] = {**(sp.get("acceptance") or {}), **acceptance}
    (dst / "summary_p2.json").write_text(json.dumps(sp, ensure_ascii=False), encoding="utf-8")
    return dst


@pytest.mark.parametrize("tone_model,expect_col,info_rung", [("M1", "p_m1", "M3"), ("M3", "p_m3", "M1")])
def test_run_backtest_v1_headline_layer_follows_accepted_tone_model(calib_short, tmp_path, tone_model, expect_col, info_rung):
    """헤드라인 결정층은 acceptance 가 배치한 단의 확률로 돌아간다 — CSV 의 p, summary_v1 의 kpis/allocation/flags,
    리포트 ②~⑤ 가 모두 그 단이고, 배포되지 않은 단은 info_layers/⑥ 에 '정보 표시(배포 안 함)' 로만 남는다."""
    from scripts import run_backtest_v1 as RB1
    results = _calib_copy(calib_short, tmp_path, {"deploy_mode": "tones", "tone_model": tone_model, "rule": "amended"})
    docs = tmp_path / "docs"
    assert RB1.main(["--results-dir", str(results), "--docs-dir", str(docs), "--no-charts"]) == 0
    oos = pd.read_csv(results / "calib_p2_walkforward.csv", index_col="date", parse_dates=["date"])
    df = pd.read_csv(results / "backtest_v1.csv", index_col="date", parse_dates=["date"])
    other = "p_m3" if expect_col == "p_m1" else "p_m1"
    assert (df["p"].to_numpy() == pytest.approx(oos[expect_col].reindex(df.index).to_numpy(), abs=1e-12))
    assert not np.allclose(df["p"].to_numpy(), oos[other].reindex(df.index).to_numpy())    # 두 단이 실제로 다르다
    js = _strict_json(results / "summary_v1.json")
    assert js["deployed"] == {"prob_rung": tone_model, "prob_column": expect_col, "deployed": True, "label": "배포",
                              "deploy_mode": "tones", "tone_model": tone_model, "source": "summary_p2.json:acceptance"}
    assert js["run"]["prob_rung"] == tone_model and js["run"]["prob_column"] == expect_col
    assert set(js["info_layers"]) == {info_rung} and js["info_layers"][info_rung]["role"] == "정보 표시(배포 안 함)"
    assert (js["m1_layer"] is None) is (tone_model == "M1")          # M1 이 배포되면 M1 은 정보층이 아니다
    # 헤드라인 KPI == 배포 단 결정층의 상태 계열 (정보층의 것이 아니다)
    from mrl import decision as DEC
    st = DEC.run(oos[expect_col].reindex(df.index), oos["clim"].reindex(df.index), DEC.config_from_name("default"))
    assert (st["state"].to_numpy() == df["state"].to_numpy()).all()
    assert js["kpis"]["n_changes"] == int(df["changed"].astype(bool).sum())
    assert js["info_layers"][info_rung]["prob_column"] == ("p_m1" if info_rung == "M1" else "p_m3")
    # (짧은 2003~06 창에서는 두 단의 상태 경로가 우연히 같을 수 있다 — 층의 정체는 위의 p 열·상태 계열로 확인한다)
    html = (docs / "backtest_v1.html").read_text(encoding="utf-8")
    txt = _visible_text(html)
    assert not BAD_TOKEN.search(txt), BAD_TOKEN.findall(txt)[:5]
    assert f"헤드라인 <b>{tone_model}</b> 배포" in html and "정보 표시(배포 안 함)" in txt
    assert f"② 결정층 KPI — {tone_model} 배포" in txt and 'id="s6"' in html and 'id="s8"' in html
    for i in range(1, 9):
        assert f'id="s{i}"' in html


def test_run_backtest_v1_info_only_labels_display_layer_and_claims_no_tone(calib_short, tmp_path):
    """배치된 단이 없으면(info_only) 생산 모델 M3 를 '정보 표시(배포 안 함)' 로 싣고 톤·비중을 주장하지 않는다."""
    from scripts import run_backtest_v1 as RB1
    results = _calib_copy(calib_short, tmp_path, {"deploy_mode": "info_only", "tone_model": None, "rule": "literal"})
    docs = tmp_path / "docs"
    assert RB1.main(["--results-dir", str(results), "--docs-dir", str(docs), "--no-charts"]) == 0
    oos = pd.read_csv(results / "calib_p2_walkforward.csv", index_col="date", parse_dates=["date"])
    df = pd.read_csv(results / "backtest_v1.csv", index_col="date", parse_dates=["date"])
    assert df["p"].to_numpy() == pytest.approx(oos["p_m3"].reindex(df.index).to_numpy(), abs=1e-12)   # 오늘 동작 유지
    js = _strict_json(results / "summary_v1.json")
    assert js["deployed"]["prob_rung"] == "M3" and js["deployed"]["deployed"] is False
    assert js["deployed"]["label"] == "정보 표시(배포 안 함)" and js["deploy_mode"] == "info_only"
    assert set(js["info_layers"]) == {"M1"} and js["m1_layer"] is not None
    html = (docs / "backtest_v1.html").read_text(encoding="utf-8")
    txt = _visible_text(html)
    assert "정보 표시(배포 안 함)" in txt and "배치된 단이 없어" in txt and "톤·비중 주장이 아닙니다" in txt
    assert "헤드라인 <b>M3</b> 정보 표시(배포 안 함)" in html
    # 배치된 단이 없는 페이지에서 "배포 단" 을 주장하지 않는다(⑥ 해설·⑦ 캡션·범례)
    assert "배포된 단은 <b>M3</b>" not in html and "배포 단(M3)" not in txt
    assert "배포된 단이 없어(deploy info_only) 맨 윗줄 <b>M3</b>" in html
    assert not BAD_TOKEN.search(txt), BAD_TOKEN.findall(txt)[:5]


def test_run_backtest_v1_refuses_when_accepted_rung_column_missing(calib_short, tmp_path):
    """배치된 단의 확률 열이 CSV 에 없으면 exit 1 — 배포되지 않은 M3 로 조용히 대체하지 않는다."""
    from scripts import run_backtest_v1 as RB1
    results = _calib_copy(calib_short, tmp_path, {"deploy_mode": "tones", "tone_model": "M2", "rule": "amended"})
    oos = pd.read_csv(results / "calib_p2_walkforward.csv", index_col="date", parse_dates=["date"])
    oos.drop(columns=["p_m2"]).to_csv(results / "calib_p2_walkforward.csv", index_label="date", lineterminator="\n", encoding="utf-8")
    with pytest.raises(SystemExit) as ex:
        RB1.main(["--results-dir", str(results), "--docs-dir", str(tmp_path / "docs"), "--no-charts"])
    assert ex.value.code == 1


def test_daily_writes_p2_ledger_columns_and_card(tmp_path):
    """daily.py --no-update 가 장부 P2 열(계약 §12)과 index.html 의 p2 카드를 쓴다. 재실행은 행을 늘리지 않는다."""
    from scripts import daily as DY
    from mrl.config import RESULTS_DIR
    if not (RESULTS_DIR / "model_p2.json").exists():
        pytest.skip("results/model_p2.json 없음 (run_calibration.py 먼저)")
    ledger, docs = tmp_path / "track_record.csv", tmp_path / "docs"
    spy = pd.read_csv(DATA_DIR / "spy_ohlc.csv", index_col="date", parse_dates=["date"])
    last = spy.index[-1]
    now = f"{last:%Y-%m-%d} 17:00"
    assert DY.main(["--no-update", "--now", now, "--ledger", str(ledger), "--docs-dir", str(docs)]) == 0
    df = pd.read_csv(ledger, dtype=str)
    assert len(df) == 1 and df.loc[0, "variant"] == "completed"
    for c in P2_LEDGER_COLUMNS:
        assert c in df.columns, c
    row = df.loc[0]
    p = float(row["prob_dd5_20"])
    assert 0.0 <= p <= 1.0 and float(row["p2_lo"]) <= p <= float(row["p2_hi"]) and row["p2_band_src"] in ("param", "calib")
    assert row["p2_state"] in ("normal", "caution", "reduce") and int(float(row["p2_days_in_state"])) >= 1
    assert row["p2_deploy_mode"] in ("info_only", "tones") and row["p2_model_id"].startswith("p2m3-")
    assert pd.isna(row["p2_input_missing"]) or row["p2_input_missing"] == ""
    d = sum(float(row[c]) for c in ("p2_d_vix", "p2_d_har", "p2_d_ma", "p2_d_refit"))
    assert abs(d) < 1.0                                                              # 일간 귀속 4항의 합 = Δp (확률 단위)
    assert abs(float(row["p2_r"]) - p / float(row["p2_clim"])) < 1e-4          # CSV 는 소수 6자리(ledger._write) → r 오차 ~1e-5
    # 재실행 → 행 불변
    assert DY.main(["--no-update", "--now", now, "--ledger", str(ledger), "--docs-dir", str(docs)]) == 0
    assert len(pd.read_csv(ledger)) == 1
    html = (docs / "index.html").read_text(encoding="utf-8")
    txt = _visible_text(html)
    assert not BAD_TOKEN.search(txt), BAD_TOKEN.findall(txt)[:5]
    assert 'id="p2"' in html and "다음 20거래일 안에 -5% 하락할 확률" in html and "100일 중 약" in html
    assert 'href="calibration_p2.html"' in html and 'href="backtest_v1.html"' in html
    assert html.index("오늘 판정") < html.index('id="p2"')                          # v0 판정 블록 아래


def _logit(rung: str, spec: str = "a" * 64, clim: float = 0.17, refit: str = "2024-08-30"):
    """합성 LogitModel — 사다리 단별 계수(값은 임의, 단끼리 다르기만 하면 된다)."""
    from mrl import model as M
    coef = {"M1": {"x_vix": 0.9}, "M2": {"x_vix": 0.8, "x_har": 0.7}, "M3": {"x_vix": 0.7, "x_har": 0.6, "x_ma": -2.3}}[rung]
    return M.LogitModel(features=tuple(coef), coef=coef, intercept=-1.0, refit_date=refit, train_start="1993-10-14",
                        train_end="2024-08-01", n_train=7754, n_pos=1329, clim=clim, feature_rule="p2|test",
                        spec_sha256=spec, model_id=f"p2{rung.lower()}-{spec[:8]}-{refit}")


def test_resolve_deployment_follows_acceptance_not_the_production_model():
    """배포 모델 결정 = summary_p2.json.acceptance. tone_model=M1 → M1 라이브 계수, M3 → 생산 모델,
    info_only → 배포 없음(M3 를 정보로). model_p2.json 과 어긋나면 acceptance 를 따르되 경고한다."""
    from scripts import daily as DY
    m3 = _logit("M3")
    live = {"M1": _logit("M1").to_dict(), "M2": _logit("M2").to_dict(), "M3": m3.to_dict()}
    w: list[str] = []
    dep = DY.resolve_deployment({"deploy_mode": "tones", "tone_model": "M1"}, m3, live, w)
    assert dep["prob_rung"] == "M1" and dep["deployed"] is True and dep["model"].model_id.startswith("p2m1-")
    assert dep["model"].features == ("x_vix",) and dep["source"] == "summary_p2.json:acceptance"
    assert any("model_p2.json 의 배치" in x for x in w)          # 생산 모델은 info_only/None 이라 불일치 경고
    w2: list[str] = []
    dep3 = DY.resolve_deployment({"deploy_mode": "tones", "tone_model": "M3"}, m3, live, w2)
    assert dep3["prob_rung"] == "M3" and dep3["model"] is m3 and dep3["deployed"] is True
    w3: list[str] = []
    dep0 = DY.resolve_deployment({"deploy_mode": "info_only", "tone_model": None}, m3, live, w3)
    assert dep0["prob_rung"] == "M3" and dep0["model"] is m3 and dep0["deployed"] is False and dep0["tone_model"] is None
    # 모순(tones 인데 tone_model 없음) → 배포 없음으로 내리고 경고
    w4: list[str] = []
    dep4 = DY.resolve_deployment({"deploy_mode": "tones", "tone_model": None}, m3, live, w4)
    assert dep4["deploy_mode"] == "info_only" and dep4["prob_rung"] == "M3" and any("tone_model 이 비어" in x for x in w4)
    # 배치된 단의 라이브 계수가 없으면 P2Fatal — M3 로 조용히 대체하지 않는다
    with pytest.raises(DY.P2Fatal, match="M1"):
        DY.resolve_deployment({"deploy_mode": "tones", "tone_model": "M1"}, m3, {"M2": live["M2"]}, [])
    with pytest.raises(DY.P2Fatal, match="spec_sha256"):
        DY.resolve_deployment({"deploy_mode": "tones", "tone_model": "M1"}, m3, {"M1": _logit("M1", spec="b" * 64).to_dict()}, [])
    # acceptance 가 없으면 model_p2.json 으로 내려가되 경고
    w5: list[str] = []
    m_dep = _logit("M3")
    m_dep.deploy_mode, m_dep.tone_model = "tones", "M1"
    dep5 = DY.resolve_deployment(None, m_dep, live, w5)
    assert dep5["prob_rung"] == "M1" and dep5["source"] == "results/model_p2.json" and any("acceptance 가 없어" in x for x in w5)


@pytest.mark.parametrize("acc,rung,prefix,deployed", [
    ({"deploy_mode": "tones", "tone_model": "M1"}, "M1", "p2m1-", True),
    ({"deploy_mode": "tones", "tone_model": "M3"}, "M3", "p2m3-", True),
    ({"deploy_mode": "info_only", "tone_model": None}, "M3", "p2m3-", False),
])
def test_daily_ledger_and_card_use_deployed_probability(tmp_path, acc, rung, prefix, deployed):
    """장부의 prob_dd5_20·p2_r·p2_state 와 카드 헤드라인은 배치된 단의 확률이다. 배포되지 않은 단은 사다리 열에 정보로만.
    info_only 면 톤·비중을 주장하지 않고 '정보 표시(배포 안 함)' 라벨을 단다."""
    import shutil
    from scripts import daily as DY
    from mrl.config import RESULTS_DIR
    if not (RESULTS_DIR / "model_p2.json").exists() or not (RESULTS_DIR / "summary_p2.json").exists():
        pytest.skip("results/model_p2.json · summary_p2.json 없음 (run_calibration.py 먼저)")
    results = tmp_path / "results"
    results.mkdir()
    for name in ("model_p2.json", "summary_p2.json"):
        shutil.copy(RESULTS_DIR / name, results / name)
    sp = json.loads((results / "summary_p2.json").read_text(encoding="utf-8"))
    sp["acceptance"] = {**(sp.get("acceptance") or {}), **acc}
    (results / "summary_p2.json").write_text(json.dumps(sp, ensure_ascii=False), encoding="utf-8")
    ledger, docs = tmp_path / "track_record.csv", tmp_path / "docs"
    spy = pd.read_csv(DATA_DIR / "spy_ohlc.csv", index_col="date", parse_dates=["date"])
    now = f"{spy.index[-1]:%Y-%m-%d} 17:00"
    assert DY.main(["--no-update", "--now", now, "--ledger", str(ledger), "--docs-dir", str(docs), "--results-dir", str(results)]) == 0
    row = pd.read_csv(ledger).iloc[0]
    p, p_m1, p_m3 = float(row["prob_dd5_20"]), float(row["p2_p_m1"]), float(row["p2_p_m3"])
    assert p == pytest.approx({"M1": p_m1, "M3": p_m3}[rung], abs=1e-9)          # 배포 단의 확률
    assert p != pytest.approx({"M1": p_m3, "M3": p_m1}[rung], abs=1e-6)          # 다른 단의 확률이 아니다
    assert str(row["p2_prob_model_id"]).startswith(prefix) and str(row["p2_model_id"]).startswith("p2m3-")
    assert row["p2_deploy_mode"] == acc["deploy_mode"]
    assert (pd.isna(row["p2_tone_model"]) if acc["tone_model"] is None else row["p2_tone_model"] == acc["tone_model"])
    assert float(row["p2_r"]) == pytest.approx(p / float(row["p2_clim"]), abs=1e-4)
    d_sum = sum(float(row[c]) for c in ("p2_d_vix", "p2_d_har", "p2_d_ma", "p2_d_refit"))
    assert math.isfinite(d_sum) and abs(d_sum) < 1.0                              # 배포 모델 기준 4항 합 = Δp
    if rung == "M1":
        assert float(row["p2_d_har"]) == 0.0 and float(row["p2_d_ma"]) == 0.0     # M1 에는 그 항이 없다(구조적 0)
    html = (docs / "index.html").read_text(encoding="utf-8")
    txt = _visible_text(html)
    assert not BAD_TOKEN.search(txt), BAD_TOKEN.findall(txt)[:5]
    i0 = html.index('id="p2"')                                                   # v0 판정·장부 블록은 그대로 두고 P2 카드만 본다
    card = html[i0:html.index("<section", i0)]
    assert f"이런 날 {report_nat_freq(p)}" in card                                # 헤드라인 = 배포 확률
    if deployed:
        assert f"톤 적용 — {rung} 배포" in card and "주식 비중" in card
        if rung != "M3":
            assert "정보 표시(배포 안 함)" in card                                 # 배포되지 않은 M3 는 정보 라벨
    else:
        assert "정보 표시(배포 안 함)" in card and "주식 비중" not in card
        assert 'class="pill"' not in card and "배포된 단이 없습니다" in _visible_text(card)


def report_nat_freq(p: float) -> str:
    from mrl import report as RPT
    return RPT._nat_freq(p)


def test_selftest_p2_section_real_cache_and_stale_spec(tmp_path, capsys):
    """selftest.py 의 Phase 2 절(§13): 실캐시·실 results 에서 FAIL 0 (ratio_checks 경계·특징 꼬리·spec·PARAM_COUNT·이벤트 표·장부 P2 열),
    spec 을 틀어 심은 model_p2.json 은 FAIL, 모델이 없으면 WARN(FAIL 아님)."""
    from scripts import selftest as ST
    from mrl.config import RESULTS_DIR
    from mrl import model as M
    ledger = RESULTS_DIR / "track_record.csv"
    rep = ST.run(DATA_DIR, max_age_days=10 ** 6)                  # 최근성 검사는 날짜에 좌우되므로 여기서는 끈다
    assert rep.n_fail == 0, [ln for ln in rep.lines if ln.startswith("[FAIL]")]
    ST.run_p2(rep, DATA_DIR, RESULTS_DIR, ledger)
    fails = [ln for ln in rep.lines if ln.startswith("[FAIL]")]
    assert not fails, fails
    txt = "\n".join(rep.lines)
    for key in ("(GK+OV)/CC rolling-250", "Parkinson/CC RV20", "특징 표 마지막 세션", "이벤트 표 잔여"):
        assert key in txt, key
    if (RESULTS_DIR / "model_p2.json").exists():
        assert f"PARAM_COUNT {M.PARAM_COUNT}" in txt and "spec_sha256" in txt
        # spec 불일치 → FAIL (daily.py 의 exit 1 조건을 미리 잡는다)
        stale = tmp_path / "stale_results"
        stale.mkdir()
        m = json.loads((RESULTS_DIR / "model_p2.json").read_text(encoding="utf-8"))
        m["spec_sha256"] = "f" * 64
        (stale / "model_p2.json").write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
        rep2 = ST.Report()
        ST.run_p2(rep2, DATA_DIR, stale, tmp_path / "no_ledger.csv")
        assert rep2.n_fail >= 1 and any("spec_sha256" in ln and ln.startswith("[FAIL]") for ln in rep2.lines)
    # 모델 없음 → WARN (Phase 2 미실행 상태는 계약 위반이 아니다)
    empty = tmp_path / "empty_results"
    empty.mkdir()
    rep3 = ST.Report()
    ST.run_p2(rep3, DATA_DIR, empty, tmp_path / "no_ledger.csv")
    assert rep3.n_fail == 0 and any("model_p2.json 없음" in ln for ln in rep3.lines)
    # CLI 종료 코드: --no-p2 · 기본 모두 0 (실캐시가 오래됐어도 FAIL 로 잡히지 않게 max-age 를 크게)
    assert ST.main(["--max-age-days", str(10 ** 6), "--no-p2"]) == 0
    assert ST.main(["--max-age-days", str(10 ** 6), "--results-dir", str(RESULTS_DIR), "--ledger", str(ledger)]) == 0
    out = capsys.readouterr().out
    assert "결과: FAIL 0" in out


def test_daily_fails_without_model_or_with_stale_spec(tmp_path):
    """model_p2.json 없음 → exit 1 (장부·페이지 쓰지 않음); spec_sha256 불일치 → exit 1."""
    from scripts import daily as DY
    from mrl.config import RESULTS_DIR
    spy = pd.read_csv(DATA_DIR / "spy_ohlc.csv", index_col="date", parse_dates=["date"])
    now = f"{spy.index[-1]:%Y-%m-%d} 17:00"
    empty = tmp_path / "empty_results"
    empty.mkdir()
    ledger, docs = tmp_path / "track_record.csv", tmp_path / "docs"
    assert DY.main(["--no-update", "--now", now, "--ledger", str(ledger), "--docs-dir", str(docs), "--results-dir", str(empty)]) == 1
    assert not ledger.exists() and not (docs / "index.html").exists()
    src = RESULTS_DIR / "model_p2.json"
    if not src.exists():
        pytest.skip("results/model_p2.json 없음")
    stale = tmp_path / "stale_results"
    stale.mkdir()
    m = json.loads(src.read_text(encoding="utf-8"))
    m["spec_sha256"] = "f" * 64
    (stale / "model_p2.json").write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    assert DY.main(["--no-update", "--now", now, "--ledger", str(ledger), "--docs-dir", str(docs), "--results-dir", str(stale)]) == 1
    assert not ledger.exists()
