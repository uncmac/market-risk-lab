# -*- coding: utf-8 -*-
"""mrl/replay.py · scripts/run_backtest_v0.py · scripts/daily.py 스모크 테스트.

* 실캐시(data/)를 쓴다 — 캐시가 없으면 skip 하지 않고 실패한다(재현이 곧 이 플랫폼의 목적이므로).
* 40거래일 재현이 예외 없이 돌고 열 계약을 만족하는가, 축소 번들(_WindowSlicer)이 전체 번들과 비트 동일한가
  (30일 표본 × 두 변형), 125칸 점유표, 스크립트 두 개가 임시 디렉터리에 모든 산출물을 쓰는가.
* results/·docs/ 의 실제 산출물은 건드리지 않는다 (tmp_path 만 사용).
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrl import config as C                                   # noqa: E402
from mrl import replay as R                                   # noqa: E402
from mrl import signals_v0 as S                               # noqa: E402
from mrl.data import load_cache                               # noqa: E402

N_SMOKE_DAYS = 40
SMOKE_START = "2025-02-03"


def _load_script(name: str):
    """scripts/<name>.py 를 모듈로 import (scripts 는 패키지가 아님)."""
    spec = importlib.util.spec_from_file_location(f"script_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bundle():
    missing = [n for n in ("close.csv", "spy_ohlc.csv", "fg_history.csv", "spy_eod.csv", "meta.json")
               if not (C.DATA_DIR / n).exists()]
    if missing:
        pytest.fail(f"캐시 파일 없음 {missing} — scripts/build_cache.py 를 먼저 실행하라")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_cache(C.DATA_DIR)


@pytest.fixture(scope="module")
def smoke_days(bundle) -> pd.DatetimeIndex:
    idx = bundle.spy_ohlc.index
    days = idx[idx >= SMOKE_START][:N_SMOKE_DAYS]
    assert len(days) == N_SMOKE_DAYS
    return days


@pytest.fixture(scope="module")
def replay_f(bundle, smoke_days) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return R.replay_v0(bundle, smoke_days[0], smoke_days[-1], variant="faithful", basket="v0", progress=False)


@pytest.fixture(scope="module")
def replay_c(bundle, smoke_days) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return R.replay_v0(bundle, smoke_days[0], smoke_days[-1], variant="completed", basket="v0", progress=False)


# ------------------------------------------------------------------
# replay_v0 — 열 계약
# ------------------------------------------------------------------
def test_replay_40_days_column_contract(replay_f, smoke_days):
    df = replay_f
    assert len(df) == N_SMOKE_DAYS
    assert df.index.equals(smoke_days) and df.index.name == "date"
    assert df.index.is_unique and df.index.is_monotonic_increasing and df.index.tz is None
    # 계약 열이 이 순서로 맨 앞에 온다; 부가 열은 그 뒤
    assert list(df.columns[: len(R.REPLAY_COLUMNS)]) == R.REPLAY_COLUMNS
    assert list(df.columns[len(R.REPLAY_COLUMNS):]) == R.EXTRA_COLUMNS
    expected = ([f"state_{k}" for k in C.V0_SIGNALS] + ["score_d", "score_w", "score_m", "trend_d", "trend_w", "trend_m",
                "overall_d", "overall_w", "overall_m", "tone", "verdict_ko", "n_watch_avail", "fg_avail", "eod_avail", "n_leaders"])
    assert R.REPLAY_COLUMNS == expected
    assert not df[R.REPLAY_COLUMNS].isna().any().any()
    # 값 도메인
    for k in C.V0_SIGNALS:
        assert set(df[f"state_{k}"].unique()) <= set(C.V0_SC), k
    for x in ("d", "w", "m"):
        assert set(df[f"overall_{x}"].unique()) <= set(C.V0_SC)
        assert set(df[f"trend_{x}"].unique()) <= {-1, 0, 1}
        assert df[f"score_{x}"].between(-1.0, 1.0).all()
        assert df[f"score_{x}"].dtype == float and df[f"trend_{x}"].dtype.kind == "i"
    assert set(df["tone"].unique()) <= set(C.TONES)
    assert df["fg_avail"].dtype == bool and df["eod_avail"].dtype == bool
    assert df["fg_avail"].all()                       # 2025 년: F&G 이력 존재
    assert (df["n_watch_avail"] >= 20).all() and df["n_watch_avail"].dtype.kind == "i"
    assert (df["n_leaders"] >= 0).all() and df["n_leaders"].dtype.kind == "i"
    assert (df["spy_close"] > 0).all() and (df["vix_close"] > 0).all()
    # attrs
    assert df.attrs["variant"] == "faithful" and df.attrs["basket"] == "v0" and df.attrs["n_days"] == N_SMOKE_DAYS
    assert df.attrs["runtime_sec"] > 0 and df.attrs["ms_per_day"] > 0
    assert isinstance(df.attrs["warnings"], dict)


def test_replay_rows_equal_v0_day(bundle, replay_f):
    """재현 행은 v0_day(전체 번들) 결과와 완전히 같다 (표본 5일 직접 비교 + verify_replay)."""
    for ts in replay_f.index[[0, 7, 19, 31, 39]]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = S.v0_day(bundle, ts, variant="faithful", basket="v0")
        row = replay_f.loc[ts]
        for k in C.V0_SIGNALS:
            assert row[f"state_{k}"] == out["states_d"][k]
        for x in ("d", "w", "m"):
            assert row[f"score_{x}"] == out[f"score_{x}"]
            assert row[f"trend_{x}"] == out[f"trend_{x}"] and row[f"overall_{x}"] == out[f"overall_{x}"]
        assert row["tone"] == out["tone"] and row["verdict_ko"] == out["verdict_ko"]
        assert row["n_leaders"] == len(out["leaders"]) and row["n_watch_avail"] == out["n_watch_avail"]
        assert bool(row["fg_avail"]) == out["fg_avail"] and bool(row["eod_avail"]) == out["eod_avail"]
        assert row["spy_close"] == out["spy_close"] and row["vix_close"] == out["vix_close"]
    v = R.verify_replay(bundle, replay_f, n_samples=10, seed=1)
    assert v["n_checked"] == 10 and len(v["dates"]) == 10


def test_completed_variant_shares_daily_block(replay_f, replay_c, smoke_days):
    """completed 는 주/월 재표집만 다르다 → 일간 상태·일간 점수는 faithful 과 같고, 주/월은 다를 수 있다."""
    assert list(replay_c.columns) == list(replay_f.columns) and replay_c.index.equals(smoke_days)
    assert replay_c.attrs["variant"] == "completed"
    for k in C.V0_SIGNALS:
        assert (replay_c[f"state_{k}"] == replay_f[f"state_{k}"]).all(), k
    assert (replay_c["score_d"] == replay_f["score_d"]).all()
    assert (replay_c["fg_avail"] == replay_f["fg_avail"]).all() and (replay_c["n_leaders"] == replay_f["n_leaders"]).all()
    # 40 일 안에 부분 주/월이 있으므로 주간 점수는 어딘가 달라야 한다 (부분 봉 문제의 존재)
    assert (replay_c["score_w"] != replay_f["score_w"]).any()


# ------------------------------------------------------------------
# 축소 번들의 동일성 — 2015~2026 표본 30일 × 두 변형
# ------------------------------------------------------------------
def test_day_bundle_identity_30_sampled_days(bundle):
    idx = bundle.spy_ohlc.index
    days = idx[idx >= C.BACKTEST_START]
    rng = np.random.default_rng(0)
    picks = sorted(rng.choice(len(days), size=30, replace=False).tolist())
    checked = 0
    for p in picks:
        ts = days[p]
        sub = R.day_bundle(bundle, ts)
        # 점 원칙: 축소 번들에 asof 이후 행이 없다
        assert sub.close.index.max() <= ts and sub.spy_ohlc.index.max() == ts
        # 축소 번들은 달력 창 (asof-2y, asof] 의 상위집합이되 훨씬 크지 않다 (SPY 2y 는 500~507거래일)
        assert sub.spy_ohlc.index[0] >= ts - S.WINDOW_SPAN["spy"] and len(sub.spy_ohlc) <= 510
        assert len(sub.close) < len(bundle.close)
        for v in S.VARIANTS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                full = R._row_from_day(S.v0_day(bundle, ts, variant=v))
                small = R._row_from_day(S.v0_day(sub, ts, variant=v))
            assert full == small, (ts, v)      # float 은 == 로 비트 동일, 나머지는 값 동일
            checked += 1
    assert checked == 60


# ------------------------------------------------------------------
# 점유표
# ------------------------------------------------------------------
def test_cell_table(replay_f):
    ct = R.cell_table(replay_f)
    assert list(ct.columns) == R.CELL_COLUMNS and len(ct) == 125
    assert int(ct["n"].sum()) == len(replay_f) and math.isclose(float(ct["share"].sum()), 1.0)
    assert ct["n"].is_monotonic_decreasing
    assert set(ct["tone"]) <= set(C.TONES) and ct["rule"].between(0, 10).all()
    combos = {tuple(r) for r in ct[["overall_m", "overall_w", "overall_d"]].itertuples(index=False)}
    assert len(combos) == 125
    observed = replay_f.groupby(["overall_m", "overall_w", "overall_d"]).size()
    assert int((ct["n"] > 0).sum()) == len(observed)
    for (mo, wk, dy), n in observed.items():
        row = ct[(ct.overall_m == mo) & (ct.overall_w == wk) & (ct.overall_d == dy)].iloc[0]
        assert row["n"] == n
        name, _a, tone = S.v0_combo(mo, wk, dy)
        assert row["tone"] == tone and row["verdict_ko"] == name
        assert row["rule"] == R.RULE_NAMES_KO.index(name) + 1 if name in R.RULE_NAMES_KO else row["rule"] == 0
    # 국면명은 규칙 10개 + 폴백 3개뿐
    assert set(ct["verdict_ko"]) <= (set(R.RULE_NAMES_KO) | set(R.FALLBACK_NAMES_KO))
    assert len(set(ct["verdict_ko"])) == len(R.RULE_NAMES_KO) + len(R.FALLBACK_NAMES_KO)
    # 톤이 v0_combo 와 어긋난 replay 는 거부 (재현 결과 손상 탐지)
    bad = replay_f.copy()
    bad.iloc[0, bad.columns.get_loc("tone")] = "buy" if bad["tone"].iloc[0] != "buy" else "reduce"
    with pytest.raises(ValueError):
        R.cell_table(bad)
    empty = R.cell_table(replay_f.iloc[0:0])
    assert len(empty) == 125 and int(empty["n"].sum()) == 0


# ------------------------------------------------------------------
# 오류 경로
# ------------------------------------------------------------------
def test_replay_error_paths(bundle):
    with pytest.raises(ValueError):
        R.replay_v0(bundle, "2025-02-03", "2025-02-10", variant="partial", progress=False)
    with pytest.raises(ValueError):
        R.replay_v0(bundle, "2025-02-03", "2025-02-10", basket="cap", progress=False)
    with pytest.raises(ValueError):
        R.replay_v0(bundle, "2025-03-01", "2025-02-01", progress=False)
    with pytest.raises(ValueError):
        R.replay_v0(bundle, "1990-01-01", "1990-02-01", progress=False)
    with pytest.raises(ValueError):
        R.resolve_range(bundle.spy_ohlc.index, "2025-02-01", "2099-01-01")
    days = R.resolve_range(bundle.spy_ohlc.index, "2025-02-01", "2025-02-09")
    assert days[0] == pd.Timestamp("2025-02-03") and days[-1] == pd.Timestamp("2025-02-07")
    # BTC 이력 이전(2010): 하루라도 v0_day 가 실패하면 날짜를 붙여 RuntimeError
    with pytest.raises(RuntimeError, match="2010-01-04"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            R.replay_v0(bundle, "2010-01-04", "2010-01-06", progress=False)
    with pytest.raises(RuntimeError):
        R.verify_replay(bundle, pd.DataFrame({"tone": ["buy"]}, index=pd.DatetimeIndex(["2025-02-03"])).assign(
            **{c: 0 for c in R.REPLAY_COLUMNS if c != "tone"}, spy_close=0.0, vix_close=0.0), n_samples=1)


# ------------------------------------------------------------------
# scripts/run_backtest_v0.py — 짧은 구간에 모든 산출물
# ------------------------------------------------------------------
def test_run_backtest_script_short_range(tmp_path):
    mod = _load_script("run_backtest_v0")
    results, docs = tmp_path / "results", tmp_path / "docs"
    rc = mod.main(["--variant", "both", "--start", "2025-01-02", "--end", "2025-03-31",
                   "--results-dir", str(results), "--docs-dir", str(docs), "--verify", "3", "--no-progress"])
    assert rc == 0
    for v in S.VARIANTS:
        csv = results / f"backtest_v0_{v}.csv"
        js = results / f"summary_v0_{v}.json"
        html = docs / f"backtest_v0_{v}.html"
        assert csv.exists() and js.exists() and html.exists()
        df = pd.read_csv(csv, index_col="date", parse_dates=["date"])
        assert list(df.columns) == R.REPLAY_COLUMNS + R.EXTRA_COLUMNS and len(df) == 60
        assert df.index[0] == pd.Timestamp("2025-01-02") and df.index[-1] == pd.Timestamp("2025-03-31")
        with open(js, encoding="utf-8") as f:
            s = json.load(f)
        assert s["variant"] == v and s["run"]["n_days"] == 60 and s["run"]["verify"]["n_checked"] == 3
        for k in ("meta", "base_rates", "scorecard", "episodes", "allocation", "switching", "headline", "cells", "artifacts"):
            assert k in s, k
        assert set(s["episodes"]) == {"5", "10", "20"} and len(s["cells"]) == 125
        assert s["run"]["n_episodes_1993"] == {"5": 36, "10": 12, "20": 4}      # VALIDATION.md §2 감사 실측치
        assert s["headline"]["baseline_20"] is not None
        page = html.read_text(encoding="utf-8")
        assert page.startswith("<!doctype html>") and 'id="s9"' in page and "data:image/png;base64," in page
        assert "2025-01-02" in page and v in page
        assert b"\r\n" not in html.read_bytes()
    both = docs / "backtest_v0.html"
    assert both.exists()
    page = both.read_text(encoding="utf-8")
    assert "faithful" in page and "completed" in page and "일치도" in page and page.count("data:image/png;base64,") == 8
    assert 'href="backtest_v0_faithful.html"' in page and 'href="backtest_v0_completed.html"' in page
    with open(results / "summary_v0.json", encoding="utf-8") as f:
        comb = json.load(f)
    assert set(comb["variants"]) == set(S.VARIANTS)
    # 재사용 경로: 재현 없이 평가·리포트만 다시 (검증은 CSV 반올림 허용)
    rc = mod.main(["--variant", "faithful", "--start", "2025-01-02", "--end", "2025-03-31", "--results-dir", str(results),
                   "--docs-dir", str(docs), "--verify", "2", "--no-progress", "--reuse-replay"])
    assert rc == 0
    single = (docs / "backtest_v0.html").read_text(encoding="utf-8")
    assert 'id="s9"' in single                      # 변형 하나면 그 변형의 전체 리포트가 backtest_v0.html


# ------------------------------------------------------------------
# scripts/daily.py — 오프라인, 시각 고정
# ------------------------------------------------------------------
def test_daily_script_incomplete_then_current(tmp_path, bundle):
    mod = _load_script("daily")
    ledger = tmp_path / "results" / "track_record.csv"
    docs = tmp_path / "docs"
    last = bundle.spy_ohlc.index[-1]
    prev = bundle.spy_ohlc.index[-2]
    # 1) 장중(12:00 ET) → 오늘 봉 미완성 → 전일 기준으로 기록
    rc = mod.main(["--no-update", "--now", f"{last:%Y-%m-%d} 12:00", "--ledger", str(ledger), "--docs-dir", str(docs)])
    assert rc == 0
    df = pd.read_csv(ledger)
    assert len(df) == 1 and df["asof"].iloc[0] == prev.strftime("%Y-%m-%d") and df["variant"].iloc[0] == "completed"
    assert df["tone_faithful"].iloc[0] in C.TONES and df["market_status"].iloc[0] == "incomplete"
    assert set(["y_sign_20", "y_sign_60", "y_dd5_20", "fwd_ret_20", "fwd_ret_60"]) <= set(df.columns)
    assert df["y_sign_20"].isna().all()               # 아직 20일이 지나지 않음 → 채우지 않는다
    html = (docs / "index.html").read_text(encoding="utf-8")
    assert "미완성" in html and prev.strftime("%Y-%m-%d") in html and "faithful" in html and "completed" in html
    # 2) 마감 후(17:00 ET) → 오늘 기준 기록
    rc = mod.main(["--no-update", "--now", f"{last:%Y-%m-%d} 17:00", "--ledger", str(ledger), "--docs-dir", str(docs)])
    assert rc == 0
    df = pd.read_csv(ledger)
    assert len(df) == 2 and df["asof"].iloc[-1] == last.strftime("%Y-%m-%d") and df["market_status"].iloc[-1] == "current"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expect = S.v0_day(bundle, last, "completed")
    assert df["tone"].iloc[-1] == expect["tone"] and abs(df["spy_close"].iloc[-1] - expect["spy_close"]) < 1e-5
    # 3) 같은 날 다시 실행 → 중복 추가 없음 (변형 표기·상태 열은 그대로)
    rc = mod.main(["--no-update", "--now", f"{last:%Y-%m-%d} 18:00", "--ledger", str(ledger), "--docs-dir", str(docs)])
    assert rc == 0 and len(pd.read_csv(ledger)) == 2
    # 4) 상태 판정 단위 검사
    idx = bundle.spy_ohlc.index
    asof, st = mod.determine_asof(idx, mod.parse_now(f"{last:%Y-%m-%d} 16:04"))
    assert st == "incomplete" and asof == prev
    asof, st = mod.determine_asof(idx, mod.parse_now(f"{last:%Y-%m-%d} 16:06"))
    assert st == "current" and asof == last
    sat = last + pd.Timedelta(days=(5 - last.weekday()) % 7 or 7)
    asof, st = mod.determine_asof(idx, mod.parse_now(f"{sat:%Y-%m-%d} 10:00"))
    assert st == "weekend" and asof == last
