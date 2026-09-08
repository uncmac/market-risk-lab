#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""reference fetch_real() 의 실제 라이브 페이로드를 한 번 기록해 tests/fixtures/fetch_real_<date>.json 으로 남긴다.

목적: tests/test_parity.py::test_window_reproduces_recorded_fetch_real 이 mrl.signals_v0.window() 를
      포팅 자체의 상수가 아니라 **실제 Yahoo/CNN 응답의 모양**(행 수·첫/마지막 일자·열·EOD 세션·F&G 값)과 비교하게 한다.
사용: python tests/fixtures/record_fetch_real.py     (네트워크 필요; 값이 아니라 창의 모양만 저장한다)

기록 내용 (프레임별): rows, first, last(raw — 유령 행 포함 그대로), columns / watch 는 열별 {rows, first, last} /
eod 는 세션 일자 목록과 bool 값·수익률(%) / fg 는 라이브 score·d_daily·d_weekly·d_monthly.
fetch_news / fetch_shorts 는 창 재현과 무관하고 느리므로 빈 목록으로 대체한다.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REF_PATH = ROOT / "reference" / "market_dashboard_v0.py"
OUT_DIR = Path(__file__).resolve().parent


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _frame(obj) -> dict:
    return {"rows": int(len(obj)), "first": _d(obj.index[0]), "last": _d(obj.index[-1])}


def main() -> int:
    spec = importlib.util.spec_from_file_location("market_dashboard_v0_record", REF_PATH)
    mod = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        spec.loader.exec_module(mod)
    mod.fetch_news = lambda limit=5: []          # 창 재현과 무관 (느리고 흔들림)
    mod.fetch_shorts = lambda: []
    fetched_at = dt.datetime.now(dt.timezone.utc)
    data = mod.fetch_real()
    import yfinance
    watch = data["watch"]
    fx = {
        "recorded_at_utc": fetched_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "yfinance_version": getattr(yfinance, "__version__", "?"),
        "period": mod.CONFIG["period"], "watch_period": "1y", "eod_period": "30d",
        "spy": _frame(data["spy"]),
        "vix": _frame(data["vix"]),
        "btc": _frame(data["btc"]),
        "fang": {**_frame(data["fang"]), "columns": [str(c) for c in data["fang"].columns]},
        "watch": {"columns": [str(c) for c in watch.columns], **_frame(watch),
                  "per_column": {str(c): _frame(watch[c].dropna()) for c in watch.columns}},
        "eod": {"dates": [_d(x) for x in data["eod"].index],
                "values": [bool(v) for v in data["eod"].values],
                "ret_pct": [round(float(v), 6) for v in data["eod_vals"].values]},
        "fg": (None if data["fg"] is None else
               {k: (int(data["fg"][k]) if data["fg"][k] is not None else None)
                for k in ("score", "d_daily", "d_weekly", "d_monthly")}),
    }
    out = OUT_DIR / f"fetch_real_{fetched_at:%Y-%m-%d}.json"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(fx, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"저장: {out}")
    print(json.dumps({k: v for k, v in fx.items() if k not in ("eod", "watch")}, ensure_ascii=False, indent=1))
    print("watch per_column rows:", {k: v["rows"] for k, v in fx["watch"]["per_column"].items()})
    print("eod:", fx["eod"]["dates"][0], "..", fx["eod"]["dates"][-1], len(fx["eod"]["dates"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
