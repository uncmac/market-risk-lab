#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""전량 캐시 구축 (최초 1회 · 일요일 보정 작업).

사용:
    python scripts/build_cache.py                 # 가격 전량 재구축 + fg/eod/cboe append + meta.json
    python scripts/build_cache.py --update        # update_daily (가격 전량, fg/eod/cboe 는 새 날짜만)
    python scripts/build_cache.py --data-dir DIR  # 다른 디렉터리에 구축 (실험용)
    python scripts/build_cache.py --force         # 이전 캐시 대비 퇴행 검사(티커 소실·행 수 급감·SPY 일자 후퇴) 생략 —
                                                  # 원천이 정당하게 이력을 줄였을 때만 수동으로 쓴다

종료 코드: 0 성공, 1 실패(예외 — 퇴행한 다운로드도 여기 포함: 기존 캐시는 그대로 남는다). 경고는 meta.json["warnings"] 에
남고 여기서도 출력한다.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrl import data as D                     # noqa: E402
from mrl.config import DATA_DIR               # noqa: E402


def _utf8_stdout() -> None:
    """Windows 콘솔(cp1252)에서 한국어 출력이 깨지지 않도록."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _span(df, col=None) -> str:
    """'rows, first ~ last' 한 줄 요약."""
    if df is None or len(df) == 0:
        return "0행 (없음)"
    s = df[col].dropna() if col is not None else df
    if len(s) == 0:
        return "0행 (전부 NaN)"
    return f"{len(s):,}행, {s.index[0]:%Y-%m-%d} ~ {s.index[-1]:%Y-%m-%d}"


def summarize(bundle: D.Bundle, data_dir: Path) -> str:
    lines = ["=== 캐시 요약 ==="]
    lines.append(f"디렉터리: {data_dir}")
    m = bundle.meta
    lines.append(f"수집 시각: {m.get('fetched_at_utc')} (UTC) / {m.get('fetched_at_et')}  "
                 f"yfinance {m.get('yfinance_version')}  모드 {m.get('mode')}  {m.get('elapsed_sec')}초")
    lines.append(f"close   : {len(bundle.close):,}행 × {bundle.close.shape[1]}열 "
                 f"({bundle.close.index[0]:%Y-%m-%d} ~ {bundle.close.index[-1]:%Y-%m-%d})")
    for t in ("SPY", "^VIX", "BTC-USD"):
        lines.append(f"  {t:<8}: " + (_span(bundle.close, t) if t in bundle.close.columns else "열 없음"))
    lines.append(f"spy_ohlc: {_span(bundle.spy_ohlc)}")
    lines.append(f"cboe    : {_span(bundle.cboe)}  열 {list(bundle.cboe.columns)}")
    lines.append(f"fg      : {_span(bundle.fg)}  열 {len(bundle.fg.columns)}개")
    eod = bundle.eod
    n_half = int(eod["half_day"].sum()) if len(eod) else 0
    lines.append(f"eod     : {_span(eod)}  반일장 {n_half}일")
    if m.get("missing_tickers"):
        lines.append(f"누락 티커: {m['missing_tickers']}")
    lines.append(f"SPY 마지막 봉 완성: {m.get('spy_last_bar_complete')}")
    sizes = []
    for name in D.FILES.values():
        p = data_dir / name
        if p.exists():
            sizes.append(f"{name} {p.stat().st_size/1024:,.0f}KB")
    lines.append("파일: " + ", ".join(sizes))
    warns = m.get("warnings", [])
    lines.append(f"경고 {len(warns)}건:" if warns else "경고 없음")
    for w in warns:
        lines.append(f"  - {w}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser(description="market-risk-lab 데이터 캐시 구축")
    ap.add_argument("--update", action="store_true", help="update_daily 모드 (fg/eod/cboe 새 날짜만)")
    ap.add_argument("--data-dir", default=str(DATA_DIR), help=f"캐시 디렉터리 (기본 {DATA_DIR})")
    ap.add_argument("--force", action="store_true", help="이전 캐시 대비 퇴행 검사를 생략 (수동 재구축 전용)")
    args = ap.parse_args(argv)
    data_dir = Path(args.data_dir)

    t0 = time.time()
    print(f"[build_cache] {'update_daily' if args.update else 'build_cache'} → {data_dir}" + (" (--force)" if args.force else ""))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # 경고는 meta.warnings 로 모아 아래에서 출력한다
        bundle = (D.update_daily(data_dir, force=args.force) if args.update
                  else D.build_cache(data_dir, force=args.force))
    print(summarize(bundle, data_dir))
    print(f"[build_cache] 완료 {time.time() - t0:.1f}초")
    return 0


if __name__ == "__main__":
    sys.exit(main())
