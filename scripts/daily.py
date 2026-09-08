#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""매일 판정: update_daily → apply_guards → 완성 봉 기준일 결정 → v0_day(faithful·completed) → 장부 → docs/index.html

사용:
    python scripts/daily.py                       # 캐시 갱신(네트워크) 후 오늘 판정 기록
    python scripts/daily.py --no-update           # 캐시 갱신 없이(오프라인) 현재 캐시로 판정 (테스트·재생성용)
    python scripts/daily.py --now "2026-09-04 12:00"   # ET 시각을 고정 (테스트: 장중이면 '미완성' 경로)

흐름 (ARCHITECTURE.md 스크립트 절)
  1. bundle = update_daily()(또는 load_cache) → apply_guards (멱등).
  2. asof = SPY 마지막 일자. 지금(ET)이 그 세션 마감(16:05 ET) 전이면 상태 'incomplete' 로 표시하고 **직전 완성 세션**을
     asof 로 쓴다. 오늘 봉이 없으면 주말/휴장/개장 전/데이터 지연으로 구분해 표시하고 마지막 완성 세션을 쓴다.
  3. v0_day 를 두 변형으로 계산. 장부에는 variant='completed' 행이 정식 기록이며, faithful 톤은 같은 행의
     추가 열 `tone_faithful` 에 둔다 (장부 모듈은 추가 열을 보존한다). 같은 날이 이미 있으면 추가하지 않는다.
  4. ledger.backfill — 결과 열은 **완성 세션까지의 SPY 종가**로만 채운다(미완성 봉 사용 금지).
  5. report.render_index → docs/index.html.
  6. GitHub Actions 안이면($GITHUB_OUTPUT) asof / market_status / appended 를 스텝 출력으로 내보낸다 — daily.yml 은
     market_status == 'current' 인 실행만 게이트가 인식하는 제목(`daily: <날짜>`)으로 커밋하고, 그 밖(incomplete/pre_open/
     stale/…)은 `daily(<상태>): <날짜>` 로 커밋해 예약 실행이 그날을 건너뛰지 않게 한다.
종료 코드: 0 성공, 1 실패(예외 — 판정이 없으면 페이지를 만들지 않는다: 조용한 실패 금지).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from datetime import datetime, time as dtime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrl import calendar_us as CAL                                      # noqa: E402
from mrl import data as D                                               # noqa: E402
from mrl import ledger as L                                             # noqa: E402
from mrl import report as RPT                                           # noqa: E402
from mrl import signals_v0 as S                                         # noqa: E402
from mrl.config import DATA_DIR, DOCS_DIR, ET                           # noqa: E402

MARKET_OPEN = dtime(9, 30)
STATUS_NOTE = {
    "current": "",
    "incomplete": "장 마감 전 실행 — 오늘 봉이 아직 미완성이라 전일(마지막 완성 세션) 기준으로 판정·기록",
    "weekend": "주말 휴장 — 직전 영업일 종가 기준",
    "holiday": "휴장일 — 직전 영업일 종가 기준",
    "pre_open": "개장 전 — 전일 종가 기준",
    "stale": "개장 시간인데 오늘자 봉이 없음 → 데이터 지연으로 간주, 직전 영업일 종가 기준 (조용히 넘기지 않고 경고)",
}


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _log(msg: str) -> None:
    print(msg, flush=True)


def _messages(caught) -> list[str]:
    """catch_warnings(record=True) 로 잡은 경고 → 문구 목록(순서 유지·중복 제거).
    ResourceWarning('unclosed database …')은 yfinance 내부 sqlite 캐시가 GC 될 때 나는 소음이라 제외한다 —
    데이터·계약 경고(UserWarning 등)는 전부 남긴다(조용한 실패 금지)."""
    out: list[str] = []
    for w in caught:
        if issubclass(w.category, ResourceWarning):
            continue
        m = str(w.message)
        if m not in out:
            out.append(m)
    return out


def parse_now(s: str | None) -> datetime:
    """--now 'YYYY-MM-DD HH:MM' (ET, tz-naive) → ET aware datetime. 없으면 현재 ET."""
    tz = ZoneInfo(ET)
    if not s:
        return datetime.now(tz)
    ts = pd.Timestamp(s)
    if pd.isna(ts):
        raise ValueError(f"--now 해석 불가: {s!r}")
    dt = ts.to_pydatetime()
    return dt.replace(tzinfo=tz) if dt.tzinfo is None else dt.astimezone(tz)


def determine_asof(spy_idx: pd.DatetimeIndex, now_et: datetime) -> tuple[pd.Timestamp, str]:
    """(기준일, 상태). 상태 ∈ current | incomplete | weekend | holiday | pre_open | stale.

    * 마지막 SPY 봉이 오늘(ET) 것이면: 16:05 ET 를 지났으면 current, 아니면 incomplete(→ 직전 세션).
    * 오늘 봉이 없으면 사유를 구분하되 기준일은 마지막(완성) 세션.
    """
    if len(spy_idx) < 2:
        raise ValueError("SPY 인덱스에 세션이 2개 미만 — 캐시 손상")
    last = spy_idx[-1]
    today = now_et.date()
    if last.date() >= today:
        if CAL.session_complete(now_et, last):
            return last, "current"
        return spy_idx[-2], "incomplete"
    if today.weekday() >= 5:
        return last, "weekend"
    if not CAL.is_trading_day(today):
        return last, "holiday"
    if now_et.time() < MARKET_OPEN:
        return last, "pre_open"
    return last, "stale"


def _set_extra_columns(path: Path, asof: str, variant: str, extras: dict) -> None:
    """장부의 (asof, variant) 행에 추가 열을 기록한다 (ledger 는 계약 외 열을 보존한다)."""
    df = L._read(path)
    mask = (df["asof"] == asof) & (df["variant"] == variant)
    if int(mask.sum()) != 1:
        raise RuntimeError(f"장부에서 ({asof}, {variant}) 행을 하나로 찾지 못함: {int(mask.sum())}건")
    for col, val in extras.items():
        if col not in df.columns or not pd.api.types.is_object_dtype(df[col]):
            # 문자열 열: 전부 결측이면 float64 로 읽히므로 object 로 맞춘 뒤 기록 (dtype 경고 방지)
            df[col] = (df[col] if col in df.columns else pd.Series(np.nan, index=df.index)).astype(object)
        df.loc[mask, col] = val
    L._write(df, path)


def run(args) -> dict:
    t0 = time.perf_counter()
    now_et = parse_now(args.now)
    now_utc = now_et.astimezone(timezone.utc)
    data_dir, docs_dir, ledger_path = Path(args.data_dir), Path(args.docs_dir), Path(args.ledger)
    warns: list[str] = []

    # 1) 캐시
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle = D.load_cache(data_dir) if args.no_update else D.update_daily(data_dir)
        bundle = D.apply_guards(bundle)
    warns += [w for w in bundle.meta.get("warnings", []) if w not in warns]
    warns += [f"캐시: {m}" for m in _messages(caught) if m not in bundle.meta.get("warnings", [])]
    _log(f"[daily] 캐시 {'로드' if args.no_update else '갱신'} 완료 · SPY 마지막 {bundle.spy_ohlc.index[-1]:%Y-%m-%d} · "
         f"수집 {bundle.meta.get('fetched_at_utc')}")

    # 2) 기준일
    asof, status = determine_asof(bundle.spy_ohlc.index, now_et)
    asof_s = asof.strftime("%Y-%m-%d")
    note = STATUS_NOTE.get(status, status)
    if status == "incomplete":
        note += f" (오늘 {bundle.spy_ohlc.index[-1]:%Y-%m-%d} 봉 제외 → {asof_s})"
    if status not in ("current",):
        warns.append(f"상태 {status}: {note}")
    _log(f"[daily] 지금 {now_et:%Y-%m-%d %H:%M} ET · 기준일 {asof_s} · 상태 {status}")

    # 3) 판정 (두 변형)
    days: dict[str, dict] = {}
    for v in S.VARIANTS:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            days[v] = S.v0_day(bundle, asof, variant=v, basket=args.basket)
        for m0 in _messages(caught):
            m = f"[{v}] {m0}"
            if m not in warns:
                warns.append(m)
        for m in days[v].get("warnings", []):
            mm = f"[{v}] {m}"
            if mm not in warns:
                warns.append(mm)
        d = days[v]
        _log(f"[daily] {v:9s} 월 {d['overall_m']} · 주 {d['overall_w']} · 일 {d['overall_d']} → {d['tone']} · {d['verdict_ko']}"
             f" (F&G {'O' if d['fg_avail'] else 'X'}, 마감봉 {'O' if d['eod_avail'] else 'X'}, 워치 {d['n_watch_avail']})")

    # 4) 장부: completed 가 정식 행, faithful 톤은 추가 열
    run_id = f"daily-{now_utc:%Y%m%dT%H%M%SZ}"
    row = {**days["completed"], "run_id": run_id, "recorded_at_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        appended = L.append_today(row, ledger_path)
    warns += [f"장부: {m}" for m in _messages(caught)]
    if appended:
        _set_extra_columns(ledger_path, asof_s, "completed",
                           {"tone_faithful": days["faithful"]["tone"], "market_status": status})
        _log(f"[daily] 장부 추가 {asof_s} completed={days['completed']['tone']} (faithful={days['faithful']['tone']}) · {run_id}")
    else:
        _log(f"[daily] 장부에 {asof_s} 행이 이미 있음 → 추가하지 않음")
    spy_complete = bundle.spy_ohlc["Close"].astype(float).loc[:asof]      # 미완성 봉은 결과 열에 쓰지 않는다
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        L.backfill(ledger_path, spy_complete)
        ls = L.summary(ledger_path, trading_days=bundle.spy_ohlc.index)
    warns += [f"장부: {m}" for m in _messages(caught)]
    _log(f"[daily] 장부 {ls.get('n')}행 · 결과 확정 20일 {ls.get('n_with_outcome')} / 60일 {ls.get('n_with_outcome_60')}"
         + (f" · 결측 거래일 {ls.get('n_missing_days')}일" if ls.get("n_missing_days") else ""))

    # 5) index.html
    today = {
        "asof": asof_s, "market_status": status, "note": note or None,
        "generated_at": now_et.strftime("%Y-%m-%d %H:%M ET"), "generated_at_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "spy_close": days["completed"]["spy_close"], "vix_close": days["completed"]["vix_close"],
        "spy_last_in_cache": bundle.spy_ohlc.index[-1].strftime("%Y-%m-%d"),
        "run_id": run_id, "ledger_appended": bool(appended), "tone_faithful": days["faithful"]["tone"],
        "warnings": warns,
        "faithful": days["faithful"], "completed": days["completed"],
    }
    out_html = docs_dir / "index.html"
    RPT.render_index(today, ls, out_html)
    _log(f"[daily] {out_html} 저장 · 경고 {len(warns)}건 · {time.perf_counter() - t0:.1f}s")
    for w in warns:
        _log(f"  - {w}")
    return {"asof": asof_s, "status": status, "appended": appended, "days": days, "ledger_summary": ls,
            "warnings": warns, "index_html": out_html}


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser(description="매일 v0 판정 기록 (완성 봉 기준)")
    ap.add_argument("--no-update", action="store_true", help="네트워크 갱신 없이 현재 캐시로 실행")
    ap.add_argument("--now", default=None, help="ET 시각 고정 'YYYY-MM-DD HH:MM' (테스트용)")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--docs-dir", default=str(DOCS_DIR))
    ap.add_argument("--ledger", default=str(L.LEDGER), help=f"장부 CSV 경로 (기본 {L.LEDGER})")
    ap.add_argument("--basket", choices=list(S.BASKETS), default="v0")
    args = ap.parse_args(argv)
    res = run(args)
    write_github_output(res)
    return 0


def write_github_output(res: dict, path: str | None = None) -> bool:
    """GitHub Actions 스텝 출력(asof, market_status, appended). $GITHUB_OUTPUT 이 없으면(로컬) 아무것도 하지 않는다."""
    gh_out = path if path is not None else os.environ.get("GITHUB_OUTPUT")
    if not gh_out:
        return False
    with open(gh_out, "a", encoding="utf-8") as f:
        f.write(f"asof={res['asof']}\nmarket_status={res['status']}\nappended={str(bool(res['appended'])).lower()}\n")
    return True


if __name__ == "__main__":
    sys.exit(main())
