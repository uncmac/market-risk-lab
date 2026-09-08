#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""매일 판정: update_daily → apply_guards → 완성 봉 기준일 결정 → v0_day(faithful·completed) → P2 확률·상태·귀속 → 장부 → docs/index.html

사용:
    python scripts/daily.py                       # 캐시 갱신(네트워크) 후 오늘 판정 기록
    python scripts/daily.py --no-update           # 캐시 갱신 없이(오프라인) 현재 캐시로 판정 (테스트·재생성용)
    python scripts/daily.py --now "2026-09-04 12:00"   # ET 시각을 고정 (테스트: 장중이면 '미완성' 경로)

흐름 (ARCHITECTURE.md 스크립트 절 + ARCHITECTURE_PHASE2.md §13)
  1. bundle = update_daily()(또는 load_cache) → apply_guards (멱등).
  2. asof = SPY 마지막 일자. 지금(ET)이 그 세션 마감(16:05 ET) 전이면 상태 'incomplete' 로 표시하고 **직전 완성 세션**을
     asof 로 쓴다. 오늘 봉이 없으면 주말/휴장/개장 전/데이터 지연으로 구분해 표시하고 마지막 완성 세션을 쓴다.
  3. v0_day 를 두 변형으로 계산. 장부에는 variant='completed' 행이 정식 기록이며, faithful 톤은 같은 행의
     추가 열 `tone_faithful` 에 둔다 (장부 모듈은 추가 열을 보존한다). 같은 날이 이미 있으면 추가하지 않는다.
  3b. **Phase 2 (재적합 금지)**: results/model_p2.json 을 읽기만 한다(spec_sha256 이 현재 코드와 다르면 exit 1 — 코드가 모델보다
     새롭다). build_features(bundle, asof) → input_status → p = **배포 확률**(입력 결측이면 NaN + 사유) → 장부의 마지막 유효 P2 행에서
     상태를 이어 decision.step → 밴드 = max(파라미터 밴드(최근 5회 재적합 + 라이브), 신뢰도 구간 Wilson) → 수준 귀속·일간 귀속(어제·5일)
     → HAR 예측(summary_p2.json.har.live_coef) → 이벤트 → 장부 P2 열 + index.html 의 p2 카드.
     **배포 확률의 정의(2026-09-08 정정)**: summary_p2.json.acceptance 가 배치한 단(deploy_mode="tones" → tone_model 의 확률;
     오늘은 M1). deploy_mode="info_only" 면 배포된 단이 없으므로 생산 모델 M3 의 확률을 '정보 표시(배포 안 함)' 로 기록·표시하고
     톤·비중은 주장하지 않는다. 상태·r·톤·장부 prob_dd5_20 은 모두 이 배포 확률에서 나온다 — 배포되지 않은 단은 사다리
     (p2_p_m1/p2_p_m2/p2_p_m3)에 '정보' 로만 남는다.
     실패(exit 1): model_p2.json 없음 · spec 불일치 · 배치된 단의 라이브 계수 없음/spec 불일치 · 홀드아웃 해제 후 1월 첫 주간 실행
     이후 refit_year ≠ 올해.
     입력 결측은 실패가 아니라 "확률 계산 불가" 경로(상태 유지, days 증가, p2_input_missing 사유).
  4. ledger.backfill — 결과 열은 **완성 세션까지의 SPY 종가**로만 채운다(미완성 봉 사용 금지).
  5. report.render_index → docs/index.html (v0 판정 블록 아래 p2 카드).
  6. GitHub Actions 안이면($GITHUB_OUTPUT) asof / market_status / appended 를 스텝 출력으로 내보낸다 — daily.yml 은
     market_status == 'current' 인 실행만 게이트가 인식하는 제목(`daily: <날짜>`)으로 커밋하고, 그 밖(incomplete/pre_open/
     stale/…)은 `daily(<상태>): <날짜>` 로 커밋해 예약 실행이 그날을 건너뛰지 않게 한다.
종료 코드: 0 성공, 1 실패(예외 — 판정이 없으면 페이지를 만들지 않는다: 조용한 실패 금지).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrl import calendar_us as CAL                                      # noqa: E402
from mrl import data as D                                               # noqa: E402
from mrl import decision as DEC                                         # noqa: E402
from mrl import events as EV                                            # noqa: E402
from mrl import features as F                                           # noqa: E402
from mrl import ledger as L                                             # noqa: E402
from mrl import model as M                                              # noqa: E402
from mrl import report as RPT                                           # noqa: E402
from mrl import signals_v0 as S                                         # noqa: E402
from mrl import vol as V                                                # noqa: E402
from mrl.config import (DATA_DIR, DOCS_DIR, ET, HOLDOUT_UNLOCK_PATH, MODEL_P2_PATH, P2,   # noqa: E402
                        RESULTS_DIR)

MARKET_OPEN = dtime(9, 30)
STATUS_NOTE = {
    "current": "",
    "incomplete": "장 마감 전 실행 — 오늘 봉이 아직 미완성이라 전일(마지막 완성 세션) 기준으로 판정·기록",
    "weekend": "주말 휴장 — 직전 영업일 종가 기준",
    "holiday": "휴장일 — 직전 영업일 종가 기준",
    "pre_open": "개장 전 — 전일 종가 기준",
    "stale": "개장 시간인데 오늘자 봉이 없음 → 데이터 지연으로 간주, 직전 영업일 종가 기준 (조용히 넘기지 않고 경고)",
}
SUMMARY_P2_NAME = "summary_p2.json"
P2_REPRICE_TOL = 0.01          # §16 5: 어제 저장값과 오늘 재계산값의 |Δp| > 0.01 이면 경고
P2_DOD_5D = 5                  # 5일 누적 귀속
P2_RUNGS = ("M1", "M2", "M3")  # 사다리(정보). 배포 단은 acceptance 가 고른다
P2_INFO_RUNG = "M3"            # 배포 단이 없을 때(info_only) 정보로 표시·기록하는 생산 모델
P2_RELIABILITY_KEY = {"M3": "reliability", "M1": "reliability_m1"}   # 단별 신뢰도 표(보정 구간용)


class P2Fatal(RuntimeError):
    """Phase 2 실패 조건(§13): model_p2.json 없음 · spec 불일치 · 재적합 연도 규칙 위반 → exit 1."""


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


def _fill_p2_if_empty(path: Path, asof: str, variant: str, p2_cols: dict) -> bool:
    """이미 있는 (asof, variant) 행의 P2 열이 전부 비어 있으면(Phase 1 시절 행) P2 열만 채운다. 값이 있으면 건드리지 않는다."""
    df = L._read(path)
    mask = (df["asof"] == asof) & (df["variant"] == variant)
    if int(mask.sum()) != 1:
        return False
    i = df.index[mask][0]
    if not (L._is_missing(df.at[i, "prob_dd5_20"]) and L._is_missing(df.at[i, "p2_state"])):
        return False
    base = {k: v for k, v in df.loc[i].to_dict().items() if k not in L.OUTCOME_COLUMNS}
    norm = L._normalize_row({**base, **p2_cols, "asof": asof, "variant": variant,
                             "tone": df.at[i, "tone"], "recorded_at_utc": df.at[i, "recorded_at_utc"]})
    for col in L.P2_COLUMNS + ["prob_dd5_20"]:
        if col in L.P2_STRING_COLUMNS and not pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].astype(object)
        df.at[i, col] = norm[col]
    L._write(df, path)
    return True


# ------------------------------------------------------------------
# Phase 2 — 일간 추론 (재적합 금지, model_p2.json 읽기 전용)
# ------------------------------------------------------------------
def _fnum(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return math.nan
    return f


def _weekly_refit_anchor(sessions, year: int) -> date | None:
    """그 해 1월 첫 거래일 이후의 첫 일요일 — 주간 작업(일요일 13:00 UTC)이 그 해로 재적합할 수 있는 최초 시점.

    달력 첫 일요일을 쓰면(1월 1일이 금·토·일인 해) 첫 거래일보다 앞서서, 주간 작업이 정상 실행됐는데도
    daily 가 죽는다(2027·2028·2033·2034 …). 올해 1월 세션이 아직 인덱스에 없으면 None(가드 미적용).
    sessions 는 SPY 세션 인덱스여야 한다 — bundle.close 는 BTC 주말을 포함한 합집합이라 쓰면 안 된다."""
    idx = pd.DatetimeIndex(sessions)
    jan = idx[(idx.year == year) & (idx.month == 1)]
    if not len(jan):
        return None
    d = jan[0].date()
    return d + timedelta(days=(6 - d.weekday()) % 7)   # 세션은 평일이므로 항상 그 세션 뒤(또는 같은 주)의 일요일


def _model_from_params(rec: dict, m_live: M.LogitModel) -> M.LogitModel:
    """summary_p2.json.params_by_refit 기록 → LogitModel (계수만 있으면 되므로 메타는 기록값·라이브 모델에서).
    m_live 는 같은 단(rung)의 라이브 모델이어야 한다 — 특징 목록·model_id 접두사가 여기서 나온다."""
    rung = (M.rung_of(tuple(rec.get("features") or m_live.features)) or rec.get("rung") or "m3")
    return M.LogitModel(features=tuple(rec.get("features") or m_live.features), coef=dict(rec["coef"]), intercept=float(rec["intercept"]),
                        refit_date=str(rec.get("refit_date")), train_start=str(rec.get("train_start") or m_live.train_start),
                        train_end=str(rec.get("train_end") or ""), n_train=int(rec.get("n_train") or 0), n_pos=int(rec.get("n_pos") or 0),
                        clim=float(rec.get("clim") or m_live.clim), feature_rule=m_live.feature_rule, spec_sha256=m_live.spec_sha256,
                        model_id=str(rec.get("model_id") or f"p2{str(rung).lower()}-{m_live.spec_sha256[:8]}-{rec.get('refit_date')}"))


def _dpp_term(d_pp: dict, feat: str, m_dep: M.LogitModel) -> float:
    """장부의 일간 귀속 한 항(확률 단위). 배포 모델에 있는 항은 그 값, 없는 항은 구조적 0(다른 항이 유효할 때만), 결측이면 NaN.

    M1 을 배포하면 x_har·x_ma 항 자체가 없다 — 0 은 지어낸 값이 아니라 '그 모델에 그 항이 없다' 는 뜻이고,
    덕분에 네 항의 합 == Δ(배포 확률) 이 그대로 유지된다."""
    if feat in m_dep.features:
        return _fnum(d_pp.get(feat))
    vals = [_fnum(d_pp.get(f)) for f in m_dep.features] + [_fnum(d_pp.get("refit"))]
    return 0.0 if (vals and all(math.isfinite(v) for v in vals)) else math.nan


def _model_dict(mm: M.LogitModel, dep: dict) -> dict:
    """카드·정직 스트립용 모델 요약(계수는 그 모델의 것만)."""
    return {"model_id": mm.model_id, "coef": dict(mm.coef), "intercept": mm.intercept, "clim": mm.clim, "refit_date": mm.refit_date,
            "spec_sha256": mm.spec_sha256, "deploy_mode": dep["deploy_mode"], "tone_model": dep["tone_model"],
            "rung": M.rung_of(mm.features), "n_params": mm.n_params(),
            "train_start": mm.train_start, "train_end": mm.train_end, "n_train": mm.n_train, "n_pos": mm.n_pos}


def _rung_of_model_id(model_id) -> str | None:
    """'p2m1-<spec8>-<refit>' → 'M1'. 형식이 다르면 None (지어내지 않는다)."""
    if model_id is None:
        return None
    s = str(model_id)
    head = s.split("-", 1)[0].lower()
    if head.startswith("p2m") and head[3:].isdigit():
        return f"M{head[3:]}"
    return None


def resolve_deployment(acc, m: M.LogitModel, live_models: dict, warns: list[str]) -> dict:
    """배치 판정 → 오늘의 배포 모델. 반환 {deploy_mode, tone_model, prob_rung, model, source, deployed}.

    * 진실의 출처는 summary_p2.json.acceptance (계약: 이 스크립트는 판정을 바꾸지 않고 그대로 옮긴다).
      acceptance 가 없으면 results/model_p2.json 의 deploy_mode/tone_model 로 내려가며 경고한다.
    * deploy_mode == "tones" → prob_rung = tone_model (그 단의 라이브 계수로 확률을 만든다).
      deploy_mode == "info_only" → 배포된 단이 없다 → prob_rung = M3(생산 모델)을 '정보 표시(배포 안 함)' 로 쓴다.
    * 배치된 단의 라이브 계수가 없거나 spec 이 다르면 P2Fatal — 배포되지 않은 모델의 확률을 배포된 것처럼 기록하지 않는다.
    """
    acc = acc if isinstance(acc, dict) else {}
    def _tm(v) -> str:
        return "없음" if v is None or str(v).strip() in ("", "None") else str(v)

    if acc.get("deploy_mode"):
        deploy, tone, source = str(acc.get("deploy_mode")), acc.get("tone_model"), "summary_p2.json:acceptance"
        if (str(m.deploy_mode) != deploy) or ((m.tone_model or None) != (tone or None)):
            warns.append(f"model_p2.json 의 배치(deploy {m.deploy_mode} · tone_model {_tm(m.tone_model)}) 와 "
                         f"summary_p2.json.acceptance(deploy {deploy} · tone_model {_tm(tone)}) 가 다릅니다 → acceptance 를 따릅니다 "
                         "(run_calibration.py 를 다시 실행해 두 파일을 맞추세요)")
    else:
        deploy, tone, source = str(m.deploy_mode), m.tone_model, "results/model_p2.json"
        warns.append(f"summary_p2.json.acceptance 가 없어 배치 정보를 {source} 에서 읽습니다")
    tone = None if (tone is None or str(tone).strip() == "") else str(tone)
    if deploy not in L.P2_DEPLOY_MODES:
        warns.append(f"알 수 없는 deploy_mode {deploy!r} → 배포 없음(info_only)으로 다룹니다")
        deploy = "info_only"
    if deploy == "tones" and not tone:
        warns.append("deploy_mode 는 'tones' 인데 tone_model 이 비어 있습니다 → 배포 없음(info_only)으로 다룹니다")
        deploy = "info_only"
    if deploy != "tones":
        tone = None
    rung = tone if deploy == "tones" else P2_INFO_RUNG
    if rung == P2_INFO_RUNG:
        model = m
    else:
        rec = (live_models or {}).get(rung)
        if not isinstance(rec, dict):
            raise P2Fatal(f"배치된 톤 모델 {rung} 의 라이브 계수를 summary_p2.json.live_models 에서 찾지 못했습니다 — "
                          "run_calibration.py 를 다시 실행하라 (배포되지 않은 M3 로 대신 기록하지 않는다)")
        try:
            model = M.LogitModel.from_dict(rec)
        except (ValueError, KeyError, TypeError) as e:
            raise P2Fatal(f"배치된 톤 모델 {rung} 의 라이브 계수를 읽을 수 없습니다({type(e).__name__}: {e})") from e
        if model.spec_sha256 != m.spec_sha256:
            raise P2Fatal(f"live_models.{rung} 의 spec_sha256 {model.spec_sha256[:12]} ≠ model_p2.json {m.spec_sha256[:12]} — "
                          "run_calibration.py 로 다시 만들라")
        if abs(float(model.clim) - float(m.clim)) > 1e-12:
            warns.append(f"P2: live_models.{rung} 의 clim {model.clim:.6f} ≠ model_p2.json {m.clim:.6f} — 배포 단의 clim 으로 r 을 계산합니다")
    return {"deploy_mode": deploy, "tone_model": tone, "prob_rung": rung, "model": model, "source": source,
            "deployed": deploy == "tones"}


def _load_summary_p2(path: Path, warns: list[str]) -> dict:
    if not path.exists():
        warns.append(f"P2: {path.name} 없음 → 파라미터 밴드·신뢰도 구간·HAR 예측·판정 상자를 채울 수 없음(카드에 '—')")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        warns.append(f"P2: {path.name} 읽기 실패({type(e).__name__}) → 밴드·HAR 생략")
        return {}


def _prev_p2_rows(ledger_path: Path, asof_s: str) -> pd.DataFrame:
    """장부의 completed 변형 P2 행(asof < 오늘, 상태 있음) — asof 오름차순."""
    df = L._read(ledger_path)
    if len(df) == 0:
        return df
    sub = df[(df["variant"] == "completed") & (df["asof"] < asof_s) & df["p2_state"].notna()]
    return sub.sort_values("asof", kind="stable")


def _calib_bin(reliability: list, p: float) -> dict | None:
    if not isinstance(reliability, list) or math.isnan(p):
        return None
    for r in reliability:
        lo, hi = _fnum(r.get("lo")), _fnum(r.get("hi"))
        if math.isnan(lo) or math.isnan(hi):
            continue
        last = r is reliability[-1]
        if lo <= p < hi or (last and p == hi):
            return r
    return None


def _prob_model_id_at(prev_rows: pd.DataFrame, d: pd.Timestamp):
    """그 날 장부 행이 기록한 '확률을 만든 모델' 의 id. 구 행(p2_prob_model_id 없음)은 p2_model_id 로 내려간다."""
    ds = d.strftime("%Y-%m-%d")
    row = prev_rows[prev_rows["asof"] == ds]
    if len(row) == 0:
        return None
    r0 = row.iloc[-1]
    mid = r0.get("p2_prob_model_id") if "p2_prob_model_id" in row.columns else None
    if L._is_missing(mid):
        mid = r0.get("p2_model_id")
    return None if L._is_missing(mid) else str(mid)


def _model_for_date(prev_rows: pd.DataFrame, d: pd.Timestamp, m_live: M.LogitModel, params: list, warns: list[str]) -> M.LogitModel:
    """그 날 장부에 기록된 확률 모델(같은 단의 그때 계수). 없거나 같으면 라이브 모델; 다른 단·모르는 id 면 경고 + 라이브.

    params 는 **배포 단**의 재적합 기록만 담긴 목록이어야 한다(단이 섞이면 day_over_day 가 특징 불일치로 죽는다)."""
    mid = _prob_model_id_at(prev_rows, d)
    if mid is None or mid == m_live.model_id:
        return m_live
    for rec in params:
        if str(rec.get("model_id")) == mid:
            return _model_from_params(rec, m_live)
    rung_prev, rung_now = _rung_of_model_id(mid), _rung_of_model_id(m_live.model_id)
    ds = d.strftime("%Y-%m-%d")
    if rung_prev and rung_now and rung_prev != rung_now:
        warns.append(f"P2: 장부 {ds} 의 확률은 {rung_prev} 단({mid}), 오늘 배포 단은 {rung_now} — 일간 귀속은 오늘 계수로 계산(재적합 항 0), "
                     "두 날의 확률을 한 계열로 읽지 마세요")
    else:
        warns.append(f"P2: 장부의 {ds} 모델 id {mid!r} 를 summary_p2.json.params_by_refit 에서 찾지 못함 → 일간 귀속은 라이브 계수로(재적합 항 0)")
    return m_live


def compute_p2(bundle, asof: pd.Timestamp, ledger_path: Path, results_dir: Path, warns: list[str]) -> dict:
    """§13 daily 확장. 반환 {"ledger": P2 열 dict, "card": p2_card 입력 dict, "log": 한 줄, "ok": bool,
    "deployment": {deploy_mode, tone_model, prob_rung, deployed, source}}.

    확률·상태·귀속·밴드는 모두 **배포 단**(resolve_deployment)의 모델로 계산하고, 배포되지 않은 단은 사다리
    (p2_p_m1/p2_p_m2/p2_p_m3 · 카드 §2)에 정보로만 남긴다."""
    asof_s = asof.strftime("%Y-%m-%d")
    model_path = results_dir / MODEL_P2_PATH.name
    unlock_path = results_dir / HOLDOUT_UNLOCK_PATH.name
    if not model_path.exists():
        raise P2Fatal(f"{model_path} 없음 — scripts/run_calibration.py(주간 작업)가 먼저 만들어야 한다. daily 는 재적합하지 않는다")
    m = M.load_model(model_path)
    sha_now = F.spec_sha256()
    if m.spec_sha256 != sha_now:
        raise P2Fatal(f"model_p2.json 의 spec_sha256 {m.spec_sha256[:12]} ≠ 현재 코드 {sha_now[:12]} — 코드가 모델보다 새롭다: "
                      "run_calibration.py 로 모델을 다시 만들라(daily 는 재적합 금지)")
    refit_year = int(str(m.refit_date)[:4])
    anchor = _weekly_refit_anchor(bundle.spy_ohlc.index, asof.year)
    if unlock_path.exists() and refit_year < asof.year and anchor is not None and asof.date() > anchor:
        raise P2Fatal(f"홀드아웃 해제 후 규칙: {anchor:%Y-%m-%d} 주간 실행 이후인데 model_p2.json 의 refit_year {refit_year} ≠ 올해 {asof.year} — "
                      "주간 작업(run_calibration.py)이 실패했거나 실행되지 않았다")
    sp2 = _load_summary_p2(results_dir / SUMMARY_P2_NAME, warns)
    live_models = sp2.get("live_models") or {}
    p2_warns: list[str] = []
    dep = resolve_deployment(sp2.get("acceptance"), m, live_models, p2_warns)
    m_dep, prob_rung = dep["model"], dep["prob_rung"]
    params = [p for p in (sp2.get("params_by_refit") or []) if p.get("rung") == prob_rung]   # 배포 단의 재적합 기록만
    if not params:
        p2_warns.append(f"summary_p2.json.params_by_refit 에 {prob_rung} 기록이 없음 → 파라미터 밴드는 라이브 계수 하나로만(폭 0)")

    # 특징 (완성 봉 번들, asof 로 자름 — 점 원칙)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        feats = F.build_features(bundle, asof=asof)
    p2_warns += [f"features: {w}" for w in _messages(caught)]
    if feats.index[-1] != asof:
        raise P2Fatal(f"특징 표의 마지막 세션 {feats.index[-1]:%Y-%m-%d} ≠ 기준일 {asof_s}")
    if feats.attrs.get("spec_sha256") != m.spec_sha256:
        raise P2Fatal("build_features 의 spec_sha256 이 모델과 다름")
    x_now = feats.iloc[-1]
    ok, why = F.input_status(x_now)
    p = m_dep.predict_one(x_now) if ok else math.nan            # 배포 확률 — 상태·톤·장부·헤드라인이 모두 이 값에서 나온다
    if ok and not math.isfinite(p):
        ok, why = False, "확률 계산 불가: 모델 예측이 유한하지 않음"
    # 사다리 오늘값(정보) — 배포 단의 열은 p 와 같은 값이 된다
    p_rungs = {}
    for rung in P2_RUNGS:
        if rung == prob_rung:
            p_rungs[rung] = p
            continue
        if rung == P2_INFO_RUNG:                                # 생산 모델(M3) 은 model_p2.json 이 진실
            p_rungs[rung] = m.predict_one(x_now) if ok else math.nan
            continue
        rec = live_models.get(rung)
        if isinstance(rec, dict) and ok:
            try:
                p_rungs[rung] = M.LogitModel.from_dict(rec).predict_one(x_now)
            except (ValueError, KeyError) as e:
                p2_warns.append(f"live_models.{rung} 적재 실패({type(e).__name__}) → 사다리 오늘값 생략")
                p_rungs[rung] = math.nan
        else:
            p_rungs[rung] = math.nan
    if not live_models:
        p2_warns.append("summary_p2.json.live_models 없음 → 사다리 오늘값(M1·M2) 생략")

    # 상태 (장부에서 이어서; 첫날은 normal/0)
    prev_rows = _prev_p2_rows(ledger_path, asof_s)
    if len(prev_rows):
        prev = prev_rows.iloc[-1]
        prev_state, prev_days = str(prev["p2_state"]), int(float(prev["p2_days_in_state"])) if not L._is_missing(prev["p2_days_in_state"]) else 0
        prev_asof = pd.Timestamp(prev["asof"])
    else:
        prev, prev_state, prev_days, prev_asof = None, "normal", 0, None
    clim = float(m_dep.clim)                                    # 배포 단의 기후학 — r 은 배포 확률 ÷ 이 값
    r = (p / clim) if ok else None
    state, days, reason = DEC.step(r, prev_state, prev_days)
    changed = state != prev_state
    thresholds = DEC.next_thresholds(state, days, clim)
    # churn: 장부 P2 행(최근 251) + 오늘
    if len(prev_rows):
        st_hist = prev_rows["p2_state"].astype(str).tolist()[-(DEC.CHURN_WINDOW - 1):]
        seq = st_hist + [state]
        churn = int(sum(1 for a, b in zip(seq[:-1], seq[1:]) if a != b))
        if len(prev_rows) < DEC.CHURN_WINDOW - 1:
            churn_note = f"장부 P2 행 {len(prev_rows)}개(252세션 미만) 기준"
        else:
            churn_note = ""
    else:
        churn, churn_note = 0, "첫 P2 기록"
    churn_alert = churn > DEC.DecisionConfig().churn_alert

    # 밴드: 파라미터 밴드(최근 5회 재적합 + 라이브) vs 신뢰도 구간 Wilson → 넓은 쪽
    param_band, calib_band, lo, hi, src = (math.nan, math.nan), (math.nan, math.nan), math.nan, math.nan, None
    cbin = None
    if ok:
        models = [_model_from_params(rec, m_dep) for rec in sorted(params, key=lambda q: str(q.get("refit_date")))[-P2["param_band_refits"]:]] + [m_dep]
        try:
            param_band = M.parameter_band(models, x_now)
        except ValueError as e:
            p2_warns.append(f"파라미터 밴드 실패: {e}")
        rel_key = P2_RELIABILITY_KEY.get(prob_rung)
        rel = sp2.get(rel_key) if rel_key else None
        if rel is None and sp2:
            p2_warns.append(f"summary_p2.json 에 {prob_rung} 단의 신뢰도 표({rel_key or '없음'})가 없어 보정 구간을 만들 수 없음 "
                            "→ 파라미터 밴드만 사용(다른 단의 표를 대신 쓰지 않는다)")
        cbin = _calib_bin(rel, p)
        if cbin is not None:
            calib_band = (_fnum(cbin.get("wilson_lo")), _fnum(cbin.get("wilson_hi")))
        wp = param_band[1] - param_band[0] if all(math.isfinite(v) for v in param_band) else -1.0
        wc = calib_band[1] - calib_band[0] if all(math.isfinite(v) for v in calib_band) else -1.0
        if wp < 0 and wc < 0:
            p2_warns.append("구간을 만들 수 없음(파라미터 밴드·신뢰도 구간 모두 없음)")
        elif wc > wp:
            lo, hi, src = calib_band[0], calib_band[1], "calib"
        else:
            lo, hi, src = param_band[0], param_band[1], "param"

    # 귀속: 수준 + 일간(어제) + 5일 누적 — 모두 **배포 모델** 기준(합 == Δ 배포 확률).
    # 어제 = 특징 표의 직전 세션(장부에 없어도 계산 가능), 모델 = 그날 장부에 기록된 확률 모델 id
    pos = int(feats.index.get_loc(asof))
    contrib = M.contributions(m_dep, x_now) if ok else None
    contrib_m3 = M.contributions(m, x_now) if (ok and prob_rung != P2_INFO_RUNG) else None    # 정보 표시용(배포 아님)

    def _dod(k: int) -> dict:
        if pos - k < 0:
            return {"d_logit": {}, "d_pp": {}, "d_p": math.nan, "refit": False, "gap_sessions": None}
        x_prev = feats.iloc[pos - k]
        m_prev = _model_for_date(prev_rows, feats.index[pos - k], m_dep, params, p2_warns)
        return M.day_over_day(m_dep, x_now, m_prev, x_prev)

    dod = _dod(1)
    dod_5d = _dod(P2_DOD_5D)
    # §16 5: 어제 저장값 vs 오늘 재계산값 — 같은 모델이 만든 값일 때만 비교한다(단이 바뀐 날을 캐시 수정으로 오인하지 않게)
    if prev is not None and prev_asof in feats.index and not L._is_missing(prev.get("prob_dd5_20")):
        prev_mid = _prob_model_id_at(prev_rows, prev_asof)
        same_rung = prev_mid is None or _rung_of_model_id(prev_mid) in (None, prob_rung)
        if not same_rung:
            p2_warns.append(f"어제({prev_asof:%Y-%m-%d}) 저장 확률은 {_rung_of_model_id(prev_mid)} 단({prev_mid})이고 오늘 배포 단은 "
                            f"{prob_rung} → 재계산 대조(§16 5) 건너뜀. 배포 단이 바뀐 경계입니다")
        else:
            m_prev = _model_for_date(prev_rows, prev_asof, m_dep, params, p2_warns)
            p_prev_now = m_prev.predict_one(feats.loc[prev_asof])
            if math.isfinite(p_prev_now) and abs(p_prev_now - float(prev["prob_dd5_20"])) > P2_REPRICE_TOL:
                p2_warns.append(f"어제({prev_asof:%Y-%m-%d}) 저장 확률 {float(prev['prob_dd5_20']):.4f} vs 오늘 재계산 {p_prev_now:.4f} — "
                                f"|Δp| > {P2_REPRICE_TOL} (캐시 재조정·수정 의심)")

    # HAR 예측 (적합 log-HAR 라이브 계수, 예산 밖)
    har = sp2.get("har") or {}
    har_fc = math.nan
    live_coef = har.get("live_coef") if isinstance(har.get("live_coef"), dict) else None
    if live_coef:
        comp = feats.iloc[[-1]][["rv1", "rv5", "rv22"]]
        try:
            har_fc = float(np.exp(V.har_predict(live_coef, comp).iloc[0]))
        except (ValueError, KeyError) as e:
            p2_warns.append(f"HAR 예측 실패({type(e).__name__}: {e})")
    else:
        p2_warns.append("summary_p2.json.har.live_coef 없음 → HAR 예측 생략")

    # 표시 전용 문맥
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pct_row = F.factor_percentiles(feats).iloc[-1]
    percentiles = {c[4:]: (None if pd.isna(v) else float(v)) for c, v in pct_row.items()}
    context = {}
    cb = bundle.cboe
    if asof in cb.index:
        for c in ("VIX3M", "VIX9D", "VVIX", "SKEW"):
            if c in cb.columns and not pd.isna(cb.at[asof, c]):
                context[c] = float(cb.at[asof, c])
    fg = bundle.fg
    if asof in fg.index and "score" in fg.columns and not pd.isna(fg.at[asof, "score"]):
        context["fg"] = float(fg.at[asof, "score"])
    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        events = EV.upcoming(asof, EV.DEFAULT_HORIZON_SESSIONS)
        horizon = EV.table_horizon(asof)
    p2_warns += [f"events: {w}" for w in _messages(caught2)]
    if horizon.get("warn"):
        p2_warns.append(f"이벤트 표 잔여 {horizon.get('days_left')}일 < {horizon.get('warn_days')}일 — FOMC 표 갱신 필요")

    d_pp = dod.get("d_pp") or {}
    ledger_cols = {
        # prob_dd5_20 = 배포 확률(= p_rungs[prob_rung]). 사다리 세 단은 정보로 나란히 남긴다
        "prob_dd5_20": p, "p2_p_m1": p_rungs.get("M1", math.nan), "p2_p_m2": p_rungs.get("M2", math.nan),
        "p2_p_m3": p_rungs.get("M3", math.nan),
        "p2_p_vix": _fnum(x_now.get("p_vix")), "p2_p_vix_bgk": _fnum(x_now.get("p_vix_bgk")), "p2_clim": clim,
        "p2_lo": lo, "p2_hi": hi, "p2_band_src": src, "p2_x_vix": _fnum(x_now.get("x_vix")), "p2_x_har": _fnum(x_now.get("x_har")),
        "p2_x_ma": _fnum(x_now.get("x_ma")), "p2_har_vol_20": _fnum(x_now.get("har_vol_20")), "p2_har_fc_20": har_fc,
        "p2_r": (r if r is not None else math.nan), "p2_state": state, "p2_days_in_state": days,
        "p2_tone_model": dep["tone_model"], "p2_deploy_mode": dep["deploy_mode"],
        # 일간 귀속(확률 단위; ×100 = pp) — 배포 모델 기준. 합 == Δ(배포 확률) (잔차 0).
        # 배포 모델에 없는 항은 구조적으로 0(예: M1 의 har·ma), 확률을 못 구한 날은 NaN.
        "p2_d_vix": _dpp_term(d_pp, "x_vix", m_dep), "p2_d_har": _dpp_term(d_pp, "x_har", m_dep),
        "p2_d_ma": _dpp_term(d_pp, "x_ma", m_dep), "p2_d_refit": _fnum(d_pp.get("refit")),
        "p2_input_missing": ("" if ok else why),
        "p2_model_id": m.model_id,                  # 생산 모델(model_p2.json = M3) — 재적합 신선도 추적
        "p2_prob_model_id": m_dep.model_id,         # prob_dd5_20 을 실제로 만든 모델
    }
    model_dict = _model_dict(m_dep, dep)            # 카드의 '이 숫자를 만든 모델'
    prod_dict = _model_dict(m, dep) if prob_rung != P2_INFO_RUNG else None
    card = {
        "asof": asof_s, "p": p, "p_m1": p_rungs.get("M1"), "p_m2": p_rungs.get("M2"), "p_m3": p_rungs.get("M3"),
        "p_vix": ledger_cols["p2_p_vix"],
        "p_vix_bgk": ledger_cols["p2_p_vix_bgk"], "p_vix_driftless": _fnum(x_now.get("p_vix_driftless")), "clim": clim,
        "lo": lo, "hi": hi, "band_src": src, "param_band": list(param_band), "calib_band": list(calib_band), "calib_bin": cbin,
        "x_vix": ledger_cols["p2_x_vix"], "x_har": ledger_cols["p2_x_har"], "x_ma": ledger_cols["p2_x_ma"], "vix": _fnum(x_now.get("vix")),
        "har_vol_20": ledger_cols["p2_har_vol_20"], "har_fc_20": har_fc, "rv20_cc": _fnum(x_now.get("rv20_cc")), "ts_diag": _fnum(x_now.get("ts_diag")),
        "percentiles": percentiles, "context": context,
        "r": ledger_cols["p2_r"], "state": state, "days_in_state": days, "deploy_mode": dep["deploy_mode"], "tone_model": dep["tone_model"],
        "prob_rung": prob_rung, "deployed": dep["deployed"], "acceptance_source": dep["source"],
        "thresholds": thresholds, "churn_252": churn, "churn_alert": bool(churn_alert), "reason_ko": reason + (f" ({churn_note})" if churn_note else ""),
        "contributions": contrib, "contributions_m3": contrib_m3, "dod": dod, "dod_5d": dod_5d, "input_missing": (None if ok else why),
        "events": events, "events_horizon": horizon, "model": model_dict, "prod_model": prod_dict, "acceptance": sp2.get("acceptance"),
        "holdout": sp2.get("holdout"), "har_oos_log_mae": har.get("oos_log_mae"), "warnings": p2_warns,
        "changed": bool(changed), "prev_state": prev_state, "prev_asof": (prev_asof.strftime("%Y-%m-%d") if prev_asof is not None else None),
    }
    warns.extend(f"P2: {w}" for w in p2_warns)
    who = f"{prob_rung} {'배포' if dep['deployed'] else '정보 표시(배포 안 함)'}"
    log = (f"[daily] P2 {m_dep.model_id} ({who}) · p={p * 100:.1f}% (사다리 M1 {p_rungs.get('M1', math.nan) * 100:.1f}% · "
           f"M3 {p_rungs.get('M3', math.nan) * 100:.1f}% · VIX {ledger_cols['p2_p_vix'] * 100:.1f}% · clim {clim * 100:.1f}%) · "
           f"r={r:.2f} · {prev_state}→{state} (체류 {days}) · {reason} · 구간 [{lo * 100:.1f}%, {hi * 100:.1f}%] ({src}) · "
           f"어제 대비 {dod.get('d_p', math.nan) * 100:+.1f}pp · HAR {har_fc * 100:.1f}% · deploy {dep['deploy_mode']}"
           if ok else
           f"[daily] P2 {m_dep.model_id} ({who}) · {why} · 상태 유지 {state} (체류 {days}) · deploy {dep['deploy_mode']}")
    return {"ledger": ledger_cols, "card": card, "log": log, "ok": ok, "deployment": {k: v for k, v in dep.items() if k != "model"}}


def run(args) -> dict:
    t0 = time.perf_counter()
    now_et = parse_now(args.now)
    now_utc = now_et.astimezone(timezone.utc)
    data_dir, docs_dir, ledger_path = Path(args.data_dir), Path(args.docs_dir), Path(args.ledger)
    results_dir = Path(args.results_dir)
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

    # 3b) Phase 2 (재적합 금지 — model_p2.json 읽기 전용). 완성 봉 번들을 asof 로 잘라 특징을 만든다.
    bundle_complete = D.Bundle(close=bundle.close.loc[:asof], spy_ohlc=bundle.spy_ohlc.loc[:asof], cboe=bundle.cboe.loc[:asof],
                               fg=bundle.fg.loc[:asof], eod=bundle.eod.loc[:asof], meta=dict(bundle.meta))
    p2 = compute_p2(bundle_complete, asof, ledger_path, results_dir, warns)
    _log(p2["log"])

    # 4) 장부: completed 가 정식 행(P2 열 포함), faithful 톤은 추가 열
    run_id = f"daily-{now_utc:%Y%m%dT%H%M%SZ}"
    row = {**days["completed"], **p2["ledger"], "run_id": run_id, "recorded_at_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        appended = L.append_today(row, ledger_path)
    warns += [f"장부: {m}" for m in _messages(caught)]
    if appended:
        _set_extra_columns(ledger_path, asof_s, "completed",
                           {"tone_faithful": days["faithful"]["tone"], "market_status": status})
        _log(f"[daily] 장부 추가 {asof_s} completed={days['completed']['tone']} (faithful={days['faithful']['tone']}) · "
             f"P2 {p2['ledger']['p2_state']} · {run_id}")
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            filled = _fill_p2_if_empty(ledger_path, asof_s, "completed", p2["ledger"])
        warns += [f"장부: {m}" for m in _messages(caught)]
        _log(f"[daily] 장부에 {asof_s} 행이 이미 있음 → 추가하지 않음" + (" (비어 있던 P2 열만 채움)" if filled else ""))
    spy_complete = bundle.spy_ohlc["Close"].astype(float).loc[:asof]      # 미완성 봉은 결과 열에 쓰지 않는다
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        L.backfill(ledger_path, spy_complete)
        ls = L.summary(ledger_path, trading_days=bundle.spy_ohlc.index, spy_close=spy_complete)
    warns += [f"장부: {m}" for m in _messages(caught)]
    warns += [f"장부 P2: {n}" for n in (ls.get("p2", {}).get("notes") or [])]   # 출처 혼재·홀드아웃 잠금 등은 조용히 넘기지 않는다
    _log(f"[daily] 장부 {ls.get('n')}행 · 결과 확정 20일 {ls.get('n_with_outcome')} / 60일 {ls.get('n_with_outcome_60')}"
         + (f" · 결측 거래일 {ls.get('n_missing_days')}일" if ls.get("n_missing_days") else "")
         + f" · P2 채점 {ls.get('p2', {}).get('n_scored')}행")

    # 5) index.html (v0 판정 블록 아래 p2 카드; live = 장부 요약 p2)
    today = {
        "asof": asof_s, "market_status": status, "note": note or None,
        "generated_at": now_et.strftime("%Y-%m-%d %H:%M ET"), "generated_at_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "spy_close": days["completed"]["spy_close"], "vix_close": days["completed"]["vix_close"],
        "spy_last_in_cache": bundle.spy_ohlc.index[-1].strftime("%Y-%m-%d"),
        "run_id": run_id, "ledger_appended": bool(appended), "tone_faithful": days["faithful"]["tone"],
        "warnings": warns,
        "faithful": days["faithful"], "completed": days["completed"],
        "p2": {**p2["card"], "live": ls.get("p2")},
    }
    out_html = docs_dir / "index.html"
    RPT.render_index(today, ls, out_html)
    _log(f"[daily] {out_html} 저장 · 경고 {len(warns)}건 · {time.perf_counter() - t0:.1f}s")
    for w in warns:
        _log(f"  - {w}")
    return {"asof": asof_s, "status": status, "appended": appended, "days": days, "ledger_summary": ls, "p2": p2,
            "warnings": warns, "index_html": out_html}


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser(description="매일 v0 판정 + Phase 2 확률 기록 (완성 봉 기준, 재적합 금지)")
    ap.add_argument("--no-update", action="store_true", help="네트워크 갱신 없이 현재 캐시로 실행")
    ap.add_argument("--now", default=None, help="ET 시각 고정 'YYYY-MM-DD HH:MM' (테스트용)")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--docs-dir", default=str(DOCS_DIR))
    ap.add_argument("--results-dir", default=str(RESULTS_DIR), help="model_p2.json · summary_p2.json 위치 (읽기 전용)")
    ap.add_argument("--ledger", default=str(L.LEDGER), help=f"장부 CSV 경로 (기본 {L.LEDGER})")
    ap.add_argument("--basket", choices=list(S.BASKETS), default="v0")
    args = ap.parse_args(argv)
    try:
        res = run(args)
    except P2Fatal as e:
        _log(f"::error::[daily] Phase 2 실패 — {e}")
        return 1
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
