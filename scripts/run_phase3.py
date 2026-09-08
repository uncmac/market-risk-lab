#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""주간 Phase 3 산출: 국면 walk-forward → 그림자 등록부·채택 검정 → 비중 백테스트·예산 사다리 → 시나리오 표 →
킬룰 재현 → results/ + docs/{sizing_p3,regime_p3,track_record}.html (ARCHITECTURE_PHASE3.md §11·§12).

사용:
    python scripts/run_phase3.py                          # 하드컷(HOLDOUT_START 직전 세션)까지, 기본 D_max 0.35
    python scripts/run_phase3.py --all-sensitivities      # 민감도 ~15행 전부 표에 (워크플로가 쓰는 형태)
    python scripts/run_phase3.py --end 2006-12-29 --results-dir results/tmp --docs-dir results/tmp/docs
    python scripts/run_phase3.py --holdout-final          # #2b 와 **같은 세션**에서만 (정보 기록; 채택엔 쓰지 않는다)

흐름 (§11)
  load_cache → apply_guards → 완성 봉 자르기 → 하드컷(--end)
  → 읽기: calib_p2_walkforward.csv + summary_p2.json + model_p2.json + backtest_v1.csv
    (spec_sha256·feature_rule 일치 assert, 아니면 exit 1 — 옛 확률 위에 Phase 3 를 세우지 않는다)
  → build_features(P2) → make_targets → regime.observations → hmm_walk_forward(1월 퍼지 재적합, warm start)
  → ensemble.platt_walk_forward(H) → member_probs → combine/disagreement → 결정층 ens3(정보)
  → 사다리 확장(M1→H, M3→H, M3→ens(M3,H), M3→ens(M3,M1), M3→ens3, M3→M4a(#3 닫힘), B1→H)
  → admission_test(H, M1; BLOCKS_24·BLOCKS_18) · 검정력 표 · 등록부 표
  → sizing: ewma_vol → target_vol(--d-max) → run → backtest_table(세 창) → 민감도 → 유지 조건 → 사다리 → 에피소드 손익
  → scenarios: bin_table/state_table/episode_conditionals/coverage
  → track: kill_replay(p2·M1·H) → window_distribution 등 참조 분포 → summary_p3(장부에서 라이브 패널)
  → 결정론 검사(θ·Platt·비중 CSV 해시) → 저장

산출: results/{oos_p3.csv, backtest_p3.csv, summary_p3.json, model_p3.json, hmm_p3.json, track_record_p3.json}
      docs/{sizing_p3.html, regime_p3.html, track_record.html}

원칙
  * **홀드아웃 하드컷**: --holdout-final 없이는 모든 입력 프레임을 HOLDOUT_START 직전 세션에서 자른다(§2).
  * **생산 확률 불변**: Phase 3 는 적합 파라미터를 0개 더한다. 멤버 H 의 Platt 2 + HMM θ 12 는 그림자이며
    `ensemble.assert_shadow_only` 가 매 실행에서 이를 강제한다(§14).
  * **결정론**: 난수 seed 0. registry_sha·sizing_sha·입력 지문이 모두 같은데 θ·Platt·비중 경로가 1e-7·해시 밖이면
    exit 1(파일을 쓰지 않는다). sha 가 다르면 코드 변경 → 새 파일 + run.spec_changed=true + 장부 기재 요구 로그.
    입력 지문이 다르면(캐시 재다운로드) 결정론 검사가 아니므로 이동 폭을 기록·경고하고 새 파일을 쓴다.
  * **조용한 실패 금지**: 계약 함수(mrl.report 의 Phase 3 렌더러)가 없으면 시작 즉시 exit 1 로 무엇이 없는지 말한다.
    표·창을 만들지 못한 경우는 summary_p3.json.warnings 에 사유가 남는다.
종료 코드: 0 성공 · 1 실패(예외·spec 불일치·결정론 검사 실패·계약 함수 없음) · 2 --holdout-final 전제 위반.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrl import calibrate as C                                            # noqa: E402
from mrl import data as D                                                 # noqa: E402
from mrl import decision as DEC                                           # noqa: E402
from mrl import ensemble as EN                                            # noqa: E402
from mrl import evaluate as E                                             # noqa: E402
from mrl import features as F                                             # noqa: E402
from mrl import ledger as L                                               # noqa: E402
from mrl import model as M                                                # noqa: E402
from mrl import regime as RG                                              # noqa: E402
from mrl import report as RPT                                             # noqa: E402
from mrl import scenarios as SC                                           # noqa: E402
from mrl import sizing as SZ                                              # noqa: E402
from mrl import targets as T                                              # noqa: E402
from mrl import track as TR                                               # noqa: E402
from mrl import vol as V                                                  # noqa: E402
from mrl.config import (ALARMS_PATH, BLOCKS_18, BLOCKS_24, DATA_DIR, DOCS_DIR, ENSEMBLE_P3,   # noqa: E402
                        HMM_P3, HMM_P3_PATH, HOLDOUT_START, HOLDOUT_UNLOCK_PATH, KILL_MANUAL_PATH,
                        KILL_P3, KILL_RECORD_PATH, MODEL_P2_PATH, MODEL_P3_PATH, P2, P3, P3_DEPLOY_MODES,
                        P3_DRIFT, RESULTS_DIR, TONE_EXPOSURE, TRACK_P3_PATH)

# ------------------------------------------------------------------
# 계약 (동시에 확장 중인 모듈의 함수 이름 — 없으면 조용히 건너뛰지 않고 시작 즉시 실패한다)
# ------------------------------------------------------------------
REPORT_P3_FUNCS = ("charts_p3", "render_sizing_report", "render_regime_report", "render_track_record")
LEDGER_P3_NAMES = ("P3_COLUMNS",)

CALIB_CSV = "calib_p2_walkforward.csv"
SUMMARY_P2 = "summary_p2.json"
BACKTEST_V1 = "backtest_v1.csv"
SUMMARY_V1 = "summary_v1.json"
SUMMARY_V0 = "summary_v0_completed.json"
BACKTEST_V0 = "backtest_v0_completed.csv"

OOS_CSV_NAME = "oos_p3.csv"
BACKTEST_CSV_NAME = "backtest_p3.csv"
SUMMARY_NAME = "summary_p3.json"
SIZING_HTML = "sizing_p3.html"
REGIME_HTML = "regime_p3.html"
TRACK_HTML = "track_record.html"

OOS_P3_COLUMNS = ("y", "clim", "p_p2", "p_m1", "p_h", "p_hmm_high", "p20", "q20", "x_hmm", "p_ens",
                  "lo", "hi", "state_p2", "state_ens3", "refit_year", "block24", "block18", "theta_id")
BACKTEST_P3_BASE = ("sigma_ewma", "sigma_har_fc", "state", "mult", "w_vol", "w_target", "w_exec", "reason",
                    "ret_rule", "ret_bh", "dd_rule")
# 사다리 확장(§5.4·#3): 관측 기록 위의 채점만 — 채택 근거가 아니다
LADDER_STEPS_P3 = (("M1", "H"), ("M3", "H"), ("M3", "ens_m3h"), ("M3", "ens_m3m1"), ("M3", "ens3"),
                   ("M3", "M4a"), ("B1", "H"))
M4A_FEATURES = ("x_vix", "x_har", "x_ma", "x_hmm")          # 5번째 슬롯 후보(#3) — 채점만, 채택 아님
HEADLINE_SENSITIVITIES = ("S-nofloor", "S-volonly", "S-VT14")     # --all-sensitivities 없이 표에 남기는 행
DISCLOSURE = "#2 사전 관측 + 설계 단계 비중 관측 참조"
TOL_THETA = float(HMM_P3["tol_theta"])                      # 1e-7 (러너·로컬의 BLAS 차이)
TOL_PROB = float(HMM_P3["tol_prob"])
HOLDOUT_SESSION_MINUTES = 180                               # --holdout-final: unlock 파일이 '같은 세션' 인지 보는 창
P3_D3_WINDOW = int(P3_DRIFT["D3_vol_fc"]["window"])         # 60 — 참조 분포도 경보와 같은 창을 쓴다


class P3Fatal(RuntimeError):
    """Phase 3 실패 조건(§11): spec 불일치 · 계약 함수 없음 · 결정론 검사 실패 → exit 1."""


class HoldoutRefused(RuntimeError):
    """--holdout-final 전제 위반 → exit 2 (#2b 와 같은 세션이 아니거나 이미 채점됨)."""


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _log(msg: str) -> None:
    print(msg, flush=True)


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dstr(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _messages(caught) -> list[str]:
    out: list[str] = []
    for w in caught:
        if issubclass(w.category, ResourceWarning):
            continue
        m = str(w.message)
        if m not in out:
            out.append(m)
    return out


def _pct(v, d: int = 1) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    return "—" if not math.isfinite(f) else f"{f * 100:.{d}f}%"


def _num(v, d: int = 1) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    return "—" if not math.isfinite(f) else f"{f:.{d}f}"


def _fnum(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return math.nan
    return f if math.isfinite(f) else math.nan


def missing_contract(module, names, module_label: str) -> list[str]:
    """계약 이름 가운데 아직 없는 것 (동시 작성 중인 모듈을 조용히 건너뛰지 않기 위한 검사)."""
    return [f"{module_label}.{n}" for n in names if not hasattr(module, n)]


def require_contract(*, docs: bool) -> None:
    """§9·§10 계약 이름이 없으면 무엇이 없는지 말하고 exit 1. `--no-docs` 면 리포트 계약은 보지 않는다."""
    missing = missing_contract(L, LEDGER_P3_NAMES, "mrl.ledger")
    if int(getattr(L, "SCHEMA_VERSION", 0)) < 3:
        missing.append(f"mrl.ledger.SCHEMA_VERSION >= 3 (현재 {getattr(L, 'SCHEMA_VERSION', None)})")
    if docs:
        missing += missing_contract(RPT, REPORT_P3_FUNCS, "mrl.report")
    if missing:
        raise P3Fatal(
            "Phase 3 계약 함수가 없습니다: " + ", ".join(missing) + " — ARCHITECTURE_PHASE3.md §10 의 이름 그대로 "
            "mrl/report.py 에 추가돼야 합니다(동시 작업 중이면 그 에이전트가 끝난 뒤 다시 실행하라). "
            "결과 산출물만 필요하면 --no-docs 로 리포트 계약을 건너뛸 수 있습니다(워크플로는 쓰지 않는다)")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _data_sha256(ohlc: pd.DataFrame, vix: pd.Series) -> str:
    """하드컷된 입력(SPY OHLC + 정렬된 VIX)의 지문 — run_calibration._data_sha256 과 같은 정의.
    결정론 검사 실패 시 '자료가 바뀐 것' 과 '코드가 비결정적인 것' 을 구분한다."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(ohlc[list(V.OHLC_COLUMNS)].to_numpy(dtype=float)).tobytes())
    h.update(np.ascontiguousarray(pd.Series(vix).to_numpy(dtype=float)).tobytes())
    h.update(pd.Timestamp(ohlc.index[-1]).strftime("%Y-%m-%d").encode())
    return h.hexdigest()


def _csv_bytes(df: pd.DataFrame, index_label: str = "date") -> bytes:
    """산출 CSV 의 바이트(결정론 해시 비교와 저장이 같은 문자열을 쓰도록 한 곳에서 만든다)."""
    return df.to_csv(index_label=index_label, float_format="%.10g", lineterminator="\n").encode("utf-8")


def _csv_rows(b: bytes) -> tuple[list[str], list[list[str]]]:
    """CSV 바이트를 (헤더, 행들)로 — 값은 문자열 그대로 둔다(재포맷하면 비교가 근사가 된다)."""
    rows = list(csv.reader(io.StringIO(b.decode("utf-8"), newline="")))
    return (rows[0], rows[1:]) if rows else ([], [])


def _csv_shared_columns_equal(old_b: bytes, new_b: bytes) -> dict:
    """저장본과 새 CSV 를 **공통 열**에서 바이트 수준으로 비교한다.

    민감도 열 `w_S-*` 의 집합은 `--all-sensitivities` 라는 **실행 옵션**이 정하므로 코드·자료의
    결정성과 무관하다. 옵션이 다른 두 실행을 전체 해시로 비교하면 결정론 실패로 오판한다(§2 결정론은
    '같은 sha·같은 입력이면 같은 산출' 이지 '같은 옵션이면' 이 아니다). 그래서 공통 열에서 비교하고,
    빠지거나 늘어난 열이 민감도 열이 아니면(계약 열이 바뀐 것) 비교 자체를 실패로 본다."""
    old_h, old_rows = _csv_rows(old_b)
    new_h, new_rows = _csv_rows(new_b)
    only_old = [c for c in old_h if c not in new_h]
    only_new = [c for c in new_h if c not in old_h]
    out = {"shared_equal": False, "columns_same": not (only_old or only_new),
           "only_stored": only_old, "only_new": only_new,
           "option_only": bool((only_old or only_new)
                               and all(c.startswith("w_S-") for c in only_old + only_new)),
           "rows_same": len(old_rows) == len(new_rows)}
    if not out["rows_same"]:
        return out
    shared = [c for c in new_h if c in old_h]
    oi = {c: old_h.index(c) for c in shared}
    ni = {c: new_h.index(c) for c in shared}
    out["shared_equal"] = all(
        all(o[oi[c]] == n[ni[c]] for c in shared) for o, n in zip(old_rows, new_rows))
    return out


def _records(obj):
    return E._jsonable(obj)


def _drop_none(obj):
    """dict 를 그대로 문자열로 찍는 리포트 칸(채택 판정·신선 블록)에서 값이 None 인 키를 뺀다.
    `.get()` 결과는 그대로 None 이므로 뜻은 같고, 페이지에 'None' 이라는 불량 토큰이 새지 않는다."""
    if isinstance(obj, dict):
        return {k: _drop_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_drop_none(v) for v in obj]
    return obj


def _auc(y, s) -> float | None:
    """AUC (라벨이 한 종류뿐이거나 표본이 없으면 None — 0.5 로 채우지 않는다)."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(y) & np.isfinite(s)
    if ok.sum() < 2 or len(np.unique(y[ok])) < 2:
        return None
    from sklearn.metrics import roc_auc_score                       # noqa: WPS433 - 지연 import
    return float(roc_auc_score(y[ok], s[ok]))


# ------------------------------------------------------------------
# 1) 입력 — Phase 2 산출물 계약 (§2 '계약 열')
# ------------------------------------------------------------------
def load_p2(results_dir: Path) -> dict:
    """calib_p2_walkforward.csv + summary_p2.json + model_p2.json + backtest_v1.csv.
    spec_sha256·feature_rule 이 현재 코드와 다르면 P3Fatal — 대체 없음(§16 11)."""
    csv, js, mj, v1p = (results_dir / CALIB_CSV, results_dir / SUMMARY_P2,
                        results_dir / MODEL_P2_PATH.name, results_dir / BACKTEST_V1)
    for p in (csv, js, mj, v1p):
        if not p.exists():
            raise P3Fatal(f"{p} 없음 — scripts/run_calibration.py · scripts/run_backtest_v1.py 를 먼저 실행하라")
    sp2 = json.loads(js.read_text(encoding="utf-8"))
    run = sp2.get("run") or {}
    sha_now, rule_now = F.spec_sha256(), F.FEATURE_RULE
    if run.get("spec_sha256") != sha_now or run.get("feature_rule") != rule_now:
        raise P3Fatal(f"summary_p2.json 의 spec_sha256 {str(run.get('spec_sha256'))[:12]} / feature_rule 이 현재 코드"
                      f"({sha_now[:12]})와 다름 — run_calibration.py 를 다시 실행하라 (옛 확률 위에 Phase 3 를 세우지 않는다)")
    m2 = M.load_model(mj)
    if m2.spec_sha256 != sha_now:
        raise P3Fatal(f"model_p2.json 의 spec_sha256 {m2.spec_sha256[:12]} ≠ 현재 코드 {sha_now[:12]}")
    oos = pd.read_csv(csv, index_col="date", parse_dates=["date"], encoding="utf-8")
    need = {"y", "clim", "p_vix", "p_vix_bgk", "p_m1", "p_m3", "har_fc_20", "refit_year", "block24", "block18"}
    miss = sorted(need - set(oos.columns))
    if miss:
        raise P3Fatal(f"{CALIB_CSV} 에 계약 열이 없음: {miss}")
    if oos.index.has_duplicates or not oos.index.is_monotonic_increasing:
        raise P3Fatal(f"{CALIB_CSV} 인덱스가 중복/비정렬")
    v1 = pd.read_csv(v1p, index_col="date", parse_dates=["date"], encoding="utf-8")
    if "state" not in v1.columns:
        raise P3Fatal(f"{BACKTEST_V1} 에 state 열이 없음 — run_backtest_v1.py 를 다시 실행하라")
    sv1 = {}
    if (results_dir / SUMMARY_V1).exists():
        sv1 = json.loads((results_dir / SUMMARY_V1).read_text(encoding="utf-8"))
    if sv1 and sv1.get("run", {}).get("spec_sha256") not in (None, sha_now):
        raise P3Fatal(f"{SUMMARY_V1} 의 spec_sha256 이 현재 코드와 다름 — run_backtest_v1.py 를 다시 실행하라")
    acc = sp2.get("acceptance") or {}
    deploy_mode = str(acc.get("deploy_mode") or "info_only")
    tone_model = acc.get("tone_model")
    tone_model = None if tone_model in (None, "") else str(tone_model)
    # 배포 확률 = acceptance 가 배치한 단. info_only 면 생산 모델 M3 를 '정보 표시(배포 안 함)' 로 쓴다
    prob_rung = tone_model if (deploy_mode == "tones" and tone_model) else "M3"
    prob_col = C.rung_col(prob_rung)
    if prob_col not in oos.columns:
        raise P3Fatal(f"배치된 단 {prob_rung} 의 확률 열 {prob_col} 이 {CALIB_CSV} 에 없음 (조용한 대체 없음)")
    return {"oos": oos, "summary_p2": sp2, "model_p2": m2, "v1": v1, "summary_v1": sv1,
            "deploy_mode": deploy_mode, "tone_model": tone_model, "prob_rung": prob_rung, "prob_col": prob_col,
            "spec_sha256": sha_now, "feature_rule": rule_now, "first_refit": run.get("first_refit", P2["first_refit"])}


def load_v0(results_dir: Path, warns: list[str]) -> tuple[dict, pd.Series | None]:
    """v0 completed 요약 줄(영구) + 톤 경로(비중 참조 행). window_rule·signals_v0_sha256 이 현재 코드와 다르면 참조 행을 뺀다."""
    for d in (results_dir, RESULTS_DIR):
        js, csv = d / SUMMARY_V0, d / BACKTEST_V0
        if not js.exists():
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = RPT.load_summary_v0(js)
        tone = None
        if csv.exists():
            rep = pd.read_csv(csv, index_col="date", parse_dates=["date"], encoding="utf-8")
            if "tone" in rep.columns:
                tone = rep["tone"].astype(str)
        # 재현 출처 검사(§11): 창 규칙·signals_v0 해시가 현재 코드와 같을 때만 참조 행으로 쓴다
        from mrl import signals_v0 as S0                          # noqa: WPS433 - 지연 import
        head = RPT.v0_headline(s)
        rule_now = getattr(S0, "WINDOW_RULE", None)
        sha_now = hashlib.sha256(Path(S0.__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        rule_old, sha_old = head.get("window_rule"), head.get("signals_v0_sha256")
        if (rule_old is not None and rule_now is not None and rule_old != rule_now) or \
                (sha_old is not None and sha_old != sha_now):
            warns.append("v0 참조 행 생략: summary_v0_completed.json 의 window_rule/signals_v0 해시가 현재 코드와 다름 "
                         "(다른 규칙으로 만든 벤치마크를 비중 표에 섞지 않는다)")
            tone = None
        return s, tone
    warns.append(f"{SUMMARY_V0} 없음 → 비중 표의 v0 참조 행·영구 요약 줄에 숫자가 없다")
    return {}, None


# ------------------------------------------------------------------
# 2) 국면 · 등록부
# ------------------------------------------------------------------
def build_members(spy: pd.Series, y: pd.Series, refits, oos: pd.DataFrame, v1: pd.DataFrame, prob_col: str,
                  prev_thetas, warns: list[str]) -> dict:
    """멤버 H(HMM → Platt) walk-forward + 결합·불일치 + 결정층 ens3(정보). 반환 {oos_p3, thetas, platts, hmm_attrs}."""
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hdf, platts, thetas = EN.member_h_walk_forward(spy, y, refits, prev_thetas=prev_thetas)
    warns += [f"멤버 H: {m}" for m in _messages(caught)]
    warns += [f"멤버 H: {m}" for m in (hdf.attrs.get("warnings") or [])]
    hmm_attrs = dict(hdf.attrs.get("hmm") or {})
    warns += [f"HMM: {m}" for m in (hmm_attrs.get("warnings") or [])]
    seconds = round(time.perf_counter() - t0, 1)

    idx = oos.index
    oo = pd.DataFrame(index=idx)
    oo.index.name = "date"
    oo["y"] = pd.to_numeric(oos["y"], errors="coerce")
    oo["clim"] = pd.to_numeric(oos["clim"], errors="coerce")
    oo["p_p2"] = pd.to_numeric(oos[prob_col], errors="coerce")          # 배포 단(오늘은 info_only → M3 정보)
    oo["p_m1"] = pd.to_numeric(oos["p_m1"], errors="coerce")
    oo["p_h"] = hdf["p_h"].reindex(idx)
    oo["p_hmm_high"] = hdf["p_high"].reindex(idx)
    for c in ("p20", "q20", "x_hmm"):
        oo[c] = hdf[c].reindex(idx)

    statuses = EN.day1_statuses()
    EN.assert_shadow_only(statuses, where="run_phase3")                 # 생산 확률 보호(§14): admitted = ∅
    probs = EN.member_probs(oo)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        oo["p_ens"] = EN.combine(probs, statuses)                       # admitted = ∅ → p2 와 비트 동일
        p2_lo = pd.to_numeric(oos["p2_lo"], errors="coerce") if "p2_lo" in oos.columns else pd.Series(np.nan, index=idx)
        p2_hi = pd.to_numeric(oos["p2_hi"], errors="coerce") if "p2_hi" in oos.columns else pd.Series(np.nan, index=idx)
        dis = EN.disagreement(probs, statuses, p2_lo, p2_hi)
    warns += [f"등록부: {m}" for m in _messages(caught)]
    oo["lo"], oo["hi"] = dis["lo"], dis["hi"]
    oo["state_p2"] = v1["state"].astype(str).reindex(idx)
    ens3 = probs.mean(axis=1, skipna=False)                             # 정보용 3-멤버 평균(생산 아님)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        st3 = DEC.run(ens3, oo["clim"], DEC.DecisionConfig())
    warns += [f"결정층(ens3, 정보): {m}" for m in _messages(caught)]
    oo["state_ens3"] = st3["state"].reindex(idx)
    for c in ("refit_year", "block24", "block18"):
        oo[c] = pd.to_numeric(oos[c], errors="coerce").astype("Int64")
    oo["theta_id"] = hdf["theta_id"].reindex(idx).astype("string")
    oo = oo.reindex(columns=list(OOS_P3_COLUMNS))
    return {"oos_p3": oo, "thetas": thetas, "platts": platts, "hmm_attrs": hmm_attrs, "seconds": seconds,
            "probs": probs, "ens3": ens3, "disagreement": dis, "statuses": statuses,
            "x_hmm_full": hdf["x_hmm"], "p_high_full": hdf["p_high"], "q20_full": hdf["q20"]}


def ladder_extension(oos_p3: pd.DataFrame, oos: pd.DataFrame, feats: pd.DataFrame, y: pd.Series,
                     x_hmm_full: pd.Series, first_refit, end, warns: list[str]) -> dict:
    """§5.4 사다리 확장: M1→H, M3→H, M3→ens(M3,H), M3→ens(M3,M1), M3→ens3, M3→M4a(#3), B1→H.
    M4a = M3 + b4·x_hmm (5 파라미터) 를 같은 일정으로 walk-forward 해 **채점만** 한다(채택 아님)."""
    lad = pd.DataFrame(index=oos_p3.index)
    lad["y"] = oos_p3["y"]
    lad["clim"] = oos_p3["clim"]
    lad["p_vix"] = pd.to_numeric(oos["p_vix"], errors="coerce")
    lad["p_vix_bgk"] = pd.to_numeric(oos["p_vix_bgk"], errors="coerce")
    lad["p_m1"] = oos_p3["p_m1"]
    lad["p_m3"] = pd.to_numeric(oos["p_m3"], errors="coerce")
    lad["p_h"] = oos_p3["p_h"]
    lad["p_ens_m3h"] = (lad["p_m3"] + lad["p_h"]) / 2.0
    lad["p_ens_m3m1"] = (lad["p_m3"] + lad["p_m1"]) / 2.0
    lad["p_ens3"] = (lad["p_m3"] + lad["p_m1"] + lad["p_h"]) / 3.0

    m4a = {"fitted": False, "note": "x_hmm 을 붙인 5번째 슬롯 후보 — 채점만(§5.4 · 장부 #3)"}
    feats4 = feats.copy()
    feats4["x_hmm"] = pd.Series(x_hmm_full).reindex(feats.index)      # 전 이력(학습 구간 포함) — OOS 창만 넣으면 학습 행이 없다
    if feats4["x_hmm"].notna().sum() < 500:
        warns.append("사다리: x_hmm 유효 행이 부족해 M4a 단을 건너뜀")
    else:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                oos4, params4 = C.walk_forward(feats4, y, ladder={"M4a": M4A_FEATURES},
                                               first_refit=first_refit, end=end)
            warns += [f"M4a: {m}" for m in _messages(caught)]
            lad["p_m4a"] = oos4["p_m4a"].reindex(lad.index)
            b4 = [_fnum((p.get("coef") or {}).get("x_hmm")) for p in params4]
            b4 = [v for v in b4 if math.isfinite(v)]
            m4a = {"fitted": True, "features": list(M4A_FEATURES), "n_params": 5, "n_refits": len(params4),
                   "b4_min": (min(b4) if b4 else None), "b4_max": (max(b4) if b4 else None),
                   "note": "5번째 슬롯 후보(#3): 채점만 — 생산 확률에 넣지 않는다"}
        except Exception as e:                                    # noqa: BLE001 - 조용한 실패 금지
            warns.append(f"사다리: M4a walk-forward 실패({type(e).__name__}: {e}) → 그 단만 건너뜀")

    steps = tuple(s for s in LADDER_STEPS_P3 if C.rung_col(s[1]) in lad.columns)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tbl24 = C.ladder_table(lad, [(a, b) for a, b in BLOCKS_24], steps=steps)
        tbl18 = C.ladder_table(lad, [(a, b) for a, b in BLOCKS_18], steps=steps)
    warns += [f"사다리: {m}" for m in _messages(caught)]
    return {"steps": [f"{a}->{b}" for a, b in steps], "blocks24": tbl24, "blocks18": tbl18, "m4a": m4a,
            "frame": lad}


def admissions(oos_p3: pd.DataFrame, statuses: dict, warns: list[str]) -> dict:
    """채택 검정(§5.4) — 관측 기록(BLOCKS_24·BLOCKS_18)은 **기각만** 할 수 있다. 채택은 신선 블록에서만."""
    out: dict = {"note": "관측 기록(2003~2024-08)의 채택 검정은 기각 전용 — ADMIT 는 라이브 신선 블록 ≥ 11개에서만(§5.4)",
                 "power_table": _records(EN.admission_power_table()),
                 "fresh_blocks_min": int(ENSEMBLE_P3["fresh_blocks_min"]), "admit_frac": float(ENSEMBLE_P3["admit_frac"]),
                 "order": list(ENSEMBLE_P3["member_order"])}
    for label, blocks in (("24", BLOCKS_24), ("18", BLOCKS_18)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            seq = EN.admission_sequence(oos_p3, [(a, b) for a, b in blocks], statuses=statuses)
        warns += [f"채택 검정({label}): {m}" for m in _messages(caught)]
        out[f"blocks{label}"] = {"order": list(seq["order"]), "admitted": list(seq["admitted"]),
                                 "skipped": seq["skipped"],
                                 "results": {k: _drop_none(_records(v)) for k, v in seq["results"].items()}}
    admitted24 = set(out["blocks24"]["admitted"]) - {"p2"}
    if admitted24:
        warns.append(f"채택 검정이 관측 기록에서 ADMIT 를 냈습니다({sorted(admitted24)}) — §5.4 는 관측 기록으로 "
                     "채택할 수 없다고 사전 등록했다. 상태는 바꾸지 않고 기록만 한다(장부 항목 필요)")
    return out


def member_health(oos_p3: pd.DataFrame, theta_tbl: pd.DataFrame, warns: list[str]) -> dict:
    """§5.4 멤버 탈락 검사: H 의 풀링 BSS_clim ≤ 0 또는 AUC < 0.60, guard 3회 이상 → candidate_rejected."""
    ok = oos_p3["y"].notna() & oos_p3["p_h"].notna() & oos_p3["clim"].notna()
    n = int(ok.sum())
    res = {"n": n, "brier": None, "bss_clim": None, "auc": None, "guard_trips": 0, "status": "shadow", "reasons": []}
    if n:
        yv = oos_p3.loc[ok, "y"].to_numpy(dtype=float)
        ph = oos_p3.loc[ok, "p_h"].to_numpy(dtype=float)
        cl = oos_p3.loc[ok, "clim"].to_numpy(dtype=float)
        bs, ref = float(np.mean((ph - yv) ** 2)), float(np.mean((cl - yv) ** 2))
        res["brier"] = bs
        res["bss_clim"] = float(1.0 - bs / ref) if ref > 0 else None
        res["auc"] = _auc(yv, ph)
    if theta_tbl is not None and len(theta_tbl) and "guard" in theta_tbl.columns:
        res["guard_trips"] = int((theta_tbl["guard"].astype(str) != "ok").sum())
    if res["bss_clim"] is not None and res["bss_clim"] <= 0:
        res["reasons"].append(f"풀링 BSS_clim {res['bss_clim']:+.4f} ≤ 0")
    if res["auc"] is not None and math.isfinite(res["auc"]) and res["auc"] < 0.60:
        res["reasons"].append(f"AUC {res['auc']:.3f} < 0.60")
    if res["guard_trips"] >= 3:
        res["reasons"].append(f"guard {res['guard_trips']}회 트립")
    if res["reasons"]:
        res["status"] = "candidate_rejected"
        warns.append("멤버 H 탈락(§5.4): " + " · ".join(res["reasons"]) + " → candidate_rejected (게이지 숨김)")
    return res


def era_auc_p3(oos_p3: pd.DataFrame, oos: pd.DataFrame, feats: pd.DataFrame, eras=P2["eras"]) -> list[dict]:
    """시대별 AUC 표 하나(§10 regime ⑥ 'P_high vs p_vix vs M3'). Phase 2 의 특징·단 AUC(calibrate.era_auc)에
    HMM 열(auc_p_hmm · auc_p_h · auc_p_p2)을 같은 행으로 붙인다. AUC 가 불가능한 시대는 None — 0.5 로 채우지 않는다."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                              # 라벨이 한 종류뿐인 시대의 경고는 값(None)으로 대신한다
        base = C.era_auc(feats, oos)
    by_era = {str(r.get("era")): dict(r) for r in _records(base)}
    rows = []
    idx = oos_p3.index
    for a, b in eras:
        key = f"{_dstr(a)}..{_dstr(b)}"
        m = (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
        sub = oos_p3[m]
        y = sub["y"].to_numpy(dtype=float)
        rec = dict(by_era.get(key) or {"era": key, "start": _dstr(a), "end": _dstr(b)})
        rec["n"] = int(np.isfinite(y).sum())
        rec["n_pos"] = int(np.nansum(y)) if np.isfinite(y).any() else 0
        rec["auc_p_hmm"] = _auc(y, sub["p_hmm_high"].to_numpy(dtype=float))    # 필터 P_high (그림자)
        rec["auc_p_h"] = _auc(y, sub["p_h"].to_numpy(dtype=float))             # 멤버 H (Platt)
        rec["auc_p_p2"] = _auc(y, sub["p_p2"].to_numpy(dtype=float))           # 배포 단
        rows.append(rec)
    return rows


# ------------------------------------------------------------------
# 3) 비중 (§6)
# ------------------------------------------------------------------
def build_sizing(spy: pd.Series, oos: pd.DataFrame, v1: pd.DataFrame, d_max: float, end, v0_tone,
                 all_sens: bool, warns: list[str]) -> dict:
    """§6 비중: σ̂ → σ_T → 채택 규칙 경로 → 세 창 백테스트 표 → 민감도 → 유지 조건 → 예산 사다리 → 에피소드 손익."""
    try:
        sigma_target = SZ.target_vol(d_max)
    except ValueError as e:                                        # 사다리 아래의 예산선은 제공하지 않는다(§6.1.2)
        raise P3Fatal(f"--d-max {d_max} 를 쓸 수 없다: {e}") from e
    sigma = SZ.ewma_vol(spy)
    states = v1["state"].astype(str)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        path = SZ.run(sigma.reindex(states.index), states, sigma_target, deploy=True)
        sens = SZ.sensitivities(oos, states, spy, sigma_target=sigma_target)
    warns += [f"비중: {m}" for m in _messages(caught)]

    # 1993~ 변동성 단독(결정층 없음) — 같은 실행 규칙, 상태 배수만 없다
    vo_idx = spy.loc[pd.Timestamp(P3["vol_only_start"]):].index
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        w_vol_full = SZ.exposure_target(sigma.reindex(vo_idx), 1.0, sigma_target)["w_vol"]
        vol_only_full = SZ.execute(w_vol_full, pd.Series(1.0, index=vo_idx))["w_exec"]
    warns += [f"비중(1993~ 변동성 단독): {m}" for m in _messages(caught)]

    keep = dict(sens) if all_sens else {k: v for k, v in sens.items() if k in HEADLINE_SENSITIVITIES}
    paths = {SZ.ROW_ADOPTED: path["w_exec"], **keep}
    refs_eval = {SZ.ROW_P2_DECISION: SZ.state_multiplier(states)}
    refs_v0 = dict(refs_eval)
    if v0_tone is not None:
        refs_v0["v0_completed"] = v0_tone.map(TONE_EXPOSURE).astype(float).dropna()
    cost = {n: (10.0 if n == "S-10bp" else float(P3["cost_bps"]))
            for n in set(paths) | set(refs_v0) | {SZ.ROW_VOL_ONLY}}

    windows: dict[str, tuple] = {SZ.WINDOW_EVAL: (P3["eval_start"], end)}
    tables = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tables.append(SZ.backtest_table(paths, spy, windows, cost_bps=cost, references=refs_eval,
                                        sigma_target=sigma_target))
        if (spy.index >= pd.Timestamp(P3["v0_start"])).sum() >= 60:
            tables.append(SZ.backtest_table(paths, spy, {SZ.WINDOW_V0: (P3["v0_start"], end)}, cost_bps=cost,
                                            references=refs_v0, sigma_target=sigma_target))
        else:
            warns.append(f"비중 표: {SZ.WINDOW_V0} 창의 거래일이 부족해 생략(짧은 --end)")
        tables.append(SZ.backtest_table({SZ.ROW_VOL_ONLY: vol_only_full}, spy,
                                        {SZ.WINDOW_VOL_ONLY: (P3["vol_only_start"], end)},
                                        cost_bps=float(P3["cost_bps"]), sigma_target=sigma_target))
    warns += [f"비중 표: {m}" for m in _messages(caught)]
    table = pd.concat(tables, ignore_index=True)
    table.attrs = dict(tables[0].attrs)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ret = SZ.retention(table, sigma_target)
    warns += [f"유지 조건: {m}" for m in _messages(caught)]

    # 민감도 표(§10 사이징 페이지 ⑥) — **모든** 변형을 채점 창에서 같은 규약으로 점수 낸다.
    # ④ 의 표는 --all-sensitivities 유무에 따라 일부만 싣지만 ⑥ 은 '규칙의 가족' 전체를 보여야 한다
    # (§16 6). 채택 규칙 행을 같이 넣어 읽는 사람이 기준선 없이 비교하지 않게 한다.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sens_cost = {n: (10.0 if n == "S-10bp" else float(P3["cost_bps"])) for n in {SZ.ROW_ADOPTED, *sens}}
        sens_table = SZ.backtest_table({SZ.ROW_ADOPTED: path["w_exec"], **sens}, spy, windows,
                                       cost_bps=sens_cost, sigma_target=sigma_target)
    warns += [f"민감도 표: {m}" for m in _messages(caught)]

    # 예산 사다리: 격자의 모든 σ_T 를 같은 규칙으로 다시 돌려 채운다(근사와 공식을 섞지 않는다)
    ladder_results: dict[float, dict] = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for g in P3["vol_grid"]:
            g = float(g)
            w_dec = SZ.run(sigma.reindex(states.index), states, g, deploy=True)["w_exec"]
            w_vo = SZ.execute(SZ.exposure_target(sigma.reindex(vo_idx), 1.0, g)["w_vol"],
                              pd.Series(1.0, index=vo_idx))["w_exec"]
            a_dec = SZ.allocation_from_weights(w_dec, spy.loc[w_dec.index[0]:], cost_bps=float(P3["cost_bps"]))
            a_vo = SZ.allocation_from_weights(w_vo, spy.loc[w_vo.index[0]:], cost_bps=float(P3["cost_bps"]))
            rel = SZ.calendar_year_relative(a_dec["series"]["daily_ret"], a_dec["series"]["daily_ret_bh"])
            ladder_results[g] = {
                "maxdd_vol_only": a_vo["max_dd"],
                "maxdd_vol_only_date": None if pd.isna(a_vo["max_dd_date"]) else _dstr(a_vo["max_dd_date"]),
                "maxdd_decision": a_dec["max_dd"],
                "maxdd_decision_date": None if pd.isna(a_dec["max_dd_date"]) else _dstr(a_dec["max_dd_date"]),
                "worst_month": a_dec["worst_month"], "worst_month_label": a_dec["worst_month_label"],
                "cagr": a_dec["cagr"], "avg_exposure": a_dec["avg_exposure"],
                "switches_per_year": a_dec["switches_per_year"], "d_cagr_vs_bh": a_dec["excess_cagr"],
                "share_behind_years": rel.get("share_behind"),
            }
        ladder = SZ.budget_ladder(results=ladder_results)
    warns += [f"예산 사다리: {m}" for m in _messages(caught)]

    # 채택 규칙 경로의 일간 수익·낙폭 (backtest_p3.csv · 상대성과 · 참조 분포)
    alloc = SZ.allocation_from_weights(path["w_exec"], spy.loc[path.index[0]:], cost_bps=float(P3["cost_bps"]))
    ret_rule = alloc["series"]["daily_ret"]
    ret_bh = alloc["series"]["daily_ret_bh"]
    equity = alloc["series"]["equity"]
    dd_rule = equity / equity.cummax() - 1.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ep5 = T.episodes(spy, 0.05, split=False)
        pnl = SZ.episode_pnl(path["w_exec"], spy, ep5)
        relative = {str(L): SZ.rolling_relative(ret_rule, ret_bh, L) for L in P3["reference_windows"] if L <= len(ret_rule)}
        cal_rel = SZ.calendar_year_relative(ret_rule, ret_bh)
    warns += [f"에피소드 손익: {m}" for m in _messages(caught)]
    return {"sigma": sigma, "sigma_target": sigma_target, "d_max": float(d_max), "path": path,
            "sensitivities": sens, "sensitivities_used": sorted(keep), "vol_only_full": vol_only_full,
            "sensitivity_table": sens_table,
            "table": table, "retention": ret, "ladder": ladder, "episode_pnl": pnl, "alloc": alloc,
            "ret_rule": ret_rule, "ret_bh": ret_bh, "dd_rule": dd_rule, "relative": relative,
            "calendar_relative": cal_rel, "states": states}


def backtest_p3_frame(sz: dict, oos: pd.DataFrame, all_sens: bool) -> pd.DataFrame:
    """results/backtest_p3.csv (§11): 채택 규칙 경로 + 수익·낙폭 + 민감도 비중 열 w_S_*."""
    path = sz["path"]
    out = pd.DataFrame(index=path.index)
    out.index.name = "date"
    out["sigma_ewma"] = path["sigma"]
    out["sigma_har_fc"] = pd.to_numeric(oos["har_fc_20"], errors="coerce").reindex(path.index)
    out["state"] = sz["states"].reindex(path.index).astype(str)
    for c in ("mult", "w_vol", "w_target", "w_exec", "reason"):
        out[c] = path[c]
    out["ret_rule"] = sz["ret_rule"].reindex(path.index)
    out["ret_bh"] = sz["ret_bh"].reindex(path.index)
    out["dd_rule"] = sz["dd_rule"].reindex(path.index)
    names = sorted(sz["sensitivities"]) if all_sens else sorted(sz["sensitivities_used"])
    for name in names:
        out[f"w_{name}"] = sz["sensitivities"][name].reindex(path.index)
    return out.reindex(columns=list(BACKTEST_P3_BASE) + [c for c in out.columns if c.startswith("w_S-")])


# ------------------------------------------------------------------
# 4) 시나리오 (§7)
# ------------------------------------------------------------------
def build_scenarios(spy: pd.Series, oos_p3: pd.DataFrame, feats: pd.DataFrame, oos: pd.DataFrame, tgts: pd.DataFrame,
                    end, warns: list[str]) -> dict:
    """§7 시나리오 표 — 세기만 한다. 모든 행에 n·n_eff·구간; n_eff < 20 은 풀링(코드가 강제)."""
    idx = oos_p3.index
    fwd_ret = tgts["fwd_ret_20"].reindex(idx)
    fwd_dd = tgts["fwd_maxdd_20"].reindex(idx)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ep10 = SC.episode_starts(spy, 0.10, end=end)
        bins = SC.bin_table(oos_p3["p_p2"], oos_p3["y"], fwd_ret, fwd_dd, episodes10=ep10)
        states = SC.state_table(oos_p3["state_p2"], oos_p3["y"], fwd_ret, fwd_dd)
        episodes = SC.episode_conditionals(spy, end=end, states=oos_p3["state_p2"])
        cov = SC.coverage(fwd_ret, feats["vix"].reindex(idx), pd.to_numeric(oos["har_fc_20"], errors="coerce"))
        dd_now = SC.current_drawdown(spy, end)
    warns += [f"시나리오: {m}" for m in _messages(caught)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SC.assert_displayable(bins, name="확률 구간 표")
    warns += [f"시나리오(표시 검사): {m}" for m in _messages(caught)]
    spot, vix_now = float(spy.iloc[-1]), _fnum(feats["vix"].reindex(idx).iloc[-1])
    har_now = _fnum(pd.to_numeric(oos["har_fc_20"], errors="coerce").iloc[-1])
    ranges = {"vix": (SC.implied_range(spot, vix_now) if math.isfinite(vix_now) else None),
              "har": (SC.har_range(spot, har_now) if math.isfinite(har_now) else None)}
    return {"bins": bins, "states": states, "episodes": episodes, "coverage": cov, "drawdown": dd_now,
            "ranges": ranges, "spot": spot, "vix": vix_now, "har_fc_20": har_now,
            "spread_sentence": SC.SPREAD_SENTENCE}


# ------------------------------------------------------------------
# 5) 트랙레코드 참조 분포 · 킬룰 재현 (§8)
# ------------------------------------------------------------------
def build_reference(oos_p3: pd.DataFrame, sz: dict, scen: dict, feats: pd.DataFrame, oos: pd.DataFrame,
                    spy: pd.Series, ledger_df, warns: list[str]) -> dict:
    """`summary_p3.json.reference` — 라이브 숫자를 옆에 세울 백테스트 분포(§8.2·§8.5). 임계는 config 에서만 온다."""
    ref: dict = {}
    ref.update(TR.window_distribution(oos_p3, "p_p2"))                     # rolling_bss · quantiles · step
    p = oos_p3["p_p2"].dropna()
    ref["p_level"] = {"mean": _fnum(p.mean()), "share_gt_030": _fnum((p > 0.30).mean()), "n": int(len(p))}
    st = oos_p3["state_p2"].astype("string").dropna()
    ref["occupancy"] = {s: _fnum((st == s).mean()) for s in ("normal", "caution", "reduce")}
    table = sz["table"]

    def _row(name, win):
        r = table[(table["row"] == name) & (table["window"] == win)]
        return None if not len(r) else r.iloc[0]
    dec, adopted = _row(SZ.ROW_P2_DECISION, SZ.WINDOW_EVAL), _row(SZ.ROW_ADOPTED, SZ.WINDOW_EVAL)
    ref["switching"] = {"per_year": (_fnum(dec["switches_per_year"]) if dec is not None else None)}
    ref["churn"] = {"per_year": (_fnum(adopted["switches_per_year"]) if adopted is not None else None)}
    w = sz["path"]["w_exec"].dropna()
    ref["exposure"] = {"avg_w": _fnum(w.mean()),
                       "share_floor": _fnum((w <= float(P3["w_min"]) + 1e-9).mean()),
                       "share_full": _fnum((w >= float(P3["w_max"]) - 1e-9).mean())}
    ref["vol"] = _vol_reference(sz, spy, oos, warns)
    ref["coverage"] = {"vix80": _fnum((scen["coverage"].get("vix") or {}).get("hit_80")),
                       "har80": _fnum((scen["coverage"].get("har") or {}).get("hit_80"))}
    rel: dict = {}
    for k, v in (sz["relative"] or {}).items():                            # pp 단위(§8.5 패널 (viii))
        rel[k] = [_fnum(100.0 * _fnum(v.get("p10"))), _fnum(100.0 * _fnum(v.get("p50"))),
                  _fnum(100.0 * _fnum(v.get("p90")))]
    dd = sz["dd_rule"]
    rel["maxdd_252_median"] = _fnum(dd.rolling(252).min().median()) if len(dd) >= 252 else None
    rel["maxdd_252_worst"] = _fnum(dd.min()) if len(dd) else None
    ref["relative"] = rel
    fr: dict = {}
    for name in ("x_vix", "x_har", "x_ma"):
        s = pd.to_numeric(feats[name], errors="coerce").dropna() if name in feats.columns else pd.Series(dtype=float)
        if len(s):
            fr[name] = [float(s.min()), float(s.max())]
    xh = oos_p3["x_hmm"].dropna()
    if len(xh):
        fr["x_hmm"] = [float(xh.min()), float(xh.max())]
    ref["feature_range"] = fr
    ref["fresh_blocks"] = _fresh_block_progress(ledger_df, warns)
    return ref


def _realized_rv20(spy: pd.Series) -> pd.Series:
    """앞으로 20세션의 실현 변동성(연율) — 장부 backfill 의 rv20_realized 와 같은 정의(종가 로그수익 20일 std·√252)."""
    h = int(P2["h"])
    r = np.log(spy.astype(float) / spy.astype(float).shift(1))
    # r.shift(-h) 의 위치 t 는 r[t+h]; rolling(h) 는 t-h+1..t → r[t+1..t+h] (ddof=1, ledger._backfill_p3 과 같은 정의)
    return r.shift(-h).rolling(h).std() * math.sqrt(252.0)


def _vol_reference(sz: dict, spy: pd.Series, oos: pd.DataFrame, warns: list[str]) -> dict:
    """HAR log-MAE 중앙값(60세션 창)과 ln(실현 RV20/σ̂_EWMA) 편향 범위 — D3 의 참조 분포."""
    out = {"har_log_mae_p50": None, "ewma_bias_range": None, "window": int(P3_D3_WINDOW)}
    try:
        idx = sz["path"].index
        realized = _realized_rv20(spy).reindex(idx)
        har = pd.to_numeric(oos["har_fc_20"], errors="coerce").reindex(idx)
        ew = pd.to_numeric(sz["path"]["sigma"], errors="coerce")
        m = realized.notna() & har.notna() & (realized > 0) & (har > 0)
        if int(m.sum()) >= P3_D3_WINDOW:
            mae = (np.log(har[m]) - np.log(realized[m])).abs().rolling(P3_D3_WINDOW).mean().dropna()
            out["har_log_mae_p50"] = _fnum(mae.median())
        else:
            warns.append(f"참조 분포(변동성): HAR·실현 RV20 유효 행 {int(m.sum())} < {P3_D3_WINDOW} → log-MAE 미산출")
        m2 = realized.notna() & ew.notna() & (realized > 0) & (ew > 0)
        if int(m2.sum()) >= P3_D3_WINDOW:
            bias = np.log(realized[m2] / ew[m2]).rolling(P3_D3_WINDOW).mean().dropna()
            out["ewma_bias_range"] = [_fnum(bias.quantile(0.05)), _fnum(bias.quantile(0.95))]
        else:
            warns.append(f"참조 분포(변동성): EWMA·실현 RV20 유효 행 {int(m2.sum())} < {P3_D3_WINDOW} → 편향 범위 미산출")
    except Exception as e:                                        # noqa: BLE001 - 참조 분포는 표시용이지만 조용히 넘기지 않는다
        warns.append(f"참조 분포(변동성): 계산 실패({type(e).__name__}: {e})")
    return out


def _fresh_block_progress(ledger_df, warns: list[str]) -> dict | None:
    """신선 블록 진행 'k/11' (라이브 시작부터). 라이브 장부가 없으면 None — 없는 진행을 지어내지 않는다."""
    if ledger_df is None or not len(ledger_df):
        return None
    try:
        ls = TR.live_start(ledger_df)
    except Exception as e:                                        # noqa: BLE001
        warns.append(f"신선 블록: live_start 계산 실패({type(e).__name__}: {e})")
        return None
    if ls is None:
        return {"complete": 0, "need": int(ENSEMBLE_P3["fresh_blocks_min"]),
                "note": "Phase 2 톤 가동 전 — 신선 블록은 라이브 시작부터 센다(§5.4)"}
    try:
        idx = pd.DatetimeIndex(pd.to_datetime(ledger_df["asof"], errors="coerce").dropna().unique()).sort_values()
        tbl = EN.fresh_block_table(idx, ls)
        return _drop_none({"complete": int(tbl["complete"].sum()), "need": int(ENSEMBLE_P3["fresh_blocks_min"]),
                           "live_start": _dstr(ls), "table": _records(tbl)})
    except ValueError as e:
        warns.append(f"신선 블록: {e}")
        return None


def build_ablations(oos_p3: pd.DataFrame, q20_full: pd.Series, y: pd.Series, refits, warns: list[str]) -> list[dict]:
    """§10 regime ⑦ 소거 — **선택에 쓰지 않는다**. 이 실행이 실제로 채점하는 것은 q20 Platt 하나이고,
    나머지(A 관측형·B 동결형·대각 공분산)는 설계 단계 관측(§4.3)이라고 그대로 밝힌다(근사와 공식을 섞지 않는다)."""
    rows: list[dict] = []
    yv = oos_p3["y"].to_numpy(dtype=float)
    base = {"name": "H (채택: Platt on logit P_high)", "computed": True,
            "auc": _auc(yv, oos_p3["p_h"].to_numpy(dtype=float)),
            "note": "이 실행의 공식 수치"}
    rows.append(base)
    try:                                                            # q20 Platt: 같은 일정·같은 Platt, 특징만 q20
        q = pd.Series(q20_full).astype(float).clip(float(HMM_P3["clip"]), 1.0 - float(HMM_P3["clip"]))
        x_q = pd.Series(RG.logit(q.to_numpy(dtype=float)), index=q.index)      # 전 이력(학습 구간 포함)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df_q, _models = EN.platt_walk_forward(x_q, y, refits)
        warns += [f"소거(q20 Platt): {m}" for m in _messages(caught)]
        p_q = df_q["p_h"].reindex(oos_p3.index).to_numpy(dtype=float)
        ok = np.isfinite(p_q) & np.isfinite(yv)
        cl = oos_p3["clim"].to_numpy(dtype=float)
        bs = float(np.mean((p_q[ok] - yv[ok]) ** 2)) if ok.any() else float("nan")
        ref = float(np.mean((cl[ok] - yv[ok]) ** 2)) if ok.any() else float("nan")
        rows.append({"name": "q20 Platt (P_high 대신 20세션 안 고변동 확률)", "computed": True,
                     "auc": _auc(yv, p_q), "brier": bs,
                     "bss_clim": (float(1.0 - bs / ref) if ref and math.isfinite(ref) and ref > 0 else None),
                     "note": "특징 하나만 바꾼 소거 — 선택에 쓰지 않는다(§4.1: x_hmm 만 Platt 에 들어간다)"})
    except Exception as e:                                          # noqa: BLE001 - 조용한 실패 금지
        warns.append(f"소거(q20 Platt) 실패({type(e).__name__}: {e})")
        rows.append({"name": "q20 Platt", "computed": False, "note": f"실패: {type(e).__name__}"})
    rows += [
        {"name": "A 관측형 [r, ln σ_GK+OV] 대각 공분산", "computed": False,
         "note": "설계 단계 관측(§4.3): 전체표본 체류 38/31 · ξ AUC 0.685 · q20 0.667 — 이 실행은 계산하지 않는다"},
        {"name": "B 일변량 r · 1993~2014 동결 θ", "computed": False,
         "note": "설계 단계 관측(§4.3): 2015~24 AUC 0.665 vs x_vix 0.625 · 0.5 교차 12회/년(상태기계로 못 쓴다)"},
        {"name": "대각 공분산 HMM", "computed": False,
         "note": "설계 단계 관측(§4.3): 같은 정보 · 차이는 잡음 안 — 등록 사양은 완전 공분산 하나뿐(§3 HMM_P3)"},
    ]
    return rows


def build_selftest(mem: dict, thetas, spy: pd.Series, det: dict, registry_sha: str, sizing_sha: str,
                   warns: list[str]) -> dict:
    """§11 주간 자기검사 — 파라미터 회계·PIT 비트 동일성·그림자 보호. 실패는 경고가 아니라 flags 로 올라간다."""
    out: dict = {"hmm_n_params": None, "hmm_param_count_ok": None, "p2_param_count_ok": bool(M.PARAM_COUNT == 4),
                 "sizing_n_params": SZ.n_params(), "sizing_n_params_ok": bool(SZ.n_params() == 0),
                 "production_shadow_only": None, "pit_bit_identical": None, "pit_cuts": [],
                 "registry_sha": registry_sha, "sizing_sha": sizing_sha,
                 "registry_tuple_sha": EN.registry_tuple_sha256(), "determinism": det.get("status")}
    if thetas:
        n = RG.n_params(thetas[-1])
        out["hmm_n_params"] = int(n)
        out["hmm_param_count_ok"] = bool(n == RG.HMM_PARAM_COUNT == int(HMM_P3["param_count"]))
    try:
        EN.assert_shadow_only(mem["statuses"], where="selftest")
        out["production_shadow_only"] = True
    except ValueError as e:
        out["production_shadow_only"] = False
        warns.append(f"자기검사: 생산 평균에 그림자 멤버가 들어갔다 — {e}")
    # PIT: 무작위 절단 T 에서 filter_probabilities(obs[:T]) 가 전체 결과의 앞부분과 비트 동일해야 한다(§2 점 원칙)
    try:
        if thetas:
            obs = RG.observations(spy)
            full = RG.filter_probabilities(obs, thetas[-1])
            rng = np.random.default_rng(0)
            cuts = sorted(int(v) for v in rng.integers(200, len(obs), size=3))
            ok_all = True
            for t_cut in cuts:
                part = RG.filter_probabilities(obs.iloc[:t_cut], thetas[-1])
                same = bool(np.array_equal(part.to_numpy(dtype=float), full.iloc[:t_cut].to_numpy(dtype=float),
                                           equal_nan=True))
                out["pit_cuts"].append({"T": t_cut, "bit_identical": same})
                ok_all = ok_all and same
            out["pit_bit_identical"] = ok_all
            if not ok_all:
                warns.append("자기검사: PIT 비트 동일성 실패 — 전방 필터가 잘린 자료에서 다른 값을 냈다(§4.4)")
    except Exception as e:                                          # noqa: BLE001 - 조용한 실패 금지
        warns.append(f"자기검사(PIT) 실패({type(e).__name__}: {e})")
    return out


def build_kill_power(oos_p3: pd.DataFrame, spy: pd.Series, warns: list[str]) -> dict:
    """§8.3 킬룰 재현 — 오기각 ~12% · 검증 ~20% · 무정보 모델 기각 ~78% 를 매주 다시 만든다(못 만들면 쓰지 않는다)."""
    out: dict = {}
    for name, col in (("p2", "p_p2"), ("M1", "p_m1"), ("H", "p_h")):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                out[name] = TR.kill_replay(oos_p3, spy, col)
            warns += [f"킬룰 재현({name}): {m}" for m in _messages(caught)]
        except Exception as e:                                    # noqa: BLE001 - 조용한 실패 금지
            warns.append(f"킬룰 재현({name}) 실패({type(e).__name__}: {e}) — 카드·페이지에 검정력 문장을 쓰지 않는다")
            out[name] = None
    return out


# ------------------------------------------------------------------
# 6) 결정론 검사 (§2·§11)
# ------------------------------------------------------------------
def _theta_diff(old, new) -> float:
    """저장된 θ 와 새 θ 의 최대 절대차(같은 refit_date 끼리). 없는 재적합일은 비교 대상이 아니다."""
    if not old or not new:
        return math.nan
    by = {t.refit_date: t for t in old}
    worst = 0.0
    seen = 0
    for t in new:
        o = by.get(t.refit_date)
        if o is None:
            continue
        seen += 1
        for a, b in ((o.A, t.A), (o.mu, t.mu), (o.cov, t.cov), (o.pi, t.pi)):
            worst = max(worst, float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))))
    return worst if seen else math.nan


def determinism_check(results_dir: Path, thetas, platt_live: dict, backtest_bytes: bytes,
                      registry_sha: str, sizing_sha: str, data_sha: str, warns: list[str]) -> dict:
    """저장본과 새 산출을 비교한다. sha·입력 지문이 모두 같은데 θ·Platt·비중 경로가 허용오차 밖이면 ok=False → exit 1."""
    info = {"status": "first_run", "ok": True, "theta_max_abs_diff": None, "platt_max_abs_diff": None,
            "backtest_csv_same": None, "registry_sha_same": None, "sizing_sha_same": None,
            "data_sha_same": None, "spec_changed": False, "tol": TOL_THETA,
            "note": "저장된 Phase 3 산출물이 없어 비교 대상이 없다(첫 실행)"}
    hmm_path, model_path = results_dir / HMM_P3_PATH.name, results_dir / MODEL_P3_PATH.name
    csv_path = results_dir / BACKTEST_CSV_NAME
    if not (hmm_path.exists() and model_path.exists()):
        return info
    try:
        old_thetas, _old_live = RG.load_thetas(hmm_path)
        old_model = json.loads(model_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError) as e:
        info.update({"status": "unreadable", "ok": True,
                     "note": f"저장본을 읽지 못해 비교를 건너뛴다({type(e).__name__}: {e}) → 새 파일을 쓴다"})
        warns.append(info["note"])
        return info
    info["registry_sha_same"] = bool(old_model.get("registry_sha") == registry_sha)
    info["sizing_sha_same"] = bool(old_model.get("sizing_sha") == sizing_sha)
    info["data_sha_same"] = bool((old_model.get("run") or {}).get("data_sha256") == data_sha)
    info["theta_max_abs_diff"] = _theta_diff(old_thetas, thetas)
    old_platt = old_model.get("platt") or {}
    diffs = []
    for k in ("intercept",):
        if k in old_platt and k in platt_live:
            diffs.append(abs(_fnum(old_platt[k]) - _fnum(platt_live[k])))
    oc, nc = (old_platt.get("coef") or {}), (platt_live.get("coef") or {})
    for k in set(oc) | set(nc):
        diffs.append(abs(_fnum(oc.get(k)) - _fnum(nc.get(k))))
    info["platt_max_abs_diff"] = float(max(diffs)) if diffs else None
    info["backtest_csv_columns_same"] = None
    info["backtest_csv_sens_only_stored"] = []
    info["backtest_csv_sens_only_new"] = []
    if not csv_path.exists():
        info["backtest_csv_same"] = None
    else:
        old_csv = csv_path.read_bytes()
        if _sha256_bytes(old_csv) == _sha256_bytes(backtest_bytes):
            info["backtest_csv_same"] = True
            info["backtest_csv_columns_same"] = True
        else:
            cmp = _csv_shared_columns_equal(old_csv, backtest_bytes)
            info["backtest_csv_columns_same"] = cmp["columns_same"]
            info["backtest_csv_sens_only_stored"] = cmp["only_stored"]
            info["backtest_csv_sens_only_new"] = cmp["only_new"]
            if cmp["option_only"]:
                # 민감도 열 집합만 다르다(--all-sensitivities 유무) → 공통 열로 판정하고 차이는 공개한다
                info["backtest_csv_same"] = bool(cmp["shared_equal"])
                note = (f"{BACKTEST_CSV_NAME} 민감도 열 집합이 저장본과 다르다(--all-sensitivities 유무): "
                        f"저장본에만 {cmp['only_stored']} · 새 산출에만 {cmp['only_new']} — "
                        f"결정론 판정은 공통 열 바이트 비교로 했다"
                        f"({'동일' if cmp['shared_equal'] else '불일치'})")
                warns.append(note)
            else:
                info["backtest_csv_same"] = False
    if not (info["registry_sha_same"] and info["sizing_sha_same"]):
        info.update({"status": "spec_changed", "ok": True, "spec_changed": True,
                     "note": "registry_sha/sizing_sha 변경(코드 변경) → 새 파일을 쓴다. "
                             "VALIDATION.md §8 장부에 코드 변경 항목을 기재하라"})
        warns.append(info["note"])
        return info
    if not info["data_sha_same"]:
        info.update({"status": "data_changed", "ok": True,
                     "note": f"입력 지문이 다르다(캐시 재다운로드·배당 재조정) → 결정론 검사가 아니다. "
                             f"θ 이동 {info['theta_max_abs_diff']:.2e} · Platt 이동 "
                             f"{(info['platt_max_abs_diff'] if info['platt_max_abs_diff'] is not None else float('nan')):.2e} "
                             f"— 장부 검토 대상(§16 10)"})
        warns.append(info["note"])
        return info
    bad = []
    if info["theta_max_abs_diff"] is not None and math.isfinite(info["theta_max_abs_diff"]) \
            and info["theta_max_abs_diff"] > TOL_THETA:
        bad.append(f"θ 최대 차 {info['theta_max_abs_diff']:.2e} > {TOL_THETA:.0e}")
    if info["platt_max_abs_diff"] is not None and info["platt_max_abs_diff"] > TOL_PROB:
        bad.append(f"Platt 계수 최대 차 {info['platt_max_abs_diff']:.2e} > {TOL_PROB:.0e}")
    if info["backtest_csv_same"] is False:
        bad.append(f"{BACKTEST_CSV_NAME} 해시 불일치"
                   + ("" if info["backtest_csv_columns_same"] is not False else
                      f" (열 집합도 다르다 — 저장본에만 {info['backtest_csv_sens_only_stored']} · "
                      f"새 산출에만 {info['backtest_csv_sens_only_new']})"))
    if bad:
        info.update({"status": "mismatch", "ok": False,
                     "note": "결정론 검사 실패(같은 sha·같은 입력인데 산출이 다르다): " + " · ".join(bad)})
        return info
    info.update({"status": "compared", "ok": True,
                 "note": f"같은 sha·같은 입력 → θ·Platt 1e-7 안, {BACKTEST_CSV_NAME} "
                         + ("해시 동일" if info["backtest_csv_columns_same"] is not False else
                            "공통 열 바이트 동일(민감도 열 집합은 실행 옵션 차이)")})
    return info


# ------------------------------------------------------------------
# 7) model_p3.json (킬은 여기서 풀리지 않는다 — sticky)
# ------------------------------------------------------------------
def build_model_p3(results_dir: Path, registry_sha: str, sizing_sha: str, platt_live: dict, sz: dict,
                   adm: dict, health: dict, data_sha: str, spec_sha: str, warns: list[str]) -> dict:
    """`results/model_p3.json`. 이전 파일의 kill 상태·mode_history 를 이어받고, **킬은 절대 풀지 않는다**(§8.3)."""
    path = results_dir / MODEL_P3_PATH.name
    prev = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            warns.append(f"model_p3.json 읽기 실패({type(e).__name__}: {e}) → 새로 만든다(킬 기록은 kill_record.json 이 남긴다)")
    killed_file = (results_dir / KILL_RECORD_PATH.name).exists() or KILL_RECORD_PATH.exists() \
        or (results_dir / KILL_MANUAL_PATH.name).exists() or KILL_MANUAL_PATH.exists()
    prev_mode = str(prev.get("deploy_mode") or "tones")
    deploy_mode = "info_only" if (killed_file or prev_mode == "info_only") else "tones"
    if deploy_mode not in P3_DEPLOY_MODES:
        raise P3Fatal(f"deploy_mode {deploy_mode!r} 가 {P3_DEPLOY_MODES} 밖")
    if killed_file:
        warns.append("킬 기록이 있어 model_p3.deploy_mode 를 info_only 로 유지한다 — 주간 재적합은 킬을 풀지 못한다(§8.3)")
    # 번호 붙인 장부 항목(#8x)으로 채택된 멤버는 주간 재적합이 되돌리지 못한다(§5.4). 1일차 상태에서
    # 다시 세우되, admitted 는 발효일(effective_refit)·채택 근거와 **한 덩어리로** 이월한다.
    # (옛 코드는 아무도 쓰지 않는 'effective_from' 만 가져와 사실상 매주 채택을 지웠다.)
    statuses = dict(EN.day1_statuses())
    prev_members = prev.get("members") or {}
    carried: dict[str, dict] = {}
    for m in EN.REGISTRY:
        pm = prev_members.get(m.name) or {}
        if not isinstance(pm, dict) or str(pm.get("status")) != "admitted":
            continue
        if not pm.get("effective_refit"):                        # 조용한 실패 금지 — 이월하면 effective_statuses 가 ValueError
            warns.append(f"model_p3.json.members.{m.name} 이 admitted 인데 effective_refit 가 없다 → "
                         "이월하지 않고 1일차 상태로 둔다(번호 붙인 장부 항목으로 다시 기재하라)")
            continue
        statuses[m.name] = "admitted"
        carried[m.name] = {"effective_refit": str(pm["effective_refit"]), "admitted_on": pm.get("admitted_on"),
                           "ledger_no": str(pm.get("ledger_no") or m.ledger_no), "admission": pm.get("admission")}
    if health.get("status") == "candidate_rejected":
        if carried.pop("H", None):
            warns.append("멤버 H 가 채택 상태였으나 §5.4 탈락 검사에 걸렸다 → candidate_rejected "
                         "(강등은 번호 붙인 장부 항목으로 남겨야 한다)")
        statuses["H"] = "candidate_rejected"
    members = {}
    for m in EN.REGISTRY:
        c = carried.get(m.name) or {}
        members[m.name] = {"status": statuses[m.name], "K_s": m.k_s, "K_u": m.k_u,
                           "ledger_no": c.get("ledger_no") or m.ledger_no, "source": m.source,
                           "admission": (c.get("admission")
                                         or ((adm.get("blocks24") or {}).get("results") or {}).get(m.name))}
        if c:
            members[m.name]["effective_refit"] = c["effective_refit"]
            members[m.name]["admitted_on"] = c.get("admitted_on")
    members["H"]["health"] = health
    history = list(prev.get("mode_history") or [])
    if not history or history[-1].get("deploy_mode") != deploy_mode:
        history.append({"asof_utc": _now_utc(), "deploy_mode": deploy_mode,
                        "reason": ("kill_record" if killed_file else "weekly")})
    return {
        "schema_version": 1, "registry_sha": registry_sha, "sizing_sha": sizing_sha,
        "deploy_mode": deploy_mode, "deploy_sizing": bool(sz["retention"]["deploy_sizing"]),
        "members": members,
        "platt": platt_live,
        "sizing": {"d_max": sz["d_max"], "sigma_target": sz["sigma_target"], "k_slow": float(P3["k_slow"]),
                   "k_fast": float(P3["k_fast"]), "ledger_no": "6", "rule": SZ.SIZING_RULE,
                   "constants": SZ.sizing_constants(), "n_params": SZ.n_params()},
        "kill": prev.get("kill") or {"state": "not_started", "killed": False, "stage1_done": False,
                                     "stage2_done": False, "last_eval_month": None},
        "mode_history": history,
        "run": {"data_sha256": data_sha, "spec_sha256": spec_sha},
        "created_at_utc": _now_utc(),
        "note": "유효 모드 = p2.deploy_mode ∧ p3.deploy_mode ∧ ¬kill_record ∧ ¬kill_manual (§8.3). "
                "이 파일은 킬을 풀지 않는다 — 킬 해제는 번호 붙인 장부 항목으로만.",
    }


def platt_live_of(platts, warns: list[str]) -> dict:
    """가장 최근 재적합의 Platt 계수(= 라이브). 없으면 빈 dict + 경고."""
    if not platts:
        warns.append("Platt 모델이 없어 model_p3.platt 를 비운다 — daily 의 멤버 H 는 NaN 으로 기록된다")
        return {}
    m = platts[-1]
    return {"coef": {k: float(v) for k, v in dict(m.coef).items()}, "intercept": float(m.intercept),
            "refit_date": str(m.refit_date), "train_start": str(m.train_start), "train_end": str(m.train_end),
            "n_train": int(m.n_train), "clim": float(m.clim), "feature": EN.PLATT_FEATURE,
            "model_id": str(m.model_id), "n_params": 2}


# ------------------------------------------------------------------
# 8) --holdout-final (정보로만; #2b 와 같은 세션 1회)
# ------------------------------------------------------------------
def validate_unlock(unlock_path: Path, spec_sha: str) -> dict:
    """해제 파일의 전제(1회성·같은 코드·같은 세션)를 본다 — **자료를 만지기 전에** 부른다(§11).
    통과하면 파싱된 dict 를 돌려준다. 실패는 HoldoutRefused(종료 코드 2)."""
    unlock = json.loads(unlock_path.read_text(encoding="utf-8"))
    if unlock.get("members"):
        raise HoldoutRefused(f"{unlock_path} 에 members 블록이 이미 있다 — 멤버 홀드아웃 채점은 1회뿐이다(§6)")
    if unlock.get("spec_sha256") not in (None, spec_sha):
        raise HoldoutRefused(f"{unlock_path}.spec_sha256 {str(unlock.get('spec_sha256'))[:12]} ≠ 현재 코드 {spec_sha[:12]} — "
                             "같은 실행이 만든 unlock 파일이 아니다")
    ts = unlock.get("timestamp_utc")
    try:
        age_min = (datetime.now(timezone.utc) - datetime.strptime(str(ts), "%Y-%m-%dT%H:%M:%SZ")
                   .replace(tzinfo=timezone.utc)).total_seconds() / 60.0
    except (TypeError, ValueError) as e:
        raise HoldoutRefused(f"{unlock_path}.timestamp_utc({ts!r}) 를 읽을 수 없다 — 같은 세션인지 확인할 수 없다") from e
    if age_min > HOLDOUT_SESSION_MINUTES:
        raise HoldoutRefused(f"{unlock_path} 이 {age_min:.0f}분 전에 만들어졌다(> {HOLDOUT_SESSION_MINUTES}분) — "
                             "#2b(run_calibration.py --holdout-final)와 **같은 세션**에서만 채점한다")
    return unlock


def holdout_members(bundle, spy_all: pd.Series, first_refit, unlock_path: Path, spec_sha: str,
                    warns: list[str]) -> dict:
    """홀드아웃(2024-09-03~)에서 멤버 H·M1·ens 를 **1회** 채점해 unlock 파일의 members 블록에 적는다.
    정보로만 — 채택에 쓰지 않는다(§11 · §16 12)."""
    unlock = validate_unlock(unlock_path, spec_sha)               # 전제는 run() 이 이미 봤다 — 여기서 한 번 더(방어)
    end_full = spy_all.index[-1]
    spy = spy_all.loc[:end_full]
    tg = T.make_targets(spy)
    y = tg["y_dd5_20"]
    rd = C.refit_dates(spy.index, first=first_refit, end=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hdf, _platts, _thetas = EN.member_h_walk_forward(spy, y, rd)
        feats = F.build_features(bundle, asof=end_full)
        oos_full, _params = C.walk_forward(feats, y, ladder=M.LADDER, first_refit=first_refit, end=None,
                                           holdout_final=True)
    warns += [f"홀드아웃: {m}" for m in _messages(caught)]
    ho_idx = oos_full.index[oos_full.index >= pd.Timestamp(HOLDOUT_START)]
    fr = pd.DataFrame(index=ho_idx)
    fr["y"] = oos_full["y"].reindex(ho_idx)
    fr["clim"] = oos_full["clim"].reindex(ho_idx)
    fr["p2"] = oos_full["p_m3"].reindex(ho_idx)
    fr["M1"] = oos_full["p_m1"].reindex(ho_idx)
    fr["H"] = hdf["p_h"].reindex(ho_idx)
    fr["ens"] = fr[["p2", "M1", "H"]].mean(axis=1)
    ok = fr["y"].notna()
    out = {"start": _dstr(ho_idx[0]), "end": _dstr(ho_idx[-1]), "n": int(ok.sum()),
           "n_eff": round(int(ok.sum()) / float(P2["n_eff_div"]), 3),
           "base_rate": _fnum(fr.loc[ok, "y"].mean()), "scores": {},
           "note": "정보로만 — 채택에 쓰지 않는다(§5.4). #2b 가 이미 실행된 뒤라면 '해제 후 관측' 이다.",
           "scored_at_utc": _now_utc(), "ledger_entry": "2b"}
    yv = fr.loc[ok, "y"].to_numpy(dtype=float)
    cl = fr.loc[ok, "clim"].to_numpy(dtype=float)
    ref = float(np.mean((cl - yv) ** 2))
    for name in ("p2", "M1", "H", "ens"):
        pv = fr.loc[ok, name].to_numpy(dtype=float)
        m = np.isfinite(pv)
        if m.sum() < 2:
            out["scores"][name] = None
            continue
        bs = float(np.mean((pv[m] - yv[m]) ** 2))
        out["scores"][name] = {"n": int(m.sum()), "brier": bs,
                               "bss_clim": (float(1.0 - bs / ref) if ref > 0 else None),
                               "auc": _auc(yv[m], pv[m])}
    unlock["members"] = out
    unlock_path.write_text(json.dumps(E._jsonable(unlock), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    _log(f"[holdout] 멤버 채점 {out['start']}~{out['end']} · n={out['n']} · "
         + " · ".join(f"{k} BSS {(v or {}).get('bss_clim'):+.3f}" if v else f"{k} —" for k, v in out["scores"].items()))
    return out


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """진입점 — 실패는 예외가 아니라 종료 코드로 나간다(0 성공 · 1 실패 · 2 홀드아웃 거부)."""
    _utf8_stdout()
    try:
        return run(argv)
    except HoldoutRefused as e:
        _log(f"::error::[run_phase3] 홀드아웃 거부 — {e}")
        return 2
    except P3Fatal as e:
        _log(f"::error::[run_phase3] {e}")
        return 1


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="주간 Phase 3 산출 (국면·등록부·비중·시나리오·킬룰)")
    ap.add_argument("--end", default=None, help="하드컷 끝 (기본: HOLDOUT_START 직전 세션)")
    ap.add_argument("--d-max", type=float, default=float(P3["d_max_default"]), help="가족 위험예산 D_max (기본 0.35 → σ_T 10%%)")
    ap.add_argument("--all-sensitivities", action="store_true", help="민감도 전부를 표·CSV 에 (워크플로 기본)")
    ap.add_argument("--holdout-final", action="store_true", help="#2b 와 같은 세션에서만: 홀드아웃 멤버 채점(정보)")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--docs-dir", default=str(DOCS_DIR))
    ap.add_argument("--ledger", default=str(L.LEDGER), help="장부 CSV (라이브 패널·신선 블록; 읽기 전용)")
    ap.add_argument("--no-charts", action="store_true", help="차트 생략(테스트·디버그)")
    ap.add_argument("--no-docs", action="store_true", help="docs/ 렌더 생략 — 결과 산출물만(테스트·디버그; 워크플로는 쓰지 않는다)")
    args = ap.parse_args(argv)

    t_all = time.perf_counter()
    results_dir, docs_dir, data_dir = Path(args.results_dir), Path(args.docs_dir), Path(args.data_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    warns: list[str] = []
    require_contract(docs=not args.no_docs)                      # 계약 함수가 없으면 여기서 멈춘다(4분을 낭비하지 않는다)

    # 0) Phase 2 계약 입력(파일만 읽는다) · --holdout-final 전제
    #    자료를 만지기 전에 본다 — run_calibration 과 같은 순서(잠긴 뒤에 숫자). 전제가 깨지면
    #    특징·HMM·비중·시나리오를 한 줄도 계산·기록하지 않고 종료 코드 2 로 나간다.
    p2 = load_p2(results_dir)
    unlock = None
    if args.holdout_final:
        unlock = next((p for p in (results_dir / HOLDOUT_UNLOCK_PATH.name, HOLDOUT_UNLOCK_PATH) if p.exists()), None)
        if unlock is None:
            raise HoldoutRefused("holdout_unlock.json 이 없다 — run_calibration.py --holdout-final 을 **같은 세션**에서 "
                                 "먼저 실행해야 멤버를 채점할 수 있다(§11)")
        validate_unlock(unlock, p2["spec_sha256"])               # 1회성·같은 코드·같은 세션

    # 1) 캐시 · 하드컷
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle = D.apply_guards(D.load_cache(data_dir))
    for m in _messages(caught):
        _log(f"[캐시 경고] {m[:300]}")
    spy_all = bundle.spy_ohlc["Close"].astype(float)
    if args.end:
        end = pd.Timestamp(args.end)
        if end not in spy_all.index:
            prior = spy_all.index[spy_all.index <= end]
            if not len(prior):
                raise P3Fatal(f"--end {args.end} 이전에 세션이 없습니다")
            end = prior[-1]
    else:
        prior = spy_all.index[spy_all.index < pd.Timestamp(HOLDOUT_START)]
        end = prior[-1]
    if end >= pd.Timestamp(HOLDOUT_START):
        # 해제 파일이 없으면(= 평소) 하드컷은 그대로 위반이다. 있어도 사전 등록 OOS 는 하드컷 그대로 —
        # 홀드아웃 채점은 8) 단계가 자르지 않은 bundle/spy_all 로 따로 한다(run_calibration.py 와 같은 규칙).
        if unlock is None:
            raise P3Fatal(f"--end {_dstr(end)} 이 홀드아웃({HOLDOUT_START}) 안입니다 — 하드컷 위반(§2)")
        pre = spy_all.index[spy_all.index < pd.Timestamp(HOLDOUT_START)][-1]
        warns.append(f"--holdout-final: --end {_dstr(end)} 이 홀드아웃 안이라 하드컷 {_dstr(pre)} 로 되돌렸다 — "
                     "1993+ 비중 행·예산 사다리·시나리오 표는 홀드아웃 이전 자료로만 만든다")
        _log(f"[run_phase3] --holdout-final: --end 를 홀드아웃 직전 {_dstr(pre)} 로 되돌린다")
        end = pre
    b_cut = D.Bundle(close=bundle.close.loc[:end], spy_ohlc=bundle.spy_ohlc.loc[:end], cboe=bundle.cboe.loc[:end],
                     fg=bundle.fg.loc[:end], eod=bundle.eod.loc[:end], meta=dict(bundle.meta))
    spy = spy_all.loc[:end].rename("SPY")
    _log(f"[run_phase3] 하드컷 {_dstr(spy.index[0])} ~ {_dstr(end)} ({len(spy):,}세션) · D_max {args.d_max:.2f}")

    # 2) Phase 2 계약 입력 (0) 에서 이미 읽었다 — 여기서는 하드컷과의 정합만 본다)
    oos, sp2, v1 = p2["oos"], p2["summary_p2"], p2["v1"]
    if oos.index[-1] > end:
        raise P3Fatal(f"{CALIB_CSV} 의 마지막 세션 {_dstr(oos.index[-1])} 이 하드컷 {_dstr(end)} 보다 뒤 — 입력이 어긋난다")
    _log(f"[run_phase3] P2 OOS {_dstr(oos.index[0])}~{_dstr(oos.index[-1])} ({len(oos):,}) · deploy {p2['deploy_mode']} · "
         f"tone_model {p2['tone_model']} · 헤드라인 확률 {p2['prob_rung']}({p2['prob_col']})")
    if p2["deploy_mode"] != "tones":
        warns.append("p2 가 info_only 이므로 유효 모드도 info_only — 카드의 비중·상태 블록은 뜨지 않는다(§15 단계 0). "
                     "이 실행의 비중 백테스트·시나리오·게이지는 정보로만 남는다")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        feats = F.build_features(b_cut, asof=end)
    warns += [f"특징: {m}" for m in _messages(caught)]
    if feats.attrs.get("spec_sha256") != p2["spec_sha256"]:
        raise P3Fatal("build_features 의 spec_sha256 이 summary_p2.json 과 다름")
    data_sha = _data_sha256(b_cut.spy_ohlc, feats["vix"])
    tgts = T.make_targets(spy)
    y = tgts["y_dd5_20"]
    refits = C.refit_dates(spy.index, first=p2["first_refit"], end=HOLDOUT_START)

    # 3) 국면 · 등록부
    prev_thetas = None
    hmm_path = results_dir / HMM_P3_PATH.name
    if hmm_path.exists():
        try:
            prev_thetas, _ = RG.load_thetas(hmm_path)                 # warm start (결정적; 난수 재시작 없음)
        except (OSError, ValueError, KeyError) as e:
            warns.append(f"hmm_p3.json warm start 실패({type(e).__name__}: {e}) → 결정적 초기화로 처음부터 적합")
    mem = build_members(spy, y, refits, oos, v1, p2["prob_col"], prev_thetas, warns)
    oos_p3, thetas = mem["oos_p3"], mem["thetas"]
    theta_tbl = RG.theta_table(thetas)
    _log(f"[run_phase3] HMM 재적합 {len(thetas)}회 {mem['seconds']}s · guard "
         f"{'모두 통과' if (theta_tbl['guard'].astype(str) == 'ok').all() else '트립 있음'} · "
         f"P_high 점유 {_pct(oos_p3['p_hmm_high'].mean())} · 멤버 H AUC "
         f"{_num(_auc(oos_p3['y'].to_numpy(float), oos_p3['p_h'].to_numpy(float)), 3)}")

    lad = ladder_extension(oos_p3, oos, feats, y, mem["x_hmm_full"], p2["first_refit"], HOLDOUT_START, warns)
    adm = admissions(oos_p3, mem["statuses"], warns)
    health = member_health(oos_p3, theta_tbl, warns)
    _log("[run_phase3] 채택 검정(관측 기록, 기각 전용): "
         + " · ".join(f"{k} {v['wins']}/{v['n_blocks']}(need {v['need']}) {v['verdict']}"
                      for k, v in adm["blocks24"]["results"].items()))

    # 4) 비중
    v0_summary, v0_tone = load_v0(results_dir, warns)
    sz = build_sizing(spy, oos, v1, args.d_max, end, v0_tone, args.all_sensitivities, warns)
    sz["spy_close"] = spy
    a_row = sz["table"][(sz["table"]["row"] == SZ.ROW_ADOPTED) & (sz["table"]["window"] == SZ.WINDOW_EVAL)]
    if len(a_row):
        r0 = a_row.iloc[0]
        _log(f"[run_phase3] 비중 σ_T {_pct(sz['sigma_target'], 0)} · 채택 규칙 {SZ.WINDOW_EVAL}: CAGR {_pct(r0['cagr'], 2)} · "
             f"MaxDD {_pct(r0['max_dd'], 1)} ({r0['max_dd_date']}) · 최악 월 {_pct(r0['worst_month'])} · "
             f"변경 {_num(r0['switches_per_year'])}회/년 · 평균 비중 {_pct(r0['avg_exposure'], 0)} · "
             f"유지 조건 {'통과' if sz['retention']['deploy_sizing'] else '위반 ' + str(sz['retention']['failed'] + sz['retention']['undecided'])}")

    # 5) 시나리오
    scen = build_scenarios(spy, oos_p3, feats, oos, tgts, end, warns)

    # 6) 트랙레코드 참조 · 킬룰 재현 · 라이브 패널
    ledger_path = Path(args.ledger)
    ledger_df = None
    if ledger_path.exists():
        try:
            ledger_df = L._read(ledger_path)
        except Exception as e:                                     # noqa: BLE001 - 조용한 실패 금지
            warns.append(f"장부 읽기 실패({type(e).__name__}: {e}) → 라이브 패널 없이 진행")
    else:
        warns.append(f"장부 {ledger_path.name} 없음 → 라이브 패널·신선 블록은 비어 있다(주간 산출은 계속)")
    reference = build_reference(oos_p3, sz, scen, feats, oos, spy, ledger_df, warns)
    kill_power = build_kill_power(oos_p3, spy, warns)
    # 주간 재현(D10): 이번 실행이 다시 만든 p·P_high·w_exec 를 장부와 대 본다. 장부는 기록이라 고치지 않고 **세기만** 한다.
    # 하드컷 때문에 겹치는 세션이 없으면(라이브 행이 전부 홀드아웃 뒤) replay_check 가 n_compared=0 으로 정직하게 말한다.
    recomputed = pd.DataFrame({
        "asof": [_dstr(d) for d in oos_p3.index],
        "prob_dd5_20": oos_p3["p_p2"].to_numpy(dtype=float),
        "p3_hmm_p_high": oos_p3["p_hmm_high"].to_numpy(dtype=float),
        "p3_w_exec": sz["path"]["w_exec"].reindex(oos_p3.index).to_numpy(dtype=float),
    })
    # model_p3 을 여기서 만든다(파일만 읽는다): 트랙 블록의 유효 모드와 킬 상태는 model_p3 이 단일 원천이다.
    # 리터럴 "tones" 를 넘기면 AND 에서 p3 항이 사라져, 킬이 걸린 모델을 라이브 기록 페이지가
    # '배치됨' 으로 광고한다(§8.3). prev_kill 을 빼면 주간 재적합이 킬을 푼다(§16-11).
    registry_sha, sizing_sha = EN.registry_sha256(), SZ.sizing_sha256()
    platt_live = platt_live_of(mem["platts"], warns)
    model_p3 = build_model_p3(results_dir, registry_sha, sizing_sha, platt_live, sz, adm, health,
                              data_sha, p2["spec_sha256"], warns)
    prev_kill = model_p3.get("kill")                             # 직전 파일의 sticky 상태(killed·stage·재평가 리듬)
    if str(model_p3["deploy_mode"]) == "info_only" and not (prev_kill or {}).get("killed"):
        # 킬 기록 파일은 있는데 직전 kill 블록이 비어 있는 경우(수동 킬·백업 복원) — 어떤 경우에도 풀지 않는다
        prev_kill = {**(prev_kill or {}), "killed": True}
    eff_mode = TR.effective_mode(p2["deploy_mode"], model_p3["deploy_mode"],
                                 kill_record_path=results_dir / KILL_RECORD_PATH.name,
                                 kill_manual_path=results_dir / KILL_MANUAL_PATH.name)
    track_p3 = None
    if ledger_df is not None and len(ledger_df):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                track_p3 = TR.summary_p3(ledger_df, spy_all, reference, ledger_df["asof"].max(),
                                         prev_kill=prev_kill,
                                         recomputed=recomputed,
                                         p2_deploy_mode=p2["deploy_mode"],
                                         p3_deploy_mode=model_p3["deploy_mode"],
                                         deploy_sizing=bool(sz["retention"]["deploy_sizing"]))
            warns += [f"트랙레코드: {m}" for m in _messages(caught)]
            warns += [f"트랙레코드: {m}" for m in (track_p3.get("notes") or [])]
            tm = str(track_p3.get("effective_mode") or eff_mode)
            if tm != eff_mode:                                   # 페이지와 요약이 서로 다른 모드를 말하면 쓰지 않는다(§8.3)
                raise P3Fatal(f"유효 모드 불일치 — track_record_p3 {tm} ≠ summary {eff_mode} "
                              f"(p2 {p2['deploy_mode']} · p3 {model_p3['deploy_mode']}). "
                              "라이브 기록 페이지가 죽은 모델의 톤을 광고하게 둘 수 없다")
            n_al = TR.append_alarms(track_p3.get("alarms") or [], results_dir / ALARMS_PATH.name)
            if n_al:
                warns.append(f"경보 {n_al}건을 {ALARMS_PATH.name} 에 기록(주간)")
        except P3Fatal:                                            # 모드 불일치는 경고로 낮추지 않는다
            raise
        except Exception as e:                                     # noqa: BLE001
            warns.append(f"track.summary_p3 실패({type(e).__name__}: {e}) → track_record_p3.json 을 쓰지 않는다")
            track_p3 = None

    # 7) 홀드아웃 멤버 채점 (정보; #2b 와 같은 세션 1회)
    holdout_block = None
    if args.holdout_final:                                       # 전제는 0) 에서 이미 통과했다(unlock 은 None 이 아니다)
        holdout_block = holdout_members(bundle, spy_all, p2["first_refit"], unlock, p2["spec_sha256"], warns)

    # 8) 산출물 조립 (registry_sha·sizing_sha·platt_live·model_p3 은 6) 에서 이미 만들었다)
    bt = backtest_p3_frame(sz, oos, args.all_sensitivities)
    bt_bytes = _csv_bytes(bt)
    det = determinism_check(results_dir, thetas, platt_live, bt_bytes, registry_sha, sizing_sha, data_sha, warns)
    if not det["ok"]:
        _log(f"::error::{det['note']}")
        return 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        blk24 = C.block_scores(lad["frame"], [(a, b) for a, b in BLOCKS_24], model_col="p_h")
        blk18 = C.block_scores(lad["frame"], [(a, b) for a, b in BLOCKS_18], model_col="p_h")
    warns += [f"블록 표(H): {m}" for m in _messages(caught)]
    ablations = build_ablations(oos_p3, mem["q20_full"], y, refits, warns)
    selftest = build_selftest(mem, thetas, spy, det, registry_sha, sizing_sha, warns)

    flags: list[str] = []
    if not sz["retention"]["deploy_sizing"]:
        flags.append(f"유지 조건 위반/미판정 {sz['retention']['failed'] + sz['retention']['undecided']} → 비중 카드 숨김(§6.4)")
    if sz["retention"]["narrow"]:
        flags.append(f"유지 조건 근소 통과 {sz['retention']['narrow']} — 자료가 조금만 바뀌어도 뒤집힌다(§6.4)")
    if health["status"] == "candidate_rejected":
        flags.append("멤버 H candidate_rejected — 게이지 숨김(§5.4)")
    if det.get("spec_changed"):
        flags.append("registry_sha/sizing_sha 변경 — VALIDATION §8 장부 기재 요구")
    for k in ("hmm_param_count_ok", "p2_param_count_ok", "sizing_n_params_ok", "production_shadow_only",
              "pit_bit_identical"):
        if selftest.get(k) is False:
            flags.append(f"자기검사 실패: {k} (§11)")
    if p2["deploy_mode"] != "tones":
        flags.append("p2 info_only → 유효 모드 info_only: 톤·비중·상태 카드 없음(§15 단계 0)")

    import sklearn                                                  # noqa: WPS433 - 버전 기록용
    summary = {
        "schema_version": 1,
        "disclosure": DISCLOSURE,
        "run": {"generated_at_utc": _now_utc(), "end": _dstr(end), "start": _dstr(spy.index[0]),
                "n_sessions": int(len(spy)), "n_oos": int(len(oos_p3)),
                "spec_sha256": p2["spec_sha256"], "feature_rule": p2["feature_rule"],
                "registry_sha": registry_sha, "sizing_sha": sizing_sha, "data_sha256": data_sha,
                "regime_sha": RG.regime_sha256(), "registry_tuple_sha": EN.registry_tuple_sha256(),
                "d_max": float(args.d_max), "sigma_target": sz["sigma_target"],
                "holdout_start": HOLDOUT_START, "hard_cut": True,
                "holdout_final": bool(args.holdout_final), "all_sensitivities": bool(args.all_sensitivities),
                "first_refit": p2["first_refit"], "n_refits": len(thetas), "seed": int(KILL_P3["seed"]),
                "purge": int(HMM_P3["purge"]), "spec_changed": bool(det.get("spec_changed")),
                "obs_spec": RG.OBS_SPEC, "cost_bps": float(P3["cost_bps"]), "cash_return": float(P3["cash_return"]),
                "p3_deploy_mode": model_p3["deploy_mode"], "deploy_mode": model_p3["deploy_mode"],
                "p2_deploy_mode": p2["deploy_mode"], "p2_tone_model": p2["tone_model"],
                "p2_prob_rung": p2["prob_rung"], "effective_mode": eff_mode,
                "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                "sklearn": sklearn.__version__, "platform": platform.platform(),
                "cache": {"fetched_at_utc": bundle.meta.get("fetched_at_utc"), "spy_last": bundle.meta.get("spy_last"),
                          "spy_last_used": _dstr(end)},
                "runtime_sec": None},
        "params": {"production_K_s": EN.production_param_count(mem["statuses"]),
                   "shadow": EN.shadow_param_counts(mem["statuses"]),
                   "sizing_n_params": SZ.n_params(), "hmm_param_count": RG.HMM_PARAM_COUNT,
                   "note": "Phase 3 는 생산 확률에 적합 파라미터를 0개 더한다 (§14)"},
        "v0_reference": _records(RPT.v0_headline(v0_summary)),
        "registry": _drop_none(_records(EN.registry_table(
            oos_p3, mem["statuses"],
            admissions={k: (adm["blocks24"]["results"] or {}).get(k) for k in EN.REGISTRY_NAMES}))),
        "ladder": _records(lad["blocks24"]),                       # §10 regime ② 는 이 표를 그대로 그린다
        "ladder_blocks18": _records(lad["blocks18"]), "ladder_steps": lad["steps"], "ladder_m4a": lad["m4a"],
        "blocks": {"24": _records(blk24), "18": _records(blk18),
                   "definition": {"24": [list(b) for b in BLOCKS_24], "18": [list(b) for b in BLOCKS_18]},
                   "model_col": "p_h"},
        "admission": _drop_none(adm), "admission_power": adm["power_table"],
        "ablations": ablations,
        "selftest": selftest,
        "acceptance": (sp2.get("acceptance") or {}),
        "deploy_mode": model_p3["deploy_mode"],
        "model_p3": {"deploy_mode": model_p3["deploy_mode"], "registry_sha": registry_sha, "sizing_sha": sizing_sha,
                     "deploy_sizing": model_p3["deploy_sizing"], "kill": model_p3["kill"]},
        "track": (_records(track_p3) if track_p3 is not None else {}),
        "kill": _records((track_p3 or {}).get("kill") or model_p3["kill"]),
        "alarms": _records((track_p3 or {}).get("alarms") or []),
        "reliability": {"H": _records(C.reliability_table(oos_p3["p_h"], oos_p3["y"])),
                        "ens": _records(C.reliability_table(mem["ens3"], oos_p3["y"]))},
        "murphy": {"H": _records(C.murphy_decomposition(oos_p3["p_h"], oos_p3["y"]))},
        "era_auc": era_auc_p3(oos_p3, oos, feats),
        "hmm": {"theta_table": _records(theta_tbl), "guard_log": mem["hmm_attrs"].get("guard_log"),
                "obs_spec": RG.OBS_SPEC, "param_count": RG.HMM_PARAM_COUNT,
                "timing": {"walk_forward_sec": mem["seconds"], "n_refits": len(thetas)},
                "occupancy": _fnum(oos_p3["p_hmm_high"].mean()),
                "auc_p_high": _auc(oos_p3["y"].to_numpy(float), oos_p3["p_hmm_high"].to_numpy(float)),
                "health": health},
        "decision_ens3": {"note": "3-멤버 평균 위의 결정층 — 정보로만(생산 상태는 backtest_v1.csv 의 배포 단)",
                          "occupancy": {s: _fnum((oos_p3["state_ens3"].astype("string") == s).mean())
                                        for s in ("normal", "caution", "reduce")},
                          "n_changes": int((oos_p3["state_ens3"].astype("string")
                                            != oos_p3["state_ens3"].astype("string").shift(1)).sum() - 1)},
        "sizing": {"table": _records(sz["table"]), "ladder": _records(sz["ladder"]),
                   # 리포트 ⑥ 은 §6.3 형식의 표를 읽는다 — 이름만 넘기면 페이지가 비어 버린다
                   "sensitivities": _records(sz["sensitivity_table"]),
                   "sensitivity_names": sorted(sz["sensitivities"]),
                   "sensitivities_in_table": sz["sensitivities_used"],
                   "retention": _records(sz["retention"]), "deploy_sizing": bool(sz["retention"]["deploy_sizing"]),
                   "episode_pnl": _records(sz["episode_pnl"]),
                   "relative": _records({**sz["relative"], "calendar_year": sz["calendar_relative"]}),
                   "budget_sentence": SZ.budget_sentence(args.d_max, sz["sigma_target"],
                                                         maxdd_decision=(_fnum(a_row.iloc[0]["max_dd"]) if len(a_row) else None),
                                                         maxdd_vol_only=_vol_only_maxdd(sz["table"])),
                   "honest_reading": SZ.honest_reading(_honest_results(sz)), "honest_results": _honest_results(sz),
                   "rule": SZ.SIZING_RULE, "constants": SZ.sizing_constants(),
                   "columns_ko": dict(SZ.BACKTEST_COLUMNS_KO)},
        "scenarios": {"bins": _records(scen["bins"]), "states": _records(scen["states"]),
                      "episodes": _records(scen["episodes"]), "coverage": _records(scen["coverage"]),
                      "drawdown": _records(scen["drawdown"]), "ranges": _records(scen["ranges"]),
                      "spread_sentence": scen["spread_sentence"]},
        "kill_power": _records(kill_power),
        "reference": _records(reference),
        "holdout_members": _records(holdout_block),
        "determinism": _records(det),
        "flags": flags,
        "warnings": warns,
        "artifacts": {"oos_csv": str(results_dir / OOS_CSV_NAME), "backtest_csv": str(results_dir / BACKTEST_CSV_NAME),
                      "summary_json": str(results_dir / SUMMARY_NAME), "model_json": str(results_dir / MODEL_P3_PATH.name),
                      "hmm_json": str(results_dir / HMM_P3_PATH.name),
                      "track_json": str(results_dir / TRACK_P3_PATH.name),
                      "sizing_html": str(docs_dir / SIZING_HTML), "regime_html": str(docs_dir / REGIME_HTML),
                      "track_html": str(docs_dir / TRACK_HTML)},
    }

    # 9) 저장 — 쓰기 직전 하드컷 재확인(run_backtest_v1.py 와 같은 안전망; 조용히 새 나가지 않는다)
    #    --holdout-final 에서도 끄지 않는다: 사전 등록 산출물은 어느 모드에서나 하드컷이다
    #    (홀드아웃 채점은 8) 이 unlock 파일 안에만 적는다). 예외를 두면 안전망이 정작 필요할 때 꺼져 있다.
    oos_out = oos_p3.copy()
    for name, idx in ((OOS_CSV_NAME, oos_out.index), (BACKTEST_CSV_NAME, bt.index)):
        leak = pd.DatetimeIndex(idx)[pd.DatetimeIndex(idx) >= pd.Timestamp(HOLDOUT_START)]
        if len(leak):
            raise P3Fatal(f"{name} 에 홀드아웃({HOLDOUT_START}~) 행 {len(leak)}개 "
                          f"({_dstr(leak[0])}~{_dstr(leak[-1])}) — 하드컷 위반(§2), 쓰지 않는다")
    (results_dir / OOS_CSV_NAME).write_bytes(_csv_bytes(oos_out))
    (results_dir / BACKTEST_CSV_NAME).write_bytes(bt_bytes)
    RG.save_thetas(thetas, results_dir / HMM_P3_PATH.name, live=(thetas[-1] if thetas else None),
                   guard_log=mem["hmm_attrs"].get("guard_log"), registry_sha=registry_sha)
    with open(results_dir / MODEL_P3_PATH.name, "w", encoding="utf-8", newline="\n") as f:
        json.dump(E._jsonable(model_p3), f, ensure_ascii=False, indent=1, allow_nan=False)
        f.write("\n")
    if track_p3 is not None:
        TR.save_summary_p3(track_p3, results_dir / TRACK_P3_PATH.name)
    summary["run"]["runtime_sec"] = round(time.perf_counter() - t_all, 1)
    summary_json = E._jsonable(summary)
    with open(results_dir / SUMMARY_NAME, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=1, allow_nan=False)
        f.write("\n")

    # 10) 페이지
    if not args.no_docs:
        charts: dict[str, bytes] = {}
        if not args.no_charts:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                charts = RPT.charts_p3(oos_p3, bt, spy, summary_json, ledger_df)
            warns += [f"차트: {m}" for m in _messages(caught)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            RPT.render_sizing_report(summary_json, p2["summary_v1"], v0_summary, docs_dir / SIZING_HTML, charts)
            RPT.render_regime_report(summary_json, docs_dir / REGIME_HTML, charts)
            RPT.render_track_record(E._jsonable(track_p3) if track_p3 is not None else {}, summary_json,
                                    docs_dir / TRACK_HTML, charts)
        for m in _messages(caught):
            _log(f"[리포트 경고] {m[:300]}")

    _log(f"[run_phase3] 저장: {OOS_CSV_NAME}, {BACKTEST_CSV_NAME}, {SUMMARY_NAME}, {MODEL_P3_PATH.name}, "
         f"{HMM_P3_PATH.name}" + (f", {TRACK_P3_PATH.name}" if track_p3 is not None else "")
         + ("" if args.no_docs else f", {SIZING_HTML}, {REGIME_HTML}, {TRACK_HTML}")
         + f" · 결정론 {det['status']} · 경고 {len(warns)}건 · 총 {summary['run']['runtime_sec']}s")
    for fl in flags:
        _log(f"  ! {fl}")
    for w in warns:
        _log(f"  - {w[:300]}")
    return 0


def _vol_only_maxdd(table: pd.DataFrame):
    row = table[(table["row"] == SZ.ROW_VOL_ONLY) & (table["window"] == SZ.WINDOW_VOL_ONLY)]
    return _fnum(row.iloc[0]["max_dd"]) if len(row) else None


def _signed_pct(v, digits: int = 1) -> str:
    """카드·문장용 부호 표기(−는 U+2212; §6.5 문장이 이 형식으로 쓰여 있다)."""
    f = _fnum(v)
    if not math.isfinite(f):
        return "—"
    return ("−" if f < 0 else "") + f"{abs(f) * 100:.{digits}f}%"


def _honest_results(sz: dict) -> dict | None:
    """§6.5 정직한 읽기의 공식 수치(문자열) — 필요한 행이 하나라도 없으면 None(설계 단계 근사 라벨을 유지한다)."""
    t = sz["table"]

    def _row(name, win):
        r = t[(t["row"] == name) & (t["window"] == win)]
        return None if not len(r) else r.iloc[0]
    vt14, dec = _row(SZ.ROW_VT14, SZ.WINDOW_EVAL), _row(SZ.ROW_P2_DECISION, SZ.WINDOW_EVAL)
    volo, nofl = _row(SZ.ROW_VOL_ONLY, SZ.WINDOW_EVAL), _row("S-nofloor", SZ.WINDOW_EVAL)
    adopted, vo93 = _row(SZ.ROW_ADOPTED, SZ.WINDOW_EVAL), _row(SZ.ROW_VOL_ONLY, SZ.WINDOW_VOL_ONLY)
    if any(r is None for r in (vt14, dec, volo, nofl, adopted, vo93)):
        return None
    return {"vt14_cagr": f"{float(vt14['cagr']) * 100:.2f}%", "vt14_dd": _signed_pct(vt14["max_dd"]),
            "p2_cagr": f"{float(dec['cagr']) * 100:.2f}%", "p2_dd": _signed_pct(dec["max_dd"]),
            "vt14_sw": f"{float(vt14['switches_per_year']):.1f}", "p2_sw": f"{float(dec['switches_per_year']):.1f}",
            "volonly_dd": _signed_pct(volo["max_dd"]), "nofloor_dd": _signed_pct(nofl["max_dd"]),
            "adopted_dd": _signed_pct(adopted["max_dd"]), "mild_bear_dd": _signed_pct(vo93["max_dd"]),
            "share_behind": _signed_pct(sz["calendar_relative"].get("share_behind"), 0)}


if __name__ == "__main__":
    sys.exit(main())
