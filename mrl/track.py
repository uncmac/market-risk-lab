# -*- coding: utf-8 -*-
"""트랙레코드 · 드리프트 경보 · 킬룰 (ARCHITECTURE_PHASE3.md §8).

이 모듈이 하는 일은 네 가지뿐이다 — **세고, 비교하고, 플래그를 세우고, 출구를 연다**. 적합 파라미터는 0개다.

1. **라이브 창과 채점**(§8.1): 장부(`results/track_record.csv`)에서만 다시 계산한다. 새 백테스트에서 숫자를
   가져오지 않는다 — 자료 개정이 기록을 바꾸면 D7·D10 이 그것을 잡는 것이지 기록을 다시 쓰지 않는다.
2. **드리프트 경보 D1~D11**(§8.2): 사전 등록 임계(`config.P3_DRIFT`; 2003~24 참조분포의 p5/p95)를 넘으면
   **플래그·기록만** 한다. 자동 재조정은 없다. 임계 변경은 번호 붙인 장부 항목으로만.
3. **2단계 킬룰**(§8.3): 1차 = 36개월(BSS_clim 점추정 ≤ 0 → `info_only`, sticky; > 0 → `provisional`),
   2차 = 36개월 이후 실현 ≥5% 에피소드 8회 또는 60개월 중 **먼저** 오는 때, 이후 12개월마다 재평가.
   라벨은 `validated`(부트스트랩 CI 하한 > 0) / `provisional` / `info_only`. 수동 킬은 `kill_manual.json`.
   복귀는 코드에 없다(문서 절차: 새 장부 항목 + 12개월 + 전체 라이브 창 CI 하한 > 0).
4. **지평별 표시 규칙**(§8.4)과 **라이브-vs-백테스트 패널**(§8.5): 소표본 문구를 산문이 아니라 코드로 고정한다.
   n_eff < 6 이면 BSS 숫자가 아예 나오지 않고, 판정일 전에는 판정어가 나오지 않는다.

**홀드아웃 보호(VALIDATION.md §6 · 과제 계약)**: `results/holdout_unlock.json` 이 없으면
`HOLDOUT_START`(2024-09-01) 이후 세션은 **채점되지 않는다**. `mrl/ledger.py::_p2_block` 과 같은 의미론이며
(그 파일은 건드리지 않는다), 여기서도 킬룰 시계·경보·패널 전부에 같은 게이트를 건다. `unlock_path=None` 을
주면 게이트를 끈다(테스트·사후 분석 전용).

**킬은 `model_p2.json` 을 건드리지 않는다**: 유효 모드 = `p2.deploy_mode == "tones"` ∧ `p3.deploy_mode == "tones"`
∧ ¬`kill_record.json` ∧ ¬`kill_manual.json` 의 AND 이므로(=`effective_mode()`), 주간 `run_calibration.py` 가
무엇을 다시 쓰더라도 킬이 풀리지 않는다.

결정론: 모든 부트스트랩·널 표본은 seed 0. 시각(UTC)은 감사 기록용 필드에만 들어가고 어떤 판정에도 쓰이지 않는다.
"""
from __future__ import annotations

import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from mrl.config import (ALARMS_PATH, HOLDOUT_START, HOLDOUT_UNLOCK_PATH, KILL_MANUAL_PATH, KILL_P3,
                        KILL_RECORD_PATH, MODEL_P3_PATH, P2, P3, P3_DRIFT, TRACK_P3_PATH)
from mrl.targets import episodes as _episodes_table

__all__ = [
    "SCHEMA_VERSION", "SCORE_REFS", "MEMBER_COLUMNS", "KILL_STATES", "ALARM_CODES", "ALARM_ACTIONS",
    "HORIZON_STAGES", "HORIZON_SENTENCES", "HORIZON_FOOTNOTE", "HORIZON_HIDDEN", "HORIZON_SHOWN",
    "holdout_gate", "live_start", "episodes_realized", "breach_dates", "months_elapsed", "bss_block_ci",
    "score", "hist_pct", "window_distribution", "kill_status", "kill_apply", "kill_replay",
    "alarms", "append_alarms", "replay_check", "horizon_stage", "horizon_copy", "may_show",
    "live_panel", "summary_p3", "effective_mode",
]

SCHEMA_VERSION = 3
N_EFF_DIV = int(P2["n_eff_div"])                     # 20 — 독립 창 환산(라벨 지평 20세션)
_DD5_WINDOW = int(P2["h"])                           # 20

# 채점 기준(장부 열). ledger._P2_SCORE_REFS 와 같은 이름·같은 열을 쓴다(같은 숫자가 나와야 한다).
SCORE_REFS = (("clim", "p2_clim"), ("m1", "p2_p_m1"), ("vix", "p2_p_vix"), ("vix_bgk", "p2_p_vix_bgk"))
# 등록부 멤버 → 장부의 확률 열 (§5.1; 생산 확률은 prob_dd5_20 = 배포 단)
MEMBER_COLUMNS = {"p2": "prob_dd5_20", "M1": "p2_p_m1", "H": "p3_p_h"}

KILL_STATES = ("not_started", "not_due", "provisional", "validated", "info_only", "manual_kill")
_KILL_PREV_DEFAULT = {"killed": False, "stage1_done": False, "stage2_done": False, "last_eval_month": None}

ALARM_CODES = ("D1_p_level", "D2_bss", "D3_vol_fc", "D4_vol_target", "D4b_budget", "D5_stuck", "D6_churn",
               "D7_parity", "D8_coverage", "D9_hmm", "D10_replay", "D11_feature_range")
# 효과(§8.2 표). 'exit_1' 은 daily.py 가 커밋 없이 종료해야 하는 유일한 코드다.
ALARM_ACTIONS = {"D1_p_level": "display", "D2_bss": "warn_before_kill", "D3_vol_fc": "display",
                 "D4_vol_target": "halt_sizing_card", "D4b_budget": "require_ledger_entry",
                 "D5_stuck": "display", "D6_churn": "halt_sizing_card", "D7_parity": "exit_1",
                 "D8_coverage": "display", "D9_hmm": "hide_gauge_reject_member", "D10_replay": "display",
                 "D11_feature_range": "display"}
ALARM_CSV_COLUMNS = ("asof", "code", "value", "threshold")

# ------------------------------------------------------------------
# 지평별 표시 규칙 (§8.4) — 문장은 문서에서 그대로 옮긴 것이며 코드가 강제한다.
# ------------------------------------------------------------------
HORIZON_STAGES = ("0-3m", "6m", "12m", "36m")
HORIZON_SENTENCES = {
    "0-3m": "독립 창 3개 — 어떤 결론도 없음. 지금 보이는 것은 파이프라인이 매일 돌고 기록이 맞게 쌓인다는 것뿐. "
            "과거 3개월 창의 skill 은 −0.17~+0.85(10/90분위)였다.",
    "6m": "6개 창: 구간 폭이 skill 자체보다 넓다(과거 6개월 창의 36% 가 음수, 10/90 −0.12~+0.81).",
    "12m": "1년은 모델의 생사를 말하지 못한다: 같은 모델의 과거 1년 BSS 는 −0.07(하위 10%)에서 +0.50 사이였고 "
           "CI 하한 > 0 은 26~29% 뿐. 킬룰은 36개월.",
    "36m": "≥8 에피소드로는 큰 실패만 걸러진다; +1~2% 우위는 인증 불가(P2 §16.4).",
}
HORIZON_FOOTNOTE = ("라이브 기록은 해마다 독립 창 ~12개·에피소드 ~1개가 쌓인다. 판정일은 36개월(1차)과 "
                    "8회 또는 60개월(2차); 그때까지 이 페이지는 일기이지 판결이 아니다.")
# 보이는 것(§8.4 표) — 상위 단계는 하위 단계의 항목을 모두 포함한다.
HORIZON_SHOWN = {
    "0-3m": ("sessions", "missing_days", "alarms", "mean_p_vs_clim", "state_days", "w_path",
             "rule_vs_bh_63", "band_hit_raw", "vol_log_mae", "shadow_members"),
    "6m": ("brier", "bss", "ci", "reliability_2bin", "episodes", "drift_table", "w_distribution", "hist_pct"),
    "12m": ("bss_vs_backtest_252", "reliability_pooled", "w_annual_card", "coverage_wilson",
            "realized_vol_ratio", "midterm_check"),
    "36m": ("verdict", "ci_label", "mode_switch_record", "window_distribution_36m", "member_verdicts",
            "fresh_block_admission"),
}
# 보이지 않는 것 — 단계별로 **코드가 막는** 키(하위 단계일수록 더 많이 막는다).
HORIZON_HIDDEN = {
    "0-3m": ("brier", "bss", "ci", "reliability_2bin", "reliability_pooled", "hist_pct", "verdict", "ci_label"),
    "6m": ("verdict", "ci_label"),
    "12m": ("verdict", "ci_label"),
    "36m": (),
}
_HIDDEN_REASON = {
    "0-3m": "n_eff < 6 — §8.4: 3개월 창의 skill 은 잡음(10/90분위 ±0.8)이라 회색으로도 표시하지 않는다",
    "6m": "판정일(36개월) 전 — §8.4: 판정어 금지",
    "12m": "판정일(36개월) 전 — §8.4: 판정어 금지",
}


# ==================================================================
# 0. 공용 유틸
# ==================================================================
def _utcnow_iso() -> str:
    """감사 기록용 UTC 시각. 어떤 판정에도 쓰이지 않는다(결정론 규칙)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ts(v):
    """None/NaT 를 그대로 흘리는 Timestamp 정규화(자정)."""
    if v is None:
        return None
    t = pd.Timestamp(v)
    if pd.isna(t):
        return None
    if t.tz is not None:
        t = t.tz_localize(None)
    return t.normalize()


def _jf(v):
    """JSON 안전 float (NaN/inf → None)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(f) else f


def _close_series(spy_close) -> pd.Series:
    """SPY 종가를 tz-naive·정렬·중복제거된 Series 로. (ledger._prep_close 와 같은 의미론)"""
    if isinstance(spy_close, pd.DataFrame):
        if "Close" in spy_close.columns:
            spy_close = spy_close["Close"]
        elif "SPY" in spy_close.columns:
            spy_close = spy_close["SPY"]
        elif spy_close.shape[1] == 1:
            spy_close = spy_close.iloc[:, 0]
        else:
            raise ValueError("spy_close DataFrame 에서 종가 열을 찾을 수 없습니다 (Close/SPY)")
    s = pd.Series(spy_close).dropna().astype(float)
    if len(s) == 0:
        raise ValueError("spy_close 가 비어 있습니다")
    idx = pd.DatetimeIndex(pd.to_datetime(s.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _frame_dates(frame: pd.DataFrame) -> pd.Series:
    """프레임의 세션 날짜: 'asof' 열 → 'date' 열 → DatetimeIndex 순. 어느 것도 없으면 ValueError
    (조용한 실패 금지 — 날짜를 모르면 홀드아웃 게이트를 걸 수 없다)."""
    for c in ("asof", "date"):
        if c in frame.columns:
            return pd.to_datetime(frame[c], errors="coerce")
    idx = frame.index
    if isinstance(idx, pd.DatetimeIndex):
        return pd.Series(pd.to_datetime(idx), index=frame.index)
    raise ValueError("프레임에서 세션 날짜를 찾을 수 없습니다 ('asof'/'date' 열 또는 DatetimeIndex 필요)")


def _resolve_unlock(unlock_path):
    """홀드아웃 해제 파일 경로 정규화. None 이면 게이트 해제, 디렉터리를 주면 그 안의 표준 파일명."""
    if unlock_path is None:
        return None
    p = Path(unlock_path)
    if p.is_dir():
        p = p / HOLDOUT_UNLOCK_PATH.name
    return p


def holdout_gate(frame: pd.DataFrame, unlock_path=HOLDOUT_UNLOCK_PATH,
                 notes: list | None = None) -> tuple[pd.DataFrame, bool]:
    """홀드아웃 보호(VALIDATION.md §6 · `ledger._p2_block` 과 같은 의미론).

    `unlock_path` 파일이 **없으면** `HOLDOUT_START`(2024-09-01) 이후 세션을 잘라낸다. 라이브 세션은 전부
    홀드아웃 구간 안이므로(끝이 없다), 해제 전에 여기서 채점하면 `run_calibration --holdout-final` 이
    "오염되지 않은 1회" 로 쓸 세션을 미리 태우는 셈이 된다.

    반환 (게이트를 통과한 프레임, locked). `unlock_path=None` 이면 게이트를 끈다(테스트·사후 분석 전용).
    """
    p = _resolve_unlock(unlock_path)
    if p is None or len(frame) == 0:
        return frame, False
    if p.exists():
        return frame, False
    dates = _frame_dates(frame)
    in_ho = dates >= pd.Timestamp(HOLDOUT_START)
    if not bool(in_ho.any()):
        return frame, False
    if notes is not None:
        notes.append(f"홀드아웃 미해제({p.name} 없음) → {HOLDOUT_START} 이후 {int(in_ho.sum())}행은 "
                     "라이브 채점·킬룰·경보에서 제외 (최종 검증 1회 전 접근 금지 — VALIDATION.md §6)")
    return frame.loc[~in_ho.to_numpy()], True


def _numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _tail(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df if (n is None or len(df) <= n) else df.iloc[-int(n):]


def _auc(y, s) -> float:
    """Mann-Whitney AUC (동률은 평균 순위). 한 쪽 라벨이 없으면 NaN."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    pos = y > 0.5
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1))


def _count_changes(v: np.ndarray, tol: float = 1e-12) -> int:
    """유한값 사이의 변경 횟수(결측은 건너뛰고 직전 유효값과 비교)."""
    prev = None
    n = 0
    for x in np.asarray(v, dtype=float):
        if not math.isfinite(x):
            continue
        if prev is not None and abs(x - prev) > tol:
            n += 1
        prev = x
    return n


def _max_run(flags: np.ndarray) -> int:
    """True 가 연속된 최장 런 길이."""
    best = cur = 0
    for f in np.asarray(flags, dtype=bool):
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return int(best)


def effective_mode(p2_deploy_mode, p3_deploy_mode, *, kill_record_path=KILL_RECORD_PATH,
                   kill_manual_path=KILL_MANUAL_PATH) -> str:
    """유효 배치 모드 = p2 ∧ p3 ∧ ¬kill_record ∧ ¬kill_manual (§8.3 '모드 스위치 역학').

    킬은 `model_p2.json` 을 건드리지 않으므로 주간 재적합이 킬을 풀 수 없다 — 이 AND 가 유일한 판정이다.
    """
    ok = (str(p2_deploy_mode) == "tones" and str(p3_deploy_mode) == "tones"
          and not Path(kill_record_path).exists() and not Path(kill_manual_path).exists())
    return "tones" if ok else "info_only"


# ==================================================================
# 1. 라이브 창 (§8.1)
# ==================================================================
def live_start(ledger: pd.DataFrame, col: str = "prob_dd5_20",
               mode_col: str | None = "p2_deploy_mode") -> pd.Timestamp | None:
    """라이브 시작일.

    * 생산(production): `col` 값이 있고 `mode_col == "tones"` 인 **첫 행**의 asof = Phase 2 톤 가동일.
      그 전 행(Phase 2 info_only 기간·홀드아웃 재현 행)은 '재현 — 라이브 아님' 이며 킬룰·채택에 들어가지 않는다.
    * 그림자 멤버: `mode_col=None` 으로 부르면 자기 열(`p3_p_h`, `p2_p_m1`)이 처음 기록된 날.
    * 해당 행이 없으면 None(= 아직 라이브가 시작되지 않음).
    """
    if ledger is None or len(ledger) == 0 or col not in ledger.columns:
        return None
    df = ledger.copy()
    df["_asof"] = pd.to_datetime(_frame_dates(df), errors="coerce")
    ok = pd.to_numeric(df[col], errors="coerce").notna()
    if mode_col is not None:
        if mode_col not in df.columns:
            warnings.warn(f"live_start: 장부에 {mode_col} 열이 없습니다 → 배치 모드 조건을 적용할 수 없습니다")
            return None
        ok &= df[mode_col].astype("string").fillna("") == "tones"
    ok &= df["_asof"].notna()
    if not bool(ok.any()):
        return None
    return _ts(df.loc[ok, "_asof"].min())


def months_elapsed(start, asof) -> int:
    """경과 개월 = 달력 월(Period) 차. 2026-01-31 → 2029-01-02 는 36."""
    a, b = _ts(start), _ts(asof)
    if a is None or b is None:
        return 0
    return int((pd.Period(b, "M") - pd.Period(a, "M")).n)


def breach_dates(spy_close, dd: float = 0.05, asof=None) -> list[pd.Timestamp]:
    """`targets.episodes(close, dd, split=False)` 각 에피소드의 **돌파일**(수중 구간 안에서 종가가
    ATH 대비 −dd 이하로 처음 내려간 날) 목록.

    에피소드 표는 peak/trough 만 주므로 peak 이후 첫 −dd 돌파를 다시 찾는다. peak 과 돌파 사이에는
    정의상 신고점이 없으므로(있었다면 peak 이 옮겨간다) 이 재탐색은 `targets` 의 판정과 같은 날을 준다.
    """
    s = _close_series(spy_close)
    if asof is not None:
        s = s[s.index <= _ts(asof)]
    if len(s) < 2:
        return []
    ep = _episodes_table(s, dd, split=False)
    if len(ep) == 0:
        return []
    idx, vals = s.index, s.to_numpy(dtype=float)
    out: list[pd.Timestamp] = []
    for pk in pd.to_datetime(ep["peak_date"]):
        pk = _ts(pk)
        if pk is None or pk not in idx:
            continue
        p0 = int(idx.get_loc(pk))
        rel = vals[p0 + 1:] / vals[p0] - 1.0
        hit = np.nonzero(rel <= -dd)[0]
        if len(hit) == 0:
            continue
        out.append(idx[p0 + 1 + int(hit[0])])
    return out


def episodes_realized(spy_close, start, asof, dd: float = 0.05,
                      confirm: int = 20) -> tuple[int, list[str]]:
    """라이브 시작 이후 **실현된** ≥dd 낙폭 에피소드 수와 돌파일 목록 (§8.3).

    규약: `targets.episodes(close, dd, split=False)` · 돌파일 기준 · 시작 시점에 이미 진행 중이던 구간 제외
    (= 돌파일 < start 인 구간은 세지 않는다) · 저점은 `confirm`(20) 세션 뒤 확정
    (= 돌파일로부터 confirm 세션이 지나지 않았으면 아직 세지 않는다).

    asof 이후의 종가는 쓰지 않는다(점 원칙): 미래 자료를 붙여도 이미 센 에피소드는 바뀌지 않는다.
    """
    s = _close_series(spy_close)
    a = _ts(asof)
    s = s[s.index <= a] if a is not None else s
    if len(s) < 2:
        return 0, []
    st = _ts(start)
    bd = breach_dates(s, dd=dd)
    idx = s.index
    n = len(idx)
    out: list[str] = []
    for b in bd:
        if st is not None and b < st:
            continue                                     # 시작 시점에 이미 진행 중/시작 전 돌파 → 제외
        pos = int(idx.get_loc(b))
        if (n - 1 - pos) < int(confirm):
            continue                                     # 저점 미확정(20세션 전)
        out.append(str(pd.Timestamp(b).date()))
    return len(out), out


# ==================================================================
# 2. 채점 (§8.1)
# ==================================================================
def bss_block_ci(loss_p, loss_ref, block: int = int(KILL_P3["block"]), n_boot: int = int(KILL_P3["n_boot"]),
                 seed: int = 0, ci: float = float(KILL_P3["ci"])) -> tuple[float, float]:
    """짝지은(paired) 순환 블록 부트스트랩으로 BSS = 1 − mean(loss_p)/mean(loss_ref) 의 백분위 구간.

    `ledger._bss_block_ci`(Phase 2)의 공개 승격판이다 — **같은 블록 추출·같은 seed 라 같은 숫자**가 나온다
    (`tests/test_track.py::test_bss_block_ci_matches_ledger` 가 비트 동일성을 검사한다). ledger.py 는
    이 단계에서 소유하지 않으므로 그 파일을 고치는 대신 같은 알고리즘을 여기에 둔다.

    블록 시작점은 0..n-1 전체에서 뽑고 끝에서 감아 붙인다(순환) — 모든 관측이 정확히 block 번 뽑혀
    부트스트랩 평균의 기대값이 표본 평균과 같다. 재표본에서 mean(loss_ref) == 0 인 경우는 제외.
    """
    lp = np.asarray(loss_p, dtype=float)
    lr = np.asarray(loss_ref, dtype=float)
    n = len(lp)
    if n == 0 or len(lr) != n:
        return (float("nan"), float("nan"))
    b = min(int(block), n)
    k = int(math.ceil(n / b))
    rng = np.random.default_rng(seed)
    offs = np.arange(b)
    bss = np.empty(int(n_boot))
    chunk = max(1, min(int(n_boot), int(2_000_000 // max(k * b, 1))))
    done = 0
    while done < n_boot:
        m = min(chunk, int(n_boot) - done)
        starts = rng.integers(0, n, size=(m, k))
        sel = ((starts[:, :, None] + offs[None, None, :]) % n).reshape(m, k * b)[:, :n]
        mp = lp[sel].mean(axis=1)
        mr = lr[sel].mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            bss[done: done + m] = np.where(mr > 0, 1.0 - mp / mr, np.nan)
        done += m
    bss = bss[np.isfinite(bss)]
    if len(bss) == 0:
        return (float("nan"), float("nan"))
    alpha = (1.0 - float(ci)) / 2.0
    lo, hi = np.quantile(bss, [alpha, 1.0 - alpha])
    return (float(lo), float(hi))


def _ci_label(lo, hi) -> str | None:
    """CI 라벨: 하한 > 0 → 'validated', 상한 < 0 → 'rejected', 그 밖 → 'undecided'."""
    if lo is None or hi is None or not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    if lo > 0:
        return "validated"
    if hi < 0:
        return "rejected"
    return "undecided"


def score(rows: pd.DataFrame, p_col: str, ref_cols: dict | None = None, cfg: dict = KILL_P3, *,
          target: str | None = None, unlock_path=HOLDOUT_UNLOCK_PATH, notes: list | None = None) -> dict:
    """라이브 행의 Brier·skill·CI (§8.1).

    * 채점 행 = `p_col` 과 목표(`cfg["target"]`, 기본 y_dd5_20)가 **둘 다** 있는 행. 20세션 backfill 이
      끝난 행만 자동으로 포함된다.
    * `d_t = (clim − y)² − (p − y)²` 의 부호가 곧 모델이 기후학보다 나은지다; `bss = mean(d)/mean((clim−y)²)`.
    * `ci_bss_clim` 은 n ≥ cfg["block"](40)일 때만 — 블록 하나도 못 채우면 순환 재표본이 원본과 같아
      구간이 퇴화한다.
    * 홀드아웃 게이트가 기본으로 걸린다(`unlock_path=None` 으로 해제).
    """
    refs = dict(SCORE_REFS) if ref_cols is None else dict(ref_cols)
    tgt = target or str(cfg.get("target", "y_dd5_20"))
    out: dict = {"n": 0, "n_eff": 0.0, "brier": None, "base_rate": None, "mean_p": None,
                 "ci_bss_clim": None, "ci_label": None, "holdout_locked": False,
                 "brier_ref": {}, "n_ref": {}, "p_col": p_col, "target": tgt,
                 "ci_method": f"짝지은 순환 블록 부트스트랩(block={int(cfg['block'])}, n_boot={int(cfg['n_boot'])}, "
                              f"seed={int(cfg.get('seed', 0))}) — n ≥ {int(cfg['block'])} 일 때만"}
    for name in refs:
        out[f"bss_{name}"] = None
    for name in ("clim", "m1", "vix", "vix_bgk"):
        out.setdefault(f"bss_{name}", None)
    _notes = notes if notes is not None else []

    if rows is None or len(rows) == 0 or p_col not in rows.columns:
        _notes.append(f"score: {p_col} 열이 없거나 행이 없습니다 → 채점 없음")
        return out
    if tgt not in rows.columns:
        _notes.append(f"score: 목표 열 {tgt} 이 없습니다 → 채점 없음")
        return out
    df, locked = holdout_gate(rows, unlock_path, _notes)
    out["holdout_locked"] = bool(locked)
    df = _numeric(df, [p_col, tgt] + list(refs.values()))
    scored = df[df[p_col].notna() & df[tgt].notna()]
    out["n"] = int(len(scored))
    out["n_eff"] = round(len(scored) / N_EFF_DIV, 4)
    if len(scored) == 0:
        _notes.append("score: 채점된 행이 없습니다 (y 확정 전이거나 홀드아웃 잠금)")
        return out

    p = scored[p_col].to_numpy(dtype=float)
    y = scored[tgt].to_numpy(dtype=float)
    loss_p = (p - y) ** 2
    out["brier"] = float(loss_p.mean())
    out["base_rate"] = float(y.mean())
    out["mean_p"] = float(p.mean())
    for name, col in refs.items():
        if col not in scored.columns:
            _notes.append(f"score: {col} 이 없어 {name} 대비 skill 을 계산할 수 없습니다")
            continue
        ok = scored[col].notna().to_numpy()
        out["n_ref"][name] = int(ok.sum())
        if not ok.any():
            _notes.append(f"score: {col} 이 비어 {name} 대비 skill 을 계산할 수 없습니다")
            continue
        ref = scored[col].to_numpy(dtype=float)[ok]
        loss_ref = (ref - y[ok]) ** 2
        b_ref = float(loss_ref.mean())
        out["brier_ref"][name] = b_ref
        if b_ref <= 0:
            _notes.append(f"score: {name} 기준 Brier 가 0 이라 skill 을 정의할 수 없습니다")
            continue
        out[f"bss_{name}"] = float(1.0 - loss_p[ok].mean() / b_ref)
        if name == "clim":
            if int(ok.sum()) >= int(cfg["block"]):
                lo, hi = bss_block_ci(loss_p[ok], loss_ref, block=int(cfg["block"]),
                                      n_boot=int(cfg["n_boot"]), seed=int(cfg.get("seed", 0)),
                                      ci=float(cfg.get("ci", 0.95)))
                if math.isfinite(lo) and math.isfinite(hi):
                    out["ci_bss_clim"] = [float(lo), float(hi)]
                    out["ci_label"] = _ci_label(lo, hi)
            else:
                _notes.append(f"score: 채점 행 {int(ok.sum())} < 블록 {int(cfg['block'])} → ci_bss_clim 생략")
    for k in ("brier", "base_rate", "mean_p"):
        out[k] = _jf(out[k])
    return out


def hist_pct(bss, L: int, reference: dict) -> float | None:
    """라이브 BSS 가 같은 길이 L 의 과거 이동 창 분포에서 차지하는 백분위(0~1).

    `reference["rolling_bss"][str(L)]` = window_distribution() 이 만든 값 목록(step 21).
    분포가 없거나 bss 가 유한하지 않으면 None(조용히 0 을 만들지 않는다).
    """
    if bss is None or not math.isfinite(float(bss)):
        return None
    rb = (reference or {}).get("rolling_bss") or {}
    dist = rb.get(str(L), rb.get(int(L) if str(L).isdigit() else L))
    if isinstance(dist, dict):
        dist = dist.get("values")
    if dist is None:
        return None
    arr = np.asarray(list(dist), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None
    return float(np.mean(arr <= float(bss)))


def window_distribution(oos: pd.DataFrame, p_col: str, *, y_col: str = "y", ref_col: str = "clim",
                        windows=tuple(P3["reference_windows"]), step: int = 21,
                        unlock_path=HOLDOUT_UNLOCK_PATH) -> dict:
    """같은 길이 L 의 과거 이동 창 BSS 분포(step 21) — `hist_pct` 가 쓰는 참조.

    계약상 이 함수는 `evaluate.py` 에 놓이지만(§1) 이 단계에서 그 파일을 소유하지 않으므로 여기에 둔다
    (배선 단계에서 재수출하면 된다). 반환: {"rolling_bss": {str(L): [...]}, "quantiles": {str(L): {...}}, "step"}.
    """
    if oos is None or len(oos) == 0:
        return {"rolling_bss": {}, "quantiles": {}, "step": int(step)}
    df, _ = holdout_gate(oos, unlock_path)
    df = _numeric(df, [p_col, y_col, ref_col]).dropna(subset=[p_col, y_col, ref_col])
    p = df[p_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    r = df[ref_col].to_numpy(dtype=float)
    lp, lr = (p - y) ** 2, (r - y) ** 2
    roll: dict[str, list[float]] = {}
    quant: dict[str, dict] = {}
    n = len(df)
    for L in windows:
        L = int(L)
        vals: list[float] = []
        for s in range(0, max(n - L + 1, 0), int(step)):
            a, b = lp[s:s + L], lr[s:s + L]
            mr = float(b.mean())
            if mr > 0:
                vals.append(float(1.0 - float(a.mean()) / mr))
        roll[str(L)] = vals
        if vals:
            arr = np.asarray(vals, dtype=float)
            quant[str(L)] = {"n": len(vals), "p5": float(np.quantile(arr, 0.05)),
                             "p10": float(np.quantile(arr, 0.10)), "p50": float(np.quantile(arr, 0.50)),
                             "p90": float(np.quantile(arr, 0.90)), "p95": float(np.quantile(arr, 0.95)),
                             "min": float(arr.min()), "max": float(arr.max()),
                             "share_lt_0": float(np.mean(arr < 0))}
    return {"rolling_bss": roll, "quantiles": quant, "step": int(step)}


# ==================================================================
# 3. 킬룰 (§8.3)
# ==================================================================
def _prev_kill(prev: dict | None) -> dict:
    d = dict(_KILL_PREV_DEFAULT)
    if isinstance(prev, dict):
        for k in d:
            if k in prev and prev[k] is not None:
                d[k] = prev[k]
        d["killed"] = bool(d["killed"])
        d["stage1_done"] = bool(d["stage1_done"])
        d["stage2_done"] = bool(d["stage2_done"])
        d["last_eval_month"] = None if d["last_eval_month"] is None else int(d["last_eval_month"])
    return d


def _read_manual(manual_path, notes: list) -> dict | None:
    """수동 킬 파일(`results/kill_manual.json`). ledger_entry 는 필수 — 없으면 경고하고 그래도 킬한다
    (킬 방향은 안전한 쪽; 조용히 무시하지 않는다)."""
    p = Path(manual_path) if manual_path is not None else None
    if p is None or not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        warnings.warn(f"kill_manual.json 을 읽을 수 없습니다({type(e).__name__}: {e}) → 수동 킬로 간주")
        notes.append(f"kill_manual.json 읽기 실패({type(e).__name__}) → 안전한 쪽(킬)으로 처리")
        return {"ledger_entry": None, "_parse_error": str(e)}
    if not isinstance(d, dict):
        d = {"value": d, "ledger_entry": None}
    if not d.get("ledger_entry"):
        warnings.warn("kill_manual.json 에 ledger_entry 가 없습니다 — VALIDATION §8 장부 항목이 필요합니다")
        notes.append("kill_manual.json 에 ledger_entry 없음 → 장부 항목을 기재하세요(킬은 적용됨)")
    return d


def kill_status(ledger: pd.DataFrame, spy_close, asof, prev: dict | None = None, cfg: dict = KILL_P3,
                members: dict | None = None, *, unlock_path=HOLDOUT_UNLOCK_PATH,
                manual_path=KILL_MANUAL_PATH, p_col: str = "prob_dd5_20",
                mode_col: str = "p2_deploy_mode", ref_cols: dict | None = None) -> dict:
    """2단계 킬룰의 오늘 상태 (§8.3).

    핵심 논리(문서의 코드 계약 그대로):
    ```
    stage1_due = months >= 36
    stage2_due = stage1_due and (n_ep >= 8 or months >= 60)
    due_now    = (stage1_due and not prev.stage1_done) or (stage2_due and not prev.stage2_done)
                 or (stage2_due and months - prev.last_eval_month >= 12)
    killed     = prev.killed or (due_now and not (bss > 0))          # 점추정 트리거; sticky
    state      = info_only if killed else validated if (stage2_due and lo > 0)
                 else provisional if stage1_due else not_due
    ```
    12·24개월 중간 점검은 숫자만 보여 주고 **킬할 수 없다**(§7 의 3년 하한). 중대 실패
    (BSS ≤ −0.05 ∧ CI 상한 ≤ 0)는 배지로만 표시된다.

    자료가 없어 BSS 가 정의되지 않는데 판정일이 온 경우는 **킬하지 않고** 판정을 미룬다
    (`evaluated_now=False`, 사유를 notes 에 기록) — 결측을 실패로 읽으면 조용한 오판이 된다.
    """
    notes: list[str] = []
    prev_d = _prev_kill(prev)
    asof_ts = _ts(asof)
    out: dict = {
        "state": "not_started", "live_start": None, "clock_end": None, "n": 0, "n_eff": 0.0, "months": 0,
        "episodes5": 0, "breach_dates": [], "bss_clim": None, "ci": None, "ci_label": None,
        "stage1_due": False, "stage2_due": False, "evaluated_now": False, "killed": False,
        "stage1_done": prev_d["stage1_done"], "stage2_done": prev_d["stage2_done"],
        "last_eval_month": prev_d["last_eval_month"],
        "countdown": {"episodes": f"0/{int(cfg['min_episodes'])}", "months": f"0/{int(cfg['min_months'])}",
                      "cap": f"0/{int(cfg['max_months'])}"},
        "gross_failure_badge": False, "manual": None, "holdout_locked": False, "members": {},
        "score": None, "asof": None if asof_ts is None else str(asof_ts.date()),
        "rule": ("1차 36개월(점추정 ≤ 0 → info_only, sticky) · 2차 = 8회 또는 60개월 중 먼저 · 이후 12개월마다 "
                 "재평가 · 라벨은 부트스트랩 CI 하한(VALIDATION.md §7 실행 해석, 장부 #7)"),
        "rearm": {"automatic": False,
                  "procedure": ("복귀는 코드에 없다: 새 번호 장부 항목 + 킬 이후 ≥ "
                                f"{int(cfg['rearm']['min_months'])}개월 + **전체 라이브 창**의 CI 하한 > "
                                f"{cfg['rearm']['ci_lo_gt']} (부분 창 금지)")},
        "notes": notes,
    }

    manual = _read_manual(manual_path, notes)

    if ledger is None or len(ledger) == 0:
        notes.append("장부가 비어 있습니다 → 라이브 시작 전")
        if manual is not None:
            out["manual"], out["killed"], out["state"] = manual, True, "manual_kill"
        return out

    df = ledger.copy()
    df["_asof"] = pd.to_datetime(_frame_dates(df), errors="coerce")
    df = df[df["_asof"].notna()].sort_values("_asof", kind="stable")
    if "variant" in df.columns and (df["variant"].astype("string") == "completed").any():
        df = df[df["variant"].astype("string") == "completed"]
    df = df.drop_duplicates("_asof", keep="first")
    if asof_ts is not None:
        df = df[df["_asof"] <= asof_ts]

    ls = live_start(df, col=p_col, mode_col=mode_col)
    if ls is None:
        notes.append(f"라이브 시작 전: {p_col} 이 있고 {mode_col} == 'tones' 인 행이 없습니다 "
                     "(Phase 2 info_only 기간·재현 행은 라이브가 아닙니다 — §8.1)")
        if manual is not None:
            out["manual"], out["killed"], out["state"] = manual, True, "manual_kill"
        return out

    live = df[df["_asof"] >= ls]
    scorable, locked = holdout_gate(live, unlock_path, notes)
    out["holdout_locked"] = bool(locked)
    out["live_start"] = str(ls.date())
    if len(scorable) == 0:
        notes.append("킬룰 시계 보류: 채점 가능한 라이브 행이 없습니다(홀드아웃 미해제)")
        if manual is not None:
            out["manual"], out["killed"], out["state"] = manual, True, "manual_kill"
        return out

    ls_eff = _ts(scorable["_asof"].min())
    clock_end = _ts(scorable["_asof"].max()) if locked else (asof_ts or _ts(scorable["_asof"].max()))
    out["clock_end"] = str(clock_end.date())
    months = months_elapsed(ls_eff, clock_end)
    out["months"] = int(months)

    sc = score(scorable.drop(columns=["_asof"]), p_col, ref_cols, cfg, unlock_path=None, notes=notes)
    out["score"] = sc
    out["n"], out["n_eff"] = sc["n"], sc["n_eff"]
    bss = sc.get("bss_clim")
    out["bss_clim"] = bss
    out["ci"] = sc.get("ci_bss_clim")
    out["ci_label"] = sc.get("ci_label")

    n_ep, bds = 0, []
    if spy_close is not None:
        try:
            n_ep, bds = episodes_realized(spy_close, ls_eff, clock_end, dd=float(cfg["dd"]),
                                          confirm=int(cfg["confirm_sessions"]))
        except Exception as e:                                     # 조용한 실패 금지
            warnings.warn(f"episodes_realized 실패({type(e).__name__}: {e}) → 에피소드 0 으로 두고 기록")
            notes.append(f"episodes_realized 실패({type(e).__name__}: {e}) → 에피소드 카운트 0")
    else:
        notes.append("spy_close 가 없어 실현 에피소드를 셀 수 없습니다 → 0 (2차 판정은 60개월 상한으로만)")
    out["episodes5"], out["breach_dates"] = int(n_ep), list(bds)

    min_m, min_ep = int(cfg["min_months"]), int(cfg["min_episodes"])
    max_m, reeval = int(cfg["max_months"]), int(cfg["reeval_months"])
    stage1_due = months >= min_m
    stage2_due = bool(stage1_due and (n_ep >= min_ep or months >= max_m))
    lem = prev_d["last_eval_month"]
    due_now = bool((stage1_due and not prev_d["stage1_done"])
                   or (stage2_due and not prev_d["stage2_done"])
                   or (stage2_due and lem is not None and (months - lem) >= reeval))
    out["stage1_due"], out["stage2_due"] = bool(stage1_due), bool(stage2_due)

    bss_defined = bss is not None and math.isfinite(float(bss))
    evaluated_now = bool(due_now and bss_defined)
    if due_now and not bss_defined:
        notes.append("판정일이지만 BSS_clim 이 정의되지 않았습니다(채점 행 없음) → 판정 보류(킬하지 않음). "
                     "다음 세션에 다시 시도합니다")
    out["evaluated_now"] = evaluated_now

    killed = bool(prev_d["killed"] or (evaluated_now and not (float(bss) > 0)))
    out["killed"] = killed
    out["stage1_done"] = bool(prev_d["stage1_done"] or (stage1_due and evaluated_now))
    out["stage2_done"] = bool(prev_d["stage2_done"] or (stage2_due and evaluated_now))
    out["last_eval_month"] = int(months) if evaluated_now else lem

    ci = out["ci"]
    lo = ci[0] if isinstance(ci, (list, tuple)) and len(ci) == 2 else None
    if manual is not None:
        out["manual"], out["killed"], out["state"] = manual, True, "manual_kill"
    elif killed:
        out["state"] = "info_only"
    elif stage2_due and lo is not None and lo > 0:
        out["state"] = "validated"
    elif stage1_due:
        out["state"] = "provisional"
    else:
        out["state"] = "not_due"

    out["countdown"] = {"episodes": f"{int(n_ep)}/{min_ep}", "months": f"{int(months)}/{min_m}",
                        "cap": f"{int(months)}/{max_m}"}
    hi = ci[1] if isinstance(ci, (list, tuple)) and len(ci) == 2 else None
    out["gross_failure_badge"] = bool(bss_defined and float(bss) <= -0.05 and hi is not None and hi <= 0)
    if out["gross_failure_badge"] and not stage1_due:
        notes.append("중대 실패 배지(BSS ≤ −0.05 ∧ CI 상한 ≤ 0) — 중간 점검이라 킬할 수 없습니다(§7 3년 하한)")

    # 등록 멤버(그림자 포함)도 같은 날짜·같은 통계로 판정한다 — 정보이며 생산 확률엔 영향이 없다.
    mem = MEMBER_COLUMNS if members is None else dict(members)
    for name, col in mem.items():
        if col == p_col:
            continue
        m_ls = live_start(df, col=col, mode_col=None) if col in df.columns else None
        if m_ls is None:
            out["members"][name] = {"live_start": None, "n": 0, "n_eff": 0.0, "brier": None,
                                    "bss_clim": None, "verdict": None,
                                    "note": f"{col} 이 아직 장부에 없습니다"}
            continue
        m_rows = df[df["_asof"] >= m_ls]
        m_rows, _ = holdout_gate(m_rows, unlock_path)
        m_sc = score(m_rows.drop(columns=["_asof"]), col, ref_cols, cfg, unlock_path=None)
        m_bss = m_sc.get("bss_clim")
        m_ok = m_bss is not None and math.isfinite(float(m_bss))
        verdict = None
        if stage1_due and m_ok:
            verdict = "killed" if not (float(m_bss) > 0) else (
                "validated" if (stage2_due and m_sc.get("ci_bss_clim") and m_sc["ci_bss_clim"][0] > 0)
                else "provisional")
        out["members"][name] = {"live_start": str(m_ls.date()), "verdict": verdict, **m_sc}
    return out


def kill_apply(status: dict, model_p3_path=MODEL_P3_PATH, record_path=KILL_RECORD_PATH, *,
               extra: dict | None = None) -> bool:
    """킬 판정을 산출물에 반영한다 — **멱등**. (§8.3 '모드 스위치 역학')

    * 언제나: `model_p3.json["kill"]` 에 오늘 상태(stage 플래그·last_eval_month)를 기록해 내일의 `prev` 가 된다.
    * 킬이면(state ∈ {info_only, manual_kill}): `model_p3.json.deploy_mode = "info_only"` 로 바꾸고
      `kill_record.json` 을 **한 번만** 쓴다(이미 있으면 덮지 않는다 — 기록은 다시 쓰지 않는다).
    * `model_p2.json` 은 절대 건드리지 않는다. 킬 해제도 하지 않는다(sticky).

    반환: 이번 호출이 **킬을 새로 적용**했으면 True(파일 생성 또는 deploy_mode 전환), 아니면 False.
    `extra` 로 git_sha·registry_sha·spec_sha256 등 배선 단계의 메타를 넣을 수 있다.
    """
    mp = Path(model_p3_path)
    rp = Path(record_path)
    killed = bool(status.get("killed")) or str(status.get("state")) in ("info_only", "manual_kill")

    model: dict = {}
    if mp.exists():
        try:
            model = json.loads(mp.read_text(encoding="utf-8"))
            if not isinstance(model, dict):
                raise ValueError("model_p3.json 의 최상위가 객체가 아닙니다")
        except Exception as e:
            warnings.warn(f"model_p3.json 을 읽을 수 없습니다({type(e).__name__}: {e}) → 새로 만듭니다")
            model = {}
    else:
        warnings.warn(f"{mp} 가 없습니다 → 킬 상태만 담은 최소 파일을 만듭니다 (run_phase3.py 가 아직 돌지 않았습니다)")
    model.setdefault("schema_version", SCHEMA_VERSION)
    model.setdefault("deploy_mode", "info_only")

    changed = False
    if killed and str(model.get("deploy_mode")) != "info_only":
        model["deploy_mode"] = "info_only"
        hist = model.setdefault("mode_history", [])
        hist.append({"asof": status.get("asof"), "from": "tones", "to": "info_only",
                     "reason": str(status.get("state")), "recorded_at_utc": _utcnow_iso()})
        changed = True
    model["kill"] = {k: status.get(k) for k in ("state", "live_start", "n", "n_eff", "months", "episodes5",
                                                "bss_clim", "ci", "ci_label", "killed", "stage1_due",
                                                "stage2_due", "stage1_done", "stage2_done", "last_eval_month",
                                                "countdown", "gross_failure_badge", "asof")}
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(model, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    if killed and not rp.exists():
        rec = {"asof": status.get("asof"), "state": status.get("state"), "live_start": status.get("live_start"),
               "n": status.get("n"), "n_eff": status.get("n_eff"), "bss": status.get("bss_clim"),
               "ci": status.get("ci"), "ci_label": status.get("ci_label"), "n_ep": status.get("episodes5"),
               "breach_dates": status.get("breach_dates"), "months": status.get("months"),
               "ledger_entry": "7x", "git_sha": None, "registry_sha": None, "spec_sha256": None,
               "recorded_at_utc": _utcnow_iso(),
               "rule": status.get("rule"), "rearm": status.get("rearm")}
        if extra:
            rec.update({k: v for k, v in extra.items()})
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        changed = True
    return bool(changed)


def kill_replay(oos: pd.DataFrame, spy_close, p_col: str, step: int = 21, cfg: dict = KILL_P3, *,
                y_col: str = "y", ref_col: str = "clim", ci_windows: int = 75, ci_n_boot: int = 1000,
                null_noise_sd: float | None = None, n_shuffle: int = 20, seed: int = 0,
                unlock_path=HOLDOUT_UNLOCK_PATH) -> dict:
    """과거 OOS 기록에 live_start 를 `step` 세션마다 놓고 **1차 규칙(36개월)** 을 재현한다 (§8.3).

    카드·페이지의 문장(`이 규칙은 과거 36개월 창에서 진짜 +0.09 모델을 X% 오기각하고 '검증' 은 Y% 에서만 준다;
    정보 없는 같은 분포의 모델은 Z%, 기후학+잡음은 100% 기각한다`)은 **이 함수의 산출로만** 만든다.
    재생성되지 않으면 페이지에 쓰지 않는다.

    * `false_kill_share` = 36개월 창의 BSS 점추정 ≤ 0 인 비중 (= 진짜 skill 있는 모델을 오기각할 확률)
    * `validated_share` = 창의 부트스트랩 CI 하한 > 0 인 비중 (계산량 때문에 균등 간격 `ci_windows` 개 부표본,
      `ci_n_boot` 회; seed 0 이라 결정적)
    * `null_noise_pass` = 기후학 + 독립 가우시안 잡음 모델이 통과하는 비중 → 0 이어야 한다.
      잡음 크기 `null_noise_sd` 의 기본값은 **후보 모델의 실제 분산** std(p − ref) 다(임의 상수 대신
      "같은 폭으로 흔들리지만 정보는 0" 인 널). 블록 셔플 널과 기대 손해(Var(p))는 같지만 잡음이 iid 라
      창별 표본변동이 훨씬 작아 사실상 100% 기각된다 — 두 널의 차이가 곧 자기상관의 대가다.
    * `null_block_shuffle_pass` = p 를 블록(cfg["block"]) 단위로 섞어 시점 정보만 없앤 모델이 통과하는 비중
      (`n_shuffle` 회 × 창) → 1 − 이 값이 규칙의 검정력
    * `months_to_8` = 각 시작점에서 실현 ≥5% 에피소드 8회에 도달하는 데 걸린 개월 수의 분위
    """
    rng = np.random.default_rng(int(seed))
    out: dict = {"p_col": p_col, "step": int(step), "window_months": int(cfg["min_months"]),
                 "n_windows": 0, "false_kill_share": None, "validated_share": None,
                 "bss": {}, "null_noise_pass": None, "null_block_shuffle_pass": None,
                 "months_to_8": {}, "ci_windows": 0, "ci_n_boot": int(ci_n_boot), "seed": int(seed),
                 "notes": []}
    notes: list[str] = out["notes"]
    if oos is None or len(oos) == 0:
        notes.append("kill_replay: OOS 기록이 비어 있습니다")
        return out
    df, _ = holdout_gate(oos, unlock_path, notes)
    dates = pd.to_datetime(_frame_dates(df), errors="coerce")
    df = df.assign(_d=dates.to_numpy())
    df = _numeric(df, [p_col, y_col, ref_col]).dropna(subset=["_d", p_col, y_col, ref_col])
    df = df.sort_values("_d", kind="stable")
    if len(df) == 0:
        notes.append("kill_replay: 채점 가능한 OOS 행이 없습니다")
        return out

    d = pd.DatetimeIndex(df["_d"])
    p = df[p_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    r = df[ref_col].to_numpy(dtype=float)
    lp, lr = (p - y) ** 2, (r - y) ** 2
    n = len(df)
    months = int(cfg["min_months"])
    last = d[-1]

    wins: list[tuple[int, int]] = []
    for s in range(0, n, int(step)):
        end = d[s] + pd.DateOffset(months=months)
        if end > last:
            break                                        # 36개월 창이 완결되지 않음
        e = int(np.searchsorted(d.to_numpy(), np.datetime64(end), side="left"))
        if e - s < int(cfg["block"]):
            continue
        wins.append((s, e))
    out["n_windows"] = len(wins)
    if not wins:
        notes.append("kill_replay: 36개월 완결 창이 없습니다")
        return out

    def _bss(a: np.ndarray, b: np.ndarray) -> float:
        mr = float(b.mean())
        return float("nan") if mr <= 0 else float(1.0 - float(a.mean()) / mr)

    bss_vals = np.array([_bss(lp[s:e], lr[s:e]) for s, e in wins], dtype=float)
    ok = np.isfinite(bss_vals)
    out["false_kill_share"] = float(np.mean(bss_vals[ok] <= 0)) if ok.any() else None
    out["bss"] = {"p10": float(np.quantile(bss_vals[ok], 0.10)), "p50": float(np.quantile(bss_vals[ok], 0.50)),
                  "p90": float(np.quantile(bss_vals[ok], 0.90)), "mean": float(bss_vals[ok].mean())} if ok.any() else {}

    # CI 하한 > 0 ('검증') 비중 — 균등 간격 부표본(결정적)
    k = min(int(ci_windows), len(wins))
    pick = np.unique(np.linspace(0, len(wins) - 1, k).round().astype(int)) if k > 0 else np.array([], dtype=int)
    val = []
    for i in pick:
        s, e = wins[int(i)]
        lo, _hi = bss_block_ci(lp[s:e], lr[s:e], block=int(cfg["block"]), n_boot=int(ci_n_boot),
                               seed=int(cfg.get("seed", 0)), ci=float(cfg.get("ci", 0.95)))
        if math.isfinite(lo):
            val.append(lo > 0)
    out["ci_windows"] = len(val)
    out["validated_share"] = float(np.mean(val)) if val else None

    # 널 1: 기후학 + 잡음 (정보 0, 게다가 잡음만큼 손해) → 통과 0% 이어야 한다
    sd = float(np.std(p - r, ddof=1)) if null_noise_sd is None else float(null_noise_sd)
    out["null_noise_sd"] = sd
    p_noise = np.clip(r + rng.normal(0.0, sd, size=n), 1e-6, 1 - 1e-6)
    ln = (p_noise - y) ** 2
    b_noise = np.array([_bss(ln[s:e], lr[s:e]) for s, e in wins], dtype=float)
    okn = np.isfinite(b_noise)
    out["null_noise_pass"] = float(np.mean(b_noise[okn] > 0)) if okn.any() else None

    # 널 2: 블록 셔플 (주변분포·자기상관 구조는 남기고 시점 정보만 없앤다) → 1 − pass = 검정력
    block = int(cfg["block"])
    n_blocks = int(math.ceil(n / block))
    shuf_pass: list[float] = []
    for _ in range(int(n_shuffle)):
        order = rng.permutation(n_blocks)
        idx = np.concatenate([np.arange(j * block, min((j + 1) * block, n)) for j in order])
        ls_ = (p[idx] - y) ** 2
        bb = np.array([_bss(ls_[s:e], lr[s:e]) for s, e in wins], dtype=float)
        okb = np.isfinite(bb)
        if okb.any():
            shuf_pass.append(float(np.mean(bb[okb] > 0)))
    out["null_block_shuffle_pass"] = float(np.mean(shuf_pass)) if shuf_pass else None
    out["power_vs_block_shuffle"] = (None if out["null_block_shuffle_pass"] is None
                                     else float(1.0 - out["null_block_shuffle_pass"]))

    # 문자 그대로의 '8회' 가 얼마나 먼지 — 각 시작점에서 8번째 돌파일까지의 개월 수
    if spy_close is not None:
        try:
            bd = pd.DatetimeIndex(breach_dates(spy_close, dd=float(cfg["dd"]), asof=last))
            need = int(cfg["min_episodes"])
            got, censored = [], 0
            for s, _e in wins:
                after = bd[bd >= d[s]]
                if len(after) >= need:
                    got.append(months_elapsed(d[s], after[need - 1]))
                else:
                    censored += 1
            if got:
                arr = np.asarray(got, dtype=float)
                out["months_to_8"] = {"n": len(got), "censored": int(censored),
                                      "p10": float(np.quantile(arr, 0.10)), "p50": float(np.quantile(arr, 0.50)),
                                      "p90": float(np.quantile(arr, 0.90))}
            else:
                out["months_to_8"] = {"n": 0, "censored": int(censored)}
        except Exception as e:                                    # 조용한 실패 금지
            warnings.warn(f"kill_replay: months_to_8 계산 실패({type(e).__name__}: {e})")
            notes.append(f"months_to_8 계산 실패({type(e).__name__}: {e})")
    else:
        notes.append("spy_close 가 없어 months_to_8 을 계산하지 않았습니다")

    fk = out["false_kill_share"]
    vs = out["validated_share"]
    pw = out["power_vs_block_shuffle"]
    out["sentence_ko"] = (
        f"이 규칙은 과거 {months}개월 창에서 이 모델을 {0 if fk is None else round(100 * fk):.0f}% 오기각하고 "
        f"'검증' 은 {0 if vs is None else round(100 * vs):.0f}% 에서만 준다; 정보 없는 같은 분포의 모델은 "
        f"{0 if pw is None else round(100 * pw):.0f}%, 기후학+잡음은 "
        f"{100 if out['null_noise_pass'] is None else round(100 * (1 - out['null_noise_pass'])):.0f}% 기각한다.")
    return out


# ==================================================================
# 4. 드리프트 경보 D1~D11 (§8.2)
# ==================================================================
def _alarm(code: str, value, threshold, **extra) -> dict:
    d = {"code": code, "value": _jf(value) if not isinstance(value, (str, list, dict)) else value,
         "threshold": threshold, "action": ALARM_ACTIONS.get(code, "display")}
    d.update(extra)
    return d


def alarms(ledger: pd.DataFrame, reference: dict | None = None, cfg: dict = P3_DRIFT,
           feats_today: pd.Series | None = None, asof=None, *, recomputed: pd.DataFrame | None = None,
           unlock_path=HOLDOUT_UNLOCK_PATH, notes: list | None = None) -> list[dict]:
    """드리프트 경보 D1~D11 (§8.2) — **플래그·기록만** 한다. 자동 재조정은 없다.

    임계는 전부 `config.P3_DRIFT`(2003~24 참조분포의 p5/p95, post hoc)에서 온다. 이 함수는 임계를
    라이브 자료로 다시 맞추지 않는다 — 변경은 번호 붙인 장부 항목으로만.

    반환: 발화한 경보만 `[{code, value, threshold, action, ...}]`. 창을 채우지 못해 평가하지 못한 코드는
    `notes` 리스트(주면)에 사유가 남는다(조용한 실패 금지).
    D7·D10 은 재계산 프레임(`recomputed`)이 있어야 평가된다 — daily.py 가 최근 20세션 재계산을 넘긴다.
    """
    ref = reference or {}
    _n = notes if notes is not None else []
    out: list[dict] = []
    if ledger is None or len(ledger) == 0:
        _n.append("alarms: 장부가 비어 있습니다")
        return out

    df = ledger.copy()
    df["_asof"] = pd.to_datetime(_frame_dates(df), errors="coerce")
    df = df[df["_asof"].notna()].sort_values("_asof", kind="stable")
    if asof is not None:
        df = df[df["_asof"] <= _ts(asof)]
    df, _locked = holdout_gate(df, unlock_path, _n)
    if len(df) == 0:
        _n.append("alarms: 채점 가능한 행이 없습니다(홀드아웃 잠금)")
        return out
    asof_s = str(_ts(df["_asof"].max()).date())

    num = ["prob_dd5_20", "p2_clim", "y_dd5_20", "p2_har_fc_20", "rv20_realized", "p3_sigma_ewma",
           "p3_sigma_target", "p3_w_exec", "p3_rule_dd", "p3_d_max", "spy_close", "p3_hmm_p_high",
           "ret20_in_vix80", "ret20_in_har80", "p2_x_vix", "p2_x_har", "p2_x_ma", "p3_x_hmm"]
    df = _numeric(df, num)

    # ---- D1 p_level: 120세션 평균 p 가 [0.04, 0.35] 밖 ----------------
    c1 = cfg["D1_p_level"]
    s = df["prob_dd5_20"].dropna() if "prob_dd5_20" in df.columns else pd.Series(dtype=float)
    if len(s) >= int(c1["window"]):
        v = float(s.iloc[-int(c1["window"]):].mean())
        if v < float(c1["lo"]) or v > float(c1["hi"]):
            out.append(_alarm("D1_p_level", v, [float(c1["lo"]), float(c1["hi"])], asof=asof_s,
                              window=int(c1["window"])))
    else:
        _n.append(f"D1: p 기록 {len(s)} < {int(c1['window'])}세션 → 미평가")

    # ---- D2 bss: 756세션 BSS < −0.05 빨강 / 252세션 < −0.10 노랑 -------
    c2 = cfg["D2_bss"]
    scored = df[df["prob_dd5_20"].notna() & df["y_dd5_20"].notna() & df["p2_clim"].notna()] \
        if {"prob_dd5_20", "y_dd5_20", "p2_clim"}.issubset(df.columns) else df.iloc[0:0]

    def _bss_tail(k: int):
        if len(scored) < k:                  # 부분 창 금지: 임계는 k세션 참조분포의 p5 다 (§8.2·§8.3)
            _n.append(f"D2({k}세션): 채점된 행 {len(scored)} < {k} → 미평가")
            return None
        sub = _tail(scored, k)
        if len(sub) == 0:
            return None
        y = sub["y_dd5_20"].to_numpy(dtype=float)
        lp = (sub["prob_dd5_20"].to_numpy(dtype=float) - y) ** 2
        lr = (sub["p2_clim"].to_numpy(dtype=float) - y) ** 2
        m = float(lr.mean())
        return None if m <= 0 else float(1.0 - float(lp.mean()) / m)

    if len(scored) == 0:
        _n.append("D2: 채점된 행이 없습니다 → 미평가")
    else:
        red_w, red_lt = int(c2["red"]["window"]), float(c2["red"]["lt"])
        yel_w, yel_lt = int(c2["yellow"]["window"]), float(c2["yellow"]["lt"])
        b_red, b_yel = _bss_tail(red_w), _bss_tail(yel_w)
        if b_red is not None and b_red < red_lt:
            out.append(_alarm("D2_bss", b_red, red_lt, level="red", window=red_w, asof=asof_s,
                              note="킬룰 전 경고 배지"))
        elif b_yel is not None and b_yel < yel_lt:
            out.append(_alarm("D2_bss", b_yel, yel_lt, level="yellow", window=yel_w, asof=asof_s))

    # ---- D3 vol_fc: HAR log-MAE / EWMA 편향 ---------------------------
    c3 = cfg["D3_vol_fc"]
    w3 = int(c3["window"])
    if {"p2_har_fc_20", "rv20_realized"}.issubset(df.columns):
        sub = df[(df["p2_har_fc_20"] > 0) & (df["rv20_realized"] > 0)]
        sub = _tail(sub, w3)
        if len(sub) >= w3:
            mae = float(np.mean(np.abs(np.log(sub["p2_har_fc_20"].to_numpy(dtype=float))
                                       - np.log(sub["rv20_realized"].to_numpy(dtype=float)))))
            if mae > float(c3["har_log_mae_gt"]):
                out.append(_alarm("D3_vol_fc", mae, float(c3["har_log_mae_gt"]), kind="har_log_mae",
                                  window=w3, asof=asof_s))
        else:
            _n.append(f"D3(har_log_mae): 유효 행 {len(sub)} < {w3} → 미평가")
    else:
        _n.append("D3(har_log_mae): p2_har_fc_20/rv20_realized 열이 없습니다 → 미평가")
    if {"rv20_realized", "p3_sigma_ewma"}.issubset(df.columns) and not any(a["code"] == "D3_vol_fc" for a in out):
        sub = df[(df["rv20_realized"] > 0) & (df["p3_sigma_ewma"] > 0)]
        sub = _tail(sub, w3)
        if len(sub) >= w3:
            bias = float(np.mean(np.log(sub["rv20_realized"].to_numpy(dtype=float)
                                        / sub["p3_sigma_ewma"].to_numpy(dtype=float))))
            lo_b, hi_b = (float(x) for x in c3["ewma_bias"])
            if bias < lo_b or bias > hi_b:
                out.append(_alarm("D3_vol_fc", bias, [lo_b, hi_b], kind="ewma_bias", window=w3, asof=asof_s))
        else:
            _n.append(f"D3(ewma_bias): 유효 행 {len(sub)} < {w3} → 미평가")

    # ---- D4 vol_target: 규칙 수익률 실현변동성 / σ_T > 1.5 -------------
    c4 = cfg["D4_vol_target"]
    w4 = int(c4["window"])
    if {"p3_w_exec", "spy_close", "p3_sigma_target"}.issubset(df.columns):
        sub = df[["_asof", "p3_w_exec", "spy_close", "p3_sigma_target"]].dropna(subset=["spy_close"])
        if len(sub) >= w4 + 1:
            ret = sub["spy_close"].astype(float).pct_change()
            w_prev = sub["p3_w_exec"].astype(float).shift(1)
            rule = (w_prev * ret).dropna()
            rule = _tail(rule.to_frame("r"), w4)["r"]
            st = df["p3_sigma_target"].dropna()
            if len(rule) >= w4 and len(st) > 0 and float(st.iloc[-1]) > 0:
                rv = float(rule.std(ddof=1) * math.sqrt(252.0))
                ratio = rv / float(st.iloc[-1])
                if ratio > float(c4["ratio_gt"]):
                    out.append(_alarm("D4_vol_target", ratio, float(c4["ratio_gt"]), window=w4, asof=asof_s,
                                      realized_vol=rv, sigma_target=float(st.iloc[-1])))
            else:
                _n.append(f"D4: 규칙 수익률 {len(rule)} < {w4} 또는 σ_T 결측 → 미평가")
        else:
            _n.append(f"D4: 행 {len(sub)} < {w4 + 1} → 미평가")
    else:
        _n.append("D4: p3_w_exec/spy_close/p3_sigma_target 열이 없습니다 → 미평가")

    # ---- D4b budget: 라이브 규칙 MaxDD < −D_max -----------------------
    if cfg.get("D4b_budget") and {"p3_rule_dd", "p3_d_max"}.issubset(df.columns):
        dd = df["p3_rule_dd"].dropna()
        dm = df["p3_d_max"].dropna()
        if len(dd) and len(dm):
            worst, d_max = float(dd.min()), float(dm.iloc[-1])
            if worst < -abs(d_max):
                out.append(_alarm("D4b_budget", worst, -abs(d_max), asof=asof_s,
                                  note="예산 초과 — 장부 항목이 필요합니다"))
        else:
            _n.append("D4b: p3_rule_dd/p3_d_max 값이 없습니다 → 미평가")
    elif cfg.get("D4b_budget"):
        _n.append("D4b: p3_rule_dd/p3_d_max 열이 없습니다 → 미평가")

    # ---- D5 stuck: 비정상 점유 > 0.75 또는 평균 w_exec < 0.30 ---------
    c5 = cfg["D5_stuck"]
    w5 = int(c5["window"])
    sub5 = _tail(df, w5)
    if len(sub5) >= w5 and "p2_state" in sub5.columns:
        st = sub5["p2_state"].astype("string")
        occ = float((st.notna() & (st != "normal")).mean())
        if occ > float(c5["warn_occ_gt"]):
            out.append(_alarm("D5_stuck", occ, float(c5["warn_occ_gt"]), kind="warn_occupancy",
                              window=w5, asof=asof_s))
    if len(sub5) >= w5 and "p3_w_exec" in sub5.columns and not any(a["code"] == "D5_stuck" for a in out):
        wv = sub5["p3_w_exec"].dropna()
        if len(wv) >= w5:
            avg = float(wv.mean())
            if avg < float(c5["avg_w_lt"]):
                out.append(_alarm("D5_stuck", avg, float(c5["avg_w_lt"]), kind="avg_w_exec",
                                  window=w5, asof=asof_s))
    if len(sub5) < w5:
        _n.append(f"D5: 행 {len(sub5)} < {w5} → 미평가")

    # ---- D6 churn: 252세션 비중 변경 > 24 -----------------------------
    c6 = cfg["D6_churn"]
    w6 = int(c6["window"])
    if "p3_w_exec" in df.columns:
        sub6 = _tail(df[df["p3_w_exec"].notna()], w6)
        if len(sub6) >= min(w6, 2):
            ch = _count_changes(sub6["p3_w_exec"].to_numpy(dtype=float))
            if ch > int(c6["gt"]):
                out.append(_alarm("D6_churn", ch, int(c6["gt"]), window=w6, asof=asof_s,
                                  note="비중 카드 정지 — 장부 항목 기재까지"))
        else:
            _n.append("D6: p3_w_exec 행이 부족합니다 → 미평가")
    else:
        _n.append("D6: p3_w_exec 열이 없습니다 → 미평가")

    # ---- D7 parity: 최근 20세션 재계산 |Δp| > 0.01 또는 w_exec 불일치 ---
    c7 = cfg["D7_parity"]
    if recomputed is not None:
        rc = replay_check(df, recomputed, cols=("prob_dd5_20", "p3_w_exec"), tol=float(c7["dp_gt"]),
                          sessions=int(c7["sessions"]))
        if rc["n_mismatch"] > 0:
            out.append(_alarm("D7_parity", rc["max_abs_diff"], float(c7["dp_gt"]),
                              sessions=int(c7["sessions"]), n_mismatch=int(rc["n_mismatch"]),
                              n_recompute_missing=int(rc["n_recompute_missing"]),
                              by_col=rc["by_col"], asof=asof_s,
                              note="daily 는 커밋 없이 exit 1 — 코드가 장부와 다른 확률·비중을 냅니다"))
    else:
        _n.append("D7: recomputed 프레임이 없어 재계산 정합을 검사하지 못했습니다")

    # ---- D8 coverage: 80% 밴드 적중률 < 0.5 (n_eff ≥ 12) --------------
    c8 = cfg["D8_coverage"]
    worst = None
    for col, band in (("ret20_in_vix80", "vix80"), ("ret20_in_har80", "har80")):
        if col not in df.columns:
            continue
        v = df[col].dropna()
        n_eff = len(v) / N_EFF_DIV
        if n_eff < float(c8["min_n_eff"]):
            _n.append(f"D8({band}): n_eff {n_eff:.1f} < {c8['min_n_eff']} → 미평가")
            continue
        hit = float(v.mean())
        if hit < float(c8["hit_lt"]) and (worst is None or hit < worst[1]):
            worst = (band, hit, n_eff)
    if worst is not None:
        out.append(_alarm("D8_coverage", worst[1], float(c8["hit_lt"]), band=worst[0],
                          n_eff=round(worst[2], 2), asof=asof_s))

    # ---- D9 hmm: 후행 756세션 P_high AUC < 0.5 ------------------------
    c9 = cfg["D9_hmm"]
    w9 = int(c9["window"])
    if {"p3_hmm_p_high", "y_dd5_20"}.issubset(df.columns):
        sub9 = _tail(df[df["p3_hmm_p_high"].notna() & df["y_dd5_20"].notna()], w9)
        if len(sub9) >= w9:
            a = _auc(sub9["y_dd5_20"].to_numpy(dtype=float), sub9["p3_hmm_p_high"].to_numpy(dtype=float))
            if math.isfinite(a) and a < float(c9["auc_lt"]):
                out.append(_alarm("D9_hmm", a, float(c9["auc_lt"]), window=w9, asof=asof_s,
                                  note="게이지 숨김 · 멤버 H 를 candidate_rejected 로"))
        else:
            _n.append(f"D9: 채점된 P_high 행 {len(sub9)} < {w9} → 미평가")
    else:
        _n.append("D9: p3_hmm_p_high/y_dd5_20 열이 없습니다 → 미평가")

    # ---- D10 replay: 주간 재현 불일치 개수 ----------------------------
    c10 = cfg["D10_replay"]
    if recomputed is not None:
        rc10 = replay_check(df, recomputed, tol=float(c10["dp_gt"]))
        if rc10["n_mismatch"] > 0:
            out.append(_alarm("D10_replay", int(rc10["n_mismatch"]), float(c10["dp_gt"]), asof=asof_s,
                              by_col=rc10["by_col"], max_abs_diff=rc10["max_abs_diff"],
                              n_recompute_missing=int(rc10["n_recompute_missing"]),
                              note="자료 개정(Yahoo) 또는 코드 변경 — 원인을 로그에 남길 것"))
    else:
        _n.append("D10: recomputed 프레임이 없어 재현 검사를 하지 못했습니다")

    # ---- D11 feature_range: 특징이 1993~2024 범위 밖 ≥5세션 연속 ------
    c11 = cfg["D11_feature_range"]
    fr = (ref or {}).get("feature_range") or {}
    if not fr:
        _n.append("D11: reference.feature_range 가 없습니다 → 미평가")
    else:
        cols = {"x_vix": "p2_x_vix", "x_har": "p2_x_har", "x_ma": "p2_x_ma", "x_hmm": "p3_x_hmm"}
        for name, col in cols.items():
            rng_ = fr.get(name)
            if rng_ is None or col not in df.columns:
                continue
            lo_r, hi_r = float(rng_[0]), float(rng_[1])
            v = df[col]
            out_of = ((v < lo_r) | (v > hi_r)).fillna(False).to_numpy()
            run = _max_run(out_of)
            if run >= int(c11["sessions"]):
                last_v = v.dropna()
                out.append(_alarm("D11_feature_range", None if len(last_v) == 0 else float(last_v.iloc[-1]),
                                  [lo_r, hi_r], feature=name, sessions=int(run), asof=asof_s))
                break                                   # 코드당 한 줄 — 어느 특징인지는 feature 필드로
    return out


def append_alarms(rows, path=ALARMS_PATH) -> int:
    """경보 이력을 `results/alarms.csv`(asof, code, value, threshold)에 덧붙인다. 같은 (asof, code)는 다시 쓰지 않는다."""
    rows = list(rows or [])
    p = Path(path)
    new = pd.DataFrame([{k: (json.dumps(r.get(k), ensure_ascii=False) if isinstance(r.get(k), (list, dict))
                             else r.get(k)) for k in ALARM_CSV_COLUMNS} for r in rows],
                       columns=list(ALARM_CSV_COLUMNS))
    if len(new) == 0:
        return 0
    old = pd.DataFrame(columns=list(ALARM_CSV_COLUMNS))
    if p.exists():
        old = pd.read_csv(p, dtype=str, keep_default_na=False)
        for c in ALARM_CSV_COLUMNS:
            if c not in old.columns:
                old[c] = ""
    both = pd.concat([old.astype(str), new.astype(str)], ignore_index=True)
    both = both.drop_duplicates(subset=["asof", "code"], keep="first")
    p.parent.mkdir(parents=True, exist_ok=True)
    both.to_csv(p, index=False, lineterminator="\n", encoding="utf-8")
    return int(len(both) - len(old))


def replay_check(ledger: pd.DataFrame, recomputed: pd.DataFrame,
                 cols=("prob_dd5_20", "p3_hmm_p_high", "p3_w_exec"), tol: float = 1e-6, *,
                 sessions: int | None = None) -> dict:
    """장부 값과 재계산 값의 정합(D7·D10). 장부는 **기록**이며 다시 쓰지 않는다 — 여기서는 세기만 한다.

    두 프레임을 asof 로 정렬해 겹치는 세션만 비교한다. `sessions` 를 주면 최근 그만큼만(D7 은 20).

    **결측도 불일치다**: 장부에 값이 있는데 재계산이 NaN(또는 그 열 자체가 없음)이면 코드가 장부를
    재현하지 못한 것이므로 `n_mismatch`·`n_recompute_missing` 에 센다(§2 조용한 실패 금지).
    반대 방향(장부 결측 · 재계산 값)은 구 스키마 행을 채운 것이라 `n_ledger_missing` 에 기록만 하고
    불일치로 세지 않는다 — daily 를 exit 1 시키지 않는다.
    """
    out = {"tol": float(tol), "cols": list(cols), "n_compared": 0, "n_mismatch": 0,
           "max_abs_diff": None, "by_col": {}, "mismatch_rows": [], "missing_in_recomputed": 0,
           "n_recompute_missing": 0, "n_ledger_missing": 0}
    if ledger is None or recomputed is None or len(ledger) == 0 or len(recomputed) == 0:
        return out
    a = ledger.copy()
    b = recomputed.copy()
    a["_asof"] = pd.to_datetime(_frame_dates(a), errors="coerce")
    b["_asof"] = pd.to_datetime(_frame_dates(b), errors="coerce")
    a = a[a["_asof"].notna()].drop_duplicates("_asof", keep="last").set_index("_asof").sort_index()
    b = b[b["_asof"].notna()].drop_duplicates("_asof", keep="last").set_index("_asof").sort_index()
    if sessions:
        a = a.iloc[-int(sessions):]
    common = a.index.intersection(b.index)
    out["missing_in_recomputed"] = int(len(a.index.difference(b.index)))
    if len(common) == 0:
        return out
    out["n_compared"] = int(len(common))
    worst = 0.0
    idx_all = np.asarray(common)
    for c in cols:
        in_a, in_b = c in a.columns, c in b.columns
        if not (in_a or in_b):                            # 양쪽 다 없는 열(아직 안 쓰는 P3 열) — 조용히 건너뛴다
            continue
        if in_a and not in_b:                             # 장부에만 있는 열 = 재현 실패(조용히 건너뛰지 않는다)
            n_bad = int(len(common))
            warnings.warn(f"replay_check: {c} 열이 재계산 프레임에 없습니다 → 겹치는 {n_bad}세션 전부 "
                          "불일치로 셉니다")
            out["by_col"][c] = {"n": 0, "n_mismatch": n_bad, "max_abs_diff": None,
                                "n_recompute_missing": n_bad, "n_ledger_missing": 0,
                                "note": "재계산 프레임에 열이 없습니다"}
            out["n_mismatch"] += n_bad
            out["n_recompute_missing"] += n_bad
            continue
        if in_b and not in_a:                             # 재계산에만 있는 열(신규 열 도입) — 기록만, exit 1 아님
            n_new = int(len(common))
            warnings.warn(f"replay_check: {c} 열이 장부에 없습니다 → 비교하지 않습니다(불일치 아님)")
            out["by_col"][c] = {"n": 0, "n_mismatch": 0, "max_abs_diff": None,
                                "n_recompute_missing": 0, "n_ledger_missing": n_new,
                                "note": "장부에 열이 없습니다"}
            out["n_ledger_missing"] += n_new
            continue
        va = pd.to_numeric(a.loc[common, c], errors="coerce").to_numpy(dtype=float)
        vb = pd.to_numeric(b.loc[common, c], errors="coerce").to_numpy(dtype=float)
        fa, fb = np.isfinite(va), np.isfinite(vb)
        ok = fa & fb
        rec_missing = fa & ~fb        # 장부엔 값 · 재계산은 결측 → 불일치(코드가 장부를 재현하지 못한다)
        led_missing = (~fa) & fb      # 장부가 결측(구 행 backfill) → 기록만, exit 1 아님
        d = np.abs(va[ok] - vb[ok]) if ok.any() else np.zeros(0)
        bad = d > float(tol)
        n_bad = int(bad.sum()) + int(rec_missing.sum())
        mad = float(d.max()) if ok.any() else None
        out["by_col"][c] = {"n": int(ok.sum()), "n_mismatch": n_bad, "max_abs_diff": mad,
                            "n_recompute_missing": int(rec_missing.sum()),
                            "n_ledger_missing": int(led_missing.sum())}
        out["n_mismatch"] += n_bad
        out["n_recompute_missing"] += int(rec_missing.sum())
        out["n_ledger_missing"] += int(led_missing.sum())
        if mad is not None:
            worst = max(worst, mad)
        for ts in idx_all[rec_missing][:10]:
            out["mismatch_rows"].append({"asof": str(pd.Timestamp(ts).date()), "col": c,
                                         "abs_diff": None, "kind": "recompute_missing"})
        if bad.any():
            idx = idx_all[ok][bad]
            for ts, dv in list(zip(idx, d[bad]))[:10]:
                out["mismatch_rows"].append({"asof": str(pd.Timestamp(ts).date()), "col": c,
                                             "abs_diff": float(dv), "kind": "value"})
    out["max_abs_diff"] = float(worst)
    return out


# ==================================================================
# 5. 지평별 표시 규칙 (§8.4)
# ==================================================================
def horizon_stage(n_eff: float, months: int) -> str:
    """표시 단계. n_eff < 6 이면 무조건 '0-3m'(§8.4: 3개월 창의 skill 은 잡음이라 숫자를 내지 않는다)."""
    n_eff = 0.0 if n_eff is None else float(n_eff)
    months = 0 if months is None else int(months)
    if n_eff < 6 or months < 6:
        return "0-3m"
    if n_eff < 12 or months < 12:
        return "6m"
    if months < 36:
        return "12m"
    return "36m"


def may_show(stage: str, key: str) -> bool:
    """이 단계에서 `key` 를 보여도 되는가. 코드가 강제하는 유일한 판정(리포트·패널이 이 함수를 통과해야 한다)."""
    if stage not in HORIZON_HIDDEN:
        raise ValueError(f"알 수 없는 지평 단계: {stage} (허용 {HORIZON_STAGES})")
    return key not in HORIZON_HIDDEN[stage]


def horizon_copy(summary_p3: dict) -> dict:
    """§8.4 의 지평별 표시 규칙을 dict 로. {stage, show, hide, sentences_ko, ...}

    입력은 `summary_p3()` 의 산출(또는 최소한 n_eff·months_elapsed·kill 을 가진 dict).
    """
    s = summary_p3 or {}
    n_eff = s.get("n_eff")
    if n_eff is None:
        n = s.get("n_scored") or 0
        n_eff = float(n) / N_EFF_DIV
    months = int(s.get("months_elapsed") or 0)
    stage = horizon_stage(n_eff, months)
    i = HORIZON_STAGES.index(stage)
    show: list[str] = []
    for st in HORIZON_STAGES[:i + 1]:
        show.extend(HORIZON_SHOWN[st])
    hide = list(HORIZON_HIDDEN[stage])
    show = [k for k in show if k not in hide]
    kill = s.get("kill") or {}
    return {"stage": stage, "n_eff": round(float(n_eff), 4), "months": months,
            "show": show, "hide": hide,
            "sentences_ko": [HORIZON_SENTENCES[st] for st in HORIZON_STAGES[:i + 1]],
            "sentence_ko": HORIZON_SENTENCES[stage],
            "footnote_ko": HORIZON_FOOTNOTE,
            "hidden_reason": _HIDDEN_REASON.get(stage),
            "may_show_bss": may_show(stage, "bss"),
            "may_show_verdict": may_show(stage, "verdict"),
            "verdict_state": kill.get("state") if may_show(stage, "verdict") else None}


def _redact(block: dict, stage: str, keys) -> dict:
    """지평 규칙이 막는 키를 None 으로 바꾸고 사유를 남긴다(조용히 지우지 않는다)."""
    hidden = [k for k in keys if not may_show(stage, k)]
    if not hidden:
        return block
    out = dict(block)
    for k in list(out):
        base = k.split("_")[0]
        if k in hidden or base in hidden or any(k.startswith(h) for h in hidden):
            out[k] = None
    out["hidden"] = hidden
    out["hidden_reason"] = _HIDDEN_REASON.get(stage)
    return out


# ==================================================================
# 6. 라이브-vs-백테스트 패널 (§8.5) · 단일 요약 (§9)
# ==================================================================
def _ref(reference: dict | None, *path, default=None):
    cur = reference or {}
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def live_panel(ledger: pd.DataFrame, spy_close, reference: dict | None, asof, *,
               unlock_path=HOLDOUT_UNLOCK_PATH, prev_kill: dict | None = None,
               recomputed: pd.DataFrame | None = None, cfg: dict = P3_DRIFT,
               kill: dict | None = None, notes: list | None = None) -> dict:
    """§8.5 라이브-vs-백테스트 패널 (i)~(xi). 같은 행에 n·n_eff·백테스트 참조를 둔다.

    §8.4 의 지평 규칙을 **여기서 강제한다**: 단계가 막는 항목(n_eff < 6 의 BSS·CI·신뢰도, 판정일 전의 판정어)은
    값이 None 으로 나오고 `hidden`·`hidden_reason` 이 붙는다.
    """
    _n = notes if notes is not None else []
    ks = kill if kill is not None else kill_status(ledger, spy_close, asof, prev=prev_kill,
                                                   unlock_path=unlock_path)
    sc = ks.get("score") or {}
    n_eff = float(sc.get("n_eff") or 0.0)
    stage = horizon_stage(n_eff, int(ks.get("months") or 0))

    df = pd.DataFrame() if ledger is None or len(ledger) == 0 else ledger.copy()
    if len(df):
        df["_asof"] = pd.to_datetime(_frame_dates(df), errors="coerce")
        df = df[df["_asof"].notna()].sort_values("_asof", kind="stable")
        if asof is not None:
            df = df[df["_asof"] <= _ts(asof)]
        ls = ks.get("live_start")
        if ls:
            df = df[df["_asof"] >= pd.Timestamp(ls)]
        df, _lk = holdout_gate(df, unlock_path, _n)
        df = _numeric(df, ["prob_dd5_20", "p2_clim", "y_dd5_20", "p3_w_exec", "p3_sigma_ewma",
                           "p3_sigma_target", "p2_har_fc_20", "rv20_realized", "ret20_in_vix80",
                           "ret20_in_har80", "rule_ret_20", "bh_ret_20", "p3_rule_dd", "p3_lo", "p3_hi",
                           "spy_close"])

    def _col(c):
        return df[c].dropna() if (len(df) and c in df.columns) else pd.Series(dtype=float)

    p = _col("prob_dd5_20")
    w = _col("p3_w_exec")
    state = df["p2_state"].astype("string").dropna() if (len(df) and "p2_state" in df.columns) else pd.Series(dtype="string")

    # (i) 평균 p · p > 0.30 비중
    i_p = {"n": int(len(p)), "mean_p": _jf(p.mean()) if len(p) else None,
           "share_gt_030": _jf((p > 0.30).mean()) if len(p) else None,
           "backtest": {"mean_p": _ref(reference, "p_level", "mean", default=0.162),
                        "share_gt_030": _ref(reference, "p_level", "share_gt_030", default=0.10)}}
    # (ii) 상태 점유
    occ = {s: _jf((state == s).mean()) for s in ("normal", "caution", "reduce")} if len(state) else {}
    ii_state = {"n": int(len(state)), "occupancy": occ,
                "backtest": _ref(reference, "occupancy", default={"normal": 0.79, "caution": 0.16, "reduce": 0.05})}
    # (iii) 결정 전환 · 비중 변경 (연 환산)
    yrs = (len(df) / 252.0) if len(df) else 0.0
    n_sw = _count_changes(pd.Categorical(state).codes.astype(float)) if len(state) else 0
    n_ch = _count_changes(w.to_numpy(dtype=float)) if len(w) else 0
    iii_sw = {"switches_per_year": _jf(n_sw / yrs) if yrs > 0 else None,
              "w_changes_per_year": _jf(n_ch / yrs) if yrs > 0 else None,
              "n_changes_252": int(_count_changes(_tail(w.to_frame("w"), 252)["w"].to_numpy(dtype=float))) if len(w) else 0,
              "churn_ceiling": int(P3["churn_ceiling"]),
              "backtest": {"switches_per_year": _ref(reference, "switching", "per_year", default=3.9),
                           "w_changes_per_year": _ref(reference, "churn", "per_year", default=10.9)}}
    # (iv) 비중 분포
    iv_w = {"n": int(len(w)), "avg_w": _jf(w.mean()) if len(w) else None,
            "share_floor": _jf((w <= float(P3["w_min"]) + 1e-9).mean()) if len(w) else None,
            "share_full": _jf((w >= float(P3["w_max"]) - 1e-9).mean()) if len(w) else None,
            "backtest": _ref(reference, "exposure", default={"avg_w": 0.69, "share_floor": 0.19, "share_full": 0.07})}
    # (v) 변동성 예측
    har, rv, ew = _col("p2_har_fc_20"), _col("rv20_realized"), _col("p3_sigma_ewma")
    joint = df.dropna(subset=[c for c in ("p2_har_fc_20", "rv20_realized") if c in df.columns]) if len(df) else df
    log_mae = None
    if len(joint) and {"p2_har_fc_20", "rv20_realized"}.issubset(joint.columns):
        jj = joint[(joint["p2_har_fc_20"] > 0) & (joint["rv20_realized"] > 0)]
        if len(jj):
            log_mae = _jf(np.mean(np.abs(np.log(jj["p2_har_fc_20"].to_numpy(float))
                                         - np.log(jj["rv20_realized"].to_numpy(float)))))
    bias = None
    if len(df) and {"rv20_realized", "p3_sigma_ewma"}.issubset(df.columns):
        jj = df[(df["rv20_realized"] > 0) & (df["p3_sigma_ewma"] > 0)]
        if len(jj):
            bias = _jf(np.mean(np.log(jj["rv20_realized"].to_numpy(float) / jj["p3_sigma_ewma"].to_numpy(float))))
    v_vol = {"har_log_mae": log_mae, "ewma_log_bias": bias, "n_har": int(len(har)), "n_rv": int(len(rv)),
             "n_ewma": int(len(ew)),
             "backtest": {"har_log_mae_p50": _ref(reference, "vol", "har_log_mae_p50", default=0.26),
                          "ewma_bias_range": _ref(reference, "vol", "ewma_bias_range", default=[-0.34, 0.31])}}
    # (vi) 밴드 포함률
    vi_cov = {}
    for col, band, bt in (("ret20_in_vix80", "vix80", 0.936), ("ret20_in_har80", "har80", 0.80)):
        v = _col(col)
        vi_cov[band] = {"n": int(len(v)), "n_eff": round(len(v) / N_EFF_DIV, 3),
                        "hit": _jf(v.mean()) if len(v) else None,
                        "backtest": _ref(reference, "coverage", band, default=bt)}
    # (vii) Brier·BSS + 같은 길이 창의 백테스트 분포
    vii = {"n": sc.get("n", 0), "n_eff": sc.get("n_eff", 0.0), "brier": sc.get("brier"),
           "base_rate": sc.get("base_rate"), "bss_clim": sc.get("bss_clim"), "bss_m1": sc.get("bss_m1"),
           "bss_vix": sc.get("bss_vix"), "bss_vix_bgk": sc.get("bss_vix_bgk"),
           "ci_bss_clim": sc.get("ci_bss_clim"), "ci_label": sc.get("ci_label"),
           "hist_pct": {str(L): hist_pct(sc.get("bss_clim"), L, reference or {})
                        for L in P3["reference_windows"]},
           "backtest_quantiles": _ref(reference, "quantiles", default={})}
    # "ci_label" 을 후보로 주지 않으면 6m·12m 에서 판정어가 그대로 나간다(0-3m 은 "ci" 접두사에 우연히 걸릴 뿐).
    vii = _redact(vii, stage, ("brier", "bss", "ci", "hist_pct", "ci_label"))
    # (viii) 규칙 vs 보유
    rr, bh = _col("rule_ret_20"), _col("bh_ret_20")
    viii = {"n": int(min(len(rr), len(bh))),
            "rule_minus_bh_20_mean_pp": _jf(100.0 * (rr.mean() - bh.mean())) if (len(rr) and len(bh)) else None,
            "rule_dd_worst": _jf(_col("p3_rule_dd").min()) if len(_col("p3_rule_dd")) else None,
            "backtest": _ref(reference, "relative", default={"63": [-5.7, -1.0, 3.6], "126": [-7.5, -2.1, 2.6],
                                                             "252": [-11.7, -5.9, -1.3],
                                                             "maxdd_252_median": -7.1, "maxdd_252_worst": -12.9})}
    # (ix) 킬 카운트다운·중간 점검
    ix = {"state": ks.get("state"), "countdown": ks.get("countdown"), "episodes5": ks.get("episodes5"),
          "months": ks.get("months"), "stage1_due": ks.get("stage1_due"), "stage2_due": ks.get("stage2_due"),
          "gross_failure_badge": ks.get("gross_failure_badge"),
          "midterm_checks": {"12m": int(ks.get("months") or 0) >= 12, "24m": int(ks.get("months") or 0) >= 24,
                             "can_kill": bool(ks.get("stage1_due"))}}
    if not may_show(stage, "verdict"):                       # 판정일 전에는 판정어를 내지 않는다(§8.4)
        ix["state"] = None
        ix["hidden"] = ["verdict"]
        ix["hidden_reason"] = _HIDDEN_REASON.get(stage)
    # (x) 경보·D10
    al = alarms(ledger, reference, cfg, asof=asof, recomputed=recomputed, unlock_path=unlock_path, notes=_n)
    rc = replay_check(df, recomputed) if recomputed is not None else None
    x_al = {"alarms": al, "codes": sorted({a["code"] for a in al}),
            "replay_mismatch_count": None if rc is None else int(rc["n_mismatch"])}
    # (xi) 등록부: 멤버별 라이브 Brier · 불일치 폭 · 신선 블록 진행
    lo_s, hi_s = _col("p3_lo"), _col("p3_hi")
    width = (hi_s - lo_s).dropna() if (len(lo_s) and len(hi_s)) else pd.Series(dtype=float)
    xi = {"members": {k: {kk: v.get(kk) for kk in ("live_start", "n", "n_eff", "brier", "bss_clim", "verdict")}
                      for k, v in (ks.get("members") or {}).items()},
          "disagreement": {"median_width": _jf(width.median()) if len(width) else None,
                           "p90_width": _jf(width.quantile(0.90)) if len(width) else None,
                           "flag_sessions": int(df["p3_disagree_flag"].astype("string").fillna("").ne("").sum())
                           if (len(df) and "p3_disagree_flag" in df.columns) else 0},
          "fresh_blocks": _ref(reference, "fresh_blocks", default=None)}
    if not may_show(stage, "bss"):                           # 멤버 줄도 같은 규칙을 받는다
        xi["members"] = {k: _redact(v, stage, ("brier", "bss", "verdict")) for k, v in xi["members"].items()}

    return {"stage": stage, "asof": ks.get("asof"), "live_start": ks.get("live_start"),
            "i_p_level": i_p, "ii_state_occupancy": ii_state, "iii_switching": iii_sw, "iv_exposure": iv_w,
            "v_vol_forecast": v_vol, "vi_coverage": vi_cov, "vii_skill": vii, "viii_relative": viii,
            "ix_kill": ix, "x_alarms": x_al, "xi_registry": xi,
            "horizon": horizon_copy({"n_eff": n_eff, "months_elapsed": ks.get("months"), "kill": ks}),
            "notes": _n}


def summary_p3(ledger: pd.DataFrame, spy_close, reference: dict | None, asof,
               prev_kill: dict | None = None, *, unlock_path=HOLDOUT_UNLOCK_PATH,
               recomputed: pd.DataFrame | None = None, p2_deploy_mode=None, p3_deploy_mode=None,
               deploy_sizing=None) -> dict:
    """`results/track_record_p3.json` 의 **단일 원천** (§9 `summary().p3` 계약).

    키: schema_version, n_rows, n_scored, n_eff, live_start, months_elapsed, effective_mode, deploy_sizing,
        sizing{...}, members{...}, disagreement{...}, kill{...}, alarms[...], replay_mismatch_count,
        coverage{vix80, har80}, horizon{...}, panel{...}, notes[]
    """
    notes: list[str] = []
    ks = kill_status(ledger, spy_close, asof, prev=prev_kill, unlock_path=unlock_path)
    notes.extend(ks.get("notes") or [])
    panel = live_panel(ledger, spy_close, reference, asof, unlock_path=unlock_path, kill=ks,
                       recomputed=recomputed, notes=notes)
    sc = ks.get("score") or {}
    n_rows = 0 if ledger is None else int(len(ledger))
    mode = effective_mode(p2_deploy_mode, p3_deploy_mode) if (p2_deploy_mode or p3_deploy_mode) else None
    rc = replay_check(ledger, recomputed) if (recomputed is not None and ledger is not None) else None

    out = {
        "schema_version": SCHEMA_VERSION,
        "n_rows": n_rows,
        "n_scored": int(sc.get("n") or 0),
        "n_eff": float(sc.get("n_eff") or 0.0),
        "live_start": ks.get("live_start"),
        "months_elapsed": int(ks.get("months") or 0),
        "effective_mode": mode,
        "deploy_sizing": deploy_sizing,
        "sizing": {"avg_w": panel["iv_exposure"]["avg_w"],
                   "n_changes_252": panel["iii_switching"]["n_changes_252"],
                   "realized_vol_ratio": None, "rule_dd": panel["viii_relative"]["rule_dd_worst"],
                   "rule_vs_bh_20": panel["viii_relative"]["rule_minus_bh_20_mean_pp"]},
        "members": {k: {kk: v.get(kk) for kk in ("live_start", "n", "n_eff", "brier", "bss_clim", "verdict")}
                    for k, v in (ks.get("members") or {}).items()},
        "disagreement": panel["xi_registry"]["disagreement"],
        "kill": {k: v for k, v in ks.items() if k != "notes"},
        "alarms": panel["x_alarms"]["alarms"],
        "replay_mismatch_count": None if rc is None else int(rc["n_mismatch"]),
        "coverage": {b: panel["vi_coverage"][b]["hit"] for b in ("vix80", "har80") if b in panel["vi_coverage"]},
        "horizon": panel["horizon"],
        "panel": panel,
        "holdout_locked": bool(ks.get("holdout_locked")),
        "notes": notes,
    }
    # 실현변동성/σ_T — D4 와 같은 계산(60세션)
    for a in out["alarms"]:
        if a["code"] == "D4_vol_target":
            out["sizing"]["realized_vol_ratio"] = a["value"]
    return out


def save_summary_p3(summary: dict, path=TRACK_P3_PATH) -> Path:
    """summary_p3() 결과를 엄격 JSON 으로 저장한다(NaN 없음)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return p
