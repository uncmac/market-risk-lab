# -*- coding: utf-8 -*-
"""Phase 3 멤버 등록부·앙상블 — 전부 그림자(shadow) (ARCHITECTURE_PHASE3.md §5).

계약:
    REGISTRY (동결 튜플) · registry_repr() · registry_tuple_sha256() · registry_sha256()
    platt_walk_forward(x_hmm, y, refit_dates, purge, train_start, C) -> (DataFrame, [LogitModel])
    combine(probs, statuses) -> Series
    disagreement(probs, statuses, p2_lo, p2_hi, cfg) -> DataFrame
    admission_test(oos, candidate, admitted, blocks, *, frac, block, n_boot, seed, live_start=None) -> dict
    fresh_blocks(idx, live_start, months=3, label_h=20) -> [(Timestamp, Timestamp)]
    admission_power_table(n_blocks=11, need=8, qs=(...)) -> DataFrame
    registry_table(oos, statuses, admissions, live_scores) -> [dict]
    apply_admission(model_p3, result, ledger_no, effective_refit) -> dict

설계 원칙 (§0 표·§2·§5 그대로)
* **생산 확률은 Phase 3 에서 한 글자도 바뀌지 않는다.** 1일차 admitted = {p2} 뿐이라 `combine()` 은 p2 와
  비트 동일하고, 이 모듈의 어떤 산출도 생산 확률 열(`prob_dd5_20`)이나 `model_p2.json` 에 쓰이지 않는다.
  멤버 H·M1 은 `shadow` 로 등록되어 매일 장부에 기록·채점되지만 확률·결정층·비중·킬룰에 들어가지 않는다.
  이 모듈은 Phase 2 생산 모델 파일의 경로 상수도 저장 함수도 import 하지 않고, 어떤 파일에도 쓰지 않는다
  (산출물 기록은 주간 `run_phase3.py` 의 몫) — 그림자가 생산으로 새어 들어갈 통로 자체가 없다.
* **채택은 신선 자료로만**(§5.4): 2003~2024-08 관측 기록과 홀드아웃(2024-09-01~)은 **기각만** 할 수 있다.
  `admission_test(live_start=None)` 은 통계를 다 계산하되 verdict 를 절대 "ADMIT" 로 내지 않고
  (`admissible=False`, 통계상의 판정은 `verdict_raw`), 홀드아웃 행이 검정에 들어오면 ValueError 다.
  `fresh_blocks` 는 라이브 장부 시작일이 HOLDOUT_START 이하이면 ValueError(홀드아웃 2회 접근 금지).
* **적합 파라미터 회계**(§14): 생산 확률 4개(Phase 2) + Phase 3 **0개**. 멤버 H 의 Platt 2(K_s, 지도)와
  HMM θ 12(K_u, 비지도)는 예산 밖 공개값이며 카드 정직 스트립에 줄로 나온다.
* **Platt 레시피 동일성**: 그림자 멤버 H 는 Phase 2 의 M1/M3 와 **같은** sklearn 사양으로 적합한다 —
  `calibrate.fit_refit`(→ `model.fit_logit`, `model.FIT_KWARGS`: l2 · C=1.0 · lbfgs · tol 1e-8 · 표준화 없음,
  20세션 퍼지 · train_start 1993-10-14). 별도 적합 코드를 두지 않는다(레시피가 갈라지지 않게).
* 조용한 실패 금지(멤버 결측·상태 미지정은 ValueError 또는 warn+기록) · 결정론 seed=0 ·
  점(point-in-time): Platt walk-forward 는 퍼지된 학습창만 본다.
* `mrl.regime` 은 **지연 import** 한다(모듈 경계; regime 이 없으면 H 특징 생성 함수만 실패하고 등록부는 산다).
"""
from __future__ import annotations

import hashlib
import math
import warnings
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from mrl.calibrate import assign_blocks, fit_refit, loss_diff_ci
from mrl.config import ENSEMBLE_P3, HMM_P3, HOLDOUT_START, ROOT
from mrl.evaluate import brier
from mrl.model import PARAM_COUNT, LogitModel

__all__ = [
    "SCHEMA_VERSION", "Member", "REGISTRY", "REGISTRY_NAMES", "MEMBER_STATUSES", "COMBINING_STATUSES",
    "BAND_EXCLUDED_STATUSES", "UNREGISTERED", "PLATT_FEATURE", "MEMBER_COLUMN_CANDIDATES",
    "registry_repr", "registry_tuple_sha256", "registry_sha256", "member", "day1_statuses", "check_statuses",
    "production_members", "production_param_count", "shadow_param_counts", "assert_shadow_only",
    "effective_statuses", "member_probs", "combine", "disagreement",
    "platt_walk_forward", "hmm_x_walk_forward", "member_h_walk_forward",
    "admission_test", "admission_sequence", "admission_due", "fresh_blocks", "fresh_block_table",
    "admission_power_table", "registry_table", "apply_admission", "next_january",
]

SCHEMA_VERSION = 1

# ------------------------------------------------------------------
# 5.1 등록부 (동결 튜플; 변경은 번호 붙인 장부 항목으로만 — tests/test_ensemble.py 가 해시를 고정한다)
# ------------------------------------------------------------------
MEMBER_STATUSES = ("seed", "admitted", "shadow", "candidate_rejected", "killed")
COMBINING_STATUSES = ("seed", "admitted")        # 생산 평균(동일가중)에 들어가는 상태
BAND_EXCLUDED_STATUSES = ("killed",)             # 불일치 구간에서 빠지는 상태 (§5.1: killed 만)


@dataclass(frozen=True)
class Member:
    """등록부 한 줄 (§5.1 표). frozen — 실행 중 변경 불가, 상태는 별도 dict(statuses)로 다룬다."""
    name: str
    kind: str            # "seed" | "ladder" | "hmm_platt"
    source: str          # 확률 원천 (장부/OOS 열 또는 사상 설명)
    k_s: int             # 지도(라벨 대면) 적합 파라미터
    k_u: int             # 비지도 적합 파라미터
    day1_status: str     # 1일차 상태
    ledger_no: str       # 장부 항목 번호

    def __post_init__(self):
        if self.day1_status not in MEMBER_STATUSES:
            raise ValueError(f"알 수 없는 1일차 상태: {self.day1_status!r}")
        if self.k_s < 0 or self.k_u < 0:
            raise ValueError("파라미터 수는 음수일 수 없습니다")


REGISTRY: tuple[Member, ...] = (
    Member("p2", "seed", "prob_dd5_20", 4, 0, "seed", "#2"),
    Member("M1", "ladder", "p2_p_m1", 2, 0, "shadow", "#8"),
    Member("H", "hmm_platt", "platt(x_hmm)", 2, 12, "shadow", "#3a"),
)
REGISTRY_NAMES: tuple[str, ...] = tuple(m.name for m in REGISTRY)

# (미등록) 폭/신용 멤버 B — 단독 BSS −0.055·AUC 0.495 로 등록하지 않는다. 수치만 기록(§0 표·§5.1).
UNREGISTERED: tuple[Member, ...] = (
    Member("B", "logit", "-d63 ln(HYG/IEF), -d63 ln(RSP/SPY)", 3, 0, "candidate_rejected", "-"),
)

PLATT_FEATURE = "x_hmm"                          # 멤버 H 의 유일한 Platt 특징 (§4.1: logit P_high)

# 프레임에서 멤버 확률 열을 찾는 순서(고정). 첫 번째로 존재하는 열을 쓴다 — 못 찾으면 ValueError(조용한 대체 없음).
MEMBER_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "p2": ("p2", "p_p2", "prob_dd5_20", "p_m3"),
    "M1": ("M1", "p_m1", "p2_p_m1"),
    "H": ("H", "p_h", "p3_p_h"),
}

_SHA_FILES = ("ensemble.py", "regime.py")        # §2 결정론: registry_sha256 = 두 소스 + REGISTRY
_HOLDOUT = pd.Timestamp(HOLDOUT_START)


# ------------------------------------------------------------------
# 해시 (등록부 동결)
# ------------------------------------------------------------------
def registry_repr() -> str:
    """REGISTRY 의 정규 텍스트. 한 줄 = 멤버 하나 (§5.1 표의 열 순서)."""
    lines = [f"ensemble-registry-v{SCHEMA_VERSION}"]
    lines += [f"{m.name}|{m.kind}|{m.source}|{m.k_s}|{m.k_u}|{m.day1_status}|{m.ledger_no}" for m in REGISTRY]
    return "\n".join(lines) + "\n"


def registry_tuple_sha256() -> str:
    """REGISTRY 튜플만의 sha256 — 소스 편집과 무관하게 **등록부 변경**만 잡는다(테스트가 고정)."""
    return hashlib.sha256(registry_repr().encode("utf-8")).hexdigest()


def registry_sha256() -> str:
    """`ensemble.py` + `regime.py` 소스(CRLF→LF 정규화; 없으면 'MISSING:<name>') + REGISTRY 의 sha256.
    모든 P3 산출물·장부 행에 기록한다. regime.py 가 생기면 값이 바뀐다 — 그것이 '코드가 산출물보다 새롭다'
    신호이며 `daily.py` 는 불일치 시 exit 1 이다(§2)."""
    h = hashlib.sha256()
    for name in _SHA_FILES:
        p = ROOT / "mrl" / name
        h.update(name.encode("utf-8") + b"\n")
        h.update(p.read_bytes().replace(b"\r\n", b"\n") if p.exists() else b"MISSING:" + name.encode("utf-8"))
        h.update(b"\n")
    h.update(registry_repr().encode("utf-8"))
    return h.hexdigest()


def member(name: str) -> Member:
    """이름으로 등록 멤버를 찾는다. 미등록 이름은 ValueError(등록부는 동결이다)."""
    for m in REGISTRY:
        if m.name == name:
            return m
    raise ValueError(f"등록부에 없는 멤버: {name!r} (등록: {list(REGISTRY_NAMES)})")


# ------------------------------------------------------------------
# 상태 (statuses)
# ------------------------------------------------------------------
def day1_statuses() -> dict[str, str]:
    """1일차 상태 — {'p2': 'seed', 'M1': 'shadow', 'H': 'shadow'}. admitted = ∅ 이므로 평균 ≡ p2."""
    return {m.name: m.day1_status for m in REGISTRY}


def check_statuses(statuses: dict[str, str], names: Sequence[str] | None = None) -> dict[str, str]:
    """상태 dict 검증: 키는 등록 멤버, 값은 MEMBER_STATUSES. names 를 주면 전부 덮는지도 본다.
    조용히 기본값을 채우지 않는다 — 빠진 멤버는 ValueError."""
    if not isinstance(statuses, dict):
        raise TypeError("statuses 는 {멤버: 상태} dict 여야 합니다")
    out: dict[str, str] = {}
    for k, v in statuses.items():
        if k not in REGISTRY_NAMES:
            raise ValueError(f"등록부에 없는 멤버의 상태: {k!r} (등록: {list(REGISTRY_NAMES)})")
        if v not in MEMBER_STATUSES:
            raise ValueError(f"{k}: 알 수 없는 상태 {v!r} (허용: {MEMBER_STATUSES})")
        out[k] = str(v)
    if names is not None:
        missing = [n for n in names if n not in out]
        if missing:
            raise ValueError(f"상태가 지정되지 않은 멤버: {missing}")
    return out


def production_members(statuses: dict[str, str]) -> tuple[str, ...]:
    """동일가중 평균에 실제로 들어가는 멤버(등록 순서 고정). 1일차엔 ('p2',) 하나뿐이다."""
    st = check_statuses(statuses)
    return tuple(n for n in REGISTRY_NAMES if st.get(n) in COMBINING_STATUSES)


def production_param_count(statuses: dict[str, str]) -> int:
    """생산 확률에 닿는 지도 적합 파라미터 합계 K_s. 1일차 = 4 (= mrl.model.PARAM_COUNT). 상한 ENSEMBLE_P3['K_s_cap']."""
    total = sum(member(n).k_s for n in production_members(statuses))
    cap = int(ENSEMBLE_P3["K_s_cap"])
    if total > cap:
        raise ValueError(f"K_s 합계 {total} > 상한 {cap} (§6-c)")
    return int(total)


def shadow_param_counts(statuses: dict[str, str] | None = None) -> dict[str, int]:
    """공개용 파라미터 회계: {'K_s_production', 'K_u_production', 'K_s_shadow', 'K_u_shadow',
    'phase3_added_to_production'}. K_u 총계는 채택 여부와 무관하게 **보존**된다 — 멤버 H 가 생산 평균에
    들어가면 HMM θ 의 비지도 12개는 그림자가 아니라 **생산** 줄로 옮겨 공개한다
    (§14: 채택 시 'M3+H = K_s 6 · K_u 12'; §6-c(b): K_u 는 별도 줄로 세어 공개, 상한 K_u_cap)."""
    st = check_statuses(statuses if statuses is not None else day1_statuses(), REGISTRY_NAMES)
    prod = production_members(st)
    k_s_prod = sum(member(n).k_s for n in prod)
    k_u_prod = sum(member(n).k_u for n in prod)
    k_s_shadow = sum(m.k_s for m in REGISTRY if m.name not in prod)
    k_u_shadow = sum(m.k_u for m in REGISTRY if m.name not in prod)
    cap_u = int(ENSEMBLE_P3["K_u_cap"])
    if k_u_prod + k_u_shadow > cap_u:
        raise ValueError(f"K_u 합계 {k_u_prod + k_u_shadow} > 상한 {cap_u} (§6-c(b))")
    return {"K_s_production": int(k_s_prod), "K_u_production": int(k_u_prod),
            "K_s_shadow": int(k_s_shadow), "K_u_shadow": int(k_u_shadow),
            "phase3_added_to_production": int(k_s_prod - PARAM_COUNT)}


def assert_shadow_only(statuses: dict[str, str] | None = None, *, where: str = "") -> None:
    """생산 확률 보호 가드: seed(p2) 밖의 어떤 멤버도 평균에 들어가 있으면 ValueError.
    채택은 `apply_admission`(신선 블록 검정 통과 + 다음 1월 발효)로만 일어나야 한다."""
    st = check_statuses(statuses if statuses is not None else day1_statuses())
    leak = [n for n in production_members(st) if n != "p2"]
    if leak:
        raise ValueError(f"그림자 전용 위반{(' @' + where) if where else ''}: {leak} 가 생산 평균에 들어 있습니다 "
                         f"(§5.4 채택 규칙과 다음 1월 발효를 거치지 않은 상태 변경)")
    if int(sum(member(n).k_s for n in production_members(st))) != PARAM_COUNT:
        raise ValueError(f"생산 확률의 적합 파라미터가 {PARAM_COUNT} 가 아닙니다 (§2 예산)")


def effective_statuses(model_p3: dict | None, asof) -> dict[str, str]:
    """`model_p3.json.members` 의 상태를 **발효일**로 걸러 오늘 유효한 상태를 만든다(§5.4: 연중 변경 금지).
    status == 'admitted' 인데 effective_refit > asof 이면 아직 'shadow' 다."""
    asof = pd.Timestamp(asof)
    out = day1_statuses()
    members = (model_p3 or {}).get("members") or {}
    if not isinstance(members, dict):
        raise TypeError("model_p3['members'] 는 dict 여야 합니다")
    for name, info in members.items():
        if name not in REGISTRY_NAMES:
            raise ValueError(f"model_p3.members 에 등록부 밖 멤버: {name!r}")
        if not isinstance(info, dict) or "status" not in info:
            raise ValueError(f"model_p3.members[{name!r}] 에 status 가 없습니다")
        status = str(info["status"])
        if status not in MEMBER_STATUSES:
            raise ValueError(f"{name}: 알 수 없는 상태 {status!r}")
        eff = info.get("effective_refit")
        if status == "admitted":
            if eff is None:
                raise ValueError(f"{name}: admitted 인데 effective_refit 가 없습니다 (§5.4 다음 1월 발효)")
            if pd.Timestamp(eff) > asof:
                status = "shadow"                       # 아직 발효 전 — 평균에 들어가지 않는다
        out[name] = status
    return out


# ------------------------------------------------------------------
# 멤버 확률 프레임
# ------------------------------------------------------------------
def _idx(obj) -> pd.DatetimeIndex:
    idx = obj if isinstance(obj, pd.Index) else pd.Index(obj)
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(idx)
    if idx.has_duplicates or not idx.is_monotonic_increasing:
        raise ValueError("인덱스는 중복 없는 오름차순 거래일이어야 합니다")
    return idx


def _resolve_column(frame: pd.DataFrame, name: str, required: bool = True) -> str | None:
    for c in MEMBER_COLUMN_CANDIDATES[name]:
        if c in frame.columns:
            return c
    if required:
        raise ValueError(f"멤버 {name!r} 의 확률 열을 찾지 못했습니다 "
                         f"(찾은 이름: {MEMBER_COLUMN_CANDIDATES[name]}; 프레임 열: {list(frame.columns)[:12]})")
    return None


def member_probs(frame: pd.DataFrame, members: Sequence[str] | None = None, *, required: bool = True,
                 columns: dict[str, str] | None = None) -> pd.DataFrame:
    """OOS·장부 프레임에서 멤버 확률만 뽑아 **멤버 이름 열**로 돌려준다(등록 순서 고정).
    columns 로 열 이름을 직접 줄 수 있다. 열을 못 찾으면 required=True 면 ValueError, False 면 건너뛰고 경고."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame 은 DataFrame 이어야 합니다")
    names = tuple(members) if members is not None else REGISTRY_NAMES
    for n in names:
        if n not in REGISTRY_NAMES:
            raise ValueError(f"등록부에 없는 멤버: {n!r}")
    out = pd.DataFrame(index=frame.index)
    out.index.name = frame.index.name
    missing: list[str] = []
    for n in [m for m in REGISTRY_NAMES if m in names]:
        col = (columns or {}).get(n) or _resolve_column(frame, n, required=required)
        if col is None:
            missing.append(n)
            continue
        if col not in frame.columns:
            raise ValueError(f"멤버 {n!r} 의 열 {col!r} 이 프레임에 없습니다")
        out[n] = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
    if missing:
        warnings.warn(f"member_probs: 확률 열이 없는 멤버 {missing} — 프레임에서 제외했습니다", RuntimeWarning)
    out.attrs = {"missing_members": missing}
    return out


def _check_probs(probs: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(probs, pd.DataFrame):
        raise TypeError("probs 는 멤버 이름을 열로 갖는 DataFrame 이어야 합니다")
    unknown = [c for c in probs.columns if c not in REGISTRY_NAMES]
    if unknown:
        raise ValueError(f"probs 에 등록부 밖 열이 있습니다: {unknown} (등록: {list(REGISTRY_NAMES)})")
    A = probs.to_numpy(dtype=float)
    fin = np.isfinite(A)
    if fin.any() and ((A[fin] < 0.0) | (A[fin] > 1.0)).any():
        raise ValueError("멤버 확률이 [0,1] 밖입니다")
    return probs


def _row_mean(probs: pd.DataFrame, names: Sequence[str]) -> pd.Series:
    """동일가중 평균. **NaN 전파**: 하나라도 NaN 이면 그 행은 NaN(조용히 빼고 재조정하지 않는다).
    멤버가 하나면 그 열을 그대로 복사한다(비트 동일)."""
    names = list(names)
    if not names:
        return pd.Series(np.nan, index=probs.index, dtype=float)
    if len(names) == 1:
        return probs[names[0]].astype(float).copy()
    return pd.Series(probs[names].to_numpy(dtype=float).mean(axis=1), index=probs.index, dtype=float)


# ------------------------------------------------------------------
# 5.3 결합 · 불일치 구간
# ------------------------------------------------------------------
def combine(probs: pd.DataFrame, statuses: dict[str, str]) -> pd.Series:
    """admitted(seed 포함)·가용 멤버의 **동일가중 평균**(확률 공간). 가중 재조정 없음.

    * '가용' = 그 멤버 열이 probs 에 있다는 뜻이다. 행 단위 결측은 **전파**한다 —
      admitted 멤버 하나가 NaN 이면 그 행의 평균은 NaN(§13 test_ensemble 계약).
    * admitted 가 하나도 없으면 전 행 NaN + 경고(호출자가 상태를 이월한다; P2 §2 규칙).
    * **1일차 admitted = {p2} 이므로 결과는 p2 와 비트 동일하다** — Phase 3 는 생산 확률을 만들지 않는다.
    """
    probs = _check_probs(probs)
    st = check_statuses(statuses, tuple(probs.columns))
    names = [n for n in REGISTRY_NAMES if n in probs.columns and st[n] in COMBINING_STATUSES]
    out = _row_mean(probs, names)
    out.name = "p_ens"
    if not names:
        warnings.warn("combine: 평균에 들어갈 멤버가 없습니다 (전 행 NaN — 상태 이월)", RuntimeWarning)
    out.attrs = {"members": tuple(names), "weight": (1.0 / len(names)) if names else float("nan"),
                 "registry_sha": registry_sha256()}
    return out


def _runs_at_least(mask: np.ndarray, min_len: int) -> np.ndarray:
    """True 런의 길이가 min_len 이상인 구간 전체를 True 로 표시한다."""
    out = np.zeros(len(mask), dtype=bool)
    start = None
    for i, v in enumerate(list(mask) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= int(min_len):
                out[start:i] = True
            start = None
    return out


def disagreement(probs: pd.DataFrame, statuses: dict, p2_lo: pd.Series, p2_hi: pd.Series,
                 cfg: dict = ENSEMBLE_P3["disagree_flag"]) -> pd.DataFrame:
    """표시 전용 불일치 구간. lo = min(p2_lo, 등록·비killed 멤버 min), hi = max(p2_hi, 멤버 max).

    열: lo, hi, width, src('members'|'p2'), flag. flag 는 width > cfg['width'] 가 cfg['sessions'] 세션 이상
    **연속**인 구간 전체에 True. killed 멤버는 구간에서 빠진다(§5.1). p2 밴드는 언제나 포함된다.
    """
    probs = _check_probs(probs)
    st = check_statuses(statuses, tuple(probs.columns))
    idx = probs.index
    lo2 = pd.Series(p2_lo, dtype=float).reindex(idx).to_numpy(dtype=float)
    hi2 = pd.Series(p2_hi, dtype=float).reindex(idx).to_numpy(dtype=float)
    bad = np.isfinite(lo2) & np.isfinite(hi2) & (lo2 > hi2)
    if bad.any():
        raise ValueError(f"p2 밴드가 뒤집힌 행이 {int(bad.sum())}개 있습니다 (lo > hi)")

    names = [n for n in REGISTRY_NAMES if n in probs.columns and st[n] not in BAND_EXCLUDED_STATUSES]
    if names:
        A = probs[names].to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # 전부 NaN 인 행 → NaN (경고 억제)
            mlo = np.nanmin(A, axis=1)
            mhi = np.nanmax(A, axis=1)
    else:
        mlo = np.full(len(idx), np.nan)
        mhi = np.full(len(idx), np.nan)

    lo = np.fmin(lo2, mlo)                       # fmin/fmax 는 한쪽이 NaN 이면 다른 쪽을 준다
    hi = np.fmax(hi2, mhi)
    width = hi - lo
    ext = ((mlo < lo2) | (~np.isfinite(lo2) & np.isfinite(mlo))
           | (mhi > hi2) | (~np.isfinite(hi2) & np.isfinite(mhi)))
    src = np.where(ext, "members", "p2")
    wide = np.isfinite(width) & (width > float(cfg["width"]))
    flag = _runs_at_least(wide, int(cfg["sessions"]))
    out = pd.DataFrame({"lo": lo, "hi": hi, "width": width, "src": src, "flag": flag}, index=idx)
    out.index.name = idx.name
    out.attrs = {"members": tuple(names), "cfg": dict(cfg), "excluded": tuple(n for n in probs.columns
                                                                              if n not in names)}
    return out


# ------------------------------------------------------------------
# 5.2 그림자 멤버 H — Platt (Phase 2 M1/M3 와 같은 sklearn 레시피)
# ------------------------------------------------------------------
def _dstr(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def platt_walk_forward(x_hmm, y, refit_dates, purge: int = HMM_P3["purge"], train_start=HMM_P3["train_start"],
                       C: float = HMM_P3["platt_C"], *,
                       train_feats: dict | None = None) -> tuple[pd.DataFrame, list[LogitModel]]:
    """멤버 H 의 Platt walk-forward (§5.2).

    재적합일 R_y 마다 `calibrate.fit_refit` 로 **Phase 2 와 같은** 로지스틱(l2, C=1.0, lbfgs, tol 1e-8,
    표준화 없음)을 퍼지된 학습창(train_start ≤ t, pos(t) ≤ pos(R_y) − purge − 1, y·x 유한)에 적합하고
    [R_y, R_{y+1}) 를 채점한다. 첫 재적합일 이전 행은 θ/Platt_1 로 채우되 `in_sample=True` 로 표시한다
    (구간·사다리 표시용; 채점에 쓰면 안 된다).

    `train_feats` 는 재적합일 문자열 → x_hmm Series 의 dict 로, **§5.2/§4.1 의 학습 특징**
    (θ_y 를 학습 구간 전체에 전방 필터한 값)이다. 주지 않으면 채점용 walk-forward 모자이크로 적합하며
    (계약 이탈) `attrs['warnings']` 에 남긴다 — 조용히 다른 것을 적합하지 않는다.

    반환 (df, models): df 열 p_h, refit_year, in_sample. 각 적합은 K_s = 2 (assert).
    """
    x = pd.Series(x_hmm, dtype=float).copy()
    idx = _idx(x.index)
    x.index = idx
    ys = pd.Series(y, dtype=float)
    ys = ys.reindex(idx) if not ys.index.equals(idx) else ys
    feats = x.to_frame(PLATT_FEATURE)

    rds = [pd.Timestamp(d) for d in refit_dates]
    if not rds:
        raise ValueError("refit_dates 가 비어 있습니다")
    if any(b <= a for a, b in zip(rds, rds[1:])):
        raise ValueError("refit_dates 는 오름차순이어야 합니다")
    miss = [_dstr(R) for R in rds if R not in idx]
    if miss:
        raise ValueError(f"refit_date 가 인덱스에 없습니다: {miss[:5]}")

    df = pd.DataFrame({"p_h": np.nan, "refit_year": 0, "in_sample": False}, index=idx)
    df.index.name = idx.name
    df["in_sample"] = df["in_sample"].astype(bool)
    models: list[LogitModel] = []
    warn_list: list[str] = []
    k_s = member("H").k_s

    if train_feats is None and len(rds) > 1:
        warn_list.append("train_feats 미지정 — 학습 특징이 walk-forward 모자이크입니다 (§5.2 이탈)")

    for i, R in enumerate(rds):
        R_next = rds[i + 1] if i + 1 < len(rds) else None
        sel = (idx >= R) if R_next is None else ((idx >= R) & (idx < R_next))
        # §5.2/§4.1: 학습 특징은 θ_y 를 학습 구간에 전방 필터한 값 — 채점(feats)은 walk-forward 모자이크 그대로
        if train_feats is None:
            tf = feats
        else:
            key = _dstr(R)
            if key not in train_feats:
                raise ValueError(f"train_feats 에 재적합일 {key} 의 x_hmm 이 없습니다 (§5.2)")
            tf = pd.Series(train_feats[key], dtype=float).reindex(idx).to_frame(PLATT_FEATURE)
        m = fit_refit(tf, ys, R, (PLATT_FEATURE,), C=float(C), purge=int(purge), train_start=train_start, rung="H")
        if m.n_params() != k_s:
            raise AssertionError(f"H@{_dstr(R)}: Platt 파라미터 {m.n_params()} ≠ K_s {k_s}")
        m.model_id = f"p3h-{m.spec_sha256[:8]}-{m.refit_date}"
        m.extra = {**m.extra, "member": "H", "feature": PLATT_FEATURE}
        models.append(m)
        score_idx = idx[sel]
        if len(score_idx):
            p = m.predict_proba(feats.loc[score_idx])
            df.loc[score_idx, "p_h"] = p.to_numpy()
            df.loc[score_idx, "refit_year"] = int(R.year)
            n_nan = int(p.isna().sum())
            if n_nan:
                warn_list.append(f"H@{_dstr(R)}: OOS {n_nan}행의 x_hmm 결측 → p_h NaN")

    pre = idx[idx < rds[0]]                       # 첫 재적합 이전 — Platt 학습·표시 전용(in_sample)
    if len(pre):
        p = models[0].predict_proba(feats.loc[pre])
        df.loc[pre, "p_h"] = p.to_numpy()
        df.loc[pre, "refit_year"] = int(rds[0].year)
        df.loc[pre, "in_sample"] = True

    df["refit_year"] = df["refit_year"].astype(int)
    df["in_sample"] = df["in_sample"].astype(bool)
    df.attrs = {"n_refits": len(rds), "purge": int(purge), "train_start": _dstr(train_start), "C": float(C),
                "feature": PLATT_FEATURE, "k_s": int(k_s), "warnings": warn_list,
                "train_feats": train_feats is not None,
                "registry_sha": registry_sha256()}
    return df, models


def _regime():
    """mrl.regime 지연 import (동시 작성 중인 모듈; 없으면 명확히 실패)."""
    try:
        from mrl import regime                                 # noqa: WPS433 - 지연 import
    except ImportError as e:                                   # pragma: no cover - regime 부재 경로
        raise ImportError("mrl/regime.py 가 필요합니다 (§4 국면 엔진). 등록부·결합·채택은 regime 없이도 동작합니다.") from e
    return regime


def hmm_x_walk_forward(close: pd.Series, refit_dates, *, purge: int = HMM_P3["purge"],
                       train_start=HMM_P3["train_start"], cfg: dict = HMM_P3, prev_thetas=None):
    """regime 계약(§4.2)으로 x_hmm 경로를 만든다: observations → hmm_walk_forward. 반환 (wf_df, thetas)."""
    regime = _regime()
    obs = regime.observations(close, rv_window=int(cfg["rv_window"]))
    return regime.hmm_walk_forward(obs, refit_dates, purge=int(purge), train_start=train_start, cfg=cfg,
                                   prev_thetas=prev_thetas)


def member_h_walk_forward(close: pd.Series, y, refit_dates, *, purge: int = HMM_P3["purge"],
                          train_start=HMM_P3["train_start"], cfg: dict = HMM_P3, prev_thetas=None,
                          C: float = HMM_P3["platt_C"]):
    """멤버 H 전체 경로(주간 작업용): HMM walk-forward → x_hmm → Platt walk-forward.
    반환 (df, platt_models, thetas). df = Platt 열 + regime 열(p_high, p20, q20, x_hmm, theta_id)."""
    wf, thetas = hmm_x_walk_forward(close, refit_dates, purge=purge, train_start=train_start, cfg=cfg,
                                    prev_thetas=prev_thetas)
    if "x_hmm" not in wf.columns:
        raise ValueError("regime.hmm_walk_forward 결과에 x_hmm 열이 없습니다 (§4.2 계약)")
    regime = _regime()
    obs = regime.observations(close, rv_window=int(cfg["rv_window"]))
    clip = float(cfg["clip"])
    train_feats: dict[str, pd.Series] = {}
    for th in thetas:                      # θ_y 를 학습 구간 전체에 전방 필터 (§4.1; 전방만이라 누수 없음)
        v = regime.filter_probabilities(obs, th).to_numpy(dtype=float)
        ok = np.isfinite(v)
        xv = np.full(len(v), np.nan)
        xv[ok] = regime.logit(np.clip(v[ok], clip, 1.0 - clip))
        train_feats[th.refit_date] = pd.Series(xv, index=obs.index)
    df, models = platt_walk_forward(wf["x_hmm"], y, refit_dates, purge=purge, train_start=train_start, C=C,
                                    train_feats=train_feats)
    keep = [c for c in ("p_high", "p20", "q20", "x_hmm", "theta_id") if c in wf.columns]
    out = df.join(wf[keep])
    out.attrs = {**df.attrs, "hmm": dict(wf.attrs)}
    return out, models, thetas


# ------------------------------------------------------------------
# 5.4 채택 규칙 — 관측 기록은 기각만, 채택은 신선 자료로만
# ------------------------------------------------------------------
def fresh_blocks(idx, live_start, months: int = ENSEMBLE_P3["fresh_block_months"], label_h: int = 20
                 ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """라이브 장부 시작일부터 `months` 달력월 타일([a, b) 반개구간) 중 **완결된** 블록만.

    완결 = 블록에 세션이 있고, 블록 마지막 세션 뒤에 라벨 지평(label_h) 만큼의 세션이 idx 에 실현돼 있다.
    live_start ≤ HOLDOUT_START 이면 ValueError — 홀드아웃 행은 채택에 절대 쓰지 않는다(§2·§6 홀드아웃 1회 규칙).
    """
    idx = _idx(idx)
    ls = pd.Timestamp(live_start)
    if ls <= _HOLDOUT:
        raise ValueError(f"신선 블록의 시작 {_dstr(ls)} 이 홀드아웃 시작 {HOLDOUT_START} 이하입니다 — "
                         "채택 증거로 홀드아웃/관측 기록을 쓸 수 없습니다 (§5.4)")
    if int(months) <= 0 or int(label_h) < 0:
        raise ValueError("months 는 양수, label_h 는 0 이상이어야 합니다")
    tbl = fresh_block_table(idx, ls, months=months, label_h=label_h)
    return [(pd.Timestamp(r.start), pd.Timestamp(r.end)) for r in tbl.itertuples() if bool(r.complete)]


def fresh_block_table(idx, live_start, months: int = ENSEMBLE_P3["fresh_block_months"], label_h: int = 20
                      ) -> pd.DataFrame:
    """신선 블록 진행 표: block, start, end, n, last_session, sessions_after, complete (미완결 포함)."""
    idx = _idx(idx)
    ls = pd.Timestamp(live_start)
    if ls <= _HOLDOUT:
        raise ValueError(f"신선 블록의 시작 {_dstr(ls)} 이 홀드아웃 시작 {HOLDOUT_START} 이하입니다 (§5.4)")
    rows = []
    k = 0
    while True:
        a = ls + pd.DateOffset(months=int(months) * k)
        b = ls + pd.DateOffset(months=int(months) * (k + 1))
        if len(idx) == 0 or a > idx[-1]:
            break
        sub = idx[(idx >= a) & (idx < b)]
        n = int(len(sub))
        last = sub[-1] if n else None
        after = int((idx > last).sum()) if last is not None else 0
        rows.append({"block": k + 1, "start": _dstr(a), "end": _dstr(b), "n": n,
                     "last_session": _dstr(last) if last is not None else None,
                     "sessions_after": after, "complete": bool(n > 0 and after >= int(label_h))})
        k += 1
        if k > 10_000:                                          # 방어적 상한 (달력 오프셋 오류 방지)
            raise RuntimeError("신선 블록 타일이 지나치게 많습니다 — 입력을 확인하세요")
    return pd.DataFrame(rows, columns=["block", "start", "end", "n", "last_session", "sessions_after", "complete"])


def admission_power_table(n_blocks: int = ENSEMBLE_P3["fresh_blocks_min"], need: int = 8,
                          qs: Iterable[float] = (0.5, 0.6, 0.7, 0.8, 0.9)) -> pd.DataFrame:
    """검정력 표: 블록별 개선확률 q 일 때 P(≥need / n_blocks) — 이항 꼬리합.
    등록부 페이지에 그대로 인쇄한다(11/8: 0.113 / 0.296 / 0.570 / 0.839)."""
    n, k0 = int(n_blocks), int(need)
    if n <= 0 or not (0 <= k0 <= n):
        raise ValueError(f"n_blocks={n}, need={k0} 가 유효하지 않습니다")
    rows = []
    for q in qs:
        q = float(q)
        if not (0.0 <= q <= 1.0):
            raise ValueError(f"q 는 [0,1] 이어야 합니다: {q}")
        p = sum(math.comb(n, k) * (q ** k) * ((1.0 - q) ** (n - k)) for k in range(k0, n + 1))
        rows.append({"q": q, "n_blocks": n, "need": k0, "p_at_least": float(p)})
    return pd.DataFrame(rows, columns=["q", "n_blocks", "need", "p_at_least"])


def _blocks_as_pairs(blocks) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    out = []
    for a, b in blocks:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if b <= a:
            raise ValueError(f"블록 길이가 0 이하입니다: {_dstr(a)}~{_dstr(b)}")
        out.append((a, b))
    return out


def admission_test(oos: pd.DataFrame, candidate: str, admitted: tuple[str, ...], blocks, *,
                   frac: float = ENSEMBLE_P3["admit_frac"], block: int = 40, n_boot: int = 4000, seed: int = 0,
                   live_start=None, y_col: str = "y", columns: dict[str, str] | None = None) -> dict:
    """사전 등록 채택 검정(§5.4). 블록 b 마다 Brier(mean(admitted ∪ {c}))_b vs Brier(mean(admitted))_b 를
    **c 가용 행**에서 비교한다. wins = 개선 블록 수, need = ceil(frac · n_avail_blocks),
    no_harm = loss_diff_ci(mean(admitted), mean(admitted∪{c})).hi > 0.

    **증거 구분(계약의 핵심)**:
      * `live_start=None` → 증거 = 관측 기록(2003~2024-08). 통계는 다 계산하지만 verdict 는 절대 "ADMIT"
        가 될 수 없다(`admissible=False`, `verdict_raw` 에 통계상의 판정, `reason='observed_record_cannot_admit'`).
        관측 기록과 홀드아웃은 **기각만** 할 수 있다.
      * `live_start` 지정 → 신선 라이브 블록. live_start ≤ HOLDOUT_START 이면 ValueError,
        블록이 live_start 보다 앞서 시작해도 ValueError, 채점 행에 홀드아웃/관측 행이 섞여도 ValueError.

    반환 {candidate, admitted, evidence, wins, n_blocks, need, per_block, blocks_detail, pooled, ci, no_harm,
          verdict, verdict_raw, admissible, reason, n, first_session, last_session, frac, block, n_boot, seed,
          registry_sha}
    """
    if not isinstance(oos, pd.DataFrame):
        raise TypeError("oos 는 DataFrame 이어야 합니다")
    if candidate not in REGISTRY_NAMES:
        raise ValueError(f"등록부에 없는 후보: {candidate!r}")
    adm = tuple(admitted)
    for a in adm:
        if a not in REGISTRY_NAMES:
            raise ValueError(f"등록부에 없는 admitted 멤버: {a!r}")
    if candidate in adm:
        raise ValueError(f"후보 {candidate!r} 가 이미 admitted 입니다")
    if not adm:
        raise ValueError("admitted 가 비어 있습니다 (seed p2 는 항상 들어간다 — §5.1)")
    if y_col not in oos.columns:
        raise ValueError(f"oos 에 라벨 열 {y_col!r} 이 없습니다")

    idx = _idx(oos.index)
    frame = oos.copy()
    frame.index = idx
    probs = _check_probs(member_probs(frame, tuple(adm) + (candidate,), required=True, columns=columns))
    ys = pd.to_numeric(frame[y_col], errors="coerce").astype(float)

    pairs = _blocks_as_pairs(blocks)
    if not pairs:
        raise ValueError("blocks 가 비어 있습니다")
    bid = assign_blocks(idx, pairs)

    # --- 증거 검증: 홀드아웃·관측 기록은 채택에 못 쓴다 ---
    scored_mask = bid > 0
    scored_idx = idx[scored_mask]
    if live_start is None:
        evidence = "observed"
        bad = scored_idx[scored_idx >= _HOLDOUT]
        if len(bad):
            raise ValueError(f"관측 기록 검정에 홀드아웃 행 {len(bad)}개({_dstr(bad[0])}~)가 들어 있습니다 — "
                             f"홀드아웃은 채택 검정에 절대 쓰지 않습니다 (§2·§5.4)")
    else:
        evidence = "fresh"
        ls = pd.Timestamp(live_start)
        if ls <= _HOLDOUT:
            raise ValueError(f"live_start {_dstr(ls)} ≤ 홀드아웃 시작 {HOLDOUT_START} — 신선 자료가 아닙니다 (§5.4)")
        early = [f"{_dstr(a)}~{_dstr(b)}" for a, b in pairs if a < ls]
        if early:
            raise ValueError(f"라이브 시작 {_dstr(ls)} 이전에서 시작하는 블록이 있습니다: {early[:3]} — "
                             "채택 증거는 라이브 장부의 신선 블록뿐입니다 (§5.4)")
        bad = scored_idx[scored_idx < ls]
        if len(bad):
            raise ValueError(f"채점 행에 라이브 시작 이전 행 {len(bad)}개({_dstr(bad[0])}~)가 있습니다 (§5.4)")

    base = _row_mean(probs, [n for n in REGISTRY_NAMES if n in adm])
    with_c = _row_mean(probs, [n for n in REGISTRY_NAMES if n in adm or n == candidate])
    use = scored_mask & np.isfinite(base.to_numpy()) & np.isfinite(with_c.to_numpy()) & np.isfinite(ys.to_numpy())
    n_used = int(use.sum())
    if n_used == 0:
        raise ValueError("검정에 쓸 유효 행이 없습니다 (라벨·멤버 확률이 모두 유한한 행 0)")

    per_block: list[float] = []
    detail: list[dict] = []
    wins = 0
    for k, (a, b) in enumerate(pairs, start=1):
        m = use & (bid == k)
        n_b = int(m.sum())
        if n_b == 0:
            detail.append({"block": k, "start": _dstr(a), "end": _dstr(b), "n": 0,
                           "dbrier_1e4": None, "win": None, "available": False})
            continue
        yb = ys[m]
        d = brier(base[m], yb) - brier(with_c[m], yb)          # 양수 = 후보가 개선
        win = bool(d > 0)
        wins += int(win)
        per_block.append(float(d * 1e4))
        detail.append({"block": k, "start": _dstr(a), "end": _dstr(b), "n": n_b,
                       "dbrier_1e4": float(d * 1e4), "win": win, "available": True,
                       "brier_base": float(brier(base[m], yb)), "brier_with": float(brier(with_c[m], yb))})

    n_avail = len(per_block)
    if n_avail == 0:
        raise ValueError("후보가 가용한 블록이 하나도 없습니다")
    need = int(math.ceil(float(frac) * n_avail))
    ci = loss_diff_ci(base[use], with_c[use], ys[use], block=int(block), n_boot=int(n_boot), seed=int(seed))
    no_harm = bool(np.isfinite(ci["hi"]) and ci["hi"] > 0.0)
    verdict_raw = "ADMIT" if (wins >= need and no_harm) else "SHADOW"
    min_blocks = int(ENSEMBLE_P3["fresh_blocks_min"])          # §5.4: 완결 신선 블록 ≥ 11개
    admissible = (evidence == "fresh") and (n_avail >= min_blocks)
    verdict = verdict_raw if admissible else "SHADOW"
    reason = (None if admissible
              else "observed_record_cannot_admit" if evidence != "fresh"
              else f"insufficient_fresh_blocks_{n_avail}/{min_blocks}")
    if not admissible and verdict_raw == "ADMIT":
        warnings.warn(
            ("admission_test: 관측 기록에서 통계는 통과했지만 채택은 신선 자료로만 가능합니다 "
             "(verdict=SHADOW, verdict_raw=ADMIT) — §5.4") if evidence != "fresh"
            else (f"admission_test: 통계는 통과했지만 완결 신선 블록 {n_avail}/{min_blocks} 로 "
                  "사전 등록 최소치에 못 미칩니다 (verdict=SHADOW, verdict_raw=ADMIT) — §5.4"),
            RuntimeWarning)

    return {"candidate": candidate, "admitted": tuple(adm), "evidence": evidence,
            "live_start": _dstr(live_start) if live_start is not None else None,
            "wins": int(wins), "n_blocks": int(n_avail), "need": int(need), "frac": float(frac),
            "min_blocks": int(min_blocks),
            "per_block": per_block, "blocks_detail": detail,
            "pooled": float(ci["mean"]), "ci": {"lo": float(ci["lo"]), "hi": float(ci["hi"])}, "no_harm": no_harm,
            "verdict": verdict, "verdict_raw": verdict_raw, "admissible": bool(admissible), "reason": reason,
            "n": n_used, "first_session": _dstr(idx[use][0]), "last_session": _dstr(idx[use][-1]),
            "block": int(block), "n_boot": int(n_boot), "seed": int(seed),
            "registry_sha": registry_sha256()}


def admission_due(n_complete_blocks: int, last_eval_blocks: int | None = None, cfg: dict = ENSEMBLE_P3) -> dict:
    """채택 재검 시점(§5.4): 완결 신선 블록이 `fresh_blocks_min`(11) 이상이면 최초 검정, 그 뒤로는
    `reeval_every_blocks`(4 = 12개월)마다. 반환 {due, n_blocks, min_blocks, next_at, reason}."""
    n = int(n_complete_blocks)
    need_first = int(cfg["fresh_blocks_min"])
    every = int(cfg["reeval_every_blocks"])
    if n < need_first:
        return {"due": False, "n_blocks": n, "min_blocks": need_first, "next_at": need_first,
                "reason": f"완결 신선 블록 {n}/{need_first}"}
    if last_eval_blocks is None:
        return {"due": True, "n_blocks": n, "min_blocks": need_first, "next_at": n + every, "reason": "최초 검정"}
    last = int(last_eval_blocks)
    if last < need_first:
        raise ValueError(f"last_eval_blocks={last} 가 최초 검정 기준 {need_first} 보다 작습니다")
    due = (n - last) >= every
    return {"due": bool(due), "n_blocks": n, "min_blocks": need_first, "next_at": last + every,
            "reason": f"직전 검정 이후 {n - last}블록 (재검 주기 {every})"}


def admission_sequence(oos: pd.DataFrame, blocks, *, statuses: dict[str, str] | None = None,
                       order: Sequence[str] = ENSEMBLE_P3["member_order"], live_start=None, **kwargs) -> dict:
    """고정 순서 **탐욕적 전진** 채택 검정 (§5.4: H → M1, 재배열 금지).

    현재 admitted 집합에 후보를 하나씩 붙여 `admission_test` 를 돌리고, ADMIT 이면 admitted 에 넣고 다음으로
    간다(관측 기록 증거에서는 ADMIT 가 나올 수 없으므로 admitted 는 그대로다). 순서를 바꿔 부르면 ValueError.
    반환 {order, admitted, results{name: result}, skipped{name: 사유}}
    """
    if tuple(order) != tuple(ENSEMBLE_P3["member_order"]):
        raise ValueError(f"채택 검정 순서는 고정입니다: {tuple(ENSEMBLE_P3['member_order'])} (받은 값 {tuple(order)}) — "
                         "재배열 금지 (§5.4)")
    st = check_statuses(statuses if statuses is not None else day1_statuses(), REGISTRY_NAMES)
    admitted = list(production_members(st))
    results: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    for name in order:
        if name in admitted:
            skipped[name] = "already_admitted"
            continue
        if st[name] in ("killed", "candidate_rejected"):
            skipped[name] = st[name]
            continue
        res = admission_test(oos, name, tuple(admitted), blocks, live_start=live_start, **kwargs)
        results[name] = res
        if res["verdict"] == "ADMIT":
            admitted.append(name)
    return {"order": tuple(order), "admitted": tuple(admitted), "results": results, "skipped": skipped}


def next_january(asof, idx=None) -> str:
    """`asof` **다음**의 1월 재적합일(발효일). idx 를 주면 그 인덱스의 1월 첫 거래일, 없으면 달력으로 계산."""
    a = pd.Timestamp(asof)
    if idx is not None:
        ix = _idx(idx)
        jan = ix[ix.month == 1]
        if len(jan):
            firsts = pd.DatetimeIndex(jan.to_series().groupby(jan.year).first().tolist())
            later = firsts[firsts > a]
            if len(later):
                return _dstr(later[0])
    from mrl.calendar_us import next_trading_day, is_trading_day     # noqa: WPS433 - 지연 import
    d = pd.Timestamp(year=a.year + 1, month=1, day=1).date()
    return _dstr(d if is_trading_day(d) else next_trading_day(d))


def apply_admission(model_p3: dict, result: dict, ledger_no: str, effective_refit) -> dict:
    """ADMIT 판정을 `model_p3.json` dict 에 반영(§5.4). **발효는 다음 1월 재적합** — 상태는 'admitted' 로
    바꾸되 `effective_refit` 가 오기 전에는 `effective_statuses()` 가 'shadow' 로 읽는다(연중 변경 금지).

    신선 증거가 아니거나 verdict 가 ADMIT 가 아니면 ValueError — 관측 기록·홀드아웃으로는 채택이 불가능하다.
    파일에 쓰지 않는다(주간 작업 `run_phase3.py` 의 몫). 입력 dict 는 바꾸지 않고 새 dict 를 돌려준다.
    """
    if not isinstance(model_p3, dict):
        raise TypeError("model_p3 는 dict 여야 합니다")
    if not isinstance(result, dict):
        raise TypeError("result 는 admission_test 의 반환 dict 여야 합니다")
    if result.get("evidence") != "fresh":
        raise ValueError(f"채택 불가: 증거가 신선 자료가 아닙니다 (evidence={result.get('evidence')!r}, "
                         f"reason={result.get('reason')!r}) — 관측 기록·홀드아웃은 기각만 할 수 있습니다 (§5.4)")
    if not result.get("admissible", False):
        raise ValueError(f"채택 불가: admissible=False (reason={result.get('reason')!r}) — "
                         "신선 자료이지만 §5.4 사전 등록 조건을 채우지 못했습니다")
    if result.get("verdict") != "ADMIT":
        raise ValueError(f"채택 불가: verdict={result.get('verdict')!r} (ADMIT 아님)")
    min_blocks = int(ENSEMBLE_P3["fresh_blocks_min"])          # 손으로 만든 result 로도 우회할 수 없다
    n_blocks = int(result.get("n_blocks", 0))
    if n_blocks < min_blocks:
        raise ValueError(f"채택 불가: 완결 신선 블록 {n_blocks}/{min_blocks} (§5.4 사전 등록 최소치)")
    name = str(result["candidate"])
    m = member(name)
    if not str(ledger_no).strip():
        raise ValueError("ledger_no(장부 항목 번호)가 필요합니다 — 등록부 변경은 장부에만 남는다")
    eff = pd.Timestamp(effective_refit)
    if eff.month != 1:
        raise ValueError(f"발효일 {_dstr(eff)} 이 1월이 아닙니다 (§5.4: 다음 1월 재적합부터 발효)")
    last = pd.Timestamp(result["last_session"])
    if eff <= last:
        raise ValueError(f"발효일 {_dstr(eff)} 이 검정 마지막 세션 {_dstr(last)} 이후여야 합니다")

    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in model_p3.items()}
    members = {k: dict(v) for k, v in (out.get("members") or {}).items()}
    for n in REGISTRY_NAMES:                                    # 등록부 전체를 항상 채운다(결측 없음)
        mm = member(n)
        members.setdefault(n, {"status": mm.day1_status, "K_s": mm.k_s, "K_u": mm.k_u, "ledger_no": mm.ledger_no})
    entry = dict(members[name])
    entry.update({"status": "admitted", "K_s": m.k_s, "K_u": m.k_u, "ledger_no": str(ledger_no),
                  "effective_refit": _dstr(eff), "admitted_on": str(result["last_session"]),
                  "admission": {k: result[k] for k in ("evidence", "live_start", "wins", "n_blocks", "need",
                                                       "pooled", "ci", "no_harm", "verdict", "n",
                                                       "first_session", "last_session")}})
    members[name] = entry
    out["members"] = members
    hist = list(out.get("mode_history") or [])
    hist.append({"event": "admission", "member": name, "ledger_no": str(ledger_no), "effective_refit": _dstr(eff),
                 "asof": str(result["last_session"]), "wins": int(result["wins"]), "need": int(result["need"])})
    out["mode_history"] = hist
    out["registry_sha"] = registry_sha256()
    return out


# ------------------------------------------------------------------
# 등록부 표 (주간 페이지·카드 줄)
# ------------------------------------------------------------------
def _auc(y: np.ndarray, s: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(s)
    if ok.sum() < 2 or len(np.unique(y[ok])) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score                    # noqa: WPS433 - 지연 import
    return float(roc_auc_score(y[ok], s[ok]))


def registry_table(oos: pd.DataFrame | None = None, statuses: dict[str, str] | None = None,
                   admissions: dict[str, dict] | None = None,
                   live_scores: dict[str, dict] | None = None) -> list[dict]:
    """등록부 표 한 줄 = 멤버 하나. oos(라벨 y·clim 포함)를 주면 관측 기록 점수를 함께 채운다.
    라이브 점수·채택 판정은 호출자가 준 것을 그대로 싣는다(이 모듈은 라이브 채점을 하지 않는다 — track.py)."""
    st = check_statuses(statuses if statuses is not None else day1_statuses(), REGISTRY_NAMES)
    prod = production_members(st)
    probs = None
    y = clim = None
    if oos is not None:
        if not isinstance(oos, pd.DataFrame):
            raise TypeError("oos 는 DataFrame 이어야 합니다")
        probs = member_probs(oos, required=False)
        y = pd.to_numeric(oos["y"], errors="coerce").to_numpy(dtype=float) if "y" in oos.columns else None
        clim = pd.to_numeric(oos["clim"], errors="coerce") if "clim" in oos.columns else None

    rows: list[dict] = []
    for m in REGISTRY:
        row = {"member": m.name, "kind": m.kind, "source": m.source, "K_s": m.k_s, "K_u": m.k_u,
               "status": st[m.name], "ledger_no": m.ledger_no, "in_average": m.name in prod,
               "n": None, "brier": None, "bss_clim": None, "auc": None,
               "admission": (admissions or {}).get(m.name), "live": (live_scores or {}).get(m.name)}
        if probs is not None and m.name in probs.columns and y is not None:
            p = probs[m.name].to_numpy(dtype=float)
            ok = np.isfinite(p) & np.isfinite(y)
            row["n"] = int(ok.sum())
            if ok.any():
                yy = pd.Series(y[ok])
                bs = brier(pd.Series(p[ok]), yy)
                row["brier"] = float(bs)
                row["auc"] = _auc(y[ok], p[ok])
                if clim is not None:
                    c = clim.to_numpy(dtype=float)[ok]
                    if np.isfinite(c).all():
                        bs_ref = float(np.mean((c - y[ok]) ** 2))
                        row["bss_clim"] = float(1.0 - bs / bs_ref) if bs_ref > 0 else None
        rows.append(row)
    return rows
