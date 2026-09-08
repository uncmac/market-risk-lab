# -*- coding: utf-8 -*-
"""장부 Phase 3 확장(mrl/ledger.py — ARCHITECTURE_PHASE3.md §9, schema_version=3) 테스트.

* P3_COLUMNS 가 P2_COLUMNS 뒤에 계약 순서 그대로, 그 뒤에 OUTCOME_COLUMNS_P3 — 기존 열은 재배열도 개명도 없다.
* **schema 2 파일이 제자리에서 승격**된다: 이미 기록된 행의 바이트가 그대로 남고(역사를 다시 쓰지 않는다),
  알 수 없는 추가 열은 맨 뒤에 보존되며, summary() 는 파일을 아예 건드리지 않는다.
* append_today 가 평면 p3_* 키·중첩 dict row["p3"]·별칭을 같은 열로 정규화하고 열거값·확률 범위를 검사한다.
* backfill 이 §9 의 P3 결과 열을 채운다: fwd_maxdd_20 · rv20_realized · ret20_in_vix80/har80 · bh_ret_20 ·
  rule_ret_20(장부 자신의 p3_w_exec 경로 × SPY, 5bp) · 매일의 p3_rule_dd. 전부 손계산·evaluate 와 대조.
* summary()["p3"] 가 track.summary_p3 로 위임되고 JSON 직렬화 가능(NaN 없음)·결정론이다.
임시 디렉터리만 사용한다(results/ 의 실제 장부는 건드리지 않음).
"""
from __future__ import annotations

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

from mrl import evaluate, ledger, track                       # noqa: E402
from mrl.config import ENSEMBLE_P3, P3, V0_SIGNALS            # noqa: E402

# ------------------------------------------------------------------
# 계약 사본 (ARCHITECTURE_PHASE3.md §9 를 그대로 옮긴 것 — 코드가 이 순서에서 벗어나면 실패한다)
# ------------------------------------------------------------------
CONTRACT_P3 = [
    "p3_sigma_ewma", "p3_sigma_har_fc", "p3_sigma_target", "p3_d_max", "p3_w_vol", "p3_state_mult", "p3_w_target",
    "p3_w_exec", "p3_w_reason", "p3_next_check", "p3_sigma_down", "p3_sigma_up", "p3_deploy_sizing",
    "p3_hmm_p_high", "p3_hmm_p20", "p3_hmm_q20", "p3_x_hmm", "p3_p_h", "p3_hmm_theta_id", "p3_hmm_gauge",
    "p3_members", "p3_lo", "p3_hi", "p3_band_src", "p3_disagree_flag",
    "p3_range_vix_lo", "p3_range_vix_hi", "p3_range_har_lo", "p3_range_har_hi", "p3_scen_bin", "p3_dd_from_ath",
    "p3_kill_state", "p3_kill_n_ep", "p3_kill_months", "p3_deploy_mode", "p3_effective_mode", "p3_alarms",
    "p3_registry_sha", "p3_sizing_sha", "p3_input_missing", "p3_run_id",
]
CONTRACT_P3_STRING = ("p3_w_reason", "p3_next_check", "p3_hmm_theta_id", "p3_hmm_gauge", "p3_members", "p3_band_src",
                      "p3_scen_bin", "p3_kill_state", "p3_deploy_mode", "p3_effective_mode", "p3_alarms",
                      "p3_registry_sha", "p3_sizing_sha", "p3_input_missing", "p3_run_id")
CONTRACT_P3_PROB = ("p3_hmm_p_high", "p3_hmm_p20", "p3_hmm_q20", "p3_p_h", "p3_lo", "p3_hi")
CONTRACT_W_REASONS = ("init", "weekly", "escalation", "hold", "input_missing", "info_only")
CONTRACT_KILL_STATES = ("not_started", "not_due", "provisional", "validated", "info_only", "manual_kill")
CONTRACT_OUTCOME_P3 = ["fwd_maxdd_20", "rv20_realized", "ret20_in_vix80", "ret20_in_har80", "rule_ret_20",
                       "bh_ret_20", "p3_rule_dd"]


# ------------------------------------------------------------------
# 픽스처
# ------------------------------------------------------------------
def _row(asof, tone="hold", variant="completed", **kw) -> dict:
    r = {"asof": asof, "variant": variant, "tone": tone, "states_d": {k: "GREEN" for k in V0_SIGNALS},
         "score_d": 0.3, "score_w": 0.1, "score_m": 0.5, "overall_d": "GREEN", "overall_w": "GREEN",
         "overall_m": "GREEN", "spy_close": 512.34, "vix_close": 15.2, "run_id": "test",
         "prob_dd5_20": 0.2, "p2_clim": 0.16, "p2_state": "normal", "p2_deploy_mode": "tones",
         "p2_tone_model": "M1", "p2_p_m1": 0.2}
    r.update(kw)
    return r


def _p3_flat(**kw) -> dict:
    d = {"p3_sigma_ewma": 0.154, "p3_sigma_har_fc": 0.131, "p3_sigma_target": 0.10, "p3_d_max": 0.35,
         "p3_w_vol": 0.65, "p3_state_mult": 1.0, "p3_w_target": 0.65, "p3_w_exec": 0.65, "p3_w_reason": "weekly",
         "p3_next_check": "2026-09-11", "p3_sigma_down": 0.182, "p3_sigma_up": 0.133, "p3_deploy_sizing": True,
         "p3_hmm_p_high": 0.18, "p3_hmm_p20": 0.31, "p3_hmm_q20": 0.42, "p3_x_hmm": -1.52, "p3_p_h": 0.147,
         "p3_hmm_theta_id": "hmm-2024-abcd1234", "p3_hmm_gauge": "low",
         "p3_members": {"p2": 0.2, "M1": 0.19, "H": 0.147}, "p3_lo": 0.12, "p3_hi": 0.31,
         "p3_band_src": "members", "p3_disagree_flag": False,
         "p3_range_vix_lo": 730.5, "p3_range_vix_hi": 810.2, "p3_range_har_lo": 742.1, "p3_range_har_hi": 798.6,
         "p3_scen_bin": "[0.20,0.25)", "p3_dd_from_ath": -0.031,
         "p3_kill_state": "not_due", "p3_kill_n_ep": 1, "p3_kill_months": 4, "p3_deploy_mode": "tones",
         "p3_effective_mode": "tones", "p3_alarms": None, "p3_registry_sha": "reg-abc123",
         "p3_sizing_sha": "siz-def456", "p3_input_missing": None, "p3_run_id": "p3-20260908T000000Z"}
    d.update(kw)
    return d


def _spy(n=70, seed=3) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2026-01-05", periods=n)
    ret = rng.normal(0.0004, 0.006, n)
    ret[30:34] = -0.02                                     # 에피소드 하나
    return pd.Series(100.0 * np.cumprod(1.0 + ret), index=idx)


def _unlock(tmp_path) -> Path:
    """장부 옆 홀드아웃 해제 스텁 — 없으면 2024-09-01 이후 행은 채점되지 않는다(VALIDATION.md §6)."""
    f = Path(tmp_path) / "holdout_unlock.json"
    f.write_text('{"ledger_entry": "2b", "note": "test stub"}', encoding="utf-8")
    return f


# ==================================================================
# 1. 스키마 — 계약 순서·집합
# ==================================================================
def test_schema_v3_columns_append_after_v2_in_contract_order():
    assert ledger.SCHEMA_VERSION == 3
    assert ledger.P3_COLUMNS == CONTRACT_P3
    assert ledger.OUTCOME_COLUMNS_P3 == CONTRACT_OUTCOME_P3
    assert ledger.P3_STRING_COLUMNS == CONTRACT_P3_STRING
    assert ledger.P3_PROB_COLUMNS == CONTRACT_P3_PROB
    assert ledger.P3_W_REASONS == CONTRACT_W_REASONS
    assert ledger.P3_KILL_STATES == CONTRACT_KILL_STATES
    # 덧붙이기만: schema 2 의 전체 열이 schema 3 전체 열의 **접두사**다
    assert ledger.LEDGER_COLUMNS_V2 == ledger.LEDGER_COLUMNS_V1 + ledger.P2_COLUMNS
    assert ledger.LEDGER_COLUMNS == ledger.LEDGER_COLUMNS_V2 + CONTRACT_P3 + CONTRACT_OUTCOME_P3
    assert ledger.LEDGER_COLUMNS[:len(ledger.LEDGER_COLUMNS_V2)] == ledger.LEDGER_COLUMNS_V2
    assert len(set(ledger.LEDGER_COLUMNS)) == len(ledger.LEDGER_COLUMNS)          # 이름 충돌 없음
    assert ledger.ALL_OUTCOME_COLUMNS == ledger.OUTCOME_COLUMNS + CONTRACT_OUTCOME_P3
    # 파생 집합
    assert set(ledger.P3_STRING_COLUMNS) < set(CONTRACT_P3)
    assert set(ledger.P3_NUMERIC_COLUMNS) == set(CONTRACT_P3) - set(CONTRACT_P3_STRING)
    assert set(ledger.P3_PROB_COLUMNS) <= set(ledger.P3_NUMERIC_COLUMNS)
    # 킬 상태·배치 모드는 track 과 같은 어휘여야 한다(두 모듈이 다른 말을 쓰면 카드가 갈라진다)
    assert set(ledger.P3_KILL_STATES) == set(track.KILL_STATES)
    assert ledger.P3_DEPLOY_MODES == ("info_only", "tones")
    assert ledger.P3_MEMBER_ORDER == tuple(ENSEMBLE_P3["members"])


def test_p3_colname_resolves_contract_aliases_and_nested_names():
    assert ledger._p3_colname("p3_w_exec") == "p3_w_exec"
    # 중첩 dict 는 접두사를 떼도 된다(sizing.run() 의 열 이름이 그대로 닿는다)
    assert ledger._p3_colname("w_exec") == "p3_w_exec"
    assert ledger._p3_colname("sigma") == "p3_sigma_ewma"
    assert ledger._p3_colname("mult") == "p3_state_mult"
    assert ledger._p3_colname("reason") == "p3_w_reason"
    assert ledger._p3_colname("nonsense") is None
    # 평면(최상위) 키는 반드시 p3_ 접두사 — 접두사 없는 이름을 최상위에서 낚아채면 v0/P2 키와 충돌한다
    assert all(k.startswith("p3_") for k in ledger.P3_ALIASES)


# ==================================================================
# 2. schema 2 → 3 제자리 승격 (역사를 다시 쓰지 않는다)
# ==================================================================
def _write_schema2_file(path: Path) -> tuple[list[str], list[str]]:
    """schema 2 로 기록된 장부를 손으로 만든다(계약 열 + 알 수 없는 추가 열 2개).
    반환 (헤더 필드, 원본 데이터 줄들)."""
    header = list(ledger.LEDGER_COLUMNS_V2) + ["tone_faithful", "market_status"]
    vals = {c: "" for c in header}
    vals.update({
        "asof": "2026-01-05", "recorded_at_utc": "2026-01-05T22:00:00Z", "variant": "completed",
        **{f"state_{k}": "GREEN" for k in V0_SIGNALS},
        "score_d": "0.676471", "score_w": "0.500000", "score_m": "0.600000",
        "overall_d": "GREEN", "overall_w": "GREEN", "overall_m": "GREEN", "tone": "hold",
        "spy_close": "100.000000", "vix_close": "14.530000", "prob_dd5_20": "0.083596",
        "run_id": "daily-20260105T220000Z", "p2_clim": "0.171395", "p2_p_m1": "0.100548",
        "p2_state": "normal", "p2_deploy_mode": "tones", "p2_tone_model": "M1",
        "p2_model_id": "p2m3-6f786d9f-2024-08-30", "tone_faithful": "hold", "market_status": "open",
    })
    line1 = ",".join(vals[c] for c in header)
    vals2 = dict(vals)
    vals2.update({"asof": "2026-01-06", "recorded_at_utc": "2026-01-06T22:00:00Z",
                  "spy_close": "101.000000", "prob_dd5_20": "0.090000"})
    line2 = ",".join(vals2[c] for c in header)
    path.write_text(",".join(header) + "\n" + line1 + "\n" + line2 + "\n", encoding="utf-8", newline="")
    return header, [line1, line2]


def test_schema2_file_upgrades_in_place_without_rewriting_history(tmp_path):
    path = tmp_path / "track_record.csv"
    old_header, old_lines = _write_schema2_file(path)
    before_bytes = path.read_bytes()
    n_v2 = len(ledger.LEDGER_COLUMNS_V2)
    v2_prefix = [",".join(ln.split(",")[:n_v2]) for ln in old_lines]      # 승격 뒤에도 이 접두사가 바이트 그대로여야 한다

    # (a) 읽기: 없는 P3 열은 NaN 으로 보강, 알 수 없는 추가 열은 맨 뒤에 보존, 기존 값은 그대로
    df = ledger._read(path)
    assert list(df.columns) == list(ledger.LEDGER_COLUMNS) + ["tone_faithful", "market_status"]
    assert df[CONTRACT_P3 + CONTRACT_OUTCOME_P3].isna().all().all()
    assert list(df["asof"]) == ["2026-01-05", "2026-01-06"]
    assert list(df["tone_faithful"]) == ["hold", "hold"]

    # (b) summary(): 파일을 **한 바이트도** 건드리지 않는다(읽기 전용)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ledger.summary(path, trading_days=pd.bdate_range("2026-01-05", periods=2))
    assert path.read_bytes() == before_bytes
    assert s["schema_version"] == 3 and s["n"] == 2
    assert s["p2"]["n_rows"] == 2 and s["p2"]["last_prob"] == pytest.approx(0.09)      # P2 판독 불변
    assert s["p3"]["n_p3_rows"] == 0
    assert any("P3 행이 없습니다" in n for n in s["p3"]["notes"])

    # (c) 새 P3 행을 붙이면 헤더가 승격되고, **구 행의 기존 열 텍스트는 바이트 동일**
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert ledger.append_today(_row("2026-01-07", **_p3_flat()), path) is True
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == list(ledger.LEDGER_COLUMNS) + ["tone_faithful", "market_status"]
    for i, pre in enumerate(v2_prefix):
        assert lines[1 + i].startswith(pre + ","), f"구 행 {i} 의 기존 열이 다시 쓰였습니다"
    # 구 행의 새 열은 전부 빈 값(0 으로도, 오늘 값으로도 채우지 않는다)
    up = pd.read_csv(path)
    old_rows = up[up["asof"].isin(["2026-01-05", "2026-01-06"])]
    assert old_rows[CONTRACT_P3 + CONTRACT_OUTCOME_P3].isna().all().all()
    assert list(up["tone_faithful"].iloc[:2]) == ["hold", "hold"] and bool(up["tone_faithful"].isna().iloc[2])
    assert list(up["market_status"].iloc[:2]) == ["open", "open"]

    # (d) backfill 도 구 행의 이미 채워진 값을 덮어쓰지 않는다
    spy = _spy(40)
    spy.index = pd.DatetimeIndex(["2026-01-05", "2026-01-06", "2026-01-07"]
                                 + [str(d.date()) for d in pd.bdate_range("2026-01-08", periods=37)])
    before_parsed = pd.read_csv(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ledger.backfill(path, spy)
    after = pd.read_csv(path)
    keep = [c for c in ledger.LEDGER_COLUMNS_V2 if c not in ledger.OUTCOME_COLUMNS] + ["tone_faithful", "market_status"]
    pd.testing.assert_frame_equal(after[keep], before_parsed[keep], check_exact=True)


def test_real_repo_ledger_recorded_under_schema2_still_loads_and_summarizes(tmp_path):
    """실제 `results/track_record.csv` 의 Phase 2 시절 행이 그대로 읽히고 요약된다 — 원본은 건드리지 않는다.

    파일은 schema 2 로 기록됐다가 `daily.py` 가 처음 P3 열을 쓸 때 **뒤에 열만 덧붙여** schema 3 이 된다
    (§9: 이관 절차 없음). 그러니 두 상태 모두에서 통과해야 한다 — 검사할 불변량은 '앞쪽 열이 그대로' 이지
    'P3 열이 없다' 가 아니다(후자는 승격 직후 영구히 실패한다)."""
    real = ROOT / "results" / "track_record.csv"
    if not real.exists():
        pytest.skip("results/track_record.csv 없음 (아직 라이브 기록 없음)")
    import hashlib
    sha_before = hashlib.sha256(real.read_bytes()).hexdigest()
    header = real.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header[:len(ledger.LEDGER_COLUMNS_V2)] == list(ledger.LEDGER_COLUMNS_V2)   # 기존 열 이름·순서 불변
    # 승격됐다면 계약 열이 그 순서 그대로 앞쪽에 다 있어야 하고(뒤의 여분 열은 daily 가 붙인 것),
    # 아직이면 P3 열이 하나도 없어야 한다 — 그 사이(일부만 있는 상태)는 없다
    assert (header[:len(ledger.LEDGER_COLUMNS)] == list(ledger.LEDGER_COLUMNS)
            or not (set(CONTRACT_P3) & set(header)))

    df = ledger._read(real)                                                            # 제자리 승격(메모리에서만)
    assert list(df.columns)[:len(ledger.LEDGER_COLUMNS)] == list(ledger.LEDGER_COLUMNS)
    assert df[CONTRACT_P3 + CONTRACT_OUTCOME_P3].isna().all().all()
    n = len(df)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ledger.summary(real)                                                       # 읽기 전용
    assert hashlib.sha256(real.read_bytes()).hexdigest() == sha_before                  # 파일을 다시 쓰지 않았다
    assert s["schema_version"] == 3 and s["n"] == n and s["p2"]["n_rows"] == n
    assert s["p3"]["n_p3_rows"] == 0
    json.dumps(s, ensure_ascii=False, allow_nan=False)

    # 복사본에 P3 행을 붙여도 원본 행의 기존 열 텍스트는 바이트 그대로다
    copy = tmp_path / "track_record.csv"
    copy.write_bytes(real.read_bytes())
    old_lines = copy.read_text(encoding="utf-8").splitlines()[1:]
    n_v2 = len(ledger.LEDGER_COLUMNS_V2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ledger.append_today(_row("2026-09-07", **_p3_flat()), copy)
    new_lines = copy.read_text(encoding="utf-8").splitlines()[1:]
    for old in old_lines:
        pre = ",".join(old.split(",")[:n_v2])
        assert any(ln.startswith(pre + ",") for ln in new_lines), "구 행의 기존 열이 다시 쓰였습니다"


def test_old_phase1_and_phase2_rows_keep_nan_p3_columns(tmp_path):
    path = tmp_path / "t.csv"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ledger.append_today(_row("2026-01-05"), path)                       # P3 없이 기록(Phase 2 호출)
        ledger.append_today(_row("2026-01-06", **_p3_flat()), path)
    df = pd.read_csv(path)
    assert df.loc[0, CONTRACT_P3].isna().all()
    assert not df.loc[1, ["p3_w_exec", "p3_members", "p3_run_id"]].isna().any()


# ==================================================================
# 3. append_today — P3 열 왕복·정규화
# ==================================================================
def test_append_flat_p3_keys_round_trip_and_types(tmp_path):
    path = tmp_path / "t.csv"
    assert ledger.append_today(_row("2026-09-04", **_p3_flat()), path) is True
    df = pd.read_csv(path)
    assert list(df.columns) == ledger.LEDGER_COLUMNS
    r = df.iloc[0]
    assert r["p3_sigma_ewma"] == pytest.approx(0.154) and r["p3_w_exec"] == pytest.approx(0.65)
    assert r["p3_w_reason"] == "weekly" and r["p3_next_check"] == "2026-09-11"
    assert r["p3_kill_state"] == "not_due" and r["p3_kill_n_ep"] == 1 and r["p3_kill_months"] == 4
    assert r["p3_band_src"] == "members" and r["p3_scen_bin"] == "[0.20,0.25)"
    assert r["p3_deploy_sizing"] == 1.0                                     # True → 1.0
    assert pd.isna(r["p3_disagree_flag"])                                   # 플래그가 서지 않은 날은 빈 값
    assert pd.isna(r["p3_alarms"]) and pd.isna(r["p3_input_missing"])       # 사건 없음 = 빈 값
    assert json.loads(r["p3_members"]) == {"p2": 0.2, "M1": 0.19, "H": 0.147}
    assert df[CONTRACT_OUTCOME_P3].isna().all().all()                       # 결과 열은 backfill 소관


def test_append_nested_p3_dict_and_aliases(tmp_path):
    path = tmp_path / "t.csv"
    # sizing.run() 의 열 이름을 그대로 중첩 dict 로 준다
    row = _row("2026-09-04", prob_dd5_20=0.21,
               p3={"sigma": 0.154, "w_vol": 0.65, "mult": 0.5, "w_target": 0.33, "w_exec": 0.35,
                   "reason": "escalation", "deploy_mode": "tones"},
               p3_sigma_har=0.131, p3_w=0.40)                               # 평면 별칭이 중첩보다 우선
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ledger.append_today(row, path)
    r = pd.read_csv(path).iloc[0]
    assert r["p3_sigma_ewma"] == pytest.approx(0.154) and r["p3_state_mult"] == pytest.approx(0.5)
    assert r["p3_w_reason"] == "escalation" and r["p3_sigma_har_fc"] == pytest.approx(0.131)
    assert r["p3_w_exec"] == pytest.approx(0.40)                            # 평면 p3_w 가 중첩 w_exec 를 이긴다


def test_members_json_is_canonical_and_deterministic(tmp_path):
    a = ledger._members_json({"H": 0.147, "p2": 0.2, "M1": 0.19})
    b = ledger._members_json({"p2": 0.2, "M1": 0.19, "H": 0.147})
    assert a == b == '{"p2": 0.2, "M1": 0.19, "H": 0.147}'                  # 등록부 순서 고정(호출자 dict 순서 무관)
    assert ledger._members_json(a) == a                                     # 문자열 재입력도 같은 문자열
    assert ledger._members_json({"H": None, "p2": 0.2}) == '{"p2": 0.2, "H": null}'
    assert ledger._members_json({"p2": 0.2, "Z": 0.3, "A": 0.1}) == '{"p2": 0.2, "A": 0.1, "Z": 0.3}'
    with pytest.raises(ValueError):
        ledger._members_json({"p2": 1.4})                                   # [0,1] 밖
    with pytest.raises(ValueError):
        ledger._members_json({"p2": "높음"})
    with pytest.raises(ValueError):
        ledger._members_json("not json")
    with pytest.raises(ValueError):
        ledger._members_json([0.2, 0.19])


def test_alarms_normalized_to_sorted_comma_string(tmp_path):
    path = tmp_path / "t.csv"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ledger.append_today(_row("2026-09-04", **_p3_flat(p3_alarms=["D6_churn", "D1_p_level", "D6_churn"])), path)
        ledger.append_today(_row("2026-09-07", **_p3_flat(p3_alarms=[])), path)
    df = pd.read_csv(path)
    assert df.loc[0, "p3_alarms"] == "D1_p_level,D6_churn"                   # 이름순·중복 제거(결정론)
    assert pd.isna(df.loc[1, "p3_alarms"])                                   # 빈 목록은 빈 값


def test_disagree_flag_records_only_when_raised(tmp_path):
    path = tmp_path / "t.csv"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ledger.append_today(_row("2026-09-01", **_p3_flat(p3_disagree_flag=True)), path)
        ledger.append_today(_row("2026-09-02", **_p3_flat(p3_disagree_flag=False)), path)
        ledger.append_today(_row("2026-09-03", **_p3_flat(p3_disagree_flag=0)), path)
        ledger.append_today(_row("2026-09-04", **_p3_flat(p3_disagree_flag=np.nan)), path)
    col = pd.read_csv(path)["p3_disagree_flag"]
    assert col.iloc[0] == 1.0 and col.iloc[1:].isna().all()
    # 이 규약이 곧 track.live_panel 의 flag_sessions 계수(빈 값이 아닌 행 수)와 같아야 한다
    assert int(col.astype("string").fillna("").ne("").sum()) == 1


def test_deploy_sizing_false_is_recorded_as_zero_not_missing(tmp_path):
    """0 은 '유지 조건 위반 → 비중 카드 숨김' 이라는 정보다 — 결측으로 지워서는 안 된다(§6.4)."""
    path = tmp_path / "t.csv"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ledger.append_today(_row("2026-09-04", **_p3_flat(p3_deploy_sizing=False)), path)
    assert pd.read_csv(path).loc[0, "p3_deploy_sizing"] == 0.0


def test_boolean_columns_survive_csv_round_trip_strings(tmp_path):
    """CSV 왕복으로 'False' 문자열이 들어와도 뒤집히지 않는다(`bool("False") is True` 함정)."""
    assert ledger._flag_true("False") is False and ledger._flag_true("True") is True
    assert ledger._flag_true("0") is False and ledger._flag_true(np.True_) is True
    with pytest.raises(ValueError):
        ledger._flag_true("아마도")
    path = tmp_path / "t.csv"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ledger.append_today(_row("2026-09-04", **_p3_flat(p3_deploy_sizing="False",
                                                          p3_disagree_flag="True")), path)
    r = pd.read_csv(path).iloc[0]
    assert r["p3_deploy_sizing"] == 0.0 and r["p3_disagree_flag"] == 1.0


def test_next_check_accepts_dates_and_strings(tmp_path):
    path = tmp_path / "t.csv"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ledger.append_today(_row("2026-09-04", **_p3_flat(p3_next_check=pd.Timestamp("2026-09-11"))), path)
    assert pd.read_csv(path).loc[0, "p3_next_check"] == "2026-09-11"


# ==================================================================
# 4. 값 검사 — 조용한 실패 금지
# ==================================================================
@pytest.mark.parametrize("kw", [
    {"p3_w_reason": "rebalance"},                    # 닫힌 열거값
    {"p3_kill_state": "zombie"},
    {"p3_p_h": 1.4}, {"p3_lo": -0.1}, {"p3_hmm_q20": 2.0},
    {"p3_lo": 0.5, "p3_hi": 0.3},                    # 뒤집힌 구간
    {"p3_kill_n_ep": -1}, {"p3_kill_months": 2.5},
    {"p3_sigma_ewma": "높음"},
])
def test_bad_p3_values_raise_value_error(tmp_path, kw):
    with pytest.raises(ValueError):
        ledger.append_today(_row("2026-09-04", **_p3_flat(**kw)), tmp_path / "t.csv")


def test_unknown_enum_values_warn_but_are_recorded(tmp_path):
    path = tmp_path / "t.csv"
    with pytest.warns(UserWarning, match="알 수 없는 p3_deploy_mode"):
        ledger.append_today(_row("2026-09-04", **_p3_flat(p3_deploy_mode="shadow")), path)
    assert pd.read_csv(path).loc[0, "p3_deploy_mode"] == "shadow"
    with pytest.warns(UserWarning, match="알 수 없는 p3_band_src"):
        ledger.append_today(_row("2026-09-07", **_p3_flat(p3_band_src="ensemble")), path)


def test_unknown_p3_keys_warn_and_are_not_recorded(tmp_path):
    path = tmp_path / "t.csv"
    with pytest.warns(UserWarning, match="알 수 없는 P3 키"):
        ledger.append_today(_row("2026-09-04", **_p3_flat(), p3_secret_signal=1.0), path)
    assert "p3_secret_signal" not in pd.read_csv(path).columns
    with pytest.warns(UserWarning, match=r"알 수 없는 P3 키.*p3\.nope"):
        ledger.append_today(_row("2026-09-07", p3={"nope": 1.0, "w_exec": 0.5, "reason": "hold"}), path)


def test_missing_w_exec_without_reason_warns(tmp_path):
    path = tmp_path / "t.csv"
    with pytest.warns(UserWarning, match="p3_w_exec 이 NaN 인데"):
        ledger.append_today(_row("2026-09-04", p3={"sigma": 0.15, "w_vol": 0.65}), path)
    # 사유가 있으면 조용하다 (info_only = §15 단계 0 의 정상 상태)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ledger.append_today(_row("2026-09-07", p3={"sigma": 0.15, "w_vol": 0.65, "reason": "info_only"}), path)
    assert not [c for c in caught if "p3_w_exec" in str(c.message)]


def test_info_only_row_must_not_carry_a_weight(tmp_path):
    """§6.1-4 · §15 단계 0: 정보 제공 전용에서는 비중을 제안하지 않는다 — 기록되면 경고한다."""
    with pytest.warns(UserWarning, match="info_only"):
        ledger.append_today(_row("2026-09-04", **_p3_flat(p3_w_reason="info_only")), tmp_path / "t.csv")


def test_band_not_containing_deployed_probability_warns(tmp_path):
    with pytest.warns(UserWarning, match="등록부 구간이 배포 확률을 포함하지 않음"):
        ledger.append_today(_row("2026-09-04", prob_dd5_20=0.05, **_p3_flat()), tmp_path / "t.csv")


def test_p3_outcome_columns_are_rejected_on_append(tmp_path):
    with pytest.warns(UserWarning, match="결과 열"):
        ledger.append_today(_row("2026-09-04", **_p3_flat(), rule_ret_20=0.05, p3_rule_dd=-0.02),
                            tmp_path / "t.csv")


def test_clean_p3_row_appends_without_warnings(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ledger.append_today(_row("2026-09-04", **_p3_flat()), tmp_path / "t.csv")
    assert [str(c.message) for c in caught] == []


def test_append_is_deterministic(tmp_path):
    """같은 입력 → 같은 바이트(recorded_at_utc 는 고정해 시각만 제외한다)."""
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    row = _row("2026-09-04", recorded_at_utc="2026-09-04T22:00:00Z", **_p3_flat())
    ledger.append_today(dict(row), a)
    ledger.append_today(dict(row), b)
    assert a.read_bytes() == b.read_bytes()


# ==================================================================
# 5. backfill — §9 의 P3 결과 열 (손계산 · evaluate 와 대조)
# ==================================================================
def _build_ledger(path: Path, spy: pd.Series, n_rows: int, weights) -> None:
    idx = spy.index
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n_rows):
            c = float(spy.iloc[i])
            ledger.append_today(_row(idx[i], spy_close=c, **_p3_flat(
                p3_w_exec=float(weights[i]), p3_w_reason="init" if i == 0 else "hold",
                p3_range_vix_lo=c * 0.95, p3_range_vix_hi=c * 1.05,
                p3_range_har_lo=c * 0.99, p3_range_har_hi=c * 1.01)), path)


def test_backfill_p3_outcomes_match_hand_calculation(tmp_path):
    path = tmp_path / "t.csv"
    spy = _spy(70)
    n_rows = 45
    w = np.where(np.arange(n_rows) < 10, 1.0, 0.5)
    _build_ledger(path, spy, n_rows, w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = ledger.backfill(path, spy).set_index("asof")

    c = spy.to_numpy()
    ln = np.log(c)
    for pos in (0, 7, 23):
        r = df.loc[spy.index[pos].strftime("%Y-%m-%d")]
        assert r["fwd_maxdd_20"] == pytest.approx(round(c[pos + 1:pos + 21].min() / c[pos] - 1.0, 6), abs=1e-6)
        lr = np.diff(ln[pos:pos + 21])
        assert r["rv20_realized"] == pytest.approx(round(float(lr.std(ddof=1) * math.sqrt(252)), 6), abs=1e-6)
        assert r["bh_ret_20"] == pytest.approx(round(c[pos + 20] / c[pos] - 1.0, 6), abs=1e-6)
        # 20일 보유 수익은 fwd_ret_20 과 같은 수여야 한다(같은 정의, 다른 열)
        assert r["bh_ret_20"] == pytest.approx(r["fwd_ret_20"], abs=1e-6)
        # 기록한 80% 밴드 안인가 = close[t+20] ∈ [lo, hi]
        assert r["ret20_in_vix80"] == float(c[pos] * 0.95 <= c[pos + 20] <= c[pos] * 1.05)
        assert r["ret20_in_har80"] == float(c[pos] * 0.99 <= c[pos + 20] <= c[pos] * 1.01)
    # 밴드가 실제로 두 값을 다 만든다(상수 열이면 테스트가 아무것도 증명하지 못한다)
    assert set(df["ret20_in_har80"].dropna().unique()) == {0.0, 1.0}


def test_backfill_rule_return_uses_ledger_own_weight_path(tmp_path):
    """rule_ret_20·p3_rule_dd 는 **장부 자신의** p3_w_exec 경로에서 나오며 evaluate.allocation_from_weights 와 같은 규약."""
    path = tmp_path / "t.csv"
    spy = _spy(70)
    n_rows = 45
    w = np.where(np.arange(n_rows) < 10, 1.0, 0.5)
    _build_ledger(path, spy, n_rows, w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = ledger.backfill(path, spy).set_index("asof")

    ws = pd.Series(w, index=spy.index[:n_rows])
    ref = evaluate.allocation_from_weights(ws, spy.iloc[:n_rows], cost_bps=P3["cost_bps"])
    eq = pd.concat([pd.Series([1.0], index=[spy.index[0]]), ref["series"]["equity"]])   # 첫 세션 자본 = 1.0
    dd = eq / eq.cummax() - 1.0
    for pos in (0, 5, 12, 24):
        key = spy.index[pos].strftime("%Y-%m-%d")
        assert df.loc[key, "p3_rule_dd"] == pytest.approx(round(float(dd.iloc[pos]), 6), abs=1e-6)
        if pos + 20 < n_rows:
            want = float(eq.iloc[pos + 20] / eq.iloc[pos] - 1.0)
            assert df.loc[key, "rule_ret_20"] == pytest.approx(round(want, 6), abs=1e-6)
    # 비중을 줄인 뒤 규칙 낙폭은 보유보다 얕다(급락 구간이 자료에 있다)
    assert df["p3_rule_dd"].min() > (spy / spy.cummax() - 1.0).iloc[:n_rows].min()
    # p3_rule_dd 는 미래 자료가 필요 없다 → 마지막 행까지 매일 채워진다(rule_ret_20 은 20세션 뒤에만)
    assert df["p3_rule_dd"].notna().sum() == n_rows
    assert df["rule_ret_20"].notna().sum() == n_rows - 20
    assert df["p3_rule_dd"].iloc[0] == 0.0


def test_backfill_p3_is_point_in_time_and_write_once(tmp_path):
    path = tmp_path / "t.csv"
    spy = _spy(70)
    _build_ledger(path, spy, 45, np.full(45, 0.6))
    short = spy.iloc[:30]                                       # 완성 세션이 30개뿐인 상태로 먼저 채운다
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d1 = ledger.backfill(path, short).set_index("asof")
    assert d1["fwd_maxdd_20"].notna().sum() == 30 - 20          # 20세션이 지난 행만
    assert d1.loc[spy.index[25].strftime("%Y-%m-%d"), ["fwd_maxdd_20", "rule_ret_20"]].isna().all()
    first = d1.loc[spy.index[0].strftime("%Y-%m-%d"), CONTRACT_OUTCOME_P3].copy()

    # 자료를 늘려 다시 돌리면 새 행만 채워지고 이미 채운 값은 그대로(기록 안정성)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d2 = ledger.backfill(path, spy).set_index("asof")
    pd.testing.assert_series_equal(d2.loc[spy.index[0].strftime("%Y-%m-%d"), CONTRACT_OUTCOME_P3].astype(float),
                                   first.astype(float), check_names=False)
    assert d2["fwd_maxdd_20"].notna().sum() == 45


def test_backfill_skips_scoring_when_band_was_not_recorded(tmp_path):
    """밴드를 기록하지 않은 날은 0 으로 채우지 않는다(조용히 '빗나감' 으로 세면 포함률이 거짓이 된다)."""
    path = tmp_path / "t.csv"
    spy = _spy(70)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(30):
            kw = _p3_flat(p3_w_exec=0.6, p3_w_reason="hold")
            kw.update({"p3_range_vix_lo": None, "p3_range_vix_hi": None})
            ledger.append_today(_row(spy.index[i], **kw), path)
        df = ledger.backfill(path, spy)
    assert df["ret20_in_vix80"].isna().all()
    assert df["ret20_in_har80"].notna().sum() == 30           # HAR 밴드는 기록했으므로 채점된다


def test_backfill_variants_do_not_share_a_weight_path(tmp_path):
    path = tmp_path / "t.csv"
    spy = _spy(70)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(30):
            ledger.append_today(_row(spy.index[i], variant="completed",
                                     **_p3_flat(p3_w_exec=1.0, p3_w_reason="hold")), path)
            ledger.append_today(_row(spy.index[i], variant="faithful",
                                     **_p3_flat(p3_w_exec=0.25, p3_w_reason="hold")), path)
        df = ledger.backfill(path, spy)
    comp = df[df["variant"] == "completed"].set_index("asof")
    faith = df[df["variant"] == "faithful"].set_index("asof")
    key = spy.index[0].strftime("%Y-%m-%d")
    # 같은 날 두 변형의 규칙 수익이 비중 비율만큼 다르다 — 섞이지 않았다는 증거
    assert comp.loc[key, "rule_ret_20"] != pytest.approx(faith.loc[key, "rule_ret_20"], abs=1e-6)
    assert abs(faith.loc[key, "rule_ret_20"]) < abs(comp.loc[key, "rule_ret_20"])


def test_backfill_fills_gaps_with_hold_and_says_so(tmp_path):
    """장부에 빠진 거래일은 규칙의 'hold'(직전 비중 유지)로 메우고 **경고한다** — 조용히 메우지 않는다."""
    path = tmp_path / "t.csv"
    spy = _spy(70)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in list(range(0, 10)) + list(range(12, 40)):          # 10, 11 결측
            ledger.append_today(_row(spy.index[i], **_p3_flat(p3_w_exec=0.6, p3_w_reason="hold")), path)
    with pytest.warns(UserWarning, match="직전 w_exec 을 유지"):
        ledger.backfill(path, spy)


def test_backfill_without_p3_rows_leaves_p3_outcomes_empty(tmp_path):
    path = tmp_path / "t.csv"
    spy = _spy(70)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(30):
            ledger.append_today(_row(spy.index[i]), path)            # P3 열 없음
        df = ledger.backfill(path, spy)
    assert df[["rule_ret_20", "p3_rule_dd", "ret20_in_vix80", "ret20_in_har80"]].isna().all().all()
    assert df["fwd_maxdd_20"].notna().sum() == 30                   # 종가만으로 되는 열은 그래도 채워진다
    assert df["rv20_realized"].notna().sum() == 30 and df["bh_ret_20"].notna().sum() == 30


# ==================================================================
# 6. summary()["p3"] — track.summary_p3 위임
# ==================================================================
def test_summary_p3_delegates_to_track_and_is_json_safe(tmp_path):
    path = tmp_path / "t.csv"
    _unlock(tmp_path)
    spy = _spy(70)
    n_rows = 45
    _build_ledger(path, spy, n_rows, np.where(np.arange(n_rows) < 10, 1.0, 0.5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ledger.backfill(path, spy)
        s = ledger.summary(path, trading_days=spy.index, spy_close=spy)
    p3 = s["p3"]
    for k in ("schema_version", "n_rows", "n_scored", "n_eff", "live_start", "months_elapsed", "effective_mode",
              "deploy_sizing", "sizing", "members", "disagreement", "kill", "alarms", "replay_mismatch_count",
              "coverage", "notes"):
        assert k in p3, k
    assert p3["schema_version"] == 3 and p3["n_rows"] == n_rows and p3["n_p3_rows"] == n_rows
    assert p3["live_start"] == spy.index[0].strftime("%Y-%m-%d")
    assert p3["sizing"]["avg_w"] == pytest.approx(np.mean(np.where(np.arange(n_rows) < 10, 1.0, 0.5)))
    assert p3["sizing"]["rule_dd"] is not None and p3["kill"]["state"] in ledger.P3_KILL_STATES
    assert set(p3["coverage"]) == {"vix80", "har80"}
    # p2 가 tones · p3 가 tones · 킬 기록 없음 → 유효 모드 tones (§8.3 의 AND)
    assert p3["effective_mode"] in ("tones", "info_only")
    # JSON 엄격 직렬화(NaN·Infinity 없음)
    txt = json.dumps(s, ensure_ascii=False, allow_nan=False)
    assert "NaN" not in txt and "Infinity" not in txt
    # 결정론: 두 번 불러도 같은 결과
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s2 = ledger.summary(path, trading_days=spy.index, spy_close=spy)
    assert json.dumps(s2["p3"], ensure_ascii=False, sort_keys=True) == json.dumps(p3, ensure_ascii=False, sort_keys=True)


def test_summary_p3_effective_mode_follows_p2_info_only(tmp_path):
    """§15 단계 0: p2 가 info_only 면 유효 모드도 info_only — 비중·상태 카드가 뜨면 안 된다."""
    path = tmp_path / "t.csv"
    _unlock(tmp_path)
    spy = _spy(40)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(20):
            ledger.append_today(_row(spy.index[i], p2_deploy_mode="info_only", p2_tone_model=None,
                                     p3={"sigma": 0.15, "w_vol": 0.65, "reason": "info_only",
                                         "deploy_mode": "info_only"}), path)
        s = ledger.summary(path, trading_days=spy.index, spy_close=spy)
    p3 = s["p3"]
    assert p3["effective_mode"] == "info_only"
    # p2 가 info_only 인 동안은 라이브가 시작되지 않는다(§8.1) → 채점·킬룰 시계도 시작 전
    assert p3["live_start"] is None and p3["n_scored"] == 0
    assert p3["kill"]["state"] == "not_started"


def test_summary_p3_disagreement_counts_only_flagged_sessions(tmp_path):
    path = tmp_path / "t.csv"
    _unlock(tmp_path)
    spy = _spy(40)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(20):
            ledger.append_today(_row(spy.index[i], **_p3_flat(p3_lo=0.10, p3_hi=0.34,
                                                              p3_disagree_flag=(i < 6))), path)
        s = ledger.summary(path, trading_days=spy.index, spy_close=spy)
    dis = s["p3"]["disagreement"]
    assert dis["flag_sessions"] == 6
    assert dis["median_width"] == pytest.approx(0.24, abs=1e-9)


def test_summary_p3_empty_ledger_and_missing_file(tmp_path):
    empty = ledger.summary(tmp_path / "none.csv")
    assert empty["p3"]["n_rows"] == 0 and empty["p3"]["kill"] == {}
    assert empty["schema_version"] == 3
    json.dumps(empty, allow_nan=False)


def test_summary_p3_failure_is_loud_not_silent(tmp_path, monkeypatch):
    """track.summary_p3 가 터져도 summary() 는 나오되 **사유가 크게 남는다**(조용한 실패 금지)."""
    path = tmp_path / "t.csv"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ledger.append_today(_row("2026-09-04", **_p3_flat()), path)

    def boom(*a, **k):
        raise RuntimeError("합성 고장")

    monkeypatch.setattr(track, "summary_p3", boom)
    s = ledger.summary(path)
    # summary() 는 p3 계산의 경고를 삼키지 않고 자기 warnings 목록에 'p3: ...' 로 올린다
    assert s["p3"]["n_rows"] == 0
    assert any("합성 고장" in n for n in s["p3"]["notes"])
    assert any(w.startswith("p3: ") and "합성 고장" in w for w in s["warnings"])
    assert s["p2"]["n_rows"] == 1                     # 나머지 요약은 그대로 나온다


def test_summary_p2_kill_rule_block_survives_alongside_p3(tmp_path):
    """§9: p2.kill_rule('둘 다' 문자 그대로 판독)은 남기되 카드·페이지는 p3.kill 만 읽는다."""
    path = tmp_path / "t.csv"
    _unlock(tmp_path)
    spy = _spy(70)
    _build_ledger(path, spy, 45, np.full(45, 0.6))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ledger.backfill(path, spy)
        s = ledger.summary(path, trading_days=spy.index, spy_close=spy)
    assert s["p2"]["kill_rule"]["episodes_required"] == 8 and s["p2"]["kill_rule"]["months_required"] == 36
    assert s["p2"]["kill_rule_due"] is False
    assert s["p3"]["kill"]["countdown"]["months"].endswith("/36")
    assert "rule" in s["p3"]["kill"] and "1차 36개월" in s["p3"]["kill"]["rule"]
