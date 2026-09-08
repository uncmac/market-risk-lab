# -*- coding: utf-8 -*-
"""트랙레코드·드리프트 경보·킬룰(mrl/track.py — ARCHITECTURE_PHASE3.md §8) 테스트.

* `episodes_realized`: 돌파일 기준 · 시작 시점에 이미 진행 중이던 구간 제외 · 저점 20세션 확정 · 점 원칙.
* `months_elapsed`(달력 월 Period 차), `score`/`bss_block_ci` 부호와 `ledger._bss_block_ci` 와의 **비트 동일성**.
* **킬룰 단위 테스트(합성 장부)**: not_due(30개월) / provisional(36개월·BSS > 0) / info_only(36개월·BSS ≤ 0) +
  `kill_apply` 멱등 / validated(8회·CI 하한 > 0) / 60개월 상한(7회) / 2차 뒤 12개월 재평가 리듬(11개월엔 미실행) /
  sticky / 멤버별 판정 / 수동 킬 파일 / 복귀는 함수로 불가.
* **홀드아웃 게이트**: `results/holdout_unlock.json` 이 없으면 HOLDOUT_START 이후 세션은 채점되지 않는다
  (`mrl/ledger.py` 와 같은 의미론 — 그 파일은 건드리지 않는다).
* **재현 테스트**: 실제 2003~2024-08 기록에서 `kill_replay` 의 오기각·검증·널 통과 비중이 사전 등록 구간 안.
* **경보**: 각 코드가 자기 조건에서만 발화하고 정상 장부에서는 무발화. D7 은 exit 1 행동을 단다.
* **지평 문구**: n_eff < 6 에 BSS 숫자 없음, 판정일 전 판정어 없음 — 코드(may_show/live_panel)가 강제.

실제 `results/` 산출물은 절대 건드리지 않는다(모든 경로는 tmp_path).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrl import track                                                      # noqa: E402
from mrl.config import DATA_DIR, HOLDOUT_START, KILL_P3, P3_DRIFT, RESULTS_DIR   # noqa: E402

FAST = dict(KILL_P3, n_boot=800)          # 부트스트랩만 줄인 사본 — 판정 논리는 그대로


# ------------------------------------------------------------------
# 합성 장부·종가 픽스처
# ------------------------------------------------------------------
def sessions(start: str, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def rising_close(dates, step: float = 0.0008) -> pd.Series:
    """단조 상승 종가 — ≥5% 에피소드 0회."""
    return pd.Series(100.0 * np.exp(np.arange(len(dates)) * step), index=dates)


def episode_close(dates, n_ep: int, dd: float = 0.07, cycle: int = 60, drift: float = 0.0008) -> pd.Series:
    """정확히 `n_ep` 개의 ≥5% 돌파를 만드는 톱니 종가.
    한 주기: 20세션 신고점 갱신 → 1세션 −7% 돌파 → 39세션 회복(다음 주기 초에 ATH 재돌파)."""
    n = len(dates)
    step = np.empty(n, dtype=float)
    for i in range(n):
        j = i % cycle
        step[i] = (math.log(1 - dd) if (j == 20 and (i // cycle) < n_ep)
                   else (-math.log(1 - dd) / (cycle - 1) + drift if (i // cycle) < n_ep else drift))
    return pd.Series(100.0 * np.exp(np.cumsum(step)), index=dates)


def y_pattern(n: int, every: int = 6) -> np.ndarray:
    """결정적 라벨(기저율 ≈ 1/every)."""
    y = np.zeros(n, dtype=float)
    y[::every] = 1.0
    return y


def skilled_p(y: np.ndarray, good: bool = True) -> np.ndarray:
    """BSS_clim 이 크게 양(+0.74)/음(−1.96)이 되는 결정적 확률."""
    return (0.05 + 0.50 * y) if good else (0.55 - 0.50 * y)


def make_ledger(dates, *, good: bool = True, clim: float = 0.16, mode: str = "tones",
                m1_good: bool | None = None, **extra) -> pd.DataFrame:
    """킬룰·채점용 최소 장부(계약 열 이름 그대로)."""
    n = len(dates)
    y = y_pattern(n)
    df = pd.DataFrame({
        "asof": [str(d.date()) for d in dates],
        "variant": "completed",
        "prob_dd5_20": skilled_p(y, good),
        "p2_clim": clim,
        "y_dd5_20": y,
        "p2_deploy_mode": mode,
        "p2_state": "normal",
    })
    if m1_good is not None:
        df["p2_p_m1"] = skilled_p(y, m1_good)
    for k, v in extra.items():
        df[k] = v
    return df


def unlock(tmp_path) -> Path:
    """홀드아웃 해제 스텁 — 이 파일이 있어야 HOLDOUT_START 이후 세션이 채점된다."""
    p = Path(tmp_path) / "holdout_unlock.json"
    p.write_text('{"ledger_entry": "2b", "note": "test stub"}', encoding="utf-8")
    return p


def no_unlock(tmp_path) -> Path:
    return Path(tmp_path) / "holdout_unlock.json"          # 존재하지 않는 경로 = 잠금


# ==================================================================
# 1. 기본 통계
# ==================================================================
def test_bss_block_ci_matches_ledger_bitwise():
    """`ledger._bss_block_ci` 의 공개 승격판 — 같은 블록 추출·같은 seed 라 **비트 동일**해야 한다."""
    from mrl.ledger import _bss_block_ci as ledger_ci
    rng = np.random.default_rng(7)
    for n in (60, 300, 1013):
        lp, lr = rng.random(n) * 0.3, rng.random(n) * 0.4
        assert track.bss_block_ci(lp, lr) == ledger_ci(lp, lr)


def test_bss_block_ci_sign_and_degenerate():
    y = y_pattern(600)
    good, bad, clim = skilled_p(y, True), skilled_p(y, False), np.full(600, 0.16)
    lr = (clim - y) ** 2
    lo_g, hi_g = track.bss_block_ci((good - y) ** 2, lr)
    lo_b, hi_b = track.bss_block_ci((bad - y) ** 2, lr)
    assert lo_g > 0 and hi_g > lo_g                    # 진짜 skill → 하한 > 0
    assert hi_b < 0                                    # 반대로 맞추는 모델 → 상한 < 0
    assert track._ci_label(lo_g, hi_g) == "validated"
    assert track._ci_label(lo_b, hi_b) == "rejected"
    assert track._ci_label(-0.1, 0.2) == "undecided"
    assert all(math.isnan(v) for v in track.bss_block_ci([], []))


def test_months_elapsed_calendar_period():
    assert track.months_elapsed("2026-01-31", "2029-01-02") == 36
    assert track.months_elapsed("2026-10-01", "2029-09-30") == 35
    assert track.months_elapsed("2026-10-01", "2031-10-01") == 60
    assert track.months_elapsed(None, "2029-01-02") == 0


def test_score_hand_calculation():
    df = pd.DataFrame({"asof": [f"2016-01-0{i}" for i in (4, 5, 6, 7, 8)],
                       "prob_dd5_20": [0.1, 0.1, 0.1, 0.1, 0.9],
                       "p2_clim": 0.2, "y_dd5_20": [0.0, 0.0, 0.0, 0.0, 1.0]})
    s = track.score(df, "prob_dd5_20", {"clim": "p2_clim"}, FAST, unlock_path=None)
    assert s["n"] == 5 and s["n_eff"] == pytest.approx(0.25)
    assert s["brier"] == pytest.approx(0.01)
    assert s["bss_clim"] == pytest.approx(1.0 - 0.01 / 0.16)
    assert s["ci_bss_clim"] is None and s["ci_label"] is None      # n < block 40 → 구간 생략
    assert s["base_rate"] == pytest.approx(0.2)


def test_score_skips_unscored_rows_and_reports_missing_ref():
    notes: list[str] = []
    df = pd.DataFrame({"asof": ["2016-01-04", "2016-01-05"], "prob_dd5_20": [0.2, np.nan],
                       "y_dd5_20": [1.0, 1.0]})
    s = track.score(df, "prob_dd5_20", {"clim": "p2_clim"}, FAST, unlock_path=None, notes=notes)
    assert s["n"] == 1 and s["bss_clim"] is None
    assert any("p2_clim" in n for n in notes)          # 조용한 실패 금지


# ==================================================================
# 2. 에피소드 (돌파일·확정·점 원칙)
# ==================================================================
def test_episodes_realized_counts_breaches_not_troughs():
    d = sessions("2016-01-04", 800)
    c = episode_close(d, 8)
    n, bd = track.episodes_realized(c, d[0], d[-1])
    assert n == 8 and len(bd) == 8
    assert bd == sorted(bd)
    # 돌파일은 종가가 ATH 대비 −5% 이하로 처음 내려간 날이다(저점이 아니다)
    for b in bd:
        ts = pd.Timestamp(b)
        prior_max = c[c.index <= ts].iloc[:-1].max()
        assert c.loc[ts] / prior_max - 1.0 <= -0.05


def test_episodes_realized_excludes_in_progress_at_start():
    d = sessions("2016-01-04", 800)
    c = episode_close(d, 8)
    all_bd = track.episodes_realized(c, d[0], d[-1])[1]
    first = pd.Timestamp(all_bd[0])
    # 첫 돌파 다음 날부터 세면 그 구간(이미 진행 중)은 빠진다
    n2, bd2 = track.episodes_realized(c, first + pd.Timedelta(days=1), d[-1])
    assert n2 == 7 and all_bd[0] not in bd2
    # 돌파일 당일부터 세면 포함된다(경계는 닫힌 쪽)
    assert track.episodes_realized(c, first, d[-1])[0] == 8


def test_episodes_realized_requires_20_session_confirmation():
    d = sessions("2016-01-04", 800)
    c = episode_close(d, 8)
    bd = track.episodes_realized(c, d[0], d[-1])[1]
    last = pd.Timestamp(bd[-1])
    pos = int(c.index.get_loc(last))
    assert track.episodes_realized(c, d[0], c.index[pos + 19])[0] == 7    # 19세션: 아직 미확정
    assert track.episodes_realized(c, d[0], c.index[pos + 20])[0] == 8    # 20세션: 확정


def test_episodes_realized_is_point_in_time():
    d = sessions("2016-01-04", 800)
    c = episode_close(d, 8)
    cut = c.index[500]
    a = track.episodes_realized(c, d[0], cut)
    b = track.episodes_realized(c.iloc[:501], d[0], cut)
    assert a == b                                       # 미래 행을 붙여도 과거 판정 불변


def test_episodes_realized_zero_on_monotone_close():
    d = sessions("2016-01-04", 400)
    assert track.episodes_realized(rising_close(d), d[0], d[-1]) == (0, [])


# ==================================================================
# 3. 홀드아웃 게이트 (계약: unlock 파일 없으면 라이브 채점 금지)
# ==================================================================
def test_holdout_gate_blocks_rows_without_unlock(tmp_path):
    d = pd.DatetimeIndex(["2024-08-29", "2024-08-30", "2024-09-03", "2024-09-04"])
    df = pd.DataFrame({"asof": [str(x.date()) for x in d], "v": 1.0})
    notes: list[str] = []
    out, locked = track.holdout_gate(df, no_unlock(tmp_path), notes)
    assert locked and len(out) == 2 and notes and HOLDOUT_START in notes[0]
    out2, locked2 = track.holdout_gate(df, unlock(tmp_path))
    assert (not locked2) and len(out2) == 4
    out3, locked3 = track.holdout_gate(df, None)         # None = 게이트 해제(테스트·사후 분석)
    assert (not locked3) and len(out3) == 4


def test_holdout_gate_requires_dates():
    with pytest.raises(ValueError):
        track.holdout_gate(pd.DataFrame({"v": [1.0]}), Path("nope.json"))


def test_score_does_not_score_holdout_without_unlock(tmp_path):
    d = sessions("2024-08-01", 200)                     # 대부분 HOLDOUT_START 이후
    led = make_ledger(d)
    locked = track.score(led, "prob_dd5_20", None, FAST, unlock_path=no_unlock(tmp_path))
    open_ = track.score(led, "prob_dd5_20", None, FAST, unlock_path=unlock(tmp_path))
    assert locked["holdout_locked"] and locked["n"] < open_["n"] == 200
    assert locked["n"] == int((pd.to_datetime(led["asof"]) < pd.Timestamp(HOLDOUT_START)).sum())


def test_kill_status_holdout_locked_is_not_started(tmp_path):
    d = sessions("2024-09-03", 800)                     # 전부 홀드아웃 안
    led = make_ledger(d)
    ks = track.kill_status(led, rising_close(d), d[-1], cfg=FAST,
                           unlock_path=no_unlock(tmp_path), manual_path=tmp_path / "none.json")
    assert ks["state"] == "not_started" and ks["months"] == 0
    assert any("홀드아웃" in n for n in ks["notes"])
    ks2 = track.kill_status(led, rising_close(d), d[-1], cfg=FAST,
                            unlock_path=unlock(tmp_path), manual_path=tmp_path / "none.json")
    assert ks2["state"] != "not_started" and ks2["n"] == 800


# ==================================================================
# 4. 킬룰 분기 (§8.3)
# ==================================================================
def _ks(led, close, asof, prev=None, tmp=None, **kw):
    return track.kill_status(led, close, asof, prev=prev, cfg=FAST,
                             unlock_path=None, manual_path=(Path(tmp) / "no_manual.json") if tmp else None,
                             **kw)


def test_kill_not_started_before_tones():
    d = sessions("2016-01-04", 900)
    led = make_ledger(d, mode="info_only")              # Phase 2 가 아직 info_only → 라이브 아님
    ks = _ks(led, rising_close(d), d[-1])
    assert ks["state"] == "not_started" and ks["live_start"] is None
    assert any("라이브 시작 전" in n for n in ks["notes"])


def test_kill_not_due_at_30_months():
    d = sessions("2016-01-04", 653)                     # ≈30개월
    led = make_ledger(d, good=True)
    ks = _ks(led, rising_close(d), d[-1])
    assert ks["months"] == 30 and ks["state"] == "not_due"
    assert not ks["stage1_due"] and not ks["evaluated_now"] and not ks["killed"]
    assert ks["countdown"] == {"episodes": "0/8", "months": "30/36", "cap": "30/60"}


def test_kill_provisional_at_36_months_positive_bss():
    d = sessions("2016-01-04", 790)                     # ≈36개월
    led = make_ledger(d, good=True)
    ks = _ks(led, rising_close(d), d[-1])
    assert ks["months"] >= 36 and ks["stage1_due"] and not ks["stage2_due"]
    assert ks["evaluated_now"] and ks["stage1_done"] and not ks["stage2_done"]
    assert ks["bss_clim"] > 0 and ks["state"] == "provisional" and not ks["killed"]


def test_kill_info_only_at_36_months_and_kill_apply_is_idempotent(tmp_path):
    d = sessions("2016-01-04", 790)
    led = make_ledger(d, good=False)                    # BSS ≤ 0
    ks = _ks(led, rising_close(d), d[-1])
    assert ks["stage1_due"] and ks["bss_clim"] < 0
    assert ks["killed"] and ks["state"] == "info_only" and ks["stage1_done"]

    model, record = tmp_path / "model_p3.json", tmp_path / "kill_record.json"
    model.write_text(json.dumps({"schema_version": 3, "deploy_mode": "tones"}), encoding="utf-8")
    assert track.kill_apply(ks, model, record) is True
    first = record.read_text(encoding="utf-8")
    m1 = json.loads(model.read_text(encoding="utf-8"))
    assert m1["deploy_mode"] == "info_only" and len(m1["mode_history"]) == 1
    assert m1["kill"]["state"] == "info_only" and m1["kill"]["killed"] is True

    assert track.kill_apply(ks, model, record) is False          # 멱등
    m2 = json.loads(model.read_text(encoding="utf-8"))
    assert m2["deploy_mode"] == "info_only" and len(m2["mode_history"]) == 1
    assert record.read_text(encoding="utf-8") == first           # 기록은 다시 쓰지 않는다
    rec = json.loads(first)
    for k in ("asof", "n", "n_eff", "bss", "ci", "ci_label", "n_ep", "breach_dates", "months", "ledger_entry"):
        assert k in rec


def test_kill_apply_never_touches_model_p2(tmp_path):
    d = sessions("2016-01-04", 790)
    ks = _ks(make_ledger(d, good=False), rising_close(d), d[-1])
    p2 = tmp_path / "model_p2.json"
    p2.write_text('{"deploy_mode": "tones"}', encoding="utf-8")
    p3 = tmp_path / "model_p3.json"
    p3.write_text('{"schema_version": 3, "deploy_mode": "tones"}', encoding="utf-8")
    track.kill_apply(ks, p3, tmp_path / "kill_record.json")
    assert json.loads(p2.read_text(encoding="utf-8"))["deploy_mode"] == "tones"
    assert json.loads(p3.read_text(encoding="utf-8"))["deploy_mode"] == "info_only"


def test_kill_apply_warns_when_model_p3_missing(tmp_path):
    """산출물이 아직 없으면 조용히 넘어가지 않는다 — 경고하고 최소 파일을 만든다."""
    d = sessions("2016-01-04", 790)
    ks = _ks(make_ledger(d, good=False), rising_close(d), d[-1])
    with pytest.warns(UserWarning, match="model_p3.json"):
        assert track.kill_apply(ks, tmp_path / "model_p3.json", tmp_path / "kill_record.json") is True


def test_kill_validated_requires_stage2_and_ci_lower_above_zero():
    d = sessions("2016-01-04", 790)
    led = make_ledger(d, good=True)
    close8 = episode_close(d, 8)
    ks = track.kill_status(led, close8, d[-1], cfg=FAST, unlock_path=None, manual_path=None)
    assert ks["episodes5"] == 8 and ks["stage2_due"] and ks["stage2_done"]
    assert ks["ci"][0] > 0 and ks["ci_label"] == "validated" and ks["state"] == "validated"
    # 같은 기록이라도 에피소드가 부족하고 60개월 전이면 2차가 아니라 provisional
    ks7 = track.kill_status(led, episode_close(d, 7), d[-1], cfg=FAST, unlock_path=None, manual_path=None)
    assert ks7["episodes5"] == 7 and not ks7["stage2_due"] and ks7["state"] == "provisional"


def test_kill_stage2_by_60_month_cap_with_seven_episodes():
    d = sessions("2016-01-04", 1310)                    # ≈60개월
    led = make_ledger(d, good=True)
    ks = track.kill_status(led, episode_close(d, 7), d[-1], cfg=FAST, unlock_path=None, manual_path=None,
                           prev={"killed": False, "stage1_done": True, "stage2_done": False,
                                 "last_eval_month": 36})
    assert ks["months"] >= 60 and ks["episodes5"] == 7
    assert ks["stage2_due"] and ks["evaluated_now"] and ks["stage2_done"]
    assert ks["last_eval_month"] == ks["months"]


def test_kill_reeval_rhythm_is_twelve_months():
    d = sessions("2016-01-04", 1550)                    # ≈71개월
    led = make_ledger(d, good=True)
    close = rising_close(d)
    prev = {"killed": False, "stage1_done": True, "stage2_done": True, "last_eval_month": 60}
    ks11 = track.kill_status(led, close, d[-1], cfg=FAST, unlock_path=None, manual_path=None, prev=prev)
    assert ks11["months"] == 71 and ks11["stage2_due"] and not ks11["evaluated_now"]
    assert ks11["last_eval_month"] == 60                # 11개월 → 미실행

    d2 = sessions("2016-01-04", 1571)                   # ≈72개월
    led2 = make_ledger(d2, good=True)
    ks12 = track.kill_status(led2, rising_close(d2), d2[-1], cfg=FAST, unlock_path=None,
                             manual_path=None, prev=prev)
    assert ks12["months"] == 72 and ks12["evaluated_now"] and ks12["last_eval_month"] == 72


def test_kill_is_sticky_even_when_later_window_is_good():
    d = sessions("2016-01-04", 900)
    led = make_ledger(d, good=True)                     # 나중 창은 좋다
    prev = {"killed": True, "stage1_done": True, "stage2_done": False, "last_eval_month": 36}
    ks = _ks(led, rising_close(d), d[-1], prev=prev)
    assert ks["bss_clim"] > 0 and ks["killed"] and ks["state"] == "info_only"


def test_kill_defers_when_bss_undefined_instead_of_killing():
    """판정일인데 채점 행이 없다 → 결측을 실패로 읽지 않는다(킬 없음, 사유 기록, 재시도)."""
    d = sessions("2016-01-04", 790)
    led = make_ledger(d, good=True)
    led["y_dd5_20"] = np.nan                            # backfill 전
    ks = _ks(led, rising_close(d), d[-1])
    assert ks["stage1_due"] and ks["bss_clim"] is None
    assert not ks["evaluated_now"] and not ks["killed"] and not ks["stage1_done"]
    assert any("판정 보류" in n for n in ks["notes"])


def test_kill_members_scored_with_same_statistic():
    d = sessions("2016-01-04", 790)
    led = make_ledger(d, good=True, m1_good=False)      # 그림자 M1 은 반대로 맞춘다
    ks = _ks(led, rising_close(d), d[-1])
    m1 = ks["members"]["M1"]
    assert m1["live_start"] == str(d[0].date()) and m1["n"] == 790
    assert m1["bss_clim"] < 0 and m1["verdict"] == "killed"
    assert ks["state"] == "provisional"                 # 멤버 판정은 생산 확률을 바꾸지 않는다
    assert ks["members"]["H"]["verdict"] is None        # p3_p_h 가 장부에 없다
    assert "아직 장부에 없습니다" in ks["members"]["H"]["note"]


def test_kill_manual_file(tmp_path):
    d = sessions("2016-01-04", 200)
    led = make_ledger(d, good=True)
    man = tmp_path / "kill_manual.json"
    man.write_text(json.dumps({"ledger_entry": "7b", "reason": "소유자 판단"}, ensure_ascii=False),
                   encoding="utf-8")
    ks = track.kill_status(led, rising_close(d), d[-1], cfg=FAST, unlock_path=None, manual_path=man)
    assert ks["state"] == "manual_kill" and ks["killed"] and ks["manual"]["ledger_entry"] == "7b"
    model, record = tmp_path / "model_p3.json", tmp_path / "kill_record.json"
    model.write_text('{"schema_version": 3, "deploy_mode": "tones"}', encoding="utf-8")
    assert track.kill_apply(ks, model, record) is True
    assert json.loads(record.read_text(encoding="utf-8"))["state"] == "manual_kill"


def test_manual_kill_without_ledger_entry_warns_but_still_kills(tmp_path):
    d = sessions("2016-01-04", 200)
    man = tmp_path / "kill_manual.json"
    man.write_text("{}", encoding="utf-8")
    with pytest.warns(UserWarning, match="ledger_entry"):
        ks = track.kill_status(make_ledger(d), rising_close(d), d[-1], cfg=FAST,
                               unlock_path=None, manual_path=man)
    assert ks["state"] == "manual_kill" and ks["killed"]


def test_no_automatic_rearm_path(tmp_path):
    """복귀는 코드에 없다(문서 절차만) — kill_apply 는 kill_record 를 지우지 않고 모드를 되돌리지 않는다."""
    assert not any(n for n in dir(track) if n.startswith(("rearm", "unkill", "revive", "clear_kill")))
    d = sessions("2016-01-04", 790)
    bad = _ks(make_ledger(d, good=False), rising_close(d), d[-1])
    model, record = tmp_path / "model_p3.json", tmp_path / "kill_record.json"
    model.write_text('{"schema_version": 3, "deploy_mode": "tones"}', encoding="utf-8")
    track.kill_apply(bad, model, record)
    good = _ks(make_ledger(d, good=True), rising_close(d), d[-1],
               prev={"killed": True, "stage1_done": True, "stage2_done": False, "last_eval_month": 36})
    assert track.kill_apply(good, model, record) is False
    assert record.exists()
    assert json.loads(model.read_text(encoding="utf-8"))["deploy_mode"] == "info_only"
    ks = track.kill_status(make_ledger(d, good=True), rising_close(d), d[-1], cfg=FAST, unlock_path=None,
                           manual_path=None, prev=json.loads(model.read_text(encoding="utf-8"))["kill"])
    assert ks["killed"] and ks["state"] == "info_only"


def test_effective_mode_and_gross_failure_badge(tmp_path):
    rec, man = tmp_path / "kill_record.json", tmp_path / "kill_manual.json"
    assert track.effective_mode("tones", "tones", kill_record_path=rec, kill_manual_path=man) == "tones"
    assert track.effective_mode("info_only", "tones", kill_record_path=rec, kill_manual_path=man) == "info_only"
    rec.write_text("{}", encoding="utf-8")
    assert track.effective_mode("tones", "tones", kill_record_path=rec, kill_manual_path=man) == "info_only"
    # 중대 실패 배지는 12·24개월 중간 점검에서 표시만 되고 킬하지 못한다(§7 3년 하한)
    d = sessions("2016-01-04", 530)                     # ≈24개월
    ks = _ks(make_ledger(d, good=False), rising_close(d), d[-1])
    assert ks["months"] < 36 and ks["gross_failure_badge"] and ks["state"] == "not_due"
    assert not ks["killed"]


# ==================================================================
# 5. 재현(kill_replay)
# ==================================================================
def test_kill_replay_on_synthetic_record():
    d = sessions("2003-01-02", 2600)
    y = y_pattern(len(d))
    oos = pd.DataFrame({"date": d, "y": y, "clim": 0.16, "p_m3": skilled_p(y, True)})
    r = track.kill_replay(oos, rising_close(d), "p_m3", cfg=FAST, ci_windows=10, ci_n_boot=200, n_shuffle=3)
    assert r["n_windows"] > 20 and r["false_kill_share"] == 0.0     # 완벽한 모델은 오기각되지 않는다
    assert r["null_noise_pass"] == 0.0
    assert 0.0 <= r["null_block_shuffle_pass"] <= 1.0
    assert r["power_vs_block_shuffle"] == pytest.approx(1.0 - r["null_block_shuffle_pass"])
    assert "오기각" in r["sentence_ko"]


@pytest.mark.skipif(not (RESULTS_DIR / "calib_p2_walkforward.csv").exists() or not DATA_DIR.exists(),
                    reason="실제 캐시·Phase 2 walk-forward 산출물이 필요합니다")
def test_kill_replay_reproduces_registered_bounds_on_real_record():
    """실제 2003-01-02~2024-08-30 OOS 기록(홀드아웃 미접근)에서 §8.3 사전 등록 수치를 재생성한다.

    측정(2026-09-08, 223창·step 21): M3 오기각 11.66% · 검증 20.0% · BSS p10/p50/p90 −0.0094/+0.0649/+0.2312 ·
    기후학+잡음 0% 통과 · 블록 셔플 21.2% 통과(검정력 78.8%). M1 오기각 15.25%.
    """
    from mrl.data import load_cache
    oos = pd.read_csv(RESULTS_DIR / "calib_p2_walkforward.csv", parse_dates=["date"])
    close = load_cache().close["SPY"].dropna()
    assert oos["date"].max() < pd.Timestamp(HOLDOUT_START)          # 하드컷 확인

    r3 = track.kill_replay(oos, close, "p_m3")
    assert r3["n_windows"] == 223
    assert 0.10 <= r3["false_kill_share"] <= 0.14                   # 문서: ~12%
    assert 0.15 <= r3["validated_share"] <= 0.25                    # 문서: ~20%
    assert r3["null_noise_pass"] == 0.0                             # 문서: 100% 기각
    assert 0.15 <= r3["null_block_shuffle_pass"] <= 0.30            # 문서: ~22% → 검정력 ~78%
    assert r3["bss"]["p10"] == pytest.approx(-0.009, abs=0.01)
    assert r3["bss"]["p50"] == pytest.approx(0.065, abs=0.01)
    assert r3["bss"]["p90"] == pytest.approx(0.232, abs=0.02)
    assert r3["months_to_8"]["p50"] >= 60                           # 문자 그대로의 '8회' 는 중앙 5년 이상

    r1 = track.kill_replay(oos, close, "p_m1")
    assert 0.12 <= r1["false_kill_share"] <= 0.18                   # 문서: ~15%
    # 결정론: 같은 입력 → 같은 숫자
    assert track.kill_replay(oos, close, "p_m3")["validated_share"] == r3["validated_share"]


# ==================================================================
# 6. 드리프트 경보 D1~D11
# ==================================================================
REFERENCE = {"feature_range": {"x_vix": [-4.0, 4.0], "x_har": [-4.0, 4.0],
                               "x_ma": [-4.0, 4.0], "x_hmm": [-9.5, 9.5]},
             "rolling_bss": {"252": [-0.1, 0.0, 0.05, 0.1, 0.2], "756": [0.0, 0.05, 0.1]}}


def _codes(rows) -> set:
    return {r["code"] for r in rows}


def _frame(n: int, start: str = "2016-01-04", **cols) -> pd.DataFrame:
    d = sessions(start, n)
    out = pd.DataFrame({"asof": [str(x.date()) for x in d]})
    for k, v in cols.items():
        out[k] = v
    return out


def test_alarm_d1_p_level_only():
    a = track.alarms(_frame(150, prob_dd5_20=0.50), REFERENCE, unlock_path=None)
    assert _codes(a) == {"D1_p_level"}
    assert a[0]["value"] == pytest.approx(0.50) and a[0]["threshold"] == [0.04, 0.35]
    assert a[0]["action"] == "display"
    assert track.alarms(_frame(150, prob_dd5_20=0.16), REFERENCE, unlock_path=None) == []


def test_alarm_d2_bss_red_and_yellow():
    n = 800
    y = y_pattern(n)
    p = np.where(y > 0.5, 0.05, 0.30)                   # 평균 0.26(D1 무발화) · 반대로 맞춘다
    red = track.alarms(_frame(n, prob_dd5_20=p, p2_clim=0.16, y_dd5_20=y), REFERENCE, unlock_path=None)
    assert _codes(red) == {"D2_bss"} and red[0]["level"] == "red" and red[0]["value"] < -0.05
    quiet = track.alarms(_frame(n, prob_dd5_20=np.where(y > 0.5, 0.30, 0.05), p2_clim=0.16, y_dd5_20=y),
                         REFERENCE, unlock_path=None)
    assert quiet == []


def test_alarm_d2_bss_needs_a_full_window():
    """§8.2 임계는 756/252세션 참조분포의 p5 다 — 부분 창에 그 배지를 붙이지 않는다."""
    n = 40                                               # 라이브 ~2개월
    y = y_pattern(n)
    p = np.where(y > 0.5, 0.05, 0.30)                    # BSS 크게 음수
    notes: list = []
    a = track.alarms(_frame(n, prob_dd5_20=p, p2_clim=0.16, y_dd5_20=y), REFERENCE,
                     unlock_path=None, notes=notes)
    assert "D2_bss" not in _codes(a)
    assert any("D2(756세션)" in s for s in notes) and any("D2(252세션)" in s for s in notes)

    n = 300                                              # 252 는 차지만 756 은 아니다 → 창 표시가 정직해야
    y = y_pattern(n)
    b = track.alarms(_frame(n, prob_dd5_20=np.where(y > 0.5, 0.05, 0.30), p2_clim=0.16, y_dd5_20=y),
                     REFERENCE, unlock_path=None)
    d2 = next(r for r in b if r["code"] == "D2_bss")
    assert d2["level"] == "yellow" and d2["window"] == 252


def test_alarm_d3_har_mae_and_ewma_bias():
    a = track.alarms(_frame(80, p2_har_fc_20=0.10, rv20_realized=0.30), REFERENCE, unlock_path=None)
    assert _codes(a) == {"D3_vol_fc"} and a[0]["kind"] == "har_log_mae" and a[0]["value"] > 0.5
    b = track.alarms(_frame(80, rv20_realized=0.30, p3_sigma_ewma=0.10), REFERENCE, unlock_path=None)
    assert _codes(b) == {"D3_vol_fc"} and b[0]["kind"] == "ewma_bias"
    assert track.alarms(_frame(80, p2_har_fc_20=0.21, rv20_realized=0.20), REFERENCE, unlock_path=None) == []


def test_alarm_d4_vol_target_and_d4b_budget():
    n = 100
    ret = np.where(np.arange(n) % 2 == 0, 0.03, -0.03)
    close = 100.0 * np.cumprod(1.0 + ret)
    a = track.alarms(_frame(n, p3_w_exec=1.0, spy_close=close, p3_sigma_target=0.10),
                     REFERENCE, unlock_path=None)
    assert _codes(a) == {"D4_vol_target"} and a[0]["value"] > 1.5
    assert a[0]["action"] == "halt_sizing_card"
    b = track.alarms(_frame(50, p3_rule_dd=-0.40, p3_d_max=0.35), REFERENCE, unlock_path=None)
    assert _codes(b) == {"D4b_budget"} and b[0]["action"] == "require_ledger_entry"
    assert track.alarms(_frame(50, p3_rule_dd=-0.10, p3_d_max=0.35), REFERENCE, unlock_path=None) == []


def test_alarm_d5_stuck_and_d6_churn():
    a = track.alarms(_frame(260, p2_state="caution"), REFERENCE, unlock_path=None)
    assert _codes(a) == {"D5_stuck"} and a[0]["kind"] == "warn_occupancy"
    b = track.alarms(_frame(260, p3_w_exec=0.20), REFERENCE, unlock_path=None)
    assert _codes(b) == {"D5_stuck"} and b[0]["kind"] == "avg_w_exec"
    w = np.where(np.arange(260) % 2 == 0, 0.50, 0.60)
    c = track.alarms(_frame(260, p3_w_exec=w), REFERENCE, unlock_path=None)
    assert _codes(c) == {"D6_churn"} and c[0]["action"] == "halt_sizing_card"
    assert c[0]["value"] == 251 and c[0]["window"] == 252         # 창은 후행 252세션(260행 중)
    assert track.alarms(_frame(260, p2_state="normal", p3_w_exec=0.70), REFERENCE, unlock_path=None) == []


def test_alarm_d7_parity_is_exit1_and_d10_counts_small_drift():
    led = _frame(40, prob_dd5_20=0.20, p3_w_exec=0.70)
    big = led.copy()
    big["prob_dd5_20"] = 0.26                            # |Δp| = 0.06 > 0.01
    a = track.alarms(led, REFERENCE, unlock_path=None, recomputed=big)
    assert "D7_parity" in _codes(a)
    d7 = next(r for r in a if r["code"] == "D7_parity")
    assert d7["action"] == "exit_1" and d7["n_mismatch"] > 0
    small = led.copy()
    small["prob_dd5_20"] = 0.2001                        # 1e-4: D10 만(자료 개정) — daily 를 죽이지 않는다
    b = track.alarms(led, REFERENCE, unlock_path=None, recomputed=small)
    assert _codes(b) == {"D10_replay"} and b[0]["value"] == 40
    assert track.alarms(led, REFERENCE, unlock_path=None, recomputed=led.copy()) == []


def test_alarm_d8_coverage_and_d9_hmm():
    a = track.alarms(_frame(300, ret20_in_vix80=np.where(np.arange(300) % 10 < 3, 1.0, 0.0)),
                     REFERENCE, unlock_path=None)
    assert _codes(a) == {"D8_coverage"} and a[0]["band"] == "vix80" and a[0]["value"] == pytest.approx(0.3)
    assert track.alarms(_frame(300, ret20_in_vix80=0.94), REFERENCE, unlock_path=None) == []
    assert track.alarms(_frame(100, ret20_in_vix80=0.0), REFERENCE, unlock_path=None) == []   # n_eff 5 < 12

    n, y = 800, y_pattern(800)
    b = track.alarms(_frame(n, p3_hmm_p_high=np.where(y > 0.5, 0.10, 0.90), y_dd5_20=y),
                     REFERENCE, unlock_path=None)
    assert _codes(b) == {"D9_hmm"} and b[0]["value"] < 0.5
    assert track.alarms(_frame(n, p3_hmm_p_high=np.where(y > 0.5, 0.90, 0.10), y_dd5_20=y),
                        REFERENCE, unlock_path=None) == []


def test_alarm_d11_feature_range_needs_five_consecutive():
    x = np.full(60, 0.5)
    x[10:14] = 9.0                                       # 4세션 → 무발화
    assert track.alarms(_frame(60, p2_x_vix=x), REFERENCE, unlock_path=None) == []
    x[10:15] = 9.0                                       # 5세션 연속 → 발화
    a = track.alarms(_frame(60, p2_x_vix=x), REFERENCE, unlock_path=None)
    assert _codes(a) == {"D11_feature_range"} and a[0]["feature"] == "x_vix" and a[0]["sessions"] >= 5


def test_alarms_quiet_on_well_behaved_ledger_and_notes_are_recorded():
    n = 800
    y = y_pattern(n)
    ret = np.where(np.arange(n) % 2 == 0, 0.004, -0.003)
    notes: list[str] = []
    led = _frame(n, prob_dd5_20=skilled_p(y, True), p2_clim=0.16, y_dd5_20=y, p2_state="normal",
                 p2_har_fc_20=0.21, rv20_realized=0.20, p3_sigma_ewma=0.20, p3_w_exec=0.70,
                 spy_close=100.0 * np.cumprod(1.0 + ret), p3_sigma_target=0.10, p3_rule_dd=-0.05,
                 p3_d_max=0.35, ret20_in_vix80=0.94, ret20_in_har80=0.80,
                 p3_hmm_p_high=np.where(y > 0.5, 0.9, 0.1), p2_x_vix=0.5, p2_x_har=0.1, p2_x_ma=-0.2,
                 p3_x_hmm=0.3)
    assert track.alarms(led, REFERENCE, unlock_path=None, notes=notes) == []
    # 재계산 프레임이 없으면 D7·D10 은 '미평가' 로 남는다 — 조용히 통과시키지 않는다
    assert any("D7" in x for x in notes) and any("D10" in x for x in notes)


def test_alarms_respect_holdout_gate(tmp_path):
    led = _frame(150, start="2024-08-01", prob_dd5_20=0.50)
    assert track.alarms(led, REFERENCE, unlock_path=no_unlock(tmp_path)) == []   # 창을 못 채운다
    assert _codes(track.alarms(led, REFERENCE, unlock_path=unlock(tmp_path))) == {"D1_p_level"}


def test_append_alarms_csv_is_append_only(tmp_path):
    p = tmp_path / "alarms.csv"
    rows = [{"asof": "2026-10-01", "code": "D1_p_level", "value": 0.5, "threshold": [0.04, 0.35],
             "action": "display"}]
    assert track.append_alarms(rows, p) == 1
    assert track.append_alarms(rows, p) == 0                       # 같은 (asof, code) 는 다시 쓰지 않는다
    df = pd.read_csv(p)
    assert list(df.columns) == list(track.ALARM_CSV_COLUMNS) and len(df) == 1


def test_alarm_codes_and_actions_cover_the_contract():
    assert set(track.ALARM_CODES) == set(track.ALARM_ACTIONS)
    assert len(track.ALARM_CODES) == 12
    assert track.ALARM_ACTIONS["D7_parity"] == "exit_1"
    # 임계는 전부 config.P3_DRIFT 에서만 온다(모듈 안에 다시 적어 두지 않는다 — 변경은 장부 항목으로만)
    assert set(track.ALARM_CODES) == set(P3_DRIFT)
    a = track.alarms(_frame(150, prob_dd5_20=0.50), REFERENCE, unlock_path=None)
    assert a[0]["threshold"] == [P3_DRIFT["D1_p_level"]["lo"], P3_DRIFT["D1_p_level"]["hi"]]


# ==================================================================
# 7. 재현 검사 (replay_check)
# ==================================================================
def test_replay_check_counts_mismatches_only():
    led = _frame(30, prob_dd5_20=0.2, p3_hmm_p_high=0.4, p3_w_exec=0.7)
    rec = led.copy()
    rec.loc[5, "prob_dd5_20"] = 0.25
    rec.loc[7, "p3_w_exec"] = 0.75
    out = track.replay_check(led, rec)
    assert out["n_compared"] == 30 and out["n_mismatch"] == 2
    assert out["by_col"]["prob_dd5_20"]["n_mismatch"] == 1
    assert out["by_col"]["p3_hmm_p_high"]["n_mismatch"] == 0
    assert out["max_abs_diff"] == pytest.approx(0.05)
    assert track.replay_check(led, led.copy())["n_mismatch"] == 0
    assert track.replay_check(led, rec, sessions=3)["n_mismatch"] == 0     # 최근 3세션만(D7 창)


def test_replay_check_counts_missing_recompute_as_mismatch():
    """장부에 값이 있는데 재계산이 결측이면 재현 실패다 — 조용히 건너뛰면 D7 이 안 뜬다 (§2·§8.2)."""
    led = _frame(30, prob_dd5_20=0.2, p3_hmm_p_high=0.4, p3_w_exec=0.7)
    rec = led.copy()
    rec.loc[29, "p3_w_exec"] = np.nan
    rec.loc[28, "prob_dd5_20"] = np.nan
    out = track.replay_check(led, rec)
    assert out["n_compared"] == 30 and out["n_mismatch"] == 2
    assert out["n_recompute_missing"] == 2 and out["n_ledger_missing"] == 0
    assert out["by_col"]["p3_w_exec"]["n_recompute_missing"] == 1
    assert out["by_col"]["prob_dd5_20"]["n_recompute_missing"] == 1
    assert out["by_col"]["p3_hmm_p_high"]["n_mismatch"] == 0
    assert {r["kind"] for r in out["mismatch_rows"]} == {"recompute_missing"}
    a = track.alarms(led, REFERENCE, unlock_path=None, recomputed=rec)
    d7 = next(r for r in a if r["code"] == "D7_parity")
    assert d7["action"] == "exit_1" and d7["n_recompute_missing"] == 2
    assert "D10_replay" in _codes(a)

    allnan = led.copy()                                   # 재계산이 통째로 실패한 경우
    for c in ("prob_dd5_20", "p3_w_exec"):
        allnan[c] = np.nan
    out2 = track.replay_check(led, allnan)
    assert out2["n_mismatch"] == 60 and out2["max_abs_diff"] == 0.0
    assert "D7_parity" in _codes(track.alarms(led, REFERENCE, unlock_path=None, recomputed=allnan))

    dropped = led.drop(columns=["p3_w_exec"])             # 열 자체가 사라진 경우
    with pytest.warns(UserWarning, match="재계산 프레임에 없습니다"):
        out3 = track.replay_check(led, dropped)
    assert out3["n_mismatch"] == 30 and out3["by_col"]["p3_w_exec"]["n_recompute_missing"] == 30


def test_replay_check_ledger_missing_is_recorded_not_a_mismatch():
    """반대 방향(구 장부 행을 재계산이 채움)은 기록만 — daily 를 exit 1 시키지 않는다."""
    led = _frame(30, prob_dd5_20=0.2, p3_w_exec=0.7)
    led.loc[0:4, "p3_w_exec"] = np.nan                    # 롤아웃 이전 5행
    rec = _frame(30, prob_dd5_20=0.2, p3_w_exec=0.7)
    out = track.replay_check(led, rec)
    assert out["n_mismatch"] == 0 and out["n_ledger_missing"] == 5
    assert out["by_col"]["p3_w_exec"]["n_ledger_missing"] == 5
    assert track.alarms(led, REFERENCE, unlock_path=None, recomputed=rec) == []
    with pytest.warns(UserWarning, match="장부에 없습니다"):   # 신규 열 도입도 불일치가 아니다
        out2 = track.replay_check(_frame(30, prob_dd5_20=0.2), rec)
    assert out2["n_mismatch"] == 0 and out2["by_col"]["p3_w_exec"]["n_ledger_missing"] == 30


# ==================================================================
# 8. 지평별 표시 규칙 (§8.4)
# ==================================================================
def test_horizon_stage_boundaries():
    assert track.horizon_stage(3.0, 3) == "0-3m"
    assert track.horizon_stage(20.0, 3) == "0-3m"        # 개월이 모자라면 올라가지 않는다
    assert track.horizon_stage(5.9, 24) == "0-3m"        # n_eff < 6 이면 무조건 0-3m
    assert track.horizon_stage(6.5, 8) == "6m"
    assert track.horizon_stage(13.0, 20) == "12m"
    assert track.horizon_stage(40.0, 36) == "36m"
    with pytest.raises(ValueError):
        track.may_show("9m", "bss")


@pytest.mark.parametrize("n_eff, months, stage, bss_ok, verdict_ok", [
    (3.0, 3, "0-3m", False, False),
    (6.5, 7, "6m", True, False),
    (13.0, 13, "12m", True, False),
    (40.0, 40, "36m", True, True),
])
def test_horizon_copy_rules(n_eff, months, stage, bss_ok, verdict_ok):
    c = track.horizon_copy({"n_eff": n_eff, "months_elapsed": months, "kill": {"state": "provisional"}})
    assert c["stage"] == stage
    assert c["may_show_bss"] is bss_ok and c["may_show_verdict"] is verdict_ok
    assert c["sentence_ko"] == track.HORIZON_SENTENCES[stage]
    assert c["footnote_ko"] == track.HORIZON_FOOTNOTE
    assert len(c["sentences_ko"]) == track.HORIZON_STAGES.index(stage) + 1
    assert ("bss" in c["hide"]) is (not bss_ok)
    assert ("verdict" in c["hide"]) is (not verdict_ok)
    assert not set(c["show"]) & set(c["hide"])
    assert (c["verdict_state"] == "provisional") is verdict_ok
    if not bss_ok:
        assert "bss" not in c["show"] and "ci" not in c["show"]


def test_live_panel_hides_skill_numbers_below_six_neff():
    d = sessions("2016-01-04", 63)                       # ≈3개월, n_eff 3.15
    led = make_ledger(d, good=True)
    panel = track.live_panel(led, rising_close(d), REFERENCE, d[-1], unlock_path=None)
    assert panel["stage"] == "0-3m"
    v = panel["vii_skill"]
    assert v["brier"] is None and v["bss_clim"] is None and v["ci_bss_clim"] is None
    assert v["hist_pct"] is None and "bss" in v["hidden"] and v["hidden_reason"]
    assert panel["ix_kill"]["state"] is None and panel["ix_kill"]["hidden"] == ["verdict"]
    assert panel["i_p_level"]["mean_p"] is not None       # 세는 것은 계속 보인다
    assert panel["iv_exposure"]["n"] == 0                 # 비중 열이 없으면 0 (조용한 1.0 금지)


def test_live_panel_shows_numbers_and_verdict_after_thirty_six_months():
    d = sessions("2016-01-04", 790)
    led = make_ledger(d, good=True, m1_good=True)
    panel = track.live_panel(led, episode_close(d, 8), REFERENCE, d[-1], unlock_path=None)
    assert panel["stage"] == "36m"
    v = panel["vii_skill"]
    assert v["bss_clim"] > 0 and v["ci_bss_clim"] is not None and "hidden" not in v
    assert v["hist_pct"]["252"] is not None
    assert panel["ix_kill"]["state"] in ("provisional", "validated")
    assert panel["ix_kill"]["countdown"]["episodes"] == "8/8"
    assert set(panel) >= {"i_p_level", "ii_state_occupancy", "iii_switching", "iv_exposure",
                          "v_vol_forecast", "vi_coverage", "vii_skill", "viii_relative", "ix_kill",
                          "x_alarms", "xi_registry"}


def test_live_panel_six_month_stage_shows_bss_but_no_verdict():
    d = sessions("2016-01-04", 145)                      # ≈7개월, n_eff 7.25
    led = make_ledger(d, good=True)
    panel = track.live_panel(led, rising_close(d), REFERENCE, d[-1], unlock_path=None)
    assert panel["stage"] == "6m"
    assert panel["vii_skill"]["bss_clim"] is not None
    assert panel["ix_kill"]["state"] is None              # 판정어 금지
    assert panel["horizon"]["sentence_ko"] == track.HORIZON_SENTENCES["6m"]


# ==================================================================
# 9. 단일 요약 (§9 summary().p3)
# ==================================================================
def test_summary_p3_contract_keys_and_strict_json(tmp_path):
    d = sessions("2016-01-04", 790)
    led = make_ledger(d, good=True, m1_good=True, p3_w_exec=0.70, p3_lo=0.10, p3_hi=0.30)
    s = track.summary_p3(led, episode_close(d, 8), REFERENCE, d[-1], unlock_path=None,
                         p2_deploy_mode="tones", p3_deploy_mode="tones", deploy_sizing=True)
    for k in ("schema_version", "n_rows", "n_scored", "n_eff", "live_start", "months_elapsed",
              "effective_mode", "deploy_sizing", "sizing", "members", "disagreement", "kill",
              "alarms", "replay_mismatch_count", "coverage", "notes"):
        assert k in s, k
    assert s["schema_version"] == 3 and s["n_rows"] == 790 and s["n_scored"] == 790
    assert s["n_eff"] == pytest.approx(39.5)
    assert s["kill"]["state"] in ("provisional", "validated")
    assert s["horizon"]["stage"] == "36m"
    assert s["effective_mode"] == "tones"
    assert s["disagreement"]["median_width"] == pytest.approx(0.20)
    out = track.save_summary_p3(s, tmp_path / "track_record_p3.json")     # allow_nan=False (엄격 JSON)
    assert json.loads(out.read_text(encoding="utf-8"))["n_rows"] == 790


def test_summary_p3_empty_ledger_is_not_started():
    s = track.summary_p3(pd.DataFrame(columns=["asof"]), None, {}, "2026-10-01", unlock_path=None)
    assert s["n_rows"] == 0 and s["kill"]["state"] == "not_started" and s["alarms"] == []


def test_window_distribution_and_hist_pct():
    d = sessions("2003-01-02", 1600)
    y = y_pattern(len(d))
    oos = pd.DataFrame({"date": d, "y": y, "clim": 0.16, "p": skilled_p(y, True)})
    wd = track.window_distribution(oos, "p", windows=(252, 756), unlock_path=None)
    assert set(wd["rolling_bss"]) == {"252", "756"} and len(wd["rolling_bss"]["252"]) > 10
    assert wd["quantiles"]["252"]["p50"] > 0
    ref = {"rolling_bss": {"252": [-0.1, 0.0, 0.1, 0.2]}}
    assert track.hist_pct(0.05, 252, ref) == pytest.approx(0.5)
    assert track.hist_pct(0.30, 252, ref) == pytest.approx(1.0)
    assert track.hist_pct(0.05, 63, ref) is None and track.hist_pct(float("nan"), 252, ref) is None


def test_live_start_requires_tones_mode():
    d = sessions("2016-01-04", 10)
    led = make_ledger(d, mode="info_only")
    led.loc[4:, "p2_deploy_mode"] = "tones"
    assert track.live_start(led) == d[4]
    assert track.live_start(led, mode_col=None) == d[0]           # 그림자 멤버 규칙
    led2 = make_ledger(d, mode="info_only")
    assert track.live_start(led2) is None
