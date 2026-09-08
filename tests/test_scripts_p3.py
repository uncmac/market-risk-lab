# -*- coding: utf-8 -*-
"""Phase 3 스크립트 통합 테스트 — run_phase3.py · daily.py 의 P3 블록 · selftest.py 의 P3 절 · weekly.yml
(ARCHITECTURE_PHASE3.md §11·§12·§13).

* 실제 캐시(data/)는 읽기만 하고 산출물은 tmp_path 에만 쓴다 (results/ · docs/ · track_record.csv 는 건드리지 않는다).
* 짧은 창(--end 2006-12-29)으로 Phase 2 산출물을 만들고 그 위에 Phase 3 를 돌린다 — 모듈 안에서 한 번만(fixture).
* 검사: 산출물 계약(열·키·엄격 JSON·하드컷) · 결정론(두 번 실행 == 바이트 동일, 훼손 시 exit 1) · spec/sha 게이트 ·
  --holdout-final 거부(exit 2) · 계약 함수 부재 시 큰 소리로 실패 · daily 의 P3 장부 열·카드·유효 모드·GitHub 출력 ·
  킬 기록이 있으면 카드가 info_only · selftest 의 P3 절 · 워크플로 YAML(파이썬 yaml 로 파싱).
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrl.config import DATA_DIR, HOLDOUT_START, RESULTS_DIR  # noqa: E402

pytestmark = pytest.mark.skipif(not (DATA_DIR / "close.csv").exists(), reason="data/ 캐시 없음")

BAD_TOKEN = re.compile(r"\b(undefined|NaN|nan|None|null|NaT)\b")
SHORT_END = "2006-12-29"
P3_LEDGER_HEAD = ("p3_sigma_ewma", "p3_sigma_target", "p3_w_vol", "p3_w_exec", "p3_w_reason", "p3_hmm_p_high",
                  "p3_p_h", "p3_hmm_theta_id", "p3_members", "p3_lo", "p3_hi", "p3_kill_state",
                  "p3_deploy_mode", "p3_effective_mode", "p3_registry_sha", "p3_sizing_sha", "p3_run_id")


def _strict_json(path: Path) -> dict:
    def _no_const(c):
        raise ValueError(f"엄격하지 않은 JSON 상수: {c}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_no_const)


def _walk_nan(o, path="") -> list[str]:
    bad: list[str] = []
    if isinstance(o, dict):
        for k, v in o.items():
            bad += _walk_nan(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            bad += _walk_nan(v, f"{path}[{i}]")
    elif isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        bad.append(path)
    elif isinstance(o, str) and o.strip().lower() in ("nan", "none", "inf", "-inf", "nat"):
        bad.append(f"{path}={o}")
    return bad


def _visible_text(html: str) -> str:
    t = re.sub(r"<style.*?</style>", "", html, flags=re.S)
    t = re.sub(r"<script.*?</script>", "", t, flags=re.S)
    return re.sub(r"<[^>]+>", " ", t)


def _last_session() -> pd.Timestamp:
    spy = pd.read_csv(DATA_DIR / "spy_ohlc.csv", index_col="date", parse_dates=["date"])
    return spy.index[-1]


def _report_p3_ready() -> bool:
    from mrl import report as RPT
    return all(hasattr(RPT, n) for n in ("charts_p3", "render_sizing_report", "render_regime_report",
                                         "render_track_record", "p3_card"))


# ------------------------------------------------------------------
# fixture — 짧은 창의 Phase 2 산출물 위에 Phase 3 (모듈 안에서 한 번)
# ------------------------------------------------------------------
@pytest.fixture(scope="module")
def p3_short(tmp_path_factory):
    from scripts import run_backtest_v1 as RB1
    from scripts import run_calibration as RC
    from scripts import run_phase3 as RP3
    base = tmp_path_factory.mktemp("p3")
    results, docs = base / "results", base / "docs"
    assert RC.main(["--end", SHORT_END, "--first-refit", "2003-01-02", "--results-dir", str(results),
                    "--docs-dir", str(docs), "--no-charts"]) == 0
    assert RB1.main(["--results-dir", str(results), "--docs-dir", str(docs), "--no-charts"]) == 0
    argv = ["--end", SHORT_END, "--results-dir", str(results), "--docs-dir", str(docs), "--no-charts"]
    if not _report_p3_ready():
        argv.append("--no-docs")
    assert RP3.main(argv) == 0
    return {"results": results, "docs": docs, "base": base}


# ------------------------------------------------------------------
# run_phase3.py
# ------------------------------------------------------------------
def test_run_phase3_writes_all_artifacts(p3_short):
    """산출물 6개 + 엄격 JSON + §11 키 계약 + CSV 열 + 홀드아웃 하드컷 + 파라미터 회계(생산 추가 0)."""
    from mrl import ensemble as EN
    from mrl import regime as RG
    from mrl import sizing as SZ
    results = p3_short["results"]
    for name in ("oos_p3.csv", "backtest_p3.csv", "summary_p3.json", "model_p3.json", "hmm_p3.json"):
        assert (results / name).exists(), name

    js = _strict_json(results / "summary_p3.json")
    for k in ("schema_version", "disclosure", "run", "params", "v0_reference", "registry", "ladder", "blocks",
              "admission", "admission_power", "ablations", "selftest", "reliability", "era_auc", "hmm",
              "decision_ens3", "sizing", "scenarios", "kill_power", "reference", "determinism", "flags",
              "warnings", "artifacts"):
        assert k in js, f"summary_p3.json 에 {k} 없음"
    assert _walk_nan(js) == []
    assert js["disclosure"] == "#2 사전 관측 + 설계 단계 비중 관측 참조"
    run = js["run"]
    assert run["end"] == SHORT_END and run["hard_cut"] is True and run["holdout_final"] is False
    assert run["registry_sha"] == EN.registry_sha256() and run["sizing_sha"] == SZ.sizing_sha256()
    assert run["obs_spec"] == RG.OBS_SPEC and run["sigma_target"] == 0.10 and run["d_max"] == 0.35
    for k in ("python", "pandas", "numpy", "sklearn", "cache", "spec_sha256", "data_sha256", "runtime_sec"):
        assert k in run
    # §14 파라미터 회계: 생산 확률에 Phase 3 가 더한 적합값은 0
    assert js["params"]["production_K_s"] == 4 and js["params"]["sizing_n_params"] == 0
    assert js["params"]["shadow"]["phase3_added_to_production"] == 0
    assert js["params"]["hmm_param_count"] == 12
    st = js["selftest"]
    assert st["hmm_param_count_ok"] and st["p2_param_count_ok"] and st["sizing_n_params_ok"]
    assert st["production_shadow_only"] is True and st["pit_bit_identical"] is True

    # OOS CSV — §11 열 계약, 하드컷, 멤버 확률 [0,1]
    oos = pd.read_csv(results / "oos_p3.csv", index_col="date", parse_dates=["date"])
    assert list(oos.columns) == ["y", "clim", "p_p2", "p_m1", "p_h", "p_hmm_high", "p20", "q20", "x_hmm",
                                 "p_ens", "lo", "hi", "state_p2", "state_ens3", "refit_year", "block24",
                                 "block18", "theta_id"]
    assert oos.index[0] == pd.Timestamp("2003-01-02") and oos.index[-1] == pd.Timestamp(SHORT_END)
    assert (oos.index < pd.Timestamp(HOLDOUT_START)).all()
    for c in ("p_p2", "p_m1", "p_h", "p_hmm_high", "p20", "q20", "p_ens", "lo", "hi"):
        v = oos[c].dropna()
        assert ((v >= 0) & (v <= 1)).all(), c
    assert (oos["lo"] <= oos["hi"]).all()
    # admitted = ∅ 이므로 생산 평균 ≡ p2 (Phase 3 는 별도의 생산 확률을 만들지 않는다)
    both = oos["p_ens"].notna() & oos["p_p2"].notna()
    assert np.allclose(oos.loc[both, "p_ens"], oos.loc[both, "p_p2"], atol=0, rtol=0)
    assert set(oos["state_p2"].dropna().unique()) <= {"normal", "caution", "reduce"}

    # 비중 CSV
    bt = pd.read_csv(results / "backtest_p3.csv", index_col="date", parse_dates=["date"])
    for c in ("sigma_ewma", "sigma_har_fc", "state", "mult", "w_vol", "w_target", "w_exec", "reason",
              "ret_rule", "ret_bh", "dd_rule"):
        assert c in bt.columns, c
    assert set(bt["reason"].dropna().unique()) <= set(SZ.REASONS)
    w = bt["w_exec"].dropna()
    assert ((w >= SZ.P3["w_min"] - 1e-9) & (w <= SZ.P3["w_max"] + 1e-9)).all()
    assert (bt["dd_rule"].dropna() <= 1e-12).all()

    # model_p3.json — 킬 상태·멤버·비중 상수
    m3 = _strict_json(results / "model_p3.json")
    assert m3["registry_sha"] == EN.registry_sha256() and m3["sizing_sha"] == SZ.sizing_sha256()
    assert m3["deploy_mode"] in ("tones", "info_only")
    assert {"p2", "M1", "H"} == set(m3["members"]) and m3["members"]["p2"]["status"] == "seed"
    assert m3["members"]["M1"]["status"] in ("shadow", "candidate_rejected", "killed")
    assert m3["sizing"]["n_params"] == 0 and m3["sizing"]["sigma_target"] == 0.10
    assert m3["kill"]["state"] in ("not_started", "not_due", "provisional", "validated", "info_only", "manual_kill")

    # hmm_p3.json — θ 12개 파라미터 · 재적합 4회(2003~2006)
    hm = _strict_json(results / "hmm_p3.json")
    assert hm["param_count"] == 12 and len(hm["thetas"]) == 4
    assert [t["refit_date"] for t in hm["thetas"]] == ["2003-01-02", "2004-01-02", "2005-01-03", "2006-01-03"]


def test_run_phase3_registry_and_admission_are_rejection_only(p3_short):
    """등록부는 1일차 전부 그림자이고, 관측 기록의 채택 검정은 **기각만** 한다(§5.4). 검정력 표가 함께 인쇄된다."""
    js = _strict_json(p3_short["results"] / "summary_p3.json")
    reg = {r["member"]: r for r in js["registry"]}
    assert set(reg) == {"p2", "M1", "H"}
    assert reg["p2"]["in_average"] is True and reg["M1"]["in_average"] is False and reg["H"]["in_average"] is False
    assert reg["H"]["K_s"] == 2 and reg["H"]["K_u"] == 12 and reg["M1"]["K_s"] == 2
    adm = js["admission"]
    assert adm["order"] == ["H", "M1"] and adm["fresh_blocks_min"] == 11
    for table in ("blocks24", "blocks18"):
        assert adm[table]["admitted"] == ["p2"], f"{table}: 관측 기록으로 채택되면 안 된다"
        for name, res in adm[table]["results"].items():
            assert res["verdict"] in ("SHADOW", "ADMIT") and res["evidence"] == "observed"
            assert res["verdict"] == "SHADOW", f"{table}/{name} 이 관측 기록으로 ADMIT 되었다"
    pw = {round(float(r["q"]), 2): float(r["p_at_least"]) for r in js["admission_power"]}
    assert pw[0.5] == pytest.approx(0.113, abs=0.002) and pw[0.8] == pytest.approx(0.839, abs=0.002)


def test_run_phase3_is_deterministic_and_refuses_tampered_theta(p3_short):
    """같은 입력·같은 sha 로 두 번 → CSV 바이트 동일 · determinism.status == 'compared'.
    저장된 θ 를 1e-6 틀면 결정론 검사가 실패해 exit 1 이고 **파일을 쓰지 않는다**."""
    from scripts import run_phase3 as RP3
    results, docs = p3_short["results"], p3_short["docs"]
    before_oos = (results / "oos_p3.csv").read_bytes()
    before_bt = (results / "backtest_p3.csv").read_bytes()
    argv = ["--end", SHORT_END, "--results-dir", str(results), "--docs-dir", str(docs), "--no-charts", "--no-docs"]
    assert RP3.main(argv) == 0
    assert (results / "oos_p3.csv").read_bytes() == before_oos
    assert (results / "backtest_p3.csv").read_bytes() == before_bt
    det = _strict_json(results / "summary_p3.json")["determinism"]
    assert det["status"] == "compared" and det["ok"] is True
    assert det["theta_max_abs_diff"] == 0.0 and det["backtest_csv_same"] is True
    assert det["registry_sha_same"] and det["sizing_sha_same"] and det["data_sha_same"]

    hmm_path = results / "hmm_p3.json"
    good = hmm_path.read_bytes()
    hm = json.loads(good.decode("utf-8"))
    hm["thetas"][-1]["A"][0][0] += 1e-6                     # 허용오차 1e-7 밖
    tampered = json.dumps(hm, ensure_ascii=False, indent=1) + "\n"
    hmm_path.write_text(tampered, encoding="utf-8")
    assert RP3.main(argv) == 1
    assert hmm_path.read_text(encoding="utf-8") == tampered  # 실패한 실행은 산출물을 덮지 않는다
    assert (results / "backtest_p3.csv").read_bytes() == before_bt
    hmm_path.write_bytes(good)
    assert RP3.main(argv) == 0                               # 복구 후 다시 통과 (다른 테스트가 이 fixture 를 쓴다)


def test_determinism_check_separates_sensitivity_option_from_nondeterminism(p3_short, tmp_path):
    """`--all-sensitivities` 는 실행 **옵션**이라 `backtest_p3.csv` 의 `w_S-*` 열 집합만 바꾼다.

    옵션이 다른 두 실행을 전체 해시로 비교하면 결정론 실패로 오판한다(§2: 결정론은 '같은 sha·같은 입력이면
    같은 산출' 이지 '같은 옵션이면' 이 아니다). 공통 열은 바이트 동일이어야 통과하고, 계약 열(`w_S-` 가 아닌
    열)이 달라지면 그때는 진짜 실패여야 한다."""
    import shutil
    from scripts import run_phase3 as RP3
    dst, docs = tmp_path / "results", tmp_path / "docs"
    shutil.copytree(p3_short["results"], dst)
    base = ["--end", SHORT_END, "--results-dir", str(dst), "--docs-dir", str(docs), "--no-charts", "--no-docs"]

    assert RP3.main(base + ["--all-sensitivities"]) == 0          # 저장본을 넓은 열 집합으로 만든다
    wide = (dst / "backtest_p3.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert RP3.main(base) == 0                                     # 좁은 열 집합으로 다시 — 실패하면 안 된다
    narrow = (dst / "backtest_p3.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    det = _strict_json(dst / "summary_p3.json")["determinism"]
    dropped = [c for c in wide if c not in narrow]
    assert dropped and all(c.startswith("w_S-") for c in dropped)   # 사라진 것은 민감도 열뿐
    assert det["status"] == "compared" and det["ok"] is True
    assert det["backtest_csv_same"] is True and det["backtest_csv_columns_same"] is False
    assert det["backtest_csv_sens_only_stored"] == dropped

    # 공통 열의 값이 틀어지면(계약 열 훼손) 진짜 실패로 잡아야 한다
    lines = (dst / "backtest_p3.csv").read_text(encoding="utf-8").splitlines()
    cells = lines[2].split(",")
    cells[narrow.index("w_exec")] = "0.123456"
    lines[2] = ",".join(cells)
    (dst / "backtest_p3.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert RP3.main(base) == 1


def test_summary_p3_sizing_sensitivities_is_a_scored_table_not_names(p3_short):
    """`summary_p3.json.sizing.sensitivities` 는 §6.3 형식의 **채점된 표**다 — 리포트 ⑥ 이 이걸 읽는다.

    이름 목록만 넣으면 `_p3_window_tables` 가 레코드를 하나도 못 찾아 페이지가 조용히 빈다."""
    sz = _strict_json(p3_short["results"] / "summary_p3.json")["sizing"]
    rows = sz["sensitivities"]
    assert isinstance(rows, list) and rows and all(isinstance(r, dict) for r in rows)
    for key in ("row", "window", "start", "end", "cagr", "max_dd", "avg_exposure", "switches_per_year"):
        assert key in rows[0], f"민감도 표에 §6.3 열 {key} 가 없다"
    names = {str(r["row"]) for r in rows}
    # --all-sensitivities 없이도 변형 전부가 실려야 한다(④ 의 표와 달리 ⑥ 은 '규칙의 가족' 전체)
    assert set(sz["sensitivity_names"]) <= names and len(sz["sensitivity_names"]) > len(sz["sensitivities_in_table"])
    assert "adopted" in names, "기준선 없이 민감도만 읽게 두면 안 된다"


def test_run_phase3_refuses_stale_p2_spec(p3_short, tmp_path):
    """summary_p2.json 의 spec_sha256 이 현재 코드와 다르면 exit 1 — 옛 확률 위에 Phase 3 를 세우지 않는다."""
    import shutil
    from scripts import run_phase3 as RP3
    dst = tmp_path / "results"
    shutil.copytree(p3_short["results"], dst)
    sp = json.loads((dst / "summary_p2.json").read_text(encoding="utf-8"))
    sp["run"]["spec_sha256"] = "0" * 64
    (dst / "summary_p2.json").write_text(json.dumps(sp, ensure_ascii=False), encoding="utf-8")
    assert RP3.main(["--end", SHORT_END, "--results-dir", str(dst), "--docs-dir", str(tmp_path / "docs"),
                     "--no-charts", "--no-docs"]) == 1


def test_run_phase3_refuses_holdout_end_and_holdout_final_without_unlock(p3_short, tmp_path):
    """--end 가 홀드아웃 안이면 exit 1(하드컷 위반), --holdout-final 은 unlock 파일이 없으면 exit 2."""
    import shutil
    from scripts import run_phase3 as RP3
    dst = tmp_path / "results"
    shutil.copytree(p3_short["results"], dst)
    docs = tmp_path / "docs"
    assert RP3.main(["--end", "2025-01-02", "--results-dir", str(dst), "--docs-dir", str(docs),
                     "--no-charts", "--no-docs"]) == 1
    assert RP3.main(["--holdout-final", "--results-dir", str(dst), "--docs-dir", str(docs),
                     "--no-charts", "--no-docs"]) == 2


def test_run_phase3_fails_loudly_when_contract_missing(p3_short, tmp_path, monkeypatch, capsys):
    """§9·§10 계약 이름이 없으면 **조용히 건너뛰지 않고** 무엇이 없는지 말하며 exit 1 한다(동시 확장 중인 모듈 대비)."""
    from mrl import ledger as L
    from mrl import report as RPT
    from scripts import run_phase3 as RP3
    args = ["--end", SHORT_END, "--results-dir", str(p3_short["results"]),
            "--docs-dir", str(tmp_path / "docs"), "--no-charts"]
    monkeypatch.delattr(RPT, "charts_p3", raising=False)
    rc = RP3.main(args)
    out = capsys.readouterr().out
    assert rc == 1 and "mrl.report.charts_p3" in out and "--no-docs" in out
    monkeypatch.undo()
    monkeypatch.delattr(L, "P3_COLUMNS", raising=False)              # 장부 계약은 --no-docs 여도 본다
    rc2 = RP3.main(args + ["--no-docs"])
    out2 = capsys.readouterr().out
    assert rc2 == 1 and "mrl.ledger.P3_COLUMNS" in out2

    # daily 도 같은 계약을 본다
    from scripts import daily as DY
    with pytest.raises(DY.P3Fatal, match="mrl.ledger.P3_COLUMNS"):
        DY.require_p3_contract()


def test_run_phase3_kill_record_keeps_info_only(p3_short, tmp_path):
    """킬 기록이 있으면 주간 재적합이 model_p3.deploy_mode 를 tones 로 되돌리지 못한다(sticky · §8.3·§16 11)."""
    import shutil
    from scripts import run_phase3 as RP3
    dst = tmp_path / "results"
    shutil.copytree(p3_short["results"], dst)
    (dst / "kill_record.json").write_text(json.dumps(
        {"asof": SHORT_END, "bss": -0.01, "n": 800, "n_eff": 40.0, "ledger_entry": "7x"}, ensure_ascii=False),
        encoding="utf-8")
    assert RP3.main(["--end", SHORT_END, "--results-dir", str(dst), "--docs-dir", str(tmp_path / "docs"),
                     "--no-charts", "--no-docs"]) == 0
    m3 = _strict_json(dst / "model_p3.json")
    assert m3["deploy_mode"] == "info_only"
    js = _strict_json(dst / "summary_p3.json")
    assert js["run"]["effective_mode"] == "info_only"
    assert m3["mode_history"][-1]["deploy_mode"] == "info_only"


@pytest.mark.skipif(not _report_p3_ready(), reason="mrl.report 의 Phase 3 렌더러가 아직 없음")
def test_run_phase3_renders_three_pages(p3_short):
    """docs/{sizing_p3,regime_p3,track_record}.html 이 불량 토큰 없이 섹션을 모두 갖는다(§10)."""
    docs = p3_short["docs"]
    for name in ("sizing_p3.html", "regime_p3.html", "track_record.html"):
        p = docs / name
        assert p.exists(), name
        html = p.read_text(encoding="utf-8")
        txt = _visible_text(html)
        assert not BAD_TOKEN.search(txt), f"{name}: {BAD_TOKEN.findall(txt)[:5]}"
        for i in range(1, 10):
            assert f'id="s{i}"' in html, f"{name}: 섹션 s{i} 없음"
        assert b"\r\n" not in p.read_bytes()


# ------------------------------------------------------------------
# daily.py — Phase 3 블록
# ------------------------------------------------------------------
def _daily_results(tmp_path, src_results: Path) -> Path:
    """Phase 2·3 산출물을 tmp 로 복사(장부와 짝이 되는 results 디렉터리)."""
    import shutil
    dst = tmp_path / "results"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("model_p2.json", "summary_p2.json", "model_p3.json", "hmm_p3.json", "summary_p3.json"):
        if (src_results / name).exists():
            shutil.copy(src_results / name, dst / name)
    return dst


@pytest.mark.skipif(not (RESULTS_DIR / "model_p3.json").exists(), reason="results/model_p3.json 없음 (run_phase3.py 먼저)")
def test_daily_writes_p3_ledger_columns_and_card(tmp_path):
    """daily.py --no-update 가 P3 장부 열(§9)과 index.html 의 p3 카드(§10)를 쓴다. 재실행은 행을 늘리지 않는다."""
    from scripts import daily as DY
    from mrl import ensemble as EN
    from mrl import sizing as SZ
    results = _daily_results(tmp_path, RESULTS_DIR)
    ledger, docs = tmp_path / "track_record.csv", tmp_path / "docs"
    now = f"{_last_session():%Y-%m-%d} 17:00"
    args = ["--no-update", "--now", now, "--ledger", str(ledger), "--docs-dir", str(docs),
            "--results-dir", str(results)]
    assert DY.main(args) == 0
    df = pd.read_csv(ledger, dtype=str)
    assert len(df) == 1
    for c in P3_LEDGER_HEAD:
        assert c in df.columns, c
    row = df.loc[0]
    assert float(row["p3_sigma_ewma"]) > 0 and float(row["p3_sigma_target"]) == 0.10
    assert 0.25 <= float(row["p3_w_vol"]) <= 1.0
    assert row["p3_w_reason"] in SZ.REASONS and row["p3_kill_state"] in ("not_started", "not_due", "provisional",
                                                                        "validated", "info_only", "manual_kill")
    assert row["p3_effective_mode"] in ("info_only", "tones") and row["p3_deploy_mode"] in ("info_only", "tones")
    assert row["p3_registry_sha"] == EN.registry_sha256() and row["p3_sizing_sha"] == SZ.sizing_sha256()
    assert 0.0 <= float(row["p3_hmm_p_high"]) <= 1.0 and 0.0 <= float(row["p3_p_h"]) <= 1.0
    mem = json.loads(row["p3_members"])
    assert list(mem) == ["p2", "M1", "H"] and all(v is None or 0.0 <= v <= 1.0 for v in mem.values())
    assert float(row["p3_lo"]) <= float(row["p3_hi"])
    assert float(row["p3_range_vix_lo"]) < float(row["p3_range_vix_hi"])       # 20세션 80% 밴드(장부 채점용)
    # 재실행 → 행 불변
    assert DY.main(args) == 0
    assert len(pd.read_csv(ledger)) == 1
    if _report_p3_ready():
        html = (docs / "index.html").read_text(encoding="utf-8")
        txt = _visible_text(html)
        assert not BAD_TOKEN.search(txt), BAD_TOKEN.findall(txt)[:5]
        assert 'id="p3"' in html and html.index('id="p2"') < html.index('id="p3"')   # P2 카드 아래
        assert "오늘의 주식 비중과 그 근거" in txt


@pytest.mark.skipif(not (RESULTS_DIR / "model_p3.json").exists(), reason="results/model_p3.json 없음")
def test_daily_p3_is_info_only_while_p2_is(tmp_path):
    """p2 가 info_only 인 동안 유효 모드도 info_only — 카드에 비중·상태·사다리가 없고 §7 문구가 보인다(§15 단계 0)."""
    from scripts import daily as DY
    results = _daily_results(tmp_path, RESULTS_DIR)
    sp = json.loads((results / "summary_p2.json").read_text(encoding="utf-8"))
    if (sp.get("acceptance") or {}).get("deploy_mode") == "tones":
        pytest.skip("p2 가 tones 로 배치됨 — 이 테스트는 info_only 경로를 본다")
    ledger, docs = tmp_path / "track_record.csv", tmp_path / "docs"
    now = f"{_last_session():%Y-%m-%d} 17:00"
    assert DY.main(["--no-update", "--now", now, "--ledger", str(ledger), "--docs-dir", str(docs),
                    "--results-dir", str(results)]) == 0
    row = pd.read_csv(ledger, dtype=str).loc[0]
    assert row["p3_effective_mode"] == "info_only" and row["p3_w_reason"] == "info_only"
    assert pd.isna(row["p3_w_exec"]) and pd.isna(row["p3_state_mult"])          # 비중·상태 배수를 제안하지 않는다
    assert not pd.isna(row["p3_w_vol"])                                         # 변동성 단독 가정치는 기록한다
    if _report_p3_ready():
        html = (docs / "index.html").read_text(encoding="utf-8")
        card = html[html.index('id="p3"'):]
        txt = _visible_text(card)
        assert "정보 제공 전용 — 톤·비중 제안 숨김" in txt
        assert "예산 사다리는 유효 배치 모드에서만 표시합니다" in txt


@pytest.mark.skipif(not (RESULTS_DIR / "model_p3.json").exists(), reason="results/model_p3.json 없음")
def test_daily_p3_kill_record_forces_info_only(tmp_path):
    """kill_record.json 이 있으면 model_p3.deploy_mode 가 tones 여도 유효 모드는 info_only 다(§8.3 sticky)."""
    from scripts import daily as DY
    results = _daily_results(tmp_path, RESULTS_DIR)
    m3 = json.loads((results / "model_p3.json").read_text(encoding="utf-8"))
    m3["deploy_mode"] = "tones"
    (results / "model_p3.json").write_text(json.dumps(m3, ensure_ascii=False), encoding="utf-8")
    (results / "kill_record.json").write_text(json.dumps(
        {"asof": "2026-09-08", "bss": -0.02, "ci_label": "rejected", "ledger_entry": "7x"}, ensure_ascii=False),
        encoding="utf-8")
    ledger, docs = tmp_path / "track_record.csv", tmp_path / "docs"
    now = f"{_last_session():%Y-%m-%d} 17:00"
    assert DY.main(["--no-update", "--now", now, "--ledger", str(ledger), "--docs-dir", str(docs),
                    "--results-dir", str(results)]) == 0
    row = pd.read_csv(ledger, dtype=str).loc[0]
    assert row["p3_deploy_mode"] == "tones" and row["p3_effective_mode"] == "info_only"
    assert row["p3_w_reason"] == "info_only" and pd.isna(row["p3_w_exec"])
    if _report_p3_ready():
        txt = _visible_text((docs / "index.html").read_text(encoding="utf-8"))
        assert "킬룰 발동" in txt or "정보 제공 전용 — 톤·비중 제안 숨김" in txt


@pytest.mark.skipif(not (RESULTS_DIR / "model_p3.json").exists(), reason="results/model_p3.json 없음")
def test_daily_fails_on_stale_p3_sha_and_on_half_built_artifacts(tmp_path):
    """registry_sha/sizing_sha 불일치 → exit 1(코드가 산출물보다 새롭다). hmm_p3.json 만 없어도 exit 1."""
    from scripts import daily as DY
    now = f"{_last_session():%Y-%m-%d} 17:00"

    stale = _daily_results(tmp_path / "a", RESULTS_DIR)
    m3 = json.loads((stale / "model_p3.json").read_text(encoding="utf-8"))
    m3["sizing_sha"] = "f" * 64
    (stale / "model_p3.json").write_text(json.dumps(m3, ensure_ascii=False), encoding="utf-8")
    led_a = tmp_path / "a" / "track_record.csv"
    assert DY.main(["--no-update", "--now", now, "--ledger", str(led_a),
                    "--docs-dir", str(tmp_path / "a" / "docs"), "--results-dir", str(stale)]) == 1
    assert not led_a.exists()

    half = _daily_results(tmp_path / "b", RESULTS_DIR)
    (half / "hmm_p3.json").unlink()
    led_b = tmp_path / "b" / "track_record.csv"
    assert DY.main(["--no-update", "--now", now, "--ledger", str(led_b),
                    "--docs-dir", str(tmp_path / "b" / "docs"), "--results-dir", str(half)]) == 1
    assert not led_b.exists()


def test_daily_without_phase3_artifacts_warns_and_keeps_going(tmp_path, capsys):
    """주간 작업이 아직 돌지 않은 results 디렉터리(P3 산출물 둘 다 없음)에서는 **소리 내어** 건너뛰고 계속한다.
    --require-p3 를 주면 §11 문자 그대로 exit 1."""
    import shutil
    from scripts import daily as DY
    if not (RESULTS_DIR / "model_p2.json").exists():
        pytest.skip("results/model_p2.json 없음")
    results = tmp_path / "results"
    results.mkdir()
    for name in ("model_p2.json", "summary_p2.json"):
        shutil.copy(RESULTS_DIR / name, results / name)
    ledger, docs = tmp_path / "track_record.csv", tmp_path / "docs"
    now = f"{_last_session():%Y-%m-%d} 17:00"
    args = ["--no-update", "--now", now, "--ledger", str(ledger), "--docs-dir", str(docs),
            "--results-dir", str(results)]
    assert DY.main(args) == 0
    out = capsys.readouterr().out
    assert "::warning::" in out and "model_p3.json" in out
    df = pd.read_csv(ledger, dtype=str)
    assert len(df) == 1 and pd.isna(df.loc[0, "p3_w_reason"])          # P3 열은 비어 있다(스키마는 3)
    assert DY.main(args + ["--require-p3"]) == 1


@pytest.mark.skipif(not (RESULTS_DIR / "model_p3.json").exists(), reason="results/model_p3.json 없음")
def test_daily_github_output_carries_p3_fields(tmp_path, monkeypatch):
    """GitHub 스텝 출력에 p3_w_exec · p3_reason · kill_state · alarms · effective_mode 가 추가된다(§11)."""
    from mrl import sizing as SZ
    from scripts import daily as DY
    results = _daily_results(tmp_path, RESULTS_DIR)
    ledger, docs = tmp_path / "track_record.csv", tmp_path / "docs"
    gh = tmp_path / "gh_output.txt"
    gh.write_text("", encoding="utf-8")
    monkeypatch.setenv("GITHUB_OUTPUT", str(gh))
    now = f"{_last_session():%Y-%m-%d} 17:00"
    assert DY.main(["--no-update", "--now", now, "--ledger", str(ledger), "--docs-dir", str(docs),
                    "--results-dir", str(results)]) == 0
    kv = dict(line.split("=", 1) for line in gh.read_text(encoding="utf-8").splitlines() if "=" in line)
    for k in ("asof", "market_status", "appended", "p3_w_exec", "p3_reason", "kill_state", "alarms",
              "effective_mode"):
        assert k in kv, k
    assert kv["p3_reason"] in ("", *SZ.REASONS)
    assert kv["effective_mode"] in ("info_only", "tones")
    assert kv["kill_state"] in ("", "not_started", "not_due", "provisional", "validated", "info_only", "manual_kill")


# ------------------------------------------------------------------
# selftest.py — Phase 3 절
# ------------------------------------------------------------------
def test_selftest_p3_section(p3_short, tmp_path, capsys):
    """§11 selftest: sha 일치·θ guard·D_max·자기검사·kill 정합에서 FAIL 0, sizing_sha 를 틀면 FAIL."""
    from scripts import selftest as ST
    results = p3_short["results"]
    rep = ST.Report()
    ST.run_p3(rep, DATA_DIR, results, tmp_path / "no_ledger.csv")
    fails = [ln for ln in rep.lines if ln.startswith("[FAIL]")]
    assert not fails, fails
    txt = "\n".join(rep.lines)
    for key in ("smoothed_probabilities", "registry_sha", "sizing_sha", "D_max", "θ guard 재검사",
                "신선 블록", "장부 계약"):
        assert key in txt, key

    stale = tmp_path / "stale"
    stale.mkdir()
    m3 = json.loads((results / "model_p3.json").read_text(encoding="utf-8"))
    m3["sizing_sha"] = "f" * 64
    (stale / "model_p3.json").write_text(json.dumps(m3, ensure_ascii=False), encoding="utf-8")
    rep2 = ST.Report()
    ST.run_p3(rep2, DATA_DIR, stale, tmp_path / "no_ledger.csv")
    assert rep2.n_fail >= 1 and any("sizing_sha" in ln and ln.startswith("[FAIL]") for ln in rep2.lines)

    empty = tmp_path / "empty"
    empty.mkdir()
    rep3 = ST.Report()
    ST.run_p3(rep3, DATA_DIR, empty, tmp_path / "no_ledger.csv")
    assert rep3.n_fail == 0 and any("model_p3.json 없음" in ln for ln in rep3.lines)   # 미실행은 계약 위반이 아니다


def test_selftest_p3_kill_record_must_force_info_only(p3_short, tmp_path):
    """kill_record.json 이 있는데 model_p3.deploy_mode 가 tones 면 FAIL (킬은 sticky)."""
    from scripts import selftest as ST
    bad = tmp_path / "bad"
    bad.mkdir()
    m3 = json.loads((p3_short["results"] / "model_p3.json").read_text(encoding="utf-8"))
    m3["deploy_mode"] = "tones"
    (bad / "model_p3.json").write_text(json.dumps(m3, ensure_ascii=False), encoding="utf-8")
    (bad / "kill_record.json").write_text(json.dumps({"asof": "2026-01-02", "ledger_entry": "7x"}), encoding="utf-8")
    rep = ST.Report()
    ST.run_p3(rep, DATA_DIR, bad, tmp_path / "no_ledger.csv")
    assert any("sticky" in ln and ln.startswith("[FAIL]") for ln in rep.lines), rep.lines


def test_smoothed_probabilities_is_never_called_outside_regime():
    """§2 점 원칙: 평활 확률은 미래를 읽는다 — regime.py 밖에서 호출되면 안 된다(selftest 의 grep 계약)."""
    from scripts import selftest as ST
    assert ST._grep_smoothed(ROOT) == []


def test_selftest_cli_runs_all_three_sections(capsys):
    """CLI 기본 실행이 캐시·P2·P3 세 절을 모두 돌고 FAIL 0 으로 끝난다(--no-p3 는 P3 절만 끈다)."""
    from scripts import selftest as ST
    assert ST.main(["--max-age-days", str(10 ** 6)]) == 0
    out = capsys.readouterr().out
    assert "Phase 2 자기점검" in out and "Phase 3 자기점검" in out and "결과: FAIL 0" in out
    assert ST.main(["--max-age-days", str(10 ** 6), "--no-p3"]) == 0
    out2 = capsys.readouterr().out
    assert "Phase 3 자기점검" not in out2


# ------------------------------------------------------------------
# 워크플로 (§12)
# ------------------------------------------------------------------
def _load_yaml(name: str) -> dict:
    yaml = pytest.importorskip("yaml", reason="PyYAML 없음")
    return yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8"))


def test_weekly_workflow_runs_phase3_after_backtest_v1_inside_the_gate():
    """§12: run_phase3.py 는 run_backtest_v1.py **뒤**, 결정론 게이트·커밋 **앞**에 온다. --all-sensitivities 를 쓰고
    --holdout-final 은 워크플로에 없다. timeout-minutes 60 유지."""
    d = _load_yaml("weekly.yml")
    job = d["jobs"]["rebuild"]
    assert job["timeout-minutes"] == 60
    steps = job["steps"]
    runs = [str(s.get("run") or "") for s in steps]
    def _idx(token):
        return next(i for i, r in enumerate(runs) if token in r)
    i_v1 = _idx("scripts/run_backtest_v1.py")
    i_p3 = _idx("scripts/run_phase3.py")
    i_gate = next(i for i, s in enumerate(steps) if "결정론 게이트" in str(s.get("name") or ""))
    i_commit = next(i for i, s in enumerate(steps) if "커밋 및 푸시" in str(s.get("name") or ""))
    assert i_v1 < i_p3 < i_gate < i_commit
    assert "--all-sensitivities" in runs[i_p3] and "--holdout-final" not in runs[i_p3]
    gate = runs[i_gate]
    assert "summary_p3.json" in gate and "determinism" in gate
    for k in ("hmm_param_count_ok", "production_shadow_only", "pit_bit_identical"):
        assert k in gate, k


def test_daily_workflow_is_unchanged_and_parses():
    """§12: daily.yml 은 그대로 (daily.py 만 확장). 두 워크플로 YAML 이 파이썬 yaml 로 파싱된다."""
    d = _load_yaml("daily.yml")
    assert set(d["jobs"]) == {"gate", "run"}
    runs = [str(s.get("run") or "") for s in d["jobs"]["run"]["steps"]]
    assert any("scripts/daily.py" in r for r in runs)
    assert not any("run_phase3" in r for r in runs)


def test_workflows_have_no_crlf_in_weekly():
    """UTF-8 LF (프로젝트 규약)."""
    assert b"\r\n" not in (ROOT / ".github" / "workflows" / "weekly.yml").read_bytes()


# ==================================================================
# 회귀 — 검토에서 확인된 결함
# ==================================================================
def test_holdout_refusal_precedes_every_disclosure(p3_short, tmp_path, capsys):
    """§11: --holdout-final 은 unlock 전제를 **자료를 만지기 전에** 본다.

    전에는 하드컷·HMM 재적합·채택 검정·비중 헤드라인을 전부 찍고 로그로 흘린 뒤에야 거부했다 —
    거부한 실행이 이미 홀드아웃이 섞인 숫자를 공개한 것이다.
    """
    import shutil
    from scripts import run_phase3 as RP3
    dst = tmp_path / "results"
    shutil.copytree(p3_short["results"], dst)
    assert not (dst / "holdout_unlock.json").exists()
    capsys.readouterr()
    rc = RP3.main(["--holdout-final", "--end", "2025-06-30", "--results-dir", str(dst),
                   "--docs-dir", str(tmp_path / "docs"), "--no-charts", "--no-docs"])
    out = capsys.readouterr().out
    assert rc == 2, out
    for leaked in ("하드컷", "HMM 재적합", "비중 σ_T", "채택 검정"):
        assert leaked not in out, f"거부 전에 {leaked!r} 를 공개했다:\n{out}"


def test_recompute_frame_survives_kill_and_budget_change(p3_short, tmp_path):
    """D7: 되감기는 오늘의 모드·예산이 아니라 **그날 장부가 적어 둔 것**으로 한다.

    킬(tones→info_only)이나 D_max 변경이 D7 오경보가 되면 daily 가 오늘 행을 쓰지 못하고,
    같은 20행이 남아 매일 같은 실패가 되풀이된다(킬이 걸린 그날 daily 가 영구히 죽는다).
    """
    import math as _m
    from mrl import data as D
    from mrl import sizing as SZ
    from mrl import track as TR
    from scripts import daily as DY

    spy = D.apply_guards(D.load_cache(DATA_DIR)).spy_ohlc["Close"].astype(float)
    idx = spy.index[-25:]
    sig = SZ.ewma_vol(spy).reindex(idx)
    week_end = SZ.execute(pd.Series(0.5, index=idx), pd.Series(1.0, index=idx))["week_end"].to_numpy(bool)

    def ledger(states, sigma_ts, deploys):
        """daily.py 와 같은 방식(SZ.step 한 행씩)으로 '진짜 장부' 를 만든다."""
        rows, pw, pm = [], _m.nan, _m.nan
        for i, d in enumerate(idx):
            mult = SZ.state_multiplier(states[i])
            wt = float(SZ.exposure_target(float(sig.loc[d]), mult, sigma_ts[i])["w_target"].iloc[0])
            w, why = SZ.step(wt, mult, pm, pw, bool(week_end[i]), deploy=deploys[i])
            rows.append({"asof": d.strftime("%Y-%m-%d"), "p2_state": states[i], "p3_sigma_target": sigma_ts[i],
                         "p3_state_mult": mult, "p3_w_exec": w, "p3_w_reason": why})
            if _m.isfinite(w):
                pw = w
            if _m.isfinite(mult):
                pm = mult
        return pd.DataFrame(rows)

    def mismatches(led, sigma_today, deploy_today):
        rec = DY._recompute_frame(led, pd.DataFrame(index=pd.DatetimeIndex([])), spy,
                                  pd.DataFrame(index=idx), [], None, None,
                                  sigma_today, deploy_today, [], sessions=20)
        return TR.replay_check(led, rec, cols=("p3_w_exec",), tol=0.01, sessions=20)

    n = len(idx)
    normal = ["normal"] * n
    # (대조군) 설정 그대로 — 회귀 방지
    assert mismatches(ledger(normal, [0.10] * n, [True] * n), 0.10, True)["n_mismatch"] == 0
    # (a) 킬이 발동해 오늘부터 info_only
    assert mismatches(ledger(normal, [0.10] * n, [True] * 19 + [False] * 6), 0.10, False)["n_mismatch"] == 0
    # (b) 소유자가 D_max 0.35 → 0.30 (σ_T 0.10 → 0.08) 로 낮춘 장부 기록
    assert mismatches(ledger(normal, [0.10] * 20 + [0.08] * 5, [True] * n), 0.08, True)["n_mismatch"] == 0
    # (c) 상태 변화 + 킬이 섞인 창
    mixed = ["normal"] * 10 + ["caution"] * 5 + ["reduce"] * 10
    assert mismatches(ledger(mixed, [0.10] * n, [True] * 15 + [False] * 10), 0.10, False)["n_mismatch"] == 0
    # (음성 대조군) 장부 한 행을 조작하면 **여전히** 잡아야 한다 — 검사에 이빨이 남아 있는지
    led = ledger(normal, [0.10] * n, [True] * n)
    rec = DY._recompute_frame(led, pd.DataFrame(index=pd.DatetimeIndex([])), spy, pd.DataFrame(index=idx),
                              [], None, None, 0.10, True, [], sessions=20)
    led.loc[led.index[-3], "p3_w_exec"] = 0.30
    assert TR.replay_check(led, rec, cols=("p3_w_exec",), tol=0.01, sessions=20)["n_mismatch"] == 1


def test_build_model_p3_keeps_admission_across_weekly_rebuild(p3_short, tmp_path):
    """§5.4: 번호 붙인 장부 항목으로 채택된 멤버를 주간 재적합이 조용히 되돌리면 안 된다."""
    import shutil
    from mrl import ensemble as EN
    from scripts import run_phase3 as RP3
    dst = tmp_path / "results"
    shutil.copytree(p3_short["results"], dst)
    m3 = json.loads((dst / "model_p3.json").read_text(encoding="utf-8"))
    m3["members"]["H"] = {"status": "admitted", "K_s": 2, "K_u": 12, "ledger_no": "#8x",
                          "effective_refit": "2030-01-02", "admitted_on": "2029-11-30",
                          "admission": {"verdict": "ADMIT", "wins": 9, "n_blocks": 11}}
    (dst / "model_p3.json").write_text(json.dumps(m3, ensure_ascii=False), encoding="utf-8")
    warns: list[str] = []
    out = RP3.build_model_p3(dst, "r" * 64, "s" * 64, {}, {"retention": {"deploy_sizing": True},
                             "d_max": 0.35, "sigma_target": 0.10}, {}, {}, "d" * 64, "sp" * 32, warns)
    assert out["members"]["H"]["status"] == "admitted"
    assert out["members"]["H"]["effective_refit"] == "2030-01-02"
    assert out["members"]["H"]["ledger_no"] == "#8x"
    assert "effective_from" not in out["members"]["H"]           # 아무도 쓰지 않던 죽은 키
    # §5.4 1월 발효 게이트가 재조립 뒤에도 그대로
    assert EN.effective_statuses(out, "2029-12-31")["H"] == "shadow"
    assert EN.effective_statuses(out, "2030-01-05")["H"] == "admitted"


def test_build_model_p3_refuses_malformed_admission_loudly(p3_short, tmp_path):
    """effective_refit 없는 admitted 를 이월하면 effective_statuses 가 ValueError → daily 가 죽는다.
    이월하지 않고 **경고와 함께** 1일차 상태로 둔다(조용한 실패 금지)."""
    import shutil
    from mrl import ensemble as EN
    from scripts import run_phase3 as RP3
    dst = tmp_path / "results"
    shutil.copytree(p3_short["results"], dst)
    m3 = json.loads((dst / "model_p3.json").read_text(encoding="utf-8"))
    m3["members"]["H"] = {"status": "admitted"}                  # effective_refit 없음
    (dst / "model_p3.json").write_text(json.dumps(m3, ensure_ascii=False), encoding="utf-8")
    warns: list[str] = []
    out = RP3.build_model_p3(dst, "r" * 64, "s" * 64, {}, {"retention": {"deploy_sizing": True},
                             "d_max": 0.35, "sigma_target": 0.10}, {}, {}, "d" * 64, "sp" * 32, warns)
    assert out["members"]["H"]["status"] == "shadow"
    assert any("effective_refit" in w for w in warns), warns
    EN.effective_statuses(out, "2030-01-05")                     # 죽지 않는다


def test_weekly_track_block_is_info_only_when_kill_record_exists(p3_short, tmp_path):
    """§8.3: 킬 기록이 있으면 요약·트랙 블록·track_record_p3.json 이 **모두** info_only 여야 한다.

    전에는 track 호출부가 p3_deploy_mode="tones" 리터럴을 넘겨 AND 에서 p3 항이 사라졌고,
    prev_kill 도 넘기지 않아 주간 재적합이 킬을 풀 수 있었다.
    """
    import shutil
    from scripts import run_phase3 as RP3
    dst = tmp_path / "results"
    shutil.copytree(p3_short["results"], dst)
    (dst / "kill_record.json").write_text(json.dumps({"killed": True, "ledger_entry": "7x"}), encoding="utf-8")
    rc = RP3.main(["--end", SHORT_END, "--results-dir", str(dst), "--docs-dir", str(tmp_path / "docs"),
                   "--no-charts", "--no-docs"])
    assert rc == 0
    s = json.loads((dst / "summary_p3.json").read_text(encoding="utf-8"))
    assert s["run"]["effective_mode"] == "info_only"
    assert s["run"]["p3_deploy_mode"] == "info_only"
    if s.get("track"):
        assert s["track"]["effective_mode"] == "info_only"
    tp = dst / "track_record_p3.json"
    if tp.exists():
        assert json.loads(tp.read_text(encoding="utf-8"))["effective_mode"] == "info_only"
