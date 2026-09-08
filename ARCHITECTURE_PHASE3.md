# market-risk-lab — Phase 3 모듈 계약 (국면 엔진 `regime` / 등록부 `ensemble` / 비중 `sizing` / 시나리오 `scenarios` / 트랙레코드·킬룰 `track`)

목적: Phase 2 가 만든 보정 확률 `p2`(4-파라미터 중첩 로지스틱 M3, 또는 `acceptance()` 가 배치한 단)와 3단계 결정층을 **한 글자도 바꾸지 않고**, 그 위에 (1) 가족 위험예산에서 나오는 **적합 파라미터 0개**의 변동성 목표 비중 규칙, (2) 확률 구간·결정 상태·에피소드 표에서 **세기만 한** 조건부 시나리오와 시장 내재 20일 범위, (3) 라이브 장부의 트랙레코드 패널(소표본 문구가 코드로 고정된)과 수치 드리프트 경보, (4) VALIDATION §7 킬룰의 실행 가능한 코드화, (5) 2-상태 가우시안 HMM(직접 구현한 numpy EM, 연 1회 퍼지 재적합)을 **그림자(shadow) 멤버**로 매일 기록·채점하는 사전 등록 멤버 등록부를 쌓는다. 생산 확률은 Phase 3 에서 바뀌지 않는다: 어떤 후보도 보정 VIX 위에 측정 가능한 Brier 이득을 주지 않았고(§0), 그래서 이 단계가 사는 것은 skill 이 아니라 **비중·시나리오·구간·트랙레코드·출구**다.
v0(`reference/`, `mrl/signals_v0.py`, `mrl/config.V0*`)와 Phase 2 모듈(`features.py`, `vol.py`, `model.py`, `calibrate.py`, `decision.py`, `events.py`)은 한 줄도 건드리지 않는다. 확장은 `config.py`·`ledger.py`·`report.py`·`evaluate.py`·`daily.py`·`selftest.py` 에 **추가**만 한다. v0 결과는 모든 리포트 첫 줄에 영구 표기한다(VALIDATION §4).

## 0. 설계 채택과 접목 (심사 결과의 해석)

세 설계의 심사 합계: **P3-B 116**(34/42/40) · P3-C 97(31/31/35) · P3-A 94(30/32/32). 기반 설계 = **P3-B "예산 우선 비중"** (p2 동결, 변동성 목표 × 상태 배수, 세는 시나리오, 라이브 패널, §7 코드화, HMM 은 진단). 세 심사가 독립 스크래치 재현으로 세 설계의 수치를 소수 셋째 자리까지 확인했으므로(하드컷 2024-08-30, 홀드아웃 미접근) 아래 수치는 신뢰할 수 있는 **설계 단계 근사**이며, 공식 수치는 `run_phase3.py` 산출로 대체한다(근사와 공식을 섞지 않는다).

심사 공통 결정 사실: (a) 어떤 확률 후보도 M3 위에 측정 가능한 skill 을 더하지 못한다 — 앙상블 p3_A(M3+M1+HMM) 손실차 −0.00024 [CI −0.00251, +0.00183], ens(M3,H) dBrier +0.0000(6/11 블록), 5번째 슬롯 M4a b4≈0.01(3/11 블록), 폭/신용 멤버 BSS −0.055. (b) 가족의 낙폭을 실제로 바꾸는 것은 비중 규칙이며, 변동성 일치 기준으로 파라미터 0개의 변동성 목표(VT14)가 p2 결정층을 재현한다(10.14% / −30.3% vs 10.19% / −31.0%). (c) 모든 변동성 목표 규칙은 2000~02 완만한 약세장에서 예산의 3.0~3.5배를 잃는다. (d) 문자 그대로의 §7 '늦은 쪽'은 8회 도달까지 중앙 ~8년이다.

Phase 2 의 공식 결과(VALIDATION #2, `run_calibration.py` 2026-09-07): M3 Brier 0.1210 · BSS_clim +0.091 · vs B1 +0.200 · vs M1 +0.012 · AUC 0.687; 24개월 블록 기후학 대비 10/11(2019-20 −0.037), B1 대비 9/11 → **§6 문자 그대로 실패 → `model_p2.json` deploy_mode=info_only, tone_model=None**; #2a 완화안(사후)으로는 M1 만 A∧B∧C 통과. 따라서 Phase 3 의 seed 는 "소유자가 #2a 를 기재하면 M1, 아니면 없음(M3 를 그림자 표시)" 이며, 유효 배치 모드가 info_only 인 동안 비중·상태 카드는 뜨지 않는다(§2·§15). 이 문서의 설계 단계 비중 수치는 M3 상태로 계산된 것이라 공식 수치(배치 단의 상태)와 조금 다를 수 있다.

| 항목 | P3-B(기반) | 접목/충돌 | 최종 결정 |
|---|---|---|---|
| 생산 확률 | p2 동결(M3 또는 배치 단), 새 적합 0 | A: 8-파라미터 동일가중 앙상블 p3 + §6 사후 개정 · C: 등록부, 신선 자료로만 채택 | **B.** 생산 확률 = p2 그대로. C 의 등록부·채택 규칙을 **그림자 전용**으로 접목(§5): 관측 기록·홀드아웃은 기각만, 채택은 라이브 장부의 3개월 신선 블록으로만. §6-c 개정 문구는 지금 쓰되 채택 시에만 발효. A 의 앙상블은 Δ −0.0002 에 사후 개정이므로 기각 |
| HMM | 일변량 수익률, 1993~2014 동결, 게이지 전용 | A: [r, ln σ_GK+OV] 대각, 1월 퍼지 재적합·warm start·guard·q20 · C: [100·r, ln RV10] 완전공분산, 재적합, Platt 그림자 멤버 | **C 관측·완전공분산**(종가만 써서 1993~95 OHLC 품질 문제에 면역) + **A 의 퍼지 재적합·guard·PIT·누수 카나리 테스트** + **C/J 의 그림자 멤버**(Platt 2, 매일 장부 기록·채점). k-스텝 전이확률(p20·q20)은 A 식으로 계산·표시하되 Platt 특징은 `logit P_high`(C) 하나로 고정 |
| 5번째 슬롯 M4a(#3) | 유일한 승격 경로 | A 실측: b4≈0.01, BSS 0.0901 vs 0.0910, CI [−0.0005, +0.0003] | **#3 닫힘(무이득·미채택)**. 승격 경로는 그림자 멤버 → 신선 자료 채택(#3a)뿐 |
| 폭/신용 멤버 B(HYG/IEF·RSP/SPY) | 없음 | C: 2007+ 그림자 멤버 | **미등록.** 단독 BSS −0.055·AUC 0.495(2010~24)·사상 불안정; 구간을 잡음으로 넓힌다(J2·J3). #6 예약 후보로 수치만 기록 |
| 불일치 구간 | 없음(p2 밴드만) | C: 등록 멤버 min/max ∪ p2 밴드 | **C 접목(표시 전용).** 멤버 {p2, M1, H}; 폭 > 0.20 이 5세션 이상이면 '멤버 불일치' 플래그 |
| 예산 → 목표 변동성 | σ_T = D_max/2.0 (급락형 실측 1.5~1.8) | A: D_max/3.5 를 격자 {6,8,10,12,15}% 로 내림(1993+ 실측 3.1~3.5) · C: k=3.3 | **헤드라인 k_slow = 3.5**(실측 최악 3.46) + **급락형 k_fast = 2.0 을 낙관선으로 병기**, A 격자. 기본 D_max = 35% → σ_T 10%. 사다리는 두 창(1993~ 변동성 단독·2003~ 결정층 포함)의 MaxDD 를 나란히 |
| 비중 결합 형태 | 곱 w_vol × m(1/.5/.25), 재클립 없음(비중 5~10% 가능) | A: min(w_vol, 상태 상한) · C: 곱 × (1/.75/.5), [0.25, 1] 재클립 | **곱 × TONE_EXPOSURE, 곱 뒤 [0.25, 1] 재클립**(J3). 바닥 0.25 는 v0·p2 의 'reduce' 바닥과 같은 숫자라 새 상수가 아니며, 재클립 없는 곱은 등록된 어떤 배분표에도 없는 비중 0.0625~0.125 를 만든다. 가격을 인쇄한다: MaxDD −17.9%(바닥 없음) → −23.5%(바닥). 바닥 없음·min 규칙·1/.75/.5 는 민감도 행 |
| 리듬·churn | 주 마지막 세션 + 격상 즉시, 5pp 격자, 10pp 밴드(11.4회/년) | A: 0.25 격자 + 히스테리시스 0.175(3.6~6회) · C: 월초 + 상태 변경(7~8회) | **B**(J1·J2 다수). C 월간·A 격자는 민감도 행. 비중 churn 상한 24회/252세션(D6) |
| 변동성 예측(비중용) | EWMA λ=0.94 | A·C 동일 | **EWMA94.** 적합 HAR(OLS 4, 예산 밖)은 표시·민감도 S-HAR 전용 |
| 킬룰 시점 | 문자 그대로 '둘 다' + 60개월 상한(#7) | A: 2단계(36개월 1차·sticky·provisional) · C: 문자 그대로 + 중간 중대실패 조항 | **A 2단계.** 1차 = 36개월(점추정 ≤ 0 → info_only, sticky; > 0 → provisional). 2차 = 36개월 이후 8회 도달 또는 **60개월 중 먼저**(B 상한을 외곽으로 유지), 이후 12개월마다. C 의 중간 조항(BSS ≤ −0.05 ∧ CI 상한 ≤ 0)은 1차보다 약해 흡수됨 — 12·24개월 점검에서 '중대 실패' 배지로만 표시(§7 의 3년 하한 때문에 킬 불가) |
| 킬룰 판정 | 점추정 > 0, CI 는 라벨 | A: validated/provisional/info_only 라벨 + 주간 재현 자기검사 · C: 멤버별 판정 | **셋 다.** 점추정 트리거; 라벨; 그림자 멤버도 같은 통계로 판정(정보); 주간 자기검사가 과거 36개월 창의 오기각 비중(~12%)과 검정력 문구를 **재생성**(J1 정정: 무정보 모델은 동전 던지기가 아니라 ~78% 기각) |
| 신선 블록 시작 | — | C: 2024-09-01(홀드아웃) | **라이브 장부 시작일**(세 심사 일치). 홀드아웃은 #2b 1회 관측; H 의 홀드아웃 점수는 정보로만 기록 |
| 시나리오 표시 | 위에서부터 n_eff ≥ 20 까지 풀링 | A: n_eff < 5 회색 · C: < 10 회색, 3줄 스트립 순서 | **B 풀링** + **C 의 3줄 순서**(시장 범위 → 오늘 같은 날 → 에피소드가 시작되면) + **A 의 문장**('높은 구간의 수익 분포는 좁아지지 않고 넓어진다') |
| 드리프트 경보 | D1~D9(p5/p95 참조분포) | A: 주간 재현 |Δp| > 1e-6 + 특징 범위 이탈 | **B + D10(재현) + D11(특징 범위)** |
| 모듈 이름 | drift.py · kill.py · regime.py(게이지) | 과제 계약: regime · ensemble · sizing · scenarios · track | **과제 계약대로.** `track.py` = 라이브 채점 + 경보 + 킬룰; `ensemble.py` = 등록부 + 결합 + 채택 + 구간 |
| 결정론 허용오차(HMM) | 1e-9 | J2: 러너 BLAS 차이 → 1e-7 | **θ 1e-7·확률 1e-7**, 장부에 기록 |
| Phase 2 가 info_only 로 착지하면 | 비중 제안 없음, w_vol 은 회색 기록 | — | **B**(지금 등록해 결과를 보고 정하지 않음) |
| 카드 첫 줄 | 다음 임계 문장 | A: 숫자 하나 + 문장 하나 | **A 첫 줄 + B 둘째 줄** |

## 1. 디렉터리 변경

```
market-risk-lab/
  mrl/
    regime.py       2-상태 가우시안 HMM: 관측 · 결정적 초기화 · numpy EM(스케일드 전방-후방, einsum ξ) · 전방 필터 · k-스텝 전이확률 · 퍼지 walk-forward · guard (신규)
    ensemble.py     멤버 등록부(REGISTRY) · Platt 그림자 멤버 · 동일가중 결합 · 불일치 구간 · 채택 규칙(신선 블록) · 검정력 표 (신규)
    sizing.py       EWMA 변동성 · 예산→σ_T 사다리 · 비중 목표/실행(격자·밴드·주간·격상) · 백테스트 표 · 민감도 · 유지 조건 (신규)
    scenarios.py    확률 구간 표(풀링) · 결정 상태 표 · 에피소드 조건부 · 시장 내재/HAR 20일 범위 · 오늘 문맥 문장 (신규)
    track.py        라이브 창 채점 · 과거 동일 길이 창 백분위 · 드리프트 경보 D1~D11 · 킬룰(상태·적용·재현) · 지평별 문구 규칙 (신규)
    ledger.py       P3 열 · schema_version 3 · backfill 확장 · summary().p3 (확장)
    report.py       p3_card · render_sizing_report · render_regime_report · render_track_record · charts_p3 (확장)
    evaluate.py     allocation_from_weights · window_distribution (추가)
    config.py       P3 · HMM_P3 · ENSEMBLE_P3 · KILL_P3 · P3_DRIFT · SCENARIO_P3 · 경로 (추가; v0·P2 블록 불변)
  scripts/
    run_phase3.py   주간: HMM walk-forward → 그림자 멤버 → 채택 검정 → 비중 백테스트 → 시나리오 표 → 킬룰 재현 → results/·docs/ (신규)
    daily.py        P2 블록 뒤에 P3 블록: σ̂ → 비중 → HMM 필터 → 구간 → 시나리오 문장 → 경보 → 킬룰 → 장부 → 카드 (확장; 재적합 금지)
    selftest.py     등록부·비중 규칙 해시, θ guard, 장부 P3 결측, 경보 파일, kill_record 정합 (확장)
  results/
    oos_p3.csv · summary_p3.json · model_p3.json · hmm_p3.json · backtest_p3.csv · track_record_p3.json · alarms.csv
    kill_record.json (킬 발동 후에만) · kill_manual.json (소유자 수동 킬) · track_record.csv (P3 열 추가)
  docs/
    sizing_p3.html · regime_p3.html · track_record.html · index.html (P3 카드)
  tests/
    test_regime.py · test_ensemble.py · test_sizing.py · test_scenarios.py · test_track.py (신규)
    test_ledger.py · test_report.py · test_scripts_integration.py (확장)
```

## 2. 공통 규칙 추가

* **파라미터 예산(VALIDATION §6)**: 생산 확률의 적합 파라미터는 Phase 2 그대로 **4개**(b0, b1, b2, b3), 5번째 슬롯은 비어 있다. Phase 3 는 생산 확률에 적합값을 **0개** 더한다. `mrl.model.PARAM_COUNT == 4` 는 그대로 assert 된다. Phase 3 가 만드는 적합값은 전부 예산 밖이며 카드 정직 스트립에 줄별로 공개한다: 그림자 멤버 H 의 Platt 2(K_s, 지도)·HMM θ 12(K_u, 비지도), Phase 2 부터 있던 HAR OLS 4·v0 Platt 2. 비중·시나리오·킬룰·경보는 적합 파라미터 0개여야 하고(§6-b), 그 상수들은 1993~2024-08 기록으로 ~40 변형을 본 뒤 골랐으므로 그 기록에 대해 **post hoc** 이다 — 지금 동결하고 라이브 장부로 재조정하지 않는다.
* **고정 상수(적합 아님, 사전 등록)**: EWMA λ=0.94·seed 60세션, 비중 하한 0.25·상한 1.0, k_slow 3.5·k_fast 2.0, σ_T 격자 {6,8,10,12,15}%, 기본 D_max 0.35, 격자 0.05, 무거래 밴드 0.10, 주간 리듬·격상 즉시, 상태 배수 = `TONE_EXPOSURE[STATE_TO_TONE[state]]`(새 숫자 없음), 비용 5bp·현금 0%; HMM K=2·완전공분산·초기 A·tol·jitter·guard 경계·클립 1e-4·turn_h 20; 등록부 채택 8/11·신선 블록 3개월·11개; 킬룰 36/8/60/12·블록 40·4,000회·seed 0; 경보 임계 P3_DRIFT; 시나리오 구간 = `P2["reliability_bins"]`·풀링 n_eff ≥ 20·z 1.2816/1.645.
* **홀드아웃 하드컷**: `run_phase3.py` 는 `--holdout-final` 없이는 Phase 2 와 같은 방식으로 **모든 입력 프레임**을 2024-08-30 에서 자른다. 신선 블록(§5.5)은 **라이브 장부 시작일부터** 센다 — 홀드아웃 행은 채택에 절대 쓰지 않는다(§6 홀드아웃 1회 규칙의 두 번째 접근이 되기 때문).
* **점(point-in-time)**: HMM 필터는 전방(forward)만 쓴다 — 평활(smoothing)은 미래를 읽는다. `tests/test_regime.py::test_filter_point_in_time` 이 20개 무작위 T 에서 비트 동일성을, `test_leakage_canary` 가 평활 확률이 필터보다 **높게** 채점됨(=필터가 실제로 돌고 있음)을 검사한다. 비중 w_exec_t 는 t 종가에 확정되어 t+1 수익률에 적용된다.
* **결정론**: 난수는 전부 seed 0. HMM EM 은 결정적 초기화·warm start 이며 난수 재시작이 없다. 주간 작업은 새 θ·Platt·비중 경로와 저장된 `hmm_p3.json`·`model_p3.json`·`backtest_p3.csv` 를 허용오차 **1e-7**(θ·확률; 러너와 로컬의 BLAS 차이)·CSV 해시 동일로 비교하고 다르면 실패(커밋 없음). `registry_sha256()`(ensemble.py+regime.py 소스+REGISTRY 튜플)와 `sizing_sha256()`(sizing.py 소스+SIZING_RULE)이 모든 산출물에 기록되며 `daily.py` 는 불일치 시 exit 1(코드가 산출물보다 새롭다).
* **재적합·채택은 주간 작업에서만**: `daily.py` 는 `model_p3.json`·`hmm_p3.json` 을 읽기만 한다. 등록부 상태 변경은 다음 1월 재적합부터 발효한다.
* **조용한 실패 금지**: 입력 결측(σ̂·상태·θ 결측)이면 비중은 **직전 값 유지**(reason `input_missing`; 조용히 1.0 으로 돌아가지 않는다), HMM 확률은 NaN, 구간은 p2 밴드만, 카드에 사유, 장부 `p3_input_missing`.
* **계약 열(Phase 2 와의 결합)**: `results/calib_p2_walkforward.csv`(p_m1, p_m3, clim, y), `summary_p2.json`(acceptance.tone_model, deploy_mode, spec_sha256, params_by_refit, reliability), `model_p2.json`, `backtest_v1.csv`(state), 장부 열 `prob_dd5_20`·`p2_p_m1`·`p2_p_vix`·`p2_clim`·`p2_state`·`p2_days_in_state`·`p2_har_fc_20`·`p2_deploy_mode`·`p2_tone_model`·`p2_lo`·`p2_hi`. spec_sha256 불일치는 exit 1 — 대체 없음.
* **사전 관측 공개**: 세 설계와 세 심사는 2003~2024-08 OOS 기록과 1993~2024 비중 백테스트를 이미 보았다. 이 문서 이후의 어떤 상수·규칙 변경도 그 기록에 대해 post hoc 이며 장부에 그렇게 적는다. 오염되지 않은 자료는 홀드아웃(#2b 1회)과 라이브 장부뿐이다.

## 3. 설정 — `mrl/config.py` (추가; v0·P2 블록 불변)

```python
# ------------------------------------------------------------------
# Phase 3 — 비중 p3 / 그림자 등록부 / 킬룰 (ARCHITECTURE_PHASE3.md §3; v0·P2 블록 불변)
# ------------------------------------------------------------------
P3 = {
    "sigma_model": "ewma", "ewma_lambda": 0.94, "ewma_seed_sessions": 60,   # RiskMetrics 상수 — 적합 아님
    "sigma_sensitivity": "har_fc_20",                                        # 적합 HAR 예측은 표시·민감도 S-HAR 전용
    "k_slow": 3.5, "k_fast": 2.0,                                            # σ_T = D_max/k_slow (헤드라인) ; k_fast 는 낙관선 병기
    "vol_grid": (0.06, 0.08, 0.10, 0.12, 0.15),                              # σ_T 격자 (내림)
    "d_max_default": 0.35,                                                   # 기본 가족 예산 → σ_T 10%
    "w_min": 0.25, "w_max": 1.0, "grid": 0.05, "band": 0.10,                 # 바닥/상한/격자/무거래 밴드
    "floor_after_multiplier": True,                                          # 곱 뒤 [w_min, w_max] 재클립
    "cadence": "weekly", "escalate_immediately": True,                       # 주 마지막 세션 점검 + 결정층 격상 시 즉시 하향
    "cost_bps": 5, "cash_return": 0.0,
    "eval_start": "2003-01-02", "vol_only_start": "1993-10-14", "v0_start": BACKTEST_START,
    "reference_windows": (63, 126, 252, 756), "churn_ceiling": 24,          # 비중 변경 상한(252세션)
}
HMM_P3 = {
    "obs": ("r100", "ln_rv10"), "rv_window": 10, "k_states": 2, "cov": "full",
    "init_A": ((0.98, 0.02), (0.05, 0.95)), "tol": 1e-6, "max_iter_first": 300, "max_iter_refit": 100,
    "cov_jitter": 1e-6, "clip": 1e-4, "param_count": 12,                     # μ 2×2 + Σ 2×3 + A 2 (π = 정상분포, 자유도 0)
    "guard": {"p_min": 0.90, "p_max": 0.999, "gap_lnvol": 0.3, "occ_min": 0.05},
    "turn_h": 20, "platt_C": 1.0, "train_start": P2["train_start"], "purge": P2["purge"],
    "first_refit": P2["first_refit"], "tol_theta": 1e-7, "tol_prob": 1e-7,
}
ENSEMBLE_P3 = {
    "members": ("p2", "M1", "H"), "member_order": ("H", "M1"),               # 채택 검정 순서(고정, 재배열 금지)
    "fresh_block_months": 3, "fresh_blocks_min": 11, "admit_frac": 8 / 11, "reeval_every_blocks": 4,
    "disagree_flag": {"width": 0.20, "sessions": 5}, "K_s_cap": 12, "K_u_cap": 15,
}
KILL_P3 = {
    "min_months": 36, "min_episodes": 8, "max_months": 60, "reeval_months": 12,     # 2단계(§8.4)
    "block": P2["boot_block"], "n_boot": P2["n_boot"], "seed": 0, "ci": 0.95,
    "dd": 0.05, "confirm_sessions": 20, "target": "y_dd5_20", "reference": "p2_clim", "criterion": "point",
    "rearm": {"min_months": 12, "ci_lo_gt": 0.0},                            # 복귀: 새 장부 항목 + 전체 라이브 창 CI 하한 > 0
}
P3_DRIFT = {                                                                 # 임계 = 2003~24 참조분포 p5/p95 (post hoc; 라이브로 재조정 금지)
    "D1_p_level": {"window": 120, "lo": 0.04, "hi": 0.35},
    "D2_bss": {"red": {"window": 756, "lt": -0.05}, "yellow": {"window": 252, "lt": -0.10}},
    "D3_vol_fc": {"window": 60, "har_log_mae_gt": 0.50, "ewma_bias": (-0.35, 0.35)},
    "D4_vol_target": {"window": 60, "ratio_gt": 1.5}, "D4b_budget": True,
    "D5_stuck": {"window": 252, "warn_occ_gt": 0.75, "avg_w_lt": 0.30},
    "D6_churn": {"window": 252, "gt": 24},
    "D7_parity": {"sessions": 20, "dp_gt": 0.01},                             # exit 1
    "D8_coverage": {"min_n_eff": 12, "hit_lt": 0.5},
    "D9_hmm": {"window": 756, "auc_lt": 0.5},
    "D10_replay": {"dp_gt": 1e-6},                                            # 주간 재현(코드·자료 개정 탐지)
    "D11_feature_range": {"sessions": 5},                                     # x_vix·x_har·x_ma·x_hmm 가 1993~2024 범위 밖
}
SCENARIO_P3 = {"bins": P2["reliability_bins"], "min_n_eff": 20, "n_eff_div": P2["n_eff_div"], "episode_split": False,
               "h": P2["h"], "z": {"1s": 1.0, "80": 1.2816, "90": 1.645}, "dd_levels": (0.05, 0.10, 0.15, 0.20)}
P3_DEPLOY_MODES = ("info_only", "tones")                                     # p2 와 같은 두 값; 유효 모드 = 둘의 AND
MODEL_P3_PATH = RESULTS_DIR / "model_p3.json"
HMM_P3_PATH = RESULTS_DIR / "hmm_p3.json"
KILL_RECORD_PATH = RESULTS_DIR / "kill_record.json"
KILL_MANUAL_PATH = RESULTS_DIR / "kill_manual.json"
ALARMS_PATH = RESULTS_DIR / "alarms.csv"
TRACK_P3_PATH = RESULTS_DIR / "track_record_p3.json"
```

## 4. 국면 엔진 — `mrl/regime.py` (2-상태 가우시안 HMM, 직접 구현 numpy EM, 비지도)

### 4.1 관측·모형·추정

* 관측 `o_t = [100·r_t, ln RV10_t]`, `r_t = ln(C_t/C_{t−1})`(조정 종가, 비율이라 조정 불변), `RV10_t = sqrt(252·mean(r²_{t−9..t}))` 를 `P2["var_floor"]` 로 하한. 첫 유효 관측 1993-02-12(수익률 10개). OHLC 를 쓰지 않으므로 1993~95 OHLC 품질 문제(시가==고/저가 30%)에 면역.
* 모형: K=2, 상태별 완전 2×2 공분산 가우시안(μ_k ∈ R², Σ_k ∈ R^{2×2}), 전이행렬 A(2×2, 행 확률), 초기분포 π = A 의 정상분포(적합 아님 — 퇴화 추정치 하나를 없앤다). 파라미터 수 **12** = μ 4 + Σ 6 + A 2(`n_params(theta) == HMM_P3["param_count"]` assert).
* 결정적 초기화(난수 재시작 없음): 관측을 ln RV10 중앙값에서 둘로 나눠 상태 0 = 하위 절반의 평균·공분산, 상태 1 = 상위 절반; A0 = `HMM_P3["init_A"]`; π0 = stationary(A0).
* EM(스케일드 전방-후방): 로그 방출 `logB[t,k] = −ln(2π) − ½ln|Σ_k| − ½(o_t−μ_k)ᵀΣ_k⁻¹(o_t−μ_k)`; 전방 `α_t = normalize((α_{t−1}A) ⊙ B_t)`, 정규화 상수 c_t, `loglik = Σ ln c_t`(행별 최대값 이동으로 언더플로 방지); 후방 `β_t = A(B_{t+1} ⊙ β_{t+1}) / c_{t+1}`; `γ_t = α_t ⊙ β_t`(정규화); `ξ_t` 는 `numpy.einsum("ti,ij,tj->ij", α[:−1], A, B[1:]⊙β[1:]/c[1:])` 로 t 루프 없이 누적. M-스텝: `A_ij = Σξ_ij / Σγ_i`, `μ_k = Σγ_tk o_t / Σγ_tk`, `Σ_k = Σγ_tk (o_t−μ_k)(o_t−μ_k)ᵀ / Σγ_tk + jitter·I`(jitter 1e-6). 종료: Δloglik < 1e-6 또는 첫 적합 300회·재적합 100회(warm start). 종료 후 **라벨 고정**: 상태 1 = 수익률 분산이 큰 쪽('고변동 상태'; 카드 명칭은 '위기'가 아니라 '고변동 상태'). 그다음 π = stationary(A).
* **재적합 일정** = `calibrate.refit_dates`(각 해 1월 첫 거래일, 2003-01-02 ~ ; 라이브 재적합은 주간 작업에서만). 학습 관측 = `training_mask(idx, R_y, purge=20, train_start)` 와 같은 행(`pos(t) ≤ pos(R_y) − 21`; EM 은 라벨을 보지 않지만 모든 적합 객체의 학습 끝을 동일하게 둔다), θ_{y−1} 에서 warm start(첫 재적합은 결정적 초기화). θ_y 가 [R_y, R_{y+1}) 을 채점.
* **퇴화 guard**(위반 시 θ_{y−1} 유지 + 경고 + `hmm_p3.json.guard_log`): `p00, p11 ∈ [0.90, 0.999]`, 두 상태의 ln RV10 평균 차 ≥ 0.3, 두 상태 점유(Σγ/T) ≥ 5%, 공분산 양정치. 3회 이상 트립하면 그림자 멤버 H 를 `candidate_rejected` 로 내린다(§5.4).
* **필터(점 원칙)**: θ_y 로 1993-02-12 부터 전방 재귀만 돌려 `P_high_t = α_t[1] = P(state_t = high | o_1..o_t)`. 평활 γ 는 테스트에서만 계산(카나리). 그 해의 OOS 값은 t ∈ [R_y, R_{y+1}) 의 P_high_t. Platt 학습용(§5.2)으로는 θ_y 를 학습 구간 전체에 전방 필터한 값을 쓴다(비지도 look-back 공개; 라벨은 θ 에 절대 안 들어감).
* **k-스텝 전이확률**(A 의 닫힌 식, 표시·소거 전용): `p_k = (ξ_t·A^k)[high]`(k 세션 뒤 고변동일 확률), `q_k = 1 − [ξ_t·p10·p00^{k−1} + (1−ξ_t)·p00^k]`(t+1..t+k 에 고변동 세션이 하나라도 있을 확률; k = `turn_h` = 20 = 라벨 지평). 특징 `x_hmm_t = logit(clip(P_high_t, 1e-4, 1−1e-4))` 하나만 Platt 에 들어간다(A 의 q20 변형은 소거 행으로 보고, 선택 불가).
* 일간: `daily.py` 가 `hmm_p3.json` 의 올해 θ 로 완성 봉 이력 전체를 다시 필터(8.5k 행 < 0.05초)하고, 장부의 어제 `p3_hmm_p_high` 에서 한 걸음 갱신한 값과 1e-7 안에서 일치하는지 assert. 종가 결측 → P_high NaN, x_hmm NaN, 구간은 p2 밴드만.

### 4.2 함수

```python
HMM_PARAM_COUNT = 12
@dataclass(frozen=True)
class HMMTheta:
    A: tuple; mu: tuple; cov: tuple; pi: tuple                 # 2×2 / 2×2 / 2×2×2 / 2 (JSON 직렬화용 튜플; 계산은 np.asarray)
    refit_date: str; train_start: str; train_end: str; n_obs: int; loglik: float; n_iter: int; converged: bool
    obs_spec: str; theta_id: str; guard: dict                  # theta_id = sha256(1e-10 반올림 파라미터)[:12]
def observations(close: pd.Series, asof=None, rv_window: int = 10) -> pd.DataFrame     # 열 r100, ln_rv10 ; asof 로 먼저 자른 뒤 계산
def deterministic_init(obs: np.ndarray, init_A=HMM_P3["init_A"]) -> HMMTheta
def stationary(A: np.ndarray) -> np.ndarray
def log_emissions(obs: np.ndarray, theta: HMMTheta) -> np.ndarray                     # T×2
def forward_filter(obs: np.ndarray, theta: HMMTheta) -> tuple[np.ndarray, np.ndarray, float]   # (α T×2, c T, loglik) — 전방만
def em_fit(obs: np.ndarray, init: HMMTheta, max_iter: int, tol: float = 1e-6, jitter: float = 1e-6) -> HMMTheta
def guard_check(theta: HMMTheta, occupancy: np.ndarray, prev: HMMTheta | None, cfg=HMM_P3["guard"]) -> tuple[bool, list[str]]
def filter_probabilities(obs_df: pd.DataFrame, theta: HMMTheta) -> pd.Series           # P_high (전방), NaN 관측 행은 NaN·이월 없음
def smoothed_probabilities(obs_df, theta) -> pd.Series                                # 테스트 카나리 전용 (모듈 밖에서 호출 금지 — selftest 가 grep)
def k_step(theta: HMMTheta, p_high_t: float, k: int = HMM_P3["turn_h"]) -> dict          # {"p_k", "q_k", "k"}
def hmm_walk_forward(obs_df: pd.DataFrame, refit_dates, *, purge=20, train_start=HMM_P3["train_start"], cfg=HMM_P3,
                     prev_thetas: list[HMMTheta] | None = None) -> tuple[pd.DataFrame, list[HMMTheta]]
    # 반환 df: 인덱스 = 관측 세션 전체. 열: p_high(OOS: θ_y 로 [R_y,R_{y+1}); 2003 이전 행: θ_2003 — Platt 학습 전용, 'in_sample'=True),
    #      p20, q20, x_hmm, refit_year, theta_id, in_sample. thetas: 재적합별 HMMTheta. .attrs = {guard_log, warnings, obs_spec, n_refits, seconds}
def n_params(theta: HMMTheta) -> int
def save_thetas(thetas: list[HMMTheta], path=HMM_P3_PATH, *, live: HMMTheta | None = None) -> None
    # JSON {schema_version, obs_spec, param_count, thetas[...], live{...}, guard_log, registry_sha, created_at_utc}
def load_thetas(path=HMM_P3_PATH) -> tuple[list[HMMTheta], HMMTheta]
def theta_table(thetas) -> pd.DataFrame                                                # refit_date, p00, p11, dur0, dur1, vol0, vol1, mu_r0, mu_r1, n_iter, guard
```

### 4.3 설계 단계 실측(심사 재현, 1993-02-12~2024-08-30; 공식 수치로 대체)

* 첫 적합(관측 < 2003-01-02 퍼지): 14~24회 반복 < 1초, A = [[0.9803, 0.0197], [0.0170, 0.9830]], 기대 체류 51/59세션, 상태 일수익 표준편차 0.63%/1.50%(연 9.9%/23.8%), 드리프트 +0.088%/−0.015%/일. walk-forward 22회 재적합 27초(einsum ξ; 벡터화 전 목표 < 30초), p00 0.980~0.984 · p11 0.977~0.984(모든 해 guard 통과), 2024 재적합 A = [[0.9807, 0.0193], [0.0217, 0.9783]].
* OOS 2003-01-02~2024-08-02(라벨 5,433행·271 독립 창·기저율 15.4%): 필터 P_high 점유 42.8%, AUC vs `y_dd5_20` **0.697**(p_vix 0.688, x_har 0.625, −x_ma 0.643); 시대별 P_high vs p_vix: 2003~07 0.648/0.662, 2008~12 0.675/0.651, 2013~19 0.581/0.571, **2020~24 0.666/0.594**(VIX 판별력 감쇠 시대가 HMM 이 밥값을 하는 곳). 카나리: 평활 γ 의 AUC 0.722 vs 필터 0.687(전체표본 θ) — 필터가 도는지 판별할 만큼 차이가 크다.
* 참고(선택 불가 소거): A 관측 [r, ln σ_GK+OV] 대각형 전체표본 A = [[0.9737, 0.0263], [0.0327, 0.9673]], 체류 38/31, 정상 P(고변동) 0.446, ξ AUC 0.685·q20 0.667; B 일변량 r 1993~2014 동결 A = [[0.9889, 0.0111], [0.0241, 0.9759]], 체류 90/41, 2015~24 AUC 0.665 vs x_vix 0.625, 0.5 교차 12회/년(상태기계로 못 쓴다는 근거). 셋 다 같은 정보(수익률 지속성·실현변동성)이며 차이는 잡음 안이다.
* **표시**: 게이지(히스테리시스 밴드 0.2/0.8 → 낮음/중간/높음; 밴드 변경 4.9회/년), 오늘 P_high·p20·q20, 기대 체류, θ 경로 표, 라벨 '그림자 — 확률·비중·결정층·킬룰에 들어가지 않음'.

### 4.4 검증·탈락

같은 OOS 행·블록(BLOCKS_24·18·시대)에서 P_high·q20 의 AUC, θ 안정성 표, guard 로그, 결정론(두 실행 CSV 동일·θ 1e-7), PIT 비트 동일(20개 무작위 절단), 카나리(평활 > 필터), 지연 특징(x_hmm 을 20/250세션 지연 → AUC 엄격 감소). 탈락(게이지 숨김·멤버 `candidate_rejected`): 후행 756세션 AUC < 0.5(D9), guard 3회 이상, 주간 결정론 assert 실패.

## 5. 등록부·앙상블 — `mrl/ensemble.py` (멤버 등록부 · 동일가중 결합 · 채택 규칙 · 불일치 구간)

### 5.1 등록부(동결 튜플; 변경은 번호 붙인 장부 항목으로만)

| 멤버 | 종류 | 확률 원천 | K_s(지도) | K_u(비지도) | 1일차 상태 | 장부 |
|---|---|---|---|---|---|---|
| `p2` | seed | `prob_dd5_20`(= `summary_p2.json.acceptance.tone_model` 의 단; info_only 면 M3 를 그림자 표시) | 4 | 0 | `seed` | #2 |
| `M1` | ladder | `p2_p_m1`(보정 VIX; M3 를 M1 쪽으로 수축시키는 정규화이자 정직한 잣대) | 2 | 0 | `shadow` | #8 |
| `H` | hmm_platt | `logit p_H = a_y + b_y·x_hmm`, x_hmm = logit P_high (§4) | 2 | 12 | `shadow` | #3a |
| (미등록) B 폭/신용 | logit | −Δ63 ln(HYG/IEF), −Δ63 ln(RSP/SPY), 2010 재적합부터 | 3 | 0 | `candidate_rejected`(수치만 기록) | — 기록 |

* 생산 평균 = `admitted ∪ {seed}` 의 동일가중 평균(확률 공간). **1일차 admitted = ∅** 이므로 생산 확률 ≡ p2 — 헤드라인 숫자는 Phase 2 의 것이고 Phase 3 는 별도의 생산 확률을 만들지 않는다. 가중치는 상수(적합 0).
* 등록 멤버는 모두 매일 장부에 기록되고(`p3_members` JSON), 같은 날짜·같은 라벨로 채점되며(라이브 Brier vs clim·M1·B1), 불일치 구간에 들어간다. `killed` 멤버는 평균과 구간 모두에서 빠진다.
* `registry_sha256()` = `ensemble.py`·`regime.py` 소스 + REGISTRY 튜플의 sha256. 모든 P3 산출물·장부 행에 기록.

### 5.2 그림자 멤버 H 의 Platt

`platt_walk_forward(x_hmm, y, refit_dates, purge=20, train_start="1993-10-14", C=1.0)`: 재적합일 R_y 마다 `model.FIT_KWARGS` 와 같은 sklearn 로지스틱(l2, C=1.0, lbfgs, tol 1e-8, 표준화 없음)을 학습 행(y 유효 ∧ `training_mask`)에 적합; x_hmm 은 θ_y 를 학습 구간에 전방 필터한 값(§4.1). OOS [R_y, R_{y+1}) 는 같은 θ_y. 라이브 계수는 `model_p3.json.platt`(`refit_date="2024-08-30"`; #2b 뒤부터 매년 1월). 학습 시작 1993-10-14 는 다른 멤버와의 동등성 때문(x_hmm 자체는 1993-02-12 부터 유효).

### 5.3 결합·구간

```python
def combine(probs: pd.DataFrame, statuses: dict[str, str]) -> pd.Series
    # admitted(seed 포함)·가용·유한 멤버의 평균; 하나도 없으면 NaN(상태 이월 — P2 §2 규칙). 가중 재조정 없음.
def disagreement(probs: pd.DataFrame, statuses: dict, p2_lo: pd.Series, p2_hi: pd.Series, cfg=ENSEMBLE_P3["disagree_flag"]) -> pd.DataFrame
    # lo = min(p2_lo, min 등록·비killed 멤버), hi = max(p2_hi, max 멤버), width, src('members'|'p2'), flag(width>0.20 가 ≥5세션 연속)
```
헤드라인 문구: `다음 20거래일 안에 -5%: 이런 날 100일 중 약 N일 (기저율 16 · 모델들은 L~H 로 갈린다)`. 사전 관측: (M1, M3, H) 불일치 중앙 4.5pp, p90 12.1pp; 20pp 초과는 급변 국면에서만.

### 5.4 채택 규칙(사전 등록; 관측 기록은 기각만, 채택은 신선 자료만)

```python
def admission_test(oos: pd.DataFrame, candidate: str, admitted: tuple[str, ...], blocks, *, frac=8/11,
                   block=40, n_boot=4000, seed=0) -> dict
    # 블록 b 마다 Brier(mean(admitted∪{c}))_b vs Brier(mean(admitted))_b (c 가용 행에서). wins = 개선 블록 수,
    # need = ceil(frac·n_avail_blocks). no_harm = loss_diff_ci(mean(admitted), mean(admitted∪{c})).hi > 0.
    # verdict = "ADMIT" iff wins ≥ need and no_harm else "SHADOW". 반환 {wins, n_blocks, need, per_block(dBrier×1e4), pooled, ci, verdict}
def fresh_blocks(idx: pd.DatetimeIndex, live_start, months: int = 3, label_h: int = 20) -> list[tuple[pd.Timestamp, pd.Timestamp]]
    # live_start 부터 3달력월 타일; 마지막 세션 + 20세션의 라벨이 실현된 블록만 '완결'
def admission_power_table(n_blocks: int = 11, need: int = 8, qs=(0.5, 0.6, 0.7, 0.8, 0.9)) -> pd.DataFrame   # 이항 P(≥need | q)
def registry_table(oos, statuses, admissions, live_scores) -> list[dict]                   # 주간 페이지·카드 줄
def apply_admission(model_p3: dict, result: dict, ledger_no: str, effective_refit: str) -> dict   # 상태 변경은 다음 1월 재적합부터
```

* 순서 고정 H → M1(탐욕적 전진, 재배열 금지). 관측 기록(2003~2024-08, BLOCKS_24) 사전 관측: **H 6/11**(풀링 dBrier +0.00000, CI [−0.00300, +0.00257]; 블록별 ×1e4 [−2.3, −24.3, −103.2, +29.7, +9.7, −30.1, +9.5, +34.8, +29.0, +51.7, −5.2]), **M1 5/11**(−0.00048, CI [−0.00137, +0.00037]), 참고 M1→M3 자체가 6/11. 아무것도 통과하지 못한다 — 예상된 결과이며 이 규칙의 값은 '기각만 할 수 있는 구조'다. 검정력: P(≥8/11 | q) = 0.113(q=.5) / 0.296(.6) / 0.570(.7) / 0.839(.8) — 등록부 페이지에 그대로 인쇄해 '몇 년째 안 바뀌는 등록부'가 게으름이 아니라 설계임을 보인다.
* **신선 자료 채택**: 완결 신선 블록 ≥ 11개(라이브 시작 + ≈33개월 + 20세션; 2026-10 시작이면 최초 2029-07)에서 같은 검정(need = ceil(8/11·n)), 이후 4블록(12개월)마다 재검. ADMIT 이면 `model_p3.json.members[c].status = "admitted"`, 장부 항목 #8x, 발효 = 다음 1월 재적합(연중 변경 금지) — 그때부터 §6-c 가 발효하고 K_s 가 4 → 6(H) 또는 6(M1) 이 되며 카드 파라미터 줄이 바뀐다.
* 멤버 탈락: 공식 실행에서 H 의 풀링 BSS_clim ≤ 0 또는 AUC < 0.60, guard 3회, D9 → `candidate_rejected`. 라이브에서 §7 판정일에 멤버의 BSS_clim ≤ 0 → `killed`(§8.4).
* 5번째 슬롯 M4a(M3 + b4·x_hmm, 5 파라미터)는 사다리 단 M3→M4a 로 **채점만** 한다(#3 닫힘 근거의 공식 수치): 사전 관측 b4 ∈ [−0.10, +0.02], BSS 0.0901 vs 0.0910, 손실차 CI [−0.0005, +0.0003] — x_vix·x_har 와 공선.

### 5.5 설계 단계 실측(공식 수치로 대체)

멤버 H: Brier 0.1243, BSS_clim +0.066, BSS_vix(B1) +0.177, BSS_m1 −0.017, AUC 0.658; BLOCKS_24 BSS_clim [0.151, 0.416, 0.058, 0.023, 0.060, 0.111, 0.023, 0.068, −0.026, 0.112, −0.010](9/11 > 0); 시대 BSS_clim M3/H: 2003~12 .137/.080, 2013~19 .062/.059, **2020~24 .025/.039**. ens(M3,H) BSS_clim 0.0911 vs M3 0.0910; ens3(M1,M3,H) 0.0903. A 의 p3(M3+M1+HMM_A) 0.0892, 손실차 M3→p3 −0.00024 [−0.00251, +0.00183], 최악 블록 −0.037 → −0.013 — 사전 관측 뒤에 본 이득이라 채택 근거로 쓰지 않는다. 결정층을 ens3 에 돌리면 3.5회/년, 점유 76.7/19.9/3.4% (M3: 3.9회, 79.4/16.0/4.7%).

## 6. 비중 — `mrl/sizing.py` (가족 위험예산 → 변동성 목표 × 상태 배수, 적합 파라미터 0)

### 6.1 규칙(세션 t, 자료 ≤ t)

1. **변동성 예측**: `σ̂_t = sqrt(252·v_t)`, `v_t = λ·v_{t−1} + (1−λ)·r_t²`, λ = 0.94(RiskMetrics 상수), `r_t = ln(C_t/C_{t−1})`(조정 종가), v 는 첫 60세션(1993-02-01~)의 표본분산으로 seed. 왜 EWMA 인가: 같은 5pp 격자·10pp 밴드에서 고정가중 HAR 특징은 116~137회/년, 적합 log-HAR 예측은 74회/년(일간)·13회/년(주간), EWMA94 는 35회(일간)·9.6~11.4회(주간)로 예측 오차는 거의 같다(20일 실현 RV 대비 log-MAE 2003~24: EWMA94 0.306, HAR-고정 0.299, HAR-OLS 0.279, VIX 0.374; 중앙 편향 ln(실현/예측) −0.04/+0.03/−0.03/−0.29). 적합 HAR(`p2_har_fc_20`)은 '예상 변동성' 표시와 민감도 S-HAR 전용이며 생산 비중에 절대 안 들어간다.
2. **예산 → 목표**: `σ_T = target_vol(D_max) = max{g ∈ vol_grid : g ≤ D_max / k_slow}`, k_slow = 3.5, 기본 D_max = 0.35 → **σ_T = 10%**. D_max < 0.21(σ_T < 6%)은 제공하지 않는다(바닥 0.25 만으로도 55% 급락에서 ≥14% 를 잃으므로). 사다리(§6.2)는 같은 σ_T 를 두 예산선으로 읽게 한다: 완만한 약세장선 D_slow = 3.5·σ_T, 급락형선 D_fast = 2.0·σ_T.
3. **변동성 비중**: `w_vol_t = clip(σ_T / σ̂_t, 0.25, 1.00)`.
4. **상태 배수**: `m_t = TONE_EXPOSURE[STATE_TO_TONE[p2_state_t]]` = 1.00 / 0.50 / 0.25 (새 상수 없음). `w_target_t = clip(w_vol_t × m_t, 0.25, 1.00)` — 곱 뒤 재클립(가족은 주식 25% 아래로 내려가지 않는다; v0·p2 와 같은 바닥). 유효 배치 모드(`p2_deploy_mode == "tones"` ∧ `model_p3.json.deploy_mode == "tones"`)가 아니면 m 은 적용하지 않고 비중을 **제안하지 않는다**; w_vol 은 장부에 회색 '가정치'로만 남는다(reason `info_only`).
5. **실행(churn 제어; 상수 전부 사전 등록)**: `cand_t = round(w_target_t/0.05)·0.05`; `esc_t = (m_t < m_{t−1})`; `week_end_t` = asof 가 그 ISO 주의 마지막 거래일(`calendar_us.is_period_end(asof, "W")`, 백테스트에선 `period_end_from_index`).
   * `w_exec_{t−1}` 없음 → `w_exec_t = cand_t`, reason `init`
   * elif `esc_t` → `w_exec_t = min(cand_t, w_exec_{t−1})`, reason `escalation`(즉시, 밴드 무시, 하향만)
   * elif `week_end_t` ∧ `|w_target_t − w_exec_{t−1}| ≥ 0.10` → `w_exec_t = cand_t`, reason `weekly`
   * else `w_exec_t = w_exec_{t−1}`, reason `hold`(σ̂·상태·θ 결측이면 `input_missing`, 절대 1.0 으로 복귀하지 않음)
   구조적 상한: 주 1회 + 격상 1회 → 어떤 5세션 창에서도 ≤ 3회(결정층과 같은 경계). 격하는 주말까지 기다린다(구조상).
6. **가족용 파생값(표시 전용)**: 한 단계 하향이 일어나는 예상 변동성 `σ_down = σ_T·m_t / (w_exec_t − 0.10)`, 상향 `σ_up = σ_T·m_t / (w_exec_t + 0.10)`, 다음 점검일(다음 주 마지막 거래일), D_max 사다리의 자기 행.
7. **점 원칙·비용**: w_exec_t 는 t 종가에 확정되어 t+1 수익률에 적용('내일부터 주식 비중 X%'); 비용 5bp × |Δw|(변경 시), 현금 0%(보수적; 평균 현금 ~31% 에 2% 단기금리면 +0.6pp/년 — 각주로만).

### 6.2 예산 사다리(카드·`docs/sizing_p3.html`; `run_phase3.py` 가 채움)

| σ_T | D_slow(3.5×) | D_fast(2.0×) | MaxDD 1993~ 변동성 단독 | MaxDD 2003~ 결정층 포함(바닥 .25) | 최악 월 | CAGR 2003~ | 평균 비중 | 변경/년 |
|---|---|---|---|---|---|---|---|---|
| 6% | 21% | 12% | −19.0% (2002-10-09) | (채움) | (채움) | (채움) | (채움) | (채움) |
| 8% | 28% | 16% | −24.2% (2002-10-09) | (채움) | | | | |
| **10% (기본)** | **35%** | **20%** | **−30.7% (2002-10-09)** | **−23.5% (2009-03-09)** | **−7.8% (2018-10)** | **7.22%** | **69%** | **10.9** |
| 12% | 42% | 24% | −36.8% (2002-07-23; 12.5%) | (채움) | | | | |
| 15% | 52.5% | 30% | −41.8% (2002-10-09) | (채움) | | | | |
| 보유 | — | — | −55.2% | −55.2% | −16.5% | 10.85% | 100% | 0 |

설계 단계 실측 비율 MaxDD/σ_T: 1993~ 변동성 단독 3.2/3.0/3.1/2.9/2.8(B 주간 규칙), 3.2/3.1/3.5/2.9/2.8(A 0.25 격자 규칙; 최악 **3.46** 가 k_slow 3.5 의 근거); 2003~ 결정층 포함·바닥 없음 1.7/1.6/1.5~1.8/1.6/1.5, 바닥 있음 10% 에서 2.35. 각 행에 보유 대비 CAGR 차와 보유에 뒤진 달력연도 비율(10%: 20/22 = 91%, 중앙 −5.8pp; 10/90 분위 −14.1/−1.2pp)을 인쇄한다. 소유자는 가구 기본 D_max 를 장부 항목으로 등록하고(0.35), 다른 가족은 자기 행을 읽는다. 카드 문장: `예산 −35%(완만한 약세장 기준) ≈ 급락형 −20% → 목표 변동성 10%. 2003년 이후 기록에서는 −23.5% 였고, 2000~02년형 완만한 약세장에서는 −30.7% 까지 갔다 — 예산은 보장이 아니다.`

### 6.3 백테스트 표(형식 고정; `results/summary_p3.json.sizing` · `docs/sizing_p3.html`)

열: 기간 | CAGR | MaxDD(일자) | 최악 월(월) | 연변동성 | 전환/년 | 평균 비중 | 누적 비용 | ΔCAGR vs 보유 | ΔMaxDD vs 보유 | 실현변동성/σ_T | MaxDD/σ_T. 규약 = `evaluate.allocation_sim` 그대로(w[t]→r[t+1], 5bp×|Δw|, 현금 0%). 항상 세 참조 행(보유·v0 completed(2015+)·p2 결정층 100/50/25)과 변동성 일치 참조 행(VT14)을 넣는다. 설계 단계 근사(심사 재현; 공식 산출로 대체, 그 전엔 '설계 단계 근사' 라벨):

**2003-01-02 ~ 2024-08-30 (결정층 OOS, 21.7년)**

| 행 | CAGR | MaxDD | 최악 월 | 변동성 | 전환/년 | 평균 비중 | 비용 |
|---|---|---|---|---|---|---|---|
| 보유 | 10.85% | −55.2% (2009-03-09) | −16.5% (2008-10) | 18.7% | 0 | 100% | 0 |
| p2 결정층 100/50/25 | 10.19% | −31.0% (2009-03-09) | −8.6% (2018-10) | 12.4% | 3.9 | 89% | 1.9% |
| **채택: 10% × 상태, 주간 + 즉시 격상, 곱 뒤 바닥 .25** | **7.22%** | **−23.5% (2009-03-09)** | **−7.8% (2018-10)** | **9.3%** | **10.9** | **69%** | **2.2%** |
| S-nofloor: 같은 규칙, 재클립 없음(B 원안) | 6.94% | −17.9% | −7.8% | 8.9% | 11.4 | 67% | 2.3% |
| 변동성 단독 10%, 주간 10pp | 7.84% | −24.1% | −6.8% (2020-02) | 10.2% | 9.6 | 72% | 1.5% |
| S-monthly: C 월초+상태변경 × TONE_EXPOSURE, 바닥 .25 | 7.49% | −23.2% | −7.6% | 9.5% | 8.4 | 69% | 2.1% |
| S-mild: C 월간 × (1/.75/.5), 바닥 .25 | 7.63% | −23.6% | −7.1% | 9.8% | 8.8 | 71% | 1.7% |
| S-grid: A 0.25 격자·히스테리시스 .175, VT10 단독 / min(VT10, 상태) | 8.07% / 7.85% | −23.8% / −23.9% | −6.2% | 10.2% | 6.1 / 6.4 | 74% / 73% | 1.7% / 1.8% |
| S-VT14: A 격자 VT14 단독(p2 결정층과 변동성 일치) | 10.14% | −30.3% | −7.2% (2008-09) | 12.5% | 3.6 | 87% | 1.0% |
| S-noesc(바닥 없음) / 1세션 지연 / 10bp / S-HAR / 일간 5pp | 6.82 / 6.87 / 6.83 / 6.77 / 7.17% | −15.5 / −15.6 / −18.1 / −17.8 / −18.3% | −6.9 / −9.4 / −7.9 / −7.0 / −7.2% | | 11.2 / 11.4 / 11.4 / 14.1 / 35.5 | | |
| 목표 7.5 / 12.5 / 15% × 상태(바닥 없음) | 5.47 / 7.85 / 8.53% | −12.5 / −21.0 / −22.9% | −6.8 / −7.8 / −8.4% | 7.0 / 9.8 / 10.5% | 12.0 / 8.0 / 6.0 | 54 / 74 / 78% | |

**2015-01-02 ~ 2024-08-30 (v0 창, 홀드아웃 제외)**: 보유 12.98% / −33.7% (2020-03-23) / −12.5%; v0 completed 10.43% / −16.1% (2022-06-16) / −7.7% / 55.3회 / 71% / 비용 14.5%; p2 결정층 10.79% / −19.8% (2022-10-12) / −8.6% / 5.2 / 89%; 규칙 10%(바닥 없음) 7.07% / −12.9% (2018-12-24) / −7.8% / 12.8 / 68%; 12.5% 7.75% / −14.5%.
**1993-10-14 ~ 2024-08-30 (2003 이전 변동성 단독; 완만한 약세장 진단)**: 10% 규칙 6.5% / −30.7% (2002-10-09) / −7.8% / 10.5회 / 66% vs 보유 10.4% / −55.2%. VT-only 격자 규칙 6/8/10/12/15%: 4.54/6.25/7.03/7.88/9.25% · −19.2/−24.8/−34.6/−35.2/−41.6% · 최악 월 −6.1/−5.6/−7.2/−8.2/−10.0%.
**에피소드 손익(10% 규칙 vs 보유, 평균 비중)**: 1998 LTCM −10.1 vs −18.2%(44%) · 2000~02 −30.7 vs −47.5%(47%, 결정층 없음) · 2008 −13.5 vs −55.2%(25%) · 2010 −8.4 vs −15.7% · 2011 −10.8 vs −18.6% · 2015~16 −9.8 vs −13.0% · 2018-02 −9.0 vs −10.1%(90%; 저변동 급락엔 보호 없음) · 2018-Q4 −12.2 vs −19.3% · 2020 −12.9 vs −33.7%(32%) · 2022 −11.0 vs −24.5%(29%); 최악 12개월 −21.4% vs −47.4%. 돌파일 2003+ 의 결정 상태: normal 11 / caution 8 — 모델이 첫날을 맞추리라 기대하지 말라는 줄이 페이지에 남는다.

### 6.4 유지 조건(사전 등록; 공식 산출로 판정, 근사로 판정 금지)

(a) MaxDD_2003~24(채택 규칙) ≥ −2.5·σ_T(10% → −25%); (b) MaxDD_1993~24(변동성 단독) ≥ −k_slow·σ_T(−35%); (c) p2 결정층 대비 MaxDD 개선 ≥ 5pp; (d) 전환 ≤ 24회/년 ∧ 누적 비용 ≤ 5%(2003~24); (e) 최악 월이 p2 결정층보다 2pp 이상 나쁘지 않음; (f) 실현변동성/σ_T ∈ [0.75, 1.15]. 하나라도 위반 → `summary_p3.json.sizing.deploy_sizing = false`, 카드의 비중 블록 숨김(장부는 계속 기록). 라이브: D4(60세션 실현변동성/σ_T > 1.5)·D4b(라이브 MaxDD < −D_max)·D6(변경 > 24/252세션)은 소유자가 장부 항목을 쓸 때까지 비중 카드를 정지한다. §7 킬 스위치는 비중 성적표와 무관하게 상태 배수·비중 제안을 숨긴다. 설계 단계 근사로는 (a) −23.5 ≥ −25 ✓ (b) −30.7/−34.6 ≥ −35 ✓ (c) 7.5pp ✓ (d) 10.9·2.2% ✓ (e) −7.8 vs −8.6 ✓ (f) 0.93 ✓ — 근소한 (a)·(b) 는 공식 수치에서 뒤집힐 수 있다.

### 6.5 정직한 읽기(표 위에 그대로 인쇄)

(a) 변동성을 맞추면 파라미터 0개의 변동성 목표(VT14)가 p2 결정층과 같다(10.14%/−30.3% vs 10.19%/−31.0%, 3.6 vs 3.9회) — 비중은 확률 모델 위에 서 있지 않다. (b) 확률층이 MaxDD 를 낮추는 것은 **바닥 없는 곱**일 때뿐(−24.1 → −17.9%; 2008-09~2009-06 평균 비중 0.12, 2020-03~05 0.10 = 사실상 퇴장)이며, 바닥을 두면 −23.5%(변동성 단독 −24.1% 와 같다); min 규칙은 상한이 0.7~11.6% 의 세션에서만 묶이고 CAGR 만 0.2~1.0pp 깎는다. 채택 규칙에서 상태 배수의 가치는 낙폭이 아니라 가족용 상태·킬룰 시험 대상이라는 점이다. (c) 변동성 목표는 완만한 약세장(2000~02: −30.7~−34.6%)을 막지 못한다 — k_slow 3.5 의 이유. (d) 모든 규칙 행이 보유 CAGR 에 0.4~4.6pp 뒤지고, 10% 규칙은 달력연도의 ~90% 에서 뒤진다(중앙 −5.8pp; 63/126/252세션 창에서 뒤질 확률 79/88/90%) — 상품은 낙폭 통제이며 그 말을 그대로 쓴다. 현금 0%·5bp 는 분기별로 늦게 실행하는 가족의 실제 마찰을 과소평가한다.

### 6.6 함수

```python
SIZING_RULE = ("p3|sigma=ewma(lambda=0.94,seed=60)|sT=floor_grid(Dmax/3.5)|w_vol=clip(sT/sigma,.25,1)"
               "|m=TONE_EXPOSURE[STATE_TO_TONE[p2_state]]|w=clip(w_vol*m,.25,1)|grid=.05|band=.10|weekly+escalate|cost=5bp|cash=0")
def sizing_sha256() -> str                                                                   # sizing.py 소스 + SIZING_RULE
def ewma_vol(close: pd.Series, lam: float = 0.94, seed_sessions: int = 60, asof=None) -> pd.Series   # 연율화 σ̂; asof 로 먼저 자름
def target_vol(d_max: float, k: float = P3["k_slow"], grid=P3["vol_grid"]) -> float          # 내림; d_max/k < grid[0] 면 ValueError
def budget_ladder(grid=P3["vol_grid"], k_slow=3.5, k_fast=2.0, results: dict | None = None) -> pd.DataFrame
def state_multiplier(states: pd.Series) -> pd.Series                                         # STATE_TO_TONE → TONE_EXPOSURE; 결측은 NaN
def exposure_target(sigma: pd.Series, mult: pd.Series, sigma_target: float, w_min=0.25, w_max=1.0,
                    floor_after=True) -> pd.DataFrame                                         # w_vol, mult, w_target
def step(w_target_t, mult_t, mult_prev, w_exec_prev, week_end: bool, cfg=P3) -> tuple[float, str]   # 한 걸음(daily.py, 재개 안전)
def execute(w_target: pd.Series, mult: pd.Series, *, grid=0.05, band=0.10, cadence="weekly", escalate=True,
            init_w=None) -> pd.DataFrame                                                     # w_exec, reason, cand, week_end
def next_thresholds(w_exec: float, sigma_target: float, mult: float, asof, band=0.10) -> dict # sigma_down, sigma_up, next_check
def run(sigma: pd.Series, states: pd.Series, sigma_target: float, cfg=P3, deploy: bool = True) -> pd.DataFrame
    # 열: sigma, w_vol, mult, w_target, cand, w_exec, reason, week_end ; deploy=False 면 mult 미적용·w_exec NaN·reason 'info_only'
def backtest_table(paths: dict[str, pd.Series], spy_close: pd.Series, windows: dict[str, tuple], cost_bps=5,
                   references: dict | None = None) -> pd.DataFrame                           # §6.3 형식; evaluate.allocation_from_weights
def sensitivities(feats, states, spy_close, cfg=P3) -> dict[str, pd.Series]                  # S-nofloor, S-monthly, S-mild, S-grid, S-VT14, S-noesc, S-lag, S-10bp, S-HAR, S-daily, 목표 격자
def retention(table: pd.DataFrame, sigma_target: float, cfg=P3) -> dict                      # (a)~(f) 판정 + deploy_sizing
def episode_pnl(w_exec: pd.Series, spy_close: pd.Series, episodes5: pd.DataFrame) -> pd.DataFrame
def rolling_relative(ret_rule: pd.Series, ret_bh: pd.Series, L: int) -> dict                 # 비겹침 창 diff p10/p50/p90, share_behind
```
`mrl/evaluate.py` 추가: `allocation_from_weights(w: pd.Series, spy_close, cost_bps=5) -> dict`(`allocation_sim` 과 같은 `_perf_stats`·키; 톤 경로로 만든 w 에서 `allocation_sim` 과 비트 동일 — 테스트), `window_distribution(series: pd.Series, L: int, step: int = 21) -> dict`(참조 분위 p5/p25/p50/p75/p95·min·max·share<0).

## 7. 시나리오 — `mrl/scenarios.py` (세기만 한다; 적합 0; 모든 행에 n·n_eff·구간)

모든 수치는 OOS 기록(2003-01-02~2024-08-30; #2b 뒤에는 + 홀드아웃; 라이브 창은 라벨이 닫히는 대로 별도 열)에서만 온다. 표본 수는 언제나 `n_eff = n/20`(또는 에피소드 수)이고 구간은 `calibrate.wilson(p̂, n_eff)`; 중앙값 구간은 40세션 블록 부트스트랩.

**A. 확률 구간 표**(`bin_table`): p = 배치 단의 `prob_dd5_20`(1일차 = p2), 구간 = `P2["reliability_bins"]`. **풀링 규칙(사전 등록)**: 위에서부터 인접 구간을 합쳐 표시되는 모든 구간이 n_eff ≥ 20 이 되게 한다(설계 단계 [0.30, 0.40) 이 n_eff 18 에서 0.229 로 꺼지는 비단조 — 손으로 고치지 않고 규칙으로 없앤다). 열: n, n_eff, mean_p, 관측 dd5 비율 + Wilson, fwd_ret_20 p10/p50/p90, fwd_maxdd_20 p10/p50, ≥10% 에피소드가 이 구간에서 시작한 비율. 설계 단계 근사(M3): [0,.08) n 1,422 · n_eff 71 · 관측 0.076 (0.03~0.16) · 수익 −3.0/+1.0/+3.3% · 최대낙폭 p10 −4.5%; [.08,.12) 1,137/57/0.102 (0.05~0.21)/−3.6/+1.2/+4.0/−5.0; [.12,.16) 869/43/0.116 (0.05~0.24)/−3.9/+1.7/+4.9/−5.3; [.16,.20) 620/31/0.161 (0.07~0.33)/−5.3/+2.1/+5.5/−6.7; [.20,.25) 453/23/0.236 (0.11~0.44)/−6.1/+2.7/+6.6/−7.9; **[.25, 1] 풀링** 932/47/≈0.33/수익 p10 ≈ −9%/최대낙폭 p10 ≈ −15% (풀리기 전 [.40,.50) n_eff 5: 수익 −14.0/+2.2/+8.4%, 최대낙폭 p10 −24.6%). 카드에는 오늘 행과 이웃 두 행만('오늘은 이 줄'), 주간 페이지엔 전체 + 라이브 열.
**B. 결정 상태 표**(`state_table`): normal n_eff 215 · dd5 0.113 (0.08~0.16) · 수익 −3.7/+1.4/+4.5%; caution 43 · 0.262 (0.15~0.41) · −5.2/+2.4/+7.3%; reduce 12 · 0.482 (0.24~0.73) · −13.8/+2.5/+12.4%. 카드에 반드시 인쇄하는 문장: `모든 상태·구간에서 20일 수익의 중앙값은 양수다 — 모델은 방향이 아니라 결과의 폭을 예측한다(VALIDATION §0). 높은 위험 구간일수록 수익 분포는 좁아지지 않고 넓어진다(급락과 반등이 같이 산다).`
**C. 에피소드 조건부**(`episode_conditionals`; `targets.episodes(close, 0.05, split=False)`, 홀드아웃 해제 전엔 고점 < 2024-09-01 만 → 33건; VALIDATION §2 의 36건은 2026-09-04 까지·split=False 규약이라는 점을 페이지에 명시): P(≥10% | −5% 돌파) = 11/33 = 0.33 (0.20~0.50); P(≥20% | −5%) = 4/33 = 0.12 (0.05~0.27); P(≥15% | ≥10%) = 6/11; P(≥20% | ≥10%) = 4/11 (0.15~0.65); 깊이 분위 p90/p75/p50/p25/p10 = −5.5/−6.0/−7.6/−11.2/−23.5%; 돌파 종가 이후 추가 손실 p10/p25/p50 = −19.4/−6.5/−2.8%, 11/33 이 −5% 보다 더; 돌파→저점 세션 p10/p50/p90 = 0/3/106, 고점→저점 중앙 22, 저점→회복 12/32/121; 빈도 1.04/년·군집(1997: 4, 2001~06: 0). 표시는 ATH 대비 현재 낙폭에 조건부: [−5%, 0) 기저율 문맥 / ≤ −5% '지금 −5% 를 뚫었다면' / ≤ −10% 는 ≥10% 행. 돌파일 결정 상태(normal 11 / caution 8)도 같이.
**D. 시장 내재 20세션 범위**(`implied_range`): `s = VIX_t/100·sqrt(20/252)`; 1σ = spot·exp(±s), 80% = ±1.2816s, 90% = ±1.645s; `p_vix`(B1)·`p_vix_bgk` 병기. 과거 포함률(2003~24, 실현 ln fwd_ret_20/s): 1σ 밴드 **83.8%**(명목 68.3%), 80% 밴드 93.6%, 90% 밴드 97.3%, −1.645s 아래 꼬리 2.4%(명목 5%), RV20/VIX 0.81 → 라벨 `시장 내재 (보수적: 과거 84% 포함)`. 적합 HAR(`p2_har_fc_20`) 기준 같은 밴드는 64% / 꼬리 5.8% → `예상 변동성 기준 (중심: 과거 64%)`. 둘 다 인쇄, 넓은 쪽이 헤드라인. 예 2026-09-04(VIX 14.53): 1σ ±4.1%, 80% ±5.2%, 90% ±6.7%.

**카드 '시나리오' 블록**(P2 귀속 아래, 한국어 자연빈도, 3줄 고정 순서):
1. `시장이 보는 20일 범위(VIX 14.5): ±5.2% (80%; 과거 94% 포함, 보수적) · 예상 변동성 기준 ±3.9% (과거 ~80%)`
2. `오늘 같은 날(확률대 20~25%): 과거 독립 23창 중 5창(11~44%)이 20일 안에 −5%; 그때 20일 수익 10~90%: −6.1%~+6.6%, 최대낙폭 나쁜 10% −7.9%`
3. `만약 −5% 에피소드가 시작되면: 33번 중 11번(20~50%)은 −10% 까지, 4번(5~27%)은 −20% 까지; 절반은 −7.6% 안에서 멈춘다; 추가 손실 중앙 −2.8%, 나쁜 10% −19%`
풀링 전 n_eff < 20 인 행은 단독 표시 금지(코드로 강제). 라이브 채점은 80% 밴드 적중 수(VIX/HAR 각각, Wilson(n_eff))와 구간 표의 라이브 열(관측 비율, n_eff)뿐이며, 라이브 비율이 백테스트 Wilson 구간 밖이면 '보정 드리프트' 플래그(D8 과 별개 표시). `calib_p2_walkforward.csv` 가 없거나 spec_sha256 이 배치 모델과 다르면 주간 exit 1, 일간 회색 'n/a'.

```python
def bin_table(p, y, fwd_ret_20, fwd_maxdd_20, bins=SCENARIO_P3["bins"], min_n_eff=20, n_eff_div=20, episodes10=None) -> pd.DataFrame
    # 열: bin_lo, bin_hi, pooled(bool), n, n_eff, mean_p, obs, wilson_lo, wilson_hi, ret_p10, ret_p50, ret_p90, ret_p50_lo/hi(블록 부트스트랩), mdd_p10, mdd_p50, share_ep10_start
def state_table(states, y, fwd_ret_20, fwd_maxdd_20) -> pd.DataFrame
def episode_conditionals(spy_close: pd.Series, *, end=None, levels=SCENARIO_P3["dd_levels"], split=False) -> dict
    # {n, p_ge10_given5:(k,n,lo,hi), p_ge20_given5, p_ge15_given10, p_ge20_given10, depth_q{p90..p10}, extra_loss_q, breach_to_trough_q, trough_to_recovery_q, per_year, by_year, table(DataFrame)}
def current_drawdown(spy_close: pd.Series, asof) -> dict                 # dd_from_ath, ath_date, breached_5, breached_10, sessions_since_breach
def implied_range(spot: float, vix: float, h: int = 20, z=SCENARIO_P3["z"]) -> dict      # {"1s":(lo,hi), "80":(lo,hi), "90":(lo,hi), s}
def har_range(spot: float, har_fc_20: float, h: int = 20, z=...) -> dict
def coverage(fwd_ret_20: pd.Series, vix: pd.Series, har_fc: pd.Series) -> dict          # 과거·라이브 포함률(|z|≤1, ≤1.2816, ≤1.645, 좌꼬리)
def today_context(p_today, state_today, vix, har_fc, spot, tables: dict, dd: dict) -> dict   # 3줄 문장(한국어) + 오늘 행 + 라벨
```

## 8. 트랙레코드·드리프트·킬룰 — `mrl/track.py`

### 8.1 라이브 창과 채점

* **live_start(production)** = 장부에서 `prob_dd5_20` 이 있고 `p2_deploy_mode == "tones"` 인 첫 행(Phase 2 톤 가동일). 그 전 행(Phase 2 info_only 기간·홀드아웃 재현 행)은 모든 표에서 `재현 — 라이브 아님` 으로 라벨링되고 킬룰·채택에 들어가지 않는다. 그림자 멤버의 live_start 는 자기 열(`p3_p_h`, `p2_p_m1`)이 처음 기록된 날.
* 채점 행 = 라이브 창 안에서 `y_dd5_20` 이 backfill 된 행(asof ≤ 오늘 − 20세션). `d_t = (clim − y)² − (p − y)²`(> 0 이면 모델이 기후학보다 나음), `bss = mean(d)/mean((clim−y)²)`, 구간 = `bss_block_ci`(`ledger._bss_block_ci` 를 공개 함수로 승격: 짝지은 순환 블록 부트스트랩, block 40, 4,000회, seed 0; n_scored ≥ 40 일 때만). 같은 통계를 참조 M1·B1·BGK 와 각 등록 멤버에 대해서도 낸다.
* `hist_pct(bss, L)` = 라이브 BSS 가 같은 길이 L 의 과거 이동 창 분포(`summary_p3.json.reference.rolling_bss[L]`, step 21)에서 차지하는 백분위.
* 모든 라이브 통계는 **장부에서만** 다시 계산한다(새 백테스트에서 가져오지 않는다). 자료 개정은 D7·D10 이 잡지 기록을 바꾸지 않는다.

### 8.2 드리프트 경보(`alarms`; 플래그·기록만, 자동 재조정 없음)

| 코드 | 조건 | 참조(2003~24 설계 단계; `summary_p3.json.reference` 가 재산출) | 효과 |
|---|---|---|---|
| D1 p_level | 120세션 평균 p ∉ [0.04, 0.35] | p5 0.047 · p95 0.344 | 표시 |
| D2 bss | 756세션 BSS_clim < −0.05 → 빨강; 252세션 < −0.10 → 노랑 | 756: p5 −0.031·min −0.085·share<0 11.7% ; 252: p5 −0.094·share<0 30% | 빨강 = '킬룰 전 경고' 배지 |
| D3 vol_fc | 60세션 HAR log-MAE > 0.50 또는 60세션 mean ln(실현 RV20/σ̂_EWMA) ∉ [−0.35, +0.35] | p95 0.503 ; p5 −0.337·p95 +0.305 | 표시 |
| D4 vol_target | 60세션 규칙 수익률 실현변동성 / σ_T > 1.5 | p95 1.16 · max 1.49 | 비중 카드 정지 |
| D4b budget | 라이브 규칙 MaxDD < −D_max | — | 장부 항목 요구 |
| D5 stuck | 252세션 비정상 점유 > 0.75 또는 252세션 평균 w_exec < 0.30 | p95 0.758 ; p5 0.269 | 표시 |
| D6 churn | 252세션 비중 변경 > 24 | p95 19 · max 23 | 비중 카드 정지 |
| D7 parity | 최근 20세션 p·w_exec 재계산 |Δp| > 0.01 또는 w_exec 불일치 | — | **daily exit 1(커밋 없음)** |
| D8 coverage | 80% 밴드 적중률 < 0.5 (n_eff ≥ 12) | 93.6% / ~80% | 표시 |
| D9 hmm | 후행 756세션 P_high AUC < 0.5 | 0.581~0.697 | 게이지 숨김·멤버 rejected |
| D10 replay | 주간 재현 p·P_high·w_exec 와 장부 값 |Δ| > 1e-6 (개수) | — | 개수 표시; 원인(Yahoo 개정/코드) 로그 |
| D11 feature_range | x_vix·x_har·x_ma·x_hmm 중 하나가 1993~2024 범위 밖으로 ≥ 5세션 연속 | 범위는 `summary_p3.json.reference.feature_range` | 표시 |

임계는 `P3_DRIFT` 에 버전 고정; 변경은 번호 붙인 장부 항목. 경보 이력은 `results/alarms.csv`(asof, code, value, threshold). 카드 정지(D4/D6)는 소유자가 장부 항목을 기재할 때까지 지속.

### 8.3 킬룰(VALIDATION §7 의 실행 해석 — 라이브 행 기록 전 등록 → 사후 아님)

§7 원문 '≥5% 에피소드 8회 이상 또는 3년 경과 후(늦은 쪽)' 의 문자 그대로(둘 다)는 8회 도달까지 임의 시작점에서 중앙 7.9~8.8년(p10/p90 3.6/13.7; 36개월 창이 8회를 담는 비율 5~7%, 2003 이후엔 0)이라 가족의 첫 판정을 ~2033년으로 미룬다. 등록 해석(가족 쪽으로 더 엄격):

* **1차(36개월)**: 라이브 BSS_clim 점추정 ≤ 0 → `info_only`(킬, sticky); > 0 → `provisional`(톤·비중 유지, 배지 표시).
* **2차(최종)**: 36개월 이후 8회 도달 또는 **60개월** 중 먼저 오는 때; 같은 검정; 이후 12개월마다 재평가. 킬은 sticky.
* 라벨: `validated` = 2차 이후 부트스트랩 CI 하한 > 0(진짜 +0.09 모델도 36개월 창의 20% 에서만); `provisional` = 점추정 > 0 ≥ 하한; `info_only` = 점추정 ≤ 0; 그 전엔 `not_due`(12·24개월 중간 점검은 숫자만 보여 주고 킬할 수 없다 — §7 의 3년 하한). C 의 중대실패 조항(BSS ≤ −0.05 ∧ CI 상한 ≤ 0)은 중간 점검에서 '중대 실패' 배지로만.
* 멤버별: 등록 멤버(그림자 포함)도 같은 날짜·같은 통계로 판정; `killed` 멤버는 평균·구간에서 빠진다(정보; 생산 확률엔 영향 없음).
* 에피소드 = `targets.episodes(close, 0.05, split=False)` 규약, 라이브 시작 이후 **돌파일**(수중 구간 안에서 종가 ≤ 0.95·ATH 인 첫날) 기준, 시작 시점에 이미 진행 중이던 구간은 제외, 저점은 20세션 뒤 확정.
* 복귀: 자동 없음. 새 번호 장부 항목 + 킬 이후 ≥ 12개월 + **전체 라이브 창**(부분 창 금지)의 CI 하한 > 0. 입증 책임은 모델 쪽.
* 수동 킬: 소유자가 `results/kill_manual.json`(같은 필드 + ledger_entry 필수)을 만들면 `manual_kill`.

```python
def live_start(ledger: pd.DataFrame, col: str = "prob_dd5_20", mode_col: str = "p2_deploy_mode") -> pd.Timestamp | None
def episodes_realized(spy_close: pd.Series, start, asof, dd: float = 0.05, confirm: int = 20) -> tuple[int, list[str]]
def months_elapsed(start, asof) -> int                                        # 달력 월(Period 차)
def bss_block_ci(loss_p: np.ndarray, loss_ref: np.ndarray, block=40, n_boot=4000, seed=0, ci=0.95) -> tuple[float, float]
def score(rows: pd.DataFrame, p_col: str, ref_cols: dict[str, str], cfg=KILL_P3) -> dict
    # {n, n_eff, brier, base_rate, mean_p, bss_{clim,m1,vix,vix_bgk}, ci_bss_clim, ci_label('validated'|'undecided'|'rejected')}
def hist_pct(bss: float, L: int, reference: dict) -> float | None
def kill_status(ledger: pd.DataFrame, spy_close: pd.Series, asof, prev: dict | None, cfg=KILL_P3,
                members: dict[str, str] | None = None) -> dict
    # 반환 {state: not_started|not_due|provisional|validated|info_only|manual_kill, live_start, n, n_eff, months, episodes5, breach_dates,
    #       bss_clim, ci, ci_label, stage1_due, stage2_due, evaluated_now, killed, stage1_done, stage2_done, last_eval_month,
    #       countdown{episodes:'X/8', months:'Y/36', cap:'Y/60'}, gross_failure_badge(bool), members{name: {...score, verdict}}}
def kill_apply(status: dict, model_p3_path=MODEL_P3_PATH, record_path=KILL_RECORD_PATH) -> bool   # 멱등; 킬이면 deploy_mode='info_only' + kill_record.json
def kill_replay(oos: pd.DataFrame, spy_close: pd.Series, p_col: str, step: int = 21, cfg=KILL_P3) -> dict
    # 과거 OOS 기록에 live_start 를 21세션마다 놓고 1차 규칙을 재현: false_kill_share(M3 ~12%, M1 ~15%), validated_share(~20%),
    # bss quantiles(p10/p50/p90 −0.009/+0.065/+0.232), null_noise_pass(0%), null_block_shuffle_pass(~22% → 검정력 ~78%), months_to_8(p10/p50/p90)
```

핵심 논리(코드 계약):
```python
stage1_due = months >= 36
stage2_due = stage1_due and (n_ep >= 8 or months >= 60)
due_now = (stage1_due and not prev.stage1_done) or (stage2_due and not prev.stage2_done) \
          or (stage2_due and months - prev.last_eval_month >= 12)
killed = prev.killed or (due_now and not (bss > 0))              # 점추정 트리거; sticky
state = ("info_only" if killed else "validated" if (stage2_due and lo > 0) else "provisional" if stage1_due else "not_due")
```
**모드 스위치 역학**: `daily.py` 가 `kill_status(prev=model_p3.json["kill"])` 를 매일 부르고, 킬이면 `model_p3.json.deploy_mode = "info_only"` 와 `kill_record.json{asof, git_sha, registry_sha, spec_sha256, n, n_eff, bss, ci, ci_label, n_ep, breach_dates, months, ledger_entry:"7x"}` 를 쓴다(git 에 남는다). **`model_p2.json` 은 건드리지 않는다** — 유효 모드 = `p2.deploy_mode == "tones"` ∧ `p3.deploy_mode == "tones"` ∧ ¬`kill_record.json` 을 `daily.py`·`report.p3_card`·`selftest` 가 각자 계산하므로 주간 `run_calibration.py` 가 무엇을 다시 써도 킬이 풀리지 않는다. info_only 에서 카드는 상태·톤·비중·D_max 사다리를 숨기고 VALIDATION §7 문구 `정보 제공 전용 — 톤·비중 제안 숨김` 과 기록된 숫자(bss, CI, n_eff, 에피소드)를 보이며, 장부는 p·상태·w_vol 을 reason `info_only` 로 계속 기록하고(가정치), 시나리오·변동성·이벤트·귀속·게이지는 남는다. 주간 자기검사 `kill_replay` 가 12%/20%/78% 를 재생성해 카드 문장을 만든다: `이 규칙은 과거 36개월 창에서 진짜 +0.09 모델을 12% 오기각하고 '검증' 은 20% 에서만 준다; 정보 없는 같은 분포의 모델은 78%, 기후학+잡음은 100% 기각한다.` 카운트다운은 매일: `실현 ≥5% 에피소드 X/8 · 경과 Y/36개월(상한 60) · 현재 skill ±CI · 과거 동일 길이 창 대비 백분위`.

### 8.4 지평별 표시 규칙(문장이 코드로 고정; 테스트가 합성 장부로 검사)

| 경과 | 보이는 것 | 보이지 않는 것 | 고정 문장 |
|---|---|---|---|
| 0~3개월(n_eff ≤ 3) | 기록 세션·결측일, 경보, 평균 p vs clim, 상태 일수, 비중 경로(평균·최소·최대·변경 수 vs 기대 ~3), 규칙 vs 보유 수익(63세션 과거 밴드 −5.7/−1.0/+3.6pp 옆), 80% 밴드 적중 '3/3', 변동성 log-MAE, 그림자 멤버 값 | Brier·skill 숫자(회색도 없음), 신뢰도, 판정어 | `독립 창 3개 — 어떤 결론도 없음. 지금 보이는 것은 파이프라인이 매일 돌고 기록이 맞게 쌓인다는 것뿐. 과거 3개월 창의 skill 은 −0.17~+0.85(10/90분위)였다.` |
| 6개월(n_eff ≈ 6) | Brier/BSS vs clim·M1·B1 + CI, 2구간 신뢰도(p < clim / p ≥ clim), 실현 에피소드(기대 0~1), 드리프트 표, 비중 분포 vs 백테스트, hist_pct | 판정어 | `6개 창: 구간 폭이 skill 자체보다 넓다(과거 6개월 창의 36% 가 음수, 10/90 −0.12~+0.81).` |
| 12개월(n_eff ≈ 12) | BSS+CI 를 252세션 백테스트 분포(p10/p50/p90 −0.07/+0.06/+0.50; 29% 음수) 옆에, 풀링 신뢰도, 비중 연간 카드 vs 보유·p2, 밴드 포함률(Wilson), 실현변동성/σ_T(과거 p5~p95 0.42~1.16), 12개월 중간 점검(not_due) | 판정어 | `1년은 모델의 생사를 말하지 못한다: 같은 모델의 과거 1년 BSS 는 −0.07(하위 10%)에서 +0.50 사이였고 CI 하한 > 0 은 26~29% 뿐. 킬룰은 36개월.` |
| 36개월·판정일 | §8.3 판정·CI 라벨·모드 전환 기록, 과거 3년 창 분포(−0.009/+0.065/+0.232), 멤버별 판정, 신선 블록 채택 재검(≥ 11개 완결 시) | — | `≥8 에피소드로는 큰 실패만 걸러진다; +1~2% 우위는 인증 불가(P2 §16.4).` |

영구 각주: `라이브 기록은 해마다 독립 창 ~12개·에피소드 ~1개가 쌓인다. 판정일은 36개월(1차)과 8회 또는 60개월(2차); 그때까지 이 페이지는 일기이지 판결이 아니다.`

### 8.5 라이브-vs-백테스트 패널(`docs/track_record.html`; 같은 행에 n·n_eff·백테스트 참조)

(i) 평균 p·p > 0.30 비율 vs OOS(0.162 / ~10%) (ii) 상태 점유 vs 79/16/5% (iii) 결정 전환 vs 3.9/년, 비중 변경 vs 10.9/년 (iv) 비중 분포 vs 백테스트(평균 69%, 바닥 19%, 100% 7%) (v) 변동성 log-MAE vs p50 0.26(60세션), ln(실현/EWMA) 편향 vs [−0.34, +0.31] (vi) 밴드 포함률 vs 93.6%(VIX 80%) / ~80%(HAR) (vii) Brier·BSS vs clim/M1/B1 + CI 를 같은 길이 창의 백테스트 분포 옆에(252: p5 −0.094/p50 +0.064/share<0 30%; 756: −0.031/+0.065/12%), hist_pct (viii) 규칙 vs 보유 수익·MaxDD 를 같은 길이 과거 분포 옆에(63: −5.7/−1.0/+3.6pp, 뒤질 확률 79%; 126: −7.5/−2.1/+2.6, 88%; 252: −11.7/−5.9/−1.3, 90%; 252세션 MaxDD 중앙 −7.1%, 최악 −12.9%) (ix) 킬 카운트다운·중간 점검 (x) 경보 표·D10 불일치 수 (xi) 등록부: 멤버별 라이브 Brier, 불일치 폭 분포, 신선 블록 진행 'k/11'.

```python
def alarms(ledger: pd.DataFrame, reference: dict, cfg=P3_DRIFT, feats_today: pd.Series | None = None, asof=None) -> list[dict]   # [{code, value, threshold, action}]
def replay_check(ledger: pd.DataFrame, recomputed: pd.DataFrame, cols=("prob_dd5_20", "p3_hmm_p_high", "p3_w_exec"), tol=1e-6) -> dict
def horizon_copy(summary_p3: dict) -> dict            # {stage: '0-3m'|'6m'|'12m'|'36m', show: [...], hide: [...], sentences_ko: [...]}
def live_panel(ledger: pd.DataFrame, spy_close: pd.Series, reference: dict, asof) -> dict      # (i)~(xi)
def summary_p3(ledger: pd.DataFrame, spy_close: pd.Series, reference: dict, asof, prev_kill: dict | None) -> dict   # results/track_record_p3.json 의 단일 원천
```

## 9. 장부 — `mrl/ledger.py` (확장; 기존 열 순서 불변, `schema_version=3`)

`P3_COLUMNS` 를 `P2_COLUMNS` 뒤에 추가(구 행은 NaN; 알 수 없는 추가 열은 그 뒤에 보존):

```python
SCHEMA_VERSION = 3
P3_COLUMNS = [
    # 비중
    "p3_sigma_ewma", "p3_sigma_har_fc", "p3_sigma_target", "p3_d_max", "p3_w_vol", "p3_state_mult", "p3_w_target",
    "p3_w_exec", "p3_w_reason", "p3_next_check", "p3_sigma_down", "p3_sigma_up", "p3_deploy_sizing",
    # 국면·등록부·구간
    "p3_hmm_p_high", "p3_hmm_p20", "p3_hmm_q20", "p3_x_hmm", "p3_p_h", "p3_hmm_theta_id", "p3_hmm_gauge",
    "p3_members", "p3_lo", "p3_hi", "p3_band_src", "p3_disagree_flag",
    # 시나리오
    "p3_range_vix_lo", "p3_range_vix_hi", "p3_range_har_lo", "p3_range_har_hi", "p3_scen_bin", "p3_dd_from_ath",
    # 킬룰·경보·정합
    "p3_kill_state", "p3_kill_n_ep", "p3_kill_months", "p3_deploy_mode", "p3_effective_mode", "p3_alarms",
    "p3_registry_sha", "p3_sizing_sha", "p3_input_missing", "p3_run_id",
]
P3_STRING_COLUMNS = ("p3_w_reason", "p3_next_check", "p3_hmm_theta_id", "p3_hmm_gauge", "p3_members", "p3_band_src",
                     "p3_scen_bin", "p3_kill_state", "p3_deploy_mode", "p3_effective_mode", "p3_alarms",
                     "p3_registry_sha", "p3_sizing_sha", "p3_input_missing", "p3_run_id")
P3_PROB_COLUMNS = ("p3_hmm_p_high", "p3_hmm_p20", "p3_hmm_q20", "p3_p_h", "p3_lo", "p3_hi")          # [0,1] 검사
P3_W_REASONS = ("init", "weekly", "escalation", "hold", "input_missing", "info_only")
P3_KILL_STATES = ("not_started", "not_due", "provisional", "validated", "info_only", "manual_kill")
OUTCOME_COLUMNS_P3 = ["fwd_maxdd_20", "rv20_realized", "ret20_in_vix80", "ret20_in_har80", "rule_ret_20", "bh_ret_20", "p3_rule_dd"]
```

* `append_today` 는 P3 열을 정규화(숫자 NaN 허용, 문자열 object, `p3_members` 는 JSON 문자열 `{"p2": p, "M1": p, "H": p}`), 확률 열 [0,1]·reason·state 열거값 검사.
* `backfill(path, spy_close)` 확장: 20세션 뒤 `fwd_maxdd_20`, `rv20_realized`(종가 로그수익 20일 std·√252), `ret20_in_vix80`/`ret20_in_har80`(실현 fwd_ret_20 이 기록된 80% 밴드 안인가), `rule_ret_20`(장부 자신의 `p3_w_exec` 경로 × SPY 로 복리, 5bp), `bh_ret_20`; 매일 `p3_rule_dd`(규칙 라이브 자본곡선의 현재 낙폭, 라이브 시작부터). 완성 세션까지의 종가만 사용.
* `summary()` 추가 키 `p3` = `track.summary_p3(...)` 위임: `{schema_version, n_rows, n_scored, n_eff, live_start, months_elapsed, effective_mode, deploy_sizing, sizing{avg_w, n_changes_252, realized_vol_ratio, rule_dd, rule_vs_bh_20}, members{name: score}, disagreement{median_width, p90_width, flag_sessions}, kill{...kill_status...}, alarms[...], replay_mismatch_count, coverage{vix80, har80}, notes[]}`. 기존 `summary().p2.kill_rule`(문자 그대로의 '둘 다' 판독)은 남기되 페이지·카드는 `p3.kill` 만 쓴다(주석으로 명시).

## 10. 리포트 — `mrl/report.py` (추가)

```python
def p3_card(today_p3: dict, effective_mode: str) -> str          # index.html 의 Phase 3 카드(P2 카드 아래; v0·P2 카드는 그대로)
def render_sizing_report(summary_p3: dict, summary_v1: dict, summary_v0: dict, out_html: Path, charts: dict[str, bytes]) -> None   # docs/sizing_p3.html
def render_regime_report(summary_p3: dict, out_html: Path, charts: dict[str, bytes]) -> None                                       # docs/regime_p3.html
def render_track_record(track_p3: dict, summary_p3: dict, out_html: Path, charts: dict[str, bytes]) -> None                        # docs/track_record.html
def charts_p3(oos_p3, backtest_p3, spy_close, summary_p3, ledger_df) -> dict[str, bytes]
    # 비중 경로 밴드 + SPY · 규칙/보유/p2 결정층/v0 누적수익 · 낙폭 비교 · D_max 사다리(MaxDD-vs-σ_T 프론티어, 두 창) · 멤버 확률 + 불일치 밴드 vs SPY ·
    # HMM P_high 밴드 + θ 경로 · 시대별 AUC · 구간 표(백테스트 vs 라이브) · 라이브 BSS vs 동일 길이 과거 분포 · 경보 타임라인 · 킬 카운트다운
```

**일간 카드(`docs/index.html`, 한국어, 기존 다크 팔레트; P2 카드 아래)** — 위에서 아래로:
1. **숫자 하나 + 문장 하나**(A): `오늘 주식 비중 0.65 — 변동성 규칙 0.65 (목표 10% ÷ 예상 15.4%) × 상태 배수 1.0 (normal)`; 둘째 줄(B): `다음 점검 금요일 · 예상 변동성이 18.2% 를 넘으면 0.55, 13.3% 아래면 0.75 · 결정층이 격상되면 즉시 하향`. 변경이 있었으면 사유(`weekly`/`escalation`). 유효 모드가 info_only 면 이 블록 대신 회색 `정보 제공 전용 — 톤·비중 제안 숨김`(VALIDATION §7) + 기록 숫자.
2. **예산 줄**: `예산 −35%(완만한 약세장 기준) ≈ 급락형 −20% → 목표 변동성 10% · 2003~ 기록 −23.5% · 2000~02년형 −30.7%` + 사다리 링크; 정직 문장 (a)~(d) 의 축약 `변동성을 맞추면 파라미터 0개 규칙이 p2 결정층과 같다 · 낙폭 통제의 값은 보유 대비 −3.6pp/년`.
3. **확률 헤드라인(P2 카드의 숫자를 반복하지 않음)**: `모델들은 100일 중 L~H 일로 갈린다 (p2 M3 N · 보정 VIX M · HMM K; 불일치 W pp)`; 불일치 플래그.
4. **시나리오 3줄**(§7 순서: 시장 범위 → 오늘 같은 날 → 에피소드가 시작되면) + ATH 대비 현재 낙폭.
5. **국면 게이지**(그림자): 낮음/중간/높음, P_high·q20, 기대 체류, 라벨 `그림자 — 확률·비중·결정층·킬룰에 들어가지 않음`.
6. **정직 스트립**: 파라미터 줄 `생산 확률 적합 4/5 (Phase 2) · Phase 3 추가 0 · 그림자 H: Platt 2 (K_s) + HMM θ 12 (K_u) · 예산 밖 HAR OLS 4 · v0 Platt 2`, registry_sha·sizing_sha·model_id·theta_id, 유효 모드(p2 ∧ p3 ∧ ¬kill), 킬 카운트다운 `실현 ≥5% 에피소드 X/8 · 경과 Y/36개월(상한 60) · skill ±CI · 과거 동일 길이 창 백분위`, 지평 문장(§8.4), 경보 코드, D10 불일치 수, 신선 블록 `k/11`, 각주 `이 비중은 사전 등록 규칙을 SPY 100% 슬리브에 적용한 것이다; 가족의 현금·채권은 모델 밖 · 투자 조언 아님`.

**주간 `docs/sizing_p3.html`**: ① v0 completed 요약 줄(영구) ② p2 결정층 줄 ③ 정직한 읽기 (a)~(d) ④ 백테스트 표 §6.3(세 창) ⑤ D_max 사다리(두 창·프론티어 차트) ⑥ 민감도(선택에 쓰지 않음) ⑦ 에피소드 손익 ⑧ 유지 조건 (a)~(f) 판정 ⑨ 63/126/252 창 상대성과 분포 ⑩ 규약(5bp·현금 0%·점 원칙)·sizing_sha.
**주간 `docs/regime_p3.html`**: ① 등록부 표(멤버·K_s/K_u·상태·장부 번호·블록 기록·채택 판정·검정력 표·신선 블록 진행) ② 사다리 확장(M1→H, M3→H, M3→ens(M3,H), M3→ens(M3,M1), M3→M4a(#3 닫힘), B1→H; loss-diff CI·DM-HAC·위상 오프셋) ③ 블록 표(24·18) ④ θ 경로·guard 로그·체류·상태 변동성 ⑤ P_high 밴드 차트 ⑥ 시대별 AUC(P_high vs p_vix vs M3) ⑦ 소거(q20 Platt, A 관측형, B 동결형, 완전/대각 공분산) ⑧ 신뢰도(H, ens) ⑨ 결정론·PIT·카나리 결과·spec.
**주간 `docs/track_record.html`**: §8.4 지평 규칙에 따른 §8.5 패널 (i)~(xi) + 경보 이력 + 킬 기록.

## 11. 스크립트

* `scripts/run_phase3.py [--end (기본: HOLDOUT_START 직전 세션)] [--d-max 0.35] [--all-sensitivities] [--holdout-final] [--results-dir] [--docs-dir]`
  흐름: `load_cache → apply_guards → 완성 봉 자르기 → 하드컷(--end) → 읽기: calib_p2_walkforward.csv + summary_p2.json + model_p2.json + backtest_v1.csv (spec_sha256·feature_rule 일치 assert, 아니면 exit 1) → build_features(P2, 특징 범위 참조용) → make_targets → regime.observations → hmm_walk_forward(refit_dates, purge 20; warm start from hmm_p3.json) → ensemble.platt_walk_forward(H) → member_probs → admission_test(H, M1; BLOCKS_24·18) → 사다리 확장·block_scores·reliability/murphy/era_auc(H, ens) → combine/disagreement → decision.run(p2 배치 단; 병행 ens3 정보) → sizing: ewma_vol → target_vol(--d-max) → run → backtest_table(세 창; 참조 행 v0(results/backtest_v0_completed.csv, window_rule·signals_v0_sha256 일치 시)·p2 결정층·VT14) → sensitivities → retention → episode_pnl → rolling_relative → scenarios: bin_table/state_table/episode_conditionals/coverage → track.kill_replay(M3, M1, H) → reference 분포(window_distribution: rolling_bss 63/126/252/756, p_level, log-MAE, bias, occupancy, exposure, churn, vol_ratio, feature_range) → 결정론 assert(θ·Platt·backtest CSV vs 저장본, 1e-7/해시) → 저장`.
  산출: `results/oos_p3.csv`(date, y, clim, p_p2, p_m1, p_h, p_hmm_high, p20, q20, x_hmm, p_ens, lo, hi, state_p2, state_ens3, refit_year, block24, block18, theta_id), `results/backtest_p3.csv`(date, sigma_ewma, sigma_har_fc, state, mult, w_vol, w_target, w_exec, reason, ret_rule, ret_bh, dd_rule, + 민감도 열 `w_S_*`), `results/summary_p3.json`(run{generated_at_utc, end, spec_sha256, registry_sha, sizing_sha, d_max, sigma_target, python/pandas/numpy/sklearn, cache}, v0_reference, registry, ladder, blocks, admission, reliability, era_auc, hmm{theta_table, guard_log, timing}, decision_ens3, sizing{table, ladder, sensitivities, retention, deploy_sizing, episode_pnl, relative}, scenarios{bins, states, episodes, coverage}, kill_power, reference, flags, warnings, disclosure="#2 사전 관측 + 설계 단계 비중 관측 참조"), `results/model_p3.json`(schema_version, registry_sha, sizing_sha, deploy_mode, deploy_sizing, members{name:{status, K_s, K_u, ledger_no, admission{...}}}, platt{coef, intercept, refit_date}, sizing{d_max, sigma_target, k_slow, k_fast, ledger_no}, kill{...prev state...}, mode_history[], created_at_utc), `results/hmm_p3.json`, `docs/sizing_p3.html`, `docs/regime_p3.html`, `docs/track_record.html`.
  `--holdout-final`: `run_calibration.py --holdout-final` 과 **같은 세션**에서만(unlock 파일에 `members` 블록이 없고 같은 실행이 만든 timestamp 여야 함; 아니면 exit 2) 2024-09-03~ 를 R_2024/25/26 θ·Platt 로 채점해 `holdout_unlock.json.members{H, M1, ens}` 에 기록. **정보로만** — 채택엔 쓰지 않는다. #2b 가 이미 실행된 뒤라면 H 의 홀드아웃 점수는 '해제 후 관측' 으로 표기하고 역시 정보로만.
  주간 결정론 assert: registry_sha·sizing_sha 가 저장본과 같은데 θ·Platt·비중 경로가 1e-7·해시 밖이면 exit 1. sha 가 다르면(코드 변경) 새 파일을 쓰되 `run.spec_changed=true` 와 장부 기재 요구를 로그.
* `scripts/daily.py` (P2 블록 뒤에 추가; 재적합·채택 금지; 읽기 전용 모델)
  ```
  m3 = load_json(MODEL_P3_PATH); thetas, theta_live = regime.load_thetas()
  assert m3.registry_sha == ensemble.registry_sha256() and m3.sizing_sha == sizing.sizing_sha256()   # 아니면 exit 1: 코드가 산출물보다 새롭다
  effective = (p2.deploy_mode == "tones") and (m3.deploy_mode == "tones") and not KILL_RECORD_PATH.exists() and not KILL_MANUAL_PATH.exists()
  sigma = sizing.ewma_vol(spy_close_complete, asof=asof).loc[asof]            # 완성 봉 종가만
  prev = 장부의 마지막 유효 P3 행 (없으면 w_exec None, reason 'init')
  mult = sizing.state_multiplier(p2_state_today) if effective else NaN
  w_vol, w_target = sizing.exposure_target(sigma, mult, m3.sizing.sigma_target)  ; w_exec, reason = sizing.step(w_target, mult, prev.mult, prev.w_exec, is_period_end(asof,"W"))
  thr = sizing.next_thresholds(w_exec, sigma_target, mult, asof)
  obs = regime.observations(spy_close_complete, asof) ; p_high = regime.filter_probabilities(obs, theta_live).loc[asof]
  assert |p_high − one_step(prev.p3_hmm_p_high, obs.loc[asof], theta_live)| < 1e-7 (prev 있고 theta_id 같을 때)
  x_hmm, p_h = logit(clip(p_high)), sigmoid(a + b·x_hmm) ; k = regime.k_step(theta_live, p_high)
  lo, hi, src, flag = ensemble.disagreement({p2, M1, H}, statuses, p2_lo, p2_hi)
  scen = scenarios.today_context(p2_today, state, vix, har_fc, spot, summary_p3.scenarios, current_drawdown(...))
  al = track.alarms(ledger, summary_p3.reference, feats_today) ; D7 parity(최근 20세션 재계산) 실패 → exit 1
  ks = track.kill_status(ledger, spy_close_complete, asof, prev=m3["kill"]) ; track.kill_apply(ks)  # 킬이면 model_p3.json 갱신 + kill_record.json
  ledger.append_today(row | P3 열) ; ledger.backfill ; render_index(today | {"p3": {...}}, ls, out_html)
  GitHub 스텝 출력 추가: p3_w_exec, p3_reason, kill_state, alarms
  ```
  실패(exit 1): model_p3.json/hmm_p3.json 없음 · sha 불일치 · D7 · kill_record 가 있는데 유효 모드가 info_only 가 아님. 입력 결측은 실패가 아니라 `input_missing`(비중 유지) 경로.
* `scripts/selftest.py` 추가: registry_sha·sizing_sha 일치, θ guard 재검사, `smoothed_probabilities` 가 regime.py 밖에서 호출되지 않음(grep), 장부 P3 열 결측일, `alarms.csv` 읽힘, `kill_record.json` 과 `model_p3.json.deploy_mode` 정합, D_max 가 사다리 위(≥ 0.21), `summary_p3.json.run.d_max == model_p3.json.sizing.d_max`, 신선 블록 카운트.

## 12. 워크플로

* `.github/workflows/weekly.yml` — Phase 2 가 추가한 `run_backtest_v1.py` 단계 뒤에:
  ```yaml
      - name: scripts/run_phase3.py (HMM walk-forward → 그림자 멤버·채택 검정 → 비중 백테스트·사다리 → 시나리오 → 킬룰 재현 → results/ + docs/{sizing_p3,regime_p3,track_record}.html; 결정론 검사)
        run: python scripts/run_phase3.py --all-sensitivities
  ```
  실패 시 커밋 단계에 이르지 못한다. `timeout-minutes: 60` 유지. `--holdout-final` 은 워크플로에 넣지 않는다(로컬 1회, #2b 와 같은 세션, 커밋 메시지 `holdout-final: 2b`).
* `daily.yml` — 변경 없음(`daily.py` 확장; 커밋 제목 규칙 그대로).
* `requirements.txt` — 변경 없음(numpy EM; hmmlearn·statsmodels 불필요; scipy 는 이미 있음).
* 계산 예산(ubuntu 러너, 설계 단계 실측): HMM 22회 재적합 ~27초(einsum ξ, warm start; 상한 60초) · Platt 22회 < 3초 · 채택 검정·사다리 부트스트랩 4,000회 × ~12 비교 < 30초 · 킬룰 재현(223창 × 3 모델, 부트스트랩은 75창 × 1,000회) < 90초 · 비중 경로 ~15 변형 × 3 창 < 10초 · 참조 분포 < 5초 · 차트 < 30초 → `run_phase3.py` **< 4분**(주간 총계 여전히 ≪ 60분). 일간 추가 < 5초(EWMA + 8.5k 행 전방 필터 + 라이브 d_t 부트스트랩 1회; 10분 한도 무관).

## 13. 테스트 (필수)

* `tests/test_regime.py` — **EM 회복**: 합성 2-상태 자료(n=20,000, seed 0; A=[[.98,.02],[.05,.95]], 상태 변동성 1%/2.5%)에서 A 오차 < 0.02·변동성 오차 < 5%·μ 부호 회복 · **EM 결정론**: 같은 관측에 두 번 적합 → θ 비트 동일; warm start 와 결정적 초기화 모두 · **라벨 고정**: 초기 상태를 바꿔 넣어도 상태 1 = 큰 수익률 분산 · `n_params == 12`, π == stationary(A)(1e-12) · **PIT**: 20개 무작위 절단 T 에서 `filter_probabilities(obs[:T])` == 전체 결과의 [:T] 비트 동일 · **누수 카나리**: 실캐시 2003+ 에서 `smoothed_probabilities` 의 y_dd5_20 AUC 가 `filter_probabilities` 보다 ≥ 0.02 높음(필터가 실제로 돌고 있음을 증명); 합성 자료에서 평활 vs 필터 상태 정확도 차 > 0 · **지연 특징 이동 검사(leakage shift)**: x_hmm 을 +1/+5/+20세션 **앞당긴**(미래) 변형은 AUC 가 원본보다 엄격히 높고, −20/−250 지연 변형은 엄격히 낮으며 −250 에서는 원본 초과분의 25% 미만 — 원본이 그 사이에 있어야 통과(정보가 시점에 맞게 흐름) · **퍼지**: 모든 재적합에서 max(학습 pos)+20 < pos(R); 홀드아웃 관측이 어떤 학습 마스크에도 없음 · guard: 퇴화 θ(p00 0.999+, 점유 1%)를 넣으면 False + 사유, 이전 θ 유지 · k_step: k=1 이 A 행과 일치, q_k 단조 증가, k=20 닫힌 식 == 몬테카를로(seed 0, 1e-3) · 실캐시 walk-forward 22회가 < 60초·모든 해 guard 통과·p00/p11 ∈ [0.97, 0.99].
* `tests/test_ensemble.py` — REGISTRY 동결 해시(튜플 변경 시 실패해 장부 기재를 강제) · 파라미터 수 4/2/(2+12) · `combine`: admitted=∅ 이면 p2 와 비트 동일; 가용성 마스크·NaN 전파(멤버 NaN → 평균 제외가 아니라 admitted 멤버 NaN 이면 NaN) · `disagreement` 가 p2 밴드를 포함하고 killed 멤버를 제외 · `admission_test`: 합성 블록에서 8/11 ADMIT, 7/11 SHADOW, 8/11 이어도 풀링 CI 상한 ≤ 0 이면 SHADOW(거부권) · `fresh_blocks`: 라이브 시작부터 3달력월 타일, 라벨 미실현 블록은 미완결, 2024-09-01 시작으로 호출하면 ValueError(홀드아웃 금지) · 검정력 표 이항값(0.113/0.296/0.570/0.839) · **walk-forward vs 오프라인 parity**: OOS 무작위 10일 t 에 대해 `oos_p3.p_h[t]` == 그 해 θ·Platt 로 `observations(close[:t])` 를 필터·예측한 값(1e-12) · `apply_admission` 이 상태를 바꾸되 발효일 = 다음 1월 재적합.
* `tests/test_sizing.py` — EWMA 손계산(합성 수익률 5개) · seed 60 · `target_vol`: 0.35→0.10, 0.28→0.08, 0.30→0.08(내림), 0.20→ValueError · 사다리 두 예산선 21/28/35/42/52.5 · 클립·곱·재클립 범위 [0.25, 1] · **실행 의미론**(합성 경로): 밴드 안 진동(|Δ| < 0.10) → 무변경; 화요일 격상 → 즉시 하향(min 규칙), 격하는 금요일까지 대기; 어떤 5세션 창에서도 ≤ 3회(적대적 경로) · 결측 σ̂/상태 → `input_missing` 이며 w_exec 유지(1.0 복귀 금지) · **재개**: 장부 상태에서 `step` 으로 이어 계산 == `execute` 로 처음부터 계산 · **점 원칙**: 미래 행을 덧붙여도 w_exec[:t] 불변 · **백테스트 재현성**: `run` 두 번 → CSV 해시 동일; `allocation_from_weights(w)` == `allocation_sim(tone_df)` 톤 매핑 비중에서 비트 동일; 비용은 변경일에만 · 유지 조건 (a)~(f) 분기(합성 표) · 실캐시: 채택 규칙 2003~24 의 전환/년 ≤ 24·평균 비중 ∈ [0.5, 0.9]·실현변동성/σ_T ∈ [0.75, 1.15].
* `tests/test_scenarios.py` — 풀링: [.25,.30) n_eff 16 이 위 구간과 합쳐져 n_eff ≥ 20 이 될 때까지 · Wilson 이 n/20 기준 · 에피소드 조건부: 합성 종가(진행 중 에피소드 포함)에서 11/33 형 계수·split=False 규약·홀드아웃 고점 제외 · `implied_range`: VIX 15 → 1σ ±4.2%, 80% ±5.4%; VIX 14.53 → ±4.1/±5.2/±6.7% · `coverage` 계수 · `today_context` 가 낙폭 상태별로 다른 줄을 고르고 n_eff < 20 행을 절대 단독 표시하지 않음.
* `tests/test_track.py` — `episodes_realized`: 시작 시점 진행 중 에피소드 제외, 돌파일 기준, 20세션 확정 · `months_elapsed` · `score`/`bss_block_ci` 부호(합성 d) · **킬룰 단위 테스트(합성 장부)**: 30개월 → `not_due`; 36개월·BSS > 0 → `provisional`(evaluated_now, stage1_done); 36개월·BSS ≤ 0 → `info_only` + `kill_apply` 가 deploy_mode 를 한 번만 바꾸고 멱등; 8회·CI 하한 > 0 → `validated`; 60개월·7회 → 2차 실행(상한); 2차 뒤 12개월 재평가 리듬(11개월엔 미실행); sticky(뒤 창이 좋아도 killed 유지); 멤버 killed 가 구간에서 빠짐; 수동 킬 파일; 복귀는 함수로 불가(문서 절차만) · **재현 테스트**: `kill_replay(oos_p3, M3)` 의 false_kill_share ∈ [0.10, 0.14], validated_share ∈ [0.15, 0.25], null_noise_pass == 0, null_block_shuffle_pass ∈ [0.15, 0.30](공식 수치 확정 후 ±2pp 로 조인다) · 경보: 임계를 넘는 합성 장부에서 각 코드가 정확히 자기만 발화, 참조 분포 자체에선 무발화 · D7 이 exit 1 경로 · 지평 문구: 3/6/12/36개월 합성 장부에서 n_eff < 6 에 BSS 숫자 없음·판정어는 due 전 없음.
* `tests/test_ledger.py` 확장 — P3 열 왕복·구 행 NaN·`p3_members` JSON·열거값 검사·backfill 의 `rule_ret_20`(장부 경로에서 복리)·`p3_rule_dd`·`ret20_in_vix80`.
* `tests/test_report.py` 확장 — `p3_card` 가 `BAD_TOKEN` 없이 두 모드로 렌더; info_only 에서 비중·상태·사다리 숨김과 §7 문구 노출; 지평별 문장 규칙; sizing/regime/track_record 페이지 섹션 전부 존재; 파라미터 줄 텍스트 정확.
* `tests/test_scripts_integration.py` 확장 — `run_phase3.py --end 2006-12-29`(tmp results/docs; Phase 2 통합 픽스처의 calib_p2_walkforward.csv·summary_p2.json·backtest_v1.csv 사용)가 모든 산출물을 쓰고 summary 가 엄격 JSON·키 계약을 만족·홀드아웃 하드컷 준수 · `--holdout-final` 이 unlock 파일 부재/불일치 시 exit 2 · `daily.py --no-update` 가 P3 장부 열과 카드를 쓰고, `kill_record.json` 을 넣으면 카드가 info_only.

## 14. 파라미터 회계 (카드 정직 스트립·등록부 페이지에 이 표를 그대로)

| 줄 | 객체 | 지도(K_s, 라벨 대면) | 비지도(K_u) | 생산 확률에? | 예산 줄 |
|---|---|---|---|---|---|
| 생산 확률 p2 | b0, b1(x_vix), b2(x_har), b3(x_ma) | **4** | 0 | 예 | VALIDATION §6 (상한 5; 1 슬롯 예약) — Phase 3 변경 없음, `PARAM_COUNT==4` assert 유지 |
| 결정층 | 밴드 1.5/1.2·2.5/2.0, dwell 5 | 0 | 0 | — | Phase 2 상수 동결 |
| 비중 규칙(§6) | λ .94, seed 60, 바닥 .25/상한 1, k_slow 3.5·k_fast 2.0, 격자 {6..15}%, D_max .35, 격자 .05, 밴드 .10, 주간+격상, 배수 = TONE_EXPOSURE, 비용 5bp·현금 0 | 0 | 0 | — | §6-b: 적합 0 필수. ~40 변형을 1993~2024 로 본 뒤 선택 → **post hoc**, 재조정 금지 |
| 시나리오·경보·킬룰 | 구간·풀링 20·z·D1~D11 임계·36/8/60/12·블록 40·4,000·seed 0 | 0 | 0 | — | 상수(설계 단계 관측 후 고정; 경보 임계는 p5/p95 에서 → post hoc) |
| 그림자 M1 | b0, b1 | 2 | 0 | 아니오(#8) | 등록 시험(Phase 2 사다리 재사용) |
| 그림자 H | Platt a, b / HMM μ 4·Σ 6·A 2 | 2 | **12** | 아니오(#3a) | 등록 시험; K_u 는 라벨을 보지 않지만 **파라미터로 공개** |
| 앙상블 가중 | 동일 | 0 | 0 | — | 상수 |
| 예산 밖(Phase 2, 불변) | HAR OLS 4 · v0 Platt 2 | 표시 전용 | | 아니오 | |

생산 확률에 닿는 적합값: **4(예산 안) + 0(Phase 3)**. 가족이 보는 어떤 숫자에든 닿는 적합값 합계: 4 + 0 + 4(변동성 표시) + 2 + 2 + 12(그림자) + 2(v0 줄) = 26, 그중 예산 안 4. **§6 는 오늘 문자 그대로 성립한다**(개정 불필요). 개정(§6-c)이 필요해지는 유일한 경우는 그림자 멤버가 신선 자료로 채택될 때(M3+H = K_s 6 · K_u 12; M3+M1 = 6)이며, 그 자료는 아직 존재하지 않으므로 개정 문구를 **지금** 쓰는 것은 사후가 아니다 — 다만 후보와 규칙 자체가 #2 사전 관측을 본 뒤 설계됐다는 약한 의미의 post hoc 은 장부에 적는다. 설계 자유도(파라미터가 아니라 선택; 검토자가 할인할 수 있게 나열): HMM 관측형(C) · Platt 특징(P_high) · 바닥 재클립 · 주간 리듬 · k_slow 3.5 · 풀링 20 · 2단계 킬 해석 · 경보 임계.

## 15. 단계별 도입 (지금 짓는 것 vs 장부가 쌓여야 하는 것)

| 단계 | 시점 | 짓는·보이는 것 | 금지 |
|---|---|---|---|
| 0 구축 | 지금(Phase 2 에이전트 완료 후, 홀드아웃 미접근) | 순서: config → regime + 테스트 → ensemble + 채택 검정 → sizing + evaluate 추가 → scenarios → track → ledger/report → run_phase3.py(주간 산출·결정론) → daily.py 배선 → selftest → VALIDATION 항목. 산출: oos_p3·backtest_p3·summary_p3·model_p3·hmm_p3·세 페이지. 유효 모드는 p2 결과에 종속 — **#2a 소유자 결정 전(또는 p2 info_only)에는 비중·상태 카드가 뜨지 않고** 변동성 단독 규칙·게이지·시나리오·구간만 회색/정보로 보인다. 홀드아웃 채점은 #2b 와 같은 세션 1회(정보) | 홀드아웃 접근(#2b 외), 상수 재조정, 근사 수치를 공식 표에 섞기 |
| 1 가동 | Phase 2 톤 가동일 = live_start | 카드: 비중 한 숫자·한 문장, 예산 줄, 불일치 구간, 시나리오 3줄, 게이지, 정직 스트립; 장부 P3 열 매일; 그림자 H·M1 매일 채점; 경보 D1~D11; 킬 카운트다운 | Brier·skill 숫자, 판정어 |
| 2 0~6개월 | n_eff < 6 | 계수·경보·비중 경로·에피소드 수·밴드 적중·변동성 log-MAE; 주간 D10 재현; 첫 라이브 vs 백테스트 상대성과(63세션 밴드 옆) | skill 숫자(회색도 없음), 상수 변경 |
| 3 6~12개월 | n_eff 6~12 | BSS ± CI(고정 문구 동반), 2구간 신뢰도, hist_pct, 12개월 중간 점검(not_due) | 판정어, 등록부 변경 |
| 4 12~36개월 | | 풀링 신뢰도 라이브 열, 비중 연간 카드, 24개월 중간 점검, 신선 블록 진행 k/11, D4b 예산 검사 | 킬·채택·복귀 |
| 5 36개월(1차) | | 킬룰 1차(≤ 0 → info_only sticky / > 0 → provisional); 신선 블록 11개 완결(≈33개월+20세션) 뒤 첫 채택 재검 → 통과 시 다음 1월부터 §6-c 발효·K_s 갱신 | 부분 창 판정 |
| 6 8회 또는 60개월(2차) | 이후 12개월마다 | 최종 라벨(validated/provisional/info_only), 멤버별 판정, 채택 재검 4블록마다; 복귀 = 새 장부 항목 + 전체 라이브 창 CI 하한 > 0 | 자동 복귀, 홀드아웃 재사용 |

지금 짓지만 **12~24개월의 장부 없이는 의미가 없는 것**(코드는 완성하되 페이지에서 회색·'미확정'): 라이브 BSS·CI·hist_pct(≥ 6개월), 구간 표 라이브 열(≥ 20 창 ≈ 12개월), 예산 준수 판정 D4b(≥ 36개월), 채택 재검(≥ 33개월), 킬 판정(≥ 36개월), 멤버 killed(≥ 36개월). 지금 짓고 **즉시 의미 있는 것**: 비중 규칙과 그 백테스트(모델 무관), 사다리, 시나리오 표, 시장 범위, 경보 파이프라인, 그림자 기록, 재현 자기검사.

## 16. 정직 문구와 리스크 (리포트·카드 정직 스트립에 상시)

1. **Phase 3 는 skill 을 더하지 않는다.** 어떤 후보도 보정 VIX 위에 측정 가능한 Brier 이득이 없다(M3→p3_A −0.0002, ens(M3,H) +0.0000, M4a 무이득, 폭/신용 −0.055). 가치는 비중 규칙·시나리오·구간·트랙레코드·킬 장치이며 모든 페이지 첫 줄에 그렇게 쓴다.
2. **Phase 2 의 공식 결과가 Phase 3 의 전제다.** 2026-09-07 실행에서 §6 문자 그대로의 규칙은 실패(블록 3·8·9)해 `model_p2.json` 은 info_only·tone_model None 이고, #2a 완화안(사후)으로는 M1 만 A∧B∧C 를 통과한다. 소유자가 #2a 를 기재하기 전에는 유효 모드가 info_only 이므로 **비중·상태 카드는 뜨지 않는다**; 기재하면 seed = M1 이고 결정층·비중은 M1 의 상태로 돈다(설계 단계 비중 수치는 M3 상태 기준이라 공식 수치가 조금 다를 수 있다). 이 결정은 Phase 3 코드가 아니라 장부에서 내려진다.
3. **비중 규칙의 가격**: σ_T 10% 에서 보유보다 ~3.6pp/년 낮고 달력연도의 ~90% 에서 뒤진다; 현금 0%(+0.6pp/년 미반영); 5bp 는 분기별로 늦게 실행하는 가족의 실제 마찰(중개·세금)을 과소평가한다. 가족이 긴 평온기 뒤 최악의 시점에 규칙을 조용히 버릴 위험 — 가격을 카드에 두고, 사다리로 더 높은 예산 행(12~15% → 평균 비중 74~80%)을 고를 수 있게 하며, p2 결정층(89% 비중)이 같은 페이지의 가벼운 대안으로 남는다.
4. **완만한 약세장 실패 모드**: 2000~02 변동성 단독 −30.7~−34.6%(σ_T 10%); k_slow 3.5 는 한 에피소드로 보정한 상수라 1970년대형 침체는 넘길 수 있다 — '예산은 보장이 아니다' 를 사다리 위에 둔다. 저변동 급락(2018-02, 2020-02 첫 주)은 규칙이 반응하기 전에 맞는다(비중 ~90%); 돌파일 결정 상태는 대부분 normal(11/19).
5. **바닥의 대가**: 곱 뒤 0.25 재클립은 MaxDD 를 −17.9% 에서 −23.5% 로 올린다(변동성 단독과 같다). 상태 배수의 낙폭 기여는 바닥 없이만 나타나고 그 형태는 2008~09·2020 에 사실상 퇴장(비중 0.05~0.10)이었다. 두 수치를 나란히 두고, 바닥 없는 변형은 민감도 행으로만 남긴다.
6. **모든 비중 상수는 post hoc**(λ, 격자, 밴드, 리듬, k_slow, 바닥 처리; ~40 변형). 합리적 변형 사이의 차이는 작아(−23~−24%, 바닥 없으면 −16~−18%) '규칙의 가족' 으로 읽고 재조정하지 않는다. 유지 조건 (a)·(b) 는 근소해 공식 수치에서 뒤집힐 수 있다 — 그러면 카드가 숨겨지고 장부만 쌓인다.
7. **킬룰 검정력**: 진짜 +0.09 모델도 36개월에 ~12% 오기각, `validated` 는 20%; 무정보 모델(블록 셔플 M3)은 78% 기각, 기후학+잡음은 100%. 2단계 해석은 문자 그대로보다 엄격하다(의도적, #7 문서화). 주간 자기검사가 이 수치를 재생성하지 않으면 페이지에 쓰지 않는다.
8. **소표본 영구**: 연 ~12 독립 창·~1 에피소드; 3·6개월 skill 은 잡음(10/90분위 ±0.8)이라 억제한다 — 가족이 답답해도 규칙. 행운의 첫해 뒤 문구를 빼고 싶은 유혹은 테스트(n_eff < 6 에 BSS 없음, due 전 판정어 없음)로 막는다.
9. **HMM**: 라벨 스위칭(분산 순서 고정), EM 국소해(결정적 초기화·warm start), 지속성 0.98 이라 급락을 늦게 본다(P2 §16.3 과 같은 실패 모드), 점유 43% 는 '위기' 가 아니라 '고변동 상태'(카드 명칭), 비지도 12개도 파라미터로 공개, 러너 BLAS 차이(1e-7). AUC 0.697 이 x_vix 보다 커 보여도 ~270 독립 창의 한 숫자이고 같은 정보(실현변동성 지속성)다. 등록부는 몇 년째 안 바뀌어 보일 것 — 검정력 표로 설명.
10. **데이터 개정**: Yahoo 조정 종가 재다운로드가 σ̂·P_high·비중 이력을 조금 바꾼다 → D10 이 불일치를 세고 D7 은 daily 를 실패시킨다; 장부는 기록이며 다시 쓰지 않는다. HMM 관측은 종가만 쓰므로 1993~95 OHLC 품질 문제는 없다.
11. **결합 위험**: 열 이름(`prob_dd5_20`, `p2_p_m1`, `p2_state`, `p2_clim`, `p2_har_fc_20`, `p2_deploy_mode`, `p2_tone_model`, `p2_lo/hi`)과 spec_sha256·registry_sha·sizing_sha 가 계약; 불일치는 exit 1, 대체 없음. 킬은 `model_p2.json` 을 건드리지 않고 유효 모드의 AND 로만 작동하므로 주간 재적합이 킬을 풀 수 없다.
12. **홀드아웃**: H 의 홀드아웃 점수는 #2b 와 같은 세션에서만 1회 채점되고(M3/M1 은 두 번째 관측 — 라벨링) 채택엔 쓰지 않는다. 신선 블록은 라이브 시작부터만.
13. **제품 위험**: 확률·상태기계·변동성 규칙 세 겹은 v0 톤보다 설명이 어렵다 → 카드는 한 숫자·한 문장으로 시작하고, 정직 스트립은 접힌다. 시나리오 셀은 얇다(reduce n_eff 12, 0.40 이상 n_eff < 8, 에피소드 33/11/4) — 회색·풀링 규칙은 산문이 아니라 코드로 강제.
14. **계산**: 주간 +< 4분, 일간 +< 5초; 새 의존성 없음. EM tol 1e-6 에서 반복 14~48회; 1e-9 로 조이면 ~3배지만 예산 안.

## 17. VALIDATION.md 변경 (§5/§6/§7 개정 문구 + §8 장부 항목; post hoc 표시)

기존 항목은 지우지 않는다. 아래 문구를 소유자가 그대로 옮긴다(수치는 `run_phase3.py` 산출로 채우고, 설계 단계 근사는 '사전 관측' 으로만 남긴다).

**§5 추가(6번 항목)**: `6. (Phase 3부터) 비중 성적표: CAGR·MaxDD(일자)·최악 월·연변동성·전환/년·평균 비중·누적 비용·실현변동성/σ_T·MaxDD/σ_T — 보유·v0·p2 결정층·변동성 일치 참조(VT14) 옆에; D_max 사다리는 1993~(변동성 단독)·2003~(결정층 포함) 두 창의 MaxDD 를 두 예산선(3.5×σ_T·2.0×σ_T)과 나란히. 시나리오 표는 n·n_eff(=n/20)·Wilson 없이 표시 금지, n_eff < 20 은 풀링.`

**§6 추가 문단 6-b (Phase 3 비중; 사후 표시)**: `비중 규칙(Phase 3 sizing)은 확률 모델의 적합 파라미터 예산(5개)에 포함되지 않으며 적합 파라미터 0개여야 한다. 사전 등록 상수: EWMA λ=0.94(seed 60세션), 비중 하한 0.25·상한 1.0, σ_T = D_max/3.5 를 격자 {6,8,10,12,15}% 로 내림(급락형 예산선 2.0×σ_T 병기; 기본 D_max=0.35 → 10%), 격자 0.05, 무거래 밴드 0.10, 주 마지막 세션 갱신 + 결정층 격상 시 즉시 하향, 상태 배수 = TONE_EXPOSURE(1/.5/.25), 곱 뒤 [0.25, 1] 재클립, 비용 5bp·현금 0%. 이 상수들은 1993~2024-08 기록을 보고(~40개 변형) 정했으므로 그 기록에 대해 사후(post hoc)이며, 라이브 장부 자료로 재조정하지 않는다. 상수·D_max 변경은 번호 붙인 장부 항목으로만. 유지 조건 (a)~(f)(ARCHITECTURE_PHASE3 §6.4)를 공식 산출로 판정해 위반 시 비중 카드를 숨긴다. 유효 배치 모드가 아니면(p2 info_only 또는 §7 킬) 비중을 제안하지 않는다.`

**§6 추가 문단 6-c (등록부·앙상블; 지금 작성, 발효는 신선 자료 채택 시)**: `(a) 생산 확률은 사전 등록 멤버의 동일가중 평균일 수 있다. 적합 파라미터 상한 5개는 멤버 각각의 지도 사상(특징→확률)에 적용하고, 멤버 합계 K_s 는 12개 이하로 카드에 공개한다. (b) 라벨을 보지 않는 비지도 상태모형(HMM θ)의 파라미터는 별도 줄 K_u(상한 15)로 세어 공개하고 같은 1월 일정·20일 퍼지로 walk-forward 재추정하며 K_s 에 섞지 않는다. (c) 동일 가중·가용성 마스크·결정층 밴드·비중 상수·시나리오 구간은 상수이지 적합값이 아니다. (d) 멤버는 ARCHITECTURE_PHASE3 §5.4 의 채택 규칙(라이브 장부 시작부터 3개월 신선 블록 ≥ 11개 완결, ≥ ceil(8/11·n) 블록 개선 ∧ 풀링 손실차 CI 상한 > 0, 순서 H→M1, 이후 4블록마다 재검, 발효는 다음 1월)으로만 생산 평균에 들어간다; 2003~2024-08 기록과 홀드아웃은 기각만 할 수 있다. 정직 표기: 모든 후보와 이 규칙은 #2 사전 관측을 본 뒤 설계됐고(약한 의미의 post hoc), 관측 기록의 채택 결과는 H 6/11·M1 5/11·M4a 3/11 로 아무것도 통과하지 못하므로 이 문단은 오늘 아무것도 바꾸지 않는다(1일차 생산 파라미터 4/5).`

**§7 각주(실행 해석; 라이브 P3 행 기록 전 등록 → 사후 아님)**: `'8회 이상 또는 3년 경과 후(늦은 쪽)' 의 실행 해석(#7): 1차 평가 = 36개월(y_dd5_20 BSS_clim 점추정 ≤ 0 이면 정보 제공 전용·sticky; > 0 이면 '잠정'), 2차(최종) = 36개월 이후 8회 도달 또는 60개월 중 먼저, 이후 12개월마다; '검증' 라벨은 블록 부트스트랩(40, 4,000회, seed 0) 95% 하한 > 0 일 때만; 12·24개월 중간 점검은 킬할 수 없다. 에피소드 = split=False 규약, 장부 시작 이후 돌파일 기준, 저점 20세션 확정. 등록 멤버도 같은 통계로 판정(정보). 복귀 = 새 장부 항목 + 킬 후 ≥ 12개월 + 전체 라이브 창 CI 하한 > 0. 사유: 문자 그대로의 '늦은 쪽' 은 8회 도달까지 중앙 ~8년(1.04회/년·군집); 36개월 오기각 ~12%·검증 20%·무정보 모델 기각 ~78% 를 매주 재생성해 공개.`

**§8 장부 행**:

| # | 날짜 | 변경 | 사유 | 결과 | 채택 |
|---|---|---|---|---|---|
| 3 | (갱신 2026-09-07) | 5번째 슬롯 M4a(M3 + b4·x_hmm, 5 파라미터, 사다리 단 M3→M4a): 사전 관측 b4 ∈ [−0.10, +0.02] · 3/11 블록 · BSS 0.0901 vs 0.0910 · CI [−0.0005, +0.0003] → **무이득·미채택**; 공식 수치는 run_phase3.py 사다리 확장으로 채점만 | x_vix·x_har 와 공선 (post hoc 관측) | (채움) | 미채택 → #3a 로 대체 |
| 3a | 2026-09-07 | 그림자 멤버 H: 2-상태 완전공분산 가우시안 HMM on [100·r, ln RV10], numpy EM(결정적 초기화·warm start·guard), 1월 퍼지 재적합, 전방 필터 P(고변동), Platt 2(x_hmm = logit P_high); 매일 장부 기록·채점, 생산 확률 밖, 불일치 구간 멤버; 승격은 §6-c(d) 신선 자료 채택으로만 | 2020~24 시대의 VIX 판별력 감쇠를 보완(AUC 0.666 vs 0.594; 풀링 0.697 vs 0.688), 단 Brier 이득 0 (post hoc 관측) | (run_phase3.py: 블록 기록·채택 검정·guard 로그·홀드아웃 정보 점수) | 그림자 |
| 6 | 2026-09-07 | Phase 3 비중 사양(§6-b): clip(σ_T/σ_EWMA94, .25, 1) × TONE_EXPOSURE(p2 결정층 상태), 곱 뒤 [.25, 1] 재클립, σ_T = D_max/3.5 격자(기본 35% → 10%; 급락형 2.0× 병기), 5pp 격자·10pp 밴드·주간 + 즉시 격상, 유지 조건 (a)~(f) | §6 예산 불변(4/5), 새 적합 0. **사전 관측 공개(설계 단계 근사, 2003-01-02~2024-08-30, M3 상태, 5bp)**: 보유 10.85%/−55.2%; p2 결정층 10.19%/−31.0%/3.9회; 채택 규칙 7.22%/−23.5%/최악 월 −7.8%/10.9회/비중 69%; 바닥 없음 6.94%/−17.9%; 변동성 단독 7.84%/−24.1%; VT14 10.14%/−30.3%; 1993~ 변동성 단독 10% −30.7%(주간)·−34.6%(0.25 격자) → k_slow 3.5. 상수는 ~40 변형을 본 뒤 선택 → post hoc | (run_phase3.py 산출로 채움; 배치 단의 상태로 재계산; 유지 조건 (a)~(f)) | (채움) |
| 6a | (같은 날) | 비중 민감도(선택에 쓰지 않음): 바닥 없음(B 원안), C 월간 × TONE_EXPOSURE, C 월간 × (1/.75/.5), A 0.25 격자·min 규칙, VT14 변동성 일치 참조, 즉시 격상 없음, 1세션 지연, 10bp, S-HAR, 일간 5pp, 목표 격자 6~15% | 보고 전용 | (채움) | — |
| 7 | 2026-09-07 (라이브 P3 행 기록 전) | §7 실행 해석: 2단계(36개월 1차 · 8회 또는 60개월 2차 · 12개월 재평가), 점추정 트리거, CI 라벨(validated/provisional/info_only), 돌파일 에피소드·20세션 확정, 멤버별 동일 판정, 복귀 규칙, 수동 킬 파일, 킬은 model_p2.json 을 건드리지 않음(유효 모드 AND), 허용오차 1e-7 | 문자 그대로는 중앙 ~8년; 36개월 오기각 ~12%·검증 20%·무정보 기각 ~78% 를 주간 자기검사로 재생성 | (kill_replay 산출로 채움) | 사전 등록(사후 아님) |
| 8 | 2026-09-07 | §6-c 등록부·채택 규칙·신선 블록(라이브 시작부터)·검정력 표; 그림자 멤버 M1(p2 사다리 재사용, Platt 2) 등록; 폭/신용 후보 B 는 미등록(단독 BSS −0.055·AUC 0.495, 2010~24; 사상 불안정) | 앙상블을 5개 한 줄에 넣을 수 없음; 관측 기록은 기각만(H 6/11·M1 5/11·M4a 3/11) | (run_phase3.py: 채택 검정 표) | 문구만 발효(1일차 admitted = ∅, 생산 4/5) |
| 9 | 2026-09-07 | 드리프트 경보 D1~D11 임계(P3_DRIFT; 2003~24 참조분포 p5/p95)와 지평별 표시 문구 규칙(0~3/6/12/36개월; n_eff < 6 에 BSS 없음, due 전 판정어 없음) | 임계는 관측 후 고정 → post hoc; 라이브로 재조정 금지; 변경은 번호 항목 | (summary_p3.json.reference) | 최초 배치 시 고정 |
| 10 | 2026-09-07 | 시나리오 표 A~D 와 표시 규칙(위에서부터 n_eff ≥ 20 풀링, 에피소드 split=False 33회·홀드아웃 고점 제외, 시장 내재 범위 포함률 84/94/97%, HAR 범위 64%, 3줄 순서, '분포는 넓어진다' 문장) | 세는 것뿐, 적합 없음 | (채움) | 표시 규칙 |

이 문서의 근사 수치 출처(스크래치, 리포 밖): 심사 재현 `scratchpad/judge/judge_{common,hmm,sizing,kill}.py`·`judge_p3/judge_{base,verify}.py`, 설계 B `apprB_run2.py`·`apprB_run3.py`, 설계 A `p3_check*.py`, 설계 C `p3c_*.py` — 하드컷 2024-08-30, 홀드아웃 미접근. 공식 수치가 채워지면 근사는 '사전 관측' 열에만 남는다.
