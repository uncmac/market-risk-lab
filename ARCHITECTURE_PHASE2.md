# market-risk-lab — Phase 2 모듈 계약 (보정 모델 `p2` / 결정층 `v1`)

목적: Phase 1 이 동결한 v0 벤치마크 위에, **적합 파라미터 ≤ 5개**의 보정 모델로 주 목표변수 `y_dd5_20`(다음 20거래일 안에 종가 -5% 이하)의 **확률**을 만들고, 그 확률을 (1) 기후학(기저율)과 (2) VIX 내재 확률 대비 Brier skill 로 채점하고, (3) 히스테리시스·최소 체류를 갖는 3단계 결정층으로 가족용 톤에 매핑하며, (4) HAR-RV 변동성 예측·이벤트 달력·일간 변화 귀속을 보조 출력으로 낸다.
v0(`reference/`, `mrl/signals_v0.py`, `mrl/config.V0*`)은 한 줄도 건드리지 않는다. v0 결과는 모든 Phase 2 리포트 첫 줄에 영구 표기한다(VALIDATION §4).

## 0. 설계 채택과 접목 (심사 결과의 해석)

세 설계의 심사 합계: **P2-B 124**(39/44/41) · P2-A 114(36/37/41) · C 84(28/26/30). 기반 설계 = **P2-B "vol-anchored calibration"** (x_vix + GK-HAR 갭 + MA180, 4-파라미터 중첩 로지스틱). 심사위원이 권고한 접목을 아래 표대로 반영하고, 충돌은 명시적으로 해소한다.

| 항목 | P2-B(기반) | 접목/충돌 | 최종 결정 |
|---|---|---|---|
| 확률 모델 | 4-파라미터 중첩 로지스틱 M0→M1→M2→M3 | A: 고정가중 점수+Platt(2), C: HMM 앙상블 | **B 유지.** 5번째 슬롯은 비워 두고 장부에 예약(#3~#5). HMM·구리/금·SOX 는 Phase 3 후보로만 등록 |
| 실현변동성 추정 | Garman-Klass + 야간갭(GK+OV) | A: Parkinson | **GK+OV.** Parkinson 은 이름 붙인 소거실험(M3-PK)과 1993~95 OHLC 품질 자기점검으로만 |
| VIX 내재 벤치마크 식 | 드리프트 m=-s²/2 반사원리 | A/C: 무드리프트 2Φ(b/s) | **B 식 = B1(사전 등록 벤치마크).** 무드리프트·BGK 이산관측 보정판은 같은 함수의 옵션으로 열에 병기(차이 <1pp) |
| 기후학 | 재적합 연도의 퍼지된 학습창 평균(연 1회 갱신) | A: 매일 갱신(21일 지연 확장평균) | **B.** 결정층 밴드가 clim 배수이므로 연 1회 상수여야 재현·재개가 결정적 |
| 결정층 밴드 | r=P/clim 배수 (1.5/1.2, 2.5/2.0), 대칭 dwell 5 | A: 절대확률 밴드, 즉시 격상·격하만 dwell | **B 밴드 + A 비대칭 dwell**(격상 즉시, 격하는 5세션 체류 후). 최대 변경 횟수 상한은 두지 않되 12회/년 KPI 상한 + 252세션 이동 churn 경보 |
| 구간(interval) | 예측 십분위의 Wilson | A: 파라미터 밴드(최근 5회 재적합) ∨ 고정 구간 Wilson(n/20) | **A 방식.** 고정 구간 `[0,.08,.12,.16,.20,.25,.30,.40,.50,1]`, 둘 다 표시·헤드라인은 넓은 쪽 |
| 홀드아웃 보호 | 라벨 마스크 + `--force-holdout` | A: 입력 프레임 하드컷 + 1회용 unlock 파일 | **A.** `--force-holdout` 삭제. 재실행은 unlock 파일을 손으로 지우는 행위(=git 에 남음)뿐이며 장부 기재 의무 |
| 첫 재적합 | 2003 고정 | A: 규칙(≥2,000행 AND 학습창 안 완결 ≥20% 에피소드) | **A 규칙** → 2003-01-02(2,302행·2000~02 약세장 충족). 1999-01-04(1,298행, 규칙 미충족)는 민감도 |
| 검정 | 40일 블록 부트스트랩 | A: DM-HAC lag 19 + 20 위상 오프셋 부분표본 | **셋 다** 사다리 모든 단·모든 블록에 |
| HAR-RV 보조 출력 | 고정가중 HAR(특징) | A: 적합 log-HAR(OLS 4, 예산 밖) + QLIKE/MSE | **둘 다.** 고정가중은 모델 특징 `x_har` 전용, 적합 HAR 은 보조 출력 전용(확률에 절대 안 들어감) |
| 귀속 | Δlogit 분해 | A: 수준 귀속 + 재적합 항 + pp 정확 합산 | **A.** 수준(절편에서 순차 대입)·일간(할선 기울기로 pp 배분, 재적합 항 포함, 잔차 0) |
| 결정론 검사 | (없음) | C: 비트 동일 / J1: 수치 허용오차 | **허용오차 1e-9**(러너와 로컬의 BLAS 차이) |
| 입력 결측 | 장부 재개 | C: 상태 유지 + '확률 계산 불가' | **C** |
| 이벤트 지평 | 10세션 | A: 20세션(라벨 지평) + 60일 잔여 경고 | **20세션 + 60일 경고** |
| v0 종합점수 | (없음) | C: 앙상블 멤버 | **참조선만**(Platt 2개, 모델 밖, 주간 페이지) |
| §6 문자 그대로의 채택 규칙 | 모두 실패 예상 | J2/J3: 검정력 기반 수정안 | 규칙은 그대로 채점하고, 수정안은 **#2a 로 사후(post hoc) 표시**하여 소유자가 홀드아웃 해제 전에 (a)/(b) 선택 (§13) |

## 1. 디렉터리 변경

```
market-risk-lab/
  mrl/
    features.py     x_vix · x_ma · VIX 내재 확률 식 · build_features · FEATURE_SPEC/FEATURE_RULE (신규)
    vol.py          GK+OV 일변동 · HAR 성분 · 고정가중 HAR(특징) · 적합 log-HAR(보조 출력) · 자기점검 (신규)
    model.py        LogitModel · 사다리 사양 · PARAM_COUNT · 저장/적재 · 귀속 · 파라미터 밴드 (신규)
    calibrate.py    재적합 일정 · 퍼지 · 기후학 · walk_forward · 블록 · 검정 · 신뢰도 · 수용 판정 (신규)
    decision.py     3단계 상태기계(히스테리시스·비대칭 dwell) · KPI (신규)
    events.py       FOMC 표(2025~2027) · OPEX/쿼드위칭 규칙 · NFP 규칙 · CPI 표(선택) (신규)
    ledger.py       P2 열 추가 · 라이브 Brier 요약 (확장)
    report.py       render_calibration_report · render_backtest_v1 · p2_card · charts_p2 (확장)
    config.py       P2 · DECISION_P2 · BLOCKS_* · 경로 상수 (추가; v0 블록 불변)
  scripts/
    run_calibration.py    특징 → walk-forward 사다리 → 벤치마크 → 검정 → HAR-RV → results/·docs/calibration_p2.html (신규)
    run_backtest_v1.py    OOS 확률 → 결정층 → 톤 → v0 와 같은 성적표 → results/·docs/backtest_v1.html (신규)
    daily.py              v0 판정 뒤 P2 확률·상태·귀속·장부·카드 (확장; 재적합 금지)
    selftest.py           (GK+OV)/CC 비율 · 특징 NaN 꼬리 · model_p2.json 해시 · 이벤트 표 잔여 (확장)
  results/
    calib_p2_walkforward.csv · summary_p2.json · model_p2.json · backtest_v1.csv · summary_v1.json
    holdout_unlock.json (최종 검증 1회 후에만 존재) · track_record.csv (P2 열 추가)
  docs/
    calibration_p2.html · backtest_v1.html · index.html (P2 카드)
  tests/
    test_features.py · test_vol.py · test_model.py · test_calibrate.py · test_decision.py · test_events.py (신규)
    test_ledger.py · test_report.py · test_scripts_integration.py (확장)
```

## 2. 공통 규칙 추가

* **파라미터 예산(VALIDATION §6)**: 확률 모델의 적합 파라미터는 정확히 4개 — `b0`(절편), `b1`(x_vix), `b2`(x_har), `b3`(x_ma). 5번째 슬롯은 비어 있고 장부 예약(Phase 3 #3~#5). 예산 밖이지만 공개하는 적합값: 보조 출력 HAR-RV OLS 4개(확률과 결합 금지), v0 참조선 Platt 2개(주간 페이지 참조선 전용). `mrl.model.PARAM_COUNT == 4` 를 `walk_forward` 가 매 재적합마다 assert 한다.
* **고정 상수(적합 아님, 사전 등록)**: 낙폭 5%/지평 20(= `DD_TARGETS["y_dd5_20"]`), GK 식, 야간갭 항, 분산 하한 1e-8, HAR 창 (1,5,22)·로그 동일가중 1/3, MA 창 180(= `V0["ma_window"]`), 반사원리 식 상수, L2 C=1.0(sklearn 기본값), 퍼지 20, 학습 시작 1993-10-14, 첫 재적합 규칙, 블록 표, 결정층 배수·dwell, 신뢰도 구간 경계, 부트스트랩 블록 40·4,000회·seed 0, HAC lag 19.
* **홀드아웃 하드컷**: `HOLDOUT_START="2024-09-01"`. `scripts/run_calibration.py` 는 `--holdout-final` 없이는 **모든 입력 프레임**(bundle.close, spy_ohlc, cboe, targets)을 2024-08-30 에서 자른다. 라벨 마스크가 아니라 데이터 자체를 자른다.
* **점(point-in-time)**: `build_features(bundle, asof)` 는 asof 이후 행을 먼저 버리고 계산한다. 중심 창·미래 행 참조 금지. `tests/test_features.py::test_point_in_time_truncation` 이 20개 무작위 T 에서 비트 동일성을 검사한다.
* **결정론**: 난수는 전부 `seed=0`. 두 번 실행한 `walk_forward` 산출 CSV 는 동일해야 한다. 일요일 작업은 재적합 결과와 `results/model_p2.json` 의 계수를 허용오차 1e-9 로 비교하고 다르면 실패(커밋 없음).
* **재적합은 주간 작업에서만**: `daily.py` 는 `model_p2.json` 을 읽기만 한다.
* **조용한 실패 금지**: 입력 결측(그날 VIX·OHLC 없음)이면 확률·귀속은 NaN, 상태는 유지, 카드에 "확률 계산 불가", `meta.warnings` 와 장부 `p2_input_missing` 에 사유.
* **홀드아웃 관측 사실의 공개**: 세 설계는 모두 2003~2024-08 OOS 기록을 이미 보았다(수치는 §13 #2 의 '사전 관측'). 이 문서 이후의 어떤 규칙 변경도 그 기록에 대해 post hoc 이며 장부에 그렇게 적는다. 홀드아웃(2024-09-03~, 504세션·라벨 484행·기저율 12.0%)만이 오염되지 않았다.

## 3. 설정 — `mrl/config.py` (추가; v0 블록 불변)

```python
P2 = {
    "dd": 0.05, "h": 20,                                  # = DD_TARGETS["y_dd5_20"]
    "features": ("x_vix", "x_har", "x_ma"),
    "har_lookbacks": (1, 5, 22), "har_weights": (1/3, 1/3, 1/3), "var_floor": 1e-8,   # (1bp 일변동)^2
    "ma_window": V0["ma_window"],                         # 180 — v0 와 같은 창(새 상수 없음)
    "vix_ffill_limit": 3,                                 # SPY 세션에 VIX 없으면 ≤3세션 ffill, 초과 시 NaN
    "C": 1.0, "purge": 20, "train_start": "1993-10-14",
    "first_refit": "2003-01-02", "first_refit_sensitivity": "1999-01-04",
    "first_refit_rule": {"min_rows": 2000, "min_episode_depth": 0.20},
    "boot_block": 40, "n_boot": 4000, "hac_lag": 19, "phase_offsets": 20,
    "reliability_bins": (0.0, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40, 0.50, 1.0),
    "n_eff_div": 20, "param_band_refits": 5,
    "budget": 5, "param_count": 4,
    "eras": (("1993-01-29", "1999-12-31"), ("2000-01-01", "2007-12-31"), ("2008-01-01", "2012-12-31"),
             ("2013-01-01", "2019-12-31"), ("2020-01-01", "2024-08-30")),
    "har_train_start_sensitivity": "1996-01-02",
}
# 평가 블록 (달력 경계, [a, b) 반개구간; 세션은 SPY 인덱스로 결정) — §7
BLOCKS_24 = [("2003-01-01", "2005-01-01"), ("2005-01-01", "2007-01-01"), ("2007-01-01", "2009-01-01"),
             ("2009-01-01", "2011-01-01"), ("2011-01-01", "2013-01-01"), ("2013-01-01", "2015-01-01"),
             ("2015-01-01", "2017-01-01"), ("2017-01-01", "2019-01-01"), ("2019-01-01", "2021-01-01"),
             ("2021-01-01", "2023-01-01"), ("2023-01-01", HOLDOUT_START)]                      # 11개, 마지막 20개월
BLOCKS_18 = [("2003-01-01", "2004-07-01"), ("2004-07-01", "2006-01-01"), ("2006-01-01", "2007-07-01"),
             ("2007-07-01", "2009-01-01"), ("2009-01-01", "2010-07-01"), ("2010-07-01", "2012-01-01"),
             ("2012-01-01", "2013-07-01"), ("2013-07-01", "2015-01-01"), ("2015-01-01", "2016-07-01"),
             ("2016-07-01", "2018-01-01"), ("2018-01-01", "2019-09-01"), ("2019-09-01", "2021-05-01"),
             ("2021-05-01", "2023-01-01"), ("2023-01-01", HOLDOUT_START)]                      # 14개: 18개월×10 + 20개월×4
BLOCKS_24_FROM_1999 = [("1999-01-01", "2001-01-01"), ("2001-01-01", "2003-01-01")] + BLOCKS_24  # 13개(민감도)
DECISION_P2 = {"enter_caution": 1.5, "exit_caution": 1.2, "enter_reduce": 2.5, "exit_reduce": 2.0,
               "dwell": 5, "churn_alert": 12, "kpi_max_switches_per_year": 12}
DECISION_P2_SENSITIVITY = {"wide": {"enter_caution": 1.75, "exit_caution": 1.25, "enter_reduce": 3.0, "exit_reduce": 2.25},
                           "symmetric_dwell": {"dwell_escalate": 5}, "no_dwell": {"dwell": 0}}
P2_STATES = ("normal", "caution", "reduce")
STATE_TO_TONE = {"normal": "hold", "caution": "caution", "reduce": "reduce"}   # TONE_EXPOSURE 그대로 재사용
MODEL_P2_PATH = RESULTS_DIR / "model_p2.json"
HOLDOUT_UNLOCK_PATH = RESULTS_DIR / "holdout_unlock.json"
```

## 4. 특징 — `mrl/features.py`

세 특징 모두 **척도 불변 비율/로짓**이라 표준화(z-score·백분위)를 하지 않는다 — 확장창 표준화가 1996~2002 VIX 신호를 파괴한다는 A 의 실측을 받아들인 것. 백분위는 표시 전용.

| 열 | 정의 | 원천(캐시 열) | 창 | 최초 유효일 |
|---|---|---|---|---|
| `vix` | `close.csv['^VIX']` 를 SPY 세션(`spy_ohlc.csv` 인덱스)에 reindex, ffill ≤ 3세션(초과 NaN + 경고). SPY 마지막 세션 이후 유령 행(예: 2026-05-25 휴장일 행)은 `apply_guards` 가 이미 제거 | close.csv `^VIX` | 0 | 1993-01-29 |
| `p_vix` | B1 벤치마크(§6 식, 드리프트 m=-s²/2) | vix | 0 | 1993-01-29 |
| `p_vix_driftless` | 2·Φ(b/s) (A/C 표기와 대조용, 표시·열 전용) | vix | 0 | 1993-01-29 |
| `p_vix_bgk` | BGK 이산관측 보정판(§6) | vix | 0 | 1993-01-29 |
| **`x_vix`** | `ln(p_vix/(1-p_vix))` | vix | 0 | 1993-01-29 |
| `var_gkov` | v_t = GK_t + OV_t (mrl/vol.py) | spy_ohlc Open/High/Low/Close | 1(+전일 종가) | 1993-02-01 |
| `rv1`, `rv5`, `rv22` | √(252·mean(v_{t-k+1..t})), 각 평균은 `var_floor`=1e-8 로 하한 | var_gkov | 1/5/22 | 1993-02-01 / 02-05 / 03-03 |
| `har_vol_20` | σ_har = exp((ln rv1 + ln rv5 + ln rv22)/3) — 고정 동일가중(적합 아님) | rv1/5/22 | 22 | 1993-03-03 |
| **`x_har`** | `ln(σ_har) − ln(vix/100)` | har_vol_20, vix | 22 | 1993-03-03 |
| `sma180` | mean(Close_{t-179..t}), min_periods=180 (조정 종가; 비율이라 조정 불변) | spy_ohlc Close | 180 | 1993-10-14 |
| **`x_ma`** | `Close_t / sma180_t − 1` | Close, sma180 | 180 | **1993-10-14 (전체 특징의 구속 시작일)** |
| `rv20_cc` | 종가 로그수익 20일 std·√252 (targets 의 RV 규약과 동일; 표시·자기점검) | Close | 20 | 1993-02-26 |
| `ts_diag` | vix / VIX3M (표시 전용, 모델 입력 아님; 2009-09-18~) | vix, cboe.csv VIX3M | 0 | 2009-09-18 |

```python
FEATURE_SPEC: dict[str, dict]   # 열 → {source, columns, lookback, first_date, formula, label_ko, model_input: bool}
FEATURE_RULE = ("p2|x_vix=logit(refl(VIX,dd=0.05,h=20,drift=-s2/2))"
                "|x_har=mean_log(rv1,rv5,rv22;GK+OV;floor=1e-8)-ln(VIX/100)|x_ma=C/SMA180-1"
                "|vix_ffill<=3|purge=20|C=1.0|train_start=1993-10-14")
def spec_sha256() -> str            # mrl/features.py + mrl/vol.py + mrl/model.py 소스의 sha256 — 모든 산출물에 기록(WINDOW_RULE 과 같은 보호)
def align_vix(close: pd.DataFrame, spy_index: pd.DatetimeIndex, limit=P2["vix_ffill_limit"]) -> tuple[pd.Series, list[str]]
def vix_implied_prob(vix, dd=0.05, h=20, drift="martingale"|"none", monitoring="continuous"|"bgk") -> pd.Series | float
    # §6 식. vix ≤ 0 이면 ValueError. 벡터화(scipy.special.ndtr — scipy 는 scikit-learn 의존성으로 이미 설치됨)
def ma_distance(close: pd.Series, window=180) -> pd.Series
def build_features(bundle, asof=None) -> pd.DataFrame
    # 인덱스 = SPY 세션(asof 까지 자른 뒤 계산). 열: vix, p_vix, p_vix_driftless, p_vix_bgk, x_vix, var_gkov, rv1, rv5, rv22,
    #      har_vol_20, x_har, sma180, x_ma, rv20_cc, ts_diag. 각 열은 first_date 전 NaN. 반환 .attrs = {"feature_rule", "spec_sha256", "warnings"}
def factor_percentiles(feats: pd.DataFrame, window=2520) -> pd.DataFrame   # 표시 전용 10년 이동 백분위 (모델 입력 금지)
def input_status(feats_row: pd.Series) -> tuple[bool, str]                  # 세 모델 입력이 모두 유한한가, 아니면 사유 문자열
```

## 5. 변동성 — `mrl/vol.py`

```python
def garman_klass_variance(ohlc: pd.DataFrame, overnight=True, floor=1e-8) -> pd.Series
    # GK_t = 0.5·ln(H/L)^2 − (2ln2−1)·ln(C/O)^2 ; OV_t = ln(O_t/C_{t−1})^2 ; v_t = GK_t + OV_t (overnight=False 면 GK 만)
    # 야간갭은 필수: SPY 야간 분산 비중이 1990년대 ~20% → 2020년대 ~45% 로 올라 GK 단독은 종가 변동성 대비 표류한다(B 실측 GK/CC 0.85→0.60)
def parkinson_variance(ohlc, floor=1e-8) -> pd.Series          # ln(H/L)^2/(4 ln 2) — 소거실험 M3-PK 과 자기점검 전용
def har_components(var: pd.Series, lookbacks=(1, 5, 22), floor=1e-8) -> pd.DataFrame   # rv1, rv5, rv22 (연율화 %가 아니라 소수)
def har_log_vol(comp: pd.DataFrame, weights=(1/3, 1/3, 1/3)) -> pd.Series             # 고정가중 — 모델 특징 전용
# ---- 보조 출력: 적합 log-HAR (OLS 4개, 확률 예산 밖, 확률에 결합 금지) ----
def har_target(close: pd.Series, h=20) -> pd.Series           # ln(std(logret_{t+1..t+h})·√252) — targets 의 종가 RV 규약; 마지막 h 행 NaN
def har_fit(X: pd.DataFrame, y_ln: pd.Series) -> dict         # OLS ln RV20_fwd ~ 1 + ln rv1 + ln rv5 + ln rv22 → {coef(4), n, refit_date}
def har_walk_forward(comp, y_ln, refit_dates, purge=20, train_start="1993-03-03") -> tuple[pd.Series, pd.DataFrame]
    # 확률 모델과 같은 연 1회 재적합·20일 퍼지. 반환 (ln 예측 Series, 계수 표). 민감도: train_start=P2["har_train_start_sensitivity"]
def vol_scorecard(ln_fc: pd.Series, ln_rv_fwd: pd.Series, vix: pd.Series, rv22: pd.Series, blocks) -> pd.DataFrame
    # 블록별·전체: MSE(log), QLIKE = RV²/σ̂² − ln(RV²/σ̂²) − 1, R²(log), 그리고 같은 지표를 'VIX 를 예측으로'(σ̂=VIX/100)·'RV22 지속'에 대해; y_vol_20 AUC
def ratio_checks(ohlc: pd.DataFrame) -> dict
    # 자기점검: (a) rolling-250 (GK+OV)/CC 비율이 1996년 이후 [0.8, 1.3] 안 (1993~95 값은 보고만: 실측 0.87)
    #           (b) Parkinson RV20 / CC RV20 ∈ [0.3, 3] ; (c) 1993~95 Open==High|Low 비중(실측 29.9%)·중앙 로그 범위(0.62% vs 1996+ 1.09%) 보고
```

## 6. VIX 내재 벤치마크 식 (사전 등록)

기호: V = VIX_t/100, T = 20/252, s = V·√T (지평 변동성), b = ln(1−0.05) = −0.051293 (장벽), Φ = 표준정규 cdf.

* **B1 (사전 등록 벤치마크, 기반 설계의 특징 x_vix 의 원천)** — 드리프트 m = −s²/2 (r=0 위험중립 마팅게일)인 브라운 운동 로그가격의 **최저값(running minimum) 반사원리**:
  `p_vix = Φ((b − m)/s) + exp(2·m·b/s²)·Φ((b + m)/s)`.
  값: VIX 12/16/20/30/45 → 0.133 / 0.262 / 0.372 / 0.558 / 0.703.
* 무드리프트 형(A/C 표기; m=0 이면 위 식이 정확히 `2·Φ(b/s)` 로 줄어든다): VIX 12/15/20/30 → 0.129/0.225/0.363/0.544 (B1 과 <1pp 차이). 열 `p_vix_driftless` 로 병기.
* BGK 이산(일 1회 종가) 관측 보정(Broadie–Glasserman–Kou): 장벽을 `b' = b − 0.5826·V·√(1/252)` 로 옮겨 같은 식 → `p_vix_bgk`. 값: 0.102/0.211/0.307/0.475/0.613 (20종가 몬테카를로와 ±0.005). 라벨이 종가 기준이므로 물리적으로 더 맞는 형이지만, §6 이 이름 붙인 벤치마크는 B1 이고 BGK 는 '더 어려운 벤치마크'로 병기한다.
* 실측(설계 단계 공개): 1993-02~2024-08 평균 p_vix 0.318~0.33 vs 실현 0.167, Brier 0.155~0.158 vs 기후학 0.139 → skill −0.11(BGK −0.002). 즉 **보정된 것은 무엇이든 B1 을 이기므로 정직한 잣대는 M1(보정 VIX)** 이며, 사다리·검정은 M1 대비를 항상 함께 낸다.
* 기후학(참조, 멤버 아님): `clim_y` = 재적합 R_y 의 퍼지된 학습창(1993-10-14~, pos ≤ pos(R_y)−21)의 `y_dd5_20` 평균. 연중 상수. 2003~2024 재적합에서 0.150~0.197. 결정층의 r = p/clim 과 카드의 '기저율'이 이 값.

## 7. 모델 — `mrl/model.py`

`logit p_t = b0 + b1·x_vix_t + b2·x_har_t + b3·x_ma_t`, `p_t = 1/(1+exp(−logit))`.
적합: `sklearn.linear_model.LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", fit_intercept=True, max_iter=1000, tol=1e-8)`, 절편 비벌칙(lbfgs 기본), 클래스·표본 가중치 없음, 표준화 없음. C=1.0 은 조정 대상이 아니다(2,300~7,900행에서 벌칙은 무시할 크기; 민감도 C∈{0.1,1,10} 보고). 두 번째 Platt·isotonic 없음 — 로지스틱 자체가 보정 사상이며 (b0,b1)=(0,1),(b2,b3)=(0,0)이면 정확히 p_vix 를 재현(중첩).

```python
PARAM_COUNT = 4 ; BUDGET = 5
LADDER = {"M0": (), "M1": ("x_vix",), "M2": ("x_vix", "x_har"), "M3": ("x_vix", "x_har", "x_ma")}   # 채택 = M3
ABLATIONS = {"M3-PK": "GK+OV 대신 Parkinson 분산으로 x_har", "M3-HAR96": "학습 시작 1996-01-02",
             "R-v0": "v0 종합 s_t=-(score_d+score_w+score_m)/3 에 Platt(2) — 참조선, 2017-01-03 재적합부터(학습 ≥2역년·≥40양성)"}

@dataclass
class LogitModel:
    features: tuple[str, ...]; coef: dict[str, float]; intercept: float
    refit_date: str; train_start: str; train_end: str; n_train: int; n_pos: int
    clim: float; feature_rule: str; spec_sha256: str; model_id: str     # model_id = f"p2m3-{spec_sha256[:8]}-{refit_date}"
    def predict_proba(self, X: pd.DataFrame) -> pd.Series                 # 입력 NaN 행 → NaN (조용히 채우지 않음)
    def n_params(self) -> int                                             # len(coef)+1 ; M3 == 4
def fit_logit(X: pd.DataFrame, y: pd.Series, features, C=1.0, meta: dict) -> LogitModel   # y ∈ {0,1}; 유한 행만; n_pos < 20 이면 ValueError
def save_model(m: LogitModel, path=MODEL_P2_PATH) -> None ; def load_model(path=MODEL_P2_PATH) -> LogitModel
    # JSON: {schema_version:1, model_id, features, coef, intercept, clim, refit_date, train_start/end, n_train, n_pos, feature_rule,
    #        spec_sha256, deploy_mode ("info_only"|"tones"), tone_model ("M1"|"M3"|None), acceptance_ref ("summary_p2.json:acceptance"), created_at_utc}
# ---- 귀속(정확 합산) ----
def contributions(m: LogitModel, x: pd.Series) -> pd.DataFrame
    # 행: intercept, x_vix, x_har, x_ma. 열: logit(= b_k·x_k, 절편은 b0), pp(순차 대입: p_k = σ(b0+Σ_{j≤k} b_j x_j), pp_k = p_k − p_{k−1}, 순서 vix→har→ma 고정)
    # Σ logit == logit p ; σ(b0) + Σ pp == p (1e-12)
def day_over_day(m_now: LogitModel, x_now: pd.Series, m_prev: LogitModel, x_prev: pd.Series) -> dict
    # Δlogit 항: k∈{x_vix,x_har,x_ma}: b_k^now·(x_k^now − x_k^prev) ; refit: (b0^now − b0^prev) + Σ_k (b_k^now − b_k^prev)·x_k^prev (재적합 없으면 0)
    # pp 항: pp_k = m·Δlogit_k, m = (p_now − p_prev)/(logit_now − logit_prev) (할선 기울기; |Δlogit|<1e-12 면 p(1−p)) → Σ pp == Δp 정확, 순서 무관, 잔차 0
    # 반환 {"d_logit": {...}, "d_pp": {...}, "d_p": float, "refit": bool, "gap_sessions": int}
def parameter_band(models_last5: list[LogitModel], x: pd.Series) -> tuple[float, float]   # 최근 5회 연간 재적합 계수를 오늘 x 에 적용한 min/max p
```

## 8. 보정·평가 — `mrl/calibrate.py`

### 8.1 walk-forward 일정
* 학습 행: 1993-10-14 이후, `y_dd5_20` 비-NaN, 그리고 `pos(t) ≤ pos(R) − 21` (라벨 창 t+1..t+20 이 R 전에 완전히 실현 = **20거래일 퍼지**).
* 재적합일 R_y = 각 해 1월 첫 거래일(1999-01-04, 2003-01-02, 2007-01-03(1/2 포드 국장 휴장), …, 2024-01-02). R_y 의 계수가 [R_y, R_{y+1}) 을 채점.
* 첫 재적합 규칙(사전 등록): 학습 행 ≥ 2,000 **AND** 학습창 안에 저점이 확정된 ≥20% 에피소드 1개 이상 → **2003-01-02**(2,302행; 2000-03~2002-10 약세장). 1999-01-04(1,298행, 규칙 미충족)는 **민감도**로 병행 보고(2000~02 약세장을 OOS 로 채점).
* OOS 기록(주): **2003-01-02 ~ 2024-08-30, 5,453세션, 겹치지 않는 20일 창 ~272개, 기저율 15.4%**.
* 홀드아웃: 2024-09-03 ~ 마지막 완성 세션(현재 2026-09-04; 504세션, 라벨 484행, 기저율 12.0%). `--holdout-final` 1회만.
* 라이브 계수(홀드아웃 검증 전): 홀드아웃 직전까지의 퍼지된 전 자료(학습 마지막 행 = 2024-08-30 의 21세션 전 ≈ 2024-08-01)로 적합한 모델을 `model_p2.json` 에 둔다(`refit_date="2024-08-30"`). 홀드아웃 검증(#2b)이 장부에 기록된 뒤부터는 매년 1월 재적합이 전 자료를 포함한다.

### 8.2 블록 (24개월 주, 18개월 민감도; SPY 세션 실측)

| # | BLOCKS_24 | 첫/마지막 세션 | n | 양성(캐시 2026-09-04) |
|---|---|---|---|---|
| 1 | 2003-01-01~2005-01-01 | 2003-01-02 / 2004-12-31 | 504 | 27 |
| 2 | 2005~2006 | 2005-01-03 / 2006-12-29 | 503 | 6 |
| 3 | 2007~2008 | 2007-01-03 / 2008-12-31 | 504 | 197 |
| 4 | 2009~2010 | 2009-01-02 / 2010-12-31 | 504 | 116 |
| 5 | 2011~2012 | 2011-01-03 / 2012-12-31 | 502 | 95 |
| 6 | 2013~2014 | 2013-01-02 / 2014-12-31 | 504 | 26 |
| 7 | 2015~2016 | 2015-01-02 / 2016-12-30 | 504 | 49 |
| 8 | 2017~2018 | 2017-01-03 / 2018-12-31 | 502 | 78 |
| 9 | 2019~2020 | 2019-01-02 / 2020-12-31 | 505 | 75 |
| 10 | 2021~2022 | 2021-01-04 / 2022-12-30 | 503 | 128 |
| 11 | 2023-01-01~2024-09-01 (20개월) | 2023-01-03 / 2024-08-30 | 418 | 41 |

BLOCKS_18(14개): 2003-01-02/2004-06-30(376), 2004-07-01/2005-12-30(380), 2006-01-03/2007-06-29(375), 2007-07-02/2008-12-31(380), 2009-01-02/2010-06-30(376), 2010-07-01/2011-12-30(380), 2012-01-03/2013-06-28(374), 2013-07-01/2014-12-31(380), 2015-01-02/2016-06-30(377), 2016-07-01/2017-12-29(378), 2018-01-02/2019-08-30(419), 2019-09-03/2021-04-30(419), 2021-05-03/2022-12-30(421), 2023-01-03/2024-08-30(418). 모든 블록 길이 ∈ [18, 24]개월, 두 표 모두 [2003-01-01, 2024-09-01) 을 정확히 타일링(테스트). 2007년 이후 블록(#3~#11)은 캐시의 모든 자산군이 존재하는 구간이지만 이 모델은 SPY·VIX 만 쓰므로 구분이 필요 없다.

### 8.3 함수

```python
def refit_dates(idx: pd.DatetimeIndex, first="2003-01-02", end=HOLDOUT_START) -> list[pd.Timestamp]
def first_refit_ok(idx, y, episodes20: pd.DataFrame, refit_date, rule=P2["first_refit_rule"]) -> tuple[bool, dict]
def training_mask(idx, refit_date, purge=20, train_start=P2["train_start"]) -> np.ndarray[bool]
def pit_climatology(y, mask) -> float
def walk_forward(feats, y, ladder=LADDER, first_refit=..., end=HOLDOUT_START, C=1.0, purge=20) -> tuple[pd.DataFrame, list[dict]]
    # 반환 oos: 인덱스 = OOS 세션. 열: y, clim, p_vix, p_vix_driftless, p_vix_bgk, p_m1, p_m2, p_m3, refit_year, block24, block18
    #      params: 재적합별 {refit_date, rung, coef, intercept, n_train, n_pos, clim}. 매 재적합에서 assert n_params("M3") == PARAM_COUNT
def block_scores(oos, blocks, model_col="p_m3") -> pd.DataFrame
    # 블록별: n, n_blocks(=n//20), n_pos, base, mean_p, brier, bss_clim, bss_vix(B1), bss_vix_bgk, bss_m1, auc, calib_in_large(mean_p − base)
def loss_diff_ci(p_a, p_b, y, block=40, n_boot=4000, seed=0) -> dict     # d_t=(p_a−y)²−(p_b−y)² (b 가 좋으면 양수): mean, lo, hi (evaluate.block_bootstrap_ci)
def dm_test(loss_a, loss_b, lag=19) -> dict                              # Diebold–Mariano, Newey–West(Bartlett) HAC lag 19 (20일 라벨 겹침): t, p
def phase_offset_skill(p, y, ref, h=20) -> pd.DataFrame                  # 20개 오프셋(매 20번째 세션) 각각의 BSS; min/median/max/share>0
def reliability_table(p, y, bins=P2["reliability_bins"], n_eff_div=20) -> pd.DataFrame   # bin, n, n_eff=n/20, mean_p, obs, wilson_lo/hi(n_eff)
def murphy_decomposition(p, y, bins) -> dict                             # reliability, resolution, uncertainty
def era_auc(feats, oos, eras=P2["eras"]) -> pd.DataFrame                 # x_vix, x_har, −x_ma, p_m1, p_m3 의 시대별 AUC(VIX 판별력 감쇠 0.72→0.57 상시 표시)
def ladder_table(oos, blocks) -> pd.DataFrame
    # 단(M0→M1, M1→M2, M2→M3, M1→M3, clim→M3, B1→M3, BGK→M3): 전체·블록별 loss_diff_ci + dm_test + phase_offset 요약
def acceptance(block_scores24, ladder, pooled) -> dict
    # literal (VALIDATION §6 그대로): pass_clim = all(bss_clim > 0) ; pass_vix = all(bss_vix ≥ 0) ; failing_blocks
    # amended (#2a 후보, 사후 표시; §13): A) 전체 bss_clim>0 이고 loss_diff_ci(clim→M).lo>0, 모든 블록 bss_clim ≥ −0.05, 부호검정 ≥8/11 블록 >0
    #                                     B) 전체 bss_vix(B1) ≥0 이고 loss_diff_ci(B1→M).lo ≥0, ≥8/11 블록 ≥0
    #                                     C) 정보 단(M_{k−1}→M_k) 의 loss_diff_ci.lo > 0
    # deploy: rule ∈ {"literal","amended"} 에 따라 tone_model = 조건을 만족하는 최상위 단(amended: A∧B 를 만족하고 자기 단의 C 를 만족하는 가장 높은 단; literal: M3 가 통과하면 M3)
    #         없으면 deploy_mode="info_only". 반환 {rule, literal{...}, amended{...}, deploy_mode, tone_model, rationale}
def v0_reference(replay_completed: pd.DataFrame, y, refit_dates) -> pd.Series   # s_t Platt 참조선(2017-01-03~), 예산 밖
```

## 9. 결정층 — `mrl/decision.py`

상태 3개 `normal / caution / reduce` (v0 의 5톤·125칸이 churn 의 원인이었으므로 1차원 확률엔 3단계). 신호 `r_t = p_t / clim_y`(기저율 배수라 기저율 표류에도 점유율이 유지됨; clim≈0.16 이면 절대값 caution≈0.24, reduce≈0.40).

* 격상(즉시, dwell 없음): `normal→caution` r ≥ 1.5 ; 어느 상태에서든 `→reduce` r ≥ 2.5.
* 격하(현 상태 체류 ≥ 5세션일 때만): `reduce→caution` r < 2.0 ; `reduce→normal` r < 1.2 ; `caution→normal` r < 1.2.
* [1.2, 1.5) 안에서 진동해도 caution 은 그대로(히스테리시스). 상태가 바뀌면 days_in_state = 1 부터 다시.
* 최대 변경 횟수: 하드캡 없음(숨은 파라미터가 됨). 구조적 상한 = 어떤 5세션 창에서도 ≤ 3회(격하 1 + 격상 2). KPI 상한 12회/년(사전 등록), **churn 경보** = 직전 252세션 변경 > 12 → 카드에 "결정층 잦은 전환" + summary_v1.json.flags, 장부 §8 검토 대상(자동 재조정 없음).
* 입력 결측: 상태 유지, days_in_state 계속 증가, r=NaN 기록, 카드 "확률 계산 불가".
* 재적합일: 상태 이월(재적합 자체로 격하 불가; dwell 은 그대로 센다).
* 상태 → 톤: `STATE_TO_TONE` → `TONE_EXPOSURE`(100/50/25%) 그대로, 비용 5bp → v0 표와 비교 가능.
* 민감도(선택에 쓰지 않음, 보고만): wide (1.75/1.25, 3.0/2.25); symmetric_dwell; no_dwell.

```python
@dataclass(frozen=True)
class DecisionConfig: enter_caution=1.5; exit_caution=1.2; enter_reduce=2.5; exit_reduce=2.0; dwell=5; churn_alert=12
def step(r: float | None, prev_state: str, days_in_state: int, cfg=DecisionConfig()) -> tuple[str, int, str]   # (state, days, reason_ko)
def run(p: pd.Series, clim: pd.Series, cfg) -> pd.DataFrame        # 열: r, state, days_in_state, changed, reason_ko, churn_252, churn_alert
def to_tone(states: pd.Series) -> pd.Series                         # evaluate.* 가 받는 'tone' 열
def kpis(states_df, y_dd: pd.Series, spy_close) -> dict
    # switches_per_year, occupancy{normal,caution,reduce}, dd5_rate_by_state, median_warn_run, n_warn_runs,
    # true_alarm_share(경고 런 시작일, evaluate._true_alarm_share) vs 기저율 vs v0(11~13%), max_changes_any_5_sessions,
    # allocation = evaluate.allocation_sim(tone_df, spy_close) 전체, kpi_ceiling_ok(≤12/yr)
```

## 10. 이벤트 — `mrl/events.py` (표시 전용, 파라미터 0, 모델 입력 아님, 네트워크 없음)

```python
FOMC_DECISION_DAYS = {   # 2일 회의의 둘째 날(성명 14:00 ET). 출처: federalreserve.gov 회의 달력·보도자료(2024-08-09, 2025-09-05). 매년 12월 손으로 갱신
    2025: ["2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10"],
    2026: ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"],
    2027: ["2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09", "2027-07-28", "2027-09-15", "2027-10-27", "2027-12-08"],   # 잠정(직전 회의에서 확정)
}
OPEX_EXPECTED = {        # 규칙 산출과의 parity 표(테스트): 셋째 금요일, 휴장이면 직전 거래일
    2025: ["2025-01-17", "2025-02-21", "2025-03-21", "2025-04-17", "2025-05-16", "2025-06-20", "2025-07-18", "2025-08-15", "2025-09-19", "2025-10-17", "2025-11-21", "2025-12-19"],
    2026: ["2026-01-16", "2026-02-20", "2026-03-20", "2026-04-17", "2026-05-15", "2026-06-18", "2026-07-17", "2026-08-21", "2026-09-18", "2026-10-16", "2026-11-20", "2026-12-18"],
    2027: ["2027-01-15", "2027-02-19", "2027-03-19", "2027-04-16", "2027-05-21", "2027-06-17", "2027-07-16", "2027-08-20", "2027-09-17", "2027-10-15", "2027-11-19", "2027-12-17"],
}   # 2025-04-17(성금요일 4/18), 2026-06-18(준틴스 6/19), 2027-06-17(준틴스 6/19 토→금 6/18 관측 휴장)
CPI_RELEASE_DAYS: dict[int, list[str]] = {}       # 선택: BLS 표를 넣으면 표시, 비어 있으면 카드에 "CPI 일정 미등록"
NFP_OVERRIDES: dict[str, str] = {}                # "YYYY-MM" → 날짜 (BLS 예외; 기본 규칙: 그 달 첫 금요일, 연방 휴일이면 직전 목요일)
def opex_dates(year) -> list[date]                # 규칙: 셋째 금요일 → calendar_us.is_trading_day 아니면 직전 거래일
def quad_witching(year) -> list[date]             # 3·6·9·12월 OPEX
def nfp_dates(year) -> list[date] ; def fomc_dates(year) -> list[date] ; def cpi_dates(year) -> list[date]
def upcoming(asof, n_sessions=20) -> pd.DataFrame # 열: date, kind(FOMC|OPEX|QUAD|NFP|CPI), sessions_ahead, label_ko ; asof 이후 거래일만
def table_horizon(asof) -> dict                   # {last_fomc, days_left, warn: days_left < 60} — selftest 가 경고, 카드에 표시
```

## 11. 리포트 — `mrl/report.py` (추가)

```python
def p2_card(today_p2: dict) -> str                # index.html 의 Phase 2 카드 조각 (v0 판정 블록 아래, v0 는 그대로 보임)
def render_calibration_report(summary_p2: dict, out_html: Path, charts: dict[str, bytes]) -> None
def render_backtest_v1(summary_v1: dict, summary_v0: dict, out_html: Path, charts: dict[str, bytes]) -> None
def charts_p2(oos, blocks, summary_p2, spy_close) -> dict[str, bytes]     # 신뢰도 다이어그램(고정 구간, Wilson), 블록 skill 막대(4 벤치마크), 계수 경로, 사다리 loss-diff CI, 시대별 AUC, 상태 밴드+SPY, 누적수익 vs 보유 vs v0
```

**일간 카드(`docs/index.html`, 한국어, 기존 다크 팔레트)** — 위에서 아래로:
1. 헤드라인(자연빈도): `다음 20거래일 안에 -5% 하락: 이런 날 100일 중 약 N일 (기저율 100일 중 16일 · VIX 공식 100일 중 22일 · VIX 보정 100일 중 M일)`. 구간 = max(파라미터 밴드, 보정 밴드). 문장: `모델이 과거에 20~25% 라고 말했을 때 100번 중 26번 일어났습니다 (독립 사례 22개, 95% 구간 12~47%)`.
2. 사다리 오늘값: `VIX 공식(보정 전) / VIX 공식(일별 관측 보정) / VIX 보정만(M1) / +실현변동성(M2) / +추세(M3)` — "오늘 추가 요인이 더한 것: ±X pp".
3. 상태: `deploy_mode=="tones"` 이면 상태·톤(`TONE_KO`)·r·days_in_state·다음 임계(확률로 환산: "caution 해제까지 p < 19%")·churn 경보. `info_only` 이면 같은 정보를 회색으로 `시험 운용 — 비중 제안 아님`(VALIDATION §7 문구) 라벨과 함께.
4. 수준 귀속 3막대(VIX / 실현-내재 갭 / 추세; logit·pp) + 일간 변화: `어제 대비 +1.3pp: VIX +0.9 · 실현변동성 +0.6 · 추세 −0.2 (· 재적합 0.0)`, 5일 누적도.
5. 요인 문맥(표시 전용): 각 원시값 + 10년 백분위 + 한 줄 해설. VIX3M 기간구조·SKEW·VVIX·F&G 는 변동성 카드 문맥으로만.
6. 보조 A — `예상 변동성(HAR) N% · VIX N% · 최근 20일 실현 N%`, 차이 = 현재 변동성 프리미엄, OOS log-MAE.
7. 보조 B — 다음 20거래일 이벤트(FOMC/OPEX/쿼드위칭/NFP/CPI), 표 잔여 60일 미만이면 경고.
8. 정직 스트립: model_id, 계수 4개·clim·밴드, 실험 #2 §6 판정(literal/amended 각각), 홀드아웃 결과(해제 후), Phase 3 킬룰 카운트다운 `실현 ≥5% 에피소드 X/8 · 경과 Y/36개월`, 장부 기준 라이브 Brier(vs 기후학·vs M1·vs B1), 각주 `AUC 0.63~0.70: 위험 확률은 예측 가능하지만 방향은 아니다 · VIX 대비 skill 의 대부분은 편향 보정 · 확률은 매일 참고, 배분은 상태 기계로만`.

**주간 `docs/calibration_p2.html`**: ① v0 completed 요약 줄(영구) ② 사다리 표(Brier, BSS_clim, BSS_vix(B1), BSS_bgk, BSS_m1, AUC, loss-diff CI, DM t, 위상 오프셋 min/med/max) ③ 블록 표 24개월(+18개월, +1999 시작) 과 §6 literal/amended 판정 ④ 신뢰도(고정 구간·Wilson)·Murphy ⑤ 계수 경로·파라미터 밴드 ⑥ 시대별 AUC ⑦ 소거(M3-PK, M3-HAR96, C 민감도)·v0 참조선 ⑧ HAR-RV 성적표 ⑨ 데이터 범위·자기점검(GK+OV/CC, 1993~95 품질)·경고·spec_sha256·FEATURE_RULE.
**주간 `docs/backtest_v1.html`**: v0 줄 먼저 → 결정층 KPI(전환/년, 점유, 상태별 dd5 비율, 진짜 경보 비중 vs 기저율 vs v0, 최대 5세션 변경) → `evaluate.episode_eval`·`allocation_sim`(v1 vs v0 vs 보유) → 민감도 3종 → 상태 밴드 차트.

## 12. 장부 — `mrl/ledger.py` (확장; 기존 열 순서 불변, `schema_version=2`)

`P2_COLUMNS` 를 `LEDGER_COLUMNS` 뒤에 추가(구 행은 NaN): `prob_dd5_20`(기존 예약 열, = p_m3), `p2_p_m1`, `p2_p_m2`, `p2_p_vix`, `p2_p_vix_bgk`, `p2_clim`, `p2_lo`, `p2_hi`, `p2_band_src`(param|calib), `p2_x_vix`, `p2_x_har`, `p2_x_ma`, `p2_har_vol_20`, `p2_har_fc_20`, `p2_r`, `p2_state`, `p2_days_in_state`, `p2_tone_model`, `p2_deploy_mode`, `p2_d_vix`, `p2_d_har`, `p2_d_ma`, `p2_d_refit`(pp), `p2_input_missing`(str|''), `p2_model_id`.
`append_today` 는 이 열을 받아 정규화(숫자 NaN 허용, 문자열은 object). `backfill` 은 그대로(`y_dd5_20` 이 채워지면 라이브 Brier 가 자동 누적). `summary()` 추가 키: `p2: {n_scored, brier, brier_clim, brier_m1, brier_vix, bss_clim, bss_m1, bss_vix, ci_bss_clim(블록 40 부트스트랩), episodes5_observed, months_elapsed, kill_rule_due}`.

## 13. 스크립트

* `scripts/run_calibration.py [--end (기본: HOLDOUT_START 직전 세션)] [--first-refit 2003-01-02|1999-01-04] [--blocks 24|18] [--har-train-start 1993-03-03|1996-01-02] [--holdout-final] [--results-dir] [--docs-dir]`
  흐름: `load_cache → apply_guards → 완성 봉 자르기 → 하드컷(--end) → build_features → make_targets → episodes(0.20) → first_refit_ok(assert) → walk_forward(사다리+소거) → block_scores(24·18·1999) → ladder_table → reliability/murphy/era_auc → acceptance → har_walk_forward+vol_scorecard → v0_reference(results/backtest_v0_completed.csv 의 run.window_rule·signals_v0_sha256 이 현재 코드와 같을 때만; 아니면 참조선 생략+경고)`.
  산출: `results/calib_p2_walkforward.csv`(date, y, clim, p_vix, p_vix_driftless, p_vix_bgk, p_m1, p_m2, p_m3, p_m3_pk, p_m3_har96, p_v0ref, har_fc_20, refit_year, block24, block18), `results/summary_p2.json`(run{generated_at_utc, end, first_refit, spec_sha256, feature_rule, python/pandas/numpy/sklearn, cache}, v0_reference, ladder, blocks24/18/from1999, reliability, murphy, era_auc, acceptance, params_by_refit, har, selftest, warnings, disclosure="#2 사전 관측 참조"), `results/model_p2.json`(라이브 계수 + deploy_mode/tone_model = acceptance 결과), `docs/calibration_p2.html`.
  `--holdout-final`: `HOLDOUT_UNLOCK_PATH` 가 존재하면 즉시 비정상 종료(exit 2). 없으면 하드컷 없이 2024-09-03~ 를 R_2024·R_2025·R_2026 재적합(각각 자기 재적합일 전 퍼지 자료로만 학습)으로 채점해 `summary_p2.json.holdout` 에 기록하고 `holdout_unlock.json{timestamp_utc, git_sha, spec_sha256, ledger_entry:"2b", n, n_pos, bss_clim, bss_vix, bss_m1, ci}` 를 쓴다. `--force` 류 옵션은 없다.
  주간 결정론 assert: `--end` 기본 실행에서 새로 적합한 라이브 계수와 기존 `model_p2.json` 의 계수 차가 1e-9 를 넘으면(그리고 spec_sha256 이 같으면) exit 1. spec_sha256 이 다르면(코드 변경) 새 파일을 쓰되 `run.spec_changed=true` 와 장부 기재 요구를 로그.
* `scripts/run_backtest_v1.py [--config default|wide|symmetric_dwell|no_dwell] [--all-configs]`
  `results/calib_p2_walkforward.csv` 와 `summary_p2.json` 을 읽어(spec_sha256·feature_rule 일치 확인, 아니면 exit 1) `decision.run(p_m3, clim)`(tone_model 이 M1 이면 p_m1 도 병행) → `to_tone` → `evaluate.episode_eval / allocation_sim / directional_scorecard`(방향은 참고만) → `decision.kpis` → `results/backtest_v1.csv`(date, p, clim, r, state, days_in_state, tone, changed) + `results/summary_v1.json`(v0_reference 줄, kpis, allocation, episodes, sensitivities, flags) + `docs/backtest_v1.html`.
* `scripts/daily.py` (v0 단계 뒤에 추가; 재적합 금지)
  ```
  feats = build_features(bundle, asof)                      # 완성 봉 번들, asof 로 자름
  m = load_model(MODEL_P2_PATH); assert m.spec_sha256 == spec_sha256() (다르면 exit 1: 코드가 모델보다 새롭다)
  ok, why = input_status(feats.loc[asof]) ; p = m.predict_proba(...) if ok else NaN
  prev = 장부의 마지막 유효 P2 행 (없으면 state="normal", days=0)
  state, days, reason = decision.step(p/m.clim if ok else None, prev.p2_state, prev.p2_days_in_state)
  band = max(parameter_band(최근 5회 계수: summary_p2.json.params_by_refit), reliability bin Wilson(summary_p2.json.reliability))
  dod = day_over_day(m, x_now, m_prev(= prev.p2_model_id 의 계수; 같은 모델이면 refit 항 0), x_prev)
  har_fc = summary_p2.json.har.live_coef 로 계산 ; events = upcoming(asof, 20)
  ledger.append_today(row | P2 열) ; render_index(today | {"p2": {...}}, ls, out_html)
  ```
  실패 조건(exit 1): model_p2.json 없음 · spec 불일치 · 1월 첫 주간 실행 이후 refit_year ≠ 올해(홀드아웃 해제 후 규칙). 입력 결측은 실패가 아니라 "확률 계산 불가" 경로.
* `scripts/selftest.py` 추가: `vol.ratio_checks` 경계, 특징 NaN 꼬리(마지막 세션에 x_* 결측이면 경고), `model_p2.json.spec_sha256 == spec_sha256()`, `events.table_horizon(asof).warn`, 장부 P2 열 결측일.

## 14. 워크플로

* `.github/workflows/weekly.yml` — `run_backtest_v0.py` 단계 뒤에 추가:
  ```yaml
      - name: scripts/run_calibration.py (특징 → walk-forward 사다리 → 검정 → HAR-RV → results/ + docs/calibration_p2.html; 라이브 계수 결정론 검사)
        run: python scripts/run_calibration.py
      - name: scripts/run_backtest_v1.py (결정층 → 톤 → v0 와 같은 성적표 → results/ + docs/backtest_v1.html)
        run: python scripts/run_backtest_v1.py --all-configs
  ```
  실패 시 커밋 단계에 이르지 못한다(조용한 실패 금지). `timeout-minutes: 60` 유지. `--holdout-final` 은 워크플로에 넣지 않는다(로컬 1회 수동, 커밋 메시지 `holdout-final: 2b`).
* `daily.yml` — 변경 없음(`daily.py` 확장; 추가 시간 < 3초).
* `requirements.txt` — 변경 없음(scipy 는 scikit-learn 의존성으로 이미 설치; hmmlearn·statsmodels 불필요).
* 계산 예산(ubuntu 러너): 특징 < 2초 · 22회×(4단+2소거+참조선) 로지스틱 < 20초 · 부트스트랩 4,000회 × ~60개 비교 < 40초 · HAR OLS < 2초 · 차트 < 20초 → `run_calibration.py` < 2분, `run_backtest_v1.py` < 30초. 일간 추가 < 3초(10분 한도 무관).

## 15. 테스트 (필수)

* `tests/test_features.py` — 최초 유효일(x_vix 1993-01-29, x_har 1993-03-03, x_ma 1993-10-14) 실캐시 확인 · **점 원칙**: 무작위 T 20개에서 `build_features(bundle_cut_at_T)` 와 전체 계산의 [:T] 가 NaN 포함 비트 동일 · VIX 정렬(유령 행 제거, ≤3 ffill+경고, 초과 NaN) · `vix_implied_prob`: VIX 12/16/20/30/45 → 0.133/0.262/0.372/0.558/0.703(±0.001), drift="none" 에서 20 → 0.363, VIX 단조, bgk < martingale < driftless(장벽 관계) · (slow) 200k 경로·20일 일별 GBM 몬테카를로(seed 0)의 20종가 최저값 확률이 BGK 와 ±0.005(VIX 12/20/30) · x_vix 중첩: σ(x_vix)==p_vix · 백분위는 build_features 열에 없음(모델 입력 금지).
* `tests/test_vol.py` — GK·OV 손계산 대조(합성 OHLC) · 하한 적용 · 실캐시 rolling-250 (GK+OV)/CC ∈ [0.8,1.3](1996~) · Parkinson/CC RV20 ∈ [0.3,3] · 고정가중 HAR 동일가중 · `har_fit` 합성 회복(계수 4개, 오차 < 0.05) · 미래 행 추가 시 t 까지의 rv/har 불변.
* `tests/test_model.py` — `PARAM_COUNT==4`, `n_params("M3")==4` · 합성 로지스틱(n=20,000) 계수 회복 오차 < 0.05 · coef(0,1,0,0)·intercept 0 → predict == p_vix · save/load 왕복 · `contributions` 합 == logit, σ(b0)+Σpp == p (1e-12) · `day_over_day` 합 == Δp (재적합 있는 날 포함; Δlogit≈0 경계) · 파라미터 밴드가 현재 p 를 포함.
* `tests/test_calibrate.py` — 재적합일 = 1월 첫 거래일(1999-01-04, 2003-01-02, 2007-01-03, 2024-01-02) · **퍼지**: 모든 재적합에서 max(학습 pos)+20 < pos(R) · 홀드아웃 행이 어떤 학습 마스크에도 없음 · 기후학 == 퍼지 학습 라벨 평균 · `first_refit_ok`: 2003-01-02 True(2,302행), 1999-01-04 False(1,298행) · BLOCKS_24/18 이 [2003-01-01, HOLDOUT_START) 를 정확히 타일링, 길이 ∈ [18,24]개월, 세션 수 = §8.2 표 · **결정론**: `walk_forward` 두 번 → CSV 해시 동일 · **walk-forward vs 오프라인 parity**: OOS 무작위 10일 t 에 대해 `oos.p_m3[t]` == 그 해 재적합 계수로 `build_features(bundle_cut_at_t).loc[t]` 를 예측한 값(1e-12) · **누수 카나리(합성)**: y_t = 1[x_{t+1} > 0] 인 자료에서 퍼지 20 walk-forward 의 OOS BSS ∈ [−0.05, 0.05]; 같은 파이프라인에 purge=0 이고 특징이 라벨 창을 읽도록 고의로 망가뜨린 변형은 BSS > 0.5(테스트가 누수를 실제로 탐지함을 증명) · **지연 특징(실캐시)**: 특징을 20/250세션 지연시켜 걸린 skill 이 원본보다 엄격히 낮고, 250세션 지연에서는 원본의 25% 미만 · `acceptance`: 합성 블록 표에서 literal/amended 분기·tone_model 선택 · 사다리 M0 == p_vix · `loss_diff_ci` 부호(합성) · `dm_test` HAC 분산 > 0 · `phase_offset_skill` 20행 · `reliability_table` Wilson 이 n/20 기준.
* `tests/test_decision.py` — 밴드 안 진동 시 무변경 · 격상 즉시 · 격하는 5세션 체류 후 · reduce→normal 경로 · 재개(장부 상태에서 이어서 계산 == 처음부터 계산) · 입력 결측 시 상태 유지·days 증가 · churn 경보 · 적대적 p 경로에서 5세션 창 최대 3회 · `kpis` 키.
* `tests/test_events.py` — `opex_dates` 2025~2027 == `OPEX_EXPECTED`(36개; 2025-04-17·2026-06-18·2027-06-17 이동 포함) · 쿼드위칭 12개 · FOMC 표 24개 전부 거래일·수요일 · NFP 규칙·override · `upcoming` 은 asof 이후 거래일만 · `table_horizon` 경고 경계(60일).
* `tests/test_ledger.py` 확장 — P2 열 왕복·구 행 NaN 보존·`summary().p2` 라이브 Brier.
* `tests/test_report.py` — p2_card 가 `BAD_TOKEN` 없이 렌더, info_only 에서 톤 문구 숨김·시험 운용 라벨 노출, calibration/backtest_v1 리포트 섹션 ①~⑨ 존재.
* `tests/test_scripts_integration.py` 확장 — `run_calibration.py --end 2006-12-29 --first-refit 2003-01-02`(tmp results/docs) 가 모든 산출물을 쓰고 summary 가 엄격 JSON·키 계약을 만족 · `--holdout-final` 이 unlock 파일 존재 시 exit 2 · `run_backtest_v1.py` 가 그 산출물로 완주 · `daily.py --no-update` 가 P2 장부 열과 카드를 씀.

## 16. 정직 문구와 리스크 (리포트 ⑨·카드 정직 스트립에 상시)

1. §6 문자 그대로의 규칙은 설계 단계 관측에서 **모든 후보가 실패**(M3: 기후학 대비 2019-20 −0.038, 2023-24 −0.000; B1 대비 2007-08 −0.082, 2017-18 −0.029; 보정 VIX 단독도 2019-20 실패). 블록당 ~25개 독립 창에서 BSS 의 표준편차는 ~0.1 이라 '모든 블록 > 0' 은 참 skill +0.05~0.09 에 대해 검정력이 낮다 — 그래도 규칙은 그대로 채점하고, 완화는 #2a 로 사후 표시한다.
2. VIX 대비 skill 의 대부분은 편향 보정(M1 +0.19 of M3 +0.20); HAR·MA 의 정보 이득 +13.6×1e-4 (CI −2.8..31.4) — 사다리를 숨기지 않는다. 가족용 문구는 "보정된 VIX 에 조금 더".
3. 구조적 실패 모드: 저변동 상태에서의 급락(2018-02, 2020-02)은 과소, 급락 뒤 반등기(2020-04~05, 2003)는 과대. 자연빈도+기저율 표시가 완화책이지 해결책이 아니다.
4. 홀드아웃(~24 독립 창, 2025-04 관세 급락 포함)은 큰 실패만 기각할 수 있고 +1~2% 우위를 인증할 수 없다. 진짜 시험은 Phase 3 라이브 장부(킬룰 §7).
5. 데이터: 1993~95 SPY OHLC 는 시가==고/저가 30%·범위 절반(자기점검·HAR96 민감도); Yahoo 조정 종가 재다운로드는 O/H/L/C 비율·x_ma 를 바꾸지 않지만 어제 저장값과 오늘 재계산값의 |Δp| > 0.01 이면 경고; ^VIX 휴장일 유령 행.
6. 겹치는 라벨: 모든 구간·검정·신뢰도 표본 수는 블록 방법 또는 n/20 — 5,453 을 '관측치'로 읽는 검토자를 막기 위해 표마다 n_blocks 병기.
7. 'reduce' 는 VIX>50 국면 밖에서 거의 켜지지 않는다 → 사실상 2단계임을 표시.
8. 설계 단계에서 결정층 구성 4종을 시뮬레이션했다 — 채택 구성은 지금 동결하고 같은 자료로 재조정하지 않는다.

## 17. VALIDATION.md 변경 (§0 정정 + §8 장부 항목)

§0 12번째 줄: `20일 내 -5% 낙폭 AUC 0.76` 을 취소선으로 남기고 각주: "0.76(기저율 25.1%)은 창 안 고점→저점 정의에서만 재현; 구현된 `y_dd5_20` 에서는 0.700(홀드아웃 제외)/0.633(2013+)/0.624(2015+), 20일 RV 0.689 — 실험 #1c". §6 에 각주: "실험 #2 채점은 문자 그대로; 완화안은 #2a(사후)".

| # | 날짜 | 변경 | 사유 | 결과 | 채택 |
|---|---|---|---|---|---|
| 1c | 2026-09-07 | 목표변수 정의 대조: 감사의 VIX AUC 0.76/0.69·기저율 ~25% 는 창 [t,t+20] 안 고점→저점 -5% 정의(0.760, 기저율 25.1%)에서만 재현; `mrl/targets.py` 의 `y_dd5_20`(t 종가 대비)에서는 0.700(1993~2024-08, n=7,954)/0.633(2013+)/0.624(2015+), 20일 RV20 0.689. §0 문구 정정 | 세 설계·세 심사가 독립 재현. 목표변수는 바꾸지 않음(사후 변경 금지) — 문제는 감사보다 어렵다 | 문서 정정 | 보고 방식만 |
| 2 | 2026-09-07 | Phase 2 사양(이 문서): 특징 x_vix(B1 로짓)·x_har(GK+OV 고정가중 HAR − ln VIX)·x_ma(Close/SMA180−1); 4-파라미터 L2 로지스틱(C=1.0), 사다리 M0~M3; 학습 1993-10-14~, 1월 첫 거래일 재적합, 20일 퍼지, 첫 재적합 규칙(≥2,000행∧완결 ≥20% 에피소드 → 2003-01-02; 1999 민감도); BLOCKS_24 11개(마지막 20개월)·BLOCKS_18 14개; 벤치마크 B1(드리프트 반사원리)·B1-BGK·B2=M1·기후학(재적합 학습창 평균); 검정 = 40일 블록 부트스트랩 4,000회 + DM-HAC lag 19 + 20 위상 오프셋; 고정 구간 신뢰도·Wilson(n/20); 결정층 r=p/clim (1.5/1.2, 2.5/2.0), 격상 즉시·격하 5세션, 12회/년 KPI·252세션 churn 경보; 소거 M3-PK·M3-HAR96·C∈{0.1,10}; 참조선 v0 Platt(2017~); 보조 출력 HAR OLS(4, 예산 밖)·이벤트 표·귀속; 홀드아웃 하드컷 + unlock 1회 | §6 예산(4/5)·규칙 이행. **사전 관측 공개**: 설계 단계에서 2003~2024-08 OOS 를 이미 봄 — M3 Brier 0.1205, BSS_clim +0.087, BSS_vix +0.202, BSS_bgk +0.102, 24개월 블록 기후학 대비 9/11 >0 (2019-20 −0.038, 2023-24 −0.000), B1 대비 9/11 ≥0 (2007-08 −0.082, 2017-18 −0.029), M1→M3 +13.6×1e-4 (CI −2.8..31.4); 결정층 2003~24: 4.4회/년, 경고 점유 21.7%, 경고 dd5 31.2% vs 정상 11.0% (v0: 83회/년, 46.5%, 18~19% vs 11~12%). 따라서 이 항목 이후 어떤 규칙 변경도 post hoc | (run_calibration.py / run_backtest_v1.py 산출로 채움: literal 판정, 사다리, 블록 표, KPI) | (채움) — literal 실패 시 사전 등록 결과 = 정보 제공 전용, v0 톤 유지 |
| 2a | (홀드아웃 해제 전, 소유자 결정) | §6 완화안 — **사후(post hoc)**: A) 전체 BSS_clim>0 이고 loss-diff 95% 하한>0, 모든 블록 ≥−0.05, ≥8/11 블록 >0; B) B1 대비 전체 ≥0·하한 ≥0·≥8/11 블록 ≥0; C) 정보 단의 하한>0. 톤 모델 = A∧B 와 자기 단의 C 를 만족하는 최상위 단(예상: M1; M3 는 정보 표시), 없으면 정보 제공 전용. 경로 (a) 문자 그대로 → 정보 제공 전용 / (b) 이 완화안 | 블록당 ~25창의 검정력(BSS sd ~0.1). #2 의 사전 관측을 본 뒤의 결정이므로 post hoc 표시 필수 | (소유자 기재) | (소유자 기재) |
| 2b | (1회) | 홀드아웃 최종 검증 `run_calibration.py --holdout-final`: 2024-09-03~ 를 R_2024/2025/2026 계수로 채점, BSS_clim·BSS_vix·BSS_m1 + 40일 블록 CI, `results/holdout_unlock.json` 생성(재실행 거부) | §6 홀드아웃 1회 규칙. #2/#2a 의 판정을 바꿀 수 없고 Phase 3 킬룰 감시의 첫 행이 된다 | (채움) | — |
| 3 | (Phase 3 예약) | 5번째 슬롯 후보 M4a: 2-상태 가우시안 HMM 필터 P(고변동) on [r_t, ln RV10], θ 1993~2014 동결(적합 파라미터 0), 사다리 단 M3→M4a 로만 채점 | C 실측 AUC 0.650(2015+) > VIX 0.624, θ 동결 시 AUC 변화 <0.001, VIX 판별력 감쇠 보완 | 미실행 | 미정 |
| 4 | (Phase 3 예약) | 후보 M4b: 구리/금 63일 로그 모멘텀 −Δln(HG/GC) (2000-11-29~, 가용성 대체 없이 자기 구간만) | A 실측 +2.0% walk-forward skill(2005+), 그러나 ~20개 후보 중 선택 → 승자의 저주 할인 | 미실행 | 미정 |
| 5 | (Phase 3 예약) | 후보 M4c: SOX/^GSPC 63일 상대수익(1994-08-03~; 라벨 '주도'이지 '폭' 아님) | A 실측 +0.4%(1999+) | 미실행 | 미정 |
| — | 기록 | v0 종합점수 s_t 는 OOS 정보 없음(C 실측 AUC 0.526, skill −0.027; 2026년 선정 워치리스트의 생존편향이 입력에 스며듦) → 모델 입력 후보에서 영구 제외, 참조선으로만 | | | |

출처(이벤트 표): [FOMC 회의 달력](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) · [2025·2026 잠정 일정 보도자료](https://www.federalreserve.gov/newsevents/pressreleases/monetary20240809a.htm) · [2027 잠정 일정 보도자료](https://www.federalreserve.gov/newsevents/pressreleases/monetary20250905a.htm)