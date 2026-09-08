# market-risk-lab — 아키텍처 및 모듈 계약 (Phase 1)

목적: 기존 `market-brief`(v0)를 **한 줄도 건드리지 않고**, 별도 플랫폼에서
(1) 30년 데이터 캐시, (2) 완성 봉 기반 v0 신호의 충실한 재현, (3) 사전 등록된 목표변수·에피소드 표,
(4) v0 규칙의 정직한 백테스트 리포트, (5) 매일 판정 기록 장부를 구축한다.
Phase 2(보정 모델)·Phase 3(국면 모델·앙상블·비중)은 이 위에 쌓는다.

기존 시스템 참조본: `reference/market_dashboard_v0.py` (읽기 전용, 수정 금지).

## 디렉터리

```
market-risk-lab/
  mrl/            패키지 (아래 모듈 계약)
  scripts/        build_cache.py · run_backtest_v0.py · daily.py
  tests/          pytest (parity 테스트 필수)
  data/           커밋되는 캐시 (CSV) — 전량 재다운로드 방식, 증분 append 금지(자동조정 재계산 때문)
                  예외: fg_history.csv, spy_eod.csv, cboe.csv 는 append (원천이 과거를 다시 주지 않음)
  results/        백테스트 산출물 (CSV/JSON), track_record.csv
  docs/           GitHub Pages 산출 HTML (index.html, backtest_v0.html)
  reference/      v0 원본 사본 (읽기 전용)
  VALIDATION.md   사전 등록: 목표변수·평가·수용/킬 규칙·실험 장부
```

## 공통 규칙

* Python 3.12, pandas 2.2, numpy 2.1, yfinance 1.2.1 (requirements.txt 고정).
* 모든 날짜 인덱스는 tz-naive `DatetimeIndex`(거래일). yfinance 반환값은 `tz_localize(None)` 처리.
* **점(point-in-time) 원칙**: 날짜 t의 계산은 t 이전(또는 t 종가까지)의 데이터만 사용. 미래 행을 슬라이스로 걸러내는 것을 각 함수가 책임진다.
* **완성 봉 원칙**: 일봉은 장 마감(16:00 ET) 후에만 "완성". 주봉/월봉은 해당 기간의 마지막 거래일 종가 이후에만 완성. `variant="faithful"`은 v0의 부분 봉 채점을 그대로 재현하고, `variant="completed"`는 완성 봉만 쓴다.
* 실패 시 조용히 넘어가지 않는다: 예외를 던지거나 `warnings.warn` + 반환값의 `meta` 필드에 기록.
* 한국어 주석·문구, 영어 식별자.

## 데이터 계약 — `mrl/data.py`

```python
@dataclass
class Bundle:
    close: pd.DataFrame          # 조정 종가. 열 = ALL_TICKERS 중 존재하는 것. 인덱스 = 거래일(BTC 주말 포함 시 union)
    spy_ohlc: pd.DataFrame       # SPY Open/High/Low/Close/Volume (조정)
    cboe: pd.DataFrame           # 열 VIX3M, VIX9D, VVIX, SKEW (일자 인덱스)
    fg: pd.DataFrame             # 열 score + CNN 구성요소(있는 대로). 2020-08-03~. 인덱스 = 날짜
    eod: pd.DataFrame            # SPY 마지막 1시간봉(15:30~16:00 ET): 열 open, close, ret (ret = close/open-1)
    meta: dict                   # fetched_at_utc, first/last per ticker, yfinance_version, warnings[list]

def fetch_prices(tickers: list[str], period="max") -> pd.DataFrame        # yf.download 일괄, auto_adjust=True, threads=True, progress=False
def fetch_cboe() -> pd.DataFrame
def fetch_fg_history(seed_date=CNN_FG_SEED_DATE) -> pd.DataFrame          # graphdata/{date} 파싱, 점수+구성요소
def fetch_fg_today() -> dict                                              # graphdata (날짜 없음) → 오늘 점수·구성요소
def fetch_spy_eod(period="730d") -> pd.DataFrame                          # interval="1h", 세션별 마지막 봉(15:30). 반일장(마지막 봉 11:30)은 half_day=True 열로 표시
def build_cache(data_dir=DATA_DIR) -> Bundle                              # 전량 재구축(가격) + append(fg/eod/cboe) + meta.json + 저장
def update_daily(data_dir=DATA_DIR) -> Bundle                             # 가격 전량 재다운로드(4초), fg/eod/cboe 는 새 날짜만 append
def load_cache(data_dir=DATA_DIR) -> Bundle
def apply_guards(bundle: Bundle) -> Bundle                                # (a) SPY 마지막 일자 이후의 유령 행 제거(24/7 자산 제외) (b) 마지막 일자가 SPY보다 3거래일 이상 뒤처진 티커는 meta.warnings 에 기록
```

파일: `data/close.csv`, `data/spy_ohlc.csv`, `data/cboe.csv`, `data/fg_history.csv`, `data/spy_eod.csv`, `data/meta.json`.
CSV는 `index_label="date"`, 소수 6자리.

## 거래 캘린더 — `mrl/calendar_us.py`

```python
def nyse_holidays(start_year: int, end_year: int) -> set[date]     # NYSE 규칙 기반 (신정·MLK(1998+)·대통령의날·성금요일·현충일·준틴스(2022+)·독립기념일·노동절·추수감사절·성탄절, 관측일 규칙) + 특별휴장(2001-09-11~14, 2012-10-29~30, 2004-06-11, 2007-01-02, 2018-12-05, 2025-01-09)
def is_trading_day(d: date) -> bool
def next_trading_day(d: date) -> date
def is_period_end(d: date, freq: "W"|"M") -> bool                   # d 가 그 주(금요일 마감)/그 달의 마지막 거래일인가
def session_complete(now_et: datetime, bar_date: date) -> bool      # bar_date 의 일봉이 확정됐는가 (now_et > bar_date 16:05 ET)
def completed_daily(df: pd.DataFrame, now_et: datetime|None) -> pd.DataFrame   # 미완성 마지막 일봉 제거
def resample_close(s: pd.Series, freq: "W"|"M", completed_only: bool) -> pd.Series  # v0 방식('W-FRI'/'ME' last) 재현; completed_only=True 면 마지막 부분 기간 제거(마지막 관측일이 is_period_end 가 아니면 제거)
```
백테스트에서는 SPY 인덱스가 실제 거래일 진실이므로 `is_period_end` 판정 시 인덱스의 다음 관측일도 교차 확인하는 헬퍼 `period_end_from_index(idx)` 를 제공한다(살아있는 날짜엔 규칙 기반).

## v0 신호 포팅 — `mrl/signals_v0.py`

reference 의 `resample_close / macd_phase / ret / build_metrics / assess / composite / overall / combo_advice` 를 **그대로** 옮기되,
데이터 입력을 Bundle + `asof` 로 바꾸고 창(window)을 명시적으로 에뮬레이션한다.

```python
def window(bundle, asof, variant) -> dict      # v0 라이브가 보는 데이터 재현:
    # spy/vix/fang: (asof-2y, asof] 달력 창 — Yahoo period="2y" 의 실제 규칙(마지막 봉 기준, 시작 경계 배타), 500~507거래일
    #               (504 로 고정하면 ~0.5% 의 날에 P1·월간 MACD 상태가 뒤집힌다; tests/fixtures/fetch_real_*.json 으로 검증)
    # watch: (asof-1y, asof] (≈252거래일) · btc: [마지막 BTC 봉-2y, 마지막 BTC 봉] 포함 경계(주말 포함, 731행)
    # variant="faithful": asof 당일 행 포함(부분 봉 그대로), 주/월 부분 기간 그대로
    # variant="completed": 일봉은 asof 까지 완성분만(백테스트에선 asof 종가는 완성으로 간주), 주/월은 completed_only
    # 반환 dict 키: spy(DataFrame OHLC), vix(Series), btc(Series), fang(DataFrame), watch(DataFrame), eod(Series[bool]), eod_vals, fg(dict|None)
def v0_metrics(win: dict, tf: str, cut: int = 0, basket: "v0"|"equal") -> dict   # reference build_metrics 와 동일 키
def v0_assess(m: dict) -> dict[str, state]
def v0_composite(states: dict, fg_missing: bool, eod_missing: bool=False, lead_missing: bool=False) -> float
    # v0: fg_missing 이면 fg 가중치 제외. 재현에서 데이터가 없는 신호(P3 2020-08 이전, P6 2023-10 이전)도 같은 방식으로 제외 — 이것이 VALIDATION.md 의 '대체 규약'
def v0_overall(score: float, trend: int) -> state
def v0_combo(mo, wk, dy) -> tuple[name_ko, action_ko, tone]
def v0_day(bundle, asof, variant="faithful", basket="v0") -> dict
    # 하루 전체: 3개 타임프레임 × (now, prev=cut) → states, composite, trend, overall, combo. reference build_payload 의 계산 부분과 동일 흐름
    # 반환 키: asof, states_d/w/m(dict), score_d/w/m, trend_d/w/m, overall_d/w/m, tone, verdict_ko, n_watch_avail, fg_avail(bool), eod_avail(bool), leaders(list)
```
* P3: `bundle.fg` 에 asof 가 있으면 사용(`d_daily`=전일 대비, `d_weekly`=5거래일 전 대비, `d_monthly`=21거래일 전 대비 — CNN 라이브 값의 정의와 가장 가까운 근사), 없으면 None → 제외.
* P6: `bundle.eod` 의 asof 이전 30거래일로 streak/eod20 계산(라이브와 동일 규칙 `ret > eod_ret`), 반일장 제외, asof 미포함 시 제외.
* P7: 워치리스트 중 asof 시점에 65개 이상 봉이 있는 종목만(라이브의 `len(s)<65` 스킵과 동일). `n_watch_avail` 기록.
* `basket="v0"`는 FANG 바스켓을 창 첫날 정규화 평균(라이브 동일), `"equal"`은 일별 동일가중 리밸런스 수익률 누적(Phase 2 실험용).

## 재현 — `mrl/replay.py`

```python
def replay_v0(bundle, start=BACKTEST_START, end=None, variant="faithful", basket="v0", progress=True) -> pd.DataFrame
    # 인덱스 = 거래일. 열: state_fang..state_btc(일간 상태), score_d/w/m, trend_d/w/m, overall_d/w/m, tone, verdict_ko,
    #        n_watch_avail, fg_avail, eod_avail, n_leaders. 성능 목표: 2015~2026 (~2,950일) 10분 이내(로컬), 가능한 부분은 벡터화.
def cell_table(replay: pd.DataFrame) -> pd.DataFrame    # (overall_m, overall_w, overall_d) 125칸 점유표: n일, 톤, 규칙 번호
```

## 목표변수·에피소드 — `mrl/targets.py`

```python
def make_targets(spy_close: pd.Series) -> pd.DataFrame
    # y_sign_5/20/60: close[t+h]/close[t]-1 > 0
    # y_dd5_20: min(close[t+1..t+20])/close[t]-1 <= -0.05 ; y_dd10_60 동일(0.10, 60)
    # y_vol_20: 다음 20일 실현변동성(로그수익률 std*sqrt(252)) > 직전 252일 20일 RV 의 중앙값
    # fwd_ret_5/20/60 (연속값), fwd_maxdd_20/60 (연속값). 마지막 h 일은 NaN.
def base_rates(targets: pd.DataFrame, start=None, end=None) -> pd.Series
def episodes(spy_close: pd.Series, threshold: float) -> pd.DataFrame
    # 고점→저점 낙폭 ≥ threshold 인 구간. 열: peak_date, trough_date, depth, days_to_trough, recovery_date, days_to_recover
    # 병합 규칙(사전 등록): 저점 후 신고점 전에 다시 threshold 하락하면 같은 에피소드의 연장이 아니라 새 에피소드로 세되 recovery_date 는 신고점 회복일
def independent_blocks(n_days: int, h: int) -> int                     # 겹치지 않는 창 수 (n // h)
```

## 평가 — `mrl/evaluate.py`

```python
def directional_scorecard(replay, targets) -> pd.DataFrame
    # 톤별 × 지평(5/20/60): n, n_blocks(독립 창 수), hit(톤이 buy/hold/neutral 이면 y_sign=1 을 예측한 것으로, caution/reduce 이면 0 예측), 기준선(always-up 적중률), fwd_ret mean/median/p10/p90, y_dd5_20 비율
def episode_eval(replay, episodes_df, targets) -> tuple[pd.DataFrame, dict]
    # 에피소드별: 고점 전/후 첫 caution·reduce 톤 등장일, lead_days(양수=고점 전 경고), 저점까지 경고 유지 여부, 놓침 여부
    # 요약: 탐지율, 중앙 리드타임, 연간 오경보(caution/reduce 런 뒤 20일 내 -5% 없음) 횟수, 톤 전환 횟수/년, 중앙 런 길이
def allocation_sim(replay, spy_close, exposure=TONE_EXPOSURE, cost_bps=5) -> dict
    # 톤→비중 적용 일별 수익, CAGR, MaxDD, 최악 월, 전환 횟수/년, buy&hold 대비. 전환 시 비용 반영.
def brier(prob: pd.Series, y: pd.Series) -> float ; def brier_skill(prob, y, ref_prob) -> float
def block_bootstrap_ci(values: pd.Series, block: int, n_boot=2000, ci=0.95, seed=0) -> tuple
def summarize_v0(replay, targets, episodes5, episodes10, spy_close) -> dict   # 리포트가 쓰는 단일 dict (JSON 직렬화 가능)
```

## 리포트 — `mrl/report.py`

```python
def render_backtest_report(summary: dict, out_html: Path, charts: dict[str, bytes]) -> None
    # 자체 완결 HTML(한국어). 섹션: ① 정직한 요약(숫자), ② 방향 성적표(기준선 옆에), ③ 에피소드 표(1993~) + v0 리드타임, ④ 톤별 선행수익 분포,
    # ⑤ 배분 시뮬 vs 보유, ⑥ 판정 전환 빈도, ⑦ 125칸 점유표, ⑧ 데이터 범위·대체 규약·경고, ⑨ 방법 설명. 차트는 base64 PNG 인라인.
def render_index(today: dict, ledger_summary: dict, out_html: Path) -> None
    # 오늘 판정(완성 봉 기준, faithful/completed 둘 다), 마지막 갱신, 백테스트 링크, 장부 요약(표본 수·현재까지 적중률·경고), 정직 문구
def charts_for(summary, replay, spy_close, targets) -> dict[str, bytes]      # matplotlib → PNG bytes: 누적수익 vs 배분, 에피소드 타임라인+톤 밴드, 톤별 선행수익 박스, 전환 빈도
```
스타일: 기존 대시보드와 같은 다크 팔레트(`#0b1220` 배경, `#111a2b` 카드, 텍스트 `#e6eaf2`), 시스템 폰트(외부 폰트 금지), 모바일 대응.

## 장부 — `mrl/ledger.py`

```python
LEDGER = RESULTS_DIR / "track_record.csv"
def append_today(row: dict, path=LEDGER) -> bool       # asof 중복이면 False. 열: asof, recorded_at_utc, variant, states(8), score_d/w/m, overall_d/w/m, tone, spy_close, vix_close, prob_dd5_20(None; Phase 2), run_id
def backfill(path, spy_close) -> pd.DataFrame           # 20/60일 지난 행의 y_sign_20/60, y_dd5_20, fwd_ret_20/60 채움
def summary(path) -> dict                               # n, n_with_outcome, 톤별 적중, 기준선, 마지막 기록일, 결측일(거래일인데 행 없음) 목록
```

## 스크립트

* `scripts/build_cache.py` — 전량 캐시 구축 (최초 1회, 일요일 보정 작업).
* `scripts/run_backtest_v0.py [--variant faithful|completed|both] [--start ... --end ...]` — replay → targets → evaluate → `results/backtest_v0_{variant}.csv` + `results/summary_v0.json` + `docs/backtest_v0.html`.
  완성 봉 원칙: 캐시의 마지막 SPY 봉이 미완성이면(meta.spy_last_bar_complete=False 또는 지금 ET 기준 16:05 이전) end 와 목표변수·에피소드용 종가를 마지막 완성 세션까지로 자르고, 캐시 meta.warnings 를 리포트 ⑧ 에 싣는다.
* `scripts/daily.py` — `update_daily` → 오늘 완성 봉 판정(faithful·completed) → 장부 append/backfill → `docs/index.html`. 장 마감 전이면 "미완성" 경고와 함께 전일 기준.
* `scripts/selftest.py` — 캐시·계약 자기점검(빈 열, 유령 행, 최근성).

## 테스트 (필수)

* `tests/test_parity.py`: reference 모듈을 import 해 같은 입력(캐시에서 만든 2y 창)에 대해 `macd_phase/assess/composite/overall/combo_advice` 결과가 **비트 단위로 동일**한지, 그리고 `v0_day(variant="faithful")`가 reference `build_metrics`+`assess` 흐름과 동일한 상태를 내는지 확인. `mrl.config.V0*` 상수가 reference 의 CONFIG/WEIGHTS/SC/DAILY_ONLY 와 동일한지 확인.
* `tests/test_calendar.py`: 알려진 휴장일·기간말 판정, 완성 봉 제거.
* `tests/test_targets.py`: 합성 시계열로 목표변수·에피소드 정의 검증.
* `tests/test_replay_smoke.py`: 60거래일 구간 replay 가 예외 없이 돌고 열 계약을 만족.

## 워크플로 (.github/workflows)

* `daily.yml`: 평일 21:20/22:20 UTC 후보(ET 16:20 게이트, 서머타임 양쪽), `scripts/daily.py` → `data/ docs/ results/` 커밋. 메일 없음. 실패 시 GitHub 알림.
  게이트의 중복 실행 판정 키는 커밋 제목 `daily: YYYY-MM-DD` 이며, `daily.py` 의 스텝 출력 `market_status` 가 `current` 인 실행만
  그 제목을 쓴다(그 밖은 `daily(<상태>): YYYY-MM-DD`) — 장 마감 전 수동 실행·지연된 원천이 그날을 장부에서 지우지 않게.
* `weekly.yml`: 일요일 13:00 UTC, `build_cache.py` + `run_backtest_v0.py --variant both` → 커밋. (Phase 2: 보정 작업 추가)
* 공통: `concurrency: risk-lab`, `git pull --rebase -X theirs` 후 push, requirements 고정 설치.
