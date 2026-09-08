# market-risk-lab

기존 가족용 시장 대시보드(`market-brief`, v0)의 규칙을 **동결한 채로** 30년 데이터 위에서 정직하게 검증하고,
매일의 판정을 장부에 남기는 연구 플랫폼. 게시 주소: <https://uncmac.github.io/market-risk-lab/>

> 이 저장소는 투자 조언이 아니라 **v0 규칙이 얼마나 맞았는지 재는 도구**다.
> 모든 성적표는 "항상-상승" 기준선 옆에 놓고 읽는다(아래 2절).

---

## 1. 목적 — `market-brief` 와의 관계

* `market-brief`(기존 대시보드, 매일 가족에게 게시되는 v0)는 **한 줄도 건드리지 않는다.** 그대로 계속 돈다.
* 이 저장소는 별도 플랫폼이다. v0 원본은 `reference/market_dashboard_v0.py` 에 **읽기 전용 사본**으로 두고,
  `mrl/signals_v0.py` 가 같은 규칙을 그대로 옮긴다. `tests/test_parity.py` 가 두 구현이 비트 단위로 같음을 검사한다.
* Phase 1 에서 만드는 것 다섯 가지:
  1. 30년 가격·변동성·심리 데이터 캐시(`data/`)
  2. 완성 봉 기준 v0 신호의 충실한 재현(`mrl/replay.py`)
  3. 사전 등록된 목표변수·에피소드 표(`mrl/targets.py`, `VALIDATION.md`)
  4. v0 규칙의 정직한 백테스트 리포트(`docs/backtest_v0.html`)
  5. 매일 판정을 기록하는 장부(`results/track_record.csv`)

---

## 2. 정직한 전제 (`VALIDATION.md` §0 요약)

SPY 1993-01-29 ~ 2026-09-04, 8,458거래일 실측:

* **상승 비율**: 일 54.1% · 20거래일 창 65.4%(겹치지 않는 창 422개 기준 64.0%) · 60거래일 창 72.0%(140개 기준 70.0%).
  2013년 이후는 더 강세(55.3% / 68~69% / 77~79%). 즉 "항상 오른다"고만 말해도 한 달 뒤 적중률은 3분의 2다.
* **방향(오를지 내릴지)은 사실상 예측 불가.** 학계 최고 모형의 월간 out-of-sample R² 는 0.5~1%이며 이는 항상-상승 대비 적중률 +0.3%p 수준.
  5%p 우위를 α=5%, 검정력 80%로 확인하려면 20일 창 717개(≈57년)가 필요하다.
* **위험(변동성·낙폭)은 예측 가능.** VIX 수준만으로 다음 20일 실현변동성 상위 여부 AUC 0.87, 20일 내 -5% 낙폭 AUC 0.76,
  반면 다음 20일 상승 여부 AUC 는 0.52(동전 던지기 수준).
* **따라서 이 플랫폼의 주 목표는 "다음 한 달의 위험"(`y_dd5_20`)이며, 방향 적중률은 항상 기준선 옆에 참고로만 보고한다.**
* 표본이 작다: 1993년 이후 -10% 이상 하락 에피소드는 12회, -5% 이상은 36회, -20% 이상은 4회.
  어떤 성적표도 "12회 중 X회" 수준의 넓은 구간을 벗어날 수 없다.
* 워치리스트(주도주 신호 P7)는 2026년에 고른 종목이므로 **생존편향을 제거할 수 없다.** 리포트에 `n_watch_avail` 을 함께 표시한다.

---

## 3. 아키텍처 요약 (자세한 계약은 `ARCHITECTURE.md`)

```
market-risk-lab/
  mrl/            패키지
    config.py       v0 동결 상수(V0, V0_WEIGHTS, V0_SC …) · 캐시 티커 · 목표변수 상수 · 경로
    data.py         yfinance/CBOE/CNN 수집, Bundle 캐시(build_cache / update_daily / load_cache / apply_guards)
    calendar_us.py  NYSE 휴장일 · 완성 봉 판정(session_complete) · 주/월 기간말 판정 · v0 방식 리샘플
    signals_v0.py   v0 신호 포팅: window(asof, variant) → v0_metrics → v0_assess → v0_composite → v0_overall → v0_combo → v0_day
    replay.py       거래일마다 v0_day 를 돌려 판정 시계열 생성(replay_v0) · 125칸 점유표(cell_table)
    targets.py      목표변수(y_sign_h, y_dd5_20, y_dd10_60, y_vol_20, fwd_ret_h …) · 낙폭 에피소드 · 독립 창 수
    evaluate.py     방향 성적표 · 에피소드 평가 · 배분 시뮬 · Brier · 블록 부트스트랩 · summarize_v0
    report.py       backtest_v0.html · index.html 렌더(다크 팔레트, 시스템 폰트, 모바일 대응)
    ledger.py       track_record.csv append / backfill / summary
  scripts/        build_cache.py · run_backtest_v0.py · daily.py · selftest.py
  tests/          parity(필수) · calendar · targets · replay smoke
  data/           커밋되는 캐시(CSV + meta.json)
  results/        백테스트 산출물, track_record.csv
  docs/           GitHub Pages(index.html, backtest_v0.html, .nojekyll)
  reference/      v0 원본 사본(읽기 전용)
  .github/workflows/  daily.yml · weekly.yml
```

데이터 흐름:

```
yfinance · CBOE CSV · CNN F&G  ──build_cache / update_daily──▶  data/*.csv (Bundle)
Bundle + asof ──window(variant)──▶ v0_day ──replay_v0──▶ 판정 시계열 ──evaluate──▶ summary_v0.json ──report──▶ docs/backtest_v0.html
Bundle(오늘) ──v0_day(faithful, completed)──▶ ledger.append_today / backfill ──report.render_index──▶ docs/index.html
```

지켜야 할 세 가지 원칙:

* **점(point-in-time) 원칙** — 날짜 t 의 계산은 t 종가까지의 데이터만 쓴다. 미래 행은 각 함수가 직접 걸러낸다.
* **완성 봉 원칙** — 일봉은 16:00 ET 마감 후에만 완성, 주봉/월봉은 그 기간 마지막 거래일 종가 이후에만 완성.
  `variant="faithful"` 은 v0 라이브가 보던 부분 봉을 그대로 재현하고, `variant="completed"` 는 완성 봉만 쓴다.
  **두 결과의 차이 자체가 부분 봉 문제의 크기다.**
* **조용한 실패 금지** — 예외를 던지거나 `warnings.warn` + `meta.warnings` 에 기록한다.

---

## 4. 로컬 실행

```bash
git clone https://github.com/uncmac/market-risk-lab.git
cd market-risk-lab
python -m venv .venv
# Windows: .venv\Scripts\activate  /  macOS·Linux: source .venv/bin/activate
pip install -r requirements.txt          # Python 3.12 · 버전 고정
```

Windows 콘솔에서 한글 출력이 깨지면 실행 전에 `set PYTHONIOENCODING=utf-8`(cmd) 또는
`$env:PYTHONIOENCODING="utf-8"`(PowerShell)을 설정한다.

순서대로 세 단계:

```bash
# 1) 캐시 전량 구축 (최초 1회 · 이후 매주 일요일 자동). yfinance 일괄 다운로드 + CBOE + CNN F&G 이력 → data/
python scripts/build_cache.py

# 2) v0 백테스트: replay → targets → evaluate → results/backtest_v0_{variant}.csv · results/summary_v0.json · docs/backtest_v0.html
python scripts/run_backtest_v0.py --variant both          # faithful | completed | both, --start/--end 선택
#    2015~2026(약 2,950거래일) 한 변형에 10분 이내가 목표

# 3) 오늘 판정: update_daily → 완성 봉 판정(faithful·completed) → 장부 append/backfill → docs/index.html
python scripts/daily.py
#    장 마감(16:00 ET) 전에 돌리면 "미완성" 경고와 함께 전일 기준으로 기록한다
```

검증:

```bash
pytest -q                      # test_parity(비트 단위 동일성) · test_calendar · test_targets · test_replay_smoke
python scripts/selftest.py     # 캐시 자기점검: 빈 열 · 유령 행 · 최근성
```

산출물을 보려면 `docs/index.html`, `docs/backtest_v0.html` 을 브라우저에서 연다(자체 완결 HTML, 외부 자원 없음).

---

## 5. 리포트 읽는 법

### `docs/index.html` — 오늘 판정

* **오늘 판정**은 완성 봉 기준으로 `faithful` 과 `completed` 두 변형을 나란히 보여 준다. 둘이 다르면 그날은
  부분 봉(주/월 미완성 기간)이 판정을 흔들고 있다는 뜻이다.
* **마지막 갱신** 시각과 데이터 경고(`meta.warnings`: 뒤처진 티커, F&G 조회 실패 등)를 확인한다.
* **장부 요약**: 지금까지 기록된 표본 수, 결과가 확정된 행 수(20/60일 지난 행), 톤별 적중률과 기준선, 결측일 목록.
  표본이 수십 개일 때의 적중률은 의미가 없다 — 킬 규칙(6절)의 문턱까지는 숫자를 보되 결론을 내리지 않는다.

### `docs/backtest_v0.html` — v0 백테스트

| 절 | 내용 | 읽을 때 |
|---|---|---|
| ① 정직한 요약 | 핵심 숫자 몇 개 | 첫 줄의 v0 결과는 이후 모든 실험 리포트에 영구 표기된다 |
| ② 방향 성적표 | 톤별 × 지평(5/20/60일) 적중률, n, 독립 창 수, 95% 구간 | **항상 오른쪽의 "항상-상승" 기준선과 비교.** 기준선을 못 넘으면 방향 정보는 없는 것 |
| ③ 에피소드 표 | 1993년 이후 ≥5%/≥10% 하락마다 첫 경고일, 리드타임, 놓침 여부, 연간 오경보 | "12회 중 X회" 형식. 리드타임 양수 = 고점 전 경고 |
| ④ 톤별 선행수익 분포 | 평균·중앙·10/90분위, `y_dd5_20` 비율 | caution/reduce 뒤 낙폭 비율이 기저율보다 높아야 위험 신호로서 값어치가 있다 |
| ⑤ 배분 시뮬 vs 보유 | 톤→비중(buy/hold/neutral 100%, caution 50%, reduce 25%), 비용 5bp | CAGR·MaxDD·최악 월·전환 횟수를 buy&hold 와 비교 |
| ⑥ 판정 전환 빈도 | 톤 전환 횟수/년, 중앙 런 길이 | 너무 잦은 전환은 비용과 피로 |
| ⑦ 125칸 점유표 | (월간, 주간, 일간) 상태 조합별 일수·톤·규칙 번호 | 실제로 쓰인 칸이 몇 개인지, 한 번도 안 나온 규칙이 무엇인지 |
| ⑧ 데이터 범위·대체 규약·경고 | 신호별 재현 가능 구간, `n_watch_avail`, 경고 목록 | P3(F&G) 2020-08 이후, P6(마감 30분) 2023-10 이후만 실데이터 |
| ⑨ 방법 설명 | 창 에뮬레이션, 변형 정의, 목표변수 정의 | |

용어:

* **톤(tone)**: `buy` / `hold` / `neutral` / `caution` / `reduce` — v0 의 월·주·일 판정 조합(`combo_advice`)이 내는 최종 결론.
* **독립 창 수(n_blocks)**: 겹치지 않는 20일(또는 60일) 창의 수. 적중률의 실질 표본 크기는 n 이 아니라 이 숫자다.
* **대체 규약**: 데이터가 없는 구간의 신호(P3·P6)는 v0 라이브에서 "조회 실패 시 가중치 제외"하는 것과 같은 방식으로 제외한다.

### `results/track_record.csv` — 장부

열: `asof, recorded_at_utc, variant, states(8개 신호), score_d/w/m, overall_d/w/m, tone, spy_close, vix_close, prob_dd5_20, run_id`
+ 20/60일이 지난 뒤 채워지는 `y_sign_20/60, y_dd5_20, fwd_ret_20/60`. 같은 `asof` 는 한 번만 기록된다.
`prob_dd5_20` 은 **배포 확률** — `summary_p2.json.acceptance` 가 배치한 단의 확률이다(2026-09-08 현재 `deploy_mode=tones`, `tone_model=M1`; `info_only` 면 생산 모델 M3 를 정보로 기록). 행마다 출처 단은 `p2_prob_model_id`·`p2_tone_model`·`p2_deploy_mode` 로 확인한다(계약: `mrl/ledger.py` 열 의미 절). 그 옆의 `p2_*` **26개** 열(§12)이 사다리 오늘값(`p2_p_m1/m2/m3`)·구간·상태·귀속을 담는다(Phase 1 시절 행은 NaN).

---

## 6. 로드맵

| Phase | 내용 | 상태 |
|---|---|---|
| **1** | 캐시 · v0 동결 포팅(parity) · 목표변수/에피소드 · 정직한 백테스트 리포트 · 일일 장부 · 자동화 | 진행 중 |
| **2** | 보정 모델: `y_dd5_20` 확률 추정. **적합 파라미터 최대 5개**(구현: 정확히 4개), 확장창 walk-forward(연 1회 재적합, 20거래일 퍼지, 블록 18~24개월). 홀드아웃 `2024-09-01` 이후는 최종 검증 1회 외 접근 금지. 채택 조건: 모든 블록에서 Brier skill(vs 기저율) > 0 **이고** VIX 내재 확률 대비 skill ≥ 0. 위반 시 v0 유지. 실험은 `VALIDATION.md` §8 장부에 번호를 붙여 기록 | 구현 완료 — 실험 #2 literal 실패 → 장부 #2a **경로 (b) 채택**(post hoc, 소유자 결정 2026-09-08) → `deploy_mode=tones`, 톤 모델 **M1**, M3 는 정보 표시(§12) |
| **3** | 국면 모델 · 앙상블 · 비중 제안. **킬 규칙**: 라이브 장부에 ≥5% 낙폭 에피소드 8회 이상 또는 3년 경과 후(늦은 쪽) 블록 부트스트랩 95% 구간으로 평가, `y_dd5_20` Brier skill 이 0 을 넘지 못하면 대시보드를 "정보 제공 전용"으로 전환(톤·비중 제안 숨김) | 예정 |

검토했으나 채택하지 않은 방향(`VALIDATION.md` §9): 계절성, 뉴스 LLM 감성, 딥러닝 가격 예측, 거시 나우캐스팅.

---

## 7. 데이터 소스와 이력 깊이

정확한 티커별 시작/끝 날짜는 캐시 구축 때 `data/meta.json` 에 기록된다(`first/last per ticker`). 아래 연도는 대략값이다.

| 파일 | 원천 | 내용 | 이력 깊이 | 갱신 방식 |
|---|---|---|---|---|
| `data/close.csv` | Yahoo Finance (yfinance 1.2.1, `period="max"`, `auto_adjust=True`) | 조정 종가. 핵심 SPY·^VIX·BTC-USD, FANG 8종, 워치리스트 28종, Phase 2 입력(폭·신용·금리·거시·팩터) | SPY 1993-01-29~, ^VIX 1990~, BTC-USD 2014-09~. FANG 은 META(2012-05) 상장 이후 8종 완비. 워치리스트는 2013년 22개 → 2026년 27~28개(예: PLTR 2020-09, IBIT 2024-01 상장). 폭·섹터 ETF 1998~2003~, 신용 ETF 2002~2007~, 금리 지수(^TNX 등) 1960년대~, 상품 선물 2000~, 팩터 ETF 2011~2013~ | 매일·매주 **전량 재다운로드**(증분 append 금지 — 배당·분할 자동조정이 과거를 다시 쓰기 때문) |
| `data/spy_ohlc.csv` | Yahoo Finance | SPY Open/High/Low/Close/Volume(조정) | 1993~ | 전량 재다운로드 |
| `data/spy_eod.csv` | Yahoo Finance `interval="1h"` | 세션 마지막 1시간봉(15:30~16:00 ET)의 open/close/ret, 반일장 표시 | Yahoo 가 1시간봉을 **최근 730일**만 주므로 2023-10~. 그래서 append 로 쌓아 간다 | append(새 날짜만) |
| `data/cboe.csv` | Cboe 공개 CSV (`cdn.cboe.com`, 키 불필요) | VIX3M, VIX9D, VVIX, SKEW | Cboe 가 배포하는 전체 이력(시리즈별로 다름: SKEW 1990년대~, VIX3M·VVIX 2000년대 중반~, VIX9D 2011~ 무렵) | append |
| `data/fg_history.csv` | CNN Fear & Greed 비공식 엔드포인트(`graphdata/{date}`) | 점수 + 구성요소 | **2020-08-03~** (날짜 지정 엔드포인트의 이력 시작, 2026-09-07 확인). 그 이전 구간의 P3 는 대체 규약대로 제외 | append. 브라우저 수준 헤더가 필요하며 조회 실패 시 v0 와 같이 해당 신호 제외 |
| `data/meta.json` | — | 수집 시각(UTC), 티커별 첫/마지막 날짜, yfinance 버전, 경고 목록 | | 매 구축 |

신호별 재현 가능 구간(`VALIDATION.md` §3): 6개 신호(P1·P2·P4·P5·P7·P9)의 공통 구간 시작은 **2015-01-02**(`BACKTEST_START`),
P3(F&G)는 2020-08-03~, P6(마감 30분 매수)는 2023-10~.

캐시 CSV 는 `index_label="date"`, 소수 6자리, tz-naive 거래일 인덱스(BTC-USD 는 주말 포함이라 union). 다운로드 후
`apply_guards` 가 SPY 마지막 일자 이후의 유령 행을 제거하고, 3거래일 이상 뒤처진 티커를 `meta.warnings` 에 기록한다.

---

## 8. Yahoo Finance 데이터 이용 조건에 관한 메모

* `yfinance` 는 Yahoo 와 무관한 오픈소스 커뮤니티 도구이며, Yahoo Finance 데이터는 Yahoo 의 이용 약관에 따라
  **개인적·비상업적 용도**로만 쓸 수 있다.
* 이 저장소의 `data/` 캐시는 원시 시세가 아니라 **분할·배당 자동조정을 거친 파생 종가 시계열**로, 오직 이 연구
  (v0 규칙 검증)를 재현하기 위해 커밋해 둔 것이다. 재배포·상업적 이용 목적이 아니다.
* 저장소를 포크하거나 다른 목적으로 쓰려면 캐시를 그대로 가져가지 말고 `python scripts/build_cache.py` 로 직접
  받아, 각자 Yahoo·Cboe·CNN 의 약관을 확인한 뒤 사용한다.
* Cboe CSV 는 Cboe 가 공개 배포하는 자료이고, CNN Fear & Greed 는 문서화되지 않은 비공식 엔드포인트라 언제든 막힐 수 있다.
  막히면 P3 신호는 자동으로 제외되며(대체 규약), 리포트의 ⑧ 절에 경고로 남는다.

---

## 9. 자동화 (GitHub Actions)

두 워크플로 모두 `concurrency: risk-lab`(동시 커밋 방지, 진행 중 취소 없음), `permissions: contents: write`,
Python 3.12, `requirements.txt` 고정 설치, 커밋 전 `git pull --rebase -X theirs`(충돌 시 방금 만든 로컬 커밋 편) 후 push.
메일 알림은 없고, 실패하면 GitHub 의 워크플로 실패 알림이 온다.

### `daily.yml` — 평일 장 마감 후 판정 기록

* cron `20 21,22 * * 1-5`: UTC 고정 스케줄이 서머타임 양쪽을 덮도록 후보를 둘 둔다.

  | UTC | 서머타임(EDT, UTC-4) | 표준시(EST, UTC-5) |
  |---|---|---|
  | 21:20 | 17:20 ET | 16:20 ET |
  | 22:20 | 18:20 ET | 17:20 ET |

* 게이트(`gate` 잡, bash): 뉴욕 시각이 **16:10 이후**이고 **평일**이어야 통과. 두 후보 모두 어느 계절에나 16:10 을 넘기므로
  **같은 뉴욕 날짜의 `daily: YYYY-MM-DD` 커밋이 이미 있으면 건너뛴다.** 결과적으로 하루 한 번만 실행되고, 첫 후보가
  실패하거나 GitHub 가 스케줄을 누락했을 때만 두 번째 후보가 대신 뛴다. cron 지연이 흔해서 시간 "창"이 아니라 "이후" 조건을 쓴다.
* 수동 실행(`workflow_dispatch`)은 게이트를 거치지 않는다. 장 마감 전이면 `daily.py` 가 "미완성" 경고와 함께 전일 기준으로 기록한다.
* **커밋 제목은 `daily.py` 의 스텝 출력 `market_status` 로 정한다.** 오늘 세션을 실제로 기록한 실행(`current`)만
  게이트가 인식하는 `daily: YYYY-MM-DD` 를 쓰고, 그 밖(`incomplete`·`pre_open`·`stale`·`weekend`·`holiday`)은
  `daily(<상태>): YYYY-MM-DD` 로 커밋한다. 그래서 장 마감 전 수동 실행이나 Yahoo 응답이 늦은 첫 후보가 그날의 게이트를
  잠그지 않고, 두 번째 후보(또는 다음 수동 실행)가 마감 봉이 생긴 뒤 그날을 기록한다. (전에는 어떤 실행이든 같은 제목을
  써서, 예컨대 14:00 ET 수동 실행 한 번이 그날을 장부에서 영구히 비웠다.)
* 휴장일(평일)에도 실행되지만 `daily.py` 가 전일 기준으로 처리하고, 장부는 같은 `asof` 를 두 번 쓰지 않는다.
* 커밋 대상: `data/ docs/ results/`. 변경이 없으면 커밋하지 않는다. 잡 타임아웃 30분.

### `weekly.yml` — 일요일 보정 작업

* cron `0 13 * * 0`(일요일 09:00 EDT / 08:00 EST) + 수동 실행(변형 선택 가능, 기본 `both`).
* `scripts/build_cache.py` → `scripts/run_backtest_v0.py --variant both` → `weekly: YYYY-MM-DD` 커밋. 타임아웃 60분.
* Phase 2 단계가 그 뒤에 붙는다: `scripts/run_calibration.py --acceptance-rule amended`(하드컷 walk-forward + 라이브 계수 결정론 검사; `amended` 는 장부 #2a 경로 (b) 결정에 따른 사후 채택 규칙) →
  `scripts/run_backtest_v1.py --all-configs` → `summary_p2.json.determinism.ok` 확인 → 커밋. 어느 단계든 실패하면 커밋하지 않는다(§12).

### GitHub Pages

저장소 Settings → Pages → Source: *Deploy from a branch* → `main` / `/docs`. `docs/.nojekyll` 은 Jekyll 빌드를 건너뛰고
파일을 그대로 서빙하게 하는 표식이다(빌드 지연·밑줄 파일 무시 문제 방지).

---

## 10. 테스트

| 파일 | 검사 내용 |
|---|---|
| `tests/test_parity.py` | reference 모듈을 import 해 같은 입력에 대해 `macd_phase / assess / composite / overall / combo_advice` 가 **비트 단위로 동일**한지, `v0_day(variant="faithful")` 가 reference 흐름과 같은 상태를 내는지, `mrl.config.V0*` 가 reference 의 CONFIG/WEIGHTS/SC/DAILY_ONLY 와 같은지 |
| `tests/test_calendar.py` | 알려진 휴장일 · 주/월 기간말 판정 · 미완성 봉 제거 |
| `tests/test_targets.py` | 합성 시계열로 목표변수 · 에피소드 정의 검증 |
| `tests/test_replay_smoke.py` | 60거래일 구간 replay 가 예외 없이 돌고 열 계약을 만족 |
| `tests/test_features.py` · `test_vol.py` · `test_model.py` · `test_calibrate.py` · `test_decision.py` · `test_events.py` | Phase 2 모듈 계약: 점 원칙(무작위 T 20개 비트 동일), VIX 내재 확률 식, 퍼지·기후학·블록 타일링, walk-forward 결정론·누수 카나리, 4-파라미터 예산, 결정층 히스테리시스·dwell·churn, 이벤트 표 |
| `tests/test_ledger_p2.py` · `test_report_p2.py` | 장부 P2 열 왕복·라이브 Brier, p2 카드·보정/결정층 리포트 렌더(BAD_TOKEN 없음) |
| `tests/test_scripts_integration.py` (Phase 2 부분) | `run_calibration.py --end 2006-12-29` 산출물·엄격 JSON·하드컷(마지막 20 라벨 NaN)·결정론 게이트(같은 입력 두 번 = 같은 CSV, 계수를 틀어 심으면 exit 1)·홀드아웃 거부(exit 1)·unlock 존재 시 exit 2, `run_backtest_v1.py` 완주·spec 불일치 exit 1, `daily.py` P2 장부 열·카드·모델 없음/spec 불일치 exit 1 |

`test_parity.py` 통과가 "v0 동결 성공"의 정의다(`VALIDATION.md` §4).

---

## 11. 문서

* `ARCHITECTURE.md` — 모듈 계약(함수 시그니처·열 이름·원칙). 다른 모듈이 의존하므로 계약을 바꾸면 여기부터 고친다.
* `ARCHITECTURE_PHASE2.md` — Phase 2 계약(보정 모델 p2 · 결정층 v1 · 스크립트 · 워크플로 · 테스트). 아래 §12 의 근거.
* `VALIDATION.md` — 사전 등록 문서. 목표변수·평가·수용/킬 규칙·실험 장부. **결과를 본 뒤에는 항목을 지우지 않고 취소선으로 남긴다.**

---

## 12. Phase 2 — 보정 모델 p2 · 결정층 v1 (실험 #2)

v0 벤치마크 위에, **적합 파라미터 정확히 4개**(절편 + x_vix·x_har·x_ma 계수)의 로지스틱으로 `y_dd5_20`(다음 20거래일 안에 -5%)의
**확률**을 만들고, 그 확률을 기후학(기저율)·VIX 내재 확률 대비 Brier skill 로 채점하며, 3단계 상태기계(정상/주의/축소)로 가족용 톤에
맵핑한다. v0 는 한 줄도 건드리지 않았고 모든 Phase 2 리포트 첫 줄에 v0 결과를 영구 표기한다. 계약은 `ARCHITECTURE_PHASE2.md`.

### 12.1 무엇이 어디에

```
mrl/features.py    x_vix = logit(VIX 내재 확률 B1) · x_har = ln(HAR σ, GK+야간갭) − ln(VIX/100) · x_ma = Close/SMA180 − 1 (표준화 없음)
mrl/vol.py         GK+OV 일변동 · HAR 성분 · 적합 log-HAR(보조 출력, 확률에 결합 금지) · 자기점검
mrl/model.py       LogitModel(4 파라미터) · 사다리 M0(VIX 공식)→M1(보정)→M2(+실현변동성)→M3(+추세) · 귀속 · 저장/적재
mrl/calibrate.py   재적합 일정(1월 첫 거래일) · 20일 퍼지 · 기후학 · walk-forward · 블록 채점 · 부트스트랩/DM/위상 검정 · §6 판정
mrl/decision.py    r = p/clim 상태기계(격상 즉시, 격하 5세션 체류) · KPI · churn 경보
mrl/events.py      FOMC/OPEX/쿼드위칭/NFP 달력(표시 전용)
scripts/run_calibration.py   주간: 하드컷 → 사다리 walk-forward → 검정 → HAR → results/calib_p2_walkforward.csv · summary_p2.json · model_p2.json · docs/calibration_p2.html
scripts/run_backtest_v1.py   주간: OOS 확률 → 결정층 → v0 와 같은 성적표 → results/backtest_v1.csv · summary_v1.json · docs/backtest_v1.html
scripts/daily.py             매일: model_p2.json 을 **읽기만** 해서 오늘 확률·상태·귀속 → 장부 p2_* 열 · index.html 의 p2 카드 (재적합 금지)
```

### 12.2 실행

```bash
python scripts/run_calibration.py                 # 하드컷(2024-08-30)·walk-forward·검정·리포트. 로컬 ~11초
python scripts/run_backtest_v1.py --all-configs   # 결정층 기본 + 민감도 3종(wide·symmetric_dwell·no_dwell, 보고만). ~15초
python scripts/daily.py                           # v0 판정 뒤 P2 확률·상태·귀속을 장부에 쓰고 카드를 그린다 (+<1초)
python scripts/run_calibration.py --holdout-final # 홀드아웃 최종 검증 **1회**. results/holdout_unlock.json 이 있으면 exit 2
pytest -q tests/test_scripts_integration.py       # 짧은 창(2003~2006)으로 세 스크립트를 끝까지 돌려 산출물 계약을 검사
```

### 12.3 지켜지는 규칙 (스크립트가 강제)

* **홀드아웃 하드컷** — `run_calibration.py` 는 `--holdout-final` 없이는 `bundle.close·spy_ohlc·cboe·fg·eod` 와 목표변수를 모두
  2024-08-30 에서 **데이터 자체를** 자른다(라벨 마스크가 아님). 그래서 2024-08 의 마지막 20세션 라벨은 NaN 이고 24개월 블록 #11 의
  채점 행은 418 이 아니라 398 이다. `--end` 가 2024-09-01 이후면 exit 1. `run_backtest_v1.py` 도 OOS 끝에서 종가·에피소드를 자른다.
* **점 원칙** — `build_features(bundle, asof)` 는 asof 이후 행을 먼저 버리고 계산한다(무작위 T 20개 비트 동일성 테스트).
* **파라미터 예산** — 매 재적합에서 M3 의 `n_params == 4` 를 assert. 5번째 슬롯은 비어 있고 장부 #3~#5 에 예약.
* **결정론** — 난수는 전부 seed 0. 같은 입력의 두 실행은 CSV 바이트가 같다(테스트). 주간 작업은 새로 적합한 라이브 계수를
  `results/model_p2.json` 과 비교한다: **입력 데이터 지문(SPY OHLC + VIX)이 같은데 계수 차 > 1e-9 면 exit 1(파일·커밋 없음)**.
  지문이 다르면(Yahoo 재다운로드의 배당 조정계수 반올림 — SPY 조정 OHLC 가 상대 ~1.5e-6, 계수가 ~9e-6 움직인다, 실측 2026-09-08;
  6자리 CSV 반올림만이면 ~2e-8) 결정론 검사가 아니므로 이동 폭을 `summary_p2.json.determinism` 에 기록·경고하고 새 파일을
  쓴다(1e-4 초과는 '데이터 수정 의심' 경고 → 장부 검토; 손으로 고친 자료는 ≥1e-3 움직인다).
  코드가 바뀌면(spec_sha256 변경) 새 파일 + `run.spec_changed=true` + 장부 기재 요구. 복구는 `results/model_p2.json` 을 손으로
  지우는 것뿐이며 git 에 남는다.
* **재적합은 주간에만** — `daily.py` 는 `model_p2.json` 이 없거나 `spec_sha256` 이 현재 코드와 다르면 exit 1(장부·페이지를 쓰지 않는다).
  입력 결측(VIX·OHLC 없음)은 실패가 아니라 "확률 계산 불가": 확률·귀속 NaN, 상태 유지, 장부 `p2_input_missing` 에 사유.
* **조용한 실패 금지** — v0 참조선만 조건 미충족 시 생략(+경고); 나머지는 예외로 죽는다. 모든 표에 n_blocks(=n/20) 병기.

### 12.4 결과 (계산 2026-09-07 · 채택 판정 2026-09-08 #2a 경로 (b) · 캐시 2026-09-04 · OOS 2003-01-02~2024-08-30, 5,453세션 · 라벨 5,433행 · 창 271 · 기저율 15.4%)

| 단 | Brier | BSS vs 기후학 | vs VIX(B1) | vs BGK | vs M1 | AUC | 24개월 블록 >0 (기후학) | ≥0 (B1) |
|---|---|---|---|---|---|---|---|---|
| M0 VIX 공식 | 0.1512 | −0.136 | 0 | −0.124 | −0.235 | 0.688 | — | — |
| M1 VIX 보정 | 0.1224 | +0.080 | +0.191 | +0.090 | 0 | 0.680 | 10/11 | 9/11 |
| M2 +실현변동성 | 0.1217 | +0.086 | +0.195 | +0.095 | +0.006 | 0.685 | 10/11 | 9/11 |
| **M3 +추세** | **0.1210** | **+0.091** | **+0.200** | **+0.101** | **+0.012** | 0.687 | **10/11** (실패 2019-20 −0.037) | **9/11** (실패 2007-08 −0.082, 2017-18 −0.029) |

* 단 간 손실차(×1e-4, 40일 블록 부트스트랩 95%, DM t HAC 19): M0→M1 +288 [173, 401] t 5.4 · M1→M2 +7.1 [−1.3, 15.1] t 1.7 ·
  M2→M3 +7.2 [−7.8, 22.5] t 1.0 · **M1→M3 +14.3 [−3.0, 32.6] t 1.6 (p 0.11)**. VIX 대비 skill 의 대부분은 편향 보정(M1)이고
  HAR·추세의 정보 이득은 0 을 포함하는 구간 안에 있다 — 가족용 문구는 "보정된 VIX 에 조금 더".
* **§6 문자 그대로의 판정: 실패**(실패 블록 3·8·9) → 사전 등록 결과는 `deploy_mode=info_only` 였다. #2a 완화안(post hoc)으로
  채점하면 M1 만 A∧B∧C 를 통과한다(M3 는 기준 C 미충족: M2→M3 손실차 95% 하한 −7.8×1e-4 ≤ 0) → **소유자가 2026-09-08 장부 #2a 에
  경로 (b) 를 기재해 채택**했고, 이후 모든 실행·주간 워크플로는 `--acceptance-rule amended` 로 채점한다 → **`deploy_mode=tones`,
  `tone_model=M1`**; M3 는 정보 표시(배포 안 함). 카드는 "톤 적용 — M1 배포" 로 상태·톤·비중을 제안한다.
* **블록 민감도(§8.2, 2026-09-08 추가)**: 같은 완화안 규칙을 사전 등록된 18개월 블록 표로 재채점하면 어느 단도 A 를 통과하지 못해
  `deploy_mode=info_only` 가 된다(M1 최소 블록 BSS 기후학 −0.0996, 2018-01~2019-09). **배치 판정은 블록 정의에 의존한다** —
  `summary_p2.json.acceptance.sensitivity_blocks` 에 기계가 읽을 수 있게 남고, 보정 리포트 머리·판정 상자·카드 정직 스트립에 상시 표시된다.
* 결정층 v1(default) — **배포 = M1**: 상태 전환 4.4회/년(v0 82.6), 경고 점유 22.5%(v0 46.5%), 경고 상태 -5%/20일 비율
  22.5%(주의)/42.9%(축소) vs 정상 12.2%, 진짜 경보 비중 34.2% vs 기저율 15.4% vs v0 12.8%, 5세션 최대 변경 2(상한 3),
  **churn 경보 68세션(최대 13/252) — `summary_v1.json.flags` 에 경고**. 배분 CAGR 8.6% vs 보유 10.8%,
  MaxDD −34.8% vs −55.2% (2003~2024-08; v0 2015~26 은 9.9% vs 13.9%, −16.1% vs −33.7% — 구간이 달라 직접 비교 불가).
* (정보 표시, 배포 안 함) 같은 규칙을 M3 로 돌리면: 3.9회/년, 경고 점유 20.6%, 26.2%/48.2% vs 정상 11.3%, 진짜 경보 50.0%,
  churn 경보 31세션(최대 14/252), CAGR 10.2%, MaxDD −31.0% — `summary_v1.json.info_layers.M3`. 톤·비중 제안이 아니다.
* 소거: Parkinson(M3-PK) Brier 0.1208, 학습 1996 시작(M3-HAR96) 0.1231, C∈{0.1,1,10} 0.1214/0.1210/0.1209. HAR-RV 보조 출력 OOS log-MAE 0.279,
  R²(log) 0.49 vs VIX 0.22. v0 참조선(Platt 2, 2017~): Brier 0.1427, BSS vs 기후학 −0.010, AUC 0.567 → 모델 입력 후보에서 영구 제외.
* 정직 문구(§16)와 리스크는 `docs/calibration_p2.html` ⑨ 와 카드의 정직 스트립에 상시 표시된다. 자세한 숫자는 `results/summary_p2.json`.
