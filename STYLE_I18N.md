# 표현 규칙 — 쉬운 한국어 + 한/영 토글 (market-risk-lab)

목표: **고등학생·대학생이 처음 봐도 이해되는 한국어**로 쓰고, 오른쪽 위 버튼으로 **한국어 ↔ English** 를 바꿀 수 있게 한다.
정직성은 절대 낮추지 않는다 — 쉬운 말로 바꾸되 숫자·조건·한계는 그대로 남긴다.

기존 `market-brief` 대시보드와 같은 방식(자바스크립트 0줄, CSS 체크박스 토글)을 쓴다.

## 1. 토글 (기존 플랫폼과 동일한 구조)

* 숨긴 체크박스 `#lang-sw` + `label.lang-btn` (오른쪽 위, 알약 모양 두 칸: `한국어` / `English`).
* 기본 = 한국어. `:checked` 이면 영어. **자바스크립트 없음** (주피터·오프라인·메일 미리보기에서도 작동).
* 좁은 화면(≤ 640px)에서는 절대위치를 풀고 헤더 위 오른쪽 정렬로 내린다 (겹침 방지 — market-brief 에서 겪은 문제).
* 페이지 안의 모든 사용자 대상 문구는 `bi(ko, en)` 로 감싼다:

```python
def bi(ko: str, en: str) -> str:
    """한/영 두 벌을 함께 심는다. 화면엔 선택된 언어만 보인다."""
    return f'<span class="lg ko">{esc(ko)}</span><span class="lg en">{esc(en)}</span>'
```

* 숫자·티커·날짜·수식 기호는 번역하지 않는다(중복 삽입 금지) — 라벨과 문장만 감싼다.
* 표 안의 셀 값은 그대로 두고 **열 제목과 각주만** 이중화한다.
* `<title>`·차트 이미지 안 글자는 영어로 통일한다(차트는 이중화하지 않는다).

## 2. 쉬운 한국어 규칙

1. **먼저 뜻, 그다음 용어.** 전문 용어는 괄호로만 남긴다.
   * ✗ `Brier skill +0.091` → ✓ `기준선보다 9.1% 더 정확 (Brier skill +0.091)`
   * ✗ `블록 부트스트랩 95% 구간` → ✓ `구간을 여러 조각으로 잘라 다시 계산한 95% 범위`
2. **비율은 자연빈도로.** `14.7%` → `10번 중 1~2번 (14.7%)`. 확률 카드는 `100일 중 약 8일` 형태를 우선한다.
3. **한 문장 = 한 가지.** 40자 넘으면 자른다. 수동태·명사형("~에 대한 판정의 수행")은 쓰지 않는다.
4. **바로 뜻이 통하는 말로 바꾼다** (아래 표).
5. **한계는 생략하지 않는다.** 대신 쉬운 말로: `표본이 적어 오차가 큽니다 (12번 중 몇 번인지 수준)`.
6. 영어판은 같은 눈높이의 평이한 영어로 쓴다(전문용어는 처음 나올 때만 괄호).

## 3. 용어 대응표 (한국어 원문 → 쉬운 한국어 → English)

| 원문 | 쉬운 한국어 | English |
|---|---|---|
| 기저율 | 평소 비율 | base rate (how often it normally happens) |
| Brier skill | 정확도 점수(기준선 대비) | accuracy score vs a naive guess |
| 기후학(climatology) | 평소 평균 | the long-run average |
| walk-forward | 과거만 보고 미래를 맞혀 본 방식 | trained only on the past, tested on the future |
| 홀드아웃 | 손대지 않고 남겨둔 최근 구간 | untouched recent data |
| 퍼지(purge) | 겹치는 구간 제거 | removing overlapping days |
| 블록 부트스트랩 | 구간을 잘라 다시 계산한 범위 | resampled range |
| 히스테리시스 | 한 번 바뀌면 쉽게 되돌리지 않기 | requires a clear move to switch back |
| dwell / 체류 | 최소 유지 일수 | minimum days before switching back |
| 결정층 | 판정 규칙 | the decision rule |
| 사다리(M0~M3) | 단계별 모델 | model steps |
| 보정(calibration) | 확률을 실제와 맞추기 | making the stated % match reality |
| 배포/배치(deploy) | 실제 사용 | in use |
| info_only | 참고용만 (실제 사용 안 함) | information only (not in use) |
| 톤 | 신호등 판정 | traffic-light call |
| 노출/비중 | 주식 비중 | how much to hold in stocks |
| 낙폭(drawdown) | 고점 대비 하락폭 | drop from the peak |
| 실현변동성 | 실제로 움직인 폭 | how much it actually moved |
| VIX 내재 확률 | 옵션 시장이 보는 확률 | the options market's own estimate |
| 국면(regime) | 시장 분위기 | market mood (calm / turbulent) |
| 앙상블 | 여러 모형의 평균 | average of several models |
| 그림자 멤버 | 후보 모형 (아직 사용 안 함) | candidate model (not in use yet) |
| 킬룰 | 성적이 나쁘면 끄는 규칙 | the rule that switches it off |
| 드리프트 경보 | 예전과 달라졌다는 경고 | drift warning |
| 오경보 | 헛경보 | false alarm |
| 에피소드 | 하락 사건 | a decline episode |
| 파라미터 예산 | 조정 가능한 숫자의 한도 | how many numbers we may tune |
| post hoc | 결과를 본 뒤에 정한 것 | decided after seeing the results |
| 사전 등록 | 미리 정해둔 것 | decided in advance |

## 4. 카드 첫 화면(가장 많이 읽는 곳) 문구 형식

```
다음 한 달 안에 5% 넘게 떨어질 확률
약 8% — 100일 중 8일쯤
평소에도 15% 정도이니, 지금은 평소보다 낮은 편입니다.
(범위 5~21% · 아직 시험 운용이라 투자 조언이 아닙니다)
```

## 5. 검사 (테스트로 강제)

* 렌더된 모든 페이지에서 `class="lg ko"` 와 `class="lg en"` 의 개수가 같아야 한다.
* 한국어 문구가 영어 스팬 안에, 영어 전용 문구가 한국어 스팬 안에 남아 있으면 실패(티커·숫자·고유명사 화이트리스트 제외).
* 좁은 화면 CSS(≤640px)에서 토글이 헤더와 겹치지 않아야 한다(규칙 존재 검사).
* 용어 대응표에 있는 "원문" 표현이 페이지에 그대로 남아 있으면 실패(쉬운 말로 바뀌었는지 검사).
