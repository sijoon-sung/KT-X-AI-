# H-0012: 서울의 미래 대여·반납을 예측하도록 사전학습하기

## 가설

프로필을 복원한 H11과 달리 서울의 **다음30분 완료 대여·반납 건수**를 맞히는 과제로 공유 표현을 학습한다.
그 encoder와 예측 신경망을 타슈에서 미세조정하면, 같은 기상·크기·타슈 학습 예산의 scratch 및 AE 대조군보다
다계절 거래 deviance를 각각1% 넘게 줄일 것으로 예상한다. 세분할 최소 이득의3일/7일 구간 하한도1%를 넘어야 지지한다.
계획은 [H-0012 동결 JSON](../research/hypotheses/H-0012.json), 출처는 [입력 명세](../research/datasets/tashu-forecast-transfer-H0012-v1.json)다.

[RegionTrans](https://ijcai.org/proceedings/2019/262)와
[Yao 외 WWW2019 저자 원문](https://faculty.ist.psu.edu/jessieli/Publications/2019-WWW-traffc-predict-multiple-cities.pdf)의
도시 간 예측 이전·미세조정 대조 및 negative transfer 문제를 참고한다. 두 방법의 전체 재현이나 새 방법의 신규성은 주장하지 않는다.

## 실험

계획·구현 게시 후 전처리와 본 실험을 실행 중이며 정확도 결과는 아직 없다. 로컬 RTX4060Laptop8GB를 사용한다.
원본·가중치·개별 기록은 비공개이며 공개 PR에는 해시·분할·요약·실패를 기록한다.
계획 게시 커밋 `77a746998f4347e547b2d24bfbe432c05d7fa7e2`, 주 구현 커밋 `1aee0f33fa04bb5f0d38668d190434721e4b0259`.
계획 해시 `bcdf20d2120fb9b3e39a4551498ee64c04a2813f93f44fc08c4d27bf2685a226`.
입력 명세의 서울 ZIP 개별 `role`에는 H11의 상속 문구가 남아 있으며 H12의 실제 역할은 같은 명세의
`source_seoul_role`과 동결 설계에 명시한 미래 건수 예측/AE 대조다. 원본·고정 해시는 바꾸지 않는다.

### 예측 사전학습과 대조군

서울2024년1~8월로 코호트·프로필·OD·달력률을 만들고9월에서 checkpoint를 선택한다.
10~12월 source 보조 점수도 보고한다. source 자료는H11에서 이미 사용한 원본이므로 미열람 독립시험이라고 부르지 않는다.
타슈는H11과 같은 봄/여름/겨울이후 세 시간 분할에서 각각 다시 학습한다.
과거 다운로드 이력만 학습·선택하며 재고 API는 읽지 않는다.

공유 encoder는420차원 주간 프로필→256→64차원이다. source와 target의 관측 월 지원이 달라지는 문제를 줄이기 위해
encoder에서는 월별 입력을 제외한다. 월·계절 정보는 기존 달력/계절률 맥락으로 남긴다.
source에도 실제 운영 때 구할 수 없는 최근 거래 입력을 넣지 않는다. 가중치만 이전하며 대여소번호를 맞추지 않는다.
미세조정 중 encoder와 GCN까지 모두 학습한다. graph를 거쳐도 encoder에 gradient가 전달되는지 검사한다.

| 후보 | 사전학습과 기상 | 비교 목적 |
|---|---|---|
| legacy_weather_scratch | 기존13개 기상 입력만, 처음부터 학습 | 기상 보완 효과 |
| weather_scratch | 보완54개 입력, 처음부터 학습 | source 없는 주대조군 |
| weather_ae | 서울 주간 프로필 복원 encoder→전체 미세조정 | 예측 과제와 다른 사전학습 주대조군 |
| weather_forecast_encoder | 서울 예측 encoder만 이전 | encoder와 예측 신경망 이전 구분 |
| weather_forecast_full | 서울 예측 encoder+예측 신경망 이전 | 사전 주후보 |

source AE/forecast각2seed·24round+0·각round128step×8192표본, target각2seed·16epoch+0·각epoch64step×8192표본이다.
optimizer step/표본 수를 맞춰도 source 과제 정보량·FLOPs가 같다는 뜻은 아니므로 시간·모수를 별도로 기록한다.
target후보의 구조·미세조정 예산·선택 규칙은 같다. 각seed의 과거 validation에서만 checkpoint를 고른다.

### 기상 보완의 실제 범위

- 기존 [NOAA GHCNh](https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly) 기상에
  서울 김포·대전 인근 청주공항 관측과 최근 현상코드를 추가한다. 공항 관측은 도시·대여소의 실측값과 구분한다.
- primary의 일반 강수 보고는 source223의 nominal-hour/T만 별도로 쓰고 누적기간이 불명확한 source412 generic 강수는 제외한다.
- [기상청 날씨누리 서울108/대전133 과거 일별자료](https://www.weather.go.kr/w/weather/land/past-obs/obs-by-day.do?stn=133&yy=2025&mm=8)의
  실제34개월 HTML을 확보했다. numeric·대시·numeric0을 구분하고 대시를 임의의0강수 관측으로 해석하지 않는다.
- 예측 시점에 그날의 최종 강수를 미리 알 수 없으므로 일별은 **KST날짜−2**만 쓴다.
  시간별은 예측시작1시간 이전/최대4시간 나이, 미래 실측·당일 일요약은 입력하지 않는다.
- GHCNd 추가 다운로드는 최근 기간이 끝나고 희소해 제외했다. 공항 amount 강수 열도 빈값이어서 값을 만들지 않는다.
  시간 강수의 완전한 관측망을 확보한 것으로 보고하지 않는다.

아카이브 수정·실시간 수신은 검증하지 못한 hindcast 맥락 실험이다. 관측소 대표성·보고체계 차이는 한계로 남긴다.
정답은 기록 건수이며 실측 재고·잠재 수요·운영 개선이 아니다.

## 결과

실행 중. 원본 확보·스키마/날짜·출처 확인을 사전학습 완료와 구분한다.
H12 의미 검사7개와 변경한 H11 source callback의 회귀 검사8개를 통과했다.
이틀 전 일별 입력·미래 보고 제외·누적기간 불명 강수 제외·현상코드·동일 encoder 초기화·graph를 통한 gradient·source 미래 label 비의존을 검사했다.
다음 행동은 source 예측 사전학습·동일 대조군·세분할 미세조정을 마치고 모든 결과를 기록하는 것이다.
실패·효과 없음도 같은 기록에 보존하고 결과가 나온 뒤 유리하게 예산이나 후보를 바꾸지 않는다.

고정 모델 예측 CLI는 다음30분 완료 대여·반납 기대건수를 반환한다. 실측 재고 예측을 반환한다고 하지 않는다.
`forecast_transfer_predict.py`는 context/가중치 해시·지원 대여소·시간을 확인하고 고정encoder를 재사용한다.

```text
python research_code/forecast_transfer.py --source-dir PRIVATE_RAW --history-dir PRIVATE_HISTORY --out PRIVATE_OUTPUT
python research_code/forecast_transfer_predict.py --model-dir PRIVATE_OUTPUT/winter --station HISTORICAL_ID --timestamp KST_DATETIME --weather-dir PRIVATE_RAW
```
