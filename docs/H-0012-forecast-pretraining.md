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

계획·구현 게시 후 로컬 RTX4060Laptop8GB에서 전처리·서울 사전학습·타슈 미세조정·비교를 완료했다.
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

**사전 주가설 판정: `not_supported`. 표현만 이전한 보조 후보가 유망하다.**
서울2698대여소에서 예측/AE 사전학습 각2seed, 타슈5후보×3분할×2seed의30개 미세조정을 완료했다.
모든 후보와 실패/제약을 보존한다. 실행 근거는
[manifest](../research/runs/H-0012-20261031-forecast-transfer/manifest.json)와
[전체 집계·선택 trace·날짜별 오차](../research/runs/H-0012-20261031-forecast-transfer/summary.json)다.

| 기록 건수 Poisson deviance, 낮을수록 좋음 | 봄1174대여소 | 여름1254대여소 | 겨울이후1318대여소 | 분할 등가중 평균 |
|---|---:|---:|---:|---:|
| 기존 기상·처음부터 학습 | 0.961067 | 0.873200 | 0.889879 | 0.908049 |
| 보완 기상·처음부터 학습 | 0.917307 | 0.849094 | 0.891644 | 0.886015 |
| 보완 기상·서울 AE encoder | 0.946994 | 0.845811 | 0.881186 | 0.891331 |
| 보완 기상·서울 예측 encoder만 이전 | 0.895880 | 0.848250 | 0.864778 | 0.869636 |
| 보완 기상·서울 예측망 전체 이전, 주후보 | 0.892549 | 0.881122 | 0.890068 | 0.887913 |

같은 분할의 표본은 모든 후보에서 같다. 시험 channel 노출은4,015,080 /4,288,680 /2,293,320,
관측대상 날짜57/57/29일이다. H11과 신경망·예산·표준화가 달라 H11 점수와의 차이를 사전학습 단독 효과로 부르지 않는다.
H12 내부에서는 구조·예산을 맞췄으며 neural모수313,026개(encoder124,352 포함)다.
달력/계절률 표·OD·프로필 캐시는 별도다. sourceAE는248,932개이며 encoder는 같은 구조·초기값이다.

### 주가설과 보조 비교를 나눈 해석

- **주후보인 전체 이전:** scratch 대비0.214% 악화, AE 대비0.383% 감소.
  두 주대조군 최소 이득의3일95%구간[-0.951%,0.420%],7일[-1.141%,0.437%].
  상한도 사전1% 조건에 못 미쳐 `not_supported`다. 효과의 부호 자체를 확정한 것은 아니다.
- **예측 encoder만 이전한 사전 보조 후보:** scratch 대비평균1.849% 감소.
  각분할 감소는봄2.336%·여름0.099%·겨울이후3.013%다.
  사후 계산한3일구간[1.569%,2.222%],7일[1.480%,2.202%]가 관측됐다.
  [보조 구간 계산](../research/runs/H-0012-20261031-forecast-transfer/auxiliary-intervals.json)은 다중 비교를 보정하지 않은 기술 결과이며 주가설을 대체하지 않는다.
- **기상 보완 묶음:** scratch동일구조에서평균2.426% 감소.
  봄4.553%·여름2.761% 감소, 겨울이후0.198% 악화다. 기상청 강수나 공항 관측 한 요소의 단독 효과는 분리하지 못했다.

서울 예측 과제는 같은 calendar prior의 epoch0 source9월 deviance1.520501에서
seed11의1.353022(epoch5),seed29의1.317067(epoch9)로 선택됐다.
source10~12월 보조 deviance1.215975, 86관측일·13,921,680channel노출이다.
AE는2418개 source학습/280개source선택 대여소, 각각round12/11에서MSE0.027669/0.027705를 골랐다.
서로 다른 정답의 MSE와deviance를 직접 순위 비교하지 않는다.
AE는station holdout, 예측 과제는시간 holdout이므로source학습대상·선택구조도 다르다.
관측된 차이를 loss함수 하나의 인과 효과로 확정하지 않는다.

공유 원본 parser는H11 호환용 전체연도 profile파일도 생성하지만 H12학습에는 사용하지 않는다.
H12 source코호트·OD·프로필은1~8월로 별도 구성했다. 원본4385만 행의 수락/제외 규칙은H11과 같으며
타슈 수락9,116,463건도 canonical reader와 일치한다.

### 기상 보완에서 실제로 얻은 것

| 시험 기상 입력 확보율 | 봄 | 여름 | 겨울이후 |
|---|---:|---:|---:|
| primary기온/풍속/습도 | 100% | 97.13% | 86.21% |
| 별도 공항기온/풍속/습도 | 100% | 99.94% | 99.08% |
| primary nominal-hour 강수 보고 | 19.65% | 26.08% | 10.80% |
| 기상청D−2일별표시, numeric 또는dash | 100% | 100% | 100% |

기상청 numeric/dash는 원문 표시다. 이를 현재 시간 강수의100% 관측으로 해석하지 않는다.
기존3시간 강수는 시험에서 여전히 결측이고, nominal-hour 보고도 희소하다.
공항은 별도 관측소 맥락으로 주며 primary의 빈값을 도시 실측값인 것처럼 덮어쓰지 않는다.
새54차원에는 값 외에 mask·나이·현상코드가 포함된다. 사후 수정/수신시각과 공간 대표성 한계도 남는다.

### 실행 확인과 다음 행동

모델 본 실험은첫실행에서완료했고 계획·예산을 바꾸지 않았다. 전체756.1초 중
source학습/선택74.2초·target30개학습/선택243.0초다. 나머지는 파일 읽기·집계·기상 처리·보조 평가·저장 등이다.
최대GPU할당815.2MiB이며 실제장치는로컬RTX4060Laptop이다.
H12 의미 검사7개와 변경한 H11 source callback의 회귀 검사8개를 통과했다.
이틀 전 일별 입력·미래 보고 제외·누적기간 불명 강수 제외·현상코드·동일 encoder 초기화·graph를 통한 gradient·source 미래 label 비의존을 검사했다.
[고정 모델 인수 검사](../research/runs/H-0012-20261031-forecast-transfer/prediction-acceptance.json)에서
다섯후보·10개CPU/GPU예측의최대차8.94e-8, 허용절대/상대오차각2e-5를 확인했다.
잘못된대여소·반시간밖·야간·학습이전 backcast4개를거부하고 기상미제공을missing으로처리했다.

**후속 주후보는 ‘서울 예측 encoder 초기값 + 타슈 전용 GCN 예측망’으로 좁힌다.**
신경망 전체의 반응까지 이전하는 이득은 이번 설정에서 없었고, 도시별 관계/기상 반응의 차이나
미세조정의 망각이 원인 후보다. 원인을 분리해 입증한 것은 아니다.
후속 비교는 encoder/targethead를 분리한 학습률·동결/warm-up·정규화를 검토하고,
그 표현에 조건화한 단순계수형 분포와 Flow Matching을 새 가설에서 비교한다.
H12 주후보를 사후에encoder-only로 바꾸지 않고 주가설 실패를 보존한다.
연구 구현·보완 자료·재현 가능한 후보를 채택하며 새 재고 성능이나 운영 정책 효용은 아직 주장하지 않는다.

고정 모델 예측 CLI는 다음30분 완료 대여·반납 기대건수를 반환한다. 실측 재고 예측을 반환한다고 하지 않는다.
`forecast_transfer_predict.py`는 context/가중치 해시·지원 대여소·시간을 확인하고 고정encoder를 재사용한다.

```text
python research_code/forecast_transfer.py --source-dir PRIVATE_RAW --history-dir PRIVATE_HISTORY --out PRIVATE_OUTPUT
python research_code/forecast_transfer_predict.py --model-dir PRIVATE_OUTPUT/winter --station HISTORICAL_ID --timestamp KST_DATETIME --weather-dir PRIVATE_RAW --kind weather_forecast_encoder
```
