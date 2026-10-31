# 연구 질문에 연결한 선행연구 비교

확인일: 2026-10-31. 문헌의 성능을 타슈에서 재현한 결과로 인용하지 않는다.
현재 [자료 계약](training-validation-contract.md)은 과거 이력 학습·API 외부 검증 전용이다.
원문을 읽은 범위와 미확인 부분을 적고, [큰 연구 방향](research-program.md)의 선택 근거로 사용한다.

## 핵심 비교

### 추가 근거: 도시 간 이전과 기상 입력 (H-0011)

- [RegionTrans, IJCAI2019](https://ijcai.org/proceedings/2019/262): 학회 초록과
  [저자 원문](https://arxiv.org/pdf/1802.00386)의 §4.1 지역 상관 매칭, 지역 표현·이전 손실,
  시간/기상 맥락의 역할을 확인했다. 서비스는 도시 격자의 crowd flow이며 타슈 재고가 아니다.
  H11은 같은 좌표계의 대여소 프로필 인코더를 시험한다. 논문의 전체 모델·매칭·점수는 재현하지 않는다.
- [서울 OA-15182 원천 대여이력](https://data.seoul.go.kr/dataList/OA-15182/F/1/datasetView.do):
  공식 설명·2024 ZIP 다운로드·12 CSV 헤더·공공누리 제1유형을 확인했다.
  source 대여소 표현에 사용하고 개인속성·타슈와의 번호 매칭을 사용하지 않는다.
- [NOAA GHCNh](https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly):
  공식 제품 설명과 v1.1.0 문서 원본의 변수 단위·QC·관측기간 코드를 확인했다.
  대전2024~2026 파일을 실제 확보했다. 구 ISD의 2026 부재를 확인해 후속 제품을 사용한다.
  기상은 지연된 관측값과 결측 표시이며 미래 날씨·타슈 재고 API 학습을 의미하지 않는다.

[H-0011](H-0011-cross-city-weather.md)은 이 근거로 다계절 과거 건수를 비교한다.
관측소 대표성·아카이브 수정·운영시 수신 지연·타도시 분포 이동은 검증 한계다.

| 문헌 | 문제·자료·출력 | 실제 확인 범위 | 이번 연구에서 확인할 질문 |
|---|---|---|---|
| [Gast 외, CIKM 2015](https://www.pure.ed.ac.uk/ws/portalfiles/portal/23608488/Gast_et_al_2015_Probablistic_Forecasts.pdf), DOI 10.1145/2806416.2806569 | Vélib 가용성의 시간 비동질 큐 모형과 확률 예측, scoring-rule 평가 | 새 저자 PDF 경로로 §2.2~3.1 재확인. 대여/반납 Poisson 과정·유한용량·현재 상태 조건부 예측·비어있지/가득차지 않은 시간의 사건률 추정·대여소 독립 근사의 한계 확인 | 과거 재고가 없는 우리 자료에서 노출시간 보정을 재현할 수 없음. 완료 거래율·상한 없는 과정 근사의 검증을 H-0005로 분리 |
| [Feng 외, 2017](https://www.pure.ed.ac.uk/ws/portalfiles/portal/44197453/peva_bike.pdf) | 현재 상태와 역사적 이용 자료로 미래 재고 분포를 근사하는 과정 모형 | 저자 공개 원고 초록·방법 개요 확인. 구현·자료 재현 미완료 | 과정 기반 접근에서 어떤 사건률과 관계 자료가 필요한가? API만 있을 때 어디까지 식별되는가? |
| [Lin·He·Peeta, TRC 2018](https://arxiv.org/pdf/1712.04997), DOI 10.1016/j.trc.2018.10.011 | Citi Bike 거래 기반 시간당 대여량, 여러 사전 그래프와 학습 필터·시간 블록 비교 | 저자 원고 §3~4의 관계 정의·DDGF·학습 구간의 관계 생성과 순차 분할 재확인 | 거리는 가능한 관계 중 하나다. 9월에 없는 최근 거래 입력 없이 공간 정보의 추가 가치를 어떻게 대조할까? |
| [Wu 외, Graph WaveNet, IJCAI 2019](https://arxiv.org/pdf/1906.00121) | 교통 시계열, 사전 관계와 학습 의존 행렬·시간 모형 | 저자 원고의 self-adaptive adjacency 설명과 Table 3 비교 확인 | 학습 그래프가 자기 이력·공통 정보보다 추가 가치를 주는가? 관계의 의미·방향·안정성은 무엇인가? |
| [Salinas 외, DeepAR, arXiv v3 2019](https://arxiv.org/pdf/1704.04110) | 관련 시계열의 조건부 확률 예측, 정수 자료에 음이항 likelihood | §3.1의 분포·평균/분산 모수화와 자기회귀 표본 생성 설명 확인 | 시간 인코더와 관측 분포를 나눠 설계할 수 있는가? 재고의 조건부 분산에 어떤 계수형 분포가 맞는가? |
| [Shmueli 외, JRSS C 2005](https://doi.org/10.1111/j.1467-9876.2005.00474.x) | 과산포·저산포를 표현하는 Conway–Maxwell–Poisson 계수형 분포 | [대학 보관 출판 PDF](https://ics.uci.edu/~johnsong/papers/Shmueli%20et.%20al%20-%20A%20Useful%20Distribution%20for%20Fitting%20Discrete%20Data%20%28revival%20of%20the%20Conway-Maxwell-Poisson%20distribution%29.pdf)의 요약·서론·§2에서 분포 정의와 분산 유연성 확인. 타슈 적합·구현 재현 없음 | 과거 학습 이력의 건수 분포와 관측 범위에서 비교 이유가 성립하는가? API H-0003으로 선택하지 않음. 정규화·꼬리·계산 비용은? |
| [Gammelli 외, TRC 2020](https://arxiv.org/abs/2001.07402) | 공급에 의해 절단된 잠재 수요의 GP 추정 | 이전 원문 검토에서 censored likelihood와 자전거 모사 절단 실험 확인 | 표시 재고·실측 거래·미충족 수요의 관측 범위를 어떻게 구분할 것인가? |
| [Lipman 외, ICLR 2023](https://arxiv.org/html/2210.02747v2)·[Tong 외, 2024](https://arxiv.org/html/2302.00482v4) | 확률 경로·벡터장 회귀, 독립 결합과 OT 결합 | Lipman §2~3의 확률 경로·벡터장 회귀와 생성 내부 시간 재확인, Tong §3.2.2~3.2.3 확인 | 어떤 사전분포·결합·확률 경로를 쓸 이유가 있는가? 생성 시간과 물리 시간을 혼동하지 않는가? |
| [Kollovieh 외, TSFlow, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/ee1a1ecc92f35702b5c29dad3dc909ea-Paper-Conference.pdf) | GP와 조건부 사전분포를 이용한 시계열 생성·확률 예측 | §3의 조건부 prior/generative 분해, informed prior 및 App. B.2 설명 확인 | 과거에 조건부인 사전분포가 필요한가? H-0002의 독립 Gaussian 설정과 무엇이 다른가? |
| [Gat 외, Discrete Flow Matching, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/f0d629a734b56a642701bba7bc8bb3ed-Paper-Conference.pdf) | 유한 이산 상태 공간의 확률 경로·표본 생성, 언어 중심 실험 | 서론·§2의 상태 공간과 probability velocity 설명 확인 | 정수 재고를 연속 dequantization과 이산 상태 중 어떻게 표현할까? 상태 범위·큰 대수·순서 정보를 어떻게 다룰까? |
| [Scheuerer·Hamill, 2015](https://doi.org/10.1175/MWR-D-14-00269.1) | 다변량 예측 의존성을 평가하는 variogram 기반 proper score | 출판사 초록·설명 확인, 전체 구현 재현 미완료 | 결합 시나리오의 가치를 주변 Brier 외에 어떻게 검증할까? |
| [Gammelli 외, TRC 2022](https://arxiv.org/abs/2108.00858) | 대여·반납률 예측과 재고 의사결정 성능 연결 | 이전 공개 출판 원문에서 방법·예측/의사결정 비교 확인 | 어떤 출력이 운영 판단에 필요한가? 예측 지표와 정책 효과를 어떻게 나눠 평가할까? |
| [Schuijbroek·Hampshire·van Hoeve 저자 공개 원고](https://www.andrew.cmu.edu/user/vanhoeve/papers/bike_sharing.pdf) | 대여소 서비스 수준의 재고 요구와 차량 경로를 결합, 구역 후 경로 휴리스틱 | 공개 원고 초록·서론 확인. 타슈 구현·성능 재현 없음 | 예측 위험을 수량·차량 제약에 어떻게 연결할까? 유한 거치대·차량 용량 가정을 타슈의 실측 제약과 구분해야 함 |
| [Gneiting, JASA2011 저자 원문](https://arxiv.org/pdf/0912.0902) | 점예측의 손실·요약 통계·Bayes action 일치 | §1.3 식3·Table5에서 제곱손실↔평균·절대손실↔중앙값과 일관성 설명 확인 | 같은분포 점출력의 차이를 H6로 진단. 이론적 최적성과 실제 성능을 구분 |
| [Luo 외, Transportation2026](https://link.springer.com/article/10.1007/s11116-026-10753-7) | 거래와 상태 관측의 불일치로 재배치 추정 | Data and method의 같은기간 거래·10분 상태자료, 식1의 불일치 기반 식별 확인. 경로 최적화·타슈 재현 없음 | 과거 실측재고/동시9월거래가 없는 우리 자료로는 이 방법을 그대로 재현할 수 없음. API잔차를 학습 라벨로 쓰지 않음 |

## 문헌에서 옮겨 올 때 지킬 구분

- 대여량을 demand라고 부르는 논문에서도 그 정답은 실현된 거래일 수 있다. 미충족 수요의 실측 정답과 구분한다.
- 재고 상태의 조건부 분포와 대여·반납 사건의 발생률은 변수와 단위가 다르다.
- 학습 그래프는 예측 의존성의 표현일 수 있다. 실제 이동·수요 대체·인과 효과를 자동으로 입증하지 않는다.
- Flow Matching의 생성 시간, 재고 과정의 실제 시간, 관측 수집 간격을 구분한다.
- 이산 생성의 언어 성능이 순서가 있는 비음수 재고에서의 타당성을 입증하지 않는다.
- 시계열 사전분포와 관측 분포를 바꾸는 것은 모델링 가정 변경이다. 동결된 가설 안에서 사후 조정하지 않는다.
- 점 예측·주변 분포·결합 분포·운영 손실의 평가 대상을 맞춰 비교한다.

## 현재의 해석과 아직 할 일

위 연구들은 설계 공간의 근거다. 타슈를 위한 최종 조합이나 신규성을 확정하지 않는다.
계수형 시계열의 더 넓은 계열(INAR 등), 경계가 있는 재고 과정, 이산·연속 결합 예측의 가까운 대조군은
추가 원문 검토 대상이다. 이번 문서에서 확인하지 못한 정리·수식·성능을 읽은 것처럼 인용하지 않는다.

후속 문헌 카드는 최소한 질문, 관측 자료, 정답, 시간 범위, 가정, 출력 분포, 평가,
코드/자료 접근, 확인 범위, 우리 자료에 적용되지 않는 조건을 같은 형식으로 채운다.
이 비교가 다음 자료 진단과 모형군 선택의 근거가 된다.

## 과거 API 진단의 현재 위치

[H-0003 결과](H-0003-results.md)는 개발 기간의 현재 2~10대 적격 층에서 30분 조건부 변동비 중앙값 0.449를 보였다.
이는 과거 API 탐색 기록이며 최신 계약에서는 후속 모형군을 고르는 근거로 사용하지 않는다.
분포·상태 전이·그래프·생성 방식은 과거 학습 이력의 정답과 구조를 근거로 다시 검토한다.
CMP의 문헌상 유연성이 타슈의 예측 성능 우위를 보장하지 않으며, 이 진단은 해당 모형의 채택 실험이 아니다.
Gast 원문 심화 확인과 [H-0005 결과](H-0005-results.md)를 연결했다. 노출시간 보정·물리 용량을 재현하지 않은 근사의 별도 실험이다.
그 결과는 GCN과 Flow Matching의 채택 근거가 아니며 두 선택은 여전히 열려 있다.

## 관리자 운영 근거와 의사결정 검증

[관리자 재배치 기획](operator-rebalancing-plan.md)에 2024년 대전시 정비센터 개소와 확대 계획,
기존 AI 배치 사업 보도, 2026년 출근 후 지역별 집중·부족 보도, 공식 지정 대여소 안내를 연결했다.
2026년 보험 대상 차량 14대는 재게시본에서 확인한 잠정 자료이며 나라장터 원문 첨부는 시스템 점검으로 대조하지 못했다.
실가동 재배치 차량 수·적재량·거점으로 확정하지 않는다. 예측 지표와 실제 운영 개선은 별도 평가다.


## H-0013 문헌과 적용 경계

### 가설

완료 거래의 확률 예측과 관측 재고 연결, 관리자 행동을 다른 검증 대상으로 둔다.
다음 원문은 설계 근거이며 이번 설정의 성능 우위를 보장하는 자료가 아니다.

### 실험

- [Lipman 외, Flow Matching](https://arxiv.org/html/2210.02747v2): 확인 범위는 원문§3의 조건부 벡터장 회귀다.
  질문은 과거 완료 대여/반납2D log 잔차 분포를 생성할 수 있는가다. 원문의 일반 연속 생성과 우리의10분 계수 정답은 다르다.
  독립Gaussian coupling·dequantization·역변환을 명시하고 과거 flow loss/energy와 별도 재고 fairCRPS로 평가한다.
  구현은 저장소 `rolling_inventory_model.py`·`scaled_flow_core.py`다. 도시 공동 보존·물리 재배치나 신규성을 입증한 재현으로 부르지 않는다.
- [Ferro, Fair scores for ensemble forecasts](https://empslocal.ex.ac.uk/people/staff/ferro/Publications/ferro2013.pdf): 저자 원문 초록·독립 ensemble 표본 해석을 확인했다.
  질문은 유한128입자와 pointmass를 비교할 때 표본 수 영향을 어떻게 구분하는가다. 일반 예측/검증값을 다루며 특정 교통 예측시간을 정하는 논문이 아니다.
  IID seed-mixture 가정 아래 M(M−1) 보정 fairCRPS를 주지표로, 일반CRPS·fairBrier를 보조로 사용한다.
  저장소 `rolling_inventory_metrics.py`의 pairwise 등가 검사를 제공한다. 이 보정은 확률 보정 완료나 독립시험을 보장하지 않는다.
- [Liang 외, RL bike rebalancing](https://www.cirrelt.ca/documentstravail/cirrelt-2024-04.pdf): 확인 범위는 초록의 운영 MDP·행동·보상·시뮬레이터다.
  질문은 예측 상태를 실제 수량/이동 행동으로 연결하는 데 무엇이 필요한가다. 이번 자료에는 미래 재고와 과거 완료 거래만 있고 action/cost 정답이 없다.
  논문의 상세 simulator·실험 성능·시간 설정을 재현했다고 하지 않는다. 이번 구현은 상태예측·로그 필드·관리자 참고목록이며 정책/운영 보상 평가는 미실행이다.

### 결과

[H-0013](H-0013-rolling-api-flow.md)의 재사용9월에서 CFM 추가이득 주가설은 지지하지 못했다.
서울/GCN 단독효과는 이번 대조에서 분리하지 않았으며 CFM/Poisson은 head·목적함수 묶음 대조다.
후속 설계의 분포·정규화·선택 기준은 과거 내부 정답으로 새 계획에서 정한다. API 관측 통계를 모형군 선택 근거로 재사용하지 않는다.


## H-0014 · 결합 역할을 검증하는 문헌 연결

### 가설

FM의 역할을 대여소 간 공동 시나리오로 정의하고 개별 분포와 추가 효과를 분리한다.

### 실험

[설계 문서](H-0014-dependence-flow-design.md)와 [원문별 확인 범위](sources/H-0014-flow-matching-literature.json)에
ECC·저차원 Gaussian copula·Variogram·GP-prior FM·Count-FM을 정리했다. 원문 설정/확인 범위와 우리 제안을 구분한다.
특히 단변량 GP-prior 연구의 결과를 대여소 그래프의 개선 증거로 쓰지 않고, Count-FM의 생성 시간을 실제 자전거 이동시간으로 바꾸지 않는다.

### 결과

설계 근거를 확보한 단계다. 타슈 새 성능 실험·현장 비용절감·정책 학습은 미실행이며 신규성도 확정하지 않는다.
