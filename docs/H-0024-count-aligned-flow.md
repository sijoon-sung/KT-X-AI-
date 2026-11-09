# H-0024: 작은 encoder와 건수 정합 Flow Matching

## 가설

기상을 유지하고 불필요한 encoder 경로를 줄인 뒤, 포아송을 시작 분포로 사용하는 FM이 포아송 단독보다 건수 분포를 개선하는지 검증한다. 기존 연속 FM의 정수화·손실 선택 문제를 분리한다. 무보정 선택은 FM의 개선으로 세지 않는다.

## 실험

- Encoder: H13 재학습, 작은 지역 MLP, 작은 GCN, 작은 주간 임베딩 MLP. 기존 기상·달력·흐름 입력을 유지하며 새 특징은 넣지 않는다.
- Decoder: 기존 Gaussian FM의 손실 선택, 같은 궤적의 건수 energy 선택, 포아송에서 출발하는 연속 FM, 정수 birth-death count-FM.
- Encoder 선택은 2월 전반, decoder 선택은 2월 후반, 완료 거래 비교는 이미 노출된 3월로 분리한다. 선택을 잠근 뒤 평가한다. API는 열지 않는다.
- [count-FM 원문](https://arxiv.org/html/2605.07746v3)의 binomial bridge·rate matching·정수 sampler를 확인해 조건부 포아송 출발 분포에 적용한다. OT·CFG를 구현한 것으로 설명하지 않는다.
- 비교 예산·수식·통계 판정은 [계획](../research/hypotheses/H-0024.json)에 동결하고 연산은 dislab에서 수행한다.

## 결과

계획 시점에는 미실행. Draft PR과 동결 계획을 실제 게시한 뒤 실행한다. 배포 모델은 자동 교체하지 않는다.
