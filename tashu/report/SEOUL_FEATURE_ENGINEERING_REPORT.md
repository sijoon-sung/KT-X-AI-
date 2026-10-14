# 서울 따릉이 빅데이터를 대전 타슈의 파운데이션 피처(Feature)로 가공한 체계 보고서

본 보고서는 멘토님의 핵심 디렉션인 **'서울 따릉이의 거대한 데이터 볼륨을 타슈 모델의 입력 피처(Feature Representation)로 변환하라'**는 과제를 수리적·공학적으로 완벽히 구현하여 구축한 피처 엔지니어링 결과이다.

## 1. 피처 가공의 핵심 철학 (Why Big Data as Feature?)

- **타슈의 한계 (Small Data Overfitting)**: 대전 데이터는 1.5개년으로 짧아 기상 이변이나 특정 요일/시간 조합의 표본 수가 적어 분산(Variance)이 큼.
- **따릉이의 역할 (Foundation Prior)**: 서울의 8,760시간 전수 데이터와 2,586개소의 극한 환경에서 추출한 통행 법칙을 **'사전확률 및 프로토타입 임베딩'**으로 타슈 모델에 주입하여 정규화(Regularization) 수행.

## 2. 가공된 4단계 서울 파운데이션 피처 체계

산출물: [`data/seoul_derived_tashu_features.npz`](file:///c:/Users/DISLAB/Desktop/KT%20X%20AI/tashu/data/seoul_derived_tashu_features.npz)

### ① 시간 축 외생 피처: `temporal_features` (T=14,616시간 x 2차원)
1. **`seoul_macro_intensity_prior`**: 서울 8,760시간에서 학습된 신경망이 추정한 시각 $t$의 전국 공통 기저 활동 강도 [0, 1].
2. **`seoul_rain_suppression_factor`**: 서울 데이터에서 실측된 우천 시 수요 감쇄율 (평시 대비 26.8% 수준으로 급락하는 물리적 감쇄 계수).

### ② 공간 축 대여소 프로토타입 임베딩 피처: `station_features` (N=1,425소 x 6차원)
서울 2,586개소에서 추출한 5대 극한 프로토타입과의 **소프트 코사인 유사도(Soft-Attention Affinity)**:
1. **`proto0_transit_hub_affinity`**: 서울 초역세권(여의도/강남역) 출퇴근 첨두 프로토타입과의 유사도.
2. **`proto1_residential_drain_affinity`**: 서울 외곽 대형 베드타운(노원/강서) 아침 순유출 프로토타입과의 유사도.
3. **`proto2_youth_commercial_affinity`**: 홍대·신촌 주야간 지속 상업/대학 순환 프로토타입과의 유사도 (KAIST/충남대 매핑).
4. **`proto3_river_leisure_affinity`**: 한강 수변 레저 주말형 프로토타입과의 유사도 (갑천/유등천 매핑).
5. **`proto4_neighborhood_affinity`**: 주택가 골목길 완만 분산 프로토타입과의 유사도.
6. **`seoul_fitted_transit_feeder_index`**: 서울 589개 전철역에서 실측 피팅된 도보 거리 감쇠 함수 $e^{-d/600\text{m}}$ 기반 타슈 대여소별 철도 결합도.

## 3. 멘토 미팅용 논리 구성 요약

1. **"따릉이를 단순히 타슈 옆에 나열해 비교하는 것이 아니라, 따릉이의 방대한 데이터로부터 전국 단위 기저 활동 강도(Temporal Prior)와 5대 도시 프로토타입 사전(Codebook)을 구축했습니다."**
2. **"이를 통해 타슈의 1,425개 대여소마다 '서울의 5대 행동 패턴과 얼마나 닮았는가'를 나타내는 6차원 임베딩 벡터와 매 시간대별 거시 사전확률 피처를 생성하여 GCN 모델의 외생 변수로 결합했습니다."**
3. **"이로써 데이터가 적은 중소도시에서도 대도시의 파운데이션 지식을 사전 피처(Pretrained Feature)로 전이받는 구조가 완성되었습니다."**
