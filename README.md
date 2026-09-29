# KT-X-AI: 대전 공영자전거 '타슈(Tashu)' 300m Super-Station 기반 AI 재배치 및 베이지안 스마트 관제 시스템

본 레포지토리는 대전광역시 공영자전거 '타슈' 1,200여 개 대여소의 911만 건 통행 이력과 실시간 API 스냅샷 데이터를 바탕으로 구축된 **차세대 AI 수요 예측, 비대칭 품절 방지 재배치 최적화, 차량 동선 최적화, 그리고 베이지안 고장 감지 시스템**의 전체 소스코드와 파이프라인입니다.

---

## 📌 핵심 연구 및 개발 성과 요약 (Executive Highlights)

| 구분 | 기존 방식 (대여소 단위 MSE 예측) | 최종 혁신 솔루션 (300m Q85 슈퍼스테이션) | 현장 개선 효과 |
| :--- | :--- | :--- | :--- |
| **공간 단위** | 1,200여 개 개별 대여소 (50~200m 초근접 분산) | **460개 300m 생활권 Super-Station** | 데이터 희소성(Sparsity) 해소, 앵커 거점 도보 대체성 확보 |
| **네트워크 특성** | 대여소 자체의 시계열 래그만 고려 | **OD 통행 87.4% 유출입(`lag1_ret`) 피처 반영** | 피처 중요도 53.41% 확보, 유입 회랑(Corridor) 선제 파악 |
| **손실 함수** | MSE (평균제곱오차: $\pm 2$대 동일 취급) | **Quantile 85% 비대칭 손실 ($\tau=0.85$)** | 부족 오차에 5.67배 높은 벌점 부여, 피크 수요 안전 버퍼 자동 확보 |
| **품절률 (Top 20)** | **38.0% 품절 방치** (10회 중 4회 0대) | **13.3%로 급감** (결품 위험 65% 차단) | **월간 83,565명의 결품 잠재 수요 완벽 방어** |
| **피크 재배치 효과** | 외곽 0대 대여소 헛걸음 배차 (구제 인원 2명) | **핵심 회랑 핀포인트 배차 (퇴근 1시간 60~73명 구제)** | **핵심 거점 결품 방어율 88.8%, 유효 회전율 30.2% 달성** |
| **트럭 동선 최적화** | 탁상공론식 다중 정류장 순회 | **[상차 1곳 → 하차 1~2곳] 물리적 골든타임 동선** | **평균 7.5km 주행, 48.5분 만에 176대 완벽 하차** |
| **고장 자전거 감지** | 시민 민원 접수 후 뒤늦은 수거 | **API 스냅샷 기반 포아송 베이지안 추론** | **자전거 ID 없이도 1대 고착 대여소 99.9% 특정** |

---

## 🛠️ 시스템 아키텍처 (System Architecture)

```
[1. 데이터 수집 및 공간 클러스터링]
  - 911만 건 통행 로그 (trips.parquet) + 기상청 기상 + OSM 도로망
  - 300m 생활권 제약 군집화 (1,200개 대여소 → 460개 Super-Station 압축)

[2. 머신러닝 예측 및 비대칭 목표 설정]
  - 87.4% 타 군집 이동 네트워크 O/D 회랑 유입량(lag1_ret) 피처 주입
  - Native Quantile 85% Loss XGBoost (부족 오차에 5.67배 페널티)
  - 시간대별 피크 결품 위험 거점 및 배차 목표 수량 산출

[3. 물리 제약 기반 트럭 10대 동선 최적화 (Fleet Routing)]
  - 정차 3분, 상차 45초/대, 하차 30초/대, 퇴근길 시내 22km/h 속도 제약
  - 1-ton 트럭 10대: [상차 1곳 → 하차 1~2곳] 48.5분 만에 176대 핀포인트 하차

[4. 실시간 API 스냅샷 기반 베이지안 관제 (Online Serving)]
  - 공공 API 재고 차분(ΔS)을 통한 실시간 대여량/유입량 역산 복원
  - 포아송 베이지안 생존 분석으로 1대 고착 유령 자전거(Ghost Bike) 99.9% 적발 및 유효 재고 차감

[5. 미래 도시교통 확장 분석]
  - 자전거 도로 단절 및 최고 위험 구간 Top 10 도출 (OSM 오버레이)
  - 대전시 ESG 탄소 감축량 (701.8톤) 및 지하철 1호선 환승 편익 정량화
  - 2028년 개통 예정 대전 도시철도 2호선 트램(38.8km) 연계 시너지 분석 (전체 통행의 42.4% 점유)
```

---

## 📂 폴더 및 파일 구조 (Directory Structure)

```
KT-X-AI/
├── scripts/                                  # 프로덕션 실행 및 분석 스크립트
│   ├── build_final_model.py                  # 300m 군집화 및 XGBoost 최종 모델 학습
│   ├── calculate_exact_dispatch_manifest.py  # 2, 3, 9월 실전 배차 지시서 생성
│   ├── evaluate_final_metrics.py             # 3계층 최종 평가 지표 (Pinball, Coverage, Turnover)
│   ├── optimize_fleet_routing.py             # 상하차 시간 제약 기반 트럭 10대 동선 최적화
│   ├── bayesian_broken_bikes.py              # 통행 이력 기반 베이지안 튕김(Bounce) 고장 감지
│   ├── simulate_september_api_pipeline.py    # API 스냅샷 기반 1대 고착 포아송 추론 및 배차
│   ├── calculate_esg_and_subway_feeder.py    # ESG 탄소 감축량 및 지하철 1호선 연계 분석
│   ├── analyze_cycleway_safety_and_gaps.py   # 자전거 도로 단절 위험 구간 Top 10 도출
│   └── analyze_tram_line2_synergy.py         # 대전 트램 2호선(38.8km) 연계 수요 분석
│
├── src/                                      # 전처리, 피처 엔지니어링, 베이스라인 모듈
│   ├── preprocess.py                         # 원본 데이터 전처리 및 클렌징
│   ├── features.py                           # 시계열 래그 및 공간 피처 엔지니어링
│   ├── spatial_features.py                   # 공간적 O/D 매트릭스 계산
│   └── common.py                             # 공통 유틸리티 및 메트릭
│
├── outputs/                                  # 모델 가중치 및 핵심 결과 요약
│   ├── final_model/                          # 최종 모델 (xgb_cluster_300m_q85.json, top_partners.json)
│   └── report_figs/                          # EDA 및 분석 시각화 차트
│
├── data/processed/
│   └── cluster_mapping_300m.csv              # 460개 생활권 Super-Station 매핑 테이블
│
├── requirements.txt                          # 의존성 패키지 목록
└── .gitignore                                # 대용량 바이너리/데이터 제외 규칙
```

---

## 🚀 빠른 시작 (Quick Start)

### 1. 환경 설정
```bash
git clone https://github.com/sijoon-sung/KT-X-AI-.git
cd KT-X-AI-
git checkout 성시준
pip install -r requirements.txt
```

### 2. 주요 스크립트 실행
```bash
# 1. 300m 생활권 Super-Station 기반 Quantile 85% 모델 학습
python scripts/build_final_model.py

# 2. 계절별(2, 3, 9월) 17:00 퇴근길 실전 배차 지시서 산출
python scripts/calculate_exact_dispatch_manifest.py

# 3. 1-ton 트럭 10대 상하차 시간 및 도로 주행 동선 최적화
python scripts/optimize_fleet_routing.py

# 4. 실시간 API 스냅샷 기반 고장 유령 자전거 감지 및 배차 파이프라인
python scripts/simulate_september_api_pipeline.py

# 5. ESG 탄소 감축량 및 지하철 1호선 연계 편익 산출
python scripts/calculate_esg_and_subway_feeder.py

# 6. 자전거 전용도로 단절 위험 구간 Top 10 도출
python scripts/analyze_cycleway_safety_and_gaps.py

# 7. 대전 도시철도 2호선 트램(Tram) 연계 수요 분석
python scripts/analyze_tram_line2_synergy.py
```

---

## 📊 주요 실증 결과 요약

### ① 17:00 퇴근 피크 배차 효과 (Before vs After)
* **집중 투하 5대 거점 결품 피해**: 82명 $\rightarrow$ **9명 (88.8% 결품 차단 성공)**
* **퇴근길 1시간 즉시 구제 인원**: **73명 즉시 구제** (일일 출·퇴근 150명, 월간 4,500명 헛걸음 해소)

### ② 트럭 10대 동선 최적화 실측치
* **평균 이동 거리**: **7.5 km** (최소 유류비 동선)
* **평균 작업+주행 시간**: **48.5분** (퇴근 60분 골든타임 이내 100% 안착)
* **총 재배치 물량**: **176대 완벽 하차**

### ③ API 스냅샷 기반 베이지안 고장 자전거 감지
* 시간당 수요 4~8대/h 고수요 거점에서 1대 고착 시: **포아송 사후 고장 확률 99.9% 확정**
* 자전거 ID가 없는 공공 API에서도 **혼자 서 있는 고장 유령 자전거를 핀포인트 특정하여 수거**

### ④ ESG 탄소 감축 및 대중교통 연계
* **이산화탄소 순 감축량**: **701.8 톤 (ton $\text{CO}_2$)** (소나무 106,337 그루 식재 효과)
* **시민 유류비 절감액**: **약 6.89억 원**
* **지하철 1호선 연계**: 전체 통행의 **19.37% (176.6만 건)**가 지하철 1호선과 직결
* **트램 2호선(2028년 개통) 연계**: 전체 타슈 이용의 **42.4% (386.3만 건, 연간 232만 건)**가 트램 노선 축선에서 발생

---

## 👥 Authors & Contributors
* **성시준 (Sijun Sung)** - KT-X-AI Project Lead & Modeling Engineer
* **Google Deepmind Antigravity** - AI Pair-Programming Assistant