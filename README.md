# 타슈 관측 기반 예측 연구

**[확대·따릉이·Flow Matching 연구 모델](docs/H-0009-scaled-seoul-flow.md)을 실제 학습하고 시제품에 추가했습니다.**
큰GCN75,394개+CFM50,818개/시드·2시드, 과거30분 거래 학습·2월 선택·9월API평가입니다.
공식 재고 CRPS 최소 대조군 상대 감소 -0.371%, 사전 판정 `not_supported`. 서비스 연결과 성능 판정을 구분합니다.

**공모전용 예측 시제품 [타슈다음](docs/prediction-system.md)을 구현했습니다.**
대여소·현재 대수·시각으로30분 뒤 재고·0대 확률·예측 범위를 확인합니다.
GCN을 시연 기본 경로로 사용하고 달력 참고출력을 함께 제공합니다.
[H-0008 인수검증](docs/H-0008-prediction-system.md):64/64 고정출력 재현, 서비스8개·프로토콜14개·HTTP5개 검사 통과.
9월기록재생·수동입력·현재스냅샷과JSON내려받기를지원합니다. 연산은dislab입니다.

**현재 자료 계약: 과거 대여·반납 기록으로 학습·내부 검증하고, 타슈 API는 외부 검증에만 사용합니다.**
**공식 재고·품절 평가는 2026년 9월 API에 고정합니다.** [평가 대상과 기간](docs/september-evaluation-contract.md)을 따릅니다.
2·3월에는 실측 재고 정답이 없습니다. 그 기간의 거래 건수 비교를 프로젝트의 재고 성능으로 보고하지 않습니다.
[자료 사용 계약](docs/training-validation-contract.md)과 [과거 원본 감사](docs/historical-source-audit.md)를 우선합니다.
아래 H-0001~H-0003은 이전 API 개발·탐색 기록입니다. 과거 이력 학습→API 검증을 수행한 결과로 승계하지 않습니다.

[H-0004 보조 거래 건수 학습](docs/H-0004-results.md)은 dislab에서 완료했습니다.
대여소 1,192개의 두 평균 예측 모델을 학습·저장하고 재로딩을 확인했습니다.
3월 내부 보류에서 요일·시간대 평균은 고정 평균보다 Poisson deviance를 11.35% 줄였습니다.
이 결과는 기록 대여·반납 건수의 보조 평균 기준선입니다.

[H-0005 공식 9월 평가](docs/H-0005-results.md)에서는 고정한 과거 흐름과 현재 API 시작 재고로 30분 뒤 상태를 예측했습니다.
현재 양수 131,118쌍의 0대 Brier는 0.149934→0.119369로 사전 가설을 지지했습니다.
전체 재고 수량 MAE는 0.579567→0.632669대로 악화했습니다. API 학습·튜닝·보정은 없었습니다.
[아키텍처 아이디어별 실험](docs/architecture-ideas-to-experiments.md)에서 GCN·사건 과정·FM의 역할을 분리하고 점진적으로 비교합니다.

[H-0006 원인 진단](docs/H-0006-results.md)에서는 같은분포 중앙값으로 MAE가0.576346대까지 낮아졌습니다.
현재 유지0.579567대 대비 차이의 두구간은0을 포함합니다. 평균은 RMSE에서 현재 유지보다 낮았습니다.
[원인별 개선 계획](docs/failure-analysis-and-improvement-plan.md)에 손실/점출력·정보·공간·과정·결합 생성 문제를 나눴습니다.

[H-0007 OD-GCN 비교](docs/H-0007-results.md)는 dislab GPU에서 세관계·각3시드 학습 뒤 공식9월 평가를 완료했습니다.
현재양수 Brier는 OD0.120347·자기0.120283·치환0.120387·고정달력0.119369였습니다.
사전 공간 추가가치 가설은 지지하지 않았습니다. 한종류 정적관계·가용입력 조건의 결과이며 GCN전체나FM의 기각이 아닙니다.

현재 연구 방향은 [큰 질문에서 모형군으로 좁혀 가는 연구 설계](docs/research-program.md)에서 정리합니다.
[선행연구 비교](docs/research-evidence-map.md)와 [결정 기록](docs/research-decisions.md)을 근거로
문제·관측 과정·의존성·확률 표현을 먼저 정하며, 최종 아키텍처는 열어 둡니다.

새 연구는 [가설별 Draft PR 진행 체계](docs/research-workflow.md)를 따릅니다.
[연구 기록](research/README.md)에서 계획·실행·판단을 추적합니다. 아래 내용은 기존 연구 기록이며 새 연구의 재현 검증을 대신하지 않습니다.

새 연구의 출발점은 [프로젝트 문제의식과 해결 계획](docs/project-charter.md)입니다.
첫 가설 [H-0001: 관측 간격과 0대 비율](docs/H-0001-observation-resolution.md)은 완료했습니다.
관측 0대 비율의 평균 절대차는 1.24%p로, 사전 기준 5%p 초과 가설을 지지하지 않았습니다.

[H-0002 아키텍처 비교](docs/H-0002-architecture-results.md)는 dislab RTX 3080에서 세 시드로 완료했습니다.
현재 재고 양수인 표본의 미래 관측 0대 Brier는 자기 이력 0.079564, 좌표 그래프 직접 예측 0.078826,
그래프 CFM 0.082581이었습니다. 이번 소형 재고 적응 CFM의 추가 가치 가설은 지지하지 않았습니다.
기존 전체 모델·정책 효과의 검증으로 확대하지 않습니다. 후속 연구·학습 연산은 dislab에서 수행합니다.

[H-0003 자료 구조 진단](docs/H-0003-results.md)은 dislab CPU에서 완료했습니다.
같은 대여소·같은 현재 2~10대에서 30분 뒤 재고의 분산/평균 비율 중앙값은 0.449였습니다.
이미 API를 탐색한 진단 기록으로 보존하며, 후속 학습의 분포·아키텍처 선택 근거에서는 제외합니다.

## Spatio-Temporal Continuous Flow Matching for Urban Micro-Mobility & Transit Policy Interventions

This repository contains the core methodology implementation of **Multimodal Transit Continuous Flow Matching (Transit-CFM)**, a graph-informed continuous generative framework designed for city-wide bike-sharing demand modeling and counterfactual transit policy simulations.

---

### 1. Proposed Methodology (`tashu/multimodal_transit_flow_matching.py`)

The proposed method models joint spatial velocity fields along Optimal Transport (OT) displacement trajectories, conditioned on multimodal transit connectivity and meteorological forcing:

1. **Optimal Transport Continuous Flow Matching (OT-CFM):**
   - Straight displacement paths: $\mathbf{x}_t = (1 - t)\mathbf{x}_0 + t\mathbf{x}_1$ for $t \in [0, 1]$.
   - Target velocity field: $\mathbf{u}_t = \mathbf{x}_1 - \mathbf{x}_0$.
   - Objective: $\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1} \| v_\theta(\mathbf{x}_t, t, \mathbf{C}, \tilde{\mathbf{A}}) - (\mathbf{x}_1 - \mathbf{x}_0) \|^2$.

2. **Multimodal Transit & External Conditioning ($\mathbf{C} \in \mathbb{R}^{B \times N \times 36}$):**
   - **Metro Line 1 Coupling (22 Stations):** Geodesic distance decay $e^{-d / 500\text{m}}$ and commuter rush waves (08h inflow / 18h outflow).
   - **Bus Transit Coupling (159 Hubs):** Last-mile feeder index $e^{-d / 300\text{m}}$ and transfer wave intensity.
   - **6-Class Urban Functional Typologies:** One-hot embeddings for Subway Hub, Bus Transit, Residential Origin, Campus/Research, Civic/Commercial, and Park/Leisure.
   - **Hourly Dynamic Weather:** Real-time temperature, precipitation, wind speed, relative humidity, and rain-transit suppression interactions.

3. **High-Capacity Residual GCN Backbone:**
   - 6 Residual GCN Blocks with LayerNorm and 2x FFN Expansion (dim 512).
   - Continuous sinusoidal time embedding with per-block AdaLN/FiLM modulation.
   - Dense Jumping Knowledge (JK) multi-scale skip connections.

4. **Numerical ODE Inference & What-If Policy Engine:**
   - Supports Euler, Midpoint (Runge-Kutta 2nd order), and Classical RK4 solvers.
   - Solves forward and counterfactual trajectories: $\hat{\mathbf{x}}_1 = \mathbf{x}_0 + \int_0^1 v_\theta(\mathbf{x}_t, t, \mathbf{C}, \tilde{\mathbf{A}}) dt$.

---

### 2. Method Code Structure

```
├── tashu/
│   ├── multimodal_transit_flow_matching.py   # [Main Proposed Method] Full Transit-CFM pipeline
│   ├── flow_matching_st.py                   # [Foundation Method] Scaled ST-CFM architecture
│   └── data/
│       ├── multimodal_transit_benchmark_results.json # Benchmark metrics & policy simulation results
│       └── flow_matching_benchmark_results.json      # Pure ST-CFM benchmark results
├── .gitignore                                # Strict exclusion of *.pt checkpoints & raw datasets
└── README.md                                 # Methodology documentation
```

---

### 3. How to Run the Methodology Code

#### Requirements
- Python >= 3.10
- PyTorch >= 2.0 (CUDA enabled)
- NumPy, SciPy, LightGBM, XGBoost

#### Execution Command
```bash
cd tashu

# Run full 10-epoch training and 2026 held-out evaluation
python multimodal_transit_flow_matching.py --epochs 10 --batch-size 16 --solver midpoint --run-policy-sim

# Fast smoke-test mode (3 epochs, quick verification)
python multimodal_transit_flow_matching.py --fast --run-policy-sim
```

---

### 4. Empirical Evaluation Summary (Held-Out 2026 Data)

| Period | Model | MAE | WAPE | Zero-Demand Pred | Peak MAE ($y \ge 3$) | Rainy Hours MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Feb 2026** (Winter) | Historical Smoothed Mean | 0.5281 | 1.1622 | 0.3731 | 2.0950 | 0.5120 |
| | **Multimodal Transit-CFM (Proposed)** | **0.3395** | **0.7471** | **0.0787** | **2.1458** | **0.1171** |
| **Mar 2026** (Spring) | Historical Smoothed Mean | 0.5264 | 0.9311 | 0.2778 | 2.5899 | 0.5230 |
| | **Multimodal Transit-CFM (Proposed)** | **0.3809** | **0.6738** | **0.0811** | **2.1855** | **0.0568** |

- **Inference Latency:** 12.7 ms per city-wide hour (1,283 stations simultaneously on RTX 3080).
- **Counterfactual Policy Simulation:**
  - *Subway Hub 50% Bike Supply Expansion:* +3.60% station catchment inflow.
  - *Commuter Transit Disruption Shock:* +10.74% subway catchment surge, +11.18% city-wide last-mile absorption.
