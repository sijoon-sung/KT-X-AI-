# 타슈 공모전: 품절 예측과 대수별 재배치 추천

**실행은 [run.md](run.md)를 기준으로 합니다.** 설치·빠른 실행·모델 크기 비교·실데이터
학습·재고/이동 대수 계산·테스트 명령을 한곳에서 관리하며, 실행 절차 변경 시 함께 갱신합니다.

검증 자료에서 건수 편향과 현재 0대/재고 보유별 GCN·Flow 비중을 조정하는 보정 모델을
함께 평가합니다. 별도 시험 기록으로 재배치 전후를 비교하고, 기존 모델도 재학습 없이
평가할 수 있습니다. 실데이터 점검과 모든 실행은 [run.md 12번](run.md#12-보정대수회복재배치-평가-추가)을 따릅니다.

RTX 3080용 `rtx3080` 설정은 폭 512, GCN 4층, Flow 6블록으로 확대합니다.
CUDA 혼합 정밀도·배치 누적·시나리오 분할을 적용하며 GPU 설치 확인부터 학습까지
모든 명령은 [run.md의 RTX 3080 절차](run.md#0-rtx-3080-확대-실행)에 있습니다.

**현재 대수 → 미래 재고 구간 → 보충 필요 대수 → 어디에서 몇 대를 옮길지**를 검토합니다.
XGBoost, 일반 GCN, GCN + Flow Matching을 같은 입력·시간 분할에서 학습하고 비교합니다.
기존 `tashu/`의 데이터·가중치·성능표는 자동 사용하지 않습니다.
이전 README는 [참고용 기록](docs/legacy_methodology.md)으로 보존했습니다.

> `demo`는 새로 생성한 **합성 데이터 실행 예제**입니다. 실제 타슈 성능이나 재배치
> 효과를 나타내지 않습니다. 실데이터 성능은 독립된 입력과 평가가 필요합니다.

## 설치와 전체 실행

Python 3.11 이상, 3.12 권장. CPU가 기본값이며 CUDA는 선택 사항입니다.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"

# 데이터 생성 → 세 모델 학습 → 저장 모델 재로딩 → 공통 평가 → 대수별 이동 추천
tashu demo --out runs/demo --budget 20 --truck-capacity 10
```

이번에 검증한 CPU 패키지 버전을 그대로 설치하려면 Python 3.12 환경에서
`python -m pip install -r requirements-cpu-py312.txt`를 실행한 뒤
`python -m pip install -e . --no-deps`를 사용하세요.

완료 후 `runs/demo/recommendation/report.html`을 브라우저로 엽니다. 인터넷 없이 볼 수
있습니다. 기존 실행을 보존하므로 재실행은 `runs/demo2` 등 새 폴더를 사용하세요.

```bash
# 빠른 확인
tashu demo --out runs/quick --days 12 --stations 6 --epochs 2 --trees 30 --samples 16 --ode-steps 4
```

## 한 단계 큰 모델

기본 `base`는 은닉 차원 48, `medium`은 96입니다. 그래프 계층 수와 입력·학습 횟수·
XGBoost 트리 수·생성 시나리오 수는 그대로 두고 폭만 키워 효과를 비교합니다.

```bash
tashu demo --profile base --out runs/size_base
tashu demo --profile medium --out runs/size_medium
# 같은 실데이터와 같은 분할로 확대 모델 학습
tashu train --profile medium --panel data/hourly_panel.csv --train-end 2026-02-01 --valid-end 2026-03-01 --out runs/real_medium
```

`--hidden`을 직접 지정하면 profile보다 우선합니다. 기본값은 기존 base를 유지합니다.
새 실행 metadata.json에는 전체/학습/고정 파라미터 수와 학습 시간을 기록하고,
evaluation.json에는 warm-up을 제외한 도시 전체 한 예측 시점의 시나리오 생성 시간을
기록합니다. GCN + Flow Matching 학습 시간은 잔차 모델 단독/기본 GCN 포함을 구분합니다.
품절 Brier·CRPS 등 **검증 구간**의 이득과 실행 비용을 보고 크기를 선택하세요.
합성 데이터에서 커진 모델이 낫더라도 실데이터 성능으로 해석하지 않습니다.

## 실제 데이터로 학습

[입력 규약](docs/DATA_CONTRACT.md)에 맞춘 시간별 CSV를 사용합니다. 이번 구현은 원본
ZIP이나 이전 NPZ를 신뢰한다고 가정해 자동 변환하거나 재고를 임의 복원하지 않습니다.
원본 어댑터는 실제 파일 구조·수집 범위 확인 후 추가할 수 있습니다.

```bash
tashu train --panel data/hourly_panel.csv --train-end 2026-02-01 --valid-end 2026-03-01 --out runs/real01
```

- `train-end` 전만 학습/그래프 생성, 그 이후 `valid-end` 전은 검증, 나머지는 최종 평가입니다.
- 여러 시간의 정답이 분할 경계를 걸치면 해당 예측 창을 제외합니다.
- 검증 품절 Brier로 모델을 선택하고, 측정 재고가 없으면 검증 count CRPS를 사용합니다.
- 기본 예측 길이는 3시간입니다. `--horizon 6` 등으로 바꿀 수 있습니다.
- 메모리 관리를 위해 기본 `--max-training-rows 250000`, 검증/평가 각각
  `--max-eval-origins 96`을 사용합니다. 선택 시각을 전부 기록하며, 전체 평가 시 한도를 늘리세요.
- 입력 SHA-256, 코드 커밋/수정 상태, 시드, 버전, 분할·모델·그래프 설정을 기록합니다.

## 예상 대수와 이동 대수

```bash
tashu recommend --run runs/real01 --panel data/current_panel.csv --at "2026-04-01 08:00:00" --snapshot data/current_stock.csv --budget 40 --truck-capacity 10 --service-level 0.9 --max-distance-km 5 --out runs/real01/plan_0800
```

패널에는 예측 시각까지의 과거 이력과 해당 시각 행이 필요합니다. 해당 시각의
대여/반납 정답은 비워도 됩니다. `--snapshot`은 그 시각의 실측 `station_id,stock`이며
모든 모델 대여소가 한 번씩 있어야 합니다. 패널에 시작 재고가 있으면 생략할 수 있습니다.

```bash
# 검증 선택 모델(기본 auto)
tashu recommend --run runs/demo/run --panel runs/demo/synthetic_panel.csv --out runs/demo/auto_plan
# 특정 모델 비교
tashu recommend --run runs/demo/run --panel runs/demo/synthetic_panel.csv --model gcn_flow --out runs/demo/flow_plan
```

| 결과 | 의미 |
|---|---|
| 현재 대수 | 해당 시각 실측 재고 |
| 미래 예상 대수 | 이동 후 매 시간 재고 중앙값과 10~90% 구간 |
| 목표 대수 | 시나리오의 지정 서비스 수준에 따른 시작 재고 후보 |
| 보충 필요 | 목표보다 부족한 수량 |
| 이동 대수 | 실제 공급처·수요처·예산·적재 한도·거리 조건을 만족한 추천 |
| 남은 보충 필요 | 공급 부족이나 제약으로 채우지 못한 수량 |

**보충 필요 대수와 실행 가능한 이동 대수는 다릅니다.** 자전거를 새로 만들지 않고,
공급처는 자기 목표 대수를 남깁니다. 수거 측 서비스 악화보다 도착 측 개선이 클 때만
옮깁니다. 추천만 내며 외부 시스템에 출동 요청을 보내지 않습니다. 한 배치는 직접
수거→보충 이동이고, 다중 정차 최적 트럭 경로는 아닙니다.

## 결과 파일

```text
runs/demo/
  synthetic_panel.csv
  run/
    metadata.json           # 입력·버전·시드·분할·설정
    graph.npy               # 학습 자료만으로 생성
    evaluation.json         # 검증/시험을 구분한 세 모델 비교
    models/
      xgb_0.json ...
      gcn.pt / flow.pt      # 검증에서 선택한 가중치
      training.json
  recommendation/
    report.html             # 오프라인 보고서
    recommendation.json     # 시간별 예상 대수·전체 추천
    stations.csv            # 현재/목표/보충/위험
    hourly_forecasts.csv    # 대여소·시간별 예상 재고 대수와 대여/반납량
    transfers.csv           # 출발지·도착지·이동 대수
```

## 모델과 0대 처리

- XGBoost: 과거 대여/반납, 측정 재고, 달력, 위치를 사용하는 Poisson 회귀.
- 일반 GCN: 두 공간 전파 계층과 비음수 대여/반납 평균. 시간 의존은 lag 입력으로 반영.
- GCN + Flow Matching: 같은 GCN 인코더·평균을 고정하고 공동 log-count 잔차 시나리오를 학습.
  독립 결합 CFM이며 OT 최적 매칭을 주장하지 않습니다.
- 그래프는 거리와 학습 기간의 시간별 거래 패턴으로 생성합니다.
  `--graph distance` 또는 `--graph identity`로 구성을 비교할 수 있습니다.
- 누락은 0대/0건과 구분합니다. 현재 재고가 있는 곳의 미래 품절과 현재 0대인 곳의 회복을 따로 평가합니다.
- 관측되지 않은 잠재 수요의 정답을 복원했다고 주장하지 않습니다.

자세한 가정은 [모델·운영 설명](docs/MODEL_AND_OPERATIONS.md)을 확인하세요.

## 테스트와 구조

```bash
python -m pytest -q
```

정보 유출, 분할 경계, 누락/0 구분, 비음수·정수·용량·예산·대수 보존, 저장/재로딩,
전체 실행을 검사합니다. 기존 대규모 GPU 벤치마크는 자동 테스트 수집에서 제외합니다.
GitHub Actions에는 Windows/Linux CPU 테스트를 설정했습니다.

```text
src/tashu_contest/   # 새 공식 실행 경로
tests/              # 재현 가능한 작은 검증
docs/               # 입력·모델·운영·브랜치 설명
tashu/              # 이전 실험 기록. 새 코드의 의존성 아님
```

[브랜치 운영](docs/BRANCHES.md): 새 구현은 `contest/rebuild-pipeline`에 모으고 기존
`main`과 개인 브랜치는 검토 전 강제 병합·삭제·히스토리 재작성하지 않습니다.
