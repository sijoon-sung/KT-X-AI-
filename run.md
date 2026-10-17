# 실행 안내 — 모든 실행은 이 문서 기준

앞으로 설치·학습·평가·예측·재배치 추천·테스트 명령은 **이 파일에서 관리합니다.**
실행 옵션이나 절차를 변경하면 이 문서를 함께 갱신합니다.
README는 프로젝트 설명, 이 문서는 실제 실행 순서의 기준입니다.

**Windows PowerShell / 저장소 루트에서 실행합니다.** 필요한 코드 블록만 터미널에
복사하세요. Markdown 파일 자체를 프로그램처럼 실행하는 방식은 아닙니다.
가상환경 활성화 없이 `.venv`의 Python을 직접 호출하므로 터미널을 다시 열어도 같은 방식입니다.

| 하려는 일 | 이동 |
|---|---|
| RTX 3080에서 크게 학습 | [0. RTX 3080](#0-rtx-3080-확대-실행) |
| 처음 설치 | [1. 설치](#1-설치-최초-1회) |
| 코드가 도는지 빠르게 확인 | [2. 빠른 확인](#2-빠른-확인-합성-데이터) |
| 확대 모델까지 전체 시연 | [3. Medium 전체 실행](#3-medium-전체-실행-합성-데이터) |
| 기본형과 확대형 비교 | [4. 크기 비교](#4-base와-medium-비교-합성-데이터) |
| 실제 자료로 세 모델 학습·평가 | [5. 실제 데이터 학습](#5-실제-데이터-학습평가) |
| 예상 재고와 이동 대수 계산 | [6. 재배치 추천](#6-예상-재고-대수와-재배치-추천) |
| 기존 저장 모델로 합성 시연 다시 계산 | [7. 재학습 없이 추천](#7-재학습-없이-합성-예제-추천만-다시-계산) |
| 결과 위치와 옵션 확인 | [8. 결과](#8-결과-파일), [9. 설정](#9-자주-바꾸는-설정) |
| 테스트 | [10. 테스트](#10-테스트와-실행-도움말) |
| Git에 결과 요약 올리기 | [11. 결과 공유](#11-git에-올릴-결과-요약) |

## 0. RTX 3080 확대 실행

현재 브랜치 `contest/model-medium`에서 `base`, `medium`, `large`, `rtx3080`을 모두 선택합니다.
RTX 3080의 10GB/12GB 변형을 고려한 시작 설정이며 실제 사용량은 대여소 수에 따라 달라집니다.
폭뿐 아니라 층수와 학습량도 바꾸므로 기존 medium과의 비교는 **전체 설정 비교**입니다.
성능 향상은 같은 데이터·분할의 검증 점수로 확인해야 합니다.

| 항목 | medium | rtx3080 | 100m (신규) |
|---|---|---|---|
| 은닉 폭 / GCN 층 / Flow 블록 | 96 / 2 / 1 | 512 / 4 / 6 | 1024 / 6 / 18 |
| 총 파라미터 수 | 약 6만 개 | 약 872만 개 | **약 9,996만 개 (딱 100M)** |
| 최대 에포크 / 조기 종료 patience | 15 / 비활성 | 60 / 12 | 60 / 12 |
| 배치 × 누적 | 32 × 1 | 4 × 4 (유효 16) | 4 × 4 (유효 16) |
| 학습률 / dropout | 0.002 / 0 | 0.0005 / 0.1 | 0.0003 / 0.1 |
| XGBoost 트리 / 깊이 / 학습 장치 | 150 / 4 / CPU | 600 / 6 / CUDA | 1000 / 8 / CUDA |
| 시나리오 / ODE 단계 | 32 / 8 | 64 / 12 | 64 / 12 |
| 평가 배치 / 시나리오 묶음 | 8 / 32 | 2 / 4 | 2 / 4 |
| 실측 피크 VRAM (RTX 3080 10GB) | < 0.2 GiB | ~0.54 GiB | **~1.87 GiB (여유 8GB+)** |

### CUDA 환경 확인

**이미 CUDA가 작동하는 `.venv`라면 1번의 CPU 전용 설치를 다시 실행하지 마세요.**
아래는 기존 CUDA PyTorch를 유지하며 프로젝트 의존성을 설치합니다.
새 GPU 환경은 [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서
Windows/Pip/Python/CUDA에 맞는 명령으로 PyTorch를 먼저 설치한 뒤 진행합니다.

```powershell
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
if ($LASTEXITCODE -ne 0) { throw "의존성 설치 실패" }
& .\.venv\Scripts\python.exe -c "import torch; print('torch:',torch.__version__,'CUDA:',torch.version.cuda); assert torch.cuda.is_available(), 'CUDA를 사용할 수 없습니다'; p=torch.cuda.get_device_properties(0); print(p.name, round(p.total_memory/1024**3,2),'GiB')"
if ($LASTEXITCODE -ne 0) { throw "CUDA 설치와 드라이버를 확인하세요." }
```

### 먼저 GPU 짧은 실행

512 폭과 전체 층수는 유지하고 반복 횟수만 줄입니다. 이 검사가 성공하면 전체 학습으로 갑니다.
합성 자료의 실행 확인이며 실제 타슈 성능 검증은 아닙니다.

```powershell
$GpuSmokeRoot = Join-Path "runs" ("rtx3080-smoke-" + (Get-Date -Format "yyyyMMdd-HHmmss-fff"))
& .\.venv\Scripts\python.exe -m tashu_contest.cli demo --profile rtx3080 --out $GpuSmokeRoot --days 35 --stations 64 --epochs 2 --trees 30 --samples 8 --ode-steps 4 --max-eval-origins 8 --seed 42
if ($LASTEXITCODE -ne 0) { throw "GPU 짧은 실행 실패: 오류와 아래 메모리 대응을 확인하세요." }
Write-Host "보고서: $GpuSmokeRoot/recommendation/report.html"
```

### 실제 자료 전체 학습

파일과 분할 날짜를 자기 데이터에 맞게 수정합니다. 각 분할은 history/horizon을 확보할 만큼 길어야 합니다.
`rtx3080`은 CUDA와 혼합 정밀도를 기본으로 사용하며 각 신경망을 최대 60에포크 학습합니다.

```powershell
$PanelFile = "data/hourly_panel.csv"
$TrainEnd = "2026-08-31 23:00:00"
$ValidEnd = "2026-09-15 23:00:00"
if (-not (Test-Path -LiteralPath $PanelFile)) { throw "실제 패널 CSV 경로를 지정하세요." }
$GpuTrainRoot = Join-Path "runs" ("rtx3080-real-" + (Get-Date -Format "yyyyMMdd-HHmmss-fff"))
& .\.venv\Scripts\python.exe -m tashu_contest.cli train --profile rtx3080 --panel $PanelFile --train-end $TrainEnd --valid-end $ValidEnd --out $GpuTrainRoot --max-training-rows 1000000 --seed 42
if ($LASTEXITCODE -ne 0) { throw "학습 실패: 위 오류를 확인하세요." }
Write-Host "평가: $GpuTrainRoot/evaluation.json"
Write-Host "GPU 정보와 실제 설정: $GpuTrainRoot/metadata.json"
```

실데이터 없이 확대 설정으로 전체 과정을 확인하려면 다음 합성 실행을 사용합니다.
이 명령은 128개 대여소·90일이며 실행 시간이 길어질 수 있습니다.

```powershell
$GpuDemoRoot = Join-Path "runs" ("rtx3080-demo-" + (Get-Date -Format "yyyyMMdd-HHmmss-fff"))
& .\.venv\Scripts\python.exe -m tashu_contest.cli demo --profile rtx3080 --out $GpuDemoRoot --days 90 --stations 128 --seed 42
if ($LASTEXITCODE -ne 0) { throw "확대 합성 실행 실패" }
Write-Host "보고서: $GpuDemoRoot/recommendation/report.html"
```

### 100M 파라미터 극대화 실행 (딱 100M, 99.96M)

99,964,940개(GCN 7.39M + Flow 92.57M) 파라미터의 초거대 모델로 학습·평가 및 재배치 추천을 실행합니다.

```powershell
$Gpu100mRoot = Join-Path "runs" ("100m-demo-" + (Get-Date -Format "yyyyMMdd-HHmmss-fff"))
& .\.venv\Scripts\python.exe -m tashu_contest.cli demo --profile 100m --out $Gpu100mRoot --days 90 --stations 128 --seed 42
if ($LASTEXITCODE -ne 0) { throw "100M 실행 실패" }
Write-Host "100M 보고서: $Gpu100mRoot/recommendation/report.html"
```


### 메모리가 부족할 때

- train/demo 명령에 `--batch-size 2 --accumulate 8 --eval-batch-size 1 --sample-chunk 1`을 추가해 새 출력 폴더로 재실행합니다. 유효 학습 배치는 16으로 유지합니다.
- recommend는 `--device cuda --sample-chunk 1`을 사용합니다. 일반 GPU 추천은 6·7번 명령에 `--device cuda --sample-chunk 4`를 추가합니다.
- XGBoost CUDA 지원이 없는 설치에서는 `--xgb-device cpu`를 명시합니다. 신경망의 CUDA 설정은 유지됩니다.
- 수치 문제가 발생하면 `--no-amp`로 float32를 사용합니다. GPU 메모리는 더 필요합니다.
- CPU에서 코드만 확인할 때는 `--device cpu --xgb-device cpu`를 함께 지정합니다. GPU 성능 확인으로 해석하지 않습니다.

혼합 정밀도, 실제 GPU 정보, 에포크별 최대 GPU 할당량, 조기 종료 시점은 결과 파일에 기록됩니다.
현재 개발 환경의 검증은 CPU에서 수행했으며 RTX 3080에서의 최대 메모리·학습 속도를 실측하지 않았습니다.

## 1. 설치 (최초 1회)

이 절은 **CPU 전용** 설치입니다. 기존 CUDA 환경은 0번을 사용합니다.

Python **3.12**를 설치한 환경에서 시작합니다. 아래 첫 명령으로 버전을 확인하세요.
현재 브랜치는 `contest/model-medium`이며 기본형과 확대형 코드를 모두 포함합니다.

```powershell
python --version
git branch --show-current
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-cpu-py312.txt
& .\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

설치 명령이 실패하면 먼저 해당 오류를 해결하고 다음 단계로 진행하세요.
위 설치는 검증한 **CPU 버전**입니다. CUDA 실행에는 사용 중인 GPU/드라이버에 맞는
PyTorch CUDA 설치가 별도로 필요합니다. CPU 환경에 `--device cuda`만 붙여서는 사용할 수 없습니다.

## 2. 빠른 확인 (합성 데이터)

세 모델 학습 → 저장/재로딩 → 검증/평가 → 재고·이동 대수 추천을 짧게 확인합니다.
매번 새 폴더가 생성되며 기존 결과를 덮어쓰지 않습니다.

```powershell
$QuickRun = Join-Path "runs" ("quick-" + (Get-Date -Format "yyyyMMdd-HHmmss-fff"))
& .\.venv\Scripts\python.exe -m tashu_contest.cli demo --out $QuickRun --days 12 --stations 6 --epochs 2 --trees 30 --samples 16 --ode-steps 4
if ($LASTEXITCODE -ne 0) { throw "빠른 확인 실패: 위 오류를 확인하세요." }
Write-Host "보고서: $QuickRun/recommendation/report.html"
```

이 결과는 **합성 데이터 실행 확인**이며 실제 타슈 성능이 아닙니다.

## 3. Medium 전체 실행 (합성 데이터)

은닉 차원 96의 확대형으로 세 모델을 학습·비교합니다. 최종 추천의 `auto`는 검증
점수가 좋은 모델을 선택하므로 GCN + Flow Matching이 항상 선택되지는 않습니다.

```powershell
$DemoRoot = Join-Path "runs" ("medium-demo-" + (Get-Date -Format "yyyyMMdd-HHmmss-fff"))
& .\.venv\Scripts\python.exe -m tashu_contest.cli demo --profile medium --out $DemoRoot --days 35 --stations 12 --epochs 15 --samples 32 --ode-steps 8 --budget 20 --truck-capacity 5
if ($LASTEXITCODE -ne 0) { throw "전체 실행 실패: 위 오류를 확인하세요." }
Write-Host "보고서: $DemoRoot/recommendation/report.html"
Write-Host "모델 비교: $DemoRoot/run/evaluation.json"
```

## 4. Base와 Medium 비교 (합성 데이터)

동일한 데이터 생성 시드, 학습 횟수, 시나리오 수로 실행합니다. 다른 점은 은닉 차원
48/96입니다. 아래 한 블록이 두 실행과 검증 지표 표/CSV까지 만듭니다.

```powershell
$SizeRunRoot = Join-Path "runs" ("size-compare-" + (Get-Date -Format "yyyyMMdd-HHmmss-fff"))
$SizeRows = @()
foreach ($ModelSize in @("base", "medium")) {
    $SizeRun = Join-Path $SizeRunRoot $ModelSize
    & .\.venv\Scripts\python.exe -m tashu_contest.cli demo --profile $ModelSize --out $SizeRun --days 35 --stations 12 --epochs 15 --samples 32 --ode-steps 8 --seed 42 --budget 20 --truck-capacity 5
    if ($LASTEXITCODE -ne 0) { throw "$ModelSize 실행 실패" }
    $SizeMeta = Get-Content (Join-Path $SizeRun "run/metadata.json") -Raw -Encoding UTF8 | ConvertFrom-Json
    $SizeEval = Get-Content (Join-Path $SizeRun "run/evaluation.json") -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($ForecastModel in @("xgboost", "gcn", "gcn_flow")) {
        $ValidationResult = $SizeEval.models.$ForecastModel.validation
        $SizeRows += [PSCustomObject]@{
            Size = $ModelSize
            Model = $ForecastModel
            Width = $SizeMeta.hidden
            MAE = $ValidationResult.counts.mae
            CRPS = $ValidationResult.counts.crps
            StockBrier = $ValidationResult.stock.brier
            InventoryMAE = $ValidationResult.stock.net_inventory_mae
            InferenceMs = $ValidationResult.inference.ms_per_city_origin
        }
    }
}
$SizeRows | Format-Table -AutoSize
$SizeRows | Export-Csv (Join-Path $SizeRunRoot "validation_comparison.csv") -NoTypeInformation -Encoding UTF8
Write-Host "비교 CSV: $SizeRunRoot/validation_comparison.csv"
```

MAE·CRPS·Brier는 낮을수록 좋습니다. 크기 선택에는 **검증 구간**을 사용합니다.
합성 예제/단일 실행의 속도와 오차를 실데이터 성능으로 해석하지 않습니다.

## 5. 실제 데이터 학습·평가

먼저 [입력 규약](docs/DATA_CONTRACT.md)에 맞는 새 CSV를 준비합니다.
`data/hourly_panel.csv`는 사용자가 준비할 경로이며 저장소에 있다고 가정하지 않습니다.
기존 NPZ/가중치/성능표를 자동으로 가져오지 않습니다.
아래 날짜는 예시이므로 **실제 데이터 범위에 맞춰** 수정하세요.

```powershell
$PanelPath = "data/hourly_panel.csv"
$TrainCutoff = "2026-02-01"
$ValidationCutoff = "2026-03-01"
$TrainingRun = Join-Path "runs" ("real-medium-" + (Get-Date -Format "yyyyMMdd-HHmmss-fff"))
if (-not (Test-Path -LiteralPath $PanelPath)) { throw "입력 CSV를 먼저 준비하세요: $PanelPath" }
& .\.venv\Scripts\python.exe -m tashu_contest.cli train --profile medium --panel $PanelPath --train-end $TrainCutoff --valid-end $ValidationCutoff --out $TrainingRun --history 24 --horizon 3 --epochs 15 --samples 32 --ode-steps 8 --device cpu
if ($LASTEXITCODE -ne 0) { throw "학습 실패: 위 오류를 확인하세요." }
Write-Host "저장 모델: $TrainingRun/models"
Write-Host "모델 비교: $TrainingRun/evaluation.json"
```

학습 종료 전 / 검증 종료 전 / 나머지 기간을 분리합니다. 한 번의 train 명령이
XGBoost·일반 GCN·GCN + Flow Matching을 모두 학습하고 평가합니다.
현재 구현에는 별도 `evaluate` 명령이 없으며 평가 결과는 train/demo 실행에 포함됩니다.
측정 재고가 없으면 거래 예측만 평가하며, 품절 정답을 임의 생성하지 않습니다.

## 6. 예상 재고 대수와 재배치 추천

실제 학습을 마친 뒤 실행합니다. `$ModelRun`을 5번에서 출력된 실제 학습 폴더로,
패널·재고 파일·시각을 실제 입력에 맞춰 수정하세요.
이 블록은 저장 모델을 불러오며 다시 학습하지 않습니다.

```powershell
$ModelRun = "runs/real-medium-실제실행폴더"
$CurrentPanel = "data/current_panel.csv"
$CurrentStock = "data/current_stock.csv"
$ForecastAt = "2026-04-01 08:00:00"
$PlanRoot = Join-Path "runs" ("plan-" + (Get-Date -Format "yyyyMMdd-HHmmss-fff"))
foreach ($RequiredFile in @((Join-Path $ModelRun "metadata.json"), $CurrentPanel, $CurrentStock)) {
    if (-not (Test-Path -LiteralPath $RequiredFile)) { throw "경로를 수정하거나 파일을 준비하세요: $RequiredFile" }
}
& .\.venv\Scripts\python.exe -m tashu_contest.cli recommend --run $ModelRun --panel $CurrentPanel --snapshot $CurrentStock --at $ForecastAt --model auto --samples 32 --ode-steps 8 --budget 40 --truck-capacity 10 --service-level 0.9 --max-distance-km 5 --out $PlanRoot
if ($LASTEXITCODE -ne 0) { throw "추천 실패: 위 오류를 확인하세요." }
Write-Host "보고서: $PlanRoot/report.html"
Write-Host "시간별 대수: $PlanRoot/hourly_forecasts.csv"
Write-Host "이동 대수: $PlanRoot/transfers.csv"
```

- 특정 모델을 쓰려면 `--model auto`를 `gcn_flow`, `gcn`, `xgboost`로 바꿉니다.
- 패널에 현재 실측 시작 재고가 있으면 `--snapshot $CurrentStock`을 생략할 수 있습니다.
  이 경우 위 파일 확인 목록에서도 `$CurrentStock`을 제외하세요.
- 현재 시각의 대여/반납 정답은 비워도 됩니다. 해당 시각 행과 과거 이력은 필요합니다.
- 미래 예상 재고, 보충 필요 대수, 실제 이동 가능한 대수는 서로 다른 값입니다.
- 추천은 예측 시작 전에 이동이 끝난다는 가정의 직접 수거→보충 배치이며 자동 출동하지 않습니다.

## 7. 재학습 없이 합성 예제 추천만 다시 계산

3번을 실행한 **같은 터미널**에서 사용할 수 있습니다. 다시 연 터미널이라면 첫 줄의
`$DemoRoot`를 자신이 생성한 medium-demo 폴더로 설정하세요.

```powershell
# 다른 터미널이라면 실제 폴더로 지정: $DemoRoot = "runs/medium-demo-실제실행폴더"
if (-not $DemoRoot) { throw "3번을 실행하거나 DemoRoot를 실제 폴더로 설정하세요." }
$DemoPlanRoot = Join-Path $DemoRoot ("flow-plan-" + (Get-Date -Format "yyyyMMdd-HHmmss-fff"))
& .\.venv\Scripts\python.exe -m tashu_contest.cli recommend --run (Join-Path $DemoRoot "run") --panel (Join-Path $DemoRoot "synthetic_panel.csv") --model gcn_flow --budget 20 --truck-capacity 5 --out $DemoPlanRoot
if ($LASTEXITCODE -ne 0) { throw "추천 실패: 위 오류를 확인하세요." }
Write-Host "보고서: $DemoPlanRoot/report.html"
```

## 8. 결과 파일

보고서는 출력된 `report.html`을 직접 열어 봅니다. 외부 서버 없이 사용할 수 있습니다.

| 파일 | 확인할 내용 |
|---|---|
| `metadata.json` | 입력 해시·시드·분할·설정·파라미터 수·학습 시간 |
| `evaluation.json` | 세 모델 검증/시험 점수와 시나리오 생성 시간 |
| `models/training.json` | 에포크별 학습/검증 손실 |
| `report.html` | 모델 비교, 대여소별 재고·품절, 이동 배치 |
| `stations.csv` | 현재 대수·목표 대수·보충 필요·이동 후 대수·품절/회복 확률 |
| `hourly_forecasts.csv` | 대여소·시간별 예상 재고 p10/p50/p90와 대여/반납량 |
| `transfers.csv` | 배치별 출발 대여소·도착 대여소·이동 대수·직선거리 |
| `recommendation.json` | 추천 전체 결과 |

train 출력에는 metadata/evaluation/models가 바로 들어갑니다.
demo 출력은 `run/`에 학습 결과, `recommendation/`에 추천 결과를 구분합니다.

## 9. 자주 바꾸는 설정

| 옵션 | 기본값 | 바꾸는 이유 |
|---|---|---|
| `--profile` | `base` (48) | `medium` 96 / `large` 256 / `rtx3080` 512. 학습/demo에 사용 |
| `--hidden` | profile 사용 | 직접 폭 지정. profile보다 우선 |
| `--epochs` | 15 | 학습 반복 횟수 |
| `--history` / `--horizon` | 24 / 3 | 과거 문맥 길이 / 미래 예측 길이 |
| `--samples` / `--ode-steps` | 32 / 8 | 시나리오 수 / Flow Matching 적분 단계 |
| `--graph` | `hybrid` | `distance`, `identity`로 그래프 효과 비교 |
| `--max-training-rows` | 250000 | 학습 시 선택 행 수의 메모리 관리 기준 |
| `--max-eval-origins` | 96 | 검증/시험 각각 선택할 최대 시점 수 |
| `--budget` | 20 | 전체 이동 가능한 자전거 대수 |
| `--truck-capacity` | 10 | 직접 이동 배치당 적재 가능한 대수 |
| `--service-level` | 0.9 | 시나리오 기반 목표 재고 분위수 |
| `--max-distance-km` | 5 | 수거→보충 최대 직선거리 |
| `--device` | `cpu` | CUDA 설치 확인 후 `cuda` 선택 |

위 기본값은 `base` 기준이며 확대 profile은 0번 표처럼 바뀝니다. 명시한 옵션이 profile보다 우선합니다.
`--gcn-layers`, `--flow-blocks`, `--expansion`, `--dropout`으로 구조를 바꿀 수 있습니다.
`--accumulate`는 학습 배치 누적, `--patience`는 조기 종료 대기 에포크(0은 비활성),
`--eval-batch-size`와 `--sample-chunk`는 예측 메모리를 관리합니다.
`--amp`/`--no-amp`는 CUDA 혼합 정밀도, `--xgb-depth`와 `--xgb-device`는 XGBoost 설정입니다.
recommend에서 시나리오 수·ODE 단계·묶음 크기를 생략하면 저장된 실행 설정을 사용합니다.

학습 옵션은 train/demo, 이동 옵션은 recommend/demo에서 사용합니다.
**모델 폭을 바꾸려면 재학습**해야 합니다. recommend에서 profile을 바꿀 수는 없습니다.
큰 자료는 학습 행 수/평가 시점 수 제한을 확인하고 실행하세요. 선택한 시각은 metadata에 기록됩니다.

## 10. 테스트와 실행 도움말

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m tashu_contest.cli --help
& .\.venv\Scripts\python.exe -m tashu_contest.cli train --help
& .\.venv\Scripts\python.exe -m tashu_contest.cli recommend --help
```

CUDA 전용 테스트는 실제 CUDA가 사용 가능할 때 실행되고 CPU 환경에서는 건너뜁니다.
CPU에서 테스트가 통과해도 실제 GPU 검증이 끝났다는 뜻은 아닙니다.

## 11. Git에 올릴 결과 요약

`runs/`는 Git에서 제외됩니다. 실행 스크립트만 올리면 실제 점수는 확인할 수 없습니다.
학습 완료 후 아래처럼 작은 요약만 `results/`로 복사해 커밋·업로드하세요.
train이면 출력 폴더, demo이면 그 아래 `run`을 지정합니다.

```powershell
$FinishedRun = "runs/실제완료폴더/run"
$ResultFiles = @((Join-Path $FinishedRun "metadata.json"), (Join-Path $FinishedRun "evaluation.json"), (Join-Path $FinishedRun "models/training.json"))
foreach ($ResultFile in $ResultFiles) {
    if (-not (Test-Path -LiteralPath $ResultFile)) { throw "완료된 실행 파일이 없습니다: $ResultFile" }
}
$ResultShareRoot = Join-Path "results" ("run-" + (Get-Date -Format "yyyyMMdd-HHmmss-fff"))
New-Item -ItemType Directory -Path $ResultShareRoot -Force | Out-Null
Copy-Item -LiteralPath $ResultFiles -Destination $ResultShareRoot
Write-Host "Git에 올릴 결과: $ResultShareRoot"
git status --short
```

위 폴더에는 원본 데이터·모델 가중치를 복사하지 않습니다. metadata에 입력 해시와
대여소 목록이 포함되므로 공유 전에 내용을 확인하세요. `run_size_comparison.ps1`도 이제
base/medium 요약과 비교 CSV를 `results/size-compare-...`에 자동 복사합니다.
해당 비교 스크립트를 실행할 때도 이 문서의 다음 명령을 사용합니다.

```powershell
& .\run_size_comparison.ps1
```

## 실행 전후 확인

1. CSV의 누락을 실제 0건/0대로 채우지 않았는지 확인합니다.
2. 날짜·시드·모델 크기·예산·출력 폴더를 확인합니다.
3. 명령이 성공한 뒤 출력된 폴더의 결과를 확인합니다. 이전 폴더의 결과와 혼동하지 않습니다.
4. 합성 결과와 실제 자료 결과를 구분합니다. 크기 선택은 검증 점수로 합니다.
5. 새 실행 절차가 생기면 **run.md에 먼저 명령과 결과 위치를 추가합니다.**
