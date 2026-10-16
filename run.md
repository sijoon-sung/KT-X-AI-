# 실행 안내 — 모든 실행은 이 문서 기준

앞으로 설치·학습·평가·예측·재배치 추천·테스트 명령은 **이 파일에서 관리합니다.**
실행 옵션이나 절차를 변경하면 이 문서를 함께 갱신합니다.
README는 프로젝트 설명, 이 문서는 실제 실행 순서의 기준입니다.

**Windows PowerShell / 저장소 루트에서 실행합니다.** 필요한 코드 블록만 터미널에
복사하세요. Markdown 파일 자체를 프로그램처럼 실행하는 방식은 아닙니다.
가상환경 활성화 없이 `.venv`의 Python을 직접 호출하므로 터미널을 다시 열어도 같은 방식입니다.

| 하려는 일 | 이동 |
|---|---|
| 처음 설치 | [1. 설치](#1-설치-최초-1회) |
| 코드가 도는지 빠르게 확인 | [2. 빠른 확인](#2-빠른-확인-합성-데이터) |
| 확대 모델까지 전체 시연 | [3. Medium 전체 실행](#3-medium-전체-실행-합성-데이터) |
| 기본형과 확대형 비교 | [4. 크기 비교](#4-base와-medium-비교-합성-데이터) |
| 실제 자료로 세 모델 학습·평가 | [5. 실제 데이터 학습](#5-실제-데이터-학습평가) |
| 예상 재고와 이동 대수 계산 | [6. 재배치 추천](#6-예상-재고-대수와-재배치-추천) |
| 기존 저장 모델로 합성 시연 다시 계산 | [7. 재학습 없이 추천](#7-재학습-없이-합성-예제-추천만-다시-계산) |
| 결과 위치와 옵션 확인 | [8. 결과](#8-결과-파일), [9. 설정](#9-자주-바꾸는-설정) |
| 테스트 | [10. 테스트](#10-테스트와-실행-도움말) |

## 1. 설치 (최초 1회)

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
| `--profile` | `base` (48) | `medium`은 96. 학습/demo에 사용 |
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

## 실행 전후 확인

1. CSV의 누락을 실제 0건/0대로 채우지 않았는지 확인합니다.
2. 날짜·시드·모델 크기·예산·출력 폴더를 확인합니다.
3. 명령이 성공한 뒤 출력된 폴더의 결과를 확인합니다. 이전 폴더의 결과와 혼동하지 않습니다.
4. 합성 결과와 실제 자료 결과를 구분합니다. 크기 선택은 검증 점수로 합니다.
5. 새 실행 절차가 생기면 **run.md에 먼저 명령과 결과 위치를 추가합니다.**
