# 모델 실행과 API

계산은 dislab에서 실행한다. 로컬 컴퓨터에서는 홈페이지 표시와 HTTP 요청·응답 확인만 한다. 아래 `run.py` 명령은 계산 서버의 PowerShell 기준이다.

## 1. 내려받기

[고정 모델 실행 묶음](https://github.com/sijoon-sung/KT-X-AI-/releases/tag/h13-runtime-v1)에서 ZIP과 SHA-256 파일을 받는다.

```powershell
$base = 'https://github.com/sijoon-sung/KT-X-AI-/releases/download/h13-runtime-v1'
Invoke-WebRequest "$base/tashu-h13-runtime-v1.zip" -OutFile tashu-h13-runtime-v1.zip
Invoke-WebRequest "$base/tashu-h13-runtime-v1.zip.sha256" -OutFile tashu-h13-runtime-v1.zip.sha256
$expected = ((Get-Content tashu-h13-runtime-v1.zip.sha256) -split '\s+')[0]
if ((Get-FileHash tashu-h13-runtime-v1.zip -Algorithm SHA256).Hash.ToLower() -ne $expected) { throw 'ZIP hash mismatch' }
Expand-Archive tashu-h13-runtime-v1.zip -DestinationPath .
Set-Location tashu-h13-runtime-v1
```

ZIP에는 `run.py`, `code/`, `model/`, `requirements.txt`, `MODEL_CARD.md`, `model-info.json`, `package-manifest.json`이 들어 있다. 가중치만 복사하면 과거 특성 파일이 없어 실행되지 않는다.

## 2. 환경 설치와 합성 입력 실행

Python 3.12를 사용한다. 기존 실험 환경을 바꾸지 않도록 별도 환경을 만든다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py verify
.\.venv\Scripts\python.exe run.py inspect --device cpu
.\.venv\Scripts\python.exe run.py demo --device cpu --out outputs/demo.json
Get-Content outputs/demo.json
```

`demo`는 원본 고정 모델에 가상 재고 5개를 입력한다. API 키와 외부 통신이 필요 없다. 출력 `mode`는 `synthetic_offline_demo`다. 재현·설치 확인용이며 실제 타슈 예측이나 정확도 평가로 사용하지 않는다. 기존 출력 파일을 덮어쓰지 않는다.

GPU를 쓰려면 계산 서버의 CUDA에 맞는 PyTorch를 설치한 뒤 `--device cuda`를 쓴다. 기존 검증 환경은 PyTorch 2.7.1+cu118이다. 설치 명령은 [PyTorch 공식 안내](https://pytorch.org/get-started/previous-versions/)의 2.7.1 항목을 따른다. README가 최신 CUDA 드라이버를 설치하거나 다른 GPU 작업을 종료하지는 않는다.

## 3. 타슈 API에서 한 번 관측

공식 대여소 API는 `https://bikeapp.tashu.or.kr:50041/v1/openapi/station`이며 인증 헤더는 `api-token`이다. 키를 명령행 인자나 문서에 넣지 않는다. 환경변수 `TASHU_API_KEY` 또는 기존 Windows 계정의 암호화 키 파일을 사용한다.

```powershell
# 기존 암호화 키 파일이 있는 계산 서버에서 실행한다.
$env:TASHU_CREDENTIAL_FILE = 'PRIVATE_ENCRYPTED_KEY_FILE'
# 시각 동기화를 확인하고 native UTC 대비 보정값·불확실성을 넣는다.
$env:TASHU_CLOCK_OFFSET = 'VERIFIED_OFFSET_SECONDS'
$env:TASHU_CLOCK_UNCERTAINTY = 'VERIFIED_UNCERTAINTY_SECONDS'
.\.venv\Scripts\python.exe run.py capture --credential $env:TASHU_CREDENTIAL_FILE --clock-offset $env:TASHU_CLOCK_OFFSET --clock-uncertainty $env:TASHU_CLOCK_UNCERTAINTY --wait --out private/current.json
```

`capture`는 한 번 요청한 뒤 끝난다. `--wait`는 다음 10분 경계까지 기다리는 옵션이며 반복 수집이 아니다. 키가 이미 환경변수에 있으면 `--credential`을 생략한다. 기존 H18 시험 수집의 기한·143회 상한을 연장하거나 종료된 수집을 자동 재시작하지 않는다. 새 수집은 별도 운영 조건에 맞춰 실행한다.

과거 dislab 시계 지연이 있었으므로 이전 실험의 보정값을 현재 기본값으로 복사하지 않는다. 동기화가 확인된 시스템은 offset=0과 측정한 uncertainty를 사용한다. 불확실성이 2초를 넘거나 보정값이 없으면 live 명령은 거부한다.

저장 형식:

```json
{
  "scheduled_at_unix_seconds": 1793610000,
  "observed_at_unix_seconds": 1793610002,
  "payload": {"results": [{"id": "EXAMPLE_ID", "parking_count": 3}]}
}
```

위 JSON은 형식 예시다. 실제 예측에는 그 시점에 수집한 파일을 쓴다. `scheduled_at`은 600초 경계, 수신 지연은 120초 이내여야 한다. 누락 대여소는 0으로 채우지 않는다. 미지원 ID는 채점 대상이 아니며, 실제 지원 대여소 수는 `supported_valid_stations`로 확인한다.

## 4. 한 번 예측

```powershell
.\.venv\Scripts\python.exe run.py predict --device cpu --current private/current.json --allow-missing-weather --clock-offset $env:TASHU_CLOCK_OFFSET --clock-uncertainty $env:TASHU_CLOCK_UNCERTAINTY --out outputs/forecast.json
```

직전 10분 관측이 있으면 `--previous private/previous.json`을 추가한다. 없으면 기존 모델의 최근 입력 결측 마스크를 사용한다. `--allow-missing-weather`는 원래 기상 결측 인코딩을 명시적으로 허용한다. 출력은 대여소별 평균·중앙값·0대 확률·5/95% 표본 분위수이며 원본 예측 코드와 가중치를 호출한다.

현재 관측이 만료됐거나 미래 정답이 들어오면 예측을 발행하지 않는다. 재실행으로 과거 발행 시각을 만들지 않는다. 미래 관측 채점은 저장소의 `research_code/rolling_inventory_predict.py score`와 원래 NPZ 예측 기록을 사용한다. 이 간편 실행기의 요약 JSON은 원래 NPZ 채점 파일을 대신하지 않는다.

## 5. HTTP 예측 서버

```powershell
.\.venv\Scripts\python.exe run.py serve --device cpu --port 8780 --clock-offset $env:TASHU_CLOCK_OFFSET --clock-uncertainty $env:TASHU_CLOCK_UNCERTAINTY
```

| 경로 | 요청 | 응답 |
|---|---|---|
| `/health` | GET | 준비 상태·모델 잠금 해시 |
| `/api/model` | GET | 파라미터 수·시드·지원 대여소 수 |
| `/api/predict` | POST | 현재·직전 관측을 이용한 10분 예측 |

계산 서버의 다른 터미널:

```powershell
Invoke-RestMethod http://127.0.0.1:8780/health
Invoke-RestMethod http://127.0.0.1:8780/api/model
$body = @{current=(Get-Content private/current.json -Raw | ConvertFrom-Json); allow_missing_weather=$true} | ConvertTo-Json -Depth 15
Invoke-RestMethod http://127.0.0.1:8780/api/predict -Method Post -ContentType 'application/json' -Body $body
```

로컬 컴퓨터에서 접속하려면 기존 SSH 설정으로 포트만 전달한다. 개인 접속 주소는 저장소에 기록하지 않는다.

```powershell
ssh -N -L 8780:127.0.0.1:8780 $env:DISLAB_SSH_TARGET
```

예측 서버는 127.0.0.1에만 연결한다. API 키는 예측 요청 본문에 넣지 않는다. 동시 추론은 직렬화하며 잘못된·만료된 입력은 422, 모델 실행 실패는 503으로 반환한다. 0대 확률은 모델 출력이며 미충족 수요나 배차 실행 결과가 아니다.

## 6. 재배치와 홈페이지

예측 API는 재고 분포를 반환한다. H21 수량·동선 계산기는 저장소의 `research_code/bulk_dispatch.py`이며 별도 관측 스트림과 운영 조건을 사용한다. 기존 관리자 경로의 API는 `GET /api/dispatch/info?fresh=1`, `POST /api/dispatch/optimize`다. 요청에는 해당 관측의 `source_version`과 조건을 넣는다. 소스가 갱신되면 409와 최신 정보를 반환하고, 만료 시 이전 경로를 현재 결과로 쓰지 않는다.

[재배치 실행 안내](dispatch-runbook.md)에서 서버·계산기·표시 화면을 구분한다. 홈페이지는 `maintenance/homepage-methods` 브랜치의 정적 파일이며 모델 실행 묶음과 별도다. 홈페이지 폴더에서 `python -m http.server 8771 --bind 127.0.0.1`로 연다.
