# 타슈 재배치 운영 도우미

과거 이용 흐름과 대여소 관계를 예측해 현재 관측과 연결하고, 부족 위험에 따라 수거·배치 수량과 방문 순서를 계산한다. 담당자는 차량 조건을 바꾸고 외부 LLM에 전체 계획을 물어볼 수 있다. GCN·포아송·Flow Matching은 이 운영 과정의 예측 부분을 맡는다. 현재 검증한 재고 예측 구간은 10분이다.

홈페이지·실행 도구·연구 기록을 이 브랜치에 모았다. 새로 시작할 때는 아래 실행 안내를 따르면 된다.

[공모전 발표·도입 계획](docs/competition-guide.md)은 대전의 빈 대여소 문제, 실제 시연 순서와 4주 시범 운영안을 설명한다. [외부 LLM 실행](docs/quickstart-external-llm.md)을 따르면 지도에서 계산한 이동 대수와 방문 순서를 Antigravity 외부 모델에 물어볼 수 있다. 자료 준비·답변 검사는 dislab에서 실행한다. 이전 [소형 Qwen 실행](docs/quickstart-llm.md)과 시험 기록도 보존했다.

[설문을 반영한 IR 발표자료](docs/competition-ir-deck.md)와 [무음 발표 영상](assets/demo/tashu-dispatch-presentation.mp4)을 제공한다. 대전 전체 이용 연결에서 작업 권역으로 확대해 대여·반납 시나리오와 세 권역의 재배치 경로를 보여준다. [시나리오 계산과 출발 전 검토](docs/scenario-dispatch.md)

## 실행

[고정 모델 실행 묶음](https://github.com/sijoon-sung/KT-X-AI-/releases/tag/h13-runtime-v1.0.1)을 내려받으면 재학습 없이 원본 가중치를 실행할 수 있다. 묶음에는 가중치, 과거 특성, 추론 코드, 해시 목록이 포함된다.

계산은 dislab에서 실행한다. 로컬에서는 홈페이지 표시와 HTTP 요청·응답 확인을 한다. 자세한 설치·다운로드·API 명령은 [빠른 실행 안내](docs/quickstart.md)에 정리했다.

```powershell
# 모델 ZIP을 푼 계산 서버의 tashu-h13-runtime-v1.0.1 폴더에서
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py verify
.\.venv\Scripts\python.exe run.py demo --device cpu --out outputs/demo.json
```

`demo`는 가상 재고를 입력한 설치 확인이다. API 키 없이 실행하며 실제 타슈 관측·정확도 결과가 아니다.

| 실행 명령 | 용도 |
|---|---|
| `run.py verify` | 원 잠금 해시·모든 파일 검증 |
| `run.py inspect --device cpu` | 파라미터 수·시드·대여소 수 확인 |
| `run.py demo --device cpu` | 가상 5개 대여소 예측 |
| `run.py capture ...` | 공식 재고 API 한 번 수신 |
| `run.py predict ...` | 현재·직전 관측으로 10분 예측 |
| `run.py serve ...` | `/health`, `/api/model`, `/api/predict` 서버 |

실시간 명령은 현재 확인한 시계 보정·불확실성이 필요하다. API 키는 환경변수 `TASHU_API_KEY` 또는 기존 암호화 키 파일을 사용한다. 키와 원본 API 응답은 저장소에 올리지 않는다. 명령별 필수 인자와 요청 예시는 [API 사용법](docs/quickstart.md)에 있다.

## 97MB 확대 모델

[GCN·Flow Matching 확대 모델](https://github.com/sijoon-sung/KT-X-AI-/releases/tag/h22-100mb-v1.0.1)은 실제 가중치97.00MB, 매개변수24,245,640개다. 두 시드의 체크포인트를 함께 사용한다. 설치와API 실행은 [100MB 모델 안내](docs/quickstart-h22.md)를 따른다.

3월 거래 분포 점수는 기존 FM보다 0.58% 나빠졌다. 9월 2,458,501쌍의 fairCRPS는 0.180022로 기존 FM보다 0.15% 높았다. 확대 모델은 별도로 제공하며 기본 예측·재배치는 H13을 유지한다. [전체 비교](docs/H-0022-100mb-model.md)에서 조건과 지표를 확인할 수 있다.

## 기존 모델 크기

| 구성 | 파라미터 |
|---|---:|
| GCN 1개, encoder 포함 | 317,122 |
| FM head 1개 | 67,202 |
| 두 시드의 GCN + FM 예측 경로 | 768,648 |
| 비교 기준선까지 로딩한 전체 | 1,402,892 |

시드 11·29, 표본 128개, Heun 16단계다. 가중치 외에 약 200MB의 대여소별 과거 특성 파일이 필요하다. 학습 대여소는 1,318개이며 실제 입력·평가에 대응되는 대여소 수는 관측마다 다르다. 모델 구조와 원 파일 해시는 [모델 카드](docs/model-card-h13.md)와 릴리즈의 `model-info.json`에 있다.

## API와 재배치

모델 API는 재고 분포를 반환한다. 관리자 UI·H21 계산기를 시작하는 방법과 요청 형식은 [재배치 실행 안내](docs/dispatch-runbook.md)에 정리했다.

```powershell
# 저장소 폴더, 계산 서버에서. 이미 발행된 관측 스트림 필요.
python runtime/dispatch_server.py --stream-dir PRIVATE_EXISTING_STREAM --port 8770
```

이 명령은 API 수집을 재시작하지 않는다. H18의 기존 24시간·143회 시험 수집과 새 운영 수집을 구분한다. 수량·차량 적재·주차·상하차·복귀를 검사하며 실제 차량과 작업시간은 아직 가정값이다.

홈페이지는 이 저장소의 `index.html`이다. 저장소 폴더에서 `python -m http.server 8771 --bind 127.0.0.1`로 표시한다.

## 교통 연계 예시

[가상 입력과 실행 코드](examples/transport/README.md)로 버스 ETA·과거 도시철도 하차량을 대여소 특성에 대응하고, 도로 혼잡에 따라 같은 공급량의 방문 순서를 바꾸는 과정을 보여준다.

```powershell
python examples/transport/example.py --out private/transport-example.json
```

현재 H13에 새로운 교통 특성을 넣어 재예측한 결과는 아니다. 새 예측 입력을 쓰는 모델은 과거 자료로 별도 학습·선택해야 한다.

## 평가와 자료

2026-09-17~30의 1,298개 대여소·2,458,501쌍에서 현재 유지 대비 fairCRPS가 포아송 15.49%, FM 15.22% 감소했다. 재고 MAE는 현재 유지 0.212015대, 포아송 0.212039대, FM 0.211982대였다. FM fairCRPS는 포아송보다 0.327% 높았다.

- [H13 평가·실행 이력](docs/H-0013-rolling-api-flow.md)
- [H21 재배치 계획·운영 가정](docs/H-0021-bulk-rebalancing.md)
- [학습과 API 사용 계약](docs/training-validation-contract.md)
- [9월 평가 기간·정답](docs/september-evaluation-contract.md)
- [배포·검증 기록](docs/runtime-release.md)
- [기존 연구 기록](https://github.com/sijoon-sung/KT-X-AI-/blob/f218c83b015fffbac6fe1db686cdbe22f36a992e/README.md)

다운로드한 과거 거래로 학습·내부 선택하고, API는 고정 모델의 당시 입력과 미래 평가 정답으로 분리한다. API 학습·튜닝·보정·모델 선택은 하지 않는다. 9월 자료는 반복 사용한 평가 자료다. 2·3월 거래 기록에는 실측 재고 정답이 없다. H13 로컬 학습 위치 위반과 dislab 평가 전환은 원 기록에 보존했다.
