# 질문으로 이동 계획 확인하기

“몇 대를 옮겨?”, “어떤 순서로 가면 돼?”라고 물으면 이미 계산된 예측·계획을 설명한다. Qwen3-1.7B를 dislab GPU에서 실행하며 외부 LLM API를 사용하지 않는다. 새 예측이나 배차 지시를 만드는 기능은 아니다.

## 계산 서버에서 시작

기존 CUDA용 PyTorch 환경이 있는 계산 서버의 저장소 루트에서 실행한다. 별도 가상환경으로 기존 예측 환경과 분리한다. 설치 확인 환경은 Torch 2.7.1+cu118, Transformers 4.57.1, RTX 3080이다. 모델 revision과 파일 해시는 입력 명세에 고정돼 있다.

```powershell
python -m venv --system-site-packages .venv-llm
.\.venv-llm\Scripts\python.exe -m pip install -r runtime/requirements-llm.txt
.\.venv-llm\Scripts\python.exe runtime/operator_assistant.py download --out PRIVATE_MODEL_DIR

# 키 없이 실행하는 가상 대여소 예시
.\.venv-llm\Scripts\python.exe runtime/operator_assistant.py ask --model-dir PRIVATE_MODEL_DIR --context examples/assistant-demo.json --question "자전거를 총 몇 대 옮기는 계획이야?"

# 기존 공개 기록에 담긴 실제 대여소·가정 차량 조건의 이동 계획 재생
.\.venv-llm\Scripts\python.exe runtime/operator_assistant.py serve --model-dir PRIVATE_MODEL_DIR --plan research/runs/H-0021-20261101-bulk/reference-replay.json --port 8782
```

`PRIVATE_MODEL_DIR`은 내려받을 폴더 경로로 바꾼다. 서버는 `127.0.0.1:8782`에만 열리며 종료는 실행 창에서 Ctrl+C다. 원격 연결을 끊어도 계속 실행하려면 운영 서버의 작업 관리 방식으로 등록한다. 이 프로젝트는 SSH·방화벽 설정을 바꾸지 않는다.

Qwen은 실제 1,720,574,976개 파라미터다. 원본 safetensors는 두 파일 합계 4,063,515,592바이트다. 실행은 FP16이며 시험 중 GPU 할당량은 약 3.5GB 수준이었다. 디스크 파일 크기와 GPU 메모리는 다른 값이다. 기존 H13 예측 가중치는 별도로 유지한다.

## 내 컴퓨터에서 화면 열기

SSH 별칭 `dislab`이 기존 키로 연결되는 환경에서 아래 명령을 실행한다. 이 프로그램은 화면과 요청만 전달하며, 답변 생성과 자료 요약은 계산 서버에서 한다. SSH 포트 포워딩이 제한된 서버에서도 사용할 수 있다.

```powershell
python runtime/assistant_proxy.py --ssh-host dislab --remote-port 8782 --port 8772
```

브라우저에서 `http://127.0.0.1:8772/`를 연다. 로컬 프록시 종료는 Ctrl+C다. SSH 비밀번호나 키를 코드·채팅에 넣지 않는다. `dislab`은 사용자 SSH 설정의 별칭이며 접속 정보는 공개 저장소에 넣지 않는다.

## 기존 예측·계획 연결

```powershell
# 고정 H13 runtime/run.py가 저장한 요약 JSON
python runtime/operator_assistant.py ask --model-dir PRIVATE_MODEL_DIR --forecast PRIVATE_FORECAST.json --question "자료에 있는 대여소 중 자전거가 없는 곳을 알려줘."

# 같은 시각·같은 운영 상황에서 발행한 예측과 이동 계획
python runtime/operator_assistant.py serve --model-dir PRIVATE_MODEL_DIR --forecast PRIVATE_FORECAST.json --plan PRIVATE_PLAN.json --port 8782
```

H13 모델 잠금 해시가 맞는 예측만 받는다. 예측과 계획의 시각이 다르면 거부한다. 시각 일치만으로 같은 상황임이 증명되지는 않으므로, 합성 예측과 실제 기록의 계획을 섞지 않는다. 실제 연결 시 같은 발행 버전으로 파일을 함께 저장한다. 서버는 요청마다 파일을 읽으므로 새 결과 파일을 원자적으로 바꾸면 다음 질문부터 반영된다. 날짜·시각과 예시/기록 여부는 화면에 표시한다. 이 버전은 직접 실시간 API를 수집하지 않는다.

## HTTP

```powershell
Invoke-RestMethod http://127.0.0.1:8772/api/health
Invoke-RestMethod http://127.0.0.1:8772/api/context
$body=@{question='자전거를 총 몇 대 옮기는 계획이야?'} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8772/api/ask -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

응답의 `engine=llm`은 실제 Qwen 생성, `engine=facts`는 원 자료 문장 표시다. 후자는 LLM 성공으로 세지 않는다. 근거 ID는 서버가 붙이며 LLM의 독립 검증 결과가 아니다. 수치·단일 방문 수량·방문지 누락 검사를 하지만 모든 문장 의미를 보장하지는 않는다. 현장 배차는 원래 계획표와 담당자 확인을 따른다.

설치·고정 질문 시험·실패 및 수정 기록은 [H26 결과](H-0026-operator-llm.md), 공모전 구성은 [발표와 도입 계획](competition-guide.md)에 있다.
