# 외부 LLM과 재배치 화면 실행

화면의 ‘전체 재배치 계획 알려줘’ 버튼 하나로 총대수·거리·예상 시간과 모든 방문지의 수거·배치 대수를 함께 확인한다.

현재 답변 모델은 Antigravity의 `gemini-3.8-flash-low`다. 정상 로그인한 공식 CLI가 외부 모델에 요청을 보내고, dislab은 예측·계획 자료를 준비하고 답변의 숫자·방문지를 검사한다. 로컬 GPU로 LLM을 실행하지 않는다. H13 예측 모델도 그대로 유지한다.

## 현재 실행 화면

- [질문하기](http://127.0.0.1:8772/assistant)
- [지도·수량·동선](http://127.0.0.1:8772/dispatch)
- [연결 확인](http://127.0.0.1:8772/api/health)

2026년 11월 10일 20:50 관측 기록을 재생한다. 적재량을 바꾸면 같은 기록으로 이동 계획을 다시 계산한다. 지도와 질문 화면은 같은 관측 버전과 가장 최근에 계산한 계획을 읽는다. 실제 배차를 실행하는 화면은 아니다.

이 컴퓨터에는 `TashuExternalAssistantH27` 로그인 작업을 등록했다. 로그인한 사용자 권한으로만 실행하고 `127.0.0.1`에만 연결한다. SSH·방화벽·외부 공개 범위는 바꾸지 않았다. Python과 정상 로그인한 Antigravity CLI, 기존 dislab SSH 연결이 필요하다.

```powershell
# 시작 / 중지 / 상태
Start-ScheduledTask -TaskName TashuExternalAssistantH27
Stop-ScheduledTask -TaskName TashuExternalAssistantH27
Get-ScheduledTask -TaskName TashuExternalAssistantH27

# HTTP 확인
Invoke-RestMethod http://127.0.0.1:8772/api/health
$body = @{ question = '전체 재배치 계획을 한 번에 정리해줘.' } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8772/api/ask -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

## 다른 환경에서 설치

원격 계산 서버에 이 브랜치와 NumPy가 설치된 Python 환경을 준비한다. 과거 API 관측 스트림의 `status.json`·고정 예측 상태 파일이 있어야 한다. 재학습하거나 새 API 키를 코드에 넣을 필요는 없다.

저장소 밖에 원격 설정 `external-source.private.json`을 만든다. 아래 경로는 각 환경의 실제 경로로 바꾼다. HMAC 키 파일은 첫 실행 때 생성된다.

```json
{
  "provider": "antigravity",
  "model": "gemini-3.8-flash-low",
  "ticket_key": "PRIVATE_DIRECTORY/answer-ticket.key",
  "source": {
    "stream": "PRIVATE_OBSERVATION_STREAM_DIRECTORY",
    "mode": "recorded_replay",
    "replay_offset_seconds": 15,
    "plan_cache": "PRIVATE_DIRECTORY/latest-plan.json"
  }
}
```

로컬 설정도 저장소 밖에 둔다. `remote_repo`와 `remote_python`은 SSH 서버의 경로다. 로그인 토큰을 복사하지 않고 이 컴퓨터의 공식 `agy` 로그인을 사용한다.

```json
{
  "ssh_host": "dislab",
  "remote_repo": "REMOTE_CHECKOUT_DIRECTORY",
  "remote_python": "REMOTE_PYTHON_EXECUTABLE",
  "remote_config": "REMOTE_PRIVATE_CONFIG_FILE",
  "relay_workspace": "LOCAL_EMPTY_RELAY_DIRECTORY",
  "agy": "LOCAL_AGY_EXECUTABLE"
}
```

```powershell
python runtime/external_proxy.py --config PRIVATE_LOCAL_CONFIG_FILE --port 8772

# Windows 로그인 시 자동 시작. 사용자 권한으로 등록한다.
.\runtime\manage_external.ps1 -Action install -Config PRIVATE_LOCAL_CONFIG_FILE -Pythonw PATH_TO_PYTHONW -Port 8772
```

공식 CLI는 모델 고정·JSON 스키마·90초 제한·동시 1요청으로 실행한다. 파일·명령·웹·MCP 도구를 허용하지 않는다. 외부 모델에는 질문과 허용된 짧은 사실만 전달한다. 인증키, 원본 거래, 개인 경로는 전달하지 않는다.

## 새 관측을 연결하려면

기존 [API 실행 안내](quickstart.md)에 따라 시계를 확인하고 상한 있는 수집을 시작한다. 원격 설정의 `source.stream`을 새 관측 폴더, `source.mode`를 `live`로 지정한다. `/dispatch`가 새 회차를 읽으면 계획을 다시 계산한다. 시작 직후에는 첫 관측·예측 준비를 기다린다. 중지·누락·만료된 자료로는 새 실시간 계획을 발행하지 않는다. LLM 답변 중 계획이 바뀌어도 오래된 답변을 그대로 내보내지 않는다.

이번 H27 시험 수집은 7회·70분 상한을 마쳐 중지됐다. 수집을 계속하고 있다고 표시하지 않는다. API는 고정 모델의 관측 입력·검증에만 사용하고 학습·보정·모형 선택에는 사용하지 않는다.

## 일반 외부 HTTP API 사용

`external_assistant.py`에는 OpenAI 호환 Chat Completions 어댑터도 있다. 원격 설정에 아래 항목을 넣고 서버 환경변수에 키를 저장한다. 이 경로는 키가 준비된 배포 환경용이며, 이번 실제 연결 검증은 Antigravity로 수행했다.

```json
{
  "provider": "external-http",
  "model": "PROVIDER_MODEL_ID",
  "http": {
    "url": "https://api.openai.com/v1/chat/completions",
    "key_env": "OPENAI_API_KEY"
  }
}
```

위 설정에도 `source`와 `ticket_key`가 필요하다. API 모델이 JSON 스키마 출력을 지원해야 한다.

```powershell
# 계산 서버에서 실행. 기본 연결은 loopback이다.
python runtime/external_assistant.py --config PRIVATE_REMOTE_CONFIG_FILE --port 8783
```

## 오류가 나면

`/api/health`는 원격 준비 경로를 확인한다. 실제 외부 생성 성공은 `/api/ask` 응답의 `engine: external`로 확인한다. `engine: facts`는 자료 문장을 표시한 것이므로 외부 모델 성공과 구분한다. 503이면 SSH·정상 CLI 로그인·서비스 한도를 확인한다. 없는 재고를 0으로 만들거나 지난 자료를 현재 관측으로 바꾸지 않는다.

[H27 결과](H-0027-external-assistant.md) · [시연 대본](../design/demo-video-script.json)
