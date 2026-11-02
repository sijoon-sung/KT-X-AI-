# 재배치 실행

## 구성

| 실행부 | 역할 | 필요한 자료 |
|---|---|---|
| `runtime/run.py` | 잠금 검증·고정 모델 예측 API | 릴리즈의 model/·원래 코드 |
| `live_api_service.py` | H18의 한정된 수집·예측·다음 관측 채점 | 암호화 키·시계 보정·기상·고정 모델·운영 조건 |
| `runtime/dispatch_server.py` | 기존 관리자 UI와 H21 RPC 연결 | 이미 발행된 private 관측 스트림 |
| 홈페이지 | 기술·평가·기록 사례 표시 | 홈페이지 브랜치 정적 파일 |

`runtime/run.py`의 예측 요약 JSON은 H18 스트림 전체가 아니다. 예측 JSON 이름만 바꿔 H21 스트림으로 사용하지 않는다. 현재 재배치 계산기는 기존 스트림의 발행 시각·잠금 해시·표본·대여소 metadata·단조 시계 anchor를 읽는다.

## 기존 스트림으로 관리자 화면 실행

아래 명령은 **dislab**에서 실행한다. 저장소의 `maintenance/model-package-quickstart` 브랜치를 사용한다.

```powershell
python runtime/dispatch_server.py --stream-dir PRIVATE_EXISTING_STREAM --port 8770
```

다른 터미널에서:

```powershell
$info = Invoke-RestMethod 'http://127.0.0.1:8770/api/dispatch/info?fresh=1'
$body = @{source_version=$info.source_version; automatic=$true; capacity=20; budget=20; target=6; drop_limit=10; handling_seconds=20} | ConvertTo-Json
Invoke-RestMethod 'http://127.0.0.1:8770/api/dispatch/optimize' -Method Post -ContentType 'application/json' -Body $body
```

조건 필드는 `bulk_dispatch.py`의 RPC 계약을 따른다. 원래 코드의 시각·남은 시간·공급지 보호·적재·복귀 검사를 그대로 사용한다. 409 응답이면 반환된 `current_info`에서 최신 `source_version`을 다시 읽는다. 만료한 관측이나 재부팅 전 단조 시계 anchor로 새 계획을 발행하지 않는다.

로컬 화면으로 전달하려면 기존 SSH 설정에서 `ssh -N -L 8770:127.0.0.1:8770 $env:DISLAB_SSH_TARGET`를 실행하고 `http://127.0.0.1:8770/dispatch`를 연다. 로컬 GPU에서는 계산하지 않는다.

## 수집 상태와 종료

H18은 최대 24시간·143회의 시험 수집이다. 종료 후 자료는 기록이며 현재 관측으로 사용하지 않는다. 위 명령은 표시·계산 서버만 시작하고 API 수집은 재개하지 않는다. 새 수집을 실행할 때는 인증·기간·쿼터·시계 보정·입력과 평가의 분리를 다시 확정한다.

차량 수·적재·주차·작업시간은 가정값이다. 화면의 수량과 경로는 계획이며 실제 출동 명령이나 현장 효과 측정이 아니다.
