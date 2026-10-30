# 관측 기반 연구 실행

이 폴더는 기존 `tashu/` 구현의 점수를 재사용하지 않는 새 연구 코드다.
각 실행 전에 해당 가설의 동결 계획과 Draft PR을 확인한다.
사용자의 최신 지정 계산 환경은 **dislab**이다. 아래 학습·분석 명령은 dislab에서 실행하고,
이 컴퓨터에서는 접속·파일 전송·문서 편집·게시만 수행한다.

## 입력

원본은 사용자가 제공한 타슈 API 백업과 그로부터 만든 비공개 패널이다.
공개 파일 목록·해시는 [입력 manifest](../research/datasets/tashu-primary-v1.json)에 있다.
원본 배포 권한을 검증한 것이 아니므로 저장소에는 원본을 포함하지 않는다.
API 표시 재고만 정답으로 쓰며, 누락을 0으로 바꾸지 않는다.

`audit_inputs.py`는 원본 88개 파일과 기존 패널의 값·요청 시각을 독립적으로 대조한다.
재감사에 사용한 환경은 Python 3.12.14, NumPy 2.5.4, pandas 3.0.6이다.
시간대는 한국 시간 조건부 해석이고 서버 갱신 시각은 확인하지 못했다.

## 실행

저장소 루트에서 실행한다. 아래의 대문자 인자는 본인의 비공개 파일 위치로 지정한다.
실제 실행 정보는 각 `research/runs/*/manifest.json`을 기준으로 한다.

```text
python research_code/audit_inputs.py --archive PRIVATE_ARCHIVE --prepared PRIVATE_PREPARED_FOLDER --out research/datasets/tashu-primary-v1.json
python research_code/observation_resolution.py --panel PRIVATE_PANEL --run-id H-0001-20261031-resolution
python research_code/architecture_benchmark.py --prepared PRIVATE_PREPARED_FOLDER --run-id H-0002-20261031-dislab --device cuda --sample-chunk 4
```

완료된 가설을 다시 실행할 때는 기존 기록을 덮어쓰지 않도록 별도 재현 브랜치·새 실행 ID를 사용한다.
실험 대상과 설계를 바꾸면 새 가설 ID가 필요하다.

H-0002는 CUDA 사용 가능 여부와 입력·계획 해시, 커밋된 코드 상태를 점검한다.
GPU 메모리 한도는 전체의 최대 15%로 두며, 사용 가능한 메모리에 따라 더 작게 설정한다.
메모리 부족 시 모형·학습량을 바꾸지 않고 표본 생성 chunk만 줄일 수 있다.
CPU에서는 `--device cpu`와 같은 계획의 4개 스레드를 사용한다.
실측 실행 장치·패키지·메모리 사용은 manifest에 기록한다.

## 검증

```text
python -m unittest discover -s research_code/tests -v
python tools/research.py check
```

합성 자료 검사는 누락과 실제 0 구분, 관측 간격 민감도, 시간 분할, 미래 정보로 대여소를 선택하지 않는지,
그래프의 배치 분리와 CPU/GPU 일치, 미관측 미래 정답이 CFM 손실에 섞이지 않는지 등을 확인한다.
CUDA가 없으면 GPU 일치 검사 하나는 건너뛴다. 이 검사가 통과해도 실증 타당성을 보장하지 않는다.

공개 산출물은 계획·집계·실행 manifest다. 가중치와 개별 예측은 실행 폴더의 `artifacts/`에 보관하며
Git에서 제외한다. 시험 자료의 탐색·재사용 여부와 부정·불확실한 결론도 기록한다.

공개 입력·결과 JSON의 해시는 파일의 정확한 바이트를 대상으로 한다.
`.gitattributes`에서 이 파일들의 줄바꿈 자동 변환을 막아, Windows와 다른 운영체제의 Git 다운로드에서도
실행 당시 해시가 유지되게 한다. 코드·계획의 의미나 관측값을 바꾼 조치가 아니다.
자동 검사에서도 공개 입력 및 결과 파일의 실제 바이트를 기록된 해시와 비교한다.
