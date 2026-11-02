# 관측 기반 연구 실행

현재 학습 경로는 [H-0004](../docs/H-0004-historical-calendar.md)의 다운로드 이력 기준선이다.
API는 후속 외부 검증 전용이며 학습·튜닝·모형 선택에 사용하지 않는다.
아래 H-0001~H-0003의 API 학습·진단 설명은 과거 재현 기록으로 구분한다.

## H-0004 과거 이력 학습

dislab에서 다운로드 이력 ZIP을 사용한다. 원본·내부 CSV 해시와 동결된 입력 계약을 확인한다.
코드는 API 입력 경로를 받지 않으며 고정 평균·요일×시각 평활 평균 두 모델을 학습한다.

```text
python research_code/historical_calendar.py fit --archive PRIVATE_DOWNLOADED_HISTORY --run-id H-0004-20261031-historical
python research_code/historical_calendar.py predict --model PRIVATE_CALENDAR_MODEL --station-id TRAINING_COHORT_ID --timestamp 2026-04-02T08:00:00+09:00
```

출력은 정시부터 1시간의 적격 기록 대여·반납 기대 건수다. 실측 재고·잠재 수요 예측으로 해석하지 않는다.
학습 대여소와 07~21시 정시만 지원하며, 범위 밖 입력은 오류로 알린다.
`artifacts/constant.npz`, `calendar.npz`, `selection.json`, `prepared_panel.npz`는 비공개 실행 폴더에 보관한다.
완료 실행을 덮어쓰지 않는다. 파생 패널의 0/누락 의미와 평가 한계는 동결 계획을 따른다.

## 이전 API 개발 코드

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
python research_code/stock_dispersion.py --panel PRIVATE_PANEL --run-id H-0003-20261031-structure
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

`stock_dispersion.py`는 H-0003의 개발 자료 진단이다. dislab CPU에서 실행하며 GPU 학습을 하지 않는다.
현재와 세 미래 시점의 공통 완전 관측, 정확한 현재 재고 층, 고정 적격 집합의 도시 날짜 블록 부트스트랩을 사용한다.
합성 자료 검사 `python -m unittest discover -s research_code/tests -p test_stock_dispersion.py -v`의 9개 항목은
분할 경계·누락·실제 요청 시차·조건부 분산·블록 재표집·큰 변화 보존·판정 조건을 점검한다.
이 검사는 dislab에서 통과했으며, [진단 결과](../docs/H-0003-results.md)의 실증 해석과 구분한다.

## H-0005: 고정 과거 흐름에서 미래 재고로 연결

`history_to_stock.py`는 H-0004의 비공개 고정 가중치를 해시로 확인하고, 새 학습 없이 공식 9월을 평가한다.
현재 API는 검증 시작 상태이며 미래 API는 정답이다. API 학습·선택·보정은 하지 않는다.
계획의 반복 상한·꼬리 게이트와 과거 계수만으로 API를 읽기 전에 전이표를 만든다.
실제 ID·시간·누락·현재 양수/0 표본, Brier와 수량 MAE, 도시 날짜 블록 구간을 보고한다.

```text
python -m unittest discover -s research_code/tests -p test_history_to_stock.py -v
python research_code/history_to_stock.py --panel <PRIVATE_AUDITED_API_PANEL> --rate-model-dir <PRIVATE_FROZEN_HISTORY_MODELS> --run-id <NEW_RUN_ID>
```

완료 기록에서의 재현은 별도 재현 브랜치·새 실행 ID가 필요하다. [동결 계획](../docs/H-0005-history-to-stock.md)과
[결과](../docs/H-0005-results.md)를 따른다. 일반 모형군의 성능이나 최종 아키텍처의 증명으로 해석하지 않는다.

## H-0006: 분포를 고정하고 손실에 맞는 점출력 확인

`loss_consistent_stock.py`는 동일한 고정 과거율·0반사 과정의 전방 PMF를 계산한다.
수량 중앙값/평균·같은0대확률·고정분위수와CRPS를 구분하며, 미래값은 별도 점수 함수에서만 사용한다.
API를 이용한 학습·분포/구조/분위수 선택·보정은 없다. 음수 상태 확률·수치 질량·모형내 중앙값 최적성을 검사한다.

```text
python -m unittest discover -s research_code/tests -p test_loss_consistent_stock.py -v
python research_code/loss_consistent_stock.py --panel <PRIVATE_AUDITED_API_PANEL> --rate-model-dir <PRIVATE_FROZEN_HISTORY_MODELS> --run-id <NEW_RUN_ID>
```

[동결 계획](../docs/H-0006-loss-consistent-action.md)·[결과](../docs/H-0006-results.md)를 따른다.
완료 기록은 별도 재현 브랜치·새 실행ID로만 재현한다. 원본·개별 예측은 공개하지 않는다.

## H-0007: 학습 전용 OD 관계와 동일 예산 비교

`historical_od_gcn.py train`은과거원본/패널/고정prior만읽고세관계·각3seed를학습한다. API입력인자가없다.
선택epoch·평균률을저장한뒤3월보조건수진단을한다. `evaluate`는그해시를확인한뒤공식9월을평가한다.
자기관계·OD·치환은같은모수/손실/예산을사용하며9월의승자를선택하지않는다.

```text
python research_code/historical_od_gcn.py train --archive <PRIVATE_HISTORY_ARCHIVE> --history-model-dir <PRIVATE_H4_MODELS> --run-id <NEW_RUN_ID>
python research_code/historical_od_gcn.py evaluate --panel <PRIVATE_AUDITED_API_PANEL> --run-id <SAME_RUN_ID>
```

환경은dislabCUDA학습·CPU평가,CUBLAS_WORKSPACE_CONFIG=:4096:8·OMP/MKL/OPENBLAS4스레드.
[동결계획](../docs/H-0007-historical-od-gcn.md)·[결과](../docs/H-0007-results.md)를따른다.재현은별도브랜치·새실행ID로보존한다.
