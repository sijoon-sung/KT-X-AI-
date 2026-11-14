# 타슈 권역별 재배치 시나리오

[시나리오 뷰어](../assets/demo/dispatch-scenarios.html) · [무음 발표 영상](../assets/demo/tashu-dispatch-presentation.mp4)

발표 영상에는 지도, 작업 장소와 순서, 묶음 수량을 표시한다. 거리와 예상 시간, 예측 확률 등 발표자가 설명하는 세부 수치는 아래 기록과 시나리오 뷰어에서 확인할 수 있다.

기존 영상의 단일 왕복을 일괄 경로 계산과 권역별 비교로 확장했다. 차량 출발 위치, 복귀 마감, 배치할 빈자리가 달라질 때 수량과 경로가 어떻게 바뀌는지 볼 수 있다. 대전 전체 화면에는 모델이 지원하는 대여소 1,294곳과 과거 대여소 연결 일부를 표시하고 태평·문화권으로 확대한다. 가까운 지도에서는 고정 Flow Matching의 실제 보정값과 대여·반납 표본을 보여 준다. [지도 출처](https://www.openstreetmap.org/relation/2349984)

## 경로 계산 프로그램

`tools/scenario_dispatch.py`는 저장된 H13 예측과 [시나리오 설정](../design/dispatch-scenarios.json)을 읽어 OSRM 도로 경로를 계산한다. dislab에서 기존 H28 Python 환경을 사용한다. `STREAM`은 `state-<Unix 시각>.json`과 그에 대응하는 `forecast-<Unix 시각>.npz`가 있는 비공개 폴더다.

```powershell
python tools/scenario_dispatch.py --stream STREAM --spec design/dispatch-scenarios.json --roads ROAD_CACHE --out OUTPUT
```

`plans.json`, 시나리오별 GeoJSON, `work-orders.csv`가 생성된다. GeoJSON에는 왕복 경로와 순서가 지정된 작업 위치가 들어 있다. HTML은 내려받아 브라우저에서 열고, 시나리오를 선택해 결과를 확인할 수 있다. 이 파일은 계산 결과를 조회하고 선택한 경로를 내보낸다. 새 계산은 위 프로그램에서 실행한다.

## 조건과 결과

기록 재생은 2026년 11월 10일의 19:50, 20:20, 20:50 관측을 사용했다. 수거와 배치에 한 곳 최소 3대, 정차 준비 60초, 상하차 20초/대를 적용했다. 대여소 작업은 예측 시각 30초 전까지 마치는 조건이다. 기본 복귀 마감은 출발 후 15분이다.

| 조건 | 수거·배치 순서 | 이동 | 왕복 | 작업시간 |
| --- | --- | ---: | ---: | ---: |
| 태평 19:50 | 버드내초등학교 → 유천동 278-30 | 6대 | 2.58km | 9.0분 |
| 태평 20:20 | 버드내초등학교 → 유천동 278-30 | 6대 | 2.58km | 9.0분 |
| 태평·문화 20:50 | 버드내초등학교 → 센트럴파크 | 6대 | 4.33km | 10.6분 |
| 둔산 20:50 | 탄방동 e편한세상 → 둔산동 샤크존 | 6대 | 2.46km | 10.1분 |
| 대전역 20:50 | 홍도아파트 버스정류장 → 대전역 동광장 | 4대 | 5.61km | 11.7분 |
| 태평 20:50, 복귀 8분 | 버드내초등학교 → 태평오거리 | 4대 | 1.83km | 7.4분 |
| 태평·문화 20:50, 빈자리 3대 가정 | 버드내초등학교 → 센트럴파크 | 3대 | 4.33km | 8.6분 |

20:50의 기본 세 권역은 차량 3대가 각 권역에서 독립적으로 출발한다는 가정이다. 작업하는 여섯 곳이 겹치지 않았고 합계 16대를 옮기는 계획을 계산했다. 세 권역의 이동 시간은 차량별 왕복 시간이다. 빈자리 3대 조건은 한 곳에 3대 배치를 선택했다. 모든 계산은 당시 예측에 기반한 기록 재생이다.

## 출발 전 검토

```powershell
python tools/review_dispatch_scenarios.py --plans OUTPUT/plans.json --prepare --out OUTPUT/review-request.json
python tools/run_dispatch_review_relay.py --request OUTPUT/review-request.json --relay-workspace RELAY --out OUTPUT/llm-relay.json
```

dislab에서 먼저 수량 보존, 적재량, 작업·복귀 시각과 작업 대상 중복을 검사한다. Antigravity에 넘기는 것은 검증한 요약과 선택 가능한 항목 ID다. `llm-relay.json`의 `candidate` 객체를 `candidate.json`으로 저장하고, 다음 명령으로 그 선택을 다시 검증한다. 선택이 유효하면 코드가 작성한 근거 문장을 원문 그대로 표시한다. 후보 파일을 생략하면 규칙에 따른 검토 순서를 만든다.

```powershell
python tools/review_dispatch_scenarios.py --plans OUTPUT/plans.json --candidate OUTPUT/candidate.json --out OUTPUT/review.json
```

실제 LLM 호출에서 첫 항목은 대전역의 배치 마감 여유 13초였다. 다음은 둔산의 배치 마감 여유 34초와 복귀 8분 조건의 복귀 여유 37초였다. 담당자는 마감 여유가 짧은 경로의 도로 지연과 현장 작업시간을 먼저 확인할 수 있다.

## 시연 영상 구성

| 시간 | 화면 |
| --- | --- |
| 0:08–0:30 | 대전 전체 이용 연결과 태평·문화권 확대 |
| 0:30–1:16 | 권역 연결, Flow Matching 표본, 재고 부족 위험 |
| 1:16–1:46 | 3개 권역 합계 16대와 마감·빈자리 조건별 경로 |
| 1:46–1:57 | 작업표 검사 및 출발 전 검토 |
| 1:57–2:10 | 수거·배치·복귀 경로와 작업 위치 |
