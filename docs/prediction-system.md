# 타슈다음: 30분 재고 예측 시제품

## 가설

H-0008은 고정된 과거전용 GCN을 실제 입력·예측·화면에 연결하는 인수검증이다.
사용자 요청에 따라 GCN을 공모전 시연의 기본 출력으로 사용한다.
GCN의 성능 우위를 뜻하지 않으며 [H7의 비교 결과](H-0007-results.md)는 보존한다.

## 실험

입력은 **대여소·한국시간 기준시각·현재 표시 대수**, 출력은 **30분 뒤 중앙값·평균·0대 확률·5~95% 분위수와 전체 분포**다.
과거 대여·반납 이력에서 만든 OD-GCN 흐름과 달력 참고 흐름을 각각 현재 재고에 연결한다.
공간 관계와 고정률은 같은 대여소 순서로 검증하고 체크포인트 CPU 계산과 저장한 고정률의 일치 여부를 시작할 때 확인한다.
입력이 과거 프로필·요일·시간대로 고정되어 있어 검증한 고정률을 캐시로 사용한다. 새 학습이나 API 튜닝은 없다.

화면에는 세 가지 입력 방식이 있다.

1. **9월 기록 재생:** 현재 관측으로 먼저 예측한다. 미래 관측은 별도로 요청해야 보이며 모델 입력으로 전달하지 않는다.
2. **직접 입력:** 학습 범위 내 대여소와 현재 대수를 직접 넣는 시나리오다.
3. **현재 스냅샷 JSON:** 현재 관측 1~64개를 불러온다. 누락은 `null`로 남기고 미지원 ID를 임의 예측하지 않는다.

고정 예측시간은 30분, 지원 발행시간은 한국시간 07~21시다.
정시 외 시나리오는 해당 시간대의 평균률을 30분간 고정하며 정시 발행의 기존 검증 범위와 구분한다.
API 표시 재고는 실제 대여 성공이나 자전거 상태를 보장하지 않는다.
실시간 API 수집·날씨 연동·현장 운영 배포는 이번 구현 범위에 포함하지 않는다.

```json
{
  "schema_version": 1,
  "observed_at": "2026-10-31T15:00:00+09:00",
  "stations": [
    {"station_id": "ST0003", "parking_count": 4},
    {"station_id": "ST0004", "parking_count": null}
  ]
}
```

다음 명령은 **dislab**에서 실행한다. 대문자 자리표시자는 비공개 입력 위치이며 원본·가중치를 공개 저장소에 넣지 않는다.

```bash
python research_code/prediction_service.py --h7-artifacts H7_ARTIFACTS --h4-artifacts H4_ARTIFACTS --station-metadata DISPLAY_METADATA serve --replay-panel AUDITED_API_PANEL --port 8768
python research_code/prediction_service.py --h7-artifacts H7_ARTIFACTS --h4-artifacts H4_ARTIFACTS predict --station-id ST0003 --parking-count 4 --issued-at 2026-10-31T15:00:00+09:00
```

`GET /api/info`는 모델 출처와 지원 대여소를, `POST /api/forecast`는 세 필드의 예측 요청을 받는다.
현재 스냅샷은 `/api/snapshot/forecast`, 재생 예측은 `/api/replay/forecast`, 별도 정답 조회는 `/api/replay/truth`다.
HTTP 서버는 loopback에만 연결한다. 개인 장비 연결 정보는 저장소에 포함하지 않는다.
화면의 JSON 내려받기는 현재 예측을 저장하며 정답을 별도 조회한 경우에만 평가 관측 항목을 추가한다.

## 결과

인수검증 결과는 [H-0008 기록](H-0008-prediction-system.md)에 연결한다.
예측 계산·검증·서버는 dislab에서 수행하고 로컬 화면은 입력·표시만 담당한다.
시제품 연결의 채택과 예측 성능·운영 적합성 판정은 구분한다.
