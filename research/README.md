# 연구 기록

**현재 원칙: 과거 이력 학습·내부 검증, API 외부 검증 전용.** [자료 계약](../docs/training-validation-contract.md)을 따른다.
아래 표는 과거 API 개발·탐색 실행의 기록이다. 새 계약을 준수한 학습·외부 검증 결과로 승계하지 않는다.

진행 절차: [연구·개발 체계](../docs/research-workflow.md).
상위 설계: [연구 프로그램](../docs/research-program.md), [문헌 비교](../docs/research-evidence-map.md),
[결정 기록](../docs/research-decisions.md). 최종 모형군·아키텍처는 아직 확정하지 않았다.

기획 배경: [프로젝트 문제의식과 해결 계획](../docs/project-charter.md).

| 기록 | 가설 | 실험 | 결과 |
|---|---|---|---|
| [H-0001](../docs/H-0001-observation-resolution.md) | 10분·30분 관측의 대여소별 0대 비율 평균 절대차가 5%p를 넘는다 | 완전 관측 대여소·날짜 57,800개·42일의 짝지은 비교, 도시 날짜 블록 평가 | **지지하지 않음.** 평균 절대차 1.24%p. 연속 시간의 품절 관측 충분성을 입증하지 않음 |
| [H-0002](../docs/H-0002-architecture-results.md) | 소형 그래프 CFM이 직접 예측보다 30분 뒤 0대 예측을 개선한다 | dislab에서 1,369개 대여소·세 시드·7일 시험. 자기 이력·그래프 직접 예측과 비교 | **지지하지 않음.** 현재 양수 표본 Brier: 자기 이력 0.079564, 그래프 직접 0.078826, CFM 0.082581. 전체 방법군으로 확대하지 않음 |
| [H-0003](../docs/H-0003-results.md) | 같은 대여소·현재 2~10대 층의 30분 조건부 변동비 중앙값이 0.8 미만이다 | dislab CPU에서 개발 28일·1,111개 적격 대여소·192,147쌍, 3일·7일 블록 각 2,000회 | **지지.** 중앙값 0.449, 구간 0.383~0.473·0.389~0.470. 과거 API 탐색 결과. 후속 모형 선택 근거에서 제외 |

다음 행동은 [과거 대여이력 원본 감사](../docs/historical-source-audit.md)를 토대로 학습 입력·정답·내부 시간 분할을 정하는 것이다.
API로 학습한 H-0002의 가중치·점수, H-0003의 분산 통계는 새 모델 선택에 사용하지 않는다.
GCN·포아송·Flow Matching의 선택은 과거 이력과 문헌으로 검토하고 새 가설을 동결한 뒤 dislab에서 수행한다.
API는 모델·평가 규칙을 고정한 뒤 외부 검증에만 사용하며, 이미 열람한 수집분과 새 독립 기간을 구분한다.

모든 후속 학습·분석 연산은 사용자가 지정한 dislab에서 수행한다.
원본·가중치·개별 예측은 공개하지 않고, 완료 결과와 중단된 실행을 함께 기록한다.

- `hypotheses/H-0001.json`: 가설, 선행연구, 동결 계획, 실행 연결, 결론의 기준 기록.
- `runs/<run-id>/manifest.json`: 한 실행의 재현 정보와 공개 가능한 지표.
- `templates/`: 새 가설과 실행 기록의 양식. 실제 연구 결과가 아니다.

```bash
python tools/research.py new H-0001 --title "검증할 질문"
python tools/research.py freeze H-0001
python tools/research.py check
python tools/research.py list
```

실험 전에 Draft PR을 열고 동결 계획을 push한다. 원본·가중치·큰 예측 출력은 저장소 밖에 둔다.
