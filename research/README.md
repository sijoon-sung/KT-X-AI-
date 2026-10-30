# 연구 기록

진행 절차: [연구·개발 체계](../docs/research-workflow.md).
상위 설계: [연구 프로그램](../docs/research-program.md), [문헌 비교](../docs/research-evidence-map.md),
[결정 기록](../docs/research-decisions.md). 최종 모형군·아키텍처는 아직 확정하지 않았다.

기획 배경: [프로젝트 문제의식과 해결 계획](../docs/project-charter.md).

| 가설 | 질문 | 상태 |
|---|---|---|
| [H-0001](hypotheses/H-0001.json) | [10분·30분 관측에서의 0대 비율 민감도](../docs/H-0001-observation-resolution.md) | 완료, 5%p 초과 가설을 지지하지 않음 |
| [H-0002](hypotheses/H-0002.json) | [소형 API 재고 그래프 CFM의 추가 가치](../docs/H-0002-architecture-results.md) | dislab에서 완료, 직접 분류기 대비 개선 가설을 지지하지 않음 |

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
