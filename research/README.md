# 연구 기록

진행 절차: [연구·개발 체계](../docs/research-workflow.md).

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
