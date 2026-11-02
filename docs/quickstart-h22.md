# 100MB 모델 실행

가중치는 `large-11.pt`와 `large-29.pt` 두 개를 함께 사용한다. 정적 특징표는 가중치 크기에 포함하지 않는다.

실행 묶음은 [GitHub Releases](https://github.com/sijoon-sung/KT-X-AI-/releases/tag/h22-100mb-v1.0.1)에서 받는다. Python3.12 환경에서 압축을 풀고 `tashu-h22-100mb-v1.0.1` 폴더에서 실행한다. 연구 연산은 dislab에서 수행한다.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python large_run.py verify
.venv\Scripts\python large_run.py inspect
.venv\Scripts\python large_run.py demo --device cuda --out demo.json
```

`demo`는 합성 입력 기능 검사다. 실제 API를 쓰는 방법·시계 보정·키 보관은 저장소의 [API 실행 안내](quickstart.md)를 따른다. 명령의 `run.py`를 `large_run.py`로 바꾸면 같은 입력 형식과 HTTP 경로를 사용한다.

```powershell
python large_run.py predict --device cuda --current private/current.json --previous private/previous.json --allow-missing-weather --clock-offset OFFSET_SECONDS --clock-uncertainty UNCERTAINTY_SECONDS --out private/prediction.json
python large_run.py serve --device cuda --port 8780 --clock-offset OFFSET_SECONDS --clock-uncertainty UNCERTAINTY_SECONDS
```

`OFFSET_SECONDS`와 `UNCERTAINTY_SECONDS`는 측정한 시계 차이와 오차로 바꾼다. API 키는 환경변수 `TASHU_API_KEY` 또는 기존 암호화 파일로 전달한다. 예측 서버는 키를 요구하지 않으며 수집기와 분리된다. `/health`, `/api/model`, `/api/predict`를 제공한다. API 입력으로 가중치를 갱신하지 않는다.

이 모델은 H13과 별도 릴리즈다. 기존 관리자 화면의 모델을 자동 교체하지 않는다. 전체 자료에서의 검증 점수는 모델 카드와 실험 결과를 확인한다.
