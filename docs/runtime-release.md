# H13 모델 배포와 실행 검증

## 가설

비실험 유지보수 목적: 기존 H13 모델의 원 가중치·필수 과거 특성·잠금 해시·추론 코드를 묶고 실행과 API 사용법을 정리한다. 새 학습·모델 선택·정확도 개선 실험은 수행하지 않는다.

## 실험

계산은 dislab에서 수행했다. 원본 모델 잠금 SHA-256은 `767061e696f2b72fd48a14dbd4ea1851921bd3481405f6887e48d28ee8acbe09`다. 가중치6개와 잠금 대상8개를 원 해시에 대조하고 상태 사전을 strict 로딩했다. 원본 거래 행·재고 API·인증정보·개인 경로는 배포하지 않았다.

최종 실행 코드·릴리즈 target은 `4004f21f71938284a311e7f9e8751ba26bbc73af`다. 패키지는 자동 생성 bytecode를 제외한32개 파일이다. 압축을 풀고 CPU 가상5대여소 예측→해시 검증→RTX3080 가상5대여소 예측→해시 검증→HTTP 검사14개→해시 검증을 확인했다. 예측 코드와 원 가중치는 바꾸지 않았다. CPU·GPU 출력의 새 정확도 비교를 수행한 것은 아니다.

[API 검사](runtime-api-checks.json)는 health/model/predict 정상 경로와 만료·결측 허용 누락·미래 정답·중복ID·미지원 경로 거부, 키 오류 문구 비노출, 잠금 손상 거부, 출력 덮어쓰기 거부, 관리자 wrapper 표시·만료 자료 거부다. 실제 타슈 API 호출은0회다. [패키지 검사](runtime-package-checks.json)에 반복 실행과 업로드 확인을 기록했다.

실행 환경은 Python3.12.10, torch2.7.1+cu118, numpy1.26.4, scipy1.15.3, pandas2.3.3이다. 이 버전의 기존 환경에서 실행했으며 별도의 새 가상환경 설치까지 검사한 것은 아니다. GitHub Research protocol run156은 최종 실행 코드4004f21에서 success였다.

## 결과

[h13-runtime-v1.0.1](https://github.com/sijoon-sung/KT-X-AI-/releases/tag/h13-runtime-v1.0.1)을 게시했다. GitHub API에서 draft=false·prerelease=true, 첨부3개 모두 uploaded를 확인했다. GitHub가 보고한 ZIP digest가 전송 전 SHA-256과 일치했다.

| 항목 | 확인값 |
|---|---|
| GCN / FM head, 시드당 | 317,122 / 67,202 파라미터 |
| 두 시드 FM 경로 / 비교 기준선 포함 | 768,648 / 1,402,892 파라미터 |
| 체크포인트6개 | 5,639,320바이트 |
| 과거 특성 context.npz | 200,047,986바이트 |
| 실행 ZIP | 200,449,072바이트 |
| ZIP SHA-256 | `08ebba6d9213076dd47baef6b58afbac2e34c3ad60d8a4c707a3d40053126fb3` |

원 파일 감사는 [model-size-audit.json](model-size-audit.json), 최종 모델·배포 메타데이터는 [model-publication.json](model-publication.json), 실행은 [quickstart.md](quickstart.md), 관리자 경로는 [dispatch-runbook.md](dispatch-runbook.md)에 정리했다. PR31은 미병합이며 모델 파일은 실행용 prerelease로 제공한다.

교통 연계는 [가상 예시](../examples/transport/README.md)로 실행했다. 버스 ETA·전날 도시철도 하차량을 대여소별 입력 후보로 대응하고,18:02 제공 값은18:00 입력에서 제외했다. 같은3대 공급량에서 기본 순서420초, 혼잡 반영 시 같은 순서660초, 대안 순서450초를 계산했다. 마감600초를 지켰다. 홈페이지의 상태 전환도 실제 확인했다. 현재 H13에 새 교통 특성을 입력하거나 실제 교통의 추가 성능을 평가하지 않았다.

### v1 배포 오류와 복구

첫 v1은 빌드 과정에서 import가 만든 pyc를 manifest·ZIP에 포함했다. 최초 CPU 실행은 통과했지만 이후 GPU 실행 전 검증에서 캐시 hash 변경으로 실패했다. 원 가중치와 특성은 손상되지 않았다. 원 자산과 [v1 메타데이터](releases/h13-runtime-v1.json)를 보존하고 릴리즈에 사용 중단·수정본 안내를 표시했다. v1.0.1은 빌드 시 bytecode 생성을 끄고 pyc·__pycache__를 manifest와 ZIP에서 제외했다.

수정본 HTTP 검사 과정에서는 도우미의 압축 해제 폴더 참조를 고쳤고, 별도 회차에서는 발행 마감 직전의 가상 입력이422로 거부됐다. 정상 회차의14검사와 분리해 보존했다. 최종 패키지는 반복 실행 후에도 검증을 통과했다. API 수집 재시작·쿼터 연장·PR 병합은 수행하지 않았다.
