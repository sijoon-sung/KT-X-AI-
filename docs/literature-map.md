# 새 연구의 출발점

조사 기준일: 2026-10-31. 기존 구현의 이름·성능표 대신 관측 가능한 문제와 선행연구에서 시작한다.
이 문서는 1차 문헌 지도이며 신규성을 확정한 연구 결과가 아니다.
모형군을 선택하기 위한 후속 정리는 [연구 질문별 문헌 비교](research-evidence-map.md)와
[큰 연구 방향](research-program.md)에서 이어 간다.

| 연구 | 이미 다룬 문제 | 다음에 확인할 부분 |
|---|---|---|
| [행정안전부 타슈 분석, 2017](https://www.mois.go.kr/frt/bbs/type010/commonSelectBoardArticle.do?bbsId=BBSMSTR_000000000008&nttId=59951) | 타슈 대여·반납과 주변 환경 결합, 기계학습 기반 신규 대여소 후보 선정 | 현재 재고 관측과 당시 거래 자료·사업 목표의 차이. 공식 보도자료 확인, 모형 재현은 미완료 |
| [Feng 외, 2017](https://doi.org/10.1016/j.peva.2017.09.004) | 현재 상태에서 미래 재고 확률분포 예측 | 거래·재고 입력과 순수 상태 관측 입력의 차이 |
| [Gammelli 외, 2020](https://arxiv.org/abs/2001.07402) | 공급으로 절단된 잠재 수요 추정 | 모사한 절단 정답과 실제 관측 정답 구분 |
| [Zhao 외, 2021](https://www.research-collection.ethz.ch/entities/publication/3b834016-d6c4-4c58-90c3-7c08c19d00dd) | 관측 자료 처리와 샘플링의 영향 | 차량별 위치 자료와 대여소 집계 재고의 차이 |
| [Gammelli 외, 2022](https://doi.org/10.1016/j.trc.2022.103571) | 예측 품질과 재고 의사결정 품질의 불일치 | 예측 검증과 정책 효과 검증의 분리 |
| [Rodrigues, 2025](https://ijcai.org/proceedings/2025/857) | 절단과 대체 공급처로의 수요 전이 | 수요 전이 모사 실험의 가정과 자료 요건 |
| [Byrd 외, 2026](https://pubsonline.informs.org/doi/10.1287/msom.2025.0391) | 거의 분 단위 재고·빈 거치대 상태 자료 | 배포·사용 조건·정답 품질을 실제 데이터에서 확인 |
| [Nejadshamsi 외, 2026](https://doi.org/10.1016/j.knosys.2025.114986) | 동적 그래프 기반 대여·반납 공동 예측 | 거래량과 재고 상태 예측의 차이 |

출판사·학회·저자 원문을 우선 확인했다. Feng·Zhao·Nejadshamsi·Byrd 자료는 초록·방법 설명 중심이며,
전체 데이터와 구현을 재현 검증한 것이 아니다. 후속 가설의 `verified_scope`에 실제 확인 범위를 다시 기록한다.

추가 확인: [타슈 공식 API 예시](https://www.tashu.or.kr/main.do)의 `parking_count` 의미와
[Feng 외 저자 공개 원고](https://www.pure.ed.ac.uk/ws/portalfiles/portal/44197453/peva_bike.pdf)의 초록·방법 개요를 확인했다.
Zhao 외 연구의 출판사 검색 노출 자료·샘플링 절에서는 전동자전거 이용이력 대조 검증과
전동킥보드 위치 가용성 자료 분석을 구분한다. 대여소 집계 재고와 같은 입력이 아니다.

첫 가설은 [H-0001 관측 간격과 0대 비율](H-0001-observation-resolution.md)로 좁혔다.
동결 뒤 실행했으며 평균 절대차 1.24%p로 5%p 초과 가설을 지지하지 않았다.
현재 상태별 미래 0대·회복 확률의 신뢰성, 자기 이력 대비 주변 정보의 추가 가치는 후속 질문이다.
구체적인 실험으로 정할 때 각각 별도 가설 PR을 만든다.

API 표시 재고는 수집 시점의 관측값이다. 실제 고장 없는 이용 가능 대수, 개별 대여·반납,
관측 사이의 짧은 품절, 미충족 수요, 재배치 개입 효과를 관측했다고 취급하지 않는다.

## 아키텍처 검토 중 추가 확인한 문헌

아래 문헌은 H-0002 계획 동결 후 추가로 확인했다. 동결된 비교·판정 기준은 변경하지 않는다.

| 문헌 | 실제 확인 범위 | 우리 연구에 주는 제약 |
|---|---|---|
| [Lipman 외, Flow Matching](https://arxiv.org/abs/2210.02747) | 저자 초록·확률 경로 설명 | CFM 자체는 기존 생성 모형이다 |
| [Tong 외, 2024](https://arxiv.org/html/2302.00482v4) | 원문 §3.2.2 독립 결합, §3.2.3 OT 결합 | 직선 경로와 분포 사이 OT 결합을 구분해야 한다 |
| [Kollovieh 외, TSFlow, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/ee1a1ecc92f35702b5c29dad3dc909ea-Paper-Conference.pdf) | 공식 학회 원문의 초록·도입, 저자 초록. 전체 실험 재현 미완료 | CFM을 시계열 확률 예측에 쓰는 것과 사전분포 설계도 이미 연구되었다 |
| [ElGazzar·van Gerven, FlowTime, 2026 개정](https://arxiv.org/abs/2503.10375) | 저자 초록. 전체 방법·코드 재현 미완료 | 관측 이력에 조건부인 미래 궤적 생성과 자기회귀 CFM도 기존 연구다 |
| [Wu 외, AsyncCouple-Flow, 2026](https://arxiv.org/abs/2609.16573) | 저자 초록과 저자가 기재한 ICONIP 2026 채택 안내. 학회 편집본·재현 미확인 | 다중 자료·그래프·Flow Matching의 결합과 누락 대응도 선행 사례가 있다. 타슈 재고에서의 성능은 별도 검증해야 한다 |

따라서 그래프·날씨·대중교통·CFM을 조합했다는 사실만으로 신규성을 주장하지 않는다.
구조의 타당성은 [코드 검토](architecture-audit.md)와 H-0002의 실제 비교를 함께 해석한다.
