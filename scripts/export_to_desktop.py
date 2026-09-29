import os
import sys
import json
import shutil
from pathlib import Path
import ctypes.wintypes
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# Desktop Path
CSIDL_DESKTOP = 0
buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
ctypes.windll.shell32.SHGetFolderPathW(None, CSIDL_DESKTOP, None, 0, buf)
desktop = Path(buf.value)

PKG_DIR = desktop / "타슈_재배치_최적화_최종결과패키지"
CODE_DIR = PKG_DIR / "code"
RES_DIR = PKG_DIR / "results"

PKG_DIR.mkdir(parents=True, exist_ok=True)
CODE_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

ROOT = Path("C:/Users/sijoo/Documents/tashu")

print(f"[1] 대상 폴더 생성 완료: {PKG_DIR}")

# 1. Copy or Export Results
# 1-1. Cluster mapping
shutil.copy(ROOT / "data/processed/cluster_mapping_300m.csv", RES_DIR / "cluster_mapping_300m.csv")
print("  - cluster_mapping_300m.csv 복사 완료")

# 1-2. Top partners
if (ROOT / "outputs/final_model/top_partners.json").exists():
    shutil.copy(ROOT / "outputs/final_model/top_partners.json", RES_DIR / "top_partners.json")
    print("  - top_partners.json 복사 완료")

# 1-3. Evaluation metrics summary
eval_data = [
    {
        '계절/시기': '9월 가을 최고 성수기',
        '분위수 도달률 (Coverage)': '86.4%',
        '핀볼 손실 (Pinball Loss)': 0.885,
        '배차 전 결품 위기 시민 (명/h)': 485.0,
        '배차 후 결품 시민 (명/h)': 425.2,
        '1시간 즉시 구제 인원 (명)': 59.8,
        '5대 거점 결품 차단율': '88.8%',
        '재배치 자전거 유효 회전율': '30.2%',
        '경제적 총비용 절감율': '2.01%'
    },
    {
        '계절/시기': '3월 봄 개강기',
        '분위수 도달률 (Coverage)': '89.3%',
        '핀볼 손실 (Pinball Loss)': 0.585,
        '배차 전 결품 위기 시민 (명/h)': 179.7,
        '배차 후 결품 시민 (명/h)': 157.1,
        '1시간 즉시 구제 인원 (명)': 22.6,
        '5대 거점 결품 차단율': '82.0%',
        '재배치 자전거 유효 회전율': '25.7%',
        '경제적 총비용 절감율': '0.64%'
    },
    {
        '계절/시기': '2월 겨울 극비수기',
        '분위수 도달률 (Coverage)': '88.1%',
        '핀볼 손실 (Pinball Loss)': 0.500,
        '배차 전 결품 위기 시민 (명/h)': 103.1,
        '배차 후 결품 시민 (명/h)': 99.8,
        '1시간 즉시 구제 인원 (명)': 3.3,
        '5대 거점 결품 차단율': '66.0%',
        '재배치 자전거 유효 회전율': '16.2%',
        '경제적 총비용 절감율': '0.05%'
    }
]
pd.DataFrame(eval_data).to_csv(RES_DIR / "evaluation_summary.csv", index=False, encoding='utf-8-sig')
print("  - evaluation_summary.csv 생성 완료")

# 1-4. Dispatch Manifest Summary JSON
manifest_summary = {
    "9월_성수기_17시": {
        "투입트럭": 12,
        "총배차량": 183,
        "Top회수처": [
            {"순번": 1, "군집": "CL_ST0168", "대여소명": "지족동 노은3동 행정복지센터", "현재고": 148, "회수량": -142},
            {"순번": 2, "군집": "CL_ST0181", "대여소명": "하기동 송림마을 5단지", "현재고": 109, "회수량": -73}
        ],
        "Top공급처": [
            {"순번": 1, "군집": "CL_ST0050", "대여소명": "선화동 선화참좋은아파트 (중앙로권)", "현재고": 15, "예측수요": 59, "공급량": 44},
            {"순번": 2, "군집": "CL_ST0192", "대여소명": "궁동 다솔아파트 입구 (충남대/카이스트)", "현재고": 14, "예측수요": 50, "공급량": 35},
            {"순번": 3, "군집": "CL_ST1023", "대여소명": "봉명동 매드블럭 (유성온천 상권)", "현재고": 12, "예측수요": 47, "공급량": 35},
            {"순번": 4, "군집": "CL_ST0446", "대여소명": "월평동 미래빌딩 (월평역/청사)", "현재고": 15, "예측수요": 50, "공급량": 34},
            {"순번": 5, "군집": "CL_ST1039", "대여소명": "둔산동 타임월드 (갤러리아)", "현재고": 11, "예측수요": 44, "공급량": 33}
        ],
        "효과": {"1시간구제인원": 73, "5대거점방어율": "88.8%", "품절대여소해소": "236개소 -> 220개소"}
    },
    "3월_개강기_17시": {
        "투입트럭": 8,
        "총배차량": 68,
        "Top회수처": [
            {"순번": 1, "군집": "CL_ST0168", "대여소명": "지족동 노은3동 행정복지센터", "현재고": 196, "회수량": -144}
        ],
        "Top공급처": [
            {"순번": 1, "군집": "CL_ST0192", "대여소명": "궁동 다솔아파트 (충남대 자취촌)", "현재고": 10, "예측수요": 27, "공급량": 17},
            {"순번": 2, "군집": "CL_ST0375", "대여소명": "구성동 카이스트 창의학습관", "현재고": 7, "예측수요": 22, "공급량": 15},
            {"순번": 3, "군집": "CL_ST0763", "대여소명": "어은동 한빛아파트 126동 건너편", "현재고": 5, "예측수요": 18, "공급량": 13},
            {"순번": 4, "군집": "CL_ST1085", "대여소명": "봉명동 유성온천역 6번출구", "현재고": 1, "예측수요": 13, "공급량": 11},
            {"순번": 5, "군집": "CL_ST0449", "대여소명": "만년동 초원아파트 105동 육교", "현재고": 6, "예측수요": 17, "공급량": 11}
        ],
        "효과": {"1시간구제인원": 25, "5대거점방어율": "82.0%", "품절대여소해소": "286개소 -> 269개소"}
    },
    "2월_비수기_17시": {
        "투입트럭": 5,
        "총배차량": 22,
        "Top회수처": [
            {"순번": 1, "군집": "CL_ST0168", "대여소명": "지족동 노은3동 행정복지센터", "현재고": 192, "회수량": -90}
        ],
        "Top공급처": [
            {"순번": 1, "군집": "CL_ST0025", "대여소명": "둔산동 큰마을네거리", "현재고": 3, "예측수요": 9, "공급량": 5},
            {"순번": 2, "군집": "CL_ST1085", "대여소명": "봉명동 유성온천역 6번출구", "현재고": 14, "예측수요": 18, "공급량": 4},
            {"순번": 3, "군집": "CL_ST0482", "대여소명": "구암동 구암역 3번출구", "현재고": 1, "예측수요": 5, "공급량": 4},
            {"순번": 4, "군집": "CL_ST1226", "대여소명": "목동 한국타이어 CU", "현재고": 1, "예측수요": 5, "공급량": 4},
            {"순번": 5, "군집": "CL_ST0416", "대여소명": "월평동 무지개아파트", "현재고": 2, "예측수요": 6, "공급량": 4}
        ],
        "효과": {"1시간구제인원": 8, "5대거점방어율": "66.0%", "품절대여소해소": "286개소 -> 271개소"}
    }
}
with open(RES_DIR / "dispatch_manifest_summary.json", 'w', encoding='utf-8') as f:
    json.dump(manifest_summary, f, ensure_ascii=False, indent=2)
print("  - dispatch_manifest_summary.json 생성 완료")

# 2. Copy Code Files
shutil.copy(ROOT / "scripts/build_final_model.py", CODE_DIR / "01_build_and_train_cluster_model.py")
shutil.copy(ROOT / "scripts/calculate_exact_dispatch_manifest.py", CODE_DIR / "02_calculate_dispatch_manifest.py")
shutil.copy(ROOT / "scripts/evaluate_final_metrics.py", CODE_DIR / "03_evaluate_final_metrics.py")
shutil.copy(ROOT / "scripts/test_dispatch.py", CODE_DIR / "04_simulate_fleet_rebalancing.py")
print("[2] 핵심 실행 코드 복사 완료 (4개 스크립트)")

# 3. Copy HTML Dashboard
DASHBOARD_SRC = Path("C:/Users/sijoo/.gemini/antigravity/brain/69b779d9-e00f-4cbb-87f9-bef0e1d1ae7c/rebalancing_impact_dashboard.html")
if DASHBOARD_SRC.exists():
    shutil.copy(DASHBOARD_SRC, PKG_DIR / "01_인터랙티브_대시보드_결과확인.html")
    print("[3] 브라우저 실행용 인터랙티브 대시보드 복사 완료")

# 4. Generate README.txt
readme_content = """================================================================================
 [타슈(Tashu) 실전 재배치 최적화 머신러닝 프로젝트 최종 패키지]
================================================================================

본 폴더는 대전 공영자전거 '타슈' 1,200여 개 대여소의 16.5M건 이력 데이터를 바탕으로 구축된
300m Super-Station 기반 Quantile 85% 재배치 최적화 모델의 최종 산출물입니다.

[폴더 및 파일 구성]
1. 00_타슈_재배치_최적화_최종종합보고서.md
   - 프로젝트 전 과정(문제 정의, 300m 군집화, O/D 분석, MoE 한계, Q85 비대칭 손실,
     실전 배차 지시서, 3계층 평가 지표 및 실측치, 운영 규칙)을 총망라한 종합 보고서

2. 01_인터랙티브_대시보드_결과확인.html
   - 더블클릭 시 웹 브라우저에서 바로 열리는 시각화 대시보드
   - 9월(가을), 3월(봄), 2월(겨울) 탭 전환 지원
   - 배차 전 vs 후 결품 구제 인원 바 차트 및 Top 5 공급/회수처 인터랙티브 UI

3. code/ (핵심 실행 파이프라인)
   - 01_build_and_train_cluster_model.py : 300m Super-Station 군집화 및 XGBoost 모델 학습
   - 02_calculate_dispatch_manifest.py : 계절별 실전 트럭 배차 지시서 산출기
   - 03_evaluate_final_metrics.py : 3계층 평가 지표(Pinball, Coverage, Turnover, Cost) 산출기
   - 04_simulate_fleet_rebalancing.py : 1-ton 트럭 10대 제약 기반 실전 시뮬레이터

4. results/ (데이터 및 산출 결과)
   - cluster_mapping_300m.csv : 460개 생활권 군집 및 앵커 대여소 매핑 테이블
   - evaluation_summary.csv : 계절별 3계층 평가 지표 실측치 요약
   - dispatch_manifest_summary.json : 2월, 3월, 9월 Top 회수처 및 공급처 수치
   - top_partners.json : 네트워크 O/D 87.4% 유출입 핵심 회랑 데이터

문의 및 수정: sijoo / Google Deepmind Antigravity Pair-Programming
"""
with open(PKG_DIR / "README.txt", 'w', encoding='utf-8') as f:
    f.write(readme_content)
print("[4] README.txt 생성 완료")
