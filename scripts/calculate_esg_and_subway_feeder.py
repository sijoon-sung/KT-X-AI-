import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path("C:/Users/sijoo/Documents/tashu")
NODES_PATH = ROOT / "data/processed/v2/nodes.parquet"
TRIPS_PATH = ROOT / "data/processed/trips.parquet"

nodes = pd.read_parquet(NODES_PATH)
trips = pd.read_parquet(TRIPS_PATH, columns=['rent_st', 'ret_st', 'dist_km', 'dur_min', 'is_valid', 'rent_ts'])
trips = trips[trips['is_valid']].copy()
trips['rent_st'] = trips['rent_st'].astype(str)
trips['ret_st'] = trips['ret_st'].astype(str)

# 1. 대전 지하철 1호선 22개 역 매핑 (대여소명에 '역'이 포함되거나 역세권 대여소)
subway_keywords = [
    '반석역', '지족역', '노은역', '월드컵경기장역', '현충원역', '구암역', '유성온천역', 
    '갑천역', '월평역', '갈마역', '정부청사역', '시청역', '탄방역', '용문역', 
    '오룡역', '서대전네거리역', '중구청역', '중앙로역', '대전역', '대동역', '신흥역', '판암역'
]

def find_subway_station(name):
    for kw in subway_keywords:
        if kw in name:
            return kw
    return None

nodes['subway_name'] = nodes['name'].apply(find_subway_station)
st_to_subway = nodes.set_index('station_id')['subway_name'].dropna().to_dict()

# 2. 통행별 First-Mile / Last-Mile 분류
trips['origin_subway'] = trips['rent_st'].map(st_to_subway)
trips['dest_subway'] = trips['ret_st'].map(st_to_subway)

trips['is_first_mile'] = trips['dest_subway'].notna() & trips['origin_subway'].isna()
trips['is_last_mile'] = trips['origin_subway'].notna() & trips['dest_subway'].isna()
trips['is_subway_both'] = trips['origin_subway'].notna() & trips['dest_subway'].notna()
trips['is_subway_linked'] = trips['origin_subway'].notna() | trips['dest_subway'].notna()

# 3. ESG 탄소 감축량 계산
total_trips = len(trips)
total_km = trips['dist_km'].sum()
total_hours = trips['dur_min'].sum() / 60.0

# 환경부 / 국토교통부 표준 원단위
EMISSION_FACTOR = 140.0 # g CO2 / km (승용차 평균)
MODE_SHIFT_RATIO = 0.30 # 공영자전거의 승용차/택시 전환율 (KOTI 표준 30%)
GAS_PRICE_PER_LITER = 1650 # 원/L
AVG_FUEL_EFFICIENCY = 12.0 # km/L

# 순 감축 탄소량 (톤)
co2_reduced_ton = (total_km * MODE_SHIFT_RATIO * EMISSION_FACTOR) / 1_000_000.0
# 소나무 식재 효과 (30년생 소나무 1그루당 연간 6.6kg 흡수)
pine_trees_equivalent = (co2_reduced_ton * 1000.0) / 6.6
# 시민 유류비 절감액 (원)
fuel_cost_saved_krw = (total_km * MODE_SHIFT_RATIO / AVG_FUEL_EFFICIENCY) * GAS_PRICE_PER_LITER

# 4. 지하철 연계 통행 정량화
n_first_mile = trips['is_first_mile'].sum()
n_last_mile = trips['is_last_mile'].sum()
n_subway_both = trips['is_subway_both'].sum()
n_subway_total = trips['is_subway_linked'].sum()
subway_share = (n_subway_total / total_trips) * 100.0

# 역별 환승 연계 통행량 Top 10
# 출발 or 도착으로 집계
subway_counts = {}
for sw in subway_keywords:
    c_out = (trips['origin_subway'] == sw).sum()
    c_in = (trips['dest_subway'] == sw).sum()
    subway_counts[sw] = {
        'total': int(c_out + c_in),
        'last_mile_out': int(c_out), # 역에서 출발
        'first_mile_in': int(c_in)   # 역으로 도착
    }

df_sw_rank = pd.DataFrame.from_dict(subway_counts, orient='index').sort_values('total', ascending=False)

print("="*85)
print(" [대전 타슈(Tashu) ESG 탄소 감축 및 지하철 1호선 연계 편익 정량 평가 결과] ")
print("="*85)

print("\n1. [ESG 친환경 탄소 배출 감축 효과]")
print(f"  - 총 분석 통행량: {total_trips:,} 건 (2024.08 ~ 2026.03, 20개월)")
print(f"  - 총 누적 주행거리: {total_km:,.1f} km (지구 {total_km/40075:.1f} 바퀴 순환)")
print(f"  - **이산화탄소(CO2) 순 감축량**: **{co2_reduced_ton:,.1f} 톤 (ton CO2)**")
print(f"  - **소나무 식재 환산 효과**: **30년생 소나무 {int(pine_trees_equivalent):,} 그루** 식재 효과!")
print(f"  - **대전 시민 유류비(기름값) 절감액**: **약 {fuel_cost_saved_krw/100_000_000:.2f} 억 원 ({int(fuel_cost_saved_krw):,} 원)**")

print("\n2. [대전 도시철도 1호선 연계 편익 (First-Mile / Last-Mile)]")
print(f"  - 지하철역 연계 총 통행량: **{n_subway_total:,} 건** (전체 통행의 **{subway_share:.2f}%**가 지하철과 직결!)")
print(f"    ▶ [First-Mile] 집/직장 → 지하철역 도착 통행: **{n_first_mile:,} 건** ({n_first_mile/total_trips*100:.1f}%)")
print(f"    ▶ [Last-Mile] 지하철역 하차 → 학교/직장/집 이동: **{n_last_mile:,} 건** ({n_last_mile/total_trips*100:.1f}%)")
print(f"    ▶ [역간 이동] 지하철역 ↔ 타 지하철역 통행: **{n_subway_both:,} 건**")
print(f"  - 지하철 연계 평균 통행 거리 / 시간: **{trips[trips['is_subway_linked']]['dist_km'].mean():.2f} km / {trips[trips['is_subway_linked']]['dur_min'].mean():.1f} 분**")

print("\n3. [타슈 환승 연계 최상위 지하철역 Top 10]")
print(f"| 순위 | 지하철 1호선 역명 | 총 연계 통행량 | Last-Mile (역에서 대여) | First-Mile (역으로 반납) | 주요 환승 목적지 |")
print(f"| :---: | :--- | :---: | :---: | :---: | :--- |")
dest_hints = {
    '유성온천역': '충남대, 카이스트, 봉명동 상권',
    '정부청사역': '정부청사, 둔산동 학원가, 갤러리아',
    '시청역': '대전시청, 법원, 탄방동 오피스',
    '용문역': '롯데백화점, 용문/탄방 주거지',
    '반석역': '세종시 BRT 환승, 노은지구 주거',
    '월평역': '카이스트, 이마트 트레이더스',
    '탄방역': '둔산 로데오, 탄방동 주거지',
    '서대전네거리역': '서대전역(KTX), 세이백화점',
    '갈마역': '갈마동 주거지, 한밭고등학교',
    '노은역': '노은지구 학원가, 열매마을 단지'
}

for idx, (sw_name, r) in enumerate(df_sw_rank.head(10).iterrows(), 1):
    hint = dest_hints.get(sw_name, '인근 주거/상업지구')
    print(f"| {idx:2d}위 | {sw_name:12s} | **{r['total']:,} 건** | {r['last_mile_out']:,} 건 | {r['first_mile_in']:,} 건 | {hint} |")
