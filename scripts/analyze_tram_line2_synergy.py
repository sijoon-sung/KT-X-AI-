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

total_trips = len(trips)

# 대전 도시철도 2호선 트램 12대 핵심 축 정의
tram_corridors = {
    '카이스트/대덕연구단지 축': ['카이스트', '구성동', '어은동', '신성동'],
    '도안신도시 주거 축': ['도안', '원신흥', '상대'],
    '대전역/대동 환승 축': ['대전역', '대동역', '신흥'],
    '유천/태평 주거 축': ['유천', '태평'],
    '충남대/대학로 축': ['충남대', '궁동'],
    '관저/건양대병원 주거 축': ['관저', '건양대병원'],
    '유성온천 환승역 축': ['유성온천'],
    '한남대/오정 단절 해소 축': ['한남대', '오정'],
    '도마/배재대 생활 축': ['도마', '배재'],
    '정부청사/둔산 중심 축': ['정부청사', '둔산대공원'],
    '서대전역 KTX 환승 축': ['서대전역', '서대전네거리'],
    '복합터미널 광역교통 축': ['복합터미널', '용전']
}

all_tram_stations = set()
corridor_stats = []

for c_name, kws in tram_corridors.items():
    pattern = '|'.join(kws)
    st_matches = set(nodes[nodes['name'].str.contains(pattern, na=False)]['station_id'])
    all_tram_stations.update(st_matches)
    
    # 해당 축 연계 통행
    linked = trips[trips['rent_st'].isin(st_matches) | trips['ret_st'].isin(st_matches)]
    n_cnt = len(linked)
    avg_km = linked['dist_km'].mean()
    avg_dur = linked['dur_min'].mean()
    
    corridor_stats.append({
        '축_명칭': c_name,
        '소속_타슈대여소수': len(st_matches),
        '20개월_연계통행량': n_cnt,
        '연간_환산통행량': int(n_cnt * 12.0 / 20.0),
        '평균통행거리_km': round(avg_km, 2),
        '평균통행시간_분': round(avg_dur, 1),
        '전체통행_비중': round(n_cnt / total_trips * 100.0, 2)
    })

df_tram = pd.DataFrame(corridor_stats).sort_values('20개월_연계통행량', ascending=False)

# 트램 2호선 전체 영향권 통행량
tram_all_trips = len(trips[trips['rent_st'].isin(all_tram_stations) | trips['ret_st'].isin(all_tram_stations)])
tram_share = (tram_all_trips / total_trips) * 100.0

print("="*85)
print(" [대전 도시철도 2호선 트램(Tram) - 타슈(Tashu) 연계 편익 및 수요 분석 결과] ")
print("="*85)
print(f"▶ 트램 2호선 38.8km 순환선 권역 내 타슈 대여소: 총 {len(all_tram_stations)}개소 (전체의 {len(all_tram_stations)/len(nodes)*100:.1f}%)")
print(f"▶ 트램 권역 내 발생 타슈 통행량: **{tram_all_trips:,} 건 (전체 타슈 이용의 {tram_share:.1f}%!)**")
print(f"▶ 연간 환산 통행량: **약 {int(tram_all_trips * 12.0 / 20.0):,} 건/년**")
print("-" * 85)

print("\n[트램 2호선 12대 핵심 회랑별 타슈 연계 통행량 순위]")
print(f"| 순위 | 트램 2호선 회랑 명칭 | 타슈 대여소 | 20개월 통행량 | 연간 환산 통행량 | 평균거리 | 통행 비중 |")
print(f"| :---: | :--- | :---: | :---: | :---: | :---: | :---: |")

for idx, r in enumerate(df_tram.iterrows(), 1):
    d = r[1]
    print(f"| {idx:2d}위 | {d['축_명칭']:20s} | {d['소속_타슈대여소수']:3d}개소 | **{d['20개월_연계통행량']:,}건** | **{d['연간_환산통행량']:,}건** | {d['평균통행거리_km']}km | {d['전체통행_비중']}% |")

# 결과를 JSON으로 저장
tram_summary = {
    '트램_총_영향권_대여소수': len(all_tram_stations),
    '트램_총_연계통행량_20개월': int(tram_all_trips),
    '트램_통행_점유율': f"{tram_share:.1f}%",
    '연간_환산_통행량': int(tram_all_trips * 12.0 / 20.0),
    '회랑별_상세순위': df_tram.to_dict('records')
}

with open(ROOT / "outputs/final_model/tram_synergy_summary.json", 'w', encoding='utf-8') as f:
    json.dump(tram_summary, f, ensure_ascii=False, indent=2)

print("\n분석 완료 및 outputs/final_model/tram_synergy_summary.json 저장 완료.")
