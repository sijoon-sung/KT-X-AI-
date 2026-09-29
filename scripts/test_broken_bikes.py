import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path("C:/Users/sijoo/Documents/tashu")
trips = pd.read_parquet(ROOT / "data/processed/trips.parquet", 
                        columns=['bike_id', 'rent_ts', 'ret_ts', 'rent_st', 'ret_st', 'dur_min', 'dist_km', 'is_valid'])

print(f"Total trips loaded: {len(trips):,}")

# Bouncing Trip Definition (즉시 반납 징후)
# 대여 후 3분 이내 반납 및 제자리 반납(또는 이동거리 0.1km 이하)
same_st = trips['rent_st'].astype(str) == trips['ret_st'].astype(str)
is_bounce = (trips['dur_min'] <= 3) & (same_st | (trips['dist_km'] <= 0.1))
trips['is_bounce'] = is_bounce

n_bounce = trips['is_bounce'].sum()
pct_bounce = trips['is_bounce'].mean() * 100
print(f"Total bounce trips: {n_bounce:,} ({pct_bounce:.2f}%)")
print(f"Total unique bikes: {trips['bike_id'].nunique():,}")

# 자전거별 통계
bike_stats = trips.groupby('bike_id').agg(
    total_trips=('is_bounce', 'count'),
    bounces=('is_bounce', 'sum'),
    total_dist=('dist_km', 'sum'),
    total_dur=('dur_min', 'sum')
).reset_index()

bike_stats['bounce_rate'] = bike_stats['bounces'] / bike_stats['total_trips']

print("\n--- [고장 의심 자전거 상위 10대 (최소 30회 이상 운행)] ---")
top_broken = bike_stats[bike_stats['total_trips'] >= 30].sort_values('bounce_rate', ascending=False).head(10)
print(top_broken.to_string(index=False))

# 전체 시스템 평균 바운스율 (Prior 모수 추정용)
alpha_prior = bike_stats['bounces'].sum()
beta_prior = (bike_stats['total_trips'] - bike_stats['bounces']).sum()
global_bounce_rate = alpha_prior / (alpha_prior + beta_prior)
print(f"\n전체 타슈 자전거 평균 튕김(Bounce) 비율: {global_bounce_rate*100:.3f}%")
