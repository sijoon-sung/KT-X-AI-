import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "data/processed/v3/inventory_hourly.parquet"

df = pd.read_parquet(INVENTORY_PATH,
                     columns=['ts', 'node', 'stock_min', 'stock_mean', 'rent', 'ret'],
                     filters=[('ts', '>=', pd.Timestamp('2025-09-01')), 
                              ('ts', '<=', pd.Timestamp('2025-09-30 23:59:59'))])

df = df.sort_values(['node', 'ts']).reset_index(drop=True)

# 1. Stagnant: rent == 0 and ret == 0 while stock >= 1
df['is_stagnant'] = ((df['stock_min'] >= 1) & (df['rent'] == 0) & (df['ret'] == 0)).astype(int)

# Continuous stagnant episodes across ALL stations
df['stag_block'] = (df['is_stagnant'] != df.groupby('node')['is_stagnant'].shift(1)).cumsum()
stag_episodes = df[df['is_stagnant'] == 1].groupby(['node', 'stag_block']).agg(
    duration=('ts', 'count'),
    mean_stock=('stock_mean', 'mean')
).reset_index()

print("="*70)
print(" [통계 1] 자전거가 대여소에 멈춰서 단 1대도 안 움직인 시간 (무변동 에피소드 전수)")
print("="*70)
print(f"총 발생 건수: {len(stag_episodes):,}건")
print(f"평균 지속 시간: {stag_episodes['duration'].mean():.1f}시간")
print(f"중앙값(Median): {stag_episodes['duration'].median():.1f}시간")
print(f"12시간(반나절) 이상 멈춤: {(stag_episodes['duration'] >= 12).mean()*100:.1f}% ({sum(stag_episodes['duration'] >= 12):,}건)")
print(f"24시간(1일) 이상 멈춤: {(stag_episodes['duration'] >= 24).mean()*100:.1f}% ({sum(stag_episodes['duration'] >= 24):,}건)")
print(f"48시간(2일) 이상 멈춤: {(stag_episodes['duration'] >= 48).mean()*100:.1f}% ({sum(stag_episodes['duration'] >= 48):,}건)")
print(f"72시간(3일) 이상 멈춤: {(stag_episodes['duration'] >= 72).mean()*100:.1f}% ({sum(stag_episodes['duration'] >= 72):,}건)")
print(f"최장 기록: {stag_episodes['duration'].max():.0f}시간 ({stag_episodes['duration'].max()/24:.1f}일)")

# 2. Daily average idle hours per station
station_idle = df.groupby('node').agg(
    total_stagnant_hours=('is_stagnant', 'sum'),
    total_hours=('ts', 'count'),
    avg_stock=('stock_mean', 'mean')
).reset_index()
station_idle['idle_ratio'] = station_idle['total_stagnant_hours'] / station_idle['total_hours'] * 100
station_idle['daily_idle_hours'] = station_idle['total_stagnant_hours'] / 30.0

print("\n" + "="*70)
print(" [통계 2] 대전시 대여소 1곳당 하루 평균 자전거 멈춤(유휴) 시간")
print("="*70)
print(f"대여소 1곳당 하루 평균 멈춤 시간: {station_idle['daily_idle_hours'].mean():.1f}시간 / 24시간 (하루의 {station_idle['idle_ratio'].mean():.1f}%)")
print(f"하루 18시간 이상 멈춰있는 대여소 비율: {(station_idle['daily_idle_hours'] >= 18).mean()*100:.1f}% (전체의 {(station_idle['daily_idle_hours'] >= 18).sum()}개소)")
print(f"하루 12시간 이상 멈춰있는 대여소 비율: {(station_idle['daily_idle_hours'] >= 12).mean()*100:.1f}% (전체의 {(station_idle['daily_idle_hours'] >= 12).sum()}개소)")

# 3. High stock (Over-capacity >= 30 bikes) duration
df['is_overflow_30'] = (df['stock_min'] >= 30).astype(int)
df['over_block'] = (df['is_overflow_30'] != df.groupby('node')['is_overflow_30'].shift(1)).cumsum()
over_episodes = df[df['is_overflow_30'] == 1].groupby(['node', 'over_block']).agg(
    duration=('ts', 'count')
).reset_index()

if len(over_episodes) > 0:
    print("\n" + "="*70)
    print(" [통계 3] 30대 이상 대량 과적(포화) 상태가 지속된 시간")
    print("="*70)
    print(f"평균 지속 시간: {over_episodes['duration'].mean():.1f}시간")
    print(f"최장 지속 시간: {over_episodes['duration'].max():.0f}시간 ({over_episodes['duration'].max()/24:.1f}일간 풀가동 포화)")
