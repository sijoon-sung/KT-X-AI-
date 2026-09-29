import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path("C:/Users/sijoo/Documents/tashu")
trips = pd.read_parquet(ROOT / "data/processed/trips.parquet", 
                        columns=['bike_id', 'rent_ts', 'ret_ts', 'rent_st', 'ret_st', 'dur_min', 'dist_km'])

# 1. 튕김(Bounce) 정의: 3분 이내 반납 + 제자리 또는 0.1km 이하
same_st = trips['rent_st'].astype(str) == trips['ret_st'].astype(str)
trips['is_bounce'] = (trips['dur_min'] <= 3) & (same_st | (trips['dist_km'] <= 0.1))

# 자전거별 집계
b_agg = trips.groupby('bike_id').agg(
    n_trips=('is_bounce', 'count'),
    k_bounces=('is_bounce', 'sum'),
    total_km=('dist_km', 'sum'),
    total_min=('dur_min', 'sum')
).reset_index()

# 2. Empirical Bayes (경험적 베이즈)를 통한 Prior 모수 (alpha_0, beta_0) 추정
# 전체 함대의 정상적인 기본 바운스율 분포 피팅
b_agg['raw_rate'] = b_agg['k_bounces'] / b_agg['n_trips']
# 모멘트법(Method of Moments)으로 Beta Prior 모수 추정
mu = b_agg['raw_rate'].mean()
var = b_agg['raw_rate'].var()

# Beta 분포 모수 공식: alpha = mu * (mu*(1-mu)/var - 1)
factor = (mu * (1 - mu) / var) - 1.0
alpha_0 = max(1.0, mu * factor)
beta_0 = max(1.0, (1 - mu) * factor)

print(f"=== [1. 베이지안 사전 분포 (Beta Prior) 모수] ===")
print(f"  - 평균 바운스율 (mu): {mu*100:.2f}%")
print(f"  - 분산 (var): {var:.5f}")
print(f"  - 사전 분포: Beta(alpha_0={alpha_0:.2f}, beta_0={beta_0:.2f})\n")

# 3. 사후 분포 (Beta Posterior) 계산
# Posterior for bike i: Beta(alpha_0 + k, beta_0 + n - k)
b_agg['post_alpha'] = alpha_0 + b_agg['k_bounces']
b_agg['post_beta'] = beta_0 + b_agg['n_trips'] - b_agg['k_bounces']

# 사후 기댓값 (Posterior Mean)
b_agg['post_mean'] = b_agg['post_alpha'] / (b_agg['post_alpha'] + b_agg['post_beta'])

# 95% 신용 구간 (Credible Interval: 2.5% ~ 97.5%)
b_agg['ci_lower'] = stats.beta.ppf(0.025, b_agg['post_alpha'], b_agg['post_beta'])
b_agg['ci_upper'] = stats.beta.ppf(0.975, b_agg['post_alpha'], b_agg['post_beta'])

# 고장 확률: P(theta > 0.25 | Data) -> 바운스율이 25%를 넘을 사후 확률
b_agg['prob_broken'] = 1.0 - stats.beta.cdf(0.25, b_agg['post_alpha'], b_agg['post_beta'])

print("=== [2. 사후 고장 확률 상위 15대 (긴급 정비 수거 대상)] ===")
top_faulty = b_agg.sort_values('prob_broken', ascending=False).head(15)
cols = ['bike_id', 'n_trips', 'k_bounces', 'raw_rate', 'post_mean', 'ci_lower', 'ci_upper', 'prob_broken']
df_show = top_faulty[cols].copy()
df_show['raw_rate'] = (df_show['raw_rate'] * 100).round(1).astype(str) + '%'
df_show['post_mean'] = (df_show['post_mean'] * 100).round(1).astype(str) + '%'
df_show['95% CI'] = df_show['ci_lower'].apply(lambda x: f"{x*100:.1f}%") + " ~ " + df_show['ci_upper'].apply(lambda x: f"{x*100:.1f}%")
df_show['고장판정확률'] = (df_show['prob_broken'] * 100).round(2).astype(str) + '%'

print(df_show[['bike_id', 'n_trips', 'k_bounces', 'post_mean', '95% CI', '고장판정확률']].to_string(index=False))

# 4. 전체 함대 내 고장 확정 자전거 통계
critically_broken = b_agg[b_agg['prob_broken'] >= 0.95]
suspicious = b_agg[(b_agg['prob_broken'] >= 0.50) & (b_agg['prob_broken'] < 0.95)]

print(f"\n=== [3. 타슈 전체 6,776대 베이지안 진단 결과] ===")
print(f"  - [긴급 수거 확정] 고장 확률 >= 95% : {len(critically_broken):,} 대 ({len(critically_broken)/len(b_agg)*100:.2f}%)")
print(f"  - [주의/점검 필요] 고장 확률 50~95% : {len(suspicious):,} 대 ({len(suspicious)/len(b_agg)*100:.2f}%)")
print(f"  - [정상 운행 상태] 고장 확률 < 50%   : {len(b_agg)-len(critically_broken)-len(suspicious):,} 대")

# 5. 연속 튕김(Consecutive Bounce) 실시간 베이즈 업데이트 증명
print("\n=== [4. 실시간 연속 바운스 발생 시 베이지안 사후 확률 갱신 전이] ===")
p_healthy = 0.047
p_broken = 0.70 # 고장 시 바운스 확률
prior_broken = 0.03 # 전체 함대 기본 고장 비율 (3%)

p_b = prior_broken
print(f"  - 대여 전 사전 확률(Prior): P(고장) = {p_b*100:.1f}%")

for bounce_cnt in range(1, 5):
    # Bayes Update: P(B | bounce) = P(bounce | B) * P(B) / P(bounce)
    p_evidence = p_broken * p_b + p_healthy * (1.0 - p_b)
    p_b = (p_broken * p_b) / p_evidence
    print(f"  - 연속 {bounce_cnt}회 즉시 반납 발생 시 사후 확률: P(고장 | {bounce_cnt}회 바운스) = {p_b*100:.2f}%")
