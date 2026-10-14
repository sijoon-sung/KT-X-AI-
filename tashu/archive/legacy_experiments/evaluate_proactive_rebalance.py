"""출근 피크 선제적 사전 최적 배치(Proactive Pre-allocation) 종합 검증 및 대전시 황금 배치표 산출 스크립트.

포함 내용:
1. 관측 거래(Observed) 및 복원된 잠재 수요(Latent Demand) 하에서의 시뮬레이션 맞비교.
2. 만차 방치(Overflow) 감축률 검증 (거치대 여유 공간 최적 확보 효과).
3. 대전시 1,425개 대여소별 '새벽 6시 황금 초기 재고 분배표' 및 만성 결손/잉여 대여소 리포트.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from kinetic_flux_demand import DEV, load_dataset
from neural_flux_operator import HighDimKineticModel
from proactive_optimizer import (
    compute_cumulative_flux,
    simulate_commute_peak,
    solve_golden_allocation,
)


def evaluate_weekdays_rebalance(model, data, start_date, end_date, capacities):
    """
    지정된 기간 내 모든 평일(월~금) 06:00 시점에 대해,
    실제 대전시 배치(Actual) vs AI 황금 사전 배치(Golden)의 출근 4시간 피크(07~11시) 시뮬레이션 비교.
    """
    bin_start = data['bin_start']
    n_stations = len(data['stations'])

    days = bin_start.astype('datetime64[D]')
    hours = (bin_start - days).astype('timedelta64[h]').astype(int)
    weekdays = (days.astype(int) + 3) % 7

    in_period = (bin_start >= np.datetime64(start_date)) & (bin_start < np.datetime64(end_date))
    six_am_mask = in_period & (hours == 6) & (weekdays < 5) & data['valid_hours']
    six_am_indices = np.flatnonzero(six_am_mask)

    if len(six_am_indices) == 0:
        return None

    total_actual_unmet_obs = 0.0
    total_golden_unmet_obs = 0.0
    total_actual_unmet_latent = 0.0
    total_golden_unmet_latent = 0.0
    total_actual_overflow = 0.0
    total_golden_overflow = 0.0
    total_latent_demand = 0.0

    cumulative_recommended_alloc = np.zeros(n_stations, dtype=np.float64)
    cumulative_actual_alloc = np.zeros(n_stations, dtype=np.float64)
    evaluated_days_count = 0

    for idx_6am in six_am_indices:
        commute_hours = [idx_6am + 1, idx_6am + 2, idx_6am + 3, idx_6am + 4]
        if commute_hours[-1] >= len(bin_start) or not np.all(data['valid_hours'][commute_hours]):
            continue

        active = data['active'][idx_6am] & data['station_ok']
        actual_inv_6am = data['inv_start'][idx_6am].astype(np.float32) * active.astype(np.float32)
        total_bikes = float(actual_inv_6am.sum())
        if total_bikes < 100:
            continue

        # 1. AI 4시간 연속 유체 플럭스 적분
        flux_info = compute_cumulative_flux(model, data, commute_hours)

        # 2. 동일한 총 자전거 예산 하에서의 황금 초기 배치 산출
        golden_inv_6am = solve_golden_allocation(
            flux_info['d_peak'], flux_info['s_peak'],
            total_bikes=total_bikes,
            capacities=capacities,
            active_mask=active,
            safety_margin=2.0
        )

        actual_rent = data['rentals'][commute_hours].astype(np.float32)
        actual_ret = data['returns'][commute_hours].astype(np.float32)
        exp = data['exposure'][commute_hours]

        # 3. 시뮬레이션 1: 관측된 실현 대여량 기준
        sim_act_obs = simulate_commute_peak(actual_inv_6am, actual_rent, actual_ret, capacities, active)
        sim_gld_obs = simulate_commute_peak(golden_inv_6am, actual_rent, actual_ret, capacities, active)

        # 4. 시뮬레이션 2: 화학 흡착식으로 복원된 진짜 잠재 수요 기준 (재고 0으로 가려졌던 수요 포함)
        latent_rent = np.where(exp >= 0.9, actual_rent,
                               np.maximum(actual_rent, flux_info['hourly_lambda']))
        sim_act_lat = simulate_commute_peak(actual_inv_6am, latent_rent, actual_ret, capacities, active)
        sim_gld_lat = simulate_commute_peak(golden_inv_6am, latent_rent, actual_ret, capacities, active)

        total_actual_unmet_obs += sim_act_obs['total_unmet_demand']
        total_golden_unmet_obs += sim_gld_obs['total_unmet_demand']
        total_actual_unmet_latent += sim_act_lat['total_unmet_demand']
        total_golden_unmet_latent += sim_gld_lat['total_unmet_demand']
        total_actual_overflow += sim_act_obs['total_overflow']
        total_golden_overflow += sim_gld_obs['total_overflow']
        total_latent_demand += float((latent_rent * active.astype(np.float32)).sum())

        cumulative_recommended_alloc += golden_inv_6am
        cumulative_actual_alloc += actual_inv_6am
        evaluated_days_count += 1

    latent_unmet_reduction = (1.0 - (total_golden_unmet_latent / max(total_actual_unmet_latent, 1.0))) * 100.0
    overflow_reduction = (1.0 - (total_golden_overflow / max(total_actual_overflow, 1.0))) * 100.0

    mean_golden = cumulative_recommended_alloc / max(evaluated_days_count, 1)
    mean_actual = cumulative_actual_alloc / max(evaluated_days_count, 1)
    diff = mean_golden - mean_actual

    starved_indices = np.argsort(-diff)[:10]
    wasted_indices = np.argsort(diff)[:10]

    return {
        'evaluated_days': evaluated_days_count,
        'overall': {
            'total_latent_demand': total_latent_demand,
            'actual_unmet_latent_demand': total_actual_unmet_latent,
            'golden_unmet_latent_demand': total_golden_unmet_latent,
            'latent_unmet_reduction_pct': latent_unmet_reduction,
            'actual_overflow': total_actual_overflow,
            'golden_overflow': total_golden_overflow,
            'overflow_reduction_pct': overflow_reduction,
        },
        'top_starved_stations': [
            {'idx': int(i), 'mean_actual': float(mean_actual[i]),
             'mean_golden': float(mean_golden[i]), 'needed_increase': float(diff[i])}
            for i in starved_indices
        ],
        'top_wasted_stations': [
            {'idx': int(i), 'mean_actual': float(mean_actual[i]),
             'mean_golden': float(mean_golden[i]), 'unnecessary_excess': float(-diff[i])}
            for i in wasted_indices
        ],
        'golden_allocation_table': mean_golden.tolist()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/processed/tashu_60min.npz')
    parser.add_argument('--model', default='data/neural_flux_model.pt')
    parser.add_argument('--out_json', default='data/proactive_rebalance_evaluation.json')
    args = parser.parse_args()

    print(f"Loading dataset from {args.data}...")
    data = load_dataset(args.data)
    n_stations = len(data['stations'])

    inv = data['inv_start']
    p99 = np.percentile(inv, 99.5, axis=0)
    capacities = np.clip(np.maximum(15.0, p99), 15.0, 50.0).astype(np.float32)

    print(f"Loading Neural-KFDM model from {args.model}...")
    ckpt = torch.load(args.model, map_location=DEV, weights_only=False)
    od_path = Path('data/processed/od_train_before_2026-02-01.npy')
    od = np.load(od_path) if od_path.exists() else data['od']
    model = HighDimKineticModel(n_stations, od, k=32, d_model=32).to(DEV)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    results = {}

    print("Simulating February 2026 Weekday Morning Commute Peaks (07:00 ~ 11:00)...")
    results['february_2026'] = evaluate_weekdays_rebalance(
        model, data, '2026-02-01', '2026-03-01', capacities
    )

    print("Simulating March 2026 Weekday Morning Commute Peaks (07:00 ~ 11:00)...")
    results['march_2026'] = evaluate_weekdays_rebalance(
        model, data, '2026-03-01', '2026-04-01', capacities
    )

    with open('data/processed/stations.csv', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        st_meta = {int(r['idx']): f"{r['station_id']} {r['name']} ({r['gu']})" for r in reader}

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Saved evaluation to {args.out_json}")

    print("\n" + "=" * 78)
    print("      출근 피크(07~11시) 선제적 사전 최적 배치(Proactive Golden Allocation) 결과")
    print("=" * 78)
    for period in ('february_2026', 'march_2026'):
        res = results[period]
        if not res:
            continue
        ov = res['overall']
        print(f"\n[{period.upper()}] ({res['evaluated_days']}개 평일 출근 피크 시뮬레이션)")
        print(f" - 총 복원된 출근 잠재 수요: {ov['total_latent_demand']:,.0f} 건")
        print(f" - 현실 대전시 실제 잠재 결손: {ov['actual_unmet_latent_demand']:,.0f} 건")
        print(f" - AI 황금 사전 배치 결손    : {ov['golden_unmet_latent_demand']:,.0f} 건")
        print(f" ▶ 진짜 잠재 수요 결손 감축률: 【 -{ov['latent_unmet_reduction_pct']:.1f}% 감축 】 "
              f"(시민 {ov['actual_unmet_latent_demand'] - ov['golden_unmet_latent_demand']:,.0f}명의 탑승 기회 구출)")
        print(f" ▶ 거치대 만차 방치 감축률    : 【 -{ov['overflow_reduction_pct']:.1f}% 감축 】 "
              f"(도착지 거치대 부족으로 인한 노상 방치 원천 차단)")

        print(f"\n [새벽 6시에 자전거가 가장 절실했던 만성 품절 대여소 Top 5]")
        for item in res['top_starved_stations'][:5]:
            info = st_meta.get(item['idx'], f"idx {item['idx']}")
            print(f"   · {info}: 실제 평균 {item['mean_actual']:.1f}대 -> 권장 {item['mean_golden']:.1f}대 (+{item['needed_increase']:.1f}대 보충 필요)")

        print(f"\n [새벽 6시에 자전거가 헛되이 방치되었던 만성 잉여 대여소 Top 5]")
        for item in res['top_wasted_stations'][:5]:
            info = st_meta.get(item['idx'], f"idx {item['idx']}")
            print(f"   · {info}: 실제 평균 {item['mean_actual']:.1f}대 -> 권장 {item['mean_golden']:.1f}대 (-{item['unnecessary_excess']:.1f}대 회수 필요)")

    print("=" * 78)


if __name__ == '__main__':
    main()
