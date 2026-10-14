"""동역학적 유량 모델(KFDM) 학습, 2026년 2·3월 평가 및 분포 최적도 진단 스크립트."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kinetic_flux_demand import (
    DEV,
    NON_CUSTOMER,
    KineticFluxDemandModel,
    evaluate_distribution_optimality,
    load_dataset,
    predict_kinetic_hour,
    train_kinetic_model,
)


def evaluate_period(model, data, start, end, name="period"):
    """특정 기간에 대해 대여, 반납, 총량 및 분포 최적도를 종합 평가한다."""
    test_mask = (
        data['valid_hours']
        & (data['bin_start'] >= np.datetime64(start))
        & (data['bin_start'] < np.datetime64(end))
    )
    test_times = np.flatnonzero(test_mask)
    if len(test_times) == 0:
        return None

    total_pred_rent = 0.0
    total_actual_rent = 0.0
    total_abs_error_rent = 0.0
    full_exposure_cells = 0

    total_pred_return = 0.0
    total_actual_return = 0.0
    total_abs_error_return = 0.0
    return_cells = 0

    hourly_total_errors = []
    optimality_scores = []
    total_deficits = []
    total_excesses = []

    # 샘플 진단용 시간대 (아침 출근 8시, 퇴근 18시)
    commute_diagnostics = []

    for t in test_times:
        pred = predict_kinetic_hour(model, data, t)
        pred_rent = pred['lambda_rent']
        pred_return = pred['mu_return']

        mask = data['active'][t] & data['station_ok']
        full_exp = mask & (data['exposure'][t] == 1.0)

        # 1. 완전 노출 대여 평가 (정답이 절단되지 않은 셀)
        if full_exp.any():
            act_r = data['rentals'][t, full_exp].astype(np.float64)
            p_r = pred_rent[full_exp].astype(np.float64)
            total_pred_rent += p_r.sum()
            total_actual_rent += act_r.sum()
            total_abs_error_rent += np.abs(p_r - act_r).sum()
            full_exposure_cells += int(full_exp.sum())

        # 2. 반납 평가 (활성 대여소 전체)
        if mask.any():
            act_ret = data['returns'][t, mask].astype(np.float64)
            p_ret = pred_return[mask].astype(np.float64)
            total_pred_return += p_ret.sum()
            total_actual_return += act_ret.sum()
            total_abs_error_return += np.abs(p_ret - act_ret).sum()
            return_cells += int(mask.sum())

            # 시간당 총량 오차 (완전 노출 기준)
            hourly_total_errors.append(abs(p_r.sum() - act_r.sum()) if full_exp.any() else 0.0)

        # 3. 분포 최적도 진단 (실제 시작 재고 inv_start 대조)
        inv = data['inv_start'][t].astype(np.float32)
        opt_res = evaluate_distribution_optimality(pred_rent, pred_return, inv)
        optimality_scores.append(opt_res['optimality_score'])
        total_deficits.append(opt_res['expected_deficit'])
        total_excesses.append(opt_res['expected_excess'])

        # 대표 출퇴근 시간 기록
        h = data['hour'][t]
        if h in (8, 18) and len(commute_diagnostics) < 4:
            commute_diagnostics.append({
                'time': str(data['bin_start'][t]),
                'hour': int(h),
                'macro_demand': float(pred['macro_total']),
                'optimality_score': float(opt_res['optimality_score']),
                'expected_deficit': float(opt_res['expected_deficit']),
                'expected_excess': float(opt_res['expected_excess']),
                'top_rebalance_in': [
                    int(idx) for idx in np.argsort(-opt_res['rebalance_priority'])[:5]
                ],
                'top_rebalance_out': [
                    int(idx) for idx in np.argsort(opt_res['rebalance_priority'])[:5]
                ],
            })

    mae_rent = total_abs_error_rent / max(full_exposure_cells, 1)
    ratio_rent = total_pred_rent / max(total_actual_rent, 1)

    mae_return = total_abs_error_return / max(return_cells, 1)
    ratio_return = total_pred_return / max(total_actual_return, 1)

    return {
        'hours': len(test_times),
        'rental_full_exposure': {
            'cells': full_exposure_cells,
            'predicted': float(total_pred_rent),
            'actual': float(total_actual_rent),
            'abs_error': float(total_abs_error_rent),
            'mae': float(mae_rent),
            'pred_actual_ratio': float(ratio_rent),
        },
        'return_all_active': {
            'cells': return_cells,
            'predicted': float(total_pred_return),
            'actual': float(total_actual_return),
            'abs_error': float(total_abs_error_return),
            'mae': float(mae_return),
            'pred_actual_ratio': float(ratio_return),
        },
        'hourly_total_mae': float(np.mean(hourly_total_errors)),
        'distribution_optimality': {
            'mean_optimality_score': float(np.mean(optimality_scores)),
            'mean_hourly_deficit': float(np.mean(total_deficits)),
            'mean_hourly_excess': float(np.mean(total_excesses)),
        },
        'commute_samples': commute_diagnostics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/processed/tashu_60min.npz')
    parser.add_argument('--cutoff', default='2026-02-01')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_hours', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--out_model', default='data/kinetic_flux_demand.pt')
    parser.add_argument('--out_json', default='data/kinetic_flux_evaluation.json')
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    data = load_dataset(args.data)

    model_path = Path(args.out_model)
    if model_path.exists():
        print(f"Found existing model at {model_path}, loading...")
        ckpt = torch.load(model_path, map_location=DEV, weights_only=False)
        n_stations = len(data['stations'])
        od_path = Path('data/processed/od_train_before_2026-02-01.npy')
        od = np.load(od_path) if od_path.exists() else data['od']
        model = KineticFluxDemandModel(n_stations, od).to(DEV)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
    else:
        print(f"Training KineticFluxDemandModel before {args.cutoff}...")
        model = train_kinetic_model(
            data, cutoff=args.cutoff, epochs=args.epochs,
            batch_hours=args.batch_hours, lr=args.lr
        )
        torch.save({'state_dict': model.state_dict(),
                    'stations': data['stations'].astype(str).tolist()}, model_path)
        print(f"Saved model to {model_path}")

    results = {'cutoff': args.cutoff, 'model': 'KineticFluxDemandModel'}

    print("Evaluating February 2026 (Winter holdout)...")
    results['february_2026'] = evaluate_period(model, data, '2026-02-01', '2026-03-01', 'Feb')

    print("Evaluating March 2026 (Spring holdout)...")
    results['march_2026'] = evaluate_period(model, data, '2026-03-01', '2026-04-01', 'Mar')

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Saved evaluation results to {args.out_json}")

    # Summary table
    print("\n" + "=" * 65)
    print("           동역학적 유량 모델(KFDM) 종합 평가 결과")
    print("=" * 65)
    for period_name in ('february_2026', 'march_2026'):
        p = results[period_name]
        if not p:
            continue
        print(f"\n[{period_name.upper()}] ({p['hours']} 시간)")
        print(f" - 대여 MAE (완전 노출)   : {p['rental_full_exposure']['mae']:.4f}")
        print(f" - 대여 예측/실제 비율     : {p['rental_full_exposure']['pred_actual_ratio']:.4f} "
              f"({p['rental_full_exposure']['predicted']:.0f} / {p['rental_full_exposure']['actual']:.0f})")
        print(f" - 반납 MAE (이류 보존)   : {p['return_all_active']['mae']:.4f}")
        print(f" - 시간당 총량 오차 MAE   : {p['hourly_total_mae']:.2f} 건")
        print(f" - 평균 분포 최적도 점수  : {p['distribution_optimality']['mean_optimality_score']:.2f}%")
        print(f" - 시간당 평균 품절 결손  : {p['distribution_optimality']['mean_hourly_deficit']:.2f} 대")
        print(f" - 시간당 평균 만차 잉여  : {p['distribution_optimality']['mean_hourly_excess']:.2f} 대")
    print("=" * 65)


if __name__ == '__main__':
    main()
