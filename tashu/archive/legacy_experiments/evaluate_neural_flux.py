"""고차원 신경망 유속 모델(Neural-KFDM) 학습 및 2026년 2·3월 맞비교 평가 스크립트."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kinetic_flux_demand import (
    DEV,
    evaluate_distribution_optimality,
    load_dataset,
)
from neural_flux_operator import (
    HighDimKineticModel,
    predict_neural_hour,
    train_neural_flux_model,
)


def evaluate_neural_period(model, data, start, end):
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

    for t in test_times:
        pred = predict_neural_hour(model, data, t)
        pred_rent = pred['lambda_rent']
        pred_return = pred['mu_return']

        mask = data['active'][t] & data['station_ok']
        full_exp = mask & (data['exposure'][t] == 1.0)

        # 1. 완전 노출 대여 평가
        if full_exp.any():
            act_r = data['rentals'][t, full_exp].astype(np.float64)
            p_r = pred_rent[full_exp].astype(np.float64)
            total_pred_rent += p_r.sum()
            total_actual_rent += act_r.sum()
            total_abs_error_rent += np.abs(p_r - act_r).sum()
            full_exposure_cells += int(full_exp.sum())

        # 2. 반납 평가 (동적 신경망 유속장에 의한 전파)
        if mask.any():
            act_ret = data['returns'][t, mask].astype(np.float64)
            p_ret = pred_return[mask].astype(np.float64)
            total_pred_return += p_ret.sum()
            total_actual_return += act_ret.sum()
            total_abs_error_return += np.abs(p_ret - act_ret).sum()
            return_cells += int(mask.sum())

            hourly_total_errors.append(abs(p_r.sum() - act_r.sum()) if full_exp.any() else 0.0)

        # 3. 분포 최적도 진단
        inv = data['inv_start'][t].astype(np.float32)
        opt_res = evaluate_distribution_optimality(pred_rent, pred_return, inv)
        optimality_scores.append(opt_res['optimality_score'])
        total_deficits.append(opt_res['expected_deficit'])
        total_excesses.append(opt_res['expected_excess'])

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
    }


def analyze_directional_reversal(model, data):
    """아침 8시(출근)와 저녁 18시(퇴근)의 신경망 유속장(P_dyn) 방향 변화를 분석한다."""
    h_morn = torch.tensor([8], dtype=torch.long, device=DEV)
    h_eve = torch.tensor([18], dtype=torch.long, device=DEV)
    w = torch.tensor([1], dtype=torch.long, device=DEV)  # 화요일
    m = torch.tensor([2], dtype=torch.long, device=DEV)  # 3월

    P_morn = model.advection_op.compute_dynamic_transition(h_morn, w, m).squeeze(0).detach().cpu().numpy()
    P_eve = model.advection_op.compute_dynamic_transition(h_eve, w, m).squeeze(0).detach().cpu().numpy()

    # 각 대여소별 아침과 저녁 전이 분포의 차이 (L1 distance)
    diff = np.abs(P_morn - P_eve).sum(axis=-1)
    mean_diff = float(diff.mean())
    top_divergent = np.argsort(-diff)[:5]

    return {
        'mean_directional_shift_l1': mean_diff,
        'top_shift_station_indices': [int(x) for x in top_divergent],
        'top_shift_values': [float(diff[x]) for x in top_divergent],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/processed/tashu_60min.npz')
    parser.add_argument('--cutoff', default='2026-02-01')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch_hours', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--k', type=int, default=32)
    parser.add_argument('--out_model', default='data/neural_flux_model.pt')
    parser.add_argument('--out_json', default='data/neural_flux_evaluation.json')
    args = parser.parse_args()

    print(f"Loading dataset from {args.data}...")
    data = load_dataset(args.data)

    model_path = Path(args.out_model)
    if model_path.exists():
        print(f"Loading existing model from {model_path}...")
        ckpt = torch.load(model_path, map_location=DEV, weights_only=False)
        n_stations = len(data['stations'])
        od_path = Path('data/processed/od_train_before_2026-02-01.npy')
        od = np.load(od_path) if od_path.exists() else data['od']
        model = HighDimKineticModel(n_stations, od, k=args.k, d_model=32).to(DEV)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
    else:
        print(f"Training HighDimKineticModel (Neural-KFDM) before {args.cutoff}...")
        model = train_neural_flux_model(
            data, cutoff=args.cutoff, epochs=args.epochs,
            batch_hours=args.batch_hours, lr=args.lr, k=args.k
        )
        torch.save({'state_dict': model.state_dict(),
                    'stations': data['stations'].astype(str).tolist()}, model_path)
        print(f"Saved neural model to {model_path}")

    results = {'cutoff': args.cutoff, 'model': 'HighDimKineticModel (Neural-KFDM)'}

    print("Evaluating February 2026...")
    results['february_2026'] = evaluate_neural_period(model, data, '2026-02-01', '2026-03-01')

    print("Evaluating March 2026...")
    results['march_2026'] = evaluate_neural_period(model, data, '2026-03-01', '2026-04-01')

    print("Analyzing Directional Reversal (Morning 8 AM vs Evening 6 PM)...")
    results['directional_reversal'] = analyze_directional_reversal(model, data)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Saved evaluation results to {args.out_json}")

    # Load static model results for comparison if available
    static_path = Path('data/kinetic_flux_evaluation.json')
    static_res = json.loads(static_path.read_text(encoding='utf-8')) if static_path.exists() else None

    print("\n" + "=" * 70)
    print("      고차원 신경망 유속 모델(Neural-KFDM) vs 정적 물리 모델 맞비교")
    print("=" * 70)
    for period in ('february_2026', 'march_2026'):
        p_neu = results[period]
        p_sta = static_res[period] if static_res else None
        print(f"\n[{period.upper()}]")
        print(f" - 대여 MAE (완전 노출)   : Neural {p_neu['rental_full_exposure']['mae']:.4f} "
              + (f"vs Static {p_sta['rental_full_exposure']['mae']:.4f}" if p_sta else ""))
        print(f" - 반납 MAE (신경망 유속장): Neural {p_neu['return_all_active']['mae']:.4f} "
              + (f"vs Static {p_sta['return_all_active']['mae']:.4f}" if p_sta else ""))
        print(f" - 시간당 총량 오차 MAE   : Neural {p_neu['hourly_total_mae']:.2f} "
              + (f"vs Static {p_sta['hourly_total_mae']:.2f}" if p_sta else ""))
        print(f" - 평균 분포 최적도 점수  : Neural {p_neu['distribution_optimality']['mean_optimality_score']:.2f}% "
              + (f"vs Static {p_sta['distribution_optimality']['mean_optimality_score']:.2f}%" if p_sta else ""))

    dr = results['directional_reversal']
    print(f"\n[출퇴근 유속장 방향 반전 분석]")
    print(f" - 아침(8시) vs 저녁(18시) 평균 전이 확률 변화 (L1): {dr['mean_directional_shift_l1']:.4f}")
    print(f" - 가장 큰 방향 전환을 보인 대여소 인덱스: {dr['top_shift_station_indices']}")
    print("=" * 70)


if __name__ == '__main__':
    main()
