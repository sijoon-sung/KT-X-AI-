"""Proactive Pre-allocation Optimizer (선제적 사전 최적 배치 엔진).

핵심 기능:
1. 다중 시간 누적 유체 적분 (Cumulative Flux Integration):
   - 출근 피크 구간(07:00 ~ 11:00) 4시간 동안의 잠재 수요 Lambda_rent와 반납 이류 mu_return을 적분.
   - 대여소별 최대 누적 고갈량(D_peak)과 최대 누적 유입량(S_peak)을 연속 방정식으로 산출.
2. 수력학적 수위 배분 (Water-Filling Golden Allocation):
   - 도시 전체 자전거 M_total(약 2,000~3,500대) 제약 하에서, 전 대여소의 품절/만차 손실을
     최소화하는 새벽 06:00의 이상적 초기 재고 벡터 I_0*를 볼츠만 등위 배분으로 도출.
3. 출근 피크 현실 시뮬레이터 (Commute Peak Simulator):
   - 실제 Ground Truth 대여/반납 데이터로 시뮬레이션을 실행하여:
     A) 실제 대전시 배치(Actual) vs B) AI 황금 배치(Golden Pre-allocation)의
     실제 품절 발생 건수(Unmet Demand)와 만차 발생 건수(Overflow)를 정밀 비교.
"""
import numpy as np
import torch

from kinetic_flux_demand import DEV
from neural_flux_operator import predict_neural_hour


def compute_cumulative_flux(model, data, time_indices):
    """
    주어진 시간 인덱스들(예: 07시, 08시, 09시, 10시)에 대해
    AI 모델을 실행하고, 대여소별 누적 유체 플럭스(D_peak, S_peak)를 적분한다.
    """
    n_stations = len(data['stations'])
    T = len(time_indices)

    all_lambda = np.zeros((T, n_stations), dtype=np.float32)
    all_mu = np.zeros((T, n_stations), dtype=np.float32)

    for step, t in enumerate(time_indices):
        pred = predict_neural_hour(model, data, t)
        all_lambda[step] = pred['lambda_rent']
        all_mu[step] = pred['mu_return']

    # 시간대별 순유출입 (Net Flux = Demand - Return)
    net_flux = all_lambda - all_mu  # [T, N]

    # 시간 경과에 따른 누적 유량 적분곡선: cum_flux[t, i] = sum_{tau=0}^t net_flux[tau, i]
    cum_flux = np.cumsum(net_flux, axis=0)  # [T, N]

    # 1. 최대 누적 고갈량 (Peak Drawdown = 최소 필요 완충 재고)
    # cum_flux가 양수로 치솟는 최대 지점
    d_peak = np.maximum(0.0, np.max(cum_flux, axis=0))

    # 2. 최대 누적 유입량 (Peak Inflow = 최소 필요 빈 거치 공간)
    # cum_flux가 음수로 떨어지는 최대 지점
    s_peak = np.maximum(0.0, np.max(-cum_flux, axis=0))

    # 4시간 동안의 총 대여 수요
    total_demand = np.sum(all_lambda, axis=0)

    return {
        'd_peak': d_peak,
        's_peak': s_peak,
        'total_demand': total_demand,
        'hourly_lambda': all_lambda,
        'hourly_mu': all_mu,
    }


def solve_golden_allocation(d_peak, s_peak, total_bikes, capacities,
                            active_mask, safety_margin=2.0):
    """
    수력학적 등위 배분 (Water-Filling Optimization):
    총 보유 자전거 total_bikes 제약 하에서, 4시간 동안 품절과 만차가 발생하지 않는
    새벽 6시의 최적 초기 재고 I_0*를 도출한다.
    """
    n = len(d_peak)
    opt_inv = np.zeros(n, dtype=np.float32)

    # 비활성 대여소 제외
    valid = active_mask & (capacities > 0)
    if not valid.any():
        return opt_inv

    # 1. 대여소별 이상적 완충 요구량 계산
    # L_i: 품절을 막기 위한 최소 필요 시작 재고 (누적 고갈량 + 안전 마진)
    # U_i: 만차를 막기 위한 최대 허용 시작 재고 (용량 - 누적 유입량 - 안전 마진)
    L = np.zeros(n, dtype=np.float32)
    U = np.zeros(n, dtype=np.float32)

    for i in np.flatnonzero(valid):
        L[i] = min(capacities[i], d_peak[i] + safety_margin if d_peak[i] > 0.5 else 1.0)
        u_val = capacities[i] - (s_peak[i] + safety_margin if s_peak[i] > 0.5 else 0.0)
        U[i] = max(L[i], min(capacities[i], u_val))

    # 2. 우선순위 가중치 기반 수위 배분 (Water-filling via binary search)
    # 목표: sum(I_0) == total_bikes, 0 <= I_0 <= capacities
    # 각 대여소의 기본 요구량 합산
    sum_L = float(L.sum())

    if total_bikes >= sum_L:
        # 자전거가 충분한 경우: 모든 대여소의 최소 필요 재고 L을 보장하고,
        # 남는 잉여 자전거를 유동 규모(total_demand)와 수용 여력(U - L)에 비례 배분
        surplus = total_bikes - sum_L
        headroom = np.maximum(0.0, U - L)
        sum_headroom = float(headroom.sum())
        if sum_headroom > 0:
            alloc_ratio = min(1.0, surplus / sum_headroom)
            opt_inv = L + alloc_ratio * headroom
            # 그래도 남으면 용량(capacities - opt_inv)에 비례해 균등 배분
            rem = total_bikes - float(opt_inv.sum())
            if rem > 0:
                rem_room = np.maximum(0.0, capacities - opt_inv)
                if rem_room.sum() > 0:
                    opt_inv += (rem / rem_room.sum()) * rem_room
        else:
            opt_inv = L.copy()
    else:
        # 자전거가 부족한 극단적 결손 상황:
        # 고갈 위험(d_peak)이 가장 큰 대여소부터 수위를 채워나가는 Water-filling
        low, high = -10.0, 50.0
        for _ in range(50):
            mid = (low + high) / 2.0
            # I_i = clip(d_peak - mid, 0, L_i)
            test_alloc = np.clip(L - mid, 0.0, capacities) * valid.astype(np.float32)
            if test_alloc.sum() > total_bikes:
                low = mid
            else:
                high = mid
        opt_inv = np.clip(L - (low + high) / 2.0, 0.0, capacities) * valid.astype(np.float32)

    # 반올림 및 정수 보정 (총합 일치 보장)
    opt_inv = np.round(opt_inv).astype(np.float32)
    diff = int(total_bikes - opt_inv.sum())
    if diff != 0:
        # 남거나 모자란 대수를 우선순위 상위 대여소에 1대씩 조정
        prio = np.argsort(-d_peak) if diff > 0 else np.argsort(d_peak)
        for idx in prio:
            if diff == 0:
                break
            if diff > 0 and opt_inv[idx] < capacities[idx] and valid[idx]:
                opt_inv[idx] += 1
                diff -= 1
            elif diff < 0 and opt_inv[idx] > 0 and valid[idx]:
                opt_inv[idx] -= 1
                diff += 1

    return opt_inv


def simulate_commute_peak(initial_inv, actual_rentals, actual_returns,
                          capacities, active_mask):
    """
    실제 4시간 동안의 대여·반납 트랜잭션을 바탕으로 한 시뮬레이터.

    매개변수:
        initial_inv: [N] 06:00 시작 재고
        actual_rentals: [T, N] 시간별 실제 대여 요청량
        actual_returns: [T, N] 시간별 실제 반납 유입량
        capacities: [N] 대여소 거치 용량
        active_mask: [N] 활성 대여소 마스크

    반환값:
        dict:
            total_unmet_demand: 재고 부족으로 대여 실패한 총 건수
            total_overflow: 거치대 부족으로 만차 방치된 총 건수
            total_served_demand: 성공적으로 대여된 총 건수
            service_rate: 서비스 성공률 (Served / (Served + Unmet) * 100%)
            hourly_unmet: [T] 시간별 품절 건수
    """
    T, n = actual_rentals.shape
    current_inv = initial_inv.copy().astype(np.float32)

    total_unmet = 0.0
    total_overflow = 0.0
    total_served = 0.0
    hourly_unmet = []

    for t in range(T):
        r_req = actual_rentals[t] * active_mask.astype(np.float32)
        r_ret = actual_returns[t] * active_mask.astype(np.float32)

        # 1. 대여 서비스 처리
        served = np.minimum(current_inv, r_req)
        unmet = np.maximum(0.0, r_req - current_inv)

        total_served += float(served.sum())
        total_unmet += float(unmet.sum())
        hourly_unmet.append(float(unmet.sum()))

        # 대여 후 잔여 재고
        inv_after_rent = current_inv - served

        # 2. 반납 수용 처리
        free_docks = np.maximum(0.0, capacities - inv_after_rent)
        docked = np.minimum(free_docks, r_ret)
        overflow = np.maximum(0.0, r_ret - free_docks)

        total_overflow += float(overflow.sum())

        # 3. 다음 시간 시작 재고
        current_inv = inv_after_rent + docked

    total_requests = total_served + total_unmet
    service_rate = (total_served / max(total_requests, 1.0)) * 100.0

    return {
        'total_unmet_demand': total_unmet,
        'total_overflow': total_overflow,
        'total_served_demand': total_served,
        'service_rate': service_rate,
        'hourly_unmet': hourly_unmet,
        'final_inventory': current_inv,
    }
