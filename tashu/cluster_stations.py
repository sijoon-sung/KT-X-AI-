"""Tashu Bicycle Sharing Station Clustering & Functional Profiling.

Categorizes 1,425 Tashu stations into actionable operational clusters based on:
1. Hourly rental and return temporal profiles (morning, daytime, evening rush)
2. Directional Net Flow Asymmetry (morning outflow residential vs morning inflow transit/office)
3. Leisure and circular trip characteristics (self-loop ratio, weekend/weekday ratio)
4. Usage intensity (daily trip volume, stockout frequency)
5. Spatial coordinates (district distribution)

Outputs:
- data/clustering_stations.csv: Per-station cluster labels, metrics, and actions
- data/clustering_summary.json: Cluster profile summaries and operational guidelines
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

NON_CUSTOMER = ('ST0001', 'ST1220', 'ST1198')


def load_data(npz_path, stations_csv_path, train_start, cutoff, min_active=0.90):
    """Loads processed npz data and station metadata."""
    npz = np.load(npz_path, allow_pickle=True)
    stations = npz['stations']
    rentals = npz['rentals']
    returns = npz['returns']
    exposure = npz['exposure']
    active = npz['active']
    valid = npz['valid'] & ~npz['closed']
    bin_start = npz['bin_start']
    lat = npz['lat']
    lon = npz['lon']
    od = npz['od']

    # Station metadata
    st_meta = {}
    if Path(stations_csv_path).exists():
        with open(stations_csv_path, 'r', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                st_meta[r['station_id']] = {
                    'name': r['name'],
                    'gu': r['gu'],
                    'first_seen': r.get('first_seen', ''),
                    'last_seen': r.get('last_seen', '')
                }

    # Training time filter
    train = valid & (bin_start >= np.datetime64(train_start)) & (bin_start < np.datetime64(cutoff))
    if train.sum() < 100:
        raise ValueError(f"Insufficient training data between {train_start} and {cutoff}")

    # Active station filter
    active_rate = active[train].mean(0)
    selected = (active_rate >= min_active) & ~np.isin(stations, NON_CUSTOMER)
    sel_idx = np.where(selected)[0]

    # Calendar extraction
    ts = bin_start.astype('datetime64[h]')
    hour = (ts.astype('int64') % 24)
    # Days since 1970-01-01 (Thursday, weekday=3)
    day = ((ts.astype('int64') // 24) + 3) % 7
    is_weekend = (day >= 5)

    return {
        'npz': npz,
        'stations': stations,
        'st_meta': st_meta,
        'train': train,
        'selected': selected,
        'sel_idx': sel_idx,
        'rentals': rentals,
        'returns': returns,
        'exposure': exposure,
        'hour': hour,
        'is_weekend': is_weekend,
        'lat': lat,
        'lon': lon,
        'od': od
    }


def extract_features(data):
    """Extracts domain-engineered features for clustering."""
    train = data['train']
    sel_idx = data['sel_idx']
    N_sel = len(sel_idx)

    r_train = data['rentals'][train][:, sel_idx]
    ret_train = data['returns'][train][:, sel_idx]
    exp_train = data['exposure'][train][:, sel_idx]
    h_train = data['hour'][train]
    w_train = data['is_weekend'][train]

    # 1. Hourly averages for weekday and weekend
    wd_r = np.zeros((24, N_sel), dtype=np.float32)
    wd_ret = np.zeros((24, N_sel), dtype=np.float32)
    we_r = np.zeros((24, N_sel), dtype=np.float32)
    we_ret = np.zeros((24, N_sel), dtype=np.float32)

    for h in range(24):
        m_wd = (~w_train) & (h_train == h)
        if m_wd.any():
            wd_r[h] = r_train[m_wd].mean(0)
            wd_ret[h] = ret_train[m_wd].mean(0)
        m_we = w_train & (h_train == h)
        if m_we.any():
            we_r[h] = r_train[m_we].mean(0)
            we_ret[h] = ret_train[m_we].mean(0)

    # 2. Total Daily Volume & Log Scale
    n_days = train.sum() / 24.0
    daily_trips = (r_train.sum(0) + ret_train.sum(0)) / n_days
    log_vol = np.log1p(daily_trips)

    # 3. Weekend to Weekday Ratio
    we_wd_ratio = we_r.sum(0) / np.maximum(wd_r.sum(0), 0.05)

    # 4. Net Flow Asymmetry during key windows
    # Morning rush: 07:00 ~ 09:59
    am_rent = wd_r[7:10].sum(0)
    am_ret = wd_ret[7:10].sum(0)
    net_am = (am_ret - am_rent) / np.maximum(am_ret + am_rent, 0.1)

    # Evening rush: 17:00 ~ 19:59
    pm_rent = wd_r[17:20].sum(0)
    pm_ret = wd_ret[17:20].sum(0)
    net_pm = (pm_ret - pm_rent) / np.maximum(pm_ret + pm_rent, 0.1)

    # Daytime share: 11:00 ~ 15:59
    day_rent = wd_r[11:16].sum(0)
    day_share = day_rent / np.maximum(wd_r.sum(0), 0.05)

    # 5. Self-loop Ratio (circular leisure trips)
    od = data['od']
    departures = od[sel_idx].sum(1)
    self_loop = np.diag(od)[sel_idx] / np.maximum(departures, 1.0)

    # 6. Stockout Rate (exposure < 1)
    stockout_rate = 1.0 - exp_train.mean(0)

    # 7. Coordinate centering
    lat_sel = data['lat'][sel_idx]
    lon_sel = data['lon'][sel_idx]
    lat_km = (lat_sel - np.mean(lat_sel)) * 111.0
    lon_km = (lon_sel - np.mean(lon_sel)) * 89.0

    # Hourly shape vectors (normalized 24h proportions)
    r_shape = wd_r / np.maximum(wd_r.sum(0, keepdims=True), 1e-4)
    ret_shape = wd_ret / np.maximum(wd_ret.sum(0, keepdims=True), 1e-4)

    # Feature matrix with domain weights
    feature_matrix = np.column_stack([
        net_am * 3.5,            # High weight on morning net flow
        net_pm * 2.5,            # High weight on evening net flow
        self_loop * 4.0,         # High weight on circular / leisure behavior
        we_wd_ratio * 1.5,       # Weekend leisure concentration
        day_share * 1.5,         # Daytime sustained flow
        log_vol * 1.2,           # Traffic volume scale
        stockout_rate * 1.0,     # Stockout risk
        lat_km * 0.10,           # Subtle spatial cohesion
        lon_km * 0.10,
        r_shape[7:10].sum(0) * 2.0,   # Morning rental peak share
        ret_shape[7:10].sum(0) * 2.0, # Morning return peak share
        r_shape[17:20].sum(0) * 1.5,  # Evening rental peak share
        ret_shape[17:20].sum(0) * 1.5 # Evening return peak share
    ])

    return {
        'X': feature_matrix,
        'daily_trips': daily_trips,
        'net_am': net_am,
        'net_pm': net_pm,
        'day_share': day_share,
        'self_loop': self_loop,
        'we_wd_ratio': we_wd_ratio,
        'stockout_rate': stockout_rate,
        'lat': lat_sel,
        'lon': lon_sel
    }


def kmeans_fit(X, k, n_init=25, max_iter=80, seed=42):
    """K-Means clustering with K-Means++ initialization and multiple restarts."""
    rng = np.random.default_rng(seed)
    N, D = X.shape

    # Standardize features
    mu = X.mean(0)
    sigma = np.maximum(X.std(0), 1e-4)
    Z = (X - mu) / sigma

    best_inertia = float('inf')
    best_labels = None
    best_centers = None

    for init in range(n_init):
        # K-Means++ init
        centers = np.empty((k, D), dtype=np.float64)
        centers[0] = Z[rng.integers(N)]
        dist_sq = ((Z - centers[0]) ** 2).sum(1)

        for j in range(1, k):
            probs = dist_sq / np.maximum(dist_sq.sum(), 1e-12)
            centers[j] = Z[rng.choice(N, p=probs)]
            dist_sq = np.minimum(dist_sq, ((Z - centers[j]) ** 2).sum(1))

        # Iterative updates
        for _ in range(max_iter):
            # Compute distance matrix [N, k]
            diff = Z[:, None, :] - centers[None, :, :]
            dists = (diff ** 2).sum(2)
            labels = dists.argmin(1)

            # Update centroids
            new_centers = np.zeros_like(centers)
            for j in range(k):
                mask = (labels == j)
                if np.any(mask):
                    new_centers[j] = Z[mask].mean(0)
                else:
                    new_centers[j] = Z[rng.integers(N)]

            if np.allclose(centers, new_centers, atol=1e-4):
                break
            centers = new_centers

        # Inertia
        diff = Z - centers[labels]
        inertia = float((diff ** 2).sum())

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centers = centers

    return best_labels, best_centers, best_inertia, Z


def assign_cluster_identities(labels, stats, k):
    """Analyzes cluster centroids and assigns intuitive, distinctive functional identities."""
    identities = {}

    for c in range(k):
        idx = np.where(labels == c)[0]
        n = len(idx)
        avg_vol = float(stats['daily_trips'][idx].mean())
        avg_net_am = float(stats['net_am'][idx].mean())
        avg_net_pm = float(stats['net_pm'][idx].mean())
        avg_self = float(stats['self_loop'][idx].mean())
        avg_we_wd = float(stats['we_wd_ratio'][idx].mean())
        avg_stockout = float(stats['stockout_rate'][idx].mean())

        # Determine distinctive functional profile
        if avg_vol < 8.0:
            c_type = "외곽 저수요 롱테일 (Low-demand Peripheral)"
            code = "LOW_DEMAND_PERIPHERAL"
            desc = "일평균 8건 미만의 외곽/한적한 대여소. 결품 및 초과 적재 위험 낮음."
            priority = "낮음 (Low Priority)"
            action = "정기 트럭 순회 최소화, 시민 앱 신고 또는 정비 주기 시 수시 방문"
        elif avg_net_am <= -0.35:
            c_type = "주거단지 출근 유출 거점 (Residential Morning Outflow)"
            code = "RESIDENTIAL_MORNING_OUTFLOW"
            desc = "아파트/주거 밀집지로 아침(07~09시) 출근 대여 급증 및 심각한 순유출. 오전 품절 위험 최고."
            priority = "최우선 (High Priority)"
            action = "06:30~07:30 이전 자전거 집중 사전 공급(Morning Pre-fill) 필수. 09시 이후 추가 공급 불필요."
        elif avg_net_am >= +0.45:
            c_type = "정부청사·업무·연구 집중 유입 거점 (Workplace & Campus Morning Peak Inflow)"
            code = "WORKPLACE_PEAK_INFLOW"
            desc = "정부청사, 연구소, 핵심 강의동 등 아침 출근·등교 반납 폭증. 오후 퇴근 시 역방향 유출."
            priority = "최우선 (High Priority)"
            action = "08:30~10:00 거치대 만차 방지용 긴급 트럭 수거(Drain), 오후 17시 이전에는 공급 불필요."
        elif avg_vol >= 35.0 and abs(avg_net_am) < 0.15:
            c_type = "도심·역세권·여가 초고수요 거점 (Core Urban Hubs & Transit Centers)"
            code = "CORE_URBAN_HIGH_DEMAND"
            desc = "대전역, 서대전역, 카이스트 관문, 타임월드, 한밭수목원 등 종일 회전율이 높은 대형 거점."
            priority = "상시집중 (Continuous Monitoring)"
            action = "종일 상시 모니터링, 피크 시간대(점심/퇴근/주말) 거치대 부족 시 인접 분산 유도 및 수시 정돈."
        elif avg_net_am >= +0.10:
            c_type = "대학 캠퍼스 및 연구·행정 완만 유입형 (Campus Academic & Institutional Inflow)"
            code = "CAMPUS_ACADEMIC_INFLOW"
            desc = "충남대 단과대, 학생회관, 연구단지 등으로 오전부터 낮 시간대까지 완만한 유입 지속."
            priority = "중간 (Medium Priority)"
            action = "낮 시간대(11~14시) 교내 거치 현황 점검, 기숙사/식당과 단과대 간 내부 균형 재배치."
        elif avg_net_am <= -0.15:
            c_type = "생활권 보조 주거 거점 (Secondary Residential Commuter)"
            code = "SECONDARY_RESIDENTIAL"
            desc = "중소형 주거지역 및 환승 정류장 주변으로 완만한 출근 유출 발생."
            priority = "중간 (Medium Priority)"
            action = "출근 전 최소 기본 수량(3~5대) 유지 점검, 주거지 유출 거점 트럭 순회 경로에 연계."
        else:
            c_type = "상업·생활지구 균형 순환형 (Balanced Urban Commercial)"
            code = "BALANCED_COMMERCIAL"
            desc = "대여와 반납이 시간대별로 균형을 이루며 주야간 지속 순환."
            priority = "보통 (Standard Priority)"
            action = "자연 순환율이 높으므로 고장·방치 차량 회수 위주 점검."

        identities[c] = {
            'cluster_id': c,
            'name': c_type,
            'code': code,
            'description': desc,
            'rebalance_priority': priority,
            'recommended_action': action,
            'station_count': n,
            'avg_daily_trips': round(avg_vol, 1),
            'avg_net_flow_am': round(avg_net_am, 2),
            'avg_net_flow_pm': round(avg_net_pm, 2),
            'avg_self_loop_ratio': round(avg_self, 3),
            'avg_weekend_to_weekday': round(avg_we_wd, 2),
            'avg_stockout_rate': round(avg_stockout, 3)
        }

    return identities


def export_results(data, labels, identities, out_csv_path, out_json_path):
    """Exports CSV with station-level cluster tags and JSON with cluster summaries."""
    stations = data['stations']
    st_meta = data['st_meta']
    sel_idx = data['sel_idx']
    stats = data['stats']

    # 1. Station-level CSV
    rows = []
    for i, idx in enumerate(sel_idx):
        sid = str(stations[idx])
        c = int(labels[i])
        info = identities[c]
        meta = st_meta.get(sid, {'name': '미등록', 'gu': '미등록'})

        self_loop_val = float(stats['self_loop'][i])
        is_leisure = "Y" if (self_loop_val >= 0.15 or float(stats['we_wd_ratio'][i]) >= 1.3) else "N"

        rows.append({
            'station_id': sid,
            'name': meta['name'],
            'gu': meta['gu'],
            'lat': f"{data['lat'][idx]:.6f}",
            'lon': f"{data['lon'][idx]:.6f}",
            'cluster_id': c,
            'cluster_name': info['name'],
            'cluster_code': info['code'],
            'daily_trips': f"{stats['daily_trips'][i]:.1f}",
            'net_flow_am': f"{stats['net_am'][i]:+.2f}",
            'net_flow_pm': f"{stats['net_pm'][i]:+.2f}",
            'self_loop_ratio': f"{self_loop_val:.3f}",
            'is_leisure_high_circular': is_leisure,
            'weekend_ratio': f"{stats['we_wd_ratio'][i]:.2f}",
            'stockout_rate': f"{stats['stockout_rate'][i]:.1%}",
            'rebalance_priority': info['rebalance_priority'],
            'recommended_action': info['recommended_action']
        })

    # Sort by cluster then daily trips descending
    rows.sort(key=lambda r: (int(r['cluster_id']), -float(r['daily_trips'])))

    Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 2. Cluster summary JSON
    summary_report = {
        'total_stations_evaluated': len(sel_idx),
        'number_of_clusters': len(identities),
        'clusters': {}
    }

    for c, info in identities.items():
        c_rows = [r for r in rows if int(r['cluster_id']) == c]
        top_stations = [{
            'station_id': r['station_id'],
            'name': r['name'],
            'gu': r['gu'],
            'daily_trips': float(r['daily_trips']),
            'net_am': r['net_flow_am'],
            'self_loop': r['self_loop_ratio']
        } for r in c_rows[:5]]

        # District breakdown
        gus = {}
        for r in c_rows:
            gus[r['gu']] = gus.get(r['gu'], 0) + 1

        summary_report['clusters'][str(c)] = {
            **info,
            'district_breakdown': gus,
            'top_representative_stations': top_stations
        }

    Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    return rows, summary_report


def print_summary_table(identities, summary_report):
    """Prints a clean, formatted Markdown table of cluster characteristics to the console."""
    sys.stdout.reconfigure(encoding='utf-8')
    print("\n" + "=" * 90)
    print("                     [타슈 대여소 행동 패턴 클러스터링 분석 결과]")
    print("=" * 90)
    print(f"{'ID':<3} | {'군집 유형':<28} | {'대여소수':<6} | {'일평균이용':<8} | {'출근순흐름':<8} | {'퇴근순흐름':<8} | {'제자리반납':<8} | {'재배치우선순위'}")
    print("-" * 90)
    for c, info in identities.items():
        print(f"{c:<3} | {info['name']:<28} | {info['station_count']:<6} | {info['avg_daily_trips']:>6.1f}건 | {info['avg_net_flow_am']:>+6.2f} | {info['avg_net_flow_pm']:>+6.2f} | {info['avg_self_loop_ratio']:>6.1%} | {info['rebalance_priority']}")
    print("-" * 90)

    print("\n[군집별 대표 대여소 및 운영 재배치 전략]")
    for c, info in identities.items():
        c_data = summary_report['clusters'][str(c)]
        top_st_names = [f"{st['name']}({st['daily_trips']}건/일)" for st in c_data['top_representative_stations'][:3]]
        print(f"\n■ 클러스터 {c}: {info['name']} (총 {info['station_count']}개 대여소)")
        print(f"  - 대표 거점: {', '.join(top_st_names)}")
        print(f"  - 주요 특성: {info['description']}")
        print(f"  - 운영 지침: {info['recommended_action']}")
    print("\n" + "=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Tashu Station Clustering Pipeline")
    parser.add_argument('--data', default='data/processed/tashu_60min.npz', help='Path to tashu npz')
    parser.add_argument('--stations-csv', default='data/processed/stations.csv', help='Path to stations metadata csv')
    parser.add_argument('--train-start', default='2025-08-01', help='Start date for clustering profile')
    parser.add_argument('--cutoff', default='2026-02-01', help='Cutoff date for clustering profile')
    parser.add_argument('--k', type=int, default=6, help='Number of clusters (default: 6)')
    parser.add_argument('--out-csv', default='data/clustering_stations.csv', help='Output CSV for station clusters')
    parser.add_argument('--out-json', default='data/clustering_summary.json', help='Output JSON for cluster summary')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    print(f"1. Loading data from {args.data} (Training window: {args.train_start} ~ {args.cutoff})...")
    data = load_data(args.data, args.stations_csv, args.train_start, args.cutoff)
    print(f"   -> Evaluated {len(data['sel_idx'])} active customer stations.")

    print("2. Extracting multi-dimensional behavioral features...")
    stats = extract_features(data)
    data['stats'] = stats

    print(f"3. Running K-Means clustering (K={args.k}, Seed={args.seed})...")
    labels, centers, inertia, Z = kmeans_fit(stats['X'], args.k, n_init=30, seed=args.seed)
    print(f"   -> Clustering converged. Total Inertia: {inertia:.2f}")

    print("4. Profiling clusters and assigning operational identities...")
    identities = assign_cluster_identities(labels, stats, args.k)

    print("5. Exporting results...")
    rows, summary_report = export_results(data, labels, identities, args.out_csv, args.out_json)
    print(f"   -> Saved station assignments to: {args.out_csv}")
    print(f"   -> Saved cluster summary to: {args.out_json}")

    print_summary_table(identities, summary_report)


if __name__ == '__main__':
    main()
