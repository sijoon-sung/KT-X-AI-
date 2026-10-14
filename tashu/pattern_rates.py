"""계절·시간·장소 패턴 분석 → 포아송 요율 구조를 단계적으로 조립하고 2026-02/03 에서 검증.

요율 분해:  λ(s, h, d, m) = scale[s] × shape[g(s), h, d] × season[m] × level
  scale  대여소 규모        (장소)
  shape  시각×평일/휴일 모양 (시간; g(s) 는 대여소 유형)
  season 월 효과            (계절)
  level  운영 규모          (증차)

평가 셀: 재고가 한 시간 내내 있었던(exposure==1) 고객 대여소×시간, 06~22시.
지표: 셀당 포아송 로그우도(높을수록 좋음), 예측합/실제합, 시각별 WAPE.
학습: 2024-08 ~ 2026-01 (계절 추정에 2025-02/03 필요).  시험: 2026-02, 2026-03.
"""
import json
import numpy as np

NON_CUSTOMER = ('ST0001', 'ST1220', 'ST1198')


def load(path):
    with np.load(path) as f:
        z = {k: f[k] for k in ('rentals', 'returns', 'exposure', 'active', 'valid', 'closed', 'bin_start', 'stations', 'lat', 'lon')}
    bs = z['bin_start']; tb = bs.astype('datetime64[h]').astype(np.int64)
    z['hour'] = tb % 24; z['dow'] = ((tb // 24) + 3) % 7; z['day'] = tb // 24
    z['moy'] = (bs.astype('datetime64[M]').astype(np.int64) % 12)          # month of year 0..11
    z['ym'] = bs.astype('datetime64[M]')
    z['daytype'] = (z['dow'] >= 5).astype(int)
    z['ok'] = z['valid'] & ~z['closed'] & (z['hour'] >= 6) & (z['hour'] < 22)
    z['cust'] = ~np.isin(z['stations'].astype(str), NON_CUSTOMER)
    return z


def poisson_ll(y, mu):
    from math import lgamma
    gammaln = np.vectorize(lgamma)
    mu = np.maximum(mu, 1e-6)
    return y * np.log(mu) - mu - gammaln(y + 1)


def main():
    z = load('data/processed/tashu_60min.npz')
    T, N = z['rentals'].shape
    full = (z['exposure'] >= 0.999) & z['active'] & z['ok'][:, None] & z['cust'][None]
    train_t = (z['bin_start'] >= np.datetime64('2024-08-01')) & (z['bin_start'] < np.datetime64('2026-02-01'))
    tests = {'feb_2026': (np.datetime64('2026-02-01'), np.datetime64('2026-03-01')),
             'mar_2026': (np.datetime64('2026-03-01'), np.datetime64('2026-04-01'))}
    Y = z['rentals'].astype(float)
    R = {}

    # ------------------------------------------------ 패턴 1: 계절 (도시 전체, 활동 자전거당)
    # 월별 도시 시간당 대여(전체 대여소 합, ok 시간 평균) 과 그 달 활동 자전거 수(trips.npz 에서 미리 계산한 값)
    bikes_by_month = {'2024-08': 2810, '2024-09': 2635, '2024-10': 2710, '2024-11': 2747, '2024-12': 2350, '2025-01': 2127,
                      '2025-02': 2339, '2025-03': 2773, '2025-04': 3356, '2025-05': 3535, '2025-06': 3860, '2025-07': 4001,
                      '2025-08': 3868, '2025-09': 4047, '2025-10': 4058, '2025-11': 3845, '2026-01': 3498, '2026-02': 4057, '2026-03': 4944}
    season = {}
    for ym in np.unique(z['ym']):
        sel = (z['ym'] == ym) & z['ok']
        if sel.sum() < 100:
            continue
        per_h = (Y[sel][:, z['cust']]).sum(1).mean()
        b = bikes_by_month.get(str(ym))
        season[str(ym)] = {'city_rentals_per_hour': round(float(per_h), 1), 'active_bikes': b,
                           'per_bike_per_hour': round(float(per_h / b), 4) if b else None}
    R['season_city'] = season

    # ------------------------------------------------ 패턴 2: 시간 모양은 대여소마다 얼마나 다른가
    tr = train_t & z['ok']
    key48 = z['hour'] * 2 + z['daytype']
    cnt48 = np.zeros((48, N)); hrs48 = np.zeros((48, N))
    for k in range(48):
        sel = tr & (key48 == k)
        m = full[sel]
        cnt48[k] = (Y[sel] * m).sum(0); hrs48[k] = m.sum(0)
    rate48 = cnt48 / np.maximum(hrs48, 1)
    tot = cnt48.sum(0)
    slot_ok = hrs48.sum(1) > 0                                             # 06~22시 슬롯만 존재
    big = (tot >= 500) & (hrs48[slot_ok].min(0) >= 10) & z['cust']
    shape_s = rate48[:, big] / rate48[:, big].sum(0, keepdims=True)           # 대여소별 정규화 모양 [48, n]
    city_shape = shape_s.mean(1)
    # 모양의 이질성: 대여소 모양과 도시 모양의 L1 거리, 그리고 k-means 유형
    R['shape_l1_to_city_quantiles'] = {q: float(np.quantile(np.abs(shape_s - city_shape[:, None]).sum(0), q)) for q in (0.25, 0.5, 0.75, 0.9)}
    rng = np.random.default_rng(0)
    X = shape_s.T
    best = None
    for k in (2, 3, 4, 6):
        cen = X[rng.choice(len(X), k, replace=False)]
        for _ in range(30):
            lab = np.argmin(((X[:, None] - cen[None]) ** 2).sum(2), 1)
            cen = np.array([X[lab == j].mean(0) if (lab == j).any() else cen[j] for j in range(k)])
        within = float(((X - cen[lab]) ** 2).sum())
        R.setdefault('shape_kmeans_within_ss', {})[k] = round(within, 3)
        if k == 4:
            best = (lab, cen)
    lab4, cen4 = best
    types = np.full(N, -1); types[np.flatnonzero(big)] = lab4
    R['shape_types_k4'] = {int(j): {'stations': int((lab4 == j).sum()),
                                    'weekday_peak_hour': int(np.argmax(cen4[j][0::2])), 'weekend_peak_hour': int(np.argmax(cen4[j][1::2])),
                                    'am_share_wd(7-9)': round(float(cen4[j][14:20:2].sum() / cen4[j][0::2].sum()), 3),
                                    'pm_share_wd(17-19)': round(float(cen4[j][34:40:2].sum() / cen4[j][0::2].sum()), 3),
                                    'weekend_over_weekday': round(float(cen4[j][1::2].sum() / cen4[j][0::2].sum()), 3)}
                           for j in range(4)}

    # ------------------------------------------------ 패턴 3: 계절 × 장소 (월별 대여소 비중이 움직이는가: 캠퍼스 vs 나머지)
    # 이름으로 캠퍼스 대여소 표시 (충남대, 카이스트, 한밭대, 대전대, 목원대, 배재대, 우송대, 한남대)
    names = np.array([str(s) for s in np.load('data/processed/tashu_60min.npz')['stations']])
    import csv
    nm = {}
    with open('data/processed/stations.csv', encoding='utf-8-sig') as f:
        for r_ in csv.DictReader(f):
            nm[r_['station_id']] = r_['name']
    campus_kw = ('충남대', '카이스트', 'KAIST', '한밭대', '대전대', '목원대', '배재대', '우송대', '한남대', '대덕대', '기숙사')
    campus = np.array([any(k in nm.get(s, '') for k in campus_kw) for s in names]) & z['cust']
    R['campus_stations'] = int(campus.sum())
    share = {}
    for ym in np.unique(z['ym']):
        sel = (z['ym'] == ym) & z['ok']
        if sel.sum() < 100:
            continue
        tot_m = Y[sel][:, z['cust']].sum(); cam = Y[sel][:, campus].sum()
        share[str(ym)] = round(float(cam / max(tot_m, 1)), 4)
    R['campus_share_by_month'] = share

    # ------------------------------------------------ 요율 구조를 단계적으로 조립하고 검증
    def eval_model(name, mu_fn):
        out = {}
        for lab, (a, b) in tests.items():
            sel_t = (z['bin_start'] >= a) & (z['bin_start'] < b) & z['ok']
            ts = np.flatnonzero(sel_t)
            ll = 0; n = 0; pred = 0; act = 0
            wape_num = np.zeros(24); wape_den = np.zeros(24)
            for t in ts:
                m = full[t]
                if not m.any():
                    continue
                mu = mu_fn(t)[m]; y = Y[t][m]
                ll += poisson_ll(y, mu).sum(); n += m.sum(); pred += mu.sum(); act += y.sum()
                wape_num[z['hour'][t]] += np.abs(mu - y).sum(); wape_den[z['hour'][t]] += y.sum()
            out[lab] = {'ll_per_cell': round(float(ll / n), 4), 'pred_over_actual': round(float(pred / act), 3),
                        'wape_hourly_cells': round(float(wape_num.sum() / wape_den.sum()), 3)}
        R.setdefault('models', {})[name] = out
        print(name, json.dumps(out), flush=True)

    # 학습 통계 (2025-08~2026-01: 최근 6개월 규모용; 전체 18개월: 계절용)
    recent = (z['bin_start'] >= np.datetime64('2025-08-01')) & (z['bin_start'] < np.datetime64('2026-02-01')) & z['ok']
    c48 = np.zeros((48, N)); h48 = np.zeros((48, N))
    for k in range(48):
        sel = recent & (key48 == k); m = full[sel]
        c48[k] = (Y[sel] * m).sum(0); h48[k] = m.sum(0)
    flat = c48 / np.maximum(h48, 1)                                     # M0: 표 그대로
    eval_model('M0_flat_table', lambda t: flat[key48[t]])

    scale = c48.sum(0) / np.maximum(h48.sum(0), 1)                       # 시간당 평균 규모
    cshape = city_shape * 48 / city_shape.sum()                          # 평균 1 로 정규화 (요일유형 가중 무시)
    # 요일유형별로 따로 정규화: shape[k] = city rate at k / city mean rate
    city_rate48 = c48.sum(1) / np.maximum(h48.sum(1), 1)
    cshape = city_rate48 / city_rate48[slot_ok].mean()
    eval_model('M1_scale_x_cityshape', lambda t: scale * cshape[key48[t]])

    # 유형별 모양 (k=4, 소규모 대여소는 도시 모양)
    tshape = np.tile(cshape[:, None], (1, N))
    for j in range(4):
        idx = np.flatnonzero(types == j)
        rj = c48[:, idx].sum(1) / np.maximum(h48[:, idx].sum(1), 1)
        tshape[:, idx] = (rj / rj[slot_ok].mean())[:, None]
    eval_model('M2_scale_x_typeshape', lambda t: scale * tshape[key48[t], :])

    # 계절: 월(연중) 효과 = 그 달 도시 시간당 대여 / 학습 기간 평균, 2025년 값 사용 (2026-02/03 은 2025-02/03 의 계절)
    city_m = {}
    for ym in np.unique(z['ym'][train_t]):
        sel = (z['ym'] == ym) & z['ok']
        if sel.sum() >= 100:
            city_m[str(ym)] = Y[sel][:, z['cust']].sum(1).mean()
    base_level = np.mean([city_m[k] for k in ('2025-08', '2025-09', '2025-10', '2025-11', '2026-01')])
    seas = {'feb_2026': city_m['2025-02'] / base_level, 'mar_2026': city_m['2025-03'] / base_level}
    R['season_factor_from_2025'] = {k: round(float(v), 3) for k, v in seas.items()}

    def with_season(t):
        s = seas['feb_2026'] if z['ym'][t] == np.datetime64('2026-02') else seas['mar_2026']
        return scale * tshape[key48[t], :] * s
    eval_model('M3_+season(2025 same month)', with_season)

    # 수준(증차): 2026-02/03 활동 자전거 / 2025-02/03 활동 자전거  (오라클: 그 달 값을 안다고 가정)
    lvl = {'feb_2026': bikes_by_month['2026-02'] / bikes_by_month['2025-02'], 'mar_2026': bikes_by_month['2026-03'] / bikes_by_month['2025-03']}
    # 기준 수준도 맞춰야 함: base_level 은 2025-08~2026-01 의 자전거 수 기준 → 상대비 = (2026-mm 자전거 / base 기간 평균 자전거)
    base_bikes = np.mean([bikes_by_month[k] for k in ('2025-08', '2025-09', '2025-10', '2025-11', '2026-01')])
    lvl = {'feb_2026': bikes_by_month['2026-02'] / bikes_by_month['2025-02'] * 0 + (bikes_by_month['2026-02'] / base_bikes) / (bikes_by_month['2025-02'] / base_bikes),
           'mar_2026': (bikes_by_month['2026-03'] / base_bikes) / (bikes_by_month['2025-03'] / base_bikes)}
    R['level_factor_active_bikes_oracle'] = {k: round(float(v), 3) for k, v in lvl.items()}

    def with_level(t):
        k = 'feb_2026' if z['ym'][t] == np.datetime64('2026-02') else 'mar_2026'
        return scale * tshape[key48[t], :] * seas[k] * lvl[k]
    eval_model('M4_+level(active bikes, oracle)', with_level)

    # 계절×장소: 캠퍼스 대여소는 2025-02/03 의 캠퍼스 비중 변화를 별도 적용
    cam_adj = {}
    for k, ym25, ymb in (('feb_2026', '2025-02', None), ('mar_2026', '2025-03', None)):
        base_share = np.mean([share[m] for m in ('2025-08', '2025-09', '2025-10', '2025-11', '2026-01')])
        cam_adj[k] = share[ym25] / base_share
    R['campus_share_adjust'] = {k: round(float(v), 3) for k, v in cam_adj.items()}

    def with_campus(t):
        k = 'feb_2026' if z['ym'][t] == np.datetime64('2026-02') else 'mar_2026'
        mu = scale * tshape[key48[t], :] * seas[k] * lvl[k]
        a = cam_adj[k]; f = (1 - share_base_w * a) / (1 - share_base_w)   # 나머지 대여소로 재분배 (총량 보존)
        return np.where(campus, mu * a, mu * f)
    share_base_w = np.mean([share[m] for m in ('2025-08', '2025-09', '2025-10', '2025-11', '2026-01')])
    eval_model('M5_+campus season', with_campus)

    json.dump(R, open('data/pattern_rates.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in R.items() if k != 'models'}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
