"""대여소 특성 분석과 군집화 (과거 대여이력만 사용).

A. 수요 규모      대여소별 크기 분포, 집중도(지니), 상하위 비교
B. 시간 구조      시각·요일·월 패턴, 기온/강수와의 관계, 평일 대비 휴일
C. 이동 구조      이용시간·거리 분포, 자기순환, 상위 OD, 출근시간 순유출입
D. 유형 군집화    시간 프로필로 k-평균 (충분한 기록이 있는 대여소만, 실루엣으로 k 선택)
E. OD 커뮤니티    Louvain 분할, 내부 통행 비율, 커뮤니티가 불균형을 가두는 정도
F. 재고 관점      복원 재고로 유형별 빈 시간, 결품 지속시간 분포
그림: outputs/eda2/*.png

실행:  python src/analyze_stations.py [--k 6] [--min-rent 300]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
V2, V3 = PROC / "v2", PROC / "v3"
OUT = ROOT / "outputs" / "eda2"
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main(args):
    OUT.mkdir(parents=True, exist_ok=True)
    rep: dict = {}
    z = np.load(V2 / "panel.npz")
    ts = pd.to_datetime(z["ts"].astype("datetime64[h]"))
    rent, ret = z["rent"], z["ret"]
    obs = z["node_active"] & z["hour_ok"][:, None]
    nodes = pd.read_parquet(V2 / "nodes.parquet")
    sp = pd.read_parquet(V2 / "node_spatial.parquet")
    T, N = rent.shape
    hour, dow, month = ts.hour.to_numpy(), ts.dayofweek.to_numpy(), ts.month.to_numpy()
    weekend = dow >= 5
    obs_h = obs.sum(0)
    tot_rent = (rent * obs).sum(0)
    tot_ret = (ret * obs).sum(0)
    per_h = tot_rent / np.maximum(obs_h, 1)

    # ---------------------------------------------------------- A. 수요 규모
    order = np.sort(tot_rent)[::-1]
    rep["A_규모"] = {
        "대여소": int(N), "총 대여": int(tot_rent.sum()),
        "대여소당 19개월 대여 중앙값": int(np.median(tot_rent)),
        "시간당 대여 중앙값": round(float(np.median(per_h)), 3),
        "지니계수": round(gini(tot_rent), 3),
        "상위 10% 비중": round(float(order[: N // 10].sum() / tot_rent.sum()), 3),
        "상위 25% 비중": round(float(order[: N // 4].sum() / tot_rent.sum()), 3),
        "하위 50% 비중": round(float(order[N // 2 :].sum() / tot_rent.sum()), 3),
        "대여 절반을 만드는 대여소 수": int(np.searchsorted(np.cumsum(order) / tot_rent.sum(), 0.5) + 1),
    }
    bins = [0, 0.1, 0.3, 0.5, 1, 2, 1e9]
    labs = ["<0.1", "0.1~0.3", "0.3~0.5", "0.5~1", "1~2", "2+"]
    grp = pd.cut(per_h, bins, labels=labs, right=False)
    rep["A_규모구간"] = {str(k): {"대여소": int(v), "대여비중": round(float(tot_rent[grp == k].sum() / tot_rent.sum()), 3)}
                    for k, v in pd.Series(grp).value_counts().sort_index().items()}

    # ---------------------------------------------------------- B. 시간 구조
    city_h = (rent * obs).sum(1) / np.maximum(obs.sum(1), 1)
    rep["B_시각"] = {int(h): round(float(city_h[hour == h].mean()), 3) for h in range(24)}
    rep["B_요일"] = {int(d): round(float(city_h[dow == d].mean()), 3) for d in range(7)}
    rep["B_월"] = {int(m): round(float(city_h[month == m].mean()), 3) for m in range(1, 13)}
    rep["B_평일대비휴일"] = round(float(city_h[weekend].mean() / city_h[~weekend].mean()), 3)
    wp = ROOT / "data/raw/weather_openmeteo_daejeon.parquet"
    if wp.exists():
        w = pd.read_parquet(wp)
        d = pd.DataFrame({"y": city_h, "rain": w.precipitation.to_numpy()[:T], "temp": w.temperature_2m.to_numpy()[:T],
                          "ok": obs.any(1)})
        d = d[d.ok]
        rep["B_날씨"] = {
            "비 안 올 때 시간당": round(float(d.y[d.rain == 0].mean()), 3),
            "0~1mm": round(float(d.y[(d.rain > 0) & (d.rain <= 1)].mean()), 3),
            "1mm 이상": round(float(d.y[d.rain > 1].mean()), 3),
            "기온 상관": round(float(np.corrcoef(d.y, d.temp)[0, 1]), 3),
            "기온 10도 미만": round(float(d.y[d.temp < 10].mean()), 3),
            "기온 20~28도": round(float(d.y[(d.temp >= 20) & (d.temp < 28)].mean()), 3),
            "기온 30도 이상": round(float(d.y[d.temp >= 30].mean()), 3),
        }

    # ---------------------------------------------------------- C. 이동 구조
    tr = pd.read_parquet(PROC / "trips.parquet", columns=["rent_ts", "rent_st", "ret_st", "dur_min", "dist_km"])
    tr["rent_st"] = tr.rent_st.astype(str)
    tr["ret_st"] = tr.ret_st.astype(str)
    tr = tr[(tr.dur_min <= 1440) & ~((tr.rent_st == tr.ret_st) & (tr.dur_min <= 1))]
    rep["C_이용"] = {
        "이용시간 분위(분)": {str(q): float(tr.dur_min.quantile(q)) for q in [.25, .5, .75, .9, .99]},
        "같은 대여소 반납 비율": round(float((tr.rent_st == tr.ret_st).mean()), 3),
        "관제센터 관련 비율": round(float(((tr.rent_st == "ST1220") | (tr.ret_st == "ST1220")).mean()), 3),
    }
    od = pd.read_parquet(V2 / "od_month.parquet").groupby(["src", "dst"], as_index=False).n.sum()
    od_ns = od[od.src != od.dst].sort_values("n", ascending=False)
    top = od_ns.head(12).copy()
    top["from"] = nodes["name"].to_numpy()[top.src]
    top["to"] = nodes["name"].to_numpy()[top.dst]
    rep["C_상위OD"] = top[["from", "to", "n"]].to_dict("records")
    peak = (hour >= 7) & (hour <= 9) & ~weekend
    net_peak = (ret[peak] * obs[peak]).sum(0) - (rent[peak] * obs[peak]).sum(0)
    ev_peak = (ret[peak] * obs[peak]).sum(0) + (rent[peak] * obs[peak]).sum(0)
    rep["C_출근불균형"] = {
        "|순유입| 합": int(np.abs(net_peak).sum()),
        "가장 많이 들어오는 곳": [{"name": nodes['name'][i], "net": int(net_peak[i])} for i in np.argsort(-net_peak)[:5]],
        "가장 많이 빠지는 곳": [{"name": nodes['name'][i], "net": int(net_peak[i])} for i in np.argsort(net_peak)[:5]],
        "불균형이 큰 대여소 비율(|순유입|>총이동 30%)": round(float((np.abs(net_peak) > 0.3 * np.maximum(ev_peak, 1)).mean()), 3),
    }

    # ---------------------------------------------------------- D. 유형 군집화
    def prof(mat, mask):
        p = np.zeros((24, N), np.float32)
        for h in range(24):
            m = mask & (hour == h)
            cnt = obs[m].sum(0)
            p[h] = np.where(cnt > 0, (mat[m] * obs[m]).sum(0) / np.maximum(cnt, 1), 0)
        return p

    blocks = {"대여평일": prof(rent, ~weekend), "대여휴일": prof(rent, weekend),
              "반납평일": prof(ret, ~weekend), "반납휴일": prof(ret, weekend)}
    keep = tot_rent >= args.min_rent
    cols = [f"{k}_{h}" for k in blocks for h in range(24)]
    P = np.concatenate([blocks[k].T for k in blocks], 1)  # (N,96) 시간당 평균 건수
    Pk = P[keep]
    Pk = (Pk + 0.01) / (Pk.sum(1, keepdims=True) + 0.96)  # 모양만 남기고 크기 제거, 작은 값 보정
    X = np.sqrt(Pk)  # 큰 봉우리에 덜 끌리게
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    rep["D_군집대상"] = {"쓴 대여소": int(keep.sum()), "제외(기록 적음)": int((~keep).sum()),
                    "제외분이 차지하는 대여 비중": round(float(tot_rent[~keep].sum() / tot_rent.sum()), 3)}
    sil = {}
    for k in range(3, 11):
        km0 = KMeans(k, n_init=10, random_state=0).fit(X)
        sil[k] = round(float(silhouette_score(X, km0.labels_, sample_size=min(3000, len(X)), random_state=0)), 4)
    rep["D_실루엣"] = sil
    km = KMeans(args.k, n_init=30, random_state=0).fit(X)
    lab = np.full(N, -1)
    lab[keep] = km.labels_
    prof_df = pd.DataFrame(np.vstack([Pk[km.labels_ == c].mean(0) for c in range(args.k)]), columns=cols)

    def share(c, block, hrs):
        v = prof_df.loc[c, [f"{block}_{h}" for h in hrs]].sum()
        return float(v / prof_df.loc[c, [f"{block}_{h}" for h in range(24)]].sum())

    rows = []
    for c in range(args.k):
        m = lab == c
        rows.append({
            "유형": c, "대여소수": int(m.sum()),
            "대여비중": round(float(tot_rent[m].sum() / tot_rent.sum()), 3),
            "시간당대여": round(float(per_h[m].mean()), 2),
            "출근대여(7-9)": round(share(c, "대여평일", [7, 8, 9]), 3),
            "퇴근대여(17-19)": round(share(c, "대여평일", [17, 18, 19]), 3),
            "심야(0-5)": round(share(c, "대여평일", range(5)), 3),
            "휴일낮(11-17)": round(share(c, "대여휴일", range(11, 18)), 3),
            "출근_반납−대여": round(share(c, "반납평일", [7, 8, 9]) - share(c, "대여평일", [7, 8, 9]), 3),
            "휴일/평일": round(float(prof_df.loc[c, [f"대여휴일_{h}" for h in range(24)]].sum()
                               / max(prof_df.loc[c, [f"대여평일_{h}" for h in range(24)]].sum(), 1e-9)), 2),
            "역거리m": int(sp.loc[m, "dist_station_m"].median()),
            "대학거리m": int(sp.loc[m, "dist_univ_m"].median()),
            "상가300m": int(sp.loc[m, "n_commerce_300m"].median()),
            "아파트500m": int(sp.loc[m, "n_apt_bldg_500m"].median()),
            "고도m": int(sp.loc[m, "elevation_m"].median()),
            "예시": ", ".join(nodes.loc[m].assign(r=tot_rent[m]).nlargest(3, "r")["name"].tolist()),
        })
    sm = pd.DataFrame(rows)
    sm.to_csv(OUT / "cluster_summary.csv", index=False, encoding="utf-8-sig")
    prof_df.insert(0, "n", [int((lab == c).sum()) for c in range(args.k)])
    prof_df.to_csv(OUT / "cluster_profiles.csv", index_label="cluster", encoding="utf-8-sig")

    # ---------------------------------------------------------- E. OD 커뮤니티
    G = nx.Graph()
    agg = od[od.src != od.dst].groupby([od[["src", "dst"]].min(1), od[["src", "dst"]].max(1)]).n.sum()
    for (a, b), n in agg.items():
        G.add_edge(int(a), int(b), weight=float(n))
    best = None
    for res in [0.6, 1.0, 1.5, 2.0, 3.0]:
        cs = nx.community.louvain_communities(G, weight="weight", seed=0, resolution=res)
        q = nx.community.modularity(G, cs, weight="weight")
        if best is None or res == args.resolution:
            pass
        rep.setdefault("E_해상도별", {})[res] = {"커뮤니티": len(cs), "모듈러리티": round(float(q), 3)}
        if res == args.resolution:
            best = cs
    comms = best
    comm = np.full(N, -1)
    for i, cset in enumerate(comms):
        for v in cset:
            comm[v] = i
    ee = od[od.src != od.dst]
    inner = ee[(comm[ee.src.values] == comm[ee.dst.values]) & (comm[ee.src.values] >= 0)].n.sum()
    cdf = pd.DataFrame({"comm": comm, "net_peak": net_peak, "rent": tot_rent})
    cg = cdf[cdf.comm >= 0].groupby("comm").agg(대여소수=("rent", "size"), 총대여=("rent", "sum"), 출근순유입=("net_peak", "sum"))
    rep["E_커뮤니티"] = {
        "해상도": args.resolution, "커뮤니티 수": len(comms),
        "모듈러리티": round(float(nx.community.modularity(G, comms, weight="weight")), 3),
        "커뮤니티 안에서 끝나는 이동": round(float(inner / ee.n.sum()), 3),
        "크기 중앙값": int(cg["대여소수"].median()), "최대": int(cg["대여소수"].max()),
        "출근 불균형이 커뮤니티 안에서 상쇄되는 비율": round(float(1 - cg["출근순유입"].abs().sum() / np.abs(net_peak).sum()), 3),
    }
    cg.sort_values("총대여", ascending=False).to_csv(OUT / "od_communities.csv", encoding="utf-8-sig")

    # ---------------------------------------------------------- F. 재고 관점
    piv = None
    inv_p = V3 / "inventory_hourly.parquet"
    if inv_p.exists():
        inv = pd.read_parquet(inv_p, columns=["ts", "node", "stock_mean", "empty_frac", "rent"])
        inv["cluster"] = lab[inv.node.to_numpy()]
        inv["hour"] = inv.ts.dt.hour
        d = inv[inv.cluster >= 0].groupby(["cluster", "hour"]).agg(빈비율=("empty_frac", "mean"), 재고=("stock_mean", "mean")).reset_index()
        d.to_csv(OUT / "cluster_stock_hour.csv", index=False, encoding="utf-8-sig")
        piv = d.pivot(index="hour", columns="cluster", values="빈비율").round(3)
        rep["F_유형별_빈비율"] = {"전체평균": piv.mean().round(3).to_dict(), "07시": piv.loc[7].to_dict(), "18시": piv.loc[18].to_dict()}
        st = inv.groupby("node").empty_frac.mean()
        rep["F_빈시간분포"] = {str(q): round(float(st.quantile(q)), 3) for q in [.1, .25, .5, .75, .9]}
        rep["F_대여가_빈시간에_일어난_비중"] = round(float(inv.loc[inv.empty_frac > 0.9, "rent"].sum() / inv.rent.sum()), 3)

    # ---------------------------------------------------------- 그림
    k = args.k
    fig, ax = plt.subplots(2, (k + 1) // 2, figsize=(4 * ((k + 1) // 2), 7), sharex=True)
    for c in range(k):
        a = ax.flat[c]
        a.plot(range(24), prof_df.loc[c, [f"대여평일_{h}" for h in range(24)]].to_numpy(), lw=2, label="대여 평일")
        a.plot(range(24), prof_df.loc[c, [f"반납평일_{h}" for h in range(24)]].to_numpy(), lw=2, ls="--", label="반납 평일")
        a.plot(range(24), prof_df.loc[c, [f"대여휴일_{h}" for h in range(24)]].to_numpy(), lw=1.3, alpha=.75, label="대여 휴일")
        a.set_title(f"유형 {c} · {int((lab == c).sum())}곳 · 대여 {sm.loc[c, '대여비중']:.0%}", fontsize=10)
        a.set_xticks([0, 6, 9, 12, 15, 18, 21])
        if c == 0:
            a.legend(fontsize=8)
    fig.suptitle("대여소 유형별 하루 모양")
    fig.tight_layout()
    fig.savefig(OUT / "cluster_profiles.png", dpi=130)

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    m = lab >= 0
    ax[0].scatter(nodes.lon[m], nodes.lat[m], c=lab[m], cmap="tab10", s=np.sqrt(tot_rent[m]) / 5 + 3, alpha=.85)
    ax[0].set_title(f"유형 지도 ({int(m.sum())}곳)")
    big = pd.Series(comm[comm >= 0]).value_counts().head(10).index
    mm = np.isin(comm, big)
    ax[1].scatter(nodes.lon[mm], nodes.lat[mm], c=comm[mm], cmap="tab20", s=np.sqrt(tot_rent[mm]) / 5 + 3, alpha=.85)
    ax[1].set_title(f"OD 커뮤니티 지도 (큰 10개 / 총 {len(comms)}개)")
    for a in ax:
        a.set_aspect(1 / np.cos(np.radians(36.35)))
    fig.tight_layout()
    fig.savefig(OUT / "cluster_map.png", dpi=130)

    if piv is not None:
        fig, a = plt.subplots(figsize=(8, 4.5))
        for c in piv.columns:
            a.plot(piv.index, piv[c], label=f"유형 {c}")
        a.set_xlabel("시각"); a.set_ylabel("재고 0 인 시간 비율"); a.legend(fontsize=8)
        a.set_title("유형별 빈 대여소 시간대 (복원 재고)")
        fig.tight_layout(); fig.savefig(OUT / "cluster_empty_hour.png", dpi=130)

    p2 = PCA(2, random_state=0).fit_transform(X)
    fig, a = plt.subplots(figsize=(6.5, 5.5))
    a.scatter(p2[:, 0], p2[:, 1], c=km.labels_, cmap="tab10", s=7, alpha=.65)
    a.set_title("프로필 PCA")
    fig.tight_layout(); fig.savefig(OUT / "cluster_pca.png", dpi=130)

    pd.DataFrame({"node": np.arange(N), "station_id": nodes.station_id, "name": nodes["name"], "lat": nodes.lat,
                  "lon": nodes.lon, "cluster": lab, "od_community": comm, "total_rent": tot_rent,
                  "rent_per_hour": per_h.round(3), "net_peak_in": net_peak}).to_csv(
        OUT / "station_clusters.csv", index=False, encoding="utf-8-sig")
    (OUT / "analysis_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")

    print(json.dumps({k2: rep[k2] for k2 in rep if not k2.startswith("C_상위")}, ensure_ascii=False, indent=1))
    print("\n=== 유형 요약")
    print(sm.drop(columns=["예시"]).to_string(index=False))
    for r in rows:
        print(f"유형 {r['유형']}: {r['예시']}")
    print("\n=== 상위 이동 (대여소 쌍)")
    print(pd.DataFrame(rep["C_상위OD"]).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--min-rent", type=int, default=300, help="19개월 총 대여가 이 값 이상인 대여소만 군집화")
    p.add_argument("--resolution", type=float, default=2.0, help="Louvain 해상도 (클수록 잘게 나뉨)")
    main(p.parse_args())
