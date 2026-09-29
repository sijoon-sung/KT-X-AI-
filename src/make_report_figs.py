"""보고서용 그림 생성 (과거 대여이력 + 복원 재고만 사용). outputs/report_figs/*.png"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
V2, V3 = PROC / "v2", PROC / "v3"
FIG = ROOT / "outputs" / "report_figs"
plt.rcParams.update({"font.family": ["Malgun Gothic", "DejaVu Sans"], "axes.unicode_minus": False,
                     "axes.grid": True, "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10})
ACC, ACC2, WARN, GREY = "#1C7A58", "#3B82C4", "#B4741A", "#8A9A94"


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("saved", name, round((FIG / f"{name}.png").stat().st_size / 1e3), "KB", flush=True)


def main():
    z = np.load(V2 / "panel.npz")
    ts = pd.to_datetime(z["ts"].astype("datetime64[h]"))
    rent, ret = z["rent"], z["ret"]
    obs = z["node_active"] & z["hour_ok"][:, None]
    nodes = pd.read_parquet(V2 / "nodes.parquet")
    N = rent.shape[1]
    hour, dow, month = ts.hour.to_numpy(), ts.dayofweek.to_numpy(), ts.month.to_numpy()
    weekend = dow >= 5
    city = (rent * obs).sum(1)
    n_obs = obs.sum(1)
    per_st = np.where(n_obs > 0, city / np.maximum(n_obs, 1), np.nan)

    # F1 시각별 (평일/휴일) + 반납
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for m, lbl, c in [(~weekend, "평일", ACC), (weekend, "휴일", ACC2)]:
        y = [np.nanmean(per_st[m & (hour == h)]) for h in range(24)]
        ax.plot(range(24), y, lw=2.2, color=c, label=f"{lbl} 대여")
    yr = [np.nanmean(np.where(n_obs > 0, (ret * obs).sum(1) / np.maximum(n_obs, 1), np.nan)[~weekend & (hour == h)]) for h in range(24)]
    ax.plot(range(24), yr, lw=1.6, ls="--", color=GREY, label="평일 반납")
    ax.set_xticks(range(0, 24, 3)); ax.set_xlabel("시각"); ax.set_ylabel("대여소 한 곳당 시간당 건수")
    ax.set_title("시각별 이용: 퇴근이 출근보다 크다")
    ax.legend(frameon=False)
    save(fig, "f1_hour")

    # F2 월별 추이 + 결측
    d = pd.Series(city, index=ts).resample("D").sum()
    ok = pd.Series(obs.any(1), index=ts).resample("D").mean()
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.fill_between(d.index, 0, d.values, color=ACC, alpha=.25)
    ax.plot(d.index, d.values, color=ACC, lw=.9)
    for a, b, t in [("2025-02-26", "2025-03-04", "시스템 장애"), ("2025-12-01", "2025-12-31", "원본 결측")]:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=WARN, alpha=.18)
        ax.text(pd.Timestamp(a), d.max() * .93, f" {t}", fontsize=9, color=WARN)
    ax.set_ylabel("하루 대여 건수"); ax.set_title("19개월 일별 대여량과 비어 있는 구간")
    save(fig, "f2_daily")

    # F3 날씨
    w = pd.read_parquet(ROOT / "data/raw/weather_openmeteo_daejeon.parquet")
    df = pd.DataFrame({"y": per_st, "rain": w.precipitation.to_numpy()[: len(ts)], "temp": w.temperature_2m.to_numpy()[: len(ts)]}).dropna()
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    rb = pd.cut(df.rain, [-.01, 0, 0.5, 1, 3, 100], labels=["0", "~0.5", "0.5~1", "1~3", "3mm+"])
    g = df.groupby(rb, observed=True).y.mean()
    ax[0].bar(range(len(g)), g.values, color=[ACC] + [WARN] * 4)
    ax[0].set_xticks(range(len(g))); ax[0].set_xticklabels(g.index.astype(str))
    ax[0].set_title("시간당 강수량별 이용"); ax[0].set_ylabel("대여소당 시간당 건수")
    tb = pd.cut(df.temp, [-20, 0, 10, 20, 28, 45], labels=["0도 미만", "0~10", "10~20", "20~28", "28도+"])
    g2 = df.groupby(tb, observed=True).y.mean()
    ax[1].bar(range(len(g2)), g2.values, color=ACC2)
    ax[1].set_xticks(range(len(g2))); ax[1].set_xticklabels(g2.index.astype(str))
    ax[1].set_title("기온별 이용")
    save(fig, "f3_weather")

    # F4 집중도 로렌츠
    tot = (rent * obs).sum(0)
    s = np.sort(tot)[::-1]
    cum = np.cumsum(s) / s.sum()
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(np.arange(1, N + 1) / N * 100, cum * 100, color=ACC, lw=2.2)
    ax.plot([0, 100], [0, 100], color=GREY, ls=":", lw=1.2)
    i50 = int(np.searchsorted(cum, .5)) + 1
    ax.scatter([i50 / N * 100], [50], color=WARN, zorder=5)
    ax.annotate(f"{i50}곳(상위 {i50/N*100:.0f}%)이 절반", (i50 / N * 100, 50), (25, 35), fontsize=9, color=WARN,
                arrowprops=dict(arrowstyle="->", color=WARN))
    ax.set_xlabel("대여소 누적 비율 (많은 순)"); ax.set_ylabel("대여 누적 비율 (%)")
    ax.set_title("수요 집중도 (지니 0.59)")
    save(fig, "f4_lorenz")

    # F5 상위 OD 흐름 지도
    od = pd.read_parquet(V2 / "od_month.parquet").groupby(["src", "dst"], as_index=False).n.sum()
    od = od[od.src != od.dst].nlargest(120, "n")
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.scatter(nodes.lon, nodes.lat, s=np.sqrt(tot) / 6 + 1.5, color=GREY, alpha=.35, lw=0)
    mx = od.n.max()
    for r in od.itertuples():
        ax.plot([nodes.lon[r.src], nodes.lon[r.dst]], [nodes.lat[r.src], nodes.lat[r.dst]],
                color=ACC, alpha=.35 + .5 * r.n / mx, lw=.6 + 3 * r.n / mx)
    ax.set_aspect(1 / np.cos(np.radians(36.35)))
    ax.set_title("가장 굵은 이동 120쌍")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    save(fig, "f5_odmap")

    # F6 복원 재고: 도시 전체 + 대여소 예시
    inv = np.load(V3 / "inventory_15min.npz")
    slots = pd.to_datetime(inv["slots"].astype("datetime64[m]"))
    stock = inv["stock"].astype(np.float32)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    cityst = pd.Series(stock.sum(1), index=slots).resample("D").mean()
    ax.plot(cityst.index, cityst.values, color=ACC, lw=1.1)
    ax.set_ylabel("복원 재고 합 (대)"); ax.set_title("도시 전체 복원 재고 (하루 평균)")
    save(fig, "f6_city_stock")

    top = np.argsort(-tot)[:3]
    wk = (slots >= "2025-05-12") & (slots < "2025-05-19")
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    for i, nd in enumerate(top):
        ax.plot(slots[wk], stock[wk, nd], lw=1.3, label=nodes["name"][nd][:18], color=[ACC, ACC2, WARN][i])
    ax.set_ylabel("복원 재고 (대)"); ax.set_title("붐비는 대여소 3곳의 한 주 재고 (2025-05-12 ~ 18)")
    ax.legend(frameon=False, fontsize=9)
    save(fig, "f7_station_stock")

    # F8 재고 0 비율 분포 + 시각별
    ih = pd.read_parquet(V3 / "inventory_hourly.parquet", columns=["ts", "node", "empty_frac", "rent", "stock_mean"])
    byst = ih.groupby("node").empty_frac.mean()
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    ax[0].hist(byst.values, bins=40, color=ACC, alpha=.85)
    ax[0].set_xlabel("대여소가 비어 있던 시간 비율"); ax[0].set_ylabel("대여소 수")
    ax[0].set_title("대여소별 결품 시간 비율")
    hh = ih.assign(h=ih.ts.dt.hour).groupby("h").agg(e=("empty_frac", "mean"), r=("rent", "mean"))
    ax2 = ax[1]
    ax2.plot(hh.index, hh.e, color=WARN, lw=2, label="비어 있는 비율")
    ax2.set_ylabel("비어 있는 비율", color=WARN); ax2.set_xlabel("시각")
    a3 = ax2.twinx(); a3.plot(hh.index, hh.r, color=ACC, lw=2, label="대여")
    a3.set_ylabel("시간당 대여", color=ACC); a3.grid(False)
    ax2.set_title("시각별 결품과 이용")
    save(fig, "f8_empty")

    json.dump({"figs": sorted(p.name for p in FIG.glob("*.png"))}, open(FIG / "index.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
