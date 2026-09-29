"""대여소(노드) 주변 공간 특징 만들기.

출처
  - OpenStreetMap (Overpass API): 지하철역, 버스정류장, 대학, 초중고, 상가·음식점·카페, 아파트, 공원, 자전거도로, 하천
  - Open-Meteo Elevation API: 고도 (Copernicus DEM 90m)
  - 대여소 좌표 자체: 주변 대여소 밀도, 시청(둔산)까지 거리

원자료는 data/raw/osm_*.json, elevation_nodes.json 으로 저장해 다시 받지 않는다.
산출: data/processed/v2/node_spatial.parquet (node 별 특징), report 출력
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
V2 = ROOT / "data" / "processed" / "v2"
BBOX = (36.18, 127.24, 36.52, 127.58)  # 대전 (남, 서, 북, 동)
OVERPASS = "https://overpass-api.de/api/interpreter"

QUERIES = {
    "subway": 'node["railway"="station"]["station"="subway"]{b};node["railway"="station"]{b};',
    "bus_stop": 'node["highway"="bus_stop"]{b};',
    "university": 'nwr["amenity"~"^(university|college)$"]{b};',
    "school": 'nwr["amenity"="school"]{b};',
    "commerce": 'node["shop"]{b};node["amenity"~"^(restaurant|cafe|fast_food|bar|pub)$"]{b};',
    "apartment": 'way["building"="apartments"]{b};',
    "park": 'nwr["leisure"="park"]{b};',
    "cycleway": 'way["highway"="cycleway"]{b};way["cycleway"~"^(lane|track)$"]{b};way["bicycle"="designated"]{b};',
    "river": 'way["waterway"="river"]{b};',
}
GEOM = {"cycleway", "river"}


def hav_m(a1, o1, a2, o2):
    a1, o1, a2, o2 = (np.radians(np.asarray(x, float)) for x in (a1, o1, a2, o2))
    return 2 * 6371000 * np.arcsin(np.sqrt(np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin((o2 - o1) / 2) ** 2))


def fetch_osm(name: str) -> dict:
    path = RAW / f"osm_{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    b = "({},{},{},{})".format(*BBOX)
    out = "out geom;" if name in GEOM else "out center;"
    q = f"[out:json][timeout:180];({QUERIES[name].format(b=b)});{out}"
    for attempt in range(4):
        r = requests.post(OVERPASS, data={"data": q}, timeout=300, headers={"User-Agent": "tashu-demand-study"})
        if r.status_code == 200:
            js = r.json()
            path.write_text(json.dumps(js, ensure_ascii=False), encoding="utf-8")
            return js
        print(f"  overpass {name} {r.status_code}, retry", flush=True)
        time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"overpass failed: {name}")


def points(js):
    pts = []
    for e in js["elements"]:
        if "lat" in e:
            pts.append((e["lat"], e["lon"]))
        elif "center" in e:
            pts.append((e["center"]["lat"], e["center"]["lon"]))
        elif "geometry" in e:
            g = e["geometry"]
            pts.append((np.mean([p["lat"] for p in g]), np.mean([p["lon"] for p in g])))
    return np.array(sorted(set(pts))) if pts else np.zeros((0, 2))


def segments(js):
    """선 geometry -> (중점 위도, 중점 경도, 길이 m) 배열."""
    segs = []
    seen = set()
    for e in js["elements"]:
        if "geometry" not in e or e["id"] in seen:
            continue
        seen.add(e["id"])
        g = e["geometry"]
        for a, b in zip(g[:-1], g[1:]):
            L = hav_m(a["lat"], a["lon"], b["lat"], b["lon"])
            segs.append(((a["lat"] + b["lat"]) / 2, (a["lon"] + b["lon"]) / 2, float(L)))
    return np.array(segs) if segs else np.zeros((0, 3))


def elevation(lat, lon):
    path = RAW / "elevation_nodes.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    keys = [f"{a:.5f},{o:.5f}" for a, o in zip(lat, lon)]
    todo = [k for k in dict.fromkeys(keys) if k not in cache]
    for i in range(0, len(todo), 100):
        chunk = todo[i : i + 100]
        la = ",".join(k.split(",")[0] for k in chunk)
        lo = ",".join(k.split(",")[1] for k in chunk)
        for attempt in range(8):
            r = requests.get("https://api.open-meteo.com/v1/elevation", params={"latitude": la, "longitude": lo}, timeout=60)
            if r.status_code == 200:
                break
            print(f"  elevation {r.status_code}, wait", flush=True)
            time.sleep(30 * (attempt + 1))
        r.raise_for_status()
        for k, v in zip(chunk, r.json()["elevation"]):
            cache[k] = v
        time.sleep(3)
        path.write_text(json.dumps(cache))
    path.write_text(json.dumps(cache))
    return np.array([cache[k] for k in keys], float)


def main():
    nodes = pd.read_parquet(V2 / "nodes.parquet")
    lat, lon = nodes.lat.to_numpy(), nodes.lon.to_numpy()
    feat = pd.DataFrame({"node": nodes.node})
    rep = {}

    def dist_to(P):
        if len(P) == 0:
            return np.full(len(lat), np.nan)
        return hav_m(lat[:, None], lon[:, None], P[None, :, 0], P[None, :, 1])

    for name in QUERIES:
        js = fetch_osm(name)
        if name in GEOM:
            S = segments(js)
            rep[name] = f"{len(S)} segments"
            D = dist_to(S[:, :2])
            if name == "cycleway":
                feat["cycleway_km_500m"] = (np.where(D <= 500, S[None, :, 2], 0).sum(1) / 1000)
                feat["dist_cycleway_m"] = D.min(1)
            else:
                feat["dist_river_m"] = D.min(1)
            continue
        P = points(js)
        rep[name] = f"{len(P)} points"
        D = dist_to(P)
        if name == "subway":
            feat["dist_station_m"] = D.min(1)
        elif name == "bus_stop":
            feat["n_bus_300m"] = (D <= 300).sum(1)
        elif name == "university":
            feat["dist_univ_m"] = D.min(1)
            feat["n_univ_1km"] = (D <= 1000).sum(1)
        elif name == "school":
            feat["n_school_500m"] = (D <= 500).sum(1)
        elif name == "commerce":
            feat["n_commerce_300m"] = (D <= 300).sum(1)
        elif name == "apartment":
            feat["n_apt_bldg_500m"] = (D <= 500).sum(1)
        elif name == "park":
            feat["dist_park_m"] = D.min(1)

    elev = elevation(lat, lon)
    feat["elevation_m"] = elev
    DD = hav_m(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    np.fill_diagonal(DD, np.inf)
    near = DD <= 1000
    feat["elev_minus_1km_mean"] = elev - np.array([elev[m].mean() if m.any() else e for m, e in zip(near, elev)])
    feat["n_tashu_300m"] = (DD <= 300).sum(1)
    feat["n_tashu_1km"] = near.sum(1)
    feat["dist_nearest_tashu_m"] = DD.min(1)
    feat["dist_cityhall_m"] = hav_m(lat, lon, 36.3504, 127.3845)

    feat.to_parquet(V2 / "node_spatial.parquet", index=False)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    print(feat.describe().T.round(2).to_string())
    # 공간 특징이 대여소 평균 수요와 얼마나 관련 있나 (스피어만)
    z = np.load(V2 / "panel.npz")
    obs = z["node_active"] & z["hour_ok"][:, None]
    mean_rent = (z["rent"] * obs).sum(0) / np.maximum(obs.sum(0), 1)
    corr = feat.drop(columns="node").apply(lambda c: pd.Series(c).corr(pd.Series(mean_rent), method="spearman"))
    print("\n대여소 평균 수요와 순위상관")
    print(corr.sort_values(key=np.abs, ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
