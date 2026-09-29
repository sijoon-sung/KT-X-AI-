"""2단계: 15분 집계 -> 대여소×시간 패널 -> 학습용 피처/타깃.

제안서 설계대로 세 구간을 입력으로 쓴다.
  - 직전 L 시간 흐름 (기본 3시간: lag 1..L 의 rent/ret/net, 누적합)
  - 전일 같은 시간대
  - 전주 같은 요일·시간대
타깃: 앞으로 H 시간 동안의 순유출입 net = ret - rent (음수면 자전거가 빠져나감),
      그리고 같은 구간의 rent, ret 도 함께 저장해 --target 으로 고를 수 있게 한다.

실행:  python src/features.py --freq 60 --horizon 1 --lags 3
산출:  data/processed/dataset_f{freq}_h{horizon}/{train,val,test}.parquet + meta.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def build_panel(freq_min: int):
    counts = pd.read_parquet(PROC / "counts_15min.parquet")
    counts["ts"] = counts["ts"].dt.floor(f"{freq_min}min")
    counts = counts.groupby(["ts", "station_id"], as_index=False)[["rent", "ret"]].sum()

    stations = pd.read_parquet(PROC / "stations.parquet")
    st_ids = sorted(counts["station_id"].unique())
    stations = stations.set_index("station_id").reindex(st_ids).reset_index()
    # 마지막 날은 반납 기록만 있어(대여 0) 가짜 0 이 되므로, 대여 기록이 있는 마지막 스텝까지만 쓴다
    last_rent = counts.loc[counts["rent"] > 0, "ts"].max()
    ts_index = pd.date_range(counts["ts"].min().floor("D"), last_rent, freq=f"{freq_min}min")
    counts = counts[counts["ts"] <= last_rent]

    S, T = len(st_ids), len(ts_index)
    si = pd.Index(st_ids).get_indexer(counts["station_id"])
    ti = ts_index.get_indexer(counts["ts"])
    rent = np.zeros((T, S), dtype=np.float32)
    ret = np.zeros((T, S), dtype=np.float32)
    rent[ti, si] = counts["rent"].to_numpy()
    ret[ti, si] = counts["ret"].to_numpy()
    return ts_index, stations, rent, ret


def shift(a: np.ndarray, k: int) -> np.ndarray:
    """시간축(0번 축)으로 k 스텝 과거 값. 앞쪽은 NaN."""
    out = np.full_like(a, np.nan)
    if k > 0:
        out[k:] = a[:-k]
    elif k < 0:
        out[:k] = a[-k:]
    else:
        out[:] = a
    return out


def fwd_sum(a: np.ndarray, n: int) -> np.ndarray:
    """t+1 .. t+n 구간 합 (미래). 뒤쪽 n 스텝은 NaN."""
    cs = np.concatenate([np.zeros((1,) + a.shape[1:], dtype=np.float64), np.cumsum(a, axis=0, dtype=np.float64)])
    out = np.full(a.shape, np.nan, dtype=np.float32)
    out[: a.shape[0] - n] = (cs[n + 1 :] - cs[1 : a.shape[0] - n + 1]).astype(np.float32)
    return out


def bwd_sum(a: np.ndarray, n: int) -> np.ndarray:
    """t-n+1 .. t 구간 합 (과거, 현재 포함)."""
    cs = np.concatenate([np.zeros((1,) + a.shape[1:], dtype=np.float64), np.cumsum(a, axis=0, dtype=np.float64)])
    out = np.full(a.shape, np.nan, dtype=np.float32)
    out[n - 1 :] = (cs[n:] - cs[:-n]).astype(np.float32)
    return out


def main(args):
    freq = args.freq
    per_hour = 60 // freq
    per_day = 24 * per_hour
    L = args.lags * per_hour  # 직전 창 스텝 수
    H = args.horizon * per_hour  # 예측 구간 스텝 수

    ts_index, stations, rent, ret = build_panel(freq)
    net = ret - rent
    T, S = rent.shape
    print(f"panel: {T:,} steps x {S:,} stations, freq={freq}min")

    # ----- 결측일 마스크: 해당 스텝 자체 + 그 스텝을 참조하는 lag/target 이 모두 무효 -----
    missing_days = set(json.loads((PROC / "missing_days.json").read_text()))
    day_str = ts_index.strftime("%Y-%m-%d")
    bad_step = np.array([d in missing_days for d in day_str])  # (T,)
    # 첫날은 05:00 부터 기록 시작 → 첫날도 제외
    bad_step |= ts_index.normalize() == ts_index.normalize().min()

    feats: dict[str, np.ndarray] = {}
    # ----- 직전 흐름 -----
    for k in range(1, L + 1):
        feats[f"rent_lag{k}"] = shift(rent, k)
        feats[f"ret_lag{k}"] = shift(ret, k)
        feats[f"net_lag{k}"] = shift(net, k)
    feats[f"rent_sum{args.lags}h"] = shift(bwd_sum(rent, L), 1)
    feats[f"ret_sum{args.lags}h"] = shift(bwd_sum(ret, L), 1)
    feats[f"net_sum{args.lags}h"] = shift(bwd_sum(net, L), 1)
    feats["rent_sum6h"] = shift(bwd_sum(rent, 6 * per_hour), 1)
    feats["net_sum6h"] = shift(bwd_sum(net, 6 * per_hour), 1)
    # ----- 전일 / 전주 같은 시간대 (예측 구간과 같은 길이 H 의 합) -----
    fs_rent, fs_ret, fs_net = fwd_sum(rent, H), fwd_sum(ret, H), fwd_sum(net, H)
    feats["rent_next_yday"] = shift(fs_rent, per_day)
    feats["ret_next_yday"] = shift(fs_ret, per_day)
    feats["net_next_yday"] = shift(fs_net, per_day)
    feats["rent_next_lweek"] = shift(fs_rent, 7 * per_day)
    feats["ret_next_lweek"] = shift(fs_ret, 7 * per_day)
    feats["net_next_lweek"] = shift(fs_net, 7 * per_day)
    # 최근 7일 일평균 대여 (요즘 이용 수준)
    feats["rent_mean7d"] = shift(bwd_sum(rent, 7 * per_day), 1) / 7.0

    # ----- 타깃 -----
    targets = {"y_rent": fs_rent, "y_ret": fs_ret, "y_net": fs_net}

    # ----- 유효 행: 참조하는 모든 스텝이 결측일이 아니어야 함 -----
    valid = ~bad_step.copy()
    bad_cs = np.concatenate([[0], np.cumsum(bad_step)])

    def window_has_bad(start_off, end_off):
        """각 t 에 대해 [t+start_off, t+end_off] 구간에 결측 스텝이 있는지 (범위 밖이면 True)."""
        idx = np.arange(T)
        lo = idx + start_off
        hi = idx + end_off
        out = (lo < 0) | (hi >= T)
        lo_c = np.clip(lo, 0, T)
        hi_c = np.clip(hi + 1, 0, T)
        out |= (bad_cs[hi_c] - bad_cs[lo_c]) > 0
        return out

    valid &= ~window_has_bad(-7 * per_day, 0)  # 과거 7일 (lags, 전주, 7일 평균 모두 포함)
    valid &= ~window_has_bad(1, H)  # 미래 H 스텝
    valid &= ~window_has_bad(-7 * per_day + 1, -7 * per_day + H)  # 전주의 미래 구간
    valid &= ~window_has_bad(-per_day + 1, -per_day + H)  # 전일의 미래 구간

    # ----- 지난 4주 같은 시간대 평균 (시점마다 과거만 사용 → 어느 기간을 시험해도 정답이 섞이지 않음) -----
    # 결측일은 건너뛰고 평균, 유효한 날이 7일 미만이면 행 제외
    acc_rent = np.zeros((T, S), np.float32)
    acc_ret = np.zeros((T, S), np.float32)
    acc_all = np.zeros((T, S), np.float32)
    cnt = np.zeros(T, np.float32)
    for k in range(1, 29):
        ok = ~window_has_bad(-k * per_day + 1, -k * per_day + H)
        okf = ok.astype(np.float32)[:, None]
        acc_rent += np.nan_to_num(shift(fs_rent, k * per_day)) * okf
        acc_ret += np.nan_to_num(shift(fs_ret, k * per_day)) * okf
        cnt += ok
    den = np.maximum(cnt, 1.0)[:, None]
    feats["prof_rent_4w"] = acc_rent / den
    feats["prof_ret_4w"] = acc_ret / den
    feats["prof_net_4w"] = feats["prof_ret_4w"] - feats["prof_rent_4w"]
    del acc_rent, acc_ret, acc_all
    # 지난 28일 일평균 대여 (결측일 제외)
    day_ok = (~bad_step).astype(np.float32)[:, None]
    n_ok = shift(bwd_sum(np.broadcast_to(day_ok, (T, 1)).astype(np.float32), 28 * per_day), 1)[:, 0] / per_day
    feats["rent_mean28d"] = shift(bwd_sum(rent * day_ok, 28 * per_day), 1) / np.maximum(np.nan_to_num(n_ok), 1.0)[:, None]
    valid &= cnt >= 7

    # ----- 시간 피처 -----
    kr_hol = holidays.KR(years=range(ts_index.year.min(), ts_index.year.max() + 1))
    hour = ts_index.hour.to_numpy() + ts_index.minute.to_numpy() / 60.0
    dow = ts_index.dayofweek.to_numpy()
    is_hol = np.array([d in kr_hol for d in ts_index.date])
    time_feats = {
        "hour": hour,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow": dow,
        "is_weekend": (dow >= 5).astype(np.float32),
        "is_holiday": is_hol.astype(np.float32),
        "is_offday": ((dow >= 5) | is_hol).astype(np.float32),
        "month": ts_index.month.to_numpy(),
        "doy_sin": np.sin(2 * np.pi * ts_index.dayofyear.to_numpy() / 365.25),
        "doy_cos": np.cos(2 * np.pi * ts_index.dayofyear.to_numpy() / 365.25),
    }
    # ----- 대여소 정적 피처 -----
    st_feats = {
        "lat": stations["lat"].to_numpy(np.float32),
        "lon": stations["lon"].to_numpy(np.float32),
        "capacity": stations["capacity"].to_numpy(np.float32),
    }

    # ----- 분할 -----
    if args.layout == "all":
        split_bounds = {"all": (None, None)}
    else:
        split_bounds = {
            "train": (None, pd.Timestamp(args.val_start)),
            "val": (pd.Timestamp(args.val_start), pd.Timestamp(args.test_start)),
            "test": (pd.Timestamp(args.test_start), None),
        }
    # 대여소 거르기용 (피처 아님): 전체 기간 유효일 기준 일평균 대여
    st_activity = rent[~bad_step].sum(0) / max(1.0, (~bad_step).sum() / per_day)

    out_dir = PROC / f"dataset_f{freq}_h{args.horizon}"
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_names = list(feats) + list(time_feats) + list(st_feats)
    step_idx = np.arange(T)
    st_idx = np.arange(S)
    for name, (lo, hi) in split_bounds.items():
        m = valid.copy()
        if lo is not None:
            m &= ts_index >= lo
        if hi is not None:
            m &= ts_index < hi
        if args.min_station_rent > 0:
            m = m[:, None] & (st_activity >= args.min_station_rent)[None, :]
        else:
            m = np.broadcast_to(m[:, None], (T, S))
        tt, ss = np.nonzero(m)
        cols = {"ts_idx": tt.astype(np.int32), "st_idx": ss.astype(np.int32)}
        for k, v in feats.items():
            cols[k] = v[tt, ss]
        for k, v in time_feats.items():
            cols[k] = np.asarray(v, dtype=np.float32)[tt]
        for k, v in st_feats.items():
            cols[k] = v[ss]
        for k, v in targets.items():
            cols[k] = v[tt, ss]
        df = pd.DataFrame(cols)
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
        print(f"{name}: {len(df):,} rows  ({ts_index[tt.min()]} ~ {ts_index[tt.max()]})")

    meta = {
        "freq_min": freq,
        "horizon_h": args.horizon,
        "lags_h": args.lags,
        "features": feat_names,
        "targets": list(targets),
        "n_stations": S,
        "station_ids": stations["station_id"].tolist(),
        "ts_start": str(ts_index[0]),
        "val_start": args.val_start,
        "test_start": args.test_start,
        "layout": args.layout,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.DataFrame({"ts_idx": step_idx, "ts": ts_index}).to_parquet(out_dir / "ts_index.parquet", index=False)
    stations.assign(st_idx=st_idx).to_parquet(out_dir / "stations.parquet", index=False)
    print("saved to", out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--freq", type=int, default=60, help="집계 단위(분): 15 또는 60")
    p.add_argument("--horizon", type=int, default=1, help="예측 구간(시간)")
    p.add_argument("--lags", type=int, default=3, help="직전 창 길이(시간)")
    p.add_argument("--val-start", default="2026-01-01")
    p.add_argument("--test-start", default="2026-02-01")
    p.add_argument("--layout", choices=["split", "all"], default="split", help="all: 한 파일로 저장 (교차검증용)")
    p.add_argument("--min-station-rent", type=float, default=0.0, help="train 일평균 대여가 이 값 미만인 대여소 제외")
    main(p.parse_args())
