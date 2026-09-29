"""전처리 재설계를 위한 원자료 점검. 결과는 표준출력으로."""
import hashlib, re
import numpy as np, pandas as pd
from pathlib import Path
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
ROOT = Path(__file__).resolve().parents[1]
t = pd.read_parquet(ROOT/"data/processed/trips.parquet")
for c in ["rent_st","ret_st","rent_gu","ret_gu","rent_dong","ret_dong","src_file"]:
    t[c] = t[c].astype(str)
N = len(t)
def pct(x): return f"{x/N*100:.2f}%"
def sec(s): print(f"\n########## {s}")

sec("0. 202512 파일이 202501 과 같은지")
def md5(p):
    h = hashlib.md5()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(1<<22), b""): h.update(b)
    return h.hexdigest()
print("202501", md5(ROOT/"data/raw/202501.csv")); print("202512", md5(ROOT/"data/raw/202512.csv"))

sec("1. 시각 정밀도: 초가 00 인 비율 (월별)")
print(t.groupby(t.rent_ts.dt.to_period("M")).apply(lambda g: round((g.rent_ts.dt.second==0).mean(),3)).to_dict())

sec("2. 이용시간 열 vs 실제 시각 차이")
real = (t.ret_ts - t.rent_ts).dt.total_seconds()/60
diff = t.dur_min - np.floor(real)
print("이용시간(분) - floor(반납-대여) 분포:", diff.clip(-5,5).value_counts().sort_index().to_dict())
print("이용시간 분위:", t.dur_min.quantile([.001,.01,.05,.5,.95,.99,.999,.9999]).to_dict())

sec("3. 거리")
print("거리 분위:", t.dist_km.quantile([.01,.05,.25,.5,.95,.99,.999,.9999]).to_dict())
R=6371
la1,lo1,la2,lo2 = [np.radians(t[c].astype(float)) for c in ["rent_lat","rent_lon","ret_lat","ret_lon"]]
straight = 2*R*np.arcsin(np.sqrt(np.sin((la2-la1)/2)**2+np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2))
t["straight_km"]=straight
speed = t.dist_km/(real/60).replace(0,np.nan)
print("평균속도(km/h) 분위:", speed.quantile([.5,.9,.99,.999]).round(1).to_dict())
print("속도 30km/h 초과:", int((speed>30).sum()), pct((speed>30).sum()))
print("기록거리 < 직선거리*0.8 (다른 대여소, 직선 0.3km 이상):", int(((t.dist_km<t.straight_km*0.8)&(t.straight_km>0.3)).sum()))
print("거리 0 인데 다른 대여소 반납:", int(((t.dist_km==0)&(t.rent_st!=t.ret_st)).sum()))

sec("4. 같은 곳 반납 · 짧은 이용")
same = t.rent_st==t.ret_st
print("같은 곳 반납:", int(same.sum()), pct(same.sum()))
bins=[-1,0,1,2,3,5,10,30,60,1e9]
tab = pd.crosstab(pd.cut(t.dur_min,bins), same, normalize=False)
tab.columns=["다른곳","같은곳"]; print(tab)
print("같은 곳 반납 & 거리 0 의 이용시간 분포:", pd.cut(t.dur_min[same&(t.dist_km==0)],bins).value_counts().sort_index().to_dict())
print("같은 곳 반납 & 이용시간 5분 이상의 거리 분위:", t.dist_km[same&(t.dur_min>=5)].quantile([.1,.25,.5,.75]).to_dict())

sec("5. 대여소: 좌표·이름·구동 일관성")
g = t.groupby("rent_st").agg(n=("rent_st","size"), n_name=("rent_name","nunique"),
    lat_sd=("rent_lat","std"), lon_sd=("rent_lon","std"), n_gu=("rent_gu","nunique"),
    first=("rent_ts","min"), last=("rent_ts","max"))
print("대여소 수:", len(g), "| 이름 2개 이상:", int((g.n_name>1).sum()), "| 좌표 흔들림(>0.001도):", int(((g.lat_sd>1e-3)|(g.lon_sd>1e-3)).sum()), "| 구 2개 이상:", int((g.n_gu>1).sum()))
multi = t[t.rent_st.isin(g[g.n_name>1].index)].groupby(["rent_st","rent_name"]).agg(n=("rent_ts","size"),first=("rent_ts","min"),last=("rent_ts","max"))
print(multi.head(20))
print("좌표 범위:", t.rent_lat.min(), t.rent_lat.max(), t.rent_lon.min(), t.rent_lon.max())
print("대전 밖/0 좌표 기록:", int(((t.rent_lat<36.1)|(t.rent_lat>36.6)|(t.rent_lon<127.2)|(t.rent_lon>127.6)).sum()))
print("구 값:", t.rent_gu.value_counts().to_dict())
coords = t.groupby("rent_st")[["rent_lat","rent_lon"]].median().round(5)
dupc = coords[coords.duplicated(keep=False)].sort_values(["rent_lat","rent_lon"])
print("좌표가 완전히 같은 서로 다른 대여소:", len(dupc)); print(dupc.head(10))

sec("6. 대여소 수명: 늦게 생긴 곳 / 일찍 사라진 곳")
end = t.rent_ts.max(); start=t.rent_ts.min()
print("첫 등장이 2024-09 이후:", int((g["first"]>"2024-09-01").sum()), "| 마지막 등장이 2026-02 이전:", int((g["last"]<"2026-02-01").sum()))
print("월별 활동 대여소 수:", t.groupby(t.rent_ts.dt.to_period("M")).rent_st.nunique().to_dict())

sec("7. 이름으로 본 운영용 대여소 후보")
names = t.groupby("rent_st").rent_name.agg(lambda s: s.mode().iat[0])
pat = re.compile(r"관제|센터|창고|정비|테스트|임시|TEST|test|사무|본사|수리|보관|거점")
cand = names[names.str.contains(pat)]
print(pd.DataFrame({"name":cand, "n_rent":g.n.reindex(cand.index)}).sort_values("n_rent",ascending=False).head(30))

sec("8. 대여 수 vs 반납 수 불균형이 심한 대여소 (재배치 거점 의심)")
rc = t.rent_st.value_counts(); rt = t.ret_st.value_counts()
bal = pd.DataFrame({"rent":rc,"ret":rt}).fillna(0)
bal["ret_minus_rent"]=bal.ret-bal.rent; bal["ratio"]=(bal.ret+1)/(bal.rent+1); bal["name"]=names.reindex(bal.index)
print(bal[bal.rent+bal.ret>2000].sort_values("ratio").head(8)); print(bal[bal.rent+bal.ret>2000].sort_values("ratio").tail(8))

sec("9. 자전거별 흐름: 다음 대여 위치가 직전 반납 위치와 다른가 (=누군가 옮김)")
b = t[["bike_id","rent_ts","ret_ts","rent_st","ret_st"]].sort_values(["bike_id","rent_ts"])
nxt_st = b.groupby("bike_id").rent_st.shift(-1); nxt_ts = b.groupby("bike_id").rent_ts.shift(-1)
has = nxt_st.notna()
moved = has & (nxt_st!=b.ret_st)
overlap = has & (nxt_ts < b.ret_ts)
print("다음 대여가 있는 기록:", int(has.sum()), "| 위치 달라짐:", int(moved.sum()), f"{moved.sum()/has.sum()*100:.2f}%", "| 반납 전에 다음 대여 시작(겹침):", int(overlap.sum()))
gap_h = (nxt_ts - b.ret_ts).dt.total_seconds()/3600
print("위치 달라진 경우 간격(시간) 분위:", gap_h[moved].quantile([.1,.25,.5,.75,.9]).round(2).to_dict())
print("위치 달라진 경우 '옮겨진 도착지' 상위:", nxt_st[moved].map(names).value_counts().head(8).to_dict())
print("위치 달라진 경우 '떠난 곳' 상위:", b.ret_st[moved].map(names).value_counts().head(8).to_dict())
print("위치 달라짐 시간대:", nxt_ts[moved].dt.hour.value_counts().sort_index().to_dict())

sec("10. 자전거 수: 하루에 한 번 이상 쓰인 자전거")
print("번호 앞자리:", t.bike_id.str.extract(r"^([A-Z]+\d)")[0].value_counts().to_dict())
d = t.groupby(t.rent_ts.dt.to_period("M")).bike_id.nunique(); print("월별 쓰인 자전거 수:", d.to_dict())

sec("11. 몰림: 한 대여소에서 같은 1분에 5건 이상")
k = t.groupby(["rent_st", t.rent_ts.dt.floor("min")]).size()
burst = k[k>=5]
print("그런 분 수:", len(burst), "| 해당 대여 수:", int(burst.sum()), pct(burst.sum()))
print("대여소 상위:", burst.groupby(level=0).sum().sort_values(ascending=False).head(8).rename(names).to_dict())

sec("12. 관제센터(ST1220) 상세")
c = t[t.rent_st=="ST1220"]
print("좌표:", c.rent_lat.median(), c.rent_lon.median(), "| 구/동:", c.rent_gu.mode().iat[0], c.rent_dong.mode().iat[0])
cb = b[b.rent_st=="ST1220"].index
prev_ret = b.groupby("bike_id").ret_st.shift(1)
print("관제센터 대여 직전 반납 위치 상위:", prev_ret.loc[cb].map(names).value_counts().head(8).to_dict())
print("관제센터 대여 직전 반납 == 관제센터 비율:", round((prev_ret.loc[cb]=="ST1220").mean(),3))
prev_ret_ts = b.groupby("bike_id").ret_ts.shift(1)
gh = (b.rent_ts - prev_ret_ts).dt.total_seconds()/60
print("관제센터 대여 직전 반납과의 간격(분) 분위:", gh.loc[cb].quantile([.1,.25,.5,.75,.9]).round(1).to_dict())
print("관제센터 대여 중 같은 곳 반납·거리0:", int(((c.ret_st=="ST1220")&(c.dist_km==0)).sum()), "| 이용시간 분위:", c.dur_min.quantile([.25,.5,.75]).to_dict())
