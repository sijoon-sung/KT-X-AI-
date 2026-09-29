"""'같은 곳' 판단 기준 비교: 대여소 번호 / 구 / 동 / 위도·경도."""
import numpy as np, pandas as pd
from pathlib import Path
pd.set_option("display.width", 250)
ROOT = Path(__file__).resolve().parents[1]
t = pd.read_parquet(ROOT/"data/processed/trips.parquet")
for c in ["rent_st","ret_st","rent_gu","ret_gu","rent_dong","ret_dong"]: t[c]=t[c].astype(str)
N=len(t)
def hav_m(a1,o1,a2,o2):
    a1,o1,a2,o2=[np.radians(np.asarray(x,float)) for x in (a1,o1,a2,o2)]
    return 2*6371000*np.arcsin(np.sqrt(np.sin((a2-a1)/2)**2+np.cos(a1)*np.cos(a2)*np.sin((o2-o1)/2)**2))
t["d_m"]=hav_m(t.rent_lat,t.rent_lon,t.ret_lat,t.ret_lon)
def sec(s): print(f"\n########## {s}")

sec("1. 행마다 좌표가 대여소 고정 좌표인가, 자전거 GPS 인가")
k = t.groupby(["rent_st", t.rent_ts.dt.to_period("M")]).agg(n_coord=("rent_lat", lambda s: s.round(6).nunique()), n=("rent_lat","size"))
print("대여소-월별 서로 다른 좌표 개수 분포:", k.n_coord.value_counts().sort_index().head(8).to_dict())

sec("2. 네 기준의 '같은 곳 반납' 건수")
same_id = t.rent_st==t.ret_st
same_gu = t.rent_gu==t.ret_gu
same_dong = (t.rent_gu==t.ret_gu)&(t.rent_dong==t.ret_dong)
same_xy = t.d_m < 1
print(pd.DataFrame({"기준":["대여소 번호","구","구+동","좌표 1m 이내"],
    "건수":[same_id.sum(),same_gu.sum(),same_dong.sum(),same_xy.sum()]}).assign(비율=lambda d:(d.건수/N*100).round(2)).to_string(index=False))

sec("3. 번호 기준과 좌표 기준이 어긋나는 경우")
print("번호 같음 & 좌표 1m 넘게 다름:", int((same_id&~same_xy).sum()))
print("  그 거리(m) 분위:", pd.Series(t.d_m[same_id&~same_xy]).quantile([.5,.9,.99]).round(0).to_dict() if (same_id&~same_xy).any() else "-")
print("번호 다름 & 좌표 1m 이내:", int((~same_id&same_xy).sum()))
x = t[~same_id&same_xy]
if len(x): print(x.groupby(["rent_st","ret_st"]).size().sort_values(ascending=False).head(8).to_dict())
print("번호 다름 & 50m 이내:", int((~same_id&(t.d_m<50)).sum()), "| 100m 이내:", int((~same_id&(t.d_m<100)).sum()))

sec("4. 구·동이 같은데 다른 대여소로 간 기록 (서구→서구 등)")
diff_id = ~same_id & (t.rent_st!="ST1220") & (t.ret_st!="ST1220")
print("다른 대여소 이동 중 같은 구:", round((same_gu&diff_id).sum()/diff_id.sum()*100,1),"%",
      "| 같은 구+동:", round((same_dong&diff_id).sum()/diff_id.sum()*100,1),"%")
print("같은 구·다른 동 이동 거리(m) 분위:", t.d_m[diff_id&same_gu&~same_dong].quantile([.1,.5,.9]).round(0).to_dict())
print("같은 구·같은 동·다른 대여소 이동 거리(m) 분위:", t.d_m[diff_id&same_dong].quantile([.1,.5,.9]).round(0).to_dict())

sec("5. 구·동 글자와 좌표가 맞는가: 한 좌표에 구·동 값이 여러 개인 경우")
cg = t.groupby([t.rent_lat.round(5), t.rent_lon.round(5)]).agg(n_gu=("rent_gu","nunique"), n_dong=("rent_dong","nunique"), n=("rent_gu","size"))
print("좌표 수:", len(cg), "| 구 2개 이상인 좌표:", int((cg.n_gu>1).sum()), "| 동 2개 이상인 좌표:", int((cg.n_dong>1).sum()))
bad = cg[cg.n_dong>1].sort_values("n",ascending=False).head(5)
for (la,lo),_ in bad.iterrows():
    q=t[(t.rent_lat.round(5)==la)&(t.rent_lon.round(5)==lo)]
    print(la,lo, q.groupby(["rent_st","rent_gu","rent_dong"]).agg(n=("rent_ts","size"),first=("rent_ts","min"),last=("rent_ts","max")).to_dict("index"))
print("동 값 예시(상위 15):", t.rent_dong.value_counts().head(15).to_dict())
print("'신탄진' 구 값 기록:", t[t.rent_gu=="신탄진"][["rent_st","rent_name","rent_dong"]].drop_duplicates().to_dict("records"))

sec("6. 취소 추정 기록을 좌표 기준으로 다시 세기")
short = t.dur_min<=1
print("번호 같음 & 1분 이하:", int((same_id&short).sum()), "| 좌표 1m 이내 & 1분 이하:", int((same_xy&short).sum()),
      "| 좌표 50m 이내 & 1분 이하:", int(((t.d_m<50)&short).sum()))

sec("7. '옮겨짐'을 좌표 기준으로 다시 세기 (관제센터 제외)")
b = t[["bike_id","rent_ts","ret_ts","rent_st","ret_st","rent_lat","rent_lon","ret_lat","ret_lon"]].sort_values(["bike_id","rent_ts"])
g = b.groupby("bike_id")
nx_st, nx_la, nx_lo = g.rent_st.shift(-1), g.rent_lat.shift(-1), g.rent_lon.shift(-1)
has = nx_st.notna() & (b.ret_st!="ST1220") & (nx_st!="ST1220")
dm = hav_m(b.ret_lat, b.ret_lon, nx_la.fillna(0), nx_lo.fillna(0))
print("번호 다름:", int((has&(nx_st!=b.ret_st)).sum()), "| 좌표 50m 넘게 다름:", int((has&(dm>50)).sum()),
      "| 번호 다른데 50m 이내:", int((has&(nx_st!=b.ret_st)&(dm<=50)).sum()), "| 번호 같은데 50m 넘게 다름:", int((has&(nx_st==b.ret_st)&(dm>50)).sum()))
y = has&(nx_st==b.ret_st)&(dm>50)
if y.any(): print("  번호 같은데 멀어진 경우 거리(m) 분위:", pd.Series(dm[y]).quantile([.5,.9]).round(0).to_dict(), "| 대여소 상위:", b.ret_st[y].value_counts().head(5).to_dict())
