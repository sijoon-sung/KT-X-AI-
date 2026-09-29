import numpy as np, pandas as pd
from pathlib import Path
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
ROOT = Path(__file__).resolve().parents[1]
t = pd.read_parquet(ROOT/"data/processed/trips.parquet")
for c in ["rent_st","ret_st","rent_gu","ret_gu","rent_dong","ret_dong"]: t[c]=t[c].astype(str)
names = t.groupby("rent_st").rent_name.agg(lambda s: s.mode().iat[0])
R=6371
def hav(a1,o1,a2,o2):
    a1,o1,a2,o2=[np.radians(np.asarray(x,float)) for x in (a1,o1,a2,o2)]
    return 2*R*np.arcsin(np.sqrt(np.sin((a2-a1)/2)**2+np.cos(a1)*np.cos(a2)*np.sin((o2-o1)/2)**2))
t["straight"]=hav(t.rent_lat,t.rent_lon,t.ret_lat,t.ret_lon)
real=(t.ret_ts-t.rent_ts).dt.total_seconds()/60
def sec(s): print(f"\n########## {s}")

sec("A. 관제센터로 '반납'된 기록")
r = t[(t.ret_st=="ST1220")&(t.rent_st!="ST1220")]
print("건수", len(r), "| 월별", r.groupby(r.ret_ts.dt.to_period("M")).size().to_dict())
print("기록 거리 분위", r.dist_km.quantile([.25,.5,.75]).to_dict(), "| 대여소→관제센터 직선거리 분위", r.straight.quantile([.25,.5,.75]).round(1).to_dict())
print("이용시간 분위", r.dur_min.quantile([.25,.5,.75]).to_dict())
print("대여 구 분포", r.rent_gu.value_counts().to_dict())
c = t[(t.rent_st=="ST1220")&(t.ret_st!="ST1220")]
print("관제센터→다른곳 기록 거리 분위", c.dist_km.quantile([.25,.5,.75]).to_dict(), "| 직선거리 분위", c.straight.quantile([.25,.5,.75]).round(1).to_dict())
cc = t[(t.rent_st=="ST1220")&(t.ret_st=="ST1220")]
print("관제센터→관제센터", len(cc), "거리 분위", cc.dist_km.quantile([.25,.5,.75]).to_dict(), "이용시간 분위", cc.dur_min.quantile([.25,.5,.75]).to_dict())
# 관제센터로 반납된 자전거가 다음에 어디서 빌려지나
b = t[["bike_id","rent_ts","ret_ts","rent_st","ret_st"]].sort_values(["bike_id","rent_ts"]).reset_index(drop=True)
b["next_st"]=b.groupby("bike_id").rent_st.shift(-1); b["next_ts"]=b.groupby("bike_id").rent_ts.shift(-1)
x=b[b.ret_st=="ST1220"]
print("관제센터 반납 뒤 다음 대여 위치: 관제센터", round((x.next_st=="ST1220").mean(),3), "| 다른 곳", round(((x.next_st!="ST1220")&x.next_st.notna()).mean(),3))
print("관제센터 반납→다음 대여 간격(분) 분위", ((x.next_ts-x.ret_ts).dt.total_seconds()/60).quantile([.1,.25,.5,.75,.9]).round(0).to_dict())

sec("B. 속도 30km/h 초과 기록의 이용시간")
sp = t.dist_km/(real/60).replace(0,np.nan)
f = sp>30
print(pd.cut(t.dur_min[f],[-1,0,1,2,5,10,30,1e9]).value_counts().sort_index().to_dict())
print("그중 같은 곳 반납", round((t.rent_st[f]==t.ret_st[f]).mean(),3), "| 거리 분위", t.dist_km[f].quantile([.25,.5,.75,.99]).to_dict())

sec("C. 다른 대여소로 1분 이하 반납")
g = t[(t.rent_st!=t.ret_st)&(t.dur_min<=1)]
print("건수", len(g), "| 직선거리 분위(km)", g.straight.quantile([.25,.5,.75,.9]).round(2).to_dict())
print("관제센터 관련 비율", round(((g.rent_st=="ST1220")|(g.ret_st=="ST1220")).mean(),3))

sec("D. 기록 거리 vs 직선거리 (다른 대여소, 관제센터 제외, 직선 0.5km 이상)")
h = t[(t.rent_st!=t.ret_st)&(t.rent_st!="ST1220")&(t.ret_st!="ST1220")&(t.straight>=0.5)]
ratio = h.dist_km/h.straight
print("비율 분위", ratio.quantile([.05,.1,.25,.5,.75,.9]).round(2).to_dict(), "| 기록거리 0 비율", round((h.dist_km==0).mean(),3))
print("월별 기록거리0 비율", h.groupby(h.rent_ts.dt.to_period("M")).apply(lambda q: round((q.dist_km==0).mean(),3)).to_dict())

sec("E. 구가 2개인 대여소 / 좌표가 움직인 대여소")
gg = t.groupby("rent_st").rent_gu.agg(lambda s: s.value_counts().to_dict())
two = gg[gg.map(len)>1]; print(two.head(8).to_dict())
co = t.groupby(["rent_st", t.rent_ts.dt.to_period("M")])[["rent_lat","rent_lon"]].median().reset_index()
jump = co.groupby("rent_st").apply(lambda q: hav(q.rent_lat.min(),q.rent_lon.min(),q.rent_lat.max(),q.rent_lon.max()).item()*1000)
print("월별 중앙 좌표 최대 이동(m) 분위", jump.quantile([.5,.9,.95,.99,1]).round(0).to_dict())
print("200m 이상 옮겨진 대여소", int((jump>200).sum())); print(jump[jump>200].sort_values(ascending=False).head(8).rename(names).round(0).to_dict())

sec("F. 대여소 운영 기간: 첫/마지막 기록, 중간의 긴 공백")
ev = pd.concat([t[["rent_st","rent_ts"]].set_axis(["st","ts"],axis=1), t[["ret_st","ret_ts"]].set_axis(["st","ts"],axis=1)])
ev["d"]=ev.ts.dt.floor("D")
days = ev.groupby("st").d.agg(["min","max","nunique"])
all_days = pd.date_range("2024-08-01","2026-03-31")
missing = pd.to_datetime(pd.read_json(ROOT/"data/processed/missing_days.json")[0])
valid_days = all_days.difference(missing)
days["span_valid_days"] = days.apply(lambda r: ((valid_days>=r["min"])&(valid_days<=r["max"])).sum(), axis=1)
days["active_share"] = days["nunique"]/days["span_valid_days"]
print("첫 기록이 데이터 시작 7일 이후인 대여소", int((days["min"]>"2024-08-08").sum()), "| 마지막 기록이 끝 7일 이전", int((days["max"]<"2026-03-24").sum()))
print("운영기간 중 기록 있는 날 비율 분위", days.active_share.quantile([.01,.05,.1,.25,.5]).round(2).to_dict())
# 가장 긴 연속 공백(유효일 기준)
dd = ev.groupby("st").d.apply(lambda s: np.sort(s.unique()))
def longest_gap(arr):
    if len(arr)<2: return 0
    idx = valid_days.get_indexer(pd.DatetimeIndex(arr)); idx=idx[idx>=0]
    return int(np.diff(np.sort(idx)).max()-1) if len(idx)>1 else 0
lg = dd.map(longest_gap)
print("가장 긴 무기록 구간(유효일) 분위", lg.quantile([.5,.9,.95,.99]).to_dict(), "| 30일 이상 공백 있는 대여소", int((lg>=30).sum()))
panel_days = len(valid_days)*len(days)
pre = days.apply(lambda r: ((valid_days<r["min"])|(valid_days>r["max"])).sum(), axis=1).sum()
print(f"현재 패널에서 '생기기 전/없어진 뒤' 인데 0 으로 채워진 대여소-날: {pre:,} / {panel_days:,} ({pre/panel_days*100:.1f}%)")

sec("G. 재고 복원 가능성: 자전거 위치 이어붙이기")
b["moved"] = b.next_st.notna() & (b.next_st!=b.ret_st)
print("월별 '옮겨짐' 건수", b[b.moved].groupby(b.next_ts[b.moved].dt.to_period("M")).size().to_dict())
nb = b[b.moved & (b.ret_st!="ST1220") & (b.next_st!="ST1220")]
print("관제센터 빼고 옮겨짐", len(nb), "| 하루 평균", round(len(nb)/600,1))
print("옮겨짐 거리(km) 분위", pd.Series(hav(*[nb.ret_st.map(t.groupby('ret_st')[k].median()) for k in ['ret_lat','ret_lon']], *[nb.next_st.map(t.groupby('rent_st')[k].median()) for k in ['rent_lat','rent_lon']])).quantile([.1,.25,.5,.75,.9]).round(2).to_dict())
first = t.groupby("bike_id").rent_ts.min(); last = t.groupby("bike_id").ret_ts.max()
print("자전거 첫 등장 월 분포", first.dt.to_period("M").value_counts().sort_index().to_dict())
print("자전거 마지막 등장 월 분포", last.dt.to_period("M").value_counts().sort_index().to_dict())
gap = (b.next_ts-b.ret_ts).dt.total_seconds()/86400
print("반납 후 다음 대여까지 간격(일) 분위", gap.quantile([.5,.9,.99,.999]).round(2).to_dict(), "| 14일 이상 쉬는 경우", int((gap>14).sum()))
