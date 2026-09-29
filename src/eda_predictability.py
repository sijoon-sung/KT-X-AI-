"""90% 정확도가 어디서 가능한지: 데이터 특징 + 저장된 예측(겨울·봄 GCN+GRU)으로 점검.
정확도 정의 두 가지
  R²       : 실제 값의 흔들림 중 몇 %를 설명했나
  acc      : 1 - (|오차| 합 / 실제 대여 합)   = 전체 대여량 대비 맞힌 비율 (수요예측에서 흔히 쓰는 WMAPE 기반)
  ceiling  : 기대값을 완벽히 알아도 남는 우연(푸아송)만 고려한 R² 상한
"""
import numpy as np, pandas as pd
from pathlib import Path
pd.set_option("display.width", 250)
ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT/"data/processed/v2"
z = np.load(V2/"panel.npz"); rent = z["rent"]; obs = z["node_active"] & z["hour_ok"][:, None]
nodes = pd.read_parquet(V2/"nodes.parquet"); T, N = rent.shape
ts = pd.date_range("2024-08-01", periods=T, freq="h")
def r2(y,p): y=np.asarray(y,float); p=np.asarray(p,float); return 1-((y-p)**2).sum()/((y-y.mean())**2).sum()
def acc(y,p): y=np.asarray(y,float); p=np.asarray(p,float); return 1-np.abs(y-p).sum()/y.sum()
def ceil_(y): y=np.asarray(y,float); return 1-y.mean()/y.var()
def sec(s): print(f"\n########## {s}")

sec("1. 수요가 얼마나 흩어져 있나 (유효 칸 전체)")
yv = rent[obs]
print(f"유효 칸 {obs.sum():,} | 0건 비율 {np.mean(yv==0):.3f} | 1건 {np.mean(yv==1):.3f} | 2건 {np.mean(yv==2):.3f} | 3건 이상 {np.mean(yv>=3):.3f}")
node_mean = (rent*obs).sum(0)/np.maximum(obs.sum(0),1)
node_tot = (rent*obs).sum(0)
bins=[0,0.1,0.3,0.5,1,2,1e9]; lab=["<0.1","0.1~0.3","0.3~0.5","0.5~1","1~2","2이상"]
g = pd.cut(node_mean,bins,labels=lab,right=False)
t1 = pd.DataFrame({"grp":g,"tot":node_tot}).groupby("grp",observed=False).agg(대여소수=("tot","size"),대여합=("tot","sum"))
t1["대여소비율"]=(t1.대여소수/N).round(3); t1["대여비율"]=(t1.대여합/node_tot.sum()).round(3)
print("시간당 평균 대여로 나눈 대여소 그룹"); print(t1.to_string())
o = np.sort(node_tot)[::-1]; cs=np.cumsum(o)/o.sum()
print("대여의 50%를 만드는 대여소 수:", int(np.searchsorted(cs,0.5))+1, "| 80%:", int(np.searchsorted(cs,0.8))+1, "/", N)

sec("2. 같은 시간 지난주와 얼마나 닮았나 (주간 반복성, 상관계수)")
def weekcorr(A, M):
    a, b, m = A[168:], A[:-168], M[168:] & M[:-168]
    return np.corrcoef(a[m], b[m])[0,1]
print("대여소×1시간:", round(weekcorr(rent, obs),3))
city = (rent*obs).sum(1); cm = obs.any(1)
print("대전 전체×1시간:", round(np.corrcoef(city[168:][cm[168:]&cm[:-168]], city[:-168][cm[168:]&cm[:-168]])[0,1],3))

sec("3. 저장된 예측으로 본 정확도 (겨울+봄, GCN+GRU)")
parts=[]
for f in ["winter","spring"]:
    d=np.load(ROOT/f"outputs/st/{f}_gcn_gru_pred.npz"); parts.append(pd.DataFrame({k:d[k] for k in ["h","n","y","p"]}).assign(fold=f))
df=pd.concat(parts,ignore_index=True)
df["hour"]=ts[df.h].hour; df["grp"]=np.asarray(g)[df.n]
lat=nodes.lat.to_numpy(); lon=nodes.lon.to_numpy()
def grid(m): return (np.floor(lat/(0.009*m/1000)).astype(int)*100000+np.floor(lon/(0.0112*m/1000)).astype(int))
rows=[]
def add(name, keys, note=""):
    q=df.groupby(keys)[["y","p"]].sum()
    rows.append({"단위":name,"평균 실제대여":round(q.y.mean(),2),"R2":round(r2(q.y,q.p),3),"R2 한계":round(ceil_(q.y),3),"acc":round(acc(q.y,q.p),3),"비고":note})
add("대여소×1시간",[df.n,df.h])
add("대여소×3시간",[df.n,df.h//3],"1시간 예측 합")
add("대여소×하루",[df.n,df.h//24],"1시간 예측 합")
for m in [300,500,1000,2000]:
    add(f"{m}m 격자×1시간",[grid(m)[df.n],df.h])
add("1km 격자×3시간",[grid(1000)[df.n],df.h//3],"1시간 예측 합")
add("대전 전체×1시간",[df.h])
print(pd.DataFrame(rows).to_string(index=False))

sec("4. 대여소 붐빔 정도별 (대여소×1시간)")
r4=[]
for k,q in df.groupby("grp",observed=True):
    r4.append({"시간당 평균":k,"칸 비율":round(len(q)/len(df),3),"대여 비율":round(q.y.sum()/df.y.sum(),3),"R2":round(r2(q.y,q.p),3),"R2 한계":round(ceil_(q.y),3),"acc":round(acc(q.y,q.p),3)})
print(pd.DataFrame(r4).to_string(index=False))

sec("5. 시간대별 (대여소×1시간 / 1km 격자×1시간)")
r5=[]
for hh,q in df.groupby("hour"):
    gq=q.groupby([grid(1000)[q.n],q.h])[["y","p"]].sum()
    r5.append({"시":hh,"평균대여":round(q.y.mean(),2),"대여소 R2":round(r2(q.y,q.p),3),"대여소 acc":round(acc(q.y,q.p),3),"격자 R2":round(r2(gq.y,gq.p),3),"격자 acc":round(acc(gq.y,gq.p),3)})
print(pd.DataFrame(r5).to_string(index=False))

sec("6. '정확도'를 분류로 재면 생기는 착시")
yb, pb = df.y>=1, df.p>=0.5
print(f"다음 1시간 대여 1건 이상인가: 정확도 {np.mean(yb==pb):.3f} | 전부 '없음'이라 찍어도 {np.mean(~yb):.3f} | 실제 발생 중 맞힘 {np.mean(pb[yb]):.3f}")
yb, pb = df.y>=3, df.p>=3
print(f"다음 1시간 대여 3건 이상인가: 정확도 {np.mean(yb==pb):.3f} | 전부 '아님'이라 찍어도 {np.mean(~yb):.3f} | 실제 발생 중 맞힘 {np.mean(pb[yb]):.3f}")

sec("7. 날씨: 대전 전체 하루 오차가 비·기온과 관련 있나")
w=pd.read_parquet(ROOT/"data/raw/weather_openmeteo_daejeon.parquet"); w["h"]=np.arange(len(w))
day=df.groupby(df.h//24)[["y","p"]].sum(); day["err%"]=(day.p-day.y)/day.y*100
wd=w.groupby(w.h//24).agg(rain=("precipitation","sum"),temp=("temperature_2m","mean"),wind=("wind_speed_10m","mean"))
day=day.join(wd); day["date"]=ts[day.index*24].date
print("하루 오차(%)와 상관: 강수", round(day["err%"].corr(day.rain),3), "| 기온", round(day["err%"].corr(day.temp),3), "| 바람", round(day["err%"].corr(day.wind),3))
print("비 5mm 이상인 날 평균 과대예측(%):", round(day[day.rain>=5]["err%"].mean(),1), f"({(day.rain>=5).sum()}일)", "| 비 없는 날:", round(day[day.rain==0]["err%"].mean(),1))
print("가장 크게 틀린 날 10개"); print(day.reindex(day["err%"].abs().sort_values(ascending=False).index).head(10)[["date","y","p","err%","rain","temp"]].round(1).to_string(index=False))
hr=df.merge(w[["h","precipitation"]],on="h"); ch=hr.groupby("h")[["y","p","precipitation"]].agg({"y":"sum","p":"sum","precipitation":"first"})
print("대전 전체×1시간 R2: 전체", round(r2(ch.y,ch.p),3), "| 비 안 온 시간만", round(r2(ch.y[ch.precipitation==0],ch.p[ch.precipitation==0]),3))
