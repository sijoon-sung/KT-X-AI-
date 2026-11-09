"""Render H23 aggregate results on dislab CPU; no model fit or new selection."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NAMES={'full':'Full GCN','no_graph':'No graph (MLP)','shuffled_neighbors':'Shuffled neighbors','no_weather':'No weather','no_calendar':'No explicit calendar','no_recent':'No recent flow','no_embedding':'No profile embedding','no_pretrain':'No Seoul pretraining','prior':'Historical prior','xgboost':'XGBoost','locked_h13':'Frozen H13','locked_h22':'Frozen H22'}
KO={'no_graph':'이웃 전체 / GCN','shuffled_neighbors':'올바른 이웃 대응','no_weather':'기상 입력','no_calendar':'명시적 달력·10분 위치','no_recent':'최근 순흐름','no_embedding':'주간 임베딩','no_pretrain':'서울 사전학습'}

def save(fig,out,name):
 for ext in ['png','svg']:
  target=out/(name+'.'+ext);fig.savefig(target,dpi=200,bbox_inches='tight',facecolor='white')
  if ext=='svg':target.write_text('\n'.join(s.rstrip() for s in target.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
 plt.close(fig)

def main(a):
 r=json.loads(Path(a.result).read_text(encoding='utf-8'));out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
 order=r['test_rank_by_deviance'];y=np.arange(len(order));colors=['#16796c' if k=='full' else '#c58627' if k=='xgboost' else '#65798c' for k in order]
 fig,axes=plt.subplots(1,2,figsize=(13,6.7),sharey=True)
 for ax,key,title in zip(axes,['deviance','mean_count_mae'],['Completed-count Poisson deviance','Mean count MAE']):
  vals=[r['test'][k][key] for k in order];ax.barh(y,vals,color=colors,height=.63);ax.set_yticks(y,[NAMES[k] for k in order]);ax.set_ylim(len(order)-.5,-.5);ax.set_title(title,pad=16);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
  for j,v in enumerate(vals):ax.text(v+max(vals)*.012,j,f'{v:.5f}',va='center',fontsize=9)
  ax.set_xlim(0,max(vals)*1.18)
 fig.suptitle('H23 | Same historical split, fixed features',fontsize=17,y=1.025)
 fig.text(.01,-.025,'March 2026 previously exposed completed trips; not measured inventory. Lower is better. Frozen references used different original training budgets.',fontsize=9,color='#5d6875')
 fig.tight_layout();save(fig,out,'model-comparison')

 keys=list(r['feature_effects']);fig,ax=plt.subplots(figsize=(11,6.2));ax.axvline(0,color='#3c4855',lw=1);ax.axvline(1,color='#9ca7b0',ls=':',lw=1)
 for i,k in enumerate(keys):
  e=r['feature_effects'][k];g=e['relative_reduction']*100
  for b,dy,col in [('3',-.13,'#16796c'),('7',.13,'#7287a0')]:
   lo,hi=[v*100 for v in e[b]['feature_simultaneous_ci']];ax.plot([lo,hi],[i+dy,i+dy],color=col,lw=2);ax.scatter([g],[i+dy],color=col,s=27,zorder=3)
 component_names={'no_graph':'Graph neighborhood','shuffled_neighbors':'Correct neighbor mapping','no_weather':'Weather inputs','no_calendar':'Explicit calendar / phase','no_recent':'Recent net flow','no_embedding':'Weekly profile embedding','no_pretrain':'Seoul pretraining'}
 ax.set_yticks(range(len(keys)),[component_names[k] for k in keys]);ax.invert_yaxis();ax.grid(axis='x',alpha=.18);ax.set_xlabel('Deviance reduction from retaining the component (%)');ax.set_title('Feature contribution after retraining',fontsize=17,pad=20)
 fig.text(.04,.015,'Green: 3-day blocks. Blue: 7-day blocks. Simultaneous intervals across 7 contrasts × 2 block lengths. Positive favors Full.',fontsize=9,color='#5d6875');fig.tight_layout(rect=(0,.045,1,1));save(fig,out,'feature-effects')

 keys=list(r['distributions']);fig,ax=plt.subplots(figsize=(11,5.7));vals=[r['distributions'][k]['energy'] for k in keys]
 labels=[k.replace('locked_h13','Frozen H13').replace('locked_h22','Frozen H22').replace('full','Full GCN').replace('no_graph','No graph').replace('_poisson',' / Poisson').replace('_fm',' / FM') for k in keys]
 ax.barh(range(len(keys)),vals,color=['#8056a3' if k.endswith('_fm') else '#b3883f' for k in keys]);ax.set_yticks(range(len(keys)),labels);ax.invert_yaxis();ax.set_xlim(0,max(vals)*1.22)
 for j,v in enumerate(vals):ax.text(v+max(vals)*.012,j,f'{v:.5f}',va='center')
 ax.set_xlabel('Count energy score (lower is better)');ax.set_title('Poisson and Flow Matching distributions',fontsize=17,pad=20);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
 fig.text(.02,.01,'Same 32,768 station-times, 128 IID particles. Historical completed counts, not stock or dispatch benefit.',fontsize=9,color='#5d6875');fig.tight_layout(rect=(0,.04,1,1));save(fig,out,'distribution-comparison')

 p=r['primary'];fm=r['fm_effect'];best=order[0]
 lines=['# H-0023 결과','', '## 가설','', '고정 특징을 모두 사용한 GCN이 2월에서 선택한 비그래프 대조군보다 3월 거래 deviance를1% 이상 줄이고 두 블록 구간의 하한이 양수인지 확인했다.','', '## 실험','',f"dislab {r['environment']['gpu']}에서 특징8구성×2시드, XGBoost3.1.1, 포아송/FM과 기존 H13/H22를 비교했다. 학습은2월 전, 선택은2월, 시험은 이미 노출된3월이다. 시험은 같은 대여소·시각262,144개이며 분포 평가는32,768개×128표본이다. API는 열지 않았다.",'', '신경망 제거 실험끼리는 같은 예산이다. XGBoost는 별도 고정 예산으로52만개 학습 표본과 최대500tree를 사용했다. 동일 계산량이나 동일 고유 표본 수의 비교는 아니다.','', '## 결과','', f"주가설 판정: **{p['decision']}**. 2월에서 선택한 대조군은 **{NAMES[p['control']]}**이며 full의 상대 deviance 감소는 **{p['relative_reduction']*100:.3f}%**다.", '', '| 모델 | 3월 count deviance ↓ | mean count MAE ↓ | prior 대비 deviance 감소 |','|---|---:|---:|---:|']
 for k in order:
  s=r['test'][k];lines.append(f"| {NAMES[k]} | {s['deviance']:.6f} | {s['mean_count_mae']:.6f} | {s['gain_vs_prior']*100:.3f}% |")
 lines+=['',f"이번 시험에서 deviance가 가장 낮은 모델은 **{NAMES[best]}**다. 이 순위는 한 번의 시간 분할과 고정된 모델별 예산 안의 관측 결과이며 새 운영 모델 선택으로 자동 적용하지 않았다.",'','![모델 비교](model-comparison.png)','','### 특징별 추가 가치','','양수는 해당 요소를 유지한 full이 제거 모델보다 낫다는 뜻이다. 다른 입력을 남긴 상태에서 재학습한 조건부 비교이며 인과적 중요도가 아니다. 명시적 달력 제거 후에도 prior 안의 달력 정보는 남아 있다.','','| 유지한 요소 | deviance 감소 | 3일 동시 구간 | 7일 동시 구간 | 판정 |','|---|---:|---|---|---|']
 for k,e in r['feature_effects'].items():
  ci=lambda b:'['+', '.join(f'{v*100:.3f}%' for v in e[b]['feature_simultaneous_ci'])+']'
  lines.append(f"| {KO[k]} | {e['relative_reduction']*100:.3f}% | {ci('3')} | {ci('7')} | {e['decision']} |")
 lines+=['','![특징별 비교](feature-effects.png)','','### 포아송과 FM','','| 구성 | count energy ↓ | 표본 중앙값 count MAE ↓ |','|---|---:|---:|']
 for k,v in r['distributions'].items():lines.append(f"| {k} | {v['energy']:.6f} | {v['median_count_mae']:.6f} |")
 lines+=['',f"새 full에서 FM의 포아송 대비 energy 감소: **{fm['relative_reduction']*100:.3f}%**, 판정 **{fm['decision']}**.",'','![분포 비교](distribution-comparison.png)','','### 해석 범위와 다음 행동','','- 2·3월 거래 건수는 API 재고 정확도와 다른 지표다. 이번 시험으로 재고·0대 예측의 최종 우승자를 정하지 않는다.','- 기상 입력은 과거 시점 제약을 유지했지만 현재 운영 래퍼의 결측 기상과 다르다. 과거 거래 순흐름도 실시간 API 순재고 변화와 다르다.','- 새 특징을 늘리거나 GCN/FM을 기본 모델로 자동 교체하지 않는다. 유지 근거가 없는 요소는 후속 고정 입력 후보에서 단순화 대상으로 둔다.','- 실제 재고 후보는 과거 선택을 먼저 끝내고 별도 동결 계획으로9월 API 외부 평가를 한다. 같은 API로 구조를 반복 선택하지 않는다.','- 전체 추적값·시드별 점수·일별 집계·그룹 점수는 summary.json에 보존했다.','',f"실행 {r['seconds']:.1f}초. 원 실행 코드 `{r['code_commit']}`. 모델 가중치는 비공개 산출물로 보존하며 배포 가중치를 덮어쓰지 않았다."]
 (out/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 summary={k:r[k] for k in ['hypothesis_id','scope','started_at_utc','finished_at_utc','code_commit','primary','test','test_rank_by_deviance','feature_effects','distributions','fm_effect','feature_audit','test_dates','distribution_test_dates','test_target_zero_fraction','seconds','limitations']}
 (out/'compact-result.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--out',required=True);main(p.parse_args())
