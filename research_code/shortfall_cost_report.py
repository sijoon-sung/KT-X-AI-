"""Render H25 aggregate evidence on dislab after numerical evaluation ends."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shortfall_cost_evaluate import MODELS,TARGETS,COSTS

NAMES={'persistence':'Current stock','h13_poisson':'H13 GCN / Poisson','h13_fm':'H13 GCN / Gaussian FM',
       'h22_poisson':'H22 GCN / Poisson','h22_fm':'H22 GCN / Gaussian FM',
       'h24_poisson':'H24 MLP / Poisson','h24_fm':'H24 MLP / Poisson-start FM'}
COLORS=['#a3acb9','#3969ac','#83a6cd','#6f4e7c','#bd8ec6','#d18b25','#087f8c']

def render(a):
    src=Path(a.source);out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
    result=json.loads((src/'results.json').read_text(encoding='utf-8'));m=result['metrics']
    def row(k,target=3,cu=5,group='current_positive'):
        return m[f'target{target}/cost{cu}/{group}/{k}']
    # For integer inventory, target1 gap is exactly the zero-inventory indicator.
    # Standard Brier = (M-1)/M * fairBrier + mean|p-z|/M, M=128.
    # This recovers the prespecified secondary score from aggregate sufficient statistics.
    toy=np.array([[0,0,1,1],[1,1,1,1]],float);ty=np.array([1,0.])
    from rolling_inventory_metrics import scores
    ts=scores(toy,ty)
    assert np.allclose(.75*ts['fair_crps']+np.abs(toy.mean(1)-ty)/4,ts['brier'],atol=1e-12)
    zero={}
    for group in ('all','current_zero','current_positive','current1_2','current3plus'):
        zero[group]={}
        for k in MODELS:
            r=row(k,1,1,group)
            zero[group][k]={'n':r['n'],'brier':127/128*r['gap_fair_crps']+r['gap_mean_abs']/128,
                            'fair_brier':r['gap_fair_crps']}
    (out/'derived-zero-brier.json').write_text(json.dumps({'identity_hand_check':True,'method':'target1 gap equals zero-inventory indicator; standard Brier recovered from fairCRPS and mean absolute probability error','groups':zero},indent=2)+'\n',encoding='utf-8')
    notes={'unrecorded_secondary_metrics':['Inventory median MAE for this hourly-origin cohort'],
           'reason':'Initial evaluator recorded shortfall mean/median MAE but omitted uncapped inventory median MAE. It cannot be recovered from the capped gap aggregates. This omission is preserved, not silently replaced with the old full-cohort MAE.',
           'primary_metrics_affected':False,'frozen_plan_changed':False,'new_model_selection':False}
    (out/'execution-notes.json').write_text(json.dumps(notes,indent=2)+'\n',encoding='utf-8')
    secondary={}
    for target in TARGETS:
        for cu in COSTS:
            for family in ('h13','h22','h24'):
                p=row(family+'_poisson',target,cu);f=row(family+'_fm',target,cu)
                secondary[f'target{target}/cost{cu}/{family}']={
                    'relative_cost_reduction':1-f['cost']/p['cost'],
                    'relative_under_bikes_reduction':1-f['under_bikes']/p['under_bikes'],
                    'poisson_under_per1000':p['under_bikes']*1000,'fm_under_per1000':f['under_bikes']*1000,
                    'poisson_over_per1000':p['over_bikes']*1000,'fm_over_per1000':f['over_bikes']*1000}
    (out/'secondary-comparisons.json').write_text(json.dumps({'scope':'All prespecified target/cost conditions, current-positive group. Descriptive point comparisons only; not new confirmatory tests or API winner selection.','comparisons':secondary},indent=2)+'\n',encoding='utf-8')
    def save(fig,name):
        fig.savefig(out/f'{name}.png',dpi=170,bbox_inches='tight',facecolor='white')
        fig.savefig(out/f'{name}.svg',bbox_inches='tight',facecolor='white');plt.close(fig)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    names=[NAMES[k] for k in MODELS];y=np.arange(len(MODELS))
    fig,axes=plt.subplots(1,3,figsize=(15.7,5.1),gridspec_kw={'wspace':.44})
    for ax,key,title in zip(axes,('cost','under_bikes','over_bikes'),('Asymmetric forecast cost','Underestimated gap','Overestimated gap')):
        ax.barh(y,[row(k)[key] for k in MODELS],color=COLORS,height=.65)
        ax.set_yticks(y,names if ax is axes[0] else ['']*len(names));ax.invert_yaxis();ax.set_title(title,loc='left',fontweight='bold')
        ax.set_xlabel('Assumed cost units / forecast' if key=='cost' else 'Bikes / forecast');ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
    fig.suptitle('Inventory shortfall: magnitude and asymmetric error cost',x=.02,ha='left',fontsize=15,fontweight='bold')
    fig.text(.02,.005,'Current stock > 0 | assumed target = 3 bikes | under:over cost = 5:1 | hourly September origins, +10 min.\nForecast errors only; not measured lost rentals, dispatched bikes or realized operating cost.',fontsize=9,color='#536273')
    fig.subplots_adjust(bottom=.19,top=.85);save(fig,'shortfall-main')

    fig,axes=plt.subplots(1,3,figsize=(14,4.9),sharey=False)
    for ax,target in zip(axes,TARGETS):
        for k,color in zip(MODELS,COLORS):
            ax.plot(COSTS,[row(k,target,c)['cost'] for c in COSTS],marker='o',lw=1.5,color=color,label=NAMES[k])
        ax.set_title(f'Assumed target = {target} bikes',loc='left',fontweight='bold');ax.set_xticks(COSTS);ax.set_xlabel('Underprediction cost (overprediction = 1)');ax.grid(alpha=.15)
    axes[0].set_ylabel('Assumed forecast cost / valid origin')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.01))
    fig.suptitle('Sensitivity to target inventory and asymmetric cost',x=.02,ha='left',fontsize=14,fontweight='bold')
    fig.subplots_adjust(bottom=.31,top=.83,wspace=.28);save(fig,'shortfall-sensitivity')

    lines=['# H-0025: 재고 부족량과 비대칭 비용','','## 가설','',
           '동결한 H24 FM이 같은 encoder의 포아송보다 현재 양수 대여소의 목표 3대 부족량에 대한 5:1 비대칭 비용을 1% 이상 줄이는지 검증했다. 기존 H24 판정과 가중치는 바꾸지 않았다.','',
           '## 실험','',
           'dislab에서 고정 모델 7종을 공식 9월 API로 재생했다. KST 정각마다 현재·직전 관측으로 10분 뒤 표시 재고를 예측했다. 누락을 0대로 채우지 않았고 미래 유효성은 발행 뒤 채점에만 사용했다. API로 학습·모형 선택·보정하지 않았다.','',
           f"유효 {result['coverage']['valid_pairs']:,}쌍, 현재 양수 {result['coverage']['positive_pairs']:,}쌍, {result['coverage']['scored_ids']:,}개 대여소, {result['coverage']['scored_days']}일. 수치 실험 {result['seconds']:.1f}초. 전체 H13 평가와 다른 정각 원점 표본이며 9월은 이미 노출된 자료다.",'',
           '부족량은 `Z=max(0,목표-미래표시재고)`다. 목표 1·3·6대와 부족/과잉 비용 1·2·5·10:1은 운영 미확인 가정이다. 비용 비율에 맞춘 부족량 분위수를 Q로 예측하고 `cu*max(0,Z-Q)+max(0,Q-Z)`를 채점했다. 실제 배차량이나 배차 후 효과가 아니다.','',
           '## 결과','',f"주가설 판정: **{result['primary']['decision']}**. H24 FM의 같은 encoder 포아송 대비 비용 상대 감소는 **{100*result['primary']['relative_reduction']:.3f}%**다."]
    for v in result['primary']['intervals']:
        lo,hi=v['relative_reduction_95'];lines.append(f"- {v['block_days']}일 블록 95% 구간: [{lo*100:.3f}%, {hi*100:.3f}%]")
    lines+=['',f"부족량 과소예측량 guard: {result['primary']['underprediction_guard']}. 자료 충분 조건: {result['primary']['sufficient_data']}.",'',
            '### 주조건: 현재 양수·목표 3대·부족 비용 5배','',
            '| 모델 | 가정 비용 ↓ | 부족량 과소예측, 대 ↓ | 부족량 과대예측, 대 ↓ | 부족량 분포 fairCRPS ↓ |',
            '|---|---:|---:|---:|---:|']
    for k in MODELS:
        r=row(k);lines.append(f"| {NAMES[k]} | {r['cost']:.6f} | {r['under_bikes']:.6f} | {r['over_bikes']:.6f} | {r['gap_fair_crps']:.6f} |")
    lines+=['','![부족량 주조건](shortfall-main.png)','','### 비용 기준에 따른 차이','','| 모델 | 1:1 | 2:1 | 5:1 | 10:1 |','|---|---:|---:|---:|---:|']
    for k in MODELS:lines.append('| '+NAMES[k]+' | '+' | '.join(f"{row(k,3,c)['cost']:.6f}" for c in COSTS)+' |')
    lines+=['','### 같은 encoder에서 FM 추가 전후: 기술적 점추정','',
            '아래는 사전 비교군의 기술적 비교다. H13/H22를 새 확인적 주가설로 바꾸거나 API로 승자를 선택하지 않는다. 양수는 FM 비용 감소, 음수는 증가다.','',
            '| 구조 | 비용 1:1에서 FM 비용 감소율 | 비용 5:1에서 FM 비용 감소율 |',
            '|---|---:|---:|']
    for family in ('h13','h22','h24'):
        lines.append(f"| {family.upper()} | {100*secondary[f'target3/cost1/{family}']['relative_cost_reduction']:.3f}% | {100*secondary[f'target3/cost5/{family}']['relative_cost_reduction']:.3f}% |")
    lines+=['','![목표·비용별 결과](shortfall-sensitivity.png)','','### 현재 재고별 결과','',
            '| 현재 상태 | 표본 수 | 유지 비용 | H13 Poisson | H13 FM | H22 Poisson | H22 FM | H24 Poisson | H24 FM |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for group in ('all','current_zero','current_positive','current1_2','current3plus'):
        lines.append('| '+group+f" | {row('persistence',group=group)['n']:,} | "+' | '.join(f"{row(k,group=group)['cost']:.6f}" for k in MODELS)+' |')
    lines+=['','### 해석 범위와 다음 행동','',
            '보조 0대 Brier는 목표 1대 부족량이 재고 0대 지시값과 같다는 관계로 집계값에서 정확히 복원했다. 수작업 예제 검사를 통과했다. 같은 정각 표본의 재고 중앙값 MAE는 초기 집계에서 누락되어 미집계로 남겼다. 기존 전체 표본 MAE를 대신 붙이지 않았다. 부족량 MAE와 주지표는 집계했다.','',
            '목표 대비 몇 대가 부족한지를 채점했다. 실제 대여 시도·포기 기록이 없으므로 미충족 수요 인원, 품절 때문에 사라진 거래량, 운영자가 옮긴 뒤의 개선을 측정한 결과로 읽지 않는다. 부족량 분위수는 비용 가정에 맞춘 예측 출력이며 실제 배차 권고량이 아니다.','',
            'API 점수로 승자·분포·목표·비용 비율을 다시 고르지 않았다. 순위는 이번 고정 조건의 기술적 비교이며 기본 H13·관리자는 유지한다. 실제 운영 의사결정에는 목표 재고의 근거와 운영 단가, 동일 기간의 거래·재고·개입 로그가 필요하다.','',
            '다음 행동은 모델 용량이나 특징을 늘리기보다 재고 목표·부족 비용의 운영 근거와 미충족 수요 관측 가능성을 확인하는 것이다. 새 학습이 필요하면 과거 이력 안에서 별도 가설로 손실을 고정하고, API는 외부 검증으로 유지한다.']
    (out/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (out/'all-conditions.csv').open('w',encoding='utf-8',newline='') as f:
        keys=next(iter(m.values())).keys();writer=csv.DictWriter(f,fieldnames=['condition',*keys]);writer.writeheader()
        for name,r in m.items():writer.writerow({'condition':name,**r})
    card={key:result[key] for key in ('hypothesis_id','code_commit','plan_sha256','input_manifest_sha256','coverage','primary','seconds','weights_updated','api_selected','limitations')}
    card['main_condition']={k:row(k) for k in MODELS}
    (out/'compact-result.json').write_text(json.dumps(card,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'render_complete':True,'files':[p.name for p in out.iterdir()]}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);render(p.parse_args())
