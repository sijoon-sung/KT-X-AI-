"""H24 aggregate plots, report and actual selected-model diagram. Run on dislab CPU."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch
import numpy as np

ENC={'full_retrain':'Full H13 retrained','compact_local':'Local MLP128','compact_graph':'Light GCN128','compact_profile':'Profile MLP128','frozen_h13':'Frozen H13'}
DEC={'poisson':'Poisson','gaussian_mse':'Gaussian FM / loss selection','gaussian_score':'Gaussian FM / count-score selection','poisson_continuous_score':'Poisson → continuous FM','count_jump_score':'Poisson → integer count-FM','frozen_h13_poisson':'Frozen H13 / Poisson','frozen_h13_gaussian':'Frozen H13 / Gaussian FM'}

def save(fig,out,name):
    for ext in ('png','svg'):
        p=out/(name+'.'+ext);fig.savefig(p,dpi=200,bbox_inches='tight',facecolor='white')
        if ext=='svg':p.write_text('\n'.join(s.rstrip() for s in p.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)

def architecture(r,out):
    kind=r['selected_encoder'];mode=r['selected_fm'];choice=r['flow_choices'][mode];active=sum(choice['active']);small=kind!='full_retrain';width=128 if small else 256
    own=164 if kind in ('full_retrain','compact_profile') else 100;graph=kind in ('full_retrain','compact_graph');profile=kind in ('full_retrain','compact_profile')
    fig=plt.figure(figsize=(15.6,7.2));ax=fig.add_axes((0,0,1,1));ax.set_xlim(0,1560);ax.set_ylim(720,0);ax.axis('off')
    colors={'blue':('#eaf2fa','#326b9b'),'gold':('#fcf2dd','#a77823'),'green':('#e4f3ed','#247f68'),'purple':('#f1eaf8','#8053a5'),'gray':('#f3f4f6','#73818d')}
    def txt(x,y,s,size=12,**kw):ax.text(x,y,s,fontsize=size,ha='center',va='center',**kw)
    def box(x,y,w,h,title,sub='',color='blue'):
        bg,edge=colors[color];ax.add_patch(Rectangle((x,y),w,h,fc=bg,ec=edge,lw=1.3,zorder=3));txt(x+w/2,y+h*(.35 if sub else .5),title,13,weight='bold',zorder=4)
        if sub:txt(x+w/2,y+h*.72,sub,10,color='#52606d',zorder=4)
    def arrow(points,color='#425466',dashed=False):
        if len(points)>2:ax.plot([p[0] for p in points[:-1]],[p[1] for p in points[:-1]],c=color,lw=1.3,ls='--' if dashed else '-',zorder=1)
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=12,color=color,lw=1.3,linestyle='--' if dashed else '-',zorder=1))
    txt(780,28,'Weather-preserving compact encoder and count-aligned Flow Matching',21,weight='bold')
    txt(780,65,f"Selected from February only | {ENC[kind]} | {DEC[mode]} | active FM seeds: {active}/2",11,color='#52606d')
    for x,w,label,col in [(15,460,'(a) Encoder','#326b9b'),(500,625,'(b) Poisson source and correction','#8053a5'),(1150,395,'(c) Count prediction','#247f68')]:
        ax.add_patch(Rectangle((x,100),w,520,fill=False,ec=col,lw=1,ls='--'));ax.text(x+12,122,label,fontsize=13,weight='bold',color=col,ha='left')
    box(40,161,410,110,'Existing inputs',f'priors6 / calendar35 / phase3 / weather54\nrecent2'+(' / weekly embedding64' if profile else ''),'blue')
    if graph:
        box(40,300,160,75,'Own projection',f'{own} → {width}','blue');box(233,300,217,75,'Neighbor projection',f"{'72' if kind=='full_retrain' else '8'} → {width}; fixed OD",'blue')
        arrow([(245,273),(245,286),(120,286),(120,298)]);arrow([(245,286),(340,286),(340,298)])
        arrow([(120,377),(120,396),(225,396)]);arrow([(340,377),(340,396),(265,396)]);txt(245,395,'+',16)
        arrow([(245,406),(245,412)])
    else:
        box(90,300,310,75,'Station projection',f'Linear {own} → {width} + SiLU','blue');arrow([(245,273),(245,298)]);arrow([(245,377),(245,412)])
    box(90,414,310,90,'Context MLP',f'[Linear {width} → {width} + SiLU] × 2','blue')
    txt(245,555,rf'$h_i\in\mathbb{{R}}^{{{width}}}$',17,color='#326b9b');arrow([(245,506),(245,535)])
    box(540,162,220,88,'Rate head',f'{width} → 2 / Poisson likelihood','gold')
    arrow([(327,555),(487,555),(487,205),(538,205)],'#326b9b')
    txt(825,206,r'$\lambda_i$',22,color='#a77823');arrow([(762,205),(795,205)],'#a77823')
    box(889,162,207,88,'Poisson source',r'$x_0\sim\mathrm{Pois}(\lambda_i)$','gold');arrow([(853,205),(887,205)],'#a77823')
    if active:
        title='Integer count-FM' if mode=='count_jump_score' else 'Continuous residual FM'
        sub=f'Birth / death rates: {width+8} → 128 → 128 → 128 → 4\n64 integer steps; zero death at0' if mode=='count_jump_score' else f'Poisson dequantized source\nv: {width+8} → 128 → 128 → 128 → 2\n16 Heun steps / integer decode'
        box(622,358,474,133,title,sub,'purple')
        arrow([(993,252),(993,356)],'#a77823');arrow([(487,555),(572,555),(572,425),(620,425)],'#326b9b',True)
        arrow([(825,230),(825,318),(900,318),(900,356)],'#a77823',True)
        txt(845,548,'Generative time s ∈ [0, 0.999]' if mode=='count_jump_score' else 'Generative time s ∈ [0,1]',11,color='#8053a5')
        arrow([(1098,425),(1188,425)],'#8053a5')
        if active<2:
            arrow([(1098,205),(1138,205),(1138,425),(1188,425)],'#73818d',True)
            txt(1158,278,'Identity-seed bypass',9,rotation=90,color='#73818d')
    else:
        box(622,358,474,133,'Direct Poisson selected','FM checkpoints were trained,\nbut no learned correction was selected','gray')
        arrow([(993,252),(993,356)],'#a77823');arrow([(1098,425),(1188,425)],'#73818d')
    box(1190,360,315,128,'Rental / return samples','128 IID particles\nnonnegative integer counts','green')
    txt(1347,552,r'$Y_{i,t:t+10}=(Y_R,Y_D)$',17,color='#247f68')
    txt(780,660,f"Rate-network parameters per seed: {r['mean_scores'][kind]['parameters_per_seed']:,}. Reduction: {r['encoder_parameter_reduction']*100:.1f}%.",12)
    txt(780,700,'Generative steps are not physical bike movements. This experiment evaluates completed counts, not inventory or dispatch effects.',10,color='#52606d')
    save(fig,out,'selected-architecture')

def main(a):
    r=json.loads(Path(a.result).read_text(encoding='utf-8'));out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    keys=list(r['mean_scores']);fig,axes=plt.subplots(1,3,figsize=(14.8,4.8))
    for ax,field,title in zip(axes,['parameters_per_seed','deviance','mean_count_mae'],['Rate parameters / seed','March count deviance','March mean count MAE']):
        vals=[r['mean_scores'][k][field] for k in keys];colors=['#247f68' if k==r['selected_encoder'] else '#728497' for k in keys];ax.barh(range(len(keys)),vals,color=colors);ax.set_yticks(range(len(keys)),[ENC[k] for k in keys]);ax.invert_yaxis();ax.set_title(title,pad=18);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        for j,v in enumerate(vals):ax.text(v+max(vals)*.012,j,f'{int(v):,}' if field=='parameters_per_seed' else f'{v:.5f}',va='center',fontsize=9)
        ax.set_xlim(0,max(vals)*1.2)
    fig.suptitle('H24 | Encoder simplification with existing weather',fontsize=17,y=1.04);fig.tight_layout();save(fig,out,'encoder-comparison')
    keys=list(r['distributions']);fig,axes=plt.subplots(1,2,figsize=(14,6.1))
    for ax,field,title in zip(axes,['energy','zero_count_brier'],['Count energy (lower is better)','Zero-count Brier (not stock-out)']):
        vals=[r['distributions'][k][field] for k in keys];ax.barh(range(len(keys)),vals,color=['#247f68' if k==r['selected_fm'] else '#b58b43' if k=='poisson' else '#8c6aa7' for k in keys]);ax.set_yticks(range(len(keys)),[DEC[k] for k in keys]);ax.invert_yaxis();ax.set_title(title,pad=18);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        for j,v in enumerate(vals):ax.text(v+max(vals)*.012,j,f'{v:.5f}',va='center',fontsize=9)
        ax.set_xlim(0,max(vals)*1.25)
    fig.suptitle('H24 | Distribution and selection-path comparison',fontsize=17,y=1.03);fig.tight_layout();save(fig,out,'distribution-comparison')
    architecture(r,out)
    p=r['primary'];choice=r['flow_choices'][r['selected_fm']]
    flow_kind,selector=r['selected_fm'].rsplit('_',1)
    ids=[f'encoder-{r["selected_encoder"]}-{s}.pt' for s in (11,29)]+[f'flow-{flow_kind}-{selector}-{s}.pt' for s in (11,29)]
    weight_bytes=sum(r['checkpoint_bytes'][k] for k in ids)
    flow_parameters=[t['selected'][selector]['parameters'] for t in r['flow_training'] if t['kind']==flow_kind]
    model_card={'hypothesis_id':'H-0024','encoder_kind':r['selected_encoder'],'flow_kind':flow_kind,'selection':selector,'seeds':[11,29],'flow_epochs':choice['epochs'],'flow_active':choice['active'],'weight_files':ids,'selected_checkpoint_bytes':weight_bytes,'total_selected_parameters':2*r['mean_scores'][r['selected_encoder']]['parameters_per_seed']+sum(flow_parameters),'encoder_parameter_reduction':r['encoder_parameter_reduction'],'weather_dimensions':54,'station_context_width':128 if r['selected_encoder']!='full_retrain' else 256,'particles':128,'source':'conditional Poisson','solver':'Heun16' if flow_kind=='poisson_continuous' else 'integer local jump64 to.999','epoch0_semantics':'direct Poisson bypass,not learned FM','weights_publication':'private checkpoints identified by pre-test-lock.json,not rawGit','static_historical_context_included_in_weight_size':False,'serving_integration':'Research candidate; existing runtime not replaced and API inference not executed','numerical_training_commit':r['code_commit']}
    (out/'model-card.json').write_text(json.dumps(model_card,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# H-0024 결과','','## 가설','','기상은 유지하고 encoder를 줄인 뒤, 건수 정합 FM이 포아송보다 분포 오차를1% 이상 줄이는지 검증했다. 무보정 선택은 FM 이득으로 세지 않았다.','','## 실험','',f"dislab {r['environment']['gpu']}에서 encoder4종×2시드, FM3학습 경로×2시드를 비교했다. Encoder는2월전반, decoder는2월후반에서 선택하고 잠근 뒤 이미 노출된3월을 평가했다. API를 열지 않았다. 총{r['seconds']:.1f}초다.",'','최초 시도는 호출 인자 누락으로 첫 학습 checkpoint 전에 실패했다. 조건·예산·계획을 바꾸지 않고 인자만 수정했으며 원 시도 파일과기록을 보존했다.','','## 결과','',f"선택 encoder: **{ENC[r['selected_encoder']]}**. Rate parameter가 **{r['encoder_parameter_reduction']*100:.2f}%** 줄었다. 선택 decoder: **{DEC[r['selected_fm']]}**, epoch **{choice['epochs']}**, 활성 FM **{sum(choice['active'])}/2**시드. 주가설 판정 **{p['decision']}**.",'','| Encoder | seed당 rate parameters | March count deviance ↓ | mean count MAE ↓ |','|---|---:|---:|---:|']
    for k,s in r['mean_scores'].items():lines.append(f"| {ENC[k]} | {s['parameters_per_seed']:,} | {s['deviance']:.6f} | {s['mean_count_mae']:.6f} |")
    lines+=['',f"선택한 두 시드 encoder+FM 가중치는 **{weight_bytes:,}바이트**, 총 **{model_card['total_selected_parameters']:,}개 파라미터**다. 과거 특성 cache는 이 크기에 포함하지 않았다. [모델 카드](model-card.json).",'','![encoder 비교](encoder-comparison.png)','','| Decoder | count energy ↓ | median count MAE ↓ | count fairCRPS ↓ | zero-count Brier ↓ | negative raw mass |','|---|---:|---:|---:|---:|---:|']
    for k,s in r['distributions'].items():lines.append(f"| {DEC[k]} | {s['energy']:.6f} | {s['median_count_mae']:.6f} | {s['fair_channel_crps']:.6f} | {s['zero_count_brier']:.6f} | {s['negative_continuous_raw_fraction']*100:.3f}% |")
    lines+=['',f"선택 FM의 같은 encoder 포아송 대비 energy 감소: **{p['relative_reduction']*100:.3f}%**."]
    for block in ('3','7'):lines.append(f"{block}일 블록95% 구간: [{p[block][0]*100:.3f}%,{p[block][1]*100:.3f}%].")
    lines+=['','![분포 비교](distribution-comparison.png)','','### 원인 분리','','| 비교 | energy 감소 |','|---|---:|']
    for name,s in r['comparisons'].items():lines.append(f"| {name} | {s['relative_reduction']*100:.3f}% |")
    lines+=['','### 수치 적분 검사','','| 경로 | 같은8192표본 기본단계 energy | 추가단계 energy |','|---|---:|---:|']
    for name,s in r['solver_diagnostics'].items():lines.append(f"| {name} / {s['extra_steps']}steps | {s['primary_steps_same_prefix']['energy']:.6f} | {s['extra']['energy']:.6f} |")
    lines+=['','추가 단계 점수는 진단이며 3월 점수로 실제 sampler를 선택하지 않았다. Zero-count는 대여/반납0건이며 대여소0대와 다르다.','','### 선택한 구조','','![선택 구조](selected-architecture.png)','','### 다음 행동','','- 특징·모델 크기 확대는 하지 않는다. 배포H13과 관리자 경로는 유지한다.','- 작은 encoder는 검증 허용 오차와 parameter 기준으로 선택했으며 동일 성능을 증명한 것은 아니다.','- 무보정이 선택됐다면 활성FM 개선으로 발표하지 않는다. Gaussian 대비 개선과 Poisson 대비 개선을 구분한다.','- 모든 선택과 학습은 과거 기록만으로 마쳤다. 실제 재고 후보는 별도 동결 계획으로9월 외부 평가한다. API로 재학습·선택·보정하지 않는다.','- Count-FM은[원문](https://arxiv.org/html/2605.07746v3)의 정수 bridge와 rate matching을 조건부Poisson에 적용한 것이다. OT·CFG·원문 전체 재현을 주장하지 않는다.','',f"실행 코드 `{r['code_commit']}`. 시드·trace·실패·잠금·진단·전체 지표는 summary와manifest에 보존했다."]
    (out/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    compact={k:r[k] for k in ('hypothesis_id','code_commit','primary','selected_encoder','encoder_parameter_reduction','selected_fm','flow_choices','mean_scores','distributions','comparisons','solver_diagnostics','seconds','kernel_checks','limitations')};compact['model_card']=model_card
    (out/'compact-result.json').write_text(json.dumps(compact,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(compact,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--out',required=True);main(p.parse_args())
