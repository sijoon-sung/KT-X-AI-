export default async function({presentation,input}){
 const s=presentation.slides.getItem(7),C={navy:'#173E50',teal:'#1A8585',line:'#B7C9D0',blue:'#F0F6FB',green:'#EFF7F4',orange:'#FFF4E5',muted:'#526A75',white:'#FFFFFF'};
 const root=input.assetDirectory;
 if(input.phase==='icons-data'||input.phase==='icons-people'){
  const all=input.phase==='icons-data'?[['history',49,208,66,66],['inventory',49,288,66,66],['weather',49,368,66,66]]:[['operator',992,206,234,193],['truck',1012,440,202,94]];
  return {pictures:all.map(([name,x,y,w,h])=>s.images.add({path:root+name+'.png',alt:name+' service illustration',fit:'contain',position:{left:x,top:y,width:w,height:h}}).id)};
 }
 presentation.delete(input.remove);
 const ids=[];
 const shape=(name,x,y,w,h,fill=C.white,line=C.line,geo='roundRect')=>{const a=s.shapes.add({name:'H27P '+name,geometry:geo,position:{left:x,top:y,width:w,height:h},fill,line:{fill:line,width:1.1},...(['rect','textbox','roundRect'].includes(geo)?{borderRadius:8}:{})});ids.push(a.id);return a;};
 const text=(name,value,x,y,w,h,size=20,color=C.navy,bold=false,align='left')=>{const a=shape(name,x,y,w,h,'none','none','textbox');a.text=value;a.text.style={fontSize:size,color,bold,alignment:align,verticalAlignment:'middle',insets:0,wrap:true};return a;};
 const box=(name,value,x,y,w,h,fill,size=20,color=C.navy,bold=false)=>{const a=shape(name,x,y,w,h,fill);a.text=value;a.text.style={fontSize:size,color,bold,alignment:'center',verticalAlignment:'middle',insets:8,wrap:true};return a;};
 const connect=(a,b,from='right',to='left',color=C.teal,kind='straight',arrow=true,width=1.8)=>{const c=s.shapes.connect(a,b,{kind,fromSide:from,toSide:to,line:{fill:color,width},...(arrow?{tail:{type:'triangle',width:'sm',length:'sm'}}:{})});c.bringToFront();ids.push(c.id);return c;};
 text('heading','GCN 기반 예측을 이동 대수와 동선으로 연결합니다',48,83,1170,45,29,C.navy,true);
 text('data heading','들어오는 자료',44,151,217,34,22,C.navy,true);
 const source=[];for(const [name,value,y] of [['history','대여·반납 이력',208],['inventory','대여소 재고 API',288],['weather','날씨·요일·시간',368]]){const a=box(name,value,44,y,225,70,C.white,18.5);a.text.insets={left:78,right:7,top:8,bottom:8};a.text.alignment='left';source.push(a);}
 shape('platform boundary',298,151,656,346,'#F7FAFB',C.line);
 box('platform title','타슈 예측·재배치 플랫폼',312,162,628,42,C.navy,23,C.white,true);
 const forecast=shape('forecast region',316,219,390,260,C.blue,'#C1D5E5'),planner=shape('planning region',724,219,212,260,C.orange,'#E5CFAD');
 text('forecast title','GCN + 조건부 예측 모델',333,230,356,32,21,C.navy,true);
 // Feature planes and fixed graph schematic inspired by GCN Figure 1.
 shape('feature plane back',344,274,107,97,'none','#CBD7DE','rect');shape('feature plane middle',340,278,107,97,'none','#B7C9D3','rect');shape('feature plane front',336,282,107,97,C.white,'#9DB8C4','rect');
 const nodes=[];for(const [x,y,r,fill] of [[385,326,11,'#D8ECEB'],[353,298,8,'#E2EAF4'],[421,296,8,'#FFE7CA'],[423,354,8,'#E2EAF4'],[352,355,8,'#FFE7CA']])nodes.push(shape('GCN node',x-r,y-r,r*2,r*2,fill,C.navy,'ellipse'));
 for(const [a,b] of [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4]])connect(nodes[a],nodes[b],'right','left','#657B88','straight',false,1);
 for(const n of nodes)n.bringToFront();text('GCN label','GCN',334,382,113,27,21,C.navy,true,'center');
 const latent=box('graph condition','h',459,311,28,56,C.white,18,C.navy);
 connect(nodes[0],latent,'right','left',C.navy,'straight',true,1.4);
 const mean=box('Poisson mean','포아송 평균',502,266,185,25,'#F9EDD8',16,C.navy);
 const flow=shape('FM frame',502,312,185,96,C.white,'#A8BFC7','rect');
 text('FM title','Flow Matching',509,314,171,22,17,C.navy,true,'center');
 connect(latent,mean,'top','left','#708794','straight',true,1.2);connect(latent,flow,'right','left',C.navy,'straight',true,1.2);connect(mean,flow,'bottom','top','#708794','straight',true,1.2);
 // Gaussian source, learned vector-field trajectories, and count-residual distribution.
 shape('noise region',511,338,35,34,'#E7EEF7','none','ellipse');shape('target cluster upper',649,337,21,18,'#DFEFEC','none','ellipse');shape('target cluster lower',654,358,22,17,'#DFEFEC','none','ellipse');
 const start=[],end=[];
 for(const [dx,dy] of [[3,16],[9,4],[14,22],[19,10],[25,28],[28,13]])start.push(shape('Gaussian particle',513+dx,338+dy,3.5,3.5,'#617C9C','none','ellipse'));
 for(const [dx,dy] of [[0,3],[8,0],[13,9],[6,24],[15,29],[3,31]])end.push(shape('forecast particle',651+dx,339+dy,3.5,3.5,C.teal,'none','ellipse'));
 for(let i=0;i<3;i++)connect(start[i],end[i+2],'right','left',['#567AA1','#298C87','#A88352'][i],'curved',true,1.1);
 for(const a of [...start,...end])a.bringToFront();
 text('noise caption','잡음',509,379,42,20,14,C.muted,false,'center');text('distribution caption','분포',643,379,39,20,14,C.muted,false,'center');
 const stock=box('forecast output','10분 뒤 재고·0대 가능성',332,428,358,36,C.white,19,C.navy,true);connect(flow,stock,'bottom','top',C.navy,'straight',true,1.4);
 text('planning title','재배치 계획',739,230,182,32,21,C.navy,true);
 const target=box('station selection','부족·공급 대여소',739,278,182,48,C.white,18);
 const route=box('quantity and route','이동 대수\n방문 순서',739,346,182,58,C.white,20,C.navy,true);
 box('vehicle conditions','적재량·위치\n작업 시간',739,422,182,42,C.white,16.5,C.muted);
 connect(target,route,'bottom','top');connect(forecast,planner);for(const a of source)connect(a,forecast);
 text('operator heading','운영자와 현장',988,151,240,34,22,C.navy,true);
 const monitor=shape('operator anchor',995,206,231,193,'none','none','rect');text('operator task','지도·계획표 확인',999,402,221,28,19,C.navy,true,'center');
 const truck=shape('vehicle anchor',1012,440,202,94,'none','none','rect');text('vehicle task','차량으로 수거·배치',997,534,225,25,18,C.navy,false,'center');
 connect(route,monitor);connect(monitor,truck,'bottom','top');
 const llm=box('external explanation','설명 LLM · 외부 API\n“어디에 몇 대를 옮길까?”',379,532,464,63,C.green,21,C.navy,true);
 connect(forecast,llm,'bottom','top');connect(planner,llm,'bottom','top');connect(llm,monitor,'right','left','#7EA5A6','elbow',true,1.3);
 text('difference label','서비스 차별점',46,549,236,28,19,C.teal,true);text('difference detail','미래 부족까지 확인\n수량과 동선을 함께 계산',46,582,286,52,18,C.navy);
 text('refresh','새 관측이 오면 예측과 계획을 다시 계산',378,606,605,27,18,C.teal,false,'center');
 box('records','이용 이력     |     재고 관측     |     예측·계획 기록',381,644,811,36,'#F0F4F5',18,C.muted);shape('database',335,648,30,29,C.white,C.teal,'can');text('records label','기록 저장',44,648,210,28,18,C.muted);
 s.speakerNotes.text='GCN 도식은 Kipf & Welling(2017) Figure 1의 그래프·특징 표현을 참고해 직접 그렸습니다. 실제 H13은 정적 OD 관계를 한 번 집계하며 원 논문의 다층 전체를 그대로 구현한 것은 아닙니다. FM은 Lipman et al.(2023) Figures 2–3의 확률 경로·벡터장 표현을 참고해 Gaussian 잡음에서 예측 분포로 가는 개념도로 그렸습니다. 실제 표본·적합한 궤적이 아니며, 재배치 차량 경로나 최적수송 해법을 뜻하지 않습니다. 포아송은 평균 학습과 FM 조건이며 FM의 출발 표본이 아닙니다. 데이터·사용자·차량 아이콘은 생성한 개별 PNG이고 도형·문자·연결선은 모두 편집 가능합니다. 예측 후 별도 계획을 계산하고 담당자가 확인합니다. 외부 LLM은 설명 역할입니다. 참고: https://arxiv.org/pdf/1609.02907 ; https://arxiv.org/html/2210.02747v2';
 return {nativeShapeIds:ids};
}
