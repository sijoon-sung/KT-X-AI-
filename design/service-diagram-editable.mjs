export default async function({presentation,input}) {
 const s=presentation.slides.getItem(input.slideIndex ?? 7);
 const C={navy:'#173E50',teal:'#1A8585',line:'#BACDD5',blue:'#F0F6FB',green:'#EFF7F4',orange:'#FFF4E5',muted:'#526A75',white:'#FFFFFF'};
 const root=input.assetDirectory;
 if(input.phase==='icons-data'||input.phase==='icons-people'||input.phase==='icon-gpt') {
  const all=input.phase==='icons-data'?[['history',44,226,76,76],['inventory',44,330,76,76],['weather',44,434,76,76]]:input.phase==='icons-people'?[['operator',834,426,185,159],['truck',1045,444,176,105]]:[['openai-blossom-black',276,527,132,132]];
  return {pictures:all.map(([name,x,y,w,h])=>s.images.add({path:root+'/'+name+'.png',alt:name==='openai-blossom-black'?'OpenAI GPT logo, external LLM option':name+' service illustration',fit:'contain',position:{left:x,top:y,width:w,height:h}}).id)};
 }
 presentation.delete(input.remove);
 const ids=[];
 const shape=(name,x,y,w,h,fill=C.white,line=C.line,geo='roundRect')=>{const a=s.shapes.add({name:'H27L '+name,geometry:geo,position:{left:x,top:y,width:w,height:h},fill,line:{fill:line,width:1.3},...(['rect','textbox','roundRect'].includes(geo)?{borderRadius:9}:{})});ids.push(a.id);return a;};
 const text=(name,value,x,y,w,h,size=24,color=C.navy,bold=false,align='left')=>{const a=shape(name,x,y,w,h,'none','none','textbox');a.text=value;a.text.style={fontSize:size,color,bold,alignment:align,verticalAlignment:'middle',insets:0,wrap:true};return a;};
 const box=(name,value,x,y,w,h,fill,size=24,color=C.navy,bold=false)=>{const a=shape(name,x,y,w,h,fill);a.text=value;a.text.style={fontSize:size,color,bold,alignment:'center',verticalAlignment:'middle',insets:7,wrap:true};return a;};
 const connect=(a,b,from='right',to='left',color=C.teal,kind='straight',arrow=true,width=2.1)=>{const c=s.shapes.connect(a,b,{kind,fromSide:from,toSide:to,line:{fill:color,width},...(arrow?{tail:{type:'triangle',width:'sm',length:'sm'}}:{})});c.bringToFront();ids.push(c.id);return c;};
 text('heading','부족 위험을 예측하고, 수거·배치 수량과 동선을 정합니다',43,83,1190,48,31,C.navy,true);
 text('data heading','대여소와 주변 정보',42,169,229,35,24,C.navy,true);
 const source=[];
 for(const [name,value,y] of [['history','대여·반납\n이력',224],['inventory','실시간\n재고 API',328],['weather','날씨·요일\n시간',432]]) {
  const a=box(name,value,40,y,215,80,C.white,23);a.text.insets={left:86,right:6,top:5,bottom:5};a.text.alignment='left';source.push(a);
 }
 const forecast=shape('prediction platform',292,187,495,326,C.blue,'#B2CCD9');
 box('prediction heading','이용 흐름·부족 위험 예측',304,198,471,43,C.navy,26,C.white,true);
 text('GCN heading','GCN',320,252,166,32,26,C.navy,true,'center');
 shape('feature plane back',330,290,146,123,'none','#C4D4DF','rect');shape('feature plane middle',324,296,146,123,'none','#ADC5D2','rect');shape('feature plane front',318,302,146,123,C.white,'#8CA9BA','rect');
 const nodes=[];
 for(const [x,y,r,fill] of [[392,362,16,'#BFE0DC'],[342,324,11,'#CDDCF0'],[444,324,11,'#F4D2A9'],[444,400,11,'#CDDCF0'],[342,400,11,'#F4D2A9']])nodes.push(shape('GCN node',x-r,y-r,2*r,2*r,fill,C.navy,'ellipse'));
 for(const [a,b] of [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4]])connect(nodes[a],nodes[b],'right','left','#657B88','straight',false,1.8);
 for(const n of nodes)n.bringToFront();
 const graph=shape('GCN output anchor',318,302,146,123,'none','none','rect');
 const mean=box('Poisson mean','포아송 평균',524,259,240,36,'#FAEDD6',22,C.navy,true);
 const flow=shape('FM frame',524,314,240,115,C.white,'#A2BAC7','rect');
 text('FM heading','Flow Matching',532,317,224,30,23,C.navy,true,'center');
 connect(graph,mean,'right','left',C.navy,'elbow',true,1.7);connect(graph,flow,'right','left',C.navy,'straight',true,1.7);connect(mean,flow,'bottom','top',C.navy,'straight',true,1.7);
 shape('Gaussian source region',534,351,47,40,'#E6EDF6','none','ellipse');shape('target cluster upper',707,350,33,24,'#D9EBE5','none','ellipse');shape('target cluster lower',715,375,33,23,'#D9EBE5','none','ellipse');
 const start=[],end=[];
 for(const [dx,dy] of [[4,18],[12,5],[20,27],[27,11],[34,30],[38,18]])start.push(shape('Gaussian particle',535+dx,351+dy,5,5,'#617C9C','none','ellipse'));
 for(const [dx,dy] of [[0,4],[13,0],[22,11],[9,28],[25,33],[3,35]])end.push(shape('prediction particle',710+dx,353+dy,5,5,C.teal,'none','ellipse'));
 for(let i=0;i<3;i++)connect(start[i],end[i+2],'right','left',['#567AA1','#298C87','#A88352'][i],'curved',true,1.7);
 for(const n of [...start,...end])n.bringToFront();
 text('noise label','잡음',532,401,52,24,18,C.muted,false,'center');text('distribution label','예측 분포',671,402,90,22,18,C.muted,false,'center');
 const stock=box('forecast output','자전거가 부족해질 대여소',304,461,471,42,C.white,24,C.navy,true);connect(flow,stock,'bottom','top',C.navy,'straight',true,1.7);
 for(const a of source)connect(a,forecast);
 const planner=shape('dispatch planning',826,187,400,167,C.orange,'#E0C99F');
 text('planning heading','재배치 계획',847,199,358,38,27,C.navy,true);
 box('quantity and route','어디에 몇 대를, 어떤 순서로',846,247,360,50,C.white,25,C.navy,true);
 text('vehicle conditions','차량 적재량 · 출발 위치 · 작업 시간',846,309,360,29,20,C.muted);
 connect(forecast,planner);
 const action=shape('operator and vehicle',826,387,400,234,'#F8FAFB','#CCD8DE');
 text('operation heading','담당자 확인 · 현장 이동',844,395,365,37,25,C.navy,true);
 text('operator task','지도·계획표 확인',838,585,184,28,22,C.navy,true,'center');text('vehicle task','수거·배치',1043,557,180,31,23,C.navy,true,'center');
 const operator=shape('operator anchor',834,426,185,159,'none','none','rect'),truck=shape('truck anchor',1045,444,176,105,'none','none','rect');
 connect(planner,action,'bottom','top');connect(operator,truck,'right','left');
 const llm=shape('external LLM',292,547,495,92,C.green,'#A7CCC3');
 text('LLM title','외부 LLM',396,554,365,33,26,C.navy,true);text('LLM question','“어디에 몇 대를 옮길까?”',396,592,365,33,23,C.navy);
 connect(forecast,llm,'bottom','top');connect(llm,action,'right','left','#679B96','elbow',true,1.9);
 text('record label','관측·예측 기록 저장',42,559,218,61,22,C.muted);
 text('service advantage','미래 부족 예측   +   이동 수량·동선 계산   +   근거 확인',67,654,1146,32,23,C.teal,true,'center');
 s.speakerNotes.text='GCN은 Kipf & Welling(2017) Figure 1의 그래프·특징 표현을 참고해 편집 가능한 도형으로 다시 그렸습니다. H13은 정적 OD 관계를 한 번 집계합니다. FM은 Lipman et al.(2023) Figures 2–3의 확률 경로·벡터장 표현을 참고한 개념도이며 실측 궤적이나 차량 동선이 아닙니다. 포아송 평균은 FM 조건이고 출발 표본은 Gaussian입니다. 예측 후 별도 계획을 계산하고 담당자가 확인합니다. LLM은 예측·계획의 근거를 설명합니다. OpenAI Blossom은 사용자가 요청한 GPT 외부 모델 선택지를 표시하며 실제 검증한 연결은 Antigravity입니다. GPT API 접속 완료를 뜻하지 않습니다. OpenAI 로고의 권리는 OpenAI에 있으며 공식 배포본을 변경 없이 사용했습니다. 개별 PNG 아이콘 외에 문자·도형·연결선은 편집할 수 있습니다. 출처: https://arxiv.org/pdf/1609.02907 ; https://arxiv.org/html/2210.02747v2 ; https://openai.com/brand/ ; https://cdn.openai.com/brand/openai-logos.zip';
 return {nativeShapeIds:ids};
}
