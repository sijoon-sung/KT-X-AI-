'use strict';
const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const km=v=>(v/1000).toFixed(2),mins=v=>v.toFixed(1),labels={balanced:'균형',distance:'거리 우선',time:'시간 우선'};
let info,stationById={},ids=[],depot='',objective='balanced',plan=null,map=null,baseLayer,routeLayer,line,vehicle,allVisible=true;
let generation=0,debounce=null,animation=null,toastTimer=null,expiryTimer=null,recoveryTimer=null,recoveryAttempt=0;
let initialized=false,planAbort=null;const infoRequests=new Map();
function toast(text){$('toast').textContent=text;$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,4500);}
function stopAnimation(){clearInterval(animation);animation=null;if(vehicle&&map){map.removeLayer(vehicle);vehicle=null;}$('previewRoute').textContent='▶ 경로 미리보기';}
function controls(){return {...(info?.planner==='setup-aware-batch'?{minimum_transfer:+$('minimumTransfer').value,stop_setup_seconds:+$('stopSetup').value}:{}),...(info?.source_version?{automatic:$('automatic').checked}:{}),station_ids:ids.slice(),depot_id:depot,capacity:+$('capacity').value,target:+$('target').value,deadline_minutes:+$('deadline').value,budget:+$('budget').value,drop_limit:+$('dropLimit').value,handling_seconds:+$('handling').value,max_visits:8,objective,...(info?.source_version?{source_version:info.source_version}:{})};}
function candidates(){
 $('candidateCount').textContent=$('automatic').checked?`자동 ${ids.length}곳`:`수동 ${ids.length} / 29`;
 $('candidates').innerHTML=ids.map(id=>{const s=stationById[id];return `<div class="candidate"><span class="candidate-icon">${id===depot?'⌂':'◉'}</span><div class="candidate-name">${esc(s.name)}<span>현재 ${s.current}대 · 10분 예측 ${s.forecast.median}대</span></div><button class="remove-candidate" data-id="${esc(id)}" aria-label="${esc(s.name)} 후보 제외">×</button></div>`;}).join('');
 $('depot').innerHTML=info.stations.map(s=>s.id).sort((a,b)=>stationById[a].name.localeCompare(stationById[b].name,'ko')).map(id=>`<option value="${esc(id)}">${esc(stationById[id].name)}</option>`).join('');$('depot').value=depot;
 $('candidates').querySelectorAll('button').forEach(b=>b.onclick=()=>removeCandidate(b.dataset.id));
}
function addCandidate(id){
 if(ids.includes(id)){if(map)map.closePopup();return;}
 if(ids.length>=29){toast('직접 지정 후보는 최대29곳입니다.');return;}
 $('automatic').checked=false;
 ids.push(id);if(!depot)depot=id;candidates();invalidate();drawBase();if(map)map.closePopup();
}
function removeCandidate(id){
 if(ids.length<=1){toast('방문 후보를 최소 한 곳 남겨 주세요.');return;}
 $('automatic').checked=false;ids=ids.filter(x=>x!==id);candidates();invalidate();drawBase();
}
function invalidate(){
 clearTimeout(expiryTimer);clearTimeout(recoveryTimer);if(planAbort)planAbort.abort();generation++;stopAnimation();$('summary').classList.add('stale');$('planState').textContent='다시 계산';$('planState').classList.add('busy');
 $('calculationStatus').textContent='조건 변경 · 계산 대기';$('previewRoute').disabled=true;$('export').disabled=true;
 if(line)line.setStyle({opacity:.25,dashArray:'6 6'});
 clearTimeout(debounce);debounce=setTimeout(calculate,550);
}
function clearFailedPlan(message){
 plan=null;stopAnimation();clearTimeout(expiryTimer);$('summary').classList.add('stale');
 for(const id of ['totalDistance','totalTime','totalBikes','totalStops'])$(id).textContent='—';
 if(plan.operations){const o=plan.operations;$('coverage').innerHTML+=`<p>정차 준비 ${mins(o.setup_minutes)}분 · 상하차 ${mins(o.handling_minutes)}분</p><p>이동 1대당 ${o.work_minutes_per_delivered_bike===null?'계획 없음':mins(o.work_minutes_per_delivered_bike)+'분 (복귀 포함)'}</p><small>최소 수량 미만: 공급 ${o.below_minimum_donor_count}곳 · 부족 ${o.below_minimum_receiver_count}곳. 개별 확인 필요.</small>`;}
 $('itinerary').innerHTML='';$('returnLeg').innerHTML='';$('coverage').hidden=true;$('comparison').hidden=true;
 $('planEmpty').hidden=false;$('planEmpty').textContent=message;$('previewRoute').disabled=true;$('export').disabled=true;
 if(routeLayer)routeLayer.clearLayers();line=null;
}
function retryLater(message){
 clearTimeout(recoveryTimer);const delay=[1000,2500,5000,10000,15000][Math.min(recoveryAttempt++,4)];
 $('planState').textContent='자동 재연결';$('calculationStatus').textContent=message;
 recoveryTimer=setTimeout(async()=>{try{if(!initialized){await init();return;}await refreshFeed({fresh:true,replan:false});await calculate();}catch(e){retryLater('최신 관측 연결을 다시 확인하고 있습니다.');}},delay);
}
async function calculate(){
 if(!info)return;clearTimeout(expiryTimer);clearTimeout(recoveryTimer);clearTimeout(debounce);if(planAbort)planAbort.abort();
 const request=++generation;planAbort=new AbortController();const abort=planAbort;const requestTimeout=setTimeout(()=>abort.abort(),70000);
 $('planState').textContent='계산 중';$('planState').classList.add('busy');$('calculationStatus').textContent='수량과 도로 경로를 계산하고 있습니다…';$('recalculate').disabled=true;
 $('summary').classList.add('stale');$('previewRoute').disabled=true;$('export').disabled=true;stopAnimation();
 try{
  for(let attempt=0;attempt<2;attempt++){
   const body=controls();const response=await fetch('/api/dispatch/optimize',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:abort.signal}),value=await response.json();
   if(request!==generation)return;
   if(response.ok){
    if(value.source_version&&value.source_version!==info.source_version)throw Object.assign(Error('최신 관측과 계획을 다시 맞추고 있습니다.'),{status:409});
    plan=value;if(body.automatic){ids=value.candidate_ids.slice();candidates();}recoveryAttempt=0;renderPlan();$('mapBanner').hidden=true;
    if(value.geometry_status==='unavailable')banner('도로시간은 계산했지만 경로선 조회가 실패했습니다. 점선은 방문 순서 연결입니다.');return;
   }
   if(response.status===409&&(!value.code||value.code==='source_changed')){
    const previous=info.source_version;clearFailedPlan('최신 관측을 받아 계산을 이어 갑니다.');$('planState').textContent='관측 갱신 중';
    if(value.current_info)applyInfo(value.current_info,false);else await refreshFeed({fresh:true,replan:false});
    if(request!==generation)return;if(attempt===0&&info.source_version!==previous)continue;
   }
   throw Object.assign(Error(value.error||'계산을 완료하지 못했습니다.'),{status:response.status});
  }
 }catch(e){
  if(request!==generation)return;const message=e.name==='AbortError'?'계산 응답이 늦어 다시 연결합니다.':e.message||'계산 장비 연결을 다시 확인하고 있습니다.';
  clearFailedPlan(message);banner(message);
  if(!e.status||e.status===409||e.status>=500)retryLater(message);else{$('planState').textContent='조건 확인';$('calculationStatus').textContent=message;}
 }finally{clearTimeout(requestTimeout);if(request===generation){$('recalculate').disabled=false;$('planState').classList.remove('busy');}}
}

function banner(message){$('mapBanner').textContent=message;$('mapBanner').hidden=false;}
function renderPlan(){
 const r=plan.route;$('summary').classList.remove('stale');$('totalDistance').innerHTML=`${km(r.distance_meters)}<small>km</small>`;
 $('totalTime').innerHTML=`${mins(r.return_minutes)}<small>분</small>`;$('totalBikes').innerHTML=`${plan.moved_bikes}<small>대</small>`;$('totalStops').innerHTML=`${r.stops.length}<small>곳</small>`;
 $('planState').textContent=plan.status==='no_action'?'이동 없음':'계산 완료';$('planEmpty').hidden=r.stops.length>0;
 $('planEmpty').textContent=plan.no_action_reason||'입력 조건에서 가능한 이동이 없습니다.';
 const c=plan.coverage;$('coverage').hidden=!c;if(c){const until=new Date(plan.valid_until_utc).toLocaleTimeString('ko-KR',{timeZone:'Asia/Seoul',hour:'2-digit',minute:'2-digit',hour12:false});$('coverage').innerHTML=`<b>${c.scanned}곳 탐색 → 후보 ${c.shortlisted}곳</b><p>현재 0대 ${c.current_zero}곳 중 <strong>${c.zero_sites_supplied}곳 공급 계획</strong> · 이번 계획 외 ${c.unserved_zero}곳</p><p>공급 가능 ${c.donors}곳 / 부족 후보 ${c.receivers}곳</p><small>${until} 예측 · ${plan.latest_departure_at_utc&&plan.moved_bikes>0?new Date(plan.latest_departure_at_utc).toLocaleTimeString('ko-KR',{timeZone:'Asia/Seoul',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false})+'까지 출발 · ':''}최대 ${mins(plan.parameters.effective_deadline)}분 안에 복귀 ${plan.operations?` · 작업 완료 ${mins(plan.parameters.action_deadline_minutes)}분 이내 · 정차 준비 ${plan.parameters.stop_setup_seconds}초/곳 · 최소 ${plan.parameters.minimum_transfer}대/곳`:''} · 상하차 ${plan.parameters.handling_seconds??6}초/대 · 차량·주차 가정</small>`;}
 if(plan.operations){const o=plan.operations;$('coverage').innerHTML+=`<p>정차 준비 ${mins(o.setup_minutes)}분 · 상하차 ${mins(o.handling_minutes)}분</p><p>이동 1대당 ${o.work_minutes_per_delivered_bike===null?'계획 없음':mins(o.work_minutes_per_delivered_bike)+'분 (복귀 포함)'}</p><small>최소 수량 미만: 공급 ${o.below_minimum_donor_count}곳 · 부족 ${o.below_minimum_receiver_count}곳. 개별 확인 필요.</small>`;}
 $('itinerary').innerHTML=r.stops.map((s,i)=>`<div class="stop ${s.action}" data-stop="${i}" tabindex="0" role="button"><span class="stop-number">${i+1}</span><div class="stop-main"><div class="stop-name">${esc(stationById[s.station_id].name)}</div><div class="stop-action">${s.action==='pickup'?'수거':'배치'} ${s.bikes}대 <span class="load-label">작업 후 적재 ${s.vehicle_load_after} / ${plan.parameters.capacity}대</span></div><div class="stop-leg">이전 지점에서 ${km(s.leg_meters)}km · ${mins(s.leg_drive_minutes)}분 / 도착 +${mins(s.eta_minutes)}분${plan.operations?` · 작업 완료 +${mins(s.completion_minutes)}분`:''}</div></div></div>`).join('');
 $('itinerary').querySelectorAll('[data-stop]').forEach(el=>{const action=()=>selectStop(+el.dataset.stop);el.onclick=action;el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();action();}};});
 $('returnLeg').innerHTML=r.stops.length?`<div class="return-leg">↩ ${esc(stationById[depot].name)} 복귀<br><strong>${km(r.return_leg_meters)}km · ${mins(r.return_leg_minutes)}분</strong> / 전체 ${mins(r.return_minutes)}분</div>`:'';
 $('comparison').hidden=!!plan.comparison;const names={direct_capacity:'기존 단일 탐색',smaller_capacity:'작은 적재량 대안',no_action:'이동 없음',best_pair:'두 곳 이동',nearest_pair:'최근접 두 곳',legacy_fixed5:'기존 고정 후보'};$('alternatives').innerHTML=plan.alternatives.filter(a=>plan.planner?['legacy_fixed5','nearest_pair','direct_capacity'].includes(a.objective):a.objective!=='best_pair').map(a=>`<div class="comparison-row ${a.objective===objective?'active':''}" ${names[a.objective]?'':`data-mode="${a.objective}" role="button" tabindex="0"`}><span>${names[a.objective]||labels[a.objective]}<small>${a.moved_bikes}대 이동 · ${mins(a.return_minutes)}분</small></span><b>${km(a.distance_meters)} km</b></div>`).join('');
 $('alternatives').querySelectorAll('[data-mode]').forEach(el=>{el.onclick=()=>setObjective(el.dataset.mode);el.onkeydown=e=>{if(e.key==='Enter')setObjective(el.dataset.mode);};});
 $('previewRoute').disabled=!r.stops.length;$('export').disabled=false;
 const date=new Date(info.recorded_at_utc).toLocaleString('ko-KR',{timeZone:'Asia/Seoul',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false});
 $('calculationStatus').textContent=`${date} 관측 · ${plan.parameters.automatic?'자동 후보':'수동 후보'} ${ids.length}곳 · 계획 계산 완료`;$('conditionSummary').innerHTML=`적재 ${plan.parameters.capacity}대 · 목표 ${plan.parameters.target}대 · 복귀 ${plan.parameters.deadline_minutes}분${plan.operations?` · 최소 ${plan.parameters.minimum_transfer}대/곳`:''} <span>설정 ›</span>`;drawBase();drawRoute();fit();
 clearTimeout(expiryTimer);if(plan.latest_departure_at_utc&&plan.moved_bikes>0){expiryTimer=setTimeout(()=>{plan=null;invalidate();banner('출발 가능 시각이 지나 현재 조건으로 다시 계산합니다.');},Math.max(0,Date.parse(plan.latest_departure_at_utc)-Date.now()));}
}
function icon(type,text){return L.divIcon({className:'',html:`<div class="route-pin ${type}">${text}</div>`,iconSize:type==='depot'?[22,22]:[30,30],iconAnchor:type==='depot'?[11,11]:[15,15]});}
function popup(s){
 const div=document.createElement('div');div.className='popup';div.innerHTML=`<h3>${esc(s.name)}</h3><p>${esc(s.id)}<br>현재 ${s.current}대 · 10분 예측 ${s.forecast.median}대<br>예측 범위 ${s.forecast.low}–${s.forecast.high}대</p>`;
 const b=document.createElement('button');b.textContent=ids.includes(s.id)?'방문 후보에서 제외':'방문 후보에 추가';b.onclick=()=>ids.includes(s.id)?removeCandidate(s.id):addCandidate(s.id);div.appendChild(b);const start=document.createElement('button');start.textContent='차량 출발·복귀 위치로 설정';start.onclick=()=>{depot=s.id;candidates();invalidate();if(map)map.closePopup();};div.appendChild(start);return div;
}
function drawBase(){
 if(!map){fallback();return;}baseLayer.clearLayers();
 const visible=allVisible?info.stations:ids.map(id=>stationById[id]);
 for(const s of visible){const selected=ids.includes(s.id);L.circleMarker([s.lat,s.lon],{radius:selected?6:3,weight:selected?2:0,color:'#fff',fillColor:selected?'#748db2':s.current===0?'#d2a967':'#96adc0',fillOpacity:selected?.9:.48}).addTo(baseLayer).bindPopup(()=>popup(s));}
}
function drawRoute(){
 if(!map){fallback();return;}routeLayer.clearLayers();line=null;if(!plan)return;
 const r=plan.route,anchor=stationById[depot],pts=plan.geometry?.coordinates.map(p=>[p[1],p[0]])||[[anchor.lat,anchor.lon],...r.stops.map(s=>{const p=stationById[s.station_id];return [p.lat,p.lon];}),[anchor.lat,anchor.lon]];
 if(r.stops.length){L.polyline(pts,{color:'#fff',weight:8,opacity:.85}).addTo(routeLayer);line=L.polyline(pts,{color:'#477bea',weight:4,opacity:.95,dashArray:plan.geometry?null:'7 6'}).addTo(routeLayer);}
 r.stops.forEach((s,i)=>L.marker([stationById[s.station_id].lat,stationById[s.station_id].lon],{icon:icon(s.action,String(i+1))}).addTo(routeLayer).bindTooltip(`${esc(stationById[s.station_id].name)} · ${s.action==='pickup'?'수거':'배치'} ${s.bikes}대`).on('click',()=>selectStop(i)));
 L.marker([anchor.lat,anchor.lon],{icon:icon('depot','⌂'),zIndexOffset:200}).addTo(routeLayer).bindTooltip('출발 · 복귀 위치 (가정)');
}
function fit(){if(!map)return;const points=[stationById[depot],...(plan?.route.stops.length?plan.route.stops.map(s=>stationById[s.station_id]):ids.map(id=>stationById[id]))].filter(Boolean);if(points.length)map.fitBounds(points.map(s=>[s.lat,s.lon]),{paddingTopLeft:[40,60],paddingBottomRight:[map.getSize().x>480&&!$('comparison').hidden?260:40,60],maxZoom:15});}
function selectStop(i){
 const s=plan.route.stops[i];document.querySelectorAll('[data-stop]').forEach((el,j)=>el.classList.toggle('active',j===i));
 if(map)map.panTo([stationById[s.station_id].lat,stationById[s.station_id].lon]);
 toast(`${i+1}. ${stationById[s.station_id].name} · ${s.action==='pickup'?'수거':'배치'} ${s.bikes}대 · 적재 ${s.vehicle_load_after}대`);
}
function fallback(){
 if(map)return;const svg=$('fallbackMap');svg.hidden=false;const rows=ids.map(id=>stationById[id]);if(!rows.length)return;
 const xs=rows.map(s=>s.lon),ys=rows.map(s=>s.lat),x0=Math.min(...xs),y0=Math.min(...ys),dx=Math.max(...xs)-x0||.001,dy=Math.max(...ys)-y0||.001;
 const point=s=>[80+(s.lon-x0)/dx*650,590-(s.lat-y0)/dy*450];let html='<text x="40" y="50" fill="#8da2b1" font-size="14">지도 배경을 불러오지 못해 좌표 위치도를 표시합니다.</text>';
 if(plan?.route.stops.length){const order=[depot,...plan.route.stops.map(s=>s.station_id),depot];html+=`<polyline points="${order.map(id=>point(stationById[id]).join(',')).join(' ')}" fill="none" stroke="#477bea" stroke-width="4" stroke-dasharray="8 6"/>`;}
 html+=rows.map(s=>{const [x,y]=point(s);return `<circle cx="${x}" cy="${y}" r="10" fill="#608bd2"/><text x="${x}" y="${y-20}" text-anchor="middle" fill="#6c8296" font-size="12">${esc(s.name)}</text>`;}).join('');svg.innerHTML=html;
}
function setObjective(mode){objective=mode;document.querySelectorAll('[data-objective]').forEach(b=>b.classList.toggle('active',b.dataset.objective===mode));$('objectiveHint').textContent={balanced:'부족 예상과 이동 비용을 함께 고려합니다.',distance:'이동거리에 더 높은 비중을 두고 수량과 순서를 정합니다.',time:'이동과 작업에 걸리는 시간을 우선 고려합니다.'}[mode];invalidate();}
function preview(){
 if(animation){stopAnimation();return;}if(!plan?.route.stops.length)return;
 if(!map){toast('좌표 위치도에서는 방문 순서를 작업표에서 확인하세요.');return;}
 const anchor=stationById[depot],points=plan.geometry?.coordinates.map(p=>[p[1],p[0]])||[[anchor.lat,anchor.lon],...plan.route.stops.map(s=>[stationById[s.station_id].lat,stationById[s.station_id].lon]),[anchor.lat,anchor.lon]];
 let i=0;vehicle=L.marker(points[0],{icon:icon('vehicle','↗'),zIndexOffset:500}).addTo(map);$('previewRoute').textContent='Ⅱ 미리보기 중지';
 animation=setInterval(()=>{i+=Math.max(1,Math.ceil(points.length/45));if(i>=points.length){stopAnimation();return;}vehicle.setLatLng(points[i]);},160);
}
function exportPlan(){
 const lines=[['기준시각',info.recorded_at_utc],['계산기준',labels[objective]],['총거리_m',plan.route.distance_meters],['총시간_분',plan.route.return_minutes],[],['순서','대여소ID','대여소','작업','대수','구간거리_m','구간이동_분','도착_분','적재_대']];
 plan.route.stops.forEach((s,i)=>lines.push([i+1,s.station_id,stationById[s.station_id].name,s.action==='pickup'?'수거':'배치',s.bikes,s.leg_meters,s.leg_drive_minutes,s.eta_minutes,s.vehicle_load_after]));lines.push(['복귀',depot,stationById[depot].name,'복귀',0,plan.route.return_leg_meters,plan.route.return_leg_minutes,plan.route.return_minutes,0]);
 lines.push([],['조건','차량·거점·주차·작업시간 가정. 실제 배차 없음.']);const csv='\ufeff'+lines.map(row=>row.map(x=>'"'+String(x).replaceAll('"','""')+'"').join(',')).join('\r\n');
 const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download='타슈_재배치_작업표.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function renderFeed(){
const f=info.feed;if(!f)return;$('feedBar').hidden=false;const live=info.mode==='live_observation_shadow',clock=t=>t?new Date(t).toLocaleTimeString('ko-KR',{timeZone:'Asia/Seoul',hour:'2-digit',minute:'2-digit',hour12:false}):'대기';
const ended=['failed','stopped'].includes(f.phase);document.querySelector('.record-badge').textContent=live?(ended?'수집 중지':'API 관측'):'기록 재생';
$('feedTitle').textContent=!live?'기록 재생 · 표시된 시각을 기준으로 계산':ended?'수집 종료 · 마지막 관측 표시':'API 관측 → 10분 예측 → 자동 재배치 계획';
$('feedSub').textContent=`관측 ${clock(f.latest_observation?.observed_at_utc)} · 예측 대상 ${clock(f.latest_forecast?.target_at_utc)} · 지원 ${f.latest_forecast?.stations??'—'}곳 / 관측 당시 0대 ${f.latest_forecast?.current_zero??'—'}곳`;
if(!live)$('dataModeHelp').textContent='표시된 시각의 실제 관측과 예측을 재생합니다. 차량 조건을 바꾸면 같은 기록에서 이동 수량과 동선을 다시 계산합니다.';
if(live){$('recordTime').textContent=new Date(info.recorded_at_utc).toLocaleString('ko-KR',{timeZone:'Asia/Seoul',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false})+' 관측 기준';$('dataModeHelp').textContent='10분 주기의 실제 API 관측과 고정 예측입니다. 수량·동선은 관측 시점을 출발점으로 하는 가정 계획이며 실제 배차가 아닙니다.';}
const score=f.latest_score,all=score?.groups?.all,metric=(v,d=3)=>typeof v==='number'?v.toFixed(d):'—';
$('feedDetails').innerHTML=score?`<b>최근 관측으로 확인한 예측 · ${clock(score.target_at_utc)}</b><div class="feed-score"><span>채점 대여소</span><b>${score.pairs}곳</b></div><div class="feed-score"><span>GCN+FM 수량 오차 MAE</span><b>${metric(all?.recent_cfm?.median_mae)}대</b></div><div class="feed-score"><span>현재 유지 수량 오차 MAE</span><b>${metric(all?.persistence?.median_mae)}대</b></div><div class="feed-score"><span>0대 → 양수 관측</span><b>${score.zero_to_positive}곳</b></div><div class="feed-score"><span>양수 → 0대 관측</span><b>${score.positive_to_zero}곳</b></div><p>신규 회차의 진단값이며 기존 9월 공식 평가를 대체하지 않습니다. API로 모델을 학습하거나 보정하지 않았습니다.</p>`:`<b>첫 검증 점수 대기</b><p>예측을 먼저 발행한 뒤 다음 관측을 받아야 채점할 수 있습니다. 아직 관측되지 않은 정답의 점수는 표시하지 않습니다.</p>`;
if(f.latest_forecast?.previous_proxy_present===0)$('feedDetails').innerHTML+='<p>초기 회차는 직전 10분 관측이 없어 해당 입력을 결측으로 표시했습니다.</p>';
 const overdue=live&&Date.now()-Date.parse(info.recorded_at_utc)>720000;if(overdue){$('feedTitle').textContent='관측 갱신 지연 · 마지막 자료 표시';document.querySelector('.record-badge').textContent='연결 확인 필요';}
}
function applyInfo(next,replan=true){
 if(!next||!Array.isArray(next.stations)||!next.stations.length)throw Error('관측 자료를 불러오지 못했습니다.');
 const currentTime=Date.parse(info?.recorded_at_utc),nextTime=Date.parse(next.recorded_at_utc);
 if(Number.isFinite(currentTime)&&Number.isFinite(nextTime)&&nextTime<currentTime)return false;
 const changed=!!(info&&next.source_version&&next.source_version!==info.source_version);info=next;stationById=Object.fromEntries(info.stations.map(s=>[s.id,s]));
 if(changed){ids=ids.filter(id=>stationById[id]);if(!ids.length)ids=info.default_ids.slice();if(!stationById[depot])depot=info.default_depot_id;candidates();drawBase();if(replan)invalidate();}
 renderFeed();return changed;
}
async function refreshFeed(options={}){
 const key=options.fresh?'fresh':'cached';let pending=infoRequests.get(key);
 if(!pending){pending=(async()=>{const r=await fetch('/api/dispatch/info'+(options.fresh?'?fresh=1':''),{cache:'no-store'});if(!r.ok)throw Error('최신 관측 연결을 확인하고 있습니다.');return r.json();})();infoRequests.set(key,pending);}
 try{
  const next=await pending;const changed=applyInfo(next,options.replan!==false);
  if(info.mode==='live_observation_shadow'&&options.replan!==false&&plan?.latest_departure_at_utc&&plan.moved_bikes>0&&Date.now()>Date.parse(plan.latest_departure_at_utc)){plan=null;banner('출발 가능 시각이 지나 현재 조건으로 다시 계산합니다.');invalidate();}
  if(info.mode==='live_observation_shadow'&&plan?.valid_until_utc&&Date.now()>Date.parse(plan.valid_until_utc)){clearFailedPlan('예측 유효시간이 끝나 다음 관측을 기다립니다.');retryLater('다음 관측을 기다립니다.');}
  return changed;
 }finally{if(infoRequests.get(key)===pending)infoRequests.delete(key);}
}

async function init(){
 try{
  const r=await fetch('/api/dispatch/info');if(!r.ok)throw Error('관측 기록을 불러오지 못했습니다.');info=await r.json();stationById=Object.fromEntries(info.stations.map(s=>[s.id,s]));ids=info.default_ids.slice();depot=info.default_depot_id;if(!info.source_version){$('automatic').checked=false;$('automatic').disabled=true;}
  const cached=info.mode==='recorded_replay'?info.initial_plan:null;
  if(info.planner==='setup-aware-batch'){
   ['batchField','setupField','batchHint'].forEach(id=>$(id).hidden=false);
   const settings=cached?.parameters||info.operating_defaults;
   const fields={deadline:['deadline_minutes','분'],minimumTransfer:['minimum_transfer','대'],stopSetup:['stop_setup_seconds','초 / 곳'],capacity:['capacity','대'],target:['target','대'],budget:['budget','대'],dropLimit:['drop_limit','대'],handling:['handling_seconds','초']};
   for(const [id,[key,unit]] of Object.entries(fields)){
    const field=$(id),value=String(settings[key]);
    if(![...field.options].some(option=>option.value===value))field.add(new Option(value+unit,value));
    field.value=value;
   }
   if(cached){
    depot=settings.depot_id;ids=cached.candidate_ids.slice();objective=settings.objective;
    $('automatic').checked=settings.automatic;
    document.querySelectorAll('[data-objective]').forEach(button=>button.classList.toggle('active',button.dataset.objective===objective));
   }
  }else{[...$('deadline').options].filter(o=>+o.value>10).forEach(o=>o.remove());}
  $('recordTime').textContent=new Date(info.recorded_at_utc).toLocaleString('ko-KR',{timeZone:'Asia/Seoul',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false})+' 기준';
  if(window.L){map=L.map('map',{zoomControl:false}).setView([36.32,127.39],14);L.control.zoom({position:'topright'}).addTo(map);L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',keepBuffer:1}).addTo(map);baseLayer=L.layerGroup().addTo(map);routeLayer=L.layerGroup().addTo(map);map.on('resize',fit);}
  candidates();drawBase();fit();$('recalculate').disabled=false;
  ['capacity','target','deadline','budget','automatic','dropLimit','handling','minimumTransfer','stopSetup'].forEach(id=>$(id).onchange=invalidate);$('depot').onchange=()=>{depot=$('depot').value;candidates();invalidate();};
  const tab=setting=>{$('controlScroll').className='control-scroll '+(setting?'view-settings':'view-route');$('settingsTab').classList.toggle('active',setting);$('routeTab').classList.toggle('active',!setting);};$('settingsTab').onclick=()=>tab(true);$('routeTab').onclick=()=>tab(false);$('conditionSummary').onclick=()=>tab(true);
  $('restore').onclick=()=>{$('automatic').checked=true;candidates();invalidate();};
  document.querySelectorAll('[data-objective]').forEach(b=>b.onclick=()=>setObjective(b.dataset.objective));$('recalculate').onclick=calculate;
  $('fitMap').onclick=fit;$('showAll').onclick=()=>{allVisible=!allVisible;$('showAll').classList.toggle('active',allVisible);drawBase();};
  $('search').oninput=()=>{const q=$('search').value.trim().toLowerCase();$('searchResults').hidden=!q;if(!q)return;const matches=info.stations.filter(s=>s.name.toLowerCase().includes(q)||s.id.toLowerCase().includes(q)).slice(0,8);$('searchResults').innerHTML=matches.map(s=>`<button data-id="${esc(s.id)}">${esc(s.name)}<small>${esc(s.id)} · 현재 ${s.current}대</small></button>`).join('')||'<button disabled>검색 결과가 없습니다.</button>';$('searchResults').querySelectorAll('[data-id]').forEach(b=>b.onclick=()=>{const s=stationById[b.dataset.id];$('searchResults').hidden=true;if(map){map.setView([s.lat,s.lon],16);L.popup().setLatLng([s.lat,s.lon]).setContent(popup(s)).openOn(map);}else addCandidate(s.id);});};
  $('previewRoute').onclick=preview;$('export').onclick=exportPlan;$('help').onclick=()=>$('helpDialog').showModal();$('closeHelp').onclick=()=>$('helpDialog').close();$('feedInspect').onclick=()=>$('helpDialog').showModal();
  document.querySelector('.rail-active').onclick=fit;renderFeed();initialized=true;setInterval(()=>refreshFeed().catch(()=>{$('feedTitle').textContent='관측 상태 확인 지연 · 자동 재연결';}),30000);
  if(cached){plan=cached;renderPlan();}else{await calculate();}
 }catch(e){clearFailedPlan(e.message);banner(e.message);retryLater('관측 자료 연결을 다시 확인하고 있습니다.');}
}
init();
