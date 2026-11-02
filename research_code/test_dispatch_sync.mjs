// Transport/DOM regression only. Execute on dislab; no model inference locally.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const source=fs.readFileSync(new URL('./forecast_web/dispatch.js',import.meta.url),'utf8');
const future='2099-01-01T00:10:00Z';
const snapshot=(version,minute)=>({source_version:version,recorded_at_utc:`2099-01-01T00:0${minute}:00Z`,target_at_utc:future,default_ids:['a'],default_depot_id:'a',stations:[{id:'a',name:'Station A',current:8,lat:36.35,lon:127.38,forecast:{median:8,low:7,high:9}}]});
const A=snapshot('H18-100',0),B=snapshot('H18-200',1);
const plan=version=>({source_version:version,status:'no_action',route:{stops:[],distance_meters:0,return_minutes:0},moved_bikes:0,candidate_ids:['a'],parameters:{capacity:6,target:2,deadline_minutes:10},alternatives:[],geometry_status:'not_requested'});
function environment(fetcher){
 const nodes=new Map(),timers=new Map();let id=0;const calls=[];
 const element=key=>{if(!nodes.has(key))nodes.set(key,{value:({capacity:'6',target:'2',budget:'6',deadline:'10'})[key]||'',checked:true,hidden:false,disabled:false,innerHTML:'',textContent:'',classList:{add(){},remove(){},toggle(){}},querySelectorAll(){return [];}});return nodes.get(key);};
 const context={document:{getElementById:element,querySelectorAll:()=>[],querySelector:()=>element('query'),createElement:()=>element('created')},window:{},console,AbortController,Date,Map,Set,
  setTimeout:(fn,delay)=>{timers.set(++id,{fn,delay});return id;},clearTimeout:key=>timers.delete(key),setInterval:()=>++id,clearInterval(){},fetch:async(url,options={})=>{calls.push({url,body:options.body&&JSON.parse(options.body)});const r=await fetcher(url,options,calls);return {ok:r.status===200,status:r.status,json:async()=>r.body};}};
 vm.createContext(context);vm.runInContext(source,context);const api=vm.runInContext('({calculate,refreshFeed,applyInfo,invalidate,controls,getInfo:()=>info})',context);
 return {context,nodes,element,timers,calls,api};
}
const settle=async()=>{for(let i=0;i<30;i++)await new Promise(resolve=>setImmediate(resolve));};
const results=[];async function test(name,fn){await fn();results.push({name,passed:true});}
await test('409 latest snapshot immediately retries without click',async()=>{
 let posts=0;const e=environment(async(url)=>url.startsWith('/api/dispatch/info')?{status:200,body:A}:++posts===1?{status:409,body:{code:'source_changed',current_info:B}}:{status:200,body:plan(B.source_version)});await settle();
 assert.deepEqual(e.calls.filter(c=>c.body).map(c=>c.body.source_version),[A.source_version,B.source_version]);assert.equal(e.element('planState').textContent,'이동 없음');assert.equal(e.element('recalculate').disabled,false);
});
await test('legacy409 awaits authoritative GET before retry',async()=>{
 let posts=0;const e=environment(async(url)=>url.includes('fresh=1')?{status:200,body:B}:url.startsWith('/api/dispatch/info')?{status:200,body:A}:++posts===1?{status:409,body:{error:'changed'}}:{status:200,body:plan(B.source_version)});await settle();
 assert.equal(posts,2);assert(e.calls.some(c=>c.url.includes('fresh=1')));assert.equal(e.api.getInfo().source_version,B.source_version);
});
await test('older cached response never rolls snapshot backwards',async()=>{
 const e=environment(async(url)=>({status:200,body:url.startsWith('/api/dispatch/info')?A:plan(A.source_version)}));await settle();e.api.applyInfo(B,false);await e.api.refreshFeed({replan:false});assert.equal(e.api.getInfo().source_version,B.source_version);
});
await test('temporary503 clears fake zero and automatically reconnects',async()=>{
 let posts=0;const e=environment(async(url)=>url.startsWith('/api/dispatch/info')?{status:200,body:B}:++posts===1?{status:503,body:{error:'temporary'}}:{status:200,body:plan(B.source_version)});await settle();assert.equal(e.element('totalBikes').textContent,'—');assert.equal(e.element('export').disabled,true);
 const timer=[...e.timers.values()].find(t=>t.delay===1000);assert(timer);await timer.fn();await settle();assert.equal(posts,2);assert.equal(e.element('planState').textContent,'이동 없음');
});
await test('forecast expiry waits instead of spinning immediate requests',async()=>{
 const e=environment(async(url)=>({status:url.startsWith('/api/dispatch/info')?200:409,body:url.startsWith('/api/dispatch/info')?A:{code:'forecast_expired',error:'waiting'}}));await settle();assert.equal(e.calls.filter(c=>c.body).length,1);assert.equal(e.element('planState').textContent,'자동 재연결');assert.equal(e.element('export').disabled,true);
});
await test('invalid422 does not continuously retry operator input',async()=>{
 const e=environment(async(url)=>({status:url.startsWith('/api/dispatch/info')?200:422,body:url.startsWith('/api/dispatch/info')?A:{error:'invalid'}}));await settle();assert.equal(e.element('planState').textContent,'조건 확인');assert.equal(e.timers.size,0);
});
await test('initial connection failure recovers complete initialization',async()=>{
 let reads=0;const e=environment(async(url)=>url.startsWith('/api/dispatch/info')?++reads===1?{status:503,body:{error:'offline'}}:{status:200,body:B}:{status:200,body:plan(B.source_version)});await settle();const timer=[...e.timers.values()].find(t=>t.delay===1000);assert(timer);await timer.fn();await settle();assert.equal(e.api.controls().depot_id,'a');assert.equal(e.element('planState').textContent,'이동 없음');assert.equal(typeof e.element('recalculate').onclick,'function');
});
console.log(JSON.stringify({checks:results.length,failures:0,results}));
