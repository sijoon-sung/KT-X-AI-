'use strict';
let result;
function showCase(congested) {
  const plan = congested ? result.traffic_plan : result.baseline_plan;
  document.getElementById('normal').setAttribute('aria-pressed', String(!congested));
  document.getElementById('congested').setAttribute('aria-pressed', String(congested));
  document.getElementById('case-description').textContent = congested
    ? 'S→A 구간의 속도 10km/h를 반영했다.' : '도로 구간의 기본 이동시간을 사용했다.';
  const route = document.getElementById('route');
  route.replaceChildren();
  plan.path.forEach((name, i) => {
    if (i) {const arrow = document.createElement('b'); arrow.textContent = '→'; route.append(arrow);}
    const node = document.createElement('span'); node.textContent = name; route.append(node);
  });
  document.getElementById('distance').textContent = `${(plan.meters / 1000).toFixed(1)}km`;
  document.getElementById('duration').textContent = `${((congested ? plan.traffic_seconds : plan.baseline_seconds) / 60).toFixed(1)}분`;
  const note = document.getElementById('route-note'); note.hidden = !congested;
  note.textContent = '기존 S→A→B→S는 11분으로 마감을 넘었다. 공급량을 유지하고 S→B→A→S로 순서를 바꿨다.';
}
fetch('assets/transport-example-result.json').then(response => {
  if (!response.ok) throw Error('Result unavailable');
  return response.json();
}).then(value => {
  result = value;
  document.getElementById('contexts').textContent = JSON.stringify(result.candidate_contexts, null, 2);
  document.getElementById('normal').addEventListener('click', () => showCase(false));
  document.getElementById('congested').addEventListener('click', () => showCase(true));
  showCase(false);
}).catch(() => {document.getElementById('load-error').hidden = false;});
