'use strict';
const sections = [...document.querySelectorAll('.slide')];
const controls = document.getElementById('presentation-controls');
const toggle = document.getElementById('present');
const status = document.getElementById('slide-status');
const previous = document.getElementById('previous');
const next = document.getElementById('next');
let active = 0;
function update() {
  status.textContent = `${active + 1} / ${sections.length} · ${sections[active].dataset.title}`;
  previous.disabled = active === 0;
  next.disabled = active === sections.length - 1;
}
function show(index) {
  active = Math.max(0, Math.min(sections.length - 1, index));
  sections[active].scrollIntoView({block:'start', behavior:'instant'});
  history.replaceState(null, '', `#${sections[active].id}`);
  update();
}
function present(enabled) {
  document.body.classList.toggle('presenting', enabled);
  toggle.setAttribute('aria-pressed', String(enabled));
  toggle.textContent = enabled ? '발표 모드 종료' : '발표 모드 ↗';
  controls.hidden = !enabled;
  if (enabled) {
    const index = sections.findIndex(s => `#${s.id}` === location.hash);
    if (index >= 0) active = index;
    show(active);
    next.focus({preventScroll:true});
  } else toggle.focus({preventScroll:true});
}
toggle.addEventListener('click', () => present(controls.hidden));
document.getElementById('exit').addEventListener('click', () => present(false));
previous.addEventListener('click', () => show(active - 1));
next.addEventListener('click', () => show(active + 1));
document.querySelectorAll('a[href^="#"]').forEach(link => link.addEventListener('click', event => {
  if (controls.hidden) return;
  const index = sections.findIndex(section => `#${section.id}` === link.getAttribute('href'));
  if (index >= 0) {event.preventDefault(); show(index);}
}));
document.getElementById('print').addEventListener('click', () => window.print());
document.addEventListener('keydown', event => {
  if (controls.hidden || /INPUT|TEXTAREA|SELECT/.test(event.target.tagName) || event.target.isContentEditable) return;
  if (event.key === 'Escape') present(false);
  if (['ArrowRight','PageDown'].includes(event.key)) {event.preventDefault(); show(active + 1);}
  if (['ArrowLeft','PageUp'].includes(event.key)) {event.preventDefault(); show(active - 1);}
});
const observer = new IntersectionObserver(entries => {
  const visible = entries.filter(e => e.isIntersecting).sort((a,b) => b.intersectionRatio-a.intersectionRatio);
  if (visible.length) {active = sections.indexOf(visible[0].target); update();}
}, {rootMargin:'-78px 0px -45% 0px', threshold:[0,.2,.5]});
sections.forEach(s => observer.observe(s));
if (!['localhost','127.0.0.1','[::1]'].includes(location.hostname)) {
  const link = document.getElementById('dispatch-link');
  link.href = '#dispatch';
  link.textContent = '날짜가 명시된 기록 사례 ↑';
}
update();
