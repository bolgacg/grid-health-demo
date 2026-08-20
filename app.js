/* The Night Shift - control room for a simulated 10 kV network.
   All data precomputed offline (study2/run_study.py, seeds 0-4); this file only renders. */
'use strict';
(function(){
const TL = window.TL, RS = window.RS;
const META = TL.meta;
const BUDGETS = META.budgets;               // [3.6 .. 2.0] z-units
const N_DAYS = META.n_days, TRAIN = META.train_days, REGIME = META.regime_day;
const ARCHS = [
  {key:'naive',  label:'static limit'},
  {key:'single', label:'forecaster residual'},
  {key:'vote',   label:'multi-method vote'},
  {key:'gated',  label:'vote + uncertainty weighting'},
  {key:'seq',    label:'full sequential gate'},
  {key:'recal',  label:'gate + weekly retraining', rsLabel:'sequential gate + rolling recalibration', shiftOnly:true},
];
const rsLabel = a => a.rsLabel || a.label;

const S = { scenario:'stationary', arch:'seq', bIdx:2, day:0, sel:4, playing:false };
try{ const q=new URLSearchParams(location.search);
  if(q.get('w')) S.scenario=q.get('w');
  if(q.get('a')) S.arch=q.get('a');
  if(q.get('t')) S.sel=+q.get('t');
}catch(e){}

/* ---------- topology ---------- */
const FEEDER_Y = [92, 218, 344, 470];
const NODE_X = [235, 345, 455, 565, 675, 785];
const BUS_X = 150;
function assetPos(a){ const f = Math.floor(a/6), i = a%6; return {x:NODE_X[i], y:FEEDER_Y[f], f}; }
const FNAMES = ['F1','F2','F3','F4'];

/* ---------- helpers ---------- */
const $ = s => document.querySelector(s);
const el = (tag, attrs, parent) => {
  const ns = 'http://www.w3.org/2000/svg';
  const e = ['svg','g','path','circle','rect','line','text','polyline','polygon'].includes(tag)
    ? document.createElementNS(ns, tag) : document.createElement(tag);
  for (const k in attrs||{}) {
    if (k==='text') e.textContent = attrs[k];
    else e.setAttribute(k, attrs[k]);
  }
  if (parent) parent.appendChild(e);
  return e;
};
const fmt1 = x => (Math.round(x*10)/10).toString();
const bKey = () => String(BUDGETS[S.bIdx]);
const data = () => TL[S.scenario];
const archAvailable = a => !(a.shiftOnly && S.scenario!=='shift');
function currentArch(){ const a = ARCHS.find(x=>x.key===S.arch); return archAvailable(a) ? a : ARCHS[4]; }
function maxRamp(a){ const r = data().ramp[a]; let m=0; for (const v of r) if (v>m) m=v; return m; }
function isFault(a){ return data().fault_assets.includes(a); }
function ticketsFor(archKey){ return data().tickets[bKey()][archKey] || {}; }
function sigmaOf(a){
  const d = data(); let s=0, n=0;
  for (let t=5;t<TRAIN;t++){ const e = d.hotspot[a][t]-d.pred[a][t]; s+=e*e; n++; }
  return Math.sqrt(s/n);
}
function kMean(a){ const K=data().K[a]; let s=0; for (let t=0;t<TRAIN;t++) s+=K[t]; return s/TRAIN; }

/* ---------- the map ---------- */
const MAP = {nodes:[], pins:null, svg:null};
function buildMap(){
  const svg = el('svg', {viewBox:'0 0 858 560', role:'img',
    'aria-label':'One-line diagram of the simulated 10 kV network'}, $('#gridmap'));
  MAP.svg = svg;
  // bus + substation
  el('line',{x1:BUS_X,y1:FEEDER_Y[0]-14,x2:BUS_X,y2:FEEDER_Y[3]+14,stroke:'var(--line2)','stroke-width':5},svg);
  const sub = el('g',{},svg);
  el('rect',{x:38,y:252,width:76,height:56,rx:3,fill:'var(--panel2)',stroke:'var(--bright)','stroke-width':1.2},sub);
  el('text',{x:76,y:276,'text-anchor':'middle',fill:'var(--bright)','font-family':'var(--mono)','font-size':'11px',text:'60/10 kV'},sub);
  el('text',{x:76,y:292,'text-anchor':'middle',fill:'var(--faint)','font-family':'var(--mono)','font-size':'10px',text:'primary'},sub);
  el('line',{x1:114,y1:280,x2:BUS_X,y2:280,stroke:'var(--line2)','stroke-width':2},svg);
  // feeders
  FEEDER_Y.forEach((y,f)=>{
    el('line',{x1:BUS_X,y1:y,x2:NODE_X[5],y2:y,stroke:'var(--line)','stroke-width':1.6},svg);
    el('text',{x:BUS_X+18,y:y-13,fill:'var(--faint)','font-family':'var(--mono)','font-size':'10.5px',
      text:FNAMES[f]+' · trunk cable'},svg);
  });
  // transformer nodes: the two-circle one-line symbol
  for (let a=0;a<24;a++){
    const {x,y} = assetPos(a);
    const g = el('g',{class:'node',tabindex:0,role:'button','aria-label':'Transformer T'+String(a+1).padStart(2,'0'),
      style:'cursor:pointer'},svg);
    el('circle',{cx:x-4.5,cy:y,r:8.5,fill:'var(--void)','stroke-width':2},g).classList.add('c1');
    el('circle',{cx:x+4.5,cy:y,r:8.5,fill:'none','stroke-width':2},g).classList.add('c2');
    el('circle',{cx:x,cy:y,r:17,fill:'none',stroke:'transparent','stroke-width':1.6},g).classList.add('ring');
    el('text',{x:x,y:y+31,'text-anchor':'middle',fill:'var(--faint)','font-family':'var(--mono)','font-size':'10px',
      text:'T'+String(a+1).padStart(2,'0')},g);
    g.addEventListener('click',()=>{ S.sel=a; renderRail(); paintMap(); });
    g.addEventListener('keydown',e=>{ if(e.key==='Enter'||e.key===' '){e.preventDefault(); S.sel=a; renderRail(); paintMap();}});
    MAP.nodes.push(g);
  }
  MAP.pins = el('g',{},svg);
  el('text',{x:NODE_X[5]+55,y:548,'text-anchor':'end',fill:'var(--faint)','font-family':'var(--mono)','font-size':'10px',
    text:'Click a transformer to inspect it. Colour shows the true asset condition; markers show tickets from the selected architecture.'},svg);
}
function healthColor(a){
  const d=data(), t=S.day;
  const fd = d.fail_day[String(a)];
  if (fd!=null && t>=fd) return 'var(--signal)';
  const m = maxRamp(a); if (m<=0) return 'var(--phosphor)';
  const r = d.ramp[a][t]/m;
  if (r<0.08) return 'var(--phosphor)';
  return r<0.5 ? 'var(--sodium)' : '#b5541f';
}
function paintMap(){
  const d=data(); const tix = ticketsFor(currentArch().key);
  for (let a=0;a<24;a++){
    const g=MAP.nodes[a], col=healthColor(a);
    g.querySelector('.c1').setAttribute('stroke',col);
    g.querySelector('.c2').setAttribute('stroke',col);
    const fd = d.fail_day[String(a)];
    g.classList.toggle('pulse', fd!=null && S.day>=fd);
    g.querySelector('.ring').setAttribute('stroke', a===S.sel ? 'var(--steel)' : 'transparent');
  }
  // ticket pins
  MAP.pins.innerHTML='';
  for (const aStr in tix){
    const a=+aStr, {x,y}=assetPos(a);
    tix[aStr].forEach((td,i)=>{
      if (td>S.day) return;
      const age = S.day-td, op = age<20 ? 1 : 0.35;
      const px = x-10+((i%5)*5), py = y-21;
      el('polygon',{points:`${px},${py} ${px+7},${py} ${px},${py-9}`,
        fill: isFault(a)&&td>=(+data().onset[aStr]||1e9)-5 ? 'var(--amber-bright)' : 'var(--signal)',
        opacity:op},MAP.pins);
    });
  }
  const clock=$('#clock');
  let phase='';
  if (S.scenario==='shift' && S.day>=REGIME) phase = S.day<REGIME+25 ? ' · <span class="warn">heat wave</span>' : ' · <span class="warn">EV ramp</span>';
  else if (S.day<TRAIN) phase=' · calibration';
  clock.innerHTML='Day '+String(S.day).padStart(3,'0')+phase;
}

/* ---------- sparkline helper ---------- */
function spark(w,h,series,opts){
  const svg=el('svg',{viewBox:`0 0 ${w} ${h}`});
  const o=Object.assign({pad:6},opts||{});
  let lo=Infinity,hi=-Infinity;
  series.forEach(s=>s.pts.forEach(v=>{ if(v==null)return; if(v<lo)lo=v; if(v>hi)hi=v; }));
  if (o.lo!=null) lo=Math.min(lo,o.lo); if (o.hi!=null) hi=Math.max(hi,o.hi);
  if (hi-lo<1e-9) hi=lo+1;
  const X=t=>o.pad+(w-2*o.pad)*t/(N_DAYS-1), Y=v=>h-o.pad-(h-2*o.pad)*(v-lo)/(hi-lo);
  if (S.scenario==='shift' && !o.noRegime)
    el('rect',{x:X(REGIME),y:0,width:2,height:h,fill:'var(--sodium)',opacity:.5},svg);
  if (!o.noCursor) el('rect',{x:X(0),y:0,width:X(TRAIN)-X(0),height:h,fill:'var(--steel)',opacity:.07},svg);
  series.forEach(s=>{
    if (s.band){
      let up='',dn='';
      for(let t=0;t<s.pts.length;t++){ if(s.pts[t]==null)continue; up+=`${X(t)},${Y(s.pts[t]+s.band[t])} `; }
      for(let t=s.pts.length-1;t>=0;t--){ if(s.pts[t]==null)continue; dn+=`${X(t)},${Y(s.pts[t]-s.band[t])} `; }
      el('polygon',{points:up+dn,fill:s.color,opacity:.13},svg);
    }
    let pl='';
    for(let t=0;t<s.pts.length;t++){ if(s.pts[t]==null)continue; pl+=`${X(t)},${Y(s.pts[t])} `; }
    el('polyline',{points:pl,fill:'none',stroke:s.color,'stroke-width':s.wd||1.5,opacity:s.op??1,
      'stroke-dasharray':s.dash||'none'},svg);
  });
  (o.hlines||[]).forEach(hl=>{
    el('line',{x1:o.pad,y1:Y(hl.v),x2:w-o.pad,y2:Y(hl.v),stroke:hl.color,'stroke-width':1,
      'stroke-dasharray':'4 3',opacity:.75},svg);
    if(hl.label) el('text',{x:w-o.pad-2,y:Y(hl.v)-4,'text-anchor':'end',fill:hl.color,
      'font-family':'var(--mono)','font-size':'9.5px',text:hl.label},svg);
  });
  (o.marks||[]).forEach(m=>{
    el('line',{x1:X(m.t),y1:0,x2:X(m.t),y2:h,stroke:m.color,'stroke-width':1,opacity:.8},svg);
  });
  if (!o.noCursor) el('line',{x1:X(S.day),y1:0,x2:X(S.day),y2:h,stroke:'var(--bright)','stroke-width':1,opacity:.5},svg);
  return svg;
}

/* ---------- the rail ---------- */
function renderRail(){
  const a=S.sel, d=data(), rail=$('#rail'); rail.innerHTML='';
  const id='T'+String(a+1).padStart(2,'0'), f=Math.floor(a/6);
  const fd=d.fail_day[String(a)], on=d.onset[String(a)];
  const who=el('div',{class:'who'},rail);
  el('span',{class:'id',text:id+' · feeder '+FNAMES[f]},who);
  let tag='HEALTHY', cls='';
  if (fd!=null && S.day>=fd){tag='DAMAGE'; cls='bad';}
  else if (isFault(a) && on!=null && S.day>=+on){tag='FAULT DEVELOPING'; cls='hot';}
  el('span',{class:'tag '+cls,text:tag},who);

  // L1 perceive
  const cut=(arr)=>arr.map((v,t)=>t<=S.day?v:null);
  const sg=sigmaOf(a);
  const l1=el('div',{class:'level'},rail);
  l1.innerHTML='<div class="lv"><b>Level 1 · Perception</b> · hot-spot temperature against the model\'s expectation</div>';
  l1.appendChild(spark(430,96,[
    {pts:d.pred[a].map((v,t)=>t<=S.day?v:null),band:d.pred[a].map(()=>3*sg),color:'var(--steel)',wd:1,op:.9},
    {pts:cut(d.hotspot[a]),color:'var(--bright)',wd:1.3},
  ],{}));
  const t1=el('div',{class:'note'},l1);
  t1.innerHTML=`today ${fmt1(d.hotspot[a][S.day])} °C · model expected ${fmt1(d.pred[a][S.day])} ±${fmt1(3*sg)} · load ${d.K[a][S.day]} pu`;

  // L2 comprehend
  const b=BUDGETS[S.bIdx];
  const l2=el('div',{class:'level'},rail);
  l2.innerHTML='<div class="lv"><b>Level 2 · Comprehension</b> · standardised residual, accumulated evidence, uncertainty</div>';
  const l2hi=Math.max(8,b+2);
  l2.appendChild(spark(430,96,[
    {pts:cut(d.cus[a].map(v=>Math.min(v/3,l2hi))),color:'var(--sodium)',wd:1,op:.85},
    {pts:cut(d.z[a]),color:'var(--phosphor)',wd:1.2},
  ],{hlines:[{v:b,color:'var(--signal)',label:'alarm z='+b}],lo:-3,hi:l2hi}));
  const zt=d.z[a][S.day], ct=d.cus[a][S.day], uh=d.unc_hi[a][S.day];
  const t2=el('div',{class:'note'},l2);
  t2.innerHTML = ct>3*b
    ? `z = ${zt.toFixed(1)} · <em>accumulated evidence well past the ticket threshold</em> (requires ${(3*b).toFixed(0)})`
    : `z = ${zt.toFixed(1)} · accumulated evidence ${ct.toFixed(0)} of the ${(3*b).toFixed(0)} required`+(zt>b?' · <em>above threshold</em>':'');
  const env=el('div',{class:'envelope'+(uh?' show':'')},l2);
  env.textContent='Outside the training envelope: ensemble disagreement is high, so the vote requirement is raised';

  // L3 project
  const l3=el('div',{class:'level'},rail);
  l3.innerHTML='<div class="lv"><b>Level 3 · Projection</b> · expected trajectory</div>';
  const wb=RS.cables.weibull;
  const ticketDays=(ticketsFor(currentArch().key)[String(a)]||[]).filter(t=>t<=S.day);
  if (fd!=null && S.day>=fd){
    l3.appendChild(spark(430,96,[{pts:cut(d.z[a]),color:'var(--signal)',wd:1.2}],
      {hlines:[{v:10,color:'var(--signal)',label:'damage'}],lo:-2,hi:12,marks:[{t:fd,color:'var(--signal)'}]}));
    const t3=el('div',{class:'note'},l3);
    t3.innerHTML = ticketDays.length && ticketDays[0]<fd
      ? `damage reached on day ${fd}. This architecture's first ticket came on day ${ticketDays[0]}, <em>${fd-ticketDays[0]} days in advance</em>.`
      : `damage reached on day ${fd} <em>without a ticket from this architecture</em>. Others ticketed it earlier; compare above.`;
  } else if (ticketDays.length && zt>1){
    // straight-line extrapolation of recent z toward damage, with a doubt fan
    const t0=Math.max(TRAIN,S.day-20);
    const slope=(d.z[a][S.day]-d.z[a][t0])/Math.max(1,S.day-t0);
    const zdmg=10;
    const eta = slope>0.01 ? Math.round((zdmg-zt)/slope) : null;
    const proj=[],projHi=[],projLo=[];
    for(let t=0;t<N_DAYS;t++){
      if(t<S.day){proj.push(null);projHi.push(0);projLo.push(0);}
      else {const dz=slope*(t-S.day); proj.push(Math.min(zdmg+2,zt+dz)); projHi.push(dz*.5); projLo.push(dz*.5);}
    }
    l3.appendChild(spark(430,96,[
      {pts:cut(d.z[a]),color:'var(--phosphor)',wd:1.1,op:.6},
      {pts:proj,band:projHi,color:'var(--sodium)',wd:1.4,dash:'5 3'},
    ],{hlines:[{v:zdmg,color:'var(--signal)',label:'damage'}],lo:-2,hi:zdmg+2}));
    const t3=el('div',{class:'note'},l3);
    t3.innerHTML = eta && eta<200
      ? `a linear extrapolation reaches the damage threshold in roughly <em>${eta} days</em>. The band shows the uncertainty of that assumption.`
      : 'evidence is present, but the trend is too flat for a dated projection.';
  } else {
    const km=kMean(a), scl=wb.scale*Math.pow(km/0.7,-0.9);
    const pts=[],ages=[];
    for(let t=0;t<N_DAYS;t++){const age=60*t/(N_DAYS-1); ages.push(age); pts.push(Math.exp(-Math.pow(age/scl,wb.shape)));}
    const med=scl*Math.pow(Math.LN2,1/wb.shape);
    l3.appendChild(spark(430,96,[{pts:pts,color:'var(--steel)',wd:1.4}],
      {noRegime:true,noCursor:true,hlines:[{v:.5,color:'var(--faint)',label:'median '+fmt1(med)+' y'}],lo:0,hi:1}));
    const t3=el('div',{class:'note'},l3);
    t3.innerHTML=`no active evidence on ${id}. Cohort survival for its loading (K̄ ${km.toFixed(2)} pu): median life ${fmt1(med)} years, censoring-aware, from Study 3.`;
  }
}

/* ---------- verdict tiles ---------- */
function renderVerdicts(){
  const arch=currentArch(); const r=RS[S.scenario][rsLabel(arch)][bKey()];
  const v=$('#verdicts'); v.innerHTML='';
  const mk=(cls,k,n,s)=>{const t=el('div',{class:'tile '+cls},v);
    t.innerHTML=`<div class="k">${k}</div><div class="n">${n}</div><div class="s">${s}</div>`;};
  const det=r.detected, of=r.of;
  mk(det>=2.9?'good':det>=2?'mid':'bad','detection · mean of 5 runs',
    `${det.toFixed(1)}<small> of ${of}</small>`,
    det>=2.9 ? 'Every developing fault was ticketed before damage.' :
    det>=2 ? 'One fault is being missed; the diagram shows which.' :
    'Faults reach damage without a ticket.');
  mk(r.warning_days>=15?'good':'mid','median lead time before damage',
    `${fmt1(r.warning_days)}<small> days</small>`,
    r.warning_days>=15 ? 'Sufficient notice to plan an intervention.' :
    'Short notice: intervention becomes reactive.');
  const fa=r.fa_per_month;
  mk(fa<1?'good':fa<5?'mid':'bad','false alarms · fleet per month',
    fmt1(fa),
    fa<1 ? `Low enough for every ticket to receive attention. Precision ${(r.precision*100).toFixed(0)}%.` :
    fa<5 ? `Roughly one unnecessary call-out per week. Precision ${(r.precision*100).toFixed(0)}%.` :
    `At this rate the ticket log loses credibility. Precision ${(r.precision*100).toFixed(0)}%.`);
}

/* ---------- work orders ---------- */
function whyText(archKey,a,td){
  const d=data();
  if (archKey==='naive') return 'temperature above the commissioning limit';
  const z=d.z[a][td], uh=d.unc_hi[a][td];
  let s=`hot-spot ${z.toFixed(1)}σ over forecast`;
  if (archKey!=='single') s+=', methods agree';
  if (archKey==='seq'||archKey==='recal') s+=`, evidence sustained`;
  if (uh) s+=' · flagged: outside training envelope';
  return s;
}
function renderOrders(){
  const arch=currentArch(); const tix=ticketsFor(arch.key); const d=data();
  const rows=[];
  for (const aStr in tix) for (const td of tix[aStr]){
    if (td<TRAIN) continue;
    const a=+aStr;
    const real=isFault(a) && td>=(+d.onset[aStr])-5;
    rows.push({td,a,real});
  }
  rows.sort((x,y)=>x.td-y.td);
  const list=$('#orderlist'); list.innerHTML='';
  let shown=0, realN=0;
  rows.forEach((r,i)=>{
    const row=el('div',{class:'orow'+(r.td>S.day?' future':'')},list);
    row.innerHTML=`<span class="d">day ${String(r.td).padStart(3,'0')}</span>`+
      `<span class="a">T${String(r.a+1).padStart(2,'0')} · ${FNAMES[Math.floor(r.a/6)]}</span>`+
      `<span class="why">${whyText(arch.key,r.a,r.td)}</span>`+
      `<span class="h ${r.real?'real':'false'}">${r.real?'true positive':'false alarm'}</span>`;
    if (r.td<=S.day){shown++; if(r.real)realN++;}
  });
  if (!rows.length) list.innerHTML='<div class="empty-orders">No tickets at this threshold. Note the amber assets on the diagram that remain unreported.</div>';
  $('#ordercount').textContent=`${shown} tickets by day ${S.day} · ${realN} on genuinely degrading assets · future rows dimmed`;
}

/* ---------- studies ---------- */
function barChart(rows, maxFA, maxW){
  const w=980, rh=44, h=rows.length*rh+30;
  const svg=el('svg',{viewBox:`0 0 ${w} ${h}`});
  const lw=250, half=(w-lw-96)/2;
  el('text',{x:lw+half/2,y:14,'text-anchor':'middle',class:'axis',text:'false alarms · fleet per month'},svg);
  el('text',{x:lw+half+56+half/2,y:14,'text-anchor':'middle',class:'axis',text:'median lead time · days before damage'},svg);
  rows.forEach((r,i)=>{
    const y=30+i*rh;
    el('text',{x:lw-10,y:y+17,'text-anchor':'end',class:'barlbl',text:r.label},svg);
    if (r.note) el('text',{x:lw-10,y:y+30,'text-anchor':'end',class:'axis',fill:'var(--faint)',text:r.note},svg);
    const bw=Math.max(2,half*r.fa/maxFA);
    el('rect',{x:lw,y:y+6,width:bw,height:14,rx:2,fill:r.faColor||'var(--sodium)'},svg);
    el('text',{x:lw+bw+7,y:y+17,class:'barnum',text:fmt1(r.fa)},svg);
    const bw2=Math.max(2,half*r.warn/maxW);
    el('rect',{x:lw+half+56,y:y+6,width:bw2,height:14,rx:2,fill:'var(--phosphor)',opacity:r.dim?0.45:1},svg);
    const missed=r.det!=null && r.det<2.9;
    el('text',{x:lw+half+56+bw2+7,y:y+17,class:'barnum',text:fmt1(r.warn)},svg);
    if (missed) el('text',{x:lw+half+56,y:y+33,class:'axis',fill:'var(--signal)',
      text:'detects only '+r.det.toFixed(1)+' of 3 faults'},svg);
  });
  return svg;
}
function renderStudies(){
  const get=(sc,lbl)=>RS[sc][lbl][String(META.default_budget)];
  // study 1
  const s1=$('#study1'); s1.innerHTML='<h3>Study 1 · The contribution of each architectural stage</h3>'+
    '<p class="q">Stationary conditions, five seeded runs, threshold z='+META.default_budget+'. Each architecture adds one mechanism from the centre\'s publications to the previous one.</p>';
  const order=[['static limit',''],['forecaster residual','learned expectation model'],['multi-method vote','agreement before alarm'],
    ['vote + uncertainty weighting','vote raised under uncertainty'],['full sequential gate','sustained evidence required']];
  let rows=order.map(([lbl,note])=>{const r=get('stationary',lbl);
    return {label:lbl,note,fa:r.fa_per_month,warn:r.warning_days,det:r.detected};});
  const maxFA=Math.max(...rows.map(r=>r.fa))*1.15;
  s1.appendChild(barChart(rows,maxFA,30));
  const g1=get('stationary','full sequential gate'), n1=get('stationary','static limit'), f1=get('stationary','forecaster residual');
  el('div',{class:'verdict'},s1).innerHTML=
    `The static limit misses one fault and still produces ${fmt1(n1.fa_per_month)} false call-outs a month. The full
     sequential gate detects <b>3 of 3 at ${fmt1(g1.fa_per_month)} false call-outs a month</b> (precision
     ${(g1.precision*100).toFixed(0)}%), at a cost of ${fmt1(f1.warning_days-g1.warning_days)} days of lead time. This
     trade between alarm load and lead time is the central design decision; the threshold slider above moves along it.`;
  // study 2
  const s2=$('#study2'); s2.innerHTML='<h3>Study 2 · Behaviour under a regime change</h3>'+
    '<p class="q">The same fleet, with a heat wave arriving on day 180 and electric-vehicle charging growing thereafter. The training data no longer describes the operating conditions.</p>';
  const order2=order.concat([['sequential gate + rolling recalibration','retrained weekly']]);
  let rows2=order2.map(([lbl,note])=>{const r=get('shift',lbl);
    return {label:lbl===('sequential gate + rolling recalibration')?'gate + weekly retraining':lbl,note,fa:r.fa_per_month,warn:r.warning_days,det:r.detected,
      faColor: lbl.includes('recalibration')?'var(--signal)':undefined};});
  s2.appendChild(barChart(rows2,Math.max(...rows2.map(r=>r.fa))*1.15,30));
  const n2=get('shift','static limit'), g2=get('shift','full sequential gate'), r2=get('shift','sequential gate + rolling recalibration');
  el('div',{class:'verdict sting'},s2).innerHTML=
    `The static limit degrades to ${fmt1(n2.fa_per_month)} false call-outs a month at ${(n2.precision*100).toFixed(0)}%
     precision. The fixed sequential gate holds at ${fmt1(g2.fa_per_month)}. Weekly retraining appears strongest of all,
     ${fmt1(r2.fa_per_month)} false call-outs at ${(r2.precision*100).toFixed(0)}% precision, and it is the only
     architecture that <b>misses the slow fault, detecting ${r2.detected.toFixed(1)} of 3</b>. T22's degradation advanced
     more slowly than the retraining cadence, so each refit absorbed part of it into the estimated normal state, and
     each redeployment restarted the evidence accumulator. In the run drawn above, the fixed gate tickets T22 on day
     197, 43 days before damage; the retrained detector never does. The architecture with the cleanest record is the
     one that cannot see the slowest fault.`;
  // study 3
  const c=RS.cables;
  const s3=$('#study3'); s3.innerHTML='<h3>Study 3 · Cable replacement under censored lifetimes</h3>'+
    `<p class="q">${c.n} MV cable sections observed for 25 years; ${c.n_failures} failures observed, ${c.n_censored}
     lifetimes censored by the observation window or preventive replacement. The estimate of remaining life depends
     on how the censored records are treated.</p>`;
  const w=980,h=230,pad=46;
  const svg=el('svg',{viewBox:`0 0 ${w} ${h}`},s3);
  const X=t=>pad+(w-2*pad)*t/60, Y=s=>18+(h-58)*(1-s);
  el('line',{x1:pad,y1:Y(0),x2:w-pad,y2:Y(0),stroke:'var(--line2)'},svg);
  [0,10,20,30,40,50,60].forEach(t=>el('text',{x:X(t),y:h-18,'text-anchor':'middle',class:'axis',text:t+'y'},svg));
  el('text',{x:pad,y:12,class:'axis',text:'share of cohort still alive'},svg);
  let km='M'+X(0)+','+Y(1); let last=1;
  c.km_curve.forEach(([t,s])=>{ km+=` L${X(t)},${Y(last)} L${X(t)},${Y(s)}`; last=s; });
  el('path',{d:km,fill:'none',stroke:'var(--phosphor)','stroke-width':2},svg);
  let wbp='';
  for(let t=0;t<=60;t+=1){const s=Math.exp(-Math.pow(t/c.weibull.scale,c.weibull.shape)); wbp+=`${X(t)},${Y(s)} `;}
  el('polyline',{points:wbp,fill:'none',stroke:'var(--steel)','stroke-width':1.4,'stroke-dasharray':'6 4'},svg);
  el('line',{x1:X(c.naive_mean),y1:Y(1),x2:X(c.naive_mean),y2:Y(0),stroke:'var(--signal)','stroke-width':1.6,'stroke-dasharray':'3 3'},svg);
  el('text',{x:X(c.naive_mean)+6,y:Y(.92),class:'barnum',fill:'var(--signal)',text:'uncorrected mean: '+fmt1(c.naive_mean)+' y'},svg);
  el('text',{x:X(c.km_median)+6,y:Y(.5)-6,class:'barnum',fill:'var(--phosphor)',text:'Kaplan-Meier median '+fmt1(c.km_median)+' y'},svg);
  el('text',{x:X(2),y:Y(.13),class:'axis',text:'Weibull fit: shape '+c.weibull.shape.toFixed(1)+', scale '+c.weibull.scale.toFixed(0)+' y'},svg);
  const ro=c.rule_oldest, rh2=c.rule_hazard;
  el('div',{class:'verdict'},s3).innerHTML=
    `Averaging only the observed failures gives a fleet lifetime of ${fmt1(c.naive_mean)} years; the censoring-aware
     estimate puts the median at <b>${fmt1(c.km_median)} years</b>. The bias carries an operational cost: with the same
     budget of ${c.budget} replacements over ten years, replacing the oldest sections first prevents ${ro.prevented}
     failures and replaces ${ro.wasted} sections that would have survived, while ranking by fitted Weibull hazard (age
     and loading jointly) prevents <b>${rh2.prevented} failures</b> and replaces ${rh2.wasted} unnecessarily. The same
     budget, ${ro.in_service_failures-rh2.in_service_failures} fewer in-service failures.`;
}

/* ---------- provenance ---------- */
const PROV=[
 {what:'Residual detectors trained on healthy operation only, because faults are rare and unlabelled',
  src:[['sh','Shaker et al. 2026','Multi-method fault detection considering uncertainty through MC dropout, Energy & Buildings'],
       ['fd','the premise','fault detection without fault labels: model normal, alarm on departure']]},
 {what:'Several different residual generators vote before anything is called a fault',
  src:[['sh','Shaker et al. 2026','the enhanced-voting architecture this demo\'s vote stage copies'],
       ['fd','ensemble tradition','diverse errors cancel; agreement is evidence']]},
 {what:'When the ensemble disagrees, the vote requirement is raised, not silenced',
  src:[['sh','Shaker group 2026','probabilistic overload alarms: epistemic vs aleatoric uncertainty, Sust. Energy Grids & Netw.'],
       ['fd','Gal & Ghahramani 2016','MC dropout as Bayesian approximation; bootstrap-ensemble disagreement stands in for it here']]},
 {what:'A ticket requires sustained evidence: sequential statistics before the queue',
  src:[['sh','Shaker group 2024-2025','alarm significance (IEEE TII 2024), TFT alarm forecasting at a Danish DSO'],
       ['fd','Page 1954; Roberts 1959','CUSUM and EWMA, the sequential-evidence canon']]},
 {what:'The alarm budget is set in crew call-outs per month, and everything else follows from it',
  src:[['sh','Shaker group 2025','consequence-aware prescriptive maintenance, IEEE Trans. Smart Grid: crews are finite'],
       ['fd','alarm fatigue','a queue nobody reads is a detector nobody has']]},
 {what:'Every ticket carries its reason in plain words',
  src:[['sh','Shaker group 2025','XAI for energy maintenance review, Renew. Sust. Energy Rev.: operators cannot act on black boxes'],
       ['fd','Lundberg & Lee 2017; Ribeiro et al. 2016','SHAP and LIME, the attribution canon']]},
 {what:'Cable life is read through censoring-aware survival curves, not averages of observed deaths',
  src:[['sh','Shaker group 2026','survival models for PdM and RUL in smart energy networks, Sensors review'],
       ['sh','Mortensen & Shaker 2025','neural Weibull proportional hazards for cable replacement under data deficiency, IEEE Access'],
       ['fd','Kaplan & Meier 1958; Cox 1972; Weibull 1951','with Ishwaran 2008 and Katzman 2018 as the learned extensions']]},
 {what:'The whole page is a small digital twin: a simulated network used to interrogate operating decisions',
  src:[['bj','Jørgensen group 2024','digital-twin framework for simulating DER in distribution grids, Energies'],
       ['bj','Jørgensen group 2021','the building-level twin the framework grew from, Energy Informatics']]},
 {what:'The asset panel is organised as perceive / comprehend / project',
  src:[['bj','Jørgensen & Ma 2026','Infostructure: situation awareness in future power system control rooms, Energies'],
       ['fd','Endsley 1995','the three-level model of situation awareness']]},
 {what:'Detection stages are sequential and non-compensable: no later score can buy back a failed gate',
  src:[['bj','Ma, Cong & Jørgensen 2026','deployment feasibility as a layered construct: sequential gates, not compensatory scoring, Energies'],
       ['fd','this page\'s reading','a ticket must pass vote, uncertainty and persistence in order']]},
 {what:'Retraining cadence is treated as a first-class design variable, with a cost',
  src:[['bj','Jørgensen group 2026','MLOps platform capability mapping for energy forecasting, Information'],
       ['fd','Gama et al. 2014','concept drift: adapt too fast and slow faults become the new normal']]},
 {what:'District heating is the stated next domain for exactly this machinery',
  src:[['sh','Shaker group 2023-2026','anomaly indexing for DH decision support; DH predictive maintenance review; DH asset management pathways'],
       ['fd','kept out of this page','one grid, readable; the methods carry over']]},
];
function renderProv(){
  const p=$('#provcards');
  PROV.forEach(c=>{
    const card=el('div',{class:'card'},p);
    card.innerHTML='<div class="what">'+c.what+'</div>'+
      c.src.map(([cls,who,what])=>`<div class="src"><span class="who-tag ${cls}">${cls==='sh'?'Shaker':cls==='bj'?'Jørgensen':'foundation'}</span><b>${who}</b> · ${what}</div>`).join('');
  });
}

/* ---------- controls ---------- */
function renderArchChips(){
  const box=$('#arch-chips'); box.innerHTML='';
  ARCHS.forEach(a=>{
    if (!archAvailable(a)) return;
    const c=el('button',{class:'chip'+(currentArch().key===a.key?' on':''),text:a.label},box);
    c.addEventListener('click',()=>{S.arch=a.key; refresh();});
  });
}
function refresh(){
  document.querySelectorAll('#scenario-chips .chip').forEach(x=>
    x.classList.toggle('on', x.dataset.scenario===S.scenario));
  const sc=$('#scrubber');
  sc.style.background = S.scenario==='shift'
    ? `linear-gradient(90deg, var(--line) 0%, var(--line) ${REGIME/(N_DAYS-1)*100}%, var(--sodium) ${REGIME/(N_DAYS-1)*100}%, var(--sodium) ${REGIME/(N_DAYS-1)*100+0.8}%, var(--line) ${REGIME/(N_DAYS-1)*100+0.8}%)`
    : 'var(--line)';
  renderArchChips();
  $('#budget-val').textContent='ticket at z > '+bKey();
  paintMap(); renderRail(); renderVerdicts(); renderOrders();
}
function setDay(d){ S.day=Math.max(0,Math.min(N_DAYS-1,d)); $('#scrubber').value=S.day; paintMap(); renderRail(); renderOrders(); }

let rafId=null,lastT=0,acc=0;
function tick(ts){
  if(!S.playing) return;
  if(lastT) { acc+=(ts-lastT)/1000*16; if(acc>=1){ setDay(S.day+Math.floor(acc)); acc%=1; } }
  lastT=ts;
  if (S.day>=N_DAYS-1){ S.playing=false; $('#playbtn').innerHTML='&#9654;'; return; }
  rafId=requestAnimationFrame(tick);
}
$('#playbtn').addEventListener('click',()=>{
  S.playing=!S.playing; lastT=0;
  $('#playbtn').innerHTML=S.playing?'&#10073;&#10073;':'&#9654;';
  if(S.playing){ if(S.day>=N_DAYS-1) setDay(0); rafId=requestAnimationFrame(tick); }
});
$('#scrubber').addEventListener('input',e=>{ S.playing=false; $('#playbtn').innerHTML='&#9654;'; setDay(+e.target.value); });
document.querySelectorAll('#scenario-chips .chip').forEach(c=>{
  c.addEventListener('click',()=>{
    document.querySelectorAll('#scenario-chips .chip').forEach(x=>x.classList.remove('on'));
    c.classList.add('on');
    S.scenario=c.dataset.scenario;
    refresh();
  });
});

/* budget slider: index 0 (left, strict) .. 5 (right, loose); BUDGETS is [3.6..2.0] strict->loose.
   slider value v: 0=strict; map S.bIdx = v. Fix mapping: */
$('#budget').max=String(BUDGETS.length-1);
$('#budget').value='2';
$('#budget').addEventListener('input',e=>{ S.bIdx=+e.target.value; refresh(); },{once:false});

buildMap();
renderProv();
renderStudies();
setDay(N_DAYS-1);
refresh();
})();
