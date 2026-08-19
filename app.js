'use strict';
/* Alarm quality under regime change.
   Port of the offline Python study (study/) to the browser. Same model,
   same gates, same scoring, smaller fleet so it runs instantly. */

const HPD = 24, DAYS = 300, N = DAYS * HPD, ASSETS = 60, WARMUP = 90 * HPD;
const SHIFT_AT = Math.floor(N * 0.55);

/* ---------- seeded rng ---------- */
function rng(seed){ let s = seed >>> 0;
  return () => { s ^= s<<13; s>>>=0; s ^= s>>17; s ^= s<<5; s>>>=0; return s/4294967296; }; }
function gauss(r){ let u=0,v=0; while(!u) u=r(); while(!v) v=r();
  return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }

/* ---------- systems ---------- */
function dailyShape(i, peak1, peak2){
  const h = i % HPD, dow = Math.floor(i/HPD)%7;
  const base = 0.55 + 0.45*Math.exp(-0.5*Math.pow((h-peak1)/3,2))
                    + 0.45*Math.exp(-0.5*Math.pow((h-peak2)/2.5,2));
  return base * (dow>=5 ? 0.80 : 1.0);
}

function build(sys, shift, sev, noiseMul, seed){
  const r = rng(seed), faulty = [], fstart = [];
  const y = [], X0 = [], X1 = [];
  const ambient = new Float64Array(N);
  for(let i=0;i<N;i++)
    ambient[i] = 9 + 7*Math.sin(2*Math.PI*(i-2000)/(365*HPD))
                   + 3*Math.sin(2*Math.PI*((i%HPD)-4)/HPD);

  for(let a=0;a<ASSETS;a++){
    const isF = r() < 0.25;
    faulty.push(isF);
    fstart.push(isF ? Math.floor(N*0.45 + r()*(N*0.35)) : -1);
    const yy = new Float64Array(N), f0 = new Float64Array(N), f1 = new Float64Array(N);

    if(sys === 'tx'){
      const rated = 0.45 + r()*0.30, tau = 3;
      let th = 0;
      for(let i=0;i<N;i++){
        let ld = rated*dailyShape(i,8,19) + gauss(r)*0.035;
        if(shift && i>=SHIFT_AT) ld *= 1.28;
        ld = Math.min(1.35, Math.max(0.02, ld));
        let amb = ambient[i] + (shift && i>=SHIFT_AT ? 6 : 0);
        let k = 55;
        if(isF && i>=fstart[a])
          k *= 1 + 0.16*sev*Math.min(1,(i-fstart[a])/(30*HPD));
        th += (k*ld*ld - th)/tau;
        yy[i] = th + gauss(r)*0.9*noiseMul;
        f0[i] = ld; f1[i] = amb/30;
      }
    } else {
      const size = 0.5 + r()*0.7, eff0 = 0.72 + r()*0.08;
      for(let i=0;i<N;i++){
        const amb = ambient[i];
        let sup = 75 - 0.45*amb + (shift && i>=SHIFT_AT ? 6 : 0);
        let dem = size*dailyShape(i,7,20)*Math.min(1.6,Math.max(0.15,(18-amb)/12));
        if(shift && i>=SHIFT_AT) dem *= 1.28;
        dem = Math.max(0.02, dem + gauss(r)*0.03);
        let eff = eff0;
        if(isF && i>=fstart[a])
          eff *= 1 - 0.085*sev*Math.min(1,(i-fstart[a])/(45*HPD));
        yy[i] = sup - eff*(sup-28) + 4.5*dem + gauss(r)*0.55*noiseMul;
        f0[i] = dem; f1[i] = sup/80;
      }
    }
    y.push(yy); X0.push(f0); X1.push(f1);
  }
  return {y, X0, X1, faulty, fstart};
}

/* ---------- bootstrapped quadratic ridge ensemble ---------- */
function phi(a,b){ return [1,a,b,a*a,b*b,a*b]; }

function solve(A,b){                       // gaussian elimination, 6x6
  const n=b.length, M=A.map((row,i)=>row.concat([b[i]]));
  for(let c=0;c<n;c++){
    let p=c; for(let i=c+1;i<n;i++) if(Math.abs(M[i][c])>Math.abs(M[p][c])) p=i;
    [M[c],M[p]]=[M[p],M[c]];
    if(Math.abs(M[c][c])<1e-12) continue;
    for(let i=0;i<n;i++){ if(i===c) continue;
      const f=M[i][c]/M[c][c];
      for(let j=c;j<=n;j++) M[i][j]-=f*M[c][j]; }
  }
  return M.map((row,i)=> Math.abs(row[i])<1e-12 ? 0 : row[n]/row[i]);
}

function fitEnsemble(d, from, to, seed, K=8){
  const r = rng(seed), W=[];
  const healthy = []; for(let a=0;a<ASSETS;a++) if(!d.faulty[a]) healthy.push(a);
  const pool=[];
  for(const a of healthy) for(let i=from;i<to;i+=3) pool.push([d.X0[a][i], d.X1[a][i], d.y[a][i]]);
  const n=pool.length;
  for(let k=0;k<K;k++){
    const A=Array.from({length:6},()=>new Float64Array(6)), bb=new Float64Array(6);
    for(let s=0;s<n;s++){
      const p=pool[Math.floor(r()*n)], f=phi(p[0],p[1]);
      for(let i=0;i<6;i++){ for(let j=0;j<6;j++) A[i][j]+=f[i]*f[j]; bb[i]+=f[i]*p[2]; }
    }
    for(let i=0;i<6;i++) A[i][i] += 1e-3*n;
    W.push(solve(A.map(rw=>Array.from(rw)), Array.from(bb)));
  }
  return W;
}

function predict(W,a,b){
  const f=phi(a,b); let m=0, m2=0;
  for(const w of W){ let v=0; for(let i=0;i<6;i++) v+=w[i]*f[i]; m+=v; m2+=v*v; }
  const mean=m/W.length;
  return [mean, Math.sqrt(Math.max(0, m2/W.length - mean*mean))];
}

function residuals(d, W, from){
  const R=[], U=[];
  for(let a=0;a<ASSETS;a++){
    const rr=new Float64Array(N), uu=new Float64Array(N);
    for(let i=0;i<N;i++){
      const [m,s]=predict(W,d.X0[a][i],d.X1[a][i]);
      rr[i]=d.y[a][i]-m; uu[i]=s;
    }
    R.push(rr); U.push(uu);
  }
  return [R,U];
}

/* ---------- detection ---------- */
function stats(R){
  const Z=[],E=[],C=[];
  for(let a=0;a<ASSETS;a++){
    let m=0; for(let i=0;i<WARMUP;i++) m+=R[a][i]; m/=WARMUP;
    let v=0; for(let i=0;i<WARMUP;i++) v+=(R[a][i]-m)**2; const sd=Math.sqrt(v/WARMUP)+1e-9;
    const z=new Float64Array(N), e=new Float64Array(N), c=new Float64Array(N);
    let acc=0, cus=0; const lam=0.02, k=Math.sqrt(lam/(2-lam));
    for(let i=0;i<N;i++){
      z[i]=(R[a][i]-m)/sd;
      acc=lam*z[i]+(1-lam)*acc; e[i]=acc/k;
      cus=Math.max(0,cus+z[i]-0.5); c[i]=cus;
    }
    Z.push(z);E.push(e);C.push(c);
  }
  return [Z,E,C];
}

function quantile(arrs, q, upto){
  const v=[]; for(const a of arrs) for(let i=0;i<upto;i+=7) v.push(a[i]);
  v.sort((x,y)=>x-y);
  return v[Math.min(v.length-1, Math.floor(q*v.length))];
}

function detect(R,U,cfg){
  const [Z,E,C]=stats(R);
  const tz=quantile(Z,1-2e-4,WARMUP), te=quantile(E,1-2e-4,WARMUP), tc=quantile(C,1-2e-4,WARMUP);
  const need = cfg.vote, pers = cfg.persistence;
  const fired=[];
  for(let a=0;a<ASSETS;a++){
    let ucut=Infinity;
    if(cfg.unc){ const one=[U[a]]; ucut=quantile(one,0.98,WARMUP); }
    const f=new Uint8Array(N); let run=0;
    for(let i=0;i<N;i++){
      let v=(Z[a][i]>tz)+(E[a][i]>te)+(C[a][i]>tc);
      let ok = v>=need;
      if(ok && cfg.unc && U[a][i]>ucut) ok=false;
      run = ok ? run+1 : 0;
      f[i] = (i>=WARMUP && run>=pers) ? 1 : 0;
    }
    fired.push(f);
  }
  return fired;
}

function driftOnset(U){
  const m=new Float64Array(N);
  for(let i=0;i<N;i++){ let s=0; for(let a=0;a<ASSETS;a++) s+=U[a][i]; m[i]=s/ASSETS; }
  const cut=quantile([m],0.99,WARMUP);
  let run=0;
  for(let i=WARMUP;i<N;i++){ run = m[i]>cut ? run+1 : 0; if(run>=3*HPD) return i; }
  return null;
}

function score(fired,d){
  const REF=7*HPD;
  let tickets=0, healthyHours=0, detected=0; const delays=[];
  const marks=[];
  for(let a=0;a<ASSETS;a++){
    const end = d.faulty[a] ? d.fstart[a] : N;
    let next=WARMUP;
    for(let i=WARMUP;i<end;i++) if(fired[a][i] && i>=next){ tickets++; next=i+REF; marks.push([i,0]); }
    healthyHours += Math.max(0,end-WARMUP);
    if(d.faulty[a]){
      let first=-1;
      for(let i=d.fstart[a];i<N;i++) if(fired[a][i]){ first=i; break; }
      if(first>=0){ detected++; delays.push((first-d.fstart[a])/24); marks.push([first,1]); }
    }
  }
  const nF=d.faulty.filter(Boolean).length;
  delays.sort((x,y)=>x-y);
  const months=healthyHours/(24*30);
  return {
    fa: months? tickets/months : NaN,
    rate: nF? detected/nF : NaN,
    delay: delays.length? delays[Math.floor(delays.length/2)] : NaN,
    prec: (detected+tickets)? detected/(detected+tickets) : NaN,
    marks
  };
}

const CFGS = {
  single:{vote:1,unc:false,persistence:1},
  vote:{vote:2,unc:false,persistence:1},
  votep:{vote:2,unc:false,persistence:6},
  voteu:{vote:2,unc:true,persistence:1},
  gate:{vote:2,unc:true,persistence:6},
  recal:{vote:2,unc:true,persistence:6,recal:true}
};

/* ---------- run ---------- */
let LAST=null;
function run(){
  const sys=document.getElementById('sys').value;
  const cfgName=document.getElementById('cfg').value;
  const shift=document.getElementById('shift').value==='1';
  const sev=parseFloat(document.getElementById('sev').value);
  const noise=parseFloat(document.getElementById('noise').value);
  const cfg=CFGS[cfgName];

  const d=build(sys,shift,sev,noise,12345);
  let W=fitEnsemble(d,0,WARMUP,7);
  let [R,U]=residuals(d,W,0);

  let fired;
  if(cfg.recal){
    const onset=driftOnset(U);
    if(onset!==null && onset+44*HPD<N){
      const settle=onset+14*HPD;
      const W2=fitEnsemble(d,settle,settle+30*HPD,77);
      for(let a=0;a<ASSETS;a++)
        for(let i=settle;i<N;i++){
          const [m,s]=predict(W2,d.X0[a][i],d.X1[a][i]);
          R[a][i]=d.y[a][i]-m; U[a][i]=s;
        }
      fired=detect(R,U,cfg);
      for(let a=0;a<ASSETS;a++)
        for(let i=settle;i<settle+30*HPD;i++) fired[a][i]=0;
    } else fired=detect(R,U,cfg);
  } else fired=detect(R,U,cfg);

  const s=score(fired,d);
  LAST={d,R,U,s,shift};
  showMetrics(s);
  draw(d,R,U,s,shift);
}

function showMetrics(s){
  const f=(v,n=2)=> isNaN(v)? '&ndash;' : v.toFixed(n);
  const pc = isNaN(s.prec)? '&ndash;' : Math.round(s.prec*100)+'%';
  document.getElementById('mx').innerHTML = `
    <div class="metric"><div class="k">Tickets / asset&middot;month</div><div class="v">${f(s.fa)}</div></div>
    <div class="metric"><div class="k">Detection rate</div><div class="v">${f(s.rate*100,0)}%</div></div>
    <div class="metric"><div class="k">Median delay</div><div class="v">${f(s.delay,1)}<span style="font-size:14px"> d</span></div></div>
    <div class="metric"><div class="k">Precision</div><div class="v">${pc}</div></div>`;
}

function draw(d,R,U,s,shift){
  const cv=document.getElementById('cv'), dpr=window.devicePixelRatio||1;
  const w=cv.clientWidth, h=300;
  cv.width=w*dpr; cv.height=h*dpr;
  const g=cv.getContext('2d'); g.scale(dpr,dpr);
  g.clearRect(0,0,w,h);

  const L=46,Rm=12,T=12,B=26, pw=w-L-Rm, ph=h-T-B;
  let lo, hi;
  const x=i=>L+pw*i/N;
  const yv2=v=>T+ph*(1-(v-lo)/(hi-lo));

  g.strokeStyle='#e6e1da'; g.lineWidth=1;
  for(let k=0;k<=4;k++){ const yy=T+ph*k/4; g.beginPath(); g.moveTo(L,yy); g.lineTo(L+pw,yy); g.stroke(); }
  g.fillStyle='#767d86'; g.font='11px DM Mono, monospace'; g.textAlign='right';
  for(let k=0;k<=4;k++){ const v=hi-(hi-lo)*k/4; g.fillText(v.toFixed(1),L-6,T+ph*k/4+4); }
  g.textAlign='center';
  for(let dday=0;dday<=DAYS;dday+=50) g.fillText('d'+dday, x(dday*HPD), h-8);

  // daily means: hourly traces are unreadable over 300 days
  const D = DAYS;
  const dayMean = (arr) => { const o=new Float64Array(D);
    for(let d=0;d<D;d++){ let s2=0; for(let k=0;k<HPD;k++) s2+=arr[d*HPD+k]; o[d]=s2/HPD; } return o; };
  const xd = d => L + pw*d/D;

  lo=1e9; hi=-1e9;
  const daily=[];
  for(let a=0;a<ASSETS;a++){ const dm=dayMean(R[a]); daily.push(dm);
    for(let d=0;d<D;d++){ if(dm[d]<lo)lo=dm[d]; if(dm[d]>hi)hi=dm[d]; } }
  const pad2=(hi-lo)*0.12||1; lo-=pad2; hi+=pad2;

  g.clearRect(0,0,w,h);
  g.strokeStyle='#e6e1da'; g.lineWidth=1;
  for(let k=0;k<=4;k++){ const yy2=T+ph*k/4; g.beginPath(); g.moveTo(L,yy2); g.lineTo(L+pw,yy2); g.stroke(); }
  g.fillStyle='#767d86'; g.font='11px DM Mono, monospace'; g.textAlign='right';
  for(let k=0;k<=4;k++){ const v=hi-(hi-lo)*k/4; g.fillText(v.toFixed(1),L-6,T+ph*k/4+4); }
  g.textAlign='center';
  for(let dday=0;dday<=DAYS;dday+=50) g.fillText('d'+dday, xd(dday), h-8);

  g.lineWidth=1.1;
  for(let a=0;a<ASSETS;a++){
    g.strokeStyle = d.faulty[a] ? 'rgba(162,59,40,.60)' : 'rgba(18,89,90,.28)';
    g.beginPath();
    for(let dd=0;dd<D;dd++){ const px=xd(dd), py=yv2(daily[a][dd]); dd?g.lineTo(px,py):g.moveTo(px,py); }
    g.stroke();
  }

  const um=dayMean((()=>{ const m=new Float64Array(N);
    for(let i=0;i<N;i++){ let s3=0; for(let a=0;a<ASSETS;a++) s3+=U[a][i]; m[i]=s3/ASSETS; } return m; })());
  let umax=0; for(let dd=0;dd<D;dd++) if(um[dd]>umax) umax=um[dd];
  g.strokeStyle='#b07d29'; g.lineWidth=1.6; g.beginPath();
  for(let dd=0;dd<D;dd++){ const py=T+ph*0.72+ph*(1-um[dd]/(umax||1))*0.26;
    dd?g.lineTo(xd(dd),py):g.moveTo(xd(dd),py); }
  g.stroke();

  if(shift){ g.strokeStyle='#3c434c'; g.setLineDash([5,4]); g.lineWidth=1.2;
    g.beginPath(); g.moveTo(x(SHIFT_AT),T); g.lineTo(x(SHIFT_AT),T+ph); g.stroke(); g.setLineDash([]); }

  for(const [i,kind] of s.marks){
    g.strokeStyle = kind ? '#2f6f4f' : '#a23b28'; g.lineWidth=1.2; g.globalAlpha=.85;
    g.beginPath(); g.moveTo(x(i),T+ph); g.lineTo(x(i),T+ph-(kind?16:9)); g.stroke();
  }
  g.globalAlpha=1;
  g.strokeStyle='#c9c2b8'; g.lineWidth=1; g.strokeRect(L,T,pw,ph);
}

/* ---------- offline study results ---------- */
const STUDY = {
 "single threshold":       [0.37,1.00,7.6,0.09,  2.81,1.00,0.0,0.01],
 "multi-method vote":      [0.02,1.00,10.8,0.66, 2.67,1.00,0.0,0.01],
 "vote + persistence":     [0.00,1.00,13.5,0.99, 2.65,1.00,0.0,0.01],
 "vote + uncertainty":     [0.02,1.00,11.8,0.70, 0.83,1.00,61.9,0.04],
 "full sequential gate":   [0.00,1.00,20.8,0.99, 0.53,1.00,79.5,0.07],
 "gate + recalibration":   [null,null,null,null, 0.05,0.41,34.1,0.27]
};
function fillExec(){
  const tb=document.querySelector('#exec tbody');
  for(const [name,v] of Object.entries(STUDY)){
    const tr=document.createElement('tr');
    if(name==='gate + recalibration') tr.className='hl';
    const cell=(x,i)=>{
      if(x===null) return '<td style="color:#b9b3aa">&ndash;</td>';
      if(i===3||i===7) return `<td class="${x>=0.5?'good':(x<=0.05?'bad':'')}">${Math.round(x*100)}%</td>`;
      if(i===1||i===5) return `<td>${Math.round(x*100)}%</td>`;
      return `<td>${x.toFixed(x<1?2:1)}</td>`;
    };
    tr.innerHTML = `<td>${name}</td>` + v.map(cell).join('');
    tb.appendChild(tr);
  }
}

document.getElementById('run').addEventListener('click',run);
document.getElementById('sev').addEventListener('input',e=>
  document.getElementById('sevv').textContent=(+e.target.value).toFixed(2)+'×');
document.getElementById('noise').addEventListener('input',e=>
  document.getElementById('noisev').textContent=(+e.target.value).toFixed(2)+'×');
document.getElementById('shift').addEventListener('change',e=>
  document.getElementById('shiftv').textContent=e.target.value==='1'?'on':'off');
fillExec();
run();
