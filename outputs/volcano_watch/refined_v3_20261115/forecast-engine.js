/* Browser implementation of the saved TERRA four-branch point forecaster.
 * Source engine/checkpoint are read-only. No weights are fitted here.
 * Six input slots, explicit validity, signed LOS mm, physical32×32 grid.
 */
(()=>{'use strict';
const N=1024,K=6,F=Math.fround,ok=x=>typeof x==='number'&&Number.isFinite(x),pause=()=>new Promise(r=>setTimeout(r,0));
const median=a=>{const b=Array.from(a).sort((x,y)=>x-y),n=b.length;return n%2?b[(n-1)/2]:(b[n/2-1]+b[n/2])/2;};
const med32=a=>F(median(a));
const flatten=a=>Array.isArray(a?.[0])?a.flat():Array.from(a||[]);
function eigen(a){
 const n=a.length,A=a.map(r=>r.slice()),V=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>+(i===j)));
 for(let it=0;it<300;it++){
  let p=0,q=1,max=0;for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)if(Math.abs(A[i][j])>max){max=Math.abs(A[i][j]);p=i;q=j;}
  const scale=Math.max(1,...A.map((r,i)=>Math.abs(r[i])));if(max<scale*1e-14)break;
  const phi=.5*Math.atan2(2*A[p][q],A[q][q]-A[p][p]),c=Math.cos(phi),s=Math.sin(phi),ap=A[p][p],aq=A[q][q],b=A[p][q];
  for(let i=0;i<n;i++)if(i!==p&&i!==q){const x=A[i][p],y=A[i][q];A[i][p]=A[p][i]=c*x-s*y;A[i][q]=A[q][i]=s*x+c*y;}
  A[p][p]=c*c*ap-2*s*c*b+s*s*aq;A[q][q]=s*s*ap+2*s*c*b+c*c*aq;A[p][q]=A[q][p]=0;
  for(let i=0;i<n;i++){const x=V[i][p],y=V[i][q];V[i][p]=c*x-s*y;V[i][q]=s*x+c*y;}
 }
 return Array.from({length:n},(_,j)=>({value:A[j][j],vector:V.map(r=>r[j])})).sort((a,b)=>b.value-a.value);
}
function reflected(i){while(i<0||i>=32)i=i<0?-i-1:63-i;return i;}
function uniform5(values){const tmp=new Float32Array(N),out=new Float32Array(N);for(let y=0;y<32;y++)for(let x=0;x<32;x++){let s=0;for(let k=-2;k<=2;k++)s+=values[reflected(y+k)*32+x];tmp[y*32+x]=s/5;}for(let y=0;y<32;y++)for(let x=0;x<32;x++){let s=0;for(let k=-2;k<=2;k++)s+=tmp[y*32+reflected(x+k)];out[y*32+x]=s/5;}return out;}
function input(arg){
 if(!arg||!Array.isArray(arg.maps)||arg.maps.length<6)throw Error('The four-branch model needs six explicit history slots. Predicted slots must be labelled by the caller.');
 if(!Array.isArray(arg.dates)||arg.dates.length!==arg.maps.length||(arg.valid_masks&&arg.valid_masks.length!==arg.maps.length))throw Error('Maps, dates and masks must describe the same history slots.');
 const h=Number(arg.horizon_days);if(!Number.isFinite(h)||h<1||h>180)throw Error('Forecast horizon must be1–180days beyond the last input.');
 const dates=arg.dates?.slice(-6),times=dates?.map(d=>Date.parse(d)/86400000);
 if(!times||times.length!==6||times.some(x=>!Number.isFinite(x))||times.some((x,i)=>i&&x<=times[i-1]))throw Error('Six strictly increasing valid UTC dates are required.');
 const md=arg.metadata||{},sp=md.spacing_km;if(!Array.isArray(sp)||sp.length!==2||sp.some(v=>!Number.isFinite(v)||v<=0))throw Error('Physical row/column spacing_km is required.');
 if(typeof md.geometry_available!=='boolean'||!['aoi_block_mean32','native_center32'].includes(md.grid_convention))throw Error('Explicit geometry availability and physical grid convention are required.');
 if(md.units&&md.units!=='mm')throw Error('Input displacement must be signed LOS millimetres.');
 const los=md.los_vector||[0,0,0];if(los.length!==3||los.some(x=>!Number.isFinite(x))||(md.geometry_available&&!md.los_vector))throw Error('Known geometry needs a verified three-component LOS vector.');
 const f=[],valid=[],raw=[];let last=new Float32Array(N),seen=new Uint8Array(N);
 arg.maps.slice(-6).forEach((map,k)=>{const arr=flatten(map),mask=arg.valid_masks?flatten(arg.valid_masks.slice(-6)[k]):null;if(arr.length!==N||(mask&&mask.length!==N))throw Error('Each map and mask must have1024cells.');const values=new Float32Array(N),v=new Uint8Array(N),r=new Float32Array(N);r.fill(NaN);for(let i=0;i<N;i++){if(ok(arr[i])&&(!mask||mask[i]===true||mask[i]===1)){r[i]=F(arr[i]);values[i]=r[i];v[i]=1;seen[i]=1;}else values[i]=last[i];}raw.push(r);f.push(values);valid.push(v);last=values;});
 if(!seen.some(Boolean))throw Error('No valid historical displacement pixels.');
 const diff=[];for(let i=0;i<N;i++)if(valid[0][i]&&valid[5][i])diff.push(Math.abs(F(raw[5][i]-raw[0][i])));diff.sort((a,b)=>a-b);
 let scale=1;if(diff.length){const pos=(diff.length-1)*.75,lo=Math.floor(pos);scale=Math.max(diff[lo]+(diff[Math.ceil(pos)]-diff[lo])*(pos-lo),1);}
 return {h,times,dates,md,sp,los,f,valid,raw,last,seen,scale};
}
function smooth(p,half,i){let sum=0,weight=0;for(let k=0;k<K;k++)if(p.valid[k][i]){const w=Math.exp(-Math.LN2*(p.times[5]-p.times[k])/half);sum+=p.raw[k][i]*w;weight+=w;}return weight?sum/weight:0;}
function common(p,model){
 const temporal=new Float32Array(N),quality=new Float32Array(N),spatial=new Float32Array(N),z=Array.from({length:K},()=>new Float64Array(N));
 const b=model.spatial_basis,R=b.length,coeff=Array.from({length:K},()=>new Float64Array(R));
 for(let i=0;i<N;i++){
  temporal[i]=smooth(p,30,i);const col=p.f.map(row=>row[i]),med=med32(col),mad=F(Math.max(F(med32(col.map(v=>Math.abs(F(v-med))))*1.4826),1/3));let sw=0,sd=0;
  const lower=F(med-F(3*mad)),upper=F(med+F(3*mad));for(let k=0;k<K;k++){z[k][i]=F(p.f[k][i]-p.last[i])/p.scale;if(p.valid[k][i]){const w=Math.exp(-Math.LN2*(p.times[5]-p.times[k])/30);sd+=Math.min(upper,Math.max(lower,p.f[k][i]))*w;sw+=w;}}
  quality[i]=sw?sd/sw:0;
 }
 for(let k=0;k<K;k++)for(let r=0;r<R;r++){let s=0;for(let i=0;i<N;i++)s+=z[k][i]*b[r][i];coeff[k][r]=s;}
 const tt=p.times.map(t=>(t-p.times[5])/90),sx=tt.reduce((a,b)=>a+b,0),sxx=tt.reduce((a,b)=>a+b*b,0)+4,det=K*sxx-sx*sx;
 const forecast=new Float64Array(R);for(let r=0;r<R;r++){let sy=0,sxy=0;for(let k=0;k<K;k++){sy+=coeff[k][r];sxy+=tt[k]*coeff[k][r];}const intercept=(sxx*sy-sx*sxy)/det,slope=(K*sxy-sx*sy)/det;forecast[r]=intercept+slope*p.h/90;}
 for(let i=0;i<N;i++){let s=0;for(let r=0;r<R;r++)s+=forecast[r]*b[r][i];spatial[i]=s*p.scale+p.last[i];}
 return {temporal,quality,spatial};
}
function contextFeatures(p){
 const z=Array.from({length:K},()=>new Float64Array(N)),means=new Float64Array(N);for(let i=0;i<N;i++){let s=0;for(let k=0;k<K;k++)s+=p.f[k][i];means[i]=s/K;for(let k=0;k<K;k++)z[k][i]=p.f[k][i]-means[i];}
 const gram=Array.from({length:K},()=>Array(K).fill(0));for(let k=0;k<K;k++)for(let j=k;j<K;j++){let s=0;for(let i=0;i<N;i++)s+=z[k][i]*z[j][i];gram[k][j]=gram[j][k]=s;}
 const eig=eigen(gram).slice(0,3),vt=eig.map(e=>{const values=new Float64Array(N),s=Math.sqrt(Math.max(e.value,0));if(s>1e-8)for(let i=0;i<N;i++){let sum=0;for(let k=0;k<K;k++)sum+=e.vector[k]*z[k][i];values[i]=sum/s;}return values;});
 const coeff=Array.from({length:K},()=>new Float64Array(3));for(let k=0;k<K;k++)for(let r=0;r<3;r++){let s=0;for(let i=0;i<N;i++)s+=z[k][i]*vt[r][i];coeff[k][r]=s;}
 const uniform=uniform5(p.last),dt=p.times.slice(1).map((v,k)=>v-p.times[k]),span=p.times[5]-p.times[0],features=Array.from({length:N},()=>new Float32Array(31));
 for(let i=0;i<N;i++){
  const col=p.f.map(row=>row[i]),med=med32(col),mad=med32(col.map(v=>Math.abs(F(v-med)))),rates=dt.map((d,k)=>F(col[k+1]-col[k])/d),all=[];
  for(let j=0;j<K;j++)for(let k=j+1;k<K;k++)all.push(F(col[k]-col[j])/(p.times[k]-p.times[j]));
  let sum=0;for(const c of col)sum=F(sum+c);const mean=F(sum/K);let vs=0;for(const c of col)vs=F(vs+F(F(c-mean)**2));const std=F(Math.sqrt(F(vs/K)));
  const y=Math.floor(i/32),x=i%32;
  const gy=F(F(p.last[(y===31?y:y+1)*32+x]-p.last[(y===0?y:y-1)*32+x])/((y===0||y===31?1:2)*p.sp[0]));
  const gx=F(F(p.last[y*32+(x===31?x:x+1)]-p.last[y*32+(x===0?x:x-1)])/((x===0||x===31?1:2)*p.sp[1]));
  const count=p.valid.reduce((s,row)=>s+row[i],0),sm120=smooth(p,120,i),sm30=smooth(p,30,i),components=vt.map((row,r)=>coeff[5][r]*row[i]);
  let recon=0,zmean=0;for(let k=0;k<K;k++)zmean+=z[k][i]/K;for(const c of components)recon+=c;
  const values=[rates[4]*p.h,median(rates)*p.h,F(col[5]-col[0])/Math.max(span,1)*p.h,(rates[4]-rates[3])*p.h*p.h/180,sm120-col[5],std,gy,gx,F(uniform[i]-col[5]),count/K,p.h,Math.log1p(p.h),span,median(dt),Math.log(p.sp[0]),Math.log(p.sp[1]),...p.los,+p.md.geometry_available,col[5],med,mad,sm30-col[5],sm120-col[5],median(all)*p.h,count,z[5][i]-zmean-recon,...components];
  for(let j=0;j<31;j++)features[i][j]=Number.isFinite(values[j])?values[j]:0;
 }
 return features;
}
let cache=null;
function weights(){const model=window.TERRA_FOUR_BRANCH_MODEL;if(!model)throw Error('The four-branch checkpoint assets have not loaded.');if(!cache){cache={...model,context:{...model.context,support:Float64Array.from(model.context.support_vectors.flat())}};}return cache;}
async function predict(arg){
 const p=input(arg),model=weights(),bin=p.h<31?0:p.h<91?1:2,w=model.weights[bin],branches=common(p,model),context=new Float64Array(N);context.set(branches.temporal);
 await pause();
 // Tiny fitted fusion terms below1e-14 cannot affect reported millimetres; skip
 // the otherwise expensive SVR only when its contribution is negligible.
 if(Math.abs(w[2])>=1e-14){
  const fs=contextFeatures(p),c=model.context,sv=c.support,dual=c.dual,scratch=new Float64Array(31);
  for(let i=0;i<N;i++){
   if(!p.seen[i])continue;
   for(let j=0;j<31;j++)scratch[j]=F(F(fs[i][j]-c.mean[j])/c.scale[j]);
   let sum=c.intercept;
   for(let k=0;k<dual.length;k++){let distance=0,offset=k*31;for(let j=0;j<31;j++){const d=scratch[j]-sv[offset+j];distance+=d*d;}sum+=dual[k]*Math.exp(-c.gamma*distance);}
   context[i]=sum*c.target_scale+c.target_offset+branches.temporal[i];
   if(i%16===15)await pause();
  }
 }
 const prediction=Array(N),change=Array(N),support=Array.from(p.seen,Boolean);
 for(let i=0;i<N;i++){if(!p.seen[i]){prediction[i]=change[i]=null;continue;}prediction[i]=w[0]*branches.temporal[i]+w[1]*branches.spatial[i]+w[2]*context[i]+w[3]*branches.quality[i];change[i]=prediction[i]-p.last[i];if(!Number.isFinite(prediction[i]))throw Error('Nonfinite output on observed support.');}
 return {prediction_mm:prediction,change_mm:change,support_mask:support,origin_date:new Date(p.times[5]*86400000).toISOString(),target_date:new Date((p.times[5]+p.h)*86400000).toISOString(),horizon_days:p.h,branch_weights:w.slice(),checkpoint_sha256:model.checkpoint_sha256,model_run:model.source_run,confidence_calibrated:false,implementation:'browser-four-branch-port',context_evaluated:Math.abs(w[2])>=1e-14};
}
window.TerraFourBranch={predict,version:'browser-four-branch-v1',_debug:{input,common,contextFeatures,eigen}};
})();
