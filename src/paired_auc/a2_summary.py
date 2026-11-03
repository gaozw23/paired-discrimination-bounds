"""A2 raw probability integration; equal-cohort then equal-cell mixtures."""
from pathlib import Path
import hashlib,json,math
import numpy as np,pandas as pd
from scipy.special import logsumexp
from .a2 import STATUSES

def metrics(df):
 allk=np.ones(len(df),dtype=bool)
 for method in ('point','interval','reference'):
  valid=df[method+'_status'].eq('valid').to_numpy()
  for key in ('lower','upper','width'):yield method+'.'+key,valid,df[method+'_'+key].to_numpy(dtype=float)
  for sign in ('positive','negative','unresolved'):yield method+'.'+sign,valid,df[method+'_sign'].eq(sign).to_numpy(dtype=float)
  for status in STATUSES:yield 'status.'+method+'.'+status,allk,df[method+'_status'].eq(status).to_numpy(dtype=float)
 for ev in ('raw_CP_contains_true_pi','empirical_domain_contains_true_pi','effective_CP_contains_true_pi'):
  yield 'event.'+ev,allk,df[ev].to_numpy(dtype=float)
 both=(df.point_status.eq('valid') & df.interval_status.eq('valid')).to_numpy()
 for method in ('point','interval'):yield 'joint.point_interval.'+method+'_width',both,df[method+'_width'].to_numpy(dtype=float)
 for key in ('lower','upper','width'):
  yield 'joint.point_interval.'+key+'_change',both,(df['interval_'+key]-df['point_'+key]).to_numpy(dtype=float)
 for method in ('point','interval'):
  valid=(df[method+'_status'].eq('valid') & df.reference_status.eq('valid')).to_numpy()
  for key in ('lower','upper','width'):
   delta=(df[method+'_'+key]-df['reference_'+key]).to_numpy(dtype=float)
   yield 'joint.'+method+'_reference.'+key+'_change',valid,delta
   if key!='width':yield 'joint.'+method+'_reference.abs_'+key+'_change',valid,np.abs(delta)


def integ(logw,w,mask,h):
 logs=logw[mask];values=h[mask];weights=w[mask]
 if not len(logs):return dict(numerator=0.,denominator=0.,log_denominator=None,conditional_mean=None,valid_K_count=0,nonzero_K_count=0,zero_mass=True,underflow=False,log_abs_numerator=None,numerator_sign=0,child_count=1)
 assert np.isfinite(values).all()
 b=math.fsum(weights);lb=float(logsumexp(logs));scale=float(logs.max());scaled=np.exp(logs-scale)
 numscaled=math.fsum(scaled*values);denomscaled=math.fsum(scaled)
 a=math.fsum(weights*values);mean=numscaled/denomscaled
 nz=values!=0
 if nz.any():
  la,sg=logsumexp(logs[nz]+np.log(np.abs(values[nz])),b=np.sign(values[nz]),return_sign=True)
  la=float(la) if sg else None;sg=int(sg)
 else:la=None;sg=0
 return dict(numerator=a,denominator=b,log_denominator=lb,conditional_mean=mean,valid_K_count=len(logs),nonzero_K_count=int(nz.sum()),zero_mass=False,underflow=b==0.,log_abs_numerator=la,numerator_sign=sg,child_count=1)


def mix(g):
 count=len(g);pos=g.loc[~g.zero_mass]
 a=math.fsum(g.numerator)/count;b=math.fsum(g.denominator)/count
 if pos.empty:return dict(numerator=a,denominator=b,log_denominator=None,conditional_mean=None,valid_K_count=int(g.valid_K_count.sum()),nonzero_K_count=int(g.nonzero_K_count.sum()),zero_mass=True,underflow=False,log_abs_numerator=None,numerator_sign=0,child_count=int(g.child_count.sum()))
 logs=pos.log_denominator.to_numpy(dtype=float);scale=float(logs.max());w=np.exp(logs-scale)
 mean=math.fsum(w*pos.conditional_mean.to_numpy(dtype=float))/math.fsum(w)
 lb=float(logsumexp(logs))-math.log(count)
 # Signed log numerator uses child scaled moments, preserving subnormal masses.
 nz=g.loc[g.numerator_sign.ne(0)]
 if len(nz):
  la,sg=logsumexp(nz.log_abs_numerator.to_numpy(dtype=float),b=nz.numerator_sign.to_numpy(dtype=float),return_sign=True)
  la=float(la)-math.log(count) if sg else None;sg=int(sg)
 else:la=None;sg=0
 return dict(numerator=a,denominator=b,log_denominator=lb,conditional_mean=mean,valid_K_count=int(g.valid_K_count.sum()),nonzero_K_count=int(g.nonzero_K_count.sum()),zero_mass=False,underflow=b==0.,log_abs_numerator=la,numerator_sign=sg,child_count=int(g.child_count.sum()))


def summarize(directory):
 OUT=Path(directory)
 dest=OUT/'summaries';dest.mkdir(exist_ok=False)
 manifest=json.loads((OUT/'cohort_manifest.json').read_text('utf-8'))
 rows=[];refs=[];indices=[];qa=[]
 for item in manifest:
  stem=f"cohort_{item['cell_id']:03d}_{item['replicate_id']:04d}";path=OUT/'checkpoints'/(stem+'.parquet');meta=json.loads(path.with_suffix('.meta.json').read_text('utf-8'))
  assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256'] and meta['status']=='PASS'
  cols=pd.read_parquet(path).columns.tolist();df=pd.read_parquet(path,columns=[c for c in cols if not c.endswith('_json')])
  assert len(df)==1202 and not df.duplicated(['cell_id','replicate_id','N','K']).any()
  for n,g in df.groupby('N'):
   assert np.array_equal(g.K.to_numpy(),np.arange(n+1))
   lw=g.log_weight.to_numpy();w=g.weight.to_numpy()
   for metric,mask,values in metrics(g):rows.append({'cell_id':item['cell_id'],'replicate_id':item['replicate_id'],'N':int(n),'metric':metric,**integ(lw,w,mask,values)})
   indices.append({'cell_id':item['cell_id'],'replicate_id':item['replicate_id'],'N':int(n),'rows':len(g),'raw_mass':math.fsum(w),'raw_log_mass':float(logsumexp(lw)),'underflow_count':int(g.weight_underflow.sum()),'underflow_log_mass':float(logsumexp(lw[g.weight_underflow])) if g.weight_underflow.any() else None})
  ref=json.loads((OUT/'checkpoints'/(stem+'.reference.json')).read_text('utf-8'));ref['detail_json']=json.dumps(ref.pop('detail'),separators=(',',':'));refs.append(ref)
  qa.append({'cell_id':item['cell_id'],'replicate_id':item['replicate_id'],'rows':len(df),'max_containment_violation':meta['qa']['max_containment_violation'],'failures':len(meta['qa']['failures']),'sha256':meta['sha256']})
 cohort=pd.DataFrame(rows);cohort.to_parquet(dest/'cohort_metrics.parquet',index=False)
 pd.DataFrame(indices).to_csv(dest/'cohort_index.csv',index=False);pd.DataFrame(refs).to_parquet(OUT/'reference_table.parquet',index=False)
 pd.DataFrame(qa).to_csv(dest/'checkpoint_manifest.csv',index=False)
 cell=pd.DataFrame([dict(cell_id=int(c),N=int(n),metric=m,**mix(g)) for (c,n,m),g in cohort.groupby(['cell_id','N','metric'],sort=True)])
 cell.to_parquet(dest/'cell_metrics.parquet',index=False)
 overall=pd.DataFrame([dict(N=int(n),metric=m,**mix(g)) for (n,m),g in cell.groupby(['N','metric'],sort=True)])
 overall.to_parquet(dest/'overall_metrics.parquet',index=False);overall.to_csv(dest/'overall_metrics.csv',index=False)
 metadata=pd.DataFrame([{'cell_id':r['cell_id'],'n':r['n'],'pi_true':r['pi'],'eta':r['eta'],'target_verification_rate':r['target_verification_rate'],'mechanism':r['mechanism']} for r in manifest]).drop_duplicates('cell_id')
 joined=cell.merge(metadata,on='cell_id',validate='many_to_one');factors=[]
 for factor in ('n','pi_true','eta','target_verification_rate','mechanism'):
  for (value,n,m),g in joined.groupby([factor,'N','metric'],sort=True):factors.append(dict(factor=factor,value=str(value),N=int(n),metric=m,**mix(g)))
 factor=pd.DataFrame(factors);factor.to_parquet(dest/'factor_metrics.parquet',index=False);factor.to_csv(dest/'factor_metrics.csv',index=False)
 # Explicit probability tables distinguish empty events from exp underflow.
 for level,frame in [('cohort',cohort),('cell',cell),('overall',overall),('factor',factor)]:
  mass=frame.loc[frame.metric.str.startswith(('status.','event.'))].copy()
  mass['mass']=mass.numerator;mass['log_mass']=mass.log_abs_numerator
  mass['zero_mass']=mass.numerator_sign.eq(0);mass['underflow']=mass.mass.eq(0)&~mass.zero_mass
  keys=[c for c in ('cell_id','replicate_id','factor','value','N','metric') if c in mass.columns]
  mass['K_count']=mass.nonzero_K_count
  mass[keys+['mass','log_mass','zero_mass','underflow','K_count']].to_parquet(dest/(level+'_probability_masses.parquet'),index=False)
 table=[]
 important={'point_width_given_point_valid':'point.width','CP_width_given_CP_valid':'interval.width','reference_width_given_reference_valid':'reference.width','point_valid_probability':'status.point.valid','CP_valid_probability':'status.interval.valid','reference_valid_probability':'status.reference.valid','CP_minus_point_width_given_joint_valid':'joint.point_interval.width_change','point_width_given_point_CP_joint_valid':'joint.point_interval.point_width','CP_width_given_point_CP_joint_valid':'joint.point_interval.interval_width','point_positive_given_point_valid':'point.positive','CP_positive_given_CP_valid':'interval.positive','point_abs_lower_shift_given_point_reference_joint_valid':'joint.point_reference.abs_lower_change','CP_abs_lower_shift_given_CP_reference_joint_valid':'joint.interval_reference.abs_lower_change','point_abs_upper_shift_given_point_reference_joint_valid':'joint.point_reference.abs_upper_change','CP_abs_upper_shift_given_CP_reference_joint_valid':'joint.interval_reference.abs_upper_change','raw_CP_contains_true_pi_probability':'event.raw_CP_contains_true_pi','empirical_domain_contains_true_pi_probability':'event.empirical_domain_contains_true_pi','effective_CP_contains_true_pi_probability':'event.effective_CP_contains_true_pi'}
 for n,g in overall.groupby('N'):
  row={'N':int(n),'evaluation_cohorts':len(manifest),'cells_equal_weighted':len(metadata),'cohorts_per_cell':3,'external_count_scenarios':len(manifest)*(int(n)+1)};lookup=g.set_index('metric')
  for label,metric in important.items():
   r=lookup.loc[metric];row[label]=r.numerator if metric.startswith(('status.','event.')) else r.conditional_mean;row[label+'__unconditional_weighted_numerator']=r.numerator;row[label+'__conditioning_probability_denominator']=r.denominator
  for method in ('point','interval'):
   for st in STATUSES:
    r=lookup.loc[f'status.{method}.{st}'];row[f'{method}_{st}_raw_probability_mass']=r.numerator;row[f'{method}_{st}_log_probability_mass']=r.log_abs_numerator
  table.append(row)
 pd.DataFrame(table).to_csv(OUT/'a2_main_table.csv',index=False)
 result={'status':'PASS','scenario_rows':sum(x['rows'] for x in qa),'cohorts':len(qa),'cohort_N':len(indices),'references':len(refs),'metrics':cohort.metric.nunique(),'cohort_metric_rows':len(cohort),'cell_metric_rows':len(cell),'overall_metric_rows':len(overall),'factor_metric_rows':len(factor),'qa_failures':sum(x['failures'] for x in qa),'max_containment_violation':max(x['max_containment_violation'] for x in qa),'aggregation':'raw K integration, equal cohorts per cell, equal cells; no probability renormalization'}
 with (dest/'results.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
