"""New Gamma generator and observable-only adapter to unchanged finite methods."""
import json
REGIMES = ("exact", "bounded", "unrestricted")
TOL = 1e-12
BASE_FIELDS = {
    **{k:'int64' for k in ('cell_id','replicate_id','pair_id','n','seed_base','seed_namespace','seed_stage','total_positives_m','total_negatives','verified_count','verified_positives','verified_negatives','unverified_count','ties_a','ties_b')},
    **{k:'float64' for k in ('pi','kappa','target_verification_rate','gamma_0','gamma_y','gamma_m','gamma_g','achieved_verification_fraction','delta_population','auc_a_full','auc_b_full','delta_full_n','auc_a_cc','auc_b_cc','delta_cc','cc_error_vs_full','replicate_runtime_seconds')},
    **{k:'bool' for k in ('target_undefined','cc_undefined','technical_failure')},
    **{k:'string' for k in ('mechanism','replicate_status','warning_code','seed_signature')},
}
REGIME_FIELDS = {
    **{k:'int64' for k in ('count_set_min','count_set_max','count_set_size')},
    **{k:'float64' for k in ('paired_lower','paired_upper','paired_width','separate_lower','separate_upper','separate_width','width_ratio','percent_width_reduction')},
    **{k:'bool' for k in ('count_contains_m','degenerate_separate_width','positive_certification','negative_certification','unresolved','finite_truth_contained','paired_within_separate','lower_le_upper','unexplained_nan_inf')},
    'status':'string',
}
DIAG_FIELDS = {f'{group}_{kind}_correlation':'float64' for group in ('class0','class1','pooled','unverified') for kind in ('pearson','spearman')}
DIAG_FIELDS.update({f'unverified_d_{x}':'float64' for x in ('mean','sd','min','p05','median','p95','max')})
DIAG_FIELDS['unverified_q_fraction']='float64'
FIELDS = {**BASE_FIELDS,**DIAG_FIELDS,**{f'{g}_{k}':v for g in REGIMES for k,v in REGIME_FIELDS.items()}}

import time
import numpy as np
from scipy.special import ndtr, expit
from scipy.stats import gamma
from paired_auc import core, bounds, metrics
class NumericalGenerationError(RuntimeError):
    def __init__(self,message,record): super().__init__(message);self.record=record
def quantiles(z,shape):
    z,shape=np.broadcast_arrays(np.asarray(z,float),np.asarray(shape,float))
    lower=z<=0;tail=ndtr(np.where(lower,z,-z))
    bad=(tail<=0)|(~np.isfinite(tail))|(~np.isfinite(z))
    if bad.any(): raise NumericalGenerationError('normal tail underflow/nonfinite',{'z':z[bad].tolist(),'shape':shape[bad].tolist(),'tail':tail[bad].tolist()})
    w=np.empty_like(z)
    w[lower]=gamma.ppf(tail[lower],shape[lower],scale=1.)
    w[~lower]=gamma.isf(tail[~lower],shape[~lower],scale=1.)
    bad=~np.isfinite(w)
    if bad.any():raise NumericalGenerationError('nonfinite gamma score',{'z':z[bad].tolist(),'shape':shape[bad].tolist(),'tail':tail[bad].tolist()})
    return w
def generate(size,pi,kappa,root_parts,*,protocol):
    if size<1 or not(0<=pi<=1) or not(-1<=kappa<=1):raise ValueError('invalid generator parameters')
    root=np.random.SeedSequence(root_parts); data_seed,r_seed=root.spawn(2)
    rng=np.random.Generator(np.random.PCG64(data_seed))
    y=rng.binomial(1,pi,size=size).astype(np.int8)
    e=rng.standard_normal((size,2))
    z=np.column_stack((e[:,0],kappa*e[:,0]+np.sqrt(1-kappa*kappa)*e[:,1]))
    shape=np.where(y==0,protocol['shape0'],protocol['shape1'])[:,None]
    w=quantiles(z,shape)
    theta=np.column_stack([np.where(y==0,1.,protocol['scale_parameters'][j]['theta_1']) for j in ('A','B')])
    scores=(theta*w-2.)/np.sqrt(2.)
    if not np.isfinite(scores).all():raise NumericalGenerationError('nonfinite affine score',{'root':root_parts})
    return {'Y':y,'scores_a':scores[:,0],'scores_b':scores[:,1],'latent_z':z,'verification_seed':r_seed}
def probabilities(data,cell,gamma0,*,protocol):
    gy,gm,gg=protocol['verification_slopes'][cell['mechanism']]
    a,b=data['scores_a'],data['scores_b']
    prob=expit(gamma0+gy*data['Y']+gm*(a+b)/2+gg*np.abs(a-b))
    if not np.isfinite(prob).all():raise NumericalGenerationError('nonfinite probability',{'cell':cell})
    return prob
def analyze(scores_a,scores_b,observed_labels,verified,counts,regime):
    """Identification interface: deliberately no full labels or true count argument."""
    r=np.asarray(verified)
    if r.dtype!=bool:raise ValueError('adapter requires Boolean verification mask')
    obs=np.asarray(observed_labels,float)
    if not np.isnan(obs[~r]).all():raise ValueError('hidden labels must be absent')
    if regime=='exact':
        if len(counts)!=1:raise ValueError('exact requires one count')
        p=bounds.paired_bounds_exact_count(scores_a,scores_b,obs,r,counts[0])
        s=bounds.separate_difference_bounds_exact_count(scores_a,scores_b,obs,r,counts[0])
    else:
        p=bounds.paired_bounds_count_set(scores_a,scores_b,obs,r,counts)
        s=bounds.separate_difference_bounds_count_set(scores_a,scores_b,obs,r,counts)
    return p,s
def diagnostic(a,b,y,r):
    result={}
    for group,mask in [('class0',y==0),('class1',y==1),('pooled',np.ones(len(y),bool)),('unverified',~r)]:
        aa=a[mask];bb=b[mask]
        for kind in ('pearson','spearman'):
            if len(aa)<2: value=np.nan
            else:
                x=core.ascending_midranks(aa) if kind=='spearman' else aa
                z=core.ascending_midranks(bb) if kind=='spearman' else bb
                value=float(np.corrcoef(x,z)[0,1]) if np.ptp(x)>0 and np.ptp(z)>0 else np.nan
            result[f'{group}_{kind}_correlation']=value
    du=core.paired_rank_contrasts(a,b)[~r]/len(y)
    if len(du):
        vals=[np.mean(du),np.std(du,ddof=1) if len(du)>1 else np.nan,*np.quantile(du,[0,.05,.5,.95,1])]
    else:vals=[np.nan]*7
    result.update({f'unverified_d_{key}':float(value) for key,value in zip(('mean','sd','min','p05','median','p95','max'),vals)})
    result['unverified_q_fraction']=float(y[~r].mean()) if (~r).any() else np.nan
    return result
def blank_regime(regime,status):
    vals={k:(False if typ=='bool' else None if typ=='int64' else status if typ=='string' else np.nan) for k,typ in REGIME_FIELDS.items()}
    vals['count_set_size']=0
    return {f'{regime}_{k}':v for k,v in vals.items()}
def run_one(cell,replicate_id,gamma0,stage=4,*,protocol):
    t=time.perf_counter();parts=[20261102,33,int(stage),int(cell['cell_id']),int(replicate_id)]
    data=generate(cell['n'],cell['pi'],cell['kappa'],parts,protocol=protocol)
    y,a,b=data['Y'],data['scores_a'],data['scores_b'];n=len(y)
    prob=probabilities(data,cell,gamma0,protocol=protocol)
    r=np.random.Generator(np.random.PCG64(data['verification_seed'])).random(n)<prob
    obs=np.where(r,y.astype(float),np.nan)
    m=int(y.sum());p=int(y[r].sum());v=int(r.sum());u=n-v
    gy,gm,gg=protocol['verification_slopes'][cell['mechanism']]
    row={**cell,'replicate_id':int(replicate_id),'seed_base':20261102,'seed_namespace':33,'seed_stage':stage,'seed_signature':json.dumps(parts,separators=(',',':')),
         'gamma_0':gamma0,'gamma_y':gy,'gamma_m':gm,'gamma_g':gg,'total_positives_m':m,'total_negatives':n-m,'verified_count':v,'verified_positives':p,'verified_negatives':v-p,'unverified_count':u,
         'achieved_verification_fraction':v/n,'delta_population':.06,'target_undefined':m in (0,n),'cc_undefined':p==0 or v-p==0,'technical_failure':False,
         'auc_a_full':np.nan,'auc_b_full':np.nan,'delta_full_n':np.nan,'auc_a_cc':np.nan,'auc_b_cc':np.nan,'delta_cc':np.nan,'cc_error_vs_full':np.nan,
         'replicate_status':'target_undefined' if m in (0,n) else 'PASS','warning_code':'TARGET_UNDEFINED' if m in (0,n) else ('CC_UNDEFINED' if p==0 or v-p==0 else 'OK'),
         'ties_a':n-len(np.unique(a)),'ties_b':n-len(np.unique(b))}
    row.update(diagnostic(a,b,y,r))
    for regime in REGIMES:row.update(blank_regime(regime,'target_undefined'))
    if not row['target_undefined']:
        row['auc_a_full']=core.empirical_auc(a,y);row['auc_b_full']=core.empirical_auc(b,y)
        row['delta_full_n']=row['auc_a_full']-row['auc_b_full']
        compatible=bounds.compatible_total_positive_counts(obs,r,n)
        counts_by={'exact':(m,),'bounded':tuple(k for k in compatible if m-15<=k<=m+15),'unrestricted':compatible}
        for regime,counts in counts_by.items():
            paired,separate=analyze(a,b,obs,r,counts,regime)
            values=metrics.interval_metrics(paired,separate,row['delta_full_n'],tolerance=TOL)
            metrics.enforce_interval_gates(values,count_contains_m=m in counts,identifiers={'cell_id':cell['cell_id'],'replicate_id':replicate_id,'stage':stage,'regime':regime})
            if min(paired.lower,separate.lower)<-1-TOL or max(paired.upper,separate.upper)>1+TOL:raise RuntimeError('endpoint range gate')
            values.update(count_set_min=min(counts),count_set_max=max(counts),count_set_size=len(counts),count_contains_m=m in counts,status='valid')
            row.update({f'{regime}_{k}':value for k,value in values.items()})
        for inner,outer in [('exact','bounded'),('bounded','unrestricted')]:
            for method in ('paired','separate'):
                if row[f'{inner}_{method}_lower']<row[f'{outer}_{method}_lower']-TOL or row[f'{inner}_{method}_upper']>row[f'{outer}_{method}_upper']+TOL:raise RuntimeError('nested count regime gate')
    if not row['cc_undefined']:
        row['auc_a_cc']=core.empirical_auc(a[r],obs[r]);row['auc_b_cc']=core.empirical_auc(b[r],obs[r]);row['delta_cc']=row['auc_a_cc']-row['auc_b_cc'];row['cc_error_vs_full']=row['delta_cc']-row['delta_full_n']
    row['replicate_runtime_seconds']=time.perf_counter()-t
    if set(row)!=set(FIELDS):raise RuntimeError(f'schema difference {set(row)^set(FIELDS)}')
    arrays={'Y':y,'scores_a':a,'scores_b':b,'verified':r,'observed_labels':obs}
    return row,arrays
