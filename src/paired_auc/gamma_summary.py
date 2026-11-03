"""Metric-specific cell summaries and equal-cell aggregation for Gamma experiments."""
import math
import numpy as np
import pandas as pd
from .gamma import REGIMES
DENOM={
 'mean_achieved_verification_rate':'replicate_count','mean_paired_width':'paired_width_valid_count','mean_separate_width':'separate_width_valid_count','mean_width_ratio':'width_ratio_valid_count','mean_percent_width_reduction':'percent_width_reduction_valid_count',
 'positive_certification_rate':'sign_valid_count','negative_certification_rate':'sign_valid_count','unresolved_rate':'sign_valid_count',
 'cc_mean':'cc_defined_count','cc_bias_vs_delta_population':'cc_defined_count','cc_mean_error_vs_delta_full_n':'cc_defined_count','cc_mae_vs_delta_full_n':'cc_defined_count','cc_rmse_vs_delta_full_n':'cc_defined_count',
 'cc_undefined_rate':'replicate_count','target_undefined_rate':'replicate_count','degenerate_separate_width_fraction':'target_defined_count'}
MCSE={'mean_achieved_verification_rate':'achieved_verification_rate_mcse','mean_paired_width':'paired_width_mcse','mean_separate_width':'separate_width_mcse','mean_width_ratio':'width_ratio_mcse','mean_percent_width_reduction':'percent_width_reduction_mcse','positive_certification_rate':'positive_certification_mcse','negative_certification_rate':'negative_certification_mcse','unresolved_rate':'unresolved_mcse','cc_mean':'cc_mean_mcse','cc_bias_vs_delta_population':'cc_mean_mcse','cc_mean_error_vs_delta_full_n':'cc_mean_error_mcse','cc_mae_vs_delta_full_n':'cc_mae_mcse','cc_rmse_vs_delta_full_n':'cc_rmse_mcse_delta','cc_undefined_rate':'cc_undefined_mcse','target_undefined_rate':'target_undefined_mcse'}
def stats(x):
    x=np.asarray(x,dtype=float);x=x[np.isfinite(x)];n=len(x)
    sd=float(x.std(ddof=1)) if n>=2 else math.nan
    return (float(x.mean()) if n else math.nan,sd,sd/math.sqrt(n) if n else math.nan,n)
def prop(x):
    x=np.asarray(x,bool);n=len(x);p=float(x.mean()) if n else math.nan
    return p,math.sqrt(p*(1-p)/n) if n else math.nan,n,int(x.sum())
def cells(data):
    out=[]
    for cell_id,part in data.groupby('cell_id',sort=True):
        valid=~part.target_undefined;defined=part.loc[valid];cc=part.loc[~part.cc_undefined & part.delta_cc.notna()]
        for g in REGIMES:
            row={k:part.iloc[0][k] for k in ('cell_id','pair_id','n','pi','kappa','target_verification_rate','mechanism')}
            row.update(count_regime=g,replicate_count=len(part),target_defined_count=int(valid.sum()),target_undefined_count=int((~valid).sum()),cc_defined_count=len(cc),cc_undefined_count=int(part.cc_undefined.sum()),technical_failure_count=int(part.technical_failure.sum()))
            for base,x in [('target_undefined',part.target_undefined),('cc_undefined',part.cc_undefined)]:
                p,se,_,_=prop(x);row[base+'_rate']=p;row[base+'_mcse']=se
            m,sd,se,n=stats(part.achieved_verification_fraction);row.update(mean_achieved_verification_rate=m,achieved_verification_rate_sd=sd,achieved_verification_rate_mcse=se)
            for base in ('paired_width','separate_width','width_ratio','percent_width_reduction'):
                m,sd,se,n=stats(defined[g+'_'+base]);row['mean_'+base]=m;row[base+'_sd']=sd;row[base+'_mcse']=se;row[base+'_valid_count']=n
            for base in ('positive_certification','negative_certification','unresolved'):
                p,se,n,c=prop(defined[g+'_'+base]);row[base+'_rate']=p;row[base+'_mcse']=se;row[base+'_count']=c;row['sign_valid_count']=n
            row['sign_partition_sum']=sum(row[x+'_rate'] for x in ('positive_certification','negative_certification','unresolved'))
            p,se,n,c=prop(defined[g+'_degenerate_separate_width']);row.update(degenerate_separate_width_count=c,degenerate_separate_width_fraction=p,degenerate_separate_width_mcse=se)
            m,sd,se,n=stats(cc.delta_cc);row.update(cc_mean=m,cc_mean_sd=sd,cc_mean_mcse=se,cc_bias_vs_delta_population=m-.06)
            m,sd,se,n=stats(cc.cc_error_vs_full);row.update(cc_mean_error_vs_delta_full_n=m,cc_mean_error_sd=sd,cc_mean_error_mcse=se)
            m,sd,se,n=stats(np.abs(cc.cc_error_vs_full));row.update(cc_mae_vs_delta_full_n=m,cc_mae_sd=sd,cc_mae_mcse=se)
            m,sd,se,n=stats(cc.cc_error_vs_full**2);rmse=math.sqrt(m) if np.isfinite(m) else math.nan
            row.update(cc_rmse_vs_delta_full_n=rmse,cc_rmse_mcse_delta=se/(2*rmse) if rmse>0 else math.nan)
            out.append(row)
    return pd.DataFrame(out)
def aggregate(table,keys):
    result=[]
    for values,part in table.groupby(keys,sort=True,dropna=False):
        if not isinstance(values,tuple):values=(values,)
        row=dict(zip(keys,values));row['cells_total']=len(part);row['cohorts_total']=int(part.replicate_count.sum());row['aggregation']='valid-replicate metric -> cell mean -> equal defined cells';row['correlation_parameter']='latent Gaussian copula kappa' if part.iloc[0].dgm=='Gamma' else 'class-conditional Gaussian eta'
        for metric,den in DENOM.items():
            ok=part[metric].notna();n=int(ok.sum());row[metric]=float(part.loc[ok,metric].mean()) if n else math.nan
            row[metric+'_n_cells_valid']=n;row[metric+'_effective_replicates']=int(part.loc[ok,den].sum())
            if metric in MCSE:
                ses=part.loc[ok,MCSE[metric]].to_numpy(float)
                row[metric+'_equal_cell_mcse']=float(np.sqrt(np.sum(ses**2))/n) if n and np.isfinite(ses).all() else math.nan
        for c in ('target_undefined_count','cc_undefined_count','degenerate_separate_width_count','positive_certification_count','negative_certification_count','unresolved_count'):
            row[c]=int(part[c].sum())
        result.append(row)
    return pd.DataFrame(result)
