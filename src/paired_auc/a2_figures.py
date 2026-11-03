"""Recreate A2 common-valid comparisons and predeclared sensitivity curves."""
from pathlib import Path
import numpy as np
import pandas as pd

def make_figures(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ROOT = Path(directory)
    RUN = ROOT/'summaries'
    OUT = ROOT/'figures'
    OUT.mkdir(parents=True,exist_ok=False)
    overall=pd.read_csv(RUN/'overall_metrics.csv',float_precision='round_trip')
    chart=[]
    for N in (200,1000):
        sub=overall[overall.N==N].set_index('metric')
        for method,metric in [('Point','joint.point_interval.point_width'),('CP envelope','joint.point_interval.interval_width')]:
            row=sub.loc[metric]
            valid=sub.loc['status.'+('point' if method=='Point' else 'interval')+'.valid']
            chart.append({'N':N,'method':method,'common_valid_conditional_mean_width':row.conditional_mean,'common_valid_probability_denominator':row.denominator,'unconditional_weighted_width_numerator':row.numerator,'method_valid_raw_probability_mass':valid.numerator,'method_valid_log_probability_mass':valid.log_abs_numerator,'conditioning':'point and CP envelope jointly valid for width; each method own valid event for probability panel','SE':'not computed'})
    chart=pd.DataFrame(chart);chart.to_csv(OUT/'overall_comparison_source.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
    fig,(ax,massax)=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
    x=np.arange(2);bar=.32
    for offset,method,color in [(-bar/2,'Point','#2563A6'),(bar/2,'CP envelope','#D97624')]:
        rows=chart[chart.method==method]
        bars=ax.bar(x+offset,rows.common_valid_conditional_mean_width,width=bar,color=color,label=method)
        ax.bar_label(bars,fmt='%.4f',padding=4,fontsize=9)
        bars2=massax.bar(x+offset,rows.method_valid_raw_probability_mass,width=bar,color=color,label=method)
        massax.bar_label(bars2,fmt='%.4f',padding=4,fontsize=9)
    ax.set_xticks(x,[f'N = {N}\njoint-valid mass = {chart[chart.N==N].iloc[0].common_valid_probability_denominator:.6f}' for N in (200,1000)])
    ax.set_ylabel('Mean empirical interval width\nconditional on common validity')
    ax.set_ylim(0,float(chart.common_valid_conditional_mean_width.max())*1.2)
    ax.set_title('Widths on the same jointly valid scenarios',pad=12)
    ax.legend(loc='upper right',frameon=False)
    massax.set_xticks(x,[f'N = {N}' for N in (200,1000)])
    massax.set_ylim(0,1.12);massax.set_ylabel('Raw probability mass of valid inputs')
    massax.set_title('Validity for each method',pad=12)
    fig.suptitle('External prevalence information: empirical interval sensitivity',fontsize=12)
    ax.text(0,-.23,'Fixed evaluation cohorts; deterministic all-K integration; no sampling error bars.',transform=ax.transAxes,fontsize=9,color='#444444')
    for suffix in ('png','pdf'):fig.savefig(OUT/('overall_common_valid_width.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)

    curve_frames=[]
    for cell in (0,80,161):
        path=ROOT/'checkpoints'/f'cohort_{cell:03d}_0000.parquet'
        data=pd.read_parquet(path)
        curve=data[(data.N==1000)&(data.K%10==0)].sort_values('K')
        assert np.array_equal(curve.K.to_numpy(),np.arange(0,1001,10))
        fields=['cell_id','replicate_id','n','pi_true','N','K','pi_hat','point_status','point_lower','point_upper','empirical_feasible_lower','empirical_feasible_upper']
        curve_frames.append(curve[fields])
    curves=pd.concat(curve_frames,ignore_index=True);curves.to_csv(OUT/'fixed_cohort_sensitivity_source.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.2),constrained_layout=True,sharey=True)
    for ax,cell in zip(axes,(0,80,161)):
        d=curves[curves.cell_id==cell]
        x=d.pi_hat.to_numpy();lo=d.point_lower.to_numpy(dtype=float);hi=d.point_upper.to_numpy(dtype=float)
        ax.plot(x,lo,color='#2563A6',linewidth=1.6,label='Lower endpoint')
        ax.plot(x,hi,color='#D97624',linewidth=1.6,label='Upper endpoint')
        ax.fill_between(x,lo,hi,color='#BBD2E8',alpha=.4)
        F_lo=float(d.empirical_feasible_lower.iloc[0]);F_hi=float(d.empirical_feasible_upper.iloc[0]);truth=float(d.pi_true.iloc[0])
        ax.axvline(F_lo,color='#777777',linestyle='--',linewidth=.8)
        ax.axvline(F_hi,color='#777777',linestyle='--',linewidth=.8,label='Empirical feasible boundaries')
        ax.axvline(truth,color='#222222',linestyle=':',linewidth=1.1,label='True-pi input')
        margin=max(.03,.1*(F_hi-F_lo));ax.set_xlim(max(0,F_lo-margin),min(1,F_hi+margin))
        ax.set_title(f'Cell {cell}, replicate 0\nn = {int(d.n.iloc[0])}, true-pi input = {truth:g}')
        ax.set_xlabel('Input prevalence pi')
        ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Empirical paired-AUC difference endpoint')
    handles,labels=axes[-1].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False,fontsize=9)
    fig.suptitle('Predeclared cohort sensitivity curves (display grid only)',fontsize=12)
    for suffix in ('png','pdf'):fig.savefig(OUT/('fixed_cohort_prevalence_sensitivity.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)
