"""Continuous empirical prevalence hull. No hidden labels, old API mutations, or CI claim.

The rank coefficients and algebraic comparisons are exact for represented scores
and canonicalized domain endpoints. Exported coordinates/values are binary64.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from functools import total_ordering
from bisect import bisect_left, bisect_right
from math import isqrt, fsum, isfinite
import numpy as np

class InputError(ValueError):
    pass

def _sign(x):
    return (x > 0) - (x < 0)

def _single_sign(a, b, d):
    if not b: return _sign(a)
    if not a: return _sign(b)
    if _sign(a) == _sign(b): return _sign(a)
    return _sign(a) * _sign(a*a-b*b*d)

@total_ordering
@dataclass(frozen=True, eq=False)
class Algebraic:
    """r+s*sqrt(d); exact equality, rational isolating comparisons."""
    r: F
    s: F = F(0)
    d: int = 0

    def __post_init__(self):
        object.__setattr__(self, 'r', F(self.r))
        object.__setattr__(self, 's', F(self.s))
        if self.d < 0: raise ValueError('negative radicand')
        z=isqrt(self.d)
        if not self.s or z*z==self.d:
            object.__setattr__(self,'r',self.r+self.s*z)
            object.__setattr__(self,'s',F(0)); object.__setattr__(self,'d',0)

    def __eq__(self, other):
        if not isinstance(other, Algebraic): other=Algebraic(F(other))
        a=self.r-other.r; b=self.s; c=-other.s; d=self.d; e=other.d
        if not b: return _single_sign(a,c,e)==0
        if not c: return _single_sign(a,b,d)==0
        if not a: return b*c < 0 and b*b*d==c*c*e
        h=c*c*e-a*a-b*b*d
        t=h/(2*a*b)
        return t>=0 and t*t==d and _single_sign(a,b,d)==-_sign(c)

    def bounds(self, digits=40):
        if not self.s: return self.r,self.r
        scale=10**digits
        q=isqrt(self.d*scale*scale)
        lo,hi=F(q,scale),F(q+1,scale)
        if self.s<0:lo,hi=hi,lo
        return self.r+self.s*lo,self.r+self.s*hi

    def __lt__(self, other):
        if not isinstance(other, Algebraic): other=Algebraic(F(other))
        if not self.s and not other.s:return self.r<other.r
        if self==other:return False
        digits=40
        while True:
            lo,hi=self.bounds(digits); ol,oh=other.bounds(digits)
            if hi<ol:return True
            if oh<lo:return False
            digits*=2

    def floating(self):
        digits=40
        while True:
            lo,hi=self.bounds(digits)
            fl,fh=float(lo),float(hi)
            if fl==fh:return fl
            # Adjacent floats also bracket the exact value; midpoint rounds stably.
            if np.nextafter(fl, np.inf)>=fh:return float((lo+hi)/2)
            digits*=2

    def key(self):
        return (str(self.r),str(self.s),self.d)

    def serial(self):
        return {'rational':str(self.r),'sqrt_coefficient':str(self.s),'radicand':self.d}

def _array(x,name):
    try:a=np.asarray(x,dtype=float)
    except (ValueError,TypeError) as exc:raise InputError(name+' must be numeric') from exc
    if a.ndim!=1:raise InputError(name+' must be one dimensional')
    return a

def _double_midranks(s):
    order=np.argsort(s,kind='mergesort'); out=[0]*len(s); i=0
    while i<len(s):
        j=i+1
        while j<len(s) and s[order[j]]==s[order[i]]:j+=1
        for k in range(i,j):out[int(order[k])]=i+1+j
        i=j
    return out

def _interval(lo,hi,lc=True,hc=True,**extra):
    return {'lower':float(lo),'upper':float(hi),'lower_closed':lc,'upper_closed':hc,
            'lower_exact':str(lo),'upper_exact':str(hi),**extra}

class EmpiricalPrevalence:
    """Prepare once; labels are verified-only or NaN-masked, never hidden Y."""
    def __init__(self,scores_a,scores_b,verified,verified_labels):
        a=_array(scores_a,'scores_a');b=_array(scores_b,'scores_b')
        if len(a)<2 or len(a)!=len(b) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise InputError('at least two aligned finite scores required')
        r=_array(verified,'verified')
        if len(r)!=len(a) or not np.all(np.isfinite(r)) or not np.all((r==0)|(r==1)):
            raise InputError('verified must be aligned binary values')
        r=r.astype(bool); y=_array(verified_labels,'verified_labels')
        if len(y)==len(a):
            if not np.all(np.isnan(y[~r])):raise InputError('hidden labels forbidden; use NaN placeholders')
            observed=y[r]
        elif len(y)==int(r.sum()):observed=y
        else:raise InputError('labels must be verified-only or aligned NaN-masked')
        if not np.all(np.isfinite(observed)) or not np.all((observed==0)|(observed==1)):
            raise InputError('verified labels must be binary and finite')
        self.n=len(a);self.p=int(observed.sum());self.u=int((~r).sum())
        ra=_double_midranks(a);rb=_double_midranks(b)
        self.d2=tuple(x-y for x,y in zip(ra,rb))
        if sum(self.d2)!=0:raise ArithmeticError('rank conservation failure')
        self.cv2=sum(d*int(v) for d,v in zip(np.asarray(self.d2,dtype=object)[r],observed))
        self.cv2=int(self.cv2)
        self.z2=tuple(sorted(d for d,v in zip(self.d2,r) if not v))
        self.prefix=[0]
        for z in self.z2:self.prefix.append(self.prefix[-1]+z)
        self.scale=2*self.n*self.n
        self.knots=[F(self.p+k,self.n) for k in range(self.u+1)]
        self._canonical={float(F(m,self.n)):F(m,self.n) for m in range(self.n+1)}
        self.segments={'lower':[],'upper':[]}; self.stationary={'lower':[],'upper':[]}
        for side in self.segments:
            for k in range(self.u):
                if side=='lower':z=self.z2[k];pre=self.prefix[k]
                else:z=self.z2[self.u-k-1];pre=self.prefix[self.u]-self.prefix[self.u-k]
                A=self.cv2+pre-(self.p+k)*z; B=self.n*z
                self.segments[side].append((A,B))
                for x,v in self._roots(A,B):
                    if Algebraic(self.knots[k])<x<Algebraic(self.knots[k+1]) and Algebraic(0)<x<Algebraic(1):
                        self.stationary[side].append((x,v,k))

    def _roots(self,A,B):
        if B==0:
            if A!=0:yield Algebraic(F(1,2)),Algebraic(F(4*A,self.scale))
            return
        disc=A*(A+B)
        if disc<0:return
        # Algebraic isolation avoids subtracting nearly equal binary floats.
        for sign in (-1,1):
            x=Algebraic(F(-A,B),F(sign,B),disc)
            val=Algebraic(F(B+2*A,self.scale),F(2*sign,self.scale),disc)
            yield x,val

    def diagnose_prevalence(self,pi):
        raw=float(pi)
        if not isfinite(raw):raise InputError('prevalence must be finite')
        canon=self._canonical.get(raw,F.from_float(raw))
        p1=self.p/self.n;rho=self.u/self.n
        tau=(raw-p1)/rho if self.u else None
        clipped=min(1.,max(0.,tau)) if tau is not None else None
        feasible=self.knots[0]<=canon<=self.knots[-1]
        distance=max(float(self.knots[0])-raw,raw-float(self.knots[-1]),0.)
        near=bool(not feasible and distance<=8*np.finfo(float).eps*max(1.,abs(raw)))
        return {'pi_raw':raw,'pi_exact_used':str(canon),'canonicalized_to_knot':canon!=F.from_float(raw),
                'tau_raw':tau,'tau_clip_diagnostic':clipped,'legacy_projection_flag_formula':tau!=clipped if tau is not None else None,
                'feasible':feasible,'status':'undefined' if raw<=0 or raw>=1 else ('valid' if feasible else 'incompatible'),
                'outside_distance':distance,'near_boundary_roundoff':near,
                'substantial_incompatibility':not feasible and not near}

    def _domain(self,intervals):
        try:raw=list(intervals)
        except TypeError as exc:raise InputError('Pi must be an interval list') from exc
        if not raw:raise InputError('Pi must be nonempty')
        original=[]; frac=[]; canonical=[]
        for item in raw:
            try:
                if len(item)!=2:raise InputError('each interval needs two endpoints')
                l,h=map(float,item)
            except (TypeError,ValueError,OverflowError) as exc:raise InputError('invalid interval') from exc
            if not isfinite(l) or not isfinite(h) or not 0<=l<=h<=1:raise InputError('require 0 <= lower <= upper <= 1')
            original.append([l,h]); ends=[]
            for x in (l,h):
                q=self._canonical.get(x,F.from_float(x));ends.append(q)
                if q!=F.from_float(x):canonical.append({'raw':x,'exact_used':str(q)})
            frac.append(tuple(ends))
        merged=[]
        for l,h in sorted(frac):
            if merged and l<=merged[-1][1]:merged[-1]=(merged[-1][0],max(h,merged[-1][1]))
            else:merged.append((l,h))
        effective=[];excluded=[]
        for l,h in merged:
            lo=max(l,self.knots[0]);hi=min(h,self.knots[-1])
            if lo>hi:
                excluded.append(_interval(l,h,reason='empirical_incompatible'));continue
            if l<lo:excluded.append(_interval(l,lo,True,False,reason='empirical_incompatible'))
            if hi<h:excluded.append(_interval(hi,h,False,True,reason='empirical_incompatible'))
            if lo==0:excluded.append(_interval(F(0),F(0),reason='undefined_prevalence_zero'))
            if hi==1:excluded.append(_interval(F(1),F(1),reason='undefined_prevalence_one'))
            if lo==hi and lo in (0,1):continue
            effective.append((lo,hi))
        return effective,{'original':original,'normalized':[_interval(l,h) for l,h in merged],
             'effective':[_interval(l,h,l!=0,h!=1) for l,h in effective], 'excluded':excluded,
             'empirical_feasible_range':_interval(self.knots[0],self.knots[-1]),
             'canonicalizations':canonical,'zero_one_excluded':True}

    def _value(self,pi,side):
        if self.u==0:return Algebraic(F(self.cv2,self.scale)/pi/(1-pi)),None
        k=min(self.u-1,max(0,bisect_right(self.knots,pi)-1))
        A,B=self.segments[side][k]
        return Algebraic((A+B*pi)/self.scale/pi/(1-pi)),k

    def _candidate(self,side,x,v,kind,k,attained=True):
        pi=x.floating();value=v.floating();residual=None;scaled=None
        if kind=='stationary':
            A,B=self.segments[side][k]
            residual=fsum([B*pi*pi,2*A*pi,-A])
            denom=abs(B*pi*pi)+abs(2*A*pi)+abs(A)
            scaled=abs(residual)/denom if denom else 0.
        return {'side':side,'pi':pi,'value':value,'kind':kind,'segment':k,'attained':attained,
                'root_residual':residual,'root_residual_scaled':scaled,
                'pi_algebraic':x.serial(),'value_algebraic':v.serial(),'_x':x,'_v':v}

    def _component(self,lo,hi,index):
        all_candidates=[]
        for side in ('lower','upper'):
            entries=[];seen=set()
            def add(x,v,kind,k,attained=True):
                key=(x.key(),kind)
                if key not in seen:
                    seen.add(key); entries.append(self._candidate(side,x,v,kind,k,attained))
            for x in set((lo,hi)):
                if 0<x<1:
                    value,k=self._value(x,side)
                    add(Algebraic(x),value,'complete_data' if not self.u else 'domain_boundary',k)
            if lo==0:
                z=self.z2[0] if side=='lower' else self.z2[-1]
                add(Algebraic(0),Algebraic(F(z,2*self.n)),'limit_0',0,False)
            if hi==1:
                z=-self.z2[-1] if side=='lower' else -self.z2[0]
                add(Algebraic(1),Algebraic(F(z,2*self.n)),'limit_1',self.u-1,False)
            for q in self.knots[bisect_right(self.knots,lo):bisect_left(self.knots,hi)]:
                if 0<q<1:
                    value,k=self._value(q,side);add(Algebraic(q),value,'knot',k)
            if lo<hi:
                for x,value,k in self.stationary[side]:
                    if Algebraic(lo)<x<Algebraic(hi):add(x,value,'stationary',k)
                start=max(0,bisect_right(self.knots,lo)-1);stop=min(self.u,bisect_left(self.knots,hi)+1)
                for k in range(start,stop):
                    if self.segments[side][k]==(0,0):
                        left=max(lo,self.knots[k]);right=min(hi,self.knots[k+1])
                        if left<right:add(Algebraic((left+right)/2),Algebraic(0),'constant_interior',k)
            for entry in entries:entry['component']=index
            all_candidates.extend(entries)
        return all_candidates

    @staticmethod
    def _extreme(entries,side):
        candidates=[x for x in entries if x['side']==side]
        best=(min if side=='lower' else max)(x['_v'] for x in candidates)
        winners=[x for x in candidates if x['_v']==best]
        live=[x for x in winners if x['attained']]
        near=[x for x in candidates if x['_v']!=best and abs(x['value']-best.floating())<=1e-12]
        return best.floating(), {'attained':bool(live),'prevalences':sorted(set(x['pi'] for x in live)),
            'limits':sorted(set(x['pi'] for x in winners if not x['attained'])),
            'candidates':[EmpiricalPrevalence._public(x) for x in winners],
            'near_tie_count_not_treated_as_equal':len(near),'comparison':'exact_algebraic_for_represented_domain'}

    @staticmethod
    def _public(x):return {k:v for k,v in x.items() if not k.startswith('_')}

    def envelope(self,intervals):
        eff,domain=self._domain(intervals)
        result={'status':'empty_domain' if not eff else 'valid','lower':None,'upper':None,'width':None,'sign':None,
                'domain':domain,'n':self.n,'p':self.p,'u':self.u,'target':'empirical_population_functional_hull',
                'lower_attainment':None,'upper_attainment':None,'candidates':[],'components':[],
                'diagnostics':{'rank_sum_doubled':sum(self.d2),'cv2':self.cv2,'z2':list(self.z2),
                   'coefficient_arithmetic':'integer','root_comparison':'rational_sqrt_isolation',
                   'export_arithmetic':'binary64','complete_data_branch':self.u==0,
                   'raw_endpoint_diagnostics':[self.diagnose_prevalence(x) for pair in domain['original'] for x in pair]}}
        if not eff:return result
        entries=[]
        for i,(lo,hi) in enumerate(eff):
            local=self._component(lo,hi,i);entries.extend(local)
            lower,la=self._extreme(local,'lower');upper,ua=self._extreme(local,'upper')
            result['components'].append({'domain':domain['effective'][i],'lower':lower,'upper':upper,'lower_attainment':la,'upper_attainment':ua})
        result['lower'],result['lower_attainment']=self._extreme(entries,'lower')
        result['upper'],result['upper_attainment']=self._extreme(entries,'upper')
        result['width']=result['upper']-result['lower']
        result['sign']='positive' if result['lower']>1e-12 else ('negative' if result['upper'] < -1e-12 else 'unresolved')
        result['candidates']=[self._public(x) for x in entries]
        result['diagnostics']['max_scaled_root_residual']=max((x['root_residual_scaled'] or 0. for x in entries),default=0.)
        return result

def prevalence_envelope(scores_a,scores_b,verified,verified_labels,intervals):
    return EmpiricalPrevalence(scores_a,scores_b,verified,verified_labels).envelope(intervals)
