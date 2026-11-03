"""Rebuild the frozen Gaussian IDs and deterministic all-K A2 experiment."""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib, json, math, time
import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp
from . import simulation as sim
from .a2 import binomial_plan
from .prevalence import EmpiricalPrevalence

def array_signature(arrays):
    """Portable scientific hash; excludes ZIP headers and timestamps."""
    digest = hashlib.sha256()
    for name in ('scores_a', 'scores_b', 'verified', 'observed_labels'):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode('ascii'))
        digest.update(value.dtype.str.encode('ascii'))
        digest.update(str(value.shape).encode('ascii'))
        digest.update(value.tobytes())
    return digest.hexdigest()


def regenerate_cohort(protocol, cell, replicate_id, gamma_0, base_seed):
    """Use the original public generator and the two original child streams."""
    da, db = sim.signal_parameters(protocol)
    seed = np.random.SeedSequence([int(base_seed), cell.cell_id, int(replicate_id)])
    data_seed, verification_seed = seed.spawn(2)
    y, a, b = sim.generate_full_data_sample(
        size=cell.n, pi=cell.pi, eta=cell.eta, delta_a=da, delta_b=db,
        seed_sequence=data_seed)
    mean, difference = sim.verification_covariates(a, b)
    gy, gm, gg = sim.mechanism_coefficients(protocol, cell.mechanism)
    probability = expit(gamma_0 + gy*y + gm*mean + gg*difference)
    if not np.isfinite(probability).all():
        raise ArithmeticError('nonfinite verification probabilities')
    verified = np.random.default_rng(verification_seed).random(cell.n) < probability
    return dict(scores_a=a, scores_b=b, verified=verified,
                observed_labels=np.where(verified, y.astype(float), np.nan))


@lru_cache(maxsize=4)
def weight_plan(n, truth):
    rows, checks = binomial_plan(n, truth)
    if abs(checks['raw_mass_error']) > 1e-12:
        raise ArithmeticError('external binomial mass check failed')
    return rows, checks

def brief(result,status=None):
 return {'status':status or result['status'],'domain':result['domain'],
 'lower_attainment':result['lower_attainment'],'upper_attainment':result['upper_attainment'],
 'diagnostics':{k:v for k,v in result['diagnostics'].items() if k not in ('z2','cv2','rank_sum_doubled')}}


def method(law,pi):
 result=law.envelope(pi);status=result['status']
 if len(pi)==1 and pi[0][0]==pi[0][1]:
  v=pi[0][0];status='undefined' if v in (0,1) else ('incompatible' if status=='empty_domain' else status)
 return result,status


def fields(prefix,result,status):
 valid=status=='valid'
 return {prefix+'_'+k:(result[k] if valid else None) for k in ('lower','upper','width','sign')} | {
 prefix+'_status':status,prefix+'_lower_attained':result['lower_attainment']['attained'] if valid else None,
 prefix+'_upper_attained':result['upper_attainment']['attained'] if valid else None}


def validate_frame(df, all_k=True):
 checks={};fail=[];max_violation=0.
 def test(name,mask,scope=None):
  mask=np.atleast_1d(np.asarray(mask,dtype=bool));checks[name]={'checked':len(mask) if scope is None else int(np.sum(scope)),'failures':int((~mask).sum())}
  if not mask.all():fail.append({'name':name,'rows':df.loc[~mask,['cell_id','replicate_id','N','K']].head(20).to_dict('records') if len(mask)==len(df) else []})
 for m in ('point','interval','reference'):
  valid=df[m+'_status'].eq('valid').to_numpy();l=df[m+'_lower'].to_numpy(dtype=float);u=df[m+'_upper'].to_numpy(dtype=float);w=df[m+'_width'].to_numpy(dtype=float)
  test(m+'_validity',~valid | (np.isfinite(l)&np.isfinite(u)&np.isfinite(w)&(l>=-1-2e-12)&(u<=1+2e-12)&(l<=u+2e-12)&(np.abs(w-(u-l))<=2e-12)),valid)
  expected=np.where(l>1e-12,'positive',np.where(u < -1e-12,'negative','unresolved'))
  test(m+'_sign',~valid | df[m+'_sign'].eq(expected).to_numpy(),valid)
  test(m+'_missing',valid | (df[[m+'_lower',m+'_upper',m+'_width',m+'_sign',m+'_lower_attained',m+'_upper_attained']].isna().all(axis=1).to_numpy()))
  test(m+'_known_status',df[m+'_status'].isin(['valid','undefined','incompatible','empty_domain']).to_numpy())
 pv=df.point_status.eq('valid').to_numpy();cv=df.interval_status.eq('valid').to_numpy();rv=df.reference_status.eq('valid').to_numpy()
 test('point_implies_CP',~pv|cv,pv)
 for name,x in [('CP_lower_le_point',df.interval_lower-df.point_lower),('point_upper_le_CP',df.point_upper-df.interval_upper),('point_width_le_CP',df.point_width-df.interval_width)]:
  a=x.to_numpy(dtype=float);test(name,~pv|(a<=1e-12+1e-12*np.abs(df.point_width.to_numpy(dtype=float))),pv)
  if pv.any():max_violation=max(max_violation,float(np.nanmax(a[pv])))
 event=df.effective_CP_contains_true_pi.to_numpy(dtype=bool)
 test('effective_event_identity',event==(df.raw_CP_contains_true_pi.to_numpy(dtype=bool)&df.empirical_domain_contains_true_pi.to_numpy(dtype=bool)))
 test('effective_event_implies_ref_CP_valid',~event|(cv&rv),event)
 for name,a in [('CP_lower_le_ref',df.interval_lower-df.reference_lower),('ref_upper_le_CP',df.reference_upper-df.interval_upper)]:
  test(name,~event | (a.to_numpy(dtype=float)<=2e-12),event)
 center=df.K.to_numpy()==np.rint(df.N*df.pi_true).to_numpy()
 test('center_status',~center|df.point_status.eq(df.reference_status).to_numpy(),center)
 for k in ('lower','upper'):
  diff=np.abs(df['point_'+k]-df['reference_'+k]).to_numpy(dtype=float)
  test('center_'+k,~(center&rv)|(diff<=2e-12),center&rv)
 for n,g in df.groupby('N') if all_k else []:
  test('all_K_'+str(n),np.array_equal(g.K.to_numpy(),np.arange(n+1)) and len(g)==n+1)
  if abs(math.expm1(float(logsumexp(g.log_weight.to_numpy()))))>1e-12:fail.append({'name':'raw_mass','N':int(n)})
 return {'checks':checks,'failures':fail,'max_containment_violation':max(0.,max_violation)}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def produce(payload):
    protocol, cell_values, rid, gamma_0, design, output, all_k = payload
    cell = sim.CellSpec(**cell_values)
    output = Path(output)
    stem = f'cohort_{cell.cell_id:03d}_{rid:04d}'
    started = time.perf_counter()
    arrays = regenerate_cohort(protocol, cell, rid, gamma_0, design['base_seed'])
    law = EmpiricalPrevalence(arrays['scores_a'], arrays['scores_b'],
                             arrays['verified'], arrays['observed_labels'])
    reference, rs = method(law, [[cell.pi, cell.pi]])
    rows, weight_checks = [], []
    for n in design['external_sizes']:
        weights, check = weight_plan(n, cell.pi)
        weight_checks.append(check)
        selected = weights if all_k else [weights[k] for k in (0, 1, n//2, n-1, n)]
        for w in selected:
            k = int(w['K']); pi = k/n
            point, ps = method(law, [[pi, pi]])
            interval, cs = method(law, [[w['cp_lower'], w['cp_upper']]])
            raw = w['cp_contains_pi_true']; feasible = rs == 'valid'
            row = {'cell_id':cell.cell_id, 'replicate_id':rid, 'N':n, 'K':k,
                   'n':cell.n, 'pi_true':cell.pi, 'eta':cell.eta,
                   'target_verification_rate':cell.target_verification_rate,
                   'mechanism':cell.mechanism, 'pi_hat':pi,
                   **{key:w[key] for key in ('weight','log_weight','weight_underflow','cp_lower','cp_upper')},
                   'raw_CP_contains_true_pi':raw,
                   'empirical_domain_contains_true_pi':feasible,
                   'effective_CP_contains_true_pi':raw and feasible,
                   'empirical_feasible_lower':law.p/law.n,
                   'empirical_feasible_upper':(law.p+law.u)/law.n,
                   'point_detail_json':json.dumps(brief(point,ps), separators=(',',':'), allow_nan=False),
                   'interval_detail_json':json.dumps(brief(interval,cs), separators=(',',':'), allow_nan=False)}
            for prefix, result, status in [('point',point,ps),('interval',interval,cs),('reference',reference,rs)]:
                row.update(fields(prefix,result,status))
            rows.append(row)
    frame = pd.DataFrame(rows)
    qa = validate_frame(frame, all_k=all_k)
    if qa['failures']:
        write_json(output/(stem+'.failure.json'), qa)
        raise ArithmeticError(f'A2 validation failed for {stem}: {qa["failures"]}')
    frame.to_parquet(output/(stem+'.parquet'), index=False, compression='zstd')
    saved = pd.read_parquet(output/(stem+'.parquet'))
    pd.testing.assert_frame_equal(frame,saved,check_dtype=False,check_exact=True)
    ref = {'cell_id':cell.cell_id,'replicate_id':rid,'pi_true':cell.pi,
           **fields('reference',reference,rs),'detail':brief(reference,rs)}
    write_json(output/(stem+'.reference.json'),ref)
    write_json(output/(stem+'.law.json'),{'n':law.n,'p':law.p,'u':law.u,
        'cv2':law.cv2,'z2':law.z2,'segments':law.segments,
        'knots':[str(x) for x in law.knots],'scale':law.scale})
    digest = hashlib.sha256((output/(stem+'.parquet')).read_bytes()).hexdigest()
    record = {**asdict(cell),'replicate_id':rid,'rows':len(frame),
              'array_scientific_sha256':array_signature(arrays),
              'sha256':digest,'status':'PASS','qa':qa,
              'weights':weight_checks,'seconds':time.perf_counter()-started}
    write_json(output/(stem+'.meta.json'),record)
    return record


def run(repository, output, mode='smoke', workers=1, all_k=False):
    repository, output = Path(repository), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output/'checkpoints'; checkpoint.mkdir()
    protocol = sim.load_protocol(repository/'config/simulation_protocol.yaml')
    gamma = sim.load_gamma0_table(repository/'config/gaussian_gamma0.csv')
    design = json.loads((repository/'config/a2_protocol.json').read_text('utf-8'))
    cells = sim.enumerate_cells(protocol)
    ids = design['cell_ids'] if mode == 'full' else design['smoke_cell_ids']
    reps = design['replicate_ids'] if mode == 'full' else design['smoke_replicate_ids']
    all_k = mode == 'full' or all_k
    tasks = [(protocol,asdict(cells[cid]),rid,sim.gamma_for_cell(gamma,cells[cid]),
              design,str(checkpoint),all_k) for cid in ids for rid in reps]
    write_json(output/'execution_plan.json', {'mode':mode,'all_k':all_k,
        'cell_ids':ids,'replicate_ids':reps,'base_seed':design['base_seed'],
        'external_sizes':design['external_sizes'],'expected_cohorts':len(tasks),
        'expected_rows':len(tasks)*(1202 if all_k else 10),
        'scientific_tolerance':1e-12,'confidence_interval_claim':False})
    if workers == 1:
        records = [produce(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(produce,tasks))
    write_json(output/'cohort_manifest.json',records)
    result = {'status':'PASS','mode':mode,'all_k':all_k,'cohorts':len(records),
              'scenario_rows':sum(x['rows'] for x in records),
              'maximum_containment_violation':max(x['qa']['max_containment_violation'] for x in records)}
    if mode == 'full' and result['scenario_rows'] != design['scenarios']:
        raise ArithmeticError('full A2 scenario count does not match frozen design')
    if all_k:
        from .a2_summary import summarize
        summarize(output)
    if mode == 'full':
        actual = pd.read_csv(output/'a2_main_table.csv',float_precision='round_trip').set_index('N')
        expected = pd.read_csv(repository/'results/summaries/a2_main_table.csv',float_precision='round_trip').set_index('N')
        if not actual.index.equals(expected.index) or set(actual.columns) != set(expected.columns):
            raise ArithmeticError('full A2 reference table schema differs')
        for key in expected:
            a,e = actual[key].to_numpy(),expected[key].to_numpy()
            if not np.allclose(a,e,atol=1e-12,rtol=1e-12,equal_nan=True):
                raise ArithmeticError(f'full A2 reference table differs: {key}')
        result['historical_full_table_check'] = 'PASS'
    # Compare independent historical references, never overwritten by this run.
    expected_arrays = pd.read_csv(repository/'results/summaries/a2_array_reference.csv')
    expected_rows = pd.read_csv(repository/'results/summaries/a2_endpoint_reference.csv',float_precision='round_trip')
    comparisons = 0
    for item in records:
        cid, rid = item['cell_id'], item['replicate_id']
        expected = expected_arrays.loc[(expected_arrays.cell_id==cid)&(expected_arrays.replicate_id==rid)]
        if len(expected):
            if item['array_scientific_sha256'] != expected.iloc[0].array_scientific_sha256:
                raise ArithmeticError(f'Original stream hash differs for {cid}/{rid}')
            comparisons += 1
        expected = expected_rows.loc[(expected_rows.cell_id==cid)&(expected_rows.replicate_id==rid)]
        if len(expected):
            actual = pd.read_parquet(checkpoint/f'cohort_{cid:03d}_{rid:04d}.parquet')
            actual = actual.set_index(['N','K'])
            for row in expected.to_dict('records'):
                if (row['N'],row['K']) not in actual.index:
                    continue
                found = actual.loc[(row['N'],row['K'])]
                for name in ('point','interval','reference'):
                    if found[name+'_status'] != row[name+'_status']:
                        raise ArithmeticError('A2 historical method status changed')
                    for field in ('lower','upper','width'):
                        key = name+'_'+field
                        a,e = found[key],row[key]
                        if not ((pd.isna(a) and pd.isna(e)) or abs(a-e)<=1e-12+1e-12*abs(e)):
                            raise ArithmeticError(f'A2 historical endpoint changed: {cid}/{rid}/{key}')
                        comparisons += 1
    result['historical_reference_comparisons'] = comparisons
    write_json(output/'results.json',result)
    return result
