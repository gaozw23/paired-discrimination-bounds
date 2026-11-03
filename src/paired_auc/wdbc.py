"""WDBC semi-synthetic reproduction with verified UCI inputs and frozen random streams.

The identification API receives masked labels only. The full labels are used by
this experiment driver for masking, the empirical benchmark and the explicitly
oracle count regime. No model is fitted and no records are removed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
from numpy.random import Generator, PCG64, SeedSequence
from scipy.optimize import brentq
from scipy.special import expit

from .core import (ascending_midranks, empirical_auc, paired_auc_difference,
                   validate_score_pair, validate_observed_labels)
from .bounds import (paired_bounds_count_set, separate_difference_bounds_count_set,
                     compatible_total_positive_counts)
from .metrics import interval_metrics

MECHANISMS = ("MCAR", "score_dependent_MAR", "MNAR")
REGIMES = ("unrestricted", "oracle_exact_count")
RATES = (.4, .6, .8)
TOL = 1e-12


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def csvwrite(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def verified_download(directory, name, spec):
    """Download an official UCI file, or validate an already downloaded copy."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    if path.exists():
        body = path.read_bytes()
    else:
        request = urllib.request.Request(
            spec["url"], headers={"User-Agent": "paired-discrimination-bounds/1.1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
    actual = hashlib.sha256(body).hexdigest()
    if len(body) != spec["bytes"] or actual != spec["sha256"]:
        raise ValueError(f"UCI {name}: SHA-256/size mismatch; input was not accepted")
    if not path.exists():
        path.write_bytes(body)
    return body


def pair_auc(scores, labels):
    """Direct pairwise benchmark independent of the midrank implementation."""
    positive = scores[labels == 1, None]
    negative = scores[labels == 0][None, :]
    return float(((positive > negative).sum() + .5 * (positive == negative).sum())
                 / (positive.size * negative.size))


def tie_summary(scores, labels):
    _, counts = np.unique(scores, return_counts=True)
    return {
        "unique_values": len(counts),
        "tied_groups": int(sum(counts > 1)),
        "tied_records": int(sum(counts[counts > 1])),
        "duplicate_excess": int(sum(counts - 1)),
        "tied_unordered_pairs": int(sum(counts * (counts - 1) // 2)),
        "cross_class_tied_pairs": int((scores[labels == 1, None]
                                       == scores[labels == 0][None, :]).sum()),
    }


def load_cohort(data_directory, protocol):
    """Preserve all 569 source records, their order, labels and observed ties."""
    specs = protocol["dataset"]["files"]
    raw = verified_download(data_directory, "wdbc.data", specs["wdbc.data"])
    names = verified_download(data_directory, "wdbc.names", specs["wdbc.names"]).decode("ascii")
    if "field 3 is Mean Radius" not in names or "field 23 is Worst Radius" not in names:
        raise ValueError("UCI dictionary does not confirm the frozen score fields")
    rows = list(csv.reader(raw.decode("ascii").splitlines()))
    if len(rows) != 569 or any(len(row) != 32 for row in rows):
        raise ValueError("WDBC row/column structure mismatch")
    ids = np.asarray([row[0] for row in rows])
    diagnosis = np.asarray([row[1] for row in rows])
    labels = (diagnosis == protocol["positive_label"]).astype(np.int8)
    features = np.asarray([row[2:] for row in rows], dtype=np.float64)
    if (len(set(ids)) != len(rows) or set(diagnosis) != {"M", "B"}
            or int(labels.sum()) != 212 or not np.isfinite(features).all()):
        raise ValueError("WDBC ID/diagnosis/class/missing-value validation failed")
    a = features[:, protocol["score_A"]["zero_based_column"] - 2]
    b = features[:, protocol["score_B"]["zero_based_column"] - 2]
    ra, rb = ascending_midranks(a), ascending_midranks(b)
    contrast = ra - rb
    n = len(labels)
    va, vb = (ra - .5) / n, (rb - .5) / n
    auc_a, auc_b = pair_auc(a, labels), pair_auc(b, labels)
    benchmark = {
        "n": n, "positive_count": int(labels.sum()),
        "negative_count": int(n - labels.sum()),
        "auc_a_pairwise": auc_a, "auc_b_pairwise": auc_b,
        "delta_full_n": auc_a - auc_b,
        "auc_a_midrank": empirical_auc(a, labels),
        "auc_b_midrank": empirical_auc(b, labels),
        "delta_midrank": paired_auc_difference(a, b, labels),
        "pearson": float(np.corrcoef(a, b)[0, 1]),
        "spearman": float(np.corrcoef(ra, rb)[0, 1]),
        "rank_contrast": {
            "mean": float(contrast.mean()), "sd": float(contrast.std(ddof=1)),
            "min": float(contrast.min()), "max": float(contrast.max()),
            "q25": float(np.quantile(contrast, .25)),
            "q50": float(np.quantile(contrast, .5)),
            "q75": float(np.quantile(contrast, .75)),
            "zero_count": int((contrast == 0).sum()),
        },
        "ties_a": tie_summary(a, labels), "ties_b": tie_summary(b, labels),
    }
    np.testing.assert_allclose(
        [auc_a, auc_b, auc_a - auc_b],
        [benchmark["auc_a_midrank"], benchmark["auc_b_midrank"], benchmark["delta_midrank"]],
        atol=TOL, rtol=TOL,
    )
    return a, b, labels, (va + vb) / 2, abs(va - vb), benchmark


def calibrate(labels, mstar, gstar, protocol):
    """Recompute the nine intercepts on this fixed empirical cohort."""
    probabilities = np.empty((3, 3, len(labels)))
    records = []
    settings = protocol["calibration"]
    for h, mechanism in enumerate(protocol["mechanisms"]):
        gy = math.log(3) if h == 2 else 0.
        gm, gg = float(mechanism["gamma_m"]), float(mechanism["gamma_g"])
        offset = gy * labels + gm * mstar + gg * gstar
        for j, rate in enumerate(RATES):
            def residual(intercept):
                return float(expit(intercept + offset).mean() - rate)
            intercept = (math.log(rate / (1 - rate)) if h == 0 else brentq(
                residual, *settings["bracket"], xtol=settings["xtol"], rtol=settings["rtol"]
            ))
            probs = expit(intercept + offset)
            probabilities[h, j] = probs
            direct = np.array([
                1 / (1 + math.exp(-(intercept + gy * int(labels[i])
                                    + gm * float(mstar[i]) + gg * float(gstar[i]))))
                for i in range(len(labels))
            ])
            direct_residual = math.fsum(direct) / len(labels) - rate
            if abs(direct_residual) > settings["residual_max"]:
                raise ValueError("WDBC intercept calibration residual exceeds frozen tolerance")
            np.testing.assert_allclose(direct, probs, atol=TOL, rtol=TOL)
            records.append({
                "mechanism_id": h, "mechanism": MECHANISMS[h], "rate_id": j,
                "target_rate": rate, "gamma_0": intercept, "gamma_y": gy,
                "gamma_m": gm, "gamma_g": gg, "mean_probability": float(probs.mean()),
                "residual": float(probs.mean() - rate),
                "minimum_probability": float(probs.min()),
                "maximum_probability": float(probs.max()),
                "bracket_lower_residual": residual(-40),
                "bracket_upper_residual": residual(40),
                "method": "analytic_logit" if h == 0 else "brentq", "status": "PASS",
            })
    if not np.all(np.diff(probabilities, axis=1) > 0):
        raise ValueError("WDBC verification probabilities are not ordered across rates")
    return probabilities, records


def verification_masks(probabilities, h, replicate, seed_prefix=(20261102, 34, 1)):
    """One uniform stream per mechanism/replicate, shared by all three rates."""
    uniforms = Generator(PCG64(SeedSequence([*seed_prefix, h, replicate]))).random(
        probabilities.shape[-1]
    )
    masks = uniforms <= probabilities[h]
    if not np.all(masks[0] <= masks[1]) or not np.all(masks[1] <= masks[2]):
        raise ValueError("WDBC masks failed the frozen shared-uniform nesting check")
    return masks


def validate_protocol(protocol):
    """Reject accidental changes to the frozen scientific design."""
    expected = {
        "positive_label": "M", "negative_label": "B", "replicates": 100,
        "rates": list(RATES), "regimes": list(REGIMES),
        "seed": [20261102, 34, 1, "mechanism_id", "replicate_id"],
        "uniform_shared_across_rates": True, "smoke_replicates": [0],
        "atol": TOL, "rtol": TOL, "sign_threshold": TOL,
        "separate_degeneracy_threshold": TOL,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"Frozen WDBC protocol mismatch for {key}")
    if (protocol["score_A"]["zero_based_column"] != 2
            or protocol["score_B"]["zero_based_column"] != 22):
        raise ValueError("Frozen WDBC score-field mapping changed")
    if protocol["mechanisms"] != [
        {"id": 0, "name": MECHANISMS[0], "gamma_y": 0., "gamma_m": 0., "gamma_g": 0.},
        {"id": 1, "name": MECHANISMS[1], "gamma_y": 0., "gamma_m": .6, "gamma_g": .8},
        {"id": 2, "name": MECHANISMS[2], "gamma_y": "log(3)", "gamma_m": .6, "gamma_g": .8},
    ]:
        raise ValueError("Frozen WDBC mechanism coefficients changed")
    if protocol["calibration"] != {
        "bracket": [-40, 40], "xtol": 1e-10, "rtol": 1e-12, "residual_max": 1e-8
    }:
        raise ValueError("Frozen WDBC calibration tolerance changed")


def check_references(reference_directory, benchmark, records, main_table, mode):
    """Check archived scientific numbers, excluding timing and environment data."""
    reference_directory = Path(reference_directory)
    expected = json.loads((reference_directory / "wdbc_benchmark.json").read_text())

    def compare(actual, reference):
        if isinstance(reference, dict):
            for key, value in reference.items():
                compare(actual[key], value)
        elif isinstance(reference, (int, float)):
            np.testing.assert_allclose(actual, reference, atol=TOL, rtol=TOL)
        else:
            assert actual == reference, (actual, reference)

    compare(benchmark, expected)
    checked = {"benchmark": "PASS"}
    targets = [("wdbc_replicate0.csv", [row for row in records if row["replicate_id"] == 0])]
    if mode == "full":
        targets.append(("wdbc_main_table.csv", main_table))
    for name, actual_rows in targets:
        with (reference_directory / name).open(newline="", encoding="utf-8") as stream:
            expected_rows = list(csv.DictReader(stream))
        if len(actual_rows) != len(expected_rows):
            raise AssertionError(f"WDBC reference row count mismatch: {name}")
        key = lambda row: (int(row["mechanism_id"]), int(row["rate_id"]), row["regime"])
        actual_rows = sorted(actual_rows, key=key)
        expected_rows = sorted(expected_rows, key=key)
        for actual, reference in zip(actual_rows, expected_rows):
            for column, value in reference.items():
                if isinstance(actual[column], (str, bool)):
                    compare(actual[column], value)
                else:
                    compare(actual[column], float(value))
        checked[name] = "PASS"
    return checked


def run(protocol_path, output_directory, mode, reference_directory, *, figures=True):
    """Download, validate, calibrate, run, summarize and verify one reproduction."""
    started = time.perf_counter()
    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "analysis.jsonl").exists():
        raise FileExistsError("Use a fresh output directory for this WDBC run")
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    validate_protocol(protocol)
    if mode not in ("smoke", "full"):
        raise ValueError("mode must be smoke or full")
    replicates = protocol["smoke_replicates"] if mode == "smoke" else list(range(100))
    a, b, y, mstar, gstar, benchmark = load_cohort(out / "data", protocol)
    probabilities, calibration_rows = calibrate(y, mstar, gstar, protocol)
    dump(out / "benchmark.json", benchmark)
    csvwrite(out / "calibration.csv", calibration_rows)
    records = []
    leakage_checks = 0
    with (out / "analysis.jsonl").open("x", encoding="utf-8") as stream:
        for rep in replicates:
            for h in range(3):
                masks = verification_masks(probabilities, h, rep, protocol["seed"][:3])
                for j in range(3):
                    mask = masks[j]
                    rows = run_one(a, b, y, mask, h, j, rep, {}, benchmark["delta_full_n"])
                    if rep == 0:
                        # Keep observed inputs fixed and reverse all hidden labels.
                        altered = y.copy()
                        altered[~mask] = 1 - altered[~mask]
                        obs1, obs2 = np.where(mask, y, np.nan), np.where(mask, altered, np.nan)
                        result1 = identify(a, b, mask, obs1, unrestricted_counts(obs1, mask))
                        result2 = identify(a, b, mask, obs2, unrestricted_counts(obs2, mask))
                        if json.dumps(result1, sort_keys=True) != json.dumps(result2, sort_keys=True):
                            raise AssertionError("Identification output depended on hidden labels")
                        leakage_checks += 1
                    for row in rows:
                        stream.write(json.dumps(row, allow_nan=True) + "\n")
                    stream.flush()
                    records.extend(rows)
    csvwrite(out / "analysis.csv", [
        {key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()}
        for row in records
    ])
    main_table = summarize(out, records, replicates)
    reference_checks = check_references(reference_directory, benchmark, records, main_table, mode)
    if figures:
        generate_figures(out, replicates)
    report = {
        "mode": mode, "status": "PASS", "mask_count": 9 * len(replicates),
        "analysis_count": len(records), "strata": len(main_table),
        "replicate_ids": replicates, "technical_failures": 0,
        "hidden_label_invariance_checks": leakage_checks,
        "max_calibration_residual": max(abs(row["residual"]) for row in calibration_rows),
        "reference_checks": reference_checks,
        "input_files": protocol["dataset"]["files"],
        "dataset_doi": protocol["dataset"]["doi"], "dataset_license": "CC BY 4.0",
        "full_reproduction_executed": mode == "full", "figures_generated": figures,
        "wall_seconds": time.perf_counter() - started,
    }
    dump(out / "run_summary.json", report)
    return report

def unrestricted_counts(observed_labels,verified):
    obs,r=validate_observed_labels(observed_labels,verified,len(observed_labels))
    if not np.all(np.isnan(obs[~r])):raise ValueError('unknown labels must be NaN')
    return compatible_total_positive_counts(obs,r)

def identify(scores_a,scores_b,verified,observed_labels,counts):
    a,b=validate_score_pair(scores_a,scores_b)
    obs,r=validate_observed_labels(observed_labels,verified,len(a))
    if not np.all(np.isnan(obs[~r])):raise ValueError('unknown labels must be NaN')
    counts=tuple(counts)
    paired=paired_bounds_count_set(a,b,obs,r,counts)
    separate=separate_difference_bounds_count_set(a,b,obs,r,counts)
    result=interval_metrics(paired,separate,float('nan'))
    result.pop('finite_truth_contained') # evaluated outside identification using benchmark
    result.update(count_set_min=min(counts),count_set_max=max(counts),count_set_size=len(counts),paired_lower_attaining_counts=list(paired.lower_attaining_counts),paired_upper_attaining_counts=list(paired.upper_attaining_counts))
    for name in ('auc_a_lower','auc_a_upper','auc_b_lower','auc_b_upper'):result[name]=float(getattr(separate,name))
    # Additional reporting of marginal attaining counts; no change to finite core.
    p=int(obs[r].sum());u=int((~r).sum());m=np.asarray(counts);q=m-p;den=m*(len(a)-m);correction=m*(m+1)/2
    for label,scores in [('a',a),('b',b)]:
        ranks=ascending_midranks(scores);prefix=np.r_[0.,np.cumsum(np.sort(ranks[~r]))];known=float(obs[r]@ranks[r])
        lo=(known+prefix[q]-correction)/den;hi=(known+prefix[u]-prefix[u-q]-correction)/den
        for endpoint,values in [('lower',lo),('upper',hi)]:
            extreme=result[f'auc_{label}_{endpoint}']
            result[f'auc_{label}_{endpoint}_attaining_counts']=[int(x) for x in m[np.abs(values-extreme)<=1e-14]]
    return result


def run_one(a,b,y,R,h,rate_id,rep,signatures,target):
    """Experiment driver sees Y; only masked inputs cross identification boundary."""
    start=time.perf_counter();n=len(a);R=np.asarray(R,dtype=bool)
    obs=np.where(R,y,np.nan);p=int(y[R].sum());v=int(R.sum());u=n-v
    cc_undefined=p==0 or p==v
    ca=cb=cc=float('nan')
    if not cc_undefined:ca=empirical_auc(a[R],y[R]);cb=empirical_auc(b[R],y[R]);cc=ca-cb
    base={'mechanism_id':h,'mechanism':MECHANISMS[h],'rate_id':rate_id,'target_rate':RATES[rate_id],'replicate_id':rep,'n':n,'verified_count':v,'verified_positives':p,'verified_negatives':v-p,'unverified_count':u,'achieved_verification_fraction':v/n,'delta_full_n':target,'auc_a_cc':ca,'auc_b_cc':cb,'delta_cc':cc,'cc_signed_error':cc-target,'cc_absolute_error':abs(cc-target),'cc_squared_error':(cc-target)**2,'cc_undefined':cc_undefined,'technical_failure':False,'target_undefined':False,'status':'PASS','failure_message':'','mask_sha256':array_sha(R),'observed_labels_sha256':array_sha(obs),**signatures}
    # Main unrestricted count set constructed before and without oracle m.
    sets=[unrestricted_counts(obs,R),(int(y.sum()),)]
    output=[]
    for regime,counts in zip(REGIMES,sets):
        result=identify(a,b,R,obs,counts)
        result['finite_truth_contained']=bool(result['paired_lower']-TOL<=target<=result['paired_upper']+TOL)
        assert result['finite_truth_contained'] and result['paired_within_separate'] and result['lower_le_upper'] and not result['unexplained_nan_inf']
        assert result['separate_lower']-TOL<=target<=result['separate_upper']+TOL
        assert -1-TOL<=result['separate_lower']<=result['separate_upper']<=1+TOL
        output.append({**base,'regime':regime,**result,'runtime_seconds':time.perf_counter()-start})
    return output


METRICS=('achieved_verification_fraction','verified_count','verified_positives','verified_negatives','unverified_count','paired_width','separate_width','width_ratio','percent_width_reduction','delta_cc','cc_signed_error','cc_absolute_error','cc_squared_error')
KEY=('mechanism_id','mechanism','rate_id','target_rate','regime')
def statistics(values,name):
    x=np.asarray(values,float);x=x[np.isfinite(x)];n=len(x)
    return {f'{name}_n':n,f'{name}_mean':float(x.mean()) if n else float('nan'),f'{name}_sd':float(x.std(ddof=1)) if n>1 else float('nan'),f'{name}_mcse':float(x.std(ddof=1)/np.sqrt(n)) if n>1 else float('nan'),**{f'{name}_q{q}':float(np.quantile(x,q/100,method='linear')) if n else float('nan') for q in (25,50,75)}}
def summarize(out, rows, replicates):
    main=[]
    for h in range(3):
        for j in range(3):
            for regime in REGIMES:
                group=[r for r in rows if (r['mechanism_id'],r['rate_id'],r['regime'])==(h,j,regime)]
                valid=[r for r in group if not r['technical_failure'] and not r['target_undefined']]
                record={k:group[0][k] for k in KEY}
                record.update(design_n=len(group),valid_n=len(valid),technical_failure_n=sum(r['technical_failure'] for r in group),target_undefined_n=sum(r['target_undefined'] for r in group),degenerate_n=sum(r['degenerate_separate_width'] for r in valid),cc_valid_n=sum(not r['cc_undefined'] for r in group))
                for metric in METRICS:record.update(statistics([r[metric] for r in valid],metric))
                record['cc_MAE']=record['cc_absolute_error_mean'];record['cc_RMSE']=math.sqrt(record['cc_squared_error_mean'])
                for bound in ('paired','separate'):
                    outcomes=['positive' if r[bound+'_lower']>TOL else 'negative' if r[bound+'_upper']< -TOL else 'unresolved' for r in valid]
                    for outcome in ('positive','negative','unresolved'):
                        count=outcomes.count(outcome);rate=count/len(valid) if valid else float('nan')
                        record.update({f'{bound}_{outcome}_n':count,f'{bound}_{outcome}_rate':rate,f'{bound}_{outcome}_mcse':math.sqrt(rate*(1-rate)/len(valid)) if valid else float('nan')})
                main.append(record)
    csvwrite(out/'main_table.csv',main);dump(out/'summaries/main_table.json',main)
    by_key={(r['mechanism_id'],r['rate_id'],r['replicate_id'],r['regime']):r for r in rows};pairs=[];pair_summaries=[]
    for h in range(3):
        for j in range(3):
            local=[]
            for rep in replicates:
                u=by_key[h,j,rep,REGIMES[0]];e=by_key[h,j,rep,REGIMES[1]]
                row={k:u[k] for k in ('mechanism_id','mechanism','rate_id','target_rate','replicate_id')}
                for kind in ('paired','separate'):row[kind+'_width_difference']=u[kind+'_width']-e[kind+'_width']
                pairs.append(row);local.append(row)
            record={k:local[0][k] for k in ('mechanism_id','mechanism','rate_id','target_rate')};record['design_n']=len(local)
            for kind in ('paired','separate'):
                metric=kind+'_width_difference';record.update(statistics([r[metric] for r in local],metric))
            pair_summaries.append(record)
    csvwrite(out/'summaries/paired_count_differences.csv',pairs);dump(out/'summaries/paired_count_differences.json',pairs)
    csvwrite(out/'summaries/paired_count_summary.csv',pair_summaries);dump(out/'summaries/paired_count_summary.json',pair_summaries)
    cols=['mechanism_id','mechanism','rate_id','target_rate','replicate_id','regime']
    csvwrite(out/'figures/source/rep0_intervals.csv',[{k:r[k] for k in cols+['paired_lower','paired_upper','separate_lower','separate_upper','delta_full_n']} for r in rows if r['replicate_id']==0])
    csvwrite(out/'figures/source/mask_distributions.csv',[{k:r[k] for k in cols+['paired_width','separate_width','width_ratio','percent_width_reduction']} for r in rows])
    return main


def generate_figures(out, replicates):
    os.environ.setdefault("MPLCONFIGDIR", str(out / "cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    rows=list(csv.DictReader((out/'figures/source/rep0_intervals.csv').open()))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(3,3,figsize=(12,8),sharex=True,sharey=True)
    lower=min(float(r['separate_lower']) for r in rows);upper=max(float(r['separate_upper']) for r in rows);margin=(upper-lower)*.06
    for h in range(3):
        for j in range(3):
            ax=axes[h,j]
            for y,reg in zip([1,0],REGIMES):
                r=next(r for r in rows if int(r['mechanism_id'])==h and int(r['rate_id'])==j and r['regime']==reg)
                ax.plot([float(r['separate_lower']),float(r['separate_upper'])],[y,y],color='#B8C9DA',linewidth=9,solid_capstyle='butt')
                ax.plot([float(r['paired_lower']),float(r['paired_upper'])],[y,y],color='#126782',linewidth=4,solid_capstyle='butt')
            ax.axvline(float(r['delta_full_n']),color='#AE2012',linestyle='--',linewidth=1.5)
            ax.axvline(0,color='#777777',linewidth=.6)
            ax.set_xlim(lower-margin,upper+margin);ax.set_ylim(-.5,1.5);ax.set_yticks([0,1],['Oracle exact count','Unrestricted'])
            ax.set_title(f'{MECHANISMS[h].replace("score_dependent_MAR","Score-dependent MAR")} | target rate {RATES[j]:.1f}',fontsize=10)
            ax.grid(axis='x',alpha=.2)
            if h==2:ax.set_xlabel('AUC difference: Mean Radius - Worst Radius')
    fig.suptitle('Real-data-based semi-synthetic illustration: fixed replicate 0',fontsize=15,y=.98)
    fig.legend(handles=[Line2D([0],[0],color='#B8C9DA',lw=8,label='Separately optimized'),Line2D([0],[0],color='#126782',lw=4,label='Paired'),Line2D([0],[0],color='#AE2012',ls='--',label='Full-cohort empirical benchmark')],loc='lower center',ncol=3,bbox_to_anchor=(.5,.01),frameon=False)
    fig.tight_layout(rect=[0,.07,1,.95])
    for ext in ('png','pdf'):fig.savefig(out/f'figures/rep0_intervals.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    rows=list(csv.DictReader((out/'figures/source/mask_distributions.csv').open()))
    fig,axes=plt.subplots(3,2,figsize=(10,9),sharex=True,sharey=True)
    for h in range(3):
        for col,reg in enumerate(REGIMES):
            ax=axes[h,col]
            for k,(kind,color) in enumerate([('paired_width','#126782'),('separate_width','#E09F3E')]):
                values=[[float(r[kind]) for r in rows if int(r['mechanism_id'])==h and int(r['rate_id'])==j and r['regime']==reg] for j in range(3)]
                boxes=ax.boxplot(values,positions=np.arange(3)+(k-.5)*.3,widths=.25,patch_artist=True,manage_ticks=False,whis=1.5,showfliers=True,flierprops={'markersize':2},medianprops={'color':'black'})
                for patch in boxes['boxes']:patch.set_facecolor(color);patch.set_alpha(.75)
            ax.set_xticks(range(3),['0.4','0.6','0.8']);ax.set_title(f'{MECHANISMS[h].replace("score_dependent_MAR","Score-dependent MAR")} | {"Unrestricted" if col==0 else "Oracle exact count"}',fontsize=11)
            ax.grid(axis='y',alpha=.2)
            if col==0:ax.set_ylabel('Identification interval width')
            if h==2:ax.set_xlabel('Target verification rate')
    fig.suptitle(f'Variation across {len(replicates)} artificial masks per stratum\nOne fixed cohort; rates share uniforms within mechanism and replicate',fontsize=13,y=.98)
    fig.legend(handles=[Line2D([0],[0],color='#126782',lw=8,label='Paired'),Line2D([0],[0],color='#E09F3E',lw=8,label='Separately optimized')],loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=[0,.05,1,.94])
    for ext in ('png','pdf'):fig.savefig(out/f'figures/mask_width_distributions.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
