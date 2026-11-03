"""Reproduce the frozen 36-cell Gamma experiment and its summaries/figures."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import gamma as gamma_distribution

from paired_auc.gamma import DIAG_FIELDS, FIELDS, REGIMES, generate, probabilities, run_one
from paired_auc.gamma_summary import aggregate, cells, stats

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(data):
    result = hashlib.sha256()
    for key in ("Y", "scores_a", "scores_b", "latent_z"):
        result.update(np.ascontiguousarray(data[key]).tobytes())
    return result.hexdigest()


def calibrate(args):
    """Rebuild intercepts using the original calibration and validation streams."""
    cfg = read(args.protocol)
    grid = read(args.grid)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    rows, streams = [], []
    for pair in sorted({cell["pair_id"] for cell in grid}):
        selected = [cell for cell in grid if cell["pair_id"] == pair]
        cell = selected[0]
        cal = generate(cfg["calibration"]["size"], cell["pi"], cell["kappa"], [20261102, 33, 0, pair], protocol=cfg)
        val = generate(cfg["calibration"]["validation_size"], cell["pi"], cell["kappa"], [20261102, 33, 1, pair], protocol=cfg)
        ch, vh = array_hash(cal), array_hash(val)
        streams.append({"pair_id": pair, "calibration_root": [20261102, 33, 0, pair], "validation_root": [20261102, 33, 1, pair], "calibration_array_sha256": ch, "validation_array_sha256": vh})
        for cell in selected:
            gy, gm, gg = cfg["verification_slopes"][cell["mechanism"]]
            rate = cell["target_verification_rate"]
            linear = gy * cal["Y"] + gm * (cal["scores_a"] + cal["scores_b"]) / 2 + gg * np.abs(cal["scores_a"] - cal["scores_b"])
            fun = lambda x: float(expit(x + linear).mean() - rate)
            if cell["mechanism"] == "MCAR":
                g0, iterations, calls, converged = float(logit(rate)), 0, 0, True
            else:
                g0, solution = brentq(fun, *cfg["calibration"]["bracket"], xtol=cfg["calibration"]["xtol"], rtol=cfg["calibration"]["rtol"], full_output=True)
                iterations, calls, converged = int(solution.iterations), int(solution.function_calls), bool(solution.converged)
            cp = expit(g0 + linear)
            vp = probabilities(val, cell, g0, protocol=cfg)
            residual, difference = float(cp.mean() - rate), float(vp.mean() - rate)
            passed = converged and abs(residual) <= cfg["calibration"]["residual_limit"] and abs(difference) <= cfg["calibration"]["validation_abs_limit"]
            rows.append({**cell, "gamma_0": g0, "gamma_y": gy, "gamma_m": gm, "gamma_g": gg, "calibration_size": len(cp), "validation_size": len(vp), "calibration_mean": float(cp.mean()), "calibration_residual": residual, "validation_mean": float(vp.mean()), "validation_difference": difference, "calibration_mcse": float(cp.std(ddof=1) / np.sqrt(len(cp))), "validation_mcse": float(vp.std(ddof=1) / np.sqrt(len(vp))), "bracket_lower": -40.0, "bracket_upper": 40.0, "f_bracket_lower": fun(-40), "f_bracket_upper": fun(40), "xtol": cfg["calibration"]["xtol"], "rtol": cfg["calibration"]["rtol"], "iterations": iterations, "function_calls": calls, "converged": converged, "status": "PASS" if passed else "FAIL", "calibration_root": json.dumps([20261102, 33, 0, pair]), "validation_root": json.dumps([20261102, 33, 1, pair]), "calibration_array_sha256": ch, "validation_array_sha256": vh})
    write_json(out / "lookup.json", rows)
    write_json(out / "streams.json", streams)
    pd.DataFrame(rows).to_csv(out / "lookup.csv", index=False, float_format="%.17g")
    if not all(row["status"] == "PASS" for row in rows):
        raise RuntimeError("Gamma calibration failed; original streams and failed rows retained")
    frozen = pd.DataFrame(read(args.calibration))
    rebuilt = pd.DataFrame(rows)
    pd.testing.assert_frame_equal(frozen, rebuilt, check_exact=True)
    print(json.dumps({"status": "PASS", "cells": len(rows), "frozen_lookup_exact": True}))


def validate(data, cell, replicates):
    if set(data.columns) != set(FIELDS) or data.replicate_id.tolist() != list(range(replicates)):
        raise RuntimeError("Gamma schema or replicate ID mismatch")
    if not data.cell_id.eq(cell["cell_id"]).all() or data.technical_failure.any():
        raise RuntimeError("Gamma cell identity or technical failure")
    if not (data.total_positives_m + data.total_negatives).eq(cell["n"]).all():
        raise RuntimeError("Gamma cohort size mismatch")
    defined = data.loc[~data.target_undefined]
    for regime in REGIMES:
        for flag in ("count_contains_m", "finite_truth_contained", "paired_within_separate", "lower_le_upper"):
            if not defined[f"{regime}_{flag}"].all():
                raise RuntimeError(f"Gamma mathematical gate: {regime}_{flag}")


def worker(payload):
    cell, count, cfg, lookup, output, signature = payload
    path = Path(output) / f"cell_{cell['cell_id']:03d}.parquet"
    meta = path.with_suffix(".json")
    if path.exists() or meta.exists():
        if not (path.exists() and meta.exists()):
            raise RuntimeError(f"Incomplete checkpoint: {path.name}")
        old = read(meta)
        if old["signature"] != signature or old["sha256"] != sha(path):
            raise RuntimeError(f"Checkpoint signature/hash mismatch: {path.name}")
        validate(pd.read_parquet(path), cell, count)
        return cell["cell_id"]
    rows = []
    rep = None
    try:
        for rep in range(count):
            row, _ = run_one(cell, rep, float(lookup["gamma_0"]), stage=4, protocol=cfg)
            rows.append(row)
        frame = pd.DataFrame(rows)[list(FIELDS)]
        for key, dtype in FIELDS.items():
            if dtype == "int64":
                frame[key] = frame[key].astype("Int64")
        validate(frame, cell, count)
        frame.to_parquet(path, index=False, compression="zstd")
        pd.testing.assert_frame_equal(frame, pd.read_parquet(path), check_exact=True)
        write_json(meta, {"signature": signature, "sha256": sha(path)})
        return cell["cell_id"]
    except Exception as exc:
        write_json(Path(output) / f"failure_cell_{cell['cell_id']:03d}_rep_{rep}.json", {"cell": cell, "replicate_id": rep, "error_type": type(exc).__name__, "message": str(exc), "record": getattr(exc, "record", None)})
        raise


def run(args):
    cfg, grid, lookup = read(args.protocol), read(args.grid), read(args.calibration)
    if any(row["status"] != "PASS" for row in lookup):
        raise RuntimeError("Frozen calibration contains failed rows")
    if args.cells:
        chosen = {int(value) for value in args.cells.split(",")}
        grid = [cell for cell in grid if cell["cell_id"] in chosen]
        if len(grid) != len(chosen):
            raise ValueError("Unknown cell ID")
    count = 1 if args.mode == "smoke" else cfg["replicates"]
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "checkpoints"
    checkpoint.mkdir(exist_ok=True)
    signature = {"mode": args.mode, "replicates": count, "protocol": sha(args.protocol), "grid": sha(args.grid), "calibration": sha(args.calibration), "engine": sha(ROOT / "src/paired_auc/gamma.py")}
    by_id = {row["cell_id"]: row for row in lookup}
    start = time.perf_counter()
    payloads = [(cell, count, cfg, by_id[cell["cell_id"]], str(checkpoint), signature) for cell in grid]
    if args.workers == 1:
        completed = [worker(payload) for payload in payloads]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            completed = list(pool.map(worker, payloads))
    frames = [pd.read_parquet(checkpoint / f"cell_{cell['cell_id']:03d}.parquet") for cell in grid]
    merged = pd.concat(frames, ignore_index=True)
    merged.to_parquet(args.output / "results.parquet", index=False, compression="zstd")
    result = {"status": "PASS", "mode": args.mode, "cells": len(completed), "cohorts": len(merged), "regime_analyses": 3 * len(merged), "seed_stage": 4, "technical_failures": int(merged.technical_failure.sum()), "wall_seconds": time.perf_counter() - start, "signature": signature}
    write_json(args.output / "run.json", result)
    print(json.dumps(result))


def summarize(args):
    data = pd.read_parquet(args.input / "results.parquet")
    table = cells(data)
    table.to_csv(args.input / "cell_regime_summaries.csv", index=False, float_format="%.17g")
    old = pd.read_csv(args.gaussian_summary, float_precision="round_trip")
    if "gaussian_original_cell_id" not in old:
        old = old.loc[old.n.eq(500) & old.pi.isin([.2, .5]) & old.eta.isin([.2, .8, .95]) & old.target_verification_rate.isin([.4, .8])].copy()
        old = old.rename(columns={"cell_id": "gaussian_original_cell_id"})
        old["kappa"] = old["eta"]
        mapping = pd.DataFrame(read(args.grid)).drop(columns=["pair_id"])
        old = old.merge(mapping, on=["n", "pi", "kappa", "target_verification_rate", "mechanism"], validate="many_to_one")
    if len(old) != 108 or old.cell_id.nunique() != 36:
        raise ValueError("Gaussian comparator must contain the matched 36-cell design")
    old.to_csv(args.input / "matched_gaussian_cells.csv", index=False, float_format="%.17g")
    table["dgm"], old["dgm"] = "Gamma", "Gaussian_matched"
    both = pd.concat([table, old], ignore_index=True)
    both.to_parquet(args.input / "comparison_cells.parquet", index=False)
    aggregate(both, ["dgm", "count_regime"]).to_csv(args.input / "main_table.csv", index=False, float_format="%.17g")
    factors = []
    for name in ("pi", "kappa", "target_verification_rate", "mechanism"):
        sub = aggregate(both, ["dgm", name, "count_regime"]).rename(columns={name: "factor_level"})
        sub["factor"], sub["factor_level"] = name, sub.factor_level.astype(str)
        factors.append(sub)
    pd.concat(factors, ignore_index=True).to_csv(args.input / "factor_tables.csv", index=False, float_format="%.17g")
    diagnostics = []
    for cell, part in data.groupby("cell_id"):
        for metric in [*DIAG_FIELDS, "auc_a_full", "auc_b_full", "delta_full_n"]:
            mean, sd, se, valid = stats(part[metric])
            diagnostics.append({"cell_id": int(cell), "metric": metric, "mean": mean, "sd": sd, "mcse": se, "valid_count": valid})
    pd.DataFrame(diagnostics).to_csv(args.input / "diagnostics.csv", index=False, float_format="%.17g")
    print(json.dumps({"status": "PASS", "cohorts": len(data), "cell_regime_rows": len(table), "gaussian_comparator_sha256": sha(args.gaussian_summary)}))


def figures(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import rankdata

    cfg, grid, lookup = read(args.protocol), read(args.grid), read(args.calibration)
    dest = args.input / "figures"
    dest.mkdir(exist_ok=True)
    table = pd.read_parquet(args.input / "comparison_cells.parquet")
    source = table.groupby(["dgm", "count_regime", "mechanism", "target_verification_rate", "kappa"], as_index=False).agg(mean_reduction=("mean_percent_width_reduction", "mean"), cells=("cell_id", "count"))
    source.to_csv(dest / "gain_source.csv", index=False, float_format="%.17g")
    names, labels = ["MCAR", "score_dependent_MAR", "MNAR"], ["MCAR", "Score-dependent MAR", "MNAR"]
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
    for i, mechanism in enumerate(names):
        for j, regime in enumerate(REGIMES):
            ax = axes[i, j]
            for rate, color in ((.4, "#1769aa"), (.8, "#c25516")):
                for dgm, style, marker in (("Gamma", "-", "o"), ("Gaussian_matched", "--", "s")):
                    sub = source.loc[(source.mechanism == mechanism) & (source.count_regime == regime) & (source.target_verification_rate == rate) & (source.dgm == dgm)].sort_values("kappa")
                    ax.plot(sub.kappa, sub.mean_reduction, style, marker=marker, color=color, label=f"{dgm.replace('_matched', '')}, rate {rate:.1f}", markersize=4)
            if i == 0:
                ax.set_title(regime.capitalize())
            if j == 0:
                ax.set_ylabel(labels[i] + "\nMean reduction (%)")
            if i == 2:
                ax.set_xlabel("Dependence parameter")
            ax.set_xticks([.2, .8, .95], ["0.2", "0.8", "0.95"])
            ax.set_ylim(0, 100)
            ax.grid(alpha=.2)
    handles, legends = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legends, loc="upper center", bbox_to_anchor=(.5, .965), ncol=4, frameon=False)
    fig.suptitle("Paired versus separate interval width: matched 36-cell designs", y=.995)
    fig.text(.5, .02, "Replicate reductions -> cell means -> equal cells.\nGamma: latent copula kappa; Gaussian: class-conditional eta. No paired test or confidence band.", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, .065, 1, .94])
    for suffix in ("png", "pdf"):
        fig.savefig(dest / f"gain_matched.{suffix}", dpi=180)
    plt.close(fig)
    x = np.linspace(-np.sqrt(2), 7, 601)
    density = []
    for cls in (0, 1):
        for model in ("A", "B"):
            shape = cfg["shape0"] if cls == 0 else cfg["shape1"]
            theta = 1 if cls == 0 else cfg["scale_parameters"][model]["theta_1"]
            values = np.sqrt(2) * gamma_distribution.pdf(np.sqrt(2) * x + 2, shape, scale=theta)
            density.extend({"class": cls, "model": model, "score": xx, "density": yy} for xx, yy in zip(x, values))
    density = pd.DataFrame(density)
    density.to_csv(dest / "density_source.csv", index=False, float_format="%.17g")
    histogram = []
    by_id = {row["cell_id"]: row for row in lookup}
    edges = np.linspace(-1, 1, 51)
    for cell in grid:
        for rep in cfg["saved_reps"]:
            data = generate(cell["n"], cell["pi"], cell["kappa"], [20261102, 33, 4, cell["cell_id"], rep], protocol=cfg)
            prob = probabilities(data, cell, by_id[cell["cell_id"]]["gamma_0"], protocol=cfg)
            verified = np.random.Generator(np.random.PCG64(data["verification_seed"])).random(cell["n"]) < prob
            d = (rankdata(data["scores_a"], method="average") - rankdata(data["scores_b"], method="average")) / cell["n"]
            count, _ = np.histogram(d[~verified], bins=edges)
            if count.sum() != (~verified).sum() or not (~verified).any():
                raise RuntimeError("Rank histogram gate failed")
            histogram.extend({**cell, "replicate_id": rep, "left": edges[k], "right": edges[k + 1], "mass": value / (~verified).sum()} for k, value in enumerate(count))
    hs = pd.DataFrame(histogram).groupby(["mechanism", "kappa", "left", "right"], as_index=False).agg(mean_mass=("mass", "mean"), cohorts=("cell_id", "count"))
    hs.to_csv(dest / "rank_histogram_source.csv", index=False, float_format="%.17g")
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for cls, model, color, style in ((0, "A", "#555555", "-"), (1, "A", "#1769aa", "-"), (1, "B", "#c25516", "--")):
        sub = density.loc[(density["class"] == cls) & (density.model == model)]
        axs.flat[0].plot(sub.score, sub.density, color=color, linestyle=style, label="Class 0, both models" if cls == 0 else f"Class 1, model {model}")
    axs.flat[0].set(xlabel="Score S", ylabel="Analytic density", title="Class-conditional Gamma score distributions")
    axs.flat[0].legend(fontsize=9)
    axs.flat[0].grid(alpha=.2)
    for ax, mechanism, label in zip(list(axs.flat)[1:], names, labels):
        for kappa, color in ((.2, "#1769aa"), (.8, "#c25516"), (.95, "#398447")):
            sub = hs.loc[(hs.mechanism == mechanism) & (hs.kappa == kappa)].sort_values("left")
            ax.step((sub.left + sub.right) / 2, sub.mean_mass / (sub.right - sub.left), where="mid", color=color, label=f"kappa={kappa}")
        ax.set(xlabel="Unverified rank contrast (rank A - rank B) / n", ylabel="Mean histogram density", title=label, xlim=(-1, 1))
        ax.legend(fontsize=9)
        ax.grid(alpha=.2)
    fig.suptitle("Distribution and rank diagnostics")
    fig.text(.5, .015, "Rank panels: fixed original 108 cohorts, per-cohort normalized histograms, then equal averaging.\nAll prevalence/rate levels retained; these descriptive curves are not population confidence bands.", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, .07, 1, .96])
    for suffix in ("png", "pdf"):
        fig.savefig(dest / f"distribution_rank_diagnostics.{suffix}", dpi=180)
    plt.close(fig)
    print(json.dumps({"status": "PASS", "figures": 2, "rank_cohorts": 108}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("calibrate", "run", "summarize", "figures"))
    parser.add_argument("--protocol", type=Path, default=ROOT / "config/gamma_protocol.json")
    parser.add_argument("--grid", type=Path, default=ROOT / "config/gamma_grid.json")
    parser.add_argument("--calibration", type=Path, default=ROOT / "config/gamma_calibration.json")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke", help="smoke: 36 original stage-4 rep0 cohorts; full: 36 x 2,000 cohorts")
    parser.add_argument("--cells", help="optional comma-separated original cell IDs")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "results/gamma_smoke")
    parser.add_argument("--input", type=Path, default=ROOT / "results/gamma_smoke")
    parser.add_argument("--gaussian-summary", type=Path, default=ROOT / "results/summaries/gaussian_cell_regime.csv", help="frozen matched reference or regenerated Gaussian cell_regime_summaries.csv")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    globals()[args.command](args)


if __name__ == "__main__":
    main()
