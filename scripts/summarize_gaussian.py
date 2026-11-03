"""Original equal-cell Gaussian summaries, tables, and Figure 1."""
from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dataclasses import asdict
from paired_auc.simulation import enumerate_cells, load_protocol
class Phase6ValidationError(RuntimeError):
    """The frozen scientific validation contract was violated."""

TOLERANCE = 1e-12


REGIMES = ("exact", "bounded", "unrestricted")

REGIME_LABELS = {
    "exact": "Exact",
    "bounded": "Bounded",
    "unrestricted": "Unrestricted",
}

MECHANISM_LABELS = {
    "MCAR": "MCAR",
    "score_dependent_MAR": "Score-dependent MAR",
    "MNAR": "MNAR",
}

FACTORS = (
    "cell_id",
    "n",
    "pi",
    "eta",
    "mechanism",
    "target_verification_rate",
)

PRIMARY_METRICS = (
    "mean_paired_width",
    "mean_separate_width",
    "mean_percent_width_reduction",
    "positive_certification_rate",
    "unresolved_rate",
)

AGGREGATE_METRICS = (
    "mean_achieved_verification_rate",
    "mean_paired_width",
    "mean_separate_width",
    "mean_width_ratio",
    "mean_percent_width_reduction",
    "positive_certification_rate",
    "negative_certification_rate",
    "unresolved_rate",
    "cc_mean",
    "cc_bias_vs_delta_population",
    "cc_mean_error_vs_delta_full_n",
    "cc_mae_vs_delta_full_n",
    "cc_rmse_vs_delta_full_n",
    "cc_undefined_rate",
    "target_undefined_rate",
    "degenerate_separate_width_fraction",
)

DENOMINATORS = {
    "mean_achieved_verification_rate": "replicate_count",
    "mean_paired_width": "paired_width_valid_count",
    "mean_separate_width": "separate_width_valid_count",
    "mean_width_ratio": "width_ratio_valid_count",
    "mean_percent_width_reduction": "percent_width_reduction_valid_count",
    "positive_certification_rate": "target_defined_count",
    "negative_certification_rate": "target_defined_count",
    "unresolved_rate": "target_defined_count",
    "cc_mean": "cc_defined_count",
    "cc_bias_vs_delta_population": "cc_defined_count",
    "cc_mean_error_vs_delta_full_n": "cc_defined_count",
    "cc_mae_vs_delta_full_n": "cc_defined_count",
    "cc_rmse_vs_delta_full_n": "cc_defined_count",
    "cc_undefined_rate": "replicate_count",
    "target_undefined_rate": "replicate_count",
    "degenerate_separate_width_fraction": "target_defined_count",
}

MCSE_COLUMNS = (
    "paired_width_mcse",
    "separate_width_mcse",
    "width_ratio_mcse",
    "percent_width_reduction_mcse",
    "positive_certification_mcse",
    "negative_certification_mcse",
    "unresolved_mcse",
    "cc_mean_mcse",
    "cc_mean_error_mcse",
    "cc_mae_mcse",
    "cc_rmse_mcse_delta",
    "cc_undefined_mcse",
)

def mean_and_mcse(values: pd.Series | Iterable[float]) -> tuple[float, float, int]:
    series = pd.Series(values, dtype="float64").dropna()
    series = series[np.isfinite(series)]
    count = int(len(series))
    if count == 0:
        return math.nan, math.nan, 0
    mean = float(series.mean())
    mcse = float(series.std(ddof=1) / math.sqrt(count)) if count >= 2 else math.nan
    return mean, mcse, count

def proportion_and_mcse(values: pd.Series | Iterable[bool]) -> tuple[float, float, int]:
    series = pd.Series(values).dropna().astype(bool)
    count = int(len(series))
    if count == 0:
        return math.nan, math.nan, 0
    proportion = float(series.mean())
    return proportion, math.sqrt(proportion * (1.0 - proportion) / count), count

def _required_columns() -> list[str]:
    columns = [
        *FACTORS,
        "replicate_id",
        "seed_base",
        "seed_cell_id",
        "seed_replicate_id",
        "replicate_status",
        "warning_code",
        "target_undefined",
        "cc_undefined",
        "achieved_verification_fraction",
        "delta_population",
        "delta_full_n",
        "delta_cc",
        "cc_error_vs_full",
    ]
    for regime in REGIMES:
        columns.extend(
            [
                f"{regime}_paired_width",
                f"{regime}_separate_width",
                f"{regime}_width_ratio",
                f"{regime}_percent_width_reduction",
                f"{regime}_degenerate_separate_width",
                f"{regime}_positive_certification",
                f"{regime}_negative_certification",
                f"{regime}_unresolved",
                f"{regime}_finite_truth_contained",
                f"{regime}_paired_within_separate",
                f"{regime}_lower_le_upper",
                f"{regime}_count_contains_m",
                f"{regime}_unexplained_nan_inf",
            ]
        )
    return columns

def load_and_validate_input(input_path: Path, protocol: dict[str, Any]) -> pd.DataFrame:
    if not input_path.exists():
        raise Phase6ValidationError(f"production input missing: {input_path}")
    schema_columns = set(pq.read_schema(input_path).names)
    missing_columns = sorted(set(_required_columns()) - schema_columns)
    if missing_columns:
        raise Phase6ValidationError(f"production columns missing: {missing_columns}")
    data = pd.read_parquet(input_path, columns=_required_columns())
    production = protocol["simulation"]
    expected_rows = int(production["simulated_dataset_count"])
    expected_cells = int(production["dgm_cells"])
    repetitions = int(production["repetitions_per_dgm_cell"])
    production_seed = int(protocol["random_streams"]["base_seed"])
    if len(data) != expected_rows:
        raise Phase6ValidationError(f"expected {expected_rows} rows, found {len(data)}")
    if data["cell_id"].nunique() != expected_cells:
        raise Phase6ValidationError("unexpected number of DGM cells")
    cell_counts = data.groupby("cell_id", observed=True).size()
    if not cell_counts.eq(repetitions).all():
        raise Phase6ValidationError("a DGM cell does not have exactly 2000 replicates")
    expected_ids = pd.MultiIndex.from_product(
        [range(expected_cells), range(repetitions)],
        names=["cell_id", "replicate_id"],
    )
    actual_ids = pd.MultiIndex.from_frame(data[["cell_id", "replicate_id"]].astype(int))
    if actual_ids.has_duplicates:
        raise Phase6ValidationError("duplicate cell_id + replicate_id")
    if len(expected_ids.difference(actual_ids)) or len(actual_ids.difference(expected_ids)):
        raise Phase6ValidationError("missing or unexpected cell_id + replicate_id")
    if not data["seed_base"].eq(production_seed).all():
        raise Phase6ValidationError("wrong production base seed")
    if not data["seed_cell_id"].astype(int).eq(data["cell_id"].astype(int)).all():
        raise Phase6ValidationError("seed cell identifier mismatch")
    if not data["seed_replicate_id"].astype(int).eq(data["replicate_id"].astype(int)).all():
        raise Phase6ValidationError("seed replicate identifier mismatch")
    if not data["replicate_status"].eq("PASS").all():
        raise Phase6ValidationError("production contains non-PASS replicates")
    if not data["warning_code"].eq("OK").all():
        raise Phase6ValidationError("production contains warning codes")

    expected_cells_frame = pd.DataFrame(asdict(cell) for cell in enumerate_cells(protocol))
    actual_cells = data[list(FACTORS)].drop_duplicates().sort_values("cell_id")
    if len(actual_cells) != expected_cells:
        raise Phase6ValidationError("cell_id does not uniquely map to one factor combination")
    expected_cells_frame = expected_cells_frame.rename(
        columns={"verification_mechanism": "mechanism"}
    )
    expected_cells_frame = expected_cells_frame[list(FACTORS)].sort_values("cell_id")
    pd.testing.assert_frame_equal(
        actual_cells.reset_index(drop=True),
        expected_cells_frame.reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        atol=0.0,
        rtol=0.0,
    )

    target_defined = ~data["target_undefined"].astype(bool)
    for regime in REGIMES:
        for flag in (
            "finite_truth_contained",
            "paired_within_separate",
            "lower_le_upper",
            "count_contains_m",
        ):
            if not data.loc[target_defined, f"{regime}_{flag}"].astype(bool).all():
                raise Phase6ValidationError(f"{regime} {flag} gate failed")
        if data.loc[target_defined, f"{regime}_unexplained_nan_inf"].astype(bool).any():
            raise Phase6ValidationError(f"{regime} unexplained NaN/Inf gate failed")
        partition = data.loc[
            target_defined,
            [
                f"{regime}_positive_certification",
                f"{regime}_negative_certification",
                f"{regime}_unresolved",
            ],
        ].astype(int).sum(axis=1)
        if not partition.eq(1).all():
            raise Phase6ValidationError(f"{regime} sign classification does not partition")
        degenerate = data.loc[target_defined, f"{regime}_separate_width"].le(TOLERANCE)
        stored_flag = data.loc[
            target_defined, f"{regime}_degenerate_separate_width"
        ].astype(bool)
        if not degenerate.eq(stored_flag).all():
            raise Phase6ValidationError(f"{regime} degeneracy flag mismatch")
        for metric in ("width_ratio", "percent_width_reduction"):
            missing = data.loc[target_defined, f"{regime}_{metric}"].isna()
            if not missing.eq(degenerate).all():
                raise Phase6ValidationError(f"{regime} {metric} NA rule mismatch")
    return data

def build_cell_summaries(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cell_id, part in data.groupby("cell_id", sort=True, observed=True):
        first = part.iloc[0]
        target_defined = ~part["target_undefined"].astype(bool)
        defined_part = part.loc[target_defined]
        cc_defined = ~part["cc_undefined"].astype(bool) & part["delta_cc"].notna()
        cc_values = part.loc[cc_defined, "delta_cc"]
        cc_errors = part.loc[cc_defined, "cc_error_vs_full"]
        cc_mean, cc_mean_mcse, cc_count = mean_and_mcse(cc_values)
        cc_error_mean, cc_error_mcse, _ = mean_and_mcse(cc_errors)
        cc_mae, cc_mae_mcse, _ = mean_and_mcse(cc_errors.abs())
        squared_errors = cc_errors.pow(2)
        mean_squared_error, mean_squared_error_mcse, _ = mean_and_mcse(squared_errors)
        cc_rmse = math.sqrt(mean_squared_error) if np.isfinite(mean_squared_error) else math.nan
        if np.isfinite(cc_rmse) and cc_rmse > 0 and np.isfinite(mean_squared_error_mcse):
            cc_rmse_mcse = mean_squared_error_mcse / (2.0 * cc_rmse)
        else:
            cc_rmse_mcse = math.nan
        cc_undefined_rate, cc_undefined_mcse, _ = proportion_and_mcse(
            part["cc_undefined"]
        )
        target_undefined_rate, target_undefined_mcse, _ = proportion_and_mcse(
            part["target_undefined"]
        )
        achieved_mean, achieved_mcse, _ = mean_and_mcse(
            part["achieved_verification_fraction"]
        )
        for regime in REGIMES:
            paired_mean, paired_mcse, paired_count = mean_and_mcse(
                defined_part[f"{regime}_paired_width"]
            )
            separate_mean, separate_mcse, separate_count = mean_and_mcse(
                defined_part[f"{regime}_separate_width"]
            )
            ratio_mean, ratio_mcse, ratio_count = mean_and_mcse(
                defined_part[f"{regime}_width_ratio"]
            )
            reduction_mean, reduction_mcse, reduction_count = mean_and_mcse(
                defined_part[f"{regime}_percent_width_reduction"]
            )
            positive_rate, positive_mcse, sign_count = proportion_and_mcse(
                defined_part[f"{regime}_positive_certification"]
            )
            negative_rate, negative_mcse, _ = proportion_and_mcse(
                defined_part[f"{regime}_negative_certification"]
            )
            unresolved_rate, unresolved_mcse, _ = proportion_and_mcse(
                defined_part[f"{regime}_unresolved"]
            )
            degenerate_count = int(
                defined_part[f"{regime}_degenerate_separate_width"].astype(bool).sum()
            )
            row = {
                "cell_id": int(cell_id),
                "n": int(first["n"]),
                "pi": float(first["pi"]),
                "eta": float(first["eta"]),
                "mechanism": str(first["mechanism"]),
                "target_verification_rate": float(first["target_verification_rate"]),
                "count_regime": regime,
                "replicate_count": int(len(part)),
                "target_defined_count": int(target_defined.sum()),
                "target_undefined_count": int((~target_defined).sum()),
                "target_undefined_rate": target_undefined_rate,
                "target_undefined_mcse": target_undefined_mcse,
                "cc_defined_count": cc_count,
                "cc_undefined_count": int(part["cc_undefined"].astype(bool).sum()),
                "cc_undefined_rate": cc_undefined_rate,
                "cc_undefined_mcse": cc_undefined_mcse,
                "mean_achieved_verification_rate": achieved_mean,
                "achieved_verification_rate_mcse": achieved_mcse,
                "paired_width_valid_count": paired_count,
                "mean_paired_width": paired_mean,
                "paired_width_mcse": paired_mcse,
                "separate_width_valid_count": separate_count,
                "mean_separate_width": separate_mean,
                "separate_width_mcse": separate_mcse,
                "width_ratio_valid_count": ratio_count,
                "mean_width_ratio": ratio_mean,
                "width_ratio_mcse": ratio_mcse,
                "percent_width_reduction_valid_count": reduction_count,
                "mean_percent_width_reduction": reduction_mean,
                "percent_width_reduction_mcse": reduction_mcse,
                "degenerate_separate_width_count": degenerate_count,
                "degenerate_separate_width_fraction": (
                    degenerate_count / int(target_defined.sum())
                    if target_defined.any()
                    else math.nan
                ),
                "positive_certification_count": int(
                    defined_part[f"{regime}_positive_certification"].astype(bool).sum()
                ),
                "positive_certification_rate": positive_rate,
                "positive_certification_mcse": positive_mcse,
                "negative_certification_count": int(
                    defined_part[f"{regime}_negative_certification"].astype(bool).sum()
                ),
                "negative_certification_rate": negative_rate,
                "negative_certification_mcse": negative_mcse,
                "unresolved_count": int(
                    defined_part[f"{regime}_unresolved"].astype(bool).sum()
                ),
                "unresolved_rate": unresolved_rate,
                "unresolved_mcse": unresolved_mcse,
                "sign_valid_count": sign_count,
                "sign_partition_sum": positive_rate + negative_rate + unresolved_rate,
                "cc_mean": cc_mean,
                "cc_mean_mcse": cc_mean_mcse,
                "cc_bias_vs_delta_population": cc_mean - 0.06,
                "cc_mean_error_vs_delta_full_n": cc_error_mean,
                "cc_mean_error_mcse": cc_error_mcse,
                "cc_mae_vs_delta_full_n": cc_mae,
                "cc_mae_mcse": cc_mae_mcse,
                "cc_rmse_vs_delta_full_n": cc_rmse,
                "cc_rmse_mcse_delta": cc_rmse_mcse,
            }
            rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != 486:
        raise Phase6ValidationError(f"expected 486 cell-regime rows, found {len(result)}")
    if (result["sign_partition_sum"] - 1.0).abs().max() > TOLERANCE:
        raise Phase6ValidationError("cell-level sign rates do not sum to one")
    return result

def aggregate_cell_summaries(
    cell_summaries: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouper: str | list[str]
    grouper = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for key, part in cell_summaries.groupby(grouper, sort=False, dropna=False):
        keys = (key,) if len(group_columns) == 1 else tuple(key)
        row: dict[str, Any] = dict(zip(group_columns, keys))
        row["cells_total"] = int(len(part))
        for metric in AGGREGATE_METRICS:
            valid = part[metric].notna()
            row[metric] = float(part.loc[valid, metric].mean()) if valid.any() else math.nan
            row[f"{metric}_n_cells_valid"] = int(valid.sum())
            denominator = DENOMINATORS[metric]
            row[f"{metric}_effective_replicates"] = int(
                part.loc[valid, denominator].sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)

def _regime_order(frame: pd.DataFrame) -> pd.DataFrame:
    order = {regime: index for index, regime in enumerate(REGIMES)}
    result = frame.copy()
    result["_regime_order"] = result["count_regime"].map(order)
    sort_columns = [column for column in result.columns if column in {
        "_regime_order", "mechanism", "target_verification_rate", "eta", "n", "pi"
    }]
    result = result.sort_values(sort_columns, kind="stable").drop(columns="_regime_order")
    return result.reset_index(drop=True)

def _save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")

def build_table2(cell_summaries: pd.DataFrame) -> pd.DataFrame:
    aggregate = aggregate_cell_summaries(cell_summaries, ["count_regime"])
    aggregate = _regime_order(aggregate)
    rows: list[dict[str, Any]] = []
    for _, source in aggregate.iterrows():
        regime = str(source["count_regime"])
        row: dict[str, Any] = {
            "count_regime": regime,
            "row_label": REGIME_LABELS[regime],
            "aggregation": "cell mean over valid replicates, then unweighted mean of defined cell means",
            "cell_level_source": "cell_regime_summaries.csv",
        }
        for metric in PRIMARY_METRICS:
            value = float(source[metric])
            kind = "width" if metric in {"mean_paired_width", "mean_separate_width"} else "percent"
            row[metric] = value
            row[f"{metric}_display"] = display_value(value, kind)
            row[f"{metric}_n_cells_valid"] = int(source[f"{metric}_n_cells_valid"])
            row[f"{metric}_effective_replicates"] = int(
                source[f"{metric}_effective_replicates"]
            )
        rows.append(row)
    return pd.DataFrame(rows)

def display_value(value: float, kind: str) -> str:
    if not np.isfinite(value):
        return "NA"
    if kind == "width":
        return f"{value:.3f}"
    if kind == "percent":
        return f"{value:.1f}%" if abs(value) > 1.0 else f"{100.0 * value:.1f}%"
    return f"{value:.6g}"

def build_figure_data(cell_summaries: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (regime, eta, target), part in cell_summaries.groupby(
        ["count_regime", "eta", "target_verification_rate"],
        sort=False,
        observed=True,
    ):
        valid = part["mean_width_ratio"].notna()
        rows.append(
            {
                "panel": {"exact": "A", "bounded": "B", "unrestricted": "C"}[regime],
                "count_regime": regime,
                "eta": float(eta),
                "target_verification_rate": float(target),
                "mean_paired_separate_width_ratio": float(
                    part.loc[valid, "mean_width_ratio"].mean()
                )
                if valid.any()
                else math.nan,
                "n_cells_expected": 18,
                "n_cells_valid": int(valid.sum()),
                "n_cells_all_ratio_na": int((~valid).sum()),
                "effective_valid_replicates": int(
                    part.loc[valid, "width_ratio_valid_count"].sum()
                ),
                "minimum_cell_valid_replicates": int(
                    part.loc[valid, "width_ratio_valid_count"].min()
                )
                if valid.any()
                else 0,
                "maximum_cell_valid_replicates": int(
                    part.loc[valid, "width_ratio_valid_count"].max()
                )
                if valid.any()
                else 0,
                "aggregation": "mean valid replicate ratios within cell, then unweighted mean across n, pi, and mechanism cell means",
                "cell_level_source": "cell_regime_summaries.csv",
            }
        )
    result = pd.DataFrame(rows)
    regime_order = {regime: i for i, regime in enumerate(REGIMES)}
    result["_order"] = result["count_regime"].map(regime_order)
    return result.sort_values(
        ["_order", "target_verification_rate", "eta"],
        ascending=[True, True, True],
    ).drop(columns="_order").reset_index(drop=True)

def plot_figure1(figure_data: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
        }
    )
    styles = {
        0.4: {
            "color": "#0072B2",
            "marker": "o",
            "linestyle": "-",
            "zorder": 3,
        },
        0.6: {
            "color": "#D55E00",
            "marker": "s",
            "linestyle": "--",
            "zorder": 5,
        },
        0.8: {
            "color": "#009E73",
            "marker": "^",
            "linestyle": ":",
            "zorder": 4,
        },
    }
    titles = {
        "exact": "(A) Exact count",
        "bounded": "(B) Bounded count",
        "unrestricted": "(C) Unrestricted compatible count",
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), sharex=True, sharey=True)
    for axis, regime in zip(axes, REGIMES):
        panel = figure_data[figure_data["count_regime"].eq(regime)]
        for target in (0.4, 0.6, 0.8):
            target_match = np.isclose(
                panel["target_verification_rate"].to_numpy(dtype=float),
                target,
                rtol=0.0,
                atol=TOLERANCE,
            )
            line = panel.loc[target_match].sort_values("eta")
            if len(line) != 3:
                raise Phase6ValidationError(
                    f"Figure 1 expected three eta points for {regime}, rate={target}"
                )
            axis.plot(
                line["eta"],
                line["mean_paired_separate_width_ratio"],
                label=f"Verification rate {target:.1f}",
                linewidth=1.6,
                markersize=4.8,
                markeredgewidth=0.8,
                markerfacecolor="none",
                **styles[target],
            )
        axis.set_title(titles[regime], pad=5)
        axis.set_xticks([0.2, 0.8, 0.95], labels=["0.2", "0.8", "0.95"])
        axis.set_ylim(0.0, 0.6)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.55)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean paired/separate width ratio")
    fig.supxlabel(r"Score correlation, $\eta$", y=0.07, fontsize=8.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=3,
        frameon=False,
        handlelength=2.0,
        handletextpad=0.45,
        columnspacing=1.1,
        labelspacing=0.2,
        borderaxespad=0.1,
    )
    fig.subplots_adjust(left=0.08, right=0.995, top=0.88, bottom=0.28, wspace=0.12)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Creator": "paired_auc Phase 6 reporting pipeline",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def _macro_name(regime: str, metric: str) -> str:
    prefix = REGIME_LABELS[regime]
    suffix = {
        "mean_paired_width": "PairedWidth",
        "mean_separate_width": "SeparateWidth",
        "mean_percent_width_reduction": "Reduction",
        "positive_certification_rate": "PositiveCert",
        "unresolved_rate": "Unresolved",
        "negative_certification_rate": "NegativeCert",
    }[metric]
    return f"{prefix}{suffix}"

def build_macros(table2: pd.DataFrame, count_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    count_lookup = count_summary.set_index("count_regime")
    for _, table_row in table2.iterrows():
        regime = str(table_row["count_regime"])
        for metric in PRIMARY_METRICS:
            kind = "width" if metric in {"mean_paired_width", "mean_separate_width"} else "percent"
            macro = _macro_name(regime, metric)
            display = display_value(float(table_row[metric]), kind)
            rows.append(
                {
                    "macro_name": macro,
                    "count_regime": regime,
                    "metric": metric,
                    "full_precision_source_value": float(table_row[metric]),
                    "displayed_value": display,
                    "tex_value": display.replace("%", r"\%"),
                    "aggregation_definition": table_row["aggregation"],
                    "source_file": "table2_main.csv",
                }
            )
        negative = float(count_lookup.loc[regime, "negative_certification_rate"])
        display = f"{100.0 * negative:.3f}%"
        rows.append(
            {
                "macro_name": _macro_name(regime, "negative_certification_rate"),
                "count_regime": regime,
                "metric": "negative_certification_rate",
                "full_precision_source_value": negative,
                "displayed_value": display,
                "tex_value": display.replace("%", r"\%"),
                "aggregation_definition": "cell proportion, then unweighted mean across 162 cells",
                "source_file": "summary_by_count_regime.csv",
            }
        )
    return pd.DataFrame(rows)

def write_macros(macros: pd.DataFrame, path: Path) -> None:
    lines = [
        "% Generated by scripts/summarize_gaussian.py; do not edit manually.",
    ]
    for row in macros.itertuples(index=False):
        lines.append(f"\\newcommand{{\\{row.macro_name}}}{{{row.tex_value}}}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_table2_tex(table2: pd.DataFrame, path: Path) -> None:
    lines = [
        "% Generated by scripts/summarize_gaussian.py; do not edit manually.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Simulation summaries under prespecified equal-cell aggregation.}",
        "\\label{tab:production-main}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Count & Paired width & Separate width & Reduction (\\%) & Positive cert. (\\%) & Unresolved (\\%) \\\\",
        "\\midrule",
    ]
    for row in table2.itertuples(index=False):
        regime = row.count_regime
        lines.append(
            f"{row.row_label} & "
            f"\\{_macro_name(regime, 'mean_paired_width')} & "
            f"\\{_macro_name(regime, 'mean_separate_width')} & "
            f"\\{_macro_name(regime, 'mean_percent_width_reduction')} & "
            f"\\{_macro_name(regime, 'positive_certification_rate')} & "
            f"\\{_macro_name(regime, 'unresolved_rate')} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{minipage}{0.98\\linewidth}",
            "\\footnotesize Each entry first averages over valid replicates within a DGM cell and then averages the defined cell summaries equally across all 162 cells. Percentage reduction is computed replicate by replicate only when the separate width exceeds $10^{-12}$.",
            "\\end{minipage}",
            "\\end{table}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _tex_number(value: float, digits: int = 3) -> str:
    return "--" if not np.isfinite(value) else f"{value:.{digits}f}"

def write_supplement_tex(cell_summaries: pd.DataFrame, path: Path) -> None:
    lines = [
        "% Generated by scripts/summarize_gaussian.py; do not edit manually.",
        "% Requires booktabs, longtable, and pdflscape.",
        "\\begingroup",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
    ]
    for regime in REGIMES:
        part = cell_summaries[cell_summaries["count_regime"].eq(regime)].sort_values(
            ["n", "pi", "eta", "target_verification_rate", "mechanism"]
        )
        lines.extend(
            [
                "\\begin{landscape}",
                "\\begin{longtable}{rrrrlrrrrrrrrrrrrrr}",
                f"\\caption{{Full production grid: {REGIME_LABELS[regime]} count information.}}\\\\",
                "\\toprule",
                "$n$ & $\\pi$ & $\\eta$ & $r$ & Mechanism & $W_p$ & MCSE & $W_s$ & MCSE & Ratio & Red. (\\%) & Pos. (\\%) & Neg. (\\%) & Unres. (\\%) & CC bias & CC MAE & CC RMSE & CC undef. (\\%) & Deg. (\\%) \\\\",
                "\\midrule",
                "\\endfirsthead",
                "\\toprule",
                "$n$ & $\\pi$ & $\\eta$ & $r$ & Mechanism & $W_p$ & MCSE & $W_s$ & MCSE & Ratio & Red. (\\%) & Pos. (\\%) & Neg. (\\%) & Unres. (\\%) & CC bias & CC MAE & CC RMSE & CC undef. (\\%) & Deg. (\\%) \\\\",
                "\\midrule",
                "\\endhead",
            ]
        )
        for row in part.itertuples(index=False):
            mechanism = MECHANISM_LABELS[row.mechanism]
            lines.append(
                f"{row.n} & {row.pi:.1f} & {row.eta:.2f} & {row.target_verification_rate:.1f} & {mechanism} & "
                f"{_tex_number(row.mean_paired_width)} & {_tex_number(row.paired_width_mcse, 4)} & "
                f"{_tex_number(row.mean_separate_width)} & {_tex_number(row.separate_width_mcse, 4)} & "
                f"{_tex_number(row.mean_width_ratio)} & {_tex_number(row.mean_percent_width_reduction, 1)} & "
                f"{_tex_number(100 * row.positive_certification_rate, 1)} & "
                f"{_tex_number(100 * row.negative_certification_rate, 1)} & "
                f"{_tex_number(100 * row.unresolved_rate, 1)} & "
                f"{_tex_number(row.cc_bias_vs_delta_population, 4)} & "
                f"{_tex_number(row.cc_mae_vs_delta_full_n, 4)} & "
                f"{_tex_number(row.cc_rmse_vs_delta_full_n, 4)} & "
                f"{_tex_number(100 * row.cc_undefined_rate, 2)} & "
                f"{_tex_number(100 * row.degenerate_separate_width_fraction, 2)} \\\\"
            )
        lines.extend(
            [
                "\\bottomrule",
                "\\end{longtable}",
                "\\end{landscape}",
            ]
        )
    lines.extend(["\\endgroup"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def build_mcse_summary(cell_summaries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in MCSE_COLUMNS:
        values = cell_summaries[column].dropna()
        rows.append(
            {
                "metric": column,
                "n_cells_defined": int(len(values)),
                "maximum_mcse": float(values.max()) if len(values) else math.nan,
                "median_mcse": float(values.median()) if len(values) else math.nan,
                "minimum_mcse": float(values.min()) if len(values) else math.nan,
                "interpretation": "Monte Carlo numerical precision within DGM cells; not a scientific confidence interval",
            }
        )
    return pd.DataFrame(rows)

def build_degeneracy_outputs(cell_summaries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_cell = cell_summaries[
        [
            *FACTORS,
            "count_regime",
            "target_defined_count",
            "degenerate_separate_width_count",
            "degenerate_separate_width_fraction",
            "width_ratio_valid_count",
        ]
    ].copy()
    summary_rows = []
    for regime in REGIMES:
        part = by_cell[by_cell["count_regime"].eq(regime)]
        nonzero = part[part["degenerate_separate_width_count"].gt(0)]
        count = int(part["degenerate_separate_width_count"].sum())
        denominator = int(part["target_defined_count"].sum())
        summary_rows.append(
            {
                "count_regime": regime,
                "degenerate_replicates": count,
                "target_defined_replicates": denominator,
                "degenerate_fraction": count / denominator,
                "dgm_cells_with_degeneracy": int(len(nonzero)),
                "cell_ids_with_degeneracy": ";".join(
                    str(int(value)) for value in nonzero["cell_id"]
                ),
            }
        )
    return by_cell, pd.DataFrame(summary_rows)

def build_complete_case_summary(cell_summaries: pd.DataFrame) -> pd.DataFrame:
    cells = cell_summaries.drop_duplicates("cell_id")
    metrics = (
        "cc_mean",
        "cc_bias_vs_delta_population",
        "cc_mean_error_vs_delta_full_n",
        "cc_mae_vs_delta_full_n",
        "cc_rmse_vs_delta_full_n",
        "cc_undefined_rate",
    )
    rows = []
    for (mechanism, target), part in cells.groupby(
        ["mechanism", "target_verification_rate"], sort=False
    ):
        row: dict[str, Any] = {
            "mechanism": mechanism,
            "target_verification_rate": float(target),
            "cells_total": int(len(part)),
        }
        for metric in metrics:
            valid = part[metric].notna()
            row[metric] = float(part.loc[valid, metric].mean()) if valid.any() else math.nan
            row[f"{metric}_n_cells_valid"] = int(valid.sum())
            denominator = DENOMINATORS[metric]
            row[f"{metric}_effective_replicates"] = int(part.loc[valid, denominator].sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["mechanism", "target_verification_rate"], kind="stable"
    ).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--from-reference", action="store_true", help="Redraw the published summary, without simulating")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--output", type=Path, default=ROOT / "results/gaussian-artifacts")
    args = parser.parse_args()
    if args.from_reference:
        if args.input:
            parser.error("choose --input or --from-reference")
        cells = pd.read_csv(ROOT / "results/summaries/gaussian_cell_regime.csv")
    else:
        if not args.input:
            parser.error("--input is required unless --from-reference is selected")
        protocol = load_protocol(ROOT / "config/simulation_protocol.yaml")
        if args.mode == "full":
            data = load_and_validate_input(args.input, protocol)
        else:
            from run_simulation import validate_cell_frame
            data = pd.read_parquet(args.input)
            if set(data.cell_id) != set(range(162)):
                raise ValueError("smoke requires all 162 original cells")
            for cell_id, part in data.groupby("cell_id"):
                validate_cell_frame(part, expected_cell_id=int(cell_id), repetitions=1, simulation_base_seed=20260901)
        cells = build_cell_summaries(data)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    _save_csv(cells, out / "cell_regime_summaries.csv")
    table = build_table2(cells)
    _save_csv(table, out / "table2_main.csv")
    write_table2_tex(table, out / "table2_main.tex")
    write_supplement_tex(cells, out / "supplement_full_grid.tex")
    for name, group in [("mechanism", "mechanism"), ("verification_rate", "target_verification_rate"), ("eta", "eta"), ("n", "n"), ("pi", "pi"), ("count_regime", None)]:
        groups = ([group] if group else []) + ["count_regime"]
        _save_csv(_regime_order(aggregate_cell_summaries(cells, groups)), out / f"summary_by_{name}.csv")
    counts = _regime_order(aggregate_cell_summaries(cells, ["count_regime"]))
    write_macros(build_macros(table, counts), out / "results_macros.tex")
    _save_csv(build_mcse_summary(cells), out / "mcse_summary.csv")
    degenerate_cells, degenerate_summary = build_degeneracy_outputs(cells)
    _save_csv(degenerate_cells, out / "degeneracy_by_cell.csv")
    _save_csv(degenerate_summary, out / "degeneracy_summary.csv")
    _save_csv(build_complete_case_summary(cells), out / "complete_case_by_mechanism_and_rate.csv")
    figure = build_figure_data(cells)
    _save_csv(figure, out / "Figure_1_data.csv")
    plot_figure1(figure, out / "Figure_1.pdf", out / "Figure_1.png")
    print(f"Wrote {len(cells)} cell/regime summaries, Table 2, supplementary tables, and Figure 1 to {out}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
