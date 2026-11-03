import math

import pandas as pd

from scripts.summarize_gaussian import (
    aggregate_cell_summaries,
    display_value,
    mean_and_mcse,
    plot_figure1,
    proportion_and_mcse,
)


def test_mean_and_mcse_uses_sample_standard_deviation():
    mean, mcse, count = mean_and_mcse(pd.Series([1.0, 2.0, 3.0, math.nan]))
    assert mean == 2.0
    assert mcse == 1.0 / math.sqrt(3.0)
    assert count == 3


def test_proportion_mcse_uses_actual_valid_denominator():
    proportion, mcse, count = proportion_and_mcse(pd.Series([True, False, True, None]))
    assert proportion == 2.0 / 3.0
    assert mcse == math.sqrt((2.0 / 3.0) * (1.0 / 3.0) / 3.0)
    assert count == 3


def test_equal_cell_aggregation_does_not_pool_valid_replicates():
    rows = []
    for cell_id, cell_mean, valid_count in [(0, 0.0, 1), (1, 1.0, 3)]:
        row = {
            "cell_id": cell_id,
            "count_regime": "exact",
            "mean_width_ratio": cell_mean,
            "width_ratio_valid_count": valid_count,
        }
        for metric in (
            "mean_achieved_verification_rate",
            "mean_paired_width",
            "mean_separate_width",
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
        ):
            row[metric] = 0.0
        for denominator in {
            "replicate_count",
            "paired_width_valid_count",
            "separate_width_valid_count",
            "percent_width_reduction_valid_count",
            "target_defined_count",
            "cc_defined_count",
        }:
            row[denominator] = 1
        rows.append(row)
    result = aggregate_cell_summaries(pd.DataFrame(rows), ["count_regime"])
    assert result.loc[0, "mean_width_ratio"] == 0.5
    assert result.loc[0, "mean_width_ratio_effective_replicates"] == 4
    assert result.loc[0, "mean_width_ratio_n_cells_valid"] == 2


def test_display_precision_is_frozen():
    assert display_value(0.21496958, "width") == "0.215"
    assert display_value(66.54158, "percent") == "66.5%"
    assert display_value(0.260123, "percent") == "26.0%"


def test_figure_plot_accepts_csv_round_trip_rate_values(tmp_path):
    rows = []
    for regime in ("exact", "bounded", "unrestricted"):
        for target in (0.4, 0.5999999999999999, 0.8):
            for eta, ratio in ((0.2, 0.5), (0.8, 0.3), (0.95, 0.15)):
                rows.append(
                    {
                        "count_regime": regime,
                        "target_verification_rate": target,
                        "eta": eta,
                        "mean_paired_separate_width_ratio": ratio,
                    }
                )
    plot_figure1(
        pd.DataFrame(rows),
        tmp_path / "figure.pdf",
        tmp_path / "figure.png",
    )
    assert (tmp_path / "figure.pdf").stat().st_size > 0
    assert (tmp_path / "figure.png").stat().st_size > 0
