# Sharp bounds for paired discrimination

Version **1.1.0**, by Zhuowei Gao and Weipeng Sun. Python software accompanying
*Sharp bounds for paired comparisons of discrimination under arbitrary label
nonresponse* (Statistics and Probability Letters).

The package implements sharp finite-cohort paired-AUC bounds, empirical
envelopes over continuous prevalence sets, and four reproduction workflows:
Gaussian simulation, external-prevalence A2 analysis, Gamma simulation, and a
WDBC semi-synthetic illustration. Count information, population prevalence,
and sampling confidence sets have distinct meanings in these workflows.

## Install and test

Use Python **3.11** (`>=3.11,<3.12`; validated with 3.11.8). From this repository:

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or
`.venv\Scripts\Activate.ps1` in Windows PowerShell. Then:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

Dependencies are pinned in `requirements.txt` and `pyproject.toml`.
All commands below run from the repository root after installation. Use a
fresh output directory for a different run. Generated data and figures stay
local; `results/summaries/` contains small historical reference tables and
deterministic regression fixtures, not newly simulated full results.

## Gaussian experiment

The original 162 cells, 2,000 replicates per cell, seed 20260901, and three
count regimes give 324,000 cohorts and 972,000 analyses. Frozen scientific
intercepts are in `config/gaussian_gamma0.csv`.
Gaussian and Gamma exact and bounded regimes use the true finite-cohort
positive count as an oracle benchmark (bounded is centered on that count).
Unrestricted bounds use only verified labels and public scores.

```bash
python scripts/run_gaussian.py --mode smoke --workers 4 --output results/gaussian_smoke
python scripts/run_gaussian.py --mode full --workers 8 --output results/gaussian_full
```

Smoke uses original replicate 0 in all 162 cells and verifies its deterministic
SHA-256. Both commands generate equal-cell summaries, Table 2, supplementary
tables, and Figure 1. Full mode uses the original resumable cell runner.
Rebuild the intercepts from the original 500,000 calibration and 200,000
validation samples (the command also repeats calibration to check determinism):

```bash
python scripts/calibrate_verification.py --output-dir results/gaussian_calibration
python scripts/run_gaussian.py --mode full --workers 8 --calibration results/gaussian_calibration/gamma0_calibration.csv --output results/gaussian_recalibrated
```

To regenerate the Gaussian artifacts from an existing run or the small
published reference summary:

```bash
python scripts/summarize_gaussian.py --input results/gaussian_full/simulation_results.parquet --mode full --output results/gaussian_artifacts
python scripts/summarize_gaussian.py --from-reference --output results/gaussian_reference_artifacts
```

The reference command redraws historical summaries; it does not rerun simulation.
For the generated TeX Table 2, input `results_macros.tex` before `table2_main.tex`.

## Continuous prevalence envelopes

```bash
python scripts/run_prevalence.py --output results/prevalence_example.json
```

The default example uses a fixed empirical law. For another law, pass
`--input example.json`, with `scores_a`, `scores_b`, `verified`,
`verified_labels` (labels only for verified records), and `intervals`.
Supply a nonempty list of closed endpoint pairs, for example
`[[0.1,0.3],[0.5,0.6]]`. Undefined prevalence values 0 and 1 are excluded
internally; the output records endpoint closure and one-sided limits.
The algorithm evaluates rank knots, interior
stationary points and one-sided boundary limits; it records whether each
extremum is attained. The tests include exhaustive finite-label completions,
integer-count bridges and independent high-precision stationary calculations.

## External prevalence: A2

```bash
python scripts/run_a2.py --mode smoke --workers 4 --output results/a2_smoke
python scripts/run_a2.py --mode smoke --all-k --figures --workers 4 --output results/a2_allk
python scripts/run_a2.py --mode full --figures --workers 8 --output results/a2_full
```

Every evaluation cohort is rebuilt from the Gaussian protocol and frozen
intercepts using its original `cell_id` and `replicate_id`, without private
Parquet files. Full selects all 162 cells and replicates 0, 999, 1999:
486 cohorts. Independent external samples have N=200 or 1000, all K=0,...,N,
and 95% Clopper-Pearson intervals. This gives **584,172 scenarios**. Binomial
weights integrate over K within a cohort, followed by the original equal-cell
averaging. External intervals need not contain the realized finite-cohort
positive fraction; containment is checked only when mathematically required.

Smoke uses nine original cohorts and ten prespecified K values per cohort
(90 scenarios). `--all-k` integrates all 10,818 scenarios for those nine
cohorts and enables figures. Array SHA-256 and historical endpoints are
checked during execution; incomplete-K smoke outputs are not full A2 averages.
Full outputs include `a2_main_table.csv`, `summaries/overall_metrics.csv`,
`summaries/factor_metrics.csv`, and two figures under `figures/`.
The 584,172 scenario rows are stored in per-cohort files under `checkpoints/`.

## Gamma experiment

```bash
python scripts/run_gamma.py calibrate --output results/gamma_calibration
python scripts/run_gamma.py run --mode smoke --output results/gamma_smoke
python scripts/run_gamma.py summarize --input results/gamma_smoke
python scripts/run_gamma.py figures --input results/gamma_smoke
python scripts/run_gamma.py run --mode full --output results/gamma_full
python scripts/run_gamma.py summarize --input results/gamma_full
python scripts/run_gamma.py figures --input results/gamma_full
```

Full uses 36 cells × 2,000 replicates = **72,000 cohorts**, with the original
Gamma margins, Gaussian copula, namespace 33, and stage-4 streams. Smoke uses
original replicate 0 in every cell. Calibration rebuilds all 36 intercepts
from six paired calibration/validation streams and compares the frozen lookup
exactly. Runs use `config/gamma_calibration.json` by default; pass
`--calibration results/gamma_calibration/lookup.json` to use the rebuilt file.
The main table and two figures are generated by the summary/figure commands.
The small Gaussian reference is filtered to the same 36 factor combinations;
`summarize --gaussian-summary results/gaussian_full/artifacts/cell_regime_summaries.csv`
uses a newly generated Gaussian comparison instead.

## WDBC semi-synthetic illustration

```bash
python scripts/run_wdbc.py --mode smoke --output results/wdbc_smoke
python scripts/run_wdbc.py --mode full --output results/wdbc_full
```

The runner downloads `wdbc.data` and `wdbc.names` from the official
[UCI WDBC dataset](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic)
and verifies their frozen SHA-256 values before parsing. It preserves all
569 records, original order and ties, maps M=1/B=0, and compares Mean Radius
(column 3) against Worst Radius (column 23), in the direction A−B. No model
is trained. It regenerates intercepts, scores, masks, analyses, tables and
both figures without precomputed NPZ files. Full has 100 masks per
mechanism/rate stratum (900 masks; 1,800 count-regime analyses); smoke has
one per stratum (9 masks; 18 analyses). Both check benchmark, replicate-0
endpoints and mask hashes; full also checks the reference main table.
Exact-count results are explicitly an **oracle** comparison.

Data citation: Wolberg, W., Mangasarian, O., Street, N., and Street, W. (1993).
*Breast Cancer Wisconsin (Diagnostic)*. UCI Machine Learning Repository.
[doi:10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B).
The third-party dataset is **CC BY 4.0**; attribution and source/hash details
are in `config/wdbc_protocol.json`. Original data are downloaded locally and
are not included in this software release. The software itself is **MIT**,
as specified in `LICENSE`.

## Relationship to the paper

| Implementation / output | Paper or Supplement material |
| --- | --- |
| `core.py`, `bounds.py`, tests | Main sharp finite-cohort results; Supplement finite identities, set-valued counts and inclusion proofs |
| `population.py`, `prevalence.py` | Population counterpart, plug-in bounds; revision Supplement “Empirical envelopes over prevalence sets” |
| Gaussian Table 2 and Figure 1 | Main “Numerical study”; Supplement implementation details, additional simulation and complete-case results |
| A2 main table and figures | Revision Supplement “External prevalence information: an empirical sensitivity analysis” |
| Gamma main table and figures | Revision Supplement “A Gamma extension of the numerical design” |
| WDBC tables and figures | Revision Supplement “WDBC: a real-data-based semi-synthetic illustration”; main additional illustrations |

Revision additions are identified by descriptive section names because final
typeset numbering can change. All generated TeX/figures remain under the
chosen output directory; these commands do not build or edit a manuscript.

## Reproduction scope and citation

Smoke tests validate execution and selected original streams; they do not
establish that the large Gaussian, A2 or Gamma experiments have been rerun
in full. Full commands above regenerate the large outputs from public code,
frozen parameters and seeds. Unit tests check mathematical identities and
edge cases; SHA-256 fixtures check selected deterministic inputs/results.
Small numerical differences on another platform must be investigated against
the stated scientific tolerances, without changing seeds or dropping cases.

`SHA256SUMS` records the canonical source bytes of scientific configuration
and reference files. Cite the release using `CITATION.cff`; the version DOI
should be taken from the actual Zenodo record linked to the GitHub release.
