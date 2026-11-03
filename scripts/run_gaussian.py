"""Run the original Gaussian design; smoke uses replicate 0 of every cell."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd
from paired_auc.simulation import (
    deterministic_frame_signature, enumerate_cells, gamma_for_cell,
    load_gamma0_table, load_protocol, run_cell_worker,
)
from run_simulation import normalize_frame_types, validate_cell_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/gaussian')
    parser.add_argument('--calibration', type=Path, default=ROOT / 'config/gaussian_gamma0.csv')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.mode == 'full':
        subprocess.run([sys.executable, str(ROOT / 'scripts/run_simulation.py'),
                        '--workers', str(args.workers), '--output-dir', str(args.output),
                        '--calibration', str(args.calibration)], check=True)
    else:
        protocol = load_protocol(ROOT / 'config/simulation_protocol.yaml')
        lookup = load_gamma0_table(args.calibration)
        payloads = [(protocol, asdict(c), [0], gamma_for_cell(lookup, c), 20260901)
                    for c in enumerate_cells(protocol)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            frames = []
            for payload, (rows, _) in zip(payloads, pool.map(run_cell_worker, payloads)):
                frame = normalize_frame_types(pd.DataFrame(rows))
                validate_cell_frame(frame, expected_cell_id=payload[1]['cell_id'],
                                    repetitions=1, simulation_base_seed=20260901)
                frames.append(frame)
        frame = pd.concat(frames, ignore_index=True)
        args.output.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(args.output / 'simulation_results.parquet', index=False)
        signature = deterministic_frame_signature(frame)
        reference = ROOT / 'results/summaries/gaussian_smoke_signature.json'
        if reference.exists():
            expected = json.loads(reference.read_text())['deterministic_sha256']
            if signature != expected:
                raise RuntimeError(f'Gaussian smoke reference mismatch: {signature} != {expected}')
        report = {'mode': 'smoke', 'cohorts': 162, 'analyses': 486,
                  'replicate_id': 0, 'deterministic_sha256': signature,
                  'reference_checked': reference.exists()}
        (args.output / 'smoke_check.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report))
    subprocess.run([sys.executable, str(ROOT / 'scripts/summarize_gaussian.py'),
                    '--input', str(args.output / 'simulation_results.parquet'),
                    '--mode', args.mode, '--output', str(args.output / 'artifacts')], check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
