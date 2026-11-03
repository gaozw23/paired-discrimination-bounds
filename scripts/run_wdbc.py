#!/usr/bin/env python
"""Reproduce the WDBC fixed-cohort semi-synthetic experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paired_auc.wdbc import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--protocol", type=Path, default=ROOT / "config/wdbc_protocol.json")
    parser.add_argument("--output", type=Path, required=True, help="fresh output directory")
    parser.add_argument("--no-figures", action="store_true", help="skip PNG/PDF figure rendering")
    args = parser.parse_args()
    result = run(args.protocol, args.output, args.mode, ROOT / "results/summaries",
                 figures=not args.no_figures)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
