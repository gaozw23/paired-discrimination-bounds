"""Rebuild original evaluation cohorts and external-prevalence A2 results."""
import argparse
import json
from pathlib import Path

from paired_auc.a2_reproduction import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('smoke','full'), default='smoke')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--all-k', action='store_true', help='All 1202 K values for the nine smoke cohorts')
    parser.add_argument('--figures', action='store_true', help='Requires full or --all-k integration')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.figures and args.mode != 'full' and not args.all_k:
        parser.error('--figures requires --all-k or --mode full')
    repository = Path(__file__).resolve().parents[1]
    result = run(repository,args.output,args.mode,args.workers,args.all_k)
    if args.figures:
        from paired_auc.a2_figures import make_figures
        make_figures(args.output)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
