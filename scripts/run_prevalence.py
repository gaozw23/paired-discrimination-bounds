"""Evaluate a continuous empirical prevalence envelope from JSON or a toy case."""
import argparse
import json
from pathlib import Path

from paired_auc.prevalence import prevalence_envelope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='JSON with scores_a, scores_b, verified, verified_labels, intervals')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.input:
        data = json.loads(args.input.read_text('utf-8'))
    else:
        data = dict(scores_a=[7,1,2,5,3,6,4,8], scores_b=[7,1,3,6,8,4,2,5],
                    verified=[1,0,1,0,1,1,0,0], verified_labels=[1,0,0,1],
                    intervals=[[0.25,0.75]])
    result = prevalence_envelope(**data)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
        handle.write('\n')
    print(json.dumps({key:result[key] for key in ('status','lower','upper','width','sign')}))


if __name__ == '__main__':
    main()
