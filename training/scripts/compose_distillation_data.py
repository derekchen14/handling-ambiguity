"""Generate ensemble pseudo-label JSONLs for distillation SFT.

Both domains compose a 10v-1 ensemble from exp1a predictions via
EnsembleBootstrapper.

Output:
    training/distill_data/hugo_10v-1.jsonl
    training/distill_data/dana_10v-1.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

from helpers.bootstrap import (
    RESULTS_DIR,
    EnsembleBootstrapper,
    load_eval_sets,
)

VOTER_IDS = [
    '1a_002',
    '1a_004',
    '1a_006',
    '1a_008',
    '1a_010',
    '1a_016',
    '1a_020',
    '1a_021',
    '1a_023',
    '1a_024',
]

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'distill_data'


def main() -> None:
    eval_sets = load_eval_sets()
    bootstrapper = EnsembleBootstrapper()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for domain, eval_set in eval_sets.items():
        preds = bootstrapper.load_predictions(RESULTS_DIR, domain)
        results = bootstrapper.compose_ensemble(
            preds, VOTER_IDS, seed=1, eval_set=eval_set,
        )

        out_path = OUTPUT_DIR / f'{domain}_10v-1.jsonl'
        with open(out_path, 'w') as f:
            for item in results:
                f.write(json.dumps(item) + '\n')

        print(f'{domain}: wrote {len(results)} lines → {out_path}')


if __name__ == '__main__':
    main()
