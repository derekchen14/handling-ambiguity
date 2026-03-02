#!/usr/bin/env python3
"""CLI entry point for synthetic data augmentation of multi-turn conversations.

Usage:
    # Augment pilot batch (8 per category) for Hugo
    python augment_data.py --domain hugo --pilot

    # Augment all 128 convos for Hugo
    python augment_data.py --domain hugo --all

    # Augment just category C for Dana
    python augment_data.py --domain dana --category c

    # Regenerate specific convos that failed validation
    python augment_data.py --domain hugo --regenerate hugo_c_005,hugo_c_012

    # Validate existing output
    python augment_data.py --domain hugo --validate-only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.data_aug.generator import generate_batch
from experiments.data_aug.validator import validate_file

log = logging.getLogger(__name__)

EVAL_DIR = str(Path(__file__).parent.parent / 'eval')


def main():
    parser = argparse.ArgumentParser(
        description='Synthetic data augmentation for multi-turn conversation eval data',
    )
    parser.add_argument(
        '--domain',
        required=True,
        choices=['hugo', 'dana'],
        help='Domain to generate for',
    )
    parser.add_argument(
        '--category',
        choices=['a', 'b', 'c', 'd'],
        help='Generate a specific category only',
    )
    parser.add_argument(
        '--pilot',
        action='store_true',
        help='Pilot mode: generate 8 per category (instead of 32)',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        dest='generate_all',
        help='Generate all 128 conversations (32 per category)',
    )
    parser.add_argument(
        '--regenerate',
        type=str,
        help='Comma-separated convo_ids to regenerate',
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Validate existing output without generating',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling (default: 42)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=EVAL_DIR,
        help=f'Output directory (default: {EVAL_DIR})',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging',
    )

    args = parser.parse_args()

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    domain = args.domain

    # ── Validate only ──────────────────────────────────────────────
    if args.validate_only:
        json_path = Path(args.output_dir) / f'gen_{domain}.json'
        if not json_path.exists():
            print(f'No eval file found for {domain} at {json_path}')
            sys.exit(1)

        print(f'Validating {json_path}...')
        result = validate_file(json_path, domain)

        print(f'\nResults: {result["passed"]}/{result["total"]} passed '
              f'({result["pass_rate"]:.1%})')

        if result['failed_ids']:
            print(f'\nFailed conversations ({result["failed"]}):')
            for r in result['per_convo']:
                if not r['passed']:
                    print(f'  {r["convo_id"]}:')
                    for issue in r['issues']:
                        print(f'    - {issue}')

        if result['uniqueness_issues']:
            print(f'\nUniqueness issues ({len(result["uniqueness_issues"])}):')
            for issue in result['uniqueness_issues'][:20]:
                print(f'  - {issue}')
            if len(result['uniqueness_issues']) > 20:
                print(f'  ... and {len(result["uniqueness_issues"]) - 20} more')

        sys.exit(0 if result['pass_rate'] > 0.95 else 1)

    # ── Generate ───────────────────────────────────────────────────
    categories = None
    n_per_cat = 32

    if args.category:
        categories = [args.category]
    elif args.pilot:
        n_per_cat = 8
    elif args.generate_all:
        pass  # defaults: all categories, 32 per cat
    elif args.regenerate:
        # Parse specific IDs
        specific_ids = [s.strip() for s in args.regenerate.split(',')]
        print(f'Regenerating {len(specific_ids)} conversations: {specific_ids}')
        results = generate_batch(
            domain=domain,
            n_per_cat=n_per_cat,
            workers=args.workers,
            seed=args.seed,
            output_dir=args.output_dir,
            specific_ids=specific_ids,
        )
        print(f'Total conversations: {len(results)}')
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)

    print(f'Generating {n_per_cat} per category for {domain} '
          f'(categories: {categories or "all"}, workers: {args.workers})')

    results = generate_batch(
        domain=domain,
        categories=categories,
        n_per_cat=n_per_cat,
        workers=args.workers,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print(f'\nTotal conversations: {len(results)}')

    # Run validation on results
    json_path = Path(args.output_dir) / f'gen_{domain}.json'
    if json_path.exists():
        result = validate_file(json_path, domain)
        print(f'Validation: {result["passed"]}/{result["total"]} passed '
              f'({result["pass_rate"]:.1%})')

        if result['failed_ids']:
            print(f'Failed: {result["failed_ids"]}')
            print('Run with --regenerate to fix failures')


if __name__ == '__main__':
    main()
