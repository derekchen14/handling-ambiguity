"""One-time migration: unwrap fuzzy wrappers from eval set params.

Transforms every {"value": X, "fuzzy": true} → X in target_tools.
Keeps structured objects (dicts without "fuzzy" key) as-is.

Run:  python scripts/migrate_eval_params.py [--dry-run]

Writes back to the same files.  Review the git diff before committing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def unwrap_fuzzy(value):
    """If value is a fuzzy wrapper {"value": X, "fuzzy": true}, return X."""
    if isinstance(value, dict) and 'fuzzy' in value:
        return value.get('value')
    return value


def unwrap_tool_params(target_tools: dict) -> dict:
    """Unwrap all fuzzy wrappers in a single turn's target_tools."""
    for tool_name, params in target_tools.items():
        if not isinstance(params, dict):
            continue
        for param_name, val in params.items():
            params[param_name] = unwrap_fuzzy(val)
    return target_tools


def unwrap_eval_set(eval_set: list[dict]) -> tuple[list[dict], int]:
    """Unwrap all fuzzy wrappers in an eval set. Returns (eval_set, count)."""
    count = 0
    for convo in eval_set:
        for turn in convo.get('turns', []):
            if 'target_tools' in turn:
                for tool_name, params in turn['target_tools'].items():
                    if not isinstance(params, dict):
                        continue
                    for param_name, val in params.items():
                        if isinstance(val, dict) and 'fuzzy' in val:
                            count += 1
                unwrap_tool_params(turn['target_tools'])
    return eval_set, count


def main():
    parser = argparse.ArgumentParser(description='Unwrap fuzzy wrappers from eval set params')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print changes without writing')
    args = parser.parse_args()

    for domain in ('hugo', 'dana'):
        path = ROOT / 'datasets' / domain / 'eval_set.json'
        print(f'Processing {path}...')

        with open(path) as f:
            eval_set = json.load(f)

        migrated, count = unwrap_eval_set(eval_set)
        print(f'  Unwrapped {count} fuzzy wrappers')

        if args.dry_run:
            print(f'  Would write {len(migrated)} conversations')
        else:
            with open(path, 'w') as f:
                json.dump(migrated, f, indent=2, ensure_ascii=False)
                f.write('\n')
            print(f'  Wrote {len(migrated)} conversations')

    print('Done.')


if __name__ == '__main__':
    main()
