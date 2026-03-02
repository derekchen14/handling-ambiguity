#!/usr/bin/env python3
"""Check exp2b progress — direct tool-calling across 8 configs.

Each run = 1 config × 1 domain × 1 seed = 128 conversations × 2 user turns = 256 API calls.
8 configs × 2 domains × 3 seeds = 48 runs = 12,288 API calls.
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIGS_PATH = BASE_DIR / 'helpers' / 'configs' / 'exp2_configs.json'
EVAL_DIR = BASE_DIR / 'eval'
RESULTS_DIR = BASE_DIR / 'results' / 'exp2b'

DOMAINS = ['hugo', 'dana']
SEEDS = [1, 2, 3]

# Ordered by tier then provider — low → mid → high
CONFIG_ORDER = ['2_001', '2_002', '2_008', '2_004', '2_006', '2_024', '2_021', '2_010']

MODEL_SHORT = {
    'claude-haiku-4-5-20251001': 'Haiku',
    'claude-sonnet-4-6': 'Sonnet',
    'claude-opus-4-6': 'Opus',
    'gemini-3-flash-preview': 'Flash',
    'gemini-3-pro-preview': 'Gem 3.1',
    'gpt-5-mini': 'Mini',
    'deepseek-reasoner': 'DS R1',
    'Qwen/Qwen3-235B-A22B-Thinking-2507': 'Q3 235B',
}


def load_configs() -> dict[str, dict]:
    if not CONFIGS_PATH.exists():
        print(f'  Config file not found: {CONFIGS_PATH}', file=sys.stderr)
        sys.exit(1)
    with open(CONFIGS_PATH) as f:
        return {c['config_id']: c for c in json.load(f)}


def get_convos_per_domain() -> dict[str, int]:
    counts = {}
    for domain in DOMAINS:
        eval_path = EVAL_DIR / f'eval_{domain}.json'
        if eval_path.exists():
            with open(eval_path) as f:
                counts[domain] = len(json.load(f))
        else:
            counts[domain] = 128
    return counts


def count_convos(domain: str, config_id: str, seed: int) -> int:
    jsonl = RESULTS_DIR / f'{domain}_{config_id}_seed{seed}.jsonl'
    if not jsonl.exists():
        return 0
    with open(jsonl) as f:
        return sum(1 for line in f if line.strip())


def cell_value(domain: str, config_id: str, expected_per_seed: int) -> str:
    total = 0
    expected = len(SEEDS) * expected_per_seed
    for seed in SEEDS:
        total += count_convos(domain, config_id, seed)
    if total == 0:
        return '-'
    if total >= expected:
        return f'{total}'
    pct = total / expected * 100
    return f'{total}/{expected} ({pct:.0f}%)'


def print_table(title: str, headers: list[str], rows: list[tuple], group_size: int = 0):
    widths = [len(h) + 2 for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)) + 2)

    def hline(left, mid, right, fill='─'):
        return '  ' + left + mid.join(fill * w for w in widths) + right

    def dataline(vals):
        cells = [f' {str(v):<{widths[i]-1}}' for i, v in enumerate(vals)]
        return '  │' + '│'.join(cells) + '│'

    print(f'\n  {title}')
    print(hline('┌', '┬', '┐'))
    print(dataline(headers))
    print(hline('├', '┼', '┤'))

    for i, row in enumerate(rows):
        print(dataline(row))
        if group_size and i % group_size == group_size - 1 and i < len(rows) - 1:
            print(hline('├', '┼', '┤'))

    print(hline('└', '┴', '┘'))


def main():
    configs = load_configs()
    convos_per_domain = get_convos_per_domain()

    col_headers = []
    for cid in CONFIG_ORDER:
        if cid in configs:
            model_id = configs[cid]['model_id']
            col_headers.append(MODEL_SHORT.get(model_id, model_id[:8]))
        else:
            col_headers.append(cid)

    headers = ['Domain'] + col_headers
    rows = []

    total_runs = len(CONFIG_ORDER) * len(DOMAINS) * len(SEEDS)
    completed_runs = 0
    total_convos = 0
    expected_convos = 0

    for domain in DOMAINS:
        expected_per_seed = convos_per_domain.get(domain, 128)
        vals = []
        for cid in CONFIG_ORDER:
            vals.append(cell_value(domain, cid, expected_per_seed))
            for seed in SEEDS:
                n = count_convos(domain, cid, seed)
                total_convos += n
                expected_convos += expected_per_seed
                if n >= expected_per_seed:
                    completed_runs += 1
        rows.append((domain, *vals))

    print_table(
        f'EXP2B — Direct Tool-Calling   [{total_runs} runs target]',
        headers, rows,
    )

    api_done = total_convos * 2
    api_target = expected_convos * 2
    pct = completed_runs / total_runs * 100 if total_runs else 0
    print(f'  Runs: {completed_runs}/{total_runs} ({pct:.0f}%) | '
          f'API calls: {api_done:,}/{api_target:,}')
    print()


if __name__ == '__main__':
    main()
