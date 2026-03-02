#!/usr/bin/env python3
"""Check exp2a progress — staged NLU funnel (intent, flow, slot, tool) across 8 configs.

Each run = 1 config × 1 domain × 1 seed = 128 conversations × 2 user turns = 256 API calls.
4 modes × 8 configs × 2 domains × 3 seeds = 192 runs.
Flow mode is bootstrapped from Exp 1A (0 API calls). Other 3 modes = 144 runs = 36,864 API calls.
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIGS_PATH = BASE_DIR / 'helpers' / 'configs' / 'exp2_configs.json'
EVAL_DIR = BASE_DIR / 'datasets'
RESULTS_BASE = BASE_DIR / 'results' / 'exp2a'

DOMAINS = ['hugo', 'dana']
SEEDS = [1, 2, 3]

MODE_DIRS = {
    'intent': 'intents',
    'flow':   'flows',
    'slot':   'slots',
    'tool':   'tools',
}

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
        eval_path = EVAL_DIR / domain / 'eval_set.json'
        if eval_path.exists():
            with open(eval_path) as f:
                counts[domain] = len(json.load(f))
        else:
            counts[domain] = 128
    return counts


def count_convos(results_dir: Path, domain: str, config_id: str, seed: int) -> int:
    jsonl = results_dir / f'{domain}_{config_id}_seed{seed}.jsonl'
    if not jsonl.exists():
        return 0
    with open(jsonl) as f:
        return sum(1 for line in f if line.strip())


def cell_value(results_dir: Path, domain: str, config_id: str, expected_per_seed: int) -> str:
    total = 0
    expected = len(SEEDS) * expected_per_seed
    for seed in SEEDS:
        total += count_convos(results_dir, domain, config_id, seed)
    if total == 0:
        return '-'
    if total >= expected:
        return f'{total}/{expected} (100%)'
    pct = total / expected * 100
    return f'{total}/{expected} ({pct:.0f}%)'


def print_table(title: str, headers: list[str], rows: list[tuple], group_size: int = 2):
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

    # Build column headers from config order
    col_headers = []
    for cid in CONFIG_ORDER:
        if cid in configs:
            model_id = configs[cid]['model_id']
            col_headers.append(MODEL_SHORT.get(model_id, model_id[:8]))
        else:
            col_headers.append(cid)

    headers = ['Mode', 'Domain'] + col_headers
    rows = []

    # Flow mode is bootstrapped from Exp 1A (0 API calls)
    API_MODES = {k for k in MODE_DIRS if k != 'flow'}

    total_runs = len(CONFIG_ORDER) * len(DOMAINS) * len(SEEDS) * len(MODE_DIRS)
    completed_runs = 0
    total_convos = 0
    expected_convos = 0
    api_convos = 0
    api_expected = 0

    for mode, subdir in MODE_DIRS.items():
        results_dir = RESULTS_BASE / subdir
        for di, domain in enumerate(DOMAINS):
            mlabel = mode.capitalize() if di == 0 else ''
            expected_per_seed = convos_per_domain.get(domain, 128)
            vals = []
            for cid in CONFIG_ORDER:
                vals.append(cell_value(results_dir, domain, cid, expected_per_seed))
                for seed in SEEDS:
                    n = count_convos(results_dir, domain, cid, seed)
                    total_convos += n
                    expected_convos += expected_per_seed
                    if mode in API_MODES:
                        api_convos += n
                        api_expected += expected_per_seed
                    if n >= expected_per_seed:
                        completed_runs += 1
            rows.append((mlabel, domain, *vals))

    print_table(
        f'EXP2A — Staged NLU Funnel   [{total_runs} runs target]',
        headers, rows,
    )

    pct = completed_runs / total_runs * 100 if total_runs else 0
    print(f'  Runs: {completed_runs}/{total_runs} ({pct:.0f}%) | '
          f'API calls: {api_convos * 2:,}/{api_expected * 2:,} (flow mode bootstrapped, 0 API)')
    print()


if __name__ == '__main__':
    main()
