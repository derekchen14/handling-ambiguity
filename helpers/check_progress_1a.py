#!/usr/bin/env python3
"""Check exp1a trial run progress — box-drawing table, provider x domain rows, tier columns."""

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / 'results' / 'exp1a'
CONFIGS = Path(__file__).resolve().parent / 'configs' / 'exp1a_configs.json'
DOMAINS = ['hugo', 'dana']
SEEDS = [1, 2, 3, 4, 5]
CONVOS_PER_SEED = 128

PROVIDER_ORDER = ['anthropic', 'openai', 'google', 'qwen', 'deepseek']
PROVIDER_LABELS = {
    'anthropic': 'Anthropic',
    'openai': 'OpenAI',
    'google': 'Google',
    'qwen': 'Qwen',
    'deepseek': 'DeepSeek',
}
TIER_ORDER = ['low', 'medium', 'high']

MODEL_SHORT = {
    'claude-haiku-4-5-20251001': 'Haiku',
    'claude-sonnet-4-6': 'Sonnet',
    'claude-opus-4-6': 'Opus',
    'gemini-3-flash-preview': 'Flash',
    'gemini-3-pro-preview': 'Gem Pro',
    'gpt-5-nano': 'Nano',
    'gpt-5-mini': 'Mini',
    'gpt-5.2': 'GPT-5.2',
    'deepseek-chat': 'DS Chat',
    'deepseek-reasoner': 'DS R1',
    'Qwen/Qwen2.5-7B-Instruct-Turbo': 'Q 7B',
    'Qwen/Qwen3-Next-80B-A3B-Instruct': 'Q3 80B',
    'Qwen/Qwen3-235B-A22B-Thinking-2507': 'Q3 235B',
    'gemma-3-27b-it': 'Gemma',
}


def load_configs():
    with open(CONFIGS) as f:
        configs = json.load(f)
    for c in configs:
        if c['provider'] == 'gemma':
            c['provider'] = 'google'
    return configs


def count_convos(domain, config_id, seed):
    """Count conversations from the JSONL file directly."""
    jsonl = RESULTS / f'{domain}_{config_id}_seed{seed}.jsonl'
    if jsonl.exists():
        return sum(1 for _ in open(jsonl))
    return 0


def cell_value(domain, cfg):
    """Return formatted progress string for one (domain, config) cell."""
    if cfg is None:
        return '—'
    total = 0
    expected = len(SEEDS) * CONVOS_PER_SEED  # 640
    for seed in SEEDS:
        total += count_convos(domain, cfg['config_id'], seed)
    if total == 0:
        return '-'
    pct = total / expected * 100
    return f'{total}/{expected} ({pct:.0f}%)'


def main():
    configs = load_configs()
    lookup = {}
    for c in configs:
        lookup[(c['provider'], c['model_level'])] = c

    # Build all rows: (provider_label, domain, low_val, med_val, high_val)
    rows = []
    for provider in PROVIDER_ORDER:
        for di, domain in enumerate(DOMAINS):
            plabel = PROVIDER_LABELS[provider] if di == 0 else ''
            low = cell_value(domain, lookup.get((provider, 'low')))
            med = cell_value(domain, lookup.get((provider, 'medium')))
            high = cell_value(domain, lookup.get((provider, 'high')))
            rows.append((plabel, domain, low, med, high))

    # Compute column widths
    headers = ['Provider', 'Domain', 'Low', 'Medium', 'High']
    widths = [len(h) + 2 for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val) + 2)

    def hline(left, mid, right, fill='─'):
        return '  ' + left + mid.join(fill * w for w in widths) + right

    def dataline(vals):
        cells = []
        for i, val in enumerate(vals):
            cells.append(f' {val:<{widths[i]-1}}')
        return '  │' + '│'.join(cells) + '│'

    print()
    print(hline('┌', '┬', '┐'))
    print(dataline(headers))
    print(hline('├', '┼', '┤'))

    for i, row in enumerate(rows):
        print(dataline(row))
        # Separator between providers (every 2 rows), but not after last
        if i % 2 == 1 and i < len(rows) - 1:
            print(hline('├', '┼', '┤'))

    print(hline('└', '┴', '┘'))
    print()


if __name__ == '__main__':
    main()
