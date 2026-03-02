#!/usr/bin/env python3
"""Check exp1b progress — self-consistency runs (raw API call results only)."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_1B = BASE_DIR / 'results' / 'exp1b'

DOMAINS = ['dana', 'hugo']
CONVOS_PER_SEED = 128

# All runs that require API calls
RUNS = {
    '3v-1': ('Opus @ t=0.3',   [1, 2, 3]),
    '3v-2': ('Sonnet @ t=0.3', [1, 2, 3]),
    '3v-3': ('Flash @ t=0.3',  [1, 2, 3]),
    '3v-8': ('Sonnet @ t=0.6', [1]),
}


def count_lines(fpath: Path) -> int:
    if fpath.exists():
        return sum(1 for _ in open(fpath))
    return 0


def cell(domain: str, ens_id: str, seed: int) -> str:
    n = count_lines(RESULTS_1B / f'{domain}_{ens_id}_seed{seed}.jsonl')
    if n == 0:
        return '-'
    if n >= CONVOS_PER_SEED:
        return f'{n}'
    return f'{n}/{CONVOS_PER_SEED}'


def print_table(headers, rows, group_size=2):
    widths = [len(h) + 2 for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)) + 2)

    def hline(left, mid, right, fill='\u2500'):
        return '  ' + left + mid.join(fill * w for w in widths) + right

    def dataline(vals):
        cells = [f' {str(v):<{widths[i]-1}}' for i, v in enumerate(vals)]
        return '  \u2502' + '\u2502'.join(cells) + '\u2502'

    print()
    print(hline('\u250c', '\u252c', '\u2510'))
    print(dataline(headers))
    print(hline('\u251c', '\u253c', '\u2524'))

    for i, row in enumerate(rows):
        print(dataline(row))
        if group_size and i % group_size == group_size - 1 and i < len(rows) - 1:
            print(hline('\u251c', '\u253c', '\u2524'))

    print(hline('\u2514', '\u2534', '\u2518'))


def main():
    print('\n  EXP1B RUNS (raw API call results)')
    headers = ['Model', 'Domain', 'Seed 1', 'Seed 2', 'Seed 3', 'Done']
    rows = []
    for ens_id, (desc, seeds) in RUNS.items():
        for di, domain in enumerate(DOMAINS):
            label = f'{ens_id} ({desc})' if di == 0 else ''
            cells = [cell(domain, ens_id, s) if s in seeds else '' for s in [1, 2, 3]]
            done_count = sum(
                1 for s in seeds
                if count_lines(RESULTS_1B / f'{domain}_{ens_id}_seed{s}.jsonl') >= CONVOS_PER_SEED
            )
            done_str = f'{done_count}/{len(seeds)}'
            rows.append((label, domain, *cells, done_str))

    print_table(headers, rows)
    print()


if __name__ == '__main__':
    main()
