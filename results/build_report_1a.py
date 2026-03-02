#!/usr/bin/env python3
"""Build the exp1a HTML report by reading JSONL result files directly."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# ── Config metadata ─────────────────────────────────────────────────────
CONFIGS_PATH = Path(__file__).resolve().parent.parent / 'helpers' / 'configs' / 'exp1a_configs.json'
RESULTS_DIR = Path(__file__).resolve().parent / 'exp1a'
REPORT_PATH = Path(__file__).resolve().parent / 'reports' / 'exp1a_report.html'

PRICING = {
    'claude-haiku-4-5-20251001':       (0.80,   4.00),
    'claude-sonnet-4-6':               (3.00,  15.00),
    'claude-opus-4-6':                (15.00,  75.00),
    'gemini-3-flash-preview':          (0.15,   0.60),
    'gemini-3-pro-preview':            (1.25,  10.00),
    'gpt-5-nano':                      (0.10,   0.40),
    'gpt-5-mini':                      (0.40,   1.60),
    'gpt-5.2':                         (2.50,  10.00),
    'deepseek-chat':                   (0.27,   1.10),
    'deepseek-reasoner':               (0.55,   2.19),
    'Qwen/Qwen2.5-7B-Instruct-Turbo': (0.18,   0.18),
    'Qwen/Qwen3-Next-80B-A3B-Instruct': (0.50, 0.50),
    'Qwen/Qwen3-235B-A22B-Thinking-2507': (3.50, 3.50),
    'gemma-3-27b-it':                  (0.10, 0.10),  # Free tier / minimal cost
    'claude-sonnet-4-20250514':        (3.00,  15.00),
}

MODEL_SHORT = {
    'claude-haiku-4-5-20251001': 'Haiku 4.5',
    'claude-sonnet-4-6': 'Sonnet 4.6',
    'claude-opus-4-6': 'Opus 4.6',
    'gemini-3-flash-preview': 'Gemini 3 Flash',
    'gemini-3-pro-preview': 'Gemini 3 Pro',
    'gpt-5-nano': 'GPT-5 nano',
    'gpt-5-mini': 'GPT-5 mini',
    'gpt-5.2': 'GPT-5.2',
    'deepseek-chat': 'DeepSeek V3',
    'deepseek-reasoner': 'DeepSeek R1',
    'Qwen/Qwen2.5-7B-Instruct-Turbo': 'Qwen 7B',
    'Qwen/Qwen3-Next-80B-A3B-Instruct': 'Qwen3 80B',
    'Qwen/Qwen3-235B-A22B-Thinking-2507': 'Qwen3 235B',
    'gemma-3-27b-it': 'Gemma 27B',
    'claude-sonnet-4-20250514': 'Sonnet 4.0',
}


def load_configs() -> dict[str, dict]:
    with open(CONFIGS_PATH) as f:
        configs = json.load(f)
    return {c['config_id']: c for c in configs}


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL, deduplicating by convo_id (last record wins)."""
    by_id: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                by_id[record['convo_id']] = record
    return list(by_id.values())


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = PRICING.get(model_id, (1.0, 1.0))
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


def parse_filename(name: str):
    """Parse 'hugo_1a_004_seed3' -> (domain, config_id, seed)."""
    m = re.match(r'^(\w+)_(1a_\d+)_seed(\d+)$', name)
    if not m:
        return None, None, None
    return m.group(1), m.group(2), int(m.group(3))


def compute_run_stats(convos: list[dict]) -> dict:
    """Compute accuracy stats from conversation-level JSONL records."""
    all_turns = []
    for c in convos:
        cat = c.get('category', 'unknown')
        for t in c.get('turns', []):
            t['_category'] = cat
            all_turns.append(t)

    if not all_turns:
        return {}

    correct = sum(1 for t in all_turns if t.get('correct'))
    total = len(all_turns)

    # By category
    by_cat = defaultdict(lambda: {'correct': 0, 'total': 0})
    for t in all_turns:
        cat = t['_category']
        by_cat[cat]['total'] += 1
        if t.get('correct'):
            by_cat[cat]['correct'] += 1

    # By turn
    by_turn = defaultdict(lambda: {'correct': 0, 'total': 0})
    for t in all_turns:
        tn = t.get('turn_num', 1)
        by_turn[tn]['total'] += 1
        if t.get('correct'):
            by_turn[tn]['correct'] += 1

    # Tokens and latency
    input_tokens = sum(t.get('input_tokens', 0) for t in all_turns)
    output_tokens = sum(t.get('output_tokens', 0) for t in all_turns)
    latencies = [t.get('latency_ms', 0) for t in all_turns if t.get('latency_ms')]
    latencies.sort()

    def percentile(arr, p):
        if not arr:
            return 0
        k = (len(arr) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(arr) else f
        return arr[f] + (k - f) * (arr[c] - arr[f])

    # Confusion pairs (for error analysis)
    confusions = []
    for t in all_turns:
        if not t.get('correct'):
            expected = t.get('flow', '')
            detected = t.get('detected_flows', [])
            if expected and detected:
                for d in detected:
                    confusions.append((expected, d))

    return {
        'accuracy': correct / total if total else 0,
        'total_turns': total,
        'categories': {
            cat: vals['correct'] / vals['total'] if vals['total'] else 0
            for cat, vals in by_cat.items()
        },
        'turns': {
            tn: vals['correct'] / vals['total'] if vals['total'] else 0
            for tn, vals in by_turn.items()
        },
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'latency_p50': int(percentile(latencies, 50)),
        'latency_p95': int(percentile(latencies, 95)),
        'confusions': confusions,
    }


def main():
    configs = load_configs()
    # All tiers we have data for
    target_tiers = {'low', 'medium', 'high'}

    # Collect all JSONL files
    runs = []  # list of (domain, config_id, seed, stats)
    all_confusions_hugo = Counter()
    all_confusions_dana = Counter()

    for jsonl_path in sorted(RESULTS_DIR.glob('*.jsonl')):
        domain, config_id, seed = parse_filename(jsonl_path.stem)
        if not domain or config_id not in configs:
            continue
        cfg = configs[config_id]
        if cfg.get('model_level') not in target_tiers:
            continue

        convos = load_jsonl(jsonl_path)
        stats = compute_run_stats(convos)
        if not stats:
            continue

        model_id = cfg['model_id']
        cost = estimate_cost(model_id, stats['input_tokens'], stats['output_tokens'])
        stats['cost'] = cost
        stats['model_id'] = model_id

        runs.append((domain, config_id, seed, stats))

        # Aggregate confusions
        target = all_confusions_hugo if domain == 'hugo' else all_confusions_dana
        for pair in stats['confusions']:
            target[pair] += 1

    # ── Build report data ──────────────────────────────────────────
    # Group by config_id
    by_config = defaultdict(lambda: {'hugo': [], 'dana': []})
    for domain, config_id, seed, stats in runs:
        by_config[config_id][domain].append((seed, stats))

    def std(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return math.sqrt(variance)

    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    # Ranking: average across seeds and domains, with STD
    ranking = []

    for config_id in sorted(by_config.keys(), key=lambda x: int(x.split('_')[1])):
        cfg = configs[config_id]
        model_id = cfg['model_id']
        model_short = MODEL_SHORT.get(model_id, model_id)
        tier = cfg.get('model_level', 'unknown')
        hugo_runs = by_config[config_id]['hugo']
        dana_runs = by_config[config_id]['dana']
        all_runs = hugo_runs + dana_runs

        if not all_runs:
            continue

        hugo_accs = [s['accuracy'] for _, s in hugo_runs]
        dana_accs = [s['accuracy'] for _, s in dana_runs]
        all_accs = [s['accuracy'] for _, s in all_runs]

        # Cost and latency (average per run)
        all_costs = [s['cost'] for _, s in all_runs]
        all_lat = [s['latency_p50'] for _, s in all_runs]
        seed_count = len(set(seed for seed, _ in all_runs))

        # Category averages + STD (across ALL runs, both domains)
        def cat_avg_std(cat_name):
            vals = [s['categories'].get(cat_name, 0) for _, s in all_runs]
            return avg(vals), std(vals)

        # Turn averages + STD (across ALL runs, both domains)
        def turn_avg_std(turn_num):
            vals = [s['turns'].get(turn_num, 0) for _, s in all_runs]
            return avg(vals), std(vals)

        sf_avg, sf_std = cat_avg_std('same_flow')
        sw_avg, sw_std = cat_avg_std('switch_flow')
        a1_avg, a1_std = cat_avg_std('ambiguous_first')
        a2_avg, a2_std = cat_avg_std('ambiguous_second')
        t1_avg, t1_std = turn_avg_std(1)
        t3_avg, t3_std = turn_avg_std(3)

        ranking.append({
            'config_id': config_id,
            'model_short': model_short,
            'provider': cfg.get('provider', 'unknown'),
            'tier': tier,
            'seed_count': seed_count,
            'accuracy': avg(all_accs),
            'std': std(all_accs),
            'hugo_acc': avg(hugo_accs),
            'hugo_std': std(hugo_accs),
            'dana_acc': avg(dana_accs),
            'dana_std': std(dana_accs),
            'cost_per_run': avg(all_costs),
            'latency_p50': int(avg(all_lat)),
            # Category breakdown (averaged across both domains)
            'cat_same_flow': sf_avg, 'cat_same_flow_std': sf_std,
            'cat_switch_flow': sw_avg, 'cat_switch_flow_std': sw_std,
            'cat_ambiguous_first': a1_avg, 'cat_ambiguous_first_std': a1_std,
            'cat_ambiguous_second': a2_avg, 'cat_ambiguous_second_std': a2_std,
            # Turn breakdown (averaged across both domains)
            'turn_1': t1_avg, 'turn_1_std': t1_std,
            'turn_3': t3_avg, 'turn_3_std': t3_std,
        })

    # Sort ranking by accuracy descending
    ranking.sort(key=lambda x: x['accuracy'], reverse=True)

    # Confusion pairs (top 15 per domain)
    confusions_hugo = []
    confusions_dana = []
    hugo_conf_by_model = defaultdict(lambda: Counter())
    dana_conf_by_model = defaultdict(lambda: Counter())
    for domain, config_id, seed, stats in runs:
        cfg = configs[config_id]
        model_short = MODEL_SHORT.get(cfg['model_id'], cfg['model_id'])
        target = hugo_conf_by_model if domain == 'hugo' else dana_conf_by_model
        for pair in stats['confusions']:
            target[pair][model_short] += 1

    for counter, result_list in [
        (all_confusions_hugo, confusions_hugo),
        (all_confusions_dana, confusions_dana),
    ]:
        by_model = hugo_conf_by_model if result_list is confusions_hugo else dana_conf_by_model
        for (exp, pred), count in counter.most_common(15):
            models = by_model[(exp, pred)]
            result_list.append({
                'expected': exp,
                'predicted': pred,
                'count': count,
                'models': ', '.join(f'{m}({c})' for m, c in models.most_common(3)),
            })

    # Status table (all low+mid configs)
    status = []
    for config_id in sorted(configs.keys(), key=lambda x: int(x.split('_')[1])):
        cfg = configs[config_id]
        if cfg.get('model_level') not in target_tiers:
            continue
        model_short = MODEL_SHORT.get(cfg['model_id'], cfg['model_id'])
        hugo_seeds = len(by_config[config_id]['hugo'])
        dana_seeds = len(by_config[config_id]['dana'])
        complete = hugo_seeds == 5 and dana_seeds == 5
        partial = hugo_seeds > 0 or dana_seeds > 0
        status.append({
            'config_id': config_id,
            'model_short': model_short,
            'tier': cfg.get('model_level', ''),
            'hugo_seeds': hugo_seeds,
            'dana_seeds': dana_seeds,
            'complete': complete,
            'partial': partial and not complete,
        })

    # Assemble
    total_cost = sum(s['cost'] for _, _, _, s in runs)
    best = ranking[0] if ranking else {}

    report_data = {
        'generated': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'total_runs': len(runs),
        'unique_configs': len(by_config),
        'domains': ['Hugo', 'Dana'],
        'best_accuracy': best.get('accuracy', 0),
        'best_model': best.get('model_short', '?'),
        'total_cost': total_cost,
        'ranking': ranking,
        'confusions_hugo': confusions_hugo,
        'confusions_dana': confusions_dana,
        'status': status,
    }

    # Embed into HTML — replace the DATA assignment line
    html = REPORT_PATH.read_text()
    data_json = json.dumps(report_data, indent=None)
    # Match the entire DATA assignment (handles both placeholder and previously embedded data)
    html = re.sub(
        r'const DATA = .*?;',
        f'const DATA = {data_json};',
        html,
        count=1,
        flags=re.DOTALL,
    )
    REPORT_PATH.write_text(html)
    print(f'Report written to {REPORT_PATH}')
    print(f'  {len(runs)} runs, {len(by_config)} configs, best={best.get("model_short")} @ {best.get("accuracy", 0):.1%}')


if __name__ == '__main__':
    main()
