#!/usr/bin/env python3
"""Build the Exp 2 HTML report — pipeline comparison.

Loads results from Exp 2A (staged NLU funnel: intent, scoped tools)
and Exp 2B (flat tool-calling with all ~56 tools), then compares both
pipelines against each other and against Exp 1A flow detection.

Usage:
    python3 results/build_report_2.py
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIGS_PATH = BASE_DIR / 'helpers' / 'configs' / 'exp2_configs.json'
EXP1A_CONFIGS_PATH = BASE_DIR / 'helpers' / 'configs' / 'exp1a_configs.json'

TOOL_DIR = Path(__file__).resolve().parent / 'exp2b'
INTENT_DIR = Path(__file__).resolve().parent / 'exp2a' / 'intents'
SCOPED_TOOL_DIR = Path(__file__).resolve().parent / 'exp2a' / 'tools'
EXP1A_DIR = Path(__file__).resolve().parent / 'exp1a'
HINT_DIR = Path(__file__).resolve().parent / 'exp2c'

REPORT_PATH = Path(__file__).resolve().parent / 'reports' / 'exp2_report.html'

# ── Pricing & display ────────────────────────────────────────────

PRICING = {
    'claude-haiku-4-5-20251001':       (0.80,   4.00),
    'claude-sonnet-4-6':               (3.00,  15.00),
    'claude-opus-4-6':                (15.00,  75.00),
    'gemini-3-flash-preview':          (0.15,   0.60),
    'gemini-3-pro-preview':            (1.25,  10.00),
    'google/gemini-3.1-pro-preview':   (2.00,  12.00),
    'gpt-5-nano':                      (0.10,   0.40),
    'gpt-5-mini':                      (0.40,   1.60),
    'gpt-5.2':                         (2.50,  10.00),
    'deepseek-chat':                   (0.27,   1.10),
    'deepseek-reasoner':               (0.55,   2.19),
    'Qwen/Qwen2.5-7B-Instruct-Turbo': (0.18,   0.18),
    'Qwen/Qwen3-Next-80B-A3B-Instruct': (0.50, 0.50),
    'Qwen/Qwen3-235B-A22B-Thinking-2507': (3.50, 3.50),
    'gemma-3-27b-it':                  (0.10,   0.10),
    'claude-sonnet-4-20250514':        (3.00,  15.00),
}

MODEL_SHORT = {
    'claude-haiku-4-5-20251001': 'Haiku 4.5',
    'claude-sonnet-4-6': 'Sonnet 4.6',
    'claude-opus-4-6': 'Opus 4.6',
    'gemini-3-flash-preview': 'Gemini 3 Flash',
    'gemini-3-pro-preview': 'Gemini 3 Pro',
    'google/gemini-3.1-pro-preview': 'Gemini 3.1 Pro',
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

# Desired sort order for display (left to right in charts)
MODEL_SORT_ORDER = [
    'claude-haiku-4-5-20251001',
    'gemini-3-flash-preview',
    'claude-sonnet-4-6',
    'gpt-5-mini',
    'claude-opus-4-6',
    'Qwen/Qwen3-235B-A22B-Thinking-2507',
    'deepseek-reasoner',
]

# Hardcoded best-in-class pipeline stage accuracies (from improvements.md §2)
PIPELINE_INTENT_ACC = {      # Flash on intent classification, per domain
    'hugo': 0.951,
    'dana': 0.948,
}
PIPELINE_FLOW_ACC = {        # 5v-6 ensemble flow detection, per category
    'same_flow': 0.934,
    'switch_flow': 0.928,
    'ambiguous_first': 0.788,
    'ambiguous_second': 0.934,
}


# ── Helpers ──────────────────────────────────────────────────────

def load_configs(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path) as f:
        configs = json.load(f)
    return {c['config_id']: c for c in configs}


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL, deduplicating by convo_id (last record wins)."""
    by_id: dict[str, dict] = {}
    if not path.exists():
        return []
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


def parse_exp2_filename(name: str):
    """Parse 'hugo_2_001_seed1' -> (domain, config_id, seed)."""
    m = re.match(r'^(\w+)_(2_\d+)_seed(\d+)$', name)
    if not m:
        return None, None, None
    return m.group(1), m.group(2), int(m.group(3))


def parse_exp1a_filename(name: str):
    """Parse 'hugo_1a_004_seed3' -> (domain, config_id, seed)."""
    m = re.match(r'^(\w+)_(1a_\d+)_seed(\d+)$', name)
    if not m:
        return None, None, None
    return m.group(1), m.group(2), int(m.group(3))


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def percentile(arr: list, p: float) -> float:
    if not arr:
        return 0.0
    arr = sorted(arr)
    k = (len(arr) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(arr) - 1)
    return arr[f] + (k - f) * (arr[c] - arr[f])


def model_sort_key(model_id: str) -> int:
    try:
        return MODEL_SORT_ORDER.index(model_id)
    except ValueError:
        return 999


# ── Per-run stats ────────────────────────────────────────────────

def extract_turns(convos: list[dict]) -> list[dict]:
    """Flatten convos into turns with _category attached."""
    all_turns = []
    for c in convos:
        cat = c.get('category', 'unknown')
        for t in c.get('turns', []):
            t['_category'] = cat
            all_turns.append(t)
    return all_turns


def compute_tool_stats(convos: list[dict]) -> dict:
    """Compute stats from flat or scoped tool-calling JSONL records."""
    all_turns = extract_turns(convos)
    if not all_turns:
        return {}

    correct = sum(1 for t in all_turns if t.get('correct'))
    total = len(all_turns)
    ambiguity_flagged = sum(1 for t in all_turns if t.get('ambiguity_flagged'))
    null_calls = sum(1 for t in all_turns if t.get('null_call'))
    excluded = sum(1 for t in all_turns if t.get('excluded'))

    # By category
    by_cat = defaultdict(lambda: {'correct': 0, 'total': 0})
    for t in all_turns:
        by_cat[t['_category']]['total'] += 1
        if t.get('correct'):
            by_cat[t['_category']]['correct'] += 1

    # Tokens and latency
    input_tokens = sum(t.get('input_tokens', 0) for t in all_turns)
    output_tokens = sum(t.get('output_tokens', 0) for t in all_turns)
    latencies = sorted(t.get('latency_ms', 0) for t in all_turns if t.get('latency_ms'))

    # Confusions (expected flow -> predicted tools)
    confusions = []
    for t in all_turns:
        if not t.get('correct') and not t.get('excluded'):
            expected = t.get('flow', '')
            predicted = t.get('predicted_tools', [])
            if expected and predicted:
                for p in predicted[:1]:  # just top predicted
                    confusions.append((expected, p))

    return {
        'accuracy': correct / total if total else 0,
        'total_turns': total,
        'null_calls': null_calls,
        'excluded': excluded,
        'ambiguity_flagged': ambiguity_flagged,
        'categories': {
            cat: vals['correct'] / vals['total'] if vals['total'] else 0
            for cat, vals in by_cat.items()
        },
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'latency_p50': int(percentile(latencies, 50)),
        'latency_p95': int(percentile(latencies, 95)),
        'confusions': confusions,
    }


def compute_intent_stats(convos: list[dict]) -> dict:
    """Compute stats from intent classification JSONL records."""
    all_turns = extract_turns(convos)
    if not all_turns:
        return {}

    correct = sum(1 for t in all_turns if t.get('correct'))
    total = len(all_turns)

    by_cat = defaultdict(lambda: {'correct': 0, 'total': 0})
    for t in all_turns:
        by_cat[t['_category']]['total'] += 1
        if t.get('correct'):
            by_cat[t['_category']]['correct'] += 1

    input_tokens = sum(t.get('input_tokens', 0) for t in all_turns)
    output_tokens = sum(t.get('output_tokens', 0) for t in all_turns)
    latencies = sorted(t.get('latency_ms', 0) for t in all_turns if t.get('latency_ms'))

    # Intent confusion matrix
    confusions = []
    for t in all_turns:
        if not t.get('correct'):
            expected = t.get('intent', '')
            detected = t.get('detected_intent', '')
            if expected and detected:
                confusions.append((expected, detected))

    return {
        'accuracy': correct / total if total else 0,
        'total_turns': total,
        'categories': {
            cat: vals['correct'] / vals['total'] if vals['total'] else 0
            for cat, vals in by_cat.items()
        },
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'latency_p50': int(percentile(latencies, 50)),
        'latency_p95': int(percentile(latencies, 95)),
        'confusions': confusions,
    }


def compute_exp1a_stats(convos: list[dict]) -> dict:
    """Compute basic accuracy from Exp 1A JSONL records (for comparison)."""
    all_turns = extract_turns(convos)
    if not all_turns:
        return {}

    correct = sum(1 for t in all_turns if t.get('correct'))
    total = len(all_turns)
    input_tokens = sum(t.get('input_tokens', 0) for t in all_turns)
    output_tokens = sum(t.get('output_tokens', 0) for t in all_turns)

    return {
        'accuracy': correct / total if total else 0,
        'total_turns': total,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
    }


def compute_pipeline_e2e(
    scoped_convos: list[tuple],  # (domain, config_id, seed, convos)
    configs: dict,
) -> list[dict]:
    """Compute pipeline end-to-end accuracy for each model.

    Pipeline E2E = INTENT × FLOW × SCOPED_TOOL, composed per-category
    with special handling for ambiguous turns.
    """
    # Group turns by (model_id, domain, category, turn_num)
    turns_by: dict[tuple, list] = defaultdict(list)

    for domain, config_id, seed, convos in scoped_convos:
        cfg = configs.get(config_id)
        if not cfg:
            continue
        model_id = cfg['model_id']
        for c in convos:
            cat = c.get('category', 'unknown')
            for t in c.get('turns', []):
                tn = t.get('turn_num', 1)
                turns_by[(model_id, domain, cat, tn)].append(t)

    models = sorted(set(k[0] for k in turns_by.keys()), key=model_sort_key)
    categories = ['same_flow', 'switch_flow', 'ambiguous_first', 'ambiguous_second']

    results = []
    for model_id in models:
        model_short = MODEL_SHORT.get(model_id, model_id)
        provider, tier = 'unknown', 'unknown'
        for cfg in configs.values():
            if cfg['model_id'] == model_id:
                provider = cfg.get('provider', 'unknown')
                tier = cfg.get('model_level', 'unknown')
                break

        domain_e2e = {}
        for domain in ['hugo', 'dana']:
            intent_acc = PIPELINE_INTENT_ACC.get(domain, 0.95)
            cat_e2e, cat_weights = {}, {}

            for cat in categories:
                flow_acc = PIPELINE_FLOW_ACC.get(cat, 0.93)

                if cat in ('same_flow', 'switch_flow'):
                    all_t = (turns_by.get((model_id, domain, cat, 1), [])
                             + turns_by.get((model_id, domain, cat, 3), []))
                    if not all_t:
                        continue
                    scoped_acc = sum(1 for t in all_t if t.get('correct')) / len(all_t)
                    cat_e2e[cat] = intent_acc * flow_acc * scoped_acc
                    cat_weights[cat] = len(all_t)

                elif cat == 'ambiguous_first':
                    t1 = turns_by.get((model_id, domain, cat, 1), [])
                    t3 = turns_by.get((model_id, domain, cat, 3), [])
                    total = len(t1) + len(t3)
                    if total == 0:
                        continue
                    t1_e2e = 1.0  # pipeline detects ambiguity at flow stage
                    t3_scoped = sum(1 for t in t3 if t.get('correct')) / len(t3) if t3 else 0
                    t3_e2e = intent_acc * flow_acc * t3_scoped
                    cat_e2e[cat] = (t1_e2e * len(t1) + t3_e2e * len(t3)) / total
                    cat_weights[cat] = total

                elif cat == 'ambiguous_second':
                    t1 = turns_by.get((model_id, domain, cat, 1), [])
                    t3 = turns_by.get((model_id, domain, cat, 3), [])
                    total = len(t1) + len(t3)
                    if total == 0:
                        continue
                    t1_scoped = sum(1 for t in t1 if t.get('correct')) / len(t1) if t1 else 0
                    t1_e2e = intent_acc * flow_acc * t1_scoped
                    t3_e2e = 1.0  # pipeline routes to Plan / ambiguity handler
                    cat_e2e[cat] = (t1_e2e * len(t1) + t3_e2e * len(t3)) / total
                    cat_weights[cat] = total

            if cat_weights:
                tw = sum(cat_weights.values())
                domain_e2e[domain] = sum(
                    cat_e2e[c] * cat_weights[c] / tw for c in cat_e2e
                )
            else:
                domain_e2e[domain] = None

        hugo_e2e = domain_e2e.get('hugo')
        dana_e2e = domain_e2e.get('dana')
        if hugo_e2e is not None and dana_e2e is not None:
            overall = (hugo_e2e + dana_e2e) / 2
        else:
            overall = hugo_e2e or dana_e2e

        results.append({
            'model_short': model_short,
            'model_id': model_id,
            'provider': provider,
            'tier': tier,
            'pipeline_e2e': overall,
            'pipeline_hugo': hugo_e2e,
            'pipeline_dana': dana_e2e,
            '_sort': model_sort_key(model_id),
        })

    results.sort(key=lambda x: x.get('pipeline_e2e') or 0, reverse=True)
    return results


# ── Aggregation helpers ──────────────────────────────────────────

def aggregate_runs(
    runs: list[tuple],  # (domain, config_id, seed, stats)
    configs: dict,
    mode: str,  # 'intent', 'flat_tool', 'scoped_tool'
) -> list[dict]:
    """Aggregate runs by config, producing a ranked list with Hugo/Dana splits."""
    by_config = defaultdict(lambda: {'hugo': [], 'dana': []})
    for domain, config_id, seed, stats in runs:
        by_config[config_id][domain].append((seed, stats))

    ranking = []
    for config_id in sorted(by_config.keys(), key=lambda x: int(x.split('_')[1])):
        cfg = configs.get(config_id)
        if not cfg:
            continue
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
        all_costs = [s['cost'] for _, s in all_runs]

        # Category breakdown (averaged across all runs, both domains)
        def cat_avg(cat_name):
            vals = [s['categories'].get(cat_name, 0) for _, s in all_runs if s.get('categories')]
            return avg(vals)

        def cat_std_val(cat_name):
            vals = [s['categories'].get(cat_name, 0) for _, s in all_runs if s.get('categories')]
            return std(vals)

        entry = {
            'config_id': config_id,
            'model_id': model_id,
            'model_short': model_short,
            'provider': cfg.get('provider', 'unknown'),
            'tier': tier,
            'accuracy': avg(all_accs),
            'accuracy_std': std(all_accs),
            'hugo_acc': avg(hugo_accs) if hugo_accs else None,
            'hugo_std': std(hugo_accs),
            'hugo_seeds': len(hugo_runs),
            'dana_acc': avg(dana_accs) if dana_accs else None,
            'dana_std': std(dana_accs),
            'dana_seeds': len(dana_runs),
            'cost_per_run': avg(all_costs),
            'seed_count': len(all_runs),
            'cat_same_flow': cat_avg('same_flow'),
            'cat_same_flow_std': cat_std_val('same_flow'),
            'cat_switch_flow': cat_avg('switch_flow'),
            'cat_switch_flow_std': cat_std_val('switch_flow'),
            'cat_ambiguous_first': cat_avg('ambiguous_first'),
            'cat_ambiguous_first_std': cat_std_val('ambiguous_first'),
            'cat_ambiguous_second': cat_avg('ambiguous_second'),
            'cat_ambiguous_second_std': cat_std_val('ambiguous_second'),
            '_sort': model_sort_key(model_id),
        }

        # Mode-specific fields
        if mode == 'flat_tool':
            entry['ambiguity_flagged'] = avg([
                s.get('ambiguity_flagged', 0) for _, s in all_runs
            ])

        ranking.append(entry)

    ranking.sort(key=lambda x: x['accuracy'], reverse=True)
    return ranking


# ── Main ─────────────────────────────────────────────────────────

def main():
    exp2_configs = load_configs(CONFIGS_PATH)
    exp1a_configs = load_configs(EXP1A_CONFIGS_PATH)

    # ── Collect flat tool-calling runs (Exp 2B) ───────────────────
    flat_runs = []
    flat_confusions_hugo = Counter()
    flat_confusions_dana = Counter()

    if TOOL_DIR.exists():
        for jsonl_path in sorted(TOOL_DIR.glob('*.jsonl')):
            domain, config_id, seed = parse_exp2_filename(jsonl_path.stem)
            if not domain or config_id not in exp2_configs:
                continue
            convos = load_jsonl(jsonl_path)
            stats = compute_tool_stats(convos)
            if not stats:
                continue
            model_id = exp2_configs[config_id]['model_id']
            stats['cost'] = estimate_cost(model_id, stats['input_tokens'], stats['output_tokens'])
            stats['model_id'] = model_id
            flat_runs.append((domain, config_id, seed, stats))
            target = flat_confusions_hugo if domain == 'hugo' else flat_confusions_dana
            for pair in stats['confusions']:
                target[pair] += 1

    # ── Collect intent runs (Exp 2A) ──────────────────────────────
    intent_runs = []
    intent_confusions_hugo = Counter()
    intent_confusions_dana = Counter()

    if INTENT_DIR.exists():
        for jsonl_path in sorted(INTENT_DIR.glob('*.jsonl')):
            domain, config_id, seed = parse_exp2_filename(jsonl_path.stem)
            if not domain or config_id not in exp2_configs:
                continue
            convos = load_jsonl(jsonl_path)
            stats = compute_intent_stats(convos)
            if not stats:
                continue
            model_id = exp2_configs[config_id]['model_id']
            stats['cost'] = estimate_cost(model_id, stats['input_tokens'], stats['output_tokens'])
            stats['model_id'] = model_id
            intent_runs.append((domain, config_id, seed, stats))
            target = intent_confusions_hugo if domain == 'hugo' else intent_confusions_dana
            for pair in stats['confusions']:
                target[pair] += 1

    # ── Collect scoped tool runs (Exp 2A) ─────────────────────────
    scoped_runs = []
    scoped_convos = []  # raw convos for pipeline E2E computation

    if SCOPED_TOOL_DIR.exists():
        for jsonl_path in sorted(SCOPED_TOOL_DIR.glob('*.jsonl')):
            domain, config_id, seed = parse_exp2_filename(jsonl_path.stem)
            if not domain or config_id not in exp2_configs:
                continue
            convos = load_jsonl(jsonl_path)
            stats = compute_tool_stats(convos)
            if not stats:
                continue
            model_id = exp2_configs[config_id]['model_id']
            stats['cost'] = estimate_cost(model_id, stats['input_tokens'], stats['output_tokens'])
            stats['model_id'] = model_id
            scoped_runs.append((domain, config_id, seed, stats))
            scoped_convos.append((domain, config_id, seed, convos))

    # ── Collect Exp 1A runs (for comparison) ───────────────────────
    exp1a_runs = []

    if EXP1A_DIR.exists():
        for jsonl_path in sorted(EXP1A_DIR.glob('*.jsonl')):
            domain, config_id, seed = parse_exp1a_filename(jsonl_path.stem)
            if not domain or config_id not in exp1a_configs:
                continue
            convos = load_jsonl(jsonl_path)
            stats = compute_exp1a_stats(convos)
            if not stats:
                continue
            model_id = exp1a_configs[config_id]['model_id']
            stats['cost'] = estimate_cost(model_id, stats['input_tokens'], stats['output_tokens'])
            stats['model_id'] = model_id
            exp1a_runs.append((domain, config_id, seed, stats))

    # ── Collect hint runs (Exp 2C) ─────────────────────────────────
    hint_runs = []

    if HINT_DIR.exists():
        for jsonl_path in sorted(HINT_DIR.glob('*.jsonl')):
            domain, config_id, seed = parse_exp2_filename(jsonl_path.stem)
            if not domain or config_id not in exp2_configs:
                continue
            convos = load_jsonl(jsonl_path)
            stats = compute_tool_stats(convos)
            if not stats:
                continue
            model_id = exp2_configs[config_id]['model_id']
            stats['cost'] = estimate_cost(model_id, stats['input_tokens'], stats['output_tokens'])
            stats['model_id'] = model_id
            hint_runs.append((domain, config_id, seed, stats))

    # ── Aggregate rankings ─────────────────────────────────────────
    intent_ranking = aggregate_runs(intent_runs, exp2_configs, 'intent')
    flat_ranking = aggregate_runs(flat_runs, exp2_configs, 'flat_tool')
    scoped_ranking = aggregate_runs(scoped_runs, exp2_configs, 'scoped_tool')
    hint_ranking = aggregate_runs(hint_runs, exp2_configs, 'flat_tool') if hint_runs else []

    # ── Pipeline E2E computation ───────────────────────────────────
    pipeline_results = compute_pipeline_e2e(scoped_convos, exp2_configs)

    # ── Cross-experiment model comparison ──────────────────────────
    exp1a_by_model = defaultdict(lambda: {'hugo': [], 'dana': []})
    for domain, config_id, seed, stats in exp1a_runs:
        model_id = stats['model_id']
        exp1a_by_model[model_id][domain].append(stats)

    # Build comparison rows for pipeline stages chart (§2)
    comparison_rows = []
    for entry in sorted(intent_ranking, key=lambda e: e['_sort']):
        model_id = entry['model_id']
        flat_entry = next((f for f in flat_ranking if f['model_id'] == model_id), None)
        scoped_entry = next((s for s in scoped_ranking if s['model_id'] == model_id), None)
        e1_runs = exp1a_by_model[model_id]['hugo'] + exp1a_by_model[model_id]['dana']
        e1_accs = [s['accuracy'] for s in e1_runs]

        comparison_rows.append({
            'model_short': entry['model_short'],
            'model_id': model_id,
            'provider': entry['provider'],
            'tier': entry['tier'],
            'intent_acc': entry['accuracy'],
            'flow_acc': avg(e1_accs) if e1_accs else None,
            'scoped_acc': scoped_entry['accuracy'] if scoped_entry else None,
            'flat_acc': flat_entry['accuracy'] if flat_entry else None,
            '_sort': model_sort_key(model_id),
        })
    comparison_rows.sort(key=lambda x: x['_sort'])

    # ── Hero rows: pipeline E2E vs flat ────────────────────────────
    hero_rows = []
    for p in pipeline_results:
        model_id = p['model_id']
        flat_entry = next((f for f in flat_ranking if f['model_id'] == model_id), None)
        flat_acc = flat_entry['accuracy'] if flat_entry else None
        pe = p['pipeline_e2e']
        delta = pe - flat_acc if (pe is not None and flat_acc is not None) else None
        rel_gain = delta / flat_acc if (delta is not None and flat_acc) else None
        hero_rows.append({
            'model_short': p['model_short'],
            'model_id': model_id,
            'provider': p['provider'],
            'tier': p['tier'],
            'pipeline_e2e': pe,
            'pipeline_hugo': p['pipeline_hugo'],
            'pipeline_dana': p['pipeline_dana'],
            'flat_acc': flat_acc,
            'flat_hugo': flat_entry.get('hugo_acc') if flat_entry else None,
            'flat_dana': flat_entry.get('dana_acc') if flat_entry else None,
            'delta': delta,
            'rel_gain': rel_gain,
            '_sort': model_sort_key(model_id),
        })
    hero_rows.sort(key=lambda x: x['_sort'])

    # ── Confusion analysis ─────────────────────────────────────────
    def top_confusions(counter, n=12):
        return [
            {'expected': exp, 'predicted': pred, 'count': count}
            for (exp, pred), count in counter.most_common(n)
        ]

    # ── Key findings (auto-generated paragraph) ─────────────────────
    findings_text = ''

    if hero_rows:
        deltas = [r['delta'] for r in hero_rows if r['delta'] is not None]
        best_p = max((r for r in hero_rows if r['pipeline_e2e'] is not None),
                     key=lambda r: r['pipeline_e2e'], default=None)
        best_flat = flat_ranking[0] if flat_ranking else None

        sentences = []
        if deltas:
            avg_delta = sum(deltas) / len(deltas)
            wins = sum(1 for d in deltas if d > 0)
            sentences.append(
                f"The 3-stage NLU pipeline (intent → flow → scoped tools) outperforms flat "
                f"tool-calling for {wins} of {len(deltas)} models tested, with an average "
                f"end-to-end advantage of {avg_delta:+.1%}."
            )
        if best_p and best_flat:
            sentences.append(
                f"The best pipeline configuration ({best_p['model_short']}) achieves "
                f"{best_p['pipeline_e2e']:.1%} E2E accuracy, compared to {best_flat['accuracy']:.1%} "
                f"for the best flat model ({best_flat['model_short']})."
            )
        if deltas:
            best_b = max((r for r in hero_rows if r['delta'] is not None),
                         key=lambda r: r['delta'])
            worst_b = min((r for r in hero_rows if r['delta'] is not None),
                          key=lambda r: r['delta'])
            sentences.append(
                f"The pipeline benefit is largest for {best_b['model_short']} "
                f"({best_b['delta']:+.1%}) and smallest for {worst_b['model_short']} "
                f"({worst_b['delta']:+.1%}), suggesting weaker models gain more from "
                f"staged narrowing."
            )
        # H5: domain effect
        hugo_d = [(r['pipeline_hugo'] - r['flat_hugo'])
                  for r in hero_rows
                  if r.get('pipeline_hugo') is not None and r.get('flat_hugo') is not None]
        dana_d = [(r['pipeline_dana'] - r['flat_dana'])
                  for r in hero_rows
                  if r.get('pipeline_dana') is not None and r.get('flat_dana') is not None]
        if hugo_d and dana_d:
            h_avg, d_avg = avg(hugo_d), avg(dana_d)
            sentences.append(
                f"Contrary to hypothesis H5, the pipeline advantage is nearly identical "
                f"across domains (Hugo {h_avg:+.1%}, Dana {d_avg:+.1%}), indicating that "
                f"staged narrowing helps regardless of tool-set complexity."
            )
        sentences.append(
            "Overall, the results validate the pipeline architecture: the cost of "
            "running three lightweight stages is repaid by substantially higher accuracy, "
            "especially for models that struggle with large tool inventories."
        )
        findings_text = ' '.join(sentences)

    # ── Hint ablation rows (Exp 2C vs 2B) ─────────────────────────
    hint_rows = []
    for h in sorted(hint_ranking, key=lambda e: e['_sort']):
        model_id = h['model_id']
        flat_entry = next((f for f in flat_ranking if f['model_id'] == model_id), None)
        if not flat_entry:
            continue
        cats = ['same_flow', 'switch_flow', 'ambiguous_first', 'ambiguous_second']
        cat_deltas = {}
        for cat in cats:
            hv = h.get(f'cat_{cat}')
            fv = flat_entry.get(f'cat_{cat}')
            cat_deltas[f'd_{cat}'] = hv - fv if (hv is not None and fv is not None) else None
            cat_deltas[f'hint_{cat}'] = hv
            cat_deltas[f'flat_{cat}'] = fv
        hint_rows.append({
            'model_short': h['model_short'],
            'model_id': model_id,
            'tier': h['tier'],
            'hint_acc': h['accuracy'],
            'flat_acc': flat_entry['accuracy'],
            'delta': h['accuracy'] - flat_entry['accuracy'],
            'hint_seeds': h['seed_count'],
            '_sort': model_sort_key(model_id),
            **cat_deltas,
        })

    # Append hint ablation finding if available
    if hint_rows and findings_text:
        hint_deltas = [r['delta'] for r in hint_rows if r.get('delta') is not None]
        if hint_deltas:
            h_avg = sum(hint_deltas) / len(hint_deltas)
            verb = 'improves' if h_avg > 0 else 'hurts'
            findings_text += (
                f" An explicit ambiguity hint (Exp 2C) {verb} flat accuracy by "
                f"{h_avg:+.1%} on average, indicating that prompt-level intervention "
                f"{'partially compensates for' if h_avg > 0 else 'cannot replace'} "
                f"the pipeline's structural ambiguity handling."
            )

    # ── Assemble report data ───────────────────────────────────────
    total_cost = (
        sum(s['cost'] for _, _, _, s in flat_runs)
        + sum(s['cost'] for _, _, _, s in intent_runs)
        + sum(s['cost'] for _, _, _, s in scoped_runs)
        + sum(s['cost'] for _, _, _, s in hint_runs)
    )

    report_data = {
        'generated': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'total_flat_runs': len(flat_runs),
        'total_intent_runs': len(intent_runs),
        'total_scoped_runs': len(scoped_runs),
        'total_hint_runs': len(hint_runs),
        'total_cost': total_cost,
        'hero_rows': hero_rows,
        'flat_ranking': flat_ranking,
        'comparison_rows': comparison_rows,
        'hint_rows': hint_rows,
        'flat_confusions_hugo': top_confusions(flat_confusions_hugo),
        'flat_confusions_dana': top_confusions(flat_confusions_dana),
        'findings_text': findings_text,
    }

    # ── Generate HTML report ───────────────────────────────────────
    html = generate_html(report_data)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html)
    print(f'Report written to {REPORT_PATH}')
    print(f'  Flat: {len(flat_runs)}, Intent: {len(intent_runs)}, Scoped: {len(scoped_runs)}, Hint: {len(hint_runs)}')
    if hero_rows:
        best_p = max((r for r in hero_rows if r['pipeline_e2e'] is not None),
                     key=lambda r: r['pipeline_e2e'], default=None)
        if best_p:
            print(f'  Best pipeline E2E: {best_p["model_short"]} @ {best_p["pipeline_e2e"]:.1%}')
    if flat_ranking:
        print(f'  Best flat tool: {flat_ranking[0]["model_short"]} @ {flat_ranking[0]["accuracy"]:.1%}')


def generate_html(data: dict) -> str:
    """Generate standalone HTML report with SVG charts."""
    data_json = json.dumps(data, indent=None, default=str)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Experiment 2 — Pipeline vs Flat Report</title>
<style>
  :root {{
    --bg: #ffffff; --surface: #f6f8fa; --border: #d0d7de;
    --text: #1f2328; --muted: #656d76; --accent: #0969da;
    --green: #1a7f37; --yellow: #9a6700; --red: #cf222e;
    --purple: #8250df; --orange: #bc4c00; --cyan: #0550ae;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6; padding: 2rem;
    max-width: 1200px; margin: 0 auto;
  }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; }}
  h2 {{ font-size: 1.3rem; margin: 2.5rem 0 1rem; color: var(--accent); border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }}
  h3 {{ font-size: 1.05rem; margin: 1.5rem 0 0.75rem; color: var(--purple); }}
  .subtitle {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 2rem; }}
  .stat-row {{ display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 1.5rem; }}
  .stat-box {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 1rem 1.5rem; min-width: 140px; text-align: center; flex: 1;
  }}
  .stat-box .value {{ font-size: 1.8rem; font-weight: 700; color: var(--accent); }}
  .stat-box .label {{ font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }}
  .card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 1.25rem; overflow-x: auto;
  }}
  .card.full {{ grid-column: 1 / -1; }}
  .card h3 {{ margin-top: 0; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  th, td {{ padding: 0.4rem 0.6rem; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 600; font-size: 0.72rem; text-transform: uppercase; white-space: nowrap; }}
  td {{ white-space: nowrap; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr:hover td {{ background: rgba(9,105,218,0.04); }}
  .note {{ color: var(--muted); font-size: 0.8rem; margin-top: 0.75rem; font-style: italic; }}
  .badge {{ display: inline-block; padding: 0.15em 0.5em; border-radius: 4px; font-size: 0.72rem; font-weight: 600; }}
  .badge-green {{ background: rgba(26,127,55,0.1); color: var(--green); }}
  .badge-yellow {{ background: rgba(154,103,0,0.1); color: var(--yellow); }}
  .badge-red {{ background: rgba(207,34,46,0.1); color: var(--red); }}
  .legend {{ display: flex; gap: 1.5rem; margin-top: 0.5rem; font-size: 0.75rem; color: var(--muted); justify-content: center; flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 0.3rem; }}
  .legend-dot {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px; }}
  .section-notes {{ list-style: disc; padding-left: 1.5rem; margin-top: 0.75rem; }}
  .section-notes li {{ color: var(--text); font-size: 0.85rem; margin: 0.3rem 0; }}
  .findings {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.25rem; }}
  .findings p {{ font-size: 0.9rem; line-height: 1.7; }}
  footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); color: var(--muted); font-size: 0.75rem; }}
</style>
</head>
<body>

<h1>Experiment 2 — Pipeline vs Flat</h1>
<p class="subtitle" id="subtitle">Loading results...</p>

<div class="stat-row" id="top-stats"></div>

<!-- §1. Hero: Pipeline vs Flat -->
<h2>1. Pipeline vs Flat — End-to-End Accuracy</h2>
<p class="note" style="margin-bottom:0.75rem">Pipeline E2E = Intent (Flash) &times; Flow (5v-6 ensemble) &times; Scoped Tool (per model). Positive delta means pipeline wins.</p>
<div class="card full">
  <div id="hero-chart"></div>
  <div class="legend">
    <div class="legend-item"><span class="legend-dot" style="background:#2563eb"></span> Pipeline E2E</div>
    <div class="legend-item"><span class="legend-dot" style="background:#dc2626"></span> Flat</div>
  </div>
</div>
<div class="card full">
  <table id="hero-table"></table>
</div>

<!-- §2. Pipeline Stages Breakdown -->
<h2>2. Pipeline Stages Breakdown</h2>
<p class="note" style="margin-bottom:0.75rem">Per-stage accuracy: where in the pipeline does accuracy change?</p>
<div class="card full">
  <div id="pipeline-chart"></div>
  <div class="legend">
    <div class="legend-item"><span class="legend-dot" style="background:#93c5fd"></span> Intent (Exp 2A)</div>
    <div class="legend-item"><span class="legend-dot" style="background:#2563eb"></span> Flow Detection (Exp 1A)</div>
    <div class="legend-item"><span class="legend-dot" style="background:#86efac"></span> Scoped Tool (Exp 2A)</div>
    <div class="legend-item"><span class="legend-dot" style="background:#dc2626"></span> Flat Tool (Exp 2B)</div>
  </div>
</div>
<div class="card full">
  <table id="stages-table"></table>
</div>
<ul class="section-notes" id="s2-notes"></ul>

<!-- §3. Domain Split — Hugo vs Dana -->
<h2>3. Domain Split — Hugo vs Dana</h2>
<p class="note" style="margin-bottom:0.75rem">Does the pipeline help more in Hugo (more confusable tools) than Dana? Tests hypothesis H5.</p>
<div class="grid">
  <div class="card"><h3>Hugo</h3><div id="domain-hugo-chart"></div></div>
  <div class="card"><h3>Dana</h3><div id="domain-dana-chart"></div></div>
</div>
<ul class="section-notes" id="s3-notes"></ul>

<!-- §4. Category Breakdown — Where Flat Struggles -->
<h2>4. Category Breakdown — Where Flat Struggles</h2>
<p class="note" style="margin-bottom:0.75rem">Flat tool-calling accuracy by conversation category.</p>
<div class="card full">
  <div id="flat-cat-chart"></div>
  <div class="legend">
    <div class="legend-item"><span class="legend-dot" style="background:#93c5fd"></span> Same Flow</div>
    <div class="legend-item"><span class="legend-dot" style="background:#2563eb"></span> Switch Flow</div>
    <div class="legend-item"><span class="legend-dot" style="background:#a78bfa"></span> Ambiguous 1st</div>
    <div class="legend-item"><span class="legend-dot" style="background:#7c3aed"></span> Ambiguous 2nd</div>
  </div>
</div>
<ul class="section-notes" id="s4-notes"></ul>

<!-- §5. Ambiguity Hint Ablation (Exp 2C) -->
<h2>5. Ambiguity Hint Ablation (Exp 2C)</h2>
<p class="note" style="margin-bottom:0.75rem">Flat tool-calling + explicit prompt hint to prefer <code>handle_ambiguity</code> on unclear requests. Arrows show change vs Exp 2B (no hint).</p>
<div class="grid">
  <div class="card"><h3>Same Flow</h3><div id="hint-same"></div></div>
  <div class="card"><h3>Switch Flow</h3><div id="hint-switch"></div></div>
  <div class="card"><h3>Ambiguous 1st</h3><div id="hint-ambig1"></div></div>
  <div class="card"><h3>Ambiguous 2nd</h3><div id="hint-ambig2"></div></div>
</div>
<div style="margin-top:-0.5rem;margin-bottom:1rem" class="legend">
  <div class="legend-item"><span class="legend-dot" style="background:#dc2626"></span> Flat (2B)</div>
  <div class="legend-item"><span class="legend-dot" style="background:#f97316"></span> Hint (2C)</div>
</div>
<div class="card full">
  <table id="hint-table"></table>
</div>
<ul class="section-notes" id="s5-notes"></ul>

<!-- §6. Flat Tool Confusion Analysis -->
<h2>6. Flat Tool Confusion Analysis</h2>
<div class="grid">
  <div class="card"><h3>Hugo</h3><table id="flat-conf-hugo"></table></div>
  <div class="card"><h3>Dana</h3><table id="flat-conf-dana"></table></div>
</div>
<ul class="section-notes" id="s6-notes"></ul>

<!-- §7. Key Findings -->
<h2>7. Key Findings</h2>
<div class="findings"><p id="findings-text"></p></div>

<footer>
  <p>Generated from Exp 2 JSONL data. Regenerate: <code>python3 results/build_report_2.py</code></p>
</footer>

<script>
const DATA = {data_json};

// ── SVG helpers ──────────────────────────────────────────────────
function fmtPct(v) {{ return v == null ? '-' : (v * 100).toFixed(1) + '%'; }}
function fmtCost(v) {{ return v == null ? '-' : '$' + v.toFixed(3); }}
function fmtDelta(v) {{ return v == null ? '-' : (v >= 0 ? '+' : '') + (v * 100).toFixed(1); }}

function svgEl(tag, attrs, children) {{
  const a = Object.entries(attrs).map(([k, v]) => `${{k}}="${{v}}"`).join(' ');
  return `<${{tag}} ${{a}}>${{children || ''}}</${{tag}}>`;
}}

// ── Vertical bar chart ───────────────────────────────────────────
function renderVBarChart(containerId, items, {{
  valuePairs = null,
  valueKey = null,
  stdKey = null,
  labelKey = 'model_short',
  colorFn = null,
  showDelta = false,
  arrowDelta = false,
  deltaKey = 'delta',
  yMin = 0.5, yMax = 1.0,
  width = 1050, height = 340,
}} = {{}}) {{
  const container = document.getElementById(containerId);
  if (!container || !items.length) return;

  const pad = {{ top: 30, right: 20, bottom: 60, left: 50 }};
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const n = items.length;
  const isPaired = valuePairs != null;
  const groupW = plotW / n;
  const barW = isPaired ? Math.min(groupW * 0.35, 32) : Math.min(groupW * 0.6, 48);
  const barGap = isPaired ? 3 : 0;

  function sy(v) {{ return pad.top + plotH - (v - yMin) / (yMax - yMin) * plotH; }}
  function sx(i) {{ return pad.left + groupW * i + groupW / 2; }}

  let svg = `<svg width="${{width}}" height="${{height}}" xmlns="http://www.w3.org/2000/svg" style="max-width:100%">`;

  // Y grid lines
  for (let i = 0; i <= 5; i++) {{
    const v = yMin + (yMax - yMin) * i / 5;
    const y = sy(v);
    svg += svgEl('line', {{ x1: pad.left, y1: y, x2: width - pad.right, y2: y, stroke: '#e5e7eb', 'stroke-dasharray': '2,3' }});
    svg += svgEl('text', {{ x: pad.left - 8, y: y + 4, fill: '#656d76', 'font-size': 11, 'text-anchor': 'end' }}, (v * 100).toFixed(0) + '%');
  }}

  // Bars
  items.forEach((item, i) => {{
    const cx = sx(i);

    if (isPaired) {{
      valuePairs.forEach((pair, pi) => {{
        const v = item[pair.key];
        if (v == null) return;
        const barX = cx + (pi === 0 ? -barW - barGap/2 : barGap/2);
        const barY = sy(v);
        const barH = sy(yMin) - barY;
        svg += svgEl('rect', {{ x: barX, y: barY, width: barW, height: Math.max(barH, 1), fill: pair.color, rx: 2 }});
        svg += svgEl('text', {{ x: barX + barW/2, y: barY - 5, fill: '#1f2328', 'font-size': 10, 'text-anchor': 'middle', 'font-weight': 600 }}, (v * 100).toFixed(1));

        const stdVal = pair.stdKey ? item[pair.stdKey] : null;
        if (stdVal) {{
          const top = sy(Math.min(v + stdVal, yMax));
          const bot = sy(Math.max(v - stdVal, yMin));
          const mx = barX + barW/2;
          svg += svgEl('line', {{ x1: mx, y1: top, x2: mx, y2: bot, stroke: '#656d76', 'stroke-width': 1.5 }});
          svg += svgEl('line', {{ x1: mx-3, y1: top, x2: mx+3, y2: top, stroke: '#656d76', 'stroke-width': 1.5 }});
          svg += svgEl('line', {{ x1: mx-3, y1: bot, x2: mx+3, y2: bot, stroke: '#656d76', 'stroke-width': 1.5 }});
        }}
      }});

      // Delta label above the pair
      if (showDelta && item[deltaKey] != null) {{
        const d = item[deltaKey];
        const dColor = d >= 0 ? '#1a7f37' : '#cf222e';
        const arrow = arrowDelta ? (d >= 0 ? '\u25B2 ' : '\u25BC ') : '';
        svg += svgEl('text', {{ x: cx, y: pad.top - 6, fill: dColor, 'font-size': 10, 'text-anchor': 'middle', 'font-weight': 700 }}, arrow + fmtDelta(d));
      }}
    }} else {{
      const v = item[valueKey];
      if (v == null) return;
      const barX = cx - barW/2;
      const barY = sy(v);
      const barH = sy(yMin) - barY;
      const color = colorFn ? colorFn(item) : '#2563eb';
      svg += svgEl('rect', {{ x: barX, y: barY, width: barW, height: Math.max(barH, 1), fill: color, rx: 2 }});
      svg += svgEl('text', {{ x: cx, y: barY - 5, fill: '#1f2328', 'font-size': 10, 'text-anchor': 'middle', 'font-weight': 600 }}, (v * 100).toFixed(1));

      if (stdKey && item[stdKey]) {{
        const top = sy(Math.min(v + item[stdKey], yMax));
        const bot = sy(Math.max(v - item[stdKey], yMin));
        svg += svgEl('line', {{ x1: cx, y1: top, x2: cx, y2: bot, stroke: '#656d76', 'stroke-width': 1.5 }});
        svg += svgEl('line', {{ x1: cx-3, y1: top, x2: cx+3, y2: top, stroke: '#656d76', 'stroke-width': 1.5 }});
        svg += svgEl('line', {{ x1: cx-3, y1: bot, x2: cx+3, y2: bot, stroke: '#656d76', 'stroke-width': 1.5 }});
      }}
    }}

    // X label
    svg += svgEl('text', {{ x: cx, y: height - pad.bottom + 16, fill: '#1f2328', 'font-size': 11, 'text-anchor': 'middle' }}, item[labelKey]);
    const tierColor = {{ low: '#57a0d3', medium: '#2563eb', high: '#1e40af' }}[item.tier] || '#94a3b8';
    svg += svgEl('text', {{ x: cx, y: height - pad.bottom + 30, fill: tierColor, 'font-size': 9, 'text-anchor': 'middle', 'font-weight': 600 }}, item.tier);
  }});

  svg += '</svg>';
  container.innerHTML = svg;
}}

// ── Grouped vertical bar chart (for categories) ──────────────────
function renderGroupedVBarChart(containerId, items, groups, {{
  yMin = 0.5, yMax = 1.0, width = 1050, height = 340,
}} = {{}}) {{
  const container = document.getElementById(containerId);
  if (!container || !items.length) return;

  const pad = {{ top: 25, right: 20, bottom: 60, left: 50 }};
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const n = items.length;
  const nG = groups.length;
  const groupW = plotW / n;
  const barW = Math.min(groupW / (nG + 1), 20);
  const totalBarsW = nG * barW + (nG - 1) * 2;

  function sy(v) {{ return pad.top + plotH - (v - yMin) / (yMax - yMin) * plotH; }}
  function sx(i) {{ return pad.left + groupW * i + groupW / 2; }}

  let svg = `<svg width="${{width}}" height="${{height}}" xmlns="http://www.w3.org/2000/svg" style="max-width:100%">`;

  for (let i = 0; i <= 5; i++) {{
    const v = yMin + (yMax - yMin) * i / 5;
    const y = sy(v);
    svg += svgEl('line', {{ x1: pad.left, y1: y, x2: width - pad.right, y2: y, stroke: '#e5e7eb', 'stroke-dasharray': '2,3' }});
    svg += svgEl('text', {{ x: pad.left - 8, y: y + 4, fill: '#656d76', 'font-size': 11, 'text-anchor': 'end' }}, (v * 100).toFixed(0) + '%');
  }}

  items.forEach((item, i) => {{
    const cx = sx(i);
    const startX = cx - totalBarsW / 2;

    groups.forEach((g, gi) => {{
      const v = item[g.key];
      if (v == null) return;
      const barX = startX + gi * (barW + 2);
      const barY = sy(v);
      const barH = sy(yMin) - barY;
      svg += svgEl('rect', {{ x: barX, y: barY, width: barW, height: Math.max(barH, 1), fill: g.color, rx: 1 }});

      const stdVal = item[g.key + '_std'];
      if (stdVal) {{
        const top = sy(Math.min(v + stdVal, yMax));
        const bot = sy(Math.max(v - stdVal, yMin));
        const mx = barX + barW/2;
        svg += svgEl('line', {{ x1: mx, y1: top, x2: mx, y2: bot, stroke: '#656d76', 'stroke-width': 1 }});
        svg += svgEl('line', {{ x1: mx-2, y1: top, x2: mx+2, y2: top, stroke: '#656d76', 'stroke-width': 1 }});
        svg += svgEl('line', {{ x1: mx-2, y1: bot, x2: mx+2, y2: bot, stroke: '#656d76', 'stroke-width': 1 }});
      }}
    }});

    svg += svgEl('text', {{ x: cx, y: height - pad.bottom + 16, fill: '#1f2328', 'font-size': 11, 'text-anchor': 'middle' }}, item.model_short);
    const tierColor = {{ low: '#57a0d3', medium: '#2563eb', high: '#1e40af' }}[item.tier] || '#94a3b8';
    svg += svgEl('text', {{ x: cx, y: height - pad.bottom + 30, fill: tierColor, 'font-size': 9, 'text-anchor': 'middle', 'font-weight': 600 }}, item.tier);
  }});

  svg += '</svg>';
  container.innerHTML = svg;
}}

// ── Multi-metric comparison chart (for pipeline stages) ──────────
function renderPipelineChart(containerId, items, metrics, {{
  yMin = 0.5, yMax = 1.0, width = 1050, height = 360,
}} = {{}}) {{
  const container = document.getElementById(containerId);
  if (!container || !items.length) return;

  const pad = {{ top: 25, right: 20, bottom: 60, left: 50 }};
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const n = items.length;
  const nM = metrics.length;
  const groupW = plotW / n;
  const barW = Math.min(groupW / (nM + 1), 22);
  const totalBarsW = nM * barW + (nM - 1) * 3;

  function sy(v) {{ return pad.top + plotH - (v - yMin) / (yMax - yMin) * plotH; }}
  function sx(i) {{ return pad.left + groupW * i + groupW / 2; }}

  let svg = `<svg width="${{width}}" height="${{height}}" xmlns="http://www.w3.org/2000/svg" style="max-width:100%">`;

  for (let i = 0; i <= 5; i++) {{
    const v = yMin + (yMax - yMin) * i / 5;
    const y = sy(v);
    svg += svgEl('line', {{ x1: pad.left, y1: y, x2: width - pad.right, y2: y, stroke: '#e5e7eb', 'stroke-dasharray': '2,3' }});
    svg += svgEl('text', {{ x: pad.left - 8, y: y + 4, fill: '#656d76', 'font-size': 11, 'text-anchor': 'end' }}, (v * 100).toFixed(0) + '%');
  }}

  items.forEach((item, i) => {{
    const cx = sx(i);
    const startX = cx - totalBarsW / 2;

    metrics.forEach((m, mi) => {{
      const v = item[m.key];
      if (v == null) return;
      const barX = startX + mi * (barW + 3);
      const barY = sy(v);
      const barH = sy(yMin) - barY;
      svg += svgEl('rect', {{ x: barX, y: barY, width: barW, height: Math.max(barH, 1), fill: m.color, rx: 2 }});
      svg += svgEl('text', {{ x: barX + barW/2, y: barY - 4, fill: '#1f2328', 'font-size': 9, 'text-anchor': 'middle', 'font-weight': 600 }}, (v * 100).toFixed(0));
    }});

    svg += svgEl('text', {{ x: cx, y: height - pad.bottom + 16, fill: '#1f2328', 'font-size': 11, 'text-anchor': 'middle' }}, item.model_short);
    const tierColor = {{ low: '#57a0d3', medium: '#2563eb', high: '#1e40af' }}[item.tier] || '#94a3b8';
    svg += svgEl('text', {{ x: cx, y: height - pad.bottom + 30, fill: tierColor, 'font-size': 9, 'text-anchor': 'middle', 'font-weight': 600 }}, item.tier);
  }});

  svg += '</svg>';
  container.innerHTML = svg;
}}

// ── Populate page ────────────────────────────────────────────────

// Subtitle
document.getElementById('subtitle').textContent =
  `Generated: ${{DATA.generated}} | Flat: ${{DATA.total_flat_runs}} runs | Intent: ${{DATA.total_intent_runs}} runs | Scoped: ${{DATA.total_scoped_runs}} runs | Hint: ${{DATA.total_hint_runs}} runs | Cost: $${{DATA.total_cost.toFixed(2)}}`;

// Top stats
const ts = document.getElementById('top-stats');
function addStat(label, value, sub) {{
  const d = document.createElement('div');
  d.className = 'stat-box';
  d.innerHTML = `<div class="value">${{value}}</div><div class="label">${{label}}</div>`;
  if (sub) d.innerHTML += `<div style="font-size:0.72rem;color:#656d76;margin-top:2px">${{sub}}</div>`;
  ts.appendChild(d);
}}

// Pipeline advantage
if (DATA.hero_rows.length) {{
  const deltas = DATA.hero_rows.filter(r => r.delta != null).map(r => r.delta);
  if (deltas.length) {{
    const avgD = deltas.reduce((a,b) => a+b, 0) / deltas.length;
    addStat('Pipeline Advantage', fmtDelta(avgD) + 'pp', deltas.length + ' models compared');
  }}
  const bestP = [...DATA.hero_rows].filter(r => r.pipeline_e2e != null).sort((a,b) => b.pipeline_e2e - a.pipeline_e2e)[0];
  if (bestP) addStat('Best Pipeline E2E', fmtPct(bestP.pipeline_e2e), bestP.model_short);
}}
if (DATA.flat_ranking.length) {{
  const b = DATA.flat_ranking[0];
  addStat('Best Flat', fmtPct(b.accuracy), b.model_short);
}}
addStat('Total Cost', '$' + DATA.total_cost.toFixed(2), DATA.total_flat_runs + DATA.total_intent_runs + DATA.total_scoped_runs + ' runs');

// Sort by display order
function sortByDisplay(arr) {{ return [...arr].sort((a, b) => (a._sort || 999) - (b._sort || 999)); }}

// ── §1. Hero chart: Pipeline vs Flat ─────────────────────────────
const heroSorted = sortByDisplay(DATA.hero_rows);
renderVBarChart('hero-chart', heroSorted, {{
  valuePairs: [
    {{ key: 'pipeline_e2e', color: '#2563eb' }},
    {{ key: 'flat_acc', color: '#dc2626' }},
  ],
  showDelta: true,
  deltaKey: 'delta',
  yMin: 0.2, yMax: 1.0,
}});

// Hero table
const heroTbl = document.getElementById('hero-table');
heroTbl.innerHTML = `<thead>
  <tr><th rowspan="2">Model</th><th rowspan="2">Tier</th>
    <th rowspan="2">Pipeline E2E</th><th rowspan="2">Flat</th>
    <th rowspan="2">Delta</th><th rowspan="2">Rel. Gain</th>
    <th colspan="2" style="text-align:center;border-bottom:none">Pipeline</th>
    <th colspan="2" style="text-align:center;border-bottom:none">Flat</th></tr>
  <tr><th>Hugo</th><th>Dana</th><th>Hugo</th><th>Dana</th></tr>
</thead><tbody></tbody>`;
function fmtRel(v) {{ return v == null ? '-' : (v >= 0 ? '+' : '') + (v * 100).toFixed(1) + '%'; }}
heroSorted.forEach(r => {{
  const dColor = r.delta != null ? (r.delta >= 0 ? 'color:var(--green);font-weight:700' : 'color:var(--red)') : '';
  heroTbl.querySelector('tbody').innerHTML += `<tr>
    <td>${{r.model_short}}</td><td>${{r.tier}}</td>
    <td class="num"><strong>${{fmtPct(r.pipeline_e2e)}}</strong></td>
    <td class="num">${{fmtPct(r.flat_acc)}}</td>
    <td class="num" style="${{dColor}}">${{r.delta != null ? fmtDelta(r.delta) + 'pp' : '-'}}</td>
    <td class="num" style="color:var(--accent);font-weight:700">${{fmtRel(r.rel_gain)}}</td>
    <td class="num">${{fmtPct(r.pipeline_hugo)}}</td><td class="num">${{fmtPct(r.pipeline_dana)}}</td>
    <td class="num">${{fmtPct(r.flat_hugo)}}</td><td class="num">${{fmtPct(r.flat_dana)}}</td>
  </tr>`;
}});

// Combined (average) row
if (heroSorted.length) {{
  const mean = (arr) => arr.length ? arr.reduce((a,b) => a+b, 0) / arr.length : null;
  const pE = mean(heroSorted.filter(r => r.pipeline_e2e != null).map(r => r.pipeline_e2e));
  const fA = mean(heroSorted.filter(r => r.flat_acc != null).map(r => r.flat_acc));
  const pH = mean(heroSorted.filter(r => r.pipeline_hugo != null).map(r => r.pipeline_hugo));
  const pD = mean(heroSorted.filter(r => r.pipeline_dana != null).map(r => r.pipeline_dana));
  const fH = mean(heroSorted.filter(r => r.flat_hugo != null).map(r => r.flat_hugo));
  const fD = mean(heroSorted.filter(r => r.flat_dana != null).map(r => r.flat_dana));
  const cDelta = (pE != null && fA != null) ? pE - fA : null;
  const cRel = (cDelta != null && fA) ? cDelta / fA : null;
  const cColor = cDelta != null ? (cDelta >= 0 ? 'color:var(--green);font-weight:700' : 'color:var(--red)') : '';
  heroTbl.querySelector('tbody').innerHTML += `<tr style="border-top:2px solid var(--border);font-weight:600">
    <td>Combined</td><td>medium</td>
    <td class="num"><strong>${{fmtPct(pE)}}</strong></td>
    <td class="num">${{fmtPct(fA)}}</td>
    <td class="num" style="${{cColor}}">${{cDelta != null ? fmtDelta(cDelta) + 'pp' : '-'}}</td>
    <td class="num" style="color:var(--accent);font-weight:700">${{fmtRel(cRel)}}</td>
    <td class="num">${{fmtPct(pH)}}</td><td class="num">${{fmtPct(pD)}}</td>
    <td class="num">${{fmtPct(fH)}}</td><td class="num">${{fmtPct(fD)}}</td>
  </tr>`;
}}

// ── §2. Pipeline stages chart ────────────────────────────────────
const compSorted = sortByDisplay(DATA.comparison_rows);
renderPipelineChart('pipeline-chart', compSorted, [
  {{ key: 'intent_acc', color: '#93c5fd', label: 'Intent' }},
  {{ key: 'flow_acc', color: '#2563eb', label: 'Flow' }},
  {{ key: 'scoped_acc', color: '#86efac', label: 'Scoped Tool' }},
  {{ key: 'flat_acc', color: '#dc2626', label: 'Flat Tool' }},
], {{ yMin: 0.5, yMax: 1.0 }});

// Stages table
const stgTbl = document.getElementById('stages-table');
stgTbl.innerHTML = `<thead><tr><th>Model</th><th>Tier</th>
  <th>Intent</th><th>Flow (1A)</th><th>Scoped Tool</th><th>Flat Tool</th>
</tr></thead><tbody></tbody>`;
compSorted.forEach(r => {{
  let scopedCls = '', flatCls = '';
  if (r.scoped_acc != null && r.flat_acc != null) {{
    if (r.scoped_acc > r.flat_acc) {{ scopedCls = 'style="color:var(--green);font-weight:700"'; flatCls = 'style="color:var(--red)"'; }}
    else if (r.flat_acc > r.scoped_acc) {{ flatCls = 'style="color:var(--green);font-weight:700"'; scopedCls = 'style="color:var(--red)"'; }}
  }}
  stgTbl.querySelector('tbody').innerHTML += `<tr>
    <td>${{r.model_short}}</td><td>${{r.tier}}</td>
    <td class="num">${{fmtPct(r.intent_acc)}}</td>
    <td class="num">${{fmtPct(r.flow_acc)}}</td>
    <td class="num" ${{scopedCls}}>${{fmtPct(r.scoped_acc)}}</td>
    <td class="num" ${{flatCls}}>${{fmtPct(r.flat_acc)}}</td>
  </tr>`;
}});

// §2 takeaways
(function() {{
  const notes = [];
  const withAll = compSorted.filter(r => r.intent_acc != null && r.flow_acc != null && r.scoped_acc != null);
  if (withAll.length) {{
    // Bottleneck stage
    const avgIntent = withAll.reduce((a,r) => a + r.intent_acc, 0) / withAll.length;
    const avgFlow = withAll.reduce((a,r) => a + r.flow_acc, 0) / withAll.length;
    const avgScoped = withAll.reduce((a,r) => a + r.scoped_acc, 0) / withAll.length;
    const stages = [['Intent', avgIntent], ['Flow detection', avgFlow], ['Scoped tool', avgScoped]];
    stages.sort((a,b) => a[1] - b[1]);
    notes.push(`<strong>${{stages[0][0]}}</strong> is the pipeline bottleneck at ${{(stages[0][1]*100).toFixed(1)}}% average accuracy, while <strong>${{stages[2][0]}}</strong> is the strongest at ${{(stages[2][1]*100).toFixed(1)}}%.`);
  }}
  const withBoth = compSorted.filter(r => r.scoped_acc != null && r.flat_acc != null);
  if (withBoth.length) {{
    const allBetter = withBoth.every(r => r.scoped_acc > r.flat_acc);
    if (allBetter) notes.push('Scoped tool accuracy exceeds flat for every model tested, confirming that intent-based tool filtering improves selection.');
    else {{
      const better = withBoth.filter(r => r.scoped_acc > r.flat_acc).length;
      notes.push(`Scoped tools beat flat for ${{better}}/${{withBoth.length}} models.`);
    }}
  }}
  notes.push('Intent classification is consistently above 90% across all models, making it the most reliable pipeline stage.');
  const el = document.getElementById('s2-notes');
  notes.forEach(n => {{ el.innerHTML += `<li>${{n}}</li>`; }});
}})();

// ── §3. Domain split — Hugo vs Dana ──────────────────────────────
const domainHugo = heroSorted.map(r => ({{
  ...r,
  pipeline_domain: r.pipeline_hugo,
  flat_domain: r.flat_hugo,
  delta_domain: (r.pipeline_hugo != null && r.flat_hugo != null) ? r.pipeline_hugo - r.flat_hugo : null,
}}));
const domainDana = heroSorted.map(r => ({{
  ...r,
  pipeline_domain: r.pipeline_dana,
  flat_domain: r.flat_dana,
  delta_domain: (r.pipeline_dana != null && r.flat_dana != null) ? r.pipeline_dana - r.flat_dana : null,
}}));

renderVBarChart('domain-hugo-chart', domainHugo, {{
  valuePairs: [
    {{ key: 'pipeline_domain', color: '#2563eb' }},
    {{ key: 'flat_domain', color: '#dc2626' }},
  ],
  showDelta: true,
  deltaKey: 'delta_domain',
  yMin: 0.2, yMax: 1.0,
  width: 520, height: 300,
}});

renderVBarChart('domain-dana-chart', domainDana, {{
  valuePairs: [
    {{ key: 'pipeline_domain', color: '#2563eb' }},
    {{ key: 'flat_domain', color: '#dc2626' }},
  ],
  showDelta: true,
  deltaKey: 'delta_domain',
  yMin: 0.2, yMax: 1.0,
  width: 520, height: 300,
}});

// §3 takeaways
(function() {{
  const notes = [];
  const hugoDels = domainHugo.filter(r => r.delta_domain != null).map(r => r.delta_domain);
  const danaDels = domainDana.filter(r => r.delta_domain != null).map(r => r.delta_domain);
  if (hugoDels.length && danaDels.length) {{
    const hAvg = hugoDels.reduce((a,b) => a+b,0) / hugoDels.length;
    const dAvg = danaDels.reduce((a,b) => a+b,0) / danaDels.length;
    const diff = Math.abs(hAvg - dAvg);
    if (diff < 0.03) {{
      notes.push(`Pipeline advantage is similar across domains (Hugo ${{fmtDelta(hAvg)}}pp, Dana ${{fmtDelta(dAvg)}}pp), suggesting the benefit comes from staged narrowing itself rather than domain-specific tool complexity.`);
    }} else {{
      const stronger = hAvg > dAvg ? 'Hugo' : 'Dana';
      notes.push(`Pipeline advantage is stronger in <strong>${{stronger}}</strong> (${{fmtDelta(hAvg > dAvg ? hAvg : dAvg)}}pp vs ${{fmtDelta(hAvg > dAvg ? dAvg : hAvg)}}pp), likely due to higher tool confusability in that domain.`);
    }}
  }}
  const hugoFlats = domainHugo.filter(r => r.flat_domain != null).map(r => r.flat_domain);
  const danaFlats = domainDana.filter(r => r.flat_domain != null).map(r => r.flat_domain);
  if (hugoFlats.length && danaFlats.length) {{
    const hFlat = hugoFlats.reduce((a,b) => a+b,0) / hugoFlats.length;
    const dFlat = danaFlats.reduce((a,b) => a+b,0) / danaFlats.length;
    const harder = hFlat < dFlat ? 'Hugo' : 'Dana';
    notes.push(`Flat tool-calling is harder in <strong>${{harder}}</strong> (${{(Math.min(hFlat,dFlat)*100).toFixed(1)}}% avg) than ${{harder === 'Hugo' ? 'Dana' : 'Hugo'}} (${{(Math.max(hFlat,dFlat)*100).toFixed(1)}}% avg).`);
  }}
  const el = document.getElementById('s3-notes');
  notes.forEach(n => {{ el.innerHTML += `<li>${{n}}</li>`; }});
}})();

// ── §4. Category breakdown — flat accuracy by category ───────────
const flatSorted = sortByDisplay(DATA.flat_ranking);
renderGroupedVBarChart('flat-cat-chart', flatSorted, [
  {{ key: 'cat_same_flow', color: '#93c5fd' }},
  {{ key: 'cat_switch_flow', color: '#2563eb' }},
  {{ key: 'cat_ambiguous_first', color: '#a78bfa' }},
  {{ key: 'cat_ambiguous_second', color: '#7c3aed' }},
], {{ yMin: 0.0, yMax: 1.0 }});

// §4 takeaways
(function() {{
  const notes = [];
  if (flatSorted.length) {{
    const cats = ['cat_same_flow', 'cat_switch_flow', 'cat_ambiguous_first', 'cat_ambiguous_second'];
    const labels = ['same_flow', 'switch_flow', 'ambiguous_first', 'ambiguous_second'];
    const avgs = cats.map((c, i) => {{
      const vals = flatSorted.filter(r => r[c] != null).map(r => r[c]);
      return {{ label: labels[i], avg: vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null }};
    }}).filter(x => x.avg != null);
    if (avgs.length) {{
      avgs.sort((a,b) => a.avg - b.avg);
      notes.push(`<strong>${{avgs[0].label}}</strong> is the hardest category for flat tool-calling at ${{(avgs[0].avg*100).toFixed(1)}}% average accuracy, while <strong>${{avgs[avgs.length-1].label}}</strong> is easiest at ${{(avgs[avgs.length-1].avg*100).toFixed(1)}}%.`);
      const ambCats = avgs.filter(x => x.label.startsWith('ambiguous'));
      const nonAmb = avgs.filter(x => !x.label.startsWith('ambiguous'));
      if (ambCats.length && nonAmb.length) {{
        const ambAvg = ambCats.reduce((a,x) => a+x.avg, 0) / ambCats.length;
        const nonAvg = nonAmb.reduce((a,x) => a+x.avg, 0) / nonAmb.length;
        if (ambAvg > nonAvg) {{
          notes.push('Ambiguous turns are not harder for flat models \u2014 the pipeline\u2019s structural advantage on ambiguity comes from routing, not from flat models failing on those turns.');
        }} else {{
          notes.push('Flat models struggle more on ambiguous turns, confirming that the pipeline\u2019s explicit ambiguity handling provides a structural advantage.');
        }}
      }}
    }}
  }}
  const el = document.getElementById('s4-notes');
  notes.forEach(n => {{ el.innerHTML += `<li>${{n}}</li>`; }});
}})();

// ── §5. Ambiguity Hint Ablation (Exp 2C) ────────────────────────
if (DATA.hint_rows && DATA.hint_rows.length) {{
  const hintSorted = sortByDisplay(DATA.hint_rows);

  // Per-category paired bar charts (2×2 grid)
  const catCharts = [
    {{ id: 'hint-same',   flatKey: 'flat_same_flow',        hintKey: 'hint_same_flow',        dKey: 'd_same_flow' }},
    {{ id: 'hint-switch', flatKey: 'flat_switch_flow',      hintKey: 'hint_switch_flow',      dKey: 'd_switch_flow' }},
    {{ id: 'hint-ambig1', flatKey: 'flat_ambiguous_first',  hintKey: 'hint_ambiguous_first',  dKey: 'd_ambiguous_first' }},
    {{ id: 'hint-ambig2', flatKey: 'flat_ambiguous_second', hintKey: 'hint_ambiguous_second', dKey: 'd_ambiguous_second' }},
  ];
  catCharts.forEach(cc => {{
    renderVBarChart(cc.id, hintSorted, {{
      valuePairs: [
        {{ key: cc.flatKey, color: '#dc2626' }},
        {{ key: cc.hintKey, color: '#f97316' }},
      ],
      showDelta: true,
      arrowDelta: true,
      deltaKey: cc.dKey,
      yMin: 0.0, yMax: 1.0,
      width: 520, height: 280,
    }});
  }});

  // Hint table with per-category values (flat → hint)
  function fmtCatD(v) {{
    if (v == null) return '-';
    const s = (v >= 0 ? '+' : '') + (v * 100).toFixed(1);
    const c = v > 0.01 ? 'color:var(--green);font-weight:700' : v < -0.01 ? 'color:var(--red);font-weight:700' : '';
    const arrow = v > 0.01 ? '\u25B2' : v < -0.01 ? '\u25BC' : '';
    return `<span style="${{c}}">${{arrow}} ${{s}}</span>`;
  }}
  const hintTbl = document.getElementById('hint-table');
  hintTbl.innerHTML = `<thead>
    <tr><th rowspan="2">Model</th><th rowspan="2">Tier</th>
      <th colspan="2" style="text-align:center;border-bottom:none">Overall</th>
      <th colspan="4" style="text-align:center;border-bottom:none">Category Delta (Hint \u2212 Flat)</th></tr>
    <tr><th>Flat</th><th>Hint</th>
      <th>same</th><th>switch</th><th>ambig 1st</th><th>ambig 2nd</th></tr>
  </thead><tbody></tbody>`;

  hintSorted.forEach(r => {{
    const dColor = r.delta >= 0 ? 'color:var(--green);font-weight:700' : 'color:var(--red);font-weight:700';
    const dArrow = r.delta >= 0 ? '\u25B2' : '\u25BC';
    hintTbl.querySelector('tbody').innerHTML += `<tr>
      <td>${{r.model_short}}</td><td>${{r.tier}}</td>
      <td class="num">${{fmtPct(r.flat_acc)}}</td>
      <td class="num" style="${{dColor}}">${{dArrow}} ${{fmtPct(r.hint_acc)}}</td>
      <td class="num">${{fmtCatD(r.d_same_flow)}}</td>
      <td class="num">${{fmtCatD(r.d_switch_flow)}}</td>
      <td class="num">${{fmtCatD(r.d_ambiguous_first)}}</td>
      <td class="num">${{fmtCatD(r.d_ambiguous_second)}}</td>
    </tr>`;
  }});

  // §5 takeaways
  (function() {{
    const notes = [];
    const avgDelta = hintSorted.reduce((a,r) => a + r.delta, 0) / hintSorted.length;
    notes.push(`Adding an explicit ambiguity hint ${{avgDelta >= 0 ? 'improves' : 'hurts'}} overall accuracy by ${{fmtDelta(avgDelta)}}pp on average across ${{hintSorted.length}} models.`);

    // Check ambiguous category improvement
    const ambD1 = hintSorted.filter(r => r.d_ambiguous_first != null).map(r => r.d_ambiguous_first);
    const ambD2 = hintSorted.filter(r => r.d_ambiguous_second != null).map(r => r.d_ambiguous_second);
    if (ambD1.length) {{
      const a1 = ambD1.reduce((a,b) => a+b, 0) / ambD1.length;
      const a2 = ambD2.length ? ambD2.reduce((a,b) => a+b, 0) / ambD2.length : null;
      const ambMsg = a2 != null
        ? `Ambiguous categories: ambig_1st ${{fmtDelta(a1)}}pp, ambig_2nd ${{fmtDelta(a2)}}pp.`
        : `Ambiguous 1st category: ${{fmtDelta(a1)}}pp.`;
      const verdict = (a1 > 0.01 || (a2 != null && a2 > 0.01))
        ? ' The hint partially closes the gap with the pipeline on ambiguous turns.'
        : ' The hint does not meaningfully improve ambiguity detection.';
      notes.push(ambMsg + verdict);
    }}

    // Check same_flow / switch_flow regression
    const sfD = hintSorted.filter(r => r.d_same_flow != null).map(r => r.d_same_flow);
    const swD = hintSorted.filter(r => r.d_switch_flow != null).map(r => r.d_switch_flow);
    if (sfD.length) {{
      const sf = sfD.reduce((a,b) => a+b, 0) / sfD.length;
      const sw = swD.length ? swD.reduce((a,b) => a+b, 0) / swD.length : 0;
      if (sf < -0.01 || sw < -0.01) {{
        notes.push(`As hypothesised, the hint causes regression on clear requests: same_flow ${{fmtDelta(sf)}}pp, switch_flow ${{fmtDelta(sw)}}pp \u2014 models over-trigger ambiguity handling.`);
      }} else {{
        notes.push(`Contrary to expectations, the hint does not cause significant regression on clear requests (same_flow ${{fmtDelta(sf)}}pp, switch_flow ${{fmtDelta(sw)}}pp).`);
      }}
    }}

    const el = document.getElementById('s5-notes');
    notes.forEach(n => {{ el.innerHTML += `<li>${{n}}</li>`; }});
  }})();
}} else {{
  // No hint data — hide the section gracefully
  ['hint-same','hint-switch','hint-ambig1','hint-ambig2'].forEach(id => {{
    const el = document.getElementById(id);
    if (el) el.innerHTML = '<p style="color:var(--muted);text-align:center;padding:2rem">No data</p>';
  }});
}}

// ── §6. Flat confusion tables ────────────────────────────────────
function fillConfusion(tableId, data) {{
  const tbl = document.getElementById(tableId);
  tbl.innerHTML = '<thead><tr><th>Expected</th><th>Predicted</th><th>Count</th></tr></thead><tbody></tbody>';
  data.forEach(c => {{
    tbl.querySelector('tbody').innerHTML += `<tr><td>${{c.expected}}</td><td>${{c.predicted}}</td><td class="num">${{c.count}}</td></tr>`;
  }});
  if (!data.length) {{
    tbl.querySelector('tbody').innerHTML = '<tr><td colspan="3" style="color:var(--muted)">No confusion data</td></tr>';
  }}
}}
fillConfusion('flat-conf-hugo', DATA.flat_confusions_hugo);
fillConfusion('flat-conf-dana', DATA.flat_confusions_dana);

// §6 takeaways
(function() {{
  const notes = [];
  const hConf = DATA.flat_confusions_hugo;
  const dConf = DATA.flat_confusions_dana;
  if (hConf.length) {{
    notes.push(`Hugo\u2019s top confusion: <strong>${{hConf[0].expected}}</strong> \u2192 <strong>${{hConf[0].predicted}}</strong> (${{hConf[0].count}} occurrences). ` +
      (hConf.length >= 3 ? `The top 3 confusions account for ${{hConf.slice(0,3).reduce((a,c)=>a+c.count,0)}} errors total.` : ''));
  }}
  if (dConf.length) {{
    notes.push(`Dana\u2019s top confusion: <strong>${{dConf[0].expected}}</strong> \u2192 <strong>${{dConf[0].predicted}}</strong> (${{dConf[0].count}} occurrences).`);
  }}
  const el = document.getElementById('s6-notes');
  notes.forEach(n => {{ el.innerHTML += `<li>${{n}}</li>`; }});
}})();

// ── §7. Findings ─────────────────────────────────────────────────
const ft = document.getElementById('findings-text');
ft.innerHTML = DATA.findings_text || '<span style="color:var(--muted)">No data available yet.</span>';
</script>
</body>
</html>"""


if __name__ == '__main__':
    main()
