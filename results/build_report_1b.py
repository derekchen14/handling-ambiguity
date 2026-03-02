#!/usr/bin/env python3
"""Build the exp1b HTML report: load raw predictions, compose ensembles in memory, score, generate report."""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from helpers.bootstrap import (  # noqa: E402
    CONFIGS_DIR,
    DOMAINS,
    EXP1A_DIR,
    MODEL_COST,
    MODEL_SHORT,
    build_eval_lookup,
    load_eval_sets,
    load_predictions_1a,
    load_predictions_1b,
)
from helpers.scoring import score_turn_ensemble  # noqa: E402

# ── Paths ────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent
REPORT_PATH = RESULTS_DIR / 'reports' / 'exp1b_report.html'

SKIPPED = {'5v-4', '10v-2', '1v-temp06'}
SELF_CONSISTENCY_IDS = {'3v-1', '3v-2', '3v-3'}
BOOTSTRAP_SEED = 42
N_BOOTSTRAP = 5


def parse_n_voters(ensemble_id: str) -> int:
    """Parse voter count from ensemble ID like '3v-1' → 3."""
    m = re.match(r'^(\d+)v', ensemble_id)
    return int(m.group(1)) if m else 0


# ── Scoring ──────────────────────────────────────────────────────────────

def score_voter_set(
    voters: list[dict[str, dict[int, list[str]]]],
    eval_lookup: dict[str, dict[int, dict]],
) -> dict[str, float]:
    """Score an ensemble given voter predictions against eval gold labels.

    Uses score_turn_ensemble from runner/scoring.py for correct two-criteria
    ambiguity scoring.  Also tracks per-criterion results on ambiguous turns:
      - ambiguous_recognized: % where criterion 1 passed
      - ambiguous_selected: of recognized, % where criterion 2 passed
    """
    by_cat: dict[str, list[bool]] = defaultdict(list)
    amb_recognized: list[bool] = []
    amb_selected: list[bool] = []

    for cid, turns_info in eval_lookup.items():
        for turn_num, info in turns_info.items():
            voter_flow_lists = [
                v.get(cid, {}).get(turn_num, []) for v in voters
            ]
            correct = score_turn_ensemble(
                info['category'],
                voter_flow_lists,
                info['flow'],
                info['candidate_flows'],
            )
            by_cat[info['category']].append(correct)

            # Track per-criterion breakdown for ambiguous turns
            if info['candidate_flows']:
                active = [vf for vf in voter_flow_lists if vf]
                if not active:
                    amb_recognized.append(False)
                    continue
                # Criterion 1: ensemble recognised ambiguity?
                any_multi = any(len(vf) >= 2 for vf in active)
                set_counts: Counter[frozenset[str]] = Counter()
                for vf in active:
                    set_counts[frozenset(vf)] += 1
                top_count = set_counts.most_common(1)[0][1]
                no_majority = top_count <= len(active) / 2
                recognized = any_multi or no_majority
                amb_recognized.append(recognized)

                if recognized:
                    # Criterion 2: highest-count flow in candidate set?
                    counts: Counter[str] = Counter()
                    for vf in active:
                        for flow in vf:
                            counts[flow] += 1
                    if counts:
                        max_count = counts.most_common(1)[0][1]
                        tied = [f for f, c in counts.items() if c == max_count]
                        predicted = random.choice(tied)
                        amb_selected.append(
                            predicted in set(info['candidate_flows']),
                        )
                    else:
                        amb_selected.append(False)

    all_results = [v for vals in by_cat.values() for v in vals]

    def acc(key: str) -> float:
        vals = by_cat.get(key, [])
        return sum(vals) / len(vals) if vals else 0.0

    amb_all = by_cat.get('ambiguous_first', []) + by_cat.get('ambiguous_second', [])

    return {
        'accuracy': sum(all_results) / len(all_results) if all_results else 0.0,
        'same_flow': acc('same_flow'),
        'switch_flow': acc('switch_flow'),
        'ambiguous_first': acc('ambiguous_first'),
        'ambiguous_second': acc('ambiguous_second'),
        'ambiguous_overall': sum(amb_all) / len(amb_all) if amb_all else 0.0,
        'ambiguous_recognized': (
            sum(amb_recognized) / len(amb_recognized) if amb_recognized else 0.0
        ),
        'ambiguous_selected': (
            sum(amb_selected) / len(amb_selected) if amb_selected else 0.0
        ),
    }


# ── Ensemble Composition ─────────────────────────────────────────────────

def _zero_stds() -> dict[str, float]:
    """Return zero-STD entries for all scored metrics."""
    return {
        'metric_std': 0.0,
        'same_flow_std': 0.0,
        'switch_flow_std': 0.0,
        'ambiguous_first_std': 0.0,
        'ambiguous_second_std': 0.0,
        'ambiguous_overall_std': 0.0,
        'ambiguous_recognized_std': 0.0,
        'ambiguous_selected_std': 0.0,
    }


_BOOTSTRAP_KEYS = [
    'accuracy', 'same_flow', 'switch_flow',
    'ambiguous_first', 'ambiguous_second',
    'ambiguous_overall', 'ambiguous_recognized', 'ambiguous_selected',
]


def compose_self_consistency(
    preds_1b: dict, eid: str, eval_lookup: dict,
) -> dict | None:
    """Self-consistency: 3 exp1b seeds are the 3 voters. Single result, STD=0."""
    ens_data = preds_1b.get(eid, {})
    seeds = sorted(ens_data.keys())
    if len(seeds) < 3:
        return None
    voters = [ens_data[s] for s in seeds[:3]]
    random.seed(BOOTSTRAP_SEED)
    scores = score_voter_set(voters, eval_lookup)
    return {**_zero_stds(), **scores}


def compose_cross_model(
    preds_1a: dict, composition: list[str], eval_lookup: dict,
) -> dict | None:
    """Cross-model bootstrap: pick 1 random seed per member, repeat N_BOOTSTRAP times."""
    rng = random.Random(BOOTSTRAP_SEED)

    member_seeds: dict[str, list[int]] = {}
    for config_id in composition:
        available = sorted(preds_1a.get(config_id, {}).keys())
        if not available:
            return None
        member_seeds[config_id] = available

    results = []
    for i in range(N_BOOTSTRAP):
        random.seed(BOOTSTRAP_SEED + i)
        voters = [
            preds_1a[cid][rng.choice(member_seeds[cid])]
            for cid in composition
        ]
        results.append(score_voter_set(voters, eval_lookup))

    n = len(results)
    out: dict[str, float] = {}
    for key in _BOOTSTRAP_KEYS:
        vals = [r[key] for r in results]
        out[key] = mean(vals)
        std_key = 'metric_std' if key == 'accuracy' else key + '_std'
        out[std_key] = stdev(vals) if n > 1 else 0.0
    return out


def compose_mixed_temp(
    preds_1a: dict, preds_1b: dict, eval_lookup: dict,
) -> dict | None:
    """3v-8: Sonnet @ t=0.0 (1A 1a_004) + t=0.3 (3v-2) + t=0.6 (3v-8). Single result, STD=0."""
    voter_t0 = preds_1a.get('1a_004', {}).get(1)
    voter_t03 = preds_1b.get('3v-2', {}).get(1)
    voter_t06 = preds_1b.get('3v-8', {}).get(1)
    if not all([voter_t0, voter_t03, voter_t06]):
        return None
    random.seed(BOOTSTRAP_SEED)
    scores = score_voter_set([voter_t0, voter_t03, voter_t06], eval_lookup)
    return {**_zero_stds(), **scores}


def _ensemble_cost(cfg: dict, exp1a_configs: dict[str, dict], n_voters: int) -> float:
    """Compute relative cost for an ensemble config."""
    composition = cfg.get('composition', [])
    if len(composition) == 1 and n_voters > 1:
        # Self-consistency or single-model temperature diversity
        model_id = exp1a_configs.get(composition[0], {}).get('model_id', '')
        return n_voters * MODEL_COST.get(model_id, 1.0)
    # Cross-model: sum of each voter's cost
    total = 0.0
    for cid in composition:
        model_id = exp1a_configs.get(cid, {}).get('model_id', '')
        total += MODEL_COST.get(model_id, 1.0)
    return total


def compute_all_ensembles(
    ensemble_configs: list[dict],
    preds_1a: dict,
    preds_1b: dict,
    eval_lookup: dict,
    exp1a_configs: dict[str, dict] | None = None,
) -> list[dict]:
    """Iterate configs, dispatch by type, bootstrap where needed."""
    results = []

    for cfg in ensemble_configs:
        eid = cfg['ensemble_id']
        if eid in SKIPPED:
            continue

        if eid in SELF_CONSISTENCY_IDS:
            ens_type = 'self-consistency'
            result = compose_self_consistency(preds_1b, eid, eval_lookup)
        elif eid == '3v-8':
            ens_type = 'mixed-temp'
            result = compose_mixed_temp(preds_1a, preds_1b, eval_lookup)
        elif cfg.get('bootstrappable'):
            ens_type = 'cross-model'
            result = compose_cross_model(
                preds_1a, cfg['composition'], eval_lookup,
            )
        else:
            continue

        if result is None:
            print(f'  SKIP {eid}: insufficient data')
            continue

        n_v = parse_n_voters(eid)
        cost = _ensemble_cost(cfg, exp1a_configs or {}, n_v)

        results.append({
            'id': eid,
            'n_voters': n_v,
            'type': ens_type,
            'description': cfg.get('description', ''),
            'relative_cost': round(cost, 3),
            'metric_value': round(result['accuracy'], 5),
            'metric_std': round(result['metric_std'], 5),
            'same_flow': round(result['same_flow'], 5),
            'switch_flow': round(result['switch_flow'], 5),
            'ambiguous_first': round(result['ambiguous_first'], 5),
            'ambiguous_second': round(result['ambiguous_second'], 5),
            'ambiguous_overall': round(result['ambiguous_overall'], 5),
            'ambiguous_recognized': round(result['ambiguous_recognized'], 5),
            'ambiguous_selected': round(result['ambiguous_selected'], 5),
            'same_flow_std': round(result.get('same_flow_std', 0), 5),
            'switch_flow_std': round(result.get('switch_flow_std', 0), 5),
            'ambiguous_first_std': round(result.get('ambiguous_first_std', 0), 5),
            'ambiguous_second_std': round(result.get('ambiguous_second_std', 0), 5),
            'ambiguous_overall_std': round(result.get('ambiguous_overall_std', 0), 5),
            'ambiguous_recognized_std': round(result.get('ambiguous_recognized_std', 0), 5),
            'ambiguous_selected_std': round(result.get('ambiguous_selected_std', 0), 5),
        })

    results.sort(key=lambda x: x['metric_value'], reverse=True)
    for i, r in enumerate(results):
        r['rank'] = i + 1
    return results


# ── Baselines ────────────────────────────────────────────────────────────

def _baseline_ambiguity_detail(
    preds_1a: dict,
    eval_lookup: dict[str, dict[int, dict]],
    config_id: str,
) -> dict[str, float]:
    """Compute recognition/selection breakdown for a single-model baseline.

    Uses the single-model criteria (len >= 2 for recognition, random pick for
    selection) — NOT the ensemble path — so baseline accuracy stays canonical.
    """
    seeds = sorted(preds_1a.get(config_id, {}).keys())
    if not seeds:
        return {'ambiguous_recognized': 0.0, 'ambiguous_selected': 0.0,
                'ambiguous_overall': 0.0}

    all_recog: list[list[bool]] = []
    all_sel: list[list[bool]] = []
    all_overall: list[list[bool]] = []

    for i, seed in enumerate(seeds):
        random.seed(BOOTSTRAP_SEED + i)
        seed_preds = preds_1a[config_id][seed]
        recog: list[bool] = []
        sel: list[bool] = []
        overall: list[bool] = []
        for cid, turns_info in eval_lookup.items():
            for turn_num, info in turns_info.items():
                if not info['candidate_flows']:
                    continue
                detected = seed_preds.get(cid, {}).get(turn_num, [])
                recognized = len(detected) >= 2
                recog.append(recognized)
                if recognized:
                    predicted = random.choice(detected)
                    selected = predicted in set(info['candidate_flows'])
                    sel.append(selected)
                    overall.append(selected)
                else:
                    overall.append(False)
        all_recog.append(recog)
        all_sel.append(sel)
        all_overall.append(overall)

    def avg_rate(lists: list[list[bool]]) -> float:
        rates = [sum(l) / len(l) if l else 0.0 for l in lists]
        return mean(rates)

    def std_rate(lists: list[list[bool]]) -> float:
        rates = [sum(l) / len(l) if l else 0.0 for l in lists]
        return stdev(rates) if len(rates) > 1 else 0.0

    return {
        'ambiguous_recognized': avg_rate(all_recog),
        'ambiguous_selected': avg_rate(all_sel),
        'ambiguous_overall': avg_rate(all_overall),
        'ambiguous_recognized_std': std_rate(all_recog),
        'ambiguous_selected_std': std_rate(all_sel),
        'ambiguous_overall_std': std_rate(all_overall),
    }


def load_baselines(
    exp1a_configs: dict[str, dict],
    preds_1a: dict,
    eval_lookup: dict[str, dict[int, dict]],
) -> list[dict]:
    """Read 1A summary JSONs for accuracy, add recognition/selection from raw preds."""
    by_config: dict[str, list[dict]] = defaultdict(list)

    for path in sorted(EXP1A_DIR.glob('*_summary.json')):
        stem = path.stem.replace('_summary', '')
        parts = stem.split('_seed')
        if len(parts) != 2:
            continue
        prefix = parts[0]

        config_id = None
        for domain in DOMAINS:
            if prefix.startswith(domain + '_'):
                config_id = prefix[len(domain) + 1:]
                break
        if not config_id or config_id not in exp1a_configs:
            continue

        with open(path) as f:
            data = json.load(f)
        summary = data.get('summary', {})
        if not summary:
            continue

        cats = summary.get('accuracy_by_category', {})
        by_config[config_id].append({
            'accuracy': summary.get('accuracy_top1', 0),
            'same_flow': cats.get('same_flow', 0),
            'switch_flow': cats.get('switch_flow', 0),
            'ambiguous_first': cats.get('ambiguous_first', 0),
            'ambiguous_second': cats.get('ambiguous_second', 0),
        })

    baselines = []
    for config_id, runs in by_config.items():
        cfg = exp1a_configs[config_id]
        model_id = cfg['model_id']
        n = len(runs)

        # Recognition/selection from raw predictions
        detail = _baseline_ambiguity_detail(preds_1a, eval_lookup, config_id)

        baselines.append({
            'config_id': config_id,
            'model_id': model_id,
            'model_short': MODEL_SHORT.get(model_id, model_id),
            'relative_cost': round(MODEL_COST.get(model_id, 1.0), 3),
            'metric_value': round(mean(r['accuracy'] for r in runs), 5),
            'metric_std': round(stdev(r['accuracy'] for r in runs), 5) if n > 1 else 0.0,
            'same_flow': round(mean(r['same_flow'] for r in runs), 5),
            'switch_flow': round(mean(r['switch_flow'] for r in runs), 5),
            'ambiguous_first': round(mean(r['ambiguous_first'] for r in runs), 5),
            'ambiguous_second': round(mean(r['ambiguous_second'] for r in runs), 5),
            'ambiguous_overall': round(detail['ambiguous_overall'], 5),
            'ambiguous_recognized': round(detail['ambiguous_recognized'], 5),
            'ambiguous_selected': round(detail['ambiguous_selected'], 5),
            'same_flow_std': round(stdev(r['same_flow'] for r in runs), 5) if n > 1 else 0.0,
            'switch_flow_std': round(stdev(r['switch_flow'] for r in runs), 5) if n > 1 else 0.0,
            'ambiguous_first_std': round(stdev(r['ambiguous_first'] for r in runs), 5) if n > 1 else 0.0,
            'ambiguous_second_std': round(stdev(r['ambiguous_second'] for r in runs), 5) if n > 1 else 0.0,
            'ambiguous_overall_std': round(detail.get('ambiguous_overall_std', 0), 5),
            'ambiguous_recognized_std': round(detail.get('ambiguous_recognized_std', 0), 5),
            'ambiguous_selected_std': round(detail.get('ambiguous_selected_std', 0), 5),
        })

    baselines.sort(key=lambda x: x['metric_value'], reverse=True)
    return baselines[:3]


# ── Report Building ──────────────────────────────────────────────────────

def build_report_data(ranking: list[dict], baselines: list[dict]) -> dict:
    """Assemble the DATA dict for the HTML template."""
    best_ens = ranking[0] if ranking else {}
    best_base = baselines[0] if baselines else {}
    sc_entries = [r for r in ranking if r['type'] == 'self-consistency']

    return {
        'generated': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'n_ensembles': len(ranking),
        'best_ensemble_acc': best_ens.get('metric_value', 0),
        'best_ensemble_id': best_ens.get('id', '?'),
        'best_baseline_acc': best_base.get('metric_value', 0),
        'best_baseline_model': best_base.get('model_short', '?'),
        'ranking': ranking,
        'baselines': baselines,
        'self_consistency': sc_entries,
    }


def inject_into_html(data: dict) -> None:
    """Regex-replace const DATA in the HTML template."""
    html = REPORT_PATH.read_text()
    data_json = json.dumps(data, indent=None)
    replacement = f'const DATA = {data_json};'
    html = re.sub(
        r'const DATA = .*?;',
        lambda m: replacement,
        html,
        count=1,
        flags=re.DOTALL,
    )
    REPORT_PATH.write_text(html)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print('Loading eval sets...')
    eval_sets = load_eval_sets()
    eval_lookup = build_eval_lookup(eval_sets)
    n_convos = sum(len(v) for v in eval_sets.values())
    n_turns = sum(len(t) for t in eval_lookup.values())
    print(f'  {n_convos} conversations, {n_turns} scored turns')

    print('Loading 1A predictions...')
    preds_1a = load_predictions_1a()
    print(f'  {len(preds_1a)} configs')

    print('Loading 1B predictions...')
    preds_1b = load_predictions_1b()
    print(f'  {len(preds_1b)} ensemble IDs')

    with open(CONFIGS_DIR / 'exp1b_ensembles_resolved.json') as f:
        ensemble_configs = json.load(f)
    with open(CONFIGS_DIR / 'exp1a_configs.json') as f:
        exp1a_configs = {c['config_id']: c for c in json.load(f)}

    print('Computing ensembles...')
    ranking = compute_all_ensembles(
        ensemble_configs, preds_1a, preds_1b, eval_lookup, exp1a_configs,
    )
    print(f'  {len(ranking)} ensembles scored')

    print('Loading baselines...')
    baselines = load_baselines(exp1a_configs, preds_1a, eval_lookup)
    print(f'  {len(baselines)} baselines')

    data = build_report_data(ranking, baselines)
    inject_into_html(data)

    print(f'\nReport written to {REPORT_PATH}')
    print(f'  {len(ranking)} ensembles')
    if ranking:
        best = ranking[0]
        print(
            f'  Best: {best["id"]} @ {best["metric_value"]:.1%}'
            f' (\u00b1{best["metric_std"]:.4f})',
        )
    if baselines:
        bl = baselines[0]
        print(f'  Best baseline: {bl["model_short"]} @ {bl["metric_value"]:.1%}')


if __name__ == '__main__':
    main()
