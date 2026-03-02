#!/usr/bin/env python3
"""Build the exp1c HTML report: confidence calibration analysis.

Reuses 1A/1B prediction data (0 API calls). Computes calibration metrics
for the top 10 ensembles + top 4 single-model baselines (14 configs total).

Single models: each detected flow becomes a pseudo-voter, so
confidence = 1 / len(detected_flows). This gives a natural calibration
signal — models outputting multiple flows on ambiguous turns are less
confident.

Ensembles: confidence = voter agreement (via tally_votes_multi).
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from helpers.bootstrap import (  # noqa: E402
    CONFIGS_DIR,
    MODEL_COST,
    build_eval_lookup,
    load_baselines,
    load_eval_sets,
    load_predictions_1a,
    load_predictions_1b,
)
from helpers.metrics import (  # noqa: E402
    brier_score,
    ece,
    mce,
    overconfidence_rate,
    reliability_diagram,
    underconfidence_rate,
)
from helpers.scoring import score_turn, score_turn_ensemble, tally_votes_multi  # noqa: E402

# ── Paths ────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent
REPORT_PATH = RESULTS_DIR / 'reports' / 'exp1c_report.html'

DOMAINS = ['dana', 'hugo']
SKIPPED = {'5v-4', '10v-2', '1v-temp06'}
SELF_CONSISTENCY_IDS = {'3v-1', '3v-2', '3v-3'}
BOOTSTRAP_SEED = 42
N_BOOTSTRAP = 5
CATEGORIES = ['same_flow', 'switch_flow', 'ambiguous_first', 'ambiguous_second']


def parse_n_voters(ensemble_id: str) -> int:
    m = re.match(r'^(\d+)v', ensemble_id)
    return int(m.group(1)) if m else 0


# ── Confidence-aware Scoring ─────────────────────────────────────────────

def score_with_confidence_ensemble(
    voters: list[dict[str, dict[int, list[str]]]],
    eval_lookup: dict[str, dict[int, dict]],
) -> list[dict]:
    """Score an ensemble and record per-turn confidence + correctness.

    Returns list of {confidence, correct, category, domain} records.
    """
    records = []
    for cid, turns_info in eval_lookup.items():
        for turn_num, info in turns_info.items():
            voter_flow_lists = [
                v.get(cid, {}).get(turn_num, []) for v in voters
            ]
            _, confidence = tally_votes_multi(voter_flow_lists)
            correct = score_turn_ensemble(
                info['category'],
                voter_flow_lists,
                info['flow'],
                info['candidate_flows'],
            )
            records.append({
                'confidence': confidence,
                'correct': correct,
                'category': info['category'],
                'domain': info['domain'],
            })
    return records


def score_with_confidence_single(
    preds: dict[str, dict[int, list[str]]],
    eval_lookup: dict[str, dict[int, dict]],
) -> list[dict]:
    """Score a single model with decomposed pseudo-voter confidence.

    Each detected flow becomes a pseudo-voter with a single flow.
    confidence = 1 / len(detected_flows) via tally_votes_multi.
    """
    records = []
    for cid, turns_info in eval_lookup.items():
        for turn_num, info in turns_info.items():
            detected_flows = preds.get(cid, {}).get(turn_num, [])

            if detected_flows:
                pseudo_voters = [[f] for f in detected_flows]
                _, confidence = tally_votes_multi(pseudo_voters)
            else:
                confidence = 0.0

            correct = score_turn(
                info['category'],
                detected_flows,
                info['flow'],
                info['candidate_flows'],
            )
            records.append({
                'confidence': confidence,
                'correct': correct,
                'category': info['category'],
                'domain': info['domain'],
            })
    return records


# ── Calibration Metrics ──────────────────────────────────────────────────

def compute_calibration(records: list[dict]) -> dict:
    """Compute calibration metrics from a list of {confidence, correct} records."""
    if not records:
        return {
            'ece': 0.0, 'mce': 0.0, 'brier': 0.0,
            'overconfidence_rate': 0.0, 'underconfidence_rate': 0.0,
            'spearman_r': 0.0, 'mean_confidence': 0.0, 'accuracy': 0.0,
            'n_turns': 0, 'bin_confs': [], 'bin_accs': [],
        }

    confs = np.array([r['confidence'] for r in records])
    correct = np.array([r['correct'] for r in records], dtype=float)

    bin_confs_arr, bin_accs_arr, bin_counts_arr = reliability_diagram(confs, correct)

    if np.std(confs) < 1e-10 or np.std(correct) < 1e-10:
        sp_r = 0.0
    else:
        sp_r = float(spearmanr(confs, correct).statistic)

    return {
        'ece': round(float(ece(confs, correct)), 5),
        'mce': round(float(mce(confs, correct)), 5),
        'brier': round(float(brier_score(confs, correct)), 5),
        'overconfidence_rate': round(float(overconfidence_rate(confs, correct)), 5),
        'underconfidence_rate': round(float(underconfidence_rate(confs, correct)), 5),
        'spearman_r': round(sp_r, 5),
        'mean_confidence': round(float(confs.mean()), 5),
        'accuracy': round(float(correct.mean()), 5),
        'n_turns': len(records),
        'bin_confs': [round(float(x), 5) if not np.isnan(x) else None for x in bin_confs_arr],
        'bin_accs': [round(float(x), 5) if not np.isnan(x) else None for x in bin_accs_arr],
        'bin_counts': [int(x) for x in bin_counts_arr],
    }


def compute_calibration_grouped(records: list[dict]) -> dict:
    """Compute calibration overall + by domain + by category."""
    overall = compute_calibration(records)

    by_domain = {}
    domain_groups = defaultdict(list)
    for r in records:
        domain_groups[r['domain']].append(r)
    for domain in DOMAINS:
        by_domain[domain] = compute_calibration(domain_groups.get(domain, []))

    by_category = {}
    cat_groups = defaultdict(list)
    for r in records:
        cat_groups[r['category']].append(r)
    for cat in CATEGORIES:
        by_category[cat] = compute_calibration(cat_groups.get(cat, []))

    return {
        'overall': overall,
        'by_domain': by_domain,
        'by_category': by_category,
    }


# ── Ensemble Composition (mirrors 1B dispatch) ──────────────────────────

def calibrate_cross_model(
    preds_1a: dict,
    composition: list[str],
    eval_lookup: dict,
) -> list[dict] | None:
    """Cross-model bootstrap: N_BOOTSTRAP passes, collect all per-turn records."""
    rng = random.Random(BOOTSTRAP_SEED)

    member_seeds: dict[str, list[int]] = {}
    for config_id in composition:
        available = sorted(preds_1a.get(config_id, {}).keys())
        if not available:
            return None
        member_seeds[config_id] = available

    all_records = []
    for i in range(N_BOOTSTRAP):
        random.seed(BOOTSTRAP_SEED + i)
        voters = [
            preds_1a[cid][rng.choice(member_seeds[cid])]
            for cid in composition
        ]
        records = score_with_confidence_ensemble(voters, eval_lookup)
        all_records.extend(records)
    return all_records


def calibrate_self_consistency(
    preds_1b: dict, eid: str, eval_lookup: dict,
) -> list[dict] | None:
    """Self-consistency: 3 exp1b seeds are the 3 voters. Single pass."""
    ens_data = preds_1b.get(eid, {})
    seeds = sorted(ens_data.keys())
    if len(seeds) < 3:
        return None
    voters = [ens_data[s] for s in seeds[:3]]
    random.seed(BOOTSTRAP_SEED)
    return score_with_confidence_ensemble(voters, eval_lookup)


def calibrate_mixed_temp(
    preds_1a: dict, preds_1b: dict, eval_lookup: dict,
) -> list[dict] | None:
    """3v-8: Sonnet @ t=0.0 (1A 1a_004) + t=0.3 (3v-2) + t=0.6 (3v-8)."""
    voter_t0 = preds_1a.get('1a_004', {}).get(1)
    voter_t03 = preds_1b.get('3v-2', {}).get(1)
    voter_t06 = preds_1b.get('3v-8', {}).get(1)
    if not all([voter_t0, voter_t03, voter_t06]):
        return None
    random.seed(BOOTSTRAP_SEED)
    return score_with_confidence_ensemble([voter_t0, voter_t03, voter_t06], eval_lookup)


def calibrate_single_model(
    preds_1a: dict,
    config_id: str,
    eval_lookup: dict,
) -> list[dict] | None:
    """Single model: average across seeds. Each detected flow -> pseudo-voter."""
    seeds = sorted(preds_1a.get(config_id, {}).keys())
    if not seeds:
        return None

    all_records = []
    for i, seed in enumerate(seeds):
        random.seed(BOOTSTRAP_SEED + i)
        seed_preds = preds_1a[config_id][seed]
        records = score_with_confidence_single(seed_preds, eval_lookup)
        all_records.extend(records)
    return all_records


# ── Ensemble cost ────────────────────────────────────────────────────────

def _ensemble_cost(cfg: dict, exp1a_configs: dict[str, dict], n_voters: int) -> float:
    composition = cfg.get('composition', [])
    if len(composition) == 1 and n_voters > 1:
        model_id = exp1a_configs.get(composition[0], {}).get('model_id', '')
        return n_voters * MODEL_COST.get(model_id, 1.0)
    total = 0.0
    for cid in composition:
        model_id = exp1a_configs.get(cid, {}).get('model_id', '')
        total += MODEL_COST.get(model_id, 1.0)
    return total


# ── Accuracy scoring (for ensemble ranking) ─────────────────────────────

def score_voter_set(
    voters: list[dict[str, dict[int, list[str]]]],
    eval_lookup: dict[str, dict[int, dict]],
) -> dict[str, float]:
    """Score an ensemble for accuracy. Returns {accuracy}."""
    results: list[bool] = []
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
            results.append(correct)
    return {'accuracy': sum(results) / len(results) if results else 0.0}


def compute_all_ensembles_ranked(
    ensemble_configs: list[dict],
    preds_1a: dict,
    preds_1b: dict,
    eval_lookup: dict,
    exp1a_configs: dict[str, dict],
) -> list[dict]:
    """Compute all ensemble accuracies and return ranked list."""
    results = []

    for cfg in ensemble_configs:
        eid = cfg['ensemble_id']
        if eid in SKIPPED:
            continue

        if eid in SELF_CONSISTENCY_IDS:
            ens_type = 'self-consistency'
            ens_data = preds_1b.get(eid, {})
            seeds = sorted(ens_data.keys())
            if len(seeds) < 3:
                continue
            voters = [ens_data[s] for s in seeds[:3]]
            random.seed(BOOTSTRAP_SEED)
            result = score_voter_set(voters, eval_lookup)
        elif eid == '3v-8':
            ens_type = 'mixed-temp'
            voter_t0 = preds_1a.get('1a_004', {}).get(1)
            voter_t03 = preds_1b.get('3v-2', {}).get(1)
            voter_t06 = preds_1b.get('3v-8', {}).get(1)
            if not all([voter_t0, voter_t03, voter_t06]):
                continue
            random.seed(BOOTSTRAP_SEED)
            result = score_voter_set([voter_t0, voter_t03, voter_t06], eval_lookup)
        elif cfg.get('bootstrappable'):
            ens_type = 'cross-model'
            rng = random.Random(BOOTSTRAP_SEED)
            member_seeds: dict[str, list[int]] = {}
            skip = False
            for config_id in cfg['composition']:
                available = sorted(preds_1a.get(config_id, {}).keys())
                if not available:
                    skip = True
                    break
                member_seeds[config_id] = available
            if skip:
                continue
            boot_results = []
            for i in range(N_BOOTSTRAP):
                random.seed(BOOTSTRAP_SEED + i)
                voters = [
                    preds_1a[cid][rng.choice(member_seeds[cid])]
                    for cid in cfg['composition']
                ]
                boot_results.append(score_voter_set(voters, eval_lookup))
            result = {'accuracy': mean(r['accuracy'] for r in boot_results)}
        else:
            continue

        n_v = parse_n_voters(eid)
        cost = _ensemble_cost(cfg, exp1a_configs, n_v)

        results.append({
            'id': eid,
            'n_voters': n_v,
            'type': ens_type,
            'description': cfg.get('description', ''),
            'relative_cost': round(cost, 3),
            'accuracy': round(result['accuracy'], 5),
        })

    results.sort(key=lambda x: x['accuracy'], reverse=True)
    return results


# ── Main Pipeline ────────────────────────────────────────────────────────

def _ensemble_group(ens: dict) -> str:
    """Map an ensemble to its voter group key."""
    etype = ens.get('type', '')
    if etype in ('self-consistency', 'mixed-temp'):
        return 'temp'
    nv = ens.get('n_voters', 0)
    if nv == 3:
        return '3v'
    if nv == 5:
        return '5v'
    if nv >= 10:
        return '10v'
    return '3v'


def _config_label(cfg: dict) -> str:
    if 'model_short' in cfg:
        return cfg['model_short']
    return cfg.get('id', '?')


def _serialize_configs(configs: list[dict]) -> list[dict]:
    """Flatten config data for JSON serialization."""
    out = []
    for c in configs:
        cal = c['calibration']
        overall = cal['overall']
        out.append({
            'id': c.get('id', c.get('model_short', '?')),
            'label': _config_label(c),
            'group': c['group'],
            'accuracy': overall['accuracy'],
            'relative_cost': c.get('relative_cost', 0),
            'n_voters': c.get('n_voters', 1),
            'type': c.get('type', 'single-model'),
            'description': c.get('description', c.get('model_short', '')),
            'overall': overall,
            'by_domain': cal['by_domain'],
            'by_category': cal['by_category'],
        })
    return out


def build_report_data(
    ensemble_configs: list[dict],
    exp1a_configs: dict[str, dict],
    preds_1a: dict,
    preds_1b: dict,
    eval_lookup: dict,
) -> dict:
    """Run the full 1C pipeline and assemble the DATA dict."""

    # Step 1: Rank ensembles by accuracy, cap per group, select top 10
    print('Ranking ensembles by accuracy...')
    ranked = compute_all_ensembles_ranked(
        ensemble_configs, preds_1a, preds_1b, eval_lookup, exp1a_configs,
    )
    # Cap each voter group to 5 so no single group dominates the top 10
    group_counts: dict[str, int] = {}
    capped = []
    for ens in ranked:
        grp = _ensemble_group(ens)
        group_counts[grp] = group_counts.get(grp, 0) + 1
        if group_counts[grp] <= 5:
            capped.append(ens)
    top10 = capped[:10]
    print(f'  Top 10 ensembles: {[r["id"] for r in top10]}')

    # Step 2: Load top 4 baselines
    print('Loading baselines...')
    baselines = load_baselines(exp1a_configs, top_n=4)
    print(f'  Top 4 baselines: {[b["model_short"] for b in baselines]}')

    # Step 3: Build config-to-ensemble mapping for dispatch
    ens_cfg_map = {cfg['ensemble_id']: cfg for cfg in ensemble_configs}

    # Step 4: Calibrate each config
    print('Computing calibration metrics...')
    ensemble_results = []
    for ens in top10:
        eid = ens['id']
        cfg = ens_cfg_map.get(eid, {})
        print(f'  Calibrating {eid}...')

        if eid in SELF_CONSISTENCY_IDS:
            records = calibrate_self_consistency(preds_1b, eid, eval_lookup)
        elif eid == '3v-8':
            records = calibrate_mixed_temp(preds_1a, preds_1b, eval_lookup)
        elif cfg.get('bootstrappable'):
            records = calibrate_cross_model(preds_1a, cfg['composition'], eval_lookup)
        else:
            print(f'    SKIP {eid}: no dispatch path')
            continue

        if records is None:
            print(f'    SKIP {eid}: insufficient data')
            continue

        cal = compute_calibration_grouped(records)
        ensemble_results.append({**ens, 'group': _ensemble_group(ens), 'calibration': cal})

    baseline_results = []
    for bl in baselines:
        cid = bl['config_id']
        print(f'  Calibrating baseline {bl["model_short"]} ({cid})...')
        records = calibrate_single_model(preds_1a, cid, eval_lookup)
        if records is None:
            print(f'    SKIP {bl["model_short"]}: no data')
            continue
        cal = compute_calibration_grouped(records)
        baseline_results.append({**bl, 'group': 'baseline', 'calibration': cal})

    # Step 5: Assemble report data
    all_configs = ensemble_results + baseline_results

    best_ece_ens = min(
        (c for c in ensemble_results),
        key=lambda c: c['calibration']['overall']['ece'],
        default=None,
    )
    best_ece_single = min(
        (c for c in baseline_results),
        key=lambda c: c['calibration']['overall']['ece'],
        default=None,
    )
    worst_overconf = max(
        (c for c in all_configs),
        key=lambda c: c['calibration']['overall']['overconfidence_rate'],
        default=None,
    )

    return {
        'generated': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'n_configs': len(all_configs),
        'n_ensembles': len(ensemble_results),
        'n_baselines': len(baseline_results),
        'best_ece_ensemble': {
            'id': best_ece_ens['id'] if best_ece_ens else '?',
            'ece': best_ece_ens['calibration']['overall']['ece'] if best_ece_ens else 0,
        },
        'best_ece_single': {
            'id': best_ece_single['model_short'] if best_ece_single else '?',
            'ece': best_ece_single['calibration']['overall']['ece'] if best_ece_single else 0,
        },
        'worst_overconfidence': {
            'id': _config_label(worst_overconf) if worst_overconf else '?',
            'rate': worst_overconf['calibration']['overall']['overconfidence_rate'] if worst_overconf else 0,
        },
        'configs': _serialize_configs(all_configs),
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


def main():
    print('=== Experiment 1C — Confidence Calibration Report ===\n')

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

    data = build_report_data(
        ensemble_configs, exp1a_configs, preds_1a, preds_1b, eval_lookup,
    )

    inject_into_html(data)

    print(f'\nReport written to {REPORT_PATH}')
    print(f'  {data["n_configs"]} configs ({data["n_ensembles"]} ensembles + {data["n_baselines"]} baselines)')
    print(f'  Best ECE (ensemble): {data["best_ece_ensemble"]["id"]} @ {data["best_ece_ensemble"]["ece"]:.4f}')
    print(f'  Best ECE (single):   {data["best_ece_single"]["id"]} @ {data["best_ece_single"]["ece"]:.4f}')
    print(f'  Worst overconfidence: {data["worst_overconfidence"]["id"]} @ {data["worst_overconfidence"]["rate"]:.1%}')


if __name__ == '__main__':
    main()
