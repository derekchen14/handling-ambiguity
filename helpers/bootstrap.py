"""Ensemble composition from Experiment 1A data — zero API calls.

Also provides shared data-loading helpers used by report builders
(build_report_1b.py, build_report_1c.py).
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

log = logging.getLogger(__name__)

# ── Shared paths and constants ───────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / 'results'
EXP1A_DIR = RESULTS_DIR / 'exp1a'
EXP1B_DIR = RESULTS_DIR / 'exp1b'
EVAL_DIR = BASE_DIR / 'eval'
CONFIGS_DIR = BASE_DIR / 'helpers' / 'configs'

DOMAINS = ['dana', 'hugo']

MODEL_COST = {
    'claude-haiku-4-5-20251001': 0.27,
    'claude-sonnet-4-6': 1.00,
    'claude-opus-4-6': 5.00,
    'gemini-3-flash-preview': 0.05,
    'gemini-3-pro-preview': 0.83,
    'gpt-5-nano': 0.03,
    'gpt-5-mini': 0.13,
    'gpt-5.2': 1.00,
    'deepseek-chat': 0.09,
    'deepseek-reasoner': 0.18,
    'Qwen/Qwen2.5-7B-Instruct-Turbo': 0.05,
    'Qwen/Qwen3-Next-80B-A3B-Instruct': 0.10,
    'Qwen/Qwen3-235B-A22B-Thinking-2507': 0.50,
    'gemma-3-27b-it': 0.03,
}

MODEL_SHORT = {
    'claude-haiku-4-5-20251001': 'Haiku 4.5',
    'claude-sonnet-4-6': 'Sonnet 4.6',
    'claude-opus-4-6': 'Opus 4.6',
    'gemini-3-flash-preview': 'Gemini 3 Flash',
    'gemini-3-pro-preview': 'Gemini 3 Pro',
    'gpt-5-nano': 'GPT-5 Nano',
    'gpt-5-mini': 'GPT-5 Mini',
    'gpt-5.2': 'GPT-5.2',
    'deepseek-chat': 'DeepSeek V3',
    'deepseek-reasoner': 'DeepSeek R1',
    'Qwen/Qwen2.5-7B-Instruct-Turbo': 'Qwen 7B',
    'Qwen/Qwen3-Next-80B-A3B-Instruct': 'Qwen3 80B',
    'Qwen/Qwen3-235B-A22B-Thinking-2507': 'Qwen3 235B',
    'gemma-3-27b-it': 'Gemma 27B',
}


# ── Shared data-loading helpers ──────────────────────────────────────────

def load_eval_sets() -> dict[str, list[dict]]:
    """Load eval sets -> {domain: [convo_dicts]}."""
    result = {}
    for domain in DOMAINS:
        with open(EVAL_DIR / f'eval_{domain}.json') as f:
            result[domain] = json.load(f)
    return result


def build_eval_lookup(
    eval_sets: dict[str, list[dict]],
) -> dict[str, dict[int, dict]]:
    """Build {convo_id: {turn_num: {flow, candidate_flows, category, domain}}}."""
    lookup: dict[str, dict[int, dict]] = {}
    for domain, convos in eval_sets.items():
        for convo in convos:
            cid = convo['convo_id']
            category = convo.get('category', 'unknown')
            turns: dict[int, dict] = {}
            for turn in convo.get('turns', []):
                if turn.get('speaker') != 'user':
                    continue
                turns[turn['turn_num']] = {
                    'flow': turn.get('flow', ''),
                    'candidate_flows': turn.get('candidate_flows'),
                    'category': category,
                    'domain': domain,
                }
            lookup[cid] = turns
    return lookup


def _load_jsonl_predictions(directory: Path) -> dict:
    """Load per-turn predictions from JSONL files in a results directory.

    Returns: {key: {seed: {convo_id: {turn_num: detected_flows}}}}
    where key is config_id (1A) or ensemble_id (1B).
    """
    predictions: dict = {}
    for domain in DOMAINS:
        for path in sorted(directory.glob(f'{domain}_*.jsonl')):
            stem = path.stem
            parts = stem.split('_seed')
            if len(parts) != 2:
                continue
            seed = int(parts[1])
            prefix = parts[0]
            key = prefix[len(domain) + 1:]

            key_preds = predictions.setdefault(key, {})
            seed_preds = key_preds.setdefault(seed, {})

            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        cid = record['convo_id']
                        convo_preds = seed_preds.setdefault(cid, {})
                        for turn in record.get('turns', []):
                            convo_preds[turn['turn_num']] = turn.get(
                                'detected_flows', [],
                            )
                    except (json.JSONDecodeError, KeyError):
                        continue
    return predictions


def load_predictions_1a() -> dict:
    return _load_jsonl_predictions(EXP1A_DIR)


def load_predictions_1b() -> dict:
    return _load_jsonl_predictions(EXP1B_DIR)


def load_baselines(
    exp1a_configs: dict[str, dict],
    top_n: int = 3,
) -> list[dict]:
    """Read 1A summary JSONs for accuracy. Returns top N by accuracy.

    Each baseline entry: {config_id, model_id, model_short, relative_cost, accuracy}.
    """
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

        by_config[config_id].append({
            'accuracy': summary.get('accuracy_top1', 0),
        })

    baselines = []
    for config_id, runs in by_config.items():
        cfg = exp1a_configs[config_id]
        model_id = cfg['model_id']
        baselines.append({
            'config_id': config_id,
            'model_id': model_id,
            'model_short': MODEL_SHORT.get(model_id, model_id),
            'relative_cost': round(MODEL_COST.get(model_id, 1.0), 3),
            'accuracy': round(mean(r['accuracy'] for r in runs), 5),
        })

    baselines.sort(key=lambda x: x['accuracy'], reverse=True)
    return baselines[:top_n]


# ── EnsembleBootstrapper (original class) ────────────────────────────────

class EnsembleBootstrapper:
    """Compose cross-model ensembles from 1A per-turn predictions."""

    def load_predictions(
        self, results_dir: Path, domain: str,
    ) -> dict[str, dict[int, dict[str, dict[int, list[str]]]]]:
        """Load all per-turn predictions from exp1a results.

        Returns:
            {config_id: {seed: {convo_id: {turn_num: detected_flows_list}}}}
        """
        exp1a_dir = results_dir / 'exp1a'
        predictions: dict = {}

        if not exp1a_dir.exists():
            log.warning('No exp1a results found at %s', exp1a_dir)
            return predictions

        for path in sorted(exp1a_dir.glob(f'{domain}_*.jsonl')):
            # Parse filename: {domain}_{config_id}_seed{n}.jsonl
            stem = path.stem
            parts = stem.split('_seed')
            if len(parts) != 2:
                continue

            prefix = parts[0]
            seed = int(parts[1])
            # config_id is everything after domain_
            config_id = prefix[len(domain) + 1:]

            config_preds = predictions.setdefault(config_id, {})
            seed_preds = config_preds.setdefault(seed, {})

            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        convo_id = record['convo_id']
                        convo_preds = seed_preds.setdefault(convo_id, {})
                        for turn in record.get('turns', []):
                            turn_num = turn['turn_num']
                            convo_preds[turn_num] = turn.get('detected_flows', [])
                    except (json.JSONDecodeError, KeyError):
                        continue

        log.info('Loaded predictions for %d configs', len(predictions))
        return predictions

    def compose_ensemble(
        self,
        predictions: dict,
        voter_config_ids: list[str],
        seed: int,
        eval_set: list[dict],
        weights: list[float] | None = None,
    ) -> list[dict]:
        """Combine predictions from multiple 1A configs into ensemble results.

        Args:
            predictions: Output of load_predictions()
            voter_config_ids: Which 1A configs to use as voters
            seed: Which seed's predictions to use
            eval_set: Original eval conversations (for expected labels)
            weights: Optional per-voter weights (defaults to uniform)

        Returns:
            List of conversation results with ensemble predictions
        """
        if weights is None:
            weights = [1.0] * len(voter_config_ids)

        results = []
        for convo in eval_set:
            convo_id = convo['convo_id']
            category = convo.get('category', 'unknown')
            turns_spec = convo.get('turns', [])
            turn_results = []

            for turn in turns_spec:
                if turn.get('speaker') != 'user':
                    continue

                turn_num = turn['turn_num']
                expected = turn['flow']
                candidate_flows = turn.get('candidate_flows')

                # Gather votes from each voter (each is a list of detected_flows)
                voter_flow_lists: list[list[str]] = []
                for vid in voter_config_ids:
                    config_preds = predictions.get(vid, {})
                    seed_preds = config_preds.get(seed, {})
                    convo_preds = seed_preds.get(convo_id, {})
                    detected = convo_preds.get(turn_num, [])
                    voter_flow_lists.append(detected if detected else [])

                ensemble_flows, confidence = self.tally_votes_multi(
                    voter_flow_lists, weights,
                )

                correct = self._score_turn(turn, category, ensemble_flows)

                turn_results.append({
                    'turn_num': turn_num,
                    'utterance': turn['utterance'],
                    'flow': expected,
                    'candidate_flows': candidate_flows,
                    'intent': turn.get('intent'),
                    'detected_flows': ensemble_flows,
                    'correct': correct,
                    'confidence': confidence,
                    'votes': voter_flow_lists,
                    'voter_ids': voter_config_ids,
                })

            results.append({
                'convo_id': convo_id,
                'category': category,
                'turns': turn_results,
            })

        return results

    @staticmethod
    def tally_votes(
        votes: list[str | None],
        weights: list[float] | None = None,
    ) -> tuple[str | None, float]:
        """Weighted majority vote (single-flow, legacy).

        Returns:
            (winning_flow, confidence) where confidence =
            sum(weights for winner) / sum(all weights).
        """
        if weights is None:
            weights = [1.0] * len(votes)

        weighted_counts: Counter[str] = Counter()
        total_weight = 0.0

        for vote, weight in zip(votes, weights):
            if vote is not None:
                weighted_counts[vote] += weight
                total_weight += weight

        if not weighted_counts:
            return None, 0.0

        winner = weighted_counts.most_common(1)[0][0]
        confidence = weighted_counts[winner] / total_weight if total_weight > 0 else 0.0

        return winner, round(confidence, 4)

    @staticmethod
    def tally_votes_multi(
        voter_flow_lists: list[list[str]],
        weights: list[float] | None = None,
    ) -> tuple[list[str], float]:
        """Majority vote on complete answer sets (order-independent).

        Each voter's output is a set of flows. Majority = >50% of weighted
        voters produced the exact same set.

        Examples:
            A / A / B          → A (majority)
            [A,B] / [B,A] / C  → {A,B} (majority, order ignored)
            A / B / C           → A (plurality fallback, no majority)
            [A,B] / A / B       → no majority, plurality fallback

        Returns:
            (ensemble_flows, confidence)
        """
        if weights is None:
            weights = [1.0] * len(voter_flow_lists)

        # Count weighted support per answer-set (frozenset, order-independent)
        set_counts: Counter[frozenset] = Counter()
        total_weight = 0.0
        for flows, w in zip(voter_flow_lists, weights):
            if flows:
                total_weight += w
                set_counts[frozenset(flows)] += w

        if not set_counts:
            return [], 0.0

        # Winner = answer-set with most weighted support
        winner_set, winner_weight = set_counts.most_common(1)[0]
        confidence = winner_weight / total_weight if total_weight > 0 else 0.0

        return sorted(winner_set), round(confidence, 4)

    @staticmethod
    def _score_turn(turn: dict, category: str, detected_flows: list[str]) -> bool:
        """Score using same logic as harness._score_turn."""
        candidate_flows = turn.get('candidate_flows')

        if candidate_flows:
            expected_set = set(candidate_flows)
            if category == 'ambiguous_first':
                return set(detected_flows) == expected_set
            else:
                return expected_set.issubset(set(detected_flows))
        else:
            expected = turn.get('flow', '')
            return len(detected_flows) == 1 and detected_flows[0] == expected
