"""Shared scoring functions for flow detection experiments.

Centralises the scoring logic used by harness.py (exp1a), bootstrap_10v.py
(exp1b), analyze_ambiguity_detection.py, and exp2 pipeline comparison.
All callers use this module as the single source of truth.

Scoring rules (v2 — February 2026):

Clear turns (no candidate_flows):
    Exactly one detected flow matching the gold label → correct, else wrong.

Ambiguous turns (candidate_flows present):
    Two criteria, both required:
    1. Recognise ambiguity: len(detected_flows) >= 2
    2. Predicted label in candidate set:
       Pick one detected flow at random; check membership in candidate set.

    For ensembles the "predicted label" is determined by flattening all
    voters' outputs into individual flow mentions, counting each flow,
    and picking the highest-count flow.  Ties broken by random coin flip.

Exp 2 adds two additional scorers:
    score_tool_turn — Tool-calling (flat and scoped)
    score_nlu_staged_funnel — Pipeline A (staged NLU funnel)

All scores are boolean (True/False).  No partial credit.
"""

from __future__ import annotations

import math
import random
from collections import Counter


# ── Constants ────────────────────────────────────────────────────

FREEBIE_TOOLS_HUGO = frozenset({'read_post', 'read_outline'})
FREEBIE_TOOLS_DANA = frozenset({'describe_stats', 'list_datasets'})


# ── Single-model scoring ─────────────────────────────────────────

def score_turn(
    category: str,
    detected_flows: list[str],
    expected_flow: str,
    candidate_flows: list[str] | None,
) -> bool:
    """Score a single turn.  Returns True/False.

    Works for both exp1a (single model) and clear turns in exp1b.
    For ambiguous turns in exp1b, use ``score_turn_ensemble`` instead.
    """
    if candidate_flows:
        return _score_ambiguous(detected_flows, candidate_flows)
    # Clear turn: exactly one correct flow
    return len(detected_flows) == 1 and detected_flows[0] == expected_flow


def _score_ambiguous(
    detected_flows: list[str],
    candidate_flows: list[str],
) -> bool:
    """Score a single model on an ambiguous turn.

    Criterion 1 — recognise ambiguity:  len(detected_flows) >= 2
    Criterion 2 — pick one detected flow at random, check if it's in
        the candidate set.

    Examples (candidate_flows = [expand, rework]):
        [expand, rework]        → pick one → always in set → True
        [expand, polish]        → pick one → 50/50 coin flip
        [expand]                → fail criterion 1 → False
        [tone, polish, rework]  → pick one → 1/3 chance True
    """
    if len(detected_flows) < 2:
        return False
    predicted = random.choice(detected_flows)
    return predicted in set(candidate_flows)


# ── Ensemble scoring ─────────────────────────────────────────────

def score_turn_ensemble(
    category: str,
    voter_flow_lists: list[list[str]],
    expected_flow: str,
    candidate_flows: list[str] | None,
) -> bool:
    """Score a turn using raw voter outputs (for ensembles).

    For clear turns, delegates to the standard ``score_turn`` after
    tallying votes via set-level majority.

    For ambiguous turns, uses flattened vote counting to determine the
    predicted label (see ``_score_ambiguous_ensemble``).
    """
    if candidate_flows:
        return _score_ambiguous_ensemble(voter_flow_lists, candidate_flows)
    # Clear turn: tally votes (set-level majority) then score normally
    detected, _ = tally_votes_multi(voter_flow_lists)
    return score_turn(category, detected, expected_flow, None)


def _score_ambiguous_ensemble(
    voter_flow_lists: list[list[str]],
    candidate_flows: list[str],
) -> bool:
    """Ensemble scoring for ambiguous turns.

    Criterion 1 — the ensemble recognised ambiguity.  This is true when
        EITHER (a) any voter output 2+ flows, OR (b) voters disagree
        (no single answer-set holds a majority).  A unanimous single-flow
        answer means the ensemble is confident it's not ambiguous → fail.
    Criterion 2 — flatten all voters' outputs into individual flow mentions,
        count each flow, pick the highest count.  Ties broken by coin flip.
        Check if the pick is in the candidate set.

    Examples (candidate_flows = [expand, rework]):
        Votes: [expand,rework], [expand,polish], [expand]
            → criterion 1 pass (voters 1,2 output 2+)
            → flatten: expand=3, rework=1, polish=1
            → predicted = expand (in set) → True
        Votes: [polish], [tone], [rework]
            → criterion 1 pass (no majority — all disagree)
            → flatten: polish=1, tone=1, rework=1 — 3-way tie
            → coin flip among tied → 1/3 chance rework (in set) → maybe True
        Votes: [expand], [expand], [expand]
            → criterion 1 FAIL (unanimous single-flow agreement)
            → False
    """
    active = [vf for vf in voter_flow_lists if vf]
    if not active:
        return False

    # Criterion 1: ensemble recognised ambiguity?
    # (a) Any voter output 2+ flows
    any_multi = any(len(vf) >= 2 for vf in active)
    # (b) No majority — voters disagree on the answer-set
    set_counts: Counter[frozenset[str]] = Counter()
    for vf in active:
        set_counts[frozenset(vf)] += 1
    top_count = set_counts.most_common(1)[0][1]
    no_majority = top_count <= len(active) / 2

    if not any_multi and not no_majority:
        return False

    # Flatten all votes into flow counts
    counts: Counter[str] = Counter()
    for vf in active:
        for flow in vf:
            counts[flow] += 1

    if not counts:
        return False

    # Pick highest-count flow; break ties with coin flip
    max_count = counts.most_common(1)[0][1]
    tied = [f for f, c in counts.items() if c == max_count]
    predicted = random.choice(tied)
    return predicted in set(candidate_flows)


# ── Voting helper (used by ensemble clear-turn scoring) ──────────

def tally_votes_multi(
    voter_flow_lists: list[list[str]],
    weights: list[float] | None = None,
) -> tuple[list[str], float]:
    """Majority vote on complete answer sets (order-independent).

    Each voter's output is a set of flows.  The winner is the set with
    the most weighted support.  Confidence = winner weight / total weight.
    """
    if weights is None:
        weights = [1.0] * len(voter_flow_lists)

    set_counts: Counter[frozenset[str]] = Counter()
    total_weight = 0.0
    for flows, w in zip(voter_flow_lists, weights):
        if flows:
            total_weight += w
            set_counts[frozenset(flows)] += w

    if not set_counts:
        return [], 0.0

    winner_set, winner_weight = set_counts.most_common(1)[0]
    confidence = winner_weight / total_weight if total_weight > 0 else 0.0

    return sorted(winner_set), round(confidence, 4)


# ── Tool manifest helpers ────────────────────────────────────────

def build_tool_flow_map(tools: list[dict]) -> dict[str, list[str]]:
    """Build a mapping from tool name → list of flow names from ``_flows``.

    Reads the ``_flows`` metadata field from each tool definition.
    Wildcard tools (``_flows: ["*"]``) are stored with ``["*"]``.
    """
    mapping: dict[str, list[str]] = {}
    for tool in tools:
        name = tool.get('name', '')
        flows = tool.get('_flows', [])
        mapping[name] = list(flows)
    return mapping


# ── Exp 2 Pipeline A: Intent classification scorer ───────────────

def score_intent(
    detected_intent: str | None,
    expected_intent: str | None,
    candidate_intents: list[str] | None,
) -> bool:
    """Score an intent classification prediction.

    Clear turns (no candidate_intents):
        Exact match on expected_intent.

    Ambiguous turns (candidate_intents present):
        Correct if detected_intent is in candidate_intents.
        The eval data includes 'Plan' in candidate_intents for
        ambiguous turns, so no special-case logic is needed.
    """
    if detected_intent is None:
        return False

    det = detected_intent.strip().lower()

    # Ambiguous turn: check membership in candidate set
    if candidate_intents:
        return det in {c.lower() for c in candidate_intents}

    # Clear turn: exact match
    if not expected_intent:
        return False
    return det == expected_intent.strip().lower()


# ── Exp 2: Tool-calling scorer ───────────────────────────────────

def score_tool_turn(
    predicted_tools: list[str],
    gold_tools: list[str],
    candidate_flows: list[str] | None = None,
    domain: str = 'hugo',
) -> dict:
    """Score a tool-calling turn with strict precision gating.

    Tool calls have side effects (revise, format, etc.), so calling
    spurious action tools is penalised — unlike classification where
    over-prediction is harmless.  Read-only precursor tools (freebies)
    are exempt.

    Precision = |non_freebie_predicted ∩ gold| / |non_freebie_predicted|
        Freebie tools are excluded from the precision denominator.
        Hugo freebies: read_post, read_outline (precursor read tools).
        Dana freebies: describe_stats, list_datasets (info-gathering tools).

    Recall floor = ceil(n/2) for n > 1, else n:
        1 gold → must hit 1, 2 gold → hit 1, 3 gold → hit 2, etc.

    Correct =
        hits ≥ recall floor
        AND (non_freebie_predicted ≤ 2 OR precision ≥ 0.50)

    The precision gate applies when the model calls 3+ non-freebie
    tools.  At that point, at least half must be gold to count as
    correct.  1–2 extra freebie tools are always forgiven.

    Special cases:
        - Null call (no tools) → always incorrect.
        - handle_ambiguity on ambiguous turn → always correct.
        - conversational_response is a regular tool — must be in gold set.
        - All-freebie predictions → precision = 1.0 (vacuously correct),
          but recall must still pass.

    Args:
        predicted_tools: tool names the model called (empty for null call).
        gold_tools: expected tool names (keys of target_tools dict).
        candidate_flows: if present, turn is ambiguous.
        domain: 'hugo' or 'dana' — determines which freebie set to use.

    Returns dict with:
        correct, precision, recall, hits, min_hits, freebies_called,
        ambiguity_flagged, null_call, num_predicted, predicted, gold
    """
    freebie_tools = FREEBIE_TOOLS_HUGO if domain == 'hugo' else FREEBIE_TOOLS_DANA

    pred_set = set(predicted_tools)
    gold_set = set(gold_tools)

    n_gold = len(gold_set)
    min_hits = n_gold if n_gold <= 1 else math.ceil(n_gold / 2)

    result = {
        'correct': False,
        'precision': 0.0,
        'recall': 0.0,
        'hits': 0,
        'min_hits': min_hits,
        'freebies_called': sorted(pred_set & freebie_tools),
        'ambiguity_flagged': 'handle_ambiguity' in pred_set,
        'null_call': len(pred_set) == 0,
        'num_predicted': len(pred_set),
        'predicted': sorted(pred_set),
        'gold': sorted(gold_set),
    }

    # No gold tools → unmapped flow; correct if model also didn't call anything
    if not gold_set:
        result['correct'] = not pred_set
        result['recall'] = 1.0 if not pred_set else 0.0
        result['precision'] = 1.0 if not pred_set else 0.0
        return result

    # Null calls with gold tools → always incorrect
    if not pred_set:
        return result

    # handle_ambiguity on ambiguous turn → always correct
    if result['ambiguity_flagged'] and candidate_flows:
        result['correct'] = True
        result['precision'] = 1.0
        result['recall'] = 1.0
        result['hits'] = min_hits  # credit full recall
        return result

    # Compute hits
    hits_in_gold = pred_set & gold_set
    result['hits'] = len(hits_in_gold)

    # Precision: only non-freebie predictions count in denominator
    non_freebie_pred = pred_set - freebie_tools
    non_freebie_hits = non_freebie_pred & gold_set

    if non_freebie_pred:
        result['precision'] = len(non_freebie_hits) / len(non_freebie_pred)
    else:
        # All predictions are freebies → vacuously correct (no "mistakes")
        result['precision'] = 1.0

    # Recall: hits vs floor
    result['recall'] = len(hits_in_gold) / n_gold if n_gold > 0 else 1.0
    recall_passes = len(hits_in_gold) >= min_hits

    # Precision gate: applied for 3+ non-freebie predictions
    # Tool calls have side effects — spurious actions are penalised.
    # 1-2 non-freebie tools → precision forgiven (minor over-prediction)
    # 3+ non-freebie tools → at least 50% must be gold
    if len(non_freebie_pred) >= 3:
        precision_passes = result['precision'] >= 0.50
    else:
        precision_passes = True

    result['correct'] = precision_passes and recall_passes

    return result


# ── Exp 2: Tool parameter scoring ─────────────────────────────

def score_tool_params(
    predicted_tools_with_args: list[dict],
    gold_target_tools: dict,
    fuzzy_evaluator=None,
    param_schema_index: dict | None = None,
) -> dict:
    """Score predicted tool parameters against gold target_tools.

    Match method is determined from the tool manifest schema via
    param_schema_index (built by build_param_schema_index). When no
    schema is available, defaults to fuzzy matching for strings.

    Args:
        predicted_tools_with_args: [{'name': ..., 'args': {...}}, ...]
        gold_target_tools: {tool_name: {param: value|null}}
        fuzzy_evaluator: callable(gold, predicted) -> bool (for NL params)
        param_schema_index: {(tool_name, param_name): property_schema}

    Returns dict with param_accuracy, matched/total counts, and per-param details.
    """
    from helpers.schema_utils import classify_match_method

    # Build lookup: first occurrence of each tool name wins
    pred_lookup: dict[str, dict] = {}
    for t in predicted_tools_with_args:
        name = t.get('name', '')
        if name and name not in pred_lookup:
            pred_lookup[name] = t.get('args') or {}

    details = []
    matched = 0
    total = 0
    fuzzy_matched = 0
    fuzzy_total = 0
    exact_matched = 0
    exact_total = 0

    for tool_name, gold_params in gold_target_tools.items():
        if not isinstance(gold_params, dict):
            continue
        if tool_name not in pred_lookup:
            # Tool not predicted — all non-null params are misses
            for param_name, gold_val in gold_params.items():
                if gold_val is None:
                    continue
                total += 1
                exact_total += 1
                details.append({
                    'tool': tool_name, 'param': param_name,
                    'gold': gold_val, 'predicted': None,
                    'match': False, 'method': 'missing_tool',
                })
            continue

        pred_args = pred_lookup[tool_name]

        for param_name, gold_val in gold_params.items():
            if gold_val is None:
                continue

            # Schema-driven match method dispatch
            param_schema = (param_schema_index or {}).get((tool_name, param_name), {})
            method = classify_match_method(param_schema)

            pred_val = pred_args.get(param_name)

            if pred_val is None:
                total += 1
                if method == 'fuzzy':
                    fuzzy_total += 1
                else:
                    exact_total += 1
                details.append({
                    'tool': tool_name, 'param': param_name,
                    'gold': gold_val, 'predicted': None,
                    'match': False, 'method': 'missing',
                })
                continue

            total += 1

            if method == 'fuzzy':
                fuzzy_total += 1
                if fuzzy_evaluator:
                    match = fuzzy_evaluator(gold_val, pred_val)
                else:
                    # Fallback: case-insensitive string comparison
                    match = str(gold_val).strip().lower() == str(pred_val).strip().lower()
                if match:
                    fuzzy_matched += 1
            elif method == 'structured':
                match = _match_structured(gold_val, pred_val)
                exact_total += 1
                if match:
                    exact_matched += 1
            else:
                # Exact match (enums, booleans, numbers)
                match = _normalize_for_match(gold_val) == _normalize_for_match(pred_val)
                exact_total += 1
                if match:
                    exact_matched += 1

            if match:
                matched += 1

            details.append({
                'tool': tool_name, 'param': param_name,
                'gold': gold_val, 'predicted': pred_val,
                'match': match, 'method': method,
            })

    return {
        'param_accuracy': matched / total if total > 0 else 1.0,
        'matched_params': matched,
        'total_scored_params': total,
        'fuzzy_matched': fuzzy_matched,
        'fuzzy_total': fuzzy_total,
        'exact_matched': exact_matched,
        'exact_total': exact_total,
        'param_details': details,
    }


def _normalize_for_match(val) -> str:
    """Normalize a value for exact comparison."""
    if val is None:
        return ''
    return str(val).strip().lower()


def _match_structured(gold: dict, predicted) -> bool:
    """Match a structured gold object against a predicted value.

    The predicted value may be a dict (ideal) or a string (model didn't
    follow structured format).  For dicts, all gold keys must match.
    """
    if not isinstance(predicted, dict):
        return False
    for k, v in gold.items():
        pred_v = predicted.get(k)
        if pred_v is None:
            return False
        if str(v).strip().lower() != str(pred_v).strip().lower():
            return False
    return True


def build_fuzzy_evaluator(client) -> callable:
    """Build a fuzzy evaluator using Haiku for NL param comparison."""
    def evaluate(gold_value, predicted_value) -> bool:
        if gold_value == predicted_value:
            return True
        # Quick normalization check before calling LLM
        if str(gold_value).strip().lower() == str(predicted_value).strip().lower():
            return True
        prompt = (
            'Are these two tool parameter values semantically equivalent? '
            'They do not need to be word-for-word identical, but must convey '
            'the same intent/instruction.\n'
            f'Expected: {gold_value}\n'
            f'Predicted: {predicted_value}\n'
            'Reply YES or NO only.'
        )
        resp = client.call_simple(model='claude-haiku-4-5-20251001', prompt=prompt)
        return resp.strip().upper().startswith('YES')
    return evaluate


# ── Exp 2 Pipeline A: Staged NLU funnel scorer ──────────────────

def score_nlu_staged_funnel(
    detected_intent: str | None,
    detected_flow: str | None,
    detected_slots: dict | None,
    expected_intent: str,
    expected_flow: str,
    expected_slots: dict | None,
    flow_slot_schema: dict | None,
) -> dict:
    """Score a single turn for Pipeline A (staged NLU funnel).

    Returns a dict with:
        intent_correct    — bool
        flow_correct      — bool
        slots_score       — dict with slot_precision, slot_recall, slot_f1
        end_to_end_correct — bool (all stages correct)
        details           — per-stage detail dict
    """
    # Intent stage
    intent_correct = (
        detected_intent is not None
        and detected_intent.lower() == expected_intent.lower()
    )

    # Flow stage
    flow_correct = (
        detected_flow is not None
        and detected_flow.lower() == expected_flow.lower()
    )

    # Slot stage (structural scoring)
    slots_score = _score_slots_structural(
        detected_slots or {},
        expected_slots or {},
        flow_slot_schema or {},
    )

    # End-to-end: all stages correct
    slot_threshold = 0.5
    end_to_end = (
        intent_correct
        and flow_correct
        and slots_score['slot_f1'] >= slot_threshold
    )

    return {
        'intent_correct': intent_correct,
        'flow_correct': flow_correct,
        'slots_score': slots_score,
        'end_to_end_correct': end_to_end,
        'details': {
            'detected_intent': detected_intent,
            'expected_intent': expected_intent,
            'detected_flow': detected_flow,
            'expected_flow': expected_flow,
            'detected_slots': detected_slots,
            'expected_slots': expected_slots,
        },
    }


def _score_slots_structural(
    detected: dict,
    expected: dict,
    schema: dict,
) -> dict:
    """Structural slot scoring — checks keys, not values.

    - Correct keys: detected keys that appear in the schema
    - Required slots filled: schema keys with priority='required' that have non-null values
    - No hallucinated keys: detected keys not in schema

    Returns: slot_precision, slot_recall, slot_f1, details
    """
    schema_keys = set(schema.keys())

    # Filter detected to non-null entries
    detected_present = {k for k, v in detected.items() if v is not None}

    if not schema_keys:
        # No schema → can't evaluate structurally
        return {
            'slot_precision': 1.0 if not detected_present else 0.0,
            'slot_recall': 1.0,
            'slot_f1': 1.0 if not detected_present else 0.0,
            'valid_keys': [],
            'hallucinated_keys': list(detected_present),
            'missing_required': [],
        }

    # Valid keys = detected keys that are in the schema
    valid_keys = detected_present & schema_keys
    hallucinated = detected_present - schema_keys

    # Required slots from schema
    required_keys = {
        k for k, v in schema.items()
        if isinstance(v, dict) and v.get('priority') == 'required'
    }
    required_filled = required_keys & detected_present
    missing_required = required_keys - detected_present

    # Precision: fraction of detected keys that are valid
    precision = len(valid_keys) / len(detected_present) if detected_present else 1.0
    # Recall: fraction of schema keys that were detected
    recall = len(valid_keys) / len(schema_keys) if schema_keys else 1.0
    # F1
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        'slot_precision': round(precision, 4),
        'slot_recall': round(recall, 4),
        'slot_f1': round(f1, 4),
        'valid_keys': sorted(valid_keys),
        'hallucinated_keys': sorted(hallucinated),
        'missing_required': sorted(missing_required),
    }
