"""Pure metric functions — accuracy, calibration, agreement."""

from __future__ import annotations

from collections import Counter

import numpy as np


# ── Accuracy metrics ──────────────────────────────────────────────

def accuracy_top1(results: list[dict]) -> float:
    """Overall accuracy across all labeled turns."""
    if not results:
        return 0.0
    correct = sum(1 for r in results if r.get('correct'))
    return correct / len(results)


def accuracy_per_intent(
    results: list[dict], ontology: dict,
) -> dict[str, float]:
    """Accuracy broken down by intent (uses ontology to map flow→intent)."""
    flow_to_intent = {
        name: flow['intent'].value if hasattr(flow['intent'], 'value') else str(flow['intent'])
        for name, flow in ontology.items()
    }

    by_intent: dict[str, list[bool]] = {}
    for r in results:
        intent = flow_to_intent.get(r.get('flow', ''), 'Unknown')
        by_intent.setdefault(intent, []).append(bool(r.get('correct')))

    return {
        intent: sum(vals) / len(vals)
        for intent, vals in by_intent.items()
        if vals
    }


def accuracy_per_category(results: list[dict]) -> dict[str, float]:
    """Accuracy broken down by conversation category."""
    by_cat: dict[str, list[bool]] = {}
    for r in results:
        cat = r.get('category', 'unknown')
        by_cat.setdefault(cat, []).append(bool(r.get('correct')))

    return {
        cat: sum(vals) / len(vals)
        for cat, vals in by_cat.items()
        if vals
    }


def accuracy_per_turn(results: list[dict]) -> dict[int, float]:
    """Accuracy broken down by turn number."""
    by_turn: dict[int, list[bool]] = {}
    for r in results:
        turn = r.get('turn_num', 1)
        by_turn.setdefault(turn, []).append(bool(r.get('correct')))

    return {
        turn: sum(vals) / len(vals)
        for turn, vals in by_turn.items()
        if vals
    }


def confusion_matrix(
    results: list[dict], flow_names: list[str],
) -> np.ndarray:
    """Build confusion matrix: rows=expected, cols=detected."""
    idx = {name: i for i, name in enumerate(flow_names)}
    n = len(flow_names)
    mat = np.zeros((n, n), dtype=int)

    for r in results:
        expected = r.get('flow', '')
        detected = r.get('detected_flow', '')
        if expected in idx and detected in idx:
            mat[idx[expected], idx[detected]] += 1

    return mat


def near_miss_rate(results: list[dict], ontology: dict) -> float:
    """Rate of errors where detected flow is an edge_flow of the expected flow."""
    errors = [r for r in results if not r.get('correct')]
    if not errors:
        return 0.0

    near_misses = 0
    for r in errors:
        expected = r.get('flow', '')
        detected = r.get('detected_flow', '')
        flow_info = ontology.get(expected, {})
        if detected in flow_info.get('edge_flows', []):
            near_misses += 1

    return near_misses / len(errors)


# ── Calibration metrics ──────────────────────────────────────────

def ece(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10,
) -> float:
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(confidences)
    if total == 0:
        return 0.0

    ece_val = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        avg_conf = confidences[mask].mean()
        avg_acc = correct[mask].mean()
        ece_val += (n_bin / total) * abs(avg_acc - avg_conf)

    return float(ece_val)


def mce(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10,
) -> float:
    """Maximum Calibration Error."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    max_gap = 0.0

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        avg_conf = confidences[mask].mean()
        avg_acc = correct[mask].mean()
        max_gap = max(max_gap, abs(avg_acc - avg_conf))

    return float(max_gap)


def brier_score(
    confidences: np.ndarray, correct: np.ndarray,
) -> float:
    """Brier score: mean squared error between confidence and correctness."""
    if len(confidences) == 0:
        return 0.0
    return float(np.mean((confidences - correct.astype(float)) ** 2))


def reliability_diagram(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (mean_predicted_confidence, fraction_correct, bin_counts) per bin."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confs = []
    bin_accs = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        n_bin = mask.sum()
        bin_counts.append(int(n_bin))
        if n_bin == 0:
            bin_confs.append(np.nan)
            bin_accs.append(np.nan)
        else:
            bin_confs.append(confidences[mask].mean())
            bin_accs.append(correct[mask].mean())

    return np.array(bin_confs), np.array(bin_accs), np.array(bin_counts)


def overconfidence_rate(
    confidences: np.ndarray, correct: np.ndarray, threshold: float = 0.64,
) -> float:
    """Fraction of incorrect predictions where confidence >= threshold."""
    incorrect = ~correct.astype(bool)
    if incorrect.sum() == 0:
        return 0.0
    return float((confidences[incorrect] >= threshold).mean())


def underconfidence_rate(
    confidences: np.ndarray, correct: np.ndarray, threshold: float = 0.64,
) -> float:
    """Fraction of correct predictions where confidence < threshold."""
    correct_mask = correct.astype(bool)
    if correct_mask.sum() == 0:
        return 0.0
    return float((confidences[correct_mask] < threshold).mean())


# ── Agreement metrics ────────────────────────────────────────────

def fleiss_kappa(ratings: np.ndarray) -> float:
    """Fleiss' kappa for inter-rater agreement.

    Args:
        ratings: (n_items, n_categories) matrix where each cell is the
                 count of raters who assigned that category to that item.
    """
    n_items, n_categories = ratings.shape
    n_raters = ratings[0].sum()

    if n_items == 0 or n_raters <= 1:
        return 0.0

    # Proportion of all assignments to each category
    p_j = ratings.sum(axis=0) / (n_items * n_raters)

    # P_i: extent of agreement for each item
    P_i = (np.sum(ratings ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))

    P_bar = P_i.mean()
    P_e = np.sum(p_j ** 2)

    if abs(1.0 - P_e) < 1e-10:
        return 1.0 if abs(P_bar - 1.0) < 1e-10 else 0.0

    return float((P_bar - P_e) / (1.0 - P_e))
