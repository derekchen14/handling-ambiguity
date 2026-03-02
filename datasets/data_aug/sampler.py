"""Flow + scenario sampling per category and DAX decomposition."""

from __future__ import annotations

import random
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ── Scenario Banks ────────────────────────────────────────────────────

SCENARIOS = {
    'hugo': [
        'Technical ML/AI blog -- writing about transformer architectures',
        'Travel blog -- series on Southeast Asia backpacking',
        'Personal finance -- investment strategies for beginners',
        'Company engineering blog -- migration from monolith to microservices',
        'Food/cooking blog -- seasonal recipes with local ingredients',
        'Product review site -- reviewing developer tools and IDEs',
        'Thought leadership -- future of remote work in tech',
        'Tutorial/how-to -- step-by-step guide to building a REST API',
        'Newsletter (Substack) -- weekly AI research roundup',
        'Creative writing / essays -- personal essays on urban living',
    ],
    'dana': [
        'E-commerce sales -- Q4 sales data with 50K orders across 6 regions',
        'HR/people analytics -- employee attrition dataset with 2,000 records',
        'Marketing campaigns -- email campaign performance across 12 segments',
        'Financial reporting -- monthly P&L across 4 business units',
        'Healthcare -- patient outcomes dataset from 3 hospital sites',
        'Customer churn -- SaaS subscription data with usage metrics',
        'Supply chain -- inventory and shipping data across 8 warehouses',
        'Social media -- engagement metrics for Instagram and TikTok',
        'Product analytics -- feature usage telemetry from 100K users',
        'Survey/research -- customer satisfaction survey with 5,000 responses',
    ],
}

PERSONA_HINTS = [
    'The user is terse and uses short sentences.',
    'The user is verbose and provides lots of context.',
    'The user is technical and uses precise terminology.',
    'The user is casual and uses informal language.',
    'The user is a beginner and may use imprecise terms.',
]

# Probability of adding a persona hint to a prompt
PERSONA_HINT_PROB = 0.3


# ── Helpers ───────────────────────────────────────────────────────────

def _load_domain(domain: str):
    """Import FLOW_CATALOG, DACT_CATALOG, and Intent for a domain."""
    if domain == 'hugo':
        from datasets.hugo.ontology import (
            DACT_CATALOG, FLOW_CATALOG, Intent,
        )
    elif domain == 'dana':
        from datasets.dana.ontology import (
            DACT_CATALOG, FLOW_CATALOG, Intent,
        )
    else:
        raise ValueError(f'Unknown domain: {domain}')
    return FLOW_CATALOG, DACT_CATALOG, Intent


def _user_facing_flows(flow_catalog: dict, Intent) -> dict:
    """Return only user-facing flows (exclude Plan and Internal)."""
    result = {}
    for name, flow in flow_catalog.items():
        intent_val = flow['intent'].value if hasattr(flow['intent'], 'value') else str(flow['intent'])
        if intent_val not in ('Plan', 'Internal'):
            result[name] = flow
    return result


def _flows_by_intent(flows: dict) -> dict[str, list[str]]:
    """Group flow names by intent value."""
    groups: dict[str, list[str]] = {}
    for name, flow in flows.items():
        intent_val = flow['intent'].value if hasattr(flow['intent'], 'value') else str(flow['intent'])
        groups.setdefault(intent_val, []).append(name)
    return groups


def _plan_orchestrator(domain: str) -> tuple[str, str]:
    """Return (flow_name, intent) for the domain's Plan orchestrator."""
    if domain == 'hugo':
        return 'blueprint', 'Plan'
    elif domain == 'dana':
        return 'outline', 'Plan'
    raise ValueError(f'Unknown domain: {domain}')


# ── DAX Decomposition ────────────────────────────────────────────────

def decompose_dax(dax_code: str, dact_catalog: dict) -> str:
    """Decompose a DAX code into its dact primitives.

    Example: '{03A}' with Hugo catalog → 'chat(0) + compose(3) + post(A)'
    """
    # Build reverse lookup: hex digit → dact name
    hex_to_name = {}
    for dact_name, info in dact_catalog.items():
        hex_to_name[info['hex'].upper()] = dact_name

    # Strip braces
    code = dax_code.strip('{}')

    parts = []
    for digit in code:
        name = hex_to_name.get(digit.upper(), f'?{digit}')
        parts.append(f'{name}({digit.upper()})')

    return ' + '.join(parts)


def _flow_info(name: str, flow: dict, dact_catalog: dict) -> dict:
    """Build a flow info dict with decomposed DAX."""
    intent_val = flow['intent'].value if hasattr(flow['intent'], 'value') else str(flow['intent'])
    return {
        'name': name,
        'intent': intent_val,
        'description': flow.get('description', ''),
        'dax': flow.get('dax', ''),
        'dax_decomposed': decompose_dax(flow.get('dax', '{}'), dact_catalog),
        'edge_flows': flow.get('edge_flows', []),
    }


# ── Scenario Sampling ────────────────────────────────────────────────

def sample_scenarios(domain: str, n: int = 32) -> list[str]:
    """Round-robin from scenario bank."""
    bank = SCENARIOS[domain]
    return [bank[i % len(bank)] for i in range(n)]


def sample_persona_hint(rng: random.Random) -> str | None:
    """Return a persona hint with PERSONA_HINT_PROB probability."""
    if rng.random() < PERSONA_HINT_PROB:
        return rng.choice(PERSONA_HINTS)
    return None


# ── Category A — Same Flow (32 convos) ───────────────────────────────

def sample_category_a(
    flow_catalog: dict,
    dact_catalog: dict,
    Intent,
    domain: str,
    n: int = 32,
    seed: int = 42,
) -> list[dict]:
    """Sample 32 flows for same-flow conversations.

    Near-uniform coverage: each flow appears ~1x. Every intent has at least
    4 representatives. Top-frequency flows can appear 2x if needed.
    """
    rng = random.Random(seed)
    flows = _user_facing_flows(flow_catalog, Intent)
    by_intent = _flows_by_intent(flows)

    # Ensure every intent has at least 4 representatives
    selected = []
    for intent_name, flow_names in sorted(by_intent.items()):
        picks = rng.sample(flow_names, min(4, len(flow_names)))
        for fname in picks:
            selected.append(fname)

    # Fill remaining slots with random flows (allowing duplicates for top flows)
    remaining = n - len(selected)
    all_flow_names = list(flows.keys())
    # Flows not yet selected get priority
    unselected = [f for f in all_flow_names if f not in selected]
    rng.shuffle(unselected)

    if remaining > 0:
        # First use unselected flows
        fill = unselected[:remaining]
        remaining -= len(fill)
        selected.extend(fill)

    if remaining > 0:
        # Allow duplicates from the full pool
        extra = rng.choices(all_flow_names, k=remaining)
        selected.extend(extra)

    # Trim to exactly n
    selected = selected[:n]
    rng.shuffle(selected)

    scenarios = sample_scenarios(domain, n)

    return [
        {
            'convo_idx': i,
            'category': 'same_flow',
            'flow': _flow_info(fname, flows[fname], dact_catalog),
            'scenario': scenarios[i],
        }
        for i, fname in enumerate(selected)
    ]


# ── Category B — Switch Flow (32 convos) ─────────────────────────────

def sample_category_b(
    flow_catalog: dict,
    dact_catalog: dict,
    Intent,
    domain: str,
    n: int = 32,
    seed: int = 42,
) -> list[dict]:
    """Sample 32 (flow_x, flow_y) pairs for switch-flow conversations.

    Mix: ~n/2 same-intent pairs + ~n/2 cross-intent pairs.
    Maximizes flow coverage across both slots.
    """
    rng = random.Random(seed)
    flows = _user_facing_flows(flow_catalog, Intent)
    by_intent = _flows_by_intent(flows)

    same_intent_pairs = []
    cross_intent_pairs = []

    # Generate same-intent pairs
    for intent_name, flow_names in sorted(by_intent.items()):
        if len(flow_names) < 2:
            continue
        shuffled = flow_names[:]
        rng.shuffle(shuffled)
        for i in range(0, len(shuffled) - 1, 2):
            same_intent_pairs.append((shuffled[i], shuffled[i + 1]))

    # Generate cross-intent pairs
    intent_names = sorted(by_intent.keys())
    for i in range(len(intent_names)):
        for j in range(i + 1, len(intent_names)):
            f1 = rng.choice(by_intent[intent_names[i]])
            f2 = rng.choice(by_intent[intent_names[j]])
            cross_intent_pairs.append((f1, f2))

    rng.shuffle(same_intent_pairs)
    rng.shuffle(cross_intent_pairs)

    # Take ~n/2 from each
    half = n // 2
    selected_same = same_intent_pairs[:half]
    selected_cross = cross_intent_pairs[:n - half]

    # If we don't have enough, fill from the other
    if len(selected_same) < half:
        deficit = half - len(selected_same)
        selected_cross = cross_intent_pairs[:n - len(selected_same)]
    if len(selected_cross) < n - half:
        deficit = (n - half) - len(selected_cross)
        selected_same = same_intent_pairs[:half + deficit]

    pairs = selected_same + selected_cross
    # Ensure exactly n pairs, fill with random if needed
    while len(pairs) < n:
        all_names = list(flows.keys())
        f1, f2 = rng.sample(all_names, 2)
        pairs.append((f1, f2))
    pairs = pairs[:n]
    rng.shuffle(pairs)

    scenarios = sample_scenarios(domain, n)

    return [
        {
            'convo_idx': i,
            'category': 'switch_flow',
            'flow_x': _flow_info(fx, flows[fx], dact_catalog),
            'flow_y': _flow_info(fy, flows[fy], dact_catalog),
            'scenario': scenarios[i],
        }
        for i, (fx, fy) in enumerate(pairs)
    ]


# ── Category C — Ambiguous First Turn (32 convos) ────────────────────

def _find_confusable_pairs(flows: dict) -> list[tuple[str, str]]:
    """Find pairs of flows that are confusable via edge_flows."""
    pairs = set()
    for name, flow in flows.items():
        for edge in flow.get('edge_flows', []):
            if edge in flows:
                pair = tuple(sorted([name, edge]))
                pairs.add(pair)
    return list(pairs)


def sample_category_c(
    flow_catalog: dict,
    dact_catalog: dict,
    Intent,
    domain: str,
    n: int = 32,
    seed: int = 42,
) -> list[dict]:
    """Sample 32 confusable pairs using edge_flows.

    Turn 1: genuinely ambiguous between the pair.
    Turn 3: user resolves to one specific flow.
    """
    rng = random.Random(seed)
    flows = _user_facing_flows(flow_catalog, Intent)
    confusable = _find_confusable_pairs(flows)
    rng.shuffle(confusable)

    selected = []
    while len(selected) < n:
        for pair in confusable:
            if len(selected) >= n:
                break
            # Randomly choose which flow resolves
            fa, fb = pair
            if rng.random() < 0.5:
                fa, fb = fb, fa
            selected.append((fa, fb))

    selected = selected[:n]

    scenarios = sample_scenarios(domain, n)

    return [
        {
            'convo_idx': i,
            'category': 'ambiguous_first',
            'flow_a': _flow_info(fa, flows[fa], dact_catalog),
            'flow_b': _flow_info(fb, flows[fb], dact_catalog),
            'resolves_to': fa,  # Turn 3 resolves to flow_a
            'scenario': scenarios[i],
        }
        for i, (fa, fb) in enumerate(selected)
    ]


# ── Category D — Ambiguous Second Turn (32 convos) ───────────────────

def sample_category_d(
    flow_catalog: dict,
    dact_catalog: dict,
    Intent,
    domain: str,
    n: int = 32,
    seed: int = 42,
) -> list[dict]:
    """Sample 32 (clear_flow, pair_of_flows) for multi-request.

    Turn 1: clear single flow.
    Turn 3: user asks for TWO things — routes to Plan orchestrator.
    """
    rng = random.Random(seed)
    flows = _user_facing_flows(flow_catalog, Intent)
    by_intent = _flows_by_intent(flows)
    all_flow_names = list(flows.keys())
    intent_names = sorted(by_intent.keys())

    orchestrator_name, orchestrator_intent = _plan_orchestrator(domain)

    selected = []
    for _ in range(n):
        # Turn 1: any flow
        flow_x = rng.choice(all_flow_names)

        # Turn 3: two flows from preferably different intents
        intent_y = rng.choice(intent_names)
        remaining_intents = [i for i in intent_names if i != intent_y]
        intent_z = rng.choice(remaining_intents) if remaining_intents else intent_y

        flow_y = rng.choice(by_intent[intent_y])
        flow_z = rng.choice(by_intent[intent_z])

        # Ensure flow_y != flow_z
        attempts = 0
        while flow_y == flow_z and attempts < 20:
            flow_z = rng.choice(all_flow_names)
            attempts += 1

        selected.append((flow_x, flow_y, flow_z))

    scenarios = sample_scenarios(domain, n)

    result = []
    for i, (fx, fy, fz) in enumerate(selected):
        result.append({
            'convo_idx': i,
            'category': 'ambiguous_second',
            'flow_x': _flow_info(fx, flows[fx], dact_catalog),
            'flow_y': _flow_info(fy, flows[fy], dact_catalog),
            'flow_z': _flow_info(fz, flows[fz], dact_catalog),
            'orchestrator': {'name': orchestrator_name, 'intent': orchestrator_intent},
            'scenario': scenarios[i],
        })
    return result


# ── Master Sampler ────────────────────────────────────────────────────

def sample_all(domain: str, n_per_cat: int = 32, seed: int = 42) -> dict[str, list[dict]]:
    """Sample all 4 categories for a domain.

    Returns:
        Dict with keys 'a', 'b', 'c', 'd', each a list of sample dicts.
    """
    flow_catalog, dact_catalog, Intent = _load_domain(domain)

    return {
        'a': sample_category_a(flow_catalog, dact_catalog, Intent, domain, n_per_cat, seed),
        'b': sample_category_b(flow_catalog, dact_catalog, Intent, domain, n_per_cat, seed),
        'c': sample_category_c(flow_catalog, dact_catalog, Intent, domain, n_per_cat, seed),
        'd': sample_category_d(flow_catalog, dact_catalog, Intent, domain, n_per_cat, seed),
    }
