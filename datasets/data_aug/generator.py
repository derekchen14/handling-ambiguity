"""LLM orchestration for synthetic data augmentation of multi-turn conversations."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.data_aug.prompts import (
    build_system_prompt,
    build_user_prompt,
)
from experiments.data_aug.sampler import (
    _load_domain,
    sample_all,
    sample_persona_hint,
)

log = logging.getLogger(__name__)

# All categories use Opus. Sonnet was tested and found insufficient for
# generating natural, non-formulaic conversations.
MODEL = 'claude-opus-4-6'

TEMPERATURE = 0.8
MAX_TOKENS = 1024


# ── Label Attachment ──────────────────────────────────────────────────

def _attach_labels_a(sample: dict, turns: list[dict], convo_id: str, domain: str) -> dict:
    """Attach labels for Category A (same flow)."""
    flow = sample['flow']
    return {
        'convo_id': convo_id,
        'category': 'same_flow',
        'scenario': _sanitize_text(sample['scenario']),
        'turns': [
            {
                'turn_num': 1,
                'flow': flow['name'],
                'intent': flow['intent'],
                'speaker': 'user',
                'utterance': turns[0]['utterance'],
            },
            {
                'turn_num': 2,
                'speaker': 'agent',
                'utterance': turns[1]['utterance'],
            },
            {
                'turn_num': 3,
                'flow': flow['name'],
                'intent': flow['intent'],
                'speaker': 'user',
                'utterance': turns[2]['utterance'],
            },
        ],
    }


def _attach_labels_b(sample: dict, turns: list[dict], convo_id: str, domain: str) -> dict:
    """Attach labels for Category B (switch flow)."""
    flow_x = sample['flow_x']
    flow_y = sample['flow_y']
    return {
        'convo_id': convo_id,
        'category': 'switch_flow',
        'scenario': _sanitize_text(sample['scenario']),
        'turns': [
            {
                'turn_num': 1,
                'flow': flow_x['name'],
                'intent': flow_x['intent'],
                'speaker': 'user',
                'utterance': turns[0]['utterance'],
            },
            {
                'turn_num': 2,
                'speaker': 'agent',
                'utterance': turns[1]['utterance'],
            },
            {
                'turn_num': 3,
                'flow': flow_y['name'],
                'intent': flow_y['intent'],
                'speaker': 'user',
                'utterance': turns[2]['utterance'],
            },
        ],
    }


def _attach_labels_c(sample: dict, turns: list[dict], convo_id: str, domain: str) -> dict:
    """Attach labels for Category C (ambiguous first, clarified third)."""
    flow_a = sample['flow_a']
    flow_b = sample['flow_b']
    resolves_to = sample['resolves_to']
    resolved_flow = flow_a if resolves_to == flow_a['name'] else flow_b
    return {
        'convo_id': convo_id,
        'category': 'ambiguous_first',
        'scenario': _sanitize_text(sample['scenario']),
        'turns': [
            {
                'turn_num': 1,
                'flow': 'ambiguous',
                'candidate_flows': [flow_a['name'], flow_b['name']],
                'candidate_intents': [flow_a['intent'], flow_b['intent']],
                'speaker': 'user',
                'utterance': turns[0]['utterance'],
            },
            {
                'turn_num': 2,
                'speaker': 'agent',
                'utterance': turns[1]['utterance'],
            },
            {
                'turn_num': 3,
                'flow': resolved_flow['name'],
                'intent': resolved_flow['intent'],
                'speaker': 'user',
                'utterance': turns[2]['utterance'],
            },
        ],
    }


def _generate_rationale_d(flow_y: dict, flow_z: dict) -> str:
    """Generate a template rationale for Category D multi-request."""
    fy = f"'{flow_y['name']}', {flow_y['intent']}"
    fz = f"'{flow_z['name']}', {flow_z['intent']}"
    if flow_y['intent'] == flow_z['intent']:
        return (
            f"Two distinct {flow_y['intent']} operations ({fy}) and ({fz}) "
            f"require separate flow stacks despite sharing an intent."
        )
    return f"({fy}) and ({fz}) are operations from different intents."


def _attach_labels_d(sample: dict, turns: list[dict], convo_id: str, domain: str) -> dict:
    """Attach labels for Category D (clear first, multi-request third)."""
    flow_x = sample['flow_x']
    flow_y = sample['flow_y']
    flow_z = sample['flow_z']
    orch = sample['orchestrator']
    return {
        'convo_id': convo_id,
        'category': 'ambiguous_second',
        'scenario': _sanitize_text(sample['scenario']),
        'turns': [
            {
                'turn_num': 1,
                'flow': flow_x['name'],
                'intent': flow_x['intent'],
                'speaker': 'user',
                'utterance': turns[0]['utterance'],
            },
            {
                'turn_num': 2,
                'speaker': 'agent',
                'utterance': turns[1]['utterance'],
            },
            {
                'turn_num': 3,
                'flow': orch['name'],
                'intent': orch['intent'],
                'candidate_flows': [flow_y['name'], flow_z['name']],
                'candidate_intents': [flow_y['intent'], flow_z['intent']],
                'speaker': 'user',
                'utterance': turns[2]['utterance'],
                'rationale': _generate_rationale_d(flow_y, flow_z),
            },
        ],
    }


LABEL_ATTACHERS = {
    'a': _attach_labels_a,
    'b': _attach_labels_b,
    'c': _attach_labels_c,
    'd': _attach_labels_d,
}


# ── LLM Call ──────────────────────────────────────────────────────────

def _call_anthropic(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> dict:
    """Call Anthropic API and return parsed turns."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

    t0 = time.perf_counter()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{'role': 'user', 'content': user_prompt}],
    )
    latency_ms = round((time.perf_counter() - t0) * 1000)

    raw_text = response.content[0].text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    # Parse JSON from response
    turns = _parse_turns(raw_text)

    return {
        'turns': turns,
        'raw_response': raw_text,
        'latency_ms': latency_ms,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'model': model,
    }


# Unicode characters that LLMs like to produce but we want plain ASCII.
_UNICODE_REPLACEMENTS = {
    '\u2013': ' - ',    # en dash
    '\u2018': "'",      # left single quote
    '\u2019': "'",      # right single quote
    '\u201c': '"',      # left double quote
    '\u201d': '"',      # right double quote
    '\u2026': '...',    # ellipsis
    '\u00a0': ' ',      # non-breaking space
}

# Varied replacements for em dashes to reduce LLM tells.
_EM_DASH_REPLACEMENTS = [', ', '. ', '; ', ' - ', ' -- ']
_EM_DASH_WEIGHTS = [0.35, 0.20, 0.15, 0.15, 0.15]

# Module-level RNG for em-dash variation (seeded per-run via generate_one).
_sanitize_rng = random.Random(42)


def _sanitize_text(text: str) -> str:
    """Replace fancy unicode and vary em-dash substitutions."""
    # First pass: replace non-em-dash unicode
    for old, new in _UNICODE_REPLACEMENTS.items():
        text = text.replace(old, new)

    # Second pass: replace em dashes with varied punctuation
    if '\u2014' in text:
        parts = text.split('\u2014')
        result = parts[0]
        for part in parts[1:]:
            replacement = _sanitize_rng.choices(
                _EM_DASH_REPLACEMENTS, weights=_EM_DASH_WEIGHTS, k=1,
            )[0]
            # Capitalize after period
            if replacement == '. ' and part:
                part = part.lstrip()
                part = part[0].upper() + part[1:] if part else part
            result += replacement + part
        text = result

    return text


def _parse_turns(raw_text: str) -> list[dict]:
    """Parse the LLM's JSON response into a list of turn dicts."""
    # Strip markdown fences if present
    text = raw_text.strip()
    if text.startswith('```'):
        lines = text.split('\n')
        # Remove first and last lines (fences)
        lines = [l for l in lines if not l.strip().startswith('```')]
        text = '\n'.join(lines)

    data = json.loads(text)
    turns = data.get('turns', data if isinstance(data, list) else [])

    # Sanitize all text fields
    for turn in turns:
        if turn.get('utterance'):
            turn['utterance'] = _sanitize_text(turn['utterance'])

    return turns


# ── Single Conversation Generator ─────────────────────────────────────

def generate_one(
    domain: str,
    category: str,
    sample: dict,
    system_prompt: str,
    seed: int = 42,
) -> dict | None:
    """Generate a single conversation and attach labels.

    Returns a labeled conversation dict, or None on failure.
    """
    rng = random.Random(seed + sample['convo_idx'])
    persona_hint = sample_persona_hint(rng)
    convo_id = f'{domain}_{category}_{sample["convo_idx"] + 1:03d}'

    user_prompt = build_user_prompt(category, sample, persona_hint)

    try:
        result = _call_anthropic(system_prompt, user_prompt, MODEL)
        turns = result['turns']

        if len(turns) != 3:
            log.warning(f'{convo_id}: Expected 3 turns, got {len(turns)}')
            return None

        attacher = LABEL_ATTACHERS[category]
        labeled = attacher(sample, turns, convo_id, domain)
        labeled['_meta'] = {
            'model': result['model'],
            'latency_ms': result['latency_ms'],
            'input_tokens': result['input_tokens'],
            'output_tokens': result['output_tokens'],
        }
        return labeled

    except Exception as e:
        log.error(f'{convo_id}: Generation failed: {e}')
        return None


# ── Batch Generator ───────────────────────────────────────────────────

def _load_existing(output_path: Path) -> list[dict]:
    """Load existing conversations from a JSON file for resume support."""
    if output_path.exists():
        with open(output_path) as f:
            try:
                return json.load(f)
            except (json.JSONDecodeError, ValueError):
                return []
    return []


def _compact_json(convos: list[dict]) -> str:
    """Render conversations as compact, human-readable JSON."""
    lines = ['[']
    for ci, convo in enumerate(convos):
        lines.append('  {')
        lines.append(f'    "convo_id": {json.dumps(convo["convo_id"])}, '
                     f'"category": {json.dumps(convo["category"])},')
        lines.append(f'    "scenario": {json.dumps(convo["scenario"])},')
        lines.append('    "turns": [')

        turns = convo['turns']
        for ti, turn in enumerate(turns):
            meta_parts = [f'"turn_num": {turn["turn_num"]}']
            for key in ('flow', 'intent', 'candidate_flows', 'candidate_intents'):
                if key in turn:
                    meta_parts.append(f'"{key}": {json.dumps(turn[key])}')
            meta_parts.append(f'"speaker": {json.dumps(turn["speaker"])}')

            if ti == 0:
                lines.append('      {')
            lines.append(f'        {", ".join(meta_parts)},')
            lines.append(f'        "utterance": {json.dumps(turn["utterance"])}')
            if 'rationale' in turn:
                lines[-1] += ','
                lines.append(f'        "rationale": {json.dumps(turn["rationale"])}')
            lines.append('      }, {' if ti < len(turns) - 1 else '      }')

        lines.append('    ]')
        lines.append('  },' if ci < len(convos) - 1 else '  }')

    lines.append(']')
    return '\n'.join(lines)


def _save_json(convos: list[dict], output_path: Path, domain: str = '') -> None:
    """Write conversations to compact JSON, sequentially numbered, _meta stripped."""
    clean = []
    for c in convos:
        c2 = {k: v for k, v in c.items() if k != '_meta'}
        clean.append(c2)
    clean.sort(key=lambda c: c['convo_id'])

    # Assign final sequential IDs: {domain}_001, {domain}_002, ...
    if not domain:
        # Infer domain from filename: gen_hugo.json -> hugo
        domain = output_path.stem.replace('gen_', '')
    for i, c in enumerate(clean):
        c['convo_id'] = f'{domain}_{i + 1:03d}'
    with open(output_path, 'w') as f:
        f.write(_compact_json(clean) + '\n')


def generate_batch(
    domain: str,
    categories: list[str] | None = None,
    n_per_cat: int = 32,
    workers: int = 4,
    seed: int = 42,
    output_dir: str | None = None,
    specific_ids: list[str] | None = None,
) -> list[dict]:
    """Generate conversations for one or more categories.

    Args:
        domain: 'hugo' or 'dana'
        categories: List of category letters ('a', 'b', 'c', 'd'). None = all.
        n_per_cat: Conversations per category.
        workers: Number of parallel workers.
        seed: Random seed for sampling.
        output_dir: Directory for JSON output. Saves after each conversation.
        specific_ids: If set, only generate these convo_ids (for regeneration).

    Returns:
        List of all generated (labeled) conversations.
    """
    if categories is None:
        categories = ['a', 'b', 'c', 'd']

    flow_catalog, dact_catalog, Intent = _load_domain(domain)
    samples = sample_all(domain, n_per_cat, seed)

    system_prompt = build_system_prompt(domain, flow_catalog, Intent)

    # Set up output path
    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent / 'eval')
    output_path = Path(output_dir) / f'gen_{domain}.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: load existing conversations
    existing = _load_existing(output_path)
    existing_ids = {c['convo_id'] for c in existing}

    # For regeneration: remove the specific IDs from existing so they get replaced
    if specific_ids:
        existing = [c for c in existing if c['convo_id'] not in specific_ids]
        existing_ids -= set(specific_ids)

    # Count existing conversations per category (for expansion from pilot data).
    # Pilot data uses final IDs (hugo_001) not temp IDs (hugo_a_001), so we
    # count by category field rather than matching temp IDs.
    _CAT_MAP = {
        'a': 'same_flow', 'b': 'switch_flow',
        'c': 'ambiguous_first', 'd': 'ambiguous_second',
    }
    existing_per_cat = {}
    for c in existing:
        cat_name = c.get('category', '')
        existing_per_cat[cat_name] = existing_per_cat.get(cat_name, 0) + 1

    # Build work items
    work_items = []
    for cat in categories:
        cat_name = _CAT_MAP[cat]
        n_existing = existing_per_cat.get(cat_name, 0)
        if n_existing > 0:
            log.info(f'Category {cat} ({cat_name}): {n_existing} pilot convos exist, '
                     f'generating indices {n_existing}-{n_per_cat - 1}')

        for sample in samples[cat]:
            cid = f'{domain}_{cat}_{sample["convo_idx"] + 1:03d}'

            # Skip indices covered by pilot data
            if sample['convo_idx'] < n_existing and not specific_ids:
                continue

            # Skip if already generated (resume support for temp IDs)
            if cid in existing_ids:
                log.info(f'Skipping {cid} (already exists)')
                continue

            # If regenerating specific IDs, only include those
            if specific_ids and cid not in specific_ids:
                continue

            work_items.append((cat, sample))

    if not work_items:
        log.info('No work items to generate.')
        return existing

    log.info(f'Generating {len(work_items)} conversations for {domain} '
             f'(categories: {categories}, workers: {workers})')

    all_convos = list(existing)
    failed = []
    _lock = __import__('threading').Lock()

    def _gen(item):
        cat, sample = item
        return generate_one(domain, cat, sample, system_prompt, seed)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_gen, item): item for item in work_items}

        for future in as_completed(futures):
            item = futures[future]
            cat, sample = item
            cid = f'{domain}_{cat}_{sample["convo_idx"] + 1:03d}'

            try:
                result = future.result()
                if result is not None:
                    with _lock:
                        all_convos.append(result)
                        _save_json(all_convos, output_path, domain)
                    log.info(f'Generated {cid}')
                else:
                    failed.append(cid)
            except Exception as e:
                log.error(f'{cid}: Unexpected error: {e}')
                failed.append(cid)

    if failed:
        log.warning(f'Failed to generate {len(failed)} conversations: {failed}')

    log.info(f'Generated {len(all_convos) - len(existing)} conversations, {len(failed)} failed')
    return all_convos
