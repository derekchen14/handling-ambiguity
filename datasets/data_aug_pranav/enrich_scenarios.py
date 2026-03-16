"""Enrich scenarios with natural flow sequences using multiple LLM families.

Step 2 of the synthetic data pipeline: for each scenario produced in Step 1,
generate a natural sequence of 5-7 flows that a user would progress through.
These sequences later support assigning each scenario to any of the 4
conversation categories (same_flow, switch_flow, ambiguous_first,
ambiguous_second).

Usage:
    python datasets/data_aug_pranav/enrich_scenarios.py \
        --domain hugo --batch-size 8 --seed 42 --max-threads 20

    # Single-model smoke test:
    python datasets/data_aug_pranav/enrich_scenarios.py \
        --domain hugo --models anthropic --batch-size 4

    # Dry run (print prompts, no API calls):
    python datasets/data_aug_pranav/enrich_scenarios.py \
        --domain hugo --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
from pathlib import Path

from tqdm import tqdm

# ── Path setup ───────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_SCRIPT_DIR = Path(__file__).resolve().parent

# ── Logging ──────────────────────────────────────────────────────────

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

MODEL_CONFIGS = [
    {
        'name': 'anthropic',
        'provider': 'anthropic',
        'model_id': 'claude-sonnet-4-6',
    },
    {
        'name': 'openai',
        'provider': 'openai',
        'model_id': 'gpt-5.2',
    },
    {
        'name': 'gemini',
        'provider': 'openrouter',
        'model_id': 'google/gemini-3-pro-preview',
    },
    {
        'name': 'deepseek',
        'provider': 'openrouter',
        'model_id': 'deepseek/deepseek-chat',
    },
]

DOMAIN_DESCRIPTIONS = {
    'hugo': 'a blog writing assistant that helps users research, draft, revise, and publish blog posts',
    'dana': 'a data analysis assistant that helps users clean, transform, analyze, and report on datasets',
}

DOMAIN_NAMES = {
    'hugo': 'Hugo',
    'dana': 'Dana',
}

MAX_RETRIES = 3
BACKOFF_BASE_S = 1.0


# ── Ontology helpers ─────────────────────────────────────────────────

def _load_domain(domain: str):
    """Import FLOW_CATALOG, DACT_CATALOG, and Intent for a domain."""
    if domain == 'hugo':
        from datasets.hugo.ontology import DACT_CATALOG, FLOW_CATALOG, Intent
    elif domain == 'dana':
        from datasets.dana.ontology import DACT_CATALOG, FLOW_CATALOG, Intent
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


# ── Async provider call functions ────────────────────────────────────

async def _call_anthropic_async(system_prompt: str, user_prompt: str) -> str:
    """Call Anthropic API using async client."""
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
    try:
        resp = await client.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=4096,
            temperature=0.7,
            system=system_prompt,
            messages=[{'role': 'user', 'content': user_prompt}],
        )
        return resp.content[0].text if resp.content else ''
    finally:
        await client.close()


async def _call_openai_async(system_prompt: str, user_prompt: str) -> str:
    """Call OpenAI API using async client."""
    import openai

    client = openai.AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
    try:
        resp = await client.chat.completions.create(
            model='gpt-5.2',
            max_completion_tokens=4096,
            temperature=0.7,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
        )
        return resp.choices[0].message.content or ''
    finally:
        await client.close()


async def _call_openrouter_async(system_prompt: str, user_prompt: str, model_id: str) -> str:
    """Call OpenRouter's OpenAI-compatible API using async client."""
    import openai

    client = openai.AsyncOpenAI(
        api_key=os.environ['OPEN_ROUTER_API_KEY'],
        base_url='https://openrouter.ai/api/v1',
    )
    try:
        resp = await client.chat.completions.create(
            model=model_id,
            max_completion_tokens=4096,
            temperature=0.7,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
        )
        if not resp or not resp.choices:
            raise RuntimeError('OpenRouter returned empty choices (transient error)')
        return resp.choices[0].message.content or ''
    finally:
        await client.close()


async def _call_model_async(config: dict, system_prompt: str, user_prompt: str) -> str:
    """Async dispatch to the right provider with retries."""
    provider = config['provider']
    last_error = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            if provider == 'anthropic':
                return await _call_anthropic_async(system_prompt, user_prompt)
            elif provider == 'openai':
                return await _call_openai_async(system_prompt, user_prompt)
            elif provider == 'openrouter':
                return await _call_openrouter_async(system_prompt, user_prompt, config['model_id'])
            else:
                raise ValueError(f'Unknown provider: {provider}')
        except Exception as e:
            last_error = e
            err_str = f'{type(e).__name__}: {e}'.lower()
            is_retryable = any(kw in err_str for kw in (
                'ratelimit', 'rate_limit', 'rate limit', 'resource_exhausted',
                'timeout', 'internal', 'server', '429', '500', '503',
            ))
            if not is_retryable:
                raise
            if attempt < MAX_RETRIES:
                delay = max(BACKOFF_BASE_S * (2 ** attempt), 5.0 if '429' in err_str else 1.0)
                log.warning(
                    'Retry %d/%d for %s after %s: %s',
                    attempt + 1, MAX_RETRIES, config['name'], type(e).__name__, e,
                )
                await asyncio.sleep(delay)

    raise last_error  # type: ignore[misc]


async def _run_wave(batch_specs: list[dict], system_prompt: str, semaphore: asyncio.Semaphore) -> list:
    """Run a wave of batches concurrently, bounded by semaphore."""
    async def _run_one(spec):
        async with semaphore:
            return await _call_model_async(spec['model_config'], system_prompt, spec['user_prompt'])
    return await asyncio.gather(*[_run_one(s) for s in batch_specs], return_exceptions=True)


# ── Prompt builders ──────────────────────────────────────────────────

def _build_system_prompt(domain: str, user_facing: dict) -> str:
    """Build the system prompt for flow sequence enrichment."""
    domain_name = DOMAIN_NAMES[domain]
    domain_desc = DOMAIN_DESCRIPTIONS[domain]

    # Build flow catalog section
    flow_lines = []
    for name, flow in user_facing.items():
        intent_val = flow['intent'].value if hasattr(flow['intent'], 'value') else str(flow['intent'])
        desc = flow.get('description', '')
        edges = flow.get('edge_flows', [])
        edges_str = ', '.join(edges) if edges else 'none'
        flow_lines.append(f'  - {name} (intent: {intent_val}) — {desc}\n    edge_flows: [{edges_str}]')
    flow_catalog_section = '\n'.join(flow_lines)

    return f"""You are a workflow analyst for {domain_name}, {domain_desc}.

Your task: given a scenario, identify the natural sequence of 5-7 flows a user would progress through to accomplish their goal in that scenario.

## Flow Catalog

{flow_catalog_section}

## Rules

1. Return exactly 5-7 flows per scenario. No fewer, no more.
2. The sequence must use at least 2 different intents (e.g. Research + Draft, or Draft + Revise).
3. No duplicate flows in a single sequence — each flow appears at most once.
4. Every flow name must come from the catalog above. Do not invent flow names.
5. Include at least one adjacent pair where the second flow appears in the first flow's edge_flows list.
6. Order flows by natural workflow progression — earlier phases (Research) before later phases (Publish).
7. Consider the scenario's example_utterances to understand what the user actually needs.

## Output Format

Return a JSON array of objects, one per scenario. Each object must have:
- "scenario_id": the scenario_id from the input
- "flow_sequence": an array of objects, each with "flow" (flow name) and "intent" (the intent of that flow from the catalog)

Return valid JSON only. No markdown fences, no explanation outside the JSON array."""


def _build_user_prompt(scenarios: list[dict]) -> str:
    """Build the per-batch user prompt with scenario details."""
    items = []
    for s in scenarios:
        items.append({
            'scenario_id': s['scenario_id'],
            'scenario': s['scenario'],
            'example_utterances': s['example_utterances'],
            'grounding_flows': s.get('grounding_flows', []),
            'grounding_intents': s.get('grounding_intents', []),
        })

    return f"""Enrich the following {len(scenarios)} scenarios with flow sequences.

{json.dumps(items, indent=2, ensure_ascii=False)}

Return a JSON array of {len(scenarios)} objects in the same order, each with "scenario_id" and "flow_sequence"."""


# ── Response parser ──────────────────────────────────────────────────

def _parse_enrichments(raw: str) -> list[dict]:
    """Extract enrichment list from LLM JSON response."""
    cleaned = raw.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', cleaned, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                log.warning('Failed to parse JSON array from response')
                return []
        else:
            log.warning('No JSON array found in response')
            return []

    if not isinstance(parsed, list):
        log.warning('Parsed result is not a list: %s', type(parsed))
        return []

    valid = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        if 'scenario_id' not in item:
            continue
        if 'flow_sequence' not in item or not isinstance(item['flow_sequence'], list):
            continue
        valid.append(item)

    return valid


# ── Validation and auto-repair ───────────────────────────────────────

def _validate_flow_sequence(flow_seq: list[dict], user_facing: dict) -> list[str]:
    """Validate a flow sequence. Returns list of error strings (empty = valid)."""
    errors = []

    # 1. Length 5-7
    if len(flow_seq) < 5:
        errors.append(f'too_short: {len(flow_seq)} flows (need 5-7)')
    elif len(flow_seq) > 7:
        errors.append(f'too_long: {len(flow_seq)} flows (need 5-7)')

    # 2. All flow names exist
    for step in flow_seq:
        if step.get('flow') not in user_facing:
            errors.append(f'unknown_flow: {step.get("flow")}')

    # 3. Intent matches catalog for each flow
    for step in flow_seq:
        fname = step.get('flow')
        if fname in user_facing:
            expected = user_facing[fname]['intent']
            expected_val = expected.value if hasattr(expected, 'value') else str(expected)
            if step.get('intent') != expected_val:
                errors.append(f'wrong_intent: {fname} should be {expected_val}, got {step.get("intent")}')

    # 4. At least 2 distinct intents
    intents = set()
    for step in flow_seq:
        fname = step.get('flow')
        if fname in user_facing:
            iv = user_facing[fname]['intent']
            intents.add(iv.value if hasattr(iv, 'value') else str(iv))
    if len(intents) < 2:
        errors.append(f'single_intent: only {intents}')

    # 5. No duplicate flows
    flow_names = [s.get('flow') for s in flow_seq]
    if len(flow_names) != len(set(flow_names)):
        dupes = [f for f in flow_names if flow_names.count(f) > 1]
        errors.append(f'duplicate_flows: {set(dupes)}')

    # 6. At least one adjacent edge_flow pair
    has_edge_pair = False
    for i in range(len(flow_seq) - 1):
        a = flow_seq[i].get('flow')
        b = flow_seq[i + 1].get('flow')
        if a in user_facing and b in user_facing.get(a, {}).get('edge_flows', []):
            has_edge_pair = True
            break
    if not has_edge_pair:
        errors.append('no_edge_pair: no adjacent flows connected by edge_flows')

    return errors


def _auto_repair(flow_seq: list[dict], user_facing: dict, rng: random.Random) -> list[dict]:
    """Attempt to auto-repair a flow sequence. Returns repaired copy."""
    seq = [dict(step) for step in flow_seq]  # shallow copy each step

    # Remove unknown flow names
    seq = [s for s in seq if s.get('flow') in user_facing]

    # Fix wrong intent labels
    for step in seq:
        fname = step.get('flow')
        if fname in user_facing:
            expected = user_facing[fname]['intent']
            step['intent'] = expected.value if hasattr(expected, 'value') else str(expected)

    # Remove duplicates (keep first occurrence)
    seen = set()
    deduped = []
    for step in seq:
        if step['flow'] not in seen:
            seen.add(step['flow'])
            deduped.append(step)
    seq = deduped

    # Too long → truncate to 7
    if len(seq) > 7:
        seq = seq[:7]

    # Too short → append from edge_flows of last flow
    attempts = 0
    while len(seq) < 5 and attempts < 10:
        attempts += 1
        last_flow = seq[-1]['flow'] if seq else None
        if last_flow and last_flow in user_facing:
            edges = user_facing[last_flow].get('edge_flows', [])
            # Filter to user-facing and not already in sequence
            candidates = [e for e in edges if e in user_facing and e not in {s['flow'] for s in seq}]
            if candidates:
                pick = rng.choice(candidates)
                intent_val = user_facing[pick]['intent']
                seq.append({
                    'flow': pick,
                    'intent': intent_val.value if hasattr(intent_val, 'value') else str(intent_val),
                })
                continue
        # Fallback: pick any user-facing flow not already in sequence
        remaining = [f for f in user_facing if f not in {s['flow'] for s in seq}]
        if remaining:
            pick = rng.choice(remaining)
            intent_val = user_facing[pick]['intent']
            seq.append({
                'flow': pick,
                'intent': intent_val.value if hasattr(intent_val, 'value') else str(intent_val),
            })
        else:
            break

    # No edge_flow pair → swap last flow with one from edge_flows of second-to-last
    has_edge_pair = False
    for i in range(len(seq) - 1):
        a = seq[i]['flow']
        b = seq[i + 1]['flow']
        if b in user_facing.get(a, {}).get('edge_flows', []):
            has_edge_pair = True
            break

    if not has_edge_pair and len(seq) >= 2:
        second_last = seq[-2]['flow']
        if second_last in user_facing:
            edges = user_facing[second_last].get('edge_flows', [])
            candidates = [e for e in edges if e in user_facing and e not in {s['flow'] for s in seq}]
            if candidates:
                pick = rng.choice(candidates)
                intent_val = user_facing[pick]['intent']
                seq[-1] = {
                    'flow': pick,
                    'intent': intent_val.value if hasattr(intent_val, 'value') else str(intent_val),
                }

    return seq


# ── edge_flow_pairs extraction ───────────────────────────────────────

def _extract_edge_pairs(flow_seq: list[dict], user_facing: dict) -> list[list[str]]:
    """Extract adjacent pairs connected by edge_flows in the ontology."""
    pairs = []
    for i in range(len(flow_seq) - 1):
        a, b = flow_seq[i]['flow'], flow_seq[i + 1]['flow']
        if b in user_facing.get(a, {}).get('edge_flows', []):
            pairs.append([a, b])
    return pairs


# ── Orchestrator ─────────────────────────────────────────────────────

def enrich_scenarios(
    domain: str,
    batch_size: int = 8,
    seed: int = 42,
    models_filter: list[str] | None = None,
    dry_run: bool = False,
    verbose: bool = False,
    max_threads: int | None = None,
) -> Path:
    """Main orchestrator: enrich scenarios with flow sequences.

    Returns path to the output JSONL file.
    """
    rng = random.Random(seed)

    # Load domain ontology
    flow_catalog, dact_catalog, Intent = _load_domain(domain)
    user_facing = _user_facing_flows(flow_catalog, Intent)
    by_intent = _flows_by_intent(user_facing)

    # Filter models if requested
    active_models = MODEL_CONFIGS[:]
    if models_filter:
        active_models = [m for m in active_models if m['name'] in models_filter]
        if not active_models:
            raise ValueError(f'No models match filter: {models_filter}')

    # Input/output paths
    input_jsonl = _SCRIPT_DIR / f'scenarios_{domain}.jsonl'
    output_jsonl = _SCRIPT_DIR / f'scenarios_{domain}_enriched.jsonl'
    output_meta = _SCRIPT_DIR / f'scenarios_{domain}_enriched_meta.json'

    # Load all input scenarios
    all_scenarios: list[dict] = []
    if not input_jsonl.exists():
        raise FileNotFoundError(f'Input not found: {input_jsonl}')
    with open(input_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                all_scenarios.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    log.info('Loaded %d scenarios from %s', len(all_scenarios), input_jsonl)

    # Resume: load already-enriched scenario_ids
    enriched_ids: set[str] = set()
    if output_jsonl.exists():
        with open(output_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    enriched_ids.add(obj['scenario_id'])
                except (json.JSONDecodeError, KeyError):
                    continue

    # Filter remaining
    remaining_scenarios = [s for s in all_scenarios if s['scenario_id'] not in enriched_ids]

    if not remaining_scenarios:
        log.info('All %d scenarios already enriched, nothing to do', len(all_scenarios))
        return output_jsonl

    log.info(
        'Enriching %d remaining scenarios for %s (have %d, total %d)',
        len(remaining_scenarios), domain, len(enriched_ids), len(all_scenarios),
    )

    # Build system prompt once
    system_prompt = _build_system_prompt(domain, user_facing)

    # Concurrency setup
    if max_threads is None:
        max_threads = len(active_models)

    # Model assignment: sort remaining by index, split into equal chunks per model
    # Sort by scenario_id to ensure deterministic assignment
    remaining_scenarios.sort(key=lambda s: s['scenario_id'])
    chunk_size = (len(remaining_scenarios) + len(active_models) - 1) // len(active_models)

    # Assign model to each scenario
    scenario_model_map: dict[str, dict] = {}
    for i, scenario in enumerate(remaining_scenarios):
        model_idx = i // chunk_size if chunk_size > 0 else 0
        model_idx = min(model_idx, len(active_models) - 1)
        scenario_model_map[scenario['scenario_id']] = active_models[model_idx]

    # Group into batches of batch_size, all same model
    batches: list[tuple[dict, list[dict]]] = []  # (model_config, scenarios)
    for model_config in active_models:
        model_scenarios = [
            s for s in remaining_scenarios
            if scenario_model_map[s['scenario_id']]['name'] == model_config['name']
        ]
        for i in range(0, len(model_scenarios), batch_size):
            batch = model_scenarios[i:i + batch_size]
            batches.append((model_config, batch))

    # Shuffle batches to interleave models (better load distribution)
    rng.shuffle(batches)

    # --- Dry-run path ---
    if dry_run:
        for batch_num, (model_config, batch_scenarios) in enumerate(batches):
            user_prompt = _build_user_prompt(batch_scenarios)
            print(f'\n{"="*70}')
            print(f'BATCH {batch_num + 1}/{len(batches)}')
            print(f'Model: {model_config["name"]} ({model_config["model_id"]})')
            print(f'Scenarios: {[s["scenario_id"] for s in batch_scenarios]}')
            print(f'{"="*70}')
            print(f'\n--- SYSTEM PROMPT ---\n{system_prompt[:500]}...')
            print(f'\n--- USER PROMPT ---\n{user_prompt[:1000]}...')

        # Model distribution
        model_dist = {}
        for s in remaining_scenarios:
            mname = scenario_model_map[s['scenario_id']]['name']
            model_dist[mname] = model_dist.get(mname, 0) + 1
        print(f'\n[DRY RUN] Would enrich {len(remaining_scenarios)} scenarios '
              f'across {len(batches)} batches (batch_size={batch_size})')
        print(f'Model distribution: {model_dist}')
        return output_jsonl

    # --- Live path: wave-based async orchestration ---
    total_enriched: list[dict] = []
    total_failed: list[dict] = []
    wave_num = 0
    batch_cursor = 0
    pbar = tqdm(total=len(remaining_scenarios), unit='scenarios', desc='Enriching')

    while batch_cursor < len(batches):
        # Build wave specs
        wave_specs: list[dict] = []
        wave_batch_scenarios: list[list[dict]] = []

        for _ in range(max_threads):
            if batch_cursor >= len(batches):
                break
            model_config, batch_scenarios = batches[batch_cursor]
            user_prompt = _build_user_prompt(batch_scenarios)

            wave_specs.append({
                'model_config': model_config,
                'user_prompt': user_prompt,
                'batch_num': batch_cursor,
            })
            wave_batch_scenarios.append(batch_scenarios)
            batch_cursor += 1

        if not wave_specs:
            break

        # Fire wave
        semaphore = asyncio.Semaphore(max_threads)
        results = asyncio.run(_run_wave(wave_specs, system_prompt, semaphore))

        # Process results
        wave_valid: list[dict] = []

        for spec, batch_scenarios, result in zip(wave_specs, wave_batch_scenarios, results):
            bnum = spec['batch_num'] + 1
            mname = spec['model_config']['name']

            if isinstance(result, BaseException):
                pbar.write(f'ERROR: Batch {bnum} failed ({mname}): {result}')
                total_failed.extend(batch_scenarios)
                continue

            parsed = _parse_enrichments(result)
            if not parsed:
                pbar.write(f'WARNING: Batch {bnum}: no valid enrichments parsed from {mname}')
                total_failed.extend(batch_scenarios)
                continue

            # Index parsed by scenario_id
            parsed_map = {p['scenario_id']: p for p in parsed}

            for scenario in batch_scenarios:
                sid = scenario['scenario_id']
                enrichment = parsed_map.get(sid)

                if not enrichment:
                    log.debug('Missing enrichment for %s', sid)
                    total_failed.append(scenario)
                    continue

                flow_seq = enrichment['flow_sequence']

                # Auto-repair before validation
                flow_seq = _auto_repair(flow_seq, user_facing, rng)

                # Validate
                errors = _validate_flow_sequence(flow_seq, user_facing)
                if errors:
                    log.debug('Validation failed for %s after repair: %s', sid, errors)
                    total_failed.append(scenario)
                    continue

                # Compute edge_flow_pairs
                edge_pairs = _extract_edge_pairs(flow_seq, user_facing)

                # Merge into original scenario
                mc = scenario_model_map[sid]
                output_obj = dict(scenario)
                output_obj['flow_sequence'] = flow_seq
                output_obj['edge_flow_pairs'] = edge_pairs
                output_obj['enrichment_model'] = mc['model_id']
                output_obj['enrichment_provider'] = mc['provider']

                wave_valid.append(output_obj)

        # Flush wave results to JSONL
        if wave_valid:
            with open(output_jsonl, 'a') as f:
                for obj in wave_valid:
                    f.write(json.dumps(obj, ensure_ascii=False) + '\n')
            total_enriched.extend(wave_valid)
            pbar.update(len(wave_valid))

        wave_num += 1
        pbar.set_postfix_str(f'wave {wave_num}')

    pbar.close()

    # Retry failed scenarios (up to 2 retry passes with smaller batch size)
    retry_batch_size = max(batch_size // 2, 1)
    for retry_pass in range(2):
        if not total_failed:
            break

        log.info('Retry pass %d: %d failed scenarios (batch_size=%d)',
                 retry_pass + 1, len(total_failed), retry_batch_size)

        # Re-batch failed scenarios
        retry_batches: list[tuple[dict, list[dict]]] = []
        for scenario in total_failed:
            mc = scenario_model_map[scenario['scenario_id']]
            # Find or create a batch for this model
            if not retry_batches or retry_batches[-1][0]['name'] != mc['name'] or len(retry_batches[-1][1]) >= retry_batch_size:
                retry_batches.append((mc, []))
            retry_batches[-1][1].append(scenario)

        rng.shuffle(retry_batches)
        retry_failed: list[dict] = []
        retry_enriched: list[dict] = []

        pbar = tqdm(total=len(total_failed), unit='scenarios', desc=f'Retry {retry_pass + 1}')
        batch_cursor = 0

        while batch_cursor < len(retry_batches):
            wave_specs = []
            wave_batch_scenarios = []

            for _ in range(max_threads):
                if batch_cursor >= len(retry_batches):
                    break
                mc, batch_scenarios = retry_batches[batch_cursor]
                user_prompt = _build_user_prompt(batch_scenarios)

                wave_specs.append({
                    'model_config': mc,
                    'user_prompt': user_prompt,
                    'batch_num': batch_cursor,
                })
                wave_batch_scenarios.append(batch_scenarios)
                batch_cursor += 1

            if not wave_specs:
                break

            semaphore = asyncio.Semaphore(max_threads)
            results = asyncio.run(_run_wave(wave_specs, system_prompt, semaphore))

            wave_valid = []
            for spec, batch_scenarios, result in zip(wave_specs, wave_batch_scenarios, results):
                bnum = spec['batch_num'] + 1
                mname = spec['model_config']['name']

                if isinstance(result, BaseException):
                    pbar.write(f'RETRY ERROR: Batch {bnum} ({mname}): {result}')
                    retry_failed.extend(batch_scenarios)
                    continue

                parsed = _parse_enrichments(result)
                if not parsed:
                    pbar.write(f'RETRY WARNING: Batch {bnum}: no valid enrichments from {mname}')
                    retry_failed.extend(batch_scenarios)
                    continue

                parsed_map = {p['scenario_id']: p for p in parsed}

                for scenario in batch_scenarios:
                    sid = scenario['scenario_id']
                    enrichment = parsed_map.get(sid)

                    if not enrichment:
                        retry_failed.append(scenario)
                        continue

                    flow_seq = _auto_repair(enrichment['flow_sequence'], user_facing, rng)
                    errors = _validate_flow_sequence(flow_seq, user_facing)
                    if errors:
                        log.debug('Retry validation failed for %s: %s', sid, errors)
                        retry_failed.append(scenario)
                        continue

                    edge_pairs = _extract_edge_pairs(flow_seq, user_facing)
                    mc = scenario_model_map[sid]
                    output_obj = dict(scenario)
                    output_obj['flow_sequence'] = flow_seq
                    output_obj['edge_flow_pairs'] = edge_pairs
                    output_obj['enrichment_model'] = mc['model_id']
                    output_obj['enrichment_provider'] = mc['provider']

                    wave_valid.append(output_obj)

            if wave_valid:
                with open(output_jsonl, 'a') as f:
                    for obj in wave_valid:
                        f.write(json.dumps(obj, ensure_ascii=False) + '\n')
                retry_enriched.extend(wave_valid)
                pbar.update(len(wave_valid))

        pbar.close()
        total_enriched.extend(retry_enriched)
        total_failed = retry_failed

    if total_failed:
        log.warning('%d scenarios could not be enriched after retries: %s',
                    len(total_failed),
                    [s['scenario_id'] for s in total_failed[:20]])

    # Write summary meta JSON
    model_counts: dict[str, int] = {}
    for s in total_enriched:
        model_counts[s['enrichment_model']] = model_counts.get(s['enrichment_model'], 0) + 1

    meta = {
        'domain': domain,
        'total_input': len(all_scenarios),
        'total_enriched': len(enriched_ids) + len(total_enriched),
        'new_enriched': len(total_enriched),
        'resumed_from': len(enriched_ids),
        'failed': len(total_failed),
        'failed_ids': [s['scenario_id'] for s in total_failed],
        'model_distribution': model_counts,
        'seed': seed,
        'batch_size': batch_size,
        'max_threads': max_threads,
        'waves': wave_num,
        'output_file': str(output_jsonl),
    }

    with open(output_meta, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    log.info('Summary: %s', json.dumps(meta, indent=2))

    return output_jsonl


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    # Load .env
    env_path = _PROJECT_ROOT / '.env'
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)

    parser = argparse.ArgumentParser(
        description='Enrich scenarios with flow sequences using multiple LLM families',
    )
    parser.add_argument(
        '--domain', required=True, choices=['hugo', 'dana'],
        help='Domain to enrich scenarios for',
    )
    parser.add_argument(
        '--batch-size', type=int, default=8,
        help='Scenarios per LLM call (default: 8)',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)',
    )
    parser.add_argument(
        '--models', type=str, default=None,
        help='Comma-separated model filter (e.g. "anthropic,openai")',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print prompts without calling LLMs',
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable debug logging',
    )
    parser.add_argument(
        '--max-threads', type=int, default=None,
        help='Max concurrent API calls (default: number of active models)',
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    # Only our logger gets DEBUG/INFO — keep httpcore/httpx silent
    log.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    models_filter = None
    if args.models:
        models_filter = [m.strip() for m in args.models.split(',')]

    output_path = enrich_scenarios(
        domain=args.domain,
        batch_size=args.batch_size,
        seed=args.seed,
        models_filter=models_filter,
        dry_run=args.dry_run,
        verbose=args.verbose,
        max_threads=args.max_threads,
    )

    print(f'\nOutput: {output_path}')


if __name__ == '__main__':
    main()
