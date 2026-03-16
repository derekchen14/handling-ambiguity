"""Generate diverse scenario objects using multiple LLM families.

Step 1 of the synthetic data pipeline: produce a large, diverse set of rich
scenario objects (description + example utterances) to be consumed by
downstream generation steps.

Usage:
    python datasets/data_aug_pranav/generate_scenarios.py \
        --domain hugo --target 200 --batch-size 12 --seed 42

    # Single-batch smoke test:
    python datasets/data_aug_pranav/generate_scenarios.py \
        --domain hugo --target 12 --models anthropic

    # Dry run (print prompts, no API calls):
    python datasets/data_aug_pranav/generate_scenarios.py \
        --domain hugo --target 24 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

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

DIVERSITY_AXES = [
    'enterprise and corporate contexts',
    'personal and hobbyist contexts',
    'academic and research contexts',
    'startup and small-business contexts',
    'non-profit and public-sector contexts',
    'creative and artistic contexts',
    'technical and engineering contexts',
    'marketing and growth contexts',
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


def _sample_flows(flows: dict, rng: random.Random, n: int = 7) -> list[tuple[str, dict]]:
    """Sample n flows from the catalog for grounding."""
    names = list(flows.keys())
    k = min(n, len(names))
    sampled = rng.sample(names, k)
    return [(name, flows[name]) for name in sampled]


# ── Prompt builders ──────────────────────────────────────────────────

def _build_system_prompt(domain: str) -> str:
    """Build the system prompt for scenario generation."""
    domain_name = DOMAIN_NAMES[domain]
    domain_desc = DOMAIN_DESCRIPTIONS[domain]

    return f"""You are a scenario generator for creating evaluation data for an AI assistant called {domain_name}.

{domain_name} is {domain_desc}.

Your job is to generate diverse, realistic scenario descriptions along with example user utterances. Each scenario describes a specific use case or context in which a user would interact with {domain_name}.

## Output Requirements

Return a JSON array of scenario objects. Each object must have:
- "scenario": A specific, detailed scenario description (1-2 sentences). Include the domain/topic, the user's goal, and any relevant context.
- "example_utterances": A list of 3-5 short, realistic user messages (10-30 words each) that someone in this scenario might type on their phone. These should be terse, natural, and varied in style.

## Rules

1. Scenarios must be specific and concrete — not generic. Bad: "Writing a blog post". Good: "Corporate cybersecurity newsletter — explaining zero-trust architecture to non-technical executives".
2. Example utterances must sound like real phone-typed messages: short, terse, skip pleasantries, assume shared context.
3. NEVER use technical flow names or system terminology in utterances. Use natural language only.
4. Vary sentence structure, length, and register across utterances. Mix terse commands with casual questions.
5. Avoid em dashes, fancy Unicode punctuation, and overly polished prose. Use commas, periods, and plain language.
6. Return valid JSON only. No markdown fences, no explanation outside the JSON array."""


def _build_user_prompt(
    domain: str,
    batch_size: int,
    diversity_axis: str,
    grounding_flows: list[tuple[str, dict]],
    exclusion_list: list[str],
) -> str:
    """Build the per-batch user prompt."""
    domain_name = DOMAIN_NAMES[domain]

    # Format grounding flows
    flow_lines = []
    for name, flow in grounding_flows:
        intent_val = flow['intent'].value if hasattr(flow['intent'], 'value') else str(flow['intent'])
        desc = flow.get('description', '')
        flow_lines.append(f'  - {name} ({intent_val}): {desc}')
    flow_section = '\n'.join(flow_lines)

    # Format exclusion list
    if exclusion_list:
        # Show up to 40 most recent to keep prompt manageable
        shown = exclusion_list[-40:]
        exclusion_section = (
            '\n## Already Generated (do NOT repeat or closely paraphrase these)\n'
            + '\n'.join(f'  - {s}' for s in shown)
        )
    else:
        exclusion_section = ''

    return f"""Generate exactly {batch_size} scenario objects for {domain_name}.

## Diversity Focus
Focus on: **{diversity_axis}**. All scenarios in this batch should relate to {diversity_axis}.

## Grounding Flows
These are some of {domain_name}'s capabilities. Use them to inspire scenarios where users would need these features:
{flow_section}

For each scenario, pick 1-3 of these flows as relevant and include their names in a "grounding_flows" field. Also include a "grounding_intents" field listing the unique intent categories (e.g. "Draft", "Revise") of those flows.
{exclusion_section}

Return a JSON array of {batch_size} objects, each with:
- "scenario": string (specific scenario description)
- "example_utterances": list of 3-5 strings
- "grounding_flows": list of flow name strings from the grounding flows above
- "grounding_intents": list of intent category strings

Example:
[
  {{
    "scenario": "Corporate cybersecurity newsletter -- explaining zero-trust architecture to non-technical executives",
    "example_utterances": [
      "Can you check if the zero-trust piece reads okay for a non-technical audience?",
      "I need an outline for the next issue, focusing on phishing trends",
      "The intro is too jargon-heavy, tone it down"
    ],
    "grounding_flows": ["write", "tone", "outline"],
    "grounding_intents": ["Draft", "Revise"]
  }}
]

Return ONLY the JSON array."""


# ── Provider call functions ──────────────────────────────────────────

def _call_anthropic(system_prompt: str, user_prompt: str) -> str:
    """Call Anthropic API directly."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
    resp = client.messages.create(
        model='claude-sonnet-4-6',
        max_tokens=4096,
        temperature=0.9,
        system=system_prompt,
        messages=[{'role': 'user', 'content': user_prompt}],
    )
    return resp.content[0].text if resp.content else ''


def _call_openai(system_prompt: str, user_prompt: str) -> str:
    """Call OpenAI API directly."""
    import openai

    client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    resp = client.chat.completions.create(
        model='gpt-5.2',
        max_completion_tokens=4096,
        temperature=0.9,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
    )
    return resp.choices[0].message.content or ''


def _call_openrouter(system_prompt: str, user_prompt: str, model_id: str) -> str:
    """Call OpenRouter's OpenAI-compatible API (for Gemini, DeepSeek, etc.)."""
    import openai

    client = openai.OpenAI(
        api_key=os.environ['OPEN_ROUTER_API_KEY'],
        base_url='https://openrouter.ai/api/v1',
    )
    resp = client.chat.completions.create(
        model=model_id,
        max_completion_tokens=4096,
        temperature=0.9,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
    )
    if not resp or not resp.choices:
        raise RuntimeError('OpenRouter returned empty choices (transient error)')
    return resp.choices[0].message.content or ''


def _call_model(config: dict, system_prompt: str, user_prompt: str) -> str:
    """Dispatch to the right provider with retries."""
    provider = config['provider']
    last_error = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            if provider == 'anthropic':
                return _call_anthropic(system_prompt, user_prompt)
            elif provider == 'openai':
                return _call_openai(system_prompt, user_prompt)
            elif provider == 'openrouter':
                return _call_openrouter(system_prompt, user_prompt, config['model_id'])
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
                time.sleep(delay)

    raise last_error  # type: ignore[misc]


# ── Response parser ──────────────────────────────────────────────────

def _parse_scenarios(raw: str) -> list[dict]:
    """Extract scenario list from LLM JSON response."""
    # Strip markdown fences if present
    cleaned = raw.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find a JSON array in the response
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

    # Validate each scenario object
    valid = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        if 'scenario' not in item:
            continue
        if 'example_utterances' not in item or not isinstance(item['example_utterances'], list):
            continue
        valid.append(item)

    return valid


# ── Dedup via Jaccard ────────────────────────────────────────────────

def _jaccard_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity between two strings (token-level)."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _is_duplicate(scenario: str, existing: list[str], threshold: float = 0.5) -> bool:
    """Check if a scenario is too similar to any existing one."""
    for existing_scenario in existing:
        if _jaccard_similarity(scenario, existing_scenario) > threshold:
            return True
    return False


# ── Orchestrator ─────────────────────────────────────────────────────

def generate_scenarios(
    domain: str,
    target: int = 200,
    batch_size: int = 12,
    seed: int = 42,
    models_filter: list[str] | None = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> Path:
    """Main orchestrator: round-robin models x diversity axes, dedup, JSONL append.

    Returns path to the output JSONL file.
    """
    rng = random.Random(seed)

    # Load domain ontology
    flow_catalog, dact_catalog, Intent = _load_domain(domain)
    flows = _user_facing_flows(flow_catalog, Intent)
    by_intent = _flows_by_intent(flows)

    # Filter models if requested
    active_models = MODEL_CONFIGS[:]
    if models_filter:
        active_models = [m for m in active_models if m['name'] in models_filter]
        if not active_models:
            raise ValueError(f'No models match filter: {models_filter}')

    # Output paths
    output_jsonl = _SCRIPT_DIR / f'scenarios_{domain}.jsonl'
    output_meta = _SCRIPT_DIR / f'scenarios_{domain}_meta.json'

    # Resume support: load existing scenarios
    existing_scenarios: list[str] = []
    existing_ids: set[str] = set()
    scenario_counter = 0

    if output_jsonl.exists():
        with open(output_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    existing_scenarios.append(obj['scenario'])
                    existing_ids.add(obj['scenario_id'])
                    # Track the highest counter
                    sid = obj['scenario_id']
                    parts = sid.rsplit('_', 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        scenario_counter = max(scenario_counter, int(parts[1]) + 1)
                except (json.JSONDecodeError, KeyError):
                    continue

    remaining = target - len(existing_scenarios)
    if remaining <= 0:
        log.info('Already have %d scenarios (target=%d), nothing to do', len(existing_scenarios), target)
        return output_jsonl

    log.info(
        'Generating %d more scenarios for %s (have %d, target %d)',
        remaining, domain, len(existing_scenarios), target,
    )

    # Build system prompt once
    system_prompt = _build_system_prompt(domain)

    # Round-robin state
    model_idx = 0
    axis_idx = 0
    all_generated: list[str] = list(existing_scenarios)  # for dedup
    new_scenarios: list[dict] = []

    num_batches = (remaining + batch_size - 1) // batch_size

    for batch_num in range(num_batches):
        if len(new_scenarios) >= remaining:
            break

        # Pick model and diversity axis
        model_config = active_models[model_idx % len(active_models)]
        diversity_axis = DIVERSITY_AXES[axis_idx % len(DIVERSITY_AXES)]

        # Sample grounding flows
        grounding = _sample_flows(flows, rng, n=7)

        # Build user prompt
        user_prompt = _build_user_prompt(
            domain=domain,
            batch_size=batch_size,
            diversity_axis=diversity_axis,
            grounding_flows=grounding,
            exclusion_list=all_generated,
        )

        if dry_run:
            print(f'\n{"="*70}')
            print(f'BATCH {batch_num + 1}/{num_batches}')
            print(f'Model: {model_config["name"]} ({model_config["model_id"]})')
            print(f'Diversity axis: {diversity_axis}')
            print(f'Grounding flows: {[name for name, _ in grounding]}')
            print(f'{"="*70}')
            print(f'\n--- SYSTEM PROMPT ---\n{system_prompt[:500]}...')
            print(f'\n--- USER PROMPT ---\n{user_prompt[:1000]}...')
            model_idx += 1
            axis_idx += 1
            continue

        # Call LLM
        log.info(
            'Batch %d/%d: model=%s, axis=%s',
            batch_num + 1, num_batches, model_config['name'], diversity_axis,
        )

        try:
            raw_response = _call_model(model_config, system_prompt, user_prompt)
        except Exception as e:
            log.error('Batch %d failed (%s): %s', batch_num + 1, model_config['name'], e)
            # Advance to next model to avoid getting stuck
            model_idx += 1
            continue

        # Parse and validate
        batch_scenarios = _parse_scenarios(raw_response)
        if not batch_scenarios:
            log.warning('Batch %d: no valid scenarios parsed from %s', batch_num + 1, model_config['name'])
            model_idx += 1
            continue

        if verbose:
            log.info('Batch %d: parsed %d scenarios', batch_num + 1, len(batch_scenarios))

        # Dedup and write
        grounding_flow_names = [name for name, _ in grounding]
        for scenario_obj in batch_scenarios:
            if len(new_scenarios) >= remaining:
                break

            scenario_desc = scenario_obj['scenario']

            # Skip duplicates
            if _is_duplicate(scenario_desc, all_generated):
                if verbose:
                    log.info('Skipping duplicate: %s', scenario_desc[:60])
                continue

            # Build rich output object
            scenario_id = f'{domain}_{scenario_counter:03d}'
            scenario_counter += 1

            # Validate grounding_flows against actual flow names
            raw_grounding = scenario_obj.get('grounding_flows', [])
            valid_grounding = [f for f in raw_grounding if f in flows]
            if not valid_grounding:
                # Fall back to a random subset of the grounding flows we provided
                valid_grounding = rng.sample(
                    grounding_flow_names,
                    min(2, len(grounding_flow_names)),
                )

            # Validate grounding_intents
            raw_intents = scenario_obj.get('grounding_intents', [])
            all_intent_names = set(by_intent.keys())
            valid_intents = [i for i in raw_intents if i in all_intent_names]
            if not valid_intents:
                # Derive from valid_grounding flows
                valid_intents = list({
                    flows[f]['intent'].value
                    for f in valid_grounding
                    if f in flows
                })

            output_obj = {
                'scenario_id': scenario_id,
                'domain': domain,
                'scenario': scenario_desc,
                'example_utterances': scenario_obj['example_utterances'],
                'grounding_flows': valid_grounding,
                'grounding_intents': valid_intents,
                'diversity_axis': diversity_axis,
                'model': model_config['model_id'],
                'provider': model_config['provider'],
            }

            new_scenarios.append(output_obj)
            all_generated.append(scenario_desc)

        # Advance round-robin
        model_idx += 1
        axis_idx += 1

    if dry_run:
        print(f'\n[DRY RUN] Would generate ~{num_batches * batch_size} scenarios across {num_batches} batches')
        return output_jsonl

    # Append new scenarios to JSONL
    if new_scenarios:
        with open(output_jsonl, 'a') as f:
            for obj in new_scenarios:
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')

        log.info('Wrote %d new scenarios to %s', len(new_scenarios), output_jsonl)

    # Write summary meta JSON
    total = len(existing_scenarios) + len(new_scenarios)
    model_counts: dict[str, int] = {}
    axis_counts: dict[str, int] = {}
    for s in new_scenarios:
        model_counts[s['model']] = model_counts.get(s['model'], 0) + 1
        axis_counts[s['diversity_axis']] = axis_counts.get(s['diversity_axis'], 0) + 1

    meta = {
        'domain': domain,
        'target': target,
        'total_scenarios': total,
        'new_scenarios': len(new_scenarios),
        'resumed_from': len(existing_scenarios),
        'model_distribution': model_counts,
        'diversity_distribution': axis_counts,
        'seed': seed,
        'batch_size': batch_size,
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
        description='Generate diverse scenario objects using multiple LLM families',
    )
    parser.add_argument(
        '--domain', required=True, choices=['hugo', 'dana'],
        help='Domain to generate scenarios for',
    )
    parser.add_argument(
        '--target', type=int, default=200,
        help='Target number of scenarios (default: 200)',
    )
    parser.add_argument(
        '--batch-size', type=int, default=12,
        help='Scenarios per LLM call (default: 12)',
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

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    models_filter = None
    if args.models:
        models_filter = [m.strip() for m in args.models.split(',')]

    output_path = generate_scenarios(
        domain=args.domain,
        target=args.target,
        batch_size=args.batch_size,
        seed=args.seed,
        models_filter=models_filter,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    print(f'\nOutput: {output_path}')


if __name__ == '__main__':
    main()
