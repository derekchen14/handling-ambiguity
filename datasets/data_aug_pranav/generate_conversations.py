"""Generate 3-turn conversations from enriched scenarios.

Step 4 of the synthetic data pipeline: for each enriched+deduped scenario,
generate a 3-turn conversation (user → agent → user) in one of 4 categories
(same_flow, switch_flow, ambiguous_first, ambiguous_second).

Usage:
    python datasets/data_aug_pranav/generate_conversations.py \
        --domain hugo --seed 42

    python datasets/data_aug_pranav/generate_conversations.py \
        --domain dana --seed 42 --max-threads 8

    # Dry run:
    python datasets/data_aug_pranav/generate_conversations.py \
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

CATEGORIES = ['same_flow', 'switch_flow', 'ambiguous_first', 'ambiguous_second']

# Plan orchestrator flow per domain
PLAN_ORCHESTRATOR = {
    'dana': 'outline',
    'hugo': 'blueprint',
}

DOMAIN_DESCRIPTIONS = {
    'dana': (
        'Dana is a data-analyst copilot. Users work with tables/datasets: '
        'cleaning, transforming, analyzing, charting, and exporting data. '
        'The context field should contain a realistic table name and column names.'
    ),
    'hugo': (
        'Hugo is a blog-writing copilot. Users draft, revise, publish, and manage '
        'blog posts. The context field should contain a realistic post_id and post_title.'
    ),
}

CONTEXT_SCHEMA = {
    'dana': '{"table": "<table_name>", "columns": ["col1", "col2", ...]}',
    'hugo': '{"post_id": "<post_NNN>", "post_title": "<Title>", "platform": "<optional: wordpress/substack/medium>"}',
}

MAX_RETRIES = 3
BACKOFF_BASE_S = 1.0

# Intents that are user-facing (exclude Plan, Internal)
USER_FACING_INTENTS = {
    'dana': ['Clean', 'Transform', 'Analyze', 'Report', 'Converse'],
    'hugo': ['Research', 'Draft', 'Revise', 'Publish', 'Converse'],
}


# ── Load helpers ─────────────────────────────────────────────────────

def _load_scenarios(domain: str) -> list[dict]:
    """Load enriched deduped scenarios."""
    path = _SCRIPT_DIR / f'scenarios_{domain}_enriched_deduped.jsonl'
    scenarios = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                scenarios.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return scenarios


def _load_flow_tool_mapping(domain: str) -> str:
    """Load the flow-to-tool mapping markdown as a string."""
    path = _PROJECT_ROOT / 'tools' / f'flow_tool_mapping_{domain}.md'
    return path.read_text()


def _load_tool_manifest(domain: str) -> list[dict]:
    """Load the tool manifest JSON."""
    path = _PROJECT_ROOT / 'tools' / f'tool_manifest_{domain}.json'
    with open(path) as f:
        return json.load(f)


def _get_user_facing_flows(flow_mapping_text: str) -> dict[str, str]:
    """Parse flow names and their intents from flow_tool_mapping markdown.

    Returns {flow_name: intent_name}.
    """
    flows = {}
    current_intent = None
    for line in flow_mapping_text.splitlines():
        # Section headers like "## Clean (8 flows)"
        m = re.match(r'^##\s+(\w+)\s+\(', line)
        if m:
            current_intent = m.group(1)
            continue
        # Table rows like "| update | 9 | ..."
        if current_intent and line.startswith('|'):
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 3 and parts[1] and parts[1] != 'Flow' and parts[1] != '---':
                flow_name = parts[1]
                if flow_name.startswith('-'):
                    continue
                flows[flow_name] = current_intent
    return flows


def _tool_manifest_compact(manifest: list[dict]) -> str:
    """Compact tool manifest: name, description, params (no full schema)."""
    lines = []
    for tool in manifest:
        if tool.get('internal_component'):
            continue
        params = tool.get('input_schema', {}).get('properties', {})
        param_names = list(params.keys())
        required = tool.get('input_schema', {}).get('required', [])
        param_descs = []
        for pname in param_names:
            pinfo = params[pname]
            req = '(required)' if pname in required else '(optional)'
            ptype = pinfo.get('type', 'any')
            desc = pinfo.get('description', '')
            if len(desc) > 80:
                desc = desc[:80] + '...'
            param_descs.append(f'    {pname}: {ptype} {req} — {desc}')
        flows = tool.get('_flows', [])
        lines.append(f'- **{tool["name"]}** (flows: {", ".join(flows)})')
        lines.append(f'  {tool.get("description", "")}')
        for pd in param_descs:
            lines.append(pd)
    return '\n'.join(lines)


# ── Category assignment ──────────────────────────────────────────────

def assign_categories(
    scenarios: list[dict],
    rng: random.Random,
) -> dict[str, list[dict]]:
    """Split scenarios into 4 equal category buckets.

    Each scenario is augmented with category-specific fields:
      - assigned_category
      - assigned_flows (dict with category-specific flow assignments)
    """
    shuffled = list(scenarios)
    rng.shuffle(shuffled)

    n = len(shuffled)
    chunk = n // 4
    remainder = n % 4

    buckets: dict[str, list[dict]] = {}
    idx = 0
    for i, cat in enumerate(CATEGORIES):
        size = chunk + (1 if i < remainder else 0)
        bucket = shuffled[idx:idx + size]
        idx += size

        assigned = []
        for sc in bucket:
            sc_copy = dict(sc)
            sc_copy['assigned_category'] = cat
            sc_copy['assigned_flows'] = _pick_flows_for_category(sc_copy, cat, rng)
            assigned.append(sc_copy)
        buckets[cat] = assigned

    return buckets


def _pick_flows_for_category(
    scenario: dict,
    category: str,
    rng: random.Random,
) -> dict:
    """Pick specific flows for a scenario based on its category."""
    seq = scenario.get('flow_sequence', [])
    edges = scenario.get('edge_flow_pairs', [])

    if category == 'same_flow':
        # Pick one flow, use for both turns
        flow_obj = rng.choice(seq)
        return {
            'turn1_flow': flow_obj['flow'],
            'turn1_intent': flow_obj['intent'],
            'turn3_flow': flow_obj['flow'],
            'turn3_intent': flow_obj['intent'],
        }

    elif category == 'switch_flow':
        # Pick two different flows
        if len(seq) >= 2:
            pair = rng.sample(seq, 2)
        else:
            pair = [seq[0], seq[0]]  # fallback, shouldn't happen
        return {
            'turn1_flow': pair[0]['flow'],
            'turn1_intent': pair[0]['intent'],
            'turn3_flow': pair[1]['flow'],
            'turn3_intent': pair[1]['intent'],
        }

    elif category == 'ambiguous_first':
        # Use edge_flow_pair for candidate flows
        if edges:
            edge = rng.choice(edges)
            cand_a, cand_b = edge[0], edge[1]
        else:
            pair = rng.sample(seq, min(2, len(seq)))
            cand_a = pair[0]['flow']
            cand_b = pair[1]['flow'] if len(pair) > 1 else pair[0]['flow']

        # Find intents for the candidate flows
        flow_intent_map = {fs['flow']: fs['intent'] for fs in seq}
        intent_a = flow_intent_map.get(cand_a, 'unknown')
        intent_b = flow_intent_map.get(cand_b, 'unknown')

        # Resolve turn 3 to one of the candidates
        resolved = rng.choice([cand_a, cand_b])
        resolved_intent = flow_intent_map.get(resolved, intent_a)

        # Decide sub-type: partial or confirmation (50/50)
        sub_type = rng.choice(['partial', 'confirmation'])

        return {
            'candidate_flows': [cand_a, cand_b],
            'candidate_intents': list(set([intent_a, intent_b, 'Plan'])),
            'turn3_flow': resolved,
            'turn3_intent': resolved_intent,
            'sub_type': sub_type,
        }

    elif category == 'ambiguous_second':
        # Turn 1: clear flow; Turn 3: multi-request → Plan orchestrator
        if len(seq) >= 3:
            # Pick turn 1 flow and two flows for the multi-request
            turn1 = seq[0]
            multi = rng.sample(seq[1:], 2)
        elif len(seq) >= 2:
            turn1 = seq[0]
            multi = [seq[1], seq[0]]
        else:
            turn1 = seq[0]
            multi = [seq[0], seq[0]]

        flow_intent_map = {fs['flow']: fs['intent'] for fs in seq}

        return {
            'turn1_flow': turn1['flow'],
            'turn1_intent': turn1['intent'],
            'candidate_flows': [m['flow'] for m in multi],
            'candidate_intents': list(set(
                [m['intent'] for m in multi] + ['Plan']
            )),
        }

    return {}


# ── Prompt construction ──────────────────────────────────────────────

def _build_system_prompt(
    domain: str,
    flow_mapping_text: str,
    tool_manifest_compact: str,
) -> str:
    """Build the system prompt with full domain context."""
    return f"""You are an expert conversation designer for the {domain.upper()} copilot.

## Domain
{DOMAIN_DESCRIPTIONS[domain]}

## All Flows and Their Tools
{flow_mapping_text}

## Tool Reference (name, description, parameters)
{tool_manifest_compact}

## Output Format
You MUST output ONLY a single valid JSON object (no markdown fencing, no explanation).
The JSON must conform exactly to the schema described in the user prompt.

## Quality Rules
- Utterances must sound like a real person talking to an AI assistant — casual, varied, natural.
- NEVER include the flow name, intent name, or tool name in the user utterance.
- Turn 2 (agent) should be 1-2 sentences, directly responding to turn 1.
- The context field must be realistic and specific to the scenario.
- target_tools must use actual tool names and realistic parameter values from the tool reference above.
- For parameters where the exact value depends on data the user hasn't provided, use null.
"""


def _build_user_prompt_same_flow(
    scenario: dict,
    domain: str,
    convo_id: str,
) -> str:
    af = scenario['assigned_flows']
    return f"""Generate a 3-turn conversation for category "same_flow".

## Rules for same_flow
- Turn 1 and Turn 3 must BOTH use the flow "{af['turn1_flow']}" (intent: {af['turn1_intent']}).
- Turn 3 should be a natural follow-up or continuation — NOT a repetition of turn 1.
- The agent (turn 2) responds to turn 1 helpfully, setting up the follow-up.

## Scenario
{scenario['scenario']}

Example utterances for inspiration (DO NOT copy these verbatim):
{json.dumps(scenario.get('example_utterances', []))}

## Required JSON Schema
{{
  "convo_id": "{convo_id}",
  "category": "same_flow",
  "scenario": "<short scenario description, like: 'E-commerce sales -- Q4 data with 50K orders'>",
  "context": {CONTEXT_SCHEMA[domain]},
  "turns": [
    {{
      "turn_num": 1,
      "flow": "{af['turn1_flow']}",
      "intent": "{af['turn1_intent']}",
      "speaker": "user",
      "utterance": "<natural user message>",
      "target_tools": {{
        "<tool_name>": {{ <realistic params> }}
      }}
    }},
    {{
      "turn_num": 2,
      "speaker": "agent",
      "utterance": "<1-2 sentence agent response>"
    }},
    {{
      "turn_num": 3,
      "flow": "{af['turn3_flow']}",
      "intent": "{af['turn3_intent']}",
      "speaker": "user",
      "utterance": "<natural follow-up in the SAME flow>",
      "target_tools": {{
        "<tool_name>": {{ <realistic params> }}
      }}
    }}
  ]
}}

Output ONLY the JSON object. No markdown, no commentary."""


def _build_user_prompt_switch_flow(
    scenario: dict,
    domain: str,
    convo_id: str,
) -> str:
    af = scenario['assigned_flows']
    return f"""Generate a 3-turn conversation for category "switch_flow".

## Rules for switch_flow
- Turn 1 uses flow "{af['turn1_flow']}" (intent: {af['turn1_intent']}).
- Turn 3 SWITCHES to a different flow: "{af['turn3_flow']}" (intent: {af['turn3_intent']}).
- The switch should feel natural — the user finishes one task and moves to another.
- The agent (turn 2) responds to turn 1, then the user pivots.

## Scenario
{scenario['scenario']}

Example utterances for inspiration (DO NOT copy these verbatim):
{json.dumps(scenario.get('example_utterances', []))}

## Required JSON Schema
{{
  "convo_id": "{convo_id}",
  "category": "switch_flow",
  "scenario": "<short scenario description>",
  "context": {CONTEXT_SCHEMA[domain]},
  "turns": [
    {{
      "turn_num": 1,
      "flow": "{af['turn1_flow']}",
      "intent": "{af['turn1_intent']}",
      "speaker": "user",
      "utterance": "<natural user message for {af['turn1_flow']}>",
      "target_tools": {{
        "<tool_name>": {{ <realistic params> }}
      }}
    }},
    {{
      "turn_num": 2,
      "speaker": "agent",
      "utterance": "<1-2 sentence agent response>"
    }},
    {{
      "turn_num": 3,
      "flow": "{af['turn3_flow']}",
      "intent": "{af['turn3_intent']}",
      "speaker": "user",
      "utterance": "<natural user message switching to {af['turn3_flow']}>",
      "target_tools": {{
        "<tool_name>": {{ <realistic params> }}
      }}
    }}
  ]
}}

Output ONLY the JSON object. No markdown, no commentary."""


def _build_user_prompt_ambiguous_first(
    scenario: dict,
    domain: str,
    convo_id: str,
) -> str:
    af = scenario['assigned_flows']
    sub_type = af.get('sub_type', 'partial')

    if sub_type == 'partial':
        sub_desc = (
            'This is a "partial" ambiguity: the user\'s intent/preference is unclear '
            'and cannot be resolved by inspecting the data. The agent must ask a '
            'clarifying question to distinguish between the candidate flows.'
        )
        agent_behavior = 'Ask a clarifying question to distinguish between the two possible actions.'
    else:
        sub_desc = (
            'This is a "confirmation" ambiguity: the agent CAN partially resolve '
            'the ambiguity by inspecting the data/context, but needs user confirmation. '
            'The agent proposes an action and asks the user to confirm.'
        )
        agent_behavior = (
            'Inspect the data/context, propose a specific action based on what you find, '
            'and ask the user to confirm.'
        )

    return f"""Generate a 3-turn conversation for category "ambiguous_first".

## Rules for ambiguous_first
- Turn 1 is AMBIGUOUS — the utterance could reasonably map to either of these flows:
  candidate_flows: {json.dumps(af['candidate_flows'])}
- {sub_desc}
- Turn 2: agent responds. {agent_behavior}
- Turn 3: user CLARIFIES, resolving to flow "{af['turn3_flow']}" (intent: {af['turn3_intent']}).

## Scenario
{scenario['scenario']}

Example utterances for inspiration (DO NOT copy these verbatim):
{json.dumps(scenario.get('example_utterances', []))}

## Required JSON Schema
{{
  "convo_id": "{convo_id}",
  "category": "ambiguous_first",
  "scenario": "<short scenario description>",
  "context": {CONTEXT_SCHEMA[domain]},
  "turns": [
    {{
      "turn_num": 1,
      "flow": "ambiguous",
      "candidate_flows": {json.dumps(af['candidate_flows'])},
      "candidate_intents": {json.dumps(af['candidate_intents'])},
      "speaker": "user",
      "utterance": "<genuinely ambiguous user message>",
      "rationale": "<1-2 sentences explaining WHY this is ambiguous between the candidate flows>",
      "target_tools": {{
        "handle_ambiguity": {{
          "clarification": null,
          "candidates": ["<tool_from_flow_A>", "<tool_from_flow_B>"]
        }},
        "<tool_for_flow_A>": {{ <params> }},
        "<tool_for_flow_B>": {{ <params> }}
      }}
    }},
    {{
      "turn_num": 2,
      "speaker": "agent",
      "utterance": "<clarifying question or proposal with confirmation request>"
    }},
    {{
      "turn_num": 3,
      "flow": "{af['turn3_flow']}",
      "intent": "{af['turn3_intent']}",
      "speaker": "user",
      "utterance": "<user clarifies, resolving to {af['turn3_flow']}>",
      "candidate_intents": {json.dumps(af['candidate_intents'])},
      "target_tools": {{
        "<tool_for_resolved_flow>": {{ <params> }}
      }}
    }}
  ]
}}

Output ONLY the JSON object. No markdown, no commentary."""


def _build_user_prompt_ambiguous_second(
    scenario: dict,
    domain: str,
    convo_id: str,
) -> str:
    af = scenario['assigned_flows']
    plan_flow = PLAN_ORCHESTRATOR[domain]

    return f"""Generate a 3-turn conversation for category "ambiguous_second".

## Rules for ambiguous_second
- Turn 1 is CLEAR — uses flow "{af['turn1_flow']}" (intent: {af['turn1_intent']}).
- Turn 2: agent responds to turn 1 normally.
- Turn 3 is a MULTI-REQUEST — the user asks for TWO things at once that require different flows:
  candidate_flows: {json.dumps(af['candidate_flows'])}
  This routes to the Plan orchestrator flow "{plan_flow}" (intent: Plan).
- The multi-request should feel natural — the user combines two related actions in one sentence.

## Scenario
{scenario['scenario']}

Example utterances for inspiration (DO NOT copy these verbatim):
{json.dumps(scenario.get('example_utterances', []))}

## Required JSON Schema
{{
  "convo_id": "{convo_id}",
  "category": "ambiguous_second",
  "scenario": "<short scenario description>",
  "context": {CONTEXT_SCHEMA[domain]},
  "turns": [
    {{
      "turn_num": 1,
      "flow": "{af['turn1_flow']}",
      "intent": "{af['turn1_intent']}",
      "speaker": "user",
      "utterance": "<clear user message for {af['turn1_flow']}>",
      "target_tools": {{
        "<tool_name>": {{ <params> }}
      }}
    }},
    {{
      "turn_num": 2,
      "speaker": "agent",
      "utterance": "<1-2 sentence agent response>"
    }},
    {{
      "turn_num": 3,
      "flow": "{plan_flow}",
      "intent": "Plan",
      "candidate_flows": {json.dumps(af['candidate_flows'])},
      "candidate_intents": {json.dumps(af['candidate_intents'])},
      "speaker": "user",
      "utterance": "<natural multi-request combining both flows in one sentence>",
      "rationale": "<1-2 sentences explaining why this is a multi-request>",
      "target_tools": {{
        "<tool_for_flow_1>": {{ <params> }},
        "<tool_for_flow_2>": {{ <params> }}
      }}
    }}
  ]
}}

Output ONLY the JSON object. No markdown, no commentary."""


PROMPT_BUILDERS = {
    'same_flow': _build_user_prompt_same_flow,
    'switch_flow': _build_user_prompt_switch_flow,
    'ambiguous_first': _build_user_prompt_ambiguous_first,
    'ambiguous_second': _build_user_prompt_ambiguous_second,
}


# ── LLM calling ─────────────────────────────────────────────────────

async def _call_anthropic_async(system_prompt: str, user_prompt: str, model_id: str) -> str:
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
    try:
        resp = await client.messages.create(
            model=model_id,
            max_tokens=4096,
            temperature=0.8,
            system=system_prompt,
            messages=[{'role': 'user', 'content': user_prompt}],
        )
        return next(b.text for b in resp.content if hasattr(b, 'text'))
    finally:
        await client.close()


async def _call_openai_async(system_prompt: str, user_prompt: str, model_id: str) -> str:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
    try:
        resp = await client.chat.completions.create(
            model=model_id,
            max_completion_tokens=4096,
            temperature=0.8,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
        )
        return resp.choices[0].message.content or ''
    finally:
        await client.close()


async def _call_openrouter_async(system_prompt: str, user_prompt: str, model_id: str) -> str:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        base_url='https://openrouter.ai/api/v1',
        api_key=os.environ['OPEN_ROUTER_API_KEY'],
    )
    try:
        resp = await client.chat.completions.create(
            model=model_id,
            max_completion_tokens=4096,
            temperature=0.8,
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


async def _call_model_async(
    system_prompt: str,
    user_prompt: str,
    model_config: dict,
) -> str:
    """Call a model with retry logic."""
    provider = model_config['provider']
    model_id = model_config['model_id']

    for attempt in range(MAX_RETRIES):
        try:
            if provider == 'anthropic':
                return await _call_anthropic_async(system_prompt, user_prompt, model_id)
            elif provider == 'openai':
                return await _call_openai_async(system_prompt, user_prompt, model_id)
            elif provider == 'openrouter':
                return await _call_openrouter_async(system_prompt, user_prompt, model_id)
            else:
                raise ValueError(f'Unknown provider: {provider}')
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = BACKOFF_BASE_S * (2 ** attempt)
                log.warning(
                    'Attempt %d failed for %s: %s. Retrying in %.1fs',
                    attempt + 1, model_config['name'], e, wait,
                )
                await asyncio.sleep(wait)
            else:
                raise


# ── Response parsing ─────────────────────────────────────────────────

def _parse_conversation(raw: str) -> dict | None:
    """Extract and validate conversation JSON from LLM response."""
    # Strip markdown fencing if present
    text = raw.strip()
    if text.startswith('```'):
        # Remove first line (```json or ```)
        lines = text.split('\n')
        lines = lines[1:]
        # Remove last ``` line
        if lines and lines[-1].strip().startswith('```'):
            lines = lines[:-1]
        text = '\n'.join(lines)

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                obj = json.loads(match.group())
            except json.JSONDecodeError:
                log.warning('Failed to parse JSON from response')
                return None
        else:
            log.warning('No JSON object found in response')
            return None

    # Validate basic structure
    if not isinstance(obj, dict):
        return None
    if 'turns' not in obj or not isinstance(obj['turns'], list):
        return None
    if len(obj['turns']) != 3:
        log.warning('Expected 3 turns, got %d', len(obj['turns']))
        return None

    # Validate turn structure
    for turn in obj['turns']:
        if 'turn_num' not in turn or 'speaker' not in turn or 'utterance' not in turn:
            return None

    return obj


# ── Wave-based async orchestration ───────────────────────────────────

async def _run_single(
    spec: dict,
    system_prompt: str,
    semaphore: asyncio.Semaphore,
) -> str | BaseException:
    async with semaphore:
        try:
            return await _call_model_async(
                system_prompt,
                spec['user_prompt'],
                spec['model_config'],
            )
        except Exception as e:
            return e


async def _run_wave(
    specs: list[dict],
    system_prompt: str,
    semaphore: asyncio.Semaphore,
) -> list[str | BaseException]:
    tasks = [_run_single(spec, system_prompt, semaphore) for spec in specs]
    return await asyncio.gather(*tasks, return_exceptions=True)


# ── Output finalization ──────────────────────────────────────────────

def _finalize_output(domain: str, raw_jsonl_path: Path, output_json_path: Path) -> None:
    """Convert JSONL to a sorted JSON array matching eval_set.json format."""
    convos = []
    with open(raw_jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                convos.append(json.loads(line))
    cat_order = {'same_flow': 0, 'switch_flow': 1, 'ambiguous_first': 2, 'ambiguous_second': 3}
    convos.sort(key=lambda c: (cat_order.get(c.get('category', ''), 99), c.get('convo_id', '')))
    with open(output_json_path, 'w') as f:
        json.dump(convos, f, indent=2, ensure_ascii=False)
    log.info('Finalized output: %d conversations -> %s', len(convos), output_json_path)


# ── Main orchestrator ────────────────────────────────────────────────

def generate_conversations(
    domain: str,
    seed: int = 42,
    models_filter: list[str] | None = None,
    dry_run: bool = False,
    verbose: bool = False,
    max_threads: int | None = None,
) -> Path:
    """Main orchestrator: assign categories, generate conversations, write output.

    Returns path to the output JSONL file.
    """
    rng = random.Random(seed)

    # Filter models
    active_models = MODEL_CONFIGS[:]
    if models_filter:
        active_models = [m for m in active_models if m['name'] in models_filter]
        if not active_models:
            raise ValueError(f'No models match filter: {models_filter}')

    if max_threads is None:
        max_threads = len(active_models)

    # Load data
    scenarios = _load_scenarios(domain)
    flow_mapping_text = _load_flow_tool_mapping(domain)
    tool_manifest = _load_tool_manifest(domain)
    manifest_compact = _tool_manifest_compact(tool_manifest)

    log.info('Loaded %d scenarios for %s', len(scenarios), domain)

    # Output paths
    output_jsonl = _SCRIPT_DIR / f'conversations_{domain}_raw.jsonl'
    output_json = _SCRIPT_DIR / f'conversations_{domain}.json'
    output_meta = _SCRIPT_DIR / f'conversations_{domain}_meta.json'

    # Resume support: load existing conversations
    existing_ids: set[str] = set()
    if output_jsonl.exists():
        with open(output_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    existing_ids.add(obj.get('convo_id', ''))
                except json.JSONDecodeError:
                    continue
        log.info('Resuming: %d conversations already exist', len(existing_ids))

    # Assign categories
    buckets = assign_categories(scenarios, rng)

    # Build work items: assign models round-robin within each category
    work_items: list[dict] = []
    convo_counter = 1

    for cat in CATEGORIES:
        cat_scenarios = buckets[cat]
        for i, sc in enumerate(cat_scenarios):
            convo_id = f'{domain}_{convo_counter:03d}'
            convo_counter += 1

            if convo_id in existing_ids:
                continue

            model_config = active_models[i % len(active_models)]

            prompt_builder = PROMPT_BUILDERS[cat]
            user_prompt = prompt_builder(sc, domain, convo_id)

            work_items.append({
                'convo_id': convo_id,
                'category': cat,
                'scenario': sc,
                'model_config': model_config,
                'user_prompt': user_prompt,
            })

    if not work_items:
        log.info('All conversations already generated')
        if output_jsonl.exists():
            _finalize_output(domain, output_jsonl, output_json)
        return output_jsonl

    log.info(
        'Generating %d conversations for %s (%d existing, %d models)',
        len(work_items), domain, len(existing_ids), len(active_models),
    )

    # Build system prompt
    system_prompt = _build_system_prompt(domain, flow_mapping_text, manifest_compact)

    if dry_run:
        for wi in work_items[:4]:
            print(f'\n{"="*70}')
            print(f'CONVO: {wi["convo_id"]} | Category: {wi["category"]} | Model: {wi["model_config"]["name"]}')
            print(f'{"="*70}')
            print(f'\n--- SYSTEM PROMPT (first 500 chars) ---\n{system_prompt[:500]}...')
            print(f'\n--- USER PROMPT ---\n{wi["user_prompt"][:1500]}...')
        print(f'\n[DRY RUN] Would generate {len(work_items)} conversations')
        return output_jsonl

    # Wave-based generation
    pbar = tqdm(total=len(work_items), unit='convos', desc='Generating')
    wave_num = 0
    cursor = 0
    failed = 0
    generated = 0
    failed_items: list[dict] = []

    while cursor < len(work_items):
        wave_specs = work_items[cursor:cursor + max_threads]
        cursor += len(wave_specs)

        semaphore = asyncio.Semaphore(max_threads)
        results = asyncio.run(_run_wave(wave_specs, system_prompt, semaphore))

        wave_new: list[dict] = []
        for spec, result in zip(wave_specs, results):
            if isinstance(result, BaseException):
                pbar.write(f'ERROR: {spec["convo_id"]} ({spec["model_config"]["name"]}): {result}')
                failed += 1
                failed_items.append(spec)
                continue

            convo = _parse_conversation(result)
            if convo is None:
                pbar.write(f'WARNING: {spec["convo_id"]}: failed to parse response')
                failed += 1
                failed_items.append(spec)
                continue

            # Ensure correct metadata
            convo['convo_id'] = spec['convo_id']
            convo['category'] = spec['category']
            convo['_model'] = spec['model_config']['model_id']
            convo['_provider'] = spec['model_config']['provider']

            wave_new.append(convo)

        # Flush to disk
        if wave_new:
            with open(output_jsonl, 'a') as f:
                for obj in wave_new:
                    f.write(json.dumps(obj, ensure_ascii=False) + '\n')
            generated += len(wave_new)
            pbar.update(len(wave_new))

        wave_num += 1
        pbar.set_postfix_str(f'wave {wave_num}, ok={generated}, fail={failed}')

    pbar.close()

    # Retry failed conversations (up to 2 retry passes)
    for retry_pass in range(2):
        if not failed_items:
            break

        log.info('Retry pass %d: %d failed conversations', retry_pass + 1, len(failed_items))
        retry_failed: list[dict] = []

        pbar = tqdm(total=len(failed_items), unit='convos', desc=f'Retry {retry_pass + 1}')
        retry_cursor = 0

        while retry_cursor < len(failed_items):
            wave_specs = failed_items[retry_cursor:retry_cursor + max_threads]
            retry_cursor += len(wave_specs)

            semaphore = asyncio.Semaphore(max_threads)
            results = asyncio.run(_run_wave(wave_specs, system_prompt, semaphore))

            wave_new = []
            for spec, result in zip(wave_specs, results):
                if isinstance(result, BaseException):
                    pbar.write(f'RETRY ERROR: {spec["convo_id"]} ({spec["model_config"]["name"]}): {result}')
                    retry_failed.append(spec)
                    continue

                convo = _parse_conversation(result)
                if convo is None:
                    pbar.write(f'RETRY WARNING: {spec["convo_id"]}: failed to parse response')
                    retry_failed.append(spec)
                    continue

                convo['convo_id'] = spec['convo_id']
                convo['category'] = spec['category']
                convo['_model'] = spec['model_config']['model_id']
                convo['_provider'] = spec['model_config']['provider']

                wave_new.append(convo)

            if wave_new:
                with open(output_jsonl, 'a') as f:
                    for obj in wave_new:
                        f.write(json.dumps(obj, ensure_ascii=False) + '\n')
                generated += len(wave_new)
                failed -= len(wave_new)
                pbar.update(len(wave_new))

        pbar.close()
        failed_items = retry_failed

    if failed_items:
        log.warning('%d conversations could not be generated after retries: %s',
                    len(failed_items),
                    [s['convo_id'] for s in failed_items[:20]])

    # Write meta
    meta = {
        'domain': domain,
        'seed': seed,
        'total_scenarios': len(scenarios),
        'total_generated': generated,
        'total_failed': failed,
        'categories': {cat: len(buckets[cat]) for cat in CATEGORIES},
        'models': [m['name'] for m in active_models],
    }
    with open(output_meta, 'w') as f:
        json.dump(meta, f, indent=2)

    # Finalize: convert JSONL to sorted JSON array
    if output_jsonl.exists():
        _finalize_output(domain, output_jsonl, output_json)

    log.info(
        'Done: %d conversations generated, %d failed. Output: %s',
        generated, failed, output_jsonl,
    )
    return output_jsonl


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    # Load .env
    env_path = _PROJECT_ROOT / '.env'
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)

    parser = argparse.ArgumentParser(description='Generate conversations from enriched scenarios')
    parser.add_argument('--domain', required=True, choices=['hugo', 'dana'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated model filter (e.g. "anthropic,openai")')
    parser.add_argument('--max-threads', type=int, default=None)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    log.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    models_filter = None
    if args.models:
        models_filter = [m.strip() for m in args.models.split(',')]

    generate_conversations(
        domain=args.domain,
        seed=args.seed,
        models_filter=models_filter,
        dry_run=args.dry_run,
        verbose=args.verbose,
        max_threads=args.max_threads,
    )


if __name__ == '__main__':
    main()
