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

from datasets.data_aug_pranav.compute_metrics import (
    check_conversation,
    check_leakage_llm,
)

# ── Path setup ───────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SCRIPT_DIR / "data"

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
MAX_QUALITY_RETRIES = 3
BACKOFF_BASE_S = 1.0

# Intents that are user-facing (exclude Plan, Internal)
USER_FACING_INTENTS = {
    'dana': ['Clean', 'Transform', 'Analyze', 'Report', 'Converse'],
    'hugo': ['Research', 'Draft', 'Revise', 'Publish', 'Converse'],
}


# ── Load helpers ─────────────────────────────────────────────────────

def _load_scenarios(domain: str) -> list[dict]:
    """Load enriched deduped scenarios."""
    path = _DATA_DIR / f'scenarios_{domain}_enriched_deduped.jsonl'
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

    # Build a lookup from flow name to its assigned_tools
    flow_tools_map = {fs['flow']: fs.get('assigned_tools', []) for fs in seq}

    if category == 'same_flow':
        # Pick one flow, use for both turns
        flow_obj = rng.choice(seq)
        tools = flow_obj.get('assigned_tools', [])
        return {
            'turn1_flow': flow_obj['flow'],
            'turn1_intent': flow_obj['intent'],
            'turn3_flow': flow_obj['flow'],
            'turn3_intent': flow_obj['intent'],
            'turn1_tools': tools[:1],   # first tool for turn 1
            'turn3_tools': tools[1:2],  # second tool for turn 3 (different)
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
            'turn1_tools': pair[0].get('assigned_tools', [])[:1],
            'turn3_tools': pair[1].get('assigned_tools', [])[:1],
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

        # Tool assignments for candidates
        cand_a_tools = flow_tools_map.get(cand_a, [])[:1]
        cand_b_tools = flow_tools_map.get(cand_b, [])[:1]
        resolved_tools = flow_tools_map.get(resolved, [])[1:2] or flow_tools_map.get(resolved, [])[:1]

        return {
            'candidate_flows': [cand_a, cand_b],
            'candidate_intents': list(set([intent_a, intent_b, 'Plan'])),
            'turn3_flow': resolved,
            'turn3_intent': resolved_intent,
            'sub_type': sub_type,
            'cand_a_tools': cand_a_tools,
            'cand_b_tools': cand_b_tools,
            'turn3_tools': resolved_tools,
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
            'turn1_tools': turn1.get('assigned_tools', [])[:1],
            'candidate_flows': [m['flow'] for m in multi],
            'candidate_intents': list(set(
                [m['intent'] for m in multi] + ['Plan']
            )),
            'turn3_tools': [
                multi[0].get('assigned_tools', [])[:1],
                multi[1].get('assigned_tools', [])[:1],
            ],
        }

    return {}


# ── Tool constraint helpers ───────────────────────────────────────────

def _required_tools_section(constraints: list[str]) -> str:
    """Build a ## Required Tools section from a list of constraint lines.

    Returns empty string if no constraints (e.g. all flows are toolless).
    """
    if not constraints:
        return ''
    lines = '\n'.join(f'- {c}' for c in constraints)
    return f"""
## Required Tools
{lines}
- You MAY add additional tools from the same flow, but the specified tool(s) MUST appear.
"""


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

### User utterances
- IMAGINE THE USER IS TYPING ON THEIR SMARTPHONE. Mobile users are terse, rely on shared context, skip pleasantries, and don't repeat what's on screen.
- USERS OBSERVE, THEY DON'T COMMAND. Describe what you see or what's wrong, not what the agent should do. "The ICD codes aren't matching" beats "Flag anything that doesn't match." "I think I see repeated employees?" beats "Check for duplicate employee IDs and clean those out."
- DROP IMPERATIVES. State the problem without telling the agent what to do. "Signup dates are all over the place." Period. Don't append "Standardize them."
- DON'T GIVE AWAY THE ANSWER. Even unambiguous turns shouldn't spell out the operation. Wrong: "Can you expand and develop my bullet points into fully written paragraphs?" Right: "Flesh them out into actual paragraphs."
- MORE IMPLICIT, LESS EXPLICIT. Users assume shared context. "Too smooth now, try 3 days instead" beats "Can you also perform the same rolling average smoothing operation on the revenue column using a 3-day window?"
- Turn 1: 8-20 words (avg ~14). Turn 3: 2-9 words (avg ~5). Turn 3 is EXTREMELY terse — context from turns 1-2 makes elaboration unnecessary.
- Turn 3 MUST depend on turns 1-2. Use anaphora: "Same thing for...", "That one too", "What about X instead?"
- Turn 3 can be a follow-up question: "What's the percent change?", "Which sites?"
- Use domain abbreviations: "MoM" not "month-over-month", "CTR" not "click-through rate".
- NEVER include the flow name, intent name, or tool name in the user utterance.
- Vary register: mix observations ("dates look weird"), terse commands ("fix the intro"), casual questions ("what's the word count?"), and brief follow-ups ("same thing for the conclusion").
- Avoid em dashes, fancy punctuation, and overly polished prose. Use plain commas and periods.

### Agent utterances (Turn 2)
- 1-2 sentences max. Directly responds to turn 1.
- Agent explains WHY it's taking an action: "since that's the most common format" or "Biggest jump this quarter."
- CRITICAL: Agent responds ONLY to what the user actually said. The agent does NOT know the user's downstream goal, the flow label, or what the user will ask next. Do NOT infer unstated goals.
- No over-acknowledgment: NEVER start with "Absolutely!", "Great question!", "I'd be happy to!", "Sure thing!"
- No filler: NEVER use "Just to confirm —", "To clarify,", "So what you're saying is..."

### General
- The context field must be realistic and specific to the scenario.
- target_tools MUST ONLY include the required tools specified in the Tool Constraints section below. Do NOT add extra tools beyond those listed. Use realistic parameter values from the tool reference.
- For parameters where the exact value depends on data the user hasn't provided, use null.

### Anti-patterns (NEVER do these)
- Filler openers: "Before I...", "Just to confirm —", "I was wondering if..."
- Hedging: "Can you maybe...", "Would it be possible to..."
- Over-explaining intent: "so that I can then..." / "which will help me..."
- Restating what's obvious: "Good call. Now, moving on to the next thing..."
- Padding: "right now" / "at this point" / "at the moment"
- Agent leaking label info: agent inferring user's goal that wasn't stated (e.g., user asks to "pull posts" and agent says "so we can compare wording")

## Style Examples — BAD vs GOOD (for calibration — DO NOT copy verbatim)

User turn 1:
  BAD:  "Before I cross-post the Lisbon guide, can you check if it's already live on Medium and whether the sync looks okay?"
  GOOD: "Is the Lisbon guide live on Medium yet?"
  BAD:  "Can you pull my beginner baking posts from the last 6 months?"
  GOOD: "beginner baking posts from the last 6 months"
  BAD:  "The order_date column needs to be converted from string format to a proper datetime type."
  GOOD: "The order_date is showing up as a generic string. That doesn't seem right."

User turn 3:
  BAD:  "Good. Can you pull up all my connected platforms and show me which ones have working integrations right now?"
  GOOD: "Which platforms are working?"
  BAD:  "Just publish it, we can worry about the rest later."
  GOOD: "Publish now, worry about the rest later"
  BAD:  "Just the ones tagged beginner or 101, and include anything that mentions laminated dough."
  GOOD: "beginner tag is good, also anything that mentions laminated dough"
  GOOD: "Same thing for multi-head attention."
  GOOD: "That's a terrible title, please try again."
  GOOD: "What about job satisfaction rather than department?"

Agent turn 2:
  BAD:  "Yep, I'll list your recent beginner-friendly baking drafts and published posts so we can compare wording."
  GOOD: "Are beginner posts those that contain the word '101' or those that have the tag 'beginner'?"
  BAD:  "Just to confirm — do you want to publish it live on the blog right now, or push it across your connected platforms?"
  GOOD: "Publish it live on the blog, or push it across all platforms?"
  BAD:  "Absolutely! I'll smooth out the transitions and tighten up the word choice in the intro for you."
  GOOD: "I'll tighten the intro — the transitions between paragraphs are the main issue."
"""


def _build_user_prompt_same_flow(
    scenario: dict,
    domain: str,
    convo_id: str,
) -> str:
    af = scenario['assigned_flows']
    t1_tools = af.get('turn1_tools', [])
    t3_tools = af.get('turn3_tools', [])

    # Build tool constraints
    constraints = []
    if t1_tools:
        constraints.append(f'Turn 1 MUST include `{t1_tools[0]}` in target_tools with realistic parameters.')
    if t3_tools:
        constraints.append(f'Turn 3 MUST include `{t3_tools[0]}` in target_tools with realistic parameters.')
    tool_section = _required_tools_section(constraints)

    # Concrete tool names for schema placeholders
    t1_tool_name = t1_tools[0] if t1_tools else '<tool_name>'
    t3_tool_name = t3_tools[0] if t3_tools else '<tool_name>'

    return f"""Generate a 3-turn conversation for category "same_flow".

## Rules for same_flow
- Turn 1 and Turn 3 must BOTH use the flow "{af['turn1_flow']}" (intent: {af['turn1_intent']}).
- Turn 3 should be a natural follow-up or continuation — NOT a repetition of turn 1. Turn 3 is EXTREMELY terse (2-9 words).
- The agent (turn 2) responds to turn 1 helpfully, setting up the follow-up.
- The conversation must follow ONE of these patterns:
  (a) Slot-value missing — turn 1 omits a parameter, agent asks, turn 3 provides it
  (b) User builds on previous — turn 1 acts on entity X, turn 3 does same flow on entity Y
  (c) User adjusts agent proposal — turn 1 requests, agent proposes specifics, turn 3 corrects/adjusts
- Anti-pattern: two independent same-flow requests. "Rename column A" then "Rename column B" are unrelated. Turns must have a narrative thread.

## Scenario
{scenario['scenario']}

Example utterances for inspiration (DO NOT copy these verbatim):
{json.dumps(scenario.get('example_utterances', []))}
{tool_section}
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
        "{t1_tool_name}": {{ <realistic params> }}
      }}
    }},
    {{
      "turn_num": 2,
      "speaker": "agent",
      "utterance": "<1-2 sentence response to what the user ACTUALLY said — explain WHY you're taking this action, do NOT infer unstated goals>"
    }},
    {{
      "turn_num": 3,
      "flow": "{af['turn3_flow']}",
      "intent": "{af['turn3_intent']}",
      "speaker": "user",
      "utterance": "<natural follow-up in the SAME flow>",
      "target_tools": {{
        "{t3_tool_name}": {{ <realistic params> }}
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
    t1_tools = af.get('turn1_tools', [])
    t3_tools = af.get('turn3_tools', [])

    constraints = []
    if t1_tools:
        constraints.append(f'Turn 1 MUST include `{t1_tools[0]}` in target_tools with realistic parameters.')
    if t3_tools:
        constraints.append(f'Turn 3 MUST include `{t3_tools[0]}` in target_tools with realistic parameters.')
    tool_section = _required_tools_section(constraints)

    t1_tool_name = t1_tools[0] if t1_tools else '<tool_name>'
    t3_tool_name = t3_tools[0] if t3_tools else '<tool_name>'

    return f"""Generate a 3-turn conversation for category "switch_flow".

## Rules for switch_flow
- Turn 1 uses flow "{af['turn1_flow']}" (intent: {af['turn1_intent']}).
- Turn 3 SWITCHES to a different flow: "{af['turn3_flow']}" (intent: {af['turn3_intent']}).
- The switch should feel natural — the user finishes one task and moves to another.
- The agent (turn 2) responds to turn 1, then the user pivots.
- The switch should emerge naturally from the conversation:
  - brainstorm -> endorse: ask for ideas / agent suggests / "Let's go with that"
  - tone -> preference: request style change / agent adjusts / "Remember that going forward"
  - explain -> browse: "What'd you just do?" / agent explains / "What else have I got?"

## Scenario
{scenario['scenario']}

Example utterances for inspiration (DO NOT copy these verbatim):
{json.dumps(scenario.get('example_utterances', []))}
{tool_section}
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
        "{t1_tool_name}": {{ <realistic params> }}
      }}
    }},
    {{
      "turn_num": 2,
      "speaker": "agent",
      "utterance": "<1-2 sentence response to what the user ACTUALLY said — explain WHY you're taking this action, do NOT infer unstated goals>"
    }},
    {{
      "turn_num": 3,
      "flow": "{af['turn3_flow']}",
      "intent": "{af['turn3_intent']}",
      "speaker": "user",
      "utterance": "<natural user message switching to {af['turn3_flow']}>",
      "target_tools": {{
        "{t3_tool_name}": {{ <realistic params> }}
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

    cand_a_tools = af.get('cand_a_tools', [])
    cand_b_tools = af.get('cand_b_tools', [])
    t3_tools = af.get('turn3_tools', [])

    constraints = []
    if cand_a_tools:
        constraints.append(f'Turn 1 MUST include `{cand_a_tools[0]}` (from flow {af["candidate_flows"][0]}) in target_tools.')
    if cand_b_tools:
        constraints.append(f'Turn 1 MUST include `{cand_b_tools[0]}` (from flow {af["candidate_flows"][1]}) in target_tools.')
    if t3_tools:
        constraints.append(f'Turn 3 MUST include `{t3_tools[0]}` in target_tools with realistic parameters.')
    tool_section = _required_tools_section(constraints)

    tool_a_name = cand_a_tools[0] if cand_a_tools else '<tool_from_flow_A>'
    tool_b_name = cand_b_tools[0] if cand_b_tools else '<tool_from_flow_B>'
    t3_tool_name = t3_tools[0] if t3_tools else '<tool_for_resolved_flow>'

    return f"""Generate a 3-turn conversation for category "ambiguous_first".

## Rules for ambiguous_first
- Turn 1 is AMBIGUOUS — the utterance could reasonably map to either of these flows:
  candidate_flows: {json.dumps(af['candidate_flows'])}
- {sub_desc}
- Turn 2: agent responds. {agent_behavior}
- Turn 3: user CLARIFIES, resolving to flow "{af['turn3_flow']}" (intent: {af['turn3_intent']}). Clarifications can be terse ("The second one") or more detailed.

## Scenario
{scenario['scenario']}

Example utterances for inspiration (DO NOT copy these verbatim):
{json.dumps(scenario.get('example_utterances', []))}
{tool_section}
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
          "candidates": ["{tool_a_name}", "{tool_b_name}"]
        }},
        "{tool_a_name}": {{ <params> }},
        "{tool_b_name}": {{ <params> }}
      }}
    }},
    {{
      "turn_num": 2,
      "speaker": "agent",
      "utterance": "<clarifying question or proposal — respond ONLY to what the user said, do NOT infer unstated goals>"
    }},
    {{
      "turn_num": 3,
      "flow": "{af['turn3_flow']}",
      "intent": "{af['turn3_intent']}",
      "speaker": "user",
      "utterance": "<user clarifies, resolving to {af['turn3_flow']}>",
      "candidate_intents": {json.dumps(af['candidate_intents'])},
      "target_tools": {{
        "{t3_tool_name}": {{ <params> }}
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

    t1_tools = af.get('turn1_tools', [])
    # turn3_tools is [[tool_a], [tool_b]] for multi-request
    t3_tools_raw = af.get('turn3_tools', [[], []])
    t3_tool_a = t3_tools_raw[0][0] if t3_tools_raw and t3_tools_raw[0] else None
    t3_tool_b = t3_tools_raw[1][0] if len(t3_tools_raw) > 1 and t3_tools_raw[1] else None

    constraints = []
    if t1_tools:
        constraints.append(f'Turn 1 MUST include `{t1_tools[0]}` in target_tools with realistic parameters.')
    if t3_tool_a:
        constraints.append(f'Turn 3 MUST include `{t3_tool_a}` (from flow {af["candidate_flows"][0]}) in target_tools.')
    if t3_tool_b:
        constraints.append(f'Turn 3 MUST include `{t3_tool_b}` (from flow {af["candidate_flows"][1]}) in target_tools.')
    tool_section = _required_tools_section(constraints)

    t1_tool_name = t1_tools[0] if t1_tools else '<tool_name>'
    t3_tool_a_name = t3_tool_a or '<tool_for_flow_1>'
    t3_tool_b_name = t3_tool_b or '<tool_for_flow_2>'

    return f"""Generate a 3-turn conversation for category "ambiguous_second".

## Rules for ambiguous_second
- Turn 1 is CLEAR — uses flow "{af['turn1_flow']}" (intent: {af['turn1_intent']}).
- Turn 2: agent responds to turn 1 normally.
- Turn 3 is a MULTI-REQUEST — the user asks for TWO things at once that require different flows:
  candidate_flows: {json.dumps(af['candidate_flows'])}
  This routes to the Plan orchestrator flow "{plan_flow}" (intent: Plan).
- The multi-request should feel natural — the user combines two related actions in one sentence.
- The multi-request MUST use a natural connector, NOT bare "and":
  - Causal: "so I can", "so we can"
  - Prerequisite: "before", "I need X before Y"
  - Sequential: "and then" (ordered steps)
  - Embedded qualifier: second operation as constraint on first
  - Implicit: goal implied, never named ("send it off Friday")
  - Two sentences with period between
  - Question + request: "Also rename X. What do Y look like?"

## Scenario
{scenario['scenario']}

Example utterances for inspiration (DO NOT copy these verbatim):
{json.dumps(scenario.get('example_utterances', []))}
{tool_section}
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
        "{t1_tool_name}": {{ <params> }}
      }}
    }},
    {{
      "turn_num": 2,
      "speaker": "agent",
      "utterance": "<1-2 sentence response to what the user ACTUALLY said — explain WHY you're taking this action, do NOT infer unstated goals>"
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
        "{t3_tool_a_name}": {{ <params> }},
        "{t3_tool_b_name}": {{ <params> }}
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


async def _run_leakage_checks(
    candidates: list[tuple[dict, dict]],
) -> list:
    """Run LLM leakage judge in parallel for a batch of (spec, convo) pairs."""
    async def _check_one(spec, convo):
        return await check_leakage_llm(
            convo,
            scenario=spec.get('scenario', {}),
            assigned_flows=spec.get('scenario', {}).get('assigned_flows', {}),
        )
    tasks = [_check_one(spec, convo) for spec, convo in candidates]
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
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_jsonl = _DATA_DIR / f'conversations_{domain}_raw.jsonl'
    output_json = _DATA_DIR / f'conversations_{domain}.json'
    output_meta = _DATA_DIR / f'conversations_{domain}_meta.json'

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
    quality_retries_total = 0
    quality_warnings_count = 0
    failed_items: list[dict] = []

    while cursor < len(work_items):
        wave_specs = work_items[cursor:cursor + max_threads]
        cursor += len(wave_specs)

        semaphore = asyncio.Semaphore(max_threads)
        results = asyncio.run(_run_wave(wave_specs, system_prompt, semaphore))

        wave_new: list[dict] = []
        # Candidates that passed regex checks, pending leakage LLM judge
        leakage_candidates: list[tuple[dict, dict]] = []  # (spec, convo)

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

            # Regex quality gate (fast)
            qr = check_conversation(convo, spec['category'])
            if not qr.passed:
                retries_so_far = spec.get('_quality_retries', 0)
                if retries_so_far < MAX_QUALITY_RETRIES:
                    spec['_quality_retries'] = retries_so_far + 1
                    quality_retries_total += 1
                    pbar.write(f'QUALITY: {spec["convo_id"]} retry {retries_so_far + 1}/{MAX_QUALITY_RETRIES}: {qr.reasons}')
                    failed_items.append(spec)
                    continue
                else:
                    pbar.write(f'QUALITY: {spec["convo_id"]} accepted with warnings: {qr.reasons}')
                    convo['_quality_warnings'] = qr.reasons
                    quality_warnings_count += 1

            # Ensure correct metadata
            convo['convo_id'] = spec['convo_id']
            convo['category'] = spec['category']
            convo['_model'] = spec['model_config']['model_id']
            convo['_provider'] = spec['model_config']['provider']
            convo['_assigned_tools'] = spec['scenario'].get('assigned_flows', {})

            leakage_candidates.append((spec, convo))

        # LLM leakage judge (async, batched)
        if leakage_candidates:
            leakage_results = asyncio.run(_run_leakage_checks(leakage_candidates))
            for (spec, convo), lr in zip(leakage_candidates, leakage_results):
                if isinstance(lr, BaseException):
                    pbar.write(f'LEAKAGE CHECK ERROR: {spec["convo_id"]}: {lr}')
                    wave_new.append(convo)  # accept on judge error
                elif not lr.passed:
                    retries_so_far = spec.get('_quality_retries', 0)
                    if retries_so_far < MAX_QUALITY_RETRIES:
                        spec['_quality_retries'] = retries_so_far + 1
                        quality_retries_total += 1
                        pbar.write(f'LEAKAGE: {spec["convo_id"]} retry {retries_so_far + 1}/{MAX_QUALITY_RETRIES}: {lr.reasons}')
                        failed_items.append(spec)
                    else:
                        pbar.write(f'LEAKAGE: {spec["convo_id"]} accepted with warnings: {lr.reasons}')
                        convo.setdefault('_quality_warnings', []).extend(lr.reasons)
                        quality_warnings_count += 1
                        wave_new.append(convo)
                else:
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
            leakage_candidates = []

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

                # Regex quality gate (fast)
                qr = check_conversation(convo, spec['category'])
                if not qr.passed:
                    retries_so_far = spec.get('_quality_retries', 0)
                    if retries_so_far < MAX_QUALITY_RETRIES:
                        spec['_quality_retries'] = retries_so_far + 1
                        quality_retries_total += 1
                        pbar.write(f'QUALITY: {spec["convo_id"]} retry {retries_so_far + 1}/{MAX_QUALITY_RETRIES}: {qr.reasons}')
                        retry_failed.append(spec)
                        continue
                    else:
                        pbar.write(f'QUALITY: {spec["convo_id"]} accepted with warnings: {qr.reasons}')
                        convo['_quality_warnings'] = qr.reasons
                        quality_warnings_count += 1

                convo['convo_id'] = spec['convo_id']
                convo['category'] = spec['category']
                convo['_model'] = spec['model_config']['model_id']
                convo['_provider'] = spec['model_config']['provider']
                convo['_assigned_tools'] = spec['scenario'].get('assigned_flows', {})

                leakage_candidates.append((spec, convo))

            # LLM leakage judge (async, batched)
            if leakage_candidates:
                leakage_results = asyncio.run(_run_leakage_checks(leakage_candidates))
                for (spec, convo), lr in zip(leakage_candidates, leakage_results):
                    if isinstance(lr, BaseException):
                        pbar.write(f'LEAKAGE CHECK ERROR: {spec["convo_id"]}: {lr}')
                        wave_new.append(convo)
                    elif not lr.passed:
                        retries_so_far = spec.get('_quality_retries', 0)
                        if retries_so_far < MAX_QUALITY_RETRIES:
                            spec['_quality_retries'] = retries_so_far + 1
                            quality_retries_total += 1
                            pbar.write(f'LEAKAGE: {spec["convo_id"]} retry {retries_so_far + 1}/{MAX_QUALITY_RETRIES}: {lr.reasons}')
                            retry_failed.append(spec)
                        else:
                            pbar.write(f'LEAKAGE: {spec["convo_id"]} accepted with warnings: {lr.reasons}')
                            convo.setdefault('_quality_warnings', []).extend(lr.reasons)
                            quality_warnings_count += 1
                            wave_new.append(convo)
                    else:
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
        'quality_retries_total': quality_retries_total,
        'quality_warnings_count': quality_warnings_count,
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
