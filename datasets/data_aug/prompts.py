"""Prompt templates for synthetic data augmentation of multi-turn conversations."""

from __future__ import annotations

DOMAIN_DESCRIPTIONS = {
    'hugo': 'a blog writing assistant that helps users research, draft, revise, and publish blog posts',
    'dana': 'a data analysis assistant that helps users clean, transform, analyze, and report on datasets',
}

DOMAIN_NAMES = {
    'hugo': 'Hugo',
    'dana': 'Dana',
}


def build_flow_catalog_section(flow_catalog: dict, Intent) -> str:
    """Format all user-facing flows grouped by intent for the system prompt."""
    groups: dict[str, list[str]] = {}

    for name, flow in flow_catalog.items():
        intent_val = flow['intent'].value if hasattr(flow['intent'], 'value') else str(flow['intent'])
        if intent_val in ('Plan', 'Internal'):
            continue
        desc = flow.get('description', '')
        line = f'  - **{name}**: {desc}'
        groups.setdefault(intent_val, []).append(line)

    parts = []
    for intent_name in sorted(groups):
        parts.append(f'### {intent_name}')
        parts.extend(groups[intent_name])
        parts.append('')

    return '\n'.join(parts).strip()


def build_system_prompt(domain: str, flow_catalog: dict, Intent) -> str:
    """Build the shared system prompt for conversation generation."""
    domain_name = DOMAIN_NAMES[domain]
    domain_desc = DOMAIN_DESCRIPTIONS[domain]
    flow_section = build_flow_catalog_section(flow_catalog, Intent)

    return f"""You are a synthetic data generator for evaluating NLU flow detection \
in a task-oriented dialogue system. You generate realistic multi-turn \
conversations between a user and an AI assistant called {domain_name}.

{domain_name} is {domain_desc}.

## Available Flows (what the assistant can do)

{flow_section}

## Rules

1. **Imagine the user is typing on their smartphone.** Messages are short \
(10-40 words, many under 15), terse, assume shared context, and skip \
pleasantries. If an utterance feels too long or formal for a phone \
keyboard, rewrite it shorter.
2. NEVER use the flow name as a word in the user's utterance. The user \
doesn't know the internal flow names.
3. Don't spell out the operation in technical terms. Avoid trivially \
obvious keyword matches. Use indirect language, context, and natural \
paraphrasing. Wrong: "Can you expand and develop my bullet points into \
fully written paragraphs?" Right: "Flesh them out into actual paragraphs."
4. Vary sentence structure, length, and register. Mix terse and casual freely.
5. The assistant's response should be brief (1-2 sentences), specific to \
what it did or found. No filler, no robotic acknowledgments like \
"Sure thing!" or "I'll get right on that!"
6. Turn 3 MUST depend on turns 1-2. Use anaphora ("Same thing for...", \
"That one too"), implicit context ("Too smooth, try 3 days"), or \
reactions to what the agent said. If turns 1 and 3 could be swapped \
or appear independently, the conversation fails.
7. Avoid em dashes, fancy Unicode punctuation, and overly polished prose. \
Use commas, periods, and plain language. These are chat messages, not essays.
8. Return valid JSON only. No markdown fences, no explanation outside the JSON."""


# -- Per-Category User Prompts ------------------------------------------------

def _flow_spec_block(flow_info: dict) -> str:
    """Format a flow's details for the prompt."""
    return (
        f'Flow: {flow_info["name"]} ({flow_info["intent"]}): {flow_info["description"]}\n'
        f'DAX: {flow_info["dax"]} = {flow_info["dax_decomposed"]}'
    )


def build_category_a_prompt(
    sample: dict,
    persona_hint: str | None = None,
) -> str:
    """Build user prompt for Category A (same flow, turn 1 and turn 3)."""
    flow = sample['flow']
    scenario = sample['scenario']

    persona_line = f'\nPersona: {persona_hint}\n' if persona_hint else ''

    return f"""## Specification

Category: same_flow
Scenario: {scenario}
{_flow_spec_block(flow)}
{persona_line}
Generate a 3-turn conversation where both user turns map to "{flow['name']}".

The conversation must follow ONE of these patterns:

(a) Slot missing: Turn 1 omits a required parameter. Agent asks which one. \
Turn 3 provides it.
Example: "Can you go over it?" / "Which post?" / "The butternut squash one."

(b) User builds on prior: Turn 1 targets entity X. Agent does it. Turn 3 \
targets entity Y, referencing the prior action.
Example: "Tighten up the intro." / "Done, smoothed transitions..." / \
"Same thing for the benchmarks section."

(c) User corrects agent: Turn 1 makes a request. Agent proposes specifics \
(threshold, name, window). Turn 3 adjusts the proposal.
Example: "Smooth out the weekly spend." / "Applied 7-day rolling average." / \
"Too smooth, try 3 days instead."

AVOID these anti-patterns:
- Two independent requests that coincidentally use the same flow \
("Rename col A" then "Rename col B" with no narrative thread)
- Converse flows (chat, preference) as same-flow pairs (almost always contrived)

The user should NOT use the word "{flow['name']}" directly.

Return JSON:
{{"turns": [{{"role": "user", "utterance": "..."}}, {{"role": "assistant", "utterance": "..."}}, {{"role": "user", "utterance": "..."}}]}}"""


def build_category_b_prompt(
    sample: dict,
    persona_hint: str | None = None,
) -> str:
    """Build user prompt for Category B (switch flow between turns)."""
    flow_x = sample['flow_x']
    flow_y = sample['flow_y']
    scenario = sample['scenario']

    persona_line = f'\nPersona: {persona_hint}\n' if persona_hint else ''

    return f"""## Specification

Category: switch_flow
Scenario: {scenario}
{_flow_spec_block(flow_x)}

{_flow_spec_block(flow_y)}
{persona_line}
Generate a 3-turn conversation where:
- Turn 1: The user clearly wants "{flow_x['name']}"
- Turn 2: The assistant responds with something SPECIFIC about what it did \
or found (a count, a finding, a status, a problem it noticed)
- Turn 3: The user switches to "{flow_y['name']}", TRIGGERED by what the \
assistant said in turn 2

CRITICAL: The switch must emerge from the conversation. The assistant's \
response in turn 2 should reveal information or create context that \
naturally leads the user to request "{flow_y['name']}". Examples:
- Agent reports "800 words, mostly bullet points" -> user asks to flesh them out
- Agent says "formatted for WordPress, tags still empty" -> user wants a preview
- Agent appends data, mentions total -> user worries about duplicates

If turn 3 makes sense without reading turn 2, the conversation fails.

Turn 1 must unambiguously map to "{flow_x['name']}".
Turn 3 must unambiguously map to "{flow_y['name']}".
The user should NOT use the words "{flow_x['name']}" or "{flow_y['name']}" directly.

Return JSON:
{{"turns": [{{"role": "user", "utterance": "..."}}, {{"role": "assistant", "utterance": "..."}}, {{"role": "user", "utterance": "..."}}]}}"""


def build_category_c_prompt(
    sample: dict,
    persona_hint: str | None = None,
) -> str:
    """Build user prompt for Category C (ambiguous first turn, clarified third)."""
    flow_a = sample['flow_a']
    flow_b = sample['flow_b']
    resolves_to = sample['resolves_to']
    scenario = sample['scenario']

    persona_line = f'\nPersona: {persona_hint}\n' if persona_hint else ''

    return f"""## Specification

Category: ambiguous_first
Scenario: {scenario}
Confusable pair: {flow_a['name']} ({flow_a['intent']}) vs {flow_b['name']} ({flow_b['intent']})
Resolution: {resolves_to}

{_flow_spec_block(flow_a)}

{_flow_spec_block(flow_b)}
{persona_line}
Generate a 3-turn conversation where:
- Turn 1: The user's request is genuinely ambiguous, it could plausibly \
be either "{flow_a['name']}" or "{flow_b['name']}". Neither should be obviously more likely.
- Turn 2: The assistant asks a brief clarifying question to disambiguate.
- Turn 3: The user's response makes it clear they want "{resolves_to}".

The ambiguity in turn 1 must be real, not artificial.
The user should NOT use any flow name directly.

Return JSON:
{{"turns": [{{"role": "user", "utterance": "..."}}, {{"role": "assistant", "utterance": "..."}}, {{"role": "user", "utterance": "..."}}]}}"""


def build_category_d_prompt(
    sample: dict,
    persona_hint: str | None = None,
) -> str:
    """Build user prompt for Category D (clear first, multi-request third)."""
    flow_x = sample['flow_x']
    flow_y = sample['flow_y']
    flow_z = sample['flow_z']
    scenario = sample['scenario']

    persona_line = f'\nPersona: {persona_hint}\n' if persona_hint else ''

    return f"""## Specification

Category: ambiguous_second
Scenario: {scenario}
Clear flow (turn 1): {flow_x['name']} ({flow_x['intent']}): {flow_x['description']}
Combined flows (turn 3): {flow_y['name']} ({flow_y['intent']}) + {flow_z['name']} ({flow_z['intent']})

{_flow_spec_block(flow_x)}

{_flow_spec_block(flow_y)}

{_flow_spec_block(flow_z)}
{persona_line}
Generate a 3-turn conversation where:
- Turn 1: The user clearly wants "{flow_x['name']}"
- Turn 2: The assistant responds with something specific about what it did \
or found (a count, a finding, a status)
- Turn 3: The user asks for TWO things in a single utterance, one that \
maps to "{flow_y['name']}" and one that maps to "{flow_z['name']}".

CRITICAL: Do NOT just join two commands with "and". Use one of these natural patterns:
- Causal: "Fix X so I can do Y" (one operation enables the other)
- Prerequisite: "I need X before we do Y"
- Sequential: "Do X, and then do Y" (ordered steps)
- Embedded qualifier: "Do X for all valid Y, meaning the Z ones"
- Observation + question: notice a problem (implies action) + ask something else. \
Example: "I think there are dupes in here. How many rows total?"
- Two sentences: "Do X. How about Y?" (period separates, no conjunction)

One of the two requests CAN be implicit (e.g., observing a problem implies \
the fix without commanding it). Both flows can be from the same intent.

Turn 1 should be unambiguous. Turn 3 must contain two distinct requests.
The user should NOT use any flow name directly.

Return JSON:
{{"turns": [{{"role": "user", "utterance": "..."}}, {{"role": "assistant", "utterance": "..."}}, {{"role": "user", "utterance": "..."}}]}}"""


# -- Prompt Builder -----------------------------------------------------------

CATEGORY_BUILDERS = {
    'a': build_category_a_prompt,
    'b': build_category_b_prompt,
    'c': build_category_c_prompt,
    'd': build_category_d_prompt,
}


def build_user_prompt(category: str, sample: dict, persona_hint: str | None = None) -> str:
    """Build the user prompt for a given category and sample."""
    builder = CATEGORY_BUILDERS[category]
    return builder(sample, persona_hint)
