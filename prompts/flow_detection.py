"""Build system prompts for flow detection experiments (Exp 1)."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root so we can import assistant ontologies
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_flow_catalog(domain: str) -> tuple:
    """Import the FLOW_CATALOG for a domain."""
    if domain == 'hugo':
        from assistants.Hugo.schemas.ontology import FLOW_CATALOG, Intent
    elif domain == 'dana':
        from assistants.Dana.schemas.ontology import FLOW_CATALOG, Intent
    else:
        raise ValueError(f'Unknown domain: {domain}')
    return FLOW_CATALOG, Intent


def determine_candidate_flows(domain: str, intents: list[str] | None = None) -> list[tuple[str, str]]:
    """Determine the candidate flow set from intent(s) + edge_flows.

    When intents is provided, collects flows belonging to those intents plus
    any flows referenced by their edge_flows.  When intents is None, returns
    all user-facing flows (everything except Internal).

    Returns sorted list of (flow_name, description).
    """
    flow_catalog, _ = _load_flow_catalog(domain)

    if intents is None:
        # All user-facing flows
        result = []
        for name in sorted(flow_catalog):
            flow = flow_catalog[name]
            if flow['intent'].value == 'Internal':
                continue
            result.append((name, flow.get('description', '')))
        return result

    # Step 1: Collect flows from the specified intents
    base_names = set()
    for name, flow in flow_catalog.items():
        intent_val = flow['intent'].value
        if intent_val in intents:
            base_names.add(name)

    # Step 2: Collect edge_flows from base candidates
    candidate_names = set(base_names)
    for name in base_names:
        edges = flow_catalog[name].get('edge_flows', [])
        candidate_names.update(edges)

    # Step 3: Filter out Internal flows (system-level, never user-invoked)
    result = []
    for name in sorted(candidate_names):
        if name not in flow_catalog:
            continue
        flow = flow_catalog[name]
        intent_val = flow['intent'].value
        if intent_val == 'Internal':
            continue
        desc = flow.get('description', '')
        result.append((name, desc))

    return result


# ── Domain-specific exemplars ─────────────────────────────────

_HUGO_EXEMPLARS = """### Example 1

User: "Add a section on offshore wind farms after the solar panel discussion."

```json
{"reasoning": "'Add a section' at a specific position in the post structure. This inserts a new section placeholder, not generating prose content.", "flows": ["add"]}
```

### Example 2

User: "Outline a post comparing React and Vue for enterprise apps."
Agent: "Here's a five-section outline: architecture, ecosystem, performance, hiring, and migration."
User: "Move hiring before performance, that's what CTOs care about first."

```json
{"reasoning": "Reordering sections within an existing outline. 'Move ... before' = structural adjustment, not new content.", "flows": ["refine"]}
```

### Example 3

User: "Have I published anything about container orchestration before?"

```json
{"reasoning": "'Published ... before about' implies searching previous posts by topic to find existing content.", "flows": ["find"]}
```

### Example 4
User: "How does the conclusion of the DevOps post read?"
Agent: "180 words, summarizes three benefits and ends with a call to try Terraform."
User: "Tighten the last paragraph, it rambles."

```json
{"reasoning": "'Tighten' and 'rambles' refer to surface-level conciseness improvements on a specific section. Light editing of text, not a structural rewrite of the outline.", "flows": ["polish"]}
```

### Example 5

User: "The accessibility section needs concrete WCAG examples and the whole post should go out to the newsletter."

```json
{"reasoning": "'Needs concrete WCAG examples' = substantive content revision, adding missing detail. 'Go out to the newsletter' = a separate distribution action to a secondary platform. Two distinct operations.", "flows": ["rework", "syndicate"]}
```"""


_DANA_EXEMPLARS = """### Example 1

User: "How many columns does the inventory table have and what are their types?"

```json
{"reasoning": "Asking for table structure metadata: column count and data types. Basic dataset inspection.", "flows": ["describe"]}
```

### Example 2

User: "Show me AOV by customer tier."
Agent: "Gold $142, Silver $89, Bronze $54."
User: "Now break that down by quarter."

```json
{"reasoning": "'Break down by quarter' adds a time-based grouping dimension to the previous analysis.", "flows": ["segment"]}
```

### Example 3

User: "The phone numbers have dashes, parentheses, and spaces mixed in. Normalize them to digits only."

```json
{"reasoning": "Inconsistent representations of the same data need standardization. 'Normalize' + mixed formatting = fixing the surface form, not checking validity.", "flows": ["format"]}
```

### Example 4

User: "What does the churn_risk column look like?"
Agent: "Numeric float 0-1, mean 0.34, 2100 nulls out of 50K rows."
User: "Are there any customer_segment values that look wrong?"

```json
{"reasoning": "'Values that look wrong' means checking against valid options. Validating correctness of existing data, not fixing formatting.", "flows": ["validate"]}
```

### Example 5

User: "There are gaps in the temperature readings and I need a line chart of temperature over time."

```json
{"reasoning": "'Gaps in temperature readings' = missing data that needs imputation. 'Line chart over time' = time-series visualization. Two independent operations.", "flows": ["fill", "trend"]}
```"""


def build_flow_detection_prompt(domain: str, intents: list[str] | None = None) -> str:
    """Build the system prompt for flow detection.

    Scopes candidate flows to the given intent(s) plus their edge_flows,
    instructs the model to return JSON with reasoning and a list of flows,
    and includes 5 domain-specific exemplars.

    Args:
        domain: 'hugo' or 'dana'
        intents: Intent name(s) to scope candidates. One for clear turns,
                 two for ambiguous turns that span intents. None = all
                 user-facing flows (used by self-consistency ensembles).
    """
    domain_label = domain.capitalize()

    candidates = determine_candidate_flows(domain, intents)
    flow_lines = [f'- {name}: {desc}' for name, desc in candidates]
    flow_list = '\n'.join(flow_lines)

    exemplars = _HUGO_EXEMPLARS if domain == 'hugo' else _DANA_EXEMPLARS

    prompt = f"""You are a flow classifier for {domain_label}, a conversational assistant.

Given the conversation history, classify the final user utterance into one or more flows from the candidate list below.
You should only be focused on the last user utterance, with the conversation history only being used to provide context.

## Candidate Flows

{flow_list}

## Instructions

1. Read the user's utterance and any prior conversation context.
2. Identify the key patterns, words, or phrases that signal what the user wants.
3. Determine which flow(s) match the user's request. Many utterances map to one flow, but some may be ambiguous or contain multiple requests.
4. If the utterance could reasonably be interpreted as two different flows, then output all plausible flows to cover all bases.
5. Only output flows from the Candidate Flows list above.
6. Return ONLY valid JSON. No markdown fences, no explanation outside the JSON.

## Output Format

{{"reasoning": "<What key words or phrases hint at the answer? How did you reach your conclusion?>", "flows": ["<flow_name>", "<optional_second_flow>"]}}

## Examples

{exemplars}

Respond with exactly one JSON object. Nothing else."""

    return prompt
