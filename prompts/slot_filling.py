"""Build system prompts for slot-filling (Exp 2A stage 3)."""

from __future__ import annotations

import sys
from pathlib import Path

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


def _get_flow_slot_schema(domain: str, flow_name: str) -> dict:
    """Get the slot schema for a specific flow from the ontology."""
    flow_catalog, _ = _load_flow_catalog(domain)
    flow = flow_catalog.get(flow_name, {})
    return flow.get('slots', {})


def get_flow_slot_schema(domain: str, flow_name: str) -> dict:
    """Public accessor for flow slot schema (used by harness)."""
    return _get_flow_slot_schema(domain, flow_name)


def build_slot_filling_prompt(domain: str, flow_name: str) -> str:
    """Build the system prompt for slot-filling (Exp 2A stage 3).

    Given the detected flow, extracts slot values from the user's message.
    Uses the flow's slot schema from the ontology.
    """
    domain_label = domain.capitalize()
    slots = _get_flow_slot_schema(domain, flow_name)

    if not slots:
        slot_desc = '(This flow has no defined slots.)'
    else:
        lines = []
        for name, info in slots.items():
            stype = info.get('type', 'unknown')
            priority = info.get('priority', 'optional')
            lines.append(f'- **{name}** (type: {stype}, priority: {priority})')
        slot_desc = '\n'.join(lines)

    prompt = f"""You are a slot-filling module for {domain_label}. The user's intent has been classified and the detected flow is **{flow_name}**.

Your job: extract slot values from the user's most recent message.

## Slot Schema for `{flow_name}`

{slot_desc}

## Instructions

1. Extract values for each slot that are explicitly stated or clearly implied in the user's message.
2. Use conversation history to resolve references ("it", "that one", "the same").
3. Output `null` for slots whose values are not stated and cannot be inferred.
4. Do NOT invent or hallucinate values — only extract what is present.
5. Only use slot names from the schema above. Do not add extra keys.
6. Return ONLY valid JSON. No markdown fences, no explanation outside the JSON.

## Output Format

{{"reasoning": "<Brief explanation of extracted values>", "slots": {{"slot_name": "value_or_null"}}}}

## Examples

### Example 1
Flow: outline
User: "Outline a post comparing React and Vue for enterprise apps."
```json
{{"reasoning": "Topic is explicitly stated: comparing React and Vue for enterprise apps. No depth specified.", "slots": {{"topic": "comparing React and Vue for enterprise apps", "depth": null}}}}
```

### Example 2
Flow: schedule
User: "Schedule the DevOps post for next Tuesday on Medium."
```json
{{"reasoning": "Post reference is 'the DevOps post', platform is Medium, datetime is 'next Tuesday'.", "slots": {{"post_id": "DevOps post", "platform": "Medium", "datetime": "next Tuesday"}}}}
```

Respond with exactly one JSON object. Nothing else."""

    return prompt
