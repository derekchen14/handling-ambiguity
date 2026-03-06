"""Build system prompts for param extraction eval (--mode slot)."""

from __future__ import annotations


_DOMAIN_INTROS = {
    'hugo': 'Hugo is a blog writing assistant that helps users research topics, draft posts, revise content, and publish to platforms.',
    'dana': 'Dana is a data analysis assistant that helps users clean datasets, transform data, run analyses, and create reports and visualizations.',
}

_PARAM_FORMAT_GUIDELINES = """- **Dates/times**: Use a structured object: {"day": "friday", "time": "08:00", "tz": "EST"}. Use lowercase day names. Use 24h time format (e.g. "08:00", "18:00"). Omit fields you cannot infer.
- **Enums**: Use the exact enum value from the schema (lowercase where specified).
- **IDs**: Use null when not explicitly stated.
- **Platform names**: Use lowercase (e.g. "substack", not "Substack")."""

_HUGO_EXAMPLE = """### Example
Conversation history:
User: "The roundup needs to go out on Medium next Tuesday at 8am."

Tool: `manage_schedule`
Parameters schema:
  - post_id (string): Post to manage schedule for
  - action (string, enum: view|reschedule|cancel|schedule): Schedule management action
  - platform (string, enum: blog|substack|twitter|linkedin|medium|wordpress): Platform
  - datetime (object): Structured date/time with day, time, tz fields

Output:
```json
{"reasoning": "User wants to schedule a post for Tuesday 8am on Medium.", "params": {"post_id": null, "action": "schedule", "platform": "medium", "datetime": {"day": "tuesday", "time": "08:00"}}}
```"""

_DANA_EXAMPLE = """### Example
Conversation history:
User: "Show me average tenure by department, broken down by attrition status, from q4_sales."

Tool: `pivot_tables`
Parameters schema:
  - table (string): Source table name
  - index (string): Row grouping column
  - columns (string): Column grouping column
  - values (string): Values column to aggregate
  - aggfunc (string, enum: mean|sum|count|min|max): Aggregation function

Output:
```json
{"reasoning": "User wants a pivot table with department as rows, attrition_status as columns, averaging tenure.", "params": {"table": "q4_sales", "index": "department", "columns": "attrition_status", "values": "tenure", "aggfunc": "mean"}}
```"""


def _render_param_schema(tool_def: dict) -> str:
    """Render parameter schema as a readable list from a tool definition."""
    props = tool_def.get('input_schema', {}).get('properties', {})
    required = set(tool_def.get('input_schema', {}).get('required', []))
    lines = []
    for name, schema in props.items():
        ptype = schema.get('type', 'string')
        desc = schema.get('description', '')
        parts = [f'  - {name} ({ptype}']
        if 'enum' in schema:
            parts.append(f', enum: {"|".join(schema["enum"])}')
        parts.append(f'): {desc}')
        if name in required:
            parts.append(' [required]')
        lines.append(''.join(parts))
    return '\n'.join(lines)


def build_param_extraction_prompt(
    domain: str, tool_name: str, tool_schema: dict, context: dict | None = None,
) -> str:
    """Build a prompt asking the model to extract params for a single tool."""
    intro = _DOMAIN_INTROS.get(domain, f'{domain.capitalize()} is a conversational assistant.')
    example = _HUGO_EXAMPLE if domain == 'hugo' else _DANA_EXAMPLE
    param_list = _render_param_schema(tool_schema)

    context_section = ''
    if context:
        if domain == 'hugo':
            pid = context.get('post_id', '')
            title = context.get('post_title', '')
            platform = context.get('platform', '')
            parts = []
            if pid and title:
                parts.append(f'Active post: **{pid}** ("{title}").')
            elif pid:
                parts.append(f'Active post: **{pid}**.')
            if platform:
                parts.append(f'Target platform: {platform}.')
            context_section = '\n\n## Active Context\n' + ' '.join(parts)
        elif domain == 'dana':
            table = context.get('table', '')
            columns = context.get('columns', [])
            if table:
                context_section = f'\n\n## Active Context\nActive dataset: `{table}`'
                if columns:
                    context_section += f' with columns [{", ".join(columns)}]'
                context_section += '.'

    return f"""{intro}

The correct tool to call is `{tool_name}`. Here are its parameters:
{param_list}

## Instructions

Extract parameter values from the most recent user message and conversation history.
- Use `null` for parameters whose values are not stated and cannot be inferred.
- Respect enum values exactly (use the exact string from the schema).
- Use structured datetime objects where applicable.

## Parameter Format Guidelines

{_PARAM_FORMAT_GUIDELINES}{context_section}

{example}

---

Now extract the parameters. Respond with JSON only — no other text.
Output format: {{"reasoning": "...", "params": {{"param_name": value}}}}
"""


def build_batch_param_extraction_prompt(
    domain: str, tools_with_schemas: list[dict], context: dict | None = None,
) -> str:
    """Build a prompt asking the model to extract params for multiple tools at once.

    Each entry in tools_with_schemas: {"name": str, "schema": dict}
    """
    intro = _DOMAIN_INTROS.get(domain, f'{domain.capitalize()} is a conversational assistant.')
    example = _HUGO_EXAMPLE if domain == 'hugo' else _DANA_EXAMPLE

    context_section = ''
    if context:
        if domain == 'hugo':
            pid = context.get('post_id', '')
            title = context.get('post_title', '')
            platform = context.get('platform', '')
            parts = []
            if pid and title:
                parts.append(f'Active post: **{pid}** ("{title}").')
            elif pid:
                parts.append(f'Active post: **{pid}**.')
            if platform:
                parts.append(f'Target platform: {platform}.')
            context_section = '\n\n## Active Context\n' + ' '.join(parts)
        elif domain == 'dana':
            table = context.get('table', '')
            columns = context.get('columns', [])
            if table:
                context_section = f'\n\n## Active Context\nActive dataset: `{table}`'
                if columns:
                    context_section += f' with columns [{", ".join(columns)}]'
                context_section += '.'

    tool_sections = []
    for entry in tools_with_schemas:
        name = entry['name']
        param_list = _render_param_schema(entry['schema'])
        tool_sections.append(f'### `{name}`\n{param_list}')
    tools_block = '\n\n'.join(tool_sections)

    return f"""{intro}

The correct tools to call are listed below with their parameters:

{tools_block}

## Instructions

Extract parameter values for each tool from the most recent user message and conversation history.
- Use `null` for parameters whose values are not stated and cannot be inferred.
- Respect enum values exactly (use the exact string from the schema).
- Use structured datetime objects where applicable.

## Parameter Format Guidelines

{_PARAM_FORMAT_GUIDELINES}{context_section}

{example}

---

Now extract the parameters for all tools. Respond with JSON only — no other text.
Output format: {{"tools": [{{"name": "tool_name", "params": {{...}}}}, ...]}}
"""
