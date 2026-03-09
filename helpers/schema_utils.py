"""Utilities for introspecting tool manifests and classifying param match methods."""

from __future__ import annotations


def build_param_schema_index(manifest: list[dict]) -> dict[tuple[str, str], dict]:
    """Build {(tool_name, param_name): property_schema} from a tool manifest."""
    index = {}
    for tool in manifest:
        name = tool.get('name', '')
        props = tool.get('input_schema', {}).get('properties', {})
        for param_name, param_schema in props.items():
            index[(name, param_name)] = param_schema
    return index


def classify_match_method(param_schema: dict) -> str:
    """Classify a param's match method from its JSON Schema definition.

    Returns: "exact" | "fuzzy" | "structured"
    - "exact": has "enum", or type is "boolean"/"integer"/"number"
    - "structured": type is "object" or "array"
    - "fuzzy": plain "type": "string" with no enum (default)
    """
    ptype = param_schema.get('type', '')

    if 'enum' in param_schema:
        return 'exact'
    elif not param_schema:
        return 'fuzzy'
    elif ptype in ('boolean', 'integer', 'number'):
        return 'exact'
    elif ptype in ('object', 'array'):
        return 'structured'

    return 'fuzzy'
