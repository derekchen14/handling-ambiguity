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
    """Read the explicit match_method from a parameter schema.

    Returns: "exact" | "fuzzy" | "structured"
    Raises ValueError if the field is missing or invalid.
    """
    method = param_schema.get('match_method')
    if method is None:
        raise ValueError(f"Parameter schema missing required 'match_method' field: {param_schema}")
    if method not in ('exact', 'fuzzy', 'structured'):
        raise ValueError(f"Invalid match_method '{method}', must be exact/fuzzy/structured")
    return method
