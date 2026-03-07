#!/usr/bin/env python3
"""Find enum violations in eval sets against tool manifests."""

import json
import sys
from pathlib import Path


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_enum_map(manifest):
    """Build a map of (tool_name, param_name) -> list of allowed enum values."""
    enum_map = {}
    for tool in manifest:
        tool_name = tool["name"]
        props = tool.get("input_schema", {}).get("properties", {})
        for param_name, param_def in props.items():
            if "enum" in param_def:
                enum_map[(tool_name, param_name)] = param_def["enum"]
            # Also check nested object properties (e.g., panels -> items -> properties -> type)
            if param_def.get("type") == "array" and "items" in param_def:
                items = param_def["items"]
                if isinstance(items, dict) and "properties" in items:
                    for nested_name, nested_def in items["properties"].items():
                        if "enum" in nested_def:
                            enum_map[(tool_name, f"{param_name}[].{nested_name}")] = nested_def["enum"]
    return enum_map


def extract_value(val):
    """Extract the actual value from a possibly-fuzzy wrapper."""
    if isinstance(val, dict) and "fuzzy" in val:
        return val.get("value")
    return val


def check_turn(tool_name, params, enum_map, violations, convo_id, turn_num):
    """Check a single tool call's parameters against the enum map."""
    if not isinstance(params, dict):
        return
    for param_name, param_val in params.items():
        actual = extract_value(param_val)
        if actual is None:
            continue
        key = (tool_name, param_name)
        if key in enum_map:
            allowed = enum_map[key]
            if actual not in allowed:
                violations.append({
                    "convo_id": convo_id,
                    "turn_num": turn_num,
                    "tool_name": tool_name,
                    "param_name": param_name,
                    "actual_value": actual,
                    "allowed_enum": allowed,
                })
        # Also handle nested arrays (like panels)
        if isinstance(actual, list):
            for i, item in enumerate(actual):
                if isinstance(item, dict):
                    for nested_name, nested_val in item.items():
                        nested_actual = extract_value(nested_val)
                        if nested_actual is None:
                            continue
                        nested_key = (tool_name, f"{param_name}[].{nested_name}")
                        if nested_key in enum_map:
                            allowed = enum_map[nested_key]
                            if nested_actual not in allowed:
                                violations.append({
                                    "convo_id": convo_id,
                                    "turn_num": turn_num,
                                    "tool_name": tool_name,
                                    "param_name": f"{param_name}[{i}].{nested_name}",
                                    "actual_value": nested_actual,
                                    "allowed_enum": allowed,
                                })


def find_violations(manifest_path, eval_path):
    manifest = load_json(manifest_path)
    eval_set = load_json(eval_path)
    enum_map = build_enum_map(manifest)

    print(f"\n{'='*80}")
    print(f"Manifest: {manifest_path}")
    print(f"Eval set: {eval_path}")
    print(f"{'='*80}")

    print(f"\nEnum-constrained parameters found in manifest:")
    for (tool, param), values in sorted(enum_map.items()):
        print(f"  {tool}.{param}: {values}")

    violations = []
    for convo in eval_set:
        convo_id = convo.get("convo_id", "?")
        for turn in convo.get("turns", []):
            if turn.get("speaker") != "user":
                continue
            turn_num = turn.get("turn_num", "?")
            target_tools = turn.get("target_tools", {})
            if not isinstance(target_tools, dict):
                continue
            for tool_name, params in target_tools.items():
                check_turn(tool_name, params, enum_map, violations, convo_id, turn_num)

    print(f"\n--- Violations ({len(violations)}) ---")
    if violations:
        for v in violations:
            print(f"\n  convo_id:    {v['convo_id']}")
            print(f"  turn_num:    {v['turn_num']}")
            print(f"  tool_name:   {v['tool_name']}")
            print(f"  param_name:  {v['param_name']}")
            print(f"  actual:      {v['actual_value']!r}")
            print(f"  allowed:     {v['allowed_enum']}")
    else:
        print("  None found.")

    return violations


def main():
    base = Path("/Users/pranavraja/Documents/research/handling-ambiguity")

    all_violations = []

    # Hugo
    v = find_violations(
        base / "tools" / "tool_manifest_hugo.json",
        base / "datasets" / "hugo" / "eval_set.json",
    )
    all_violations.extend(v)

    # Dana
    v = find_violations(
        base / "tools" / "tool_manifest_dana.json",
        base / "datasets" / "dana" / "eval_set.json",
    )
    all_violations.extend(v)

    print(f"\n{'='*80}")
    print(f"TOTAL VIOLATIONS: {len(all_violations)}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
