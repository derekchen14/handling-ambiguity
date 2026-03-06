"""One-time fix: correct eval set enum violations to match tool manifests.

Applies value corrections for both Hugo and Dana eval sets.
Run: python scripts/fix_eval_enum_violations.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Replacement maps ─────────────────────────────────────────────
# Key: (tool_name, param_name, bad_value) → corrected_value

HUGO_FIXES = {}

# revise_content.focus: all free-text values → "general"
# (The enum is about revision focus *type*, not content targets)
HUGO_FOCUS_ENUM = {'structure', 'clarity', 'conciseness', 'flow', 'accuracy', 'general'}
# We'll handle this dynamically: any value not in enum → "general"

# format_content.platform: case mismatch → lowercase
HUGO_FIXES[('format_content', 'platform', 'WordPress')] = 'wordpress'
HUGO_FIXES[('format_content', 'platform', 'Medium')] = 'medium'
HUGO_FIXES[('format_content', 'platform', 'Substack')] = 'substack'
HUGO_FIXES[('format_content', 'platform', 'LinkedIn')] = 'linkedin'

# check_platform.platform: case mismatch → lowercase
HUGO_FIXES[('check_platform', 'platform', 'Substack')] = 'substack'

# adjust_tone.intensity
HUGO_FIXES[('adjust_tone', 'intensity', 'lighter')] = 'subtle'

# explain_action.scope
HUGO_FIXES[('explain_action', 'scope', 'routing section')] = 'detailed'

# update_metadata.status
HUGO_FIXES[('update_metadata', 'status', 'pinned')] = 'published'
HUGO_FIXES[('update_metadata', 'status', 'featured')] = 'published'

# ── Dana fixes ───────────────────────────────────────────────────

DANA_FIXES = {
    ('validate_column', 'check', 'enum'): 'in_list',
    ('validate_column', 'check', 'format'): 'regex',
    ('validate_column', 'check', 'type'): 'in_range',
    ('flash_fill', 'strategy', 'forward_fill'): 'ffill',
    ('flash_fill', 'strategy', 'rolling_average'): 'mean',
    ('compare_metrics', 'method', 'trend'): 'summary',
    ('dimension_breakdown', 'aggregation', 'avg'): 'mean',
    ('run_interpolation', 'method', 'model'): 'linear',
    ('cast_column', 'target_type', 'integer'): 'int',
    ('cast_column', 'target_type', 'numeric'): 'float',
    ('semantic_layer', 'scope', 'table'): 'tables',
    ('semantic_layer', 'scope', 'column'): 'columns',
    ('export_dataset', 'format', 'xlsx'): 'excel',
    ('apply_style', 'format', 'bold'): 'highlight',
    ('modify_table', 'operation', 'transpose'): 'reorder_columns',
}


def fix_eval_set(eval_set: list[dict], fixes: dict, focus_enum: set | None = None) -> int:
    """Apply fixes to an eval set. Returns count of changes."""
    count = 0
    for convo in eval_set:
        for turn in convo.get('turns', []):
            for tool_name, params in turn.get('target_tools', {}).items():
                if not isinstance(params, dict):
                    continue
                for param_name, val in list(params.items()):
                    if val is None or isinstance(val, (dict, list)):
                        continue
                    # Check explicit fix map
                    key = (tool_name, param_name, val)
                    if key in fixes:
                        params[param_name] = fixes[key]
                        count += 1
                    # Dynamic: revise_content.focus not in enum → "general"
                    elif focus_enum and tool_name == 'revise_content' and param_name == 'focus':
                        if val not in focus_enum:
                            params[param_name] = 'general'
                            count += 1
    return count


def main():
    # Hugo
    hugo_path = ROOT / 'datasets' / 'hugo' / 'eval_set.json'
    with open(hugo_path) as f:
        hugo_eval = json.load(f)
    hugo_count = fix_eval_set(hugo_eval, HUGO_FIXES, focus_enum=HUGO_FOCUS_ENUM)
    with open(hugo_path, 'w') as f:
        json.dump(hugo_eval, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(f'Hugo: fixed {hugo_count} enum violations')

    # Dana
    dana_path = ROOT / 'datasets' / 'dana' / 'eval_set.json'
    with open(dana_path) as f:
        dana_eval = json.load(f)
    dana_count = fix_eval_set(dana_eval, DANA_FIXES)
    with open(dana_path, 'w') as f:
        json.dump(dana_eval, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(f'Dana: fixed {dana_count} enum violations')


if __name__ == '__main__':
    main()
