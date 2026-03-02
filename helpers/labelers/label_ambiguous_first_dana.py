"""Add target_tools labels to ambiguous_first Dana eval turns.

Pattern:
  T1 (flow=ambiguous): handle_ambiguity is always correct, plus tools
      for each candidate_flow (model picks one interpretation).
  T3 (resolved): specific tool(s) for the resolved flow.
"""

import json

LABELS = {
    # dana_011: T1=ambiguous [trend, lookup], T3=trend
    ("dana_011", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["render_chart", "compare_metrics", "lookup_metric"]},
        "render_chart": {"chart_type": "line", "x": "month", "y": "CTR"},
        "compare_metrics": {"column_a": "CTR", "method": "trend"},
        "lookup_metric": {"term": "CTR"},
    },
    ("dana_011", 3): {
        "render_chart": {"chart_type": "line", "x": "month", "y": "CTR"},
        "compare_metrics": {"column_a": "CTR", "method": "trend"},
    },

    # dana_017: T1=ambiguous [lookup, query], T3=lookup
    ("dana_017", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["lookup_metric", "execute_sql"]},
        "lookup_metric": {"term": "gross_margin"},
        "execute_sql": {"query": None},
    },
    ("dana_017", 3): {
        "lookup_metric": {"term": "gross_margin"},
    },

    # dana_018: T1=ambiguous [fill, interpolate], T3=interpolate
    ("dana_018", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["flash_fill", "run_interpolation"]},
        "flash_fill": {"column": "salary", "strategy": None},
        "run_interpolation": {"column": "salary", "method": None},
    },
    # T3: "Yeah, go for it." — confirms agent's suggestion to estimate from correlates
    ("dana_018", 3): {
        "run_interpolation": {"column": "salary", "method": "model"},
    },

    # dana_019: T1=ambiguous [append, insert], T3=append
    ("dana_019", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["append_tables", "insert_rows"]},
        "append_tables": {"source": None, "target": "performance"},
        "insert_rows": {"table": "performance", "values": None},
    },
    ("dana_019", 3): {
        "append_tables": {"source": None, "target": None},
    },

    # dana_020: T1=ambiguous [compare, trend], T3=compare
    ("dana_020", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["compare_metrics", "render_chart"]},
        "compare_metrics": {"column_a": None, "group_by": "platform"},
        "render_chart": {"chart_type": "line", "x": None, "y": None},
    },
    ("dana_020", 3): {
        "compare_metrics": {"column_a": None, "group_by": "platform", "method": "summary"},
    },

    # dana_021: T1=ambiguous [export, dashboard], T3=export
    ("dana_021", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["export_dataset", "compose_dashboard"]},
        "export_dataset": {"format": None},
        "compose_dashboard": {"title": None, "panels": None},
    },
    ("dana_021", 3): {
        "export_dataset": {"format": None},
        "save_dataset": {"format": None},
    },

    # dana_022: T1=ambiguous [style, design], T3=style
    ("dana_022", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["apply_style", "style_chart", "modify_chart"]},
        "apply_style": {"condition": None, "format": None},
        "style_chart": {"chart_id": None},
        "modify_chart": {"chart_id": None},
    },
    ("dana_022", 3): {
        "apply_style": {"condition": None, "format": "color_scale"},
    },

    # dana_023: T1=ambiguous [define, query], T3=define
    ("dana_023", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["define_metric", "execute_sql"]},
        "define_metric": {"name": None, "formula": "avg(days_to_ship) / total_inventory"},
        "execute_sql": {"query": None},
    },
    ("dana_023", 3): {
        "define_metric": {"name": None, "formula": "avg(days_to_ship) / total_inventory"},
    },

    # dana_024: T1=ambiguous [merge, join], T3=merge
    ("dana_024", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["merge_columns", "merge_tables"]},
        "merge_columns": {"columns": ["platform", "handle"], "name": None},
        "merge_tables": {"left": None, "right": None, "key": None},
    },
    # T3: "Exactly." — confirms concatenation
    ("dana_024", 3): {
        "merge_columns": {"columns": ["platform", "handle"], "name": None, "separator": "_"},
    },

    # dana_044: T1=ambiguous [query, segment], T3=query
    ("dana_044", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["execute_sql", "dimension_breakdown"]},
        "execute_sql": {"query": None},
        "dimension_breakdown": {"metric": "satisfaction_score", "dimension": None},
    },
    ("dana_044", 3): {
        "execute_sql": {"query": None},
    },

    # dana_081: T1=ambiguous [reshape, pivot], T3=reshape
    # Both candidates can map to pivot_tables, but the ambiguity is
    # whether user wants stats (pivot/analyze) or restructure (reshape)
    ("dana_081", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["pivot_tables", "dimension_breakdown"]},
        "pivot_tables": {"columns": "feature"},
        "dimension_breakdown": {"metric": None, "dimension": "feature"},
    },
    ("dana_081", 3): {
        "pivot_tables": {"columns": "feature"},
    },

    # dana_082: T1=ambiguous [reshape, pivot], T3=reshape
    ("dana_082", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["pivot_tables"]},
        "pivot_tables": {"columns": "question"},
    },
    ("dana_082", 3): {
        "pivot_tables": {"index": "respondent_id", "columns": "question"},
    },

    # dana_083: T1=ambiguous [exist, describe], T3=exist
    ("dana_083", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["semantic_layer", "describe_stats"]},
        "semantic_layer": {"scope": "columns"},
        "describe_stats": {"column": "gift_wrap"},
    },
    ("dana_083", 3): {
        "semantic_layer": {"scope": "columns"},
        "describe_stats": {"column": "gift_wrap"},
    },

    # dana_084: T1=ambiguous [validate, describe], T3=replace
    # T3: user bypassed ambiguity entirely — gave a direct replace action
    ("dana_084", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["validate_column", "describe_stats"]},
        "validate_column": {"column": "department", "check": None},
        "describe_stats": {"column": "department"},
    },
    ("dana_084", 3): {
        "replace_values": {"column": "department", "find": "R&D", "replacement": "Research & Development"},
    },

    # dana_086: T1=ambiguous [lookup, define], T3=lookup
    ("dana_086", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["lookup_metric", "define_metric"]},
        "lookup_metric": {"term": "gross_margin"},
        "define_metric": {"name": "gross_margin", "formula": None},
    },
    ("dana_086", 3): {
        "lookup_metric": {"term": "gross_margin"},
    },

    # dana_087: T1=ambiguous [describe, datatype], T3=describe
    ("dana_087", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["describe_stats", "validate_column", "cast_column"]},
        "describe_stats": {"table": None},
        "validate_column": {"column": None, "check": None},
    },
    ("dana_087", 3): {
        "describe_stats": {"table": None},
    },

    # dana_088: T1=ambiguous [chat, approve], T3=chat
    # REVISIT: "yeah that sounds good" (approve) + "what does churn rate mean" (chat)
    # Both signals present — handle_ambiguity is cleanest
    ("dana_088", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["no_tool_needed", "explain_content"]},
        "no_tool_needed": {"reason": "user asking about definition"},
        "explain_content": {"topic": "churn rate"},
    },
    ("dana_088", 3): {
        "no_tool_needed": {"reason": "conversational — asking for definition"},
        "explain_content": {"topic": "churn rate definition"},
    },

    # dana_089: T1=ambiguous [trend, compare], T3=trend
    ("dana_089", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["render_chart", "compare_metrics"]},
        "render_chart": {"chart_type": "line", "x": None, "y": "fulfillment_rate"},
        "compare_metrics": {"column_a": "fulfillment_rate", "group_by": None},
    },
    ("dana_089", 3): {
        "render_chart": {"chart_type": "line", "x": "week", "y": "fulfillment_rate"},
        "compare_metrics": {"column_a": "fulfillment_rate", "method": "trend"},
    },

    # dana_091: T1=ambiguous [dashboard, export], T3=dashboard
    ("dana_091", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["compose_dashboard", "export_dataset"]},
        "compose_dashboard": {"title": None, "panels": None},
        "export_dataset": {"format": None},
    },
    ("dana_091", 3): {
        "compose_dashboard": {"title": None, "panels": None},
    },

    # dana_092: T1=ambiguous [describe, segment], T3=describe
    ("dana_092", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["describe_stats", "dimension_breakdown"]},
        "describe_stats": {"column": "satisfaction_score"},
        "dimension_breakdown": {"metric": "satisfaction_score", "dimension": "user_group"},
    },
    ("dana_092", 3): {
        "describe_stats": {"column": "satisfaction_score"},
    },

    # dana_093: T1=ambiguous [validate, format], T3=validate
    ("dana_093", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["validate_column", "format_column"]},
        "validate_column": {"column": "region", "check": None},
        "format_column": {"column": "region", "type": None},
    },
    ("dana_093", 3): {
        "validate_column": {"column": "region", "check": None, "action": "flag"},
    },

    # dana_094: T1=ambiguous [segment, summarize], T3=summarize
    ("dana_094", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["dimension_breakdown", "summarize_content"]},
        "dimension_breakdown": {"metric": "attrition", "dimension": "month"},
        "summarize_content": {"focus": "attrition"},
    },
    ("dana_094", 3): {
        "summarize_content": {"artifact_type": "table", "focus": "attrition patterns"},
    },

    # dana_095: T1=ambiguous [recommend, chat], T3=recommend
    # REVISIT: No direct 'recommend' tool. describe_stats has 'recommend' in _flows.
    # "Suggest something" is advisory — agent uses data to inform suggestion.
    ("dana_095", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["describe_stats", "no_tool_needed"]},
        "describe_stats": {"table": None},
        "no_tool_needed": {"reason": "conversational question about next steps"},
    },
    ("dana_095", 3): {
        "describe_stats": {"table": None},
        "execute_sql": {"query": None},
    },

    # dana_096: T1=ambiguous [insert, fill], T3=insert
    ("dana_096", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["insert_columns", "apply_formula", "flash_fill"]},
        "insert_columns": {"column": "rolling_avg", "expression": None},
        "flash_fill": {"column": "contracted_spend", "strategy": "rolling_average"},
    },
    ("dana_096", 3): {
        "insert_columns": {"column": None, "expression": None},
        "apply_formula": {"column": None, "formula": None},
    },

    # dana_097: T1=ambiguous [preference, format], T3=preference
    ("dana_097", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["manage_memory", "format_column"]},
        "manage_memory": {"operation": "write", "level": "L2", "key": "date_format", "value": "dd/mm/yyyy"},
        "format_column": {"column": None, "type": "date"},
    },
    ("dana_097", 3): {
        "manage_memory": {"operation": "write", "level": "L2", "key": "date_format", "value": "dd/mm/yyyy"},
    },

    # dana_098: T1=ambiguous [datatype, update], T3=datatype
    ("dana_098", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["cast_column", "validate_column", "modify_column"]},
        "cast_column": {"column": "signup_date", "target_type": "datetime"},
        "validate_column": {"column": "signup_date", "check": None},
    },
    # T3: "scan it first" — investigate before casting
    ("dana_098", 3): {
        "validate_column": {"column": "signup_date", "check": None},
        "describe_stats": {"column": "signup_date"},
        "cast_column": {"column": "signup_date", "target_type": "datetime"},
    },

    # dana_099: T1=ambiguous [join, merge], T3=join
    ("dana_099", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["merge_tables", "merge_by_key", "merge_columns"]},
        "merge_tables": {"left": "inventory", "right": "shipping", "key": None},
        "merge_columns": {"columns": None, "name": None},
    },
    ("dana_099", 3): {
        "merge_tables": {"left": "inventory", "right": "shipping", "key": "warehouse_id"},
        "merge_by_key": {"left": "inventory", "right": "shipping", "key": "warehouse_id"},
    },

    # dana_100: T1=ambiguous [replace, validate], T3=replace
    ("dana_100", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["replace_values", "validate_column"]},
        "replace_values": {"column": "platform", "find": "IG", "replacement": "Instagram"},
        "validate_column": {"column": "platform", "check": None},
    },
    ("dana_100", 3): {
        "replace_values": {"column": "platform", "find": "IG", "replacement": "Instagram"},
    },

    # dana_101: T1=ambiguous [plot, compare], T3=plot
    ("dana_101", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["render_chart", "compare_metrics"]},
        "render_chart": {"chart_type": None, "x": "segment", "y": "adoption"},
        "compare_metrics": {"column_a": "adoption", "group_by": "segment"},
    },
    ("dana_101", 3): {
        "render_chart": {"chart_type": "bar", "x": "segment", "y": "adoption"},
    },

    # dana_102: T1=ambiguous [describe, fill], T3=describe
    ("dana_102", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["describe_stats", "flash_fill"]},
        "describe_stats": {"column": "satisfaction_score"},
        "flash_fill": {"column": "satisfaction_score", "strategy": None},
    },
    ("dana_102", 3): {
        "describe_stats": {"column": "satisfaction_score"},
    },

    # dana_103: T1=ambiguous [trend, dashboard], T3=trend
    ("dana_103", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["render_chart", "compose_dashboard"]},
        "render_chart": {"chart_type": "line", "x": "week", "y": "sales"},
        "compose_dashboard": {"title": None, "panels": None},
    },
    ("dana_103", 3): {
        "render_chart": {"chart_type": "line", "x": "week", "y": "sales", "color": "region"},
    },

    # dana_104: T1=ambiguous [segment, plot], T3=summarize
    # T3: user chose summarize — outside the original candidates
    ("dana_104", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["dimension_breakdown", "render_chart"]},
        "dimension_breakdown": {"metric": "attrition", "dimension": "department"},
        "render_chart": {"chart_type": None, "x": "department", "y": "attrition"},
    },
    ("dana_104", 3): {
        "summarize_content": {"artifact_type": "table", "focus": "attrition by department"},
    },
}


def main():
    with open("datasets/dana/eval_set.json") as f:
        data = json.load(f)

    labeled = 0
    missing = 0
    for convo in data:
        if convo["category"] != "ambiguous_first":
            continue
        cid = convo["convo_id"]
        for turn in convo["turns"]:
            if turn.get("speaker") != "user":
                continue
            key = (cid, turn["turn_num"])
            if key in LABELS:
                turn["target_tools"] = LABELS[key]
                labeled += 1
            else:
                print(f"WARNING: No label for {key}")
                missing += 1

    with open("datasets/dana/eval_set.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"Labeled {labeled} turns, {missing} missing")


if __name__ == "__main__":
    main()
