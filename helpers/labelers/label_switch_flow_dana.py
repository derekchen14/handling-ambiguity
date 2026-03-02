"""Add target_tools labels to switch_flow Dana eval turns."""

import json

LABELS = {
    # dana_009: T1=replace, T3=reject
    ("dana_009", 1): {
        "replace_values": {"column": "region", "find": "APAC", "replacement": "Asia-Pacific"},
    },
    # T3: "Everywhere means everywhere" — user corrects scope. Reject flow but
    # the implicit action is to redo replace across all columns.
    ("dana_009", 3): {
        "replace_values": {"column": None, "find": "APAC", "replacement": "Asia-Pacific"},
        "no_tool_needed": {"reason": "user is correcting the scope of the previous action"},
    },

    # dana_010: T1=append, T3=dedupe
    ("dana_010", 1): {
        "append_tables": {"source": "Q2_headcount", "target": "Q1_data"},
    },
    ("dana_010", 3): {
        "dedupe_single_col": {"column": "employee_id"},
        "dedupe_columns": {"columns": ["employee_id"]},
    },

    # dana_012: T1=trend, T3=insert
    ("dana_012", 1): {
        "render_chart": {"chart_type": "line", "x": "month", "y": "operating_expenses"},
        "compare_metrics": {"column_a": "operating_expenses", "method": "trend", "group_by": "business_unit"},
    },
    ("dana_012", 3): {
        "insert_columns": {"column": "pct_change", "expression": "percent change of operating_expenses"},
        "apply_formula": {"column": "pct_change", "formula": "pct_change(operating_expenses)"},
    },

    # dana_013: T1=validate, T3=query
    ("dana_013", 1): {
        "validate_column": {"column": "icd_code", "check": "format"},
    },
    ("dana_013", 3): {
        "execute_sql": {"query": "SELECT site FROM patients WHERE icd_code_invalid GROUP BY site"},
    },

    # dana_014: T1=format, T3=describe
    ("dana_014", 1): {
        "format_column": {"column": "signup_date", "type": "date"},
    },
    ("dana_014", 3): {
        "describe_stats": {"table": None, "column": None},
        "semantic_layer": {"scope": "table"},
    },

    # dana_015: T1=exist, T3=query
    ("dana_015", 1): {
        "execute_sql": {"query": "SELECT COUNT(*) FROM shipping WHERE warehouse = 'Phoenix'"},
        "describe_stats": {"table": "shipping"},
    },
    ("dana_015", 3): {
        "execute_sql": {"query": "SELECT item, SUM(volume) FROM shipping WHERE warehouse = 'Phoenix' GROUP BY item ORDER BY 2 DESC LIMIT 5"},
    },

    # dana_016: T1=reshape, T3=plot
    ("dana_016", 1): {
        "pivot_tables": {"table": None},
        "modify_table": {"operation": "transpose"},
    },
    ("dana_016", 3): {
        "render_chart": {"chart_type": "scatter", "x": "likes", "y": "shares"},
    },

    # dana_057: T1=lookup, T3=fill
    ("dana_057", 1): {
        "lookup_metric": {"term": "power_user_score"},
    },
    ("dana_057", 3): {
        "flash_fill": {"column": "days_active_last_30", "strategy": "forward_fill"},
    },

    # dana_058: T1=approve, T3=export
    # T1: user confirms a suggested breakdown → the tool is the analytical action
    ("dana_058", 1): {
        "dimension_breakdown": {"metric": "satisfaction_score", "dimension": "region"},
    },
    ("dana_058", 3): {
        "export_dataset": {"format": "csv"},
        "save_dataset": {"format": "csv"},
    },

    # dana_059: T1=segment, T3=preference
    ("dana_059", 1): {
        "dimension_breakdown": {"metric": "revenue", "dimension": "region"},
    },
    # T3: save a display preference → manage_memory (L2 user preferences)
    ("dana_059", 3): {
        "manage_memory": {"operation": "write", "level": "L2", "key": "number_format", "value": "full_dollar_amounts"},
    },

    # dana_060: T1=replace, T3=reject
    ("dana_060", 1): {
        "replace_values": {"column": "department", "find": "HR", "replacement": "Human Resources"},
    },
    # T3: user declines the suggestion — no action needed
    ("dana_060", 3): {
        "no_tool_needed": {"reason": "user declined the agent's suggestion to standardize casing"},
    },

    # dana_061: T1=compare, T3=pivot
    ("dana_061", 1): {
        "compare_metrics": {"column_a": "open_rate", "column_b": "CTR", "group_by": "campaign_type"},
        "execute_sql": {"query": None},
    },
    ("dana_061", 3): {
        "pivot_tables": {"index": "segment", "columns": "campaign_type", "values": "open_rate"},
    },

    # dana_062: T1=define, T3=append
    ("dana_062", 1): {
        "define_metric": {"name": "gross_margin", "formula": "revenue_total - cogs_total"},
    },
    ("dana_062", 3): {
        "append_tables": {"source": "East", "target": "consolidated"},
    },

    # dana_063: T1=export, T3=delete
    ("dana_063", 1): {
        "export_dataset": {"format": "xlsx"},
    },
    ("dana_063", 3): {
        "delete_rows": {"condition": "site_id is null"},
    },

    # dana_064: T1=join, T3=design
    ("dana_064", 1): {
        "merge_tables": {"left": "subscriptions", "right": "usage_metrics", "key": "account_id"},
        "merge_by_key": {"left": "subscriptions", "right": "usage_metrics", "key": "account_id"},
    },
    # T3: legend position + colors → style_chart
    ("dana_064", 3): {
        "style_chart": {"legend": "bottom", "colors": None},
    },

    # dana_065: T1=dashboard, T3=style
    ("dana_065", 1): {
        "compose_dashboard": {"title": "Weekly Stock & Shipping Report", "panels": None},
    },
    ("dana_065", 3): {
        "apply_style": {"condition": "inventory < 500", "format": "bold", "color": "red"},
    },

    # dana_066: T1=compare, T3=fill
    ("dana_066", 1): {
        "compare_metrics": {"column_a": "engagement_rate", "group_by": "platform", "method": None},
    },
    ("dana_066", 3): {
        "flash_fill": {"column": "engagement", "strategy": "forward_fill"},
    },

    # dana_067: T1=approve, T3=split
    # T1: user confirms the suggested feature adoption breakdown
    ("dana_067", 1): {
        "dimension_breakdown": {"metric": "feature_adoption", "dimension": None},
    },
    ("dana_067", 3): {
        "split_column": {"column": "event_source", "delimiter": "/"},
    },

    # dana_068: T1=plot, T3=export
    ("dana_068", 1): {
        "render_chart": {"chart_type": "bar", "x": "age_group", "y": "satisfaction_score"},
    },
    ("dana_068", 3): {
        "export_dataset": {"format": "xlsx"},
    },

    # dana_069: T1=undo, T3=delete
    ("dana_069", 1): {
        "rollback_dataset": {"steps": 1},
    },
    ("dana_069", 3): {
        "delete_rows": {"condition": "shipping_fee is null"},
    },

    # dana_070: T1=undo, T3=define
    ("dana_070", 1): {
        "rollback_dataset": {"steps": 1},
    },
    ("dana_070", 3): {
        "define_metric": {"name": "attrition_cost", "formula": "annual_salary * 0.33"},
        "apply_formula": {"column": "attrition_cost", "formula": "annual_salary * 0.33"},
    },

    # dana_071: T1=describe, T3=delete
    ("dana_071", 1): {
        "describe_stats": {"table": "email_campaigns"},
    },
    # T3: two operations — delete columns AND delete rows. Both tools needed.
    ("dana_071", 3): {
        "delete_columns": {"columns": ["internal_id", "debug_notes", "etl_batch"]},
        "delete_rows": {"condition": "test_flag is null"},
    },

    # dana_072: T1=replace, T3=dedupe
    ("dana_072", 1): {
        "replace_values": {"column": "business_unit", "find": "Corp Services", "replacement": "Corporate Services"},
    },
    ("dana_072", 3): {
        "dedupe_columns": {"columns": ["business_unit", "month", "line_item"], "keep": "first"},
    },

    # dana_073: T1=interpolate, T3=update
    ("dana_073", 1): {
        "run_interpolation": {"column": "BMI", "method": "model"},
        "execute_python": {"code": "df['BMI'] = df['weight_kg'] / (df['height_cm']/100)**2"},
    },
    # T3: rename column ht_cm → height_cm (header rename)
    ("dana_073", 3): {
        "modify_row": {"prev_values": ["ht_cm"], "new_values": ["height_cm"], "is_header": True},
    },

    # dana_074: T1=format, T3=validate
    ("dana_074", 1): {
        "format_column": {"column": "contact", "type": "phone"},
    },
    ("dana_074", 3): {
        "validate_column": {"column": "contact", "check": "format"},
    },

    # dana_075: T1=delete, T3=insert
    ("dana_075", 1): {
        "delete_rows": {"table": "shipping", "condition": "warehouse is null"},
    },
    ("dana_075", 3): {
        "insert_columns": {"column": "shipping_cost", "default": 12.50},
    },

    # dana_076: T1=lookup, T3=describe
    ("dana_076", 1): {
        "lookup_metric": {"term": "engagement_rate"},
    },
    ("dana_076", 3): {
        "describe_stats": {"table": "tiktok_posts"},
    },

    # dana_077: T1=query, T3=plot
    ("dana_077", 1): {
        "execute_sql": {"query": "SELECT COUNT(*) FROM telemetry WHERE feature='export' AND use_count > 5"},
    },
    ("dana_077", 3): {
        "render_chart": {"chart_type": "histogram", "x": "use_count"},
    },

    # dana_078: T1=segment, T3=query
    ("dana_078", 1): {
        "dimension_breakdown": {"metric": "satisfaction_score", "dimension": "age_group", "aggregation": "avg"},
    },
    ("dana_078", 3): {
        "execute_sql": {"query": "SELECT * FROM survey WHERE age_group = '55+' AND score <= 2"},
    },

    # dana_079: T1=replace, T3=summarize
    ("dana_079", 1): {
        "replace_values": {"column": "region", "find": "W. Coast", "replacement": "West"},
    },
    ("dana_079", 3): {
        "summarize_content": {"focus": "West region vs other regions"},
    },

    # dana_080: T1=split, T3=reshape
    ("dana_080", 1): {
        "split_column": {"column": "location", "delimiter": "-", "names": ["city", "state"]},
    },
    ("dana_080", 3): {
        "pivot_tables": {"index": "employee_id", "columns": "survey_year"},
    },

    # dana_085: T1=exist, T3=query
    ("dana_085", 1): {
        "semantic_layer": {"scope": "column", "name": "bounce_rate"},
        "describe_stats": {"table": "email_campaigns"},
    },
    ("dana_085", 3): {
        "execute_sql": {"query": "SELECT bounce_rate FROM campaigns WHERE campaign = 'Coding Agents'"},
    },
}


def main():
    with open("eval/eval_dana.json") as f:
        data = json.load(f)

    labeled = 0
    missing = 0
    for convo in data:
        if convo["category"] != "switch_flow":
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

    with open("eval/eval_dana.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"Labeled {labeled} turns, {missing} missing")


if __name__ == "__main__":
    main()
