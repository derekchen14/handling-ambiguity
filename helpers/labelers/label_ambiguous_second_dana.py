"""Add target_tools labels to ambiguous_second Dana eval turns.

Pattern:
  T1: clear flow, standard tool assignment.
  T3: flow=outline (Plan), multiple candidate_flows — user requests
      2+ operations at once. Include tools for ALL operations.
"""

import json

LABELS = {
    # dana_025: T1=describe, T3=outline [fill, trend]
    ("dana_025", 1): {
        "describe_stats": {"column": "session_duration"},
    },
    # T3: "Fill the nulls with the median so I can see the session duration over time."
    ("dana_025", 3): {
        "flash_fill": {"column": "session_duration", "strategy": "median"},
        "render_chart": {"chart_type": "line", "x": None, "y": "session_duration"},
        "compare_metrics": {"column_a": "session_duration", "method": "trend"},
    },

    # dana_026: T1=plot, T3=outline [replace, update]
    ("dana_026", 1): {
        "render_chart": {"chart_type": "bar", "x": "department", "y": "attrition"},
    },
    # T3: fix typos "Slaes"→"Sales", "Resarch"→"Research", then lowercase
    ("dana_026", 3): {
        "replace_values": {"column": "department", "find": "Slaes", "replacement": "Sales"},
        "modify_column": {"column": "department", "value": None},
        "format_column": {"column": "department", "type": "lower"},
    },

    # dana_027: T1=interpolate, T3=outline [split, compare]
    ("dana_027", 1): {
        "run_interpolation": {"column": "satisfaction_score"},
    },
    # T3: split respondent_id on underscore, then compare scores by region
    ("dana_027", 3): {
        "split_column": {"column": "respondent_id", "delimiter": "_"},
        "compare_metrics": {"column_a": "satisfaction_score", "group_by": "region"},
    },

    # dana_028: T1=merge, T3=outline [validate, plot]
    ("dana_028", 1): {
        "merge_columns": {"columns": ["unit", "currency"], "name": None},
    },
    # T3: chart DAUs per quarter for valid (active) subscription types
    ("dana_028", 3): {
        "validate_column": {"column": "subscription_type", "check": "in_list"},
        "render_chart": {"chart_type": None, "x": "quarter", "y": "DAU"},
        "execute_sql": {"query": None},
    },

    # dana_029: T1=segment, T3=outline [format, dashboard]
    ("dana_029", 1): {
        "dimension_breakdown": {"metric": "readmission_rate", "dimension": "hospital_site"},
    },
    # T3: clean dates, then set up weekly readmission report
    ("dana_029", 3): {
        "format_column": {"column": None, "type": "date"},
        "compose_dashboard": {"title": "Weekly Readmission Report", "panels": None},
    },

    # dana_030: T1=export, T3=outline [update, describe]
    ("dana_030", 1): {
        "export_dataset": {"format": "excel"},
        "save_dataset": {"format": "excel"},
    },
    # T3: rename column to ChurnStatus, how many unique values
    ("dana_030", 3): {
        "modify_row": {"is_header": True, "prev_values": None, "new_values": ["ChurnStatus"]},
        "describe_stats": {"column": None},
    },

    # dana_031: T1=delete, T3=outline [datatype, style]
    ("dana_031", 1): {
        "delete_rows": {"condition": "shipping_status is null"},
    },
    # T3: fix sendoff_times to timestamps, then color-code by delivery status
    ("dana_031", 3): {
        "cast_column": {"column": "sendoff_times", "target_type": "datetime"},
        "apply_style": {"column": "delivery_status", "condition": None, "format": "highlight"},
    },

    # dana_032: T1=append, T3=outline [dedupe, describe]
    ("dana_032", 1): {
        "append_tables": {"source": None, "target": "outreach"},
    },
    # T3: dupes observed, how many rows now
    ("dana_032", 3): {
        "dedupe_columns": {"columns": None},
        "dedupe_single_col": {"column": None},
        "describe_stats": {"table": None},
    },

    # dana_105: T1=query, T3=outline [compare, delete]
    ("dana_105", 1): {
        "execute_sql": {"query": None},
    },
    # T3: drop bottom 3 features, compare top one vs Search week over week
    ("dana_105", 3): {
        "delete_rows": {"condition": None},
        "compare_metrics": {"column_a": None, "column_b": None},
        "render_chart": {"chart_type": "line", "x": "week", "y": None},
    },

    # dana_106: T1=trend, T3=outline [segment, style]
    ("dana_106", 1): {
        "render_chart": {"chart_type": "line", "x": "month", "y": "satisfaction_score"},
        "compare_metrics": {"column_a": "satisfaction_score", "method": "trend"},
    },
    # T3: break score down by age group, red cells where < 3.5
    ("dana_106", 3): {
        "dimension_breakdown": {"metric": "satisfaction_score", "dimension": "age_group"},
        "apply_style": {"condition": "value < 3.5", "format": "highlight", "color": "red"},
    },

    # dana_107: T1=plot, T3=outline [delete, update]
    ("dana_107", 1): {
        "render_chart": {"chart_type": "bar", "x": "region", "y": "revenue"},
    },
    # T3: delete null-region rows, rename region_cd → region
    ("dana_107", 3): {
        "delete_rows": {"condition": "region is null"},
        "modify_row": {"is_header": True, "prev_values": ["region_cd"], "new_values": ["region"]},
    },

    # dana_108: T1=split, T3=outline [reject, lookup]
    ("dana_108", 1): {
        "split_column": {"column": "location", "delimiter": "-", "names": ["city", "state"]},
    },
    # T3: reject suggestion + ask about attrition risk score formula
    ("dana_108", 3): {
        "no_tool_needed": {"reason": "user rejected suggestion to fix 13 unparsed rows"},
        "lookup_metric": {"term": "attrition_risk_score"},
    },

    # dana_109: T1=validate, T3=outline [export, approve]
    ("dana_109", 1): {
        "validate_column": {"column": "campaign_type", "check": "in_list", "value": "Drip,Blast,Nurture,Welcome"},
    },
    # T3: approve cleanup + export as CSV
    # Approve maps to the corrective action (fixing invalid campaign_type values)
    ("dana_109", 3): {
        "replace_values": {"column": "campaign_type", "find": None, "replacement": None},
        "export_dataset": {"format": "csv"},
    },

    # dana_110: T1=plot, T3=outline [replace, trend]
    ("dana_110", 1): {
        "render_chart": {"chart_type": "bar", "x": "month", "y": "net_income", "color": "BU"},
    },
    # T3: fix "Ret." → "Retail" in BU column, then show MoM net income growth
    ("dana_110", 3): {
        "replace_values": {"column": "BU", "find": "Ret.", "replacement": "Retail"},
        "render_chart": {"chart_type": "line", "x": "month", "y": "net_income"},
        "compare_metrics": {"column_a": "net_income", "method": "trend"},
    },

    # dana_111: T1=lookup, T3=outline [compare, approve]
    ("dana_111", 1): {
        "lookup_metric": {"term": "30_day_readmission_rate"},
    },
    # T3: approve agent's suggestion + compare Site A vs Site B mortality
    # Approve: agent showed formula, likely suggested applying it
    ("dana_111", 3): {
        "apply_formula": {"column": None, "formula": None},
        "execute_sql": {"query": None},
        "compare_metrics": {"column_a": "mortality_rate", "group_by": "site"},
    },

    # dana_112: T1=define, T3=outline [validate, replace]
    ("dana_112", 1): {
        "define_metric": {"name": "churn_rate", "formula": "cancelled_subscriptions / total_subscriptions"},
    },
    # T3: fix "premum" → "premium", check for other bad plan_type values
    ("dana_112", 3): {
        "replace_values": {"column": "plan_type", "find": "premum", "replacement": "premium"},
        "validate_column": {"column": "plan_type", "check": "in_list"},
    },

    # dana_113: T1=approve, T3=outline [update, trend]
    # T1: "Do it." — confirms stockout risk analysis
    ("dana_113", 1): {
        "execute_sql": {"query": None},
        "execute_python": {"code": None},
        "dimension_breakdown": {"metric": None, "dimension": "warehouse"},
    },
    # T3: fix "Newark" → "Newark NJ", show stockout counts week over week
    ("dana_113", 3): {
        "replace_values": {"column": None, "find": "Newark", "replacement": "Newark NJ"},
        "modify_column": {"column": None, "value": None, "condition": None},
        "render_chart": {"chart_type": "line", "x": "week", "y": "stockout_count"},
    },

    # dana_114: T1=dedupe, T3=outline [delete, reshape]
    ("dana_114", 1): {
        "dedupe_single_col": {"column": "post_id"},
        "dedupe_columns": {"columns": ["post_id"]},
    },
    # T3: drop TikTok rows, pivot so each platform is its own column
    ("dana_114", 3): {
        "delete_rows": {"condition": "platform = 'TikTok'"},
        "pivot_tables": {"columns": "platform"},
    },

    # dana_115: T1=merge, T3=outline [append, approve]
    ("dana_115", 1): {
        "merge_columns": {"columns": ["os_name", "os_version"], "name": "os", "separator": "-"},
    },
    # T3: stack last month's data + approve suggested dedup
    ("dana_115", 3): {
        "append_tables": {"source": None, "target": None},
        "dedupe_single_col": {"column": None},
        "dedupe_columns": {"columns": None},
    },

    # dana_116: T1=chat, T3=outline [trend, reshape]
    ("dana_116", 1): {
        "no_tool_needed": {"reason": "conversational question about sample size"},
        "explain_content": {"topic": "sample size adequacy"},
    },
    # T3: flip question columns into rows (unpivot), show avg satisfaction MoM
    ("dana_116", 3): {
        "pivot_tables": {"table": None},
        "render_chart": {"chart_type": "line", "x": "month", "y": "satisfaction_score"},
    },

    # dana_117: T1=insert, T3=outline [validate, summarize, plot]
    # REVISIT: 3 candidate flows. "How does monthly revenue look" = summarize or plot.
    ("dana_117", 1): {
        "insert_columns": {"column": "profit", "expression": "revenue - cost"},
        "apply_formula": {"column": "profit", "formula": "revenue - cost"},
    },
    # T3: flag bad regions + show monthly revenue by region (chart or summary)
    ("dana_117", 3): {
        "validate_column": {"column": "region", "check": None, "action": "flag"},
        "render_chart": {"chart_type": "line", "x": "month", "y": "revenue", "color": "region"},
        "summarize_content": {"focus": "monthly revenue by region"},
    },

    # dana_118: T1=dashboard, T3=outline [chat, format]
    ("dana_118", 1): {
        "compose_dashboard": {"title": "Weekly Attrition Report", "panels": None},
    },
    # T3: clean hire_date format + ask about healthy attrition rate
    ("dana_118", 3): {
        "format_column": {"column": "hire_date", "type": "date"},
        "no_tool_needed": {"reason": "conversational question about attrition benchmarks"},
        "explain_content": {"topic": "healthy attrition rate"},
    },

    # dana_119: T1=replace, T3=outline [lookup, dedupe]
    ("dana_119", 1): {
        "replace_values": {"column": "segment_name", "find": "Email - Promo", "replacement": "Promotional Email"},
    },
    # T3: dedupe campaign_id + segment, look up engagement_rate formula
    ("dana_119", 3): {
        "dedupe_columns": {"columns": ["campaign_id", "segment"]},
        "lookup_metric": {"term": "engagement_rate"},
    },

    # dana_120: T1=summarize, T3=outline [split, describe]
    ("dana_120", 1): {
        "summarize_content": {"artifact_type": "table", "focus": "monthly P&L"},
    },
    # T3: split BU_period on dash, full stats on table
    ("dana_120", 3): {
        "split_column": {"column": "BU_period", "delimiter": "-"},
        "describe_stats": {"table": None},
    },

    # dana_121: T1=design, T3=outline [insert, approve]
    ("dana_121", 1): {
        "style_chart": {"chart_id": None, "colors": None},
        "modify_chart": {"chart_id": None},
    },
    # T3: add length_of_stay column + approve outlier cleanup
    ("dana_121", 3): {
        "insert_columns": {"column": "length_of_stay_days", "expression": "discharge_date - admission_date"},
        "apply_formula": {"column": "length_of_stay_days", "formula": "discharge_date - admission_date"},
        "delete_rows": {"condition": None},
    },

    # dana_122: T1=format, T3=outline [merge, explain]
    ("dana_122", 1): {
        "format_column": {"column": "contact", "type": "phone"},
    },
    # T3: combine first_name + last_name, explain why 23 were left unchanged
    ("dana_122", 3): {
        "merge_columns": {"columns": ["first_name", "last_name"], "name": "full_name", "separator": " "},
        "explain_content": {"topic": "why 23 entries were left unchanged"},
    },

    # dana_123: T1=pivot, T3=outline [export, update]
    ("dana_123", 1): {
        "pivot_tables": {"index": "warehouse", "columns": "month", "values": "shipment_volume"},
    },
    # T3: fix header "Chicago_IL" → "CHI", export as Excel
    ("dana_123", 3): {
        "modify_row": {"is_header": True, "prev_values": ["Chicago_IL"], "new_values": ["CHI"]},
        "replace_values": {"column": None, "find": "Chicago_IL", "replacement": "CHI"},
        "export_dataset": {"format": "excel"},
    },

    # dana_124: T1=query, T3=outline [delete, interpolate]
    ("dana_124", 1): {
        "execute_sql": {"query": None},
    },
    # T3: fill impressions from reach + engagement rate, drop story_views column
    ("dana_124", 3): {
        "run_interpolation": {"column": "impressions"},
        "delete_columns": {"columns": ["story_views"]},
    },

    # dana_125: T1=preference, T3=outline [replace, format]
    ("dana_125", 1): {
        "manage_memory": {"operation": "write", "level": "L2", "key": "date_format", "value": "YYYY-MM-DD"},
    },
    # T3: "web app" → "web", clean up messy timestamps
    ("dana_125", 3): {
        "replace_values": {"column": None, "find": "web app", "replacement": "web"},
        "format_column": {"column": None, "type": "date"},
    },

    # dana_126: T1=join, T3=outline [merge, describe]
    ("dana_126", 1): {
        "merge_tables": {"left": "survey_responses", "right": "customer_demographics", "key": "customer_id"},
        "merge_by_key": {"left": "survey_responses", "right": "customer_demographics", "key": "customer_id"},
    },
    # T3: combine first_name + last_name into full_name, overall stats
    ("dana_126", 3): {
        "merge_columns": {"columns": ["first_name", "last_name"], "name": "full_name", "separator": " "},
        "describe_stats": {"table": None},
    },

    # dana_127: T1=datatype, T3=outline [segment, compare]
    ("dana_127", 1): {
        "cast_column": {"column": "order_date", "target_type": "datetime"},
        "describe_stats": {"table": None},
        "validate_column": {"column": None, "check": None},
    },
    # T3: revenue by month, August vs July
    ("dana_127", 3): {
        "dimension_breakdown": {"metric": "revenue", "dimension": "month"},
        "compare_metrics": {"column_a": "revenue", "group_by": "month"},
        "execute_sql": {"query": None},
    },

    # dana_128: T1=describe, T3=outline [fill, compare]
    ("dana_128", 1): {
        "describe_stats": {"table": "attrition"},
    },
    # T3: forward-fill satisfaction_score gaps, compare remote vs onsite attrition
    ("dana_128", 3): {
        "flash_fill": {"column": "satisfaction_score", "strategy": "ffill"},
        "compare_metrics": {"column_a": "attrition_rate", "group_by": "work_location"},
        "dimension_breakdown": {"metric": "attrition_rate", "dimension": "work_location"},
    },
}


def main():
    with open("datasets/dana/eval_set.json") as f:
        data = json.load(f)

    labeled = 0
    missing = 0
    for convo in data:
        if convo["category"] != "ambiguous_second":
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
