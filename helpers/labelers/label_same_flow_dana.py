"""Add target_tools labels to same_flow Dana eval turns."""

import json

# Map: (convo_id, turn_num) -> target_tools dict
# Each target_tools entry: {tool_name: {param: value, ...}}
# Multiple tools = multiple acceptable answers

LABELS = {
    # dana_001: datatype (Clean) — Q4 sales, order_date is string
    # T1 is clear; T3 is 'specific' ambiguity (target_type unknown) — revisit later
    ("dana_001", 1): {
        "cast_column": {"column": "order_date", "target_type": "datetime"},
    },
    # ("dana_001", 3): SKIP — specific ambiguity, revisit

    # dana_002: pivot (Analyze) — HR attrition, tenure by department × attrition
    ("dana_002", 1): {
        "pivot_tables": {"index": "department", "columns": "attrition_status", "values": "tenure", "aggfunc": "mean"},
    },
    ("dana_002", 3): {
        "pivot_tables": {"index": "job_satisfaction", "columns": "attrition_status", "values": "tenure", "aggfunc": "mean"},
    },

    # dana_003: validate (Clean) — campaign segments, approved list
    ("dana_003", 1): {
        "validate_column": {"column": "segment", "check": "enum", "value": None},
    },
    ("dana_003", 3): {
        "validate_column": {"column": "segment", "check": "enum", "value": ["active", "paused", "completed", "scheduled"]},
    },

    # dana_004: fill (Clean) — P&L, smooth weekly spend with rolling average
    ("dana_004", 1): {
        "flash_fill": {"column": "weekly_spend", "strategy": "rolling_average"},
    },
    ("dana_004", 3): {
        "flash_fill": {"column": "weekly_spend", "strategy": "rolling_average", "value": 3},
    },

    # dana_005: update (Clean) — patient outcomes, rename ER → Emergency in headers
    ("dana_005", 1): {
        "modify_row": {"prev_values": ["ER"], "new_values": ["Emergency"], "is_header": True},
    },
    ("dana_005", 3): {
        "modify_row": {"prev_values": ["Admission Type", "Discharge Date"], "new_values": ["admission type", "discharge date"], "is_header": True},
    },

    # dana_006: dashboard (Report) — SaaS churn, weekly anomaly dashboard
    ("dana_006", 1): {
        "compose_dashboard": {"title": "Weekly Anomaly Report", "panels": None},
    },
    ("dana_006", 3): {
        "compose_dashboard": {"title": None, "panels": None},
        "render_chart": {"chart_type": "bar", "x": "severity", "y": "count"},
    },

    # dana_007: join (Transform) — inventory + shipping data
    ("dana_007", 1): {
        "merge_tables": {"left": "inventory", "right": "shipping", "key": None},
        "merge_by_key": {"left": "inventory", "right": "shipping", "key": None},
    },
    ("dana_007", 3): {
        "merge_tables": {"left": "inventory", "right": "shipping", "key": ["warehouse_id", "order_id"]},
        "merge_by_key": {"left": "inventory", "right": "shipping", "key": ["warehouse_id", "order_id"]},
    },

    # dana_008: split (Transform) — social media, split username on underscore
    ("dana_008", 1): {
        "split_column": {"column": "username", "delimiter": "_"},
    },
    ("dana_008", 3): {
        "split_column": {"column": "username", "delimiter": "_", "names": ["platform", "account_name"]},
    },

    # dana_033: delete (Transform) — telemetry, drop zero-duration sessions
    ("dana_033", 1): {
        "delete_rows": {"condition": "session_duration == 0"},
    },
    ("dana_033", 3): {
        "delete_rows": {"condition": "feature_id is null"},
    },

    # dana_034: datatype (Clean) — survey data, check column types
    ("dana_034", 1): {
        "describe_stats": {"table": "survey"},
        "validate_column": {"column": None, "check": "type"},
    },
    ("dana_034", 3): {
        "cast_column": {"column": "satisfaction_score", "target_type": "integer"},
    },

    # dana_035: lookup (Analyze) — Q4 sales, semantic layer formula lookup
    ("dana_035", 1): {
        "lookup_metric": {"term": "gross_margin"},
    },
    ("dana_035", 3): {
        "lookup_metric": {"term": "customer_lifetime_value"},
    },

    # dana_036: merge (Transform) — HR, combine name columns
    ("dana_036", 1): {
        "merge_columns": {"columns": ["first_name", "last_name"], "name": "full_name", "separator": " "},
    },
    ("dana_036", 3): {
        "merge_columns": {"columns": ["department", "job_role"], "name": None, "separator": "-"},
    },

    # dana_037: fill (Clean) — campaigns, forward fill open_rate
    ("dana_037", 1): {
        "flash_fill": {"column": "open_rate", "strategy": "forward_fill"},
    },
    ("dana_037", 3): {
        "flash_fill": {"column": "click_through_rate", "strategy": "rolling_average", "value": 3},
    },

    # dana_038: join (Transform) — P&L, join revenue + cost tables
    ("dana_038", 1): {
        "merge_tables": {"left": "revenue", "right": "cost", "key": ["business_unit_id", "month"], "how": "inner"},
    },
    ("dana_038", 3): {
        "merge_tables": {"right": "headcount", "key": ["business_unit_id", "month"]},
    },

    # dana_039: replace (Clean) — hospital names standardization
    ("dana_039", 1): {
        "replace_values": {"column": "hospital", "find": "St Marys", "replacement": "St. Mary's"},
    },
    ("dana_039", 3): {
        "replace_values": {"column": "hospital", "find": "Mercy Gen", "replacement": "Mercy General"},
    },

    # dana_040: trend (Report) — churn over time
    ("dana_040", 1): {
        "render_chart": {"chart_type": "line", "x": "month", "y": "churn_rate"},
        "compare_metrics": {"column_a": "churn_rate", "column_b": None, "method": "trend"},
    },
    ("dana_040", 3): {
        "render_chart": {"chart_type": "line", "x": "quarter", "y": "churn_rate"},
    },

    # dana_041: format (Clean) — warehouse phone numbers
    ("dana_041", 1): {
        "format_column": {"column": "phone", "type": "phone"},
    },
    ("dana_041", 3): {
        "format_custom": {"column": "shipping_address", "pattern": None},
    },

    # dana_042: define (Transform) — social media, save engagement rate formula
    ("dana_042", 1): {
        "define_metric": {"name": "engagement_rate", "formula": "(likes + comments + shares) / impressions"},
    },
    ("dana_042", 3): {
        "define_metric": {"name": "virality", "formula": "shares / impressions * 100"},
    },

    # dana_043: plot (Report) — feature adoption bar chart
    ("dana_043", 1): {
        "render_chart": {"chart_type": "bar", "x": "feature", "y": "user_count"},
    },
    ("dana_043", 3): {
        "render_chart": {"chart_type": "bar", "x": "feature", "y": "user_count"},
    },

    # dana_045: segment (Analyze) — Q4 returns by product category
    ("dana_045", 1): {
        "dimension_breakdown": {"metric": "returns", "dimension": "product_category"},
    },
    ("dana_045", 3): {
        "dimension_breakdown": {"metric": "returns", "dimension": "region"},
    },

    # dana_046: reshape (Transform) — flip attrition table
    ("dana_046", 1): {
        "pivot_tables": {"columns": "department"},
    },
    ("dana_046", 3): {
        "pivot_tables": {"index": "tenure_band", "columns": "department", "values": "attrition_rate"},
    },

    # dana_047: describe (Analyze) — campaign data profile
    ("dana_047", 1): {
        "describe_stats": {"table": "campaigns"},
    },
    ("dana_047", 3): {
        "describe_stats": {"table": "campaigns", "column": "open_rate"},
    },

    # dana_048: explain (Converse) — walk through P&L calculation
    ("dana_048", 1): {
        "explain_content": {"topic": "consolidated P&L calculation"},
    },
    ("dana_048", 3): {
        "explain_content": {"topic": "intercompany eliminations"},
    },

    # dana_049: insert (Transform) — add risk_flag column
    ("dana_049", 1): {
        "insert_columns": {"column": "risk_flag", "expression": "IF(age > 65, 'High Risk', 'Standard')"},
    },
    ("dana_049", 3): {
        "insert_columns": {"column": "readmission_flag", "expression": "IF(readmission_days < 30, 'Critical', 'Standard')"},
    },

    # dana_050: summarize (Report) — chart takeaways
    ("dana_050", 1): {
        "summarize_content": {"artifact_type": "chart", "focus": "churn by plan tier"},
    },
    ("dana_050", 3): {
        "summarize_content": {"artifact_type": "table", "focus": "usage metrics"},
    },

    # dana_051: segment (Analyze) — shipping delay by warehouse
    ("dana_051", 1): {
        "dimension_breakdown": {"metric": "shipping_delay", "dimension": "warehouse", "aggregation": "avg"},
    },
    ("dana_051", 3): {
        "dimension_breakdown": {"metric": "shipping_delay", "dimension": "carrier", "aggregation": "avg"},
    },

    # dana_052: pivot (Analyze) — engagement rate by platform × content type
    ("dana_052", 1): {
        "pivot_tables": {"index": "content_type", "columns": "platform", "values": "engagement_rate", "aggfunc": "mean"},
    },
    ("dana_052", 3): {
        "pivot_tables": {"index": "content_type", "columns": "platform", "values": "impressions", "aggfunc": "sum"},
    },

    # dana_053: dedupe (Clean) — duplicated users in telemetry
    ("dana_053", 1): {
        "dedupe_single_col": {"column": "user_id"},
        "dedupe_columns": {"columns": ["user_id", "session_id"]},
    },
    ("dana_053", 3): {
        "dedupe_single_col": {"column": "device_id"},
    },

    # dana_054: interpolate (Clean) — estimate missing age from other columns
    ("dana_054", 1): {
        "run_interpolation": {"column": "age", "method": "model"},
    },
    ("dana_054", 3): {
        "run_interpolation": {"column": "household_size"},
    },

    # dana_055: validate (Clean) — check region for weird values
    ("dana_055", 1): {
        "validate_column": {"column": "region", "check": "enum"},
    },
    ("dana_055", 3): {
        "validate_column": {"column": "order_status", "check": "enum", "value": ["shipped", "delivered", "returned", "cancelled"]},
    },

    # dana_056: datatype (Clean) — revenue column is text
    ("dana_056", 1): {
        "cast_column": {"column": "revenue_total", "target_type": "numeric"},
    },
    ("dana_056", 3): {
        "cast_column": {"column": "cost", "target_type": "numeric"},
    },

    # dana_090: define (Transform) — engagement rate formula (duplicate of 042 scenario)
    ("dana_090", 1): {
        "define_metric": {"name": "engagement_rate", "formula": "(likes + comments + shares) / impressions"},
    },
    ("dana_090", 3): {
        "define_metric": {"name": "engagement_rate", "formula": "(likes + comments + shares) / impressions"},
    },
}


def main():
    with open("datasets/dana/eval_set.json") as f:
        data = json.load(f)

    labeled = 0
    for convo in data:
        if convo["category"] != "same_flow":
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

    with open("datasets/dana/eval_set.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"Labeled {labeled} turns across same_flow category")


if __name__ == "__main__":
    main()
