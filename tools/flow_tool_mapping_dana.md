# Dana — Flow → Tool Mapping

All flows also use component tools (+3): `coordinate_context`, `manage_memory`, `read_flow_stack`

---

## Clean (8 flows)

All Clean flows also have access to: `describe_stats` (peek), `semantic_layer`

| Flow | Tools | Domain Tools | Selection logic |
|------|-------|-------------|-----------------|
| update | 9 | `modify_column` · `modify_row` · `modify_cell` · `modify_table` · `execute_python` | Choose by scope: column→bulk column values, row→fix records (is_header=true for header renames), cell→surgical point edits, table→sort/reindex; execute_python for complex transforms |
| datatype | 7 | `cast_column` · `validate_column` | cast_column for type conversion; validate_column to check type-valid values first |
| dedupe | 8 | `dedupe_single_col` · `dedupe_columns` · `execute_python` | Single key column → dedupe_single_col; composite key → dedupe_columns; execute_python for custom dedup logic |
| fill | 7 | `flash_fill` · `execute_python` | flash_fill for standard strategies; execute_python for custom fill logic |
| interpolate | 7 | `run_interpolation` · `execute_python` | run_interpolation for standard methods; execute_python for custom models |
| replace | 8 | `replace_values` · `cut_n_paste` · `execute_python` | Simple find/replace → replace_values; move data blocks → cut_n_paste; execute_python for regex/complex replacements |
| validate | 6 | `validate_column` | 1 call per check type; multiple checks = multiple calls |
| format | 7 | `format_column` · `format_custom` | Built-in patterns → format_column; user-defined → format_custom (requires preview) |

## Transform (8 flows)

All Transform flows also have access to: `describe_stats` (peek), `semantic_layer`

| Flow | Tools | Domain Tools | Selection logic |
|------|-------|-------------|-----------------|
| insert | 8 | `insert_rows` · `insert_columns` · `load_dataset` | Adding rows → insert_rows; adding columns → insert_columns; load_dataset to bring in external data |
| delete | 7 | `delete_rows` · `delete_columns` | Removing rows → delete_rows; removing columns → delete_columns |
| join | 8 | `merge_tables` · `merge_by_key` | merge_by_key for clean FK relationships; merge_tables for complex multi-key or cross joins |
| append | 6 | `append_tables` | Always 1 tool — vertical stacking |
| reshape | 8 | `pivot_tables` · `modify_table` · `cut_n_paste` | Pivot/melt → pivot_tables; sort/reorder → modify_table; move blocks → cut_n_paste |
| merge | 6 | `merge_columns` | Always 1 tool — combines columns |
| split | 6 | `split_column` | Always 1 tool — splits column |
| define | 7 | `define_metric` · `apply_formula` | Save to semantic layer → define_metric; compute the column → apply_formula |

## Analyze (7 flows)

All Analyze flows also have access to: `describe_stats` (peek), `execute_python` (calculate), `semantic_layer`

| Flow | Tools | Domain Tools | Selection logic |
|------|-------|-------------|-----------------|
| query | 7 | `execute_sql` | Always 1 primary tool |
| lookup | 7 | `lookup_metric` | Always 1 primary tool — semantic layer read |
| pivot | 8 | `execute_sql` · `pivot_tables` | SQL for aggregation, pivot_tables for layout |
| describe | 7 | `execute_sql` · `describe_stats` | describe_stats for profiling; SQL for ad-hoc exploration |
| compare | 8 | `compare_metrics` · `compute_correlation` | General group comparison → compare_metrics; relationship strength → compute_correlation |
| exist | 8 | `execute_sql` · `describe_stats` · `semantic_layer` · `list_datasets` | SQL for value checks; describe_stats for column/table presence; semantic_layer for schema-level existence; list_datasets for dataset-level existence |
| segment | 9 | `execute_sql` · `root_cause_analysis` · `dimension_breakdown` | SQL for grouping; root_cause_analysis for why-did-it-change; dimension_breakdown for metric-by-dimension drilldown |

## Report (7 flows)

Plot, trend, summarize, and style also have access to: `describe_stats` (peek)

| Flow | Tools | Domain Tools | Selection logic |
|------|-------|-------------|-----------------|
| plot | 7 | `render_chart` · `execute_sql` · `execute_python` | render_chart to create; execute_sql to prepare data; execute_python for custom transforms |
| trend | 7 | `render_chart` · `compare_metrics` · `execute_python` | render_chart for time-series; compare_metrics for period-over-period significance; execute_python for growth calcs |
| dashboard | 6 | `compose_dashboard` · `render_chart` · `modify_chart` | compose_dashboard for layout; render_chart for each panel; modify_chart to adjust individual panels |
| export | 5 | `export_dataset` · `save_dataset` | export_dataset for download; save_dataset for persisting state |
| summarize | 7 | `summarize_content` · `semantic_layer` · `search_reference` | summarize_content for LLM narrative; semantic_layer for column context; search_reference for domain terminology |
| style | 5 | `apply_style` | Always 1 tool — conditional formatting |
| design | 5 | `modify_chart` · `style_chart` | Labels/type → modify_chart; colors/legend → style_chart |

## Converse (7 flows)

| Flow | Tools | Domain Tools | Selection logic |
|------|-------|-------------|-----------------|
| explain | 4 | `explain_content` | Always 1 tool — LLM-powered |
| chat | 3 | *(component tools only)* | No domain tool — pure conversation |
| preference | 3 | *(manage_memory L2 write)* | Component tool only |
| recommend | 5 | `execute_sql` · `describe_stats` | SQL to probe data; describe_stats for quick stats |
| undo | 4 | `rollback_dataset` | Always 1 tool |
| approve | — | *(routes to target flow's tools)* | Triggers the tool(s) from the approved suggestion |
| reject | 3 | *(component tools only)* | No domain tool — decline and note preference |

## Plan (5 flows)

| Flow | Tools | Domain Tools | Selection logic |
|------|-------|-------------|-----------------|
| insight | — | *(orchestrates Analyze + Report tools)* | No unique tool — chains query, segment, trend, summarize |
| pipeline | — | `load_dataset` · *(orchestrates Clean + Transform tools)* | load_dataset to start; chains dedupe, join, reshape, export |
| blank | 4 | `describe_stats` | Diagnoses nulls across columns |
| issue | 5 | `describe_stats` · `validate_column` | describe_stats for profiling; validate for rule checks |
| outline | — | `load_dataset` · *(orchestrates all intents' tools)* | load_dataset to start; plans across all intents |

## Internal (6 flows)

Internal flows use 2 component tools (`coordinate_context`, `manage_memory`) — no `read_flow_stack`

| Flow | Tools | Domain Tools | Selection logic |
|------|-------|-------------|-----------------|
| recap | 2 | *(manage_memory L1 read)* | Component tool only — session scratchpad |
| calculate | 3 | `execute_python` | Quick arithmetic, date math, unit conversions |
| search | 3 | `search_reference` | Vetted FAQs and curated reference content |
| peek | 4 | `describe_stats` · `heads_or_tails` | describe_stats for profiling; heads_or_tails for head()/tail() preview |
| recall | 2 | *(manage_memory L2 read)* | Component tool only — stored preferences |
| retrieve | 2 | *(manage_memory L3 read)* | Component tool only — unvetted business docs and domain knowledge |

---

## Summary

| Category | Tools | Count |
|----------|-------|-------|
| Infrastructure | list_datasets, load_dataset, save_dataset | 3 |
| Code Execution | execute_sql, execute_python | 2 |
| Modify | modify_column, modify_row, modify_cell, modify_table | 4 |
| Structure | insert_rows, insert_columns, delete_rows, delete_columns, rollback_dataset | 5 |
| Clean | cast_column, dedupe_single_col, dedupe_columns, flash_fill, run_interpolation, replace_values, validate_column, format_column, format_custom | 9 |
| Transform | merge_tables, merge_by_key, pivot_tables, append_tables, split_column, merge_columns, cut_n_paste | 7 |
| Metric / Formula | define_metric, lookup_metric, apply_formula | 3 |
| Analysis | describe_stats, compare_metrics, compute_correlation, semantic_layer, heads_or_tails, root_cause_analysis, dimension_breakdown | 7 |
| Report / Chart | render_chart, modify_chart, style_chart, compose_dashboard, apply_style | 5 |
| Export | export_dataset | 1 |
| Content / LLM | summarize_content, explain_content | 2 |
| Knowledge | search_reference | 1 |
| Component | coordinate_context, manage_memory, read_flow_stack | 3 |
| **Total** | | **52** |

### Coverage

| Intent | Flows | With domain tools | Component-only | Orchestrators |
|--------|-------|--------------------|----------------|---------------|
| Clean | 8 | 8 | 0 | 0 |
| Transform | 8 | 8 | 0 | 0 |
| Analyze | 7 | 7 | 0 | 0 |
| Report | 7 | 7 | 0 | 0 |
| Converse | 7 | 3 | 4 | 0 |
| Plan | 5 | 2 | 0 | 3 |
| Internal | 6 | 3 | 3 | 0 |
| **Total** | **48** | **38** | **7** | **3** |

### Parameter counts (max 5 enforced)

| Params | Tools |
|--------|-------|
| 0 | — |
| 2 | list_datasets, execute_sql, execute_python, rollback_dataset, lookup_metric, explain_content, coordinate_context, read_flow_stack, semantic_layer |
| 3 | load_dataset, save_dataset, insert_rows, delete_rows, delete_columns, dedupe_single_col, dedupe_columns, apply_formula, describe_stats, append_tables, export_dataset, heads_or_tails, merge_by_key |
| 4 | modify_column, modify_row, modify_cell, modify_table, flash_fill, format_column, merge_tables, cut_n_paste, summarize_content, style_chart, manage_memory, search_reference, compute_correlation |
| 5 | insert_columns, cast_column, run_interpolation, replace_values, validate_column, format_custom, pivot_tables, split_column, merge_columns, define_metric, compare_metrics, render_chart, modify_chart, compose_dashboard, apply_style, root_cause_analysis, dimension_breakdown |
