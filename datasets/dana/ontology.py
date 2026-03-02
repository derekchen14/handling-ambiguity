from enum import Enum


class Intent(str, Enum):
    PLAN = 'Plan'
    CONVERSE = 'Converse'
    INTERNAL = 'Internal'
    CLEAN = 'Clean'          # update, cast types, deduplicate, fill, replace, validate, format
    TRANSFORM = 'Transform'  # insert, delete, join, reshape, merge, split, define
    ANALYZE = 'Analyze'      # query, describe, compare, segment, lookup, pivot, exist
    REPORT = 'Report'        # plot, trend, dashboard, export, explain, style, design


class FlowLifecycle(str, Enum):
    PENDING = 'Pending'
    ACTIVE = 'Active'
    COMPLETED = 'Completed'
    INVALID = 'Invalid'


class AmbiguityLevel(str, Enum):
    GENERAL = 'general'
    PARTIAL = 'partial'
    SPECIFIC = 'specific'
    CONFIRMATION = 'confirmation'


DACT_CATALOG = {
    'chat':     {'hex': '0', 'pos': 'noun'},
    'query':    {'hex': '1', 'pos': 'verb'},
    'measure':  {'hex': '2', 'pos': 'verb'},
    'plot':     {'hex': '3', 'pos': 'verb'},
    'retrieve': {'hex': '4', 'pos': 'verb'},
    'insert':   {'hex': '5', 'pos': 'verb'},
    'update':   {'hex': '6', 'pos': 'verb'},
    'delete':   {'hex': '7', 'pos': 'verb'},
    'user':     {'hex': '8', 'pos': 'adj'},
    'agent':    {'hex': '9', 'pos': 'adj'},
    'table':    {'hex': 'A', 'pos': 'noun'},
    'row':      {'hex': 'B', 'pos': 'noun'},
    'column':   {'hex': 'C', 'pos': 'noun'},
    'multiple': {'hex': 'D', 'pos': 'adj'},
    'confirm':  {'hex': 'E', 'pos': 'adj'},
    'deny':     {'hex': 'F', 'pos': 'adj'},
}

FLOW_CATALOG = {

    # ── Clean (8 flows) ────────────────────────────────────────────

    'update': {
        'dax': '{006}',
        'intent': Intent.CLEAN,
        'description': 'Modify cell values, column types, or column names in place — the catch-all for simple edits like renaming a column, changing a single value, or recasting a type. Use validate or format for bulk corrections',
        'slots': {
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
            'column': {'type': 'SourceSlot', 'entity_part': 'column', 'priority': 'required'},
            'row': {'type': 'SourceSlot', 'entity_part': 'row', 'priority': 'optional'},
            'value': {'type': 'FreeTextSlot', 'priority': 'required'},
        },
        'output': 'toast',
        'edge_flows': ['replace', 'delete', 'insert'],
        'policy_path': 'policies.clean.update',
    },
    'datatype': {
        'dax': '{06E}',
        'intent': Intent.CLEAN,
        'description': 'Validate and cast column data types — string to date, int to float, object to category. Auto-detects mismatched types and proposes corrections',
        'slots': {
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
            'column': {'type': 'SourceSlot', 'entity_part': 'column', 'priority': 'required'},
            'type': {'type': 'CategorySlot', 'priority': 'required'},
        },
        'output': 'toast',
        'edge_flows': ['validate', 'update'],
        'policy_path': 'policies.clean.datatype',
    },
    'dedupe': {
        'dax': '{7BD}',
        'intent': Intent.CLEAN,
        'description': 'Remove duplicate rows based on one or more key columns — keeps first, last, or none of the duplicates. Reports how many rows were removed',
        'slots': {
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
            'key_columns': {'type': 'SourceSlot', 'entity_part': 'column', 'priority': 'optional'},
        },
        'output': 'toast',
        'edge_flows': ['fill', 'update'],
        'policy_path': 'policies.clean.dedupe',
    },
    'fill': {
        'dax': '{5BD}',
        'intent': Intent.CLEAN,
        'description': 'Flash fill — each cell looks at the row(s) above it to decide the new value. Forward fill, backward fill, rolling average, or carry-down. Operates row-wise; use interpolate when inferring from neighboring columns',
        'slots': {
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
            'column': {'type': 'SourceSlot', 'entity_part': 'column', 'priority': 'elective'},
            'strategy': {'type': 'CategorySlot', 'priority': 'elective'},
        },
        'output': 'toast',
        'edge_flows': ['interpolate'],
        'policy_path': 'policies.clean.fill',
    },
    'interpolate': {
        'dax': '{02B}',
        'intent': Intent.CLEAN,
        'description': 'Estimate missing values by looking at surrounding columns to infer what the cell should be — e.g., infer state from city, or use numerical methods like linear or spline interpolation. Operates column-wise; use fill when carrying values down from rows above',
        'slots': {
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
            'column': {'type': 'SourceSlot', 'entity_part': 'column', 'priority': 'required'},
            'method': {'type': 'CategorySlot', 'priority': 'optional'},
        },
        'output': 'toast',
        'edge_flows': ['fill'],
        'policy_path': 'policies.clean.interpolate',
    },
    'replace': {
        'dax': '{04C}',
        'intent': Intent.CLEAN,
        'description': 'Find and replace values across a cell or column — supports exact match, regex patterns, and case-insensitive substitution. Changes that affect one row or one record are better served by update. For bulk corrections against a valid set, use validate instead',
        'slots': {
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
            'column': {'type': 'SourceSlot', 'entity_part': 'column', 'priority': 'required'},
            'find': {'type': 'ExactSlot', 'priority': 'required'},
            'replacement': {'type': 'ExactSlot', 'priority': 'required'},
        },
        'output': 'toast',
        'edge_flows': ['update', 'validate'],
        'policy_path': 'policies.clean.replace',
    },
    'validate': {
        'dax': '{16E}',
        'intent': Intent.CLEAN,
        'description': 'Check that values belong to a valid set of options — enum constraints, allowed ranges, and business rules. Flags violations but does not fix formatting; use format for correcting form (emails, dates, phones)',
        'slots': {
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
            'column': {'type': 'SourceSlot', 'entity_part': 'column', 'priority': 'elective'},
            'rules': {'type': 'GroupSlot', 'priority': 'elective'},
        },
        'output': 'toast',
        'edge_flows': ['format', 'datatype'],
        'policy_path': 'policies.clean.validate',
    },
    'format': {
        'dax': '{06B}',
        'intent': Intent.CLEAN,
        'description': 'Normalize values into the correct form — standardize emails, phone numbers, addresses, dates, or apply custom patterns. Fixes formatting, not validity; use validate to check whether values are allowed',
        'slots': {
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
            'column': {'type': 'SourceSlot', 'entity_part': 'column', 'priority': 'required'},
            'pattern': {'type': 'CategorySlot', 'priority': 'required'},
        },
        'output': 'toast',
        'edge_flows': ['validate', 'update'],
        'policy_path': 'policies.clean.format',
    },

    # ── Transform (8 flows) ────────────────────────────────────────

    'insert': {
        'dax': '{005}',
        'intent': Intent.TRANSFORM,
        'description': 'Add a new row or column to the table — a computed column from an expression, an empty column with a default, or a manually specified row of values',
        'slots': {
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
            'column': {'type': 'TargetSlot', 'entity_part': 'column', 'priority': 'elective'},
            'row': {'type': 'TargetSlot', 'entity_part': 'row', 'priority': 'elective'},
        },
        'output': 'toast',
        'edge_flows': ['join', 'append', 'merge'],
        'policy_path': 'policies.transform.insert',
    },
    'delete': {
        'dax': '{007}',
        'intent': Intent.TRANSFORM,
        'description': 'Remove rows or columns from the table — delete by name, index, or condition (e.g., drop rows where revenue is null, remove the temp_id column). Requires confirmation',
        'slots': {
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
            'target': {'type': 'RemovalSlot', 'priority': 'required'},
        },
        'output': 'confirmation',
        'edge_flows': ['insert', 'replace'],
        'policy_path': 'policies.transform.delete',
    },
    'join': {
        'dax': '{05A}',
        'intent': Intent.TRANSFORM,
        'description': 'Combine two tables on a shared key column — left, inner, outer, or cross join. Produces a wider table with columns from both sources',
        'slots': {
            'left': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
            'right': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
            'key': {'type': 'SourceSlot', 'entity_part': 'column', 'priority': 'required'},
            'how': {'type': 'CategorySlot', 'priority': 'optional'},
        },
        'output': 'table',
        'edge_flows': ['append', 'merge'],
        'policy_path': 'policies.transform.join',
    },
    'append': {
        'dax': '{05B}',
        'intent': Intent.TRANSFORM,
        'description': 'Stack rows from one table onto another vertically — columns must align by name or position. Use join when combining tables side-by-side on a key',
        'slots': {
            'source': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
            'target': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
        },
        'output': 'toast',
        'edge_flows': ['join', 'insert'],
        'policy_path': 'policies.transform.append',
    },
    'reshape': {
        'dax': '{06A}',
        'intent': Intent.TRANSFORM,
        'description': 'Restructure the table layout — pivot (long to wide), unpivot/melt (wide to long), or transpose (swap rows and columns). Changes shape without changing data values',
        'slots': {
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
            'method': {'type': 'CategorySlot', 'priority': 'required'},
        },
        'output': 'table',
        'edge_flows': ['pivot', 'describe'],
        'policy_path': 'policies.transform.reshape',
    },
    'merge': {
        'dax': '{56C}',
        'intent': Intent.TRANSFORM,
        'description': 'Combine two or more columns into one — concatenate with a separator or apply a custom expression (e.g., first_name + " " + last_name -> full_name)',
        'slots': {
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
            'columns': {'type': 'SourceSlot', 'entity_part': 'column', 'min_size': 2, 'priority': 'required'},
            'name': {'type': 'TargetSlot', 'entity_part': 'column', 'priority': 'required'},
        },
        'output': 'toast',
        'edge_flows': ['split'],
        'policy_path': 'policies.transform.merge',
    },
    'split': {
        'dax': '{5CD}',
        'intent': Intent.TRANSFORM,
        'description': 'Split one column into multiple new columns by a delimiter or pattern — e.g., "full_name" on space becomes first_name and last_name. Inverse of merge',
        'slots': {
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
            'column': {'type': 'SourceSlot', 'entity_part': 'column', 'priority': 'required'},
            'delimiter': {'type': 'ExactSlot', 'priority': 'optional'},
        },
        'output': 'toast',
        'edge_flows': ['merge', 'update'],
        'policy_path': 'policies.transform.split',
    },
    'define': {
        'dax': '{28C}',
        'intent': Intent.TRANSFORM,
        'description': 'Create a named, reusable metric formula saved to the semantic layer — e.g., profit = revenue - cost, conversion_rate = purchases / visits. Referenced by lookup and query',
        'slots': {
            'name': {'type': 'TargetSlot', 'entity_part': 'column', 'priority': 'required'},
            'formula': {'type': 'FreeTextSlot', 'priority': 'required'},
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'optional'},
        },
        'output': 'toast',
        'edge_flows': ['lookup', 'query'],
        'policy_path': 'policies.transform.define',
    },

    # ── Analyze (7 flows) ──────────────────────────────────────────

    'query': {
        'dax': '{001}',
        'intent': Intent.ANALYZE,
        'description': 'Run a SQL-like query against the data — supports SELECT, WHERE, GROUP BY, HAVING, ORDER BY, and aggregations. The general-purpose analysis tool when no specialized flow fits',
        'slots': {
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
            'query': {'type': 'FreeTextSlot', 'priority': 'optional'},
        },
        'output': 'table',
        'edge_flows': ['pivot', 'describe', 'lookup', 'plot', 'compare', 'segment'],
        'policy_path': 'policies.analyze.query',
    },
    'lookup': {
        'dax': '{002}',
        'intent': Intent.ANALYZE,
        'description': 'Find the definition and value of a metric, term, or concept in the semantic layer — returns the formula, source columns, and business context. The structured equivalent of search for vetted definitions. Can also apply basic filters.',
        'slots': {
            'term': {'type': 'FreeTextSlot', 'priority': 'required'},
            'source': {'type': 'FreeTextSlot', 'priority': 'optional'},
        },
        'output': 'card',
        'edge_flows': ['query', 'describe', 'define'],
        'policy_path': 'policies.analyze.lookup',
    },
    'pivot': {
        'dax': '{01A}',
        'intent': Intent.ANALYZE,
        'description': 'Cross-tabulate data by two dimensions — e.g., sales by region and quarter. Produces a matrix with row and column headers and aggregated values in each cell',
        'slots': {
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
            'row_dim': {'type': 'SourceSlot', 'entity_part': 'column', 'priority': 'required'},
            'col_dim': {'type': 'SourceSlot', 'entity_part': 'column', 'priority': 'required'},
        },
        'output': 'table',
        'edge_flows': ['query', 'reshape'],
        'policy_path': 'policies.analyze.pivot',
    },
    'describe': {
        'dax': '{02A}',
        'intent': Intent.ANALYZE,
        'description': 'Profile a dataset or column — row count, column names, data types, null counts, unique values, and summary statistics (mean, median, min, max, standard deviation)',
        'slots': {
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
            'column': {'type': 'SourceSlot', 'entity_part': 'column', 'priority': 'optional'},
        },
        'output': 'card',
        'edge_flows': ['query', 'compare', 'exist'],
        'policy_path': 'policies.analyze.describe',
    },
    'compare': {
        'dax': '{12A}',
        'intent': Intent.ANALYZE,
        'description': 'Compare two variables, groups, or time periods side by side — correlation, A/B difference, before-vs-after, or distribution overlap. Use segment for breaking one metric down by dimension',
        'slots': {
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
            'column_a': {'type': 'SourceSlot', 'entity_part': 'column', 'priority': 'required'},
            'column_b': {'type': 'SourceSlot', 'entity_part': 'column', 'priority': 'required'},
            'method': {'type': 'CategorySlot', 'priority': 'optional'},
        },
        'output': 'chart',
        'edge_flows': ['trend', 'plot', 'query', 'segment'],
        'policy_path': 'policies.analyze.compare',
    },
    'exist': {
        'dax': '{14C}',
        'intent': Intent.ANALYZE,
        'description': 'Check whether specific data exists in the workspace — does a column, table, or value appear? Returns a yes/no answer with counts and sample matches if found',
        'slots': {
            'query': {'type': 'FreeTextSlot', 'priority': 'required'},
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'optional'},
        },
        'output': 'card',
        'edge_flows': ['query', 'describe'],
        'policy_path': 'policies.analyze.exist',
    },
    'segment': {
        'dax': '{12C}',
        'intent': Intent.ANALYZE,
        'description': 'Break down a single metric by one dimension for drilldown analysis — e.g., MAU by platform, revenue by region, churn by cohort. The inverse of compare, which puts two variables side by side',
        'slots': {
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
            'metric': {'type': 'SourceSlot', 'entity_part': 'column', 'priority': 'required'},
            'dimension': {'type': 'SourceSlot', 'entity_part': 'column', 'priority': 'required'},
        },
        'output': 'table',
        'edge_flows': ['compare', 'describe'],
        'policy_path': 'policies.analyze.segment',
    },

    # ── Report (7 flows) ──────────────────────────────────────────

    'plot': {
        'dax': '{003}',
        'intent': Intent.REPORT,
        'description': 'Create a chart from data — bar, line, pie, scatter, histogram, or heatmap. Includes setting the title and axis labels for new charts; for adjusting an existing chart\'s visuals or metadata use "design"',
        'slots': {
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
            'chart_type': {'type': 'CategorySlot', 'priority': 'optional'},
        },
        'output': 'chart',
        'edge_flows': ['trend', 'dashboard', 'query', 'summarize', 'lookup'],
        'policy_path': 'policies.report.plot',
    },
    'trend': {
        'dax': '{023}',
        'intent': Intent.REPORT,
        'description': 'Perform cohort analysis, or apply period-over-period comparisons (week-over-week, month-over-month), growth rates, and trajectory analysis. Time must be an axis',
        'slots': {
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
            'column': {'type': 'SourceSlot', 'entity_part': 'column', 'priority': 'required'},
            'time_col': {'type': 'SourceSlot', 'entity_part': 'column', 'priority': 'required'},
            'group_by': {'type': 'SourceSlot', 'entity_part': 'column', 'priority': 'optional'},
        },
        'output': 'chart',
        'edge_flows': ['plot', 'compare'],
        'policy_path': 'policies.report.trend',
    },
    'dashboard': {
        'dax': '{03A}',
        'intent': Intent.REPORT,
        'description': 'Schedule recurring reports — daily updates, weekly summaries, or periodic multi-panel views that refresh automatically. Combines multiple charts and tables into a single deliverable',
        'slots': {
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
            'charts': {'type': 'GroupSlot', 'priority': 'required'},
        },
        'output': 'chart',
        'edge_flows': ['plot', 'trend'],
        'policy_path': 'policies.report.dashboard',
    },
    'export': {
        'dax': '{03D}',
        'intent': Intent.REPORT,
        'description': 'Export a dataset or query result to a downloadable file — CSV, Excel, JSON, or Parquet. Applies any active style formatting when exporting to Excel',
        'slots': {
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
            'format': {'type': 'CategorySlot', 'priority': 'optional'},
        },
        'output': 'toast',
        'edge_flows': ['plot', 'dashboard'],
        'policy_path': 'policies.report.export',
    },
    'summarize': {
        'dax': '{019}',
        'intent': Intent.REPORT,
        'description': 'Summarize a specific chart or table in plain language — grounded to a concrete artifact, extracts key takeaways, patterns, and implications. Different from explain, which describes the agent\'s process',
        'slots': {
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
            'chart': {'type': 'FreeTextSlot', 'priority': 'elective'},
            'table': {'type': 'FreeTextSlot', 'priority': 'elective'},
        },
        'output': 'card',
        'edge_flows': ['plot', 'trend'],
        'policy_path': 'policies.report.summarize',
    },
    'style': {
        'dax': '{03E}',
        'intent': Intent.REPORT,
        'description': 'Apply conditional formatting to a table display — color scales, highlight rules, borders, and number formatting (e.g., red for negative values, bold for totals). Visual only; does not change underlying data',
        'slots': {
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
            'condition': {'type': 'FreeTextSlot', 'priority': 'required'},
            'format': {'type': 'FreeTextSlot', 'priority': 'optional'},
        },
        'output': 'table',
        'edge_flows': ['plot', 'design'],
        'policy_path': 'policies.report.style',
    },
    'design': {
        'dax': '{038}',
        'intent': Intent.REPORT,
        'description': 'Adjust an existing chart\'s visual properties — colors, axis labels, legend position, title, gridlines, and chart type. Modifies presentation of an already-created chart; use plot to create one from scratch',
        'slots': {
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
            'chart': {'type': 'FreeTextSlot', 'priority': 'required'},
            'element': {'type': 'FreeTextSlot', 'priority': 'optional'},
        },
        'output': 'chart',
        'edge_flows': ['plot', 'style'],
        'policy_path': 'policies.report.design',
    },

    # ── Converse (7 flows) ────────────────────────────────────────

    'explain': {
        'dax': '{009}',
        'intent': Intent.CONVERSE,
        'description': 'Dana explains what it did or plans to do — transparency into the analysis process, why a particular method was chosen, or what steps are coming next. About process, not data; use summarize for artifact-level takeaways',
        'slots': {
            'topic': {'type': 'FreeTextSlot', 'priority': 'optional'},
        },
        'output': 'card',
        'edge_flows': ['chat', 'recommend'],
        'policy_path': 'policies.converse.explain',
    },
    'chat': {
        'dax': '{000}',
        'intent': Intent.CONVERSE,
        'description': 'Open-ended conversation — general Q&A about data literacy, methodology advice, analytics strategy, or any topic not tied to a specific dataset action',
        'slots': {
            'topic': {'type': 'FreeTextSlot', 'priority': 'optional'},
        },
        'output': 'card',
        'edge_flows': ['recommend', 'preference'],
        'policy_path': 'policies.converse.chat',
    },
    'preference': {
        'dax': '{048}',
        'intent': Intent.CONVERSE,
        'description': 'Set a persistent analysis preference stored in Memory Manager (L2) — default chart colors, preferred date format, decimal precision, timezone, or display density',
        'slots': {
            'key': {'type': 'FreeTextSlot', 'priority': 'required'},
            'value': {'type': 'FreeTextSlot', 'priority': 'required'},
        },
        'output': 'toast',
        'edge_flows': ['chat', 'recommend'],
        'policy_path': 'policies.converse.preference',
    },
    'recommend': {
        'dax': '{049}',
        'intent': Intent.CONVERSE,
        'description': 'Dana proactively suggests a next step based on current data context — a follow-up query, a chart that would clarify results, a data quality issue worth investigating, or an analysis angle to explore',
        'slots': {},
        'output': 'card',
        'edge_flows': ['chat', 'preference'],
        'policy_path': 'policies.converse.recommend',
    },
    'undo': {
        'dax': '{08F}',
        'intent': Intent.CONVERSE,
        'description': 'Reverse the most recent data action — rolls back the last Clean, Transform, or Analyze operation and restores the dataset to its previous state',
        'slots': {
            'action': {'type': 'FreeTextSlot', 'priority': 'optional'},
        },
        'output': 'toast',
        'edge_flows': ['chat'],
        'policy_path': 'policies.converse.undo',
    },
    'approve': {
        'dax': '{09E}',
        'intent': Intent.CONVERSE,
        'description': 'Accept Dana\'s proactive suggestion and trigger the corresponding action — e.g., a recommended query, a data quality fix, or a chart that Dana offered via recommend',
        'slots': {
            'action': {'type': 'FreeTextSlot', 'priority': 'required'},
        },
        'output': 'toast',
        'edge_flows': ['reject', 'chat'],
        'policy_path': 'policies.converse.approve',
    },
    'reject': {
        'dax': '{09F}',
        'intent': Intent.CONVERSE,
        'description': 'Decline Dana\'s proactive suggestion — optionally provide a reason so Dana can adjust future recommendations. Dana notes the preference and moves on',
        'slots': {
            'action': {'type': 'FreeTextSlot', 'priority': 'required'},
            'reason': {'type': 'FreeTextSlot', 'priority': 'optional'},
        },
        'output': 'toast',
        'edge_flows': ['approve', 'chat'],
        'policy_path': 'policies.converse.reject',
    },

    # ── Plan (5 flows) ────────────────────────────────────────────

    'insight': {
        'dax': '{146}',
        'intent': Intent.PLAN,
        'description': 'Plan a multi-step analysis to answer a complex question — chains Analyze and Report flows (e.g., "What drove the revenue drop?" triggers query, segment, trend, summarize)',
        'slots': {
            'question': {'type': 'FreeTextSlot', 'priority': 'required'},
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
        },
        'output': 'list',
        'edge_flows': ['pipeline', 'query'],
        'policy_path': 'policies.plan.insight',
    },
    'pipeline': {
        'dax': '{156}',
        'intent': Intent.PLAN,
        'description': 'Plan a reusable ETL sequence — chains Clean and Transform flows into a saved pipeline that can be replayed on new data (e.g., load, dedupe, join, reshape, export)',
        'slots': {
            'steps': {'type': 'GroupSlot', 'priority': 'required'},
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
        },
        'output': 'list',
        'edge_flows': ['insight', 'join'],
        'policy_path': 'policies.plan.pipeline',
    },
    'blank': {
        'dax': '{46B}',
        'intent': Intent.PLAN,
        'description': 'Diagnose null or empty cells across the dataset — scans all columns, reports counts and percentages, then recommends fill or interpolate to fix them. Diagnoses the problem; Clean flows fix it',
        'slots': {
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
            'column': {'type': 'SourceSlot', 'entity_part': 'column', 'priority': 'elective'},
            'strategy': {'type': 'CategorySlot', 'priority': 'elective'},
        },
        'output': 'list',
        'edge_flows': ['issue', 'validate', 'fill', 'interpolate'],
        'policy_path': 'policies.plan.blank',
    },
    'issue': {
        'dax': '{16F}',
        'intent': Intent.PLAN,
        'description': 'Diagnose data quality issues — outliers, anomalies, inconsistent formats, or suspicious values. Reports findings and recommends Clean flows (validate, format, replace) to fix them',
        'slots': {
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'required'},
            'column': {'type': 'SourceSlot', 'entity_part': 'column', 'priority': 'optional'},
            'type': {'type': 'CategorySlot', 'priority': 'optional'},
        },
        'output': 'list',
        'edge_flows': ['validate', 'blank'],
        'policy_path': 'policies.plan.issue',
    },
    'outline': {
        'dax': '{16D}',
        'intent': Intent.PLAN,
        'description': 'Execute a multi-step plan from instructions — orchestrates flows across Clean, Transform, Analyze, and Report into a sequenced checklist with dependencies. The mandatory Plan orchestrator',
        'slots': {
            'instructions': {'type': 'FreeTextSlot', 'priority': 'required'},
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'optional'},
        },
        'output': 'list',
        'edge_flows': ['insight', 'pipeline'],
        'policy_path': 'policies.plan.outline',
    },

    # ── Internal (6 flows) ────────────────────────────────────────

    'recap': {
        'dax': '{018}',
        'intent': Intent.INTERNAL,
        'description': 'Read back a previously noted fact from the current session scratchpad (L1) — a decision, constraint, dataset reference, or intermediate result the agent stored earlier',
        'slots': {
            'key': {'type': 'FreeTextSlot', 'priority': 'optional'},
        },
        'output': '(internal)',
        'edge_flows': ['recall', 'retrieve'],
        'policy_path': 'policies.internal.recap',
    },
    'calculate': {
        'dax': '{129}',
        'intent': Intent.INTERNAL,
        'description': 'Perform quick arithmetic or comparisons internally — unit conversions, percentage calculations, date arithmetic, or sanity checks before responding to the user',
        'slots': {
            'expression': {'type': 'FreeTextSlot', 'priority': 'required'},
        },
        'output': '(internal)',
        'edge_flows': ['recap', 'lookup'],
        'policy_path': 'policies.internal.calculate',
    },
    'search': {
        'dax': '{149}',
        'intent': Intent.INTERNAL,
        'description': 'Look up vetted FAQs and curated reference content — the unstructured equivalent of lookup in the semantic layer',
        'slots': {
            'query': {'type': 'FreeTextSlot', 'priority': 'required'},
        },
        'output': '(internal)',
        'edge_flows': ['retrieve', 'recap'],
        'policy_path': 'policies.internal.search',
    },
    'peek': {
        'dax': '{189}',
        'intent': Intent.INTERNAL,
        'description': 'Quick internal glance at data state before responding — checks row count, column names, data types, or sample values without surfacing results to the user',
        'slots': {
            'dataset': {'type': 'SourceSlot', 'entity_part': 'table', 'priority': 'optional'},
        },
        'output': '(internal)',
        'edge_flows': ['recap', 'calculate'],
        'policy_path': 'policies.internal.peek',
    },
    'recall': {
        'dax': '{489}',
        'intent': Intent.INTERNAL,
        'description': 'Look up persistent user preferences from Memory Manager (L2) — default chart style, date format, decimal precision, or timezone set via the preference flow',
        'slots': {
            'key': {'type': 'FreeTextSlot', 'priority': 'optional'},
            'scope': {'type': 'CategorySlot', 'priority': 'optional'},
        },
        'output': '(internal)',
        'edge_flows': ['recap', 'retrieve'],
        'policy_path': 'policies.internal.recall',
    },
    'retrieve': {
        'dax': '{004}',
        'intent': Intent.INTERNAL,
        'description': 'Fetch general business context from Memory Manager — unvetted documents, reports, or domain knowledge from any source',
        'slots': {
            'topic': {'type': 'FreeTextSlot', 'priority': 'required'},
            'source': {'type': 'FreeTextSlot', 'priority': 'optional'},
        },
        'output': '(internal)',
        'edge_flows': ['recall', 'search'],
        'policy_path': 'policies.internal.retrieve',
    },
}

KEY_ENTITIES = ['table', 'row', 'column']
