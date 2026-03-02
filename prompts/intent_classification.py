"""Build system prompts for intent classification (Exp 2A stage 1)."""

from __future__ import annotations


_DOMAIN_INTROS = {
    'hugo': 'Hugo is a blog writing assistant that helps users research topics, draft posts, revise content, and publish to platforms.',
    'dana': 'Dana is a data analysis assistant that helps users clean datasets, transform data, run analyses, and create reports and visualizations.',
}

_INTENT_DESCRIPTIONS = {
    'hugo': {
        'Research': 'Reading and inspecting what already exists. The user gets back information about their content: search results, post metadata, word counts, version diffs. Nothing is created or changed. Research includes comparing versions and reviewing diffs — "show me what changed" is Research because the user gets back information, not modified text.',
        'Draft': 'Creating new content from scratch or developing existing notes into prose. The user gets back new text that did not exist before: outlines, sections, expanded bullet points, brainstormed ideas. Use Draft when content is being *added*, not *edited*.',
        'Revise': 'Editing and improving existing text. The user gets back a modified version of content that already existed: tightened prose, adjusted tone, reformatted structure, style audits. Use Revise when content is being *changed*, not created from nothing. If the user is just viewing or comparing versions without requesting changes, that is Research.',
        'Publish': 'Distributing content to platforms and managing publication lifecycle. The user gets back a published post, a scheduled date, a preview, or platform status. Anything related to getting content out the door.',
        'Converse': 'Non-content interaction with the agent. The user gets back explanations of what the agent did, preference confirmations, approval/rejection of suggestions, or general writing advice. No content is created, edited, or published.',
        'Plan': 'Multi-step orchestration that spans two or more other intents. The user describes a workflow requiring many actions (e.g., "research, then draft, then publish"). If the user makes a request that is vague or could go in multiple directions, this is also a Plan.',
    },
    'dana': {
        'Clean': 'Fixing data quality problems. The user gets back data with corrected values: deduplicated rows, filled gaps, standardized formats, validated entries, fixed types. The data structure stays the same, but the values become more reliable.',
        'Transform': 'Changing data structure. The user gets back a reshaped dataset: new columns, deleted rows, joined tables, split or merged columns, pivoted layouts. The shape of the data changes, not just the values within it.',
        'Analyze': 'Running new computations on data. The user gets back numbers, metrics, comparisons, or existence checks: "What is X?", "How does A compare to B?", "Do we have Y?" Nothing is changed or created, just computed. Analyze requires a new computation — if the user is asking about an already-generated chart or table ("what\'s the takeaway from that chart"), that is Report, not Analyze. Asking about formulas or definitions in the semantic layer is also Analyze.',
        'Report': 'Producing visual artifacts, exports, and interpreting existing outputs. The user gets back a chart, a styled table, a dashboard, a CSV file, or a written summary. Use Report when the output is a deliverable, not just an answer. Report also covers interpreting or summarizing existing charts and tables — "what\'s the takeaway" or "give me a plain english rundown" of an already-generated artifact is Report.',
        'Converse': 'Non-data interaction with the agent. The user gets back explanations of what the agent did, preference confirmations, approval/rejection of suggestions, or general data advice. No data is changed, analyzed, or exported. Converse is strictly meta-interaction. Terse utterances about data operations ("I want X and Y columns", "save a formula for Z", "what\'s the formula behind X") are domain actions (Transform, Analyze), not Converse.',
        'Plan': 'Multi-step orchestration that spans two or more other intents. The user describes a workflow requiring many actions (e.g., "clean the dates, then analyze, then chart it"). If the user makes a request that is vague or could go in multiple directions, this is also a Plan.',
    },
}

_HUGO_INTENT_EXEMPLARS = """### Example 1
User: "Have I written anything about container orchestration before?"
```json
{"reasoning": "Searching existing content by topic to see what's been published.", "intent": "Research"}
```

### Example 2
User: "Start a new post on edge computing for IoT."
Agent: "I've created a blank post titled 'Edge Computing for IoT'. Want me to outline it?"
User: "How long is the serverless post? I want a similar length."
```json
{"reasoning": "Checking metadata of another post for reference. Despite the Draft context, the last message is inspecting an existing post.", "intent": "Research"}
```

### Example 3
User: "I've got bullet points for the noise pollution section. Flesh them out into actual paragraphs."
```json
{"reasoning": "Taking existing notes and developing them into full prose.", "intent": "Draft"}
```

### Example 4
User: "Outline a post comparing React and Vue for enterprise apps."
Agent: "Here's a five-section outline: architecture, ecosystem, performance, hiring, and migration."
User: "Add a section on developer experience after ecosystem."
```json
{"reasoning": "Inserting a new section into the outline. Prior context was also drafting, and this continues building the structure.", "intent": "Draft"}
```

### Example 5
User: "Get the remote work post ready for Medium. Fix the heading levels, add tags, set a featured image, and make sure the code blocks render right."
```json
{"reasoning": "Multiple formatting tasks, but all are preparing existing content for publication — fixing headings, tags, image, code blocks. This is still Revise (formatting and cleanup), not Plan, because every step falls within the same intent.", "intent": "Revise"}
```

### Example 6
User: "Make the whole post sound more conversational, less academic."
```json
{"reasoning": "Adjusting the tone and voice across the post.", "intent": "Revise"}
```

### Example 7
User: "Push the React post to Medium."
```json
{"reasoning": "Distributing content to a secondary platform.", "intent": "Publish"}
```

### Example 8
User: "The conclusion is too wordy, can you make it less verbose?"
Agent: "Done, trimmed it to 90 words. Ready to go?"
User: "Yep, schedule it for Tuesday morning."
```json
{"reasoning": "Scheduling the post for publication. Prior context was revision, but the last message is about publishing.", "intent": "Publish"}
```

### Example 9
User: "Thanks, that looks great."
Agent: "Happy to help! Let me know if you want to tweak anything else."
User: "Actually, always default to AP style going forward."
```json
{"reasoning": "Setting a persistent writing preference for future sessions.", "intent": "Converse"}
```

### Example 10
User: "Wait, what did you just change?"
```json
{"reasoning": "Asking the agent to explain its most recent action.", "intent": "Converse"}
```

### Example 11
User: "I want to do a 5-part series on cloud migration, one post per week."
```json
{"reasoning": "Planning a multi-post series with a publishing schedule, requires orchestrating multiple steps.", "intent": "Plan"}
```

### Example 12
User: "Check if I have anything on Kubernetes already."
Agent: "You have two drafts: 'K8s for Beginners' (1200 words) and 'Helm Charts Deep Dive' (800 words, stale)."
User: "OK, revise both of those and then publish whichever is stronger."
```json
{"reasoning": "Multiple steps across revision and publishing, conditional on comparison. This requires multi-step orchestration.", "intent": "Plan"}
```

### Example 13
User: "Can you show me what changed between the last two edits on the Bangkok section?"
```json
{"reasoning": "Comparing versions to see what changed. This is inspecting existing content (a diff), not making edits. The user gets back information, not modified text.", "intent": "Research"}
```"""

_DANA_INTENT_EXEMPLARS = """### Example 1
User: "The phone numbers are all over the place, dashes and parens mixed in."
```json
{"reasoning": "Inconsistent formatting across values needs standardization.", "intent": "Clean"}
```

### Example 2
User: "What's AOV by customer tier?"
Agent: "Gold $142, Silver $89, Bronze $54."
User: "I think there are dupes in the customer table, that might be skewing it."
```json
{"reasoning": "Suspecting duplicate rows that need removal. Prior context was analysis, but the last message is about data quality.", "intent": "Clean"}
```

### Example 3
User: "Add a profit margin column from revenue and cost."
```json
{"reasoning": "Creating a new derived column via computation.", "intent": "Transform"}
```

### Example 4
User: "The phone numbers are all over the place, dashes and parens mixed in."
Agent: "Standardized 4,200 phone numbers to digits only. 12 couldn't be parsed."
User: "Split the area code into its own column."
```json
{"reasoning": "Splitting a column into parts. Prior context was cleaning, but the last message is a structural transformation.", "intent": "Transform"}
```

### Example 5
User: "What's AOV by customer tier?"
```json
{"reasoning": "Computing an aggregate metric segmented by a dimension.", "intent": "Analyze"}
```

### Example 6
User: "Combine the first_name and last_name columns."
Agent: "Done, merged into full_name. 8,400 rows updated."
User: "Do we even have email data in here?"
```json
{"reasoning": "Checking whether specific data exists. Prior context was transformation, but the last message is about data discovery.", "intent": "Analyze"}
```

### Example 7
User: "Set up a weekly report with stock levels by warehouse, shipping delays over time, and a summary table with conditional formatting on low-inventory rows."
```json
{"reasoning": "Multiple components (charts, table, formatting), but all are parts of a single report deliverable. This is still Report, not Plan, because every element falls within the same intent.", "intent": "Report"}
```

### Example 8
User: "What's AOV by customer tier?"
Agent: "Gold $142, Silver $89, Bronze $54."
User: "Chart that as a bar, and export to CSV too."
```json
{"reasoning": "Creating a visualization and exporting data. Prior context was analysis, but the last message is about producing output.", "intent": "Report"}
```

### Example 9
User: "Standardized 4,200 phone numbers. 12 couldn't be parsed."
Agent: "Want me to flag those 12 for manual review?"
User: "Yep, go with that."
```json
{"reasoning": "Approving the agent's suggestion. Prior context was cleaning, and the last message provides approval of the suggestion.", "intent": "Converse"}
```

### Example 10
User: "Always show me percentages with one decimal."
```json
{"reasoning": "Setting a persistent formatting preference.", "intent": "Converse"}
```

### Example 11
User: "Clean the dates, then break down conversion rates by channel and quarter."
```json
{"reasoning": "Multiple steps across cleaning and analysis, needs multi-step orchestration.", "intent": "Plan"}
```

### Example 12
User: "What does the churn table look like?"
Agent: "12 columns, 50K rows. churn_risk is float 0-1, 2100 nulls. customer_segment has 5 values."
User: "OK, fill in those nulls, segment by tier, and build me a dashboard."
```json
{"reasoning": "Multiple steps: fill missing values, segment analysis, and build dashboard. This requires multi-step orchestration.", "intent": "Plan"}
```

### Example 13
User: "What's AOV by customer tier?"
Agent: "Gold $142, Silver $89, Bronze $54."
User: "What's the takeaway from that breakdown? Summarize the key patterns."
```json
{"reasoning": "Interpreting an already-computed result. No new computation needed — the analysis is done and the user wants a written summary of it.", "intent": "Report"}
```

### Example 14
User: "hey can you save a formula for engagement rate? likes plus comments plus shares divided by impressions"
```json
{"reasoning": "Creating a derived calculation in the data layer. Despite the casual tone, this is a structural data operation, not conversation.", "intent": "Transform"}
```"""


def build_intent_classification_prompt(domain: str) -> str:
    """Build the system prompt for intent classification (Exp 2A stage 1).

    Classifies the user's most recent utterance into one of 6 user-facing
    intents (Internal excluded — system-level, never user-invoked).
    """
    domain_label = domain.capitalize()
    intro = _DOMAIN_INTROS.get(domain, f'{domain_label} is a conversational assistant.')

    intents = _INTENT_DESCRIPTIONS.get(domain, {})
    intent_lines = [f'- **{name}**: {desc}' for name, desc in intents.items()]
    intent_list = '\n'.join(intent_lines)

    exemplars = _HUGO_INTENT_EXEMPLARS if domain == 'hugo' else _DANA_INTENT_EXEMPLARS

    prompt = f"""You are an intent classifier for {domain_label}, a conversational assistant. {intro}

Given the conversation history, classify the most recent user message into exactly one intent.

## Intents

{intent_list}

## Instructions

1. Classify ONLY the most recent user message. Prior turns may belong to a different intent.
2. Use conversation history for context (references, anaphora, implicit meaning), but do not let prior intents bias your classification.
3. Classify into exactly one intent.
4. Return ONLY valid JSON. No markdown fences, no explanation outside the JSON.

## Output Format

{{"reasoning": "<Brief explanation of why this intent>", "intent": "<IntentName>"}}

## Examples

{exemplars}

Respond with exactly one JSON object. Nothing else."""

    return prompt
