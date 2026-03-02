version_1 = """
1. Read the user's utterance and any prior conversation context.
2. Identify the key patterns, words, or phrases that signal what the user wants.
3. Select the single flow that best matches the captured concept. The vast majority of utterances map to exactly one flow.
4. If the utterance is genuinely ambiguous and contains requests for two distinct operations that cannot be handled by a single flow, output all relevant flows. This is uncommon — only do this when the utterance clearly heads in multiple directions.
5. Only output flows from the Candidate Flows list above.
6. Return ONLY valid JSON. No markdown fences, no explanation outside the JSON.
"""

version_2 = """
1. Read the user's utterance and any prior conversation context.
2. Identify the key patterns, words, or phrases that signal what the user wants.
3. Determine which flow(s) match the user's request. Many utterances map to one flow, but some are ambiguous or contain multiple requests.
4. If the utterance could reasonably be interpreted as two different flows, output all plausible flows. It is better to include a second plausible flow than to miss it.
5. Only output flows from the Candidate Flows list above.
6. Return ONLY valid JSON. No markdown fences, no explanation outside the JSON.
"""

version_3 = """
1. Read the user's utterance and any prior conversation context.
2. Identify the key patterns, words, or phrases that signal what the user wants.
3. Consider your top candidate flow. Then ask: "Could a reasonable person read this utterance and conclude a *different* flow is intended?" If yes, include that second flow too.
4. When the utterance is the first thing the user says (no prior context), be especially alert to ambiguity — the lack of context means multiple interpretations are more likely, not less.
5. Output one flow when the match is clear and unambiguous. Output two flows when the utterance has two plausible readings or explicitly requests two distinct operations.
6. Only output flows from the Candidate Flows list above.
7. Return ONLY valid JSON. No markdown fences, no explanation outside the JSON.
"""

version_4 = """
1. Read the user's utterance and any prior conversation context.
2. Identify the key patterns, words, or phrases that signal what the user wants.
3. Determine which flow(s) match the user's request. Many utterances map to one flow, but some may be ambiguous or contain multiple requests.
4. If the utterance could reasonably be interpreted as two different flows, then output all plausible flows to cover all bases.
5. Only output flows from the Candidate Flows list above.
6. Return ONLY valid JSON. No markdown fences, no explanation outside the JSON.
"""