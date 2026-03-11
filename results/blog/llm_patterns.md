# LLM Patterns

## Word Choice

### 1. Magic Adverbs

- **Count**: 10 occurrences found
- **Threshold**: 3
- **Status**: VIOLATION (10 - 3 = 7 to fix)

Occurrences:
1. Line 8: "Confidently. Wrongly." — "Confidently" used as a standalone adverb for dramatic effect.
2. Line 12: "confidently executed" — adverb signaling significance.
3. Line 29: "naturally talk to systems" — "naturally" is a magic adverb here.
4. Line 31: "must simultaneously do two things" — "simultaneously" adds mild flavor but is functional.
5. Line 44: "show disproportionate drops" — "disproportionate" functions adverbially to inflate the claim.
6. Line 69: "statistically likely" — used twice in same sentence; functional but contributes to adverb density.
7. Line 85: "largest precisely on the turn categories" — "precisely" signals significance rather than showing it.
8. Line 87: "performs markedly better" — "markedly" is a classic magic adverb.
9. Line 87: "reflects the genuine difficulty" — "genuine" used to add weight without specifics.
10. Line 133: "empirically, benefits from" — "empirically" used as a filler qualifier.

**FIX** (7 most egregious):

1. Line 8: "Confidently. Wrongly."
   - Remove the dramatic fragment pair. Fold into prior sentence.
   - Proposed: "In flat tool-calling, the agent picks the most probable tool and executes — often incorrectly."

2. Line 85: "largest precisely on the turn categories"
   - "precisely" adds no information here.
   - Proposed: "The pipeline's structural advantage is largest on the turn categories where flat tool-calling breaks down."

3. Line 87: "performs markedly better"
   - Replace with specific comparison.
   - Proposed: "the flow detection ensemble performs better by 15+ percentage points"

4. Line 87: "reflects the genuine difficulty"
   - "genuine" is hollow emphasis.
   - Proposed: "reflects the difficulty of the ambiguous-first case"

5. Line 133: "empirically, benefits from"
   - Drop the filler adverb.
   - Proposed: "a task that benefits from a dedicated classification step"

6. Line 44: "show disproportionate drops"
   - Replace with specifics.
   - Proposed: "show large accuracy drops on ambiguous-first."

7. Line 29: "naturally talk to systems"
   - Proposed: "talk to systems they expect to understand them."

### 2. Overused Vocabulary

- **Count**: 0 occurrences found
- **Threshold**: 4
- **Status**: PASS

No instances of "delve", "certainly", "utilize", "leverage", "robust", "streamline", "harness", "tapestry", "landscape", "paradigm", "synergy", "ecosystem", or "framework" found.

### 3. Pompous Verbs

- **Count**: 1 occurrence found
- **Threshold**: 4
- **Status**: PASS

Occurrences:
1. Line 83: "it represents a ~31 percentage point structural improvement" — "represents" is a pompous substitute for "is".


## Sentence Structure

### 4. Negative Parallelism

- **Count**: 5 occurrences found
- **Threshold**: 1
- **Status**: VIOLATION (5 - 1 = 4 to fix)

Occurrences:
1. Line 10: "The failure mode is distinct from **hallucination**... The problem is in the routing decision" — sets up a "not X, it's Y" frame across sentences.
2. Line 31: "The problem isn't that these utterances are somehow malformed. The problem is that the flat tool-selection model must simultaneously do two things" — classic "It's not X — it's Y" structure.
3. Line 69: "The model isn't failing randomly. It's defaulting to semantically plausible tools" — "not X. It's Y."
4. Line 117: "The prompt tells it to be cautious — but 'cautious' without a prior classification signal means occasionally being cautious about clear requests" — negation-pivot structure.
5. Line 119: "Structural problems require structural solutions." — while not a negation per se, this follows and caps a negation-framed paragraph.

**FIX** (4 most egregious):

1. Line 31: "The problem isn't that these utterances are somehow malformed. The problem is that the flat tool-selection model must simultaneously do two things"
   - Proposed: "These utterances are well-formed. The flat tool-selection model just has to do two things at once in a single inference step:"

2. Line 69: "The model isn't failing randomly. It's defaulting to semantically plausible tools"
   - Proposed: "The model defaults to semantically plausible tools — reading, searching, analyzing — because those are the most likely responses to underspecified inputs."

3. Line 10: "The failure mode is distinct from **hallucination**... The problem is in the routing decision"
   - Proposed: "This failure mode differs from hallucination in a specific way: the routing decision happens before any content is generated. The model selected the wrong tool for the user's underlying intent, and nothing in the output surface revealed that."

4. Line 117: "The prompt tells it to be cautious — but 'cautious' without a prior classification signal means occasionally being cautious about clear requests"
   - Proposed: "The prompt tells it to be cautious, but without a prior classification signal, the model sometimes applies that caution to clear requests too, which degrades performance on the categories where the model was already doing well."

### 5. Rhetorical Questions

- **Count**: 2 occurrences found
- **Threshold**: 1
- **Status**: VIOLATION (2 - 1 = 1 to fix)

Occurrences:
1. Line 57: "**Why does this happen?** Two reasons compound." — poses a question then immediately answers it.
2. Line 91/125: "Can You Prompt Your Way Out of This?" (section heading, line 91) and "What does the pipeline provide that flat tool-calling doesn't?" (line 125) — rhetorical questions as section openers.

Note: Line 91 is a section heading, which is a common and acceptable use. Line 125 is the more problematic case — a rhetorical question posed and immediately answered.

**FIX** (1 most egregious):

1. Line 57: "**Why does this happen?** Two reasons compound."
   - Proposed: "Two reasons compound to produce this failure mode."

### 6. Filler Transitions

- **Count**: 0 occurrences found
- **Threshold**: 4
- **Status**: PASS

No instances of "It's worth noting", "It bears mentioning", "Importantly", "Interestingly", or "Notably" found.

### 7. Show, Don't Tell

- **Count**: 1 occurrence found
- **Threshold**: 1
- **Status**: PASS

Occurrences:
1. Line 63: "The confusion tables from our evaluation are illustrative." — tells the reader the tables are illustrative rather than letting them speak for themselves. However, it does follow with concrete data, so it's borderline.

### 8. False Ranges

- **Count**: 1 occurrence found
- **Threshold**: 1
- **Status**: PASS

Occurrences:
1. Line 14: "ranged from **37.5% to 57.3%**" — this is a legitimate data range with a real spectrum, not a false range dressing up two loosely related things. This is appropriate use.

### 9. Short Punchy Fragments

- **Count**: 1 occurrence found
- **Threshold**: 2
- **Status**: PASS

Occurrences:
1. Line 8: "Confidently. Wrongly." — two standalone single-word fragments used for dramatic effect.


## Tone

### 10. Unnecessary Metaphors

- **Count**: 1 occurrence found
- **Threshold**: 1
- **Status**: PASS

Occurrences:
1. Line 6: "Imagine a user says: *'Can you check on that last post?'*" — this is a concrete scenario rather than a metaphor. It's functional and illustrative, so it passes.

No actual metaphors found. The post is direct and technical throughout.

### 11. False Vulnerability

- **Count**: 0 occurrences found
- **Threshold**: 1
- **Status**: PASS

No instances of simulated candor or false self-awareness found.

### 12. Go Directly to the Point

- **Count**: 2 occurrences found
- **Threshold**: 1
- **Status**: VIOLATION (2 - 1 = 1 to fix)

Occurrences:
1. Line 79: "> **The key insight**: you can't fix the ambiguity problem at the tool-selection layer" — labels the point as "the key insight" before stating it, telling the reader what to think.
2. Line 119: "Structural problems require structural solutions." — aphoristic assertion that tells the reader the conclusion is simple/clear rather than proving it. The prior paragraph already made the argument.

**FIX** (1 most egregious):

1. Line 79: "> **The key insight**: you can't fix the ambiguity problem at the tool-selection layer"
   - Drop the label and let the insight stand on its own.
   - Proposed: "> You can't fix the ambiguity problem at the tool-selection layer, because the model at that layer has no structural access to the information that would tell it something is ambiguous. You have to fix it upstream."

### 13. Grandiose Stakes Inflation

- **Count**: 1 occurrence found
- **Threshold**: 1
- **Status**: PASS

Occurrences:
1. Line 133: "This generalizes beyond ambiguity handling." — mild stakes inflation, suggesting the finding applies broadly. However, the sentence follows with a specific claim grounded in the data, so it's borderline acceptable.

### 14. Vague Attributions

- **Count**: 0 occurrences found
- **Threshold**: 0
- **Status**: PASS

No unnamed experts, vague studies, or inflated consensus claims found. All data points are attributed to "our evaluation" or specific experiments.

### 15. Invented Concept Labels

- **Count**: 3 occurrences found
- **Threshold**: 2
- **Status**: VIOLATION (3 - 2 = 1 to fix)

Occurrences:
1. Lines 12, 16: "confident misdirection" — coined compound label, introduced with "We call this" and used twice (lines 12 and 16).
2. Line 61: "**high-entropy tool distributions**" — compound label bolded and treated as an established term.
3. Line 133: "**structured intent disambiguation**" — compound label bolded and treated as an established term.

**FIX** (1 most egregious):

1. Line 133: "**structured intent disambiguation**"
   - This label is used once, bolded, and treated as a term of art. It adds jargon without earning it.
   - Proposed: "The flow detection stage is doing intent disambiguation through a dedicated classification step rather than being folded into the tool-selection inference."


## Formatting

### 16. Em-Dash Addiction

- **Count**: 27 occurrences found
- **Threshold**: 2
- **Status**: VIOLATION (27 - 2 = 25 to fix)

Occurrences (all lines with em dashes):
1. Line 10: "before any content was generated — the model selected"
2. Line 12: "confidently executed — and for the wrong task"
3. Line 14: "it's a consistent structural gap" (em dash before "it's")
4. Line 16: "confident misdirection — below 50%"
5. Line 26: "is underspecified — the user's intent"
6. Line 29: "they rely on shared context, anaphora, and implicit reference" (em dash before "they rely")
7. Line 29: "these are how people naturally talk" (em dash before "these are")
8. Line 36: "it just sees an underspecified utterance" (em dash before "it just")
9. Line 42: "roughly 21 percentage points below" (em dash before "roughly")
10. Line 55: "the strongest flat model in our benchmark at 76.4% overall — only reaches"
11. Line 59: "but without a prior classification step" (em dash before "but")
12. Line 69: "reading, searching, analyzing — because"
13. Line 77: "as it will be for underspecified inputs — the system routes"
14. Line 81: "meaning it correctly identified" (em dash before "meaning")
15. Line 87: "these are utterances where even human annotators might disagree about intent — and"
16. Line 87: "the genuine difficulty of the ambiguous-first case — these are"
17. Line 109: "modest, and concentrated" (em dash before "modest")
18. Line 115: "the hint does help Haiku on ambiguous-first" (em dash before "the hint")
19. Line 117: "but 'cautious' without a prior classification signal" (em dash before "but")
20. Line 127: "flow detection produces a distribution over flows, not just a point estimate" (em dash after "A confidence signal")
21. Line 128: "the ambiguity handler is structurally reachable" (em dash after "A dedicated routing path")
22. Line 129: "detecting ambiguity and selecting a tool are two distinct decisions" (em dash after "Separation of concerns")
23. Line 131: "not just prompt-instructed" (em dash before "not just")
24. Line 133: "a task that, empirically, benefits" (em dash after "structured intent disambiguation")
25. Line 141: "more diverse, less controlled, potentially higher entropy" (em dash after "a different distribution")
26. Line 148: two em dashes in the footer references

Note: Lines 127-129 are a numbered list where the em dash separates a bolded label from its explanation. This is a consistent formatting choice for the list and could be considered acceptable (3 uses in one structure). Still, the overall count is extreme.

**FIX** (25 to fix; showing the 10 most egregious — the rest should follow the same pattern of replacing with commas, colons, periods, or parentheses):

1. Line 10: "before any content was generated — the model selected the wrong tool"
   - Proposed: "before any content was generated. The model selected the wrong tool for the user's underlying intent, and nothing in the output surface revealed that."

2. Line 16: "confident misdirection — below 50% accuracy"
   - Proposed: "confident misdirection: below 50% accuracy on ambiguous turns for most models."

3. Line 36: "it's in an ambiguous situation — it just sees"
   - Proposed: "it's in an ambiguous situation. It just sees an underspecified utterance and assigns probability mass across the tool inventory."

4. Line 59: "when uncertain" — but without a prior classification step"
   - Proposed: "when uncertain,' but without a prior classification step, the model can't reliably distinguish uncertain from certain contexts."

5. Line 69: "reading, searching, analyzing — because those are"
   - Proposed: "reading, searching, analyzing. Those are the most statistically likely responses to underspecified inputs."

6. Line 77: "as it will be for underspecified inputs — the system routes"
   - Proposed: "as it will be for underspecified inputs. In those cases, the system routes to the `ambiguity_handler` intent rather than propagating uncertainty down to the tool-selection stage."

7. Line 87: "the genuine difficulty of the ambiguous-first case — these are utterances"
   - Proposed: "the genuine difficulty of the ambiguous-first case. These are utterances where even human annotators might disagree about intent."

8. Line 109: "**+2.8%** — modest, and concentrated"
   - Proposed: "**+2.8%**, which is modest and concentrated in the overall accuracy rather than specifically in ambiguous-turn categories."

9. Line 115: "But note the side effect: Gemini Flash's ambiguous-second accuracy" (already uses colon; the em dash before "the hint does help" should become a period)
   - Proposed: "There's a real signal here. The hint does help Haiku on ambiguous-first."

10. Line 131: "before tool selection — not just prompt-instructed"
    - Proposed: "before tool selection, not just through prompt instructions."

### 17. Bold-First Bullets

- **Count**: 6 occurrences found
- **Threshold**: 1
- **Status**: VIOLATION (6 - 1 = 5 to fix)

Occurrences:
1. Line 24: "- **same-flow**: the user continues..."
2. Line 25: "- **switch-flow**: the user pivots..."
3. Line 26: "- **ambiguous-first**: the opening message..."
4. Line 27: "- **ambiguous-second**: the follow-up turn..."
5. Line 127: "1. **A confidence signal** — flow detection produces..."
6. Line 128: "2. **A dedicated routing path** — the ambiguity handler..."
7. Line 129: "3. **Separation of concerns** — detecting ambiguity..."

Note: Lines 24-27 are defining dataset categories — the bold terms are labels being introduced for the first time. This is a standard and defensible use for a technical post defining terms. Lines 127-129 are the more problematic case, as they use bold-first bullets for rhetorical emphasis.

**FIX** (5 most egregious — focusing on lines 127-129 and acknowledging lines 24-27 are borderline):

1. Lines 127-129: The numbered list in Section 6 uses bold labels followed by em dashes.
   - Proposed rewrite as prose: "The pipeline provides three things flat tool-calling doesn't. First, flow detection produces a distribution over flows, not just a point estimate — low confidence is a signal, not noise. Second, the ambiguity handler is structurally reachable before the tool-selection stage, not just nominally available as one of 56 tools. Third, detecting ambiguity and selecting a tool are two distinct decisions, handled by two distinct stages."

2. Lines 24-27: The four-item definition list uses bold labels. These are category definitions used throughout the post, so they serve a reference function. However, they could be reformatted.
   - Proposed: Convert to a table or inline definitions in prose rather than a bold-first bullet list.


## Composition

### 18. Filler Sentences

- **Count**: 0 occurrences found
- **Threshold**: 0
- **Status**: PASS

No instances of "Here's the kicker", "Here's the thing", "Here's where it gets interesting", "Let's break this down", "Let's unpack this", "Let's explore", or "Let's dive in" found.

### 19. Duplicate Content

- **Count**: 3 occurrences found
- **Threshold**: 1
- **Status**: VIOLATION (3 - 1 = 2 to fix)

Occurrences:
1. Lines 14 and 42: The accuracy range "37.5% to 57.3% across 8 models, averaging 47.2%" appears verbatim in both the introduction (line 14) and Section 3's blockquote (line 42). The blockquote also restates "flat tool-calling accuracy" on "ambiguous-first turns."
2. Lines 14 and 81: Line 81 repeats "flat accuracy on ambiguous-first of 47.2% average" — restating the same stat from the introduction.
3. Lines 75 and 131: Both state the same core recommendation — that the ambiguity handler should be structurally reachable before tool selection. Line 75: "the system can route to an ambiguity handler instead of guessing at a tool." Line 131: "make it **structurally reachable** before tool selection — not just prompt-instructed."

**FIX** (2 most egregious):

1. Line 42: The blockquote restates the exact stat from line 14.
   - The introduction already establishes the key number. The blockquote should add new information or context rather than repeating it.
   - Proposed: Remove the blockquote or rephrase to emphasize the comparison rather than restating the number: "> Ambiguous-first turns lagged switch-flow turns by roughly 21 percentage points across all 8 models — the largest per-category gap in our evaluation."

2. Line 81: "This compares to flat accuracy on ambiguous-first of 47.2% average."
   - The reader already knows this number from lines 14 and 42. Drop the restatement.
   - Proposed: "For ambiguous-first turns, our flow detection ensemble achieved **78.8% accuracy** — a ~31 percentage point improvement over the flat baseline."

### 20. Repetitive Language

- **Count**: 4 occurrences found
- **Threshold**: 2
- **Status**: VIOLATION (4 - 2 = 2 to fix)

Occurrences:
1. The phrase "structurally reachable" appears on lines 128 and 131 within a few lines of each other.
2. The phrase "the model has no structural" / "no structural signal" / "no structural access" / "lacks the structural context" appears on lines 36, 79, 117 — "structural" used as a recurring modifier throughout the piece (lines 14, 36, 75, 79, 83, 85, 117, 119, 128, 129, 131, 133 — 12 uses of "structural/structurally").
3. The phrase "not just" appears on lines 127 ("not just a point estimate"), 128 ("not just nominally available"), and 131 ("not just prompt-instructed") — three times in five lines.
4. The word "ambiguous-first" appears 13 times throughout the post (lines 26, 42, 44, 53, 55, 59, 81, 83, 87, 109, 111, 113, 142). While it's a defined term, the repetition is heavy.

**FIX** (2 most egregious):

1. Lines 127-131: "not just" appears three times in five lines.
   - Proposed for line 127: "flow detection produces a distribution over flows rather than a point estimate."
   - Proposed for line 131: "make it **structurally reachable** before tool selection, rather than relying on prompt instructions alone."

2. Lines 128 and 131: "structurally reachable" appears twice in close proximity.
   - Proposed for line 128: "the ambiguity handler is accessible before the tool-selection stage, not nominally available as one of 56 tools."

