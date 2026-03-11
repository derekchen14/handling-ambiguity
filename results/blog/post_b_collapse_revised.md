# Direct Tool-Calling Collapses on Ambiguous Requests


## 1. Introduction

Imagine a user says: *"Can you check on that last post?"*

This could mean at least three things: pull up the post to review it, check its publication status, or analyze how it's performing. In a well-designed system, the agent asks a clarifying question. In direct tool-calling, the agent picks the most probable tool and executes, often incorrectly.

This failure mode differs from **hallucination** in a specific way: the routing decision happens before any content is generated. The model selected the wrong tool for the user's underlying intent, and nothing in the output revealed that. The response is fluent. The parameters look reasonable. If a human glanced at the output, nothing would seem amiss.

We call this **confident misdirection**: fluent, grammatically correct, but executed for the wrong task.

In our evaluation, direct tool-calling accuracy on underspecified first-turn requests ranged from **37.5% to 57.3%** across 8 models, averaging **47.2%**. This is not a failure of a single weak model. It is a consistent gap across every model we tested.

When the model doesn't know what it doesn't know, you get confident misdirection: below 50% accuracy on ambiguous turns for most models.

---

## 2. What Ambiguity Looks Like in Dialogue

Our evaluation dataset includes four conversational turn categories:

- **same-flow**: the user continues the same task as the previous turn (*"Same thing for the performance benchmarks section"*)
- **switch-flow**: the user pivots to a different task mid-conversation (*"Now I want to check when this is scheduled to go out"*)
- **ambiguous-first**: the opening message is underspecified, and the user's intent can't be resolved without context or clarification (*"Can you help me with the developer tools post?"*)
- **ambiguous-second**: the follow-up turn introduces context that conflicts with or overrides the apparent intent established earlier

The last two categories are inevitable in any real deployment. Users don't narrate their intent. They rely on shared context, anaphora, and implicit reference. *"That last one"*, *"the same thing"*, *"can you check on it"* are how people talk to systems they expect to understand them.

These utterances are well-formed. The direct tool-selection model just has to do two things at once in a single inference step:

1. **Detect** whether the request is ambiguous
2. **Select** the right tool, given that judgment

Both tasks get conflated into a single inference step. The model has no signal that it's in an ambiguous situation. It sees an underspecified utterance and assigns probability mass across the tool inventory.

---

## 3. The Flat Baseline: Where It Breaks

> **Ambiguous-first turns lagged switch-flow turns by roughly 21 percentage points across all 8 models, the largest per-category gap in our evaluation.**

**Figure 1**: *Flat tool-calling accuracy by turn category per model. Ambiguous-first is the hardest category for every model. Models that perform well on clear turns (same-flow, switch-flow) show large accuracy drops on ambiguous-first.*

The per-category breakdown tells the story:

| Category | Accuracy Range | Avg |
|----------|---------------|-----|
| switch-flow | 56.3%–88.8% | 74.3% |
| same-flow | 51.8%–84.9% | 68.3% |
| ambiguous-second | 45.1%–78.1% | 63.9% |
| ambiguous-first | **37.5%–57.3%** | **47.2%** |

The gradient is consistent across capability tiers: high-tier models (Opus, Qwen3) score 45.6% and 50.5% respectively on ambiguous-first. Mid-tier Sonnet manages 43.5%. Even Gemini 3.1 Pro, the strongest direct model in our benchmark at 76.4% overall, only reaches 55.2% on ambiguous-first. GPT-5 mini scores highest (57.3%) but this is still well below any model's performance on clear turns.

Two reasons compound to produce this failure mode. First, the model has no mechanism to recognize it's in an ambiguous situation before committing to a tool call. It receives the utterance, generates logits over the tool vocabulary, and picks the argmax. The prompt can say "call `handle_ambiguity` when uncertain," but without a prior classification step, the model can't reliably distinguish uncertain from certain contexts.

Second, ambiguous turns expose the model to high-entropy tool distributions. On a clear expansion request, the probability mass concentrates on `expand_content`. On *"can you check on that post?"*, the mass spreads across `read_post`, `check_platform`, `analyze_content`, `search_posts`, and `handle_ambiguity`. The model picks the plurality winner, which may or may not be the right call.

Under direct tool-calling, the most common errors on ambiguous Hugo turns were:
- `ambiguous` → `search_posts` (97 occurrences across models/seeds)
- `ambiguous` → `read_post` (78 occurrences)
- `ambiguous` → `analyze_content` (58 occurrences)
- `ambiguous` → `publish_post` (57 occurrences)

The model defaults to semantically plausible tools (reading, searching, analyzing) because those are the most likely responses to underspecified inputs. But "likely" and "correct" diverge badly when the right action is to ask a clarifying question.

---

## 4. The Pipeline Solution: Catching Ambiguity Early

The pipeline addresses this at the architectural level. **Flow detection** runs before tool selection. When the flow detector is uncertain about which specific user intent applies, the system can route to an ambiguity handler instead of guessing at a tool.

The mechanism: the flow detection stage produces a confidence signal over the ~42 candidate flows. When confidence is low, as it will be for underspecified inputs, the system routes to the `ambiguity_handler` intent rather than propagating uncertainty down to the tool-selection stage.

> You can't fix the ambiguity problem at the tool-selection layer, because the model at that layer has no access to the information that would tell it something is ambiguous. You have to fix it upstream.

For ambiguous-first turns, our flow detection ensemble achieved **78.8% accuracy**, a ~31 percentage point improvement over the direct baseline.

That's still imperfect. 78.8% flow detection on ambiguous-first leaves 21.2% of cases reaching the tool-selection stage unresolved. But it is achievable with a dedicated ensemble rather than hoping the base model self-reports uncertainty.

**Figure 2**: *Pipeline E2E vs. direct accuracy, with ambiguous categories highlighted. The pipeline's advantage is largest on the turn categories where direct tool-calling breaks down.*

For other turn categories, the flow detection ensemble performs better by 15+ percentage points: 93.4% on same-flow and ambiguous-second, 92.8% on switch-flow. The 78.8% on ambiguous-first reflects the difficulty of the case. These are utterances where even human annotators might disagree about intent. And it's still better than what direct mode achieves.

---

## 5. Can You Prompt Your Way Out of This?

The natural counter-proposal: add an explicit instruction to the direct prompt.

We tested this in our Experiment 2C, adding the following to the system prompt:

> *"When a request is ambiguous or underspecified, prefer calling `handle_ambiguity` rather than guessing at the user's intent."*

**Figure 3**: *Hint ablation: direct vs. hint accuracy by turn category, across 3 models (Haiku, Gemini Flash, Sonnet). Per-category deltas show the hint's effect is minimal on ambiguous turns and introduces regression on some clear-turn categories.*

The results across 3 models:

| Model | Flat Acc | Hint Acc | Delta |
|-------|---------|---------|-------|
| Haiku 4.5 | 54.6% | 59.7% | +5.1% |
| Gemini 3 Flash | 73.0% | 74.2% | +1.3% |
| Sonnet 4.6 | 61.5% | 63.7% | +2.1% |

The average improvement is **+2.8%**, which is modest and concentrated in overall accuracy rather than in ambiguous-turn categories. Looking at ambiguous-first specifically:

- Haiku: 37.5% → 49.2% (+11.7%)
- Gemini Flash: 47.7% → 52.1% (+4.5%)
- Sonnet: 43.5% → 48.4% (+4.9%)

There's a real signal here. The hint does help Haiku on ambiguous-first, but note the side effect: Gemini Flash's ambiguous-second accuracy *dropped* from 77.6% to 72.7% with the hint. The model over-triggers `handle_ambiguity` on turns that aren't actually ambiguous.

This is the core problem with the prompt approach: **the model lacks the context to know when it's in an ambiguous situation**. The prompt tells it to be cautious, but without a prior classification signal, the model sometimes applies that caution to clear requests too, which degrades performance on the categories where it was already doing well.

---

## 6. Designing for Uncertainty

The pipeline provides three things direct tool-calling doesn't.

First, flow detection produces a distribution over flows rather than a point estimate. Low confidence is a signal, not noise. Second, the ambiguity handler is accessible before the tool-selection stage, not nominally available as one of 56 tools. Third, detecting ambiguity and selecting a tool are two distinct decisions, handled by two distinct stages.

The practical recommendation for agent builders: if your system has a "clarify" or "ask_followup" tool, make it reachable before tool selection, rather than relying on prompt instructions alone. An upstream classifier that routes to clarification before the tool-selection stage is doing work that the tool-selection model cannot replicate through prompting alone.

This generalizes beyond ambiguity handling. The flow detection stage is doing intent disambiguation through a dedicated classification step rather than being folded into the tool-selection inference. Any multi-intent agent where users can make underspecified requests should consider a similar architecture.

The broader principle: identify which decisions require routing and which decisions require content understanding. Flat tool-calling conflates both. The pipeline separates them.

---

## 7. Limitations and Open Questions

- Our ambiguous turn categories were **synthetically constructed** to have specific ambiguity properties. Real-world ambiguity has a different distribution: more diverse, less controlled, potentially higher entropy.
- The **78.8% flow detection accuracy** on ambiguous-first is imperfect. The ensemble approach helps, but there is meaningful headroom remaining.
- We tested prompt-level intervention with a single instruction variant. **Chain-of-thought**, **few-shot examples of ambiguity recognition**, or **retrieval-augmented context injection** may achieve larger gains via prompting than our simple hint experiment suggests. We did not test these.
- The study measures routing accuracy, not downstream quality. Even when the pipeline routes correctly to the ambiguity handler, the quality of the clarification question generated is a separate issue we did not evaluate.
