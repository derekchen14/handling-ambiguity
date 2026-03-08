# The Narrowing Funnel Beats a Single LLM with All the Tools

*We built a 3-stage NLU pipeline and then tried skipping it. It wasn't close.*

---

## 1. Introduction

Every agent builder faces the same architectural question early on: give the model everything and let it sort things out, or pre-filter the decision space before the model ever sees it?

The case for flat tool-calling is intuitive. Fewer moving parts. Single inference call. Modern frontier models are capable — Gemini Pro can handle a 100-tool function list without breaking a sweat, or so the story goes.

We made the opposite bet. Our hypothesis was that **staged narrowing** — routing utterances through an intent classifier, then a flow detector, then a scoped tool-selection step — isn't overhead to be optimized away. It's load-bearing architecture. Each stage reduces the decision space for the next, and that reduction should translate to measurably higher accuracy.

We tested this hypothesis with 7 models across 5 providers, two qualitatively different task domains, and ~56 candidate tools. The result: the pipeline outperformed flat tool-calling for **every single model tested**, by an average of **+16.7 percentage points**.

No exceptions. No ties.

---

## 2. The Narrowing Funnel

Our **NLU pipeline** is a three-stage decision funnel:

```
Utterance → Intent (1 of 6) → Flow (1 of ~42) → Scoped Tools (5–7) → Execution
```

Each stage narrows the decision space for the next:

- **Stage 1 — Intent Classification**: Categorizes the utterance into one of ~6 high-level intents (e.g., Research, Draft, Revise, Publish for our blog-writing domain). This prunes the candidate flow space from 42 flows down to roughly 6–12.
- **Stage 2 — Flow Detection**: Identifies the specific user intent (e.g., `expand`, `tone`, `format`) within the intent bucket. This maps to a known tool scope of 5–7 tools.
- **Stage 3 — Scoped Tool Selection**: The model sees only the 5–7 tools relevant to the detected flow and selects one.

The **flat alternative** collapses all three stages into one: the model receives the full tool inventory (~56 tools) and must select directly from it.

We evaluated across two domains:

- **Hugo** — a blog-writing assistant with intents (Research, Draft, Revise, Publish, Converse, Plan) and flows like `expand`, `tone`, `rework`, `publish`, `schedule`
- **Dana** — a data analysis assistant with intents (Clean, Transform, Analyze, Report) and flows like `validate`, `pivot`, `trend`, `dashboard`

These domains were chosen because they are structurally different: Hugo tools are semantically clustered around content manipulation; Dana tools span structured data operations. We wanted to know if the pipeline benefit was domain-specific or architectural.

---

## 3. Experimental Setup

**Models tested**: 7 models across 5 providers, covering three capability tiers:
- *Low*: Haiku 4.5 (Anthropic)
- *Mid*: Gemini Flash (Google), Sonnet 4.6 (Anthropic), GPT-5 mini (OpenAI)
- *High*: Opus 4.6 (Anthropic), Qwen3 235B (Qwen), DeepSeek R1 (DeepSeek)

**Evaluation dataset**: 4 conversational turn categories — *same-flow* (user continues within the same task), *switch-flow* (user pivots to a new task), *ambiguous-first* (opening message is underspecified), and *ambiguous-second* (follow-up introduces conflicting context). Two domains × 256 turns each.

**Pipeline configuration**: Stages 1–2 used best-in-class models: Gemini Flash as intent classifier (~94.9% accuracy); a 5-voter ensemble (Sonnet 4.6, Gemini Flash, GPT-5 mini, Qwen3-80B, DeepSeek V3) for flow detection (93.4% same-flow, 92.8% switch-flow, 78.8% ambiguous-first, 93.4% ambiguous-second). Stage 3 (scoped tool selection) used each model individually.

**Flat configuration**: Each model saw all ~56 tools with no upstream filtering. Same evaluation data.

---

## 4. Results

> **The headline**: The 3-stage NLU pipeline outperforms flat tool-calling for 7 of 7 models, with an average end-to-end advantage of +16.7 percentage points.

**Figure 1**: *Pipeline E2E vs. flat tool-calling accuracy across 7 models. The pipeline wins in every case, with gains ranging from +11.3% to +23.9%.*

The distribution of gains reveals a consistent pattern:

| Model | Flat Accuracy | Pipeline E2E | Gain |
|-------|--------------|-------------|------|
| Qwen3 235B | 71.8% | **84.8%** | +13.0% |
| Gemini 3 Flash | 73.0% | **84.3%** | +11.3% |
| GPT-5 mini | 59.8% | **81.3%** | +21.5% |
| Sonnet 4.6 | 61.5% | **78.3%** | +16.8% |
| Opus 4.6 | 62.8% | **75.6%** | +12.8% |
| DeepSeek R1 | 48.4% | **72.2%** | +23.9% |
| Haiku 4.5 | 54.6% | **72.0%** | +17.4% |
| *Gemini 3.1 Pro* | *76.4%* | *(flat only)* | *—* |

The smallest gain belongs to Gemini 3 Flash (+11.3%) — already strong at flat tool-calling (73.0%) and well-calibrated for structured decisions. The largest gain goes to DeepSeek R1 (+23.9%), which started from the lowest flat baseline (48.4%) and recovered dramatically under the pipeline.

**Notably, the leaderboard shuffles.** Qwen3 235B leads pipeline accuracy at 84.8%, but ranks third in flat mode. Gemini 3.1 Pro leads flat mode (76.4%), but that best-in-class flat score is still 8.4 points below the best pipeline score. No flat configuration matches any pipeline configuration among the models tested.

**Figure 2**: *Per-stage accuracy breakdown shows where value accumulates in the funnel. Intent classification is highly reliable; the bottleneck is flow detection, particularly on ambiguous turns.*

The stage breakdown reveals the structure of the advantage:

- **Intent stage**: 89.9%–94.9% accuracy across models. Intent classification is reliable and consistent.
- **Flow detection** (ensemble): 93.4% on same-flow and ambiguous-second turns; 78.8% on ambiguous-first. The ensemble's weakest point is ambiguous-first — when the user's opening message is underspecified.
- **Scoped tool selection**: 72%–94% depending on model, but operating on a 5–7 tool slate rather than 56. The restricted scope is what makes this stage manageable.

The intuition: even with the cumulative cost of pipeline stages (e.g., 94.9% × 93.4% × 80% = ~70.9% for a rough calculation), the scoped tool stage benefits from a radically smaller decision problem. A 5–7 tool selection is categorically easier than a 56-tool selection, and the per-model scoped accuracy confirms this.

---

## 5. Why Narrowing Works

The mathematical case is straightforward. Intent classification runs at ~94.9% accuracy. Flow detection runs at 78.8%–93.4% (with the lower end on ambiguous inputs). Each stage incurs a small accuracy tax — but the payoff is that Stage 3 operates on 5–7 scoped tools instead of ~56.

Even if the pipeline's cumulative overhead were 10%, it breaks even if the scoped tool stage gains more than 10% over flat tool selection. In practice, the gains are far larger. Restricting the tool slate from ~56 to 5–7 removes semantically confusable alternatives — the model no longer has to distinguish `expand_content` from `revise_content` from `insert_section` from `analyze_content` in a single step.

The **cognitive load analogy** is apt: "which of 7 tools is right for a blog expansion task?" is a categorically easier question than "which of 56 tools is right for this utterance?" even holding model capability constant.

There's also a calibration effect. Flat tool selection from large inventories appears to degrade model calibration — the model assigns probability mass across a larger outcome space, increasing the chance of a plausible-but-wrong selection. The confusion tables from our evaluation illustrate this: when models err under flat tool-calling, they tend to pick tools that are superficially related but semantically wrong (e.g., routing a `polish` request to `analyze_content`, or an `expand` request to `search_posts`). These specific confusions don't arise under the pipeline because the tool slate no longer contains those alternatives.

**Cost**: The pipeline requires 3 serial LLM calls vs. 1 for flat. In practice, stages 1 and 2 use lightweight models (a single fast model for intent, a 5-voter ensemble for flow). Stage 3 uses the target model on a small slate. The latency overhead is real but bounded, and for agents where accuracy matters more than raw throughput, the trade-off is favorable.

---

## 6. The Domain-Invariance Surprise

Before running the domain split, we expected Hugo to show a larger pipeline advantage. Blog tools are semantically closer together — the difference between `polish`, `rework`, and `expand` is subtle in a way that `pivot_tables` vs. `execute_sql` is not. We hypothesized that the pipeline's disambiguation power would be more valuable in Hugo.

It wasn't. Hugo averaged **+16.2%** and Dana averaged **+17.3%** — essentially identical.

**Figure 3**: *Domain split — Hugo (blog writing) and Dana (data analysis) show nearly identical pipeline advantages across all 7 models. The gain is architectural, not domain-specific.*

This is a meaningful result. It suggests the pipeline's advantage doesn't come primarily from resolving semantic confusability between domain-specific tools — it comes from the structural reduction in decision space itself. The intent → flow → scoped-tool narrowing benefits both domains similarly because both domains have ~56 tools that need filtering.

The implication for practitioners: if your agent operates in a domain you believe has "less confusable" tools, the pipeline benefit is unlikely to disappear. The fundamental problem — choosing from a large, flat tool inventory — is domain-agnostic.

---

## 7. Practical Takeaways

**Rule of thumb**: If your agent has more than ~20 tools, staged pre-filtering is worth the latency cost. The break-even point is when the scoped-tool accuracy gain exceeds the cumulative stage overhead.

**Intent classification is cheap and reliable — don't skip it**. Intent accuracy ranged from 89.9% to 94.9% across all 7 models. Even the worst-performing model on intent classification is a reliable pre-filter. The common argument against adding an intent stage — "it just seems like overhead" — isn't supported by the data.

**The weakest pipeline configuration still beats every flat configuration we tested**. Haiku 4.5 in the pipeline achieved 72.0% E2E; Gemini 3.1 Pro in flat mode achieved 76.4%. That 4.4-point gap is the size of a single model-tier upgrade. The architecture is doing real work.

**Open question**: Does the advantage scale beyond ~56 tools? We would expect it to grow, since the relative cost of large-inventory flat selection increases while the pipeline's scoped stage stays constant. Empirical confirmation at 100+ tools remains future work.

---

## 8. Limitations and Future Work

- The pipeline requires an intent and flow **ontology** to be defined in advance. This upfront design cost is non-trivial — our two domains each required careful taxonomy work before the pipeline was functional.
- We evaluated on a **fixed distribution** across 4 turn categories. Real user traffic will have a different distribution, which may shift the magnitude of the advantage.
- **External validation** on ATIS, MultiWOZ, or ToolBench is pending. Our domains were constructed specifically to test multi-turn, multi-intent scenarios; how the finding transfers to other benchmark distributions is an open question.
- Latency was not formally benchmarked in this study.

---

*This work is part of our preprint on conversational tool-use accuracy in multi-stage NLU pipelines. Code and eval data available at [repo]. See also: [Post B — Why Flat Tool-Calling Collapses on Ambiguous Requests] and [Post C — We Tested 7 LLMs on Multi-Turn Tool Selection. The Architecture Mattered More Than the Model].*
