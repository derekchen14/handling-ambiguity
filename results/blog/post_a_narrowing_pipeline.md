# Improving Model Reliability in Ambiguous Situations

*When the model doesn't know what it doesn't know, you get confident misdirection — as low as 37.5% accuracy on ambiguous turns.*

---

## 1. Introduction

Picture a user message: *"Can you check on that last post?"*

There's no single right answer here. "Check on" could mean read it, verify its publication status, run a quality check, or pull performance metrics. A thoughtful collaborator would ask for clarification. An agent built on direct tool-calling will pick one interpretation, commit to it, and proceed — fluently, confidently, and frequently wrong.

This is what we mean by **confident misdirection**: a response that passes every surface-quality check (grammatical, formatted, tool-schema-compliant) while serving the wrong intent. Unlike hallucination, which surfaces as factual errors in content, confident misdirection is invisible at the output layer. The agent moves on. The user has to catch it.

The design question that motivates this work: should an agent see every available tool for every turn, or should the decision space be pre-filtered based on what the agent understands about the user's current intent? The intuitive argument for the simpler approach — direct tool-calling — is that it removes moving parts. Frontier models are trained on massive tool-use datasets; they should be able to handle a 56-tool vocabulary without explicit routing help.

Our bet was different. We believed that **staged narrowing is load-bearing to understanding**, not overhead to be optimized away. Each pipeline stage is not just an accuracy filter — it is a representation of what the system believes about the user's current goal. That representation is what lets the agent recognize when it *doesn't know* what the user wants, and ask rather than assume.

We ran this comparison across 15 models from 5 providers and 3 capability tiers, reporting results here for the top 8 by overall accuracy. The pipeline outperformed direct tool-calling for every reported model. Average improvement: **+15.9 percentage points**. Here is what we found and why it matters.

---

## 2. Ambiguity Occurs in Dialogue

The failure mode we're studying is not exotic. Every real conversation with an AI agent produces ambiguous turns — inputs where the user's intent cannot be determined from the message alone.

Four categories capture the range:

**Same-flow turns** are the straightforward case: the user continues a task already underway. *"Same thing for the header section."* The prior context resolves the anaphora. Agents handle this reasonably well.

**Switch-flow turns** are trickier: the user changes direction mid-conversation. *"Actually, when does this post go out?"* The agent must recognize the pivot and reset its intent model. Most capable models manage this adequately on clear requests.

**Ambiguous-first turns** are where systems break down. The user's opening message is genuinely underspecified. *"Can you help me with the developer tools post?"* There is no prior context to disambiguate — draft it, revise it, research it, schedule it, or analyze its performance are all plausible readings. The right response is a clarifying question. Direct tool-calling produces a guess.

**Ambiguous-second turns** introduce a follow-up that reframes or conflicts with the apparent prior trajectory. The user may have seemed to be revising content, and then: *"Wait, is this already published? I need to check first."* The second message changes what was needed all along.

Categories three and four are not edge cases. They emerge from the basic fact that users communicate with AI agents the way they communicate with human collaborators: using shared context, implicit reference, and conversational shortcuts that any knowledgeable partner would understand. An agent that can't track this context will systematically misroute a non-trivial fraction of all real user requests.

---

## 3. Problem Setup

We evaluated direct tool-calling against a multi-stage pipeline on a conversational tool-use benchmark.

**Domains**:
- **Hugo** — a blog-writing assistant with ~56 tools spanning Research, Draft, Revise, and Publish workflows
- **Dana** — a data analysis assistant with a comparable tool inventory spanning Clean, Transform, Analyze, and Report workflows

These domains were chosen because they are structurally different: Hugo tools are semantically clustered around more open-ended content operations; Dana tools span structured data operations. We wanted to know if the pipeline benefit was domain-specific or architectural.

**Models**: We ran 15 models across 5 providers and 3 capability tiers. Results below report the top 8 by overall accuracy; the full 15-model ranking follows the same pattern.

| Level | Anthropic | Google | OpenAI | Qwen | DeepSeek |
|-------|-----------|--------|--------|------|---------|
| **Low** | Haiku 4.5 | Gemma 27B | GPT-5 nano | Qwen2.5-7B | DeepSeek Chat |
| **Mid** | Sonnet 4.6 | Gemini 3 Flash | GPT-5 mini | Qwen3-80B | DeepSeek Chat |
| **High** | Opus 4.6 | Gemini 3.1 Pro | GPT-5.2 | Qwen3-235B | DeepSeek R1 |

**Evaluation**: 256 turns per domain balanced across same-flow, switch-flow, ambiguous-first, and ambiguous-second categories. Accuracy is measured as: did the agent select the correct tool for the turn?

**Direct condition**: each model receives all ~56 tools and must select one per turn.

**Pipeline condition**: a three-stage architecture — intent classification → flow detection → scoped tool selection — filters the tool inventory before the model makes a selection. Stages 1 and 2 use a shared best-in-class configuration (Gemini Flash for intent, a 5-voter ensemble for flow detection). Stage 3 uses each model individually on a 5–7 tool slate.

---

## 4. Can You Prompt Your Way Out of This?

The first hypothesis to test: maybe good models with good instructions already handle this. Tool-calling is a capability that frontier labs optimize heavily. Adding an explicit prompt instruction — *"ask for clarification when uncertain"* — should be sufficient, no?

**Figure 1**: *Direct tool-calling accuracy by turn category across 8 models. The drop on ambiguous-first is consistent regardless of provider or capability tier.*

The per-category accuracy under direct tool-calling:

| Turn category | Avg accuracy | Range |
|---------------|-------------|-------|
| switch-flow | 74.3% | 56.3%–87.2% |
| same-flow | 68.3% | 51.8%–84.9% |
| ambiguous-second | 63.9% | 45.1%–78.1% |
| **ambiguous-first** | **47.2%** | **37.5%–57.3%** |

The gap is not model-specific. Gemini 3.1 Pro — the strongest model in our benchmark — achieves 55.2% on ambiguous-first. DeepSeek R1 (a high-tier reasoning model) reaches 40.4%. Haiku 4.5 falls to 37.5%. The consistent pattern across eight models from five providers rules out the "weak model" explanation. This is a structural problem.

The structural issue: the model must simultaneously detect ambiguity *and* select the right tool in a single inference step. It has no access to information that would tell it whether the current situation warrants hesitation. It sees an underspecified utterance, generates logits over a 56-item vocabulary, and picks the argmax.

What happens when you add an explicit hint? We added this to the system prompt: *"When a request is ambiguous or underspecified, prefer calling `handle_ambiguity` rather than guessing."* Three models, six seeds each.

**Figure 2**: *Hint ablation — flat vs. hint accuracy per category. Gains on ambiguous-first are small and accompanied by regression on switch-flow and ambiguous-second.*

| Model | Direct | Hint | Delta |
|-------|--------|------|-------|
| Haiku 4.5 | 54.6% | 59.7% | +5.1% |
| Gemini 3 Flash | 73.0% | 74.2% | +1.3% |
| Sonnet 4.6 | 61.5% | 63.7% | +2.1% |

Ambiguous-first improves modestly (Haiku +11.7 pp, Flash +4.5 pp, Sonnet +4.9 pp). But the hint also causes regression on unambiguous turns: Gemini Flash's ambiguous-second accuracy dropped 4.9 points. The model over-triggers ambiguity handling on turns that don't need it.

This is the fundamental limitation of prompt-level intervention: the model doesn't have the information it would need to know when the instruction applies. Telling it to be cautious makes it globally more cautious — not selectively cautious where caution is warranted. For that, you need a prior classification stage.

---

## 5. The Pipeline Solution: Catching Ambiguity Early

The pipeline's solution is architectural: **flow detection runs before tool selection**, making the system's uncertainty about user intent explicit and actionable.

The three-stage structure:

```
Utterance → Intent (1 of 6) → Flow (1 of ~42) → Scoped Tools (5–7)
```

At the flow detection stage, the 5-voter ensemble produces a confidence distribution over candidate flows. When confidence is low — as it reliably is for underspecified inputs — the system routes to the ambiguity handler before any tool list is assembled. The model never reaches the tool selection step for turns where it shouldn't guess.

> The mechanism: intent × flow is a confidence signal. When flow confidence is low, the system asks a clarifying question instead of executing a guess.

**Figure 3**: *Per-stage accuracy breakdown. Intent classification is the most reliable stage (89–95%); flow detection is the bottleneck, especially on ambiguous-first turns (78.8%).*

Each stage narrows the decision space for the next:

- **Stage 1 — Intent Classification**: Categorizes the utterance into one of ~6 high-level intents (e.g., Research, Draft, Revise, Publish for our blog-writing domain). This prunes the candidate flow space from 42 flows down to roughly 6–12.
- **Stage 2 — Flow Detection**: Identifies the specific user intent (e.g., `expand`, `tone`, `format`) within the intent bucket. This maps to a known tool scope of 5–7 tools. Notably, this can be composed using a 5-voter ensemble of individual models, each of which may be less accurate, but which together outperform the best single model on the task.
- **Stage 3 — Scoped Tool Selection**: The model sees only the 5–7 tools relevant to the detected flow and selects one.

The result:

**Figure 4**: *Pipeline E2E vs. direct tool-calling per model. 8 of 8 models improve.*

| Model | Direct | Pipeline | Gain |
|-------|--------|---------|------|
| Gemini 3.1 Pro | 76.4% | **87.1%** | +10.7% |
| Qwen3 235B | 71.8% | **84.8%** | +13.0% |
| Gemini 3 Flash | 73.0% | **84.3%** | +11.3% |
| GPT-5 mini | 59.8% | **81.3%** | +21.5% |
| Sonnet 4.6 | 61.5% | **78.3%** | +16.8% |
| Opus 4.6 | 62.8% | **75.6%** | +12.8% |
| DeepSeek R1 | 48.4% | **72.2%** | +23.9% |
| Haiku 4.5 | 54.6% | **72.0%** | +17.4% |

The gains are largest for models that struggled most under direct tool-calling (DeepSeek R1 +23.9%, GPT-5 mini +21.5%) and smallest for models that were already strong (Gemini 3.1 Pro +10.7%, Gemini Flash +11.3%). Every model benefits.

The domain split confirms that the advantage is architectural, not domain-specific: Hugo gained +15.4% on average, Dana gained +16.7%. Two structurally different tool inventories, essentially identical results.

---

## 6. Discussion

### a. This sounds like reducing hallucination — isn't that a known problem?

The failure mode here is distinct from hallucination, and the distinction matters for how you address it.

Hallucination is a content problem: the model generates tokens that don't correspond to reality — fabricated facts, wrong dates, invented citations. The model assigns too-high probability to low-probability tokens given the evidence.

What we're measuring is a **routing problem**: the model assigns high probability to a plausible tool given underspecified context. The response can be entirely factually grounded — the tool exists, the parameters are syntactically correct, the output format is right — and still serve the wrong intent. There is no factual error to check against. The only way to detect it is to know what the user actually wanted, which requires the system to track user intent explicitly.

This is why the solutions are different. Hallucination is addressed by better calibration, retrieval, and factual grounding. Confident misdirection is addressed by explicit intent modeling — which is what the flow detection stage provides.

### b. Won't model tool-calling accuracy improve and make this unnecessary?

Counterpoint one: the pipeline does tool-calling too. As models improve at tool selection, the pipeline's scoped tool stage benefits equally. The architecture compounds with model improvement rather than competing with it.

Counterpoint two: accuracy is not the only dimension that matters. The pipeline provides practical engineering properties that direct tool-calling cannot replicate regardless of model quality:

- **Modularity**: intent, flow, and tool stages are independently iterable. A regression in flow detection doesn't require retraining the tool selection component.
- **Interpretability**: intent and flow labels are human-readable. When the agent misroutes a request, the stage of failure is visible.
- **Debuggability**: production monitoring can be applied at each stage independently. Error analysis is tractable.
- **Parallelism**: separate teams can own intent taxonomy, flow detection, and tool design without coordination overhead.

Counterpoint three: the pipeline provides **theory of mind** that direct tool-calling cannot. Tracking user intent as an explicit variable — and maintaining uncertainty about that variable — is how the system knows when to ask rather than assume. That capability doesn't automatically emerge from better tool-call fine-tuning.

### c. Doesn't building ontologies go against the Bitter Lesson?

A fair challenge. The Bitter Lesson argues that hand-crafted structure loses to learned representations at scale. Building intent taxonomies by hand looks like the kind of manual feature engineering the lesson warns against.

Two responses.

First, this is now largely automated. The actor-critic pattern works well here: a frontier LLM proposes an ontology, a second LLM acting as critic — given a rubric of common failure modes (slots masquerading as flows, near-synonym flows, trivially simple flows that should be merged) — iterates toward convergence. This process typically resolves in less than a day for a new domain. The upfront cost is real but substantially lower than it appears.

Second, the pipeline does not resist scale — it benefits from it. Better models mean better ontology proposals from the actor. More compute means more thorough critique. More data means better coverage of real user request distributions. The structure is a scaffold for directing scale, not a barrier to it.

---

## 7. Practical Takeaways

**Known limitations**:

- Flow detection accuracy on ambiguous-first (78.8%) leaves headroom. Retrieval-augmented context injection — giving the flow detector access to conversation history — might improve this; we have not tested it.
- The 5-voter ensemble was selected empirically. A purpose-trained classifier might outperform it. What training data looks like for flow detection is an open question.
- We tested 8 models. A larger-scale comparison (15–30 models) would provide stronger evidence for the cross-model consistency of the finding.

**Engineering guidance**:

- If your agent has more than ~20 tools, a pre-filtering stage pays for itself in accuracy. Intent classification runs at 89–95% accuracy across diverse models; the overhead of adding it is bounded and the benefit is consistently positive.
- A model in the pipeline typically outperforms the same model in direct tool-calling by more than a full model tier's worth of capability gap. Architecture restructuring before model upgrading is a reasonable priority ordering.
- Cost the stages differently: intent classification and flow detection can run on small, cheap models. The target model's contribution is concentrated at scoped tool selection, where accuracy and quality matter most.

---

## 8. Conclusion: Designing for Uncertainty

The headline finding is clean: a multi-stage pipeline that routes uncertain inputs to an ambiguity handler before tool selection outperforms direct tool-calling for every model we tested, by an average of 15.9 percentage points.

But the more important finding is structural. The pipeline provides three capabilities that direct tool-calling lacks:

1. **A confidence signal**: flow detection produces a distribution, not just a point estimate. Low confidence is a first-class signal that something is worth asking about.
2. **A dedicated routing path**: the ambiguity handler is reachable from an upstream classification stage — not just nominally present as one item in a 56-tool menu.
3. **Conversational memory**: intent and flow context accumulate across turns, letting the system recognize when a follow-up contradicts or reframes what was previously established.

Together, these properties give the agent something like theory of mind: a representation of what the user believes they asked for, what the agent understands that to be, and — critically — when those two things might not yet be aligned.

The practical recommendation for multi-intent agent builders: if your system has an ambiguity or clarification tool, make it **structurally reachable** through an upstream routing decision — not just mentioned in the prompt. A prompt that says "ask when uncertain" gives the model an instruction it cannot reliably follow without the structural context to know when it applies. A flow detection stage gives it that context.

Structural problems require structural solutions. Intent disambiguation that relies purely on the model's internal calibration will not scale reliably to the full diversity of real user requests. A pipeline that explicitly models user intent uncertainty — and routes accordingly — will.

---

*This work is part of our preprint on conversational tool-use accuracy in multi-stage NLU pipelines. Code and eval data available at [repo]. See also: [Post A — Draft 1].*

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
