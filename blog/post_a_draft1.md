# Improving Model Reliability in Ambiguous Situations: A Case Study on AI Agents

*When the model doesn't know what it doesn't know, you get confident misdirection — as low as 37.5% accuracy on ambiguous turns.*

---

## 1. Introduction

*"Can you check on that last post?"*

An agent built on direct tool-calling will respond immediately to this. It will pick the most statistically probable tool — perhaps `read_post`, perhaps `analyze_content`, perhaps `check_platform` — and execute it. The response will be fluent. The parameters will look plausible. Nothing in the output will signal that the agent picked the wrong intent entirely.

This is the failure mode we set out to characterize: **confident misdirection**. Not hallucination, not a formatting error, not a refusal. A grammatically correct, professionally formatted response to the wrong task.

Every agent builder faces the same architectural question early in design: give the model everything and let it figure it out, or pre-filter the decision space before tool selection happens? The case for the simpler path is real — fewer components, single inference call, and modern frontier models are genuinely capable. If you ask Claude or GPT-5 to pick from 56 tools in one shot, it usually gets close.

Our case for a multi-stage pipeline is that "usually gets close" is structurally insufficient when users communicate the way people actually communicate — implicitly, with shared context assumed, with follow-up messages that reframe rather than restate. The pipeline's stages are not overhead. They are the mechanism by which the system develops something like theory of mind: the ability to track what the user believes, what the agent knows, and when those two things are not yet aligned.

We tested both approaches across 15 models from 5 providers and 3 capability tiers. Results reported here cover the top 8 by overall accuracy; the full sweep follows the same pattern. The pipeline outperformed direct tool-calling for every model reported, with an average improvement of **+15.9 percentage points**. No exceptions. We will explain why.

---

## 2. Ambiguity Occurs in Dialogue

Ambiguous requests are not pathological inputs. They are how people naturally talk to systems they expect to understand them.

Consider four archetypes:

- **same-flow**: The user continues a task already in progress. *"Same thing for the intro section."* The referent is implicit; the intent is unambiguous to a system with context.
- **switch-flow**: The user changes direction. *"Actually, hold off on that — when is this scheduled to go live?"* Intent changes; the prior task is suspended.
- **ambiguous-first**: The opening message is underspecified. *"Can you help me with the developer tools post?"* Could mean write it, revise it, research it, or schedule it.
- **ambiguous-second**: A follow-up introduces context that conflicts with or overrides the apparent trajectory. *"Wait, the parmesan brand is all wrong! Are you kidding me? The brand manager will kill me."* The user's panic signals something — but what?

The last two categories are structurally unavoidable in any real deployment. Users rely on anaphora ("that last one"), implicit frames ("can you check on it?"), and contextual shortcuts that any human collaborator would understand. The agent that cannot track this context cannot serve as a reliable collaborator. These are not edge cases — they appear in every extended conversation.

The problem for direct tool-calling: recognizing that a request is ambiguous and selecting the right tool are two separate cognitive tasks that get conflated into a single inference step. When the model can't tell whether "check on that post" means read, analyze, or verify publication status, it assigns probability mass across the tool vocabulary and picks the most probable — which is not the same as asking the user what they actually meant.

---

## 3. Problem Setup

We evaluated two approaches — direct tool-calling and a multi-stage pipeline — on a conversational tool-use benchmark spanning two task domains.

**Domains**:
- **Hugo** (blog writing): intents include Research, Draft, Revise, Publish; 42 flows; ~56 tools total
- **Dana** (data analysis): intents include Clean, Transform, Analyze, Report; 48 flows; comparable inventory

Both domains were chosen to be qualitatively different. Hugo tools are semantically dense (the difference between `polish` and `rework` is subtle). Dana tools span structured data operations with more distinct action types. We wanted to know whether any pipeline benefit was domain-specific or structural.

**Models**: We ran 15 models across 5 providers and 3 capability tiers. Results below report the top 8 by overall performance to keep the comparison tractable; the full ranking follows the same pattern.

| Level | Anthropic | Google | OpenAI | Qwen | DeepSeek |
|-------|-----------|--------|--------|------|---------|
| **Low** | Haiku 4.5 | Gemma 27B | GPT-5 nano | Qwen2.5-7B | DeepSeek Chat |
| **Mid** | Sonnet 4.6 | Gemini 3 Flash | GPT-5 mini | Qwen3-80B | DeepSeek Chat |
| **High** | Opus 4.6 | Gemini 3.1 Pro | GPT-5.2 | Qwen3-235B | DeepSeek R1 |

**Evaluation**: 256 turns per domain, balanced across the four turn categories (same-flow, switch-flow, ambiguous-first, ambiguous-second). Each turn required a single tool call. Accuracy = correct tool selected.

**Direct condition**: each model receives all ~56 tools per turn with no upstream filtering.

**Pipeline condition**: utterance passes through intent classification (~94.9% accuracy using Gemini Flash) → flow detection (5-voter ensemble, 78.8%–93.4% accuracy depending on turn type) → per-model scoped tool selection (5–7 tools).

---

## 4. Can You Prompt Your Way Out of This?

The natural first hypothesis: modern models are specifically trained for tool-calling. They should handle this already.

**Figure 1**: *Direct tool-calling accuracy by turn category per model. Ambiguous-first is consistently the hardest category across all models.*

The data tells a different story. When models select tools directly from the full ~56-tool inventory, accuracy by category looks like this:

| Category | Range across models | Average |
|----------|---------------------|---------|
| switch-flow | 56.3%–87.2% | 74.3% |
| same-flow | 51.8%–84.9% | 68.3% |
| ambiguous-second | 45.1%–78.1% | 63.9% |
| **ambiguous-first** | **37.5%–57.3%** | **47.2%** |

The drop on ambiguous-first is consistent across all models — it is not a weakness of any particular provider or tier. Gemini 3.1 Pro, the best-performing model overall, reaches only 55.2% on ambiguous-first. Haiku 4.5 drops to 37.5%. High-tier reasoning models like DeepSeek R1 hit 40.4%.

**Why?** The model must simultaneously (1) detect that the request is ambiguous and (2) decide what to do about it — all in a single forward pass over a 56-tool vocabulary. There is no structural moment for uncertainty to surface. The model assigns logits, picks the argmax, and executes.

The natural follow-up: what if the prompt explicitly instructed the model to recognize ambiguity?

We tested this (Experiment 2C) by adding an explicit instruction: *"When a request is ambiguous or underspecified, prefer calling `handle_ambiguity` rather than guessing at the user's intent."*

The result — **+2.8% average improvement overall**:

| Model | Direct | With Hint | Delta |
|-------|--------|-----------|-------|
| Haiku 4.5 | 54.6% | 59.7% | +5.1% |
| Gemini 3 Flash | 73.0% | 74.2% | +1.3% |
| Sonnet 4.6 | 61.5% | 63.7% | +2.1% |

On ambiguous-first specifically: Haiku improves from 37.5% → 49.2% (+11.7%), Flash from 47.7% → 52.1% (+4.5%), Sonnet from 43.5% → 48.4% (+4.9%).

The hint helps — but it also introduces regression on unambiguous turns. Gemini Flash's ambiguous-second accuracy *dropped* from 77.6% to 72.7% with the hint. The model over-triggers `handle_ambiguity` on clear requests, because the prompt makes it cautious but gives it no structural signal to distinguish cautious-warranted from cautious-unwarranted situations.

The problem is not the instruction. The problem is that without a prior classification step, the model cannot reliably tell when it is in an ambiguous situation. Structural problems require structural solutions.

---

## 5. The Pipeline Solution: Catching Ambiguity Early

The key architectural insight: **flow detection precedes tool selection**. The model's uncertainty about which specific user intent applies is captured as a classification confidence signal — before any tool is called.

When flow detection confidence is low (as it will be for underspecified inputs), the pipeline routes to an ambiguity handler. Tool selection is never reached. The agent asks a clarifying question instead of guessing.

**Figure 2**: *Pipeline E2E vs. direct tool-calling accuracy per model. The pipeline wins in every case.*

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

8 of 8 models improve. Average gain: **+15.9 percentage points**.

**How the stages work** (Figure 3):

- **Intent classification** (~94.9% accuracy for Gemini Flash, 89.9%–94.7% across all models): Maps the utterance to one of ~6 high-level intents. This prunes 42 candidate flows to roughly 6–12. Intent classification is consistent and reliable across all models tested.
- **Flow detection** (78.8%–93.4% by turn type, using a 5-voter ensemble): Identifies the specific user intent within the intent bucket. At this stage, if confidence is low, the ambiguity handler is invoked — the model never sees the downstream tool list.
- **Scoped tool selection** (per-model, now operating on 5–7 tools): The model selects from a curated slate that corresponds to the detected flow. Accuracy is substantially higher here than in the direct condition because confusable alternatives are absent.

The bottleneck in the direct condition is tool selection from ~56 options. The bottleneck in the pipeline is flow detection, particularly on ambiguous-first turns (78.8%). Even this imperfect flow detection is a structural improvement over direct tool-calling's 47.2% average on the same turns.

The implication: model choice matters most at the scoped tool selection stage. Intent classification is reliable across all models; it does not require the most capable model. The ensemble architecture at the flow detection stage is where architectural investment pays off.

---

## 6. Discussion

### a. Is this just hallucination by another name?

No. Hallucination is the generation of content that doesn't correspond to reality — fabricated facts, invented citations, wrong token sequences. What we're measuring is something different: **wrong intent routing**. The content of the agent's response may be entirely grounded and fluent. The problem is that it is answering the wrong question.

More precisely: hallucination typically arises from the model assigning too-high probability to low-probability tokens given the evidence. What we're describing is the model assigning high probability to a plausible tool given an underspecified context — it's confident, and the confidence is not obviously wrong from the output surface. This makes the failure mode harder to detect in production.

The ambiguous-first accuracy of 47.2% (direct, averaged across models) means nearly half of all underspecified first turns are silently misrouted. No error is raised. The agent proceeds.

### b. Won't model tool-calling accuracy just improve over time?

Two counterpoints.

First, the pipeline *also* does tool-calling. As models improve their tool selection capabilities, the pipeline's scoped tool stage benefits equally. The architecture does not compete with model improvement — it compounds with it.

Second, the pipeline provides practical engineering benefits orthogonal to accuracy:
- **Modularity**: each stage can be iterated, monitored, and improved independently
- **Interpretability**: intent and flow labels are human-readable routing artifacts
- **Debuggability**: misclassifications are observable at the stage where they occur, not just at the final output
- **Parallelism**: teams can work on intent classification, flow detection, and tool design independently

Third — and most importantly — the pipeline's core contribution is architectural: it explicitly models what the user wants to do (flow detection) as a first-class variable. This is a form of **theory of mind** applied to agent design. The system maintains a representation of the user's current intent and tracks uncertainty about it. Direct tool-calling conflates intent detection with tool selection; the pipeline separates them. That separation is load-bearing regardless of how good the underlying models get.

### c. Doesn't building ontologies go against the Bitter Lesson?

The Bitter Lesson argues that learned representations outperform hand-crafted ones at scale. Building intent taxonomies and flow ontologies by hand looks like the kind of manual feature engineering that the lesson warns against.

Two responses.

First, the process is now largely automated. Given any domain, a competent LLM can propose an intent/flow ontology, and a second LLM acting as a critic — given a rubric of common failure modes (slot masquerading as flow, near-synonym flows, too-simple flows) — can iterate the proposal to convergence. This actor-critic loop typically resolves in less than a day. The "manual cost" is much lower than it appears.

Second, this approach benefits directly from the Bitter Lesson trends. More compute means a better critic. More data means better actor proposals. More search means better ontology coverage. The pipeline's structure is a scaffold for directing scale, not a resistance to it.

---

## 7. Practical Takeaways

**Limitations**:
- Flow detection accuracy on ambiguous-first turns is 78.8% — meaningful headroom remains. Retrieval-augmented context could plausibly help; we have not tested this.
- The flow detection ensemble was hand-selected; a trained classifier might outperform it. Training data for flow detection is itself an investment.
- Does this generalize beyond ~56 tools? We expect the advantage to grow, since the difficulty of direct tool selection scales with inventory size while the pipeline's scoped stage stays constant.

**Rules of thumb**:
- If your agent has more than ~20 tools, a pre-filtering stage is worth the latency cost. Intent classification runs at 89–95% accuracy across models; don't skip it because it seems like overhead.
- Even the weakest pipeline configuration (Haiku 4.5, 72.0% E2E) outperforms the second-best direct configuration (Gemini 3 Flash, 73.0% direct) in this benchmark. The architecture compounds with model choice rather than replacing it.
- Cost the pipeline stages intelligently: intent classification can use a small, cheap model. The target model does only scoped tool selection — the stage where it actually needs to reason.

---

## 8. Conclusion: Designing for Uncertainty

On this benchmark, the same model (Gemini 3.1 Pro) ranked first under direct tool-calling and would have ranked first under the pipeline as well — but its pipeline score (87.1%) exceeded its direct score (76.4%) by 10.7 percentage points. DeepSeek R1 moved from last place (48.4% direct) to a mid-tier position (72.2% pipeline) — a 23.9-point gain from architecture alone.

What the pipeline provides that direct tool-calling does not:

1. **A confidence signal**: flow detection produces a distribution over intents, not just a point estimate. Low confidence is actionable.
2. **A dedicated routing path**: the ambiguity handler is structurally reachable before tool selection — not just one of 56 options in a flat list.
3. **Persistent context**: the flow state carries conversation history into each routing decision.

The practical recommendation: before upgrading the model tier, consider restructuring the architecture. Making the ambiguity handler a first-class routing destination — reachable from a classification stage, not invocable only via prompt instruction — yields larger accuracy gains than moving between adjacent model tiers in our benchmark.

The broader principle is this: **structural problems require structural solutions**. A prompt that says "ask when uncertain" cannot give the model the information it would need to reliably know when uncertainty is warranted. A flow detection stage that explicitly models intent uncertainty can. That is theory of mind applied to agent design.

---

*This work is part of our preprint on conversational tool-use accuracy in multi-stage NLU pipelines. Code and eval data available at [repo]. See also: [Post B — Draft 2].*
