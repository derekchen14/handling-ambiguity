# Patterns to Avoid

In order to write effectively, we must minimize these patterns. Your job as the Editor Agent is to detect these patterns and correct them.

Process:
  1. Read the text and all 20 patterns
     a. Most patterns are acceptable in small doses. The problem is when they are overused.
     b. Thus, we want to count the number of times each pattern appears and compare it to the allowed threshold
  2. Keep a tally in a temp file for each of the occurrences. Each case should include:
     a. The line number(s)
     b. A brief description of how to fix it
     c. A proposal of the rewritten sentence or section
  3. If the number of pattern occurrences exceeds the threshold, then send those violators back to the Writer Agent for revision
     a. If the threshold is 2, and the number of occurrences is 3, then send back the one violation which is most egregious
     b. If the threshold is 4, and the number of occurrences is 3, then there is nothing to do for this pattern
     c. If the threshold is 0, that means every single occurrence should be fixed

---

## Word Choice

### 1. Magic Adverbs

Don't reach for adverbs like "quietly", "deeply", "fundamentally", "remarkably", or "arguably" to make mundane descriptions feel significant. If something is important, show it — don't signal it with an adverb.

**Examples:**
- "quietly orchestrating workflows, decisions, and interactions"
- "the one that quietly suffocates everything else"
- "a quiet intelligence behind it"

**Fix:**
- Replace with more specific or vivid language
  - "the one that quietly suffocates" -> "the one that slowly smothers"
  - "a quiet intelligence" -> "an unnoticed intelligence"
- Simply remove the adverb

**Threshold:** 3 occurrences

### 2. Overused Vocabulary

Don't use "delve", "certainly", "utilize", "leverage" (as a verb), "robust", "streamline", or "harness". Prefer plain, specific alternatives. Also minimizes the use of "tapestry" and "landscape", "paradigm", "synergy", "ecosystem", or "framework".

**Examples:**
- "Let's delve into the details..."
- "Delving deeper into this topic..."
- "We certainly need to leverage these robust frameworks..."
- "The rich tapestry of human experience..."
- "Navigating the complex landscape of modern AI..."

**Fix:**
- Replace with more specific or vivid language
  - "Delving deeper" -> "Exploring further"
  - "Leverage" -> "Use"
  - "Robust" -> "Strong"
- Use a simple synonym

**Threshold:** 4 occurrences

### 3. Pompous Verbs

Prefer "is" or "are" over pompous substitutes like "serves as", "stands as", "marks", or "represents".

**Examples:**
- "The building serves as a reminder of the city's heritage."
- "Gallery 825 serves as LAAA's exhibition space for contemporary art."
- "The station marks a pivotal moment in the evolution of regional transit."

**Fix:**
- Replace with more specific or vivid language
  - "serves as" -> "is"
  - "marks" -> "is"
  - "represents" -> "is"

**Threshold:** 4 occurrences

## Sentence Structure

### 4. Negative Parallelism

Don't frame points as "It's not X — it's Y." This creates false profundity. Don't build fake tension by negating two things before landing a point. One such construction in a piece can work; more than that insults the reader.

**Examples:**
- "It's not bold. It's backwards."
- "Feeding isn't nutrition. It's dialysis."
- "Half the bugs you chase aren't in your code. They're in your head."
- "Not a bug. Not a feature. A fundamental design flaw."

**Fix:**
- "It's not bold. It's backwards." -> "It's backwards, not bold."
- "Feeding isn't nutrition. It's dialysis." -> "Feeding is a form of dialysis, rather than nutrition."
- "Not a bug. Not a feature. A fundamental design flaw." -> "It's a fundamental design flaw, not a bug nor a feature."

**Threshold:** 1 occurrence

### 5. Rhetorical Questions

Don't pose rhetorical questions nobody asked, then immediately answer them for dramatic effect.

**Examples:**
- "The result? Devastating."
- "The worst part? Nobody saw it coming."
- "The scary part? This attack vector is perfect for developers."

**Fix:**
- "The result? Devastating." -> "The result is devastating."
- "The worst part? Nobody saw it coming." -> "The worst part is that nobody saw it coming."
- "The scary part? This attack vector is perfect for developers." -> "Scarily enough, this attack vector is perfect for developers."

**Threshold:** 1 occurrence

### 6. Filler Transitions

Don't use filler transitions that introduce points without connecting them. Cut "It's worth noting", "It bears mentioning", "Importantly", "Interestingly", and "Notably" — or restructure so the connection is explicit.

**Examples:**
- "It's worth noting that this approach has limitations."
- "Importantly, we must consider the broader implications."
- "Interestingly, this pattern repeats across industries."

**Fix:**
- "It's worth noting that this approach has limitations." -> "This approach has limitations."
- "Importantly, we must consider the broader implications." -> "We must also consider the broader implications."
- "Interestingly, this pattern repeats across industries." -> "This pattern also repeats across industries."

**Threshold:** 4 occurrences

### 7. Show, Don't Tell.

Don't append a present participle phrase to inject shallow significance. If an observation needs "highlighting its importance" tacked on, rewrite the observation so it speaks for itself to avoid superficial analysis.

**Examples:**
- "contributing to the region's rich cultural heritage"
- "This etymology highlights the enduring legacy of the community's resistance and the transformative power of unity in shaping its identity."
- "underscoring its role as a dynamic hub of activity and culture"

**Fix:**
- "contributing to the region's rich cultural heritage" -> explain why the region is culturally important
- "This etymology highlights the enduring legacy of the community's resistance ..." -> show examples of the community's resistance instead
- "underscoring its role as a dynamic hub of activity and culture" -> give concrete examples of how the hub is dynamic and active

**Threshold:** 1 occurrence

### 8. False Ranges

Only use "from X to Y" when there is a real spectrum with a meaningful middle. Don't use it to dress up a list of two loosely related things.

**Examples:**
- "From innovation to implementation to cultural transformation."
- "From the singularity of the Big Bang to the grand cosmic web."
- "From problem-solving and tool-making to scientific discovery, artistic expression, and technological innovation."

**Fix:**
- Just drop these phrases completely. They add no value.
- Find a different way to express the same idea

**Threshold:** 1 occurrence

### 9. Short Punchy Fragments

Don't use a string of very short sentences or fragments as standalone paragraphs to manufacture emphasis. Don't follow a claim with a stream of verbless gerund fragments to illustrate a point since this is an inhuman writing style. Use it sparingly and deliberately, rather than as a default cadence.

**Examples:**
- "He published this. Openly. In a book. As a priest."
- "Shipping faster. Moving quicker. Delivering more."
- "These weren't just products. And the software side matched. Then it professionalised. But I adapted."
- "Reviewing pull requests. Debugging edge cases. Attending architecture meetings."

**Fix:**
- Combine the fragments into a single sentence
- Replace the fragments with a more fleshed-out paragraph

**Threshold:** 2 occurrences

## Tone

### 10. Unncessary Metaphors

Don't assume the reader needs a metaphor to understand. Only reach for analogy when the analogy is genuinely more illuminating than the direct explanation.

**Examples:**
- "Think of it like a highway system for data."
- "Think of it as a Swiss Army knife for your workflow."
- "It's like asking someone to buy a car they're only allowed to sit in while it's parked."

**Fix:**
- Just drop the sentence since they add no value
- "Think of it like a highway system for data." -> "Data is like a highway system."

**Threshold:** 1 occurrence

### 11. False Vulnerability

Don't perform self-awareness. Simulated candor — pretending to break the fourth wall or admit a bias — reads as hollow. Real honesty is specific and has stakes; don't fake it.

**Examples:**
- "And yes, I'm openly in love with the platform model"
- "And yes, since we're being honest: I'm looking at you, OpenAI, Google, Anthropic, Meta"
- "This is not a rant; it's a diagnosis"

**Fix:**
- Just drop the sentence. That's all.
- "And yes, I'm openly in love with the platform model" -> "The platform model is wonderful."

**Threshold:** 1 occurrence

### 12. Go Directly to the Point

Don't assert that a point is obvious, clear, or simple — prove it. Telling the reader your point is clear is a signal it isn't. Don't adopt a teacher-student tone with a reader who hasn't asked for it. Don't follow the formula of acknowledging problems only to immediately dismiss them with an optimistic pivot. If there are real challenges, engage with them.

**Examples:**
- "The reality is simpler and less flattering"
- "History is clear, the metrics are clear, the examples are clear"
- "Despite these challenges, [optimistic conclusion]."

**Fix:**
- Replace the sentence with the actual point
- "The reality is simpler and less flattering" -> "The reality is... [actual point]"
- Just go directly into the point that needs unpacking

**Threshold:** 1 occurrence

### 13. Grandiose Stakes Inflation

Don't inflate the significance of every argument to world-historical scale. Match the stakes of your claims to what you're actually demonstrating. Don't open an argument by asking the reader to imagine an appealing future. Make the argument directly.

**Examples:**
- "This will fundamentally reshape how we think about everything."
- "will define the next era of computing"
- "Imagine a world where every tool you use -- your calendar, your inbox, your documents, your CRM, your code editor -- has a quiet intelligence behind it..."

**Fix:**
- The last example include em-dashes, 'quiet', and grandiose language. The whole sentence should be rewritten.
- "will define the next era of computing" -> "will impact computing in the following ways:"

**Threshold:** 1 occurrence

### 14. Vague Attributions

Don't cite unnamed authorities. If you can't name the expert, the study, or the publication, don't invoke them. Don't inflate one source into "several publications" or one person's view into a widely held consensus.

**Examples:**
- "Experts argue that this approach has significant drawbacks."
- "Industry reports suggest that adoption is accelerating."
- "Observers have cited the initiative as a turning point."

**Fix:**
- Either call out a specific expert or publication, or remove the attribution entirely
- "Experts argue that this approach has significant drawbacks." -> "This approach has significant drawbacks."

**Threshold:** 0 occurrences

### 15. Invented Concept Labels

Don't coin compound labels — "supervision paradox", "acceleration trap", "workload creep" — and treat them as established terms. Name things precisely, or make the argument without the label.

**Examples:**
- "the supervision paradox"
- "the acceleration trap"
- "workload creep"

**Fix:**
- Replace the label with a more precise description
- Try to use industry standard jargon if a concept is well established
- "the supervision paradox" -> "the supervision problem"

**Threshold:** 2 occurrences

## Formatting

### 16. Em-Dash Addiction

Use em dashes sparingly — two or three per piece at most. Don't use them as a default mechanism for asides and pivots.

**Examples:**
- "The problem -- and this is the part nobody talks about -- is systemic."
- "The tinkerer spirit didn't die of natural causes -- it was bought out."
- "Not recklessly, not completely -- but enough -- enough to matter."

**Fix:**
- Replace the em dash with a comma, or rewrite the sentence to avoid the aside entirely
- "The problem -- and this is the part nobody talks about -- is systemic." -> "The problem, which is the part nobody talks about, is systemic."
- "Not recklessly, not completely -- but enough -- enough to matter." -> "The situation was not a complete wreck, but enough so that it felt unsalvageable."

**Threshold:** 2 occurrences

### 17. Bold-First Bullets

Don't begin every bullet with a bolded phrase. If you need bullets, let the content carry the list, rather than not typographic decoration. Lists in general should be kept to a minimum.

**Examples:**
- Every single bullet point begins with a bold keyword.
- "**Security**: Environment-based configuration with..."
- "**Performance**: Lazy loading of expensive resources..."

**Fix:**
- Remove the bolding entirely, or remove the keyword at the beginning of the bullet
- Rewrite the list as a paragraph

**Threshold:** 1 occurrence

## Composition

### 18. Filler Sentences

Don't use false-suspense transitions to manufacture drama before an unremarkable point. Cut "Here's the kicker", "Here's the thing", "Here's where it gets interesting", and "Here's what most people miss". Cut "Let's break this down", "Let's unpack this", "Let's explore", "Let's dive in".

**Examples:**
- "Here's the kicker."
- "Here's the thing about AI adoption."
- "Here's where it gets interesting."
- "Let's break this down step by step."

**Fix:**
- Just drop the sentence or phrase. That's all.

**Threshold:** 0 occurrences

### 19. Duplicate Content

Don't repeat entire sections or paragraphs verbatim within the same piece. Read back what you've written before continuing. Avoid restating the summary at every level of the document. This is too much. Instead, state the summary a single time. Not every section needs a summary that repeats what was already said before.

**Examples:**
- "In this section, we'll explore... [3000 words later] ...as we've seen in this section."
- "A conclusion that restates every point already made in the previous 3000 words"
- Paragraph 3 and paragraph 17 being the same sentence reworded
- The same point, restated eight ways across 4000 words.

**Fix:**
- Remove the duplicate content entirely
- If a summary is needed, make it a new section that adds value, rather than repeating what was already said

**Threshold:** 1 occurrence

### 20. Repetitive Language

Don't repeat the same sentence opening or metaphor multiple times in quick succession. Certainly don't repeat the same phrase multiple times in a single paragraph. Whereas 'Duplicate Content' warns against repeating entire sentences, this rule is about avoiding the repetition of short phrases.

**Examples:**
- "They assume that users will pay... They assume that developers will build... They assume that ecosystems will emerge... They assume that..."
- "They could expose... They could offer... They could provide... They could create... They could let... They could unlock..."
- "They have built engines, but not vehicles. They have built power, but not leverage. They have built walls, but not doors."
- "Walls and doors used 30+ times in the same article"
- Every paragraph finds a way to say "primitives" again

**Fix:**
- Vary the sentence structure and avoid repeating the same opening or phrase.
- If a metaphor is overused, replace it with a different one

**Threshold:** 2 occurrences