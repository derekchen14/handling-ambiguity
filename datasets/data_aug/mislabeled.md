# Mislabeled Turn Review — 53 Labels (15/15 Models Wrong)

**Note:** "15/15 models wrong" means 14 models confirmed wrong across 5 seeds each + 1a_006 (Gemini Pro) confirmed wrong on 1 seed only (incomplete data). No cases exist where Gemini Pro was the sole correct model, so the signal is robust.

**Status key:**
- `open` — not reviewed yet
- `skip` — label is correct, models just made a mistake
- `relabel` — change the gold label (flow or category)
- `rewrite` — change the utterance to preserve category balance
- `reconsider` — ontology change needed (merge/split flows)

---

## Hugo — ambiguous_first (17 labels)

### H1. hugo_018 t1 — `skip`
- **Category:** ambiguous_first
- **Utterance:** "Change the overnight train part which is pretty bare bones. Turn it into something that actually reads well?"
- **Label:** ambiguous [expand, rework]
- **Models say:** expand (37x), rework (15x), polish (9x)
- **Agent clarifies t2:** "I see bullet points and some scattered notes. I can develop those into full prose, or take a different angle entirely."
- **Recommendation:** `rewrite` — The models are split 37/15 between expand and rework, which shows genuine uncertainty. But 37/71 picking one flow isn't "ambiguous" enough since models need to return BOTH flows. Rewrite the utterance to be more balanced between the two candidates. Something like "The overnight train part is pretty thin — not sure if it needs more content or a total rewrite?" would make both flows equally plausible.
- **Feedback:** You said yourself, this shows 'genuine uncertainty'. The label is correct, and the models should pick up on the nuance.

### H2. hugo_020 t1 — `skip`
- **Category:** ambiguous_first
- **Utterance:** "Do we have anything on fermentation?"
- **Label:** ambiguous [browse, find]
- **Models say:** find (71x)
- **Recommendation:** `rewrite` — 71/71 say `find`, zero say `browse`. The distinction: `find` = search previous posts by keyword, `browse` = browse available topic ideas. "Do we have anything on X" maps perfectly to keyword search (find), not browsing ideas. To make it genuinely ambiguous, rewrite to something like "Fermentation — have I written about it, or is it just on my ideas list?" which could be either searching posts (find) or browsing saved ideas (browse).
- **Feedback:** 'fermentation' could easily be a topic idea, rather than a keyword within a post. The label is correct, and the models should pick up on the nuance.

### H3. hugo_022 t1 — `rewrite`
- **Category:** ambiguous_first
- **Utterance:** "The AI roundup is ready. Let's go."
- **Label:** ambiguous [promote, release]
- **Models say:** release (71x)
- **Recommendation:** `rewrite` — "Let's go" strongly signals immediate publish action (release). Nobody reads this as "amplify reach" (promote). To make it ambiguous: "The AI roundup is ready. Time to get it out there." — "get it out there" could mean publish (release) or amplify/share widely (promote).
- **Feedback:** Sure, we can change it to "The AI roundup is ready. Time to get it out there.", but the models should have still picked on this regardless.

### H4. hugo_023 t1 — `rewrite`
- **Category:** ambiguous_first
- **Utterance:** "Can you set up the urban living essay for Medium?"
- **Label:** ambiguous [syndicate, schedule]
- **Models say:** syndicate (49x), [format, syndicate] (19x)
- **Recommendation:** `rewrite` — "Set up for Medium" reads as cross-posting (syndicate), not scheduling for a future date. To make it ambiguous: "Can you get the urban living essay queued for Medium?" — "queued" could mean cross-post now or schedule for later.
- **Feedback:** Ok, lets rewrite to: "Can you get the urban living essay queued for Medium?"

### H5. hugo_024 t1 — `relabel`
- **Category:** ambiguous_first
- **Utterance:** "How's the appliance section looking?"
- **Label:** ambiguous [survey, inspect]
- **Models say:** inspect (34x), view (15x), [inspect, view] (12x)
- **Recommendation:** `rewrite` — Models pick inspect (content quality check) over survey (platform health), but neither candidate pair gets returned together. The "appliance section" phrasing points to content, not publishing infrastructure. To make [survey, inspect] ambiguous: "How's the appliance review looking across our channels?" — "across our channels" introduces the platform dimension that makes survey plausible.
- **Feedback:** This should be relabeled as [inspect, view]

### H6. hugo_082 t1 — `rewrite`
- **Category:** ambiguous_first
- **Utterance:** "Hey the voice in my urban living essay feels off, can you fix it"
- **Label:** ambiguous [tone, audit]
- **Models say:** tone (70x)
- **Recommendation:** `rewrite` — "Fix the voice" is unambiguously `tone` (adjust register). `audit` = check consistency with published history, which isn't signaled here. To make it ambiguous: "The voice in my urban living essay feels off — is it just this post or have I been drifting?" — "have I been drifting" introduces the audit (compare against past posts) possibility.
- **Feedback:** Change to "Hey the voice in my Urban Living essay feels off, can you fix it to follow the tone and structure from the Rural Living post?" This should make it ambiguous because elements of tone and audit are both present.

### H7. hugo_083 t1 — `rewrite+relabel`
- **Category:** ambiguous_first
- **Utterance:** "The attention mechanism section needs a lot of work before it goes out"
- **Label:** ambiguous [rework, format]
- **Models say:** rework (60x), [release, rework] (9x)
- **Recommendation:** `rewrite` — "A lot of work" clearly signals content revision (rework), not platform formatting (format). `format` = heading levels, code blocks, image placement for publication. To make it ambiguous: "The attention mechanism section isn't ready for Medium yet" — could mean content needs rework OR formatting for the platform.
- **Feedback:** relabel to [rework, tidy], rewrite to: "The attention mechanism post needs a lot of work before it goes out." This ensures that the alternate label is 'tidy', rather than 'polish'.

### H8. hugo_084 t1 — `relabel`
- **Category:** ambiguous_first
- **Utterance:** "That last change to the Thailand section wasn't right"
- **Label:** ambiguous [undo, chat]
- **Models say:** amend (35x), [amend, undo] (21x), undo (15x)
- **Recommendation:** `rewrite` — Interesting: models don't pick either candidate! They pick `amend` (push back on revision) instead. The undo/chat pair is unusual — chat is rarely the right ambiguous partner. The utterance naturally signals amend or undo, not chat. To make [undo, chat] genuinely ambiguous: "That last change to the Thailand section... I don't know about that" — vague enough to be either "roll it back" (undo) or "let's discuss what felt off" (chat).
- **Feedback:** Relabel to [undo, amend]

### H9. hugo_085 t1 — `rewrite`
- **Category:** ambiguous_first
- **Utterance:** "Can you push the investment strategies post to Medium too"
- **Label:** ambiguous [syndicate, schedule]
- **Models say:** syndicate (71x)
- **Recommendation:** `rewrite` — "Push to Medium too" clearly means cross-post now (syndicate). No time signal for schedule. Same pattern as H4. To make it ambiguous: "The investment strategies post needs to go on Medium — want to time it with the newsletter." — "time it" introduces scheduling possibility.
- **Feedback:** I have rewritten to: "Can you push the investment strategies post to Medium too. The whitepaper goes out on Friday." Now the utterance contains a time signal, which should make it ambiguous.

### H10. hugo_090 t1 — `skip`
- **Category:** ambiguous_first
- **Utterance:** "Hey can you do the authentication section for my REST API tutorial"
- **Label:** ambiguous [write, expand]
- **Models say:** write (71x)
- **Recommendation:** `rewrite` — "Can you do the section" doesn't signal whether content exists. Models default to `write` (from scratch). `expand` requires existing content. To make it ambiguous: "Hey can you flesh out the authentication section for my REST API tutorial" — "flesh out" could mean write from nothing or expand bullet points.
- **Feedback:** As you mentioned, the signals are not present in the original utterance. Depending on context, this might even be 'add' flow. Therefore, this utterance is clearly ambiguous and should be recognized as such.

### H11. hugo_093 t1 — `relabel`
- **Category:** ambiguous_first
- **Utterance:** "What's the move with the transformer section"
- **Label:** ambiguous [suggest, endorse]
- **Models say:** suggest (41x), [explain, suggest] (20x)
- **Recommendation:** `rewrite` — "What's the move" reads as "give me advice" (suggest), not "go ahead with your prior recommendation" (endorse). To make it ambiguous: "What's the move with the transformer section — should we go with what you said earlier?" — the trailing question makes endorse (accepting prior suggestion) plausible.
- **Feedback:** Relabel to [suggest, explain]

### H12. hugo_098 t1 — `rewrite` (swapped to switch_flow)
- **Category:** ~~ambiguous_first~~ → switch_flow
- **Utterance:** "the VS Code section reads kinda rough, can you clean it up"
- **Label:** ambiguous [polish, rework]
- **Models say:** polish (71x)
- **Recommendation:** `rewrite` — "Reads rough, clean it up" = light editing (polish). 71/71 agreement. To make it ambiguous: "The VS Code section reads kinda rough. Not sure if it just needs smoothing or a bigger overhaul." — explicitly raises both options.
- **Feedback:** This is ambiguous because 'clean it up' could mean light editing (polish) or heavy editing (rework). The label is correct, and the models should pick up on the nuance.
- **Swap note:** Conversation rewritten by user to be switch_flow. Category changed to balance hugo_014 moving to ambiguous_first.

### H13. hugo_099 t1 — `rewrite+relabel` (swapped to switch_flow)
- **Category:** ~~ambiguous_first~~ → switch_flow
- **Utterance:** "Hey can you push the remote work piece live"
- **Label:** ambiguous [syndicate, release]
- **Models say:** release (71x)
- **Recommendation:** `rewrite` — "Push live" = publish to main blog (release). Nobody reads "live" as cross-posting (syndicate). To make it ambiguous: "Hey can you get the remote work piece out there" — "out there" is vague enough to mean main blog or secondary platforms.
- **Feedback:** Yes, we should rewrite to: "Can I take a quick look at the remote work piece to check how long it is?" — "how long is it" could be a 'inspect' (word count) or a 'preview' (read it and see for yourself). The new label should be [inspect, preview]
- **Swap note:** Conversation rewritten by user to be switch_flow. Category changed to balance hugo_016 moving to ambiguous_first.

### H14. hugo_100 t1 — `rewrite+relabel`
- **Category:** ambiguous_first
- **Utterance:** "Hey can you put the REST API tutorial up on the blog"
- **Label:** ambiguous [release, preview]
- **Models say:** release (71x)
- **Recommendation:** `rewrite` — "Put up on the blog" = publish (release). No preview signal. To make it ambiguous: "Hey can you put the REST API tutorial up so I can see how it looks on the blog" — "see how it looks" could be preview OR publish-then-check.
- **Feedback:** Rewrite the utterance to: "Oh my gosh, it's already Tuesday, we really should have gotten the post out yesterday! Jane is gonna be so mad!" — This should make it ambiguous because the user is expressing urgency, but not specifying whether they want to release the post immediately. The label should be [release, chat]. You could even argue for 'dismiss' or 'schedule' as alternate labels, but regardless, the new utterance is clearly ambiguous.

### H15. hugo_103 t1 — `rewrite+relabel`
- **Category:** ambiguous_first
- **Utterance:** "Can you work on the attention mechanism section? It's pretty rough right now"
- **Label:** ambiguous [expand, refine]
- **Models say:** rework (26x), [polish, rework] (24x)
- **Recommendation:** `rewrite` — Models don't pick either candidate! They say rework/polish, not expand/refine. "Pretty rough" triggers Revise intent, not Draft (where expand and refine live). To make [expand, refine] genuinely ambiguous: "The attention mechanism section is just bullet points. Can you work on it?" — "just bullet points" signals Draft territory where expand vs refine makes sense.
- **Feedback:** Rewrite to "Can you fill in the details on all the sections? It's just a skeleton right now." The new labels should be [expand, rework].

### H16. hugo_093 t3 — `relabel`
- **Category:** ambiguous_first (turn 3, after clarification)
- **Utterance:** "No I want fresh ideas, what angles am I missing"
- **Label:** suggest
- **Models say:** brainstorm (70x)
- **Recommendation:** `relabel` — "Fresh ideas, what angles am I missing" is clearly `brainstorm` (generate angles, hooks, perspectives), not `suggest` (proactive next-step recommendation). The ontology distinction: brainstorm = generate creative ideas for a topic, suggest = recommend what to do next. This utterance is about content ideas, not process next-steps. Relabel to `brainstorm`.
- **Feedback:** Agreed, relabeled to `brainstorm`

## Hugo — ambiguous_second (4 labels)

### H17. hugo_029 t1 — `relabel`
- **Category:** ambiguous_second
- **Utterance:** "How's the tomato galette post looking?"
- **Label:** inspect
- **Models say:** [check, view] (13x), view (10x), preview (10x)
- **Full convo:** t1 asks about status → t2 gives metrics → t3 "sourcing note could use love before we feature it"
- **Recommendation:** `relabel` — The label says `inspect` (analyze content metrics) but models split across `check` (workflow status), `view` (view full post), and `preview`. "How's X looking?" is genuinely ambiguous between status-check and content-quality, but `inspect` specifically means "word count, reading time, readability" while `check` means "draft/published/scheduled status". The agent's response gives metrics, suggesting inspect is defensible. However, 0/71 models returned `inspect` as a single-flow answer. Consider relabeling to `check` (the most common single answer) since "how's it looking" is more naturally a status check than a deep metrics analysis.
- **Feedback:** Relabel as [check, view, inspect], which means the model should hopefully pick up on the ambiguity.

### H18. hugo_031 t3 — `relabel`
- **Category:** ambiguous_second
- **Utterance:** "That async culture draft I started. Pull it up so I can plan Part 2."
- **Label:** blueprint, candidates=[find, outline]
- **Models say:** view (54x), [outline, view] (13x)
- **Recommendation:** `relabel` — Label says `blueprint` (plan full post workflow) with candidates [find, outline]. But "pull it up" = view/find, and "plan Part 2" could be blueprint or outline. Models overwhelmingly say `view` since "pull it up" dominates the utterance. The candidates should include `view` — relabel candidates to [view, blueprint] or [view, outline]. The user wants to SEE the draft AND plan the next part.
- **Feedback:** Relabel so that the candidates are [view, outline]

### H19. hugo_032 t3 — `relabel`
- **Category:** ambiguous_second
- **Utterance:** "Also flesh out the Laos section. Then double check all the grammar and tone for it."
- **Label:** blueprint, candidates=[expand, audit]
- **Models say:** [expand, polish] (47x), [expand, polish, tone] (24x)
- **Recommendation:** `relabel` — Two issues: (1) The label says `blueprint` but models correctly detect the actual constituent flows. (2) "Double check grammar and tone" maps to `polish` or `tone`, not `audit` (which checks consistency with publishing history). Relabel candidates to [expand, polish] or [expand, tone]. The `blueprint` label also seems wrong — this is a multi-step instruction (Plan/outline territory), but the constituent flows are expand + polish, not expand + audit.
- **Feedback:** Relabel so that the candidates are [expand, polish, tone]

### H20. hugo_116 t3 — `relabel`
- **Category:** ambiguous_second
- **Utterance:** "Add a section on sirens becoming background music. Then do the same cleanup on the whole post."
- **Label:** blueprint, candidates=[write, tidy]
- **Models say:** [add, tidy] (71x)
- **Recommendation:** `relabel` — 71/71 agree on [add, tidy]. "Add a section" = `add` (create section placeholder), not `write` (write section from scratch). The ontology is clear: `add` creates an empty section with a heading, `write` generates prose from a topic. Relabel candidates to [add, tidy].
- **Feedback:** Relabel candidates to [add, tidy]

## Hugo — same_flow (1 label)

### H21. hugo_035 t1 — `relabel`
- **Category:** same_flow
- **Utterance:** "What have I got in the pipeline?"
- **Label:** browse
- **Models say:** check (71x)
- **Recommendation:** `relabel` — "What have I got in the pipeline?" asks about workflow status of posts (draft/scheduled/published). That's `check` (workflow status), not `browse` (browse topic ideas). The ontology: browse = "trending subjects, saved ideas, content gaps", check = "draft, scheduled, published, unpublished; which platforms; last edited date". Relabel to `check`. Need to also change t3 ("Just the spring stuff") which is also labeled `browse` but should be `check`.
- **Feedback:** Yup, relabel to `check`

## Hugo — switch_flow (2 labels)

### H22. hugo_014 t1 — `relabel` (swapped to ambiguous_first)
- **Category:** ~~switch_flow~~ → ambiguous_first
- **Utterance:** "The last post hit in error when trying to publish to WordPress"
- **Label:** format (Revise intent)
- **Models say:** release (49x), [format, release] (7x)
- **Recommendation:** `relabel` — The utterance describes a publish error, not a formatting request. Models say `release` (which would trigger a retry or debug of the publish action). The label `format` assumes the error was caused by bad formatting, but the user doesn't say that — they report a publish failure. However, looking at the agent's t2 response ("I have edited the code blocks... fixed heading levels... metadata fields filled in"), the conversation was designed with format in mind. The utterance just doesn't convey it. Recommend `rewrite` to: "The last post errored out on WordPress — I think the heading levels or code blocks are wrong" to make the format intent clear. OR `relabel` to `release` if we want to match what the utterance actually says.
- **Feedback:** This can be relabeled to [format, release]

### H23. hugo_016 t3 — `relabel` (swapped to ambiguous_first)
- **Category:** ~~switch_flow~~ → ambiguous_first
- **Utterance:** "Fix the dryness, pull from here: https://www.example.com/reviews/dishwashers"
- **Label:** polish
- **Models say:** rework (36x), [polish, rework] (10x)
- **Recommendation:** `relabel` or `rewrite` — "Fix the dryness" could be polish (light editing) or rework (deeper revision). But "pull from here [URL]" implies incorporating external source material, which is heavier than polish (word choice/transitions). `rework` = "restructures arguments, replaces weak sections" seems more appropriate. Relabel to `rework`. OR if we want to keep `polish`, rewrite to "The performance section is a bit dry — can you smooth out the transitions?"
- **Feedback:** Relabel to [polish, rework] then

## Dana — ambiguous_first (22 labels)

### D1. dana_017 t1 — `skip`
- **Category:** ambiguous_first
- **Utterance:** "What's gross margin in June for users who joined in Q1 cohort?"
- **Label:** ambiguous [lookup, query]
- **Models say:** query (53x), segment (9x)
- **Recommendation:** `rewrite` — "What's gross margin for X cohort in June" is a data question with filters = `query`. `lookup` = "find the definition of a metric in the semantic layer". The utterance asks for a computed number, not a definition. To make it genuinely ambiguous: "What's gross margin? I need it for the Q1 cohort in June." — first sentence could be lookup (what's the formula), second sentence adds query context.
- **Feedback:** This is an ambiguous question. The user is asking for a specific value, but it's not clear whether cohorted data is a filter that can be found in a column (simple 'lookup') or needs to be derived (which would be 'query'). Skip.

### D2. dana_017 t3 — `skip`
- **Category:** ambiguous_first (turn 3)
- **Utterance:** "You can just use the timestamp column and break it down by month."
- **Label:** lookup
- **Models say:** query (27x), segment (26x)
- **Recommendation:** `relabel` — "Break it down by month" is clearly an analysis action (query or segment), not a definition lookup. The user is giving instructions for HOW to compute, not asking WHAT a term means. Relabel to `query` or `segment`. Given the context (breaking down a metric by a dimension), `segment` is most precise.
- **Feedback:** This would be 'segment' if viewed in isolation, but in context it's a follow-up to determine how to find the cohort data. Since the cohort data can be easily filtered for by looking at the timestamp column, it should be 'lookup'.

### D3. dana_020 t1 — `rewrite`
- **Category:** ambiguous_first
- **Utterance:** "How do Instagram and TikTok stack up?"
- **Label:** ambiguous [compare, trend]
- **Models say:** compare (71x)
- **Recommendation:** `rewrite` — "Stack up" is unambiguously `compare` (side-by-side). `trend` requires a time dimension. To make it ambiguous: "How have Instagram and TikTok been doing?" — "been doing" introduces a time element that makes trend plausible alongside compare.
- **Feedback:** Alright, let's rewrite to: "How have Instagram and TikTok been doing lately?"

### D4. dana_021 t1 — `relabel`
- **Category:** ambiguous_first
- **Utterance:** "I need to send the patient outcomes to the team."
- **Label:** ambiguous [export, summarize]
- **Models say:** export (62x), [dashboard, export] (8x)
- **Recommendation:** `rewrite` — "Send to the team" strongly signals export (download file). Summarize = write up key takeaways. To make it ambiguous: "I need to share the patient outcomes with the team — they want a quick read." — "quick read" makes summarize plausible.
- **Feedback:** We can relabel to [export, dashboard]

### D5. dana_022 t1 — `rewrite`
- **Category:** ambiguous_first
- **Utterance:** "The churn table needs to stand out more."
- **Label:** ambiguous [style, design]
- **Models say:** style (67x)
- **Recommendation:** `rewrite` — `style` = conditional formatting on a table, `design` = adjust an existing chart's visuals. "The table needs to stand out" points to table formatting (style). To make it ambiguous: "The churn visual needs to stand out more." — "visual" is ambiguous between table and chart.
- **Feedback:** Rewrite to "The customer churn results needs to stand out more."

### D6. dana_024 t1 — `rewrite`
- **Category:** ambiguous_first
- **Utterance:** "Can you combine the platform and handle columns?"
- **Label:** ambiguous [merge, join]
- **Models say:** merge (71x)
- **Recommendation:** `reconsider` or `rewrite` — `merge` = combine columns within one table, `join` = combine two tables on a key. "Combine columns" is unambiguously merge. The ontology boundary is actually clear here — merge operates on columns, join operates on tables. To make it ambiguous: "Can you combine the platform data with the handle data?" — "data" is vague enough to mean columns or tables.
- **Feedback:** Rewrite to "Can you combine the platform and handle data?"

### D7. dana_081 t1 — `relabel`
- **Category:** ambiguous_first
- **Utterance:** "Can you break down the feature usage table so each feature is its own column"
- **Label:** ambiguous [reshape, describe]
- **Models say:** pivot (30x), reshape (23x)
- **Recommendation:** `rewrite` — "Each feature is its own column" = pivot/reshape (long to wide). `describe` = profile dataset stats. These aren't even close. The utterance clearly describes a structural transformation. To make [reshape, describe] ambiguous: "Can you break down the feature usage table for me?" — without the structural detail, "break down" could mean reshape OR describe.
- **Feedback:** Relabel to [reshape, pivot]

### D8. dana_084 t1 — `relabel`
- **Category:** ambiguous_first
- **Utterance:** "Hey can you check the department column in the attrition dataset, some of the values look off"
- **Label:** ambiguous [update, describe]
- **Models say:** validate (64x), [describe, validate] (7x)
- **Recommendation:** `rewrite` — Models pick `validate` (check values against valid set), which makes sense for "values look off". Neither candidate (update or describe) is what models choose. `update` = modify in place, `describe` = profile stats. To make [update, describe] genuinely ambiguous, or better yet, pick better candidates: "Hey can you look at the department column in the attrition dataset" — without "values look off", it could be describe (profile) or validate (check values).
- **Feedback:** Relabel to [validate, describe]

### D9. dana_084 t3 — `relabel`
- **Category:** ambiguous_first (turn 3)
- **Utterance:** "I already know, change "R&D" to "Research & Development" everywhere"
- **Label:** update
- **Models say:** replace (66x)
- **Recommendation:** `relabel` — "Change X to Y everywhere" is textbook `replace` (find and replace across a column), not `update` (modify a cell/type/name). The ontology: replace = "find and replace values, supports regex, case-insensitive", update = "modify cell values, column types, or column names in place". "Everywhere" signals bulk replacement. Relabel to `replace`.
- **Feedback:** relabel to `replace`

### D10. dana_085 t1 — `relabel` (swapped to switch_flow)
- **Category:** ambiguous_first
- **Utterance:** "Do we have bounce rate in the email campaign data?"
- **Label:** ambiguous [exist, query]
- **Models say:** exist (70x)
- **Recommendation:** `rewrite` — "Do we have X in Y" is a pure existence check. To make it ambiguous: "What's the bounce rate situation in the email campaign data?" — "situation" could mean check if it exists OR pull the actual numbers.
- **Feedback:** Relabel to just `exist`, move to switch_flow. Change the follow-up utterances as well. Second turn should be "Yes, the bounce rate for email campaign is 12.8%". Third turn should be "OK, but what about just the Coding Agents campaign?", which should be relabeled to `query`. This will swap with dana_011

### D11. dana_086 t1 — `rewrite`
- **Category:** ambiguous_first
- **Utterance:** "What's the gross margin formula in our P&L? Want to make sure I've got it right before pulling numbers."
- **Label:** ambiguous [lookup, define]
- **Models say:** lookup (70x)
- **Recommendation:** `rewrite` — "What's the formula" + "make sure I've got it right" = checking an existing definition (lookup). `define` = create/save a new formula. To make it ambiguous: "The gross margin formula in our P&L — is it set up right, or should I redo it?" — "redo it" makes define plausible.
- **Feedback:** Rewrite to: "The gross margin formula in our P&L — is it set up right, or should I redo it?"

### D12. dana_087 t1 — `rewrite`
- **Category:** ambiguous_first
- **Utterance:** "Can you check the columns in the patient outcomes table? Want to know what we're working with across the 3 sites"
- **Label:** ambiguous [describe, datatype]
- **Models say:** describe (66x), [describe, segment] (4x)
- **Recommendation:** `rewrite` — "Check the columns" + "what we're working with" = profiling (describe). `datatype` = validate column types. To make it ambiguous: "Can you check the columns in the patient outcomes table? Some of them might be the wrong type." — adds the type-checking signal.
- **Feedback:** Rewrite to: "Can you check the columns in the patient outcomes table? Some of them might be the wrong type."

### D13. dana_090 t1 — `relabel` (swapped to same_flow)
- **Category:** ambiguous_first
- **Utterance:** "engagement rate should be (likes + comments + shares) / impressions for both instagram and tiktok"
- **Label:** ambiguous [define, query]
- **Models say:** define (70x)
- **Recommendation:** `rewrite` — Stating a formula = saving a definition (define). `query` = run analysis. To make it ambiguous: "What's the engagement rate — (likes + comments + shares) / impressions — for both Instagram and TikTok?" — asking "what is" + providing formula could be define (save it) or query (compute it now).
- **Feedback:** Relabel as define. Move to same_flow. Swap with dana_044

### D14. dana_091 t1 — `rewrite`
- **Category:** ambiguous_first
- **Utterance:** "I need to pull together the feature usage charts and the engagement table so I can share them with the team every Monday"
- **Label:** ambiguous [dashboard, export]
- **Models say:** dashboard (69x)
- **Recommendation:** `rewrite` — "Every Monday" signals recurring reports (dashboard). To make it ambiguous: "I need to pull together the feature usage charts and the engagement table for the team." — without "every Monday", it could be a one-time export or a recurring dashboard.
- **Feedback:** Rewrite to: "I need to pull together the feature usage charts and the engagement table for the team to take a look at."

### D15. dana_092 t1 — `rewrite`
- **Category:** ambiguous_first
- **Utterance:** "Can you give me a rundown of the satisfaction scores in the survey data"
- **Label:** ambiguous [describe, segment]
- **Models say:** describe (71x)
- **Recommendation:** `rewrite` — "Give me a rundown" = profile/describe. `segment` = break down by dimension. To make it ambiguous: "Can you give me a rundown of the satisfaction scores across the different regions?" — "across regions" makes segment plausible.
- **Feedback:** Rewrite to: "Give me a rundown of the satisfaction scores in the survey data across each user group"

### D16. dana_094 t1 — `relabel`
- **Category:** ambiguous_first
- **Utterance:** "Can you break down the attrition over the last 12 months for me"
- **Label:** ambiguous [summarize, trend]
- **Models say:** trend (51x), segment (10x)
- **Recommendation:** `rewrite` — "Over the last 12 months" = time-based analysis = trend. To make it ambiguous: "Can you break down the attrition for me" — without the time qualifier, it could be a written summary or a trend chart.
- **Feedback:** Relabel as [segment, summarize]

### D17. dana_095 t1 — `relabel`
- **Category:** ambiguous_first
- **Utterance:** "so what should I be thinking about next with these 12 segments"
- **Label:** ambiguous [recommend, explain]
- **Models say:** recommend (52x), [chat, recommend] (9x)
- **Recommendation:** `rewrite` — "What should I be thinking about next" = proactive suggestion (recommend). `explain` = explain what Dana did. To make it ambiguous: "So what was the thinking behind those 12 segments, and where do we go from here?" — first half = explain, second half = recommend.
- **Feedback:** Relabel as [recommend, chat]

### D18. dana_096 t1 — `rewrite+relabel`
- **Category:** ambiguous_first
- **Utterance:** "The account_detail column has the BU name and the GL code jammed together. Can you fix that so they're separate fields in the P&L?"
- **Label:** ambiguous [split, merge]
- **Models say:** split (71x)
- **Recommendation:** `rewrite` — "Jammed together... separate fields" is textbook `split`. `merge` is the opposite operation. To make it ambiguous, pick different candidates. [split, merge] can never be simultaneously plausible for the same utterance — they're inverses. Better candidate pair: [split, update] or [split, replace]. Rewrite: "The account_detail column has the BU name and GL code — can you fix it?" — without specifying split vs merge, it's ambiguous what "fix" means.
- **Feedback:** Rewrite as "There should be a column for the weekly rolling average based on the contracted_spend column. Can you add that?" This is then relabeled as [insert, fill]. You will have to write out new turns 2 & 3 for this one.

### D19. dana_097 t1 — `relabel`
- **Category:** ambiguous_first
- **Utterance:** "For patient outcomes, I like dates as dd/mm/yyyy since these hospitals are UK."
- **Label:** ambiguous [preference, recommend]
- **Models say:** preference (71x)
- **Recommendation:** `rewrite` — "I like dates as X" = setting a preference. `recommend` = Dana suggests next steps. These aren't even close. To make it ambiguous: "For patient outcomes, what's the best date format? These are UK hospitals." — asking for advice (recommend) vs stating a preference.
- **Feedback:** Relabel as [preference, format]

### D20. dana_098 t1 — `skip`
- **Category:** ambiguous_first
- **Utterance:** "hey the signup_date column is all strings, needs to be dates"
- **Label:** ambiguous [datatype, update]
- **Models say:** datatype (65x), update (4x)
- **Recommendation:** `rewrite` — "All strings, needs to be dates" = type casting (datatype). There's a small model split (65 vs 4) but it's overwhelmingly datatype. To make it more balanced: "hey the signup_date column looks off, can you fix it" — could be datatype issue or a value correction (update).
- **Feedback:** The models were mixed because this is ambiguous. Skip.

### D21. dana_099 t1 — `skip`
- **Category:** ambiguous_first
- **Utterance:** "I need to combine the inventory data with the shipping records for all 8 warehouses"
- **Label:** ambiguous [join, merge]
- **Models say:** join (69x)
- **Recommendation:** `reconsider` or `rewrite` — Same issue as D6. `join` = combine tables, `merge` = combine columns. "Combine the inventory data with the shipping records" = two separate tables = join. The join/merge boundary is well-defined in the ontology, and this utterance is clearly on the join side. To make it ambiguous: "I need to combine inventory and shipping into one view for the 8 warehouses" — "one view" could mean join tables or merge certain columns.
- **Feedback:** Skip. The 'inventory data' and 'shipping records' could be columns as well.

### D22. dana_100 t1 — `skip`
- **Category:** ambiguous_first
- **Utterance:** "The platform column has a bunch of entries that say IG instead of Instagram. Can you fix those?"
- **Label:** ambiguous [replace, validate]
- **Models say:** replace (69x), [format, replace] (2x)
- **Recommendation:** `rewrite` — "IG instead of Instagram, fix those" = find and replace. `validate` = check against valid options, flag violations. To make it ambiguous: "The platform column has some inconsistent values. Can you fix it?" — without specifying the exact fix, it could be validate (check what's wrong) or replace (swap values).
- **Feedback:** Skip. The models were mixed because this is ambiguous.

### D23. dana_101 t1 — `relabel`
- **Category:** ambiguous_first
- **Utterance:** "Can you show me how feature adoption looks across our user segments"
- **Label:** ambiguous [plot, trend]
- **Models say:** [compare, plot] (20x), segment (19x)
- **Recommendation:** `rewrite` — Models are split between plot/compare/segment but nobody says trend. "Across user segments" = cross-sectional (not time-based), ruling out trend. To make [plot, trend] ambiguous: "Can you show me how feature adoption has been going?" — "has been going" introduces the time element for trend.
- **Feedback:** Relabel as [plot, compare]

### D24. dana_104 t1 — `relabel`
- **Category:** ambiguous_first
- **Utterance:** "can you show me what's going on with attrition by department"
- **Label:** ambiguous [summarize, plot]
- **Models say:** segment (20x), plot (19x)
- **Recommendation:** `rewrite` — Models split between segment and plot, but neither picks summarize as a strong candidate. "Show me" biases toward visual output (plot) or drilldown (segment). To make [summarize, plot] ambiguous: "can you tell me what's going on with attrition by department" — "tell me" makes the written summary (summarize) more plausible.
- **Feedback:**  Relabel as [segment, plot]

## Dana — ambiguous_second (2 labels)

### D25. dana_028 t3 — `skip`
- **Category:** ambiguous_second
- **Utterance:** "Give me a chart with the DAUs per quarter for all valid subscription types, meaning the active ones."
- **Label:** blueprint, candidates=[validate, plot]
- **Models say:** trend (30x), [trend, validate] (13x)
- **Recommendation:** `relabel` — "Chart with DAUs per quarter" = time-series visualization = `trend` (not plot — plot is a generic chart, trend has time as an axis). "Valid subscription types, meaning active ones" adds a filter, not a validate step. Models correctly identify trend. Relabel candidates to [trend, validate] or just relabel to `trend` as the primary flow. The `blueprint` parent label should probably be `outline` (multi-step plan).
- **Feedback:** Plot is correct here. The user is asking for a chart, not a trend line. Plots can also include time-series data. Skip

### D26. dana_117 t3 — `relabel`
- **Category:** ambiguous_second
- **Utterance:** "Flag the bad region values. How does monthly revenue look by region?"
- **Label:** blueprint, candidates=[validate, summarize]
- **Models say:** [trend, validate] (50x)
- **Recommendation:** `relabel` — "Flag bad region values" = validate. "Monthly revenue by region" = trend (time-series by dimension). Models correctly say [trend, validate]. The `summarize` candidate is wrong — "how does it look" with "monthly" is a visual/trend question, not a written summary. Relabel candidates to [validate, trend].
- **Feedback:** I have reconsidered how to define 'trend' so that the distinction between 'trend' and 'plot' is more clear. This example should be relabeled as [validate, summarize, plot].

## Dana — same_flow (2 labels)

### D27. dana_005 t1 — `rewrite`
- **Category:** same_flow
- **Utterance:** "Can you expand 'ER' to 'Emergency' so it's more readable?"
- **Label:** update
- **Models say:** replace (64x), [format, replace] (5x)
- **Recommendation:** `relabel` — "Expand ER to Emergency" = find "ER" and replace with "Emergency" = `replace`. The ontology: update = modify a cell/type/name, replace = find and replace values across a column. This is a bulk substitution, not a single-cell edit. Relabel to `replace`. Also need to check t3 ("lowercase those too") — that should probably stay as `update` (changing column name casing) or could be `format`.
- **Feedback:** Rewrite to "Can you change 'ER' to 'Emergency' in the headers so it's more readable." — this makes it clear that the user wants to update the values in just the header columns, not replace values across all cells.

### D28. dana_044 t3 — `relabel` (swapped to ambiguous_first)
- **Category:** same_flow
- **Utterance:** "Now pull the same but grouped by age bracket"
- **Label:** query
- **Models say:** segment (61x), [query, segment] (8x)
- **Recommendation:** `relabel` — "Grouped by age bracket" = break down a metric by a dimension = `segment`. The ontology: segment = "break down a single metric by one dimension for drilldown", query = general-purpose SQL-like analysis. This is a textbook segment operation. Relabel to `segment`. Note: this changes the conversation from same_flow to switch_flow since t1 is query and t3 would be segment.
- **Feedback:** Relabel as ambiguous with [query, segment]. Swap with dana_090

## Dana — switch_flow (2 labels)

### D29. dana_011 t1 — `relabel` (swapped to ambiguous_first)
- **Category:** switch_flow
- **Utterance:** "How's CTR doing lately?"
- **Label:** lookup
- **Models say:** trend (71x)
- **Recommendation:** `relabel` — "How's X doing lately?" = time-based analysis = `trend`. `lookup` = find a metric definition in the semantic layer. The user isn't asking for a definition; they want to see recent CTR values/trajectory. Relabel to `trend`.
- **Feedback:** Relabel to ambiguous [trend, lookup]. The metric is 'Click-Through Rate', which is a well-defined metric in the semantic layer. The user could be asking about the recent trajectory (trend) OR the definition (lookup). Move this into ambiguous_first. This will swap with dana_085

### D30. dana_012 t3 — `rewrite`
- **Category:** switch_flow
- **Utterance:** "What's the percent change?"
- **Label:** insert
- **Models say:** query (53x), trend (7x)
- **Recommendation:** `relabel` — "What's the percent change?" asks for a computed value from existing data = `query`. `insert` = add a new row/column. The user isn't asking to add a column; they want to see the calculation. Relabel to `query`. If the intent was "add a percent-change column", the utterance should say "Add a percent change column" explicitly.
- **Feedback:** Rewrite to "What's the percent change per state? I want to see it next to the expenses." — this makes it clear that the user wants to see the calculation in a new column (insert).
