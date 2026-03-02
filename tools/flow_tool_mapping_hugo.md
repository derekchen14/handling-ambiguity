# Hugo — Flow → Tool Mapping

All flows also use component tools (+3): `coordinate_context`, `manage_memory`, `read_flow_stack`

---

## Tool Design Ladders

### Read granularity ladder

```
get_post         → metadata only (title, status, tags, section titles, word counts)
read_outline     → structural overview (headings, key points per section)
heads_or_tails   → peek at first/last N paragraphs
read_post        → full content loaded into LLM context
```

### Write quality ladder

```
draft_content    → rough, fast, exploratory (less LLM thinking)
write_section    → context-aware, reads surrounding sections for coherence
generate_prose   → polished, careful, standalone writing from a brief
```

---

## Research (6 flows)

| Flow | Tools | Domain Tools | Selection logic |
|------|-------|-------------|-----------------|
| browse | 8 | `browse_topics` · `search_posts` · `suggest_keywords` · `search_by_time` · `search_inspiration` | browse_topics for trending/saved/gaps; search_posts for keyword match against existing posts; suggest_keywords for SEO-informed discovery; search_by_time for recent activity; search_inspiration for external content landscape |
| view | 4 | `read_post` | Always 1 tool — full content load into context |
| check | 6 | `search_posts` · `check_platform` · `search_by_time` | search_posts for local workflow status; check_platform for publication status on external platforms; search_by_time for date-filtered status checks |
| inspect | 9 | `get_post` · `analyze_content` · `check_readability` · `analyze_seo` · `check_links` · `heads_or_tails` | get_post for metadata; analyze_content for structure/completeness; check_readability for grade-level metrics; analyze_seo for keyword/meta checks; check_links for broken URLs; heads_or_tails for quick content peek |
| find | 5 | `search_posts` · `search_by_time` | search_posts for keyword search; search_by_time for date-range search |
| compare | 5 | `compare_posts` · `read_post` | read_post to load each post; compare_posts for style/structure analysis |

## Draft (7 flows)

| Flow | Tools | Domain Tools | Selection logic |
|------|-------|-------------|-----------------|
| outline | 6 | `generate_outline` · `suggest_keywords` · `search_inspiration` | generate_outline for section structure; suggest_keywords for SEO-aware headings; search_inspiration for angle research |
| refine | 7 | `read_post` · `read_outline` · `revise_content` · `update_section` | read_outline for current structure; read_post for section content; revise_content to regenerate with feedback; update_section to save |
| expand | 7 | `read_post` · `expand_content` · `update_section` · `insert_media` | read_post to load sparse content; expand_content to develop into prose; update_section to save; insert_media if expansion includes images |
| write | 10 | `generate_prose` · `draft_content` · `write_section` · `update_section` · `insert_media` · `search_sources` · `search_evidence` | draft_content for rough first pass; generate_prose for polished writing; write_section for context-aware section writing; update_section to save; search_sources/search_evidence for references and data |
| add | 4 | `insert_section` | Always 1 tool — structural section insertion |
| create | 4 | `create_post` | Always 1 tool — initializes new draft |
| brainstorm | 5 | `brainstorm_ideas` · `search_inspiration` | brainstorm_ideas for creative angles; search_inspiration for what others have written |

## Revise (8 flows)

| Flow | Tools | Domain Tools | Selection logic |
|------|-------|-------------|-----------------|
| rework | 9 | `read_post` · `read_outline` · `revise_content` · `write_section` · `update_section` · `delete_section` · `reorder_sections` | read_post + read_outline to understand current state; revise_content for heavy rewrite; write_section for new sections; delete_section to remove weak parts; reorder_sections to restructure; update_section to save |
| polish | 8 | `read_post` · `revise_content` · `update_section` · `check_grammar` · `detect_issues` | read_post to load content; check_grammar + detect_issues to find problems; revise_content for light editing; update_section to save |
| tone | 6 | `read_post` · `adjust_tone` · `update_section` | read_post to load content; adjust_tone for voice shift; update_section to save |
| audit | 5 | `read_post` · `audit_style` | read_post to load current draft; audit_style to compare against published history |
| format | 8 | `get_post` · `format_content` · `update_metadata` · `analyze_seo` · `insert_media` | get_post for metadata/structure; format_content for platform conventions; update_metadata for tags/excerpt/status; analyze_seo for pre-publish optimization; insert_media for featured image |
| amend | 6 | `read_post` · `revise_content` · `update_section` | read_post to load content; revise_content with user feedback as instructions; update_section to save |
| diff | 4 | `diff_versions` | Always 1 tool — side-by-side version comparison |
| tidy | 9 | `get_post` · `normalize_structure` · `update_section` · `update_metadata` · `check_grammar` · `check_links` | get_post for metadata; normalize_structure for heading/spacing consistency; check_grammar for mechanical cleanup; check_links for URL validation; update_section + update_metadata to save |

## Publish (7 flows)

| Flow | Tools | Domain Tools | Selection logic |
|------|-------|-------------|-----------------|
| release | 5 | `get_post` · `publish_post` | get_post to verify readiness; publish_post to make live |
| syndicate | 6 | `get_post` · `publish_post` · `format_content` | get_post to verify metadata; format_content to adapt for target platform; publish_post to cross-post |
| schedule | 4 | `manage_schedule` | Always 1 tool — sets future publication datetime |
| preview | 5 | `get_post` · `render_preview` | get_post to load metadata; render_preview for platform-specific appearance |
| confirm | — | *(component tools only)* | No domain tool — final yes/no gate before release or syndicate |
| cancel | 4 | `manage_schedule` | Always 1 tool (action=cancel) — removes scheduled publication |
| survey | 5 | `list_platforms` · `check_platform` | list_platforms for all configured platforms; check_platform for per-post publication status |

## Converse (7 flows)

| Flow | Tools | Domain Tools | Selection logic |
|------|-------|-------------|-----------------|
| explain | 4 | `explain_action` | Always 1 tool — LLM-powered transparency |
| chat | 3 | *(component tools only)* | No domain tool — pure conversation |
| preference | 3 | *(manage_memory L2 write)* | Component tool only |
| suggest | 5 | `brainstorm_ideas` · `search_posts` | brainstorm_ideas for creative next steps; search_posts for context-aware suggestions based on existing content |
| undo | 4 | `rollback_post` | Always 1 tool — reverts to previous version |
| endorse | — | *(routes to target flow's tools)* | Triggers the tool(s) from the endorsed suggestion |
| dismiss | 3 | *(component tools only)* | No domain tool — decline and note preference |

## Plan (6 flows)

| Flow | Tools | Domain Tools | Selection logic |
|------|-------|-------------|-----------------|
| blueprint | — | *(orchestrates Research + Draft + Revise + Publish tools)* | No unique tool — plans a full post workflow from idea to publication |
| triage | 8 | `analyze_content` · `check_readability` · `detect_issues` · `heads_or_tails` · `read_outline` | read_outline for structural overview; heads_or_tails for content peek; analyze_content for structural gaps; check_readability for grade-level issues; detect_issues for quality problems; produces prioritized revision checklist |
| calendar | 5 | `plan_calendar` · `search_by_time` | plan_calendar for schedule generation; search_by_time to check what's already scheduled and avoid conflicts |
| scope | 8 | `search_posts` · `browse_topics` · `search_sources` · `search_inspiration` · `read_outline` | search_posts for what's already written; browse_topics for gaps; read_outline for structure of existing posts; search_sources + search_inspiration for external research planning |
| digest | 4 | `plan_series` | Always 1 tool — multi-part series planning |
| remember | — | *(orchestrates Internal memory flows)* | No unique tool — routes to recap, store, recall, or retrieve |

## Internal (7 flows)

Internal flows use 2 component tools (`coordinate_context`, `manage_memory`) — no `read_flow_stack`

| Flow | Tools | Domain Tools | Selection logic |
|------|-------|-------------|-----------------|
| recap | 2 | *(manage_memory L1 read)* | Component tool only — session scratchpad read |
| store | 2 | *(manage_memory L1 write)* | Component tool only — session scratchpad write |
| recall | 2 | *(manage_memory L2 read)* | Component tool only — persistent preferences |
| retrieve | 2 | *(manage_memory L3 read)* | Component tool only — unvetted business docs and domain knowledge |
| search | 3 | `search_reference` | Vetted FAQs and curated editorial guidelines |
| reference | 3 | `lookup_word` | Dictionary definitions, synonyms, antonyms, usage examples |
| study | 3 | `read_post` | Internally loads a previous post into agent context to match voice and patterns |

---

## Summary

| Category | Tools | Count |
|----------|-------|-------|
| Post CRUD | search_posts, get_post, create_post, delete_post, rollback_post | 5 |
| Post Read | read_post, read_outline, heads_or_tails, search_by_time | 4 |
| Section | update_section, insert_section, delete_section, reorder_sections | 4 |
| Metadata | update_metadata | 1 |
| Generation | generate_outline, generate_prose, expand_content, revise_content, brainstorm_ideas, explain_action, write_section, draft_content | 8 |
| Tone & Style | adjust_tone, audit_style, compare_posts | 3 |
| Writing Quality | check_grammar, check_readability, detect_issues, analyze_content | 4 |
| Reference | lookup_word, search_reference | 2 |
| SEO | analyze_seo, suggest_keywords | 2 |
| Web Research | search_sources, search_evidence, search_inspiration | 3 |
| Formatting | format_content, normalize_structure | 2 |
| Diff | diff_versions | 1 |
| Media | insert_media, check_links | 2 |
| Topic | browse_topics | 1 |
| Platform | publish_post, check_platform, list_platforms, render_preview | 4 |
| Scheduling | manage_schedule | 1 |
| Planning | plan_calendar, plan_series | 2 |
| Component | coordinate_context, manage_memory, read_flow_stack | 3 |
| **Total** | | **52** |

### Coverage

| Intent | Flows | With domain tools | Component-only | Orchestrators |
|--------|-------|--------------------|----------------|---------------|
| Research | 6 | 6 | 0 | 0 |
| Draft | 7 | 7 | 0 | 0 |
| Revise | 8 | 8 | 0 | 0 |
| Publish | 7 | 6 | 1 | 0 |
| Converse | 7 | 3 | 4 | 0 |
| Plan | 6 | 4 | 0 | 2 |
| Internal | 7 | 3 | 4 | 0 |
| **Total** | **48** | **37** | **9** | **2** |

### Parameter counts (max 5 enforced)

| Params | Tools |
|--------|-------|
| 0 | list_platforms |
| 1 | delete_post, check_links, read_outline |
| 2 | create_post, rollback_post, get_post, read_post, delete_section, reorder_sections, explain_action, compare_posts, check_grammar, check_readability, detect_issues, analyze_content, lookup_word, search_reference, analyze_seo, suggest_keywords, normalize_structure, check_platform, render_preview, coordinate_context, read_flow_stack |
| 3 | update_section, insert_section, generate_outline, expand_content, brainstorm_ideas, heads_or_tails, draft_content, search_sources, search_evidence, search_inspiration, adjust_tone, audit_style, format_content, browse_topics, publish_post |
| 4 | search_posts, search_by_time, generate_prose, revise_content, write_section, diff_versions, manage_schedule, plan_calendar, plan_series, manage_memory |
| 5 | update_metadata, insert_media |
