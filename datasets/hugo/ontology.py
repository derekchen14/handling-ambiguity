from enum import Enum


class Intent:
    PLAN = 'Plan'
    CONVERSE = 'Converse'
    INTERNAL = 'Internal'
    RESEARCH = 'Research'     # browse, summarize, check, inspect, find, compare, diff
    DRAFT = 'Draft'           # outline, refine, cite, compose, add, create, brainstorm
    REVISE = 'Revise'         # rework, polish, tone, audit, simplify, remove, tidy
    PUBLISH = 'Publish'       # release, syndicate, schedule, preview, confirm, cancel, survey


class FlowLifecycle(str, Enum):
    PENDING = 'Pending'
    ACTIVE = 'Active'
    COMPLETED = 'Completed'
    INVALID = 'Invalid'


class AmbiguityLevel(str, Enum):
    GENERAL = 'general'
    PARTIAL = 'partial'
    SPECIFIC = 'specific'
    CONFIRMATION = 'confirmation'


DACT_CATALOG = {
    'chat':     {'hex': '0', 'pos': 'noun'},
    'search':   {'hex': '1', 'pos': 'verb'},
    'outline':  {'hex': '2', 'pos': 'verb'},
    'compose':  {'hex': '3', 'pos': 'verb'},
    'share':    {'hex': '4', 'pos': 'verb'},
    'insert':   {'hex': '5', 'pos': 'verb'},
    'update':   {'hex': '6', 'pos': 'verb'},
    'delete':   {'hex': '7', 'pos': 'verb'},
    'user':     {'hex': '8', 'pos': 'adj'},
    'agent':    {'hex': '9', 'pos': 'adj'},
    'post':     {'hex': 'A', 'pos': 'noun'},
    'section':  {'hex': 'B', 'pos': 'noun'},
    'channel':  {'hex': 'C', 'pos': 'noun'},
    'rough':    {'hex': 'D', 'pos': 'adj'},
    'approve':  {'hex': 'E', 'pos': 'adj'},
    'reject':   {'hex': 'F', 'pos': 'adj'},
}



FLOW_CATALOG = {

    # ── Research (7 flows) ─────────────────────────────────

    'browse': {
        'dax': '{012}',
        'intent': Intent.RESEARCH,
        'description': 'Browse available topics, notes, trending subjects, saved ideas, and content gaps filtered by category. Excludes drafts and posts which use the "find" flow instead.',
        'output': 'list',
        'edge_flows': ['find', 'brainstorm', 'search', 'audit'],
        'policy_path': 'policies.research.browse',
    },
    'summarize': {
        'dax': '{19A}',
        'intent': Intent.RESEARCH,
        'description': 'Synthesize a post into a short paragraph capturing the core argument, target audience, and main takeaways — useful for excerpts, SEO descriptions, or pre-reads before writing a follow-up',
        'output': 'card',
        'edge_flows': ['find', 'check', 'preview'],
        'policy_path': 'policies.research.summarize',
    },
    'check': {
        'dax': '{0AD}',
        'intent': Intent.RESEARCH,
        'description': 'Check the technical metadata surrounding a post — category tags, has_featured_image, publication date, last edited date, scheduled date, channels, status: draft, scheduled, published, or unpublished',
        'output': 'list',
        'edge_flows': ['summarize', 'inspect', 'preview', 'compare'],
        'policy_path': 'policies.research.check',
    },
    'inspect': {
        'dax': '{1BD}',
        'intent': Intent.RESEARCH,
        'description': 'Report numeric content metrics — word count, section count, reading time, image count, post size (MB); optionally filtered to a single metric (word_count, section_count, time_to_read, image_count, post_size). Use check for post metadata',
        'output': 'card',
        'edge_flows': ['check', 'tidy', 'audit'],
        'policy_path': 'policies.research.inspect',
    },
    'find': {
        'dax': '{001}',
        'intent': Intent.RESEARCH,
        'description': 'Search previous posts by keyword or topic — returns matching titles, excerpts, and publication dates sorted by relevance',
        'output': 'list',
        'edge_flows': ['browse', 'audit', 'summarize'],
        'policy_path': 'policies.research.find',
    },
    'compare': {
        'dax': '{18A}',
        'intent': Intent.RESEARCH,
        'description': 'Compare style or structure across two or more posts — sentence length, paragraph density, heading patterns, vocabulary, and tonal consistency',
        'output': 'compare',
        'edge_flows': ['find', 'audit'],
        'policy_path': 'policies.research.compare',
    },
    'diff': {
        'dax': '{0BD}',
        'intent': Intent.RESEARCH,
        'description': 'Compare two versions of a section side by side — shows additions, deletions, and modifications highlighted so the user can evaluate what changed',
        'output': 'card',
        'edge_flows': ['polish', 'amend'],
        'policy_path': 'policies.research.diff',
    },

    # ── Draft (7 flows) ─────────────────────────────────────────────

    'outline': {
        'dax': '{002}',
        'intent': Intent.DRAFT,
        'description': 'Generate an outline including section headings, key bullet points, estimated word counts, and suggested reading order',
        'output': 'list',
        'edge_flows': ['refine', 'brainstorm', 'write'],
        'policy_path': 'policies.draft.outline',
    },
    'refine': {
        'dax': '{02B}',
        'intent': Intent.DRAFT,
        'description': 'Refine the bullet points in the outline; which can include adjust headings, reorder points, add or remove subsections, and incorporate feedback',
        'output': 'card',
        'edge_flows': ['outline', 'compose', 'polish'],
        'policy_path': 'policies.draft.refine',
    },
    'cite': {
        'dax': '{15B}',
        'intent': Intent.DRAFT,
        'description': 'Add a citation to a note — if a URL is provided, attach it directly; if only a note is provided, search the web for a supporting source and propose it for user confirmation',
        'output': 'card',
        'edge_flows': ['brainstorm', 'rework'],
        'policy_path': 'policies.draft.cite',
    },
    'compose': {
        'dax': '{003}',
        'intent': Intent.DRAFT,
        'description': 'Write a section from scratch based on instructions or an outline. If only given a topic, generate an outline first; For editing existing content, use rework',
        'output': 'card',
        'edge_flows': ['rework', 'polish'],
        'policy_path': 'policies.draft.compose',
    },
    'add': {
        'dax': '{005}',
        'intent': Intent.DRAFT,
        'description': 'Add a new section to the post — creates an empty section placeholder with a heading, inserted at a specific position in the post structure',
        'output': 'toast',
        'edge_flows': ['write', 'refine'],
        'policy_path': 'policies.draft.add',
    },
    'create': {
        'dax': '{05A}',
        'intent': Intent.DRAFT,
        'description': 'Start a new post from scratch — initializes a post record with title, topic, and empty sections. Does not generate content; use outline or write to fill sections',
        'output': 'form',
        'panel': 'top',
        'edge_flows': ['outline', 'blueprint', 'add'],
        'policy_path': 'policies.draft.create',
    },
    'brainstorm': {
        'dax': '{29A}',
        'intent': Intent.DRAFT,
        'description': 'Come up with new ideas or angles for a given topic, word, or phrase. This may include hooks, opening lines, synonyms, or new perspectives the user can choose from',
        'output': 'list',
        'edge_flows': ['browse', 'outline'],
        'policy_path': 'policies.draft.brainstorm',
    },

    # ── Revise (7 flows) ─────────────────────────────

    'rework': {
        'dax': '{006}',
        'intent': Intent.REVISE,
        'description': 'Major revision of draft content — restructures arguments, replaces weak sections, addresses reviewer comments. Scope can go across the whole post, or an entire section. For smaller changes, use polish',
        'output': 'card',
        'edge_flows': ['compose', 'polish', 'simplify', 'refine'],
        'policy_path': 'policies.revise.rework',
    },
    'polish': {
        'dax': '{3BD}',
        'intent': Intent.REVISE,
        'description': 'Editing of a specific paragraph, sentence or phrase — improves word choice, tightens sentences, fixes transitions, and smooths flow without changing meaning or structure. The scope is within a single paragraph or image, not across the whole post',
        'output': 'card',
        'edge_flows': ['rework', 'write', 'refine'],
        'policy_path': 'policies.revise.polish',
    },
    'tone': {
        'dax': '{38A}',
        'intent': Intent.REVISE,
        'description': 'Adjust tone or voice across the entire post — shifts register (formal, casual, technical, academic, witty, natural), adjusts sentence length and vocabulary complexity',
        'output': 'card',
        'edge_flows': ['rework', 'audit'],
        'policy_path': 'policies.revise.tone',
    },
    'audit': {
        'dax': '{13A}',
        'intent': Intent.REVISE,
        'description': "Check that the post is written in the user's voice rather than sounding like AI, compares voice, terminology, formatting conventions, and stylistic patterns against previous posts",
        'output': 'card',
        'edge_flows': ['tone', 'compare'],
        'policy_path': 'policies.revise.audit',
    },
    'simplify': {
        'dax': '{7BD}',
        'intent': Intent.REVISE,
        'description': 'Reduce complexity of a section or note — shorten paragraphs, simplify sentence structure, remove redundancy; image simplification means replacing with a simpler alternative or removing entirely',
        'output': 'card',
        'edge_flows': ['polish', 'remove', 'tidy'],
        'policy_path': 'policies.revise.simplify',
    },
    'remove': {
        'dax': '{007}',
        'intent': Intent.REVISE,
        'description': 'Remove a section from the post, delete a draft or note',
        'output': 'toast',
        'edge_flows': ['audit', 'rework'],
        'policy_path': 'policies.revise.amend',
    },
    'tidy': {
        'dax': '{3AB}',
        'intent': Intent.REVISE,
        'description': 'Normalize structural formatting across the post — consistent heading hierarchy, list indentation, paragraph spacing, and whitespace cleanup. Does not change wording',
        'output': 'card',
        'edge_flows': ['simplify', 'rework'],
        'policy_path': 'policies.revise.tidy',
    },

    # ── Publish (7 flows) ──────────────────────────────────────────

    'release': {
        'dax': '{04A}',
        'intent': Intent.PUBLISH,
        'description': 'Publish the post to the primary blog; makes the post live immediately on the main channel. Use syndicate to cross-post, promote to amplify reach after publishing',
        'output': 'toast',
        'edge_flows': ['syndicate', 'promote'],
        'policy_path': 'policies.publish.release',
    },
    'syndicate': {
        'dax': '{04C}',
        'intent': Intent.PUBLISH,
        'description': 'Cross-post to a secondary channel — adapts formatting for the target (Medium, Dev.to, LinkedIn, Substack) and publishes a tailored version',
        'output': 'toast',
        'edge_flows': ['release', 'schedule'],
        'policy_path': 'policies.publish.syndicate',
    },
    'schedule': {
        'dax': '{4AC}',
        'intent': Intent.PUBLISH,
        'description': 'Schedule a post for future publication — sets a specific date and time for automatic publishing on a given channel',
        'output': 'toast',
        'edge_flows': ['release', 'syndicate'],
        'policy_path': 'policies.publish.schedule',
    },
    'preview': {
        'dax': '{4AD}',
        'intent': Intent.PUBLISH,
        'description': 'Preview how the post will look when published — renders the post in the target channel\'s format so the user can review layout, images, and formatting before going live',
        'output': 'card',
        'edge_flows': ['tidy', 'release', 'summarize', 'compare'],
        'policy_path': 'policies.publish.preview',
    },
    'promote': {
        'dax': '{004}',
        'intent': Intent.PUBLISH,
        'description': 'Make a published post more prominent — pin to the top of the blog, mark as featured, announce to subscribers, or share to social channels and email lists. Amplifies reach after release or syndicate',
        'output': 'toast',
        'edge_flows': ['release', 'syndicate'],
        'policy_path': 'policies.publish.promote',
    },
    'cancel': {
        'dax': '{04F}',
        'intent': Intent.PUBLISH,
        'description': 'Cancel a scheduled publication or unpublish a live post — reverts to draft status or removes from the channel entirely',
        'output': 'confirmation',
        'edge_flows': ['release', 'schedule'],
        'policy_path': 'policies.publish.cancel',
    },
    'survey': {
        'dax': '{01C}',
        'intent': Intent.PUBLISH,
        'description': 'View configured publishing channels and their health — lists connected channels (WordPress, Medium, etc.), API status, last sync date, and credential validity',
        'output': 'list',
        'edge_flows': ['syndicate', 'schedule', 'find'],
        'policy_path': 'policies.publish.survey',
    },

    # ── Converse (7 flows) ──────────────────────────────────────────

    'explain': {
        'dax': '{009}',
        'intent': Intent.CONVERSE,
        'description': 'Hugo explains what it did or plans to do — transparency into the writing process and recent actions',
        'output': 'card',
        'edge_flows': ['chat', 'suggest'],
        'policy_path': 'policies.converse.explain',
    },
    'chat': {
        'dax': '{000}',
        'intent': Intent.CONVERSE,
        'description': 'Open-ended conversation — general Q&A about writing craft, blogging strategy, SEO, audience engagement, or any topic not tied to a specific post action',
        'output': 'card',
        'edge_flows': ['suggest', 'explain'],
        'policy_path': 'policies.converse.chat',
    },
    'preference': {
        'dax': '{08A}',
        'intent': Intent.CONVERSE,
        'description': 'Set a persistent writing preference stored in Memory Manager (L2) — preferred tone, default post length, heading style, Oxford comma usage, or channel defaults',
        'output': 'toast',
        'edge_flows': ['chat', 'tone'],
        'policy_path': 'policies.converse.preference',
    },
    'suggest': {
        'dax': '{29B}',
        'intent': Intent.CONVERSE,
        'description': 'Hugo proactively suggests a next step based on current context — what to write next, which section needs attention, a new angle to explore, or an improvement to try',
        'output': 'card',
        'edge_flows': ['chat', 'explain', 'brainstorm'],
        'policy_path': 'policies.converse.suggest',
    },
    'undo': {
        'dax': '{08F}',
        'intent': Intent.CONVERSE,
        'description': 'Reverse the most recent writing action — rolls back the last edit, addition, deletion, or formatting change and restores the previous version of the affected section',
        'output': 'toast',
        'edge_flows': ['chat', 'amend'],
        'policy_path': 'policies.converse.undo',
    },
    'endorse': {
        'dax': '{09E}',
        'intent': Intent.CONVERSE,
        'description': "Accept Hugo's proactive suggestion and trigger the corresponding action — e.g., a recommended edit, topic idea, or next step that Hugo offered via suggest",
        'output': 'toast',
        'edge_flows': ['amend', 'suggest'],
        'policy_path': 'policies.converse.endorse',
    },
    'dismiss': {
        'dax': '{09F}',
        'intent': Intent.CONVERSE,
        'description': "Decline Hugo's proactive suggestion without providing feedback — Hugo notes the preference and moves on without further prompting",
        'output': 'toast',
        'edge_flows': ['amend', 'chat', 'cancel'],
        'policy_path': 'policies.converse.dismiss',
    },

    # ── Plan (6 flows) ──────────────────────────────────────────────

    'blueprint': {
        'dax': '{25A}',
        'intent': Intent.PLAN,
        'description': 'Plan the full post creation workflow from idea to publication — orchestrates Research, Draft, Revise, and Publish flows into a sequenced checklist with dependencies',
        'output': 'list',
        'edge_flows': ['create', 'outline'],
        'policy_path': 'policies.plan.blueprint',
    },
    'triage': {
        'dax': '{23A}',
        'intent': Intent.PLAN,
        'description': 'Plan a revision sequence — examines a draft and prioritizes which sections need rework, polish, or restructuring; produces an ordered checklist of revision tasks',
        'output': 'list',
        'edge_flows': ['rework', 'tidy'],
        'policy_path': 'policies.plan.triage',
    },
    'calendar': {
        'dax': '{24A}',
        'intent': Intent.PLAN,
        'description': 'Plan a content calendar — lays out a publishing schedule over weeks or months: which topics to draft, target publish dates, and how to space content for consistency',
        'output': 'list',
        'edge_flows': ['schedule', 'browse'],
        'policy_path': 'policies.plan.calendar',
    },
    'scope': {
        'dax': '{12A}',
        'intent': Intent.PLAN,
        'description': 'Plan topic research before writing — defines what information to gather, which previous posts to reference, and what questions to answer before drafting begins',
        'output': 'list',
        'edge_flows': ['browse', 'find'],
        'policy_path': 'policies.plan.scope',
    },
    'digest': {
        'dax': '{25B}',
        'intent': Intent.PLAN,
        'description': 'Plan a multi-part blog series — splits a broad theme into installments, defines the narrative arc, assigns subtopics to each part, and sets a suggested publication sequence',
        'output': 'list',
        'edge_flows': ['blueprint', 'create'],
        'policy_path': 'policies.plan.digest',
    },
    'remember': {
        'dax': '{19B}',
        'intent': Intent.PLAN,
        'description': 'Plan a memory operation — determines whether information should be stored (L1 scratchpad), saved as a preference (L2), or retrieved from business context (L3), then orchestrates the appropriate internal flows',
        'output': 'list',
        'edge_flows': ['recall', 'retrieve'],
        'policy_path': 'policies.plan.remember',
    },

    # ── Internal (7 flows: -tidy) ───────────────────────────────────

    'recap': {
        'dax': '{018}',
        'intent': Intent.INTERNAL,
        'description': 'Read back a previously noted fact from the current session scratchpad (L1) — a decision, constraint, topic preference, or reference the agent stored earlier via store',
        'output': '(internal)',
        'edge_flows': ['store', 'recall'],
        'policy_path': 'policies.internal.recap',
    },
    'store': {
        'dax': '{058}',
        'intent': Intent.INTERNAL,
        'description': 'Save a key-value pair to the session scratchpad (L1) for later use in the same session — topic preferences, user corrections, interim decisions, or reference snippets',
        'output': '(internal)',
        'edge_flows': ['recap', 'recall'],
        'policy_path': 'policies.internal.store',
    },
    'recall': {
        'dax': '{289}',
        'intent': Intent.INTERNAL,
        'description': 'Look up persistent user preferences from Memory Manager (L2) — default tone, word count targets, stylistic rules, or channel credentials set via the preference flow',
        'output': '(internal)',
        'edge_flows': ['recap', 'retrieve'],
        'policy_path': 'policies.internal.recall',
    },
    'retrieve': {
        'dax': '{049}',
        'intent': Intent.INTERNAL,
        'description': 'Fetch general business context from Memory Manager — unvetted documents, style guides, or domain knowledge (L3)',
        'output': '(internal)',
        'edge_flows': ['recall', 'search'],
        'policy_path': 'policies.internal.retrieve',
    },
    'search': {
        'dax': '{189}',
        'intent': Intent.INTERNAL,
        'description': 'Look up vetted FAQs and curated editorial guidelines — the unstructured equivalent of a style manual',
        'output': '(internal)',
        'edge_flows': ['retrieve', 'recap'],
        'policy_path': 'policies.internal.search',
    },
    'reference': {
        'dax': '{139}',
        'intent': Intent.INTERNAL,
        'description': 'Look up word definitions, synonyms, antonyms, or usage examples via dictionary and thesaurus — e.g., "synonym for important", "definition of ephemeral", "formal alternatives to good"',
        'output': '(internal)',
        'edge_flows': ['study', 'recap'],
        'policy_path': 'policies.internal.reference',
    },
    'study': {
        'dax': '{1AC}',
        'intent': Intent.INTERNAL,
        'description': 'Internally load a previous post into agent context without showing it to the user — used to match voice, structure, or vocabulary patterns when writing new content',
        'output': '(internal)',
        'edge_flows': ['find', 'reference'],
        'policy_path': 'policies.internal.study',
    },
}

KEY_ENTITIES = ['post', 'section', 'note', 'channel']
