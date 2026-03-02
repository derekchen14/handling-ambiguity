"""Add target_tools labels to ambiguous_second Hugo eval turns.

Pattern:
  T1: clear flow — standard tool labeling.
  T3: flow=blueprint (Plan intent) with candidate_flows — user requests
      multiple operations at once. Include tools for ALL candidate operations.
  Exception: hugo_029 T1 is ambiguous [check, view, inspect].
"""

import json

LABELS = {
    # hugo_025: T1=view, T3=blueprint [rework, schedule]
    ("hugo_025", 1): {
        "read_post": {"post_id": None, "section": "authentication middleware"},
    },
    # T3: "Rewrite the code examples for Express 5 so we can send it off on Friday."
    ("hugo_025", 3): {
        "revise_content": {"focus": "code examples", "instructions": "rewrite for Express 5"},
        "update_section": {"section": None},
        "manage_schedule": {"action": "schedule", "date": "Friday"},
    },

    # hugo_026: T1=tone, T3=blueprint [brainstorm, outline]
    ("hugo_026", 1): {
        "adjust_tone": {"source_content": None, "target_tone": "casual, conversational"},
    },
    # T3: "throw me some angles and map out what the sections would be"
    ("hugo_026", 3): {
        "brainstorm_ideas": {"topic": "rent control follow-up"},
        "generate_outline": {"topic": "rent control follow-up"},
    },

    # hugo_027: T1=polish, T3=blueprint [inspect, release]
    ("hugo_027", 1): {
        "revise_content": {"focus": "alert fatigue", "instructions": "finalize"},
        "check_grammar": {"section": "alert fatigue"},
    },
    # T3: "Check the rest first before publishing."
    ("hugo_027", 3): {
        "analyze_content": {"post_id": None},
        "check_readability": {"content": None},
        "check_links": {"post_id": None},
        "publish_post": {"post_id": None, "action": "publish"},
    },

    # hugo_028: T1=schedule, T3=blueprint [diff, create]
    ("hugo_028", 1): {
        "manage_schedule": {"action": "schedule", "date": "Thursday 6pm"},
    },
    # T3: "Show me what changed since last week's issue, and then start a fresh post on reasoning models."
    ("hugo_028", 3): {
        "diff_versions": {"section": None},
        "create_post": {"title": "reasoning models"},
    },

    # hugo_029: T1=ambiguous [check, view, inspect], T3=blueprint [polish, promote]
    ("hugo_029", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["get_post", "read_post", "analyze_content"]},
        "get_post": {"post_id": None},
        "read_post": {"post_id": None},
        "analyze_content": {"post_id": None},
        "check_readability": {"content": None},
    },
    # T3: "The sourcing note could use some love before we feature it on the homepage."
    ("hugo_029", 3): {
        "revise_content": {"focus": "sourcing note", "instructions": "improve"},
        "check_grammar": {"section": "sourcing note"},
        "publish_post": {"post_id": None, "action": "promote"},
    },

    # hugo_030: T1=write, T3=blueprint [compare, preview]
    ("hugo_030", 1): {
        "write_section": {"section": "performance benchmarks"},
        "generate_prose": {"topic": "startup time, memory, extension load, Copilot latency"},
    },
    # T3: "Pull it up next to the VS Code review so I can see how it'll look on WordPress."
    ("hugo_030", 3): {
        "compare_posts": {"post_ids": None},
        "read_post": {"post_id": None},
        "render_preview": {"post_id": None},
    },

    # hugo_031: T1=release, T3=blueprint [view, outline]
    ("hugo_031", 1): {
        "publish_post": {"post_id": None, "action": "publish"},
    },
    # T3: "That async culture draft I started. Pull it up so I can plan Part 2."
    ("hugo_031", 3): {
        "read_post": {"post_id": None},
        "generate_outline": {"topic": "async culture Part 2"},
    },

    # hugo_032: T1=tidy, T3=blueprint [expand, polish, tone] — 3 candidates
    ("hugo_032", 1): {
        "normalize_structure": {"post_id": None},
    },
    # T3: "Also flesh out the Laos section. Then double check all the grammar and tone for it."
    ("hugo_032", 3): {
        "expand_content": {"source_content": None, "instructions": "flesh out Laos section"},
        "revise_content": {"focus": None, "instructions": "check grammar"},
        "check_grammar": {"section": None},
        "adjust_tone": {"source_content": None, "target_tone": None},
    },

    # hugo_105: T1=syndicate, T3=blueprint [survey, browse]
    ("hugo_105", 1): {
        "format_content": {"platform": "Substack"},
        "publish_post": {"platform": "Substack", "action": "publish"},
    },
    # T3: "can you check those? Also what's trending in AI right now for next week's edition"
    ("hugo_105", 3): {
        "check_platform": {"platform": None},
        "list_platforms": {},
        "browse_topics": {"category": "AI"},
        "search_inspiration": {"topic": "AI"},
    },

    # hugo_106: T1=chat, T3=blueprint [diff, explain]
    ("hugo_106", 1): {
        "no_tool_needed": {"reason": "conversational question about blog niche"},
    },
    # T3: "Show me what changed in the intro. Why'd you rewrite it that way?"
    ("hugo_106", 3): {
        "diff_versions": {"section": "intro"},
        "explain_action": {"topic": "intro rewrite"},
    },

    # hugo_107: T1=amend, T3=blueprint [audit, cancel]
    ("hugo_107", 1): {
        "revise_content": {"focus": "attention mechanism", "instructions": "restore computational complexity argument"},
    },
    # T3: "check if the terminology matches my older ML posts. Actually pull that scheduled Thursday publish too"
    ("hugo_107", 3): {
        "audit_style": {"post_id": None, "aspects": ["terminology"]},
        "read_post": {"post_id": None},
        "manage_schedule": {"action": "cancel"},
    },

    # hugo_108: T1=find, T3=blueprint [dismiss, schedule]
    ("hugo_108", 1): {
        "search_posts": {"query": "Vietnam"},
        "search_by_time": {"query": "Vietnam"},
    },
    # T3: "Nah skip that idea. Just queue the Ha Giang one for next Tuesday on WordPress"
    ("hugo_108", 3): {
        "no_tool_needed": {"reason": "user declined idea"},
        "manage_schedule": {"action": "schedule", "date": "next Tuesday", "platform": "WordPress"},
    },

    # hugo_109: T1=compare, T3=blueprint [write, preference]
    ("hugo_109", 1): {
        "compare_posts": {"post_ids": None, "aspects": ["structure", "vocabulary"]},
    },
    # T3: "Draft a DCA section for the index funds post. Also default to grade 7 readability for my beginner stuff."
    ("hugo_109", 3): {
        "write_section": {"section": "DCA"},
        "generate_prose": {"topic": "dollar cost averaging"},
        "manage_memory": {"operation": "write", "level": "L2", "key": "readability", "value": "grade 7"},
    },

    # hugo_110: T1=chat, T3=blueprint [schedule, audit]
    ("hugo_110", 1): {
        "no_tool_needed": {"reason": "conversational question about posting frequency"},
    },
    # T3: "Queue part 3 for next Tuesday 9am. But first check if the language matches the two we already published."
    ("hugo_110", 3): {
        "manage_schedule": {"action": "schedule", "date": "next Tuesday 9am"},
        "audit_style": {"post_id": None, "aspects": ["language", "consistency"]},
    },

    # hugo_111: T1=rework, T3=blueprint [promote, check]
    ("hugo_111", 1): {
        "revise_content": {"instructions": "complete overhaul, fix structure and intro"},
    },
    # T3: "Push it to the subscriber list. How's the rest of the seasonal series looking?"
    ("hugo_111", 3): {
        "publish_post": {"post_id": None, "action": "promote"},
        "search_posts": {"query": "seasonal series"},
        "search_by_time": {"query": "seasonal"},
    },

    # hugo_112: T1=add, T3=blueprint [tone, write]
    ("hugo_112", 1): {
        "insert_section": {"title": "Pricing Breakdown", "position": 4},
    },
    # T3: "Way too formal for devs. Also fill in Pricing Breakdown, just free vs paid and student discounts."
    ("hugo_112", 3): {
        "adjust_tone": {"source_content": None, "target_tone": "casual, developer-friendly"},
        "write_section": {"section": "Pricing Breakdown"},
        "generate_prose": {"topic": "free vs paid, student discounts"},
    },

    # hugo_113: T1=preview, T3=blueprint [cancel, expand]
    ("hugo_113", 1): {
        "render_preview": {"post_id": None},
    },
    # T3: "pull it off the schedule entirely. Also section 2 is still just bullet points, turn those into real paragraphs"
    ("hugo_113", 3): {
        "manage_schedule": {"action": "cancel"},
        "expand_content": {"source_content": None, "instructions": "turn bullet points into paragraphs"},
    },

    # hugo_114: T1=inspect, T3=blueprint [brainstorm, expand]
    ("hugo_114", 1): {
        "analyze_content": {"post_id": None},
        "check_readability": {"content": None},
        "check_links": {"post_id": None},
    },
    # T3: "ideas for error handling and flesh out those auth bullets"
    ("hugo_114", 3): {
        "brainstorm_ideas": {"topic": "error handling"},
        "expand_content": {"source_content": None, "instructions": "flesh out auth bullets"},
    },

    # hugo_115: T1=promote, T3=blueprint [syndicate, schedule]
    ("hugo_115", 1): {
        "publish_post": {"post_id": None, "action": "promote"},
    },
    # T3: "Push it to LinkedIn too and while you're at it queue next week's edition for Thursday same time"
    ("hugo_115", 3): {
        "format_content": {"platform": "LinkedIn"},
        "publish_post": {"platform": "LinkedIn", "action": "publish"},
        "manage_schedule": {"action": "schedule", "date": "Thursday"},
    },

    # hugo_116: T1=tidy, T3=blueprint [add, tidy]
    ("hugo_116", 1): {
        "normalize_structure": {"post_id": None},
    },
    # T3: "Add a section on sirens becoming background music. Then do the same cleanup on the whole post."
    ("hugo_116", 3): {
        "insert_section": {"title": "sirens becoming background music"},
        "normalize_structure": {"post_id": None},
    },

    # hugo_117: T1=write, T3=blueprint [expand, write]
    ("hugo_117", 1): {
        "write_section": {"section": "self-attention mechanisms"},
        "generate_prose": {"topic": "QKV matrices, scaled dot product"},
    },
    # T3: "The positional encoding notes need fleshing out and I want a new section on sparse attention after that"
    ("hugo_117", 3): {
        "expand_content": {"source_content": None, "instructions": "flesh out positional encoding notes"},
        "write_section": {"section": "sparse attention"},
        "generate_prose": {"topic": "sparse attention"},
    },

    # hugo_118: T1=dismiss, T3=blueprint [promote, preview]
    ("hugo_118", 1): {
        "no_tool_needed": {"reason": "user declining/skipping"},
    },
    # T3: "let me see how the Thailand post looks before you blast it to the list"
    ("hugo_118", 3): {
        "render_preview": {"post_id": None},
        "publish_post": {"post_id": None, "action": "promote"},
    },

    # hugo_119: T1=schedule, T3=blueprint [add, preview]
    ("hugo_119", 1): {
        "manage_schedule": {"action": "schedule", "date": "next Tuesday 9am", "platform": "WordPress"},
    },
    # T3: "stick a FAQ section at the end before that goes out. Also let me see how it'll look on WordPress first"
    ("hugo_119", 3): {
        "insert_section": {"title": "FAQ"},
        "render_preview": {"post_id": None},
    },

    # hugo_120: T1=compare, T3=blueprint [chat, create]
    ("hugo_120", 1): {
        "compare_posts": {"post_ids": None, "aspects": ["structure", "tone"]},
    },
    # T3: "do you think our readers will even notice the tonal drift? If not then we're ready to just kick off the next one about Kubernetes."
    # REVISIT: conditional — chat (opinion) + create (conditional on answer)
    ("hugo_120", 3): {
        "no_tool_needed": {"reason": "asking for opinion about tonal drift"},
        "create_post": {"title": "Kubernetes"},
    },

    # hugo_121: T1=add, T3=blueprint [inspect, suggest]
    ("hugo_121", 1): {
        "insert_section": {"title": "Farmers Market Finds"},
    },
    # T3: "how's the post looking overall, like word count and anything missing? Also what do you think I should tackle next"
    ("hugo_121", 3): {
        "analyze_content": {"post_id": None},
        "check_readability": {"content": None},
        "brainstorm_ideas": {"topic": None},
    },

    # hugo_122: T1=chat, T3=blueprint [rework, tone]
    ("hugo_122", 1): {
        "no_tool_needed": {"reason": "conversational question about IDE comparison posts"},
    },
    # T3: "The JetBrains review needs a complete rebuild and the Sublime one is way too dry"
    ("hugo_122", 3): {
        "revise_content": {"instructions": "complete rebuild"},
        "adjust_tone": {"source_content": None, "target_tone": None},
    },

    # hugo_123: T1=refine, T3=blueprint [amend, release]
    ("hugo_123", 1): {
        "revise_content": {"focus": "hybrid models", "instructions": "trim and reorder"},
        "reorder_sections": {},
    },
    # T3: "That lost the timezone equity nuance. Fix it before we push it live."
    ("hugo_123", 3): {
        "revise_content": {"focus": "timezone equity", "instructions": "restore nuance"},
        "publish_post": {"post_id": None, "action": "publish"},
    },

    # hugo_124: T1=amend, T3=blueprint [inspect, outline]
    ("hugo_124", 1): {
        "revise_content": {"focus": "authentication", "instructions": "restore JWT expiration"},
    },
    # T3: "can you check if anything's missing so far? Empty parts, broken links, whatever."
    # + "Before I map out the remaining sections"
    ("hugo_124", 3): {
        "analyze_content": {"post_id": None},
        "check_links": {"post_id": None},
        "check_readability": {"content": None},
        "generate_outline": {"topic": "REST API guide remaining sections"},
    },

    # hugo_125: T1=audit, T3=blueprint [polish, brainstorm]
    ("hugo_125", 1): {
        "audit_style": {"post_id": None, "aspects": ["voice", "style"]},
    },
    # T3: "Clean up that intro so it reads more like my usual stuff. Also throw me some fresh angles for next week's issue."
    ("hugo_125", 3): {
        "revise_content": {"focus": "intro", "instructions": "match usual style"},
        "check_grammar": {"section": "intro"},
        "brainstorm_ideas": {"topic": "AI roundup next week"},
    },

    # hugo_126: T1=diff, T3=blueprint [tone, expand]
    ("hugo_126", 1): {
        "diff_versions": {"section": "neighborhood noise"},
    },
    # T3: "Too academic. Make it conversational, and flesh out the rent section bullets into paragraphs."
    ("hugo_126", 3): {
        "adjust_tone": {"source_content": None, "target_tone": "conversational"},
        "expand_content": {"source_content": None, "instructions": "flesh out rent section bullets into paragraphs"},
    },

    # hugo_127: T1=add, T3=blueprint [amend, suggest]
    ("hugo_127", 1): {
        "insert_section": {"title": "attention heads"},
    },
    # T3: "The self-attention rewrite lost my math notation. How should we tackle that, any ideas?"
    ("hugo_127", 3): {
        "revise_content": {"focus": "self-attention", "instructions": "restore math notation"},
        "brainstorm_ideas": {"topic": "math notation in blog posts"},
    },

    # hugo_128: T1=view, T3=blueprint [dismiss, check]
    ("hugo_128", 1): {
        "read_post": {"post_id": None},
    },
    # T3: "Nah skip that idea. Where do all my Southeast Asia posts stand right now"
    ("hugo_128", 3): {
        "no_tool_needed": {"reason": "user declined idea"},
        "search_posts": {"query": "Southeast Asia"},
        "search_by_time": {"query": "Southeast Asia"},
    },
}


def main():
    with open("datasets/hugo/eval_set.json") as f:
        data = json.load(f)

    labeled = 0
    missing = 0
    for convo in data:
        if convo["category"] != "ambiguous_second":
            continue
        cid = convo["convo_id"]
        for turn in convo["turns"]:
            if turn.get("speaker") != "user":
                continue
            key = (cid, turn["turn_num"])
            if key in LABELS:
                turn["target_tools"] = LABELS[key]
                labeled += 1
            else:
                print(f"WARNING: No label for {key}")
                missing += 1

    with open("datasets/hugo/eval_set.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"Labeled {labeled} turns, {missing} missing")


if __name__ == "__main__":
    main()
