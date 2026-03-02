"""Add target_tools labels to ambiguous_first Hugo eval turns.

Pattern varies:
  Most convos: T1=ambiguous (handle_ambiguity + candidate tools), T3=resolved.
  Some convos: T1=clear, T3=ambiguous (e.g., hugo_016).
  Some T3: user resolves to a flow outside the original candidates.
"""

import json

LABELS = {
    # hugo_014: T1=ambiguous [format, release], T3=preview
    ("hugo_014", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["format_content", "publish_post"]},
        "format_content": {"post_id": None, "platform": "WordPress"},
        "publish_post": {"post_id": None, "platform": "WordPress", "action": "publish"},
    },
    # T3: "Is this how it will look when published?" — preview (not a candidate)
    ("hugo_014", 3): {
        "render_preview": {"post_id": None},
    },

    # hugo_016: T1=inspect (clear), T3=ambiguous [polish, rework]
    ("hugo_016", 1): {
        "analyze_content": {"post_id": None},
        "check_readability": {"content": None},
    },
    # T3: "Fix the dryness, pull from here: [URL]"
    # REVISIT: includes URL for external source — search_sources/search_evidence?
    ("hugo_016", 3): {
        "handle_ambiguity": {"clarification": None, "candidates": ["revise_content", "search_evidence"]},
        "revise_content": {"focus": None, "instructions": "fix dryness, incorporate external content"},
        "search_evidence": {"query": None},
        "search_sources": {"query": None},
    },

    # hugo_017: T1=ambiguous [amend, diff], T3=amend
    ("hugo_017", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["revise_content", "diff_versions"]},
        "revise_content": {"focus": "attention", "instructions": None},
        "diff_versions": {"section": "attention"},
    },
    ("hugo_017", 3): {
        "revise_content": {"focus": "attention", "instructions": "fix parallel processing analogy"},
        "update_section": {"section": "attention"},
    },

    # hugo_018: T1=ambiguous [expand, rework], T3=expand
    ("hugo_018", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["expand_content", "revise_content"]},
        "expand_content": {"source_content": None, "instructions": "develop into full prose"},
        "revise_content": {"focus": "overnight train", "instructions": "restructure"},
    },
    ("hugo_018", 3): {
        "expand_content": {"source_content": None, "instructions": "turn into paragraphs"},
    },

    # hugo_019: T1=ambiguous [tidy, format], T3=tidy
    ("hugo_019", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["normalize_structure", "format_content"]},
        "normalize_structure": {"post_id": None},
        "format_content": {"post_id": None, "platform": None},
    },
    ("hugo_019", 3): {
        "normalize_structure": {"post_id": None},
    },

    # hugo_020: T1=ambiguous [browse, find], T3=find
    ("hugo_020", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["search_posts", "browse_topics", "search_inspiration"]},
        "search_posts": {"query": "fermentation"},
        "browse_topics": {"category": "fermentation"},
        "search_inspiration": {"topic": "fermentation"},
    },
    ("hugo_020", 3): {
        "search_posts": {"query": "fermentation"},
    },

    # hugo_021: T1=ambiguous [outline, brainstorm], T3=brainstorm
    ("hugo_021", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["generate_outline", "brainstorm_ideas"]},
        "generate_outline": {"topic": "async communication culture"},
        "brainstorm_ideas": {"topic": "async communication culture"},
    },
    ("hugo_021", 3): {
        "brainstorm_ideas": {"topic": "async communication culture"},
    },

    # hugo_022: T1=ambiguous [promote, release], T3=release
    ("hugo_022", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["publish_post", "list_platforms"]},
        "publish_post": {"post_id": None, "action": "publish"},
        "list_platforms": {},
    },
    ("hugo_022", 3): {
        "publish_post": {"post_id": None, "action": "publish"},
    },

    # hugo_023: T1=ambiguous [syndicate, schedule], T3=syndicate
    ("hugo_023", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["format_content", "publish_post", "manage_schedule"]},
        "format_content": {"platform": "Medium"},
        "publish_post": {"platform": "Medium", "action": "publish"},
        "manage_schedule": {"action": "schedule", "platform": "Medium"},
    },
    ("hugo_023", 3): {
        "format_content": {"platform": "Medium"},
        "publish_post": {"platform": "Medium", "action": "publish"},
    },

    # hugo_024: T1=ambiguous [inspect, view], T3=view
    ("hugo_024", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["analyze_content", "read_post"]},
        "analyze_content": {"post_id": None},
        "read_post": {"post_id": None},
    },
    ("hugo_024", 3): {
        "read_post": {"post_id": None},
    },

    # hugo_081: T1=ambiguous [survey, syndicate], T3=survey
    ("hugo_081", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["check_platform", "list_platforms", "format_content"]},
        "check_platform": {"platform": "Substack"},
        "list_platforms": {},
        "format_content": {"platform": "Substack"},
    },
    ("hugo_081", 3): {
        "check_platform": {"platform": "Substack"},
        "list_platforms": {},
    },

    # hugo_082: T1=ambiguous [tone, audit], T3=tone
    ("hugo_082", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["adjust_tone", "audit_style"]},
        "adjust_tone": {"source_content": None, "target_tone": None},
        "audit_style": {"post_id": None, "aspects": ["voice"]},
    },
    ("hugo_082", 3): {
        "adjust_tone": {"source_content": None, "target_tone": "conversational, loose"},
    },

    # hugo_083: T1=ambiguous [rework, tidy], T3=rework
    ("hugo_083", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["revise_content", "normalize_structure"]},
        "revise_content": {"instructions": "major revision"},
        "normalize_structure": {"post_id": None},
    },
    ("hugo_083", 3): {
        "revise_content": {"focus": "self-attention", "instructions": "restructure argument"},
    },

    # hugo_084: T1=ambiguous [undo, amend], T3=undo
    ("hugo_084", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["rollback_post", "revise_content"]},
        "rollback_post": {"post_id": None},
        "revise_content": {"focus": "Thailand", "instructions": "fix last change"},
    },
    ("hugo_084", 3): {
        "rollback_post": {"post_id": None},
    },

    # hugo_085: T1=ambiguous [syndicate, schedule], T3=syndicate
    ("hugo_085", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["format_content", "publish_post", "manage_schedule"]},
        "format_content": {"platform": "Medium"},
        "publish_post": {"platform": "Medium", "action": "publish"},
        "manage_schedule": {"action": "schedule"},
    },
    ("hugo_085", 3): {
        "format_content": {"platform": "Medium"},
        "publish_post": {"platform": "Medium", "action": "publish"},
    },

    # hugo_086: T1=ambiguous [polish, diff], T3=polish
    ("hugo_086", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["revise_content", "diff_versions"]},
        "revise_content": {"focus": "database migration", "instructions": None},
        "diff_versions": {"section": "database migration"},
    },
    ("hugo_086", 3): {
        "revise_content": {"focus": "database migration", "instructions": "tighten transitions, shorten sentences"},
    },

    # hugo_087: T1=ambiguous [write, add], T3=write
    ("hugo_087", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["write_section", "generate_prose", "insert_section"]},
        "write_section": {"section": "preserving summer tomatoes"},
        "generate_prose": {"topic": "preserving summer tomatoes"},
        "insert_section": {"title": "Preserving Summer Tomatoes"},
    },
    ("hugo_087", 3): {
        "write_section": {"section": "preserving summer tomatoes"},
        "generate_prose": {"topic": "water bath canning and freezing methods"},
    },

    # hugo_088: T1=ambiguous [suggest, explain], T3=suggest
    # Same pattern as dana_095 — suggest/recommend flow, advisory
    ("hugo_088", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["brainstorm_ideas", "detect_issues", "explain_action"]},
        "brainstorm_ideas": {"topic": "VS Code review"},
        "detect_issues": {"content": None},
        "explain_action": {"topic": "VS Code review"},
    },
    ("hugo_088", 3): {
        "brainstorm_ideas": {"topic": "VS Code review", "angle": "what to work on next"},
        "detect_issues": {"content": None},
    },

    # hugo_089: T1=ambiguous [preference, tone], T3=preference
    ("hugo_089", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["manage_memory", "adjust_tone"]},
        "manage_memory": {"operation": "write", "level": "L2", "key": "tone", "value": "authoritative"},
        "adjust_tone": {"source_content": None, "target_tone": "authoritative, less casual"},
    },
    ("hugo_089", 3): {
        "manage_memory": {"operation": "write", "level": "L2", "key": "tone", "value": "authoritative"},
    },

    # hugo_090: T1=ambiguous [write, expand], T3=write
    ("hugo_090", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["write_section", "generate_prose", "expand_content"]},
        "write_section": {"section": "authentication"},
        "generate_prose": {"topic": "authentication"},
        "expand_content": {"source_content": None, "instructions": "develop authentication section"},
    },
    ("hugo_090", 3): {
        "write_section": {"section": "authentication"},
        "generate_prose": {"topic": "authentication"},
    },

    # hugo_091: T1=ambiguous [find, browse], T3=find
    ("hugo_091", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["search_posts", "browse_topics"]},
        "search_posts": {"query": "reinforcement learning"},
        "browse_topics": {"category": "reinforcement learning"},
    },
    ("hugo_091", 3): {
        "search_posts": {"query": "RLHF"},
    },

    # hugo_092: T1=ambiguous [view, find], T3=view
    ("hugo_092", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["read_post", "search_posts"]},
        "read_post": {"post_id": None},
        "search_posts": {"query": "urban living"},
    },
    ("hugo_092", 3): {
        "read_post": {"post_id": None},
    },

    # hugo_093: T1=ambiguous [suggest, explain], T3=brainstorm (shifted outside candidates)
    ("hugo_093", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["brainstorm_ideas", "detect_issues", "explain_action"]},
        "brainstorm_ideas": {"topic": "transformer section"},
        "detect_issues": {"content": None},
        "explain_action": {"topic": "transformer section"},
    },
    ("hugo_093", 3): {
        "brainstorm_ideas": {"topic": "transformer section", "angle": "missing angles"},
    },

    # hugo_094: T1=ambiguous [format, tidy], T3=format
    ("hugo_094", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["format_content", "normalize_structure"]},
        "format_content": {"post_id": None, "platform": None},
        "normalize_structure": {"post_id": None},
    },
    ("hugo_094", 3): {
        "format_content": {"post_id": None, "platform": "Medium"},
        "update_metadata": {"tags": None},
    },

    # hugo_095: T1=ambiguous [brainstorm, outline], T3=brainstorm
    ("hugo_095", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["brainstorm_ideas", "generate_outline"]},
        "brainstorm_ideas": {"topic": "beginner investing"},
        "generate_outline": {"topic": "beginner investing"},
    },
    ("hugo_095", 3): {
        "brainstorm_ideas": {"topic": "beginner investing"},
    },

    # hugo_096: T1=ambiguous [audit, compare], T3=audit
    ("hugo_096", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["audit_style", "compare_posts"]},
        "audit_style": {"post_id": None, "aspects": ["style"]},
        "compare_posts": {"post_ids": None, "aspects": ["style"]},
    },
    ("hugo_096", 3): {
        "audit_style": {"post_id": None, "aspects": ["voice"]},
    },

    # hugo_097: T1=ambiguous [syndicate, promote], T3=syndicate
    ("hugo_097", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["format_content", "publish_post", "list_platforms"]},
        "format_content": {"platform": None},
        "publish_post": {"action": "promote"},
        "list_platforms": {},
    },
    ("hugo_097", 3): {
        "format_content": {"platform": "Substack"},
        "publish_post": {"platform": "Substack", "action": "publish"},
    },

    # hugo_100: T1=ambiguous [release, chat], T3=release
    ("hugo_100", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["publish_post", "no_tool_needed"]},
        "publish_post": {"post_id": None, "action": "publish"},
        "no_tool_needed": {"reason": "user venting about missed deadline"},
    },
    ("hugo_100", 3): {
        "publish_post": {"post_id": None, "action": "publish"},
    },

    # hugo_101: T1=ambiguous [diff, view], T3=diff
    ("hugo_101", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["diff_versions", "read_post"]},
        "diff_versions": {"section": "intro"},
        "read_post": {"post_id": None},
    },
    ("hugo_101", 3): {
        "diff_versions": {"section": "intro"},
    },

    # hugo_102: T1=ambiguous [rework, tidy], T3=rework
    ("hugo_102", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["revise_content", "normalize_structure"]},
        "revise_content": {"instructions": "major revision"},
        "normalize_structure": {"post_id": None},
    },
    ("hugo_102", 3): {
        "revise_content": {"focus": "commuter isolation", "instructions": "rewrite, argument falls apart in section 3"},
    },

    # hugo_103: T1=ambiguous [expand, rework], T3=expand
    ("hugo_103", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["expand_content", "revise_content"]},
        "expand_content": {"source_content": None, "instructions": "fill in skeleton"},
        "revise_content": {"instructions": "structural overhaul"},
    },
    ("hugo_103", 3): {
        "expand_content": {"source_content": None, "instructions": "flesh out into full content"},
    },

    # hugo_104: T1=ambiguous [compare, find], T3=compare
    ("hugo_104", 1): {
        "handle_ambiguity": {"clarification": None, "candidates": ["compare_posts", "search_posts", "read_post"]},
        "compare_posts": {"post_ids": None},
        "search_posts": {"query": "Thailand Vietnam"},
        "read_post": {"post_id": None},
    },
    ("hugo_104", 3): {
        "compare_posts": {"post_ids": None, "aspects": ["tone", "style"]},
    },
}


def main():
    with open("datasets/hugo/eval_set.json") as f:
        data = json.load(f)

    labeled = 0
    missing = 0
    for convo in data:
        if convo["category"] != "ambiguous_first":
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
