"""Add target_tools labels to same_flow Hugo eval turns."""

import json

LABELS = {
    # hugo_001: refine (Draft) — transformer post, reorder self-attention section
    ("hugo_001", 1): {
        "revise_content": {"source_content": None, "instructions": "reorder from Q/K/V intuition into scaled dot-product", "focus": "self-attention"},
        "update_section": {"section": "self-attention", "content": None},
    },
    ("hugo_001", 3): {
        "revise_content": {"source_content": None, "instructions": "same reordering", "focus": "multi-head attention"},
        "update_section": {"section": "multi-head attention", "content": None},
    },

    # hugo_002: create (Draft) — new monolith post
    ("hugo_002", 1): {
        "create_post": {"title": "Why We Broke Apart Our Monolith", "topic": "monolith to microservices migration"},
    },
    # T3: "That's a terrible title, please try again" — regenerate the title
    ("hugo_002", 3): {
        "brainstorm_ideas": {"topic": "monolith to microservices migration", "angle": "blog post titles"},
        "update_metadata": {"title": None},
    },

    # hugo_003: inspect (Research) — food blog, check for missing things
    # T1: "it" is unspecified — agent asks. Tool still applies, post_id unknown.
    ("hugo_003", 1): {
        "analyze_content": {"post_id": None},
    },
    ("hugo_003", 3): {
        "analyze_content": {"post_id": "butternut_squash_soup"},
    },

    # hugo_004: polish (Revise) — developer tools review, tighten intro
    ("hugo_004", 1): {
        "revise_content": {"source_content": None, "instructions": "tighten, less clunky", "focus": "intro"},
    },
    ("hugo_004", 3): {
        "revise_content": {"source_content": None, "instructions": "tighten, less wordy", "focus": "performance benchmarks"},
    },

    # hugo_005: rework (Revise) — remote work, restructure sections
    ("hugo_005", 1): {
        "revise_content": {"source_content": None, "instructions": "restructure, argument isn't landing", "focus": "async communication"},
    },
    ("hugo_005", 3): {
        "revise_content": {"source_content": None, "instructions": "rewrite, trails off", "focus": "conclusion"},
    },

    # hugo_006: compare (Research) — compare API tutorials
    ("hugo_006", 1): {
        "compare_posts": {"post_ids": None, "aspects": ["word_count", "sentence_length", "vocabulary"]},
        "search_posts": {"query": "API tutorial"},
    },
    ("hugo_006", 3): {
        "compare_posts": {"post_ids": None, "aspects": None},
    },

    # hugo_007: schedule (Publish) — newsletter schedule
    ("hugo_007", 1): {
        "manage_schedule": {"post_id": None, "action": "schedule", "datetime": "Friday 8:00 AM EST"},
    },
    ("hugo_007", 3): {
        "manage_schedule": {"post_id": None, "action": "reschedule", "datetime": "Thursday evening"},
    },

    # hugo_008: expand (Draft) — flesh out bullet points
    ("hugo_008", 1): {
        "expand_content": {"source_content": "noise pollution", "instructions": "flesh out into full paragraph"},
    },
    ("hugo_008", 3): {
        "expand_content": {"source_content": "commuting", "instructions": "flesh out into full paragraph"},
    },

    # hugo_033: brainstorm (Draft) — AI roundup angles
    ("hugo_033", 1): {
        "brainstorm_ideas": {"topic": "AI research roundup"},
    },
    ("hugo_033", 3): {
        "brainstorm_ideas": {"topic": "AI safety papers nobody's reading; weird niche applications from arxiv", "angle": "hooks"},
    },

    # hugo_034: survey (Publish) — which platforms are connected
    ("hugo_034", 1): {
        "list_platforms": {},
    },
    ("hugo_034", 3): {
        "check_platform": {"platform": "Substack"},
    },

    # hugo_035: check (Research) — what's in the pipeline
    ("hugo_035", 1): {
        "search_posts": {"status": "draft"},
    },
    ("hugo_035", 3): {
        "search_posts": {"query": "spring", "status": "draft"},
        "search_by_time": {"start_date": None},
    },

    # hugo_036: diff (Revise) — version comparison
    ("hugo_036", 1): {
        "diff_versions": {"section": "Bangkok"},
    },
    ("hugo_036", 3): {
        "diff_versions": {"section": "Chiang Mai"},
    },

    # hugo_037: tidy (Revise) — formatting cleanup, no word changes
    ("hugo_037", 1): {
        "normalize_structure": {"post_id": None},
    },
    ("hugo_037", 3): {
        "normalize_structure": {"post_id": "compound_interest"},
    },

    # hugo_038: tone (Revise) — shift from casual to professional
    ("hugo_038", 1): {
        "adjust_tone": {"source_content": None, "target_tone": "professional technical"},
    },
    ("hugo_038", 3): {
        "adjust_tone": {"source_content": None, "target_tone": "professional but accessible", "intensity": "lighter"},
    },

    # hugo_039: polish (Revise) — clean up writing in food blog
    ("hugo_039", 1): {
        "revise_content": {"source_content": None, "instructions": "clean up, less clunky", "focus": "intro"},
    },
    ("hugo_039", 3): {
        "revise_content": {"source_content": None, "instructions": "clean up", "focus": "sourcing ingredients locally"},
    },

    # hugo_040: undo (Converse) — rollback last change
    ("hugo_040", 1): {
        "rollback_post": {"post_id": None},
    },
    ("hugo_040", 3): {
        "rollback_post": {"post_id": None, "version": None},
    },

    # hugo_041: browse (Research) — trending thought leadership topics
    ("hugo_041", 1): {
        "browse_topics": {"category": "thought leadership"},
        "search_inspiration": {"topic": "thought leadership"},
    },
    ("hugo_041", 3): {
        "browse_topics": {"category": "remote work"},
    },

    # hugo_042: explain (Converse) — what did you change
    ("hugo_042", 1): {
        "explain_action": {"topic": "REST API tutorial changes"},
    },
    ("hugo_042", 3): {
        "explain_action": {"topic": "endpoint routing changes", "scope": "routing section"},
    },

    # hugo_043: schedule (Publish) — set publication time
    ("hugo_043", 1): {
        "manage_schedule": {"action": "schedule", "platform": "Substack", "datetime": "next Tuesday morning"},
    },
    ("hugo_043", 3): {
        "manage_schedule": {"action": "schedule", "datetime": "8am EST"},
    },

    # hugo_044: format (Revise) — prepare for publishing on Medium
    ("hugo_044", 1): {
        "format_content": {"post_id": None, "platform": None},
    },
    ("hugo_044", 3): {
        "format_content": {"post_id": None, "platform": "Medium"},
        "update_metadata": {"excerpt": None, "tags": None},
    },

    # hugo_045: write (Draft) — draft new sections for transformer post
    ("hugo_045", 1): {
        "write_section": {"section": "multi-head attention"},
        "generate_prose": {"topic": "multi-head attention"},
    },
    ("hugo_045", 3): {
        "write_section": {"section": "positional encoding", "instructions": "focus on sinusoidal vs learned embeddings"},
        "generate_prose": {"topic": "positional encoding", "instructions": "focus on sinusoidal vs learned embeddings"},
    },

    # hugo_046: release (Publish) — push posts live
    ("hugo_046", 1): {
        "publish_post": {"post_id": None, "platform": "WordPress", "action": "publish"},
    },
    ("hugo_046", 3): {
        "publish_post": {"post_id": None, "action": "publish"},
    },

    # hugo_047: add (Draft) — insert new sections
    ("hugo_047", 1): {
        "insert_section": {"title": "Index Funds", "position": "after intro"},
    },
    ("hugo_047", 3): {
        "insert_section": {"title": "Compound Interest", "position": "before Index Funds"},
    },

    # hugo_048: find (Research) — search for prior posts
    ("hugo_048", 1): {
        "search_posts": {"query": "microservices"},
    },
    ("hugo_048", 3): {
        "search_posts": {"query": "database sharding"},
    },

    # hugo_049: view (Research) — load full post content
    ("hugo_049", 1): {
        "read_post": {"post_id": "rhubarb_galette"},
    },
    ("hugo_049", 3): {
        "read_post": {"post_id": "spring_onions"},
    },

    # hugo_050: check (Research) — post status overview
    ("hugo_050", 1): {
        "search_posts": {"category": "reviews"},
        "get_post": {"post_id": None},
    },
    ("hugo_050", 3): {
        "search_posts": {"status": "draft", "category": "reviews"},
    },

    # hugo_051: promote (Publish) — distribute to subscribers and social
    ("hugo_051", 1): {
        "publish_post": {"post_id": None, "action": "publish"},
    },
    # T3: "Pin it" — closest tool is publish_post or update_metadata. Revisit.
    ("hugo_051", 3): {
        "publish_post": {"post_id": None, "action": "pin"},
        "update_metadata": {"post_id": None, "status": "pinned"},
    },

    # hugo_052: expand (Draft) — flesh out bullet points
    ("hugo_052", 1): {
        "expand_content": {"source_content": None, "instructions": "flesh out bullet points in authentication section"},
    },
    ("hugo_052", 3): {
        "expand_content": {"source_content": None, "instructions": "flesh out notes in error handling section"},
    },

    # hugo_053: create (Draft) — new newsletter post
    ("hugo_053", 1): {
        "create_post": {"title": None, "topic": "AI research roundup"},
    },
    ("hugo_053", 3): {
        "update_metadata": {"title": "AI Weekly #12: Agents, Benchmarks, and Open Models"},
        "create_post": {"title": "AI Weekly #12: Agents, Benchmarks, and Open Models"},
    },

    # hugo_054: refine (Draft) — restructure essay section
    # T1 involves reorder + delete — multiple operations
    ("hugo_054", 1): {
        "revise_content": {"instructions": "move safety stats before walkability, drop parking subsection", "focus": "neighborhood comparison"},
        "reorder_sections": {"order": None},
    },
    ("hugo_054", 3): {
        "revise_content": {"instructions": "fold parking into walkability section", "focus": "walkability"},
        "update_section": {"section": "walkability", "content": None},
    },

    # hugo_055: cancel (Publish) — cancel scheduled publication
    ("hugo_055", 1): {
        "manage_schedule": {"post_id": None, "action": "cancel"},
    },
    ("hugo_055", 3): {
        "manage_schedule": {"post_id": None, "action": "cancel"},
    },

    # hugo_056: check (Research) — post status for SE Asia series
    ("hugo_056", 1): {
        "search_posts": {"query": "Southeast Asia"},
    },
    ("hugo_056", 3): {
        "get_post": {"post_id": "vietnam_by_train", "include_history": True},
        "search_by_time": {"date_field": "modified"},
    },
}


def main():
    with open("eval/eval_hugo.json") as f:
        data = json.load(f)

    labeled = 0
    missing = 0
    for convo in data:
        if convo["category"] != "same_flow":
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

    with open("eval/eval_hugo.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"Labeled {labeled} turns, {missing} missing")


if __name__ == "__main__":
    main()
