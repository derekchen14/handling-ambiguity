"""Add target_tools labels to switch_flow Hugo eval turns."""

import json

LABELS = {
    # hugo_009: T1=tone, T3=preference
    ("hugo_009", 1): {
        "adjust_tone": {"source_content": None, "target_tone": "casual first-person, like a letter to a friend"},
    },
    ("hugo_009", 3): {
        "manage_memory": {"operation": "write", "level": "L2", "key": "tone", "value": "casual first-person storytelling"},
    },

    # hugo_010: T1=add, T3=release
    ("hugo_010", 1): {
        "insert_section": {"title": "Disclaimer", "position": "bottom"},
    },
    ("hugo_010", 3): {
        "publish_post": {"post_id": None, "action": "publish"},
    },

    # hugo_011: T1=write, T3=audit
    ("hugo_011", 1): {
        "write_section": {"section": "positional encoding"},
        "generate_prose": {"topic": "positional encoding"},
    },
    ("hugo_011", 3): {
        "audit_style": {"post_id": None, "aspects": ["voice"]},
    },

    # hugo_012: T1=find, T3=expand
    ("hugo_012", 1): {
        "search_posts": {"query": "Vietnam"},
    },
    ("hugo_012", 3): {
        "expand_content": {"source_content": None, "instructions": "build Hai Van Pass notes into real paragraphs"},
    },

    # hugo_013: T1=diff, T3=amend
    ("hugo_013", 1): {
        "diff_versions": {"section": "compound interest"},
    },
    # T3: "Put the original opening back and keep everything else" — selective restore
    ("hugo_013", 3): {
        "revise_content": {"instructions": "restore original opening, keep table and everything else"},
        "rollback_post": {"post_id": None},
    },

    # hugo_015: T1=view, T3=cancel
    ("hugo_015", 1): {
        "read_post": {"post_id": "asparagus_risotto"},
    },
    # T3: post is already live ("3 weeks") — unpublish, not cancel schedule
    ("hugo_015", 3): {
        "publish_post": {"post_id": None, "action": "unpublish"},
    },

    # hugo_057: T1=dismiss, T3=schedule
    ("hugo_057", 1): {
        "no_tool_needed": {"reason": "user dismissed suggestion"},
    },
    ("hugo_057", 3): {
        "manage_schedule": {"action": "schedule", "datetime": "Thursday 8am"},
    },

    # hugo_058: T1=suggest, T3=polish
    # T1: "I'm stuck, any ideas what I should work on next?" — triage, not brainstorm
    ("hugo_058", 1): {
        "brainstorm_ideas": {"topic": "urban living essay", "angle": "what to work on next"},
        "detect_issues": {"content": None},
    },
    ("hugo_058", 3): {
        "revise_content": {"focus": "Noise as Neighbors", "instructions": "tighten writing, don't change meaning"},
    },

    # hugo_059: T1=expand, T3=find
    ("hugo_059", 1): {
        "expand_content": {"source_content": None, "instructions": "flesh out bullet points into paragraphs"},
    },
    ("hugo_059", 3): {
        "search_posts": {"query": "multi-head attention"},
    },

    # hugo_060: T1=inspect, T3=polish
    ("hugo_060", 1): {
        "analyze_content": {"post_id": None},
        "check_readability": {"content": None},
    },
    ("hugo_060", 3): {
        "revise_content": {"focus": "Khao San Road", "instructions": "smooth out the flow"},
    },

    # hugo_061: T1=outline, T3=create
    ("hugo_061", 1): {
        "generate_outline": {"topic": "beginner investing strategies"},
    },
    ("hugo_061", 3): {
        "create_post": {"topic": "beginner investing strategies"},
    },

    # hugo_062: T1=promote, T3=syndicate
    ("hugo_062", 1): {
        "publish_post": {"post_id": None, "action": "promote"},
    },
    # T3: cross-post as native LinkedIn post → format + publish
    ("hugo_062", 3): {
        "publish_post": {"post_id": None, "platform": "LinkedIn", "action": "publish"},
        "format_content": {"post_id": None, "platform": "LinkedIn"},
    },

    # hugo_063: T1=expand, T3=survey
    ("hugo_063", 1): {
        "expand_content": {"source_content": None, "instructions": "turn winter squash bullet points into paragraphs"},
    },
    ("hugo_063", 3): {
        "list_platforms": {},
    },

    # hugo_064: T1=tone, T3=add
    ("hugo_064", 1): {
        "adjust_tone": {"source_content": None, "target_tone": "casual, conversational"},
    },
    ("hugo_064", 3): {
        "insert_section": {"title": "Pricing", "position": "before final verdict"},
    },

    # hugo_065: T1=browse, T3=explain
    ("hugo_065", 1): {
        "search_posts": {"query": "remote work"},
        "browse_topics": {"category": "remote work"},
    },
    ("hugo_065", 3): {
        "explain_action": {"topic": "async communication section changes"},
    },

    # hugo_066: T1=tidy, T3=inspect
    ("hugo_066", 1): {
        "normalize_structure": {"post_id": None},
    },
    ("hugo_066", 3): {
        "analyze_content": {"post_id": None},
    },

    # hugo_067: T1=format, T3=polish
    ("hugo_067", 1): {
        "format_content": {"post_id": None, "platform": "Substack"},
    },
    ("hugo_067", 3): {
        "revise_content": {"focus": "intro", "instructions": "tighten wording, don't change meaning"},
    },

    # hugo_068: T1=check, T3=compare
    ("hugo_068", 1): {
        "search_posts": {"query": "urban loneliness"},
        "get_post": {"post_id": None},
    },
    ("hugo_068", 3): {
        "compare_posts": {"post_ids": None, "aspects": ["tone", "sentence_patterns"]},
    },

    # hugo_069: T1=cancel, T3=inspect
    ("hugo_069", 1): {
        "publish_post": {"post_id": None, "platform": "Medium", "action": "unpublish"},
    },
    ("hugo_069", 3): {
        "analyze_content": {"post_id": None},
    },

    # hugo_070: T1=add, T3=dismiss
    ("hugo_070", 1): {
        "insert_section": {"title": "Street Food", "position": "after Bangkok intro"},
    },
    ("hugo_070", 3): {
        "no_tool_needed": {"reason": "user declined suggestion to reorder sections"},
    },

    # hugo_071: T1=promote, T3=polish
    ("hugo_071", 1): {
        "publish_post": {"post_id": None, "action": "promote"},
    },
    ("hugo_071", 3): {
        "revise_content": {"focus": "intro", "instructions": "clean up transitions, tighten wording"},
    },

    # hugo_072: T1=view, T3=find
    ("hugo_072", 1): {
        "read_post": {"post_id": None},
    },
    ("hugo_072", 3): {
        "search_posts": {"query": "service boundaries"},
    },

    # hugo_073: T1=polish, T3=rework
    ("hugo_073", 1): {
        "revise_content": {"focus": "intro", "instructions": "smooth it out"},
    },
    ("hugo_073", 3): {
        "revise_content": {"focus": "intro", "instructions": "restructure, get to recipe faster"},
    },

    # hugo_074: T1=amend, T3=preference
    ("hugo_074", 1): {
        "revise_content": {"instructions": "restore previous version of intro"},
        "rollback_post": {"post_id": None},
    },
    ("hugo_074", 3): {
        "manage_memory": {"operation": "write", "level": "L2", "key": "tone", "value": "casual"},
    },

    # hugo_075: T1=diff, T3=format
    ("hugo_075", 1): {
        "diff_versions": {"section": "remote work trends"},
    },
    ("hugo_075", 3): {
        "format_content": {"post_id": None, "platform": "Medium"},
        "update_metadata": {"tags": None},
    },

    # hugo_076: T1=amend, T3=tone
    ("hugo_076", 1): {
        "revise_content": {"instructions": "restore JWT vs session token comparison in authentication section"},
    },
    ("hugo_076", 3): {
        "adjust_tone": {"source_content": None, "target_tone": "conversational, walking someone through it"},
    },

    # hugo_077: T1=expand, T3=rework
    ("hugo_077", 1): {
        "expand_content": {"source_content": None, "instructions": "turn bullet points into paragraphs"},
    },
    ("hugo_077", 3): {
        "revise_content": {"focus": "diffusion models", "instructions": "restructure argument, cut fluff"},
    },

    # hugo_078: T1=expand, T3=add
    ("hugo_078", 1): {
        "expand_content": {"source_content": None, "instructions": "turn neighborhood sounds notes into paragraphs"},
    },
    ("hugo_078", 3): {
        "insert_section": {"title": "The Silence Between the Noise", "position": "after neighborhood sounds"},
    },

    # hugo_079: T1=format, T3=find
    ("hugo_079", 1): {
        "format_content": {"post_id": None, "platform": "Medium"},
    },
    ("hugo_079", 3): {
        "search_posts": {"query": "attention transformers"},
    },

    # hugo_080: T1=tidy, T3=audit
    ("hugo_080", 1): {
        "normalize_structure": {"post_id": None},
    },
    ("hugo_080", 3): {
        "audit_style": {"post_id": None, "aspects": ["style consistency"]},
    },

    # hugo_098: T1=rework, T3=polish
    ("hugo_098", 1): {
        "revise_content": {"focus": "VS Code", "instructions": "major revision"},
    },
    ("hugo_098", 3): {
        "revise_content": {"focus": "VS Code", "instructions": "make sentences more concise"},
    },

    # hugo_099: T1=release, T3=inspect
    ("hugo_099", 1): {
        "publish_post": {"post_id": None, "action": "publish"},
    },
    ("hugo_099", 3): {
        "analyze_content": {"post_id": None},
        "get_post": {"post_id": None},
    },
}


def main():
    with open("datasets/hugo/eval_set.json") as f:
        data = json.load(f)

    labeled = 0
    missing = 0
    for convo in data:
        if convo["category"] != "switch_flow":
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
