"""Apply Round 1 fixes to gen_hugo.json and gen_dana.json."""

import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[1] / 'eval'

# ── Punctuation artifact fix ─────────────────────────────────────────

def fix_punctuation_artifacts(text: str) -> str:
    """Fix em-dash replacement artifacts: ' ,  ' -> ', ' etc."""
    text = text.replace(' ,  ', ', ')
    text = text.replace(' ;  ', '; ')
    text = text.replace('  -  ', ' - ')
    return text


# ── Per-conversation fixes ───────────────────────────────────────────

HUGO_FIXES = {
    'hugo_001': {
        'turns': {
            0: {'utterance': "The self-attention section feels off. Reorder it from Q/K/V intuition into scaled dot-product."},
            2: {'utterance': "Same thing for multi-head attention."},
        }
    },
    'hugo_004': {
        'turns': {
            1: {'utterance': "Smoothing transitions and trimming redundant phrasing. The second paragraph especially had some awkward jumps between ideas."},
        }
    },
    'hugo_005': {
        'turns': {
            1: {'utterance': "The main argument wanders between three unrelated points. I'll restructure around a single throughline with stronger examples."},
        }
    },
    'hugo_006': {
        'turns': {
            1: {'utterance': "Comparing word count, sentence length, and vocabulary patterns between both. Should show if there's style drift."},
        }
    },
    'hugo_009': {
        'turns': {
            0: {'utterance': "Make the whole Southeast Asia series feel like a letter to a friend taking the same trip."},
        }
    },
    'hugo_016': {
        'turns': {
            2: {'utterance': "Fix the dryness, pull from here: https://www.example.com/reviews/dishwashers"},
        }
    },
    'hugo_033': {
        'turns': {
            2: {'utterance': "Love 3 and 4. Dig deeper with some hooks?"},
        }
    },
    'hugo_036': {
        'turns': {
            2: {'utterance': "Do Chiang Mai too, did the temple stuff get cut?"},
        }
    },
    'hugo_037': {
        'turns': {
            2: {'utterance': "Do compound interest too, bullets are a mess."},
        }
    },
    'hugo_038': {
        'turns': {
            0: {'utterance': "The microservices post reads like a Slack message. Needs to sound like a proper engineering blog."},
            2: {'utterance': "Too far on the intro. Reads like a white paper now."},
        }
    },
    'hugo_048': {
        'turns': {
            2: {'utterance': "What about database sharding? Don't wanna repeat ourselves."},
        }
    },
    # hugo_049: dismiss+dismiss -> view+view (structural fix)
    'hugo_049': {
        'turns': {
            0: {'utterance': "Pull up the rhubarb galette post", 'flow': 'view', 'intent': 'Research'},
            1: {'utterance': "Here it is. 900 words across 4 sections, still in draft. No featured image set."},
            2: {'utterance': "The spring onions one too", 'flow': 'view', 'intent': 'Research'},
        },
        'delete_keys': {0: ['flow', 'intent'], 2: ['flow', 'intent']},
    },
    'hugo_068': {
        'turns': {
            2: {'utterance': "Compare the tone and sentence patterns between the loneliness one and Noise as Neighbor?"},
        }
    },
    'hugo_075': {
        'turns': {
            2: {'utterance': "Keep the new version. Get it ready for Medium, fix the headings and add tags."},
        }
    },
    'hugo_083': {
        'turns': {
            2: {'utterance': "The content. The argument doesn't build on the self-attention part."},
        }
    },
    'hugo_089': {
        'turns': {
            2: {'utterance': "Save it as my default."},
        }
    },
    'hugo_092': {
        'turns': {
            2: {'utterance': "Show me the whole thing."},
        }
    },
    'hugo_095': {
        'turns': {
            2: {'utterance': "Angles and hooks first."},
        }
    },
    'hugo_096': {
        'turns': {
            2: {'utterance': "The second one. Make sure it matches our usual voice."},
        }
    },
    'hugo_104': {
        'turns': {
            2: {'utterance': "The style. Are they consistent or is one more formal?"},
        }
    },
    'hugo_106': {
        'turns': {
            2: {'utterance': "Show me what changed in the intro. Why'd you rewrite it that way?"},
        }
    },
    'hugo_109': {
        'turns': {
            2: {'utterance': "Draft a DCA section for the index funds post. Also default to grade 7 readability for my beginner stuff."},
        }
    },
    # hugo_110: candidate mismatch - "check if language matches" = audit, not expand
    'hugo_110': {
        'turns': {
            2: {
                'utterance': "Queue part 3 for next Tuesday 9am. But first check if the language matches the two we already published.",
                'candidate_flows': ['schedule', 'audit'],
                'candidate_intents': ['Publish', 'Revise'],
            },
        },
        'rationale': "('schedule', Publish) and ('audit', Revise) are operations from different intents.",
    },
    # hugo_111: bare "and" + not distinct ops -> rewrite with distinct ops
    'hugo_111': {
        'turns': {
            0: {'utterance': "The spring asparagus post needs a complete overhaul, structure is off and the intro is weak."},
            2: {
                'utterance': "Push it to the subscriber list. How's the rest of the seasonal series looking?",
                'candidate_flows': ['promote', 'check'],
                'candidate_intents': ['Publish', 'Research'],
            },
        },
        'rationale': "('promote', Publish) and ('check', Research) are operations from different intents.",
    },
    'hugo_112': {
        'turns': {
            2: {'utterance': "Way too formal for devs. Also fill in Pricing Breakdown, just free vs paid and student discounts."},
        }
    },
    'hugo_114': {
        'turns': {
            2: {'utterance': "Ideas for the error handling section before I start? Also default my posts to 3000 words."},
        }
    },
    'hugo_116': {
        'turns': {
            2: {'utterance': "Add a section on sirens becoming background music. Then do the same cleanup on the whole post."},
        }
    },
    'hugo_117': {
        'turns': {
            2: {'utterance': "Cross-post the transformer post to Dev.to. Also start a new one on diffusion models."},
        }
    },
    'hugo_122': {
        'turns': {
            2: {'utterance': "Tear apart the JetBrains review and rebuild it. Also pull Sublime Text off Friday's schedule."},
        }
    },
    'hugo_123': {
        'turns': {
            0: {'utterance': "The hybrid models section feels bloated. Trim it and move async communication before tooling."},
        }
    },
    'hugo_126': {
        'turns': {
            2: {'utterance': "Too academic. Make it conversational, and flesh out the rent section bullets into paragraphs."},
        }
    },
    'hugo_127': {
        'turns': {
            2: {'utterance': "The self-attention rewrite lost my math notation. What should I tackle next?"},
        }
    },
}


DANA_FIXES = {
    # dana_005: T3 reads as approve, not update -> make it an active update request
    'dana_005': {
        'turns': {
            2: {'utterance': "Yeah, lowercase those too while you're at it."},
        }
    },
    # dana_006: T3 reads as approve, not dashboard -> make it a dashboard refinement
    'dana_006': {
        'turns': {
            2: {'utterance': "Yea, and add a ticket severity breakdown too."},
        }
    },
    # dana_016: Agent T2 has no WHY
    'dana_016': {
        'turns': {
            1: {'utterance': "Done, transposed the table so each campaign is a row now. 14 campaigns across 6 engagement metrics."},
        }
    },
    'dana_032': {
        'turns': {
            2: {'utterance': "I think there are dupes, two Chris Moore's and a Jessica Mulaney. How many rows now?"},
        }
    },
    'dana_033': {
        'turns': {
            2: {'utterance': "Yeah do it. Also drop the null feature_ids."},
        }
    },
    'dana_037': {
        'turns': {
            2: {'utterance': "Same for click_through_rate, use a 3-row rolling average."},
        }
    },
    'dana_041': {
        'turns': {
            0: {'utterance': "The phone numbers in warehouse contacts are a mess, dashes, dots, nothing."},
            2: {'utterance': "Same mess in shipping_address. Hit that too."},
        }
    },
    'dana_042': {
        'turns': {
            2: {'utterance': "Both. Also make virality = shares / impressions * 100."},
        }
    },
    # dana_045: approve+approve -> segment+segment (structural fix)
    'dana_045': {
        'turns': {
            0: {'utterance': "Break down Q4 returns by product category", 'flow': 'segment', 'intent': 'Analyze'},
            1: {'utterance': "Electronics had the highest return rate at 14.2%, followed by Apparel at 11.8%. Want me to drill into a specific category?"},
            2: {'utterance': "Yeah, break Electronics down by region too.", 'flow': 'segment', 'intent': 'Analyze'},
        },
    },
    'dana_049': {
        'turns': {
            2: {'utterance': "Same for readmission, mark critical if under 30 days."},
        }
    },
    'dana_053': {
        'turns': {
            2: {'utterance': "Go back one more, the dedup dropped rows I needed."},
        }
    },
    'dana_054': {
        'turns': {
            0: {'utterance': "A lot of age fields are blank. Can you estimate from income and education?"},
        }
    },
    'dana_055': {
        'turns': {
            2: {'utterance': "Flag those. Same check on order_status: shipped, delivered, returned, cancelled only."},
        }
    },
    'dana_068': {
        'turns': {
            2: {'utterance': "Bigger than expected. Send that as Excel, leadership meeting tomorrow."},
        }
    },
    'dana_074': {
        'turns': {
            0: {'utterance': "Phone numbers in the contact column are a mess, dashes, parens, inconsistent."},
        }
    },
    # dana_081: T3 uses "pivot" (flow name)
    'dana_081': {
        'turns': {
            2: {'utterance': "Restructure it, long to wide."},
        }
    },
    # dana_082: T1 uses "pivot" (candidate flow name), T3 too long
    'dana_082': {
        'turns': {
            0: {'utterance': "I need the survey data reshaped so each question is its own column instead of rows"},
            2: {'utterance': "Just restructure it, one row per respondent."},
        }
    },
    'dana_083': {
        'turns': {
            2: {'utterance': "Just whether it exists, not sure it made this export."},
        }
    },
    'dana_096': {
        'turns': {
            2: {'utterance': "One column, like 'RetailOps-4100'. Break apart by the dash."},
        }
    },
    'dana_097': {
        'turns': {
            0: {'utterance': "For patient outcomes, I like dates as dd/mm/yyyy since these hospitals are UK."},
        }
    },
    'dana_103': {
        'turns': {
            2: {'utterance': "Just the chart, week over week by region on one graph."},
        }
    },
    'dana_112': {
        'turns': {
            2: {'utterance': "Some plan_type values have typos. Also flip the table so each month is its own column."},
        }
    },
    'dana_113': {
        'turns': {
            2: {'utterance': "Newark should be 'Newark NJ'. Also show stockout counts week over week."},
        }
    },
    'dana_115': {
        'turns': {
            2: {'utterance': "Stack last month's telemetry below. Also do that dedup you suggested."},
        }
    },
    # dana_117: bare "and" connector -> observation + question pattern
    'dana_117': {
        'turns': {
            2: {'utterance': "Flag the bad region values. How does monthly revenue look by region?"},
        }
    },
    'dana_118': {
        'turns': {
            2: {'utterance': "Clean hire_date to yyyy-mm-dd, four styles mixed in there. What's a healthy attrition rate?"},
        }
    },
    'dana_119': {
        'turns': {
            2: {'utterance': "Seeing repeated campaign_id + segment rows. What's the formula for engagement_rate?"},
        }
    },
    'dana_120': {
        'turns': {
            2: {'utterance': "BU_period has 'Retail-2024Q3' jammed together, split that. Also full stats on the table."},
        }
    },
    'dana_123': {
        'turns': {
            2: {'utterance': "CHI header says 'Chicago_IL' but the rest use codes. Fix it, then export as Excel."},
        }
    },
    'dana_124': {
        'turns': {
            2: {'utterance': "Impressions has gaps, fill from reach and engagement rate. Also drop story_views."},
        }
    },
    'dana_125': {
        'turns': {
            2: {'utterance': "Entries saying 'web app' should just be 'web'. Also pull in the subscription table by user_id."},
        }
    },
    'dana_127': {
        'turns': {
            2: {'utterance': "Revenue by region? Also the colors on that last chart need more contrast."},
        }
    },
}


# ── Apply fixes ──────────────────────────────────────────────────────

def apply_fixes(convos: list[dict], fixes: dict) -> int:
    """Apply all fixes to a list of conversations. Returns count of modified convos."""
    modified = 0
    convo_map = {c['convo_id']: c for c in convos}

    for cid, fix_spec in fixes.items():
        convo = convo_map.get(cid)
        if not convo:
            print(f"  WARNING: {cid} not found")
            continue

        changed = False
        turn_fixes = fix_spec.get('turns', {})
        for turn_idx, updates in turn_fixes.items():
            turn = convo['turns'][turn_idx]
            for key, val in updates.items():
                if key in ('flow', 'intent', 'candidate_flows', 'candidate_intents'):
                    turn[key] = val
                    changed = True
                elif key == 'utterance':
                    if turn.get('utterance') != val:
                        turn['utterance'] = val
                        changed = True

        # Update rationale if specified
        if 'rationale' in fix_spec:
            # Find the turn with rationale (usually T3)
            for turn in convo['turns']:
                if 'rationale' in turn:
                    turn['rationale'] = fix_spec['rationale']
                    changed = True

        if changed:
            modified += 1

    return modified


def fix_all_punctuation(convos: list[dict]) -> int:
    """Fix punctuation artifacts across all conversations. Returns count of fixes."""
    fixes = 0
    for convo in convos:
        for turn in convo.get('turns', []):
            old = turn.get('utterance', '')
            new = fix_punctuation_artifacts(old)
            if old != new:
                turn['utterance'] = new
                fixes += 1
    return fixes


def _write_compact(convos: list[dict], path: Path) -> None:
    """Write conversations in the original compact format."""
    lines = ['[']
    for ci, convo in enumerate(convos):
        # Conversation header
        hdr_parts = [f'"convo_id": {json.dumps(convo["convo_id"])}']
        hdr_parts.append(f'"category": {json.dumps(convo["category"])}')
        lines.append(f'  {{{", ".join(hdr_parts)},')
        lines.append(f'    "scenario": {json.dumps(convo["scenario"])},')
        lines.append(f'    "turns": [')

        for ti, turn in enumerate(convo['turns']):
            # Build turn metadata line (everything except utterance and rationale)
            meta = {}
            for k in ('turn_num', 'flow', 'intent', 'candidate_flows', 'candidate_intents', 'speaker'):
                if k in turn:
                    meta[k] = turn[k]
            meta_str = json.dumps(meta, ensure_ascii=False)[1:-1]  # strip { }

            is_last_turn = (ti == len(convo['turns']) - 1)
            turn_end = '' if is_last_turn else ','

            if 'rationale' in turn:
                lines.append(f'      {{{meta_str},')
                lines.append(f'        "utterance": {json.dumps(turn["utterance"], ensure_ascii=False)},')
                lines.append(f'        "rationale": {json.dumps(turn["rationale"], ensure_ascii=False)}')
                lines.append(f'      }}{turn_end}')
            else:
                lines.append(f'      {{{meta_str},')
                lines.append(f'        "utterance": {json.dumps(turn["utterance"], ensure_ascii=False)}')
                lines.append(f'      }}{turn_end}')

        is_last_convo = (ci == len(convos) - 1)
        convo_end = '' if is_last_convo else ','
        lines.append(f'    ]')
        lines.append(f'  }}{convo_end}')

    lines.append(']')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    # Fix Hugo
    hugo_path = EVAL_DIR / 'gen_hugo.json'
    with open(hugo_path) as f:
        hugo_convos = json.load(f)

    punct_fixes = fix_all_punctuation(hugo_convos)
    print(f"Hugo: {punct_fixes} punctuation artifacts fixed")

    convo_fixes = apply_fixes(hugo_convos, HUGO_FIXES)
    print(f"Hugo: {convo_fixes} conversations modified")

    _write_compact(hugo_convos, hugo_path)
    print(f"Hugo: saved to {hugo_path}")

    # Fix Dana
    dana_path = EVAL_DIR / 'gen_dana.json'
    with open(dana_path) as f:
        dana_convos = json.load(f)

    punct_fixes = fix_all_punctuation(dana_convos)
    print(f"Dana: {punct_fixes} punctuation artifacts fixed")

    convo_fixes = apply_fixes(dana_convos, DANA_FIXES)
    print(f"Dana: {convo_fixes} conversations modified")

    _write_compact(dana_convos, dana_path)
    print(f"Dana: saved to {dana_path}")


if __name__ == '__main__':
    main()
