---
name: build-campaign-plan
description: Design multi-day marketing campaigns with objectives, audiences, channels, timeline, and daily content angles
category: strategy
priority: 1
dependencies: [campaigns.py, brain.py, content_types.py, scheduler.py, guidelines.py]
---

# Build Campaign Plan

## Purpose
Generate a structured multi-day campaign plan that feeds directly into BrandMover's campaigns.py system. Covers objectives, target audience segments, channel strategy, daily content angles, and KPIs — all aligned to FOID brand voice.

## Trigger Conditions
- User requests a campaign, launch plan, or multi-day content push
- A new product, partnership, or event needs coordinated marketing
- Seasonal or narrative-driven campaign opportunity identified

## Inputs
- Campaign goal (awareness, engagement, conversion, community)
- Duration (default: 7 days)
- Key product/feature to promote
- Any hard dates or external events to anchor around
- Optional: budget constraints, partner handles

## Execution Steps
1. Load brand context via `guidelines.get_brand_context()` for voice and visual constraints
2. Query `content_types.ALL_CONTENT_TYPES` to map available formats to campaign phases
3. Define campaign arc: tease (days 1-2) → educate (days 3-5) → activate (days 6-7) → sustain (day 8+)
4. Assign one primary content angle per day with backup angle
5. Map each day to a content type from `content_types.py` and a posting slot from `schedule.json`
6. Define KPIs per phase (impressions for tease, replies for educate, clicks for activate)
7. Generate campaign JSON compatible with `campaigns.py` schema
8. Run `brand_check.check_brand_compliance()` on all copy drafts
9. Score plan coherence via `scoring.score_draft()` on sample outputs
10. Output final plan for operator review before activation

## Output Format
```json
{
  "campaign_name": "string",
  "objective": "string",
  "duration_days": 7,
  "phases": [
    {
      "phase": "tease",
      "days": [1, 2],
      "goal": "string",
      "kpi": "impressions > 10k"
    }
  ],
  "daily_plan": [
    {
      "day": 1,
      "date": "2026-03-18",
      "angle": "string",
      "content_type": "string",
      "post_time": "11:11",
      "copy_draft": "string",
      "image_prompt": "string or null"
    }
  ],
  "kpis": {}
}
```

## Quality Checks
- Every day has a unique angle (no repetition)
- Content types are varied across the campaign (no 3 days of same type in a row)
- All copy drafts pass brand voice check (lowercase, dry wit, declarative)
- Campaign arc has clear escalation and payoff
- Post times align with `schedule.json` slots

## Tools & Modules Used
- `guidelines.py` — brand context loading
- `campaigns.py` — campaign creation and management
- `content_types.py` — available content formats
- `scheduler.py` — time slot mapping
- `brain.py` — LLM generation for copy drafts
- `scoring.py` — draft quality scoring
- `brand_check.py` — brand compliance validation

## Edge Cases & Learnings
<!-- populated through use -->

## Examples

**Example: 9-day Fluent Network launch campaign**
- Objective: drive awareness of FOID's Fluent L2 deployment
- Phase 1 (days 1-3): tease with cryptic "something is moving" posts, loreboard screenshots with redacted details
- Phase 2 (days 4-6): educate with threads on what Fluent means for prayer rituals, meme curation speed, MiFOID minting
- Phase 3 (days 7-9): activate with live minting event, community challenge, partnership announcement
- Daily angles: mystery → speed reveal → prayer on L2 → loreboard upgrade → MiFOID teaser → builder spotlight → mint day → community ritual → recap thread
