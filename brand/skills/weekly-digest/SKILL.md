---
name: weekly-digest
description: Auto-summarize weekly performance — posts published, engagement, approval rates, top content
category: analytics
priority: 2
dependencies: [generation_history.py, feedback.py, auto_state.py, report_generator.py, guidelines.py]
---

# Weekly Digest

## Purpose
Generate a branded weekly performance report summarizing what was published, what performed well, approval/rejection patterns, and recommendations for the coming week. Delivered as an HTML report via `report_generator.py`.

## Trigger Conditions
- End of week (Sunday evening or Monday morning)
- Operator requests a performance summary
- Before planning next week's content calendar

## Inputs
- Date range (default: last 7 days)
- Optional: specific metrics to highlight
- Optional: comparison period (previous week)

## Execution Steps
1. Pull all entries from `generation_history.py` for the date range
2. Pull approval/rejection data from `feedback.py` for the same period
3. Query `auto_state.py` for auto-post success/failure counts
4. Calculate metrics: total posts, approval rate, content type distribution, avg generation cost
5. Identify top 3 posts by approval speed (fastest approved = highest confidence)
6. Identify rejection patterns (common feedback themes via `feedback.get_learned_preferences()`)
7. Compare against previous period if available
8. Generate narrative summary via `brain.py` in FOID voice
9. Build HTML report via `report_generator.py`
10. Send report to operator with key takeaways

## Output Format
```
## FOID Weekly Digest — Mar 11-17, 2026

### Numbers
- Posts published: 12
- Approval rate: 83%
- Content types: 4 image, 3 text, 2 thread, 2 meme, 1 video
- Avg drafts per approval: 1.4
- Total generation cost: $2.18

### Top Content
1. "loreboard update: 47 new grids this week" — approved in 12s
2. Monday prayer prompt — highest engagement slot
3. Fluent L2 thread — 6 posts, zero rejections

### Patterns
- Meme posts had 100% first-draft approval
- Thread hooks needed rework 40% of the time
- Evening posts (20:00) had slower approval than morning (11:11)

### Recommendations
- Increase meme frequency (high approval, low effort)
- Pre-write thread hooks for faster iteration
- Test a 14:00 weekend slot
```

## Quality Checks
- All metrics are sourced from actual logs (no estimates)
- Approval rate calculation excludes pending drafts
- Cost tracking includes all LLM + image generation costs
- Recommendations are specific and actionable (not generic)
- Report renders correctly in HTML via report_generator

## Tools & Modules Used
- `generation_history.py` — post log with timestamps and costs
- `feedback.py` — approval/rejection data and learned preferences
- `auto_state.py` — auto-post scheduling outcomes
- `report_generator.py` — HTML report rendering
- `brain.py` — narrative summary generation
- `guidelines.py` — brand context for report voice

## Edge Cases & Learnings
<!-- populated through use -->

## Examples

**Example: FOID Week 12 digest highlights**
- 14 posts published across 5 content types
- 86% approval rate, up from 71% previous week
- Prayer prompt posts: 100% first-draft approval (brand voice locked in)
- Threads: hook rewrites dropped from 60% to 40% after feedback loop adjustment
- Recommendation: "the prayer slot at 11:11 is the strongest performer. consider a daily ritual series."
