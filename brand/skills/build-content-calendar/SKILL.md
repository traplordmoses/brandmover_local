---
name: build-content-calendar
description: Generate weekly/monthly content calendars balancing themes, types, and posting cadence
category: strategy
priority: 1
dependencies: [content_types.py, scheduler.py, auto_state.py, guidelines.py, generation_history.py]
---

# Build Content Calendar

## Purpose
Produce a structured content calendar that balances content types, themes, and posting frequency across a week or month. References existing schedule slots and avoids repeating recent content patterns.

## Trigger Conditions
- Start of a new week/month needing planned content
- Operator requests a content plan or posting schedule
- After a campaign ends and regular cadence needs restoring

## Inputs
- Time range (week or month, with start date)
- Posts per day (default: 2)
- Theme priorities (e.g., "heavy on Loreboard this week")
- Exclusions (topics to avoid, dates to skip)
- Optional: active campaign to weave around

## Execution Steps
1. Load `schedule.json` for available posting slots
2. Pull last 14 days from `generation_history.py` to avoid content fatigue
3. Load `content_types.ALL_CONTENT_TYPES` for format options
4. Build theme rotation: culture (2x/week), product (2x/week), community (1x/week), meme (1x/week), ritual (1x/week)
5. Assign content types ensuring variety (no same type on consecutive days)
6. Map each slot to a theme + content type + topic suggestion
7. Flag high-effort slots (threads, videos) for early production
8. Check against `auto_state.py` for rate limiting compliance
9. Output calendar grid with all assignments
10. Generate first drafts for next 3 days via `brain.py`

## Output Format
```
| Date       | Time  | Theme     | Type          | Topic                          | Status  |
|------------|-------|-----------|---------------|--------------------------------|---------|
| 2026-03-18 | 11:11 | ritual    | image_post    | monday morning prayer prompt   | draft   |
| 2026-03-18 | 18:00 | culture   | text_post     | loreboard curator spotlight    | planned |
```

## Quality Checks
- No content type appears more than 3 times per week
- Themes are distributed evenly (no 3 consecutive culture posts)
- High-effort content (threads, video) scheduled with 48h lead time
- Weekend posts are lighter/memier than weekday posts
- No topic overlap with last 14 days of generation history

## Tools & Modules Used
- `content_types.py` — format definitions
- `scheduler.py` — cron slot parsing
- `auto_state.py` — rate limit and dedup checks
- `generation_history.py` — recent content audit
- `guidelines.py` — brand context for theme alignment
- `brain.py` — draft generation for near-term slots

## Edge Cases & Learnings
<!-- populated through use -->

## Examples

**Example: FOID week of 2026-03-18**
- Monday: prayer prompt image (11:11), loreboard spotlight text (18:00)
- Tuesday: Fluent L2 thread (11:11), meme reaction post (20:00)
- Wednesday: MiFOID identity explainer image (11:11), community quote RT (18:00)
- Thursday: culture essay thread (11:11), Foid Mommy lore post (20:00)
- Friday: weekly recap image (11:11), weekend ritual teaser (18:00)
- Saturday: meme post (14:00)
- Sunday: quiet — one reflective text post (18:00)
