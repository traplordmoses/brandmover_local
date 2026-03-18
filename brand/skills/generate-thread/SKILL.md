---
name: generate-thread
description: Create multi-post X threads with hooks, pacing, education, and CTAs
category: content_creation
priority: 1
dependencies: [brain.py, unified_brain.py, guidelines.py, publisher.py, brand_check.py, scoring.py]
---

# Generate Thread

## Purpose
Produce a complete X/Twitter thread — from hook to CTA — with proper pacing, narrative arc, and FOID brand voice. Each post respects the 280-char limit. The thread is structured for maximum engagement and readability.

## Trigger Conditions
- Topic requires more depth than a single post
- Explainers, announcements, or storytelling content
- Operator explicitly requests a thread
- Campaign plan calls for a thread-type content piece

## Inputs
- Topic or angle (e.g., "why meme curation matters")
- Thread length (default: 5-7 posts)
- Tone modifier (default: standard FOID voice)
- Optional: key facts, links, or images to include
- Optional: target audience segment

## Execution Steps
1. Load brand context via `guidelines.get_brand_context()`
2. Generate thread outline via `brain.py`: hook, 3-5 body posts, CTA
3. Write hook post first — must be under 280 chars, no links, pure intrigue
4. Write body posts with one idea per post, building on previous
5. Add a "turn" at post 3-4 (surprise, counterpoint, or reveal)
6. Write CTA as final post — clear action, link if needed
7. Add thread numbering (1/, 2/, etc.) only if thread exceeds 5 posts
8. Run each post through `brand_check.check_brand_compliance()`
9. Score full thread via `scoring.score_draft()` for coherence
10. Format as numbered array for operator review

## Output Format
```json
{
  "thread_topic": "string",
  "post_count": 6,
  "posts": [
    {"position": 1, "type": "hook", "text": "string", "char_count": 142},
    {"position": 2, "type": "body", "text": "string", "char_count": 267},
    {"position": 3, "type": "body", "text": "string", "char_count": 251},
    {"position": 4, "type": "turn", "text": "string", "char_count": 220},
    {"position": 5, "type": "body", "text": "string", "char_count": 274},
    {"position": 6, "type": "cta", "text": "string", "char_count": 189}
  ]
}
```

## Quality Checks
- Hook post is under 280 chars and contains no links or hashtags
- Every post is independently readable (no orphan references)
- No post exceeds 280 characters
- Thread has a clear narrative arc (setup → tension → resolution)
- CTA post has exactly one clear action
- All posts pass brand voice check (lowercase, declarative, no exclamation marks)
- Thread reads naturally when posts are viewed individually in timeline

## Tools & Modules Used
- `brain.py` / `unified_brain.py` — LLM generation
- `guidelines.py` — brand voice context
- `brand_check.py` — compliance per post
- `scoring.py` — thread quality scoring
- `publisher.py` — thread posting to X (reply chain)

## Edge Cases & Learnings
<!-- populated through use -->

## Examples

**Example: "why we built loreboard" thread (6 posts)**
1. "most meme platforms show you what's trending. we wanted to show you what's worth remembering." (hook)
2. "loreboard is a curation grid. not an algorithm. humans pick what goes on it. that's the whole point." (setup)
3. "the internet used to have zines, mixtapes, forum threads with actual taste. we lost that to engagement metrics." (context)
4. "but here's the thing — curation is identity. what you choose to surface says more than what you create." (turn)
5. "every loreboard grid is a statement. a cultural snapshot. proof that someone was paying attention." (build)
6. "curate your first grid at foid.fun/loreboard" (CTA)
