---
name: launch-checklist
description: Generate pre/during/post launch marketing checklists with countdown content assignments
category: strategy
priority: 3
dependencies: [campaigns.py, content_types.py, scheduler.py, brain.py, guidelines.py, publisher.py]
---

# Launch Checklist

## Purpose
Produce a comprehensive launch marketing checklist covering pre-launch buildup, launch day execution, and post-launch sustain. Each item has a date, owner, content assignment, and status. Designed for product drops, partnerships, and events.

## Trigger Conditions
- New product or feature launching (Loreboard update, MiFOID drop, new ritual)
- Partnership announcement with a fixed date
- Event or AMA with coordinated marketing
- Operator says "we're launching X on [date]"

## Inputs
- What is launching (product, feature, event)
- Launch date and time
- Key messaging points (3-5 bullet points)
- Available channels (X, Telegram, Discord, website)
- Team members / roles (who approves, who designs)
- Optional: partner handles to coordinate with

## Execution Steps
1. Set launch date as T-0 and work backward/forward
2. Build pre-launch checklist (T-7 to T-1): tease content, asset preparation, copy drafts
3. Build launch day checklist (T-0): coordinated posting sequence, real-time monitoring
4. Build post-launch checklist (T+1 to T+7): recap, community response, performance review
5. Assign content types from `content_types.py` to each checklist item
6. Map posting times to `scheduler.py` slots
7. Generate draft copy for T-3 through T+1 via `brain.py`
8. Flag dependencies (e.g., "partner must approve before T-2 post")
9. Create campaign entry in `campaigns.py` format
10. Output checklist for operator review

## Output Format
```
## Launch Checklist: [Product] — [Date]

### Pre-Launch (T-7 to T-1)
- [ ] T-7: Cryptic teaser post (image_post) — 11:11
- [ ] T-5: Behind-the-scenes snippet (text_post) — 18:00
- [ ] T-3: "something is coming" thread (thread) — 11:11
- [ ] T-2: Partner co-announcement draft — needs approval
- [ ] T-1: Countdown post with preview image — 20:00

### Launch Day (T-0)
- [ ] 09:00: Final asset check — all images generated and approved
- [ ] 11:11: Main announcement post with image
- [ ] 11:12: Reply thread with details (3 posts)
- [ ] 14:00: Community engagement — reply to responses
- [ ] 18:00: Recap/highlight post
- [ ] 20:00: "day one" reflection post

### Post-Launch (T+1 to T+7)
- [ ] T+1: Community highlight / quote retweets
- [ ] T+3: "what we learned" thread
- [ ] T+7: Performance digest via weekly-digest skill
```

## Quality Checks
- Every checklist item has a specific date, time, and content type
- Dependencies are clearly marked with blockers
- Launch day has no gaps longer than 4 hours
- Pre-launch teases escalate in specificity (vague → concrete)
- Post-launch content references actual launch reception, not pre-written

## Tools & Modules Used
- `campaigns.py` — campaign structure creation
- `content_types.py` — content format assignments
- `scheduler.py` — time slot mapping
- `brain.py` — copy draft generation
- `guidelines.py` — brand voice alignment
- `publisher.py` — posting execution

## Edge Cases & Learnings
<!-- populated through use -->

## Examples

**Example: MiFOID identity NFT mint launch**
- T-7: "who are you on-chain?" — abstract identity teaser image
- T-5: Loreboard grid showing MiFOID preview assets (blurred)
- T-3: Thread on identity, memes, and what MiFOID represents
- T-1: "tomorrow you get to answer" — single declarative post
- T-0 11:11: "MiFOID is live. mint your identity at foid.fun/mifoid"
- T-0 11:12: Reply thread — how it works, what's in it, why it matters
- T+1: Community spotlight — first minters, creative uses
- T+3: "the first 48 hours" recap thread with stats
