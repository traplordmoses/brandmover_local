---
name: analyze-narrative
description: Read market/ecosystem conditions and recommend current narrative angles for content
category: strategy
priority: 2
dependencies: [web_fetch.py, brain.py, guidelines.py, generation_history.py, brand_check.py]
---

# Analyze Narrative

## Purpose
Assess the current crypto/web3 narrative landscape and recommend which angles FOID should lean into, avoid, or counter. Produces a narrative brief that guides content creation for the coming days or weeks.

## Trigger Conditions
- Start of week content planning session
- Market conditions shift significantly (pump, crash, regulatory news)
- New ecosystem narrative emerges (L2 season, meme coin wave, identity discourse)
- Before building a campaign plan or content calendar

## Inputs
- Current date and recent market context
- FOID's recent content themes (from generation_history)
- Optional: specific events or trends to assess
- Optional: narratives to explicitly evaluate

## Execution Steps
1. Fetch current crypto/web3 discourse via `web_fetch.py` (CT timeline, ecosystem news)
2. Pull FOID's last 14 days of content from `generation_history.py`
3. Load brand positioning from `guidelines.get_brand_context()`
4. Identify 3-5 active narratives in the ecosystem
5. Score each narrative on: relevance to FOID, saturation level, brand alignment
6. Classify: lean in (high relevance, low saturation), monitor (medium), avoid (off-brand or oversaturated)
7. Generate specific content angles for "lean in" narratives
8. Identify counter-narrative opportunities (what everyone is saying that FOID can push back on)
9. Check recommendations against `brand_check.py` for voice alignment
10. Output narrative brief with concrete examples

## Output Format
```
## Narrative Brief — Week of [Date]

### Active Narratives
1. **[Narrative]** — Lean In
   - Relevance: high | Saturation: low | Brand fit: strong
   - Angle: [specific content angle]
   - Example post: "[draft]"

2. **[Narrative]** — Monitor
   - Relevance: medium | Saturation: high | Brand fit: partial
   - Note: [why to watch but not chase]

3. **[Narrative]** — Avoid
   - Relevance: low | Saturation: extreme | Brand fit: off
   - Risk: [what happens if FOID engages]

### Counter-Narrative Opportunity
- Everyone is saying [X]. FOID can say [Y] because [reason].

### Recommended Content Angles (next 7 days)
1. [angle + content type + draft hook]
2. [angle + content type + draft hook]
3. [angle + content type + draft hook]
```

## Quality Checks
- Narratives are sourced from actual current discourse (not generic)
- Recommendations align with FOID's brand positioning (culture, ritual, identity)
- "Avoid" classifications have clear reasoning
- Content angles are specific enough to immediately draft from
- Counter-narrative suggestions are contrarian but authentic, not edgy for shock value

## Tools & Modules Used
- `web_fetch.py` — current discourse research
- `brain.py` — narrative analysis and recommendation
- `guidelines.py` — brand positioning context
- `generation_history.py` — recent content audit (avoid redundancy)
- `brand_check.py` — angle alignment validation

## Edge Cases & Learnings
<!-- populated through use -->

## Examples

**Example: narrative brief during "L2 season"**
- Lean In: "L2 as cultural infrastructure" — FOID is on Fluent, can own the "L2 isn't just for speed, it's for culture" angle. Low saturation, most L2 talk is technical.
  - Example post: "everyone's comparing L2 tps. we're comparing L2 vibes. fluent is where the rituals live."
- Monitor: "meme coin supercycle" — relevant but oversaturated. Don't chase the hype, but acknowledge the energy.
- Avoid: "ETH vs SOL tribalism" — off-brand, no upside, FOID is on Fluent and doesn't pick sides.
- Counter-narrative: "everyone is talking about what's fast. nobody is talking about what's meaningful. that's FOID's lane."
