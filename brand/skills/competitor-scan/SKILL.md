---
name: competitor-scan
description: Research competitor social presence, positioning, and content strategy with actionable takeaways
category: strategy
priority: 2
dependencies: [web_fetch.py, brain.py, guidelines.py, report_generator.py]
---

# Competitor Scan

## Purpose
Analyze competitor or adjacent project social media presence, content strategy, and positioning. Produces a structured comparison with gaps, opportunities, and specific content recommendations for FOID to differentiate.

## Trigger Conditions
- New competitor enters the meme/culture/web3 space
- Pre-campaign research to identify whitespace
- Quarterly competitive landscape review
- Operator asks "what is [project] doing on socials?"

## Inputs
- Competitor name(s) and X handle(s)
- Competitor website URL(s)
- Focus areas (content style, posting frequency, audience, narrative)
- Optional: specific questions to answer

## Execution Steps
1. Fetch competitor X profile and recent posts via `web_fetch.py`
2. Fetch competitor website landing page via `web_fetch.py`
3. Analyze posting frequency, content types, tone, engagement patterns
4. Identify their narrative positioning (what story are they telling?)
5. Map their content type distribution (memes vs threads vs announcements)
6. Compare against FOID's brand context from `guidelines.py`
7. Identify gaps: what they do that FOID doesn't, and vice versa
8. Identify whitespace: what neither is doing that FOID could own
9. Generate 5 specific content recommendations based on findings
10. Build comparison report via `report_generator.py`

## Output Format
```
## Competitor Scan: [Name] vs FOID

### Profile
- Handle: @competitor
- Followers: ~XX,XXX
- Posting cadence: X posts/day
- Primary content type: threads/memes/announcements

### Positioning
[2-3 sentences on their narrative]

### Content Breakdown
| Type        | Competitor | FOID |
|-------------|-----------|------|
| Memes       | 40%       | 25%  |
| Threads     | 20%       | 15%  |
| Announcements| 30%      | 20%  |

### Gaps & Opportunities
1. [specific gap]
2. [specific gap]
3. [whitespace opportunity]

### Recommendations
1. [actionable content suggestion]
2. [actionable content suggestion]
```

## Quality Checks
- All data sourced from actual web fetches (no assumptions)
- Recommendations are specific to FOID's brand and capabilities
- Tone is analytical, not dismissive of competitors
- Opportunities are realistic given FOID's current resources
- Report avoids speculation on competitor metrics we cannot verify

## Tools & Modules Used
- `web_fetch.py` — fetch competitor profiles and websites
- `brain.py` — analysis and recommendation generation
- `guidelines.py` — FOID brand context for comparison
- `report_generator.py` — formatted report output

## Edge Cases & Learnings
<!-- populated through use -->

## Examples

**Example: scan of a meme-culture L2 competitor**
- Finding: competitor posts 4x/day but 80% are retweets with no original voice
- Finding: no ritual/spiritual narrative in their content (FOID whitespace)
- Finding: they dominate meme reaction content but have zero long-form threads
- Recommendation: "lean harder into the prayer ritual narrative — nobody in this space owns spiritual meme culture. publish a weekly 'proof of prayer' thread to claim that territory."
- Recommendation: "their meme volume is high but generic. FOID's curated loreboard angle is a differentiator — post loreboard grids as counter-programming."
