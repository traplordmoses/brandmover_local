---
name: grant-application
description: Write ecosystem grant applications — project description, milestones, budget, impact
category: crypto_web3
priority: 3
dependencies: [brain.py, unified_brain.py, guidelines.py, web_fetch.py, brand_check.py]
---

# Grant Application

## Purpose
Draft complete ecosystem grant applications for FOID Foundation. Covers project overview, problem statement, proposed solution, milestones, budget, team, and expected impact. Tailored to the specific grant program's requirements and evaluation criteria.

## Trigger Conditions
- Grant program opens for a relevant ecosystem (Fluent, Ethereum Foundation, etc.)
- Operator identifies a grant opportunity
- New product/feature needs funding and has a grant-eligible scope

## Inputs
- Grant program name and URL
- Grant amount range
- Evaluation criteria (if published)
- Project to fund (which FOID product/initiative)
- Timeline constraints
- Team members to list
- Optional: previous grant history

## Execution Steps
1. Fetch grant program details via `web_fetch.py` (requirements, criteria, past winners)
2. Load FOID brand context and product details from `guidelines.get_brand_context()`
3. Draft project overview — what FOID is, what the grant funds, why it matters
4. Write problem statement — what gap exists in the ecosystem
5. Write proposed solution — how FOID addresses it, with specifics
6. Define milestones (3-5) with deliverables, dates, and verification methods
7. Build budget breakdown tied to milestones
8. Write impact section — quantifiable outcomes and ecosystem benefit
9. Adjust tone: more formal than social content, but still recognizably FOID
10. Run through `brand_check.py` with formal-mode flag

## Output Format
```
## Grant Application: [Program Name]

### Project Title
[Concise title]

### Project Overview
[2-3 paragraphs: what, why, how]

### Problem Statement
[What gap or need this addresses]

### Proposed Solution
[Specific deliverables and approach]

### Milestones
| # | Milestone | Deliverable | Date | Budget |
|---|-----------|------------|------|--------|
| 1 | [name]    | [output]   | [date]| $X,XXX |

### Budget Breakdown
| Category | Amount | Justification |
|----------|--------|--------------|
| Development | $XX,XXX | [specifics] |

### Team
[Names, roles, relevant experience]

### Expected Impact
[Quantifiable outcomes, ecosystem benefit]

### Previous Work
[Links to existing products, traction metrics]
```

## Quality Checks
- Application addresses all stated evaluation criteria
- Milestones are specific and verifiable (not vague)
- Budget is realistic and justified (no padding)
- Tone is professional but retains FOID identity
- Impact metrics are quantifiable, not aspirational fluff
- All claims about FOID products are accurate and verifiable

## Tools & Modules Used
- `brain.py` / `unified_brain.py` — application drafting
- `guidelines.py` — brand context and product details
- `web_fetch.py` — grant program research
- `brand_check.py` — voice compliance (formal mode)

## Edge Cases & Learnings
<!-- populated through use -->

## Examples

**Example: Fluent Ecosystem Grant for Loreboard**
- Title: "Loreboard: Cultural Curation Infrastructure on Fluent"
- Problem: "L2 ecosystems have DeFi dashboards and NFT marketplaces but lack cultural infrastructure — tools for communities to curate, preserve, and celebrate their meme culture on-chain."
- Solution: Loreboard v2 with on-chain curation proofs, community voting, and cross-L2 meme indexing
- Milestones: (1) on-chain curation contracts, (2) community voting module, (3) cross-chain indexer, (4) public launch + 500 curated grids
- Budget: $40k — $25k dev, $8k infra, $4k community incentives, $3k audit. Impact: 500+ grids, 2k+ curators
