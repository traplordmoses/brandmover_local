---
name: crisis-playbook
description: Pre-written response templates for negative events — exploits, FUD, outages, with escalation paths
category: strategy
priority: 3
dependencies: [brain.py, guidelines.py, brand_check.py, publisher.py]
---

# Crisis Playbook

## Purpose
Generate pre-written response templates and escalation procedures for crisis scenarios common in crypto/web3. Ensures FOID responds quickly, honestly, and on-brand during high-pressure moments instead of going silent or sounding corporate.

## Trigger Conditions
- Operator requests crisis preparation templates
- Negative event detected (manual flag or community alert)
- Pre-launch risk assessment needed
- Quarterly playbook refresh

## Inputs
- Crisis type (exploit, FUD, outage, partner incident, community conflict, regulatory)
- Severity level (1-3: monitor, respond, escalate)
- Affected products/services
- Known facts vs unknowns
- Optional: specific rumors or claims to address

## Execution Steps
1. Load brand context via `guidelines.get_brand_context()` — voice must stay consistent even in crisis
2. Classify crisis type and severity
3. Generate initial holding statement (acknowledge, no speculation, timeline for update)
4. Generate detailed response template with fact placeholders
5. Generate follow-up update template (new information, next steps)
6. Generate resolution/post-mortem template
7. Define tone adjustments: drop irony, keep lowercase, increase earnestness
8. Map response timeline: first response within 30 min, update every 2 hours
9. Run all templates through `brand_check.py` — must sound like FOID, not a legal team
10. Package as playbook document with decision tree

## Output Format
```
## Crisis Playbook: [Type]

### Severity: [1-3]
### Response Timeline
- T+0-30min: Holding statement
- T+2h: First update
- T+6h: Detailed response
- T+24h: Post-mortem

### Holding Statement
"we're aware of [issue]. looking into it now. will update within [timeframe]."

### Detailed Response Template
"here's what happened: [facts]. here's what we're doing: [actions]. here's what's next: [timeline]."

### Resolution Template
"[issue] is resolved. here's the full picture: [summary]. here's what we changed: [fixes]."

### Tone Guidelines
- Drop all irony and humor
- Stay lowercase (still FOID)
- Be direct and honest about unknowns
- Never blame community members
- Never speculate on causes before investigation

### Escalation Path
1. Marketing (this bot) posts holding statement
2. Core team reviews within 1 hour
3. Technical post-mortem drafted by engineering
4. Marketing translates post-mortem to community language
```

## Quality Checks
- Holding statement is postable within 5 minutes (no dependencies)
- Templates have clear [placeholder] markers for facts
- Tone is honest and direct, never defensive or dismissive
- No template promises specific outcomes or timelines that can't be met
- All templates pass brand check (still sounds like FOID, not a corporate PR firm)

## Tools & Modules Used
- `brain.py` — template generation with crisis-aware prompting
- `guidelines.py` — brand voice (adjusted for crisis tone)
- `brand_check.py` — compliance even under crisis constraints
- `publisher.py` — rapid posting capability

## Edge Cases & Learnings
<!-- populated through use -->

## Examples

**Example: smart contract vulnerability reported**
- Holding: "we're aware of a reported vulnerability in the MiFOID contract. team is investigating. funds are safe. update coming within 2 hours."
- Update: "confirmed: the issue affects [specific function]. no funds at risk. we've paused minting while we patch. fix deploying within 24h."
- Resolution: "MiFOID minting is back. here's what happened, what we fixed, and what we're doing to prevent it. full post-mortem: [link]"

**Example: FUD tweet goes viral**
- Assessment: is the claim factual, partially true, or false?
- If false: "saw this going around. here's the actual situation: [facts]. we get why it looked that way — [context]."
- If partially true: "this is partly right. here's the full picture: [facts + context]. we should have communicated this better."
