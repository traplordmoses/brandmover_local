# BrandMover Architectural Analysis

## The Core Problem

The bot has **two entirely disconnected brains** that never talk to each other:

| | Casual Chat Path | Content Generation Path |
|---|---|---|
| **Identity** | `brandmover_uwu.exe` — glitchy, warm, opinionated | `"BrandMover, an autonomous AI marketing agent"` — nameless workhorse |
| **Personality** | Full character from `brand/personality/system_prompt.md` | None |
| **Memory** | `brand/personality/memory.md` (knows operator is Moses, knows his preferences) | None |
| **Conversation history** | Up to 20 turns preserved | Starts from scratch every time |
| **Brand voice** | Voice & tone sections from guidelines | Full guidelines via tool call or parameter |
| **Model** | Sonnet | Sonnet (pipeline) or configured agent model |
| **Triggered by** | Intent = `casual_chat` or `greeting` | Intent = `generate_content` or fallback |

The intent router is the **hard wall** between these two brains. The moment it classifies a message as `generate_content`, all personality, memory, and conversation context is dropped. The user goes from talking to a character to feeding a pipeline.

---

## 1. Message Flow (End to End)

```
User sends text message in Telegram
    │
    ▼
handle_message() [handlers.py:1822]
    │
    ├── Auth check (_can_operate)
    ├── Template region update intercept
    ├── Onboarding intercept
    │
    ▼
Intent Router [handlers.py:1872]
    │
    ├── Short message table lookup (117 entries, 0 cost)
    │   "yes"/"ok"/"ship it" → approve
    │   "hi"/"hey"/"gm" → greeting
    │   "try again"/"redo" → reroll
    │
    ├── If not in table → Claude Haiku call (5s timeout, 256 tokens)
    │   System prompt lists 17 intents with descriptions
    │   Receives: message + last_bot_action + pending_draft + recent_intents
    │
    ▼
_route_intent() dispatches [handlers.py:1901]
    │
    ├── greeting (any confidence)
    │   └── chat.handle_greeting() → Sonnet + personality + memory
    │
    ├── casual_chat (≥0.5)
    │   └── chat.handle_casual_chat() → Sonnet + personality + memory + history
    │
    ├── approve/reject/reroll (≥0.8)
    │   └── Direct action on pending draft
    │
    ├── modify_last (≥0.5)
    │   └── chat.handle_modify_last() → Sonnet (NO personality)
    │
    ├── generate_content / unknown / low confidence
    │   └── Falls through to generation
    │
    ▼
Generation fallback [handlers.py:1881]
    │
    ├── Rate limit check (10s cooldown)
    ├── Pending draft check (blocks if exists)
    │
    ▼
    ├── AGENT_MODE=agent → _handle_agent_mode()
    │   └── engine.run_agent() [15 max turns, 8 tools]
    │       System prompt: skill_prompt.build_system_prompt()
    │       NO personality. NO memory. NO conversation history.
    │       First turn forced to call a tool.
    │       Tools: read_brand_guidelines, generate_image, img2img,
    │              read_feedback_history, check_figma_design,
    │              read_references, log_resource_usage, execute_openclaw_script
    │
    └── AGENT_MODE=pipeline → _handle_pipeline_mode()
        └── brain.pipeline_generate() [4 steps]
            Step 1: Analyze (600 tokens)
            Step 2: Plan (800 tokens)
            Step 3: Verify (500 tokens)
            Step 4: Generate (1500 tokens)
            NO personality. NO memory. NO conversation history.
            Brand context passed as raw string parameter.
            Then: image_gen.generate_image() called separately.
    │
    ▼
_send_draft() [handlers.py:2988]
    │
    ├── Synthesize title/subtitle from caption if missing
    ├── _maybe_compose() → template > compositor > raw image
    ├── _prepare_photo() → convert local path to BytesIO
    ├── Send photo with caption + random review prompt
    │   "how does this look?" / "what do you think?" / etc.
    ├── Inline buttons: [Approve] [Reject] [Edit] [Reroll]
    │
    ▼
User taps button or sends text
    │
    ├── Approve → _do_approve()
    │   Post to X/Twitter, log feedback, save to library,
    │   add to LoRA training, clear pending
    │
    ├── Reject → prompts for feedback → _do_reject()
    │   Log feedback, clear pending, re-run generation with
    │   revision_context prepended
    │
    ├── Edit → prompts for feedback → _do_reject() (same path)
    │
    └── Reroll → clear pending, re-run generation with
        original_request (fresh start, no revision context)
```

---

## 2. The Personality System

### What exists

**Character definition** (`brand/personality/system_prompt.md`):
- Name: `brandmover_uwu.exe`
- Identity: "An AI brand agent who takes her job very seriously but has a glitch — when she cares too much, her enthusiasm overflows"
- Uses `[MEMORY LEAK]`, `[BUFFER OVERFLOW]`, `[PROCESSING]` as emotional punctuation
- Starts professional, trails into softness
- Never says: "I'd be happy to help!", "Great question!", "As an AI..."
- Addresses operator by name or "operator"

**Memory** (`brand/personality/memory.md`):
- Knows operator is Moses
- Moses prefers reasoning/thinking before generating
- Moses values personality and conversational depth over raw speed
- Sparse — mostly placeholder sections

**Chat system** (`agent/chat.py`):
- `_build_chat_system_prompt()` assembles: base identity + personality + brand voice + memory + behavioral rules
- `handle_casual_chat()` uses Sonnet with full conversation history (up to 20 turns)
- `handle_greeting()` delegates to `handle_casual_chat()` if personality file exists

### Where it's used

Only in the casual chat path:
- Intent = `greeting` → `chat.handle_greeting()` → personality-aware
- Intent = `casual_chat` → `chat.handle_casual_chat()` → personality-aware

### Where it's NOT used

Everything else:
- **Agent mode generation** (`engine.run_agent`) — system prompt from `skill_prompt.py`, zero personality
- **Pipeline mode generation** (`brain.pipeline_generate`) — clinical step-by-step prompts, zero personality
- **Draft revision** (`_handle_agent_revision`, `_handle_pipeline_revision`) — zero personality
- **Draft modification** (`chat.handle_modify_last`) — bare-bones "content editor" prompt despite being in `chat.py`
- **Auto-post** (`scripts/auto_post.py`) — calls `engine.run_agent()` directly, zero personality
- **All bot status messages** — hardcoded English strings
- **Review prompts** — hardcoded list, not personality-aware
- **Error messages** — hardcoded
- **Draft callback prompts** ("What should I change?") — hardcoded

### The personality lobotomy

A user who has a warm, character-driven conversation and then says "make me a post about our launch" will experience an instant identity switch. The personality they were talking to vanishes. The generation pipeline has no idea that conversation happened. It starts from scratch with a blank `messages` array containing only their latest text.

Memory says Moses "prefers the bot to reason and think with him before generating." But the agent mode system prompt says "Be concise in your reasoning. Keep tool calls purposeful, not chatty." These directly contradict each other — and the agent never sees the memory file.

---

## 3. Brand Configuration

### BrandConfig (`agent/compositor_config.py`)

Parsed from `brand/guidelines.md` into a dataclass:
- **Identity**: brand_name, tagline, website, x_handle
- **Colors**: Dict of role → (hex, rgb) for primary, accent_1-3, background, text, etc.
- **Fonts**: Display/body/terminal font families with download URLs
- **Visual**: canvas size, layout profiles, glass-morphism effects, orb specs
- **Content**: style keywords, avoid terms, product description, voice traits
- **Controls**: compositor_enabled, badge_text, default_mode

Cached by MD5 hash of guidelines.md content. Overridden by `brand/config.json` for compositor settings.

### Brand context for LLM (`agent/guidelines.py`)

Assembled from three sources:
1. `brand/guidelines.md` (full text)
2. `brand/examples/articles/*.txt` (example posts)
3. `brand/references/` (PDFs, markdown, text files — newest first, capped at 50K chars)

Output format:
```
Brand Name: FOID Foundation
--- BRAND GUIDELINES ---
{guidelines.md}
--- EXAMPLE POSTS ---
{examples}
--- REFERENCE MATERIALS ---
{reference files}
```

Mtime-cached with 30-second check interval.

### How brand context reaches the LLM

- **Agent mode**: The agent must call `read_brand_guidelines` tool (forced on turn 0 via `tool_choice: any`). Returns the full assembled string.
- **Pipeline mode**: `guidelines.get_brand_context()` is called by the handler and passed as a parameter to `brain.pipeline_generate()`. Truncated to 4000 chars for intermediate steps, full for final generation.
- **Casual chat**: Only voice & tone sections are extracted from guidelines.md and injected into the chat system prompt.

---

## 4. State Management

### Per-user state

| File | Contents | Scope |
|---|---|---|
| `state/state.json` | Admin's pending draft, draft history (last 20), last generated image, last composed path, reference image path | Admin user |
| `state/draft_{uid}.json` | Same structure per operator | Per operator |
| `state/conversation.json` | Per-user: conversation history, recent intents, last bot action, user name, timestamps | All users |
| `state/feedback.json` | Append-only log of all approvals/rejections with feedback text, 134KB currently | Global |
| `state/learned_preferences.md` | Claude-summarized patterns from feedback (3.4KB) | Global |
| `state/generation_history.json` | Last 500 generations with costs, model IDs, statuses (10.6KB) | Global |
| `state/auto_post_state.json` | Scheduler state: posts today, rotation indices, recent captions, paused flag | Global |

### What the agent knows about past work

When the agent calls `read_feedback_history`:
1. The full `learned_preferences.md` content (patterns, dos/don'ts, common rejections)
2. Last 10 feedback entries: `[APPROVED/REJECTED] Request: ... | Caption: ... -- Feedback: ...`

What it does NOT know:
- The full feedback history (only last 10)
- Generation history (costs, models used, approval rates)
- Draft version history
- What it generated in the last conversation
- What the user said before the generation request
- Style profile history

### Draft lifecycle

```
save_pending() → state.json
    │
    ├── approve → post to X, log feedback, archive to references,
    │              add to library, add to LoRA training set,
    │              clear pending
    │
    ├── reject → log feedback, clear pending, re-generate with
    │             revision context
    │
    ├── edit → image manipulation via compositor, re-save pending
    │
    └── reroll → clear pending + draft history, re-generate fresh
```

---

## 5. Scheduler and Automation

### Schedule config (`config/schedule.json`)

4 time slots:
- `engagement_morning` (09:00 UTC ±30min) — community questions
- `onchain_midday` (12:00 UTC ±30min) — Loreboard activity
- `onchain_afternoon` (17:00 UTC ±30min) — second on-chain review
- `brand_meme` (21:00 UTC ±45min) — branded image + caption

Global: 120min minimum gap, 6 posts/day max.

### Auto-post flow

`scripts/auto_post.py` runs as a background asyncio task launched at bot startup:
1. Loops every 5 minutes
2. Checks each slot against current time + jitter
3. Calls `engine.run_agent()` directly (no personality, no conversation context)
4. Sends draft to Telegram via `handlers.send_auto_draft()`
5. User must still `/approve` to actually post

### User-facing

- `/autostatus` — shows slot status, post counts, paused state
- `/autopause` / `/autoresume` — toggle scheduler
- `/autoforce [slot]` — trigger immediate generation for a slot
- `/schedule` / `/unschedule` — add/remove time slots

The scheduler is **purely background infrastructure**. It doesn't proactively tell the user "I'm about to generate your evening post" or "it's been quiet today, want me to draft something?"

### Rate limiting

- `auto_state.can_post()`: checks paused flag, daily limit (6), minimum gap (120min)
- `auto_state.is_duplicate_caption()`: Jaccard similarity > 0.6 against last 20 captions
- Rotation indices per content category for prompt variety

---

## 6. Module Dependency Map

```
bot/telegram_bot.py ─── registers handlers ──→ bot/handlers.py
                                                    │
                    ┌───────────────────────────────┤
                    │                               │
              Intent Router                    Generation
        agent/intent_router.py            ┌────────┴────────┐
           │        │                     │                  │
      casual_chat  generate          Agent Mode         Pipeline Mode
           │                     agent/engine.py      agent/brain.py
           ▼                          │                     │
     agent/chat.py              agent/tools.py              │
           │                    agent/skill_prompt.py        │
    ┌──────┴──────┐                   │                     │
    │             │              ┌────┴────┐                │
personality/   memory/      agent/        agent/            │
system_prompt  memory.md    image_gen.py  guidelines.py     │
                            agent/        agent/            │
                            compositor.py feedback.py       │
                                                            │
                            agent/state.py ◄────────────────┘
                            agent/auto_state.py
                            agent/generation_history.py
                            agent/publisher.py (X/Twitter)

Brand Data:
    brand/guidelines.md ──→ compositor_config.py (BrandConfig dataclass)
                       ──→ guidelines.py (raw text for LLM context)
    brand/personality/  ──→ chat.py ONLY (never reaches generation)
    brand/config.json   ──→ compositor_config.py (overrides)
    brand/styles.json   ──→ state.py (style profiles)
    brand/references/   ──→ guidelines.py (reference materials)
    brand/examples/     ──→ guidelines.py (example posts)
    brand/prompts/      ──→ tools.py (custom image prompt templates)
```

---

## 7. Why It Feels Robotic

### The pipeline is a one-way conveyor belt

```
User request → [classify] → [generate] → [show draft] → [approve/reject]
                                              ↑
                                     No reasoning shown
                                     No creative choices explained
                                     No opinion offered
                                     Same 4 buttons every time
```

The user has no visibility into *why* the bot made the choices it did. There's no "I went with a punchy tone since your engagement posts have been doing well short" or "I used the campaign template because this feels like a launch moment." Just: here's a draft, yes or no?

### Every interaction follows the same script

1. User says something
2. Bot classifies intent
3. Bot either chats (personality) or generates (no personality)
4. If generated: same 4 buttons, same random review prompt
5. User taps a button
6. Repeat

There's no adaptation based on:
- How many times the user has rejected in a row
- Whether this is the user's first request of the day
- Whether the user's request is vague or specific
- Whether the bot has relevant context from the conversation that preceded the request

### The bot never initiates

It only responds. It never says:
- "I noticed you haven't posted in 2 days — want me to draft something?"
- "Your last 3 meme posts got way more engagement than announcements — should we lean into that?"
- "I learned from your last few rejections that you prefer shorter captions — I'll keep that in mind"
- "Based on your schedule, you have an evening slot coming up in 2 hours"

### Agent mode's "thinking" is invisible

The agent does 3-15 turns of tool-use reasoning internally. All the user sees is a single status message that gets edited with tool call names. The rich reasoning — why it chose a certain content type, how it interpreted the brand voice, what it learned from feedback — is thrown away. Only the final JSON draft survives.

### The generation-to-review handoff is a hard cut

When the draft appears, it arrives with zero context. The user doesn't know:
- What content type was chosen and why
- What the image prompt was (unless they check state)
- What feedback history influenced the output
- Whether the bot tried something different from last time

### Conversation history doesn't cross the boundary

If the user says "I want something edgy for our community" and the bot chats about it, then the user says "ok make it" — the generation pipeline receives only "ok make it" with zero context from the preceding conversation. The intent router classifies it as `generate_content` and the conversation vanishes.

---

## 8. Architectural Seams

### Seam 1: Intent router as hard wall
`intent_router.classify_intent()` returns a single intent string. This intent determines which "brain" handles the message. There is no mechanism to carry context, personality, or conversation state across the boundary.

### Seam 2: Engine has no personality parameter
`engine.run_agent(request, on_tool_call, revision_context)` has no way to receive personality, memory, or conversation history. The system prompt is built by `skill_prompt.build_system_prompt()` which reads only from brand config and content type definitions.

### Seam 3: Pipeline has no personality parameter
`brain.pipeline_generate(request, brand_context, on_step)` receives raw brand context as a string. No personality, no memory, no conversation history.

### Seam 4: _send_draft is stateless presentation
`_send_draft()` receives a draft dict and renders it. It has no access to the agent's reasoning, the conversation that led to the request, or any context about creative choices.

### Seam 5: Feedback is indirect
The agent sees learned preferences (a summarized document) and the last 10 feedback entries. But it doesn't know "this specific user rejected the last draft for being too long" in real-time. The feedback summary only updates every 10 approvals.

### Seam 6: Auto-post is personality-blind
The scheduler calls `engine.run_agent()` directly. Even if personality were injected into the engine, the auto-post path bypasses all Telegram handler logic and conversation context.
