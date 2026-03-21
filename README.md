# BrandMover Local

An autonomous AI marketing agent that runs via Telegram. Send a natural language request, get a branded post draft with a generated image, review it, and publish to X with one command. Includes a web dashboard for mission control.

**Pipeline:** Telegram message → Read brand guidelines → LLM generates caption + image prompt → Smart model routing generates image → Template composition with text overlay → Draft sent to Telegram for review → /approve posts to X.

## Features

### Content Generation
- **Agent mode** — Claude tool-use loop with 15 tools: brand guidelines, Figma, feedback, image generation, skills, memory search, subagent delegation, and more
- **Skills system** — the agent creates and saves reusable capabilities. Skills persist across sessions, so the agent gets cumulatively smarter over time
- **Smart image routing** — auto-selects the best Replicate model per content type (Flux 1.1 Pro general, Nano Banana for text overlays, Recraft SVG for brand assets, Seedream for lifestyle)
- **Template system** — upload custom templates (Figma exports, meme frames, etc.), Claude Vision analyzes regions, alpha-composite layering preserves transparency
- **brand_3d pipeline** — dedicated 3D asset generation with master prompt splicing, reference image routing, optional LoRA trigger, and parallel N=3 option generation
- **Kinetic typography videos** — word-by-word reveals, staggered animations, crossfade transitions via Playwright HTML→PNG frames + ffmpeg

### Creative Workbench
- **Brand asset ingestion** — upload an image with "add to brand", get AI analysis (colors, style, brand alignment, suggested captions), then `/save_asset` to catalog it into the visual library
- **Video style reverse-engineering** — upload a reference video with "break this down", bot extracts keyframes and breaks down scene-by-scene (timing, typography, colors, transitions), then `/remake` to recreate it in your brand style
- **Conversational guidelines editor** — `/brand_edit make the tone more casual` edits your brand guidelines through natural language with diff preview and `/confirm_edit` to apply

### Autonomous Posting
- **Heartbeat scheduler** — assess → reason → dispatch cycle replaces dumb cron. Claude decides what to post and when based on signals
- **Campaign system** — multi-day campaigns with structured posts, progress tracking, auto-execution
- **Exact-copy posting** — scheduled posts with verbatim text bypass the agent entirely, no unwanted image generation
- **Content planner** — AI-driven content calendar balancing themes, types, and cadence
- **Proactive generation** — when the feed goes quiet, the heartbeat picks a topic angle and generates a draft

### Intelligence
- **Semantic memory** — searches past generations by relevance to find what worked before, with temporal decay
- **Feedback learning** — learns from approve/reject history, auto-extracts preferences with temporal decay
- **Preference engine** — scores drafts against learned patterns before submission, auto-rejects low-quality output
- **Diversity tracker** — tracks content structure (hooks, body, CTAs) to prevent repetition
- **Topic bank** — rotating library of content angles with LRU selection for variety

### Self-Modification
- **Claude Code CLI** (`/code`) — run Claude Code directly from Telegram to fix, extend, or debug the bot's own codebase
- **Self-healing** — when the agent hits an error, auto-escalation (`CLAUDE_CODE_AUTO_ESCALATE=true`) spawns Claude Code to diagnose and patch the issue
- **Session resume** — `/code resume <follow-up>` continues the last Claude Code session with additional instructions
- **Safety rails** — syntax validation before hot-reload, one-click revert via inline button, daily usage cap, concurrency lock, full audit log

### Design Studio (Telegram Mini App)
- **Visual design builder** — full-screen Mini App embedded in Telegram (`/design`), also accessible at `/design` in the dashboard
- **Brand Board** — visual swatches, fonts, style keywords, voice traits; upload reference images for Claude Vision analysis
- **Design Agent** — conversational AI (Haiku) that refines design briefs before expensive generation, outputs structured JSON specs
- **Composer** — structured form with content type selector, layout presets (16:9/9:16/1:1), text inputs, JSON spec preview, SSE progress streaming
- **Templates** — searchable grid gallery, tap to pre-load into the composer
- **History** — browse past generations with expandable detail cards

### Infrastructure
- **Model fallback** — automatic retry with fallback models on API errors
- **Subagent delegation** — spawn lightweight sub-agents for research and analysis
- **Session transcripts** — JSONL per-user logs for debugging and analytics
- **Multi-brand support** — run multiple brands from the same codebase with isolated config, state, and assets
- **Workspace injection** — operator-editable personality and memory files that customize behavior without touching code

### Dashboard
- **Content calendar** — week view of all scheduled, pending, posted, and rejected content
- **Bot status** — live heartbeat log, activity feed, pause/resume toggle
- **Brand docs viewer/editor** — review and edit brand guidelines with markdown preview
- **Campaign overview** — campaign cards with progress bars and slot timelines
- **Design Studio** — visual design builder (same as the Telegram Mini App, accessible at `/design`)
- **Settings** — schedule editor, generation stats, cost tracking, learned preferences

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/traplordmoses/brandmover_local.git
cd brandmover_local
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. Fill in `.env`

**Required API keys:**

| Variable | How to get it |
|----------|---------------|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) — create an API key |
| `REPLICATE_API_TOKEN` | [replicate.com](https://replicate.com) — Account Settings > API tokens |
| `TELEGRAM_BOT_TOKEN` | Message [@BotFather](https://t.me/BotFather) on Telegram, send `/newbot` |
| `TELEGRAM_ALLOWED_USER_ID` | Message [@userinfobot](https://t.me/userinfobot) — it replies with your user ID |

**For X/Twitter posting:**

| Variable | How to get it |
|----------|---------------|
| `X_API_KEY` | [developer.twitter.com](https://developer.twitter.com) — create a project/app |
| `X_API_SECRET` | Same app, under Keys and Tokens |
| `X_ACCESS_TOKEN` | Same app — generate Access Token with **Read+Write** permissions |
| `X_ACCESS_SECRET` | Same app — Access Token Secret |
| `X_BEARER_TOKEN` | Same app — Bearer Token |

**Key settings:**

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MODE` | `pipeline` | Set to `agent` for the full tool-use loop (recommended) |
| `AGENT_MODEL` | `claude-sonnet-4-6` | Which Claude model to use |
| `BRAND_NAME` | `MyBrand` | Your brand name |
| `IMAGE_MODEL` | `auto` | `auto` routes by content type, or force a specific model |
| `AUTO_POST_ENABLED` | `false` | Enable scheduled auto-posting |
| `HEARTBEAT_ENABLED` | `true` | Enable the heartbeat reasoning layer |
| `CLAUDE_CODE_ENABLED` | `false` | Enable `/code` command for self-modification |
| `CLAUDE_CODE_AUTO_ESCALATE` | `false` | Auto-spawn Claude Code on agent errors |
| `CLAUDE_CODE_DAILY_LIMIT` | `10` | Max Claude Code invocations per day |
| `CLAUDE_CODE_TIMEOUT_SECONDS` | `300` | Timeout per Claude Code session |
| `MINIAPP_URL` | — | HTTPS URL for the Design Studio Mini App |
| `DASHBOARD_CORS_ORIGINS` | — | Allowed CORS origins for dashboard/Mini App |

### 3. Set up your brand

**Option A — Interactive onboarding (recommended):** Send `/onboard` in Telegram and the bot walks you through setup step by step.

**Option B — Write guidelines manually:** Create `brand/guidelines.md` with your brand's voice, tone, colors, and style rules. See `brand/guidelines.md.example` for the format.

**Option C — Bootstrap from PDF:** Upload a brand guidelines PDF to the bot and it auto-extracts structured guidelines via Claude Vision.

Customize the agent's personality by editing:
- `brand/personality/system_prompt.md` — the agent's character and tone
- `brand/personality/memory.md` — persistent notes about you and your preferences

Place your logo at `brand/assets/logo.png` — the compositor overlays it on generated images.

### 4. Run

```bash
python3 main.py
```

The bot validates your `.env` on startup and tells you exactly which variables are missing.

### 5. Launch the dashboard (optional)

```bash
# Install dashboard dependencies
pip install fastapi uvicorn
cd dashboard/frontend && npm install && cd ../..

# Start both servers
python3 -m uvicorn dashboard.backend.main:app --port 8100 --reload &
cd dashboard/frontend && npm run dev &
```

Open [http://localhost:5173](http://localhost:5173) — the dashboard reads from the same data stores as the bot.

## Usage

Message the bot on Telegram:

### Content Generation
- **"write a post about our new feature"** — generates a draft with image
- **"make a meme about X"** — generates a meme with Impact font top/bottom text
- **/approve [N]** — approve the draft (option N if multiple images)
- **/reject make it more urgent** — revises the draft with your feedback
- **/edit make the background darker** — surgical img2img edit on the last image
- **/post** — publish the approved draft to X/Twitter

### Brand Building
- **Upload image + "add to brand"** — AI analyzes colors, style, suggests captions
- **/save_asset [type]** — catalog the analyzed image into the brand library
- **Upload video + "break this down"** — scene-by-scene style breakdown
- **/remake** — recreate the analyzed video in your brand style
- **/brand_edit \<instruction\>** — edit guidelines via natural language
- **/confirm_edit** / **/cancel_edit** — apply or discard the edit

### Scheduling
- **/schedule 3pm tomorrow post about our launch** — schedule a post
- **/autopause** / **/autoresume** — pause/resume auto-posting
- **/autostatus** — show scheduler status

### Management
- **/status** — show the current pending draft
- **/cancel** — clear the pending draft
- **/feedback** — show approval/rejection stats
- **/preferences** — view/manage learned preferences
- **/analytics** — show generation stats and cost tracking
- **/help** — show all available commands

## Setting Up a Second Brand

BrandMover supports running multiple brands from the same codebase. Each brand gets its own Telegram bot, X/Twitter account, guidelines, personality, and state.

```bash
# Bootstrap
./scripts/new_brand.sh mybrand

# Configure .env.mybrand with new API keys

# Launch alongside main brand
python3 main.py                    # Terminal 1 — main brand
./scripts/launch_brand.sh mybrand  # Terminal 2 — second brand
```

## Project Structure

```
agent/              Core logic (no Telegram dependency)
  engine.py           Tool-use agent loop (main architecture)
  tools.py            15 tool definitions and handlers
  brain.py            Claude LLM calls (pipeline + agent modes)
  image_gen.py        Replicate image generation (Flux, Seedream, Recraft, Nano)
  compositor.py       PIL image composition (glass-morphism, text overlay)
  asset_ingest.py     Brand asset analysis + cataloging via Claude Vision
  video_reverse.py    Video style reverse-engineering (keyframes + scene breakdown)
  guidelines_editor.py  Conversational guidelines editing via Claude
  heartbeat.py        Assess → reason → dispatch scheduling layer
  skills.py           Persistent agent-created capabilities
  session.py          Session memory with temporal decay
  feedback.py         Feedback log + learned preferences
  publisher.py        X/Twitter posting via tweepy

bot/                Telegram interface
  telegram_bot.py     Bot setup, handler registration, scheduler launch
  handlers/           Command and message handlers (modular)

config/             Configuration
  settings.py         .env loader with startup validation
  schedule.json       Auto-post time slots

dashboard/          Web dashboard (FastAPI + React)
  backend/            FastAPI API reading from bot's data stores
  frontend/           React SPA with Tailwind (Vite)

brand/              Your brand assets (per-instance, mostly gitignored)
  guidelines.md       Brand voice, tone, colors, style rules
  personality/        Agent personality and operator notes
  skills/             Agent-created reusable capabilities
  assets/             Logo, fonts, images, asset library
  templates/          Template PNGs + manifest
  prompts/            Custom image prompt templates per content type
  references/         Reference images for style consistency

scripts/            Setup and management
  auto_post.py        Background scheduler loop
  new_brand.sh        Bootstrap a new brand instance
  launch_brand.sh     Launch a specific brand instance

state/              Runtime state (gitignored, per-instance)
```

## Testing

```bash
python3 -m pytest tests/ -v
```

## Updating

```bash
git pull origin main
pip install -r requirements.txt
python3 main.py
```

Your `brand/`, `.env`, and `state/` are all gitignored, so `git pull` won't overwrite your data.
