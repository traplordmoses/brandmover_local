# BrandMover Local

An autonomous AI marketing agent that runs via Telegram. Send a natural language request, get a branded post draft with a generated image, review it, and publish to X with one command.

**Pipeline:** Telegram message -> Read brand guidelines -> LLM generates caption + image prompt -> Smart model routing generates image -> Template composition with text overlay -> Draft sent to Telegram for review -> /approve posts to X.

## Features

- **Agent mode** — Claude tool-use loop with 15 tools: brand guidelines, Figma, feedback, image generation, skills, memory search, subagent delegation, and more
- **Skills system** — the agent creates and saves reusable capabilities. Skills persist across sessions, so the agent gets cumulatively smarter over time
- **Smart image routing** — auto-selects the best Replicate model per content type (Flux 1.1 Pro general, Nano Banana for text overlays, Recraft SVG for brand assets, Seedream for lifestyle)
- **Template system** — upload custom templates (Figma exports, meme frames, etc.), Claude Vision analyzes regions, alpha-composite layering preserves transparency
- **brand_3d pipeline** — dedicated 3D asset generation with master prompt splicing, reference image routing, optional LoRA trigger, and parallel N=3 option generation
- **Semantic memory** — searches past generations by relevance to find what worked before, with temporal decay so recent work is weighted higher
- **Subagent delegation** — the agent can spawn lightweight sub-agents for research and analysis tasks
- **Model fallback** — automatic retry with fallback models when the primary API returns errors (429/500/503)
- **Event hooks** — async pub/sub system for decoupled side effects (analytics, notifications, logging)
- **Session transcripts** — JSONL per-user logs of all agent interactions for debugging and analytics
- **Channel abstraction** — normalized message envelope for multi-channel publishing (X/Twitter, with pluggable support for Discord, LinkedIn, etc.)
- **Workspace injection** — operator-editable personality and memory files that customize agent behavior without touching code
- **Feedback learning** — learns from approve/reject history, auto-extracts preferences with temporal decay
- **Multi-brand support** — run multiple brand instances from the same codebase with isolated config, state, and brand assets
- **Style profiles** — named collections of reference images that apply a consistent visual style via img2img
- **Adaptive compositor** — fallback branded image composition with glass-morphism backgrounds, text overlay, and platform badges
- **PDF brand bootstrap** — upload a brand guidelines PDF and auto-extract structured guidelines

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
| `HEARTBEAT_ENABLED` | `true` | Enable proactive content generation |

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

## Setting Up a Second Brand

BrandMover supports running multiple brands from the same codebase. Each brand gets its own Telegram bot, X/Twitter account, guidelines, personality, and state.

### 1. Bootstrap the new brand

```bash
./scripts/new_brand.sh mybrand
```

This creates:
- `.env.mybrand` — config file (fill in your API keys)
- `brand_mybrand/` — brand directory with placeholder guidelines and personality
- `state_mybrand/` — isolated state directory

### 2. Configure it

Edit `.env.mybrand` and fill in:
- A **new Telegram bot token** (create one via @BotFather)
- Your **Telegram user ID** (same as your main brand, or different)
- **X/Twitter credentials** for the new brand's account
- **Anthropic and Replicate keys** (can share with your main brand)

Edit `brand_mybrand/guidelines.md` with the new brand's voice, tone, and style. Customize `brand_mybrand/personality/system_prompt.md` for the agent's character.

### 3. Launch it

```bash
./scripts/launch_brand.sh mybrand
```

Run it alongside your main brand in a separate terminal:

```bash
# Terminal 1 — main brand
python3 main.py

# Terminal 2 — second brand
./scripts/launch_brand.sh mybrand
```

Each brand runs as a completely isolated instance — different bot, different state, different personality, different skills.

## Usage

Message the bot on Telegram:

- **"write a post about our new feature launch"** — generates a draft with image
- **"make a meme about X"** — generates a meme with Impact font top/bottom text
- **/approve [N]** — approve the draft (option N if multiple images)
- **/reject make it more urgent** — revises the draft with your feedback
- **/edit make the background darker** — surgical img2img edit on the last image
- **/post** — publish the approved draft to X/Twitter
- **/status** — show the current pending draft
- **/cancel** — clear the pending draft
- **/feedback** — show approval/rejection stats
- **/preferences** — view/manage learned preferences
- **/analytics** — show generation stats and cost tracking
- **/help** — show all available commands

Upload a photo to use as a reference image. Add a caption to immediately generate with it.

## Templates

Upload branded frames that wrap your generated images:

```
/template_upload meme        # Upload a template image
/template_test meme          # Preview with placeholder content
/template on                 # Enable template composition
/template off                # Disable templates
```

Templates named "meme" get classic Impact font styling with top/bottom text.

## Style Profiles

Train the bot's visual identity from reference images:

```
/style create 3d_card Revolut-style 3D floating card visuals
/style 3d_card announcement  # Apply to all announcements
```

Upload reference photos with caption `3d_card` to build the style.

## Project Structure

```
agent/              Core logic (no Telegram dependency)
  engine.py           Tool-use agent loop (main architecture)
  tools.py            15 tool definitions and handlers
  skills.py           Persistent agent-created capabilities
  skill_prompt.py     System prompt builder with workspace injection
  context_engine.py   Token-budget context assembly
  memory.py           Semantic search over past generations
  subagent.py         Sub-agent delegation for parallel tasks
  model_fallback.py   Automatic model fallback on API errors
  hooks.py            Async event pub/sub system
  transcript.py       JSONL session transcript logger
  channels/           Multi-channel publishing abstraction
  paths.py            Centralized path definitions (enables multi-brand)
  image_gen.py        Replicate image generation
  compositor.py       PIL image composition
  state.py            Pending draft management
  session.py          Persistent session memory with temporal decay
  feedback.py         Feedback log
  publisher.py        X/Twitter posting via tweepy

bot/                Telegram interface
  telegram_bot.py     Bot setup, handler registration
  handlers.py         All command and message handlers

config/             Configuration
  settings.py         .env loader with startup validation

brand/              Your brand assets (per-instance, mostly gitignored)
  guidelines.md       Brand voice, tone, colors, style rules
  personality/        Agent personality and operator notes
  skills/             Agent-created reusable capabilities
  assets/             Logo, fonts, images
  templates/          Template PNGs + manifest
  references/         Reference images for style consistency

scripts/            Setup and management
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
