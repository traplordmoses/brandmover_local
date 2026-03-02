# BrandMover Local

An autonomous AI marketing agent that runs via Telegram. Send a natural language request, get a branded post draft with a generated image, review it, and publish to X with one command.

**Pipeline:** Telegram message -> Read brand guidelines -> LLM generates caption + hashtags + image prompt -> Smart model routing generates image -> Template composition with text overlay -> Draft sent to Telegram for review -> /approve posts to X.

## Features

- **Agent mode** — Claude tool-use loop with brand guidelines, Figma, feedback history, and onchain scripts
- **Smart image routing** — auto-selects the best Replicate model per content type (Flux 1.1 Pro Ultra general, Nano Banana for text overlays, Recraft SVG for brand assets, Seedream for lifestyle)
- **Template system** — upload custom templates (Figma exports, meme frames, etc.), Claude Vision analyzes regions, alpha-composite layering preserves transparency. Meme templates get Impact font with classic top/bottom text.
- **brand_3d pipeline** — dedicated 3D asset generation with master prompt splicing, category-based reference image routing, optional LoRA trigger, and parallel N=3 option generation
- **Parallel image options** — brand_3d generates 3 options simultaneously; pick the best with `/approve 1`, `/approve 2`, or `/approve 3`
- **Surgical /edit** — apply targeted img2img edits to the last generated image without re-running the full pipeline (`/edit make the background darker`)
- **Adaptive compositor** — fallback branded image composition with glass-morphism backgrounds, text overlay, and platform badges
- **Style profiles** — named collections of reference images that apply a consistent visual style via img2img
- **Mascot generation** — character-consistent generation using multi-reference stitched grids
- **Feedback learning** — learns from approve/reject history and auto-summarizes preferences
- **PDF brand bootstrap** — upload a brand guidelines PDF and auto-extract structured guidelines
- **Natural language template editing** — after uploading a template, describe region positions in plain English ("top text across top 15%, image fills full canvas") and Claude converts to pixel coordinates

## Setup

### 1. Clone and install

```bash
git clone https://github.com/traplordmoses/brandmover_local.git
cd brandmover_local
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
```

### 2. Fill in `.env`

You need API keys for the following services:

**Required:**

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
| `X_ACCESS_TOKEN` | Same app — generate Access Token with Read+Write permissions |
| `X_ACCESS_SECRET` | Same app — Access Token Secret |
| `X_BEARER_TOKEN` | Same app — Bearer Token |

Make sure your X app has **Read and Write** permissions.

**Optional:**

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `anthropic` | LLM provider (`anthropic`, `openai`, `gemini`) |
| `AGENT_MODE` | `pipeline` | `pipeline` (4-step) or `agent` (Claude tool-use loop) |
| `AGENT_MODEL` | `claude-sonnet-4-6` | Which Claude model to use for agent mode |
| `BRAND_NAME` | `FOID Foundation` | Your brand name — change this! |
| `IMAGE_MODEL` | `auto` | `auto` routes by content type, or force a specific model |
| `AUTO_POST_ENABLED` | `false` | Enable scheduled auto-posting |
| `FIGMA_ACCESS_TOKEN` | — | Figma personal access token for design reference integration |

### 3. Set up your brand

Replace `brand/guidelines.md` with your own brand guidelines. You can either:

**Option A — Write it manually:** Edit `brand/guidelines.md` with your brand's voice, tone, colors, hashtags, and style rules. The agent reads this file before every generation.

**Option B — Bootstrap from PDF:** Upload a brand guidelines PDF to the bot and it auto-extracts structured guidelines via Claude Vision.

**Option C — Interactive onboarding:** Send `/onboard` in Telegram and the bot walks you through setup step by step.

Place your logo at `brand/assets/logo.png` — the compositor overlays it on generated images.

### 4. Run

```bash
python main.py
```

The bot validates your `.env` on startup and will tell you exactly which variables are missing.

## Updating from a Previous Version

If you already have a clone of this repo:

```bash
cd brandmover_local
git pull origin main
source venv/bin/activate
pip install -r requirements.txt  # in case dependencies changed
```

Your `brand/` folder, `.env`, and `state/` are all gitignored or user-specific, so `git pull` won't overwrite them.

**If you get merge conflicts in `brand/guidelines.md`:** This file ships with example content. Keep your version — just `git checkout --ours brand/guidelines.md`.

After pulling, restart the bot:
```bash
python main.py
```

## Usage

Message the bot on Telegram:

- **"write a post about our new feature launch"** — generates a draft with image
- **"make a meme about X"** — generates a meme with Impact font top/bottom text on your meme template
- **/approve [N]** — approve and post to X (option N if multiple images)
- **/reject make it more urgent** — revises the draft with your feedback
- **/edit make the background darker** — surgical img2img edit on the last image
- **/status** — show the current pending draft
- **/cancel** — clear the pending draft
- **/refs** — show loaded reference materials
- **/feedback** — show approval/rejection stats
- **/learn** — trigger preference learning from feedback history
- **/analytics** — show generation stats and cost tracking
- **/history** — show recent generations
- **/help** — show all available commands

Upload a photo to use as a reference image. Add a caption to immediately generate with it, or reply with `reference`, `mascot`, `style <name>`, or `background`.

Only messages from your authorized Telegram user ID are processed. Everyone else is ignored.

## Templates

Templates let you define branded frames that wrap your generated images. Upload a PNG with transparent regions, and the bot composites generated images into the frame with text overlay.

### Upload a template

```
/template_upload meme
```

Then send the template image. Claude Vision analyzes the image and detects regions (image, text, logo areas).

### Adjust regions with natural language

After uploading, describe the layout in plain English:

```
top text across the top 15%, bottom text across the bottom 15%, image fills the full canvas
```

Claude converts this to pixel coordinates and updates the template.

### Test a template

```
/template_test meme
```

Generates a preview with placeholder content so you can verify alignment.

### Template commands

| Command | Description |
|---------|-------------|
| `/template` | Show compositor status and active templates |
| `/template on` | Enable template composition |
| `/template off` | Disable templates and compositor |
| `/template_upload <name>` | Upload a custom template image |
| `/template_test [type]` | Preview a template with placeholder content |
| `/template_from_reference` | Generate a template from a reference image |

### Meme templates

Templates named "meme" automatically get classic meme styling:
- **Impact font** (falls back to bold sans-serif if unavailable)
- **ALL CAPS** text
- **3px black outline** with white fill
- **Letter spacing** for readability
- Title at top, subtitle at bottom

### How it works

1. Template is stored as a PNG in `brand/templates/` with a JSON manifest
2. When generating content, the bot matches templates by `content_type` (exact match first, then universal templates by aspect ratio)
3. Generated image is composited under the template using alpha blending — transparent areas show the image, opaque frame sits on top
4. Text is fitted into text regions with binary-search font sizing and word wrap
5. The composed image is what gets sent to Telegram and posted to X

## Style Profiles

Style profiles let you train the bot's visual identity from reference images.

### Quick start

```
/style create 3d_card Revolut-style 3D floating card visuals
```
Upload reference photos with caption `3d_card` to add them to the profile.

```
/style 3d_card announcement
```
Now every announcement uses the 3D card references at 0.3 strength via img2img.

### Commands

| Command | Description |
|---------|-------------|
| `/style` | List all profiles with image counts and active mappings |
| `/style create <name> <description>` | Create a new profile |
| `/style <name> <content_type>` | Set profile as active for a content type |
| `/style <name> info` | Show profile details |
| `/style <name> remove` | Remove profile from all active mappings |

## Content Types

The agent selects the best content type for each request, which determines image model routing and template selection:

| Type | Description |
|------|-------------|
| `announcement` | Product launches, updates, news, partnerships |
| `campaign` | Marketing campaigns, launches |
| `meme` | Memes, humor, viral content |
| `engagement` | Conversation starters, polls |
| `advice` | Tips, recommendations, guidance |
| `lifestyle` | Aspirational, day-in-the-life, culture |
| `event` | Conferences, AMAs, meetups |
| `educational` | Tutorials, explainers, how-tos |
| `brand_asset` | Logos, icons, badges, graphics |
| `community` | Giveaways, engagement posts |
| `market_commentary` | Market analysis, price action, trends |
| `brand_3d` | 3D product illustrations, objects |

## Project Structure

```
agent/           Core logic (no Telegram dependency)
  brain.py         Claude LLM calls (pipeline + agent modes)
  engine.py        Tool-use agent loop
  tools.py         Tool definitions and handlers
  image_gen.py     Replicate image generation
  compositor.py    PIL image composition (fallback when no template)
  template_memory.py  Template storage, analysis, and composition
  content_types.py    Content type definitions (single source of truth)
  state.py         Pending draft management
  feedback.py      Feedback log + learned preferences
  publisher.py     X/Twitter posting via tweepy

bot/             Telegram interface
  telegram_bot.py  Bot setup, handler registration
  handlers.py      All command and message handlers

config/          Configuration
  settings.py      .env loader with startup validation

brand/           Your brand assets (user-specific, mostly gitignored)
  guidelines.md    Brand voice, tone, colors, style rules
  assets/          Logo, fonts, images
  templates/       Template PNGs + manifest.json
  references/      Reference images for style consistency

state/           Runtime state (gitignored)
```

## Testing

```bash
python -m pytest tests/ -v
```
