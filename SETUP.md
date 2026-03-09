# BrandMover Setup Guide

## Prerequisites

- Python 3.11+
- Telegram bot token (from @BotFather)
- Anthropic API key (for Claude)
- Replicate API token (for image generation)

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url> brandmover
cd brandmover
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

Required keys:
- `ANTHROPIC_API_KEY` — Claude API access
- `TELEGRAM_BOT_TOKEN` — from @BotFather
- `TELEGRAM_ALLOWED_USER_ID` — your Telegram user ID (admin)
- `REPLICATE_API_TOKEN` — for image generation

### 3. Set up your brand

Option A — **Bootstrap from a PDF** (recommended):

```bash
python scripts/bootstrap_brand.py path/to/brand-deck.pdf
```

This extracts colors, fonts, voice, and visual identity into `brand/guidelines.md`.

Option B — **Manual setup**:

```bash
cp brand/guidelines.md.example brand/guidelines.md
cp brand/styles.json.example brand/styles.json
cp brand/personality/system_prompt.md.example brand/personality/system_prompt.md
cp brand/personality/memory.md.example brand/personality/memory.md
```

Edit each file to match your brand.

### 4. Run

```bash
python main.py
```

The bot will start listening for messages on Telegram.

## Configuration

### Key `.env` settings

| Setting | Default | Description |
|---------|---------|-------------|
| `BRAND_NAME` | `MyBrand` | Display name for your brand |
| `AGENT_MODE` | `pipeline` | `pipeline` (4-step) or `agent` (tool-use loop) |
| `UNIFIED_BRAIN_ENABLED` | `false` | Enable the unified agent brain (recommended) |
| `PIPELINE_MODE` | `full` | `full` (4-step) or `fast` (3-step) |
| `IMAGE_MODEL` | `auto` | Auto-routes by content type, or force a specific model |
| `WHISPER_ENABLED` | auto | Voice message transcription (auto-enabled with OpenAI key) |
| `AUTO_POST_ENABLED` | `false` | Scheduled auto-posting to X/Twitter |

### X/Twitter posting

Fill in the X API keys in `.env` to enable posting. The bot generates drafts for review — nothing posts without your approval (unless auto-post is enabled).

### Auto-posting

1. Set `AUTO_POST_ENABLED=true` in `.env`
2. Configure schedule in `config/schedule.json` (copy from `config/schedule.example.json`)
3. The scheduler runs alongside the bot and posts at configured times

### Discord cross-posting

Set `DISCORD_BOT_TOKEN` and `DISCORD_GUILD_ID` in `.env` to enable cross-posting approved content to Discord.

## Multi-Brand Usage

Each brand instance gets its own:
- `brand/` folder (gitignored — brand-specific assets)
- `state/` folder (gitignored — runtime state)
- `.env` file (gitignored — API keys and settings)

The framework code is brand-agnostic. Multiple brands can pull from the same repo without conflicts.

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/start` | Show welcome message |
| `/help` | List available commands |
| `/generate` | Generate a standalone asset |
| `/logo` | Generate a brand logo |
| `/review` | Run agent self-review |
| `/feedback` | Show learned preferences |
| `/ingest` | Ingest brand materials |
| `/schedule` | View/manage auto-post schedule |
| `/history` | View generation history |
| `/cost` | Show resource usage stats |

Send any message to generate content. The bot understands natural language — "make me a hype post about the new feature" works.

## Troubleshooting

- **Bot doesn't respond**: Check `TELEGRAM_ALLOWED_USER_ID` matches your Telegram user ID
- **Image generation fails**: Verify `REPLICATE_API_TOKEN` is valid
- **No brand context**: Ensure `brand/guidelines.md` exists (run bootstrap or copy example)
- **Tests**: Run `python -m pytest tests/ -v` to verify everything works
