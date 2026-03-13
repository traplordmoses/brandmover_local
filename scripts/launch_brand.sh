#!/usr/bin/env bash
# Launch a BrandMover instance for a specific brand.
#
# Usage:
#   ./scripts/launch_brand.sh <brand_name>
#   ./scripts/launch_brand.sh brand2
#
# This loads .env.<brand_name> and runs the bot.
# Each brand needs:
#   1. .env.<brand_name>  — config with unique TELEGRAM_BOT_TOKEN, BRAND_FOLDER, etc.
#   2. A brand directory   — e.g. brand2/ with guidelines.md, personality/, etc.
#   3. A state directory   — auto-created at state_<brand_name>/
#
# To set up a new brand:
#   ./scripts/new_brand.sh <brand_name>

set -euo pipefail

BRAND="${1:?Usage: $0 <brand_name>}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${PROJECT_DIR}/.env.${BRAND}"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE not found."
    echo "Run: ./scripts/new_brand.sh $BRAND"
    exit 1
fi

echo "Launching BrandMover for: $BRAND"
echo "Config: $ENV_FILE"
echo "---"

# Override .env path — python-dotenv loads from project root by default,
# so we set DOTENV_PATH and patch the load in settings.py
export BRANDMOVER_ENV_FILE="$ENV_FILE"

cd "$PROJECT_DIR"
exec python3 main.py
