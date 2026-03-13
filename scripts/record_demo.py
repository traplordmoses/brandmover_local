#!/usr/bin/env python3
"""CLI entry point for recording demo walkthroughs.

Usage:
    python scripts/record_demo.py demos/scripts/swipe-feature-demo.json
    python scripts/record_demo.py demos/scripts/swipe-feature-demo.json --mode screenshot
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.demo_recorder import record_demo


def main():
    parser = argparse.ArgumentParser(description="Record a demo walkthrough")
    parser.add_argument("script", help="Path to demo script JSON file")
    parser.add_argument(
        "--mode",
        choices=["video", "screenshot"],
        default=None,
        help="Override recording mode (default: use script's mode field)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    result = asyncio.run(record_demo(args.script, mode_override=args.mode))

    if result.error:
        print(f"\nRecording failed: {result.error}", file=sys.stderr)
        sys.exit(1)

    print(f"\nDemo: {result.script_name}")
    print(f"Mode: {result.mode}")
    print(f"Duration: {result.duration_seconds}s")

    if result.video_path:
        print(f"Video: {result.video_path}")

    if result.screenshot_paths:
        print(f"Screenshots ({len(result.screenshot_paths)}):")
        for p in result.screenshot_paths:
            print(f"  {p}")


if __name__ == "__main__":
    main()
