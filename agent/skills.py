"""
Skills system — persistent, agent-created capabilities.

ARCHITECTURE:
A skill is a directory containing a SKILL.md (markdown instructions) + optional
scripts and references. Skills are the agent's way of remembering HOW to do
things it figured out, so it never has to re-derive the same solution twice.

The design follows OpenClaw's approach: skills are primarily markdown-based
instructions, not compiled code. The agent reads a skill's SKILL.md, understands
the pattern, and executes it using tools it already has (execute_code, generate_image,
web_fetch, etc.). Scripts are optional pre-written code the agent can run directly.

PROGRESSIVE LOADING:
- On every agent run: only skill names + one-line descriptions are loaded (~50 tokens each)
- When agent calls use_skill: full SKILL.md + scripts loaded into context
- This keeps the base prompt lean while giving access to unlimited capabilities

LIFECYCLE:
1. Agent encounters a novel problem → solves it with existing tools
2. Agent recognizes the solution is reusable → calls create_skill to save it
3. Future sessions → agent sees skill in registry → calls use_skill → done in 1 turn

STORAGE:
brand/skills/
├── REGISTRY.json           # Name + description index (loaded every session)
├── meme-generator/
│   ├── SKILL.md            # Instructions + usage guide
│   └── scripts/
│       └── make_meme.py    # Reusable script
└── trending-topics/
    ├── SKILL.md
    └── references/
        └── sources.json
"""

import json
import logging
import shutil
from datetime import date, datetime
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

_SKILLS_DIR = Path(settings.BRAND_FOLDER) / "skills"
_REGISTRY_FILE = _SKILLS_DIR / "REGISTRY.json"


def _ensure_dir():
    """Create the skills directory if it doesn't exist."""
    _SKILLS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Registry operations
# ---------------------------------------------------------------------------

def load_registry() -> list[dict]:
    """Load the skills registry. Returns list of skill entries."""
    if not _REGISTRY_FILE.exists():
        return []
    try:
        data = json.loads(_REGISTRY_FILE.read_text(encoding="utf-8"))
        return data.get("skills", [])
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read skills registry: %s", e)
        return []


def _save_registry(skills: list[dict]) -> None:
    """Write the skills registry to disk."""
    _ensure_dir()
    _REGISTRY_FILE.write_text(
        json.dumps({"skills": skills}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_skill_summary() -> str:
    """One-line-per-skill summary for system prompt injection.

    Returns empty string if no skills exist — the system prompt
    conditionally includes the skills section only when non-empty.
    """
    skills = load_registry()
    if not skills:
        return ""
    lines = []
    for s in skills:
        if s.get("status", "active") == "active":
            lines.append(f"- **{s['name']}**: {s['description']}")
    return "\n".join(lines) if lines else ""


# ---------------------------------------------------------------------------
# Skill CRUD
# ---------------------------------------------------------------------------

def load_skill(name: str) -> dict | None:
    """Load a skill's full content for agent consumption.

    Returns {name, content, scripts, references, path} or None if not found.
    The agent receives the full SKILL.md text and any script files, so it can
    follow the instructions and execute code directly.
    """
    import re as _re
    if not name or not _re.fullmatch(r"[A-Za-z0-9_-]+", name):
        logger.warning("Rejected invalid skill name in load_skill: %s", name)
        return None

    skill_dir = _SKILLS_DIR / name
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.exists():
        return None

    content = skill_file.read_text(encoding="utf-8")

    # Collect scripts
    scripts = {}
    scripts_dir = skill_dir / "scripts"
    if scripts_dir.is_dir():
        for f in sorted(scripts_dir.iterdir()):
            if f.is_file():
                try:
                    scripts[f.name] = f.read_text(encoding="utf-8")
                except OSError:
                    scripts[f.name] = "(failed to read)"

    # Collect references
    references = {}
    refs_dir = skill_dir / "references"
    if refs_dir.is_dir():
        for f in sorted(refs_dir.iterdir()):
            if f.is_file() and f.stat().st_size < 50_000:  # skip large files
                try:
                    references[f.name] = f.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    references[f.name] = "(binary or unreadable)"

    return {
        "name": name,
        "content": content,
        "scripts": scripts,
        "references": references,
        "path": str(skill_dir),
    }


def create_skill(
    name: str,
    description: str,
    skill_md: str,
    scripts: dict[str, str] | None = None,
    author: str = "agent",
    overwrite: bool = False,
) -> dict:
    """Create a new skill. Returns {success, path, message}.

    Args:
        name: Skill identifier (alphanumeric, hyphens, underscores).
        description: One-line description shown in registry.
        skill_md: Full SKILL.md content (markdown with YAML frontmatter).
        scripts: Optional {filename: code} dict for scripts/ directory.
        author: "agent" or "operator".
        overwrite: If True, replace an existing skill.
    """
    # Validate name
    if not name or not all(c.isalnum() or c in "-_" for c in name):
        return {
            "success": False,
            "message": f"Invalid skill name: '{name}'. Use alphanumeric, hyphens, underscores only.",
        }

    skill_dir = _SKILLS_DIR / name
    if skill_dir.exists() and not overwrite:
        return {
            "success": False,
            "message": f"Skill '{name}' already exists. Set overwrite=true to replace it.",
        }

    # Clean existing if overwriting
    if skill_dir.exists() and overwrite:
        shutil.rmtree(skill_dir)

    # Create directory structure
    skill_dir.mkdir(parents=True, exist_ok=True)

    # Write SKILL.md
    (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")

    # Write scripts — validate filenames to prevent path traversal
    if scripts:
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        for filename, code in scripts.items():
            safe_name = Path(filename).name  # Strip any directory components
            if not safe_name or safe_name.startswith(".") or "/" in filename or "\\" in filename:
                logger.warning("Rejected unsafe script filename: %s", filename)
                continue
            (scripts_dir / safe_name).write_text(code, encoding="utf-8")

    # Verify resolved path is under _SKILLS_DIR (defense-in-depth)
    if not skill_dir.resolve().is_relative_to(_SKILLS_DIR.resolve()):
        shutil.rmtree(skill_dir, ignore_errors=True)
        return {"success": False, "message": "Invalid skill path."}

    # Limit skill content size
    if len(skill_md) > 100_000:
        shutil.rmtree(skill_dir, ignore_errors=True)
        return {"success": False, "message": "Skill content too large (max 100KB)."}

    # Update registry
    registry = load_registry()
    # Remove existing entry if overwriting
    registry = [s for s in registry if s["name"] != name]
    registry.append({
        "name": name,
        "description": description,
        "created": date.today().isoformat(),
        "author": author,
        "status": "active",
    })
    _save_registry(registry)

    logger.info("Created skill '%s' with %d scripts", name, len(scripts or {}))
    return {
        "success": True,
        "path": str(skill_dir),
        "message": f"Skill '{name}' created and registered. It will be available in all future sessions.",
    }


def update_skill(
    name: str,
    skill_md: str | None = None,
    scripts: dict[str, str] | None = None,
    description: str | None = None,
) -> dict:
    """Update an existing skill's content, scripts, or description."""
    skill_dir = _SKILLS_DIR / name
    if not skill_dir.exists():
        return {"success": False, "message": f"Skill '{name}' not found."}

    if skill_md:
        (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")

    if scripts:
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        for filename, code in scripts.items():
            safe_name = Path(filename).name
            if not safe_name or safe_name.startswith(".") or "/" in filename or "\\" in filename:
                logger.warning("Rejected unsafe script filename in update: %s", filename)
                continue
            (scripts_dir / safe_name).write_text(code, encoding="utf-8")

    if description:
        registry = load_registry()
        for s in registry:
            if s["name"] == name:
                s["description"] = description
                s["updated"] = date.today().isoformat()
                break
        _save_registry(registry)

    logger.info("Updated skill '%s'", name)
    return {"success": True, "message": f"Skill '{name}' updated."}


def delete_skill(name: str) -> dict:
    """Delete a skill and remove it from the registry."""
    skill_dir = _SKILLS_DIR / name
    if not skill_dir.exists():
        return {"success": False, "message": f"Skill '{name}' not found."}
    if not skill_dir.resolve().is_relative_to(_SKILLS_DIR.resolve()):
        return {"success": False, "message": "Invalid skill path."}

    shutil.rmtree(skill_dir)

    registry = load_registry()
    registry = [s for s in registry if s["name"] != name]
    _save_registry(registry)

    logger.info("Deleted skill '%s'", name)
    return {"success": True, "message": f"Skill '{name}' deleted."}


# ---------------------------------------------------------------------------
# Learning infrastructure
# ---------------------------------------------------------------------------

def append_skill_learning(name: str, learning: str, source: str = "feedback") -> dict:
    """Append a learning to a skill's Edge Cases & Learnings section.

    Args:
        name: Skill name.
        learning: The learning text (one line, actionable).
        source: "feedback", "self_review", or "operator".

    Finds the '## Edge Cases & Learnings' section in SKILL.md and appends
    the learning. If no such section exists, appends one at the end.
    Returns {success, message}.
    """
    skill_dir = _SKILLS_DIR / name
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.exists():
        return {"success": False, "message": f"Skill '{name}' not found."}

    timestamp = date.today().isoformat()
    entry = f"- [{timestamp}] {learning} (source: {source})"

    content = skill_file.read_text(encoding="utf-8")

    section_header = "## Edge Cases & Learnings"
    if section_header in content:
        # Find the section and append after it (before the next ## or end of file)
        idx = content.index(section_header)
        rest = content[idx + len(section_header):]
        # Find the next ## heading (if any)
        import re
        next_section = re.search(r"\n## ", rest)
        if next_section:
            insert_pos = idx + len(section_header) + next_section.start()
            content = content[:insert_pos] + "\n" + entry + content[insert_pos:]
        else:
            content = content.rstrip() + "\n" + entry + "\n"
    else:
        content = content.rstrip() + "\n\n" + section_header + "\n" + entry + "\n"

    skill_file.write_text(content, encoding="utf-8")
    logger.info("Appended learning to skill '%s': %s", name, learning[:60])
    return {"success": True, "message": f"Learning appended to '{name}'."}


def get_skill_stats(name: str) -> dict:
    """Get usage and performance stats for a skill.

    Returns {uses, approvals, rejections, approval_rate, last_used, learnings_count}.
    Reads from REGISTRY.json stats field.
    """
    registry = load_registry()
    entry = next((s for s in registry if s["name"] == name), None)
    if not entry:
        return {"error": f"Skill '{name}' not found in registry."}

    stats = entry.get("stats", {})
    uses = stats.get("uses", 0)
    approvals = stats.get("approvals", 0)
    rejections = stats.get("rejections", 0)

    # Count learnings from SKILL.md
    learnings_count = 0
    skill_file = _SKILLS_DIR / name / "SKILL.md"
    if skill_file.exists():
        content = skill_file.read_text(encoding="utf-8")
        section_header = "## Edge Cases & Learnings"
        if section_header in content:
            section = content[content.index(section_header):]
            # Count lines starting with "- ["
            learnings_count = sum(
                1 for line in section.splitlines() if line.strip().startswith("- [")
            )

    return {
        "uses": uses,
        "approvals": approvals,
        "rejections": rejections,
        "approval_rate": round(approvals / uses, 2) if uses > 0 else 0.0,
        "last_used": stats.get("last_used"),
        "learnings_count": learnings_count,
    }


def record_skill_use(name: str, approved: bool | None = None) -> None:
    """Record that a skill was used. Called when feedback comes in.

    Updates the skill's stats in REGISTRY.json:
    - Increments uses count
    - If approved is True/False, increments approvals/rejections
    - Updates last_used timestamp
    """
    registry = load_registry()
    entry = next((s for s in registry if s["name"] == name), None)
    if not entry:
        logger.warning("record_skill_use: skill '%s' not in registry", name)
        return

    stats = entry.setdefault("stats", {
        "uses": 0, "approvals": 0, "rejections": 0, "last_used": None,
    })
    stats["uses"] = stats.get("uses", 0) + 1
    stats["last_used"] = datetime.now().isoformat(timespec="seconds")

    if approved is True:
        stats["approvals"] = stats.get("approvals", 0) + 1
    elif approved is False:
        stats["rejections"] = stats.get("rejections", 0) + 1

    _save_registry(registry)


def get_skills_for_routing(max_tokens: int = 500) -> str:
    """Get a compact skill summary optimized for routing decisions.

    Returns a tighter format than get_skill_summary():
    - name | description | approval_rate | uses
    Sorted by usage frequency (most used first).
    Only includes active skills.
    """
    skills = load_registry()
    active = [s for s in skills if s.get("status", "active") == "active"]
    if not active:
        return ""

    # Sort by usage frequency (most used first)
    def _uses(s):
        return s.get("stats", {}).get("uses", 0)

    active.sort(key=_uses, reverse=True)

    lines = []
    char_count = 0
    for s in active:
        stats = s.get("stats", {})
        uses = stats.get("uses", 0)
        approvals = stats.get("approvals", 0)
        rate = f"{approvals / uses:.0%}" if uses > 0 else "n/a"
        line = f"{s['name']} | {s['description']} | {rate} | {uses} uses"
        # Rough token estimate: 1 token ~= 4 chars
        if char_count + len(line) > max_tokens * 4:
            break
        lines.append(line)
        char_count += len(line) + 1

    return "\n".join(lines)
