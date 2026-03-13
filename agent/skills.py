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
from datetime import date
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
