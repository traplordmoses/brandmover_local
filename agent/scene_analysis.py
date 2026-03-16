"""
Scene Analysis — Descript-inspired document model for video editing.

Turns raw screen recordings into a structured alignment map of scene tokens,
enabling LLM-driven editing via natural language intent instead of manual
timestamp picking.

Pipeline:
  Record → analyze_video() → AlignmentMap → edit_by_intent() → EditPlan → render

The alignment map is the spine: every millisecond of video is covered by exactly
one SceneToken. The LLM operates on scene IDs and types, never raw timestamps.
Deterministic code below the LLM layer handles the actual video manipulation.
"""

import asyncio
import base64
import io
import json
import logging
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

from config import settings

logger = logging.getLogger(__name__)

# Models — Haiku for cheap frame classification, Sonnet for edit reasoning
_CLASSIFIER_MODEL = settings.HAIKU_MODEL
_EDITOR_MODEL = settings.SONNET_MODEL


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class SceneToken:
    """A contiguous segment of video with a single content type."""

    scene_id: str  # e.g. "sc_001"
    start_ms: int
    end_ms: int
    content_type: str  # static | animation | loading | transition | interaction
    description: str  # 1-line from Claude Vision
    quality_score: float = 1.0  # 0.0-1.0
    frame_indices: list[int] = field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "content_type": self.content_type,
            "description": self.description,
            "quality_score": self.quality_score,
        }


@dataclass
class AlignmentMap:
    """Ordered, contiguous, non-overlapping scene tokens covering the full video."""

    video_path: str
    duration_ms: int
    scenes: list[SceneToken]
    created_at: float = field(default_factory=time.time)

    def by_type(self, content_type: str) -> list[SceneToken]:
        return [s for s in self.scenes if s.content_type == content_type]

    def by_time_range(self, start_ms: int, end_ms: int) -> list[SceneToken]:
        return [s for s in self.scenes if s.start_ms < end_ms and s.end_ms > start_ms]

    def by_id(self, scene_id: str) -> SceneToken | None:
        return next((s for s in self.scenes if s.scene_id == scene_id), None)

    def summary(self) -> str:
        """Human-readable summary for LLM context."""
        lines = []
        for s in self.scenes:
            dur = s.duration_ms / 1000
            lines.append(
                f"  {s.scene_id}: [{s.start_ms / 1000:.1f}s-{s.end_ms / 1000:.1f}s] "
                f"({dur:.1f}s) {s.content_type} — {s.description} "
                f"(quality: {s.quality_score:.1f})"
            )
        return f"Video: {self.duration_ms / 1000:.1f}s, {len(self.scenes)} scenes\n" + "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "video_path": self.video_path,
            "duration_ms": self.duration_ms,
            "scenes": [s.to_dict() for s in self.scenes],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AlignmentMap":
        scenes = [
            SceneToken(
                scene_id=s["scene_id"],
                start_ms=s["start_ms"],
                end_ms=s["end_ms"],
                content_type=s["content_type"],
                description=s["description"],
                quality_score=s.get("quality_score", 1.0),
            )
            for s in d["scenes"]
        ]
        return cls(
            video_path=d["video_path"],
            duration_ms=d["duration_ms"],
            scenes=scenes,
            created_at=d.get("created_at", time.time()),
        )


@dataclass
class EditOp:
    """A single non-destructive edit operation against the alignment map."""

    op_type: str  # DELETE_SEGMENT | INSERT_PAUSE | REORDER | ADD_NARRATION | TRIM
    target_scenes: list[str]  # scene_ids
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "op_type": self.op_type,
            "target_scenes": self.target_scenes,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EditOp":
        return cls(
            op_type=d["op_type"],
            target_scenes=d["target_scenes"],
            params=d.get("params", {}),
        )


@dataclass
class EditPlan:
    """A set of edit operations against an alignment map, with LLM rationale."""

    alignment_map: AlignmentMap
    ops: list[EditOp]
    narrations: list[dict] = field(default_factory=list)  # [{"text", "scene_id", "position"}]
    rationale: str = ""

    def to_segments(self) -> tuple[list[dict], list[dict]]:
        """Flatten ops into segment list + narration list for cut_and_stitch / apply_style.

        Returns:
            (segments, narrations) where segments = [{"start": float, "end": float, "label": str}]
            and narrations = [{"text": str, "start": float, "end": float}]
        """
        # Start with all scenes in order
        working = list(self.alignment_map.scenes)

        # Apply operations
        for op in self.ops:
            targets = set(op.target_scenes)

            if op.op_type == "DELETE_SEGMENT":
                working = [s for s in working if s.scene_id not in targets]

            elif op.op_type == "TRIM":
                trim_start = op.params.get("trim_start_ms", 0)
                trim_end = op.params.get("trim_end_ms", 0)
                for i, s in enumerate(working):
                    if s.scene_id in targets:
                        working[i] = SceneToken(
                            scene_id=s.scene_id,
                            start_ms=s.start_ms + trim_start,
                            end_ms=s.end_ms - trim_end,
                            content_type=s.content_type,
                            description=s.description,
                            quality_score=s.quality_score,
                        )

            elif op.op_type == "REORDER":
                to_move = [s for s in working if s.scene_id in targets]
                working = [s for s in working if s.scene_id not in targets]
                pos = min(op.params.get("new_position", len(working)), len(working))
                for j, s in enumerate(to_move):
                    working.insert(pos + j, s)

            elif op.op_type == "ADD_NARRATION":
                self.narrations.append({
                    "text": op.params.get("text", ""),
                    "scene_id": op.target_scenes[0] if op.target_scenes else "",
                    "position": op.params.get("position", "overlay"),
                })

        # Convert to segment list (seconds)
        segments = []
        for s in working:
            if s.end_ms > s.start_ms:
                segments.append({
                    "start": s.start_ms / 1000,
                    "end": s.end_ms / 1000,
                    "label": s.description,
                })

        # Compute narration timestamps from final scene positions
        narration_texts = []
        edit_offset = 0.0
        scene_positions = {}
        for s in working:
            dur = (s.end_ms - s.start_ms) / 1000
            scene_positions[s.scene_id] = {"start": edit_offset, "end": edit_offset + dur}
            edit_offset += dur

        for nar in self.narrations:
            sid = nar.get("scene_id", "")
            pos = scene_positions.get(sid)
            if pos:
                narration_texts.append({
                    "text": nar["text"],
                    "start": pos["start"],
                    "end": pos["end"],
                })

        return segments, narration_texts


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def _extract_frames(video_path: str, interval_ms: int = 500) -> list[dict]:
    """Extract frames at fixed intervals, return as base64 JPEG with timestamps.

    Returns list of {"timestamp_ms": int, "index": int, "image_block": {...}}
    """
    from PIL import Image

    # Probe video
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if not result.stdout:
        return []
    data = json.loads(result.stdout)
    duration_s = float(data.get("format", {}).get("duration", 10))
    duration_ms = int(duration_s * 1000)

    frames = []
    idx = 0
    ts_ms = 0

    while ts_ms < duration_ms:
        ts_s = ts_ms / 1000

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            frame_path = f.name

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{ts_s:.3f}",
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            frame_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            Path(frame_path).unlink(missing_ok=True)
            ts_ms += interval_ms
            idx += 1
            continue

        try:
            img = Image.open(frame_path)
            max_w = 400
            if img.width > max_w:
                ratio = max_w / img.width
                img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=75)
            b64 = base64.b64encode(buf.getvalue()).decode()

            frames.append({
                "timestamp_ms": ts_ms,
                "index": idx,
                "image_block": {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                },
            })
        finally:
            Path(frame_path).unlink(missing_ok=True)

        ts_ms += interval_ms
        idx += 1

    logger.info("Extracted %d frames from %s (interval=%dms)", len(frames), video_path, interval_ms)
    return frames


# ---------------------------------------------------------------------------
# Scene classification (Claude Vision)
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM = """You are a video scene classifier for screen recordings of web applications.

<task>
For each frame, classify it into exactly one content type, provide a concise description
(10 words max), and assign a quality score (0.0-1.0 where 1.0 = crisp usable content).
</task>

<content_types>
<type name="static">
Stable UI with no visible change from the previous frame. Menus, text, forms at rest.
Also includes typewriter text effects where characters appear one by one — if text is still
typing in, classify as static (the content hasn't meaningfully changed, just more text appeared).
</type>
<type name="animation">
Content visibly moving — swiping cards, scrolling, animated transitions between states.
</type>
<type name="loading">
Spinner, skeleton loader, "connecting..." text, progress bars. Content is NOT ready.
Also includes black/empty frames where content has not loaded yet.
</type>
<type name="transition">
Brief page navigation or route change. The screen is between two stable states.
If a frame shows a brief flash of a home screen or completely different page sandwiched between
two related screens, classify it as transition with quality 0.2 — this is likely a recording
artifact from take stitching.
</type>
<type name="interaction">
Active user engagement visible — modal dialogs, button presses, form input, wallet popups.
If multiple consecutive frames show similar dialogs with slight changes (e.g. wallet setup
step 1, step 2, step 3), they are all interaction.
</type>
</content_types>

<output_format>
Return ONLY a JSON array, one object per frame:
[{"frame_index": 0, "content_type": "static", "description": "Welcome screen with app logo", "quality": 0.9}, ...]
</output_format>

<examples>
<example>
Frame shows a login page with all text fully loaded, no cursor blinking:
{"frame_index": 5, "content_type": "static", "description": "Login page with email and password fields", "quality": 0.95}
</example>
<example>
Frame shows a brief flash of the home screen between two wallet modal frames:
{"frame_index": 12, "content_type": "transition", "description": "Home screen flash between takes", "quality": 0.2}
</example>
<example>
Frame shows a black screen with a small spinner in the center:
{"frame_index": 8, "content_type": "loading", "description": "Black screen with loading spinner", "quality": 0.1}
</example>
</examples>"""


def _classify_frame_batch(frames: list[dict]) -> list[dict]:
    """Send a batch of frames to Claude Vision for classification."""
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    content = []
    content.append({
        "type": "text",
        "text": f"Classify these {len(frames)} frames from a screen recording:",
    })

    for frame in frames:
        content.append({
            "type": "text",
            "text": f"Frame {frame['index']} at t={frame['timestamp_ms']}ms:",
        })
        content.append(frame["image_block"])

    response = client.messages.create(
        model=_CLASSIFIER_MODEL,
        max_tokens=2048,
        system=_CLASSIFY_SYSTEM,
        messages=[{"role": "user", "content": content}],
    )

    text = response.content[0].text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\[[\s\S]*\]', text)
        if match:
            return json.loads(match.group())
        logger.error("Failed to parse classification response: %s", text[:200])
        return []


# ---------------------------------------------------------------------------
# Scene merging
# ---------------------------------------------------------------------------


def _merge_into_scenes(
    classifications: list[dict],
    frames: list[dict],
    interval_ms: int,
    duration_ms: int,
) -> list[SceneToken]:
    """Merge consecutive frames with the same content_type into SceneTokens."""
    if not classifications:
        return [SceneToken(
            scene_id="sc_001",
            start_ms=0,
            end_ms=duration_ms,
            content_type="static",
            description="unclassified video",
            quality_score=0.5,
        )]

    # Build frame-level lookup
    frame_map = {}
    for c in classifications:
        frame_map[c["frame_index"]] = c

    scenes = []
    scene_idx = 1
    current_type = None
    current_start = 0
    current_descriptions = []
    current_qualities = []
    current_frames = []

    for frame in frames:
        idx = frame["index"]
        ts = frame["timestamp_ms"]
        c = frame_map.get(idx, {
            "content_type": "static",
            "description": "unclassified",
            "quality": 0.5,
        })

        ctype = c.get("content_type", "static")

        if current_type is None:
            # First frame
            current_type = ctype
            current_start = ts
            current_descriptions.append(c.get("description", ""))
            current_qualities.append(c.get("quality", 0.5))
            current_frames.append(idx)
        elif ctype == current_type:
            # Same type — extend current scene
            current_descriptions.append(c.get("description", ""))
            current_qualities.append(c.get("quality", 0.5))
            current_frames.append(idx)
        else:
            # Type changed — finalize current scene
            scenes.append(SceneToken(
                scene_id=f"sc_{scene_idx:03d}",
                start_ms=current_start,
                end_ms=ts,
                content_type=current_type,
                description=current_descriptions[len(current_descriptions) // 2],  # median desc
                quality_score=sum(current_qualities) / len(current_qualities),
                frame_indices=current_frames,
            ))
            scene_idx += 1
            current_type = ctype
            current_start = ts
            current_descriptions = [c.get("description", "")]
            current_qualities = [c.get("quality", 0.5)]
            current_frames = [idx]

    # Finalize last scene
    if current_type is not None:
        scenes.append(SceneToken(
            scene_id=f"sc_{scene_idx:03d}",
            start_ms=current_start,
            end_ms=duration_ms,
            content_type=current_type,
            description=current_descriptions[len(current_descriptions) // 2],
            quality_score=sum(current_qualities) / len(current_qualities),
            frame_indices=current_frames,
        ))

    return scenes


# ---------------------------------------------------------------------------
# Smart compression passes (run after merging, before returning AlignmentMap)
# ---------------------------------------------------------------------------


def _compress_static_scenes(
    scenes: list[SceneToken], max_static_ms: int = 2500,
) -> list[SceneToken]:
    """Trim long static scenes to keep only the tail (full-text state).

    Typewriter text animations create long static scenes where the only change
    is more text appearing. We keep the last max_static_ms where text is fully
    rendered, trimming the slow buildup.
    """
    result = []
    for s in scenes:
        if s.content_type == "static" and s.duration_ms > max_static_ms:
            trimmed = SceneToken(
                scene_id=s.scene_id,
                start_ms=s.end_ms - max_static_ms,
                end_ms=s.end_ms,
                content_type=s.content_type,
                description=s.description,
                quality_score=s.quality_score,
                frame_indices=s.frame_indices,
            )
            logger.info(
                "Compressed static scene %s: %dms → %dms",
                s.scene_id, s.duration_ms, trimmed.duration_ms,
            )
            result.append(trimmed)
        else:
            result.append(s)
    return result


def _detect_seam_artifacts(
    scenes: list[SceneToken], flash_threshold_ms: int = 1200,
) -> list[SceneToken]:
    """Remove short low-quality scenes that are likely take-stitching artifacts.

    When two takes are stitched together, there's often a brief flash of the
    home screen or a loading frame at the seam. These are short (< 1.2s),
    low quality, and sandwiched between longer content scenes.
    """
    if len(scenes) < 3:
        return scenes

    result = []
    for i, s in enumerate(scenes):
        is_short = s.duration_ms < flash_threshold_ms
        is_low_quality = s.quality_score < 0.4
        is_artifact_type = s.content_type in ("transition", "loading")

        # Check if sandwiched between two longer scenes
        has_neighbors = (i > 0 and i < len(scenes) - 1)
        neighbors_longer = False
        if has_neighbors:
            prev_long = scenes[i - 1].duration_ms > flash_threshold_ms
            next_long = scenes[i + 1].duration_ms > flash_threshold_ms
            neighbors_longer = prev_long and next_long

        if is_short and (is_low_quality or is_artifact_type) and neighbors_longer:
            logger.info(
                "Removing seam artifact %s: %dms %s (quality=%.1f)",
                s.scene_id, s.duration_ms, s.content_type, s.quality_score,
            )
            continue

        result.append(s)

    # Re-number scene IDs
    for i, s in enumerate(result):
        s.scene_id = f"sc_{i + 1:03d}"

    return result


def _compress_repetitive_ui(
    scenes: list[SceneToken],
    max_repetitive_ms: int = 4000,
    min_group_size: int = 3,
) -> list[SceneToken]:
    """Compress consecutive interaction scenes with similar descriptions.

    Wallet connection flows, modal sequences, and multi-step dialogs often
    produce 4-6 consecutive interaction scenes that are nearly identical.
    Keep the first and last of each group, trim total duration to max_repetitive_ms.
    """
    def _word_overlap(a: str, b: str) -> float:
        """Simple word-overlap similarity between two descriptions."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union) if union else 0.0

    result = []
    i = 0
    while i < len(scenes):
        s = scenes[i]
        if s.content_type != "interaction":
            result.append(s)
            i += 1
            continue

        # Find run of consecutive interaction scenes
        group = [s]
        j = i + 1
        while j < len(scenes) and scenes[j].content_type == "interaction":
            group.append(scenes[j])
            j += 1

        if len(group) < min_group_size:
            result.extend(group)
            i = j
            continue

        # Check if descriptions are similar (repetitive UI flow)
        similarities = []
        for k in range(1, len(group)):
            similarities.append(_word_overlap(group[0].description, group[k].description))
        avg_sim = sum(similarities) / len(similarities) if similarities else 0

        if avg_sim < 0.3:
            # Not actually repetitive — keep all
            result.extend(group)
            i = j
            continue

        # Compress: keep first and last, trim durations
        total_ms = sum(g.duration_ms for g in group)
        if total_ms <= max_repetitive_ms:
            result.extend(group)
            i = j
            continue

        first = group[0]
        last = group[-1]
        # Split budget: 40% first, 60% last (entry context + completion state)
        first_budget = int(max_repetitive_ms * 0.4)
        last_budget = max_repetitive_ms - first_budget

        if first.duration_ms > first_budget:
            first = SceneToken(
                scene_id=first.scene_id,
                start_ms=first.start_ms,
                end_ms=first.start_ms + first_budget,
                content_type=first.content_type,
                description=first.description,
                quality_score=first.quality_score,
                frame_indices=first.frame_indices,
            )
        if last.duration_ms > last_budget:
            last = SceneToken(
                scene_id=last.scene_id,
                start_ms=last.end_ms - last_budget,
                end_ms=last.end_ms,
                content_type=last.content_type,
                description=last.description,
                quality_score=last.quality_score,
                frame_indices=last.frame_indices,
            )

        logger.info(
            "Compressed %d repetitive UI scenes (%dms → %dms): %s...%s",
            len(group), total_ms, first.duration_ms + last.duration_ms,
            first.description[:30], last.description[:30],
        )
        result.append(first)
        result.append(last)
        i = j

    # Re-number scene IDs
    for idx, s in enumerate(result):
        s.scene_id = f"sc_{idx + 1:03d}"

    return result


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------


def analyze_video(video_path: str, interval_ms: int = 500) -> AlignmentMap:
    """Analyze a video into a structured alignment map of scene tokens.

    Extracts frames at interval_ms, classifies each via Claude Vision,
    merges consecutive same-type frames into SceneTokens.
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Get duration
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration_s = float(json.loads(result.stdout).get("format", {}).get("duration", 10))
    duration_ms = int(duration_s * 1000)

    logger.info("Analyzing %s (%.1fs, interval=%dms)", video_path, duration_s, interval_ms)

    # Extract frames
    frames = _extract_frames(video_path, interval_ms)
    if not frames:
        raise RuntimeError(f"Could not extract frames from {video_path}")

    # Classify in batches of 12
    batch_size = 12
    all_classifications = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        logger.info("Classifying batch %d/%d (%d frames)",
                     i // batch_size + 1,
                     (len(frames) + batch_size - 1) // batch_size,
                     len(batch))
        classifications = _classify_frame_batch(batch)
        all_classifications.extend(classifications)

    # Merge into scenes
    scenes = _merge_into_scenes(all_classifications, frames, interval_ms, duration_ms)

    # Smart compression passes
    pre_count = len(scenes)
    scenes = _compress_static_scenes(scenes)
    scenes = _detect_seam_artifacts(scenes)
    scenes = _compress_repetitive_ui(scenes)
    if len(scenes) != pre_count:
        logger.info(
            "Smart compression: %d scenes → %d scenes",
            pre_count, len(scenes),
        )

    alignment_map = AlignmentMap(
        video_path=video_path,
        duration_ms=duration_ms,
        scenes=scenes,
    )

    logger.info(
        "Analysis complete: %d scenes from %d frames\n%s",
        len(scenes), len(frames), alignment_map.summary(),
    )
    return alignment_map


async def async_analyze_video(video_path: str, interval_ms: int = 500) -> AlignmentMap:
    """Async wrapper for analyze_video."""
    return await asyncio.to_thread(analyze_video, video_path, interval_ms)


# ---------------------------------------------------------------------------
# Intent-to-EditOps translation (Sonnet)
# ---------------------------------------------------------------------------

_EDIT_INTENT_SYSTEM = """You are a video editor AI that translates natural language editing instructions
into structured edit operations against a scene-by-scene breakdown.

<available_operations>
<op name="DELETE_SEGMENT">
Remove scenes entirely. Use for loading screens, errors, dead time.
Example: {"op_type": "DELETE_SEGMENT", "target_scenes": ["sc_003", "sc_004"], "params": {}}
</op>
<op name="TRIM">
Trim milliseconds from the start/end of a scene. Use for cutting partial dead time.
Example: {"op_type": "TRIM", "target_scenes": ["sc_002"], "params": {"trim_start_ms": 500, "trim_end_ms": 0}}
</op>
<op name="REORDER">
Move scenes to a new position (0-indexed). Use for rearranging flow.
Example: {"op_type": "REORDER", "target_scenes": ["sc_007"], "params": {"new_position": 2}}
</op>
<op name="ADD_NARRATION">
Add text overlay on a scene. Use for explanatory captions.
Example: {"op_type": "ADD_NARRATION", "target_scenes": ["sc_001"], "params": {"text": "Welcome to the app", "position": "overlay"}}
</op>
</available_operations>

<rules>
- Delete ALL loading scenes unless the user specifically says to keep them.
- Static scenes have already been auto-trimmed to 2.5s max. Do NOT trim them further
  unless the user explicitly asks — they need at least 2s to be readable.
- Short seam artifacts and repetitive UI sequences have already been auto-compressed.
- Keep interaction and animation scenes — these are the interesting parts.
- Preserve chronological order unless explicitly asked to reorder.
- If the video still feels too long, prefer deleting entire scenes over trimming partial seconds.
- Add narration to key moments when the user asks for captions or narration.
</rules>

<output_format>
Return ONLY a JSON object:
{"ops": [list of EditOp objects], "rationale": "Brief explanation of what you did and why"}
</output_format>

<example>
User says: "Remove loading screens and add narration to the wallet section"
Scene breakdown shows sc_003 is loading, sc_007 is wallet interaction.
Response:
{"ops": [
  {"op_type": "DELETE_SEGMENT", "target_scenes": ["sc_003"], "params": {}},
  {"op_type": "ADD_NARRATION", "target_scenes": ["sc_007"], "params": {"text": "Connect your wallet to get started", "position": "overlay"}}
], "rationale": "Removed loading screen and added wallet narration per user request"}
</example>"""


def translate_intent(alignment_map: AlignmentMap, intent: str) -> EditPlan:
    """Use Sonnet to translate a natural language editing intent into EditOps."""
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    response = client.messages.create(
        model=_EDITOR_MODEL,
        max_tokens=2048,
        system=_EDIT_INTENT_SYSTEM,
        messages=[{
            "role": "user",
            "content": (
                f"## Scene Breakdown\n\n{alignment_map.summary()}\n\n"
                f"## Edit Instruction\n\n{intent}"
            ),
        }],
    )

    text = response.content[0].text.strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            result = json.loads(match.group())
        else:
            logger.error("Failed to parse edit intent response: %s", text[:300])
            return EditPlan(
                alignment_map=alignment_map,
                ops=[],
                rationale="Failed to parse LLM response",
            )

    ops = [EditOp.from_dict(op) for op in result.get("ops", [])]
    rationale = result.get("rationale", "")

    plan = EditPlan(
        alignment_map=alignment_map,
        ops=ops,
        rationale=rationale,
    )

    logger.info("Edit plan: %d ops — %s", len(ops), rationale)
    return plan


async def async_translate_intent(alignment_map: AlignmentMap, intent: str) -> EditPlan:
    """Async wrapper for translate_intent."""
    return await asyncio.to_thread(translate_intent, alignment_map, intent)


# ---------------------------------------------------------------------------
# Full pipeline: analyze → intent → render
# ---------------------------------------------------------------------------


def execute_edit(
    video_path: str,
    alignment_map: AlignmentMap,
    intent: str,
    apply_style: bool = True,
    crossfade_duration: float = 0.15,
    output_path: str | None = None,
) -> dict:
    """Full pipeline: translate intent → flatten to segments → render.

    Returns dict with output_path, segments, narrations, rationale.
    """
    from agent.video_styler import cut_and_stitch, apply_style as style_video, VideoStyle

    # Translate intent to edit plan
    plan = translate_intent(alignment_map, intent)
    segments, narrations = plan.to_segments()

    if not segments:
        return {
            "error": "Edit plan produced no segments — nothing to render",
            "rationale": plan.rationale,
            "ops": [op.to_dict() for op in plan.ops],
        }

    # Cut and stitch
    edited_path = cut_and_stitch(
        video_path, segments,
        crossfade_duration=crossfade_duration,
    )

    # Optionally apply phone mockup + gradient
    if apply_style:
        final_path = style_video(
            edited_path,
            style=VideoStyle(frame_mode="phone"),
            narration_texts=narrations if narrations else None,
            output_path=output_path,
        )
    else:
        final_path = edited_path

    return {
        "output_path": final_path,
        "segments_used": len(segments),
        "narrations": narrations,
        "scenes_kept": len(segments),
        "scenes_removed": len(alignment_map.scenes) - len(segments),
        "rationale": plan.rationale,
        "ops": [op.to_dict() for op in plan.ops],
    }


async def async_execute_edit(
    video_path: str,
    alignment_map: AlignmentMap,
    intent: str,
    apply_style: bool = True,
    crossfade_duration: float = 0.15,
    output_path: str | None = None,
) -> dict:
    """Async wrapper for execute_edit."""
    return await asyncio.to_thread(
        execute_edit, video_path, alignment_map, intent,
        apply_style, crossfade_duration, output_path,
    )
