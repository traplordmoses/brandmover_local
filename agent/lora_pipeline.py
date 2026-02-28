"""
LoRA training data collection + Replicate training trigger.

Approved images are accumulated in brand/training_data/. When the threshold
is reached, /train_lora zips them and kicks off ostris/flux-dev-lora-trainer.
"""

import asyncio
import io
import json
import logging
import shutil
import time
import zipfile
from pathlib import Path

import httpx

from config import settings

logger = logging.getLogger(__name__)

_TRAINING_DIR = Path(settings.BRAND_FOLDER) / "training_data"
_IMAGES_DIR = _TRAINING_DIR / "images"
_MANIFEST_PATH = _TRAINING_DIR / "manifest.json"

_DEFAULT_THRESHOLD = 20
_DEFAULT_TRIGGER_WORD = "BRAND3D"

from agent.content_types import LORA_ELIGIBLE_TYPES as _ELIGIBLE_CONTENT_TYPES


def _ensure_dirs() -> None:
    _TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    _IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def _load_manifest() -> dict:
    _ensure_dirs()
    if _MANIFEST_PATH.exists():
        try:
            return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"images": [], "versions": [], "threshold": _DEFAULT_THRESHOLD}


def _save_manifest(manifest: dict) -> None:
    _ensure_dirs()
    _MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def add_training_image(
    image_path: str, prompt: str, content_type: str,
) -> tuple[int, bool]:
    """Copy an image into the training set and record it in the manifest.

    Returns:
        (total_image_count, threshold_hit) — threshold_hit is True when
        the count just reached the training threshold.
    """
    if content_type not in _ELIGIBLE_CONTENT_TYPES:
        return len(_load_manifest()["images"]), False

    manifest = _load_manifest()
    _ensure_dirs()

    src = Path(image_path)
    if not src.exists():
        logger.warning("Training image not found: %s", image_path)
        return len(manifest["images"]), False

    ts = int(time.time())
    dest_name = f"{content_type}_{ts}{src.suffix}"
    dest = _IMAGES_DIR / dest_name
    shutil.copy2(str(src), str(dest))

    manifest["images"].append({
        "filename": dest_name,
        "prompt": prompt,
        "content_type": content_type,
        "added_at": ts,
    })
    _save_manifest(manifest)

    count = len(manifest["images"])
    threshold = manifest.get("threshold", _DEFAULT_THRESHOLD)
    threshold_hit = count == threshold

    logger.info("Training image added: %s (%d total, threshold=%d)", dest_name, count, threshold)
    return count, threshold_hit


async def add_training_image_from_url(
    url: str, prompt: str, content_type: str,
) -> tuple[int, bool]:
    """Download an image from URL and add it to the training set."""
    if content_type not in _ELIGIBLE_CONTENT_TYPES:
        return len(_load_manifest()["images"]), False

    _ensure_dirs()
    ts = int(time.time())
    tmp_path = _IMAGES_DIR / f"_dl_{ts}.jpg"

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            tmp_path.write_bytes(resp.content)
    except Exception as e:
        logger.warning("Failed to download training image from %s: %s", url[:80], e)
        return len(_load_manifest()["images"]), False

    count, hit = add_training_image(str(tmp_path), prompt, content_type)

    # Clean up the temp download (add_training_image already copied it)
    try:
        tmp_path.unlink(missing_ok=True)
    except OSError:
        pass

    return count, hit


def get_training_stats() -> dict:
    """Return training set stats for display."""
    manifest = _load_manifest()
    lora_manifest = _load_lora_manifest()
    return {
        "total_images": len(manifest["images"]),
        "threshold": manifest.get("threshold", _DEFAULT_THRESHOLD),
        "versions": manifest.get("versions", []),
        "lora_manifest": lora_manifest,
    }


_LORA_DIR = Path(settings.BRAND_FOLDER) / "loras"
_LORA_MANIFEST_PATH = _LORA_DIR / "manifest.json"
_ACTIVE_WEIGHTS = _LORA_DIR / "brand3d.safetensors"
_POLL_INTERVAL = 30  # seconds
_POLL_TIMEOUT = 90 * 60  # 90 minutes


# ---------------------------------------------------------------------------
# LoRA version manifest (brand/loras/manifest.json)
# ---------------------------------------------------------------------------

def _load_lora_manifest() -> dict:
    """Load the LoRA versions manifest from brand/loras/manifest.json."""
    _LORA_DIR.mkdir(parents=True, exist_ok=True)
    if _LORA_MANIFEST_PATH.exists():
        try:
            return json.loads(_LORA_MANIFEST_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"versions": [], "active_version": None}


def _save_lora_manifest(manifest: dict) -> None:
    _LORA_DIR.mkdir(parents=True, exist_ok=True)
    _LORA_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def get_lora_manifest() -> dict:
    """Public accessor for the LoRA versions manifest."""
    return _load_lora_manifest()


def _next_version_number(manifest: dict) -> int:
    """Determine the next version number from the manifest."""
    versions = manifest.get("versions", [])
    if not versions:
        return 1
    return max(v.get("version_number", 0) for v in versions) + 1


def _record_version(
    version_number: int,
    prediction_id: str,
    image_count: int,
    content_types: list[str],
    trigger_word: str,
    file_path: str,
    weights_url: str = "",
    status: str = "active",
) -> dict:
    """Add a completed version to the lora manifest and set it as active."""
    manifest = _load_lora_manifest()

    # Deactivate the currently active version
    for v in manifest.get("versions", []):
        if v.get("status") == "active":
            v["status"] = "inactive"

    version_record = {
        "version_number": version_number,
        "version_name": f"v{version_number}",
        "prediction_id": prediction_id,
        "training_date": int(time.time()),
        "image_count": image_count,
        "content_types": content_types,
        "file_path": file_path,
        "weights_url": weights_url,
        "trigger_word": trigger_word,
        "status": status,
    }
    manifest.setdefault("versions", []).append(version_record)
    manifest["active_version"] = version_number

    _save_lora_manifest(manifest)
    logger.info("LoRA version v%d recorded as active", version_number)
    return version_record


def switch_active_version(version_number: int) -> dict | str:
    """Switch the active LoRA to a specific version.

    Copies the version's weights file to brand3d.safetensors and updates
    the manifest.

    Returns the version record on success, or an error string on failure.
    """
    manifest = _load_lora_manifest()
    versions = manifest.get("versions", [])

    target = None
    for v in versions:
        if v.get("version_number") == version_number:
            target = v
            break

    if target is None:
        available = [v.get("version_number") for v in versions]
        return f"Version {version_number} not found. Available: {available}"

    # Check that the weights file exists
    weights_path = Path(target["file_path"])
    if not weights_path.exists():
        return f"Weights file not found: {weights_path}"

    # Copy weights to active path
    shutil.copy2(str(weights_path), str(_ACTIVE_WEIGHTS))

    # Update statuses in manifest
    for v in versions:
        if v.get("status") == "active":
            v["status"] = "inactive"
    target["status"] = "active"
    manifest["active_version"] = version_number

    _save_lora_manifest(manifest)
    logger.info("Switched active LoRA to v%d", version_number)
    return target


def rollback_version() -> dict | str:
    """Roll back to the previous LoRA version (N-1).

    Returns the version record on success, or an error string on failure.
    """
    manifest = _load_lora_manifest()
    active = manifest.get("active_version")
    versions = manifest.get("versions", [])

    if not versions:
        return "No LoRA versions available."

    if active is None:
        return "No active version to roll back from."

    # Find the previous version
    prev_version = active - 1
    if prev_version < 1:
        return "Already at version 1 — no earlier version to roll back to."

    return switch_active_version(prev_version)


def format_versions_list(manifest: dict) -> str:
    """Format the LoRA versions manifest into a readable list."""
    versions = manifest.get("versions", [])
    active = manifest.get("active_version")

    if not versions:
        return "No LoRA versions trained yet."

    lines: list[str] = []
    for v in versions:
        vn = v.get("version_number", "?")
        status = v.get("status", "unknown")
        image_count = v.get("image_count", "?")
        date_ts = v.get("training_date", 0)

        # Format date
        if date_ts:
            import datetime
            dt = datetime.datetime.fromtimestamp(date_ts)
            date_str = dt.strftime("%Y-%m-%d %H:%M")
        else:
            date_str = "unknown"

        # Active indicator
        active_marker = " (active)" if vn == active else ""
        status_icon = {
            "active": "\u2705",
            "inactive": "\u2796",
        }.get(status, "\u2753")

        lines.append(
            f"  {status_icon} v{vn}{active_marker} — "
            f"{date_str}, {image_count} images"
        )

    return "\n".join(lines)


async def poll_training(prediction_id: str) -> dict:
    """Poll Replicate for training prediction status until terminal state.

    Returns the final prediction dict with status succeeded/failed/canceled.
    Timeout after 90 minutes.
    """
    headers = {"Authorization": f"Bearer {settings.REPLICATE_API_TOKEN}"}
    url = f"https://api.replicate.com/v1/predictions/{prediction_id}"
    deadline = time.time() + _POLL_TIMEOUT

    async with httpx.AsyncClient(timeout=30) as client:
        while time.time() < deadline:
            try:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                status = data.get("status", "")
                if status in ("succeeded", "failed", "canceled"):
                    logger.info("Training %s completed with status: %s", prediction_id, status)
                    return data
                logger.debug("Training %s status: %s", prediction_id, status)
            except Exception as e:
                logger.warning("Poll error for %s: %s", prediction_id, e)
            await asyncio.sleep(_POLL_INTERVAL)

    return {"status": "timeout", "error": "Polling timed out after 90 minutes"}


async def download_weights(
    output_url: str,
    prediction_id: str = "",
    image_count: int = 0,
    content_types: list[str] | None = None,
    trigger_word: str = _DEFAULT_TRIGGER_WORD,
) -> Path:
    """Download trained LoRA weights to brand/loras/.

    Saves as brand3d_v{N}.safetensors and copies to brand3d.safetensors (active).
    Records the version in brand/loras/manifest.json.
    """
    _LORA_DIR.mkdir(parents=True, exist_ok=True)

    # Determine version number from lora manifest
    lora_manifest = _load_lora_manifest()
    version_num = _next_version_number(lora_manifest)

    versioned_path = _LORA_DIR / f"brand3d_v{version_num}.safetensors"

    async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
        resp = await client.get(output_url)
        resp.raise_for_status()
        data = resp.content

    versioned_path.write_bytes(data)
    shutil.copy2(str(versioned_path), str(_ACTIVE_WEIGHTS))

    # Record in lora manifest
    _record_version(
        version_number=version_num,
        prediction_id=prediction_id,
        image_count=image_count,
        content_types=content_types or [],
        trigger_word=trigger_word,
        file_path=str(versioned_path),
        weights_url=output_url,
    )

    logger.info(
        "LoRA weights downloaded: %s (%.1f MB) + active copy %s",
        versioned_path, len(data) / 1024 / 1024, _ACTIVE_WEIGHTS,
    )
    return _ACTIVE_WEIGHTS


def get_active_lora() -> dict | None:
    """Return the active LoRA version info, or None.

    Checks brand/loras/manifest.json first (versioned), then falls back to
    the training manifest for backwards compatibility.

    Returns: {"version": str, "model_url": str, "trigger_word": str,
              "weights_path": str} or None.
    """
    # Check lora manifest (versioned system)
    lora_manifest = _load_lora_manifest()
    for v in lora_manifest.get("versions", []):
        if v.get("status") == "active":
            return {
                "version": v.get("version_name", f"v{v.get('version_number', '?')}"),
                "model_url": v.get("weights_url", ""),
                "trigger_word": v.get("trigger_word", _DEFAULT_TRIGGER_WORD),
                "weights_path": v.get("file_path", ""),
            }

    # Fallback: check training manifest (pre-versioning)
    manifest = _load_manifest()
    versions = manifest.get("versions", [])
    for v in reversed(versions):
        if v.get("status") == "completed":
            return {
                "version": v.get("version", ""),
                "model_url": v.get("weights_url", ""),
                "trigger_word": v.get("trigger_word", _DEFAULT_TRIGGER_WORD),
                "weights_path": v.get("weights_path", ""),
            }
    return None


async def _background_poll(
    prediction_id: str,
    version_name: str,
    bot=None,
    chat_id: int = 0,
) -> None:
    """Background task: poll → download → update manifest → notify."""
    try:
        result = await poll_training(prediction_id)
        status = result.get("status", "failed")

        manifest = _load_manifest()
        for v in manifest.get("versions", []):
            if v.get("prediction_id") == prediction_id:
                v["status"] = status
                v["completed_at"] = int(time.time())
                if status == "succeeded":
                    # Extract output URL (weights)
                    output = result.get("output")
                    output_url = None
                    if isinstance(output, str):
                        output_url = output
                    elif isinstance(output, dict):
                        output_url = output.get("weights") or output.get("url", "")
                    elif isinstance(output, list) and output:
                        output_url = str(output[0])

                    if output_url:
                        # Collect content types from training images
                        train_manifest = _load_manifest()
                        ct_set = set()
                        for img in train_manifest.get("images", []):
                            ct = img.get("content_type", "")
                            if ct:
                                ct_set.add(ct)

                        weights_path = await download_weights(
                            output_url,
                            prediction_id=prediction_id,
                            image_count=v.get("image_count", 0),
                            content_types=sorted(ct_set),
                            trigger_word=v.get("trigger_word", _DEFAULT_TRIGGER_WORD),
                        )
                        v["weights_url"] = output_url
                        v["weights_path"] = str(weights_path)
                break
        _save_manifest(manifest)

        # Notify via Telegram
        if bot and chat_id:
            try:
                if status == "succeeded":
                    await bot.send_message(
                        chat_id=chat_id,
                        text=(
                            f"LoRA training complete!\n\n"
                            f"Version: {version_name}\n"
                            f"Weights downloaded to brand/loras/\n"
                            f"Use /lora_status to see details."
                        ),
                    )
                else:
                    error = result.get("error", "Unknown error")
                    await bot.send_message(
                        chat_id=chat_id,
                        text=f"LoRA training {status}: {version_name}\n{error}",
                    )
            except Exception as e:
                logger.warning("Failed to send training notification: %s", e)

    except Exception as e:
        logger.error("Background poll failed for %s: %s", prediction_id, e)
        # Update manifest with failure
        manifest = _load_manifest()
        for v in manifest.get("versions", []):
            if v.get("prediction_id") == prediction_id:
                v["status"] = "failed"
                v["error"] = str(e)
                break
        _save_manifest(manifest)


async def trigger_training(
    trigger_word: str = _DEFAULT_TRIGGER_WORD,
    bot=None,
    chat_id: int = 0,
) -> dict:
    """Zip training images, upload to Replicate, and start LoRA training.

    Returns:
        {"version": str, "prediction_id": str, "image_count": int,
         "trigger_word": str} on success, or {"error": str} on failure.
    """
    if not settings.REPLICATE_API_TOKEN:
        return {"error": "REPLICATE_API_TOKEN not set"}

    manifest = _load_manifest()
    images = manifest["images"]
    threshold = manifest.get("threshold", _DEFAULT_THRESHOLD)

    if len(images) < threshold:
        return {"error": f"Need {threshold} images, have {len(images)}"}

    # Determine version number
    versions = manifest.get("versions", [])
    version_num = len(versions) + 1
    version_name = f"v{version_num}"

    # Zip all training images
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for entry in images:
            img_path = _IMAGES_DIR / entry["filename"]
            if img_path.exists():
                zf.write(str(img_path), entry["filename"])
                # Include caption file for the trainer
                caption_name = Path(entry["filename"]).stem + ".txt"
                zf.writestr(caption_name, entry.get("prompt", ""))
    zip_buffer.seek(0)
    zip_bytes = zip_buffer.getvalue()

    logger.info("Training zip created: %d bytes, %d images", len(zip_bytes), len(images))

    headers = {
        "Authorization": f"Bearer {settings.REPLICATE_API_TOKEN}",
    }

    # Upload zip to Replicate files API
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            upload_resp = await client.post(
                "https://api.replicate.com/v1/files",
                headers=headers,
                files={"content": (f"training_{version_name}.zip", zip_bytes, "application/zip")},
            )
            upload_resp.raise_for_status()
            file_data = upload_resp.json()
            file_url = file_data.get("urls", {}).get("get", "")
            if not file_url:
                return {"error": "Upload succeeded but no file URL returned"}
            logger.info("Training zip uploaded: %s", file_url)
    except Exception as e:
        return {"error": f"File upload failed: {e}"}

    # Start training via Replicate
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            train_resp = await client.post(
                "https://api.replicate.com/v1/models/ostris/flux-dev-lora-trainer/predictions",
                headers={**headers, "Content-Type": "application/json"},
                json={
                    "input": {
                        "input_images": file_url,
                        "trigger_word": trigger_word,
                        "steps": 1000,
                        "learning_rate": 0.0004,
                        "batch_size": 1,
                        "resolution": "512,768,1024",
                        "autocaption": True,
                    },
                },
            )
            train_resp.raise_for_status()
            train_data = train_resp.json()
            prediction_id = train_data.get("id", "unknown")
    except Exception as e:
        return {"error": f"Training request failed: {e}"}

    # Record version in manifest
    version_record = {
        "version": version_name,
        "prediction_id": prediction_id,
        "image_count": len(images),
        "trigger_word": trigger_word,
        "started_at": int(time.time()),
        "status": "training",
    }
    manifest.setdefault("versions", []).append(version_record)
    _save_manifest(manifest)

    logger.info("LoRA training started: %s (prediction=%s)", version_name, prediction_id)

    # Launch background polling task
    asyncio.create_task(_background_poll(prediction_id, version_name, bot, chat_id))
    logger.info("Background poll task launched for %s", prediction_id)

    return {
        "version": version_name,
        "prediction_id": prediction_id,
        "image_count": len(images),
        "trigger_word": trigger_word,
    }
