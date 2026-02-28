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
    return {
        "total_images": len(manifest["images"]),
        "threshold": manifest.get("threshold", _DEFAULT_THRESHOLD),
        "versions": manifest.get("versions", []),
    }


_LORA_DIR = Path(settings.BRAND_FOLDER) / "loras"
_POLL_INTERVAL = 30  # seconds
_POLL_TIMEOUT = 90 * 60  # 90 minutes


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


async def download_weights(output_url: str) -> Path:
    """Download trained LoRA weights to brand/loras/.

    Saves as brand3d.safetensors (active) + versioned copy brand3d_v{N}.safetensors.
    """
    _LORA_DIR.mkdir(parents=True, exist_ok=True)

    # Determine version number from existing files
    existing = sorted(_LORA_DIR.glob("brand3d_v*.safetensors"))
    version_num = len(existing) + 1

    active_path = _LORA_DIR / "brand3d.safetensors"
    versioned_path = _LORA_DIR / f"brand3d_v{version_num}.safetensors"

    async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
        resp = await client.get(output_url)
        resp.raise_for_status()
        data = resp.content

    versioned_path.write_bytes(data)
    shutil.copy2(str(versioned_path), str(active_path))

    logger.info(
        "LoRA weights downloaded: %s (%.1f MB) + active copy %s",
        versioned_path, len(data) / 1024 / 1024, active_path,
    )
    return active_path


def get_active_lora() -> dict | None:
    """Return the latest completed LoRA version info, or None.

    Returns: {"version": str, "model_url": str, "trigger_word": str,
              "weights_path": str} or None.
    """
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
                        weights_path = await download_weights(output_url)
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
