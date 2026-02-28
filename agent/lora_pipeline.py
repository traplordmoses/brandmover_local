"""
LoRA training data collection + Replicate training trigger.

Approved images are accumulated in brand/training_data/. When the threshold
is reached, /train_lora zips them and kicks off ostris/flux-dev-lora-trainer.
"""

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


async def trigger_training(trigger_word: str = _DEFAULT_TRIGGER_WORD) -> dict:
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

    return {
        "version": version_name,
        "prediction_id": prediction_id,
        "image_count": len(images),
        "trigger_word": trigger_word,
    }
