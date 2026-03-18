"""FastAPI backend for the BrandMover dashboard."""

import sys
from pathlib import Path

# Ensure project root is importable
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from dashboard.backend.routes import calendar, status, documents, campaigns, settings

app = FastAPI(title="BrandMover Dashboard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount route modules
app.include_router(calendar.router)
app.include_router(status.router)
app.include_router(documents.router)
app.include_router(campaigns.router)
app.include_router(settings.router)

# Serve generated images and outputs as static files
_state_dir = _project_root / "state"
_images_dir = _state_dir / "images"
_outputs_dir = _state_dir / "outputs"

if _images_dir.exists():
    app.mount("/static/images", StaticFiles(directory=str(_images_dir)), name="images")
if _outputs_dir.exists():
    app.mount("/static/outputs", StaticFiles(directory=str(_outputs_dir)), name="outputs")
