"""FastAPI backend for the BrandMover dashboard."""

import os
import sys
from pathlib import Path

# Ensure project root is importable
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from dashboard.backend.routes import calendar, status, documents, campaigns, settings
from dashboard.backend.routes.design import router as design_router

app = FastAPI(title="BrandMover Dashboard API", version="1.0.0")

_DASHBOARD_API_KEY = os.environ.get("DASHBOARD_API_KEY")
_LOCALHOST_IPS = {"127.0.0.1", "::1"}


_TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # Telegram Mini App auth — validate initData HMAC
    tg_init_data = request.headers.get("X-Telegram-InitData")
    if tg_init_data and _TELEGRAM_BOT_TOKEN:
        from dashboard.backend.services.auth import validate_telegram_init_data
        user = validate_telegram_init_data(tg_init_data, _TELEGRAM_BOT_TOKEN)
        if user:
            request.state.telegram_user = user
            return await call_next(request)
        return JSONResponse(status_code=403, content={"detail": "Invalid Telegram auth"})

    if _DASHBOARD_API_KEY:
        # API-key mode: every request must carry the correct key
        if request.headers.get("X-API-Key") != _DASHBOARD_API_KEY:
            return JSONResponse(status_code=403, content={"detail": "Invalid or missing API key"})
    else:
        # No key configured — restrict to localhost only
        client_ip = request.client.host if request.client else None
        if client_ip not in _LOCALHOST_IPS:
            return JSONResponse(status_code=403, content={"detail": "Remote access denied; set DASHBOARD_API_KEY to enable"})
    return await call_next(request)

# CORS: default origins + any extras from DASHBOARD_CORS_ORIGINS env var
_default_origins = ["http://localhost:5173"]
_extra_origins = os.environ.get("DASHBOARD_CORS_ORIGINS", "")
_all_origins = _default_origins + [
    o.strip() for o in _extra_origins.split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_all_origins,
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
app.include_router(design_router)

# Serve generated images and outputs as static files
_state_dir = _project_root / "state"
_images_dir = _state_dir / "images"
_outputs_dir = _state_dir / "outputs"
_brand_dir = _project_root / "brand"

if _images_dir.exists():
    app.mount("/static/images", StaticFiles(directory=str(_images_dir)), name="images")
if _outputs_dir.exists():
    app.mount("/static/outputs", StaticFiles(directory=str(_outputs_dir)), name="outputs")
if _brand_dir.exists():
    app.mount("/static/brand", StaticFiles(directory=str(_brand_dir)), name="brand_assets")
