"""Brand documents endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from dashboard.backend.services import data_bridge as db

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("")
def list_documents():
    return {"files": db.list_brand_documents()}


@router.get("/content")
def read_document(path: str):
    content = db.read_brand_document(path)
    if content is None:
        raise HTTPException(status_code=404, detail="Document not found or outside brand directory")
    return {"path": path, "content": content}


class WriteDocBody(BaseModel):
    path: str
    content: str


@router.put("/content")
def write_document(body: WriteDocBody):
    if db.write_brand_document(body.path, body.content):
        return {"ok": True}
    raise HTTPException(status_code=400, detail="Path outside brand directory")
