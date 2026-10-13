"""Local-only document research demo (not a public multi-user SaaS)."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from service.core import DocumentQASystem, MAX_BYTES

system = DocumentQASystem()


class Question(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    top_k: int = Field(default=3, ge=1, le=5)


app = FastAPI(
    title="NeuroEvoRAG Local Document Q&A",
    version="0.1.0",
    description="Source-bound extractive reader; no cloud LLM required. Local research use only.",
)


def translate_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc.args[0]))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=500, detail="Internal processing error")


@app.get("/")
def home():
    return FileResponse(Path(__file__).with_name("index.html"), media_type="text/html")


@app.get("/health")
def health():
    return {"status": "ok", "local_only": True, "public_authentication_configured": False,
            "reader": "pretrained DistilBERT, lazily loaded on first question"}


@app.post("/api/workspaces", status_code=201)
def create_workspace():
    return {"workspace_id": system.create_workspace(), "ephemeral": True}


@app.get("/api/documents")
def list_documents(x_workspace_id: str = Header(...)):
    try:
        return {"documents": system.list_documents(x_workspace_id)}
    except (KeyError, ValueError) as exc:
        raise translate_error(exc) from exc


@app.post("/api/documents", status_code=201)
async def upload_document(file: UploadFile = File(...),
                          x_workspace_id: str = Header(...)):
    # Enforce hard request budget regardless of Content-Length or client claims.
    raw = await file.read(MAX_BYTES + 1)
    try:
        return system.add_document(x_workspace_id, file.filename or "", raw)
    except (KeyError, ValueError) as exc:
        raise translate_error(exc) from exc
    finally:
        await file.close()


@app.delete("/api/documents/{document_id}", status_code=204)
def remove_document(document_id: str, x_workspace_id: str = Header(...)):
    try:
        system.remove_document(x_workspace_id, document_id)
    except (KeyError, ValueError) as exc:
        raise translate_error(exc) from exc


@app.post("/api/ask")
def ask(payload: Question, x_workspace_id: str = Header(...)):
    try:
        return system.ask(x_workspace_id, payload.question, payload.top_k)
    except (KeyError, ValueError) as exc:
        raise translate_error(exc) from exc
