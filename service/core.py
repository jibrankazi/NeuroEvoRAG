"""Bounded local document QA with verbatim source citations.

In-memory per-workspace indexes; workspace IDs are unguessable capabilities but
not user authentication. Do not publish this development app without adding
authentication, rate limiting, storage isolation, and data retention controls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
import re
import secrets
import threading
from typing import Any

from benchmarks.real_squad_qa import PretrainedExtractiveReader, TfidfOriginalCorpus

MAX_BYTES = 1_048_576
MAX_CHARS = 150_000
MAX_DOCUMENTS = 12
MAX_CHUNKS = 1000
CHUNK_WORDS = 170


def extract_document(filename: str, data: bytes) -> str:
    """Decode text/Markdown/PDF; reject binary, scans and oversized uploads."""
    suffix = Path(filename).suffix.lower()
    if len(data) > MAX_BYTES or not data:
        raise ValueError("Document must contain 1 byte to 1 MiB")
    if suffix in {".txt", ".md"}:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("Text files must use UTF-8") from exc
    elif suffix == ".pdf":
        if not data.startswith(b"%PDF-"):
            raise ValueError("Invalid PDF header")
        from pypdf import PdfReader
        try:
            reader = PdfReader(BytesIO(data), strict=True)
            if len(reader.pages) > 30 or reader.is_encrypted:
                raise ValueError("Only unencrypted PDFs with up to 30 pages are supported")
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("PDF could not be parsed; scanned PDFs need OCR (not supported)") from exc
    else:
        raise ValueError("Supported types: .txt, .md, .pdf")
    if len(text) > MAX_CHARS:
        raise ValueError("Extracted document exceeds 150,000 characters")
    if not text.strip():
        raise ValueError("Document has no selectable text")
    if "\x00" in text:
        raise ValueError("Binary content is not allowed")
    return text


def source_chunks(text: str, document_id: str, filename: str) -> list[dict[str, Any]]:
    """Use original source substring positions for independently verifiable quotes."""
    words = list(re.finditer(r"\S+", text))
    chunks = []
    for start in range(0, len(words), CHUNK_WORDS):
        group = words[start:start + CHUNK_WORDS]
        begin, end = group[0].start(), group[-1].end()
        chunks.append({
            "document_id": document_id,
            "filename": filename,
            "begin": begin,
            "end": end,
            "title": filename,
            "context": text[begin:end],
        })
    return chunks


@dataclass
class Workspace:
    documents: dict[str, tuple[str, str]] = field(default_factory=dict)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    retriever: Any = None


class DocumentQASystem:
    def __init__(self, reader=None):
        self.workspaces: dict[str, Workspace] = {}
        self.reader = reader
        self.lock = threading.RLock()

    def create_workspace(self) -> str:
        with self.lock:
            token = secrets.token_urlsafe(24)
            self.workspaces[token] = Workspace()
            return token

    def _get(self, workspace_id: str) -> Workspace:
        if not workspace_id or workspace_id not in self.workspaces:
            raise KeyError("Unknown workspace. Create a new one; in-memory data does not survive restart.")
        return self.workspaces[workspace_id]

    def list_documents(self, workspace_id: str) -> list[dict[str, Any]]:
        with self.lock:
            ws = self._get(workspace_id)
            return [{"id": ident, "filename": filename, "characters": len(text)}
                    for ident, (filename, text) in ws.documents.items()]

    def add_document(self, workspace_id: str, filename: str, data: bytes) -> dict[str, Any]:
        text = extract_document(filename, data)
        safe_name = Path(filename.replace("\\", "/")).name.strip()
        if not safe_name or len(safe_name) > 120:
            raise ValueError("Document filename must be 1–120 characters")
        with self.lock:
            ws = self._get(workspace_id)
            if len(ws.documents) >= MAX_DOCUMENTS:
                raise ValueError("Workspace has reached its 12-document limit")
            ident = secrets.token_urlsafe(12)
            chunks = source_chunks(text, ident, safe_name)
            if len(ws.chunks) + len(chunks) > MAX_CHUNKS:
                raise ValueError("Workspace has reached its indexed chunk limit")
            combined = ws.chunks + chunks
            # Create index before committing, so errors cannot leave partial documents.
            indexed = TfidfOriginalCorpus(combined, ngram_range=(1, 2))
            ws.documents[ident] = (safe_name, text)
            ws.chunks = combined
            ws.retriever = indexed
            return {"id": ident, "filename": safe_name,
                    "characters": len(text), "indexed_chunks": len(chunks)}

    def remove_document(self, workspace_id: str, document_id: str) -> None:
        with self.lock:
            ws = self._get(workspace_id)
            if document_id not in ws.documents:
                raise KeyError("Document not found in this workspace")
            new_chunks = [x for x in ws.chunks if x["document_id"] != document_id]
            indexed = TfidfOriginalCorpus(new_chunks, ngram_range=(1, 2)) if new_chunks else None
            ws.documents.pop(document_id)
            ws.chunks = new_chunks
            ws.retriever = indexed

    def ask(self, workspace_id: str, question: str, top_k: int = 3) -> dict[str, Any]:
        if not isinstance(question, str) or not 2 <= len(question.strip()) <= 500:
            raise ValueError("Question must be between 2 and 500 characters")
        if not isinstance(top_k, int) or not 1 <= top_k <= 5:
            raise ValueError("top_k must be between 1 and 5")
        with self.lock:
            ws = self._get(workspace_id)
            if ws.retriever is None:
                raise ValueError("Upload at least one document first")
            hits = ws.retriever.search(question, top_k)
            if not hits or max(float(x["cosine_tfidf"]) for x in hits) <= 0.0:
                return {"question": question, "answer": None, "abstained": True,
                        "reason": "No matching indexed passage", "citations": []}
            if self.reader is None:
                self.reader = PretrainedExtractiveReader()
            prediction = self.reader.answer(question, hits)
            if not prediction or not prediction.get("evidence_supported"):
                return {"question": question, "answer": None, "abstained": True,
                        "reason": "Reader could not verify a source span", "citations": []}
            chosen = next((x for x in hits
                           if x["document_id"] == prediction["document_id"]), None)
            if chosen is None:
                raise ValueError("Reader cited an unknown chunk")
            span = prediction["answer"]
            if not span or span not in chosen["context"]:
                raise ValueError("Answer is not an exact substring of cited source")
            chunk = ws.chunks[chosen["document_id"]]
            filename, original = ws.documents[chunk["document_id"]]
            offset = chunk["context"].find(span)
            begin = chunk["begin"] + offset
            assert original[begin:begin+len(span)] == span
            return {
                "question": question, "answer": span, "abstained": False,
                "extractive_not_generative": True, "model_score_not_truth_probability":
                    float(prediction.get("score", 0)),
                "citations": [{
                    "document_id": chunk["document_id"], "filename": filename,
                    "chunk_index": chosen["document_id"], "start_char": begin,
                    "end_char": begin + len(span), "quoted_span": span,
                    "retrieved_context": chunk["context"],
                    "retrieval_cosine": float(chosen["cosine_tfidf"]),
                }],
            }
