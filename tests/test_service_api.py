"""Local application API tests with synthetic documents, never invented benchmark scores."""
from io import BytesIO

import pytest
from fastapi.testclient import TestClient

from service import api
from service.core import DocumentQASystem, extract_document, source_chunks


class FixtureReader:
    """Stubbed software-only reader: test citation provenance without model download."""
    def answer(self, question, hits):
        selected = hits[0]
        phrase = "water conservation" if "water conservation" in selected["context"] else "research"
        return {"answer": phrase, "evidence_text": phrase, "evidence_supported": True,
                "document_id": selected["document_id"], "source_title": selected["source_title"],
                "score": .8}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "system", DocumentQASystem(reader=FixtureReader()))
    return TestClient(api.app)


def new(client):
    response = client.post("/api/workspaces")
    assert response.status_code == 201
    return {"X-Workspace-ID": response.json()["workspace_id"]}


def test_full_browser_to_api_upload_ask_citation_delete(client):
    assert client.get("/").status_code == 200
    assert "Upload a local text" in client.get("/").text
    assert client.get("/health").json()["local_only"] is True
    auth = new(client)
    original = "This original research note discusses water conservation at a municipal facility."
    posted = client.post("/api/documents", headers=auth, files={
        "file": ("source.md", original.encode("utf-8"), "text/markdown")
    })
    assert posted.status_code == 201, posted.text
    ident = posted.json()["id"]
    assert len(client.get("/api/documents", headers=auth).json()["documents"]) == 1
    prediction = client.post("/api/ask", headers=auth,
                             json={"question": "What does the research note discuss about water conservation?"})
    assert prediction.status_code == 200, prediction.text
    data = prediction.json()
    assert data["answer"] == "water conservation"
    assert data["abstained"] is False
    cite = data["citations"][0]
    assert cite["filename"] == "source.md"
    assert original[cite["start_char"]:cite["end_char"]] == data["answer"]
    assert cite["quoted_span"] == data["answer"]
    assert client.delete("/api/documents/" + ident, headers=auth).status_code == 204
    assert client.post("/api/ask", headers=auth, json={
        "question": "What is here?"}).status_code == 422
    assert client.get("/api/documents", headers=auth).json()["documents"] == []


def test_workspace_separation_and_revoked_document_visibility(client):
    alice, bob = new(client), new(client)
    res = client.post("/api/documents", headers=alice, files={
        "file": ("evidence.txt", b"Water conservation is a public research priority.")
    })
    assert res.status_code == 201
    assert client.get("/api/documents", headers=bob).json()["documents"] == []
    assert client.post("/api/ask", headers=bob, json={"question": "Water conservation?"}
                       ).status_code == 422
    assert client.delete("/api/documents/" + res.json()["id"], headers=bob).status_code == 404
    assert client.get("/api/documents", headers={"X-Workspace-ID": "invalid"}).status_code == 404


@pytest.mark.parametrize("filename,data", [
    ("secret.exe", b"some content"),
    ("wrong.txt", b"\xff\xfe\x00\xdd"),
    ("huge.txt", b"a" * (1024 * 1024 + 1)),
    ("empty.md", b" "),
    ("blank.pdf", b"%PDF-fake"),
])
def test_upload_rejects_incompatible_or_bounded_content(client, filename, data):
    auth = new(client)
    response = client.post("/api/documents", headers=auth, files={
        "file": (filename, data)
    })
    assert response.status_code == 422
    assert client.get("/api/documents", headers=auth).json()["documents"] == []


def test_abstains_on_unrelated_query_and_refuses_invalid_question(client):
    auth = new(client)
    posted = client.post("/api/documents", headers=auth, files={
        "file": ("observed.txt", b"This note addresses water conservation policies.")
    })
    assert posted.status_code == 201
    answer = client.post("/api/ask", headers=auth, json={
        "question": "What about extraterrestrial octopus locomotion?"})
    assert answer.status_code == 200
    assert answer.json()["abstained"]
    assert answer.json()["answer"] is None
    assert client.post("/api/ask", headers=auth, json={
        "question": "x", "top_k": 99}).status_code == 422


def test_detected_provenance_uses_original_text_offsets_not_token_positions():
    content = "Leading whitespace.\n\n  The original source says water conservation matters."
    chunks = source_chunks(content, "doc", "name.txt")
    assert chunks
    for chunk in chunks:
        assert content[chunk["begin"]:chunk["end"]] == chunk["context"]
    assert extract_document("one.txt", b"\xef\xbb\xbfText") == "Text"
