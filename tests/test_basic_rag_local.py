"""Local application flow tests; these tiny inputs are unit fixtures, not measured research."""
import pytest

from examples.basic_rag_working import BasicRAGPipeline


class EchoingMockReader:
    def answer(self, question, hits):
        assert question
        assert hits
        first = hits[0]
        span = first["context"].split()[0]
        return {"answer": span, "score": .8, "document_id": first["document_id"],
                "source_title": first["source_title"],
                "evidence_text": span, "evidence_supported": True}


def test_local_rag_real_index_and_citations_without_api_secrets():
    pipeline = BasicRAGPipeline(chunk_size=500, reader=EchoingMockReader())
    assert pipeline.add_documents([
        "Canada has an iconic maple leaf flag.",
        "Venus has an extremely hot surface.",
        "People travel upstream to see salmon."
    ]) == 3
    results = pipeline.query("Which country has the maple leaf?", top_k=2)
    assert results["question"] == "Which country has the maple leaf?"
    assert results["context"]
    assert results["cited_source_id"].startswith("original_doc_")
    assert results["verbatim_supporting_span"] == results["answer"]
    assert results["is_extractive_answer"]
    assert pipeline.retrieve("Venus surface", top_k=1)[0].startswith("Venus")


def test_no_empty_docs_and_no_invented_answers():
    pipeline = BasicRAGPipeline(reader=EchoingMockReader())
    with pytest.raises(ValueError):
        pipeline.query("Any question?")
    with pytest.raises(ValueError):
        pipeline.add_documents([])
    with pytest.raises(ValueError):
        pipeline.add_documents(["  "])
    with pytest.raises(ValueError):
        pipeline.generate("What?", [])
    with pytest.raises(ValueError):
        BasicRAGPipeline(chunk_size=0)


def test_chunking_keeps_original_text_and_word_boundaries():
    text = "Canada has maple leaves. The autumn colours are beautiful. " * 8
    rag = BasicRAGPipeline(chunk_size=104, reader=EchoingMockReader())
    parts = rag.chunk_text(text)
    assert len(parts) > 1
    assert all(p.strip() and len(p) <= 104 for p in parts)
    assert all(not p.startswith(" ") and not p.endswith(" ") for p in parts)
    assert "Canada has maple leaves." in parts[0]


def test_add_documents_reindexes_previous_real_documents():
    rag = BasicRAGPipeline(reader=EchoingMockReader())
    assert rag.add_documents(["First scientific description of Mars"]) == 1
    assert rag.add_documents(["The maple leaf appears on Canada's flag"]) == 1
    assert "maple" in rag.retrieve("Which nation has a maple leaf flag?", 1)[0].lower()
