"""Software contract tests. All literal phrases below are fixtures, not research evidence."""
import pytest

from rag_pipelines.agentic_generator import AgenticGenerator


def test_agentic_generator_no_longer_returns_fake_placeholder():
    generator = AgenticGenerator(llm=lambda prompt: "Canada")
    assert generator.generate("Name the country", ["The country is Canada."]) == "Canada"
    with pytest.raises(ValueError, match="not grounded"):
        AgenticGenerator(llm=lambda prompt: "France").generate(
            "Name the country", ["The country is Canada."])
    with pytest.raises(ValueError, match="nonempty source"):
        AgenticGenerator().generate("Name the country", [])


def test_local_reader_delegation_with_original_evidence():
    generator = AgenticGenerator()
    class Reader:
        def answer(self, question, hits):
            assert question == "Which country?"
            assert hits[0]["context"] == "Canada."
            return {"answer": "Canada", "evidence_supported": True}
    generator._reader = Reader()
    assert generator.generate("Which country?", ["Canada."]) == "Canada"
