"""Evidence-bound extractive answer generator, not an agentic LLM orchestrator.

Formerly returned the fabricated string "This is a placeholder answer..."
when unconfigured. The default now runs an actual pretrained local QA model.
"""
from __future__ import annotations
from typing import Any, List


class AgenticGenerator:
    """Backward-compatible minimal generator interface with source checking.

    The name is historical; this is a real source-bound extractive QA module,
    not multi-agent planning, tool calling, or a generative LLM benchmark.
    """

    def __init__(self, llm: Any = None):
        # Optional custom callable can be supplied, but still must quote source.
        self.llm = llm
        self._reader = None

    def generate(self, query: str, context: List[str]) -> str:
        """Extract a supported answer from actual retrieved text; fail closed."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Cannot answer without an actual question")
        if not context or any(not isinstance(c, str) or not c.strip() for c in context):
            raise ValueError("Cannot answer without actual nonempty source paragraphs")
        if self.llm is not None:
            prompt = ("Answer with an EXACT substring of a provided paragraph.\n"
                      "Context:\n" + "\n\n".join(context) +
                      "\nQuestion: " + query + "\nAnswer:")
            answer = self.llm(prompt)
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Optional reader did not return any evidence-backed text")
            if not any(answer.strip() in original for original in context):
                raise ValueError("Optional reader produced text not grounded in provided sources")
            return answer.strip()
        if self._reader is None:
            from benchmarks.real_squad_qa import PretrainedExtractiveReader
            self._reader = PretrainedExtractiveReader()
        hits = [
            {"document_id": idx, "source_title": f"original_context_{idx}", "context": text}
            for idx, text in enumerate(context)
        ]
        record = self._reader.answer(query, hits)
        if not record["evidence_supported"] or not record["answer"]:
            raise ValueError("Pretrained reader returned no cited evidence")
        return record["answer"]
