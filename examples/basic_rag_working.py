"""Working local retrieval + pretrained extractive QA; no OpenAI API key.

The original demo needed an OPENAI_API_KEY, OpenAI model access,
SentenceTransformer and Chroma even for simple questions. This version
uses a local TF-IDF retriever and an actual pretrained CPU QA model,
cites its original evidence span, and can run on genuine SQuAD paragraphs.

This is a closed-corpus extractive RAG baseline, not chat generation or NEAT.
"""
from __future__ import annotations
import re

from benchmarks.real_squad_qa import PretrainedExtractiveReader, TfidfOriginalCorpus


class BasicRAGPipeline:
    def __init__(self, chunk_size: int = 700, reader=None) -> None:
        if chunk_size < 100:
            raise ValueError("chunk_size must be at least 100 characters")
        self.chunk_size = chunk_size
        self._documents = []
        self._chunks = []
        self._retriever = None
        # Dependency injection avoids expensive model downloads in unit tests.
        # Real transformer downloaded on first actual query, not init.
        self._reader = reader

    def chunk_text(self, text: str) -> list[str]:
        """Preserve original capitalization and punctuation for span citations."""
        if not isinstance(text, str) or not text.strip():
            return []
        pieces = list(re.finditer(r"\S+", text))
        result = []
        first = 0
        while first < len(pieces):
            end = first
            while (end + 1 < len(pieces) and
                   pieces[end+1].end() - pieces[first].start() <= self.chunk_size):
                end += 1
            result.append(text[pieces[first].start():pieces[end].end()])
            first = end + 1
        return result

    def add_documents(self, documents: list[str]) -> int:
        if not documents:
            raise ValueError("No authentic source documents supplied for indexing")
        new_chunks = []
        base_id = len(self._documents)
        for offset, document in enumerate(documents):
            if not isinstance(document, str):
                raise ValueError("Original document must be text")
            for position, chunk in enumerate(self.chunk_text(document)):
                new_chunks.append({
                    "title": f"original_doc_{base_id+offset}_chunk_{position}",
                    "context": chunk,
                })
        if not new_chunks:
            raise ValueError("No text was found in supplied original documents")
        self._documents.extend(documents)
        self._chunks.extend(new_chunks)
        self._retriever = TfidfOriginalCorpus(self._chunks, ngram_range=(1, 2))
        return len(new_chunks)

    def _search(self, question: str, top_k: int):
        if self._retriever is None:
            raise ValueError("Call add_documents with original source text before querying")
        if not question.strip():
            raise ValueError("Question must be nonempty")
        return self._retriever.search(question, top_k)

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        return [item["context"] for item in self._search(query, top_k)]

    def _get_reader(self):
        if self._reader is None:
            self._reader = PretrainedExtractiveReader()
        return self._reader

    def generate(self, query: str, context_chunks: list[str]) -> str:
        """Neural extractive QA over explicitly provided original context chunks."""
        if not context_chunks:
            raise ValueError("No actual retrieved evidence supplied; refusing to invent answer")
        hits = [{"document_id": i, "source_title": "provided_context",
                 "context": chunk} for i, chunk in enumerate(context_chunks)]
        return self._get_reader().answer(query, hits)["answer"]

    def query(self, question: str, top_k: int = 3) -> dict[str, object]:
        retrieved = self._search(question, top_k)
        answer = self._get_reader().answer(question, retrieved)
        if not answer["evidence_supported"]:
            raise ValueError("No source-supported span found; refusing unsupported answer")
        return {
            "question": question,
            "context": [x["context"] for x in retrieved],
            "answer": answer["answer"],
            "cited_original_chunk": answer["document_id"],
            "cited_source_id": answer.get("source_title"),
            "verbatim_supporting_span": answer["evidence_text"],
            "model_span_confidence": answer["score"],
            "model_checkpoint": "distilbert/distilbert-base-cased-distilled-squad",
            "is_extractive_answer": True,
        }


def main():
    from benchmarks.real_squad_qa import (
        construct_closed_corpus, download_actual_dataset
    )
    paragraphs, digest = download_actual_dataset()
    corpus, _, evaluation_cases = construct_closed_corpus(
        paragraphs, corpus_size=120, train_questions=10,
        heldout_questions=10, seed=42)
    question = evaluation_cases[0]["question"]
    originals = [item["context"] for item in corpus]
    rag = BasicRAGPipeline(chunk_size=1500)
    chunks = rag.add_documents(originals)
    result = rag.query(question, top_k=3)
    print("Source: official Stanford SQuAD 1.1 (SHA256:", digest, ")")
    print("Indexed original real Wikipedia source chunks:", chunks)
    print("Original annotated question:", result["question"])
    print("Original published gold answers:", evaluation_cases[0]["gold_answers"])
    print("Live model answer:", result["answer"])
    print("Actual cited source chunk:", result["cited_source_id"])
    print("Verbatim supporting source span:", result["verbatim_supporting_span"])


if __name__ == "__main__":
    main()
