"""
A simple, working RAG pipeline without evolution.
This serves as the baseline for comparison.
"""

from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI
import os


class BasicRAGPipeline:
    def __init__(self, chunk_size: int = 512) -> None:
        """Initialize a basic Retrieval-Augmented Generation pipeline.

        Args:
            chunk_size: Size of text chunks when splitting documents.
        """
        # Use sentence-transformers for embeddings (free, no API key needed)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunk_size = chunk_size

        # Use ChromaDB for vector storage (simple, local)
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("rag_docs")

        # Use OpenAI for generation (requires API key)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set."
            )
        self.llm = OpenAI(api_key=api_key)

    def chunk_text(self, text: str) -> list[str]:
        """Simple chunking by character count.

        Args:
            text: The document to split into chunks.

        Returns:
            A list of chunk strings.
        """
        chunks: list[str] = []
        for i in range(0, len(text), self.chunk_size):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def add_documents(self, documents: list[str]) -> int:
        """Chunk and index documents.

        Args:
            documents: A list of raw documents.

        Returns:
            The number of chunks added to the vector store.
        """
        all_chunks: list[dict[str, str]] = []
        for doc_id, doc in enumerate(documents):
            chunks = self.chunk_text(doc)
            for chunk_id, chunk in enumerate(chunks):
                all_chunks.append({
                    "id": f"doc{doc_id}_chunk{chunk_id}",
                    "text": chunk,
                })

        # Add to ChromaDB
        self.collection.add(
            documents=[c["text"] for c in all_chunks],
            ids=[c["id"] for c in all_chunks],
            embeddings=[self.encoder.encode(c["text"]).tolist() for c in all_chunks],
        )

        return len(all_chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Retrieve relevant chunks from the vector store.

        Args:
            query: The user query.
            top_k: Number of top results to return.

        Returns:
            A list of retrieved document chunks.
        """
        query_embedding = self.encoder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k
        )
        return results["documents"][0]

    def generate(self, query: str, context_chunks: list[str]) -> str:
        """Generate an answer using retrieved context.

        Args:
            query: The user question.
            context_chunks: List of retrieved document chunks.

        Returns:
            The generated answer text.
        """
        context = "\n\n".join(context_chunks)

        prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {query}

Answer:"""

        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )

        return response.choices[0].message.content

    def query(self, question: str) -> dict[str, object]:
        """Full RAG pipeline: retrieve + generate.

        Args:
            question: The user question.

        Returns:
            A dictionary containing the question, context, and answer.
        """
        context = self.retrieve(question, top_k=3)
        answer = self.generate(question, context)
        return {
            "question": question,
            "context": context,
            "answer": answer,
        }


def main() -> None:
    """Run a simple test of the basic RAG pipeline."""
    # Sample documents about AI
    docs = [
        "Artificial intelligence is the simulation of human intelligence by machines. It includes learning, reasoning, and self-correction.",
        "Machine learning is a subset of AI that enables systems to learn from data without explicit programming.",
        "Deep learning uses neural networks with multiple layers to analyze data patterns.",
    ]

    pipeline = BasicRAGPipeline(chunk_size=200)
    num_chunks = pipeline.add_documents(docs)
    print(f"Added {num_chunks} chunks to the vector store.")

    result = pipeline.query("What is machine learning?")
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")


if __name__ == "__main__":
    main()
