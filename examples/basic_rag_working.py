from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline

class BasicRAGPipeline:
    def __init__(self, chunk_size: int = 512) -> None:
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunk_size = chunk_size

        self.client = chromadb.Client()
        self.collection = self.client.create_collection("rag_docs")

        self.llm = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)

    def chunk_text(self, text: str) -> list[str]:
        chunks: list[str] = []
        for i in range(0, len(text), self.chunk_size):
            chunks.append(text[i:i + self.chunk_size])
        return chunks

    def add_documents(self, documents: list[str]) -> int:
        all_chunks: list[dict[str, str]] = []
        for doc_id, doc in enumerate(documents):
            chunks = self.chunk_text(doc)
            for chunk_id, chunk in enumerate(chunks):
                all_chunks.append({
                    'id': f"doc{doc_id}_chunk{chunk_id}",
                    'text': chunk
                })

        self.collection.add(
            documents=[c['text'] for c in all_chunks],
            ids=[c['id'] for c in all_chunks],
            embeddings=[self.encoder.encode(c['text']).tolist() for c in all_chunks]
        )

        return len(all_chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        query_embedding = self.encoder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results['documents'][0]

    def generate(self, query: str, context_chunks: list[str]) -> str:
        context = "\n\n".join(context_chunks)

        prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {query}

Answer:"""

        response = self.llm(prompt, max_length=150, num_return_sequences=1)
        return response[0]['generated_text']

    def query(self, question: str) -> dict:
        context = self.retrieve(question, top_k=3)
        answer = self.generate(question, context)
        return {
            'question': question,
            'context': context,
            'answer': answer
        }


if __name__ == "__main__":
    docs = [
        "Artificial intelligence is the simulation of human intelligence by machines. It includes learning, reasoning, and self-correction.",
        "Machine learning is a subset of AI that enables systems to learn from data without explicit programming.",
        "Deep learning uses neural networks with multiple layers to analyze data patterns."
    ]

    pipeline_instance = BasicRAGPipeline(chunk_size=200)
    pipeline_instance.add_documents(docs)

    result = pipeline_instance.query("What is machine learning?")
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
