"""Small, REAL offline retrieval optimization experiment with synthetic documents.

This validates retrieval fitness optimization only: NO LLM generation, no RAGAS,
no HotpotQA, and no claim of production-scale neuroevolution.
"""
import random
import re
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer

DOCUMENTS = [
    "Banks check customer identity to reduce fraud and prevent abuse.",
    "Interest rates influence loan payments and mortgage affordability.",
    "Fraud investigations use transaction histories to identify suspicious activity.",
    "A secure authentication system requires multi factor verification.",
    "Weather reports help cities plan for heavy rainfall and flooding.",
    "A retrieval system finds passages that contain relevant facts for questions.",
]
QUESTIONS = [
    ("How do banks prevent fraud?", 0),
    ("What affects mortgage payments?", 1),
    ("Which data helps investigate fraud?", 2),
    ("How can systems verify account identity?", 3),
]


def split_words(text, size):
    words = re.findall(r"\w+", text.lower())
    if size < 1:
        raise ValueError("size must be positive")
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def retrieval_accuracy(chunk_words=16, top_k=1):
    """Fraction of questions whose cited document is in retrieved top-k chunks."""
    chunks, origins = [], []
    for doc_id, doc in enumerate(DOCUMENTS):
        for chunk in split_words(doc, chunk_words):
            chunks.append(chunk)
            origins.append(doc_id)
    vectors = TfidfVectorizer(stop_words="english").fit(chunks + [q for q, _ in QUESTIONS])
    matrix = vectors.transform(chunks)
    correct = 0
    for question, expected_doc in QUESTIONS:
        qv = vectors.transform([question])
        scores = (matrix @ qv.T).toarray().reshape(-1)
        selected = scores.argsort()[::-1][:min(top_k, len(chunks))]
        if expected_doc in {origins[int(j)] for j in selected}:
            correct += 1
    return correct / len(QUESTIONS)


def optimize(generations=4, population_size=8, seed=42):
    """Reproducible evolutionary search over retrieval chunk length and top-k."""
    if generations < 1 or population_size < 2:
        raise ValueError("Need >=1 generation and >=2 individuals")
    rng = random.Random(seed)
    space = [(s, k) for s in (3, 6, 12, 24) for k in (1, 2, 3)]
    baseline = (12, 1)
    population = [baseline] + [rng.choice(space) for _ in range(population_size - 1)]
    history = []
    all_best = (baseline, retrieval_accuracy(*baseline))
    for generation in range(generations):
        ranked = sorted([(ind, retrieval_accuracy(*ind)) for ind in population], key=lambda x: x[1], reverse=True)
        if ranked[0][1] > all_best[1]:
            all_best = ranked[0]
        history.append({"generation": generation, "best_accuracy": ranked[0][1], "params": ranked[0][0]})
        elites = [p for p, _ in ranked[:max(2, population_size // 3)]]
        population = [baseline] + elites[:population_size - 1]
        while len(population) < population_size:
            parent = rng.choice(elites)
            population.append((rng.choice([parent[0], rng.choice((3, 6, 12, 24))]), rng.choice([parent[1], rng.choice((1, 2, 3))])))
    return {"baseline_accuracy": retrieval_accuracy(*baseline), "best_params": all_best[0], "best_accuracy": all_best[1], "history": history}


if __name__ == "__main__":
    import json
    print(json.dumps(optimize(), indent=2))
