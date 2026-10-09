"""Real Stanford SQuAD 1.1 retrieval and evolutionary parameter selection.

Public, CC BY-SA 4.0 observational crowdsourced questions / Wikipedia contexts.
Nothing here uses fabricated article text, questions or scores.
Retrieval recall@k is NOT full RAG answer accuracy.
"""
import argparse
import json
import random
import re
from pathlib import Path
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

SQUAD_DEV = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
CHUNK_SIZES = (40, 80, 120, 200)
TOP_KS = (1, 2, 3)


def split_words(text, n):
    if n <= 0:
        raise ValueError("chunk size must be positive")
    words = re.findall(r"\w+", text.lower())
    return [" ".join(words[i:i+n]) for i in range(0, len(words), n)]


def parse_squad(payload):
    """Return true (question, source paragraph) pairings from official JSON."""
    rows = []
    if "data" not in payload:
        raise ValueError("Invalid SQuAD JSON: missing data")
    for article in payload["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for question in paragraph["qas"]:
                if question.get("question") and question.get("answers"):
                    rows.append((question["question"], context))
    if not rows:
        raise ValueError("SQuAD contains no answerable real questions")
    return rows


def load_squad(url=SQUAD_DEV, limit=80, seed=42, session=None):
    """Download real source with no synthetic fallback, sample deterministic IDs."""
    client = session or requests
    response = client.get(url, timeout=60)
    response.raise_for_status()
    rows = parse_squad(response.json())
    if limit < 10:
        raise ValueError("Require at least ten evaluation questions")
    rng = random.Random(seed)
    chosen = rng.sample(rows, min(limit, len(rows)))
    return chosen


def recall_at_k(rows, reference_contexts, chunk_words=80, top_k=2):
    """Return fraction of questions retrieving a chunk from their actual paragraph."""
    if not rows or not reference_contexts:
        raise ValueError("Real questions and reference contexts required")
    unique_contexts = list(dict.fromkeys(reference_contexts))
    chunks, origins = [], []
    for context_idx, text in enumerate(unique_contexts):
        for chunk in split_words(text, chunk_words):
            chunks.append(chunk)
            origins.append(context_idx)
    tfidf = TfidfVectorizer(stop_words="english", max_features=10000)
    matrix = tfidf.fit_transform(chunks)
    correct = 0
    lookup = {text: i for i, text in enumerate(unique_contexts)}
    for question, gold_context in rows:
        if gold_context not in lookup:
            raise ValueError("Gold paragraph not indexed")
        scores = (matrix @ tfidf.transform([question]).T).toarray().flatten()
        ranks = scores.argsort()[::-1][:min(top_k, len(chunks))]
        if lookup[gold_context] in {origins[i] for i in ranks}:
            correct += 1
    return correct / len(rows)


def optimize(rows, generations=3, population_size=5, seed=42):
    if len(rows) < 10 or generations < 1 or population_size < 2:
        raise ValueError("Need 10+ genuine question records and valid search budget")
    ntrain = int(len(rows) * 0.7)
    train_rows, heldout_rows = rows[:ntrain], rows[ntrain:]
    contexts = list(dict.fromkeys(context for _, context in rows))
    cache = {}
    def fitness(params):
        if params not in cache:
            cache[params] = recall_at_k(train_rows, contexts, *params)
        return cache[params]
    rng = random.Random(seed)
    baseline = (80, 1)
    candidates = [baseline] + [(rng.choice(CHUNK_SIZES), rng.choice(TOP_KS)) for _ in range(population_size-1)]
    best = baseline
    history = []
    for gen in range(generations):
        ranked = sorted(candidates, key=fitness, reverse=True)
        if fitness(ranked[0]) > fitness(best):
            best = ranked[0]
        history.append({"generation": gen, "best_training_recall": fitness(ranked[0]), "parameters": list(ranked[0])})
        elites = ranked[:max(2, population_size//3)]
        candidates = [baseline] + elites[:population_size-1]
        while len(candidates) < population_size:
            parent = rng.choice(elites)
            candidates.append((rng.choice([parent[0], rng.choice(CHUNK_SIZES)]), rng.choice([parent[1], rng.choice(TOP_KS)])))
    return {
        "source": SQUAD_DEV,
        "license": "CC BY-SA 4.0",
        "questions_train": len(train_rows),
        "questions_heldout": len(heldout_rows),
        "baseline_parameters": list(baseline),
        "best_parameters": list(best),
        "train_recall": fitness(best),
        "heldout_baseline_recall": recall_at_k(heldout_rows, contexts, *baseline),
        "heldout_selected_recall": recall_at_k(heldout_rows, contexts, *best),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-questions", type=int, default=80)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", help="Optional metrics JSON file")
    args = parser.parse_args()
    rows = load_squad(limit=args.max_questions, seed=args.seed)
    result = optimize(rows, generations=args.generations, population_size=args.population_size, seed=args.seed)
    print(json.dumps(result, indent=2))
    if args.output:
        dest = Path(args.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
