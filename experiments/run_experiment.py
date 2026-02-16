#!/usr/bin/env python3
import os
import sys
import time
import pickle
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from examples.basic_rag_working import BasicRAGPipeline

NUM_SAMPLES = 10
NUM_GENERATIONS = 5
POPULATION_SIZE = 8

CHUNK_SIZES = [128, 256, 512, 1024]
TOP_K_RANGE = (2, 10)
TEMP_RANGE = (0.3, 1.0)


def load_data(n_samples):
    print(f"Loading HotpotQA ({n_samples} samples)...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    sample = ds.select(range(n_samples))

    questions = [item['question'] for item in sample]
    answers = [item['answer'] for item in sample]
    contexts = []
    for item in sample:
        ctx = []
        for sentences in item['context']['sentences']:
            ctx.extend(sentences)
        contexts.append(ctx)

    return questions, answers, contexts


def evaluate_config(questions, answers, all_contexts, chunk_size, top_k, run_id):
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        from transformers import pipeline as hf_pipeline

        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        client = chromadb.Client()

        collection_name = f"rag_{run_id}"
        try:
            client.delete_collection(collection_name)
        except:
            pass
        collection = client.create_collection(collection_name)

        llm = hf_pipeline("text-generation", model="gpt2", device=-1, max_new_tokens=50)

        flat_contexts = []
        for ctx_list in all_contexts:
            flat_contexts.extend(ctx_list)

        all_chunks = []
        for doc_id, doc in enumerate(flat_contexts):
            for i in range(0, len(doc), chunk_size):
                chunk = doc[i:i + chunk_size]
                if chunk.strip():
                    all_chunks.append({'id': f"d{doc_id}_c{i}", 'text': chunk})

        if all_chunks:
            collection.add(
                documents=[c['text'] for c in all_chunks],
                ids=[c['id'] for c in all_chunks],
                embeddings=[encoder.encode(c['text']).tolist() for c in all_chunks]
            )

        correct = 0
        for i, q in enumerate(questions):
            query_emb = encoder.encode(q).tolist()
            results = collection.query(query_embeddings=[query_emb], n_results=top_k)
            context_chunks = results['documents'][0] if results['documents'] else []

            context = "\n\n".join(context_chunks[:3])
            prompt = f"Context: {context[:500]}\nQ: {q}\nA:"
            response = llm(prompt, max_new_tokens=30, num_return_sequences=1, pad_token_id=50256)
            answer = response[0]['generated_text'][len(prompt):]

            if answers[i].lower() in answer.lower():
                correct += 1

        client.delete_collection(collection_name)
        return correct / len(questions)
    except Exception as e:
        print(f"  Error: {e}")
        return 0.0


def run_baseline(questions, answers, contexts):
    print("\n" + "="*60)
    print("BASELINE: chunk_size=512, top_k=5")
    print("="*60)

    start = time.time()
    score = evaluate_config(questions, answers, contexts, chunk_size=512, top_k=5, run_id="baseline")
    elapsed = time.time() - start

    print(f"Accuracy: {score:.1%}")
    print(f"Time: {elapsed:.1f}s")
    return score


def evolve(questions, answers, contexts):
    print("\n" + "="*60)
    print(f"EVOLUTION: {NUM_GENERATIONS} generations, population {POPULATION_SIZE}")
    print("="*60)

    population = []
    for _ in range(POPULATION_SIZE):
        genome = {
            'chunk_size': random.choice(CHUNK_SIZES),
            'top_k': random.randint(*TOP_K_RANGE),
        }
        population.append(genome)

    best_ever = population[0].copy()
    best_score = 0.0
    history = []

    for gen in range(NUM_GENERATIONS):
        print(f"\nGeneration {gen + 1}/{NUM_GENERATIONS}")

        scores = []
        for i, genome in enumerate(population):
            score = evaluate_config(
                questions, answers, contexts,
                genome['chunk_size'], genome['top_k'],
                run_id=f"gen{gen}_ind{i}"
            )
            scores.append((genome, score))
            print(f"  [{i+1}/{POPULATION_SIZE}] chunk={genome['chunk_size']}, k={genome['top_k']} -> {score:.1%}")

        scores.sort(key=lambda x: x[1], reverse=True)

        gen_best = scores[0]
        if gen_best[1] > best_score:
            best_score = gen_best[1]
            best_ever = gen_best[0].copy()

        history.append({
            'generation': gen + 1,
            'best_score': scores[0][1],
            'avg_score': sum(s[1] for s in scores) / len(scores),
            'best_genome': scores[0][0].copy()
        })

        print(f"  Best this gen: {scores[0][1]:.1%} | Best ever: {best_score:.1%}")

        elite_count = 2
        new_population = [g.copy() for g, _ in scores[:elite_count]]

        while len(new_population) < POPULATION_SIZE:
            parent = random.choice([g for g, _ in scores[:4]])
            child = parent.copy()

            if random.random() < 0.3:
                child['chunk_size'] = random.choice(CHUNK_SIZES)
            if random.random() < 0.3:
                child['top_k'] = max(1, min(15, child['top_k'] + random.randint(-2, 2)))

            new_population.append(child)

        population = new_population

    return best_ever, best_score, history


def run_evolved(questions, answers, contexts, genome):
    print("\n" + "="*60)
    print(f"EVOLVED: chunk_size={genome['chunk_size']}, top_k={genome['top_k']}")
    print("="*60)

    start = time.time()
    score = evaluate_config(
        questions, answers, contexts,
        genome['chunk_size'], genome['top_k'],
        run_id="evolved_final"
    )
    elapsed = time.time() - start

    print(f"Accuracy: {score:.1%}")
    print(f"Time: {elapsed:.1f}s")
    return score


def main():
    print("NeuroEvoRAG Experiment")
    print("="*60)

    questions, answers, contexts = load_data(NUM_SAMPLES)
    print(f"Loaded {len(questions)} QA pairs")

    baseline_score = run_baseline(questions, answers, contexts)

    best_genome, evolution_best, history = evolve(questions, answers, contexts)

    evolved_score = run_evolved(questions, answers, contexts, best_genome)

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Baseline (chunk=512, k=5):     {baseline_score:.1%}")
    print(f"Evolved  (chunk={best_genome['chunk_size']}, k={best_genome['top_k']}): {evolved_score:.1%}")
    print(f"Improvement: {(evolved_score - baseline_score) * 100:+.1f} percentage points")

    results = {
        'baseline': {'chunk_size': 512, 'top_k': 5, 'score': baseline_score},
        'evolved': {'genome': best_genome, 'score': evolved_score},
        'history': history,
        'config': {
            'num_samples': NUM_SAMPLES,
            'num_generations': NUM_GENERATIONS,
            'population_size': POPULATION_SIZE
        }
    }

    os.makedirs('outputs', exist_ok=True)
    with open('outputs/experiment_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("\nResults saved to outputs/experiment_results.pkl")


if __name__ == "__main__":
    main()
