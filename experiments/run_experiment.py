#!/usr/bin/env python3
import os
import sys
import time
import json
import pickle
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

NUM_SAMPLES = 15
NUM_GENERATIONS = 3
POPULATION_SIZE = 5

CHUNK_SIZES = [128, 256, 512, 1024, 2048]
TOP_K_RANGE = (1, 12)
TEMP_RANGE = (0.1, 1.5)

_encoder = None
_client = None
_tokenizer = None
_model = None


def get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer('all-MiniLM-L6-v2')
    return _encoder


def get_chroma_client():
    global _client
    if _client is None:
        import chromadb
        _client = chromadb.Client()
    return _client


def get_llm():
    global _tokenizer, _model
    if _tokenizer is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        _tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
        _model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')
    return _tokenizer, _model


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


def f1_score(prediction, ground_truth):
    pred_tokens = prediction.lower().split()
    truth_tokens = ground_truth.lower().split()
    if not pred_tokens or not truth_tokens:
        return 0.0
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def generate_answer(question, context, temperature):
    import torch

    tokenizer, model = get_llm()

    prompt = f"Answer the question concisely based on the context. Give only the answer.\n\nContext:\n{context[:1500]}\n\nQuestion: {question}\n\nAnswer:"

    inputs = tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=max(temperature, 0.01),
            do_sample=temperature > 0.1,
            top_p=0.9 if temperature > 0.1 else 1.0,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def evaluate_config(questions, answers, all_contexts, chunk_size, top_k, temperature, run_id):
    try:
        encoder = get_encoder()
        client = get_chroma_client()

        collection_name = f"rag_{run_id}"
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        collection = client.create_collection(collection_name)

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
            batch_size = 500
            for start in range(0, len(all_chunks), batch_size):
                batch = all_chunks[start:start + batch_size]
                collection.add(
                    documents=[c['text'] for c in batch],
                    ids=[c['id'] for c in batch],
                    embeddings=[encoder.encode(c['text']).tolist() for c in batch]
                )

        f1_total = 0.0
        exact_match = 0
        start_time = time.time()

        for i, q in enumerate(questions):
            query_emb = encoder.encode(q).tolist()
            actual_k = min(top_k, collection.count())
            if actual_k == 0:
                continue
            results = collection.query(query_embeddings=[query_emb], n_results=actual_k)
            context_chunks = results['documents'][0] if results['documents'] else []
            context = "\n".join(context_chunks)

            answer = generate_answer(q, context, temperature)

            f1 = f1_score(answer, answers[i])
            f1_total += f1
            if answers[i].lower() in answer.lower():
                exact_match += 1

        elapsed = time.time() - start_time
        client.delete_collection(collection_name)

        n = len(questions)
        return {
            'f1': f1_total / n if n > 0 else 0.0,
            'exact_match': exact_match / n if n > 0 else 0.0,
            'latency': elapsed / n if n > 0 else 0.0,
        }
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return {'f1': 0.0, 'exact_match': 0.0, 'latency': 999.0}


def fitness(metrics):
    return 0.6 * metrics['f1'] + 0.3 * metrics['exact_match'] + 0.1 * max(0, 1.0 - metrics['latency'])


def run_baseline(questions, answers, contexts):
    print("\n" + "="*60)
    print("BASELINE: chunk_size=512, top_k=5, temp=0.3")
    print("="*60)

    metrics = evaluate_config(questions, answers, contexts,
                              chunk_size=512, top_k=5, temperature=0.3,
                              run_id="baseline")
    fit = fitness(metrics)
    print(f"  F1: {metrics['f1']:.3f}")
    print(f"  Exact Match: {metrics['exact_match']:.1%}")
    print(f"  Latency: {metrics['latency']:.2f}s/q")
    print(f"  Fitness: {fit:.3f}")
    return metrics, fit


def evolve(questions, answers, contexts):
    print("\n" + "="*60)
    print(f"EVOLUTION: {NUM_GENERATIONS} generations, population {POPULATION_SIZE}")
    print("="*60)

    population = []
    for _ in range(POPULATION_SIZE):
        genome = {
            'chunk_size': random.choice(CHUNK_SIZES),
            'top_k': random.randint(*TOP_K_RANGE),
            'temperature': round(random.uniform(*TEMP_RANGE), 2),
        }
        population.append(genome)

    best_ever = population[0].copy()
    best_fitness = 0.0
    history = []

    for gen in range(NUM_GENERATIONS):
        print(f"\nGeneration {gen + 1}/{NUM_GENERATIONS}")

        scored = []
        for i, genome in enumerate(population):
            metrics = evaluate_config(
                questions, answers, contexts,
                genome['chunk_size'], genome['top_k'], genome['temperature'],
                run_id=f"g{gen}_i{i}"
            )
            fit = fitness(metrics)
            scored.append((genome, metrics, fit))
            print(f"  [{i+1}/{POPULATION_SIZE}] chunk={genome['chunk_size']:>4}, k={genome['top_k']:>2}, t={genome['temperature']:.2f} -> F1={metrics['f1']:.3f} EM={metrics['exact_match']:.1%} fit={fit:.3f}")

        scored.sort(key=lambda x: x[2], reverse=True)

        if scored[0][2] > best_fitness:
            best_fitness = scored[0][2]
            best_ever = scored[0][0].copy()

        gen_scores = [s[2] for s in scored]
        history.append({
            'generation': gen + 1,
            'best_fitness': scored[0][2],
            'avg_fitness': sum(gen_scores) / len(gen_scores),
            'best_f1': scored[0][1]['f1'],
            'avg_f1': sum(s[1]['f1'] for s in scored) / len(scored),
            'best_em': scored[0][1]['exact_match'],
            'best_genome': scored[0][0].copy()
        })

        print(f"  >> Gen best: {scored[0][2]:.3f} | Overall best: {best_fitness:.3f}")

        elite_count = 2
        new_pop = [g.copy() for g, _, _ in scored[:elite_count]]

        while len(new_pop) < POPULATION_SIZE:
            tournament = random.sample(scored[:6], min(3, len(scored[:6])))
            parent = max(tournament, key=lambda x: x[2])[0]
            child = parent.copy()

            if random.random() < 0.3 and len(scored) >= 2:
                parent2 = random.choice(scored[:4])[0]
                if random.random() < 0.5:
                    child['chunk_size'] = parent2['chunk_size']
                if random.random() < 0.5:
                    child['top_k'] = parent2['top_k']
                if random.random() < 0.5:
                    child['temperature'] = parent2['temperature']

            if random.random() < 0.35:
                child['chunk_size'] = random.choice(CHUNK_SIZES)
            if random.random() < 0.35:
                child['top_k'] = max(1, min(15, child['top_k'] + random.randint(-3, 3)))
            if random.random() < 0.35:
                child['temperature'] = round(max(0.1, min(1.5, child['temperature'] + random.uniform(-0.2, 0.2))), 2)

            new_pop.append(child)

        population = new_pop

    return best_ever, best_fitness, history


def main():
    random.seed(42)
    print("NeuroEvoRAG Experiment v2")
    print("="*60)
    print(f"Samples: {NUM_SAMPLES}, Generations: {NUM_GENERATIONS}, Population: {POPULATION_SIZE}")
    print(f"LLM: google/flan-t5-small (local)")

    questions, answers, contexts = load_data(NUM_SAMPLES)
    print(f"Loaded {len(questions)} QA pairs")

    baseline_metrics, baseline_fit = run_baseline(questions, answers, contexts)

    best_genome, best_fit, history = evolve(questions, answers, contexts)

    print("\n" + "="*60)
    print(f"RE-EVALUATING BEST: chunk={best_genome['chunk_size']}, k={best_genome['top_k']}, t={best_genome['temperature']}")
    print("="*60)
    evolved_metrics = evaluate_config(
        questions, answers, contexts,
        best_genome['chunk_size'], best_genome['top_k'], best_genome['temperature'],
        run_id="final_evolved"
    )
    evolved_fit = fitness(evolved_metrics)
    print(f"  F1: {evolved_metrics['f1']:.3f}")
    print(f"  Exact Match: {evolved_metrics['exact_match']:.1%}")
    print(f"  Latency: {evolved_metrics['latency']:.2f}s/q")
    print(f"  Fitness: {evolved_fit:.3f}")

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Baseline  (chunk=512, k=5, t=0.3):")
    print(f"  F1={baseline_metrics['f1']:.3f}, EM={baseline_metrics['exact_match']:.1%}, Fitness={baseline_fit:.3f}")
    print(f"Evolved   (chunk={best_genome['chunk_size']}, k={best_genome['top_k']}, t={best_genome['temperature']}):")
    print(f"  F1={evolved_metrics['f1']:.3f}, EM={evolved_metrics['exact_match']:.1%}, Fitness={evolved_fit:.3f}")
    improvement = evolved_fit - baseline_fit
    print(f"Fitness improvement: {improvement:+.3f} ({improvement/max(baseline_fit, 0.001)*100:+.1f}%)")

    results = {
        'baseline': {
            'config': {'chunk_size': 512, 'top_k': 5, 'temperature': 0.3},
            'metrics': baseline_metrics,
            'fitness': baseline_fit
        },
        'evolved': {
            'config': best_genome,
            'metrics': evolved_metrics,
            'fitness': evolved_fit
        },
        'history': history,
        'experiment': {
            'num_samples': NUM_SAMPLES,
            'num_generations': NUM_GENERATIONS,
            'population_size': POPULATION_SIZE,
            'llm': 'google/flan-t5-small',
            'fitness_weights': '0.6*F1 + 0.3*EM + 0.1*(1-latency)',
            'date': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    }

    os.makedirs('outputs', exist_ok=True)
    with open('outputs/experiment_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    with open('outputs/experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\nResults saved to outputs/")


if __name__ == "__main__":
    main()
