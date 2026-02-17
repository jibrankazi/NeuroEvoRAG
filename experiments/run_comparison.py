#!/usr/bin/env python3
"""Compare optimization methods: Evolution vs Optuna vs Grid Search vs Random Search.

All methods get the same evaluation budget for a fair comparison.
"""
import os
import sys
import time
import json
import random
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_experiment import (
    load_data, evaluate_config, fitness, get_encoder, get_llm,
    CHUNK_SIZES, TOP_K_RANGE, TEMP_RANGE,
)

EVAL_BUDGET = 15
NUM_SAMPLES = 15
SEED = 42


def run_random_search(questions, answers, contexts, budget):
    print("\n" + "=" * 60)
    print(f"RANDOM SEARCH ({budget} evaluations)")
    print("=" * 60)

    random.seed(SEED)
    best_config = None
    best_fit = 0.0
    history = []

    for i in range(budget):
        config = {
            'chunk_size': random.choice(CHUNK_SIZES),
            'top_k': random.randint(*TOP_K_RANGE),
            'temperature': round(random.uniform(*TEMP_RANGE), 2),
        }
        metrics = evaluate_config(
            questions, answers, contexts,
            config['chunk_size'], config['top_k'], config['temperature'],
            run_id=f"rand_{i}"
        )
        fit = fitness(metrics)
        print(f"  [{i+1}/{budget}] chunk={config['chunk_size']:>4}, k={config['top_k']:>2}, t={config['temperature']:.2f} -> fit={fit:.3f}")

        if fit > best_fit:
            best_fit = fit
            best_config = config.copy()

        history.append({'eval': i + 1, 'best_fitness': best_fit, 'config': config, 'fitness': fit})

    return best_config, best_fit, history


def run_grid_search(questions, answers, contexts, budget):
    print("\n" + "=" * 60)
    print(f"GRID SEARCH ({budget} evaluations)")
    print("=" * 60)

    chunk_grid = [128, 512, 2048]
    k_grid = [2, 5, 10]
    temp_grid = [0.3, 0.7, 1.2]
    all_combos = list(itertools.product(chunk_grid, k_grid, temp_grid))

    random.seed(SEED)
    random.shuffle(all_combos)
    combos = all_combos[:budget]

    best_config = None
    best_fit = 0.0
    history = []

    for i, (chunk, k, temp) in enumerate(combos):
        config = {'chunk_size': chunk, 'top_k': k, 'temperature': temp}
        metrics = evaluate_config(
            questions, answers, contexts,
            chunk, k, temp,
            run_id=f"grid_{i}"
        )
        fit = fitness(metrics)
        print(f"  [{i+1}/{len(combos)}] chunk={chunk:>4}, k={k:>2}, t={temp:.2f} -> fit={fit:.3f}")

        if fit > best_fit:
            best_fit = fit
            best_config = config.copy()

        history.append({'eval': i + 1, 'best_fitness': best_fit, 'config': config, 'fitness': fit})

    return best_config, best_fit, history


def run_optuna_search(questions, answers, contexts, budget):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("\n" + "=" * 60)
    print(f"OPTUNA (Bayesian) ({budget} trials)")
    print("=" * 60)

    history = []
    best_fit_so_far = [0.0]

    def objective(trial):
        chunk_size = trial.suggest_categorical('chunk_size', CHUNK_SIZES)
        top_k = trial.suggest_int('top_k', *TOP_K_RANGE)
        temperature = trial.suggest_float('temperature', *TEMP_RANGE)

        metrics = evaluate_config(
            questions, answers, contexts,
            chunk_size, top_k, temperature,
            run_id=f"optuna_{trial.number}"
        )
        fit = fitness(metrics)

        best_fit_so_far[0] = max(best_fit_so_far[0], fit)
        history.append({
            'eval': trial.number + 1,
            'best_fitness': best_fit_so_far[0],
            'config': {'chunk_size': chunk_size, 'top_k': top_k, 'temperature': temperature},
            'fitness': fit,
        })

        print(f"  [{trial.number+1}/{budget}] chunk={chunk_size:>4}, k={top_k:>2}, t={temperature:.2f} -> fit={fit:.3f}")
        return fit

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=budget)

    best = study.best_trial
    best_config = {
        'chunk_size': best.params['chunk_size'],
        'top_k': best.params['top_k'],
        'temperature': round(best.params['temperature'], 2),
    }
    return best_config, best.value, history


def run_evolution(questions, answers, contexts, budget):
    print("\n" + "=" * 60)
    print(f"EVOLUTION ({budget} evaluations)")
    print("=" * 60)

    random.seed(SEED)
    pop_size = 5
    num_gens = budget // pop_size

    population = []
    for _ in range(pop_size):
        genome = {
            'chunk_size': random.choice(CHUNK_SIZES),
            'top_k': random.randint(*TOP_K_RANGE),
            'temperature': round(random.uniform(*TEMP_RANGE), 2),
        }
        population.append(genome)

    best_ever = population[0].copy()
    best_fitness = 0.0
    history = []
    eval_count = 0

    for gen in range(num_gens):
        scored = []
        for i, genome in enumerate(population):
            metrics = evaluate_config(
                questions, answers, contexts,
                genome['chunk_size'], genome['top_k'], genome['temperature'],
                run_id=f"evo_g{gen}_i{i}"
            )
            fit = fitness(metrics)
            scored.append((genome, metrics, fit))
            eval_count += 1

            if fit > best_fitness:
                best_fitness = fit
                best_ever = genome.copy()

            history.append({'eval': eval_count, 'best_fitness': best_fitness, 'config': genome.copy(), 'fitness': fit})
            print(f"  [{eval_count}/{budget}] chunk={genome['chunk_size']:>4}, k={genome['top_k']:>2}, t={genome['temperature']:.2f} -> fit={fit:.3f}")

        scored.sort(key=lambda x: x[2], reverse=True)
        print(f"  >> Gen {gen+1} best: {scored[0][2]:.3f} | Overall best: {best_fitness:.3f}")

        new_pop = [g.copy() for g, _, _ in scored[:2]]
        while len(new_pop) < pop_size:
            tournament = random.sample(scored[:4], min(3, len(scored[:4])))
            parent = max(tournament, key=lambda x: x[2])[0]
            child = parent.copy()

            if random.random() < 0.3 and len(scored) >= 2:
                parent2 = random.choice(scored[:3])[0]
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
    print("NeuroEvoRAG: Multi-Method Comparison")
    print("=" * 60)
    print(f"Evaluation budget: {EVAL_BUDGET} per method")
    print(f"Samples: {NUM_SAMPLES}, Seed: {SEED}")
    print(f"LLM: google/flan-t5-small")

    questions, answers, contexts = load_data(NUM_SAMPLES)
    print(f"Loaded {len(questions)} QA pairs")

    # Warm up models
    print("\nWarming up models...")
    get_encoder()
    get_llm()

    # Baseline
    print("\n" + "=" * 60)
    print("HAND-TUNED BASELINE: chunk=512, k=5, t=0.3")
    print("=" * 60)
    baseline_metrics = evaluate_config(
        questions, answers, contexts, 512, 5, 0.3, run_id="baseline"
    )
    baseline_fit = fitness(baseline_metrics)
    print(f"  Fitness: {baseline_fit:.3f}")

    start = time.time()
    results = {}

    for name, fn in [
        ("Random Search", run_random_search),
        ("Grid Search", run_grid_search),
        ("Optuna (Bayesian)", run_optuna_search),
        ("Evolution", run_evolution),
    ]:
        t0 = time.time()
        config, fit, hist = fn(questions, answers, contexts, EVAL_BUDGET)
        elapsed = time.time() - t0
        results[name] = {
            'best_config': config,
            'best_fitness': fit,
            'history': hist,
            'time': elapsed,
        }

    total_time = time.time() - start

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Method':<22} {'Best Fitness':>12} {'Config':>35} {'Time':>8}")
    print("-" * 80)
    print(f"{'Hand-tuned baseline':<22} {baseline_fit:>12.3f} {'chunk=512, k=5, t=0.3':>35} {'--':>8}")

    for name, r in sorted(results.items(), key=lambda x: x[1]['best_fitness'], reverse=True):
        c = r['best_config']
        cfg_str = f"chunk={c['chunk_size']}, k={c['top_k']}, t={c['temperature']}"
        print(f"{name:<22} {r['best_fitness']:>12.3f} {cfg_str:>35} {r['time']:>7.1f}s")

    best_method = max(results.items(), key=lambda x: x[1]['best_fitness'])
    print(f"\nBest method: {best_method[0]} (fitness={best_method[1]['best_fitness']:.3f})")
    print(f"Improvement over baseline: {best_method[1]['best_fitness'] - baseline_fit:+.3f} ({(best_method[1]['best_fitness'] - baseline_fit) / max(baseline_fit, 0.001) * 100:+.1f}%)")
    print(f"Total time: {total_time:.0f}s")

    # Save results
    output = {
        'baseline': {'config': {'chunk_size': 512, 'top_k': 5, 'temperature': 0.3}, 'fitness': baseline_fit},
        'methods': {name: {'best_config': r['best_config'], 'best_fitness': r['best_fitness'], 'time': r['time']}
                    for name, r in results.items()},
        'eval_budget': EVAL_BUDGET,
        'num_samples': NUM_SAMPLES,
        'seed': SEED,
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    os.makedirs('outputs', exist_ok=True)
    with open('outputs/comparison_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\nResults saved to outputs/comparison_results.json")


if __name__ == "__main__":
    main()
