# Experimental Results

## Experiment 1: Initial Evolution Run

**Date**: 2026-02-16

### Setup

- **Dataset**: HotpotQA validation (10 samples)
- **Baseline**: chunk_size=512, top_k=5
- **Evolution**: 5 generations, population size 8
- **Evolved parameters**: chunk_size (128/256/512/1024), top_k (2-10)
- **LLM**: GPT-2 (local, CPU)
- **Metric**: Substring match accuracy

### Results

| Configuration | Accuracy |
|--------------|----------|
| Baseline (chunk=512, k=5) | 20% |
| Best during evolution (chunk=256, k=2) | 30% |
| Final evolved (chunk=256, k=2) | 20% |

### Evolution History

```
Gen 1: best=20%, avg=10% -> chunk=256, k=3
Gen 2: best=20%, avg=12% -> chunk=256, k=3
Gen 3: best=20%, avg=6%  -> chunk=256, k=3
Gen 4: best=30%, avg=14% -> chunk=256, k=2  <-- peak
Gen 5: best=20%, avg=11% -> chunk=256, k=2
```

### Observations

1. Evolution converged to smaller chunks (256 vs 512) and fewer retrievals (k=2-3 vs k=5)
2. Peak accuracy (30%) exceeded baseline (20%) by 10 percentage points
3. High variance between runs due to:
   - Small sample size (10 questions)
   - Non-deterministic GPT-2 generation
   - Simple substring match metric

### Limitations

- GPT-2 is weak at QA tasks compared to instruction-tuned models
- Substring match doesn't capture semantic correctness
- 10 samples too few for statistical significance

## Experiment 2: Flan-T5 with F1/EM Metrics

**Date**: 2026-02-17

### Setup

- **Dataset**: HotpotQA validation (15 samples)
- **Baseline**: chunk_size=512, top_k=5, temperature=0.3
- **Evolution**: 3 generations, population size 5
- **Evolved parameters**: chunk_size (128/256/512/1024/2048), top_k (1-12), temperature (0.1-1.5)
- **LLM**: google/flan-t5-small (instruction-tuned, local CPU)
- **Metrics**: F1 score, Exact Match
- **Fitness function**: 0.6*F1 + 0.3*EM + 0.1*(1-latency)

### Results

| Configuration | F1 | Exact Match | Fitness |
|--------------|-----|-------------|---------|
| Baseline (chunk=512, k=5, t=0.3) | 0.116 | 6.7% | 0.170 |
| Evolved (chunk=2048, k=2, t=0.93) | 0.224 | 20.0% | 0.284 |
| **Improvement** | +93% | +199% | **+66.9%** |

### Evolution History

```
Gen 1: best=0.409 | avg fitness across population
       -> Best genome: chunk=2048, k=2, t=0.93
Gen 2: best=0.403 | Overall best: 0.409
       -> chunk=128, k=11, t=1.14 (local optimum)
Gen 3: best=0.373 | Overall best: 0.409
       -> Convergence to chunk=2048, k=2
```

### Key Findings

1. **Larger chunks work better**: 2048 >> 512 for multi-hop QA
   - Preserves more context per chunk
   - Reduces fragmentation of related sentences

2. **Fewer retrievals with larger chunks**: k=2 vs k=5
   - Larger chunks contain more info per retrieval
   - Less noise from irrelevant chunks

3. **Higher temperature helps**: 0.93 vs 0.3
   - More diverse generation explores answer space
   - Small models benefit from sampling

4. **Instruction-tuned model outperforms base model**:
   - Flan-T5-small achieves better absolute scores than GPT-2
   - Understands QA task format better

### Improvements Over Experiment 1

| Aspect | Exp 1 | Exp 2 |
|--------|-------|-------|
| LLM | GPT-2 | flan-t5-small |
| Metric | Substring match | F1 + Exact Match |
| Parameters | 2 (chunk, k) | 3 (chunk, k, temp) |
| Baseline fitness | N/A | 0.170 |
| Best fitness | N/A | 0.284 |

## Experiment 3: Multi-Method Comparison

**Date**: 2026-02-17

### Setup

- **Dataset**: HotpotQA validation (15 samples)
- **Evaluation budget**: 15 evaluations per method (fair comparison)
- **Methods**: Hand-tuned baseline, Random Search, Grid Search, Optuna (TPE), Evolution
- **LLM**: google/flan-t5-small (local CPU)
- **Search space**: chunk_size {128,256,512,1024,2048}, top_k [1-12], temperature [0.1-1.5]
- **Fitness function**: 0.6*F1 + 0.3*EM + 0.1*(1-latency)

### Results

| Method | Best Fitness | Best Config | Time |
|--------|-------------|-------------|------|
| Hand-tuned baseline | 0.125 | chunk=512, k=5, t=0.3 | -- |
| Grid Search | 0.401 | chunk=128, k=5, t=1.2 | 138s |
| Optuna (Bayesian/TPE) | 0.431 | chunk=2048, k=10, t=0.4 | 142s |
| Evolution | 0.500 | chunk=2048, k=2, t=0.93 | 159s |
| Random Search | 0.595 | chunk=128, k=11, t=1.14 | 147s |

All methods use the same evaluation budget (15 configs evaluated) for fair comparison.

### Analysis

1. **All methods beat the hand-tuned baseline** (0.125) by a large margin, confirming that the default RAG parameters are far from optimal.

2. **Random search won this round** (0.595) — this is expected at small budgets. With only 15 evaluations, random search can get lucky. Research by Bergstra & Bengio (2012) showed random search is competitive with grid search and sometimes beats more sophisticated methods at low budgets.

3. **Evolution placed second** (0.500) and showed structured convergence: it identified chunk=2048, k=2 as a strong region and exploited it across generations. With more budget, this structured search would likely outperform random.

4. **Optuna (TPE)** achieved 0.431. Its Bayesian model needs more evaluations to build an accurate surrogate, so 15 trials limits its advantage.

5. **Grid Search** (0.401) is limited by the discretization of the search space.

### Key Takeaway

At small evaluation budgets, simple methods compete with sophisticated optimization. The value of evolutionary search emerges at larger scales where its ability to combine good partial solutions (crossover) and exploit promising regions (selection pressure) provides an advantage over random sampling.

### Limitations

- 15 samples is too small for statistical significance
- 15 evaluations per method is a very low budget
- Single seed (no variance estimates)
- Need multiple runs with different seeds to draw robust conclusions

## Visualizing Results

Run the Streamlit dashboard to see interactive charts:

```bash
streamlit run app/dashboard.py
```

Run the multi-method comparison:

```bash
cd experiments && python run_comparison.py
```

## Next Steps

1. Scale to 50+ samples and 50+ evaluations per method for significance
2. Run 5+ seeds and report mean +/- std
3. Test on NaturalQuestions and TriviaQA
4. Try larger models (flan-t5-base/large) with GPU
5. Add RAGAS metrics (faithfulness, relevancy) when API available
6. Investigate at what evaluation budget evolution overtakes random search
