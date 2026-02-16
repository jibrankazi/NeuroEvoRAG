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

## Next Steps

1. Use instruction-tuned LLM (flan-t5, llama) for better QA
2. Increase sample size to 50+ for stability
3. Add RAGAS metrics (faithfulness, relevancy) instead of substring match
4. Run multiple seeds and report mean/std
5. Add temperature as evolvable parameter
