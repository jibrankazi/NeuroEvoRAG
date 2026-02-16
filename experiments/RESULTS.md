# Experimental Results

Pending first experiment run.

## Planned experiment

- **Dataset**: HotpotQA (20 validation examples)
- **Baseline**: chunk_size=512, top_k=5
- **Evolution**: 10 generations, population size 50
- **Evolved parameters**: chunk_size, top_k
- **Metrics**: RAGAS (faithfulness, answer_relevancy, context_precision)

## Next steps

1. Scale to larger dataset (100+ examples)
2. Add more evolvable parameters
3. Run multiple seeds for statistical validation
4. Compare against standard baselines
