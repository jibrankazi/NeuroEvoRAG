## LinkedIn Post (copy-paste below)

---

**Can we evolve better RAG pipelines instead of hand-tuning them?**

I built NeuroEvoRAG -- a framework that applies evolutionary optimization to Retrieval-Augmented Generation hyperparameters. Instead of guessing chunk sizes and retrieval depths, the system evolves them automatically.

**The problem:**
RAG pipelines depend on hyperparameters (chunk size, retrieval depth, temperature) that are typically set by rules of thumb. These defaults are rarely optimal.

**What I built:**
- End-to-end RAG pipeline with ChromaDB + sentence-transformers + flan-t5
- Evolutionary search over 3 parameters: chunk_size, top_k, temperature
- Fair comparison against 3 baselines: Optuna (Bayesian/TPE), Grid Search, Random Search
- 80 unit tests, Streamlit dashboard, reproducible experiments

**Results on HotpotQA (multi-hop QA):**

| Method | Best Fitness | vs Baseline |
|--------|-------------|-------------|
| Hand-tuned default | 0.125 | -- |
| Grid Search | 0.401 | +221% |
| Optuna (TPE) | 0.431 | +245% |
| Evolution | 0.500 | +300% |
| Random Search | 0.595 | +376% |

**Key findings:**
1. Every automated method crushed the hand-tuned baseline by 3-4x
2. At small evaluation budgets, random search is competitive (consistent with Bergstra & Bengio, 2012)
3. Evolution showed structured convergence, identifying two distinct retrieval strategies: large-chunk/low-k vs small-chunk/high-k
4. The conventional chunk_size=512, k=5 default is a poor compromise

**Honest limitations:** Small-scale experiment (15 samples, 15 evals per method). Needs larger experiments and multiple seeds for statistical significance. Paper in progress.

**Tech stack:** Python, ChromaDB, sentence-transformers, HuggingFace, Optuna, flan-t5-small, Streamlit, pytest

Code: github.com/jibrankazi/NeuroEvoRAG

#MachineLearning #NLP #RAG #EvolutionaryAlgorithms #HyperparameterOptimization #Research

---

## Posting instructions:

1. Copy the text above
2. Attach images in this order:
   - 1_comparison.png (main results chart -- attach first, this is the hook)
   - 3_architecture.png (system architecture)
   - 2_improvement.png (improvement percentages)
   - 4_parameters.png (parameter analysis)
3. Post on LinkedIn

## Why this post works:

- Opens with a question (hooks readers)
- States the problem clearly
- Shows real numbers (not vague claims)
- Includes honest limitations (builds credibility)
- Has a results table (easy to scan)
- Links to code (verifiable)
- No overclaiming or fabricated metrics
