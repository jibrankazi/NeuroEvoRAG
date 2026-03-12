# NeuroEvoRAG

**Evolutionary Optimization of Retrieval-Augmented Generation Pipelines**

## Overview

NeuroEvoRAG empirically compares four hyperparameter optimization methods 
for RAG pipelines on multi-hop question answering. Rather than hand-tuning 
chunk sizes, retrieval depths, and temperatures, the system evaluates 
evolutionary search, Bayesian optimization (Optuna/TPE), grid search, and 
random search under equal evaluation budgets.

## Key Finding

All automated methods dramatically outperform hand-tuned defaults. At small 
budgets (15 evaluations), random search is competitive with evolution — 
consistent with Bergstra & Bengio (2012). Evolution shows structured 
convergence and identifies promising regions across generations.

## Results

Evaluated on HotpotQA multi-hop QA with equal budget (15 evaluations):

| Method | Best Fitness | vs Baseline |
|---|---|---|
| Hand-tuned baseline | 0.125 | -- |
| Grid Search | 0.401 | +221% |
| Optuna (TPE) | 0.431 | +245% |
| Evolution | 0.500 | +300% |
| Random Search | 0.595 | +376% |

Fitness = 0.6 × F1 + 0.3 × Exact_Match + 0.1 × (1 - latency)

## Method

**Search Space:**
- `chunk_size` ∈ {128, 256, 512, 1024, 2048}
- `top_k` ∈ [1, 12]
- `temperature` ∈ [0.1, 1.5]

**Evolution:** Tournament selection, uniform crossover, Gaussian 
mutation, elitism.

**Two retrieval strategies emerged:**
- Large chunks + few retrievals (chunk=2048, k=2)
- Small chunks + many retrievals (chunk=128, k=11)

## Stack

| Component | Technology |
|---|---|
| LLM | flan-t5-small (local, no API key) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB |
| Bayesian Opt | Optuna (TPE) |
| Dataset | HotpotQA |
| Dashboard | Streamlit |
| Tests | pytest (80 tests) |

## Quick Start
```bash
git clone https://github.com/jibrankazi/NeuroEvoRAG.git
cd NeuroEvoRAG
pip install -r requirements.txt
cd experiments && python run_comparison.py
streamlit run app/dashboard.py
```

## Structure
```
NeuroEvoRAG/
├── evolution/          # Genome, evolution loop, fitness
├── rag_pipelines/      # Chunking, retrieval, generation
├── agents/             # Retriever, Critic, Synthesizer
├── experiments/        # 4-method comparison + RESULTS.md
├── benchmarks/         # RAGAS evaluation suite
├── app/                # Streamlit dashboard
├── paper/              # 4-page workshop paper (LaTeX, 16 citations)
└── tests/              # 80 unit tests
```

## Limitations

- Small scale: 15 samples, 15 evaluations per method, single seed
- CPU-only inference (flan-t5-small)
- Single dataset (HotpotQA)
- Budget threshold where evolution beats random search not yet determined

## Future Work

- Scale to 50+ samples with multiple seeds
- GPU experiments with larger models
- RAGAS faithfulness and relevancy metrics
- Test on NaturalQuestions and TriviaQA

## Paper

A 4-page workshop paper with literature review, methodology, results, 
and limitations is in `paper/main.tex` (16 citations). 
Compile with LaTeX or upload to Overleaf.

## Citation
```
@misc{kazi2026neuroevorag,
  title={Evolutionary Optimization of RAG Pipelines},
  author={Kazi, Jibran},
  year={2026},
  url={https://github.com/jibrankazi/NeuroEvoRAG}
}
```

## License MIT

---
**Kazi Jibran Rafat Samie** | Toronto, Canada | 
jibrankazi@gmail.com | github.com/jibrankazi
