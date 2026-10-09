# Real-data retrieval benchmark (current)

**This repository's validated retrieval experiment now uses real Stanford SQuAD 1.1 Wikipedia passages and crowdsourced question annotations**, not constructed sample passages. Source: https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json (CC BY-SA 4.0).

```bash
pip install requests scikit-learn numpy pytest
python -m benchmarks.offline_retrieval --max-questions 80 --generations 3 --population-size 5 --output results/squad_retrieval.json
```

The experiment samples real questions deterministically, optimizes retrieval chunk length/top-k against a training subset, then reports held-out retrieval recall. **This is not proof of answer faithfulness, neural model training, RAGAS/NEAT convergence, or the numerical claims elsewhere in this repository.** The `examples/basic_rag_working.py` example also loads real SQuAD passages but requires configured model credentials and separate end-to-end testing.

Unit tests use limited parser and metric fixtures for software correctness; empirical results must originate from the downloaded dataset. The old synthetic experiment has been replaced. The earlier documentation below is historical background.

---

# NeuroEvoRAG — validated offline retrieval experiment

**Implemented and tested in the new CI:** a seeded, synthetic TF-IDF retrieval benchmark with actual evaluation and a simple evolutionary search over chunk sizes and retrieval depths. Unlike the previous smoke workflow, this runs code and checks measurable outputs.

```sh
pip install numpy scikit-learn pytest
python -m pytest -q tests/test_offline_retrieval.py
python -m benchmarks.offline_retrieval
```

**Not verified end-to-end:** production RAG generation, LLM integrations, HotpotQA results in the original README, RAGAS/NEAT performance, 80 claimed tests, and the README's optimizer-comparison figures. The standalone offline benchmark **does not establish** those claims. Additional workflows which download external datasets require independent setup and may fail.

The original research concept and unverified historical figures are retained below for context only.

---

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
