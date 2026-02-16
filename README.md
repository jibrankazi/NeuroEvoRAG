# NeuroEvoRAG

**Neuro-Evolutionary Retrieval-Augmented Generation**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Status: Early development / research prototype**

NeuroEvoRAG explores using NEAT (NeuroEvolution of Augmenting Topologies) to automatically optimize RAG pipelines. It evolves components like chunkers, embedders, and retrievers for multi-hop question answering.

## Goals

- Use evolutionary algorithms to tune RAG hyperparameters
- Support text, image, and audio retrieval (planned)
- Evolve pipelines based on faithfulness, latency, and cost
- Provide a framework for RAG architecture search

## What's working

- NEAT configuration and custom RAGGenome with hyperparameter encoding
- Agent classes (Retriever, Critic, Synthesizer)
- Dataset download utilities for HotpotQA, MMQA, and others
- GitHub Actions workflows
- NEAT-based evolution loop
- RAGAS evaluation metrics integration
- Working baseline RAG pipeline (no API keys needed)
- Test suite (80 unit tests, runs without heavy deps)
- Evolution experiment (chunk=256, k=2 beat baseline by 10%)

## In progress

- Multimodal retrieval implementations
- Additional chunking strategies
- Vector database integrations beyond ChromaDB
- Larger-scale experiments with better LLMs

## Planned

- Dashboard for visualizing evolution progress
- Comprehensive evaluation on multiple benchmarks

## Quick start

### Prerequisites
- Python 3.11+
- (Optional) GPU for faster model inference
- Hugging Face account and token for dataset downloads

### Installation

```bash
git clone https://github.com/jibrankazi/NeuroEvoRAG.git
cd NeuroEvoRAG

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Usage

```bash
# Download benchmark datasets (requires HF_TOKEN)
export HF_TOKEN=your_huggingface_token
python benchmarks/download_datasets.py

# Run evolution
bash run_evolution.sh
```

## Project structure

```
NeuroEvoRAG/
├── agents/                 # Retriever, Critic, Synthesizer agents
├── rag_pipelines/         # Chunker, retriever, generator components
├── evolution/             # NEAT genome, evolution loop, fitness
├── benchmarks/            # Dataset download and RAGAS evaluation
├── examples/              # Working baseline pipeline
├── tests/                 # Unit tests
├── app/                   # Streamlit dashboard (planned)
└── experiments/           # Analysis and results
```

## Tech stack

- **Evolution**: NEAT-Python
- **RAG frameworks**: LangChain, LlamaIndex
- **LLMs**: OpenAI, Anthropic, open-source (flan-t5-small for baseline)
- **Vector storage**: ChromaDB (Milvus, Qdrant planned)
- **Evaluation**: RAGAS metrics
- **Orchestration**: LangGraph (planned)

## Background

Draws from GraphRAG, Self-RAG, NEAT, and Population-Based Training.

## Contributing

See `CONTRIBUTING.md`. Issues, PRs, and ideas are welcome.

## License

MIT License - see LICENSE file for details.

## Contact

GitHub: [@jibrankazi](https://github.com/jibrankazi)
