# NeuroEvoRAG 🚀

**Neuro-Evolutionary Retrieval-Augmented Generation: An Experimental Framework for Self-Optimizing RAG Pipelines**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

⚠️ **Project Status: Early Development / Research Prototype** ⚠️

NeuroEvoRAG is an experimental research project exploring how neuroevolution (specifically NEAT - NeuroEvolution of Augmenting Topologies) can be applied to automatically optimize RAG (Retrieval-Augmented Generation) pipelines. The core idea is to evolve components like chunkers, embedders, and retrievers to improve performance on multi-hop question answering tasks.

## 🎯 Project Goals

- **Automated Pipeline Optimization**: Use evolutionary algorithms to tune RAG hyperparameters
- **Multimodal Support**: Design for text, image, and audio retrieval (planned)
- **Metrics-Driven Evolution**: Evolve pipelines based on faithfulness, latency, and cost metrics
- **Research Platform**: Provide a framework for experimenting with RAG architecture search

## 🚧 Current Implementation Status

This is a **research prototype** with the following components:

### ✅ Implemented
- Basic project structure and module organization
- NEAT configuration for evolutionary optimization
- Placeholder agent classes (Retriever, Critic, Synthesizer)
- Dataset download utilities for HotpotQA, MMQA, and others
- GitHub Actions workflows for CI/CD

### 🚧 In Progress
- NEAT-based evolution loop
- Integration with RAGAS evaluation metrics
- Multimodal retrieval implementations
- Complete RAG pipeline implementations

### 📋 Planned
- Actual neuroevolution experiments with baseline comparisons
- Dashboard for visualizing evolution progress
- Integration with vector databases (Milvus, Qdrant, Chroma)
- Comprehensive evaluation on multiple benchmarks
- Publication-ready results and analysis

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- (Optional) GPU for faster model inference
- Hugging Face account and token for dataset downloads

### Installation

```bash
# Clone the repository
git clone https://github.com/jibrankazi/NeuroEvoRAG.git
cd NeuroEvoRAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Download benchmark datasets (requires HF_TOKEN environment variable)
export HF_TOKEN=your_huggingface_token
python benchmarks/download_datasets.py

# Run a basic evolution experiment (work in progress)
bash run_evolution.sh
```

## 📊 Project Structure

```
NeuroEvoRAG/
├── agents/                 # Agent classes for agentic RAG
│   ├── CriticAgent.py     # Answer evaluation agent
│   ├── RetrieverAgent.py  # Retrieval orchestration agent
│   └── SynthesizerAgent.py # Context synthesis agent
├── rag_pipelines/         # RAG component implementations
│   ├── dynamic_chunker.py # Evolvable chunking strategies
│   ├── multimodal_retriever.py # Multi-modal retrieval
│   └── agentic_generator.py # LLM-based generation
├── evolution/             # Neuroevolution engine
│   ├── genome.py         # NEAT genome definition
│   ├── evolve.py         # Main evolution loop
│   ├── reward_model.py   # Fitness evaluation
│   └── neat_config.txt   # NEAT hyperparameters
├── benchmarks/           # Evaluation datasets and scripts
│   ├── download_datasets.py
│   └── eval_suite.py
├── notebooks/            # Jupyter notebooks for experiments
├── app/                  # Streamlit dashboard (planned)
└── mutation_zoo/         # Mutation operators library (planned)
```

## 🛠 Tech Stack

- **Evolutionary Algorithm**: NEAT-Python
- **RAG Frameworks**: LangChain, LlamaIndex
- **LLMs**: OpenAI GPT-4, Anthropic Claude, Open-source models
- **Vector Databases**: Milvus, Qdrant, ChromaDB (integration planned)
- **Evaluation**: RAGAS metrics
- **Orchestration**: LangGraph (planned)

## 📖 Research Context

This project draws inspiration from:
- **GraphRAG**: Knowledge graph-enhanced retrieval
- **Self-RAG**: Self-reflective retrieval augmentation
- **NEAT**: NeuroEvolution of Augmenting Topologies
- **Population-Based Training**: Hyperparameter optimization through evolution

## 🤝 Contributing

This is a research project in active development. Contributions, suggestions, and discussions are welcome! Please feel free to:
- Open issues for bugs or feature requests
- Submit pull requests with improvements
- Share ideas for experiments or evaluations

## 📝 License

MIT License - see LICENSE file for details

## 🔗 Contact

- GitHub: [@jibrankazi](https://github.com/jibrankazi)

---

**Note**: This is an experimental research project. Results and claims should be verified independently. The project is under active development and APIs may change.
