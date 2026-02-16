# Contributing to NeuroEvoRAG

## Ways to contribute

- **Code**: implement missing features, fix bugs, improve performance
- **Testing**: add unit tests, integration tests
- **Research**: experiment with different evolution strategies, benchmark datasets
- **Ideas**: suggest features, evaluation metrics, or research directions

## Getting started

### Set up development environment

```bash
git clone https://github.com/YOUR_USERNAME/NeuroEvoRAG.git
cd NeuroEvoRAG

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install pytest pytest-mock black flake8
```

### Create a branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### Test your changes

```bash
pytest tests/
```

### Submit a pull request

- Push your branch to your fork
- Create a PR against `main`
- Describe what changed and why
- Reference related issues if any

## Code style

PEP 8 with type hints. Use `black` for formatting.

Keep docstrings short. Don't document what's obvious from the signature.

### Import order

```python
import os
from pathlib import Path

import neat
import torch

from .genome import RAGGenome
```

## Testing

Put tests in `tests/`. The suite runs without heavy dependencies
(no GPU, no API keys, no large downloads). See `tests/conftest.py` for how
third-party packages are stubbed.

```bash
pytest tests/ -v
```

## Priority areas

**High**: genome-to-pipeline conversion, RAGAS integration, end-to-end execution

**Medium**: chunking strategies, vector DB integrations, latency/cost tracking

**Low**: multimodal support, dashboard, additional benchmarks

## Reporting bugs

Include: OS/Python version, steps to reproduce, expected vs actual behavior, full traceback.

## Contact

Open an issue on GitHub or start a discussion.
