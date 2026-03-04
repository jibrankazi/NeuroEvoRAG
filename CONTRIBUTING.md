# Contributing to NeuroEvoRAG

Thank you for your interest in contributing to NeuroEvoRAG! This document provides guidelines and information for contributors.

## 🌟 Ways to Contribute

- **Code**: Implement missing features, fix bugs, improve performance
- **Documentation**: Improve README, add docstrings, create tutorials
- **Research**: Experiment with different evolution strategies, benchmark datasets
- **Testing**: Add unit tests, integration tests, validation scripts
- **Ideas**: Suggest new features, evaluation metrics, or research directions

## 🚀 Getting Started

### 1. Set Up Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/NeuroEvoRAG.git
cd NeuroEvoRAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy jupyter
```

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 3. Make Your Changes

Follow the code style guidelines below and ensure your changes:
- Are well-documented with docstrings
- Include type hints where appropriate
- Have descriptive commit messages
- Don't break existing functionality

### 4. Test Your Changes

```bash
# Run any existing tests
pytest tests/

# Test your specific changes manually
python your_module.py
```

### 5. Submit a Pull Request

- Push your branch to your fork
- Create a PR against the `main` branch
- Describe what you changed and why
- Reference any related issues

## 📝 Code Style Guidelines

### Python Style

We follow PEP 8 with some modifications:

```python
# Use type hints
def evaluate_pipeline(genome: RAGGenome, dataset: List[Dict]) -> float:
    """Evaluate a genome on a dataset.
    
    Args:
        genome: The genome to evaluate
        dataset: List of evaluation examples
        
    Returns:
        Fitness score (higher is better)
    """
    pass

# Use descriptive variable names
chunk_size = 512  # Good
cs = 512  # Bad

# Document complex logic
# Calculate fitness as weighted sum of metrics
fitness = (
    faithfulness * 0.4 +      # Most important: factual accuracy
    (1.0 / latency) * 0.3 +   # Speed matters
    -cost * 0.2 +              # Lower cost is better
    diversity * 0.1            # Encourage exploration
)
```

### Docstring Format

Use Google-style docstrings:

```python
def my_function(arg1: int, arg2: str) -> bool:
    """Short description of function.
    
    Longer description if needed, explaining the purpose,
    algorithm, or important details.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When invalid input is provided
        
    Example:
        >>> my_function(5, "test")
        True
    """
    pass
```

### Module Organization

```python
# 1. Standard library imports
import os
from pathlib import Path
from typing import List, Dict, Any

# 2. Third-party imports
import neat
import torch
from ragas import evaluate

# 3. Local imports
from .genome import RAGGenome
from .reward_model import evaluate_pipeline
```

## 🧪 Testing Guidelines

### Unit Tests

Write unit tests for individual functions:

```python
# tests/test_genome.py
import pytest
from evolution.genome import RAGGenome, genome_to_pipeline

def test_genome_initialization():
    genome = RAGGenome(1)
    assert genome.chunk_size in [128, 256, 512, 1024, 2048]
    assert 1 <= genome.top_k <= 10

def test_genome_to_pipeline():
    genome = RAGGenome(1)
    config = genome_to_pipeline(genome)
    assert "chunk_size" in config
    assert "top_k" in config
```

### Integration Tests

Test component interactions:

```python
# tests/test_pipeline.py
def test_full_pipeline():
    """Test that a pipeline can process a question end-to-end."""
    pipeline = create_test_pipeline()
    question = "What is the capital of France?"
    answer = pipeline.run(question)
    assert isinstance(answer, str)
    assert len(answer) > 0
```

## 📚 Documentation Guidelines

### Code Comments

```python
# Use comments to explain WHY, not WHAT
# Good: "Use exponential decay to favor recent mutations"
# Bad: "Set rate to 0.95"
mutation_rate *= 0.95

# Comment complex algorithms
# Implement tournament selection (k=3):
# 1. Sample k genomes randomly
# 2. Select the fittest one
# 3. Repeat until population is filled
```

### README Updates

When adding new features:
1. Update the feature list in README.md
2. Add usage examples if applicable
3. Update the tech stack section if new dependencies are added

## 🔍 Code Review Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guidelines
- [ ] All functions have docstrings
- [ ] Complex logic is commented
- [ ] Type hints are used
- [ ] No TODOs left in production code
- [ ] Tests pass (if applicable)
- [ ] Documentation is updated
- [ ] Commit messages are descriptive

## 🐛 Reporting Issues

When reporting bugs, include:

1. **Environment**:
   - OS (Windows/Linux/Mac)
   - Python version
   - Relevant package versions

2. **Steps to Reproduce**:
   ```bash
   python script.py --arg value
   # Error occurs here
   ```

3. **Expected vs Actual Behavior**:
   - What you expected to happen
   - What actually happened

4. **Error Messages**:
   ```
   Full error traceback here
   ```

5. **What You've Tried**:
   - Any debugging steps you've already taken

## 💡 Feature Requests

When suggesting features:

1. Describe the problem you're trying to solve
2. Explain why this feature would be useful
3. Propose a potential implementation approach
4. Consider edge cases and limitations

## 🎯 Priority Areas

Current priority areas for contribution:

### High Priority
- Genome-to-pipeline conversion implementation
- RAGAS metric integration
- Basic embedding and retrieval working
- First end-to-end pipeline execution

### Medium Priority
- Additional chunking strategies
- Vector database integrations
- Latency and cost tracking
- Evaluation notebooks

### Low Priority
- Multimodal support
- Advanced visualization
- Dashboard development
- Additional benchmarks

## 📲 Getting Help

- Open an issue for bugs or questions
- Start a discussion for ideas or research questions
- Check existing issues and documentation first

## 🙏 Recognition

Contributors will be:
- Listed in the project README
- Credited in any resulting publications
- Appreciated for their time and effort!

---

Thank you for helping make NeuroEvoRAG better! 🚀
