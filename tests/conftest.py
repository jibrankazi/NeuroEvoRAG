"""
Shared fixtures for NeuroEvoRAG test suite.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure project root is on sys.path so imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def mock_retriever():
    """A mock retriever that returns canned results."""
    retriever = MagicMock()
    retriever.retrieve.return_value = ["chunk_a", "chunk_b", "chunk_c"]
    return retriever


@pytest.fixture
def mock_evaluator():
    """A mock evaluator function that returns a fixed score."""
    return MagicMock(return_value=0.85)


@pytest.fixture
def mock_llm():
    """A mock LLM callable that returns a fixed answer."""
    return MagicMock(return_value="Mock LLM answer based on context.")
