"""
Shared fixtures and dependency stubs for NeuroEvoRAG test suite.

Heavy optional dependencies (neat-python, ragas, datasets, sentence_transformers,
chromadb, transformers) are stubbed out at the sys.modules level so that tests
can run in *any* environment — even one with only pytest installed.
"""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# 1.  Ensure project root is on sys.path so source imports work
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# 2.  Stub out heavy third-party packages that may not be installed.
#     Each stub is only injected when the real package is *missing*.
# ---------------------------------------------------------------------------

def _ensure_module(name: str, attrs: dict | None = None):
    """Insert a fake module into sys.modules if the real one is absent."""
    if name not in sys.modules:
        mod = types.ModuleType(name)
        for attr, value in (attrs or {}).items():
            setattr(mod, attr, value)
        sys.modules[name] = mod
    return sys.modules[name]


# -- neat-python -------------------------------------------------------------
_ensure_module("neat", {"DefaultGenome": type("DefaultGenome", (), {
    "__init__": lambda self, key: setattr(self, "key", key),
    "configure_new": lambda self, config: None,
    "mutate": lambda self, config: None,
})})

# -- ragas & sub-modules -----------------------------------------------------
_ensure_module("ragas", {"evaluate": MagicMock()})
_ragas_metrics = _ensure_module("ragas.metrics", {
    "faithfulness": MagicMock(),
    "answer_relevancy": MagicMock(),
    "context_precision": MagicMock(),
})
_ensure_module("ragas.metrics.collections", {
    "faithfulness": MagicMock(),
    "answer_relevancy": MagicMock(),
    "context_precision": MagicMock(),
})

# -- datasets ----------------------------------------------------------------
_ensure_module("datasets", {
    "load_dataset": MagicMock(),
    "Dataset": MagicMock(),
})

# -- sentence_transformers ---------------------------------------------------
_ensure_module("sentence_transformers", {
    "SentenceTransformer": MagicMock(),
})

# -- chromadb ----------------------------------------------------------------
_ensure_module("chromadb", {"Client": MagicMock()})

# -- transformers ------------------------------------------------------------
_ensure_module("transformers", {"pipeline": MagicMock()})

# ---------------------------------------------------------------------------
# 3.  Shared test fixtures
# ---------------------------------------------------------------------------

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
