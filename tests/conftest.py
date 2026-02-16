import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _ensure_module(name: str, attrs: dict | None = None):
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


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.retrieve.return_value = ["chunk_a", "chunk_b", "chunk_c"]
    return retriever


@pytest.fixture
def mock_evaluator():
    return MagicMock(return_value=0.85)


@pytest.fixture
def mock_llm():
    return MagicMock(return_value="Mock LLM answer based on context.")
