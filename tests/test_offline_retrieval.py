from benchmarks.offline_retrieval import split_words, retrieval_accuracy, optimize
import pytest


def test_chunking():
    assert split_words("one two three", 2) == ["one two", "three"]


def test_accuracy_bound():
    assert 0 <= retrieval_accuracy(6, 2) <= 1


def test_optimization_improves_or_matches_baseline():
    out = optimize(generations=3, seed=42)
    assert out["best_accuracy"] >= out["baseline_accuracy"]
    assert len(out["history"]) == 3
    assert optimize(generations=3, seed=42) == out


def test_invalid():
    with pytest.raises(ValueError):
        optimize(generations=0)
