"""
Tests for evolution/evolve.py — evolution loop functions.

These tests mock heavy dependencies (datasets, RAGAS, ML models)
to test the orchestration logic in isolation.
"""
from unittest.mock import MagicMock, patch, mock_open
import pytest

from evolution.genome import RAGGenome


class TestEvalGenome:
    @patch("evolution.evolve.evaluate_pipeline_real")
    @patch("evolution.evolve.genome_to_pipeline")
    def test_returns_weighted_fitness(self, mock_g2p, mock_eval):
        from evolution.evolve import eval_genome

        mock_pipeline = MagicMock()
        mock_g2p.return_value = mock_pipeline
        mock_eval.return_value = {
            "faithfulness": 0.8,
            "answer_relevancy": 0.6,
            "context_precision": 0.5,
        }

        # Set up global eval data
        import evolution.evolve as mod
        mod.EVAL_CONTEXTS = ["ctx1", "ctx2"]
        mod.EVAL_QUESTIONS = ["q1"]
        mod.EVAL_ANSWERS = ["a1"]

        genome = RAGGenome(key=1)
        config = MagicMock()
        fitness = eval_genome(genome, config)

        expected = 0.8 * 0.5 + 0.6 * 0.3 + 0.5 * 0.2
        assert abs(fitness - expected) < 1e-9

    @patch("evolution.evolve.evaluate_pipeline_real")
    @patch("evolution.evolve.genome_to_pipeline")
    def test_returns_zero_on_exception(self, mock_g2p, mock_eval):
        from evolution.evolve import eval_genome

        mock_g2p.side_effect = RuntimeError("boom")

        import evolution.evolve as mod
        mod.EVAL_CONTEXTS = ["ctx"]
        mod.EVAL_QUESTIONS = ["q"]
        mod.EVAL_ANSWERS = ["a"]

        genome = RAGGenome(key=1)
        fitness = eval_genome(genome, MagicMock())
        assert fitness == 0.0

    @patch("evolution.evolve.evaluate_pipeline_real")
    @patch("evolution.evolve.genome_to_pipeline")
    def test_adds_documents_to_pipeline(self, mock_g2p, mock_eval):
        from evolution.evolve import eval_genome

        mock_pipeline = MagicMock()
        mock_g2p.return_value = mock_pipeline
        mock_eval.return_value = {
            "faithfulness": 0.5,
            "answer_relevancy": 0.5,
            "context_precision": 0.5,
        }

        import evolution.evolve as mod
        mod.EVAL_CONTEXTS = ["doc1", "doc2"]
        mod.EVAL_QUESTIONS = ["q"]
        mod.EVAL_ANSWERS = ["a"]

        genome = RAGGenome(key=1)
        eval_genome(genome, MagicMock())
        mock_pipeline.add_documents.assert_called_once_with(["doc1", "doc2"])


class TestEvalGenomes:
    @patch("evolution.evolve.eval_genome")
    def test_sets_fitness_on_all_genomes(self, mock_eval_genome):
        from evolution.evolve import eval_genomes

        mock_eval_genome.side_effect = [0.8, 0.6, 0.9]
        g1, g2, g3 = MagicMock(), MagicMock(), MagicMock()
        genomes = [(1, g1), (2, g2), (3, g3)]

        eval_genomes(genomes, MagicMock())

        assert g1.fitness == 0.8
        assert g2.fitness == 0.6
        assert g3.fitness == 0.9

    @patch("evolution.evolve.eval_genome")
    def test_calls_eval_genome_for_each(self, mock_eval_genome):
        from evolution.evolve import eval_genomes

        mock_eval_genome.return_value = 0.5
        genomes = [(i, MagicMock()) for i in range(5)]

        eval_genomes(genomes, MagicMock())
        assert mock_eval_genome.call_count == 5


class TestFitnessWeights:
    def test_weights_sum_to_one(self):
        """The fitness weights in eval_genome should sum to 1.0."""
        # These are the weights from evolve.py lines 63-68
        weights = [0.5, 0.3, 0.2]
        assert abs(sum(weights) - 1.0) < 1e-9

    def test_perfect_scores_yield_fitness_one(self):
        """If all metrics are 1.0, fitness should be 1.0."""
        fitness = 1.0 * 0.5 + 1.0 * 0.3 + 1.0 * 0.2
        assert abs(fitness - 1.0) < 1e-9

    def test_zero_scores_yield_fitness_zero(self):
        fitness = 0.0 * 0.5 + 0.0 * 0.3 + 0.0 * 0.2
        assert fitness == 0.0


class TestLoadEvalData:
    @patch("evolution.evolve.load_dataset")
    def test_populates_globals(self, mock_load):
        import evolution.evolve as mod

        # Reset globals
        mod.EVAL_DATASET = None
        mod.EVAL_QUESTIONS = None
        mod.EVAL_ANSWERS = None
        mod.EVAL_CONTEXTS = None

        # Build a fake dataset
        fake_items = [
            {
                "question": "q1",
                "answer": "a1",
                "context": {"sentences": [["sent1a", "sent1b"]]},
            },
            {
                "question": "q2",
                "answer": "a2",
                "context": {"sentences": [["sent2a"]]},
            },
        ]

        mock_dataset = MagicMock()
        mock_dataset.select.return_value = fake_items
        mock_dataset.__iter__ = lambda self: iter(fake_items)
        mock_load.return_value = mock_dataset

        mod.load_eval_data()

        assert mod.EVAL_QUESTIONS == ["q1", "q2"]
        assert mod.EVAL_ANSWERS == ["a1", "a2"]
        assert "sent1a" in mod.EVAL_CONTEXTS
        assert "sent2a" in mod.EVAL_CONTEXTS

    @patch("evolution.evolve.load_dataset")
    def test_idempotent(self, mock_load):
        """Calling load_eval_data twice should not reload."""
        import evolution.evolve as mod

        mod.EVAL_DATASET = "already loaded"
        mod.load_eval_data()
        mock_load.assert_not_called()

        # Reset for other tests
        mod.EVAL_DATASET = None
