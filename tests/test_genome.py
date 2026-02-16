import random
from unittest.mock import MagicMock, patch

import pytest

from evolution.genome import RAGGenome, genome_to_pipeline

VALID_CHUNK_SIZES = [128, 256, 512, 1024, 2048]


class TestRAGGenomeInit:
    def test_default_chunk_size(self):
        genome = RAGGenome(key=1)
        assert genome.chunk_size == 512

    def test_default_top_k(self):
        genome = RAGGenome(key=1)
        assert genome.top_k == 5

    def test_default_temperature(self):
        genome = RAGGenome(key=1)
        assert genome.temperature == 0.7

    def test_key_is_stored(self):
        genome = RAGGenome(key=42)
        assert genome.key == 42


class TestRAGGenomeConfigureNew:
    def test_chunk_size_in_valid_set(self):
        genome = RAGGenome(key=1)
        config = MagicMock()
        # Mock the parent configure_new to avoid NEAT config requirements
        with patch.object(RAGGenome.__bases__[0], "configure_new"):
            for _ in range(50):
                genome.configure_new(config)
                assert genome.chunk_size in VALID_CHUNK_SIZES

    def test_top_k_in_range(self):
        genome = RAGGenome(key=1)
        config = MagicMock()
        with patch.object(RAGGenome.__bases__[0], "configure_new"):
            for _ in range(50):
                genome.configure_new(config)
                assert 3 <= genome.top_k <= 10

    def test_temperature_in_range(self):
        genome = RAGGenome(key=1)
        config = MagicMock()
        with patch.object(RAGGenome.__bases__[0], "configure_new"):
            for _ in range(50):
                genome.configure_new(config)
                assert 0.3 <= genome.temperature <= 1.0


class TestRAGGenomeMutate:
    def _make_genome(self):
        genome = RAGGenome(key=1)
        genome.chunk_size = 512
        genome.top_k = 5
        genome.temperature = 0.7
        return genome

    def test_chunk_size_stays_valid_after_mutation(self):
        config = MagicMock()
        with patch.object(RAGGenome.__bases__[0], "mutate"):
            for _ in range(100):
                genome = self._make_genome()
                genome.mutate(config)
                assert genome.chunk_size in VALID_CHUNK_SIZES

    def test_top_k_stays_in_bounds(self):
        config = MagicMock()
        with patch.object(RAGGenome.__bases__[0], "mutate"):
            genome = self._make_genome()
            genome.top_k = 1  # start at lower boundary
            for _ in range(200):
                genome.mutate(config)
                assert 1 <= genome.top_k <= 15

    def test_top_k_upper_boundary(self):
        config = MagicMock()
        with patch.object(RAGGenome.__bases__[0], "mutate"):
            genome = self._make_genome()
            genome.top_k = 15  # start at upper boundary
            for _ in range(200):
                genome.mutate(config)
                assert 1 <= genome.top_k <= 15

    def test_temperature_stays_in_bounds(self):
        config = MagicMock()
        with patch.object(RAGGenome.__bases__[0], "mutate"):
            genome = self._make_genome()
            genome.temperature = 0.1  # start at lower boundary
            for _ in range(200):
                genome.mutate(config)
                assert 0.1 <= genome.temperature <= 1.5

    def test_temperature_upper_boundary(self):
        config = MagicMock()
        with patch.object(RAGGenome.__bases__[0], "mutate"):
            genome = self._make_genome()
            genome.temperature = 1.5  # start at upper boundary
            for _ in range(200):
                genome.mutate(config)
                assert 0.1 <= genome.temperature <= 1.5

    def test_mutation_sometimes_changes_values(self):
        config = MagicMock()
        changed = False
        with patch.object(RAGGenome.__bases__[0], "mutate"):
            random.seed(0)
            for _ in range(100):
                genome = self._make_genome()
                orig = (genome.chunk_size, genome.top_k, genome.temperature)
                genome.mutate(config)
                if (genome.chunk_size, genome.top_k, genome.temperature) != orig:
                    changed = True
                    break
        assert changed, "Mutation never changed any parameter in 100 attempts"


class TestGenomeToPipeline:
    def _call_with_mock(self, genome):
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        # Create a fake module to replace examples.basic_rag_working
        import types
        fake_module = types.ModuleType("examples.basic_rag_working")
        fake_module.BasicRAGPipeline = mock_cls

        import sys
        with patch.dict(sys.modules, {"examples.basic_rag_working": fake_module}):
            result = genome_to_pipeline(genome)
        return mock_cls, mock_instance, result

    def test_pipeline_receives_chunk_size(self):
        genome = RAGGenome(key=1)
        genome.chunk_size = 256
        genome.top_k = 8
        genome.temperature = 0.5

        mock_cls, _, _ = self._call_with_mock(genome)
        mock_cls.assert_called_once_with(chunk_size=256)

    def test_pipeline_receives_top_k(self):
        genome = RAGGenome(key=1)
        genome.top_k = 12

        _, mock_instance, _ = self._call_with_mock(genome)
        assert mock_instance.top_k == 12

    def test_pipeline_receives_temperature(self):
        genome = RAGGenome(key=1)
        genome.temperature = 0.3

        _, mock_instance, _ = self._call_with_mock(genome)
        assert mock_instance.temperature == 0.3
