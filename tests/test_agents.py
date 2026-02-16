"""
Tests for agents/ — RetrieverAgent, CriticAgent, SynthesizerAgent.
"""
from unittest.mock import MagicMock

import pytest

from agents.RetrieverAgent import RetrieverAgent
from agents.CriticAgent import CriticAgent
from agents.SynthesizerAgent import SynthesizerAgent


class TestRetrieverAgent:
    def test_act_delegates_to_retriever(self, mock_retriever):
        agent = RetrieverAgent(mock_retriever)
        result = agent.act("test query")
        mock_retriever.retrieve.assert_called_once_with(
            "test query", modality="text", top_k=5
        )
        assert result == ["chunk_a", "chunk_b", "chunk_c"]

    def test_act_passes_modality(self, mock_retriever):
        agent = RetrieverAgent(mock_retriever)
        agent.act("query", modality="image")
        mock_retriever.retrieve.assert_called_once_with(
            "query", modality="image", top_k=5
        )

    def test_act_passes_top_k(self, mock_retriever):
        agent = RetrieverAgent(mock_retriever)
        agent.act("query", top_k=10)
        mock_retriever.retrieve.assert_called_once_with(
            "query", modality="text", top_k=10
        )

    def test_act_passes_all_params(self, mock_retriever):
        agent = RetrieverAgent(mock_retriever)
        agent.act("q", modality="audio", top_k=3)
        mock_retriever.retrieve.assert_called_once_with(
            "q", modality="audio", top_k=3
        )


class TestCriticAgent:
    def test_returns_zero_without_evaluator(self):
        agent = CriticAgent()
        score = agent.critique("query", ["ctx"], "answer")
        assert score == 0.0

    def test_returns_zero_with_none_evaluator(self):
        agent = CriticAgent(evaluator=None)
        score = agent.critique("query", ["ctx"], "answer")
        assert score == 0.0

    def test_delegates_to_evaluator(self, mock_evaluator):
        agent = CriticAgent(evaluator=mock_evaluator)
        score = agent.critique("q", ["c1", "c2"], "ans")
        mock_evaluator.assert_called_once_with("q", ["c1", "c2"], "ans")
        assert score == 0.85

    def test_evaluator_receives_correct_types(self):
        evaluator = MagicMock(return_value=0.5)
        agent = CriticAgent(evaluator=evaluator)
        agent.critique("question", [], "answer text")
        args = evaluator.call_args[0]
        assert isinstance(args[0], str)
        assert isinstance(args[1], list)
        assert isinstance(args[2], str)


class TestSynthesizerAgent:
    def test_synthesize_empty_list(self):
        agent = SynthesizerAgent()
        assert agent.synthesize([]) == ""

    def test_synthesize_single_item(self):
        agent = SynthesizerAgent()
        assert agent.synthesize(["only one"]) == "only one"

    def test_synthesize_multiple_items(self):
        agent = SynthesizerAgent()
        result = agent.synthesize(["first", "second", "third"])
        assert result == "first\nsecond\nthird"

    def test_synthesize_preserves_content(self):
        agent = SynthesizerAgent()
        contexts = ["alpha", "beta", "gamma"]
        result = agent.synthesize(contexts)
        for ctx in contexts:
            assert ctx in result
