"""
Tests for rag_pipelines/agentic_generator.py — AgenticGenerator.
"""
from unittest.mock import MagicMock

import pytest

from rag_pipelines.agentic_generator import AgenticGenerator


class TestAgenticGenerator:
    def test_placeholder_response_when_no_llm(self):
        gen = AgenticGenerator(llm=None)
        result = gen.generate("What is AI?", ["context about AI"])
        assert isinstance(result, str)
        assert "placeholder" in result.lower()

    def test_placeholder_response_default_init(self):
        gen = AgenticGenerator()
        result = gen.generate("query", [])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_calls_llm_with_prompt(self, mock_llm):
        gen = AgenticGenerator(llm=mock_llm)
        gen.generate("What is AI?", ["AI is intelligence by machines."])
        mock_llm.assert_called_once()
        prompt = mock_llm.call_args[0][0]
        assert "What is AI?" in prompt
        assert "AI is intelligence by machines." in prompt

    def test_prompt_contains_context_header(self, mock_llm):
        gen = AgenticGenerator(llm=mock_llm)
        gen.generate("query", ["some context"])
        prompt = mock_llm.call_args[0][0]
        assert "Context:" in prompt

    def test_prompt_contains_question_label(self, mock_llm):
        gen = AgenticGenerator(llm=mock_llm)
        gen.generate("my question", ["ctx"])
        prompt = mock_llm.call_args[0][0]
        assert "Question: my question" in prompt

    def test_multiple_contexts_joined(self, mock_llm):
        gen = AgenticGenerator(llm=mock_llm)
        gen.generate("q", ["ctx1", "ctx2", "ctx3"])
        prompt = mock_llm.call_args[0][0]
        assert "ctx1" in prompt
        assert "ctx2" in prompt
        assert "ctx3" in prompt

    def test_empty_context_skips_context_header(self, mock_llm):
        gen = AgenticGenerator(llm=mock_llm)
        gen.generate("q", [])
        prompt = mock_llm.call_args[0][0]
        assert "Context:" not in prompt

    def test_returns_llm_output(self, mock_llm):
        gen = AgenticGenerator(llm=mock_llm)
        result = gen.generate("q", ["ctx"])
        assert result == "Mock LLM answer based on context."
