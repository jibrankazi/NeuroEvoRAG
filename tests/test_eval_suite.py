"""
Tests for benchmarks/eval_suite.py — RAGAS evaluation wrapper.

Mocks the RAGAS evaluate call to test the data preparation logic.
"""
from unittest.mock import MagicMock, patch

import pytest


class TestEvaluatePipelineReal:
    @patch("benchmarks.eval_suite.evaluate")
    def test_returns_expected_keys(self, mock_evaluate):
        from benchmarks.eval_suite import evaluate_pipeline_real

        mock_evaluate.return_value = {
            "faithfulness": 0.7,
            "answer_relevancy": 0.8,
            "context_precision": 0.6,
        }

        pipeline = MagicMock()
        pipeline.query.return_value = {
            "answer": "test answer",
            "context": ["ctx1"],
        }

        result = evaluate_pipeline_real(
            pipeline, ["q1", "q2"], ["a1", "a2"]
        )

        assert "faithfulness" in result
        assert "answer_relevancy" in result
        assert "context_precision" in result
        assert "num_questions" in result

    @patch("benchmarks.eval_suite.evaluate")
    def test_num_questions_matches_input(self, mock_evaluate):
        from benchmarks.eval_suite import evaluate_pipeline_real

        mock_evaluate.return_value = {
            "faithfulness": 0.5,
            "answer_relevancy": 0.5,
            "context_precision": 0.5,
        }

        pipeline = MagicMock()
        pipeline.query.return_value = {"answer": "a", "context": ["c"]}

        questions = ["q1", "q2", "q3"]
        result = evaluate_pipeline_real(pipeline, questions, ["a1", "a2", "a3"])
        assert result["num_questions"] == 3

    @patch("benchmarks.eval_suite.evaluate")
    def test_calls_pipeline_query_for_each_question(self, mock_evaluate):
        from benchmarks.eval_suite import evaluate_pipeline_real

        mock_evaluate.return_value = {
            "faithfulness": 0.5,
            "answer_relevancy": 0.5,
            "context_precision": 0.5,
        }

        pipeline = MagicMock()
        pipeline.query.return_value = {"answer": "a", "context": ["c"]}

        questions = ["q1", "q2"]
        evaluate_pipeline_real(pipeline, questions, ["a1", "a2"])
        assert pipeline.query.call_count == 2
        pipeline.query.assert_any_call("q1")
        pipeline.query.assert_any_call("q2")

    @patch("benchmarks.eval_suite.Dataset")
    @patch("benchmarks.eval_suite.evaluate")
    def test_dataset_constructed_with_correct_keys(self, mock_evaluate, mock_dataset_cls):
        from benchmarks.eval_suite import evaluate_pipeline_real

        mock_evaluate.return_value = {
            "faithfulness": 0.5,
            "answer_relevancy": 0.5,
            "context_precision": 0.5,
        }

        pipeline = MagicMock()
        pipeline.query.return_value = {"answer": "ans", "context": ["ctx"]}

        evaluate_pipeline_real(pipeline, ["q"], ["gt"])

        # Verify Dataset.from_dict was called
        mock_dataset_cls.from_dict.assert_called_once()
        data_dict = mock_dataset_cls.from_dict.call_args[0][0]
        assert "question" in data_dict
        assert "answer" in data_dict
        assert "contexts" in data_dict
        assert "ground_truth" in data_dict
