"""Synthetic *software fixtures only*; source-backed metrics run in GitHub CI."""
import json

import pytest

from benchmarks import robust_squad_multiseed as robust


def fixtures(n=145):
    return [{
        "title": f"Sample topic {i}",
        "context": (
            f"Topic {i}: The provincial research archive lists special keyword "
            f"artifact{i} and associated evidence from the local records. "
            + "The full paragraph contains original sample fixture text. " * 4
        ),
        "qa": [{
            "id": f"fixture-question-{i}",
            "question": f"Which keyword appears on topic {i}?",
            "gold_answers": [f"artifact{i}"],
        }],
    } for i in range(n)]


def test_bm25_returns_expected_original_context_and_enforces_positive_k():
    corpus = [
        {"title": "red", "context": "The maple leaf and Canada's flag."},
        {"title": "blue", "context": "Mars is a planet with rocks."},
        {"title": "green", "context": "An arctic salmon migrates upstream."},
    ]
    bm25 = robust.BM25Corpus(corpus)
    assert bm25.search("salmon migrating upstream")[0]["document_id"] == 2
    assert bm25.search("maple flag")[0]["document_id"] == 0
    assert len(bm25.search("anything", 2)) == 2
    with pytest.raises(ValueError):
        bm25.search("something", 0)


class FixtureOnlyReader:
    def answer(self, question, hits):
        hit = hits[0]
        return {
            "answer": "Topic",
            "evidence_text": "Topic",
            "evidence_supported": True,
            "document_id": hit["document_id"],
        }


def test_repeated_seed_original_data_pipeline_contract_without_network(
    monkeypatch, tmp_path,
):
    monkeypatch.setattr(robust, "download_actual_dataset", lambda: (fixtures(), "b" * 64))
    monkeypatch.setattr(robust, "PretrainedExtractiveReader", FixtureOnlyReader)
    dest = tmp_path / "robust"
    report = robust.run(dest, seeds=(7, 9), corpus_size=115,
                        train_count=15, heldout_count=14,
                        reader_count=3)
    assert report["total_retrieval_questions_scored"] == 28
    assert report["total_neural_questions_scored"] == 6
    assert report["source_documents_from_real_published_articles"] is True
    assert report["official_source_sha256"] == "b" * 64
    assert len(report["per_seed_runs"]) == 2
    assert report["train_test_gold_paragraphs_disjoint_within_seed"]
    assert len((dest / "retrieval_cases.jsonl").read_text().splitlines()) == 28
    assert len((dest / "neural_qa_cases.jsonl").read_text().splitlines()) == 6
    assert report["neural_qa_scores"]["evidence_substring_count"] == 6
    assert json.loads((dest / "benchmark.json").read_text()) == report
    for method in ("unigram", "selected_tfidf", "bm25"):
        scores = report["aggregate_retrieval_scores"][method]
        assert all(0 <= scores[f"recall_at_{k}"] <= 1 for k in (1, 3, 5))
        assert scores["recall_at_1"] <= scores["recall_at_3"] <= scores["recall_at_5"]


def test_no_reader_still_measures_real_retrieval_contract(monkeypatch, tmp_path):
    monkeypatch.setattr(robust, "download_actual_dataset", lambda: (fixtures(), "c" * 64))
    report = robust.run(tmp_path, seeds=(17, 21), corpus_size=115,
                        train_count=11, heldout_count=10,
                        reader_count=0, without_reader=True)
    assert report["total_retrieval_questions_scored"] == 20
    assert report["total_neural_questions_scored"] == 0
    assert report["neural_qa_scores"] is None


@pytest.mark.parametrize("params", [
    {"seeds": (42, 42)}, {"seeds": (42,)},
    {"corpus_size": 80}, {"train_count": 3},
    {"heldout_count": 9}, {"reader_count": 180},
])
def test_bad_experiment_design_refused_before_any_network(params, monkeypatch, tmp_path):
    monkeypatch.setattr(robust, "download_actual_dataset",
                        lambda: (_ for _ in ()).throw(AssertionError("no network")))
    with pytest.raises(ValueError):
        robust.run(tmp_path, without_reader=False, **params)


def test_neural_scoring_counts_only_real_observed_exact_matches():
    data = [
        {"exact_match": 1, "token_f1": 1.0, "source_span_verified": True,
         "gold_paragraph_retrieved_at_3": True},
        {"exact_match": 0, "token_f1": .25, "source_span_verified": True,
         "gold_paragraph_retrieved_at_3": False},
    ]
    summary = robust.span_metrics(data)
    assert summary["exact_match_count"] == 1
    assert summary["exact_match"] == .5
    assert summary["mean_answer_f1"] == .625
    assert summary["gold_paragraph_in_top3_count"] == 1
    with pytest.raises(ValueError):
        robust.span_metrics([])
