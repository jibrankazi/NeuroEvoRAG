"""Isolated software-contract tests; toy snippets are NOT empirical SQuAD data."""
import pytest

from benchmarks.real_squad_qa import (
    PretrainedExtractiveReader, TfidfOriginalCorpus, answer_f1,
    calculate_scores, construct_closed_corpus, exact_match, normalize_answer,
    parse_published_squad, train_only_retriever_selection,
)


def test_official_version_contract_and_source_answer_substring():
    context = "The first astronomical record is preserved in the original document. " * 2
    fixture = {"version": "1.1", "data": [{"title": "Synthetic test-only",
        "paragraphs": [{"context": context,
                        "qas": [{"id": "t1", "question": "What was recorded?",
                                 "answers": [{"text": "astronomical record"}]}]}]}]}
    rows = parse_published_squad(fixture)
    assert len(rows) == 1
    assert rows[0]["qa"][0]["gold_answers"] == ["astronomical record"]
    with pytest.raises(ValueError):
        parse_published_squad({"version": "2.0", "data": fixture["data"]})


def test_corpus_and_training_separation_from_heldout():
    paragraphs = [
        {"title": f"fixture {i}", "context": f"Paragraph {i}: " + "different topic " * 30,
         "qa": [{"question": f"What paragraph is {i}?", "gold_answers": [str(i)],
                 "id": f"id-{i}"}]}
        for i in range(150)
    ]
    corpus, train, heldout = construct_closed_corpus(
        paragraphs, corpus_size=120, train_questions=12, heldout_questions=11)
    assert len(corpus) == 120 and len(train) == 12 and len(heldout) == 11
    assert not ({x["gold_document_id"] for x in train} &
                {x["gold_document_id"] for x in heldout})
    assert len({x["question_id"] for x in train + heldout}) == 23


def test_tfidf_retrieves_actually_matching_source():
    corpus = [
        {"title": "alpha", "context": "Venus has dense clouds and a very hot surface."},
        {"title": "beta", "context": "The maple leaf is a symbol on Canada's flag."},
        {"title": "gamma", "context": "Freshwater salmon travel upstream to spawn."},
    ]
    retriever = TfidfOriginalCorpus(corpus)
    hits = retriever.search("Which country uses the maple leaf as its flag symbol?", 2)
    assert hits[0]["document_id"] == 1
    assert hits[0]["source_title"] == "beta"
    assert hits[0]["cosine_tfidf"] > 0
    with pytest.raises(ValueError):
        retriever.search("anything", top_k=0)


def test_train_only_retrieval_selection():
    corpus = [
        {"title": "flag", "context": "Canada flag has a maple leaf."},
        {"title": "planet", "context": "Mars is a rocky planet."},
        {"title": "river", "context": "The Nile is a river."},
    ]
    questions = [
        {"question": "Which country flag has maple?", "gold_document_id": 0},
        {"question": "Which planet is rocky?", "gold_document_id": 1},
    ]
    selected, training_scores, params = train_only_retriever_selection(corpus, questions)
    assert params in ([1, 1], [1, 2])
    assert 0 <= training_scores[str((1, 1))] <= 1
    assert selected.search("Nile river", 1)[0]["document_id"] == 2


def test_actual_span_verification_without_download():
    reader = PretrainedExtractiveReader.__new__(PretrainedExtractiveReader)
    reader.model = lambda **kw: {"start": 4, "end": 10,
                                 "answer": "Canada", "score": 0.91}
    hits = [{"document_id": 2, "source_title": "Test", "context": "The Canada flag."}]
    assert reader.answer("Which country?", hits)["answer"] == "Canada"
    assert reader.answer("Which country?", hits)["evidence_supported"]
    reader.model = lambda **kw: {"start": 4, "end": 10,
                                 "answer": "France", "score": 0.91}
    with pytest.raises(ValueError, match="not a precise substring"):
        reader.answer("Which country?", hits)


def test_squad_official_metrics():
    assert normalize_answer("The, Quick Fox!") == "quick fox"
    assert exact_match("a Blackbird", ["blackbird"]) == 1
    assert answer_f1("red blue", ["red blue green"]) == pytest.approx(0.8)
    assert answer_f1("guessed", ["real answer"]) == 0


def test_heldout_metrics_not_invented():
    sample = [{
        "gold_paragraph_retrieved_at_3": True, "answer_exact_match": 1,
        "answer_token_f1": 1.0, "answer_evidence_supported": True
    }, {
        "gold_paragraph_retrieved_at_3": False, "answer_exact_match": 0,
        "answer_token_f1": 0.0, "answer_evidence_supported": True
    }]
    scores = calculate_scores(sample)
    assert scores["gold_paragraph_recall_at_3"] == 0.5
    assert scores["retrieved_qa_exact_match"] == 0.5
    assert scores["source_substring_verified_rate"] == 1.0
    with pytest.raises(ValueError):
        calculate_scores([])
