"""Repeated-seed original SQuAD retrieval stress-test and neural QA audit.

Downloads genuine source once, uses disjoint training/test gold paragraphs
within each run, indexes 1200+ authentic paragraphs (including reference
passages), compares unigram, training-selected TF-IDF, and Okapi BM25.
Optional independently-scored pretrained reader evaluates a predeclared
prefix of heldout queries without selecting on test answers.

IMPORTANT: QA reader is pretrained on SQuAD-style data, not trained here;
gold references are deliberately present in the indexed corpus; repeated
seeds can share source passages. These are not out-of-domain/general RAG
claims, and scores are NOT verified until CI executes source data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from math import log1p
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from benchmarks.real_squad_qa import (
    QA_CHECKPOINT, SQUAD_DEV, PretrainedExtractiveReader,
    TfidfOriginalCorpus, answer_f1, construct_closed_corpus,
    download_actual_dataset, exact_match, train_only_retriever_selection,
)


class BM25Corpus:
    """True lexical Okapi BM25 on arbitrary original indexed paragraphs."""
    def __init__(self, paragraphs, *, k1=1.2, b=0.75):
        if not paragraphs or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires nonempty documents and valid parameters")
        self.paragraphs = paragraphs
        self.vectorizer = CountVectorizer(stop_words="english", max_features=50000)
        counts = self.vectorizer.fit_transform(
            [x["context"] for x in paragraphs]).tocsr()
        n = counts.shape[0]
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        avg_length = float(lengths.mean())
        if avg_length <= 0:
            raise ValueError("Empty corpus vocabulary")
        df = np.asarray((counts > 0).sum(axis=0)).ravel()
        idf = np.log1p((n - df + .5) / (df + .5))
        weighted = counts.astype(float).copy()
        doc_ids = np.repeat(np.arange(n), np.diff(weighted.indptr))
        weighted.data *= (k1 + 1) / (
            weighted.data + k1 * (1 - b + b * lengths[doc_ids] / avg_length))
        weighted.data *= idf[weighted.indices]
        self.matrix = weighted

    def search(self, question, top_k=3):
        if top_k < 1:
            raise ValueError("top_k must be >=1")
        query = self.vectorizer.transform([question])
        scores = np.asarray(self.matrix @ query.T).reshape(-1)
        indices = np.argsort(-scores, kind="stable")[:min(top_k, len(scores))]
        return [
            {"document_id": int(i),
             "source_title": self.paragraphs[int(i)]["title"],
             "context": self.paragraphs[int(i)]["context"],
             "bm25": float(scores[i])} for i in indices
        ]


def span_metrics(records):
    if not records:
        raise ValueError("No independently evaluated QA records")
    return {
        "questions": len(records),
        "exact_match_count": sum(int(x["exact_match"]) for x in records),
        "exact_match": sum(int(x["exact_match"]) for x in records) / len(records),
        "mean_answer_f1": sum(x["token_f1"] for x in records) / len(records),
        "evidence_substring_count": sum(int(x["source_span_verified"]) for x in records),
        "gold_paragraph_in_top3_count": sum(
            int(x["gold_paragraph_retrieved_at_3"]) for x in records),
    }


def run(
    output="results/squad_multiseed",
    *,
    seeds=(17, 42, 91),
    corpus_size=1200,
    train_count=60,
    heldout_count=120,
    reader_count=24,
    without_reader=False,
):
    seeds = tuple(seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Need at least two distinct prespecified seeds")
    if not (100 <= corpus_size and
            10 <= train_count and 10 <= heldout_count and
            train_count + heldout_count < corpus_size):
        raise ValueError("Invalid corpus/training/evaluation sizes")
    if not 0 <= reader_count <= heldout_count:
        raise ValueError("reader_count outside heldout set")
    if not without_reader and reader_count < 1:
        raise ValueError("Neural QA needs at least one heldout question")
    paragraphs, source_sha = download_actual_dataset()
    reader = None if without_reader else PretrainedExtractiveReader()
    experiments = []
    receipts = []
    qa_records = []
    for seed in seeds:
        corpus, training, heldout = construct_closed_corpus(
            paragraphs, corpus_size, train_count, heldout_count, seed)
        baseline = TfidfOriginalCorpus(corpus, ngram_range=(1, 1))
        selected, training_cv, selected_ngrams = train_only_retriever_selection(
            corpus, training)
        bm25 = BM25Corpus(corpus)
        counts = {name: Counter() for name in ("unigram", "selected_tfidf", "bm25")}
        for index, case in enumerate(heldout):
            docids = {}
            for name, retriever in [
                ("unigram", baseline), ("selected_tfidf", selected), ("bm25", bm25)
            ]:
                hits = retriever.search(case["question"], 5)
                ids = [item["document_id"] for item in hits]
                docids[name] = ids
                for k in (1, 3, 5):
                    counts[name][f"recall_at_{k}"] += int(case["gold_document_id"] in ids[:k])
                counts[name]["reciprocal_rank_sum"] += (
                    1.0 / (ids.index(case["gold_document_id"]) + 1)
                    if case["gold_document_id"] in ids else 0.0)
            item = {
                "seed": seed, "original_question_id": case["question_id"],
                "question": case["question"],
                "source_title": case["gold_source_title"],
                "gold_paragraph_index": case["gold_document_id"],
                "genuine_source_gold_answers": case["gold_answers"],
                "retrieved_indices_by_method": docids,
                "source_gold_paragraph_in_index_by_design": True,
            }
            receipts.append(item)
            if reader is not None and index < reader_count:
                # Predeclare the first N test questions for QA before observing
                # any correct/incorrect outputs. No answer-based cherry picking.
                hits = selected.search(case["question"], 3)
                answer = reader.answer(case["question"], hits)
                valid_span = bool(answer["evidence_supported"]
                                  and answer["evidence_text"] == answer["answer"]
                                  and answer["document_id"] in docids["selected_tfidf"][:3])
                if not valid_span:
                    raise ValueError("Neural answer is not verbatim from retrieved source")
                qa_records.append({
                    "seed": seed, "original_question_id": case["question_id"],
                    "question": case["question"],
                    "published_answers": case["gold_answers"],
                    "predicted_original_text_span": answer["answer"],
                    "cited_paragraph_index": answer["document_id"],
                    "gold_original_paragraph_index": case["gold_document_id"],
                    "gold_paragraph_retrieved_at_3":
                        case["gold_document_id"] in docids["selected_tfidf"][:3],
                    "exact_match": exact_match(answer["answer"], case["gold_answers"]),
                    "token_f1": answer_f1(answer["answer"], case["gold_answers"]),
                    "source_span_verified": valid_span,
                })
        experiments.append({
            "seed": seed,
            "corpus_size": len(corpus),
            "training_questions": len(training),
            "heldout_questions": len(heldout),
            "distinct_training_gold_paragraphs": len({
                x["gold_document_id"] for x in training}),
            "distinct_heldout_gold_paragraphs": len({
                x["gold_document_id"] for x in heldout}),
            "train_test_gold_paragraph_overlap": len(
                {x["gold_document_id"] for x in training} &
                {x["gold_document_id"] for x in heldout}),
            "selected_tfidf_ngrams": selected_ngrams,
            "train_only_retrieval_selection_scores": training_cv,
            "retrieval": {
                name: {
                    **{f"recall_at_{k}": counts[name][f"recall_at_{k}"] / len(heldout)
                       for k in (1, 3, 5)},
                    **{f"correct_at_{k}": counts[name][f"recall_at_{k}"]
                       for k in (1, 3, 5)},
                    "truncated_mrr_at_5": counts[name]["reciprocal_rank_sum"] / len(heldout),
                } for name in counts
            },
        })
        print(json.dumps({
            "seed": seed,
            "retrieval_at_3": {name: exp["recall_at_3"]
                               for name, exp in experiments[-1]["retrieval"].items()},
            "neural_questions_scored_to_date": len(qa_records),
        }), flush=True)
    n = len(experiments) * heldout_count
    aggregates = {
        method: {
            f"recall_at_{k}": sum(
                r["retrieval"][method][f"correct_at_{k}"] for r in experiments) / n
            for k in (1, 3, 5)
        } for method in ("unigram", "selected_tfidf", "bm25")
    }
    mean_by_seed = {
        method: {
            "mean_recall_at_3_over_seeds": float(np.mean(
                [r["retrieval"][method]["recall_at_3"] for r in experiments])),
            "minimum_seed_recall_at_3": float(min(
                r["retrieval"][method]["recall_at_3"] for r in experiments)),
            "maximum_seed_recall_at_3": float(max(
                r["retrieval"][method]["recall_at_3"] for r in experiments)),
        } for method in aggregates
    }
    report = {
        "evaluation": "Original Stanford SQuAD v1.1 source; document-indexed retrieval, optionally pretrained neural extractive QA",
        "official_url": SQUAD_DEV,
        "official_source_sha256": source_sha,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_published_paragraphs_available": len(paragraphs),
        "seed_protocol_fixed_before_evaluation": list(seeds),
        "indexed_genuine_paragraphs_per_seed": corpus_size,
        "training_question_paragraphs_per_seed": train_count,
        "heldout_question_paragraphs_per_seed": heldout_count,
        "neural_question_prefix_per_seed": 0 if without_reader else reader_count,
        "total_retrieval_questions_scored": n,
        "total_neural_questions_scored": len(qa_records),
        "exact_neural_checkpoint": None if without_reader else QA_CHECKPOINT,
        "per_seed_runs": experiments,
        "aggregate_retrieval_scores": aggregates,
        "seed_variation": mean_by_seed,
        "neural_qa_scores": span_metrics(qa_records) if qa_records else None,
        "train_test_gold_paragraphs_disjoint_within_seed": all(
            x["train_test_gold_paragraph_overlap"] == 0 for x in experiments),
        "original_gold_passages_present_in_retriever_corpus_by_design": True,
        "retrieval_and_pretrained_reader_train_independently": False,
        "original_SQuAD_source_question_labels_used_for_scoring_only": True,
        "source_documents_from_real_published_articles": True,
        "new_neural_model_training_performed": False,
        "NEAT_or_evolutionary_learning_performed": False,
        "production_generalization_established": False,
        "limitations": (
            "The reader is a pre-existing fine-tuned SQuAD model and may share "
            "domain/training content with the SQuAD validation set. "
            "Each real QA gold document is deliberately included among genuine "
            "distractor paragraphs; no external unknown-answer corpus test. "
            "Within each seed model selection uses training questions only, "
            "but seeds can reuse paragraphs and are not independent populations. "
            "No confidence intervals or real product deployment claims."
        ),
    }
    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    with (dest / "retrieval_cases.jsonl").open("w") as f:
        for row in receipts:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (dest / "neural_qa_cases.jsonl").open("w") as f:
        for row in qa_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("per_seed_runs",)}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/squad_multiseed")
    parser.add_argument("--corpus-size", type=int, default=1200)
    parser.add_argument("--train-questions", type=int, default=60)
    parser.add_argument("--heldout-questions", type=int, default=120)
    parser.add_argument("--reader-count", type=int, default=24)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 91])
    parser.add_argument("--without-reader", action="store_true")
    args = parser.parse_args()
    run(args.output, seeds=args.seeds, corpus_size=args.corpus_size,
        train_count=args.train_questions, heldout_count=args.heldout_questions,
        reader_count=args.reader_count, without_reader=args.without_reader)


if __name__ == "__main__":
    main()
