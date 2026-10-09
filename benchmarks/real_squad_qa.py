"""Genuine SQuAD -> retrieval -> pretrained extractive QA -> scored evidence.

One reproducible small-scale, no-API-key, neural retrieval-augmented QA run.
This is *not* evolutionary/NEAT optimization, an LLM chat system, or a
production/multihop benchmark. No invented documents or predictions.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import random
import re
import string

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

SQUAD_DEV = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
QA_CHECKPOINT = "distilbert/distilbert-base-cased-distilled-squad"


def parse_published_squad(payload):
    """Extract genuine original Wikipedia paragraphs + crowd-authored QA labels."""
    if payload.get("version") != "1.1" or not payload.get("data"):
        raise ValueError("Expected official SQuAD version 1.1 source structure")
    paragraphs = []
    observed = set()
    for article in payload["data"]:
        title = article.get("title", "")
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")
            if len(context) < 80 or context in observed:
                continue
            question_records = []
            for qa in paragraph.get("qas", []):
                text = qa.get("question", "").strip()
                answers = [a["text"].strip() for a in qa.get("answers", [])
                           if isinstance(a.get("text"), str) and a["text"].strip()]
                if text and answers and any(ans in context for ans in answers):
                    question_records.append({"id": str(qa.get("id", "")),
                                             "question": text,
                                             "gold_answers": list(dict.fromkeys(answers))})
            if question_records:
                observed.add(context)
                paragraphs.append({"title": title, "context": context, "qa": question_records})
    return paragraphs


def download_actual_dataset():
    response = requests.get(SQUAD_DEV, timeout=85)
    response.raise_for_status()
    raw = response.content
    if len(raw) < 2_000_000:
        raise ValueError("Official published SQuAD dev response seems partial or empty")
    paragraphs = parse_published_squad(response.json())
    if len(paragraphs) < 1500:
        raise ValueError(f"Only {len(paragraphs)} distinct real answer-bearing paragraphs")
    return paragraphs, sha256(raw).hexdigest()


def construct_closed_corpus(paragraphs, corpus_size=240, train_questions=20,
                            heldout_questions=16, seed=42):
    """Pick real corpus and disjoint heldout gold *paragraphs* before fitting."""
    if corpus_size < 100 or train_questions < 10 or heldout_questions < 10:
        raise ValueError("Must use >100 real paragraphs and >10 questions per split")
    if train_questions + heldout_questions >= corpus_size:
        raise ValueError("Not enough distractor paragraphs for a meaningful retrieval corpus")
    if len(paragraphs) < corpus_size:
        raise ValueError("Not enough original SQuAD reference paragraphs")
    rng = random.Random(seed)
    corpus = rng.sample(paragraphs, corpus_size)
    gold_indices = rng.sample(range(corpus_size), train_questions + heldout_questions)
    cases = []
    for idx in gold_indices:
        original = rng.choice(corpus[idx]["qa"])
        cases.append({"question_id": original["id"], "question": original["question"],
                      "gold_answers": original["gold_answers"],
                      "gold_document_id": idx, "gold_source_title": corpus[idx]["title"]})
    return corpus, cases[:train_questions], cases[train_questions:]


class TfidfOriginalCorpus:
    def __init__(self, paragraphs, ngram_range=(1, 1)):
        self.paragraphs = paragraphs
        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_features=50000, ngram_range=ngram_range)
        self.matrix = self.vectorizer.fit_transform(
            [paragraph["context"] for paragraph in paragraphs])

    def search(self, question, top_k=3):
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        q = self.vectorizer.transform([question])
        scores = (self.matrix @ q.T).toarray().ravel()
        indexes = np.argsort(-scores, kind="stable")[:min(top_k, len(scores))]
        return [{"document_id": int(i), "source_title": self.paragraphs[int(i)]["title"],
                 "context": self.paragraphs[int(i)]["context"],
                 "cosine_tfidf": float(scores[i])} for i in indexes]


def train_only_retriever_selection(corpus, training_cases):
    """Select on train only, never on the 16 scored heldout questions."""
    candidates = {}
    for params in ((1, 1), (1, 2)):
        candidate = TfidfOriginalCorpus(corpus, ngram_range=params)
        top3_hits = sum(any(item["document_id"] == q["gold_document_id"]
                            for item in candidate.search(q["question"], top_k=3))
                        for q in training_cases)
        candidates[str(params)] = {"train_gold_recall_at_3": top3_hits / len(training_cases),
                                   "trained_candidate": candidate, "ngram_range": params}
    # Prefer simpler unigram representation in a tie.
    selected = max(candidates.values(),
                   key=lambda x: (x["train_gold_recall_at_3"],
                                  -x["ngram_range"][1]))
    return selected["trained_candidate"], {
        str(v["ngram_range"]): v["train_gold_recall_at_3"] for v in candidates.values()
    }, list(selected["ngram_range"])


def normalize_answer(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, truths):
    return int(any(normalize_answer(prediction) == normalize_answer(x) for x in truths))


def answer_f1(prediction, truths):
    a = normalize_answer(prediction).split()
    best = 0.0
    for truth in truths:
        b = normalize_answer(truth).split()
        common = sum((Counter(a) & Counter(b)).values())
        if not a or not b:
            f1 = 1.0 if a == b else 0.0
        elif common == 0:
            f1 = 0.0
        else:
            precision, recall = common / len(a), common / len(b)
            f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


class PretrainedExtractiveReader:
    def __init__(self):
        from transformers import pipeline
        self.model = pipeline("question-answering", model=QA_CHECKPOINT,
                              tokenizer=QA_CHECKPOINT, device=-1)

    def answer(self, question, hits):
        """Return a verifiable passage span, not a fabricated model response."""
        if not hits:
            return {"answer": "", "score": 0.0, "document_id": None,
                    "evidence_text": "", "evidence_supported": False}
        best = None
        for item in hits:
            result = self.model(question=question, context=item["context"],
                                max_answer_len=35)
            start, end = int(result["start"]), int(result["end"])
            if not (0 <= start < end <= len(item["context"])):
                raise ValueError("Pretrained reader returned invalid original-source span")
            span = item["context"][start:end]
            if normalize_answer(span) != normalize_answer(result["answer"]):
                raise ValueError("Reader output is not a precise substring of cited original text")
            candidate = {"answer": span, "score": float(result["score"]),
                         "document_id": item["document_id"],
                         "source_title": item["source_title"],
                         "evidence_text": span, "evidence_supported": True}
            if not np.isfinite(candidate["score"]) or not 0 <= candidate["score"] <= 1:
                raise ValueError("Pretrained reader invalid confidence score")
            if best is None or candidate["score"] > best["score"]:
                best = candidate
        return best


def calculate_scores(records):
    count = len(records)
    if count == 0:
        raise ValueError("Cannot claim measured QA results with no original cases")
    return {
        "evaluated_questions": count,
        "gold_paragraph_retrieved_at_3": sum(x["gold_paragraph_retrieved_at_3"] for x in records),
        "gold_paragraph_recall_at_3": sum(x["gold_paragraph_retrieved_at_3"] for x in records) / count,
        "retrieved_qa_exact_match": sum(x["answer_exact_match"] for x in records) / count,
        "retrieved_qa_mean_token_f1": sum(x["answer_token_f1"] for x in records) / count,
        "source_substring_verified_rate": sum(x["answer_evidence_supported"] for x in records) / count,
    }


def evaluate(output="results/real_squad_qa", corpus_size=240,
             train_questions=20, heldout_questions=16, seed=42):
    paragraphs, source_sha = download_actual_dataset()
    corpus, training_cases, test_cases = construct_closed_corpus(
        paragraphs, corpus_size, train_questions, heldout_questions, seed)
    baseline = TfidfOriginalCorpus(corpus, ngram_range=(1, 1))
    selected, training_scores, chosen_ngrams = train_only_retriever_selection(
        corpus, training_cases)
    reader = PretrainedExtractiveReader()  # an actual downloaded pretrained neural model
    records = []
    for i, q in enumerate(test_cases):
        hits = selected.search(q["question"], top_k=3)
        baseline_hits = baseline.search(q["question"], top_k=3)
        generated = reader.answer(q["question"], hits)
        record = {
            "source_question_id": q["question_id"],
            "source_article": q["gold_source_title"],
            "question": q["question"],
            "original_gold_answers": q["gold_answers"],
            "gold_original_paragraph_index": q["gold_document_id"],
            "retrieved_paragraph_indices": [h["document_id"] for h in hits],
            "baseline_gold_paragraph_retrieved_at_3":
                q["gold_document_id"] in [x["document_id"] for x in baseline_hits],
            "gold_paragraph_retrieved_at_3":
                q["gold_document_id"] in [h["document_id"] for h in hits],
            "model_answer": generated["answer"],
            "cited_evidence_original_paragraph_index": generated["document_id"],
            "cited_article": generated.get("source_title"),
            "cited_original_source_span": generated["evidence_text"],
            "model_span_confidence": generated["score"],
            "answer_evidence_supported": generated["evidence_supported"],
            "answer_exact_match": exact_match(generated["answer"], q["gold_answers"]),
            "answer_token_f1": answer_f1(generated["answer"], q["gold_answers"]),
        }
        records.append(record)
        print(f"Evaluated observed SQuAD question {i+1}/{len(test_cases)}; "
              f"retrieved gold={record['gold_paragraph_retrieved_at_3']}; "
              f"exact={record['answer_exact_match']}", flush=True)
    scores = calculate_scores(records)
    scores["baseline_unigram_gold_paragraph_recall_at_3"] = sum(
        x["baseline_gold_paragraph_retrieved_at_3"] for x in records) / len(records)
    results = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_source_url": SQUAD_DEV,
        "official_source_sha256_raw_response": source_sha,
        "published_squad_version": "1.1",
        "pretrained_reader": QA_CHECKPOINT,
        "experiment_type": "real pretrained extractive neural QA after TF-IDF retrieval",
        "genuine_distinct_corpus_paragraphs": len(corpus),
        "selection_training_question_count": len(training_cases),
        "heldout_question_count": len(test_cases),
        "original_reference_passages_disjoint_across_train_and_heldout": True,
        "corpus_includes_heldout_original_paragraphs": True,
        "heldout_not_used_for_retriever_selection": True,
        "retriever_config_training_scores": training_scores,
        "selected_ngram_range": chosen_ngrams,
        "retrieved_qa_holdout": scores,
        "question_results": records,
        "limitations": (
            "Small one-seed closed-world sample of authentic SQuAD v1.1 dev QA; "
            "evaluation gold passages deliberately indexed among many real distractors. "
            "Training used only different paragraphs for retriever selection. "
            "Reader is pretrained on SQuAD-related training material, not retrained here. "
            "Evidence-supported substring does not establish semantic truth. "
            "Not independent general-domain assessment, multi-hop reasoning, "
            "evolutionary/NEAT training, production RAG, or statistical confidence intervals."
        ),
    }
    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "heldout_results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (dest / "original_heldout_qa_evidence.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "question_results"},
                     indent=2, allow_nan=False), flush=True)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="results/real_squad_qa")
    p.add_argument("--corpus-size", type=int, default=240)
    p.add_argument("--train-questions", type=int, default=20)
    p.add_argument("--heldout-questions", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    evaluate(a.output, a.corpus_size, a.train_questions, a.heldout_questions, a.seed)
