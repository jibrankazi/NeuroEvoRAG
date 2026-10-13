# NeuroEvoRAG — Executed Real-Data Neural Retrieval & Question Answering

NeuroEvoRAG now has an executable **no-API-key, local** document retrieval → pretrained neural answer extraction → original-source evidence → heldout QA evaluation path. It processes authentic published [Stanford SQuAD v1.1 validation questions and Wikipedia passages](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json); it does **not** generate demonstration answers or silently substitute fabricated documents.

**Working neural source-to-evaluation pipeline:** [October 9, 2026 successful GitHub Actions run](https://github.com/jibrankazi/NeuroEvoRAG/actions/runs/37997618487). The workflow installed the reader, fetched the original public dataset, passed unit tests, scored genuine heldout questions with a pretrained DistilBERT QA model, validated the original cited evidence, and published downloadable per-question JSON and JSONL outputs.

## October 13, 2026: Larger independent-source repeated-seed stress test

**[Completed, fully passing original SQuAD and pretrained neural evaluation](https://github.com/jibrankazi/NeuroEvoRAG/actions/runs/38344505317)**

The new [`benchmarks/robust_squad_multiseed.py`](benchmarks/robust_squad_multiseed.py)
fetches the publisher's complete actual SQuAD v1.1 validation file, verifies its SHA-256,
and builds a **1,200 genuine Wikipedia-paragraph index per seed**, including the test
questions' gold source passages. It chooses TF-IDF unigram vs bigram using
only **60 different training-question paragraphs** per seed. It then scores
**120 held-out questions** per seed, three prespecified seeds 17, 42 and 91,
on unigram TF-IDF, selected TF-IDF and independently implemented Okapi BM25.
The trained DistilBERT extractive reader is run on the first **24 prespecified
heldout questions per seed**, for **72 actual neural answers** and original
publisher-answer comparisons, with every cited answer checked against the
original retrieved text span.

| Actual independent publisher questions | TF-IDF unigram | CV-selected TF-IDF | BM25 |
| --- | ---: | ---: | ---: |
| Original gold paragraph in top 1 of 1,200 indexed paragraphs | **62.50%** | 62.78% | **81.94%** |
| In top 3 | 82.50% | 81.94% | **91.67%** |
| In top 5 | 86.67% | 85.56% | **94.44%** |

These are **retrieval outcomes**, not answer accuracy. The prior TF-IDF
selection did **not** improve test retrieval against the simpler unigram;
BM25 performed better in this observed experimental condition. That outcome
has not been used to pretend an earlier heldout test was untouched for future
model selection.

The *pretrained neural reader attached to CV-selected TF-IDF* produced:
**34/72 exact original answers (47.22%)**, **mean answer-token F1 0.51245**,
**59/72 test gold paragraphs in its retrieved top 3**, and **72/72 verifiable
answer spans within an actual retrieved source paragraph**. The older
single-seed 240-paragraph/16-question run showed 11/16 exact answers
(68.75%) but was much smaller and was under a different corpus and seed
protocol. The two results are **not interchangeable nor a clean model
improvement/regression test**.

**Boundaries:** The correct paragraph was **deliberately indexed** for each
test question, so this is a closed-corpus retriever benchmark. The reader
is an existing fine-tuned SQuAD QA model, **not a newly trained neural
reader**, and there is no multi-hop/HotpotQA, large language generation,
trained NEAT or live knowledge base. Within each seed, train/test gold
paragraphs do not overlap; across seeds, source passages can repeat. The
larger experiment is a **more credible prototype diagnostic, not proof of
production RAG accuracy**.

Reproduce in a Python 3.11 environment:

```bash
python -m pip install requests numpy scikit-learn transformers==4.44.2 torch pytest
python -m pytest -q tests/test_robust_squad_multiseed.py tests/test_real_squad_qa.py
python -m benchmarks.robust_squad_multiseed --seeds 17 42 91 --corpus-size 1200 --train-questions 60 --heldout-questions 120 --reader-count 24
```

GitHub Actions also independently re-evaluates **all 360 original retrieval
hits** in the emitted per-question JSONL rather than trusting only printed
aggregate percentages. Download the three evidence files from the completed
run's `authentic-squad-3-seed-1200doc-360retrieval-72neuralqa` artifact.

---

## Actually measured results (one small historical experiment)

| Genuine original-data measurement | Executed result |
| --- | ---: |
| Distinct original SQuAD paragraphs indexed, including real distractors | **240** |
| Real questions used for retriever n-gram selection | **20** |
| Separate real questions, from different gold paragraphs, evaluated after selection | **16** |
| Correct original paragraph included among top 3 for heldout questions | **16/16 (100%)** |
| Exact answer match against crowd-annotated heldout reference | **11/16 (68.75%)** |
| Mean SQuAD-style answer token F1 on heldout questions | **0.7092** |
| Model answers traceable to a verbatim original retrieved paragraph substring | **16/16 (100%)** |

The sample deliberately indexes each test question's gold paragraph among 240 candidate paragraphs; **this is a small, unusually favorable closed-corpus benchmark, not a claim of 100% retrieval in a real knowledge base.** The neural reader is pretrained on SQuAD-related material. Tests use different *evaluation passages* to choose n-grams, but this is **not an externally independent SQuAD generalization study**. Verbatim source support is not proof of factual correctness; the EM/F1 results measure answer quality separately.

The [GitHub Actions run](https://github.com/jibrankazi/NeuroEvoRAG/actions/runs/37997618487) contains an artifact named `real-squad-240doc-pretrained-qa-heldout-evidence`. It includes the original SQuAD response SHA-256, model checkpoint, recorded configurations, all 16 true questions/gold answers and observed neural predictions, retrieval IDs, evidence spans, and aggregate metrics.

## Run it yourself — no external API key required

Python 3.11+ is recommended. Model weights are fetched from Hugging Face the first time you run, and **the model runs on your CPU**.

```bash
git clone https://github.com/jibrankazi/NeuroEvoRAG.git
cd NeuroEvoRAG
python -m pip install requests numpy scikit-learn transformers==4.44.2 torch pytest
python -m pytest -q tests/test_real_squad_qa.py tests/test_basic_rag_local.py
python -m benchmarks.real_squad_qa --corpus-size 240 --train-questions 20 --heldout-questions 16 --seed 42
```

Alternatively, run the actual local application demo:

```bash
python -m examples.basic_rag_working
```

The replacement `BasicRAGPipeline` now uses actual local TF-IDF indexing and pretrained neural answer extraction. Its `query()` includes the verbatim supporting span and original indexed source ID. **It no longer needs `OPENAI_API_KEY` or a paid model account.** Missing original data, unsupported answers, or missing model access fail rather than fabricating answers.

The benchmark writes:
- `results/real_squad_qa/heldout_results.json` — actual publisher URL, raw-response fingerprint, true question and answer evaluation results and limitations.
- `results/real_squad_qa/original_heldout_qa_evidence.jsonl` — evidence records for each of the 16 original heldout QA examples.

## Also runnable: genuine-source retrieval parameter experiment

```bash
python -m benchmarks.offline_retrieval --max-questions 80 --generations 3 --population-size 5 --output results/squad_retrieval.json
```

This separate seeded experiment searches retrieval chunk length and top-k for real published SQuAD questions; it measures **retrieval recall only**, not model answers or full neuroevolution. CI for that experiment is in `.github/workflows/evolve_smoke.yml`. It must not be confused with full end-to-end answer generation.

## What remains unsupported

The repo's original research proposal described multi-hop HotpotQA, NEAT, RAGAS faithfulness, multimodal image/audio models, and performance superiority of genetic search over Optuna/random search. **Those claimed comparisons and the original numerical HotpotQA table have not been verified as completed experiments.** The older skeleton modules may remain as designs, but are not evidence of measured results.

Verified now: original document source → retriever → pretrained neural *extractive* QA → explicit source span → SQuAD token-level scoring → GitHub CI artifact.

Still to build and evaluate: true generative LLM RAG with learned retriever training, research-grade multi-hop datasets, NEAT/Optuna equal-budget comparisons, repeated-seed confidence intervals, reliable document provenance in larger unseen corpora, and deployment monitoring. No paper or production-readiness claim is implied.

This repository is an academic/portfolio prototype, not a validated production AI product.
