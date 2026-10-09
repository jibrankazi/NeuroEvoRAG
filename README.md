# NeuroEvoRAG — Executed Real-Data Neural Retrieval & Question Answering

NeuroEvoRAG now has an executable **no-API-key, local** document retrieval → pretrained neural answer extraction → original-source evidence → heldout QA evaluation path. It processes authentic published [Stanford SQuAD v1.1 validation questions and Wikipedia passages](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json); it does **not** generate demonstration answers or silently substitute fabricated documents.

**Working neural source-to-evaluation pipeline:** [October 9, 2026 successful GitHub Actions run](https://github.com/jibrankazi/NeuroEvoRAG/actions/runs/37997618487). The workflow installed the reader, fetched the original public dataset, passed unit tests, scored genuine heldout questions with a pretrained DistilBERT QA model, validated the original cited evidence, and published downloadable per-question JSON and JSONL outputs.

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
