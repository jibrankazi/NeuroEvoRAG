# NeuroEvoRAG — verified implementation status (October 9, 2026)

## Tested original-data functionality

- **Executed published SQuAD source ingestion**: downloads Stanford SQuAD dev 1.1 JSON, validates the genuine original Wikipedia paragraphs, original crowdsourced questions and source response checksum. No generated data is substituted.
- **Executed original document retrieval**: indexes 240 distinct actual Wikipedia paragraphs with TF-IDF (including many real distractors), selects unigram/bigram settings using **20 original questions** whose gold paragraphs do not overlap the heldout questions' gold paragraphs. Baseline and chosen configurations are measured.
- **Executed pretrained neural answer extraction**: `distilbert/distilbert-base-cased-distilled-squad` produces answers from actual top-three retrieved paragraphs on CPU, without an OpenAI API key.
- **Verified original-source citations**: each neural answer is a verbatim substring of a cited retrieved original paragraph, with original question ID, source title and document index preserved.
- **Measured real labelled QA performance**: **16 heldout authentic SQuAD questions; 16/16 original paragraphs retrieved top 3; 11/16 exact answer match (68.75%); mean token F1 0.7092; 16/16 verbatim cited spans**. This is a favorable small closed-corpus experiment, not a statistical estimate of full SQuAD/general-domain performance.
- **End-to-end GitHub workflow and artifact**: [successful observed-data run](https://github.com/jibrankazi/NeuroEvoRAG/actions/runs/37997618487), `real-squad-240doc-pretrained-qa-heldout-evidence`.
- **Local no-token demo**: `python -m examples.basic_rag_working` now uses local document indexing and pretrained extractive reader, not the earlier `OPENAI_API_KEY`-dependent demonstration.
- **Separate lightweight genuine-source retrieval-only search**: `benchmarks/offline_retrieval.py` works with SQuAD and simple seeded parameter search. It does not establish full NEAT evolution.

## Still unverified or not implemented

- **Real NEAT neuroevolution of a trained RAG system**, neural retriever training, and comparable multi-seed Optuna/grid/random benchmarks. Placeholder evolution classes are not results.
- **Multi-hop HotpotQA question-answering benchmarks**, RAGAS faithfulness scores, end-to-end generative LLM evaluation, multimodal image/audio retrieval, and calibrated abstention under domain shift.
- **Real-world production readiness**: access control, API deployment, attribution audits at scale, adversarial evaluation, telemetry, operational SLAs and cost budgets.
- **Independent statistical validity**: 16 questions is far too small for precise population-level accuracy inference; the pretrained reader was trained on SQuAD-related material, and gold contexts are explicitly included in the experimental index.

## Reproduction

```bash
python -m pip install requests numpy scikit-learn transformers==4.44.2 torch pytest
python -m pytest -q tests/test_real_squad_qa.py tests/test_basic_rag_local.py
python -m benchmarks.real_squad_qa --corpus-size 240 --train-questions 20 --heldout-questions 16 --seed 42
python -m examples.basic_rag_working
```

Review [the primary README](README.md) for the explanation of the heldout design, source/answer scope, files and GitHub verification. The old aspirational README figures have been removed because their experimental basis was not demonstrated. This project is an implemented research baseline with known limitations, **not** a production AI service.
