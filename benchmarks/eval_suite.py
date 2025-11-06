from datasets import load_from_disk
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision


def run_evaluation(pipeline, dataset_name="hotpotqa"):
    """
    Run evaluation on a given RAG pipeline using a specified dataset.
    The dataset is loaded from the benchmarks/datasets directory on disk and
    evaluated with RAGAS metrics. A subset of 100 examples is used to reduce evaluation time.
    """
    dataset = load_from_disk(f"benchmarks/datasets/{dataset_name}")["test"]
    sampled_data = dataset.select(range(100))
    results = evaluate(
        pipeline,
        data=sampled_data,
        metrics=[faithfulness, answer_relevancy, context_precision]
    )
    return results
