"""
Real RAGAS evaluation that actually computes metrics.
"""
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset


def evaluate_pipeline_real(pipeline, questions, ground_truths, contexts=None):
    """
    Evaluate a RAG pipeline using RAGAS metrics for real.
    
    Args:
        pipeline: RAG pipeline with .query() method
        questions: List of question strings
        ground_truths: List of ground truth answer strings
        contexts: Optional list of context lists (if None, will retrieve)
    
    Returns:
        Dict with RAGAS metric scores
    """
    # Run pipeline on all questions
    results = []
    for i, question in enumerate(questions):
        result = pipeline.query(question)
        results.append({
            'question': question,
            'answer': result['answer'],
            'contexts': [result['context']],  # RAGAS expects list of contexts
            'ground_truth': ground_truths[i]
        })
    
    # Convert to RAGAS dataset format
    dataset = Dataset.from_dict({
        'question': [r['question'] for r in results],
        'answer': [r['answer'] for r in results],
        'contexts': [r['contexts'] for r in results],
        'ground_truth': [r['ground_truth'] for r in results]
    })
    
    # Evaluate with RAGAS
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision]
    )
    
    return {
        'faithfulness': scores['faithfulness'],
        'answer_relevancy': scores['answer_relevancy'],
        'context_precision': scores['context_precision'],
        'num_questions': len(questions)
    }


# Example usage
if __name__ == "__main__":
    from examples.basic_rag_working import BasicRAGPipeline
    
    # Load a few HotpotQA examples
    from datasets import load_dataset
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    
    # Take just 10 examples for quick test
    sample = dataset.select(range(10))
    
    # Set up pipeline
    pipeline = BasicRAGPipeline(chunk_size=512)
    
    # Add documents from the dataset
    all_contexts = []
    for item in sample:
        for context in item['context']['sentences']:
            all_contexts.extend(context)
    pipeline.add_documents(all_contexts)
    
    # Evaluate
    questions = [item['question'] for item in sample]
    ground_truths = [item['answer'] for item in sample]
    
    scores = evaluate_pipeline_real(pipeline, questions, ground_truths)
    
    print("BASELINE SCORES (no evolution):")
    print(f"Faithfulness: {scores['faithfulness']:.3f}")
    print(f"Answer Relevancy: {scores['answer_relevancy']:.3f}")
    print(f"Context Precision: {scores['context_precision']:.3f}")
