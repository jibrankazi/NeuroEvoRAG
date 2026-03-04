"""
NEAT-based evolution that actually runs.
"""
import neat
import pickle
from pathlib import Path
from evolution.genome import RAGGenome, genome_to_pipeline
from benchmarks.eval_suite import evaluate_pipeline_real
from datasets import load_dataset

# Global dataset (loaded once)
EVAL_DATASET = None
EVAL_QUESTIONS = None
EVAL_ANSWERS = None
EVAL_CONTEXTS = None

def load_eval_data():
    """Load evaluation dataset once."""
    global EVAL_DATASET, EVAL_QUESTIONS, EVAL_ANSWERS, EVAL_CONTEXTS
    
    if EVAL_DATASET is None:
        print("Loading HotpotQA evaluation dataset...")
        dataset = load_dataset("hotpot_qa", "distractor", split="validation")
        sample = dataset.select(range(20))  # Use 20 examples for speed
        
        EVAL_DATASET = sample
        EVAL_QUESTIONS = [item['question'] for item in sample]
        EVAL_ANSWERS = [item['answer'] for item in sample]
        
        # Flatten contexts
        all_contexts = []
        for item in sample:
            for context in item['context']['sentences']:
                all_contexts.extend(context)
        EVAL_CONTEXTS = all_contexts


def eval_genome(genome, config):
    """
    Evaluate a single genome's fitness.
    
    Args:
        genome: RAGGenome to evaluate
        config: NEAT config
        
    Returns:
        Fitness score (higher is better)
    """
    try:
        # Convert genome to pipeline
        pipeline = genome_to_pipeline(genome)
        
        # Add documents
        pipeline.add_documents(EVAL_CONTEXTS)
        
        # Evaluate with RAGAS
        scores = evaluate_pipeline_real(
            pipeline,
            EVAL_QUESTIONS,
            EVAL_ANSWERS
        )
        
        # Composite fitness (weighted average)
        fitness = (
            scores['faithfulness'] * 0.5 +
            scores['answer_relevancy'] * 0.3 +
            scores['context_precision'] * 0.2
        )
        
        print(f"Genome {genome.key}: chunk_size={genome.chunk_size}, "
              f"top_k={genome.top_k}, fitness={fitness:.3f}")
        
        return fitness
        
    except Exception as e:
        print(f"Error evaluating genome {genome.key}: {e}")
        return 0.0


def eval_genomes(genomes, config):
    """Evaluate all genomes in population."""
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run_evolution(generations=10):
    """
    Run NEAT evolution on RAG pipelines.
    
    Args:
        generations: Number of generations to evolve
    """
    # Load evaluation data
    load_eval_data()
    
    # Load NEAT config
    config_path = Path(__file__).parent / "neat_config.txt"
    config = neat.Config(
        RAGGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path)
    )
    
    # Create population
    pop = neat.Population(config)
    
    # Add reporters
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    
    # Run evolution
    print(f"\n{'='*60}")
    print(f"Starting evolution for {generations} generations")
    print(f"{'='*60}\n")
    
    winner = pop.run(eval_genomes, generations)
    
    # Save winner
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "best_genome.pkl", 'wb') as f:
        pickle.dump(winner, f)
    
    # Save stats
    with open(output_dir / "evolution_stats.pkl", 'wb') as f:
        pickle.dump(stats, f)
    
    print(f"\n{'='*60}")
    print("Evolution complete!")
    print(f"Best genome: {winner.key}")
    print(f"Best fitness: {winner.fitness:.3f}")
    print(f"Best chunk_size: {winner.chunk_size}")
    print(f"Best top_k: {winner.top_k}")
    print(f"{'='*60}\n")
    
    return winner, stats


if __name__ == "__main__":
    winner, stats = run_evolution(generations=10)
