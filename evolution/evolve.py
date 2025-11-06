from neat import Population
import joblib
from .reward_model import evaluate_pipeline


def evolve_rag_pipeline(config, initial_genomes, generations=100):
    """
    Run neuroevolution on the RAG pipeline population.

    Args:
        config: NEAT configuration object.
        initial_genomes: Initial population of genomes.
        generations: Number of generations to evolve.

    Returns:
        The winning genome after evolution.
    """
    pop = Population(config, initial_genomes)
    winner = pop.run(fitness_function=evaluate_pipeline, generations=generations)
    # After evolution, save the best pipeline for later use
    joblib.dump(winner, "best_rag_pipeline_ever.pkl")
    return winner
