"""
Visualize evolution results.
"""
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

def plot_evolution_results():
    """Plot fitness over generations."""
    # Load stats
    with open("outputs/evolution_stats.pkl", 'rb') as f:
        stats = pickle.load(f)

    # Get fitness statistics
    generations = range(len(stats.most_fit_genomes))
    best_fitness = [g.fitness for g in stats.most_fit_genomes]
    mean_fitness = stats.get_fitness_mean()

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_fitness, 'b-', label='Best Fitness', linewidth=2)
    plt.plot(generations, mean_fitness, 'r--', label='Mean Fitness', linewidth=2)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness Score', fontsize=12)
    plt.title('RAG Pipeline Evolution: Fitness Over Generations', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save
    plt.savefig('outputs/evolution_fitness.png', dpi=300, bbox_inches='tight')
    print("Saved plot to outputs/evolution_fitness.png")

    # Print summary
    print(f"\n{'='*60}")
    print("EVOLUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Initial best fitness: {best_fitness[0]:.3f}")
    print(f"Final best fitness: {best_fitness[-1]:.3f}")
    improvement_percent = ((best_fitness[-1] - best_fitness[0]) / best_fitness[0] * 100) if best_fitness[0] != 0 else 0
    print(f"Improvement: {improvement_percent:.1f}%")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    plot_evolution_results()
