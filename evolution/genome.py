"""
RAG genome that actually encodes pipeline parameters.
"""
import random
from neat import DefaultGenome

class RAGGenome(DefaultGenome):
    """
    Genome that encodes RAG pipeline hyperparameters.
    """
    
    def __init__(self, key):
        super().__init__(key)
        # RAG-specific hyperparameters
        self.chunk_size = 512
        self.top_k = 5
        self.temperature = 0.7
    
    def configure_new(self, config):
        """Initialize with random hyperparameters."""
        super().configure_new(config)
        
        # Randomly initialize hyperparameters
        self.chunk_size = random.choice([128, 256, 512, 1024, 2048])
        self.top_k = random.randint(3, 10)
        self.temperature = random.uniform(0.3, 1.0)
    
    def mutate(self, config):
        """Mutate both NEAT structure and RAG hyperparameters."""
        super().mutate(config)
        
        # Mutate chunk_size (30% chance)
        if random.random() < 0.3:
            self.chunk_size = random.choice([128, 256, 512, 1024, 2048])
        
        # Mutate top_k (30% chance)
        if random.random() < 0.3:
            self.top_k = max(1, min(15, self.top_k + random.randint(-2, 2)))
        
        # Mutate temperature (30% chance)
        if random.random() < 0.3:
            self.temperature = max(0.1, min(1.5, self.temperature + random.uniform(-0.2, 0.2)))


def genome_to_pipeline(genome):
    """
    Convert a genome to a working RAG pipeline.
    
    Args:
        genome: RAGGenome instance
        
    Returns:
        BasicRAGPipeline configured with genome's parameters
    """
    from examples.basic_rag_working import BasicRAGPipeline
    
    # Create pipeline with genome's hyperparameters
    pipeline = BasicRAGPipeline(chunk_size=genome.chunk_size)
    pipeline.top_k = genome.top_k
    pipeline.temperature = genome.temperature
    
    return pipeline
