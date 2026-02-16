import random
from neat import DefaultGenome

class RAGGenome(DefaultGenome):
    def __init__(self, key):
        super().__init__(key)
        self.chunk_size = 512
        self.top_k = 5
        self.temperature = 0.7

    def configure_new(self, config):
        super().configure_new(config)
        self.chunk_size = random.choice([128, 256, 512, 1024, 2048])
        self.top_k = random.randint(3, 10)
        self.temperature = random.uniform(0.3, 1.0)

    def mutate(self, config):
        super().mutate(config)

        if random.random() < 0.3:
            self.chunk_size = random.choice([128, 256, 512, 1024, 2048])

        if random.random() < 0.3:
            self.top_k = max(1, min(15, self.top_k + random.randint(-2, 2)))

        if random.random() < 0.3:
            self.temperature = max(0.1, min(1.5, self.temperature + random.uniform(-0.2, 0.2)))


def genome_to_pipeline(genome):
    from examples.basic_rag_working import BasicRAGPipeline

    pipeline = BasicRAGPipeline(chunk_size=genome.chunk_size)
    pipeline.top_k = genome.top_k
    pipeline.temperature = genome.temperature

    return pipeline
