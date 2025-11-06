import random
from neat import Config, DefaultGenome

class RAGGenome(DefaultGenome):
    def configure_new(self, config):
        super().configure_new(config)
        # Nodes: 0=query, 1-10=chunkers, 11-20=embedders, ..., 50=output
        # Connections encode pipeline flow + hyperparameters
        self.nodes[5].bias = random.uniform(128, 2048)  # chunk_size evolves!
