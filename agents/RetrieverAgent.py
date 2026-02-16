class RetrieverAgent:
    def __init__(self, retriever):
        self.retriever = retriever

    def act(self, query: str, modality: str = "text", top_k: int = 5):
        return self.retriever.retrieve(query, modality=modality, top_k=top_k)
