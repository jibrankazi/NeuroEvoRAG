class RetrieverAgent:
    """
    Agent responsible for executing retrieval steps and interacting with the evolution process.
    """
    def __init__(self, retriever):
        self.retriever = retriever

    def act(self, query: str, modality: str = "text", top_k: int = 5):
        """
        Use the underlying retriever to fetch relevant context.

        Args:
            query: The user query.
            modality: Which modality to search ('text', 'image', 'audio', or 'mixed').
            top_k: Number of results to fetch.

        Returns:
            Retrieved context results from the retriever.
        """
        return self.retriever.retrieve(query, modality=modality, top_k=top_k)
