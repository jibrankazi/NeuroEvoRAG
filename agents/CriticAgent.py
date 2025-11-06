class CriticAgent:
    """
    Agent that critiques generated answers and provides feedback for improvements.
    """

    def __init__(self, evaluator=None):
        """
        Initialize the critic agent with an evaluator function or model.
        """
        self.evaluator = evaluator

    def critique(self, query: str, context: list, answer: str) -> float:
        """
        Evaluate the answer with respect to query and context.

        Args:
            query: The user question.
            context: Context used to generate the answer.
            answer: The generated answer.

        Returns:
            A numeric score or feedback rating.
        """
        if self.evaluator is None:
            # Placeholder: return 0.0 if no evaluator provided
            return 0.0
        return self.evaluator(query, context, answer)
