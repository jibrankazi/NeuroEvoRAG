class CriticAgent:
    def __init__(self, evaluator=None):
        self.evaluator = evaluator

    def critique(self, query: str, context: list, answer: str) -> float:
        if self.evaluator is None:
            return 0.0
        return self.evaluator(query, context, answer)
