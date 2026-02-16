from typing import List, Any

class AgenticGenerator:
    def __init__(self, llm: Any = None):
        self.llm = llm

    def generate(self, query: str, context: List[str]) -> str:
        prompt = ""
        if context:
            prompt += "Context:\n" + "\n".join(context) + "\n\n"
        prompt += f"Question: {query}\nAnswer: "
        if self.llm is None:
            return "This is a placeholder answer based on the provided context."
        return self.llm(prompt)
