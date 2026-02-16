class DynamicChunker:
    def __init__(self, max_chunk_size: int = 512):
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str):
        return [text[i:i + self.max_chunk_size] for i in range(0, len(text), self.max_chunk_size)]
