from typing import List, Any

class MultimodalRetriever:
    def __init__(self, text_retriever: Any = None, image_retriever: Any = None, audio_retriever: Any = None):
        self.text_retriever = text_retriever
        self.image_retriever = image_retriever
        self.audio_retriever = audio_retriever

    def retrieve(self, query: str, modality: str = "text", top_k: int = 5) -> List[Any]:
        if modality == "text":
            if self.text_retriever is None:
                return []
            return self.text_retriever.retrieve(query, top_k=top_k)
        elif modality == "image":
            if self.image_retriever is None:
                return []
            return self.image_retriever.retrieve(query, top_k=top_k)
        elif modality == "audio":
            if self.audio_retriever is None:
                return []
            return self.audio_retriever.retrieve(query, top_k=top_k)
        else:
            results = []
            if self.text_retriever:
                results.extend(self.text_retriever.retrieve(query, top_k=top_k))
            if self.image_retriever:
                results.extend(self.image_retriever.retrieve(query, top_k=top_k))
            if self.audio_retriever:
                results.extend(self.audio_retriever.retrieve(query, top_k=top_k))
            return results[:top_k]
