from unittest.mock import MagicMock

import pytest

from rag_pipelines.multimodal_retriever import MultimodalRetriever


class TestMultimodalRetrieverNoRetrievers:

    def test_text_modality_returns_empty(self):
        r = MultimodalRetriever()
        assert r.retrieve("query", modality="text") == []

    def test_image_modality_returns_empty(self):
        r = MultimodalRetriever()
        assert r.retrieve("query", modality="image") == []

    def test_audio_modality_returns_empty(self):
        r = MultimodalRetriever()
        assert r.retrieve("query", modality="audio") == []

    def test_mixed_modality_returns_empty(self):
        r = MultimodalRetriever()
        assert r.retrieve("query", modality="mixed") == []


class TestMultimodalRetrieverWithMocks:

    def test_text_delegates_to_text_retriever(self):
        text_r = MagicMock()
        text_r.retrieve.return_value = ["doc1", "doc2"]
        r = MultimodalRetriever(text_retriever=text_r)
        result = r.retrieve("q", modality="text", top_k=2)
        text_r.retrieve.assert_called_once_with("q", top_k=2)
        assert result == ["doc1", "doc2"]

    def test_image_delegates_to_image_retriever(self):
        img_r = MagicMock()
        img_r.retrieve.return_value = ["img1"]
        r = MultimodalRetriever(image_retriever=img_r)
        result = r.retrieve("q", modality="image", top_k=1)
        img_r.retrieve.assert_called_once_with("q", top_k=1)
        assert result == ["img1"]

    def test_audio_delegates_to_audio_retriever(self):
        audio_r = MagicMock()
        audio_r.retrieve.return_value = ["audio1"]
        r = MultimodalRetriever(audio_retriever=audio_r)
        result = r.retrieve("q", modality="audio", top_k=1)
        audio_r.retrieve.assert_called_once_with("q", top_k=1)
        assert result == ["audio1"]

    def test_mixed_combines_all_retrievers(self):
        text_r = MagicMock()
        text_r.retrieve.return_value = ["t1", "t2"]
        img_r = MagicMock()
        img_r.retrieve.return_value = ["i1"]
        audio_r = MagicMock()
        audio_r.retrieve.return_value = ["a1"]

        r = MultimodalRetriever(
            text_retriever=text_r,
            image_retriever=img_r,
            audio_retriever=audio_r,
        )
        result = r.retrieve("q", modality="mixed", top_k=10)
        assert "t1" in result
        assert "i1" in result
        assert "a1" in result

    def test_mixed_truncates_to_top_k(self):
        text_r = MagicMock()
        text_r.retrieve.return_value = ["t1", "t2", "t3"]
        img_r = MagicMock()
        img_r.retrieve.return_value = ["i1", "i2", "i3"]

        r = MultimodalRetriever(text_retriever=text_r, image_retriever=img_r)
        result = r.retrieve("q", modality="mixed", top_k=4)
        assert len(result) == 4

    def test_mixed_with_only_text_retriever(self):
        text_r = MagicMock()
        text_r.retrieve.return_value = ["t1"]
        r = MultimodalRetriever(text_retriever=text_r)
        result = r.retrieve("q", modality="mixed", top_k=5)
        assert result == ["t1"]

    def test_default_modality_is_text(self):
        text_r = MagicMock()
        text_r.retrieve.return_value = ["doc"]
        r = MultimodalRetriever(text_retriever=text_r)
        result = r.retrieve("q")
        text_r.retrieve.assert_called_once_with("q", top_k=5)

    def test_default_top_k_is_five(self):
        text_r = MagicMock()
        text_r.retrieve.return_value = []
        r = MultimodalRetriever(text_retriever=text_r)
        r.retrieve("q", modality="text")
        text_r.retrieve.assert_called_once_with("q", top_k=5)
