import pytest

from rag_pipelines.dynamic_chunker import DynamicChunker


class TestDynamicChunker:
    def test_default_max_chunk_size(self):
        chunker = DynamicChunker()
        assert chunker.max_chunk_size == 512

    def test_custom_max_chunk_size(self):
        chunker = DynamicChunker(max_chunk_size=256)
        assert chunker.max_chunk_size == 256

    def test_empty_string(self):
        chunker = DynamicChunker(max_chunk_size=100)
        assert chunker.chunk("") == []

    def test_text_shorter_than_chunk_size(self):
        chunker = DynamicChunker(max_chunk_size=100)
        result = chunker.chunk("short text")
        assert result == ["short text"]

    def test_text_exactly_chunk_size(self):
        chunker = DynamicChunker(max_chunk_size=10)
        result = chunker.chunk("0123456789")
        assert result == ["0123456789"]

    def test_text_splits_correctly(self):
        chunker = DynamicChunker(max_chunk_size=5)
        result = chunker.chunk("abcdefghij")
        assert result == ["abcde", "fghij"]

    def test_last_chunk_can_be_shorter(self):
        chunker = DynamicChunker(max_chunk_size=4)
        result = chunker.chunk("abcdefg")
        assert result == ["abcd", "efg"]

    def test_no_data_loss(self):
        chunker = DynamicChunker(max_chunk_size=7)
        text = "The quick brown fox jumps over the lazy dog."
        chunks = chunker.chunk(text)
        assert "".join(chunks) == text

    def test_all_chunks_respect_max_size(self):
        chunker = DynamicChunker(max_chunk_size=10)
        text = "a" * 55
        chunks = chunker.chunk(text)
        for chunk in chunks:
            assert len(chunk) <= 10

    def test_chunk_count(self):
        chunker = DynamicChunker(max_chunk_size=10)
        text = "a" * 25
        chunks = chunker.chunk(text)
        assert len(chunks) == 3  # 10 + 10 + 5
