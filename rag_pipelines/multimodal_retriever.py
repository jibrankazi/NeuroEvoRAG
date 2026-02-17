"""Multimodal retrieval: text (ChromaDB + sentence-transformers) and image (CLIP)."""

import uuid
import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful imports -- degrade to None so callers get clear errors at runtime
# rather than import-time crashes when optional deps are missing.
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    logger.warning(
        "sentence-transformers is not installed. "
        "TextRetriever and ImageRetriever will not be available."
    )

try:
    import chromadb
except ImportError:
    chromadb = None
    logger.warning(
        "chromadb is not installed. TextRetriever will not be available."
    )

try:
    import numpy as np
except ImportError:
    np = None
    logger.warning(
        "numpy is not installed. ImageRetriever will not be available."
    )


# ---------------------------------------------------------------------------
# TextRetriever -- all-MiniLM-L6-v2 + ChromaDB
# ---------------------------------------------------------------------------
class TextRetriever:
    """Dense text retriever backed by ChromaDB and a sentence-transformer model.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for the sentence-transformer encoder.
    collection_name : str
        Name of the ChromaDB collection to use.
    persist_directory : str or None
        If provided, ChromaDB will persist data to this directory.
        If None, an ephemeral in-memory client is used.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "text_documents",
        persist_directory: Optional[str] = None,
    ):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for TextRetriever. "
                "Install it with: pip install sentence-transformers"
            )
        if chromadb is None:
            raise ImportError(
                "chromadb is required for TextRetriever. "
                "Install it with: pip install chromadb"
            )

        self.model = SentenceTransformer(model_name)
        self._embedding_dim = self.model.get_sentence_embedding_dimension()

        if persist_directory is not None:
            self.chroma_client = chromadb.PersistentClient(
                path=persist_directory,
            )
        else:
            self.chroma_client = chromadb.Client()

        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    def add_documents(self, documents: List[str]) -> int:
        """Embed and store *documents* in the ChromaDB collection.

        Returns the total number of documents in the collection after insert.
        """
        if not documents:
            return self.collection.count()

        embeddings = self.model.encode(documents, show_progress_bar=False)
        ids = [str(uuid.uuid4()) for _ in documents]

        self.collection.add(
            ids=ids,
            embeddings=[emb.tolist() for emb in embeddings],
            documents=documents,
        )
        return self.collection.count()

    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Return up to *top_k* documents most similar to *query*."""
        if self.collection.count() == 0:
            return []

        top_k = min(top_k, self.collection.count())
        query_embedding = self.model.encode([query], show_progress_bar=False)

        results = self.collection.query(
            query_embeddings=[query_embedding[0].tolist()],
            n_results=top_k,
        )

        # results["documents"] is a list-of-lists (one list per query)
        return results["documents"][0] if results["documents"] else []


# ---------------------------------------------------------------------------
# ImageRetriever -- CLIP (clip-ViT-B-32)
# ---------------------------------------------------------------------------
class ImageRetriever:
    """Image retriever using CLIP embeddings for text-to-image matching.

    Images are represented by their file paths and associated captions.
    Retrieval is done by encoding the text query with CLIP and comparing
    against pre-encoded caption/image embeddings.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for the CLIP model.
    """

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for ImageRetriever. "
                "Install it with: pip install sentence-transformers"
            )
        if np is None:
            raise ImportError(
                "numpy is required for ImageRetriever. "
                "Install it with: pip install numpy"
            )

        self.model = SentenceTransformer(model_name)

        # Internal storage
        self._image_paths: List[str] = []
        self._captions: List[str] = []
        self._embeddings: Optional[Any] = None  # np.ndarray | None

    # ------------------------------------------------------------------
    def add_images(self, image_paths: List[str], captions: List[str]) -> int:
        """Register images with their captions and compute CLIP embeddings.

        Parameters
        ----------
        image_paths : list[str]
            File-system paths (or URLs) to images.
        captions : list[str]
            Human-readable captions, one per image.  These are what the
            CLIP text encoder will embed for matching.

        Returns
        -------
        int
            Total number of images stored after this call.
        """
        if len(image_paths) != len(captions):
            raise ValueError(
                f"image_paths ({len(image_paths)}) and captions "
                f"({len(captions)}) must have the same length."
            )
        if not image_paths:
            return len(self._image_paths)

        new_embeddings = self.model.encode(captions, show_progress_bar=False)

        self._image_paths.extend(image_paths)
        self._captions.extend(captions)

        if self._embeddings is None:
            self._embeddings = np.array(new_embeddings)
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])

        return len(self._image_paths)

    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Return captions (with paths) of images most relevant to *query*.

        Each returned string has the format ``"<caption> [<path>]"``.
        """
        if self._embeddings is None or len(self._image_paths) == 0:
            return []

        query_embedding = self.model.encode([query], show_progress_bar=False)
        query_vec = np.array(query_embedding[0])

        # Cosine similarity
        norms = np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_vec)
        # Guard against zero-norm vectors
        norms = np.where(norms == 0, 1e-10, norms)
        similarities = np.dot(self._embeddings, query_vec) / norms

        top_k = min(top_k, len(self._image_paths))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            f"{self._captions[i]} [{self._image_paths[i]}]"
            for i in top_indices
        ]


# ---------------------------------------------------------------------------
# MultimodalRetriever -- orchestrator
# ---------------------------------------------------------------------------
class MultimodalRetriever:
    """Orchestrates retrieval across text, image, and audio modalities.

    Each modality retriever is optional; if a retriever is ``None``, queries
    for that modality simply return an empty list.
    """

    def __init__(
        self,
        text_retriever: Any = None,
        image_retriever: Any = None,
        audio_retriever: Any = None,
    ):
        self.text_retriever = text_retriever
        self.image_retriever = image_retriever
        self.audio_retriever = audio_retriever

    def retrieve(
        self, query: str, modality: str = "text", top_k: int = 5
    ) -> List[Any]:
        """Retrieve results for *query* from one or all modalities.

        Parameters
        ----------
        query : str
            The search query.
        modality : str
            One of ``"text"``, ``"image"``, ``"audio"``, or ``"all"``.
        top_k : int
            Maximum number of results to return.
        """
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
            # "all" or any unrecognised value -> merge across modalities
            results: List[Any] = []
            if self.text_retriever:
                results.extend(self.text_retriever.retrieve(query, top_k=top_k))
            if self.image_retriever:
                results.extend(self.image_retriever.retrieve(query, top_k=top_k))
            if self.audio_retriever:
                results.extend(self.audio_retriever.retrieve(query, top_k=top_k))
            return results[:top_k]


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
def create_multimodal_retriever(
    persist_directory: Optional[str] = None,
    text_model: str = "all-MiniLM-L6-v2",
    clip_model: str = "clip-ViT-B-32",
    collection_name: str = "text_documents",
) -> MultimodalRetriever:
    """Build a MultimodalRetriever with TextRetriever and ImageRetriever.

    Either retriever is silently skipped if its dependencies are missing,
    so the returned object always has a usable ``.retrieve()`` method.

    Parameters
    ----------
    persist_directory : str or None
        Passed through to ``TextRetriever`` for ChromaDB persistence.
    text_model : str
        Sentence-transformer model name for text embeddings.
    clip_model : str
        Sentence-transformer model name for CLIP image-text matching.
    collection_name : str
        ChromaDB collection name for text documents.

    Returns
    -------
    MultimodalRetriever
    """
    text_retriever = None
    image_retriever = None

    # --- Text ---
    try:
        text_retriever = TextRetriever(
            model_name=text_model,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        logger.info("TextRetriever initialised with model=%s", text_model)
    except (ImportError, Exception) as exc:
        logger.warning("Could not initialise TextRetriever: %s", exc)

    # --- Image ---
    try:
        image_retriever = ImageRetriever(model_name=clip_model)
        logger.info("ImageRetriever initialised with model=%s", clip_model)
    except (ImportError, Exception) as exc:
        logger.warning("Could not initialise ImageRetriever: %s", exc)

    return MultimodalRetriever(
        text_retriever=text_retriever,
        image_retriever=image_retriever,
    )
