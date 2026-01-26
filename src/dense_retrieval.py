"""
Dense Vector Retrieval Module

Uses sentence-transformers for embeddings and FAISS for similarity search.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class DenseRetriever:
    """Dense vector retrieval using sentence embeddings and FAISS."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        data_dir: str = "data"
    ):
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.indices_dir = self.data_dir / "indices"
        self.indices_dir.mkdir(parents=True, exist_ok=True)

        # Load embedding model
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # FAISS index and metadata
        self.index: Optional[faiss.Index] = None
        self.chunk_ids: List[str] = []
        self.chunk_metadata: Dict[str, Dict] = {}

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-10)

    def build_index(
        self,
        chunks: List[Dict],
        batch_size: int = 64,
        show_progress: bool = True
    ) -> None:
        """
        Build FAISS index from chunks.

        Args:
            chunks: List of chunk dicts with 'chunk_id', 'content', 'url', 'title'
            batch_size: Batch size for encoding
            show_progress: Show progress bar
        """
        print(f"Building dense index with {len(chunks)} chunks...")

        # Store metadata
        self.chunk_ids = [c['chunk_id'] for c in chunks]
        self.chunk_metadata = {
            c['chunk_id']: {
                'url': c['url'],
                'title': c['title'],
                'content': c['content']
            }
            for c in chunks
        }

        # Extract texts for encoding
        texts = [c['content'] for c in chunks]

        # Encode in batches
        all_embeddings = []
        iterator = range(0, len(texts), batch_size)

        if show_progress:
            iterator = tqdm(iterator, desc="Encoding chunks", total=len(texts)//batch_size + 1)

        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            all_embeddings.append(batch_embeddings)

        embeddings = np.vstack(all_embeddings).astype('float32')

        # Normalize for cosine similarity
        embeddings = self._normalize_embeddings(embeddings)

        # Build FAISS index (Inner Product = Cosine Similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)

        print(f"Dense index built: {self.index.ntotal} vectors")

    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve top-K chunks for a query.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (chunk_id, similarity_score, metadata) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Encode query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        ).astype('float32')

        # Normalize
        query_embedding = self._normalize_embeddings(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.chunk_ids):
                chunk_id = self.chunk_ids[idx]
                metadata = self.chunk_metadata.get(chunk_id, {})
                results.append((chunk_id, float(score), metadata))

        return results

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 10
    ) -> List[List[Tuple[str, float, Dict]]]:
        """Batch retrieve for multiple queries."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Encode all queries
        query_embeddings = self.model.encode(
            queries,
            convert_to_numpy=True,
            show_progress_bar=True
        ).astype('float32')

        # Normalize
        query_embeddings = self._normalize_embeddings(query_embeddings)

        # Batch search
        all_scores, all_indices = self.index.search(query_embeddings, top_k)

        all_results = []
        for scores, indices in zip(all_scores, all_indices):
            results = []
            for idx, score in zip(indices, scores):
                if idx < len(self.chunk_ids):
                    chunk_id = self.chunk_ids[idx]
                    metadata = self.chunk_metadata.get(chunk_id, {})
                    results.append((chunk_id, float(score), metadata))
            all_results.append(results)

        return all_results

    def save_index(self, name: str = "dense_index"):
        """Save FAISS index and metadata."""
        if self.index is None:
            raise ValueError("No index to save")

        # Save FAISS index
        index_path = self.indices_dir / f"{name}.faiss"
        faiss.write_index(self.index, str(index_path))

        # Save metadata
        meta_path = self.indices_dir / f"{name}_meta.pkl"
        with open(meta_path, 'wb') as f:
            pickle.dump({
                'chunk_ids': self.chunk_ids,
                'chunk_metadata': self.chunk_metadata,
                'model_name': self.model_name
            }, f)

        print(f"Dense index saved to {index_path}")

    def load_index(self, name: str = "dense_index"):
        """Load FAISS index and metadata."""
        index_path = self.indices_dir / f"{name}.faiss"
        meta_path = self.indices_dir / f"{name}_meta.pkl"

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Index files not found at {index_path}")

        # Load FAISS index
        self.index = faiss.read_index(str(index_path))

        # Load metadata
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        self.chunk_ids = meta['chunk_ids']
        self.chunk_metadata = meta['chunk_metadata']

        print(f"Dense index loaded: {self.index.ntotal} vectors")


if __name__ == "__main__":
    # Test dense retrieval
    retriever = DenseRetriever()

    sample_chunks = [
        {"chunk_id": "test_001", "content": "Python is a programming language", "url": "http://test1", "title": "Python"},
        {"chunk_id": "test_002", "content": "Machine learning uses algorithms", "url": "http://test2", "title": "ML"},
        {"chunk_id": "test_003", "content": "Natural language processing handles text", "url": "http://test3", "title": "NLP"},
    ]

    retriever.build_index(sample_chunks)

    results = retriever.retrieve("What is Python?", top_k=2)
    print("\nQuery: What is Python?")
    for chunk_id, score, meta in results:
        print(f"  {chunk_id}: {score:.4f} - {meta['title']}")
