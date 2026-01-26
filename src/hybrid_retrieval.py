"""
Hybrid Retrieval Module

Combines dense and sparse retrieval using Reciprocal Rank Fusion (RRF).
"""

from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import time

from .dense_retrieval import DenseRetriever
from .sparse_retrieval import SparseRetriever


class HybridRetriever:
    """
    Hybrid retrieval combining dense semantic search and sparse BM25
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[SparseRetriever] = None,
        rrf_k: int = 60,
        data_dir: str = "data"
    ):
        """
        Initialize hybrid retriever.

        Args:
            dense_retriever: Pre-initialized dense retriever (or None to create)
            sparse_retriever: Pre-initialized sparse retriever (or None to create)
            rrf_k: RRF constant (default 60 as per standard)
            data_dir: Data directory path
        """
        self.dense = dense_retriever or DenseRetriever(data_dir=data_dir)
        self.sparse = sparse_retriever or SparseRetriever(data_dir=data_dir)
        self.rrf_k = rrf_k

        self.chunks: List[Dict] = []

    def build_indices(self, chunks: List[Dict]) -> None:
        """
        Build both dense and sparse indices.

        Args:
            chunks: List of chunk dicts with 'chunk_id', 'content', 'url', 'title'
        """
        self.chunks = chunks

        print("Building hybrid indices...")
        print("-" * 50)

        # Build dense index
        self.dense.build_index(chunks)

        # Build sparse index
        self.sparse.build_index(chunks)

        print("-" * 50)
        print("Hybrid indices built successfully!")

    @staticmethod
    def reciprocal_rank_fusion(
        rankings: List[List[Tuple[str, float, Dict]]],
        k: int = 60
    ) -> List[Tuple[str, float, Dict]]:
        """
        Apply Reciprocal Rank Fusion to multiple rankings.

        RRF_score(d) = Σ 1/(k + rank_i(d))

        Args:
            rankings: List of rankings, each is [(chunk_id, score, metadata), ...]
            k: RRF constant (default 60)

        Returns:
            Fused ranking sorted by RRF score
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        metadata_map: Dict[str, Dict] = {}
        original_scores: Dict[str, Dict[str, float]] = defaultdict(dict)

        for ranking_idx, ranking in enumerate(rankings):
            for rank, (chunk_id, score, metadata) in enumerate(ranking, start=1):
                # RRF formula: 1 / (k + rank)
                rrf_scores[chunk_id] += 1.0 / (k + rank)

                # Store metadata
                if chunk_id not in metadata_map:
                    metadata_map[chunk_id] = metadata

                # Store original scores for debugging
                source_name = f"ranking_{ranking_idx}"
                original_scores[chunk_id][source_name] = score

        # Sort by RRF score (descending)
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Build result with metadata
        results = []
        for chunk_id, rrf_score in sorted_items:
            metadata = metadata_map.get(chunk_id, {})
            # Add RRF score and original scores to metadata
            metadata = metadata.copy()
            metadata['rrf_score'] = rrf_score
            metadata['original_scores'] = original_scores.get(chunk_id, {})
            results.append((chunk_id, rrf_score, metadata))

        return results

    def retrieve(
        self,
        query: str,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        final_top_n: int = 10,
        include_timings: bool = False
    ) -> Dict:
        """
        Retrieve chunks using hybrid approach.

        Args:
            query: Search query
            dense_top_k: Top-K for dense retrieval
            sparse_top_k: Top-K for sparse retrieval
            final_top_n: Final number of chunks to return after RRF
            include_timings: Include timing breakdown in response

        Returns:
            Dict with 'results', 'dense_results', 'sparse_results', and optionally 'timings'
        """
        timings = {}

        # Dense retrieval
        start = time.time()
        dense_results = self.dense.retrieve(query, top_k=dense_top_k)
        timings['dense_ms'] = (time.time() - start) * 1000

        # Sparse retrieval
        start = time.time()
        sparse_results = self.sparse.retrieve(query, top_k=sparse_top_k)
        timings['sparse_ms'] = (time.time() - start) * 1000

        # RRF fusion
        start = time.time()
        fused_results = self.reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=self.rrf_k
        )
        timings['rrf_ms'] = (time.time() - start) * 1000

        # Take top N
        final_results = fused_results[:final_top_n]

        response = {
            'results': final_results,
            'dense_results': dense_results[:final_top_n],
            'sparse_results': sparse_results[:final_top_n],
            'query': query,
            'params': {
                'dense_top_k': dense_top_k,
                'sparse_top_k': sparse_top_k,
                'final_top_n': final_top_n,
                'rrf_k': self.rrf_k
            }
        }

        if include_timings:
            response['timings'] = timings
            response['timings']['total_ms'] = sum(timings.values())

        return response

    def retrieve_dense_only(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[str, float, Dict]]:
        """Retrieve using only dense retrieval (for ablation studies)."""
        return self.dense.retrieve(query, top_k=top_k)

    def retrieve_sparse_only(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[str, float, Dict]]:
        """Retrieve using only sparse retrieval (for ablation studies)."""
        return self.sparse.retrieve(query, top_k=top_k)

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Get chunk metadata by ID."""
        return self.dense.chunk_metadata.get(chunk_id)

    def save_indices(self, name: str = "hybrid"):
        """Save both indices."""
        self.dense.save_index(f"{name}_dense")
        self.sparse.save_index(f"{name}_sparse")
        print(f"Hybrid indices saved with prefix: {name}")

    def load_indices(self, name: str = "hybrid"):
        """Load both indices."""
        self.dense.load_index(f"{name}_dense")
        self.sparse.load_index(f"{name}_sparse")
        print(f"Hybrid indices loaded with prefix: {name}")


def format_retrieval_results(results: List[Tuple[str, float, Dict]], max_content_len: int = 200) -> str:
    """Format retrieval results for display."""
    output = []

    for i, (chunk_id, score, metadata) in enumerate(results, 1):
        content = metadata.get('content', '')[:max_content_len]
        if len(metadata.get('content', '')) > max_content_len:
            content += "..."

        output.append(f"[{i}] {metadata.get('title', 'Unknown')} (Score: {score:.4f})")
        output.append(f"    URL: {metadata.get('url', 'N/A')}")
        output.append(f"    {content}")
        output.append("")

    return "\n".join(output)


if __name__ == "__main__":
    # Test hybrid retrieval
    hybrid = HybridRetriever()

    sample_chunks = [
        {"chunk_id": "test_001", "content": "Python is a high-level programming language known for its readability and versatility", "url": "http://test1", "title": "Python"},
        {"chunk_id": "test_002", "content": "Machine learning is a subset of artificial intelligence that learns from data", "url": "http://test2", "title": "Machine Learning"},
        {"chunk_id": "test_003", "content": "Natural language processing enables computers to understand human language", "url": "http://test3", "title": "NLP"},
        {"chunk_id": "test_004", "content": "Deep learning uses neural networks with multiple layers for complex patterns", "url": "http://test4", "title": "Deep Learning"},
    ]

    hybrid.build_indices(sample_chunks)

    # Test hybrid retrieval
    query = "programming artificial intelligence"
    results = hybrid.retrieve(query, final_top_n=3, include_timings=True)

    print(f"\nQuery: {query}")
    print("=" * 60)
    print("\nHybrid Results:")
    print(format_retrieval_results(results['results']))

    print("\nTimings:", results.get('timings', {}))

    # Compare with single methods
    print("\n--- Ablation Study ---")
    dense_only = hybrid.retrieve_dense_only(query, top_k=3)
    sparse_only = hybrid.retrieve_sparse_only(query, top_k=3)

    print("\nDense Only:", [r[0] for r in dense_only])
    print("Sparse Only:", [r[0] for r in sparse_only])
    print("Hybrid:", [r[0] for r in results['results']])
