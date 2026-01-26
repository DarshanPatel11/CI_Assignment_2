"""
Sparse Keyword Retrieval Module

Implements BM25 algorithm for keyword-based retrieval.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re

import numpy as np
from rank_bm25 import BM25Okapi
import nltk
from tqdm import tqdm

# Download NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords


class SparseRetriever:
    """Sparse keyword retrieval using BM25."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.indices_dir = self.data_dir / "indices"
        self.indices_dir.mkdir(parents=True, exist_ok=True)

        # BM25 index
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []
        self.chunk_ids: List[str] = []
        self.chunk_metadata: Dict[str, Dict] = {}

        # Stopwords for preprocessing
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and preprocess text for BM25."""
        # Lowercase
        text = text.lower()

        # Remove special characters, keep alphanumeric
        text = re.sub(r'[^a-z0-9\s]', ' ', text)

        # Tokenize
        tokens = text.split()

        # Remove stopwords and short tokens
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]

        return tokens

    def build_index(
        self,
        chunks: List[Dict],
        show_progress: bool = True
    ) -> None:
        """
        Build BM25 index from chunks.

        Args:
            chunks: List of chunk dicts with 'chunk_id', 'content', 'url', 'title'
            show_progress: Show progress bar
        """
        print(f"Building BM25 index with {len(chunks)} chunks...")

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

        # Tokenize corpus
        iterator = chunks
        if show_progress:
            iterator = tqdm(chunks, desc="Tokenizing for BM25")

        self.tokenized_corpus = [
            self._tokenize(c['content'])
            for c in iterator
        ]

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print(f"BM25 index built: {len(self.tokenized_corpus)} documents")

    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve top-K chunks for a query using BM25.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (chunk_id, bm25_score, metadata) tuples
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Tokenize query
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-K indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include positive scores
                chunk_id = self.chunk_ids[idx]
                metadata = self.chunk_metadata.get(chunk_id, {})
                results.append((chunk_id, float(scores[idx]), metadata))

        return results

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 10
    ) -> List[List[Tuple[str, float, Dict]]]:
        """Batch retrieve for multiple queries."""
        all_results = []

        for query in tqdm(queries, desc="BM25 retrieval"):
            results = self.retrieve(query, top_k)
            all_results.append(results)

        return all_results

    def get_document_frequencies(self, top_n: int = 20) -> Dict[str, int]:
        """Get top N most frequent terms in the corpus."""
        if not self.tokenized_corpus:
            return {}

        term_freq = {}
        for doc in self.tokenized_corpus:
            for term in set(doc):  # Count each term once per doc
                term_freq[term] = term_freq.get(term, 0) + 1

        # Sort by frequency
        sorted_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)

        return dict(sorted_terms[:top_n])

    def save_index(self, name: str = "sparse_index"):
        """Save BM25 index and metadata."""
        if self.bm25 is None:
            raise ValueError("No index to save")

        index_path = self.indices_dir / f"{name}.pkl"

        with open(index_path, 'wb') as f:
            pickle.dump({
                'tokenized_corpus': self.tokenized_corpus,
                'chunk_ids': self.chunk_ids,
                'chunk_metadata': self.chunk_metadata
            }, f)

        print(f"BM25 index saved to {index_path}")

    def load_index(self, name: str = "sparse_index"):
        """Load BM25 index and metadata."""
        index_path = self.indices_dir / f"{name}.pkl"

        if not index_path.exists():
            raise FileNotFoundError(f"Index not found at {index_path}")

        with open(index_path, 'rb') as f:
            data = pickle.load(f)

        self.tokenized_corpus = data['tokenized_corpus']
        self.chunk_ids = data['chunk_ids']
        self.chunk_metadata = data['chunk_metadata']

        # Rebuild BM25 from tokenized corpus
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print(f"BM25 index loaded: {len(self.tokenized_corpus)} documents")


if __name__ == "__main__":
    # Test sparse retrieval
    retriever = SparseRetriever()

    sample_chunks = [
        {"chunk_id": "test_001", "content": "Python is a high-level programming language known for readability", "url": "http://test1", "title": "Python"},
        {"chunk_id": "test_002", "content": "Machine learning algorithms learn from data patterns", "url": "http://test2", "title": "ML"},
        {"chunk_id": "test_003", "content": "Natural language processing analyzes human language text", "url": "http://test3", "title": "NLP"},
    ]

    retriever.build_index(sample_chunks)

    results = retriever.retrieve("programming language Python", top_k=2)
    print("\nQuery: programming language Python")
    for chunk_id, score, meta in results:
        print(f"  {chunk_id}: {score:.4f} - {meta['title']}")

    # Document frequencies
    print("\nTop terms:", retriever.get_document_frequencies(5))
