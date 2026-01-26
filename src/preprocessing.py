"""
Text Preprocessing Module

Handles chunking of Wikipedia articles with:
- 200-400 tokens per chunk
- 50-token overlap
- Unique chunk IDs and metadata
"""

import hashlib
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

import tiktoken
import nltk
from tqdm import tqdm

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    chunk_id: str
    url: str
    title: str
    content: str
    token_count: int
    chunk_index: int
    total_chunks: int

    def to_dict(self) -> dict:
        return asdict(self)


class TextPreprocessor:
    """Preprocesses and chunks Wikipedia articles."""

    def __init__(
        self,
        min_tokens: int = 200,
        max_tokens: int = 400,
        overlap_tokens: int = 50,
        data_dir: str = "data"
    ):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.data_dir = Path(data_dir)

        # Use tiktoken for accurate token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.tokenizer.encode(text))

    def _generate_chunk_id(self, url: str, chunk_index: int) -> str:
        """Generate unique chunk ID from URL and index."""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        return f"{url_hash}_{chunk_index:04d}"

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK."""
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception:
            # Fallback: simple splitting
            sentences = text.replace('!', '.').replace('?', '.').split('.')
            sentences = [s.strip() + '.' for s in sentences if s.strip()]
        return sentences

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = ' '.join(text.split())

        # Remove very short lines (likely headers/navigation)
        lines = text.split('\n')
        lines = [line for line in lines if len(line.split()) > 3]
        text = ' '.join(lines)

        return text.strip()

    def chunk_article(
        self,
        url: str,
        title: str,
        content: str
    ) -> List[Chunk]:
        """
        Chunk an article into 200-400 token segments with 50-token overlap.
        Uses sentence boundaries for natural splits.
        """
        content = self._clean_text(content)
        sentences = self._split_into_sentences(content)

        if not sentences:
            return []

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            # If single sentence exceeds max, split by words
            if sentence_tokens > self.max_tokens:
                # Flush current chunk if needed
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    if self._count_tokens(chunk_text) >= self.min_tokens:
                        chunks.append(chunk_text)
                    current_chunk = []
                    current_tokens = 0

                # Split long sentence by words
                words = sentence.split()
                word_chunk = []
                word_tokens = 0

                for word in words:
                    word_t = self._count_tokens(word + ' ')
                    if word_tokens + word_t > self.max_tokens:
                        if word_chunk:
                            chunks.append(' '.join(word_chunk))
                        word_chunk = [word]
                        word_tokens = word_t
                    else:
                        word_chunk.append(word)
                        word_tokens += word_t

                if word_chunk:
                    current_chunk = word_chunk
                    current_tokens = word_tokens
                continue

            # Check if adding this sentence exceeds max
            if current_tokens + sentence_tokens > self.max_tokens:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                if self._count_tokens(chunk_text) >= self.min_tokens:
                    chunks.append(chunk_text)

                # Start new chunk with overlap
                # Find overlap sentences from end of current chunk
                overlap_text = ""
                overlap_sentences = []

                for s in reversed(current_chunk):
                    test_text = s + " " + overlap_text
                    if self._count_tokens(test_text) <= self.overlap_tokens:
                        overlap_sentences.insert(0, s)
                        overlap_text = test_text.strip()
                    else:
                        break

                current_chunk = overlap_sentences + [sentence]
                current_tokens = self._count_tokens(' '.join(current_chunk))
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if self._count_tokens(chunk_text) >= self.min_tokens // 2:  # Allow smaller final chunk
                chunks.append(chunk_text)

        # Create Chunk objects with metadata
        chunk_objects = []
        for idx, text in enumerate(chunks):
            chunk_objects.append(Chunk(
                chunk_id=self._generate_chunk_id(url, idx),
                url=url,
                title=title,
                content=text,
                token_count=self._count_tokens(text),
                chunk_index=idx,
                total_chunks=len(chunks)
            ))

        return chunk_objects

    def process_articles(
        self,
        articles: List[Dict]
    ) -> List[Chunk]:
        """
        Process multiple articles and return all chunks.

        Args:
            articles: List of dicts with 'url', 'title', 'content' keys

        Returns:
            List of Chunk objects
        """
        all_chunks = []

        for article in tqdm(articles, desc="Chunking articles"):
            chunks = self.chunk_article(
                url=article['url'],
                title=article['title'],
                content=article['content']
            )
            all_chunks.extend(chunks)

        print(f"Created {len(all_chunks)} chunks from {len(articles)} articles")
        return all_chunks

    def save_chunks(
        self,
        chunks: List[Chunk],
        filename: str = "chunks.json"
    ) -> Path:
        """Save chunks to JSON file."""
        chunks_path = self.data_dir / "corpus" / filename
        chunks_path.parent.mkdir(parents=True, exist_ok=True)

        # Create summary statistics
        token_counts = [c.token_count for c in chunks]

        with open(chunks_path, 'w') as f:
            json.dump({
                "total_chunks": len(chunks),
                "total_articles": len(set(c.url for c in chunks)),
                "avg_tokens": sum(token_counts) / len(token_counts) if token_counts else 0,
                "min_tokens": min(token_counts) if token_counts else 0,
                "max_tokens": max(token_counts) if token_counts else 0,
                "chunks": [c.to_dict() for c in chunks]
            }, f, indent=2)

        print(f"Saved {len(chunks)} chunks to {chunks_path}")
        return chunks_path

    def load_chunks(self, filename: str = "chunks.json") -> List[Chunk]:
        """Load chunks from JSON file."""
        chunks_path = self.data_dir / "corpus" / filename

        with open(chunks_path, 'r') as f:
            data = json.load(f)

        return [Chunk(**c) for c in data["chunks"]]


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = TextPreprocessor()

    sample_articles = [{
        "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "title": "Python (programming language)",
        "content": """
        Python is a high-level, general-purpose programming language. Its design philosophy
        emphasizes code readability with the use of significant indentation. Python is
        dynamically typed and garbage-collected. It supports multiple programming paradigms,
        including structured, object-oriented and functional programming. Python was conceived
        in the late 1980s by Guido van Rossum at Centrum Wiskunde & Informatica in the
        Netherlands as a successor to the ABC programming language, which was inspired by
        SETL, capable of exception handling and interfacing with the Amoeba operating system.
        Its implementation began in December 1989. Python consistently ranks as one of the
        most popular programming languages.
        """ * 10  # Repeat to make it longer
    }]

    chunks = preprocessor.process_articles(sample_articles)
    print(f"\nCreated {len(chunks)} chunks")

    for chunk in chunks[:3]:
        print(f"  Chunk {chunk.chunk_id}: {chunk.token_count} tokens")
        print(f"    Preview: {chunk.content[:100]}...")
