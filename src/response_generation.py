"""
Response Generation Module

Uses Flan-T5 for answer generation from retrieved context.
"""

import time
from typing import List, Dict, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class ResponseGenerator:
    """Generates answers using Flan-T5 language model."""

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        max_context_tokens: int = 450,
        max_answer_tokens: int = 150,
        device: Optional[str] = None
    ):
        """
        Initialize the response generator.

        Args:
            model_name: HuggingFace model name
            max_context_tokens: Maximum tokens for context
            max_answer_tokens: Maximum tokens for generated answer
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.max_context_tokens = max_context_tokens
        self.max_answer_tokens = max_answer_tokens

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading model: {model_name} on {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully")

    def _truncate_context(
        self,
        chunks: List[Dict],
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Combine chunks into context, truncating to fit token limit.

        Args:
            chunks: List of chunk dicts with 'content' key
            max_tokens: Maximum context tokens (uses default if None)

        Returns:
            Combined context string
        """
        max_tokens = max_tokens or self.max_context_tokens

        context_parts = []
        current_tokens = 0

        for chunk in chunks:
            content = chunk.get('content', '')
            chunk_tokens = len(self.tokenizer.encode(content))

            if current_tokens + chunk_tokens <= max_tokens:
                context_parts.append(content)
                current_tokens += chunk_tokens
            else:
                # Add partial chunk if possible
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 50:  # Only add if meaningful
                    tokens = self.tokenizer.encode(content)[:remaining_tokens]
                    partial = self.tokenizer.decode(tokens, skip_special_tokens=True)
                    context_parts.append(partial + "...")
                break

        return "\n\n".join(context_parts)

    def _build_prompt(self, query: str, context: str) -> str:
        """Build the prompt for the model."""
        prompt = f"""Answer the following question based on the provided context.
If the context doesn't contain enough information, say "I cannot answer this based on the provided context."

Context:
{context}

Question: {query}

Answer:"""
        return prompt

    def generate(
        self,
        query: str,
        chunks: List[Dict],
        temperature: float = 0.7,
        num_beams: int = 4,
        include_timing: bool = False
    ) -> Dict:
        """
        Generate an answer for the query using retrieved chunks.

        Args:
            query: User question
            chunks: Retrieved chunks (list of dicts with 'content', 'url', 'title')
            temperature: Sampling temperature
            num_beams: Number of beams for beam search
            include_timing: Include timing information

        Returns:
            Dict with 'answer', 'context_used', and optionally 'timing_ms'
        """
        start_time = time.time()

        # Build context from chunks
        context = self._truncate_context(chunks)

        # Build prompt
        prompt = self._build_prompt(query, context)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_answer_tokens,
                temperature=temperature,
                num_beams=num_beams,
                do_sample=temperature > 0,
                early_stopping=True
            )

        # Decode answer
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Calculate timing
        timing_ms = (time.time() - start_time) * 1000

        result = {
            'answer': answer.strip(),
            'query': query,
            'context_used': context,
            'num_chunks': len(chunks),
            'sources': [
                {'url': c.get('url', ''), 'title': c.get('title', '')}
                for c in chunks
            ]
        }

        if include_timing:
            result['timing_ms'] = timing_ms

        return result

    def batch_generate(
        self,
        queries: List[str],
        batch_chunks: List[List[Dict]],
        **kwargs
    ) -> List[Dict]:
        """Generate answers for multiple queries."""
        results = []

        for query, chunks in zip(queries, batch_chunks):
            result = self.generate(query, chunks, **kwargs)
            results.append(result)

        return results


class RAGPipeline:
    """Complete RAG pipeline combining retrieval and generation."""

    def __init__(
        self,
        hybrid_retriever,
        generator: Optional[ResponseGenerator] = None
    ):
        """
        Initialize RAG pipeline.

        Args:
            hybrid_retriever: HybridRetriever instance
            generator: ResponseGenerator instance (or None to create)
        """
        self.retriever = hybrid_retriever
        self.generator = generator or ResponseGenerator()

    def answer(
        self,
        query: str,
        top_k: int = 5,
        include_details: bool = True
    ) -> Dict:
        """
        Answer a question using the full RAG pipeline.

        Args:
            query: User question
            top_k: Number of chunks to retrieve
            include_details: Include retrieval details in response

        Returns:
            Dict with 'answer', 'sources', and retrieval details
        """
        start_time = time.time()

        # Retrieve relevant chunks
        retrieval_result = self.retriever.retrieve(
            query,
            dense_top_k=top_k * 2,
            sparse_top_k=top_k * 2,
            final_top_n=top_k,
            include_timings=True
        )

        # Extract chunks for generation
        chunks = [
            {
                'content': meta.get('content', ''),
                'url': meta.get('url', ''),
                'title': meta.get('title', '')
            }
            for chunk_id, score, meta in retrieval_result['results']
        ]

        # Generate answer
        gen_result = self.generator.generate(
            query,
            chunks,
            include_timing=True
        )

        total_time = (time.time() - start_time) * 1000

        response = {
            'answer': gen_result['answer'],
            'query': query,
            'sources': gen_result['sources'],
            'timing': {
                'total_ms': total_time,
                'retrieval_ms': retrieval_result.get('timings', {}).get('total_ms', 0),
                'generation_ms': gen_result.get('timing_ms', 0)
            }
        }

        if include_details:
            response['retrieval'] = {
                'hybrid_results': [
                    {
                        'chunk_id': cid,
                        'score': score,
                        'title': meta.get('title', ''),
                        'url': meta.get('url', '')
                    }
                    for cid, score, meta in retrieval_result['results']
                ],
                'dense_results': [
                    {'chunk_id': cid, 'score': score}
                    for cid, score, _ in retrieval_result['dense_results']
                ],
                'sparse_results': [
                    {'chunk_id': cid, 'score': score}
                    for cid, score, _ in retrieval_result['sparse_results']
                ]
            }
            response['context_used'] = gen_result['context_used']

        return response


if __name__ == "__main__":
    # Test response generation
    generator = ResponseGenerator()

    sample_chunks = [
        {
            "content": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.",
            "url": "https://en.wikipedia.org/wiki/Python",
            "title": "Python"
        },
        {
            "content": "Python was conceived in the late 1980s by Guido van Rossum at Centrum Wiskunde & Informatica in the Netherlands.",
            "url": "https://en.wikipedia.org/wiki/Python",
            "title": "Python"
        }
    ]

    result = generator.generate(
        query="When was Python created and by whom?",
        chunks=sample_chunks,
        include_timing=True
    )

    print(f"Query: {result['query']}")
    print(f"Answer: {result['answer']}")
    print(f"Time: {result.get('timing_ms', 0):.1f}ms")
