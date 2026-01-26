"""
Evaluation Metrics Module

Implements:
1. MRR at URL level (mandatory)
2. Faithfulness Score (LLM-as-Judge) - custom metric 1
3. Context Precision - custom metric 2
"""

import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class EvaluationMetrics:
    """
    Evaluation metrics for RAG system including:
    - MRR (Mean Reciprocal Rank) at URL level
    - Faithfulness Score (hallucination detection)
    - Context Precision (retrieval ranking quality)
    """

    def __init__(
        self,
        llm_model: str = "google/flan-t5-base",
        embedding_model: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize metrics calculator.

        Args:
            llm_model: Model for faithfulness evaluation
            embedding_model: Model for semantic similarity
            device: Device for computation
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # LLM for faithfulness evaluation
        print(f"Loading evaluation LLM: {llm_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.llm = AutoModelForSeq2SeqLM.from_pretrained(llm_model)
        self.llm.to(self.device)
        self.llm.eval()

        # Embedding model for semantic similarity
        print(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)

    # =========================================================================
    # MRR (Mean Reciprocal Rank) at URL Level - MANDATORY METRIC
    # =========================================================================

    def calculate_mrr_single(
        self,
        retrieved_urls: List[str],
        ground_truth_url: str
    ) -> float:
        """
        Calculate MRR for a single query at URL level.

        MRR = 1/rank where rank is the position of the first correct URL.

        Args:
            retrieved_urls: List of retrieved URLs in rank order
            ground_truth_url: The correct source URL

        Returns:
            Reciprocal rank (0 if not found)
        """
        # Normalize URLs for comparison
        gt_normalized = self._normalize_url(ground_truth_url)

        for rank, url in enumerate(retrieved_urls, start=1):
            if self._normalize_url(url) == gt_normalized:
                return 1.0 / rank

        return 0.0

    def calculate_mrr(
        self,
        all_retrieved_urls: List[List[str]],
        all_ground_truth_urls: List[str]
    ) -> Dict:
        """
        Calculate Mean Reciprocal Rank across all queries.

        Args:
            all_retrieved_urls: List of retrieved URL lists per query
            all_ground_truth_urls: List of ground truth URLs

        Returns:
            Dict with 'mrr', 'individual_rrs', 'hit_rate'
        """
        reciprocal_ranks = []

        for retrieved, gt in zip(all_retrieved_urls, all_ground_truth_urls):
            rr = self.calculate_mrr_single(retrieved, gt)
            reciprocal_ranks.append(rr)

        mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
        hit_rate = sum(1 for rr in reciprocal_ranks if rr > 0) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

        return {
            'mrr': float(mrr),
            'individual_rrs': reciprocal_ranks,
            'hit_rate': float(hit_rate),
            'total_queries': len(reciprocal_ranks)
        }

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for comparison."""
        url = url.lower().strip()
        url = url.rstrip('/')
        # Remove protocol
        url = re.sub(r'^https?://', '', url)
        return url

    # =========================================================================
    # Faithfulness Score (LLM-as-Judge) - CUSTOM METRIC 1
    # =========================================================================
    #
    # JUSTIFICATION:
    # Faithfulness measures whether the generated answer is grounded in the
    # retrieved context. Unlike ROUGE or BLEU which measure surface-level
    # similarity, faithfulness detects hallucinations - cases where the model
    # generates plausible but unsupported information. This is critical for
    # RAG systems where factual accuracy depends on proper context usage.
    #
    # CALCULATION:
    # 1. Extract atomic claims from the generated answer
    # 2. For each claim, verify if it's supported by the retrieved context
    # 3. Faithfulness = (supported_claims / total_claims)
    #
    # INTERPRETATION:
    # - 1.0 = Fully grounded (all claims supported by context)
    # - 0.7-0.9 = Generally reliable (minor unsupported details)
    # - 0.5-0.7 = Partially reliable (significant unsupported content)
    # - < 0.5 = Unreliable (substantial hallucination)
    # =========================================================================

    def calculate_faithfulness_single(
        self,
        answer: str,
        context: str
    ) -> Dict:
        """
        Calculate faithfulness score for a single answer.

        Uses LLM to:
        1. Extract claims from the answer
        2. Verify each claim against context

        Args:
            answer: Generated answer
            context: Retrieved context used for generation

        Returns:
            Dict with 'score', 'claims', 'supported_claims', 'claim_details'
        """
        # Step 1: Extract claims from answer
        claims = self._extract_claims(answer)

        if not claims:
            return {
                'score': 1.0,  # No claims = nothing to verify
                'claims': [],
                'supported_claims': 0,
                'total_claims': 0,
                'claim_details': []
            }

        # Step 2: Verify each claim
        claim_details = []
        supported_count = 0

        for claim in claims:
            is_supported, confidence = self._verify_claim(claim, context)
            claim_details.append({
                'claim': claim,
                'supported': is_supported,
                'confidence': confidence
            })
            if is_supported:
                supported_count += 1

        # Calculate final score
        faithfulness_score = supported_count / len(claims)

        return {
            'score': float(faithfulness_score),
            'claims': claims,
            'supported_claims': supported_count,
            'total_claims': len(claims),
            'claim_details': claim_details
        }

    def _extract_claims(self, answer: str) -> List[str]:
        """Extract atomic claims from an answer using LLM."""
        if not answer or len(answer.strip()) < 10:
            return []

        prompt = f"""Extract the key factual claims from this answer. List each claim on a new line.

Answer: {answer}

Claims:"""

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=256,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.3,
                    do_sample=False
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Parse claims
            claims = [c.strip() for c in response.split('\n') if c.strip() and len(c.strip()) > 10]

            # Limit to reasonable number
            return claims[:5]

        except Exception:
            # Fallback: split by sentences
            sentences = answer.split('.')
            return [s.strip() + '.' for s in sentences if len(s.strip()) > 15][:5]

    def _verify_claim(self, claim: str, context: str) -> Tuple[bool, float]:
        """Verify if a claim is supported by context."""
        prompt = f"""Determine if the following claim is supported by the context.
Answer with 'Yes' if supported, 'No' if not supported, 'Partial' if partially supported.

Context: {context[:1000]}

Claim: {claim}

Is this claim supported by the context?"""

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).lower()

            if 'yes' in response:
                return True, 1.0
            elif 'partial' in response:
                return True, 0.5
            else:
                return False, 0.0

        except Exception:
            # Fallback: use semantic similarity
            return self._semantic_verify(claim, context)

    def _semantic_verify(self, claim: str, context: str) -> Tuple[bool, float]:
        """Fallback verification using semantic similarity."""
        claim_emb = self.embedder.encode([claim])

        # Split context into sentences
        sentences = context.split('.')
        if not sentences:
            return False, 0.0

        sentence_embs = self.embedder.encode(sentences)

        # Find max similarity
        similarities = cosine_similarity(claim_emb, sentence_embs)[0]
        max_sim = float(np.max(similarities))

        # Threshold for support
        is_supported = max_sim > 0.6

        return is_supported, max_sim

    def calculate_faithfulness(
        self,
        answers: List[str],
        contexts: List[str]
    ) -> Dict:
        """
        Calculate faithfulness scores for multiple answers.

        Returns:
            Dict with 'mean_score', 'scores', 'details'
        """
        all_results = []

        for answer, context in zip(answers, contexts):
            result = self.calculate_faithfulness_single(answer, context)
            all_results.append(result)

        scores = [r['score'] for r in all_results]

        return {
            'mean_score': float(np.mean(scores)) if scores else 0.0,
            'scores': scores,
            'details': all_results,
            'total_evaluated': len(all_results)
        }

    # =========================================================================
    # Context Precision - CUSTOM METRIC 2
    # =========================================================================
    #
    # JUSTIFICATION:
    # Context Precision evaluates the quality of retrieval ranking - whether
    # relevant chunks appear higher in the results than irrelevant ones.
    # Unlike simple Recall@K, this metric weighs the position of relevant
    # documents, rewarding systems that rank relevant content first.
    #
    # CALCULATION:
    # Context_Precision = Σ(precision@k × relevance_k) / total_relevant
    # where precision@k = (relevant docs in top k) / k
    #
    # INTERPRETATION:
    # - 1.0 = Perfect ranking (all relevant docs at the top)
    # - 0.7-0.9 = Good ranking (relevant docs mostly at top)
    # - 0.5-0.7 = Moderate (mixed ranking quality)
    # - < 0.5 = Poor (relevant docs buried in results)
    # =========================================================================

    def calculate_context_precision_single(
        self,
        retrieved_chunks: List[Dict],
        ground_truth_url: str,
        ground_truth_answer: str
    ) -> Dict:
        """
        Calculate context precision for a single query.

        Args:
            retrieved_chunks: List of retrieved chunks with 'url', 'content'
            ground_truth_url: Correct source URL
            ground_truth_answer: Ground truth answer for relevance check

        Returns:
            Dict with 'score', 'relevance_vector', 'precision_at_k'
        """
        if not retrieved_chunks:
            return {
                'score': 0.0,
                'relevance_vector': [],
                'precision_at_k': []
            }

        # Determine relevance for each chunk
        relevance_vector = []
        gt_url_normalized = self._normalize_url(ground_truth_url)

        for chunk in retrieved_chunks:
            chunk_url = chunk.get('url', '')
            chunk_content = chunk.get('content', '')

            # Check URL match
            url_match = self._normalize_url(chunk_url) == gt_url_normalized

            # Check content relevance using semantic similarity
            if ground_truth_answer:
                content_relevance = self._calculate_relevance(chunk_content, ground_truth_answer)
            else:
                content_relevance = 0.0

            # Chunk is relevant if URL matches OR high content relevance
            is_relevant = url_match or content_relevance > 0.5
            relevance_vector.append(1 if is_relevant else 0)

        # Calculate precision at each position
        precision_at_k = []
        cumulative_relevant = 0

        for k, is_rel in enumerate(relevance_vector, start=1):
            cumulative_relevant += is_rel
            precision_at_k.append(cumulative_relevant / k)

        # Calculate context precision (weighted average)
        total_relevant = sum(relevance_vector)

        if total_relevant == 0:
            context_precision = 0.0
        else:
            weighted_sum = sum(
                prec * rel
                for prec, rel in zip(precision_at_k, relevance_vector)
            )
            context_precision = weighted_sum / total_relevant

        return {
            'score': float(context_precision),
            'relevance_vector': relevance_vector,
            'precision_at_k': precision_at_k,
            'total_relevant': total_relevant,
            'total_retrieved': len(retrieved_chunks)
        }

    def _calculate_relevance(self, chunk_content: str, answer: str) -> float:
        """Calculate semantic relevance between chunk and answer."""
        if not chunk_content or not answer:
            return 0.0

        try:
            embeddings = self.embedder.encode([chunk_content, answer])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception:
            return 0.0

    def calculate_context_precision(
        self,
        all_retrieved_chunks: List[List[Dict]],
        all_ground_truth_urls: List[str],
        all_ground_truth_answers: List[str]
    ) -> Dict:
        """
        Calculate context precision for multiple queries.

        Returns:
            Dict with 'mean_score', 'scores', 'details'
        """
        all_results = []

        for chunks, gt_url, gt_answer in zip(
            all_retrieved_chunks,
            all_ground_truth_urls,
            all_ground_truth_answers
        ):
            result = self.calculate_context_precision_single(chunks, gt_url, gt_answer)
            all_results.append(result)

        scores = [r['score'] for r in all_results]

        return {
            'mean_score': float(np.mean(scores)) if scores else 0.0,
            'scores': scores,
            'details': all_results,
            'total_evaluated': len(all_results)
        }

    # =========================================================================
    # Comprehensive Evaluation
    # =========================================================================

    def evaluate_all(
        self,
        queries: List[str],
        generated_answers: List[str],
        contexts_used: List[str],
        retrieved_chunks: List[List[Dict]],
        ground_truth_urls: List[str],
        ground_truth_answers: List[str]
    ) -> Dict:
        """
        Run all evaluation metrics.

        Returns comprehensive evaluation results.
        """
        print("Evaluating with all metrics...")

        # Extract URLs from retrieved chunks
        all_retrieved_urls = [
            [chunk.get('url', '') for chunk in chunks]
            for chunks in retrieved_chunks
        ]

        # MRR
        print("  Computing MRR...")
        mrr_result = self.calculate_mrr(all_retrieved_urls, ground_truth_urls)

        # Faithfulness
        print("  Computing Faithfulness...")
        faithfulness_result = self.calculate_faithfulness(generated_answers, contexts_used)

        # Context Precision
        print("  Computing Context Precision...")
        context_precision_result = self.calculate_context_precision(
            retrieved_chunks,
            ground_truth_urls,
            ground_truth_answers
        )

        return {
            'mrr': mrr_result,
            'faithfulness': faithfulness_result,
            'context_precision': context_precision_result,
            'summary': {
                'mrr': mrr_result['mrr'],
                'hit_rate': mrr_result['hit_rate'],
                'faithfulness': faithfulness_result['mean_score'],
                'context_precision': context_precision_result['mean_score'],
                'total_queries': len(queries)
            }
        }


if __name__ == "__main__":
    # Test metrics
    metrics = EvaluationMetrics()

    # Test MRR
    retrieved_urls = [
        ["https://en.wikipedia.org/wiki/Python", "https://en.wikipedia.org/wiki/Java"],
        ["https://en.wikipedia.org/wiki/Java", "https://en.wikipedia.org/wiki/Python"]
    ]
    ground_truth_urls = ["https://en.wikipedia.org/wiki/Python", "https://en.wikipedia.org/wiki/Python"]

    mrr_result = metrics.calculate_mrr(retrieved_urls, ground_truth_urls)
    print(f"MRR: {mrr_result['mrr']:.4f}")

    # Test Faithfulness
    answer = "Python was created by Guido van Rossum in 1991."
    context = "Python was conceived in the late 1980s by Guido van Rossum."

    faith_result = metrics.calculate_faithfulness_single(answer, context)
    print(f"Faithfulness: {faith_result['score']:.4f}")

    # Test Context Precision
    chunks = [
        {"url": "https://en.wikipedia.org/wiki/Python", "content": "Python is a programming language."},
        {"url": "https://en.wikipedia.org/wiki/Java", "content": "Java is a programming language."}
    ]

    cp_result = metrics.calculate_context_precision_single(
        chunks,
        "https://en.wikipedia.org/wiki/Python",
        "Python is a programming language created by Guido."
    )
    print(f"Context Precision: {cp_result['score']:.4f}")
