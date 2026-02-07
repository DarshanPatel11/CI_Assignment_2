"""
Ablation study utilities.
Compare dense-only, sparse-only, and hybrid retrieval. Lightweight runner
that executes retrieval using the provided HybridRetriever (or creates one)
and optionally generates answers via a RAGPipeline.
"""
from typing import List, Dict, Any, Optional
import time


class AblationStudyRunner:
    """Run ablation experiments across retrieval configurations."""

    def __init__(self, retriever=None, rag_pipeline=None):
        """Provide a HybridRetriever instance and optional RAGPipeline.
        If not provided, caller must pass retriever to run methods.
        """
        self.retriever = retriever
        self.rag = rag_pipeline

    def run_retrieval_ablations(
        self,
        queries: List[str],
        ks: List[int] = [3, 5, 10],
        rrf_ks: List[int] = [20, 60, 100],
        include_details: bool = False,
        retriever=None
    ) -> Dict[str, Any]:
        """Run retrieval-only ablations.

        Returns a nested dict: {k: {rrf_k: {method: results}}}
        Each result is a list of retrieval outputs for the queries.
        """
        retriever = retriever or self.retriever
        if retriever is None:
            raise ValueError("A retriever instance is required")

        out = {}
        for k in ks:
            out[k] = {}
            for rrf in rrf_ks:
                # temporarily set rrf
                old_rrf = getattr(retriever, 'rrf_k', None)
                retriever.rrf_k = rrf

                configs = {}
                # Dense only
                dense_results = [
                    retriever.retrieve_dense_only(q, top_k=k) for q in queries
                ]
                configs['dense_only'] = dense_results

                # Sparse only
                sparse_results = [
                    retriever.retrieve_sparse_only(q, top_k=k) for q in queries
                ]
                configs['sparse_only'] = sparse_results

                # Hybrid
                hybrid_results = [
                    retriever.retrieve(q, dense_top_k=k*2, sparse_top_k=k*2, final_top_n=k) for q in queries
                ]
                configs['hybrid'] = hybrid_results

                if include_details and self.rag is not None:
                    # generate answers for hybrid only (optional)
                    gen = [self.rag.answer(q, top_k=k, include_details=True) for q in queries]
                    configs['hybrid_with_generation'] = gen

                out[k][rrf] = configs

                # restore rrf
                if old_rrf is not None:
                    retriever.rrf_k = old_rrf

        return out

    def run_end_to_end_ablation(
        self,
        qa_pairs: List[Dict],
        ks: List[int] = [3, 5],
        retriever=None
    ) -> Dict[str, Any]:
        """Run end-to-end ablation where answers are generated and basic
        metrics (e.g., exact-match vs gold) are computed. Expects qa_pairs
        to be list of {'question': str, 'answer': str} entries.
        """
        retriever = retriever or self.retriever
        if retriever is None or self.rag is None:
            raise ValueError("Both retriever and rag_pipeline must be provided for end-to-end ablation")

        results = {}
        for k in ks:
            results[k] = []
            for item in qa_pairs:
                q = item.get('question') or item.get('q')
                gold = item.get('answer') or item.get('gold')

                resp = self.rag.answer(q, top_k=k, include_details=True)
                pred = resp.get('answer', '')

                # Simple exact-match metric (case-insensitive)
                em = int(pred.strip().lower() == (gold or '').strip().lower())

                results[k].append({
                    'question': q,
                    'gold': gold,
                    'pred': pred,
                    'exact_match': em,
                    'sources': resp.get('sources', []),
                    'retrieval': resp.get('retrieval', {})
                })

        return results


if __name__ == "__main__":
    print("AblationStudyRunner module")

