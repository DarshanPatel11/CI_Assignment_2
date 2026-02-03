"""
LLM-as-judge utilities.
Provides a pluggable judge interface. By default uses lightweight heuristics,
but can accept a callable (e.g., an LLM wrapper) to perform detailed
scoring with explanations.

The judge callable signature should be:
    judge_fn(question: str, answer: str, context: str) -> dict
and return a dict with keys: factual_accuracy (1-5), completeness (1-5),
relevance (1-5), coherence (1-5), groundedness (1-5), explanation (str).
"""
from typing import Callable, Dict, Any
import math


class LLMJudge:
    def __init__(self, judge_fn: Callable[[str, str, str], Dict[str, Any]] = None):
        """Initialize with an optional judge function (e.g., an LLM wrapper).
        If not provided, use a simple heuristic-based judge.
        """
        self.judge_fn = judge_fn

    def _heuristic_judge(self, question: str, answer: str, context: str) -> Dict[str, Any]:
        """Very simple heuristic scoring and short explanation."""
        # Basic signals
        answer_len = len(answer.strip())
        context_len = len(context.strip())

        # Factual accuracy heuristic: presence of named entities and numbers from context
        factual = 3
        if any(tok in context for tok in answer.split()[:5]):
            factual = min(5, factual + 1)
        if 'cannot answer' in answer.lower():
            factual = 1

        # Completeness: length-based heuristic
        completeness = 1 if answer_len < 20 else (3 if answer_len < 80 else 5)

        # Relevance: overlap between question tokens and answer tokens
        q_tokens = set(question.lower().split())
        a_tokens = set(answer.lower().split())
        overlap = len(q_tokens & a_tokens) / max(1, len(q_tokens))
        relevance = int(max(1, min(5, round(overlap * 5))))

        # Coherence: crude check for sentence punctuation
        coherence = 5 if answer.endswith('.') or answer.endswith('?') else 3

        # Groundedness: fraction of answer tokens present in context
        ctx_tokens = set(context.lower().split())
        groundedness = int(max(1, min(5, round((len(a_tokens & ctx_tokens) / max(1, len(a_tokens))) * 5))))

        explanation = (
            f"Heuristic judge: answer_len={answer_len}, context_len={context_len}, "
            f"overlap={overlap:.2f}, grounded_fraction={len(a_tokens & ctx_tokens)}/{max(1,len(a_tokens))}"
        )

        return {
            'factual_accuracy': factual,
            'completeness': completeness,
            'relevance': relevance,
            'coherence': coherence,
            'groundedness': groundedness,
            'explanation': explanation
        }

    def judge(self, question: str, answer: str, context: str) -> Dict[str, Any]:
        """Score a single QA pair using the provided judge or heuristic."""
        if self.judge_fn is not None:
            try:
                out = self.judge_fn(question, answer, context)
                # validate minimal structure
                for k in ['factual_accuracy', 'completeness', 'relevance', 'coherence', 'groundedness']:
                    if k not in out:
                        out[k] = 3
                if 'explanation' not in out:
                    out['explanation'] = ''
                return out
            except Exception as e:
                # fallback to heuristic
                return {**self._heuristic_judge(question, answer, context), 'note': f'fallback due to {e}'}

        return self._heuristic_judge(question, answer, context)


if __name__ == "__main__":
    j = LLMJudge()
    print(j.judge("When was Python created?", "Python was created in 1991 by Guido van Rossum.", "Python was created in 1991 by Guido van Rossum."))

