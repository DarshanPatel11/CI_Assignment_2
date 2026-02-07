"""
Error analysis utilities for categorizing and summarizing failures.
"""
from typing import List, Dict, Any
import re


class ErrorAnalyzer:
    """Categorize failures into types and produce simple summaries."""

    ERROR_CATEGORIES = [
        'retrieval_failure',
        'context_insufficient',
        'generation_error',
        'hallucination',
        'partial_answer',
        'unanswerable'
    ]

    def __init__(self):
        pass

    def _extract_entities(self, text: str) -> List[str]:
        # Very simple heuristic: consecutive Capitalized words
        ents = re.findall(r"\b([A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*)\b", text)
        return list(set([e.strip() for e in ents if len(e) > 1]))

    def categorize_failure(self, question: str, gold: str, pred: str, context: str) -> str:
        """Return a category label for a single QA result."""
        # Unanswerable detection
        if 'cannot answer' in pred.lower() or 'i cannot' in pred.lower():
            return 'unanswerable'

        # Hallucination heuristic: tokens in pred not found in context
        pred_tokens = set(re.findall(r"\w+", pred.lower()))
        context_tokens = set(re.findall(r"\w+", context.lower()))

        # If many tokens in pred are not in context, label hallucination
        novel_tokens = pred_tokens - context_tokens
        if len(novel_tokens) / max(1, len(pred_tokens)) > 0.3:
            return 'hallucination'

        # Retrieval failure: gold entities not present in any retrieved context
        gold_ents = self._extract_entities(gold or '')
        ctx_ents = self._extract_entities(context or '')
        if gold_ents and not any(e in ctx_ents for e in gold_ents):
            return 'retrieval_failure'

        # Partial answer: if gold has multiple facts and pred contains only some
        gold_tokens = set(re.findall(r"\w+", (gold or '').lower()))
        if gold_tokens and len(pred_tokens & gold_tokens) / len(gold_tokens) < 0.5:
            return 'partial_answer'

        # Default: generation error (good context but poor phrasing)
        return 'generation_error'

    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze a list of result dicts. Each result should contain:
        { 'question', 'gold', 'pred', 'context' }

        Returns an aggregate summary including counts per category and
        sample examples.
        """
        summary = {k: {'count': 0, 'examples': []} for k in self.ERROR_CATEGORIES}

        for item in results:
            q = item.get('question', '')
            gold = item.get('gold', '')
            pred = item.get('pred', '')
            context = item.get('context', '')

            cat = self.categorize_failure(q, gold, pred, context)
            summary.setdefault(cat, {'count': 0, 'examples': []})
            summary[cat]['count'] += 1

            if len(summary[cat]['examples']) < 5:
                summary[cat]['examples'].append({
                    'question': q,
                    'gold': gold,
                    'pred': pred
                })

        total = len(results)
        summary['total'] = total
        summary['distribution'] = {k: summary[k]['count'] for k in summary if k != 'total'}
        return summary


if __name__ == "__main__":
    ea = ErrorAnalyzer()
    sample = [
        {'question': 'When was Python released?', 'gold': '1991', 'pred': 'I cannot answer', 'context': ''},
        {'question': 'Who created Python?', 'gold': 'Guido van Rossum', 'pred': 'John Doe', 'context': 'Python was created by Guido van Rossum.'}
    ]
    print(ea.analyze_results(sample))

