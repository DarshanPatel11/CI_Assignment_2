"""
Novel custom metrics for deeper evaluation insights.
Includes entity coverage, answer diversity, hallucination rate by topic,
and a simple semantic drift detector.
"""
from typing import List, Dict, Any
import re
from collections import Counter


class NovelMetrics:
    def __init__(self):
        pass

    def _extract_entities(self, text: str) -> List[str]:
        ents = re.findall(r"\b([A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*)\b", text)
        return list(set([e.strip() for e in ents if len(e) > 1]))

    def entity_coverage_score(self, answer: str, gold_entities: List[str]) -> float:
        if not gold_entities:
            return 0.0
        ans_ents = self._extract_entities(answer)
        hit = sum(1 for e in gold_entities if e in ans_ents)
        return hit / len(gold_entities)

    def answer_diversity_metric(self, answers: List[str]) -> float:
        # Use token-level diversity (unique tokens / total tokens)
        toks = [t for a in answers for t in re.findall(r"\w+", a.lower())]
        if not toks:
            return 0.0
        uniq = len(set(toks))
        return uniq / len(toks)

    def hallucination_rate_by_topic(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        # Expect results to include 'topic' and 'is_hallucination' boolean
        by_topic = {}
        counts = Counter()
        halluc_counts = Counter()
        for r in results:
            topic = r.get('topic', 'unknown')
            counts[topic] += 1
            if r.get('is_hallucination'):
                halluc_counts[topic] += 1

        for t in counts:
            by_topic[t] = halluc_counts[t] / counts[t]
        return by_topic

    def semantic_drift_detection(self, question: str, answer: str) -> float:
        # Measure drift as 1 - (overlap(question, answer) / average_length)
        q_toks = set(re.findall(r"\w+", question.lower()))
        a_toks = set(re.findall(r"\w+", answer.lower()))
        if not q_toks or not a_toks:
            return 1.0
        overlap = len(q_toks & a_toks)
        avg_len = (len(q_toks) + len(a_toks)) / 2.0
        drift = 1.0 - (overlap / max(1.0, avg_len))
        return max(0.0, min(1.0, drift))


if __name__ == "__main__":
    nm = NovelMetrics()
    print(nm.entity_coverage_score("Guido van Rossum created Python.", ["Guido van Rossum"]))
    print(nm.answer_diversity_metric(["Python was created in 1991.", "It was created by Guido van Rossum."]))
    print(nm.semantic_drift_detection("When was Python created?", "Guido created Python in 1991."))

