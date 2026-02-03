"""
Confidence calibration utilities.
Provides a lightweight confidence estimator and calibration analysis tools.
"""
from typing import List, Dict, Any
import math


class ConfidenceCalibrator:
    """Estimate answer confidence and produce simple calibration stats."""

    def __init__(self, weights: Dict[str, float] = None):
        # weights for combining signals
        self.weights = weights or {
            'retrieval_confidence': 0.5,
            'answer_specificity': 0.2,
            'context_agreement': 0.2,
            'question_complexity': 0.1
        }

    def measure_specificity(self, answer: str) -> float:
        # specificity: longer, more detailed answers considered more specific
        l = len(answer.strip())
        return min(1.0, l / 200.0)

    def assess_complexity(self, question: str) -> float:
        # complexity: more tokens -> higher complexity
        toks = question.split()
        return min(1.0, len(toks) / 20.0)

    def measure_context_consensus(self, retrieval_scores: List[float]) -> float:
        # consensus: if retrieval scores are high and concentrated
        if not retrieval_scores:
            return 0.0
        avg = sum(retrieval_scores) / len(retrieval_scores)
        # concentration: stddev
        mean = avg
        var = sum((x - mean) ** 2 for x in retrieval_scores) / len(retrieval_scores)
        std = math.sqrt(var)
        # map to 0..1 (heuristic)
        return max(0.0, min(1.0, avg / (avg + std + 1e-9)))

    def weighted_confidence(self, factors: Dict[str, float]) -> float:
        w = self.weights
        score = 0.0
        for k, weight in w.items():
            score += weight * factors.get(k, 0.0)
        return max(0.0, min(1.0, score))

    def estimate_confidence(self, question: str, answer: str, retrieval_scores: List[float]) -> float:
        factors = {
            'retrieval_confidence': sum(retrieval_scores) / max(1, len(retrieval_scores)) if retrieval_scores else 0.0,
            'answer_specificity': self.measure_specificity(answer),
            'context_agreement': self.measure_context_consensus(retrieval_scores),
            'question_complexity': 1.0 - self.assess_complexity(question)  # easier questions -> higher confidence
        }
        return self.weighted_confidence(factors)

    def calibration_stats(self, confidences: List[float], correctness: List[int], n_bins: int = 10) -> Dict[str, Any]:
        """Return calibration curve data: for each bin, average confidence and accuracy."""
        assert len(confidences) == len(correctness)
        bins = [0.0 for _ in range(n_bins)]
        counts = [0 for _ in range(n_bins)]
        corrects = [0 for _ in range(n_bins)]

        for c, corr in zip(confidences, correctness):
            idx = min(n_bins - 1, int(c * n_bins))
            bins[idx] += c
            counts[idx] += 1
            corrects[idx] += int(bool(corr))

        calibration = []
        for i in range(n_bins):
            if counts[i] == 0:
                calibration.append({'bin': i, 'avg_confidence': None, 'accuracy': None, 'count': 0})
            else:
                calibration.append({
                    'bin': i,
                    'avg_confidence': bins[i] / counts[i],
                    'accuracy': corrects[i] / counts[i],
                    'count': counts[i]
                })

        # Brier score as a simple aggregate
        brier = sum((c - corr) ** 2 for c, corr in zip(confidences, correctness)) / max(1, len(confidences))

        return {'calibration': calibration, 'brier_score': brier}


if __name__ == "__main__":
    cc = ConfidenceCalibrator()
    print(cc.estimate_confidence("When was Python created?", "1991", [0.8, 0.7, 0.6]))
    print(cc.calibration_stats([0.9, 0.2, 0.5], [1, 0, 1]))

