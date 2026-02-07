"""
CreativeEvaluationSuite: orchestrates the creative evaluation components
and provides a single entrypoint to run adversarial tests, ablations,
error analysis, LLM judging, confidence calibration, and novel metrics.
"""
from typing import List, Dict, Any, Optional

from .adversarial import AdversarialQuestionGenerator
from .ablation import AblationStudyRunner
from .error_analysis import ErrorAnalyzer
from .llm_judge import LLMJudge
from .confidence import ConfidenceCalibrator
from .novel_metrics import NovelMetrics
from .dashboard import EvaluationDashboard


class CreativeEvaluationSuite:
    def __init__(self, rag_pipeline=None, retriever=None):
        self.adversarial = AdversarialQuestionGenerator()
        self.ablation = AblationStudyRunner(retriever=retriever, rag_pipeline=rag_pipeline)
        self.error_analyzer = ErrorAnalyzer()
        self.judge = LLMJudge()
        self.confidence = ConfidenceCalibrator()
        self.metrics = NovelMetrics()
        self.dashboard = EvaluationDashboard()

    def run_adversarial_tests(self, question: str, rag_pipeline, top_k: int = 5) -> Dict[str, Any]:
        variants = self.adversarial.generate_adversarial_set(question)
        results = []
        for v in variants:
            q = v['q']
            resp = rag_pipeline.answer(q, top_k=top_k, include_details=True)
            results.append({
                'type': v['type'],
                'question': q,
                'answer': resp.get('answer', ''),
                'retrieval': resp.get('retrieval', {}),
                'context': resp.get('context_used', '')
            })
        return {'original': question, 'variants': results}

    def run_full_evaluation(self, qa_pairs: List[Dict[str, Any]], rag_pipeline, top_k: int = 5) -> Dict[str, Any]:
        # Run end-to-end evaluation: ablations, judging, error analysis, confidence
        # qa_pairs: list of {'question':..., 'answer':..., 'topic': optional}
        results = {'responses': []}
        for item in qa_pairs:
            q = item.get('question') or item.get('q')
            gold = item.get('answer') or item.get('gold')
            resp = rag_pipeline.answer(q, top_k=top_k, include_details=True)

            pred = resp.get('answer', '')
            context = resp.get('context_used', '')
            retrieval_scores = []
            for r in resp.get('retrieval', {}).get('hybrid_results', []):
                retrieval_scores.append(r.get('score', 0))

            conf = self.confidence.estimate_confidence(q, pred, retrieval_scores)
            judge_scores = self.judge.judge(q, pred, context)
            is_em = int(pred.strip().lower() == (gold or '').strip().lower())

            results['responses'].append({
                'question': q,
                'gold': gold,
                'pred': pred,
                'exact_match': is_em,
                'confidence': conf,
                'judge': judge_scores,
                'retrieval': resp.get('retrieval', {}),
                'context': context,
                'topic': item.get('topic', 'unknown')
            })

        # Ablation (retrieval-only)
        try:
            ablation_res = self.ablation.run_retrieval_ablations([p['question'] for p in qa_pairs])
            results['ablation'] = ablation_res
        except Exception as e:
            results['ablation_error'] = str(e)

        # Error analysis
        ea = self.error_analyzer.analyze_results([
            {'question': r['question'], 'gold': r['gold'], 'pred': r['pred'] if 'pred' in r else r['pred'] if 'pred' in r else r['pred']}
            for r in results['responses']
        ])
        results['error_analysis'] = ea

        # Novel metrics
        # Hallucination flagging (simple heuristic)
        halluc_results = []
        for r in results['responses']:
            is_hall = False
            pred_tokens = set((r.get('pred') or r.get('pred', '')).lower().split())
            context_tokens = set((r.get('context') or '').lower().split())
            if len(pred_tokens - context_tokens) / max(1, len(pred_tokens)) > 0.3:
                is_hall = True
            halluc_results.append({
                'topic': r.get('topic', 'unknown'),
                'is_hallucination': is_hall
            })

        results['hallucination_by_topic'] = self.metrics.hallucination_rate_by_topic(halluc_results)

        # Save summary
        self.dashboard.publish_summary(results, path="creative_evaluation_summary.json")
        return results


if __name__ == "__main__":
    print("CreativeEvaluationSuite ready")

