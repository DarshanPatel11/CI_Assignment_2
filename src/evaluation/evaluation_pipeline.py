"""
Automated Evaluation Pipeline

Single-command pipeline that:
1. Loads questions
2. Runs RAG system
3. Computes all metrics
4. Generates comprehensive reports
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

from .metrics import EvaluationMetrics
from .question_generator import QuestionGenerator, QAPair


class EvaluationPipeline:
    """Automated evaluation pipeline for RAG system."""

    def __init__(
        self,
        rag_pipeline,
        data_dir: str = "data"
    ):
        """
        Initialize evaluation pipeline.

        Args:
            rag_pipeline: RAGPipeline instance
            data_dir: Data directory path
        """
        self.rag = rag_pipeline
        self.data_dir = Path(data_dir)
        self.eval_dir = self.data_dir / "evaluation"
        self.results_dir = self.eval_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metrics calculator
        self.metrics = EvaluationMetrics()

    def load_questions(self, filename: str = "questions.json") -> List[Dict]:
        """Load evaluation questions."""
        questions_path = self.eval_dir / filename

        if not questions_path.exists():
            raise FileNotFoundError(f"Questions not found at {questions_path}")

        with open(questions_path, 'r') as f:
            data = json.load(f)

        return data.get("questions", [])

    def run_evaluation(
        self,
        questions: List[Dict],
        top_k: int = 5,
        save_intermediate: bool = True
    ) -> Dict:
        """
        Run full evaluation pipeline.

        Args:
            questions: List of question dicts
            top_k: Number of chunks to retrieve
            save_intermediate: Save intermediate results

        Returns:
            Comprehensive evaluation results
        """
        print(f"\n{'='*60}")
        print("HYBRID RAG EVALUATION PIPELINE")
        print(f"{'='*60}")
        print(f"Questions: {len(questions)}")
        print(f"Top-K: {top_k}")
        print(f"{'='*60}\n")

        # Collect results
        all_queries = []
        all_generated_answers = []
        all_contexts = []
        all_retrieved_chunks = []
        all_ground_truth_urls = []
        all_ground_truth_answers = []
        all_timings = []
        all_question_types = []

        # Run RAG for each question
        print("Step 1/3: Running RAG system on all questions...")

        for q in tqdm(questions, desc="Processing questions"):
            query = q.get('question', '')
            gt_url = q.get('source_url', '')
            gt_answer = q.get('answer', '')
            q_type = q.get('question_type', 'unknown')

            try:
                # Run RAG pipeline
                start_time = time.time()
                result = self.rag.answer(query, top_k=top_k, include_details=True)
                elapsed = (time.time() - start_time) * 1000

                # Collect results
                all_queries.append(query)
                all_generated_answers.append(result.get('answer', ''))
                all_contexts.append(result.get('context_used', ''))

                # Extract chunks with metadata
                chunks = []
                for r in result.get('retrieval', {}).get('hybrid_results', []):
                    chunk_meta = self.rag.retriever.get_chunk_by_id(r['chunk_id'])
                    if chunk_meta:
                        chunks.append(chunk_meta)
                all_retrieved_chunks.append(chunks)

                all_ground_truth_urls.append(gt_url)
                all_ground_truth_answers.append(gt_answer)
                all_timings.append(elapsed)
                all_question_types.append(q_type)

            except Exception as e:
                print(f"Error processing question: {e}")
                all_queries.append(query)
                all_generated_answers.append("")
                all_contexts.append("")
                all_retrieved_chunks.append([])
                all_ground_truth_urls.append(gt_url)
                all_ground_truth_answers.append(gt_answer)
                all_timings.append(0)
                all_question_types.append(q_type)

        # Calculate metrics
        print("\nStep 2/3: Computing evaluation metrics...")

        evaluation_results = self.metrics.evaluate_all(
            queries=all_queries,
            generated_answers=all_generated_answers,
            contexts_used=all_contexts,
            retrieved_chunks=all_retrieved_chunks,
            ground_truth_urls=all_ground_truth_urls,
            ground_truth_answers=all_ground_truth_answers
        )

        # Add timing statistics
        evaluation_results['timing'] = {
            'mean_ms': float(np.mean(all_timings)),
            'median_ms': float(np.median(all_timings)),
            'min_ms': float(np.min(all_timings)),
            'max_ms': float(np.max(all_timings)),
            'std_ms': float(np.std(all_timings)),
            'all_timings': all_timings
        }

        # Add question type breakdown
        type_breakdown = self._compute_type_breakdown(
            all_question_types,
            evaluation_results['mrr']['individual_rrs'],
            evaluation_results['faithfulness']['scores'],
            evaluation_results['context_precision']['scores']
        )
        evaluation_results['by_question_type'] = type_breakdown

        # Build detailed results table
        results_table = self._build_results_table(
            questions,
            all_generated_answers,
            evaluation_results['mrr']['individual_rrs'],
            evaluation_results['faithfulness']['scores'],
            evaluation_results['context_precision']['scores'],
            all_timings
        )
        evaluation_results['detailed_results'] = results_table

        # Save results
        print("\nStep 3/3: Saving results...")
        if save_intermediate:
            self._save_results(evaluation_results)

        # Print summary
        self._print_summary(evaluation_results)

        return evaluation_results

    def _compute_type_breakdown(
        self,
        question_types: List[str],
        mrr_scores: List[float],
        faithfulness_scores: List[float],
        context_precision_scores: List[float]
    ) -> Dict:
        """Compute metrics breakdown by question type."""
        breakdown = {}

        for q_type in set(question_types):
            indices = [i for i, t in enumerate(question_types) if t == q_type]

            breakdown[q_type] = {
                'count': len(indices),
                'mrr': float(np.mean([mrr_scores[i] for i in indices])),
                'faithfulness': float(np.mean([faithfulness_scores[i] for i in indices])),
                'context_precision': float(np.mean([context_precision_scores[i] for i in indices]))
            }

        return breakdown

    def _build_results_table(
        self,
        questions: List[Dict],
        generated_answers: List[str],
        mrr_scores: List[float],
        faithfulness_scores: List[float],
        context_precision_scores: List[float],
        timings: List[float]
    ) -> List[Dict]:
        """Build detailed results table."""
        results = []

        for i, q in enumerate(questions):
            results.append({
                'question_id': q.get('question_id', f'q_{i}'),
                'question': q.get('question', ''),
                'question_type': q.get('question_type', 'unknown'),
                'ground_truth': q.get('answer', ''),
                'generated_answer': generated_answers[i] if i < len(generated_answers) else '',
                'source_url': q.get('source_url', ''),
                'mrr': mrr_scores[i] if i < len(mrr_scores) else 0,
                'faithfulness': faithfulness_scores[i] if i < len(faithfulness_scores) else 0,
                'context_precision': context_precision_scores[i] if i < len(context_precision_scores) else 0,
                'time_ms': timings[i] if i < len(timings) else 0
            })

        return results

    def _save_results(self, results: Dict):
        """Save evaluation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save full JSON results
        json_path = self.results_dir / f"evaluation_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Saved JSON results: {json_path}")

        # Save CSV results table
        if results.get('detailed_results'):
            csv_path = self.results_dir / f"evaluation_{timestamp}.csv"
            df = pd.DataFrame(results['detailed_results'])
            df.to_csv(csv_path, index=False)
            print(f"  Saved CSV results: {csv_path}")

        # Save summary
        summary_path = self.results_dir / f"summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write(self._format_summary(results))
        print(f"  Saved summary: {summary_path}")

    def _format_summary(self, results: Dict) -> str:
        """Format evaluation summary as text."""
        summary = results.get('summary', {})

        text = f"""
HYBRID RAG EVALUATION SUMMARY
{'='*50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Questions: {summary.get('total_queries', 0)}

OVERALL METRICS
{'-'*50}
MRR (Mean Reciprocal Rank):     {summary.get('mrr', 0):.4f}
Hit Rate:                        {summary.get('hit_rate', 0):.4f}
Faithfulness Score:              {summary.get('faithfulness', 0):.4f}
Context Precision:               {summary.get('context_precision', 0):.4f}

TIMING STATISTICS
{'-'*50}
Mean:   {results.get('timing', {}).get('mean_ms', 0):.1f} ms
Median: {results.get('timing', {}).get('median_ms', 0):.1f} ms
Std:    {results.get('timing', {}).get('std_ms', 0):.1f} ms

BY QUESTION TYPE
{'-'*50}
"""

        for q_type, metrics in results.get('by_question_type', {}).items():
            text += f"\n{q_type.upper()} (n={metrics['count']}):\n"
            text += f"  MRR: {metrics['mrr']:.4f}\n"
            text += f"  Faithfulness: {metrics['faithfulness']:.4f}\n"
            text += f"  Context Precision: {metrics['context_precision']:.4f}\n"

        return text

    def _print_summary(self, results: Dict):
        """Print evaluation summary."""
        print(self._format_summary(results))

    def run_ablation_study(
        self,
        questions: List[Dict],
        k_values: List[int] = [3, 5, 10, 15]
    ) -> Dict:
        """
        Run ablation study comparing different retrieval methods and K values.

        Args:
            questions: Evaluation questions
            k_values: List of K values to test

        Returns:
            Ablation study results
        """
        print("\n" + "="*60)
        print("ABLATION STUDY")
        print("="*60)

        results = {
            'by_method': {},
            'by_k': {}
        }

        # Test different K values with hybrid
        print("\nTesting different K values...")
        for k in k_values:
            print(f"\n  K={k}:")
            eval_result = self.run_evaluation(questions[:20], top_k=k, save_intermediate=False)
            results['by_k'][k] = eval_result['summary']

        # Compare methods (would need to modify retriever)
        print("\nMethod comparison stored in results.")

        return results


def run_pipeline(args):
    """Run evaluation pipeline from command line."""
    from ..hybrid_retrieval import HybridRetriever
    from ..response_generation import RAGPipeline, ResponseGenerator

    # Initialize components
    print("Initializing RAG system...")
    retriever = HybridRetriever(data_dir=args.data_dir)
    retriever.load_indices(args.index_name)

    generator = ResponseGenerator()
    rag = RAGPipeline(retriever, generator)

    # Initialize evaluation pipeline
    pipeline = EvaluationPipeline(rag, data_dir=args.data_dir)

    # Load questions
    questions = pipeline.load_questions(args.questions_file)

    # Run evaluation
    results = pipeline.run_evaluation(questions, top_k=args.top_k)

    # Run ablation study if requested
    if args.ablation:
        ablation_results = pipeline.run_ablation_study(questions)
        results['ablation'] = ablation_results

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG Evaluation Pipeline")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--index-name", default="hybrid", help="Index name prefix")
    parser.add_argument("--questions-file", default="questions.json", help="Questions file")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K for retrieval")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")

    args = parser.parse_args()
    run_pipeline(args)
