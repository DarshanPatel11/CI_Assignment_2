"""
Hybrid RAG System - Main Pipeline Orchestrator

Single entry point for:
- Building indices
- Running evaluation (standard + innovative)
- Running comprehensive creative evaluation suite
- Starting the UI
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

from src.data_collection import WikipediaCollector
from src.preprocessing import TextPreprocessor
from src.hybrid_retrieval import HybridRetriever
from src.response_generation import RAGPipeline, ResponseGenerator
from src.evaluation.question_generator import QuestionGenerator
from src.evaluation.evaluation_pipeline import EvaluationPipeline
from src.creative_evaluation import (
    CreativeEvaluationSuite,
    AdversarialQuestionGenerator,
    AblationStudyRunner,
    ErrorAnalyzer,
    LLMJudge,
    ConfidenceCalibrator,
    NovelMetrics
)


def build_index(args):
    """Build the complete RAG index."""
    print("\n" + "="*60)
    print("BUILDING HYBRID RAG INDEX")
    print("="*60)

    data_dir = args.data_dir

    # Step 1: Collect Wikipedia articles
    print("\n[1/4] Collecting Wikipedia articles...")
    collector = WikipediaCollector(data_dir=data_dir)

    # Generate fixed URLs if needed
    if args.generate_fixed_urls or not (Path(data_dir) / "fixed_urls.json").exists():
        collector.generate_fixed_urls(count=args.fixed_count, force=args.generate_fixed_urls)

    # Collect articles
    articles = collector.collect_all_articles(
        fixed_count=args.fixed_count,
        random_count=args.random_count
    )
    collector.save_corpus(articles)

    # Step 2: Preprocess and chunk
    print("\n[2/4] Preprocessing and chunking articles...")
    preprocessor = TextPreprocessor(data_dir=data_dir)

    # Convert articles to dict format
    article_dicts = [a.to_dict() for a in articles]
    chunks = preprocessor.process_articles(article_dicts)
    preprocessor.save_chunks(chunks)

    # Step 3: Build hybrid indices
    print("\n[3/4] Building hybrid indices...")
    chunk_dicts = [c.to_dict() for c in chunks]

    retriever = HybridRetriever(data_dir=data_dir)
    retriever.build_indices(chunk_dicts)
    retriever.save_indices(args.index_name)

    # Step 4: Generate evaluation questions
    if args.generate_questions:
        print("\n[4/4] Generating evaluation questions...")
        question_gen = QuestionGenerator(data_dir=data_dir)
        qa_pairs = question_gen.generate_dataset(chunk_dicts, total_questions=args.num_questions)
        question_gen.save_dataset(qa_pairs)
    else:
        print("\n[4/4] Skipping question generation (use --generate-questions to enable)")

    print("\n" + "="*60)
    print("INDEX BUILT SUCCESSFULLY")
    print("="*60)
    print(f"  Articles: {len(articles)}")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Index: {args.index_name}")
    print("="*60)

def generate_fixed_urls(args):
    """Generate fixed URLs for data collection."""
    print("\n" + "="*60)
    print("GENERATING FIXED URLS")
    print("="*60)

    data_dir = args.data_dir

    collector = WikipediaCollector(data_dir=data_dir)
    collector.generate_fixed_urls(count=args.fixed_count, force=True)

    print("\n" + "="*60)
    print("FIXED URLS GENERATED SUCCESSFULLY")
    print("="*60)


def run_evaluation(args):
    """Run the evaluation pipeline with optional innovative evaluation."""
    print("\n" + "="*60)
    print("RUNNING EVALUATION PIPELINE")
    print("="*60)

    data_dir = args.data_dir

    # Load RAG system
    print("\nLoading RAG system...")
    retriever = HybridRetriever(data_dir=data_dir)
    retriever.load_indices(args.index_name)

    generator = ResponseGenerator()
    rag = RAGPipeline(retriever, generator)

    # Initialize evaluation pipeline
    pipeline = EvaluationPipeline(rag, data_dir=data_dir)

    # Load questions
    questions = pipeline.load_questions(args.questions_file)

    if args.num_questions:
        questions = questions[:args.num_questions]

    # Run standard evaluation
    results = pipeline.run_evaluation(questions, top_k=args.top_k)

    # Run ablation study if requested
    if args.ablation:
        print("\nRunning ablation study...")
        ablation_results = pipeline.run_ablation_study(questions)
        results['ablation'] = ablation_results

    # Run innovative evaluation if requested
    if args.innovative:
        print("\n" + "="*60)
        print("RUNNING INNOVATIVE EVALUATION")
        print("="*60)
        innovative_results = run_innovative_evaluation(rag, retriever, questions, args)
        results['innovative'] = innovative_results

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)

    return results


def run_innovative_evaluation(rag, retriever, questions, args):
    """Run innovative evaluation components."""
    results = {}
    data_dir = Path(args.data_dir)
    results_dir = data_dir / "evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize components
    print("\nInitializing innovative evaluation components...")
    adversarial = AdversarialQuestionGenerator()
    ablation = AblationStudyRunner(retriever=retriever, rag_pipeline=rag)
    error_analyzer = ErrorAnalyzer()
    judge = LLMJudge()
    confidence = ConfidenceCalibrator()
    novel_metrics = NovelMetrics()

    # 1. Adversarial Testing
    print("\n[1/5] Running Adversarial Testing...")
    adversarial_results = []
    sample_questions = questions[:10]  # Test on subset
    for q in sample_questions:
        original_q = q.get('question', '')
        variants = adversarial.generate_adversarial_set(original_q)
        variant_results = []
        for v in variants:
            try:
                resp = rag.answer(v['q'], top_k=args.top_k, include_details=True)
                variant_results.append({
                    'type': v['type'],
                    'question': v['q'],
                    'answer': resp.get('answer', ''),
                    'has_response': bool(resp.get('answer', '').strip())
                })
            except Exception as e:
                variant_results.append({'type': v['type'], 'error': str(e)})
        adversarial_results.append({
            'original': original_q,
            'variants': variant_results
        })
    results['adversarial'] = adversarial_results
    print(f"   Tested {len(sample_questions)} questions with adversarial variants")

    # 2. Comprehensive Ablation Study
    print("\n[2/5] Running Comprehensive Ablation Study...")
    queries = [q.get('question', '') for q in questions[:20]]
    ablation_results = ablation.run_retrieval_ablations(
        queries=queries,
        ks=[3, 5, 10],
        rrf_ks=[20, 60, 100]
    )
    results['ablation_study'] = {
        'summary': 'Compared dense-only, sparse-only, and hybrid retrieval',
        'k_values_tested': [3, 5, 10],
        'rrf_k_values_tested': [20, 60, 100],
        'num_queries': len(queries)
    }
    print(f"   Completed ablation across K=[3,5,10] and RRF_k=[20,60,100]")

    # 3. Error Analysis
    print("\n[3/5] Running Error Analysis...")
    error_analysis_data = []
    for q in questions[:30]:
        try:
            resp = rag.answer(q.get('question', ''), top_k=args.top_k, include_details=True)
            error_analysis_data.append({
                'question': q.get('question', ''),
                'gold': q.get('answer', ''),
                'pred': resp.get('answer', ''),
                'context': resp.get('context_used', '')
            })
        except:
            pass
    error_summary = error_analyzer.analyze_results(error_analysis_data)
    results['error_analysis'] = error_summary
    print(f"   Analyzed {len(error_analysis_data)} responses")
    print(f"   Error distribution: {error_summary.get('distribution', {})}")

    # 4. LLM-as-Judge Evaluation
    print("\n[4/5] Running LLM-as-Judge Evaluation...")
    judge_results = []
    for item in error_analysis_data[:20]:
        scores = judge.judge(
            item['question'],
            item['pred'],
            item['context']
        )
        judge_results.append({
            'question': item['question'],
            'scores': scores
        })
    results['llm_judge'] = {
        'results': judge_results,
        'avg_factual_accuracy': sum(r['scores']['factual_accuracy'] for r in judge_results) / max(1, len(judge_results)),
        'avg_completeness': sum(r['scores']['completeness'] for r in judge_results) / max(1, len(judge_results)),
        'avg_relevance': sum(r['scores']['relevance'] for r in judge_results) / max(1, len(judge_results)),
        'avg_coherence': sum(r['scores']['coherence'] for r in judge_results) / max(1, len(judge_results)),
        'avg_groundedness': sum(r['scores']['groundedness'] for r in judge_results) / max(1, len(judge_results))
    }
    print(f"   Judged {len(judge_results)} responses")

    # 5. Confidence Calibration
    print("\n[5/5] Running Confidence Calibration...")
    confidences = []
    correctness = []
    for item in error_analysis_data[:20]:
        # Get retrieval scores from a fresh query
        try:
            resp = rag.answer(item['question'], top_k=args.top_k, include_details=True)
            retrieval_scores = [r.get('score', 0) for r in resp.get('retrieval', {}).get('hybrid_results', [])]
            conf = confidence.estimate_confidence(item['question'], item['pred'], retrieval_scores)
            confidences.append(conf)
            # Simple correctness: check if gold appears in prediction
            gold_lower = (item['gold'] or '').lower()
            pred_lower = (item['pred'] or '').lower()
            is_correct = 1 if gold_lower and gold_lower in pred_lower else 0
            correctness.append(is_correct)
        except:
            pass
    if confidences:
        calibration_stats = confidence.calibration_stats(confidences, correctness)
        results['confidence_calibration'] = {
            'brier_score': calibration_stats['brier_score'],
            'calibration_curve': calibration_stats['calibration'],
            'avg_confidence': sum(confidences) / len(confidences),
            'avg_correctness': sum(correctness) / len(correctness)
        }
        print(f"   Brier Score: {calibration_stats['brier_score']:.4f}")

    # 6. Novel Metrics
    print("\n[6/6] Computing Novel Metrics...")
    answers = [item['pred'] for item in error_analysis_data if item['pred']]
    diversity = novel_metrics.answer_diversity_metric(answers)
    results['novel_metrics'] = {
        'answer_diversity': diversity,
        'num_answers_analyzed': len(answers)
    }
    print(f"   Answer Diversity: {diversity:.4f}")

    # Save innovative evaluation results
    innovative_path = results_dir / f"innovative_evaluation_{timestamp}.json"
    with open(innovative_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n   Saved innovative evaluation to: {innovative_path}")

    return results


def show_status(args):
    """Show system status."""
    print("\n" + "="*60)
    print("HYBRID RAG SYSTEM STATUS")
    print("="*60)

    data_dir = Path(args.data_dir)

    # Check fixed URLs
    fixed_urls_path = data_dir / "fixed_urls.json"
    if fixed_urls_path.exists():
        with open(fixed_urls_path) as f:
            data = json.load(f)
        print(f"✅ Fixed URLs: {data.get('count', 0)} URLs")
    else:
        print("❌ Fixed URLs: Not generated")

    # Check corpus
    corpus_path = data_dir / "corpus" / "chunks.json"
    if corpus_path.exists():
        with open(corpus_path) as f:
            data = json.load(f)
        print(f"✅ Corpus: {data.get('total_chunks', 0)} chunks from {data.get('total_articles', 0)} articles")
    else:
        print("❌ Corpus: Not generated")

    # Check indices
    indices_dir = data_dir / "indices"
    dense_index = indices_dir / f"{args.index_name}_dense.faiss"
    sparse_index = indices_dir / f"{args.index_name}_sparse.pkl"

    if dense_index.exists() and sparse_index.exists():
        print(f"✅ Indices: Built ({args.index_name})")
    else:
        print("❌ Indices: Not built")

    # Check questions
    questions_path = data_dir / "evaluation" / "questions.json"
    if questions_path.exists():
        with open(questions_path) as f:
            data = json.load(f)
        print(f"✅ Questions: {data.get('total_questions', 0)} Q&A pairs")
    else:
        print("❌ Questions: Not generated")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid RAG System - Main Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Build index:       python main.py --build-index
  Run evaluation:    python main.py --evaluate
  Full evaluation:   python main.py --evaluate --innovative --ablation
  Full pipeline:     python main.py --build-index --generate-questions --evaluate --innovative
  Check status:      python main.py --status
        """
    )

    # Actions
    parser.add_argument("--build-index", action="store_true", help="Build RAG indices")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation pipeline")
    parser.add_argument("--status", action="store_true", help="Show system status")

    # Data collection options
    parser.add_argument("--fixed-count", type=int, default=200, help="Number of fixed URLs")
    parser.add_argument("--random-count", type=int, default=300, help="Number of random URLs")
    parser.add_argument("--generate-fixed-urls", action="store_true", help="Force regenerate fixed URLs")

    # Index options
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--index-name", default="hybrid", help="Index name prefix")

    # Question generation options
    parser.add_argument("--generate-questions", action="store_true", help="Generate evaluation questions")
    parser.add_argument("--num-questions", type=int, default=100, help="Number of questions to generate/evaluate")

    # Evaluation options
    parser.add_argument("--questions-file", default="questions.json", help="Questions file name")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K for retrieval")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--innovative", action="store_true", help="Run innovative/creative evaluation (adversarial, LLM-judge, confidence calibration)")

    args = parser.parse_args()

    # Execute requested actions
    if args.status:
        show_status(args)

    if args.build_index:
        build_index(args)

    if args.evaluate:
        run_evaluation(args)

    if args.generate_fixed_urls:
        generate_fixed_urls(args)


    if not any([args.status, args.build_index, args.evaluate]):
        parser.print_help()


if __name__ == "__main__":
    main()
