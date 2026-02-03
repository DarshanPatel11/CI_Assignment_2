"""
Hybrid RAG System - Main Pipeline Orchestrator

Single entry point for:
- Building indices
- Running evaluation
- Starting the UI
"""

import argparse
import json
import sys
from pathlib import Path

from src.data_collection import WikipediaCollector
from src.preprocessing import TextPreprocessor
from src.hybrid_retrieval import HybridRetriever
from src.response_generation import RAGPipeline, ResponseGenerator
from src.evaluation.question_generator import QuestionGenerator
from src.evaluation.evaluation_pipeline import EvaluationPipeline


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
    """Run the evaluation pipeline."""
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

    # Run evaluation
    results = pipeline.run_evaluation(questions, top_k=args.top_k)

    # Run ablation study if requested
    if args.ablation:
        print("\nRunning ablation study...")
        ablation_results = pipeline.run_ablation_study(questions)
        results['ablation'] = ablation_results

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)

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
  Full pipeline:     python main.py --build-index --generate-questions --evaluate
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
