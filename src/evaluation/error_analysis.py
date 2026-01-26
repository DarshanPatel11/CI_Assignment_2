"""
Error Analysis Module

Provides detailed error categorization and analysis for RAG system evaluation.
"""

from typing import List, Dict, Tuple
from collections import defaultdict
import json
from pathlib import Path

import pandas as pd
import numpy as np


class ErrorAnalyzer:
    """Analyzes and categorizes errors in RAG system evaluation."""

    # Error categories
    RETRIEVAL_FAILURE = "retrieval_failure"
    GENERATION_FAILURE = "generation_failure"
    CONTEXT_ISSUE = "context_issue"
    PARTIAL_SUCCESS = "partial_success"

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.analysis_dir = self.data_dir / "evaluation" / "analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

    def categorize_error(
        self,
        mrr_score: float,
        faithfulness_score: float,
        context_precision: float
    ) -> str:
        """
        Categorize error type based on metric scores.

        Categories:
        - retrieval_failure: MRR = 0 (source not found)
        - context_issue: MRR > 0 but context_precision < 0.5
        - generation_failure: Good retrieval but faithfulness < 0.5
        - partial_success: Some issues but mostly working
        """
        if mrr_score == 0:
            return self.RETRIEVAL_FAILURE
        elif context_precision < 0.5:
            return self.CONTEXT_ISSUE
        elif faithfulness_score < 0.5:
            return self.GENERATION_FAILURE
        elif faithfulness_score < 0.7 or context_precision < 0.7:
            return self.PARTIAL_SUCCESS
        return "success"

    def analyze_results(self, detailed_results: List[Dict]) -> Dict:
        """
        Perform comprehensive error analysis.

        Args:
            detailed_results: List of result dicts from evaluation

        Returns:
            Analysis report with categorized errors
        """
        if not detailed_results:
            return {}

        df = pd.DataFrame(detailed_results)

        # Categorize each result
        categories = []
        for _, row in df.iterrows():
            cat = self.categorize_error(
                row.get('mrr', 0),
                row.get('faithfulness', 0),
                row.get('context_precision', 0)
            )
            categories.append(cat)

        df['error_category'] = categories

        # Build analysis report
        analysis = {
            'total_questions': len(df),
            'category_breakdown': {},
            'by_question_type': {},
            'worst_performers': [],
            'recommendations': []
        }

        # Category breakdown
        for cat in df['error_category'].unique():
            cat_df = df[df['error_category'] == cat]
            analysis['category_breakdown'][cat] = {
                'count': len(cat_df),
                'percentage': len(cat_df) / len(df) * 100,
                'examples': cat_df.head(3).to_dict('records')
            }

        # By question type
        for q_type in df['question_type'].unique():
            type_df = df[df['question_type'] == q_type]
            type_cats = type_df['error_category'].value_counts().to_dict()
            analysis['by_question_type'][q_type] = {
                'total': len(type_df),
                'success_rate': len(type_df[type_df['error_category'] == 'success']) / len(type_df) * 100 if len(type_df) > 0 else 0,
                'category_distribution': type_cats
            }

        # Worst performers (lowest combined scores)
        df['combined_score'] = df['mrr'] + df['faithfulness'] + df['context_precision']
        worst = df.nsmallest(5, 'combined_score')[['question', 'error_category', 'mrr', 'faithfulness', 'context_precision']]
        analysis['worst_performers'] = worst.to_dict('records')

        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)

        return analysis

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate improvement recommendations based on analysis."""
        recommendations = []
        breakdown = analysis.get('category_breakdown', {})

        # Check retrieval failures
        ret_fail = breakdown.get(self.RETRIEVAL_FAILURE, {})
        if ret_fail.get('percentage', 0) > 20:
            recommendations.append(
                "High retrieval failure rate detected. Consider: "
                "1) Increasing top-K, 2) Improving embeddings, 3) Tuning BM25 parameters"
            )

        # Check context issues
        ctx_issue = breakdown.get(self.CONTEXT_ISSUE, {})
        if ctx_issue.get('percentage', 0) > 15:
            recommendations.append(
                "Context ranking issues detected. Consider: "
                "1) Adjusting RRF k parameter, 2) Rebalancing dense/sparse weights"
            )

        # Check generation failures
        gen_fail = breakdown.get(self.GENERATION_FAILURE, {})
        if gen_fail.get('percentage', 0) > 15:
            recommendations.append(
                "Generation reliability issues detected. Consider: "
                "1) Using a larger LLM, 2) Improving prompts, 3) Reducing context length"
            )

        # Question type specific
        by_type = analysis.get('by_question_type', {})
        for q_type, stats in by_type.items():
            if stats.get('success_rate', 100) < 50:
                recommendations.append(
                    f"Low success rate for {q_type} questions ({stats['success_rate']:.1f}%). "
                    f"Consider specialized handling for this question type."
                )

        if not recommendations:
            recommendations.append("System performing well across all categories!")

        return recommendations

    def save_analysis(self, analysis: Dict, filename: str = "error_analysis.json"):
        """Save error analysis to file."""
        output_path = self.analysis_dir / filename

        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)

        print(f"Error analysis saved to {output_path}")
        return output_path

    def generate_error_summary(self, analysis: Dict) -> str:
        """Generate human-readable error summary."""
        summary = []
        summary.append("=" * 60)
        summary.append("ERROR ANALYSIS SUMMARY")
        summary.append("=" * 60)

        summary.append(f"\nTotal Questions: {analysis.get('total_questions', 0)}")

        summary.append("\nCATEGORY BREAKDOWN:")
        summary.append("-" * 40)
        for cat, stats in analysis.get('category_breakdown', {}).items():
            summary.append(f"  {cat}: {stats['count']} ({stats['percentage']:.1f}%)")

        summary.append("\nBY QUESTION TYPE:")
        summary.append("-" * 40)
        for q_type, stats in analysis.get('by_question_type', {}).items():
            summary.append(f"  {q_type}: {stats['success_rate']:.1f}% success rate")

        summary.append("\nRECOMMENDATIONS:")
        summary.append("-" * 40)
        for rec in analysis.get('recommendations', []):
            summary.append(f"  • {rec}")

        summary.append("\n" + "=" * 60)

        return "\n".join(summary)


if __name__ == "__main__":
    # Test error analysis
    analyzer = ErrorAnalyzer()

    sample_results = [
        {'question': 'What is Python?', 'question_type': 'factual',
         'mrr': 1.0, 'faithfulness': 0.9, 'context_precision': 0.8},
        {'question': 'Compare Python and Java', 'question_type': 'comparative',
         'mrr': 0.0, 'faithfulness': 0.3, 'context_precision': 0.2},
        {'question': 'Why is ML important?', 'question_type': 'inferential',
         'mrr': 0.5, 'faithfulness': 0.4, 'context_precision': 0.6},
    ]

    analysis = analyzer.analyze_results(sample_results)
    print(analyzer.generate_error_summary(analysis))
