"""
Report Generation Module

Generates comprehensive HTML/PDF reports with:
- Metrics summaries
- Visualizations
- Error analysis
- Ablation study results
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .error_analysis import ErrorAnalyzer


class ReportGenerator:
    """Generates evaluation reports with visualizations."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.results_dir = self.data_dir / "evaluation" / "results"
        self.reports_dir = self.data_dir / "evaluation" / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_metrics_comparison_chart(self, results: Dict) -> go.Figure:
        """Generate metrics comparison bar chart."""
        summary = results.get('summary', {})

        metrics = ['MRR', 'Hit Rate', 'Faithfulness', 'Context Precision']
        values = [
            summary.get('mrr', 0),
            summary.get('hit_rate', 0),
            summary.get('faithfulness', 0),
            summary.get('context_precision', 0)
        ]

        colors = ['#667eea', '#764ba2', '#43a047', '#fb8c00']

        fig = go.Figure(data=[
            go.Bar(
                x=metrics,
                y=values,
                marker_color=colors,
                text=[f'{v:.3f}' for v in values],
                textposition='outside'
            )
        ])

        fig.update_layout(
            title='Evaluation Metrics Overview',
            yaxis_title='Score',
            yaxis_range=[0, 1],
            template='plotly_white',
            height=400
        )

        return fig

    def generate_score_distribution(self, results: Dict) -> go.Figure:
        """Generate score distribution histograms."""
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['MRR Distribution', 'Faithfulness Distribution', 'Context Precision Distribution']
        )

        mrr_scores = results.get('mrr', {}).get('individual_rrs', [])
        faith_scores = results.get('faithfulness', {}).get('scores', [])
        cp_scores = results.get('context_precision', {}).get('scores', [])

        fig.add_trace(
            go.Histogram(x=mrr_scores, nbinsx=20, marker_color='#667eea', name='MRR'),
            row=1, col=1
        )

        fig.add_trace(
            go.Histogram(x=faith_scores, nbinsx=20, marker_color='#43a047', name='Faithfulness'),
            row=1, col=2
        )

        fig.add_trace(
            go.Histogram(x=cp_scores, nbinsx=20, marker_color='#fb8c00', name='Context Precision'),
            row=1, col=3
        )

        fig.update_layout(
            title='Score Distributions',
            showlegend=False,
            template='plotly_white',
            height=300
        )

        return fig

    def generate_question_type_breakdown(self, results: Dict) -> go.Figure:
        """Generate question type breakdown chart."""
        breakdown = results.get('by_question_type', {})

        if not breakdown:
            return None

        types = list(breakdown.keys())
        mrr_values = [breakdown[t]['mrr'] for t in types]
        faith_values = [breakdown[t]['faithfulness'] for t in types]
        cp_values = [breakdown[t]['context_precision'] for t in types]

        fig = go.Figure(data=[
            go.Bar(name='MRR', x=types, y=mrr_values, marker_color='#667eea'),
            go.Bar(name='Faithfulness', x=types, y=faith_values, marker_color='#43a047'),
            go.Bar(name='Context Precision', x=types, y=cp_values, marker_color='#fb8c00')
        ])

        fig.update_layout(
            title='Metrics by Question Type',
            xaxis_title='Question Type',
            yaxis_title='Score',
            barmode='group',
            template='plotly_white',
            height=400
        )

        return fig

    def generate_timing_chart(self, results: Dict) -> go.Figure:
        """Generate timing statistics chart."""
        timing = results.get('timing', {})

        fig = go.Figure()

        # Box plot of all timings
        all_timings = timing.get('all_timings', [])
        if all_timings:
            fig.add_trace(go.Box(
                y=all_timings,
                name='Response Time',
                marker_color='#667eea',
                boxpoints='outliers'
            ))

        # Add mean line
        mean_val = timing.get('mean_ms', 0)
        fig.add_hline(y=mean_val, line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {mean_val:.1f}ms")

        fig.update_layout(
            title='Response Time Distribution',
            yaxis_title='Time (ms)',
            template='plotly_white',
            height=350
        )

        return fig

    def generate_html_report(
        self,
        results: Dict,
        filename: Optional[str] = None
    ) -> Path:
        """Generate comprehensive HTML report."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"report_{timestamp}.html"
        report_path = self.reports_dir / filename

        # Generate all charts
        metrics_chart = self.generate_metrics_comparison_chart(results)
        distribution_chart = self.generate_score_distribution(results)
        type_chart = self.generate_question_type_breakdown(results)
        timing_chart = self.generate_timing_chart(results)

        # Convert charts to HTML
        charts_html = ""
        charts_html += metrics_chart.to_html(full_html=False, include_plotlyjs='cdn')
        charts_html += distribution_chart.to_html(full_html=False, include_plotlyjs=False)
        if type_chart:
            charts_html += type_chart.to_html(full_html=False, include_plotlyjs=False)
        charts_html += timing_chart.to_html(full_html=False, include_plotlyjs=False)

        # Build HTML
        summary = results.get('summary', {})
        timing = results.get('timing', {})

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hybrid RAG Evaluation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }}
        .header h1 {{ margin: 0; }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .metric-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{ color: #666; }}
        .section {{
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 0.5rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{ background: #f8f9fa; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Hybrid RAG Evaluation Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Total Questions Evaluated: {summary.get('total_queries', 0)}</p>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value">{summary.get('mrr', 0):.4f}</div>
            <div class="metric-label">MRR</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{summary.get('hit_rate', 0):.1%}</div>
            <div class="metric-label">Hit Rate</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{summary.get('faithfulness', 0):.4f}</div>
            <div class="metric-label">Faithfulness</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{summary.get('context_precision', 0):.4f}</div>
            <div class="metric-label">Context Precision</div>
        </div>
    </div>

    <div class="section">
        <h2>📊 Metrics Overview</h2>
        {charts_html}
    </div>

    <div class="section">
        <h2>⏱️ Performance Statistics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Mean Response Time</td><td>{timing.get('mean_ms', 0):.1f} ms</td></tr>
            <tr><td>Median Response Time</td><td>{timing.get('median_ms', 0):.1f} ms</td></tr>
            <tr><td>Min Response Time</td><td>{timing.get('min_ms', 0):.1f} ms</td></tr>
            <tr><td>Max Response Time</td><td>{timing.get('max_ms', 0):.1f} ms</td></tr>
            <tr><td>Std Deviation</td><td>{timing.get('std_ms', 0):.1f} ms</td></tr>
        </table>
    </div>

    <div class="section">
        <h2>📋 Metric Definitions</h2>
        <h3>MRR (Mean Reciprocal Rank) - Mandatory</h3>
        <p>Measures how quickly the system identifies the correct source document at URL level.
        MRR = average of 1/rank across all questions.</p>

        <h3>Faithfulness Score (Custom Metric 1)</h3>
        <p><strong>Justification:</strong> Measures if generated answers are grounded in retrieved context,
        detecting hallucinations. Unlike ROUGE, this evaluates factual consistency.</p>
        <p><strong>Calculation:</strong> Extract claims from answer, verify each against context.
        Score = supported_claims / total_claims.</p>
        <p><strong>Interpretation:</strong> 1.0 = fully grounded, &lt;0.7 = potential reliability issues.</p>

        <h3>Context Precision (Custom Metric 2)</h3>
        <p><strong>Justification:</strong> Evaluates retrieval ranking quality - whether relevant
        chunks appear higher in results.</p>
        <p><strong>Calculation:</strong> Weighted precision at each rank position, emphasizing top results.</p>
        <p><strong>Interpretation:</strong> 1.0 = perfect ranking, lower scores indicate ranking issues.</p>
    </div>
</body>
</html>
"""

        with open(report_path, 'w') as f:
            f.write(html_content)

        print(f"HTML report saved: {report_path}")
        return report_path

    def generate_results_dataframe(self, results: Dict) -> pd.DataFrame:
        """Convert detailed results to DataFrame."""
        detailed = results.get('detailed_results', [])
        return pd.DataFrame(detailed)

    def analyze_errors(self, results: Dict) -> Dict:
        """Perform error analysis on results."""
        detailed = results.get('detailed_results', [])

        if not detailed:
            return {}

        df = pd.DataFrame(detailed)

        # Categorize failures
        retrieval_failures = df[df['mrr'] == 0]
        low_faithfulness = df[df['faithfulness'] < 0.5]
        low_precision = df[df['context_precision'] < 0.5]

        analysis = {
            'total_questions': len(df),
            'retrieval_failures': {
                'count': len(retrieval_failures),
                'percentage': len(retrieval_failures) / len(df) * 100 if len(df) > 0 else 0,
                'examples': retrieval_failures.head(5).to_dict('records')
            },
            'low_faithfulness': {
                'count': len(low_faithfulness),
                'percentage': len(low_faithfulness) / len(df) * 100 if len(df) > 0 else 0,
                'examples': low_faithfulness.head(5).to_dict('records')
            },
            'low_precision': {
                'count': len(low_precision),
                'percentage': len(low_precision) / len(df) * 100 if len(df) > 0 else 0,
                'examples': low_precision.head(5).to_dict('records')
            }
        }

        # By question type
        type_analysis = {}
        for q_type in df['question_type'].unique():
            type_df = df[df['question_type'] == q_type]
            type_analysis[q_type] = {
                'count': len(type_df),
                'retrieval_fail_rate': len(type_df[type_df['mrr'] == 0]) / len(type_df) * 100 if len(type_df) > 0 else 0,
                'low_faith_rate': len(type_df[type_df['faithfulness'] < 0.5]) / len(type_df) * 100 if len(type_df) > 0 else 0
            }

        analysis['by_type'] = type_analysis

        return analysis


if __name__ == "__main__":
    # Test report generation
    reporter = ReportGenerator()

    # Sample results for testing
    sample_results = {
        'summary': {
            'mrr': 0.72,
            'hit_rate': 0.85,
            'faithfulness': 0.78,
            'context_precision': 0.68,
            'total_queries': 100
        },
        'mrr': {'individual_rrs': [1.0, 0.5, 0.33, 1.0, 0.0] * 20},
        'faithfulness': {'scores': [0.9, 0.7, 0.6, 0.8, 0.5] * 20},
        'context_precision': {'scores': [0.8, 0.6, 0.7, 0.5, 0.9] * 20},
        'timing': {
            'mean_ms': 250,
            'median_ms': 230,
            'min_ms': 150,
            'max_ms': 450,
            'std_ms': 50,
            'all_timings': [200, 250, 300, 280, 220] * 20
        },
        'by_question_type': {
            'factual': {'mrr': 0.8, 'faithfulness': 0.85, 'context_precision': 0.75, 'count': 40},
            'comparative': {'mrr': 0.65, 'faithfulness': 0.7, 'context_precision': 0.6, 'count': 20},
            'inferential': {'mrr': 0.7, 'faithfulness': 0.75, 'context_precision': 0.65, 'count': 25},
            'multi-hop': {'mrr': 0.55, 'faithfulness': 0.65, 'context_precision': 0.55, 'count': 15}
        },
        'detailed_results': [
            {'question_id': 'q1', 'question': 'What is Python?', 'question_type': 'factual',
             'mrr': 1.0, 'faithfulness': 0.9, 'context_precision': 0.8, 'time_ms': 250}
        ] * 20
    }

    report_path = reporter.generate_html_report(sample_results)
    print(f"Report generated: {report_path}")
