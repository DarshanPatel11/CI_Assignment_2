"""
Visualization Module

Creates comprehensive visualizations for RAG evaluation results.
"""

from typing import Dict, List, Optional
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


class Visualizer:
    """Creates evaluation visualizations."""

    def __init__(self, output_dir: str = "data/evaluation/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color scheme
        self.colors = {
            'primary': '#667eea',
            'secondary': '#764ba2',
            'success': '#43a047',
            'warning': '#fb8c00',
            'error': '#e53935'
        }

    def create_metrics_radar(self, metrics: Dict) -> go.Figure:
        """Create radar chart for metrics comparison."""
        categories = ['MRR', 'Hit Rate', 'Faithfulness', 'Context Precision']
        values = [
            metrics.get('mrr', 0),
            metrics.get('hit_rate', 0),
            metrics.get('faithfulness', 0),
            metrics.get('context_precision', 0)
        ]
        # Close the radar
        categories.append(categories[0])
        values.append(values[0])

        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor='rgba(102, 126, 234, 0.3)',
            line_color=self.colors['primary']
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title='Metrics Radar Chart',
            template='plotly_white'
        )

        return fig

    def create_question_type_heatmap(self, breakdown: Dict) -> go.Figure:
        """Create heatmap of metrics by question type."""
        types = list(breakdown.keys())
        metrics = ['mrr', 'faithfulness', 'context_precision']

        z = [[breakdown[t].get(m, 0) for m in metrics] for t in types]

        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=['MRR', 'Faithfulness', 'Context Precision'],
            y=types,
            colorscale='Viridis',
            text=[[f'{v:.2f}' for v in row] for row in z],
            texttemplate='%{text}',
            textfont={'size': 14}
        ))

        fig.update_layout(
            title='Metrics by Question Type',
            xaxis_title='Metric',
            yaxis_title='Question Type',
            height=400
        )

        return fig

    def create_error_distribution_pie(self, breakdown: Dict) -> go.Figure:
        """Create pie chart of error categories."""
        labels = list(breakdown.keys())
        values = [breakdown[cat].get('count', 0) for cat in labels]

        colors = [
            self.colors['success'] if 'success' in cat.lower()
            else self.colors['error'] if 'failure' in cat.lower()
            else self.colors['warning']
            for cat in labels
        ]

        fig = go.Figure(data=go.Pie(
            labels=labels,
            values=values,
            marker_colors=colors,
            textinfo='percent+label'
        ))

        fig.update_layout(title='Error Category Distribution')

        return fig

    def create_timing_histogram(self, timings: List[float]) -> go.Figure:
        """Create histogram of response times."""
        fig = go.Figure(data=go.Histogram(
            x=timings,
            nbinsx=30,
            marker_color=self.colors['primary'],
            opacity=0.7
        ))

        # Add mean line
        mean_val = np.mean(timings)
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_val:.1f}ms"
        )

        fig.update_layout(
            title='Response Time Distribution',
            xaxis_title='Time (ms)',
            yaxis_title='Count',
            template='plotly_white'
        )

        return fig

    def create_ablation_comparison(self, ablation_results: Dict) -> go.Figure:
        """Create bar chart comparing ablation study results."""
        by_k = ablation_results.get('by_k', {})

        if not by_k:
            return None

        k_values = list(by_k.keys())
        mrr_values = [by_k[k].get('mrr', 0) for k in k_values]

        fig = go.Figure(data=go.Bar(
            x=[f'K={k}' for k in k_values],
            y=mrr_values,
            marker_color=self.colors['primary'],
            text=[f'{v:.3f}' for v in mrr_values],
            textposition='outside'
        ))

        fig.update_layout(
            title='Ablation Study: MRR by Top-K',
            xaxis_title='Top-K Value',
            yaxis_title='MRR',
            template='plotly_white'
        )

        return fig

    def create_score_correlation(self, df: pd.DataFrame) -> go.Figure:
        """Create scatter plot showing metric correlations."""
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['MRR vs Faithfulness', 'MRR vs Context Precision', 'Faithfulness vs Context Precision']
        )

        fig.add_trace(
            go.Scatter(x=df['mrr'], y=df['faithfulness'], mode='markers',
                      marker=dict(color=self.colors['primary'], opacity=0.6)),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=df['mrr'], y=df['context_precision'], mode='markers',
                      marker=dict(color=self.colors['secondary'], opacity=0.6)),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(x=df['faithfulness'], y=df['context_precision'], mode='markers',
                      marker=dict(color=self.colors['success'], opacity=0.6)),
            row=1, col=3
        )

        fig.update_layout(
            title='Metric Correlations',
            showlegend=False,
            height=350
        )

        return fig

    def save_all_visualizations(self, results: Dict, prefix: str = "viz") -> List[Path]:
        """Save all visualizations as HTML files."""
        saved_paths = []

        # Metrics radar
        summary = results.get('summary', {})
        if summary:
            fig = self.create_metrics_radar(summary)
            path = self.output_dir / f"{prefix}_radar.html"
            fig.write_html(str(path))
            saved_paths.append(path)

        # Question type heatmap
        breakdown = results.get('by_question_type', {})
        if breakdown:
            fig = self.create_question_type_heatmap(breakdown)
            path = self.output_dir / f"{prefix}_heatmap.html"
            fig.write_html(str(path))
            saved_paths.append(path)

        # Timing histogram
        timings = results.get('timing', {}).get('all_timings', [])
        if timings:
            fig = self.create_timing_histogram(timings)
            path = self.output_dir / f"{prefix}_timing.html"
            fig.write_html(str(path))
            saved_paths.append(path)

        # Score correlations
        detailed = results.get('detailed_results', [])
        if detailed:
            df = pd.DataFrame(detailed)
            if all(col in df.columns for col in ['mrr', 'faithfulness', 'context_precision']):
                fig = self.create_score_correlation(df)
                path = self.output_dir / f"{prefix}_correlations.html"
                fig.write_html(str(path))
                saved_paths.append(path)

        print(f"Saved {len(saved_paths)} visualizations to {self.output_dir}")
        return saved_paths


if __name__ == "__main__":
    # Test visualizations
    viz = Visualizer()

    sample_results = {
        'summary': {
            'mrr': 0.72,
            'hit_rate': 0.85,
            'faithfulness': 0.78,
            'context_precision': 0.68
        },
        'by_question_type': {
            'factual': {'mrr': 0.8, 'faithfulness': 0.85, 'context_precision': 0.75},
            'comparative': {'mrr': 0.65, 'faithfulness': 0.7, 'context_precision': 0.6},
            'inferential': {'mrr': 0.7, 'faithfulness': 0.75, 'context_precision': 0.65},
            'multi-hop': {'mrr': 0.55, 'faithfulness': 0.65, 'context_precision': 0.55}
        },
        'timing': {
            'all_timings': [200, 250, 300, 280, 220, 350, 180, 240] * 10
        },
        'detailed_results': [
            {'mrr': 1.0, 'faithfulness': 0.9, 'context_precision': 0.8},
            {'mrr': 0.5, 'faithfulness': 0.7, 'context_precision': 0.6},
            {'mrr': 0.0, 'faithfulness': 0.4, 'context_precision': 0.3}
        ] * 20
    }

    paths = viz.save_all_visualizations(sample_results)
    print(f"Created visualizations: {paths}")
