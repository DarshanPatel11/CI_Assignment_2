"""
Evaluation dashboard stubs.
Provides a minimal, dependency-light dashboard interface. This is a
placeholder that can be extended to Streamlit or a web UI later.
"""
from typing import Any, Dict, List


class EvaluationDashboard:
    """Simple programmatic dashboard interface used for exporting
    evaluation summaries and generating quick reports.
    """

    def __init__(self):
        self.state = {}

    def publish_summary(self, summary: Dict[str, Any], path: str = "eval_summary.json") -> str:
        """Save a JSON summary for later inspection. Returns file path."""
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        return path

    def render_retrieval_visualization(self, retrieval_result: Dict[str, Any]) -> Dict[str, Any]:
        """Return a compact visualization payload describing retrieval.
        This can be consumed by a frontend.
        """
        payload = {
            'query': retrieval_result.get('query'),
            'top_results': [
                {
                    'chunk_id': r[0],
                    'score': float(r[1]),
                    'title': r[2].get('title')
                }
                for r in retrieval_result.get('results', [])
            ]
        }
        return payload

    def show_basic_report(self, results: Dict[str, Any]) -> None:
        """Print a short human-readable report to stdout."""
        print("\n=== Evaluation Report ===")
        for k, v in results.items():
            if isinstance(v, dict) and 'total' in v:
                print(f"{k}: Total={v['total']}")
            elif isinstance(v, list):
                print(f"{k}: {len(v)} items")
            else:
                print(f"{k}: {type(v)}")
        print("========================\n")


if __name__ == "__main__":
    db = EvaluationDashboard()
    print(db.render_retrieval_visualization({'query': 'test', 'results': [('c1', 0.9, {'title':'A'})]}))

