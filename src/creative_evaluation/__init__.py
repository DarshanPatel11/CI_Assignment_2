"""
Creative evaluation package for the Hybrid RAG system.
Exports modular evaluation components: adversarial tests, ablation runner,
error analysis, LLM-based judge (pluggable), confidence calibration,
novel metrics, a simple dashboard stub, and a suite to orchestrate runs.
"""

from .adversarial import AdversarialQuestionGenerator
from .ablation import AblationStudyRunner
from .error_analysis import ErrorAnalyzer
from .llm_judge import LLMJudge
from .confidence import ConfidenceCalibrator
from .novel_metrics import NovelMetrics
from .dashboard import EvaluationDashboard
from .suite import CreativeEvaluationSuite

__all__ = [
    "AdversarialQuestionGenerator",
    "AblationStudyRunner",
    "ErrorAnalyzer",
    "LLMJudge",
    "ConfidenceCalibrator",
    "NovelMetrics",
    "EvaluationDashboard",
    "CreativeEvaluationSuite",
]

