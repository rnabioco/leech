"""
Analysis utilities for leech models.

Modules:
- comparison: Compare multiple trained models
- feature_importance: Analyze feature contributions
- sequence_ablation: Test sequence branch impact
- visualization: Plot analysis results
"""

from leech.analysis.comparison import compare_models, load_comparison_results
from leech.analysis.feature_importance import compute_feature_importance
from leech.analysis.sequence_ablation import test_sequence_ablation

__all__ = [
    "compare_models",
    "load_comparison_results",
    "compute_feature_importance",
    "test_sequence_ablation",
]
