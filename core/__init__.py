"""
core — ALVRO Evaluation & Risk Signal
========================================
Exposes the evaluator metrics engine and the DeepSeek Sentinel.
"""
from core.evaluator import ALVROEvaluator, EvalMetrics, StepRecord, run_evaluation_episode
from core.sentinel import DeepSeekSentinel, attach_sentinel_to_env

__all__ = [
    "ALVROEvaluator",
    "EvalMetrics",
    "StepRecord",
    "run_evaluation_episode",
    "DeepSeekSentinel",
    "attach_sentinel_to_env",
]
