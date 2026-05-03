"""
ALVRO_v3 — Adaptive LVR Optimizer
===================================
FinAI Contest 2025 — Uniswap v3 Defensive Liquidity Agent

Package structure
-----------------
  env/        ALVROEnv  (Gymnasium environment)
  models/     ALVROAgent (MaskablePPO wrapper)
  data/       ALVRODataProcessor + SyntheticGenerator
  core/       ALVROEvaluator + DeepSeekSentinel
  scripts/    train, evaluate, test_env, check_data, plot_results
"""

__version__ = "3.0.0"
__author__  = "ALVRO Team"
