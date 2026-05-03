"""
scripts — ALVRO CLI Entry-Points
==================================
train.py         End-to-end PPO training pipeline
evaluate.py      Load model and evaluate on test split
test_env.py      Validate environment logic (gas gate, OOR fees, lambda)
check_data.py    Data quality diagnostics + GARCH/LVR plot
plot_results.py  Visualise training curves and episode trajectories
run_pipeline.py  One-shot full pipeline (data → train → evaluate → plot)
"""
