"""
ALVRO_v3 — End-to-End Pipeline Runner
=======================================
scripts/run_pipeline.py

Executes the complete ALVRO workflow in one command:

  Stage 1 — DATA      Generate or download processed market data
  Stage 2 — VALIDATE  Check data quality (NaN audit, LVR validity)
  Stage 3 — TEST-ENV  Smoke-test the Gymnasium environment
  Stage 4 — TRAIN     Run MaskablePPO training
  Stage 5 — EVALUATE  Evaluate on hold-out split, print report
  Stage 6 — PLOT      Generate the 6-panel results dashboard

Each stage can be skipped with --skip-<stage>.

Usage
-----
    # Full pipeline (synthetic data, no GPU):
    python -m scripts.run_pipeline --use-synthetic

    # Skip data generation (parquet already exists):
    python -m scripts.run_pipeline --skip-data

    # Quick smoke-run (100k steps, synthetic data):
    python -m scripts.run_pipeline --use-synthetic --timesteps 100000

    # With real data + DeepSeek LLM sentinel:
    python -m scripts.run_pipeline \\
        --sentinel-mode llm \\
        --timesteps 2000000 \\
        --log-dir logs/run_deepseek
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("run_pipeline")

PYTHON = sys.executable   # re-use the same interpreter


# ─── Stage runner ────────────────────────────────────────────────

def run_stage(name: str, cmd: list[str], skip: bool) -> bool:
    """
    Run a pipeline stage as a subprocess.

    Returns True on success, False on failure (does NOT raise so later
    stages can still be attempted).
    """
    bar = "─" * 60
    if skip:
        log.info("%s\n  SKIPPING: %s\n%s", bar, name, bar)
        return True

    log.info("%s\n  RUNNING: %s\n  CMD: %s\n%s", bar, name, " ".join(cmd), bar)
    result = subprocess.run(cmd, cwd=str(_ROOT))
    if result.returncode == 0:
        log.info("  ✔ %s — SUCCESS", name)
        return True
    else:
        log.error("  ✖ %s — FAILED (exit code %d)", name, result.returncode)
        return False


# ─── Argument parsing ────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ALVRO_v3 end-to-end pipeline runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Stage control ─────────────────────────────────────────────
    p.add_argument("--skip-data",     action="store_true", help="Skip Stage 1: data generation")
    p.add_argument("--skip-validate", action="store_true", help="Skip Stage 2: data quality check")
    p.add_argument("--skip-test-env", action="store_true", help="Skip Stage 3: env smoke-test")
    p.add_argument("--skip-train",    action="store_true", help="Skip Stage 4: training")
    p.add_argument("--skip-evaluate", action="store_true", help="Skip Stage 5: evaluation")
    p.add_argument("--skip-plot",     action="store_true", help="Skip Stage 6: plotting")

    # ── Data ──────────────────────────────────────────────────────
    p.add_argument("--use-synthetic", action="store_true",
                   help="Use synthetic data generator instead of cloning the real repo")
    p.add_argument("--parquet", type=str,
                   default=str(_ROOT / "data" / "processed_market.parquet"))
    p.add_argument("--n-hours", type=int, default=26_280,
                   help="Hours of synthetic data to generate")

    # ── Training ──────────────────────────────────────────────────
    p.add_argument("--timesteps",      type=int,   default=1_000_000)
    p.add_argument("--log-dir",        type=str,   default=str(_ROOT / "logs" / "run_01"))
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--sentinel-mode",  type=str,   default="rule",
                   choices=["static", "rule", "llm"])
    p.add_argument("--static-lambda",  type=float, default=1.0)
    p.add_argument("--device",         type=str,   default="auto")

    # ── Evaluate ──────────────────────────────────────────────────
    p.add_argument("--n-eval",  type=int, default=3)
    p.add_argument("--model",   type=str, default=None,
                   help="Override model path for Stage 5 (default: log_dir/alvro_ppo_final)")

    # ── Plot ──────────────────────────────────────────────────────
    p.add_argument("--plot-output", type=str,
                   default=str(_ROOT / "logs" / "plots" / "dashboard.png"))

    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║    ALVRO_v3 — Full Pipeline Runner                  ║")
    log.info("║    FinAI Contest 2025                                ║")
    log.info("╚══════════════════════════════════════════════════════╝")

    parquet   = Path(args.parquet)
    log_dir   = Path(args.log_dir)
    model_path = args.model or str(log_dir / "alvro_ppo_final")
    eval_csv   = str(log_dir / "results" / "eval_trajectory.csv")
    monitor_csv = str(log_dir / "train_monitor.csv")
    results = {}

    # ═══════════════════════════════════════════════════════════════
    # Stage 1 — DATA
    # ═══════════════════════════════════════════════════════════════
    if args.use_synthetic:
        data_cmd = [
            PYTHON, "-m", "data.synthetic_generator",
            "--n-hours", str(args.n_hours),
            "--output",  str(parquet),
            "--seed",    str(args.seed),
        ]
    else:
        data_cmd = [PYTHON, "-m", "data.processor"]

    results["data"] = run_stage(
        "Stage 1 — Data Generation",
        data_cmd,
        skip=args.skip_data or parquet.exists(),
    )

    if not parquet.exists():
        log.error("Parquet not found after data stage. Aborting.")
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════════
    # Stage 2 — VALIDATE DATA
    # ═══════════════════════════════════════════════════════════════
    results["validate"] = run_stage(
        "Stage 2 — Data Quality Check",
        [PYTHON, "-m", "scripts.check_data"],
        skip=args.skip_validate,
    )

    # ═══════════════════════════════════════════════════════════════
    # Stage 3 — TEST ENV
    # ═══════════════════════════════════════════════════════════════
    results["test_env"] = run_stage(
        "Stage 3 — Environment Smoke-Test",
        [PYTHON, "-m", "scripts.test_env",
         "--steps", "100",
         "--parquet", str(parquet)],
        skip=args.skip_test_env,
    )

    # ═══════════════════════════════════════════════════════════════
    # Stage 4 — TRAIN
    # ═══════════════════════════════════════════════════════════════
    train_cmd = [
        PYTHON, "-m", "scripts.train",
        "--parquet",       str(parquet),
        "--log-dir",       str(log_dir),
        "--timesteps",     str(args.timesteps),
        "--seed",          str(args.seed),
        "--sentinel-mode", args.sentinel_mode,
        "--static-lambda", str(args.static_lambda),
        "--device",        args.device,
    ]
    results["train"] = run_stage(
        "Stage 4 — PPO Training",
        train_cmd,
        skip=args.skip_train,
    )

    # ═══════════════════════════════════════════════════════════════
    # Stage 5 — EVALUATE
    # ═══════════════════════════════════════════════════════════════
    eval_cmd = [
        PYTHON, "-m", "scripts.evaluate",
        "--model",         model_path,
        "--parquet",       str(parquet),
        "--n-eval",        str(args.n_eval),
        "--sentinel-mode", args.sentinel_mode,
        "--output",        eval_csv,
    ]
    results["evaluate"] = run_stage(
        "Stage 5 — Evaluation",
        eval_cmd,
        skip=args.skip_evaluate,
    )

    # ═══════════════════════════════════════════════════════════════
    # Stage 6 — PLOT
    # ═══════════════════════════════════════════════════════════════
    if Path(eval_csv).exists():
        plot_cmd = [
            PYTHON, "-m", "scripts.plot_results",
            "--episode", eval_csv,
            "--output",  args.plot_output,
        ]
        if Path(monitor_csv).exists():
            plot_cmd += ["--monitor", monitor_csv]
        results["plot"] = run_stage(
            "Stage 6 — Visualisation Dashboard",
            plot_cmd,
            skip=args.skip_plot,
        )
    else:
        log.warning("Eval CSV not found — skipping plot stage.")
        results["plot"] = False

    # ═══════════════════════════════════════════════════════════════
    # Final summary
    # ═══════════════════════════════════════════════════════════════
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║  PIPELINE SUMMARY                                   ║")
    log.info("╠══════════════════════════════════════════════════════╣")
    for stage, ok in results.items():
        status = "✔ OK  " if ok else "✖ FAIL"
        log.info("║  %-10s  %s                              ║", stage, status)
    log.info("╚══════════════════════════════════════════════════════╝")

    if all(results.values()):
        log.info("All stages completed successfully.")
        if Path(args.plot_output).exists():
            log.info("Dashboard → %s", args.plot_output)
        sys.exit(0)
    else:
        n_failed = sum(1 for v in results.values() if not v)
        log.error("%d stage(s) failed. Check logs above.", n_failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
