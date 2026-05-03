"""
ALVRO_v3 — Evaluation Script
==============================
scripts/evaluate.py

Load a trained ALVRO PPO model and run a full evaluation over the
test split of the market data. Produces a comprehensive metrics report
and saves episode trajectory to a CSV for further analysis.

Metrics reported:
  1. LP-to-HODL Ratio
  2. Max Drawdown
  3. Gas Efficiency (Total Fees / Total Gas)
  4. Sharpe Ratio (annualised)
  5. Calmar Ratio
  6. In-Range %
  7. Mean / Std λ

Usage
-----
    python -m scripts.evaluate --model logs/run_01/alvro_ppo_final

    # With DeepSeek Sentinel (rule mode):
    python -m scripts.evaluate \
        --model logs/run_01/alvro_ppo_final \
        --sentinel-mode rule \
        --n-eval 5 \
        --output results/eval_run01.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from env.alvro_env import ALVROEnv
from models.ppo_agent import ALVROAgent
from core.sentinel import DeepSeekSentinel
from core.evaluator import run_evaluation_episode, ALVROEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ALVRO_v3 Evaluation")
    p.add_argument("--model",         type=str, required=True,
                   help="Path to trained model .zip (without extension)")
    p.add_argument("--parquet",       type=str,
                   default=str(_ROOT / "data" / "processed_market.parquet"))
    p.add_argument("--n-eval",        type=int,   default=5)
    p.add_argument("--seed",          type=int,   default=100)
    p.add_argument("--sentinel-mode", choices=["static", "rule", "llm"], default="rule")
    p.add_argument("--static-lambda", type=float, default=1.0)
    p.add_argument("--eval-split",    type=float, default=0.20,
                   help="Fraction of data to use as eval set (from end)")
    p.add_argument("--output",        type=str,   default=None,
                   help="Path to save episode trajectory CSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info("═══ ALVRO_v3 Evaluation — START ═══")

    # ── 1. Load data ──────────────────────────────────────────────
    parquet = Path(args.parquet)
    if not parquet.exists():
        log.error("%s not found. Run 'python -m data.processor' first.", parquet)
        sys.exit(1)

    df = pd.read_parquet(parquet)
    split = int(len(df) * (1.0 - args.eval_split))
    df_eval = df.iloc[split:].reset_index(drop=True)
    log.info("Eval split: %d rows (%.0f%% of total)", len(df_eval), args.eval_split * 100)

    # ── 2. Build Sentinel ─────────────────────────────────────────
    sentinel = None
    if args.sentinel_mode != "static":
        sentinel = DeepSeekSentinel(
            mode=args.sentinel_mode,
            static_lambda=args.static_lambda,
        )
        log.info("Sentinel: %s mode", args.sentinel_mode)

    # ── 3. Load Agent ─────────────────────────────────────────────
    model_path = args.model
    log.info("Loading model from %s …", model_path)

    def eval_env_fn():
        return ALVROEnv(market_data=df_eval)

    try:
        agent = ALVROAgent.load(
            path=model_path,
            env_fn=eval_env_fn,
        )
    except Exception as exc:
        log.error("Failed to load model: %s", exc)
        sys.exit(1)

    # ── 4. Run evaluation ─────────────────────────────────────────
    log.info("Running %d evaluation episode(s) …", args.n_eval)
    metrics = run_evaluation_episode(
        agent=agent,
        df_market=df_eval,
        sentinel=sentinel,
        n_eval=args.n_eval,
        seed=args.seed,
        deterministic=True,
        verbose=True,
    )

    # ── 5. Print final report ─────────────────────────────────────
    print(metrics.summary())

    # ── 6. Save trajectory (optional) ────────────────────────────
    if args.output:
        # Run one more detailed episode and save per-step CSV
        env = ALVROEnv(market_data=df_eval)
        obs, info = env.reset(seed=args.seed)
        records = []
        done = False
        while not done:
            if sentinel is not None:
                t = env._t
                if t < len(df_eval):
                    row = df_eval.iloc[t]
                    from core.sentinel import attach_sentinel_to_env
                    attach_sentinel_to_env(env, sentinel, row.to_dict())

            masks = env.action_masks()
            action, _ = agent.predict(obs, action_masks=masks, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            records.append({
                "step":            info.get("step"),
                "price":           info.get("price"),
                "price_lower":     info.get("price_lower"),
                "price_upper":     info.get("price_upper"),
                "action":          int(action),
                "reward":          reward,
                "fees_t":          info.get("fees_t", 0.0),
                "lvr_t":           info.get("lvr_t", 0.0),
                "lvr_penalty":     info.get("lvr_penalty", 0.0),
                "gas_t":           info.get("gas_t", 0.0),
                "in_range":        info.get("in_range", True),
                "rebalanced":      info.get("rebalanced", False),
                "net_pnl":         info.get("net_pnl"),
                "portfolio_value": info.get("portfolio_value"),
                "lambda":          info.get("lambda"),
                "initial_price":   info.get("initial_price"),
            })

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(out_path, index=False)
        log.info("Episode trajectory saved → %s", out_path)

    log.info("═══ ALVRO_v3 Evaluation — DONE ═══")


if __name__ == "__main__":
    main()
