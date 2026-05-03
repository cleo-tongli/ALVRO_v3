"""
ALVRO_v3 — Training Script
============================
scripts/train.py

End-to-end training pipeline:
  1. Load processed market data (requires data/processor.py to have run)
  2. Optionally attach the DeepSeek Sentinel (rule-based or LLM mode)
  3. Build train + eval environments
  4. Train MaskablePPO for total_timesteps
  5. Save the final model and print metrics

Usage
-----
    # Basic run (rule-based sentinel, no LLM key needed):
    python -m scripts.train

    # Custom settings:
    python -m scripts.train \
        --timesteps 2000000 \
        --sentinel-mode rule \
        --log-dir logs/run_01 \
        --seed 42

    # With DeepSeek LLM sentiment (requires DEEPSEEK_API_KEY):
    python -m scripts.train --sentinel-mode llm

    # Ablation: no sentinel (static λ=1.0):
    python -m scripts.train --sentinel-mode static --static-lambda 1.0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd

# ── project root on sys.path ──────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from env.alvro_env import ALVROEnv
from models.ppo_agent import ALVROAgent, wrap_env
from core.sentinel import DeepSeekSentinel, attach_sentinel_to_env
from core.evaluator import run_evaluation_episode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("train")

# ─────────────────────────────────────────────────────────────────
# Sentinel-aware env factory
# ─────────────────────────────────────────────────────────────────

class SentinelEnvWrapper(gym.Wrapper):
    """
    Gymnasium Wrapper that injects the DeepSeek Sentinel λ into the
    underlying ALVROEnv *before* each step, so the risk multiplier is
    always current when the agent acts.

    Inheriting from gym.Wrapper ensures SB3's DummyVecEnv, VecMonitor,
    and ActionMasker can correctly introspect observation_space,
    action_space, render_mode, and spec without breaking.
    """

    def __init__(self, env: ALVROEnv, sentinel: DeepSeekSentinel, df: pd.DataFrame):
        super().__init__(env)          # gym.Wrapper stores env as self.env
        self._sentinel = sentinel
        self._df = df

    # ── Sentinel injection ────────────────────────────────────────

    def _inject_lambda(self) -> None:
        """Read the current market row and push λ into the inner env."""
        t = self.env._t
        if t < len(self._df):
            attach_sentinel_to_env(self.env, self._sentinel, self._df.iloc[t].to_dict())

    # ── Gymnasium API ─────────────────────────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._inject_lambda()
        return obs, info

    def step(self, action):
        self._inject_lambda()
        return self.env.step(action)

    # ── Action masking (forwarded for MaskablePPO / ActionMasker) ─

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()

    # ── Episode metric proxies (used by ALVROMetricsCallback) ─────

    def update_external_risk(self, lam: float) -> None:
        self.env.update_external_risk(lam)

    def get_lambda(self) -> float:
        return self.env.get_lambda()

    def episode_sharpe(self) -> float:
        return self.env.episode_sharpe()

    def episode_max_drawdown(self) -> float:
        return self.env.episode_max_drawdown()

    def gas_efficiency(self) -> float:
        return self.env.gas_efficiency()

    def lp_to_hodl_ratio(self, *args, **kwargs) -> float:
        return self.env.lp_to_hodl_ratio(*args, **kwargs)

    @property
    def _total_rebalances(self) -> int:
        return self.env._total_rebalances


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ALVRO_v3 PPO Training")
    p.add_argument("--parquet",     type=str,   default=str(_ROOT / "data" / "processed_market.parquet"))
    p.add_argument("--log-dir",     type=str,   default=str(_ROOT / "logs" / "run_01"))
    p.add_argument("--timesteps",   type=int,   default=1_000_000)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--n-steps",     type=int,   default=2048)
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--n-epochs",    type=int,   default=10)
    p.add_argument("--gamma",       type=float, default=0.99)
    p.add_argument("--clip-range",  type=float, default=0.2)
    p.add_argument("--ent-coef",    type=float, default=0.01)
    p.add_argument("--sentinel-mode", choices=["static", "rule", "llm"], default="rule")
    p.add_argument("--static-lambda", type=float, default=1.0)
    p.add_argument("--eval-freq",   type=int,   default=20_000)
    p.add_argument("--save-freq",   type=int,   default=100_000)
    p.add_argument("--device",      type=str,   default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info("═══ ALVRO_v3 Training Pipeline — START ═══")
    log.info("Config: %s", vars(args))

    # ── 1. Load data ──────────────────────────────────────────────
    parquet = Path(args.parquet)
    if not parquet.exists():
        log.error(
            "%s not found. Run 'python -m data.processor' first.", parquet
        )
        sys.exit(1)

    df = pd.read_parquet(parquet)
    log.info("Loaded %d rows × %d cols from %s", len(df), len(df.columns), parquet.name)

    # ── 2. Build Sentinel ─────────────────────────────────────────
    sentinel = DeepSeekSentinel(
        mode=args.sentinel_mode,
        static_lambda=args.static_lambda,
    )
    log.info("Sentinel mode: %s", args.sentinel_mode)

    # ── 3. Build env factories ─────────────────────────────────────
    # Use first 80% for training, last 20% for evaluation
    split = int(len(df) * 0.80)
    df_train = df.iloc[:split].reset_index(drop=True)
    df_eval  = df.iloc[split:].reset_index(drop=True)

    log.info(
        "Train rows: %d | Eval rows: %d", len(df_train), len(df_eval)
    )

    def make_train_env():
        env = ALVROEnv(
            market_data=df_train,
            max_steps=min(2048, len(df_train) - 1),
        )
        if args.sentinel_mode != "static":
            return SentinelEnvWrapper(env, sentinel, df_train)
        return env

    def make_eval_env():
        env = ALVROEnv(
            market_data=df_eval,
            max_steps=len(df_eval) - 1,
        )
        if args.sentinel_mode != "static":
            return SentinelEnvWrapper(env, sentinel, df_eval)
        return env

    # ── 4. Build Agent ────────────────────────────────────────────
    agent = ALVROAgent(
        env_fn=make_train_env,
        log_dir=args.log_dir,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        seed=args.seed,
        device=args.device,
    )

    # ── 5. Train ──────────────────────────────────────────────────
    agent.train(
        total_timesteps=args.timesteps,
        eval_env_fn=make_eval_env,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
    )

    # ── 6. Save final model ───────────────────────────────────────
    model_path = Path(args.log_dir) / "alvro_ppo_final"
    agent.save(str(model_path))
    log.info("Final model saved → %s.zip", model_path)

    # ── 7. Quick post-training evaluation ─────────────────────────
    log.info("Running post-training evaluation on eval split …")
    try:
        metrics = run_evaluation_episode(
            agent=agent,
            df_market=df_eval,
            sentinel=sentinel if args.sentinel_mode != "static" else None,
            n_eval=3,
            verbose=True,
        )
        print(metrics.summary())
    except Exception as exc:
        log.warning("Post-training evaluation failed: %s", exc)

    log.info("═══ ALVRO_v3 Training Pipeline — DONE ═══")


if __name__ == "__main__":
    main()
