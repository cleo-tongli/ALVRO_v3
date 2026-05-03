"""
ALVRO_v3 — PPO Agent (MaskablePPO via sb3-contrib)
=====================================================
Module: models/ppo_agent.py

Uses MaskablePPO from sb3-contrib so the Hysteresis Gas Gate (action masking)
is natively integrated into the policy gradient update — masked actions
receive zero probability during both rollout collection and loss computation.

Architecture:
    - Policy:       MlpPolicy (2 × 256 hidden units, tanh activation)
    - Feature Ext:  Default (14-dim flat Box obs → policy head directly)
    - Action Mask:  Provided by env.action_masks() at every step
    - Callbacks:    EvalCallback + CheckpointCallback + GasEfficiencyCallback

Action space: Discrete(4)
    0 – Hold   (always valid)
    1 – Narrow (masked when net_benefit < gas_fee)
    2 – Mid    (masked when net_benefit < gas_fee)
    3 – Wide   (masked when net_benefit < gas_fee)

Usage
-----
    from models.ppo_agent import ALVROAgent
    from env.alvro_env import ALVROEnv

    agent = ALVROAgent(env_fn=lambda: ALVROEnv(df), log_dir="logs/")
    agent.train(total_timesteps=1_000_000)
    agent.save("models/alvro_ppo_final")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# sb3-contrib provides MaskablePPO; falls back to standard PPO if unavailable
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    from sb3_contrib.common.maskable.utils import get_action_masks
    from sb3_contrib.common.wrappers import ActionMasker
    _MASKABLE = True
except ImportError:
    from stable_baselines3 import PPO as MaskablePPO          # type: ignore
    _MASKABLE = False
    import warnings
    warnings.warn(
        "sb3-contrib not installed — falling back to standard PPO without "
        "action masking. Install with: pip install sb3-contrib",
        ImportWarning, stacklevel=2,
    )

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

log = logging.getLogger("ALVROAgent")

# ─────────────────────────────────────────────────────────────────
# Action mask wrapper (required by MaskablePPO)
# ─────────────────────────────────────────────────────────────────

def _mask_fn(env) -> np.ndarray:
    """Bridge between env.action_masks() and ActionMasker wrapper."""
    return env.action_masks()


def wrap_env(env, monitor_log: Optional[str] = None):
    """
    Wrap an ALVROEnv for MaskablePPO:
      1. ActionMasker — exposes the hysteresis gate to SB3
      2. Monitor      — episode reward / length logging
    """
    if _MASKABLE:
        env = ActionMasker(env, _mask_fn)
    if monitor_log:
        env = Monitor(env, filename=monitor_log)
    else:
        env = Monitor(env)
    return env


# ─────────────────────────────────────────────────────────────────
# Custom callback: Gas Efficiency & LP-to-HODL logging
# ─────────────────────────────────────────────────────────────────

class ALVROMetricsCallback(BaseCallback):
    """
    Logs custom ALVRO evaluation metrics to TensorBoard / stdout
    at the end of every training episode.

    Metrics emitted:
        train/gas_efficiency     — Total Fees / Total Gas Spent
        train/lp_to_hodl         — LP portfolio / HODL portfolio
        train/episode_sharpe     — In-episode Sharpe (annualised)
        train/episode_max_dd     — In-episode max drawdown
        train/total_rebalances   — Number of rebalancing events
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._last_n_ep = 0

    def _on_step(self) -> bool:
        # Access underlying unwrapped env(s)
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("episode"):
                # Episode just ended — collect metrics from the underlying env
                env = self.training_env
                # DummyVecEnv: get the first env
                try:
                    base_env = env.envs[0].env  # unwrap Monitor → ActionMasker → ALVROEnv
                    if hasattr(base_env, "env"):
                        base_env = base_env.env
                except AttributeError:
                    base_env = None

                if base_env is not None and hasattr(base_env, "gas_efficiency"):
                    ge = base_env.gas_efficiency()
                    lh = base_env.lp_to_hodl_ratio()
                    sh = base_env.episode_sharpe()
                    dd = base_env.episode_max_drawdown()
                    rb = base_env._total_rebalances

                    self.logger.record("train/gas_efficiency", ge if np.isfinite(ge) else 0.0)
                    self.logger.record("train/lp_to_hodl", lh)
                    self.logger.record("train/episode_sharpe", sh)
                    self.logger.record("train/episode_max_dd", dd)
                    self.logger.record("train/total_rebalances", rb)
                    self.logger.record("train/lambda", base_env.get_lambda())
        return True


# ─────────────────────────────────────────────────────────────────
# Agent class
# ─────────────────────────────────────────────────────────────────

class ALVROAgent:
    """
    Wrapper around MaskablePPO (or standard PPO) for the ALVRO environment.

    Parameters
    ----------
    env_fn : Callable
        Zero-argument factory returning a fresh ALVROEnv instance.
    log_dir : str
        Root directory for TensorBoard logs, checkpoints, and monitor files.
    policy_kwargs : dict | None
        Extra kwargs passed to the SB3 policy (e.g. net_arch, activation_fn).
    learning_rate : float
        Adam learning rate for the PPO optimiser.
    n_steps : int
        Rollout buffer size per env (steps before each gradient update).
    batch_size : int
        Mini-batch size for gradient updates.
    n_epochs : int
        Number of gradient epochs per PPO update cycle.
    gamma : float
        Discount factor for future rewards.
    clip_range : float
        PPO clipping coefficient ε.
    ent_coef : float
        Entropy regularisation coefficient (encourages exploration).
    seed : int
        Global random seed.
    device : str
        Torch device ('cpu', 'cuda', 'auto').
    """

    # Default policy architecture: 2×256 MLP matching the 14-dim obs space
    _DEFAULT_POLICY_KWARGS = {
        "net_arch": [{"pi": [256, 256], "vf": [256, 256]}],
        "activation_fn": None,   # None → SB3 default (tanh)
    }

    def __init__(
        self,
        env_fn: Callable,
        log_dir: str = "logs/",
        policy_kwargs: Optional[dict] = None,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        seed: int = 42,
        device: str = "auto",
    ) -> None:
        self.env_fn = env_fn
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

        # ── Build training environment ────────────────────────────
        train_env = wrap_env(
            env_fn(),
            monitor_log=str(self.log_dir / "train_monitor"),
        )
        train_vec = DummyVecEnv([lambda: train_env])

        # ── Policy kwargs ─────────────────────────────────────────
        pk = dict(self._DEFAULT_POLICY_KWARGS)
        if policy_kwargs:
            pk.update(policy_kwargs)
        # SB3 expects activation_fn as a class, not None
        if pk.get("activation_fn") is None:
            import torch.nn as nn
            pk["activation_fn"] = nn.Tanh

        # ── Build PPO model ───────────────────────────────────────
        log.info(
            "Initialising %sPPO | lr=%.1e  n_steps=%d  batch=%d  "
            "epochs=%d  gamma=%.3f  clip=%.2f  ent=%.4f",
            "Maskable" if _MASKABLE else "Standard ",
            learning_rate, n_steps, batch_size,
            n_epochs, gamma, clip_range, ent_coef,
        )
        self.model = MaskablePPO(
            policy="MlpPolicy",
            env=train_vec,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            clip_range=clip_range,
            ent_coef=ent_coef,
            policy_kwargs=pk,
            tensorboard_log=str(self.log_dir / "tb"),
            seed=seed,
            device=device,
            verbose=1,
        )

    def _build_callbacks(
        self,
        eval_env_fn: Optional[Callable] = None,
        eval_freq: int = 10_000,
        save_freq: int = 50_000,
    ) -> CallbackList:
        callbacks = [ALVROMetricsCallback()]

        # ── Checkpoint callback ───────────────────────────────────
        ckpt_cb = CheckpointCallback(
            save_freq=save_freq,
            save_path=str(self.log_dir / "checkpoints"),
            name_prefix="alvro_ppo",
            verbose=1,
        )
        callbacks.append(ckpt_cb)

        # ── Eval callback (optional) ──────────────────────────────
        if eval_env_fn is not None:
            eval_env = wrap_env(eval_env_fn())
            eval_vec = DummyVecEnv([lambda: eval_env])
            if _MASKABLE:
                eval_cb = MaskableEvalCallback(
                    eval_env=eval_vec,
                    best_model_save_path=str(self.log_dir / "best_model"),
                    log_path=str(self.log_dir / "eval"),
                    eval_freq=eval_freq,
                    n_eval_episodes=5,
                    deterministic=True,
                    verbose=1,
                )
            else:
                eval_cb = EvalCallback(
                    eval_env=eval_vec,
                    best_model_save_path=str(self.log_dir / "best_model"),
                    log_path=str(self.log_dir / "eval"),
                    eval_freq=eval_freq,
                    n_eval_episodes=5,
                    deterministic=True,
                    verbose=1,
                )
            callbacks.append(eval_cb)

        return CallbackList(callbacks)

    def train(
        self,
        total_timesteps: int = 1_000_000,
        eval_env_fn: Optional[Callable] = None,
        eval_freq: int = 10_000,
        save_freq: int = 50_000,
        reset_num_timesteps: bool = True,
    ) -> None:
        """
        Run PPO training.

        Parameters
        ----------
        total_timesteps : int
            Total environment interaction steps.
        eval_env_fn : Callable | None
            If provided, an eval environment is created for periodic evaluation.
        eval_freq : int
            Steps between evaluation runs.
        save_freq : int
            Steps between model checkpoints.
        reset_num_timesteps : bool
            If False, continue from a previous training run.
        """
        log.info("Starting training for %d timesteps …", total_timesteps)
        callbacks = self._build_callbacks(eval_env_fn, eval_freq, save_freq)
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=True,
        )
        log.info("Training complete.")

    def predict(
        self,
        obs: np.ndarray,
        action_masks: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict an action for a given observation.

        Parameters
        ----------
        obs : np.ndarray  shape (14,) or (N, 14)
        action_masks : np.ndarray | None
            Boolean mask from env.action_masks(). Required for masked policy.
        deterministic : bool
            If True, return the greedy action.
        """
        if _MASKABLE and action_masks is not None:
            return self.model.predict(
                obs,
                action_masks=action_masks,
                deterministic=deterministic,
            )
        return self.model.predict(obs, deterministic=deterministic)

    def save(self, path: str) -> None:
        """Save the model to disk."""
        self.model.save(path)
        log.info("Model saved → %s", path)

    @classmethod
    def load(
        cls,
        path: str,
        env_fn: Callable,
        log_dir: str = "logs/",
        device: str = "auto",
    ) -> "ALVROAgent":
        """
        Load a previously saved model.

        Parameters
        ----------
        path : str
            Path to the saved .zip file (without extension).
        env_fn : Callable
            Environment factory (needed to rebuild the VecEnv wrapper).
        """
        agent = object.__new__(cls)
        agent.env_fn = env_fn
        agent.log_dir = Path(log_dir)
        agent.seed = 42

        env = wrap_env(env_fn())
        vec_env = DummyVecEnv([lambda: env])

        agent.model = MaskablePPO.load(path, env=vec_env, device=device)
        log.info("Model loaded ← %s", path)
        return agent
