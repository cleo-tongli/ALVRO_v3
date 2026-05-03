"""
ALVRO_v3 — Evaluation Metrics Engine
======================================
Module: core/evaluator.py

Computes and reports the three primary contest evaluation metrics plus
a full set of diagnostics over a completed episode or backtest run.

Metrics
-------
1. LP-to-HODL Ratio
   The ratio of the LP's final portfolio value to what a passive holder would
   have earned. A ratio > 1.0 means the LP strategy outperformed HODL.

   HODL model: 50 % WETH + 50 % USDC held at the start price P₀.
       V_HODL(t) = 0.5 * V₀ * (P_t / P₀)   [WETH leg]
                 + 0.5 * V₀                  [USDC leg]
                 = V₀ * (0.5 * P_t/P₀ + 0.5)

2. Max Drawdown
   Maximum peak-to-trough decline in the cumulative reward curve.
   MDD = max over t of (peak_up_to_t - cum_reward_t).

3. Gas Efficiency
   Total_Fees_Earned / Total_Gas_Spent.
   Higher is better. ∞ when no gas was ever spent (pure-Hold strategy).

Usage
-----
    from core.evaluator import ALVROEvaluator, run_evaluation_episode
    import pandas as pd

    df_market = pd.read_parquet("data/processed_market.parquet")

    # 1. Run a full episode with a trained agent
    metrics = run_evaluation_episode(agent, df_market, n_eval=5)
    print(metrics.summary())

    # 2. Or build an evaluator from raw episode records
    ev = ALVROEvaluator.from_episode_records(records, initial_capital=100_000)
    ev.print_report()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("ALVROEvaluator")


# ─────────────────────────────────────────────────────────────────
# Data container for per-step records
# ─────────────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    """Per-step diagnostic record collected during evaluation."""
    step: int
    price: float
    action: int
    reward: float
    fees_t: float
    lvr_t: float
    gas_t: float
    rebalanced: bool
    in_range: bool
    net_pnl: float
    portfolio_value: float
    lambda_val: float


# ─────────────────────────────────────────────────────────────────
# Metrics result dataclass
# ─────────────────────────────────────────────────────────────────

@dataclass
class EvalMetrics:
    """Container for all evaluation metrics from one episode run."""
    # Primary contest metrics
    lp_to_hodl_ratio: float = 0.0
    max_drawdown: float = 0.0
    gas_efficiency: float = 0.0

    # Secondary diagnostics
    total_reward: float = 0.0
    total_fees: float = 0.0
    total_gas_spent: float = 0.0
    total_lvr: float = 0.0
    total_rebalances: int = 0
    n_steps: int = 0
    in_range_pct: float = 0.0
    final_portfolio: float = 0.0
    initial_capital: float = 100_000.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    mean_lambda: float = 0.0
    std_lambda: float = 0.0

    # Episodic price data for ratio computation
    start_price: float = 0.0
    end_price: float = 0.0

    def summary(self) -> str:
        """Return a formatted summary string."""
        ge_str = (
            f"{self.gas_efficiency:.2f}×"
            if np.isfinite(self.gas_efficiency)
            else "∞ (no gas spent)"
        )
        return (
            f"\n{'═'*58}\n"
            f"  ALVRO Evaluation Report\n"
            f"{'─'*58}\n"
            f"  LP-to-HODL Ratio          {self.lp_to_hodl_ratio:>10.4f}\n"
            f"  Max Drawdown              {self.max_drawdown:>10.4f}\n"
            f"  Gas Efficiency            {ge_str:>15}\n"
            f"{'─'*58}\n"
            f"  Total Reward              {self.total_reward:>+10.4f}\n"
            f"  Total Fees Earned         {self.total_fees:>10.4f}\n"
            f"  Total Gas Spent           {self.total_gas_spent:>10.4f}\n"
            f"  Total LVR Leakage         {self.total_lvr:>10.4f}\n"
            f"  Rebalancing Events        {self.total_rebalances:>10d}\n"
            f"  In-Range %                {self.in_range_pct:>10.1f}%\n"
            f"  Sharpe Ratio (annual.)    {self.sharpe_ratio:>10.4f}\n"
            f"  Calmar Ratio              {self.calmar_ratio:>10.4f}\n"
            f"  Mean λ                    {self.mean_lambda:>10.4f}\n"
            f"  Std  λ                    {self.std_lambda:>10.4f}\n"
            f"  Steps                     {self.n_steps:>10d}\n"
            f"  Start Price               {self.start_price:>10.2f}\n"
            f"  End Price                 {self.end_price:>10.2f}\n"
            f"  Final Portfolio           {self.final_portfolio:>10.2f}\n"
            f"{'═'*58}\n"
        )


# ─────────────────────────────────────────────────────────────────
# Evaluator class
# ─────────────────────────────────────────────────────────────────

class ALVROEvaluator:
    """
    Compute evaluation metrics from episode step records.

    Parameters
    ----------
    records : list of StepRecord
    initial_capital : float
        Starting portfolio value (used as HODL baseline denominator).
    """

    def __init__(
        self,
        records: List[StepRecord],
        initial_capital: float = 100_000.0,
    ) -> None:
        self.records = records
        self.initial_capital = initial_capital

    # ── Factory ───────────────────────────────────────────────────

    @classmethod
    def from_episode_records(
        cls,
        records: List[Dict],
        initial_capital: float = 100_000.0,
    ) -> "ALVROEvaluator":
        """
        Build from a list of raw info dicts (as returned by env.step).

        Parameters
        ----------
        records : list of dicts with keys:
            step, price, action, reward, fees_t, lvr_t, gas_t,
            rebalanced, in_range, net_pnl, portfolio_value, lambda
        """
        step_records = []
        for r in records:
            step_records.append(StepRecord(
                step=r.get("step", 0),
                price=r.get("price", 0.0),
                action=r.get("action", 0),
                reward=r.get("reward", 0.0),
                fees_t=r.get("fees_t", 0.0),
                lvr_t=r.get("lvr_t", 0.0),
                gas_t=r.get("gas_t", 0.0),
                rebalanced=bool(r.get("rebalanced", False)),
                in_range=bool(r.get("in_range", True)),
                net_pnl=r.get("net_pnl", 0.0),
                portfolio_value=r.get("portfolio_value", initial_capital),
                lambda_val=r.get("lambda", 1.0),
            ))
        return cls(step_records, initial_capital)

    # ── Primary metric computations ───────────────────────────────

    def compute_lp_to_hodl(self) -> float:
        """
        LP-to-HODL Ratio.

        HODL model: 50 % WETH + 50 % USDC at t=0.
            V_HODL(T) = V₀ × (0.5 × P_T/P₀ + 0.5)

        LP value: final portfolio_value from the env.
        """
        if not self.records:
            return 1.0

        P0 = self.records[0].price
        PT = self.records[-1].price
        if P0 <= 0:
            return 1.0

        V0 = self.initial_capital
        v_hodl = V0 * (0.5 * (PT / P0) + 0.5)
        v_lp = self.records[-1].portfolio_value

        if v_hodl < 1e-9:
            return 1.0
        return v_lp / v_hodl

    def compute_max_drawdown(self) -> float:
        """Max Drawdown of the cumulative reward curve."""
        if not self.records:
            return 0.0
        rewards = np.array([r.reward for r in self.records])
        cum = np.cumsum(rewards)
        roll_max = np.maximum.accumulate(cum)
        drawdown = roll_max - cum
        return float(np.max(drawdown))

    def compute_gas_efficiency(self) -> float:
        """Gas Efficiency = Total Fees / Total Gas Spent."""
        total_fees = sum(r.fees_t for r in self.records)
        total_gas = sum(r.gas_t for r in self.records)
        if total_gas < 1e-9:
            return float("inf")
        return total_fees / total_gas

    def compute_sharpe(self) -> float:
        """Annualised Sharpe ratio of per-step rewards."""
        rewards = np.array([r.reward for r in self.records])
        if len(rewards) < 2:
            return 0.0
        mu = np.mean(rewards)
        std = np.std(rewards, ddof=1) + 1e-9
        return float(mu / std * np.sqrt(8760))   # hourly → annual

    def compute_calmar(self) -> float:
        """Calmar Ratio = Annualised Return (%) / Max Drawdown (%)."""
        rewards = np.array([r.reward for r in self.records])
        n = len(rewards)
        if n == 0 or self.initial_capital <= 0:
            return 0.0
        total_ret_pct = float(np.sum(rewards)) / self.initial_capital
        annual_ret_pct = total_ret_pct * (8760 / n)
        mdd_pct = self.compute_max_drawdown() / self.initial_capital
        if mdd_pct < 1e-9:
            return float("inf")
        return annual_ret_pct / mdd_pct

    def compute_all(self) -> EvalMetrics:
        """Compute and return all metrics as an EvalMetrics dataclass."""
        if not self.records:
            return EvalMetrics()

        rewards = [r.reward for r in self.records]
        lambdas = [r.lambda_val for r in self.records]

        m = EvalMetrics(
            lp_to_hodl_ratio=self.compute_lp_to_hodl(),
            max_drawdown=self.compute_max_drawdown(),
            gas_efficiency=self.compute_gas_efficiency(),
            total_reward=sum(rewards),
            total_fees=sum(r.fees_t for r in self.records),
            total_gas_spent=sum(r.gas_t for r in self.records),
            total_lvr=sum(r.lvr_t for r in self.records),
            total_rebalances=sum(1 for r in self.records if r.rebalanced),
            n_steps=len(self.records),
            in_range_pct=100.0 * sum(1 for r in self.records if r.in_range) / max(len(self.records), 1),
            final_portfolio=self.records[-1].portfolio_value,
            initial_capital=self.initial_capital,
            sharpe_ratio=self.compute_sharpe(),
            calmar_ratio=self.compute_calmar(),
            mean_lambda=float(np.mean(lambdas)),
            std_lambda=float(np.std(lambdas)),
            start_price=self.records[0].price,
            end_price=self.records[-1].price,
        )
        return m

    def print_report(self) -> None:
        """Print the full evaluation report to stdout."""
        metrics = self.compute_all()
        print(metrics.summary())

    def to_dataframe(self) -> pd.DataFrame:
        """Export all step records as a DataFrame for further analysis."""
        return pd.DataFrame([vars(r) for r in self.records])


# ─────────────────────────────────────────────────────────────────
# High-level convenience: run_evaluation_episode
# ─────────────────────────────────────────────────────────────────

def run_evaluation_episode(
    agent,
    df_market: pd.DataFrame,
    sentinel=None,
    n_eval: int = 1,
    initial_capital: float = 100_000.0,
    max_steps: Optional[int] = None,
    seed: int = 0,
    deterministic: bool = True,
    verbose: bool = True,
) -> EvalMetrics:
    """
    Run `n_eval` evaluation episodes with the given agent and return
    the average EvalMetrics.

    Parameters
    ----------
    agent : ALVROAgent
        Trained PPO agent.
    df_market : pd.DataFrame
        Processed market data.
    sentinel : DeepSeekSentinel | None
        Optional risk signal generator. If provided, λ is updated
        each step from the sentinel before the agent acts.
    n_eval : int
        Number of evaluation episodes to average over.
    initial_capital : float
        Starting portfolio value (USD).
    max_steps : int | None
        Episode length cap. None → full dataset.
    seed : int
        Starting random seed (incremented per episode).
    deterministic : bool
        Use deterministic policy.
    verbose : bool
        Print per-episode metrics.

    Returns
    -------
    EvalMetrics  — averaged across all n_eval episodes.
    """
    from env.alvro_env import ALVROEnv

    all_metrics: List[EvalMetrics] = []

    for ep in range(n_eval):
        env = ALVROEnv(
            market_data=df_market,
            initial_capital=initial_capital,
            max_steps=max_steps,
        )
        obs, info = env.reset(seed=seed + ep)
        records: List[Dict] = []
        done = False

        while not done:
            # Optional: update λ from sentinel each step
            if sentinel is not None:
                row_idx = env._t
                if row_idx < len(df_market):
                    row = df_market.iloc[row_idx]
                    from core.sentinel import attach_sentinel_to_env
                    attach_sentinel_to_env(env, sentinel, row.to_dict())

            # Get action mask
            try:
                masks = env.action_masks()
            except AttributeError:
                masks = None

            action, _ = agent.predict(
                obs,
                action_masks=masks,
                deterministic=deterministic,
            )
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

            records.append({
                "step":            info.get("step", 0),
                "price":           info.get("price", 0.0),
                "action":          info.get("action", int(action)),
                "reward":          reward,
                "fees_t":          info.get("fees_t", 0.0),
                "lvr_t":           info.get("lvr_t", 0.0),
                "gas_t":           info.get("gas_t", 0.0),
                "rebalanced":      info.get("rebalanced", False),
                "in_range":        info.get("in_range", True),
                "net_pnl":         info.get("net_pnl", 0.0),
                "portfolio_value": info.get("portfolio_value", initial_capital),
                "lambda":          info.get("lambda", 1.0),
            })

        ev = ALVROEvaluator.from_episode_records(records, initial_capital)
        metrics = ev.compute_all()
        all_metrics.append(metrics)

        if verbose:
            log.info("Episode %d/%d %s", ep + 1, n_eval, metrics.summary())

    # ── Average across episodes ───────────────────────────────────
    avg = EvalMetrics(
        lp_to_hodl_ratio=float(np.mean([m.lp_to_hodl_ratio for m in all_metrics])),
        max_drawdown=float(np.mean([m.max_drawdown for m in all_metrics])),
        gas_efficiency=float(np.nanmean(
            [m.gas_efficiency for m in all_metrics if np.isfinite(m.gas_efficiency)]
        ) if any(np.isfinite(m.gas_efficiency) for m in all_metrics) else float("inf")),
        total_reward=float(np.mean([m.total_reward for m in all_metrics])),
        total_fees=float(np.mean([m.total_fees for m in all_metrics])),
        total_gas_spent=float(np.mean([m.total_gas_spent for m in all_metrics])),
        total_lvr=float(np.mean([m.total_lvr for m in all_metrics])),
        total_rebalances=int(np.mean([m.total_rebalances for m in all_metrics])),
        n_steps=int(np.mean([m.n_steps for m in all_metrics])),
        in_range_pct=float(np.mean([m.in_range_pct for m in all_metrics])),
        final_portfolio=float(np.mean([m.final_portfolio for m in all_metrics])),
        initial_capital=initial_capital,
        sharpe_ratio=float(np.mean([m.sharpe_ratio for m in all_metrics])),
        calmar_ratio=float(np.mean([m.calmar_ratio for m in all_metrics])),
        mean_lambda=float(np.mean([m.mean_lambda for m in all_metrics])),
        std_lambda=float(np.mean([m.std_lambda for m in all_metrics])),
        start_price=all_metrics[0].start_price,
        end_price=all_metrics[-1].end_price,
    )
    return avg
