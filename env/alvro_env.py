"""
ALVRO_v3 — Gymnasium Environment
==================================
Module: env/alvro_env.py

ALVROEnv simulates a Uniswap v3 Liquidity Provider acting over hourly market
data.  The PPO agent decides the width of its concentrated-liquidity tick
range; the environment accrues fees when price is in-range and penalises the
agent for LVR leakage and gas costs on every rebalancing action.

Architecture overview
---------------------
Observation (14-dim Box):
    [0]  price_rel          – current price normalised to range centre
    [1]  sigma              – GARCH(1,1) conditional volatility
    [2]  lvr_proxy          – LVR estimate for current step
    [3]  lambda_risk        – DeepSeek risk multiplier (ext. or default)
    [4]  accumulated_fees   – unrealised fee income since last rebalance
    [5]  tick_lower_norm    – lower bound, normalised to current price
    [6]  tick_upper_norm    – upper bound, normalised to current price
    [7]  log_return         – most recent log return
    [8]  rsi_14             – Relative Strength Index (14 bars)
    [9]  ma_24h_rel         – 24-h MA relative to current price
    [10] portfolio_value    – total USD portfolio value (normalised)
    [11] in_range           – binary: 1 if price inside [lower, upper]
    [12] steps_since_rebal  – steps elapsed since last rebalancing action
    [13] net_pnl            – cumulative net P&L this episode (normalised)

Action space: Discrete(4)
    0 – Hold        (±0 % – keep current range)
    1 – Narrow      (±1 %  of current price)
    2 – Mid         (±5 %  of current price)
    3 – Wide        (±15 % of current price – defensive mode)

Reward:
    Reward_t = Fees_t − (lambda_t × LVR_t) − Gas_t
    where Gas_t = GAS_FEE if action != Hold else 0.

Hysteresis Gas Gate:
    Before accepting a rebalancing action the env checks: if the predicted
    net benefit of rebalancing (Fees − lambda×LVR) over the next step is
    less than the gas fee, actions 1/2/3 are severely penalised, effectively
    masking them.

Usage:
    from env.alvro_env import ALVROEnv, make_alvro_env
    import pandas as pd

    df = pd.read_parquet("data/processed_market.parquet")
    env = ALVROEnv(market_data=df)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action=2)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

log = logging.getLogger("ALVROEnv")

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────
GAS_FEE: float = 5.0               # USD per rebalancing action
DEFAULT_LAMBDA: float = 1.0        # neutral risk multiplier
GAS_PENALTY_SCALE: float = 5.0     # multiplier used in hysteresis penalty

# Approximate total pool TVL for volume-based fee estimation (WETH/USDC 0.05%)
_POOL_TVL_REF: float = 300_000_000.0

# Range half-widths for each discrete action
RANGE_PCT: Dict[int, float] = {
    0: None,   # Hold – range unchanged
    1: 0.01,   # Narrow ±1 %
    2: 0.05,   # Mid    ±5 %
    3: 0.15,   # Wide  ±15 %
}

OBS_DIM: int = 14
N_ACTIONS: int = 4

# Normalisation denominators (prevent obs explosion)
_FEE_NORM: float = 500.0           # max expected accumulated fees
_PNL_NORM: float = 10_000.0        # normalise PnL over episode
_STEPS_NORM: float = 100.0         # normalise steps-since-rebal

# ─────────────────────────────────────────────────────────────────


class ALVROEnv(gym.Env):
    """
    Custom Gymnasium environment for the ALVRO Uniswap v3 LP agent.

    Parameters
    ----------
    market_data : pd.DataFrame
        Processed feature frame produced by data/processor.py.
        Required columns: timestamp, price, log_return, sigma,
        lvr_proxy, volume, rsi_14, ma_24h.
    initial_capital : float
        Starting USD capital (default: 100 000).
    gas_fee : float
        Fixed USD cost per rebalancing transaction (default: 5.0).
    default_lambda : float
        Fallback risk multiplier when no external signal is present.
    fee_tier : float
        Uniswap pool fee tier (e.g. 0.0005 for 0.05 %).
    max_steps : int | None
        Episode length cap.  None → use full dataset.
    render_mode : str | None
        Optional render mode ("human" for console output).
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    # ── Construction ─────────────────────────────────────────────

    def __init__(
        self,
        market_data: pd.DataFrame,
        initial_capital: float = 100_000.0,
        gas_fee: float = GAS_FEE,
        default_lambda: float = DEFAULT_LAMBDA,
        fee_tier: float = 0.0005,
        max_steps: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        # ── Validate & store market data ──────────────────────────
        required_cols = {
            "price", "log_return", "sigma", "lvr_proxy",
            "rsi_14", "ma_24h",
        }
        missing = required_cols - set(market_data.columns)
        if missing:
            raise ValueError(f"market_data missing columns: {missing}")

        self._data: np.ndarray = market_data.reset_index(drop=True).to_numpy(dtype=np.float64)
        self._col: Dict[str, int] = {
            col: i for i, col in enumerate(market_data.columns)
        }
        self._n_steps: int = len(self._data)
        self.max_steps: int = max_steps if max_steps else self._n_steps - 1

        # ── Hyper-parameters ─────────────────────────────────────
        self.initial_capital = initial_capital
        self.gas_fee = gas_fee
        self.fee_tier = fee_tier
        self.render_mode = render_mode

        # ── External risk signal (DeepSeek λ) ────────────────────
        self._lambda: float = default_lambda
        self._default_lambda: float = default_lambda

        # ── Gymnasium spaces ──────────────────────────────────────
        low = np.full(OBS_DIM, -np.inf, dtype=np.float32)
        high = np.full(OBS_DIM, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high,
                                            shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)

        # ── Episode state (initialised in reset) ──────────────────
        self._t: int = 0
        self._done: bool = False
        self._price_lower: float = 0.0
        self._price_upper: float = 0.0
        self._price_center: float = 0.0
        self._accumulated_fees: float = 0.0
        self._steps_since_rebal: int = 0
        self._portfolio_value: float = initial_capital
        self._net_pnl: float = 0.0
        self._current_range_pct: float = 0.05   # start at Mid
        self._episode_rewards: list = []
        self._initial_price: float = 0.0
        # Evaluation counters
        self._total_fees: float = 0.0
        self._total_gas: float = 0.0
        self._total_rebalances: int = 0
        self._hodl_reference: float = initial_capital  # HODL baseline value

    # ── Public API — External risk injection ─────────────────────

    def update_external_risk(self, new_lambda: float) -> None:
        """
        Inject a new risk multiplier λ from the DeepSeek-R1 LLM narrative layer.

        Parameters
        ----------
        new_lambda : float
            Risk sensitivity scalar ≥ 0.  Higher values punish LVR more
            heavily, pushing the agent toward wider (defensive) ranges.
            Typical range: [0.5, 3.0].  Values > 5.0 are clipped.
        """
        if new_lambda < 0:
            log.warning("new_lambda=%.4f is negative; clipping to 0.", new_lambda)
            new_lambda = 0.0
        if new_lambda > 5.0:
            log.warning("new_lambda=%.4f exceeds 5.0; clipping.", new_lambda)
            new_lambda = 5.0
        self._lambda = new_lambda
        log.debug("λ updated: %.4f", self._lambda)

    def get_lambda(self) -> float:
        """Return the current active risk multiplier."""
        return self._lambda

    # ── Action mask (Hysteresis Gas Gate) ───────────────────────

    def action_masks(self) -> np.ndarray:
        """
        Boolean mask for use with MaskablePPO (sb3-contrib).

        Action 0 (Hold) is always valid.

        For each rebalancing action 1/2/3, the mask evaluates the fee income
        the agent would earn if it set the *proposed* new range at the next
        step.  The action is allowed only when:

            Fees(proposed_range) − λ × LVR_next  ≥  gas_fee

        Using the proposed range (rather than the current range) lets the gas
        gate distinguish between a profitable narrow rebalance and an
        unprofitable wide one within the same market step.
        """
        if self._t + 1 >= self._n_steps:
            return np.array([True, False, False, False], dtype=bool)

        next_row = self._data[self._t + 1]
        next_sigma = self._get_col(next_row, "sigma")
        next_lvr   = self._get_col(next_row, "lvr_proxy")
        next_price = self._get_col(next_row, "price")
        next_volume = self._get_col(next_row, "volume")

        lam = max(self._lambda, 1e-3)
        lvr_cost = self._lambda * next_lvr

        mask = [True, False, False, False]   # Hold always allowed
        for action_id, pct in RANGE_PCT.items():
            if action_id == 0 or pct is None:
                continue
            # Compute bounds of the range this action would create
            lower_mult = max(0.0, 1.0 - pct * (lam ** 2))
            upper_mult = 1.0 + pct / lam
            prop_lower = next_price * lower_mult
            prop_upper = next_price * upper_mult

            fee_est = self._estimate_fees(
                next_price, next_sigma,
                prop_lower, prop_upper,
                volume=next_volume,
            )
            net_benefit = fee_est - lvr_cost
            if net_benefit >= self.gas_fee:
                mask[action_id] = True

        return np.array(mask, dtype=bool)

    # ── Gymnasium core ───────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        # Allow random start within first 20 % of data for training variety
        if options and options.get("random_start", False):
            max_start = max(1, int(self._n_steps * 0.20))
            self._t = int(self.np_random.integers(0, max_start))
        else:
            self._t = 0

        # Reset episode state
        self._done = False
        self._accumulated_fees = 0.0
        self._steps_since_rebal = 0
        self._portfolio_value = self.initial_capital
        self._net_pnl = 0.0
        self._lambda = self._default_lambda
        self._episode_rewards = []
        # Evaluation counters
        self._total_fees = 0.0
        self._total_gas = 0.0
        self._total_rebalances = 0
        self._hodl_reference = self.initial_capital

        # Initialise range at Mid (±5 %) around first price
        price = self._get_col(self._data[self._t], "price")
        self._initial_price = price
        self._set_range(price, pct=0.05)

        obs = self._build_obs()
        info = self._build_info()
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Advance the environment by one hourly timestep.

        Parameters
        ----------
        action : int  ∈ {0, 1, 2, 3}
            LP range decision from the PPO agent.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        if self._done:
            raise RuntimeError("Call reset() before stepping a terminated episode.")

        row = self._data[self._t]
        price = self._get_col(row, "price")
        sigma = self._get_col(row, "sigma")
        lvr = self._get_col(row, "lvr_proxy")
        volume = self._get_col(row, "volume")

        # ── 1. Hysteresis check before executing rebalance ────────
        gas_penalty = 0.0
        rebalanced = False

        if action != 0:
            mask = self.action_masks()
            if not mask[action]:
                # Net benefit < gas fee → penalise, keep current range
                gas_penalty = GAS_PENALTY_SCALE * self.gas_fee
                action = 0   # softly override to Hold
            else:
                # Execute rebalance with asymmetric defensive range
                new_pct = RANGE_PCT[action]
                self._set_range(price, pct=new_pct, asymmetric=True)
                self._steps_since_rebal = 0
                self._total_rebalances += 1
                rebalanced = True

        # ── 2. Fee accrual (Uniswap v3 in-range rule) ─────────────
        in_range = self._price_lower <= price <= self._price_upper
        fees_t = 0.0
        if in_range:
            fees_t = self._estimate_fees(price, sigma,
                                         self._price_lower, self._price_upper,
                                         volume=volume)
            self._accumulated_fees += fees_t

        # ── 3. Gas cost ───────────────────────────────────────────
        gas_t = self.gas_fee if rebalanced else 0.0

        # ── 4. ALVRO Reward formula ───────────────────────────────
        #   Reward = Fees_t − (lambda_t × LVR_t) − Gas_t
        lvr_penalty = self._lambda * lvr
        reward = fees_t - lvr_penalty - gas_t - gas_penalty

        # ── 5. Portfolio & PnL tracking ───────────────────────────
        self._portfolio_value += reward
        self._net_pnl += reward
        self._episode_rewards.append(reward)
        self._steps_since_rebal += 1
        # Evaluation counters
        self._total_fees += fees_t
        self._total_gas += gas_t

        # ── 6. Advance timestep ───────────────────────────────────
        self._t += 1
        terminated = self._t >= min(self._n_steps - 1, self.max_steps)
        truncated = False
        self._done = terminated or truncated

        obs = self._build_obs()
        info = self._build_info(
            fees_t=fees_t, lvr_t=lvr, lvr_penalty=lvr_penalty,
            gas_t=gas_t, gas_penalty=gas_penalty,
            reward=reward, action=action,
            in_range=in_range, rebalanced=rebalanced,
            price=price,                        # current step price, not next
            price_lower=self._price_lower,
            price_upper=self._price_upper,
        )

        if self.render_mode == "human":
            self._render_console(info)

        return obs, float(reward), terminated, truncated, info

    # ── Render ────────────────────────────────────────────────────

    def render(self) -> None:
        if self.render_mode == "human":
            row = self._data[self._t]
            price = self._get_col(row, "price")
            log.info(
                "[t=%5d] P=%.2f  [%.2f, %.2f]  λ=%.3f  PnL=%+.2f",
                self._t, price,
                self._price_lower, self._price_upper,
                self._lambda, self._net_pnl,
            )

    def close(self) -> None:
        pass

    # ── Internal helpers ──────────────────────────────────────────

    def _set_range(
        self,
        price: float,
        pct: float,
        asymmetric: bool = True,
    ) -> None:
        """
        Update [lower, upper] liquidity bounds around `price`.

        Asymmetric Defensive Range:
        ────────────────────────────────────────────
        Instead of a symmetric ±pct band, skew using the risk multiplier λ:

            P_L = P × max(0, 1 − pct × λ²)   ← aggressive crash defence
            P_H = P × (1 + pct / λ)           ← compressed upside during risk

        When λ = 1 the bounds reduce to the conventional symmetric ±pct form.
        When λ > 1 (high risk) the lower bound expands and upper compresses,
        creating a one-sided defensive shield.

        Parameters
        ----------
        price : float
            Current mid-price (USDC).
        pct : float
            Half-width fraction (e.g. 0.05 for ±5%).
        asymmetric : bool
            If True, apply the λ-skewed formula. If False, use symmetric ±pct.
        """
        if not asymmetric or self._lambda <= 0:
            self._price_lower = price * (1.0 - pct)
            self._price_upper = price * (1.0 + pct)
        else:
            lam = max(self._lambda, 1e-3)
            lower_mult = max(0.0, 1.0 - pct * (lam ** 2))
            upper_mult = 1.0 + pct / lam
            self._price_lower = price * lower_mult
            self._price_upper = price * upper_mult

        self._price_center = price
        self._current_range_pct = pct

    def _estimate_fees(
        self,
        price: float,
        sigma: float,
        lower: float,
        upper: float,
        volume: float = 0.0,
    ) -> float:
        """
        Estimate fee income for one hourly step assuming price is in range.

        When pool volume is available (volume > 0):
            Fees = fee_tier * volume * capital_fraction * L_eff
            where capital_fraction = initial_capital / _POOL_TVL_REF
            and   L_eff = 1 / range_width  (concentration bonus)

        Fallback when volume is zero or unavailable:
            Fees = fee_tier * (sigma * price) * L_eff
        """
        if lower >= upper or lower <= 0:
            return 0.0

        range_width = (upper - lower) / (price + 1e-9)
        if range_width <= 0:
            return 0.0

        l_eff = 1.0 / range_width

        if volume > 1.0:
            capital_fraction = self.initial_capital / _POOL_TVL_REF
            fees = self.fee_tier * volume * capital_fraction * l_eff
        else:
            # Capital-scaled fallback: fees proportional to position size.
            # fee_tier * sigma represents the fraction of notional earned per step;
            # multiplying by initial_capital gives USD fee income for our position.
            fees = self.initial_capital * self.fee_tier * sigma * l_eff

        return max(fees, 0.0)

    def _build_obs(self) -> np.ndarray:
        """
        Construct the 14-dimensional observation vector.

        Observation index map
        ---------------------
        [0]  price_rel          (price − center) / price
        [1]  sigma              GARCH vol (raw)
        [2]  lvr_proxy          LVR estimate (raw decimal)
        [3]  lambda_risk        current λ
        [4]  accumulated_fees   unrealised fees / _FEE_NORM
        [5]  tick_lower_norm    lower / price − 1
        [6]  tick_upper_norm    upper / price − 1
        [7]  log_return         most recent log return
        [8]  rsi_14             RSI / 100   (→ [0,1])
        [9]  ma_24h_rel         (ma − price) / price
        [10] portfolio_value    portfolio / initial_capital  (starts at 1.0)
        [11] in_range           binary float
        [12] steps_since_rebal  / _STEPS_NORM
        [13] net_pnl            net_pnl / _PNL_NORM
        """
        t = min(self._t, self._n_steps - 1)
        row = self._data[t]

        price       = self._get_col(row, "price")
        sigma       = self._get_col(row, "sigma")
        lvr         = self._get_col(row, "lvr_proxy")
        log_ret     = self._get_col(row, "log_return")
        rsi         = self._get_col(row, "rsi_14")
        ma_24h      = self._get_col(row, "ma_24h")

        center = self._price_center if self._price_center > 0 else price
        in_range = float(self._price_lower <= price <= self._price_upper)

        obs = np.array([
            (price - center) / (price + 1e-9),                      # [0]
            sigma,                                                    # [1]
            lvr,                                                      # [2]
            self._lambda,                                             # [3]
            self._accumulated_fees / _FEE_NORM,                      # [4]
            self._price_lower / (price + 1e-9) - 1.0,               # [5]
            self._price_upper / (price + 1e-9) - 1.0,               # [6]
            log_ret,                                                  # [7]
            rsi / 100.0,                                             # [8]
            (ma_24h - price) / (price + 1e-9),                      # [9]
            self._portfolio_value / self.initial_capital,            # [10]
            in_range,                                                # [11]
            self._steps_since_rebal / _STEPS_NORM,                  # [12]
            self._net_pnl / _PNL_NORM,                              # [13]
        ], dtype=np.float32)

        # Clip and nan-guard
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        return obs

    def _build_info(self, **kwargs) -> Dict[str, Any]:
        """Return an info dict with full episode diagnostics."""
        t = min(self._t, self._n_steps - 1)
        row = self._data[t]
        price = self._get_col(row, "price")

        info: Dict[str, Any] = {
            "step":               self._t,
            "price":              price,
            "price_lower":        self._price_lower,
            "price_upper":        self._price_upper,
            "lambda":             self._lambda,
            "initial_price":      self._initial_price,
            "accumulated_fees":   self._accumulated_fees,
            "total_fees":         self._total_fees,
            "total_gas":          self._total_gas,
            "total_rebalances":   self._total_rebalances,
            "portfolio_value":    self._portfolio_value,
            "hodl_reference":     self._hodl_reference,
            "net_pnl":            self._net_pnl,
            "steps_since_rebal":  self._steps_since_rebal,
            "current_range_pct":  self._current_range_pct,
        }
        info.update(kwargs)
        return info

    def _get_col(self, row: np.ndarray, col_name: str) -> float:
        """Safely read a named column from a data row."""
        idx = self._col.get(col_name)
        if idx is None:
            return 0.0
        val = row[idx]
        return float(val) if np.isfinite(val) else 0.0

    @staticmethod
    def _render_console(info: Dict[str, Any]) -> None:
        print(
            f"[t={info['step']:5d}] "
            f"P={info['price']:.2f} "
            f"[{info['price_lower']:.2f}, {info['price_upper']:.2f}] "
            f"λ={info['lambda']:.3f} "
            f"fees={info.get('fees_t', 0):.4f} "
            f"lvr={info.get('lvr_t', 0):.4f} "
            f"gas={info.get('gas_t', 0):.2f} "
            f"R={info.get('reward', 0):+.4f} "
            f"PnL={info['net_pnl']:+.2f}"
        )

    # ── Episode metrics helpers ───────────────────────────────────

    def episode_sharpe(self) -> float:
        """Compute the in-episode Sharpe Ratio from per-step rewards."""
        r = np.array(self._episode_rewards)
        if len(r) < 2:
            return 0.0
        mean_r = np.mean(r)
        std_r = np.std(r, ddof=1) + 1e-9
        # Annualise: ×√8760 (hourly → annual)
        return float(mean_r / std_r * math.sqrt(8760))

    def episode_max_drawdown(self) -> float:
        """Compute maximum drawdown of cumulative reward over episode."""
        r = np.array(self._episode_rewards)
        if len(r) == 0:
            return 0.0
        cum = np.cumsum(r)
        roll_max = np.maximum.accumulate(cum)
        drawdown = roll_max - cum
        return float(np.max(drawdown))

    def gas_efficiency(self) -> float:
        """
        Gas Efficiency = Total Fees Earned / Total Gas Spent.
        Returns inf if no gas was ever spent (pure Hold strategy).
        """
        if self._total_gas < 1e-9:
            return float("inf")
        return self._total_fees / self._total_gas

    def lp_to_hodl_ratio(self, current_price: Optional[float] = None) -> float:
        """
        LP-to-HODL Ratio = LP Portfolio Value / HODL Portfolio Value.

        HODL model: 50% WETH + 50% USDC held at the episode start price P0.
            V_HODL(t) = V0 * (0.5 * P_t/P0 + 0.5)

        Parameters
        ----------
        current_price : float | None
            Price at evaluation point. If None, uses the last seen price.
        """
        if current_price is None:
            t = min(self._t, self._n_steps - 1)
            current_price = self._get_col(self._data[t], "price")

        if self._initial_price <= 0:
            return 1.0

        hodl_value = self._hodl_reference * (
            0.5 * (current_price / self._initial_price) + 0.5
        )
        if hodl_value < 1e-9:
            return 1.0
        return self._portfolio_value / hodl_value


# ─────────────────────────────────────────────────────────────────
# Factory helper
# ─────────────────────────────────────────────────────────────────

def make_alvro_env(
    parquet_path: str = "data/processed_market.parquet",
    **env_kwargs,
) -> ALVROEnv:
    """
    Convenience factory: load the processed Parquet and return a ready ALVROEnv.

    Parameters
    ----------
    parquet_path : str
        Path to the processed market Parquet (from data/processor.py).
    **env_kwargs
        Forwarded to ALVROEnv.__init__.

    Example
    -------
    >>> env = make_alvro_env()
    >>> obs, info = env.reset()
    """
    df = pd.read_parquet(parquet_path)
    return ALVROEnv(market_data=df, **env_kwargs)


# ─────────────────────────────────────────────────────────────────
# Gymnasium registration
# ─────────────────────────────────────────────────────────────────

gym.register(
    id="ALVRO-v3",
    entry_point="env.alvro_env:ALVROEnv",
    kwargs={},
)


# ─────────────────────────────────────────────────────────────────
# Quick smoke-test (run directly)
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    parquet = Path("data/processed_market.parquet")
    if not parquet.exists():
        log.error(
            "processed_market.parquet not found. Run data/processor.py first."
        )
        sys.exit(1)

    log.info("Loading market data …")
    df = pd.read_parquet(parquet)
    log.info("Loaded %d rows × %d cols", len(df), len(df.columns))

    env = ALVROEnv(market_data=df, render_mode="human")
    obs, info = env.reset(seed=42)
    log.info("Initial obs shape: %s", obs.shape)
    log.info("Observation space: %s", env.observation_space)
    log.info("Action space:      %s", env.action_space)
    log.info("Initial λ:         %.3f", env.get_lambda())

    total_reward = 0.0
    for step_i in range(50):
        # Inject a DeepSeek risk update every 10 steps
        if step_i % 10 == 0:
            new_lambda = 1.0 + 0.5 * np.sin(step_i / 5.0)
            env.update_external_risk(new_lambda)

        mask = env.action_masks()
        valid_actions = np.where(mask)[0]
        action = int(np.random.choice(valid_actions))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            log.info("Episode ended at step %d", step_i)
            break

    log.info("─" * 50)
    log.info("50-step smoke test complete.")
    log.info("Total reward:    %+.4f", total_reward)
    log.info("Sharpe ratio:    %.4f", env.episode_sharpe())
    log.info("Max drawdown:    %.4f", env.episode_max_drawdown())
    log.info("Portfolio value: %.2f", info["portfolio_value"])
    log.info("Net PnL:         %+.2f", info["net_pnl"])
