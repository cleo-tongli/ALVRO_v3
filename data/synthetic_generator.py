"""
ALVRO_v3 — Synthetic Market Data Generator
============================================
Module: data/synthetic_generator.py

Fallback when the real Uniswap v3 dataset is unavailable (network issues,
contest environment, CI, etc.).  Generates realistic WETH/USDC-like hourly
price paths using a GARCH(1,1)-driven Geometric Brownian Motion, then runs
the same feature-engineering pipeline as the real processor.

The output is a Parquet file with identical schema to processed_market.parquet:
    timestamp, price, log_return, sigma, gamma, lvr_proxy, volume, rsi_14, ma_24h

Usage
-----
    # From the CLI:
    python -m data.synthetic_generator

    # Or programmatically:
    from data.synthetic_generator import SyntheticMarketGenerator
    df = SyntheticMarketGenerator(n_hours=26_280).run()

    # With custom output path:
    SyntheticMarketGenerator(
        n_hours=8_760,       # 1 year
        start_price=2_000.0,
        seed=7,
        output_path="data/processed_market.parquet",
    ).run()
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("SyntheticMarketGenerator")

# ─── Default constants ────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_DEFAULT_OUTPUT = _HERE / "processed_market.parquet"

_N_HOURS:       int   = 26_280        # 3 years of hourly bars
_START_PRICE:   float = 2_500.0       # WETH initial price in USDC
_MU:            float = 1e-4          # hourly drift ≈ +87 % / year
_SIGMA_BASE:    float = 0.018         # hourly base vol ≈ 58 % annual
_RANGE_PCT:     float = 0.05          # ± 5 % range for Gamma / LVR
_RSI_PERIOD:    int   = 14
_MA_WINDOW:     int   = 24

# GARCH(1,1) parameters (ω, α, β) calibrated to ETH 2021-2024
_OMEGA: float = 1e-6
_ALPHA: float = 0.08
_BETA:  float = 0.91


class SyntheticMarketGenerator:
    """
    Generate realistic synthetic WETH/USDC hourly data and export a
    processed Parquet identical in schema to the real data pipeline.

    The price path is driven by a GARCH(1,1) volatility process:
        σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
        log(P_t / P_{t-1}) = μ + σ_t · Z_t    Z_t ~ N(0,1)

    Parameters
    ----------
    n_hours : int
        Number of hourly bars to generate.
    start_price : float
        Initial WETH price in USDC.
    mu : float
        Per-step (hourly) drift.
    sigma_base : float
        Initial conditional volatility (annualise × √8760 for yearly).
    omega, alpha, beta : float
        GARCH(1,1) parameters.
    range_pct : float
        Symmetric half-width for Gamma/LVR estimation.
    seed : int
        NumPy random seed.
    output_path : Path | str | None
        Parquet destination. None → do not save automatically.
    """

    def __init__(
        self,
        n_hours:     int   = _N_HOURS,
        start_price: float = _START_PRICE,
        mu:          float = _MU,
        sigma_base:  float = _SIGMA_BASE,
        omega:       float = _OMEGA,
        alpha:       float = _ALPHA,
        beta:        float = _BETA,
        range_pct:   float = _RANGE_PCT,
        seed:        int   = 42,
        output_path: Optional[Path | str] = _DEFAULT_OUTPUT,
    ) -> None:
        self.n_hours     = n_hours
        self.start_price = start_price
        self.mu          = mu
        self.sigma_base  = sigma_base
        self.omega       = omega
        self.alpha       = alpha
        self.beta        = beta
        self.range_pct   = range_pct
        self.seed        = seed
        self.output_path = Path(output_path) if output_path else None

    # ── Price path ───────────────────────────────────────────────

    def _simulate_garch_gbm(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate GARCH(1,1)-driven log-returns and reconstruct price.

        Returns
        -------
        prices : (n_hours,) ndarray  USDC price of WETH
        sigmas : (n_hours,) ndarray  per-step conditional volatility
        """
        rng = np.random.default_rng(self.seed)
        n   = self.n_hours

        log_returns = np.zeros(n)
        sigmas      = np.zeros(n)
        prices      = np.zeros(n)

        # Initialise
        h_prev     = self.sigma_base ** 2   # initial conditional variance
        prices[0]  = self.start_price
        sigmas[0]  = self.sigma_base

        # Regime-switching: inject up to 3 random volatility spikes (crisis events).
        # Guard: only inject when there is a safe interior window to choose from.
        margin    = max(10, n // 10)          # 10 % of series as edge buffer
        spike_lo  = margin
        spike_hi  = n - margin
        n_spikes  = 3
        spike_set: set[int] = set()
        if spike_hi - spike_lo > n_spikes:
            spike_steps = rng.choice(
                range(spike_lo, spike_hi), size=n_spikes, replace=False
            )
            spike_set = set(spike_steps.tolist())

        for t in range(1, n):
            # Volatility spike injection
            if t in spike_set:
                h_prev = max(h_prev, (self.sigma_base * 4) ** 2)

            sigma_t = math.sqrt(max(h_prev, 1e-10))
            z_t     = rng.standard_normal()
            r_t     = self.mu + sigma_t * z_t

            # GARCH update: σ²_t+1 = ω + α·ε²_t + β·σ²_t
            h_next = (
                self.omega
                + self.alpha * (sigma_t * z_t) ** 2
                + self.beta  * h_prev
            )

            log_returns[t] = r_t
            sigmas[t]      = sigma_t
            prices[t]      = prices[t - 1] * math.exp(r_t)
            h_prev         = h_next

        # Ensure prices are strictly positive
        prices = np.maximum(prices, 1.0)
        return prices, sigmas, log_returns

    # ── Feature engineering ──────────────────────────────────────

    @staticmethod
    def _rsi(prices: pd.Series, period: int = _RSI_PERIOD) -> pd.Series:
        delta    = prices.diff()
        gain     = delta.clip(lower=0.0)
        loss     = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        rs       = avg_gain / avg_loss.replace(0.0, np.nan)
        return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

    @staticmethod
    def _gamma_series(prices: pd.Series, range_pct: float) -> pd.Series:
        pu    = prices * (1.0 + range_pct)
        pl    = prices * (1.0 - range_pct)
        gamma = 1.0 / (prices * (np.sqrt(pu / pl) - 1.0))
        return gamma

    @staticmethod
    def _lvr(sigma: pd.Series, price: pd.Series, gamma: pd.Series) -> pd.Series:
        """LVR = 0.5 · σ² · P · Γ · dt  (dt = 1 hour)"""
        return 0.5 * (sigma ** 2) * price * gamma * 1.0

    # ── Volume ───────────────────────────────────────────────────

    def _synthetic_volume(self, prices: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
        """
        Simulate USD volume: higher during volatile periods, log-normal base.
        """
        rng = np.random.default_rng(self.seed + 1)
        base_vol = np.exp(rng.normal(14.0, 1.5, size=len(prices)))  # ~$1M base
        vol_mult = 1.0 + 5.0 * sigmas / sigmas.mean()               # vol spike multiplier
        return base_vol * vol_mult

    # ── Main pipeline ────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Generate the full synthetic dataset, engineer features, and optionally
        save to Parquet.

        Returns
        -------
        pd.DataFrame  with columns identical to the real data processor output.
        """
        log.info("═══ SyntheticMarketGenerator — START ═══")
        log.info(
            "Params: n_hours=%d  P₀=%.2f  μ=%.1e  σ₀=%.4f  seed=%d",
            self.n_hours, self.start_price, self.mu, self.sigma_base, self.seed,
        )

        # 1. Simulate price path
        log.info("Simulating GARCH(1,1)-driven GBM price path …")
        prices, sigmas, log_returns = self._simulate_garch_gbm()

        # 2. Build timestamps (hourly, starting 2021-01-01 UTC)
        timestamps = pd.date_range(
            start="2021-01-01 00:00:00",
            periods=self.n_hours,
            freq="h",
            tz="UTC",
        )

        # 3. Assemble base DataFrame
        df = pd.DataFrame({
            "timestamp":  timestamps,
            "price":      prices,
            "log_return": log_returns,
            "sigma":      sigmas,
            "volume":     self._synthetic_volume(prices, sigmas),
        })

        # 4. Feature engineering (identical to real processor)
        log.info("Engineering Gamma, LVR, RSI, MA …")
        price_s      = df["price"]
        sigma_s      = df["sigma"]
        gamma_s      = self._gamma_series(price_s, self.range_pct)
        df["gamma"]      = gamma_s.values
        df["lvr_proxy"]  = self._lvr(sigma_s, price_s, gamma_s).values
        df["rsi_14"]     = self._rsi(price_s).values
        df["ma_24h"]     = price_s.rolling(window=_MA_WINDOW, min_periods=1).mean().values

        # 5. Select & order final columns (match real processor exactly)
        final_cols = [
            "timestamp", "price", "log_return", "sigma",
            "gamma", "lvr_proxy", "volume", "rsi_14", "ma_24h",
        ]
        df = df[final_cols].copy()

        # 6. Sanity checks
        assert (df["price"] > 0).all(),    "Negative prices detected!"
        assert (df["sigma"] > 0).all(),    "Zero/negative sigma detected!"
        assert (df["lvr_proxy"] > 0).all(), "Non-positive LVR detected!"
        log.info("Sanity checks passed ✓")

        # 7. Summary stats
        log.info(
            "Price range: %.2f → %.2f (%.0f%% total move)",
            df["price"].min(), df["price"].max(),
            100 * (df["price"].iloc[-1] / df["price"].iloc[0] - 1),
        )
        log.info("Mean σ: %.5f | Max σ: %.5f", df["sigma"].mean(), df["sigma"].max())
        log.info("Mean LVR: %.6f", df["lvr_proxy"].mean())

        # 8. Save
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self.output_path, index=False, engine="pyarrow")
            log.info("✔ Saved → %s  (%d rows × %d cols)", self.output_path, len(df), len(df.columns))

        log.info("═══ SyntheticMarketGenerator — DONE ═══")
        return df


# ── CLI ──────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Generate synthetic WETH/USDC market data for ALVRO_v3."
    )
    parser.add_argument("--n-hours",     type=int,   default=_N_HOURS)
    parser.add_argument("--start-price", type=float, default=_START_PRICE)
    parser.add_argument("--mu",          type=float, default=_MU)
    parser.add_argument("--sigma-base",  type=float, default=_SIGMA_BASE)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument(
        "--output", type=str, default=str(_DEFAULT_OUTPUT),
        help="Output Parquet path",
    )
    args = parser.parse_args()

    gen = SyntheticMarketGenerator(
        n_hours=args.n_hours,
        start_price=args.start_price,
        mu=args.mu,
        sigma_base=args.sigma_base,
        seed=args.seed,
        output_path=Path(args.output),
    )
    gen.run()


if __name__ == "__main__":
    main()
