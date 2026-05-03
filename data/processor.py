"""
ALVRO_v3 — Data Processor
==========================
Module: data/processor.py

Pipeline:
  1. Clone / pull the Uniswap v3 reference repo and locate the raw CSV.
  2. Load, validate, and clean hourly WETH/USDC price data.
  3. Fit a GARCH(1,1) model → conditional_volatility (sigma).
  4. Estimate the LVR proxy:  LVR = 0.5 * σ² * P * Γ * dt
     where Γ is derived from concentrated-liquidity geometry.
  5. Engineer RSI(14) and 24-h Moving Average momentum signals.
  6. Export the final feature frame to data/processed_market.parquet.

Usage:
    python -m data.processor
    # or
    from data.processor import ALVRODataProcessor
    df = ALVRODataProcessor().run()
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from arch import arch_model

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ALVRODataProcessor")

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent          # ALVRO_v3/data/
_ROOT = _HERE.parent                             # ALVRO_v3/
_REPO_DIR = _HERE / "deeprl-liquidity-provision-uniswapv3"
_REPO_URL = "https://github.com/Alessiobrini/deeprl-liquidity-provision-uniswapv3"
_OUTPUT_PATH = _HERE / "processed_market.parquet"


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
# Concentrated-liquidity default range: ±5 % around current price
_RANGE_PCT: float = 0.05          # 5 % half-width → price_upper = P*(1+r), price_lower = P*(1−r)
_DT: float = 1.0                  # hourly step, dt = 1 hour (annualise later if needed)
_RSI_PERIOD: int = 14
_MA_WINDOW: int = 24              # 24-hour simple moving average
_GARCH_P: int = 1
_GARCH_Q: int = 1


# ──────────────────────────────────────────────────────────────────────────────
class ALVRODataProcessor:
    """
    End-to-end data processing pipeline for ALVRO_v3.

    Parameters
    ----------
    repo_url : str
        URL of the reference GitHub repository.
    repo_dir : Path
        Local directory into which the repo will be cloned.
    output_path : Path
        Destination for the processed Parquet file.
    range_pct : float
        Half-width of the assumed liquidity range for Gamma estimation (default 5 %).
    """

    def __init__(
        self,
        repo_url: str = _REPO_URL,
        repo_dir: Path = _REPO_DIR,
        output_path: Path = _OUTPUT_PATH,
        range_pct: float = _RANGE_PCT,
    ) -> None:
        self.repo_url = repo_url
        self.repo_dir = Path(repo_dir)
        self.output_path = Path(output_path)
        self.range_pct = range_pct

    # ── 1. Repository access ─────────────────────────────────────────────────

    def _clone_or_pull_repo(self) -> None:
        """Clone the reference repo; if already present, pull latest changes."""
        if (self.repo_dir / ".git").exists():
            log.info("Repo already cloned at %s — pulling latest …", self.repo_dir)
            subprocess.run(
                ["git", "-C", str(self.repo_dir), "pull", "--ff-only"],
                check=True,
                capture_output=True,
            )
        else:
            log.info("Cloning %s → %s …", self.repo_url, self.repo_dir)
            subprocess.run(
                ["git", "clone", "--depth", "1", self.repo_url, str(self.repo_dir)],
                check=True,
                capture_output=True,
            )
        log.info("Repo ready.")

    def _find_csv(self) -> Path:
        """
        Locate the primary price/OHLCV data file inside the cloned repo.

        Strategy:
          1. Exclude files inside result/output directories (output, results,
             logs, figures, plots) — those are derived artefacts, not raw data.
          2. Score remaining files by filename keywords that signal raw time-series
             data: price, data, time, uni, hourly, 1h, ohlcv, swap, weth, usdc.
          3. Prefer shallower paths (ties broken by ascending depth).
        """
        _EXCLUDE_DIRS = {"output", "results", "logs", "figures", "plots", "images"}
        _DATA_KEYWORDS = (
            "price", "data", "time", "uni", "hourly", "1h",
            "ohlcv", "swap", "weth", "usdc", "eth",
        )

        candidates = (
            list(self.repo_dir.rglob("*.parquet"))
            + list(self.repo_dir.rglob("*.csv"))
        )
        if not candidates:
            raise FileNotFoundError(
                f"No CSV/Parquet files found inside {self.repo_dir}. "
                "Check the repo structure manually."
            )

        # Drop files that live inside a known result directory
        filtered = [
            p for p in candidates
            if not any(part.lower() in _EXCLUDE_DIRS for part in p.parts)
        ]
        if not filtered:
            log.warning("All candidates are in output/result dirs; falling back to full list.")
            filtered = candidates

        # Score by filename only (not full path) to avoid false boosts from repo name
        scored: list = []
        for path in filtered:
            name = path.name.lower()
            score = sum(1 for kw in _DATA_KEYWORDS if kw in name)
            depth = len(path.parts)
            scored.append((score, -depth, path))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        chosen = scored[0][2]
        log.info("Found data file: %s  (keyword score=%d)", chosen, scored[0][0])
        return chosen

    # ── 2. Load & validate ───────────────────────────────────────────────────

    def _load_raw(self, path: Path) -> pd.DataFrame:
        """Read raw market data; handle both Parquet and CSV automatically."""
        log.info("Loading raw data from %s …", path)
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, low_memory=False)

        log.info("Raw shape: %s | Columns: %s", df.shape, df.columns.tolist())
        return df

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardise column names to: [timestamp, price, volume].

        The reference repo columns vary; we handle the most common variants:
          - 'open_timestamp' / 'timestamp' / 'date' / 'time' / 'block_timestamp'
          - 'close' / 'price' / 'token0Price' / 'token1Price' / 'sqrtPriceX96'
          - 'volumeUSD' / 'volume' / 'volumeToken0' / 'volumeToken1'
        """
        df.columns = [c.strip().lower() for c in df.columns]

        # ── timestamp
        ts_candidates = ["open_timestamp", "timestamp", "date", "time",
                         "block_timestamp", "datetime", "period_start_unix"]
        for col in ts_candidates:
            if col in df.columns:
                df = df.rename(columns={col: "timestamp"})
                break
        else:
            # Use index if it looks like timestamps
            if hasattr(df.index, "dtype") and np.issubdtype(df.index.dtype, np.datetime64):
                df = df.reset_index().rename(columns={"index": "timestamp"})
            else:
                raise KeyError(
                    f"Cannot identify a timestamp column. Available: {df.columns.tolist()}"
                )

        # ── price (WETH price in USDC)
        price_candidates = ["close", "price", "token0price", "close_price",
                            "price_usd", "token1price", "weighted_price"]
        for col in price_candidates:
            if col in df.columns:
                df = df.rename(columns={col: "price"})
                break
        else:
            raise KeyError(
                f"Cannot identify a price column. Available: {df.columns.tolist()}"
            )

        # ── volume (optional — fill with 0 if absent)
        vol_candidates = ["volumeusd", "volume", "volumetoken0",
                          "volumetoken1", "vol"]
        for col in vol_candidates:
            if col in df.columns:
                df = df.rename(columns={col: "volume"})
                break
        else:
            log.warning("No volume column found — filling with 0.")
            df["volume"] = 0.0

        # Keep only core columns (plus extras we may use later)
        keep = ["timestamp", "price", "volume"] + [
            c for c in df.columns if c not in ("timestamp", "price", "volume")
        ]
        return df[keep].copy()

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Type-cast, sort, drop duplicates and NaN prices."""
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

        df = (
            df.dropna(subset=["timestamp", "price"])
            .drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # Sanity check: positive prices only
        n_before = len(df)
        df = df[df["price"] > 0].reset_index(drop=True)
        n_removed = n_before - len(df)
        if n_removed:
            log.warning("Removed %d rows with non-positive prices.", n_removed)

        log.info("Clean shape: %s | Date range: %s → %s",
                 df.shape, df["timestamp"].iloc[0], df["timestamp"].iloc[-1])
        return df

    # ── 3. Feature engineering ───────────────────────────────────────────────

    @staticmethod
    def _log_returns(prices: pd.Series) -> pd.Series:
        """Compute hourly log returns; first row is NaN → filled with 0."""
        lr = np.log(prices / prices.shift(1))
        return lr.fillna(0.0)

    # ── 3a. GARCH(1,1) ──────────────────────────────────────────────────────

    def _fit_garch(self, log_returns: pd.Series) -> pd.Series:
        """
        Fit a GARCH(1,1) model and return the conditional volatility series.

        The arch library expects returns in *percentage* space for numerical
        stability, so we scale by 100 and unscale before returning.
        """
        log.info("Fitting GARCH(%d,%d) on %d observations …",
                 _GARCH_P, _GARCH_Q, len(log_returns))

        returns_pct = log_returns * 100.0  # scale to % for numerical stability

        am = arch_model(
            returns_pct,
            vol="Garch",
            p=_GARCH_P,
            q=_GARCH_Q,
            dist="Normal",
            rescale=False,
        )
        res = am.fit(disp="off", show_warning=False)

        # conditional_volatility is in % → convert back to decimal
        sigma_pct = res.conditional_volatility
        sigma = sigma_pct / 100.0

        log.info(
            "GARCH fit complete. σ̄ = %.6f | ω=%.4e α=%.4f β=%.4f",
            sigma.mean(),
            res.params["omega"],
            res.params["alpha[1]"],
            res.params["beta[1]"],
        )
        return pd.Series(sigma.values, index=log_returns.index, name="sigma")

    # ── 3b. Gamma (concentrated liquidity approximation) ────────────────────

    @staticmethod
    def calculate_gamma(price: float, range_pct: float = _RANGE_PCT) -> float:
        """
        Approximate second-order price sensitivity (Gamma) for a Uniswap v3
        position with a symmetric ±range_pct range around the current price.

        Derivation (Milionis et al., 2022 / Ottina p.161):
            price_upper = P * (1 + r)
            price_lower = P * (1 − r)
            Γ ≈ 1 / [P · (√(Pu/Pl) − 1)]

        Parameters
        ----------
        price : float
            Current mid-price (USDC per WETH).
        range_pct : float
            Half-width of the liquidity range (default 0.05 → ±5 %).

        Returns
        -------
        float
            Gamma value (1/USDC).
        """
        price_upper = price * (1.0 + range_pct)
        price_lower = price * (1.0 - range_pct)
        gamma = 1.0 / (price * (math.sqrt(price_upper / price_lower) - 1.0))
        return gamma

    @staticmethod
    def _gamma_series(prices: pd.Series, range_pct: float) -> pd.Series:
        """Vectorised Gamma for every price row."""
        price_upper = prices * (1.0 + range_pct)
        price_lower = prices * (1.0 - range_pct)
        gamma = 1.0 / (prices * (np.sqrt(price_upper / price_lower) - 1.0))
        return gamma.rename("gamma")

    # ── 3c. LVR proxy ────────────────────────────────────────────────────────

    def calculate_lvr(
        self,
        sigma: pd.Series,
        price: pd.Series,
        gamma: pd.Series,
        dt: float = _DT,
    ) -> pd.Series:
        """
        Compute hourly LVR proxy.

        Formula (Milionis et al., 2022):
            LVR = 0.5 · σ² · P · Γ · dt

        Parameters
        ----------
        sigma : pd.Series
            Conditional volatility from GARCH(1,1) [decimal, per-hour].
        price : pd.Series
            Mid-price series (USDC).
        gamma : pd.Series
            Concentrated-liquidity Gamma series.
        dt : float
            Time step in hours (default 1.0).

        Returns
        -------
        pd.Series  — lvr_proxy (USDC per hour per unit of liquidity)
        """
        lvr = 0.5 * (sigma ** 2) * price * gamma * dt
        return lvr.rename("lvr_proxy")

    # ── 3d. RSI ──────────────────────────────────────────────────────────────

    @staticmethod
    def _rsi(prices: pd.Series, period: int = _RSI_PERIOD) -> pd.Series:
        """
        Relative Strength Index using Wilder's smoothing (EMA with α = 1/period).
        Returns values in [0, 100].
        """
        delta = prices.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0).rename(f"rsi_{period}")

    # ── 3e. Moving average ───────────────────────────────────────────────────

    @staticmethod
    def _ma(prices: pd.Series, window: int = _MA_WINDOW) -> pd.Series:
        """Simple Moving Average over `window` hourly bars."""
        return prices.rolling(window=window, min_periods=1).mean().rename(f"ma_{window}h")

    # ── 4. Full pipeline ─────────────────────────────────────────────────────

    def run(self, skip_clone: bool = False) -> pd.DataFrame:
        """
        Execute the full ingestion → feature-engineering → export pipeline.

        Parameters
        ----------
        skip_clone : bool
            Set True to skip the git clone/pull step (e.g. if already present).

        Returns
        -------
        pd.DataFrame
            Processed feature frame saved to self.output_path.
        """
        log.info("═══ ALVRO_v3 Data Processor — START ═══")

        # ── Step 1: acquire data ──────────────────────────────────────────
        if not skip_clone:
            self._clone_or_pull_repo()
        csv_path = self._find_csv()

        # ── Step 2: load + clean ──────────────────────────────────────────
        df = self._load_raw(csv_path)
        df = self._normalize_columns(df)
        df = self._clean(df)

        # ── Step 3: log returns ───────────────────────────────────────────
        log.info("Computing log returns …")
        df["log_return"] = self._log_returns(df["price"])

        # ── Step 4: GARCH(1,1) conditional volatility ─────────────────────
        df["sigma"] = self._fit_garch(df["log_return"])

        # ── Step 5: Gamma + LVR proxy ─────────────────────────────────────
        log.info("Computing Gamma and LVR proxy …")
        gamma_series = self._gamma_series(df["price"], self.range_pct)
        df["gamma"] = gamma_series.values
        df["lvr_proxy"] = self.calculate_lvr(df["sigma"], df["price"], gamma_series).values

        # ── Step 6: Momentum features ─────────────────────────────────────
        log.info("Computing RSI(%d) and MA(%dh) …", _RSI_PERIOD, _MA_WINDOW)
        df[f"rsi_{_RSI_PERIOD}"] = self._rsi(df["price"]).values
        df[f"ma_{_MA_WINDOW}h"] = self._ma(df["price"]).values

        # ── Step 7: Select final columns ──────────────────────────────────
        final_cols = [
            "timestamp",
            "price",
            "log_return",
            "sigma",
            "gamma",
            "lvr_proxy",
            "volume",
            f"rsi_{_RSI_PERIOD}",
            f"ma_{_MA_WINDOW}h",
        ]
        df_final = df[final_cols].copy()

        # ── Step 8: Export ────────────────────────────────────────────────
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_parquet(self.output_path, index=False, engine="pyarrow")
        log.info("✔ Processed data saved → %s  (%d rows × %d cols)",
                 self.output_path, len(df_final), len(df_final.columns))

        # ── Summary statistics ────────────────────────────────────────────
        self._print_summary(df_final)

        log.info("═══ ALVRO_v3 Data Processor — DONE ═══")
        return df_final

    @staticmethod
    def _print_summary(df: pd.DataFrame) -> None:
        """Log a concise descriptive summary of the processed dataset."""
        log.info("\n%s", "─" * 60)
        log.info("FEATURE SUMMARY")
        log.info("─" * 60)
        for col in df.select_dtypes(include=[np.number]).columns:
            s = df[col]
            log.info(
                "  %-20s  mean=%12.6f  std=%12.6f  min=%12.6f  max=%12.6f",
                col, s.mean(), s.std(), s.min(), s.max(),
            )
        log.info("─" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="ALVRO_v3 Data Processor — ingest, model, and export Uniswap v3 market data."
    )
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Skip the git clone/pull step (use existing repo).",
    )
    parser.add_argument(
        "--range-pct",
        type=float,
        default=_RANGE_PCT,
        help=f"Liquidity range half-width for Gamma estimation (default {_RANGE_PCT}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_OUTPUT_PATH),
        help=f"Output Parquet file path (default: {_OUTPUT_PATH}).",
    )
    args = parser.parse_args()

    processor = ALVRODataProcessor(
        output_path=Path(args.output),
        range_pct=args.range_pct,
    )
    processor.run(skip_clone=args.skip_clone)


if __name__ == "__main__":
    main()
