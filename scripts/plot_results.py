"""
ALVRO_v3 — Results Visualisation
===================================
Module: scripts/plot_results.py

Generates a comprehensive 6-panel dark-themed plot from:
  1. A trained episode CSV  (output of scripts/evaluate.py --output)
  2. An SB3 Monitor CSV    (logs/run_xx/train_monitor.csv)

Panels
------
  [0] Price + LP Range (price with shaded in-range bands)
  [1] Cumulative P&L   (LP vs HODL)
  [2] Reward per step  (coloured by action)
  [3] λ (DeepSeek risk signal) over time
  [4] Fee vs LVR Penalty decomposition
  [5] Training: episode reward curve (from Monitor CSV)

Usage
-----
    # Minimal (episode CSV only):
    python -m scripts.plot_results \\
        --episode logs/results/eval_trajectory.csv \\
        --output  logs/plots/dashboard.png

    # With training monitor:
    python -m scripts.plot_results \\
        --episode logs/results/eval_trajectory.csv \\
        --monitor logs/run_01/train_monitor.csv \\
        --output  logs/plots/dashboard.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("plot_results")

# ─── Design tokens (dark cyberpunk palette) ────────────────────────
BG      = "#0d0d1a"
BG2     = "#12122a"
GRID    = "#1e1e3a"
TEXT    = "#c8c8e0"
ACCENT  = "#00d4ff"     # cyan — price / main line
GREEN   = "#39ff14"     # neon green — positive reward / fees
RED     = "#ff4560"     # red — LVR / negative
AMBER   = "#ffb300"     # amber — lambda
PURPLE  = "#c77dff"     # purple — LP P&L
GOLD    = "#ffd700"     # gold — HODL reference
ACTION_COLORS = {
    0: "#555577",   # Hold  — grey
    1: "#00b0ff",   # Narrow — blue
    2: "#39ff14",   # Mid    — green
    3: "#ff4560",   # Wide   — red
}


# ─── Helpers ──────────────────────────────────────────────────────

def _style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Apply consistent dark styling to an axis."""
    ax.set_facecolor(BG2)
    ax.tick_params(colors=TEXT, labelsize=7)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(color=GRID, linewidth=0.4, linestyle="--", alpha=0.6)
    if title:
        ax.set_title(title, color=TEXT, fontsize=9, pad=4)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=7)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7)


def _load_episode(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"step", "price", "reward", "fees_t", "lvr_t", "net_pnl", "portfolio_value"}
    missing  = required - set(df.columns)
    if missing:
        log.warning("Episode CSV missing columns: %s — some panels may be skipped.", missing)
    return df


def _load_monitor(path: Path) -> pd.DataFrame:
    """Load SB3 Monitor CSV (skip 1 comment header line)."""
    df = pd.read_csv(path, comment="#")
    return df


# ─── Individual panels ────────────────────────────────────────────

def panel_price_range(ax, df: pd.DataFrame) -> None:
    """Panel 0: Price with LP range bands."""
    x = df["step"].values
    p = df["price"].values

    ax.plot(x, p, color=ACCENT, lw=0.9, label="WETH/USDC")

    if {"price_lower", "price_upper"}.issubset(df.columns):
        lo = df["price_lower"].values
        hi = df["price_upper"].values
        # Shade in-range vs out-of-range periods
        in_range = (p >= lo) & (p <= hi)
        ax.fill_between(x, lo, hi, where=in_range,  alpha=0.18, color=GREEN,  label="In-range")
        ax.fill_between(x, lo, hi, where=~in_range, alpha=0.10, color=RED,    label="Out-of-range")
        ax.plot(x, lo, lw=0.5, color=GREEN,  linestyle="--", alpha=0.6)
        ax.plot(x, hi, lw=0.5, color=RED,    linestyle="--", alpha=0.6)

    ax.legend(loc="upper left", fontsize=6, framealpha=0.15, labelcolor=TEXT)
    _style_ax(ax, "Price + LP Range", "Step", "USDC")


def panel_pnl(ax, df: pd.DataFrame, initial_capital: float = 100_000.0) -> None:
    """Panel 1: Cumulative P&L — LP vs HODL."""
    x    = df["step"].values

    # LP portfolio
    if "portfolio_value" in df.columns:
        lp_val = df["portfolio_value"].values
    else:
        lp_val = initial_capital + np.cumsum(df["reward"].values)

    ax.plot(x, lp_val, color=PURPLE, lw=1.2, label="LP Portfolio")

    # HODL reference (50/50 static hold)
    if "price" in df.columns:
        p0 = df["price"].iloc[0]
        pt = df["price"].values
        hodl = initial_capital * (0.5 * pt / p0 + 0.5)
        ax.plot(x, hodl, color=GOLD, lw=0.8, linestyle="--", label="HODL 50/50", alpha=0.85)
        # Shade LP > HODL
        ax.fill_between(x, lp_val, hodl,
                        where=(lp_val >= hodl), alpha=0.12, color=GREEN)
        ax.fill_between(x, lp_val, hodl,
                        where=(lp_val <  hodl), alpha=0.12, color=RED)

    ax.axhline(initial_capital, color=TEXT, lw=0.4, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=6, framealpha=0.15, labelcolor=TEXT)
    _style_ax(ax, "LP vs HODL Portfolio Value", "Step", "USD")


def panel_reward(ax, df: pd.DataFrame) -> None:
    """Panel 2: Per-step reward, coloured by action."""
    x = df["step"].values
    r = df["reward"].values

    if "action" in df.columns:
        for act, col in ACTION_COLORS.items():
            mask = df["action"].values == act
            if mask.any():
                ax.bar(x[mask], r[mask], color=col, width=1.0, alpha=0.85,
                       label=f"Action {act}")
        ax.legend(loc="upper right", fontsize=6, framealpha=0.15, labelcolor=TEXT,
                  title="Action", title_fontsize=6)
    else:
        pos = r >= 0
        ax.bar(x[pos],  r[pos],  color=GREEN, width=1.0, alpha=0.85)
        ax.bar(x[~pos], r[~pos], color=RED,   width=1.0, alpha=0.85)

    ax.axhline(0, color=TEXT, lw=0.5, alpha=0.4)
    _style_ax(ax, "Step Reward (by Action)", "Step", "USD")


def panel_lambda(ax, df: pd.DataFrame) -> None:
    """Panel 3: DeepSeek λ risk signal over time."""
    x = df["step"].values

    if "lambda" in df.columns:
        lam = df["lambda"].values
        ax.plot(x, lam, color=AMBER, lw=0.9, label="λ (sentinel)")
        ax.axhline(1.0, color=TEXT, lw=0.4, linestyle="--", alpha=0.5, label="λ=1 (neutral)")
        ax.fill_between(x, 1.0, lam, where=(lam > 1.0), alpha=0.12, color=RED,   label="High risk")
        ax.fill_between(x, 1.0, lam, where=(lam < 1.0), alpha=0.12, color=GREEN, label="Low risk")
        ax.set_ylim(0, max(2.2, lam.max() * 1.1))
        ax.legend(loc="upper right", fontsize=6, framealpha=0.15, labelcolor=TEXT)
    else:
        ax.text(0.5, 0.5, "λ column not found in CSV",
                ha="center", va="center", transform=ax.transAxes, color=TEXT, fontsize=8)

    _style_ax(ax, "DeepSeek Sentinel λ (Risk Multiplier)", "Step", "λ")


def panel_fee_lvr(ax, df: pd.DataFrame) -> None:
    """Panel 4: Stacked fee vs LVR decomposition."""
    x = df["step"].values

    if "fees_t" in df.columns:
        fees = df["fees_t"].values
        ax.fill_between(x, 0, fees, color=GREEN, alpha=0.65, label="Fees earned")
        ax.plot(x, fees, color=GREEN, lw=0.5, alpha=0.8)

    if "lvr_t" in df.columns and "lambda" in df.columns:
        lam = df["lambda"].values if "lambda" in df.columns else np.ones(len(x))
        lvr_penalty = lam * df["lvr_t"].values
        ax.fill_between(x, 0, -lvr_penalty, color=RED, alpha=0.55, label="λ × LVR leakage")
        ax.plot(x, -lvr_penalty, color=RED, lw=0.5, alpha=0.8)
    elif "lvr_penalty" in df.columns:
        ax.fill_between(x, 0, -df["lvr_penalty"].values,
                        color=RED, alpha=0.55, label="λ × LVR leakage")

    ax.axhline(0, color=TEXT, lw=0.5, alpha=0.5)
    ax.legend(loc="upper right", fontsize=6, framealpha=0.15, labelcolor=TEXT)
    _style_ax(ax, "Fees vs LVR Decomposition", "Step", "USD")


def panel_training(ax, monitor_df: Optional[pd.DataFrame]) -> None:
    """Panel 5: Episode reward during training (SB3 Monitor CSV)."""
    if monitor_df is None or "r" not in monitor_df.columns:
        ax.text(0.5, 0.5, "No training monitor data\n(run with --monitor path/to/monitor.csv)",
                ha="center", va="center", transform=ax.transAxes, color=TEXT, fontsize=8)
        _style_ax(ax, "Training Episode Rewards", "Episode", "Reward")
        return

    rewards = monitor_df["r"].values
    eps     = np.arange(len(rewards))

    ax.scatter(eps, rewards, color=ACCENT, s=1.5, alpha=0.4, label="Episode reward")

    # Rolling mean (window = 5% of episodes)
    w = max(10, len(rewards) // 20)
    rolling_mean = pd.Series(rewards).rolling(w, min_periods=1).mean().values
    ax.plot(eps, rolling_mean, color=AMBER, lw=1.2, label=f"Rolling mean ({w} ep)")

    ax.axhline(0, color=TEXT, lw=0.4, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=6, framealpha=0.15, labelcolor=TEXT)
    _style_ax(ax, "Training Reward Curve (SB3 Monitor)", "Episode", "Cumulative Reward")


# ─── Main plot function ───────────────────────────────────────────

def make_dashboard(
    episode_csv: Path,
    monitor_csv: Optional[Path] = None,
    output_path: Path = Path("logs/plots/dashboard.png"),
    initial_capital: float = 100_000.0,
    dpi: int = 150,
) -> Path:
    """
    Build the 6-panel ALVRO dashboard and save to PNG.

    Parameters
    ----------
    episode_csv     : CSV from evaluate.py --output
    monitor_csv     : SB3 Monitor CSV from training (optional)
    output_path     : PNG destination
    initial_capital : Starting portfolio (for HODL baseline)
    dpi             : Output resolution

    Returns
    -------
    Path  to saved PNG
    """
    log.info("Loading episode data from %s …", episode_csv)
    df = _load_episode(episode_csv)
    log.info("  %d steps loaded.", len(df))

    monitor_df = None
    if monitor_csv and monitor_csv.exists():
        log.info("Loading monitor data from %s …", monitor_csv)
        monitor_df = _load_monitor(monitor_csv)
        log.info("  %d training episodes.", len(monitor_df))

    # ── Figure layout ─────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 18), facecolor=BG)
    gs  = gridspec.GridSpec(
        3, 2,
        figure=fig,
        hspace=0.42,
        wspace=0.30,
        left=0.06, right=0.97,
        top=0.93,  bottom=0.05,
    )

    ax0 = fig.add_subplot(gs[0, 0])   # Price + Range
    ax1 = fig.add_subplot(gs[0, 1])   # P&L
    ax2 = fig.add_subplot(gs[1, 0])   # Step Reward
    ax3 = fig.add_subplot(gs[1, 1])   # Lambda
    ax4 = fig.add_subplot(gs[2, 0])   # Fee/LVR decomposition
    ax5 = fig.add_subplot(gs[2, 1])   # Training curve

    panel_price_range(ax0, df)
    panel_pnl(ax1, df, initial_capital)
    panel_reward(ax2, df)
    panel_lambda(ax3, df)
    panel_fee_lvr(ax4, df)
    panel_training(ax5, monitor_df)

    # ── Super-title ───────────────────────────────────────────────
    n_steps      = len(df)
    total_reward = df["reward"].sum() if "reward" in df.columns else 0.0
    n_rebalances = df["rebalanced"].sum() if "rebalanced" in df.columns else "—"
    in_range_pct = (
        100.0 * df["in_range"].mean() if "in_range" in df.columns else 0.0
    )

    title = (
        f"ALVRO_v3 — Adaptive LVR Optimizer  |  FinAI Contest 2025\n"
        f"Steps: {n_steps:,}   ·   Total Reward: {total_reward:+,.2f} USD"
        f"   ·   Rebalances: {n_rebalances}   ·   In-Range: {in_range_pct:.1f}%"
    )
    fig.suptitle(title, color=TEXT, fontsize=11, fontweight="bold", y=0.97)

    # ── Save ─────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.info("✔ Dashboard saved → %s", output_path)
    return output_path


# ─── CLI ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ALVRO_v3 Results Visualiser")
    p.add_argument(
        "--episode", type=str, required=True,
        help="Path to episode trajectory CSV (from evaluate.py --output)",
    )
    p.add_argument(
        "--monitor", type=str, default=None,
        help="Path to SB3 Monitor CSV (logs/run_xx/train_monitor.csv)",
    )
    p.add_argument(
        "--output", type=str,
        default="logs/plots/dashboard.png",
        help="PNG output path",
    )
    p.add_argument(
        "--initial-capital", type=float, default=100_000.0,
        help="Starting capital for HODL baseline",
    )
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ep_csv  = Path(args.episode)
    mon_csv = Path(args.monitor) if args.monitor else None
    out     = Path(args.output)

    if not ep_csv.exists():
        log.error("Episode CSV not found: %s", ep_csv)
        sys.exit(1)

    make_dashboard(
        episode_csv=ep_csv,
        monitor_csv=mon_csv,
        output_path=out,
        initial_capital=args.initial_capital,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
