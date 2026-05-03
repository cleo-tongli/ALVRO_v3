"""
ALVRO_v3 — Data Quality Check
==============================
scripts/check_data.py

Runs three diagnostic checks on data/processed_market.parquet:
  1. Head preview + NaN audit
  2. Price vs. conditional volatility (sigma) dual-axis plot
  3. LVR proxy statistics + formula validity flag

Usage:
    python -m scripts.check_data
    # or from project root:
    python scripts/check_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless-safe; swap to "TkAgg" if interactive
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── Resolve project root & parquet path ──────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_PARQUET = _ROOT / "data" / "processed_market.parquet"
_OUTPUT  = _ROOT / "logs" / "check_data_plot.png"

RESET  = "\033[0m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"

def ok(msg):  print(f"{GREEN}  ✔ {msg}{RESET}")
def warn(msg): print(f"{YELLOW}  ⚠ {msg}{RESET}")
def err(msg):  print(f"{RED}  ✖ {msg}{RESET}")
def header(msg): print(f"\n{BOLD}{'─'*60}\n  {msg}\n{'─'*60}{RESET}")


def check_load(path: Path) -> pd.DataFrame:
    """Check 1 — Load parquet, preview head, audit NaN values."""
    header("CHECK 1 — Load & NaN Audit")

    if not path.exists():
        err(f"File not found: {path}")
        err("Run  'python -m data.processor'  first to generate it.")
        sys.exit(1)

    df = pd.read_parquet(path)
    ok(f"Loaded {len(df):,} rows × {len(df.columns)} columns from {path.name}")

    print(f"\n{'─'*60}")
    print("  First 5 rows:")
    print("─"*60)
    print(df.head().to_string(index=False))

    # NaN audit
    print(f"\n{'─'*60}")
    print("  NaN counts per column:")
    print("─"*60)
    nan_counts = df.isna().sum()
    any_nan = False
    for col, n in nan_counts.items():
        if n > 0:
            warn(f"  {col:<25} {n:>6} NaN  ({100*n/len(df):.2f}%)")
            any_nan = True
        else:
            ok(f"  {col:<25} {n:>6} NaN")

    if not any_nan:
        ok("No NaN values detected — data is clean.")
    else:
        warn("NaN values found. Consider re-running processor.py with a fill strategy.")

    return df


def check_plot(df: pd.DataFrame, output: Path) -> None:
    """Check 2 — Dual-axis plot of price and sigma to verify GARCH spikes."""
    header("CHECK 2 — Price vs. Conditional Volatility (GARCH σ) Plot")

    if "sigma" not in df.columns:
        err("Column 'sigma' missing — GARCH step may not have run.")
        return

    # Use timestamp if available
    x = df["timestamp"] if "timestamp" in df.columns else pd.RangeIndex(len(df))

    fig = plt.figure(figsize=(16, 8), facecolor="#0f0f1a")
    gs  = gridspec.GridSpec(2, 1, hspace=0.08, height_ratios=[2, 1])

    # — top: price ——————————————————————————————————————————————
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#0f0f1a")
    ax1.plot(x, df["price"], color="#00d4ff", linewidth=0.8, label="WETH/USDC Price")
    ax1.set_ylabel("Price (USDC)", color="#00d4ff", fontsize=11)
    ax1.tick_params(colors="#cccccc"); ax1.yaxis.label.set_color("#00d4ff")
    ax1.spines[:].set_color("#333355")
    ax1.tick_params(axis="x", labelbottom=False)
    ax1.legend(loc="upper left", framealpha=0.2, labelcolor="#ffffff")
    ax1.set_title(
        "ALVRO_v3 — Price & GARCH Conditional Volatility",
        color="#ffffff", fontsize=13, pad=10,
    )

    # — bottom: sigma ——————————————————————————————————————————
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_facecolor("#0f0f1a")
    ax2.fill_between(x, df["sigma"], color="#ff6b6b", alpha=0.5, label="σ (GARCH)")
    ax2.plot(x, df["sigma"], color="#ff6b6b", linewidth=0.6)
    # Annotate top-5 volatility spikes
    top5_idx = df["sigma"].nlargest(5).index
    for idx in top5_idx:
        xi  = x.iloc[idx] if hasattr(x, "iloc") else idx
        yi  = df["sigma"].iloc[idx]
        ax2.annotate(
            f"σ={yi:.4f}",
            xy=(xi, yi), xytext=(0, 8), textcoords="offset points",
            fontsize=6, color="#ffdd44", ha="center",
            arrowprops=dict(arrowstyle="-", color="#ffdd44", lw=0.5),
        )
    ax2.set_ylabel("σ (GARCH)", color="#ff6b6b", fontsize=11)
    ax2.set_xlabel("Time", color="#cccccc", fontsize=10)
    ax2.tick_params(colors="#cccccc"); ax2.yaxis.label.set_color("#ff6b6b")
    ax2.spines[:].set_color("#333355")
    ax2.legend(loc="upper left", framealpha=0.2, labelcolor="#ffffff")

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    ok(f"Plot saved → {output}")


def check_lvr(df: pd.DataFrame) -> None:
    """Check 3 — LVR proxy statistics and formula validity check."""
    header("CHECK 3 — LVR Proxy Statistics & Formula Validation")

    if "lvr_proxy" not in df.columns:
        err("Column 'lvr_proxy' missing — cannot validate LVR formula.")
        return

    lvr = df["lvr_proxy"]
    mean_lvr = lvr.mean()
    max_lvr  = lvr.max()
    min_lvr  = lvr.min()
    std_lvr  = lvr.std()
    pct_zero = (lvr <= 0).sum() / len(lvr) * 100
    pct_pos  = (lvr > 0).sum() / len(lvr) * 100

    print(f"\n  {'Statistic':<30} {'Value':>15}")
    print(f"  {'─'*45}")
    print(f"  {'Count':<30} {len(lvr):>15,}")
    print(f"  {'Mean LVR':<30} {mean_lvr:>15.6f}")
    print(f"  {'Std LVR':<30} {std_lvr:>15.6f}")
    print(f"  {'Min LVR':<30} {min_lvr:>15.6f}")
    print(f"  {'Max LVR':<30} {max_lvr:>15.6f}")
    print(f"  {'% rows > 0':<30} {pct_pos:>14.2f}%")
    print(f"  {'% rows ≤ 0':<30} {pct_zero:>14.2f}%")

    # ── Validity checks ────────────────────────────────────────
    print()
    THRESHOLD_ZERO_PCT = 90.0   # flag if >90 % of rows are zero/negative

    if mean_lvr > 0 and max_lvr > 0:
        ok(f"LVR proxy is positive (mean={mean_lvr:.6f}, max={max_lvr:.6f}).")
    else:
        err(f"mean LVR={mean_lvr:.6f}, max LVR={max_lvr:.6f}")
        err("LVR proxy is consistently ≤ 0. Possible issues:")
        err("  • sigma values are zero (GARCH fit failed or data too short)")
        err("  • price or gamma values are zero / NaN")
        err("  • LVR formula: 0.5 * σ² * P * Γ * dt — check each term")

    if pct_zero > THRESHOLD_ZERO_PCT:
        err(
            f"{pct_zero:.1f}% of LVR values are ≤ 0 (>{THRESHOLD_ZERO_PCT}% threshold)."
            "\n  → Re-run data/processor.py and verify GARCH sigma output is non-zero."
        )
    elif pct_zero > 10.0:
        warn(f"{pct_zero:.1f}% of LVR values are ≤ 0. Investigate zero-sigma rows.")
    else:
        ok(f"LVR formula validity: {pct_pos:.1f}% of rows have positive LVR.")

    # ── Correlation: sigma² vs LVR (should be near 1.0) ──────
    if "sigma" in df.columns:
        sigma_sq = df["sigma"] ** 2
        corr = lvr.corr(sigma_sq)
        if corr > 0.95:
            ok(f"LVR ↔ σ² correlation = {corr:.4f}  (expected ≈ 1.0) ✓")
        elif corr > 0.7:
            warn(f"LVR ↔ σ² correlation = {corr:.4f}  (moderate; check Gamma variation)")
        else:
            err(f"LVR ↔ σ² correlation = {corr:.4f}  (low — formula may be incorrect)")


def main() -> None:
    print(f"\n{BOLD}{'═'*60}")
    print("  ALVRO_v3 — Data Quality Check")
    print(f"{'═'*60}{RESET}")

    df = check_load(_PARQUET)
    check_plot(df, _OUTPUT)
    check_lvr(df)

    print(f"\n{BOLD}{'═'*60}")
    print("  All checks complete.")
    print(f"{'═'*60}{RESET}\n")


if __name__ == "__main__":
    main()
