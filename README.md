# ALVRO — Adaptive LVR Optimizer

**A Reinforcement Learning Agent for Concentrated Liquidity Provision on Uniswap v3**

> FinAI Contest 2025 · NeurIPS Workshop Submission

---

## Overview

ALVRO trains a MaskablePPO agent to manage a concentrated liquidity position on Uniswap v3 (WETH/USDC 0.05% pool). The agent adaptively selects range widths to maximise fee income while minimising Loss-Versus-Rebalancing (LVR) and transaction gas costs.

Three innovations over naive LP strategies:

1. **Asymmetric Defensive Range** — upside compressed, crash floor extended in proportion to a risk signal λ
2. **Hysteresis Gas Gate** — action masking prevents rebalancing unless projected net benefit exceeds gas cost
3. **DeepSeek Sentinel** — LLM-generated λ ∈ [0.3, 2.0] risk multiplier, updated every 24 steps (rule-based fallback available without API key)

### Key Results (Rule-Based Sentinel, 1M training steps)

| Metric | Value |
|--------|-------|
| LP-to-HODL Ratio | **1.3235** |
| Net Reward | **+$43,284** |
| Gas Efficiency | **2.80×** |
| Annualised Sharpe | **210.2** |
| In-Range % | **82.4%** |

---

## Repository Structure

```
ALVRO_v3/
├── env/
│   └── alvro_env.py           # ALVROEnv — Gymnasium env (14-dim obs, Discrete(4))
├── models/
│   └── ppo_agent.py           # ALVROAgent — MaskablePPO wrapper + callbacks
├── data/
│   ├── processor.py           # Real data pipeline: GARCH → LVR → Parquet
│   └── synthetic_generator.py # Fallback: GARCH(1,1) GBM synthetic data
├── core/
│   ├── evaluator.py           # EvalMetrics: LP-to-HODL, Sharpe, Calmar, Gas Efficiency
│   └── sentinel.py            # DeepSeek Sentinel: λ risk signal (static / rule / llm)
├── scripts/
│   ├── train.py               # End-to-end PPO training pipeline
│   ├── evaluate.py            # Load model → evaluate on hold-out → CSV
│   ├── test_env.py            # Env smoke-test (gas gate, OOR fees, λ sensitivity)
│   ├── check_data.py          # Data quality audit + GARCH/LVR plot
│   ├── plot_results.py        # 6-panel results dashboard
│   └── run_pipeline.py        # One-shot full pipeline runner
├── paper/
│   ├── main.tex               # NeurIPS 2024 LaTeX paper
│   └── references.bib         # BibTeX references
├── config.yaml                # Full project configuration
└── requirements.txt           # Python dependencies
```

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Get data

Real WETH/USDC hourly data (May 2021 – Jan 2024, 23,995 rows):

```bash
python -m data.processor
```

Synthetic fallback (no internet required):

```bash
python -m data.synthetic_generator --n-hours 26280
```

### 3. Smoke-test

```bash
python -m scripts.test_env --steps 200 --seed 42
```

### 4. Train

```bash
# Rule-based sentinel (no API key needed):
python -m scripts.train --timesteps 1000000 --sentinel-mode rule

# LLM sentinel (set DEEPSEEK_API_KEY first):
export DEEPSEEK_API_KEY="sk-..."
python -m scripts.train --sentinel-mode llm
```

### 5. Evaluate

```bash
python -m scripts.evaluate \
    --model logs/run_01/alvro_ppo_final \
    --n-eval 5 \
    --output logs/results/eval_trajectory.csv
```

### 6. Visualise

```bash
python -m scripts.plot_results \
    --episode logs/results/eval_trajectory.csv \
    --monitor logs/run_01/train_monitor.monitor.csv \
    --output  logs/plots/dashboard.png
```

### One-shot pipeline

```bash
# Smoke run (100k steps, synthetic data):
python -m scripts.run_pipeline --use-synthetic --timesteps 100000

# Full run (1M steps, real data):
python -m scripts.run_pipeline --timesteps 1000000
```

---

## Core Math

### Reward Function

$$R_t = \text{Fees}_t - \lambda_t \cdot \text{LVR}_t - \text{Gas}_t \cdot \mathbf{1}[\text{action} \neq 0]$$

### LVR

$$\text{LVR}_t = \tfrac{1}{2}\,\sigma_t^2 \cdot P_t \cdot \Gamma_t \cdot \Delta t, \qquad \Gamma_t = \frac{L_{\text{eff}}}{2 P_t^{3/2} \sqrt{P_U - P_L}}$$

### Asymmetric Defensive Range (with $W = k\sigma$)

$$P_L = P \cdot \max\!\left(0,\; 1 - W\lambda^2\right) \qquad P_H = P \cdot \left(1 + \frac{W}{\lambda}\right)$$

### Gas Gate (action masking)

$$\mathbb{E}[\Delta\text{Fees} + \Delta\text{LVR}_\text{mitig}] > \text{Gas} \;\Rightarrow\; \text{allow rebalance}$$

---

## Agent Architecture

| Component | Detail |
|-----------|--------|
| Algorithm | MaskablePPO (`sb3-contrib`) |
| Policy | `MlpPolicy` — 2 × 256 hidden, Tanh |
| Observation | `Box(14,)` — price, σ, LVR proxy, λ, fees, range bounds, log-return, RSI, MA, portfolio, in-range, steps-since-rebal, net PnL |
| Actions | `Discrete(4)` — Hold / Narrow ±1% / Mid ±5% / Wide ±15% |
| Action masking | `env.action_masks()` → MaskablePPO zeroes masked action probabilities |

---

## Sentinel Modes

| Mode | Description |
|------|-------------|
| `rule` | Heuristic from σ, RSI, MA, LVR — no API key needed |
| `llm` | DeepSeek-chat generates λ every 24 steps |
| `static` | Fixed λ=1.0 (ablation baseline) |

---

## Data

- **Source**: WETH/USDC Uniswap v3 hourly price/volume data from [Alessio Brini's DRL-AMM repo](https://github.com/AlessioB94/DRL-AMM)
- **Period**: May 2021 – January 2024 (23,995 hours)
- **Train/Test split**: 80% / 20%
- **Features**: GARCH(1,1) conditional volatility (α=0.073, β=0.909), RSI-14, 24h MA, LVR proxy

---

## Requirements

- Python 3.10 or 3.11
- Key packages: `gymnasium`, `stable-baselines3`, `sb3-contrib` (MaskablePPO), `torch`, `arch` (GARCH), `openai` (for LLM sentinel)
- See `requirements.txt` for pinned versions

---

## Paper

Full methodology and results: [`paper/main.tex`](paper/main.tex)

> *ALVRO: Adaptive LVR Optimizer — A Reinforcement Learning Agent for Concentrated Liquidity Provision on Uniswap v3*

---

## References

- Milionis et al. (2022/2024) — *Automated Market Making and Loss-Versus-Rebalancing* — [arXiv:2208.06046](https://arxiv.org/abs/2208.06046)
- Xu & Brini (2025) — *Improving DeFi Accessibility through Efficient Liquidity Provisioning with DRL* — [arXiv:2501.07508](https://arxiv.org/abs/2501.07508)
- Schulman et al. (2017) — *Proximal Policy Optimization Algorithms* — [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Bollerslev (1986) — *Generalized Autoregressive Conditional Heteroskedasticity*
- Adams et al. (2021) — *Uniswap v3 Core Whitepaper*
