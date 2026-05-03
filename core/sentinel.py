"""
ALVRO_v3 — DeepSeek Sentinel (λ Risk Signal)
=============================================
Module: core/sentinel.py

The Sentinel is a modular wrapper that produces the risk multiplier λ ∈ [0.3, 2.0].
It supports three operating modes:

  1. STATIC   — fixed λ value (useful for ablation studies)
  2. RULE     — heuristic rules from market features (no API needed)
  3. LLM      — narrative summary sent to DeepSeek / OpenAI for a λ score
                (requires DEEPSEEK_API_KEY or OPENAI_API_KEY env var)

The computed λ is injected into ALVROEnv via env.update_external_risk(λ).

Usage
-----
    from core.sentinel import DeepSeekSentinel

    sentinel = DeepSeekSentinel(mode="rule")
    lam = sentinel.compute_lambda(
        sigma=0.032, rsi=72.4, ma_rel=-0.015, lvr=0.0021
    )
    env.update_external_risk(lam)
"""

from __future__ import annotations

import logging
import math
import os
from typing import Dict, Literal, Optional

import numpy as np

log = logging.getLogger("DeepSeekSentinel")

# ── λ bounds ──────────────────────────────────────────────────────
LAMBDA_MIN: float = 0.3
LAMBDA_MAX: float = 2.0
LAMBDA_DEFAULT: float = 1.0

# ── LLM prompt template ───────────────────────────────────────────
_SYSTEM_PROMPT = """You are a quantitative risk analyst specialising in DeFi liquidity provision.
Given the following market indicators, output ONLY a single float number for the risk
multiplier lambda (λ) in the range [{lmin}, {lmax}].
- λ close to {lmin} means low risk (bullish, calm market, wide LP range is fine)
- λ close to {lmax} means high risk (bearish, volatile market, LP should defend aggressively)
Reply with a single decimal number only. Do not include units, words, or explanations."""

_USER_TEMPLATE = """
Current market snapshot:
- GARCH(1,1) σ (hourly volatility): {sigma:.6f}
- RSI(14): {rsi:.1f}
- Price vs 24h MA (relative): {ma_rel:+.4f}
- LVR proxy (loss vs rebalancing): {lvr:.6f}
- Current λ (previous): {prev_lambda:.3f}

Provide the new λ as a single float:"""


class DeepSeekSentinel:
    """
    Adaptive risk-multiplier λ generator for the ALVRO DeepSeek Sentinel layer.

    Parameters
    ----------
    mode : 'static' | 'rule' | 'llm'
        Operating mode.
    static_lambda : float
        λ value used when mode='static'.
    llm_model : str
        Model identifier for the LLM API call (mode='llm').
    api_key_env : str
        Environment variable name that stores the API key.
    api_base_url : str | None
        Override base URL (e.g. for DeepSeek's endpoint).
    update_every : int
        Minimum number of environment steps between LLM calls
        (rate-limiting; only relevant for mode='llm').
    """

    def __init__(
        self,
        mode: Literal["static", "rule", "llm"] = "rule",
        static_lambda: float = LAMBDA_DEFAULT,
        llm_model: str = "deepseek-chat",
        api_key_env: str = "DEEPSEEK_API_KEY",
        api_base_url: Optional[str] = "https://api.deepseek.com/v1",
        update_every: int = 24,   # update once per 24 hourly steps (daily)
    ) -> None:
        self.mode = mode
        self._lambda = float(np.clip(static_lambda, LAMBDA_MIN, LAMBDA_MAX))
        self._llm_model = llm_model
        self._api_key_env = api_key_env
        self._api_base_url = api_base_url
        self._update_every = update_every
        self._step_counter: int = 0
        self._client = None   # lazy-init OpenAI client

        if mode == "llm":
            self._init_llm_client()

        log.info(
            "DeepSeekSentinel initialised | mode=%s  λ₀=%.3f",
            mode, self._lambda,
        )

    # ── Public API ────────────────────────────────────────────────

    def compute_lambda(
        self,
        sigma: float = 0.02,
        rsi: float = 50.0,
        ma_rel: float = 0.0,
        lvr: float = 0.0,
        force_update: bool = False,
    ) -> float:
        """
        Compute and return the risk multiplier λ.

        Parameters
        ----------
        sigma    : GARCH(1,1) conditional volatility (decimal, hourly).
        rsi      : RSI(14) value in [0, 100].
        ma_rel   : (MA_24h − price) / price — negative means price above MA.
        lvr      : LVR proxy for the current step.
        force_update : bypass the update_every throttle.
        """
        self._step_counter += 1
        should_update = (
            force_update
            or self._step_counter % self._update_every == 0
            or self._step_counter == 1
        )

        if not should_update:
            return self._lambda

        if self.mode == "static":
            pass   # λ unchanged

        elif self.mode == "rule":
            self._lambda = self._rule_based_lambda(sigma, rsi, ma_rel, lvr)

        elif self.mode == "llm":
            try:
                self._lambda = self._llm_lambda(sigma, rsi, ma_rel, lvr)
            except Exception as exc:
                log.warning(
                    "LLM call failed (%s); falling back to rule-based λ.", exc
                )
                self._lambda = self._rule_based_lambda(sigma, rsi, ma_rel, lvr)

        log.debug(
            "Sentinel λ=%.4f | mode=%s σ=%.4f RSI=%.1f ma_rel=%+.4f lvr=%.6f",
            self._lambda, self.mode, sigma, rsi, ma_rel, lvr,
        )
        return self._lambda

    def get_lambda(self) -> float:
        """Return the most recent computed λ."""
        return self._lambda

    # ── Mode 2: Rule-based ────────────────────────────────────────

    @staticmethod
    def _rule_based_lambda(
        sigma: float,
        rsi: float,
        ma_rel: float,
        lvr: float,
    ) -> float:
        """
        Heuristic λ using market microstructure signals.

        Logic:
            Base score (σ-driven):
                High volatility ↑σ → higher λ (more defensive)
                σ > 3 % / hr  → λ ≈ 2.0   (crisis regime)
                σ < 0.5% / hr → λ ≈ 0.3   (calm regime)

            RSI adjustment:
                RSI > 70 (overbought) → small upward pressure (+0.1)
                RSI < 30 (oversold)   → small upward pressure (+0.15) [crash risk]

            Trend adjustment:
                Price below 24h MA (ma_rel < 0) → bearish → λ ↑

            LVR adjustment:
                High LVR leakage → λ ↑

        Returns λ clipped to [LAMBDA_MIN, LAMBDA_MAX].
        """
        # ── Base: sigmoid-scaled from σ ──────────────────────────
        # σ range [0, 0.05]; λ ∈ [0.3, 2.0]
        sig_norm = sigma / 0.03   # normalise to 1 at σ=3%
        base = LAMBDA_MIN + (LAMBDA_MAX - LAMBDA_MIN) * (1 / (1 + math.exp(-3 * (sig_norm - 1))))

        # ── RSI contribution ──────────────────────────────────────
        rsi_adj = 0.0
        if rsi > 70:
            rsi_adj = 0.1                        # overbought
        elif rsi < 30:
            rsi_adj = 0.2                        # oversold / crash risk

        # ── Trend contribution ────────────────────────────────────
        # ma_rel < 0 → price above MA → slight upward bias  (momentum)
        # ma_rel > 0 → price below MA → more defensive
        trend_adj = float(np.clip(-ma_rel * 5.0, -0.2, 0.3))

        # ── LVR contribution ──────────────────────────────────────
        lvr_adj = float(np.clip(lvr * 100.0, 0.0, 0.3))

        lam = base + rsi_adj + trend_adj + lvr_adj
        return float(np.clip(lam, LAMBDA_MIN, LAMBDA_MAX))

    # ── Mode 3: LLM (DeepSeek / OpenAI) ──────────────────────────

    def _init_llm_client(self) -> None:
        """Lazily initialise the OpenAI-compatible API client."""
        try:
            from openai import OpenAI
            api_key = os.environ.get(self._api_key_env, "")
            if not api_key:
                raise EnvironmentError(
                    f"API key not found in env var '{self._api_key_env}'. "
                    "Set it before using mode='llm'."
                )
            self._client = OpenAI(
                api_key=api_key,
                base_url=self._api_base_url,
            )
            log.info("LLM client initialised (model: %s)", self._llm_model)
        except ImportError:
            raise ImportError(
                "openai package is required for mode='llm'. "
                "Install with: pip install openai"
            )

    def _llm_lambda(
        self,
        sigma: float,
        rsi: float,
        ma_rel: float,
        lvr: float,
    ) -> float:
        """Call the LLM API and parse the returned λ value."""
        if self._client is None:
            self._init_llm_client()

        system_msg = _SYSTEM_PROMPT.format(lmin=LAMBDA_MIN, lmax=LAMBDA_MAX)
        user_msg = _USER_TEMPLATE.format(
            sigma=sigma, rsi=rsi, ma_rel=ma_rel, lvr=lvr,
            prev_lambda=self._lambda,
        )

        response = self._client.chat.completions.create(
            model=self._llm_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=10,
        )

        raw = response.choices[0].message.content.strip()
        try:
            lam = float(raw)
        except ValueError:
            # Extract first float-like token from the response
            import re
            match = re.search(r"[-+]?\d*\.?\d+", raw)
            if match:
                lam = float(match.group())
            else:
                log.warning("Could not parse LLM response '%s'; keeping old λ.", raw)
                lam = self._lambda

        clipped = float(np.clip(lam, LAMBDA_MIN, LAMBDA_MAX))
        log.info("LLM λ=%s → clipped=%.4f", raw, clipped)
        return clipped


# ─────────────────────────────────────────────────────────────────
# Convenience runner (inject λ into a live env loop)
# ─────────────────────────────────────────────────────────────────

def attach_sentinel_to_env(env, sentinel: DeepSeekSentinel, row_data: Dict) -> None:
    """
    Compute λ from the current data row and inject it into the environment.

    Parameters
    ----------
    env     : ALVROEnv instance.
    sentinel: DeepSeekSentinel instance.
    row_data: dict with keys 'sigma', 'rsi_14', 'ma_24h', 'lvr_proxy', 'price'.
    """
    price = row_data.get("price", 1.0) or 1.0
    ma = row_data.get("ma_24h", price) or price
    lam = sentinel.compute_lambda(
        sigma=row_data.get("sigma", 0.02),
        rsi=row_data.get("rsi_14", 50.0),
        ma_rel=(ma - price) / (price + 1e-9),
        lvr=row_data.get("lvr_proxy", 0.0),
    )
    env.update_external_risk(lam)
