"""
data — ALVRO Data Pipeline
============================
Exposes the data processor and the synthetic data generator fallback.
Imports are lazy to avoid pulling arch/pyarrow at package-load time when
only one submodule is being used (e.g. python -m data.synthetic_generator).
"""

__all__ = ["ALVRODataProcessor", "SyntheticMarketGenerator"]


def __getattr__(name):
    if name == "ALVRODataProcessor":
        from data.processor import ALVRODataProcessor
        return ALVRODataProcessor
    if name == "SyntheticMarketGenerator":
        from data.synthetic_generator import SyntheticMarketGenerator
        return SyntheticMarketGenerator
    raise AttributeError(f"module 'data' has no attribute {name!r}")
