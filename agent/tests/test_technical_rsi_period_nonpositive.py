"""Reject non-positive RSI periods in technical-basic helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "skills"
    / "technical-basic"
    / "example_signal_engine.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("tb_rsi", _PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_compute_rsi_rejects_zero_period() -> None:
    mod = _load()
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(ValueError, match="period must be >= 1"):
        mod.compute_rsi(close, period=0)


def test_compute_rsi_default_period_works() -> None:
    mod = _load()
    close = pd.Series(range(1, 40), dtype=float)
    out = mod.compute_rsi(close)
    assert len(out) == len(close)
    assert out.iloc[-1] == out.iloc[-1]  # not NaN for long enough series
