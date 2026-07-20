"""Reject non-positive ADX periods in technical-basic helpers."""

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
    spec = importlib.util.spec_from_file_location("tb_adx", _PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_compute_adx_rejects_zero_period() -> None:
    mod = _load()
    close = pd.Series(range(1, 40), dtype=float)
    high = close + 1
    low = close - 1
    with pytest.raises(ValueError, match="period must be >= 1"):
        mod.compute_adx(high, low, close, period=0)


def test_compute_adx_default_period_works() -> None:
    mod = _load()
    close = pd.Series(range(1, 40), dtype=float)
    high = close + 1
    low = close - 1
    out = mod.compute_adx(high, low, close)
    assert "adx" in out.columns
    assert len(out) == len(close)
