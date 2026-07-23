"""Mootdx intraday map must accept lowercase 1h like documented 1H."""

from __future__ import annotations

from backtest.loaders.mootdx_loader import _INTRADAY_FREQ


def test_mootdx_intraday_accepts_lowercase_1h() -> None:
    assert _INTRADAY_FREQ["1H"] == 3
    assert _INTRADAY_FREQ["1h"] == 3
