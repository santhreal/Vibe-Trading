"""Eastmoney KLT map must accept lowercase 1h like documented 1H."""

from __future__ import annotations

from backtest.loaders.eastmoney_client import KLT_BY_INTERVAL


def test_eastmoney_klt_accepts_lowercase_1h() -> None:
    assert KLT_BY_INTERVAL["1H"] == 60
    assert KLT_BY_INTERVAL["1h"] == 60
    assert KLT_BY_INTERVAL["60m"] == 60
