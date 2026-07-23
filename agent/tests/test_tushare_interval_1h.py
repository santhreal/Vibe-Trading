"""Tushare minute fetch must accept lowercase 1h like the documented 1H token."""

from __future__ import annotations

import pandas as pd

from backtest.loaders import tushare as ts


def test_fetch_minutes_accepts_lowercase_1h(monkeypatch) -> None:
    seen: list[str] = []

    class _FakeApi:
        def stk_mins(self, **kwargs):
            seen.append(kwargs.get("freq", ""))
            return pd.DataFrame(
                {
                    "trade_time": ["2024-01-01 10:00:00"],
                    "open": [1.0],
                    "high": [1.0],
                    "low": [1.0],
                    "close": [1.0],
                    "vol": [100.0],
                }
            )

    loader = ts.DataLoader.__new__(ts.DataLoader)
    loader.api = _FakeApi()
    monkeypatch.setattr(ts, "_is_etf_listed", lambda code: False)
    monkeypatch.setattr(ts, "_is_index", lambda code: False)
    monkeypatch.setattr(ts, "_is_hk_equity", lambda code: False)
    monkeypatch.setattr(ts, "_is_us_equity", lambda code: False)
    monkeypatch.setattr(ts, "_is_crypto", lambda code: False)

    out = loader._fetch_minutes(["600000.SH"], "2024-01-01", "2024-01-02", "1h")
    assert seen == ["60min"]
    assert "600000.SH" in out

    seen.clear()
    loader._fetch_minutes(["600000.SH"], "2024-01-01", "2024-01-02", "1H")
    assert seen == ["60min"]
