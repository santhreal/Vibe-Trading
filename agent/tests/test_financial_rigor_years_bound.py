"""Bound three_scenario forecast horizon to avoid tool hangs."""

from __future__ import annotations

import json

import pytest

from src.tools.financial_rigor_tool import FinancialRigorTool, three_scenario_valuation


def test_three_scenario_rejects_huge_years() -> None:
    with pytest.raises(ValueError, match="years must be between"):
        three_scenario_valuation(10, 1, 1, 0.1, 0.1, 0.1, 15, 15, 15, years=10_000_000)


def test_three_scenario_tool_returns_error_for_huge_years() -> None:
    out = json.loads(
        FinancialRigorTool().execute(
            command="three_scenario",
            price=10,
            eps=1,
            shares=1,
            growth=[0.1, 0.1, 0.1],
            pe=[15, 15, 15],
            years=10_000_000,
        )
    )
    assert out["status"] == "error"
    assert "years" in out["error"]


def test_three_scenario_default_years_still_works() -> None:
    result = three_scenario_valuation(10, 1, 1, 0.1, 0.1, 0.1, 15, 15, 15)
    assert result["years"] == 3
    assert len(result["scenarios"]) == 3
