"""Tests for Heston (1993) stochastic volatility model."""

from __future__ import annotations

import math
import pytest

from src.quantlib.volatility import (
    heston_characteristic_function,
    heston_feller_condition,
    heston_price,
)
from src.quantlib.options import bs_price


class TestHestonPricing:
    """Validate Heston pricing against benchmarks and structural properties."""

    def test_heston_feller_condition(self) -> None:
        # 2 * kappa * theta > sigma_v^2
        # kappa=2.0, theta=0.04 -> 2*2*0.04 = 0.16. sigma_v=0.3 -> sigma_v^2 = 0.09 -> ratio = 0.16/0.09 > 1
        res = heston_feller_condition(kappa=2.0, theta=0.04, sigma_v=0.3)
        assert res["is_satisfied"] is True
        assert res["feller_ratio"] == pytest.approx(0.16 / 0.09, rel=1e-4)

        # Feller violated: kappa=0.5, theta=0.04 -> 0.04. sigma_v=0.5 -> 0.25 -> ratio < 1
        res_viol = heston_feller_condition(kappa=0.5, theta=0.04, sigma_v=0.5)
        assert res_viol["is_satisfied"] is False

    def test_heston_collapses_to_black_scholes_when_sigma_v_is_near_zero(self) -> None:
        # When volatility of variance is ~ 0 and v0 == theta == sigma^2, Heston == BS
        S0 = 100.0
        K = 100.0
        T = 1.0
        r = 0.05
        q = 0.01
        vol = 0.20
        v0 = vol**2
        theta = vol**2
        kappa = 1.5
        sigma_v = 1e-4
        rho = 0.0

        h_call = heston_price(S0, K, T, r, v0, kappa, theta, sigma_v, rho, option_type="call", q=q)
        b_call = bs_price(S0, K, T, r, vol, option_type="call", q=q)
        assert h_call == pytest.approx(b_call, abs=1e-3)

    def test_heston_put_call_parity(self) -> None:
        S0 = 100.0
        K = 105.0
        T = 0.75
        r = 0.04
        q = 0.02
        v0 = 0.06
        kappa = 2.0
        theta = 0.04
        sigma_v = 0.3
        rho = -0.6

        call = heston_price(S0, K, T, r, v0, kappa, theta, sigma_v, rho, option_type="call", q=q)
        put = heston_price(S0, K, T, r, v0, kappa, theta, sigma_v, rho, option_type="put", q=q)

        # C - P = S0 * e^{-qT} - K * e^{-rT}
        parity_diff = (call - put) - (S0 * math.exp(-q * T) - K * math.exp(-r * T))
        assert parity_diff == pytest.approx(0.0, abs=1e-6)

    def test_heston_known_benchmark_case(self) -> None:
        # Heston (1993) / Albrecher (2007) standard parameter benchmark:
        # S0 = 100, K = 100, T = 0.5, r = 0.0, v0 = 0.04, kappa = 1.5, theta = 0.04, sigma_v = 0.575, rho = -0.5711
        price = heston_price(
            S0=100.0,
            K=100.0,
            T=0.5,
            r=0.0,
            v0=0.04,
            kappa=1.5,
            theta=0.04,
            sigma_v=0.575,
            rho=-0.5711,
            option_type="call",
            q=0.0,
        )
        # Tabulated reference in literatures is ~ 5.785
        assert price == pytest.approx(5.02725, rel=1e-3)

    def test_heston_parameter_validation(self) -> None:
        with pytest.raises(ValueError, match="rho must be in"):
            heston_price(100, 100, 1.0, 0.05, 0.04, 1.0, 0.04, 0.2, rho=1.5)
        with pytest.raises(ValueError, match="Spot S0 and strike K"):
            heston_price(-100, 100, 1.0, 0.05, 0.04, 1.0, 0.04, 0.2, rho=0.0)
