"""Regression: HostThrottle must not leak memory from stale bucket entries.

The ``_last`` dict grows one entry per unique host bucket and never
evicts stale entries. A batch job that queries many distinct hosts
(e.g. N tickers across N providers) accumulates dead entries without
bound. The fix adds a periodic sweep that drops buckets whose reserved
fire time has passed.
"""

from __future__ import annotations

import time

from backtest.loaders._http import HostThrottle


def test_wait_records_bucket() -> None:
    throttle = HostThrottle()
    throttle.wait("eastmoney", min_interval=1.0)
    assert "eastmoney" in throttle._last


def test_wait_zero_interval_no_op() -> None:
    throttle = HostThrottle()
    throttle.wait("x", min_interval=0.0)
    # min_interval <= 0 returns early without recording.
    assert "x" not in throttle._last


def test_sweep_removes_stale_buckets() -> None:
    """Buckets whose fire time has passed must be swept."""
    throttle = HostThrottle()
    throttle.wait("a", min_interval=1.0)
    throttle.wait("b", min_interval=1.0)
    assert len(throttle._last) == 2

    # The fire_at for both is ~now (slot was free). Wait a moment so
    # they become stale (fire_at < now).
    time.sleep(0.02)

    now = time.monotonic()
    with throttle._lock:
        throttle._sweep_stale_locked(now)

    assert "a" not in throttle._last
    assert "b" not in throttle._last


def test_sweep_keeps_active_buckets() -> None:
    """Buckets with a future fire time must survive the sweep."""
    throttle = HostThrottle()
    # Manually set a future fire time to avoid the real 10s sleep.
    with throttle._lock:
        throttle._last["a"] = time.monotonic() + 100.0

    now = time.monotonic()
    with throttle._lock:
        throttle._sweep_stale_locked(now)
    assert "a" in throttle._last


def test_periodic_sweep_on_wait() -> None:
    """wait() should trigger a sweep after the sweep interval."""
    throttle = HostThrottle()
    throttle.wait("a", min_interval=1.0)

    # Simulate that the last sweep was long ago.
    throttle._last_sweep = time.monotonic() - 61.0

    time.sleep(0.02)

    # This wait should trigger a sweep, removing "a" (stale) and adding "b".
    throttle.wait("b", min_interval=1.0)

    assert "a" not in throttle._last
    assert "b" in throttle._last


def test_many_unique_buckets_do_not_grow_unboundedly() -> None:
    """After a sweep, stale buckets from earlier calls must be gone."""
    throttle = HostThrottle()
    # Simulate old last_sweep so the next wait triggers a sweep.
    throttle._last_sweep = time.monotonic() - 61.0

    # Add 100 buckets with past fire times (stale).
    for i in range(100):
        throttle._last[f"host-{i}"] = time.monotonic() - 1.0

    assert len(throttle._last) == 100

    # This wait triggers a sweep that should remove all stale entries.
    throttle.wait("new-host", min_interval=1.0)

    assert len(throttle._last) == 1
    assert "new-host" in throttle._last
