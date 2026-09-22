"""realized_vol.py icin birim testleri."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from realized_vol import (  # noqa: E402
    build_hourly_log_rv_series,
    garman_klass_variance,
    log_returns,
    log_rv,
    parkinson_variance,
    realized_variance_cc,
    realized_vol_over_window,
    rv_at_cutpoint,
)


def test_log_returns_basic():
    r = log_returns([100, 110, 99])
    assert r[0] == pytest.approx(np.log(1.10), abs=1e-9)
    assert r[1] == pytest.approx(np.log(99 / 110), abs=1e-9)


def test_realized_variance_cc_zero_for_constant_price():
    assert realized_variance_cc([100, 100, 100, 100]) == pytest.approx(0.0, abs=1e-12)


def test_realized_variance_cc_matches_manual_sum_of_squares():
    closes = [100, 105, 98, 102]
    r = log_returns(closes)
    expected = float(np.sum(r ** 2))
    assert realized_variance_cc(closes) == pytest.approx(expected)


def test_parkinson_variance_zero_when_no_range():
    assert parkinson_variance([100, 100], [100, 100]) == pytest.approx(0.0, abs=1e-12)


def test_parkinson_variance_positive_with_range():
    assert parkinson_variance([105, 103], [95, 97]) > 0


def test_garman_klass_zero_for_flat_candle():
    assert garman_klass_variance([100], [100], [100], [100]) == pytest.approx(0.0, abs=1e-12)


def candle(ts, o, h, l, c):
    return {"ts": ts, "open": o, "high": h, "low": l, "close": c}


def test_realized_vol_over_window_none_for_short_input():
    assert realized_vol_over_window([candle("t1", 100, 101, 99, 100)], method="cc") is None
    assert realized_vol_over_window([], method="cc") is None


def test_realized_vol_over_window_unknown_method_raises():
    candles = [candle("t1", 100, 101, 99, 100), candle("t2", 100, 102, 98, 101)]
    with pytest.raises(ValueError):
        realized_vol_over_window(candles, method="bogus")


def test_rv_at_cutpoint_forward_vs_lookback():
    candles = [candle(f"t{i}", 100 + i, 101 + i, 99 + i, 100 + i) for i in range(10)]
    idx_by_ts = {c["ts"]: i for i, c in enumerate(candles)}
    # t4'ten SONRAKI 3 saat (t5,t6,t7) ile t4'TEN ONCEKI 3 saat (t2,t3,t4) farkli olmali
    fwd = rv_at_cutpoint(candles, idx_by_ts, "t4", 3, method="cc", lookback=False)
    back = rv_at_cutpoint(candles, idx_by_ts, "t4", 3, method="cc", lookback=True)
    assert fwd is not None and back is not None


def test_rv_at_cutpoint_none_when_insufficient_future():
    candles = [candle(f"t{i}", 100, 101, 99, 100) for i in range(5)]
    idx_by_ts = {c["ts"]: i for i, c in enumerate(candles)}
    assert rv_at_cutpoint(candles, idx_by_ts, "t4", 10, lookback=False) is None


def test_log_rv_handles_none_and_zero():
    assert log_rv(None) is None
    assert log_rv(0.0) == pytest.approx(np.log(1e-12), abs=1e-6)


def test_build_hourly_log_rv_series_length_and_leading_none():
    candles = [candle(f"t{i}", 100 + i, 101 + i, 99 + i, 100 + i) for i in range(30)]
    series = build_hourly_log_rv_series(candles, window_hours=24, method="cc")
    assert len(series) == 30
    assert all(v is None for v in series[:23])
    assert all(v is not None for v in series[23:])
