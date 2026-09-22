"""eval_lib.simulate_trade ve yardimci fonksiyonlar icin birim testleri.

Calistirma: `python -m pytest tests/` (repo kok dizininden).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone  # noqa: E402

import pytest  # noqa: E402

from eval_lib import (  # noqa: E402
    aggregate_trades,
    block_bootstrap_ci,
    direction_correct,
    forecast_error,
    funding_cost_pct,
    funding_events_between,
    in_confidence_band,
    liquidation_price,
    momentum_side,
    paired_diff_significance,
    prob_above_entry,
    simulate_trade,
)


def dt(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def candle(ts, high, low, close):
    return {"ts": ts, "high": high, "low": low, "close": close}


# ---------------------------------------------------------------------------
# simulate_trade - temiz TP / SL
# ---------------------------------------------------------------------------

def test_buy_take_profit_hit_cleanly():
    candles = [candle("t1", high=105, low=98, close=104), candle("t2", high=112, low=104, close=111)]
    result = simulate_trade(entry_price=100, side="BUY", take_profit=110, stop_loss=90, future_candles=candles)
    assert result.exit_reason == "TP"
    assert result.exit_price == 110
    assert result.hours_held == 2
    assert result.pnl_pct == pytest.approx(10 - 0.1, abs=1e-9)  # %10 hareket - %0.1 komisyon


def test_sell_stop_loss_hit_cleanly():
    candles = [candle("t1", high=106, low=99, close=101)]
    result = simulate_trade(entry_price=100, side="SELL", take_profit=90, stop_loss=105, future_candles=candles)
    assert result.exit_reason == "SL"
    assert result.exit_price == 105
    # SELL: (105-100)/100 * -1 * 100 = -5%  - %0.1 komisyon
    assert result.pnl_pct == pytest.approx(-5 - 0.1, abs=1e-9)


# ---------------------------------------------------------------------------
# Ayni mumda hem TP hem SL - temkinli varsayim: kayip (SL) sayilmali
# ---------------------------------------------------------------------------

def test_buy_same_candle_tp_and_sl_counts_as_loss():
    candles = [candle("t1", high=112, low=90, close=100)]  # hem TP(110) hem SL(95) araliginda
    result = simulate_trade(entry_price=100, side="BUY", take_profit=110, stop_loss=95, future_candles=candles)
    assert result.exit_reason == "SL"
    assert result.exit_price == 95
    assert result.pnl_pct < 0


def test_sell_same_candle_tp_and_sl_counts_as_loss():
    candles = [candle("t1", high=106, low=88, close=100)]  # hem TP(90) hem SL(105) araliginda
    result = simulate_trade(entry_price=100, side="SELL", take_profit=90, stop_loss=105, future_candles=candles)
    assert result.exit_reason == "SL"
    assert result.exit_price == 105
    assert result.pnl_pct < 0


# ---------------------------------------------------------------------------
# Sure doldu / yetersiz veri
# ---------------------------------------------------------------------------

def test_expires_at_horizon_when_neither_hit():
    candles = [
        candle("t1", high=105, low=95, close=102),
        candle("t2", high=108, low=98, close=104),
        candle("t3", high=110, low=100, close=106),
    ]
    result = simulate_trade(
        entry_price=100, side="BUY", take_profit=120, stop_loss=80,
        future_candles=candles, horizon_hours=3,
    )
    assert result.exit_reason == "EXPIRED"
    assert result.exit_price == 106  # son mumun kapanisi
    assert result.hours_held == 3
    assert result.pnl_pct == pytest.approx(6 - 0.1, abs=1e-9)


def test_insufficient_data_when_fewer_candles_than_horizon():
    candles = [candle("t1", high=105, low=95, close=102)]
    result = simulate_trade(
        entry_price=100, side="BUY", take_profit=120, stop_loss=80,
        future_candles=candles, horizon_hours=48,
    )
    assert result.exit_reason == "INSUFFICIENT_DATA"
    assert result.hours_held == 1


def test_no_future_candles_at_all():
    result = simulate_trade(entry_price=100, side="BUY", take_profit=120, stop_loss=80, future_candles=[])
    assert result.exit_reason == "INSUFFICIENT_DATA"
    assert result.pnl_pct is None


# ---------------------------------------------------------------------------
# Kaldirac / likidasyon
# ---------------------------------------------------------------------------

def test_liquidation_price_formula():
    assert liquidation_price(100, "BUY", leverage=1.0) is None  # kaldiracsiz -> likidasyon yok
    assert liquidation_price(100, "BUY", leverage=10) == pytest.approx(90)
    assert liquidation_price(100, "SELL", leverage=10) == pytest.approx(110)


def test_high_leverage_triggers_liquidation_before_wider_sl():
    # SL genis (%20), ama 10x kaldiracta likidasyon %10'da - once likidasyon tetiklenmeli.
    candles = [candle("t1", high=101, low=85, close=95)]
    result = simulate_trade(
        entry_price=100, side="BUY", take_profit=150, stop_loss=80,
        future_candles=candles, leverage=10,
    )
    assert result.exit_reason == "LIQUIDATION"
    assert result.exit_price == pytest.approx(90)
    assert result.pnl_pct == -100.0


def test_leverage_1x_never_liquidates_even_with_big_drop():
    candles = [candle("t1", high=101, low=50, close=95)]  # %50 dusus ama kaldirac yok
    result = simulate_trade(
        entry_price=100, side="BUY", take_profit=150, stop_loss=40,
        future_candles=candles, leverage=1,
    )
    assert result.exit_reason != "LIQUIDATION"


# ---------------------------------------------------------------------------
# Yon dogrulugu / tahmin hatasi / guven araligi
# ---------------------------------------------------------------------------

def test_direction_correct_buy_and_sell():
    assert direction_correct("BUY", 100, 105) is True
    assert direction_correct("BUY", 100, 95) is False
    assert direction_correct("SELL", 100, 95) is True
    assert direction_correct("SELL", 100, 105) is False
    assert direction_correct("BUY", 100, None) is None


def test_forecast_error_basic():
    abs_err, pct_err = forecast_error(110, 100)
    assert abs_err == pytest.approx(10)
    assert pct_err == pytest.approx(10)
    assert forecast_error(None, 100) == (None, None)


def test_in_confidence_band():
    assert in_confidence_band(105, 100, 110) is True
    assert in_confidence_band(95, 100, 110) is False
    assert in_confidence_band(None, 100, 110) is None


def test_momentum_side():
    assert momentum_side([100, 101, 102, 110]) == "BUY"
    assert momentum_side([110, 105, 102, 100]) == "SELL"


# ---------------------------------------------------------------------------
# Toplulastirma
# ---------------------------------------------------------------------------

def test_aggregate_trades_basic():
    trades = [
        {"exit_reason": "TP", "pnl_pct": 10.0, "entry_ts": "t1", "direction_correct": True},
        {"exit_reason": "SL", "pnl_pct": -5.0, "entry_ts": "t2", "direction_correct": False},
    ]
    agg = aggregate_trades(trades)
    assert agg["count"] == 2
    assert agg["win_rate_pct"] == 50.0
    assert agg["direction_accuracy_pct"] == 50.0
    assert agg["profit_factor"] == pytest.approx(2.0)
    assert agg["total_return_pct"] == pytest.approx(5.0)
    assert agg["max_drawdown_pct"] == pytest.approx(-5.0)
    assert agg["exit_distribution"] == {"TP": 1, "SL": 1}


def test_aggregate_trades_empty():
    agg = aggregate_trades([])
    assert agg["count"] == 0
    assert agg["win_rate_pct"] is None


# ---------------------------------------------------------------------------
# Funding maliyeti
# ---------------------------------------------------------------------------

def test_funding_events_between_finds_8h_boundaries():
    events = funding_events_between(dt("2026-01-01T01:00:00Z"), dt("2026-01-01T20:00:00Z"))
    assert events == [dt("2026-01-01T08:00:00Z"), dt("2026-01-01T16:00:00Z")]


def test_funding_events_between_no_events_if_short_trade():
    events = funding_events_between(dt("2026-01-01T01:00:00Z"), dt("2026-01-01T03:00:00Z"))
    assert events == []


def test_funding_cost_long_pays_when_rate_positive():
    lookup = {dt("2026-01-01T08:00:00Z"): 0.0001, dt("2026-01-01T16:00:00Z"): 0.0001}  # %0.01 her biri
    cost = funding_cost_pct(dt("2026-01-01T01:00:00Z"), dt("2026-01-01T20:00:00Z"), "BUY", 1.0, lookup)
    assert cost == pytest.approx(0.02, abs=1e-6)  # 2 olay * %0.01


def test_funding_cost_short_gains_when_rate_positive():
    lookup = {dt("2026-01-01T08:00:00Z"): 0.0001}
    cost = funding_cost_pct(dt("2026-01-01T01:00:00Z"), dt("2026-01-01T09:00:00Z"), "SELL", 1.0, lookup)
    assert cost == pytest.approx(-0.01, abs=1e-6)


def test_funding_cost_falls_back_to_default_when_missing():
    cost = funding_cost_pct(dt("2026-01-01T01:00:00Z"), dt("2026-01-01T09:00:00Z"), "BUY", 1.0,
                             funding_lookup={}, default_rate_pct=0.02)
    assert cost == pytest.approx(0.02, abs=1e-6)


def test_funding_cost_scales_with_leverage():
    lookup = {dt("2026-01-01T08:00:00Z"): 0.0001}
    cost = funding_cost_pct(dt("2026-01-01T01:00:00Z"), dt("2026-01-01T09:00:00Z"), "BUY", 5.0, lookup)
    assert cost == pytest.approx(0.05, abs=1e-6)


# ---------------------------------------------------------------------------
# Olasilik filtresi
# ---------------------------------------------------------------------------

def test_prob_above_entry_at_median_is_half():
    # q10..q90 esit araliklarla 90..170, medyan (q50, 5. eleman) = 130
    qs = [0] + [90, 100, 110, 120, 130, 140, 150, 160, 170]
    assert prob_above_entry(130, qs) == pytest.approx(0.5, abs=1e-6)


def test_prob_above_entry_below_band_is_high():
    qs = [0] + [90, 100, 110, 120, 130, 140, 150, 160, 170]
    assert prob_above_entry(50, qs) == 0.95


def test_prob_above_entry_above_band_is_low():
    qs = [0] + [90, 100, 110, 120, 130, 140, 150, 160, 170]
    assert prob_above_entry(200, qs) == 0.05


def test_prob_above_entry_none_when_insufficient_data():
    assert prob_above_entry(100, None) is None
    assert prob_above_entry(100, [1, 2, 3]) is None


# ---------------------------------------------------------------------------
# Blok bootstrap
# ---------------------------------------------------------------------------

def test_block_bootstrap_ci_constant_series_is_tight():
    res = block_bootstrap_ci([1.0] * 100, block_size=10, n_boot=500, seed=1)
    assert res["mean"] == pytest.approx(1.0)
    assert res["ci_low"] == pytest.approx(1.0, abs=1e-9)
    assert res["ci_high"] == pytest.approx(1.0, abs=1e-9)
    assert res["effective_n"] == 10


def test_block_bootstrap_ci_empty():
    res = block_bootstrap_ci([])
    assert res["n"] == 0
    assert res["mean"] is None


def test_paired_diff_significance_detects_clear_difference():
    a = [2.0] * 50
    b = [0.0] * 50
    res = paired_diff_significance(a, b, block_size=5, n_boot=500, seed=1)
    assert res["significant"] is True
    assert res["diff_ci"]["ci_low"] > 0


def test_paired_diff_significance_no_difference_when_identical():
    a = [1.0, -1.0] * 25
    res = paired_diff_significance(a, a, block_size=5, n_boot=500, seed=1)
    assert res["significant"] is False
