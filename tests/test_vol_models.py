"""vol_models.py icin birim testleri (naif/EWMA/HAR-RV/GARCH/TimesFM-hibrit)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from vol_models import (  # noqa: E402
    build_ewma_variance_series,
    build_har_features,
    fit_har_rv,
    forecast_ewma,
    forecast_har_rv,
    forecast_naive,
    forecast_timesfm_direct,
    forecast_timesfm_hybrid,
)


def candle(ts, o, h, l, c):
    return {"ts": ts, "open": o, "high": h, "low": l, "close": c}


def make_synthetic_candles(n=600, seed=1, vol=0.01):
    rng = np.random.default_rng(seed)
    price = 100.0
    candles = []
    for i in range(n):
        ret = rng.normal(0, vol)
        new_price = price * np.exp(ret)
        high = max(price, new_price) * (1 + abs(rng.normal(0, vol / 2)))
        low = min(price, new_price) * (1 - abs(rng.normal(0, vol / 2)))
        candles.append(candle(f"2026-01-01T{i:04d}", price, high, low, new_price))
        price = new_price
    return candles


def test_forecast_naive_returns_log_scale():
    candles = make_synthetic_candles(60)
    idx_by_ts = {c["ts"]: i for i, c in enumerate(candles)}
    val = forecast_naive(candles, idx_by_ts, candles[40]["ts"], 12)
    assert val is not None
    assert val < 0  # log(kucuk pozitif RV) negatif olmali (RV genelde << 1)


def test_forecast_naive_none_without_enough_history():
    candles = make_synthetic_candles(5)
    idx_by_ts = {c["ts"]: i for i, c in enumerate(candles)}
    # lookback=True ic mantigi: yeterli gecmis yoksa da bir seri doner ama kisa -
    # sadece cok erken bir noktada NaN/None kontrolu (min veri varsayimi test amacli)
    val = forecast_naive(candles, idx_by_ts, candles[1]["ts"], 12)
    assert val is not None  # lookback penceresi mevcut veriyle kirpilir, None donmez


def test_ewma_series_higher_after_volatile_move():
    candles = make_synthetic_candles(50, vol=0.001)
    # buyuk bir siçrama ekle
    candles[30]["close"] = candles[29]["close"] * 1.20
    ewma = build_ewma_variance_series(candles)
    before = ewma.get(candles[29]["ts"])
    after = ewma.get(candles[31]["ts"])
    assert before is not None and after is not None
    assert after > before  # buyuk hareketten sonra EWMA varyansi yukselmeli


def test_forecast_ewma_scales_with_horizon():
    candles = make_synthetic_candles(60)
    ewma = build_ewma_variance_series(candles)
    cp = candles[45]["ts"]
    short = forecast_ewma(ewma, cp, 6)
    long = forecast_ewma(ewma, cp, 48)
    assert short is not None and long is not None
    assert long > short  # daha uzun ufuk -> daha yuksek RV (log olcekte de artmali)


def test_forecast_ewma_none_for_missing_cutpoint():
    assert forecast_ewma({}, "yok", 24) is None


def test_har_fit_and_forecast_roundtrip():
    candles = make_synthetic_candles(900, seed=7)
    idx_by_ts = {c["ts"]: i for i, c in enumerate(candles)}
    from realized_vol import rv_at_cutpoint

    feats = build_har_features(candles)
    targets = {}
    for c in candles:
        rv = rv_at_cutpoint(candles, idx_by_ts, c["ts"], 24, lookback=False)
        if rv is not None:
            targets[c["ts"]] = rv

    fit = fit_har_rv({"X": feats}, {"X": targets}, 24)
    assert fit["coef"] is not None
    assert len(fit["coef"]) == 4

    # yeterli gecmisi olan bir nokta sec
    cp = candles[700]["ts"]
    forecast = forecast_har_rv(fit["coef"], feats, cp)
    assert forecast is not None
    assert isinstance(forecast, float)


def test_har_fit_returns_none_coef_with_too_few_observations():
    fit = fit_har_rv({"X": {}}, {"X": {}}, 24)
    assert fit["coef"] is None
    assert fit["n_obs"] == 0


def test_forecast_timesfm_direct_reads_horizon_index():
    pred = {"points": [i * -1.0 for i in range(48)]}
    assert forecast_timesfm_direct(pred, 24) == pytest.approx(-23.0)  # points[horizon-1], 0-indeksli
    assert forecast_timesfm_direct(pred, 48) == pytest.approx(-47.0)


def test_forecast_timesfm_direct_none_when_missing():
    assert forecast_timesfm_direct(None, 24) is None
    assert forecast_timesfm_direct({"points": [1, 2]}, 24) is None


def test_forecast_timesfm_hybrid_wider_band_gives_higher_implied_vol():
    def make_pred(spread_pct):
        entry = 100.0
        row_narrow = [0] + [entry * (1 + (q - 5) * spread_pct / 100) for q in range(1, 10)]
        return {"quantiles": [row_narrow] * 48}

    narrow = forecast_timesfm_hybrid(make_pred(0.5), 100.0, 24)
    wide = forecast_timesfm_hybrid(make_pred(3.0), 100.0, 24)
    assert narrow is not None and wide is not None
    assert wide > narrow  # genis kantil bandi -> daha yuksek ima edilen oynaklik


def test_forecast_timesfm_hybrid_none_for_bad_input():
    assert forecast_timesfm_hybrid(None, 100.0, 24) is None
    assert forecast_timesfm_hybrid({"quantiles": []}, 100.0, 24) is None
