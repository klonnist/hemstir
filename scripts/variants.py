"""Onceden tanimlanmis, SABIT (taranmayan) 6 backtest varyanti (A-F):

  A) Referans: mevcut sistem - 1H ATR(14), TP=2.5xATR/SL=1.5xATR, ufuk=48s, baglam=300s.
  B) Saf yon: TP/SL yok, 48. saat kapanisinda cik.
  C) Ufka uyumlu TP/SL: ATR'yi 4H mumlardan hesapla, ayni 1.5/2.5 katsayilari.
  D) Olasilik filtresi: kantillerden P(48s sonunda giris yonunde) hesapla, sadece
     >= PROB_THRESHOLD ise islem ac (esik sabit - A'nin TP/SL kurallarini kullanir).
  E) Kisa ufuk: 24 saatlik tahmin, 24. saat kapanisinda cik (TP/SL yok).
  F) Uzun baglam: 1024 saatlik baglam, B'nin (saf yon) cikis kuraliyla - teshis
     raporunun "TP/SL erken/gurultulu cikiyor" bulgusuna dayanarak, baglam
     etkisini TP/SL karisikligi olmadan izole etmek icin B secildi.

Her varyant, scripts/predict_cache.py'nin urettigi HAM tahmin onbellegini (A/B/C/D/E
icin context=300, F icin context=1024) + ham OHLCV + (varsa) gercek funding verisini
kullanir; TimesFM BURADA TEKRAR CAGRILMAZ - sadece cikis kurali/post-processing.
"""
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_lib import direction_correct, funding_cost_pct, prob_above_entry, simulate_trade  # noqa: E402
from generate_forecasts import ATR_PERIOD, compute_tp_sl, signal_strength  # noqa: E402
from indicators import atr as compute_atr  # noqa: E402

PROB_THRESHOLD = 0.60  # Varyant D - sabit, taranmiyor
SHORT_HORIZON_HOURS = 24  # Varyant E
LONG_CONTEXT_HOURS = 1024  # Varyant F


def parse(s: str):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def fmt(dt_val) -> str:
    return dt_val.strftime("%Y-%m-%dT%H:%M:%SZ")


def _atr_1h(candles: list, idx_by_ts: dict, cutpoint_ts: str, period: int = ATR_PERIOD):
    row_idx = idx_by_ts.get(cutpoint_ts)
    if row_idx is None or row_idx + 1 < period + 1:
        return None
    window = candles[max(0, row_idx - period * 3): row_idx + 1]
    df = pd.DataFrame(window)[["high", "low", "close"]]
    series = compute_atr(df, period=period)
    val = series.iloc[-1]
    return float(val) if not np.isnan(val) else None


def _atr_4h(candles: list, idx_by_ts: dict, cutpoint_ts: str, period: int = ATR_PERIOD):
    """Son 1H mumlari 4H'e resample edip ATR(14) hesaplar (ufka daha uyumlu, daha
    az gurultulu bir volatilite olcusu icin - bkz. teshis raporundaki erken-SL bulgusu)."""
    row_idx = idx_by_ts.get(cutpoint_ts)
    if row_idx is None:
        return None
    lookback_hours = period * 4 * 6  # yeterli 4H mum icin pay birak
    window = candles[max(0, row_idx - lookback_hours): row_idx + 1]
    if len(window) < period * 4:
        return None
    df = pd.DataFrame(window)
    df["ts_dt"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts_dt")
    ohlc_4h = df.resample("4h", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    if len(ohlc_4h) < period + 1:
        return None
    series = compute_atr(ohlc_4h, period=period)
    val = series.iloc[-1]
    return float(val) if not np.isnan(val) else None


def _future_candles(candles: list, idx_by_ts: dict, cutpoint_ts: str, horizon_hours: int):
    row_idx = idx_by_ts.get(cutpoint_ts)
    if row_idx is None:
        return []
    return candles[row_idx + 1: row_idx + 1 + horizon_hours]


def _finalize(result, entry_ts, side, leverage, fee_pct, funding_lookup, default_funding_rate_pct):
    """simulate_trade sonucuna funding maliyetini ekler, JSON-yazilabilir dict doner."""
    if result.exit_ts is None or result.pnl_pct is None:
        return None
    exit_ts = parse(result.exit_ts)
    funding = funding_cost_pct(entry_ts, exit_ts, side, leverage, funding_lookup, default_funding_rate_pct)
    net_with_funding = round(result.pnl_pct - funding, 6)
    return {
        "exit_reason": result.exit_reason,
        "exit_ts": result.exit_ts,
        "exit_price": result.exit_price,
        "hours_held": result.hours_held,
        "pnl_pct_no_funding": result.pnl_pct,
        "funding_cost_pct": funding,
        "pnl_pct": net_with_funding,
    }


def _base_record(symbol, side, entry_price, entry_ts, strength=None, prob=None):
    return {
        "coin": symbol,
        "entry_ts": fmt(entry_ts),
        "entry_price": entry_price,
        "side": side,
        "strength": strength,
        "prob_direction": prob,
    }


def _direction_flag(side, entry_price, future_candles):
    if not future_candles:
        return None
    return direction_correct(side, entry_price, future_candles[-1]["close"])


# ---------------------------------------------------------------------------
# Varyantlar - hepsi ayni imzayi kullanir: (pred, candles, idx_by_ts, funding_lookup,
# leverage, fee_pct, default_funding_rate_pct, pred_long=None) -> trade dict | None
# ---------------------------------------------------------------------------

def variant_A(pred, candles, idx_by_ts, funding_lookup, leverage, fee_pct, default_funding_rate_pct, pred_long=None, side_override=None):
    symbol, cutpoint_ts = pred["coin"], pred["cutpoint_ts"]
    entry_price, points = pred["entry_price"], pred["points"]
    if len(points) < 48:
        return None
    atr_value = _atr_1h(candles, idx_by_ts, cutpoint_ts)
    if atr_value is None:
        return None
    side = side_override or ("BUY" if points[47] >= entry_price else "SELL")
    sl, tp = compute_tp_sl(entry_price, atr_value, side)
    future = _future_candles(candles, idx_by_ts, cutpoint_ts, 48)
    if len(future) < 48:
        return None
    result = simulate_trade(entry_price, side, tp, sl, future, leverage=leverage, fee_pct=fee_pct, horizon_hours=48)
    rec = _base_record(symbol, side, entry_price, parse(cutpoint_ts),
                        strength=signal_strength(points[47] - entry_price, atr_value))
    fin = _finalize(result, parse(cutpoint_ts), side, leverage, fee_pct, funding_lookup, default_funding_rate_pct)
    if fin is None:
        return None
    rec.update(fin)
    rec["direction_correct"] = _direction_flag(side, entry_price, future)
    return rec


def variant_B(pred, candles, idx_by_ts, funding_lookup, leverage, fee_pct, default_funding_rate_pct, pred_long=None, side_override=None):
    """Saf yon: TP/SL yok, 48. saat kapanisinda cik."""
    symbol, cutpoint_ts = pred["coin"], pred["cutpoint_ts"]
    entry_price, points = pred["entry_price"], pred["points"]
    if len(points) < 48:
        return None
    side = side_override or ("BUY" if points[47] >= entry_price else "SELL")
    future = _future_candles(candles, idx_by_ts, cutpoint_ts, 48)
    if len(future) < 48:
        return None
    # TP/SL'i erisilemeyecek kadar uzaga koyarak "asla tetiklenmesin" saglanir -
    # simulate_trade'in ayni mantigini (komisyon, sure dolumu) yeniden kullanmak icin.
    extreme = entry_price * 1000
    sl, tp = (0.0, extreme) if side == "BUY" else (extreme, 0.0)
    result = simulate_trade(entry_price, side, tp, sl, future, leverage=leverage, fee_pct=fee_pct, horizon_hours=48)
    rec = _base_record(symbol, side, entry_price, parse(cutpoint_ts))
    fin = _finalize(result, parse(cutpoint_ts), side, leverage, fee_pct, funding_lookup, default_funding_rate_pct)
    if fin is None:
        return None
    rec.update(fin)
    rec["direction_correct"] = _direction_flag(side, entry_price, future)
    return rec


def variant_C(pred, candles, idx_by_ts, funding_lookup, leverage, fee_pct, default_funding_rate_pct, pred_long=None, side_override=None):
    """Ufka uyumlu TP/SL: ATR 4H mumlardan (A ile ayni katsayilar, farkli ATR kaynagi)."""
    symbol, cutpoint_ts = pred["coin"], pred["cutpoint_ts"]
    entry_price, points = pred["entry_price"], pred["points"]
    if len(points) < 48:
        return None
    atr_value = _atr_4h(candles, idx_by_ts, cutpoint_ts)
    if atr_value is None:
        return None
    side = side_override or ("BUY" if points[47] >= entry_price else "SELL")
    sl, tp = compute_tp_sl(entry_price, atr_value, side)
    future = _future_candles(candles, idx_by_ts, cutpoint_ts, 48)
    if len(future) < 48:
        return None
    result = simulate_trade(entry_price, side, tp, sl, future, leverage=leverage, fee_pct=fee_pct, horizon_hours=48)
    rec = _base_record(symbol, side, entry_price, parse(cutpoint_ts),
                        strength=signal_strength(points[47] - entry_price, atr_value))
    fin = _finalize(result, parse(cutpoint_ts), side, leverage, fee_pct, funding_lookup, default_funding_rate_pct)
    if fin is None:
        return None
    rec.update(fin)
    rec["direction_correct"] = _direction_flag(side, entry_price, future)
    return rec


def variant_D(pred, candles, idx_by_ts, funding_lookup, leverage, fee_pct, default_funding_rate_pct, pred_long=None, side_override=None):
    """Olasilik filtresi: sadece P(yon dogru) >= PROB_THRESHOLD ise islem ac (A'nin
    TP/SL kurallarini kullanir). Esik altinda kalan sinyaller None doner (islem yok)."""
    symbol, cutpoint_ts = pred["coin"], pred["cutpoint_ts"]
    entry_price, points, quantiles = pred["entry_price"], pred["points"], pred.get("quantiles")
    if len(points) < 48 or not quantiles or len(quantiles) < 48:
        return None
    side = side_override or ("BUY" if points[47] >= entry_price else "SELL")
    prob_up = prob_above_entry(entry_price, quantiles[47])
    if prob_up is None:
        return None
    prob_direction = prob_up if side == "BUY" else round(1 - prob_up, 4)
    if prob_direction < PROB_THRESHOLD:
        return None  # esik altinda - islem acilmiyor
    atr_value = _atr_1h(candles, idx_by_ts, cutpoint_ts)
    if atr_value is None:
        return None
    sl, tp = compute_tp_sl(entry_price, atr_value, side)
    future = _future_candles(candles, idx_by_ts, cutpoint_ts, 48)
    if len(future) < 48:
        return None
    result = simulate_trade(entry_price, side, tp, sl, future, leverage=leverage, fee_pct=fee_pct, horizon_hours=48)
    rec = _base_record(symbol, side, entry_price, parse(cutpoint_ts),
                        strength=signal_strength(points[47] - entry_price, atr_value), prob=prob_direction)
    fin = _finalize(result, parse(cutpoint_ts), side, leverage, fee_pct, funding_lookup, default_funding_rate_pct)
    if fin is None:
        return None
    rec.update(fin)
    rec["direction_correct"] = _direction_flag(side, entry_price, future)
    return rec


def variant_E(pred, candles, idx_by_ts, funding_lookup, leverage, fee_pct, default_funding_rate_pct, pred_long=None, side_override=None):
    """Kisa ufuk: 24 saatlik tahmin, 24. saat kapanisinda cik (TP/SL yok, B'nin
    kisa-ufuk versiyonu)."""
    symbol, cutpoint_ts = pred["coin"], pred["cutpoint_ts"]
    entry_price, points = pred["entry_price"], pred["points"]
    if len(points) < SHORT_HORIZON_HOURS:
        return None
    side = side_override or ("BUY" if points[SHORT_HORIZON_HOURS - 1] >= entry_price else "SELL")
    future = _future_candles(candles, idx_by_ts, cutpoint_ts, SHORT_HORIZON_HOURS)
    if len(future) < SHORT_HORIZON_HOURS:
        return None
    extreme = entry_price * 1000
    sl, tp = (0.0, extreme) if side == "BUY" else (extreme, 0.0)
    result = simulate_trade(entry_price, side, tp, sl, future, leverage=leverage, fee_pct=fee_pct,
                             horizon_hours=SHORT_HORIZON_HOURS)
    rec = _base_record(symbol, side, entry_price, parse(cutpoint_ts))
    fin = _finalize(result, parse(cutpoint_ts), side, leverage, fee_pct, funding_lookup, default_funding_rate_pct)
    if fin is None:
        return None
    rec.update(fin)
    rec["direction_correct"] = _direction_flag(side, entry_price, future)
    return rec


def variant_F(pred, candles, idx_by_ts, funding_lookup, leverage, fee_pct, default_funding_rate_pct, pred_long=None, side_override=None):
    """Uzun baglam (1024s) + B'nin (saf yon) cikis kurali - baglam uzunlugunun etkisini
    TP/SL karmasasi olmadan izole eder. pred_long: context=1024 onbellegindeki tahmin."""
    if pred_long is None:
        return None
    symbol, cutpoint_ts = pred_long["coin"], pred_long["cutpoint_ts"]
    entry_price, points = pred_long["entry_price"], pred_long["points"]
    if len(points) < 48:
        return None
    side = side_override or ("BUY" if points[47] >= entry_price else "SELL")
    future = _future_candles(candles, idx_by_ts, cutpoint_ts, 48)
    if len(future) < 48:
        return None
    extreme = entry_price * 1000
    sl, tp = (0.0, extreme) if side == "BUY" else (extreme, 0.0)
    result = simulate_trade(entry_price, side, tp, sl, future, leverage=leverage, fee_pct=fee_pct, horizon_hours=48)
    rec = _base_record(symbol, side, entry_price, parse(cutpoint_ts))
    fin = _finalize(result, parse(cutpoint_ts), side, leverage, fee_pct, funding_lookup, default_funding_rate_pct)
    if fin is None:
        return None
    rec.update(fin)
    rec["direction_correct"] = _direction_flag(side, entry_price, future)
    return rec


VARIANTS = {
    "A": variant_A,
    "B": variant_B,
    "C": variant_C,
    "D": variant_D,
    "E": variant_E,
    "F": variant_F,
}

VARIANT_DESCRIPTIONS = {
    "A": "Referans (mevcut sistem): 1H ATR TP/SL, ufuk=48s",
    "B": "Saf yon: TP/SL yok, 48s sonunda cik",
    "C": "Ufka uyumlu TP/SL: ATR 4H mumlardan",
    "D": f"Olasilik filtresi: P(yon) >= %{int(PROB_THRESHOLD * 100)} sartiyla A'nin TP/SL'i",
    "E": "Kisa ufuk: 24s tahmin, 24s sonunda cik",
    "F": "Uzun baglam (1024s) + B'nin saf yon cikisi",
}

# Hangi varyantlarin hangi baglam onbellegini kullandigi (predict_cache.py'nin
# hangi context_hours ile calistirilmasi gerektigini belirler).
VARIANT_CONTEXT = {"A": 300, "B": 300, "C": 300, "D": 300, "E": 300, "F": LONG_CONTEXT_HOURS}
