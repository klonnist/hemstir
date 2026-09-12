"""ATR (Average True Range) hesaplamasi.

crypto-trader projesindeki (okx_trader/indicators.py) formulle birebir ayni tutuldu:
Wilder'in ustel duzgunlestirmesi (ewm alpha=1/period).
"""
import pandas as pd


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
