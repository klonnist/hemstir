"""Gerceklesen oynaklik (realized volatility, RV) hesaplama.

Iki tahminci:
  (a) RV (close-close): saatlik log getirilerin karelerinin toplami - standart,
      gurultulu (sadece kapanis fiyatlarini kullanir).
  (b) Garman-Klass / Parkinson: OHLC'nin tamamini (ya da sadece high/low) kullanir,
      ayni veri miktariyla DAHA AZ GURULTULU bir tahmindir (Parkinson 1980,
      Garman-Klass 1980 - literaturun standart sonuclari).

Modelleme HER ZAMAN log(RV) uzerinde yapilir (RV dagilimi cok sag-carpik) - bkz.
vol_models.py. Tum fonksiyonlar saf (ag erisimi yok), candles listesi uzerinde
calisir (her eleman en az {ts, open, high, low, close} icerir).
"""
import numpy as np

RV_EPSILON = 1e-12  # log(0) onlemek icin


def log_returns(closes) -> np.ndarray:
    closes = np.asarray(closes, dtype=float)
    return np.diff(np.log(closes))


def realized_variance_cc(closes) -> float:
    """Close-close RV: saatlik log getirilerin karelerinin toplami (varyans olcekli,
    yillik/normallize DEGIL - dogrudan 'bu pencerede biriken varyans')."""
    r = log_returns(closes)
    return float(np.sum(r ** 2))


def parkinson_variance(highs, lows) -> float:
    """Parkinson (1980): sadece high/low kullanir, close-close'dan daha az gurultulu."""
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    factor = 1.0 / (4.0 * np.log(2.0))
    terms = factor * (np.log(highs / lows) ** 2)
    return float(np.sum(terms))


def garman_klass_variance(opens, highs, lows, closes) -> float:
    """Garman-Klass (1980): OHLC'nin tamamini kullanir."""
    opens = np.asarray(opens, dtype=float)
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = np.asarray(closes, dtype=float)
    hl = 0.5 * (np.log(highs / lows) ** 2)
    co = (2 * np.log(2) - 1) * (np.log(closes / opens) ** 2)
    return float(np.sum(hl - co))


def realized_vol_over_window(candles: list, method: str = "cc") -> float:
    """candles: {open,high,low,close} iceren dict listesi (kronolojik, pencerenin
    KENDISI - cagiran taraf hangi araligi verdiyse onun uzerinden hesaplar).
    method: 'cc' (close-close), 'parkinson', 'gk' (Garman-Klass). Bos/yetersiz
    veri icin None doner."""
    if not candles or len(candles) < 2:
        return None
    if method == "cc":
        closes = [c["close"] for c in candles]
        return realized_variance_cc(closes)
    elif method == "parkinson":
        return parkinson_variance([c["high"] for c in candles], [c["low"] for c in candles])
    elif method == "gk":
        return garman_klass_variance([c["open"] for c in candles], [c["high"] for c in candles],
                                      [c["low"] for c in candles], [c["close"] for c in candles])
    raise ValueError(f"bilinmeyen yontem: {method}")


def rv_at_cutpoint(candles: list, idx_by_ts: dict, cutpoint_ts: str, horizon_hours: int,
                    method: str = "cc", lookback: bool = False):
    """cutpoint'ten SONRAKI (lookback=False, HEDEF/gerceklesen RV - degerlendirme icin)
    ya da ONCEKI (lookback=True, OZELLIK/gecmis RV - tahmin girdisi icin) horizon_hours
    saatlik penceredeki RV'yi hesaplar. Yeterli veri yoksa None doner."""
    row_idx = idx_by_ts.get(cutpoint_ts)
    if row_idx is None:
        return None
    if lookback:
        window = candles[max(0, row_idx - horizon_hours + 1): row_idx + 1]
    else:
        window = candles[row_idx + 1: row_idx + 1 + horizon_hours]
        if len(window) < horizon_hours:
            return None
    return realized_vol_over_window(window, method=method)


def log_rv(rv) -> float:
    if rv is None:
        return None
    return float(np.log(max(rv, 0.0) + RV_EPSILON))


def build_hourly_log_rv_series(candles: list, window_hours: int = 24, method: str = "cc") -> list:
    """Her saat icin, O SAATTEN GERIYE DOGRU window_hours'lik gercek zamanli
    (trailing) RV'nin log'unu hesaplar - TimesFM'e (ve HAR-RV bilesenlerine) 'gecmis
    RV serisi' olarak beslemek icin. candles[i] icin deger, candles[i-window_hours+1:i+1]
    penceresinden hesaplanir; yeterli gecmisi olmayan ilk `window_hours-1` nokta None'dir.
    Doner: ayni uzunlukta (candles ile) log(RV) listesi."""
    n = len(candles)
    out = [None] * n
    closes = np.array([c["close"] for c in candles], dtype=float)
    highs = np.array([c["high"] for c in candles], dtype=float)
    lows = np.array([c["low"] for c in candles], dtype=float)
    opens = np.array([c["open"] for c in candles], dtype=float)
    for i in range(window_hours - 1, n):
        lo = i - window_hours + 1
        if method == "cc":
            rv = realized_variance_cc(closes[lo:i + 1])
        elif method == "parkinson":
            rv = parkinson_variance(highs[lo:i + 1], lows[lo:i + 1])
        elif method == "gk":
            rv = garman_klass_variance(opens[lo:i + 1], highs[lo:i + 1], lows[lo:i + 1], closes[lo:i + 1])
        else:
            raise ValueError(f"bilinmeyen yontem: {method}")
        out[i] = log_rv(rv)
    return out
