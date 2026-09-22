"""TimesFM tahmin onbellegi: verilen kesim noktalari + baglam suresi (context_hours)
icin TimesFM'i BATCH halinde (tum coinler tek cagride) calistirir, HAM sonucu
(nokta tahmini + kantiller) JSONL'a APPEND eder.

Onemli: onbellek SADECE HAM MODEL CIKTISINI tutar (ATR/sinyal/TP-SL YOK) - cunku
Varyant A/B/C/D/E'nin hepsi AYNI (context=300, horizon=48) tahmini kullanir, sadece
CIKIS KURALI farklidir (bkz. scripts/variants.py). Boylece tahmin SADECE BIR KEZ
uretilir, 5 varyant icin 5 kez degil.

Ayni (coin, cutpoint_ts, context_hours) uclusu dosyada zaten varsa ATLANIR -
bir calistirma yarida kesilirse (timeout/crash), sonraki calistirma kaldigi
yerden devam eder.
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_forecasts import load_model, run_forecast  # noqa: E402

CACHE_DIR = os.environ.get("RESEARCH_CACHE_DIR", os.path.join("scripts", "cache"))


def cache_path(context_hours: int) -> str:
    return os.path.join(CACHE_DIR, "predictions", f"ctx{context_hours}.jsonl")


def fmt(dt_val) -> str:
    return dt_val.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse(s: str):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def load_cached_keys(context_hours: int) -> set:
    path = cache_path(context_hours)
    keys = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                keys.add((rec["coin"], rec["cutpoint_ts"]))
    return keys


def load_cache(context_hours: int) -> dict:
    """(coin, cutpoint_ts) -> kayit (entry_price, points, quantiles)."""
    path = cache_path(context_hours)
    cache = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cache[(rec["coin"], rec["cutpoint_ts"])] = rec
    return cache


def index_by_ts(candles: list) -> dict:
    return {c["ts"]: i for i, c in enumerate(candles)}


def ensure_predictions(coins_raw: dict, cutpoints: list, context_hours: int,
                        horizon_hours: int = 48, progress_every: int = 20) -> dict:
    """coins_raw: {symbol: [candle dict, ...]} (kronolojik sirali, 'ts'/'close' iceren).
    Eksik (coin, cutpoint) ciftlerini TimesFM ile doldurur, onbellek dosyasina APPEND
    eder, tam onbellegi (tum kayitlar) dondurur."""
    existing_keys = load_cached_keys(context_hours)
    idx = {symbol: index_by_ts(candles) for symbol, candles in coins_raw.items()}

    missing_count = 0
    for cutpoint in cutpoints:
        cp_str = fmt(cutpoint)
        for symbol in coins_raw:
            if (symbol, cp_str) not in existing_keys:
                missing_count += 1

    print(f"context={context_hours}: {len(cutpoints)} kesim noktasi x {len(coins_raw)} coin, "
          f"{missing_count} eksik tahmin var (onbellekte {len(existing_keys)} zaten mevcut).")

    if missing_count == 0:
        return load_cache(context_hours)

    model = None
    path = cache_path(context_hours)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    processed = 0
    with open(path, "a", encoding="utf-8") as f:
        for cp_i, cutpoint in enumerate(cutpoints):
            cp_str = fmt(cutpoint)
            batch_symbols, batch_inputs = [], []
            for symbol, candles in coins_raw.items():
                if (symbol, cp_str) in existing_keys:
                    continue
                row_idx = idx[symbol].get(cp_str)
                if row_idx is None or row_idx + 1 < context_hours:
                    continue
                context_slice = candles[row_idx - context_hours + 1: row_idx + 1]
                closes = np.array([c["close"] for c in context_slice], dtype=np.float64)
                batch_symbols.append(symbol)
                batch_inputs.append(closes)

            if not batch_symbols:
                continue

            if model is None:
                print(f"context={context_hours}: TimesFM yukleniyor...")
                model = load_model(batch_size=len(coins_raw), context_hours=context_hours,
                                    horizon_hours=horizon_hours)

            points_arr, quant_arr = run_forecast(model, batch_inputs, horizon_hours=horizon_hours)

            for i, symbol in enumerate(batch_symbols):
                quantiles = None
                if quant_arr is not None:
                    quantiles = [[round(float(x), 8) for x in row] for row in quant_arr[i][:horizon_hours]]
                rec = {
                    "coin": symbol,
                    "cutpoint_ts": cp_str,
                    "entry_price": round(float(batch_inputs[i][-1]), 8),
                    "points": [round(float(x), 8) for x in points_arr[i][:horizon_hours]],
                    "quantiles": quantiles,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                existing_keys.add((symbol, cp_str))
            f.flush()
            processed += len(batch_symbols)

            if (cp_i + 1) % progress_every == 0 or cp_i == len(cutpoints) - 1:
                print(f"  context={context_hours}: {cp_i + 1}/{len(cutpoints)} kesim noktasi "
                      f"({processed} yeni tahmin uretildi)...")

    return load_cache(context_hours)
