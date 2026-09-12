"""OKX'ten 10 kripto icin saatlik kapanis verisi ceker, Google'in TimesFM modeliyle
zero-shot 24-48 saatlik fiyat tahmini uretir ve sonucu docs/forecasts.json'a yazar.

Akis:
  1. Her coin icin OKX'in genel /market/candles ucundan (auth gerekmez) son
     CONTEXT_HOURS saatlik kapanis mumunu cek.
  2. Tum coinlerin kapanis serilerini TimesFM'e TEK bir batch cagrisinda ver
     (yeniden egitim yok, dogrudan zero-shot inference).
  3. Gercek gecmis veri + tahmin + (varsa) guven araligini tek bir JSON'a yaz.

TimesFM agirliklari script her calistiginda Hugging Face'ten indirilir; onceden
hicbir yerde barindirilmaz (bkz. README - lisans notlari da orada).
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

from okx_client import fetch_recent_candles

COINS = {
    "BTC": "BTC-USDT",
    "ETH": "ETH-USDT",
    "SOL": "SOL-USDT",
    "XRP": "XRP-USDT",
    "ADA": "ADA-USDT",
    "AVAX": "AVAX-USDT",
    "DOGE": "DOGE-USDT",
    "DOT": "DOT-USDT",
    "LINK": "LINK-USDT",
    "LTC": "LTC-USDT",
}

BAR = "1H"
# OKX /market/candles tek istekte en fazla 300 mum doner (~12.5 gun); daha uzun
# baglam icin /market/history-candles ile sayfalama gerekir, MVP kapsaminda tutuldu.
CONTEXT_HOURS = int(os.environ.get("CONTEXT_HOURS", "300"))
HORIZON_HOURS = int(os.environ.get("HORIZON_HOURS", "48"))

# TimesFM 2.5 (Apache-2.0) checkpoint'i - bkz. README'deki lisans notu.
CHECKPOINT_REPO = os.environ.get("TIMESFM_CHECKPOINT", "google/timesfm-2.5-200m-pytorch")

OUTPUT_FILE = os.environ.get("OUTPUT_FILE", os.path.join("docs", "forecasts.json"))


def load_model(batch_size: int):
    """TimesFM 2.5 modelini yukler ve verilen batch boyutu icin derler.

    torch_compile=False: CI'da her calistirmada sifirdan derleme (torch.compile)
    yapmak yerine dogrudan eager modda calistirir - inference suresi bu olcekte
    (10 kisa seri, CPU) onemsiz, CI guvenilirligi daha degerli."""
    import timesfm

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(CHECKPOINT_REPO, torch_compile=False)
    model.compile(
        timesfm.ForecastConfig(
            max_context=CONTEXT_HOURS,
            max_horizon=HORIZON_HOURS,
            per_core_batch_size=batch_size,  # tum coinler tek batch'te islensin
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    return model


def run_forecast(model, inputs: list) -> tuple:
    """Tum coinlerin kapanis serilerini tek batch cagrisinda tahmin eder.

    Doner: (point_forecast, quantile_forecast), sekil (n_seri, horizon) ve
    (n_seri, horizon, 10) - kantil kolonlari [0.1, 0.2, ..., 0.9] kapsar."""
    return model.forecast(horizon=HORIZON_HOURS, inputs=inputs)


def confidence_band(quantile_row) -> tuple:
    """Bir zaman adimina ait kantil satirindan (genelde [ortalama, q10..q90]) en
    genis alt/ust sinir cikarir. Kolon sirasi timesfm surumune gore degisebileceginden,
    ilk kolonu (ortalama) haric en kucuk/en buyuk degeri alinir."""
    if quantile_row is None or len(quantile_row) < 2:
        return None, None
    tail = np.asarray(quantile_row)[1:]
    return float(np.min(tail)), float(np.max(tail))


def build_series(symbol: str, inst_id: str) -> tuple:
    df = fetch_recent_candles(inst_id, BAR, limit=CONTEXT_HOURS)
    if df.empty:
        raise RuntimeError(f"{symbol} ({inst_id}) icin OKX'ten veri alinamadi.")
    history = [
        {"ts": row.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"), "close": round(float(row.close), 8)}
        for row in df.itertuples()
    ]
    return history, df["close"].to_numpy(dtype=np.float64)


def main() -> int:
    histories = {}
    series_by_symbol = {}
    for symbol, inst_id in COINS.items():
        try:
            history, closes = build_series(symbol, inst_id)
        except Exception as e:
            print(f"UYARI: {symbol} icin veri alinamadi, atlaniyor: {e}", file=sys.stderr)
            continue
        histories[symbol] = history
        series_by_symbol[symbol] = closes

    if not series_by_symbol:
        print("HATA: Hicbir coin icin veri alinamadi.", file=sys.stderr)
        return 1

    ordered_symbols = list(series_by_symbol.keys())
    inputs = [series_by_symbol[s] for s in ordered_symbols]

    print("TimesFM yukleniyor (ilk calistirmada agirliklar Hugging Face'ten indirilir)...")
    model = load_model(batch_size=len(inputs))
    point_forecast, quantile_forecast = run_forecast(model, inputs)

    coins_payload = {}
    for i, symbol in enumerate(ordered_symbols):
        last_ts = datetime.strptime(histories[symbol][-1]["ts"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        points = point_forecast[i]
        quantiles = quantile_forecast[i] if quantile_forecast is not None else None

        forecast_points = []
        for h in range(len(points)):
            ts = last_ts + timedelta(hours=h + 1)
            lower, upper = confidence_band(quantiles[h] if quantiles is not None else None)
            forecast_points.append({
                "ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "value": round(float(points[h]), 8),
                "lower": round(lower, 8) if lower is not None else None,
                "upper": round(upper, 8) if upper is not None else None,
            })

        coins_payload[symbol] = {
            "inst_id": COINS[symbol],
            "history": histories[symbol],
            "forecast": forecast_points,
        }

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": "Google TimesFM 2.5 (200M, zero-shot, yeniden egitim yok)",
        "checkpoint": CHECKPOINT_REPO,
        "bar": BAR,
        "context_hours": CONTEXT_HOURS,
        "horizon_hours": HORIZON_HOURS,
        "coins": coins_payload,
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"{len(coins_payload)} coin icin tahmin uretildi -> {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
