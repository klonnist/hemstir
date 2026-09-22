"""OKX'ten 14 kripto icin saatlik OHLCV verisi ceker, Google'in TimesFM modeliyle
zero-shot 24-48 saatlik fiyat tahmini uretir, ATR bazli bir yon (BUY/SELL) + TP/SL
seviyesi hesaplar ve sonucu docs/forecasts.json'a yazar.

Akis:
  1. Her coin icin OKX'in genel /market/candles ucundan (auth gerekmez) son
     CONTEXT_HOURS saatlik mumu cek.
  2. Tum coinlerin kapanis serilerini TimesFM'e TEK bir batch cagrisinda ver
     (yeniden egitim yok, dogrudan zero-shot inference).
  3. TimesFM'in ufuk sonundaki tahmini mevcut fiyatla kiyaslayarak yon belirle;
     TP/SL'i (crypto-trader projesindeki gibi) ATR'nin sabit katlariyla hesapla -
     TimesFM'in guven araligi risk yonetimi icin tasarlanmadigindan TP/SL icin
     kullanilmiyor, sadece grafikte gosteriliyor (bkz. README).
  4. Gercek gecmis veri + tahmin + sinyali tek bir JSON'a yaz.

TimesFM agirliklari script her calistiginda Hugging Face'ten indirilir; onceden
hicbir yerde barindirilmaz (bkz. README - lisans notlari da orada).
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

from indicators import atr as compute_atr
from okx_client import fetch_recent_candles

# Hepsi perp/vadeli (-USDT-SWAP) - kullanici kaldiracli islem actigi icin.
COINS = {
    "BTC": "BTC-USDT-SWAP",
    "ETH": "ETH-USDT-SWAP",
    "SOL": "SOL-USDT-SWAP",
    "XRP": "XRP-USDT-SWAP",
    "ADA": "ADA-USDT-SWAP",
    "AVAX": "AVAX-USDT-SWAP",
    "DOGE": "DOGE-USDT-SWAP",
    "DOT": "DOT-USDT-SWAP",
    "LINK": "LINK-USDT-SWAP",
    "LTC": "LTC-USDT-SWAP",
    "ETHFI": "ETHFI-USDT-SWAP",
    "CRV": "CRV-USDT-SWAP",
    "NEAR": "NEAR-USDT-SWAP",
    "BNB": "BNB-USDT-SWAP",
}

BAR = "1H"
# OKX /market/candles tek istekte en fazla 300 mum doner (~12.5 gun); daha uzun
# baglam icin /market/history-candles ile sayfalama gerekir, MVP kapsaminda tutuldu.
CONTEXT_HOURS = int(os.environ.get("CONTEXT_HOURS", "300"))
HORIZON_HOURS = int(os.environ.get("HORIZON_HOURS", "48"))

# TimesFM 2.5 (Apache-2.0) checkpoint'i - bkz. README'deki lisans notu.
CHECKPOINT_REPO = os.environ.get("TIMESFM_CHECKPOINT", "google/timesfm-2.5-200m-pytorch")

OUTPUT_FILE = os.environ.get("OUTPUT_FILE", os.path.join("docs", "forecasts.json"))

# ATR bazli TP/SL - crypto-trader projesindeki (okx_trader/strategy.py) katsayilarla
# birebir ayni: SL = 1.5x ATR, TP = 2.5x ATR (~1:1.67 risk/odul).
ATR_PERIOD = 14
SL_ATR_MULT = 1.5
TP_ATR_MULT = 2.5
# Yon (BUY/SELL) her zaman tahmin edilen hareketin yonune gore atanir; bu esik
# sadece "guclu/orta/zayif" etiketini belirler, sinyali gizlemez.
SIGNAL_ATR_THRESHOLDS = (1.0, 2.0)  # zayif < 1.0 ATR <= orta < 2.0 ATR <= guclu

# Tahmin arsivi: her calistirmada docs/history/<COIN>.jsonl'a bir kayit eklenir
# (bkz. build_archive_record/append_archive_record). 48 tahmin noktasinin tamami
# yerine sadece bu saatlerdeki degerler saklanir (dosya kontrolsuz buyumesin diye);
# 6/12/24/48 degerlendirme ufuklarinin hepsini kapsar, 36 grafik icin ara nokta.
ARCHIVE_CHECKPOINT_HOURS = [6, 12, 24, 36, 48]
HISTORY_DIR = os.environ.get("HISTORY_DIR", os.path.join("docs", "history"))
# coin basina ~800 kayit (6 saatte bir calisirsa ~200 gun) - daha eskisi budanir.
ARCHIVE_MAX_RECORDS = int(os.environ.get("ARCHIVE_MAX_RECORDS", "800"))
WRITE_ARCHIVE = os.environ.get("WRITE_ARCHIVE", "1") != "0"


def compute_tp_sl(entry_price: float, atr_value: float, side: str) -> tuple:
    """(stop_loss, take_profit) doner - crypto-trader/okx_trader/strategy.py ile ayni mantik."""
    if side == "BUY":
        return entry_price - SL_ATR_MULT * atr_value, entry_price + TP_ATR_MULT * atr_value
    return entry_price + SL_ATR_MULT * atr_value, entry_price - TP_ATR_MULT * atr_value


def signal_strength(expected_move: float, atr_value: float) -> str:
    ratio = abs(expected_move) / atr_value if atr_value else 0.0
    weak, medium = SIGNAL_ATR_THRESHOLDS
    if ratio >= medium:
        return "guclu"
    if ratio >= weak:
        return "orta"
    return "zayif"


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


def build_series_from_df(df) -> tuple:
    """Saf fonksiyon (ag erisimi yok) - canli akis ve backtest'in ikisi de kullanir."""
    if df.empty:
        raise RuntimeError("Bos DataFrame - OKX'ten veri alinamadi.")
    history = [
        {"ts": row.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"), "close": round(float(row.close), 8)}
        for row in df.itertuples()
    ]
    atr_series = compute_atr(df, period=ATR_PERIOD)
    atr_value = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else None
    return history, df["close"].to_numpy(dtype=np.float64), atr_value


def build_series(symbol: str, inst_id: str) -> tuple:
    df = fetch_recent_candles(inst_id, BAR, limit=CONTEXT_HOURS)
    if df.empty:
        raise RuntimeError(f"{symbol} ({inst_id}) icin OKX'ten veri alinamadi.")
    return build_series_from_df(df)


def build_forecast_points(last_ts, points, quantiles) -> list:
    """Her saat icin {ts, value, lower, upper} - canli akis ve backtest'in ikisi de kullanir."""
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
    return forecast_points


def build_signal(entry_price: float, atr_value, forecast_end_value) -> dict:
    """Son tahmin noktasina (ufuk sonu) gore BUY/SELL + ATR bazli TP/SL. Yeterli
    veri yoksa None doner. `forecasts.json`'daki `signal` alaniyla birebir ayni sekil -
    backtest.py da her kesim noktasinda bunu cagirir."""
    if atr_value is None or forecast_end_value is None:
        return None
    expected_move = float(forecast_end_value) - entry_price
    side = "BUY" if expected_move >= 0 else "SELL"
    sl, tp = compute_tp_sl(entry_price, atr_value, side)
    return {
        "side": side,
        "strength": signal_strength(expected_move, atr_value),
        "entry_price": round(entry_price, 8),
        "atr": round(atr_value, 8),
        "expected_move": round(expected_move, 8),
        "expected_move_pct": round(expected_move / entry_price * 100, 3),
        "take_profit": round(tp, 8),
        "stop_loss": round(sl, 8),
        "risk_reward": round(TP_ATR_MULT / SL_ATR_MULT, 3),
    }


def build_archive_record(symbol: str, inst_id: str, generated_at: str, entry_ts: str,
                          entry_price: float, signal, forecast_points: list) -> dict:
    """Tam 48 noktalik tahmin yerine sadece ARCHIVE_CHECKPOINT_HOURS'taki degerleri
    tutan kompakt kayit - docs/history/<COIN>.jsonl'a yazilir."""
    checkpoints = {}
    for h in ARCHIVE_CHECKPOINT_HOURS:
        idx = h - 1
        if 0 <= idx < len(forecast_points):
            fp = forecast_points[idx]
            checkpoints[str(h)] = {
                "ts": fp["ts"], "value": fp["value"], "lower": fp["lower"], "upper": fp["upper"],
            }
    record = {
        "generated_at": generated_at,
        "coin": symbol,
        "inst_id": inst_id,
        "entry_ts": entry_ts,
        "entry_price": round(entry_price, 8),
        "side": signal["side"] if signal else None,
        "strength": signal["strength"] if signal else None,
        "expected_move_pct": signal["expected_move_pct"] if signal else None,
        "take_profit": signal["take_profit"] if signal else None,
        "stop_loss": signal["stop_loss"] if signal else None,
        "atr": signal["atr"] if signal else None,
        "checkpoints": checkpoints,
    }
    return record


def append_archive_record(symbol: str, record: dict, history_dir: str = None, max_records: int = None) -> None:
    history_dir = history_dir or HISTORY_DIR
    max_records = ARCHIVE_MAX_RECORDS if max_records is None else max_records
    os.makedirs(history_dir, exist_ok=True)
    path = os.path.join(history_dir, f"{symbol}.jsonl")
    lines = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
    # Ayni (generated_at, coin) icin tekrar calistirilirsa eski kaydi degistir.
    lines = [ln for ln in lines if json.loads(ln).get("generated_at") != record["generated_at"]]
    lines.append(json.dumps(record, ensure_ascii=False))
    if len(lines) > max_records:
        lines = lines[-max_records:]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    histories = {}
    series_by_symbol = {}
    atr_by_symbol = {}
    for symbol, inst_id in COINS.items():
        try:
            history, closes, atr_value = build_series(symbol, inst_id)
        except Exception as e:
            print(f"UYARI: {symbol} icin veri alinamadi, atlaniyor: {e}", file=sys.stderr)
            continue
        histories[symbol] = history
        series_by_symbol[symbol] = closes
        atr_by_symbol[symbol] = atr_value

    if not series_by_symbol:
        print("HATA: Hicbir coin icin veri alinamadi.", file=sys.stderr)
        return 1

    ordered_symbols = list(series_by_symbol.keys())
    inputs = [series_by_symbol[s] for s in ordered_symbols]

    print("TimesFM yukleniyor (ilk calistirmada agirliklar Hugging Face'ten indirilir)...")
    model = load_model(batch_size=len(inputs))
    point_forecast, quantile_forecast = run_forecast(model, inputs)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    coins_payload = {}
    for i, symbol in enumerate(ordered_symbols):
        last_ts = datetime.strptime(histories[symbol][-1]["ts"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        points = point_forecast[i]
        quantiles = quantile_forecast[i] if quantile_forecast is not None else None

        forecast_points = build_forecast_points(last_ts, points, quantiles)
        entry_price = float(series_by_symbol[symbol][-1])
        atr_value = atr_by_symbol[symbol]
        signal = build_signal(entry_price, atr_value, points[-1] if len(points) else None)

        coins_payload[symbol] = {
            "inst_id": COINS[symbol],
            "history": histories[symbol],
            "forecast": forecast_points,
            "signal": signal,
        }

        if WRITE_ARCHIVE:
            record = build_archive_record(
                symbol, COINS[symbol], generated_at,
                histories[symbol][-1]["ts"], entry_price, signal, forecast_points,
            )
            append_archive_record(symbol, record)

    payload = {
        "generated_at": generated_at,
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
