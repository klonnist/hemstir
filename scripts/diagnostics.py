"""Teshis raporu: MEVCUT docs/backtest.json'daki (5 major coin) sonuclari, TimesFM'i
YENIDEN CALISTIRMADAN analiz eder. Sadece gercek OKX fiyat verisi cekilir (model
inference YOK) ve su sorulara cevap aranir:

  1. Brut (komisyonsuz) vs net getiri, toplam komisyon yuku.
  2. Cikis zamanlamasi: SL/TP/EXPIRED'in ilk 3/6/12 saatte mi yoksa daha gec mi
     tetiklendigi dagilimi.
  3. "Saf yon" sonucu: TP/SL YOK, aynı sinyallerle 48. saat kapanisinda cikilsaydi
     brut/net getiri ne olurdu (gercek OKX kapanis fiyatlari kullanilarak).
  4. 6/12/24/48 saatlik ufuklarda yon dogrulugu ve "her zaman AL" taban oranindan
     farki (gercek fiyat hareketinden dogrudan hesaplanir).
  5. Guven araligi kalibrasyonu: gercek fiyat, modelin q10-q90 bandinda ufka gore
     ne siklikta kaliyor (beklenen ~%80 ile kiyaslanir).

Girdi: docs/backtest.json (scripts/backtest.py'nin ciktisi - "independent" mod,
tum sinyaller, pozisyon kisitlamasi olmadan). Cikti: docs/diagnostics.json.

Kullanim: python scripts/diagnostics.py
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_forecasts import BAR, COINS  # noqa: E402
from okx_client import fetch_history_candles  # noqa: E402

DIAG_COINS = ["BTC", "ETH", "XRP", "SOL", "AVAX"]
BACKTEST_FILE = os.environ.get("BACKTEST_FILE", os.path.join("docs", "backtest.json"))
OUTPUT_FILE = os.environ.get("DIAGNOSTICS_OUTPUT_FILE", os.path.join("docs", "diagnostics.json"))
CHECKPOINT_HOURS = [6, 12, 24, 36, 48]
TIMING_BUCKETS = [3, 6, 12, 24, 48]  # "ilk N saatte cikti mi" kova sinirlari


def parse_ts(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def fetch_price_index(inst_id: str, start: datetime, end: datetime) -> dict:
    df = fetch_history_candles(inst_id, BAR, start, end)
    idx = {}
    for row in df.itertuples():
        key = row.timestamp.to_pydatetime().replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        idx[key] = float(row.close)
    return idx


def timing_bucket(hours_held) -> str:
    if hours_held is None:
        return "bilinmiyor"
    for b in TIMING_BUCKETS:
        if hours_held <= b:
            return f"<={b}s"
    return f">{TIMING_BUCKETS[-1]}s"


def main() -> int:
    if not os.path.exists(BACKTEST_FILE):
        print(f"HATA: {BACKTEST_FILE} yok - once scripts/backtest.py calistirilmali.", file=sys.stderr)
        return 1
    with open(BACKTEST_FILE, "r", encoding="utf-8") as f:
        backtest = json.load(f)

    leverage = backtest.get("leverage", 1.0)
    fee_pct = backtest.get("fee_pct", 0.05)
    fee_roundtrip_pct = 2 * fee_pct * leverage

    coins_out = {}
    for symbol in DIAG_COINS:
        coin_bt = backtest.get("coins", {}).get(symbol)
        if not coin_bt:
            print(f"UYARI: {symbol} backtest.json'da yok, atlaniyor.", file=sys.stderr)
            continue

        trades = coin_bt["strategies"]["model"]["independent"]["trades"]
        predictions = coin_bt.get("predictions", [])
        if not trades or not predictions:
            print(f"UYARI: {symbol} icin islem/tahmin verisi yok, atlaniyor.", file=sys.stderr)
            continue

        entry_times = [parse_ts(t["entry_ts"]) for t in trades] + [parse_ts(p["entry_ts"]) for p in predictions]
        start = min(entry_times)
        end = max(entry_times) + timedelta(hours=max(CHECKPOINT_HOURS) + 2)
        inst_id = COINS.get(symbol, coin_bt.get("inst_id"))
        try:
            price_index = fetch_price_index(inst_id, start, end)
        except Exception as e:
            print(f"UYARI: {symbol} icin fiyat verisi alinamadi, atlaniyor: {e}", file=sys.stderr)
            continue

        # ---- 1) Brut vs net getiri, komisyon yuku ----
        gross_trades, net_trades = [], []
        for t in trades:
            net = t.get("pnl_pct")
            if net is None:
                continue
            gross = net if t["exit_reason"] == "LIQUIDATION" else net + fee_roundtrip_pct
            gross_trades.append(gross)
            net_trades.append(net)
        total_gross = float(np.sum(gross_trades)) if gross_trades else None
        total_net = float(np.sum(net_trades)) if net_trades else None
        total_fee_cost = (total_gross - total_net) if (total_gross is not None and total_net is not None) else None

        # ---- 2) Cikis zamanlamasi ----
        timing_dist = {}
        for t in trades:
            bucket = timing_bucket(t.get("hours_held"))
            key = f"{t['exit_reason']}|{bucket}"
            timing_dist[key] = timing_dist.get(key, 0) + 1

        # ---- 3) Saf yon: ayni sinyaller, 48. saatte cik (gercek kapanisla) ----
        pure_direction_gross, pure_direction_net = [], []
        for p in predictions:
            side = p.get("side")
            if side is None:
                continue
            entry_ts = parse_ts(p["entry_ts"])
            entry_price = p["entry_price"]
            exit_ts = entry_ts + timedelta(hours=48)
            actual = price_index.get(exit_ts)
            if actual is None:
                continue
            price_move_pct = (actual - entry_price) / entry_price * (1 if side == "BUY" else -1) * 100
            gross = price_move_pct * leverage
            net = gross - fee_roundtrip_pct
            pure_direction_gross.append(gross)
            pure_direction_net.append(net)

        pure_direction = {
            "n": len(pure_direction_gross),
            "total_gross_pct": round(float(np.sum(pure_direction_gross)), 4) if pure_direction_gross else None,
            "total_net_pct": round(float(np.sum(pure_direction_net)), 4) if pure_direction_net else None,
            "win_rate_pct": round(100 * sum(1 for g in pure_direction_net if g > 0) / len(pure_direction_net), 2) if pure_direction_net else None,
        }

        # ---- 4) Ufka gore yon dogrulugu + "her zaman AL" tabani ----
        direction_by_horizon = {}
        for h in CHECKPOINT_HOURS:
            key = str(h)
            model_correct, always_buy_correct, total = 0, 0, 0
            for p in predictions:
                side = p.get("side")
                cp = (p.get("checkpoints") or {}).get(key)
                if side is None or not cp:
                    continue
                entry_ts = parse_ts(p["entry_ts"])
                actual = price_index.get(entry_ts + timedelta(hours=h))
                if actual is None:
                    continue
                total += 1
                up = actual > p["entry_price"]
                if (side == "BUY") == up:
                    model_correct += 1
                if up:
                    always_buy_correct += 1
            if total:
                model_acc = 100 * model_correct / total
                buy_acc = 100 * always_buy_correct / total
                direction_by_horizon[key] = {
                    "n": total,
                    "model_accuracy_pct": round(model_acc, 2),
                    "always_buy_accuracy_pct": round(buy_acc, 2),
                    "diff_pct_points": round(model_acc - buy_acc, 2),
                }
            else:
                direction_by_horizon[key] = {"n": 0, "model_accuracy_pct": None, "always_buy_accuracy_pct": None, "diff_pct_points": None}

        # ---- 5) Guven araligi kalibrasyonu ----
        calibration = {}
        for h in CHECKPOINT_HOURS:
            key = str(h)
            inside, total = 0, 0
            for p in predictions:
                cp = (p.get("checkpoints") or {}).get(key)
                if not cp or cp.get("lower") is None or cp.get("upper") is None:
                    continue
                entry_ts = parse_ts(p["entry_ts"])
                actual = price_index.get(entry_ts + timedelta(hours=h))
                if actual is None:
                    continue
                total += 1
                if cp["lower"] <= actual <= cp["upper"]:
                    inside += 1
            calibration[key] = {
                "n": total,
                "actual_coverage_pct": round(100 * inside / total, 2) if total else None,
                "expected_coverage_pct": 80.0,
                "diff_pct_points": round(100 * inside / total - 80.0, 2) if total else None,
            }

        coins_out[symbol] = {
            "trade_count": len(trades),
            "prediction_count": len(predictions),
            "fee_roundtrip_pct_per_trade": round(fee_roundtrip_pct, 4),
            "returns": {
                "total_gross_pct": round(total_gross, 4) if total_gross is not None else None,
                "total_net_pct": round(total_net, 4) if total_net is not None else None,
                "total_fee_cost_pct": round(total_fee_cost, 4) if total_fee_cost is not None else None,
            },
            "exit_timing_distribution": timing_dist,
            "pure_direction_48h": pure_direction,
            "direction_accuracy_by_horizon": direction_by_horizon,
            "confidence_band_calibration": calibration,
        }
        print(f"{symbol}: {len(trades)} islem, {len(predictions)} tahmin analiz edildi.")

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_backtest_generated_at": backtest.get("generated_at"),
        "coins_analyzed": DIAG_COINS,
        "leverage": leverage,
        "fee_pct": fee_pct,
        "coins": coins_out,
    }
    os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Teshis raporu -> {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
