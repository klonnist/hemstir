"""docs/history/<COIN>.jsonl'daki GERCEK gecmis tahminleri, OKX'ten cekilen
GERCEK gerceklesen fiyatlarla kiyaslar (yon dogrulugu, MAE/MAPE, guven araligi
isabeti) ve arsivdeki sinyalleri (varsa - eski/backfill kayitlarda olmayabilir)
gercek gelecek mumlarla simule eder. backtest.py'den farki: burada TimesFM
YENIDEN calistirilmiyor, arsive o an kaydedilmis tahmin/sinyal aynen kullaniliyor.

Cikti: docs/evaluation.json - backtest.json ile ayni sekle sahip (panel ikisini
ayni bilesenle gosterebilsin diye), ama "source": "archive" etiketiyle.

Kullanim (periyodik ya da elle calistirilabilir):
    python scripts/evaluate_archive.py
"""
import json
import os
import random
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_lib import (  # noqa: E402
    COMPARISON_STRATEGIES,
    aggregate_trades,
    direction_correct,
    forecast_error,
    in_confidence_band,
    price_at_horizon,
    simulate_trade,
)
from generate_forecasts import ARCHIVE_CHECKPOINT_HOURS, BAR, COINS, HISTORY_DIR, compute_tp_sl  # noqa: E402
from okx_client import fetch_history_candles  # noqa: E402

LEVERAGE = float(os.environ.get("BACKTEST_LEVERAGE", "1"))
FEE_PCT = float(os.environ.get("BACKTEST_FEE_PCT", "0.05"))
RANDOM_SEED = int(os.environ.get("BACKTEST_RANDOM_SEED", "42"))
OUTPUT_FILE = os.environ.get("EVALUATION_OUTPUT_FILE", os.path.join("docs", "evaluation.json"))
STRATEGIES = {"model": None, **COMPARISON_STRATEGIES}
MAX_HORIZON = max(ARCHIVE_CHECKPOINT_HOURS)


def load_records(symbol: str) -> list:
    path = os.path.join(HISTORY_DIR, f"{symbol}.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_ts(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def fetch_price_index(inst_id: str, start: datetime, end: datetime) -> dict:
    """saat basina yuvarlanmis ts -> {high, low, close}."""
    df = fetch_history_candles(inst_id, BAR, start, end)
    idx = {}
    for row in df.itertuples():
        key = row.timestamp.to_pydatetime().replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        idx[key] = {"high": float(row.high), "low": float(row.low), "close": float(row.close)}
    return idx


def future_candles_for(price_index: dict, entry_ts: datetime, horizon: int) -> list:
    out = []
    for h in range(1, horizon + 1):
        ts = entry_ts + timedelta(hours=h)
        row = price_index.get(ts)
        if row is None:
            break  # henuz gerceklesmemis / veri yok - burada dur (sonraki saatler de yok)
        out.append({"ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ"), **row})
    return out


def main() -> int:
    now = datetime.now(timezone.utc)
    rng = random.Random(RANDOM_SEED)

    coins_out = {}
    for symbol, inst_id in COINS.items():
        records = load_records(symbol)
        if not records:
            continue
        records.sort(key=lambda r: r["generated_at"])
        earliest_entry = parse_ts(records[0]["entry_ts"])

        try:
            price_index = fetch_price_index(inst_id, earliest_entry, now)
        except Exception as e:
            print(f"UYARI: {symbol} icin fiyat verisi alinamadi, atlaniyor: {e}", file=sys.stderr)
            continue

        evaluation = {"direction": {str(h): [] for h in ARCHIVE_CHECKPOINT_HOURS},
                      "abs_err": {str(h): [] for h in ARCHIVE_CHECKPOINT_HOURS},
                      "pct_err": {str(h): [] for h in ARCHIVE_CHECKPOINT_HOURS},
                      "in_band": {str(h): [] for h in ARCHIVE_CHECKPOINT_HOURS}}
        trades = {name: {"independent": [], "single_position": []} for name in STRATEGIES}
        next_allowed = {name: None for name in STRATEGIES}
        predictions = []  # panelde gecmis tahmin cizgilerini ustuste cizmek icin

        for rec in records:
            entry_ts = parse_ts(rec["entry_ts"])
            entry_price = rec["entry_price"]
            checkpoints = rec.get("checkpoints") or {}

            final_cp = checkpoints.get(str(MAX_HORIZON))
            final_actual_row = price_index.get(entry_ts + timedelta(hours=MAX_HORIZON))
            final_actual = final_actual_row["close"] if final_actual_row else None
            if final_cp:
                final_side = "BUY" if final_cp["value"] >= entry_price else "SELL"
                predictions.append({**rec, "direction_correct": direction_correct(final_side, entry_price, final_actual)})

            # ---- Tahmin dogrulugu (yon/MAE/MAPE/guven araligi) ----
            for h in ARCHIVE_CHECKPOINT_HOURS:
                cp = checkpoints.get(str(h))
                if not cp:
                    continue
                actual_row = price_index.get(entry_ts + timedelta(hours=h))
                actual = actual_row["close"] if actual_row else None
                predicted_side = "BUY" if cp["value"] >= entry_price else "SELL"
                d = direction_correct(predicted_side, entry_price, actual)
                abs_err, pct_err = forecast_error(cp["value"], actual)
                band_ok = in_confidence_band(actual, cp.get("lower"), cp.get("upper"))
                key = str(h)
                if d is not None:
                    evaluation["direction"][key].append(d)
                if abs_err is not None:
                    evaluation["abs_err"][key].append(abs_err)
                    evaluation["pct_err"][key].append(pct_err)
                if band_ok is not None:
                    evaluation["in_band"][key].append(band_ok)

            # ---- Islem simulasyonu (sadece gercek sinyali + ATR'si olan kayitlar) ----
            atr_value = rec.get("atr")
            if atr_value is None:
                continue
            future = future_candles_for(price_index, entry_ts, MAX_HORIZON)
            if not future:
                continue

            for strat_name, strat_fn in STRATEGIES.items():
                if strat_name == "model":
                    side = rec.get("side")
                    if side is None:
                        continue
                else:
                    side = strat_fn(context_closes=None, rng=rng)  # arsivde baglam serisi yok -> momentum de rastgele davranir
                sl, tp = compute_tp_sl(entry_price, atr_value, side)
                result = simulate_trade(
                    entry_price, side, tp, sl, future,
                    leverage=LEVERAGE, fee_pct=FEE_PCT, horizon_hours=MAX_HORIZON,
                )
                trade_record = {
                    "entry_ts": rec["entry_ts"],
                    "entry_price": round(entry_price, 8),
                    "side": side,
                    "strength": rec.get("strength") if strat_name == "model" else None,
                    "exit_reason": result.exit_reason,
                    "exit_ts": result.exit_ts,
                    "exit_price": round(result.exit_price, 8) if result.exit_price is not None else None,
                    "hours_held": result.hours_held,
                    "pnl_pct": result.pnl_pct,
                    "direction_correct": direction_correct(side, entry_price, future[-1]["close"]),
                }
                trades[strat_name]["independent"].append(trade_record)
                if next_allowed[strat_name] is None or entry_ts >= next_allowed[strat_name]:
                    trades[strat_name]["single_position"].append(trade_record)
                    held = result.hours_held or MAX_HORIZON
                    next_allowed[strat_name] = entry_ts + timedelta(hours=held)

        strategies_out = {}
        for strat_name in STRATEGIES:
            ind = trades[strat_name]["independent"]
            single = trades[strat_name]["single_position"]
            entry = {
                "independent": {**aggregate_trades(ind), "trades": ind},
                "single_position": {**aggregate_trades(single), "trades": single},
            }
            if strat_name == "model":
                for label, src in (("independent", ind), ("single_position", single)):
                    entry[label]["by_strength"] = {
                        s: aggregate_trades([t for t in src if t["strength"] == s])
                        for s in ("guclu", "orta", "zayif")
                    }
                    entry[label]["by_side"] = {
                        s: aggregate_trades([t for t in src if t["side"] == s])
                        for s in ("BUY", "SELL")
                    }
            strategies_out[strat_name] = entry

        direction_accuracy = {
            h: (round(100 * sum(v) / len(v), 2) if v else None) for h, v in evaluation["direction"].items()
        }
        mae = {h: (round(float(np.mean(v)), 6) if v else None) for h, v in evaluation["abs_err"].items()}
        mape = {h: (round(float(np.mean(v)), 3) if v else None) for h, v in evaluation["pct_err"].items()}
        in_band_pct = {
            h: (round(100 * sum(v) / len(v), 2) if v else None) for h, v in evaluation["in_band"].items()
        }

        coins_out[symbol] = {
            "inst_id": inst_id,
            "record_count": len(records),
            "sample_size": len(trades["model"]["independent"]),
            "direction_accuracy_pct": direction_accuracy,
            "in_confidence_band_pct": in_band_pct,
            "mae": mae,
            "mape_pct": mape,
            "predictions": predictions,
            "strategies": strategies_out,
        }
        print(f"{symbol}: {len(records)} arsiv kaydi, {len(trades['model']['independent'])} simule edilebilir islem.")

    payload = {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "archive",
        "leverage": LEVERAGE,
        "fee_pct": FEE_PCT,
        "strategies": list(STRATEGIES),
        "coins": coins_out,
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Arsiv degerlendirmesi tamamlandi -> {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
