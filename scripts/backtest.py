"""Walk-forward backtest: gecmiste, canli sistemle AYNI siklikta (varsayilan her
6 saatte bir) bir kesim noktasi secip, o ana kadarki (ve SADECE o ana kadarki)
veriyle TimesFM'i calistirir, generate_forecasts.py'deki AYNI sinyal/TP-SL
mantigini uygular, sonra gercek gelecek mumlarla (ayni fetch edilen veri
icinden) sonucu degerlendirir ve islemi simule eder.

Look-ahead bias olmamasi icin: her kesim noktasinda model ve ATR SADECE o ana
kadarki (CONTEXT_HOURS kadar) veriyi gorur; gelecekteki mumlar sadece
degerlendirme/simulasyon asamasinda, sinyal uretildikten SONRA kullanilir.

Cikti: docs/backtest.json (coin basina + strateji basina toplulastirilmis
metrikler ve karsilastirma stratejileri).

Ortam degiskenleri: BACKTEST_DAYS (varsayilan 75), BACKTEST_STEP_HOURS (6),
BACKTEST_LEVERAGE (1), BACKTEST_FEE_PCT (0.05), BACKTEST_MAX_CUTPOINTS
(0 = sinirsiz), BACKTEST_COINS (virgullu alt kume, bos = COINS'in tumu),
BACKTEST_RANDOM_SEED (42), BACKTEST_OUTPUT_FILE (docs/backtest.json).
"""
import json
import os
import random
import sys
import time
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
from generate_forecasts import (  # noqa: E402
    ARCHIVE_CHECKPOINT_HOURS,
    ATR_PERIOD,
    BAR,
    COINS,
    CONTEXT_HOURS,
    HORIZON_HOURS,
    build_archive_record,
    build_forecast_points,
    build_series_from_df,
    build_signal,
    compute_tp_sl,
    load_model,
    run_forecast,
    signal_strength,
)
from okx_client import fetch_history_candles

BACKTEST_DAYS = int(os.environ.get("BACKTEST_DAYS", "75"))
STEP_HOURS = int(os.environ.get("BACKTEST_STEP_HOURS", "6"))
LEVERAGE = float(os.environ.get("BACKTEST_LEVERAGE", "1"))
FEE_PCT = float(os.environ.get("BACKTEST_FEE_PCT", "0.05"))
MAX_CUTPOINTS = int(os.environ.get("BACKTEST_MAX_CUTPOINTS", "0")) or None
_coins_env = os.environ.get("BACKTEST_COINS", "").strip()
BACKTEST_COINS = [c.strip().upper() for c in _coins_env.split(",") if c.strip()] if _coins_env else list(COINS)
RANDOM_SEED = int(os.environ.get("BACKTEST_RANDOM_SEED", "42"))
OUTPUT_FILE = os.environ.get("BACKTEST_OUTPUT_FILE", os.path.join("docs", "backtest.json"))

STRATEGIES = {"model": None, **COMPARISON_STRATEGIES}  # "model" = TimesFM'in kendi tahmini


def build_cutpoints(now: datetime) -> list:
    """Canli sistemle ayni STEP_HOURS araliginda, hepsinin HORIZON_HOURS kadar
    gercek gelecek verisi olacagi kesim noktalari (eskiden yeniye)."""
    last_possible = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=HORIZON_HOURS)
    earliest = now - timedelta(days=BACKTEST_DAYS)
    points = []
    t = last_possible
    while t >= earliest:
        points.append(t)
        t -= timedelta(hours=STEP_HOURS)
    points.reverse()
    if MAX_CUTPOINTS:
        points = points[-MAX_CUTPOINTS:]
    return points


def fetch_coin_history(inst_id: str, earliest: datetime, now: datetime):
    start = earliest - timedelta(hours=CONTEXT_HOURS + 5)  # biraz pay birak
    df = fetch_history_candles(inst_id, BAR, start, now)
    return df


def df_index_by_hour(df):
    """timestamp (saat basina yuvarlanmis) -> satir pozisyonu."""
    idx = {}
    for i, ts in enumerate(df["timestamp"]):
        idx[ts.to_pydatetime().replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)] = i
    return idx


def rows_to_candles(df, start_idx: int, count: int) -> list:
    sub = df.iloc[start_idx:start_idx + count]
    return [
        {"ts": row.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"), "high": float(row.high),
         "low": float(row.low), "close": float(row.close)}
        for row in sub.itertuples()
    ]


def main() -> int:
    now = datetime.now(timezone.utc)
    cutpoints = build_cutpoints(now)
    if not cutpoints:
        print("HATA: gecerli kesim noktasi yok (BACKTEST_DAYS/BACKTEST_STEP_HOURS'u kontrol edin).",
              file=sys.stderr)
        return 1
    print(f"{len(cutpoints)} kesim noktasi, {len(BACKTEST_COINS)} coin icin backtest baslatiliyor "
          f"({cutpoints[0].isoformat()} -> {cutpoints[-1].isoformat()}).")

    earliest = cutpoints[0]
    coin_data = {}
    for symbol in BACKTEST_COINS:
        inst_id = COINS[symbol]
        try:
            df = fetch_coin_history(inst_id, earliest, now)
        except Exception as e:
            print(f"UYARI: {symbol} icin gecmis veri alinamadi, atlaniyor: {e}", file=sys.stderr)
            continue
        if df.empty:
            print(f"UYARI: {symbol} icin bos veri, atlaniyor.", file=sys.stderr)
            continue
        coin_data[symbol] = {"df": df, "idx": df_index_by_hour(df)}
    if not coin_data:
        print("HATA: hicbir coin icin gecmis veri alinamadi.", file=sys.stderr)
        return 1

    print("TimesFM yukleniyor (bir kere, tum kesim noktalarinda yeniden kullanilir)...")
    model = load_model(batch_size=len(BACKTEST_COINS))

    rng = random.Random(RANDOM_SEED)
    # trades[strategy][symbol] = {"independent": [...], "single_position": [...]}
    trades = {name: {s: {"independent": [], "single_position": []} for s in coin_data} for name in STRATEGIES}
    next_allowed = {}  # (strategy, symbol) -> bu zamandan once yeni islem alma (tek pozisyon modu)
    # predictions[symbol]: panelde gecmis tahmin cizgilerini ustuste cizmek icin
    # (sadece TimesFM'in kendi tahmini - karsilastirma stratejilerinin ayri bir
    # tahmin egrisi yok, sadece farkli yon secimi).
    predictions = {s: [] for s in coin_data}
    evaluation = {s: {"direction": {str(h): [] for h in ARCHIVE_CHECKPOINT_HOURS},
                       "abs_err": {str(h): [] for h in ARCHIVE_CHECKPOINT_HOURS},
                       "pct_err": {str(h): [] for h in ARCHIVE_CHECKPOINT_HOURS},
                       "in_band": {str(h): [] for h in ARCHIVE_CHECKPOINT_HOURS}}
                   for s in coin_data}

    t0 = time.time()
    for cp_i, cutpoint in enumerate(cutpoints):
        batch_symbols, batch_inputs, batch_ctx = [], [], {}
        for symbol, entry in coin_data.items():
            row_idx = entry["idx"].get(cutpoint)
            if row_idx is None or row_idx + 1 < CONTEXT_HOURS:
                continue
            df = entry["df"]
            context_df = df.iloc[row_idx - CONTEXT_HOURS + 1: row_idx + 1]
            if len(context_df) < CONTEXT_HOURS:
                continue
            future = rows_to_candles(df, row_idx + 1, HORIZON_HOURS)
            if len(future) < HORIZON_HOURS:
                continue  # bu kesim noktasinda tam ufuk kadar gercek veri yok, atla

            history, closes, atr_value = build_series_from_df(context_df)
            batch_symbols.append(symbol)
            batch_inputs.append(closes)
            batch_ctx[symbol] = {
                "entry_price": float(closes[-1]), "atr": atr_value, "future": future,
                "context_closes": closes,
            }

        if not batch_symbols:
            continue

        point_forecast, quantile_forecast = run_forecast(model, batch_inputs)

        for i, symbol in enumerate(batch_symbols):
            ctx = batch_ctx[symbol]
            entry_price, atr_value, future = ctx["entry_price"], ctx["atr"], ctx["future"]
            points = point_forecast[i]
            quantiles = quantile_forecast[i] if quantile_forecast is not None else None
            forecast_points = build_forecast_points(cutpoint, points, quantiles)
            forecast_end = forecast_points[-1]["value"] if forecast_points else None
            model_side = "BUY" if (forecast_end is not None and forecast_end >= entry_price) else "SELL"

            for h in ARCHIVE_CHECKPOINT_HOURS:
                idx = h - 1
                if idx >= len(forecast_points):
                    continue
                key = str(h)
                fp = forecast_points[idx]
                actual = price_at_horizon(future, h)
                abs_err, pct_err = forecast_error(fp["value"], actual)
                d = direction_correct(model_side, entry_price, actual)
                band_ok = in_confidence_band(actual, fp["lower"], fp["upper"])
                if d is not None:
                    evaluation[symbol]["direction"][key].append(d)
                if abs_err is not None:
                    evaluation[symbol]["abs_err"][key].append(abs_err)
                    evaluation[symbol]["pct_err"][key].append(pct_err)
                if band_ok is not None:
                    evaluation[symbol]["in_band"][key].append(band_ok)

            signal = build_signal(entry_price, atr_value, forecast_end)
            entry_ts = cutpoint.strftime("%Y-%m-%dT%H:%M:%SZ")

            final_actual = price_at_horizon(future, HORIZON_HOURS)
            record = build_archive_record(
                symbol, COINS[symbol], entry_ts, entry_ts, entry_price, signal, forecast_points,
            )
            record["direction_correct"] = direction_correct(model_side, entry_price, final_actual)
            predictions[symbol].append(record)

            for strat_name, strat_fn in STRATEGIES.items():
                if strat_name == "model":
                    if signal is None:
                        continue
                    side = signal["side"]
                else:
                    side = strat_fn(context_closes=ctx["context_closes"], rng=rng)
                if atr_value is None:
                    continue
                sl, tp = compute_tp_sl(entry_price, atr_value, side)
                result = simulate_trade(
                    entry_price, side, tp, sl, future,
                    leverage=LEVERAGE, fee_pct=FEE_PCT, horizon_hours=HORIZON_HOURS,
                )
                trade_record = {
                    "entry_ts": entry_ts,
                    "entry_price": round(entry_price, 8),
                    "side": side,
                    "strength": signal["strength"] if strat_name == "model" and signal else None,
                    "exit_reason": result.exit_reason,
                    "exit_ts": result.exit_ts,
                    "exit_price": round(result.exit_price, 8) if result.exit_price is not None else None,
                    "hours_held": result.hours_held,
                    "pnl_pct": result.pnl_pct,
                    "direction_correct": direction_correct(side, entry_price, future[-1]["close"]),
                }
                trades[strat_name][symbol]["independent"].append(trade_record)

                key = (strat_name, symbol)
                if next_allowed.get(key) is None or cutpoint >= next_allowed[key]:
                    trades[strat_name][symbol]["single_position"].append(trade_record)
                    held = result.hours_held or HORIZON_HOURS
                    next_allowed[key] = cutpoint + timedelta(hours=held)

        if (cp_i + 1) % 20 == 0 or cp_i == len(cutpoints) - 1:
            elapsed = time.time() - t0
            print(f"  {cp_i + 1}/{len(cutpoints)} kesim noktasi islendi ({elapsed:.0f}s)...")

    # ---- Toplulastirma ----
    coins_out = {}
    for symbol in coin_data:
        strategies_out = {}
        for strat_name in STRATEGIES:
            ind = trades[strat_name][symbol]["independent"]
            single = trades[strat_name][symbol]["single_position"]
            entry = {
                "independent": {**aggregate_trades(ind), "trades": ind},
                "single_position": {**aggregate_trades(single), "trades": single},
            }
            if strat_name == "model":
                for label in ("independent", "single_position"):
                    src = ind if label == "independent" else single
                    entry[label]["by_strength"] = {
                        s: aggregate_trades([t for t in src if t["strength"] == s])
                        for s in ("guclu", "orta", "zayif")
                    }
                    entry[label]["by_side"] = {
                        s: aggregate_trades([t for t in src if t["side"] == s])
                        for s in ("BUY", "SELL")
                    }
            strategies_out[strat_name] = entry

        ev = evaluation[symbol]
        direction_accuracy = {
            h: (round(100 * sum(v) / len(v), 2) if v else None) for h, v in ev["direction"].items()
        }
        mae = {h: (round(float(np.mean(v)), 6) if v else None) for h, v in ev["abs_err"].items()}
        mape = {h: (round(float(np.mean(v)), 3) if v else None) for h, v in ev["pct_err"].items()}
        in_band_pct = {
            h: (round(100 * sum(v) / len(v), 2) if v else None) for h, v in ev["in_band"].items()
        }

        coins_out[symbol] = {
            "inst_id": COINS[symbol],
            "sample_size": len(trades["model"][symbol]["independent"]),
            "direction_accuracy_pct": direction_accuracy,
            "in_confidence_band_pct": in_band_pct,
            "mae": mae,
            "mape_pct": mape,
            "predictions": predictions[symbol],
            "strategies": strategies_out,
        }

    payload = {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "backtest",
        "backtest_days": BACKTEST_DAYS,
        "step_hours": STEP_HOURS,
        "context_hours": CONTEXT_HOURS,
        "horizon_hours": HORIZON_HOURS,
        "leverage": LEVERAGE,
        "fee_pct": FEE_PCT,
        "cutpoint_count": len(cutpoints),
        "strategies": list(STRATEGIES),
        "coins": coins_out,
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Backtest tamamlandi -> {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
