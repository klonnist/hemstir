"""Arastirma orkestrasyonu: scripts/research_data.py'nin urettigi split.json +
ham veriyi kullanarak GELISTIRME (dev) donemi icin 6 varyanti (+ karsilastirma
stratejilerini, ayni cikis kurallariyla) calistirir, blok bootstrap guven
araliklariyla docs/research_dev.json'a yazar.

KILITLI TEST: --locked bayragiyla calisir. Bir KEZ calistiktan sonra
docs/locked_test_run.json'a kaydedilir; TEKRAR calismasi icin ACIKCA
--force-locked-rerun gerekir (aksi halde "zaten calisti" diyip cikar - veri
sizintisini/asiri uyumu onlemek icin, bkz. README).

Kullanim:
    python scripts/research_data.py          # (once) veri + split
    python scripts/run_research.py           # gelistirme donemi
    python scripts/run_research.py --locked --variants A,C   # kilitli test (bir kez)
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/ (predict_cache, variants)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo kok (eval_lib, generate_forecasts)

from eval_lib import (  # noqa: E402
    aggregate_trades,
    always_buy_side,
    block_bootstrap_ci,
    momentum_side,
    paired_diff_significance,
    random_side,
)
from generate_forecasts import COINS  # noqa: E402
from predict_cache import ensure_predictions, load_cache  # noqa: E402
from variants import VARIANT_CONTEXT, VARIANT_DESCRIPTIONS, VARIANTS  # noqa: E402

CACHE_DIR = os.environ.get("RESEARCH_CACHE_DIR", os.path.join("scripts", "cache"))
LEVERAGE = float(os.environ.get("BACKTEST_LEVERAGE", "1"))
FEE_PCT = float(os.environ.get("BACKTEST_FEE_PCT", "0.05"))
DEFAULT_FUNDING_RATE_PCT = float(os.environ.get("DEFAULT_FUNDING_RATE_PCT", "0.01"))
STEP_HOURS = int(os.environ.get("BACKTEST_STEP_HOURS", "6"))
RANDOM_SEED = int(os.environ.get("BACKTEST_RANDOM_SEED", "42"))
MAX_CUTPOINTS = int(os.environ.get("RESEARCH_MAX_CUTPOINTS", "0")) or None
BLOCK_SIZE = int(os.environ.get("BOOTSTRAP_BLOCK_SIZE", "20"))
N_BOOT = int(os.environ.get("BOOTSTRAP_N", "2000"))

COMPARISON_SIDE_FNS = {"always_buy": always_buy_side, "momentum": momentum_side, "random": random_side}
LOCKED_MARKER = os.path.join("docs", "locked_test_run.json")


def parse(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def fmt(d):
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


def load_split():
    path = os.path.join(CACHE_DIR, "split.json")
    if not os.path.exists(path):
        print(f"HATA: {path} yok - once scripts/research_data.py calistirin.", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_raw_candles():
    coins_raw = {}
    for symbol in COINS:
        path = os.path.join(CACHE_DIR, "raw", f"{symbol}.json")
        if not os.path.exists(path):
            print(f"HATA: {path} yok - once scripts/research_data.py calistirin.", file=sys.stderr)
            sys.exit(1)
        with open(path, encoding="utf-8") as f:
            coins_raw[symbol] = json.load(f)["candles"]
    return coins_raw


def load_funding_lookups():
    lookups = {}
    for symbol in COINS:
        path = os.path.join(CACHE_DIR, "funding", f"{symbol}.json")
        lookup = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for ev in data.get("events", []):
                lookup[parse(ev["ts"])] = ev["rate"]
        lookups[symbol] = lookup
    return lookups


def build_cutpoints(period_start, period_end, min_context_hours, step_hours=STEP_HOURS):
    t = period_start + timedelta(hours=min_context_hours)
    # saat basina hizala
    t = t.replace(minute=0, second=0, microsecond=0)
    points = []
    while t <= period_end:
        points.append(t)
        t += timedelta(hours=step_hours)
    if MAX_CUTPOINTS:
        points = points[-MAX_CUTPOINTS:]
    return points


def context_closes_before(candles, idx_by_ts, cutpoint_ts, hours=48):
    row_idx = idx_by_ts.get(cutpoint_ts)
    if row_idx is None:
        return []
    window = candles[max(0, row_idx - hours + 1): row_idx + 1]
    return [c["close"] for c in window]


def run_period(period_name: str, period_start: datetime, period_end: datetime,
                variant_codes: list, coins_raw: dict, funding_lookups: dict) -> dict:
    idx_by_ts = {symbol: {c["ts"]: i for i, c in enumerate(candles)} for symbol, candles in coins_raw.items()}

    contexts_needed = sorted(set(VARIANT_CONTEXT[v] for v in variant_codes))
    cutpoints_by_context = {}
    for ctx in contexts_needed:
        cutpoints_by_context[ctx] = build_cutpoints(period_start, period_end, ctx)

    pred_caches = {}
    for ctx in contexts_needed:
        cps = cutpoints_by_context[ctx]
        print(f"[{period_name}] context={ctx}: {len(cps)} kesim noktasi hazirlaniyor...")
        pred_caches[ctx] = ensure_predictions(coins_raw, cps, ctx, horizon_hours=48)

    rng = random.Random(RANDOM_SEED)
    results = {}  # variant -> strategy -> list[trade]
    for v in variant_codes:
        ctx = VARIANT_CONTEXT[v]
        base_ctx = 300  # ATR/gelecek mum erisimi hep 1H ham veriden, baglam sadece tahmin icin
        cps = cutpoints_by_context[ctx]
        base_cps = cutpoints_by_context.get(base_ctx, cutpoints_by_context[ctx])
        base_pred_cache = pred_caches.get(base_ctx, pred_caches[ctx])
        long_pred_cache = pred_caches.get(1024)

        fn = VARIANTS[v]
        strategy_trades = {name: [] for name in (["model"] + list(COMPARISON_SIDE_FNS))}

        use_cps = base_cps if v != "F" else cps
        for cutpoint in use_cps:
            cp_str = fmt(cutpoint)
            for symbol in coins_raw:
                pred = base_pred_cache.get((symbol, cp_str))
                pred_long = long_pred_cache.get((symbol, cp_str)) if long_pred_cache else None
                if pred is None and v != "F":
                    continue
                if v == "F" and pred_long is None:
                    continue
                candles = coins_raw[symbol]
                idx = idx_by_ts[symbol]
                funding_lookup = funding_lookups.get(symbol, {})

                trade = fn(pred or pred_long, candles, idx, funding_lookup, LEVERAGE, FEE_PCT,
                           DEFAULT_FUNDING_RATE_PCT, pred_long=pred_long)
                if trade is not None:
                    strategy_trades["model"].append(trade)

                ctx_closes = context_closes_before(candles, idx, cp_str, 48)
                for strat_name, side_fn in COMPARISON_SIDE_FNS.items():
                    side_override = side_fn(context_closes=ctx_closes, rng=rng)
                    trade = fn(pred or pred_long, candles, idx, funding_lookup, LEVERAGE, FEE_PCT,
                               DEFAULT_FUNDING_RATE_PCT, pred_long=pred_long, side_override=side_override)
                    if trade is not None:
                        strategy_trades[strat_name].append(trade)

        results[v] = strategy_trades
        print(f"[{period_name}] Varyant {v}: model={len(strategy_trades['model'])} islem "
              f"(karsilastirma: {', '.join(f'{k}={len(v_)}' for k, v_ in strategy_trades.items() if k != 'model')})")

    return results


def summarize(results: dict, variant_codes: list) -> dict:
    out = {}
    for v in variant_codes:
        strategy_trades = results[v]
        model_trades = strategy_trades["model"]
        model_pnls = [t["pnl_pct"] for t in model_trades]
        bootstrap = block_bootstrap_ci(model_pnls, block_size=BLOCK_SIZE, n_boot=N_BOOT, seed=RANDOM_SEED)

        comparisons = {}
        for strat_name in COMPARISON_SIDE_FNS:
            comp_trades = strategy_trades[strat_name]
            comp_pnls = [t["pnl_pct"] for t in comp_trades]
            sig = paired_diff_significance(model_pnls, comp_pnls, block_size=BLOCK_SIZE, n_boot=N_BOOT,
                                            seed=RANDOM_SEED)
            comparisons[strat_name] = {
                "summary": aggregate_trades(comp_trades),
                "bootstrap": block_bootstrap_ci(comp_pnls, block_size=BLOCK_SIZE, n_boot=N_BOOT, seed=RANDOM_SEED),
                "diff_vs_model": sig,
            }

        buy_only = [t for t in model_trades if t["side"] == "BUY"]
        buy_only_pnls = [t["pnl_pct"] for t in buy_only]
        always_buy_pnls = [t["pnl_pct"] for t in strategy_trades["always_buy"]]
        buy_only_vs_always_buy = paired_diff_significance(buy_only_pnls, always_buy_pnls, block_size=BLOCK_SIZE,
                                                            n_boot=N_BOOT, seed=RANDOM_SEED)

        by_coin = {}
        for symbol in COINS:
            coin_trades = [t for t in model_trades if t["coin"] == symbol]
            by_coin[symbol] = aggregate_trades(coin_trades)
            by_coin[symbol]["sample_size"] = len(coin_trades)
            by_coin[symbol]["reliable"] = len(coin_trades) >= 30

        out[v] = {
            "description": VARIANT_DESCRIPTIONS[v],
            "context_hours": VARIANT_CONTEXT[v],
            "model": {
                "summary": aggregate_trades(model_trades),
                "bootstrap_total_return": bootstrap,
                "n": len(model_trades),
            },
            "buy_only": {
                "summary": aggregate_trades(buy_only),
                "n": len(buy_only),
                "vs_always_buy": buy_only_vs_always_buy,
            },
            "comparisons": comparisons,
            "by_coin": by_coin,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locked", action="store_true", help="Kilitli test donemini calistir (bir kez)")
    parser.add_argument("--force-locked-rerun", action="store_true",
                         help="Kilitli test daha once calistiysa YINE DE tekrar calistir (dikkat: veri sizintisi riski)")
    parser.add_argument("--variants", default=None, help="Virgulle ayrilmis alt kume (varsayilan: hepsi)")
    args = parser.parse_args()

    variant_codes = args.variants.split(",") if args.variants else list(VARIANTS)
    for v in variant_codes:
        if v not in VARIANTS:
            print(f"HATA: bilinmeyen varyant '{v}'. Gecerliler: {list(VARIANTS)}", file=sys.stderr)
            return 1

    if args.locked and os.path.exists(LOCKED_MARKER) and not args.force_locked_rerun:
        with open(LOCKED_MARKER, encoding="utf-8") as f:
            marker = json.load(f)
        print(f"KILITLI TEST DAHA ONCE CALISTIRILDI ({marker.get('run_at')}, varyantlar: "
              f"{marker.get('variants')}) - TEKRAR CALISTIRILMADI. Zorlamak icin --force-locked-rerun kullanin "
              f"(bu veri sizintisi riski tasir, sadece bilerek yapin).")
        return 0

    split = load_split()
    coins_raw = load_raw_candles()
    funding_lookups = load_funding_lookups()

    if args.locked:
        period_name, start, end = "locked", parse(split["locked_start"]), parse(split["locked_end"])
        output_path = os.path.join("docs", "research_locked.json")
    else:
        period_name, start, end = "dev", parse(split["dev_start"]), parse(split["dev_end"])
        output_path = os.path.join("docs", "research_dev.json")

    print(f"=== {period_name.upper()} donemi: {fmt(start)} -> {fmt(end)} | varyantlar: {variant_codes} ===")
    results = run_period(period_name, start, end, variant_codes, coins_raw, funding_lookups)
    summary = summarize(results, variant_codes)

    payload = {
        "generated_at": fmt(datetime.now(timezone.utc)),
        "period": period_name,
        "period_start": fmt(start),
        "period_end": fmt(end),
        "leverage": LEVERAGE,
        "fee_pct": FEE_PCT,
        "default_funding_rate_pct": DEFAULT_FUNDING_RATE_PCT,
        "step_hours": STEP_HOURS,
        "bootstrap_block_size": BLOCK_SIZE,
        "bootstrap_n": N_BOOT,
        "variant_descriptions": {v: VARIANT_DESCRIPTIONS[v] for v in variant_codes},
        "variants": summary,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Sonuc -> {output_path}")

    if args.locked:
        marker = {"run_at": fmt(datetime.now(timezone.utc)), "variants": variant_codes,
                   "period_start": fmt(start), "period_end": fmt(end)}
        with open(LOCKED_MARKER, "w", encoding="utf-8") as f:
            json.dump(marker, f, ensure_ascii=False, indent=2)
        print(f"Kilitli test isareti yazildi -> {LOCKED_MARKER} (tekrar calismasi icin --force-locked-rerun gerekir)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
