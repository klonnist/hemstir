"""Arastirma verisi: 5 major coin icin ~12 aylik 1H OHLCV + (mumkun oldugunca)
gercek funding rate gecmisi ceker, yerel dosyalara onbellege alir ve gelistirme /
kilitli-test (dev/locked) bolumunu hesaplar.

Bolme kurali: ortak veri araliginin ilk ~%70'i "gelistirme" (dev), tam ortada
GAP_HOURS (48 saat) bosluk, son ~%30'u "kilitli test" (locked). Bolme TARIHI bir
kere hesaplanip scripts/cache/split.json'a yazilir; DOSYA VARSA bir DAHA
HESAPLANMAZ (veri zamanla guncellense bile bolme kaymasin diye) - degistirmek
icin FORCE_RESPLIT=1 gerekir (bu, gelistirme sonuclarini gormeden ONCE, ilk
kurulumda yapilmali; sonradan degistirmek veri sizintisidir).

OKX'in funding-rate-history ucu sinirli bir gecmis tutuyor (bu yazi itibariyle
~3 ay) - 12 aylik pencerenin buyuk kismi icin GERCEK funding verisi YOKTUR.
Script bunu acikca raporlar; eksik donem icin varsayilan sabit oran kullanilir
(bkz. eval_lib.DEFAULT_FUNDING_RATE_PCT / --default-funding-rate).

Kullanim: python scripts/research_data.py
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_forecasts import BAR, COINS  # noqa: E402
from okx_client import fetch_funding_rate_history, fetch_history_candles  # noqa: E402

RESEARCH_DAYS = int(os.environ.get("RESEARCH_DAYS", "365"))
CACHE_DIR = os.environ.get("RESEARCH_CACHE_DIR", os.path.join("scripts", "cache"))
DEV_FRACTION = 0.70
GAP_HOURS = 48
FORCE_RESPLIT = os.environ.get("FORCE_RESPLIT", "0") == "1"


def _raw_path(symbol):
    return os.path.join(CACHE_DIR, "raw", f"{symbol}.json")


def _funding_path(symbol):
    return os.path.join(CACHE_DIR, "funding", f"{symbol}.json")


def _split_path():
    return os.path.join(CACHE_DIR, "split.json")


def fmt(dt_val):
    return dt_val.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def fetch_ohlcv(symbol, inst_id, start, end):
    print(f"{symbol}: OHLCV cekiliyor ({start.date()} -> {end.date()})...")
    df = fetch_history_candles(inst_id, BAR, start, end)
    candles = [
        {"ts": fmt(row.timestamp.to_pydatetime().replace(tzinfo=timezone.utc)),
         "open": float(row.open), "high": float(row.high), "low": float(row.low), "close": float(row.close)}
        for row in df.itertuples()
    ]
    path = _raw_path(symbol)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"symbol": symbol, "inst_id": inst_id, "candles": candles}, f, ensure_ascii=False)
    print(f"{symbol}: {len(candles)} mum -> {path}")
    return candles


def fetch_funding(symbol, inst_id, start, end):
    try:
        df = fetch_funding_rate_history(inst_id, start, end)
    except Exception as e:
        print(f"UYARI: {symbol} funding verisi alinamadi: {e}", file=sys.stderr)
        df = None
    events = []
    if df is not None and not df.empty:
        events = [
            {"ts": fmt(row.timestamp.to_pydatetime().replace(tzinfo=timezone.utc)), "rate": float(row.funding_rate)}
            for row in df.itertuples()
        ]
    path = _funding_path(symbol)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"symbol": symbol, "inst_id": inst_id, "events": events}, f, ensure_ascii=False)
    coverage_start = events[0]["ts"] if events else None
    print(f"{symbol}: {len(events)} gercek funding kaydi (kapsam: {coverage_start} -> {fmt(end)}) -> {path}")
    return events


def main() -> int:
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=RESEARCH_DAYS)

    coin_ranges = {}
    for symbol, inst_id in COINS.items():
        candles = fetch_ohlcv(symbol, inst_id, start, end)
        fetch_funding(symbol, inst_id, start, end)
        if not candles:
            print(f"HATA: {symbol} icin veri yok.", file=sys.stderr)
            return 1
        coin_ranges[symbol] = (parse(candles[0]["ts"]), parse(candles[-1]["ts"]))

    # Tum coinlerde ORTAK olan araligi bul (bazi coinler daha gec listelenmis olabilir).
    common_start = max(r[0] for r in coin_ranges.values())
    common_end = min(r[1] for r in coin_ranges.values())

    split_path = _split_path()
    if os.path.exists(split_path) and not FORCE_RESPLIT:
        with open(split_path, encoding="utf-8") as f:
            split = json.load(f)
        print(f"Bolme tarihi zaten var, YENIDEN HESAPLANMADI (FORCE_RESPLIT=1 vermediginiz surece): {split_path}")
    else:
        total_span = common_end - common_start
        dev_end = common_start + total_span * DEV_FRACTION
        locked_start = dev_end + timedelta(hours=GAP_HOURS)
        split = {
            "computed_at": fmt(datetime.now(timezone.utc)),
            "common_start": fmt(common_start),
            "common_end": fmt(common_end),
            "dev_start": fmt(common_start),
            "dev_end": fmt(dev_end),
            "gap_hours": GAP_HOURS,
            "locked_start": fmt(locked_start),
            "locked_end": fmt(common_end),
            "dev_fraction": DEV_FRACTION,
        }
        os.makedirs(os.path.dirname(split_path), exist_ok=True)
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=False, indent=2)
        print(f"Yeni bolme tarihi hesaplandi ve KAYDEDILDI: {split_path}")

    print()
    print(f"Ortak veri araligi : {fmt(common_start)} -> {fmt(common_end)}")
    print(f"Gelistirme (dev)   : {split['dev_start']} -> {split['dev_end']}")
    print(f"Bosluk             : {split['gap_hours']} saat")
    print(f"Kilitli test       : {split['locked_start']} -> {split['locked_end']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
