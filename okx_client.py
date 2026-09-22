"""OKX genel (public) piyasa verisi API'si icin ince istemci (API anahtari gerekmez)."""
import time
from datetime import datetime, timezone

import pandas as pd
import requests

BASE_URL = "https://www.okx.com"
COLUMNS = ["ts", "open", "high", "low", "close", "volume", "vol_ccy", "vol_ccy_quote", "confirm"]


def _to_ms(dt_like) -> int:
    if isinstance(dt_like, (int, float)):
        return int(dt_like)
    dt = pd.Timestamp(dt_like)
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    return int(dt.timestamp() * 1000)


def _rows_to_df(rows) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_recent_candles(inst_id: str, bar: str, limit: int = 300) -> pd.DataFrame:
    """En guncel `limit` mum verisini doner (limit <= 300, OKX API'nin tek istek siniri)."""
    resp = requests.get(
        f"{BASE_URL}/api/v5/market/candles",
        params={"instId": inst_id, "bar": bar, "limit": min(limit, 300)},
        timeout=15,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("code") != "0":
        raise RuntimeError(f"OKX API hatasi ({inst_id}): {payload}")
    return _rows_to_df(payload["data"])


def fetch_history_candles(inst_id: str, bar: str, start, end=None, pause: float = 0.15) -> pd.DataFrame:
    """[start, end] araligini (dahil) /market/history-candles ile sayfalayarak kapsar.

    Backtest icin gecmise donuk cok sayida saatlik mum gerekince kullanilir
    (crypto-trader/okx_trader/okx_client.py ile ayni sayfalama mantigi)."""
    start_ms = _to_ms(start)
    end_ms = _to_ms(end) if end is not None else int(datetime.now(timezone.utc).timestamp() * 1000)

    all_rows = []
    after = None
    seen_ts = set()
    while True:
        params = {"instId": inst_id, "bar": bar, "limit": 100}
        if after is not None:
            params["after"] = after
        resp = requests.get(f"{BASE_URL}/api/v5/market/history-candles", params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("code") != "0":
            raise RuntimeError(f"OKX API hatasi ({inst_id}): {payload}")
        rows = payload["data"]
        if not rows:
            break

        new_rows = [r for r in rows if r[0] not in seen_ts]
        if not new_rows:
            break
        for r in new_rows:
            seen_ts.add(r[0])
        all_rows.extend(new_rows)

        oldest_ts = min(int(r[0]) for r in rows)
        after = oldest_ts
        if oldest_ts <= start_ms:
            break
        time.sleep(pause)

    df = _rows_to_df(all_rows)
    if df.empty:
        return df
    mask = (df["timestamp"] >= pd.Timestamp(start_ms, unit="ms", tz="UTC")) & (
        df["timestamp"] <= pd.Timestamp(end_ms, unit="ms", tz="UTC")
    )
    return df.loc[mask].reset_index(drop=True)
