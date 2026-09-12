"""OKX genel (public) piyasa verisi API'si icin ince istemci (API anahtari gerekmez)."""
import pandas as pd
import requests

BASE_URL = "https://www.okx.com"
COLUMNS = ["ts", "open", "high", "low", "close", "volume", "vol_ccy", "vol_ccy_quote", "confirm"]


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
