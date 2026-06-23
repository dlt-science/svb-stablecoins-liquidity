"""
Fetch historical stablecoin prices from CoinCap's free API.

Downloads hourly USDC and USDT price data for the SVB crisis period
and saves as CSV files for use by Prices.py.

CoinCap's /assets/{id}/history endpoint provides free hourly data
with an interval parameter, no API key required.

Usage:
    python fetch_stablecoin_prices.py
    python fetch_stablecoin_prices.py --output_dir ../../data/filtered
"""

import argparse
import os
import time

import pandas as pd
import requests

COINCAP_BASE = "https://api.coincap.io/v2"

STABLECOINS = {
    "USDC": "usd-coin",
    "USDT": "tether",
}

DEFAULT_START = "2023-02-28"
DEFAULT_END = "2023-03-30"


def fetch_coincap_history(asset_id, start_ms, end_ms):
    """Fetch hourly price history from CoinCap.

    CoinCap /assets/{id}/history?interval=h1 returns hourly candles.
    Max 2000 data points per request (~83 days at hourly).
    """
    url = f"{COINCAP_BASE}/assets/{asset_id}/history"
    params = {
        "interval": "h1",
        "start": int(start_ms),
        "end": int(end_ms),
    }

    print(f"  Fetching {asset_id} from CoinCap...")
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    records = data.get("data", [])
    if not records:
        raise ValueError(f"No data returned for {asset_id}")

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["close"] = df["priceUsd"].astype(float)
    df = df[["timestamp", "close"]].sort_values("timestamp").reset_index(drop=True)

    print(f"  Got {len(df)} data points: "
          f"{df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Fetch stablecoin prices from CoinCap")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "..", "..", "..", "data", "filtered"))
    parser.add_argument("--start_date", type=str, default=DEFAULT_START)
    parser.add_argument("--end_date", type=str, default=DEFAULT_END)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    start_ms = pd.Timestamp(args.start_date, tz="UTC").value // 10**6
    end_ms = pd.Timestamp(args.end_date, tz="UTC").value // 10**6

    for symbol, asset_id in STABLECOINS.items():
        df = fetch_coincap_history(asset_id, start_ms, end_ms)

        out_path = os.path.join(args.output_dir,
                                f"{symbol}_price_coincap_data.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved {out_path}")

        time.sleep(1)

    print("\nDone — stablecoin price data fetched.")


if __name__ == "__main__":
    main()
