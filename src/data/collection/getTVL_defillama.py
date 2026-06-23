"""
Fetch historical TVL data from DeFi Llama for Uniswap V3 and stablecoins.

Sources:
  - Uniswap V3 total TVL (Ethereum): DeFi Llama protocol API
  - USDC / USDT token TVL on Uniswap V3: Uniswap interface gateway
"""

import requests
import argparse
from datetime import datetime, timezone


def get_uniswap_v3_tvl(date_str=None):
    """Fetch Uniswap V3 TVL from DeFi Llama.

    Args:
        date_str: Optional date string (YYYY-MM-DD). If None, returns all history.

    Returns:
        dict with 'all_chains' and 'ethereum' TVL.
        If date_str is provided, returns the closest matching entry.
    """
    resp = requests.get("https://api.llama.fi/protocol/uniswap-v3")
    resp.raise_for_status()
    data = resp.json()

    all_chains_tvl = data.get("tvl", [])
    eth_tvl = data.get("chainTvls", {}).get("Ethereum", {}).get("tvl", [])

    if date_str:
        target_ts = int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())

        def closest(entries, ts):
            return min(entries, key=lambda e: abs(e["date"] - ts))

        all_entry = closest(all_chains_tvl, target_ts)
        eth_entry = closest(eth_tvl, target_ts) if eth_tvl else None

        return {
            "date": date_str,
            "all_chains": {
                "date": datetime.fromtimestamp(all_entry["date"]).strftime("%Y-%m-%d"),
                "tvl": all_entry["totalLiquidityUSD"],
            },
            "ethereum": {
                "date": datetime.fromtimestamp(eth_entry["date"]).strftime("%Y-%m-%d"),
                "tvl": eth_entry["totalLiquidityUSD"],
            } if eth_entry else None,
        }

    return {"all_chains": all_chains_tvl, "ethereum": eth_tvl}



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch TVL data from DeFi Llama and Uniswap")
    parser.add_argument("--date", type=str, default="2023-03-01",
                        help="Date to query (YYYY-MM-DD)")
    args = parser.parse_args()

    print(f"=== TVL on {args.date} ===\n")

    # Uniswap V3 total TVL
    uni = get_uniswap_v3_tvl(args.date)
    print(f"Uniswap V3 (all chains): ${uni['all_chains']['tvl']:,.2f}")
    if uni["ethereum"]:
        print(f"Uniswap V3 (Ethereum):   ${uni['ethereum']['tvl']:,.2f}")
