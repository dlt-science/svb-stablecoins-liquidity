import requests
import argparse
from datetime import datetime

def get_tvl(address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", chain="ETHEREUM", date_filter=None):
    """Fetch TVL data for a stablecoin. Optionally filter by date (YYYY-MM-DD)."""
    resp = requests.post("https://interface.gateway.uniswap.org/v1/graphql", json={
        "operationName": "TokenHistoricalTvls",
        "variables": {"address": address, "chain": chain, "duration": "MAX"},
        "query": """query TokenHistoricalTvls($chain: Chain!, $address: String, $duration: HistoryDuration!) {
            token(chain: $chain, address: $address) {
                id address chain
                market(currency: USD) {
                    id
                    historicalTvl(duration: $duration) { id timestamp value }
                    totalValueLocked { id value currency }
                }
            }
        }"""
    }, headers={
        "Content-Type": "application/json",
        "Origin": "https://app.uniswap.org",
        "Referer": "https://app.uniswap.org/",
    })
    resp.raise_for_status()
    data = resp.json()["data"]["token"]

    if date_filter:
        target = datetime.strptime(date_filter, "%Y-%m-%d").date()
        entries = data["market"]["historicalTvl"]
        data["market"]["historicalTvl"] = [
            e for e in entries if datetime.fromtimestamp(e["timestamp"]).date() == target
        ]
    return data

# Example usage
if __name__ == "__main__":
    # All historical TVL for USDC and USDT
    coins = {
        "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "USDT": "0xdac17f958d2ee523a2206206994597c13d831ec7"
        }
    date_filter = "2023-03-01"  # Filter for 1 March 2023
    for coin, address in coins.items():
        print(f"Fetching TVL data for {coin}...")
        result = get_tvl(address=address, date_filter=date_filter)
        print(f"Current TVL: ${result['market']['totalValueLocked']['value']:,.2f}")
        print(f"Total data points: {len(result['market']['historicalTvl'])}")

        for entry in result["market"]["historicalTvl"]:
            dt = datetime.fromtimestamp(entry["timestamp"])
            print(f"{dt}: ${entry['value']:,.2f}")