"""
Fetch daily TVL for all selected pools from the Uniswap V3 subgraph.

Queries the pool entity at historical block numbers (one per day) to get
totalValueLockedUSD for every pool — including inactive pools with no trades.
This gives 100% coverage, unlike poolDayDatas or poolHourDatas which only
exist for days/hours with trading activity.

Output: one CSV per day in data/filtered/pools_daily/{month}/{date}.csv
"""

# --- Old imports for the poolDayDatas approach ---
# from uniswapv3_async import get_pool_data_at_unix_timestamp

from datetime import datetime
from tqdm import tqdm
import asyncio
import pandas as pd
import os
import argparse
import time

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.data.collection.uniswapv3_async import query_data_async
from src.data.collection.etherscan import get_block_by_timestamp
from src.data.collection.pools_config import POOLS, get_active_pools, get_pools_to_ignore

from gql import gql

pools_to_ignore = get_pools_to_ignore()

# Active pools: {address: "TOKEN0TOKEN1<fee>"} label
POOLS_AND_TRADING_PAIRS = {
    addr: info["pair"].replace("/", "") + str(info["fee_tier"])
    for addr, info in POOLS.items()
}


# ---------------------------------------------------------------------------
# Batch pool entity query
# ---------------------------------------------------------------------------

async def fetch_pool_tvl_at_block(pool_addresses, block_number, batch_size=50):
    """Query pool entities at a historical block for TVL.

    Uses GraphQL aliases to batch up to `batch_size` pools per request,
    minimising round-trips to the subgraph.

    Returns a list of dicts: {pool_address, totalValueLockedUSD, block_number,
    token0_symbol, token1_symbol, fee_tier, pair, trading_pair}.
    """
    results = []

    for batch_start in range(0, len(pool_addresses), batch_size):
        batch = pool_addresses[batch_start:batch_start + batch_size]

        # Build aliased query: p0: pool(...) { ... }  p1: pool(...) { ... }
        fragments = []
        for i, addr in enumerate(batch):
            fragments.append(
                f'p{i}: pool(id: "{addr}", block: {{number: {block_number}}}) '
                f'{{ totalValueLockedUSD volumeUSD token0 {{ symbol }} token1 {{ symbol }} feeTier }}'
            )
        query = gql("{ " + " ".join(fragments) + " }")
        response = await query_data_async(query)

        for i, addr in enumerate(batch):
            pool_data = response["data"][f"p{i}"]
            if not pool_data:
                continue
            t0 = pool_data["token0"]["symbol"]
            t1 = pool_data["token1"]["symbol"]
            fee = int(pool_data["feeTier"])
            results.append({
                "pool_address": addr,
                "totalValueLockedUSD": float(pool_data["totalValueLockedUSD"]),
                "volumeUSD": float(pool_data["volumeUSD"]),
                "block_number": block_number,
                "token0_symbol": t0,
                "token1_symbol": t1,
                "fee_tier": fee,
                "pair": f"{t0}/{t1}",
                "trading_pair": f"{t0}{t1}{fee}",
            })

    return results


# ---------------------------------------------------------------------------
# Old approach: query poolDayDatas per pool (slow, incomplete coverage)
# ---------------------------------------------------------------------------
# async def main():
#     ...
#     for date_val in tqdm(dates_range):
#         ...
#         pools_data = await asyncio.gather(
#             *[get_pool_data_at_unix_timestamp(pool_id, unix_timestamp)
#               for pool_id in unique_pool_ids])
#         pools_data = [data for data in pools_data if data is not None]
#         pools_df = pd.concat(pools_data)
#         pools_df.rename(columns={'volumeUSD': '24hr Volume (USD)',
#                                  'tvlUSD': 'totalValueLockedUSD'}, inplace=True)
#         pools_df.to_csv(pool_stats_path, index=False)


async def main():
    parser = argparse.ArgumentParser(
        description="Fetch daily TVL for all pools via pool entity at block")

    parser.add_argument('--csv_files_save_directory', type=str,
                        help='Directory to save output CSVs',
                        default="./../../../data/filtered")
    parser.add_argument('--start_date', type=str, default="2023-02-28")
    parser.add_argument('--end_date', type=str, default="2023-04-01")
    args = parser.parse_args()

    csv_path_to_save = os.path.join(args.csv_files_save_directory, "pools_daily")
    os.makedirs(csv_path_to_save, exist_ok=True)

    # Active pool addresses (excluding ignored)
    pool_addresses = [
        addr for addr in POOLS_AND_TRADING_PAIRS
        if addr not in pools_to_ignore
    ]

    dates_range = pd.date_range(start=args.start_date, end=args.end_date, freq='D')

    for date_val in tqdm(dates_range):
        date_str = datetime.strftime(date_val, '%Y-%m-%d')
        month = datetime.strftime(date_val, '%m')

        dir_file = os.path.join(csv_path_to_save, month)
        os.makedirs(dir_file, exist_ok=True)
        pool_stats_path = os.path.join(dir_file, f"{date_str}.csv")

        # Incremental: load existing data, skip already-scraped pools
        existing_df = pd.read_csv(pool_stats_path) if os.path.exists(pool_stats_path) else None
        already_scraped = set(existing_df['pool_address'].unique()) if existing_df is not None else set()
        pools_remaining = [a for a in pool_addresses if a not in already_scraped]

        if not pools_remaining:
            continue

        if already_scraped:
            print(f"{len(already_scraped)} pools done, {len(pools_remaining)} remaining for {date_str}")

        # Get block number for midnight UTC of this date
        print(f"Processing {date_str}...")
        block_number = get_block_by_timestamp(date_val)

        # Batch query all remaining pools at this block
        rows = await fetch_pool_tvl_at_block(pools_remaining, block_number)

        if not rows:
            print(f"No data for {date_str}, skipping")
            continue

        new_df = pd.DataFrame(rows)

        # Add stablecoin column from pool config
        new_df["stablecoin"] = new_df["pool_address"].map(
            lambda a: POOLS[a]["stablecoin"] if a in POOLS else None
        )

        # Add date column
        new_df["date"] = date_str

        # Append to existing data and save
        combined = pd.concat([existing_df, new_df]) if existing_df is not None else new_df
        print(f"Saving {date_str}: {combined['pool_address'].nunique()} pools")
        combined.to_csv(pool_stats_path, index=False)


if __name__ == "__main__":
    # Retry on transient failures (e.g. subgraph rate limits)
    for i in range(100):
        print(f"Attempt {i + 1}...")
        try:
            asyncio.run(main())
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
            print("Retrying...")
            continue
