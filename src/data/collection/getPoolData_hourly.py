"""
Fetch hourly TVL for all selected pools from the Uniswap V3 subgraph.

Queries the pool entity at historical block numbers (one per hour, sourced
from the RPC parquet files in data/pools/rpc/) to get totalValueLockedUSD
and volumeUSD for every pool.  This mirrors the approach in getPoolData_daily.py
but at hourly granularity.

Output: one CSV per hour in data/filtered/pools_hourly/{month}/{datetime}.csv
"""

from tqdm import tqdm
import asyncio
import pandas as pd
import os
import argparse
import time
import logging

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.data.collection.uniswapv3_async import query_data_async
from src.data.collection.pools_config import POOLS, get_pools_to_ignore
from gql import gql

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

pools_to_ignore = get_pools_to_ignore()

# Active pools: {address: "TOKEN0TOKEN1<fee>"} label
POOLS_AND_TRADING_PAIRS = {
    addr: info["pair"].replace("/", "") + str(info["fee_tier"])
    for addr, info in POOLS.items()
}


# ---------------------------------------------------------------------------
# Block numbers from local RPC parquet files
# ---------------------------------------------------------------------------

def get_block_from_rpc(datetime_str, rpc_dir):
    """Extract block number from the RPC parquet file for a given datetime.

    Reads the parquet file, takes the first pool's block_number (all pools
    share the same block within a single hourly snapshot).

    Returns block_number or None if the file is missing.
    """
    month = datetime_str[5:7]
    parquet_path = os.path.join(rpc_dir, month, f"{datetime_str}.parquet")

    if not os.path.exists(parquet_path):
        return None

    df = pd.read_parquet(parquet_path, columns=["block_number"])
    return int(df["block_number"].iloc[0])


# ---------------------------------------------------------------------------
# Batch pool entity query
# ---------------------------------------------------------------------------

async def fetch_pool_tvl_at_block(pool_addresses, block_number, batch_size=50):
    """Query pool entities at a historical block for TVL.

    Uses GraphQL aliases to batch up to `batch_size` pools per request,
    minimising round-trips to the subgraph.

    Returns a list of dicts: {pool_address, totalValueLockedUSD, volumeUSD,
    block_number, token0_symbol, token1_symbol, fee_tier, pair, trading_pair}.
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
# Entry point
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Fetch hourly TVL for all pools via pool entity at block")

    parser.add_argument('--csv_files_save_directory', type=str,
                        help='Directory to save output CSVs',
                        default="./../../../data/filtered")
    parser.add_argument('--rpc_dir', type=str,
                        help='Directory with RPC hourly parquet files',
                        default="./../../../data/pools/rpc")
    parser.add_argument('--start_date', type=str, default="2023-02-28")
    parser.add_argument('--end_date', type=str, default="2023-04-01")
    args = parser.parse_args()

    csv_path_to_save = os.path.join(args.csv_files_save_directory, "pools_hourly")
    os.makedirs(csv_path_to_save, exist_ok=True)

    # Active pool addresses (excluding ignored)
    pool_addresses = [
        addr for addr in POOLS_AND_TRADING_PAIRS
        if addr not in pools_to_ignore
    ]

    dates_range = pd.date_range(start=args.start_date, end=args.end_date, freq='h')

    for dt in tqdm(dates_range, desc="Hours"):
        datetime_str = dt.strftime('%Y-%m-%d %H-%M-%S')
        month = dt.strftime('%m')

        dir_file = os.path.join(csv_path_to_save, month)
        os.makedirs(dir_file, exist_ok=True)
        pool_stats_path = os.path.join(dir_file, f"{datetime_str}.csv")

        # Incremental: load existing data, skip already-scraped pools
        existing_df = pd.read_csv(pool_stats_path) if os.path.exists(pool_stats_path) else None
        already_scraped = set(existing_df['pool_address'].unique()) if existing_df is not None else set()
        pools_remaining = [a for a in pool_addresses if a not in already_scraped]

        if not pools_remaining:
            continue

        if already_scraped:
            log.info(f"{len(already_scraped)} pools done, "
                     f"{len(pools_remaining)} remaining for {datetime_str}")

        # Get block number from the RPC parquet file (no Etherscan call needed)
        block_number = get_block_from_rpc(datetime_str, args.rpc_dir)
        if block_number is None:
            log.warning(f"No RPC parquet for {datetime_str}, skipping")
            continue

        log.info(f"Processing {datetime_str} at block {block_number}")

        # Batch query all remaining pools at this block
        rows = await fetch_pool_tvl_at_block(pools_remaining, block_number)

        if not rows:
            log.info(f"No data for {datetime_str}, skipping")
            continue

        new_df = pd.DataFrame(rows)

        # Add stablecoin column from pool config
        new_df["stablecoin"] = new_df["pool_address"].map(
            lambda a: POOLS[a]["stablecoin"] if a in POOLS else None
        )

        # Add datetime column
        new_df["date"] = datetime_str

        # Append to existing data and save
        combined = pd.concat([existing_df, new_df]) if existing_df is not None else new_df
        log.info(f"Saving {datetime_str}: {combined['pool_address'].nunique()} pools")
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
