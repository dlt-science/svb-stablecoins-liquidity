"""Collect raw, unaggregated Uniswap V3 NFT position data hourly.

Persists every pool-hour's full subgraph response to
data/raw/positions_hourly/{month}/{datetime}.csv with one row per active
NFT position. Each row is *self-describing*: it carries the position
primitives from the subgraph (id, owner, liquidity, tickLower, tickUpper)
plus the pool-and-snapshot context needed to reproduce per-position USD
valuation downstream — datetime, block number, trading pair, fee tier,
stablecoin leg, tick spacing, token symbols and decimals, the pool's
active tick (cross_tick), its sqrtPriceX96, and the non-stablecoin leg's
mid-price (price_in_stablecoin).

Companion to getPositionsData.py, which produces the wallet-aggregated,
fee-normalised view under data/filtered/. Both scripts share the same
subgraph paginator (fetch_positions_at_block) and pool-state loader
(load_pool_states_from_rpc), so this one only adds the persistence step
that the original pipeline performed in-memory.

Run:
    uv run python -m src.data.collection.getRawPositions \\
        --start_date 2023-02-28 --end_date 2023-04-02
"""

import argparse
import asyncio
import logging
import os
import sys

import aiohttp
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.data.collection.getPositionsData import (  # reuse — same subgraph
    fetch_positions_at_block,                       # paginator and pool-state
    load_pool_states_from_rpc,                      # loader as the aggregated
)                                                   # collection script.
from src.data.collection.pools_config import POOLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)


def _hour_grid(start_date: str, end_date: str) -> list[str]:
    """Hourly UTC timestamps formatted to match the RPC parquet filenames
    ("YYYY-MM-DD HH-MM-SS"). Inclusive of start, exclusive of end."""
    return [t.strftime("%Y-%m-%d %H-%M-%S")
            for t in pd.date_range(start_date, end_date, freq="h",
                                   inclusive="left")]


def _to_iso(datetime_str: str) -> str:
    """RPC parquet filename uses dashes in the time component
    ('2023-03-01 00-00-00'); convert to ISO-style ('2023-03-01 00:00:00')
    so downstream consumers can parse it as a UTC timestamp."""
    if " " in datetime_str:
        d, t = datetime_str.split(" ")
        return f"{d} {t.replace('-', ':')}"
    return f"{datetime_str} 00:00:00"


def _flatten(positions: list[dict], state: dict,
             pool_address: str, datetime_iso: str) -> pd.DataFrame:
    """Hoist the subgraph response's nested tickLower/tickUpper objects
    into flat columns and tag every row with the pool/snapshot context
    needed to reproduce per-position USD valuation without joining back
    to the RPC parquets."""
    return pd.DataFrame([{
        # Snapshot identifiers
        "datetime_utc":         datetime_iso,
        "block_number":         state["block_number"],
        # Pool identity
        "pool_address":         pool_address,
        "trading_pair":         state["trading_pair"],
        "fee_tier":             state["fee_tier"],
        "stablecoin":           POOLS[pool_address]["stablecoin"],
        "tick_spacing":         state["tick_spacing"],
        "token0_symbol":        state["token0_symbol"],
        "token1_symbol":        state["token1_symbol"],
        "token0_decimals":      state["token0_decimals"],
        "token1_decimals":      state["token1_decimals"],
        # Pool state at this hour (needed for USD valuation)
        "cross_tick":           state["cross_tick"],
        "sqrt_price_x96":       state["sqrt_price_x96"],
        "price_in_stablecoin":  state["price_in_stablecoin"],
        # Position primitives (literal subgraph response)
        "position_id":          p["id"],
        "owner":                p["owner"],
        "liquidity":            p["liquidity"],
        "tickLower":            int(p["tickLower"]["tickIdx"]),
        "tickUpper":            int(p["tickUpper"]["tickIdx"]),
    } for p in positions])


async def _fetch_with_retry(pool_address: str, block_number: int,
                            max_attempts: int = 6) -> list[dict]:
    """fetch_positions_at_block with exponential-backoff retry on
    transient network errors. After max_attempts, returns [] and logs;
    the caller continues with the next pool so a single flaky pool-hour
    does not abort a multi-hour run."""
    for attempt in range(1, max_attempts + 1):
        try:
            return await fetch_positions_at_block(pool_address, block_number)
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            wait = min(60, 2 ** attempt)  # 2,4,8,16,32,60 seconds
            log.warning(f"  {pool_address[:10]}... attempt {attempt}/{max_attempts}"
                        f" failed ({type(e).__name__}): retrying in {wait}s")
            await asyncio.sleep(wait)
    log.error(f"  {pool_address[:10]}... exhausted retries; skipping")
    return []


async def _process_timestamp(datetime_str: str, rpc_dir: str,
                             out_dir: str) -> None:
    """Fetch every active pool's raw positions at one hourly snapshot and
    persist them to a single CSV. Skips pools already present in the
    output file so the script is restartable."""
    month = datetime_str[5:7]
    out_path = os.path.join(out_dir, month, f"{datetime_str}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Pool states (block_number, etc.) come from the same RPC parquet the
    # aggregated collector reads — guarantees identical block alignment.
    states = load_pool_states_from_rpc(datetime_str, rpc_dir)
    if not states:
        return

    targets = [p for p in states
               if p in POOLS and POOLS[p].get("active", True)]

    # Incremental restart: skip pools already in the output file.
    existing = pd.read_csv(out_path) if os.path.exists(out_path) else None
    done = (set(existing["pool_address"].unique())
            if existing is not None else set())
    remaining = [p for p in targets if p not in done]
    if not remaining:
        log.info(f"{datetime_str}: all {len(done)} pools already processed")
        return

    log.info(f"{datetime_str}: {len(remaining)} pools to fetch "
             f"(block {states[remaining[0]]['block_number']})")

    datetime_iso = _to_iso(datetime_str)
    rows = []
    for addr in tqdm(remaining, desc=datetime_str, leave=False):
        # fetch_positions_at_block paginates with id_gt cursors until empty.
        # Retry transient network errors so a single connection blip does
        # not abort the whole multi-hour run.
        positions = await _fetch_with_retry(addr, states[addr]["block_number"])
        if positions:
            rows.append(_flatten(positions, states[addr], addr, datetime_iso))

    if rows:
        new_df = pd.concat(rows, ignore_index=True)
        combined = (pd.concat([existing, new_df], ignore_index=True)
                    if existing is not None else new_df)
        combined.to_csv(out_path, index=False)
        log.info(f"{datetime_str}: saved "
                 f"{combined['pool_address'].nunique()} pools, "
                 f"{len(combined)} positions")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=str, default="./../../../data",
                        help="base data dir (contains pools/ and raw/)")
    parser.add_argument("--start_date", type=str, default="2023-02-28",
                        help="inclusive start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2023-04-02",
                        help="exclusive end date (YYYY-MM-DD)")
    args = parser.parse_args()

    rpc_dir = os.path.join(args.data_dir, "pools", "rpc")
    out_dir = os.path.join(args.data_dir, "raw", "positions_hourly")
    os.makedirs(out_dir, exist_ok=True)

    for ts in tqdm(_hour_grid(args.start_date, args.end_date),
                   desc="Timestamps"):
        await _process_timestamp(ts, rpc_dir, out_dir)


if __name__ == "__main__":
    asyncio.run(main())
