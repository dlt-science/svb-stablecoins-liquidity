"""Collect per-LP position data for Gini coefficient analysis.

For each pool on each timestamp (daily or hourly), reads pool state (sqrtPrice,
current tick, prices, decimals, fee_tier) from RPC parquet files in
data/pools/rpc/ and fetches all active NFT positions from the Uniswap V3
subgraph at the same block number.  Computes each position's TVL in USD using
concentrated liquidity math, aggregates by LP (owner address), and normalises
so the sum matches the pool's totalValueLockedUSD from pools_daily or
pools_hourly.

Pool state source:
    data/pools/rpc/{month}/{datetime}.parquet — one row per tick per pool.
    Each pool's constant fields (cross_tick, sqrt_price_x96, block_number,
    fee_tier, price_in_stablecoin, token decimals, etc.) are extracted once
    per pool to avoid redundant subgraph calls.

Why normalisation is needed:
    The Position entity captures 99%+ of all position liquidity, but the pool's
    totalValueLockedUSD also includes uncollected trading fees sitting in the
    contract (~40-50% of total). Normalising distributes fees proportionally
    to each LP's liquidity share, which preserves the relative distribution
    needed for Gini coefficient calculation while ensuring the sum matches
    the pool's reported TVL exactly.

Modes:
    --hourly: processes every hour using pools_hourly TVL for normalisation.
              Output: data/filtered/positions_hourly/{month}/{datetime}.csv
    default:  processes midnight only using pools_daily TVL for normalisation.
              Output: data/filtered/positions_daily/{month}/{date}.csv
"""

from decimal import Decimal, getcontext
from tqdm import tqdm
import asyncio
import pandas as pd
import numpy as np
import os
import argparse
import time
import logging

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.data.collection.uniswapv3_async import query_data_async
from src.data.collection.pools_config import POOLS
from gql import gql

# High precision for sqrtPrice X96 → float conversion
getcontext().prec = 34

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pool state from local parquet files (data/pools/rpc/)
# ---------------------------------------------------------------------------

def load_pool_states_from_rpc(datetime_str, rpc_dir):
    """Load pool states from an RPC parquet file for a given datetime.

    Each parquet file has one row per tick per pool.  We extract the constant
    pool-level fields (cross_tick, sqrt_price_x96, block_number, fee_tier,
    price_in_stablecoin, token decimals, trading_pair) by taking the first
    row per pool.

    Args:
        datetime_str: Either "YYYY-MM-DD" (daily, maps to midnight file)
                      or "YYYY-MM-DD HH-MM-SS" (hourly, maps directly).

    Returns a dict keyed by lowercase pool address → pool state dict.
    """
    month = datetime_str[5:7]

    # Daily mode passes "YYYY-MM-DD", hourly passes "YYYY-MM-DD HH-MM-SS"
    if len(datetime_str) == 10:
        filename = f"{datetime_str} 00-00-00.parquet"
    else:
        filename = f"{datetime_str}.parquet"

    parquet_path = os.path.join(rpc_dir, month, filename)

    if not os.path.exists(parquet_path):
        log.warning(f"No RPC parquet for {datetime_str}: {parquet_path}")
        return {}

    df = pd.read_parquet(parquet_path)

    # Pool-level constants — same on every tick row within a pool
    pool_cols = [
        "pool_address", "cross_tick", "sqrt_price_x96", "block_number",
        "fee_tier", "price_in_stablecoin", "token0_symbol", "token1_symbol",
        "token0_decimals", "token1_decimals", "trading_pair", "tick_spacing",
    ]
    # One row per pool (all tick rows share the same pool-level values)
    pools = df.groupby("pool_address")[pool_cols].first().reset_index(drop=True)

    # Build lookup dict keyed by lowercase address (subgraph uses lowercase)
    states = {}
    for _, row in pools.iterrows():
        addr_lower = row["pool_address"].lower()
        states[addr_lower] = {
            "cross_tick": int(row["cross_tick"]),
            "sqrt_price_x96": str(row["sqrt_price_x96"]),
            "block_number": int(row["block_number"]),
            "fee_tier": int(row["fee_tier"]),
            "price_in_stablecoin": float(row["price_in_stablecoin"]),
            "token0_symbol": row["token0_symbol"],
            "token1_symbol": row["token1_symbol"],
            "token0_decimals": int(row["token0_decimals"]),
            "token1_decimals": int(row["token1_decimals"]),
            "trading_pair": row["trading_pair"],
            "tick_spacing": int(row["tick_spacing"]),
        }

    return states


# ---------------------------------------------------------------------------
# Subgraph query — only used for positions (pool state comes from parquet)
# ---------------------------------------------------------------------------

async def fetch_positions_at_block(pool_address, block_number):
    """Paginate all active NFT positions for a pool at a historical block.

    Uses id_gt cursor pagination to avoid the Graph Protocol 5000-skip limit.
    Returns a list of position dicts with owner, liquidity, and tick bounds.
    """
    positions, last_id = [], ""

    while True:
        query = gql(f"""{{
            positions(first: 1000, orderBy: id, orderDirection: asc,
                      where: {{pool: "{pool_address}", liquidity_gt: "0",
                               id_gt: "{last_id}"}},
                      block: {{number: {block_number}}}) {{
                id owner liquidity
                tickLower {{ tickIdx }}
                tickUpper {{ tickIdx }}
            }}
        }}""")
        resp = await query_data_async(query)
        batch = resp["data"]["positions"]

        if not batch:
            break

        positions.extend(batch)
        last_id = batch[-1]["id"]

    return positions


def compute_lp_tvls(positions, pool_state, pool_address):
    """Compute per-LP TVL from raw position data and pool state.

    For each position, calculates token0 and token1 amounts using the
    formulas from the Uniswap V3 whitepaper (Section 6.2):
      - above current tick  → only token0
      - below current tick  → only token1
      - spanning curr. tick → both tokens

    Converts to USD using price_in_stablecoin from the RPC parquet data.
    price_in_stablecoin = price of non-stablecoin token in stablecoin units.
      If token0 is stablecoin: TVL = amount0 + amount1 × price_in_stablecoin
      If token1 is stablecoin: TVL = amount0 × price_in_stablecoin + amount1

    Multiple positions by the same owner are aggregated into a single LP row.
    Returns (lp_dataframe, raw_sum_positions_tvl).
    """
    if not positions:
        return None

    current_tick = pool_state["cross_tick"]
    d0 = pool_state["token0_decimals"]
    d1 = pool_state["token1_decimals"]
    price = pool_state["price_in_stablecoin"]

    # Convert sqrtPriceX96 to actual sqrt(price) with Decimal precision
    sqrt_price = float(Decimal(pool_state["sqrt_price_x96"]) / Decimal(2 ** 96))

    # Determine which token is the stablecoin (from pool config)
    stablecoin = POOLS[pool_address]["stablecoin"]
    stable_is_t0 = (pool_state["token0_symbol"] == stablecoin)

    # Build numpy arrays for vectorised token amount computation
    owners = [p["owner"] for p in positions]
    L = np.array([float(p["liquidity"]) for p in positions])
    tl = np.array([int(p["tickLower"]["tickIdx"]) for p in positions])
    tu = np.array([int(p["tickUpper"]["tickIdx"]) for p in positions])

    # sqrt(price) at position bounds — Uniswap V3 whitepaper eq. 6.4
    P_a = 1.0001 ** (tl / 2.0)
    P_b = 1.0001 ** (tu / 2.0)

    # Classify positions relative to current tick
    above = tl >= current_tick     # entirely above → only token0
    below = tu <= current_tick     # entirely below → only token1
    in_range = ~above & ~below     # straddles current tick → both tokens

    # Compute raw token amounts (divided by 10^decimals to get human units)
    amt0 = np.zeros(len(L))
    amt1 = np.zeros(len(L))

    amt0[above] = L[above] * (P_b[above] - P_a[above]) / (P_a[above] * P_b[above]) / 10 ** d0
    amt1[below] = L[below] * (P_b[below] - P_a[below]) / 10 ** d1
    amt0[in_range] = L[in_range] * (P_b[in_range] - sqrt_price) / (sqrt_price * P_b[in_range]) / 10 ** d0
    amt1[in_range] = L[in_range] * (sqrt_price - P_a[in_range]) / 10 ** d1

    # Convert to stablecoin-denominated value using price_in_stablecoin
    tvl = (amt0 + amt1 * price) if stable_is_t0 else (amt0 * price + amt1)

    # Aggregate by owner — one LP can have multiple positions in the same pool
    pos_df = pd.DataFrame({"owner": owners, "position_tvl_usd": tvl})
    lp_df = pos_df.groupby("owner").agg(
        num_positions=("owner", "count"),
        lp_tvl_usd=("position_tvl_usd", "sum"),
    ).reset_index()

    # Compute each LP's share (used for normalisation below)
    raw_total = lp_df["lp_tvl_usd"].sum()
    lp_df["tvl_share"] = lp_df["lp_tvl_usd"] / raw_total if raw_total > 0 else 0
    lp_df["pool_address"] = pool_address

    return lp_df, raw_total


# ---------------------------------------------------------------------------
# Per-timestamp processing pipeline
# ---------------------------------------------------------------------------

async def process_timestamp(datetime_str, tvl_dir, rpc_dir, output_dir,
                            pool_filter=None, experiment_pool=None,
                            hourly=False):
    """Process all (or selected) pools for one timestamp (daily or hourly).

    1. Loads pool states from the RPC parquet file
    2. Loads reported TVL from pools_daily or pools_hourly CSV
    3. For each pool, paginates through positions (subgraph) and computes
       per-LP TVL using the RPC-sourced pool state
    4. Normalises each LP's TVL so the pool sum matches reported TVL exactly
    5. Saves incrementally (skips already-processed pools)
    """
    month = datetime_str[5:7]

    # Locate the TVL reference file
    if hourly:
        tvl_path = os.path.join(tvl_dir, month, f"{datetime_str}.csv")
    else:
        # Daily mode: datetime_str is "YYYY-MM-DD"
        tvl_path = os.path.join(tvl_dir, month, f"{datetime_str}.csv")

    if not os.path.exists(tvl_path):
        log.warning(f"No TVL CSV for {datetime_str}: {tvl_path}")
        return

    # Load pool states from the RPC parquet file
    rpc_states = load_pool_states_from_rpc(datetime_str, rpc_dir)
    if not rpc_states:
        return

    tvl_df = pd.read_csv(tvl_path)

    # Build TVL reference for normalisation
    tvl_ref = dict(zip(tvl_df["pool_address"], tvl_df["totalValueLockedUSD"]))

    # Determine which pools to process (intersection of rpc + tvl + config)
    if experiment_pool:
        targets = [experiment_pool]
    elif pool_filter:
        targets = pool_filter
    else:
        targets = tvl_df["pool_address"].tolist()
    targets = [p for p in targets
               if p in POOLS and POOLS[p].get("active", True)
               and p in rpc_states and p in tvl_ref]

    # Incremental: skip already-processed pools
    out_dir_month = os.path.join(output_dir, month)
    os.makedirs(out_dir_month, exist_ok=True)
    out_path = os.path.join(out_dir_month, f"{datetime_str}.csv")

    existing = pd.read_csv(out_path) if os.path.exists(out_path) else None
    done = set(existing["pool_address"].unique()) if existing is not None else set()
    remaining = [p for p in targets if p not in done]

    if not remaining:
        log.info(f"{datetime_str}: all {len(done)} pools already processed")
        return

    if done:
        log.info(f"{datetime_str}: {len(done)} done, {len(remaining)} remaining")

    # All pools share the same block number within a single parquet file
    block = rpc_states[remaining[0]]["block_number"]
    log.info(f"{datetime_str}: {len(remaining)} pools at block {block} "
             f"(pool states from RPC parquet)")

    results = []
    for addr in tqdm(remaining, desc=datetime_str, leave=False):
        state = rpc_states[addr]

        # Fetch active positions from the subgraph at the historical block
        positions = await fetch_positions_at_block(addr, state["block_number"])
        out = compute_lp_tvls(positions, state, addr)

        if out is None:
            continue

        lp_df, raw_total = out

        # Get the pool's reported TVL (normalisation target)
        reported = tvl_ref[addr]

        # Normalise: scale each LP's TVL so the pool total matches reported TVL.
        # The raw total is lower than reported because pool TVL includes
        # uncollected trading fees (~40-50%) sitting in the contract.
        # Normalisation distributes fees proportionally to each LP's liquidity
        # share, preserving the relative distribution for Gini calculation.
        if raw_total > 0:
            lp_df["lp_tvl_usd"] = lp_df["tvl_share"] * reported

        coverage_pct = raw_total / reported * 100 if reported > 0 else 0.0

        # Add metadata for downstream Gini calculation and fee-tier analysis
        lp_df["trading_pair"] = state["trading_pair"]
        lp_df["fee_tier"] = state["fee_tier"]
        lp_df["stablecoin"] = POOLS[addr]["stablecoin"]
        lp_df["num_lps"] = len(lp_df)
        lp_df["pool_tvl_usd"] = reported
        lp_df["raw_positions_tvl_usd"] = raw_total
        lp_df["liquidity_coverage_pct"] = coverage_pct
        lp_df["date"] = datetime_str

        log.info(f"  {addr[:10]}... {len(lp_df):>4} LPs | "
                 f"pool=${reported:>14,.2f} | "
                 f"fee={state['fee_tier']:>5} | "
                 f"coverage={coverage_pct:.1f}%")

        results.append(lp_df)

    if results:
        new_df = pd.concat(results)
        combined = pd.concat([existing, new_df]) if existing is not None else new_df
        combined.to_csv(out_path, index=False)
        log.info(f"{datetime_str}: saved {combined['pool_address'].nunique()} pools, "
                 f"{len(combined)} LP rows")


async def main():
    parser = argparse.ArgumentParser(
        description="Collect per-LP position data for Gini coefficient analysis")

    parser.add_argument("--data_dir", type=str,
                        default="./../../../data",
                        help="Base data directory (contains filtered/ and pools/)")
    parser.add_argument("--start_date", type=str, default="2023-02-28")
    parser.add_argument("--end_date", type=str, default="2023-04-01")

    # Hourly mode: process every hour instead of daily midnight only
    parser.add_argument("--hourly", action="store_true",
                        help="Process hourly snapshots (default: daily at midnight)")

    # Experiment mode: run a single pool on a single timestamp for validation
    parser.add_argument("--experiment", action="store_true",
                        help="Single pool/timestamp test run for validation")
    parser.add_argument("--experiment_pool", type=str, default=None,
                        help="Pool address for experiment (default: USDC/WETH 500)")
    parser.add_argument("--experiment_date", type=str, default="2023-03-01",
                        help="Date for experiment mode (daily: YYYY-MM-DD, "
                             "hourly: YYYY-MM-DD HH-MM-SS)")

    args = parser.parse_args()

    rpc_dir = os.path.join(args.data_dir, "pools", "rpc")

    if args.hourly:
        tvl_dir = os.path.join(args.data_dir, "filtered", "pools_hourly")
        out_dir = os.path.join(args.data_dir, "filtered", "positions_hourly")
    else:
        tvl_dir = os.path.join(args.data_dir, "filtered", "pools_daily")
        out_dir = os.path.join(args.data_dir, "filtered", "positions_daily")

    if args.experiment:
        pool = args.experiment_pool or "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
        # For hourly experiment, default to first hour of the day
        exp_dt = args.experiment_date
        if args.hourly and len(exp_dt) == 10:
            exp_dt = f"{exp_dt} 00-00-00"
        log.info(f"EXPERIMENT MODE: pool={pool} datetime={exp_dt} "
                 f"hourly={args.hourly}")
        await process_timestamp(exp_dt, tvl_dir, rpc_dir, out_dir,
                                experiment_pool=pool, hourly=args.hourly)
        return

    # Full run: process all active pools across the date range
    active = [a for a, info in POOLS.items() if info["active"]]
    freq = "h" if args.hourly else "D"
    timestamps = pd.date_range(args.start_date, args.end_date, freq=freq)

    for ts in tqdm(timestamps, desc="Timestamps"):
        if args.hourly:
            dt_str = ts.strftime("%Y-%m-%d %H-%M-%S")
        else:
            dt_str = ts.strftime("%Y-%m-%d")
        await process_timestamp(dt_str, tvl_dir, rpc_dir, out_dir,
                                pool_filter=active, hourly=args.hourly)


if __name__ == "__main__":
    # Retry on transient failures (e.g. subgraph rate limits)
    for attempt in range(100):
        print(f"Attempt {attempt + 1}...")
        try:
            asyncio.run(main())
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
            print("Retrying...")
            continue
