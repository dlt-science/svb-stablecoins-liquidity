"""
Discovery script: find top Uniswap V3 pools containing USDC or USDT,
sorted by TVL.

Queries The Graph's Uniswap V3 subgraph at a historical block (~Feb 28, 2023).
Writes selected pools to a JSON file loadable by pools_config.py and saves
a CSV for analysis.
"""

import asyncio
import json
import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.data.collection.etherscan import get_block_by_timestamp
from src.data.collection.uniswapv3_async import query_data_async
from src.data.collection.pools_config import (
    TOKEN_ADDRESSES, STABLECOINS, USDC_BACKED_TOKENS, is_usdc_backed_usdt_pair,
)

from gql import gql


# Minimum TVL (in USD) for a pool to be included in the selection
# MIN_TVL_USD = 1_000_000
MIN_TVL_USD = 100_000

# Minimum daily volume (in USD) on the target date for a pool to be included.
# Note: the subgraph's pool.volumeUSD is cumulative (all-time) and NOT useful
# for filtering active pools.  We query poolDayDatas for the target date to
# get the actual daily volume.
# MIN_VOLUME_USD = 500_000
MIN_VOLUME_USD = 0

# Pairs to exclude from discovery (DAI/USDC pairs are tracked separately)
EXCLUDE_USDC_PAIRS = {"DAI/USDC", "USDC/DAI"}

# Default JSON output path (relative to this file → data/discovery/)
_DEFAULT_JSON_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "discovery", "selected_pools.json"
)


# ---------------------------------------------------------------------------
# Subgraph queries
# ---------------------------------------------------------------------------

async def fetch_pools_with_token(token_address, token_position, block_number, skip=0):
    """Fetch pools where a token appears as token0 or token1.

    Args:
        token_address: Token contract address (lowercase)
        token_position: "token0" or "token1"
        block_number: Historical block to query at
        skip: Pagination offset

    Returns:
        List of pool dicts from the subgraph
    """
    all_pools = []

    while True:
        query = gql(f"""
        {{
          pools(
            first: 1000
            skip: {skip}
            where: {{{token_position}: "{token_address}"}}
            orderBy: totalValueLockedUSD
            orderDirection: desc
            block: {{number: {block_number}}}
          ) {{
            id
            token0 {{
              id
              symbol
              decimals
            }}
            token1 {{
              id
              symbol
              decimals
            }}
            feeTier
            totalValueLockedUSD
            totalValueLockedToken0
            totalValueLockedToken1
            volumeUSD
            liquidity
          }}
        }}
        """)

        response = await query_data_async(query)
        pools = response["data"]["pools"]

        if not pools:
            break

        all_pools.extend(pools)
        skip += 1000

        # Stop if TVL drops below threshold (avoid fetching dust pools)
        last_tvl = float(pools[-1]["totalValueLockedUSD"])
        if last_tvl < 10_000:
            break

    return all_pools


async def discover_pools(block_number):
    """Discover all USDC and USDT pools at a given block.

    Runs 4 queries (USDC as token0, USDC as token1, USDT as token0, USDT as token1)
    and deduplicates results.
    """
    usdc = TOKEN_ADDRESSES["USDC"]
    usdt = TOKEN_ADDRESSES["USDT"]

    results = await asyncio.gather(
        fetch_pools_with_token(usdc, "token0", block_number),
        fetch_pools_with_token(usdc, "token1", block_number),
        fetch_pools_with_token(usdt, "token0", block_number),
        fetch_pools_with_token(usdt, "token1", block_number),
    )

    # Deduplicate by pool id
    seen = set()
    all_pools = []
    for pool_list in results:
        for pool in pool_list:
            if pool["id"] not in seen:
                seen.add(pool["id"])
                all_pools.append(pool)

    return all_pools


async def fetch_daily_volumes(date_unix):
    """Fetch daily volume for all pools on a specific date.

    Queries poolDayDatas from the Uniswap V3 subgraph.  The subgraph only
    creates a poolDayData entry for days with trading activity, so pools
    absent from the result had zero volume on that date.

    Returns:
        dict of {pool_address: daily_volume_usd}
    """
    volumes = {}
    skip = 0

    while True:
        query = gql(f"""
        {{
          poolDayDatas(
            where: {{date: {date_unix}}}
            first: 1000
            skip: {skip}
            orderBy: volumeUSD
            orderDirection: desc
          ) {{
            pool {{ id }}
            volumeUSD
          }}
        }}
        """)
        response = await query_data_async(query)
        entries = response["data"]["poolDayDatas"]

        if not entries:
            break

        for entry in entries:
            volumes[entry["pool"]["id"]] = float(entry["volumeUSD"])

        skip += 1000

        # Stop pagination once volume drops below threshold
        if float(entries[-1]["volumeUSD"]) < 1000:
            break

    return volumes


# ---------------------------------------------------------------------------
# Processing helpers
# ---------------------------------------------------------------------------

def _assign_stablecoin(token0_sym, token1_sym):
    """Determine which stablecoin a pool belongs to."""
    for s in STABLECOINS:
        if token0_sym == s or token1_sym == s:
            return s
    return None



def _raw_pool_to_row(pool):
    """Convert a single raw subgraph pool dict to a flat row dict."""
    token0_sym = pool["token0"]["symbol"]
    token1_sym = pool["token1"]["symbol"]
    return {
        "address": pool["id"],
        "pair": f"{token0_sym}/{token1_sym}",
        "fee_tier": int(pool["feeTier"]),
        "tvl_usd": float(pool["totalValueLockedUSD"]),
        "cumulative_volume_usd": float(pool["volumeUSD"]),
        "stablecoin": _assign_stablecoin(token0_sym, token1_sym),
        "token0": token0_sym,
        "token1": token1_sym,
    }


def process_pools(raw_pools):
    """Convert raw subgraph data to a DataFrame, filter, and sort by TVL."""
    df = pd.DataFrame([_raw_pool_to_row(p) for p in raw_pools])

    # Exclude DAI/USDC pairs (tracked separately)
    df = df[~df["pair"].isin(EXCLUDE_USDC_PAIRS)]

    # Exclude USDT pools whose counterpart token is backed by USDC
    usdc_backed_mask = df.apply(
        lambda r: is_usdc_backed_usdt_pair(r["token0"], r["token1"], r["stablecoin"]),
        axis=1,
    )
    df = df[~usdc_backed_mask]

    # Filter out near-zero TVL
    df = df[df["tvl_usd"] > 1000]

    df = df.sort_values("tvl_usd", ascending=False).reset_index(drop=True)

    # --- Cumulative TVL percentage ---
    # (Commented out: replaced by individual TVL threshold — see select_pools_by_tvl)
    # for sc in STABLECOINS:
    #     mask = df["stablecoin"] == sc
    #     sc_total = df.loc[mask, "tvl_usd"].sum()
    #     df.loc[mask, "cumulative_tvl"] = df.loc[mask, "tvl_usd"].cumsum()
    #     df.loc[mask, "cumulative_pct"] = (
    #         df.loc[mask, "cumulative_tvl"] / sc_total * 100 if sc_total > 0 else 0
    #     )

    total_tvl = df["tvl_usd"].sum()
    return df, total_tvl


# ---------------------------------------------------------------------------
# Pool selection
# ---------------------------------------------------------------------------

def select_pools(df, min_tvl_usd=MIN_TVL_USD, min_volume_usd=MIN_VOLUME_USD):
    """Select pools meeting both TVL and volume thresholds."""
    mask = (df["tvl_usd"] >= min_tvl_usd) & (df["volume_usd"] >= min_volume_usd)
    return df[mask].copy().reset_index(drop=True)


# --- Cumulative-coverage selection (commented out: replaced by TVL threshold) ---
# def select_balanced_pools(df, coverage_pct=90):
#     """Select pools for each stablecoin independently, then balance counts.
#
#     For each stablecoin, pick the top pools by TVL that cover *coverage_pct*%
#     of that stablecoin's total TVL.  Then, if one stablecoin has fewer pools
#     than the other, add more pools from the smaller set (by TVL rank) until
#     the counts are equal.
#
#     Returns a combined DataFrame of selected pools (may contain duplicates
#     for USDC/USDT pools — these are deduplicated).
#     """
#     selected = {}
#     for sc in STABLECOINS:
#         sc_df = df[df["stablecoin"] == sc].copy()
#         # Pools needed for coverage threshold
#         within_coverage = sc_df[sc_df["cumulative_pct"] <= coverage_pct + 2]  # slight buffer
#         selected[sc] = within_coverage
#
#     # Balance: ensure both stablecoins have roughly equal pool counts
#     max_count = max(len(v) for v in selected.values())
#     for sc in STABLECOINS:
#         current = len(selected[sc])
#         if current < max_count:
#             sc_df = df[df["stablecoin"] == sc]
#             # Add more pools beyond the coverage threshold
#             selected[sc] = sc_df.head(max_count)
#
#     # Combine and deduplicate by address
#     combined = pd.concat(selected.values()).drop_duplicates(subset="address")
#     combined = combined.sort_values("tvl_usd", ascending=False).reset_index(drop=True)
#
#     return combined, selected


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def pools_df_to_config_dict(df):
    """Convert a DataFrame of selected pools to the dict format used by pools_config.

    Returns:
        dict of {address: {pair, fee_tier, stablecoin, active}}
    """
    return {
        row["address"]: {
            "pair": row["pair"],
            "fee_tier": row["fee_tier"],
            "stablecoin": row["stablecoin"],
            "active": True,
        }
        for _, row in df.iterrows()
    }


def write_pools_json(pools_dict, output_path):
    """Write the pools config dict to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(pools_dict, f, indent=4)
    print(f"Wrote {len(pools_dict)} pools to {output_path}")


def print_summary(df, selected_df, total_tvl):
    """Print a concise discovery summary to the console."""
    print(f"\n{'='*80}")
    print("UNISWAP V3 POOL DISCOVERY — USDC & USDT")
    print(f"Total TVL across all discovered pools: ${total_tvl:,.0f}")
    print(f"Selection threshold: TVL >= ${MIN_TVL_USD:,.0f}, Volume >= ${MIN_VOLUME_USD:,.0f}")
    print(f"{'='*80}\n")

    for sc in STABLECOINS:
        all_sc = df[df["stablecoin"] == sc]
        sel_sc = selected_df[selected_df["stablecoin"] == sc]
        print(f"  {sc}: {len(sel_sc)} / {len(all_sc)} pools selected "
              f"(TVL ${sel_sc['tvl_usd'].sum():,.0f})")
    print()

    # Per-stablecoin tables
    for sc in STABLECOINS:
        sel_addrs = set(selected_df["address"])
        sc_df = df[df["stablecoin"] == sc].head(30).reset_index(drop=True)

        print(f"--- {sc} pools (top 30) ---")
        print(f"{'#':<4} {'Sel':<4} {'Pair':<22} {'Fee':<8} {'TVL ($)':<18} {'Address'}")
        print("-" * 90)

        for i, row in sc_df.iterrows():
            sel = " +" if row["address"] in sel_addrs else "  "
            print(f"{i+1:<4} {sel:<4} {row['pair']:<22} {row['fee_tier']:<8} "
                  f"${row['tvl_usd']:>14,.0f}  {row['address']}")
        print()

    # Report excluded USDT pairs
    print("--- USDT pairs excluded (USDC-backed counterpart) ---")
    for token, reason in sorted(USDC_BACKED_TOKENS.items()):
        print(f"  {token}: {reason}")
    print()


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Discover top USDC/USDT pools on Uniswap V3")
    parser.add_argument("--date", type=str, default="2023-02-28",
                        help="Historical date to query (YYYY-MM-DD)")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Output CSV path (default: data/discovery/top_pools.csv)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Output JSON path (default: data/discovery/selected_pools.json)")
    parser.add_argument("--min-tvl", type=float, default=MIN_TVL_USD,
                        help=f"Minimum TVL in USD (default: {MIN_TVL_USD:,.0f})")
    parser.add_argument("--min-volume", type=float, default=MIN_VOLUME_USD,
                        help=f"Minimum volume in USD (default: {MIN_VOLUME_USD:,.0f})")
    args = parser.parse_args()

    # Get block number for the target date
    target_date = pd.Timestamp(args.date)
    print(f"Getting block number for {args.date}...")
    block_number = get_block_by_timestamp(target_date)
    print(f"Block number: {block_number}")

    # Discover pools
    print("Querying subgraph for USDC and USDT pools...")
    raw_pools = await discover_pools(block_number)
    print(f"Found {len(raw_pools)} unique pools (before filtering)")

    # Process and filter
    df, total_tvl = process_pools(raw_pools)
    print(f"After filtering: {len(df)} pools")

    # Fetch daily volumes for the target date (NOT cumulative all-time volume)
    date_unix = int(target_date.timestamp())
    print(f"Fetching daily volumes for {args.date}...")
    daily_volumes = await fetch_daily_volumes(date_unix)
    df["volume_usd"] = df["address"].map(daily_volumes).fillna(0.0)
    print(f"Pools with daily volume > 0: {(df['volume_usd'] > 0).sum()}")

    # Select pools meeting both TVL and daily volume thresholds
    selected_df = select_pools(df, min_tvl_usd=args.min_tvl, min_volume_usd=args.min_volume)

    # Print summary
    print_summary(df, selected_df, total_tvl)

    # --- Save CSV ---
    csv_path = args.output_csv or os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "data", "discovery", "top_pools.csv"
    )
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved full results to {csv_path}")

    # --- Save JSON (loadable by pools_config.py) ---
    json_path = args.output_json or _DEFAULT_JSON_PATH
    pools_dict = pools_df_to_config_dict(selected_df)
    write_pools_json(pools_dict, json_path)


if __name__ == "__main__":
    asyncio.run(main())
