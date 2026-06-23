"""
Investigate available tick levels per pool per hour.

For each pool at each hourly snapshot, determines:
  - Total number of ticks
  - Ticks above/below the current tick (available depth for buy/sell)
  - Maximum MCI level computable (based on available ticks)
  - Whether MCI=0 at level 1 and why (insufficient ticks vs zero swap deltas)

Usage:
    python investigate_tick_levels.py
    python investigate_tick_levels.py --start_date 2023-02-28 --end_date 2023-03-01
"""

import argparse
import glob
import os
import sys
from collections import OrderedDict
from datetime import datetime
from decimal import Decimal, getcontext

import pandas as pd
import numpy as np

getcontext().prec = 50

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.modeling.analytics_ticks import mci as compute_mci
from src.data.processing.MCI_ticks import (
    prepare_data, find_files, POOL_CSV_COLS, BASE_TOKENS, LEVELS,
)


def analyze_tick_levels(tick_df, pool_df, date_str, levels=LEVELS):
    """For each pool, compute available tick depth and max computable level.

    Returns a DataFrame with one row per pool.
    """
    merged = prepare_data(tick_df, pool_df)
    if merged.empty:
        return pd.DataFrame()

    rows = []
    for pair in merged["trading_pair"].unique():
        sub = merged[merged["trading_pair"] == pair].reset_index(drop=True)
        first = sub.iloc[0]

        token0 = first["token0_symbol"]
        token1 = first["token1_symbol"]
        fee = int(first["fee_tier"])
        n_ticks = len(sub)

        # Find current tick position
        current_mask = sub["isCurrentTick"] == True
        if not current_mask.any():
            rows.append({
                "date": date_str, "trading_pair": pair,
                "token0": token0, "token1": token1, "fee": fee,
                "n_ticks": n_ticks, "ticks_below": 0, "ticks_above": 0,
                "max_level": 0, "reason_mci1_zero": "no_current_tick",
            })
            continue

        current_pos = current_mask.idxmax()
        ticks_below = current_pos
        ticks_above = n_ticks - current_pos - 1

        # Max level: level <= (n_ticks / 2) - 1
        max_level = int(n_ticks / 2) - 1
        # Also limited by ticks on each side (buy needs above, sell needs below)
        max_buy_level = ticks_above
        max_sell_level = ticks_below

        # Check which standard levels are available
        available_levels = [l for l in levels if l <= max_level]

        # Determine why MCI_1=0 if applicable
        reason = "ok"
        if 1 > max_level:
            reason = "insufficient_ticks"
        else:
            # Test actual MCI computation at level 1
            decimals0 = int(first["token0_decimals"])
            decimals1 = int(first["token1_decimals"])
            sqrt_px96 = int(first["sqrt_price_x96"])
            tick_sp = first["tick_spacing"]
            mid_price = Decimal(first["price_in_stablecoin"])
            flip = token1 not in BASE_TOKENS

            buy_mci = compute_mci(
                False, mid_price, sub, tick_sp,
                decimals0, decimals1, Decimal(sqrt_px96),
                pair, date_str, flip, level=1,
            )
            sell_mci = compute_mci(
                True, mid_price, sub, tick_sp,
                decimals0, decimals1, Decimal(sqrt_px96),
                pair, date_str, flip, level=1,
            )
            if buy_mci == 0 and sell_mci == 0:
                reason = "zero_swap_deltas"
            elif buy_mci == 0:
                reason = "zero_buy_delta"
            elif sell_mci == 0:
                reason = "zero_sell_delta"

        stablecoin = token0 if (token1 not in BASE_TOKENS) else token1
        tvl = float(first.get("totalValueLockedUSD", 0))

        rows.append({
            "date": date_str,
            "trading_pair": pair,
            "stablecoin": stablecoin,
            "token0": token0,
            "token1": token1,
            "fee": fee,
            "tvl": tvl,
            "n_ticks": n_ticks,
            "current_pos": current_pos,
            "ticks_below": ticks_below,
            "ticks_above": ticks_above,
            "max_level_tick_check": max_level,
            "max_buy_level": max_buy_level,
            "max_sell_level": max_sell_level,
            "available_levels": str(available_levels),
            "n_available_levels": len(available_levels),
            "reason_mci1_zero": reason,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Investigate available tick levels per pool per hour")
    parser.add_argument("--ticks_dir", default="./../../../data/pools/rpc")
    parser.add_argument("--pools_dir", default=None)
    parser.add_argument("--output_dir", default="./../../../data/filtered")
    parser.add_argument("--start_date", default="2023-02-28")
    parser.add_argument("--end_date", default="2023-04-01")
    parser.add_argument("--sample_hours", type=int, default=0,
                        help="If >0, sample this many hours instead of all")
    args = parser.parse_args()

    if args.pools_dir is None:
        args.pools_dir = os.path.join(args.output_dir, "pools_hourly")

    tick_files = find_files(args.ticks_dir, args.start_date, args.end_date,
                            ext=".parquet")
    pool_files = find_files(args.pools_dir, args.start_date, args.end_date,
                            ext=".csv")

    print(f"Found {len(tick_files)} tick files, {len(pool_files)} pool files")

    # Match tick files to pool files by date
    all_results = []
    dates = list(tick_files.keys())
    if args.sample_hours > 0:
        step = max(1, len(dates) // args.sample_hours)
        dates = dates[::step][:args.sample_hours]
        print(f"Sampling {len(dates)} hours")

    for dt in dates:
        tick_path = tick_files[dt]
        # Find closest pool file
        pool_dt = min(pool_files.keys(), key=lambda d: abs((d - dt).total_seconds()))
        pool_path = pool_files[pool_dt]

        date_str = dt.strftime("%Y-%m-%d %H-%M-%S")
        print(f"Processing {date_str}...")

        tick_df = pd.read_parquet(tick_path, engine="fastparquet")
        pool_df = pd.read_csv(pool_path, engine="c", usecols=POOL_CSV_COLS)

        result = analyze_tick_levels(tick_df, pool_df, date_str)
        if not result.empty:
            all_results.append(result)

    if not all_results:
        print("No results!")
        return

    df = pd.concat(all_results, ignore_index=True)

    # Save
    out_path = os.path.join(args.output_dir, "tick_level_availability.parquet")
    df.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Summary statistics
    print(f"\n{'='*60}")
    print(f"SUMMARY across {df['date'].nunique()} hours, {df['trading_pair'].nunique()} pools")
    print(f"{'='*60}")

    print(f"\nTotal pool-hour observations: {len(df)}")

    # Reason for MCI_1=0
    reason_counts = df["reason_mci1_zero"].value_counts()
    print(f"\nMCI level 1 status:")
    for reason, count in reason_counts.items():
        pct = count / len(df) * 100
        print(f"  {reason}: {count} ({pct:.1f}%)")

    # Available levels distribution
    print(f"\nMax computable level (tick check) distribution:")
    for threshold in [0, 1, 2, 5, 10, 15, 20]:
        n = (df["max_level_tick_check"] >= threshold).sum()
        pct = n / len(df) * 100
        print(f"  >= {threshold}: {n} pool-hours ({pct:.1f}%)")

    # Per-pool: what's the minimum max_level across all hours?
    pool_min = df.groupby("trading_pair")["max_level_tick_check"].min()
    print(f"\nPools by minimum max_level across all hours:")
    for threshold in [1, 5, 10, 15, 20]:
        n = (pool_min >= threshold).sum()
        print(f"  Always >= {threshold}: {n} pools")

    # Pools that always have MCI_1 computable (reason=ok)
    pool_ok = df.groupby("trading_pair")["reason_mci1_zero"].apply(
        lambda g: (g == "ok").all()
    )
    print(f"\nPools with MCI_1 always computable: {pool_ok.sum()}")
    print(f"Pools with MCI_1 sometimes zero: {(~pool_ok).sum()}")


if __name__ == "__main__":
    main()
