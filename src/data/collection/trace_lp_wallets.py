"""
Trace where specific LP wallet addresses moved their liquidity on Uniswap v3
around the SVB / USDC depeg crisis (March 2023).

Queries the Uniswap v3 subgraph for:
  1. All positions held by each wallet (across ALL pools, not just our tracked set)
  2. Mint/burn transactions to see capital flows over time

Outputs a CSV with full position + transaction history for downstream plotting.

Usage:
    python trace_lp_wallets.py
"""

import asyncio
import json
import os
import sys
import time

import aiohttp
import pandas as pd
from dotenv import load_dotenv

# Load .env from the collection directory
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

SUBGRAPH_ID = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"

_API_KEYS = [
    os.getenv("UNISWAP_SUBGRAPH_API_KEY"),
]
_current_key_index = 0


def _get_url():
    return (f"https://gateway.thegraph.com/api/{_API_KEYS[_current_key_index]}"
            f"/subgraphs/id/{SUBGRAPH_ID}")


def _rotate_key():
    global _current_key_index
    if _current_key_index < len(_API_KEYS) - 1:
        _current_key_index += 1
        print(f"  [key rotation] -> key index {_current_key_index}")
        return True
    print("  [key rotation] All keys exhausted!")
    return False


async def _query(session, query_str, variables=None):
    """Execute a GraphQL query with automatic key rotation."""
    payload = {"query": query_str}
    if variables:
        payload["variables"] = variables

    while True:
        url = _get_url()
        async with session.post(url,
                                headers={"Content-Type": "application/json"},
                                data=json.dumps(payload)) as resp:
            result = await resp.json()

        if "data" in result:
            return result["data"]

        print(f"  [query error] {result}")
        if not _rotate_key():
            raise RuntimeError(f"All keys exhausted. Response: {result}")


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

# 1. All open positions for a wallet (at a specific block)
POSITIONS_BY_OWNER = """
{{
  positions(
    first: 1000
    where: {{owner: "{owner}", id_gt: "{last_id}"}}
    orderBy: id
    orderDirection: asc
    block: {{number: {block}}}
  ) {{
    id
    owner
    liquidity
    depositedToken0
    depositedToken1
    withdrawnToken0
    withdrawnToken1
    tickLower {{ tickIdx }}
    tickUpper {{ tickIdx }}
    pool {{
      id
      token0 {{ id symbol decimals }}
      token1 {{ id symbol decimals }}
      feeTier
      totalValueLockedUSD
    }}
  }}
}}
"""

# 2. Mints and Burns for a wallet across ALL pools in a time range
MINTS_BY_ORIGIN = """
{{
  mints(
    first: 1000
    orderBy: timestamp
    orderDirection: asc
    where: {{origin: "{origin}", timestamp_gte: "{ts_start}",
             timestamp_lte: "{ts_end}", id_gt: "{last_id}"}}
    subgraphError: allow
  ) {{
    id
    timestamp
    owner
    origin
    amount0
    amount1
    amountUSD
    tickLower
    tickUpper
    pool {{
      id
      token0 {{ symbol }}
      token1 {{ symbol }}
      feeTier
    }}
  }}
}}
"""

BURNS_BY_ORIGIN = """
{{
  burns(
    first: 1000
    orderBy: timestamp
    orderDirection: asc
    where: {{origin: "{origin}", timestamp_gte: "{ts_start}",
             timestamp_lte: "{ts_end}", id_gt: "{last_id}"}}
    subgraphError: allow
  ) {{
    id
    timestamp
    owner
    origin
    amount0
    amount1
    amountUSD
    tickLower
    tickUpper
    pool {{
      id
      token0 {{ symbol }}
      token1 {{ symbol }}
      feeTier
    }}
  }}
}}
"""


async def get_all_positions(session, owner, block):
    """Paginate through all positions for an owner at a block."""
    all_positions = []
    last_id = ""
    while True:
        q = POSITIONS_BY_OWNER.format(owner=owner, block=block,
                                       last_id=last_id)
        data = await _query(session, q)
        positions = data.get("positions", [])
        all_positions.extend(positions)
        if len(positions) < 1000:
            break
        last_id = positions[-1]["id"]
        await asyncio.sleep(0.3)
    return all_positions


async def get_mints_burns(session, origin, ts_start, ts_end):
    """Get all mints and burns for a wallet in a time range."""
    results = {"mints": [], "burns": []}

    for entity, template in [("mints", MINTS_BY_ORIGIN),
                              ("burns", BURNS_BY_ORIGIN)]:
        last_id = ""
        while True:
            q = template.format(origin=origin, ts_start=ts_start,
                                ts_end=ts_end, last_id=last_id)
            data = await _query(session, q)
            items = data.get(entity, [])
            results[entity].extend(items)
            if len(items) < 1000:
                break
            last_id = items[-1]["id"]
            await asyncio.sleep(0.3)

    return results


def flatten_position(pos, owner_label, snapshot_label):
    """Flatten a position dict into a row dict."""
    pool = pos["pool"]
    return {
        "owner": pos["owner"],
        "owner_label": owner_label,
        "snapshot": snapshot_label,
        "position_id": pos["id"],
        "liquidity": int(pos["liquidity"]),
        "deposited_token0": float(pos["depositedToken0"]),
        "deposited_token1": float(pos["depositedToken1"]),
        "withdrawn_token0": float(pos["withdrawnToken0"]),
        "withdrawn_token1": float(pos["withdrawnToken1"]),
        "tick_lower": int(pos["tickLower"]["tickIdx"]),
        "tick_upper": int(pos["tickUpper"]["tickIdx"]),
        "pool_address": pool["id"],
        "token0_symbol": pool["token0"]["symbol"],
        "token1_symbol": pool["token1"]["symbol"],
        "fee_tier": int(pool["feeTier"]),
        "pool_tvl_usd": float(pool["totalValueLockedUSD"]),
    }


def flatten_mint_burn(item, entity_type, owner_label):
    pool = item["pool"]
    return {
        "owner": item.get("owner", ""),
        "origin": item.get("origin", ""),
        "owner_label": owner_label,
        "type": entity_type,
        "timestamp": int(item["timestamp"]),
        "datetime": pd.Timestamp(int(item["timestamp"]), unit="s"),
        "amount0": float(item["amount0"]),
        "amount1": float(item["amount1"]),
        "amount_usd": float(item["amountUSD"]),
        "pool_address": pool["id"],
        "token0_symbol": pool["token0"]["symbol"],
        "token1_symbol": pool["token1"]["symbol"],
        "fee_tier": int(pool["feeTier"]),
        "trading_pair": (pool["token0"]["symbol"] + pool["token1"]["symbol"]
                         + pool["feeTier"]),
    }


# ---------------------------------------------------------------------------
# Target wallets & block numbers
# ---------------------------------------------------------------------------

# Key Ethereum block numbers (approximate, from etherscan):
# Mar 1 00:00 UTC  ~ 16730071
# Mar 10 00:00 UTC ~ 16795573
# Mar 11 06:00 UTC ~ 16798793  (just before Circle tweet)
# Mar 11 18:00 UTC ~ 16800409
# Mar 12 00:00 UTC ~ 16801847
# Mar 14 00:00 UTC ~ 16816310
# Mar 17 00:00 UTC ~ 16838132
# Mar 20 00:00 UTC ~ 16859867
# Mar 31 00:00 UTC ~ 16939634

BLOCKS = {
    "pre_crisis_mar10": 16795573,
    "during_depeg_mar11_18h": 16800409,
    "post_depeg_mar12": 16801847,
    "post_depeg_mar14": 16816310,
    "post_bankruptcy_mar20": 16859867,
    "end_of_month_mar31": 16939634,
}

# Pattern B wallets (depeg panic, Mar 11)
PATTERN_B_WALLETS = {
    "0x471c6a1f283d2b52ff332b9706ffa6ca4f261479": "USDC/USDM ($100M)",
    "0x1b04d574d4a3d57fb724848937a926aa21c59271": "USDC/XIDR ($1.2M)",
    "0x16c69edd1e2b977de44bc7e9cc7f97452d5334e4": "USDC/CGT ($1.1M)",
    "0xbcb27bc5106528dda8573dbfa7240254a1fcf7e5": "USDC/SAITO ($1.8M)",
    "0x4aec51064800e0b5f0160ef2e99e68760404cb98": "CNLT/USDC ($233K)",
    "0x69982e017acc0fde3d1542205089a8d3eafcd1b7": "USDC/RSS3 ($195K)",
    "0x786fdf0d8570c1637fcecdc1b06405dfe715492b": "TPRO/USDC ($79K)",
}

# Also track the top LPs from our universe for broader analysis
TOP_LP_WALLETS = {}  # Will be populated from position data


async def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                           "data", "filtered")
    os.makedirs(out_dir, exist_ok=True)

    all_wallets = {**PATTERN_B_WALLETS, **TOP_LP_WALLETS}

    async with aiohttp.ClientSession() as session:

        # ---- Part 1: Snapshot positions at key blocks ----
        print("=" * 60)
        print("Part 1: Querying positions at key block snapshots")
        print("=" * 60)

        position_rows = []
        for owner, label in all_wallets.items():
            print(f"\n--- {label} ({owner[:10]}...) ---")
            for snap_name, block in BLOCKS.items():
                positions = await get_all_positions(session, owner, block)
                active = [p for p in positions if int(p["liquidity"]) > 0]
                print(f"  {snap_name} (block {block}): "
                      f"{len(active)} active positions")
                for p in active:
                    position_rows.append(
                        flatten_position(p, label, snap_name))
                await asyncio.sleep(0.5)

        positions_df = pd.DataFrame(position_rows)
        pos_path = os.path.join(out_dir, "traced_lp_positions.csv")
        positions_df.to_csv(pos_path, index=False)
        print(f"\nSaved {len(positions_df)} position rows to {pos_path}")

        # ---- Part 2: Mint/Burn transactions (Feb 28 – Mar 31) ----
        print("\n" + "=" * 60)
        print("Part 2: Querying mint/burn transactions")
        print("=" * 60)

        ts_start = int(pd.Timestamp("2023-02-28").timestamp())
        ts_end = int(pd.Timestamp("2023-04-01").timestamp())

        tx_rows = []
        for owner, label in all_wallets.items():
            print(f"\n--- {label} ({owner[:10]}...) ---")
            mb = await get_mints_burns(session, owner, ts_start, ts_end)
            print(f"  Mints: {len(mb['mints'])}, Burns: {len(mb['burns'])}")
            for m in mb["mints"]:
                tx_rows.append(flatten_mint_burn(m, "mint", label))
            for b in mb["burns"]:
                tx_rows.append(flatten_mint_burn(b, "burn", label))
            await asyncio.sleep(0.5)

        tx_df = pd.DataFrame(tx_rows)
        tx_path = os.path.join(out_dir, "traced_lp_transactions.csv")
        tx_df.to_csv(tx_path, index=False)
        print(f"\nSaved {len(tx_df)} transaction rows to {tx_path}")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
