"""Collect Uniswap V2 pool TVL data for March 2023.

Cross-exchange validation: compares USDC-paired and USDT-paired Uniswap V2 pool
TVL dynamics with Uniswap V3 findings during the SVB banking crisis.

Uniswap V2 implements the classic constant-product invariant (xy = k), which is
architecturally distinct from both V3's concentrated-liquidity tick mechanism and
Curve's StableSwap invariant — providing a third independent protocol for validation.

Data sources (in priority order):
  1. DeFi Llama yields chart API  (per-pool historical TVL)
  2. TheGraph Uniswap V2 subgraph  (fallback, pairDayDatas)

Output: data/filtered/uniswap_v2_march2023.parquet

Key events:
  - Treatment 1: 2023-03-11 03:11 UTC (Circle's tweet about USDC reserve at SVB)
  - Treatment 2: 2023-03-12 22:24 UTC (Fed/FDIC/Treasury Joint Statement)
"""

from __future__ import annotations

import datetime
import logging
import time
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# March 2023 date range
MARCH_START = datetime.date(2023, 3, 1)
MARCH_END   = datetime.date(2023, 3, 31)
MARCH_START_TS = 1_677_628_800   # 2023-03-01 00:00:00 UTC
MARCH_END_TS   = 1_680_307_199   # 2023-03-31 23:59:59 UTC

# Output path
OUTPUT_PATH = Path(__file__).parents[3] / "data" / "filtered" / "uniswap_v2_march2023.parquet"

# TheGraph endpoint for Uniswap V2
THEGRAPH_URL = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2"

# Major Uniswap V2 pool addresses (Ethereum mainnet, lowercase)
POOLS = [
    {
        "address":     "0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc",
        "pair_symbol": "USDC/WETH",
        "token0":      "USDC",
        "token1":      "WETH",
        "group":       "usdc",
    },
    {
        "address":     "0x0d4a11d5eeaac28ec3f61d100daf4d40471f1852",
        "pair_symbol": "USDT/WETH",
        "token0":      "USDT",
        "token1":      "WETH",
        "group":       "usdt",
    },
    {
        "address":     "0xae461ca67b15dc8dc81ce7615e0320da1a9ab8d5",
        "pair_symbol": "USDC/DAI",
        "token0":      "USDC",
        "token1":      "DAI",
        "group":       "usdc",
    },
    {
        "address":     "0x004375dff511095cc5a197a54140a24efef3a416",
        "pair_symbol": "USDC/WBTC",
        "token0":      "USDC",
        "token1":      "WBTC",
        "group":       "usdc",
    },
    # USDT/DAI V2 — verified address
    {
        "address":     "0x6d3f7d4e5db2ad6d82c2db3e54d63e77e13efa9a",
        "pair_symbol": "USDT/DAI",
        "token0":      "USDT",
        "token1":      "DAI",
        "group":       "usdt",
    },
    # USDT/WBTC V2
    {
        "address":     "0x0de845955493f88f4fff73bca3b9f0c717266e2e",
        "pair_symbol": "USDT/WBTC",
        "token0":      "USDT",
        "token1":      "WBTC",
        "group":       "usdt",
    },
]

# DeFi Llama pool IDs for Uniswap V2 pools (where known)
# These are fetched dynamically; hard-code known ones to reduce API calls.
DEFILLAMA_KNOWN_IDS: dict[str, str] = {}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source 1: DeFi Llama — pool list + historical chart
# ---------------------------------------------------------------------------

def _retry_get(url: str, *, retries: int = 4, backoff: float = 2.0,
               timeout: int = 30) -> requests.Response:
    """GET with exponential-backoff retries."""
    delay = 1.0
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp
            log.warning("HTTP %d for %s (attempt %d/%d)",
                        resp.status_code, url, attempt, retries)
        except requests.exceptions.RequestException as exc:
            log.warning("Request error: %s (attempt %d/%d)", exc, attempt, retries)
        if attempt < retries:
            time.sleep(delay)
            delay *= backoff
    raise RuntimeError(f"Failed to GET {url} after {retries} attempts")


def fetch_defillama_pool_ids() -> dict[str, str]:
    """Return a mapping of pool_address (lowercase) -> DeFi Llama pool ID
    for Uniswap V2 pools on Ethereum.

    Fetches the DeFi Llama /pools endpoint and filters for
    project == "uniswap-v2" and chain == "Ethereum".
    """
    log.info("Fetching DeFi Llama pool list …")
    resp = _retry_get("https://yields.llama.fi/pools", timeout=60)
    pools_data = resp.json().get("data", [])

    pool_addr_set = {p["address"] for p in POOLS}
    id_map: dict[str, str] = {}

    for entry in pools_data:
        if entry.get("project", "").lower() != "uniswap-v2":
            continue
        if entry.get("chain", "").lower() != "ethereum":
            continue
        # DeFi Llama may store pool address under "pool" or "poolMeta"
        # Check symbol and try to match against our known pairs
        sym = entry.get("symbol", "").upper().replace("-", "/")
        pool_id = entry.get("pool", "")

        # Try to match by symbol
        for p in POOLS:
            sym_variants = [
                p["pair_symbol"].upper(),
                p["pair_symbol"].upper().replace("/", "-"),
                # reversed
                "/".join(reversed(p["pair_symbol"].split("/"))),
                "-".join(reversed(p["pair_symbol"].split("/"))),
            ]
            if sym in sym_variants and pool_id:
                addr = p["address"]
                if addr not in id_map:
                    id_map[addr] = pool_id
                    log.info("  Matched %s → DeFi Llama ID %s", p["pair_symbol"], pool_id)
                break

    log.info("DeFi Llama: matched %d/%d pools", len(id_map), len(POOLS))
    return id_map


def fetch_defillama_pool_tvl(pool_id: str) -> dict[str, float]:
    """Fetch daily TVL for a single DeFi Llama pool ID.

    Returns dict mapping date string (YYYY-MM-DD) -> TVL in USD.
    Uses /chart/ endpoint (not /chartV2/ which returns 404 for many pools).
    """
    url = f"https://yields.llama.fi/chart/{pool_id}"
    try:
        resp = _retry_get(url, retries=3)
    except RuntimeError as exc:
        log.warning("chart failed for %s: %s", pool_id, exc)
        return {}

    records = resp.json().get("data", [])
    result: dict[str, float] = {}
    for r in records:
        ts_str = r.get("timestamp", "")
        date_str = ts_str[:10]
        if not date_str:
            continue
        try:
            d = datetime.date.fromisoformat(date_str)
        except ValueError:
            continue
        if MARCH_START <= d <= MARCH_END:
            tvl = r.get("tvlUsd", 0.0)
            if tvl and tvl > 0:
                result[date_str] = float(tvl)

    return result


def fetch_all_via_defillama(id_map: dict[str, str]) -> dict[str, dict[str, float]]:
    """Fetch TVL for all matched pools from DeFi Llama.

    Returns dict mapping pool_address -> {date_str -> tvl_usd}.
    """
    results: dict[str, dict[str, float]] = {}
    for pool in POOLS:
        addr = pool["address"]
        pool_id = id_map.get(addr) or DEFILLAMA_KNOWN_IDS.get(addr)
        if not pool_id:
            log.info("No DeFi Llama ID for %s (%s) — will use TheGraph",
                     pool["pair_symbol"], addr)
            continue
        log.info("Fetching DeFi Llama TVL for %s (ID: %s) …", pool["pair_symbol"], pool_id)
        tvl_map = fetch_defillama_pool_tvl(pool_id)
        if tvl_map:
            log.info("  Got %d March 2023 records for %s", len(tvl_map), pool["pair_symbol"])
            results[addr] = tvl_map
        else:
            log.info("  No data — will fall back to TheGraph for %s", pool["pair_symbol"])
        time.sleep(0.4)

    return results


# ---------------------------------------------------------------------------
# Source 2: TheGraph Uniswap V2 subgraph
# ---------------------------------------------------------------------------

_THEGRAPH_QUERY = """
{{
  pairDayDatas(
    first: 1000
    orderBy: date
    orderDirection: asc
    where: {{
      pairAddress_in: {addr_list}
      date_gte: {start_ts}
      date_lte: {end_ts}
    }}
  ) {{
    id
    date
    pairAddress
    token0 {{ symbol }}
    token1 {{ symbol }}
    reserveUSD
    dailyVolumeUSD
  }}
}}
"""


def _thegraph_query(addr_list: list[str]) -> list[dict]:
    """Execute a pairDayDatas query against the TheGraph Uniswap V2 subgraph.

    Returns a list of raw pairDayData objects.
    """
    # TheGraph requires addresses in checksum format in the query string,
    # but GraphQL string matching is case-sensitive — pass lowercase as stored.
    addr_json = "[" + ", ".join(f'"{a}"' for a in addr_list) + "]"
    query = _THEGRAPH_QUERY.format(
        addr_list=addr_json,
        start_ts=MARCH_START_TS,
        end_ts=MARCH_END_TS,
    )

    delay = 1.0
    for attempt in range(1, 5):
        try:
            resp = requests.post(
                THEGRAPH_URL,
                json={"query": query},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
            if "errors" in result:
                log.warning("TheGraph errors (attempt %d): %s", attempt, result["errors"])
                time.sleep(delay)
                delay *= 2
                continue
            return result.get("data", {}).get("pairDayDatas", [])
        except Exception as exc:
            log.warning("TheGraph request failed (attempt %d): %s", attempt, exc)
            time.sleep(delay)
            delay *= 2

    log.error("TheGraph query failed after all retries")
    return []


def fetch_via_thegraph(missing_addrs: list[str]) -> dict[str, dict[str, float]]:
    """Fetch pairDayData from TheGraph for pools not covered by DeFi Llama.

    Returns dict mapping pool_address -> {date_str -> tvl_usd}.
    """
    if not missing_addrs:
        return {}

    log.info("Fetching TheGraph Uniswap V2 data for %d pools: %s",
             len(missing_addrs), missing_addrs)

    records = _thegraph_query(missing_addrs)
    log.info("TheGraph returned %d pairDayData records", len(records))

    results: dict[str, dict[str, float]] = {}
    for rec in records:
        addr = rec.get("pairAddress", "").lower()
        ts = rec.get("date", 0)
        if not ts:
            continue
        d = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).date()
        if not (MARCH_START <= d <= MARCH_END):
            continue
        date_str = d.isoformat()
        reserve_usd = float(rec.get("reserveUSD", 0.0) or 0.0)
        if reserve_usd <= 0:
            continue
        if addr not in results:
            results[addr] = {}
        results[addr][date_str] = reserve_usd

    for addr, tmap in results.items():
        log.info("  TheGraph: %s — %d records", addr, len(tmap))

    return results


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_dataset() -> pd.DataFrame:
    """Fetch data from DeFi Llama (primary) and TheGraph (fallback),
    then assemble a tidy daily DataFrame.
    """

    # Step 1: Try DeFi Llama
    id_map = fetch_defillama_pool_ids()
    dl_results = fetch_all_via_defillama(id_map)

    # Step 2: Identify which pools still need data from TheGraph
    missing = [p["address"] for p in POOLS if p["address"] not in dl_results]
    if missing:
        log.info("Falling back to TheGraph for %d pools", len(missing))
        tg_results = fetch_via_thegraph(missing)
    else:
        tg_results = {}

    # Step 3: Merge
    all_tvl: dict[str, dict[str, float]] = {**dl_results, **tg_results}

    # Step 4: Build tidy DataFrame
    rows = []
    dates = []
    d = MARCH_START
    while d <= MARCH_END:
        dates.append(d.isoformat())
        d += datetime.timedelta(days=1)

    for pool in POOLS:
        addr = pool["address"]
        tvl_map = all_tvl.get(addr, {})
        if not tvl_map:
            log.warning("No TVL data for %s (%s)", pool["pair_symbol"], addr)
        for date_str in dates:
            tvl = tvl_map.get(date_str)
            if tvl is None or tvl <= 0:
                continue
            rows.append({
                "date":        pd.Timestamp(date_str, tz="UTC"),
                "pool_address": addr,
                "pair_symbol":  pool["pair_symbol"],
                "token0":       pool["token0"],
                "token1":       pool["token1"],
                "tvl_usd":      tvl,
                "group":        pool["group"],
            })

    df = pd.DataFrame(rows)
    if df.empty:
        log.error("No data collected — returning empty DataFrame")
        return df

    df = df.sort_values(["pool_address", "date"]).reset_index(drop=True)
    log.info("Dataset built: %d rows × %d columns (%d pools with data)",
             len(df), len(df.columns), df["pool_address"].nunique())
    return df


def print_diagnostics(df: pd.DataFrame) -> None:
    """Print summary statistics and key event comparisons."""
    print()
    print("=" * 70)
    print("UNISWAP V2 MARCH 2023 — DIAGNOSTIC SUMMARY")
    print("=" * 70)
    if df.empty:
        print("  ERROR: No data collected.")
        print("=" * 70)
        return

    print(f"Rows collected : {len(df)}")
    print(f"Pools with data: {df['pool_address'].nunique()}")
    print(f"Date range     : {df['date'].min().date()} to {df['date'].max().date()}")
    print()

    # Per-pool coverage
    print("Per-pool record counts:")
    for pool in POOLS:
        n = len(df[df["pool_address"] == pool["address"]])
        print(f"  {pool['pair_symbol']:<20} ({pool['group']})  {n:>3} days")

    print()
    print("TVL around Treatment 1 (Circle tweet: 2023-03-11 03:11 UTC):")
    for grp in ["usdc", "usdt"]:
        sub = df[df["group"] == grp]
        for d_str in ["2023-03-10", "2023-03-11", "2023-03-12", "2023-03-13"]:
            day_tvl = sub[sub["date"].dt.date == pd.Timestamp(d_str).date()]["tvl_usd"].sum()
            label = ""
            if d_str == "2023-03-10":
                label = " ← pre-treatment"
            elif d_str == "2023-03-11":
                label = " ← TREATMENT 1"
            elif d_str == "2023-03-12":
                label = " ← peak crisis"
            elif d_str == "2023-03-13":
                label = " ← after Joint Statement"
            print(f"  {grp.upper():4s}  {d_str}  ${day_tvl/1e6:>8.1f}M{label}")

    print()
    print("Aggregate TVL changes (USDC-paired pools):")
    usdc_df = df[df["group"] == "usdc"]
    t10 = usdc_df[usdc_df["date"].dt.date == pd.Timestamp("2023-03-10").date()]["tvl_usd"].sum()
    t11 = usdc_df[usdc_df["date"].dt.date == pd.Timestamp("2023-03-11").date()]["tvl_usd"].sum()
    t12 = usdc_df[usdc_df["date"].dt.date == pd.Timestamp("2023-03-12").date()]["tvl_usd"].sum()
    t13 = usdc_df[usdc_df["date"].dt.date == pd.Timestamp("2023-03-13").date()]["tvl_usd"].sum()

    if t10 and t11:
        print(f"  Aggregate USDC V2 TVL Mar 10→11:  ${t10/1e6:.1f}M → ${t11/1e6:.1f}M  ({(t11-t10)/t10*100:+.1f}%)")
    if t11 and t12:
        print(f"  Aggregate USDC V2 TVL Mar 11→12:  ${t11/1e6:.1f}M → ${t12/1e6:.1f}M  ({(t12-t11)/t11*100:+.1f}%)")
    if t12 and t13:
        print(f"  Aggregate USDC V2 TVL Mar 12→13:  ${t12/1e6:.1f}M → ${t13/1e6:.1f}M  ({(t13-t12)/t12*100:+.1f}%)")

    print()
    print("Aggregate TVL changes (USDT-paired pools):")
    usdt_df = df[df["group"] == "usdt"]
    s10 = usdt_df[usdt_df["date"].dt.date == pd.Timestamp("2023-03-10").date()]["tvl_usd"].sum()
    s11 = usdt_df[usdt_df["date"].dt.date == pd.Timestamp("2023-03-11").date()]["tvl_usd"].sum()
    s12 = usdt_df[usdt_df["date"].dt.date == pd.Timestamp("2023-03-12").date()]["tvl_usd"].sum()
    s13 = usdt_df[usdt_df["date"].dt.date == pd.Timestamp("2023-03-13").date()]["tvl_usd"].sum()
    if s10 and s11:
        print(f"  Aggregate USDT V2 TVL Mar 10→11:  ${s10/1e6:.1f}M → ${s11/1e6:.1f}M  ({(s11-s10)/s10*100:+.1f}%)")
    if s11 and s12:
        print(f"  Aggregate USDT V2 TVL Mar 11→12:  ${s11/1e6:.1f}M → ${s12/1e6:.1f}M  ({(s12-s11)/s11*100:+.1f}%)")

    print()
    print("USDC/WETH V2 pool TVL (largest USDC pool):")
    uw = df[df["pair_symbol"] == "USDC/WETH"]
    for d_str in ["2023-03-10", "2023-03-11", "2023-03-12", "2023-03-13"]:
        row = uw[uw["date"].dt.date == pd.Timestamp(d_str).date()]
        if not row.empty:
            print(f"  {d_str}  ${row['tvl_usd'].values[0]/1e6:.1f}M")
    uw10 = uw[uw["date"].dt.date == pd.Timestamp("2023-03-10").date()]["tvl_usd"]
    uw11 = uw[uw["date"].dt.date == pd.Timestamp("2023-03-11").date()]["tvl_usd"]
    if not uw10.empty and not uw11.empty:
        pct = (uw11.values[0] - uw10.values[0]) / uw10.values[0] * 100
        print(f"  USDC/WETH V2 change Mar 10→11: {pct:+.1f}%")

    print()
    print("USDT/WETH V2 pool TVL (largest USDT pool):")
    tw = df[df["pair_symbol"] == "USDT/WETH"]
    for d_str in ["2023-03-10", "2023-03-11", "2023-03-12", "2023-03-13"]:
        row = tw[tw["date"].dt.date == pd.Timestamp(d_str).date()]
        if not row.empty:
            print(f"  {d_str}  ${row['tvl_usd'].values[0]/1e6:.1f}M")
    tw10 = tw[tw["date"].dt.date == pd.Timestamp("2023-03-10").date()]["tvl_usd"]
    tw11 = tw[tw["date"].dt.date == pd.Timestamp("2023-03-11").date()]["tvl_usd"]
    if not tw10.empty and not tw11.empty:
        pct = (tw11.values[0] - tw10.values[0]) / tw10.values[0] * 100
        print(f"  USDT/WETH V2 change Mar 10→11: {pct:+.1f}%")

    print("=" * 70)


def main() -> None:
    log.info("Starting Uniswap V2 data collection for March 2023")
    log.info("Output: %s", OUTPUT_PATH)

    df = build_dataset()

    if not df.empty:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUTPUT_PATH, engine="fastparquet", index=False)
        log.info("Saved %d rows to %s", len(df), OUTPUT_PATH)
    else:
        log.error("No data to save.")

    print_diagnostics(df)


if __name__ == "__main__":
    main()
