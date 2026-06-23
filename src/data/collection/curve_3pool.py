"""Collect Curve Finance 3pool liquidity data for March 2023.

Cross-exchange validation: compares Curve.fi 3pool (USDC/USDT/DAI) dynamics
with Uniswap V3 findings during the SVB banking crisis.

Pool: Curve 3pool, 0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7
LP token (3CRV): 0x6c3F90f043a72FA612cbac8115EE7e52BDe6E490
Data sources:
  - TVL (3pool-specific): DeFi Llama yields chart API
  - Token balances (USDC/USDT/DAI, protocol-wide): DeFi Llama protocol API
  - Virtual price + LP supply: Ethereum on-chain via publicnode.com RPC (eth_call)

Output: data/filtered/curve_3pool_march2023.parquet

Key events:
  - Treatment 1: 2023-03-11 03:11 UTC (Circle's tweet about USDC reserve at SVB)
  - Treatment 2: 2023-03-12 22:24 UTC (Fed/FDIC/Treasury Joint Statement)
"""

import datetime
import logging
import time
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POOL_ADDRESS = "0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7"
LP_TOKEN_ADDRESS = "0x6c3F90f043a72FA612cbac8115EE7e52BDe6E490"

# DeFi Llama pool ID for Curve 3pool (DAI-USDC-USDT on Ethereum)
DEFILLAMA_POOL_ID = "25171c4c-1877-449a-9f88-45a9f153ee31"

# Public Ethereum RPC (no API key required)
ETH_RPC = "https://ethereum.publicnode.com"

# ABI function selectors for 3pool contract (Vyper)
SEL_VIRTUAL_PRICE = "0xbb7b8b80"   # get_virtual_price() -> uint256
SEL_BALANCES_0 = "0x4903b0d1"      # balances(uint256) arg 0 (DAI)
SEL_BALANCES_1 = "0x4903b0d1"      # balances(uint256) arg 1 (USDC)
SEL_BALANCES_2 = "0x4903b0d1"      # balances(uint256) arg 2 (USDT)

# totalSupply() selector for 3CRV ERC-20
SEL_TOTAL_SUPPLY = "0x18160ddd"    # totalSupply() -> uint256

# Ethereum block calibration (empirical, verified)
# Block 16730000 timestamp: 2023-02-28 23:45:23 UTC
_CAL_BLOCK = 16_730_000
_CAL_TS = 1_677_627_923  # 2023-02-28 23:45:23 UTC
_AVG_BLOCK_TIME = 12.15   # seconds/block (post-Merge, March 2023)

# March 2023 date range
MARCH_START = datetime.date(2023, 3, 1)
MARCH_END = datetime.date(2023, 3, 31)

# Output path
OUTPUT_PATH = Path(__file__).parents[3] / "data" / "filtered" / "curve_3pool_march2023.parquet"

# Token decimals
DECIMALS = {"DAI": 18, "USDC": 6, "USDT": 6}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: block number estimation
# ---------------------------------------------------------------------------

def _ts_to_block(unix_ts: int) -> int:
    """Estimate Ethereum mainnet block number for a given Unix timestamp."""
    delta_secs = unix_ts - _CAL_TS
    return _CAL_BLOCK + int(delta_secs / _AVG_BLOCK_TIME)


# ---------------------------------------------------------------------------
# Source 1: DeFi Llama yields chart — 3pool-specific TVL
# ---------------------------------------------------------------------------

def fetch_3pool_tvl_defillama() -> dict[str, float]:
    """Fetch daily 3pool TVL from DeFi Llama yields chart API.

    Returns dict mapping date string (YYYY-MM-DD) -> TVL in USD.
    """
    url = f"https://yields.llama.fi/chart/{DEFILLAMA_POOL_ID}"
    log.info("Fetching 3pool TVL from DeFi Llama: %s", url)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    records = data.get("data", [])
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
            result[date_str] = float(r.get("tvlUsd") or 0.0)

    log.info("DeFi Llama yields: %d March 2023 records", len(result))
    return result


# ---------------------------------------------------------------------------
# Source 2: DeFi Llama protocol API — per-token balances (protocol-wide)
# ---------------------------------------------------------------------------

def fetch_curve_token_balances_defillama() -> dict[str, dict[str, float]]:
    """Fetch protocol-wide Curve USDC/USDT/DAI USD balances from DeFi Llama.

    Note: this is the aggregate across ALL Curve pools on Ethereum, not just 3pool.
    It provides directional signal for token-level composition shifts.

    Returns dict mapping date string -> {'usdc_balance', 'usdt_balance', 'dai_balance'}.
    """
    url = "https://api.llama.fi/protocol/curve-dex"
    log.info("Fetching Curve protocol token balances from DeFi Llama: %s", url)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    tokens_list = data.get("tokensInUsd", [])
    result: dict[str, dict[str, float]] = {}

    for entry in tokens_list:
        ts = entry.get("date", 0)
        if not (1_677_628_800 <= ts <= 1_680_307_199):
            continue
        d = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).date()
        date_str = d.isoformat()
        tokens = entry.get("tokens", {})
        result[date_str] = {
            "usdc_balance_curve_proto": float(tokens.get("USDC", 0.0)),
            "usdt_balance_curve_proto": float(tokens.get("USDT", 0.0)),
            "dai_balance_curve_proto": float(tokens.get("DAI", 0.0)),
        }

    log.info("DeFi Llama token balances: %d March 2023 records", len(result))
    return result


# ---------------------------------------------------------------------------
# Source 3: On-chain via public Ethereum RPC
# ---------------------------------------------------------------------------

def _rpc_call(method: str, params: list, rpc_id: int = 1) -> dict:
    """Single JSON-RPC call."""
    payload = {"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params}
    resp = requests.post(ETH_RPC, json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _rpc_batch(calls: list[dict]) -> list[dict]:
    """Batch JSON-RPC calls (up to 10 at a time)."""
    resp = requests.post(ETH_RPC, json=calls, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _encode_uint256_param(n: int) -> str:
    """ABI-encode a single uint256 argument as 32-byte hex."""
    return n.to_bytes(32, "big").hex()


def _call_uint256(to: str, data: str, block_hex: str) -> int:
    """eth_call a function that returns a single uint256. Returns raw integer."""
    result = _rpc_call("eth_call", [{"to": to, "data": data}, block_hex])
    raw = result.get("result", "0x0")
    if not raw or raw == "0x":
        return 0
    return int(raw, 16)


def fetch_onchain_data_daily() -> dict[str, dict]:
    """Fetch virtual price and LP token supply for each March 2023 day.

    Uses eth_call at the block closest to 12:00 UTC each day.

    Calls per day (batched):
      1. get_virtual_price() on 3pool
      2. totalSupply() on 3CRV LP token
      3. balances(0) — DAI raw balance
      4. balances(1) — USDC raw balance
      5. balances(2) — USDT raw balance

    Returns dict mapping date string -> {virtual_price, lp_supply, dai_balance, usdc_balance, usdt_balance}.
    """
    log.info("Fetching on-chain data via %s", ETH_RPC)

    dates = []
    d = MARCH_START
    while d <= MARCH_END:
        # Use 12:00 UTC as daily snapshot time
        noon_utc = datetime.datetime(d.year, d.month, d.day, 12, 0, 0,
                                     tzinfo=datetime.timezone.utc)
        ts = int(noon_utc.timestamp())
        block = _ts_to_block(ts)
        dates.append((d.isoformat(), block))
        d += datetime.timedelta(days=1)

    log.info("Computing data for %d days (blocks %d–%d)",
             len(dates), dates[0][1], dates[-1][1])

    # Build per-day batch requests (5 calls per day)
    result: dict[str, dict] = {}
    BATCH_SIZE = 5  # calls per day, process 3 days at a time to avoid RPC limits

    def _balances_call(idx: int, block_hex: str, rpc_id: int) -> dict:
        """Build an eth_call batch entry for balances(idx)."""
        data = "0x4903b0d1" + _encode_uint256_param(idx)
        return {"jsonrpc": "2.0", "id": rpc_id, "method": "eth_call",
                "params": [{"to": POOL_ADDRESS, "data": data}, block_hex]}

    days_per_batch = 6  # 6 days × 5 calls = 30 calls per HTTP request
    for batch_start in range(0, len(dates), days_per_batch):
        batch_dates = dates[batch_start: batch_start + days_per_batch]
        calls = []
        call_map: list[tuple[str, str]] = []  # (date_str, field)

        for date_str, block in batch_dates:
            bh = hex(block)
            base_id = len(calls)
            calls.append({
                "jsonrpc": "2.0", "id": base_id,
                "method": "eth_call",
                "params": [{"to": POOL_ADDRESS, "data": SEL_VIRTUAL_PRICE}, bh]
            })
            call_map.append((date_str, "virtual_price_raw"))

            calls.append({
                "jsonrpc": "2.0", "id": base_id + 1,
                "method": "eth_call",
                "params": [{"to": LP_TOKEN_ADDRESS, "data": SEL_TOTAL_SUPPLY}, bh]
            })
            call_map.append((date_str, "lp_supply_raw"))

            calls.append(_balances_call(0, bh, base_id + 2))
            call_map.append((date_str, "dai_raw"))

            calls.append(_balances_call(1, bh, base_id + 3))
            call_map.append((date_str, "usdc_raw"))

            calls.append(_balances_call(2, bh, base_id + 4))
            call_map.append((date_str, "usdt_raw"))

        # Execute batch
        try:
            responses = _rpc_batch(calls)
        except Exception as exc:
            log.warning("Batch RPC failed: %s — retrying individually", exc)
            responses = []
            for call in calls:
                try:
                    single = _rpc_call(call["method"], call["params"])
                    single["id"] = call["id"]
                    responses.append(single)
                    time.sleep(0.1)
                except Exception as e2:
                    log.error("Individual call failed: %s", e2)
                    responses.append({"id": call["id"], "result": "0x0"})

        # Parse responses (responses may be unordered)
        id_to_result = {r["id"]: r.get("result", "0x0") for r in responses}

        for i, (date_str, field) in enumerate(call_map):
            raw_hex = id_to_result.get(i, "0x0") or "0x0"
            try:
                raw_int = int(raw_hex, 16) if raw_hex != "0x" else 0
            except ValueError:
                raw_int = 0

            if date_str not in result:
                result[date_str] = {}
            result[date_str][field] = raw_int

        log.info("Batch %d–%d complete (%d/%d days)",
                 batch_start + 1,
                 min(batch_start + days_per_batch, len(dates)),
                 min(batch_start + days_per_batch, len(dates)),
                 len(dates))
        time.sleep(0.3)  # be polite to the public RPC

    # Convert raw values
    final: dict[str, dict] = {}
    for date_str, raw in result.items():
        vp_raw = raw.get("virtual_price_raw", 0)
        virtual_price = vp_raw / 1e18 if vp_raw else None

        lp_raw = raw.get("lp_supply_raw", 0)
        lp_supply = lp_raw / 1e18 if lp_raw else None  # 3CRV has 18 decimals

        dai_raw = raw.get("dai_raw", 0)
        dai_balance = dai_raw / 1e18 if dai_raw else None  # DAI: 18 decimals

        usdc_raw = raw.get("usdc_raw", 0)
        usdc_balance = usdc_raw / 1e6 if usdc_raw else None  # USDC: 6 decimals

        usdt_raw = raw.get("usdt_raw", 0)
        usdt_balance = usdt_raw / 1e6 if usdt_raw else None  # USDT: 6 decimals

        final[date_str] = {
            "virtual_price": virtual_price,
            "lp_supply": lp_supply,
            "dai_balance": dai_balance,
            "usdc_balance": usdc_balance,
            "usdt_balance": usdt_balance,
        }

    log.info("On-chain data fetched: %d days", len(final))
    return final


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_dataset() -> pd.DataFrame:
    """Fetch all data sources and merge into a single daily DataFrame."""

    # --- Fetch ---
    tvl_map = fetch_3pool_tvl_defillama()
    token_bal_map = fetch_curve_token_balances_defillama()
    onchain_map = fetch_onchain_data_daily()

    # --- Build row-by-row ---
    rows = []
    d = MARCH_START
    while d <= MARCH_END:
        date_str = d.isoformat()
        tvl = tvl_map.get(date_str, None)
        tbal = token_bal_map.get(date_str, {})
        oc = onchain_map.get(date_str, {})

        rows.append({
            "date": pd.Timestamp(date_str),
            # Source 1: 3pool-specific TVL from DeFi Llama yields chart
            "tvl_usd": tvl,
            # Source 3: On-chain token balances (raw counts, not USD)
            "usdc_balance": oc.get("usdc_balance"),
            "usdt_balance": oc.get("usdt_balance"),
            "dai_balance": oc.get("dai_balance"),
            # Source 3: On-chain virtual price and LP supply
            "virtual_price": oc.get("virtual_price"),
            "lp_supply": oc.get("lp_supply"),
            # Source 2: Protocol-wide Curve token USD balances (for directional cross-check)
            "usdc_bal_curve_proto_usd": tbal.get("usdc_balance_curve_proto"),
            "usdt_bal_curve_proto_usd": tbal.get("usdt_balance_curve_proto"),
            "dai_bal_curve_proto_usd": tbal.get("dai_balance_curve_proto"),
        })
        d += datetime.timedelta(days=1)

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)

    # Derived: total token balance in pool (USD) using on-chain counts
    # DAI: 1:1 USD (stablecoin), USDC/USDT: 1:1 USD approximate
    df["total_tokens_usd_approx"] = (
        df["dai_balance"].fillna(0)
        + df["usdc_balance"].fillna(0)
        + df["usdt_balance"].fillna(0)
    )
    # Replace zeros with NaN
    df.loc[df["total_tokens_usd_approx"] == 0, "total_tokens_usd_approx"] = float("nan")

    log.info("Dataset built: %d rows × %d columns", len(df), len(df.columns))
    return df


def main() -> None:
    log.info("Starting Curve 3pool data collection for March 2023")
    log.info("Pool:     %s", POOL_ADDRESS)
    log.info("LP token: %s", LP_TOKEN_ADDRESS)
    log.info("Output:   %s", OUTPUT_PATH)

    df = build_dataset()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, engine="fastparquet", index=False)
    log.info("Saved %d rows to %s", len(df), OUTPUT_PATH)


if __name__ == "__main__":
    main()
