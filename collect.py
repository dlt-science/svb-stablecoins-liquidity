r"""Key-free data collection from Ethereum mainnet through public JSON-RPC endpoints.

No API key and no identity is sent. The header is a generic User-Agent. You can give
your own endpoint list (`urls`) to every function. The functions read: blocks by time,
Mint and Burn events, pool state (slot0, liquidity, tick spacing), token data, pool
token balances and Chainlink USD prices.

    python collect.py events --pools data/paper_pools.csv \
        --to-block 16950602 --out events.parquet
    python collect.py snapshots --pools data/paper_pools.csv \
        --blocks data/snapshot_blocks.csv --events events.parquet --out ticks/
    python collect.py chainlink --blocks-from data/snapshot_blocks.csv \
        --out chainlink.csv
    python collect.py balances --pools data/paper_pools.csv \
        --blocks-from data/snapshot_blocks.csv --out balances.csv
    python collect.py snapshot --time "2023-03-11 04:00" \
        --pool 0xbb5e639a18631e38df1569e1fd2dc81c6d1ce1d4
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from eth_abi import decode, encode
from eth_utils import keccak, to_checksum_address

import liquidity as liq

URLS = [
    "https://eth.drpc.org",
    "https://mainnet.gateway.tenderly.co",
    "https://eth-mainnet.public.blastapi.io",
]

HEADERS = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}

# Uniswap v3 factory deployment
FACTORY_BLOCK = 12_369_621

MULTICALL3 = "0xcA11bde05977b3631167028862bE2a173976CA11"

# Event topics of a pool
MINT = "Mint(address,address,int24,int24,uint128,uint256,uint256)"
BURN = "Burn(address,int24,int24,uint128,uint256,uint256)"
T_MINT = "0x" + keccak(text=MINT).hex()
T_BURN = "0x" + keccak(text=BURN).hex()

EVENT_COLS = [
    "pool_address",
    "block",
    "log_index",
    "tx_hash",
    "event",
    "owner",
    "tick_lower",
    "tick_upper",
    "liquidity_delta",
]

# Chainlink USD feeds on Ethereum mainnet (8 decimals)
FEEDS = {
    "USDC": "0x8fFfFfd4AfB6115b954Bd326cbe7B4BA576818f6",
    "USDT": "0x3E7d1eAB13ad0104d2750B8863b489D65364e32D",
}


def sel(signature):
    """Four-byte function selector."""
    return keccak(text=signature)[:4]


def rpc(method, params, urls=URLS, tries=10):
    """One JSON-RPC call. The call goes to the next endpoint after each failure, with a
    longer wait each time."""
    request = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    last = None

    for k in range(tries):
        url = urls[k % len(urls)]

        try:
            j = requests.post(url, json=request, headers=HEADERS, timeout=120).json()

            # Some public endpoints give "0x" when they do not hold the archive state.
            # Try the next endpoint.
            if j.get("result") not in (None, "0x"):
                return j["result"]

            last = j
        except (requests.RequestException, ValueError) as e:
            last = repr(e)[:200]

        time.sleep(0.5 * (k + 1))

    raise RuntimeError(f"{method} failed: {last}")


def multicall(calls, block, urls=URLS, chunk=250):
    """Many contract reads at one block with Multicall3 aggregate3. `calls` is
    [(address, calldata)]. Returns [(success, return bytes)]."""
    aggregate3 = sel("aggregate3((address,bool,bytes)[])")
    out = []

    for i in range(0, len(calls), chunk):
        # True: one failed read does not stop the other reads.
        batch = [(to_checksum_address(a), True, c) for a, c in calls[i : i + chunk]]
        data = aggregate3 + encode(["(address,bool,bytes)[]"], [batch])

        call = {"to": MULTICALL3, "data": "0x" + data.hex()}
        ret = rpc("eth_call", [call, hex(block)], urls)

        out += decode(["(bool,bytes)[]"], bytes.fromhex(ret[2:]))[0]

    return out


def block_time(block, urls=URLS):
    """Unix time of a block."""
    return int(rpc("eth_getBlockByNumber", [hex(block), False], urls)["timestamp"], 16)


def block_at(timestamp, urls=URLS):
    """Last block with a time at or before `timestamp` (Unix seconds, or UTC text such
    as "2023-03-11 04:00"). The paper's hourly snapshots use this rule ("closest
    before")."""
    t = timestamp

    if not isinstance(t, int):
        t = int(pd.Timestamp(t).timestamp())

    lo = FACTORY_BLOCK
    hi = int(rpc("eth_blockNumber", [], urls), 16)

    # binary search on block times
    while lo < hi:
        mid = (lo + hi + 1) // 2

        if block_time(mid, urls) <= t:
            lo = mid
        else:
            hi = mid - 1

    return lo


def _logs(pools, a, b, urls):
    """Mint and Burn logs of `pools` in blocks a..b. The range is cut in half when an
    endpoint refuses it."""
    query = {
        "address": pools,
        "topics": [[T_MINT, T_BURN]],
        "fromBlock": hex(a),
        "toBlock": hex(b),
    }

    try:
        return rpc("eth_getLogs", [query], urls, tries=3)
    except RuntimeError:
        if b - a < 50:
            raise

        mid = (a + b) // 2

        return _logs(pools, a, mid, urls) + _logs(pools, mid + 1, b, urls)


def _signed(word):
    """Signed integer of a 32-byte hex word (two's complement)."""
    v = int(word, 16)

    return v - (1 << 256) if v >= 1 << 255 else v


def _event_row(lg):
    """One Mint (+L) or Burn (-L) log as a row of EVENT_COLS."""
    mint = lg["topics"][0] == T_MINT

    # Mint data: sender, amount, amount0, amount1. Burn data: amount, amount0, amount1.
    delta = int(lg["data"][66:130], 16) if mint else -int(lg["data"][2:66], 16)

    return (
        lg["address"].lower(),
        int(lg["blockNumber"], 16),
        int(lg["logIndex"], 16),
        lg["transactionHash"],
        "Mint" if mint else "Burn",
        "0x" + lg["topics"][1][-40:],
        _signed(lg["topics"][2]),
        _signed(lg["topics"][3]),
        str(delta),
    )


def parse_logs(logs):
    """Mint (+L) and Burn (-L) logs as rows of EVENT_COLS. liquidity_delta is text
    (uint128 range)."""
    rows = [_event_row(lg) for lg in logs if lg["topics"][0] in (T_MINT, T_BURN)]

    return pd.DataFrame(rows, columns=EVENT_COLS)


def fetch_events(pools, to_block, from_block=FACTORY_BLOCK, step=50_000, urls=URLS):
    """Every Mint and Burn of `pools` from the factory block to `to_block`. Principal
    and tick nets are cumulative sums, so you need the full history."""
    pools = [p.lower() for p in pools]
    logs = []

    for a in range(from_block, to_block + 1, step):
        logs += _logs(pools, a, min(a + step - 1, to_block), urls)

    events = parse_logs(logs).drop_duplicates(["tx_hash", "log_index"])

    return events.sort_values(["pool_address", "block", "log_index"])


def _text(ok, b):
    """ERC-20 symbol as string or bytes32 (some old tokens), without NUL bytes."""
    if not ok:
        return ""

    try:
        return decode(["string"], b)[0]
    except Exception:  # noqa: BLE001  (bytes32 symbols do not decode as string)
        return b[:32].rstrip(b"\x00").decode(errors="ignore")


def pool_states(pools, block, urls=URLS):
    """Pool data of each pool at `block` (one dict per pool): tokens, symbols, decimals,
    fee, tick spacing, sqrtPriceX96, tick and liquidity. Two Multicall3 batches read all
    pools."""
    fns = ("token0()", "token1()", "fee()", "tickSpacing()", "slot0()", "liquidity()")

    # six results per pool
    r = multicall([(p, sel(f)) for p in pools for f in fns], block, urls)
    r = [r[k : k + 6] for k in range(0, len(r), 6)]

    toks = [[decode(["address"], x[k][1])[0].lower() for k in (0, 1)] for x in r]

    # four results per pool: symbol and decimals of token0, then of token1
    token_fns = ("symbol()", "decimals()")
    calls = [(a, sel(f)) for t0t1 in toks for a in t0t1 for f in token_fns]
    t = multicall(calls, block, urls)

    out = []

    for p, x, (t0, t1), k in zip(pools, r, toks, range(0, len(t), 4), strict=True):
        sqrt_p, tick = decode(["uint160", "int24"], x[4][1][:64])

        out.append(
            {
                "pool_address": p.lower(),
                "block": block,
                "token0": t0,
                "token1": t1,
                "token0_symbol": _text(*t[k]),
                "token0_decimals": decode(["uint8"], t[k + 1][1])[0],
                "token1_symbol": _text(*t[k + 2]),
                "token1_decimals": decode(["uint8"], t[k + 3][1])[0],
                "fee_tier": decode(["uint24"], x[2][1])[0],
                "tick_spacing": decode(["int24"], x[3][1])[0],
                "sqrt_price_x96": sqrt_p,
                "tick": tick,
                "liquidity": decode(["uint128"], x[5][1])[0],
            }
        )

    return out


def pool_state(pool, block, urls=URLS):
    """Pool data of one pool at `block` (pool_states of one pool)."""
    return pool_states([pool], block, urls)[0]


def balances(pools, block, urls=URLS):
    """Raw token balances {pool: (balance0, balance1)} of each pool contract at `block`.
    Balances include the fees that LPs have not collected. The paper uses them as a
    check of the principal."""
    token_fns = ("token0()", "token1()")
    r = multicall([(p, sel(s)) for p in pools for s in token_fns], block, urls)
    tokens = [decode(["address"], b)[0] for _, b in r]

    # tokens holds token0 and token1 of each pool, so pool i // 2 owns token i
    balance_of = sel("balanceOf(address)")

    calls = [
        (t, balance_of + encode(["address"], [to_checksum_address(pools[i // 2])]))
        for i, t in enumerate(tokens)
    ]

    r = multicall(calls, block, urls)
    b = [decode(["uint256"], x)[0] if ok else None for ok, x in r]

    return {p.lower(): (b[2 * i], b[2 * i + 1]) for i, p in enumerate(pools)}


def chainlink_usd(block, feeds=FEEDS, urls=URLS):
    """USD price of each stablecoin at `block`: the answer of the Chainlink round in
    effect (latestRoundData). A failed feed read gives NaN."""
    r = multicall([(a, sel("latestRoundData()")) for a in feeds.values()], block, urls)
    answer = ["uint80", "int256", "uint256", "uint256", "uint80"]
    out = {"block": block}

    for s, (ok, b) in zip(feeds, r, strict=True):
        out[f"{s.lower()}_usd"] = decode(answer, b)[1] / 1e8 if ok else float("nan")

    return out


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------

STATE_COLS = (
    "pool_address",
    "token0_symbol",
    "token1_symbol",
    "token0_decimals",
    "token1_decimals",
    "fee_tier",
    "tick_spacing",
)


def blocks_of(path):
    """Sorted snapshot blocks (column block_number) of a file."""
    blocks = liq.read(path, columns=["block_number"]).block_number

    return sorted(set(blocks.astype(int)))


def table_of(s, ev):
    """Tick nets, tick table, stablecoin side and pool price of one pool state `s`
    (pool_state). `ev` holds the Mint and Burn events of this pool."""
    d0, d1 = s["token0_decimals"], s["token1_decimals"]
    nets = liq.nets_from_events(ev, s["block"])

    table = liq.tick_table(
        s["sqrt_price_x96"], s["tick"], s["liquidity"], s["tick_spacing"], nets, d0, d1
    )

    # The stablecoin is token1 if it is USDC or USDT.
    q1 = liq.name(s["token1_symbol"]) in liq.STABLE
    price = liq.stable_price(s["sqrt_price_x96"], d0, d1, q1)

    return nets, table, q1, price


def cmd_events(a):
    """Mint and Burn events of the pools since the factory block."""
    if a.pools.endswith((".csv", ".parquet")):
        pools = liq.read(a.pools).pool_address.tolist()
    else:
        pools = a.pools.split(",")

    liq.write(fetch_events(pools, a.to_block, urls=a.rpc), a.out)


def cmd_chainlink(a):
    """Chainlink USDC and USDT prices at each snapshot block."""
    rows = [chainlink_usd(b, urls=a.rpc) for b in blocks_of(a.blocks_from)]

    liq.write(pd.DataFrame(rows), a.out)


def cmd_balances(a):
    """Raw token balances of each pool at each block (the paper's check of the
    principal)."""
    pools = liq.read(a.pools).pool_address.tolist()
    rows = []

    for b in blocks_of(a.blocks_from):
        for p, (x0, x1) in balances(pools, b, a.rpc).items():
            row = {"block": b, "pool_address": p}
            rows.append(row | {"balance0": str(x0), "balance1": str(x1)})

    liq.write(pd.DataFrame(rows), a.out)


def cmd_snapshots(a):
    """Tick tables of each pool at each snapshot block (the MCI input), from the chain
    and the events file. Writes one parquet file per hour into the folder --out, with
    the columns of the released tick file. An hour that has a file is not read again, so
    you can restart a run that stopped."""
    pools = liq.read(a.pools).pool_address.str.lower().tolist()
    ev = liq.read(a.events)
    by_pool = dict(tuple(ev.groupby("pool_address")))

    if a.blocks:
        hours = liq.read(a.blocks)
    else:
        hours = pd.DataFrame(
            [
                {"datetime_utc": h, "block_number": block_at(str(h), a.rpc)}
                for h in pd.date_range(a.start, a.end, freq="h")
            ]
        )

    times = pd.to_datetime(hours.datetime_utc)
    blocks = hours.block_number.astype(int)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    for h, b in zip(times, blocks, strict=True):
        f = out / f"{h:%Y-%m-%d %H-%M-%S}.parquet"

        if f.exists():
            continue

        parts = []

        for p, s in zip(pools, pool_states(pools, b, a.rpc), strict=True):
            _, t, _, price = table_of(s, by_pool.get(p, ev.iloc[:0]))

            parts.append(
                t.assign(
                    datetime_utc=h,
                    block_number=b,
                    cross_tick=s["tick"],
                    price_in_stablecoin=price,
                    sqrt_price_x96=str(s["sqrt_price_x96"]),
                    **{k: s[k] for k in STATE_COLS},
                )
            )

        liq.write(pd.concat(parts, ignore_index=True), f)


def cmd_snapshot(a):
    """Tick table, MCI profile and principal TVL of one pool at one time, from the chain
    only."""
    block = a.block or block_at(a.time, a.rpc)
    s = pool_state(a.pool, block, a.rpc)

    ev = liq.read(a.events) if a.events else fetch_events([a.pool], block, urls=a.rpc)
    ev = ev[ev.pool_address == s["pool_address"]]

    nets, table, q1, price = table_of(s, ev)
    d0, d1 = s["token0_decimals"], s["token1_decimals"]
    sqrt_price = s["sqrt_price_x96"]

    prof = liq.mci_profile(table, q1, sqrt_price, price, s["tick_spacing"], d0, d1)

    p0, p1, _ = liq.principal_at(ev, [block], [sqrt_price / 2**96])
    tvl_stable = float(liq.tvl_stable_units(p0[0] / 10**d0, p1[0] / 10**d1, price, q1))

    stable = liq.name(s["token1_symbol"] if q1 else s["token0_symbol"])
    usd = chainlink_usd(block, urls=a.rpc)[f"{stable.lower()}_usd"]

    keys = ("pool_address", "block", "token0_symbol", "token1_symbol", "fee_tier")
    from_events = liq.active_liquidity(nets, s["tick"])

    print({k: s[k] for k in (*keys, "tick")})
    print(f"active liquidity: pool {s['liquidity']}, from events {from_events}")

    print(
        f"price {price:.6g}, principal TVL {tvl_stable:,.2f} stablecoin units"
        f" = {tvl_stable * usd:,.2f} USD"
    )

    for i in liq.LEVELS:
        print(
            f"I={i:>2}: MCI {prof[f'MCI_{i}']:.6g}  buy {prof[f'Buy_MCI_{i}']:.6g}"
            f"  sell {prof[f'Sell_MCI_{i}']:.6g}"
        )

    if a.out:
        liq.write(table, a.out)


COMMANDS = {
    "events": cmd_events,
    "snapshots": cmd_snapshots,
    "chainlink": cmd_chainlink,
    "balances": cmd_balances,
    "snapshot": cmd_snapshot,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    ap.add_argument(
        "--rpc", nargs="*", default=URLS, help="your own endpoints (optional)"
    )

    sub = ap.add_subparsers(dest="cmd", required=True)
    pools_help = "CSV/parquet with pool_address"
    blocks_help = "a file with the column block_number"

    # events
    e = sub.add_parser("events", help="Mint and Burn events since the factory block")
    e.add_argument(
        "--pools", required=True, help=pools_help + ", or addresses, by commas"
    )
    e.add_argument("--to-block", type=int, required=True)
    e.add_argument("--out", required=True)

    # snapshots
    t = sub.add_parser("snapshots", help="tick tables of many pools at each block")
    t.add_argument("--pools", required=True, help=pools_help)
    t.add_argument("--events", required=True, help="events file of the pools")
    t.add_argument("--blocks", help="CSV with datetime_utc and block_number")
    t.add_argument(
        "--start", help='first UTC hour without --blocks, "2023-02-28 00:00"'
    )
    t.add_argument("--end", help='last UTC hour without --blocks, "2023-04-01 00:00"')
    t.add_argument("--out", required=True, help="folder for one parquet file per hour")

    # chainlink
    c = sub.add_parser("chainlink", help="Chainlink USDC and USDT prices at each block")
    c.add_argument("--blocks-from", required=True, help=blocks_help)
    c.add_argument("--out", required=True)

    # balances
    b = sub.add_parser("balances", help="token balances of each pool at each block")
    b.add_argument("--pools", required=True, help=pools_help)
    b.add_argument("--blocks-from", required=True, help=blocks_help)
    b.add_argument("--out", required=True)

    # snapshot
    s = sub.add_parser("snapshot", help="tick table, MCI and TVL of one pool")
    s.add_argument("--pool", required=True)
    s.add_argument("--time", help='UTC time, for example "2023-03-11 04:00"')
    s.add_argument("--block", type=int)
    s.add_argument("--events", help="events file (else the events come from the chain)")
    s.add_argument("--out", help="write the tick table here")

    a = ap.parse_args()
    COMMANDS[a.cmd](a)
