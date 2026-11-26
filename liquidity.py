r"""Liquidity measures of Uniswap v3 pools: the tick table, the adapted Marginal Cost of
Immediacy (MCI), the TVL as LP principal, and the hourly panel of the paper.

    python liquidity.py --ticks tick_liquidity_hourly.parquet \
        --events events.parquet --chainlink chainlink.csv --out panel.parquet

--ticks is the Hugging Face tick file or a folder of hourly files named "YYYY-MM-DD
HH-MM-SS.parquet" (the folder that "collect.py snapshots" writes). The tick file has one
time column, datetime_utc. 769 hours take about 4 minutes.
"""

import argparse
import math
from bisect import bisect_right
from decimal import ROUND_UP, Decimal, localcontext
from itertools import accumulate
from pathlib import Path

import fastparquet
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------------------
# Read and write tables by file extension (.parquet, .csv or .csv.gz).


def read(path, columns=None):
    """Read a parquet or CSV file. `columns` reads only these columns."""
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path, columns=columns)

    return pd.read_csv(path, usecols=columns)


def write(df, path):
    """Write a parquet or CSV file (by extension) and print the number of rows."""
    if str(path).endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

    print(f"wrote {path}: {len(df)} rows")


# --------------------------------------------------------------------------------------
# Tick table
# --------------------------------------------------------------------------------------
# Tick table of one Uniswap v3 pool at one block: the liquidity and the token amounts of
# each tick level around the price.
#
# Paper: Section III.A.2, "Ticks' liquidity distribution", equation (4). The formulas
# come from the Uniswap v3 whitepaper (Adams, Zinsmeister, Salem, Keefer and Robinson,
# 2021): a tick i has the price 1.0001^i (eq. 6.1), the liquidity in a tick range
# changes by the net liquidity of each initialized tick that the price crosses
# (section 6.3), and the token amounts that liquidity L holds in a range follow
# eqs. 6.29 and 6.30.

TICK_BASE = Decimal("1.0001")
Q96 = Decimal(2) ** 96
MIN_TICK, MAX_TICK = -887272, 887272


def sqrt_bounds(lower, upper):
    """Square-root prices of the lower and upper tick of a level (raw units),
    sqrt(1.0001^i). Uses the caller's decimal context (the callers set 34 digits)."""
    return (TICK_BASE**lower).sqrt(), (TICK_BASE**upper).sqrt()


def stable_price(sqrt_price_x96, decimals0, decimals1, quote_is_token1):
    """Price of the paired token in units of the stablecoin (the quote token), from
    slot0.sqrtPriceX96. This is the column price_in_stablecoin of the tick snapshots.
    The steps are the steps of the paper's collector, so the float is the same."""
    with localcontext() as ctx:
        ctx.prec, ctx.rounding = 34, ROUND_UP

        ratio = (Decimal(sqrt_price_x96) / Q96) ** 2
        p = float(ratio / Decimal(10 ** (decimals1 - decimals0)))

    return p if quote_is_token1 else 1 / p


def nets_from_events(events, block):
    """Net liquidity {tick: liquidityNet} of the initialized ticks at `block`. A Mint
    adds L at tickLower and removes L at tickUpper. A Burn does the opposite. `events`
    needs the columns block, tick_lower, tick_upper and liquidity_delta (signed, as
    text)."""
    cols = ["tick_lower", "tick_upper", "liquidity_delta"]
    rows = events.loc[events.block <= block, cols]
    net = {}

    for lo, up, d in rows.itertuples(index=False):
        net[int(lo)] = net.get(int(lo), 0) + int(d)
        net[int(up)] = net.get(int(up), 0) - int(d)

    return {t: n for t, n in net.items() if n != 0}


def active_liquidity(nets, tick):
    """Liquidity of the active level (pool.liquidity()): the sum of the nets at or below
    the current tick."""
    return sum(n for t, n in nets.items() if t <= tick)


def level_liquidity(nets, liquidity, active, levels):
    """Liquidity of each level in `levels`, given `liquidity` at the `active` level
    (whitepaper, section 6.3). Between the active level and a level i, the price crosses
    the initialized ticks in between. Each tick crossed upward adds its net, each tick
    crossed downward removes it. So L(i) = L(active) + S(i) - S(active), where S(i) is
    the sum of the nets at or below i."""
    ticks = sorted(nets)

    # sums[k]: the nets of the first k ticks
    sums = [0] + list(accumulate(nets[t] for t in ticks))

    def s(i):
        """S(i): the sum of the nets at or below i."""
        return sums[bisect_right(ticks, i)]

    return [liquidity + s(i) - s(active) for i in levels]


def level_amounts(L, sqrt_price, lower, upper, tick):
    """Raw token amounts (x, y) that liquidity L holds in the level [lower, upper) when
    the pool is at `tick` (whitepaper, eqs. 6.29 and 6.30), rounded down to whole raw
    units. Below the level the pool holds only token0, above it only token1, and in it
    both."""
    sa, sb = sqrt_bounds(lower, upper)
    s = sa if tick < lower else sb if tick >= upper else sqrt_price

    return math.floor(L * (sb - s) / (s * sb)), math.floor(L * (s - sa))


def tick_table(
    sqrt_price_x96, tick, liquidity, spacing, nets, decimals0, decimals1, n_steps=None
):
    """One row per tick level, in ascending order, with `n_steps` levels on each side of
    the active level. The default `n_steps` is the number of initialized ticks, as in
    the paper's snapshots. The MCI depth rule (max_depth) depends on this default.
    Columns: tickIdx, liquidityActive (text), totalValueLockedToken0/1 (token units),
    isCurrentTick."""
    n = len(nets) if n_steps is None else n_steps

    # the level that holds the price
    a = tick // spacing * spacing

    levels = [
        i
        for i in range(a - n * spacing, a + (n + 1) * spacing, spacing)
        if MIN_TICK - spacing < i <= MAX_TICK
    ]

    liq = level_liquidity(nets, liquidity, a, levels)

    with localcontext() as ctx:
        # the context of the paper's collector
        ctx.prec, ctx.rounding = 34, ROUND_UP

        sp = Decimal(sqrt_price_x96) / Q96

        amounts = [
            level_amounts(L, sp, i, i + spacing, tick)
            for i, L in zip(levels, liq, strict=True)
        ]

        token0 = [float(Decimal(x) / 10**decimals0) for x, _ in amounts]
        token1 = [float(Decimal(y) / 10**decimals1) for _, y in amounts]

    return pd.DataFrame(
        {
            "tickIdx": levels,
            "liquidityActive": [str(L) for L in liq],
            "totalValueLockedToken0": token0,
            "totalValueLockedToken1": token1,
            "isCurrentTick": [i == a for i in levels],
        }
    )


# --------------------------------------------------------------------------------------
# Marginal Cost of Immediacy (MCI)
# --------------------------------------------------------------------------------------
# Adapted MCI of one Uniswap v3 pool snapshot.
#
# Paper: Section III.B.1, equations (9) and (10) for the swap of one tick level, (11)
# and (12) for the VWAP and the VWAPM, (13) and (14) for the buy and sell side, (15) for
# the imbalance and (16) for the mean. The MCI is in basis points per 1,000 units of the
# stablecoin (Cenesizoglu and Grass, 2018). The input is a tick table (tick_table, or
# one pool of the released hourly snapshots).

LEVELS = (1, 5, 10, 15, 20)

# A price this close to a tick edge leaves nothing on that side.
EDGE = Decimal("1e-18")


def max_depth(n_rows):
    """Deepest depth I that the table supports. The paper needs at least I+1 levels on
    each side."""
    return int(n_rows / 2) - 1


def _level_swap(down, L, lower, upper, x, y, d0, d1):
    """Token amounts (dx, dy) that one order takes from one level, in token units. The
    order takes all of one token (x or y) and pays the other token up to the level edge.
    """
    # x, y: X_real, Y_real of eq. (3), here in raw units.
    x = x * 10 ** Decimal(d0)
    y = y * 10 ** Decimal(d1)
    L = Decimal(L)
    sa, sb = sqrt_bounds(lower, upper)

    # dx, dy: the swap of one level, eq. (9) (up) and eq. (10) (down).
    if down:
        dx, dy = L * (1 / sa - 1 / sb) - x, y
    else:
        dx, dy = x, L * (sb - sa) - y

    dx = dx / 10 ** Decimal(d0)
    dy = dy / 10 ** Decimal(d1)

    return (dx, dy) if dx > 0 and dy > 0 else (0, 0)


def _walk(table, depth, down, sqrt_price_x96, spacing, d0, d1):
    """Sum of dx and dy over `depth` levels, from the active level down (sell token0) or
    up (buy token0)."""
    start = int(table.index[table.isCurrentTick][0])
    step = -1 if down else 1
    sx = sy = 0

    for i in range(start, start + step * depth, step):
        if not 0 <= i < len(table):
            break

        lower = int(table.tickIdx.iat[i])
        L = table.liquidityActive.iat[i]
        x = Decimal(table.totalValueLockedToken0.iat[i])
        y = Decimal(table.totalValueLockedToken1.iat[i])

        if i == start:
            # Active level: take the amounts from L and the pool price, not from the
            # rounded table values.
            sp = Decimal(sqrt_price_x96) / Decimal(2) ** 96
            La = Decimal(L)
            sa, sb = sqrt_bounds(lower, lower + spacing)

            x = La * (1 / sp - 1 / sb) if (sb - sp) / sp > EDGE else Decimal(0)
            y = La * (sp - sa) if (sp - sa) / sp > EDGE else Decimal(0)

            x = x / Decimal(10) ** d0
            y = y / Decimal(10) ** d1

        dx, dy = _level_swap(down, L, lower, lower + spacing, x, y, d0, d1)
        sx, sy = sx + dx, sy + dy

    return Decimal(sx), Decimal(sy)


def mci_side(
    table, depth, buy, quote_is_token1, sqrt_price_x96, price, spacing, d0, d1
):
    """MCI of one side at one depth. `buy` buys the paired token with the stablecoin.
    `price` is the paired token in stablecoin units. Returns 0.0 when the side has no
    volume or the table is too short, as in the paper. The table must have a 0-based
    index (mci_profile resets it)."""
    with localcontext() as ctx:
        ctx.prec = 34

        if depth > max_depth(len(table)):
            return 0.0

        # The walk sells token0 (moves the price down).
        down = buy != quote_is_token1
        sx, sy = _walk(table, depth, down, sqrt_price_x96, spacing, d0, d1)

        if sx == 0 or sy == 0:
            return 0.0

        # mid: token1 per token0. volume: in units of the stablecoin.
        mid = Decimal(price) if quote_is_token1 else 1 / Decimal(price)
        volume = sy if quote_is_token1 else sx
        sign = -1 if down else 1

        # eq. (11): VWAP = sy / sx, in token1 per token0.
        # eq. (12): VWAPM = ln(VWAP / mid).
        vwapm = (sy / sx / mid).ln()

        # eqs. (13) and (14): MCI = VWAPM / stablecoin volume, in basis points per
        # 1,000 units. The sign is -1 for a walk down the book. For a stablecoin as
        # token1 this is the minus sign of eq. (14). For a stablecoin as token0, sy / sx
        # is 1 / VWAP of eq. (11), so the same sign puts the ratio right.
        return float(sign * vwapm / (volume / Decimal(1000)) * Decimal(10000))


def mci_profile(
    table, quote_is_token1, sqrt_price_x96, price, spacing, d0, d1, levels=LEVELS
):
    """MCI_I (mean), Buy_MCI_I, Sell_MCI_I and MCI_imbalance_I for each depth I.
    When both sides are 0, the MCI is not defined, and the four values are NaN."""
    table = table.reset_index(drop=True)
    args = (quote_is_token1, sqrt_price_x96, price, spacing, d0, d1)
    names = ("MCI", "Buy_MCI", "Sell_MCI", "MCI_imbalance")
    out = {}

    for i in levels:
        b = mci_side(table, i, True, *args)
        s = mci_side(table, i, False, *args)

        if b or s:
            values = ((b + s) / 2, b, s, (b - s) / (b + s))
        else:
            values = (float("nan"),) * 4

        out |= {f"{k}_{i}": v for k, v in zip(names, values, strict=True)}

    return out


# --------------------------------------------------------------------------------------
# TVL as LP principal
# --------------------------------------------------------------------------------------
# Pool TVL as LP principal: the tokens in the LPs' positions, rebuilt from Mint and Burn
# events. Fees are not included.
#
# Paper: Section III.C, "TVL in USD", and its footnote. A position is (owner, tickLower,
# tickUpper). Its liquidity L is the sum of its Mint (+L) and Burn (-L) events up to the
# block. At the pool price, a range holds (sa, sb, sp: square-root prices of the lower
# tick, the upper tick and the pool):
#     below the range  sp <= sa       a0 = L (sb - sa) / (sa sb)   a1 = 0
#     in the range     sa < sp < sb   a0 = L (sb - sp) / (sp sb)   a1 = L (sp - sa)
#     above the range  sp >= sb       a0 = 0                       a1 = L (sb - sa)
# TVL in stablecoin units = stablecoin amount + other amount x pool price. TVL in USD =
# that x the Chainlink price of the stablecoin at the same block.


def range_amounts(L, sa, sb, sp):
    """Raw token amounts (a0, a1) of ranges with liquidity L at the pool square-root
    price sp (numpy arrays)."""
    below, above = sp <= sa, sp >= sb

    a0 = np.where(
        below,
        L * (sb - sa) / (sa * sb),
        np.where(above, 0.0, L * (sb - sp) / (sp * sb)),
    )

    a1 = np.where(below, 0.0, np.where(above, L * (sb - sa), L * (sp - sa)))

    return a0, a1


def sqrt_price_from_price(price, quote_is_token1, decimals0, decimals1):
    """Raw square-root price (token1 per token0) from the paired token's price in
    stablecoin units."""
    p01 = np.where(quote_is_token1, price, 1 / np.asarray(price, dtype=float))

    return np.sqrt(p01 * 10.0 ** (decimals1 - decimals0))


def principal_at(events, blocks, sqrt_prices):
    """Raw principal (amount0, amount1) of one pool at each block (sorted). Uses the
    events with block <= it. Also returns how many events left some position with
    negative liquidity (0 for a full event history)."""
    ev = events.sort_values(["block", "log_index"])
    keys = list(zip(ev.tick_lower, ev.tick_upper, strict=True))
    ranges = {k: r for r, k in enumerate(dict.fromkeys(keys))}

    sa = 1.0001 ** (np.array([k[0] for k in ranges], dtype=float) / 2)
    sb = 1.0001 ** (np.array([k[1] for k in ranges], dtype=float) / 2)

    ev_block = ev.block.to_numpy()
    ev_owner = ev.owner.to_numpy()
    ev_delta = [int(x) for x in ev.liquidity_delta]

    # Liquidity of each range as exact integers, and as floats for the amounts
    L_int = [0] * len(ranges)
    L_float = np.zeros(len(ranges))

    # Liquidity of each position (owner, range), and the count of negative positions
    pos = {}
    neg = 0

    p0 = np.full(len(blocks), np.nan)
    p1 = np.full(len(blocks), np.nan)
    j = 0

    for i, (blk, sp) in enumerate(zip(blocks, sqrt_prices, strict=True)):
        # Apply the events up to this block.
        while j < len(ev) and ev_block[j] <= blk:
            r = ranges[keys[j]]
            k = (ev_owner[j], keys[j])

            pos[k] = pos.get(k, 0) + ev_delta[j]
            neg += pos[k] < 0

            L_int[r] += ev_delta[j]
            L_float[r] = float(L_int[r])
            j += 1

        if np.isfinite(sp) and sp > 0:
            a0, a1 = range_amounts(L_float, sa, sb, sp)
            p0[i], p1[i] = a0.sum(), a1.sum()

    return p0, p1, neg


def owner_principal_at(events, block, sqrt_price):
    """Raw principal of each Mint owner at one block: DataFrame (owner, a0, a1,
    n_ranges). NFT positions have the NonfungiblePositionManager as owner; vaults have
    their own address."""
    ev = events[events.block <= block]
    position = ["owner", "tick_lower", "tick_upper"]

    L = ev.assign(L=ev.liquidity_delta.map(int)).groupby(position).L.sum()
    L = L[L > 0].reset_index()

    sa = 1.0001 ** (L.tick_lower.to_numpy(float) / 2)
    sb = 1.0001 ** (L.tick_upper.to_numpy(float) / 2)
    a0, a1 = range_amounts(L.L.map(float).to_numpy(), sa, sb, sqrt_price)

    amounts = pd.DataFrame({"owner": L.owner, "a0": a0, "a1": a1})
    totals = {"a0": ("a0", "sum"), "a1": ("a1", "sum"), "n_ranges": ("a0", "size")}

    return amounts.groupby("owner").agg(**totals).reset_index()


def tvl_stable_units(amount0, amount1, price, quote_is_token1):
    """TVL in stablecoin units from token amounts (token units) and the paired token's
    price in stablecoin units."""
    return np.where(
        quote_is_token1, amount0 * price + amount1, amount0 + amount1 * price
    )


# --------------------------------------------------------------------------------------
# Hourly panel
# --------------------------------------------------------------------------------------
# The paper's hourly pool panel: MCI and TVL of each pool-hour.
#
# MCI: from one tick snapshot per hour (the released tick dataset, or tick_table).
# TVL: the LP principal from Mint and Burn events at the snapshot block. The other token
# is valued at the snapshot's pool price. The USD value uses the Chainlink price of the
# stablecoin at the same block.

# The quote tokens of the paper
STABLE = ("USDC", "USDT")

# On-chain names -> the paper's names
NAMES = {"USD Coin": "USDC", "Tether USD": "USDT", "USDP": "PAX"}


def name(symbol):
    """Token symbol as the paper writes it (no NUL bytes; Paxos renamed PAX to USDP on
    chain)."""
    s = symbol.replace("\x00", "")

    return NAMES.get(s, s)


def mci_rows(snapshot, levels=LEVELS):
    """One row per pool of one snapshot: pool data, the deepest depth (max_level) and
    the MCI profile. The stablecoin is token1 if token1 is USDC or USDT, else token0 (so
    USDC/USDT pools quote in USDT)."""
    rows = []

    for pool, t in snapshot.groupby(snapshot.pool_address.str.lower()):
        t = t.sort_values("tickIdx").reset_index(drop=True)
        f = t.iloc[0]

        s0, s1 = name(f.token0_symbol), name(f.token1_symbol)
        d0, d1 = int(f.token0_decimals), int(f.token1_decimals)
        q1 = s1 in STABLE

        pool_data = {
            "pool_address": pool,
            "block": int(f.block_number),
            "trading_pair": f"{s0}{s1}{int(f.fee_tier)}",
            "stablecoin": s1 if q1 else s0,
            "quote_is_token1": q1,
            "token0_decimals": d0,
            "token1_decimals": d1,
            "price_in_stablecoin": float(f.price_in_stablecoin),
            "max_level": max_depth(len(t)),
        }

        profile = mci_profile(
            t,
            q1,
            int(f.sqrt_price_x96),
            f.price_in_stablecoin,
            int(f.tick_spacing),
            d0,
            d1,
            levels,
        )

        rows.append(pool_data | profile)

    return pd.DataFrame(rows)


def add_tvl(panel, events, prices):
    """Adds prin0 and prin1 (token units), tvl_stable (stablecoin units) and tvl_usd to
    each pool-hour. `events`: Mint and Burn events (collect.fetch_events). `prices`:
    Chainlink rows (collect.chainlink_usd) of every block in the panel. A pool with no
    event has a principal of 0."""
    ev = dict(tuple(events.groupby("pool_address")))
    parts = []

    for pool, g in panel.sort_values("block").groupby("pool_address"):
        d0, d1 = int(g.token0_decimals.iat[0]), int(g.token1_decimals.iat[0])
        price = g.price_in_stablecoin.to_numpy(float)
        sp = sqrt_price_from_price(price, g.quote_is_token1.to_numpy(), d0, d1)

        if pool in ev:
            p0, p1, _ = principal_at(ev[pool], g.block.to_numpy(), sp)
        else:
            p0, p1 = np.zeros(len(g)), np.zeros(len(g))

        parts.append(g.assign(prin0=p0 / 10**d0, prin1=p1 / 10**d1))

    out = pd.concat(parts)

    out["tvl_stable"] = tvl_stable_units(
        out.prin0, out.prin1, out.price_in_stablecoin, out.quote_is_token1
    )

    # Chainlink price of the stablecoin of each row, at the block of the row
    px = prices.drop_duplicates("block").set_index("block")
    rows = zip(out.block, out.stablecoin, strict=True)
    usd = [px.at[b, f"{s.lower()}_usd"] for b, s in rows]

    out["tvl_usd"] = out.tvl_stable * np.array(usd)

    return out.sort_index()


# --------------------------------------------------------------------------------------
# Command line: build the hourly panel
# --------------------------------------------------------------------------------------


def hours(path):
    """(hour, snapshot) pairs from one file or from a folder of hourly files."""
    if Path(path).is_dir():
        for f in sorted(Path(path).rglob("*.parquet")):
            yield pd.to_datetime(f.stem, format="%Y-%m-%d %H-%M-%S"), pd.read_parquet(f)
    else:
        # one row group at a time (48 hours)
        for part in fastparquet.ParquetFile(path).iter_row_groups():
            yield from part.groupby("datetime_utc")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    ap.add_argument("--ticks", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--chainlink", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = [mci_rows(snap).assign(hour=h) for h, snap in hours(a.ticks)]
    panel = add_tvl(
        pd.concat(rows, ignore_index=True), read(a.events), read(a.chainlink)
    )

    write(panel, a.out)
