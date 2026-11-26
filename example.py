"""Offline example on the small sample in data/sample (four pools of the paper, three
hours).

    python example.py

It (1) computes the MCI from the released tick tables, (2) rebuilds the tick tables from
Mint and Burn events, (3) computes the LP-principal TVL in USD, and (4) runs the DiD on
a four-pool panel. Steps 1 to 3 print the paper's values next to ours. Step 4 only shows
the call: four pools do not give the paper's estimate.
"""

from pathlib import Path

import pandas as pd

import did
import liquidity as liq

S = Path(__file__).parent / "data" / "sample"

snap = pd.read_parquet(S / "ticks.parquet")
ev = pd.read_csv(S / "events.csv")
chainlink = pd.read_csv(S / "chainlink.csv")
expected = pd.read_csv(S / "expected.csv", parse_dates=["hour"])

# (1) MCI and (3) TVL of each pool-hour, with the paper's values next to them.
key = ["hour", "pool_address"]
rows = [liq.mci_rows(g).assign(hour=h) for h, g in snap.groupby("datetime_utc")]
rows = liq.add_tvl(pd.concat(rows), ev, chainlink).set_index(key)

ours = rows[["trading_pair", "MCI_1", "MCI_5", "tvl_usd"]]
paper = expected.set_index(key)[["MCI_1", "MCI_5", "totalValueLockedUSD"]]

print(ours.join(paper, rsuffix="_paper").to_string(float_format="{:.6g}".format), "\n")

# (2) Tick table from events: the same liquidity per level as the released table.
last = snap[snap.datetime_utc == snap.datetime_utc.max()]

for pool, g in last.groupby("pool_address"):
    g = g.sort_values("tickIdx").reset_index(drop=True)
    f = g.iloc[0]

    nets = liq.nets_from_events(ev[ev.pool_address == pool], int(f.block_number))

    t = liq.tick_table(
        int(f.sqrt_price_x96),
        int(f.cross_tick),
        liq.active_liquidity(nets, int(f.cross_tick)),
        int(f.tick_spacing),
        nets,
        int(f.token0_decimals),
        int(f.token1_decimals),
    )

    same = t.liquidityActive.tolist() == g.liquidityActive.tolist()

    print(
        f"{f.trading_pair} levels: {len(t)} deepest depth: {liq.max_depth(len(t))}"
        f" same liquidity as the released table: {same}"
    )

# (4) The DiD call on four pools (the paper uses 155 pools: did.py).
p = pd.read_csv(S / "panel.csv.gz", parse_dates=["hour"])
cols = ["weighted", "depth", "beta", "se", "n_obs", "n_pools"]

print("\n", did.table(p, did.T1, levels=(1,))[cols].to_string(index=False))
