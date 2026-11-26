r"""The paper's difference-in-differences (DiD) of ln MCI (Section III.D).

eq. (17), Tables V and VI:
    ln MCI_mu,j,t = lambda_j + upsilon_t + beta (1[j in USDC] x 1[t >= tau]) + e_j,t
eq. (18), Table IV:
    ... + alpha (1[USDC] x 1[t < tau - 24h]) + tau_post (1[USDC] x 1[t >= tau]) + e_j,t
eq. (19), Figure 4:
    ... + sum_k beta_k (1[USDC] x 1[t in bin k]) + e_j,t, k = -1 the reference bin

Pool (lambda_j) and hour (upsilon_t) fixed effects, standard errors clustered by pool,
one estimate per depth I. The unweighted estimate is OLS. The weighted one (WLS) weights
each pool by its mean TVL in USD before tau. Input: an hourly panel (liquidity.py) with
the columns pool_address, hour, stablecoin, tvl_usd and MCI_I.

    python did.py --panel panel.parquet --pools data/paper_pools.csv
    python did.py --panel panel.parquet --pools data/paper_pools.csv \
        --exhibit pre-trends
    python did.py --panel panel.parquet --pools data/paper_pools.csv \
        --exhibit event-study

The first command gives Tables V and VI, the second Table IV and the third the
coefficients of Figure 4. The script estimates every depth I whose column MCI_I is in
the panel, for Treatment 1 and Treatment 2.
"""

import argparse

import numpy as np
import pandas as pd
from linearmodels import PanelOLS

from liquidity import LEVELS, read, write

# Treatment 1: Circle discloses its deposits at SVB
T1 = "2023-03-11 03:11"

# Treatment 2: Joint Statement of the Treasury, the Fed and the FDIC
T2 = "2023-03-12 22:24"

# Estimation window
START, END = "2023-03-01 00:00", "2023-03-31 00:00"

# Hours of the reference window of eqs. (18) and (19), and the bin width of eq. (19)
REF_HOURS = 24


def pools_by_depth(panel, levels=LEVELS):
    """{I: pools} of each depth I: the MCI at depth 1 and at depth I is defined (> 0) in
    every hour of the panel. The paper applies this rule to every hour from 28th
    February 00:00 to 1st April 00:00 UTC."""
    ok = panel[[f"MCI_{i}" for i in levels]].gt(0).groupby(panel.pool_address).all()

    return {i: set(ok.index[ok.MCI_1 & ok[f"MCI_{i}"]]) for i in levels}


def sample(panel, tau, depth, pools, drop_from=None, start=START, end=END):
    """The estimation sample of one depth: y = ln MCI_I, the USDC indicator and the
    weight w (the pool's mean TVL in USD before tau). `drop_from` removes the hours in
    [drop_from, tau). Treatment 2 uses it to drop the hours after Treatment 1."""
    h = panel.hour
    keep = panel.pool_address.isin(pools) & h.between(start, end)

    if drop_from:
        keep &= ~(h.ge(drop_from) & h.lt(tau))

    d = panel[keep]
    mci = d[f"MCI_{depth}"]

    d = d.assign(
        y=np.log(mci.where(mci > 0)), usdc=(d.stablecoin == "USDC").astype(int)
    )
    d = d.dropna(subset=["y"])

    # A pool with no hour before tau gets the median weight, as in the paper's
    # estimation code. No pool of the paper needs this.
    mean_tvl = d[d.hour < tau].groupby("pool_address").tvl_usd.mean()
    w = d.pool_address.map(mean_tvl)

    return d.assign(w=w.fillna(w.median()))


def fit(d, cols, weighted):
    """Two-way fixed effects (pool and hour) on the regressors `cols`, standard errors
    clustered by pool."""
    d = d.set_index(["pool_address", "hour"])

    model = PanelOLS(
        d.y,
        d[cols],
        entity_effects=True,
        time_effects=True,
        weights=d.w if weighted else None,
        drop_absorbed=True,
    )

    return model.fit(cov_type="clustered", cluster_entity=True)


def reference_end(d, tau):
    """End of the reference window of eqs. (18) and (19): tau, or the hour after the
    last hour before tau when the REF_HOURS hours before tau are dropped (Treatment 2),
    as in the paper's code."""
    tau = pd.Timestamp(tau)
    pre = d.hour[d.hour < tau]

    if (pre >= tau - pd.Timedelta(hours=REF_HOURS)).any():
        return tau

    return pre.max() + pd.Timedelta(hours=1)


def estimate(panel, tau, depth, pools, weighted=True, drop_from=None):
    """beta of equation (17) at one depth, with its standard error, p-value and sample
    size."""
    d = sample(panel, tau, depth, pools, drop_from)
    d = d.assign(treated=d.usdc * (d.hour >= tau))

    m = fit(d, ["treated"], weighted)
    b = m.params["treated"]
    usdc = d.groupby("pool_address").usdc.first()

    return {
        "depth": depth,
        "weighted": weighted,
        "beta": b,
        "se": m.std_errors["treated"],
        "p_value": m.pvalues["treated"],
        "pct_effect": (np.exp(b) - 1) * 100,
        "r2_within": m.rsquared_within,
        "n_obs": int(m.nobs),
        "n_pools": len(usdc),
        "n_usdc": int(usdc.sum()),
        "n_usdt": int((1 - usdc).sum()),
    }


def pre_trends(panel, tau, depth, pools, weighted=True, drop_from=None):
    """alpha and tau_post of equation (18) at one depth (Table IV). The reference window
    is the REF_HOURS hours before tau. The paper's reference window of Treatment 2 ends
    at the last hour before Treatment 1 and holds one hour more (reference_end)."""
    d = sample(panel, tau, depth, pools, drop_from)
    end = reference_end(d, tau)

    # one hour more for Treatment 2
    ref = end - pd.Timedelta(hours=REF_HOURS + (end != pd.Timestamp(tau)))

    d = d.assign(pre=d.usdc * (d.hour < ref), post=d.usdc * (d.hour >= tau))
    m = fit(d, ["pre", "post"], weighted)

    return {
        "depth": depth,
        "weighted": weighted,
        "alpha": m.params["pre"],
        "alpha_se": m.std_errors["pre"],
        "alpha_p": m.pvalues["pre"],
        "tau_post": m.params["post"],
        "tau_post_se": m.std_errors["post"],
        "tau_post_p": m.pvalues["post"],
        "n_obs": int(m.nobs),
        "n_pools": d.pool_address.nunique(),
    }


def event_study(panel, tau, depth, pools, weighted=True, drop_from=None):
    """beta_k of equation (19) at one depth (Figure 4): one coefficient per
    REF_HOURS-hour bin k relative to tau. Bin -1 is the reference window, with beta_-1 =
    0. The row of bin -1 is included for plots."""
    d = sample(panel, tau, depth, pools, drop_from)

    # Hours from tau count from tau. Earlier hours count from the end of the reference
    # window.
    at_tau = pd.Series(pd.Timestamp(tau), index=d.index)
    origin = at_tau.where(d.hour >= tau, reference_end(d, tau))

    k = np.floor((d.hour - origin) / pd.Timedelta(hours=REF_HOURS)).astype(int)
    cols = [f"k{b}" for b in sorted(set(k) - {-1})]

    d = d.assign(**{c: d.usdc * (k == int(c[1:])) for c in cols})
    m = fit(d, cols, weighted)

    out = pd.DataFrame(
        {
            "k": [-1] + [int(c[1:]) for c in cols],
            "beta": [0.0] + m.params[cols].tolist(),
            "se": [0.0] + m.std_errors[cols].tolist(),
            "p_value": [np.nan] + m.pvalues[cols].tolist(),
        }
    )

    return out.assign(
        ci_low=out.beta - 1.96 * out.se,
        ci_high=out.beta + 1.96 * out.se,
        depth=depth,
        weighted=weighted,
        n_obs=int(m.nobs),
        n_pools=d.pool_address.nunique(),
    )


def table(panel, tau, drop_from=None, levels=LEVELS, fn=estimate):
    """`fn` (estimate, pre_trends or event_study) at every depth, unweighted (Panel A)
    then weighted (Panel B)."""
    by_depth = pools_by_depth(panel, levels)
    group = panel.groupby("pool_address").stablecoin.first()

    # A depth needs USDC and USDT pools. Else the fixed effects absorb the treatment,
    # and the depth is skipped.
    levels = [i for i in levels if group[list(by_depth[i])].nunique() == 2]

    out = [
        fn(panel, tau, i, by_depth[i], weighted, drop_from)
        for weighted in (False, True)
        for i in levels
    ]

    if fn is event_study:
        return pd.concat(out, ignore_index=True)

    return pd.DataFrame(out)


EXHIBITS = {"did": estimate, "pre-trends": pre_trends, "event-study": event_study}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    ap.add_argument("--panel", required=True)

    ap.add_argument(
        "--pools", required=True, help="the paper's 155 pools (data/paper_pools.csv)"
    )

    ap.add_argument(
        "--exhibit",
        choices=EXHIBITS,
        default="did",
        help=(
            "did: eq. (17), Tables V and VI; pre-trends: eq. (18), Table IV; "
            "event-study: eq. (19), Figure 4"
        ),
    )

    ap.add_argument("--out", help="write all estimates to this CSV")
    a = ap.parse_args()

    p = read(a.panel)
    p = p[p.pool_address.isin(read(a.pools).pool_address)]
    p = p.assign(hour=pd.to_datetime(p.hour))

    levels = tuple(i for i in LEVELS if f"MCI_{i}" in p.columns)
    fn = EXHIBITS[a.exhibit]

    # Treatment 2 drops the hours between Treatment 1 and Treatment 2.
    t1 = table(p, T1, levels=levels, fn=fn).assign(treatment="T1")
    t2 = table(p, T2, drop_from=T1, levels=levels, fn=fn).assign(treatment="T2")

    res = pd.concat([t1, t2])
    res = res[["treatment", *res.columns[:-1]]]

    show = ("display.width", 200, "display.max_rows", 500)

    with pd.option_context(*show, "display.float_format", "{:.3f}".format):
        print(res.to_string(index=False))

    if a.out:
        write(res, a.out)
