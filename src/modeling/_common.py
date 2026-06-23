"""Shared helpers for v2 DiD / robustness scripts.

v2 framing (paper revision):
    Treatment 1 = Circle's tweet (11 March 2023 03:11 UTC)
    Treatment 2 = Joint Statement (12 March 2023 22:24 UTC)       [MAIN]
    Placebo     = Chapter 11 of SVB Financial Group (17 March 2023 13:00 UTC)

All v2 scripts import from here rather than re-declaring loading, pool
filtering, weighting, and DiD fit helpers. Run with `uv run python -m
src.modeling.<script_v2>`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels import PanelOLS

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data.collection.pools_config import get_active_pools  # noqa: E402

# ---- Constants ----------------------------------------------------------
LEVELS = [1, 5, 10, 15, 20]
DEP_VARS = [f"MCI_{i}" for i in LEVELS]
IND_VARS = ["treatment", "group", "treatment_interaction", "volatility"]
IND_VARS_NO_VOL = ["treatment", "group", "treatment_interaction"]

CIRCLE_TWEET = "2023-03-11 03:11:00"
JOINT_STATEMENT = "2023-03-12 22:24:00"
CHAPTER_11 = "2023-03-17 13:00:00"
BASELINE_START = "2023-03-01"
BASELINE_END = "2023-03-31"

POOLS_AVOID = ["USDCUSDT100", "USDCUSDT500", "USDDUSDT100"]

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "filtered"
PLOTS_DIR = ROOT / "plots" / "did"
LATEX_IMG_DIR = ROOT / "latex" / "figures" / "images"
TICK_AVAIL = OUT_DIR / "tick_level_availability.parquet"
PARQUET = OUT_DIR / "mci_detailed_summarized.parquet"


# ---- Pool filtering by tick-level availability --------------------------
def pools_by_level(tick_avail_path: Path = TICK_AVAIL) -> dict[int, set[str]]:
    """Return {level: {trading_pair}} — pools with sufficient tick depth at each level."""
    avail = pd.read_parquet(tick_avail_path)
    always_ok = avail.groupby("trading_pair")["reason_mci1_zero"].apply(lambda g: (g == "ok").all())
    level1 = set(always_ok[always_ok].index)
    min_max = avail.groupby("trading_pair")["max_level_tick_check"].min()
    return {1: level1} | {l: set(min_max[min_max >= l].index) & level1 for l in LEVELS[1:]}


# ---- Volatility helper --------------------------------------------------
def rolling_rv(x: pd.Series, window: int = 24) -> pd.Series:
    """Annualised realised volatility from squared log returns, 24 h rolling window.

    Formula: σ_{i,t} = sqrt( (8760/n) × Σ r²_{i,t-k+1} ),  r_{i,t} = ln(P_{i,t}/P_{i,t-1})
    Follows Heimbach, Schertenleib & Wattenhofer (2022, AFT '22), Section 5.2.
    8760 = 24 × 365 annualises hourly variance to an annual volatility figure.
    """
    log_ret = np.log(x / x.shift(1))                              # r_t = ln(P_t / P_{t-1})
    rv_sum  = log_ret.pow(2).rolling(window=window, min_periods=window // 2).sum()
    return np.sqrt(rv_sum * (8760 / window))                       # annualise: 8760 h / yr


# ---- Data loading (with log-transforms + volatility) --------------------
def load_panel(parquet_path: Path = PARQUET, pools_filter: set[str] | None = None) -> pd.DataFrame:
    """Load the hourly panel with log(MCI), log(volatility), and raw TVL column."""
    df = pd.read_parquet(parquet_path, engine="fastparquet")
    df["trading_pair"] = df["token0_symbol"] + df["token1_symbol"] + df["fee_tier"].astype(str)

    active = set(get_active_pools().values())
    keep = active if pools_filter is None else active & pools_filter
    df = df[df["trading_pair"].isin(keep) & ~df["trading_pair"].isin(POOLS_AVOID)]

    cols = ["date", "stablecoin", "trading_pair", "price_in_stablecoin", "totalValueLockedUSD"] + DEP_VARS
    df = df[cols].copy()
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df = df.sort_values("date").reset_index(drop=True)

    df["tvl_raw"] = df["totalValueLockedUSD"]
    df["totalValueLockedUSD"] = np.log(df["totalValueLockedUSD"])

    vol = df.groupby("trading_pair")["price_in_stablecoin"].transform(rolling_rv)
    vol = vol.replace(0, np.nan)                                 # guard against zero RV
    df["volatility"] = np.log(vol).replace(-np.inf, np.nan).ffill().bfill()

    for c in DEP_VARS:
        df[c] = np.log(df[c])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["volatility"])

    df["group"] = (df["stablecoin"] == "USDC").astype(int)
    return df


# ---- Per-treatment sample carving --------------------------------------
def carve_treatment(df: pd.DataFrame, treat_date: str,
                    exclude_before: str | None = None,
                    start: str = BASELINE_START, end: str = BASELINE_END) -> pd.DataFrame:
    """Slice a single-treatment DiD sample.

    If `exclude_before` is given, the window [exclude_before, treat_date)
    is cut so the pre-treatment baseline stays clean.
    """
    out = df.copy()
    if exclude_before is not None:
        out = out[~((out["date"] >= exclude_before) & (out["date"] < treat_date))]
    out = out[(out["date"] >= start) & (out["date"] <= end)].copy()
    out["treatment"] = (out["date"] >= treat_date).astype(int)
    out["treatment_interaction"] = out["treatment"] * out["group"]
    return out


# ---- Pre-treatment TVL weights -----------------------------------------
def pre_treatment_weights(df: pd.DataFrame, treatment_col: str = "treatment") -> pd.Series:
    """Map each (pool, date) observation to its pool's pre-treatment mean TVL."""
    avg = df[df[treatment_col] == 0].groupby("trading_pair")["tvl_raw"].mean()
    w = df.set_index(["trading_pair", "date"])["tvl_raw"].copy()
    w[:] = np.nan
    for tp, val in avg.items():
        w.loc[tp] = val
    return w.fillna(w.median())


# ---- DiD estimator ------------------------------------------------------
def fit_did(dep: str, df: pd.DataFrame, independents: list[str] = IND_VARS,
            weights: pd.Series | None = None):
    """Two-way FE PanelOLS with entity-clustered SEs."""
    panel = df.set_index(["trading_pair", "date"])
    formula = f"{dep} ~ EntityEffects + TimeEffects + " + " + ".join(independents)
    kw = {"data": panel, "drop_absorbed": True}
    if weights is not None:
        kw["weights"] = weights.reindex(panel.index)
    return PanelOLS.from_formula(formula, **kw).fit(cov_type="clustered", cluster_entity=True)


def did_row(model, dep: str, treatment_date: str, n_pools: int, extras: dict | None = None) -> dict:
    """Flatten a fitted PanelOLS into a result row."""
    b = model.params.get("treatment_interaction", np.nan)
    row = {
        "dep_var": dep,
        "treatment_date": treatment_date,
        "beta3": b,
        "pct_effect": (np.exp(b) - 1) * 100,
        "p_value": model.pvalues.get("treatment_interaction", np.nan),
        "se": model.std_errors.get("treatment_interaction", np.nan),
        "r2_within": model.rsquared_within if hasattr(model, "rsquared_within") else model.rsquared,
        "n_entities": n_pools,
        "n_obs": int(model.nobs),
    }
    return row | (extras or {})


# ---- Filtering helpers --------------------------------------------------
def level_subset(df: pd.DataFrame, level: int, pools_per_level: dict[int, set[str]] | None,
                 extra_pools: set[str] | None = None) -> pd.DataFrame:
    """Return the subset of `df` valid at depth `level` (and inside `extra_pools` if given)."""
    dep = f"MCI_{level}"
    sub = df.dropna(subset=[dep])
    if pools_per_level and level in pools_per_level:
        keep = pools_per_level[level]
        if extra_pools is not None:
            keep = keep & extra_pools
        sub = sub[sub["trading_pair"].isin(keep)]
    elif extra_pools is not None:
        sub = sub[sub["trading_pair"].isin(extra_pools)]
    return sub


def n_pools_by_group(df: pd.DataFrame) -> tuple[int, int, int]:
    """(USDC count, USDT count, total) in a filtered sample."""
    n_usdc = df[df["group"] == 1]["trading_pair"].nunique()
    n_usdt = df[df["group"] == 0]["trading_pair"].nunique()
    return n_usdc, n_usdt, n_usdc + n_usdt


# ---- Output paths -------------------------------------------------------
def ensure_dirs() -> None:
    for d in (OUT_DIR, PLOTS_DIR, LATEX_IMG_DIR):
        d.mkdir(parents=True, exist_ok=True)
