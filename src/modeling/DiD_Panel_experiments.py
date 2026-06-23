"""Main DiD estimation — pure two-way FE DiD without volatility covariate.

Treatment 1 (MAIN):     Circle's tweet (2023-03-11 03:11 UTC)
Treatment 2 (MAIN):     Joint Statement (2023-03-12 22:24 UTC)
Placebo anchor:         Chapter 11 filing (2023-03-17 13:00 UTC)

Run: uv run python -m src.modeling.DiD_Panel_experiments_v3
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from linearmodels import PanelOLS
from matplotlib.lines import Line2D
from tqdm import tqdm

from src.modeling._common import (
    BASELINE_END,
    BASELINE_START,
    CHAPTER_11,
    CIRCLE_TWEET,
    DEP_VARS,
    IND_VARS_NO_VOL,
    JOINT_STATEMENT,
    LATEX_IMG_DIR,
    LEVELS,
    OUT_DIR,
    PLOTS_DIR,
    carve_treatment,
    did_row,
    ensure_dirs,
    fit_did,
    level_subset,
    load_panel,
    n_pools_by_group,
    pools_by_level,
    pre_treatment_weights,
)


# ---- Event study (dynamic DiD) — no volatility covariate ---------------
def event_study_df(df: pd.DataFrame, treatment_date: str, freq_hours: int = 24) -> pd.DataFrame:
    """Period-dummy-by-group dynamic DiD coefficients, reference = -1.
    Pure specification: no volatility covariate.
    """
    df = df.copy()
    t = pd.Timestamp(treatment_date)
    df["_period"] = (((df["date"] - t).dt.total_seconds() / 3600) // freq_hours).astype(int)

    periods = sorted(df["_period"].unique())
    periods_no_ref = [p for p in periods if p != -1]
    col_of = lambda p: (f"pXg_n{abs(p)}" if p < 0 else f"pXg_p{p}")

    for p in periods_no_ref:
        df[col_of(p)] = ((df["_period"] == p) & (df["group"] == 1)).astype(int)

    # Pure DiD: no volatility term
    rhs = "EntityEffects + TimeEffects + " + " + ".join(col_of(p) for p in periods_no_ref)
    rows = []
    for dep in DEP_VARS:
        panel = df.dropna(subset=[dep]).set_index(["trading_pair", "date"])
        res = PanelOLS.from_formula(f"{dep} ~ {rhs}", data=panel, drop_absorbed=True).fit(
            cov_type="clustered", cluster_entity=True
        )
        rows.append({"period": -1, "dep_var": dep, "coef": 0.0, "se": 0.0, "ci_lower": 0.0, "ci_upper": 0.0})
        for p in periods_no_ref:
            c, s = res.params.get(col_of(p), np.nan), res.std_errors.get(col_of(p), np.nan)
            if np.isfinite(c):
                rows.append({
                    "period": p, "dep_var": dep, "coef": c, "se": s,
                    "ci_lower": c - 1.96 * s, "ci_upper": c + 1.96 * s,
                })
    return pd.DataFrame(rows)


def plot_event_study(es_df: pd.DataFrame, out_paths: list, freq_hours: int = 24) -> None:
    """Stacked-panel coefficient plot (one panel per depth). Saves to each out_paths entry."""
    sns.set_context("paper", font_scale=0.85)
    sns.set_style("ticks")

    dep_vars = list(es_df["dep_var"].unique())
    fig, axs = plt.subplots(len(dep_vars), 1, figsize=(3.5, 0.70 * len(dep_vars) + 0.6), sharex=True)
    axs = np.atleast_1d(axs)

    coef_c, treat_c = "#2775CA", "#ff3a47"
    for ax, dv in zip(axs, dep_vars):
        sub = es_df[es_df["dep_var"] == dv].sort_values("period")
        ax.axhline(0, color="gray", lw=0.4)
        ax.axvline(-0.5, color=treat_c, ls="--", lw=0.7, alpha=0.8)
        ax.fill_between(sub["period"], sub["ci_lower"], sub["ci_upper"], color=coef_c, alpha=0.12)
        ax.plot(sub["period"], sub["coef"], color=coef_c, lw=1.0, marker="o", ms=2, zorder=3)
        ax.annotate(dv.split("_")[-1], xy=(1.01, 0.5), xycoords="axes fraction",
                    fontsize=7, va="center", ha="left", fontweight="bold", color="#555")
        ax.tick_params(labelsize=6, pad=2, length=2)
        ax.grid(True, axis="y", alpha=0.2, lw=0.3)
        sns.despine(ax=ax)

    axs[-1].set_xlabel(f"Periods relative to treatment ({freq_hours}h)", fontsize=7)
    fig.text(0.02, 0.5, r"$\hat{\beta}_{period}$", va="center", rotation="vertical",
             fontsize=8, fontweight="bold")
    axs[0].legend(
        handles=[Line2D([0], [0], color=coef_c, lw=1.0, marker="o", ms=2, label="Coefficient"),
                 Line2D([0], [0], color=treat_c, ls="--", lw=0.7, label="Treatment")],
        loc="upper center", bbox_to_anchor=(0.5, 1.4), ncol=2, frameon=False,
        fontsize=6, handlelength=1.5, columnspacing=0.8,
    )
    fig.subplots_adjust(hspace=0.15, left=0.15, right=0.88, top=0.90, bottom=0.12)

    for path in out_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"saved {path}")
    plt.close(fig)


# ---- Full DiD sweep across depths — pure spec (no volatility) ----------
def run_sweep(df: pd.DataFrame, treatment_date: str, pools_per_level, label: str,
              weighted: bool = True) -> pd.DataFrame:
    rows = []
    w_full = pre_treatment_weights(df) if weighted else None
    for level in tqdm(LEVELS, desc=label):
        sub = level_subset(df, level, pools_per_level)
        n_usdc, n_usdt, n_total = n_pools_by_group(sub)
        w = w_full.reindex(sub.set_index(["trading_pair", "date"]).index) if w_full is not None else None
        m = fit_did(f"MCI_{level}", sub, independents=IND_VARS_NO_VOL, weights=w)
        rows.append(did_row(m, f"MCI_{level}", treatment_date, n_total,
                            extras={"n_usdc": n_usdc, "n_usdt": n_usdt}))
    return pd.DataFrame(rows)


# ---- Main ---------------------------------------------------------------
def main() -> None:
    ensure_dirs()
    ppl = pools_by_level()
    for l in LEVELS:
        print(f"  level {l}: {len(ppl[l])} pools")

    df = load_panel()

    # Treatment 1 — Circle's tweet
    df_t1 = carve_treatment(df, CIRCLE_TWEET)
    r1 = run_sweep(df_t1, CIRCLE_TWEET, ppl, label="T1 Circle tweet")
    r1.to_csv(OUT_DIR / "did_v3_treat1.csv", index=False)
    r1_u = run_sweep(df_t1, CIRCLE_TWEET, ppl, label="T1 unweighted", weighted=False)
    r1_u.to_csv(OUT_DIR / "did_v3_treat1_unweighted.csv", index=False)

    es1 = event_study_df(df_t1, CIRCLE_TWEET)
    es1.to_csv(OUT_DIR / "event_study_v3_treat1.csv", index=False)
    plot_event_study(
        es1,
        [PLOTS_DIR / "event_study_v3_treat1.pdf",
         LATEX_IMG_DIR / "event_study_treat1.pdf",
         LATEX_IMG_DIR / "event_study_B_treat1.pdf"],
    )

    # Treatment 2 (MAIN) — Joint Statement, excluding inter-treatment window
    df_t2 = carve_treatment(df, JOINT_STATEMENT, exclude_before=CIRCLE_TWEET)
    r2 = run_sweep(df_t2, JOINT_STATEMENT, ppl, label="T2 Joint Statement")
    r2.to_csv(OUT_DIR / "did_v3_treat2_jointstmt.csv", index=False)
    r2_u = run_sweep(df_t2, JOINT_STATEMENT, ppl, label="T2 unweighted", weighted=False)
    r2_u.to_csv(OUT_DIR / "did_v3_treat2_jointstmt_unweighted.csv", index=False)

    es2 = event_study_df(df_t2, JOINT_STATEMENT)
    es2.to_csv(OUT_DIR / "event_study_v3_treat2_jointstmt.csv", index=False)
    plot_event_study(
        es2,
        [PLOTS_DIR / "event_study_v3_treat2_jointstmt.pdf",
         LATEX_IMG_DIR / "event_study_B_treat2.pdf"],
    )

    # Placebo — Chapter 11 anchor, excluding the same inter-treatment window
    df_pl = carve_treatment(df, CHAPTER_11, exclude_before=CIRCLE_TWEET)
    rpl = run_sweep(df_pl, CHAPTER_11, ppl, label="Placebo Chapter 11")
    rpl.to_csv(OUT_DIR / "did_v3_placebo_chapter11.csv", index=False)
    rpl_u = run_sweep(df_pl, CHAPTER_11, ppl, label="Placebo unweighted", weighted=False)
    rpl_u.to_csv(OUT_DIR / "did_v3_placebo_chapter11_unweighted.csv", index=False)

    espl = event_study_df(df_pl, CHAPTER_11)
    espl.to_csv(OUT_DIR / "event_study_v3_placebo_chapter11.csv", index=False)
    plot_event_study(
        espl,
        [PLOTS_DIR / "event_study_v3_placebo_chapter11.pdf",
         LATEX_IMG_DIR / "event_study_treat2.pdf"],
    )

    # Compact console summary
    print("\n" + "=" * 72)
    print(f"{'Anchor':<20} {'Depth':<7} {'beta3':>8} {'pct':>8} {'p':>8} {'N':>6}")
    print("-" * 72)
    for name, rdf in [("T1 Circle", r1), ("T2 JointStmt", r2), ("Placebo Ch.11", rpl)]:
        for _, row in rdf.iterrows():
            print(f"{name:<20} {row['dep_var']:<7} "
                  f"{row['beta3']:>8.4f} {row['pct_effect']:>7.1f}% "
                  f"{row['p_value']:>8.4g} {row['n_entities']:>6}")


if __name__ == "__main__":
    main()
