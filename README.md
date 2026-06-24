# Code for the paper:

> Hernandez Cruz, W., Xu, J., Tasca, P., & Campajola, C. (2024). *Impact of Adverse Disclosures on Liquidity of Stablecoin Markets*. arXiv:2407.11716.

---

## Overview

Measures how Circle's disclosure of USDC reserves held at Silicon Valley Bank affected liquidity costs in Uniswap v3 stablecoin pools. Uses a difference-in-differences design comparing 114 USDC pools (treatment) against 37 USDT pools (control) across March 2023.

Cross-exchange validation corroborates the main finding across two architecturally distinct protocols. The Curve Finance 3pool (StableSwap invariant) saw a 26.2% TVL decline and USDC's share spike from 34% to 88% on 11 March. On Uniswap v2 (constant-product $xy=k$ AMM), USDC-paired pools fell 12.7% on the same day while USDT-paired pools rose. The paper also examines the TerraUST collapse of May 2022 on Uniswap v3 as a reverse flight-to-safety episode, where capital flowed *toward* USDC when its reserves were not in question.

---

## Setup

Requires Python ≥ 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
source .venv/bin/activate
```

### API keys

Create a `.env` file with:

```
THEGRAPH_API_KEY=...       # The Graph (Uniswap v3 subgraph)
ETHERSCAN_API_KEY=...      # Etherscan (block lookups)
```

---

## Data

The datasets are published on HuggingFace. Both cover 153 Uniswap v3 pools (151 DiD pools + 2 USDC/USDT reference pools) at hourly frequency across March 2023.

### [svb-stablecoins-hourly-tick-liquidity](https://huggingface.co/datasets/ExponentialScience/svb-stablecoins-hourly-tick-liquidity)

Hourly tick-level liquidity distributions for all pools. Each record contains the reconstructed active liquidity $L_i$ at initialized tick bands around the current price, queried from the Uniswap v3 subgraph at the historical block number for that hour. These distributions are the direct input to the Marginal Cost of Immediacy (MCI) computations at depths $I \in \{1, 5, 10, 15, 20\}$ ticks around the active tick.

### [svb-stablecoins-hourly-lp-positions](https://huggingface.co/datasets/ExponentialScience/svb-stablecoins-hourly-lp-positions)

Hourly LP position records for all pools. Each record is one liquidity provider (identified by owner address) in one pool at one hour, with their liquidity value $L_i$, tick bounds $[P_i, P_{i+1}]$, computed token amounts $X_\text{real}$ and $Y_\text{real}$, and USD-normalized TVL share. Used for LP churn, Gini coefficient concentration, and cohort-level capital flow analyses in the paper.

Download with the [HuggingFace `datasets` library](https://huggingface.co/docs/datasets):

```python
from datasets import load_dataset

ticks = load_dataset("ExponentialScience/svb-stablecoins-hourly-tick-liquidity")
positions = load_dataset("ExponentialScience/svb-stablecoins-hourly-lp-positions")
```

---

## Cite

```bibtex
@misc{cruz2024questionsaskedeffectstransparency,
  title        = {Impact of Adverse Disclosures on Liquidity of Stablecoin Markets},
  author       = {Walter Hernandez Cruz and Jiahua Xu and Paolo Tasca and Carlo Campajola},
  year         = {2024},
  eprint       = {2407.11716},
  archivePrefix = {arXiv},
  primaryClass = {q-fin.TR},
  url          = {https://arxiv.org/abs/2407.11716}
}
```
