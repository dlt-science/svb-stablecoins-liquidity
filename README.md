# Code for the paper:

> Hernandez Cruz, W., Xu, J., Tasca, P., & Campajola, C. (2024). *Impact of Adverse Disclosures on Liquidity of Stablecoin Markets*. arXiv:2407.11716.

---

## Overview

The paper measures how Circle's disclosure of USDC reserves at Silicon Valley Bank, at 03:11 UTC on 11th March 2023, changed the liquidity cost of Uniswap v3 stablecoin pools. A difference-in-differences (DiD) compares 119 USDC pools, the two DAI/USDC pools included, with 36 USDT pools at hourly frequency in March 2023, unweighted and weighted by total value locked (TVL). The liquidity cost is an adapted Marginal Cost of Immediacy (MCI) at depths of 1, 5, 10, 15 and 20 tick levels. The TVL is the principal of the liquidity providers (LPs) from Mint and Burn events, at the Chainlink price of the stablecoin and without fees. The code computes both for each pool-hour from public Ethereum data, and reruns the DiD, the parallel-trends test and the event study of the paper.

---

## Setup

The code needs Python 3.11 or later and no API key.

```bash
uv sync --extra data          # or: pip install -r requirements-lock.txt huggingface_hub
uv run python example.py      # a small sample, without a network
```

Three scripts hold the code. `collect.py` reads the data from Ethereum, `liquidity.py` computes the MCI and the TVL, and `did.py` runs the estimates.

---

## Data

Three Hugging Face datasets hold the data of the paper. Each link points to the revision `v1.0`, which gives the paper's numbers. The hourly files share one block per hour, from 00:00 UTC on 28th February 2023 to 00:00 UTC on 1st April 2023 (769 hours).

### [svb-stablecoins-hourly-tick-liquidity](https://huggingface.co/datasets/ExponentialScience/svb-stablecoins-hourly-tick-liquidity/tree/v1.0)

It holds the active liquidity of each tick range of 220 pools at each hour (`tick_liquidity_hourly.parquet`, 10,827,997 rows), the input of the MCI. `pools.csv` marks the 155 pools of the DiD.

### [svb-stablecoins-hourly-lp-positions](https://huggingface.co/datasets/ExponentialScience/svb-stablecoins-hourly-lp-positions/tree/v1.0)

It holds the NFT positions of each LP by owner wallet, pool and hour (`lp_owner_positions_hourly.parquet`, 21,097,300 rows) and the principal of every Mint owner, vaults included (`hourly_owner_principal.parquet`). The paper uses them for the LP counts, the Gini coefficients and the LP figures.

### [svb-stablecoins-replication-data](https://huggingface.co/datasets/ExponentialScience/svb-stablecoins-replication-data/tree/v1.0)

It holds the small inputs of the paper. These are the Mint and Burn events and the LP principal (`pools/principal/`), the Chainlink prices (`pools/stable_price/`), the pool balances (`pools/balances/`), the hourly MCI panel of the 155 pools (`panel/panel.parquet`) and the TerraUSD, multichain, Polygon and Arbitrum panels.

```python
import pandas as pd
from huggingface_hub import hf_hub_download

repo = "ExponentialScience/svb-stablecoins-replication-data"
path = hf_hub_download(
    repo, "panel/panel.parquet", repo_type="dataset", revision="v1.0"
)
panel = pd.read_parquet(path)
```

`python did.py --panel <path> --pools data/paper_pools.csv` reruns the DiD on this panel (`--exhibit did`, `pre-trends` or `event-study`).

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
