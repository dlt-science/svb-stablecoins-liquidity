from web3 import Web3
from web3.exceptions import Web3Exception
from tqdm import tqdm
import pandas as pd
import argparse
import time
import os
import asyncio
import logging
from collections import namedtuple
from typing import Dict, List, Tuple
from decimal import Decimal, getcontext, ROUND_DOWN, ROUND_HALF_UP, ROUND_UP, InvalidOperation
from tqdm.asyncio import tqdm_asyncio
from requests.exceptions import ConnectionError, Timeout, HTTPError

# Adding the path to be able to import the analytics module
import sys

# sys.path.append('./../../../')

# ---------------------------------------------------------------------------
# Path setup so imports work when running from src/data/processing/
# ---------------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# ---------------------------------------------------------------------------
# Logging setup — logs to stdout (captured by nohup) with timestamps
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# Force stdout to flush immediately so nohup captures lines in real-time
sys.stdout.reconfigure(line_buffering=True)
logger = logging.getLogger(__name__)

from src.data.collection.etherscan import get_block_by_timestamp
from src.data.collection.uniswapv3_async import get_initialized_ticks, get_pool_volume_in_block_async
from src.data.collection.ticksProcessing import create_bar_chart_ticks, get_token_amounts, Q96, TICK_BASE
from src.data.collection.pools_config import get_active_pools, STABLECOINS

# Load the environment variables
from dotenv import load_dotenv

load_dotenv()

# Set precision for Decimal operations (setting high enough for high accuracy in financial calculations)
getcontext().prec = 34  # Set precision to 100 decimal places
getcontext().rounding = ROUND_UP  # Set rounding method to down to match financial rounding

POOLS_AND_TRADING_PAIRS = get_active_pools()

# ---------------------------------------------------------------------------
# Free, archive-capable Ethereum RPC endpoints (no API key required).
# These support eth_call at historical blocks, which is needed for querying
# pool state at past block numbers (~Feb-Mar 2023).
# Order matters: fastest / most reliable first.
# ---------------------------------------------------------------------------
FREE_RPC_URLS = [
    "https://eth.drpc.org",
    "https://rpc.ankr.com/eth",
    "https://ethereum-rpc.publicnode.com",
    "https://1rpc.io/eth",
    "https://eth.llamarpc.com",
    "https://cloudflare-eth.com",
]

# Errors that indicate the RPC endpoint is down or rate-limited (should failover)
RPC_RETRYABLE_ERRORS = (
    ConnectionError, Timeout, HTTPError, OSError,
    Web3Exception,
)


def _build_rpc_list():
    """Build the ordered list of RPC URLs to try.

    Priority:
      1. ALCHEMY_RPC_URL / INFURA_RPC_URL from env (if set) — paid, fastest
      2. Free public archive RPCs
    """
    urls = []
    for env_var in ("ALCHEMY_RPC_URL_1", "ALCHEMY_RPC_URL", "ANKR_RPC_URL", "QUICKNODE_RPC_URL",
                    "INFURA_RPC_URL", "INFURA_RPC_URL_2"):
        val = os.getenv(env_var)
        if val:
            urls.append(val)
    urls.extend(FREE_RPC_URLS)
    return urls


class FallbackWeb3:
    """Thin wrapper around Web3 that automatically rotates RPC providers on failure.

    Usage is identical to a normal Web3 instance — attribute access is proxied
    to the underlying ``Web3`` object.  When any RPC call raises a retryable
    error the wrapper switches to the next URL in the list and retries once.

    The wrapper also keeps basic per-URL latency stats so that, on each new
    ``rotate()`` call, it prefers the URL with the lowest average latency
    (excluding URLs that have failed recently).
    """

    def __init__(self, rpc_urls=None, request_timeout=30):
        self._urls = rpc_urls or _build_rpc_list()
        self._request_timeout = request_timeout
        self._current_idx = 0
        # Track failures: url -> timestamp of last failure
        self._failures: Dict[str, float] = {}
        # Track latency: url -> list of response times (last 20)
        self._latencies: Dict[str, list] = {u: [] for u in self._urls}
        self._web3 = self._connect(self._urls[self._current_idx])
        logger.info(f"[RPC] Connected to {self._urls[self._current_idx]} ({len(self._urls)} providers available)")

    def _connect(self, url):
        provider = Web3.HTTPProvider(url, request_kwargs={"timeout": self._request_timeout})
        return Web3(provider)

    @property
    def current_url(self):
        return self._urls[self._current_idx]

    def rotate(self, failed_url=None):
        """Switch to the next available RPC URL.

        Marks *failed_url* as recently failed so it is deprioritised.
        Returns True if a new provider was activated, False if all URLs exhausted.
        """
        if failed_url:
            self._failures[failed_url] = time.time()

        # Try URLs in order, skipping those that failed in the last 60s
        now = time.time()
        candidates = [
            (i, u) for i, u in enumerate(self._urls)
            if u != failed_url and (u not in self._failures or now - self._failures[u] > 60)
        ]

        if not candidates:
            # All URLs failed recently — reset failures and try the full list
            self._failures.clear()
            candidates = [(i, u) for i, u in enumerate(self._urls)]

        # Pick the candidate with the lowest average latency (or first if no data)
        def _avg_latency(url):
            lat = self._latencies.get(url, [])
            return sum(lat) / len(lat) if lat else float("inf")

        candidates.sort(key=lambda c: _avg_latency(c[1]))
        self._current_idx = candidates[0][0]
        self._web3 = self._connect(self._urls[self._current_idx])
        logger.warning(f"[RPC] Rotated to {self._urls[self._current_idx]} (idx={self._current_idx})")
        return True

    def record_latency(self, url, elapsed):
        """Record a successful call's latency for the given URL."""
        bucket = self._latencies.setdefault(url, [])
        bucket.append(elapsed)
        if len(bucket) > 20:
            bucket.pop(0)

    def __getattr__(self, name):
        """Proxy attribute access to the underlying Web3 instance."""
        return getattr(self._web3, name)


def web3_call_with_fallback(fallback_web3, fn, *args, max_retries=3, **kwargs):
    """Execute a web3 contract call with automatic RPC failover.

    *fn* should be a callable that takes a ``web3`` instance as its first
    argument and returns the result.  Example::

        result = web3_call_with_fallback(fw3, lambda w3: w3.eth.get_block("latest"))
    """
    last_error = None
    for attempt in range(max_retries):
        url = fallback_web3.current_url
        t0 = time.time()
        try:
            result = fn(fallback_web3)
            fallback_web3.record_latency(url, time.time() - t0)
            return result
        except RPC_RETRYABLE_ERRORS as exc:
            elapsed = time.time() - t0
            last_error = exc
            logger.error(f"[RPC] Error on {url} after {elapsed:.1f}s: {exc.__class__.__name__}: {exc}")
            fallback_web3.rotate(failed_url=url)
        except Exception as exc:
            # Non-retryable error (e.g. contract revert) — raise immediately
            raise
    raise last_error

# Basic ERC20 ABI for balanceOf and decimals
ERC20_ABI = [
    {
        "inputs": [{"internalType": "address", "name": "", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "decimals",
        "outputs": [{"internalType": "uint8", "name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "symbol",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function"
    }
]

V3_ABI = [
    {
        "inputs": [],
        "name": "liquidity",
        "outputs": [{"internalType": "uint128", "name": "", "type": "uint128"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "slot0",
        "outputs": [
            {"internalType": "uint160", "name": "sqrtPriceX96", "type": "uint160"},
            {"internalType": "int24", "name": "tick", "type": "int24"},
            {"internalType": "uint16", "name": "observationIndex", "type": "uint16"},
            {"internalType": "uint16", "name": "observationCardinality", "type": "uint16"},
            {"internalType": "uint16", "name": "observationCardinalityNext", "type": "uint16"},
            {"internalType": "uint8", "name": "feeProtocol", "type": "uint8"},
            {"internalType": "bool", "name": "unlocked", "type": "bool"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "int24", "name": "", "type": "int24"}],
        "name": "ticks",
        "outputs": [
            {"internalType": "uint128", "name": "liquidityGross", "type": "uint128"},
            {"internalType": "int128", "name": "liquidityNet", "type": "int128"},
            {"internalType": "uint256", "name": "feeGrowthOutside0X128", "type": "uint256"},
            {"internalType": "uint256", "name": "feeGrowthOutside1X128", "type": "uint256"},
            {"internalType": "int56", "name": "tickCumulativeOutside", "type": "int56"},
            {"internalType": "uint160", "name": "secondsPerLiquidityOutsideX128", "type": "uint160"},
            {"internalType": "uint32", "name": "secondsOutside", "type": "uint32"},
            {"internalType": "bool", "name": "initialized", "type": "bool"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "token0",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "token1",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "fee",
        "outputs": [{"internalType": "uint24", "name": "", "type": "uint24"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "tickSpacing",
        "outputs": [{"internalType": "int24", "name": "", "type": "int24"}],
        "stateMutability": "view",
        "type": "function"
    }
]

TICK_VARIABLES = namedtuple("Tick",
                  "liquidityGross liquidityNet feeGrowthOutside0X128 feeGrowthOutside1X128 tickCumulativeOutside secondsPerLiquidityOutsideX128 secondsOutside initialized")

# Cache structure for token metadata
TokenMetadata = namedtuple('TokenMetadata', ['decimals', 'symbol'])
pool_metadata_cache: Dict[str, Tuple[TokenMetadata, TokenMetadata]] = {}


def get_position_info(contract, position_id, block_number=None):
    """Get detailed information about a specific position"""
    block_identifier = block_number if block_number is not None else 'latest'

    # Convert position_id to bytes32
    position_key = position_id.to_bytes(32, byteorder='big')

    position = contract.functions.positions(position_key).call(block_identifier=block_identifier)
    return {
        'liquidity': position[0],
        'fee_growth_inside0_last_x128': position[1],
        'fee_growth_inside1_last_x128': position[2],
        'tokens_owed0': position[3],
        'tokens_owed1': position[4]
    }

def calculate_tvl_from_balances(token0_contract, token1_contract, token0_decimals, token1_decimals,
                                token0_symbol, token1_symbol,
                                pool, price_in_stablecoin, block_number=None):
    """Calculate TVL from token balances at specific block"""
    block_identifier = block_number if block_number is not None else 'latest'

    balance0 = token0_contract.functions.balanceOf(pool).call(block_identifier=block_identifier)
    balance1 = token1_contract.functions.balanceOf(pool).call(block_identifier=block_identifier)

    adjusted_balance0 = balance0 / (10 ** token0_decimals)
    adjusted_balance1 = balance1 / (10 ** token1_decimals)

    if token1_symbol in STABLECOINS:
        total_value_in_token1 = (adjusted_balance0 * price_in_stablecoin) + adjusted_balance1
    else:
        total_value_in_token1 = (adjusted_balance1 * price_in_stablecoin) + adjusted_balance0

    return total_value_in_token1


async def get_active_liquidity(contract, BLOCK_NUMBER: int = None) -> Decimal:

    """Get tick data following Uniswap V3's active liquidity calculation methodology

    Args:
        sqrt_price_x96: Current sqrt price * 2^96
        contract: Pool contract instance
        token0_decimals: Decimals of token0
        token1_decimals: Decimals of token1
        token0_price_in_token1: Price of token0 in terms of token1
    """

    # Get current pool liquidity -
    # or "The active liquidity at the current Price is also stored in the smart contract"
    # Described in the smart contract as "The currently in range liquidity available to the pool"
    liquidity = contract.functions.liquidity().call(block_identifier=BLOCK_NUMBER)

    return Decimal(liquidity)


def calculate_tvl_from_ticks(pool_data, token0_decimals, token1_decimals, token0_price_in_token1, token1_symbol):

    # Process current tick first
    df = pd.DataFrame(pool_data)
    amounts0 = df['totalValueLockedToken0'].sum()
    amounts1 = df['totalValueLockedToken1'].sum()


    # Convert to token1 value
    if token1_symbol in STABLECOINS:
        TVL_in_stablecoin = (amounts0 * token0_price_in_token1) + amounts1
    else:
        TVL_in_stablecoin = (amounts1 * token0_price_in_token1) + amounts0

    return TVL_in_stablecoin



# async def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--csv_file_path', type=str,
#                         help='Path to the .csv file',
#                         default="./../../../data/pools/rpc")
#
#     parser.add_argument('--start_date', type=str, default="2023-03-01")
#     parser.add_argument('--end_date', type=str, default="2023-03-31")
#     args = parser.parse_args()
#
#     # Generate hourly timestamps within a date range
#     hourly_timestamps_date_range = pd.date_range(start=args.start_date, end=args.end_date, freq="H").to_list()
#
#     RPC_URL = os.getenv("INFURA_RPC_URL")
#     web3 = Web3(Web3.HTTPProvider(RPC_URL))
#     for hour_timestamp in tqdm(hourly_timestamps_date_range):
#
#         human_readable_timestamp = hour_timestamp.strftime('%Y-%m-%d %H-%M-%S')
#
#         month = hour_timestamp.strftime("%m")
#         dir_path = os.path.join(args.csv_file_path, month)
#         os.makedirs(dir_path, exist_ok=True)
#
#         file_path = os.path.join(dir_path, f"{human_readable_timestamp}.parquet")
#
#         # Check if the file already exists
#         if os.path.exists(file_path):
#             print(f"File already exists for {human_readable_timestamp}...")
#             continue
#
#         # Get the block number at timestamp
#         print(f"Getting block number for {hour_timestamp}...")
#         BLOCK_NUMBER = get_block_by_timestamp(hour_timestamp)
#
#         dfs = []
#         for POOL_ADDRESS, TRADING_PAIR in POOLS_AND_TRADING_PAIRS.items():
#
#             # BLOCK_NUMBER = "latest"
#             # # # BLOCK_NUMBER = 21497664
#             # # Get ticks data
#             # if BLOCK_NUMBER == "latest":
#             #     BLOCK_NUMBER = None
#
#             print(f"Getting data for pool {POOL_ADDRESS} ({TRADING_PAIR}) at block {BLOCK_NUMBER}...")
#
#             pool = Web3.to_checksum_address(POOL_ADDRESS)
#             contract = web3.eth.contract(address=pool, abi=V3_ABI)
#
#             # Get tokens and set up contracts
#             token0_address = contract.functions.token0().call(block_identifier=BLOCK_NUMBER)
#             token1_address = contract.functions.token1().call(block_identifier=BLOCK_NUMBER)
#             token0_contract = web3.eth.contract(address=token0_address, abi=ERC20_ABI)
#             token1_contract = web3.eth.contract(address=token1_address, abi=ERC20_ABI)
#
#             # Get token decimals
#             token0_decimals = token0_contract.functions.decimals().call(block_identifier=BLOCK_NUMBER)
#             token1_decimals = token1_contract.functions.decimals().call(block_identifier=BLOCK_NUMBER)
#             token0_symbol = token0_contract.functions.symbol().call(block_identifier=BLOCK_NUMBER)
#             token1_symbol = token1_contract.functions.symbol().call(block_identifier=BLOCK_NUMBER)
#
#             decimal_diff = token1_decimals - token0_decimals
#
#             flip_flag = False
#             if token1_symbol not in STABLECOINS:
#                 flip_flag = True
#
#             # Get current price from pool
#             slot0 = contract.functions.slot0().call(block_identifier=BLOCK_NUMBER)
#             fee = contract.functions.fee().call(block_identifier=BLOCK_NUMBER)
#             sqrt_price_x96 = slot0[0]
#             sqrtPriceCurrent = Decimal(sqrt_price_x96) / Q96
#             price = sqrtPriceCurrent ** 2
#             token0_price_in_token1 = float(price / Decimal(10 ** decimal_diff))
#             current_tick = int(slot0[1])
#
#             # Get tick spacing from contract
#             tick_spacing = contract.functions.tickSpacing().call(block_identifier=BLOCK_NUMBER)
#
#             if flip_flag:
#                 token0_price_in_token1 = 1 / token0_price_in_token1
#
#             # Calculate TVL using both methods
#             tvl_balances = calculate_tvl_from_balances(token0_contract, token1_contract, token0_decimals, token1_decimals,
#                                     token0_symbol, token1_symbol,
#                                     pool, token0_price_in_token1, BLOCK_NUMBER)
#
#             volume_24hrs = await get_pool_volume_in_block_async(POOL_ADDRESS, BLOCK_NUMBER)
#
#             # Get active liquidity, which is the "The currently in range liquidity available to the pool"
#             active_liquidity = await get_active_liquidity(contract, BLOCK_NUMBER)
#
#             graph_ticks = await get_initialized_ticks(POOL_ADDRESS, BLOCK_NUMBER)
#
#             # Get all the ticks around the current tick
#             bar_ticks = await create_bar_chart_ticks(
#                 tick_current=current_tick,
#                 pool_liquidity=int(active_liquidity),
#                 tick_spacing=tick_spacing,
#                 token0={"decimals": token0_decimals, "symbol": token0_symbol},  # Simplified token info
#                 token1={"decimals": token1_decimals, "symbol": token1_symbol},  # Simplified token info
#                 num_surrounding_ticks=len(graph_ticks),  # Adjust range to get all the ticks around the current tick
#                 # num_surrounding_ticks=2,  # Adjust range to get all the ticks around the current tick
#                 # fee_tier=tick_spacing * 50,  # Convert tick spacing to fee tier
#                 fee_tier = fee,
#                 sqrt_price_x96=sqrt_price_x96,
#                 graph_ticks=graph_ticks
#             )
#
#             df = pd.DataFrame(bar_ticks)
#
#             # Add the remaining data to the dataframe
#             df['token0_symbol'] = token0_symbol
#             df['token1_symbol'] = token1_symbol
#             df['token0_decimals'] = token0_decimals
#             df['token1_decimals'] = token1_decimals
#             df['price_in_stablecoin'] = token0_price_in_token1
#             df['cross_tick'] = current_tick
#             df['totalValueLockedUSD'] = tvl_balances
#             df['fee_tier'] = fee
#             df['sqrt_price_x96'] = Decimal(sqrt_price_x96)
#             df['block_number'] = BLOCK_NUMBER
#             df['pool_address'] = POOL_ADDRESS
#             df['trading_pair'] = TRADING_PAIR
#             df['tick_spacing'] = tick_spacing
#             df['24hr Volume (USD)'] = volume_24hrs
#
#             # # Convert large integers to string
#             df['totalValueLockedUSD'] = df['totalValueLockedUSD'].astype(str)
#             df['24hr Volume (USD)'] = df['24hr Volume (USD)'].astype(str)
#             df['token0Price'] = df['token0Price'].astype(str)
#             df['token1Price'] = df['token1Price'].astype(str)
#             df['liquidityActive'] = df['liquidityActive'].astype(str)
#             # df['sqrt_price_x96'] = df['sqrt_price_x96'].astype(np.uint64)
#             df['sqrt_price_x96'] = df['sqrt_price_x96'].astype(str)
#
#
#             print(f"Saving the processed tick data for {TRADING_PAIR}...")
#             # append the dataframe to the list
#             dfs.append(df)
#
#
#             # Get pool liquidity
#             # pool_data = await get_pool_liquidity_data(POOL_ADDRESS, BLOCK_NUMBER, current_tick,
#             #                                           tick_spacing, active_liquidity, sqrt_price_x96, tick_spacing)
#             #
#             # tick_data = TICK_VARIABLES(*contract.functions.ticks(current_tick).call(block_identifier=BLOCK_NUMBER))
#
#             # tvl_ticks = calculate_tvl_from_ticks(bar_ticks, token0_decimals,
#             #                                      token1_decimals, token0_price_in_token1, token1_symbol)
#             #
#             #
#             # print(f"\nAnalysis for block {BLOCK_NUMBER if BLOCK_NUMBER else 'latest'}:")
#             # print(f"TVL from balances (in token1): {tvl_balances:,.2f}")
#             # print(f"TVL from ticks (in token1): {tvl_ticks:,.2f}")
#             # print(f"Difference: {abs(tvl_balances - float(tvl_ticks)):,.2f}")
#             # print(f"Current price (token0/token1): {token0_price_in_token1:,.8f}")
#
#
#         # Concatenate all the dataframes
#         final_df = pd.concat(dfs)
#
#         # Save the dataframe as a parquet file
#         print(f"Saving the data to {file_path}...")
#         final_df.to_parquet(file_path, engine='fastparquet')
#
#         # break


async def get_token_metadata(web3, token_address: str, block_number: int) -> TokenMetadata:
    """Get token decimals and symbol with caching"""
    token_contract = web3.eth.contract(address=token_address, abi=ERC20_ABI)
    decimals = token_contract.functions.decimals().call(block_identifier=block_number)
    symbol = token_contract.functions.symbol().call(block_identifier=block_number)
    return TokenMetadata(decimals=decimals, symbol=symbol)


async def initialize_pool_metadata(web3, pool_address: str, block_number: int) -> None:
    """Initialize metadata for a pool if not already cached"""
    if pool_address not in pool_metadata_cache:
        contract = web3.eth.contract(address=pool_address, abi=V3_ABI)
        token0_address = contract.functions.token0().call(block_identifier=block_number)
        token1_address = contract.functions.token1().call(block_identifier=block_number)

        # Get metadata for both tokens concurrently
        token0_meta, token1_meta = await asyncio.gather(
            get_token_metadata(web3, token0_address, block_number),
            get_token_metadata(web3, token1_address, block_number)
        )
        pool_metadata_cache[pool_address] = (token0_meta, token1_meta)


async def _process_pool_data_inner(web3, pool_address: str, trading_pair: str, block_number: int) -> pd.DataFrame:
    """Process data for a single pool at a specific block (single attempt)."""
    t_pool_start = time.time()
    logger.info(f"  [POOL] Starting {trading_pair} (pool={pool_address[:10]}...) at block {block_number}")

    # Get cached metadata or initialize if not present
    await initialize_pool_metadata(web3, pool_address, block_number)
    token0_meta, token1_meta = pool_metadata_cache[pool_address]
    logger.info(f"  [POOL] {trading_pair}: token0={token0_meta.symbol} ({token0_meta.decimals}d), token1={token1_meta.symbol} ({token1_meta.decimals}d)")

    pool = Web3.to_checksum_address(pool_address)
    contract = web3.eth.contract(address=pool, abi=V3_ABI)

    # Get token contracts first as they're needed for TVL calculation
    # token0_address = contract.functions.token0().call(block_identifier=block_number)
    # token1_address = contract.functions.token1().call(block_identifier=block_number)
    # token0_contract = web3.eth.contract(address=token0_address, abi=ERC20_ABI)
    # token1_contract = web3.eth.contract(address=token1_address, abi=ERC20_ABI)

    # Get current price and other pool data
    slot0 = contract.functions.slot0().call(block_identifier=block_number)
    fee = contract.functions.fee().call(block_identifier=block_number)
    sqrt_price_x96 = slot0[0]
    current_tick = int(slot0[1])
    tick_spacing = contract.functions.tickSpacing().call(block_identifier=block_number)

    # Calculate price
    decimal_diff = token1_meta.decimals - token0_meta.decimals
    sqrtPriceCurrent = Decimal(sqrt_price_x96) / Q96
    price = sqrtPriceCurrent ** 2
    token0_price_in_token1 = float(price / Decimal(10 ** decimal_diff))

    flip_flag = token1_meta.symbol not in STABLECOINS
    if flip_flag:
        token0_price_in_token1 = 1 / token0_price_in_token1

    logger.info(f"  [POOL] {trading_pair}: slot0 fetched — currentTick={current_tick}, fee={fee}, tickSpacing={tick_spacing}")
    logger.info(f"  [POOL] {trading_pair}: price={token0_price_in_token1:.8f} (flip={'yes' if flip_flag else 'no'})")

    # Get data concurrently
    logger.info(f"  [POOL] {trading_pair}: Fetching active liquidity + initialized ticks...")
    active_liquidity, graph_ticks = await asyncio.gather(
        get_active_liquidity(contract, block_number),
        get_initialized_ticks(pool_address, block_number),
    )
    logger.info(f"  [POOL] {trading_pair}: Got {len(graph_ticks)} initialized ticks, activeLiq={active_liquidity}")

    # Get tick data
    logger.info(f"  [POOL] {trading_pair}: Building bar chart ticks...")
    bar_ticks = await create_bar_chart_ticks(
        tick_current=current_tick,
        pool_liquidity=int(active_liquidity),
        tick_spacing=tick_spacing,
        token0={"decimals": token0_meta.decimals, "symbol": token0_meta.symbol},
        token1={"decimals": token1_meta.decimals, "symbol": token1_meta.symbol},
        num_surrounding_ticks=len(graph_ticks),
        fee_tier=fee,
        sqrt_price_x96=sqrt_price_x96,
        graph_ticks=graph_ticks
    )

    # # Calculate TVL
    # tvl_balances = calculate_tvl_from_balances(
    #     token0_contract, token1_contract,
    #     token0_meta.decimals, token1_meta.decimals,
    #     token0_meta.symbol, token1_meta.symbol,
    #     pool, token0_price_in_token1, block_number
    # )
    #
    # volume_24hours = await get_pool_volume_in_block_async(pool_address, block_number)

    # Create DataFrame and add metadata
    df = pd.DataFrame(bar_ticks)

    # Add the remaining data to the dataframe
    df['token0_symbol'] = token0_meta.symbol
    df['token1_symbol'] = token1_meta.symbol
    df['token0_decimals'] = token0_meta.decimals
    df['token1_decimals'] = token1_meta.decimals
    df['price_in_stablecoin'] = token0_price_in_token1
    df['cross_tick'] = current_tick
    df['fee_tier'] = fee
    df['sqrt_price_x96'] = str(Decimal(sqrt_price_x96))
    df['block_number'] = block_number
    df['pool_address'] = pool_address
    df['trading_pair'] = trading_pair
    df['tick_spacing'] = tick_spacing
    # df['totalValueLockedUSD'] = str(tvl_balances)
    # df['24hr Volume (USD)'] = str(volume_24hours)

    # # Convert large integers to string
    df['token0Price'] = df['token0Price'].astype(str)
    df['token1Price'] = df['token1Price'].astype(str)
    df['liquidityActive'] = df['liquidityActive'].astype(str)

    elapsed = time.time() - t_pool_start
    logger.info(f"  [POOL] {trading_pair}: Done — {len(df)} rows in {elapsed:.1f}s")
    return df


async def process_pool_data(web3, pool_address: str, trading_pair: str, block_number: int,
                            max_retries: int = 3) -> pd.DataFrame:
    """Process data for a single pool with automatic RPC failover.

    On retryable RPC errors the FallbackWeb3 instance rotates to the next
    provider and the entire pool processing is retried.
    """
    last_error = None
    for attempt in range(max_retries):
        url = web3.current_url if isinstance(web3, FallbackWeb3) else "unknown"
        try:
            return await _process_pool_data_inner(web3, pool_address, trading_pair, block_number)
        except RPC_RETRYABLE_ERRORS as exc:
            last_error = exc
            logger.error(f"[RPC] Pool {trading_pair} attempt {attempt+1}/{max_retries} failed on {url}: {exc.__class__.__name__}: {exc}")
            if isinstance(web3, FallbackWeb3):
                web3.rotate(failed_url=url)
            else:
                raise
    raise last_error


async def process_timestamp(web3, hour_timestamp: pd.Timestamp, csv_file_path: str) -> None:
    """Process all pools for a single timestamp, skipping already-scraped pools."""
    t_ts_start = time.time()
    human_readable_timestamp = hour_timestamp.strftime('%Y-%m-%d %H-%M-%S')
    month = hour_timestamp.strftime("%m")
    dir_path = os.path.join(csv_file_path, month)
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, f"{human_readable_timestamp}.parquet")

    logger.info(f"[TIMESTAMP] ---- {human_readable_timestamp} ----")
    logger.info(f"[TIMESTAMP] Output file: {file_path}")

    # Load existing data (if any) and determine which pools still need scraping
    existing_df = pd.read_parquet(file_path, engine='fastparquet') if os.path.exists(file_path) else None
    # Normalise to lowercase to avoid checksum vs. non-checksum address mismatches
    already_scraped = set(addr.lower() for addr in existing_df['pool_address'].unique()) if existing_df is not None else set()
    pools_to_process = {k: v for k, v in POOLS_AND_TRADING_PAIRS.items() if k.lower() not in already_scraped}

    if not pools_to_process:
        logger.info(f"[TIMESTAMP] All {len(already_scraped)} pools already scraped for {human_readable_timestamp} — skipping")
        return

    if already_scraped:
        logger.info(f"[TIMESTAMP] {len(already_scraped)} pools already done, {len(pools_to_process)} remaining for {human_readable_timestamp}")
    else:
        logger.info(f"[TIMESTAMP] {len(pools_to_process)} pools to process for {human_readable_timestamp}")

    logger.info(f"[TIMESTAMP] Resolving block number for {hour_timestamp}...")
    block_number = get_block_by_timestamp(hour_timestamp)
    logger.info(f"[TIMESTAMP] Block {block_number} resolved for {human_readable_timestamp}")

    # Process only missing pools concurrently
    logger.info(f"[TIMESTAMP] Launching {len(pools_to_process)} pool tasks concurrently...")
    tasks = [
        process_pool_data(web3, Web3.to_checksum_address(addr), pair, block_number)
        for addr, pair in pools_to_process.items()
    ]
    new_dfs = await asyncio.gather(*tasks)

    # Append new data to existing (if any) and save
    final_df = pd.concat([existing_df] + list(new_dfs)) if existing_df is not None else pd.concat(new_dfs)
    final_df.to_parquet(file_path, engine='fastparquet')
    elapsed = time.time() - t_ts_start
    logger.info(f"[TIMESTAMP] Saved {len(final_df)} rows to {file_path} ({elapsed:.1f}s total)")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file_path', type=str, default="./../../../data/pools/rpc")
    parser.add_argument('--start_date', type=str, default="2023-02-28")
    parser.add_argument('--end_date', type=str, default="2023-04-01")
    parser.add_argument('--timestamp_frequency', type=str, default="h",
                        help="Timestamp frequency (e.g., 's' for seconds, 'min' for minutes, 'h' for hourly, 'D' for daily)")
    parser.add_argument('--max_concurrent', type=int, default=1,
                        help='Maximum number of timestamps to process concurrently')
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("getTicks.py — Uniswap V3 Tick Data Collection")
    logger.info("=" * 70)
    logger.info(f"  Date range     : {args.start_date}  →  {args.end_date}")
    logger.info(f"  Frequency      : {args.timestamp_frequency}")
    logger.info(f"  Max concurrent : {args.max_concurrent}")
    logger.info(f"  Output dir     : {os.path.abspath(args.csv_file_path)}")
    logger.info(f"  Active pools   : {len(POOLS_AND_TRADING_PAIRS)}")
    for addr, pair in POOLS_AND_TRADING_PAIRS.items():
        logger.info(f"    • {pair}  ({addr[:10]}...)")

    # Generate timestamps
    timestamps = pd.date_range(start=args.start_date, end=args.end_date, freq=args.timestamp_frequency).to_list()
    logger.info(f"  Timestamps     : {len(timestamps)} ({timestamps[0]} → {timestamps[-1]})")
    logger.info("=" * 70)

    # Use FallbackWeb3 with automatic RPC rotation.
    logger.info("[INIT] Initializing FallbackWeb3 with RPC rotation...")
    web3 = FallbackWeb3()

    # Process timestamps in batches
    total_batches = (len(timestamps) + args.max_concurrent - 1) // args.max_concurrent
    for i in range(0, len(timestamps), args.max_concurrent):
        batch_num = i // args.max_concurrent + 1
        batch = timestamps[i:i + args.max_concurrent]
        logger.info(f"[BATCH] === Batch {batch_num}/{total_batches} — {len(batch)} timestamp(s) ===")
        tasks = [process_timestamp(web3, ts, args.csv_file_path) for ts in batch]
        await tqdm_asyncio.gather(*tasks)
        logger.info(f"[BATCH] Batch {batch_num}/{total_batches} complete")


if __name__ == "__main__":

    # Resubmit the job if it fails
    for i in range(100):
        logger.info(f"[MAIN] >>>  Attempt {i + 1}/100  <<<")
        try:
            asyncio.run(main())
            logger.info("[MAIN] Job completed successfully.")
            break
        except Exception as e:
            logger.error(f"[MAIN] Fatal error on attempt {i + 1}: {e}", exc_info=True)
            logger.info("[MAIN] Sleeping 1s before retry...")
            time.sleep(1)
            logger.info("[MAIN] Retrying...")
            continue