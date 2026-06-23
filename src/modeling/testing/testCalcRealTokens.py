from web3 import Web3
import os
import math
from src.data.collection.ticksProcessing import Q96, TICK_BASE
from src.data.collection.getTicks import V3_ABI, ERC20_ABI

from decimal import Decimal, getcontext, ROUND_UP

# Load the environment variables
from dotenv import load_dotenv

load_dotenv()

# Set precision for Decimal operations (setting high enough for high accuracy in financial calculations)
getcontext().prec = 34  # Set precision to 100 decimal places
getcontext().rounding = ROUND_UP  # Set rounding method to down to match financial rounding


# Calculation of token amounts working by validating it with a pool that has no that much activity
# hence it is possible to see if the calculated tokens match
# what https://app.uniswap.org/explore/pools/ethereum/0x6f48eca74b38d2936b02ab603ff4e36a6c0e3a77 reports
# and using the reported
# values from: https://etherscan.io/address/0x6f48eca74b38d2936b02ab603ff4e36a6c0e3a77#readContract

POOL_ADDRESS = "0x6f48eca74b38d2936b02ab603ff4e36a6c0e3a77"

RPC_URL = os.environ.get("INFURA_RPC_URL")
web3 = Web3(Web3.HTTPProvider(RPC_URL))
pool = Web3.to_checksum_address(POOL_ADDRESS)
contract = web3.eth.contract(address=pool, abi=V3_ABI)

amounts0 = 0
amounts1 = 0
liquidity = 0


def get_token_amounts(liquidity, sqrt_price_x96, tick_low, tick_high, current_tick, isCurrent=False):
    """
    Calculate token amounts based on liquidity position and current tick.

    Args:
        liquidity: The amount of liquidity
        sqrt_price_x96: Current sqrt price in X96 format
        tick_low: Lower tick bound of position
        tick_high: Upper tick bound of position
        current_tick: Current tick price
        isCurrent: Boolean indicating if this is the current active tick

    Returns:
        tuple: (amount0, amount1) token amounts
    """

    # Convert to correct units
    sqrt_price = Decimal(sqrt_price_x96) / Q96

    # Calculate the correct sqrt prices for the range
    # sqrt_ratio_a = Decimal(TICK_BASE) ** (Decimal(tick_low) / 2)
    # sqrt_ratio_b = Decimal(TICK_BASE) ** (Decimal(tick_high) / 2)

    # Calculate directly with square root
    sqrt_ratio_a = Decimal(TICK_BASE ** tick_low).sqrt()
    sqrt_ratio_b = Decimal(TICK_BASE ** tick_high).sqrt()

    amount0 = Decimal(0)
    amount1 = Decimal(0)

    if isCurrent:
        # which is also equivalent to if tick_low <= current_tick < tick_high:
        # Only calculate in-range amounts for the current active tick
        amount0 = liquidity * (sqrt_ratio_b - sqrt_price) / (sqrt_price * sqrt_ratio_b)
        amount1 = liquidity * (sqrt_price - sqrt_ratio_a)
    else:
        # For non-active ticks
        # if sqrt_price >= sqrt_ratio_b:
        if current_tick >= tick_high:
            # Price is above the range - all liquidity in token1
            amount1 = liquidity * (sqrt_ratio_b - sqrt_ratio_a)

        # elif sqrt_price <= sqrt_ratio_a:
        elif current_tick < tick_low:
            # Price is below the range - all liquidity in token0
            amount0 = liquidity * (sqrt_ratio_b - sqrt_ratio_a) / (sqrt_ratio_a * sqrt_ratio_b)

    # Round down to avoid rounding errors
    amount0 = Decimal(math.floor(amount0))
    amount1 = Decimal(math.floor(amount1))

    return amount0, amount1


# Get tokens and set up contracts
token0_address = contract.functions.token0().call(block_identifier=None)
token1_address = contract.functions.token1().call(block_identifier=None)
token0_contract = web3.eth.contract(address=token0_address, abi=ERC20_ABI)
token1_contract = web3.eth.contract(address=token1_address, abi=ERC20_ABI)

# Get token decimals
token0_decimals = token0_contract.functions.decimals().call(block_identifier=None)
token1_decimals = token1_contract.functions.decimals().call(block_identifier=None)

# token0_decimals = 18
# token1_decimals = 6

# MIN_TICK = -887272
# MAX_TICK = 887272
# TICK_SPACING = 60

# Get the active liquidity
liquidity_active = contract.functions.liquidity().call(block_identifier=None)
# liquidity_active = 17197539541296086388

# Get current price from pool
slot0 = contract.functions.slot0().call(block_identifier=None)
fee = contract.functions.fee().call(block_identifier=None)
sqrt_price_x96 = slot0[0]
# sqrt_price_x96 = 79309892610667957777094
sqrtPriceCurrent = sqrt_price_x96 / Q96
price = sqrtPriceCurrent ** 2

# Get tick spacing from contract
tick_spacing = contract.functions.tickSpacing().call(block_identifier=None)

# Round current tick to nearest valid tick for the currently active tick
# current_tick = get_tick_at_sqrt_price(sqrt_price_x96)
current_tick = int(slot0[1])
# current_tick = -276304
current_tick_spaced = (current_tick // tick_spacing) * tick_spacing
lower_bound = current_tick_spaced
upper_bound = current_tick_spaced + tick_spacing

liquidity_net = 191297587435216793761

# amount0_new, amount1_new = get_token_amounts(liquidity_active, sqrt_price_x96, current_tick_spaced, current_tick_spaced + tick_spacing, current_tick)
amount0_new, amount1_new = get_token_amounts(
    liquidity_active,
    sqrt_price_x96,
    lower_bound,
    upper_bound,
    current_tick,
    True
)

amount0_new = amount0_new / (10 ** token0_decimals)
amount1_new = amount1_new / (10 ** token1_decimals)

print(f"Amount0: {amount0_new:,} and Amount1: {amount1_new:,}")


# sqrt_price_x96 = 4629954452967590501245915
# current_tick = -194961

liquidityNet = -224642369278857978042
liquidityActive = liquidity_active + liquidityNet
lower_bound = -276310
upper_bound = lower_bound + tick_spacing

# Current active tick (-195000)
amount0, amount1 = get_token_amounts(
    liquidityActive,
    sqrt_price_x96,
    lower_bound,
    upper_bound,
    current_tick,
    False
)
active_amount0 = amount0 / (10 ** token0_decimals)
active_amount1 = amount1 / (10 ** token1_decimals)

print(f"Active Amount0: {active_amount0:,} and Active Amount1: {active_amount1:,}")

liquidityNet = -29062402020421996194
liquidityActive = liquidity_active - liquidityNet
lower_bound = -276330
upper_bound = lower_bound + tick_spacing

# Upper tick (-194800)
amount0, amount1 = get_token_amounts(
    liquidityActive,
    sqrt_price_x96,
    lower_bound,
    upper_bound,
    current_tick,
    False
)
upper_amount0 = amount0 / (10 ** token0_decimals)
upper_amount1 = amount1 / (10 ** token1_decimals)

print(f"Upper Amount0: {upper_amount0:,} and Upper Amount1: {upper_amount1:,}")