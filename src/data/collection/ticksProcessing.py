from enum import IntEnum
import math
from typing import List, Dict, TypedDict, Union, Tuple

# Adding the path to be able to import the analytics module
import sys

sys.path.append('./../../../')

from src.modeling.analytics_levels import tick_to_price, get_tick_at_sqrt_price
from decimal import Decimal, getcontext, ROUND_DOWN, ROUND_HALF_UP, ROUND_UP, InvalidOperation


# Constants
Q96 = Decimal(2) ** 96

# Constants
MAX_INT128 = (2 ** 128) - 1
MIN_TICK = -887272
MAX_TICK = 887272

TICK_BASE = Decimal('1.0001')

class Direction(IntEnum):
    ASC = 0
    DESC = 1


class TickProcessed(TypedDict):
    tickIdx: int
    liquidityActive: int
    liquidityNet: int
    token0Price: float
    token1Price: float
    isCurrent: bool


class BarChartTick(TypedDict):
    tickIdx: int
    liquidityActive: int
    totalValueLockedToken0: float
    totalValueLockedToken1: float
    # price0: float
    # price1: float
    isCurrentTick: bool


class GraphTick(TypedDict):
    tickIdx: str
    liquidityGross: str
    liquidityNet: str


async def create_bar_chart_ticks(
        tick_current: int,
        pool_liquidity: int,
        tick_spacing: int,
        token0: dict,
        token1: dict,
        num_surrounding_ticks: int,
        fee_tier: int,
        sqrt_price_x96: Decimal,
        graph_ticks: List[GraphTick]
) -> List[BarChartTick]:
    """Direct port of createBarChartTicks
    Based on: https://github.com/Uniswap/examples/blob/main/v3-sdk/pool-data/src/libs/active-liquidity.ts
    and https://docs.uniswap.org/sdk/v3/guides/advanced/active-liquidity


    """

    processed_ticks = process_ticks(
        tick_current,
        pool_liquidity,
        tick_spacing,
        token0,
        token1,
        num_surrounding_ticks,
        graph_ticks
    )

    bar_ticks = []
    for tick in processed_ticks:
        bar_tick = await calculate_locked_liquidity(tick, token0, token1, tick_spacing,
                                                    fee_tier, sqrt_price_x96, tick_current)
        bar_ticks.append(bar_tick)

    return bar_ticks


def process_ticks(
        tick_current: int,
        pool_liquidity: int,
        tick_spacing: int,
        token0: dict,
        token1: dict,
        num_surrounding_ticks: int,
        graph_ticks: List[GraphTick]
) -> List[TickProcessed]:
    """Direct port of processTicks"""

    # Create tick dictionary
    tick_idx_to_tick_dictionary = {
        tick['tickIdx']: tick for tick in graph_ticks
    }

    active_tick_idx = (tick_current // tick_spacing) * tick_spacing

    if active_tick_idx <= MIN_TICK:
        active_tick_idx = MAX_TICK

    # Process active tick
    price_in_token1 = tick_to_price(tick=active_tick_idx, decimals=token0['decimals'] - token1['decimals'])
    active_tick_processed = {
        'tickIdx': active_tick_idx,
        'liquidityActive': pool_liquidity,
        'liquidityNet': 0,
        'price0': 1 / price_in_token1,
        'price1': price_in_token1,
        'isCurrent': True
    }

    # Update active tick if initialized
    active_tick = tick_idx_to_tick_dictionary.get(str(active_tick_idx))
    if active_tick:
        active_tick_processed['liquidityNet'] = int(active_tick['liquidityNet'])

    # Get surrounding ticks
    subsequent_ticks = compute_initialized_ticks(
        active_tick_processed,
        num_surrounding_ticks,
        tick_spacing,
        Direction.ASC,
        token0,
        token1,
        tick_idx_to_tick_dictionary
    )

    previous_ticks = compute_initialized_ticks(
        active_tick_processed,
        num_surrounding_ticks,
        tick_spacing,
        Direction.DESC,
        token0,
        token1,
        tick_idx_to_tick_dictionary
    )

    return previous_ticks + [active_tick_processed] + subsequent_ticks


def compute_initialized_ticks(
        active_tick_processed: TickProcessed,
        num_surrounding_ticks: int,
        tick_spacing: int,
        direction: Direction,
        token0: dict,
        token1: dict,
        tick_idx_to_tick_dictionary: Dict
) -> List[TickProcessed]:
    """
    Compute the active liquidity for the surrounding ticks based on the active tick's liquidity
    and the liquidityNet values of initialized ticks.

    Args:
        active_tick_processed: The current active tick data
        num_surrounding_ticks: Number of ticks to compute in each direction
        tick_spacing: The spacing between ticks
        direction: ASC or DESC direction for computation
        token0: Token0 info
        token1: Token1 info
        tick_idx_to_tick_dictionary: Dictionary of initialized ticks

    Returns:
        List of processed ticks with their active liquidity
    """
    previous_tick_processed = dict(active_tick_processed)
    ticks_processed = []

    for i in range(num_surrounding_ticks):
        # Calculate the next tick index based on direction
        current_tick_idx = (
            previous_tick_processed['tickIdx'] + tick_spacing
            if direction == Direction.ASC
            else previous_tick_processed['tickIdx'] - tick_spacing
        )

        # Break if we're outside the valid tick range
        if current_tick_idx < MIN_TICK or current_tick_idx > MAX_TICK:
            break

        # Calculate prices for the current tick
        price_in_token1 = tick_to_price(
            tick=current_tick_idx,
            decimals=token0['decimals'] - token1['decimals']
        )

        # Initialize the current tick with previous tick's liquidity
        current_tick_processed = {
            'tickIdx': current_tick_idx,
            'liquidityActive': previous_tick_processed['liquidityActive'],
            'liquidityNet': 0,
            'price0': 1 / price_in_token1,
            'price1': price_in_token1,
            'isCurrent': False
        }

        # If this is an initialized tick, update its liquidity
        current_initialized_tick = tick_idx_to_tick_dictionary.get(str(current_tick_idx))
        if current_initialized_tick:
            liquidity_net = int(current_initialized_tick['liquidityNet'])
            current_tick_processed['liquidityNet'] = liquidity_net

            if direction == Direction.ASC:
                # For ascending direction, add the liquidityNet to get the new active liquidity
                current_tick_processed['liquidityActive'] = (
                        previous_tick_processed['liquidityActive'] + liquidity_net
                )
            else:
                # For descending direction, subtract the previous tick's liquidityNet
                if previous_tick_processed['liquidityNet'] != 0:
                    current_tick_processed['liquidityActive'] = (
                            previous_tick_processed['liquidityActive'] -
                            previous_tick_processed['liquidityNet']
                    )

        ticks_processed.append(current_tick_processed)
        previous_tick_processed = dict(current_tick_processed)

    # Reverse the list for descending direction to maintain correct order
    if direction == Direction.DESC:
        ticks_processed.reverse()

    return ticks_processed


def get_token_amounts(liquidity, sqrt_price_x96, tick_low, tick_high, current_tick, isCurrent):
    """
    Calculate token amounts based on liquidity position and current tick.
    Based on https://blog.uniswap.org/uniswap-v3-math-primer-2

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


async def calculate_locked_liquidity(
        tick: TickProcessed,
        token0: dict,
        token1: dict,
        tick_spacing: int,
        fee_tier: int,
        sqrt_price_x96: float,
        current_tick: int
) -> BarChartTick:
    """Direct port of calculateLockedLiquidity"""

    # Simplified liquidity calculation since we can't use the full Uniswap SDK
    liquidity_active = int(tick['liquidityActive'])

    # sqrtPriceLow = 1.0001 ** ((tick['tickIdx'] - tick_spacing) // 2)
    # sqrtPriceHigh = 1.0001 ** (tick['tickIdx'] // 2)

    # sqrt_price_current = sqrt_price_x96 / (1 << 96)
    # amount0 = calculate_token0_amount(liquidity_active, sqrt_price_current, sqrtPriceLow, sqrtPriceHigh)
    # amount1 = calculate_token1_amount(liquidity_active, sqrt_price_current, sqrtPriceLow, sqrtPriceHigh)
    #
    # # Convert with decimals
    # amount0 = amount0 / (10 ** token0['decimals'])
    # amount1 = amount1 / (10 ** token1['decimals'])

    # # Get the closest tick in tick spacing
    # current_tick = get_tick_at_sqrt_price(sqrt_price_x96)
    # current_tick = (current_tick // tick_spacing) * tick_spacing

    # Active liquidity
    current_tick_spaced = (current_tick // tick_spacing) * tick_spacing

    # if tick['isCurrent']:
    #     print('Current tick:', current_tick)

    lower_bound = tick['tickIdx']
    upper_bound = tick['tickIdx'] + tick_spacing

    new_amount0, new_amount1 = get_token_amounts(liquidity_active, sqrt_price_x96, lower_bound,
                                                 upper_bound, current_tick, tick['isCurrent'])

    new_amount0 = new_amount0 / (10 ** token0['decimals'])
    new_amount1 = new_amount1 / (10 ** token1['decimals'])


    return {
        # 'tickIdx': current_tick if tick['isCurrent'] else tick['tickIdx'],
        'tickIdx': tick['tickIdx'],
        'liquidityActive': int(tick['liquidityActive']),
        'totalValueLockedToken0': new_amount0,
        'totalValueLockedToken1': new_amount1,
        'token0Price': tick['price0'],
        'token1Price': tick['price1'],
        'isCurrentTick': tick['isCurrent']
    }

