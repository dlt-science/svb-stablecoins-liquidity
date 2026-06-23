import pandas as pd
from decimal import Decimal, getcontext, ROUND_DOWN, ROUND_HALF_UP, ROUND_UP, InvalidOperation, DivisionUndefined

# Adding the path to be able to import the analytics module
import sys

sys.path.append('./../../../')

from src.modeling.analytics_levels import (get_token_amounts, get_tick_at_sqrt_price,
                                           tick_to_price_simple, get_vwap, get_vwapm, get_mci,
                                           StandardiseUniswap, TICK_BASE)


def CalcSwapDeltas(is_bid, liquidity, lower_bound, upper_bound, sqrt_price_x96, decimals0, decimals1,
                   size_trade=None, X_real=None, Y_real=None):
    """
    Function to calculate the execution deltas of a swap in a tick that has a given liquidity and real token quantities.
    Handles both size-based and level-based trading approaches.
    quantities0, quantities1: the real token quantities
    size: the size of the swap
    type: 'buy' or 'sell'
    liquidity: the squared liquidity in the tick
    size_trade (Decimal, optional): Size of trade in token1 terms. If None, uses full tick liquidity.
    price_lower, price_upper: the lower and upper bounds of the tick
    outputs delta0,delta1 i.e. the token0 and token1 quantities that have to be added or removed from the pool.
    """

    liquidity = Decimal(liquidity)
    sqrt_price_x96 = Decimal(sqrt_price_x96)
    decimals0 = Decimal(int(decimals0))
    decimals1 = Decimal(int(decimals1))

    if X_real is None and Y_real is None:
        # X_real, Y_real = quantities0, quantities1
        X_real, Y_real = get_token_amounts(liquidity, sqrt_price_x96, lower_bound, upper_bound)
    else:
        # Convert X_real and Y_real with their respective decimals format
        X_real = X_real * (10 ** decimals0)
        Y_real = Y_real * (10 ** decimals1)

    if X_real == 0 and Y_real == 0:
        return 0, 0

    # Calculate directly with square root
    sqrt_p_a = Decimal(TICK_BASE ** lower_bound).sqrt()
    sqrt_p_b = Decimal(TICK_BASE ** upper_bound).sqrt()

    # # Square root prices for lower and upper bounds
    # sqrt_p_a = tick_to_price_simple(Decimal(int(lower_bound)) / Decimal(2))
    # sqrt_p_b = tick_to_price_simple(Decimal(int(upper_bound)) / Decimal(2))

    offset0 = liquidity / sqrt_p_b
    offset1 = liquidity * sqrt_p_a

    # X_virtual, Y_virtual = virtual0, virtual1
    X_virtual = X_real + offset0
    Y_virtual = Y_real + offset1

    # Other option: L(\sqrt(P_b) - \sqrt(P_a)) Y_real

    if size_trade is not None:
        # Size should be adjusted by the token1 decimals for comparison
        size_swap_stablecoin = size_trade * (10 ** decimals1)

    # X_delta, Y_delta = delta0, delta1,
    if is_bid:
        if size_trade is not None:
            # Size-based approach for bids
            Y_delta = min(size_swap_stablecoin, Y_real)
            X_delta = liquidity ** 2 / (Y_virtual - Y_delta) - X_virtual

        else:
            Y_delta = Y_real
            # X_delta = liquidity ** 2 / (Y_virtual - Y_delta) - X_virtual

            """
            X_delta = liquidity ** 2 / (Y_virtual - Y_delta) - X_virtual
            This can be simplified to:
                X_delta = liquidity ** 2 / (liquidity * sqrt_p_a) - (X_real + (liquidity / sqrt_p_b))
                = liquidity / sqrt_p_a - (X_real + (liquidity / sqrt_p_b))
                = liquidity / sqrt_p_a - X_real - liquidity / sqrt_p_b
                = (liquidity/sqrt_p_a - liquidity/sqrt_p_b) - X_real
                = liquidity * (1/sqrt_p_a - 1/sqrt_p_b) - X_real
            """
            X_delta = liquidity * (1 / sqrt_p_a - 1 / sqrt_p_b) - X_real

    else:
        if size_trade is not None:
            size0 = (liquidity ** 2) / (Y_virtual + size_swap_stablecoin) - X_virtual
            X_delta = abs(min(size0, X_real))
            Y_delta = liquidity ** 2 / (X_virtual - X_delta) - Y_virtual

        else:
            # A buy order will consume token0 by the provided token1
            # Calculate the size in token0
            X_delta = X_real
            # Y_delta = liquidity ** 2 / (X_virtual - X_delta) - Y_virtual

            """
            Y_delta = liquidity ** 2 / (X_virtual - X_delta) - Y_virtual
            This can be simplified to:
            liquidity ** 2 / (liquidity / sqrt_p_b) - (Y_virtual + (liquidity * sqrt_p_a))
            = liquidity * sqrt_p_b - Y_virtual - liquidity * sqrt_p_a

            or using only the real values:
            = liquidity * (sqrt_p_b - sqrt_p_a) - Y_real

            """
            Y_delta = liquidity * (sqrt_p_b - sqrt_p_a) - Y_real

    # Adjust deltas to normal scale for output
    X_delta /= (10 ** decimals0)
    Y_delta /= (10 ** decimals1)

    if X_delta <= 0 or Y_delta <= 0:
        return 0, 0

    return X_delta, Y_delta


def execute_swap(is_bid,
                 ticks, tick_spacing, decimals0, decimals1, sqrt_price_x96, size_trade=None,
                 max_levels=None):
    """
    Simulate swap execution in a Uniswap V3 pool.
    Handles both size-based and level-based approaches.

    Args:
        is_bid (bool): True for bid/sell, False for ask/buy
        tick_liquidity (pd.Series): Liquidity at each tick
        ticks (pd.Series): Available ticks
        tick_spacing (int): Spacing between ticks
        decimals0 (int): Decimals for token0
        decimals1 (int): Decimals for token1
        sqrt_price_x96 (Decimal): Square root of the price scaled by 2^96
        size_trade (float, optional): Size of trade in token1 terms
        max_levels (int, optional): Maximum number of price levels to traverse

    Returns:
        tuple[list, list]: Lists of delta0 and delta1 values

    """
    tick_index = ticks[ticks['isCurrentTick'] == True]

    try:
        tick_index = tick_index.index[0]
    except IndexError:
        print(f"Index error: {tick_index}")

    deltas0 = []
    deltas1 = []

    residual_size = Decimal(size_trade) if size_trade is not None else None
    levels_remaining = max_levels if max_levels is not None else float('inf')

    X_real, Y_real = None, None

    while len(ticks.index) > tick_index >= 0:

        # To break loop after meeting the conditions
        if (size_trade is not None and residual_size <= 0) or \
                (max_levels is not None and levels_remaining <= 0):
            break

        lower_bound = ticks.iloc[tick_index]['tickIdx']
        upper_bound = lower_bound + tick_spacing
        liquidityActive = ticks["liquidityActive"].iloc[tick_index]

        if "totalValueLockedToken0" in ticks.columns:
            X_real = Decimal(ticks["totalValueLockedToken0"].iloc[tick_index])
            Y_real = Decimal(ticks["totalValueLockedToken1"].iloc[tick_index])

            # Try to calculate the real values properly
            if X_real < 0 or Y_real < 0:
                X_real, Y_real = None, None

        delta0, delta1 = CalcSwapDeltas(is_bid, liquidityActive,
                                        lower_bound, upper_bound,
                                        sqrt_price_x96, decimals0, decimals1,
                                        residual_size if size_trade is not None else None,
                                        X_real=X_real, Y_real=Y_real)

        # delta0 is negative for buy and sell orders. delta1 is always positive
        deltas0.append(delta0)
        deltas1.append(delta1)

        if residual_size is not None:
            residual_size -= delta1

        if max_levels is not None:
            levels_remaining -= 1

        tick_index = tick_index - 1 if is_bid else tick_index + 1

    return deltas0, deltas1


def get_quantity(is_bid: bool,
                 ticks: pd.DataFrame, tick_spacing: int, decimals0: Decimal,
                 decimals1: Decimal,
                 sqrt_price_x96: Decimal,
                 size_trade: float = None, level: int = None) -> tuple[Decimal, Decimal]:
    """
    Delta1: Represents the delta for the swap in stablecoin value / dollar value. E.g., USDC, USDT, etc.,
    which for this case would represent quantity
    Delta0: Represents the delta for the swap in the quantity of token0, WETH, etc., which for this case
    would represent dolvol

    Args:
        is_bid (bool): True for bid/sell, False for ask/buy
        tick_liquidity (pd.Series): Liquidity at each tick
        ticks (pd.Series): Available ticks
        tick_spacing (int): Spacing between ticks
        decimals0 (int): Decimals for token0
        decimals1 (int): Decimals for token1
        sqrt_price_x96 (Decimal): Square root of the price scaled by 2^96
        size_trade (float, optional): Size of trade in token1 terms
        level (int, optional): Number of price levels to use

    Returns:
        tuple[Decimal, Decimal]: Dollar volume and quantity
    """

    deltas0, deltas1 = execute_swap(is_bid, ticks,
                                    tick_spacing, decimals0, decimals1, sqrt_price_x96,
                                    size_trade=size_trade, max_levels=level)

    quantity = sum(deltas0)
    dolvol = sum(deltas1)

    # tick_index = ticks[ticks['isCurrentTick'] == True]
    #
    # try:
    #     tick_index = tick_index.index[0]
    # except IndexError:
    #     print(f"Index error: {tick_index}")
    #
    # depth = tick_index - level if is_bid else tick_index + level
    #
    # if is_bid:
    #     # ticks = ticks[-depth:tick_index + 1]
    #     # Consuming all of the X_real in the tick range
    #     ticks = ticks[tick_index:depth]
    # else:
    #     # ticks = ticks[tick_index:depth]
    #     # Consuming all of the Y_real in the tick range, which is Token1 or the main stablecoin
    #     ticks = ticks[-depth:tick_index + 1]
    #
    # quantity = sum(ticks["totalValueLockedToken0"])
    # dolvol = sum(ticks["totalValueLockedToken1"])

    return Decimal(dolvol), Decimal(quantity)


def mci(is_bid: bool, mid_price: Decimal,
        ticks: pd.DataFrame, tick_spacing: int, decimals0: Decimal,
        decimals1: Decimal,
        sqrt_price_x96: Decimal, pair: str, date_val: str, flip_flag: bool,
        size_trade: float = None, level: int = None) -> [float, pd.DataFrame]:
    """
    Calculate MCI with caching support using functional approach.

    Args:
        is_bid (bool): True for bid/sell, False for ask/buy
        mid_price (Decimal): Mid price
        tick_liquidity (pd.Series): Liquidity at each tick
        ticks (pd.Series): Available ticks
        tick_spacing (int): Spacing between ticks
        decimals0 (int): Decimals for token0
        decimals1 (int): Decimals for token1
        sqrt_price_x96 (Decimal): Square root of the price scaled by 2^96
        pair (str): Trading pair identifier
        flip_flag (bool): Flag to indicate if price should be flipped
        size_trade (float, optional): Size of trade in token1 terms
        level (int, optional): Number of price levels to use
        cache_dir (str, optional): Directory for caching results
        levels_df (pd.DataFrame, optional): DataFrame of previously calculated levels

    Returns:
        Union[float, tuple[float, pd.DataFrame]]: MCI value or MCI value with updated levels DataFrame

    """

    if size_trade is not None and level is not None:
        raise ValueError("Cannot specify both size_trade and level")

    if level:
        # Validate that the number of levels is within the available levels in ticks
        if level > (ticks.shape[0] / 2) - 1:
            print(f"{level} levels to consume liquidity is greater than "
                  f"the {int(ticks.shape[0] / 2)} available levels in ticks")

            return 0

    # Calculate for trade sizes
    dolvol, quantity = get_quantity(
        is_bid, ticks, tick_spacing,
        decimals0, decimals1, sqrt_price_x96,
        size_trade=size_trade, level=level
    )

    if dolvol == 0 or quantity == 0:
        return 0

    # # To get the right mid price
    if flip_flag:
        mid_price = 1 / mid_price
        quantity, dolvol = dolvol, quantity

    # try:
    vwap = get_vwap(dolvol, quantity, flip_flag)
    vwapm = get_vwapm(vwap, mid_price)
    # except (InvalidOperation, DivisionUndefined):
    #
    #     # Tries again
    #     mci_value = mci(is_bid, mid_price, ticks, tick_spacing, decimals0, decimals1,
    #                     sqrt_price_x96, pair, date_val, False, size_trade, level)
    #
    #     return mci_value

    mci_value = get_mci(is_bid, vwapm, dolvol, quantity, flip_flag)

    if mci_value < 0:
        print(f"Negative MCI: {mci_value} for pair {pair} for level {level} for date {date_val}"
              f" with a tick spacing of {tick_spacing}")

    # assert mci_value >= 0, f"Negative MCI: {mci_value}"
    return float(mci_value)
