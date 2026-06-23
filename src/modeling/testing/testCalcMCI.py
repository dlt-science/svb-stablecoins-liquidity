import pandas as pd
from decimal import Decimal, getcontext, ROUND_UP, ROUND_DOWN, ROUND_FLOOR, ROUND_CEILING, ROUND_05UP, ROUND_HALF_UP
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

getcontext().prec = 20  # High precision for financial calculations
# getcontext().rounding = ROUND_FLOOR  # Consistent rounding method

# Constants
TICK_BASE = Decimal('1.0001')
Q96 = Decimal(2) ** 96
MCI_TOLERANCE = Decimal('1e-6')  # Tolerance for negative MCI values


def get_tick_at_sqrt_price(sqrt_price_x96):
    sqrt_price_x96 = Decimal(sqrt_price_x96)
    ratio = (sqrt_price_x96 / Q96) ** 2
    tick = (ratio.ln() / TICK_BASE.ln()).to_integral_value(rounding=ROUND_HALF_UP)
    return int(tick)


def tick_to_price_simple(tick):
    price = TICK_BASE ** Decimal(tick)
    return price


def tick_to_price(tick, decimals):
    price = (TICK_BASE ** Decimal(tick)) * (10 ** Decimal(decimals))
    return price.normalize()


def get_token_amounts(liquidity, sqrt_price_x96, tick_low, tick_high):
    """
    Based on https://blog.uniswap.org/uniswap-v3-math-primer-2 for the calculation of token0 and token1 amounts
    """

    liquidity = Decimal(liquidity)
    sqrt_price_x96 = Decimal(sqrt_price_x96)
    tick_low = Decimal(int(tick_low))
    tick_high = Decimal(int(tick_high))

    sqrt_ratio_a = TICK_BASE ** (tick_low / 2)
    sqrt_ratio_b = TICK_BASE ** (tick_high / 2)

    current_tick = get_tick_at_sqrt_price(sqrt_price_x96)
    sqrt_price = sqrt_price_x96 / Q96

    amount0 = Decimal(0)
    amount1 = Decimal(0)

    if current_tick < tick_low:
        amount0 = (liquidity * ((sqrt_ratio_b - sqrt_ratio_a) / (sqrt_ratio_a * sqrt_ratio_b)))
    elif current_tick >= tick_high:
        amount1 = (liquidity * (sqrt_ratio_b - sqrt_ratio_a))
    elif tick_low <= current_tick < tick_high:
        amount0 = (liquidity * ((sqrt_ratio_b - sqrt_price) / (sqrt_price * sqrt_ratio_b)))
        amount1 = (liquidity * (sqrt_price - sqrt_ratio_a))

    return amount0, amount1


def CalcSwapDeltas(is_bid, liquidity, lower_bound, upper_bound, sqrt_price_x96, decimals0, decimals1):
    """
    Function to calculate the execution deltas of a swap in a tick that has a given liquidity and real token quantities
    quantities0, quantities1: the real token quantities
    size: the size of the swap
    type: 'buy' or 'sell'
    liquidity: the squared liquidity in the tick
    price_lower, price_upper: the lower and upper bounds of the tick
    outputs delta0,delta1 i.e. the token0 and token1 quantities that have to be added or removed from the pool.
    """
    # # Convert the current price to tick
    # # cross_tick = price_to_tick(current_price, decimals0 - decimals1)
    # quantities0, quantities1 = liquidity_to_quantities(lower_bound, upper_bound, decimals0, decimals1, liquidity,
    #                                                    sqrt_price, cross_tick)

    liquidity = Decimal(liquidity)
    sqrt_price_x96 = Decimal(sqrt_price_x96)
    decimals0 = Decimal(int(decimals0))
    decimals1 = Decimal(int(decimals1))

    quantities0, quantities1 = get_token_amounts(liquidity, sqrt_price_x96, lower_bound, upper_bound)

    if quantities0 == 0 and quantities1 == 0:
        return 0, 0

    # Square root prices for lower and upper bounds
    sqrt_p_a = tick_to_price_simple(int(lower_bound) / 2)
    sqrt_p_b = tick_to_price_simple(int(upper_bound) / 2)

    offset0 = liquidity / sqrt_p_b
    offset1 = liquidity * sqrt_p_a

    virtual0 = quantities0 + offset0
    virtual1 = quantities1 + offset1

    if is_bid:
        delta1 = quantities1
        delta0 = (liquidity ** 2) / (virtual1 - delta1) - virtual0

    else:
        # A buy order will consume token0 by the provided token1
        # Calculate the size in token0
        delta0 = quantities0
        delta1 = -(liquidity ** 2 / (virtual0 + delta0) - virtual1)

    # Adjust deltas to normal scale for output
    delta0 /= (10 ** decimals0)
    delta1 /= (10 ** decimals1)

    return delta0, delta1


def execute_swap(level, is_bid, tick_liquidity,
                 ticks, tick_spacing, decimals0, decimals1, sqrt_price_x96):
    """
    Simulate the execution of a swap in a Uniswap V3 pool
    """
    current_tick = get_tick_at_sqrt_price(sqrt_price_x96)
    tick_index = ticks[ticks == (current_tick // tick_spacing) * tick_spacing]
    try:
        tick_index = tick_index.index[0]
    except IndexError:
        print(f"Index error: {tick_index}")

    deltas0 = []
    deltas1 = []

    while level > 0 and len(ticks) > tick_index >= 0:

        lower_bound = ticks[tick_index]
        upper_bound = lower_bound + tick_spacing

        delta0, delta1 = CalcSwapDeltas(is_bid, tick_liquidity[tick_index], lower_bound, upper_bound,
                                        sqrt_price_x96, decimals0, decimals1)

        # delta0 is negative for buy and sell orders. delta1 is always positive
        deltas0.append(delta0)
        deltas1.append(delta1)

        level -= 1

        if is_bid:
            tick_index -= 1
        else:
            tick_index += 1

    return deltas0, deltas1


def get_quantity(level: int,
                 is_bid: bool, tick_liquidity: pd.Series,
                 ticks: pd.Series, tick_spacing: int, decimals0: Decimal,
                 decimals1: Decimal,
                 sqrt_price_x96: Decimal) -> tuple[Decimal, Decimal]:
    """
    Delta1: Represents the delta for the swap in stablecoin value / dollar value. E.g., USDC, USDT, etc.,
    which for this case would represent quantity
    Delta0: Represents the delta for the swap in the quantity of token0, WETH, etc., which for this case
    would represent dolvol
    """
    # execute swap function
    # token 1 balance: x1
    # token 2 balance: x2
    # calculate dolvol of token 1
    # dolvol = Decimal(xxxxx)  # add up ticks

    deltas0, deltas1 = execute_swap(level, is_bid, tick_liquidity, ticks,
                                    tick_spacing, decimals0, decimals1, sqrt_price_x96)

    # scale_factor = 10 ** Decimal(int(decimals0) - int(decimals1))
    #
    # quantity = Decimal(sum(Decimal(d) * scale_factor for d in deltas0))
    # dolvol = Decimal(sum(Decimal(d) * scale_factor for d in deltas1))

    quantity = sum(deltas0)
    dolvol = sum(deltas1)

    # # calculate quantity of token 2 to be traded for token 1 balance to move from x1 to x1 - dolvol
    # dolvol = total_liquidity * percentage_consume
    # x2_after = Decimal(xxxxx)  # add up ticks
    # quantity = x2 - x2_after

    return dolvol, quantity


def get_vwap(dolvol: Decimal, quantity: Decimal) -> Decimal:
    return Decimal(dolvol / quantity)


def get_vwapm(vwap: Decimal, mid_price: Decimal) -> Decimal:
    return Decimal(vwap / mid_price).ln()


def get_mci(is_bid: bool, vwapm: Decimal, dolvol: Decimal) -> Decimal:
    return (-1) ** is_bid * vwapm / dolvol * Decimal(1e4)


def mci(is_bid: bool, level: int, mid_price: Decimal,
        tick_liquidity: pd.Series,
        ticks: pd.Series, tick_spacing: int, decimals0: Decimal,
        decimals1: Decimal,
        sqrt_price_x96: Decimal) -> Decimal:
    dolvol, quantity = get_quantity(level, is_bid,
                                    tick_liquidity,
                                    ticks, tick_spacing,
                                    decimals0,
                                    decimals1,
                                    sqrt_price_x96)

    vwap = get_vwap(dolvol, quantity)
    vwapm = get_vwapm(vwap, mid_price)
    mci_value = get_mci(is_bid, vwapm, dolvol)

    if mci_value < 0:
        print(f"Negative MCI: {mci_value}")

    # assert mci_value >= 0, f"Negative MCI: {mci_value}"
    return mci_value


tick_spacing = 10
decimals0 = Decimal(6)
decimals1 = Decimal(18)
# sqrt_price_x96 = Decimal('79228528144533646618879')
scale_factor = 10 ** (decimals0 - decimals1)
# current_price = ((Decimal(sqrt_price_x96) / Q96) ** Decimal(2)) * scale_factor
level = 5
# ticks = pd.Series([-276390, -276380, -276370, -276360, -276350, -276340, -276330, -276320,
#                    -276310, -276300, -276290, -276280])

ticks = pd.Series(np.arange(-500000, 500000, tick_spacing))
tick_liquidity = pd.Series([4.351413179896899e+17] * len(ticks))
# ticks_price = [tick_to_price(tick, decimals0 - decimals1) for tick in ticks]

# sns.lineplot(x=ticks_price, y=tick_liquidity)
# plt.show()

# current_prices = np.arange(0.5, 1.5, 0.05)
current_prices = np.logspace(-1, 4, 10)
# current_prices = [0.99, 1, 1.01]
Y = []
X = []

for current_price in current_prices:
    sqrt_price_x96 = int(np.sqrt(Decimal(current_price)) * Decimal(10 ** ((decimals1 - decimals0) / 2) * 2 ** 96))

    dolvol, quantity = get_quantity(level, False,
                                    tick_liquidity,
                                    ticks, tick_spacing,
                                    decimals0,
                                    decimals1,
                                    sqrt_price_x96)

    vwap = get_vwap(dolvol, quantity)
    vwapm = get_vwapm(vwap, Decimal(current_price))

    Y.append(vwap / Decimal(current_price))

    X.append(current_price)

# Convert Decimal to float for plotting
Y_float = [float(y) for y in Y]

# Plot with full decimal values and legend on top of each bar
plt.figure(figsize=(10, 6))
sns.lineplot(x=X, y=Y, palette='viridis')
plt.xlabel('Price')
plt.ylabel('vwap / Price')
plt.title('Bar Plot of X vs Y')
# plt.ylim(0.98, 1.03)

# Adding full decimal value labels on top of bars for clarity
for i, v in enumerate(Y):
    plt.text(X[i], Y_float[i] + 0.00005, str(v), ha='center', va='bottom', fontsize=8)

plt.show()

# buy_MCI = mci(is_bid=False, level=level, mid_price=current_price,
#               tick_liquidity=tick_liquidity,
#               ticks=ticks, tick_spacing=tick_spacing,
#               decimals0=decimals0,
#               decimals1=decimals1,
#               sqrt_price_x96=sqrt_price_x96)
#
# sell_MCI = mci(is_bid=True, level=level, mid_price=current_price,
#                tick_liquidity=tick_liquidity,
#                ticks=ticks, tick_spacing=tick_spacing,
#                decimals0=decimals0,
#                decimals1=decimals1,
#                sqrt_price_x96=sqrt_price_x96)
#
# MCI_imbalance = (buy_MCI - sell_MCI) / (buy_MCI + sell_MCI)
#
# print(f"MCI Imbalance: {MCI_imbalance}")
