#!/usr/bin/env python3

#
# Based on: https://github.com/atiselsts/uniswap-v3-liquidity-math
#

from decimal import Decimal, getcontext, ROUND_DOWN, ROUND_HALF_UP
import math

# Set precision for Decimal operations
getcontext().prec = 120  # Further increased precision
getcontext().rounding = ROUND_HALF_UP  # Set rounding method to down to match financial rounding

# Constants
Q96 = Decimal(2) ** Decimal(96)


def get_tick_at_sqrt_price(sqrt_price_x96):
    sqrt_price_x96 = Decimal(sqrt_price_x96)
    ratio = (sqrt_price_x96 / Q96) ** 2
    tick = math.floor(math.log(ratio) / math.log(Decimal('1.0001')))
    return tick


def get_token_amounts(liquidity, sqrt_price_x96, tick_low, tick_high, decimal0, decimal1):
    liquidity = Decimal(liquidity)
    sqrt_price_x96 = Decimal(sqrt_price_x96)
    tick_low = Decimal(tick_low)
    tick_high = Decimal(tick_high)
    decimal0 = int(decimal0)
    decimal1 = int(decimal1)

    sqrt_ratio_a = Decimal('1.0001') ** (tick_low / 2)
    sqrt_ratio_b = Decimal('1.0001') ** (tick_high / 2)

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

    amount0_human = ((amount0 / (10 ** decimal0)).quantize(Decimal('1.' + '0' * decimal0), rounding=ROUND_HALF_UP)).normalize()
    amount1_human = ((amount1 / (10 ** decimal1)).quantize(Decimal('1.' + '0' * decimal1), rounding=ROUND_HALF_UP)).normalize()

    print("Amount Token0 in lowest decimal:", amount0)
    print("Amount Token1 in lowest decimal:", amount1)
    print("Amount Token0:", amount0_human)
    print("Amount Token1:", amount1_human)

    return amount0_human, amount1_human


# Example usage
# You would typically call get_token_amounts within an async context
# For example, asyncio.run(get_token_amounts(...))

# Test the function with the minted position: https://etherscan.io/tx/0x33272ae3a631c4ba9e1ee0bafcf69653c71ba5a7e7e446b949c132157e42abe1
# at block number 19818233


# Example of USDC / WETH pool current tick range (11-3-22 5pm PST) Liquidity from pool current sqrtPrice LowTick  upTick  token decimals
amount0, amount1 = get_token_amounts(3374594928805046, 1429009184740760554001848125080209,
                                     196000, 196040, 6, 18)

expected_amount0 = Decimal('252.386134')
expected_amount1 = Decimal('0.039552002356500967')

print("Difference in amount0:", amount0 - expected_amount0)
print("Difference in amount1:", amount1 - expected_amount1)
