import argparse
import time
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
from gql import gql

import aiohttp
import json
import os

from tqdm import tqdm
from typing import Dict, List, TypedDict, Optional, Tuple
from decimal import Decimal
# from getTicks import get_token_amounts

from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

# Equivalent of tickets in a range in Uniswap v3
TICK_MAP = {100: 1, 500: 10, 3000: 60, 10000: 200}

# Constants
MAX_INT128 = (2**128) - 1
MIN_TICK = -887272
MAX_TICK = 887272

SUBGRAPH_ID = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"

# API key rotation: keys are tried in order; when one is exhausted the next is used.
_API_KEYS = [
    os.getenv("UNISWAP_SUBGRAPH_API_KEY"),
]
_current_key_index = 0


def _get_current_url() -> str:
    """Return the URL for the currently active API key."""
    api_key = _API_KEYS[_current_key_index]
    return f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/{SUBGRAPH_ID}"


def _rotate_api_key() -> bool:
    """
    Advance to the next API key in the rotation list.

    Returns True if a new key is available, False if all keys are exhausted.
    """
    global _current_key_index
    if _current_key_index < len(_API_KEYS) - 1:
        _current_key_index += 1
        print(
            f"[API key rotation] Switching to key index {_current_key_index} "
            f"(UNISWAP_SUBGRAPH_API_KEY_{4 + _current_key_index})"
        )
        return True
    print("[API key rotation] All API keys exhausted.")
    return False


headers = {
    "Content-Type": "application/json"
}

# To store the ticks data as it gets processed
class ProcessedTick(TypedDict):
    tickIdx: int
    liquidityActive: Decimal
    liquidityNet: Decimal
    price0: Decimal
    price1: Decimal
    isCurrent: bool

class Direction(Enum):
    ASC = "ASC"
    DESC = "DESC"

class TickProcessed(TypedDict):
    tickIdx: int
    liquidityActive: int  # Using int for JSBI equivalent
    liquidityNet: int    # Using int for JSBI equivalent
    price0: float
    price1: float
    isCurrent: bool

class BarChartTick(TypedDict):
    tickIdx: int
    liquidityActive: float
    liquidityLockedToken0: float
    liquidityLockedToken1: float
    price0: float
    price1: float
    isCurrent: bool

class GraphTick(TypedDict):
    tickIdx: str
    liquidityGross: str
    liquidityNet: str

def calculate_ticks(fee_tier, lower, upper):
    return abs(upper - lower) // TICK_MAP[int(fee_tier)]


async def query_data_async(query, variables=None):
    """
    Queries the graph with the given query asynchronously using aiohttp.

    Automatically rotates through the available API keys (UNISWAP_SUBGRAPH_API_KEY_4,
    _5, _6) when a response does not contain a 'data' key, which indicates that the
    current key's query quota has been exhausted.
    """

    # Extract the original string representation of the query from the DocumentNode
    query_str = query.payload['query']
    payload = {"query": query_str}

    if variables:
        payload["variables"] = variables

    while True:
        current_url = _get_current_url()
        async with aiohttp.ClientSession() as session:
            async with session.post(current_url, headers=headers, data=json.dumps(payload)) as response:
                result = await response.json()

        # If the response contains 'data', the key is still valid – return immediately.
        if "data" in result:
            return result

        # 'data' is missing: the key is likely exhausted.  Log the issue and rotate.
        print(
            f"[API key rotation] 'data' not found in response "
            f"(key index {_current_key_index}). Response: {result}"
        )
        if not _rotate_api_key():
            # All keys are exhausted – raise so the caller can handle it.
            raise RuntimeError(
                f"All API keys exhausted. Last response: {result}"
            )


async def get_position_status_async(row, block_number: int):
    position_id = str(row["position_nft_id"])

    query = gql(f"""
            query {{            
              position(id: "{position_id}", block: {{number: {block_number}}}) {{
                id
                liquidity
                owner
                withdrawnToken0
                withdrawnToken1
                depositedToken0
                depositedToken1
                tickLower {{
                  tickIdx
                }}
                tickUpper {{
                  tickIdx
                }}
                pool {{
                  id
                }}
              }}
            }}
             """)

    response = await query_data_async(query)
    response = response['data']

    # if the list is empty means that we have reached the end of the data
    if not response["position"]:
        return None

    # Generate the pandas DataFrame
    position = response["position"]
    position_df = pd.DataFrame({
        "pool_address": str(position["pool"]["id"]),
        "position_nft_id": str(position["id"]),
        "depositer_address": str(position["owner"]),
        "tickLower_tickIdx": int(position["tickLower"]["tickIdx"]),
        "tickUpper_tickIdx": int(position["tickUpper"]["tickIdx"]),
        "deposited_token0": float(position["depositedToken0"]),
        "deposited_token1": float(position["depositedToken1"]),
        "withdrawn_token0": float(position["withdrawnToken0"]),
        "withdrawn_token1": float(position["withdrawnToken1"]),
        "liquidity": float(position["liquidity"]),
        "position_timestamp": row["position_timestamp"] if "position_timestamp" in row.index else np.nan,
        "readable_position_timestamp": row[
            "readable_position_timestamp"] if "readable_position_timestamp" in row else np.nan,
        "type_of_transaction": row["type_of_transaction"] if "type_of_transaction" in row.index else np.nan
    }, index=[0])

    return position_df


async def get_all_positions_current_time(pool_id: str):
    """
    Gets all the positions for a given liquidity pool.

    Parameters
    ----------
    pool_id : str
        The pool id of the liquidity pool.
    skip : int
        The number of positions to skip. Skip is used to paginate through the positions
        since the graph only allows 1000 positions to be returned at a time.
        Start the skip at 0 and increment by 1000 until the query returns an empty list.
    """

    all_positions = []
    skip = 0

    while True:

        query = gql(f"""
                query {{
                    positions(first: 1000, skip: {skip}, where: {{pool: "{pool_id}", liquidity_gt: "0"}}) {{
                        id
                        owner
                        tickLower {{
                        tickIdx
                        }}
                        tickUpper {{
                        tickIdx
                        }}
                        liquidity
                        depositedToken0
                        depositedToken1
                        withdrawnToken0
                        withdrawnToken1
                    }}
                }}
        """)

        response = await query_data_async(query)

        # if the list is empty means that we have reached the end of the data
        if not response["positions"]:
            break

        all_positions.extend(response["positions"])

        # increment the skip by 1000 because we are getting 1000 positions at a time
        skip += 1000

    return all_positions


async def get_pools_with_a_token_async(token_id: str, num_pools=10):
    """
       By default, it gets the top 10 pools that at least one of the
       tokens is the token we are interested in.

       """

    query = gql(f"""
       query {{
         token(id: "{token_id}") {{
           symbol
           name
           whitelistPools(orderBy: liquidity, orderDirection: desc, first: {num_pools}) {{
             id
             token0 {{
               id
               symbol
               decimals
             }}
             token1 {{
               id
               symbol
               decimals
             }}
             liquidity
             feeTier
             totalValueLockedUSD
             totalValueLockedToken1
             totalValueLockedToken0
             volumeUSD
           }}
         }}
       }}
       """)

    response = await query_data_async(query)
    response = response['data']

    # if the token is not found, return 0 meaning no pools within the whitelistpools
    if response["token"] is None:
        return None

    pools_result = []
    for pool in response['token']['whitelistPools']:
        pools = {}
        token_pair = pool['token0']['symbol'] + "/" + pool['token1']['symbol']
        pools['Pool ID'] = pool['id']
        pools['Pair'] = token_pair
        pools['Liquidity'] = pool['liquidity']
        pools['Total Value Locked (USD)'] = pool['totalValueLockedUSD']
        pools['Fee Tier'] = pool['feeTier']
        pools['Fee tier (%)'] = int(pool['feeTier']) / 10000

        # Filter the total locked for the token we are interested in
        # if pool['token0']['id'] == token_id:
        pools["totalValueLockedToken0"] = pool['totalValueLockedToken0']
        pools["totalValueLockedToken1"] = pool['totalValueLockedToken1']
        pools['token1_symbol'] = pool['token1']['symbol']

        # else:
        #     pools["totalValueLockedToken1"] = pool['totalValueLockedToken1']
        #     pools["totalValueLockedToken0"] = pool['totalValueLockedToken0']
        pools['token0_symbol'] = pool['token0']['symbol']

        pools_result.append(pools)

    # Convert dict to DataFrame for plotting
    pools_df = pd.DataFrame(pools_result)

    return pools_df


async def get_all_positions_in_block_async(pool_address: str, block_number: int):
    """
    Gets all the positions for a given liquidity pool asynchronously.

    Parameters
    ----------
    pool_id : str
        The pool id of the liquidity pool.
    skip : int
        The number of positions to skip. Skip is used to paginate through the positions
        since the graph only allows 1000 positions to be returned at a time.
        Start the skip at 0 and increment by 1000 until the query returns an empty list.
    """

    all_positions = []
    skip = 0

    while True:

        query = gql(f"""
        {{
          positionSnapshots(
            where: {{liquidity_gt: "0", 
                    pool: "{pool_address}"}},
            orderBy: timestamp
            orderDirection: asc
            first: 1000
            skip: {skip}
            block: {{number: {block_number}}}
          ) {{
            timestamp
            owner
            liquidity
            depositedToken0
            depositedToken1
            id
            blockNumber
            position {{
              tickLower {{
                tickIdx
              }}
              tickUpper {{
                tickIdx
              }}
              id
            }}
          }}
        }}
        """)

        response = await query_data_async(query)
        response = response['data']

        # if the list is empty means that we have reached the end of the data
        if not response["positionSnapshots"]:
            # pool_data = response["positionSnapshots"]["pool"]
            break

        all_positions.extend(response["positionSnapshots"])

        # increment the skip by 1000 because we are getting 1000 positions at a time
        skip += 1000

        if skip > 5000:
            break

    return all_positions


async def get_pool_volume_in_block_async(pool_id: str, block_number: int = None):

    # Choose query and variables based on block_number
    if block_number is not None:
        query = gql(f"""
        query {{
          pool(id: "{pool_id}", block: {{number: {block_number}}}) {{
            volumeUSD
          }}
        }}
        """)
    else:
        query = gql(f"""
        query {{
          pool(id: "{pool_id}") {{
            volumeUSD
          }}
        }}
        """)

    response = await query_data_async(query)
    response = response['data']

    if response["pool"] is None:
        return np.nan

    return response['pool']['volumeUSD']


async def generate_positions_df_async(positions, pool):
    fee_tier = int(pool["feeTier"])
    tick_spacing = TICK_MAP[fee_tier]

    data = []

    for pos in positions:
        # Calculate the number of ticks in the range
        tickLower = int(pos["tickLower"]["tickIdx"])
        tickUpper = int(pos["tickUpper"]["tickIdx"])
        depositer_address = pos["owner"]

        num_ticks = calculate_ticks(fee_tier, tickLower, tickUpper)

        # for bound in bounds:
        data.append({
            "depositer_address": depositer_address,
            "fee_tier": int(fee_tier),
            "tickLower_tickIdx": int(tickLower),
            "tickUpper_tickIdx": int(tickUpper),
            "deposited_token0": float(pos["depositedToken0"]),
            "deposited_token1": float(pos["depositedToken1"]),
            "num_ticks": num_ticks,
            "decimals_token0": int(pool["token0"]["decimals"]),
            "decimals_token1": int(pool["token1"]["decimals"]),
            "price0": float(pool["token0Price"]),
            "current_price_token0_in_token1": float(pool["token1Price"]),
            "token0Price": float(pool["token0Price"]),
            "token1Price": float(pool["token1Price"]),
            "sqrtPrice": float(pool["sqrtPrice"]),
            "totalValueLockedToken0": float(pool["totalValueLockedToken0"]),
            "totalValueLockedToken1": float(pool["totalValueLockedToken1"])
        })

    # Create a DataFrame from the positions data
    df = pd.DataFrame(data)

    # Add other values to dataframe
    df["tick_spacing"] = int(tick_spacing)

    # Add the token symbols
    df["token0_symbol"] = pool["token0"]["symbol"]
    df["token1_symbol"] = pool["token1"]["symbol"]

    return df


async def generate_positionsSnapshot_df(positions):
    data = []

    for pos in positions:
        data.append({
            "position_timestamp": pos["timestamp"],
            "blockNumber": pos["blockNumber"],
            # Convert the string Unix timestamp to a human-readable date format
            "readable_position_timestamp": datetime.fromtimestamp(int(pos["timestamp"])).strftime('%Y-%m-%d %H:%M:%S'),
            "position_id": str(pos["id"]),
            "position_nft_id": int(pos["position"]["id"]),
            "depositer_address": str(pos["owner"]),
            # "depositer_address": pos["position"]["owner"],
            "tickLower_tickIdx": int(pos["position"]["tickLower"]["tickIdx"]),
            "tickUpper_tickIdx": int(pos["position"]["tickUpper"]["tickIdx"]),
            "deposited_token0": float(pos["depositedToken0"]),
            "deposited_token1": float(pos["depositedToken1"]),
            "liquidity": float(pos["liquidity"]),
            # "deposited_token0": float(pos["position"]["depositedToken0"]),
            # "deposited_token1": float(pos["position"]["depositedToken1"]),
        })

    df = pd.DataFrame(data)

    return df


async def generate_pool_df(pool, position_df):
    fee_tier = int(pool["feeTier"])
    tick_spacing = TICK_MAP[fee_tier]

    for index, row in position_df.iterrows():
        num_ticks = calculate_ticks(fee_tier, int(row["tickLower_tickIdx"]), int(row["tickUpper_tickIdx"]))
        position_df.at[index, "num_ticks"] = num_ticks

    position_df["fee_tier"] = fee_tier

    position_df["totalValueLockedToken1"] = float(pool["totalValueLockedToken1"])
    position_df["totalValueLockedToken0"] = float(pool["totalValueLockedToken0"])
    position_df["token1Price"] = float(pool["token1Price"])
    position_df["token0Price"] = float(pool["token0Price"])
    position_df["sqrtPrice"] = float(pool["sqrtPrice"])

    position_df["tick_spacing"] = tick_spacing
    position_df["token0_symbol"] = pool["token0"]["name"]
    position_df["token1_symbol"] = pool["token1"]["name"]
    position_df["decimals_token0"] = int(pool["token0"]["decimals"])
    position_df["decimals_token1"] = int(pool["token1"]["decimals"])

    position_df["24hr Volume (USD)"] = float(pool["volumeUSD"])

    return position_df


async def update_positionSnapshots_balances(transactions_data, df):
    # output = net_balance_wallet(transactions_data)

    df_new_rows = []
    for burn in transactions_data['burns']:

        mask = (
            # (df['position_nft_id'] == burn['position_nft_id']) & (df['current_positions_alive'] != 1)
            (df['position_nft_id'] == burn['position_nft_id'])
        )

        # Check if at least one row exists with the filtered criteria
        if mask.sum() > 0:

            # Convert the found row to a float
            df.loc[mask, 'deposited_token0'] = df.loc[mask, 'deposited_token0'].astype(float)
            df.loc[mask, 'deposited_token1'] = df.loc[mask, 'deposited_token1'].astype(float)
            df.loc[mask, 'liquidity'] = df.loc[mask, 'liquidity'].astype(float)

            # Remove the mint amount from the deposited_token0 and deposited_token1 columns
            # because the minted liquidity has already been recorded by adding up the burns
            df.loc[mask, 'position_timestamp'] = burn['timestamp']
            df.loc[mask, 'readable_position_timestamp'] = datetime.fromtimestamp(int(burn['timestamp'])).strftime(
                '%Y-%m-%d %H:%M:%S')
            df.loc[mask, 'position_nft_id'] = burn['position_nft_id']
            df.loc[mask, 'blockNumber'] = burn['transaction']["blockNumber"]
            df.loc[mask, 'deposited_token0'] += float(burn['amount0'])
            df.loc[mask, 'deposited_token1'] += float(burn['amount1'])
            df.loc[mask, 'liquidity'] += float(burn['amount'])
            df.loc[mask, 'current_positions_alive'] = 0
            df.loc[mask, 'type_of_transaction'] = 'burn'

        # Otherwise, append a new row
        else:
            new_row = {
                'position_timestamp': burn['timestamp'],
                'readable_position_timestamp': datetime.fromtimestamp(int(burn['timestamp'])).strftime(
                    '%Y-%m-%d %H:%M:%S'),
                'position_nft_id': burn['position_nft_id'],
                'blockNumber': burn['transaction']["blockNumber"],
                'depositer_address': burn['origin'],
                'tickLower_tickIdx': burn['tickLower'],
                'tickUpper_tickIdx': burn['tickUpper'],
                'deposited_token0': float(burn['amount0']),
                'deposited_token1': float(burn['amount1']),
                'liquidity': float(burn['amount']),
                'current_positions_alive': 0,
                'type_of_transaction': 'burn'
            }
            df_new_rows.append(new_row)

    if df_new_rows:
        df = pd.concat([df, pd.DataFrame(df_new_rows)])

    # Reset for the next loop
    df_new_rows = []
    for mint in transactions_data['mints']:

        mask = (
            # (df['position_nft_id'] == mint['position_nft_id']) & (df['current_positions_alive'] != 1)
            (df['position_nft_id'] == mint['position_nft_id'])

        )

        # Check if at least one row exists with the filtered criteria
        if mask.sum() > 0:

            # Convert the found row to a float
            df.loc[mask, 'deposited_token0'] = df.loc[mask, 'deposited_token0'].astype(float)
            df.loc[mask, 'deposited_token1'] = df.loc[mask, 'deposited_token1'].astype(float)
            df.loc[mask, 'liquidity'] = df.loc[mask, 'liquidity'].astype(float)

            # Remove the mint amount from the deposited_token0 and deposited_token1 columns
            # because the minted liquidity has already been recorded by adding up the burns
            df.loc[mask, 'position_timestamp'] = mint['timestamp']
            df.loc[mask, 'readable_position_timestamp'] = datetime.fromtimestamp(int(mint['timestamp'])).strftime(
                '%Y-%m-%d %H:%M:%S')
            df.loc[mask, 'position_nft_id'] = mint['position_nft_id']
            df.loc[mask, 'blockNumber'] = mint['transaction']["blockNumber"]
            df.loc[mask, 'deposited_token0'] -= float(mint['amount0'])
            df.loc[mask, 'deposited_token1'] -= float(mint['amount1'])
            df.loc[mask, 'liquidity'] -= float(mint['amount'])
            df.loc[mask, 'current_positions_alive'] = 0

            # If the position has a negative liquidity, then it is remaining liquidity
            # that is burned at a later date outside the date range that we are looking at
            # Therefore, we should convert the position to a positive liquidity
            # . It is a position that is still alive
            if df.loc[mask, 'liquidity'].values[0] < 0:
                df.loc[mask, 'liquidity'] *= -1

            # If the positions have negative amounts, but liquidity is 0, then
            # the difference is because of impermanent loss
            if df.loc[mask, 'deposited_token0'].values[0] < 0 < df.loc[mask, 'liquidity'].values[0]:
                df.loc[mask, 'deposited_token0'] *= -1

            if df.loc[mask, 'deposited_token1'].values[0] < 0 < df.loc[mask, 'liquidity'].values[0]:
                df.loc[mask, 'deposited_token1'] *= -1

        # Otherwise, append a new row
        else:
            # Append the row because it means that the positions were minted before, and it is being burned at some
            # point in the future but past the end of the date range that we are looking at
            new_row = {
                'position_timestamp': mint['timestamp'],
                'readable_position_timestamp': datetime.fromtimestamp(int(mint['timestamp'])).strftime(
                    '%Y-%m-%d %H:%M:%S'),
                'position_nft_id': mint['position_nft_id'],
                'blockNumber': mint['transaction']["blockNumber"],
                'depositer_address': mint['origin'],
                'tickLower_tickIdx': mint['tickLower'],
                'tickUpper_tickIdx': mint['tickUpper'],
                'deposited_token0': float(mint['amount0']),
                'deposited_token1': float(mint['amount1']),
                'liquidity': float(mint['amount']),
                'current_positions_alive': 0,
                'type_of_transaction': 'mint'
            }
            df_new_rows.append(new_row)

    if df_new_rows:
        df = pd.concat([df, pd.DataFrame(df_new_rows)])

        # # Format position_nft_id to be an integer
        # df["position_nft_id"] = df["position_nft_id"].astype(int)

        # Forward fill the pool_address column
        df["pool_address"] = df["pool_address"].ffill()

        # Drop the rows whose liquidity is 0
        df["liquidity"] = df["liquidity"].astype(float)
        df = df[df["liquidity"] != 0]

    return df


async def get_pool_data_in_block(pool_id: str, block_number: int):
    query = gql(f"""
    query {{
      pool(id: "{pool_id}", block: {{number: {block_number}}}) {{
        id
        feeTier
        sqrtPrice
        token0Price
        token1Price
        token0 {{
          decimals
          name
          symbol
          id
        }}
        token1 {{
          decimals
          id
          name
          symbol
        }}
        totalValueLockedToken0
        totalValueLockedToken1
        totalValueLockedUSD
        volumeUSD
      }}
    }}
    """)

    response = await query_data_async(query)
    response = response['data']

    if response["pool"] is None:
        return None

    return response['pool']


async def get_pool_tvl_block(pool_id: str, block_number: int):
    query = gql(f"""
    query {{
      pool(id: "{pool_id}", block: {{number: {block_number}}}) {{
        id
        totalValueLockedUSD
      }}
    }}
    """)

    response = await query_data_async(query)
    response = response['data']

    if response["pool"] is None:
        return None

    return float(response['pool']['totalValueLockedUSD'])


async def get_mints_and_burns_within_date_range(address, start_timestamp, end_timestamp):
    """
    Gets all the mints and burns for a given liquidity pool in a given date range.
    """

    # Convert timestamps to unix timestamps
    start_timestamp = int(time.mktime(start_timestamp.timetuple()))
    end_timestamp = int(time.mktime(end_timestamp.timetuple()))

    query = gql("""
    query transactions($address: String!, $startTimestamp: BigInt!, $endTimestamp: BigInt!, $skipAmount: Int!) {
      mints(
        first: 1000
        skip: $skipAmount
        orderBy: timestamp
        orderDirection: desc
        where: {pool: $address, timestamp_gte: $startTimestamp, timestamp_lte: $endTimestamp, amount_gt: "0"}
        subgraphError: allow
      ) {
        timestamp
        transaction {
          id
          blockNumber
          __typename
        }
        id
        owner
        sender
        origin
        amount
        amount0
        amount1
        amountUSD
        tickLower
        tickUpper
        __typename
      }
      burns(
        first: 1000
        skip: $skipAmount
        orderBy: timestamp
        orderDirection: desc
        where: {pool: $address, timestamp_gte: $startTimestamp, timestamp_lte: $endTimestamp, amount_gt: "0"}
        subgraphError: allow
      ) {
        timestamp
        transaction {
          id
          blockNumber
          __typename
        }
        id
        owner
        origin
        amount
        amount0
        amount1
        amountUSD
        tickLower
        tickUpper
        __typename
      }
    }
    """)

    skip_amount = 0
    all_mints = []
    all_burns = []

    while True:
        params = {
            "address": address,
            "startTimestamp": start_timestamp,
            "endTimestamp": end_timestamp,
            "skipAmount": skip_amount
        }

        # Execute the GraphQL request
        response = await query_data_async(query, params)

        mints = response['data']["mints"]
        burns = response['data']["burns"]

        all_mints.extend(mints)
        all_burns.extend(burns)

        if not mints and not burns:
            break

        skip_amount += 1000

    if not all_mints and not all_burns:
        return None

    return {"mints": all_mints, "burns": all_burns}


async def get_crosstick_at_blocknumber(pool_id: str, block_number: int):
    """
    Get the crosstick at a given block number
    """

    query = gql(f"""
    query {{
      pool(id: "{pool_id}", block: {{number: {block_number}}}) {{
        id
        sqrtPrice
        tick
      }}
    }}
    """)

    response = await query_data_async(query)
    response = response['data']

    if response['pool'] is None:
        return None

    # Format the response as a dataframe
    df = pd.DataFrame(response['pool'], index=[0])

    # Add the block number
    df["blockNumber"] = block_number

    # Rename the columns
    df = df.rename(columns={"id": "pool_address", "tick": "current_tick"})

    return df


async def get_pool_data_at_unix_timestamp(pool_id: str, start_time: int):
    """
    The subgraph poolDay Data is from midnight UTC to midnight 24hrs, then changes to new day.

    Therefore, the data getting here is daily data.
    """

    all_data = []
    skip = 0

    while True:

        # Parameterized GraphQL query to avoid injection and improve maintainability
        query = gql("""
        query GetPoolData($poolId: String!, $skip: Int!, $startTime: Int!) {
          poolDayDatas(
            first: 1000
            skip: $skip
            orderBy: date
            orderDirection: asc
            where: {pool: $poolId, date_gte: $startTime}) 
            {
                date
                id
                volumeUSD
                tvlUSD
                feesUSD
                sqrtPrice
                tick
                liquidity
                pool {
                  feeTier
                  totalValueLockedToken0
                  totalValueLockedToken1
                  __typename
                  token0 {
                      symbol
                    }
                    token1 {
                      symbol
                    }
                }
                token0Price
                token1Price
          }
        }
        """)

        variables = {
            "poolId": pool_id,
            "startTime": start_time,
            "skip": skip
        }

        response = await query_data_async(query, variables)

        if not response['data']['poolDayDatas']:
            break

        # Format the response as a dataframe
        df = pd.DataFrame(response['data']['poolDayDatas'])

        all_data.append(df)

        if start_time in df['date'].values:
            break

        skip += 1000

    if not all_data:
        return None

    # Concatenate all the dataframes
    df = pd.concat(all_data)

    # Filter for the 24 hour volume closest to the start time unix epoch
    df['distance_from_start_time'] = abs(df['date'] - start_time)
    df = df.sort_values('distance_from_start_time')
    df = df.head(1)

    df['feeTier'] = df['pool'].iloc[0]['feeTier']
    df['totalValueLockedToken0'] = df['pool'].iloc[0]['totalValueLockedToken0']
    df['totalValueLockedToken1'] = df['pool'].iloc[0]['totalValueLockedToken1']
    df['pair'] = df['pool'].iloc[0]['token0']['symbol'] + "/" + df['pool'].iloc[0]['token1']['symbol']
    df["token0_symbol"] = df['pool'].iloc[0]['token0']['symbol']
    df["token1_symbol"] = df['pool'].iloc[0]['token1']['symbol']

    # Rename the columns
    df = df.rename(columns={"id": "pool_address", "tick": "cross_tick",
                            'feeTier': 'fee_tier'})

    # Drop the pool column
    df = df.drop(columns=['pool', 'pool_address'])

    # Add the poold address
    df["pool_address"] = pool_id

    return df

async def get_hourly_pool_data_at_unix_timestamp(pool_id: str, start_time: int):
    """
    The subgraph poolDay Data is from midnight UTC to midnight 24hrs, then changes to new day.

    Therefore, the data getting here is daily data.
    """

    all_data = []
    skip = 0

    while True:

        # Parameterized GraphQL query to avoid injection and improve maintainability
        query = gql("""
                query fetchPoolData($pool: String!, $startTime: Int!, $skip: Int!) {
                  poolHourDatas(
                    where: {periodStartUnix_gte: $startTime, periodStartUnix_lte: $startTime, pool: $pool}
                    orderBy: periodStartUnix
                    first: $limit
                    orderDirection: asc
                    skip: $skip
                  ) {
                    periodStartUnix
                    id
                    token0Price
                    token1Price
                    liquidity
                    tvlUSD
                    tick
                    sqrtPrice
                    close
                    txCount
                    pool {
                      token0 {
                        symbol
                        decimals
                      }
                      token1 {
                        symbol
                        decimals
                      }
                      feeTier
                      totalValueLockedToken0
                      totalValueLockedToken1
                    }
                  }
                }
                """)

        variables = {
            "pool": pool_id,
            "startTime": start_time,
            "skip": skip
        }

        response = await query_data_async(query, variables)

        if not response['data']['poolHourDatas']:
            break

        # Format the response as a dataframe
        df = pd.DataFrame(response['data']['poolHourDatas'])

        # Rename periodStartUnix to date
        df = df.rename(columns={"periodStartUnix": "date"})

        all_data.append(df)

        if start_time in df['date'].values:
            break

        skip += 1000

    if not all_data:
        return None

    # Concatenate all the dataframes
    df = pd.concat(all_data)

    # Filter for the 24 hour volume closest to the start time unix epoch
    df['distance_from_start_time'] = abs(df['date'] - start_time)
    df = df.sort_values('distance_from_start_time')
    df = df.head(1)

    df['feeTier'] = df['pool'].iloc[0]['feeTier']
    df['totalValueLockedToken0'] = df['pool'].iloc[0]['totalValueLockedToken0']
    df['totalValueLockedToken1'] = df['pool'].iloc[0]['totalValueLockedToken1']
    df['pair'] = df['pool'].iloc[0]['token0']['symbol'] + "/" + df['pool'].iloc[0]['token1']['symbol']
    df["token0_symbol"] = df['pool'].iloc[0]['token0']['symbol']
    df["token1_symbol"] = df['pool'].iloc[0]['token1']['symbol']

    # Rename the columns
    df = df.rename(columns={"id": "pool_address", "tick": "cross_tick",
                            'feeTier': 'fee_tier'})

    # Drop the pool column
    df = df.drop(columns=['pool', 'pool_address'])

    # Add the poold address
    df["pool_address"] = pool_id

    return df


async def get_pools_with_a_token_async(token_id: str, volume_USD: int = 1000000, start_time: int = 1677628800):
    """
       By default, it gets the top 10 pools that at least one of the
       tokens is the token we are interested in.

       """

    query = gql(f"""
       query {{
         token(id: "{token_id}") {{
                symbol
                name
                volumeUSD
                totalValueLockedUSD
                whitelistPools(
                  where: {{poolDayData_: {{date_gt: {start_time}, volumeUSD_gt: "{volume_USD}"}}
                  orderBy: volumeUSD
                  orderDirection: asc
                  first: 1000
                ) {{
                  id
                  token0 {{
                    decimals
                    symbol
                  }}
                  token1 {{
                    decimals
                    symbol
                  }}
                  totalValueLockedUSD
                  volumeUSD
                  totalValueLockedToken1
                  totalValueLockedToken0
                }}
              }}
       """)

    response = await query_data_async(query)
    response = response['data']

    # if the token is not found, return 0 meaning no pools within the whitelistpools
    if response["token"] is None:
        return None

    pools_result = []
    for pool in response['token']['whitelistPools']:
        pools = {}
        token_pair = pool['token0']['symbol'] + "/" + pool['token1']['symbol']
        pools['Pool ID'] = pool['id']
        pools['Pair'] = token_pair
        pools['Liquidity'] = pool['liquidity']
        pools['Total Value Locked (USD)'] = pool['totalValueLockedUSD']
        pools['Fee Tier'] = pool['feeTier']
        pools['Fee tier (%)'] = int(pool['feeTier']) / 10000

        # Filter the total locked for the token we are interested in
        # if pool['token0']['id'] == token_id:
        pools["totalValueLockedToken0"] = pool['totalValueLockedToken0']
        pools["totalValueLockedToken1"] = pool['totalValueLockedToken1']
        pools['token1_symbol'] = pool['token1']['symbol']

        # else:
        #     pools["totalValueLockedToken1"] = pool['totalValueLockedToken1']
        #     pools["totalValueLockedToken0"] = pool['totalValueLockedToken0']
        pools['token0_symbol'] = pool['token0']['symbol']

        pools_result.append(pools)

    # Convert dict to DataFrame for plotting
    pools_df = pd.DataFrame(pools_result)

    return pools_df


async def get_initialized_ticks(pool_address: str, block_number: int = None) -> List[Dict]:
    """
    Fetches all initialized ticks for a pool that have non-zero liquidity.
    Based on Uniswap V3 documentation.

    Args:
        pool_address: Address of the Uniswap V3 pool
        block_number: Optional block number to query historical data

    Returns:
        List of initialized ticks with their liquidity data
    """

    # Base query without block filter
    base_query = """
            query getTicks($poolAddress: String!) {
                ticks(
                    first: 1000
                    where: {
                        poolAddress: $poolAddress
                        liquidityNet_not: "0"
                    }
                    orderBy: tickIdx
                    orderDirection: asc
                ) {
                    tickIdx
                    liquidityGross
                    liquidityNet
                }
            }
        """

    # Query with block filter
    block_query = """
            query getTicks($poolAddress: String!, $blockNumber: Int!) {
                ticks(
                    first: 1000
                    where: {
                        poolAddress: $poolAddress
                        liquidityNet_not: "0"
                    }
                    block: { number: $blockNumber }
                    orderBy: tickIdx
                    orderDirection: asc
                ) {
                    tickIdx
                    liquidityGross
                    liquidityNet
                }
            }
        """

    # Choose query and variables based on block_number
    if block_number is not None:
        query = gql(block_query)
        variables = {
            "poolAddress": pool_address.lower(),
            "blockNumber": block_number
        }
    else:
        query = gql(base_query)
        variables = {
            "poolAddress": pool_address.lower()
        }

    response = await query_data_async(query, variables)
    return response['data']['ticks']


async def calculate_active_liquidity(
        pool_address: str,
        current_tick: int,
        tick_spacing: int,
        initial_liquidity: Decimal,
        block_number: int = None,
        TICK_RANGE: int = 100,
        sqrtPriceX96: Decimal = None,
        tickSpacing: int = None
) -> List[ProcessedTick]:
    """
    Calculates active liquidity for a range of ticks around the current tick.
    Implements the logic from Uniswap V3 documentation for liquidity calculation:
    https://docs.uniswap.org/sdk/v3/guides/advanced/active-liquidity

    Args:
        pool_address: Pool address
        current_tick: Current tick index
        tick_spacing: Tick spacing for the pool
        initial_liquidity: Current pool liquidity
        block_number: Optional block number for historical data
        TICK_RANGE: Calculate liquidity for surrounding ticks (100 ticks in each direction)

    Returns:
        List of processed ticks with active liquidity calculations
    """
    # Get initialized ticks from the subgraph
    initialized_ticks = await get_initialized_ticks(pool_address, block_number)

    # Create dictionary for O(1) tick lookup
    tick_dict = {int(tick['tickIdx']): tick for tick in initialized_ticks}

    # Initialize active tick
    active_tick_idx = (current_tick // tick_spacing) * tick_spacing

    processed_ticks: List[ProcessedTick] = []

    # Process current tick
    active_tick = {
        'tickIdx': active_tick_idx,
        'liquidityActive': initial_liquidity,
        'liquidityNet': Decimal(0),
        'quantity0': Decimal(0),
        'quantity1': Decimal(0),
        # 'price0': Decimal(1.0001) ** active_tick_idx,
        # 'price1': Decimal(1) / (Decimal(1.0001) ** active_tick_idx),
        'isCurrent': True
    }

    # active_tick['quantity0'], active_tick['quantity1'] = get_token_amounts(
    #     active_tick_idx['liquidityActive'],
    #     sqrtPriceX96,
    #     active_tick_idx,
    #     active_tick_idx + tickSpacing,
    #     current_tick)

    if active_tick_idx in tick_dict.keys():
        active_tick['liquidityNet'] = Decimal(tick_dict[active_tick_idx]['liquidityNet'])

    processed_ticks.append(active_tick)

    # Process ticks above current tick
    prev_tick = active_tick
    for i in range(1, TICK_RANGE + 1):
        tick_idx = active_tick_idx + (i * tick_spacing)

        new_tick = {
            'tickIdx': tick_idx,
            'liquidityActive': prev_tick['liquidityActive'],
            'liquidityNet': Decimal(0),
            'quantity0': Decimal(0),
            'quantity1': Decimal(0),
            # 'price0': Decimal(1.0001) ** tick_idx,
            # 'price1': Decimal(1) / (Decimal(1.0001) ** tick_idx),
            'isCurrent': False
        }

        # Update liquidity if tick is initialized
        if tick_idx in tick_dict.keys():
            tick_data = tick_dict[tick_idx]
            new_tick['liquidityNet'] = Decimal(tick_data['liquidityNet'])
            new_tick['liquidityActive'] = prev_tick['liquidityActive'] + Decimal(tick_data['liquidityNet'])
            # new_tick['quantity0'], new_tick['quantity1'] = get_token_amounts(
            #                                                 new_tick['liquidityActive'],
            #                                                 sqrtPriceX96,
            #                                                 tick_idx,
            #                                                 tick_idx + tickSpacing,
            #                                                 current_tick)

            processed_ticks.append(new_tick)
            prev_tick = new_tick

    # Process ticks below current tick
    prev_tick = active_tick
    for i in range(1, TICK_RANGE + 1):
        tick_idx = active_tick_idx - (i * tick_spacing)

        new_tick = {
            'tickIdx': tick_idx,
            'liquidityActive': prev_tick['liquidityActive'],
            'liquidityNet': Decimal(0),
            # 'price0': Decimal(1.0001) ** tick_idx,
            # 'price1': Decimal(1) / (Decimal(1.0001) ** tick_idx),
            'isCurrent': False
        }

        # Update liquidity if tick is initialized
        if tick_idx in tick_dict.keys():
            tick_data = tick_dict[tick_idx]
            new_tick['liquidityNet'] = Decimal(tick_data['liquidityNet'])
            new_tick['liquidityActive'] = prev_tick['liquidityActive'] - Decimal(tick_data['liquidityNet'])
            # new_tick['quantity0'], new_tick['quantity1'] = get_token_amounts(
            #                                                 new_tick['liquidityActive'],
            #                                                 sqrtPriceX96,
            #                                                 tick_idx,
            #                                                 tick_idx + tickSpacing,
            #                                                 current_tick)

            processed_ticks.append(new_tick)
            prev_tick = new_tick

    return sorted(processed_ticks, key=lambda x: x['tickIdx'])


