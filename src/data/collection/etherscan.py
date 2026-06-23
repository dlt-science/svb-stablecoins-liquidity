import asyncio

import requests
import argparse
import os
from datetime import datetime
from ratelimit import limits, sleep_and_retry
import aiohttp
import time
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

ETHERSCAN_API_URL = "https://api.etherscan.io/v2/api"
CHAIN_ID = 1  # Ethereum mainnet
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
# 5 Calls per second limit of Etherscan API
ETHERSCAN_CALLS_LIMIT = 5  # calls
ETHERSCAN_TIME_LIMIT = 1  # second


# To avoid hitting the Etherscan API rate limit,
# we use a decorator to limit the number of calls per second
@sleep_and_retry
@limits(calls=ETHERSCAN_CALLS_LIMIT, period=ETHERSCAN_TIME_LIMIT)
def get_block_by_timestamp(date_val):
    # Convert date_val string to datetime object
    # target_timestamp = datetime.strptime(date_val, "%Y-%m-%d %H:%M:%S")

    # Convert human-readable date to timestamp
    target_unix_time = int(date_val.timestamp())

    response = requests.get(ETHERSCAN_API_URL, params={
        "chainid": CHAIN_ID,
        "module": "block",
        "action": "getblocknobytime",
        "timestamp": target_unix_time,
        "closest": "before",  # Get the block before the timestamp. Change to 'after' if you need the next block.
        "apikey": ETHERSCAN_API_KEY
    })

    data = response.json()

    # Check for errors and return block number
    if data["message"] == "OK":
        return int(data["result"])

    if "Max rate limit reached" in data["result"]:
        # Wait for 1 second and try again
        print("Waiting for 1 second...")
        time.sleep(1)
        print("Trying again...")
        # Some recursion to try again
        return get_block_by_timestamp(date_val)
    else:
        raise ValueError(data["result"])


# To avoid hitting the Etherscan API rate limit,
# we use a decorator to limit the number of calls per second
@limits(calls=ETHERSCAN_CALLS_LIMIT, period=ETHERSCAN_TIME_LIMIT)
async def get_block_by_timestamp_async(date_val):
    # Convert date_val string to datetime object
    # target_timestamp = datetime.strptime(date_val, "%Y-%m-%d %H:%M:%S")

    # Convert human-readable date to timestamp
    target_unix_time = int(date_val.timestamp())

    async with aiohttp.ClientSession() as session:
        async with session.get(ETHERSCAN_API_URL, params={
            "module": "block",
            "action": "getblocknobytime",
            "timestamp": target_unix_time,
            "closest": "before",  # Get the block before the timestamp. Change to 'after' if you need the next block.
            "apikey": ETHERSCAN_API_KEY
        }) as response:
            data = await response.json()

            # Check for errors and return block number
            if data["message"] == "OK":
                return int(data["result"])
            else:
                raise ValueError(data["result"])


# To avoid hitting the Etherscan API rate limit,
# we use a decorator to limit the number of calls per second
@limits(calls=ETHERSCAN_CALLS_LIMIT, period=ETHERSCAN_TIME_LIMIT)
def get_position_token_id_for_burned_transaction(tx_hash):
    # Get transaction receipt logs
    params = {
        "module": "proxy",
        "action": "eth_getTransactionReceipt",
        "txhash": tx_hash,
        "apikey": ETHERSCAN_API_KEY
    }

    # # Convert the log index to hex
    # # The index provided by UniswapV3 is the start of the range of logs to look through
    # # We increment this by 1 to get the log index that we want to look at
    # # for DecreaseLiquidity events
    # log_index = hex(int(log_index) + 1)

    response = requests.get(ETHERSCAN_API_URL, params=params)

    if response.status_code == 200:
        data = response.json()

        if "message" in data.keys():
            # Wait for 1 second and try again
            print("Waiting for 1 second...")
            time.sleep(1)
            print("Trying again...")
            # Some recursion to try again
            return get_position_token_id_for_burned_transaction(tx_hash)

        # Ensure that the response is valid and contains logs
        if data["result"]["logs"]:
            # Look for the 'DecreaseLiquidity' event in the logs
            try:
                log = data["result"]["logs"][1]
                if log:
                    # Assuming the tokenId is an indexed parameter and is the second topic in the log
                    # Adjust if this is different for your specific contract
                    # Convert hex to int
                    token_id = int(log["topics"][1], 16)
                    return token_id
            except:
                print("It is not a burn position transaction")
                return None
    return None


# To avoid hitting the Etherscan API rate limit,
# we use a decorator to limit the number of calls per second
@limits(calls=ETHERSCAN_CALLS_LIMIT, period=ETHERSCAN_TIME_LIMIT)
def get_position_token_id_for_minted_transaction(tx_hash):
    # Get transaction receipt logs
    params = {
        "module": "proxy",
        "action": "eth_getTransactionReceipt",
        "txhash": tx_hash,
        "apikey": ETHERSCAN_API_KEY
    }

    # Convert the log index to hex
    # The index provided by UniswapV3 is the start of the range of logs to look through
    # We increment this by 1 to get the log index that we want to look at
    # for DecreaseLiquidity events

    response = requests.get(ETHERSCAN_API_URL, params=params)

    if response.status_code == 200:
        data = response.json()

        if "message" in data.keys():
            # Wait for 1 second and try again
            print("Waiting for 1 second...")
            time.sleep(1)
            print("Trying again...")
            # Some recursion to try again
            return get_position_token_id_for_minted_transaction(tx_hash)

        # Ensure that the response is valid and contains logs
        if data["result"]["logs"]:
            try:
                # Look for the 'DecreaseLiquidity' event in the logs
                log = data["result"]["logs"][-1]
                if log:
                    # Assuming the tokenId is an indexed parameter and is the second topic in the log
                    # Adjust if this is different for your specific contract
                    # Convert hex to int
                    token_id = int(log["topics"][1], 16)
                    return token_id
            except:
                print("It is not a minting position transaction")
                return None
    return None


@limits(calls=ETHERSCAN_CALLS_LIMIT, period=ETHERSCAN_TIME_LIMIT)
async def get_position_token_id_for_minted_transaction_async(tx_hash):
    async with aiohttp.ClientSession() as session:
        params = {
            "module": "proxy",
            "action": "eth_getTransactionReceipt",
            "txhash": tx_hash,
            "apikey": ETHERSCAN_API_KEY
        }
        async with session.get(ETHERSCAN_API_URL, params=params) as response:
            if response.status == 200:
                data = await response.json()

                if "message" in data and data["message"] != "OK":
                    # Wait for 1 second and try again
                    print("Waiting for 1 second...")
                    await asyncio.sleep(1)
                    print("Trying again...")
                    # Recursion with async call to try again
                    return await get_position_token_id_for_minted_transaction_async(tx_hash)

                # Ensure that the response is valid and contains logs
                if data["result"]["logs"]:

                    try:
                        # Look for the 'DecreaseLiquidity' event in the logs
                        log = data["result"]["logs"][-1]
                        if log:
                            # Assuming the tokenId is an indexed parameter and is the second topic in the log
                            # Adjust if this is different for your specific contract
                            token_id = int(log["topics"][1], 16)
                            return token_id
                    except:
                        print("It is not a minting position transaction")
                        return None
    return None


@limits(calls=ETHERSCAN_CALLS_LIMIT, period=ETHERSCAN_TIME_LIMIT)
async def get_position_token_id_for_burned_transaction_async(tx_hash):
    async with aiohttp.ClientSession() as session:
        params = {
            "module": "proxy",
            "action": "eth_getTransactionReceipt",
            "txhash": tx_hash,
            "apikey": ETHERSCAN_API_KEY
        }
        async with session.get(ETHERSCAN_API_URL, params=params) as response:
            if response.status == 200:
                data = await response.json()

                if "message" in data and data["message"] != "OK":

                    # Wait for 1 second and try again
                    print("Waiting for 1 second...")

                    await asyncio.sleep(1)
                    print("Trying again...")

                    # Recursion with async call to try again
                    return await get_position_token_id_for_burned_transaction_async(tx_hash)

                # Ensure that the response is valid and contains logs
                if data["result"]["logs"]:
                    try:
                        # The second log entry should be the 'DecreaseLiquidity' event,
                        # but this can vary based on the transaction.
                        log = data["result"]["logs"][1]  # Use the appropriate index for your case
                        if log:
                            # Assuming the tokenId is an indexed parameter and is the second topic in the log
                            token_id = int(log["topics"][1], 16)

                            if len(str(token_id)) > 8:
                                # print("It is not a burn position transaction")
                                return None

                            return token_id

                    except (KeyError, IndexError) as e:
                        print(f"Error getting burn from logs: {e}")
                        # print("It is not a burn position transaction")
                        return None
            else:
                # Handle non-200 status codes appropriately
                response.raise_for_status()

    return None


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('--start_date', type=str, default="2023-02-01")
#     parser.add_argument('--end_date', type=str, default="2023-04-30")
#
#     args = parser.parse_args()
#
#     hash_val = "0x8d9b7dce543f81daee757913beb863087ec4754af88751d19f44564c400bd917"
#     # log_index = "241"
#     #
#     nft_token_id = get_position_token_id_for_burned_transaction(hash_val)
#
#     hash_val = "0xbc6d89c0a9ee0b3f1ae7392739e88d65b24d00f8653d4608e100e6ad9ab56c4f"
#     nft_token_id = get_position_token_id_for_minted_transaction(hash_val)
#
#     # Get the block number for the start and end date
#     start_block = get_block_by_timestamp(args.start_date)
#     end_block = get_block_by_timestamp(args.end_date)
