import os
import argparse
import random
import time

from tqdm import tqdm
import pandas as pd

from GenSummaryMCI import filter_csv_files_by_date_range
import asyncio
from tqdm.asyncio import tqdm_asyncio

# Add the collection folder to the path
import sys

sys.path.append('./../')

from collection.uniswapv3 import get_pool_tvl_block
from collection.uniswapv3_async import get_pool_data_at_unix_timestamp
from collection.etherscan import get_block_by_timestamp
from multiprocessing import Pool, cpu_count


# def process_csv(args):
#     csv_file_date, csv_file_path = args
#
#     # Columns that we want to keep
#     cols = ["totalValueLockedToken1",
#             "totalValueLockedToken0",
#             "24hr Volume (USD)",
#             "token1Price",
#             "token0Price",
#             "token0_symbol",
#             "token1_symbol",
#             "fee_tier",
#             "pool_address",
#             "sqrtPrice"]
#
#     # Read csv files into pandas dataframe using
#     print(f"Processing {csv_file_path}...")
#     df = pd.read_csv(csv_file_path)
#
#     # Sum all the liquidity in the pool
#     liquidity = df.groupby(['pool_address']).liquidity.sum().reset_index()
#
#     # Get the block number at timestamp
#     print(f"Getting block number for {csv_file_date}...")
#
#     # Add a random wait to avoid getting banned because of too many requests
#     # Wait for between 1 to 2 seconds
#     wait_value = random.randint(1, 2)
#     print(f"Waiting for {wait_value} second(s) for fetching the block number...")
#     time.sleep(wait_value)
#     block_number = get_block_by_timestamp(csv_file_date)
#
#     # Get the TVL for each pool
#     for index, row in tqdm(liquidity.iterrows(), total=len(liquidity)):
#
#         # Only try to get the TVL if the column "totalValueLockedUSD" is empty
#         if "totalValueLockedUSD" not in row:
#             liquidity.loc[index, 'totalValueLockedUSD'] = get_pool_tvl_block(pool_id=row['pool_address'],
#                                                                              block_number=block_number)
#         elif pd.isnull(row['totalValueLockedUSD']):
#
#             print(f"Getting TVL for pool {row['pool_address']} at block {block_number}...")
#             liquidity.loc[index, 'totalValueLockedUSD'] = get_pool_tvl_block(pool_id=row['pool_address'],
#                                                                              block_number=block_number)
#
#     # Keep only the columns that we want
#     otherStats = df[cols].drop_duplicates()
#
#     # Merge the two dataframes
#     final_df = pd.merge(otherStats, liquidity, on=['pool_address'], how='inner')
#
#     # Add the date to the dataframe
#     final_df['date'] = csv_file_date
#     final_df['blockNumber'] = block_number
#
#     return final_df

async def bounded_fetch(semaphore, task):
    async with semaphore:
        return await task


async def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--enriched_csvs_files', type=str,
    #                     help='Path to the .csv file',
    #                     default="./../../../data/enriched")

    parser.add_argument('--raw_csvs_files', type=str,
                        help='Path to the .csv file',
                        default="./../../../data/raw")

    parser.add_argument('--filtered_csvs_files', type=str,
                        help='Path to the .csv file',
                        default="./../../../data/filtered")

    parser.add_argument('--pools_csvs_files', type=str,
                        help='Path to the .csv file',
                        default="./../../../data/filtered/pools")

    parser.add_argument('--start_date', type=str, default="2023-02-01")
    parser.add_argument('--end_date', type=str, default="2023-04-30")

    args = parser.parse_args()

    pools_csv_files = filter_csv_files_by_date_range(csv_files_path=args.pools_csvs_files,
                                                     start_date=args.start_date,
                                                     end_date=args.end_date)

    pool_csv_file = pd.read_csv(list(pools_csv_files.values())[0])
    pool_ids = pool_csv_file['pool_address'].unique()

    # Filter out the pools not needed
    pools_to_ignore = [
        # "0x6c6bc977e13df9b0de53b251522280bb72383700",  # DAIUSDC500
        # "0x6f48eca74b38d2936b02ab603ff4e36a6c0e3a77",  # DAIUSDT500
        #    "0x9a772018fbd77fcd2d25657e5c547baff3fd7d16",  # WBTCUSDC500
        #    "0x9db9e0e53058c89e5b94e29621a205198648425b",  # WBTCUSDT300
        "0xc7bbec68d12a0d1830360f8ec58fa599ba1b0e9b",  # ETHUSDT100
        # "0x3416cf6c708da44db2624d63ea0aaef7113527c6",  # USDCUSDT100
        # "0x7bea39867e4169dbe237d55c8242a8f2fcdcc387",  # USDCETH1000
        #    "0xc5af84701f98fa483ece78af83f11b6c38aca71d"  # USDTETH1000
        "0x5777d92f208679db4b9778590fa3cab3ac9e2168",  # DAIUSDC100
        "0x48da0965ab2d2cbf1c17c09cfb5cbe67ad5b1406"  # DAIUSDT100
    ]
    pool_ids = [pool_id for pool_id in pool_ids if pool_id not in pools_to_ignore]

    date_range = pd.date_range(start=args.start_date, end=args.end_date, freq='D').to_list()

    # Convert to unix timestamp with 00:00:00 time
    date_range = [int(pd.Timestamp(date).timestamp()) for date in date_range]

    total_pools = []
    semaphore_pools = asyncio.Semaphore(len(pool_ids))

    for unix_timestamp in tqdm(date_range):
        tasks = [get_pool_data_at_unix_timestamp(pool_id=pool_id, start_time=unix_timestamp) for
                 pool_id in pool_ids]

        pools = await asyncio.gather(*(asyncio.create_task(bounded_fetch(semaphore_pools, task))
                                       for task in tqdm_asyncio(tasks)))

        pools = [pool for pool in pools if pool is not None]  # Filter out None results

        # Concat all the pools
        pools = pd.concat(pools)

        # Convert date to human readable format for day
        pools['date'] = pd.to_datetime(unix_timestamp, unit='s').strftime('%Y-%m-%d')

        total_pools.append(pools)

        # Add a wait of 1 second to avoid getting banned
        await asyncio.sleep(1)

    # # get all the csv files to process
    # csv_files = filter_csv_files_by_date_range(csv_files_path=args.enriched_csvs_files,
    #                                            start_date=args.start_date,
    #                                            end_date=args.end_date)
    #
    # num_cpus = cpu_count() // 2  # Use half of the available CPUs
    # print(f"Using {num_cpus} CPUs...")
    #
    # with Pool(processes=num_cpus) as p:
    #     # To add a progress bar
    #     pool_stats = list(tqdm(p.imap(process_csv, csv_files.items()), total=len(csv_files)))

    # pool_stats_df = pd.concat(pool_stats)

    # Mark which is the stablecoin in the pool by joining with the top_five_pools.csv
    # liquidity_pools_selected = pd.read_csv(os.path.join(args.raw_csvs_files, 'liquidity_pools_selected.csv'))

    pool_stats_df = pd.concat(total_pools)

    # Find the stablecoin in the pool pair
    pool_stats_df['stablecoin'] = pool_stats_df['pair'].apply(lambda x: "USDC" if "USDC" in x else "USDT")

    # liquidity_pools_selected = liquidity_pools_selected[['Pool ID', 'Stablecoin', 'Pair', 'Fee tier (%)']]
    # pool_stats_df = pool_stats_df.rename(columns={'Pool ID': 'pool_address',
    #                                     'Pair': 'pair',
    #                                     'Fee tier (%)': 'fee_tier_percentage',
    #                                     'Stablecoin': 'stablecoin'})

    # pool_stats_df = pd.merge(pool_stats_df, liquidity_pools_selected, on=['pool_address'], how='left')

    print(f"Saving the Pool stats...")
    pool_stats_df.to_csv(os.path.join(args.filtered_csvs_files, 'pool_stats.csv'), index=False)


if __name__ == "__main__":

    # Resubmit the job if it fails
    for i in range(100):
        print(f"Attempt {i + 1}...")
        try:
            asyncio.run(main())
            break
        except Exception as e:
            print(f"Error from main logic: {e}")
            # Wait for 1 second before trying again
            time.sleep(1)
            print("Failed to run the job. Trying again...")
            continue
