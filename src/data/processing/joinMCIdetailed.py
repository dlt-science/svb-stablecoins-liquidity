import argparse
import os
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import numpy as np

# Adding the path to be able to import the analytics module
import sys

sys.path.append('./../../')
from GenSummaryMCI_levels import filter_csv_files_by_date_range


def gen_mci_detailed_summarized(csv_files, dir_file, filename="mci_detailed_summarized.parquet"):
    csv_by_date = {}
    for date, csv_file in csv_files.items():

        day_date = datetime.strftime(date, '%Y-%m-%d')

        if day_date not in csv_by_date.keys():
            csv_by_date[day_date] = []

        csv_by_date[day_date].append({date: csv_file})
    mci_detailed = []

    for day_date, csv_files in tqdm(csv_by_date.items(), total=len(csv_by_date.items())):
        dfs = []
        for csv_file in csv_files:
            for date, file in csv_file.items():
                df = pd.read_parquet(file, engine="fastparquet")
                df['date'] = pd.to_datetime(date)
                dfs.append(df)

        df = pd.concat(dfs)

        # filtered_df = df[df["MCI_1"].isin([np.nan, -np.inf])]
        # if not filtered_df.empty:
        #     print(f"Error")

        month = day_date.split("-")[1]
        file_dir = os.path.join(dir_file, month)
        os.makedirs(file_dir, exist_ok=True)
        file_path = os.path.join(file_dir, f"{day_date}.csv")
        df.to_csv(file_path, index=False)

        mci_detailed.append(df)

    mci_detailed = pd.concat(mci_detailed)
    file_path = os.path.join(args.filtered_path, filename)
    mci_detailed.to_parquet(file_path, index=False, engine="fastparquet")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mci_detailed_path', type=str,
                        help='Path to the .csv file',
                        default="./../../../data/filtered/mci_detailed_ticks")


    parser.add_argument('--mci_detailed_size_path', type=str,
                        help='Path to the .csv file',
                        default="./../../../data/filtered/mci_detailed_size")

    parser.add_argument('--filtered_path', type=str,
                        help='Path to the .csv file',
                        default="./../../../data/filtered")

    parser.add_argument('--file_extension', type=str,
                        help='Extension of the files to be read',
                        default=".parquet")

    parser.add_argument('--start_date', type=str, default="2023-02-28")
    parser.add_argument('--end_date', type=str, default="2023-04-01")

    args = parser.parse_args()

    # Format date strings to be '%Y-%m-%d %H-%M-%S'
    start_date = pd.to_datetime(args.start_date).strftime('%Y-%m-%d %H-%M-%S')
    end_date = pd.to_datetime(args.end_date).strftime('%Y-%m-%d %H-%M-%S')

    csv_files = filter_csv_files_by_date_range(csv_files_path=args.mci_detailed_path,
                                               start_date=start_date,
                                               end_date=end_date,
                                               file_ext=args.file_extension)

    csv_files_size = filter_csv_files_by_date_range(csv_files_path=args.mci_detailed_size_path,
                                                    start_date=start_date,
                                                    end_date=end_date,
                                                    file_ext=args.file_extension)

    gen_mci_detailed_summarized(csv_files, dir_file=args.mci_detailed_path, filename="mci_detailed_summarized.parquet")
    # gen_mci_detailed_summarized(csv_files_size, dir_file=args.mci_detailed_size_path, filename="mci_detailed_size_summarized.parquet")