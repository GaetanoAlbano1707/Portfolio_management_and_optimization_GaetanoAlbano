import pandas as pd
import os
from glob import glob

def combine_ticker_files(data_dir="./Tickers_file", save_path="./TEST/main_data_full.csv"):
    all_data = []
    for file in glob(f"{data_dir}/*.csv"):
        ticker = os.path.basename(file).split('.')[0]
        df = pd.read_csv(file)
        df.columns = [col.lower() for col in df.columns]  # Converti tutto in lowercase
        df["tic"] = ticker
        all_data.append(df)

    combined_df = pd.concat(all_data)
    combined_df["date"] = pd.to_datetime(combined_df["date"])
    combined_df = combined_df.sort_values(["tic", "date"]).reset_index(drop=True)
    combined_df.to_csv(save_path, index=False)
    print(f"✅ File combinato salvato in: {save_path}")
    return combined_df

# Esecuzione diretta se runnato come script
if __name__ == "__main__":
    combine_ticker_files()
