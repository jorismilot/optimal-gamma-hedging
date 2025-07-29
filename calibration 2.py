import pandas as pd
import os
import numpy as np

class Pricing:
    def __init__(self):
        self.data_dir = '/Users/joris/Documents/Master QF/Thesis/optimal-gamma-hedging/Data'
        self.coins = ['btc', 'eth']
        self.dte = 0

    def load_0dte_data(self):
        for coin in self.coins:
            # Loading data 
            file_path = os.path.join(self.data_dir,'snapshot_data', f'{coin}_full_data_transformed.csv')
            df = pd.read_csv(file_path)

            # Only keep certain colums 
            cols_base = ['timestamp', 'symbol']
            cols_rest = ['spot', 'bid_price', 'ask_price',
                         'mark_price', 'bid_iv', 'ask_iv', 'mark_iv',
                         'delta', 'gamma', 'vega', 'theta', 'rho',
                         'open_interest', 'volume']

            # Parse timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)

            # Extract from symbol:
            df[['coin', 'raw_expiry', 'strike', 'raw_opt_code']] = df['symbol'].str.split('-', expand=True).iloc[:, :4]
            df['opt_type'] = df['raw_opt_code'].map({'C': 'call', 'P': 'put'}) # Map option type

            # Compute expiration and time to maturity
            expiry_dt = pd.to_datetime(df['raw_expiry'], format='%d%b%y', errors='coerce')
            df['expiration'] = (expiry_dt + pd.Timedelta(hours=8)).dt.tz_localize('UTC')
            df['time_to_maturity'] = (df['expiration'] - df['timestamp']).dt.total_seconds()
            df['DTE'] = np.floor(df['time_to_maturity'] / 86_400).astype('Int64')

            # Drop intermediate columns
            df = df.drop(columns=['raw_expiry', 'raw_opt_code'])

            # Reorder and filter 
            df = df[cols_base + ['coin', 'expiration', 'time_to_maturity', 'DTE', 'strike', 'opt_type'] + cols_rest]
            df = df[df['DTE'] == 0]

            # Save the filtered DataFrame
            out_dir = os.path.join(self.data_dir, 'calibration_data')
            os.makedirs(out_dir, exist_ok=True)
            df.to_csv(os.path.join(out_dir, f'{coin}_0dte_data.csv'), index=False)

if __name__ == "__main__":
    Pricing().load_0dte_data()
    print("0 DTE data loaded and processed successfully.")

    # Get df where timestamp equals expiration
    for coin in ['btc', 'eth']:
        df = pd.read_csv(os.path.join('/Users/joris/Documents/Master QF/Thesis/optimal-gamma-hedging/Data/calibration_data', f'{coin}_0dte_data.csv'))
        df_exp = df[df['timestamp'] == df['expiration']]
        print(f"Data for {coin} at expiration:\n", df_exp.head())
