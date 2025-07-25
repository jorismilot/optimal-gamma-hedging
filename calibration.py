import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

class Pricing:
    def __init__(self, coins, dte):
        self.data_dir = '/Users/joris/Documents/Master QF/Thesis/optima-gamma-hedging/Data/snapshot_data'
        self.coins = ['BTC', 'ETH']
        self.dte = 0

    def load_0dte_data(self):
        for coin in self.coins:
            # ── loading data ──
            file_path = os.path.join(self.data_dir, f'{coin}_full_data_transformed.csv')
            df = pd.read_csv(file_path)

            # ── timestamps & filter ──
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)

            df[['coin', 'exp_raw', 'strike_raw', 'cp']] = df['symbol'].str.split('-', expand=True)
            expiry_dt = pd.to_datetime(df['exp_raw'], format='%d%b%y', errors='coerce')
            df['expiry'] = expiry_dt + pd.Timedelta(hours=8)   # expiry at 08:00 UTC
            df['expiry'] = df['expiry'].dt.tz_localize('UTC')  # force to UTC

            secs = (df['expiry'] - df['timestamp']).dt.total_seconds()
            df['DTE'] = np.floor(secs / 86_400).astype('Int64')

            

    