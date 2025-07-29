import pandas as pd
import os
import numpy as np

class DataLoader:
    def __init__(self):
        self.data_dir = '/Users/joris/Documents/Master QF/Thesis/optimal-gamma-hedging/Data'
        self.coins = ['btc', 'eth']
        self.dte = 0

    # DATA LOADING AND STORAGE METHODS
    def load_0dte_data(self):
        """        Load and process 0 DTE data for specified coins.         """
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
            df[['coin', 'expiry_raw', 'strike', 'raw_opt_code']] = df['symbol'].str.split('-', expand=True).iloc[:, :4]
            df['opt_type'] = df['raw_opt_code'].map({'C': 'call', 'P': 'put'}) # Map option type

            # Compute expiration and time to maturity
            df['expiry'] = (pd.to_datetime(df['expiry_raw'], format='%d%b%y', errors='coerce') + pd.Timedelta(hours=8)).dt.tz_localize('UTC')
            df['time_to_maturity'] = (df['expiry'] - df['timestamp']).dt.total_seconds()
            df = df[df['time_to_maturity'] > 0] # drop expired rows
            df['DTE'] = (np.ceil(df['time_to_maturity'] / 86_400) - 1).astype('Int64')      

            # Drop intermediate columns
            df = df.drop(columns=['expiry_raw', 'raw_opt_code'])

            # Reorder and filter 
            df = df[cols_base + ['coin', 'expiry', 'time_to_maturity', 'DTE', 'strike', 'opt_type'] + cols_rest]
            df = df[df['DTE'] == 0]

            # Save the filtered DataFrame
            out_dir = os.path.join(self.data_dir, 'calibration_data/all')
            os.makedirs(out_dir, exist_ok=True)
            df.to_csv(os.path.join(out_dir, f'{coin}_0dte_data.csv'), index=False)
    
    def store_time_specific_data(self, df, coin):
        """  Store time-specific data for a given coin.  """
        coin = coin.lower()

        # parse timestamp and extract hour
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['hour']      = df['timestamp'].dt.hour

        # Make hourly data sets
        for hour in range(24):
            # slice just this hour
            df_hour = df[df['hour'] == hour].copy()
            df_hour.drop(columns=['hour'], inplace=True)

            # prepare output directory & filename
            out_dir = os.path.join(self.data_dir, 'calibration_data', f'{hour:02d}')
            os.makedirs(out_dir, exist_ok=True)

            filename = f'{coin}_{hour:02d}_0dte_data.csv'
            out_path = os.path.join(out_dir, filename)

            # write CSV without index
            df_hour.to_csv(out_path, index=False)

class CharacteristicFunction:
    # CALIBRATION METHODS
    def charact_func_merton(r,tau,muJ,sigmaJ,sigma, xi, S0):
        i   = np.complex(0.0,1.0)
        # Term for E(exp(J)-1)
        helpExp = np.exp(muJ + 0.5 * sigmaJ * sigmaJ) - 1.0
        
        # Characteristic function for Merton's model    
        cf = lambda u: np.exp(i*u*np.log(S0)) * np.exp(i * u * (r - xi * helpExp - 0.5 * sigma * sigma) *tau \
            - 0.5 * sigma * sigma * u * u * tau + xi * tau * \
            (np.exp(i * u * muJ - 0.5 * sigmaJ * sigmaJ * u * u)-1.0))
        return cf 


if __name__ == "__main__":
    # Init
    dataloader = DataLoader()




