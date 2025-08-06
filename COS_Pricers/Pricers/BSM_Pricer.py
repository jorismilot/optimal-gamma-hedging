import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import scipy.stats as st
import scipy.optimize as optimize
from tqdm import tqdm

i = 1j    # imag unit
SEC_PER_YEAR = 365 * 24 * 3600 # seconds to years

def load_0dte_data(hour='08'):
    """
    Load 0DTE option data for BTC and ETH from CSV files.
    """
    # Using your local path structure
    btc_df_path = os.path.join(f'/Users/joris/Documents/Master QF/Thesis/optimal-gamma-hedging/Data/calibration_data/{hour}', f'btc_{hour}_0dte_data.csv')
    eth_df_path = os.path.join(f'/Users/joris/Documents/Master QF/Thesis/optimal-gamma-hedging/Data/calibration_data/{hour}', f'eth_{hour}_0dte_data.csv')
    
    btc_df = pd.read_csv(btc_df_path)
    eth_df = pd.read_csv(eth_df_path)

    # Convert the timestamps to UTC 
    btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], utc=True)
    eth_df['timestamp'] = pd.to_datetime(eth_df['timestamp'], utc=True)
    return btc_df, eth_df

def extract_inputs_from_df(df):
    return (
        df['opt_type'].values,
        df['spot'].values,
        df['strike'].values,        
        df['time_to_maturity'].values, # This is already in seconds from the file
        df['mark_price'].values,
        df['mark_iv'].values / 100.0  
        )

# Black-Scholes Prices 
def bs_price(CP, S0, K, sigma, tau_sec, r):
    """Vectorised BS call/put price. tau_sec is time‑to‑expiry in seconds."""
    with np.errstate(all='ignore'):
        tau = np.asarray(tau_sec, dtype=float) / SEC_PER_YEAR # years
        CP  = np.asarray(CP, dtype=str)
        d1  = (np.log(S0 / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d2  = d1 - sigma * np.sqrt(tau)
        price = np.where(
            CP == 'call',
            st.norm.cdf(d1) * S0 - st.norm.cdf(d2) * K * np.exp(-r * tau),
            st.norm.cdf(-d2) * K * np.exp(-r * tau) - st.norm.cdf(-d1) * S0
        )
    return price

# Calibration Code 
def calibration_filter(df):
    """
    Filters a dataframe to keep only OTM options and remove duplicate strikes.
    Assumes 'moneyness' and 'date' columns already exist.
    """
    # Filter out ITM options
    conditions = ((df['opt_type'] == 'call') & (df['moneyness'] < 0.0)) | \
                 ((df['opt_type'] == 'put')  & (df['moneyness'] > 0.0))
    df_filtered = df.loc[~conditions].copy()

    # Filter out one of the double ATM options with same strike at any day
    df_filtered = df_filtered.drop_duplicates(subset=['date','strike'], keep='first')
    return df_filtered

def calibrate_bsm_snapshot(df_snap):
    """
    Calibrates a single best-fit sigma for the BSM model against the daily IV surface.
    """
    _, _, _, _, _, iv_mkt = extract_inputs_from_df(df_snap)
    loss = lambda sigma: 100 * np.sqrt(np.mean((sigma - iv_mkt)**2))
    sigma_0 = np.mean(iv_mkt)
    res = optimize.minimize(loss, sigma_0, bounds=[(0.01, 5.0)], method='L-BFGS-B')

    return {
        'model':   'bsm',
        'theta':   res.x,
        'iv_rmse': res.fun,
        'n_strk':  len(df_snap),
        'success': res.success,
        'message': str(res.message) # Ensure message is a string
    }

if __name__ == '__main__':
    df_btc, _ = load_0dte_data('08')
    df_btc    = df_btc.dropna(subset=['mark_price', 'mark_iv'])

    # Pre-calculate columns needed for the filter 
    df_btc['moneyness'] = np.log(df_btc['strike'] / df_btc['spot'])
    df_btc['date'] = df_btc['timestamp'].dt.date
    
    # Apply the filter to the entire dataframe before looping
    df_btc_filtered = calibration_filter(df_btc)

    fits, option_frames = [], []
    skip = 0

    # Group by the new 'date' column
    for date, df_day in tqdm(df_btc_filtered.groupby('date'), desc="Calibrating BSM per day"):
        if df_day['strike'].nunique() < 8:
            print(f'Skip {date}: only {df_day.strike.nunique()} strikes')
            skip += 1
            continue

        # Shared arrays for this snapshot 
        CP, S0v, K, tau_sec, _, iv_mkt = extract_inputs_from_df(df_day)
        S0 = float(S0v[0])

        fit = calibrate_bsm_snapshot(df_day)
        fits.append({'date': date, **fit})

        theta = fit['theta']

        # Model prices & IV 
        sigma_ = float(theta[0])
        price_mod = bs_price(CP, S0, K, sigma_, tau_sec, 0.0)
        iv_mod    = np.full_like(iv_mkt, sigma_)

        enriched = df_day.copy()
        enriched['model']        = 'BSM'
        enriched['price_model']  = price_mod
        enriched['iv_model']     = iv_mod
        enriched['iv_abs_err']   = np.abs(iv_mod - iv_mkt)

    if fits: # Ensure there is something to save
        df_fits = pd.DataFrame(fits)

        # Define the output path
        output_path = '/Users/joris/Documents/Master QF/Thesis/optimal-gamma-hedging/COS_Pricers/Data/'
        
        # Create the directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Save the DataFrames to CSV files
        df_fits.to_csv(os.path.join(output_path, 'bsm_calibration_results_final.csv'), index=False)

        print(f"\n--- BSM Calibration Finished ---")
        print(f"Fit results saved to {os.path.join(output_path, 'bsm_calibration_results_final.csv')}")
        print("\n--- Fit Results (first rows) ---")
        print(df_fits.head())
    else:
        print("No data was calibrated. Output files not created.")