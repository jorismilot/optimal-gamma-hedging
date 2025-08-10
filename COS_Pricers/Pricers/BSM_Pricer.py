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

def filter_otm_calibration(df):
    """Filters a dataframe to keep only ATM/OTM options and remove duplicate strikes."""
    # Ensure 'moneyness' is calculated
    if 'moneyness' not in df.columns:
        df['moneyness'] = np.log(df['strike'] / df['spot'])

    mask = ((df['opt_type'] == 'call') & (df['moneyness'] >= 0)) | \
           ((df['opt_type'] == 'put') & (df['moneyness'] <= 0))
    df_filtered = df[mask].copy()

    # Drop duplicates for any given strike on the same day
    df_filtered.drop_duplicates(subset=['timestamp', 'strike'], keep='first', inplace=True)
    return df_filtered

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
        'theta_sigma':   res.x,
        'iv_rmse': res.fun,
        'n_strk':  len(df_snap),
        'success': res.success,
        'message': str(res.message) # Ensure message is a string
    }

if __name__ == '__main__':
    btc_raw = load_0dte_data()[0]
    btc = filter_otm_calibration(btc_raw.dropna(subset=['mark_iv', 'mark_price']))
    grouped = sorted(list(btc.groupby(btc['timestamp'].dt.date)))

    # Initialize Result Lists
    calib_summary = []
    option_fits = []

    print("Starting BSM calibration...")
    for date, snap in tqdm(grouped, desc='Calibrating BSM per-day'):
        # Calibrate Snapshot and Store Summary 
        result = calibrate_bsm_snapshot(snap)
        # Add the date to the summary dictionary before appending
        result_with_date = {'date': date, **result}
        calib_summary.append(result_with_date)

        # Calculate Detailed Fits if Successful
        if result['success']:
            sigma_opt = result['theta_sigma']

            # Extract inputs needed for pricing
            CP, S0_vec, K, tau_sec, _, iv_mkt = extract_inputs_from_df(snap)
            S0 = S0_vec[0]

            # Calculate fitted prices and IVs
            fitted_price = bs_price(CP, S0, K, sigma_opt, tau_sec, r=0.0)
            fitted_iv = np.full_like(iv_mkt, sigma_opt)

            # Add results to a copy of the day's snapshot
            detailed_snap = snap.copy()
            detailed_snap['fitted_price'] = fitted_price
            detailed_snap['fitted_iv'] = fitted_iv
            detailed_snap['SE_fitted'] = (fitted_iv - iv_mkt)**2
            option_fits.append(detailed_snap)

    # Save Results to CSV 
    output_path = '/Users/joris/Documents/Master QF/Thesis/optimal-gamma-hedging/COS_Pricers/Data/'
    os.makedirs(os.path.join(output_path, 'Calibration'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'Options'), exist_ok=True)

    # Save summary results
    if calib_summary:
        df_calib_summary = pd.DataFrame(calib_summary)
        summary_filepath = os.path.join(output_path, 'Calibration', 'bsm_calibration_summary.csv')
        df_calib_summary.to_csv(summary_filepath, index=False)
        print("\n--- Calibration Summary Finished ---")
        print(f"Summary results saved to '{summary_filepath}'")
        print(df_calib_summary.head())
    else:
        print("\nNo summary results to save.")

    # Save detailed per-option fits
    if option_fits:
        df_detailed_fits = pd.concat(option_fits, ignore_index=True)
        detailed_filepath = os.path.join(output_path, 'Options', 'bsm_per_option_fits.csv')
        df_detailed_fits.to_csv(detailed_filepath, index=False)
        print("\n--- Detailed Fits Finished ---")
        print(f"Per-option results saved to '{detailed_filepath}'")
        print(df_detailed_fits.head())
    else:
        print("\nNo successful calibrations to generate detailed fits.")

