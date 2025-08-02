import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import scipy.stats as st
import scipy.optimize as optimize
from tqdm import tqdm  
# We will remove joblib as we are switching to a sequential approach

i = 1j    # imag unit

def load_0dte_data(file_path='/Users/joris/Documents/Master QF/Thesis/optimal-gamma-hedging/Data/calibration_data/08/btc_08_0dte_data.csv'):
    """
    Load 0DTE option data for BTC from a specified CSV file.
    """
    # Use the provided filename directly
    btc_df = pd.read_csv(file_path)

    btc_df['time_to_maturity'] = btc_df['time_to_maturity'] / (365 * 24 * 3600)
    btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], utc=True)
    return btc_df

def filter_otm_calibration(df):
    df['moneyness'] = np.log(df['strike'] / df['spot'])
    mask =  ((df['opt_type'] == 'call') & (df['moneyness'] >= 0)) | \
            ((df['opt_type'] == 'put')  & (df['moneyness'] <= 0))
    df = df[mask]
    df = df.drop_duplicates(subset=['timestamp','strike'], keep='first')
    return df

def extract_inputs_from_df(df):
    return (
        df['opt_type'].values,
        df['spot'].values,
        df['strike'].values,        
        df['time_to_maturity'].values,
        df['mark_price'].values,
        df['mark_iv'].values / 100.0  
        )

# Black-Scholes formulas (no changes needed)
def bs_price(CP, S0, K, sigma, tau, r):
    tau = np.asarray(tau, dtype=float); CP  = np.asarray(CP, dtype=str)
    d1  = (np.log(S0 / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2  = d1 - sigma * np.sqrt(tau)
    price = np.where(CP == 'call', st.norm.cdf(d1) * S0 - st.norm.cdf(d2) * K * np.exp(-r * tau), st.norm.cdf(-d2) * K * np.exp(-r * tau) - st.norm.cdf(-d1) * S0)
    return price

def bs_vega(S0, K, sigma, tau, r):
    S0, K, sigma, tau, r = [np.asarray(x) for x in [S0, K, sigma, tau, r]]
    d1  = (np.log(S0 / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return S0 * st.norm.pdf(d1) * np.sqrt(tau)

# --- COS Pricing Mechanism ---

def chf_merton(u, r, tau, theta):
    sigma, xi, muJ, sigmaJ = theta
    omega_bar = xi * (np.exp(muJ + 0.5*sigmaJ**2) - 1)
    mu = r - 0.5*sigma**2 - omega_bar
    return np.exp(i*u*mu*tau -0.5*sigma**2*u**2*tau +
                  xi*tau*(np.exp(i*muJ*u - 0.5*sigmaJ**2*u**2) - 1))

def merton_cumulants(tau, r, theta):
    sigma, xi, muJ, sigmaJ = theta
    omega_bar = xi * (np.exp(muJ + 0.5 * sigmaJ**2) - 1)
    c1 = tau * (r - omega_bar - 0.5 * sigma**2 + xi * muJ)
    c2 = tau * (sigma**2 + xi * (muJ**2 + sigmaJ**2))
    c4 = tau * xi * (muJ**4 + 6 * muJ**2 * sigmaJ**2 + 3 * sigmaJ**4)
    return c1, c2, c4

def chi_psi_vec(k, a, b, c, d):
    omega = k * np.pi / (b - a)
    xi, psi = np.zeros_like(omega), np.zeros_like(omega)
    xi[0, :]  = np.exp(d) - np.exp(c)
    psi[0, :] = d - c
    if k.shape[0] > 1:
        omega_nz, k_nz = omega[1:, :], k[1:, :]
        denom = 1.0 + omega_nz**2
        xi[1:, :] = ( np.cos(omega_nz*(d-a))*np.exp(d) + omega_nz*np.sin(omega_nz*(d-a))*np.exp(d) - np.cos(omega_nz*(c-a))*np.exp(c) - omega_nz*np.sin(omega_nz*(c-a))*np.exp(c) ) / denom
        psi[1:, :] = ((b - a) / (k_nz * np.pi)) * (np.sin(omega_nz*(d - a)) - np.sin(omega_nz*(c - a)))
    return xi, psi

def payoff_coefficients_vec(CP, k, a, b):
    CP, is_call = np.asarray(CP, dtype=str), (CP == 'call')
    c = np.where(is_call, 0.0, a)
    d = np.where(is_call, b,   0.0)
    s = np.where(is_call, 1.0, -1.0)
    xi, psi = chi_psi_vec(k, a, b, c, d)
    H_k = (2.0 / (b - a)) * s * (xi - psi)
    return H_k

def cos_pricer(CP, S0, K, tau, r, theta, N=512, L=10):
    K, tau, CP = np.asarray(K), np.asarray(tau), np.asarray(CP)
    c1, c2, c4 = merton_cumulants(tau, r, theta)
    log_ratio  = np.log(S0 / K)
    a = (log_ratio + c1 - L*np.sqrt(c2 + np.sqrt(c4)))[None, :]
    b = (log_ratio + c1 + L*np.sqrt(c2 + np.sqrt(c4)))[None, :]
    k = np.arange(N, dtype=float)[:, None] # (N,1)
    u = k * np.pi / (b - a)
    phi = chf_merton(u, r, tau, theta)
    
    # Pass k directly, not k[None,:]. Helper expects a column vector.
    H_k = payoff_coefficients_vec(CP[None, :], k, a, b)
    
    w = np.real(phi * np.exp(-1j * u * (a - log_ratio[None, :])))
    w[0, :] *= 0.5
    prices = np.exp(-r * tau) * K * np.sum(w * H_k, axis=0)
    return prices.astype(float)

# --- Calibration Mechanism ---
MODEL_CFG = {
    'merton': (
        # Initial Guess [sigma, xi, muJ, sigmaJ]
        np.array([0.80, 0.20, -0.10, 0.30]),
        # Lower Bounds
        np.array([0.10, 0.00, -2.00, 0.01]),
        # Upper Bounds
        np.array([2.20, 2.00,  2.00, 2.00])
    )
}

# Here used: Enhanced numerical stability
def iv_newton(price, CP, S0, K, tau, r, sigma_init=0.5, tol=1e-10, it=40):
    sigma = np.full_like(price, sigma_init, dtype=float)
    for _ in range(it):
        diff = bs_price(CP, S0, K, sigma, tau, r) - price
        vega = bs_vega(S0, K, sigma, tau, r)
        sigma -= diff / np.where(vega > 1e-10, vega, np.inf) # Lower vega threshold
        if np.all(np.abs(diff) < tol): break
    sigma = np.where((sigma > 0) & (sigma < 5), sigma, np.nan)
    return sigma

def rmse_iv(S0, K, tau, CP, iv_mkt, theta, r=0.0):
    pr_model = cos_pricer(CP, S0, K, tau, r, theta)
    iv_model = iv_newton(pr_model, CP, S0, K, tau, r, sigma_init=np.nan_to_num(iv_mkt, nan=0.5))
    # Return a large penalty if inversion fails completely
    if np.all(np.isnan(iv_model)): return 1e6
    return 100 * np.sqrt(np.nanmean((iv_model - iv_mkt)**2))

def calibrate_merton_snapshot(df_snap, theta_0, bounds, r=0.0):
    CP, S0_vec, K, tau, _, iv_mkt = extract_inputs_from_df(df_snap)
    S0 = float(S0_vec[0])
    
    loss = lambda th: rmse_iv(S0, K, tau, CP, iv_mkt, th, r)
    
    # More robust optimizer settings
    res  = optimize.minimize(loss, theta_0,
                             bounds=bounds,
                             method='L-BFGS-B',
                             options={'ftol': 1e-8, 'maxiter': 500})
    return {
        'date'      : pd.to_datetime(df_snap['timestamp'].iloc[0]).date(),
        'model'     : 'merton',
        'theta_σ'   : res.x[0], 'theta_ξ': res.x[1], 'theta_μJ': res.x[2], 'theta_σJ': res.x[3],
        'iv_rmse'   : res.fun,
        'n_strk'    : len(df_snap),
        'success'   : res.success,
        'message'   : res.message
    }

if __name__ == '__main__':
    btc_raw = load_0dte_data() # Reads 'btc_08_0dte_data.csv' by default         
    btc = filter_otm_calibration(btc_raw)
    
    # Sort groups by date to ensure sequential processing
    grouped = sorted(list(btc.groupby(btc['timestamp'].dt.date)))

    # --- CORRECTED: Sequential Calibration with Robust "Warm Starts" ---
    calib_out = []
    
    # Get the initial guess and bounds from config
    # This `theta_0` will be our starting point and fallback
    theta_0, lb, ub = MODEL_CFG['merton']
    bounds = list(zip(lb, ub))

    print("Starting sequential calibration with robust warm starts...")
    for date, snap in tqdm(grouped, desc='Calibrating per-day Merton', unit='day'):
        # `theta_0` is the initial guess for this iteration.
        # It's either the default or the result from the last successful run.
        result = calibrate_merton_snapshot(snap, theta_0, bounds)
        
        # Append the results to our list
        calib_out.append(result)
        
        # IMPORTANT: Update the initial guess for the *next* day ONLY if this run succeeded.
        # If it failed, we'll just reuse the same `theta_0` for the next day.
        if result['success']:
            theta_0 = np.array([
                result['theta_σ'], 
                result['theta_ξ'], 
                result['theta_μJ'], 
                result['theta_σJ']
            ])

    df_calib = pd.DataFrame(calib_out)
    
    df_calib.to_csv('merton_calibration_results_final.csv', index=False)
    print("\n--- Final Calibration Finished ---")
    print("Results saved to 'merton_calibration_results_final.csv'")
    print(df_calib.head())