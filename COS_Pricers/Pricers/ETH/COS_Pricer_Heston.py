import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import scipy.stats as st
import scipy.optimize as optimize
from tqdm import tqdm  

i = 1j    # imag unit

def load_0dte_data(file_path='Data/calibration_data/08/eth_08_0dte_data.csv'):
    eth_df = pd.read_csv(file_path)
    eth_df['time_to_maturity'] = eth_df['time_to_maturity'] / (365 * 24 * 3600)
    eth_df['timestamp'] = pd.to_datetime(eth_df['timestamp'], utc=True)
    return eth_df

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

# Black-Scholes Formulas
def bs_price(CP, S0, K, sigma, tau, r):
    tau = np.asarray(tau, dtype=float); CP  = np.asarray(CP, dtype=str)
    with np.errstate(all='ignore'):
        d1  = (np.log(S0 / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d2  = d1 - sigma * np.sqrt(tau)
        price = np.where(CP == 'call', st.norm.cdf(d1) * S0 - st.norm.cdf(d2) * K * np.exp(-r * tau), st.norm.cdf(-d2) * K * np.exp(-r * tau) - st.norm.cdf(-d1) * S0)
    return price

def bs_vega(S0, K, sigma, tau, r):
    S0, K, sigma, tau, r = [np.asarray(x) for x in [S0, K, sigma, tau, r]]
    with np.errstate(all='ignore'):
        d1  = (np.log(S0 / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return S0 * st.norm.pdf(d1) * np.sqrt(tau)

# COS Pricing Mechanism for Heston 
def chf_heston(u, tau, r, theta):
    v0, v_bar, kappa, gamma, rho = theta
    kappa = 0.5 # Fix kappa
    d = np.sqrt(np.power(kappa-gamma*rho*i*u,2)+(u**2+i*u)*gamma**2)
    g  = (kappa-gamma*rho*i*u-d)/(kappa-gamma*rho*i*u+d)
    C  = (1.0-np.exp(-d*tau))/(gamma*gamma*(1.0-g*np.exp(-d*tau)))\
        *(kappa-gamma*rho*i*u-d)

    # Note that we exclude the term -r*tau, as the discounting is performed in the COS method
    A  = r * i*u *tau + kappa*v_bar*tau/gamma/gamma *(kappa-gamma*rho*i*u-d)\
        - 2*kappa*v_bar/gamma/gamma * (np.log1p(-g*np.exp(-d*tau)) - np.log1p(-g))

    return np.exp(A + C*v0) 

# Payoff coefficient functions 
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

# Heston COS pricer
def cos_pricer(CP, S0, K, tau, r, theta, N=256, L=12):
    K, tau, CP = np.asarray(K), np.asarray(tau), np.asarray(CP)
    log_ratio  = np.log(S0 / K)
    
    a = (log_ratio - L*np.sqrt(tau))[None, :] # TO DO: SHOULD THIS BE 0 OR log_ratio?
    b = (log_ratio + L*np.sqrt(tau))[None, :] # TO DO: SHOULD THIS BE 0 OR log_ratio?
    
    k = np.arange(N, dtype=float)[:, None]
    u = k * np.pi / (b - a)
    
    phi = chf_heston(u, tau, r, theta)

    H_k = payoff_coefficients_vec(CP[None, :], k, a, b)
    w = np.real(phi * np.exp(-1j * u * (a - log_ratio)))
    w[0, :] *= 0.5
    
    prices = np.exp(-r * tau) * K * np.sum(w * H_k, axis=0)
    return prices.astype(float)


# Calibration Mechanism 
MODEL_CFG = {
    'heston': (
        # Parameters: [v0, v_bar, kappa, gamma, rho] # Skip kappa as it has the same effect as gamma
        np.array([2.00, 2.00, 0.3, 0.5, -0.25]),  # initial guess
        np.array([0.001, 0.01, 0.01, 3, -0.999]),  # lower bounds
        np.array([3.0,   7.0, 12.0, 15,  0.00])   # upper bounds
    )
}
def iv_newton(price, CP, S0, K, tau, r, sigma_init=0.5, tol=1e-10, it=300):
    sigma = np.full_like(price, sigma_init, dtype=float)
    for _ in range(it):
        diff = bs_price(CP, S0, K, sigma, tau, r) - price
        vega = bs_vega(S0, K, sigma, tau, r)
        sigma -= diff / np.where(vega > 1e-10, vega, np.inf) # Lower vega threshold
        if np.all(np.abs(diff) < tol): 
            break
    sigma = np.where((sigma > 0) & (sigma < 5), sigma, np.nan)
    return sigma

def rmse_iv(S0, K, tau, CP, iv_mkt, theta, r=0.0):
    try:
        pr_model = cos_pricer(CP, S0, K, tau, r, theta)
        if np.any(~np.isfinite(pr_model)):
            return 1e6  # Bad price 

        iv_model = iv_newton(pr_model, CP, S0, K, tau, r,
                             sigma_init=np.nan_to_num(iv_mkt, nan=0.5))
        if np.all(np.isnan(iv_model)):
            return 1e6
        err = iv_model - iv_mkt
        return 100 * np.sqrt(np.nanmean(err**2))
    except FloatingPointError:
        return 1e6


def calibrate_heston_snapshot(df_snap, v0_fixed, theta_0_remaining, bounds_remaining, r=0.0):
    CP, S0_vec, K, tau, _, iv_mkt = extract_inputs_from_df(df_snap)
    S0 = float(S0_vec[0])
    
    # Quick helper
    def objective_for_optim(theta_remaining):
        theta_full = np.insert(theta_remaining, 0, v0_fixed)
        return rmse_iv(S0, K, tau, CP, iv_mkt, theta_full, r)

    # Run the optimization on the remaining parameters
    res = optimize.minimize(
        objective_for_optim, 
        theta_0_remaining, 
        bounds=bounds_remaining, 
        method='L-BFGS-B', 
        options={'ftol': 1e-8, 'maxiter': 500}
    )

    return {
        'date'        : pd.to_datetime(df_snap['timestamp'].iloc[0]).date(),
        'model'       : 'heston_fixed_v0',
        'theta_v0'    : v0_fixed,  # The fixed v0
        'theta_v_bar' : res.x[0],  # Optimal v_bar
        'theta_kappa' : res.x[1],      # Fixed kappa
        'theta_gamma' : res.x[2],  # Optimal gamma
        'theta_rho'   : res.x[3],  # Optimal rho
        'iv_rmse'     : res.fun, 
        'n_strk'      : len(df_snap), 
        'success'     : res.success, 
        'message'     : res.message
    }

if __name__ == '__main__':
    eth_raw = load_0dte_data()
    eth = filter_otm_calibration(eth_raw)
    grouped = sorted(list(eth.groupby(eth['timestamp'].dt.date)))

    calib_summary = []
    option_fits = []

    # Initial guess for all params: [v0, v_bar, gamma, rho]
    theta_full_0, lb_full, ub_full = MODEL_CFG['heston']
    
    # We will optimize [v_bar, gamma, rho]
    theta_0_remaining = theta_full_0[1:] 
    bounds_remaining = list(zip(lb_full, ub_full))[1:]

    print("Starting sequential calibration with fixed v0 from ATM IV...")
    for date, snap in tqdm(grouped, desc='Calibrating per-day heston', unit='day'):
        
        # Find ATM IV and set fixed v0 
        atm_option_idx = snap['moneyness'].abs().idxmin()
        atm_iv = snap.loc[atm_option_idx, 'mark_iv'] / 100.0
        v0_fixed = atm_iv**2
        
        # Calibrate with the fixed v0 
        result = calibrate_heston_snapshot(snap, v0_fixed, theta_0_remaining, bounds_remaining)
        calib_summary.append(result)

        # Process results
        if result['success']:
            # 1. Reconstruct the optimal full theta vector from the result dictionary
            theta_opt_full = np.array([
                result['theta_v0'],    # The fixed v0
                result['theta_v_bar'],
                result['theta_kappa'], 
                result['theta_gamma'], # Note: your chf_heston uses kappa=0.5, this is vol-of-vol
                result['theta_rho']
            ])

            # Extract inputs and get fits 
            CP, S0, K, tau, _, iv_mkt = extract_inputs_from_df(snap)
            fitted_price = cos_pricer(CP, S0[0], K, tau, r=0.0, theta=theta_opt_full)
            fitted_iv = iv_newton(fitted_price, CP, S0[0], K, tau, r=0.0, sigma_init=iv_mkt)

            # Create a copy and store detailed results
            detailed_snap = snap.copy()
            detailed_snap['fitted_price'] = fitted_price
            detailed_snap['fitted_iv'] = fitted_iv
            detailed_snap['SE_fitted'] = (fitted_iv - iv_mkt)**2
            option_fits.append(detailed_snap)

            # Update the initial guess for the NEXT day's remaining parameters, i.e. warm start
            theta_0_remaining = theta_opt_full[1:]

    # Save the final results to CSV files 
    output_path = '/Users/joris/Documents/Master QF/Thesis/optimal-gamma-hedging/COS_Pricers/Data_8am/ETH'
    os.makedirs(output_path, exist_ok=True)

    # Save the summary of fits (same as your original code)
    df_calib_summary = pd.DataFrame(calib_summary)
    summary_filepath = os.path.join(output_path, 'Calibration', 'heston_calibration_summary.csv')
    df_calib_summary.to_csv(summary_filepath, index=False)
    print(f"\n--- Calibration Summary Finished ---")
    print(f"Summary results saved to '{summary_filepath}'")
    print(df_calib_summary.head())

    # Concatenate all the detailed daily results into a single DataFrame
    if option_fits:
        df_detailed_fits = pd.concat(option_fits, ignore_index=True)
        detailed_filepath = os.path.join(output_path, 'Options', 'heston_per_option_fits.csv')
        df_detailed_fits.to_csv(detailed_filepath, index=False)
        print(f"\n--- Detailed Fits Finished ---")
        print(f"Per-option results saved to '{detailed_filepath}'")
        print(df_detailed_fits.head())
    else:
        print("\nNo successful calibrations to generate detailed fits.")
