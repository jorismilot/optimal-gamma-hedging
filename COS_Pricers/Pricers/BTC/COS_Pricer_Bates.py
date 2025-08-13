import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import scipy.stats as st
import scipy.optimize as optimize
from tqdm import tqdm  

i = 1j    # imag unit

def load_0dte_data(hour):
    btc_df_path = os.path.join(f'/Users/joris/Documents/Master QF/Thesis/optimal-gamma-hedging/Data/calibration_data/{hour}', f'btc_{hour}_0dte_data.csv')
    btc_df = pd.read_csv(btc_df_path)

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
        df['mark_iv'].values / 100.0,
        )

# Black-Scholes formulas 
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

# COS Pricing Mechanism 
def chf_bates(u, r, tau, theta):
    kappa, vbar, gamma, rho, v0, xi, muJ, sigmaJ = theta
    d = np.sqrt(np.power(kappa-gamma*rho*i*u,2)+((u**2)+i*u)*(gamma**2))
    g  = (kappa-gamma*rho*i*u-d)/(kappa-gamma*rho*i*u+d)
    C  = (1.0-np.exp(-d*tau))/((gamma**2)*(1.0-g*np.exp(-d*tau)))*(kappa-gamma*rho*i*u-d)

    # Exclude the term -r*tau, as the discounting is performed within the COS method
    AHes = r * i*u *tau + kappa*vbar*tau/gamma/gamma *(kappa-gamma*\
        rho*i*u-d) - 2*kappa*vbar/gamma/gamma*np.log((1.0-g*np.exp(-d*tau))/(1.0-g))
    
    A = AHes - xi * i * u * tau *(np.exp(muJ+0.5*sigmaJ**2) - 1.0) + \
            xi * tau * (np.exp(i*u*muJ - 0.5 * sigmaJ**2 * u**2) - 1.0)
    
    return np.exp(A + C*v0) 

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


def cos_pricer(CP, S0, K, tau, r, theta, N=256, L=12):
    K, tau, CP = np.asarray(K), np.asarray(tau), np.asarray(CP)
    log_ratio  = np.log(S0 / K)
    a = (log_ratio - L*np.sqrt(tau))[None, :] # CHECK IF WE SHOULD SET log_ratio = 0.0 
    b = (log_ratio + L*np.sqrt(tau))[None, :] # CHECK IF WE SHOULD SET log_ratio = 0.0 
    k = np.arange(N, dtype=float)[:, None] 
    u = k * np.pi / (b - a)
    phi = chf_bates(u, r, tau, theta)
    
    H_k = payoff_coefficients_vec(CP[None, :], k, a, b)
    
    w = np.real(phi * np.exp(-1j * u * (a - log_ratio[None, :])))
    w[0, :] *= 0.5
    prices = np.exp(-r * tau) * K * np.sum(w * H_k, axis=0)
    return prices.astype(float)

# Calibration Mechanism 
MODEL_CFG = {
    'bates': (
        # Parameters: [kappa, vbar, gamma, rho, v0, xi, muJ, sigmaJ]
        # Heston Params: kappa, vbar, gamma, rho, v0; Merton Params: xi, muJ, sigmaJ
        # Initial Guess
        np.array([2.0, 0.6, 0.5, -0.6, 0.6, 1.0, -0.10, 0.20]),
        # Lower Bounds
        np.array([0.1, 0.01, 0.01, -0.99, 0.01, 4.0, -0.5, 0.01]),
        # Upper Bounds
        np.array([20.0, 2.0, 2.0, 0.99, 2.0, 45.0, 0.5, 0.6])
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
    pr_model = cos_pricer(CP, S0, K, tau, r, theta)
    iv_model = iv_newton(pr_model, CP, S0, K, tau, r, sigma_init=np.nan_to_num(iv_mkt, nan=0.5))
    # Return a large penalty if inversion fails completely
    if np.all(np.isnan(iv_model)): 
        return 1e6
    return 100 * np.sqrt(np.nanmean((iv_model - iv_mkt)**2))


def calibrate_bates_snapshot(df_snap, theta_0, bounds, r=0.0):
    CP, S0_vec, K, tau, _, iv_mkt = extract_inputs_from_df(df_snap)
    S0 = float(S0_vec[0])
    
    loss = lambda th: rmse_iv(S0, K, tau, CP, iv_mkt, th, r)
    
    res  = optimize.minimize(loss, theta_0,
                             bounds=bounds,
                             method='L-BFGS-B',
                             options={'ftol': 1e-8, 'maxiter': 500})
    return {
            'date'      : pd.to_datetime(df_snap['timestamp'].iloc[0]).date(),
            'model'     : 'bates',
            'theta_kappa'   : res.x[0], 'theta_vbar': res.x[1], 'theta_gamma': res.x[2], 'theta_rho': res.x[3], 'theta_v0': res.x[4], 'theta_xi': res.x[5], 'theta_muJ': res.x[6], 'theta_sigmaJ': res.x[7],
            'iv_rmse'   : res.fun,
            'n_strk'    : len(snap),
            'success'   : res.success,
            'message'   : res.message
        }     

if __name__ == '__main__':
    hours = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
    for hour in hours:
        btc_raw = load_0dte_data(hour)
        btc = filter_otm_calibration(btc_raw) # filter for ATM/OTM
        grouped = sorted(list(btc.groupby(btc['timestamp'].dt.date)))

        calib_summary = []
        option_fits = []

        theta_0, lb, ub = MODEL_CFG['bates']
        bounds = list(zip(lb, ub))

        print("Starting sequential calibration with robust warm starts...")
        for date, snap in tqdm(grouped, desc='Calibrating per-day bates', unit='day'):
            # Calibrate the bates model for the current snapshot
            result = calibrate_bates_snapshot(snap, theta_0, bounds)
            calib_summary.append(result)

            # If the calibration for the day was successful, calculate and store detailed results
            if result['success']:
                # Reconstruct the optimal theta vector from the result dictionary
                theta_opt = np.array([
                    result['theta_kappa'],
                    result['theta_vbar'],
                    result['theta_gamma'],
                    result['theta_rho'],
                    result['theta_v0'],
                    result['theta_xi'],
                    result['theta_muJ'],
                    result['theta_sigmaJ']
                ])

                # Extract inputs and get fits 
                CP, S0, K, tau, _, iv_mkt = extract_inputs_from_df(snap)
                fitted_price = cos_pricer(CP, S0[0], K, tau, r=0.0, theta=theta_opt)
                fitted_iv = iv_newton(fitted_price, CP, S0[0], K, tau, r=0.0, sigma_init=iv_mkt)

                # Create a copy of the day's snapshot and add the new columns
                detailed_snap = snap.copy()
                detailed_snap['fitted_price'] = fitted_price
                detailed_snap['fitted_iv'] = fitted_iv
                detailed_snap['SE_fitted'] = (fitted_iv - iv_mkt)**2

                option_fits.append(detailed_snap)

                # Update the initial guess for the next day (warm start)
                theta_0 = theta_opt

        # Save the final results to CSV files 
        output_path = '/Users/joris/Documents/Master QF/Thesis/optimal-gamma-hedging/COS_Pricers/Hedging/Hourly_Results/'
        os.makedirs(os.path.join(output_path, 'Calibration', f'{hour}'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'Options', f'{hour}'), exist_ok=True)

        # Save the summary of fits (same as your original code)
        df_calib_summary = pd.DataFrame(calib_summary)
        summary_filepath = os.path.join(output_path, 'Calibration', f'{hour}', f'bates_calibration_summary_{hour}.csv')
        df_calib_summary.to_csv(summary_filepath, index=False)
        print(f"\n--- Calibration Summary Finished ---")
        print(f"Summary results saved to '{summary_filepath}'")
        print(df_calib_summary.head())

        # Concatenate all the detailed daily results into a single DataFrame
        if option_fits:
            df_detailed_fits = pd.concat(option_fits, ignore_index=True)
            detailed_filepath = os.path.join(output_path, 'Options', f'{hour}', f'bates_per_option_fits_{hour}.csv')
            df_detailed_fits.to_csv(detailed_filepath, index=False)
            print(f"\n--- Detailed Fits Finished ---")
            print(f"Per-option results saved to '{detailed_filepath}'")
            print(df_detailed_fits.head())
        else:
            print("\nNo successful calibrations to generate detailed fits.")
