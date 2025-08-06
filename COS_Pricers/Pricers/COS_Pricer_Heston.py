import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import scipy.stats as st
import scipy.optimize as optimize
from tqdm import tqdm  

i = 1j    # imag unit

def load_0dte_data(file_path='Data/calibration_data/08/btc_08_0dte_data.csv'):
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
    d = np.sqrt(np.power(kappa-gamma*rho*i*u,2)+(u**2+i*u)*gamma**2)
    g  = (kappa-gamma*rho*i*u-d)/(kappa-gamma*rho*i*u+d)
    C  = (1.0-np.exp(-d*tau))/(gamma*gamma*(1.0-g*np.exp(-d*tau)))\
        *(kappa-gamma*rho*i*u-d)

    # Note that we exclude the term -r*tau, as the discounting is performed in the COS method
    A  = r * i*u *tau + kappa*v_bar*tau/gamma/gamma *(kappa-gamma*rho*i*u-d)\
        - 2*kappa*v_bar/gamma/gamma*np.log((1.0-g*np.exp(-d*tau))/(1.0-g))

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
def cos_pricer(CP, S0, K, tau, r, theta, N=512, L=10):
    K, tau, CP = np.asarray(K), np.asarray(tau), np.asarray(CP)
    log_ratio  = np.log(S0 / K)
    
    a = (0 - L*np.sqrt(tau))[None, :] # TO DO: SHOULD THIS BE 0 OR log_ratio?
    b = (0 + L*np.sqrt(tau))[None, :] # TO DO: SHOULD THIS BE 0 OR log_ratio?
    
    k = np.arange(N, dtype=float)[:, None]
    u = k * np.pi / (b - a)
    
    phi_log_price = chf_heston(u, tau, r, theta)
    phi_log_return = phi_log_price * np.exp(-1j * u * np.log(S0))

    H_k = payoff_coefficients_vec(CP[None, :], k, a, b)
    w = np.real(phi_log_return * np.exp(-1j * u * (a - log_ratio[None, :])))
    w[0, :] *= 0.5
    
    prices = np.exp(-r * tau) * K * np.sum(w * H_k, axis=0)
    return prices.astype(float)


# Calibration Mechanism 
MODEL_CFG = {
    'heston': (
        np.array([0.6, 0.64, 2.0, 0.5, -0.7]),  # initial guess
        np.array([0.001, 0.001, 0.01, 0.05, -0.999]),  # lower bounds
        np.array([5.0,   5.0,   15.0, 5.0,   0.999])   # upper bounds
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
            return 1e6  # Bad price → penalise heavily

        iv_model = iv_newton(pr_model, CP, S0, K, tau, r,
                             sigma_init=np.nan_to_num(iv_mkt, nan=0.5))
        if np.all(np.isnan(iv_model)):
            return 1e6
        err = iv_model - iv_mkt
        return 100 * np.sqrt(np.nanmean(err**2))
    except FloatingPointError:
        return 1e6


def calibrate_heston_snapshot(df_snap, theta_0, bounds, r=0.0):
    CP, S0_vec, K, tau, _, iv_mkt = extract_inputs_from_df(df_snap)
    S0 = float(S0_vec[0])
    def penalised_loss(th):
        # Base RMSE
        base_loss = rmse_iv(S0, K, tau, CP, iv_mkt, th, r)
        # Soft Feller condition penalty
        v0, v_bar, kappa, gamma, rho = th
        feller_violation = max(0.0, gamma**2 - 2*kappa*min(v0, v_bar))
        return base_loss + 1e3 * feller_violation**2

    loss = penalised_loss
    res  = optimize.minimize(loss, theta_0, bounds=bounds, method='L-BFGS-B', options={'ftol': 1e-8, 'maxiter': 500})
    return {
        'date'      : pd.to_datetime(df_snap['timestamp'].iloc[0]).date(),
        'model'     : 'heston',
        'theta_v0'  : res.x[0], 'theta_v_bar': res.x[1], 'theta_kappa': res.x[2], 'theta_gamma': res.x[3], 'theta_rho': res.x[4],
        'iv_rmse'   : res.fun, 
        'n_strk'    : len(df_snap), 
        'success'   : res.success, 
        'message'   : res.message
    }

# Main Execution Block
if __name__ == '__main__':
    btc_raw = load_0dte_data()       
    btc = filter_otm_calibration(btc_raw)
    
    # Sort groups by date to ensure sequential processing
    grouped = sorted(list(btc.groupby(btc['timestamp'].dt.date)))

    # Sequential Calibration with Robust "Warm Starts"
    calib_out = []

    # Get the initial guess and bounds from config with `theta_0` the starting point and fallback
    theta_0, lb, ub = MODEL_CFG['heston']
    bounds = list(zip(lb, ub))

    print("Starting sequential Heston calibration with robust warm starts...")
    for date, snap in tqdm(grouped, desc='Calibrating per-day Heston', unit='day'):
        result = calibrate_heston_snapshot(snap, theta_0, bounds)

        # Append the results to our list
        calib_out.append(result)
        
        # Update the initial guess for the next day only if this run succeeded -> if failed, reuse the same `theta_0` for the next day.
        if result['success']:
            theta_0 = np.array([
                result['theta_v0'], 
                result['theta_v_bar'], 
                result['theta_kappa'], 
                result['theta_gamma'],
                result['theta_rho']
            ])
        else:
            # Fallback to some “safe” middle guess if failure
            theta_0 = MODEL_CFG['heston'][0]

     # --- Save the final results to CSV files ---
    output_path = '/Users/joris/Documents/Master QF/Thesis/optimal-gamma-hedging/COS_Pricers/Data/'
    os.makedirs(output_path, exist_ok=True)
    
    # Save the summary of fits
    df_calib = pd.DataFrame(calib_out)
    df_calib.to_csv(os.path.join(output_path, 'heston_calibration_results_final.csv'), index=False)
    print("\n--- Final Calibration Finished ---")
    print("Results saved to 'heston_calibration_results_final.csv'")
    print(df_calib.head())