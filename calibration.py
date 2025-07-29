import pandas as pd
import os
import numpy as np
import scipy.stats as st
import scipy.optimize as optimize

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
    def __init__(self):
        self.SECOND_PER_YEAR = 365 * 24 * 3600
        self.i = complex(0, 1)

    def extract_inputs_from_df(self, df):
        return (
            df['opt_type'].values,
            df['spot'].values,
            df['strike'].values,        
            df['time_to_maturity'].values,
            df['market_price'].values
            )

    def bs_call_put_prices(self, CP,S_0,K,sigma,tau,r):
        # Get Call and Put Option prices
        d1    = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0)) * (tau)) / (sigma * np.sqrt(tau))
        d2    = d1 - sigma * np.sqrt(tau)


        value = np.where(CP == 'call', st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * (tau)),
                            st.norm.cdf(-d2) * K * np.exp(-r * (tau)) - st.norm.cdf(-d1)*S_0)
        return value

    def bs_delta(self, CP,S_0,K,sigma,tau,r):
        # Get BS Delta values
        d1    = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0)) * (tau)) / (sigma * np.sqrt(tau))
        value = np.where(CP == 'call', st.norm.cdf(d1), st.norm.cdf(d1) - 1.0)
        return value

    def bs_impliedvol(self, CP,marketPrice,K,tau,S_0,r):
        func = lambda sigma: np.power(self.bs_call_put_prices(CP,S_0,K,sigma,tau,r) - marketPrice, 1.0)
        impliedVol = optimize.newton(func, 0.7, tol=1e-9)
        #impliedVol = optimize.brent(func, brack= (0.05, 2))
        return impliedVol
    
    # ------------------  MERTON  ------------------
    def chf_merton(self, u, sigma, xi, muJ, sigmaJ, r, tau_sec):
        tau = tau_sec / self.SEC_PER_YEAR
        omega_bar = xi * (np.exp(muJ + 0.5*sigmaJ**2) - 1)
        mu = r - 0.5*sigma**2 - omega_bar
        return np.exp(self.i*u*mu*tau -0.5*sigma**2*u**2*tau +
                    xi*tau*(np.exp(self.i*muJ*u - 0.5*sigmaJ**2*u**2) - 1))

    def phi_merton(self, u, sigma, xi, muJ, sigmaJ, r, tau_sec, S0, K):
        return lambda u: np.exp(self.i*u*(np.log(S0) - np.log(K)))  * self.chf_merton(u, sigma, xi, muJ, sigmaJ, r, tau_sec)

    # ------------------  KOU  ---------------------
    def chf_kou(self, u, sigma, xi, p1, alpha1, alpha2, r, tau_sec):
        p2 = 1 - p1
        tau = tau_sec / self.SECOND_PER_YEAR
        omega_bar = xi * (p1*alpha1/(alpha1-1) + (1-p1)*alpha2/(alpha2+1) - 1)
        mu = r - 0.5*sigma**2 - omega_bar
        return np.exp(self.i*u*mu*tau-0.5*sigma**2*u**2*tau +
                    xi*tau*((p1*alpha1)/(alpha1 - self.i*u) +
                            (p2*alpha2)/(alpha2 + self.i*u) - 1))

    def phi_kou(self, u, sigma, xi, p1, alpha1, alpha2, r, tau_sec, S0, K):
        return lambda u: np.exp(self.i*u*(np.log(S0) - np.log(K))) * self.chf_kou(u, sigma, xi, p1, alpha1, alpha2, r, tau_sec)

    # ------------------  HESTON  ------------------
    def chf_heston(self, u, tau_sec, r, kappa, v_bar, gamma, rho, v0):
        tau = tau_sec / self.SECOND_PER_YEAR 
        d1 = np.sqrt((kappa - self.i*rho*gamma*u)**2 + (u**2 + self.i*u)*gamma**2)
        g  = (kappa - self.i*rho*gamma*u - d1) / (kappa - self.i*rho*gamma*u + d1)
        term_r  = self.i*u*r*tau
        term_v0 = (v0 / gamma**2) * ((1 - np.exp(-d1*tau)) /
                                    (1 - g*np.exp(-d1*tau))) * (kappa - self.i*rho*gamma*u - d1)
        term_bar= (kappa*v_bar / gamma**2) * (tau*(kappa - self.i*rho*gamma*u - d1) -
                                            2*np.log((1 - g*np.exp(-d1*tau))/(1 - g)))
        return np.exp(term_r + term_v0 + term_bar)

    def phi_heston(self, u, kappa, v_bar, gamma, rho, v0, r, tau_sec, S0, K):
        return lambda u: np.exp(self.i*u*(np.log(S0) - np.log(K))) * self.chf_heston(u, tau_sec, r, kappa, v_bar, gamma, rho, v0)
    
    # ------------------  Cumulants  ------------------
    def merton_cumulants(self, tau, r, sigma, xi, muJ, sigmaJ):
        """ Calculate the cumulants for the Merton model. """
        omega_bar = tau * (np.exp(muJ + 0.5 * sigmaJ**2) - 1)
        c1 = tau * (r - omega_bar - 0.5 * sigma**2 + xi * muJ)
        c2 = tau * (sigma**2 + xi * muJ**2 + sigmaJ**2 * xi)
        c4 = tau * xi * (muJ**4 + 6 * muJ**2 * sigmaJ**2 + 3 * sigmaJ**4 * xi)
        return c1, c2, c4

    def kou_cumulants(self, tau, r, sigma, xi, alpha1, alpha2, p1):
        """ Calculate the cumulants for the Heston model. """
        p2 = 1 - p1
        omega_bar = xi * ((p1*alpha1)/(alpha1-1) + (p2*alpha2)/(alpha2+1) - 1)
        c1 = tau * (r - omega_bar - 0.5 * sigma**2 + ((xi * p1)/alpha1 - (xi * p2)/alpha2))
        c2 = tau * (sigma**2 + 2 * (xi * p1)/ alpha1**2 + 2 * (xi * p2)/alpha2**2)
        c4 = 24 * tau * xi * (p1 / alpha1**4 + p2 / alpha2**4) 
        return c1, c2, c4

    # ------------------  COS TRUNCATION WINDOW  ------------------
    def truncation_window(self, S_0, tau, model, r, sigma=None, params=None, L=8):
        """ Compute COS truncation window [a, b] for different models. """
        # TO DO: Calibrate the real parameters as well as value of L for each model and 
        params = params or {}

        if model.lower() == 'heston':
            a = -L * np.sqrt(tau)
            b =  L * np.sqrt(tau)

        elif model.lower() == 'merton':
            xi, muJ, sigmaJ, sigma  = self.merton_parameters() 
            c1, c2, c4 = self.merton_cumulants(tau, r, sigma, xi, muJ, sigmaJ)

            a = np.log(S_0) + c1 - L * np.sqrt(c2 + np.sqrt(c4))
            b = np.log(S_0) + c1 + L * np.sqrt(c2 + np.sqrt(c4))

        elif model.lower() == 'kou':
            xi, alpha1, alpha2, p1, sigma = self.kou_parameters()
            c1, c2, c4 = self.kou_cumulants(tau, r, sigma, xi, alpha1, alpha2, p1)

            a = np.log(S_0) + c1 - L * np.sqrt(c2 + np.sqrt(c4))
            b = np.log(S_0) + c1 + L * np.sqrt(c2 + np.sqrt(c4))
        return a, b

    # ------------------  COS PAYOFF COEFFICIENTS  ------------------
    def chi_psi_k(self, k, a, b):
        """Return lambda functions for chi_k(c,d) and psi_k(c,d) in the COS method."""
        omega = k * np.pi / (b - a)

        if k == 0:
            chi = lambda c, d: np.exp(d) - np.exp(c)
            psi = lambda c, d: d - c
        else:
            denom = 1 + omega**2

            chi = lambda c, d: (
                np.cos(omega * (d - a)) * np.exp(d)
                + omega * np.sin(omega * (d - a)) * np.exp(d)
                - np.cos(omega * (c - a)) * np.exp(c)
                - omega * np.sin(omega * (c - a)) * np.exp(c)
            ) / denom

            psi = lambda c, d: (b - a) / (k * np.pi) * (
                np.sin(omega * (d - a)) - np.sin(omega * (c - a))
            )

        return chi, psi

    def payoff_coefficients(self, option_type, K, k, c, d, a, b):
        # Get the chi and psi lambda functions for the COS method
        chi = self.chi_psi_k(k, a, b)[0]
        psi = self.chi_psi_k(k, a, b)[1]

        # Calculate the coefficients H_k based on the option type
        H_k = np.where(option_type == 'call', 
                (2.0 / (b - a)) * K * (chi(0,b) - psi(0,b)),
                (2.0 / (b - a)) * K * (-chi(a,0) + psi(a,0))    
        )
        return H_k

    # --------------------  TO DO: MODEL PARAMETERS  ------------------
    def merton_parameters(self): # TO DO: Calibrate the real parameters as well as value of L for each model 
        """ Return Merton model parameters. """
        xi     = 0.1  # Jump intensity
        muJ    = -0.1 # Mean of jump size
        sigmaJ = 0.2  # Std dev of jump size
        sigma  = 0.7  # Volatility of the underlying asset
        return xi, muJ, sigmaJ, sigma

    def kou_parameters(self): # TO DO: Calibrate the real parameters as well as value of L for each model
        """ Return Kou model parameters. """        
        xi      = 0.1   # Jump intensity
        alpha1  = 10.0  # Positive jump size parameter
        alpha2  = 5.0   # Negative jump size parameter
        p1      = 0.4   # Probability of positive jump
        sigma   = 0.2   # Volatility of the underlying asset
        return xi, alpha1, alpha2, p1, sigma

    def heston_parameters(self): # TO DO: Calibrate the real parameters as well as value of L for each model
        """ Return Heston model parameters. """
        kappa  = 1.0   # Mean reversion speed
        v_bar  = 0.04  # Long-run variance
        gamma  = 0.5   # Vol-of-vol
        rho    = -0.7  # Correlation between spot and vol
        v0     = 0.04  # Initial variance
        return kappa, v_bar, gamma, rho, v0

    # --------------------------------------------------------------------

    # --------------------  COS PRICER  --------------------
    # TO CHECK: 
    def cos_price_single(self, model, option_type, S0, K, tau_sec, r,
                        N=256, L=8):
        """ COS price for one European option (call/put) under Merton, Kou or Heston. """
        tau = tau_sec / self.SEC_PER_YEAR
        k   = np.arange(N, dtype=float)           # 0…N‑1

        # ----- model‑specific CF & truncation -----
        if model == 'merton':
            xi, muJ, sigmaJ, sigma = self.merton_parameters()
            a, b = self.truncation_window(S0, tau, 'merton', r, sigma)
            phi_func = self.phi_merton(sigma, xi, muJ, sigmaJ,
                                r, tau_sec, S0, K)
        elif model == 'kou':
            xi, a1, a2, p1, sigma = self.kou_parameters()
            a, b = self.truncation_window(S0, tau, 'kou', r, sigma)
            phi_func = self.phi_kou(sigma, xi, p1, a1, a2,
                            r, tau_sec, S0, K)
        elif model == 'heston':
            kappa, vbar, gamma, rho, v0 = self.heston_parameters()
            a, b = self.truncation_window(S0, tau, 'heston', r)
            phi_func = self.phi_heston(kappa, vbar, gamma, rho, v0,
                                r, tau_sec, S0, K)
        else:
            raise ValueError("model must be 'merton', 'kou', or 'heston'")

        # ----- χ, ψ, payoff coefficients -----
        H_k = self.payoff_coefficients(option_type, K, k, a=a, b=b,
                                c=None, d=None)   # c,d not needed in new helper

        u       = k * np.pi / (b - a)
        phi_u   = phi_func(u)                       # CF on grid
        weights = np.real(phi_u * np.exp(-1j*u*a))
        weights[0] *= 0.5

        price = np.exp(-r * tau) * np.sum(weights * H_k)
        return price

    def cos_price_batch(self, model, S0, K_vec, tau_sec_vec, r, option_type_vec,
                        N=256, L=8):
        """ Vectorised wrapper: price many options (possibly mixed calls/puts). """
        prices = np.empty_like(K_vec, dtype=float)
        for j, (K, tau_s, opt) in enumerate(zip(K_vec, tau_sec_vec, option_type_vec)):
            prices[j] = self.cos_price_single(model, opt, S0, K, tau_s, r, N=N, L=L)
        return prices



if __name__ == "__main__":
    # Init
    dataloader = DataLoader()




