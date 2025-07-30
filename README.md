## 📈 Advanced Option‑Pricing Research Toolkit

This repository provides a **research‑grade framework** for analysing short‑dated crypto options and measuring the relative performance of modern pricing models.  It combines **jump‑diffusion**, **double‑exponential jump**, and **stochastic‑volatility** dynamics with the **COS (Fourier‑Cosine) expansion**, a highly efficient pricing method that delivers exponential accuracy at linear computational cost.

### 🎯 Key Objectives

1. **Evaluate model fit** – Quantitatively benchmark Merton, Kou, and Heston models against observed BTC/ETH option prices and implied volatilities, with an emphasis on 0DTE maturities.
2. **Generate fair‑value surfaces** – Produce dense price/Greek grids in real time for use in hedging, relative‑value trading, or market‑making engines.
3. **Analyse truncation schemes** – Derive model‑consistent cumulant windows to guarantee fast COS convergence while avoiding arbitrage artefacts.

### 🔑 Main Features

| Module                            | Highlights                                                                                                                                            |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data ingestion**                | Hour‑ or second‑level snapshots; automatic UTC handling; one‑click filtering to “options expiring today.”                                             |
| **Characteristic‑function layer** | Vectorised implementations for Merton, Kou, and Heston; easy extension to other Lévy or SV models.                                                    |
| **Highly efficient COS pricer**   | • O(N) complexity per strike • truncation windows built from analytic cumulants • supports batch pricing across heterogeneous strikes and maturities. |
| **Calibration & error metrics**   | Plug‑and‑play loss functions on prices or IVs; optimisers ready for RMSE minimisation via L‑BFGS‑B or CMA‑ES.                                         |
| **Benchmark notebooks**           | End‑to‑end examples showing calibration, pricing comparison, and delta‑/vega‑surface visualisation.                                                   |

### 📚 Typical Workflow

1. **Load** BTC/ETH snapshot data via `load_0dte_data()`.
2. **Calibrate** each model to the IV surface (single‑day or rolling).
3. **Price** the full strike grid with the COS engine (`cos_price_batch`).
4. **Compare** model prices/Greeks to market mid‑quotes or realised PnL from hedged strategies.
5. **Iterate**: adjust jump intensity, vol‑of‑vol, or truncation length *L* for optimal trade‑off between speed and accuracy.

### 💡 Who Should Use This

* Quant researchers building **intraday hedging** or **gamma‑scalping** strategies.
* Market‑making desks that require **millisecond‑level fair values** for thousands of strikes.
* Academics studying **jump risk** and **stochastic volatility** in the rapidly evolving crypto‑derivatives space.

---

