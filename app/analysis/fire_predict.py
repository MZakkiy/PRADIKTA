# app/analysis/fire_predict.py

import numpy as np
from scipy.optimize import minimize
import pandas as pd

# Core Model Component Functions

def df_factor(pfvi_prev, temp, r0):
    """Calculates the Drying Factor (DF)."""
    return (300 - pfvi_prev) * (0.4982 * np.exp(0.0905 * temp + 1.6096) - 4.268) * 10**(-3) / (1 + 10.88 * np.exp(-0.00173582677165354 * r0))

def df_factor_kdbi(kdbi_prev, temp, r0):
    """Calculates the Drying Factor (DF)."""
    return (203 - kdbi_prev) * (0.968 * np.exp(0.0875 * temp + 1.552) - 8.3) * 10**(-3) / (1 + 10.88 * np.exp(-0.00173582677165354 * r0))

def df_factor_kdbi_adj(kdbi_prev, temp, r0):
    """Calculates the Drying Factor (DF)."""
    return (203 - kdbi_prev) * (0.4982 * np.exp(0.0905 * temp + 1.6096) - 4.268) * 10**(-3) / (1 + 10.88 * np.exp(-0.00173582677165354 * r0))

def rf_factor(rf_current, rf_before):
    """Calculates the Rainfall Factor (RF)."""
    if np.isnan(rf_before) or rf_before <= 5.1:
        if rf_current < 5.1:
            return 0
        else:
            return rf_current - 5.1
    else:
        return rf_current

def wtf_factor(aH, bH, n, h, alpha):
    """Calculates the Water Table Factor (WTF)."""
    if n <= 0 or alpha <= 0:  # Avoid math errors with invalid parameters
        return 1e9 # Return a large number to be penalized by optimizer
    m = 1 - (1 / n)
    # The term (h/alpha) becomes 0 if h=0, so theta is 1 and WTF is aH.
    theta = (1 + (h / alpha)**n)**(-m)
    return aH -  bH * ((1 - theta) * 300)

def di_obs(sm, fc, sat):
    """Calculates the Observed Drought Index (DIobs) from Soil Moisture."""
    if (sat - fc) == 0: return 0
    return 300 * (1 - ((sm - fc) / (sat - fc)))

# Main Simulation and Optimization Functions

def calculate_pfvi(params, WT, SM, Rf, Rf_b, Temp, R0, dt):
    """Runs the full PFVI simulation for a given set of parameters."""
    aH, bH, n, alpha = params
    time_steps = len(WT)
    
    # Calculate water depth (h) as a positive value from water table (WT)
    h = np.where(WT > 0, 0, -WT)
    
    x = np.zeros(time_steps + 1)
    # Initialize PFVI with the first observed value
    x[0] = di_obs(SM[0], 40, 70) 

    for i in range(time_steps):
        x0 = np.clip(x[i], 0, 300) # Ensure PFVI is within bounds [0, 300]
        
        df = df_factor(x0, Temp[i], R0) * dt
        rf = rf_factor(Rf[i], Rf_b[i])
        wtf = wtf_factor(aH, bH, n, h[i], alpha)
        
        x[i+1] = x0 + df - rf - wtf
        
    return x[1:] # Return the calculated PFVI series

def calculate_kdbi(SM, Rf, Rf_b, Temp, R0, dt):
    time_steps = len(SM)
    
    x = np.zeros(time_steps + 1)
    # Initialize PFVI with the first observed value
    x[0] = di_obs(SM[0], 40, 70) 

    for i in range(time_steps):
        x0 = np.clip(x[i], 0, 300) # Ensure PFVI is within bounds [0, 300]
        
        df = df_factor_kdbi(x0, Temp[i], R0) * dt
        rf = rf_factor(Rf[i], Rf_b[i])
    
        x[i+1] = x0 + df - rf 
        
    return x[1:] # Return the calculated PFVI series

def calculate_kdbi_adj(SM, Rf, Rf_b, Temp, R0, dt):
    time_steps = len(SM)
    
    x = np.zeros(time_steps + 1)
    # Initialize PFVI with the first observed value
    x[0] = di_obs(SM[0], 40, 70) 

    for i in range(time_steps):
        x0 = np.clip(x[i], 0, 300) # Ensure PFVI is within bounds [0, 300]
        
        df = df_factor_kdbi(x0, Temp[i], R0) * dt
        rf = rf_factor(Rf[i], Rf_b[i])
    
        x[i+1] = x0 + df - rf 
        
    return x[1:] # Return the calculated PFVI series

def objective_function(params, WT, SM, Rf, Rf_b, Temp, R0, dt):
    """Objective function to be minimized. Calculates Mean Squared Error (MSE)."""
    # Run the model with the trial parameters
    predicted_pfvi = calculate_pfvi(params, WT, SM, Rf, Rf_b, Temp, R0, dt)
    
    # Calculate the "ground truth" DIobs
    observed_di = np.array([di_obs(sm, 40, 70) for sm in SM])
    
    # Return the Mean Squared Error
    return np.mean((predicted_pfvi - observed_di)**2)

# Main User-Facing Function

def fire_predict(WT, SM, Rf, Temp, R0=3000, dt=1, optim_method="Nelder-Mead"):
    """
    Calibrates and calculates the Peat Fire Vulnerability Index (PFVI).

    Args:
        WT (list or np.array): Time series of Water Table data.
        SM (list or np.array): Time series of Soil Moisture data.
        Rf (list or np.array): Time series of Rainfall data.
        Temp (list or np.array): Time series of Temperature data.
        R0 (float, optional): Constant parameter. Defaults to 3000.
        dt (int, optional): Time step, usually 1 for daily data. Defaults to 1.

    Returns:
        np.array: The final, calibrated PFVI time series.
    """
    # --- Data Preparation ---
    WT, SM, Rf, Temp = np.array(WT), np.array(SM), np.array(Rf), np.array(Temp)
    
    # Create the "rainfall before" series by shifting the rainfall data
    Rf_b = np.roll(Rf, 1)
    Rf_b[0] = np.nan 

    # --- Optimization ---
    # The R code uses a brute-force grid search to find a good starting point.
    # We will replicate this to avoid getting stuck in a poor local minimum.
    # print("Starting optimization... This may take a moment.")
    
    min_error = float('inf')
    best_params = None
    
    # Grid search for the best starting parameters
    # This is a simplified version of the R code's grid search logic
    for i in np.arange(0.2, 1.1, 0.4):
        for j in np.arange(0.2, 1.1, 0.4):
            for k in np.arange(1.1, 2.0, 0.4): # n > 1
                 for l in np.arange(0.2, 1.1, 0.4): # alpha > 0
                    initial_params = [i, j, k, l]
                    
                    # Bounds for parameters to ensure they are physically realistic
                    # n > 1, alpha > 0
                    bnds = ((None, None), (None, None), (1.01, None), (1e-6, None))

                    result = minimize(
                        fun=objective_function,
                        x0=initial_params,
                        args=(WT, SM, Rf, Rf_b, Temp, R0, dt),
                        method=optim_method,
                        bounds=bnds
                    )
                    
                    if result.success and result.fun < min_error:
                        min_error = result.fun
                        best_params = result.x
                        # print(f"New best error: {min_error:.4f} with params: {[f'{p:.3f}' for p in best_params]}")

    # print("\nOptimization complete.")
    # print(f"Final optimized parameters (aH, bH, n, alpha): {[f'{p:.4f}' for p in best_params]}")

    # --- Final Calculation ---
    # Calculate the final PFVI series using the best parameters found
    final_pfvi = calculate_pfvi(best_params, WT, SM, Rf, Rf_b, Temp, R0, dt)
    
    # Clip the final values to be within the [0, 300] range
    final_pfvi_clipped = np.clip(final_pfvi, 0, 300)
    
    return final_pfvi_clipped, best_params


if __name__ == '__main__':
    # =========================================================================
    # Example Usage with sample data
    # =========================================================================
    # Create some synthetic time-series data for demonstration
    days = 12
    np.random.seed(42)
    WT_data = -0.5 - np.sin(np.linspace(0, 4 * np.pi, days)) * 0.5 + np.random.normal(0, 0.05, days)
    SM_data = 55 + np.sin(np.linspace(0, 4 * np.pi, days)) * 14 + np.random.normal(0, 1, days)
    Rf_data = np.random.exponential(3, days)
    Rf_data[20:25] = 15 # A heavy rain event
    Temp_data = 28 + np.random.uniform(-2, 2, days)

    # Run the prediction
    predicted_values = fire_predict(WT=WT_data, SM=SM_data, Rf=Rf_data, Temp=Temp_data)
    
    print("\n--- Final PFVI Output ---")
    print(predicted_values)