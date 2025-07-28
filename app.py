import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.integrate import solve_ivp
from filterpy.kalman import EnsembleKalmanFilter
import matplotlib.pyplot as plt
import mosqlient # Import mosqlient

st.set_page_config(page_title="Dengue Forecasting", layout="wide")

# --- Model Functions and Parameters (Copied from previous cell) ---

# Parameters
init_params = {
    'alpha_h': 1/7, 'gamma_h': 1/7, 'mu_m': 1/10, 'nu_m': 1/10, 'alpha_m': 1/5,
    'beta0': 0.05, 'beta_temp': 0.005, 'beta_precip': 0.0005,
    'ν_vac': 0.0, 'ε_vac': 0.0, 'ν_wol': 0.0, 'μ_ctrl': 0.0, 'trans_wol': 0.1,
    'Tmin': 10.0, 'Tmax': 40.0, 'R0': 50.0, 'k_r': 0.1, 'beta_humid': 0.002,
    'BIRTH_DEATH_RATE': 1 / (70 * 52) # Assuming a constant birth/death rate relative to population
}

def briere(T, Tmin, Tmax, c=1e-4):
    return np.maximum(c * T * (T - Tmin) * np.sqrt(np.maximum(Tmax - T, 0)), 0)

def logistic_rainfall(R, R0, k):
    exp_input = -k * (R - R0)
    exp_input = np.clip(exp_input, -20, 20)
    return 1 / (1 + np.exp(exp_input))

def seir_sei_control(t, state, params, climate, Nh, N_mosq):
    Sh, Eh, Ih, Rh, Vh, Sm, Em, Im, Wm = np.maximum(state, 0)
    Nh = max(Nh, 1) # Ensure human population is at least 1
    N_mosq = max(N_mosq, 1) # Ensure mosquito population is at least 1

    T, R, H = climate['tempmed'], climate['precip_tot_sum'], climate['umidmed']

    beta0 = params.get('beta0_est', params.get('beta0', init_params['beta0']))
    beta_temp = params.get('beta_temp_est', params.get('beta_temp', init_params['beta_temp']))
    beta_precip = params.get('beta_precip_est', params.get('beta_precip', init_params['beta_precip']))
    Tmin = params.get('Tmin_est', params.get('Tmin', init_params['Tmin']))
    Tmax = params.get('Tmax_est', params.get('Tmax', init_params['Tmax']))
    R0 = params.get('R0_est', params.get('R0', init_params['R0']))
    k_r = params.get('k_r_est', params.get('k_r', init_params['k_r']))
    beta_humid = params.get('beta_humid_est', params.get('beta_humid', init_params['beta_humid']))

    nu_vac = init_params['ν_vac']
    epsilon_vac = init_params['ε_vac']
    nu_wol = init_params['ν_wol']
    mu_ctrl = init_params['μ_ctrl']
    trans_wol = init_params['trans_wol']
    alpha_h = init_params['alpha_h'] # Fixed parameters
    gamma_h = init_params['gamma_h'] # Fixed parameters
    mu_m = init_params['mu_m']     # Fixed parameters
    nu_m = init_params['nu_m']     # Fixed parameters
    alpha_m = init_params['alpha_m'] # Fixed parameters
    birth_death_rate = init_params['BIRTH_DEATH_RATE'] # Fixed birth/death rate

    T_br = briere(T, Tmin, Tmax)
    R_eff = logistic_rainfall(R, R0, k_r)
    β = max(0, T_br * R_eff + beta_humid * H)

    λ_h = β * (Im + Wm * trans_wol) / Nh # Human force of infection
    λ_m = β * Ih / Nh # Mosquito force of infection (per mosquito)

    # ODE system
    dSh = birth_death_rate * Nh - λ_h * Sh - birth_death_rate * Sh - nu_vac * Sh
    dEh = λ_h * Sh - alpha_h * Eh - birth_death_rate * Eh
    dIh = alpha_h * Eh - gamma_h * Ih - birth_death_rate * Ih
    dRh = gamma_h * Ih - birth_death_rate * Rh
    dVh = nu_vac * (Sh + Rh) - epsilon_vac * λ_h * Vh - birth_death_rate * Vh
    dSm = nu_m * N_mosq - λ_m * Sm - (mu_m + mu_ctrl + nu_wol) * Sm
    dEm = λ_m * Sm - (alpha_m + mu_m + mu_ctrl) * Em
    dIm = alpha_m * Em - (mu_m + mu_ctrl) * Im
    dWm = nu_wol * (Sm + Em + Im) - (mu_m + mu_ctrl) * Wm


    return [dSh, dEh, dIh, dRh, dVh, dSm, dEm, dIm, dWm]

# Adjusted initial state template - Proportions for human (summing to 1), proportions for mosquito
init_state_template = np.array([
    0.99, 0.005, 0.005, 0, 0.0, # Proportions of Nh for Sh, Eh, Ih, Rh, Vh (sum to 1)
    0.99, 0.005, 0.005, 0.0, # Proportions of N_mosq for Sm, Em, Im, Wm (sum to 1)
    init_params['Tmin'], init_params['Tmax'], init_params['R0'], init_params['k_r'], init_params['beta_humid'] # Parameters to estimate
])

# Initial P matrix - based on proportions for human and mosquito compartments
init_P_template = np.diag([*(1.0,) * 5, *(1.0,) * 4, 1.0, 1.0, 10.0, 0.1, 0.001]) # P for proportions and estimated parameters
Q = np.diag([10] * 9 + [1e-8, 1e-8, 1e-7, 1e-9, 1e-10]) # Process noise Q
n_ens = 100 # Ensemble size
rf_R_val = 300. # R value for RF observation in fusion


# Corrected hx functions: they should only depend on the state vector
# The measurement is the number of human cases (Ih * gamma_h)
def hx_base(state):
    # state[:9] contains the compartmental states (actual counts)
    return np.array([init_params['gamma_h'] * state[2]])

def hx_fusion(state):
     # state[:9] contains the compartmental states (actual counts)
     # This function predicts the measurements given the state.
     # The first measurement is ODE-derived cases.
     # The second measurement is the RF prediction, which is external to the ODE state.
     # We return a placeholder (0) for the RF prediction in the predicted measurement.
     return np.array([init_params['gamma_h'] * state[2], 0.])


# EnKF fx - Modified to use seir_sei_control and handle 9 compartments + 5 betas + Nh + N_mosq
def fx(x, dt):
    x = np.atleast_2d(x)
    cl = fx.climate # Climate data for this step
    pf = fx.pf # Fixed parameters
    Nh = fx.Nh # Human Population for this year
    N_mosq = fx.N_mosq # Mosquito Population for this year

    X = np.zeros_like(x)
    for i, s in enumerate(x):
        s = np.asarray(s)
        # Handle potential NaNs in state - replace with a small positive number
        if not np.isfinite(s).all():
            s = np.nan_to_num(s, nan=1e-5)

        # Extract estimated parameters from state vector (indices 9 to 13)
        # Add clipping to estimated parameters to keep them within reasonable bounds
        Tmin_est = np.clip(s[9], 0.0, 30.0)
        Tmax_est = np.clip(s[10], 30.0, 50.0)
        R0_est = np.clip(s[11], 0.0, 200.0)
        k_r_est = np.clip(s[12], 0.0, 1.0)
        beta_humid_est = np.clip(s[13], 0.0, 0.1)

        estimated_params = {
            'Tmin_est': Tmin_est, 'Tmax_est': Tmax_est, 'R0_est': R0_est, 'k_r_est': k_r_est, 'beta_humid_est': beta_humid_est
        }

        # Combine fixed and estimated parameters for ODE
        ode_par = {
            **pf, # Includes fixed parameters like alpha_h, gamma_h, etc.
            **estimated_params, # Includes estimated parameters
            'BIRTH_DEATH_RATE': init_params['BIRTH_DEATH_RATE'] # Assuming constant birth/death rate
        }

        # Pass the first 9 elements of the state (compartmental states) to the ODE solver
        sol = solve_ivp(seir_sei_control, [0, 7], s[:9], args=(ode_par, cl, Nh, N_mosq), t_eval=[7], method='RK45', events=None) # Use RK45 method

        # Update the state in X
        if sol.status == 0:
            X[i, :9] = sol.y[:, -1] # Update compartmental states (actual counts)
        else:
            X[i, :9] = s[:9] # If ODE failed, keep the previous compartmental states

        # Update estimated parameters in X after clipping
        X[i, 9] = estimated_params['Tmin_est']
        X[i, 10] = estimated_params['Tmax_est']
        X[i, 11] = estimated_params['R0_est']
        X[i, 12] = estimated_params['k_r_est']
        X[i, 13] = estimated_params['beta_humid_est']

    # Ensure noise has the correct dimensions (14 elements)
    noise = np.random.multivariate_normal(np.zeros(X.shape[1]), Q, X.shape[0])
    return np.maximum(X + noise, 0) # Ensure non-negativity after adding noise

def initialize_filter_state(yearly_df, yearly_avg_pop_series, target_col, init_params, init_state_template, init_P_template):
    year = yearly_df['year'].iloc[0]
    # Explicitly cast to float and ensure positive
    current_year_pop = float(yearly_avg_pop_series.get(year, 100000))
    if current_year_pop <= 0: current_year_pop = 100000.0 # Fallback for non-positive population

    # Explicitly cast to float and ensure positive
    current_year_mosq_pop = float(0.7 * current_year_pop)
    if current_year_mosq_pop <= 0: current_year_mosq_pop = 0.7 * 100000.0 # Fallback for non-positive mosquito population


    if not yearly_df.empty:
        first_obs = yearly_df.iloc[0][target_col]

        # Set initial Sh based on the specified multiplier (0.01) and current year's human population
        initial_Sh = max(current_year_pop * 0.01, 0) # Using the specified multiplier 0.01

        # Estimate initial infected humans based on the first observation and recovery rate
        initial_Ih = max(first_obs / init_params['gamma_h'], 0)
        # Set initial Exposed humans as a proportion of Infected (can be tuned, using a fixed ratio for now)
        initial_Eh = max(initial_Ih * 0.5, 0)

        # Set initial Recovered and Vaccinated (assuming initially 0 or negligible relative to Sh, Eh, Ih)
        initial_Rh = 0.0
        initial_Vh = 0.0

        # Adjust initial compartments to sum up to current_year_pop if needed (should not happen with Sh calc)
        current_human_sum = initial_Sh + initial_Eh + initial_Ih + initial_Rh + initial_Vh
        if current_human_sum > current_year_pop:
             # If sum exceeds population, scale down proportions to fit
             scale_factor = current_year_pop / current_human_sum
             initial_Sh *= scale_factor
             initial_Eh *= scale_factor
             initial_Ih *= scale_factor
             initial_Rh *= scale_factor
             initial_Vh *= scale_factor
             # Re-calculate Sh after scaling others to ensure sum is Nh
             initial_Sh = max(current_year_pop - initial_Eh - initial_Ih - initial_Rh - initial_Vh, 0)


        # Initialize mosquito compartments based on the calculated yearly mosquito population
        initial_Sm = current_year_mosq_pop * init_state_template[5] # Use proportion from template
        initial_Em = current_year_mosq_pop * init_state_template[6] # Use proportion from template
        initial_Im = current_year_mosq_pop * init_state_template[7] # Use proportion from template
        initial_Wm = current_year_mosq_pop * init_state_template[8] # Use proportion from template


        # Initialize parameters to be estimated from the template
        initial_params_est = init_state_template[9:]

        # Combine into the full initial state vector (actual counts for human and mosquito compartments)
        filter_init_state = np.array([
            initial_Sh, initial_Eh, initial_Ih, initial_Rh, initial_Vh,
            initial_Sm, initial_Em, initial_Im, initial_Wm,
            *initial_params_est
        ])

    else:
         # Fallback if data is empty - Initialize based on scaled template proportions
         initial_human_comps = init_state_template[:5] * current_year_pop
         initial_mosq_comps = init_state_template[5:9] * current_year_mosq_pop

         filter_init_state = np.array([
             *initial_human_comps,
             *initial_mosq_comps,
             *init_state_template[9:] # Estimated parameters
             ])


    filter_init_P = init_P_template.copy() # Start with the template P matrix (based on proportions)

    # Adjust the P for human compartments based on the current year's population scale
    filter_init_P[0, 0] *= current_year_pop # Sh
    filter_init_P[1, 1] *= current_year_pop # Eh
    filter_init_P[2, 2] *= current_year_pop # Ih
    filter_init_P[3, 3] *= current_year_pop # Rh
    filter_init_P[4, 4] *= current_year_pop # Vh

    # Adjust the P for mosquito compartments based on the current year's mosquito population scale
    filter_init_P[5, 5] *= current_year_mosq_pop # Sm
    filter_init_P[6, 6] *= current_year_mosq_pop # Em
    filter_init_P[7, 7] *= current_year_mosq_pop # Im
    filter_init_P[8, 8] *= current_year_mosq_pop # Wm

    # Ensure P matrix is a float array before returning
    filter_init_P = np.asarray(filter_init_P, dtype=float)

    return filter_init_state, filter_init_P, current_year_pop, current_year_mosq_pop

# Modified to run for a single year and return data for Streamlit display
def run_yearly_forecast_streamlit(enkf, df_full, train_data, test_data, rf_model, current_year_pop, current_year_mosq_pop, climate_cols, target_col, lag_cols, init_params, hx_func, R_val=200., rf_R_val=None):

    fx.pf = init_params
    fx.Nh = current_year_pop
    fx.N_mosq = current_year_mosq_pop
    enkf.fx = fx # Assign fx function to the filter

    # Run filter over training data (warm-up)
    for i in range(len(train_data)):
        t_idx = train_data.index[i]
        obs = train_data.loc[t_idx, target_col]
        cl = train_data.loc[t_idx, climate_cols].to_dict()

        if any(pd.isna(val) for val in cl.values()):
            continue

        fx.climate = cl # Update climate for fx

        # If it's the fusion filter, get RF prediction for update
        rf_pred = np.nan
        if rf_model is not None and rf_R_val is not None and t_idx in df_full.index and not df_full.loc[[t_idx], lag_cols + climate_cols].isnull().values.any():
             rf_pred = rf_model.predict(df_full.loc[[t_idx], lag_cols + climate_cols])[0]


        enkf.predict()

        # Update based on filter type
        if rf_R_val is None: # Base filter (no RF)
             enkf.update(np.array([obs]))
        else: # Fusion filter (with RF prediction as a second measurement)
             if not np.isnan(rf_pred):
                 enkf.update(np.array([obs, rf_pred]))
             else:
                 # If RF prediction is NaN, set a very large covariance for the second measurement
                 enkf.update(np.array([obs, 0]), R=np.array([[R_val, 0.], [0., 1e10]]))


    # --- Forecasting on Test Data ---
    forecast = {'true': [], 'pred': [], 'dates': []} # Also store dates

    for i in range(len(test_data) - 1):
        t_idx = test_data.index[i]
        fut_idx = test_data.index[i + 1]

        if t_idx not in df_full.index:
            continue

        obs = df_full.loc[t_idx, target_col] # Current observation for update
        cl = df_full.loc[t_idx, climate_cols].to_dict() # Climate for the current step

        if any(pd.isna(val) for val in cl.values()):
            st.warning(f"Skipping update for {t_idx} due to missing climate data.")
            forecast['true'].append(df_full.loc[fut_idx, target_col] if fut_idx in df_full.index else np.nan)
            forecast['pred'].append(np.nan)
            forecast['dates'].append(fut_idx)
            continue

        fx.climate = cl # Update climate for fx

        # Get RF prediction for fusion update using current data (still needed for fusion update)
        rf_pred_current = np.nan
        if rf_model is not None and rf_R_val is not None and t_idx in df_full.index and not df_full.loc[[t_idx], lag_cols + climate_cols].isnull().values.any():
            rf_pred_current = rf_model.predict(df_full.loc[[t_idx], lag_cols + climate_cols])[0]

        enkf.predict()

        # Update based on filter type
        if rf_R_val is None: # Base filter
             enkf.update(np.array([obs]))
        else: # Fusion filter
             if not np.isnan(rf_pred_current):
                 enkf.update(np.array([obs, rf_pred_current]))
             else:
                 enkf.update(np.array([obs, 0]), R=np.array([[R_val, 0.], [0., 1e10]]))

        # --- Generate 1-week ahead forecast ---
        if fut_idx not in df_full.index:
            forecast['true'].append(np.nan)
            forecast['pred'].append(np.nan)
            forecast['dates'].append(fut_idx)
            continue

        true_val = df_full.loc[fut_idx, target_col]
        forecast['true'].append(true_val)
        forecast['dates'].append(fut_idx)

        future_clim = df_full.loc[fut_idx, climate_cols].to_dict() # Climate for prediction step
        if any(pd.isna(val) for val in future_clim.values()):
             st.warning(f"Skipping forecast for {fut_idx} due to missing future climate data.")
             forecast['pred'].append(np.nan)
        else:
            fx.climate = future_clim # Update climate for the prediction step
            s = enkf.x.copy()
            estimated_params = {
                'Tmin_est': s[9], 'Tmax_est': s[10], 'R0_est': s[11], 'k_r_est': s[12], 'beta_humid_est': s[13]
            }
            ode_par = {
                **init_params,
                **estimated_params,
                'BIRTH_DEATH_RATE': init_params['BIRTH_DEATH_RATE']
            }
            sol = solve_ivp(seir_sei_control, [0, 7], s[:9], args=(ode_par, future_clim, current_year_pop, current_year_mosq_pop), t_eval=[7], method='RK45', events=None)
            pred = init_params['gamma_h'] * sol.y[2, -1] if sol.status == 0 else np.nan
            forecast['pred'].append(max(pred, 0) if not np.isnan(pred) else np.nan)

    # --- Evaluate Yearly Forecast ---
    metrics = {}
    min_len = min(len(forecast['true']), len(forecast['pred']))
    y = np.array(forecast['true'][:min_len])
    f = np.array(forecast['pred'][:min_len])
    dates_eval = np.array(forecast['dates'][:min_len])

    valid = ~np.isnan(f) & ~np.isnan(y)
    if valid.sum() > 0:
        metrics['mae'] = mean_absolute_error(y[valid], f[valid])
        metrics['rmse'] = np.sqrt(mean_squared_error(y[valid], f[valid]))

        denominator = y[valid] + 1e-5
        accuracy_vals = np.abs(f[valid] - y[valid]) / denominator
        metrics['pct_acc'] = 100 - np.mean(accuracy_vals) * 100 if len(accuracy_vals) > 0 else np.nan

        # Peak Week Timing Error
        if len(y) > 0 and np.max(y) > 0 and valid.sum() > 0:
            try:
                # Find peak in valid true data
                true_peak_relative_idx = np.argmax(y[valid])
                # Map back to the original list index
                true_peak_forecast_list_idx = np.where(valid)[0][true_peak_relative_idx]
                # Get the date from the dates list
                true_peak_date = dates_eval[true_peak_forecast_list_idx]

                # Find peak in valid predicted data
                predicted_peak_relative_idx = np.argmax(f[valid])
                 # Map back to the original list index
                predicted_peak_forecast_list_idx = np.where(valid)[0][predicted_peak_relative_idx]
                # Get the date from the dates list
                predicted_peak_date = dates_eval[predicted_peak_forecast_list_idx]

                # Calculate difference in weeks
                weeks_off = np.abs((predicted_peak_date - true_peak_date).days / 7.0)
                metrics['peak_weeks_off'] = weeks_off
            except Exception as e:
                 st.warning(f"Error calculating peak week timing: {e}")
                 metrics['peak_weeks_off'] = np.nan
        else:
            metrics['peak_weeks_off'] = np.nan

    else:
        metrics = {'mae': np.nan, 'rmse': np.nan, 'pct_acc': np.nan, 'peak_weeks_off': np.nan}


    # Return data for plotting and metrics
    plot_data = pd.DataFrame({
        'Date': forecast['dates'],
        'True Cases': forecast['true'],
        'Predicted Cases': forecast['pred']
    }).set_index('Date')

    return plot_data, metrics


# --- Streamlit App Logic ---
st.title("Dengue Forecasting: Predicted vs Observed Cases")
st.header("Forecast Visualization and Evaluation")

api_key = st.text_input("Enter your Mosqlimate API Key:", type="password", key="api_key_input")
geocode = st.number_input("Enter geocode (e.g., 3304557):", value=3304557, key="geocode_input")
selected_year = st.number_input("Enter year to visualize (e.g., 2023):", min_value=2010, max_value=2024, value=2023, key="year_input") # Limit years to available data
run_button = st.button("Run Forecast", key="run_button")

if api_key and run_button:
    # --- Data Acquisition (Updated) ---
    climate_cols = ['tempmed', 'precip_tot_sum', 'umidmed']
    target_col = 'casos_est'
    pop_col = 'pop'
    rf_R_val = 300. # R value for RF observation in fusion
    n_ens = 100 # Ensemble size
    R_val = 200. # R value for case observations
    lag_cols = [] # Define lag_cols here to be populated after lag creation

    try:
        with st.spinner(f"Downloading data for geocode {geocode}..."):
            # Fetch data for a wider range to allow for lags and training data from previous years
            # Need data starting at least 8 weeks before the start of the selected year
            start_year_data = max(2010, selected_year - 2) # Get data from up to 2 years prior
            end_year_data = 2024 # Get data up to 2024

            climate_df = mosqlient.get_climate_weekly(
                api_key = api_key,
                start = f"{start_year_data}01",
                end = f"{end_year_data}52",
                geocode = geocode,
            )
            cases_df = mosqlient.get_infodengue(
                api_key = api_key,
                disease='dengue',
                start_date = f"{start_year_data}-01-01",
                end_date = f"{end_year_data}-12-31",
                geocode = geocode,
            )

        # Data cleaning and merging (Updated)
        if 'SE' in cases_df.columns:
            cases_df['SE'] = cases_df['SE'].astype(int)
        if 'epiweek' in climate_df.columns:
            climate_df['epiweek'] = climate_df['epiweek'].astype(int)
        if 'municipio_geocodigo' in cases_df.columns:
            cases_df = cases_df.rename(columns={'municipio_geocodigo': 'geocode'})
        if 'geocodigo' in climate_df.columns:
            climate_df = climate_df.rename(columns={'geocodigo': 'geocode'})

        # Ensure data_date is datetime in climate_df for merging
        climate_df['data_date'] = pd.to_datetime(climate_df['data_date'])

        merged = pd.merge(
            cases_df,
            climate_df,
            left_on=['geocode', 'SE'],
            right_on=['geocode', 'epiweek'],
            how='outer' # Use outer merge to keep all data points
        )

        # Use the cases data_iniSE as the primary date index
        merged['data_iniSE'] = pd.to_datetime(merged['data_iniSE'])
        merged.set_index('data_iniSE', inplace=True)
        merged.sort_index(inplace=True)

        st.success("Data downloaded and merged successfully.")
    except Exception as e:
        st.error(f"Error downloading or merging data: {e}")
        st.stop() # Stop execution if data loading fails


    # --- Modeling & Forecast Section (Adapted from previous code) ---
    try:
        # Ensure required columns exist after merge
        required_cols = climate_cols + [target_col, pop_col, 'SE']
        if not all(col in merged.columns for col in required_cols):
             missing = [col for col in required_cols if col not in merged.columns]
             st.error(f"Missing required columns after merging: {missing}")
             st.stop()

        # Calculate yearly average population BEFORE dropping NAs for lags
        merged['year'] = merged['SE'].astype(str).str[:4].astype(int)
        yearly_avg_pop = merged.groupby('year')[pop_col].mean()

        # Create lags (on the full merged_df before filtering for the year)
        lag_cols = []
        for feat in [target_col] + climate_cols:
            for lag in range(1, 5):
                # Ensure the column exists before creating lag
                if feat in merged.columns:
                    merged[f'{feat}_lag{lag}'] = merged[feat].shift(lag)
                    lag_cols.append(f'{feat}_lag{lag}')
                else:
                    st.warning(f"Cannot create lag for missing feature: {feat}")


        # Drop NA rows (after creating all lags) - subset only on columns used in forecasting
        subset_cols_for_dropna = lag_cols + climate_cols + [target_col, pop_col, 'SE']
        # Filter for existing columns before dropping
        subset_cols_for_dropna = [col for col in subset_cols_for_dropna if col in merged.columns]

        merged.dropna(subset=subset_cols_for_dropna, inplace=True)


        # Filter data for the selected year and the necessary training period
        # Training data: Data before the start of the test period for the selected year
        # Test data: Data within the selected year, after the training period

        # Find the start and end dates for the selected year
        start_date_year = pd.to_datetime(f'{selected_year}-01-01')
        end_date_year = pd.to_datetime(f'{selected_year}-12-31')

        # Define the end date of the training period for the selected year (first 8 weeks)
        end_of_train_date = start_date_year + pd.Timedelta(weeks=7) # End of week 8

        # Filter data for the selected year and previous year(s) needed for training
        # Need data starting at least 8 weeks before the start of the selected year
        train_start_date = start_date_year - pd.Timedelta(weeks=8)

        # Use merged data for the relevant date range
        yearly_df_full = merged.loc[train_start_date:end_date_year].copy()


        if yearly_df_full.empty:
            st.warning(f"No data available for the selected year {selected_year} and the required training period after dropping NA values.")
        else:
            # Split into train and test for the selected year
            # Train data: All data in yearly_df_full *before or including* the end_of_train_date
            # Test data: All data in yearly_df_full *after* the end_of_train_date

            train_data = yearly_df_full.loc[yearly_df_full.index <= end_of_train_date].copy()
            test_data = yearly_df_full.loc[yearly_df_full.index > end_of_train_date].copy()

            if train_data.empty:
                 st.warning(f"Insufficient training data available for {selected_year} after dropping NA values.")
            elif test_data.empty:
                 st.warning(f"Insufficient test data available for {selected_year} after dropping NA values.")
            else:
                st.write(f"Running forecast for {selected_year}...")

                # Train RF model for the year (still needed for fusion filter)
                # Train RF on all available data up to the end of the training period
                X_train_rf = train_data[lag_cols + climate_cols]
                # Target for RF is the case count 1 week ahead
                y_train_rf = train_data[target_col].shift(-1).dropna()
                # Align X_train_rf and y_train_rf indices
                X_train_rf = X_train_rf.loc[y_train_rf.index]

                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                if not X_train_rf.empty and len(y_train_rf) > 0: # Check if there is data to train
                    rf.fit(X_train_rf, y_train_rf)
                else:
                    rf = None # Keep rf as None if training data is empty or no target


                # Initialize filter state and parameters for the year
                # Use the first row of the selected year's data for initial state if available
                initial_state_df = yearly_df_full.loc[start_date_year:]
                if not initial_state_df.empty:
                    filter_init_state, filter_init_P, current_year_pop, current_year_mosq_pop = initialize_filter_state(
                        initial_state_df.iloc[:1], # Use only the first row of the selected year for initial state
                        yearly_avg_pop, target_col, init_params, init_state_template, init_P_template
                    )
                else:
                     st.error(f"Could not initialize filter state for {selected_year}. No data found for the start of the year.")
                     st.stop()


                # Initialize EnKF filters for the year
                enkf_base = EnsembleKalmanFilter(x=filter_init_state.copy(), P=filter_init_P.copy(), dim_z=1, dt=7.0, N=n_ens, fx=lambda x, dt: x, hx=hx_base)
                enkf_base.Q = Q
                enkf_base.R = np.array([[R_val]])
                enkf_base.ensembles = filter_init_state.copy() + np.random.multivariate_normal(np.zeros(len(filter_init_state)), np.asarray(filter_init_P, dtype=float), n_ens)

                enkf_fus = EnsembleKalmanFilter(x=filter_init_state.copy(), P=filter_init_P.copy(), dim_z=2, dt=7.0, N=n_ens, fx=lambda x, dt: x, hx=hx_fusion)
                enkf_fus.Q = Q
                enkf_fus.R = np.diag([R_val, rf_R_val]) # Use defined R values
                enkf_fus.ensembles = filter_init_state.copy() + np.random.multivariate_normal(np.zeros(len(filter_init_state)), np.asarray(filter_init_P, dtype=float), n_ens)


                # --- Run Forecasting and Evaluation for the Selected Year ---
                with st.spinner(f"Generating forecast for {selected_year}..."):
                    # Run Base Filter Forecast
                    forecast_data_base, metrics_base = run_yearly_forecast_streamlit(
                        enkf_base, merged, train_data, test_data, rf_model=None, current_year_pop=current_year_pop, current_year_mosq_pop=current_year_mosq_pop, climate_cols=climate_cols, target_col=target_col, lag_cols=lag_cols, init_params=init_params, hx_func=hx_base, R_val=R_val, rf_R_val=None
                    )

                    # Run Fusion Filter Forecast
                    forecast_data_fus, metrics_fus = run_yearly_forecast_streamlit(
                        enkf_fus, merged, train_data, test_data, rf_model=rf, current_year_pop=current_year_pop, current_year_mosq_pop=current_year_mosq_pop, climate_cols=climate_cols, target_col=target_col, lag_cols=lag_cols, init_params=init_params, hx_func=hx_fusion, R_val=R_val, rf_R_val=rf_R_val
                    )

                st.success("Forecast generated!")

                # --- Display Results ---
                st.subheader(f"Forecast for {selected_year}")

                # Combine forecast data for plotting
                plot_df = pd.DataFrame({
                    'True Cases': forecast_data_base['True Cases'], # True cases are the same for both
                    'BASE Prediction': forecast_data_base['Predicted Cases'],
                    'FUSION Prediction': forecast_data_fus['Predicted Cases']
                })

                st.line_chart(plot_df)

                st.subheader("Evaluation Metrics")
                metrics_df = pd.DataFrame({
                    'Metric': ['MAE', 'RMSE', 'Avg Accuracy (%)', 'Peak Weeks Off'],
                    'BASE': [metrics_base.get('mae'), metrics_base.get('rmse'), metrics_base.get('pct_acc'), metrics_base.get('peak_weeks_off')],
                    'FUSION': [metrics_fus.get('mae'), metrics_fus.get('rmse'), metrics_fus.get('pct_acc'), metrics_fus.get('peak_weeks_off')]
                }).set_index('Metric')

                st.table(metrics_df.round(2))


    except Exception as e:
        st.error(f"An error occurred during modeling or forecasting: {e}")
        st.write("Please check the data for the selected year and ensure it has the necessary columns and format.")
