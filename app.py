import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.integrate import solve_ivp
from filterpy.kalman import EnsembleKalmanFilter
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dengue Forecasting", layout="wide")

# User inputs
st.title("Dengue Forecasting: Predicted vs Observed Cases")
api_key = st.text_input("Enter your Mosqlimate API Key:", type="password")
geocode = st.number_input("Enter geocode (e.g., 3304557):", value=3304557)
year = st.number_input("Enter year to visualize (e.g., 2023):", min_value=2000, max_value=2100, value=2023)
run_button = st.button("Run Forecast")

if api_key and run_button:
    # --- Data Acquisition ---
    import mosqlient

    try:
        with st.spinner("Downloading data..."):
            climate_df = mosqlient.get_climate_weekly(
                api_key = api_key,
                start = "201001",
                end = "202452",
                geocode = geocode,
            )
            cases_df = mosqlient.get_infodengue(
                api_key = api_key,
                disease='dengue',
                start_date = "2010-01-01",
                end_date = "2024-12-31",
                geocode = geocode,
            )

        # Data cleaning and merging
        if 'SE' in cases_df.columns:
            cases_df['SE'] = cases_df['SE'].astype(int)
        if 'epiweek' in climate_df.columns:
            climate_df['epiweek'] = climate_df['epiweek'].astype(int)
        if 'municipio_geocodigo' in cases_df.columns:
            cases_df = cases_df.rename(columns={'municipio_geocodigo': 'geocode'})
        if 'geocodigo' in climate_df.columns:
            climate_df = climate_df.rename(columns={'geocodigo': 'geocode'})

        merged = pd.merge(
            cases_df,
            climate_df,
            left_on=['geocode', 'SE'],
            right_on=['geocode', 'epiweek'],
            how='outer'
        )
        merged['data_iniSE'] = pd.to_datetime(merged['data_iniSE'])
        merged.set_index('data_iniSE', inplace=True)
        st.success("Data downloaded and merged successfully.")
    except Exception as e:
        st.error(f"Error downloading or merging data: {e}")
        st.stop()

    # --- Modeling & Forecast Section (from frameworkpy) ---
    try:
        # Setup columns
        climate_cols = ['tempmed', 'precip_tot_sum', 'umidmed']
        target_col = 'casos_est'
        # Ensure required columns exist
        for col in climate_cols + [target_col]:
            if col not in merged.columns:
                st.error(f"Column {col} not found in merged data. Check data source or column names.")
                st.stop()

        # Create lags
        for feat in [target_col] + climate_cols:
            for lag in range(1, 5):
                merged[f'{feat}_lag{lag}'] = merged[feat].shift(lag)
        lag_cols = [f'{feat}_lag{lag}' for feat in [target_col] + climate_cols for lag in range(1, 5)]
        merged.dropna(subset=lag_cols, inplace=True)

        # Extract years
        merged['year'] = merged['SE'].astype(str).str[:4].astype(int)
        df = merged

        # Parameters (as in original)
        N_obs, N_mosq = 100000, 100000
        init_params = {
            'alpha_h': 1/7, 'gamma_h': 1/7, 'mu_m': 1/10, 'nu_m': 1/10, 'alpha_m': 1/5,
            'beta0': 0.05, 'beta_temp': 0.005, 'beta_precip': 0.0005,
            'ν_vac': 0.0, 'ε_vac': 0.0, 'ν_wol': 0.0, 'μ_ctrl': 0.0, 'trans_wol': 0.1,
            'Tmin': 10.0, 'Tmax': 40.0, 'R0': 50.0, 'k_r': 0.1, 'beta_humid': 0.002
        }

        def briere(T, Tmin, Tmax, c=1e-4):
            return np.maximum(c * T * (T - Tmin) * np.sqrt(np.maximum(Tmax - T, 0)), 0)
        def logistic_rainfall(R, R0, k):
            exp_input = -k * (R - R0)
            exp_input = np.clip(exp_input, -20, 20)
            return 1 / (1 + np.exp(exp_input))

        def seir_sei_control(t, state, params, climate):
            Sh, Eh, Ih, Rh, Vh, Sm, Em, Im, Wm = np.maximum(state, 0)
            Nh = Sh + Eh + Ih + Rh + Vh
            Nh = max(Nh, 1)
            T, R, H = climate['tempmed'], climate['precip_tot_sum'], climate['umidmed']
            beta0 = params.get('beta0_est', params.get('beta0', 0.05))
            beta_temp = params.get('beta_temp_est', params.get('beta_temp', 0.005))
            beta_precip = params.get('beta_precip_est', params.get('beta_precip', 0.0005))
            Tmin = params.get('Tmin_est', params.get('Tmin', 10.0))
            Tmax = params.get('Tmax_est', params.get('Tmax', 40.0))
            R0 = params.get('R0_est', params.get('R0', 50.0))
            k_r = params.get('k_r_est', params.get('k_r', 0.1))
            beta_humid = params.get('beta_humid_est', params.get('beta_humid', 0.002))
            nu_vac = params['ν_vac']
            epsilon_vac = params['ε_vac']
            nu_wol = params['ν_wol']
            mu_ctrl = params['μ_ctrl']
            trans_wol = params['trans_wol']
            alpha_h = params['alpha_h']
            gamma_h = params['gamma_h']
            mu_m = params['mu_m']
            nu_m = params['nu_m']
            alpha_m = params['alpha_m']
            T_br = briere(T, Tmin, Tmax)
            R_eff = logistic_rainfall(R, R0, k_r)
            β = max(0, T_br * R_eff + beta_humid * H)
            λ_h = β * (Im + Wm * trans_wol) / Nh
            λ_m = β * Ih / Nh
            dSh = params['BIRTH_DEATH_RATE'] * N_obs - λ_h * Sh - params['BIRTH_DEATH_RATE'] * Sh - nu_vac * Sh
            dEh = λ_h * Sh - alpha_h * Eh - params['BIRTH_DEATH_RATE'] * Eh
            dIh = alpha_h * Eh - gamma_h * Ih - params['BIRTH_DEATH_RATE'] * Ih
            dRh = gamma_h * Ih - params['BIRTH_DEATH_RATE'] * Rh
            dVh = nu_vac * (Sh + Rh) - epsilon_vac * λ_h * Vh - params['BIRTH_DEATH_RATE'] * Vh
            dSm = nu_m * N_mosq - λ_m * Sm - (mu_m + mu_ctrl + nu_wol) * Sm
            dEm = λ_m * Sm - (alpha_m + mu_m + mu_ctrl) * Em
            dIm = alpha_m * Em - (mu_m + mu_ctrl) * Im
            dWm = nu_wol * (Sm + Em + Im) - (mu_m + mu_ctrl) * Wm
            return [dSh, dEh, dIh, dRh, dVh, dSm, dEm, dIm, dWm]

        init_state = np.array([
            N_obs * 0.99, N_obs * 0.005, N_obs * 0.005, 0, N_obs * 0.0,
            N_mosq * 0.99, N_mosq * 0.005, N_mosq * 0.005, N_mosq * 0.0,
            init_params['Tmin'], init_params['Tmax'], init_params['R0'], init_params['k_r'], init_params['beta_humid']
        ])
        init_P = np.diag([*(N_obs * 0.01,) * 5, *(N_mosq * 0.01,) * 4, 1.0, 1.0, 10.0, 0.1, 0.001])
        Q = np.diag([10] * 9 + [1e-8, 1e-8, 1e-7, 1e-9, 1e-10])
        n_ens = 100

        def hx_base(state):
            return np.array([init_params['gamma_h'] * state[2]])
        def hx_fusion(state):
            return np.array([init_params['gamma_h'] * state[2], 0.])

        def fx(x, dt):
            x = np.atleast_2d(x)
            cl = fx.climate
            pf = fx.pf
            X = np.zeros_like(x)
            for i, s in enumerate(x):
                s = np.asarray(s)
                if not np.isfinite(s).all():
                    s = np.nan_to_num(s, nan=1e-5)
                Tmin_est = np.clip(s[9], 0.0, 30.0)
                Tmax_est = np.clip(s[10], 30.0, 50.0)
                R0_est = np.clip(s[11], 0.0, 200.0)
                k_r_est = np.clip(s[12], 0.0, 1.0)
                beta_humid_est = np.clip(s[13], 0.0, 0.1)
                estimated_params = {
                    'Tmin_est': Tmin_est, 'Tmax_est': Tmax_est, 'R0_est': R0_est, 'k_r_est': k_r_est, 'beta_humid_est': beta_humid_est
                }
                ode_par = {
                    **pf,
                    **estimated_params,
                    'BIRTH_DEATH_RATE': 1 / (70 * 52)
                }
                sol = solve_ivp(seir_sei_control, [0, 7], s[:9], args=(ode_par, cl), t_eval=[7], method='RK45', events=None)
                if sol.status == 0:
                    X[i, :9] = sol.y[:, -1]
                else:
                    X[i, :9] = s[:9]
                X[i, 9] = estimated_params['Tmin_est']
                X[i, 10] = estimated_params['Tmax_est']
                X[i, 11] = estimated_params['R0_est']
                X[i, 12] = estimated_params['k_r_est']
                X[i, 13] = estimated_params['beta_humid_est']
            noise = np.random.multivariate_normal(np.zeros(X.shape[1]), Q, X.shape[0])
            return np.maximum(X + noise, 0)

        # --- Forecast for selected year ---
        yearly_df = df[df['year'] == year].copy()
        if len(yearly_df) < 20:
            st.warning("Not enough data for the selected year.")
            st.stop()
        train = yearly_df.iloc[:8]
        test = yearly_df.iloc[8:]

        X_train = train[lag_cols + climate_cols]
        y_train = train[target_col].shift(-1).dropna()
        X_train = X_train.loc[y_train.index]
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        if not X_train.empty:
            rf.fit(X_train, y_train)
        else:
            rf = None

        filter_init_state = init_state.copy()
        filter_init_P = init_P.copy()
        enkf_base = EnsembleKalmanFilter(x=filter_init_state, P=filter_init_P, dim_z=1, dt=7.0, N=n_ens, fx=lambda x, dt: x, hx=hx_base)
        enkf_base.Q = Q
        enkf_base.R = np.array([[200.]])
        enkf_base.ensembles = filter_init_state.copy() + np.random.multivariate_normal(np.zeros(len(filter_init_state)), filter_init_P, n_ens)

        enkf_fus = EnsembleKalmanFilter(x=filter_init_state.copy(), P=filter_init_P.copy(), dim_z=2, dt=7.0, N=n_ens, fx=lambda x, dt: x, hx=hx_fusion)
        enkf_fus.Q = Q
        enkf_fus.R = np.diag([200., 300.])
        enkf_fus.ensembles = filter_init_state.copy() + np.random.multivariate_normal(np.zeros(len(filter_init_state)), filter_init_P, n_ens)

        forecast = {'true': [], 'base': [], 'fus': [], 'rf': []}

        for i in range(len(test) - 1):
            t_idx = test.index[i]
            fut_idx = test.index[i + 1]
            if t_idx not in df.index:
                continue
            obs = df.loc[t_idx, target_col]
            cl = df.loc[t_idx, climate_cols].to_dict()
            if any(pd.isna(val) for val in cl.values()):
                continue
            fx.climate = cl
            fx.pf = init_params
            enkf_base.fx = fx
            enkf_fus.fx = fx
            enkf_base.predict()
            enkf_base.update(np.array([obs]))
            rf_pred = np.nan
            if rf is not None and t_idx in df.index and not df.loc[[t_idx], lag_cols + climate_cols].isnull().values.any():
                rf_pred = rf.predict(df.loc[[t_idx], lag_cols + climate_cols])[0]
            enkf_fus.predict()
            if not np.isnan(rf_pred):
                enkf_fus.update(np.array([obs, rf_pred]))
            else:
                enkf_fus.update(np.array([obs, 0]), R=np.array([[200., 0.], [0., 1e10]]))
            if fut_idx not in df.index:
                for m in ['base', 'fus', 'rf']:
                    forecast[m].append(np.nan)
                forecast['true'].append(np.nan)
                continue
            true_val = df.loc[fut_idx, target_col]
            forecast['true'].append(true_val)
            future_clim = df.loc[fut_idx, climate_cols].to_dict()
            if any(pd.isna(val) for val in future_clim.values()):
                for m in ['base', 'fus', 'rf']:
                    forecast[m].append(np.nan)
                continue
            forecast_clim = future_clim
            for key, enkf in [('base', enkf_base), ('fus', enkf_fus)]:
                s = enkf.x.copy()
                estimated_params = {
                    'Tmin_est': s[9], 'Tmax_est': s[10], 'R0_est': s[11], 'k_r_est': s[12], 'beta_humid_est': s[13]
                }
                ode_par = {
                    **init_params,
                    **estimated_params,
                    'BIRTH_DEATH_RATE': 1 / (70 * 52)
                }
                sol = solve_ivp(seir_sei_control, [0, 7], s[:9], args=(ode_par, forecast_clim), t_eval=[7], method='RK45', events=None)
                pred = init_params['gamma_h'] * sol.y[2, -1] if sol.status == 0 else np.nan
                forecast[key].append(max(pred, 0) if not np.isnan(pred) else np.nan)
            rf_pred_next = np.nan
            if rf is not None and t_idx in df.index and not df.loc[[t_idx], lag_cols + climate_cols].isnull().values.any():
                rf_pred_next = rf.predict(df.loc[[t_idx], lag_cols + climate_cols])[0]
            forecast['rf'].append(max(rf_pred_next, 0) if not np.isnan(rf_pred_next) else np.nan)

        # --- Plotting ---
        st.subheader(f"Predicted vs Observed Cases for {year} (geocode: {geocode})")
        plot_len = min(len(test.index) - 1, len(forecast['true']))
        if plot_len > 0:
            plot_indices = test.index[1:plot_len+1]
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(plot_indices, forecast['true'][:plot_len], label='True Cases', marker='o')
            for m in ['base', 'fus', 'rf']:
                data_to_plot = np.array(forecast[m][:plot_len])
                ax.plot(plot_indices[~np.isnan(data_to_plot)], data_to_plot[~np.isnan(data_to_plot)], label=f'{m.upper()}', linestyle='--')
            ax.set_title(f'Dengue Case Forecast for {year}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Estimated Cases')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.warning("Not enough data to plot for the selected year.")
    except Exception as e:
        st.error(f"Error during modeling/forecasting: {e}")
