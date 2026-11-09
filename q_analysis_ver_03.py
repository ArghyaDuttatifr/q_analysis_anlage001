import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from functools import partial
import os
import io
import concurrent.futures
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as pcolors

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Q-Factor Analysis (Real Lorentzian)", layout="wide", page_icon="⚡")
st.title("⚡ Resonator Q-Factor & Background Analysis")

# ==========================================
# SIDEBAR: CONTROLS & INPUTS
# ==========================================
st.sidebar.header("1. Data Upload")

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

uploaded_files = st.sidebar.file_uploader(
    "Upload S12 Sweep Files (.txt, .csv)", 
    accept_multiple_files=True, 
    type=['txt', 'csv'],
    key=f"uploader_{st.session_state['uploader_key']}"
)

if st.sidebar.button("🗑️ Reset / Clear All Files", use_container_width=True):
    st.session_state["uploader_key"] += 1
    for key in ['master_results', 'individual_plots_data', 'bg_combined', 'failed_files']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

st.sidebar.header("2. Filtering & Options")
FILE_PREFIX = st.sidebar.text_input("File Prefix Filter", value="S12")
col1, col2 = st.sidebar.columns(2)

T_min = col1.number_input("T min (K)", value=2.0, step=0.1)
T_max = col2.number_input("T max (K)", value=15.0, step=0.1)

bg_correction = st.sidebar.checkbox("Enable Background Subtraction", value=False, 
    help="Uses the previous temperature's fitted background before fitting the next one.")

restrict_rows = st.sidebar.checkbox("Restrict Row Range")
if restrict_rows:
    row_col1, row_col2 = st.sidebar.columns(2)
    start_row = row_col1.number_input("Start Row (1-indexed)", value=1, min_value=1)
    end_row = row_col2.number_input("End Row", value=1600, min_value=1)
    row_range = (int(start_row), int(end_row))
else:
    row_range = None

with st.sidebar.expander("⚙️ Advanced Fit Settings"):
    RELAXATION_RANGE = st.number_input("Relaxation Range (GHz)", value=0.001, format="%.4f") * 1e9
    N_TRIALS = st.slider("Number of Trials (Lower = Faster)", min_value=10, max_value=100, value=25)
    USE_SMOOTHED_REFERENCE = st.checkbox("Use Smoothed Reference (for noisy data)", value=False)
    
    st.markdown("**Parameter Bounds [A1, A2, A3, del_f (GHz)]**")
    st.caption("A1, A2: -0.5 to 0.5 | A3: -5.0 to 5.0 | del_f: 1e-5 to 0.1")
    
    b_min = st.text_input("Lower Bounds (A1, A2, A3, del_f)", "-0.5, -0.5, -5.0, 0.00001")
    b_max = st.text_input("Upper Bounds (A1, A2, A3, del_f)", "0.5, 0.5, 5.0, 0.1")
    
    try:
        lower_bounds = [float(x.strip()) for x in b_min.split(',')]
        upper_bounds = [float(x.strip()) for x in b_max.split(',')]
        
        if len(lower_bounds) != 4 or len(upper_bounds) != 4:
            raise ValueError("Strict 4-parameter bounds required.")
            
        BOUNDS = (lower_bounds, upper_bounds)
    except:
        st.warning("⚠️ Invalid bounds format detected. Reverting to default 4 parameters.")
        BOUNDS = ([-0.5, -0.5, -5.0, 1e-5], [0.5, 0.5, 5.0, 0.1])


# ==========================================
# MATH & FITTING MODULES
# ==========================================
def linmag(freq, A1, A2, A3, del_f, omega_r, Sm):
    # Scale frequencies to GHz to match del_f bounds (1e-5 to 0.1)
    f_ghz = freq / 1e9
    omega_ghz = omega_r / 1e9
    delta_ghz = f_ghz - omega_ghz
    
    background = A1 + A2 * f_ghz
    # Formula uses delta_ghz and del_f directly (both in GHz)
    lorentzian = (Sm + A3 * f_ghz) * (1.0 / (1.0 + 4.0 * (delta_ghz / del_f)**2))
    return lorentzian + background

def chi_square(y_obs, y_fit, n_params=4):
    degrees_of_freedom = max(1, len(y_obs) - n_params)
    return np.sum((y_obs - y_fit) ** 2) / degrees_of_freedom

def interpolate_background(freq_prev, background_prev, freq_curr, edge_points=50):
    interpolator = interp1d(freq_prev, background_prev, kind='linear', bounds_error=False, fill_value="extrapolate")
    return interpolator(freq_curr)

def fit_resonance_core(freq, real_mag, real_mag_s, n_trials, relaxation_range, use_smoothed, bounds):
    reference = real_mag_s if use_smoothed else real_mag
    max_index = np.argmax(reference)
    omega_r_initial = freq[max_index]
    omega_r_trials = np.linspace(omega_r_initial - relaxation_range, omega_r_initial + relaxation_range, n_trials)

    best = {'chi2': 1e9}  
    median_val = float(np.median(reference))
    
    Sm_fixed = float(np.max(reference) - median_val)
    if Sm_fixed <= 0:
        Sm_fixed = 1e-6
        
    a1_guess = float(np.clip(median_val, bounds[0][0] + 1e-4, bounds[1][0] - 1e-4))
    # Initial guess for del_f (GHz): 0.005 = 5 MHz linewidth
    p0 = [a1_guess, 0.0, 0.0, 0.005]  
    
    for omega_r in omega_r_trials:
        model = partial(linmag, omega_r=omega_r, Sm=Sm_fixed)
        try:
            popt, pcov = curve_fit(model, freq, real_mag, p0=p0, bounds=bounds, maxfev=2500)
        except (RuntimeError, ValueError):
            continue
        
        p0 = list(popt)  
        fit_values = model(freq, *popt)
        chi2 = chi_square(reference, fit_values)

        if chi2 < best['chi2']:
            best.update(
                chi2=chi2, popt=popt, pcov=pcov, fit_values=fit_values,
                omega_r=omega_r, omega_r_max=freq[np.argmax(fit_values)],
                Sm=Sm_fixed,
                median_baseline=median_val
            )

    if 'popt' not in best:
        return None
    best['omega_r_initial'] = omega_r_initial
    return best

def process_single_task(args):
    try:
        T, filename, freq_full, real_mag_raw, row_range, bg_in, n_trials, relaxation_range, use_smoothed, bounds = args
        
        corrected_full = real_mag_raw - bg_in
        
        if row_range is not None:
            start_idx = max(0, row_range[0] - 1)
            end_idx = min(len(freq_full), row_range[1])
        else:
            start_idx, end_idx = 0, len(freq_full)
            
        freq_fit = freq_full[start_idx:end_idx]
        corrected_fit = corrected_full[start_idx:end_idx]
        
        if len(freq_fit) < 10:
            return None 
            
        n_fit = len(corrected_fit)
        window_length = min(9, n_fit if n_fit % 2 == 1 else n_fit - 1)
        corrected_fit_s = (corrected_fit if window_length < 5 else savgol_filter(corrected_fit, window_length=window_length, polyorder=3))
        
        result = fit_resonance_core(freq_fit, corrected_fit, corrected_fit_s, n_trials, relaxation_range, use_smoothed, bounds)
        
        if result is None:
            return None
            
        A1, A2, A3, del_f = result['popt']
        Sm = result['Sm']
        median_baseline = result['median_baseline']
        Sm_err = 0.0 
        
        pcov_diag = np.sqrt(np.maximum(0, np.diag(result['pcov'])))
        A1_err, A2_err, A3_err, del_f_err = pcov_diag
        
        f0_ghz = result['omega_r_max'] / 1e9
        
        # Calculate QL from f0 / del_f (both natively in GHz)
        QL = f0_ghz / max(del_f, 1e-9)
        QL_err = QL * (del_f_err / max(del_f, 1e-9))
        
        # Derived Q values
        Qex = QL / max(Sm, 1e-6)
        Qex_err = Qex * (QL_err / max(QL, 1e-6))
        
        Q_in_denom = (1.0 / max(QL, 1e-6)) - (1.0 / max(Qex, 1e-6))
        if Q_in_denom > 1e-12:
            Q_in = 1.0 / Q_in_denom
            Q_in_err = (Q_in**2) * np.sqrt((QL_err / max(QL**2, 1e-10))**2 + (Qex_err / max(Qex**2, 1e-10))**2)
        else:
            Q_in = np.nan
            Q_in_err = np.nan

        model_values_full = linmag(freq_full, A1, A2, A3, del_f, result['omega_r'], Sm)
        background_local_full = corrected_full - model_values_full + (A1 + A2 * (freq_full / 1e9))
        background_out = background_local_full + bg_in
        
        metrics = {
            "T (K)": T,
            "f0 (GHz)": f0_ghz,
            "del_f (GHz)": del_f,
            "Sm": Sm, "Sm_err": Sm_err,
            "Median_Baseline": median_baseline, 
            "Q_in": Q_in, "Q_in_err": Q_in_err,
            "Q_ex": Qex, "Q_ex_err": Qex_err,
            "Q_L": QL, "Q_L_err": QL_err,
            "A1": A1, "A1_err": A1_err,
            "A2": A2, "A2_err": A2_err,
            "A3": A3, "A3_err": A3_err,
            "chi_square": result['chi2'],
            "File": filename
        }
        
        plot_data = {
            "freq_fit": freq_fit,
            "corrected_fit": corrected_fit,
            "fit_values": result['fit_values'],
            "freq_full": freq_full[start_idx:end_idx],
            "bg_out": background_out[start_idx:end_idx]
        }
        
        bg_data = np.column_stack([np.full_like(freq_full[start_idx:end_idx], T), freq_full[start_idx:end_idx], background_out[start_idx:end_idx]])
        
        return metrics, plot_data, bg_data, freq_full, background_out
    
    except Exception as e:
        return args[1] 


# ==========================================
# WEBGL PLOTLY FUNCTIONS
# ==========================================
def create_plotly_fit(T, freq_fit, corrected_fit, fit_values):
    freq_ghz = freq_fit / 1e9
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=freq_ghz, y=corrected_fit, mode='markers', name='Experiment (Corrected)', 
                               marker=dict(color='#1f77b4', size=5, opacity=0.6)))
    fig.add_trace(go.Scattergl(x=freq_ghz, y=fit_values, mode='lines', name='Real Lorentzian Fit', 
                               line=dict(color='#d62728', width=2.5)))
    fig.update_layout(title=f"Fitting for T = {T} K", xaxis_title="Frequency (GHz)", yaxis_title="Magnitude",
                      hovermode="x unified", template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_plotly_bg(T, freq, background):
    freq_ghz = freq / 1e9
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=freq_ghz, y=background, mode='lines', name='Background', 
                               line=dict(color='#9467bd', width=2)))
    fig.update_layout(title=f"Background for T = {T} K", xaxis_title="Frequency (GHz)", yaxis_title="Background Magnitude",
                      hovermode="x unified", template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    return fig

@st.fragment
def render_individual_scans_fragment(df_res, individual_plots_data):
    st.markdown("Select a specific temperature from the dropdown below to view its interactive fit and background.")
    selected_T = st.selectbox("Select Temperature (K)", options=df_res['T (K)'].unique(), key="scan_select_T")
    
    if selected_T in individual_plots_data:
        plot_data = individual_plots_data[selected_T]
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            fig_fit = create_plotly_fit(selected_T, plot_data["freq_fit"], plot_data["corrected_fit"], plot_data["fit_values"])
            st.plotly_chart(fig_fit, use_container_width=True)
        with col_p2:
            fig_bg = create_plotly_bg(selected_T, plot_data["freq_full"], plot_data["bg_out"])
            st.plotly_chart(fig_bg, use_container_width=True)


# ==========================================
# MAIN EXECUTION
# ==========================================
if uploaded_files:
    entries = []
    
    for file in uploaded_files:
        if not file.name.startswith(FILE_PREFIX): continue
        if "_T_" not in file.name: continue
            
        T_str = os.path.splitext(file.name.split("_T_")[1])[0].replace("_", ".")
        try:
            T = float(T_str)
            if T_min <= T <= T_max:
                entries.append((T, file))
        except ValueError:
            continue

    entries.sort(key=lambda e: e[0])
    
    if not entries:
        st.warning(f"No matching files found in T range [{T_min}, {T_max}] with prefix '{FILE_PREFIX}'.")
    else:
        st.success(f"Found {len(entries)} matching sweep dataset(s) ready for analysis.")
        
        if st.button("🚀 Run Analysis", use_container_width=True, type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            master_results = []
            bg_vs_temp_data = []
            individual_plots_data = {}
            failed_files = []
            
            status_text.text("Pre-loading and cleaning sweep files...")
            
            preloaded_tasks = []
            for T, file in entries:
                file.seek(0)
                content = file.read()
                df_raw = pd.read_csv(io.BytesIO(content), skiprows=1, header=None)
                df_raw = df_raw.dropna(subset=[0, 3]) 
                
                freq_full = df_raw.iloc[:, 0].to_numpy(dtype=np.float64)
                real_mag_raw = df_raw.iloc[:, 3].to_numpy(dtype=np.float64)
                
                if len(freq_full) > 0:
                    preloaded_tasks.append((T, file.name, freq_full, real_mag_raw))
            
            status_text.text(f"Fitting {len(preloaded_tasks)} files using robust model across CPU cores...")
            
            if not bg_correction:
                task_args = [
                    (T, fname, freq_full, real_mag_raw, row_range, np.zeros_like(freq_full), N_TRIALS, RELAXATION_RANGE, USE_SMOOTHED_REFERENCE, BOUNDS)
                    for (T, fname, freq_full, real_mag_raw) in preloaded_tasks
                ]
                
                completed = 0
                total_tasks = len(task_args)
                
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    futures = [executor.submit(process_single_task, arg) for arg in task_args]
                    
                    for future in concurrent.futures.as_completed(futures):
                        res = future.result()
                        if isinstance(res, str):
                            failed_files.append(res)
                        elif res is not None:
                            metrics, plot_data, bg_data, _, _ = res
                            master_results.append(metrics)
                            individual_plots_data[metrics["T (K)"]] = plot_data
                            bg_vs_temp_data.append(bg_data)
                        
                        completed += 1
                        progress_bar.progress(completed / total_tasks)
            else:
                freq_prev, background_prev = None, None
                for idx, (T, fname, freq_full, real_mag_raw) in enumerate(preloaded_tasks):
                    status_text.text(f"Chaining fit for T = {T} K ({idx+1}/{len(preloaded_tasks)})")
                    
                    if background_prev is not None:
                        bg_in = interpolate_background(freq_prev, background_prev, freq_full)
                    else:
                        bg_in = np.zeros_like(freq_full)
                        
                    task_arg = (T, fname, freq_full, real_mag_raw, row_range, bg_in, N_TRIALS, RELAXATION_RANGE, USE_SMOOTHED_REFERENCE, BOUNDS)
                    res = process_single_task(task_arg)
                    
                    if isinstance(res, str):
                        failed_files.append(res)
                    elif res is not None:
                        metrics, plot_data, bg_data, f_full, bg_out = res
                        master_results.append(metrics)
                        individual_plots_data[T] = plot_data
                        bg_vs_temp_data.append(bg_data)
                        freq_prev, background_prev = f_full, bg_out
                    
                    progress_bar.progress((idx + 1) / len(preloaded_tasks))

            st.session_state['master_results'] = pd.DataFrame(master_results).sort_values(by="T (K)") if master_results else pd.DataFrame()
            st.session_state['individual_plots_data'] = individual_plots_data
            st.session_state['bg_combined'] = pd.DataFrame(np.vstack(bg_vs_temp_data), columns=['T', 'freq_Hz', 'background']) if bg_vs_temp_data else pd.DataFrame()
            st.session_state['failed_files'] = failed_files
            
            status_text.empty()
            progress_bar.empty()
            st.success("✅ Analysis Complete!")

        # ==========================================
        # DASHBOARD VIEWS
        # ==========================================
        if 'failed_files' in st.session_state and len(st.session_state['failed_files']) > 0:
            st.error(f"⚠️ Warning: {len(st.session_state['failed_files'])} file(s) failed to fit and were safely skipped (data might be too noisy or hit boundary limits).")

        if 'master_results' in st.session_state and not st.session_state['master_results'].empty:
            df_res = st.session_state['master_results']
            individual_plots_data = st.session_state.get('individual_plots_data', {})
            
            tab_trends, tab_individual, tab_data = st.tabs(["📈 Global Trends", "🔍 Individual Scans", "📊 Data Table & Export"])
            
            with tab_trends:
                st.subheader("Extracted Q-Factors vs Temperature (Log Scale)")
                fig_q_log = go.Figure()
                fig_q_log.add_trace(go.Scattergl(x=df_res['T (K)'], y=df_res['Q_in'], mode='lines+markers', name='Internal Q (Q_in)', line=dict(color='#2ca02c', width=2)))
                fig_q_log.add_trace(go.Scattergl(x=df_res['T (K)'], y=df_res['Q_L'], mode='lines+markers', name='Loaded Q (Q_L)', line=dict(color='#ff7f0e', width=2)))
                fig_q_log.add_trace(go.Scattergl(x=df_res['T (K)'], y=df_res['Q_ex'], mode='lines+markers', name='External Q (Q_ex)', line=dict(color='#1f77b4', width=2)))
                fig_q_log.update_layout(xaxis_title="Temperature (K)", yaxis_title="Q-Factor (Log Scale)", yaxis_type="log", hovermode="x unified", template="plotly_white")
                st.plotly_chart(fig_q_log, use_container_width=True)
                
                st.subheader("Internal Q-Factor (Q_in) with Error Bars (Linear Scale)")
                fig_qin_err = go.Figure()
                fig_qin_err.add_trace(go.Scattergl(
                    x=df_res['T (K)'], y=df_res['Q_in'], mode='lines+markers', name='Q_in', line=dict(color='#2ca02c', width=2),
                    error_y=dict(type='data', array=df_res['Q_in_err'], visible=True, color='#2ca02c', thickness=1.5, width=4)
                ))
                fig_qin_err.update_layout(xaxis_title="Temperature (K)", yaxis_title="Internal Q-Factor (Q_in)", hovermode="x unified", template="plotly_white")
                st.plotly_chart(fig_qin_err, use_container_width=True)

                st.subheader("Resonance Frequency (f0) vs Temperature")
                fig_f = px.line(df_res, x='T (K)', y='f0 (GHz)', markers=True, color_discrete_sequence=['#d62728'], render_mode='webgl')
                fig_f.update_layout(xaxis_title="Temperature (K)", yaxis_title="Resonance Frequency (GHz)", hovermode="x unified", template="plotly_white")
                st.plotly_chart(fig_f, use_container_width=True)

                st.subheader("Fitted Backgrounds vs Frequency (All Temperatures)")
                fig_bg_all = go.Figure()
                temps_sorted = sorted(individual_plots_data.keys())
                num_temps = len(temps_sorted)
                
                if num_temps > 0:
                    norm_indices = np.linspace(0, 1, num_temps) if num_temps > 1 else [0.5]
                    colors = pcolors.sample_colorscale('Viridis', norm_indices)
                    
                    for idx, T_val in enumerate(temps_sorted):
                        pdata = individual_plots_data[T_val]
                        fig_bg_all.add_trace(go.Scattergl(
                            x=pdata["freq_full"] / 1e9, y=pdata["bg_out"], mode='lines', name=f"{T_val:.3f} K", line=dict(color=colors[idx], width=1.2)
                        ))
                    fig_bg_all.update_layout(xaxis_title="Frequency (GHz)", yaxis_title="Background Magnitude", hovermode="x unified", template="plotly_white", legend_title="Temperature (K)")
                    st.plotly_chart(fig_bg_all, use_container_width=True)

            with tab_individual:
                render_individual_scans_fragment(df_res, individual_plots_data)

            with tab_data:
                st.dataframe(df_res.style.format({
                    "f0 (GHz)": "{:.6f}", 
                    "del_f (GHz)": "{:.6f}", 
                    "Q_in": "{:.1f}", 
                    "Q_L": "{:.1f}", 
                    "Q_ex": "{:.1f}", 
                    "Sm": "{:.4f}",
                    "Median_Baseline": "{:.4f}"
                }), use_container_width=True)
                
                col_d1, col_d2 = st.columns(2)
                col_d1.download_button(
                    label="📥 Download Q-Factor Results (CSV)",
                    data=df_res.to_csv(index=False),
                    file_name="q_factor_results.csv",
                    mime="text/csv"
                )
                if 'bg_combined' in st.session_state and not st.session_state['bg_combined'].empty:
                    col_d2.download_button(
                        label="📥 Download Background vs Temp Data (CSV)",
                        data=st.session_state['bg_combined'].to_csv(index=False),
                        file_name="background_vs_temperature.csv",
                        mime="text/csv"
                    )
