import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from itertools import product

FU_PLASMA, FU_ISF, FU_ICF = 0.58, 0.72, 0.10
KM_CELL = 3.6
KM_RENAL_SEC, KM_RENAL_REAB = 0.058, 0.691
VMAX_RENAL_SEC, VMAX_RENAL_REAB = 0.348, 0.138

def get_infusion_schedule(patient_df):
    infusions = []
    first_dose_time = None

    for _, row in patient_df.iterrows():
        if pd.notna(row['dosing rate']) and pd.notna(row['dosing time']):
            if pd.notna(row['Date']) and pd.notna(row['Time']):
                try:
                    start_dt = pd.to_datetime(f"{row['Date']} {row['Time']}")
                except Exception:
                    continue
            else:
                continue

            if first_dose_time is None:
                first_dose_time = start_dt
                start_hours = 0
            else:
                delta = (start_dt - first_dose_time).total_seconds() / 3600.0
                start_hours = delta

            duration = float(row['dosing time'])
            rate = float(row['dosing rate'])

            infusions.append((start_hours, duration, rate))

    return infusions, first_dose_time

def create_infusion_function(infusions):
    def infusion_rate(t):
        return sum(rate for start, dur, rate in infusions if start <= t <= start + dur)
    return infusion_rate

def pbpk_ode(t, m, pat, infusion_fn, k1, k2, vmax_in, vmax_out):
    mCS, mRCS, mHCS, mISF, mICF = m
    VCS, VRCS, VHCS, VISF, VICF = pat.VCS, pat.VRCS, pat.VHCS, pat.VISF, pat.VICF
    cCS, cRCS, cHCS, cISF, cICF = mCS/VCS, mRCS/VRCS, mHCS/VHCS, mISF/VISF, mICF/VICF
    cu_CS, cu_RCS, cu_HCS = cCS*FU_PLASMA, cRCS*FU_PLASMA, cHCS*FU_PLASMA
    cu_ISF, cu_ICF = cISF*FU_ISF, cICF*FU_ICF

    J_CS_ISF = k1 * pat.VISF * cu_CS - k2 * pat.VISF * cu_ISF
    J_ISF_ICF = (vmax_in * pat.VICF * cu_ISF)/(KM_CELL+cu_ISF) if cu_ISF > 0 else 0
    J_ICF_ISF = (vmax_out * pat.VICF* cu_ICF)/(KM_CELL+cu_ICF) if cu_ICF > 0 else 0

    filtration = pat.GFR * cu_RCS
    secretion = VMAX_RENAL_SEC * cu_RCS / (KM_RENAL_SEC + cu_RCS)
    reabsorption = VMAX_RENAL_REAB * cu_RCS / (KM_RENAL_REAB + cu_RCS)
    renal_clearance = filtration + secretion - reabsorption
    hepatic_clearance = pat.Q_HEP_CLEAR * cu_HCS

    dCSdt = (infusion_fn(t) - pat.Q_RCS*cCS + pat.Q_RCS*cRCS
             - pat.Q_HCS*cCS + pat.Q_HCS*cHCS - J_CS_ISF)
    dRCSdt = pat.Q_RCS*cCS - pat.Q_RCS*cRCS - renal_clearance
    dHCSdt = pat.Q_HCS*cCS - pat.Q_HCS*cHCS - hepatic_clearance
    dISFdt = J_CS_ISF - J_ISF_ICF + J_ICF_ISF
    dICFdt = J_ISF_ICF - J_ICF_ISF

    return [dCSdt, dRCSdt, dHCSdt, dISFdt, dICFdt]

def simulate_patient_concentration(pat, infusion_schedule, t_obs, k1, k2, vmax_in, vmax_out):
    if len(infusion_schedule) == 0:
        raise ValueError("No infusion schedule provided.")

    _, VCS, VRCS, VHCS = pat.compute_circulatory_volumes()
    _, _, _, VISF = pat.compute_ecf_icf()
    VICF = pat.compute_ecf_icf()[2]

    if VISF <= 0 or VICF <= 0:
        raise ValueError("Invalid ISF or ICF volume")

    pat.VCS, pat.VRCS, pat.VHCS, pat.VISF, pat.VICF = VCS/1000, VRCS/1000, VHCS/1000, VISF, VICF
    pat.Q_RCS = pat.compute_renal_plasma_flow()
    pat.Q_HCS = pat.compute_hepatic_plasma_flow()
    pat.GFR = pat.compute_gfr_ckd_epi() * 1.73 / pat.compute_bsa() * 0.06
    pat.Q_HEP_CLEAR = 0.04 * pat.GFR

    infusion_fn = create_infusion_function(infusion_schedule)
    last_infusion_time = max(start + dur for start, dur, _ in infusion_schedule)
    last_obs_time = max(t_obs) if len(t_obs) > 0 else 0
    sim_end = max(last_infusion_time, last_obs_time) + 12  # buffer of 12h for safety
    t_eval = np.linspace(0, sim_end, int(sim_end * 20))
    m0 = [0, 0, 0, 0, 0]
    sol = solve_ivp(pbpk_ode, [0, sim_end], m0, t_eval=t_eval,
                    args=(pat, infusion_fn, k1, k2, vmax_in, vmax_out),
                    method='Radau')

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    C_CS = sol.y[0] / pat.VCS
    if np.any(np.isnan(C_CS)) or np.any(np.isinf(C_CS)):
        raise ValueError("Simulated plasma concentrations contain NaN or Inf")

    if t_obs.size == 0:
        raise ValueError("No valid observation times (t_obs) parsed")
    
    #print(f"→ Observation times (t_obs): min={t_obs.min()}, max={t_obs.max()}")
    #print(f"→ Simulated time span: {sol.t[0]} to {sol.t[-1]}")
    #print(f"→ Any NaNs in t_obs? {np.any(np.isnan(t_obs))}")
    interp_fn = interp1d(sol.t, C_CS, bounds_error=False, fill_value="extrapolate")
    C_pred = interp_fn(t_obs)

    if np.any(np.isnan(C_pred)) or np.any(np.isinf(C_pred)):
        raise ValueError("Interpolated concentration contains NaN or Inf")

    return C_pred, sol.t, C_CS

def log_mse(c_exp, c_sim):
    c_exp = np.clip(c_exp, 1e-6, None)
    c_sim = np.clip(c_sim, 1e-6, None)
    return np.mean((np.log(c_exp) - np.log(c_sim))**2)

def evaluate_group_objfun(df, patient_ids, params, patient_cls):
    k1, k2, vmax_in, vmax_out = params
    errors = []
    failed_patients = []

    for pid in patient_ids:
        patient_df = df[df["ID"] == pid]
        if pid == "658.3":
            #print(f"\n🔍 Debugging Date+Time parsing for Patient {pid}...")
            for i, row in patient_df.iterrows():
                raw_date = str(row["Date"]).strip()
                raw_time = str(row["Time"]).strip()
                raw_combined = f"{raw_date} {raw_time}"
                try:
                    parsed = pd.to_datetime(raw_combined)
                    #print(f"✅ Parsed: {raw_combined} → {parsed}")
                except Exception as e:
                    print(f"❌ Failed: {raw_combined} → {e}")
        if patient_df.empty or patient_df["Concentration"].isna().all():
            failed_patients.append((pid, "No data or all concentrations missing"))
            continue

        row = patient_df.iloc[0]
        try:
            serum_creatinine = float(row['serum creatinine']) * 0.0113
            hematocrit = float(row['hematocrit']) / 100.0
            patient = patient_cls(
                row['Gender: 1, male; 2, female'],
                row['Body weight'], row['Height'], row['Age'],
                hematocrit, serum_creatinine
            )

            infusions, start_time = get_infusion_schedule(patient_df)
            if not infusions or start_time is None:
                failed_patients.append((pid, "No valid infusion schedule"))
                continue
            
            obs_df = patient_df.dropna(subset=["Concentration", "Date", "Time"]).copy()

            # Force both columns to clean strings
            date_str = obs_df["Date"].astype(str).str.strip()
            time_str = obs_df["Time"].astype(str).str.strip()

            # Combine and parse, allowing flexible datetime formats
            obs_df["dt"] = pd.to_datetime(date_str + " " + time_str, errors="coerce")
            obs_df = obs_df.dropna(subset=["dt"])
            #print(f"Start time for patient {pid}: {start_time}")

            #print(f"Start time type: {type(start_time)}")
            #print("First few dt values:", obs_df["dt"].head())
            obs_df["t_hr"] = (obs_df["dt"] - start_time).dt.total_seconds() / 3600.0
            t_obs = obs_df["t_hr"].values
            c_exp = obs_df["Concentration"].values

            c_sim, _, _ = simulate_patient_concentration(patient, infusions, t_obs, k1, k2, vmax_in, vmax_out)
            error = log_mse(c_exp, c_sim)
            errors.append(error)

        except Exception as e:
            print(f"[Patient {pid}] Skipped due to error: {e}")
            failed_patients.append((pid, str(e)))
            continue

    if len(errors) == 0:
        print(" No patients successfully simulated in this group.")

    return (np.mean(errors) if errors else np.nan), failed_patients

def evaluate_group_objfun_global(df, patient_ids, params, patient_cls):
    """
    Global MSLE across ALL time points from ALL patients (each measurement has equal weight).
    Returns (objective_value, failed_patients_list).

    Same inputs/outputs as evaluate_group_objfun, but the loss is computed on the
    concatenated arrays instead of mean-of-patient-means.
    """
    import numpy as np
    import pandas as pd

    k1, k2, vmax_in, vmax_out = params
    failed_patients = []

    all_obs = []   # list of arrays of observed conc (µmol/L)
    all_pred = []  # list of arrays of predicted conc (µmol/L)

    for pid in patient_ids:
        try:
            patient_df = df[df["ID"] == pid]
            if patient_df.empty or patient_df["Concentration"].isna().all():
                failed_patients.append((pid, "No data or all concentrations missing"))
                continue

            row = patient_df.iloc[0]

            # Serum creatinine: if your dataset is µmol/L, convert to mg/dL for CKD-EPI
            try:
                scr_umol_per_L = float(row['serum creatinine'])
                serum_creatinine_mgdl = scr_umol_per_L * 0.011312
            except Exception:
                serum_creatinine_mgdl = float(row['serum creatinine'])  # fallback

            hematocrit = float(row['hematocrit']) / 100.0
            patient = patient_cls(
                row['Gender: 1, male; 2, female'],
                row['Body weight'], row['Height'], row['Age'],
                hematocrit, serum_creatinine_mgdl
            )

            # Infusions & start time
            infusions, start_time = get_infusion_schedule(patient_df)
            if not infusions or start_time is None:
                failed_patients.append((pid, "No valid infusion schedule"))
                continue

            # Observation times
            obs_df = patient_df.dropna(subset=["Concentration", "Date", "Time"]).copy()
            if obs_df.empty:
                failed_patients.append((pid, "No valid observations with Date & Time"))
                continue

            # Parse datetimes robustly
            date_str = obs_df["Date"].astype(str).str.strip()
            time_str = obs_df["Time"].astype(str).str.strip()
            obs_df["dt"] = pd.to_datetime(date_str + " " + time_str, errors="coerce")
            obs_df = obs_df.dropna(subset=["dt"])
            if obs_df.empty:
                failed_patients.append((pid, "Datetime parsing failed"))
                continue

            # Numeric concentrations only
            obs_df["Concentration"] = pd.to_numeric(obs_df["Concentration"], errors="coerce")
            obs_df = obs_df.dropna(subset=["Concentration"])
            if obs_df.empty:
                failed_patients.append((pid, "No valid numeric concentrations"))
                continue

            # Build relative hours and simulate
            obs_df["t_hr"] = (obs_df["dt"] - start_time).dt.total_seconds() / 3600.0
            t_obs = obs_df["t_hr"].to_numpy()
            c_obs = obs_df["Concentration"].to_numpy()

            c_sim, _, _ = simulate_patient_concentration(
                patient, infusions, t_obs, k1, k2, vmax_in, vmax_out
            )

            # Collect for global loss
            all_obs.append(c_obs.astype(float))
            all_pred.append(np.asarray(c_sim, dtype=float))

        except Exception as e:
            failed_patients.append((pid, f"Error: {e}"))
            continue

    if not all_obs:
        # No successful patients
        return np.nan, failed_patients

    # Concatenate ALL points and compute global MSLE
    obs = np.concatenate(all_obs)
    pred = np.concatenate(all_pred)

    # Clip to avoid log(0)
    obs = np.clip(obs, 1e-6, None)
    pred = np.clip(pred, 1e-6, None)

    objfun = float(np.mean((np.log(obs) - np.log(pred))**2))
    return objfun, failed_patients

def log_msle_components(c_exp, c_sim):
    """
    Returns (SSLE, N) i.e., sum of squared log-errors and number of points.
    This yields a global MSLE when aggregated over all patients: SSLE_total / N_total.
    """
    c_exp = np.clip(c_exp, 1e-6, None)
    c_sim = np.clip(c_sim, 1e-6, None)
    diff = np.log(c_exp) - np.log(c_sim)
    ssle = np.sum(diff**2)
    n = diff.size
    return ssle, n

def evaluate_group_objfun_2(df, patient_ids, params, patient_cls):
    """
    Computes the *global* mean squared log error (MSLE) across all observations
    in the group (i.e., weights patients by their number of samples).
    Returns (global_msle, failed_patients).
    """
    k1, k2, vmax_in, vmax_out = params
    total_ssle = 0.0
    total_n = 0
    failed_patients = []

    for pid in patient_ids:
        patient_df = df[df["ID"] == pid]
        if patient_df.empty or patient_df["Concentration"].isna().all():
            failed_patients.append((pid, "No data or all concentrations missing"))
            continue

        row = patient_df.iloc[0]
        try:
            serum_creatinine = float(row['serum creatinine']) * 0.011312
            hematocrit = float(row['hematocrit']) / 100.0
            patient = patient_cls(
                row['Gender: 1, male; 2, female'],
                row['Body weight'], row['Height'], row['Age'],
                hematocrit, serum_creatinine
            )

            infusions, start_time = get_infusion_schedule(patient_df)
            if not infusions or start_time is None:
                failed_patients.append((pid, "No valid infusion schedule"))
                continue

            # Observations with valid time + concentration
            obs_df = patient_df.dropna(subset=["Concentration", "Date", "Time"]).copy()
            date_str = obs_df["Date"].astype(str).str.strip()
            time_str = obs_df["Time"].astype(str).str.strip()
            obs_df["dt"] = pd.to_datetime(date_str + " " + time_str, errors="coerce")
            obs_df = obs_df.dropna(subset=["dt"])

            # If no valid observations remain, mark failed and continue
            if obs_df.empty:
                failed_patients.append((pid, "No valid observation timestamps"))
                continue

            obs_df["t_hr"] = (obs_df["dt"] - start_time).dt.total_seconds() / 3600.0
            t_obs = obs_df["t_hr"].values
            c_exp = obs_df["Concentration"].values

            c_sim, _, _ = simulate_patient_concentration(
                patient, infusions, t_obs, k1, k2, vmax_in, vmax_out
            )

            # Accumulate global SSLE and N
            ssle, n = log_msle_components(c_exp, c_sim)
            total_ssle += ssle
            total_n += n

        except Exception as e:
            print(f"[Patient {pid}] Skipped due to error: {e}")
            failed_patients.append((pid, str(e)))
            continue

    if total_n == 0:
        print(" No patients successfully simulated in this group.")
        return np.nan, failed_patients

    global_msle = total_ssle / total_n
    return global_msle, failed_patients

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# --- keep your existing simulate_patient_concentration, get_infusion_schedule, etc. ---

def build_global_residuals(df, patient_ids, theta_phys, patient_cls):
    """
    Returns a 1D residual vector stacking all patients' log-errors.
    theta_phys = (k1, k2, vmax_in, vmax_out) in physical scale (positive).
    """
    k1, k2, vmax_in, vmax_out = theta_phys
    resid_list = []

    for pid in patient_ids:
        patient_df = df[df["ID"] == pid]
        if patient_df.empty or patient_df["Concentration"].isna().all():
            # skip patients with no usable data
            continue

        try:
            # Build patient object
            row = patient_df.iloc[0]
            serum_creatinine = float(row['serum creatinine']) * 0.011312
            hematocrit = float(row['hematocrit']) / 100.0
            pat = patient_cls(
                row['Gender: 1, male; 2, female'],
                row['Body weight'], row['Height'], row['Age'],
                hematocrit, serum_creatinine
            )

            # Dosing schedule
            infusions, start_time = get_infusion_schedule(patient_df)
            if not infusions or start_time is None:
                continue  # skip if no valid dosing

            # Observations with valid times
            obs_df = patient_df.dropna(subset=["Concentration", "Date", "Time"]).copy()
            date_str = obs_df["Date"].astype(str).str.strip()
            time_str = obs_df["Time"].astype(str).str.strip()
            obs_df["dt"] = pd.to_datetime(date_str + " " + time_str, errors="coerce")
            obs_df = obs_df.dropna(subset=["dt"])
            if obs_df.empty:
                continue

            obs_df["t_hr"] = (obs_df["dt"] - start_time).dt.total_seconds() / 3600.0
            t_obs = obs_df["t_hr"].values
            c_exp = obs_df["Concentration"].values

            # Simulate and compute residuals
            c_sim, _, _ = simulate_patient_concentration(pat, infusions, t_obs, k1, k2, vmax_in, vmax_out)
            c_exp = np.clip(c_exp, 1e-6, None)
            c_sim = np.clip(c_sim, 1e-6, None)
            resid = np.log(c_exp) - np.log(c_sim)

            # Append
            if np.all(np.isfinite(resid)) and resid.size > 0:
                resid_list.append(resid)

        except Exception as e:
            # Optional: log and continue
            # print(f"[Patient {pid}] skipped: {e}")
            continue

    if len(resid_list) == 0:
        # No usable residuals -> return a large dummy residual to avoid crashes
        return np.array([1e3])
    return np.concatenate(resid_list)

import numpy as np
from scipy.optimize import least_squares

def fit_by_regression_global_msle(df, patient_ids, patient_cls,
                                  x0_log=np.array([-3.0, -3.0, -3.0, -3.0]),
                                  log_bounds=((-10, 2), (-10, 2), (-10, 2), (-10, 2)),
                                  robust=False):
    """
    Nonlinear least-squares fit in log-space (x = log10 theta).
    Returns:
        best_params : np.ndarray, shape (4,)
            Parameters in physical scale (positive): (k1, k2, vmax_in, vmax_out).
        J_theta : float
            Global MSLE (mean squared log-error) over all observations:
            J = sum(residuals**2) / N_total.
        res : scipy.optimize.OptimizeResult
            Full result from scipy.optimize.least_squares.

    Notes:
    - Minimises residuals r = log(c_obs) - log(c_sim) stacked across all patients/timepoints.
    - Bounds are enforced in log-space; robust=True uses a soft-L1 loss.
    """
    lb = np.array([b[0] for b in log_bounds], dtype=float)
    ub = np.array([b[1] for b in log_bounds], dtype=float)

    # --- you should already have this imported from your utils ---
    # from your_module import build_global_residuals

    def residuals_x(x_log):
        theta = 10.0 ** x_log  # back to physical scale
        return build_global_residuals(df, patient_ids, theta, patient_cls)

    res = least_squares(
        residuals_x,
        x0=x0_log,
        bounds=(lb, ub),
        method="trf",
        loss="soft_l1" if robust else "linear",
        f_scale=1.0,
        max_nfev=200,     # more evals
        xtol=1e-8, ftol=1e-8, gtol=1e-8,  # stricter tolerances
        x_scale='jac'     # better step scaling
    )

    best_params = 10.0 ** res.x

    # Compute global MSLE J(theta) on the fitted residuals
    N_total = res.fun.size
    if N_total == 0:
        J_theta = float("nan")
    else:
        SSLE = float(np.sum(res.fun**2))
        J_theta = SSLE / N_total

    return best_params, J_theta, res

def grid_search_group(df, patient_ids, patient_cls, param_grid):
    best_score = float("inf")
    best_params = None

    for k1, k2, vmax_in, vmax_out in product(*param_grid):
        params = (k1, k2, vmax_in, vmax_out)
        objfun, failed_patients = evaluate_group_objfun(df, patient_ids, params, patient_cls)
        if not np.isnan(objfun) and objfun < best_score:
            best_score = objfun
            best_params = params
            print(f"New best: ObjFun={best_score:.4f}, Params={params}")
            if failed_patients:
                print(f" Failed patients: {[p[0] for p in failed_patients]}")
    return best_params, best_score