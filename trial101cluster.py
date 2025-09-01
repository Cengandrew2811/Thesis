import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import itertools
from math import inf

from patient import PatientExtended
from pbpk_fitting_utils import simulate_patient_concentration, get_infusion_schedule

EPS = 1e-9

# RMSE functions

def compute_rmse_original(params, df, patient_ids, PatientClass):
    """RMSE in concentration units (µM etc.)"""
    all_obs, all_sim = [], []
    k1, k2, vmax_in, vmax_out = params

    for pid in patient_ids:
        df_pat = df[df["ID"] == pid]
        if df_pat.empty:
            continue

        # build patient
        gender = df_pat["Gender: 1, male; 2, female"].iloc[0]
        weight_kg = df_pat["Body weight"].iloc[0]
        height_cm = df_pat["Height"].iloc[0]
        age = df_pat["Age"].iloc[0]
        hematocrit_fraction = float(df_pat["hematocrit"].iloc[0]) / 100.0
        serum_creatinine = float(df_pat["serum creatinine"].iloc[0]) * 0.0113
        pat = PatientClass(gender, weight_kg, height_cm, age,
                           hematocrit_fraction, serum_creatinine)

        infusions, start_time = get_infusion_schedule(df_pat)
        if not infusions or start_time is None:
            continue

        obs_df = df_pat.dropna(subset=["Concentration", "Date", "Time"]).copy()
        obs_df["dt"] = pd.to_datetime(
            obs_df["Date"].astype(str).str.strip() + " " +
            obs_df["Time"].astype(str).str.strip(),
            errors="coerce"
        )
        obs_df = obs_df.dropna(subset=["dt"])
        t_obs = (obs_df["dt"] - start_time).dt.total_seconds().values / 3600.0
        t_obs = np.clip(t_obs, 0, None)
        c_obs = obs_df["Concentration"].values

        if t_obs.size == 0:
            continue

        c_sim, _, _ = simulate_patient_concentration(
            pat, infusions, t_obs, k1, k2, vmax_in, vmax_out
        )

        all_obs.append(c_obs)
        all_sim.append(c_sim)

    if not all_obs or not all_sim:
        return 1e9

    all_obs = np.concatenate(all_obs)
    all_sim = np.concatenate(all_sim)
    return np.sqrt(np.mean((all_sim - all_obs) ** 2))


def group_residuals(params, df, patient_ids, patient_cls):
    """Residuals in log-space (dimensionless), with weights."""
    k1, k2, vmax_in, vmax_out = params
    all_res = []

    for pid in patient_ids:
        patient_df = df[df["ID"] == pid]
        if patient_df.empty or patient_df["Concentration"].isna().all():
            continue

        try:
            row = patient_df.iloc[0]
            serum_creatinine = float(row['serum creatinine']) * 0.0113
            hematocrit = float(row['hematocrit']) / 100.0
            pat = patient_cls(row['Gender: 1, male; 2, female'],
                              row['Body weight'], row['Height'], row['Age'],
                              hematocrit, serum_creatinine)

            infusions, start_time = get_infusion_schedule(patient_df)
            if not infusions or start_time is None:
                continue

            obs_df = patient_df.dropna(subset=["Concentration", "Date", "Time"]).copy()
            obs_df["dt"] = pd.to_datetime(
                obs_df["Date"].astype(str).str.strip() + " " +
                obs_df["Time"].astype(str).str.strip(),
                errors="coerce"
            )
            obs_df = obs_df.dropna(subset=["dt"])
            t_obs = (obs_df["dt"] - start_time).dt.total_seconds().values / 3600.0
            t_obs = np.clip(t_obs, 0, None)
            c_exp = np.clip(obs_df["Concentration"].values, EPS, None)

            if t_obs.size == 0:
                continue

            c_sim, _, _ = simulate_patient_concentration(
                pat, infusions, t_obs, k1, k2, vmax_in, vmax_out
            )
            c_sim = np.clip(c_sim, EPS, None)

            r = np.log(c_sim) - np.log(c_exp)

            w_patient = 1.0 / np.sqrt(len(r))
            w_time = 1.0 / np.sqrt(1.0 + t_obs)

            all_res.append(w_patient * w_time * r)

        except Exception as e:
            print(f"[Patient {pid}] skipped due to error: {e}")
            continue

    if not all_res:
        return np.array([1e3])

    return np.concatenate(all_res)


def log_rmse_seed(params, df, patient_ids, PatientClass):
    res = group_residuals(params, df, patient_ids, PatientClass)
    if res.size == 0 or not np.all(np.isfinite(res)):
        return 1e9
    return np.sqrt(np.mean(res**2))


# Grid search

def grid_search_init(df, patient_ids, PatientClass,
                     k1_grid=None, k2_grid=None, vin_grid=None, vout_grid=None,
                     max_candidates=1):
    if k1_grid is None:      k1_grid  = np.logspace(-6, -1, 6)
    if k2_grid is None:      k2_grid  = np.logspace(-6, -1, 6)
    if vin_grid is None:     vin_grid = np.logspace(-8, -3, 6)
    if vout_grid is None:    vout_grid= np.logspace(-8, -3, 6)

    results = []
    best_rmse = inf
    best_params = None

    combos = itertools.product(k1_grid, k2_grid, vin_grid, vout_grid)
    total = len(k1_grid)*len(k2_grid)*len(vin_grid)*len(vout_grid)
    print(f"[grid] evaluating {total} combos on {len(patient_ids)} patients...")

    for ix, (k1, k2, vin, vout) in enumerate(combos, 1):
        try:
            rmse = log_rmse_seed([k1, k2, vin, vout], df, patient_ids, PatientClass)
        except Exception:
            rmse = 1e9

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = [k1, k2, vin, vout]

        results.append((np.array([k1, k2, vin, vout]), rmse))

        if ix % 200 == 0:
            print(f"[grid] {ix}/{total} checked. best log-RMSE={best_rmse:.4g}")

    results.sort(key=lambda t: t[1])
    return results[:max_candidates]



if __name__ == "__main__":
    # Load PBPK data
    pbpk_data = pd.read_excel("Copy of mmc1[66].xlsx", header=0, skiprows=[1])
    pbpk_data["ID"] = pbpk_data["ID"].astype(str)

    # Load clustering file
    classification_data = pd.read_csv("patient_clusterss.csv")
    classification_data["PatientID"] = classification_data["PatientID"].astype(str)

    # Bounds
    bounds_lo = [1e-6, 1e-6, 1e-8, 1e-8]
    bounds_hi = [1.0,   1.0,   1e-2,  1e-2]

    # Loop over all clusters automatically
    for group in sorted(classification_data["Cluster"].unique()):
        all_ids = classification_data[classification_data["Cluster"] == group]["PatientID"].tolist()
        n_subset = int(len(all_ids) * 2 / 3)
        patient_ids = all_ids[:n_subset]

        print(f"\n=== Optimising for Cluster {group} with {len(patient_ids)} patients ===")

        top_inits = grid_search_init(pbpk_data, patient_ids, PatientExtended, max_candidates=3)

        best_res, best_score = None, inf
        for i, (x0, seed_score) in enumerate(top_inits, 1):
            print(f"\n[LS start {i}] seed {x0} (grid log-RMSE={seed_score:.4g})")
            res_i = least_squares(
                group_residuals, x0,
                bounds=(bounds_lo, bounds_hi),
                loss="soft_l1",
                f_scale=0.3,
                args=(pbpk_data, patient_ids, PatientExtended),
                max_nfev=200
            )
            res_score = log_rmse_seed(res_i.x, pbpk_data, patient_ids, PatientExtended)
            print(f"[LS start {i}] done. log-RMSE={res_score:.4g}, params={res_i.x}")

            if res_score < best_score:
                best_res, best_score = res_i, res_score

        residuals = group_residuals(best_res.x, pbpk_data, patient_ids, PatientExtended)
        rmse_weighted = np.sqrt(np.sum(residuals**2) / residuals.size)
        rmse_conc = compute_rmse_original(best_res.x, pbpk_data, patient_ids, PatientExtended)

        print(f"\n=== Cluster {group} results ===")
        print("params:", best_res.x)
        print("RMSE (log-space):", rmse_weighted)
        print("RMSE (concentration):", rmse_conc)