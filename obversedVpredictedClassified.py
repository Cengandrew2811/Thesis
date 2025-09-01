# Generates Observed vs Predicted plots (Training & Validation)
# Saves figures + CSVs per method/class.

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from patient import PatientExtended
from pbpk_fitting_utils import get_infusion_schedule, simulate_patient_concentration

USE_LOG_SCALE = True  # log–log scatter (True recommended)
DATA_XLSX     = "Copy of mmc1[66].xlsx"
CLASS_CSV     = "patient_7group_classification.csv"  # needs columns: PatientID, Final_Group
OUT_BASE_DIR  = "Figures_OVP_Classified"             # root output folder
EXCLUDE_IDS   = {"580.1", "613.3", "614.1"}
MIN_AGE       = 18

PARAMS_BY_METHOD_AND_CLASS = {
    "LS": {
         "A": (0.099992229, 0.101967487, 1.95E-05, 2.30E-05),
         "B": (0.044730942, 0.10055083, 4.86E-05, 0.001890823), 
         "C": (0.022273924, 0.223838963, 0.000153264, 0.0003861), 
         "D": (0.022178259, 0.161535775, 0.001730529, 0.003479109)
    },
    "DE": {
         "A": (0.155681555, 0.171142503, 0.004179796, 0.004375102), 
         "B": (0.067068231, 0.097479651, 0.000911749, 0.000911799), 
         "C": (0.023377751, 0.182975756, 4.99E-05, 4.99E-05), 
         "D": (0.033108363, 0.100942553, 0.000160617, 0.000267829)
    },
    "BO": {
         "A": (0.12309454, 0.176344445, 9.43E-05, 0.001309907), 
         "B": (0.043943275, 0.095829875, 2.33E-05, 0.009891693), 
         "C": (0.024127246, 0.174082662, 4.67E-05, 0.008230778), 
         "D": (0.036197184, 0.098084353, 0.000333318, 0.000336914)
    },
}

def _normalize_id(x):
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s2 = s.lstrip("0")
    return s if s2 == "" else s2

def build_patient(row0):
    """Make PatientExtended from a dataframe row. Serum creatinine (µmol/L) → mg/dL."""
    scr_umol_per_L = float(row0["serum creatinine"])
    scr_mgdl = scr_umol_per_L * 0.011312
    return PatientExtended(
        gender=int(row0["Gender: 1, male; 2, female"]),
        weight_kg=float(row0["Body weight"]),
        height_cm=float(row0["Height"]),
        age=float(row0["Age"]),
        hematocrit=float(row0["hematocrit"]) / 100.0,
        serum_creatinine=scr_mgdl,
    )

def stable_train_val_split(ids_in_group):
    """Deterministic split: sort IDs, take ceil(2N/3) for training."""
    ids_sorted = sorted(ids_in_group, key=lambda x: (len(x), x))
    n_total = len(ids_sorted)
    n_train = int(math.ceil(n_total * 2 / 3)) if n_total > 0 else 0
    return np.array(ids_sorted[:n_train]), np.array(ids_sorted[n_train:])

def run_split(df, ids, params):
    """Return concatenated observed, predicted, and patient IDs for a list of patient IDs."""
    all_obs, all_pred, who, failed = [], [], [], []
    k1, k2, vmax_in, vmax_out = params

    for pid in ids:
        patient_df = df[df["ID"] == pid].copy()
        if patient_df.empty or patient_df["Concentration"].isna().all():
            failed.append((pid, "No data or all concentrations missing"))
            continue
        try:
            pat = build_patient(patient_df.iloc[0])

            infusions, start_time = get_infusion_schedule(patient_df)
            if not infusions or start_time is None:
                failed.append((pid, "No valid infusion schedule"))
                continue

            obs_df = patient_df.dropna(subset=["Concentration", "Date", "Time"]).copy()
            if obs_df.empty:
                failed.append((pid, "No valid observations with Date & Time"))
                continue
            obs_df["dt"] = pd.to_datetime(
                obs_df["Date"].astype(str).str.strip() + " " + obs_df["Time"].astype(str).str.strip(),
                errors="coerce"
            )
            obs_df = obs_df.dropna(subset=["dt"])
            obs_df["Concentration"] = pd.to_numeric(obs_df["Concentration"], errors="coerce")
            obs_df = obs_df.dropna(subset=["Concentration"])
            if obs_df.empty:
                failed.append((pid, "No valid numeric concentrations"))
                continue

            obs_df["t_hr"] = (obs_df["dt"] - start_time).dt.total_seconds() / 3600.0
            t_obs = obs_df["t_hr"].to_numpy()
            c_obs = obs_df["Concentration"].to_numpy()  # µmol/L

            c_pred, _, _ = simulate_patient_concentration(pat, infusions, t_obs, k1, k2, vmax_in, vmax_out)

            all_obs.append(c_obs)
            all_pred.append(c_pred)
            who.append(np.repeat(pid, len(c_obs)))

        except Exception as e:
            failed.append((pid, f"Error: {e}"))
            continue

    if not all_obs:
        return None, None, None, failed

    obs = np.concatenate(all_obs).astype(float)
    pred = np.concatenate(all_pred).astype(float)
    pids = np.concatenate(who).astype(str)
    return obs, pred, pids, failed

def plot_and_save(obs, pred, title, save_path, use_log=True):
    """Plot scatter + metrics and save."""
    if use_log:
        mask = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
        if not np.any(mask):
            raise RuntimeError("No positive finite points to plot on log–log scale.")
        xo, yp = obs[mask], pred[mask]
    else:
        mask = np.isfinite(obs) & np.isfinite(pred)
        xo, yp = obs[mask], pred[mask]

    errors = yp - xo
    rmse = float(np.sqrt(np.mean(errors**2))) if len(errors) else np.nan

    plt.figure(figsize=(6, 6))
    plt.scatter(xo, yp, alpha=0.6, s=28)
    xy_min = float(min(xo.min(), yp.min()))
    xy_max = float(max(xo.max(), yp.max()))
    plt.plot([xy_min, xy_max], [xy_min, xy_max], linestyle="--")

    if use_log:
        plt.xscale("log"); plt.yscale("log")

    plt.xlabel("Observed (µmol/L)")
    plt.ylabel("Predicted (µmol/L)")
    plt.title(title)
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xy_min, xy_max)
    ax.set_ylim(xy_min, xy_max)

    txt = (f"n = {len(xo)}\n"
           f"RMSE = {rmse:.3g}\n")
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", alpha=0.15))

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {save_path}")

def save_csv(pids, obs, pred, save_path):
    df_out = pd.DataFrame({
        "PatientID": pids,
        "Observed_µmol_per_L": obs,
        "Predicted_µmol_per_L": pred
    })
    df_out.to_csv(save_path, index=False)
    print(f"Saved CSV: {save_path}")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

if __name__ == "__main__":
    # Load & filter cohort
    df = pd.read_excel(DATA_XLSX, header=0, skiprows=[1]).copy()
    df["ID"] = df["ID"].astype(str).map(_normalize_id)
    df = df[~df["ID"].isin(EXCLUDE_IDS)].copy()
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df[df["Age"] >= MIN_AGE].copy()

    # Load classification file
    cls = pd.read_csv(CLASS_CSV, dtype={"PatientID": str})
    if "Final_Group" in cls.columns:
        label_col = "Final_Group"
    elif "Cluster" in cls.columns:
        label_col = "Cluster"
    else:
        raise ValueError("Classification CSV must include 'Final_Group' or 'Cluster'.")

    cls["PatientID"] = cls["PatientID"].astype(str).map(_normalize_id)

    # Keep only IDs present in df
    present = set(df["ID"].dropna().unique())
    cls = cls[cls["PatientID"].isin(present)].copy()

    # Class labels (prefer A,B,C,D order if present)
    found = list(map(str, cls[label_col].dropna().unique().tolist()))
    preferred_order = ["A", "B", "C", "D"]
    class_labels = [g for g in preferred_order if g in found] + [g for g in found if g not in preferred_order]
    print(f"Classes found: {class_labels}")

    ensure_dir(OUT_BASE_DIR)

    for method, param_map in PARAMS_BY_METHOD_AND_CLASS.items():
        method_dir = os.path.join(OUT_BASE_DIR, method)
        ensure_dir(method_dir)

        for class_label in class_labels:
            params = None
            if param_map is not None:
                params = param_map.get(class_label, None)
            if params is None:
                print(f"[skip] {method} Class {class_label}: no params provided.")
                continue

            ids_in_class = cls.loc[cls[label_col].astype(str) == class_label, "PatientID"].unique()
            ids_in_class = np.array(sorted(ids_in_class, key=lambda x: (len(x), x)))
            n_total = len(ids_in_class)
            if n_total == 0:
                print(f"[skip] Class {class_label}: no patients.")
                continue

            # Train/Validation split (first 2/3 rule)
            train_ids, val_ids = stable_train_val_split(ids_in_class)
            print(f"[{method} | Class {class_label}] total={n_total}, training={len(train_ids)}, validation={len(val_ids)}")

            out_dir = os.path.join(method_dir, f"Class_{class_label}")
            ensure_dir(out_dir)

            # Training
            obs_tr, pred_tr, pids_tr, failed_tr = run_split(df, train_ids, params)
            if obs_tr is None:
                print(f"[warn] {method} Class {class_label}: no successful training predictions.")
            else:
                print(f"Training skipped patients: {[p for p,_ in failed_tr] if failed_tr else 'None'}")
                title_tr = f"Observed vs Predicted — Training Set (Class {class_label}, {method})"
                suffix = " log-log" if USE_LOG_SCALE else ""
                fig_tr = os.path.join(out_dir, f"Observed_vs_Predicted__Training__Class_{class_label}__{method}{suffix}.png")
                csv_tr = os.path.join(out_dir, f"observed_vs_predicted_training_Class_{class_label}__{method}.csv")
                plot_and_save(obs_tr, pred_tr, title_tr, fig_tr, use_log=USE_LOG_SCALE)
                save_csv(pids_tr, obs_tr, pred_tr, csv_tr)

            # Validation
            if len(val_ids) == 0:
                print(f"{method} Class {class_label}: no validation patients.")
            else:
                obs_va, pred_va, pids_va, failed_va = run_split(df, val_ids, params)
                if obs_va is None:
                    print(f"[warn] {method} Class {class_label}: no successful validation predictions.")
                else:
                    print(f"Validation skipped patients: {[p for p,_ in failed_va] if failed_va else 'None'}")
                    title_va = f"Observed vs Predicted — Validation Set (Class {class_label}, {method})"
                    suffix = " log-log" if USE_LOG_SCALE else ""
                    fig_va = os.path.join(out_dir, f"Observed_vs_Predicted__Validation__Class_{class_label}__{method}{suffix}.png")
                    csv_va = os.path.join(out_dir, f"observed_vs_predicted_validation_Class_{class_label}__{method}.csv")
                    plot_and_save(obs_va, pred_va, title_va, fig_va, use_log=USE_LOG_SCALE)
                    save_csv(pids_va, obs_va, pred_va, csv_va)

    print("\nAll requested Observed vs Predicted plots (classified) generated.")