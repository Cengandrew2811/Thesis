
# Generates Observed vs Predicted plots (Training & Validation)
# parameter set per method. Saves figures + CSVs.


import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from patient import PatientExtended
from pbpk_fitting_utils import get_infusion_schedule, simulate_patient_concentration

USE_LOG_SCALE = True                  # log–log scatter
DATA_XLSX     = "Copy of mmc1[66].xlsx"
OUT_BASE_DIR  = "Figures_OVP_Cohort"  # root output folder
EXCLUDE_IDS   = {"580.1", "613.3", "614.1"}
MIN_AGE       = 18

PARAMS_BY_METHOD = {
    "Pesenti et al": ((1.5783E-3)*60, (5.1829E-03)*60, (2.4088E-07)*60, (5.6171E-06)*60),   
    "DE": (0.00, 0.0, 0, 0),   
    "BO": (0.038795402, 0.118282556, 1.69E-06, 2.57E-06),  
}
# ==========================================

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _normalize_id(x: str) -> str:
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s2 = s.lstrip("0")
    return s if s2 == "" else s2

def build_patient(row0: pd.Series) -> PatientExtended:
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

def stable_train_val_split(all_ids):
    """Deterministic 2/3–1/3 split over the full cohort."""
    ids_sorted = sorted(all_ids, key=lambda x: (len(x), x))
    n_total = len(ids_sorted)
    n_train = int(math.ceil(n_total * 2 / 3)) if n_total > 0 else 0
    return np.array(ids_sorted[:n_train]), np.array(ids_sorted[n_train:])

def run_split(df: pd.DataFrame, ids, params):
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

            # Timestamps and numeric conc
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

from matplotlib.lines import Line2D  # add this import at top if not present

def plot_and_save(obs, pred, title, save_path, use_log=True, pids=None):
    if use_log:
        mask = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
        if not np.any(mask):
            raise RuntimeError("No positive finite points to plot on log–log scale.")
    else:
        mask = np.isfinite(obs) & np.isfinite(pred)

    xo, yp = obs[mask], pred[mask]

    errors = yp - xo
    rmse = float(np.sqrt(np.mean(errors**2))) if len(errors) else np.nan

    # Unique patient count (on the masked pairs)
    n_patients = None
    if pids is not None:
        pids_masked = np.asarray(pids)[mask]
        n_patients = len(np.unique(pids_masked))

    plt.figure(figsize=(6.4, 6.4), dpi=300)
    plt.scatter(xo, yp, alpha=0.55, s=26)

    xy_min = float(min(xo.min(), yp.min()))
    xy_max = float(max(xo.max(), yp.max()))
    plt.plot([xy_min, xy_max], [xy_min, xy_max], linestyle="--", color="0.3")

    if use_log:
        plt.xscale("log"); plt.yscale("log")

    plt.xlabel("Observed (µmol/L)")
    plt.ylabel("Predicted (µmol/L)")
    plt.title(title)

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xy_min, xy_max)
    ax.set_ylim(xy_min, xy_max)
    ax.grid(True, which="both", alpha=0.3)

    label_lines = []
    if n_patients is not None:
        label_lines.append(f"N patients = {n_patients}")
    label_lines.append(f"RMSE = {rmse:.3g}")

    handles = [Line2D([], [], linestyle="none", marker="", label=lbl) for lbl in label_lines]
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.85,
              handlelength=0, handletextpad=0.0, fontsize=10)

    plt.savefig(save_path, bbox_inches="tight")
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

if __name__ == "__main__":
    df = pd.read_excel(DATA_XLSX, header=0, skiprows=[1]).copy()
    df["ID"] = df["ID"].astype(str).map(_normalize_id)
    df = df[~df["ID"].isin(EXCLUDE_IDS)].copy()
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df[df["Age"] >= MIN_AGE].copy()

    ids_all = sorted(set(df.loc[df["Concentration"].notna(), "ID"].astype(str)))
    if not ids_all:
        raise SystemExit("No patients with concentration data found after filtering.")

    train_ids, val_ids = stable_train_val_split(ids_all)
    print(f"[Cohort] total={len(ids_all)}, training={len(train_ids)}, validation={len(val_ids)}")

    ensure_dir(OUT_BASE_DIR)

    for method, params in PARAMS_BY_METHOD.items():
        if params is None or len(params) != 4:
            print(f"[skip] {method}: parameter tuple not set properly.")
            continue

        method_dir = os.path.join(OUT_BASE_DIR, method)
        ensure_dir(method_dir)

        # Training
        obs_tr, pred_tr, pids_tr, failed_tr = run_split(df, train_ids, params)
        if obs_tr is None:
            print(f"[warn] {method}: no successful training predictions.")
        else:
            if failed_tr:
                print(f"[{method} | Train] skipped: {[p for p,_ in failed_tr]}")
            title_tr = f"Observed vs Predicted — Training Set (Cohort, {method})"
            suffix = "__loglog" if USE_LOG_SCALE else ""
            fig_tr = os.path.join(method_dir, f"Observed_vs_Predicted__Training__Cohort__{method}{suffix}.png")
            csv_tr = os.path.join(method_dir, f"observed_vs_predicted_training__Cohort__{method}.csv")
            plot_and_save(obs_tr, pred_tr, title_tr, fig_tr, use_log=USE_LOG_SCALE, pids=pids_tr)
            save_csv(pids_tr, obs_tr, pred_tr, csv_tr)

        # Validation
        if len(val_ids) == 0:
            print(f"{method}: no validation patients.")
        else:
            obs_va, pred_va, pids_va, failed_va = run_split(df, val_ids, params)
            if obs_va is None:
                print(f"[warn] {method}: no successful validation predictions.")
            else:
                if failed_va:
                    print(f"[{method} | Val] skipped: {[p for p,_ in failed_va]}")
                title_va = f"Observed vs Predicted — Validation Set (Cohort, {method})"
                suffix = "__loglog" if USE_LOG_SCALE else ""
                fig_va = os.path.join(method_dir, f"Observed_vs_Predicted__Validation__Cohort__{method}{suffix}.png")
                csv_va = os.path.join(method_dir, f"observed_vs_predicted_validation__Cohort__{method}.csv")
                plot_and_save(obs_va, pred_va, title_va, fig_va, use_log=USE_LOG_SCALE, pids=pids_va)
                save_csv(pids_va, obs_va, pred_va, csv_va)

    print("\nAll requested Observed vs Predicted plots for the cohort generated.")