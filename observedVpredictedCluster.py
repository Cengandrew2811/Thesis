# Generates Observed vs Predicted plots (Training & Validation)
# Saves figures + CSVs per method/cluster.


import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from patient import PatientExtended
from pbpk_fitting_utils import get_infusion_schedule, simulate_patient_concentration

USE_LOG_SCALE = True  # log–log scatter (True or Flase)
DATA_XLSX     = "Copy of mmc1[66].xlsx"
CLUSTERS_CSV  = "patient_clusterss.csv"  
OUT_BASE_DIR  = "Figures_OVP_Cluster"    # output folder
EXCLUDE_IDS   = {"580.1", "613.3", "614.1"}
MIN_AGE       = 18

# >>>>> FINAL fitted parameters here (not seeds) <<<<<
PARAMS_BY_METHOD_AND_CLUSTER = {
    "LS": {
         0: (0.022324186, 0.112119409, 0.000489414, 0.000907951),
         1: (0.049746178, 0.100447073, 0.000100592, 0.000240101), 
         2: (0.099481452, 0.249455537, 0.004057737, 0.009575658), 
         3: (0.022220106, 0.222779571, 4.40E-05, 0.000259331), 
         4: (0.044986819, 0.214954859, 2.21E-05, 0.000168882)
    },
    "DE": {
         0: (0.030407431, 0.105789874, 0.006248192, 0.006248212), 
         1: (0.057342067, 0.11127142, 0.003362401, 0.003362421), 
         2: (0.101289054, 0.168495142, 0.000179006, 0.000179016), 
         3: (0.032065808, 0.100468192, 0.003293166, 0.003294364), 
         4: (0.05925515, 0.171929018, 0.006182667, 0.008722706)
    },
    "BO": {
         0: (0.025784339, 0.123052523, 7.02E-07, 1.18E-06), 
         1: (0.042012367, 0.19517401, 3.53E-05, 3.93E-05), 
         2: (0.105877089, 0.171881689, 0.000107504, 0.000107514), 
         3: (0.024394393, 0.154806128, 0.000140067, 0.000140203), 
         4: (0.067144929, 0.174860766, 6.69E-05, 0.008805753)
    },
}
# ==========================================

def _normalize_id(x):
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s2 = s.lstrip("0")
    return s if s2 == "" else s2

def build_patient(row0):
    """Make PatientExtended from a dataframe row.
    Serum creatinine in dataset is µmol/L -> mg/dL.
    """
    scr_umol_per_L = float(row0["serum creatinine"])
    scr_mgdl = scr_umol_per_L * 0.011312  # µmol/L -> mg/dL
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

            # Infusions
            infusions, start_time = get_infusion_schedule(patient_df)
            if not infusions or start_time is None:
                failed.append((pid, "No valid infusion schedule"))
                continue

            # Observations
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

    # Metrics on plotted subset
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
    # Load data
    df = pd.read_excel(DATA_XLSX, header=0, skiprows=[1]).copy()
    df["ID"] = df["ID"].astype(str).map(_normalize_id)
    df = df[~df["ID"].isin(EXCLUDE_IDS)].copy()
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df[df["Age"] >= MIN_AGE].copy()

    clusters = pd.read_csv(CLUSTERS_CSV, dtype={"PatientID": str})
    clusters["PatientID"] = clusters["PatientID"].astype(str).map(_normalize_id)

    # Keep only IDs present in df
    present = set(df["ID"].dropna().unique())
    clusters = clusters[clusters["PatientID"].isin(present)].copy()

    # Which clusters exist?
    cluster_labels = sorted(pd.Series(clusters["Cluster"].unique()).dropna().astype(int).tolist())
    print(f"Clusters found: {cluster_labels}")

    ensure_dir(OUT_BASE_DIR)

    for method, param_map in PARAMS_BY_METHOD_AND_CLUSTER.items():
        method_dir = os.path.join(OUT_BASE_DIR, method)
        ensure_dir(method_dir)

        for cluster_id in cluster_labels:
            # Params check
            params = None
            if param_map is not None:
                params = param_map.get(cluster_id, None)
            if params is None:
                print(f"[skip] {method} Cluster {cluster_id}: no params provided.")
                continue

            # IDs in this cluster
            ids_in_cluster = clusters.loc[clusters["Cluster"] == cluster_id, "PatientID"].unique()
            ids_in_cluster = np.array(sorted(ids_in_cluster, key=lambda x: (len(x), x)))
            n_total = len(ids_in_cluster)
            if n_total == 0:
                print(f"[skip] Cluster {cluster_id}: no patients.")
                continue

            # Train/Validation split (first 2/3 rule)
            train_ids, val_ids = stable_train_val_split(ids_in_cluster)
            print(f"[{method} | Cluster {cluster_id}] total={n_total}, training={len(train_ids)}, validation={len(val_ids)}")

            # Output dir for this cluster
            out_dir = os.path.join(method_dir, f"Cluster_{cluster_id}")
            ensure_dir(out_dir)

            # Training
            obs_tr, pred_tr, pids_tr, failed_tr = run_split(df, train_ids, params)
            if obs_tr is None:
                print(f"[warn] {method} Cluster {cluster_id}: no successful training predictions.")
            else:
                print(f"Training skipped patients: {[p for p,_ in failed_tr] if failed_tr else 'None'}")
                title_tr = f"Observed vs Predicted — Training Set (Cluster {cluster_id}, {method})"
                suffix = " log-log" if USE_LOG_SCALE else ""
                fig_tr = os.path.join(out_dir, f"Observed_vs_Predicted__Training__Cluster_{cluster_id}__{method}{suffix}.png")
                csv_tr = os.path.join(out_dir, f"observed_vs_predicted__Training__Cluster_{cluster_id}__{method}.csv")
                plot_and_save(obs_tr, pred_tr, title_tr, fig_tr, use_log=USE_LOG_SCALE)
                save_csv(pids_tr, obs_tr, pred_tr, csv_tr)

            # Validation
            if len(val_ids) == 0:
                print(f"{method} Cluster {cluster_id}: no validation patients.")
            else:
                obs_va, pred_va, pids_va, failed_va = run_split(df, val_ids, params)
                if obs_va is None:
                    print(f"[warn] {method} Cluster {cluster_id}: no successful validation predictions.")
                else:
                    print(f"Validation skipped patients: {[p for p,_ in failed_va] if failed_va else 'None'}")
                    title_va = f"Observed vs Predicted — Validation Set (Cluster {cluster_id}, {method})"
                    suffix = " log-log" if USE_LOG_SCALE else ""
                    fig_va = os.path.join(out_dir, f"Observed_vs_Predicted__Validation__Cluster_{cluster_id}__{method}{suffix}.png")
                    csv_va = os.path.join(out_dir, f"observed_vs_predicted__Validation__Cluster_{cluster_id}__{method}.csv")
                    plot_and_save(obs_va, pred_va, title_va, fig_va, use_log=USE_LOG_SCALE)
                    save_csv(pids_va, obs_va, pred_va, csv_va)

    print("\nAll requested Observed vs Predicted plots generated.")