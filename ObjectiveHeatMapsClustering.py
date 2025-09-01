import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


from trial102clusterFINAL import (
    conc_rmse_true       as ls_conc_rmse_true,
    BOUNDS_LO            as LS_BOUNDS_LO,
    BOUNDS_HI            as LS_BOUNDS_HI,
)

from trial102DEcluster import (
    de_objective_theta,
    seed_true_to_theta   as de_seed_true_to_theta,
    BOUNDS_LO            as DE_BOUNDS_LO,
    BOUNDS_HI            as DE_BOUNDS_HI,
)

DATA_XLSX     = "Copy of mmc1[66].xlsx"
CLUSTERS_CSV  = "patient_clusterss.csv"      # columns: PatientID, Cluster
OUT_BASE_DIR  = "Figures_Objective_Maps_Clustered"
EXCLUDE_IDS   = {"580.1", "613.3", "614.1"}
MIN_AGE       = 18


USE_FIRST_TWO_THIRDS = True

# Grid resolution (balance quality vs time)
N_K1 = 60
N_K2 = 60
N_VIN = 60
N_VOUT = 60

# Colormap / levels for filled heatmaps
LEVELS = 30
CMAP = "viridis"

# Expand k1,k2 range for ALL methods (log space)
EXPANDED_K_BOUNDS = (1e-6, 1e2)

def _get_bounds(method):
    """Return (lo, hi) bounds per method (used for vin/vout)."""
    if method == "LS":
        return LS_BOUNDS_LO, LS_BOUNDS_HI
    else:  # "DE" or "BO"
        return DE_BOUNDS_LO, DE_BOUNDS_HI


PARAMS_FINAL = {

}

#Fitted Params
STARTS_TRUE = {
    "LS": {
         "0": (0.022324186, 0.112119409, 0.000489414, 0.000907951),
         "1": (0.049746178, 0.100447073, 0.000100592, 0.000240101), 
         "2": (0.099481452, 0.249455537, 0.004057737, 0.009575658), 
         "3": (0.022220106, 0.222779571, 4.40E-05, 0.000259331), 
         "4": (0.044986819, 0.214954859, 2.21E-05, 0.000168882)
    },
    "DE": {
         "0": (0.030407431, 0.105789874, 0.006248192, 0.006248212), 
         "1": (0.057342067, 0.11127142, 0.003362401, 0.003362421), 
         "2": (0.101289054, 0.168495142, 0.000179006, 0.000179016), 
         "3": (0.032065808, 0.100468192, 0.003293166, 0.003294364), 
         "4": (0.05925515, 0.171929018, 0.006182667, 0.008722706)
    },
    "BO": {
         "0": (0.025784339, 0.123052523, 7.02E-07, 1.18E-06), 
         "1": (0.042012367, 0.19517401, 3.53E-05, 3.93E-05), 
         "2": (0.105877089, 0.171881689, 0.000107504, 0.000107514), 
         "3": (0.024394393, 0.154806128, 0.000140067, 0.000140203), 
         "4": (0.067144929, 0.174860766, 6.69E-05, 0.008805753)
    },
}


def _normalize_id(x):
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s2 = s.lstrip("0")
    return s if s2 == "" else s2

def _first_two_thirds(ids_list):
    ids_sorted = sorted(ids_list, key=lambda x: (len(str(x)), str(x)))
    n = len(ids_sorted)
    n_train = max(1, int(math.ceil(2 * n / 3)))
    return ids_sorted[:n_train]

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _get_fixed_params(method, cluster_label):
    if method in PARAMS_FINAL and cluster_label in PARAMS_FINAL[method]:
        return list(PARAMS_FINAL[method][cluster_label])
    key = str(cluster_label)
    if method in STARTS_TRUE and key in STARTS_TRUE[method]:
        return list(STARTS_TRUE[method][key])
    return None

def _eval_objective(method, df, ids, k1, k2, vin, vout):
    # LS: true-space RMSE
    if method == "LS":
        return float(ls_conc_rmse_true([k1, k2, vin, vout], df, ids))
    # DE/BO: use scalar objective in theta-space (RMSE + tiny penalty as in DE/BO code)
    theta = de_seed_true_to_theta([k1, k2, vin, vout])
    return float(de_objective_theta(theta, df, ids))

def _plot_heatmap(
    X, Y, Z, xlabel, ylabel, title, out_png,
    logx=True, logy=True, mark=None, draw_diag=False
):
    plt.figure(figsize=(9, 7))

    # mask NaNs so the invalid region renders in a light grey
    Zm = np.ma.masked_invalid(np.asarray(Z, float))
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad("#eeeeee")

    cs = plt.contourf(X, Y, Zm, levels=LEVELS, cmap=cmap)
    cbar = plt.colorbar(cs); cbar.set_label("Objective value (lower is better)")

    cl = plt.contour(X, Y, Zm, levels=max(3, LEVELS//3),
                     linewidths=0.8, colors="k", alpha=0.25)
    plt.clabel(cl, inline=True, fontsize=7)

    if draw_diag:
        lo_d = max(float(np.nanmin(X)), float(np.nanmin(Y)))
        hi_d = min(float(np.nanmax(X)), float(np.nanmax(Y)))
        plt.plot([lo_d, hi_d], [lo_d, hi_d], "k--", linewidth=1)

    if logx: plt.xscale("log")
    if logy: plt.yscale("log")
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)

    if mark is not None:
        plt.scatter([mark[0]], [mark[1]], c="r", s=50, marker="x", label="fixed point")
        plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ↳ saved: {out_png}")

if __name__ == "__main__":
    # --- Load + filter cohort once ---
    df = pd.read_excel(DATA_XLSX, header=0, skiprows=[1]).copy()
    df["ID"] = df["ID"].astype(str).map(_normalize_id)
    df = df[~df["ID"].isin(EXCLUDE_IDS)].copy()
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df[df["Age"] >= MIN_AGE].copy()

    cls = pd.read_csv(CLUSTERS_CSV)
    if "PatientID" not in cls or "Cluster" not in cls:
        raise ValueError("CLUSTERS_CSV must contain columns: PatientID, Cluster")
    cls["PatientID"] = cls["PatientID"].astype(str).map(_normalize_id)
    present = set(df["ID"].dropna().unique().tolist())
    cls = cls[cls["PatientID"].isin(present)].copy()

    cluster_labels = sorted(pd.Series(cls["Cluster"].unique()).dropna().astype(str).tolist())
    print(f"Clusters found (after filters): {cluster_labels}")

    # Build cluster -> patient ID list
    cluster_ids = {
        lab: cls.loc[cls["Cluster"].astype(str) == lab, "PatientID"].tolist()
        for lab in cluster_labels
    }

    # Output root
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(OUT_BASE_DIR) / f"generated_{ts}"
    _ensure_dir(root)

    # Loop methods × clusters
    for method in ["LS", "DE", "BO"]:
        mdir = root / method
        _ensure_dir(mdir)

        # Bounds (vin/vout from method-specific, k1/k2 from expanded universal bounds)
        lo, hi = _get_bounds(method)
        k1_lo, k1_hi = EXPANDED_K_BOUNDS
        k2_lo, k2_hi = EXPANDED_K_BOUNDS

        k1_vals  = np.logspace(np.log10(k1_lo), np.log10(k1_hi), N_K1)
        k2_vals  = np.logspace(np.log10(k2_lo), np.log10(k2_hi), N_K2)
        vin_vals = np.logspace(np.log10(lo[2]), np.log10(hi[2]), N_VIN)
        vout_vals= np.logspace(np.log10(lo[3]), np.log10(hi[3]), N_VOUT)

        for lab in cluster_labels:
            ids_list = cluster_ids.get(lab, [])
            if len(ids_list) == 0:
                print(f"[warn] {method} | Cluster {lab}: no patients, skipping.")
                continue

            ids_used = _first_two_thirds(ids_list) if USE_FIRST_TWO_THIRDS else sorted(ids_list, key=lambda x: (len(x), x))
            print(f"\n=== {method} | Cluster {lab} | n_total={len(ids_list)} | used={len(ids_used)} (first 2/3={USE_FIRST_TWO_THIRDS}) ===")

            fixed = _get_fixed_params(method, lab)
            if fixed is None:
                print(f"[warn] No fixed params for {method} cluster {lab}; skipping.")
                continue
            fk1, fk2, fvin, fvout = [float(x) for x in fixed]

            odir = mdir / f"Cluster_{lab}"
            _ensure_dir(odir)

            # --- (A) k1 × k2 grid (fix vin, vout) ---
            K1, K2 = np.meshgrid(k1_vals, k2_vals, indexing="xy")
            Z = np.zeros_like(K1, dtype=float)
            total = K1.size
            print(f"  [A] k1×k2 grid: fix vin={fvin:.3e}, vout={fvout:.3e} — eval {total} points")
            for idx, (i, j) in enumerate(np.ndindex(K1.shape), start=1):
                k1 = float(K1[i, j]); k2 = float(K2[i, j])
                try:
                    Z[i, j] = _eval_objective(method, df, ids_used, k1, k2, fvin, fvout)
                except Exception:
                    Z[i, j] = np.nan
                if idx % max(1, total // 10) == 0:
                    print(f"     … {idx}/{total}")

            titleA = f"{method} — Objective map (k1×k2) — Cluster {lab}\nfixed vin={fvin:.3g}, vout={fvout:.3g}"
            outA = odir / f"ObjectiveMap__k1_k2__fixed_vin_vout__Cluster_{lab}__{method}.png"
            _plot_heatmap(K1, K2, Z,
                          r"$k_1$ [h$^{-1}$]", r"$k_2$ [h$^{-1}$]",
                          titleA, str(outA),
                          logx=True, logy=True, mark=(fk1, fk2), draw_diag=False)

            # Save raw grid too
            np.savez(odir / f"ObjectiveMap__k1_k2__grid__Cluster_{lab}__{method}.npz",
                     K1=K1, K2=K2, Z=Z, fixed_vin=fvin, fixed_vout=fvout, ids_used=np.array(ids_used, dtype=object))

            # --- (B) vin × vout grid (fix k1, k2), mask vout < vin ---
            VIN, VOUT = np.meshgrid(vin_vals, vout_vals, indexing="xy")
            Z2 = np.full_like(VIN, np.nan, dtype=float)
            mask_valid = VOUT >= VIN  # only evaluate physically valid region
            total2 = int(mask_valid.sum())
            print(f"  [B] vin×vout grid: fix k1={fk1:.3e}, k2={fk2:.3e} — eval {total2} valid points (mask vout<vin)")

            valid_indices = np.argwhere(mask_valid)
            for idx, (i, j) in enumerate(valid_indices, start=1):
                vin = float(VIN[i, j]); vout = float(VOUT[i, j])
                try:
                    Z2[i, j] = _eval_objective(method, df, ids_used, fk1, fk2, vin, vout)
                except Exception:
                    Z2[i, j] = np.nan
                if idx % max(1, total2 // 10) == 0:
                    print(f"     … {idx}/{total2}")

            titleB = f"{method} — Objective map (v_in×v_out) — Cluster {lab}\nfixed k1={fk1:.3g}, k2={fk2:.3g}"
            outB = odir / f"ObjectiveMap__vin_vout__fixed_k1_k2__Cluster_{lab}__{method}.png"
            _plot_heatmap(VIN, VOUT, Z2,
                          r"$v_{in}$ [µmol·L$^{-1}$·h$^{-1}$]",
                          r"$v_{out}$ [µmol·L$^{-1}$·h$^{-1}$]",
                          titleB, str(outB),
                          logx=True, logy=True, mark=(fvin, fvout), draw_diag=True)

            np.savez(odir / f"ObjectiveMap__vin_vout__grid__Cluster_{lab}__{method}.npz",
                     VIN=VIN, VOUT=VOUT, Z=Z2, fixed_k1=fk1, fixed_k2=fk2, ids_used=np.array(ids_used, dtype=object))

    print("\n All objective maps generated.")