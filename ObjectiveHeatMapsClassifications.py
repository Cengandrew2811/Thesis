import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# LS
from trial102clusterFINAL import (
    conc_rmse_true       as ls_conc_rmse_true,
    BOUNDS_LO            as LS_BOUNDS_LO,
    BOUNDS_HI            as LS_BOUNDS_HI,
)
# DE/BO (reuse scalar objective + theta mapping)
from trial102DEcluster import (
    de_objective_theta,
    seed_true_to_theta   as de_seed_true_to_theta,
    BOUNDS_LO            as DE_BOUNDS_LO,
    BOUNDS_HI            as DE_BOUNDS_HI,
)

DATA_XLSX     = "Copy of mmc1[66].xlsx"
CLASS_CSV     = "patient_7group_classification.csv"   # columns: PatientID + (Final_Group or Cluster)
OUT_BASE_DIR  = "Figures_Objective_Maps_Classified"
EXCLUDE_IDS   = {"580.1", "613.3", "614.1"}
MIN_AGE       = 18

USE_FIRST_TWO_THIRDS = True

# Grid resolution (balance quality vs time)
N_K1 = 60
N_K2 = 60
N_VIN = 60
N_VOUT = 60

# Colormap / levels
LEVELS = 30
CMAP = "viridis"

# Expand k1,k2 range for ALL methods (log space)
EXPANDED_K_BOUNDS = (1e-6, 1e2)

def _get_bounds(method):
    """Return (lo, hi) bounds per method (used for vin/vout grids)."""
    if method == "LS":
        return LS_BOUNDS_LO, LS_BOUNDS_HI
    else:  # "DE" or "BO"
        return DE_BOUNDS_LO, DE_BOUNDS_HI


PARAMS_FINAL = {

}

# Fitted Params
STARTS_TRUE = {
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

def _first_two_thirds(ids_list):
    ids_sorted = sorted(ids_list, key=lambda x: (len(str(x)), str(x)))
    n = len(ids_sorted)
    n_train = max(1, int(math.ceil(2 * n / 3)))
    return ids_sorted[:n_train]

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _get_fixed_params(method, class_label):
    # Prefer your final params dict; else fallback to seeds
    if method in PARAMS_FINAL and class_label in PARAMS_FINAL[method]:
        return list(PARAMS_FINAL[method][class_label])
    key = str(class_label)
    if method in STARTS_TRUE and key in STARTS_TRUE[method]:
        return list(STARTS_TRUE[method][key])
    return None

def _eval_objective(method, df, ids, k1, k2, vin, vout):
    # LS: true-space RMSE
    if method == "LS":
        return float(ls_conc_rmse_true([k1, k2, vin, vout], df, ids))
    # DE/BO: scalar objective in theta-space (RMSE + tiny penalty like your DE/BO code)
    theta = de_seed_true_to_theta([k1, k2, vin, vout])
    return float(de_objective_theta(theta, df, ids))

def _plot_heatmap(
    X, Y, Z, xlabel, ylabel, title, out_png,
    logx=True, logy=True, mark=None, draw_diag=False
):
    plt.figure(figsize=(9, 7))

    Zm = np.ma.masked_invalid(np.asarray(Z, float))
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad("#eeeeee")  # light grey for invalid/masked

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

    cls = pd.read_csv(CLASS_CSV)
    if "PatientID" not in cls.columns:
        raise ValueError("Classification CSV must contain 'PatientID'.")

    # auto-detect label column
    if "Final_Group" in cls.columns:
        label_col = "Final_Group"
    elif "Cluster" in cls.columns:
        label_col = "Cluster"
    else:
        raise ValueError("Classification CSV must contain 'Final_Group' or 'Cluster'.")

    cls["PatientID"] = cls["PatientID"].astype(str).map(_normalize_id)
    present = set(df["ID"].dropna().unique().tolist())
    cls = cls[cls["PatientID"].isin(present)].copy()

    class_labels = sorted(map(str, pd.Series(cls[label_col].dropna().unique()).tolist()))
    print(f"Classes found (after filters): {class_labels}")

    # Build class -> patient ID list
    class_ids = {
        lab: cls.loc[cls[label_col].astype(str) == lab, "PatientID"].tolist()
        for lab in class_labels
    }

    # Output root
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(OUT_BASE_DIR) / f"generated_{ts}"
    _ensure_dir(root)

    # Loop methods × classes
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

        for lab in class_labels:
            ids_list = class_ids.get(lab, [])
            if len(ids_list) == 0:
                print(f"[warn] {method} | Class {lab}: no patients, skipping.")
                continue

            ids_used = _first_two_thirds(ids_list) if USE_FIRST_TWO_THIRDS else sorted(ids_list, key=lambda x: (len(x), x))
            print(f"\n=== {method} | Class {lab} | n_total={len(ids_list)} | used={len(ids_used)} (first 2/3={USE_FIRST_TWO_THIRDS}) ===")

            fixed = _get_fixed_params(method, lab)
            if fixed is None:
                print(f"[warn] No fixed params for {method} class {lab}; skipping.")
                continue
            fk1, fk2, fvin, fvout = [float(x) for x in fixed]

            odir = mdir / f"Class_{lab}"
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

            titleA = f"{method} — Objective map (k1×k2) — Class {lab}\nfixed v_in={fvin:.3g}, v_out={fvout:.3g}"
            outA = odir / f"ObjectiveMap__k1_k2__fixed_vin_vout__Class_{lab}__{method}.png"
            _plot_heatmap(K1, K2, Z,
                          r"$k_1$ [h$^{-1}$]", r"$k_2$ [h$^{-1}$]",
                          titleA, str(outA),
                          logx=True, logy=True, mark=(fk1, fk2), draw_diag=False)

            # Save raw grid
            np.savez(odir / f"ObjectiveMap__k1_k2__grid__Class_{lab}__{method}.npz",
                     K1=K1, K2=K2, Z=Z, fixed_vin=fvin, fixed_vout=fvout,
                     ids_used=np.array(ids_used, dtype=object))

            # --- (B) v_in × v_out grid (fix k1, k2), mask v_out < v_in ---
            VIN, VOUT = np.meshgrid(vin_vals, vout_vals, indexing="xy")
            Z2 = np.full_like(VIN, np.nan, dtype=float)
            mask_valid = VOUT >= VIN  # only evaluate physically valid region
            total2 = int(mask_valid.sum())
            print(f"  [B] v_in×v_out grid: fix k1={fk1:.3e}, k2={fk2:.3e} — eval {total2} valid points (mask v_out< v_in)")

            valid_indices = np.argwhere(mask_valid)
            for idx, (i, j) in enumerate(valid_indices, start=1):
                vin = float(VIN[i, j]); vout = float(VOUT[i, j])
                try:
                    Z2[i, j] = _eval_objective(method, df, ids_used, fk1, fk2, vin, vout)
                except Exception:
                    Z2[i, j] = np.nan
                if idx % max(1, total2 // 10) == 0:
                    print(f"     … {idx}/{total2}")

            titleB = f"{method} — Objective map (v_in×v_out) — Class {lab}\nfixed k1={fk1:.3g}, k2={fk2:.3g}"
            outB = odir / f"ObjectiveMap__vin_vout__fixed_k1_k2__Class_{lab}__{method}.png"
            _plot_heatmap(VIN, VOUT, Z2,
                          r"$v_{in}$ [µmol·L$^{-1}$·h$^{-1}$]",
                          r"$v_{out}$ [µmol·L$^{-1}$·h$^{-1}$]",
                          titleB, str(outB),
                          logx=True, logy=True, mark=(fvin, fvout), draw_diag=True)

            np.savez(odir / f"ObjectiveMap__vin_vout__grid__Class_{lab}__{method}.npz",
                     VIN=VIN, VOUT=VOUT, Z=Z2, fixed_k1=fk1, fixed_k2=fk2,
                     ids_used=np.array(ids_used, dtype=object))

    print("\n All objective maps generated.")