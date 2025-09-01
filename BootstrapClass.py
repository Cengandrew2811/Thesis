# bootstrap_fast_refine_by_classification.py
import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from scipy.optimize import least_squares, minimize
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# LS (trial102clusterFINAL.py)
from trial102clusterFINAL import (
    group_residuals_theta as ls_group_residuals_theta,
    seed_true_to_theta   as ls_seed_true_to_theta,
    map_theta_to_true    as ls_map_theta_to_true,
    conc_rmse_true       as ls_conc_rmse_true,
    BOUNDS_LO            as LS_BOUNDS_LO,
    BOUNDS_HI            as LS_BOUNDS_HI,
    BETA_MAX             as LS_BETA_MAX,
    X_SCALE_TRUE         as LS_X_SCALE_TRUE,
    FTOL                 as LS_FTOL,
    XTOL                 as LS_XTOL,
    GTOL                 as LS_GTOL,
    MAX_NFEV             as LS_MAX_NFEV,
)

# DE (trial102DEcluster.py) – reuse this scalar objective for DE & BO 
from trial102DEcluster import (
    de_objective_theta,
    seed_true_to_theta   as de_seed_true_to_theta,
    map_theta_to_true    as de_map_theta_to_true,
    conc_rmse_true       as de_conc_rmse_true,
    BOUNDS_LO            as DE_BOUNDS_LO,
    BOUNDS_HI            as DE_BOUNDS_HI,
    BETA_MAX             as DE_BETA_MAX,
)

# -----------------------
# Config
# -----------------------
DATA_XLSX     = "Copy of mmc1[66].xlsx"
CLASS_CSV     = "patient_7group_classification.csv"  
EXCLUDE_IDS   = {"580.1", "613.3", "614.1"}
MIN_AGE       = 18

N_BOOT        = 500                       # per method × per class
RANDOM_SEED   = 42

# L-BFGS-B settings for DE/BO 
LBFGSB_MAXITER = 300
LBFGSB_TOL     = 1e-6

#Starting Values
STARTS_TRUE = {
    "LS": {
        "A": [0.1,         0.1,         5.00e-06, 8.47e-06],
        "B": [0.044724681, 0.099996639, 4.45e-05, 1.880192e-03],
        "C": [0.022362982, 0.223532992, 1.00e-04, 3.34162e-04],
        "D": [0.022341915, 0.128166048, 8.52e-05, 8.52e-05],
    },
    "DE": {
        "A": [0.150829383, 0.165866231, 4.215077e-03, 4.407024e-03],
        "B": [0.066796939, 0.092014176, 8.18735e-04, 8.18785e-04],
        "C": [0.02337844,  0.182863456, 4.74e-05,    4.74e-05],
        "D": [0.032779024, 0.1,         1.12156e-04, 2.20883e-04],
    },
    "BO": {
        "A": [0.121447201, 0.173399943, 1.00e-06,    1.230191e-03],
        "B": [0.043495952, 0.095801233, 8.95e-06,    9.891554e-03],
        "C": [0.0241333,   0.174073672, 4.47e-05,    8.229623e-03],
        "D": [0.031047815, 0.083126817, 4.85e-08,    3.77e-06],
    },
}


def _normalize_id(x):
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s2 = s.lstrip("0")
    return s if s2 == "" else s2


def _safe_filename(title: str) -> str:
    keep = "-_.()[] "
    name = "".join(ch if ch.isalnum() or ch in keep else "_" for ch in title).strip()
    if not name.lower().endswith(".jpg"):
        name += ".jpg"
    return name

def _save_kde_curve(series: pd.Series, title: str, out_path: Path):
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if x.size < 2 or np.allclose(np.std(x), 0.0):
        # Not enough variation to draw a KDE; skip quietly
        return
    kde = gaussian_kde(x)
    x_lo, x_hi = np.percentile(x, [0.5, 99.5])
    if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_lo == x_hi:
        x_lo, x_hi = np.min(x), np.max(x)
        if x_lo == x_hi:
            return
    xs = np.linspace(x_lo, x_hi, 512)
    ys = kde(xs)

    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, linewidth=2)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.tight_layout()
    out_path = out_path if out_path.suffix.lower() == ".jpg" else out_path.with_suffix(".jpg")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

G_DF = None

def _init_pool(data_xlsx, exclude_ids, min_age):
    global G_DF
    df = pd.read_excel(data_xlsx, header=0, skiprows=[1]).copy()
    df["ID"] = df["ID"].astype(str).map(_normalize_id)
    df = df[~df["ID"].isin(exclude_ids)].copy()
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df[df["Age"] >= min_age].copy()
    G_DF = df

def _first_two_thirds(ids_list):
    """Deterministic 'first' subset: sort IDs, then take ceil(2N/3)."""
    ids_sorted = sorted(ids_list)
    n = len(ids_sorted)
    n_train = max(3, int(math.ceil(2*n/3)))
    return ids_sorted[:n_train]

def _boot_samples_from_pool(pool_ids, n_boot, rng):
    """Sample WITH replacement from pool; each replicate has size=len(pool_ids)."""
    m = len(pool_ids)
    return [list(rng.choice(pool_ids, size=m, replace=True)) for _ in range(n_boot)]

# ls worker
def worker_ls(args):
    rep_idx, boot_ids, theta0_true = args
    theta0 = ls_seed_true_to_theta(np.asarray(theta0_true, dtype=float))
    try:
        res = least_squares(
            ls_group_residuals_theta, theta0,
            bounds=(
                [LS_BOUNDS_LO[0], LS_BOUNDS_LO[1], LS_BOUNDS_LO[2], -LS_BETA_MAX],
                [LS_BOUNDS_HI[0], LS_BOUNDS_HI[1], LS_BOUNDS_HI[2],  LS_BETA_MAX],
            ),
            loss="soft_l1", f_scale=0.3,
            args=(G_DF, boot_ids, None),
            method="trf",
            x_scale=LS_X_SCALE_TRUE + [1.0],
            ftol=LS_FTOL, xtol=LS_XTOL, gtol=LS_GTOL,
            max_nfev=LS_MAX_NFEV,
            verbose=0,
        )
        params_true = ls_map_theta_to_true(res.x)
        obj_rmse = ls_conc_rmse_true(params_true, G_DF, boot_ids)
        return {
            "replicate": rep_idx, "k1": params_true[0], "k2": params_true[1],
            "vmax_in": params_true[2], "vmax_out": params_true[3],
            "ObjFun": float(obj_rmse), "status": int(res.status),
            "message": str(res.message), "ids_used": ";".join(boot_ids),
        }
    except Exception as e:
        return {"replicate": rep_idx, "error": str(e)}

# DE/BO worker
def _theta_bounds_de():
    return [
        (DE_BOUNDS_LO[0], DE_BOUNDS_HI[0]),
        (DE_BOUNDS_LO[1], DE_BOUNDS_HI[1]),
        (DE_BOUNDS_LO[2], DE_BOUNDS_HI[2]),
        (-DE_BETA_MAX,    DE_BETA_MAX),
    ]

def worker_scalar(args):
    # Used for both DE and BO fast refine (same scalar objective: RMSE + tiny pair penalty)
    rep_idx, boot_ids, theta0_true = args
    theta0 = de_seed_true_to_theta(np.asarray(theta0_true, dtype=float))
    try:
        res = minimize(
            fun=lambda th: de_objective_theta(th, G_DF, boot_ids),
            x0=theta0,
            method="L-BFGS-B",
            bounds=_theta_bounds_de(),
            options={"maxiter": LBFGSB_MAXITER, "ftol": LBFGSB_TOL},
        )
        params_true = de_map_theta_to_true(res.x)
        obj_rmse = de_conc_rmse_true(params_true, G_DF, boot_ids)
        return {
            "replicate": rep_idx, "k1": params_true[0], "k2": params_true[1],
            "vmax_in": params_true[2], "vmax_out": params_true[3],
            "ObjFun": float(obj_rmse), "success": bool(res.success),
            "message": str(res.message), "ids_used": ";".join(boot_ids),
        }
    except Exception as e:
        return {"replicate": rep_idx, "error": str(e)}

def run_one(method, class_label, ids_in_class, seed_true, n_boot, n_procs):
    # Build resampling pool = FIRST 2/3 of the class (sorted for determinism)
    pool_ids = _first_two_thirds(ids_in_class)
    if len(pool_ids) == 0:
        print(f"[warn] Class {class_label} has 0 pool patients; skipping.")
        return None, None

    rng = np.random.default_rng(RANDOM_SEED + hash((method, str(class_label))) % (2**31 - 1))
    boots = _boot_samples_from_pool(pool_ids, n_boot, rng)
    tasks = [(i+1, boots[i], seed_true) for i in range(n_boot)]

    worker = worker_ls if method == "LS" else worker_scalar
    print(f"\n=== {method} | Class {class_label} | total_n={len(ids_in_class)} | "
          f"pool(first 2/3)={len(pool_ids)} | boot={n_boot} ===")

    results, done = [], 0
    with Pool(processes=n_procs, initializer=_init_pool,
              initargs=(DATA_XLSX, EXCLUDE_IDS, MIN_AGE)) as pool:
        for out in pool.imap_unordered(worker, tasks):
            results.append(out)
            done += 1
            if done % max(1, n_boot // 5) == 0:
                if isinstance(out, dict) and "ObjFun" in out:
                    print(f"… {done}/{n_boot} done (latest ObjFun={out['ObjFun']:.4f})")
                else:
                    print(f"… {done}/{n_boot} done (latest failed: {out.get('error')})")

    df_boot = pd.DataFrame(results)
    df_ok = df_boot.dropna(subset=["k1","k2","vmax_in","vmax_out","ObjFun"])
    summary = df_ok[["k1","k2","vmax_in","vmax_out","ObjFun"]].agg(["mean","std"]).T
    summary = summary.rename(columns={"mean":"Mean","std":"SD"})

    # Save (classification outputs go here)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("bootstrap_outputs_classified")
    out_dir.mkdir(parents=True, exist_ok=True)
    boot_csv    = out_dir / f"boot_replicates_{method}_class{class_label}_{ts}.csv"
    summary_csv = out_dir / f"boot_summary_{method}_class{class_label}_{ts}.csv"
    df_boot.to_csv(boot_csv, index=False)
    summary.to_csv(summary_csv)

    # KDE Figures
    params_to_plot = ["k1", "k2", "vmax_in", "vmax_out"]
    for p in params_to_plot:
        if p in df_ok.columns and df_ok[p].notna().sum() >= 2:
            title = f"Distribution of {p} for Group {class_label} from {method}"
            fig_path = out_dir / _safe_filename(title)
            _save_kde_curve(df_ok[p], title, fig_path)

    print(f"\n{method} Class {class_label} finished (N={n_boot}).")
    print(f"   Replicates saved: {boot_csv}")
    print(f"   Summary saved:    {summary_csv}")

    # Console summary (mean ± SD)
    print("\nSummary (mean ± SD):")
    for p in ["k1", "k2", "vmax_in", "vmax_out", "ObjFun"]:
        if p in df_ok:
            m = df_ok[p].mean()
            s = df_ok[p].std(ddof=1)
            print(f"  {p:8s} = {m:.3e} ± {s:.3e}")

    return df_boot, summary

if __name__ == "__main__":
    # Load + filter once (parent)
    df = pd.read_excel(DATA_XLSX, header=0, skiprows=[1]).copy()
    df["ID"] = df["ID"].astype(str).map(_normalize_id)
    df = df[~df["ID"].isin(EXCLUDE_IDS)].copy()
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df[df["Age"] >= MIN_AGE].copy()

    # Load classification file and keep only IDs remaining after filters
    cls = pd.read_csv(CLASS_CSV)
    # Auto-detect label column (prefers Final_Group)
    if "Final_Group" in cls.columns:
        label_col = "Final_Group"
    elif "Cluster" in cls.columns:  # fallback
        label_col = "Cluster"
    else:
        raise ValueError("Classification CSV must contain either 'Final_Group' or 'Cluster' column.")
    if "PatientID" not in cls.columns:
        raise ValueError("Classification CSV must contain 'PatientID' column.")

    cls["PatientID"] = cls["PatientID"].astype(str).map(_normalize_id)
    present = set(df["ID"].dropna().unique().tolist())
    cls = cls[cls["PatientID"].isin(present)].copy()

    # Restrict to A, B, C, D (in that order) if present
    allowed_labels = ["A", "B", "C", "D"]
    found = set(map(str, cls[label_col].unique().tolist()))
    class_labels = [lab for lab in allowed_labels if lab in found]
    print(f"Classes found (after filters): {class_labels}")

    # Build class -> patient ID list (after filters)
    class_ids = {
        lab: cls.loc[cls[label_col].astype(str) == lab, "PatientID"].tolist()
        for lab in class_labels
    }

    # Cores
    n_procs = max(1, cpu_count() - 1)

    # Run all methods × classes
    for method in ["LS", "DE", "BO"]:
        if method not in STARTS_TRUE:
            print(f"[warn] No seeds provided for method {method}; skipping.")
            continue
        for lab in class_labels:
            seed_map = STARTS_TRUE[method]
            if lab not in seed_map:
                # fallback: geometric-center of LS bounds
                seed_true = [
                    np.sqrt(LS_BOUNDS_LO[0]*LS_BOUNDS_HI[0]),
                    np.sqrt(LS_BOUNDS_LO[1]*LS_BOUNDS_HI[1]),
                    np.sqrt(LS_BOUNDS_LO[2]*LS_BOUNDS_HI[2]),
                    np.sqrt(LS_BOUNDS_LO[3]*LS_BOUNDS_HI[3]),
                ]
                print(f"[warn] No seed for {method} class {lab}; using bounds-center fallback.")
            else:
                seed_true = seed_map[lab]

            ids_list = class_ids.get(lab, [])
            if len(ids_list) == 0:
                print(f"[warn] Class {lab} has 0 patients after filters; skipping.")
                continue

            run_one(method, lab, ids_list, seed_true, N_BOOT, n_procs)