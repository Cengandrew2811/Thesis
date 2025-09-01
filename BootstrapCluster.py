# bootstrap_fast_refine_by_cluster.py
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

# DE (trial102DEcluster.py) – reuse objective for DE & BO fast refine
from trial102DEcluster import (
    de_objective_theta,
    seed_true_to_theta   as de_seed_true_to_theta,
    map_theta_to_true    as de_map_theta_to_true,
    conc_rmse_true       as de_conc_rmse_true,
    BOUNDS_LO            as DE_BOUNDS_LO,
    BOUNDS_HI            as DE_BOUNDS_HI,
    BETA_MAX             as DE_BETA_MAX,
)


DATA_XLSX     = "Copy of mmc1[66].xlsx"
CLUSTERS_CSV  = "patient_clusterss.csv"   # must have columns: PatientID, Cluster
EXCLUDE_IDS   = {"580.1", "613.3", "614.1"}
MIN_AGE       = 18

N_BOOT        = 500                       # per method × per cluster
RANDOM_SEED   = 42

# L-BFGS-B settings for DE/BO 
LBFGSB_MAXITER = 300
LBFGSB_TOL     = 1e-6

# Starting Values

STARTS_TRUE = {
    "LS": {
        "0": [0.022397071, 0.113050542, 1.04e-07, 1.04e-07],
        "1": [0.050005208, 0.100027235, 5.00e-07, 2.35e-05],
        "2": [0.099623448, 0.225095740, 3.675808e-03, 9.887156e-03],
        "3": [0.022360682, 0.223606818, 4.47e-05, 2.23629e-04],
        "4": [0.044721360, 0.223606799, 1.00e-05, 7.07e-05],
    },
    "DE": {
        "0": [0.030586722, 0.099999983, 6.237055e-03, 6.237075e-03],
        "1": [0.052803974, 0.100000000, 3.241779e-03, 3.241800e-03],
        "2": [0.100000000, 0.166274403, 1.31863e-04, 1.31863e-04],
        "3": [0.032203719, 0.100000000, 3.263337e-03, 3.264544e-03],
        "4": [0.058984085, 0.172350898, 6.275331e-03, 8.757622e-03],
    },
    "BO": {
        "0": [0.024127130, 0.123258773, 5.69e-08, 5.32e-07],
        "1": [0.039527110, 0.193405936, 1.00e-07, 4.13262475e-06],
        "2": [0.104583153, 0.168575447, 1.12e-06, 1.12e-06],
        "3": [0.024097373, 0.152195863, 8.94e-06, 9.08e-06],
        "4": [0.067109915, 0.174219777, 1.62e-05, 8.80029e-03],
    },
}


def _normalize_id(x):
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s2 = s.lstrip("0")
    return s if s2 == "" else s2

# --- KDE helpers (smooth curves only) ---
def _safe_filename(title: str) -> str:
    return title.replace("/", "-").replace(" ", "_") + ".jpg"

def _save_kde_curve(values, title, out_path):
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    if len(vals) == 0:
        return  # nothing to plot

    plt.figure(figsize=(7, 4))
    # If all values are identical, synthesize a narrow Gaussian so it's still a curve.
    if len(np.unique(vals)) < 2:
        mu = float(vals[0])
        sigma = max(1e-12, abs(mu) * 1e-3 + 1e-6)
        xs = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 400)
        ys = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
    else:
        kde = gaussian_kde(vals)
        lo = np.quantile(vals, 0.01)
        hi = np.quantile(vals, 0.99)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = np.min(vals), np.max(vals)
            pad = 0.05 * (hi - lo if hi > lo else max(1.0, abs(hi)))
            lo -= pad
            hi += pad
        xs = np.linspace(lo, hi, 512)
        ys = kde(xs)

    plt.plot(xs, ys, linewidth=2)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("density")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# Globals for workers
G_DF = None

def _init_pool(data_xlsx, EXCLUDE_IDS, MIN_AGE):
    global G_DF
    df = pd.read_excel(data_xlsx, header=0, skiprows=[1]).copy()
    df["ID"] = df["ID"].astype(str).map(_normalize_id)
    df = df[~df["ID"].isin(EXCLUDE_IDS)].copy()
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df[df["Age"] >= MIN_AGE].copy()
    G_DF = df

# training set
def _first_two_thirds(ids_list):
    """Deterministic 'first' subset: sort IDs, then take ceil(2N/3)."""
    ids_sorted = sorted(ids_list)
    n = len(ids_sorted)
    n_train = max(3, int(math.ceil(2 * n / 3)))
    return ids_sorted[:n_train]

def _boot_samples_from_pool(pool_ids, n_boot, rng):
    """Sample WITH replacement from the pool; replicate size = len(pool_ids)."""
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

# de / bo worker
def _theta_bounds_de():
    return [
        (DE_BOUNDS_LO[0], DE_BOUNDS_HI[0]),
        (DE_BOUNDS_LO[1], DE_BOUNDS_HI[1]),
        (DE_BOUNDS_LO[2], DE_BOUNDS_HI[2]),
        (-DE_BETA_MAX,    DE_BETA_MAX),
    ]

def worker_scalar(args):
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

def run_one(method, cluster_label, ids_in_cluster, seed_true, n_boot, n_procs):
    pool_ids = _first_two_thirds(ids_in_cluster)
    if len(pool_ids) == 0:
        print(f"[warn] Cluster {cluster_label} has 0 pool patients; skipping.")
        return None, None

    rng = np.random.default_rng(RANDOM_SEED + hash((method, str(cluster_label))) % (2**31 - 1))
    boots = _boot_samples_from_pool(pool_ids, n_boot, rng)
    tasks = [(i + 1, boots[i], seed_true) for i in range(n_boot)]

    worker = worker_ls if method == "LS" else worker_scalar
    print(
        f"\n=== {method} | Cluster {cluster_label} | total_n={len(ids_in_cluster)} | "
        f"pool(first 2/3)={len(pool_ids)} | boot={n_boot} ==="
    )
    results, done = [], 0
    with Pool(processes=n_procs, initializer=_init_pool, initargs=(DATA_XLSX, EXCLUDE_IDS, MIN_AGE)) as pool:
        for out in pool.imap_unordered(worker, tasks):
            results.append(out)
            done += 1
            if done % max(1, n_boot // 5) == 0:
                if "ObjFun" in out:
                    print(f"… {done}/{n_boot} done (latest ObjFun={out['ObjFun']:.4f})")
                else:
                    print(f"… {done}/{n_boot} done (latest failed: {out.get('error')})")

    df_boot = pd.DataFrame(results)
    df_ok = df_boot.dropna(subset=["k1", "k2", "vmax_in", "vmax_out", "ObjFun"])
    summary = df_ok[["k1", "k2", "vmax_in", "vmax_out", "ObjFun"]].agg(["mean", "std"]).T
    summary = summary.rename(columns={"mean": "Mean", "std": "SD"})

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("bootstrap_outputs_clustered")
    out_dir.mkdir(parents=True, exist_ok=True)
    boot_csv = out_dir / f"boot_replicates_{method}_cluster{cluster_label}_{ts}.csv"
    summary_csv = out_dir / f"boot_summary_{method}_cluster{cluster_label}_{ts}.csv"
    df_boot.to_csv(boot_csv, index=False)
    summary.to_csv(summary_csv)

    # KDE distribution figures 
    params_to_plot = ["k1", "k2", "vmax_in", "vmax_out", "ObjFun"]
    for p in params_to_plot:
        if p in df_ok.columns:
            title = f"Distribution of {p} for Cluster {cluster_label} from {method}"
            fig_path = out_dir / _safe_filename(title)
            _save_kde_curve(df_ok[p], title, fig_path)

    print(f"\n{method} Cluster {cluster_label} finished (N={n_boot}).")
    print(f"   Replicates saved: {boot_csv}")
    print(f"   Summary saved:    {summary_csv}")

    # summary (mean ± SD)
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

    # Load clusters and keep only IDs remaining after filters
    cls = pd.read_csv(CLUSTERS_CSV)
    if "PatientID" not in cls or "Cluster" not in cls:
        raise ValueError("CLUSTERS_CSV must contain columns: PatientID, Cluster")
    cls["PatientID"] = cls["PatientID"].astype(str).map(_normalize_id)
    present = set(df["ID"].dropna().unique().tolist())
    cls = cls[cls["PatientID"].isin(present)].copy()

    cluster_labels = sorted(map(str, cls["Cluster"].unique().tolist()))
    print(f"Clusters found (after filters): {cluster_labels}")

    # Build cluster -> patient ID list (after filters)
    cluster_ids = {
        lab: cls.loc[cls["Cluster"].astype(str) == lab, "PatientID"].tolist()
        for lab in cluster_labels
    }

    # Cores
    n_procs = max(1, cpu_count() - 1)

    # Run all methods × clusters
    for method in ["LS", "DE", "BO"]:
        if method not in STARTS_TRUE:
            print(f"[warn] No seeds provided for method {method}; skipping.")
            continue
        for lab in cluster_labels:
            seed_map = STARTS_TRUE[method]
            if lab not in seed_map:
                seed_true = [
                    np.sqrt(LS_BOUNDS_LO[0] * LS_BOUNDS_HI[0]),
                    np.sqrt(LS_BOUNDS_LO[1] * LS_BOUNDS_HI[1]),
                    np.sqrt(LS_BOUNDS_LO[2] * LS_BOUNDS_HI[2]),
                    np.sqrt(LS_BOUNDS_LO[3] * LS_BOUNDS_HI[3]),
                ]
                print(f"[warn] No seed for {method} cluster {lab}; using bounds-center fallback.")
            else:
                seed_true = seed_map[lab]

            ids_list = cluster_ids.get(lab, [])
            if len(ids_list) == 0:
                print(f"[warn] Cluster {lab} has 0 patients after filters; skipping.")
                continue

            run_one(method, lab, ids_list, seed_true, N_BOOT, n_procs)