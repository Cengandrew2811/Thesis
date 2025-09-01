import numpy as np
import pandas as pd
import itertools
from math import inf, log10

from patient import PatientExtended
from pbpk_fitting_utils import simulate_patient_concentration, get_infusion_schedule

from scipy.stats.qmc import Sobol
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern,
    WhiteKernel,
    ConstantKernel,
)

import warnings
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("default", category=ConvergenceWarning)
    warnings.filterwarnings("default", message="lbfgs failed to converge")
except Exception:
    pass

PBPK_XLSX = "Copy of mmc1[66].xlsx"
CLUSTERS_CSV = "patient_7group_classification.csv"

# bounds for TRUE params [k1, k2, vin, vout]
BOUNDS_LO = np.array([1e-6, 1e-6, 1e-8, 1e-8], dtype=float)
BOUNDS_HI = np.array([1.0,   1.0,   1e-2,  1e-2], dtype=float)

# grid settings (in true param space)
COARSE_POINTS_PER_DIM = 6        # 6 → 6^4 = 1296 combos
REFINE_FACTOR = 5.0              # ±5× around best seed
REFINE_POINTS_PER_DIM = 5        # 5 → 5^4 = 625 combos (local)
TOP_SEEDS_COARSE = 6             # take top seeds from coarse grid
TOP_SEEDS_REFINE = 4             # take top from local refined grid

# subset strategy
USE_FIRST_TWO_THIRDS_FOR_SEED = True   # pick subset for seeding/BO
EVAL_ALL_AT_END = True                 # also report RMSE on all patients

# penalties (keep small; set to 0.0 to disable)
LAMBDA_PAIR = 0.02   # encourages k1 ~ k2 via |log(k1/k2)|

# BO settings
SOBOL_INIT = 32                 # initial points (power of two recommended)
N_BO_ITERS = 100
XI_FRAC = 0.01                  # EI exploration parameter as fraction of |best|
EI_DE_MAXITER = 80              # DE steps to optimize EI
EI_DE_POP = 12
EI_DE_TOL = 1e-3
RANDOM_SEED = 42

# z-box around seed (local BO box)
# for log10 dims use ±log10(REFINE_FACTOR). for beta use ±BETA_WINDOW.
BETA_MAX = 30.0
BETA_WINDOW = 6.0

# tiny constants
EPS = 1e-12
VOUT_MIN_FRAC = 1e-6  # ensure vout >= vin + (HI-vin)*VOUT_MIN_FRAC in seeds/grids

def normalize_id(x):
    """Normalize IDs so '001', 1, '1.0' match."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s2 = s.lstrip("0")
    return s if s2 == "" else s2

def assert_columns(df, cols, df_name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing columns: {missing}\nAvailable: {list(df.columns)}")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def map_theta_to_true(theta):
    """
    theta = [k1, k2, vin, beta]  ->  true params [k1, k2, vin, vout]
    vout = vin + (HI_vout - vin) * sigmoid(beta)  ∈ [vin, HI_vout]
    """
    k1, k2, vin, beta = [float(v) for v in theta]
    k1 = float(np.clip(k1, BOUNDS_LO[0], BOUNDS_HI[0]))
    k2 = float(np.clip(k2, BOUNDS_LO[1], BOUNDS_HI[1]))
    vin = float(np.clip(vin, BOUNDS_LO[2], BOUNDS_HI[2]))
    gap_max = max(BOUNDS_HI[3] - vin, 0.0)
    vout = vin + gap_max * sigmoid(beta)
    return np.array([k1, k2, vin, vout], dtype=float)

def seed_true_to_theta(seed_true):
    """
    Convert seed TRUE [k1,k2,vin,vout] -> THETA=[k1,k2,vin,beta]
    beta = logit((vout - vin)/(HI_vout - vin)), clipped.
    """
    k1, k2, vin, vout = [float(x) for x in seed_true]
    vin = np.clip(vin, BOUNDS_LO[2], BOUNDS_HI[2])
    hi = BOUNDS_HI[3]
    # enforce tiny positive gap for seeds
    min_gap = (hi - vin) * VOUT_MIN_FRAC
    vout = float(np.clip(vout, vin + min_gap, hi))

    gap_max = max(hi - vin, 1e-12)
    frac = (vout - vin) / gap_max  # in (0,1]
    # clip to safe region to keep beta finite
    frac_min = 1.0 / (1.0 + np.exp(BETA_MAX))
    frac_max = 1.0 - frac_min
    frac = np.clip(frac, frac_min, frac_max)
    beta = np.log(frac / (1.0 - frac))
    return np.array([k1, k2, vin, beta], dtype=float)

def true_to_z(seed_true):
    """TRUE -> z=[log10(k1), log10(k2), log10(vin), beta]"""
    k1, k2, vin, vout = [float(x) for x in seed_true]
    beta = seed_true_to_theta([k1, k2, vin, vout])[3]
    return np.array([log10(k1), log10(k2), log10(vin), beta], dtype=float)

def z_to_true(z):
    """z=[log10 k1, log10 k2, log10 vin, beta] -> TRUE [k1,k2,vin,vout]"""
    z = np.asarray(z, dtype=float)
    k1 = 10.0 ** z[0]
    k2 = 10.0 ** z[1]
    vin = 10.0 ** z[2]
    beta = z[3]
    return map_theta_to_true([k1, k2, vin, beta])


def build_patient(row):
    serum_creatinine = float(row['serum creatinine']) * 0.0113
    hematocrit = float(row['hematocrit']) / 100.0
    return PatientExtended(row['Gender: 1, male; 2, female'],
                           row['Body weight'], row['Height'], row['Age'],
                           hematocrit, serum_creatinine)

def conc_rmse_true(params_true, df, patient_ids):
    k1, k2, vmax_in, vmax_out = params_true
    all_obs, all_sim = [], []

    for pid in patient_ids:
        pat_df = df[df["ID"] == pid]
        if pat_df.empty:
            continue
        try:
            pat = build_patient(pat_df.iloc[0])
            infusions, start_time = get_infusion_schedule(pat_df)
            if (not infusions) or (start_time is None) or (pd.isna(start_time)):
                continue

            obs_df = pat_df.dropna(subset=["Concentration", "Date", "Time"]).copy()
            obs_df["dt"] = pd.to_datetime(
                obs_df["Date"].astype(str).str.strip() + " " +
                obs_df["Time"].astype(str).str.strip(),
                errors="coerce"
            ).astype("datetime64[ns]")
            obs_df = obs_df.dropna(subset=["dt"])

            t_obs = (obs_df["dt"] - start_time).dt.total_seconds().to_numpy()
            if t_obs is None:
                continue
            with np.errstate(all='ignore'):
                t_obs = t_obs / 3600.0
            mask_time = np.isfinite(t_obs)
            if not mask_time.any():
                continue
            t_obs = np.maximum(t_obs[mask_time], 0.0)

            c_obs = obs_df["Concentration"].to_numpy()[mask_time]

            with np.errstate(all='ignore'):
                c_sim, _, _ = simulate_patient_concentration(
                    pat, infusions, t_obs, k1, k2, vmax_in, vmax_out
                )

            mask_fin = np.isfinite(c_sim) & np.isfinite(c_obs)
            if not mask_fin.any():
                continue

            all_obs.append(c_obs[mask_fin])
            all_sim.append(c_sim[mask_fin])

        except Exception:
            continue

    if not all_obs or not all_sim:
        return 1e9

    all_obs = np.concatenate(all_obs)
    all_sim = np.concatenate(all_sim)
    return float(np.sqrt(np.mean((all_sim - all_obs) ** 2)))

def pair_penalty(params_true, lam=LAMBDA_PAIR):
    """Gentle penalty to keep k1 and k2 similar in order of magnitude."""
    k1, k2, _, _ = params_true
    r = abs(np.log((k1 + EPS) / (k2 + EPS)))
    return lam * float(r)

# Grid search 
def grid_search_init(df, patient_ids,
                     k1_grid=None, k2_grid=None, vin_grid=None, vout_grid=None,
                     max_candidates=3):
    if k1_grid is None:      k1_grid  = np.logspace(-6, -1, COARSE_POINTS_PER_DIM)
    if k2_grid is None:      k2_grid  = np.logspace(-6, -1, COARSE_POINTS_PER_DIM)
    if vin_grid is None:     vin_grid = np.logspace(-8, -3, COARSE_POINTS_PER_DIM)
    if vout_grid is None:    vout_grid= np.logspace(-8, -3, COARSE_POINTS_PER_DIM)

    results = []
    best_rmse, best_params = inf, None
    total = len(k1_grid)*len(k2_grid)*len(vin_grid)*len(vout_grid)
    print(f"[grid] evaluating {total} combos on {len(patient_ids)} patients...")

    for ix, (k1, k2, vin, vout) in enumerate(itertools.product(k1_grid, k2_grid, vin_grid, vout_grid), 1):
        # enforce tiny positive gap in ranking
        hi = BOUNDS_HI[3]
        min_gap = (hi - vin) * VOUT_MIN_FRAC
        vout_eff = max(vout, vin + min_gap)
        try:
            rmse = conc_rmse_true([k1, k2, vin, vout_eff], df, patient_ids)
        except Exception:
            rmse = 1e9

        if rmse < best_rmse:
            best_rmse, best_params = rmse, [k1, k2, vin, vout_eff]

        results.append((np.array([k1, k2, vin, vout_eff], dtype=float), rmse))

        if ix % 200 == 0 or ix == total:
            print(f"[grid] {ix}/{total} checked. best conc-RMSE={best_rmse:.6g} params={best_params}")

    if not results:
        seed = np.array([BOUNDS_LO[0], BOUNDS_LO[1], 1e-6, 1e-6*(1+VOUT_MIN_FRAC)], dtype=float)
        return [(seed, conc_rmse_true(seed, df, patient_ids))]

    results.sort(key=lambda t: t[1])
    return results[:max_candidates]

def refine_grid_around(seed_true, factor=REFINE_FACTOR, points=REFINE_POINTS_PER_DIM):
    k1, k2, vin, vout = seed_true
    k1_grid  = np.geomspace(max(BOUNDS_LO[0], k1/factor),  min(BOUNDS_HI[0], k1*factor),  points)
    k2_grid  = np.geomspace(max(BOUNDS_LO[1], k2/factor),  min(BOUNDS_HI[1], k2*factor),  points)
    vin_grid = np.geomspace(max(BOUNDS_LO[2], vin/factor), min(BOUNDS_HI[2], vin*factor), points)
    # start vout slightly above vin
    vout_lo = max(vin + (BOUNDS_HI[3]-vin)*VOUT_MIN_FRAC, BOUNDS_LO[3])
    vout_hi = max(vout_lo * 1.000001, min(BOUNDS_HI[3], vout*factor))
    vout_grid= np.geomspace(vout_lo, vout_hi, points)
    return k1_grid, k2_grid, vin_grid, vout_grid

def local_box_around_seed_z(seed_true, factor=REFINE_FACTOR, beta_window=BETA_WINDOW):
    zc = true_to_z(seed_true)  # [logk1, logk2, logvin, beta]
    half = np.array([np.log10(factor)]*3 + [beta_window], dtype=float)

    # hard z-bounds from TRUE bounds
    z_lo_hard = np.array([np.log10(BOUNDS_LO[0]),
                          np.log10(BOUNDS_LO[1]),
                          np.log10(BOUNDS_LO[2]),
                          -BETA_MAX], dtype=float)
    z_hi_hard = np.array([np.log10(BOUNDS_HI[0]),
                          np.log10(BOUNDS_HI[1]),
                          np.log10(BOUNDS_HI[2]),
                          +BETA_MAX], dtype=float)

    z_lo = np.maximum(zc - half, z_lo_hard)
    z_hi = np.minimum(zc + half, z_hi_hard)
    return z_lo, z_hi, zc

def objective_z(z, df, patient_ids):
    try:
        p_true = z_to_true(z)
        rmse = conc_rmse_true(p_true, df, patient_ids)
        pen = pair_penalty(p_true, lam=LAMBDA_PAIR)
        return float(rmse + pen)
    except Exception:
        return 1e9

def fit_cluster_bo(df, ids_all):
    if len(ids_all) == 0:
        raise ValueError("Empty patient ID list for cluster.")

    # choose seed subset
    if USE_FIRST_TWO_THIRDS_FOR_SEED and len(ids_all) > 3:
        n_subset = max(3, int(len(ids_all) * 2/3))
        ids_seed = ids_all[:n_subset]
    else:
        ids_seed = ids_all
    print(f"[fit] BO on first 2/3: n={len(ids_seed)}")

    # 1) coarse grid on seed set
    coarse = grid_search_init(df, ids_seed, max_candidates=TOP_SEEDS_COARSE)
    best_seed_true = coarse[0][0]

    # 2) refined local grid around best seed (still on seed set)
    k1g, k2g, ving, voutg = refine_grid_around(best_seed_true)
    refined = grid_search_init(df, ids_seed,
                               k1_grid=k1g, k2_grid=k2g, vin_grid=ving, vout_grid=voutg,
                               max_candidates=TOP_SEEDS_REFINE)

    # pick best refined seed (TRUE) and create local z-box
    seed_true = refined[0][0]
    z_lo, z_hi, z_center = local_box_around_seed_z(seed_true)

    # pretty-print local box
    print("[BO] local bounds (z):")
    print(f"   log10(k1) in [{z_lo[0]:.3f}, {z_hi[0]:.3f}]")
    print(f"   log10(k2) in [{z_lo[1]:.3f}, {z_hi[1]:.3f}]")
    print(f"   log10(vin) in [{z_lo[2]:.3f}, {z_hi[2]:.3f}]")
    print(f"   beta in [{z_lo[3]:.3f}, {z_hi[3]:.3f}]")

    # 3) initial design via Sobol in the local z-box
    dim = 4
    # force SOBOL_INIT to power-of-two
    m = max(1, int(np.round(np.log2(max(SOBOL_INIT, 4)))))
    n_init = 2 ** m
    sob = Sobol(d=dim, scramble=True, seed=RANDOM_SEED)
    U = sob.random_base2(m=m)  # unit-cube samples
    Z = z_lo + U * (z_hi - z_lo)

    # include the refined best seed center as well
    Z = np.vstack([z_center, Z])
    Y = np.array([objective_z(z, df, ids_seed) for z in Z], dtype=float)

    X = Z.copy()   # z-space points

    def to_unit(z):   return (z - z_lo) / (z_hi - z_lo)
    def from_unit(u): return z_lo + u * (z_hi - z_lo)

    X_u = np.vstack([to_unit(z) for z in X])

    # kernel tuned for unit-cube inputs
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
        length_scale=np.ones(4), length_scale_bounds=(1e-2, 50.0), nu=2.5
    ) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=8,
        alpha=1e-8,
        random_state=RANDOM_SEED,
    )
    gp.fit(X_u, Y)

    # EI defined on unit cube
    from scipy.stats import norm
    def ei_neg_u(u, gp, y_best, xi):
        u = np.atleast_2d(u)
        mu, sigma = gp.predict(u, return_std=True)
        sigma = np.maximum(sigma, 1e-12)
        imp = y_best - mu - xi
        Zs = imp / sigma
        ei = imp * norm.cdf(Zs) + sigma * norm.pdf(Zs)
        return -ei

    for it in range(1, N_BO_ITERS + 1):
        y_best = float(Y.min())
        xi = max(1e-3, XI_FRAC * abs(y_best))

        result_ei = differential_evolution(
            func=lambda uu: ei_neg_u(uu, gp, y_best, xi),
            bounds=[(0.0, 1.0)] * 4,
            strategy="best1bin",
            maxiter=EI_DE_MAXITER,
            popsize=EI_DE_POP,
            tol=EI_DE_TOL,
            polish=True,
            updating="deferred",
            workers=1,
            seed=RANDOM_SEED,
        )
        u_next = result_ei.x
        z_next = from_unit(u_next)
        y_next = objective_z(z_next, df, ids_seed)

        X = np.vstack([X, z_next])
        X_u = np.vstack([X_u, u_next])
        Y = np.append(Y, y_next)

        gp.fit(X_u, Y)
        print(f"[BO] iter {it}/{N_BO_ITERS} — best(obj)={Y.min():.4f}")

    # 4) best on seed set
    ib = int(np.argmin(Y))
    z_best = X[ib]
    params_best_true = z_to_true(z_best)
    best_seed_obj = float(Y[ib])

    # 5) evaluate on all patients (optional report)
    if EVAL_ALL_AT_END:
        rmse_all = conc_rmse_true(params_best_true, df, ids_all)
        print(f"[Cluster ?] conc-RMSE (all patients) = {rmse_all:.4f}")

    return {
        "params": params_best_true,
        "best_obj_seed": best_seed_obj,
        "z_best": z_best,
    }


if __name__ == "__main__":
    # --- Load PBPK dataset ---
    df = pd.read_excel(PBPK_XLSX, header=0, skiprows=[1])
    assert_columns(df, ["ID","Concentration","Date","Time","Gender: 1, male; 2, female",
                        "Body weight","Height","Age","hematocrit","serum creatinine"], "PBPK Excel")
    df["ID"] = df["ID"].apply(normalize_id)

    # --- Load clustering (PatientID, Cluster) ---
    cls = pd.read_csv(CLUSTERS_CSV)
    assert_columns(cls, ["PatientID","Final_Group"], "Clustering CSV")
    cls["PatientID"] = cls["PatientID"].apply(normalize_id)

    # --- Align to common IDs ---
    present_ids = set(df["ID"].dropna().unique())
    before = len(cls)
    cls = cls[cls["PatientID"].isin(present_ids)].copy()
    dropped = before - len(cls)
    if dropped > 0:
        print(f"[info] Dropped {dropped} clustered rows with IDs not found in PBPK file.")

    cluster_labels = sorted(cls["Final_Group"].unique())
    if len(cluster_labels) == 0:
        raise ValueError("No clusters after ID matching. Check ID formats.")

    print(f"Clusters found: {cluster_labels}")

    summaries = []
    for cl in cluster_labels:
        ids_in_cluster = cls.loc[cls["Final_Group"] == cl, "PatientID"].dropna().tolist()
        print(f"\n=== Cluster {cl} : {len(ids_in_cluster)} patients ===")
        if len(ids_in_cluster) == 0:
            print(f"[warn] Cluster {cl} empty after matching. Skipping.")
            continue

        try:
            result = fit_cluster_bo(df, ids_in_cluster)
        except Exception as e:
            print(f"[error] Cluster {cl} failed: {e}")
            continue

        p = result["params"]
        print(f"[Cluster {cl}] params = {p}  # [k1, k2, vmax_in, vmax_out] (vout≥vin enforced)")
        print(f"[Cluster {cl}] conc-RMSE = {conc_rmse_true(p, df, ids_in_cluster):.4f}  (n_used={max(3, int(len(ids_in_cluster)*2/3) if USE_FIRST_TWO_THIRDS_FOR_SEED else len(ids_in_cluster))})")
        if EVAL_ALL_AT_END:
            print(f"[Cluster {cl}] conc-RMSE (all patients) = {conc_rmse_true(p, df, ids_in_cluster):.4f}")
        print(f"[Cluster {cl}] best objective (seed set) = {result['best_obj_seed']:.4f}")

        summaries.append({
            "Cluster": cl,
            "n_patients": len(ids_in_cluster),
            "k1": p[0], "k2": p[1], "vmax_in": p[2], "vmax_out": p[3],
            "best_obj_seed": result["best_obj_seed"],
        })

    if summaries:
        out = pd.DataFrame(summaries)
        out.to_csv("class_BO_summary.csv", index=False)
        print("\nSaved class-wise results to class_fit_summary.csv")
        print(out)
    else:
        print("\nNo class produced results. Check logs for ID mismatches or data issues.")