import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import itertools
from math import inf

from patient import PatientExtended
from pbpk_fitting_utils import simulate_patient_concentration, get_infusion_schedule


PBPK_XLSX = "Copy of mmc1[66].xlsx"
CLUSTERS_CSV = "patient_7group_classification.csv"

# bounds for true params [k1, k2, vin, vout]
BOUNDS_LO = [1e-6, 1e-6, 1e-8, 1e-8]
BOUNDS_HI = [1.0,   1.0,   1e-2,  1e-2]

# grid settings (in true param space)
COARSE_POINTS_PER_DIM = 6        # 6 → 6^4 = 1296 combos
REFINE_FACTOR = 5.0              # ±5× around best seed
REFINE_POINTS_PER_DIM = 5        # 5 → 5^4 = 625 combos (local)
TOP_SEEDS_COARSE = 6             # take top seeds from coarse grid
TOP_SEEDS_REFINE = 4             # take top from local refined grid
LS_STARTS = 5                    # how many seeds to actually run LS on

# subset strategy
USE_FIRST_TWO_THIRDS_FOR_SEED = True   # pick subset for seeding
USE_FULL_FOR_LS = False                # False => LS on first 2/3 only


LAMBDA_PAIR = 0.02   
LAMBDA_ORDER = 0.00  

# solver settings
X_SCALE_TRUE = [0.01, 0.1, 1e-3]  # scales for [k1, k2, vin]; beta uses 1.0
FTOL = 1e-10
XTOL = 1e-10
GTOL = 1e-10
MAX_NFEV = 2000

# reparam settings
BETA_MAX = 30.0            # allow extreme gaps; maps to frac ≈ e^{-BETA_MAX}
VOUT_MIN_FRAC = 1e-6       # ensure vout >= vin + (HI-vin)*VOUT_MIN_FRAC in seeds/grids

EPS = 1e-9

# Utilities
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


# enforce vout >= vin
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def map_theta_to_true(theta):
    """
    theta = [k1, k2, vin, beta]  ->  true params [k1, k2, vin, vout]
    vout = vin + (HI_vout - vin) * sigmoid(beta)  ∈ [vin, HI_vout]
    """
    k1, k2, vin, beta = theta
    vin = np.clip(vin, BOUNDS_LO[2], BOUNDS_HI[2])
    gap_max = max(BOUNDS_HI[3] - vin, 0.0)
    vout = vin + gap_max * sigmoid(beta)
    k1 = np.clip(k1, BOUNDS_LO[0], BOUNDS_HI[0])
    k2 = np.clip(k2, BOUNDS_LO[1], BOUNDS_HI[1])
    return np.array([k1, k2, vin, vout], dtype=float)

def seed_true_to_theta(seed_true):
    """
    Convert a seed in true space [k1,k2,vin,vout] to theta=[k1,k2,vin,beta]
    beta = logit((vout - vin)/(HI_vout - vin)), clipped to [-BETA_MAX, BETA_MAX].
    """
    k1, k2, vin, vout = [float(x) for x in seed_true]
    vin = np.clip(vin, BOUNDS_LO[2], BOUNDS_HI[2])

    # ensure seed respects vout >= vin + tiny gap
    hi = BOUNDS_HI[3]
    min_gap = (hi - vin) * VOUT_MIN_FRAC
    vout = float(np.clip(vout, vin + min_gap, hi))

    gap_max = max(hi - vin, 1e-12)
    frac = (vout - vin) / gap_max  # in (0,1]

    # Clip to beta range
    frac_min = 1.0 / (1.0 + np.exp(BETA_MAX))         # ≈ e^{-BETA_MAX}
    frac_max = 1.0 - frac_min
    frac = np.clip(frac, frac_min, frac_max)

    beta = np.log(frac / (1.0 - frac))
    return np.array([k1, k2, vin, beta], dtype=float)


def build_patient(row):
    serum_creatinine = float(row['serum creatinine']) * 0.0113
    hematocrit = float(row['hematocrit']) / 100.0
    return PatientExtended(row['Gender: 1, male; 2, female'],
                           row['Body weight'], row['Height'], row['Age'],
                           hematocrit, serum_creatinine)

# residuals (linear, finite-safe)
def linear_residuals_true(params_true, df, patient_ids):
    k1, k2, vmax_in, vmax_out = params_true
    all_res = []

    for pid in patient_ids:
        pat_df = df[df["ID"] == pid]
        if pat_df.empty or pat_df["Concentration"].isna().all():
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

            c_exp = obs_df["Concentration"].to_numpy()[mask_time]

            with np.errstate(all='ignore'):
                c_sim, _, _ = simulate_patient_concentration(
                    pat, infusions, t_obs, k1, k2, vmax_in, vmax_out
                )

            mask_fin = np.isfinite(c_sim) & np.isfinite(c_exp)
            if not mask_fin.any():
                continue

            c_sim = c_sim[mask_fin]
            c_exp = c_exp[mask_fin]

            # pure concentration RMSE: no time weighting
            w_patient = 1.0 / np.sqrt(len(c_sim))
            r_lin = (c_sim - c_exp) * w_patient
            all_res.append(r_lin)

        except Exception as e:
            print(f"[warn] Patient {pid} skipped (linear) due to error: {e}")
            continue

    if not all_res:
        return np.ones(10, dtype=float) * 1e3
    return np.concatenate(all_res)

def penalty_residuals_true(params_true, lam_pair=LAMBDA_PAIR):
    """Keep k1 and k2 in a similar order of magnitude (gentle)."""
    k1, k2, _, _ = params_true
    eps = 1e-12
    r_pair = np.log((k1+eps)/(k2+eps))
    return np.array([lam_pair * r_pair], dtype=float)

# LS optimises theta
def group_residuals_theta(theta, df, patient_ids, _unused):
    params_true = map_theta_to_true(theta)
    res = linear_residuals_true(params_true, df, patient_ids)
    pen = penalty_residuals_true(params_true)
    return np.concatenate([res, pen])

# RMSE in concentration space
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
        # enforce a tiny positive gap in grid ranking
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
    # start vout slightly above vin to avoid equal starts
    vout_lo = max(vin + (BOUNDS_HI[3]-vin)*VOUT_MIN_FRAC, BOUNDS_LO[3])
    vout_hi = max(vout_lo * 1.000001, min(BOUNDS_HI[3], vout*factor))
    vout_grid= np.geomspace(vout_lo, vout_hi, points)
    return k1_grid, k2_grid, vin_grid, vout_grid

# Fit per patient set (optimise in THETA space; linear residuals)

def fit_cluster(df, ids_all):
    if len(ids_all) == 0:
        raise ValueError("Empty patient ID list for cluster.")

    # subset for seeding
    if USE_FIRST_TWO_THIRDS_FOR_SEED and len(ids_all) > 3:
        n_subset = max(3, int(len(ids_all) * 2/3))
        ids_seed = ids_all[:n_subset]
    else:
        ids_seed = ids_all

    # choose which set to use for LS
    ids_for_ls = ids_all if USE_FULL_FOR_LS else ids_seed
    print(f"[fit] seeding on n={len(ids_seed)}; LS on n={len(ids_for_ls)}")

    # 1) coarse grid on ids_seed (rank by conc-RMSE in true space)
    coarse = grid_search_init(df, ids_seed, max_candidates=TOP_SEEDS_COARSE)
    best_seed_true = coarse[0][0]

    # 2) refined local grid around best seed (true space, still on ids_seed)
    k1g, k2g, ving, voutg = refine_grid_around(best_seed_true)
    refined = grid_search_init(df, ids_seed,
                               k1_grid=k1g, k2_grid=k2g, vin_grid=ving, vout_grid=voutg,
                               max_candidates=TOP_SEEDS_REFINE)

    # 3) candidate seeds (a few refined + a couple coarse) - in TRUE space
    seed_pool_true = refined[:min(3, len(refined))] + coarse[:max(0, LS_STARTS - 3)]

    # 4) multi-start LS in THETA space on ids_for_ls
    best_res, best_score = None, inf
    for i, (seed_true, grid_score) in enumerate(seed_pool_true, 1):
        theta0 = seed_true_to_theta(seed_true)  # convert to theta
        print(f"\n[LS start {i}] seed_true {seed_true} (grid conc-RMSE={grid_score:.6g})")
        res_i = least_squares(
            group_residuals_theta, theta0,
            # bounds in THETA space: [k1,k2,vin,beta]
            bounds=(
                [BOUNDS_LO[0], BOUNDS_LO[1], BOUNDS_LO[2], -BETA_MAX],
                [BOUNDS_HI[0], BOUNDS_HI[1], BOUNDS_HI[2],  BETA_MAX]
            ),
            loss="soft_l1",
            f_scale=0.3,
            args=(df, ids_for_ls, PatientExtended),
            method="trf",
            x_scale=X_SCALE_TRUE + [1.0],   # add scale for beta
            ftol=FTOL, xtol=XTOL, gtol=GTOL,
            max_nfev=MAX_NFEV,
            verbose=0
        )
        params_true_i = map_theta_to_true(res_i.x)
        res_score = conc_rmse_true(params_true_i, df, ids_for_ls)
        print(f"[LS start {i}] done. conc-RMSE={res_score:.6g}, params_true={params_true_i}")

        if res_score < best_score:
            best_res, best_score = res_i, res_score

    # 5) final evaluation and return (report TRUE params)
    best_params_true = map_theta_to_true(best_res.x)
    final_conc_rmse = conc_rmse_true(best_params_true, df, ids_for_ls)
    return {
        "params": best_params_true,
        "conc_rmse": final_conc_rmse,
        "n_used_for_ls": len(ids_for_ls),
        "status": int(best_res.status),
        "message": str(best_res.message)
    }

if __name__ == "__main__":
    df = pd.read_excel(PBPK_XLSX, header=0, skiprows=[1])
    assert_columns(df, ["ID","Concentration","Date","Time","Gender: 1, male; 2, female",
                        "Body weight","Height","Age","hematocrit","serum creatinine"], "PBPK Excel")
    df["ID"] = df["ID"].apply(normalize_id)

    cls = pd.read_csv(CLUSTERS_CSV)
    assert_columns(cls, ["PatientID","Final_Group"], "Clustering CSV")
    cls["PatientID"] = cls["PatientID"].apply(normalize_id)

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
            result = fit_cluster(df, ids_in_cluster)
        except Exception as e:
            print(f"[error] Cluster {cl} failed: {e}")
            continue

        p = result["params"]
        print(f"[Cluster {cl}] params = {p}")  # p[2]=vin, p[3]=vout guaranteed vout>=vin
        print(f"[Cluster {cl}] conc-RMSE = {result['conc_rmse']:.6g} (n_used_for_ls={result['n_used_for_ls']})")
        summaries.append({
            "Cluster": cl,
            "n_patients": len(ids_in_cluster),
            "n_used_for_ls": result["n_used_for_ls"],
            "k1": p[0], "k2": p[1], "vmax_in": p[2], "vmax_out": p[3],
            "conc_RMSE": result["conc_rmse"],
            "status": result["status"],
            "message": result["message"]
        })

    if summaries:
        out = pd.DataFrame(summaries)
        out.to_csv("class_fit_summary.csv", index=False)
        print("\nSaved class-wise results to cluster_fit_summary.csv")
        print(out)
    else:
        print("\nNo class produced results. Check logs for ID mismatches or data issues.")