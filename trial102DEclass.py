import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from patient import PatientExtended
from pbpk_fitting_utils import simulate_patient_concentration, get_infusion_schedule

# =========================
# Config
# =========================
PBPK_XLSX   = "Copy of mmc1[66].xlsx"
CLUSTERS_CSV= "patient_7group_classification.csv"

# True-parameter bounds: [k1, k2, vmax_in, vmax_out]
BOUNDS_LO = [1e-6, 1e-6, 1e-8, 1e-8]
BOUNDS_HI = [1.0,  1.0,  1e-2, 1e-2]

# Grid settings (true space)
COARSE_POINTS_PER_DIM = 6   # 6^4 = 1296 combos
REFINE_FACTOR         = 5.0
REFINE_POINTS_PER_DIM = 5   # 5^4 = 625 combos
TOP_SEEDS_COARSE      = 6
TOP_SEEDS_REFINE      = 4

# DE settings (θ space)
BETA_MAX     = 30.0           # θ4=β ∈ [-BETA_MAX, BETA_MAX]
LAMBDA_PAIR  = 0.02           # tiny penalty to keep k1~k2
MAXITER      = 200
TOL          = 1e-4
USE_PARALLEL = True           # set False if multiprocessing is problematic on your OS

# Training subset rule
USE_FIRST_TWO_THIRDS = True   # use only the first 2/3 of each cluster for fitting

# Minimum fraction to force vout above vin in seeds/grids
VOUT_MIN_FRAC = 1e-6

EPS = 1e-12

# =========================
# Utilities
# =========================
def normalize_id(x):
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

def build_patient(row):
    # dataset serum creatinine is µmol/L -> mg/dL
    scr_mgdl = float(row["serum creatinine"]) * 0.011312
    return PatientExtended(
        gender=int(row["Gender: 1, male; 2, female"]),
        weight_kg=float(row["Body weight"]),
        height_cm=float(row["Height"]),
        age=float(row["Age"]),
        hematocrit=float(row["hematocrit"]) / 100.0,
        serum_creatinine=scr_mgdl,
    )

# =========================
# Reparameterisation (θ -> true params) to enforce vout >= vin
# =========================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def map_theta_to_true(theta):
    """
    theta = [k1, k2, vin, beta] -> true params [k1, k2, vin, vout]
    vout = vin + (HI_vout - vin) * sigmoid(beta) ∈ [vin, HI_vout]
    """
    k1, k2, vin, beta = theta
    k1 = np.clip(k1,  BOUNDS_LO[0], BOUNDS_HI[0])
    k2 = np.clip(k2,  BOUNDS_LO[1], BOUNDS_HI[1])
    vin= np.clip(vin, BOUNDS_LO[2], BOUNDS_HI[2])
    gap_max = max(BOUNDS_HI[3] - vin, 0.0)
    vout = vin + gap_max * sigmoid(beta)
    return np.array([k1, k2, vin, vout], dtype=float)

def seed_true_to_theta(seed_true):
    """
    Convert a seed in true space [k1,k2,vin,vout] to θ=[k1,k2,vin,beta].
    beta = logit((vout - vin)/(HI - vin)), clipped to [-BETA_MAX, BETA_MAX].
    """
    k1, k2, vin, vout = [float(x) for x in seed_true]
    vin = np.clip(vin, BOUNDS_LO[2], BOUNDS_HI[2])
    hi  = BOUNDS_HI[3]
    min_gap = (hi - vin) * VOUT_MIN_FRAC
    vout = float(np.clip(vout, vin + min_gap, hi))
    gap_max = max(hi - vin, EPS)
    frac = np.clip((vout - vin) / gap_max, EPS, 1.0 - EPS)
    beta = np.log(frac / (1.0 - frac))
    beta = float(np.clip(beta, -BETA_MAX, BETA_MAX))
    return np.array([k1, k2, vin, beta], dtype=float)

# =========================
# RMSE in concentration space (true params)
# =========================
def conc_rmse_true(params_true, df, patient_ids):
    k1, k2, vmax_in, vmax_out = params_true
    all_obs, all_sim = [], []

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
                obs_df["Date"].astype(str).str.strip() + " " + obs_df["Time"].astype(str).str.strip(),
                errors="coerce"
            )
            obs_df = obs_df.dropna(subset=["dt"])
            if obs_df.empty:
                continue

            t_obs = (obs_df["dt"] - start_time).dt.total_seconds().to_numpy() / 3600.0
            mask_time = np.isfinite(t_obs)
            if not mask_time.any():
                continue
            t_obs = np.maximum(t_obs[mask_time], 0.0)

            c_obs = pd.to_numeric(obs_df["Concentration"], errors="coerce").to_numpy()[mask_time]
            mask_fin = np.isfinite(c_obs)
            if not mask_fin.any():
                continue
            t_obs = t_obs[mask_fin]
            c_obs = c_obs[mask_fin]

            c_sim, _, _ = simulate_patient_concentration(pat, infusions, t_obs, k1, k2, vmax_in, vmax_out)
            mask_both = np.isfinite(c_sim) & np.isfinite(c_obs)
            if not mask_both.any():
                continue

            all_obs.append(c_obs[mask_both])
            all_sim.append(c_sim[mask_both])
        except Exception:
            continue

    if not all_obs:
        return 1e9

    all_obs = np.concatenate(all_obs)
    all_sim = np.concatenate(all_sim)
    return float(np.sqrt(np.mean((all_sim - all_obs) ** 2)))

# =========================
# Grid search (true space) to collect seeds
# =========================
def grid_search_init(df, patient_ids,
                     k1_grid=None, k2_grid=None, vin_grid=None, vout_grid=None,
                     max_candidates=TOP_SEEDS_COARSE):
    if k1_grid  is None: k1_grid  = np.logspace(-6, -1, COARSE_POINTS_PER_DIM)
    if k2_grid  is None: k2_grid  = np.logspace(-6, -1, COARSE_POINTS_PER_DIM)
    if vin_grid is None: vin_grid = np.logspace(-8, -3, COARSE_POINTS_PER_DIM)
    if vout_grid is None: vout_grid= np.logspace(-8, -3, COARSE_POINTS_PER_DIM)

    results = []
    total = len(k1_grid)*len(k2_grid)*len(vin_grid)*len(vout_grid)
    print(f"[grid] evaluating {total} combos on {len(patient_ids)} patients...")

    ix = 0
    best_rmse, best_params = np.inf, None
    for k1 in k1_grid:
        for k2 in k2_grid:
            for vin in vin_grid:
                for vout in vout_grid:
                    ix += 1
                    # ensure vout >= vin + tiny gap
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

    results.sort(key=lambda t: t[1])
    return results[:max_candidates]

def refine_grid_around(seed_true, factor=REFINE_FACTOR, points=REFINE_POINTS_PER_DIM):
    k1, k2, vin, vout = seed_true
    k1_grid  = np.geomspace(max(BOUNDS_LO[0], k1/factor),  min(BOUNDS_HI[0], k1*factor),  points)
    k2_grid  = np.geomspace(max(BOUNDS_LO[1], k2/factor),  min(BOUNDS_HI[1], k2*factor),  points)
    vin_grid = np.geomspace(max(BOUNDS_LO[2], vin/factor), min(BOUNDS_HI[2], vin*factor), points)
    vout_lo  = max(vin + (BOUNDS_HI[3] - vin)*VOUT_MIN_FRAC, BOUNDS_LO[3])
    vout_hi  = max(vout_lo * 1.000001, min(BOUNDS_HI[3], vout*factor))
    vout_grid= np.geomspace(vout_lo, vout_hi, points)
    return k1_grid, k2_grid, vin_grid, vout_grid

# =========================
# Differential Evolution objective (TOP LEVEL — picklable)
# =========================
def de_objective_theta(theta, df, patient_ids):
    """Minimise concentration-space RMSE (+ tiny pairing penalty)."""
    params_true = map_theta_to_true(theta)  # enforces vout >= vin
    rmse = conc_rmse_true(params_true, df, patient_ids)
    k1, k2 = params_true[0], params_true[1]
    pen = LAMBDA_PAIR * abs(np.log((k1 + EPS) / (k2 + EPS)))
    return rmse + pen

# =========================
# Fit a single cluster (train on first 2/3)
# =========================
def fit_cluster_de(df, ids_all):
    if len(ids_all) == 0:
        raise ValueError("Empty patient list.")

    # pick training subset (first 2/3), as requested
    if USE_FIRST_TWO_THIRDS and len(ids_all) > 3:
        n_train = max(3, int(np.ceil(len(ids_all) * 2/3)))
        ids_seed = ids_all[:n_train]
    else:
        ids_seed = ids_all

    print(f"[fit] training on n={len(ids_seed)} patients (first 2/3 of cluster)")

    # 1) Coarse grid (true space)
    coarse = grid_search_init(df, ids_seed, max_candidates=TOP_SEEDS_COARSE)
    best_seed_true = coarse[0][0]

    # 2) Refined grid around best seed (true space)
    k1g, k2g, ving, voutg = refine_grid_around(best_seed_true)
    refined = grid_search_init(df, ids_seed,
                               k1_grid=k1g, k2_grid=k2g, vin_grid=ving, vout_grid=voutg,
                               max_candidates=TOP_SEEDS_REFINE)

    # Build an initial population for DE in θ space (optional but helpful)
    # SciPy allows an array-like 'init' of shape (npop, ndim)
    seeds_true = [s for s,_ in refined] + [s for s,_ in coarse]
    init_theta = np.array([seed_true_to_theta(s) for s in seeds_true], dtype=float)

    # DE bounds in θ space: [k1, k2, vin, beta]
    bounds_theta = [
        (BOUNDS_LO[0], BOUNDS_HI[0]),
        (BOUNDS_LO[1], BOUNDS_HI[1]),
        (BOUNDS_LO[2], BOUNDS_HI[2]),
        (-BETA_MAX, BETA_MAX),
    ]

    print("[DE] starting DE for this cluster ...")
    result = differential_evolution(
        func=de_objective_theta,
        args=(df, ids_seed),                 # pass data via args (picklable)
        bounds=bounds_theta,
        strategy='randtobest1bin',
        init=init_theta,                     # seed population (or 'latinhypercube')
        updating='deferred',                 # better with workers > 1
        polish=True,
        maxiter=MAXITER,
        tol=TOL,
        workers=(-1 if USE_PARALLEL else 1), # multiprocessing-safe (no lambdas)
        disp=True,
    )

    best_params_true = map_theta_to_true(result.x)
    final_rmse = conc_rmse_true(best_params_true, df, ids_seed)

    return {
        "params": best_params_true,
        "conc_rmse": final_rmse,
        "n_used_for_fit": len(ids_seed),
        "nit": int(result.nit),
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message),
    }

# =========================
# Main: cluster-wise, use only first 2/3 per cluster
# =========================
if __name__ == "__main__":
    # Load PBPK Excel
    df = pd.read_excel(PBPK_XLSX, header=0, skiprows=[1])
    assert_columns(df, ["ID","Concentration","Date","Time","Gender: 1, male; 2, female",
                        "Body weight","Height","Age","hematocrit","serum creatinine"], "PBPK Excel")
    df["ID"] = df["ID"].apply(normalize_id)

    # Load clusters
    cls = pd.read_csv(CLUSTERS_CSV)
    assert_columns(cls, ["PatientID","Final_Group"], "Clustering CSV")
    cls["PatientID"] = cls["PatientID"].apply(normalize_id)

    # Align to common IDs
    present_ids = set(df["ID"].dropna().unique())
    before = len(cls)
    cls = cls[cls["PatientID"].isin(present_ids)].copy()
    dropped = before - len(cls)
    if dropped > 0:
        print(f"[info] Dropped {dropped} clustered rows with IDs not found in PBPK file.")

    clusters = sorted(cls["Final_Group"].unique())
    if len(clusters) == 0:
        raise ValueError("No clusters after ID matching.")

    print(f"Clusters found: {clusters}")

    rows = []
    for cl in clusters:
        ids_in_cluster = cls.loc[cls["Final_Group"] == cl, "PatientID"].dropna().tolist()
        print(f"\n=== Cluster {cl} : {len(ids_in_cluster)} patients ===")
        if len(ids_in_cluster) == 0:
            print(f"[warn] Cluster {cl} empty. Skipping.")
            continue

        try:
            res = fit_cluster_de(df, ids_in_cluster)
        except Exception as e:
            print(f"[error] Cluster {cl} failed: {e}")
            continue

        p = res["params"]
        print(f"[Cluster {cl}] params = {p}")  # [k1, k2, vmax_in, vmax_out] with vout≥vin
        print(f"[Cluster {cl}] conc-RMSE = {res['conc_rmse']:.6g}  (n_used_for_fit={res['n_used_for_fit']})")
        rows.append({
            "Cluster": cl,
            "n_patients": len(ids_in_cluster),
            "n_used_for_fit": res["n_used_for_fit"],
            "k1": p[0], "k2": p[1], "vmax_in": p[2], "vmax_out": p[3],
            "conc_RMSE": res["conc_rmse"],
            "nit": res["nit"],
            "nfev": res["nfev"],
            "success": res["success"],
            "message": res["message"],
        })

    if rows:
        out = pd.DataFrame(rows)
        out.to_csv("de_class_fit_summary.csv", index=False)
        print("\nSaved class-wise DE results to de_class_fit_summary.csv")
        print(out)
    else:
        print("\nNo clusters produced DE results. Check logs.")