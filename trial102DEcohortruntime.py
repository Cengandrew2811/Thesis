import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import itertools
from math import inf
import time


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from patient import PatientExtended
from pbpk_fitting_utils import simulate_patient_concentration, get_infusion_schedule


PBPK_XLSX = "Copy of mmc1[66].xlsx"

# Choose which patients to optimise over:
USE_FILTERS = True
EXCLUDE_IDS = {"580.1", "613.3", "614.1"}  # only used if USE_FILTERS=True
MIN_AGE = 18

# bounds for true params [k1, k2, vin, vout]
BOUNDS_LO = [1e-6, 1e-6, 1e-8, 1e-8]
BOUNDS_HI = [1.0,   1.0,   1e-2,  1e-2]

# grid settings (in true param space)
COARSE_POINTS_PER_DIM = 6        # 6 → 6^4 = 1296 combos
REFINE_FACTOR = 5.0              # ±5× around best seed
REFINE_POINTS_PER_DIM = 5        # 5 → 5^4 = 625 combos (local)
TOP_SEEDS_COARSE = 6             # take top seeds from coarse grid
TOP_SEEDS_REFINE = 4             # take top from local refined grid

# Reparam: vout >= vin via sigmoid(beta)
BETA_MAX = 30.0                  # wide range; maps frac≈e^{-BETA_MAX}
VOUT_MIN_FRAC = 1e-6             # tiny positive gap at seeding

# Penalty (keep small)
LAMBDA_PAIR = 0.02               # soft prior: k1 ~ k2

# Differential Evolution settings
DE_STRATEGY = "randtobest1bin"
DE_MAXITER  = 200
DE_TOL      = 1e-5
DE_POPSIZE  = 15                 # population size multiplier
DE_WORKERS  = -1                 # use all cores; set to 1 on Windows if needed
DE_DISP     = True
DE_POLISH   = True               # optional local polish at the end

EPS = 1e-12


def normalize_id(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.endswith(".0"):  # Excel-y IDs like "123.0"
        s = s[:-2]
    s2 = s.lstrip("0")
    return s if s2 == "" else s2

def assert_columns(df, cols, df_name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing columns: {missing}\nAvailable: {list(df.columns)}")


def sigmoid(x):  # stable enough for our ranges
    return 1.0 / (1.0 + np.exp(-x))

def map_theta_to_true(theta):
    """
    theta = [k1, k2, vin, beta]  ->  true params [k1, k2, vin, vout]
    vout in [vin, HI_vout]
    """
    k1, k2, vin, beta = theta
    vin = float(np.clip(vin, BOUNDS_LO[2], BOUNDS_HI[2]))
    hi_v = BOUNDS_HI[3]
    gap_max = max(hi_v - vin, 0.0)
    vout = vin + gap_max * float(sigmoid(beta))
    k1 = float(np.clip(k1, BOUNDS_LO[0], BOUNDS_HI[0]))
    k2 = float(np.clip(k2, BOUNDS_LO[1], BOUNDS_HI[1]))
    return np.array([k1, k2, vin, vout], dtype=float)

def seed_true_to_theta(seed_true):
    """
    Convert a true-space seed [k1,k2,vin,vout] -> theta=[k1,k2,vin,beta]
    """
    k1, k2, vin, vout = [float(x) for x in seed_true]
    vin = np.clip(vin, BOUNDS_LO[2], BOUNDS_HI[2])
    hi = BOUNDS_HI[3]
    # ensure seed respects a tiny positive gap
    min_gap = (hi - vin) * VOUT_MIN_FRAC
    vout = float(np.clip(vout, vin + min_gap, hi))
    gap_max = max(hi - vin, 1e-12)
    frac = np.clip((vout - vin) / gap_max, 1e-12, 1-1e-12)
    beta = float(np.log(frac / (1.0 - frac)))
    return np.array([k1, k2, vin, beta], dtype=float)


def build_patient(row):
    serum_creatinine = float(row['serum creatinine']) * 0.0113  # µmol/L → mg/dL
    hematocrit = float(row['hematocrit']) / 100.0
    return PatientExtended(row['Gender: 1, male; 2, female'],
                           row['Body weight'], row['Height'], row['Age'],
                           hematocrit, serum_creatinine)

def conc_rmse_true(params_true, df, patient_ids):
    """RMSE in concentration space across patients (objective core)."""
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
                obs_df["Date"].astype(str).str.strip() + " " +
                obs_df["Time"].astype(str).str.strip(),
                errors="coerce"
            ).astype("datetime64[ns]")
            obs_df = obs_df.dropna(subset=["dt"])
            if obs_df.empty:
                continue
            t_obs = (obs_df["dt"] - start_time).dt.total_seconds().to_numpy()
            if t_obs is None:
                continue
            t_obs = t_obs / 3600.0
            mask_time = np.isfinite(t_obs)
            if not mask_time.any():
                continue
            t_obs = np.maximum(t_obs[mask_time], 0.0)
            c_obs = obs_df["Concentration"].to_numpy()[mask_time]
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

def objective_theta(theta, df, patient_ids, lam_pair=LAMBDA_PAIR):
    """DE objective in θ-space: conc-RMSE(true) + tiny penalty on |log(k1/k2)|."""
    params_true = map_theta_to_true(theta)
    rmse = conc_rmse_true(params_true, df, patient_ids)
    # tiny regulariser to keep k1 and k2 similar magnitude
    k1, k2, _, _ = params_true
    pen = lam_pair * abs(np.log((k1+EPS)/(k2+EPS)))
    return rmse + pen


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
        # enforce a tiny gap for ranking
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
    # start vout slightly above vin
    vout_lo  = max(vin + (BOUNDS_HI[3]-vin)*VOUT_MIN_FRAC, BOUNDS_LO[3])
    vout_hi  = max(vout_lo * 1.000001, min(BOUNDS_HI[3], vout*factor))
    vout_grid= np.geomspace(vout_lo, vout_hi, points)
    return k1_grid, k2_grid, vin_grid, vout_grid


if __name__ == "__main__":
    # --- Load PBPK dataset ---
    df = pd.read_excel(PBPK_XLSX, header=0, skiprows=[1])
    assert_columns(df, ["ID","Concentration","Date","Time","Gender: 1, male; 2, female",
                        "Body weight","Height","Age","hematocrit","serum creatinine"], "PBPK Excel")
    df["ID"] = df["ID"].apply(normalize_id)

    if USE_FILTERS:
        df = df[~df["ID"].isin(EXCLUDE_IDS)].copy()
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df = df[df["Age"] >= MIN_AGE].copy()

    # patient set (use entire cohort by default)
    all_ids = sorted(df["ID"].dropna().astype(str).unique().tolist())
    N = len(all_ids)
    n_train = max(3, int(np.ceil(N * 2/3)))
    patient_ids = all_ids[:n_train]
    print(f"Optimising over {len(patient_ids)} / {N} patients (first 2/3 after filters).")

    # Coarse grid 
    coarse = grid_search_init(df, patient_ids, max_candidates=TOP_SEEDS_COARSE)
    best_seed_true = coarse[0][0]
    print(f"[grid] best coarse seed: {best_seed_true}  (RMSE={coarse[0][1]:.6g})")

    # Refined local grid 
    k1g, k2g, ving, voutg = refine_grid_around(best_seed_true)
    refined = grid_search_init(df, patient_ids,
                               k1_grid=k1g, k2_grid=k2g, vin_grid=ving, vout_grid=voutg,
                               max_candidates=TOP_SEEDS_REFINE)
    print(f"[grid] top refined seeds:")
    for s, r in refined:
        print("   ", s, "RMSE=", r)

    # Build DE initial population from seeds 
    seed_thetas = []
    for seed_true, _ in (refined + coarse):
        th = seed_true_to_theta(seed_true)
        # clip theta to DE bounds just in case
        th[0] = np.clip(th[0], BOUNDS_LO[0], BOUNDS_HI[0])
        th[1] = np.clip(th[1], BOUNDS_LO[1], BOUNDS_HI[1])
        th[2] = np.clip(th[2], BOUNDS_LO[2], BOUNDS_HI[2])
        th[3] = np.clip(th[3], -BETA_MAX, BETA_MAX)
        seed_thetas.append(th)

    if seed_thetas:
        init_pop = np.unique(np.array(seed_thetas), axis=0)
    else:
        init_pop = None  # let DE LHS-init

    # Differential Evolution in θ-space 
    bounds_theta = [
        (BOUNDS_LO[0], BOUNDS_HI[0]),   # k1
        (BOUNDS_LO[1], BOUNDS_HI[1]),   # k2
        (BOUNDS_LO[2], BOUNDS_HI[2]),   # vin
        (-BETA_MAX, BETA_MAX),          # beta
    ]

    # Runtime
    trace = []  # rows: {"gen", "cum_seconds", "best_objective_so_far"}
    # Start timer immediately before DE begins
    t0 = time.perf_counter()
    gen_counter = {"g": 0}  # mutable counter for callback closure

    def de_callback(xk, convergence):
        """
        Called once per generation in the main process.
        We log cumulative runtime vs best objective so far.
        """
        gen_counter["g"] += 1
        # Compute objective at current best xk (one extra eval per generation).
        y = objective_theta(xk, df, patient_ids)
        elapsed = time.perf_counter() - t0
        best_so_far = y if not trace else min(trace[-1]["best_objective_so_far"], y)
        trace.append({
            "gen": gen_counter["g"],
            "cum_seconds": elapsed,
            "best_objective_so_far": best_so_far
        })
        return False  # continue optimisation

    print("\n[DE] Starting differential evolution...")
    result = differential_evolution(
        func=objective_theta,                 # objective (no wrapper; safe with workers)
        args=(df, patient_ids),
        bounds=bounds_theta,
        strategy=DE_STRATEGY,
        maxiter=DE_MAXITER,
        tol=DE_TOL,
        popsize=DE_POPSIZE,
        init=init_pop if init_pop is not None else "latinhypercube",
        updating="deferred",
        polish=DE_POLISH,
        workers=DE_WORKERS,                   # multiprocessing OK; callback runs in main proc
        disp=DE_DISP,
        seed=42,
        callback=de_callback,                 # <-- log per generation
    )

    # Total DE runtime (excludes grid search)
    total_de_seconds = time.perf_counter() - t0
    print(f"[DE] Total DE runtime (excluding grid search): {total_de_seconds:.2f} s")


    best_theta = result.x
    best_params_true = map_theta_to_true(best_theta)
    best_obj = objective_theta(best_theta, df, patient_ids)
    best_rmse = conc_rmse_true(best_params_true, df, patient_ids)

    print("\n DE optimisation complete.")
    print(f"   Best θ        = {best_theta}")
    print(f"   Best params   = {best_params_true}  # [k1, k2, vmax_in, vmax_out] with vout≥vin enforced")
    print(f"   Best objective= {best_obj:.6g} (conc-RMSE + penalty)")
    print(f"   Best conc-RMSE= {best_rmse:.6g}")
    print(f"   DE message    = {result.message}")

    if not trace:
        trace = [{
            "gen": 0,
            "cum_seconds": total_de_seconds,
            "best_objective_so_far": float(best_obj),
        }]

    trace_df = pd.DataFrame(trace)
    trace_csv = "de_runtime_trace.csv"
    trace_png = "de_runtime_vs_objective.png"
    trace_df.to_csv(trace_csv, index=False)

    plt.figure()
    plt.plot(trace_df["cum_seconds"], trace_df["best_objective_so_far"])
    plt.xlabel("Cumulative runtime since DE start (s)")
    plt.ylabel("Best objective so far (conc-RMSE + penalty)")
    plt.title("Differential Evolution: Runtime vs Best Objective")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(trace_png, dpi=200)
    print(f"[DE] Saved runtime trace to {trace_csv} and figure to {trace_png}")

    out = pd.DataFrame([{
        "k1": best_params_true[0],
        "k2": best_params_true[1],
        "vmax_in": best_params_true[2],
        "vmax_out": best_params_true[3],
        "best_objective": best_obj,
        "conc_RMSE": best_rmse,
        "DE_message": str(result.message),
        "n_patients": len(patient_ids),
        "DE_runtime_seconds": total_de_seconds,
    }])
    out.to_csv("pbpk_de_cohort_fit.csv", index=False)
    print("Saved: pbpk_de_cohort_fit.csv")