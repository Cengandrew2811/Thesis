import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import itertools
from math import inf
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from patient import PatientExtended
from pbpk_fitting_utils import simulate_patient_concentration, get_infusion_schedule


PBPK_XLSX = "Copy of mmc1[66].xlsx"

EXCLUDE_IDS = {"580.1", "613.3", "614.1"}
MIN_AGE = 18

# bounds for true params [k1, k2, vin, vout]
BOUNDS_LO = [1e-6, 1e-6, 1e-8, 1e-8]
BOUNDS_HI = [1.0,   1.0,   1e-2,  1e-2]

COARSE_POINTS_PER_DIM = 6
REFINE_FACTOR = 5.0
REFINE_POINTS_PER_DIM = 5
TOP_SEEDS_COARSE = 6
TOP_SEEDS_REFINE = 4
LS_STARTS = 5

X_SCALE_TRUE = [0.01, 0.1, 1e-3]
FTOL = 1e-10
XTOL = 1e-10
GTOL = 1e-10
MAX_NFEV = 2000

BETA_MAX = 30.0
VOUT_MIN_FRAC = 1e-6
LAMBDA_PAIR = 0.02
EPS = 1e-9


# Utilities

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

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def map_theta_to_true(theta):
    k1, k2, vin, beta = theta
    vin = np.clip(vin, BOUNDS_LO[2], BOUNDS_HI[2])
    gap_max = max(BOUNDS_HI[3] - vin, 0.0)
    vout = vin + gap_max * sigmoid(beta)
    k1 = np.clip(k1, BOUNDS_LO[0], BOUNDS_HI[0])
    k2 = np.clip(k2, BOUNDS_LO[1], BOUNDS_HI[1])
    return np.array([k1, k2, vin, vout], dtype=float)

def seed_true_to_theta(seed_true):
    k1, k2, vin, vout = [float(x) for x in seed_true]
    vin = np.clip(vin, BOUNDS_LO[2], BOUNDS_HI[2])
    hi = BOUNDS_HI[3]
    min_gap = (hi - vin) * VOUT_MIN_FRAC
    vout = float(np.clip(vout, vin + min_gap, hi))
    gap_max = max(hi - vin, 1e-12)
    frac = (vout - vin) / gap_max
    frac_min = 1.0 / (1.0 + np.exp(BETA_MAX))
    frac_max = 1.0 - frac_min
    frac = np.clip(frac, frac_min, frac_max)
    beta = np.log(frac / (1.0 - frac))
    return np.array([k1, k2, vin, beta], dtype=float)

def build_patient(row):
    scr = float(row['serum creatinine']) * 0.0113
    hct = float(row['hematocrit']) / 100.0
    return PatientExtended(row['Gender: 1, male; 2, female'],
                           row['Body weight'], row['Height'], row['Age'],
                           hct, scr)

# Residuals & RMSE

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
            if (not infusions) or (start_time is None):
                continue
            obs_df = pat_df.dropna(subset=["Concentration", "Date", "Time"]).copy()
            obs_df["dt"] = pd.to_datetime(
                obs_df["Date"].astype(str).str.strip() + " " +
                obs_df["Time"].astype(str).str.strip(),
                errors="coerce"
            )
            obs_df = obs_df.dropna(subset=["dt"])
            t_obs = (obs_df["dt"] - start_time).dt.total_seconds().to_numpy() / 3600.0
            mask = np.isfinite(t_obs)
            if not mask.any():
                continue
            t_obs = np.maximum(t_obs[mask], 0.0)
            c_exp = obs_df["Concentration"].to_numpy()[mask]
            c_sim, _, _ = simulate_patient_concentration(pat, infusions, t_obs, k1, k2, vmax_in, vmax_out)
            mask2 = np.isfinite(c_sim) & np.isfinite(c_exp)
            if not mask2.any():
                continue
            w = 1.0 / np.sqrt(len(c_sim))
            all_res.append((c_sim[mask2] - c_exp[mask2]) * w)
        except Exception:
            continue
    if not all_res:
        return np.ones(10) * 1e3
    return np.concatenate(all_res)

def penalty_residuals_true(params_true, lam=LAMBDA_PAIR):
    k1, k2, _, _ = params_true
    r = np.log((k1+EPS)/(k2+EPS))
    return np.array([lam * r])

def group_residuals_theta(theta, df, patient_ids, _unused=None):
    params_true = map_theta_to_true(theta)
    res = linear_residuals_true(params_true, df, patient_ids)
    pen = penalty_residuals_true(params_true)
    return np.concatenate([res, pen])

def conc_rmse_true(params_true, df, patient_ids):
    k1, k2, vmax_in, vmax_out = params_true
    all_obs, all_sim = [], []
    for pid in patient_ids:
        pat_df = df[df["ID"] == pid]
        if pat_df.empty: continue
        try:
            pat = build_patient(pat_df.iloc[0])
            infusions, start_time = get_infusion_schedule(pat_df)
            if (not infusions) or (start_time is None): continue
            obs_df = pat_df.dropna(subset=["Concentration", "Date", "Time"]).copy()
            obs_df["dt"] = pd.to_datetime(obs_df["Date"].astype(str)+" "+obs_df["Time"].astype(str), errors="coerce")
            obs_df = obs_df.dropna(subset=["dt"])
            t_obs = (obs_df["dt"] - start_time).dt.total_seconds().to_numpy()/3600.0
            mask = np.isfinite(t_obs)
            if not mask.any(): continue
            t_obs = np.maximum(t_obs[mask], 0.0)
            c_obs = obs_df["Concentration"].to_numpy()[mask]
            c_sim, _, _ = simulate_patient_concentration(pat, infusions, t_obs, k1, k2, vmax_in, vmax_out)
            mask2 = np.isfinite(c_sim) & np.isfinite(c_obs)
            if not mask2.any(): continue
            all_obs.append(c_obs[mask2])
            all_sim.append(c_sim[mask2])
        except Exception:
            continue
    if not all_obs: return 1e9
    return float(np.sqrt(np.mean((np.concatenate(all_sim) - np.concatenate(all_obs))**2)))

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
    for ix,(k1,k2,vin,vout) in enumerate(itertools.product(k1_grid,k2_grid,vin_grid,vout_grid),1):
        hi = BOUNDS_HI[3]
        min_gap = (hi - vin) * VOUT_MIN_FRAC
        vout_eff = max(vout, vin + min_gap)
        try: rmse = conc_rmse_true([k1,k2,vin,vout_eff], df, patient_ids)
        except: rmse = 1e9
        results.append((np.array([k1,k2,vin,vout_eff]), rmse))
        if ix % 200 == 0 or ix==total:
            print(f"[grid] {ix}/{total} best RMSE={min(r[1] for r in results):.6g}")
    results.sort(key=lambda t:t[1])
    return results[:max_candidates]

def refine_grid_around(seed_true, factor=REFINE_FACTOR, points=REFINE_POINTS_PER_DIM):
    k1,k2,vin,vout = seed_true
    k1g=np.geomspace(max(BOUNDS_LO[0],k1/factor),min(BOUNDS_HI[0],k1*factor),points)
    k2g=np.geomspace(max(BOUNDS_LO[1],k2/factor),min(BOUNDS_HI[1],k2*factor),points)
    ving=np.geomspace(max(BOUNDS_LO[2],vin/factor),min(BOUNDS_HI[2],vin*factor),points)
    vout_lo=max(vin+(BOUNDS_HI[3]-vin)*VOUT_MIN_FRAC,BOUNDS_LO[3])
    vout_hi=max(vout_lo*1.000001,min(BOUNDS_HI[3],vout*factor))
    voutg=np.geomspace(vout_lo,vout_hi,points)
    return k1g,k2g,ving,voutg

# =========================
# Main
# =========================
if __name__=="__main__":
    df=pd.read_excel(PBPK_XLSX,header=0,skiprows=[1])
    assert_columns(df,["ID","Concentration","Date","Time","Gender: 1, male; 2, female",
                       "Body weight","Height","Age","hematocrit","serum creatinine"],"PBPK Excel")
    df["ID"]=df["ID"].apply(normalize_id)

    # Exclusions
    df=df[~df["ID"].isin(EXCLUDE_IDS)].copy()
    df["Age"]=pd.to_numeric(df["Age"],errors="coerce")
    df=df[df["Age"]>=MIN_AGE].copy()

    patient_ids=df["ID"].dropna().unique().tolist()
    N=len(patient_ids)
    n_train=int(np.ceil(N*2/3))
    ids_train=patient_ids[:n_train]
    print(f"Total {N} patients after filtering. Using first 2/3 = {n_train} patients.")

    # Coarse + refined grid
    coarse=grid_search_init(df,ids_train,max_candidates=TOP_SEEDS_COARSE)
    best_seed_true=coarse[0][0]
    k1g,k2g,ving,voutg=refine_grid_around(best_seed_true)
    refined=grid_search_init(df,ids_train,k1g,k2g,ving,voutg,max_candidates=TOP_SEEDS_REFINE)

    # ===== Multi-start Nonlinear Least Squares (runtime trace excludes grid search) =====
    seed_pool=refined[:min(3,len(refined))]+coarse[:max(0,LS_STARTS-3)]

    # Runtime tracing
    trace = []  # rows with: start_index, eval_in_start, cum_seconds, objective_at_eval, best_objective_so_far
    best_objective_global = [np.inf]  # mutable for closures
    t0 = time.perf_counter()  # start AFTER grid search

    def make_wrapped_residuals(start_index):
        eval_counter = {"n": 0}
        def wrapped_residuals(theta):
            res = group_residuals_theta(theta, df, ids_train)
            obj = float(np.dot(res, res))  # LS objective = sum(residuals^2)
            eval_counter["n"] += 1
            elapsed = time.perf_counter() - t0
            if obj < best_objective_global[0]:
                best_objective_global[0] = obj
            trace.append({
                "start_index": start_index,
                "eval_in_start": eval_counter["n"],
                "cum_seconds": elapsed,
                "objective_at_eval": obj,
                "best_objective_so_far": best_objective_global[0],
            })
            return res
        return wrapped_residuals

    # Run starts
    best_score, best_params = inf, None
    for i,(seed_true,grid_rmse) in enumerate(seed_pool,1):
        theta0=seed_true_to_theta(seed_true)
        res=least_squares(
            make_wrapped_residuals(i), theta0,
            bounds=([BOUNDS_LO[0],BOUNDS_LO[1],BOUNDS_LO[2],-BETA_MAX],
                    [BOUNDS_HI[0],BOUNDS_HI[1],BOUNDS_HI[2], BETA_MAX]),
            loss="soft_l1", f_scale=0.3,
            method="trf", x_scale=X_SCALE_TRUE+[1.0],
            ftol=FTOL, xtol=XTOL, gtol=GTOL, max_nfev=MAX_NFEV
        )
        p_true=map_theta_to_true(res.x)
        rmse=conc_rmse_true(p_true,df,ids_train)
        print(f"[LS {i}] seed {seed_true}, RMSE={rmse:.6g}, params={p_true}")
        if rmse<best_score:
            best_score, best_params=rmse,p_true

    total_ls_seconds = time.perf_counter() - t0
    print(f"[LS] Total LS runtime (excluding grid search): {total_ls_seconds:.2f} s")

    # ---------- Save LS runtime trace (CSV + PNG) ----------
    if not trace:
        trace = [{
            "start_index": 0,
            "eval_in_start": 0,
            "cum_seconds": total_ls_seconds,
            "objective_at_eval": float("nan"),
            "best_objective_so_far": float("nan"),
        }]

    trace_df = pd.DataFrame(trace)
    trace_csv = "ls_runtime_trace.csv"
    trace_png = "ls_runtime_vs_objective.png"
    trace_df.to_csv(trace_csv, index=False)

    plt.figure()
    plt.plot(trace_df["cum_seconds"], trace_df["best_objective_so_far"])
    plt.xlabel("Cumulative runtime since LS start (s)")
    plt.ylabel("Best objective so far (∑ residuals²)")
    plt.title("Multi-start Nonlinear Least Squares: Runtime vs Best Objective")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(trace_png, dpi=200)
    print(f"[LS] Saved runtime trace to {trace_csv} and figure to {trace_png}")

    # ---------- Final report ----------
    print("\n✅ Finished cohort optimisation.")
    print(f"Best params = {best_params}  # [k1,k2,vin,vout], vout≥vin enforced")
    print(f"Best conc-RMSE (train 2/3) = {best_score:.6g}")

    # Save results summary
    out = pd.DataFrame([{
        "k1": best_params[0],
        "k2": best_params[1],
        "vmax_in": best_params[2],
        "vmax_out": best_params[3],
        "best_conc_RMSE": best_score,
        "best_objective_so_far": best_objective_global[0],
        "n_patients": len(ids_train),
        "LS_runtime_seconds": total_ls_seconds,
    }])
    out.to_csv("pbpk_ls_cohort_fit.csv", index=False)
    print("💾 Saved: pbpk_ls_cohort_fit.csv")