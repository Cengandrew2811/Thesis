# Machine‑Learning–Aided PBPK Toolkit for HDMTX (Dissertation Code)

This repository contains the analysis code used in the dissertation *“Machine Learning Approaches to Patient Stratification and Population Parameter Optimisation in a Physiologically Based Pharmacokinetic Model of High‑Dose Methotrexate (HDMTX)”*. It implements a five‑compartment PBPK model, two patient‑stratification schemes, three complementary optimisers, and a patient‑level bootstrap to quantify uncertainty. 

> If you are reviewing or re‑running the dissertation results, start with **Reproduce the study** below.


---

## Overview (what this repo does)

- **PBPK model (5 compartments)** with plasma (CS), renal/hepatic circulations (RCS/HCS), interstitial fluid (ISF), and intracellular fluid (ICF); plasma concentration is the observable. Four transport/exchange parameters are estimated from plasma data while physiological volumes/flows and binding are fixed from literature. 
- **Patient stratification** via (i) clinical bins from eGFR (CKD‑EPI 2009) crossed with body‑weight tertiles (A–D consolidated), and (ii) **k‑means** in the Weight–Age–eGFR space (K chosen by elbow & silhouette heuristics). 
- **Optimisers**: bounded **Least Squares (LS)** with robust loss, **Differential Evolution (DE)**, and **Bayesian Optimisation (BO)** with a GP surrogate and EI acquisition; all seeded by a **coarse‑to‑fine grid search**. 
- **Uncertainty**: **patient‑level bootstrap** (case resampling) to obtain distributions/intervals for parameters and RMSE (default 500 replicates per stratum). 
- **Diagnostics**: objective heatmaps, observed‑vs‑predicted panels (train/validation), and dot‑and‑error summaries of bootstrap objectives. 
---

## Repo layout (scripts you’ll run)

```
# Core analysis pipeline
further_patient_classification.py     # eGFR (CKD‑EPI) + weight tertiles → classes A–D
kmeans_filtered.py                    # K‑means on [Weight, Age, eGFR], writes patient_clusterss.csv

BootstrapClass.py                     # Bootstraps within classes (A–D) using LS/DE/BO
BootstrapCluster.py                   # Bootstraps within K‑means clusters using LS/DE/BO
BootstrapCohort.py                    # Bootstraps on the full cohort

ObjectiveHeatMapsClassifications.py   # Objective maps around seeds (classes)
ObjectiveHeatMapsClustering.py        # Objective maps around seeds (clusters)

observedVpredictedCohort.py           # OVP train/validation plots (cohort)
observedVpredictedCluster.py          # OVP plots (by cluster)
obversedVpredictedClassified.py       # OVP plots (by class)  [filename as provided]

dotanderrorplot.py                    # Scans bootstrap CSVs → dot + error summaries

# PBPK model & utilities
patient.py                            # Patient physiology, CKD‑EPI eGFR, water‑volume mapping
pbpk_fitting_utils.py                 # Infusions, simulation wrappers, objective helpers

# Objectives / bounds / runtime experiments
trial101cluster.py                    # LS Cluster-level objective/bounds (v101)
trial101LSruntime.py                  # LS Cohort-level with timing utilities
trial102class.py                      # LS Class-level objective/bounds (v102)
trial102DEclass.py                    # DE objective/bounds for classes (v102)
trial102DEcluster.py                  # DE objective/bounds for clusters (v102)
trial102BOclass.py                    # BO objective/bounds for classes (v102)
trial102BOcluster2.py                 # BO objective/bounds for clusters (v102)
trial102BOcohortruntime.py            # BO cohort runtime experiments (v102)
trial102DEcohortruntime.py            # DE cohort runtime experiments (v102)
```
> Note: `obversedVpredictedClassified.py` is spelled as in the submitted code.
---

## Data expectations

- **Primary dataset**: Excel file `Copy of mmc1[66].xlsx` with at least  
  `ID`, `Body weight` (kg), `Height` (cm), `Gender: 1, male; 2, female`, `Age` (years), `hematocrit` (%), `serum creatinine` (µmol/L).  
  Some steps also rely on each patient’s infusion and plasma concentration time series referenced by the simulators.
- **Patient exclusions & filters**: adult‑only (e.g., `Age ≥ 18`) and known outlier IDs removed as defined in scripts.
- **Derived artefacts**: `patient_clusterss.csv` (from k‑means), and bootstrap CSVs under the output folders listed below.  
These settings match the data handling described in the dissertation (eGFR via **CKD‑EPI 2009**, weight tertiles, case‑level resampling). 
---

## Installation

Python ≥ 3.10 recommended.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install numpy pandas scipy matplotlib scikit-learn
```


---

## Reproduce the study (end‑to‑end)

1) **Stratify patients**
```bash
python further_patient_classification.py     # writes class labels A–D
python kmeans_filtered.py                    # writes patient_clusterss.csv
```

2) **Run bootstraps** (choose any or all scopes)
```bash
python BootstrapCohort.py                    # cohort-level
python BootstrapClass.py                     # by classes A–D
python BootstrapCluster.py                   # by K‑means clusters
```
- Each bootstrap script runs LS, DE, and BO with seeds from a coarse‑to‑fine grid, and saves per‑replicate CSVs (parameters, RMSE, metadata). 

3) **Diagnostics & figures**
```bash
python ObjectiveHeatMapsClassifications.py
python ObjectiveHeatMapsClustering.py

python observedVpredictedCohort.py
python observedVpredictedCluster.py
python obversedVpredictedClassified.py

python dotanderrorplot.py
```
- These scripts generate: **observed‑vs‑predicted** panels, **objective heatmaps**, and **dot‑and‑error** summaries used in the dissertation. 


---

## Key model details (for reviewers)

- **Model structure**: Five well‑mixed spaces (CS/RCS/HCS/ISF/ICF) with zero‑order IV infusion, renal & hepatic elimination, linear plasma↔interstitium exchange, and saturable (Michaelis–Menten) ISF↔ICF transport on unbound drug. 
- **Free parameters** (estimated):  
  `k_CS→ISF`, `k_ISF→CS` (linear exchange), and `k_ISF→ICF^sat`, `k_ICF→ISF^sat` (saturable transport). Bounds/constraints follow physiologic plausibility; the saturable pair is parameterised with an inequality to reflect transporter directionality. 
- **Physiology inputs**: flows/volumes, protein binding, and BMI‑based water compartments; renal filtration linked to **CKD‑EPI** eGFR. 
- **Objective**: per‑patient RMSE on plasma concentrations; cohort/stratum objective is the mean of patient RMSEs. 
- **Bootstrap**: case resampling within each stratum, 500 replicates by default, parallelised; report medians and percentile intervals. 


---

## Outputs (where to look)

- **Bootstrap CSVs**  
  `bootstrap_outputs_classified/`, `bootstrap_outputs_clustered/`, `bootstrap_outputs_cohort/`  
  (one row per replicate × optimiser with parameters, RMSE, and metadata).

- **Figures**  
  `Figures_Objective_Maps_Classified/`, `Figures_Objective_Maps_Clustered/`  
  `Figures_OVP_Cohort/`, `Figures_OVP_Cluster/`, `Figures_OVP_Classified/`  
  plus elbow/silhouette plots (k‑means) and a class‑distribution chart.

> In the dissertation, BO is typically best or tied on accuracy‑vs‑compute; k‑means partitions often yield tighter errors than clinical bins, and exchange rates are identifiable whereas cellular transport is weakly informed under the current sampling design. Use the objective maps and bootstrap intervals to interpret these findings. 

---

## Configuration knobs (edit near the top of scripts)

- Data paths: `DATA_XLSX`, `CLUSTERS_CSV`
- Filtering: `exclude_ids`, `MIN_AGE`
- Bootstrap: `N_BOOT` (default 500), `RANDOM_SEED`
- Optimisers: `STARTS_TRUE`, bounds from `trial10x*` modules; LS trust‑region settings; DE (`F`, `CR`, population size); BO iterations & GP kernel
- Plotting: output folders, log‑axes for OVP



---

## AUTHOR

> Rajendra Alhakim. *Machine Learning Approaches to Patient Stratification and Population Parameter Optimisation in a PBPK Model of High‑Dose Methotrexate*, Department of Chemical Engineering, Imperial College London, 2025.


