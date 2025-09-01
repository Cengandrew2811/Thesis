import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ===================== PATHS =====================
DATA_DIR_GROUPS    = Path("/Users/rajendraalhakim/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Documents/MSc Dissertation/THESIS/bootstrap_outputs_classified")
DATA_DIR_CLUSTERS  = Path("/Users/rajendraalhakim/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Documents/MSc Dissertation/THESIS/bootstrap_outputs_clustered")
DATA_DIR_COHORT    = Path("/Users/rajendraalhakim/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Documents/MSc Dissertation/THESIS/bootstrap_outputs_cohort")

# ===================== CONFIG ====================
OBJ_COL  = "ObjFun"                   # column name in CSVs (case-insensitive)
methods  = ["LS", "DE", "BO"]
groups   = ["classa", "classb", "classc", "classd"]   # keep lowercase internally
clusters = [f"cluster{i}" for i in range(5)]          # change to range(1,6) if your clusters are 1..5

# now accepts "..._cohort_..."
fname_re = re.compile(
    r"^boot_replicates_(?P<method>BO|DE|LS)_(?P<tag>class[A-D]|cluster[0-9]+|cohort)_.+\.csv$",
    re.IGNORECASE
)

# Colors (matches matplotlib default cycler hues) — used by dot+error plots
method_color = {"LS": "C0", "DE": "C1", "BO": "C2"}

# Pastel palette just for the violin plots
violin_fill = {  # soft fills
    "LS": "#A3C4F3",  # periwinkle
    "DE": "#B9FBC0",  # mint
    "BO": "#FFD6A5",  # peach
}
violin_edge = {  # slightly darker edges for definition
    "LS": "#5C7FB7",
    "DE": "#6EBF83",
    "BO": "#E3A857",
}

# ===================== IO HELPERS =================
def load_objfun(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    col = next((c for c in df.columns if c.lower() == OBJ_COL.lower()), None)
    if col is None:
        raise ValueError(f"'{OBJ_COL}' not found in {csv_path.name}")
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        raise ValueError(f"No numeric values in '{OBJ_COL}' for {csv_path.name}")
    return s

def scan_dir(folder: Path, kind: str):
    """kind ∈ {'group','cluster','cohort'}"""
    recs = []
    for p in folder.glob("boot_replicates_*.csv"):
        m = fname_re.match(p.name)
        if not m:
            continue
        method = m.group("method").upper()
        tag = m.group("tag").lower()  # normalize (classa, cluster0, cohort)
        if kind == "group"   and not tag.startswith("class"):   continue
        if kind == "cluster" and not tag.startswith("cluster"): continue
        if kind == "cohort"  and tag != "cohort":               continue
        try:
            recs.append({
                "method": method,
                "label": tag,
                "kind": kind,
                "series": load_objfun(p),
                "source": str(p),
            })
        except Exception as e:
            print(f"[WARN] {p.name}: {e}")
    return recs

# ===================== SUMMARIES ==================
def summarize(series: pd.Series, mode: str):
    if mode == "IQR":
        center = series.median()
        q25, q75 = series.quantile([0.25, 0.75])
        return {"center": center, "err_lo": center - q25, "err_hi": q75 - center,
                "low": q25, "high": q75, "metric": "median_IQR", "n": len(series)}
    if mode == "SD":
        center = series.mean()
        sd = series.std(ddof=1)
        return {"center": center, "err_lo": sd, "err_hi": sd,
                "sd": sd, "metric": "mean_SD", "n": len(series)}
    raise ValueError("mode must be 'IQR' or 'SD'")

def build_rows(records, mode: str) -> pd.DataFrame:
    rows = []
    def fetch(kind, label, method):
        for r in records:
            if r["kind"] == kind and r["label"] == label and r["method"] == method:
                return r
        return None

    # ---- Cohort first (Cohort, LS/DE/BO)
    for m in methods:
        r = fetch("cohort", "cohort", m)
        if r is None:
            print("[INFO] missing:", ("cohort", "cohort", m))
            continue
        stats = summarize(r["series"], mode)
        rows.append({"display": f"Cohort, {m}",
                     "method": m, **stats, "source": r["source"]})

    # ---- Groups A..D (each LS, DE, BO)
    for g in groups:
        for m in methods:
            r = fetch("group", g, m)
            if r is None:
                print("[INFO] missing:", ("group", g, m))
                continue
            stats = summarize(r["series"], mode)
            rows.append({"display": f"Group {g[-1].upper()}, {m}",
                         "method": m, **stats, "source": r["source"]})

    # ---- Clusters 0..4 (each LS, DE, BO)
    for c in clusters:
        for m in methods:
            r = fetch("cluster", c, m)
            if r is None:
                print("[INFO] missing:", ("cluster", c, m))
                continue
            stats = summarize(r["series"], mode)
            rows.append({"display": f"Cluster {c.replace('cluster','')}, {m}",
                         "method": m, **stats, "source": r["source"]})
    return pd.DataFrame(rows)

# ===================== DOT+ERROR PLOTS ============
def make_dot_error_plot(df: pd.DataFrame, out_path: Path, title_suffix: str):
    y = np.arange(len(df))
    x = df["center"].to_numpy()
    xerr = np.vstack([df["err_lo"].to_numpy(), df["err_hi"].to_numpy()])
    colors = [method_color[m] for m in df["method"]]

    fig, ax = plt.subplots(figsize=(9, 14))  # a hair taller for many rows
    for xi, yi, err_lo, err_hi, cc in zip(x, y, xerr[0], xerr[1], colors):
        ax.errorbar([xi], [yi], xerr=[[err_lo], [err_hi]], fmt='o',
                    color=cc, ecolor=cc, elinewidth=2, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df["display"])
    ax.set_xlabel("RMSE")
    ax.set_title(f"RMSE by cohort, group and cluster across methods ({title_suffix})")
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    handles = [Line2D([0], [0], marker='o', linestyle='None',
                      color=method_color[m], label=m) for m in methods]
    leg = ax.legend(handles=handles, title="Method",
                    loc="lower right", ncol=len(methods),
                    frameon=True, fancybox=True, framealpha=0.9,
                    borderpad=0.4, handletextpad=0.6, columnspacing=1.0)
    leg.get_frame().set_edgecolor('0.3')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.3)
    print(f"Saved: {out_path.resolve()}")

# ===================== VIOLIN PLOTS ===============
def collect_violin_data(recs, labels_order, kind, include_cohort: bool):
    """Return (datasets, labels, methods_for_each_violin)."""
    def get_series(method, label, want_kind):
        for r in recs:
            if r["kind"] == want_kind and r["label"] == label and r["method"] == method:
                return r["series"]
        return None

    data, labels, meths = [], [], []

    # prepend cohort triplet if requested
    if include_cohort:
        for m in methods:
            s = get_series(m, "cohort", "cohort")
            if s is None:
                print("[INFO] missing:", "cohort", "cohort", m)
                data.append(pd.Series(dtype=float))
            else:
                data.append(s)
            labels.append(f"Cohort, {m}")
            meths.append(m)

    labfmt = (lambda k: f"Group {k[-1].upper()}") if kind == "group" \
             else (lambda k: f"Cluster {k.replace('cluster','')}")

    for key in labels_order:
        for m in methods:
            s = get_series(m, key, kind)
            if s is None:
                print("[INFO] missing:", kind, key, m)
                data.append(pd.Series(dtype=float))
            else:
                data.append(s)
            labels.append(f"{labfmt(key)}, {m}")
            meths.append(m)
    return data, labels, meths

def make_violins(recs, out_path_groups: Path, out_path_clusters: Path):
    # collect with cohort triplets
    data_g, labels_g, meths_g = collect_violin_data(recs, groups,   kind="group",   include_cohort=True)
    data_c, labels_c, meths_c = collect_violin_data(recs, clusters, kind="cluster", include_cohort=True)

    # shared y-lims across both figs
    all_vals = np.concatenate([s.to_numpy() for s in (data_g + data_c) if not s.empty]) \
               if any(not s.empty for s in (data_g + data_c)) else np.array([0.0, 1.0])
    ymin, ymax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    base_range = max(1e-9, ymax - ymin)
    base_pad = 0.05 * base_range
    ylims = (ymin - base_pad, ymax + base_pad)

    def _plot(datasets, labels, methods_for, outfile, title):
        N = len(datasets)
        positions = np.arange(1, N + 1)
        fig, ax = plt.subplots(figsize=(max(12, N * 0.5), 5))

        # draw violins
        vp = ax.violinplot([d.values for d in datasets], positions=positions, showmedians=True)

        # pastel, cute violins 💖
        for i, body in enumerate(vp['bodies']):
            m = methods_for[i]
            face = violin_fill.get(m, "#DAD7FE")  # fallback lilac
            edge = violin_edge.get(m, "#8E8AFA")
            body.set_facecolor(face)
            body.set_edgecolor(edge)
            body.set_alpha(0.95)
            body.set_linewidth(1.6)

        # make the median and extrema crisp and dark for contrast
        if 'cmedians' in vp:
            vp['cmedians'].set_color('#2F2F2F')
            vp['cmedians'].set_linewidth(1.8)
        for k in ('cbars', 'cmins', 'cmaxes'):
            if k in vp:
                vp[k].set_color('#6E6E6E')
                vp[k].set_linewidth(1.2)

        # soft background + grid
        ax.set_facecolor('#FCFCFF')

        # stats for labels
        means = [np.nanmean(d.values) if d.size > 0 else np.nan for d in datasets]
        sds   = [np.nanstd(d.values, ddof=1) if d.size > 1 else np.nan for d in datasets]
        tops  = [np.nanmax(d.values) if d.size > 0 else np.nan for d in datasets]

        # headroom so labels don't clip
        top_of_all = np.nanmax(tops) if np.any(~np.isnan(tops)) else ylims[1]
        HEADROOM = 0.25  # 25% of data range above tallest violin
        ax.set_ylim(ylims[0], max(ylims[1], top_of_all + HEADROOM * base_range))

        # labels above violins (mean±SD), rotated
        for x, mu, sd, top in zip(positions, means, sds, tops):
            if np.isnan(mu) or np.isnan(top): 
                continue
            ax.annotate(
                f"{mu:.2f}±{sd:.2f}" if not np.isnan(sd) else f"{mu:.2f}",
                xy=(x, top), xytext=(0, 10), textcoords="offset points",
                ha="center", va="bottom", rotation=50, rotation_mode="anchor",
                fontsize=9, clip_on=False
            )

        # axes & legend
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=50, ha="right")
        ax.set_ylabel("RMSE")
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        # pastel legend matching the violins
        handles = [
            Line2D([0], [0], marker='s', linestyle='None',
                   markerfacecolor=violin_fill[m], markeredgecolor=violin_edge[m],
                   markersize=10, label=m)
            for m in methods
        ]
        leg = ax.legend(handles=handles, title="Method",
                        loc="upper right", ncol=len(methods),
                        frameon=True, fancybox=True, framealpha=0.95,
                        borderpad=0.4, handletextpad=0.6, columnspacing=1.0)
        leg.get_frame().set_edgecolor('0.3')

        fig.tight_layout()
        fig.savefig(outfile, dpi=200, bbox_inches="tight", pad_inches=0.3)
        print(f"Saved: {Path(outfile).resolve()}")

    _plot(data_g, labels_g, meths_g, out_path_groups,   "RMSE distributions — Groups (LS / DE / BO)")
    _plot(data_c, labels_c, meths_c, out_path_clusters, "RMSE distributions — Clusters (LS / DE / BO)")

# ================= SAVE PARAMETER SUMMARY TABLES (mean/SD etc.) =============

def _collect_param_stats(recs, kind: str, labels_order, colname: str,
                         include_cohort=True) -> pd.DataFrame:
    """
    Build a tidy table with one row per (Base, Method):
      Base = Cohort or Group A..D / Cluster 0..4
      Method = LS / DE / BO
      Stats = mean, sd, median, q25, q75, n
    """
    def _fetch(kind_here, label_here, method_here):
        for r in recs:
            if r["kind"] == kind_here and r["label"] == label_here and r["method"] == method_here:
                return r
        return None

    rows = []

    def _push_row(base_disp, method, series: pd.Series):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return
        med = float(np.nanmedian(s))
        q25, q75 = np.nanpercentile(s, [25, 75])
        mu  = float(np.nanmean(s))
        sd  = float(np.nanstd(s, ddof=1)) if s.size > 1 else np.nan
        mu_str = f"{mu:.3e}"
        sd_str = f"{sd:.3e}" if np.isfinite(sd) else ""
        rows.append({
            "Base": base_disp,
            "Method": method,
            "Mean": mu_str,
            "SD": sd_str,
            "Median": med,
            "Q25": q25,
            "Q75": q75,
            "N": int(s.size)
        })

    # order: Cohort, then groups/clusters
    bases = []
    if include_cohort:
        bases.append(("cohort", "cohort", "Cohort"))
    for key in labels_order:
        disp = f"Group {key[-1].upper()}" if kind == "group" else f"Cluster {key.replace('cluster','')}"
        bases.append((kind, key, disp))

    for kind_here, key_here, base_disp in bases:
        for m in ["LS", "DE", "BO"]:
            rec = _fetch(kind_here, key_here, m)
            if rec is None:
                continue
            df = pd.read_csv(rec["source"])
            if colname not in df.columns:
                continue
            _push_row(base_disp, m, df[colname])

    return pd.DataFrame(rows)

def save_param_summary_tables(recs, out_dir: Path = Path(".")) -> None:
    """
    Writes 8 CSVs total:
      param_k1_summary_groups.csv, param_k1_summary_clusters.csv, ... etc.
    """
    specs = [
        {"col": "k1",       "stem": "param_k1"},
        {"col": "k2",       "stem": "param_k2"},
        {"col": "vmax_in",  "stem": "param_vmax_in"},
        {"col": "vmax_out", "stem": "param_vmax_out"},
    ]
    out_dir.mkdir(parents=True, exist_ok=True)

    for sp in specs:
        col = sp["col"]; stem = sp["stem"]

        df_groups = _collect_param_stats(recs, kind="group",   labels_order=groups,   colname=col)
        df_clusters = _collect_param_stats(recs, kind="cluster", labels_order=clusters, colname=col)

        # save with scientific formatting for Mean/SD (optional: keep raw; here we keep raw numbers)
        df_groups.to_csv(out_dir / f"{stem}_summary_groups.csv", index=False)
        df_clusters.to_csv(out_dir / f"{stem}_summary_clusters.csv", index=False)

        print(f"Saved: {(out_dir / f'{stem}_summary_groups.csv').resolve()}")
        print(f"Saved: {(out_dir / f'{stem}_summary_clusters.csv').resolve()}")

# ============ ONE CSV WITH ALL PARAMS (Mean & SD only) ======================

PARAM_COLUMNS = ["k1", "k2", "vmax_in", "vmax_out"]  # exact column names in your CSVs

def _fetch_record(recs, kind: str, label: str, method: str):
    for r in recs:
        if r["kind"] == kind and r["label"] == label and r["method"] == method:
            return r
    return None

def _series_mean_sd_from_source(source_path: str, colname: str):
    df = pd.read_csv(source_path)
    if colname not in df.columns:
        return np.nan, np.nan
    s = pd.to_numeric(df[colname], errors="coerce").dropna()
    if s.empty:
        return np.nan, np.nan
    mu = float(np.nanmean(s))
    sd = float(np.nanstd(s, ddof=1)) if s.size > 1 else np.nan
    return mu, sd

def save_all_params_mean_sd_one_csv(recs, out_path: Path):
    """
    Builds a single CSV with rows for Cohort + Groups A–D + Clusters 0–4,
    for each method (LS/DE/BO) and columns of Mean/SD for all parameters.
    """
    rows = []

    # Helper to push a row for a given (Type, Base label, kind/label key)
    def push_rows_for_base(base_type: str, base_disp: str, kind_key: str, label_key: str):
        for method in ["LS", "DE", "BO"]:
            rec = _fetch_record(recs, kind=kind_key, label=label_key, method=method)
            if rec is None:
                continue
            row = {"Type": base_type, "Base": base_disp, "Method": method}
            for col in PARAM_COLUMNS:
                mu, sd = _series_mean_sd_from_source(rec["source"], col)
                row[f"{col}_mean"] = f"{mu:.3e}" if np.isfinite(mu) else ""
                row[f"{col}_sd"]   = f"{sd:.3e}" if np.isfinite(sd) else ""
            rows.append(row)

    # Cohort
    push_rows_for_base("cohort", "Cohort", "cohort", "cohort")

    # Clinical groups
    for g in groups:  # e.g., classa, classb, ...
        base_disp = f"Group {g[-1].upper()}"
        push_rows_for_base("group", base_disp, "group", g)

    # Clusters
    for c in clusters:  # e.g., cluster0..cluster4
        base_disp = f"Cluster {c.replace('cluster','')}"
        push_rows_for_base("cluster", base_disp, "cluster", c)

    df = pd.DataFrame(rows)

    # optional: order columns nicely
    ordered_cols = (
        ["Type", "Base", "Method"] +
        [f"{p}_{stat}" for p in PARAM_COLUMNS for stat in ("mean", "sd")]
    )
    df = df.reindex(columns=ordered_cols)

    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path.resolve()}")

# ================== METHOD-FACETED DISTRIBUTIONS + MEAN DOT PLOTS ==================
# Each parameter → two figures:
#   • Groups: 1 row × 3 panels (LS, DE, BO); each panel overlays Group A–D
#   • Clusters: 1 row × 3 panels (LS, DE, BO); each panel overlays Cluster 0–4

# ##commands — y-axis mode for histograms when shown: 'proportion', 'percent', or 'count'
DIST_YMODE = 'percent'

# ##commands — appearance
DIST_STYLE = {
    "bins": "fd",        # Freedman–Diaconis; or an int like 40
    "alpha": 0.35,       # bar fill alpha
    "edgewidth": 1.6,    # outline width
    "fig_h": 3.2,        # row height (in)
    "fig_w_per_ax": 4.6, # width per panel (in)
    "dpi": 300,
}

# ##commands — fonts/weights (use existing FS/FW if defined)
FS = globals().get("FS", {"title":14,"panel_title":11,"axis":11,"tick":9,"legend":11})
FW = globals().get("FW", {"title":"bold","panel_title":"bold","axis":"normal","tick":"normal","legend":"normal"})

# ##commands — title toggles
SHOW_TITLES = {
    "suptitle": False,   # show big figure title
    "panel_titles": True # show LS/DE/BO over panels
}

# ##commands — base palettes
group_colors = {"classa":"#4C78A8","classb":"#F58518","classc":"#54A24B","classd":"#B279A2"}
cluster_colors = {f"cluster{i}": c for i, c in enumerate(["#4C78A8","#F58518","#54A24B","#E45756","#72B7B2"])}

# Pretty labels for parameters
PARAM_SPECS_ROW2 = [
    {"col":"k1",       "label_tex": r"$k_{\mathrm{CS}\to \mathrm{ISF}}$",         "stem":"k1"},
    {"col":"k2",       "label_tex": r"$k_{\mathrm{ISF}\to \mathrm{CS}}$",         "stem":"k2"},
    {"col":"vmax_in",  "label_tex": r"$k^{\max}_{\mathrm{ISF}\to \mathrm{ICF}}$", "stem":"vmax_in"},
    {"col":"vmax_out", "label_tex": r"$k^{\max}_{\mathrm{ICF}\to \mathrm{ISF}}$", "stem":"vmax_out"},
]

def _arr_from(rec_path: str, col: str) -> np.ndarray:
    df = pd.read_csv(rec_path)
    if col not in df.columns:
        return np.array([])
    return pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()

def _samples(recs, kind: str, label: str, method: str, col: str) -> np.ndarray:
    for r in recs:
        if r["kind"] == kind and r["label"] == label and r["method"] == method:
            return _arr_from(r["source"], col)
    return np.array([])

def _title_style(ax):
    ax.title.set_fontsize(FS["panel_title"]); ax.title.set_fontweight(FW["panel_title"])
    ax.xaxis.label.set_fontsize(FS["axis"]);  ax.xaxis.label.set_fontweight(FW["axis"])
    ax.yaxis.label.set_fontsize(FS["axis"]);  ax.yaxis.label.set_fontweight(FW["axis"])
    for t in ax.get_xticklabels(): t.set_fontsize(FS["tick"]); t.set_fontweight(FW["tick"])
    for t in ax.get_yticklabels(): t.set_fontsize(FS["tick"]); t.set_fontweight(FW["tick"])

# ---------- ##commands: KDE settings ----------
KDE_CFG = {
    "enabled": True,
    "show_hist": False,
    "n_grid": 400,
    "bw_scale": 1.0,
    "fill": False,
    "linewidth": 2.0,
    "alpha_fill": 0.18,
    "pad_frac": 0.03,
    "bw_min_frac": 0.01,   # NEW: minimum BW as a fraction of (x_max - x_min)
    "bw_single_frac": 0.03 # already added earlier; keep or add if missing
}

def _silverman_bw(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    n = x.size
    if n <= 1:
        return 1.0
    sigma = np.std(x, ddof=1)
    iqr   = np.subtract(*np.percentile(x, [75, 25]))
    s = sigma if not np.isfinite(iqr) else min(sigma, iqr / 1.349)
    return 0.9 * s * (n ** (-1.0 / 5.0))

def _kde_eval_gaussian(x: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray:
    x = np.asarray(x, float)
    if x.size == 0:
        return np.zeros_like(grid)
    if not np.isfinite(bw) or bw <= 0:
        bw = max(np.std(x, ddof=1), 1e-12)
    z = (grid[:, None] - x[None, :]) / bw
    pdf = np.exp(-0.5 * z * z).sum(axis=1) / (x.size * bw * np.sqrt(2.0 * np.pi))
    return pdf

def _make_grid_from_concat(concat: np.ndarray, n_grid: int, pad_frac: float) -> np.ndarray:
    lo = float(np.nanmin(concat)); hi = float(np.nanmax(concat))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = lo - 1e-12, hi + 1e-12
    span = hi - lo
    lo -= pad_frac * span; hi += pad_frac * span
    return np.linspace(lo, hi, n_grid)

def _plot_param_method_faceted(recs, param_col: str, param_label_tex: str,
                               kind: str, labels_order: list[str], out_path: Path):
    """
    Panels = methods (LS/DE/BO). Within each panel, overlay bases (groups/clusters).
    Optionally draw smooth KDE curves.
    """
    methods_local = ["LS", "DE", "BO"]
    n_panels = len(methods_local)
    fig_w = max(10, DIST_STYLE["fig_w_per_ax"] * n_panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, DIST_STYLE["fig_h"]),
                             dpi=DIST_STYLE["dpi"], sharey=False)
    axes = np.atleast_1d(axes)

    palette = group_colors if kind == "group" else cluster_colors

    # ---- draw panels ----
    for ax, m in zip(axes, methods_local):
        # collect arrays per base for this method
        data_per_base = {}
        for lab in labels_order:
            arr = _samples(recs, kind=kind, label=lab, method=m, col=param_col)
            data_per_base[lab] = arr[np.isfinite(arr)] if arr.size else np.array([])

        concat = np.concatenate([v for v in data_per_base.values() if v.size]) \
                 if any(v.size for v in data_per_base.values()) else np.array([0.0, 1.0])
        grid = _make_grid_from_concat(concat, KDE_CFG["n_grid"], KDE_CFG["pad_frac"])
        bins = np.histogram_bin_edges(concat, bins=DIST_STYLE["bins"])

        for bkey, x in data_per_base.items():
            if x.size == 0: 
                continue
            color = palette.get(bkey, "0.5")
            label_txt = (f"Group {bkey[-1].upper()}" if kind == "group"
                         else f"Cluster {bkey.replace('cluster','')}")

            # optional bars
            if KDE_CFG["show_hist"]:
                counts, _ = np.histogram(x, bins=bins)
                if DIST_YMODE == "proportion":
                    yvals = counts / counts.sum(); ylab = "Proportion"
                elif DIST_YMODE == "percent":
                    yvals = 100.0 * counts / counts.sum(); ylab = "Percent"
                else:
                    yvals = counts.astype(float); ylab = "Count"
                centers = 0.5 * (bins[:-1] + bins[1:])
                ax.bar(centers, yvals, width=(bins[1:] - bins[:-1]),
                       align="center", alpha=DIST_STYLE["alpha"],
                       color=color, edgecolor="none", label=label_txt)

            # KDE curve
            # smooth KDE curve (density) — robust for tightly clustered data
            if KDE_CFG["enabled"] and x.size >= 1:
                span = max(1e-12, grid[-1] - grid[0])
                if x.size == 1:
                    bw = KDE_CFG.get("bw_single_frac", 0.03) * span
                else:
                    bw = _silverman_bw(x) * float(KDE_CFG["bw_scale"])
                    # enforce a floor so ultra-narrow spikes are still visible on the grid
                    bw = max(bw, KDE_CFG.get("bw_min_frac", 0.01) * span)
                kde = _kde_eval_gaussian(x, grid, bw)
                ax.plot(grid, kde, color=color, lw=KDE_CFG["linewidth"],
                        label=None if KDE_CFG["show_hist"] else label_txt)
                if KDE_CFG["fill"]:
                    ax.fill_between(grid, kde, color=color, alpha=KDE_CFG["alpha_fill"])

        if SHOW_TITLES.get("panel_titles", True):
            ax.set_title(m)
        _title_style(ax)
        ax.set_xlabel(param_label_tex)
        if KDE_CFG["enabled"] and not KDE_CFG["show_hist"]:
            ax.set_ylabel("Density (KDE)")
        else:
            ax.set_ylabel({"proportion":"Proportion", "percent":"Percent", "count":"Count"}[DIST_YMODE])
        ax.grid(axis="y", ls="--", alpha=0.3)

        # ---------- legend (outside, with box) + layout ----------
        handles_all, labels_all, seen = [], [], set()
        for a in np.atleast_1d(axes).ravel():
            h, l = a.get_legend_handles_labels()
            for hh, ll in zip(h, l):
                if ll not in seen:
                    handles_all.append(hh); labels_all.append(ll); seen.add(ll)

        if handles_all:
            leg = fig.legend(
                handles_all, labels_all,
                ncol=len(labels_order),              # 4 for groups, 5 for clusters
                loc="lower center",
                bbox_to_anchor=(0.5, 0.9),          # outside, just above panels
                fontsize=FS["legend"],
                title="Bases", title_fontsize=FS["legend"],
                frameon=True, fancybox=True,         # << box ON
                borderaxespad=0.0, handlelength=1.8, handletextpad=0.6
            )
            # style the legend box
            frame = leg.get_frame()
            frame.set_edgecolor("0.35")
            frame.set_linewidth(1.0)
            frame.set_facecolor("white")
            frame.set_alpha(0.95)

        # optional big title
        if SHOW_TITLES.get("suptitle", False):
            fig.suptitle(
                f"{param_label_tex} — {'clinical groups' if kind=='group' else 'clusters'}",
                fontsize=FS["title"], fontweight=FW["title"], y=0.97
            )
            fig.tight_layout(rect=[0.02, 0.05, 0.99, 0.90])   # leave room for legend+title
        else:
            fig.tight_layout(rect=[0.02, 0.05, 0.99, 0.90])   # leave room for legend only

            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out_path.resolve()}")

def plot_param_distributions_by_method(recs):
    """Creates 8 figures total: 4 params × (groups+clusters), method-faceted."""
    for spec in PARAM_SPECS_ROW2:
        _plot_param_method_faceted(
            recs, spec["col"], spec["label_tex"], "group", groups,
            Path(f"{spec['stem']}_dists_METHODfaceted_groups.jpg")
        )
        _plot_param_method_faceted(
            recs, spec["col"], spec["label_tex"], "cluster", clusters,
            Path(f"{spec['stem']}_dists_METHODfaceted_clusters.jpg")
        )

# ------------------------ MEAN DOT PLOTS (4 figs) ----------------------------
# x-axis category = "Base, Method", value label above dot (rot. 90°), auto headroom

from matplotlib.ticker import ScalarFormatter

MEAN_DOT_CFG = {
    "top_pad_frac": 0.7,
    "bottom_pad_frac": 0.05,
    "label_offset_pts": 6,
    "label_fontsize": 8,
}

def _display_name(base_type: str, base_key: str) -> str:
    if base_type == "cohort": return "Cohort"
    if base_type == "group":  return f"Group {base_key[-1].upper()}"
    return f"Cluster {base_key.replace('cluster','')}"

def _mean_for(recs, kind: str, label: str, method: str, col: str):
    arr = _samples(recs, kind, label, method, col)
    return float(np.nanmean(arr)) if arr.size else np.nan

def plot_param_mean_dots(recs):
    """Build 4 mean-dot figures (one per parameter) across Cohort+Groups+Clusters × Methods.
       Uses pastel method colors from `violin_fill` with matching edges from `violin_edge`."""
    base_seq = [("cohort","cohort")] + [("group", g) for g in groups] + [("cluster", c) for c in clusters]
    methods_local = ["LS","DE","BO"]

    for spec in PARAM_SPECS_ROW2:
        x_labels, means, faces, edges = [], [], [], []

        for btype, bkey in base_seq:
            for m in methods_local:
                arr = _samples(recs, kind=btype, label=bkey, method=m, col=spec["col"])
                if arr.size == 0:
                    continue
                mu = float(np.nanmean(arr))
                x_labels.append(f"{_display_name(btype, bkey)}, {m}")
                means.append(mu)
                faces.append(violin_fill.get(m, "#CCCCCC"))
                edges.append(violin_edge.get(m, "#666666"))

        if not means:
            continue

        x = np.arange(len(means))
        fig_w = max(10, 0.42 * len(means))
        fig, ax = plt.subplots(figsize=(fig_w, 4.4), dpi=300)

        # pastel dots with darker edge for contrast
        ax.scatter(x, means, s=42, zorder=3,
                   c=faces, edgecolors=edges, linewidths=0.9)

        # labels above dots (rotate 90°)
        value_texts = []
        for xi, yi in zip(x, means):
            txt = ax.annotate(f"{yi:.3e}", xy=(xi, yi),
                              xytext=(0, MEAN_DOT_CFG["label_offset_pts"]),
                              textcoords="offset points", ha="center", va="bottom",
                              rotation=90, fontsize=MEAN_DOT_CFG["label_fontsize"],
                              color="#040404", clip_on=False, zorder=4)
            value_texts.append(txt)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=90, ha="center",
                           fontsize=FS["tick"], fontweight=FW["tick"])

        # base y-limits + auto headroom for labels
        ymin = float(np.nanmin(means)); ymax = float(np.nanmax(means))
        rng = ymax - ymin if ymax > ymin else (abs(ymax) if ymax != 0 else 1.0)
        ax.set_ylim(ymin - MEAN_DOT_CFG["bottom_pad_frac"] * rng,
                    ymax + MEAN_DOT_CFG["top_pad_frac"]    * rng)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        top_pix = ax.get_window_extent(renderer=renderer).y1
        overflow_pix = 0.0
        for t in value_texts:
            overflow_pix = max(overflow_pix, t.get_window_extent(renderer=renderer).y1 - top_pix)
        if overflow_pix > 0:
            dy0 = ax.transData.inverted().transform((0, 0))[1]
            dy1 = ax.transData.inverted().transform((0, overflow_pix))[1]
            ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + (dy1 - dy0) * 1.1)

        # sci notation ticks
        sf = ScalarFormatter(useMathText=True); sf.set_powerlimits((-3, 4))
        ax.yaxis.set_major_formatter(sf)

        ax.set_ylabel(spec["label_tex"], fontsize=FS["axis"], fontweight=FW["axis"])
        ax.set_title(f"{spec['label_tex']} — means across bases and methods",
                     fontsize=FS["title"], fontweight=FW["title"])
        ax.grid(axis="y", ls="--", alpha=0.35)

        # legend that matches the pastel fills
        handles = [
            Line2D([0],[0], marker='o', linestyle='None',
                   markerfacecolor=violin_fill[m], markeredgecolor=violin_edge[m],
                   markersize=8, label=m)
            for m in methods_local
        ]
        ax.legend(handles=handles, title="Method",
                  fontsize=8, title_fontsize=10,
                  loc="upper left", frameon=True)

        fig.tight_layout(rect=[0.02, 0.06, 0.99, 0.96])
        outp = Path(f"{spec['stem']}_means_dot.jpg")
        fig.savefig(outp, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {outp.resolve()}")
# ===================== MAIN ======================
if __name__ == "__main__":
    # scan once from all three sources
    recs = []
    recs += scan_dir(DATA_DIR_GROUPS,   "group")
    recs += scan_dir(DATA_DIR_CLUSTERS, "cluster")
    recs += scan_dir(DATA_DIR_COHORT,   "cohort")
    if not recs:
        raise SystemExit("No matching CSVs found. Check paths and filenames.")

    # dot-and-error (median+IQR)
    df_iqr = build_rows(recs, mode="IQR")
    df_iqr.to_csv("rmse_summary_IQR.csv", index=False)
    make_dot_error_plot(df_iqr, Path("rmse_dot_error_IQR.jpg"), "median + IQR")

    # dot-and-error (mean±SD)
    df_sd = build_rows(recs, mode="SD")
    df_sd.to_csv("rmse_summary_SD.csv", index=False)
    make_dot_error_plot(df_sd, Path("rmse_dot_error_SD.jpg"), "mean ± SD")

    # violins (color by method, cohort included on the left)
    make_violins(recs, Path("rmse_violins_groups.jpg"), Path("rmse_violins_clusters.jpg"))

    # Faceted parameter plots (median + IQR with mean±SD labels)
        # Method-faceted distributions (per parameter)
    #plot_param_distributions_by_method(recs)

    # Mean dot plots (4 figs, one per parameter)
    plot_param_mean_dots(recs)
