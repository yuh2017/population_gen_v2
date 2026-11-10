"""Evaluation helpers for comparing synthetic population to ground truth."""

__version__ = "v3-2026-01-28"
print(f"{__name__} LOADED — {__version__}")


from typing import List
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import Counter
from .markov import resolve_demo_attrs, DEMO_ATTRS_REFERENCE



def compare_demographics(survey_persons: List[dict], synthetic_persons: List[dict], keys=None, top_n=10):
    """Compare marginal distributions for demographic keys.

    Returns a dict of DataFrames with side-by-side proportions.
    
    Parameters
    ----------
    survey_persons : List[dict]
        List of person records from survey
    synthetic_persons : List[dict]
        List of synthetic person records
    keys : List[str] or List[int], optional
        Demographic attributes to compare. Can be specified as:
        - List of integers (indices): e.g., [0, 1, 2, 3, 4, 5, 6]
        - List of strings (attribute names): e.g., ["AGE2", "GENDER", "EDUSTATUS"]
        - None: defaults to all attributes from DEMO_ATTRS_REFERENCE
        
        Reference (index order):
          0: AGE2, 1: GENDER, 2: EDUSTATUS, 3: INCOME_RANGE, 4: RACE, 5: HISP, 6: household_size
    top_n : int
        Number of top values to display (default: 10)
    """
    if keys is None:
        # Use default demographic attributes if not specified
        keys = resolve_demo_attrs(None)  # Uses full reference list
        # Filter to only keys that actually exist in the data
        available_keys = set()
        for p in survey_persons + synthetic_persons:
            available_keys.update(p.keys())
        keys = [k for k in keys if k in available_keys]
    else:
        # Resolve indices if provided
        keys = resolve_demo_attrs(keys)

    results = {}
    for k in keys:
        s_vals = [p.get(k) for p in survey_persons if p.get(k) is not None]
        t_vals = [p.get(k) for p in synthetic_persons if p.get(k) is not None]
        s_counts = Counter(s_vals)
        t_counts = Counter(t_vals)
        all_keys = sorted(set(list(s_counts.keys()) + list(t_counts.keys())))
        rows = []
        for v in all_keys:
            rows.append({'value': v, 'survey_prop': s_counts.get(v, 0) / max(1, len(s_vals)), 'synthetic_prop': t_counts.get(v, 0) / max(1, len(t_vals))})
        df = pd.DataFrame(rows).sort_values('survey_prop', ascending=False).head(top_n)
        results[k] = df
    return results


def compare_trip_taz_distribution(synthetic_trips_df: pd.DataFrame, flows_df: pd.DataFrame, origin_col='OTAZ', gt_origin_col='origin_TAZ', top_n=20):
    """Compare origin TAZ distribution between synthetic trips and ground-truth flows.

    Returns a DataFrame with proportions and a simple divergence metric (KL approx).
    """
    # ground truth origin distribution (from flows)
    gt = flows_df.groupby(gt_origin_col)['S000'].sum().rename('gt_flow').reset_index()
    gt['gt_prop'] = gt['gt_flow'] / gt['gt_flow'].sum()

    sim = synthetic_trips_df.groupby(origin_col).size().rename('sim_count').reset_index()
    sim['sim_prop'] = sim['sim_count'] / sim['sim_count'].sum()

    merged = pd.merge(gt, sim, left_on=gt_origin_col, right_on=origin_col, how='outer').fillna(0)
    merged['kl_divergence'] = merged.apply(lambda r: _kl_term(r['gt_prop'], r['sim_prop']), axis=1)
    merged['abs_diff'] = (merged['gt_prop'] - merged['sim_prop']).abs()
    merged = merged.sort_values('gt_flow', ascending=False).head(top_n)
    return merged


def _kl_term(p, q, eps=1e-9):
    # compute p * log(p/q) safely
    p = max(p, eps)
    q = max(q, eps)
    return p * np.log(p / q)


def plot_comparison(
    taz_shp_path, gt_path,  sim_path,
    taz_id_col,  gt_taz_col, gt_weight_col,  sim_taz_col,
    flows_origin_col: str = 'h_geocode',
    flows_work_col: str = 'w_geocode',
    cmap="OrRd", thr = 0.999, title=None
):
    """
    Plot one side-by-side TAZ map comparison (GT vs Simulation).

    Parameters
    ----------
    taz_shp_path : str
        Path to TAZ polygon shapefile
    gt_path : str
        Ground truth CSV path (OD flows)
    sim_path : str
        Simulation CSV path (activity / trips)
    taz_id_col : str
        TAZ ID column in shapefile
    gt_taz_col : str
        TAZ column in ground truth data (e.g. origin_TAZ)
    gt_weight_col : str
        Column to sum in ground truth data (e.g. S000)
    sim_taz_col : str
        TAZ column in simulated data (e.g. OTAZ)
    flows_origin_col : str
        Column name for home/origin geocode in flows data. Default: 'h_geocode'
    flows_work_col : str
        Column name for work geocode in flows data. Default: 'w_geocode'
    cmap : str
        Matplotlib colormap
    title : str, optional
        Figure title
    """

    # -------------------------
    # Load data
    # -------------------------
    taz = gpd.read_file(taz_shp_path)
    gt = pd.read_csv(gt_path)
    sim = pd.read_csv(sim_path)

    taz[ taz_id_col ] = taz[ taz_id_col ].astype(str)
    gt[ gt_taz_col ] = gt[ gt_taz_col ].astype(str)
    sim[ sim_taz_col ] = sim[ sim_taz_col ].astype(str)

    if flows_origin_col == gt_taz_col or flows_work_col == gt_taz_col:
        unique_h_geocodes = set( gt[flows_origin_col].astype(str).unique())
        unique_w_geocodes = set( gt[flows_work_col].astype(str).unique())
        unique_geocodes = unique_h_geocodes.union(unique_w_geocodes)
    elif gt_taz_col in ['origin_TAZ', 'dest_TAZ', 'orig_TAZ', 'Origin_TAZ', 'Dest_TAZ', 'Orig_TAZ'] :
        unique_h_geocodes = set( gt[flows_origin_col].astype(str).unique())
        unique_w_geocodes = set( gt[flows_work_col].astype(str).unique())
        unique_geocodes = unique_h_geocodes.union(unique_w_geocodes)

    taz = taz[ taz[ taz_id_col ].astype(str).isin(unique_geocodes)].copy()
    taz[ taz_id_col ] = taz[ taz_id_col ].astype(str)
    # -------------------------
    # Aggregate to TAZ
    # -------------------------
    gt_agg = (
        gt.groupby(gt_taz_col)[gt_weight_col]
        .sum()
        .rename("gt_value")
        .reset_index()
    )

    sim_agg = (
        sim.groupby(sim_taz_col)
        .size()
        .rename("sim_value")
        .reset_index()
    )

    # -------------------------
    # Join to polygons
    # -------------------------
    taz_join = (
        taz.merge(gt_agg, left_on=taz_id_col, right_on=gt_taz_col, how="left")
           .merge(sim_agg, left_on=taz_id_col, right_on=sim_taz_col, how="left")
    )

    taz_join["gt_value"] = taz_join["gt_value"].fillna(0)
    taz_join["sim_value"] = taz_join["sim_value"].fillna(0)

    # -------------------------
    # Plot
    # -------------------------
    vmax1 = taz_join["gt_value"].max()
    vmax2 = taz_join["sim_value"].quantile([ thr ]).values[0]
    vmin = 0 

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    taz_join.plot(
        column="gt_value",
        ax=axes[0],
        cmap=cmap,
        legend=True,
        vmin=vmin, # Set fixed min
        vmax= vmax1 # Set fixed max
    )
    axes[0].set_title("Ground Truth")

    taz_join.plot(
        column="sim_value",
        ax=axes[1],
        cmap=cmap,
        legend=True,
        vmin=vmin, # Match the left side
        vmax=vmax2  # Match the left side
    )

    axes[1].set_title("Simulation")
    #axes[1].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.show()


import matplotlib as mpl

def plot_comparison_perct(
    taz_shp_path, gt_path, sim_path,
    taz_id_col, gt_taz_col, gt_weight_col, sim_taz_col,
    flows_origin_col: str = 'h_geocode',
    flows_work_col: str = 'w_geocode',
    cmap="OrRd", thr=0.99, title=None
):
    # 1. Load data
    taz = gpd.read_file(taz_shp_path)
    gt = pd.read_csv(gt_path)
    sim = pd.read_csv(sim_path)

    # Standardize IDs to string
    taz[taz_id_col]  = taz[taz_id_col].astype(str)
    gt[gt_taz_col]   = gt[gt_taz_col].astype(str)
    sim[sim_taz_col] = sim[sim_taz_col].astype(str)

    # Robust unique geocode extraction
    gt_cols = set(gt.columns)
    unique_geocodes = set()
    if flows_origin_col in gt_cols:
        unique_geocodes.update(gt[flows_origin_col].astype(str).unique())
    if flows_work_col in gt_cols:
        unique_geocodes.update(gt[flows_work_col].astype(str).unique())
    if not unique_geocodes:
        unique_geocodes = set(gt[gt_taz_col].astype(str).unique())

    taz = taz[taz[taz_id_col].astype(str).isin(unique_geocodes)].copy()
    taz[taz_id_col] = taz[taz_id_col].astype(str)

    # 2. Aggregate to TAZ (Raw Counts)
    gt_agg = gt.groupby(gt_taz_col)[gt_weight_col].sum().rename("gt_raw").reset_index()
    sim_agg = sim.groupby(sim_taz_col).size().rename("sim_raw").reset_index()

    # 3. Convert to Percentages of Total
    gt_agg["gt_pct"] = (gt_agg["gt_raw"] / gt_agg["gt_raw"].sum()) * 100
    sim_agg["sim_pct"] = (sim_agg["sim_raw"] / sim_agg["sim_raw"].sum()) * 100

    # 4. Join to polygons
    taz_join = taz.merge(gt_agg, left_on=taz_id_col, right_on=gt_taz_col, how="left")
    taz_join = taz_join.merge(sim_agg, left_on=taz_id_col, right_on=sim_taz_col, how="left")

    # Fill NAs with 0 (areas with no trips)
    taz_join[["gt_pct", "sim_pct"]] = taz_join[["gt_pct", "sim_pct"]].fillna(0)

    # 5. Distinguishable Color Normalization
    # We find a common vmax based on the percentile threshold across both datasets
    combined_values = pd.concat([taz_join["gt_pct"], taz_join["sim_pct"]])
    vmax_shared = combined_values.quantile(thr)
    vmin = 0

    # 6. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Ground Truth Plot
    taz_join.plot(
        column="gt_pct",
        ax=axes[0],
        cmap=cmap,
        legend= False,
        vmin=vmin,
        vmax=vmax_shared,
        legend_kwds={'label': "Percentage of Total Trips (%)", 'orientation': "horizontal"}
    )
    axes[0].set_title("Ground Truth (%)", fontsize=12)
    axes[0].axis("off")

    # Simulation Plot
    taz_join.plot(
        column="sim_pct",
        ax=axes[1],
        cmap=cmap,
        legend= False,
        vmin=vmin,
        vmax=vmax_shared,
        legend_kwds={'label': "Percentage of Total Trips (%)", 'orientation': "horizontal"}
    )
    axes[1].set_title("Simulation (%)", fontsize=12)
    axes[1].axis("off")

    # Shared colorbar
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax_shared)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []  # required for older matplotlib versions

    # fig.add_axes([left, bottom, width, height])
    cax = fig.add_axes([0.25, 0.01, 0.5, 0.04])
    cbar = fig.colorbar(
        sm,
        cax=cax,
        ax=axes,
        orientation = "horizontal"    )

    cbar.set_label("Percentage of Total Trips (%)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    if title:
        fig.suptitle(title, fontsize=16, y=1.05)

    plt.tight_layout()
    plt.show()


def plot_population_profile_comparison(
    taz_shp_path,
    gt_path,
    gt_geocode_col,
    gt_weight_col,
    persons_df,
    profile_zone_col,  # e.g., 'home_geoid' or 'work_geoid'
    taz_id_col="GEOID20",
    cmap="OrRd",
    thr=0.99,
    title=None,
    fig_size=(16, 7),
    font_size=12,
    title_fontsize=14,
    legend_fontsize=16,
    label_fontsize=14,
    tick_fontsize=14,
):
    """
    Compare spatial distribution of home/workplace from generated population profile to LODES RAC/WAC ground truth.
    persons_df: DataFrame with one row per person, and a column for home or work geoid.
    profile_zone_col: column in persons_df for home or work geoid (e.g., 'home_geoid' or 'work_geoid')
    gt_geocode_col: column in ground truth for geocode (e.g., 'h_geocode' or 'w_geocode')
    gt_weight_col: column in ground truth for count (e.g., 'C000')
    """
    # Load data
    taz = gpd.read_file(taz_shp_path)
    gt = pd.read_csv(gt_path)
    persons_df = persons_df.copy()

    # Standardize IDs to string
    taz[taz_id_col] = taz[taz_id_col].astype(str)
    gt[gt_geocode_col] = gt[gt_geocode_col].astype(str)
    persons_df[profile_zone_col] = persons_df[profile_zone_col].astype(str)

    # Aggregate ground truth
    gt_agg = gt.groupby(gt_geocode_col)[gt_weight_col].sum().rename("gt_raw").reset_index()

    # Aggregate simulated (one count per person)
    sim_agg = persons_df.groupby(profile_zone_col).size().rename("sim_raw").reset_index()
    sim_agg = sim_agg.rename(columns={profile_zone_col: taz_id_col})

    # Convert to percentages
    gt_agg["gt_pct"] = (gt_agg["gt_raw"] / gt_agg["gt_raw"].sum()) * 100
    sim_agg["sim_pct"] = (sim_agg["sim_raw"] / sim_agg["sim_raw"].sum()) * 100

    # Join to polygons
    taz_join = taz.merge(gt_agg, left_on=taz_id_col, right_on=gt_geocode_col, how="left")
    taz_join = taz_join.merge(sim_agg, left_on=taz_id_col, right_on=taz_id_col, how="left")

    # Fill NAs with 0
    taz_join[["gt_pct", "sim_pct"]] = taz_join[["gt_pct", "sim_pct"]].fillna(0)

    # Color normalization
    combined_values = pd.concat([taz_join["gt_pct"], taz_join["sim_pct"]])
    vmax_shared = combined_values.quantile(thr)
    vmin = 0

    fig, axes = plt.subplots(1, 2, figsize=fig_size)
    taz_join.plot(
        column="gt_pct",
        ax=axes[0],
        cmap=cmap,
        legend=False,
        vmin=vmin,
        vmax=vmax_shared,
    )
    axes[0].set_title("Ground Truth (%)", fontsize=font_size, fontweight="bold")
    axes[0].axis("off")

    taz_join.plot(
        column="sim_pct",
        ax=axes[1],
        cmap=cmap,
        legend=False,
        vmin=vmin,
        vmax=vmax_shared,
    )
    axes[1].set_title("Simulation (%)", fontsize=font_size, fontweight="bold")
    axes[1].axis("off")

    # Shared colorbar (create mappable first, then the axis)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax_shared)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.subplots_adjust(bottom=0.16)
    cax = fig.add_axes([0.25, 0.04, 0.5, 0.04])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(
        "Percentage of Total Population (%)",
        fontsize=label_fontsize,
        fontweight="bold",
    )
    cbar.ax.tick_params(labelsize=tick_fontsize)

    if title:
        fig.suptitle(title, fontsize=title_fontsize, fontweight="bold", y=0.98)

    plt.show()

def plot_comparison_perct_with_scatter(
    taz_shp_path, gt_path, sim_path,
    taz_id_col, gt_taz_col, gt_weight_col, sim_taz_col,
    flows_origin_col: str = 'h_geocode',
    flows_work_col: str = 'w_geocode',
    cmap="OrRd", thr=0.99, title=None
):
    # Load data
    taz = gpd.read_file(taz_shp_path)
    gt = pd.read_csv(gt_path)
    sim = pd.read_csv(sim_path)

    # Standardize IDs to string
    taz[taz_id_col] = taz[taz_id_col].astype(str)
    gt[gt_taz_col] = gt[gt_taz_col].astype(str)
    sim[sim_taz_col] = sim[sim_taz_col].astype(str)


    # Robust unique geocode extraction
    gt_cols = set(gt.columns)
    unique_geocodes = set()
    # Try to use both columns if available, else fallback
    if flows_origin_col in gt_cols:
        unique_geocodes.update(gt[flows_origin_col].astype(str).unique())
    if flows_work_col in gt_cols:
        unique_geocodes.update(gt[flows_work_col].astype(str).unique())
    if not unique_geocodes:
        unique_geocodes = set(gt[gt_taz_col].astype(str).unique())

    taz = taz[taz[taz_id_col].astype(str).isin(unique_geocodes)].copy()
    taz[taz_id_col] = taz[taz_id_col].astype(str)

    # Aggregate to TAZ (Raw Counts)
    gt_agg = gt.groupby(gt_taz_col)[gt_weight_col].sum().rename("gt_raw").reset_index()
    sim_agg = sim.groupby(sim_taz_col).size().rename("sim_raw").reset_index()

    # Convert to Percentages of Total    
    q_thr1 = gt_agg["gt_raw"].quantile( thr )
    q_thr2 = sim_agg["sim_raw"].quantile( thr )

    print(  gt_agg.loc[ gt_agg["gt_raw"] > q_thr1 ] )
    gt_agg = gt_agg.loc[gt_agg["gt_raw"] < q_thr1 ]
    gt_agg["gt_pct"] = (gt_agg["gt_raw"] / gt_agg["gt_raw"].sum()) * 100
    
    print( sim_agg.loc[ sim_agg["sim_raw"] > q_thr2 ] )
    sim_agg = sim_agg.loc[sim_agg["sim_raw"] < q_thr2 ]
    sim_agg["sim_pct"] = (sim_agg["sim_raw"] / sim_agg["sim_raw"].sum()) * 100

    # Join to polygons
    taz_join = taz.merge(gt_agg, left_on=taz_id_col, right_on=gt_taz_col, how="left")
    taz_join = taz_join.merge(sim_agg, left_on=taz_id_col, right_on=sim_taz_col, how="left")

    # Fill NAs with 0 (areas with no trips)
    taz_join[["gt_pct", "sim_pct"]] = taz_join[["gt_pct", "sim_pct"]].fillna(0)

    # Distinguishable Color Normalization
    combined_values = pd.concat([taz_join["gt_pct"], taz_join["sim_pct"]])
    vmax_shared = thr
    vmin = 0

    # taz_join = taz_join.loc[ (taz_join["gt_pct"] <= vmax_shared) & ( taz_join["sim_pct"] <= vmax_shared ) ]
    # Now plot the scatter plot comparing GT and Simulation percentages
    plt.figure(figsize=(8, 6))
    plt.scatter(taz_join["gt_pct"], taz_join["sim_pct"], alpha=0.6, edgecolors='k')

    # Plot 45-degree reference line
    max_val = vmax_shared *1.05
    plt.plot([vmin, max_val], [vmin, max_val], color='red', linestyle='--', linewidth=1)

    plt.xlabel("Ground Truth Percentage (%)")
    plt.ylabel("Simulation Percentage (%)")
    plt.title("Scatter Plot of Ground Truth vs Simulation Percentages")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
