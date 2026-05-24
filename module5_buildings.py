#!/usr/bin/env python3
"""
Module 5 — Building Exposure & Urban Vulnerability
Spatial join of building footprints to townships.
Computes per-township: building count, density, total footprint area,
mean building size, resonance risk (site period vs building period proxy).

Outputs:
  outputs/township_exposure.csv
  outputs/building_density_map.html
  outputs/building_exposure_static.png
"""

import os, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import folium
import branca.colormap as bcm

warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

BLDG_PATH = os.path.join(BASE_DIR, "yangon_townships_buildings.geojson")
TWP_PATH  = os.path.join(BASE_DIR, "yangon_townships.geojson")


def estimate_building_period(footprint_area_m2: float) -> float:
    """
    Very rough proxy for dominant building period based on footprint size.
    Small footprint → likely 1-2 storey (masonry/timber)  T ≈ 0.05-0.2 s
    Medium          → 3-5 storey RC frame                  T ≈ 0.3-0.5 s
    Large           → 6+ storey RC frame                   T ≈ 0.5-1.5 s
    Returns approximate building period (s).
    """
    if footprint_area_m2 < 50:
        return 0.1
    elif footprint_area_m2 < 200:
        return 0.2
    elif footprint_area_m2 < 500:
        return 0.35
    elif footprint_area_m2 < 1500:
        return 0.6
    else:
        return 1.0


def resonance_risk(T_site: float, T_building: float,
                   ratio_low=0.7, ratio_high=1.3) -> str:
    """
    Flag resonance if T_site / T_building is within [ratio_low, ratio_high].
    """
    if pd.isna(T_site) or pd.isna(T_building) or T_building == 0:
        return "Unknown"
    ratio = T_site / T_building
    if ratio_low <= ratio <= ratio_high:
        return "High"
    elif 0.5 <= ratio < ratio_low or ratio_high < ratio <= 1.5:
        return "Moderate"
    else:
        return "Low"


def main():
    # ── Load townships ─────────────────────────────────────────────────────────
    print("Loading township boundaries ...", flush=True)
    gdf_twp = gpd.read_file(TWP_PATH).to_crs(epsg=32647)   # UTM 47N for area calc

    # ── Load buildings (large file — use chunked reading via fiona) ────────────
    print(f"Loading buildings ({os.path.getsize(BLDG_PATH)/1e6:.0f} MB) ...", flush=True)
    gdf_bldg = gpd.read_file(BLDG_PATH).to_crs(epsg=32647)
    print(f"  Loaded {len(gdf_bldg):,} buildings")

    # ── Compute building footprint areas ───────────────────────────────────────
    gdf_bldg["area_m2"] = gdf_bldg.geometry.area
    gdf_bldg = gdf_bldg[gdf_bldg["area_m2"] > 0].copy()

    # ── Spatial join: tag each building with township ─────────────────────────
    print("Spatial join (buildings → townships) ...", flush=True)
    joined = gpd.sjoin(gdf_bldg, gdf_twp[["adm3_name", "adm2_name",
                                           "adm3_pcode", "area_sqkm", "geometry"]],
                       how="left", predicate="within")
    print(f"  Joined: {joined['adm3_pcode'].notna().sum():,} / {len(joined):,} buildings matched")

    # ── Aggregate per township ─────────────────────────────────────────────────
    grp = (joined.dropna(subset=["adm3_pcode"])
                 .groupby("adm3_pcode")["area_m2"])
    agg = grp.agg(
        building_count="count",
        total_footprint_m2="sum",
        mean_footprint_m2="mean",
        median_footprint_m2="median",
        p90_footprint_m2=lambda x: np.percentile(x, 90),
    ).reset_index()

    # Merge back township metadata
    twp_meta = gdf_twp[["adm3_name", "adm2_name", "adm3_pcode",
                          "area_sqkm", "center_lat", "center_lon"]].copy()
    # twp_meta is a plain DataFrame; no CRS conversion needed
    expo = twp_meta.merge(agg, on="adm3_pcode", how="left")

    # Fill missing (no buildings in township)
    expo["building_count"]       = expo["building_count"].fillna(0).astype(int)
    expo["total_footprint_m2"]   = expo["total_footprint_m2"].fillna(0)
    expo["mean_footprint_m2"]    = expo["mean_footprint_m2"].fillna(0)
    expo["median_footprint_m2"]  = expo["median_footprint_m2"].fillna(0)
    expo["p90_footprint_m2"]     = expo["p90_footprint_m2"].fillna(0)

    # Building density (buildings / km²)
    expo["building_density_km2"] = (expo["building_count"] /
                                     expo["area_sqkm"].replace(0, np.nan)).fillna(0)

    # Building coverage ratio (total footprint / township area)
    expo["coverage_ratio"] = (expo["total_footprint_m2"] /
                               (expo["area_sqkm"] * 1e6)).clip(0, 1)

    # Dominant building period proxy (from median footprint)
    expo["dom_building_period_s"] = expo["median_footprint_m2"].apply(
        estimate_building_period)

    # ── Load HVSR/microzonation for resonance check ────────────────────────────
    micro_path = os.path.join(OUT_DIR, "township_microzonation.csv")
    if os.path.exists(micro_path):
        micro_df = pd.read_csv(micro_path)
        expo = expo.merge(micro_df[["adm3_pcode", "mean_T0", "mean_f0", "mean_A0",
                                     "nehrp_class"]],
                          on="adm3_pcode", how="left")
        expo["resonance_risk"] = expo.apply(
            lambda r: resonance_risk(r.get("mean_T0", np.nan),
                                     r["dom_building_period_s"]), axis=1)
    else:
        expo["mean_T0"]          = np.nan
        expo["mean_f0"]          = np.nan
        expo["mean_A0"]          = np.nan
        expo["nehrp_class"]      = "Unknown"
        expo["resonance_risk"]   = "Unknown"
        print("  (microzonation data not found — resonance risk set to Unknown)")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    out_cols = ["adm3_name", "adm2_name", "adm3_pcode", "area_sqkm",
                "center_lat", "center_lon",
                "building_count", "building_density_km2", "coverage_ratio",
                "total_footprint_m2", "mean_footprint_m2", "median_footprint_m2",
                "p90_footprint_m2", "dom_building_period_s",
                "mean_T0", "mean_f0", "mean_A0", "nehrp_class", "resonance_risk"]
    # Keep only columns that exist
    out_cols = [c for c in out_cols if c in expo.columns]
    # Drop geometry column if present
    expo_df = pd.DataFrame(expo[out_cols].values, columns=out_cols)
    csv_path = os.path.join(OUT_DIR, "township_exposure.csv")
    expo_df.to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")

    # ── Static plots ───────────────────────────────────────────────────────────
    gdf_twp_wgs = gdf_twp.to_crs(epsg=4326)
    gdf_plot    = gdf_twp_wgs.merge(expo_df, on="adm3_pcode", how="left",
                                     suffixes=("", "_expo"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Building density
    gdf_plot.plot(column="building_density_km2", ax=axes[0],
                  cmap="YlOrRd", legend=True,
                  missing_kwds={"color": "lightgrey"})
    axes[0].set_title("Building Density (buildings/km²)", fontsize=11)
    axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")

    # Coverage ratio
    gdf_plot.plot(column="coverage_ratio", ax=axes[1],
                  cmap="Blues", legend=True,
                  missing_kwds={"color": "lightgrey"})
    axes[1].set_title("Building Coverage Ratio", fontsize=11)
    axes[1].set_xlabel("Longitude"); axes[1].set_ylabel("Latitude")

    plt.suptitle("Yangon Building Exposure", fontsize=13, fontweight="bold")
    plt.tight_layout()
    static_path = os.path.join(OUT_DIR, "building_exposure_static.png")
    plt.savefig(static_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {static_path}")

    # Resonance risk map
    if "resonance_risk" in gdf_plot.columns:
        fig2, ax2 = plt.subplots(figsize=(7, 7))
        res_colors = {"High": "#e74c3c", "Moderate": "#e67e22",
                      "Low": "#2ecc71", "Unknown": "#bdc3c7"}
        for risk_level, color in res_colors.items():
            sub = gdf_plot[gdf_plot["resonance_risk"] == risk_level]
            if len(sub):
                sub.plot(ax=ax2, color=color, edgecolor="#333", lw=0.5,
                         label=f"{risk_level} ({len(sub)})")
        gdf_plot.boundary.plot(ax=ax2, color="#333", lw=0.4)
        ax2.set_title("Resonance Risk by Township\n(Site Period vs Building Period)",
                      fontsize=11)
        ax2.legend(title="Resonance Risk", loc="lower right", fontsize=9)
        ax2.set_xlabel("Longitude"); ax2.set_ylabel("Latitude")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "resonance_risk_map.png"), dpi=150)
        plt.close(fig2)
        print(f"Saved → {os.path.join(OUT_DIR, 'resonance_risk_map.png')}")

    # ── Interactive Folium map ─────────────────────────────────────────────────
    center = [gdf_twp_wgs.geometry.centroid.y.mean(),
              gdf_twp_wgs.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

    max_dens = expo_df["building_density_km2"].max()
    colormap = bcm.LinearColormap(
        ["#ffffcc", "#feb24c", "#e31a1c"],
        vmin=0, vmax=max(max_dens, 1),
        caption="Building Density (buildings/km²)")
    colormap.add_to(m)

    for _, row in gdf_plot.iterrows():
        dens  = row.get("building_density_km2", 0) or 0
        color = colormap(float(dens))
        geom_json = row.geometry.__geo_interface__
        popup_txt = (
            f"<b>{row.get('adm3_name', '')}</b><br>"
            f"District: {row.get('adm2_name', '')}<br>"
            f"Buildings: <b>{int(row.get('building_count', 0) or 0):,}</b><br>"
            f"Density: {float(dens):.0f} bldg/km²<br>"
            f"Coverage: {float(row.get('coverage_ratio', 0) or 0):.1%}<br>"
            f"Mean footprint: {float(row.get('mean_footprint_m2', 0) or 0):.0f} m²<br>"
            f"NEHRP: {row.get('nehrp_class', 'N/A')}<br>"
            f"Resonance risk: <b>{row.get('resonance_risk', 'N/A')}</b>"
        )
        folium.GeoJson(
            geom_json,
            style_function=lambda feat, c=color: {
                "fillColor": c, "color": "#555", "weight": 0.8,
                "fillOpacity": 0.65},
            tooltip=row.get("adm3_name", ""),
            popup=folium.Popup(popup_txt, max_width=230),
        ).add_to(m)

    html_path = os.path.join(OUT_DIR, "building_density_map.html")
    m.save(html_path)
    print(f"Saved → {html_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    total_bldg = expo_df["building_count"].sum()
    print(f"\n── Building Exposure Summary ──")
    print(f"Total buildings (all townships): {total_bldg:,}")
    print(f"\nTop 10 townships by building density:")
    print(expo_df.sort_values("building_density_km2", ascending=False)
          [["adm3_name", "building_count", "building_density_km2",
            "coverage_ratio", "resonance_risk"]]
          .head(10).to_string(index=False))

    if "resonance_risk" in expo_df.columns:
        print(f"\nResonance risk distribution:")
        print(expo_df["resonance_risk"].value_counts().to_string())


if __name__ == "__main__":
    main()
