#!/usr/bin/env python3
"""
Module 6 — Integrated Risk Mapping
Combines site amplification, interpolated ground motion and building exposure
into a per-township Seismic Risk Index (SRI) and choropleth map.

Inputs (from previous modules):
  outputs/township_microzonation.csv   (Module 2 — mean_A0, mean_T0, nehrp_class)
  outputs/event_ground_motion.csv      (Module 3 — PGA per station)
  outputs/township_exposure.csv        (Module 5 — building_density, resonance_risk)
  yangon_townships.geojson             (boundaries)

Method:
  Each of three sub-scores is normalised to [0, 1]:
    hazard_score   = normalised mean amplification A0  (from microzonation)
    exposure_score = normalised building density        (buildings/km²)
    pga_score      = normalised interpolated PGA        (spatially IDW from event stations)

  SRI = 0.40 × hazard_score + 0.35 × exposure_score + 0.25 × pga_score

  SRI classes:
    0.00–0.25  Very Low
    0.25–0.50  Low
    0.50–0.65  Moderate
    0.65–0.80  High
    0.80–1.00  Very High

Outputs:
  outputs/township_risk_index.geojson
  outputs/township_risk_index.csv
  outputs/risk_map.html
  outputs/risk_map_static.png
"""

import os, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import folium

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Weights ───────────────────────────────────────────────────────────────────
W_HAZARD   = 0.40
W_EXPOSURE = 0.35
W_PGA      = 0.25

# ── SRI classification ────────────────────────────────────────────────────────
SRI_CLASSES = [
    (0.00, 0.25, "Very Low",  "#2ecc71"),
    (0.25, 0.50, "Low",       "#f1c40f"),
    (0.50, 0.65, "Moderate",  "#e67e22"),
    (0.65, 0.80, "High",      "#e74c3c"),
    (0.80, 1.00, "Very High", "#7d0000"),
]

def classify_sri(v):
    if pd.isna(v):
        return "Unknown", "#95a5a6"
    for lo, hi, label, color in SRI_CLASSES:
        if lo <= v <= hi:
            return label, color
    return "Unknown", "#95a5a6"


def minmax_norm(series: pd.Series, lower=0.0, upper=1.0) -> pd.Series:
    """Min-max normalise a Series to [lower, upper]; handles all-equal."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(np.full(len(series), (lower + upper) / 2),
                         index=series.index)
    return lower + (series - mn) / (mx - mn) * (upper - lower)


def idw_interpolate(known_lons, known_lats, known_vals,
                    query_lons, query_lats, power=2.0):
    """Simple IDW interpolation for scalar field at query points."""
    out = np.zeros(len(query_lons))
    for i, (qx, qy) in enumerate(zip(query_lons, query_lats)):
        d = np.sqrt((known_lons - qx) ** 2 + (known_lats - qy) ** 2)
        if d.min() < 1e-8:
            out[i] = known_vals[d.argmin()]
            continue
        w = 1.0 / d ** power
        out[i] = np.sum(w * known_vals) / np.sum(w)
    return out


def main():
    # ── Load input data ────────────────────────────────────────────────────────
    micro_path = os.path.join(OUT_DIR, "township_microzonation.csv")
    gm_path    = os.path.join(OUT_DIR, "event_ground_motion.csv")
    expo_path  = os.path.join(OUT_DIR, "township_exposure.csv")
    twp_path   = os.path.join(BASE_DIR, "yangon_townships.geojson")

    gdf = gpd.read_file(twp_path).to_crs(epsg=4326)

    # ── Hazard score from microzonation ───────────────────────────────────────
    if os.path.exists(micro_path):
        micro_df = pd.read_csv(micro_path)
        gdf = gdf.merge(micro_df[["adm3_pcode", "mean_A0", "mean_T0",
                                   "mean_f0", "nehrp_class"]],
                        on="adm3_pcode", how="left")
        has_micro = True
    else:
        gdf["mean_A0"]    = np.nan
        gdf["mean_T0"]    = np.nan
        gdf["mean_f0"]    = np.nan
        gdf["nehrp_class"]= "Unknown"
        has_micro = False
        print("  Warning: microzonation data not found — hazard score will be uniform")

    # ── PGA interpolation from event stations ─────────────────────────────────
    if os.path.exists(gm_path):
        gm_df = pd.read_csv(gm_path)
        gm_ok = gm_df.dropna(subset=["lat", "lon", "pga_cms2"])
        has_gm = len(gm_ok) >= 2
    else:
        gm_ok  = pd.DataFrame()
        has_gm = False
        print("  Warning: ground-motion data not found — PGA score will be uniform")

    # ── Building exposure ──────────────────────────────────────────────────────
    if os.path.exists(expo_path):
        expo_df = pd.read_csv(expo_path)
        gdf = gdf.merge(expo_df[["adm3_pcode", "building_density_km2",
                                  "coverage_ratio", "resonance_risk"]],
                        on="adm3_pcode", how="left")
        has_expo = True
    else:
        gdf["building_density_km2"] = np.nan
        gdf["coverage_ratio"]       = np.nan
        gdf["resonance_risk"]       = "Unknown"
        has_expo = False
        print("  Warning: building exposure data not found — exposure score will be uniform")

    # ── Interpolate PGA to each township centroid ─────────────────────────────
    c_lons = gdf.geometry.centroid.x.values
    c_lats = gdf.geometry.centroid.y.values

    if has_gm:
        pga_interp = idw_interpolate(
            gm_ok["lon"].values, gm_ok["lat"].values, gm_ok["pga_cms2"].values,
            c_lons, c_lats)
        gdf["interp_pga_cms2"] = pga_interp
    else:
        gdf["interp_pga_cms2"] = 1.0   # uniform placeholder

    # ── Resonance bonus: boost exposure score if resonance is High ────────────
    def res_factor(x):
        if x == "High":
            return 1.25
        elif x == "Moderate":
            return 1.10
        return 1.0

    if "resonance_risk" in gdf.columns:
        gdf["resonance_factor"] = gdf["resonance_risk"].apply(res_factor)
    else:
        gdf["resonance_factor"] = 1.0

    # ── Compute sub-scores ────────────────────────────────────────────────────
    # Hazard
    A0_col = gdf["mean_A0"].fillna(gdf["mean_A0"].median())
    if A0_col.isna().all():
        A0_col = pd.Series(np.ones(len(gdf)), index=gdf.index)
    gdf["hazard_score"] = minmax_norm(A0_col)

    # Exposure (density × resonance factor)
    dens_col = gdf["building_density_km2"].fillna(0)
    dens_weighted = dens_col * gdf["resonance_factor"]
    gdf["exposure_score"] = minmax_norm(dens_weighted)

    # PGA
    pga_col = gdf["interp_pga_cms2"].fillna(gdf["interp_pga_cms2"].median())
    if pga_col.isna().all():
        pga_col = pd.Series(np.ones(len(gdf)), index=gdf.index)
    gdf["pga_score"] = minmax_norm(pga_col)

    # ── Composite SRI ─────────────────────────────────────────────────────────
    gdf["SRI"] = (W_HAZARD   * gdf["hazard_score"] +
                  W_EXPOSURE * gdf["exposure_score"] +
                  W_PGA      * gdf["pga_score"])

    gdf["SRI_class"], gdf["SRI_color"] = zip(*gdf["SRI"].apply(classify_sri))

    # ── Save outputs ───────────────────────────────────────────────────────────
    out_cols_df = ["adm3_name", "adm2_name", "adm3_pcode", "area_sqkm",
                   "center_lat", "center_lon",
                   "mean_f0", "mean_A0", "mean_T0", "nehrp_class",
                   "building_density_km2", "coverage_ratio", "resonance_risk",
                   "interp_pga_cms2",
                   "hazard_score", "exposure_score", "pga_score",
                   "SRI", "SRI_class"]
    out_cols_df = [c for c in out_cols_df if c in gdf.columns]
    risk_df = pd.DataFrame({c: gdf[c].values for c in out_cols_df})
    risk_df = risk_df.round({
        "mean_f0": 4, "mean_A0": 4, "mean_T0": 4,
        "building_density_km2": 2, "coverage_ratio": 4,
        "interp_pga_cms2": 3,
        "hazard_score": 4, "exposure_score": 4, "pga_score": 4, "SRI": 4,
    })

    csv_path = os.path.join(OUT_DIR, "township_risk_index.csv")
    risk_df.to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")

    gdf_out = gdf[out_cols_df + ["geometry", "SRI_color"]].copy()
    geo_path = os.path.join(OUT_DIR, "township_risk_index.geojson")
    gdf_out.to_file(geo_path, driver="GeoJSON")
    print(f"Saved → {geo_path}")

    # ── Static map ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left: SRI choropleth
    ax = axes[0]
    for _, row in gdf.iterrows():
        geom = row.geometry
        color = row.get("SRI_color", "#95a5a6")
        if geom and not geom.is_empty:
            gpd.GeoSeries([geom]).plot(ax=ax, color=color, edgecolor="#333", lw=0.5)
    patches = [mpatches.Patch(color=c, label=f"{l} ({lo:.2f}–{hi:.2f})")
               for lo, hi, l, c in SRI_CLASSES]
    ax.legend(handles=patches, title="SRI Class", fontsize=8,
              loc="lower right")
    ax.set_title("Seismic Risk Index (SRI) by Township", fontsize=11)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

    # Right: component breakdown bar chart (top 15 townships by SRI)
    ax2 = axes[1]
    top = risk_df.sort_values("SRI", ascending=False).head(15)
    x   = np.arange(len(top))
    w   = 0.28
    ax2.bar(x - w,   top["hazard_score"]   * W_HAZARD,   w,
            label=f"Hazard (×{W_HAZARD})",   color="#e74c3c", alpha=0.8)
    ax2.bar(x,       top["exposure_score"] * W_EXPOSURE, w,
            label=f"Exposure (×{W_EXPOSURE})", color="#3498db", alpha=0.8)
    ax2.bar(x + w,   top["pga_score"]      * W_PGA,      w,
            label=f"PGA (×{W_PGA})",         color="#2ecc71", alpha=0.8)
    ax2.plot(x, top["SRI"], "ko-", ms=5, lw=1.5, label="SRI total")
    ax2.set_xticks(x)
    ax2.set_xticklabels(top["adm3_name"], rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Score contribution", fontsize=10)
    ax2.set_title("SRI Component Breakdown (Top 15 Townships)", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, ls=":", alpha=0.4)

    plt.suptitle("Yangon Seismic Risk Assessment", fontsize=13, fontweight="bold")
    plt.tight_layout()
    static_path = os.path.join(OUT_DIR, "risk_map_static.png")
    plt.savefig(static_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {static_path}")

    # ── Interactive Folium map ─────────────────────────────────────────────────
    center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

    for _, row in gdf.iterrows():
        color    = row.get("SRI_color", "#95a5a6")
        sri_val  = row.get("SRI", np.nan)
        sri_cls  = row.get("SRI_class", "Unknown")
        geom_json = row.geometry.__geo_interface__

        popup_txt = (
            f"<b>{row.get('adm3_name','')}</b><br>"
            f"District: {row.get('adm2_name','')}<br>"
            f"<b>SRI = {sri_val:.3f}  ({sri_cls})</b><br><hr>"
            f"Hazard score: {row.get('hazard_score', np.nan):.3f}"
            f"  (A₀ = {row.get('mean_A0', 'N/A')})<br>"
            f"Exposure score: {row.get('exposure_score', np.nan):.3f}"
            f"  ({int(row.get('building_density_km2', 0) or 0)} bldg/km²)<br>"
            f"PGA score: {row.get('pga_score', np.nan):.3f}"
            f"  ({row.get('interp_pga_cms2', np.nan):.2f} cm/s²)<br>"
            f"NEHRP: {row.get('nehrp_class','N/A')}<br>"
            f"Resonance risk: {row.get('resonance_risk','N/A')}"
        )
        folium.GeoJson(
            geom_json,
            style_function=lambda feat, c=color: {
                "fillColor": c, "color": "#222", "weight": 0.8,
                "fillOpacity": 0.70},
            tooltip=f"{row.get('adm3_name','')} — SRI {sri_val:.3f} ({sri_cls})",
            popup=folium.Popup(popup_txt, max_width=240),
        ).add_to(m)

    # Add event station PGA markers
    if has_gm and len(gm_ok):
        for _, row in gm_ok.iterrows():
            if pd.notna(row["lat"]) and pd.notna(row["lon"]):
                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=8,
                    color="black",
                    weight=2,
                    fill=True,
                    fill_color="#9b59b6",
                    fill_opacity=0.85,
                    tooltip=f"Station {row['station']}",
                    popup=folium.Popup(
                        f"<b>Station {row['station']}</b><br>"
                        f"PGA = {row['pga_cms2']:.2f} cm/s²<br>"
                        f"PGV = {row['pgv_cms']:.2f} cm/s<br>"
                        f"Dist = {row['dist_km']:.1f} km",
                        max_width=180),
                ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:10px 14px; border-radius:6px;
                border:1px solid #aaa; font-size:12px; line-height:1.6;">
    <b>Seismic Risk Index (SRI)</b><br>
    """
    for lo, hi, lbl, col in SRI_CLASSES:
        legend_html += (f'<i style="background:{col}; width:14px; height:14px;'
                        f' display:inline-block; margin-right:6px;'
                        f' border-radius:2px;"></i>'
                        f'{lbl} ({lo:.2f}–{hi:.2f})<br>')
    legend_html += (
        "<hr style='margin:4px 0;'>"
        f"<small>Weights: Hazard {W_HAZARD:.0%} | "
        f"Exposure {W_EXPOSURE:.0%} | PGA {W_PGA:.0%}</small>"
        "</div>")
    m.get_root().html.add_child(folium.Element(legend_html))

    html_path = os.path.join(OUT_DIR, "risk_map.html")
    m.save(html_path)
    print(f"Saved → {html_path}")

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n── Top 15 Highest-Risk Townships ──")
    print(risk_df.sort_values("SRI", ascending=False)
          [["adm3_name", "adm2_name", "SRI", "SRI_class",
            "nehrp_class", "resonance_risk"]]
          .head(15).to_string(index=False))

    print("\n── SRI Class Distribution ──")
    print(risk_df["SRI_class"].value_counts().to_string())

    print(f"\n── Scores used (weights: H={W_HAZARD} E={W_EXPOSURE} P={W_PGA}) ──")
    print(risk_df[["adm3_name", "hazard_score", "exposure_score",
                    "pga_score", "SRI", "SRI_class"]]
          .sort_values("SRI", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
