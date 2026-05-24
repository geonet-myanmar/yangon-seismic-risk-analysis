#!/usr/bin/env python3
"""
Module 2 — Spatial Interpolation & Microzonation Mapping
Reads hvsr_results.csv, interpolates f0 and A0 across Yangon,
computes zonal statistics per township, assigns NEHRP site classes,
and exports an interactive HTML map.

Outputs:
  outputs/microzonation_map.html
  outputs/township_site_class.geojson
  outputs/township_microzonation.csv
"""

import os, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import folium
from shapely.geometry import Point
from scipy.interpolate import griddata
from pykrige.ok import OrdinaryKriging

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Site-class thresholds (NEHRP, by site period T0 = 1/f0) ──────────────────
# Approximate proxy mapping using T0 from HVSR
#   T0 < 0.1 s  → A/B (rock)
#   0.1–0.3 s   → C (stiff soil)
#   0.3–0.6 s   → D (soft soil)
#   > 0.6 s     → E (very soft / potential liquefiable)
def assign_nehrp(T0):
    if pd.isna(T0):
        return "Unknown"
    if T0 < 0.1:
        return "A/B (Rock)"
    elif T0 < 0.3:
        return "C (Stiff Soil)"
    elif T0 < 0.6:
        return "D (Soft Soil)"
    else:
        return "E (Very Soft)"

SITE_COLORS = {
    "A/B (Rock)":      "#2ecc71",
    "C (Stiff Soil)":  "#f1c40f",
    "D (Soft Soil)":   "#e67e22",
    "E (Very Soft)":   "#e74c3c",
    "Unknown":         "#95a5a6",
}


def main():
    # ── Load HVSR results ──────────────────────────────────────────────────────
    hvsr_path = os.path.join(OUT_DIR, "hvsr_results.csv")
    if not os.path.exists(hvsr_path):
        raise FileNotFoundError(f"Run module1_hvsr.py first: {hvsr_path}")

    df = pd.read_csv(hvsr_path)
    df_ok = df.dropna(subset=["f0", "A0", "lat", "lon"]).copy()
    df_ok = df_ok[df_ok["status"] == "ok"].copy()
    print(f"Sites with valid HVSR: {len(df_ok)} / {len(df)}")

    # ── Load township boundaries ───────────────────────────────────────────────
    twp_path = os.path.join(BASE_DIR, "yangon_townships.geojson")
    gdf = gpd.read_file(twp_path).to_crs(epsg=4326)

    # ── Spatial interpolation grid ─────────────────────────────────────────────
    bounds  = gdf.total_bounds          # [minx, miny, maxx, maxy]
    grid_res = 0.005                    # ~500 m
    xi = np.arange(bounds[0], bounds[2], grid_res)
    yi = np.arange(bounds[1], bounds[3], grid_res)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    pts = df_ok[["lon", "lat"]].values
    f0_vals  = df_ok["f0"].values
    A0_vals  = df_ok["A0"].values

    # Kriging for f0 (more physically meaningful spatial structure)
    if len(df_ok) >= 4:
        try:
            ok_f0 = OrdinaryKriging(pts[:, 0], pts[:, 1], f0_vals,
                                    variogram_model="linear",
                                    verbose=False, enable_plotting=False)
            f0_krig, _ = ok_f0.execute("grid", xi, yi)
            f0_grid = np.array(f0_krig)
        except Exception as e:
            print(f"  Kriging f0 failed ({e}), falling back to IDW")
            f0_grid = griddata(pts, f0_vals, (xi_grid, yi_grid), method="linear")

        try:
            ok_A0 = OrdinaryKriging(pts[:, 0], pts[:, 1], A0_vals,
                                    variogram_model="linear",
                                    verbose=False, enable_plotting=False)
            A0_krig, _ = ok_A0.execute("grid", xi, yi)
            A0_grid = np.array(A0_krig)
        except Exception as e:
            print(f"  Kriging A0 failed ({e}), falling back to IDW")
            A0_grid = griddata(pts, A0_vals, (xi_grid, yi_grid), method="linear")
    else:
        print("  < 4 sites — using IDW interpolation")
        f0_grid = griddata(pts, f0_vals, (xi_grid, yi_grid), method="nearest")
        A0_grid = griddata(pts, A0_vals, (xi_grid, yi_grid), method="nearest")

    # Clip to 0
    f0_grid = np.clip(f0_grid, 0.01, None)
    A0_grid = np.clip(A0_grid, 0.01, None)
    T0_grid = 1.0 / f0_grid

    # ── Zonal statistics: sample grid points inside each township ─────────────
    rows = []
    for _, twp in gdf.iterrows():
        geom = twp.geometry
        # create mask of grid points inside polygon
        pts_in = []
        for ix, x in enumerate(xi):
            for iy, y in enumerate(yi):
                if geom.contains(Point(x, y)):
                    pts_in.append((iy, ix))
        if pts_in:
            f0_vals_twp = [f0_grid[r, c] for r, c in pts_in]
            A0_vals_twp = [A0_grid[r, c] for r, c in pts_in]
            T0_vals_twp = [T0_grid[r, c] for r, c in pts_in]
            mean_f0 = float(np.nanmean(f0_vals_twp))
            mean_A0 = float(np.nanmean(A0_vals_twp))
            mean_T0 = float(np.nanmean(T0_vals_twp))
        else:
            # fallback: nearest site
            dists  = np.sqrt((pts[:, 0] - twp.geometry.centroid.x) ** 2 +
                             (pts[:, 1] - twp.geometry.centroid.y) ** 2)
            nearest = np.argmin(dists)
            mean_f0 = float(f0_vals[nearest])
            mean_A0 = float(A0_vals[nearest])
            mean_T0 = 1.0 / mean_f0

        site_class = assign_nehrp(mean_T0)
        rows.append(dict(
            adm3_name=twp.adm3_name,
            adm2_name=twp.adm2_name,
            adm3_pcode=twp.adm3_pcode,
            area_sqkm=twp.area_sqkm,
            center_lat=twp.center_lat,
            center_lon=twp.center_lon,
            mean_f0=round(mean_f0, 4),
            mean_A0=round(mean_A0, 4),
            mean_T0=round(mean_T0, 4),
            nehrp_class=site_class,
        ))

    twp_df = pd.DataFrame(rows)
    gdf_out = gdf.merge(twp_df, on="adm3_pcode", how="left",
                        suffixes=("", "_stat"))

    # ── Save CSV and GeoJSON ───────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "township_microzonation.csv")
    twp_df.to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")

    geo_path = os.path.join(OUT_DIR, "township_site_class.geojson")
    gdf_out.to_file(geo_path, driver="GeoJSON")
    print(f"Saved → {geo_path}")

    # ── Static map: f0 and A0 ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, grid, label, cmap in zip(
            axes,
            [f0_grid, A0_grid],
            ["Fundamental Frequency f₀ (Hz)", "Amplification A₀"],
            ["RdYlGn_r", "Reds"]):
        im = ax.imshow(grid, origin="lower", cmap=cmap,
                       extent=[bounds[0], bounds[2], bounds[1], bounds[3]],
                       aspect="auto", alpha=0.7)
        gdf.boundary.plot(ax=ax, color="k", lw=0.5)
        if len(df_ok):
            sc = ax.scatter(df_ok["lon"], df_ok["lat"], c=df_ok["f0" if "f₀" in label else "A0"],
                            cmap=cmap, edgecolors="k", s=60, zorder=5)
            for _, r in df_ok.iterrows():
                ax.annotate(r["site_id"], (r["lon"], r["lat"]),
                            fontsize=6, ha="left", va="bottom")
        plt.colorbar(im, ax=ax, shrink=0.8, label=label)
        ax.set_title(label)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    plt.suptitle("Yangon Seismic Microzonation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    static_path = os.path.join(OUT_DIR, "microzonation_static.png")
    plt.savefig(static_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {static_path}")

    # ── Interactive Folium map ─────────────────────────────────────────────────
    center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

    # Township choropleth (NEHRP class)
    for _, row in gdf_out.iterrows():
        nc  = row.get("nehrp_class", "Unknown")
        col = SITE_COLORS.get(nc, "#95a5a6")
        geom_json = row.geometry.__geo_interface__
        popup_txt = (f"<b>{row.get('adm3_name','')}</b><br>"
                     f"District: {row.get('adm2_name', '')}<br>"
                     f"NEHRP Class: <b>{nc}</b><br>"
                     f"f₀ = {row.get('mean_f0', 'N/A')} Hz<br>"
                     f"T₀ = {row.get('mean_T0', 'N/A')} s<br>"
                     f"A₀ = {row.get('mean_A0', 'N/A')}")
        folium.GeoJson(
            geom_json,
            style_function=lambda feat, c=col: {
                "fillColor": c, "color": "#333", "weight": 0.8,
                "fillOpacity": 0.55},
            tooltip=row.get("adm3_name", ""),
            popup=folium.Popup(popup_txt, max_width=220),
        ).add_to(m)

    # HVSR measurement points
    for _, r in df.iterrows():
        nc  = assign_nehrp(r.get("T0"))
        col = SITE_COLORS.get(nc, "#95a5a6")
        status_ok = r.get("status", "") == "ok"
        icon_color = "blue" if status_ok else "gray"
        popup_txt = (f"<b>{r['site_id']}</b><br>"
                     f"f₀ = {r.get('f0', 'N/A')} Hz<br>"
                     f"T₀ = {r.get('T0', 'N/A')} s<br>"
                     f"A₀ = {r.get('A0', 'N/A')}<br>"
                     f"Status: {r.get('status', '')}<br>"
                     f"Note: {r.get('note', '')}")
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=7,
            color="white",
            weight=1.5,
            fill=True,
            fill_color=col if status_ok else "#95a5a6",
            fill_opacity=0.85,
            popup=folium.Popup(popup_txt, max_width=200),
            tooltip=r["site_id"],
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:10px; border-radius:5px;
                border:1px solid #ccc; font-size:12px;">
    <b>NEHRP Site Class</b><br>
    """
    for cls, col in SITE_COLORS.items():
        if cls == "Unknown":
            continue
        legend_html += (f'<i style="background:{col}; width:12px; height:12px;'
                        f' display:inline-block; margin-right:5px;"></i>{cls}<br>')
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    html_path = os.path.join(OUT_DIR, "microzonation_map.html")
    m.save(html_path)
    print(f"Saved → {html_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n── Township Site-Class Distribution ──")
    if "nehrp_class" in twp_df.columns:
        print(twp_df["nehrp_class"].value_counts().to_string())
    print("\n── Township Microzonation Summary (top 10) ──")
    print(twp_df.sort_values("mean_T0", ascending=False)
          [["adm3_name", "mean_f0", "mean_T0", "mean_A0", "nehrp_class"]]
          .head(10).to_string(index=False))


if __name__ == "__main__":
    main()
