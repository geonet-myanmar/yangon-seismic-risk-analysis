# Yangon Seismic Risk Analysis

A multi-module seismic hazard, site characterisation, and building exposure study
for the Yangon Region, Myanmar, anchored to the **M4.2 earthquake event of 22 May 2026**.

**Live report → [https://geonet-myanmar.github.io/yangon-seismic-risk-analysis/](https://geonet-myanmar.github.io/yangon-seismic-risk-analysis/)**

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Data Sources](#data-sources)
4. [Analysis Workflow](#analysis-workflow)
5. [Module Descriptions](#module-descriptions)
6. [Processing Steps](#processing-steps)
7. [Output Descriptions](#output-descriptions)
8. [Methodology](#methodology)
9. [Limitations and Caveats](#limitations-and-caveats)
10. [Attribution Requirements](#attribution-requirements)
11. [How to Reproduce](#how-to-reproduce)
12. [License](#license)
13. [Citation](#citation)

---

## Overview

This project implements a six-module seismic risk analysis pipeline for the Yangon metropolitan
area using four independently-sourced datasets:

| Component | Data |
|---|---|
| Administrative boundaries | 44 townships, Yangon Region (MIMU) |
| Building footprints | 639,318 polygons from the Global Building Atlas |
| Site characterisation | 25 ambient-noise HVSR recordings (Jan–Feb 2026) |
| Earthquake ground motion | 6-station M4.2 waveform recordings (22 May 2026) |

**Key findings:**
- All 44 townships classify as **NEHRP Site Class E (Very Soft Soil)**, with fundamental
  frequencies f₀ = 0.6–1.5 Hz (T₀ = 0.65–1.67 s), confirming deep Holocene alluvial deposits.
- Recorded PGA values range from 0.6 to 101 cm/s² at 3–25 km epicentral distance.
- **North Okkalapa** is the highest-risk township (SRI = 0.80), combining elevated site
  amplification, high building density, and proximity to the event.
- Five additional townships (Dagon Myothit (North), Insein, Shwepyithar, Mingaladon,
  Mayangone) carry **Moderate** composite risk.

---

## Repository Structure

```
yangon-seismic-risk-analysis/
├── index.html                        # HTML report (GitHub Pages root)
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Excludes large raw data files
├── .nojekyll                         # Disables Jekyll processing for GitHub Pages
│
├── module1_hvsr.py                   # HVSR site characterisation
├── module2_spatial.py                # Spatial interpolation & microzonation maps
├── module3_ground_motion.py          # Earthquake ground-motion processing
├── module4_amplification.py          # SSR amplification cross-check
├── module5_buildings.py              # Building exposure & vulnerability
├── module6_risk.py                   # Integrated seismic risk index
├── run_all.py                        # Orchestrator: runs modules 1–6 in sequence
│
├── yangon_townships.geojson          # Township boundaries (44 townships, WGS84)
│
├── Yangon Event Mseed 4.2 data/      # M4.2 event recordings (MiniSEED + StationXML)
│   ├── *.mseed                       #   [excluded from repo — >40 MB each]
│   └── *.xml                         #   StationXML instrument metadata
│
├── Yangon seismic Microzonation Study/  # Ambient-noise campaign data
│   ├── *.mseed                          #   [excluded from repo — >18 MB each]
│   ├── *.xml                            #   StationXML instrument metadata
│   └── YGN.xlsx                         #   Site metadata (coordinates, dates, filenames)
│
└── outputs/                          # All analysis outputs (committed)
    ├── hvsr_results.csv
    ├── event_ground_motion.csv
    ├── ssr_results.csv
    ├── township_microzonation.csv
    ├── township_exposure.csv
    ├── township_risk_index.csv
    ├── township_site_class.geojson
    ├── township_risk_index.geojson
    ├── microzonation_map.html        # Interactive Folium map
    ├── risk_map.html                 # Interactive risk map
    ├── building_density_map.html     # Interactive building density map
    ├── microzonation_static.png
    ├── risk_map_static.png
    ├── building_exposure_static.png
    ├── resonance_risk_map.png
    ├── ssr_comparison.png
    ├── pga_attenuation.png
    ├── response_spectra_all.png
    ├── event_fas.png
    ├── hvsr_plots/                   # Per-site HVSR curve plots
    ├── ssr_plots/                    # Per-station SSR curve plots
    ├── event_waveforms/              # Per-station waveform plots
    └── response_spectra/             # Per-station response spectra plots
```

> **Note:** MiniSEED waveform files and `yangon_townships_buildings.geojson` (235 MB) are
> excluded from this repository due to GitHub file size limits. To reproduce the full analysis,
> obtain these files from the data providers listed in [Data Sources](#data-sources) and place
> them in the directories above.

---

## Data Sources

### 1. Township Boundaries

| Attribute | Value |
|---|---|
| **Provider** | Myanmar Information Management Unit (MIMU) |
| **Website** | https://www.themimu.info/ |
| **File** | `yangon_townships.geojson` |
| **Coverage** | 44 townships, Yangon Region (Cocokyun Township excluded) |
| **Coordinate System** | WGS84 (EPSG:4326) |
| **Fields** | `adm3_name`, `adm2_name`, `adm3_pcode`, `area_sqkm`, `center_lat`, `center_lon` |
| **License** | Creative Commons Attribution (CC BY) — see [Attribution Requirements](#attribution-requirements) |

MIMU is the primary source for administrative boundary data for Myanmar. The Yangon township
boundaries represent the third administrative level (ADM3) within Yangon Region.

---

### 2. Building Footprints

| Attribute | Value |
|---|---|
| **Provider** | Global Building Atlas (GBA) |
| **Repository** | https://github.com/zhu-xlab/GlobalBuildingAtlas |
| **File** | `yangon_townships_buildings.geojson` *(excluded from repo — 235 MB)* |
| **Coverage** | 639,318 building footprints across 44 Yangon townships |
| **Coordinate System** | WGS84 (EPSG:4326) |
| **Geometry** | Building footprint polygons |
| **License** | See Global Building Atlas repository for license terms |

The Global Building Atlas provides AI-derived building footprints at global scale from
satellite imagery. This dataset was clipped to the 44 Yangon townships covered by this study.

**Reference:**
> Zhu, X. X., et al. (2022). "So2Sat LCZ42: A benchmark dataset for global local climate
> zones classification." *IEEE Journal of Selected Topics in Applied Earth Observations and
> Remote Sensing.*

---

### 3. Seismic Waveform Data — Microzonation Campaign

| Attribute | Value |
|---|---|
| **Provider** | Geographical Society of Myanmar |
| **Folder** | `Yangon seismic Microzonation Study/` |
| **Campaign period** | 20 January 2026 – 10 February 2026 |
| **Instruments** | Raspberry Shake 3-component sensors (RDEF5, RCAF2, R3AA4) |
| **Channels** | EHE, EHN, EHZ at 100 Hz |
| **Sites** | 25 ambient-noise measurement sites (SP01–SP25, no SP22) |
| **File format** | MiniSEED (waveforms) + StationXML (instrument metadata) |
| **Site metadata** | `YGN.xlsx` — coordinates, dates, start times, filenames |
| **Network** | AM (Raspberry Shake public network) |

Site locations span the Yangon metropolitan area from approximately 16.75°N to 16.95°N and
96.02°E to 96.28°E. Recording durations range from approximately 1 to 12 hours per site.

**Data quality note:** 10 of 25 sites have MiniSEED files that contain only horizontal
channels (EHE, EHN) without a vertical (EHZ) channel. These sites (SP04–SP09, SP12, SP14–SP16)
cannot be processed for HVSR and are excluded from site-classification analysis.

---

### 4. Seismic Waveform Data — M4.2 Earthquake Event

| Attribute | Value |
|---|---|
| **Provider** | Geographical Society of Myanmar |
| **Folder** | `Yangon Event Mseed 4.2 data/` |
| **Event date** | 22 May 2026 |
| **Magnitude** | M4.2 |
| **Approximate event time** | ~07:02 UTC (derived from S7E9A STA/LTA detection) |
| **Stations** | R7183, R8939, RDBC5, RF99C, S7E9A, T9951 |
| **Channels** | EHZ, ENE, ENN (mixed EH/EN depending on station model) |
| **File format** | MiniSEED (one file per station, full-day recording) + StationXML |
| **Network** | AM (Raspberry Shake public network) |

Station epicentral distances range from 3.4 km (R7183) to 42.0 km (T9951). Station T9951
was excluded from quantitative analysis due to a response-removal mismatch between the
recorded instrument epoch and the StationXML metadata.

---

## Analysis Workflow

The six analysis modules form a directed dependency graph:

```
Module 1 (HVSR)
    │
    ├──→ Module 2 (Spatial Microzonation)
    │         │
    │         └──→ Module 6 (Risk Index) ←──┐
    │                                        │
    └──→ Module 4 (SSR Amplification) ←──── │
              ↑                              │
Module 3 (Ground Motion) ──────────────────→┘
              │                              │
              └──────────────────────────────┘
                                             ↑
Module 5 (Building Exposure) ───────────────→┘
```

**Execution order:** 1 → 3 → 2 → 4 → 5 → 6 (or run `python run_all.py`)

---

## Module Descriptions

### Module 1 — HVSR Site Characterisation (`module1_hvsr.py`)

Computes Horizontal-to-Vertical Spectral Ratio (HVSR) from 3-component ambient-noise
recordings at each microzonation site.

**Inputs:**
- MiniSEED files and StationXML from `Yangon seismic Microzonation Study/`
- `YGN.xlsx` (site metadata: coordinates, recording dates and times, filenames)

**Outputs:**
- `outputs/hvsr_results.csv` — f₀, A₀, T₀, number of valid windows, status per site
- `outputs/hvsr_plots/*.png` — HVSR curve plots for valid sites

**Algorithm summary:**
1. Load 3-component MiniSEED; remove instrument response → velocity
2. Slice a 2-hour analysis window (3-stage fallback: UTC → local Myanmar time → full recording)
3. Divide into 30-second windows with 50% overlap; apply Hanning taper
4. Reject transient-contaminated windows (RMS(Z) > 4 × median RMS(Z))
5. Compute FFT on each window; smooth with Konno-Ohmachi (b = 40)
6. Compute geometric-mean H/V ratio per window; stack across windows
7. Identify f₀ as the peak in the 0.5–12 Hz search band

---

### Module 2 — Spatial Microzonation (`module2_spatial.py`)

Interpolates HVSR-derived site parameters (f₀, A₀) to township-level spatial averages
and produces microzonation maps.

**Inputs:**
- `outputs/hvsr_results.csv`
- `yangon_townships.geojson`

**Outputs:**
- `outputs/township_microzonation.csv` — mean f₀, A₀, T₀, NEHRP class per township
- `outputs/township_site_class.geojson` — townships with site class attributes
- `outputs/microzonation_static.png` — static choropleth map
- `outputs/microzonation_map.html` — interactive Folium map

**Algorithm summary:**
1. Grid HVSR point data to 0.005° resolution using Ordinary Kriging (PyKrige)
2. For each township, extract grid cells within the polygon boundary
3. Compute spatial mean of f₀, A₀, T₀ per township
4. Assign NEHRP site class: T₀ > 0.5 s → Class E (Very Soft Soil)

---

### Module 3 — Earthquake Ground Motion (`module3_ground_motion.py`)

Processes M4.2 event MiniSEED recordings to compute peak ground-motion parameters
and response spectra at each seismic station.

**Inputs:**
- MiniSEED and StationXML files from `Yangon Event Mseed 4.2 data/`
- Station coordinates from StationXML metadata

**Outputs:**
- `outputs/event_ground_motion.csv` — PGA, PGV, Arias Intensity per station
- `outputs/event_waveforms/*.png` — acceleration waveform plots
- `outputs/response_spectra/*.png` — 5%-damped pseudo-acceleration response spectra
- `outputs/response_spectra_all.png` — all-stations comparison
- `outputs/pga_attenuation.png` — PGA vs. epicentral distance
- `outputs/event_fas.png` — Fourier amplitude spectra

**Algorithm summary:**
1. Load MiniSEED; detect mixed-channel stations (EHZ vertical + ENE/ENN horizontals)
2. Apply STA/LTA event detection (1 s / 10 s windows; thresholds 4.0 → 2.5 → 1.8);
   fall back to peak-amplitude detection if no trigger found
3. Cut 30 s pre-onset + 90 s post-onset window
4. Remove instrument response → acceleration (pre_filt = 0.1/0.2/45/50 Hz, water_level = 60)
5. Apply PGA sanity cap: values > 5,000 cm/s² are flagged as response-removal failures
6. Compute PGA (geometric mean of horizontal channels), PGV (integrated), Arias Intensity
7. Compute 5%-damped response spectra (Newmark β-method) and Fourier amplitude spectra

---

### Module 4 — Site Amplification Cross-Check (`module4_amplification.py`)

Computes Standard Spectral Ratio (SSR) from event recordings and compares against the
nearest HVSR measurement to validate site characterisation consistency.

**Inputs:**
- MiniSEED and StationXML from `Yangon Event Mseed 4.2 data/`
- `outputs/hvsr_results.csv`

**Outputs:**
- `outputs/ssr_results.csv` — SSR f₀, A₀, comparison with nearest HVSR site
- `outputs/ssr_plots/*.png` — per-station SSR curves
- `outputs/ssr_comparison.png` — all-stations SSR comparison plot

**Algorithm summary:**
1. Compute horizontal Fourier amplitude spectra for each station (same event window as Module 3)
2. Divide each station spectrum by the reference station (S7E9A) spectrum
3. Smooth SSR with Konno-Ohmachi (b = 40)
4. Identify SSR f₀ in 0.3–10 Hz band
5. Match each station to the nearest HVSR site by great-circle distance
6. Report Δf₀ = |SSR f₀ − HVSR f₀| for each station

---

### Module 5 — Building Exposure and Vulnerability (`module5_buildings.py`)

Quantifies the building stock exposed to seismic hazard in each township, computes
density and coverage metrics, and derives a resonance risk proxy.

**Inputs:**
- `yangon_townships_buildings.geojson` *(239 MB — not in repo; see Data Sources)*
- `yangon_townships.geojson`
- `outputs/township_microzonation.csv`

**Outputs:**
- `outputs/township_exposure.csv` — building count, density, coverage ratio per township
- `outputs/building_exposure_static.png` — static maps of density and coverage
- `outputs/resonance_risk_map.png` — resonance risk by township
- `outputs/building_density_map.html` — interactive Folium map

**Algorithm summary:**
1. Project building footprints to UTM Zone 47N (EPSG:32647); compute footprint area
2. Spatial join buildings to townships using `geopandas.sjoin` with `predicate='within'`
3. Aggregate building count, total footprint, mean/median/p90 footprint per township
4. Compute density (buildings/km²) and coverage (footprint area / township area)
5. Estimate building natural period proxy: T_bldg ≈ 0.1 × √(footprint_m² / 10)
6. Compare T_bldg to site T₀: classify resonance risk as Low / Moderate / High

---

### Module 6 — Integrated Seismic Risk Index (`module6_risk.py`)

Combines site hazard, building exposure, and recorded ground motion into a composite
Seismic Risk Index (SRI) for each township.

**Inputs:**
- `outputs/township_microzonation.csv`
- `outputs/township_exposure.csv`
- `outputs/event_ground_motion.csv`
- `yangon_townships.geojson`

**Outputs:**
- `outputs/township_risk_index.csv` — SRI scores and class per township
- `outputs/township_risk_index.geojson` — townships with SRI attributes
- `outputs/risk_map_static.png` — static risk map
- `outputs/risk_map.html` — interactive risk map

**Algorithm summary:**
1. **Hazard score (H):** normalise (mean_A₀ × mean_f₀) to [0, 1] across all townships
2. **Exposure score (E):** normalise building_density_km² to [0, 1]
3. **PGA score (P):** IDW-interpolate station PGA to township centroids; normalise to [0, 1]
4. **SRI = 0.40·H + 0.35·E + 0.25·P**
5. Classify: Very Low < 0.20 ≤ Low < 0.45 ≤ Moderate < 0.70 ≤ High

---

## Processing Steps

### Prerequisites

```bash
# Python 3.10+ required
pip install -r requirements.txt
```

### Required data files (not in repository)

Place the following files in their respective directories before running:

| File(s) | Directory | Source |
|---|---|---|
| `*.mseed` (6 files) | `Yangon Event Mseed 4.2 data/` | Geographical Society of Myanmar |
| `*.mseed` (21 files) | `Yangon seismic Microzonation Study/` | Geographical Society of Myanmar |
| `yangon_townships_buildings.geojson` | Project root | Global Building Atlas |

### Run all modules

```bash
python run_all.py
```

### Run individual modules

```bash
python module1_hvsr.py          # ~2–3 min
python module2_spatial.py       # ~1 min  (requires outputs/hvsr_results.csv)
python module3_ground_motion.py # ~3–5 min
python module4_amplification.py # ~3–5 min (requires outputs/hvsr_results.csv)
python module5_buildings.py     # ~3–5 min (slow — loads 235 MB GeoJSON)
python module6_risk.py          # ~30 s   (requires all prior outputs)
```

### Expected outputs

After a full run, the `outputs/` directory will contain:
- 7 CSV files (site, ground-motion, SSR, microzonation, exposure, risk, SSR results)
- 2 GeoJSON files (site class, risk index)
- 3 interactive HTML maps (Folium)
- ~40 PNG plots

---

## Output Descriptions

| File | Description |
|---|---|
| `hvsr_results.csv` | Per-site f₀ (Hz), T₀ (s), A₀ (H/V amplitude), window count, status |
| `event_ground_motion.csv` | Per-station PGA (cm/s²), PGV (cm/s), Arias Intensity (m/s), onset UTC |
| `ssr_results.csv` | Per-station SSR f₀, A₀, nearest HVSR site, Δf₀ comparison |
| `township_microzonation.csv` | Per-township mean f₀, A₀, T₀, NEHRP class |
| `township_exposure.csv` | Per-township building count, density (bldg/km²), coverage ratio |
| `township_risk_index.csv` | Per-township hazard score, exposure score, PGA score, SRI, class |
| `township_site_class.geojson` | Townships polygon with microzonation attributes |
| `township_risk_index.geojson` | Townships polygon with full SRI attributes |
| `microzonation_map.html` | Interactive choropleth: mean T₀ and NEHRP class |
| `risk_map.html` | Interactive choropleth: SRI by township |
| `building_density_map.html` | Interactive choropleth: building density |

---

## Methodology

### HVSR (Nakamura Technique)

The Horizontal-to-Vertical Spectral Ratio technique (Nakamura, 1989) estimates the
fundamental resonance frequency of a soil column from single-station ambient-noise recordings.
The H/V peak frequency f₀ is related to the average shear-wave velocity V̄ₛ and depth to
bedrock H by the quarter-wavelength relationship: f₀ = V̄ₛ / (4H).

Processing follows SESAME guidelines (2004):
- Minimum 10 non-overlapping windows
- Peak criterion: A₀ ≥ 2.0 and f₀ ≠ f_instrument
- Clarity criterion: σ_f / f₀ < 0.25 (not strictly enforced given short deployments)

Konno-Ohmachi spectral smoothing (bandwidth parameter b = 40) is applied to each window
before computing the H/V ratio. The geometric mean of the two horizontal components is used
as the numerator.

### Spatial Interpolation

HVSR point observations (14 valid sites) are interpolated to a regular 0.005° grid
(~550 m spacing) using Ordinary Kriging with a linear variogram model. If the Kriging
covariance matrix is ill-conditioned, the method falls back to inverse-distance weighting
(power = 2). Township-level statistics are extracted by taking the mean of all grid cells
whose centroids fall within each township polygon.

### Ground-Motion Processing

Instrument response is removed using the pole-zero representation in StationXML, with a
pre-filter cosine taper at 0.1–0.2 Hz (low) and 45–50 Hz (high), and a water level of 60 dB
to avoid near-zero division at low frequencies. The output is ground acceleration in m/s².

PGA is taken as the larger of the two geometric-mean horizontal peak accelerations:
PGA = max(|ENE|, |ENN|) after instrument response removal.

Arias Intensity is computed as: Iₐ = (π / 2g) ∫ aₕ²(t) dt, using the trapezoidal rule.

5%-damped pseudo-acceleration response spectra are computed using the Newmark
constant-average-acceleration (β = 0.25) time-stepping algorithm at 50 logarithmically
spaced periods from 0.01 to 4.0 s.

### Standard Spectral Ratio (SSR)

SSR amplification relative to the reference station S7E9A is computed in the frequency
domain. The signal window (30 s) and the pre-signal noise window (5 s) are both
Hanning-tapered before computing the FFT. The ratio is smoothed with Konno-Ohmachi (b = 40).
S7E9A is used as the reference as it is the most distant station (25 km) and thus expected
to have the lowest site amplification relative to the other urban stations.

### Risk Index

The composite Seismic Risk Index is a weighted linear combination of three normalised
sub-scores, each scaled to [0, 1] across the 44-township study area:

```
SRI = 0.40 × H + 0.35 × E + 0.25 × P
```

Where:
- **H (Hazard):** proportional to the product (mean_A₀ × mean_f₀), capturing both
  amplification magnitude and frequency content
- **E (Exposure):** proportional to building density (buildings/km²), as a proxy for
  the number of structures at risk
- **P (PGA):** proportional to the IDW-interpolated PGA from the nearest event stations

SRI thresholds: Very Low < 0.20 | Low 0.20–0.45 | Moderate 0.45–0.70 | High ≥ 0.70

---

## Limitations and Caveats

1. **Incomplete HVSR coverage:** 10 of 25 microzonation sites (SP04–SP09, SP12, SP14–SP16)
   have missing EHZ (vertical) channels in their MiniSEED recordings and cannot be processed
   for HVSR. Southern and western townships are under-represented.

2. **Uniform NEHRP-E classification:** All 44 townships classify as NEHRP Site Class E.
   This reflects genuine geology but limits spatial discrimination. No bedrock reference
   sites are available within the measurement network to resolve sub-class differences or
   estimate absolute V̄ₛ30 values.

3. **SP01 anomalous A₀:** Site SP01 (NorthOkkalar) yields A₀ = 148, far outside the
   expected range (2–15). This likely results from an instrument-response epoch mismatch
   in the RDEF5 StationXML for that deployment. SP01 amplification values should be
   treated with caution.

4. **Event time uncertainty:** Most event stations used peak-amplitude fallback detection.
   Only S7E9A provided a reliable STA/LTA trigger at 07:02 UTC. Station R7183 triggered
   at 23:43 UTC, which may correspond to a different local event or a noise burst rather
   than the M4.2 main shock.

5. **T9951 excluded:** Station T9951 (42 km epicentral distance) produced a physically
   impossible PGA (~10⁷ cm/s²) due to a mismatch between the recorded instrument epoch
   and the available StationXML. It is excluded from all quantitative results.

6. **SSR reference station:** S7E9A is a soft-soil site, not a bedrock reference.
   SSR amplification values are relative, not absolute site factors.

7. **Resonance risk proxy:** Building natural periods are estimated from footprint area
   using a size proxy, not from actual storey counts or structural type data. All townships
   return "Low" resonance risk under this proxy; the result is conservative.

8. **SRI weight assumptions:** The weights (H = 0.40, E = 0.35, P = 0.25) are
   expert-judgement assignments and have not been calibrated against observed loss data.

9. **No epistemic uncertainty quantification:** This study does not propagate uncertainties
   in site parameters, ground-motion estimates, or model assumptions.

---

## Attribution Requirements

If you use this analysis or any of its component datasets, please comply with the following
attribution requirements:

### Township Boundaries (MIMU)

> Myanmar Information Management Unit (MIMU). *Yangon Region Township Boundaries.*
> Retrieved from https://www.themimu.info/. Licensed under CC BY.

When publishing maps or analyses derived from MIMU data, include the statement:
**"Administrative boundaries provided by the Myanmar Information Management Unit (MIMU)."**

### Building Footprints (Global Building Atlas)

> Zhu, X. X., et al. *Global Building Atlas.* Helmholtz AI / Technical University of Munich.
> https://github.com/zhu-xlab/GlobalBuildingAtlas

Please consult the GlobalBuildingAtlas repository for the current license terms and required
attribution text before redistribution.

### Seismic Waveform Data (Geographical Society of Myanmar)

> Seismic waveform data (microzonation campaign and M4.2 event recordings) provided by
> the **Geographical Society of Myanmar**. These data are proprietary and are not included
> in this repository. Requests for access should be directed to the Society directly.

Do not redistribute the raw waveform files without permission from the
Geographical Society of Myanmar.

---

## How to Reproduce

1. **Clone this repository:**
   ```bash
   git clone https://github.com/geonet-myanmar/yangon-seismic-risk-analysis.git
   cd yangon-seismic-risk-analysis
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Obtain the excluded large data files** from the data providers (see [Data Sources](#data-sources))
   and place them in the correct directories as described in [Processing Steps](#processing-steps).

4. **Run the full pipeline:**
   ```bash
   python run_all.py
   ```

5. **View the report locally:**
   Open `index.html` in any modern web browser.

---

## License

The **analysis code** in this repository (`.py` files) is released under the
[MIT License](LICENSE).

The **analysis outputs** (`outputs/` directory) are derived from the datasets listed above;
please comply with the original data licenses when using or redistributing them.

The **raw waveform data** (`*.mseed`, `*.xml`, `YGN.xlsx`) from the Geographical Society
of Myanmar and `yangon_townships_buildings.geojson` from the Global Building Atlas are
**not included** in this repository. Their use is subject to the terms of their respective
providers.

---

## Citation

If you use this analysis in academic or professional work, please cite:

```
Geographical Society of Myanmar and geonet-myanmar (2026).
Yangon Seismic Risk Analysis: HVSR Site Characterisation, Ground Motion,
and Building Exposure for the Yangon Region, Myanmar.
GitHub repository: https://github.com/geonet-myanmar/yangon-seismic-risk-analysis
```

---

*Analysis performed using ObsPy, GeoPandas, PyKrige, and related open-source Python libraries.
Seismic data provided by the Geographical Society of Myanmar. Administrative boundaries
by MIMU. Building footprints by the Global Building Atlas.*
