#!/usr/bin/env python3
"""
Module 3 — Ground Motion Analysis (Earthquake Event Data)
Processes 6-station M4.2 Yangon earthquake recordings.

Outputs:
  outputs/event_ground_motion.csv      PGA, PGV, Ia, epicentral distance
  outputs/response_spectra/            response-spectra plots per station
  outputs/event_waveforms/             waveform plots per station
  outputs/event_fas.png                Fourier amplitude spectra comparison
  outputs/response_spectra_all.png     all stations on one plot
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import detrend as sp_detrend
from obspy import read, read_inventory, Stream
from obspy.core import UTCDateTime
from obspy.signal.trigger import recursive_sta_lta, trigger_onset

warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
EVENT_DIR = os.path.join(BASE_DIR, "Yangon Event Mseed 4.2 data")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
SPEC_DIR  = os.path.join(OUT_DIR, "response_spectra")
WAVE_DIR  = os.path.join(OUT_DIR, "event_waveforms")
for d in [SPEC_DIR, WAVE_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Response spectra periods ──────────────────────────────────────────────────
T_PERIODS  = np.concatenate([
    np.arange(0.01, 0.10, 0.01),
    np.arange(0.10, 1.00, 0.05),
    np.arange(1.0,  5.01, 0.25),
])
DAMPING    = 0.05  # 5%

# ── Known event coordinates (Yangon M4.2, 2026-05-22) ───────────────────────
# Will be refined by STA/LTA but epicentre used for distance computation
EVENT_LAT = 16.870
EVENT_LON = 96.165

# Pre-filter for instrument response removal
PRE_FILT = (0.1, 0.5, 45.0, 50.0)


# ── Helpers ───────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi   = np.radians(lat2 - lat1)
    dlam   = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def compute_response_spectrum(acc: np.ndarray, dt: float,
                               periods=T_PERIODS, zeta=DAMPING) -> np.ndarray:
    """
    5%-damped pseudo-spectral acceleration (PSA) via Newmark β (linear accel).
    acc   : acceleration time-series (m/s²)
    dt    : time step (s)
    Returns: PSA array (g) for each period in `periods`
    """
    n   = len(acc)
    psa = np.zeros(len(periods))
    beta  = 0.25
    gamma = 0.50

    for k, T in enumerate(periods):
        if T <= 0:
            psa[k] = np.max(np.abs(acc)) / 9.81
            continue
        omega = 2 * np.pi / T
        c     = 2 * zeta * omega     # per unit mass
        k_eff = omega ** 2            # per unit mass

        u   = 0.0
        v   = 0.0
        a   = 0.0
        max_disp = 0.0

        # Newmark β constants
        a1 = 1.0 / (beta * dt ** 2) + gamma * c / (beta * dt)
        a2 = 1.0 / (beta * dt) + (gamma / beta - 1) * c
        a3 = (1 / (2 * beta) - 1) + dt * c * (gamma / (2 * beta) - 1)
        k_hat = k_eff + a1

        for i in range(n - 1):
            dp_eff = -(acc[i + 1] - acc[i]) + a1 * u + a2 * v + a3 * a
            du     = dp_eff / k_hat
            dv     = gamma / (beta * dt) * du - gamma / beta * v + dt * (1 - gamma / (2 * beta)) * a
            da     = du / (beta * dt ** 2) - v / (beta * dt) - (1 / (2 * beta)) * a
            u += du
            v += dv
            a += da
            if abs(u) > max_disp:
                max_disp = abs(u)

        # PSA = ω² × max(|u|) in g
        psa[k] = omega ** 2 * max_disp / 9.81

    return psa


def compute_arias(acc: np.ndarray, dt: float) -> float:
    """Arias Intensity (m/s)."""
    return (np.pi / (2 * 9.81)) * np.trapezoid(acc ** 2, dx=dt)


def detect_event_window(z_trace, pre_s=30.0, post_s=90.0):
    """
    Use STA/LTA to find the P-wave onset.
    Returns (t_onset, t_start, t_end) as UTCDateTime, or None if not found.
    """
    sr = z_trace.stats.sampling_rate
    sta_samp = int(1.0  * sr)
    lta_samp = int(10.0 * sr)
    cft = recursive_sta_lta(z_trace.data.astype(float), sta_samp, lta_samp)
    triggers = trigger_onset(cft, 4.0, 1.5)
    if len(triggers) == 0:
        return None
    # Pick the strongest trigger
    best = max(triggers, key=lambda t: cft[t[0]])
    onset = z_trace.stats.starttime + best[0] / sr
    return onset, onset - pre_s, onset + post_s


def get_station_coords(inv, t_event):
    """Return (lat, lon) of station valid at t_event."""
    for net in inv:
        for sta in net:
            for ch in sta.channels:
                if (ch.start_date is None or ch.start_date <= t_event) and \
                   (ch.end_date is None or ch.end_date >= t_event):
                    return sta.latitude, sta.longitude
    # fallback: last epoch
    lats, lons = [], []
    for net in inv:
        for sta in net:
            lats.append(sta.latitude)
            lons.append(sta.longitude)
    if lats:
        return lats[-1], lons[-1]
    return np.nan, np.nan


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # ── Discover station files ─────────────────────────────────────────────────
    mseed_files = sorted(glob.glob(os.path.join(EVENT_DIR, "*.mseed")))
    xml_files   = sorted(glob.glob(os.path.join(EVENT_DIR, "*.xml")))

    # Map station code → (mseed_path, xml_path)
    stations = {}
    for mf in mseed_files:
        code = None
        st_tmp = read(mf, headonly=True)
        code = st_tmp[0].stats.station
        stations[code] = {"mseed": mf, "xml": None}
    for xf in xml_files:
        inv_tmp = read_inventory(xf)
        for net in inv_tmp:
            for sta in net:
                if sta.code in stations:
                    stations[sta.code]["xml"] = xf

    records = []
    all_psa  = {}   # station → PSA array
    all_fas  = {}   # station → (freqs, fas)

    for sta_code, paths in sorted(stations.items()):
        print(f"\nProcessing station {sta_code} ...", flush=True)
        if paths["xml"] is None:
            print(f"  No XML for {sta_code} — skipping")
            continue

        # ── Load ──────────────────────────────────────────────────────────────
        st_raw = read(paths["mseed"])
        inv    = read_inventory(paths["xml"])
        st_raw.merge(method=1, fill_value="interpolate")

        # Build best 3-component set:
        # Use EHZ for vertical when available; use ENE/ENN for horizontals when
        # EHE/EHN are absent (common on Raspberry Shake 3D stations).
        chans_avail = set(tr.stats.channel for tr in st_raw)
        has_ehz = any(c == "EHZ" for c in chans_avail)
        has_ehe = any(c == "EHE" for c in chans_avail)
        has_ehn = any(c == "EHN" for c in chans_avail)
        has_enz = any(c == "ENZ" for c in chans_avail)
        has_ene = any(c == "ENE" for c in chans_avail)
        has_enn = any(c == "ENN" for c in chans_avail)

        if has_ehz and has_ehe and has_ehn:
            st = st_raw.select(channel="EH*").copy()
        elif has_enz and has_ene and has_enn:
            st = st_raw.select(channel="EN*").copy()
        elif has_ehz and has_ene and has_enn:
            # Mixed: EHZ vertical + EN horizontals
            st = Stream([tr for tr in st_raw
                         if tr.stats.channel in ("EHZ", "ENE", "ENN")]).copy()
        elif any(c.startswith("EH") for c in chans_avail):
            st = st_raw.select(channel="EH*").copy()
        else:
            st = st_raw.select(channel="EN*").copy()

        if len(st) == 0:
            print(f"  No usable channels for {sta_code}")
            continue

        # ── Detect event window (scan full day with low threshold) ────────────
        z_cands = [tr for tr in st if tr.stats.channel.endswith("Z")]
        if not z_cands:
            print(f"  No Z channel for {sta_code}")
            continue
        z_tr = z_cands[0]

        # Try progressively lower thresholds to catch the event
        detection = None
        for thr in (4.0, 2.5, 1.8):
            sr_z   = z_tr.stats.sampling_rate
            cft    = recursive_sta_lta(z_tr.data.astype(float),
                                       int(1.0 * sr_z), int(10.0 * sr_z))
            trigs  = trigger_onset(cft, thr, thr * 0.4)
            if len(trigs):
                best = max(trigs, key=lambda t: cft[t[0]])
                onset = z_tr.stats.starttime + best[0] / sr_z
                detection = (onset, onset - 30.0, onset + 90.0)
                break

        if detection is None:
            # Hard fallback: use the 30-s window around the global maximum of |Z|
            z_data = z_tr.data.astype(float)
            pk_idx = int(np.argmax(np.abs(z_data)))
            onset  = z_tr.stats.starttime + pk_idx / z_tr.stats.sampling_rate
            detection = (onset, onset - 30.0, onset + 90.0)
            print(f"  STA/LTA found no trigger — using peak-amplitude fallback: {onset}")
        else:
            t_onset, t_start, t_end = detection
            print(f"  Event onset: {t_onset}  window: [{t_start}, {t_end}]")

        t_onset, t_start, t_end = detection

        # ── Cut event window ───────────────────────────────────────────────────
        st_ev = st.slice(t_start, t_end).copy()
        if len(st_ev) == 0 or max(tr.stats.npts for tr in st_ev) < 100:
            print(f"  Empty slice for {sta_code}")
            continue

        # ── Remove instrument response → acceleration ─────────────────────────
        try:
            st_ev.remove_response(inventory=inv, output="ACC",
                                   pre_filt=PRE_FILT, water_level=60)
        except Exception as e:
            print(f"  Response removal warning: {e}")

        # Clean NaN/Inf introduced by response removal
        for tr in st_ev:
            tr.data = np.where(np.isfinite(tr.data), tr.data, 0.0)

        # ── Station coordinates ────────────────────────────────────────────────
        sta_lat, sta_lon = get_station_coords(inv, t_onset)
        dist_km = haversine(EVENT_LAT, EVENT_LON, sta_lat, sta_lon) if (
            np.isfinite(sta_lat) and np.isfinite(sta_lon)) else np.nan

        # ── Compute metrics per component; keep worst-case horizontal ─────────
        dt = 1.0 / st_ev[0].stats.sampling_rate
        metrics = {}
        for tr in st_ev:
            ch  = tr.stats.channel
            acc = sp_detrend(tr.data.astype(float), type="linear")
            pga = float(np.max(np.abs(acc)))
            # integrate to velocity
            vel = np.cumsum(acc) * dt
            vel -= np.mean(vel)
            pgv = float(np.max(np.abs(vel)))
            ia  = compute_arias(acc, dt)
            metrics[ch] = dict(pga_ms2=pga, pgv_ms=pgv, ia=ia)

        # Horizontal channels
        h_chans = [ch for ch in metrics if not ch.endswith("Z")]
        z_chans = [ch for ch in metrics if ch.endswith("Z")]
        pga_h = max((metrics[c]["pga_ms2"] for c in h_chans), default=np.nan)
        pgv_h = max((metrics[c]["pgv_ms"]  for c in h_chans), default=np.nan)
        ia_h  = max((metrics[c]["ia"]      for c in h_chans), default=np.nan)
        pga_z = metrics[z_chans[0]]["pga_ms2"] if z_chans else np.nan

        pga_g  = pga_h / 9.81
        pga_gal = pga_h * 100      # cm/s²
        pgv_cms = pgv_h * 100      # cm/s

        print(f"  PGA = {pga_gal:.3f} cm/s²  PGV = {pgv_cms:.3f} cm/s  "
              f"Ia = {ia_h:.4f} m/s  dist = {dist_km:.1f} km")

        # ── Response Spectra (strongest horizontal channel) ───────────────────
        if h_chans:
            best_h = max(h_chans, key=lambda c: metrics[c]["pga_ms2"])
            tr_h   = st_ev.select(channel=best_h)[0]
            acc_h  = sp_detrend(tr_h.data.astype(float), type="linear")
            psa    = compute_response_spectrum(acc_h, dt)
            all_psa[sta_code] = psa

            # ── Fourier Amplitude Spectrum ─────────────────────────────────────
            n_fft = len(acc_h)
            fas   = np.abs(np.fft.rfft(acc_h * np.hanning(n_fft))) * dt
            freqs = np.fft.rfftfreq(n_fft, d=dt)
            # smooth
            smooth_fas = fas.copy()
            if len(freqs) > 10:
                from scipy.signal import savgol_filter
                win = min(21, len(smooth_fas) // 10 * 2 + 1)
                if win >= 5:
                    smooth_fas = savgol_filter(smooth_fas, win, 3)
            all_fas[sta_code] = (freqs, smooth_fas)

            # ── Plot response spectrum ─────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.loglog(T_PERIODS, psa, color="navy", lw=1.8)
            ax.set_xlabel("Period T (s)", fontsize=11)
            ax.set_ylabel("PSA (g)", fontsize=11)
            ax.set_title(f"Station {sta_code} — 5% Damped Response Spectrum\n"
                         f"PGA = {pga_gal:.2f} cm/s²,  dist = {dist_km:.1f} km",
                         fontsize=10)
            ax.grid(True, which="both", ls=":", alpha=0.5)
            ax.set_xlim([0.01, 5])
            plt.tight_layout()
            plt.savefig(os.path.join(SPEC_DIR, f"{sta_code}_rs.png"), dpi=120)
            plt.close(fig)

        # ── Plot waveforms ─────────────────────────────────────────────────────
        n_ch = len(st_ev)
        fig, axes = plt.subplots(n_ch, 1, figsize=(10, 2.5 * n_ch), sharex=True)
        if n_ch == 1:
            axes = [axes]
        t_arr = np.arange(st_ev[0].stats.npts) / st_ev[0].stats.sampling_rate - 30
        for ax, tr in zip(axes, st_ev):
            ax.plot(t_arr[:len(tr.data)],
                    sp_detrend(tr.data.astype(float), type="linear") * 100,
                    color="k", lw=0.7)
            ax.set_ylabel(f"{tr.stats.channel}\n(cm/s²)", fontsize=8)
            ax.axvline(0, color="red", ls="--", lw=0.8, label="P onset")
            ax.grid(True, ls=":", alpha=0.4)
        axes[-1].set_xlabel("Time relative to P onset (s)", fontsize=10)
        axes[0].set_title(f"Station {sta_code} — M4.2 Yangon Event Waveforms", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(WAVE_DIR, f"{sta_code}_waveform.png"), dpi=120)
        plt.close(fig)

        # Sanity cap: PGA > 5000 cm/s² is physically impossible for a local M4.2
        MAX_PGA = 5000.0
        if pga_gal > MAX_PGA:
            print(f"  WARNING: PGA={pga_gal:.0f} cm/s² exceeds sanity cap — response removal likely failed. Setting to NaN.")
            pga_gal = np.nan
            pgv_cms = np.nan
            ia_h    = np.nan

        records.append(dict(
            station=sta_code,
            lat=round(float(sta_lat), 5) if np.isfinite(sta_lat) else np.nan,
            lon=round(float(sta_lon), 5) if np.isfinite(sta_lon) else np.nan,
            dist_km=round(float(dist_km), 2) if np.isfinite(dist_km) else np.nan,
            pga_cms2=round(pga_gal, 4),
            pgv_cms=round(pgv_cms, 4),
            pga_g=round(pga_g, 6),
            ia_ms=round(float(ia_h), 6),
            event_onset=str(t_onset),
            channels="|".join(sorted(metrics.keys())),
        ))

    # ── Save ground-motion table ───────────────────────────────────────────────
    gm_df = pd.DataFrame(records)
    csv_path = os.path.join(OUT_DIR, "event_ground_motion.csv")
    gm_df.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}")
    print(gm_df[["station", "dist_km", "pga_cms2", "pgv_cms", "ia_ms"]].to_string(index=False))

    # ── All-stations response spectra comparison ───────────────────────────────
    if all_psa:
        fig, ax = plt.subplots(figsize=(9, 5))
        cmap = plt.cm.get_cmap("tab10", len(all_psa))
        for idx, (sta, psa) in enumerate(sorted(all_psa.items())):
            ax.loglog(T_PERIODS, psa, lw=1.6, color=cmap(idx), label=sta)
        ax.set_xlabel("Period T (s)", fontsize=12)
        ax.set_ylabel("PSA (g)", fontsize=12)
        ax.set_title("5%-Damped Response Spectra — All Stations (M4.2 Yangon Event)",
                     fontsize=11)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.set_xlim([0.01, 5])
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "response_spectra_all.png"), dpi=150)
        plt.close(fig)
        print(f"Saved → {os.path.join(OUT_DIR, 'response_spectra_all.png')}")

    # ── Fourier spectra comparison ─────────────────────────────────────────────
    if all_fas:
        fig, ax = plt.subplots(figsize=(9, 5))
        cmap = plt.cm.get_cmap("tab10", len(all_fas))
        for idx, (sta, (freqs, fas)) in enumerate(sorted(all_fas.items())):
            mask = (freqs > 0.1) & (freqs < 30)
            ax.loglog(freqs[mask], fas[mask], lw=1.4, color=cmap(idx), label=sta)
        ax.set_xlabel("Frequency (Hz)", fontsize=12)
        ax.set_ylabel("FAS (m/s² · s)", fontsize=12)
        ax.set_title("Fourier Amplitude Spectra — All Stations (M4.2 Event)", fontsize=11)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "event_fas.png"), dpi=150)
        plt.close(fig)
        print(f"Saved → {os.path.join(OUT_DIR, 'event_fas.png')}")

    # ── PGA vs distance ────────────────────────────────────────────────────────
    if len(gm_df) > 1 and gm_df["dist_km"].notna().sum() > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(gm_df["dist_km"], gm_df["pga_cms2"], s=80, zorder=5, color="navy")
        for _, row in gm_df.iterrows():
            if np.isfinite(row["dist_km"]):
                ax.annotate(row["station"],
                            (row["dist_km"], row["pga_cms2"]),
                            fontsize=8, ha="left", va="bottom")
        ax.set_xlabel("Epicentral Distance (km)", fontsize=11)
        ax.set_ylabel("PGA (cm/s²)", fontsize=11)
        ax.set_title("PGA Attenuation — M4.2 Yangon Event", fontsize=11)
        ax.grid(True, ls=":", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "pga_attenuation.png"), dpi=120)
        plt.close(fig)
        print(f"Saved → {os.path.join(OUT_DIR, 'pga_attenuation.png')}")


if __name__ == "__main__":
    main()
