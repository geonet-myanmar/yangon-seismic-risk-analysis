#!/usr/bin/env python3
"""
Module 4 — Site Amplification Cross-Check
Computes Standard Spectral Ratios (SSR) from the earthquake event recordings
and compares them to HVSR amplification values from Module 1.

Method:
  - For each station, compute the Fourier amplitude spectrum of the
    horizontal acceleration (geometric mean of E and N components).
  - Divide by the spectrum of the station with the lowest HVSR A0
    (presumed reference / least-amplified site).
  - Compare SSR peak amplification and frequency with HVSR f0 / A0
    at the nearest microzonation measurement site.

Outputs:
  outputs/ssr_results.csv
  outputs/ssr_plots/                  SSR plots per station
  outputs/amplification_comparison.png
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import detrend as sp_detrend, savgol_filter
from obspy import read, read_inventory, Stream
from obspy.core import UTCDateTime
from obspy.signal.trigger import recursive_sta_lta, trigger_onset

warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
EVENT_DIR = os.path.join(BASE_DIR, "Yangon Event Mseed 4.2 data")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
SSR_DIR   = os.path.join(OUT_DIR, "ssr_plots")
os.makedirs(SSR_DIR, exist_ok=True)

PRE_FILT   = (0.1, 0.5, 45.0, 50.0)
FMIN_SSR   = 0.5    # Hz
FMAX_SSR   = 20.0   # Hz
B_KO       = 40


def ko_smooth(spectrum, freqs, b=40):
    out  = np.zeros_like(spectrum, dtype=float)
    lf   = np.log10(np.where(freqs > 0, freqs, 1e-30))
    for i, fi in enumerate(freqs):
        if fi <= 0:
            continue
        arg = b * (lf - lf[i])
        with np.errstate(invalid="ignore", divide="ignore"):
            w = np.where(np.abs(arg) < 1e-6, 1.0,
                         (np.sin(np.pi * arg) / (np.pi * arg)) ** 4)
        w = np.where(np.isnan(w) | (w < 0), 0.0, w)
        denom = w.sum()
        if denom > 0:
            out[i] = (w * spectrum).sum() / denom
    return out


def detect_p_onset(z_tr):
    sr = z_tr.stats.sampling_rate
    cft = recursive_sta_lta(z_tr.data.astype(float),
                             int(1.0 * sr), int(10.0 * sr))
    triggers = trigger_onset(cft, 4.0, 1.5)
    if not triggers:
        return None
    best = max(triggers, key=lambda t: cft[t[0]])
    return z_tr.stats.starttime + best[0] / sr


def station_fas(mseed_path, xml_path, pre_s=5.0, sig_s=30.0,
                fmin=FMIN_SSR, fmax=FMAX_SSR):
    """
    Load a station, detect event onset, cut signal window, remove response,
    compute geometric-mean horizontal FAS with KO smoothing.
    Returns: (freqs, fas, lat, lon, onset_utc) or None on failure.
    """
    st_raw = read(mseed_path)
    inv    = read_inventory(xml_path)
    st_raw.merge(method=1, fill_value="interpolate")

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
        from obspy import Stream as _Stream
        st = _Stream([tr for tr in st_raw
                      if tr.stats.channel in ("EHZ", "ENE", "ENN")]).copy()
    elif any(c.startswith("EH") for c in chans_avail):
        st = st_raw.select(channel="EH*").copy()
    else:
        st = st_raw.select(channel="EN*").copy()

    z_cands = [tr for tr in st if tr.stats.channel.endswith("Z")]
    if not z_cands:
        return None
    z_tr = z_cands[0]

    # Lower-threshold STA/LTA + peak-amplitude fallback
    onset = None
    for thr in (4.0, 2.5, 1.8):
        sr_z = z_tr.stats.sampling_rate
        cft  = recursive_sta_lta(z_tr.data.astype(float),
                                  int(1.0 * sr_z), int(10.0 * sr_z))
        trigs = trigger_onset(cft, thr, thr * 0.4)
        if len(trigs):
            best  = max(trigs, key=lambda t: cft[t[0]])
            onset = z_tr.stats.starttime + best[0] / sr_z
            break
    if onset is None:
        z_data = z_tr.data.astype(float)
        pk_idx = int(np.argmax(np.abs(z_data)))
        onset  = z_tr.stats.starttime + pk_idx / z_tr.stats.sampling_rate

    st_ev = st.slice(onset - pre_s, onset + sig_s).copy()
    if len(st_ev) == 0:
        return None

    try:
        st_ev.remove_response(inventory=inv, output="ACC",
                               pre_filt=PRE_FILT, water_level=60)
    except Exception:
        pass
    for tr in st_ev:
        tr.data = np.where(np.isfinite(tr.data), tr.data, 0.0)

    sr = st_ev[0].stats.sampling_rate

    # Horizontal channels
    h_trs = [tr for tr in st_ev if not tr.stats.channel.endswith("Z")]
    if not h_trs:
        return None

    # Align lengths
    n = min(tr.stats.npts for tr in h_trs)
    hanning = np.hanning(n)
    specs   = []
    for tr in h_trs:
        seg  = sp_detrend(tr.data[:n].astype(float), type="linear") * hanning
        spec = np.abs(np.fft.rfft(seg))
        specs.append(spec)

    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    # Geometric mean across horizontal components
    geo_mean = np.exp(np.mean(np.log(np.clip(np.array(specs), 1e-30, None)), axis=0))
    # KO smoothing
    mask  = (freqs >= fmin) & (freqs <= fmax)
    f_sel = freqs[mask]
    fas   = ko_smooth(geo_mean[mask], f_sel, b=B_KO)

    # Station coordinates
    lat, lon = np.nan, np.nan
    for net in inv:
        for sta in net:
            for ch in sta.channels:
                if (ch.start_date is None or ch.start_date <= onset) and \
                   (ch.end_date is None or ch.end_date >= onset):
                    lat, lon = sta.latitude, sta.longitude
                    break

    return f_sel, fas, lat, lon, onset


def main():
    import glob

    # ── Load HVSR results ──────────────────────────────────────────────────────
    hvsr_path = os.path.join(OUT_DIR, "hvsr_results.csv")
    if not os.path.exists(hvsr_path):
        raise FileNotFoundError("Run module1_hvsr.py first.")
    hvsr_df = pd.read_csv(hvsr_path)
    hvsr_ok  = hvsr_df.dropna(subset=["f0", "A0"]).copy()

    # ── Discover event stations ────────────────────────────────────────────────
    mseed_files = sorted(glob.glob(os.path.join(EVENT_DIR, "*.mseed")))
    xml_files   = sorted(glob.glob(os.path.join(EVENT_DIR, "*.xml")))

    stations = {}
    for mf in mseed_files:
        st_tmp = read(mf, headonly=True)
        code   = st_tmp[0].stats.station
        stations[code] = {"mseed": mf, "xml": None}
    for xf in xml_files:
        inv_tmp = read_inventory(xf)
        for net in inv_tmp:
            for sta in net:
                if sta.code in stations:
                    stations[sta.code]["xml"] = xf

    # ── Compute FAS per station ────────────────────────────────────────────────
    fas_dict = {}
    meta     = {}
    for code, paths in sorted(stations.items()):
        if paths["xml"] is None:
            print(f"  [{code}] no XML — skip")
            continue
        print(f"  Processing {code} ...", end=" ", flush=True)
        try:
            result = station_fas(paths["mseed"], paths["xml"])
            if result is None:
                print("failed")
                continue
            f_sel, fas, lat, lon, onset = result
            fas_dict[code] = (f_sel, fas)
            meta[code]     = dict(lat=lat, lon=lon, onset=str(onset))
            print(f"ok  (lat={lat:.4f}, lon={lon:.4f})")
        except Exception as e:
            print(f"error: {e}")

    if len(fas_dict) < 2:
        print("Not enough stations for SSR — need at least 2.")
        return

    # ── Choose reference station ───────────────────────────────────────────────
    # Station nearest to an HVSR site with lowest A0 (least amplified)
    ref_code = None
    if len(hvsr_ok) > 0:
        min_A0_site = hvsr_ok.loc[hvsr_ok["A0"].idxmin()]
        best_dist   = np.inf
        for code in fas_dict:
            lat = meta[code]["lat"]
            lon = meta[code]["lon"]
            if np.isnan(lat):
                continue
            d = ((lat - min_A0_site["lat"]) ** 2 +
                 (lon - min_A0_site["lon"]) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                ref_code  = code
    if ref_code is None:
        # fallback: station with lowest FAS peak in 0.5-5 Hz range
        ref_code = min(fas_dict,
                       key=lambda c: np.max(fas_dict[c][1]
                                            [(fas_dict[c][0] > 0.5) & (fas_dict[c][0] < 5)]))
    print(f"\nReference station: {ref_code}")

    ref_f, ref_fas = fas_dict[ref_code]

    # ── Compute SSR for each non-reference station ─────────────────────────────
    ssr_records = []
    fig_cmp, ax_cmp = plt.subplots(figsize=(9, 5))
    cmap  = plt.cm.get_cmap("tab10", len(fas_dict))
    c_idx = 0

    for code in sorted(fas_dict):
        f_sel, fas = fas_dict[code]
        # Interpolate reference onto same frequency axis
        ref_interp = np.interp(f_sel, ref_f, ref_fas, left=np.nan, right=np.nan)
        ref_safe   = np.where(ref_interp > 1e-30, ref_interp, 1e-30)
        ssr        = fas / ref_safe

        # Find SSR peak (0.5–12 Hz)
        peak_mask = (f_sel >= 0.5) & (f_sel <= 12.0)
        if peak_mask.any():
            pk_idx   = np.argmax(ssr[peak_mask])
            ssr_f0   = float(f_sel[peak_mask][pk_idx])
            ssr_A0   = float(ssr[peak_mask][pk_idx])
        else:
            ssr_f0, ssr_A0 = np.nan, np.nan

        # Find nearest HVSR site to this station
        sta_lat = meta[code]["lat"]
        sta_lon = meta[code]["lon"]
        hvsr_f0_near, hvsr_A0_near, nearest_site = np.nan, np.nan, ""
        if len(hvsr_ok) > 0 and np.isfinite(sta_lat):
            dists   = np.sqrt((hvsr_ok["lat"] - sta_lat) ** 2 +
                               (hvsr_ok["lon"] - sta_lon) ** 2)
            nearest = hvsr_ok.loc[dists.idxmin()]
            hvsr_f0_near  = nearest["f0"]
            hvsr_A0_near  = nearest["A0"]
            nearest_site  = nearest["site_id"]

        label = (f"{code}{'(ref)' if code == ref_code else ''}"
                 f"  SSR f₀={ssr_f0:.2f}Hz" if np.isfinite(ssr_f0) else code)
        color  = cmap(c_idx)
        c_idx += 1

        if code != ref_code:
            ax_cmp.semilogx(f_sel, ssr, lw=1.6, color=color, label=label)

        # Individual SSR plot
        fig, ax = plt.subplots(figsize=(7, 4))
        if code == ref_code:
            ax.semilogx(f_sel, np.ones_like(f_sel), color="gray",
                        lw=1.5, ls="--", label="Reference (ratio = 1)")
        else:
            ax.semilogx(f_sel, ssr, "b-", lw=1.8, label="SSR")
        if np.isfinite(ssr_f0):
            ax.axvline(ssr_f0, color="red", ls="--", lw=1.2,
                       label=f"SSR f₀ = {ssr_f0:.2f} Hz  A₀ = {ssr_A0:.2f}")
        if np.isfinite(hvsr_f0_near):
            ax.axvline(hvsr_f0_near, color="green", ls=":", lw=1.2,
                       label=f"HVSR {nearest_site} f₀ = {hvsr_f0_near:.2f} Hz  A₀ = {hvsr_A0_near:.2f}")
        ax.axhline(1.0, color="k", ls=":", lw=0.8)
        ax.set_xlim(0.5, 20)
        ax.set_xlabel("Frequency (Hz)", fontsize=11)
        ax.set_ylabel("SSR", fontsize=11)
        ax.set_title(f"SSR — Station {code}  (ref: {ref_code})", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(SSR_DIR, f"{code}_ssr.png"), dpi=120)
        plt.close(fig)

        ssr_records.append(dict(
            station=code,
            is_reference=(code == ref_code),
            lat=round(float(meta[code]["lat"]), 5) if np.isfinite(meta[code]["lat"]) else np.nan,
            lon=round(float(meta[code]["lon"]), 5) if np.isfinite(meta[code]["lon"]) else np.nan,
            ssr_f0=round(ssr_f0, 4) if np.isfinite(ssr_f0) else np.nan,
            ssr_A0=round(ssr_A0, 4) if np.isfinite(ssr_A0) else np.nan,
            nearest_hvsr_site=nearest_site,
            hvsr_f0=round(float(hvsr_f0_near), 4) if np.isfinite(hvsr_f0_near) else np.nan,
            hvsr_A0=round(float(hvsr_A0_near), 4) if np.isfinite(hvsr_A0_near) else np.nan,
            f0_diff=round(abs(ssr_f0 - hvsr_f0_near), 4) if (
                np.isfinite(ssr_f0) and np.isfinite(hvsr_f0_near)) else np.nan,
        ))

    # Finalize comparison plot
    ax_cmp.axhline(1.0, color="k", ls=":", lw=0.8)
    ax_cmp.set_xlim(0.5, 20)
    ax_cmp.set_xlabel("Frequency (Hz)", fontsize=12)
    ax_cmp.set_ylabel("SSR (relative to reference)", fontsize=12)
    ax_cmp.set_title(f"Standard Spectral Ratios — all stations vs {ref_code}", fontsize=11)
    ax_cmp.legend(fontsize=8, ncol=2)
    ax_cmp.grid(True, which="both", ls=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ssr_comparison.png"), dpi=150)
    plt.close(fig_cmp)

    # ── Save results ───────────────────────────────────────────────────────────
    ssr_df = pd.DataFrame(ssr_records)
    csv_path = os.path.join(OUT_DIR, "ssr_results.csv")
    ssr_df.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}")
    print(f"Saved → {os.path.join(OUT_DIR, 'ssr_comparison.png')}")
    print(f"Saved plots → {SSR_DIR}/")
    print("\n── SSR vs HVSR Comparison ──")
    cols = ["station", "ssr_f0", "ssr_A0", "nearest_hvsr_site", "hvsr_f0", "hvsr_A0", "f0_diff"]
    print(ssr_df[cols].to_string(index=False))

    # ── Scatter: SSR f0 vs HVSR f0 ────────────────────────────────────────────
    ok = ssr_df.dropna(subset=["ssr_f0", "hvsr_f0"])
    if len(ok) >= 2:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(ok["hvsr_f0"], ok["ssr_f0"], s=80, zorder=5, color="navy")
        for _, row in ok.iterrows():
            ax.annotate(row["station"], (row["hvsr_f0"], row["ssr_f0"]),
                        fontsize=8, ha="left", va="bottom")
        lims = [min(ok["hvsr_f0"].min(), ok["ssr_f0"].min()) * 0.8,
                max(ok["hvsr_f0"].max(), ok["ssr_f0"].max()) * 1.2]
        ax.plot(lims, lims, "k--", lw=0.8, label="1:1 line")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("HVSR f₀ (Hz)", fontsize=11)
        ax.set_ylabel("SSR f₀ (Hz)", fontsize=11)
        ax.set_title("SSR vs HVSR Fundamental Frequency", fontsize=11)
        ax.legend()
        ax.grid(True, ls=":", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "ssr_vs_hvsr_f0.png"), dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    main()
