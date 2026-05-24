#!/usr/bin/env python3
"""
Module 1 — HVSR Site Characterization
Computes Horizontal-to-Vertical Spectral Ratio at the 25 microzonation sites.
Outputs: outputs/hvsr_results.csv  and  outputs/hvsr_plots/*.png
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import detrend as sp_detrend
from obspy import read, read_inventory, Stream
from obspy.core import UTCDateTime

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MICRO_DIR = os.path.join(BASE_DIR, "Yangon seismic Microzonation Study")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
PLOT_DIR  = os.path.join(OUT_DIR, "hvsr_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
WIN_LEN      = 30.0     # seconds
OVERLAP      = 0.0      # fraction (0 = no overlap)
MAX_WINDOWS  = 150      # cap to keep runtime reasonable
B_KO         = 40       # Konno-Ohmachi bandwidth
FMIN         = 0.3      # Hz  (output frequency range)
FMAX         = 15.0     # Hz
F0_SEARCH_LO = 0.5      # Hz  (f0 search window)
F0_SEARCH_HI = 12.0     # Hz
TRANSIENT    = 4.0      # reject window if RMS > TRANSIENT × median RMS


# ── Konno-Ohmachi smoothing ────────────────────────────────────────────────────
def ko_smooth(spectrum: np.ndarray, freqs: np.ndarray, b: float = 40) -> np.ndarray:
    """Vectorised Konno-Ohmachi smoothing (log-domain bandwidth b)."""
    out = np.zeros_like(spectrum, dtype=float)
    lf  = np.log10(np.where(freqs > 0, freqs, 1e-30))
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


# ── Per-site HVSR ──────────────────────────────────────────────────────────────
def compute_hvsr(mseed_path, xml_path, lat, lon, site_id,
                 start_utc=None,
                 win_len=WIN_LEN, b=B_KO, fmin=FMIN, fmax=FMAX,
                 max_windows=MAX_WINDOWS, transient=TRANSIENT):
    """
    Load waveform, remove response, compute HVSR.
    Returns dict with keys: freqs, hvsr_mean, hvsr_std, f0, A0, T0, n_windows,
                             status, note
    """
    result = dict(site_id=site_id, lat=lat, lon=lon,
                  f0=np.nan, A0=np.nan, T0=np.nan,
                  n_windows=0, status="ok", note="")

    # ── Load waveform ──────────────────────────────────────────────────────────
    try:
        st_raw = read(mseed_path)
    except Exception as e:
        result.update(status="error", note=f"read error: {e}")
        return result, None, None, None

    # Check components
    chans = set(tr.stats.channel for tr in st_raw)
    has_z  = any(c.endswith("Z") for c in chans)
    has_e  = any(c.endswith("E") or c.endswith("1") for c in chans)
    has_n  = any(c.endswith("N") or c.endswith("2") for c in chans)
    if not (has_z and has_e and has_n):
        missing = [c for c, flag in [("Z", has_z), ("E/1", has_e), ("N/2", has_n)] if not flag]
        result.update(status="incomplete", note=f"missing components: {missing}")
        return result, None, None, None

    # ── Load inventory ─────────────────────────────────────────────────────────
    try:
        inv = read_inventory(xml_path)
    except Exception as e:
        result.update(status="error", note=f"XML error: {e}")
        return result, None, None, None

    # ── Select data window (time-based → same-day → full recording fallback) ────
    def _dur(stream):
        if len(stream) == 0:
            return 0.0
        return max(tr.stats.npts / tr.stats.sampling_rate for tr in stream)

    st = st_raw.copy()
    st.merge(method=1, fill_value="interpolate")

    if start_utc is not None:
        t0 = UTCDateTime(start_utc)
        st_try = st.slice(t0, t0 + 7200)        # 2-h window at stated UTC time
        if _dur(st_try) >= win_len * 5:
            st = st_try
        else:
            # Stated time might be local (UTC+6:30); try subtracting 6h30
            t0_local = t0 - 6 * 3600 - 30 * 60
            st_try2  = st.slice(t0_local, t0_local + 7200)
            if _dur(st_try2) >= win_len * 5:
                st = st_try2
            else:
                # Full-recording fallback: take 2 h starting 30 min in
                # (skip first 30 min to avoid instrument stabilisation transients)
                rec_start = min(tr.stats.starttime for tr in st)
                st_try3   = st.slice(rec_start + 1800, rec_start + 1800 + 7200)
                if _dur(st_try3) >= win_len * 5:
                    st = st_try3
                # else: keep full merged stream and let the window-builder deal with it

    if _dur(st) < win_len * 5:
        result.update(status="insufficient_data", note="< 5 windows of data after all fallbacks")
        return result, None, None, None

    # ── Remove instrument response ─────────────────────────────────────────────
    try:
        st.remove_response(inventory=inv, output="VEL",
                           pre_filt=(0.1, 0.2, 45.0, 50.0),
                           water_level=60)
    except Exception as e:
        result.update(note=f"response removal warning: {e}")
        # continue with raw data (H/V response still largely cancels)

    sr = st[0].stats.sampling_rate
    win_samp = int(win_len * sr)

    # ── Extract E, N, Z arrays ─────────────────────────────────────────────────
    def get_comp(suffix_list):
        traces = [tr for tr in st if any(tr.stats.channel.endswith(s) for s in suffix_list)]
        if not traces:
            return None
        return Stream(traces).merge(method=1, fill_value="interpolate")[0].data.astype(float)

    z  = get_comp(["Z"])
    h1 = get_comp(["E", "1"])
    h2 = get_comp(["N", "2"])
    if z is None or h1 is None or h2 is None:
        result.update(status="incomplete", note="component extraction failed")
        return result, None, None, None

    n_samp = min(len(z), len(h1), len(h2))
    z, h1, h2 = z[:n_samp], h1[:n_samp], h2[:n_samp]

    # ── Build output frequency axis (log-spaced) ───────────────────────────────
    n_fft   = win_samp
    raw_f   = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    mask    = (raw_f >= fmin) & (raw_f <= fmax)
    out_f   = raw_f[mask]
    if len(out_f) == 0:
        result.update(status="error", note="frequency axis empty")
        return result, None, None, None

    # ── Build windows (evenly spaced, capped at max_windows) ──────────────────
    step_samp   = int(win_samp * (1 - OVERLAP))
    total_wins  = max(0, (n_samp - win_samp) // step_samp + 1)
    indices     = np.linspace(0, max(0, total_wins - 1), min(total_wins, max_windows),
                               dtype=int)

    # Transient rejection: pre-compute RMS of Z windows
    rms_vals = []
    for i in indices:
        s = int(i) * step_samp
        rms_vals.append(np.sqrt(np.mean(z[s:s + win_samp] ** 2)))
    med_rms = np.median(rms_vals) if rms_vals else 1.0
    good    = np.array(rms_vals) < transient * med_rms

    # ── Accumulate spectra ─────────────────────────────────────────────────────
    hanning = np.hanning(win_samp)
    hv_list = []

    for keep, i in zip(good, indices):
        if not keep:
            continue
        s = int(i) * step_samp
        e = s + win_samp
        if e > n_samp:
            break

        def window_spec(arr):
            seg = sp_detrend(arr[s:e].copy(), type="linear") * hanning
            amp = np.abs(np.fft.rfft(seg))[mask]
            return ko_smooth(amp, out_f, b=b)

        az  = window_spec(z)
        ah1 = window_spec(h1)
        ah2 = window_spec(h2)

        az_safe = np.where(az > 1e-30, az, 1e-30)
        hv = np.sqrt(ah1 ** 2 + ah2 ** 2) / az_safe
        if np.all(np.isfinite(hv)):
            hv_list.append(hv)

    if len(hv_list) < 5:
        result.update(status="insufficient_windows",
                      note=f"only {len(hv_list)} good windows")
        return result, out_f, None, None

    hv_arr     = np.array(hv_list)
    log_hv     = np.log(np.where(hv_arr > 0, hv_arr, 1e-30))
    hvsr_mean  = np.exp(np.mean(log_hv, axis=0))
    hvsr_std   = np.exp(np.std(log_hv, axis=0))

    # ── Find f0 (peak in search window) ───────────────────────────────────────
    search = (out_f >= F0_SEARCH_LO) & (out_f <= F0_SEARCH_HI)
    if search.any():
        peak_idx = np.argmax(hvsr_mean[search])
        f0       = out_f[search][peak_idx]
        A0       = hvsr_mean[search][peak_idx]
    else:
        f0, A0 = np.nan, np.nan

    T0 = 1.0 / f0 if f0 > 0 else np.nan
    result.update(f0=round(float(f0), 4), A0=round(float(A0), 4),
                  T0=round(float(T0), 4) if np.isfinite(T0) else np.nan,
                  n_windows=len(hv_list))
    return result, out_f, hvsr_mean, hvsr_std


# ── Plot ───────────────────────────────────────────────────────────────────────
def plot_hvsr(freqs, hvsr_mean, hvsr_std, f0, A0, site_id, lat, lon, save_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(freqs, hvsr_mean, "b-", lw=1.8, label="H/V (geometric mean)")
    ax.fill_between(freqs,
                    hvsr_mean / hvsr_std,
                    hvsr_mean * hvsr_std,
                    alpha=0.25, color="blue", label="±1σ")
    if np.isfinite(f0):
        ax.axvline(f0, color="red", ls="--", lw=1.4, label=f"f₀ = {f0:.2f} Hz  A₀ = {A0:.2f}")
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_xlim(0.3, 15)
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("H/V Ratio", fontsize=11)
    ax.set_title(f"{site_id}  (lat {lat:.4f}, lon {lon:.4f})", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    import openpyxl
    wb  = openpyxl.load_workbook(os.path.join(MICRO_DIR, "YGN.xlsx"))
    ws  = wb["Sheet1"]
    rows = list(ws.iter_rows(min_row=3, values_only=True))   # skip header rows

    records = []
    for row in rows:
        sp_id, date, start_t, lat, lon, sta_name, xml_file, mseed_file, _ = row
        if sp_id is None:
            continue
        mseed_path = os.path.join(MICRO_DIR, mseed_file)
        xml_path   = os.path.join(MICRO_DIR, xml_file)
        if not os.path.exists(mseed_path):
            print(f"  [{sp_id}] MiniSEED not found: {mseed_file} — skipping")
            records.append(dict(site_id=sp_id, lat=lat, lon=lon,
                                f0=np.nan, A0=np.nan, T0=np.nan,
                                n_windows=0, status="file_missing", note=mseed_file))
            continue
        if not os.path.exists(xml_path):
            print(f"  [{sp_id}] XML not found: {xml_file} — skipping")
            records.append(dict(site_id=sp_id, lat=lat, lon=lon,
                                f0=np.nan, A0=np.nan, T0=np.nan,
                                n_windows=0, status="file_missing", note=xml_file))
            continue

        # Build UTC start time
        start_utc = None
        if date is not None and start_t is not None:
            from datetime import datetime, timedelta, timezone
            import datetime as dt_mod
            if isinstance(date, dt_mod.datetime):
                d = date.date()
            else:
                d = date
            if isinstance(start_t, dt_mod.time):
                naive_dt = datetime.combine(d, start_t)
                start_utc = naive_dt  # times in YGN.xlsx are treated as UTC
            else:
                start_utc = datetime(d.year, d.month, d.day)
        elif date is not None:
            from datetime import datetime
            import datetime as dt_mod
            if isinstance(date, dt_mod.datetime):
                d = date.date()
            else:
                d = date
            start_utc = datetime(d.year, d.month, d.day)

        print(f"Processing {sp_id} ({mseed_file}) ...", end=" ", flush=True)
        res, freqs, hv_mean, hv_std = compute_hvsr(
            mseed_path, xml_path, lat, lon, sp_id, start_utc=start_utc)

        status_str = res.get("status", "ok")
        print(f"status={status_str}  f0={res['f0']}  A0={res['A0']}  "
              f"n_win={res['n_windows']}  {res.get('note','')}")

        if freqs is not None and hv_mean is not None and hv_std is not None:
            plot_path = os.path.join(PLOT_DIR, f"{sp_id}_hvsr.png")
            plot_hvsr(freqs, hv_mean, hv_std,
                      res["f0"], res["A0"],
                      sp_id, lat, lon, plot_path)

        records.append(dict(
            site_id=sp_id, lat=lat, lon=lon,
            f0=res["f0"], A0=res["A0"], T0=res["T0"],
            n_windows=res["n_windows"],
            status=status_str,
            note=res.get("note", ""),
            mseed_file=mseed_file,
            xml_file=xml_file,
        ))

    df = pd.DataFrame(records)
    out_csv = os.path.join(OUT_DIR, "hvsr_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")
    print(df[["site_id", "lat", "lon", "f0", "A0", "T0", "n_windows", "status"]].to_string())
    print(f"\nPlots → {PLOT_DIR}")


if __name__ == "__main__":
    main()
