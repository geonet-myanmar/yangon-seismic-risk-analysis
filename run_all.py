#!/usr/bin/env python3
"""
run_all.py — Orchestrates all 6 analysis modules in dependency order.
Run from the yangon_quake directory:
    python run_all.py
"""

import subprocess, sys, time, os

PYTHON = sys.executable
MODULES = [
    ("Module 1 — HVSR Site Characterization",           "module1_hvsr.py"),
    ("Module 3 — Ground Motion (Event Data)",            "module3_ground_motion.py"),
    ("Module 4 — Site Amplification Cross-Check",        "module4_amplification.py"),
    ("Module 2 — Spatial Interpolation & Maps",          "module2_spatial.py"),
    ("Module 5 — Building Exposure & Vulnerability",     "module5_buildings.py"),
    ("Module 6 — Integrated Risk Mapping",               "module6_risk.py"),
]

def run(label, script):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    t0  = time.time()
    ret = subprocess.run([PYTHON, script], cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - t0
    status  = "✓ OK" if ret.returncode == 0 else f"✗ FAILED (exit {ret.returncode})"
    print(f"\n  [{status}]  {elapsed:.1f}s")
    return ret.returncode == 0

if __name__ == "__main__":
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"), exist_ok=True)
    results = []
    for label, script in MODULES:
        ok = run(label, script)
        results.append((label, ok))

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for label, ok in results:
        print(f"  {'✓' if ok else '✗'}  {label}")
