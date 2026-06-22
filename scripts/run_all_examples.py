#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "examples"


JOBS = [
    ("ex1_lorenz_63", "flow_distill_prune.py"),
    ("ex2_multiscale", "flow_distill_prune.py"),
    ("ex3_pendulum", "flow_distill.py"),
    ("ex4_heat_equation", "flow_distill_prune.py"),
    ("ex5_lunar_lander", "flow_distill_prune.py"),
    ("ex6_reacher", "flow_distill_prune.py"),
    ("ex7_gpr_oil", "flow_distill_prune_all.py"),
]

for name, script in JOBS:
    cwd = ROOT / name
    print(f"\n=== {name} ===", flush=True)
    subprocess.run([sys.executable, script], cwd=cwd, check=True)

print("\nDone.")
