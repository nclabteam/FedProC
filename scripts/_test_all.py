import subprocess
import sys

from strategies import STRATEGIES

skip = {"base", "dFL", "hFL", "nFL", "pFL", "tFL", "MOTAR"}
strategies = sorted(s for s in STRATEGIES if s not in skip)

results = []
for s in strategies:
    cmd = [
        sys.executable,
        "main.py",
        "--dataset",
        "ETDatasetHour",
        "--strategy",
        s,
        "--model",
        "DLinear",
        "--iterations",
        "2",
        "--times",
        "1",
        "--seed",
        "42",
    ]
    if s.startswith("DFed"):
        cmd += ["--topology", "Ring"]
    print(f"Testing {s}...", flush=True)
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=300)
        status = "PASS" if r.returncode == 0 else "FAIL"
    except subprocess.TimeoutExpired:
        status = "TIMEOUT"
    if status != "PASS":
        err = r.stderr.decode()[-300:] if status == "FAIL" else ""
        print(f"  {s}: {status} | {err.strip()}", flush=True)
    else:
        print(f"  {s}: {status}", flush=True)
    results.append((s, status))

print("\n=== SUMMARY ===")
passed = [s for s, st in results if st == "PASS"]
failed = [s for s, st in results if st != "PASS"]
print(f"PASS: {len(passed)}/{len(results)}")
for s, st in results:
    if st != "PASS":
        print(f"  {s}: {st}")
