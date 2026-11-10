"""Precompute simulator-only hardness and unsupervised arrival drift, never test metrics."""
import hashlib
import json
from pathlib import Path
import numpy as np
import loop
from common import DATA_DIR, WEEKLY_ARRIVALS, WEEKS

if __name__ == "__main__":
    root = Path(DATA_DIR)
    drop = json.loads((root / "drop.json").read_text())
    for key in ("pool_vs_test", "gold_vs_test", "tune_vs_test"):
        if drop[key]:
            raise SystemExit(f"REFUSED: duplicate leakage in {key}: {len(drop[key])}")
    state = loop.load()
    loop.oracle_hardness(state)
    ref = state["Xpool"][:3*WEEKLY_ARRIVALS]
    monitor = {}
    for week in range(4, WEEKS+1):
        arr = state["Xpool"][(week-1)*WEEKLY_ARRIVALS:week*WEEKLY_ARRIVALS]
        monitor[str(week)] = loop.drift_monitor(ref, arr, np.random.default_rng(10000+week))
    (root / "drift_monitor.json").write_text(json.dumps(monitor, indent=2))
    # Hash data without printing labels or held-out outcomes.
    manifest = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                for name in ("splits.npz", "emb.npz", "drop.json", "hardness.npy", "drift_monitor.json")}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Prepared simulator and drift caches; no held-out scores calculated.")
