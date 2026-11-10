"""Fail closed when frozen experiment inputs change before held-out scoring."""
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_freeze(data_dir, c_head, root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    frozen = json.loads((root / "frozen_inputs.json").read_text())
    if c_head != frozen["settings"]["C_HEAD"]:
        raise SystemExit("REFUSED: classification-head setting differs from preregistration")
    for folder, entries in ((Path(data_dir), frozen["data"]), (root, frozen["implementation"])):
        for name, expected in entries.items():
            if sha(folder/name) != expected:
                raise SystemExit(f"REFUSED: frozen input differs: {name}")


def validate_runs(runs):
    """Require the whole preregistered matrix, with paired and complete seeds."""
    from common import FINAL_SEEDS, WEEKS, N_HELD, N_DRIFT_HELD, N_CALIB
    from loop import configs
    def key(cfg):
        return json.dumps(cfg, sort_keys=True)
    expected = {key(c) for c in configs("all", FINAL_SEEDS, True)}
    found = [key(r["cfg"]) for r in runs]
    if len(found) != len(set(found)) or set(found) != expected:
        raise ValueError("Incomplete, extra, or duplicate experiment configuration/seed matrix")
    for r in runs:
        if [row["week"] for row in r["rows"]] != list(range(WEEKS+1)):
            raise ValueError("Incomplete weekly trace")
        last = r["rows"][-1]
        for name, n in (("scores_clean", N_HELD), ("scores_drift", N_DRIFT_HELD),
                        ("scores_calib", N_CALIB), ("scores_calib_drift", 1000)):
            if len(last[name]) != n:
                raise ValueError(f"Truncated predictions: {name}")
