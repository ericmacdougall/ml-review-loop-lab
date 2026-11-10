"""Summarize actual fixed-work benchmark repeats; no serving-capacity inference."""
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "python-ai-backend-lab/results"


def summary(values):
    n = len(values)
    if n != 10 or not all(math.isfinite(x) for x in values):
        raise ValueError("Expected ten finite measured repeats")
    mean = statistics.mean(values)
    half = 2.2621571627409915 * statistics.stdev(values) / math.sqrt(n)
    return {"n": n, "mean": mean, "ci95_mean": [mean-half, mean+half],
            "interval_scope": "variation across ten repeats on one shared host; not fleet or production uncertainty"}


def main():
    output = {"scope": "microbenchmarks; mock HTTP inference, actual Python work and memory",
              "hardware": "shared AMD EPYC7502 virtual host,48vCPU,Python3.12.3,GIL enabled",
              "latency_scope": "client service time after concurrency admission; queue wait before admission excluded",
              "rss_scope": "fanout parent-process cumulative high-water mark; memory workloads use fresh subprocesses"}
    gil = json.loads((RESULTS/"gil.json").read_text())
    fan = json.loads((RESULTS/"fanout.json").read_text())
    mem = json.loads((RESULTS/"mem.json").read_text())
    expected = [(gil, 440), (fan, 130), (mem, 50)]
    for data, n in expected:
        if len(data["runs"]) != n:
            raise ValueError(f"Incomplete benchmark matrix: expected {n}, got {len(data['runs'])}")
    groups = defaultdict(list)
    for row in gil["runs"]:
        groups[(row["kind"],row["mode"],row["workers"])].append(row["wall"])
    output["gil_wall_seconds"] = {"|".join(map(str,k)):summary(v) for k,v in groups.items()}
    groups = defaultdict(list)
    for row in fan["runs"]:
        if row["n_ok"] != fan["meta"]["calls"]:
            raise ValueError("Incomplete HTTP work cannot count as throughput")
        for metric in ("thr","p50","p99","wall","cpu"):
            groups[(row["approach"],row["conc"],metric)].append(row[metric])
    output["fanout"] = {"|".join(map(str,k)):summary(v) for k,v in groups.items()}
    groups = defaultdict(list)
    for row in mem["runs"]:
        for metric in ("secs","rss_mb","ttft_ms"):
            if metric in row:
                groups[(row["mode"],metric)].append(row[metric])
    output["memory_streaming"] = {"|".join(k):summary(v) for k,v in groups.items()}
    output["workloads"] = {"gil_units_per_cell":200,"gil_cells":44,"gil_repeats":10,
                           "http_calls_per_cell":1000,"http_cells":13,"http_repeats":10,
                           "jsonl_records_per_memory_cell":1000000,"memory_repeats":10}
    (RESULTS/"summary.json").write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print("Verified10-repeat matrices; wrote benchmark summary.")


if __name__ == "__main__":
    main()
