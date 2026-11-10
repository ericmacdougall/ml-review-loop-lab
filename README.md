# Review loops and Python execution, measured

Two connected engineering labs, with code, full results, run logs, controls and
teaching material. The ML experiment fits real classifiers on a real public
image benchmark; human-review behavior and upload drift are declared simulations.
The Python lab measures actual CPU/memory work and a local mock HTTP service.
Neither is a claim about a client's production performance.

- [Full ML experiment and decisions](ml-loop-lab/RESULTS.md)
- [Preregistration](ml-loop-lab/PREREG.md) and [bound inputs](ml-loop-lab/frozen_inputs.json)
- [Python measurements and limits](python-ai-backend-lab/RESULTS.md)
- `ml-loop-lab/src/`: selection, simulated review, refitting, calibration,
  confidence intervals, drift and deliberately flawed test selection.
- `python-ai-backend-lab/`: real concurrency, GIL, memory, streaming and typing
  mutation experiments; correctness controls refuse fast-but-wrong work.

## Setup on a Linux compute host

The measured environment is Python3.12.3 on a shared AMD EPYC7502 virtual host
with48vCPU. Heavy runs belong on the fleet, not on the Mac.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock
```

## One-command ML rerun

```bash
bash run_ml.sh
```

This downloads the pinned public dataset if missing, generates frozen CLIP
features if missing, checks split hygiene, verifies the frozen source/data and
settings, runs the complete300-configuration matrix, and recomputes the actual
intervals/plots and threshold results. Feature generation is the expensive step;
no paid model API is used. Existing verified features can be reused with
`LAB_DATA=/absolute/path/to/frozen/data bash run_ml.sh`.

The original full input hashes are enforced. A different numerical platform,
package build or upstream artifact may produce different feature bytes: the
runner **refuses** that mismatch rather than silently changing the original
confirmatory experiment. Reproduction on matching frozen artifacts is supported;
cross-platform bit-identical preprocessing is not promised. Each invocation
writes a separate `results/reproduction-*` directory and never overwrites the
original result. A repeated run on the same held-out images is reproducibility
evidence, not an independent new test sample.

## Focused controls

```bash
cd ml-loop-lab
../.venv/bin/python -m unittest test_method -v
cd ../python-ai-backend-lab
../.venv/bin/python -m unittest test_bench_controls -v
```

The original ML controls failed on the recovered implementation before repair.
They now enforce realizable tied thresholds, item-paired reviewer noise,
disjoint arrivals, immutable inputs and complete seed matrices. HTTP error and
wrong-aggregation controls protect benchmark denominators.

## Results and provenance

The original final ML run used source d50ac31 and the preregistered hybrid-versus-
random comparison. It has3,000 held-out images,600 positive examples and20seeds.
Primary mean AP difference:+0.019652,95% paired item-and-seed bootstrap interval
[+0.014731,+0.025470]. Secondary arms and the inconclusive highest-score result
are reported, not substituted for the primary.

The Python matrix has44 GIL conditions/440 timing records,13 HTTP fan-out
conditions/130 records and5 memory/streaming conditions/50 records, with ten
repeats of every condition. A total of
130,000 successful mock-service requests were measured. The exported summary
states exactly what each interval, latency and memory field means.

Machine-readable files retain original raw outputs; publication-safe JSON uses
null for undefined/non-finite values and includes hashes/provenance. Empty
auto-reject sets have undefined precision, not evidence of perfect safety.
No source image corpus, model weights, credentials or client information is
committed here. Dataset and model provenance/terms are linked in the lab report.

## License

Lab code and original documentation: MIT, see LICENSE. Upstream models and
datasets retain their own terms; this license does not relicense third-party
images, weights, packages or quoted documentation.
