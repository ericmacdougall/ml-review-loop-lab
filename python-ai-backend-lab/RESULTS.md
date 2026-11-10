# Python boundaries, with measured work

The scenario is a backend which fans out model-like HTTP calls, preprocesses
text, streams logs, and enforces typed request contracts. All measurements ran
on a remote shared48-vCPU AMD EPYC7502 virtual host, Python3.12.3, conventional
GIL-enabled build. Native BLAS/OMP/MKL worker counts were pinned to1 for the CPU
comparison. The network service is a **local mock**, not actual model inference.

## Execution model: fixed work instead of slogans

Each GIL cell performs200 units of the same work. Pools are warmed before the
parallel timing; startup and imports are not consistently part of that timed
interval. There are ten repeats for each kind/mode/worker combination,440 timing
records total. A t interval across repeats describes variation on this host.

| Work | Serial mean seconds |8 threads |8 processes |
|---|---:|---:|---:|
| Python text preprocessing |0.89549 |0.93044 |0.12012 |
|300×300 NumPy matrix multiplication |0.32369 |0.03912 |0.04613 |
| SHA256 over8MiB buffers |1.05377 |0.13754 |0.13759 |
|20ms waiting per unit |4.02183 |0.50698 |0.50406 |

For Python preprocessing the95% intervals of the mean are serial
[0.88233,0.90864], threads[0.91832,0.94256], processes[0.11612,0.12413]. Threads
did not improve this bytecode-heavy workload. For NumPy eight-thread time the
interval is[0.03739,0.04085]. Native kernels and large-buffer hashing have a
different GIL boundary. These are workload-specific microbenchmarks, not a
universal recommendation to replace thread pools or a benchmark of free-threaded
Python. The measured build explicitly reports `gil_enabled=true`.

## The actual network experiment

The mock server has lognormal service time, median50ms, sigma0.5, seed1, and
capacity256. Extra admitted work queues behind that service capacity. Each cell
completes1,000 calls, repeated10 times. Async and thread clients use concurrency
4,16,64,256,1024; process clients use4,16,64. All130 measured runs have1,000 successful
responses:130,000 observed completions. A503-response control must make every
client implementation fail, so errors cannot produce a flattering throughput.

| Async concurrency | Mean completions/second |95% interval of mean |
|---|---:|---:|
|4 |69.02 | See full summary.json |
|64 |958.26 | See full summary.json |
|256 |2084.27 |1948.67–2219.88 |
|1024 |1313.87 |1085.81–1541.92 |

The1024-client condition is slower than256 under this service definition. Its
mean per-run p99 service time is473.97ms versus182.46ms at256. This is queueing
under oversubscription, not evidence that async has a fixed scaling limit.
At64 concurrency thread mean throughput967.35/s and async958.26/s are close.
Processes are not automatically useful for waiting-heavy work.

The reported call percentiles begin **after client concurrency admission**.
They exclude waiting to acquire the client's semaphore or waiting for a pool
worker; end-user latency can be larger. The fields are means of per-run
percentiles, not a pooled percentile of every request. The fan-out `rss_mb` field
is cumulative parent-process high-water RSS: do not use it to compare total
thread/process memory. Record child memory or run each condition separately for
that separate question. Shared-host load is retained per observation.

## A million records and first output

A generated108.67MB JSONL file contains1,000,000 request-log rows across500
tenants. Every mode computes the same exact token sum, with an assertion against
the mathematically expected checksum. Each memory condition runs in a fresh
subprocess, making peak RSS interpretable within this Linux experiment.

| Mode | Mean peak RSS MiB |95% mean interval | Mean seconds |
|---|---:|---:|---:|
| Eager parsed list |654.00 |654.00–654.00 |2.96253 |
| Record generator |33.56 |30.59–36.53 |2.71777 |
| Batches of1000 |33.56 | See full summary.json |2.76666 |

The exactly repeated eager-list RSS produces a zero-width within-run interval;
it does not imply a universally exact memory requirement. Generators reduce
retained record objects, but the aggregation map still grows with distinct
tenant count. An unbounded consumer queue or materializing a list downstream
can erase the benefit. The generator is not a guarantee of constant memory for
every surrounding algorithm.

The mock stream yields40 chunks20ms apart. Handling the first chunk immediately
takes21.99ms on average,95% mean interval[21.56,22.42]. Buffering all chunks
before handling takes828.60ms,[825.03,832.17]. This improves time to first
application-visible output, not inference computation. It is a controlled mock
stream; production proxies, framing, client buffering and cancellation need
separate integration tests.

## Type checking and behavioral controls

The typing lab first requires the clean baseline to pass pytest, mypy strict,
and pyright strict. It then injects20 individual defects:14 type-shaped and6
logic-shaped. The original observed run caught all14 type-shaped defects and
none of the6 logic-shaped defects with each static checker. A fresh run on the
recovered code is recorded as typing-current.json and its log; inspect that
artifact for the final observed detector matrix, rather than treating a comment
as evidence.

A missing await, an invalid TypedDict key, optional-value misuse or wrong return
type is a plausible static-analysis target. A wrong currency denominator,
omitted budget term, swapped role filter or disabled authorization branch can
remain well typed. Unit tests and mutation controls cover a different boundary.
The small mutation corpus is deliberately constructed; it estimates no real
world detection rate.

Two benchmark correctness controls pass: every HTTP client rejects a controlled
503 instead of counting it as completed useful work; changing the streaming
aggregation to add one token makes the benchmark refuse the result. These
controls exercise failure paths, not implementation-shaped equality assertions.

## Implementation and rerun

`bench/mock_server.py` owns localhost18080. `bench_fanout.py` measures actual
HTTP completions with admission control and shared clients. `bench_gil.py`
isolates bytecode/native/waiting work. `bench_mem.py` uses fresh subprocesses and
retains its generated input beside the run. `typing_lab/run_typing.py` makes
temporary mutant worktrees under its own durable data directory.

With the repository-root virtual environment prepared, run on a Linux compute
host:

```bash
../.venv/bin/python bench/mock_server.py --port 18080 --median-ms 50 --sigma 0.5 --capacity 256 --seed 1
# In another shell, from python-ai-backend-lab:
../.venv/bin/python bench/bench_fanout.py --out results/fanout.json --repeats 10 --calls 1000
../.venv/bin/python bench/bench_gil.py --out results/gil.json --repeats 10
../.venv/bin/python bench/bench_mem.py --out results/mem.json --lines 1000000 --repeats 10
../.venv/bin/python typing_lab/run_typing.py --out results/typing-current.json
../.venv/bin/python -m unittest test_bench_controls -v
```

Stop only the mock service you started after the experiment. Keep prior results
when changing hardware or workload; write a new output directory. At repository
root, `python3 tools/report_benchmarks.py` verifies complete ten-repeat matrices
and writes the strict JSON summary with scope labels for every interval.

Official API references checked2026-11-10 UTC:
[Python3.12 asyncio tasks](https://docs.python.org/3.12/library/asyncio-task.html),
[Python3.12 executors](https://docs.python.org/3.12/library/concurrent.futures.html).
Newer documentation may describe newer interpreter capabilities; the experiment
reports the build it actually used.
