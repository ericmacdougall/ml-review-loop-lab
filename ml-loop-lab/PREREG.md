# ML review-loop preregistration

Frozen before any held-out classification score, 2026-11-10 UTC. GoodQA #15642.
Recovered implementation base: 66c8625; final code is bound by frozen_inputs.json. C=100 logistic regression, balanced class weights,
frozen CLIP ViT-B-32-quickgelu OpenAI weights. The head search used only the
1,000-item tune split and five development seeds: C100 AP 0.895796 (seed SD
0.009298). All model settings and five strategies are fixed now.

## Primary question and decision

At week 12, does hybrid selection improve clean held-out average precision over
random selection under the same 150-reviewed-items-per-week policy and the
same simulated adjudication policy? Report the mean paired difference and a
two-sided 95% interval. Improvement is supported only when its lower bound
exceeds zero; otherwise report the comparison as inconclusive or worse. No
method is required to win for the experiment to be valid. The comparison is
fixed before results and will not be replaced by a post-hoc best strategy.

Resample the same 3,000 held-out items and the same 20 final seeds jointly,
1,000 bootstrap replicates, RNG seed 20261110. Seeds are 100 through 119.
The 20 seeds do not create 60,000 independent items. Strategies use matched
item-indexed simulator draws within a seed; selection never receives true
labels, oracle hardness, or held-out outcomes. AP, ROC AUC, accuracy and balanced
accuracy curves have seed-level t intervals; the final AP difference uses the
item-and-seed bootstrap. Report class prevalence and this conditional scope.

## Data and split freeze

CIFAKE mirror dragonintelligence/CIFAKE-image-dataset, revision
6eebd6090fe3c0d9d44f899a25f0a34e3c2dad31. The upstream authors state MIT;
the mirror has no separate license tag. Do not distribute the image corpus.
The complete release provenance and citations must accompany the walkthrough.

20% positive prevalence. Training source: 18,000 arrival-pool images, 200 gold
seed images, 2,000 calibration images, and 1,000 tuning images. Dataset test
source: 3,000 held-out images and a disjoint 400-item peek split. The peek set
is used only for the explicitly contaminated test-tuning demonstration, never
for the main score. Two calibration duplicates (indices 728,1645), one shared
with pool and one with held-out, are excluded from calibration. The pool has
five within-pool duplicates; no pool/gold/tune exact-pixel matches with test
were found. Report the remaining near-duplicate/pretraining-overlap limitation.

SHA256 at freeze:

- splits.npz: 1c55bc6e2781064d0f544937ed3790b3de58e5c44b2a3771286e278f8d779e77
- emb.npz: 84addc7efc5422198110de8fa0c10bdce1d452190213d89f90c1d877be08e671
- drop.json: 8b1ed7df4273e49402544c0a4f810201768ed8e6e7ba51cf5fa5502a66155f09
- hardness.npy: 80887b7e59fb55816f0827b083d54f436382a5ebeb57841ca7c588455cce6966
- drift_monitor.json: e6894c6bb08640cd8715caf3923a508723ad0972d96523f50422e39ffc771df8

## Secondary experiments and honest limits

Twelve weeks, exactly 1,500 disjoint arrivals/week, 150 selected items/week
without re-review. Main strategies: random, uncertainty, highest positive score,
uncertainty plus k-center diversity, and half uncertainty/half random hybrid.
All use the same noisy simulated reviewers (three junior error baselines
4%,7%,10%, plus difficulty noise; senior 3% plus difficulty noise). Real images
and real model fits are used; no real people label items in this experiment.

Single-label, two-label-plus-adjudication and three-label policies share an
ITEM budget, not a vote budget. Report actual votes and quality versus votes;
do not infer equal-human-cost superiority from equal-items results. Noise
multiplier 0 and 2 are fixed sensitivity runs. Frozen-at-week-6 and last-four-
weeks retraining are fixed drift-response ablations.

Secondary drift stress test begins week 7: JPEG quality30, Gaussian blur0.7,
brightness0.9. The first 1,000 held-out originals are paired with transformed
copies; these are not an independent sample. Fixed arrival drift-monitor AUC
uses only pool embeddings and no class labels. Report all comparisons, without
post-hoc promotion to primary. Calibration targets are empirical precision95%
and recall90%, evaluated on the calibration subset only; report held-out
precision/recall with Wilson95% intervals and allow unmet targets or abstention.
These transformations and simulated reviewer rates do not establish production
performance, fraud readiness, or generalization to modern image generators.

## Execution and controls

Compute: shared t2-dev, AMD EPYC7502 virtual host, 48 logical CPUs,
Python3.12.3. BLAS/OMP/MKL threads1; six worker processes. Record hardware,
environment freeze, wall time and load. Prereg hash is checked before final.
Focused controls must pass: tied-threshold feasibility, item-paired noise,
non-reused/exhaustive arrivals. Any change after opening held-out results must
be disclosed; the changed comparison becomes exploratory without a fresh test.

Python backend measurements are separate and use ten repeats on the remote
host: concurrency against a local mock HTTP service, GIL CPU workloads,
generator memory, stream first-byte delay, and actual static typing mutants.
They are measured engineering microbenchmarks, not real-model serving claims.
