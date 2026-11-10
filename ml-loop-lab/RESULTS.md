# A measured review-and-retrain loop

This is a real model experiment on real benchmark image files, with **simulated
reviewers and an artificial upload-pipeline shift**. It is not a production fraud
deployment or evidence about any person's past work. The scenario is a platform
which needs to review image submissions, learn from labels, and decide whether
to replace its current classifier.

## What ran

The frozen OpenAI CLIP ViT-B/32 quickgelu image encoder produces512-dimensional
vectors. A balanced logistic-regression head is refit after each review batch.
Only the head learns; this is not backbone fine-tuning, LoRA, or a comparison
against a modern fraud classifier. We chose this design to make thousands of
actual model fits repeatable on a shared CPU host without a paid inference API.

The CIFAKE mirror is pinned at revision6eebd6090fe3c0d9d44f899a25f0a34e3c2dad31.
Its upstream authors publish a120,000-image real/synthetic benchmark and state
the MIT license. The mirror has no separate license tag. We retain provenance
and do not redistribute the source images. Sources:
[upstream dataset](https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images),
[paper](https://arxiv.org/abs/2303.14126). The images are32×32 and the synthetic
class comes from a particular generator; this is a constrained benchmark.

The arrival pool has18,000 images at20% synthetic prevalence. There are200 gold
seed labels,2,000 calibration images,1,000 tuning images,3,000 held-out images
(600 positive), and400 separate peek images. Twelve disjoint batches of1,500
arrive. Each round selects150 previously unreviewed items. We retain full
per-seed, per-week traces, labels, selection strategy, cumulative reviewer votes,
and final unrounded prediction scores.

The head search tested C=1,10,100,1000,10000 and a256-unit MLP on the tuning set
across five development seeds. C100 had mean tuning AP0.895796, seed SD0.009298;
the next C1000 was0.886090. These are tuning results, not held-out evidence.
Larger C means weaker regularization in this estimator. The final hyperparameter
was frozen before scoring the held-out set.

Five selection methods run under matched per-item reviewer randomness:

- Random: a uniform sample from the accumulated unreviewed backlog.
- Uncertainty: the150 scores closest to the classifier's decision boundary.
- Highest score: the150 strongest positive predictions, favoring exploitation.
- Margin plus diversity: choose an uncertain candidate set, then greedy k-center
  coverage in embedding space to avoid spending every review on near duplicates.
- Hybrid: half uncertainty, half random exploration.

The simulator has three junior reviewers, disagreement, and a senior tie breaker.
Error probability increases with cross-fitted item difficulty. That difficulty
and the true pool labels are **available only to the simulator**; selection sees
the trained model, embeddings, backlog, and random generator. It cannot inspect
those hidden labels. These rates are assumptions, not measured human behavior.

The final run executes300 configurations: five main strategies, four review-
policy arms, four noise-sensitivity arms, and two drift-response arms, each with
20 seeds100–119. Six processes completed the fixed matrix in28 seconds after
the expensive embedding work had already finished. This wall time is one run on
a shared48-vCPU AMD EPYC7502 virtual host, not a serving capacity claim.

## Primary result, fixed before the test

The primary comparison is **hybrid versus random** at week12 on clean held-out
images. Average precision measures the quality of ranking positives across
thresholds; an AP-point gain is not the same thing as an accuracy-point gain.

| Method | Mean AP |95% item-and-seed bootstrap interval | Difference versus random |95% paired interval |
|---|---:|---:|---:|---:|
| Random |0.90094 |0.88472–0.91618 | — | — |
| Hybrid, primary |0.92059 |0.90614–0.93458 |+0.01965 |+0.01473–+0.02547 |
| Uncertainty, secondary |0.93158 |0.91737–0.94440 |+0.03064 |+0.02415–+0.03752 |
| Margin+diversity, secondary |0.93237 |0.91885–0.94463 |+0.03143 |+0.02480–+0.03835 |
| Highest score, secondary |0.90721 |0.89158–0.92263 |+0.00628 |−0.00244–+0.01485 |

The preregistered primary improvement is supported under this experiment's
assumptions. The highest-score difference is inconclusive; its interval includes
zero. We do not replace the primary comparison with the best secondary method.
The bootstrap resamples the same3,000 items and20 seeds jointly in1,000 draws.
Those seeds do not create60,000 independent test images. Weekly curve bands in
the plots are instead t intervals across seeds; their scope is explicitly
narrower than item-and-seed uncertainty.

## What the comparison costs

The budget is150 **items** per week, not150 reviewer votes. Random used3,962.3
votes on average; hybrid4,053.55; uncertainty4,141.2; diversity4,133.9; highest
score3,893.6. Uncertainty produced more disagreements and acquired noisier
labels: week12 mean training-label error7.92%, versus3.25% for random.
The model can improve even while the reviewed subset gets harder to label.

At equal item counts, adjudication and three-vote majority give the same decision
rule in this simulator, but adjudication spends the third vote only when juniors
disagree. Random's three-vote policy therefore uses5,400 votes for the same AP as
its3,962.3-vote adjudication arm. This follows from our specified simulator and
must not be sold as a universal human-review efficiency estimate. Single-label
random sampling achieved AP0.84373; single-label uncertainty0.91855. Quality
versus actual votes is more informative than an equal-items chart alone.

## Drift and threshold failure

At week7 the upload pipeline adds JPEG compression, blur, and lower brightness.
This is a fixed stress test on paired transformations, not new independent data.
Week12 uncertainty AP is0.9316 on clean images and0.7814 on1,000 transformed
held-out originals. Freezing the model after week6 leaves transformed AP0.7172;
training only on the latest four weeks gives0.7528. Throwing old labels away did
not help in this test. The full-history strategy remains best of those fixed
secondary drift arms.

We calibrate an automatic-reject threshold to empirical95% precision and a
review threshold to empirical90% recall on calibration data. Those are tuning
targets, not guarantees. Across20 uncertainty models, clean held-out reject
precision averages94.81% and review recall86.29%. Calibration is not deployment
certification.

For the preregistered first seed100,454 held-out items are auto-flagged:428 true
positives and26 false positives. Precision is94.27%, Wilson95% interval
91.74–96.06%; recall71.33%, interval67.59–74.81%. With stale thresholds under
the artificial shift, reject precision averages89.32% and recall37.78% across
seeds. Recalibrating restores mean review recall to90.56%, but review precision
drops to47.21%, increasing queue load. Report both: recovering recall can move
the operational burden somewhere else.

The CLIP zero-shot auto-reject threshold flags nothing. Its precision is
undefined, **not zero-percent precision and not a successful safe classifier**.
The raw legacy export uses a zero placeholder and NaN intervals for empty
denominators; the publication export represents undefined values as null and
keeps the count. The supervised head and a prompt-only score are different
systems with different failure modes.

## A dead end retained

The separate test-tuning demo trains60 head variants over40 training draws,
selecting once on the400-item peek split and once on the tuning split. We
expected the reused small benchmark to look optimistic. This realization did
**not** show that pattern: peek-selected AP averaged0.91693 on the untouched
main held-out split, and tune-selected AP0.91741. The peek-versus-held-out mean
gap was−0.01926; its displayed2.5th–97.5th percentiles are variation across
draws, not a confidence interval of the mean. Distribution/sampling differences
can outweigh winner-selection optimism in one experiment. We report the result
without searching for another seed that tells a nicer story.

## Implementation repairs that mattered

1. A tied calibration score originally let a prefix within a tie satisfy a
   precision target which the realizable threshold could not satisfy. A failing
   fixture with two equal top scores exposed it. Thresholds now evaluate entire
   tied groups.
2. Reviewer noise originally depended on selection order. Item-indexed common
   draws now preserve pairing when strategies review the same item.
3. A prose-only preregistration hash did not bind the data or the C environment
   variable. The final runner checks source/data hashes and C100 before scoring.
4. Analysis now rejects incomplete or duplicated configuration/seed matrices,
   incomplete weekly traces and truncated predictions. Score rounding was
   removed before the final run to preserve ranking precision.
5. Exact-pixel auditing found one pool/calibration and one calibration/held-out
   duplicate. Both calibration records were excluded before threshold fitting.
   Five within-pool duplicates remain disclosed. Near duplicates and pretrained
   encoder overlap are not ruled out.

## How to use this in an engineering explanation

Say what was measured, what was simulated, the selection boundary, and the
decision. For example: “I built a reproducible public-image benchmark to test a
review-and-retrain design. The model fits are real; reviewer errors are a stated
simulation. I froze a hybrid-versus-random AP comparison before opening3,000
held-out images. The paired interval favored hybrid, while calibration targets
did not automatically carry over to held-out data. I would add an independently
audited production sample before making automatic decisions.”

For a personal work story, fill in only verified role, volume, review-team size,
business loss tolerance, ownership, and actual operational result. Do not
attribute these benchmark numbers to a client or to past production work.

## Reproduce and inspect

See PREREG.md, frozen_inputs.json, requirements.lock at repository root,
src/loop.py, src/signal_thresholds.py and the full machine-readable traces.
The source at the original final run was d50ac31. Reporting/documentation added
later do not change that frozen implementation or its original predictions.
Code changes after opening the held-out set require a disclosed new experiment;
they cannot silently replace the original score.
