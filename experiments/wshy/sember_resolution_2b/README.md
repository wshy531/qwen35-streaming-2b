# S-EMBER resolution/cost diagnostic

Independent adapter for LS, A06, native uncompressed evidence and text-only.
Existing policies, frozen delivery, prompts and default entry points are unchanged.
Job 218458 completed on 2026-10-25 and its outputs were independently verified.
See [the stage publication](../sember-resolution-stage-20261025/README.md) for
results, source bindings and portable read-only verification. Future cluster
execution still needs separate user approval.

## Scope

All three visual methods receive the **same author-annotated evidence pairs**.
This is an oracle-evidence input diagnostic on 102 already-open questions / 18
videos. It is not full-prefix streaming evaluation or evidence-retrieval quality.
Banks reset for every question; pairs are encoded once per question/resolution,
shared by LS/A06 and all five option rotations. Overlapping evidence across
different questions is re-encoded. No full-history feature list is retained.
Raw decoded RGB used by this offline diagnostic is not part of a deployment
memory claim. Report that distinction when discussing memory cost.

- low: 192x256, 6x8 projected grid, 48 tokens/pair, 384-token bank.
- mid: 288x384, 9x12 grid, 108 tokens/pair, 864-token bank.
- high: 384x512, 12x16 grid, 192 tokens/pair, 1536-token bank.

All three canvases preserve the historical 3:4 letterbox procedure.

The recent window stays at four pairs. LS history capacity stays at four pair
equivalents; A06 keeps one open root and three closed-root equivalents. Event
threshold, merge weights, stable ordering and A06 depth selector are unchanged.
Spatial coordinate bounds and readout offset spans use the actual grid.
Resolution **and token budget** grow together. This is not an equal-token test.

## Historical reuse

Job 218446 contains 510 low-native, 510 mid-native and 510 text-only answers.
`runner.history()` checks pinned artifact hashes, the complete historical source
lock and all 102 question identities. CPU checks reproduce every endpoint/text
processor input. GPU replays one SHA-selected question per source video in all
five rotations for each historical condition: 270 answers. All generation IDs,
text, input IDs, positions and frame/pixel digests must match. Any mismatch stops
the job; there is no automatic fallback to another experiment or resubmission.

New work: 510 native-high + 3060 LS/A06 answers + 270 replay = 3840 model answers,
at most 30720 generated tokens. Accuracy uses 5100 unique condition answers:
1530 reused + 3570 new. Replay is excluded. Text-only is counted once.
Five rotations are repeated measurements of 102 questions, not 510 questions.

Costs compare the same 18 SHA-selected questions x five rotations for all ten
conditions. Answer timing excludes video decode and processor preparation;
native timing includes its vision forward. Memory encoding/update times are
reported separately, charging the complete shared encode cost to each policy.
Peak GPU allocation is recorded separately by phase/resolution; memory-phase
peaks cover the shared LS/A06 run, not separately measured per-policy peaks.
No automatic default change: report the accuracy/cost frontier per strategy,
paired corrections/losses, categories, per-video results and unresolved answers.
Manual semantic review remains pending; no format-compliance metrics are used.

## Checks and entry point

`scripts/run_2b_sember_resolution.py --stage local --out <new-local-run>` runs
61 light tests (41 existing + 20 resolution tests), AST/whitespace and frozen-file checks,
historical reuse verification, and 12 real-frame checks at three resolutions (24 historical pixel matches, 12 new high-grid checks) on the four
already downloaded local videos. Each test suite uses a separate process.

`--stage plan --out <new-run>` creates the request, source and reuse locks.
`--stage cpu --out <run> --model-dir <model>` requires an approved four-CPU
Slurm allocation with GPU hidden. It runs all 61 light tests plus eleven genuine
torch tests, including exact low-grid LS/A06 tensor/merge parity. It then checks
18 videos / 306 question-resolution selections / 2040 native/text inputs,
low-grid readout embedding/position equivalence, three tiny native and six tiny
memory answers. No pretrained weights are loaded by CPU checks.

`--stage gpu --out <run>` requires a verified CPU proof. It sequentially loads
the original implementation for native/replay and the existing streaming
implementation for LS/A06: **two FP32 loads, one resident model at a time**.
Memory QA guards both banks and model RNG/positions. Reset happens only after
both methods finish all five rotations. Visual-forward counts are checked.

`--stage score --out <run>` validates completion hashes before content scoring.

`cluster_combined.sh` proposes one g078t GPU, four CPUs, 48 GiB RAM, six-hour
Slurm limit, `comment=gx_llm`, no requeue; a 21300-second process-tree timeout,
40 GiB host-RSS watchdog and 72 GiB GPU allocator ceiling also apply. It uses
only existing wsy model/data/environment and writes only inside wsy. The script
does not install packages or download data. Submission is not authorized merely
by preparing this file.

## Native high-resolution workspace

Historical B/R input-length deltas are verified on all 510 question/rotation
pairs. New high inputs are projected at up to 39439 tokens (39447 with the
8-token allowance); 8 questions exceed the previous 24K ceiling. The explicit
new ceiling is 40960, with no truncation. CPU preflight must match every projected
length. `resources.json` records the full derivation and locks it in the protocol.

At 39439 tokens and eight FP32 attention heads, one full score matrix is about
46.36 GiB; two such buffers already exceed 80 GiB. Simple two-GPU layer placement
would not split this per-layer workspace. Instead the independent native adapter
uses 512-query tiles, with ALL keys/values and the full original forward/cache.
One score tile is about 0.60 GiB; this is not a measured total-memory guarantee.
Dtype, SDPA math backend, weights, pixels, timestamps, positions and prompt stay
fixed. All native conditions use this adapter for comparable latency. The streaming
memory implementation and vision attention retain their existing kernels.

Eleven mandatory torch checks include causal/additive masks, cached masking,
future isolation, single-token decode and dispatch restoration. Three tiny native
prefill comparisons require rtol=1e-5/atol=1e-6 plus identical argmax. These are
additional forward checks, not extra generated answers. The 270 real-model
historical answers must still reproduce generation IDs/text exactly. Failure
stops before new high answers, with no fallback or automatic second allocation.
After replay, the longest high-resolution question is executed first.

Proposal remains one >=80-GiB-class GPU, 4 CPUs, 48 GiB host RAM, at most 6 hours,
64 GiB free-memory entry check, 72 GiB allocator cap and 40 GiB process-tree RSS.
One full model is resident at a time; two sequential FP32 loads. Two GPUs were
considered and permitted for planning, but this implementation does not implement
unverified tensor parallelism or claim that layer sharding solves the workspace.
The 6-hour limit is conservative scheduling headroom, not a measured runtime.
