# LS shared history with depth-priority merging

This is an isolated development comparison, not a default model change.

The six methods are M0 (last eight full pairs), fixed-seed reservoir (four
promoted old pairs plus four recent), M2, A06, original LS, and LS_depth.
All retain at most 384 visual tokens. LS has four recent complete pairs and a
shared 192-token history pool. LS_depth changes only the adjacent history
overflow selector from `(cost, index)` to `(resulting_depth, cost, index)`.
Feature merging, time labels, position heuristic, event threshold and QA
readout remain unchanged. A shallower tree need not retain better evidence.

LS originates in the partner's `experiments/wshy/online_memory_2b/shared.py`,
commit `4212d2399954f19700b568cc806ed82fece3349f`. Its unmodified snapshot is
included for CPU compatibility checks, with Git blob verified by prepare.py.
Other policies reuse the existing frozen project modules.

New real frame pairs are encoded once and update all six banks. Event flags
use one shared, bounded previous descriptor; standalone cost estimates must
charge its bytes to each event method. There is no persistent language main,
full-history feature archive, child tree archive, or saved past RGB library.
Every question opens a fresh language context, emits at most eight tokens
through callbacks, and releases that context. Bank fingerprints are checked
before and after each question.

Inputs are the unchanged 171 and T98 development questions plus 34 causal
midstream regressions, 303 question-time items across 112 source/protocol
groups. Long videos retain the original complete-pair scheduling to EOF;
short videos use the existing query-only odd-frame duplication and cannot
resume afterward. These are causal replays, not 1x wall-clock playback.

Answer correctness is the primary measure. Format compliance is separate.
Ambiguous, missing and failed answers remain in the denominator. The long
28 original questions include 16 supported, 11 ambiguous and one not yet
established question; report those separately from six manual candidates.
These opened data cannot establish unseen-source generalization.

Workflow: prepare inputs, run CPU-only preflight, freeze source/input hashes,
deploy only small code/metadata, run the same CPU preflight remotely, then
obtain the specific GPU authorization required by the cluster policy before
submitting `run_eval.sh`. No downloads, training or automatic retries.
Maximum: one FP32 load, 1818 answers, 14544 generated tokens, 3981 shared
visual encodings, one A800, four CPU cores, 24 GiB host RAM, 60 minutes.

Stop after this fixed comparison. Report paired gains and losses versus both
parents and the simple controls, depth/storage/update costs and old detail
failures. Do not sweep weights or promote the candidate by score alone.
Analyze this round before starting the separately queued Div24 experiment.
Progress belongs exclusively to the root 实验总纲.md.
