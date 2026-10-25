# S-EMBER 102 official questions

This separate adapter evaluates the existing M0, M2, M2_A06, LS_shared384_v1,
reservoir4 and text_only implementations on 102 unique original official
questions across 18 videos. The cohort was pinned before predictions, including
the four already opened videos. This is development data. File size and declared
evidence gaps influenced selection, so it is not a representative random sample.

The baseline commit is 4212d2399954f19700b568cc806ed82fece3349f. The original
three-source pilot remains separate. Frozen algorithms, thresholds, compression,
position processing, FP32, 192x256 letterbox and 1 FPS are reused.
The completed v1 run used duplicated option labels. The v2 adapter removes one
structural prefix before rendering each option, preserving content and order.
The frozen package and original input/output files remain unchanged. New runs
record `prompt_version=single-option-label-v1`; v1 receipts cannot authorize v2.

The new protocol explicitly allows 512 question suffix tokens instead of the old
128-token guard, so longer official questions remain complete. It does not alter
the tokenizer, context limit, visual bank budget or output budget. The
CPU stage checks all actual suffix lengths and refuses truncation. It also uses
the longest actual question in one of the two tiny-model query rounds.

The `media` stage creates actual RGB/PTS specifications using the frozen sampler
and matches media against the completed download receipt. The `prepare` stage
binds the source, media, input, sampling and model identities. CPU validation
includes inherited 209 discovered regressions (one platform-specific skip may
apply), 103 existing contracts, 11 new contracts, all 18 actual media prefixes,
102 complete questions and 12 tiny random-model answers. No pretrained weights
are loaded in CPU stages. They run with the allocated GPU hidden.

The GPU stage requires a matching successful CPU receipt, then loads the own
FP32 2B model once. Every complete arriving pair is visually encoded once and
shared by the five bounded visual banks. Queries at different times pause a
single continuous replay. An unmatched frame is held until its pair arrives;
queries and EOF do not pad. Every answer uses fresh language state and guards
all banks, descriptors, RNG and model position state. Reset is checked for every
source and visual arm (90 checks).

The fixed grid is 102 x 6 = 612 answers, <=8 generated tokens each, <=4896 total.
The planned single Slurm allocation is 1 GPU / 4 CPU / 24 GiB / 45 minutes,
comment=gx_llm, no requeue and no automatic resubmission. Stage failures stop
the combined script before subsequent stages. All paths and caches stay under
/home/share/czy/wsy. No environment change or additional data download is needed.

Offline scoring reports unambiguous option-content correctness, question
categories, author evidence-gap bins, actual committed sampled evidence position
relative to M0, answer-letter distributions, and per-question gains/losses versus
M0/M2/A06. Author intervals are proxies; neither temporal age nor node provenance
proves that old evidence is necessary or readable. Human semantic review remains
pending. These 102 official questions alone do not measure paired current
perception versus memory tradeoffs. They first establish comparable baselines.

Unified entry: scripts/run_2b_sember_102.py --stage media|prepare|cpu|gpu|score.
The Slurm script is a prepared artifact; local preparation does not submit it.
