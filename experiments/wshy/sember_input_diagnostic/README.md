# S-EMBER input-quality diagnostic

Approved 2026-10-24: 102 development questions from 18 sources, four arms,
five cyclic option positions, one original FP32 model load, one Slurm job.

- B: historical uncompressed evidence, 192×256, 1 FPS.
- R: identical source frames and pairing, 288×384 from original decoded pixels.
- F: 192×256, target-2-FPS candidates inside the exact old evidence endpoints,
  unioned with every old frame. For odd cardinality omit only the latest newly
  added frame. Record actual spacing; this is not a strictly uniform 2-FPS grid.
- T: text only, same original model and answer-content prompt.

Each option appears at each letter once. Map predicted letters back to original
content before scoring. Five positions remain repeated measurements of one
question, not five independent questions. No best-position selection or format
metric. Unparsed content and human semantic review remain pending.

`scripts/run_2b_sember_input_diagnostic.py --stage plan|cpu|gpu|score --out PATH`
is the entry point. CPU also requires `--model-dir PATH`. CPU/GPU stages require
the explicitly approved Slurm allocation. `cluster_combined.sh` submits neither
itself nor a retry: hidden-GPU preflight precedes GPU evaluation in one job.

The CPU stage verifies the existing 28 contracts plus 13 new contracts, checks
all real source RGB/PTS and all 2,040 processor inputs, and performs four tiny
random-model calls without loading real weights. The GPU stage first replays
all 102 historical B/shift-0 answers, including input and generated-ID hashes,
then extends the grid. Any difference stops execution. All four arms use full
original-model forward with the previously disclosed temporal-grid positional
adapter; this is not a claim of unmodified Hugging Face generate.

Budgets: 1 GPU, 4 CPUs, 48 GiB RAM, 4 hours, comment `gx_llm`, no requeue;
24,000 total context tokens, 8 generated tokens/answer, 2,040 answers,
16,320 generated tokens, 72 GiB GPU allocator limit, 40 GiB process-tree RSS.
Any failed assertion, OOM, timeout or decode error stops, with partial records
preserved and no automatic resubmission. No silent truncation or downscaling.

The higher-resolution/FPS arms use more visual tokens; this diagnoses input
bottlenecks and is not an equal-token comparison of memory algorithms. Author
evidence intervals are an offline oracle, not independently proven sufficient
evidence or a deployable streaming selector. Frozen code and old results are
read-only. New runtime status is recorded in the independent output directory.
