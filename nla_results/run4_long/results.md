# NLA results: run 4, long (overnight)

## Headline

| Metric | Value |
|---|---|
| **Test FVE (best-val weights)** | **+0.630** |
| Test MSE (standardized) | 0.352269 |
| Test predict-the-mean baseline MSE | 0.952908 |
| Test per-thought-index baseline FVE | +0.042 |
| Best val FVE | +0.623 (MSE 0.360837, RL step 1800) |
| Distinct test explanations | 2808/3000 |
| Activations (train / val / test) | 60000 / 384 / 3000 |
| Teacher text length (tokens) | mean 57.0, max 65 (limit 64) |
| Wall time (attempt 3) | 4 h 48 min (02:30 to 07:18) |
| Peak GPU memory in training | 7.21 GiB |

FVE = 1 − MSE / (predict-the-train-mean MSE), in per-dimension standardized space. Test
is the full 500-question ProsQA test split, scored once with the weights that did best
on validation (64 questions).

## Compared with earlier runs (same test split)

| Run | Train questions | Warm-start / RL steps | Test FVE |
|---|---|---|---|
| run1_large (answer-only teacher) | 5000 | 2000 / 300 | +0.298 |
| run2_small (plain lens) | 800 | 300 / 150 | +0.359 |
| Current defaults, small config | 800 | 300 / 150 | +0.384 |
| run3_moderate (both lenses) | 2000 | 1000 / 300 | +0.509 |
| **This run (current defaults)** | 10000 | 9000 / 2000 | **+0.630** |

## Training trajectory (validation)

★ = new best validation loss (those weights are kept for the test pass).

| Phase | Step | Val FVE | Teacher-text val FVE | NLL | KL | Distinct |
|---|---|---|---|---|---|---|
| start | 0 | -0.000 ★ |  |  |  | 14/384 |
| warm start | 900 | +0.508 ★ | +0.542 | 1.931 |  | 341/384 |
| warm start | 1800 | +0.556 ★ | +0.576 | 1.687 |  | 356/384 |
| warm start | 2700 | +0.573 ★ | +0.591 | 1.602 |  | 369/384 |
| warm start | 3600 | +0.584 ★ | +0.602 | 1.457 |  | 367/384 |
| warm start | 4500 | +0.597 ★ | +0.605 | 1.282 |  | 364/384 |
| warm start | 5400 | +0.603 ★ | +0.611 | 1.377 |  | 367/384 |
| warm start | 6300 | +0.603 ★ | +0.609 | 1.223 |  | 359/384 |
| warm start | 7200 | +0.601 | +0.608 | 1.189 |  | 360/384 |
| warm start | 8100 | +0.602 | +0.610 | 1.115 |  | 359/384 |
| warm start | 9000 | +0.605 ★ | +0.608 | 1.105 |  | 362/384 |
| RL | 200 | +0.616 ★ |  |  | -0.08 | 367/384 |
| RL | 400 | +0.621 ★ |  |  | -0.06 | 351/384 |
| RL | 600 | +0.618 |  |  | 0.07 | 358/384 |
| RL | 800 | +0.618 |  |  | 0.00 | 355/384 |
| RL | 1000 | +0.620 |  |  | 0.06 | 362/384 |
| RL | 1200 | +0.619 |  |  | 0.05 | 363/384 |
| RL | 1400 | +0.618 |  |  | -0.01 | 353/384 |
| RL | 1600 | +0.621 ★ |  |  | 0.14 | 352/384 |
| RL | 1800 | +0.623 ★ |  |  | 0.10 | 366/384 |
| RL | 2000 | +0.622 |  |  | 0.08 | 367/384 |

**Reading the trajectory**:

- **Warm start**: plateaued at val FVE ≈ +0.60 from about step 5400 on, with the AV's text
  within 0.01 of the teacher text's own score (≈ +0.61). Once the AV can write the teacher
  text, the ceiling is what that text contains, so more warm-start steps or data alone won't
  help much.
- **RL**: pushed past that ceiling, adding +0.018 (+0.605 → +0.623), most of it in the first
  400 steps.

## Sample test explanations

The first test question, thoughts 1, 2, 3 and 5:

```
### Sally is a hilpus. Not sterpus. Thought 1: Sally is a yumpus. Lens: ### h . Every p us ster a mp ? t z er y is | HIP HT Hale LH XD Ic HY Hernandez Zel Hyp Hai Yaz Hanson AH Hansen
### Sally is a hilpus. Not sterpus. Thought 2: Every yumpus is a yerpus. Lens: ### h p Every . us ? is y imp ster er a n w | Hale Hansen HRC Hin Heller Hyp Hy LH Tillerson Hiro Harden Hanson Hib HB OH
### Sally is a hilpus. Not sterpus. Thought 3: Every sterpus is a hilpus. Lens: ### h p . Every us is ? y imp ster n a er Is | Hale Hanson Hait HRC Heller Hin Hansen Hernandez Hy Hib Tillerson Hatch Hyp Holder Hegel
### Sally is a hilpus. Not sterpus. Thought 5: Done. Lens: ### h . p Every us is ? y imp ster n a er Is | Hale Hanson Hernandez Hin Heller HRC Hiro Hansen Hait Hib Hegel Hy Tillerson Hatch Hyp
```

There are no repetition loops. The thought number and `Done.` now vary across a question's
thoughts. The z-scored lens lists H-words, which point toward the answer "hilpus".

## Crashed attempts

Attempts 1 and 2 crashed with `RuntimeError: CUDA driver error: device not ready`. Their logs
are in `log_attempt1_crash.txt` and `log_attempt2_crash.txt`, with GPU usage in
`gpu_usage_attempt*.log`.

1. **Attempt 1** (00:34 to 01:21) crashed during setup. `lens_tokens` built the logit lens for
   all 60,000 training latents at once (~12 GB). It now works in chunks of 64 (1.2 GB peak
   at this scale, same output).
2. **Attempt 2** (01:24 to 02:29) crashed 16 minutes into the warm start, with ~7.1 of 8.2 GB in
   use. Both attempts ran with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which the
   calibration runs didn't use. It relies on CUDA virtual-memory mapping, which WSL handles
   poorly. Attempt 3 dropped it and ran clean at a flat 7.95 GB (allocator cache included).

Memory changes made for this run, none of which change the math:

- The frozen target model moves to the CPU after setup.
- Gradients are freed right after each optimizer step.
- The warm-start AR backward pass is split by text set.
- In RL, the AR trains before the AV's log-prob graph is built.

## Possible next steps

- RL gains were small after step 400. A higher KL coefficient or AV RL learning rate, or
  more samples per group, might let RL move further from the warm start.
- The warm-start ceiling is the teacher text. Richer per-thought content (the AV still writes
  the reasoning step with the wrong species, as in thought 2 above) would raise it.

## Command

See `command.sh`. Full output is in `log.txt`.
