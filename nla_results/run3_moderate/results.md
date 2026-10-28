# NLA results: run 3, moderate (plain + z-scored lens)

## Headline

| Metric | Value |
|---|---|
| **Test FVE (best-val weights)** | **+0.509** |
| Test MSE (standardized) | 0.469857 |
| Test predict-the-mean baseline MSE | 0.956382 |
| Test per-thought-index baseline FVE | +0.042 |
| Best val FVE | +0.502 (MSE 0.498384, final RL step) |
| Distinct test explanations | 2452/3000 |
| Activations (train / val / test) | 12000 / 192 / 3000 |
| Teacher text length (tokens) | mean 57.0, max 64 |
| Wall time | 56 min 13 s |
| Peak GPU memory | 8.28 GiB |

FVE = 1 − MSE / (predict-the-train-mean MSE), in per-dimension standardized space. Test
is the full 500-question ProsQA test split, scored once with the weights that did best
on validation (32 questions).

## Compared with earlier runs (same test split)

| Run | Train questions | Warm-start / RL steps | Test FVE |
|---|---|---|---|
| run1_large (answer-only teacher) | 5000 | 2000 / 300 | +0.298 |
| run2_small (plain lens) | 800 | 300 / 150 | +0.359 |
| Same settings as this run, small config | 800 | 300 / 150 | +0.384 |
| **This run** | 2000 | 1000 / 300 | **+0.509** |

## What's new since run2_small

- **Two logit lenses in the teacher text** (`--nla-lens both --nla-lens-tokens 15`). The plain
  lens lists the latent's top next-token predictions, which are mostly the same generic
  tokens for every latent (`### Every . p us`). The z-scored lens ranks tokens by logit
  z-scored against the train latents, which lists what is unusual about this latent
  (`Zi Zurich Z CentOS Zah …`). GPT-2 ties its LM head to its input embeddings, so these are
  the tokens whose embeddings point most unusually along the latent. In a linear
  bag-of-words probe the full teacher's ceiling rises from ~0.43 to ~0.48.
- **Clean lens words**: byte fragments (which decode to `�`), duplicates after stripping
  whitespace, and words that are more than one token after a space are skipped.
- **Repeat window** (`--nla-repeat-window 3`): a token among the last 3 can't be written
  again. Greedy decoding otherwise loops on rare lens tokens (`ster ster ster …`). A window
  of 3 blocks 0.4% of teacher texts; 4 would block 42%.
- The AR runs only GPT-2's transformer, skipping LM-head logits it never used, which keeps
  64-token explanations within the GPU's memory.

## Training trajectory (validation)

★ = new best validation loss (those weights are kept for the test pass).

| Phase | Step | Val FVE | Teacher-text val FVE | NLL | KL | Distinct |
|---|---|---|---|---|---|---|
| warm start | 200 | +0.316 ★ | +0.398 | 2.465 |  | 124/192 |
| warm start | 400 | +0.390 ★ | +0.458 | 2.067 |  | 139/192 |
| warm start | 600 | +0.438 ★ | +0.503 | 2.009 |  | 155/192 |
| warm start | 800 | +0.437 | +0.506 | 1.734 |  | 162/192 |
| warm start | 1000 | +0.470 ★ | +0.516 | 1.586 |  | 156/192 |
| RL | 60 | +0.489 ★ |  |  | 0.02 | 153/192 |
| RL | 120 | +0.493 ★ |  |  | 0.10 | 157/192 |
| RL | 180 | +0.494 ★ |  |  | 0.25 | 157/192 |
| RL | 240 | +0.499 ★ |  |  | -0.07 | 163/192 |
| RL | 300 | +0.502 ★ |  |  | 0.34 | 160/192 |

Neither phase had plateaued, so a longer run should do better.

## Sample test explanations

```
### Sally is a hilpus. Not yimpus. Thought 2: Every sterpus is a hilpus. Lens: ### h Every p . ? us is y imp n er a mp Is | h HI Hik HIP Hin HI Hik HIP Hin HI Hik HIP Hin Hua Hik
### Sally is a hilpus. Not tumpus. Thought 5: Done. Lens: ### h p . Every us is ? y imp n a Is er mp | h Hancock Hik HI Hin Hale Hik Hans…
```

## Known gaps

- **Period-4 loops** in the z-lens (`HI Hik HIP Hin HI Hik HIP Hin …`), which a window of 3
  can't catch. `--nla-distinct-lens` (added after this run) forbids repeating any token after
  `Lens`, which teacher texts never do.
- The AV still writes the reasoning step mostly wrong (right shape, wrong species).
- Teacher-text val FVE (+0.516) is still above what the AV achieves (+0.470 before RL).

## Command

See `command.sh`. Full output is in `log.txt`.
