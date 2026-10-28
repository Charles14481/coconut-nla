# NLA results: run 2, small (new defaults)

## Headline

| Metric | Value |
|---|---|
| **Test FVE (best-val weights)** | **+0.359** |
| Test MSE (standardized) | 0.604542 |
| Test predict-the-mean baseline MSE | 0.943019 |
| Test per-thought-index baseline FVE | +0.042 |
| Best val FVE | +0.347 (MSE 0.644169, RL step 60) |
| Distinct test explanations | 1879/3000 |
| Activations (train / val / test) | 4800 / 192 / 3000 |
| Distinct teacher texts (train) | 4800/4800 |
| Wall time | 14 min 17 s |
| Peak GPU memory | 7.04 GiB |

FVE = 1 − MSE / (predict-the-train-mean MSE), in per-dimension standardized space.
Test is the full 500-question ProsQA test split, scored once with the weights that did
best on validation. Run 1 (large) scored **+0.298** on the same test split, using
6× more training questions and 2.4× the wall time.

## What changed since run 1

- **Per-thought teacher text** (largest gain). The old teacher was Coconut's answer, identical
  for all 6 thoughts of a question, with a linear-probe ceiling of FVE ≈ 0.31. The new one adds
  the rejected option, the reasoning step thought k replaced in curriculum training, and the
  latent's top-10 logit-lens tokens (probe ceiling ≈ 0.43). Example:
  `### Tom is a zhorpus. Not jompus. Thought 2: Every bompus is a wumpus. Lens: ### z Every p us . is imp w y`
- **No-repeat sampling off** (`--nla-no-repeat` to restore). The new teacher repeats tokens, so
  the AV could not reproduce it under the old rule. Explanations are up to 48 tokens.
- **AV adapter**: 4 extra soft-prompt tokens from a zero-initialized linear map of the
  *standardized* activation. The raw activation is dominated by 5 outlier dimensions.
- **AR trained on the AV's own greedy text**, in the warm start and in RL, alongside the
  teacher text or the RL samples.
- **AR readout**: the head reads the last token's hidden state and the mean over the explanation.
- **RL**: the AR trains at 1e-5 during RL (was the warm-start rate). This is the first setup in
  which RL improves validation instead of eroding the warm start. RL samples are drawn without
  gradients and re-scored in one batched pass, which keeps the 48-token explanations within 8 GB.
- Warm-start AV/AR learning rate 1e-4 (was 3e-5).

## Small-config comparison (800 train questions, 300 warm-start + 150 RL steps, seed 0)

Earlier rows were tested on the first 64 test questions only; this run used all 500.

| Run | Best val FVE | Test FVE |
|---|---|---|
| Old code (answer teacher, no-repeat, 12 tokens, lr 3e-5) | +0.249 | +0.262 (64 q) |
| Answer teacher + adapter + AR on AV text + lr 1e-4 | +0.267 | +0.277 (64 q) |
| Full teacher + adapter + AR on AV text + lr 1e-4 | +0.322 | +0.343 (64 q) |
| … + RL AR lr 1e-5 | +0.337 | +0.359 (64 q) |
| … + AR pooled readout, warm start only | +0.331 | +0.355 (64 q) |
| **This run: all of the above** | **+0.347** | **+0.359 (500 q)** |

Tried and not adopted: dropping the reasoning step from the teacher (val +0.308), warm-start
lr 2e-4 (no change), AV RL lr 1e-5 (val +0.330, explanations start to collapse).

## Training trajectory (validation)

★ = new best validation loss (those weights are kept for the test pass).

| Phase | Step | Val MSE | Val FVE | Teacher-text val FVE | NLL | KL | Distinct |
|---|---|---|---|---|---|---|---|
| start | 0 | 0.986191 | -0.000 |  |  |  | 38/192 |
| warm start | 60 | 0.866448 | +0.121 ★ | +0.143 | 1.188 |  | 92/192 |
| warm start | 120 | 0.739120 | +0.251 ★ | +0.287 | 0.962 |  | 108/192 |
| warm start | 180 | 0.680540 | +0.310 ★ | +0.324 | 0.794 |  | 120/192 |
| warm start | 240 | 0.667034 | +0.324 ★ | +0.348 | 0.737 |  | 123/192 |
| warm start | 300 | 0.659980 | +0.331 ★ | +0.363 | 0.646 |  | 126/192 |
| RL | 30 | 0.647214 | +0.344 ★ |  |  | -0.02 | 129/192 |
| RL | 60 | 0.644169 | +0.347 ★ |  |  | 0.04 | 129/192 |
| RL | 90 | 0.649164 | +0.342 |  |  | 0.22 | 123/192 |
| RL | 120 | 0.644725 | +0.346 |  |  | 0.24 | 128/192 |
| RL | 150 | 0.644726 | +0.346 |  |  | 0.08 | 126/192 |

Sample test explanations (from the first two test questions):
`### Polly is a zumpus. Not brimpus. Thought 1: Polly is a zumpus. Lens: ### z Every . p us ae is a`,
`### Alex is a hilpus. Not sterpus. Thought 3: Every sterpus is a hilpus. Lens: ### h Every p . us is ster ?`

## Known gaps

- The AV still gets the reasoning step mostly wrong (often the right shape with the wrong
  species) and repeats the thought number across thoughts of one question.
- The warm start was still improving at step 300 (teacher-text val FVE +0.363 and rising),
  so a longer run should help.

## Command

```bash
# Run from coconut-main/ in the coconut conda env (WSL).
python -u nla.py \
  --nla-train-examples 800 --nla-val-examples 32 --nla-test-examples 500 \
  --nla-warmup-steps 300 --nla-steps 150 --nla-seed 0
```

Full output is in `log.txt`.
