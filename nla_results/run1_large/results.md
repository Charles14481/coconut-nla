# NLA results: run 1, large

## Headline

| Metric | Value |
|---|---|
| **Test FVE (best-val weights)** | **+0.298** |
| Test MSE (standardized) | 0.668985 |
| Test predict-the-mean baseline MSE | 0.953186 |
| Test per-thought-index baseline FVE | +0.042 |
| Best val FVE | +0.297 (MSE 0.680685) |
| Distinct test explanations | 407/3000 |
| Activations (train / val / test) | 30000 / 600 / 3000 |
| Distinct teacher texts (train) | 646/30000 |
| Wall time | 34 min 45 s |

FVE = 1 − MSE / (predict-the-train-mean MSE), in per-dimension standardized space.
FVE > 0 means the explanations carry information about the latent. Test is the full
500-question ProsQA test split, scored once with the weights that did best on validation.

## Training trajectory (validation)

★ = new best validation loss (those weights are kept for the test pass).

| Phase | Step | Val MSE | Val FVE | Teacher-text val FVE | NLL | KL | Distinct | Sample explanations |
|---|---|---|---|---|---|---|---|---|
| start | 0 | 0.968636 | -0.000 |  |  |  | 47/600 | `[' a quus. is imp rimpper or rem', ' a quus. is or imp lerpel ker', ' a quus. is or imp le` |
| warm start | 400 | 0.697976 | +0.279 ★ | +0.294 | 0.325 |  | 114/600 | `['### Davis is a lempus.', '### Sam is a scrompus.', '### Sam is a scrompus.']` |
| warm start | 800 | 0.712637 | +0.264 | +0.296 | 0.365 |  | 109/600 | `['### Wren is a lempus.', '### Sam is a scrompus.', '### Sam is a scrompus.']` |
| warm start | 1200 | 0.686663 | +0.291 ★ | +0.300 | 0.238 |  | 124/600 | `['### Sally is a lempus.', '### Sam is a scrompus.', '### Sam is a scrompus.']` |
| warm start | 1600 | 0.686340 | +0.291 ★ | +0.295 | 0.226 |  | 144/600 | `['### Fae is a lempus.', '### Sam is a scrompus.', '### Sam is a scrompus.']` |
| warm start | 2000 | 0.680685 | +0.297 ★ | +0.289 | 0.163 |  | 153/600 | `['### Rex is a lempus.', '### Tom is a scrompus.', '### Sam is a scrompus.']` |
| RL | 60 | 0.688431 | +0.289 |  |  | -0.08 | 149/600 | `['### Rex is a lempus.', '### Fae is a scrompus.', '### Jack is a scrompus.']` |
| RL | 120 | 0.688608 | +0.289 |  |  | 0.30 | 141/600 | `['### Rex is a lempus.', '### Tom is a scrompus.', '### Sam is a scrompus.']` |
| RL | 180 | 0.702863 | +0.274 |  |  | 0.11 | 149/600 | `['### Rex is a lempus.', '### Sam is a scrompus.', '### Sam is a scrompus.']` |
| RL | 240 | 0.702148 | +0.275 |  |  | -0.01 | 142/600 | `['### Rex is a lempus.', '### Sam is a scrompus.', '### Sam is a scrompus.']` |
| RL | 300 | 0.692351 | +0.285 |  |  | 0.01 | 142/600 | `['### Rex is a lempus.', '### Sam is a scrompus.', '### Sam is a scrompus.']` |

## Command

```bash
# Run from coconut-main/ in the coconut conda env (WSL).
python -u nla.py \
  --nla-train-examples 5000 --nla-val-examples 100 --nla-test-examples 500 \
  --nla-warmup-steps 2000 --nla-steps 300 \
  --nla-batch-size 4 --nla-group-size 4 \
  --nla-av-lr 3e-5 --nla-ar-lr 3e-5 --nla-rl-lr 3e-6 \
  --nla-explanation-tokens 12 --nla-kl-coef 0.05 --nla-seed 0
```

Full output is in `log.txt`.
