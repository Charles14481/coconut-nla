# NLA results: run 5, baseline metrics check

A small run to try three added metrics: FVE against the per-thought-index mean,
FVE in the raw (unstandardized) activation space, and a question-only AR baseline.

## Headline

| Metric | Value |
|---|---|
| **NLA test FVE (standardized, vs train mean)** | **+0.305** |
| NLA test FVE vs per-thought-index mean | +0.276 |
| NLA test FVE in raw activation space | +0.334 |
| Per-thought-index mean, test FVE | +0.040 |
| **Question-only AR test FVE** (300 steps, still improving) | **+0.485** |
| NLA minus question-only | −0.180 |
| Activations (train / val / test) | 1800 / 96 / 600 |
| Wall time | 8.6 min NLA + 3.7 min question-only AR |
| Peak GPU memory (NLA training) | 5.12 GiB |

The question-only AR has the same architecture, start weights (Coconut's GPT-2), learning
rates, batch size and step count as the NLA's warm start. It reads
`"{question}\nThought {k}:"` (~360 tokens) instead of a 64-token explanation.

## Reading

- **FVE vs the per-index mean**: this costs only ~0.03. Thought index explains 4% of the
  standardized variance, so the "Thought k" text earns the NLA almost nothing.
- **Raw-space FVE**: higher than the standardized FVE (+0.334 vs +0.305). The five outlier
  dimensions are easier to predict, so z-scoring makes the metric *stricter*, not inflated.
- **Question-only AR**: it beats this small NLA by 0.18 and is still climbing. That's expected,
  not a sign of failure. Coconut is frozen and greedy, so each latent is a fixed function of
  (question, index), and this AR *is* Coconut's own network reading that question. Its score
  measures how much of the latent this reader can recover from the full input. That makes it a
  ceiling-like reference for the AR, not a floor for the NLA. The AV never sees the question, so
  nothing in an explanation can come from the question except through the latent.

## Training trajectory (validation)

| Phase | Step | Val FVE | vs index mean | Teacher-text val FVE |
|---|---|---|---|---|
| start | 0 | -0.000 | -0.042 | |
| warm start | 100 | +0.195 | +0.161 | +0.258 |
| warm start | 200 | +0.218 | +0.185 | +0.340 |
| warm start | 300 | +0.254 | +0.222 | +0.341 |
| RL | 10 | +0.256 | +0.224 | |
| RL | 20 | +0.263 | +0.232 | |
| RL | 30 | +0.270 | +0.239 | |
| question-only AR | 100 | +0.352 | | |
| question-only AR | 200 | +0.416 | | |
| question-only AR | 300 | +0.435 | | |

## Command

See `command.sh`. Full output is in `log.txt`.
