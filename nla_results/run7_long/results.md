# NLA results: run 7, long (full metric suite)

Same settings as `run4_long` (10000 train questions, 9000 warm-start and 2000 RL steps),
plus the added metrics and a question-only AR baseline trained for 9000 steps.

## Headline (test, best-val weights, 500 questions = 3000 latents)

| Metric | Value |
|---|---|
| **Test FVE (standardized)** | **+0.631** |
| FVE vs per-thought-index mean | +0.614 |
| FVE in raw activation space | +0.743 |
| Per-thought FVE (k = 1..6) | +0.610 +0.626 +0.637 +0.640 +0.641 +0.641 |
| Teacher-text FVE (the AR reading the teacher text) | +0.629 |
| **Question-only AR test FVE** | **+0.878** (NLA minus question-only: −0.247) |
| AV SFT NLL, real vs shuffled latent | 1.689 vs 9.458 nats/token (gap +7.77) |
| Frontier recall, NLA vs linear probe (matched budget) | 0.186 vs 0.307 (precision 0.143 vs 0.236) |
| Fidelity: parse-fail / full answer / species / step / thought index | 0.001 / 0.842 / 0.928 / 0.356 / 0.575 |
| Paraphrase FVE drop: canonical / lens shuffle / reworded | 0.000 (0% changed) / **+0.142** / +0.010 |
| Splice: clean / all-AR / all-mean / all-shuffled accuracy | 0.980 / 0.980 / 0.980 / 0.980 (recovered n/a) |
| Best val FVE | +0.624 (RL step 1800) |
| Distinct test explanations | 2802/3000 |
| Coconut accuracy | val 63/64, test 490/500 |
| Wall time | 8 h 44 min: 50 min setup, 4 h 2 min NLA training, 3 h 35 min question-only AR |
| Peak GPU memory in NLA training | 5.46 GiB (`gpu_usage.log`: ~7.5 GB with the allocator cache; 7.96 GB during the question-only AR) |

## Compared with earlier runs (same test split)

| Run | Train questions | Warm-start / RL steps | Test FVE |
|---|---|---|---|
| run1_large (answer-only teacher) | 5000 | 2000 / 300 | +0.298 |
| run2_small (plain lens) | 800 | 300 / 150 | +0.359 |
| run3_moderate (both lenses) | 2000 | 1000 / 300 | +0.509 |
| run4_long | 10000 | 9000 / 2000 | +0.630 |
| **This run** (same config, rerun) | 10000 | 9000 / 2000 | **+0.631** |

The FVE trajectory matched run4_long to within ~0.005 at every checkpoint, so the
result is reproducible.

## Fidelity detail (test)

| Thought k | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Species matches Coconut's answer | 0.93 | 0.93 | 0.92 | 0.93 | 0.93 | 0.92 |
| Step matches gold step k | 0.07 | 0.02 | 0.10 | 0.35 | 0.71 | 0.87 |

| Coconut on the question | Questions | Species match | Step match |
|---|---|---|---|
| Right | 490 | 0.932 | 0.357 |
| Wrong | 10 | 0.733 | 0.283 |

Species match is measured against *Coconut's* answer. On the questions Coconut gets wrong, the
explanation still states Coconut's (wrong) answer 73% of the time, so it describes the model
rather than the gold answer. That's only 10 questions.

## Training trajectory (validation, 64 questions)

★ = new best validation loss. Teacher FVE = the AR reading the teacher text. SFT gap = shuffled −
real teacher-text NLL per token. Frontier = recall, NLA vs probe. Species/step = fidelity.
Shuffle drop = FVE lost when lens words are shuffled.

| Phase | Step | Val FVE | Teacher FVE | SFT gap | Frontier | Species | Step | Shuffle drop |
|---|---|---|---|---|---|---|---|---|
| start | 0 | -0.000 ★ | -0.000 | +0.03 | 0.000 / 0.000 | 0.000 | 0.000 | 0.000 |
| warm start | 900 | +0.510 ★ | +0.542 | +2.26 | 0.164 / 0.271 | 0.891 | 0.201 | 0.078 |
| warm start | 1800 | +0.554 ★ | +0.577 | +3.68 | 0.154 / 0.261 | 0.914 | 0.344 | 0.109 |
| warm start | 2700 | +0.575 ★ | +0.591 | +4.79 | 0.193 / 0.276 | 0.904 | 0.318 | 0.121 |
| warm start | 3600 | +0.584 ★ | +0.601 | +5.12 | 0.156 / 0.262 | 0.896 | 0.339 | 0.113 |
| warm start | 4500 | +0.602 ★ | +0.605 | +5.93 | 0.174 / 0.280 | 0.904 | 0.232 | 0.112 |
| warm start | 5400 | +0.605 ★ | +0.609 | +6.30 | 0.193 / 0.283 | 0.919 | 0.294 | 0.135 |
| warm start | 6300 | +0.597 | +0.607 | +6.49 | 0.174 / 0.279 | 0.898 | 0.302 | 0.144 |
| warm start | 7200 | +0.598 | +0.610 | +6.61 | 0.165 / 0.270 | 0.893 | 0.362 | 0.133 |
| warm start | 8100 | +0.609 ★ | +0.609 | +7.41 | 0.182 / 0.285 | 0.906 | 0.255 | 0.135 |
| warm start | 9000 | +0.607 | +0.607 | +7.81 | 0.171 / 0.276 | 0.891 | 0.359 | 0.145 |
| RL | 200 | +0.617 ★ | +0.613 | +7.91 | 0.166 / 0.274 | 0.893 | 0.359 | 0.146 |
| RL | 400 | +0.620 ★ | +0.616 | +7.91 | 0.167 / 0.276 | 0.893 | 0.359 | 0.139 |
| RL | 600 | +0.618 | +0.614 | +7.94 | 0.165 / 0.274 | 0.893 | 0.352 | 0.148 |
| RL | 800 | +0.621 ★ | +0.614 | +7.94 | 0.165 / 0.276 | 0.891 | 0.352 | 0.147 |
| RL | 1000 | +0.622 ★ | +0.616 | +7.95 | 0.166 / 0.276 | 0.891 | 0.359 | 0.149 |
| RL | 1200 | +0.622 | +0.615 | +7.96 | 0.165 / 0.276 | 0.891 | 0.357 | 0.147 |
| RL | 1400 | +0.620 | +0.614 | +7.97 | 0.168 / 0.276 | 0.891 | 0.362 | 0.145 |
| RL | 1600 | +0.623 ★ | +0.618 | +7.95 | 0.166 / 0.276 | 0.891 | 0.357 | 0.149 |
| RL | 1800 | +0.624 ★ | +0.617 | +7.96 | 0.167 / 0.276 | 0.891 | 0.354 | 0.148 |
| RL | 2000 | +0.622 | +0.616 | +7.94 | 0.166 / 0.274 | 0.893 | 0.359 | 0.154 |

| Question-only AR step | 900 | 1800 | 2700 | 3600 | 4500 | 5400 | 6300 | 7200 | 8100 | 9000 |
|---|---|---|---|---|---|---|---|---|---|---|
| Val FVE | +0.677 | +0.744 | +0.790 | +0.815 | +0.836 | +0.846 | +0.858 | +0.857 | +0.870 | +0.866 |

## Reading

- **The NLA has reached its teacher's ceiling.** Test FVE (+0.631) equals the AR reading the
  teacher text itself (+0.629). The AV now writes text as informative as the teacher's, so
  further gains need richer teacher content, not more training.
- **The AV reads the latent, heavily.** With another question's latent, the teacher text
  costs 9.5 nats/token instead of 1.7. The gap kept growing through the whole warm start and
  into RL.
- **Thought index barely matters.** FVE against the per-index mean is only 0.017 lower. Thought 1
  is slightly harder (+0.610) than thoughts 3 to 6 (~+0.64).
- **What the explanations get right and wrong:**
  - They state Coconut's answer species 93% of the time, and the full answer 84% of the time.
  - They mostly fail the early reasoning steps (step match 0.02 to 0.10 for thoughts 1 to 3).
  - Later "matches" are mostly the easy `Done.`.
  - The thought index is right only 58% of the time. Thoughts past the last step look alike
    (sample: thoughts 4 and 5 both written as "Thought 5: Done.").
- **The frontier is under-verbalized.** At the same number of names, a linear probe on the latent
  recovers more of the BFS frontier than the explanation does (0.31 vs 0.19). The teacher text
  matched the probe on val (0.295 vs 0.297), so the NLA loses frontier information exactly where
  it gets the early steps wrong.
- **Lens order carries real information.** Shuffling the lens words costs 0.14 FVE, a lot more
  than in run 6, the small sanity check (0.012). The teacher lists lens words by logit rank, so their order is
  legitimate content, not hidden encoding. Rewording the template costs only 0.01, and canonical
  re-rendering changes nothing (the AV always writes the exact template).
- **Raw-space FVE is higher (+0.743).** The five outlier dimensions are easy, so the standardized
  number is the stricter one.
- **Question-only AR, +0.878.** A copy of Coconut's GPT-2 reading the whole question (~360
  tokens) reconstructs the latents better than a 64-token explanation does. That's expected,
  because each latent is a fixed function of the question. It's a ceiling for what this reader
  can extract, not a floor the NLA should beat. It was still improving slowly at 9000 steps.
- **Splice: Coconut's answers don't depend on its latents.** For all 500 test questions, swapping
  in the train mean or another question's latents (all six, or any single one) leaves every
  answer unchanged. The answer log-prob doesn't move at 3 decimals. So "accuracy recovered" is
  undefined for this checkpoint. The self-check passed: the KV-cache loop reproduced Coconut's
  answers on 500/500 questions, with latents within 1.4e-4.

## Sample test explanations

The first test question's thoughts 1 to 3 and 5 (these are greedy decodes of 6 latents):

```
### Alex is a hilpus. Not shumpus. Thought 1: Alex is a sterpus. Lens: ### h . Every p us ? a mp ster z t er is Eva | HIP HT Hale LH Hai HY Zhao HERO Hernandez Zel Hyp Ic Hik HS Hansen
### Sally is a hilpus. Not sterpus. Thought 2: Every yumpus is a yerpus. Lens: ### h p . Every us ? is y imp ster n a er w | Hale Abedin Huma HRC Hin Heller Hansen LH Hyp Tillerson Hy Holder Hatch Hanson OH
### Sally is a hilpus. Not sterpus. Thought 3: Every yerpus is a hilpus. Lens: ### h p . Every us is ? y imp ster n a er w | Hale Abedin Huma Hernandez Hin Heller Hawai Hy HRC Hansen Tillerson Hatch Hanson Hyp
### Sally is a hilpus. Not sterpus. Thought 5: Done. Lens: ### h p . Every us is ? y ster imp n a w er | Hale Abedin Huma Hernandez Hin Heller Hawai HRC Hansen Hy Tillerson Hatch Hanson Hyp Holder
```

## Crashed attempt

Attempt 1 (01:38) exited immediately with `ModuleNotFoundError: No module named 'torch'`
(`log_attempt1_crash.txt`). It was a launcher bug: `export PATH=... && ... &` sent the whole
chain, including the `export`, into a background subshell, so the training ran with the system
Python. Attempt 2 set the environment as its own statement and ran clean.

## Command

See `command.sh`. Full output is in `log.txt`, and GPU memory/utilization per minute is in
`gpu_usage.log`.
