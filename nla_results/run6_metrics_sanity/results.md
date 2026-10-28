# NLA results: run 6, sanity check of the added metrics

A small run (300 train questions) to check the added metrics end to end before the long run.
`log_attempt1.txt` is the first attempt. It ran before the shuffled-latent splice ablation and
the species-only fidelity field were added, and is otherwise the same.

## Test results (best-val weights, 100 test questions = 600 latents)

| Metric | Value |
|---|---|
| FVE (standardized) / vs per-index mean / raw space | +0.301 / +0.271 / +0.345 |
| Per-thought FVE (k = 1..6) | +0.308 +0.277 +0.301 +0.305 +0.303 +0.305 |
| Teacher-text FVE (the AR reading the teacher text) | +0.401 |
| AV SFT NLL, real vs shuffled latent | 2.975 vs 3.687 (gap +0.712 nats/token) |
| Frontier recall, NLA vs probe (matched budget) | 0.124 vs 0.104 (precision 0.121 vs 0.102) |
| Fidelity: parse-fail / answer / species / step / index | 0.000 / 0.145 / 0.805 / 0.310 / 0.335 |
| Paraphrase FVE drop: canonical / lens shuffle / reworded | +0.000 (0% changed) / +0.012 / −0.003 |
| Splice: clean / all-AR / all-mean / all-shuffled accuracy | 0.99 / 0.99 / 0.99 / 0.99, recovered n/a |
| Question-only AR test FVE (100 steps) | +0.410 |
| Peak GPU memory (NLA training) | 5.11 GiB |

## Checks

- **Splice self-check**: the KV-cache loop reproduces Coconut. 0/100 answers differ, and the
  latent max abs diff is 9e-5.
- **Real vs shuffled latent**: the gap is ~0 at step 0 (22.31 vs 22.31) and +0.7 nats/token
  after the warm start, so the AV reads the latent.
- **Parsing**: no parse failures after the warm start. Canonical re-rendering changes 0% of
  texts, so the AV always writes the exact template and formatting carries nothing.

## Findings

- **Coconut's answer ignores its latents.** Replacing all six latents with the train mean or
  with *another question's* latents changes no answer. The log-prob of the clean answer moves by
  about 0.002 nats. A direct check of the first test question found attention from
  `<|end-latent|>` to the latent positions under 0.02 in every layer past layer 0. Swapping in
  another question's latents moves the final logits by at most 0.16. So "accuracy recovered" is
  undefined for this checkpoint (the ablation has no effect to recover). The splice metric still
  runs and reports this.
- **Species, not name.** The AV gets Coconut's answer *species* right 80% of the time but the
  full answer only 15%. The person's name is the part it misses.
- **Steps.** Gold-step match is 0 for thoughts 1 to 3 and high for later thoughts. Those are
  mostly `Done.`, which is easy.
- **The AR doesn't depend on the exact wording.** Rewording the template costs nothing.
  Shuffling the lens order costs ~0.015, so lens rank carries a little information.

## Command

See `command.sh`. Full output is in `log.txt`.
