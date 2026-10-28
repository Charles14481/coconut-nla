# Natural Language Autoencoders for Interpreting Coconut's Latent Reasoning

I think continuous CoT is a potentially powerful paradigm in the LLM capabilities space, but I think
there will need to be strong interpretability tools developed before it can be responsibly scaled.
I definitely enjoyed running the Coconut model for myself and trying to train an NLA on it.
I didn't get the results I hoped for, as the explanations mostly restate Coconut's answer. However, I think this was a good learning experience and potentially a useful negative result.

This repo trains a **natural language autoencoder (NLA)** on the latent thoughts of
[Coconut](https://arxiv.org/abs/2412.06769) (Chain of Continuous Thought). Coconut is a GPT-2
that reasons by feeding its last hidden state back in as the next input embedding, so its
"thoughts" are vectors, not text. The NLA tries to put those vectors into words:

- the **verbalizer** (AV, a copy of Coconut's GPT-2) reads one latent vector and writes a short
  text explanation of it;
- the **reconstructor** (AR, another copy plus a linear head) reads only that text and predicts
  the vector back.

If the reconstruction is good, the text could hint at what the latent tokens mean. The score is
**fraction of variance explained (FVE)** on held-out questions: 0 is no better than predicting the
mean latent, and 1 is a perfect reconstruction.

The NLA method (verbalizer, reconstructor, warm start, KL-penalized RL, FVE) is from Fraser-Taliente et al. Applying it to Coconut's latents was my idea; the code was written with Claude.

The repo is based on [facebookresearch/coconut](https://github.com/facebookresearch/coconut)
(MIT). All NLA work runs
on a pretrained ProsQA Coconut checkpoint,
[bmarti44/coconut-curriculum-checkpoints](https://huggingface.co/bmarti44/coconut-curriculum-checkpoints),
which answers 490/500 (98%) test questions correctly, similar to the accuracy reported in Meta's paper (97%).

## Results

Best run: [nla_results/run7_long](nla_results/run7_long/results.md), with 10,000
training questions (60,000 latents). It was scored once on 500 test questions (3,000 latents),
using the weights that did best on validation. It took 8 h 44 min on one RTX 2080 SUPER (8 GB).

### How FVE is measured

Each latent is a 768-dimensional vector. FVE compares the reconstructor's error with the error of
a model that ignores the text and always predicts the average training latent:

```
FVE = 1 − MSE(reconstruction, latent) / MSE(mean training latent, latent)
```

- **0** means the text tells the reconstructor nothing: it does no better than the mean.
- **1** means a perfect reconstruction.
- **Below 0** means it does worse than the mean, which an untrained model can.
- **+0.631** means the explanations remove 63% of the error the mean leaves.

Both errors are computed after **standardizing** each dimension (subtracting its training mean
and dividing by its training standard deviation). Without this, five GPT-2 "outlier" dimensions
with huge values account for ~87% of the raw error, and FVE would mostly measure those five.
Standardizing makes every dimension count equally. It's the stricter choice: FVE in the raw
space is higher (+0.743), because the outlier dimensions are easy to predict.

FVE only says how much of the vector the text pins down, not whether the text is a correct
account of the reasoning. The other rows in the table test that. For scale:

- A model that only knows which of the six thoughts it's looking at scores +0.04.
- A reconstructor that reads the whole question instead of an explanation scores +0.878. Each
  latent is a fixed function of its question, so that's a ceiling, not a bar the NLA should clear.

| Metric | Value |
|---|---|
| **Test FVE** | **+0.631** |
| FVE when the AR reads the teacher text itself (the warm start's ceiling) | +0.629 |
| FVE of a model that only knows which thought it is (1–6) | +0.04 |
| Question-only AR (reads the whole question instead of an explanation) | +0.878 |
| Verbalizer loss on teacher text, own latent vs another question's | 1.69 vs 9.46 nats/token |
| Explanation states Coconut's answer category | 93% |
| Explanation states the correct reasoning step (thoughts 1 / 2 / 3) | 7% / 2% / 10% |
| Reasoning-frontier recall, explanation vs a linear probe on the latent | 0.19 vs 0.31 |
| FVE lost when the lens words are shuffled / the template is reworded | 0.14 / 0.01 |

FVE on other runs (same test split):

| Run | Change | Test FVE |
|---|---|---|
| [run1_large](nla_results/run1_large/results.md) | teacher text = Coconut's answer only | +0.298 |
| [run2_small](nla_results/run2_small/results.md) | + rejected option, reasoning step, logit-lens tokens | +0.359 |
| [run3_moderate](nla_results/run3_moderate/results.md) | + z-scored logit lens, more data | +0.509 |
| [run4_long](nla_results/run4_long/results.md) | 10k questions, 9000 warm-start + 2000 RL steps | +0.630 |
| [run7_long](nla_results/run7_long/results.md) | same, rerun with the full metric suite | +0.631 |

Runs 5 and 6 were small checks of the added metrics (300 training questions, 100 test questions),
so they aren't in this table. Their write-ups are in [nla_results/](nla_results/).

Interpretation:

- **The NLA reaches its teacher's ceiling.** The verbalizer's own explanations reconstruct the
  latent as well as the teacher text it was trained to imitate.
- **The verbalizer reads the latent.** Given another question's latent, the correct explanation
  becomes far less likely.
- **The explanations are weakest on the early reasoning steps.** They name Coconut's answer
  almost every time but rarely the step a given early thought stands for, and a linear probe
  finds more of the reasoning frontier than the explanations mention.
- **They describe the model, not the truth.** On the questions Coconut gets wrong, the explanation
  still states Coconut's wrong answer 73% of the time (only 10 questions, though).
- **This Coconut checkpoint's answers don't depend on its latents.** Replacing all six latents with
  the mean latent, or with another question's latents, changes none of the 500 answers, and moves
  the answer's log-prob by about 0.002 nats. So splicing the reconstructions back in can't
  "recover" any accuracy: there is nothing to recover.

## How it works

1. **Collect latents.** Run frozen Coconut on each ProsQA question and keep its six latent
   thoughts. Standardize each dimension with train statistics (raw MSE is ~87% five GPT-2 outlier
   dimensions).
2. **Teacher text.** For each latent, build an explanation like
   `### Tom is a zhorpus. Not jompus. Thought 2: Every bompus is a wumpus. Lens: ...`, made of
   Coconut's answer, the option it rejected, the training step that thought replaced, and the
   latent's top logit-lens tokens (plain, and z-scored against the train latents).
3. **Warm start.** Train the verbalizer to write the teacher text from the latent alone
   (cross-entropy), and the reconstructor to recover the latent from the text (MSE). The
   verbalizer never sees the question.
4. **RL.** REINFORCE on the verbalizer with reward −MSE − 0.05 × KL to the post-warm-start
   verbalizer, 4 samples per latent, group-mean baseline.
5. **Evaluate.** Every validation pass reports FVE and the extra metrics. At the end, the best
   weights are scored once on the test split, then the splice test and the question-only
   baseline run.

## Running it

Needs one CUDA GPU with at least 8 GB of memory:

```bash
conda create --name coconut python=3.12
conda activate coconut
pip install -r requirements.txt
```

Run from the repo root. The checkpoint (~800 MB) downloads from the Hugging Face Hub on first use.

```bash
# Small run, ~45 min: test FVE ≈ +0.38
python nla.py --nla-train-examples 800 --nla-val-examples 32 --nla-test-examples 500 \
  --nla-warmup-steps 300 --nla-steps 150

# The best run, ~9 h (see nla_results/run7_long/command.sh)
python nla.py --nla-train-examples 10000 --nla-val-examples 64 --nla-test-examples 500 \
  --nla-warmup-steps 9000 --nla-steps 2000 --nla-evals-per-phase 10 --nla-question-steps 9000

python nla.py --help   # every hyperparameter; the defaults are the tuned settings
```

Each run in [nla_results/](nla_results/) has its `command.sh`, full `log.txt` and a `results.md`
write-up. Some logs predate a rename of the scripts and show their old names: `test_NLA.py` is
`nla.py` and `tsrigo.py` is `load_checkpoint.py`.

## Files

| File | What it is |
|---|---|
| [nla.py](nla.py) | The NLA: teacher text, warm start, RL, and all the metrics |
| [prosqa_inference.py](prosqa_inference.py) | Coconut's ProsQA accuracy, and its answer after 1–6 latents |
| [load_checkpoint.py](load_checkpoint.py) | Downloads a Coconut checkpoint and rebuilds the model |
| `coconut.py`, `dataset.py`, `run.py`, `utils.py`, `args/`, `preprocessing/` | Upstream Coconut (only `args/gsm_cot.yaml`'s `save_path` is changed) |

## Citations

- Coconut: Hao et al., [Training Large Language Models to Reason in a Continuous Latent
  Space](https://arxiv.org/abs/2412.06769), and the
  [official code](https://github.com/facebookresearch/coconut) this repo is based on.
- The ProsQA checkpoint: [bmarti44/coconut-curriculum-checkpoints](https://huggingface.co/bmarti44/coconut-curriculum-checkpoints).
- Fraser-Taliente, Kantamneni, Ong et al., [Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations](https://transformer-circuits.pub/2026/nla/)

## License

This repo is released under the [MIT License](LICENSE), the same license as upstream Coconut.

The copyright notice in [LICENSE](LICENSE) is Meta's and covers the upstream Coconut code.
