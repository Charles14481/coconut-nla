"""Train a natural language autoencoder (NLA) on Coconut's latent thoughts.

The activation verbalizer (AV, a copy of Coconut's GPT-2) writes a short text
explanation of one latent vector. The activation reconstructor (AR, another
copy plus a linear head) reads only that text and predicts the latent.
Training first imitates teacher explanations (the warm start), then runs
REINFORCE with the reconstruction error as the AV's reward. The score is the
fraction of variance explained (FVE) on held-out questions. Every validation
pass and the test pass also report explanation_metrics (whether the AV reads
the latent, frontier recall vs a linear probe, fidelity, paraphrase
robustness). The test pass adds splice_metrics: Coconut's accuracy with its
latents replaced by the NLA's reconstructions.
"""

import argparse
import collections
import copy
import dataclasses
import json
import random
import re
import time
from typing import Any

import torch
import torch.nn.functional as F

from prosqa_inference import make_inputs, normalize_answer
from load_checkpoint import load_model
from utils import set_seed

total_latent_thoughts: int = 6
# Splits are by question, so no question's latents appear in more than one split.
train_path: str = "data/prosqa_train.json"
val_path: str = "data/prosqa_valid.json"
test_path: str = "data/prosqa_test.json"
# Most tokens Coconut may write when answering, for the teacher text.
teacher_tokens: int = 16
# Rows decoded together during evaluation.
eval_chunk_size: int = 64
# Rows per question-only AR pass. Questions run to ~530 tokens, so training uses
# micro-batches of this many, and evaluation four times as many.
question_chunk_size: int = 4
# Rows per teacher-text NLL pass in the metrics: full-vocab logits at 64 rows are ~0.85 GB.
nll_chunk_size: int = 32
# Steps of the full-batch frontier probe fit.
probe_steps: int = 300
# A full teacher text; parse_explanation reads the AV's texts with it.
explanation_pattern = re.compile(
    r"(?P<answer>### .+?\.) Not (?P<rejected>\S+)\. Thought (?P<index>\d+): "
    r"(?P<step>.+?) Lens: (?P<lens>.*)",
    re.DOTALL,
)


def option(default: Any, help_text: str, **extra: Any) -> Any:
    """A config field that carries its command-line help text and argparse extras."""
    return dataclasses.field(default=default, metadata={"help": help_text, **extra})


@dataclasses.dataclass
class NLAConfig:
    """NLA hyperparameters. Every field is also a --nla-<field-name> command-line flag."""

    train_examples: int = option(8, "training questions (6 latents each)")
    val_examples: int = option(4, "validation questions, used to pick the best weights")
    test_examples: int = option(8, "test questions, scored once at the end")
    seed: int = option(0, "random seed")
    teacher: str = option(
        "full",
        "warm-start teacher text: 'full' is one text per thought (answer, rejected option, "
        "reasoning step, logit lens); 'answer' is Coconut's answer, shared by a question's thoughts",
        choices=["full", "answer"],
    )
    lens: str = option(
        "both",
        "logit lens in a full teacher text: 'plain' ranks tokens by logit; 'z' by logit z-scored "
        "against the train latents, which skips tokens every latent favors; 'both' lists both",
        choices=["plain", "z", "both"],
    )
    lens_tokens: int = option(15, "tokens listed per logit lens in a full teacher text")
    explanation_tokens: int = option(
        64, "most tokens in an explanation (full teacher texts with the defaults: at most 63)"
    )
    repeat_window: int = option(
        3,
        "forbid a token that appeared among the last N explanation tokens; stops greedy loops on "
        "rare lens tokens (0 allows repeats; 3 blocks 0.4%% of full teacher texts, 4 blocks 42%%; "
        "a huge N forbids any repeat)",
    )
    distinct_lens: bool = option(
        True,
        "once an explanation has written ' Lens', forbid repeating any token after it; a full "
        "teacher's lens lists never repeat one, and this stops greedy loops of any period there",
    )
    adapter_tokens: int = option(
        4, "extra AV soft-prompt tokens made from the standardized activation (0 turns them off)"
    )
    ar_pool_mean: bool = option(
        True, "AR head reads the mean hidden state over the explanation as well as the last token's"
    )
    ar_av_text: bool = option(
        True, "also train the AR on the AV's own greedy explanations, which it reads at evaluation"
    )
    warmup_steps: int = option(0, "supervised warm-start steps on teacher text")
    steps: int = option(25, "RL steps")
    batch_size: int = option(
        4, "activations per RL step; a warm-start step uses batch_size * group_size"
    )
    group_size: int = option(4, "RL samples per activation; their mean reward is the baseline")
    av_lr: float = option(1e-4, "AV learning rate in the warm start")
    ar_lr: float = option(1e-4, "AR learning rate in the warm start")
    head_lr: float = option(
        1e-4, "warm-start learning rate of the zero-initialized AR head and AV adapter"
    )
    rl_lr: float = option(1e-6, "AV learning rate in RL (1e-5 starts collapsing explanations)")
    rl_ar_lr: float = option(
        1e-5,
        "AR and head learning rate in RL, 0 to freeze them (at the warm-start rate the AR "
        "drifts on RL's few activations per step)",
    )
    kl_coef: float = option(0.05, "weight of the KL to the post-warm-start AV in the RL reward")
    evals_per_phase: int = option(5, "validation passes in each phase (warm start, RL)")
    question_steps: int = option(
        0,
        "after the NLA, train a question-only AR baseline for this many steps: a fresh AR that "
        "reads the question and thought index instead of an explanation (0 skips it)",
    )
    splice: bool = option(
        True,
        "at test, splice AR(AV(h)) back into Coconut in place of its latents and score its answers",
    )


@dataclasses.dataclass
class SplitData:
    """What the metrics need about one evaluation split. Rows are in collect_activations order."""

    items: list[dict[str, Any]]
    activations: torch.Tensor
    targets: torch.Tensor
    # Per question: Coconut's answer, and whether it matches the gold answer.
    answers: list[str]
    correct: list[bool]
    # Per row: the teacher text (None without a warm start), the frontier names, and every
    # name ranked by the frontier probe's score, best first.
    teacher: list[str] | None
    frontiers: list[set[str]]
    probe_ranking: list[list[str]]


def latent_activations(
    model: Any,
    tokenizer: Any,
    question: str,
    start_id: int,
    latent_id: int,
    end_id: int,
) -> torch.Tensor:
    """Return the frozen Coconut activations fed back into latent positions."""
    inputs = make_inputs(
        tokenizer,
        question,
        total_latent_thoughts,
        start_id,
        latent_id,
        end_id,
        next(model.parameters()).device,
    )
    position_ids = torch.arange(
        inputs["input_ids"].shape[1],
        device=inputs["input_ids"].device,
    ).unsqueeze(0)
    with torch.inference_mode():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["input_ids"],
            position_ids=position_ids,
        )
    positions = (inputs["input_ids"][0] == latent_id).nonzero(as_tuple=False).flatten()
    if not len(positions):
        raise RuntimeError("The activation target contains no latent-token positions.")
    return outputs.inputs_embeds[0, positions].detach()


def collect_activations(
    target_model: Any,
    tokenizer: Any,
    items: list[dict[str, str]],
    start_id: int,
    latent_id: int,
    end_id: int,
) -> torch.Tensor:
    """Return the latent activations of every question, stacked in question order."""
    return torch.cat(
        [
            latent_activations(
                target_model,
                tokenizer,
                item["question"],
                start_id,
                latent_id,
                end_id,
            )
            for item in items
        ]
    )


def coconut_answers(
    model: Any,
    tokenizer: Any,
    items: list[dict[str, str]],
    start_id: int,
    latent_id: int,
    end_id: int,
) -> list[str]:
    """Return what Coconut itself writes after each question's latents, one per question.

    The final ProsQA checkpoint answers ("### Tom is a lempus.") right after
    <|end-latent|> however many latents it was given, so one greedy decode per
    question, after the full latent chain, serves all of that question's thoughts.
    """
    lm = model.base_causallm
    eos_id = tokenizer.eos_token_id
    answers = []
    for item in items:
        inputs = make_inputs(
            tokenizer,
            item["question"],
            total_latent_thoughts,
            start_id,
            latent_id,
            end_id,
            next(model.parameters()).device,
        )
        position_ids = torch.arange(inputs["input_ids"].shape[1], device=inputs["input_ids"].device)
        with torch.inference_mode():
            embeds = model(
                **inputs, labels=inputs["input_ids"], position_ids=position_ids.unsqueeze(0)
            ).inputs_embeds
            outputs = lm(inputs_embeds=embeds, use_cache=True)
            generated = []
            for _ in range(teacher_tokens):
                token = outputs.logits[:, -1].argmax(dim=-1)
                if token.item() == eos_id:
                    break
                generated.append(token.item())
                outputs = lm(
                    input_ids=token.unsqueeze(1),
                    past_key_values=outputs.past_key_values,
                    use_cache=True,
                )
        answers.append(tokenizer.decode(generated))
    return answers


def lens_logits(model: Any, activations: torch.Tensor, tokenizer: Any) -> torch.Tensor:
    """Each latent's next-token logits under the LM head (the logit lens), text tokens only.

    A latent is a last-layer hidden state, so the LM head reads it exactly as
    it would read the state before Coconut decodes a token. EOS and the added
    latent tokens are dropped: they decode to nothing in an explanation.
    """
    with torch.inference_mode():
        return model.base_causallm.lm_head(activations)[:, : tokenizer.eos_token_id]


def lens_logit_stats(
    model: Any, activations: torch.Tensor, tokenizer: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token mean and std of the logit lens over `activations`, accumulated in chunks.

    All logits at once would be rows x 50k floats, gigabytes for a few thousand questions.
    """
    total = squares = None
    for start in range(0, len(activations), eval_chunk_size):
        logits = lens_logits(model, activations[start : start + eval_chunk_size], tokenizer).double()
        chunk_total, chunk_squares = logits.sum(dim=0), (logits**2).sum(dim=0)
        total = chunk_total if total is None else total + chunk_total
        squares = chunk_squares if squares is None else squares + chunk_squares
    mean = total / len(activations)
    std = (squares / len(activations) - mean**2).clamp_min(1e-6).sqrt()
    return mean.float(), std.float()


def lens_tokens(
    model: Any,
    activations: torch.Tensor,
    tokenizer: Any,
    count: int,
    logit_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
    exclude: list[list[str]] | None = None,
) -> list[list[str]]:
    """Return each latent's top `count` logit-lens tokens as words, skipping each row's `exclude`.

    With `logit_stats` (per-token train mean and std), tokens are ranked by
    z-scored logit instead. The plain ranking is dominated by tokens every
    latent favors ("### Every . p us"); the z-scored one lists what is unusual
    about this latent. GPT-2 ties its LM head to its input embeddings, so these
    are the tokens whose embeddings point most unusually along the latent.
    """
    # In chunks: all logits at once would be rows x 50k floats, 12 GB for 10k questions.
    top = []
    for start in range(0, len(activations), eval_chunk_size):
        logits = lens_logits(model, activations[start : start + eval_chunk_size], tokenizer)
        if logit_stats is not None:
            logits = (logits - logit_stats[0]) / logit_stats[1]
        # Spare candidates, for the tokens skipped below.
        top.extend(logits.topk(count * 4, dim=-1).indices.tolist())
    lists = []
    for index, row in enumerate(top):
        skip = set(exclude[index]) if exclude else set()
        words: list[str] = []
        for token in row:
            word = tokenizer.decode([token]).strip()
            # Byte fragments decode to U+FFFD and whitespace tokens to nothing; neither reads as text.
            # " Z" and "Z" strip to the same word, so it is listed once. A word must also stay one
            # token after a space ("zn" re-encodes as " z" "n"), so each costs one token and the
            # list never repeats a token. ":" and "|" delimit the lens in a teacher text.
            if (
                word
                and "\ufffd" not in word
                and word not in words
                and word not in skip
                and word not in (":", "|")
                and len(tokenizer.encode(" " + word)) == 1
            ):
                words.append(word)
            if len(words) == count:
                break
        lists.append(words)
    return lists


def teacher_explanations(
    model: Any,
    tokenizer: Any,
    items: list[dict[str, str]],
    activations: torch.Tensor,
    answers: list[str],
    config: NLAConfig,
    logit_stats: tuple[torch.Tensor, torch.Tensor],
) -> list[str]:
    """Return one teacher explanation per activation, in collect_activations order.

    The "answer" style is Coconut's own answer, shared by all of a question's
    thoughts. The "full" style adds what tells the thoughts apart: the option
    Coconut rejected, the reasoning step thought k replaced during curriculum
    training (c_thought = 1, so thought k stands in for step k), and the
    latent's top logit-lens tokens (see lens_tokens; `logit_stats` are the
    train latents' per-token logit mean and std). Linear bag-of-words probes put
    the answer's ceiling at FVE ~0.31, the full text's at ~0.43 with the plain
    lens and ~0.48 with 20 z-scored lens tokens. `answers` are coconut_answers'.
    """
    if config.teacher == "answer":
        return [answer for answer in answers for _ in range(total_latent_thoughts)]
    lenses = []
    if config.lens in ("plain", "both"):
        lenses.append(lens_tokens(model, activations, tokenizer, config.lens_tokens))
    if config.lens in ("z", "both"):
        # No word in both lists, so no token repeats anywhere after "Lens:" (see distinct_lens).
        exclude = lenses[0] if lenses else None
        lenses.append(
            lens_tokens(model, activations, tokenizer, config.lens_tokens, logit_stats, exclude)
        )
    lens = [" | ".join(" ".join(words) for words in parts) for parts in zip(*lenses)]
    explanations = []
    for item, answer in zip(items, answers):
        # The question ends "Is Tom a jompus or zhorpus?"; the rejected option is the one not answered.
        options = item["question"].rsplit(" a ", 1)[-1].rstrip("?").split(" or ")
        answered = answer.strip().rstrip(".").split()
        rejected = next((o for o in options if o not in answered), options[-1])
        for k in range(total_latent_thoughts):
            step = item["steps"][k] if k < len(item["steps"]) else "Done."
            explanations.append(
                f"{answer.strip()} Not {rejected}. Thought {k + 1}: {step} "
                f"Lens: {lens[len(explanations)]}"
            )
    return explanations


class ActivationAdapter(torch.nn.Module):
    """Turn a standardized activation into extra soft-prompt tokens for the AV.

    The raw activation is dominated by a few huge GPT-2 outlier dimensions, so
    what tells two latents apart is nearly invisible in it. Standardizing first
    makes every dimension count. The projection starts at zero, so these tokens
    start out carrying no information about the activation.
    """

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, tokens: int) -> None:
        super().__init__()
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone())
        self.tokens = tokens
        self.proj = torch.nn.Linear(len(mean), tokens * len(mean), device=mean.device)
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.zero_()

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        standardized = (activations - self.mean) / self.std
        return self.proj(standardized).view(len(activations), self.tokens, -1)


def prompt_prefix(model: Any, activations: torch.Tensor, tokenizer: Any) -> torch.Tensor:
    """Embed each activation as a soft-prompt token followed by "Activation:".

    An AV with an `nla_adapter` also gets the adapter's tokens after the raw activation.
    """
    prompt_ids = tokenizer.encode("Activation:", add_special_tokens=False)
    prompt = torch.tensor(prompt_ids, device=activations.device).expand(len(activations), -1)
    parts = [activations.unsqueeze(1)]
    if hasattr(model, "nla_adapter"):
        parts.append(model.nla_adapter(activations))
    return torch.cat(parts + [model.get_input_embeddings()(prompt)], dim=1)


@torch.no_grad()
def sample_explanations(
    model: Any,
    activations: torch.Tensor,
    tokenizer: Any,
    max_new_tokens: int,
    sample: bool = True,
    repeat_window: int = 0,
    lens_marker: int | None = None,
) -> tuple[list[str], list[list[int]]]:
    """Generate one explanation per activation row, decoding all rows together.

    A token among a row's last `repeat_window` tokens can't be drawn again,
    nor, once a row has written `lens_marker`, any token written after it.
    Returns the texts and each row's token ids (including the EOS token if it
    was drawn). Sampled tokens are scored afterwards by sequence_log_probs.
    """
    rows = len(activations)
    outputs = model(inputs_embeds=prompt_prefix(model, activations, tokenizer), use_cache=True)
    eos_id = tokenizer.eos_token_id
    done = torch.zeros(rows, dtype=torch.bool, device=activations.device)
    token_ids: list[list[int]] = [[] for _ in range(rows)]
    history: list[torch.Tensor] = []
    in_lens = torch.zeros_like(done)
    lens_seen = torch.zeros_like(outputs.logits[:, -1], dtype=torch.bool)

    for _ in range(max_new_tokens):
        logits = outputs.logits[:, -1]
        recent = history[-repeat_window:] if repeat_window else []
        if recent:
            logits = logits.scatter(1, torch.stack(recent, dim=1), float("-inf"))
        if lens_marker is not None:
            logits = logits.masked_fill(lens_seen, float("-inf"))
        tokens = (
            torch.distributions.Categorical(logits=logits).sample()
            if sample
            else logits.argmax(dim=-1)
        )
        for row, token_id in enumerate(tokens.tolist()):
            if not done[row]:
                token_ids[row].append(token_id)
        history.append(tokens)
        if lens_marker is not None:
            lens_seen[in_lens, tokens[in_lens]] = True
            in_lens = in_lens | (tokens == lens_marker)
        done = done | (tokens == eos_id)
        if done.all():
            break
        outputs = model(
            input_ids=tokens.unsqueeze(1),
            past_key_values=outputs.past_key_values,
            use_cache=True,
        )

    # Fall back to the EOS text so the reconstructor never receives an empty input.
    explanations = [
        tokenizer.decode([t for t in ids if t != eos_id], skip_special_tokens=True)
        or tokenizer.eos_token
        for ids in token_ids
    ]
    return explanations, token_ids


def sequence_log_probs(
    model: Any,
    activations: torch.Tensor,
    tokenizer: Any,
    sampled_ids: list[list[int]],
    repeat_window: int = 0,
    lens_marker: int | None = None,
) -> torch.Tensor:
    """Each row's summed log-prob of its sampled tokens, scored in one teacher-forced pass.

    Scoring after sampling, rather than keeping the sampling loop's graph,
    stores activations for one pass instead of one per generated token.
    Tokens are scored under the same repeat rules they were sampled under, so
    the AV's and the reference's log-probs are comparable.
    """
    device = activations.device
    length = max(len(ids) for ids in sampled_ids)
    padded = torch.tensor(
        [ids + [tokenizer.eos_token_id] * (length - len(ids)) for ids in sampled_ids], device=device
    )
    valid = torch.tensor(
        [[True] * len(ids) + [False] * (length - len(ids)) for ids in sampled_ids], device=device
    )
    prefix = prompt_prefix(model, activations, tokenizer)
    embeds = torch.cat([prefix, model.get_input_embeddings()(padded[:, :-1])], dim=1)
    # The logit at the last prefix position predicts the first sampled token.
    logits = model(inputs_embeds=embeds).logits[:, prefix.shape[1] - 1 :]
    blocked = torch.zeros_like(logits, dtype=torch.bool)
    # Position p may not repeat the tokens at p - repeat_window .. p - 1.
    for offset in range(1, min(repeat_window, length - 1) + 1):
        blocked[:, offset:].scatter_(2, padded[:, :-offset].unsqueeze(2), True)
    if lens_marker is not None:
        # Nor any token between the row's lens marker and p.
        rows = torch.arange(len(sampled_ids), device=device)
        lens_seen = torch.zeros_like(blocked[:, 0])
        in_lens = torch.zeros(len(sampled_ids), dtype=torch.bool, device=device)
        for position in range(1, length):
            previous = padded[:, position - 1]
            lens_seen[rows[in_lens], previous[in_lens]] = True
            in_lens = in_lens | (previous == lens_marker)
            blocked[:, position] |= lens_seen
    logits = logits.masked_fill(blocked, float("-inf"))
    token_log_probs = torch.log_softmax(logits, dim=-1).gather(2, padded.unsqueeze(2)).squeeze(2)
    # where, not multiply: a padding EOS may be masked to -inf, and -inf * 0 is nan.
    return torch.where(valid, token_log_probs, 0.0).sum(dim=1)


def explanation_nll(
    model: Any, activations: torch.Tensor, tokenizer: Any, explanations: list[str]
) -> torch.Tensor:
    """Mean per-token negative log-likelihood of each explanation, then EOS, given its activation."""
    eos_id = tokenizer.eos_token_id
    encoded = [tokenizer.encode(text, add_special_tokens=False) + [eos_id] for text in explanations]
    length = max(len(ids) for ids in encoded)
    device = activations.device
    input_ids = torch.tensor([ids + [eos_id] * (length - len(ids)) for ids in encoded], device=device)
    labels = torch.tensor([ids + [-100] * (length - len(ids)) for ids in encoded], device=device)
    prefix = prompt_prefix(model, activations, tokenizer)
    embeds = torch.cat([prefix, model.get_input_embeddings()(input_ids[:, :-1])], dim=1)
    # The logit at the last prefix position predicts the first explanation token.
    logits = model(inputs_embeds=embeds).logits[:, prefix.shape[1] - 1 :]
    return F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)


def make_head(dimension: int, device: torch.device, pool_mean: bool) -> torch.nn.Linear:
    """Map AR's final hidden state to a standardized activation, starting at zero.

    Zero is the standardized train mean, so at step 0 the NLA scores exactly
    the predict-the-mean baseline, and any improvement must come from the explanation.
    With `pool_mean`, the head also reads the mean hidden state over the explanation.
    """
    head = torch.nn.Linear(dimension * (2 if pool_mean else 1), dimension, device=device)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    return head


def reconstruct_activations(
    model: Any, head: torch.nn.Linear, tokenizer: Any, explanations: list[str]
) -> torch.Tensor:
    """Encode each explanation and map its last token's final hidden state to a reconstruction.

    A head twice as wide as the hidden state also gets the mean over the explanation's tokens.
    """
    device = next(model.parameters()).device
    encoded = [tokenizer.encode(text, add_special_tokens=True) for text in explanations]
    length = max(len(ids) for ids in encoded)
    # Right padding: under causal attention, pads never affect the last real token.
    input_ids = torch.tensor(
        [ids + [tokenizer.eos_token_id] * (length - len(ids)) for ids in encoded], device=device
    )
    attention_mask = torch.tensor(
        [[1] * len(ids) + [0] * (length - len(ids)) for ids in encoded], device=device
    )
    # The transformer alone: the LM head's logits over 50k tokens would only be thrown away.
    hidden = model.transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    last = torch.tensor([len(ids) - 1 for ids in encoded], device=device)
    features = hidden[torch.arange(len(encoded), device=device), last]
    if head.in_features > hidden.shape[-1]:
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        features = torch.cat([features, (hidden * mask).sum(dim=1) / mask.sum(dim=1)], dim=-1)
    return head(features)


def decoding_rules(tokenizer: Any, config: NLAConfig) -> dict[str, Any]:
    """The repeat rules for sample_explanations and sequence_log_probs, from the config."""
    marker = tokenizer.encode(" Lens")[0] if config.distinct_lens else None
    return {"repeat_window": config.repeat_window, "lens_marker": marker}


def evaluate_nla(
    av: Any,
    ar: Any,
    head: torch.nn.Linear,
    tokenizer: Any,
    activations: torch.Tensor,
    targets: torch.Tensor,
    config: NLAConfig,
) -> tuple[float, list[str], torch.Tensor]:
    """Return the mean greedy reconstruction MSE, the greedy explanations and the reconstructions."""
    losses = []
    explanations = []
    reconstruction_chunks = []
    with torch.inference_mode():
        for start in range(0, len(activations), eval_chunk_size):
            chunk = slice(start, start + eval_chunk_size)
            texts, _ = sample_explanations(
                av,
                activations[chunk],
                tokenizer,
                config.explanation_tokens,
                sample=False,
                **decoding_rules(tokenizer, config),
            )
            reconstructions = reconstruct_activations(ar, head, tokenizer, texts)
            losses.extend(((reconstructions - targets[chunk]) ** 2).mean(dim=-1).tolist())
            explanations.extend(texts)
            reconstruction_chunks.append(reconstructions)
    return sum(losses) / len(losses), explanations, torch.cat(reconstruction_chunks)


def mean_baseline_loss(train_targets: torch.Tensor, targets: torch.Tensor) -> float:
    """MSE of ignoring the explanation and always predicting the mean train target."""
    return ((targets - train_targets.mean(dim=0)) ** 2).mean().item()


def index_baseline_loss(train_targets: torch.Tensor, targets: torch.Tensor) -> float:
    """MSE of predicting the mean train target for each latent thought's index.

    An NLA that only learns which thought it is looking at reaches this.
    """
    index_means = torch.stack(
        [train_targets[i::total_latent_thoughts].mean(dim=0) for i in range(total_latent_thoughts)]
    )
    predictions = index_means.repeat(len(targets) // total_latent_thoughts, 1)
    return ((targets - predictions) ** 2).mean().item()


def raw_space_fve(
    reconstructions: torch.Tensor,
    targets: torch.Tensor,
    train_targets: torch.Tensor,
    std: torch.Tensor,
) -> float:
    """FVE of standardized reconstructions, measured back in the raw activation space.

    A standardized error times its dimension's std is the raw error. The
    baseline is still the train mean vector, so this differs from the
    standardized FVE only in weighting each dimension by its raw variance,
    which puts ~87% of the weight on five GPT-2 outlier dimensions.
    """
    weights = std**2
    error = (((reconstructions - targets) ** 2) * weights).sum()
    baseline = (((targets - train_targets.mean(dim=0)) ** 2) * weights).sum()
    return 1 - (error / baseline).item()


def question_texts(items: list[dict[str, str]]) -> list[str]:
    """One question-only AR input per latent, in collect_activations order."""
    return [
        f"{item['question']}\nThought {k + 1}:"
        for item in items
        for k in range(total_latent_thoughts)
    ]


def question_baseline(
    base_model: Any,
    tokenizer: Any,
    texts: tuple[list[str], list[str], list[str]],
    targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    config: NLAConfig,
    device: torch.device,
) -> tuple[float, float]:
    """Train an AR that reads the question and thought index, and return its best val and test MSE.

    Coconut is frozen and decodes greedily, so each latent is a fixed function of
    its question and index. This AR has the same architecture, start weights,
    learning rates and batch size as the NLA's, but reads the question where the
    NLA's reads an explanation. `texts` and `targets` are (train, val, test).
    """
    (train_texts, val_texts, test_texts), (train_targets, val_targets, test_targets) = texts, targets
    ar = copy.deepcopy(base_model).eval().to(device)
    head = make_head(train_targets.shape[1], device, config.ar_pool_mean)
    optimizer = torch.optim.AdamW(
        [
            {"params": ar.parameters(), "lr": config.ar_lr},
            {"params": head.parameters(), "lr": config.head_lr},
        ]
    )

    def score(split_texts: list[str], split_targets: torch.Tensor) -> float:
        chunk_size = question_chunk_size * 4
        losses = []
        with torch.inference_mode():
            for start in range(0, len(split_texts), chunk_size):
                chunk = slice(start, start + chunk_size)
                reconstructions = reconstruct_activations(ar, head, tokenizer, split_texts[chunk])
                losses.extend(((reconstructions - split_targets[chunk]) ** 2).mean(dim=-1).tolist())
        return sum(losses) / len(losses)

    start_time = time.time()
    val_baseline = mean_baseline_loss(train_targets, val_targets)
    best_val = score(val_texts, val_targets)
    best_weights = (cpu_state(ar), cpu_state(head))
    order: list[int] = []
    eval_every = max(1, config.question_steps // config.evals_per_phase)
    for step in range(config.question_steps):
        rows = draw(order, len(train_texts), config.batch_size * config.group_size)
        # Micro-batches, accumulating the batch-mean gradient: whole questions are long.
        loss = 0.0
        for start in range(0, len(rows), question_chunk_size):
            chunk = rows[start : start + question_chunk_size]
            reconstructions = reconstruct_activations(
                ar, head, tokenizer, [train_texts[row] for row in chunk]
            )
            chunk_loss = ((reconstructions - train_targets[chunk]) ** 2).mean() * len(chunk) / len(rows)
            chunk_loss.backward()
            loss += chunk_loss.item()
        optimizer.step()
        optimizer.zero_grad()
        if (step + 1) % eval_every == 0:
            val_loss = score(val_texts, val_targets)
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                best_weights = (cpu_state(ar), cpu_state(head))
            print(
                f"[{time.time() - start_time:.0f}s] Question-only AR step "
                f"{step + 1}/{config.question_steps}: train_loss={loss:.6f}, "
                f"val_loss={val_loss:.6f} (FVE {1 - val_loss / val_baseline:+.3f})"
                f"{' *' if improved else ''}"
            )
    for model, weights in zip((ar, head), best_weights):
        model.load_state_dict(weights)
    return best_val, score(test_texts, test_targets)


def frontier_names(items: list[dict[str, Any]]) -> list[set[str]]:
    """Per latent row, the names at BFS distance exactly k from the question's root, for thought k.

    Thought k stands in for reasoning step k (c_thought = 1), and step k reaches depth k.
    Past the deepest node the set is empty.
    """
    frontiers = []
    for item in items:
        children = collections.defaultdict(list)
        for parent, child in item["edges"]:
            children[parent].append(child)
        depth = {item["root"]: 0}
        queue = [item["root"]]
        for node in queue:
            for child in children[node]:
                if child not in depth:
                    depth[child] = depth[node] + 1
                    queue.append(child)
        for k in range(1, total_latent_thoughts + 1):
            frontiers.append({item["idx_to_symbol"][n] for n, d in depth.items() if d == k})
    return frontiers


def train_frontier_probe(
    targets: torch.Tensor, frontiers: list[set[str]], vocab: list[str]
) -> torch.nn.Linear:
    """Fit a linear probe from a standardized latent to each name's frontier membership.

    The baseline for the NLA's frontier recall: what a linear readout of the
    latent itself recovers. Full-batch AdamW on binary cross-entropy.
    """
    index = {name: i for i, name in enumerate(vocab)}
    labels = torch.zeros(len(targets), len(vocab), device=targets.device)
    for row, names in enumerate(frontiers):
        for name in names & index.keys():
            labels[row, index[name]] = 1
    probe = torch.nn.Linear(targets.shape[1], len(vocab), device=targets.device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=1e-2)
    for _ in range(probe_steps):
        F.binary_cross_entropy_with_logits(probe(targets), labels).backward()
        optimizer.step()
        optimizer.zero_grad()
    return probe


def probe_rankings(probe: torch.nn.Linear, targets: torch.Tensor, vocab: list[str]) -> list[list[str]]:
    """Every vocab name per row, ordered by the probe's score, best first."""
    with torch.no_grad():
        order = probe(targets).argsort(dim=-1, descending=True).tolist()
    return [[vocab[i] for i in row] for row in order]


def frontier_recall(
    explanations: list[str], split: SplitData, vocab: set[str]
) -> dict[str, float]:
    """Frontier recall and precision of the names each explanation mentions, vs the probe's.

    The probe gets the same budget as the explanation: its top m names, where m
    is how many distinct names the explanation mentions. Rows with an empty
    frontier (thoughts past the deepest node) are skipped.
    """
    hits = probe_hits = mentioned_total = frontier_total = 0
    for text, frontier, ranking in zip(explanations, split.frontiers, split.probe_ranking):
        if not frontier:
            continue
        mentioned = set(re.findall(r"[A-Za-z]+", text)) & vocab
        hits += len(mentioned & frontier)
        probe_hits += len(set(ranking[: len(mentioned)]) & frontier)
        mentioned_total += len(mentioned)
        frontier_total += len(frontier)
    return {
        "nla_recall": hits / max(frontier_total, 1),
        "nla_precision": hits / max(mentioned_total, 1),
        "probe_recall": probe_hits / max(frontier_total, 1),
        "probe_precision": probe_hits / max(mentioned_total, 1),
    }


def parse_explanation(text: str) -> dict[str, Any] | None:
    """Split a teacher-format explanation into its fields, or None if it isn't one.

    The lens is a list of word lists, one per lens ("plain | z" for lens=both).
    """
    match = explanation_pattern.fullmatch(" ".join(text.split()))
    if match is None:
        return None
    fields: dict[str, Any] = match.groupdict()
    fields["lens"] = [part.split() for part in fields["lens"].split(" | ")]
    return fields


def paraphrase(texts: list[str], tier: str, seed: int) -> list[str]:
    """Rewrite each parseable explanation without changing its content; others pass unchanged.

    - canonical: the same template, re-rendered from the parsed fields, so
      anything carried by spacing or formatting is lost.
    - lens_shuffle: the lens words in random order within each list, which
      also loses their rank.
    - reworded: other words for the same template. The AR never saw this
      wording, so a drop also measures how brittle it is, not only hidden content.
    """
    rng = random.Random(seed)
    rewritten = []
    for text in texts:
        fields = parse_explanation(text)
        if fields is None:
            rewritten.append(text)
            continue
        lens = [list(words) for words in fields["lens"]]
        if tier == "lens_shuffle":
            for words in lens:
                rng.shuffle(words)
        lens_text = " | ".join(" ".join(words) for words in lens)
        if tier == "reworded":
            step = re.sub(r"^Every (\S+) is a ", r"Each \1 is a ", fields["step"])
            step = "Finished." if step == "Done." else step
            rewritten.append(
                f"Answer: {fields['answer'].removeprefix('### ')} Rejected: {fields['rejected']}. "
                f"Step {fields['index']}: {step} Top tokens: {lens_text}"
            )
        else:
            rewritten.append(
                f"{fields['answer']} Not {fields['rejected']}. Thought {fields['index']}: "
                f"{fields['step']} Lens: {lens_text}"
            )
    return rewritten


def fidelity(explanations: list[str], split: SplitData) -> dict[str, Any]:
    """How often explanations state Coconut's answer, the gold step k and the right thought index.

    An explanation that doesn't parse counts as wrong on every field. Rates are
    reported overall, per thought index, and split by whether Coconut answered
    the question correctly (None for an empty group).
    """
    rows = []
    for row, text in enumerate(explanations):
        question, k = divmod(row, total_latent_thoughts)
        item = split.items[question]
        fields = parse_explanation(text)
        gold_step = item["steps"][k] if k < len(item["steps"]) else "Done."
        rows.append(
            {
                "k": k,
                "coconut_correct": split.correct[question],
                "parsed": fields is not None,
                "answer": fields is not None
                and " ".join(fields["answer"].split()) == " ".join(split.answers[question].split()),
                # The category alone: the person's name is in the question, not computed.
                "species": fields is not None
                and fields["answer"].rstrip(".").split()[-1:]
                == split.answers[question].strip().rstrip(".").split()[-1:],
                "step": fields is not None and fields["step"] == gold_step,
                "index": fields is not None and fields["index"] == str(k + 1),
            }
        )

    def rate(field: str, keep: Any = lambda row: True) -> float | None:
        kept = [row[field] for row in rows if keep(row)]
        return sum(kept) / len(kept) if kept else None

    result: dict[str, Any] = {"parse_fail": 1 - (rate("parsed") or 0.0)}
    for field in ("answer", "species", "step", "index"):
        result[field] = rate(field)
        result[f"{field}_per_k"] = [
            rate(field, lambda row, k=k: row["k"] == k) for k in range(total_latent_thoughts)
        ]
        for correct in (True, False):
            result[f"{field}_coconut_{'right' if correct else 'wrong'}"] = rate(
                field, lambda row, correct=correct: row["coconut_correct"] == correct
            )
    result["coconut_right_questions"] = sum(split.correct)
    result["coconut_wrong_questions"] = len(split.correct) - sum(split.correct)
    return result


def per_thought_fve(
    reconstructions: torch.Tensor, targets: torch.Tensor, train_targets: torch.Tensor
) -> list[float]:
    """FVE among the rows of each thought index, still against the overall train mean."""
    mean = train_targets.mean(dim=0)
    return [
        1
        - (
            ((reconstructions[k::total_latent_thoughts] - targets[k::total_latent_thoughts]) ** 2).mean()
            / ((targets[k::total_latent_thoughts] - mean) ** 2).mean()
        ).item()
        for k in range(total_latent_thoughts)
    ]


def reconstruction_loss(
    ar: Any, head: torch.nn.Linear, tokenizer: Any, texts: list[str], targets: torch.Tensor
) -> float:
    """Mean MSE of the AR's reconstructions from given texts, in chunks."""
    losses = []
    with torch.inference_mode():
        for start in range(0, len(texts), eval_chunk_size):
            chunk = slice(start, start + eval_chunk_size)
            reconstructions = reconstruct_activations(ar, head, tokenizer, texts[chunk])
            losses.extend(((reconstructions - targets[chunk]) ** 2).mean(dim=-1).tolist())
    return sum(losses) / len(losses)


def sft_nll(av: Any, activations: torch.Tensor, tokenizer: Any, texts: list[str]) -> float:
    """The AV's mean per-token NLL of `texts` given `activations`, in chunks."""
    total = 0.0
    with torch.inference_mode():
        for start in range(0, len(texts), nll_chunk_size):
            chunk = slice(start, start + nll_chunk_size)
            nll = explanation_nll(av, activations[chunk], tokenizer, texts[chunk])
            total += nll.item() * len(texts[chunk])
    return total / len(texts)


def explanation_metrics(
    av: Any,
    ar: Any,
    head: torch.nn.Linear,
    tokenizer: Any,
    split: SplitData,
    explanations: list[str],
    reconstructions: torch.Tensor,
    train_targets: torch.Tensor,
    std: torch.Tensor,
    vocab: set[str],
    seed: int,
) -> str:
    """Report the AV's and AR's metrics beyond FVE on one split, as indented log lines.

    - raw FVE: see raw_space_fve; per-thought FVE: see per_thought_fve.
    - SFT NLL: the AV's teacher-text NLL with each row's own latent, and with
      the latent of the same thought in the next question. No gap means the AV
      ignores the latent.
    - teacher-text FVE: the AR reading the teacher text, the warm start's ceiling.
    - frontier: see frontier_recall. fidelity: see fidelity.
    - paraphrase: FVE with the AR reading each paraphrase tier (see paraphrase),
      and its drop from the unparaphrased FVE.
    """
    baseline = mean_baseline_loss(train_targets, split.targets)
    fve = 1 - ((reconstructions - split.targets) ** 2).mean().item() / baseline
    per_thought = " ".join(
        f"{value:+.3f}" for value in per_thought_fve(reconstructions, split.targets, train_targets)
    )
    lines = [
        f"  FVE raw space {raw_space_fve(reconstructions, split.targets, train_targets, std):+.3f}, "
        f"per thought [{per_thought}]"
    ]
    if split.teacher is not None:
        real = sft_nll(av, split.activations, tokenizer, split.teacher)
        shuffled_activations = split.activations.roll(total_latent_thoughts, dims=0)
        shuffled = sft_nll(av, shuffled_activations, tokenizer, split.teacher)
        teacher_fve = 1 - reconstruction_loss(ar, head, tokenizer, split.teacher, split.targets) / baseline
        lines.append(
            f"  SFT NLL real {real:.3f}, shuffled {shuffled:.3f} (gap {shuffled - real:+.3f}); "
            f"teacher-text FVE {teacher_fve:+.3f}"
        )
    frontier = frontier_recall(explanations, split, vocab)
    lines.append(
        f"  frontier recall NLA {frontier['nla_recall']:.3f} vs probe {frontier['probe_recall']:.3f}, "
        f"precision NLA {frontier['nla_precision']:.3f} vs probe {frontier['probe_precision']:.3f}"
    )

    def rates(values: list[float | None]) -> str:
        return " ".join("n/a" if value is None else f"{value:.2f}" for value in values)

    def one(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    faithful = fidelity(explanations, split)
    lines.append(
        f"  fidelity: parse-fail {faithful['parse_fail']:.3f}; answer {one(faithful['answer'])}, "
        f"species {one(faithful['species'])}, step {one(faithful['step'])}, "
        f"index {one(faithful['index'])}; per k species [{rates(faithful['species_per_k'])}] "
        f"step [{rates(faithful['step_per_k'])}]; "
        f"Coconut right ({faithful['coconut_right_questions']} q) species "
        f"{one(faithful['species_coconut_right'])} step {one(faithful['step_coconut_right'])}, "
        f"Coconut wrong ({faithful['coconut_wrong_questions']} q) species "
        f"{one(faithful['species_coconut_wrong'])} step {one(faithful['step_coconut_wrong'])}"
    )
    tiers = []
    for tier in ("canonical", "lens_shuffle", "reworded"):
        texts = paraphrase(explanations, tier, seed)
        changed = sum(a != b for a, b in zip(texts, explanations)) / len(texts)
        tier_fve = 1 - reconstruction_loss(ar, head, tokenizer, texts, split.targets) / baseline
        tiers.append(f"{tier} {tier_fve:+.3f} (drop {fve - tier_fve:+.3f}, {changed:.0%} changed)")
    lines.append("  paraphrase FVE: " + ", ".join(tiers))
    return "\n".join(lines)


def spliced_answers(
    lm: Any,
    tokenizer: Any,
    question: str,
    overrides: list[dict[int, torch.Tensor]],
    ids: tuple[int, int, int],
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Coconut's answer to one question once per variant, with some latents replaced.

    Variant v puts overrides[v][k] in place of latent k (0-based), and every
    later latent is recomputed from it as Coconut computes them: the last-layer
    hidden state at the position before. The variants share the question's KV
    cache and decode together. coconut.py has no hook for this, and a variant
    with no overrides reproduces it, so overrides[0] should be empty.

    Returns the answers, each variant's latents (variant x thought x hidden),
    and each variant's summed log-prob of variant 0's answer and EOS, which
    moves even when the greedy answer doesn't.
    """
    start_id, _, end_id = ids
    device = next(lm.parameters()).device
    rows = len(overrides)
    prompt = tokenizer.encode(question + "\n", add_special_tokens=True) + [start_id]
    eos_id = tokenizer.eos_token_id
    with torch.inference_mode():
        outputs = lm(
            input_ids=torch.tensor([prompt], device=device), use_cache=True, output_hidden_states=True
        )
        cache = tuple(
            tuple(tensor.expand(rows, -1, -1, -1).contiguous() for tensor in layer)
            for layer in outputs.past_key_values
        )
        hidden = outputs.hidden_states[-1][:, -1].expand(rows, -1)
        latents = []
        for k in range(total_latent_thoughts):
            latent = torch.stack([override.get(k, h) for override, h in zip(overrides, hidden)])
            latents.append(latent)
            outputs = lm(
                inputs_embeds=latent.unsqueeze(1),
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
            )
            cache = outputs.past_key_values
            hidden = outputs.hidden_states[-1][:, -1]
        outputs = lm(
            input_ids=torch.full((rows, 1), end_id, device=device),
            past_key_values=cache,
            use_cache=True,
        )
        answer_start = outputs
        generated: list[list[int]] = [[] for _ in range(rows)]
        done = torch.zeros(rows, dtype=torch.bool, device=device)
        for _ in range(teacher_tokens):
            tokens = outputs.logits[:, -1].argmax(dim=-1)
            done = done | (tokens == eos_id)
            for row, token in enumerate(tokens.tolist()):
                if not done[row]:
                    generated[row].append(token)
            if done.all():
                break
            outputs = lm(
                input_ids=tokens.unsqueeze(1), past_key_values=outputs.past_key_values, use_cache=True
            )
        # Teacher-force variant 0's answer in every variant. The legacy tuple cache is never
        # modified in place, so the cache from before decoding is still intact.
        clean = torch.tensor(generated[0] + [eos_id], device=device).expand(rows, -1)
        logits = answer_start.logits
        if clean.shape[1] > 1:
            continued = lm(input_ids=clean[:, :-1], past_key_values=answer_start.past_key_values)
            logits = torch.cat([logits, continued.logits], dim=1)
        log_probs = torch.log_softmax(logits.float(), dim=-1).gather(2, clean.unsqueeze(2))
    return (
        [tokenizer.decode(tokens) for tokens in generated],
        torch.stack(latents, dim=1),
        log_probs.squeeze(2).sum(dim=1),
    )


def splice_metrics(
    lm: Any,
    tokenizer: Any,
    split: SplitData,
    reconstructions: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    ids: tuple[int, int, int],
) -> str:
    """Coconut's accuracy with its latents replaced by the NLA's reconstructions, as log lines.

    Each latent source replaces all six latents, or one thought k (later
    thoughts recomputed): AR(AV(h)) unstandardized, the train mean, and the
    next question's latent for the same thought ("shuffled"). Reported: gold
    accuracy, agreement with the clean answer, and the mean log-prob of the
    clean answer. Recovered = (AR - ablated) / (clean - ablated), n/a when the
    ablation lowers the score by under 0.01 (a ratio of noise to noise).

    The ProsQA checkpoint answers from the question: another question's six
    latents change no answer and its log-prob by ~0.002 nats, and attention
    from <|end-latent|> to the latents is under 0.02 past layer 0. So expect
    n/a here; the splice shows the latents barely matter downstream.
    """
    raw = reconstructions * std + mean
    sources = ("AR", "mean", "shuffled")
    names = (
        ["clean"]
        + [f"all {source}" for source in sources]
        + [f"{source}@{k + 1}" for source in sources for k in range(total_latent_thoughts)]
    )
    variant = {name: index for index, name in enumerate(names)}
    correct = [0] * len(names)
    agree = [0] * len(names)
    log_prob_total = torch.zeros(len(names))
    mismatches = 0
    max_difference = 0.0
    questions = len(split.items)
    for question, item in enumerate(split.items):
        rows = slice(question * total_latent_thoughts, (question + 1) * total_latent_thoughts)
        following = (question + 1) % questions
        latents_by_source = (
            raw[rows],
            mean.expand(total_latent_thoughts, -1),
            split.activations[
                following * total_latent_thoughts : (following + 1) * total_latent_thoughts
            ],
        )
        overrides = (
            [{}]
            + [dict(enumerate(latents)) for latents in latents_by_source]
            + [{k: latents[k]} for latents in latents_by_source for k in range(total_latent_thoughts)]
        )
        answers, latents, log_probs = spliced_answers(
            lm, tokenizer, item["question"], overrides, ids
        )
        log_prob_total += log_probs.cpu()
        # Self-checks: the clean variant must reproduce Coconut's own latents and answer.
        max_difference = max(
            max_difference, (latents[0] - split.activations[rows]).abs().max().item()
        )
        mismatches += answers[0] != split.answers[question]
        gold = normalize_answer(item["answer"])
        for index, answer in enumerate(answers):
            correct[index] += normalize_answer(answer) == gold
            agree[index] += answer == answers[0]

    metrics = {
        "accuracy": [count / questions for count in correct],
        "agreement": [count / questions for count in agree],
        "log-prob": (log_prob_total / questions).tolist(),
    }

    def recovered(values: list[float], spliced: str, ablated: str) -> str:
        span = values[0] - values[variant[ablated]]
        if span < 0.01:
            return "n/a"
        return f"{(values[variant[spliced]] - values[variant[ablated]]) / span:+.2f}"

    def report(template: str) -> str:
        parts = []
        for metric, values in metrics.items():
            spliced = "/".join(f"{values[variant[template.format(s)]]:.3f}" for s in sources)
            parts.append(
                f"{metric} clean {values[0]:.3f} spliced {spliced} recovered vs shuffled "
                f"{recovered(values, template.format('AR'), template.format('shuffled'))} "
                f"vs mean {recovered(values, template.format('AR'), template.format('mean'))}"
            )
        return "; ".join(parts)

    lines = [
        f"Splice self-check: clean-loop answers differing from Coconut's "
        f"{mismatches}/{questions}, latent max abs diff {max_difference:.2e}",
        f"Splice all six latents ({questions} questions, spliced = AR/mean/shuffled): "
        + report("all {}"),
    ]
    for k in range(1, total_latent_thoughts + 1):
        lines.append(f"Splice thought {k} only: " + report("{}@" + str(k)))
    return "\n".join(lines)


def cpu_state(model: Any) -> dict[str, torch.Tensor]:
    """Copy a model's weights to CPU so the best checkpoint costs no GPU memory."""
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def draw(order: list[int], size: int, count: int) -> list[int]:
    """Pop `count` indices from shuffled passes over range(size), reshuffling `order` when it empties."""
    drawn = []
    for _ in range(count):
        if not order:
            order.extend(torch.randperm(size).tolist())
        drawn.append(order.pop())
    return drawn


def run_nla_experiment(
    target_model: Any,
    tokenizer: Any,
    train_questions: list[dict[str, str]],
    val_questions: list[dict[str, str]],
    test_questions: list[dict[str, str]],
    ids: tuple[int, int, int],
    config: NLAConfig,
) -> float:
    """Train an NLA and return its held-out test MSE.

    The AR reconstructs activations standardized per dimension with train
    statistics. Raw MSE is dominated by a handful of huge GPT-2 outlier
    dimensions, so standardizing makes every dimension count.

    The starting AV writes nearly the same text for every activation, which
    leaves RL no signal. So the warm start first trains the AV (cross-entropy)
    on teacher explanations, built from what Coconut writes and the latent
    itself (see teacher_explanations), and the AR (MSE) on those texts plus the
    AV's own greedy texts. The teacher sees the question; the AV sees only the
    activation.

    Each RL step uses `batch_size` activations, drawn in shuffled order. For each
    activation, `group_size` explanations are sampled. The AV reward is
    -MSE - kl_coef * KL(AV || reference), where the reference is a frozen copy of
    the AV as it was when RL started, and each sample's advantage is its reward
    relative to the other samples of the same activation. The AR keeps training
    on the samples (and the greedy texts) at the lower `rl_ar_lr`.

    Validation MSE is tracked during training and the best weights are
    restored before the single final pass over the test split. FVE is the
    fraction of the predict-the-mean baseline's MSE that the NLA removes, so
    FVE > 0 means the explanations carry information about the activation.
    """
    if config.group_size < 2:
        raise ValueError("group_size must be at least 2 to form a baseline.")
    if not (train_questions and val_questions and test_questions):
        raise ValueError("NLA needs at least one question in each of train, val and test.")

    start_time = time.time()
    # Every model stays in eval mode: dropout would make the AV's sampling
    # distribution differ from the reference's and add noise to the KL.
    target_model.eval()
    device = next(target_model.parameters()).device
    av = copy.deepcopy(target_model.base_causallm).eval().to(device)
    ar = copy.deepcopy(target_model.base_causallm).eval().to(device)

    train_activations = collect_activations(target_model, tokenizer, train_questions, *ids)
    val_activations = collect_activations(target_model, tokenizer, val_questions, *ids)
    test_activations = collect_activations(target_model, tokenizer, test_questions, *ids)
    print(
        f"Activations: train={len(train_activations)}, "
        f"val={len(val_activations)}, test={len(test_activations)}"
    )

    mean = train_activations.mean(dim=0)
    std = train_activations.std(dim=0).clamp_min(1e-3)
    train_targets = (train_activations - mean) / std
    val_targets = (val_activations - mean) / std
    test_targets = (test_activations - mean) / std

    adapter_parameters = []
    if config.adapter_tokens:
        # A submodule of the AV, so the KL reference copy and the best-weights snapshots include it.
        av.nla_adapter = ActivationAdapter(mean, std, config.adapter_tokens)
        adapter_parameters = list(av.nla_adapter.parameters())
    adapter_ids = {id(parameter) for parameter in adapter_parameters}
    # The zero-initialized adapter and head need larger steps than the pretrained bodies.
    av_optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in av.parameters() if id(p) not in adapter_ids], "lr": config.av_lr},
            {"params": adapter_parameters, "lr": config.head_lr},
        ]
    )
    head = make_head(train_activations.shape[1], device, config.ar_pool_mean)
    ar_optimizer = torch.optim.AdamW(
        [
            {"params": ar.parameters(), "lr": config.ar_lr},
            {"params": head.parameters(), "lr": config.head_lr},
        ]
    )

    val_baseline = mean_baseline_loss(train_targets, val_targets)
    test_baseline = mean_baseline_loss(train_targets, test_targets)
    val_index_baseline = index_baseline_loss(train_targets, val_targets)
    test_index_baseline = index_baseline_loss(train_targets, test_targets)
    print(f"Predict-train-mean baseline MSE: val={val_baseline:.6f}, test={test_baseline:.6f}")
    print(
        "Per-thought-index mean MSE: "
        f"val FVE {1 - val_index_baseline / val_baseline:+.3f}, "
        f"test FVE {1 - test_index_baseline / test_baseline:+.3f}"
    )

    # Coconut's own answers: for the teacher texts, fidelity and the splice self-check.
    val_answers = coconut_answers(target_model, tokenizer, val_questions, *ids)
    test_answers = coconut_answers(target_model, tokenizer, test_questions, *ids)
    train_teacher: list[str] = []
    val_teacher: list[str] | None = None
    test_teacher: list[str] | None = None
    if config.warmup_steps:
        logit_stats = lens_logit_stats(target_model, train_activations, tokenizer)
        train_answers = coconut_answers(target_model, tokenizer, train_questions, *ids)
        train_teacher, val_teacher, test_teacher = (
            teacher_explanations(
                target_model, tokenizer, questions, activations, answers, config, logit_stats
            )
            for questions, activations, answers in (
                (train_questions, train_activations, train_answers),
                (val_questions, val_activations, val_answers),
                (test_questions, test_activations, test_answers),
            )
        )
        # Plus EOS: the AV needs this many tokens to write a teacher text in full.
        lengths = [len(tokenizer.encode(text)) + 1 for text in train_teacher]
        print(
            f"Teacher explanations: distinct={len(set(train_teacher))}/{len(train_teacher)}, "
            f"tokens mean {sum(lengths) / len(lengths):.1f} max {max(lengths)} "
            f"(explanation limit {config.explanation_tokens}), e.g. {train_teacher[:2]!r}"
        )
    # The frozen target is done (latents and teacher texts are collected) until the test
    # splice. Its ~0.5 GB is better spent on training: past 8 GB the GPU spills into system
    # RAM and slows down.
    target_model.base_causallm.to("cpu")
    torch.cuda.empty_cache()

    # Frontier names per row, and the linear probe the NLA's frontier recall is compared with.
    vocab = sorted({name for item in train_questions for name in item["idx_to_symbol"]})
    vocab_set = set(vocab)
    probe = train_frontier_probe(train_targets, frontier_names(train_questions), vocab)
    val_split, test_split = (
        SplitData(
            items=questions,
            activations=activations,
            targets=targets,
            answers=answers,
            correct=[
                normalize_answer(answer) == normalize_answer(item["answer"])
                for answer, item in zip(answers, questions)
            ],
            teacher=teacher,
            frontiers=frontier_names(questions),
            probe_ranking=probe_rankings(probe, targets, vocab),
        )
        for questions, activations, targets, answers, teacher in (
            (val_questions, val_activations, val_targets, val_answers, val_teacher),
            (test_questions, test_activations, test_targets, test_answers, test_teacher),
        )
    )
    del probe
    torch.cuda.reset_peak_memory_stats()
    sizes = [len(frontier) for frontier in val_split.frontiers]
    print(
        f"Coconut accuracy: val {sum(val_split.correct)}/{len(val_questions)}, "
        f"test {sum(test_split.correct)}/{len(test_questions)}; frontier vocab {len(vocab)} names, "
        f"val frontier size mean {sum(sizes) / len(sizes):.2f}, empty for "
        f"{sizes.count(0)}/{len(sizes)} rows"
    )
    if val_split.teacher is not None:
        frontier = frontier_recall(val_split.teacher, val_split, vocab_set)
        print(
            f"Teacher-text val frontier recall {frontier['nla_recall']:.3f} vs probe "
            f"{frontier['probe_recall']:.3f}, precision {frontier['nla_precision']:.3f} vs probe "
            f"{frontier['probe_precision']:.3f}"
        )

    rules = decoding_rules(tokenizer, config)
    best_val = float("inf")
    best_weights: tuple[dict[str, torch.Tensor], ...] = ()

    def validate(label: str, details: str = "") -> None:
        """Score the validation split, keep the weights if they are the best so far, and report."""
        nonlocal best_val, best_weights
        val_loss, explanations, reconstructions = evaluate_nla(
            av, ar, head, tokenizer, val_activations, val_targets, config
        )
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_weights = (cpu_state(av), cpu_state(ar), cpu_state(head))
        print(
            f"[{time.time() - start_time:.0f}s] {label}: {details}"
            f"val_loss={val_loss:.6f} (FVE {1 - val_loss / val_baseline:+.3f}, "
            f"vs index mean {1 - val_loss / val_index_baseline:+.3f})"
            f"{' *' if improved else ''}, "
            f"distinct={len(set(explanations))}/{len(explanations)}, "
            f"explanations={explanations[:3]!r}"
        )
        print(
            explanation_metrics(
                av,
                ar,
                head,
                tokenizer,
                val_split,
                explanations,
                reconstructions,
                train_targets,
                std,
                vocab_set,
                config.seed,
            )
        )

    validate("Step 0")

    order: list[int] = []

    warmup_eval_every = max(1, config.warmup_steps // config.evals_per_phase)
    for step in range(config.warmup_steps):
        rows = draw(order, len(train_activations), config.batch_size * config.group_size)
        texts = [train_teacher[row] for row in rows]
        nll = explanation_nll(av, train_activations[rows], tokenizer, texts)
        nll.backward()
        av_optimizer.step()
        # Gradients are freed right after each step, so the AV's and AR's never share the GPU.
        av_optimizer.zero_grad()

        text_sets = [texts]
        if config.ar_av_text:
            av_texts, _ = sample_explanations(
                av,
                train_activations[rows],
                tokenizer,
                config.explanation_tokens,
                sample=False,
                **rules,
            )
            text_sets.append(av_texts)
        # One backward pass per text set, accumulating the mean over all of them: same
        # gradient, half the activation memory.
        mse = 0.0
        for ar_texts in text_sets:
            reconstructions = reconstruct_activations(ar, head, tokenizer, ar_texts)
            set_mse = ((reconstructions - train_targets[rows]) ** 2).mean() / len(text_sets)
            set_mse.backward()
            mse += set_mse.item()
        ar_optimizer.step()
        ar_optimizer.zero_grad()

        if (step + 1) % warmup_eval_every == 0:
            validate(
                f"Warm-start step {step + 1}/{config.warmup_steps}",
                f"nll={nll.item():.3f}, train_loss={mse:.6f}, ",
            )

    # The KL anchors RL to the AV it starts from, so explanations stay readable.
    reference = copy.deepcopy(av).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    # Fresh optimizer: RL needs much smaller steps than the supervised warm start.
    av_optimizer = torch.optim.AdamW(av.parameters(), lr=config.rl_lr)
    for group in ar_optimizer.param_groups:
        group["lr"] = config.rl_ar_lr

    eval_every = max(1, config.steps // config.evals_per_phase)
    group_size = config.group_size
    for step in range(config.steps):
        # Each activation repeated group_size times, so rows [i * group_size, (i + 1) * group_size) share one.
        batch = draw(order, len(train_activations), config.batch_size)
        rows = torch.tensor(batch, device=device).repeat_interleave(group_size)
        activations, targets = train_activations[rows], train_targets[rows]

        explanations, sampled_ids = sample_explanations(
            av, activations, tokenizer, config.explanation_tokens, **rules
        )
        with torch.no_grad():
            ref_log_probs = sequence_log_probs(
                reference, activations, tokenizer, sampled_ids, **rules
            )

        # The AR trains before the AV's log-prob graph is built, so the two graphs never
        # share the GPU. The rewards use the AR's error from before its step.
        reconstructions = reconstruct_activations(ar, head, tokenizer, explanations)
        mse = ((reconstructions - targets) ** 2).mean(dim=-1)
        if config.rl_ar_lr:
            ar_loss = mse.mean()
            if config.ar_av_text:
                # Evaluation feeds the AR greedy text, so train it on that too, not only on samples.
                greedy, _ = sample_explanations(
                    av,
                    train_activations[batch],
                    tokenizer,
                    config.explanation_tokens,
                    sample=False,
                    **rules,
                )
                greedy_reconstructions = reconstruct_activations(ar, head, tokenizer, greedy)
                greedy_mse = ((greedy_reconstructions - train_targets[batch]) ** 2).mean()
                ar_loss = (ar_loss + greedy_mse) / 2
            ar_loss.backward()
            ar_optimizer.step()
            ar_optimizer.zero_grad()

        log_probs = sequence_log_probs(av, activations, tokenizer, sampled_ids, **rules)
        # Sample-based KL estimate: sum over sampled tokens of log pi_av - log pi_ref.
        kl = log_probs.detach() - ref_log_probs
        rewards = (-mse.detach() - config.kl_coef * kl).view(config.batch_size, group_size)
        # No division by the group's std: samples of one activation often score almost
        # the same, and normalizing would blow their noise up into full-size updates.
        # The targets are standardized, so raw advantages already have a sensible scale.
        advantages = rewards - rewards.mean(dim=1, keepdim=True)
        policy_loss = -(advantages.flatten() * log_probs).mean()
        policy_loss.backward()
        av_optimizer.step()
        av_optimizer.zero_grad()

        if (step + 1) % eval_every == 0 or step + 1 == config.steps:
            validate(
                f"RL step {step + 1}/{config.steps}",
                f"train_loss={mse.mean().item():.6f} (batch mean), kl={kl.mean().item():.2f}, ",
            )

    # Test once, with the weights that did best on validation.
    for model, weights in zip((av, ar, head), best_weights):
        model.load_state_dict(weights)
    test_loss, test_explanations, test_reconstructions = evaluate_nla(
        av, ar, head, tokenizer, test_activations, test_targets, config
    )
    test_fve = 1 - test_loss / test_baseline
    print(f"\nBest val loss: {best_val:.6f} (FVE {1 - best_val / val_baseline:+.3f})")
    print(
        f"Final NLA test reconstruction loss: {test_loss:.6f} "
        f"(FVE {test_fve:+.3f}, vs index mean {1 - test_loss / test_index_baseline:+.3f}, "
        f"raw space {raw_space_fve(test_reconstructions, test_targets, train_targets, std):+.3f}), "
        f"distinct={len(set(test_explanations))}/{len(test_explanations)}"
    )
    print(
        explanation_metrics(
            av,
            ar,
            head,
            tokenizer,
            test_split,
            test_explanations,
            test_reconstructions,
            train_targets,
            std,
            vocab_set,
            config.seed,
        )
    )
    print(
        f"Test explanations: {test_explanations[:6]!r}\n"
        f"elapsed: {time.time() - start_time:.0f}s, "
        f"peak GPU memory in training: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
    )

    # The NLA's models are done; the splice and the baseline AR need the GPU.
    del av, ar, head, reference, av_optimizer, ar_optimizer, best_weights
    torch.cuda.empty_cache()
    if config.splice:
        target_model.base_causallm.to(device)
        print(
            splice_metrics(
                target_model.base_causallm, tokenizer, test_split, test_reconstructions, mean, std, ids
            )
        )
        print(f"elapsed: {time.time() - start_time:.0f}s")
    if config.question_steps:
        question_val, question_test = question_baseline(
            target_model.base_causallm,
            tokenizer,
            (
                question_texts(train_questions),
                question_texts(val_questions),
                question_texts(test_questions),
            ),
            (train_targets, val_targets, test_targets),
            config,
            device,
        )
        question_fve = 1 - question_test / test_baseline
        print(
            f"Question-only AR: best val FVE {1 - question_val / val_baseline:+.3f}, "
            f"test loss {question_test:.6f} (FVE {question_fve:+.3f}); "
            f"NLA minus question-only test FVE {test_fve - question_fve:+.3f}"
        )
    return test_loss


def load_split(path: str, count: int) -> list[dict[str, str]]:
    """Load the first `count` questions of one dataset split."""
    with open(path, encoding="utf-8") as file:
        return json.load(file)[:count]


def parse_config() -> NLAConfig:
    """Build an NLAConfig from --nla-<field-name> command-line flags."""
    parser = argparse.ArgumentParser(description="Train an NLA on Coconut's ProsQA latents.")
    for field in dataclasses.fields(NLAConfig):
        options: dict[str, Any] = {"default": field.default, "help": field.metadata["help"]}
        if isinstance(field.default, bool):
            options["action"] = argparse.BooleanOptionalAction
        else:
            options["type"] = type(field.default)
        if "choices" in field.metadata:
            options["choices"] = field.metadata["choices"]
        parser.add_argument("--nla-" + field.name.replace("_", "-"), dest=field.name, **options)
    return NLAConfig(**vars(parser.parse_args()))


def main() -> None:
    """Load the Coconut checkpoint and run the NLA training experiment."""
    config = parse_config()
    set_seed(config.seed)

    model, tokenizer = load_model(
        repo_id="bmarti44/coconut-curriculum-checkpoints",
        checkpoint_filename="coconut/checkpoint_49",
        checkpoint_dir="checkpoints/prosqa_coconut",
    )
    ids = tuple(
        tokenizer.convert_tokens_to_ids(token)
        for token in ("<|start-latent|>", "<|latent|>", "<|end-latent|>")
    )
    run_nla_experiment(
        model,
        tokenizer,
        load_split(train_path, config.train_examples),
        load_split(val_path, config.val_examples),
        load_split(test_path, config.test_examples),
        ids,
        config,
    )


if __name__ == "__main__":
    main()
