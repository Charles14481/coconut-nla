import json
import re

import torch

from load_checkpoint import load_model

q_ct = 25
total_latent_thoughts = 6
interpret_q_ct = 1
dataset_path = "data/prosqa_test.json"

def normalize_answer(text):
    answer_text = text.split("#")[-1].strip().lower()
    return re.sub(r"\s+", " ", answer_text).rstrip(".!?")


def make_inputs(
    tokenizer, question, number_of_latents, start_id, latent_id, end_id, device
):
    question_tokens = tokenizer.encode(question + "\n", add_special_tokens=True)
    input_ids = question_tokens + [start_id] + [latent_id] * number_of_latents + [end_id]
    return {
        "input_ids": torch.tensor([input_ids], device=device),
        "attention_mask": torch.ones(1, len(input_ids), device=device),
    }


def probe_latent_thoughts(
    model, tokenizer, question, start_id, latent_id, end_id, device
):
    print("Latent probes:")
    for count in range(1, total_latent_thoughts + 1):
        inputs = make_inputs(
            tokenizer,
            question,
            count,
            start_id,
            latent_id,
            end_id,
            device,
        )
        with torch.inference_mode():
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=32,
            )
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = output[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"  after {count} latent thought(s): {generated_text!r}")


def main():
    model, tokenizer = load_model(
        repo_id="bmarti44/coconut-curriculum-checkpoints",
        checkpoint_filename="coconut/checkpoint_49",
        checkpoint_dir="checkpoints/prosqa_coconut",
    )

    with open(dataset_path, encoding="utf-8") as file:
        questions = json.load(file)[:q_ct]

    device = next(model.parameters()).device
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    correct = 0

    for index, item in enumerate(questions, start=1):
        inputs = make_inputs(
            tokenizer,
            item["question"],
            total_latent_thoughts,
            start_id,
            latent_id,
            end_id,
            device,
        )
        with torch.inference_mode():
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=64,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = output[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        predicted = normalize_answer(generated_text)
        expected = normalize_answer(item["answer"])
        is_correct = predicted == expected
        correct += is_correct

        print(f"\nQuestion {index}: {item['question']}")
        if index <= interpret_q_ct:
            probe_latent_thoughts(
                model,
                tokenizer,
                item["question"],
                start_id,
                latent_id,
                end_id,
                device,
            )
        print(f"Model continuation: {generated_text}")
        print(f"Expected: {item['answer']} | Predicted: {predicted} | Correct: {is_correct}")

    print(f"\nScore: {correct}/{len(questions)}")


if __name__ == "__main__":
    main()