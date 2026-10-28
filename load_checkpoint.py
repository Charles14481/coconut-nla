import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from coconut import Coconut

def load_model(
    repo_id="tsrigo/coconut",
    checkpoint_filename="stage4_checkpoint_25",
    checkpoint_dir="checkpoints/checkpointTs",
):
    load_model_path = hf_hub_download(
        repo_id=repo_id,
        filename=checkpoint_filename,
        local_dir=checkpoint_dir,
    )
    model_id = "openai-community/gpt2"
    print(f"Loading from {load_model_path}.")

    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    saved_weights = torch.load(
        load_model_path, map_location=torch.device("cuda")
    )
    
    model.resize_token_embeddings(len(tokenizer))
    embeddings = model.get_input_embeddings()
    target_id = tokenizer.convert_tokens_to_ids("<<")
    # initialize the new token embeddings with a known token
    # it helps stablize the training
    for token_id in [latent_id, start_id, end_id]:
        target_embedding = embeddings.weight.data[target_id]
        embeddings.weight.data[token_id] = target_embedding
        # The input embeddings and lm heads are tied in GPT2. So the code below is not necessary
        lm_head = model.lm_head
        lm_head.weight.data[token_id] = lm_head.weight.data[target_id]

    model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)
    print(model.load_state_dict(saved_weights, strict=False))
    model = model.to("cuda")
    model.eval()
    return model, tokenizer


def main():
    model, tokenizer = load_model()
    
    prompt = "Sally received the following scores on her math quizzes: 50, 80, 80. Find her mean score."
    prompt = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(
        **prompt,
        max_new_tokens=20
    )
    for i, o in enumerate(output):
        print(f"Output {i}: {tokenizer.decode(o, skip_special_tokens=True)}")
    
if __name__ == "__main__":
    main()
