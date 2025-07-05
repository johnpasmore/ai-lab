import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
TOKEN = os.getenv("HF_TOKEN")
if not TOKEN:
    raise RuntimeError("HF_TOKEN is not set in the environment!")

def load_baseline():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_auth_token=TOKEN
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map={"": "cpu"},
        torch_dtype=torch.float32,
        use_auth_token=TOKEN
    )
    return tokenizer, model

if __name__ == "__main__":
    tok, mdl = load_baseline()
    prompt = "Why is the sky blue?"
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    out = mdl.generate(**inputs, max_new_tokens=8)
    print(tok.decode(out[0], skip_special_tokens=True))

