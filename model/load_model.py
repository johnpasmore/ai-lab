import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ────────────────────────────────────────────────────────────────────────────────
# 1. MODEL NAME     (gated repo – requires an HF access token)
# ────────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"


# ────────────────────────────────────────────────────────────────────────────────
# 2. How to provide your HF token
#    • Option A: set env var   export HF_TOKEN="hf_xxxxxxxxxx"
#    • Option B: hard-code it  TOKEN = "hf_xxxxxxxxxx"
# ────────────────────────────────────────────────────────────────────────────────
#TOKEN = os.getenv("HF_TOKEN")           # reads from env var by default
TOKEN = "hf_LWINAyeHDQSyxxxxxxxxxxxxxxxx"
# TOKEN = "hf_your_long_token_here"     # ← uncomment & paste if you prefer

if TOKEN is None or TOKEN == "":
    raise RuntimeError(
        "Hugging Face token not found.  Set the HF_TOKEN environment variable "
        "or paste it into the TOKEN variable in this file."
    )


def load_baseline():
    """Return tokenizer + full-precision CPU model (works on Apple silicon; slow)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map={"": "cpu"},      # force everything onto CPU RAM
        torch_dtype=torch.float32,   # 32-bit floats
        token=TOKEN,                 # pass token for model weights
    )
    return tokenizer, model


if __name__ == "__main__":
    tok, mdl = load_baseline()
    prompt = "Why is the sky blue?"
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    out = mdl.generate(**inputs, max_new_tokens=64)
    print(tok.decode(out[0], skip_special_tokens=True))

