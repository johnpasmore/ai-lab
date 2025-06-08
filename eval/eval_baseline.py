import json, time, torch, tqdm
from datasets import load_dataset
from model.load_model import load_baseline

tok, mdl = load_baseline()

# TruthfulQA multiple-choice validation set (~817 Q’s) — we’ll take 50 for speed
ds = load_dataset("truthful_qa", "multiple_choice", split="validation")[:50]

right = 0
t0 = time.time()

for item in tqdm.tqdm(ds, total=len(ds)):
    q = item["question"]
    correct = item["mc1_targets"]["labels"][0]          # first listed correct answer
    prompt = f"Q: {q}\nA:"
    out = mdl.generate(**tok(prompt, return_tensors="pt").to(mdl.device),
                       max_new_tokens=32)
    reply = tok.decode(out[0], skip_special_tokens=True).split("\n")[0].strip()
    if correct.lower() in reply.lower():
        right += 1

stats = {"accuracy": right / len(ds), "seconds": time.time() - t0}
print(json.dumps(stats, indent=2))

