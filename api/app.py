from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import os

from model.load_model import load_baseline, MODEL_NAME, TOKEN

app = FastAPI(title="Local ChatGPT-like API")

class Msg(BaseModel):
    prompt: str
    max_tokens: int = 256

# Globals to hold the model once loaded
_tok = None
_mdl = None

def get_model():
    global _tok, _mdl
    if _tok is None or _mdl is None:
        _tok, _mdl = load_baseline()
    return _tok, _mdl

def stream_tokens(prompt: str, max_tokens: int):
    tok, mdl = get_model()
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    out = mdl.generate(
        **inputs,
        max_new_tokens=max_tokens,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    answer = tok.decode(out[0], skip_special_tokens=True)
    for chunk in answer.split():
        yield f"data: {chunk}\n\n"
        asyncio.sleep(0)
    yield "data: [DONE]\n\n"

@app.post("/chat")
def chat(msg: Msg):
    tok, mdl = get_model()
    inputs = tok(msg.prompt, return_tensors="pt").to(mdl.device)
    out = mdl.generate(**inputs, max_new_tokens=msg.max_tokens)
    return {"answer": tok.decode(out[0], skip_special_tokens=True)}

@app.post("/chat_stream")
def chat_stream(msg: Msg):
    return StreamingResponse(
        stream_tokens(msg.prompt, msg.max_tokens),
        media_type="text/event-stream",
    )

from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="static", html=True), name="static")
