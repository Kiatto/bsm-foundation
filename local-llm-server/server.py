"""Minimal OpenAI-compatible server around an Unsloth-loaded model, so
extract.js/plan.js can call your local GPU instead of OpenRouter's
rate-limited free tier. Exposes exactly what they need:

    POST /v1/chat/completions
    {"model": "<ignored, informational>", "max_tokens": 2000,
     "messages": [{"role": "user", "content": "..."}]}
    -> {"choices": [{"message": {"role": "assistant", "content": "..."}}]}

Run:
    pip install fastapi uvicorn unsloth
    python server.py --model unsloth/gemma-3-4b-it-bnb-4bit

Then set (see ../vercel-demo/.env.local.example):
    LLM_BASE_URL=http://localhost:8000/v1/chat/completions
    LLM_API_KEY=local   # anything non-empty, ignored by this server
"""

import argparse
import time

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
_model = None
_tokenizer = None


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "local"
    max_tokens: int = 2000
    messages: list[Message]
    temperature: float = 0.2


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    from unsloth import FastLanguageModel  # noqa: F401 (loaded lazily)
    import torch

    prompt = _tokenizer.apply_chat_template(
        [m.model_dump() for m in req.messages],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    with torch.no_grad():
        out = _model.generate(
            **inputs, max_new_tokens=req.max_tokens,
            temperature=max(req.temperature, 0.01), do_sample=True,
        )
    text = _tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return {
        "id": f"local-{int(time.time())}",
        "model": req.model,
        "choices": [{"index": 0, "message":
                    {"role": "assistant", "content": text},
                    "finish_reason": "stop"}],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/gemma-3-4b-it-bnb-4bit",
                    help="any Unsloth/HF model id or local path")
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    from unsloth import FastLanguageModel

    print(f"Loading {args.model} ...")
    _model, _tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_seq_len,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(_model)
    print(f"Ready — serving on http://localhost:{args.port}/v1/chat/completions")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
