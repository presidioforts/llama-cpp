# app.py  ── generic chat‑completion wrapper for LLaMA‑3 Instruct
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, time, uuid, os

MODEL_DIR = r"C:\AI-WorkSpace\Utilities-CI\WAT-epl-ai-logAnalyzer\finetuned_llama_v2"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

tok   = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, torch_dtype="auto",
            device_map="auto" if DEVICE == "cuda" else None
        ).to(DEVICE)

app = Flask(__name__)

@app.route("/chat/completions", methods=["POST"])
def chat_completion():
    t0      = time.time()
    body    = request.get_json(force=True)

    messages     = body.get("messages", [])
    max_tokens   = int(body.get("max_tokens", 512))
    temperature  = float(body.get("temperature", 0.7))
    top_p        = float(body.get("top_p", 0.95))

    # ❶ build prompt with LLaMA‑3 chat template
    prompt_ids = tok.apply_chat_template(
                    messages, return_tensors="pt").to(DEVICE)

    # ❷ generate
    with torch.no_grad():
        out_ids = model.generate(
                    prompt_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p)
    full_text = tok.decode(out_ids[0], skip_special_tokens=True)

    # strip the prompt so we return only the assistant part
    assistant_reply = full_text[len(tok.decode(prompt_ids[0], skip_special_tokens=True)):]

    # ── OpenAI‑style response payload ──────────────────────────────
    resp = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": os.path.basename(MODEL_DIR),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": assistant_reply.strip()},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_ids.shape[-1],
            "completion_tokens": len(out_ids[0]) - prompt_ids.shape[-1],
            "total_tokens": len(out_ids[0])
        }
    }
    app.logger.info(f"completed in {time.time()-t0:.2f}s, total_tokens={resp['usage']['total_tokens']}")
    return jsonify(resp)


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    # for production run with gunicorn/uvicorn; this is dev only
    app.run(host="0.0.0.0", port=8000)
