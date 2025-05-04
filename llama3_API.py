# app.py
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ── local model path ────────────────────────────────────────────────────────────
MODEL_DIR = r"C:\AI-WorkSpace\Utilities-CI\WAT-epl-ai-logAnalyzer\finetuned_llama_v2"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# ── load model once at startup ──────────────────────────────────────────────────
tok   = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            torch_dtype="auto",
            device_map="auto" if DEVICE == "cuda" else None
        ).to(DEVICE)

app = Flask(__name__)

# ── POST /convert  { "html": "<h1>Hello</h1>" } → { "markdown": "# Hello" } ─────
@app.route("/convert", methods=["POST"])
def convert():
    data   = request.get_json(force=True)
    html   = data.get("html", "").strip()
    tokens = int(data.get("max_new_tokens", 256))

    prompt = (
        "Convert the following HTML to clean GitHub‑flavored Markdown.\n\n"
        f"HTML:\n{html}\n\nMarkdown:"
    )

    with torch.no_grad():
        ids_in   = tok(prompt, return_tensors="pt").to(DEVICE)
        ids_out  = model.generate(**ids_in, max_new_tokens=tokens)
        markdown = tok.decode(ids_out[0], skip_special_tokens=True)

    return jsonify({"markdown": markdown})

if __name__ == "__main__":
    # for prod use gunicorn / uvicorn; this is fine for a quick test
    app.run(host="0.0.0.0", port=8000)
