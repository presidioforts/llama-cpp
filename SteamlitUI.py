#!/usr/bin/env python3
import os
import streamlit as st
import openai
from dotenv import load_dotenv

# ─── Load env vars ─────────────────────────────────────────────────────
load_dotenv()
API_BASE   = os.getenv("OPENAI_API_BASE", "http://localhost:8000/chat")
MODEL_NAME = os.getenv("OPENAI_MODEL", "llama3-finetune")
openai.api_base = API_BASE
openai.api_key  = os.getenv("OPENAI_API_KEY", "not-used")

# ─── Streamlit page setup ─────────────────────────────────────────────
st.set_page_config(page_title="Llama‑3 Instruct Chat", layout="centered")
st.title("🤖 LLaMA‑3 Instruct Chat")

# ─── Initialize chat history ──────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# ─── Display chat messages ────────────────────────────────────────────
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Assistant:**\n\n{msg['content']}", unsafe_allow_html=True)

# ─── User input ───────────────────────────────────────────────────────
user_input = st.text_input("You:", key="input")
send = st.button("Send")

if send and user_input:
    # append user message
    st.session_state.history.append({"role": "user", "content": user_input})
    # call OpenAI SDK (no /v1 prefix)
    with st.spinner("Generating..."):
        resp = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=st.session_state.history,
            max_tokens=int(os.getenv("MAX_TOKENS", 256)),
            temperature=float(os.getenv("TEMPERATURE", 0.7)),
            top_p=float(os.getenv("TOP_P", 0.95)),
            stream=False
        )
    reply = resp.choices[0].message.content.strip()
    # append assistant message
    st.session_state.history.append({"role": "assistant", "content": reply})
    # clear input and rerun to display updated chat
    st.session_state.input = ""
    st.experimental_rerun()
