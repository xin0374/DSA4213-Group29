import sys
import torch
import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Make sure we can import your modules
sys.path.append(".")

from rag_retrieval import hybrid_search, rerank_cross_encoder  # (loaded when module imports)
from rag_generation import generate_with_rag

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "finetuned_biogpt"

@st.cache_resource
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16 if DEVICE == "cuda" else None
    ).to(DEVICE).eval()

    return tok, model

tok, gen_model = load_model()

# ========== Streamlit UI ==========
st.set_page_config(page_title="MedQuAD Assistant", page_icon="🩺")
st.title("🩺 Medical Assistant")
st.write("How can I help you today?")

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"🧑‍💻 **You:** {msg['content']}")
    else:
        st.markdown(f"🩺 **Assistant:** {msg['content']}")

st.markdown("---")

# === Input form ===
with st.form("chat_form", clear_on_submit=True):
    user_q = st.text_input(
        "Your question",
        placeholder="Type your medical question here...",
        key="input_question",
    )
    submitted = st.form_submit_button("Send")

# Handle a *new* question only when the button is pressed
if submitted and user_q.strip():
    st.session_state.messages.append({"role": "user", "content": user_q})

    with st.spinner("Thinking..."):
        out = generate_with_rag(
            user_q=user_q,
            tok=tok,
            gen_model=gen_model,
            use_base_prompt=False,
            max_new_tokens=240,
        )

    answer_text = out["answer"]

    # Build top-4 sources string
    ev = out.get("evidence", pd.DataFrame())
    citation = ""
    if not ev.empty:
        lines = []
        for i, (_, row) in enumerate(ev.head(4).iterrows(), start=1):
            source = row.get("source", "Unknown")
            url = row.get("url", "")
            if url:
                lines.append(f"[{i}] **{source}**: {url}")
            else:
                lines.append(f"[{i}] **{source}**")
        citation = "\n\n**Sources:**  " + "  ".join(lines)

    final_output = answer_text + citation
    st.session_state.messages.append({"role": "assistant", "content": final_output})

    # Force UI to refresh with the new messages
    st.rerun()

