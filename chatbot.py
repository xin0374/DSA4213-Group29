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
        # Display answer
        st.markdown(f"🩺 **Assistant:** {msg['answer']}")
        
        # Display sources if available
        if msg.get('sources'):
            with st.expander(f"📚 View Sources ({len(msg['sources'])})"):
                for i, source in enumerate(msg['sources'], start=1):
                    st.markdown(f"**[{i}] {source['title']}**")
                    if source.get('url'):
                        st.markdown(f"🔗 [{source['url']}]({source['url']})")
                    if i < len(msg['sources']):
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
    
    # Extract source information
    ev = out.get("evidence", pd.DataFrame())
    sources = []
    if not ev.empty:
        for _, row in ev.head(4).iterrows():
            sources.append({
                'title': row.get("source", "Unknown Source"),
                'url': row.get("url", ""),
                'text': row.get("text", "")[:300] + "..." if len(row.get("text", "")) > 300 else row.get("text", "")
            })
    
    # Store message with separate answer and sources
    st.session_state.messages.append({
        "role": "assistant",
        "answer": answer_text,
        "sources": sources
    })
    
    # Force UI to refresh with the new messages
    st.rerun()