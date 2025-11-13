import os
import streamlit as st
import pandas as pd
import torch
import gc
from functions import (
    load_generator,
    load_retrieval_artifacts,
    generate_with_rag,
)

# ==========================
# Configuration
# ==========================
MAX_NEW_TOKENS = 180
K_BM25 = 100
K_DENSE = 100
K_FINAL = 30
TOP_N_RERANK = 4

# ==========================
# Cached loading of models and artifacts
# ==========================
@st.cache_resource
def load_models_and_artifacts():
    """Load all models and artifacts with memory optimization."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    FULLMODEL_DIR = "finetuned_biogpt"
    tok_full, gen_full = load_generator(FULLMODEL_DIR)
    train_df, bm25, bm25_corpus_tokens, dense, index, _dense_rows = load_retrieval_artifacts()
    
    return tok_full, gen_full, train_df, bm25, bm25_corpus_tokens, dense, index, _dense_rows

# Load models once
tok_full, gen_full, train_df, bm25, bm25_corpus_tokens, dense, index, _dense_rows = load_models_and_artifacts()

# ==========================
# Streamlit UI setup
# ==========================
st.set_page_config(page_title="Medical Q&A", page_icon="🧠", layout="centered")

st.title("🧠 Medical Q&A Assistant")
st.caption("Ask medical questions and get evidence-based answers")

# Initialize session state
if "show_evidence" not in st.session_state:
    st.session_state.show_evidence = True
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================
# Model answering function
# ==========================
def ask_once(user_q: str, max_new_tokens: int = MAX_NEW_TOKENS, retry_count: int = 0):
    """Generate answer with automatic retry on OOM errors."""
    try:
        out = generate_with_rag(
            user_q, 
            tok_full, 
            gen_full,
            bm25,
            train_df,
            dense,
            index,
            _dense_rows,
            max_new_tokens=max_new_tokens,
            use_base_prompt=False,
            k_bm25=K_BM25,
            k_dense=K_DENSE,
            k_final=K_FINAL,
            top_n=TOP_N_RERANK
        )
        return out["answer"], out.get("evidence", pd.DataFrame())
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and retry_count < 2:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return ask_once(user_q, max_new_tokens=max(80, max_new_tokens // 2), retry_count=retry_count + 1)
        raise

# ==========================
# Evidence display
# ==========================
def show_evidence_table(ev: pd.DataFrame, k: int = 4):
    """Display evidence table with proper formatting."""
    if ev is None or ev.empty:
        return
    
    ev_display = ev.head(k)[["answer", "source", "url", "ce_score"]].copy()
    ev_display = ev_display.rename(columns={
        "answer": "Snippet",
        "source": "Source",
        "url": "URL",
        "ce_score": "Relevance"
    })
    ev_display["Relevance"] = ev_display["Relevance"].round(3)
    
    with st.expander("📚 Sources", expanded=False):
        st.dataframe(ev_display, use_container_width=True, hide_index=True)

# ==========================
# Sidebar
# ==========================
with st.sidebar:
    st.header("Settings")
    
    # Toggle evidence display
    show_ev = st.toggle("Show evidence sources", value=st.session_state.show_evidence)
    if show_ev != st.session_state.show_evidence:
        st.session_state.show_evidence = show_ev
        st.rerun()
    
    # Clear conversation
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Stats
    st.caption(f"💬 {len(st.session_state.messages)} messages")
    device = "GPU" if torch.cuda.is_available() else "CPU"
    st.caption(f"🖥️ Running on {device}")

# ==========================
# Main chat interface
# ==========================
# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "evidence" in msg and st.session_state.show_evidence:
            show_evidence_table(msg["evidence"])

# Input box
user_input = st.chat_input("Ask a medical question...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                ans, ev = ask_once(user_input)
                
                # Display answer
                st.markdown(ans)
                
                # Store message with evidence
                msg_data = {"role": "assistant", "content": ans}
                if not ev.empty:
                    msg_data["evidence"] = ev
                st.session_state.messages.append(msg_data)
                
                # Show evidence if enabled
                if st.session_state.show_evidence and not ev.empty:
                    show_evidence_table(ev)
                
            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")
                if "out of memory" in str(e).lower():
                    st.info("💡 Try asking a simpler question or restart the app.")