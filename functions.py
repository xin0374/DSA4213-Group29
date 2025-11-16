import pandas as pd
import torch
import re
import os
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import minmax_scale
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from safetensors.torch import save_file


try:
    import faiss
except ImportError:
    # If faiss is not in Python path, try to find it in site-packages
    import site
    import sys
    for site_path in site.getsitepackages():
        if site_path not in sys.path:
            sys.path.append(site_path)
    import faiss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# Helper: tokenize text
# =============================
_word_re = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
def simple_tokenize(text: str):
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return _word_re.findall(text.lower())

# =============================
# Build doc column (Q + A + meta)
# =============================
def build_doc_row(row):
    parts = []
    q = str(row.get("question", "")).strip()
    a = str(row.get("answer", "")).strip()
    if q: parts.append(f"Q: {q}")
    if a: parts.append(f"A: {a}")

    meta_bits = []
    for lab, key in [("Entity", "entity"), ("Type", "qtype"), ("Source", "source"), ("URL", "url")]:
        val = str(row.get(key, "") or "").strip()
        if val: meta_bits.append(f"{lab}: {val}")
    if meta_bits:
        parts.append("\n".join(meta_bits))
    return "\n".join(parts).strip()

# =============================
# Save & Load Helpers
# =============================
SAVE_DIR = "./retrieval_artifacts"
os.makedirs(SAVE_DIR, exist_ok=True)

def save_retrieval_artifacts(train_df, bm25, bm25_corpus_tokens, index, dense_emb=None):
    # BM25
    with open(f"{SAVE_DIR}/bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    # BM25 tokens
    with open(f"{SAVE_DIR}/bm25_corpus_tokens.pkl", "wb") as f:
        pickle.dump(bm25_corpus_tokens, f)
    # Train df
    train_df.to_csv(f"{SAVE_DIR}/train_df.csv", index=False)
    # FAISS index
    faiss.write_index(index, f"{SAVE_DIR}/dense_index.faiss")
    # Dense embeddings
    if dense_emb is not None:
        np.save(f"{SAVE_DIR}/dense_emb.npy", dense_emb)

def load_retrieval_artifacts():
    # BM25
    with open(f"{SAVE_DIR}/bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(f"{SAVE_DIR}/bm25_corpus_tokens.pkl", "rb") as f:
        bm25_corpus_tokens = pickle.load(f)
    # Dataframe
    train_df = pd.read_csv(f"{SAVE_DIR}/train_df.csv")
    _dense_rows = train_df.reset_index(drop=True)
    # FAISS
    index = faiss.read_index(f"{SAVE_DIR}/dense_index.faiss")
    # Dense model
    dense = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO", device=device)
    return train_df, bm25, bm25_corpus_tokens, dense, index, _dense_rows

# =============================
# Build indexes (if first time)
# =============================
def build_retrieval(train_df):
    # Add 'doc' column
    train_df = train_df.copy()
    train_df["doc"] = train_df.apply(build_doc_row, axis=1)

    # BM25 corpus
    bm25_corpus_texts  = train_df["doc"].tolist()
    bm25_corpus_tokens = [simple_tokenize(t) for t in bm25_corpus_texts]
    bm25 = BM25Okapi(bm25_corpus_tokens)

    # Dense embeddings + FAISS
    dense = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO", device=device)
    corpus_texts = train_df["doc"].astype(str).tolist()
    dense_emb = dense.encode(corpus_texts, batch_size=128, normalize_embeddings=True,
                            show_progress_bar=True).astype("float32")
    dim = dense_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(dense_emb)
    _dense_rows = train_df.reset_index(drop=True)

    # Save artifacts
    save_retrieval_artifacts(train_df, bm25, bm25_corpus_tokens, index, dense_emb)
    print("Saved retrieval artifacts to", SAVE_DIR)
    return train_df, bm25, bm25_corpus_tokens, dense, index, _dense_rows


# =============================
# Search functions (with explicit parameters - NO GLOBALS)
# =============================
def bm25_search(query: str, bm25, train_df, k: int = 20):
    """BM25 search - requires bm25 and train_df to be passed in."""
    q_tokens = simple_tokenize(query)
    scores = bm25.get_scores(q_tokens)
    top_idx = np.argsort(scores)[::-1][:k]
    out = train_df.iloc[top_idx].copy()
    out["bm25_score"] = np.asarray(scores)[top_idx]
    return out

def dense_search(query: str, dense, index, _dense_rows, k: int = 200):
    """Dense search - requires dense model, index, and _dense_rows to be passed in."""
    qv = dense.encode([query], normalize_embeddings=True).astype("float32")
    scores, idx = index.search(qv, k)
    out = _dense_rows.iloc[idx[0]].copy()
    out["dense_score"] = scores[0]
    return out

def hybrid_search(query: str, bm25, train_df, dense, index, _dense_rows,
                  k_bm25=200, k_dense=200, k_final=50, alpha=0.5):
    """Hybrid search combining BM25 and dense retrieval."""
    bm = bm25_search(query, bm25, train_df, k=k_bm25)[["question","answer","source","url","bm25_score"]].copy()
    de = dense_search(query, dense, index, _dense_rows, k=k_dense)[["question","answer","source","url","dense_score"]].copy()

    merged = bm.merge(de, how="outer", on=["question","answer","source","url"])
    merged["bm25_score"]  = merged["bm25_score"].fillna(0.0)
    merged["dense_score"] = merged["dense_score"].fillna(0.0)

    merged["bm25_s"]  = minmax_scale(merged["bm25_score"].to_numpy(dtype=float), copy=True)
    merged["dense_s"] = minmax_scale(merged["dense_score"].to_numpy(dtype=float), copy=True)
    merged["hybrid"]  = (1 - alpha) * merged["bm25_s"] + alpha * merged["dense_s"]

    return merged.sort_values("hybrid", ascending=False).head(k_final).reset_index(drop=True)

# =============================
# Cross-encoder re-ranker
# =============================
RERANK_NAME = "ncbi/MedCPT-Cross-Encoder"
print(f"Using device: {device}")

# Lazy tokenizer and cross-encoder loader to avoid heavy import-time work
_ce_tok = None
def get_ce_tok(rerank_name: str = RERANK_NAME):
    """Lazy-load and cache the cross-encoder tokenizer."""
    global _ce_tok
    if _ce_tok is None:
        _ce_tok = AutoTokenizer.from_pretrained(rerank_name, model_max_length=512, truncation_side="right")
    return _ce_tok

_ce_model = None
def get_ce_model(rerank_name: str = RERANK_NAME):
    """Lazy-load and cache the sequence-classification model on the selected device."""
    global _ce_model
    if _ce_model is None:
        _ce_model = AutoModelForSequenceClassification.from_pretrained(rerank_name).to(device).eval()
    return _ce_model

def clip_passage(text: str, max_passage_tokens: int = 400) -> str:
    tok = get_ce_tok()
    ids = tok.encode(text or "", add_special_tokens=False)[:max_passage_tokens]
    return tok.decode(ids, skip_special_tokens=True)

@torch.inference_mode()
def rerank_cross_encoder(query: str, df_candidates: pd.DataFrame, top_n: int = 8,
                        batch_size: int = 16, max_length: int = 512, max_passage_tokens: int = 400):
    if df_candidates.empty:
        return df_candidates
    q = str(query or "")
    d_texts = [clip_passage(str(a), max_passage_tokens=max_passage_tokens)
            for a in df_candidates["answer"].astype(str).tolist()]
    scores = []
    for i in range(0, len(d_texts), batch_size):
        q_batch = [q]*len(d_texts[i:i+batch_size])
        d_batch = d_texts[i:i+batch_size]
        tok = get_ce_tok()
        enc = tok(q_batch, d_batch, padding="max_length", truncation="only_second",
                  max_length=max_length, return_tensors="pt")
        # move tensors to device
        enc = {k: v.to(device) for k, v in enc.items()}
        model = get_ce_model()
        logits = model(**enc).logits.squeeze(-1)
        scores.extend(logits.detach().cpu().tolist())
    out = df_candidates.copy()
    out["ce_score"] = scores
    return out.sort_values("ce_score", ascending=False).head(top_n).reset_index(drop=True)

@torch.inference_mode()
def load_generator(model_dir: str):
    """
    Loads a generator from a local directory or HuggingFace Hub.
    """
    # Check if it's a local path
    is_local = os.path.exists(model_dir)
    
    if is_local:
        print(f"Loading from local path: {model_dir}")
        tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        gen = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
    else:
        print(f"Loading from HuggingFace Hub: {model_dir}")
        tok = AutoTokenizer.from_pretrained(model_dir)
        gen = AutoModelForCausalLM.from_pretrained(model_dir)
    
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    gen.config.pad_token_id = tok.pad_token_id
    gen = gen.to(device).eval()
    return tok, gen

# ====== No RAG (Baseline 1) ======
@torch.inference_mode()
def generate_no_rag(user_q: str, tok, gen_model,
                    max_new_tokens: int = 150,
                    repetition_penalty: float = 1.1,
                    no_repeat_ngram_size: int = 3) -> dict:
    """
    Baseline 1: Finetuned BioGPT (no retrieval).
    The model relies only on what it learned during fine-tuning.
    """
    system = (
        "You are a careful medical information assistant for the general public.\n"
        "- Use short sentences and plain language.\n"
        "- Avoid diagnosis; give general guidance and next steps.\n"
        "- If you don't know, say you don't know.\n"
    )
    prompt = f"{system}\nQuestion: {user_q.strip()}\nAnswer:"

    # Tokenize safely
    inputs = tok(prompt, return_tensors="pt",
                 truncation=True, max_length=512).to(device)

    out_ids = gen_model.generate(
        **inputs,
        do_sample=False,  # greedy decoding
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=True
    )

    text = tok.decode(out_ids[0], skip_special_tokens=True)
    ans = text.split("Answer:", 1)
    ans = (ans[1] if len(ans) > 1 else text).strip()
    return {"answer": ans, "evidence": None}

# ====== RAG (Baseline 2 + Full Model) ======

SYSTEM_RAG_FINE = (
    "You are a careful medical information assistant for the general public.\n"
    "- Answer ONLY using the Context bullets; if missing, say you don't know.\n"
    "- Use short sentences and plain language. Avoid diagnosis; give general guidance and next steps.\n"
    "- Add bracketed numeric citations [1], [2] that refer to the bullets you used.\n"
)

SYSTEM_RAG_BASE = (
    "Answer the following medical question briefly using only the context below. "
    "If the context does not contain the answer, reply: 'I don't know.'\n"
)


def build_rag_prompt(user_q: str,
                    reranked_df: pd.DataFrame,
                    tok,
                    ctx_token_budget: int = 700,
                    system_text: str = SYSTEM_RAG_FINE) -> str:
    """
    Build a concise RAG prompt with a safe token budget.
    """
    bullets, used = [], 0
    for i, (_, r) in enumerate(reranked_df.iterrows(), start=1):
        block = f"[{i}] {str(r['answer']).strip()}\n" \
                f"(Source: {(r.get('source') or 'Unknown').strip()} {(r.get('url') or '').strip()})"
        ids = tok.encode(block, add_special_tokens=False)
        if used + len(ids) > ctx_token_budget:
            break
        bullets.append(block)
        used += len(ids)

    ctx = "\n".join(bullets)
    return (
        f"{system_text}\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {user_q.strip()}\n"
        f"Answer:"
    )


@torch.inference_mode()
def generate_with_rag(
    user_q: str,
    tok,
    gen_model,
    bm25,          
    train_df,      
    dense,         
    index,         
    _dense_rows,   
    use_base_prompt: bool = False,
    k_bm25=200, k_dense=200, k_final=50, alpha=0.5,
    top_n=6,
    max_new_tokens=150,
    do_sample=False,
    top_p=1.0,
    repetition_penalty=1.1,
    no_repeat_ngram_size=3
) -> dict:
    """
    Full RAG pipeline for Baseline 2 or Full Model.
    Returns: {"answer": str, "evidence": pd.DataFrame[answer, source, url, ce_score]}
    """
    # --- 1. Hybrid retrieval ---
    cand = hybrid_search(
        user_q, 
        bm25, 
        train_df, 
        dense, 
        index, 
        _dense_rows,
        k_bm25=k_bm25,
        k_dense=k_dense, 
        k_final=k_final, 
        alpha=alpha
    )

    # --- 2. Cross-encoder re-ranking ---
    rer = rerank_cross_encoder(
        user_q, cand, top_n=top_n,
        batch_size=16, max_length=512, max_passage_tokens=400
    )

    # --- 3. Build RAG prompt ---
    system_text = SYSTEM_RAG_BASE if use_base_prompt else SYSTEM_RAG_FINE
    prompt = build_rag_prompt(user_q, rer, tok=tok,
                            ctx_token_budget=700, system_text=system_text)

    # --- 4. Tokenize and Generate ---
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Store prompt length to extract only new tokens
    prompt_length = inputs["input_ids"].shape[1]

    gen_kwargs = dict(
        do_sample=do_sample,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=True
    )

    out_ids = gen_model.generate(**inputs, **gen_kwargs)
    
    # --- 5. Extract ONLY the generated tokens (exclude prompt) ---
    generated_ids = out_ids[0][prompt_length:]
    text = tok.decode(generated_ids, skip_special_tokens=True).strip()
    
    # --- 6. Clean up the answer (minimal, non-destructive) ---
    ans = text
    
    # Remove only obvious prefixes (be conservative)
    prefixes_to_remove = ["Answer:", "answer:", "A:", "a:"]
    for prefix in prefixes_to_remove:
        if ans.startswith(prefix):
            ans = ans[len(prefix):].strip()
            break  # Only remove first match
    
    # Collapse multiple spaces and newlines
    ans = " ".join(ans.split())
    
    # --- 7. Fallback only if truly empty ---
    if not ans or len(ans) < 5:
        # Try full decode as last resort
        full_text = tok.decode(out_ids[0], skip_special_tokens=True)
        if "Answer:" in full_text:
            ans = full_text.split("Answer:", 1)[1].strip()
            ans = " ".join(ans.split())
        else:
            ans = text  # Use original if nothing works

    # --- 8. Extract evidence ---
    cols = [c for c in ["answer", "source", "url", "ce_score"] if c in rer.columns]
    used = rer[cols].copy() if cols else pd.DataFrame(
        columns=["answer", "source", "url", "ce_score"]
    )

    return {"answer": ans, "evidence": used}