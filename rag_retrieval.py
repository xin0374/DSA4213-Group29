import os
import re
import pickle
import numpy as np
import pandas as pd
import torch
import faiss

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import minmax_scale
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"

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
SAVE_DIR = "retrieval_artifacts"
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
# Search functions
# =============================
def bm25_search(query: str, k: int = 20):
    q_tokens = simple_tokenize(query)
    scores = bm25.get_scores(q_tokens)
    top_idx = np.argsort(scores)[::-1][:k]
    out = train_df.iloc[top_idx].copy()
    out["bm25_score"] = np.asarray(scores)[top_idx]
    return out

def dense_search(query: str, k: int = 200):
    qv = dense.encode([query], normalize_embeddings=True).astype("float32")
    scores, idx = index.search(qv, k)
    out = _dense_rows.iloc[idx[0]].copy()
    out["dense_score"] = scores[0]
    return out

def hybrid_search(query: str, k_bm25=200, k_dense=200, k_final=50, alpha=0.5):
    bm = bm25_search(query, k=k_bm25)[["question","answer","source","url","bm25_score"]].copy()
    de = dense_search(query, k=k_dense)[["question","answer","source","url","dense_score"]].copy()

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
ce_tok = AutoTokenizer.from_pretrained(RERANK_NAME, model_max_length=512, truncation_side="right")
ce_model = AutoModelForSequenceClassification.from_pretrained(RERANK_NAME).to(device).eval()

def clip_passage(text: str, max_passage_tokens: int = 400) -> str:
    ids = ce_tok.encode(text or "", add_special_tokens=False)[:max_passage_tokens]
    return ce_tok.decode(ids, skip_special_tokens=True)

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
        enc = ce_tok(q_batch, d_batch, padding="max_length", truncation="only_second",
                     max_length=max_length, return_tensors="pt").to(device)
        logits = ce_model(**enc).logits.squeeze(-1)
        scores.extend(logits.detach().cpu().tolist())
    out = df_candidates.copy()
    out["ce_score"] = scores
    return out.sort_values("ce_score", ascending=False).head(top_n).reset_index(drop=True)

# ADDITIONAL
print("Loading retrieval artifacts...")
train_df, bm25, bm25_corpus_tokens, dense, index, _dense_rows = load_retrieval_artifacts()
print("Retrieval artifacts loaded.")
