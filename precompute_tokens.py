"""
Creates parquet files with retrieval contexts for each question.
"""
import os
import tempfile
import hashlib
import pandas as pd
from tqdm import tqdm
from functions import (
    load_retrieval_artifacts,
    hybrid_search,
    rerank_cross_encoder,
    bm25_search
)

# Load retrieval artifacts once
print("Loading retrieval artifacts:")
train_df, bm25, bm25_corpus_tokens, dense, index, _dense_rows = load_retrieval_artifacts()
print("Retrieval artifacts loaded successfully")

def _qid(text: str) -> str:
    """Stable ID for a question (case/whitespace-insensitive)"""
    t = (text or "").strip().lower()
    return hashlib.md5(t.encode("utf-8")).hexdigest()

def _atomic_save_parquet(df: pd.DataFrame, out_path: str):
    dir_ = os.path.dirname(out_path) or "."
    with tempfile.NamedTemporaryFile(delete=False, dir=dir_, suffix=".parquet") as tf:
        tmp_path = tf.name
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, out_path)

def build_context_block(rer_df: pd.DataFrame) -> str:
    """Build numbered context block from reranked results"""
    bullets = []
    for i, (_, r) in enumerate(rer_df.iterrows(), start=1):
        answer = str(r.get("answer", "")).strip()
        source = str(r.get("source", "Unknown")).strip()
        url = str(r.get("url", "")).strip()
        bullets.append(f"[{i}] {answer}\n(Source: {source} {url})")
    return "\n".join(bullets)

def precompute_contexts_resume(
    df: pd.DataFrame, 
    out_path: str,
    k_bm25=200, 
    k_dense=200, 
    k_final=50, 
    alpha=0.5,
    top_n=8, 
    batch_size=16, 
    max_length=512, 
    max_passage_tokens=400,
    checkpoint_every=500
) -> pd.DataFrame:
    """
    Resume-safe precompute:
      - If out_path exists, load and skip already processed questions
      - Checkpoint every `checkpoint_every` rows 
    """
    # Load prior progress if exists
    if os.path.exists(out_path):
        done_df = pd.read_parquet(out_path)
        if "qid" not in done_df.columns:
            done_df["qid"] = done_df["question"].map(_qid)
        done_qids = set(done_df["qid"].tolist())
        print(f"[resume] Loaded {len(done_df)} rows from {out_path}")
    else:
        done_df = pd.DataFrame(columns=["qid", "question", "answer", "context_block"])
        done_qids = set()

    # Prepare input with qids
    df = df.copy().reset_index(drop=True)
    df["qid"] = df["question"].map(_qid)

    # Filter remaining
    todo = df[~df["qid"].isin(done_qids)].reset_index(drop=True)
    print(f"[resume] Remaining to process: {len(todo)}")

    buffer_rows = []
    processed = 0

    for _, row in tqdm(todo.iterrows(), total=len(todo), desc=f"Precompute → {os.path.basename(out_path)}"):
        q = str(row["question"])
        a_gold = str(row["answer"])
        qid = row["qid"]

        try:
            # Perform hybrid search
            cand = hybrid_search(
                q, bm25, train_df, dense, index, _dense_rows,
                k_bm25=k_bm25, k_dense=k_dense, k_final=k_final, alpha=alpha
            )
            
            # Avoid label leakage: do not include the exact gold answer among candidates
            cand = cand[cand["answer"].astype(str) != a_gold]
            
            # Re-rank candidates
            rer = rerank_cross_encoder(
                q, cand, top_n=top_n, batch_size=batch_size,
                max_length=max_length, max_passage_tokens=max_passage_tokens
            )
            
            if rer.empty:
                # Final fallback
                cfb = bm25_search(q, bm25, train_df, k=top_n)
                ctx_block = build_context_block(cfb)
            else:
                ctx_block = build_context_block(rer)
                
        except Exception as e:
            print(f"Error processing question: {e}")
            # Robust fallback path
            cfb = bm25_search(q, bm25, train_df, k=top_n)
            ctx_block = build_context_block(cfb)

        buffer_rows.append({
            "qid": qid,
            "question": q,
            "answer": a_gold,
            "context_block": ctx_block
        })
        processed += 1

        # Periodic checkpoint
        if processed % checkpoint_every == 0:
            # Merge with previously done and save atomically
            out_df = pd.concat([done_df, pd.DataFrame(buffer_rows)], ignore_index=True)
            # De-dup by qid (keep first written)
            out_df = out_df.drop_duplicates(subset=["qid"], keep="first")
            _atomic_save_parquet(out_df, out_path)
            print(f"[checkpoint] Saved {len(out_df)} rows to {out_path} (processed +{len(buffer_rows)})")
            # Reset buffer, refresh done_df/done_qids
            done_df = out_df
            done_qids = set(done_df["qid"].tolist())
            buffer_rows = []

    # Final save for any remaining buffer
    if buffer_rows:
        out_df = pd.concat([done_df, pd.DataFrame(buffer_rows)], ignore_index=True)
        out_df = out_df.drop_duplicates(subset=["qid"], keep="first")
        _atomic_save_parquet(out_df, out_path)
        print(f"[final] Saved {len(out_df)} rows to {out_path} (+{len(buffer_rows)} new)")

    # Return the fully merged dataframe (freshly loaded to be safe)
    final_df = pd.read_parquet(out_path)
    # Ensure expected columns
    for col in ["question", "answer", "context_block"]:
        if col not in final_df.columns:
            final_df[col] = ""
    return final_df


if __name__ == "__main__":
    # Load train and validation datasets
    train_data = pd.read_csv("./data/medquad_train.csv")
    val_data = pd.read_csv("./data/medquad_val.csv")
    
    print("\n=== Processing Training Set ===")
    train_sft_df = precompute_contexts_resume(
        train_data, 
        "train_sft_with_context.parquet",
        k_bm25=200, 
        k_dense=200, 
        k_final=50, 
        alpha=0.5,
        top_n=8, 
        batch_size=16, 
        max_length=512, 
        max_passage_tokens=400,
        checkpoint_every=500
    )
    
    print("\n=== Processing Validation Set ===")
    val_sft_df = precompute_contexts_resume(
        val_data, 
        "val_sft_with_context.parquet",
        k_bm25=200, 
        k_dense=200, 
        k_final=50, 
        alpha=0.5,
        top_n=8, 
        batch_size=16, 
        max_length=512, 
        max_passage_tokens=400,
        checkpoint_every=250
    )
    
    print("\n=== Precomputation Complete ===")
    print(f"Train: {train_sft_df.shape}")
    print(f"Val: {val_sft_df.shape}")