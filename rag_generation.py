import sys
import torch
import pandas as pd

# Make sure Python can see rag_retrieval.py in this folder
sys.path.append(".")

from rag_retrieval import hybrid_search, rerank_cross_encoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Edited RAG prompt and generation with RAG

SYSTEM_RAG_FINE = (
    "You are a careful medical information assistant for the general public.\n"
    "- Answer ONLY using the Context bullets; if missing, say you don't know.\n"
    "- Use short sentences and plain language. Avoid diagnosis; give general guidance and next steps.\n"
    "- Add bracketed numeric citations [1], [2] that refer to the bullets you used.\n"
    "If the context does not contain the answer, reply: 'I don't know.'\n"
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
    """Build a concise RAG prompt with a safe token budget (token-capped bullets)."""
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

    # IMPORTANT: End with a hard delimiter the model can key off
    return (
        f"{system_text}\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {user_q.strip()}\n"
        f"### Answer:\n"
    )

@torch.inference_mode()
def generate_with_rag(
    user_q: str,
    tok,
    gen_model,
    use_base_prompt: bool = False,  # True if using BioGPT-base (zero-shot)
    k_bm25=200, k_dense=200, k_final=50, alpha=0.5,
    top_n=6,
    max_new_tokens=160,
    repetition_penalty=1.05,
    no_repeat_ngram_size=3
) -> dict:
    """Full RAG pipeline for Baseline 2 or Full Model."""
    # 1) Retrieval
    cand = hybrid_search(user_q, k_bm25=k_bm25, k_dense=k_dense, k_final=k_final, alpha=alpha)

    # 2) Re-ranking
    rer = rerank_cross_encoder(
        user_q, cand, top_n=top_n,
        batch_size=16, max_length=512, max_passage_tokens=400
    )

    # 3) Prompt (with delimiter)
    system_text = SYSTEM_RAG_BASE if use_base_prompt else SYSTEM_RAG_FINE
    prompt = build_rag_prompt(user_q, rer, tok=tok, ctx_token_budget=700, system_text=system_text)

    # DEBUG (optional): ensure the delimiter is present
    # print("...prompt tail:", prompt[-120:])

    # 4) Tokenize WITHOUT truncating the prompt tail
    inputs = tok(prompt, return_tensors="pt", truncation=False).to(DEVICE)

    # 5) Deterministic, stable decoding (beam search)
    gen_kwargs = dict(
        do_sample=False,
        num_beams=3,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=True,
        return_dict_in_generate=True
    )

    out = gen_model.generate(**inputs, **gen_kwargs)

    # 6) Decode ONLY continuation
    full_ids   = out.sequences[0]
    prompt_len = inputs["input_ids"].shape[1]
    gen_only   = full_ids[prompt_len:]
    raw = tok.decode(gen_only, skip_special_tokens=True)

    # 7) Minimal, non-destructive cleanup
    # If the model echoed the delimiter, cut after it
    if "### Answer:" in raw:
        raw = raw.split("### Answer:", 1)[-1]

    # Remove only obvious header echoes
    cleaned = raw.strip()
    cleaned = cleaned.replace("Answer:", "", 1).strip()  # case where it prints "Answer:" once

    # Final polish: collapse spaces
    ans = " ".join(cleaned.split())

    # 8) Fallbacks if empty (keep quality)
    if not ans:
        # Try without any cleaning
        ans = tok.decode(gen_only, skip_special_tokens=True).strip()
    if not ans:
        # One mild sampling retry to coax a sentence
        alt = gen_model.generate(
            **inputs, do_sample=True, top_p=0.92, temperature=0.7,
            max_new_tokens=max_new_tokens, repetition_penalty=1.05,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            return_dict_in_generate=True, use_cache=True
        )
        gen_only2 = alt.sequences[0][prompt_len:]
        ans = tok.decode(gen_only2, skip_special_tokens=True).strip()
        if "### Answer:" in ans:
            ans = ans.split("### Answer:", 1)[-1]
        ans = " ".join(ans.split())

    # 9) Evidence frame
    cols = [c for c in ["answer", "source", "url", "ce_score"] if c in rer.columns]
    used = rer[cols].copy() if cols else pd.DataFrame(columns=["answer", "source", "url", "ce_score"])

    return {"answer": ans, "evidence": used}
