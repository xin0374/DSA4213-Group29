# evaluation.py
import nltk
nltk.download('punkt')
nltk.download('stopwords')

import string
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction
from nltk.corpus import stopwords
import evaluate
from typing import Tuple, Dict

# ----------------------------
# Evaluation setup
# ----------------------------
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
stop_words = set(stopwords.words("english"))
smooth = SmoothingFunction().method3

# ----------------------------
# Text normalization & metrics
# ----------------------------
def normalize_text(s: str) -> str:
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = " ".join(s.split())
    return s

def exact_match(pred: str, gold: str) -> int:
    return int(normalize_text(pred) == normalize_text(gold))

def token_f1(pred: str, gold: str) -> float:
    pt = normalize_text(pred).split()
    gt = normalize_text(gold).split()
    common = set(pt) & set(gt)
    num_same = sum(min(pt.count(w), gt.count(w)) for w in common)
    if not pt or not gt:
        return float(pt == gt)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pt)
    recall    = num_same / len(gt)
    return 2 * precision * recall / (precision + recall)

def rouge_l(preds, refs) -> float:
    out = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    return float(out["rougeL"])

def content_tokens(s: str):
    toks = normalize_text(s).split()
    return [t for t in toks if (t not in stop_words) and (t.isalpha())]

def support_ratio(pred: str, contexts: list[str]) -> float:
    pred_ct = content_tokens(pred)
    if not pred_ct:
        return 1.0
    ctx = " ".join(contexts)
    ctx_set = set(content_tokens(ctx))
    supported = sum(1 for t in pred_ct if t in ctx_set)
    return supported / max(1, len(pred_ct))

def hallucinated(pred: str, contexts: list[str], thresh: float = 0.6) -> int:
    return int(support_ratio(pred, contexts) < thresh)

def bert_score(preds, refs, lang="en", model_type="bert-base-uncased"):
    result = bertscore.compute(
        predictions=preds,
        references=refs,
        lang=lang,
        model_type=model_type
    )
    return result

# ----------------------------
# Unified evaluation function
# ----------------------------
def evaluate_model(
    eval_df: pd.DataFrame,
    generation_fn,
    name: str,
    save_path: str = None
) -> Tuple[Dict, pd.DataFrame]:
    preds, refs = [], []
    em_list, f1_list = [], []
    supp_list, hall_list = [], []
    rows_out = []

    print(f"\n=== Evaluating {name} on {len(eval_df)} samples ===")

    for _, row in eval_df.iterrows():
        q = str(row["question"])
        gold = str(row["answer"])

        try:
            out = generation_fn(q)
            pred = str(out.get("answer", "") or "").strip()
            ev_df = out.get("evidence", None)
        except Exception as e:
            print(f"[WARN] Generation failed for: {q[:60]}... ({e})")
            pred, ev_df = "", None

        ctxs = []
        if ev_df is not None:
            try:
                if hasattr(ev_df, "empty") and not ev_df.empty and "answer" in ev_df.columns:
                    ctxs = ev_df["answer"].astype(str).tolist()
            except Exception:
                ctxs = []

        preds.append(pred)
        refs.append(gold)
        em_list.append(exact_match(pred, gold))
        f1_list.append(token_f1(pred, gold))
        supp_list.append(support_ratio(pred, ctxs))
        hall_list.append(hallucinated(pred, ctxs, thresh=0.6))

        def _safe_ev(i):
            try:
                return ev_df.iloc[i]["answer"] if ev_df is not None and len(ev_df) > i and "answer" in ev_df.columns else ""
            except Exception:
                return ""

        rows_out.append({
            "question": q,
            "gold_answer": gold,
            "prediction": pred,
            "EM": em_list[-1],
            "F1": f1_list[-1],
            "SupportRatio": supp_list[-1],
            "Hallucinated": hall_list[-1],
            "evidence_1": _safe_ev(0),
            "evidence_2": _safe_ev(1),
            "evidence_3": _safe_ev(2),
        })

    # Aggregate metrics
    rouge_l_f = rouge_l(preds, refs)
    any_pred_text = any(bool(p.strip()) for p in preds)
    if any_pred_text:
        bert_res = bert_score(preds, refs)
        bert_p = float(np.mean(bert_res.get("precision", [0.0])))
        bert_r = float(np.mean(bert_res.get("recall", [0.0])))
        bert_f = float(np.mean(bert_res.get("f1", [0.0])))
    else:
        bert_p = bert_r = bert_f = 0.0

    results = {
        "N": len(eval_df),
        "ExactMatch": float(np.mean(em_list)) if em_list else 0.0,
        "TokenF1": float(np.mean(f1_list)) if f1_list else 0.0,
        "ROUGE-L": float(rouge_l_f),
        "SupportRatio(avg)": float(np.mean(supp_list)) if supp_list else 0.0,
        "HallucinationRate(<0.6 support)": float(np.mean(hall_list)) if hall_list else 0.0,
        "BERTScore_P": bert_p,
        "BERTScore_R": bert_r,
        "BERTScore_F1": bert_f,
    }

    details = pd.DataFrame(rows_out)
    if save_path:
        try:
            details.to_csv(save_path, index=False)
            print(f"Saved per-sample details to {save_path}")
        except Exception as e:
            print(f"[WARN] Failed to save CSV to {save_path}: {e}")

    print(f"\n=== {name} Summary ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    return results, details
