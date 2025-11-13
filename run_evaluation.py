# run_evaluation.py
import os
import pandas as pd
from functions import generate_no_rag, generate_with_rag, tok_full, gen_full, tok_base, gen_base
from evaluation import evaluate_model

# Paths
VAL_CSV = "./data/medquad_val.csv"
SAVE_DIR_RESULTS = "./results"
os.makedirs(SAVE_DIR_RESULTS, exist_ok=True)

val_df = pd.read_csv(VAL_CSV)

# --- Baseline 1 (finetuned BioGPT only) ---
gen_fn_baseline1 = lambda q: generate_no_rag(q, tok_full, gen_full)
baseline1_results = evaluate_model(val_df, gen_fn_baseline1, "Baseline 1 – Finetuned BioGPT", os.path.join(SAVE_DIR_RESULTS, "baseline1_val.csv"))

summary_df1 = pd.DataFrame([
    {"Model": "Baseline 1 (Fine-tuned only)", **baseline1_results[0]}
])
summary_df1.drop(columns=["N"], inplace=True, errors="ignore")
summary_df1.to_csv(os.path.join(SAVE_DIR_RESULTS, "baseline1_model.csv"), index=False)

# --- Baseline 2 (RAG + Base BioGPT) ---
gen_fn_baseline2 = lambda q: generate_with_rag(q, tok_base, gen_base)
baseline2_results = evaluate_model(val_df, gen_fn_baseline2, "Baseline 2 – RAG + Base BioGPT", os.path.join(SAVE_DIR_RESULTS, "baseline2_val.csv"))

summary_df2 = pd.DataFrame([
    {"Model": "Baseline 2 (RAG + Base BioGPT)", **baseline2_results[0]}
])
summary_df2.drop(columns=["N"], inplace=True, errors="ignore")
summary_df2.to_csv(os.path.join(SAVE_DIR_RESULTS, "baseline2_model.csv"), index=False)

# --- Full Model (RAG + finetuned BioGPT) ---
gen_fn_full = lambda q: generate_with_rag(q, tok_full, gen_full)
fullmodel_results = evaluate_model(val_df, gen_fn_full, "Full Model – RAG + Finetuned BioGPT", os.path.join(SAVE_DIR_RESULTS, "fullmodel_val.csv"))

summary_df_full = pd.DataFrame([
    {"Model": "Full Model (RAG + Finetuned BioGPT)", **fullmodel_results[0]}
])
summary_df_full.drop(columns=["N"], inplace=True, errors="ignore")
summary_df_full.to_csv(os.path.join(SAVE_DIR_RESULTS, "fullmodel_model.csv"), index=False)

print("✅ Evaluation completed and CSVs saved.")
