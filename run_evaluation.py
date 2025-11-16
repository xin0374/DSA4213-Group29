import os
import pandas as pd
from functions import generate_no_rag, generate_with_rag, load_generator, load_retrieval_artifacts
from evaluation import evaluate_model

# Paths
VAL_CSV = "./data/medquad_val.csv"
SAVE_DIR_RESULTS = "./results"
os.makedirs(SAVE_DIR_RESULTS, exist_ok=True)

# Load models
# Convert to absolute paths
BASE_MODEL_DIR = os.path.abspath("./biogpt_base")
FULL_MODEL_DIR = os.path.abspath("./finetuned_biogpt")

print(f"Base model path: {BASE_MODEL_DIR}")
print(f"Full model path: {FULL_MODEL_DIR}")

# Verify paths exist
if not os.path.exists(BASE_MODEL_DIR):
    raise FileNotFoundError(f"Base model not found at: {BASE_MODEL_DIR}")
if not os.path.exists(FULL_MODEL_DIR):
    raise FileNotFoundError(f"Finetuned model not found at: {FULL_MODEL_DIR}")

print("Loading retrieval artifacts...")
train_df, bm25, bm25_corpus_tokens, dense, index, _dense_rows = load_retrieval_artifacts()
print("Retrieval artifacts loaded!")

print("Loading models...")
tok_base, gen_base = load_generator(BASE_MODEL_DIR)
tok_full, gen_full = load_generator(FULL_MODEL_DIR)
print("Models loaded successfully!")

val_df = pd.read_csv(VAL_CSV)

# --- Baseline 1 (finetuned BioGPT only) ---
print("\n=== Evaluating Baseline 1 ===")
gen_fn_baseline1 = lambda q: generate_no_rag(q, tok_full, gen_full)
baseline1_results = evaluate_model(
    val_df, 
    gen_fn_baseline1, 
    "Baseline 1 – Finetuned BioGPT", 
    os.path.join(SAVE_DIR_RESULTS, "baseline1_val.csv")
)
summary_df1 = pd.DataFrame([
    {"Model": "Baseline 1 (Fine-tuned only)", **baseline1_results[0]}
])
summary_df1.drop(columns=["N"], inplace=True, errors="ignore")
summary_df1.to_csv(os.path.join(SAVE_DIR_RESULTS, "baseline1_model.csv"), index=False)

# --- Baseline 2 (RAG + Base BioGPT) ---
print("\n=== Evaluating Baseline 2 ===")
gen_fn_baseline2 = lambda q: generate_with_rag(
    q, tok_base, gen_base, bm25, train_df, dense, index, _dense_rows
)
baseline2_results = evaluate_model(
    val_df, 
    gen_fn_baseline2, 
    "Baseline 2 – RAG + Base BioGPT", 
    os.path.join(SAVE_DIR_RESULTS, "baseline2_val.csv")
)
summary_df2 = pd.DataFrame([
    {"Model": "Baseline 2 (RAG + Base BioGPT)", **baseline2_results[0]}
])
summary_df2.drop(columns=["N"], inplace=True, errors="ignore")
summary_df2.to_csv(os.path.join(SAVE_DIR_RESULTS, "baseline2_model.csv"), index=False)

# --- Full Model (RAG + finetuned BioGPT) ---
print("\n=== Evaluating Full Model ===")
gen_fn_full = lambda q: generate_with_rag(
    q, tok_full, gen_full, bm25, train_df, dense, index, _dense_rows
)
fullmodel_results = evaluate_model(
    val_df, 
    gen_fn_full, 
    "Full Model – RAG + Finetuned BioGPT", 
    os.path.join(SAVE_DIR_RESULTS, "fullmodel_val.csv")
)
summary_df_full = pd.DataFrame([
    {"Model": "Full Model (RAG + Finetuned BioGPT)", **fullmodel_results[0]}
])
summary_df_full.drop(columns=["N"], inplace=True, errors="ignore")
summary_df_full.to_csv(os.path.join(SAVE_DIR_RESULTS, "fullmodel_model.csv"), index=False)

print("\nEvaluation completed and CSVs saved.")