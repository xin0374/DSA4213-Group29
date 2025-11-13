"""
Configuration file for training pipeline
Adjust these settings based on your hardware and requirements.
"""

# ====================
# Model Configuration
# ====================
BASE_MODEL = "microsoft/biogpt"
OUTPUT_DIR = "biogpt-medquad-lora"
MERGED_OUTPUT_DIR = "biogpt-medquad-lora_merged"

# ====================
# Data Configuration
# ====================
TRAIN_PARQUET = "train_sft_with_context.parquet"
VAL_PARQUET = "val_sft_with_context.parquet"

# ====================
# Retrieval Configuration
# ====================
RETRIEVAL_CONFIG = {
    "k_bm25": 200,           # Number of BM25 candidates
    "k_dense": 200,          # Number of dense candidates
    "k_final": 50,           # Final merged candidates
    "alpha": 0.5,            # Hybrid search weight (0=BM25 only, 1=dense only)
    "top_n": 6,              # Top results after reranking
    "batch_size": 16,        # Reranking batch size
    "max_length": 512,       # Max token length for reranking
    "max_passage_tokens": 400,  # Max tokens per passage
}

# Checkpoint settings
TRAIN_CHECKPOINT_EVERY = 500
VAL_CHECKPOINT_EVERY = 250

# ====================
# LoRA Configuration
# ====================
LORA_CONFIG = {
    "r": 8,                  # LoRA rank
    "lora_alpha": 16,        # LoRA alpha
    "lora_dropout": 0.05,    # LoRA dropout
    "task_type": "CAUSAL_LM",
}

# ====================
# Training Configuration
# ====================
TRAINING_CONFIG = {
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 8,  # Effective batch = 2 * 8 = 16
    "learning_rate": 2e-4,
    "num_train_epochs": 3,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "logging_steps": 50,
    "eval_steps": 500,
    "save_steps": 500,
    "save_total_limit": 3,
    "max_grad_norm": 1.0,
}

# ====================
# Tokenization Configuration
# ====================
MAX_PROMPT_LENGTH = 512
MAX_TARGET_LENGTH = 256

# ====================
# System Prompts
# ====================
TRAINING_SYSTEM_PROMPT = (
    "You are a careful medical assistant.\n"
    "Answer ONLY using the Context. Keep it simple for the general public. "
    "If information is missing, say you don't know. Cite sources inline as [1], [2] in order."
)

INFERENCE_SYSTEM_PROMPT = (
    "You are a careful medical information assistant for the general public.\n"
    "- Answer ONLY using the Context bullets; if missing, say you don't know.\n"
    "- Use short sentences and plain language. Avoid diagnosis; give general guidance and next steps.\n"
    "- Add bracketed numeric citations [1], [2] that refer to the bullets you used."
)