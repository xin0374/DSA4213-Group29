"""
Step 2: Fine-tune BioGPT with LoRA on precomputed contexts
This trains the model to generate medical answers from retrieved context
"""
import torch
import pandas as pd
from typing import List, Dict
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model

# Configuration
BASE_MODEL = "microsoft/biogpt"
OUTPUT_DIR = "biogpt-medquad-lora"
TRAIN_PARQUET = "train_sft_with_context.parquet"
VAL_PARQUET = "val_sft_with_context.parquet"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====================
# 1. Load Precomputed Data
# ====================
print("\n=== Loading Precomputed Contexts ===")
train_sft_df = pd.read_parquet(TRAIN_PARQUET)
val_sft_df = pd.read_parquet(VAL_PARQUET)
print(f"Train samples: {len(train_sft_df)}")
print(f"Val samples: {len(val_sft_df)}")

# ====================
# 2. Format Examples
# ====================
def format_example(row):
    """Format a training example with prompt and target answer."""
    prompt = (
        "You are a careful medical assistant.\n"
        "Answer ONLY using the Context. Keep it simple for the general public. "
        "If information is missing, say you don't know. Cite sources inline as [1], [2] in order.\n\n"
        f"Context:\n{row['context_block']}\n\n"
        f"Question: {row['question']}\n"
        "Answer:"
    )
    return {"prompt": prompt, "target": row["answer"].strip()}

print("\n=== Formatting Examples ===")
train_fmt = train_sft_df.apply(format_example, axis=1, result_type="expand")
val_fmt = val_sft_df.apply(format_example, axis=1, result_type="expand")

train_ds = Dataset.from_pandas(train_fmt)
val_ds = Dataset.from_pandas(val_fmt)
print(f"Train dataset: {len(train_ds)} examples")
print(f"Val dataset: {len(val_ds)} examples")

# ====================
# 3. Load Tokenizer
# ====================
print("\n=== Loading Tokenizer ===")
tok_g = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok_g.pad_token is None:
    tok_g.pad_token = tok_g.eos_token
print(f"Tokenizer loaded. Vocab size: {len(tok_g)}")

# ====================
# 4. Tokenization Function with Label Masking
# ====================
def tokenize(batch):
    """
    Tokenize prompts and targets with label masking.
    Only the target (answer) portion is used for loss computation.
    """
    prompts = batch["prompt"]
    targets = batch["target"]
    
    # Tokenize prompts
    model_inputs = tok_g(prompts, padding=True, truncation=True, max_length=512)
    
    # Tokenize targets
    with tok_g.as_target_tokenizer():
        labels = tok_g(targets, padding=True, truncation=True, max_length=256)
    
    input_ids, label_ids = [], []
    for x_ids, y_ids in zip(model_inputs["input_ids"], labels["input_ids"]):
        # Concatenate prompt + target + EOS
        ids = x_ids + y_ids + [tok_g.eos_token_id]
        # Mask prompt tokens with -100 (ignored in loss)
        labs = [-100] * len(x_ids) + y_ids + [tok_g.eos_token_id]
        
        input_ids.append(ids)
        label_ids.append(labs)
    
    return {"input_ids": input_ids, "labels": label_ids}

print("\n=== Tokenizing Datasets ===")
train_tok = train_ds.map(
    tokenize, 
    batched=True, 
    remove_columns=train_ds.column_names,
    desc="Tokenizing train"
)
val_tok = val_ds.map(
    tokenize, 
    batched=True, 
    remove_columns=val_ds.column_names,
    desc="Tokenizing val"
)
print("Tokenization complete!")

# ====================
# 5. Load Base Model and Apply LoRA
# ====================
print("\n=== Loading Base Model ===")
# Enable gradients for training
torch.set_grad_enabled(True)

model_g_base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

def guess_lora_targets(model):
    """Automatically detect LoRA target modules based on model architecture."""
    names = [n for n, _ in model.named_modules()]
    # GPT-2/BioGPT common module names
    if any("attn.c_attn" in n for n in names):
        return ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
    # Fallback to BERT-ish names
    return ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj", "fc1", "fc2"]

print("\n=== Applying LoRA Configuration ===")
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    target_modules=guess_lora_targets(model_g_base)
)

model_g = get_peft_model(model_g_base, lora_cfg)
model_g.print_trainable_parameters()

# ====================
# 6. Custom Data Collator (Padding-Aware)
# ====================
pad_id = tok_g.pad_token_id

def _to_list(x):
    """Convert tensor to list if needed."""
    if isinstance(x, torch.Tensor):
        return x.tolist()
    return list(x)

def lm_pad_collator(features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
    """
    Collate function that pads inputs and labels dynamically.
    Ensures labels are masked with -100 for padded positions.
    """
    # Ensure everything is plain lists
    input_ids_list = [_to_list(f["input_ids"]) for f in features]
    labels_list = [_to_list(f["labels"]) for f in features]
    
    # Find max length across batch
    max_len = max(len(ids) for ids in input_ids_list)
    
    input_ids, attention_mask, labels = [], [], []
    for ids, labs in zip(input_ids_list, labels_list):
        pad_len = max_len - len(ids)
        if pad_len < 0:
            # Defensive: shouldn't happen
            ids = ids[:max_len]
            labs = labs[:max_len]
            pad_len = 0
        
        input_ids.append(ids + [pad_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
        labels.append(labs + [-100] * pad_len)  # Ignore padded positions in loss
    
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

# ====================
# 7. Training Configuration
# ====================
print("\n=== Setting Up Training Configuration ===")
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,     
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,      
    learning_rate=2e-4,
    num_train_epochs=3,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,                 
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=torch.cuda.is_available(),
    bf16=False,
    report_to="none",                   
    push_to_hub=False,
)

# ====================
# 8. Initialize Trainer
# ====================
trainer = Trainer(
    model=model_g,
    args=args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=lm_pad_collator
)

# ====================
# 9. Train the Model
# ====================
print("\n=== Starting Training ===")
model_g.train()
train_out = trainer.train()  
print("\n=== Training Complete ===")
print(train_out)

# ====================
# 10. Save Final Model
# ====================
print("\n=== Saving Model ===")
# Save LoRA adapters (small files)
model_g.save_pretrained(OUTPUT_DIR)
tok_g.save_pretrained(OUTPUT_DIR)
print(f"Saved LoRA adapters and tokenizer to {OUTPUT_DIR}")