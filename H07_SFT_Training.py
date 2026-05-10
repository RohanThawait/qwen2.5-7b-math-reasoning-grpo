import os
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
import wandb
# ── CONFIGURATION ────────────────────────────────────────────────────────────

MODEL_NAME    = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR    = "/home/devinder/gnss_phase2/project_7/evaluation/sft_model"
max_seq_length   = 1024   # GSM8K solutions are rarely longer than this 

# ── Dataset ────────────────────────────────────────────────
# NuminaMath sample size — 20000 gives good cold start coverage
# without wasting H100 time on redundant examples.
# Increase to 30000 if you want richer coverage.
numina_sample_size= 20000

# GSM8K — use the full train split (~7473 examples)
# These are the right difficulty level for a 7B cold start.
gsm8k_use_full= True

# Seed for reproducibility — keep this fixed across all runs
seed = 42

# ── Training hyperparameters ───────────────────────────────
# These are conservative for a "cold start" SFT — we're not trying to
# achieve maximum accuracy here, just teach the format. GRPO does the
# heavy lifting of actually improving reasoning.
num_train_epochs= 2 # 2 passes over the data is enough for format learning

# Per-device batch size — how many sequences per forward pass.
# H100 80GB can handle 4-8 at max_seq_length=1024.
# If you get OOM, reduce to 2.
per_device_train_batch_size= 2

# Gradient accumulation — effective_batch_size = per_device * grad_accum
# Effective batch of 16 gives stable gradients for SFT.
gradient_accumulation_steps= 8

# Learning rate — 2e-5 is conservative and stable for 7B instruct SFT.
# If loss is erratic, reduce to 1e-5.
learning_rate= 2e-5

# Cosine scheduler with warmup — do not change the scheduler type.
lr_scheduler_type="cosine"

# Warmup ratio — 3% of total steps.
# Prevents large gradient updates at training start.
warmup_ratio= 0.03

# Weight decay — light regularization to prevent overfitting.
weight_decay= 0.01

# Gradient clipping — prevents catastrophic updates from bad batches.
# 1.0 is the standard value for LLM finetuning.
max_grad_norm= 1.0

# ── Logging and saving ─────────────────────────────────────
# Log metrics every N steps
logging_steps= 10

# Save checkpoint every N steps — also saves at end of each epoch
save_steps= 200

# Keep only the best 2 checkpoints to save Drive space
save_total_limit= 2

# Evaluate on validation split every N steps
eval_steps= 200

# ── Weights & Biases ───────────────────────────────────────
wandb_project= "H07-reasoning-model"
wandb_run_name= "sft-cold-start"


wandb.login()

os.environ["WANDB_PROJECT"] = wandb_project
os.environ["WANDB_RUN_NAME"] = wandb_run_name

# ── Load tokenizer first — needed for length analysis ──────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# CRITICAL: Left-padding for causal LMs
# Right-padding puts garbage tokens before the content the model generates from
tokenizer.padding_side = "left"

# Ensure pad token is set — Qwen uses eos as pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"  pad_token set to eos_token: {tokenizer.eos_token}")

print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")


# ── Format functions ────────────────────────────────────────────────────────
# These convert each dataset example into the format your model will learn.
# The format must be IDENTICAL to what you use at inference time.
# <think>...</think> tags are what your GRPO format reward will check.

def format_numina_example(example):
    """
    NuminaMath-CoT format:
    - problem: the math question
    - solution: chain-of-thought solution
    - answer: final answer
    """
    problem = example.get("problem", "").strip()
    solution = example.get("solution", "").strip()
    answer = example.get("answer", "").strip()

    if not problem or not solution:
        return None

    # Build the response with think tags
    # This exact format is what your GRPO format reward checks
    response = f"<think>\n{solution}\n</think>\n\nThe answer is: {answer}"

    # Apply Qwen chat template
    messages = [
        {"role": "user", "content": problem},
        {"role": "assistant", "content": response},
    ]

    # add_generation_prompt=False because we're training on complete sequences
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": formatted, "source": "numina"}


def format_gsm8k_example(example):
    """
    GSM8K format:
    - question: the math word problem
    - answer: solution with #### delimiter before final number
    """
    question = example.get("question", "").strip()
    answer_full = example.get("answer", "").strip()

    if not question or not answer_full:
        return None

    # GSM8K answer format: solution steps #### final_number
    # Split into CoT and final answer
    if "####" in answer_full:
        cot_part, final_answer = answer_full.split("####", 1)
        cot_part = cot_part.strip()
        final_answer = final_answer.strip()
    else:
        cot_part = answer_full
        final_answer = ""

    response = f"<think>\n{cot_part}\n</think>\n\nThe answer is: {final_answer}"

    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
    ]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": formatted, "source": "gsm8k"}


# ── Load NuminaMath-CoT ─────────────────────────────────────────────────────
print("\nLoading NuminaMath-CoT...")
numina_raw = load_dataset("AI-MO/NuminaMath-CoT", split="train")
print(f"  Raw NuminaMath size: {len(numina_raw)} examples")

# Shuffle before sampling so we get diverse coverage
numina_raw = numina_raw.shuffle(seed=seed)
numina_raw = numina_raw.select(range(numina_sample_size))
print(f"  Sampled: {len(numina_raw)} examples")

# Format examples
numina_formatted = []
numina_skipped = 0
for ex in numina_raw:
    result = format_numina_example(ex)
    if result:
        numina_formatted.append(result)
    else:
        numina_skipped += 1

print(f"  Formatted: {len(numina_formatted)} | Skipped (empty fields): {numina_skipped}")


# ── Load GSM8K ──────────────────────────────────────────────────────────────
print("\nLoading GSM8K...")
gsm8k_raw = load_dataset("openai/gsm8k", "main", split="train")
print(f"  Raw GSM8K train size: {len(gsm8k_raw)} examples")

gsm8k_formatted = []
gsm8k_skipped = 0
for ex in gsm8k_raw:
    result = format_gsm8k_example(ex)
    if result:
        gsm8k_formatted.append(result)
    else:
        gsm8k_skipped += 1

print(f"  Formatted: {len(gsm8k_formatted)} | Skipped: {gsm8k_skipped}")


# ── Combine and length filter ───────────────────────────────────────────────
print("\nCombining datasets...")
all_examples = numina_formatted + gsm8k_formatted
print(f"  Total before length filter: {len(all_examples)}")

# Filter by tokenized length
# This prevents batches dominated by very long sequences that force padding
def get_token_length(text):
    return len(tokenizer.encode(text, add_special_tokens=False))

print("  Computing token lengths (this takes ~2 minutes)...")
filtered = []
too_short = 0
too_long = 0

for ex in all_examples:
    length = get_token_length(ex["text"])
    if length < 50:
        too_short += 1
    elif length > max_seq_length:
        too_long += 1
    else:
        filtered.append(ex)

print(f"  After length filter: {len(filtered)} examples")
print(f"  Removed (too short <50 tokens): {too_short}")
print(f"  Removed (too long >{max_seq_length} tokens): {too_long}")


# ── Build HuggingFace Dataset and split ────────────────────────────────────
dataset = Dataset.from_list(filtered)

# Shuffle the combined dataset
dataset = dataset.shuffle(seed=seed)

# 95% train, 5% validation
split = dataset.train_test_split(test_size=0.05, seed=seed)
train_dataset = split["train"]
eval_dataset = split["test"]

print(f"\nFinal dataset split:")
print(f"  Train: {len(train_dataset)} examples")
print(f"  Validation: {len(eval_dataset)} examples")

# Show source distribution in train set
sources = [ex["source"] for ex in train_dataset]
numina_count = sources.count("numina")
gsm8k_count = sources.count("gsm8k")
print(f"  NuminaMath: {numina_count} ({numina_count/len(train_dataset)*100:.1f}%)")
print(f"  GSM8K: {gsm8k_count} ({gsm8k_count/len(train_dataset)*100:.1f}%)")

# Show a sample formatted example so you can verify the format visually
print("\n" + "="*60)
print("SAMPLE FORMATTED EXAMPLE (verify format looks correct):")
print("="*60)
print(train_dataset[0]["text"][:800])
print("...")

# ── STEP 2: TRAINING ────────────────────────────────────────────────────────

print("Loading model...")
print(f"  Model: {MODEL_NAME}")
print(f"  Dtype: bfloat16")
print(f"  Attention: flash_attention_2")
print(f"  This will take 2-3 minutes and use ~14GB VRAM for weights...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,        # BF16: stable on H100, half the memory of FP32
    attn_implementation="flash_attention_2",  # Required for efficient attention on long sequences
    device_map="auto",                 # Automatically places model on available GPU(s)
)

# Enable gradient checkpointing
# Trades compute for memory — discards intermediate activations during forward pass
# and recomputes them during backward. ~30% slower but halves activation memory.
model.gradient_checkpointing_enable()
print("  Gradient checkpointing: enabled")

# Count trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel loaded.")
print(f"  Total parameters: {total_params/1e9:.2f}B")
print(f"  Trainable parameters: {trainable_params/1e9:.2f}B ({trainable_params/total_params*100:.1f}%)")

# Report VRAM usage after model load
vram_used = torch.cuda.memory_allocated() / 1e9
vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  VRAM used: {vram_used:.1f}GB / {vram_total:.1f}GB ({vram_used/vram_total*100:.1f}%)")

# ── Training Arguments ─────────────────────────────────────────────────────
# Every argument here maps directly to what we discussed.
# Comments explain what each one does and why.

training_args = SFTConfig(
    # Output
    output_dir=OUTPUT_DIR,

    # Epochs
    num_train_epochs=num_train_epochs,

    # Batch size
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,

    # Optimizer and learning rate
    learning_rate=learning_rate,
    lr_scheduler_type=lr_scheduler_type,
    warmup_ratio=warmup_ratio,
    weight_decay=weight_decay,
    max_grad_norm=max_grad_norm,

    # Precision — BF16 for H100
    bf16=True,
    fp16=False,  # Never mix fp16 and bf16

    # Sequence length — must match CFG
    max_length=max_seq_length,

    # Logging
    logging_steps=logging_steps,
    report_to="wandb",  # Sends all metrics to W&B automatically

    # Evaluation
    eval_strategy="steps",
    eval_steps=eval_steps,

    # Saving
    save_strategy="steps",
    save_steps=save_steps,
    save_total_limit=save_total_limit,
    load_best_model_at_end=True,  # Loads best checkpoint by eval loss at end
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # DataLoader
    dataloader_num_workers=4,   # Parallel data loading
    dataloader_pin_memory=True, # Faster CPU→GPU transfer

    # Reproducibility
    seed=seed,

    # dataset_text_field tells SFTTrainer which column contains the formatted text
    dataset_text_field="text",

    # CRITICAL: Pack sequences for efficiency
    # packing=True concatenates short sequences to fill max_seq_length
    # This improves throughput but disables per-example loss masking.
    # We use packing=False to keep loss masking clean.
    packing=False,
)

# ── SFTTrainer ─────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# ── Print training plan before starting ───────────────────────────────────
steps_per_epoch = len(train_dataset) // (per_device_train_batch_size * gradient_accumulation_steps)
total_steps = steps_per_epoch * num_train_epochs
warmup_steps = int(total_steps * warmup_ratio)

print("Training plan:")
print(f"  Train examples: {len(train_dataset)}")
print(f"  Val examples: {len(eval_dataset)}")
print(f"  Effective batch size: {per_device_train_batch_size * gradient_accumulation_steps}")
print(f"  Steps per epoch: ~{steps_per_epoch}")
print(f"  Total steps: ~{total_steps}")
print(f"  Warmup steps: {warmup_steps}")
print(f"  Eval every: {eval_steps} steps")
print(f"  Save every: {save_steps} steps")
print()
print("Starting training...")
print("Monitor at: https://wandb.ai")
print()

# ── Train ──────────────────────────────────────────────────────────────────
train_result = trainer.train()

# ── Log final metrics ──────────────────────────────────────────────────────
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

print("\nTraining complete.")
print(f"  Final train loss: {metrics.get('train_loss', 'N/A'):.4f}")
print(f"  Total steps: {metrics.get('train_steps_per_second', 'N/A')}")
print(f"\n[Train] Done. Saving model to {OUTPUT_DIR}...")

#  ── Save the model ──────────────────────────────────────────────────────
final_save_path = os.path.join(OUTPUT_DIR, "final")
os.makedirs(final_save_path, exist_ok=True)

print(f"Saving final SFT checkpoint to: {final_save_path}")

# Save model weights and config
trainer.save_model(final_save_path)

# Save tokenizer — ALWAYS save alongside model
tokenizer.save_pretrained(final_save_path)

print("Saved:")
print(f"  Model: {final_save_path}")
print(f"  Tokenizer: {final_save_path}")
print()
print("This checkpoint is your GRPO starting point.")
print("In H07_GRPO_Training.ipynb, set:")
print(f'  CFG["sft_model_path"] = "{final_save_path}"')

# Final VRAM report
vram_used = torch.cuda.memory_allocated() / 1e9
print(f"\nVRAM used after training: {vram_used:.1f}GB")

#  ── Format verification ──────────────────────────────────────────────────────
print("Running format verification on 3 GSM8K test examples...")
print("Check that each output has <think>...</think> tags and a final answer.")
print()

# Load GSM8K test split for verification
gsm8k_test = load_dataset("openai/gsm8k", "main", split="test")
test_samples = gsm8k_test.select(range(3))

# Set model to eval mode
model.eval()

for i, sample in enumerate(test_samples):
    question = sample["question"]
    gt_answer = sample["answer"].split("####")[-1].strip()

    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # True at inference time
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,   # Low temperature for deterministic check
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated tokens (not the prompt)
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    # Check format
    has_open_think = "<think>" in generated
    has_close_think = "</think>" in generated
    has_answer = "The answer is" in generated
    format_ok = has_open_think and has_close_think and has_answer

    print(f"Example {i+1}:")
    print(f"  Question: {question[:100]}...")
    print(f"  Ground truth answer: {gt_answer}")
    print(f"  Format check: <think>={has_open_think} | </think>={has_close_think} | answer={has_answer} | {'✓ OK' if format_ok else '✗ BROKEN'}")
    print(f"  Generated output:")
    print(f"  {generated[:400]}")
    print()

print("="*60)
print("If all 3 examples show ✓ OK format, SFT cold start is successful.")
print("If format is broken, check your format_numina_example/format_gsm8k_example functions.")
print("="*60)

# Finish W&B run
wandb.finish()
print("\nSFT pipeline complete. Proceed to H07_GRPO_Training.py")