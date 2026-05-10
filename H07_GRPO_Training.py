import os
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
import wandb
import numpy as np
import random
import re
from typing import Optional
from collections import defaultdict

CFG = {
    # ── Model paths ────────────────────────────────────────────
    # Path to your SFT checkpoint from H07_SFT_Training.ipynb
    # UPDATE THIS to your actual Drive path
    "sft_model_path": "/home/devinder/gnss_phase2/project_7/evaluation/sft_model/final",

    # ── Output ─────────────────────────────────────────────────
    # Change run_name to 'grpo_no_format' or 'grpo_with_process' for ablations
    "run_name": "grpo_final",
    "output_dir": "/home/devinder/gnss_phase2/project_7/evaluation/grpo_model",

    # ── Dataset ────────────────────────────────────────────────
    "seed": 42,

    # ── GRPO core hyperparameters ──────────────────────────────
    # G — number of rollouts per problem per step.
    # Higher G = more stable group baseline = better gradient signal.
    # G=4: fast, reasonable signal. G=8: better but 2x compute.
    # Start with G=4 for your baseline run.
    "num_generations": 4,

    # Number of problems per batch.
    # Total forward passes per step = batch_size * num_generations
    # 8 * 4 = 32 forward passes per step — manageable on H100.
    "per_device_train_batch_size": 8,

    # Total GRPO update steps.
    # 1000 steps is sufficient to see measurable improvement on GSM8K.
    # Each step = one gradient update after processing one batch.
    "max_steps": 1000,

    # Learning rate — lower than SFT because we're doing RL updates.
    # Too high: KL divergence spikes, policy becomes unstable.
    # Too low: no learning. 5e-7 to 1e-6 is the typical GRPO range.
    "learning_rate": 5e-7,

    # KL coefficient — weight on the KL penalty term.
    # Controls how far the policy can drift from the reference model.
    # Too high: model barely moves — learning stalls.
    # Too low: reward hacking — model games reward without real learning.
    # 0.04 is a reasonable starting point (similar to DeepSeek R1 settings).
    "kl_coeff": 0.04, #beta

    # Max tokens the model generates per rollout.
    # Watch your response length metric — if it approaches this ceiling
    # the model may be trying to generate more. Increase if needed.
    "max_new_tokens": 1024,

    # Temperature for rollout generation.
    # Higher = more diverse rollouts = better exploration.
    # Lower = more deterministic = less variance in reward signal.
    # 0.9 gives good diversity without being chaotic.
    "temperature": 0.9,

    # Gradient clipping — same as SFT, prevents catastrophic updates.
    "max_grad_norm": 0.1,  # Tighter than SFT — GRPO updates should be small

    # ── Reward function flags (ablation control) ───────────────
    # Set these to False to run ablation experiments.
    # Baseline run: all True
    # Ablation 1 (no format): use_format_reward=False
    # Ablation 2 (with process): use_process_reward=True
    "use_format_reward": True,
    "use_length_penalty": True,
    "use_process_reward": False,  # Set True to run process reward ablation

    # ── Reward values ──────────────────────────────────────────
    # Correctness reward — primary signal.
    "correctness_reward": 1.0,

    # Format reward — incentivizes think tag structure.
    # Lower than correctness so format doesn't dominate.
    "format_reward": 0.5,

    # Length penalty bounds — responses outside this range get penalized.
    # GSM8K solutions typically need 100-600 tokens of reasoning.
    "length_min_tokens": 50,
    "length_max_tokens": 800,
    "length_penalty_coeff": 0.1,  # Scale of penalty per token outside bounds

    # Process reward — step-level proxy scoring (ablation only).
    "process_reward_max": 0.7,  # Maximum process reward — less than correctness

    # ── Logging ────────────────────────────────────────────────
    "logging_steps": 10,
    "save_steps": 200,
    "save_total_limit": 3,  # Keep intermediate checkpoints for learning curve analysis

    # ── W&B ────────────────────────────────────────────────────
    "wandb_project": "H07-reasoning-model",
}

print("GRPO Configuration loaded.")
print(f"  Run name: {CFG['run_name']}")
print(f"  SFT model: {CFG['sft_model_path']}")
print(f"  Output dir: {CFG['output_dir']}")
print(f"  Max steps: {CFG['max_steps']}")
print(f"  Group size (G): {CFG['num_generations']}")
print(f"  Batch size: {CFG['per_device_train_batch_size']}")
print(f"  Forward passes per step: {CFG['per_device_train_batch_size'] * CFG['num_generations']}")
print()
print("Reward stack:")
print(f"  Correctness:    ALWAYS ON  = {CFG['correctness_reward']}")
print(f"  Format:         {'ON' if CFG['use_format_reward'] else 'OFF'} = {CFG['format_reward']}")
print(f"  Length penalty: {'ON' if CFG['use_length_penalty'] else 'OFF'}")
print(f"  Process reward: {'ON' if CFG['use_process_reward'] else 'OFF'} = {CFG['process_reward_max']} max")

os.makedirs(CFG["output_dir"], exist_ok=True)

# Seeds
random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
torch.cuda.manual_seed_all(CFG["seed"])

# GPU check
if not torch.cuda.is_available():
    raise RuntimeError("No GPU detected. Switch runtime to H100.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# W&B login
wandb.login()
os.environ["WANDB_PROJECT"] = CFG["wandb_project"]
os.environ["WANDB_RUN_NAME"] = CFG["run_name"]
print(f"W&B run: {CFG['run_name']} in project {CFG['wandb_project']}")



print("Loading GSM8K train split for GRPO...")
gsm8k = load_dataset("openai/gsm8k", "main")
train_data = gsm8k["train"]

print(f"  Train problems: {len(train_data)}")
print(f"  Test problems (held out): {len(gsm8k['test'])}")
print()

# Parse ground truth answers from GSM8K format
# GSM8K answer format: 'step1\nstep2\n#### 42'
# We only need the number after ####
def parse_gsm8k_answer(answer_str: str) -> str:
    """Extract the final numeric answer from GSM8K answer string."""
    if "####" not in answer_str:
        return answer_str.strip()
    return answer_str.split("####")[-1].strip()

# Build GRPO dataset — only questions and ground truth answers
# Ground truth is needed by reward functions, not by the model
grpo_examples = []
for example in train_data:
    grpo_examples.append({
        "question": example["question"],
        "ground_truth": parse_gsm8k_answer(example["answer"]),
    })

# Shuffle
random.shuffle(grpo_examples)

print(f"GRPO dataset ready: {len(grpo_examples)} problems")
print()
print("Sample problem:")
print(f"  Question: {grpo_examples[0]['question'][:150]}...")
print(f"  Ground truth answer: {grpo_examples[0]['ground_truth']}")





# ============================================================
# ANSWER EXTRACTION
# ============================================================
# This is the most critical piece of infrastructure.
# If your extractor is wrong, your correctness reward is wrong,
# and your entire GRPO run learns the wrong thing.

def extract_answer(text: str) -> Optional[str]:
    """
    Extract the final answer from model output.
    Tries multiple patterns in order of specificity.
    Returns None if no parseable answer found.
    """
    # Pattern 1: After closing think tag
    # Matches: </think>\n\nThe answer is: 42
    pattern_after_think = r"</think>.*?(?:the answer is:?|answer:?)\s*([\d,\.\-\/]+)"
    match = re.search(pattern_after_think, text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    # Pattern 2: \boxed{} format (common in NuminaMath-style outputs)
    pattern_boxed = r"\\boxed\{([^}]+)\}"
    match = re.search(pattern_boxed, text)
    if match:
        return match.group(1).strip()

    # Pattern 3: Explicit answer markers
    pattern_explicit = r"(?:the answer is:?|final answer:?|answer:?)\s*\$?([\d,\.\-\/]+)"
    match = re.search(pattern_explicit, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Pattern 4: Last number in the response (fallback)
    # This is a weak signal — use it as last resort
    numbers = re.findall(r"[\-]?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1]

    return None


def normalize_answer(answer: str) -> Optional[float]:
    """
    Normalize an answer string to a float for comparison.
    Handles: commas in large numbers, dollar signs, units, fractions.
    Returns None if not parseable as a number.
    """
    if answer is None:
        return None

    # Remove common non-numeric chars
    cleaned = answer.strip()
    cleaned = re.sub(r"[\$,\s%]", "", cleaned)   # Remove $, commas, spaces, %
    cleaned = re.sub(r"[a-zA-Z]+$", "", cleaned)  # Remove trailing units (km, apples, etc)
    cleaned = cleaned.strip()

    # Handle simple fractions: 3/4 → 0.75
    if "/" in cleaned:
        parts = cleaned.split("/")
        if len(parts) == 2:
            try:
                return float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                return None

    try:
        return float(cleaned)
    except ValueError:
        return None


def answers_are_equal(pred: str, gt: str, tolerance: float = 1e-3) -> bool:
    """
    Compare predicted and ground truth answers.
    Uses near-equality for floats to handle rounding differences.
    Falls back to string comparison if normalization fails.
    """
    pred_norm = normalize_answer(pred)
    gt_norm = normalize_answer(gt)

    if pred_norm is not None and gt_norm is not None:
        return abs(pred_norm - gt_norm) <= tolerance

    # Fallback: string comparison after basic cleaning
    pred_clean = re.sub(r"[\s,\$]", "", pred.lower()) if pred else ""
    gt_clean = re.sub(r"[\s,\$]", "", gt.lower()) if gt else ""
    return pred_clean == gt_clean


# ============================================================
# REWARD FUNCTION 1 — CORRECTNESS
# ============================================================
# Primary reward signal. Binary — correct or not.
# This drives the actual learning in GRPO.

def correctness_reward_fn(
    response: str,
    ground_truth: str,
    reward_value: float = 1.0
) -> tuple[float, dict]:
    """
    Returns:
        reward: 1.0 if correct, 0.0 otherwise
        info: dict with extracted answer for logging
    """
    extracted = extract_answer(response)

    if extracted is None:
        return 0.0, {"extracted_answer": None, "correct": False, "parseable": False}

    is_correct = answers_are_equal(extracted, ground_truth)
    reward = reward_value if is_correct else 0.0

    return reward, {
        "extracted_answer": extracted,
        "ground_truth": ground_truth,
        "correct": is_correct,
        "parseable": True,
    }


# ============================================================
# REWARD FUNCTION 2 — FORMAT
# ============================================================
# Incentivizes the think tag structure.
# Calibrated at 0.5 — important but secondary to correctness.

def format_reward_fn(
    response: str,
    reward_value: float = 0.5
) -> tuple[float, dict]:
    """
    Checks for correct think tag structure:
    - Opening <think> tag present
    - Closing </think> tag present
    - Opening comes before closing
    - Content exists inside think block
    - Content exists after think block (the answer section)

    Returns:
        reward: 0.5 if format correct, 0.0 otherwise
        info: dict with format check details
    """
    has_open = "<think>" in response
    has_close = "</think>" in response

    if not has_open or not has_close:
        return 0.0, {"format_ok": False, "reason": "missing think tags"}

    open_idx = response.index("<think>")
    close_idx = response.index("</think>")

    if open_idx >= close_idx:
        return 0.0, {"format_ok": False, "reason": "close tag before open tag"}

    # Content inside think block
    think_content = response[open_idx + len("<think>"):close_idx].strip()
    if len(think_content) < 10:
        return 0.0, {"format_ok": False, "reason": "think block too short or empty"}

    # Content after think block (the answer)
    after_think = response[close_idx + len("</think>"):].strip()
    if len(after_think) < 3:
        return 0.0, {"format_ok": False, "reason": "no content after think block"}

    return reward_value, {"format_ok": True, "think_content_len": len(think_content)}


# ============================================================
# REWARD FUNCTION 3 — LENGTH PENALTY
# ============================================================
# Soft penalty outside acceptable length range.
# Prevents degenerate very-short or very-long responses.

def length_penalty_fn(
    response: str,
    tokenizer,
    min_tokens: int = 50,
    max_tokens: int = 800,
    penalty_coeff: float = 0.1
) -> tuple[float, dict]:
    """
    Returns 0.0 within acceptable range.
    Returns negative penalty proportional to distance outside range.

    Returns:
        reward: 0.0 (no penalty) or negative float (penalty)
        info: dict with token length
    """
    token_length = len(tokenizer.encode(response, add_special_tokens=False))

    if token_length < min_tokens:
        # Too short — likely no reasoning shown
        deficit = min_tokens - token_length
        penalty = -penalty_coeff * (deficit / min_tokens)
        return penalty, {"token_length": token_length, "penalty_reason": "too_short", "penalty": penalty}

    if token_length > max_tokens:
        # Too long — likely padding or circular reasoning
        excess = token_length - max_tokens
        penalty = -penalty_coeff * min(excess / max_tokens, 1.0)  # Cap at -0.1
        return penalty, {"token_length": token_length, "penalty_reason": "too_long", "penalty": penalty}

    return 0.0, {"token_length": token_length, "penalty_reason": "none"}


# ============================================================
# REWARD FUNCTION 4 — STEP-LEVEL PROCESS REWARD (ABLATION)
# ============================================================
# Rewards the quality of intermediate reasoning steps.
# Proxy implementation — no human annotation needed.
# Three proxy checks:
#   1. Step markers present (structural reasoning)
#   2. Equations present (mathematical operations shown)
#   3. Answer grounded in shown work (consistency)

def process_reward_fn(
    response: str,
    ground_truth: str,
    max_reward: float = 0.7
) -> tuple[float, dict]:
    """
    Returns up to max_reward (0.7) based on three proxy checks.
    Always returns less than correctness reward to keep it secondary.

    Returns:
        reward: 0.0 to max_reward
        info: dict with individual proxy scores
    """
    # Extract think block content
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if not think_match:
        return 0.0, {"process_score": 0.0, "reason": "no think block"}

    think_content = think_match.group(1)
    score = 0.0
    info = {}

    # ── Proxy 1: Step markers (0.2) ─────────────────────────────
    # Logical connectives that indicate step-by-step progression
    step_markers = [
        r"\bstep\s*\d+",           # "Step 1:", "Step 2"
        r"\btherefore\b",
        r"\bthus\b",
        r"\bso\b",
        r"\bthis means\b",
        r"\bsubstitut",             # substituting, substitute
        r"\bsimplif",               # simplifying, simplify
        r"\bfirst\b.*\bthen\b",    # "first... then..."
        r"\bnext\b",
        r"\bfinally\b",
    ]
    marker_count = sum(
        1 for p in step_markers if re.search(p, think_content, re.IGNORECASE)
    )
    has_step_markers = marker_count >= 2  # At least 2 different markers
    proxy1_score = 0.2 if has_step_markers else 0.0
    score += proxy1_score
    info["proxy1_step_markers"] = proxy1_score
    info["step_marker_count"] = marker_count

    # ── Proxy 2: Equation presence (0.2) ────────────────────────
    # Mathematical operations should be written explicitly
    equation_patterns = [
        r"\d+\s*[\+\-\*\/\=]\s*\d+",  # arithmetic: 3 + 4 = 7
        r"\d+\s*\=\s*\d+",             # equality: x = 42
        r"[a-zA-Z]\s*\=\s*\d+",        # variable assignment: n = 5
        r"\\frac",                      # LaTeX fractions
        r"\d+\.\d+",                   # decimals
    ]
    has_equations = any(
        re.search(p, think_content) for p in equation_patterns
    )
    proxy2_score = 0.2 if has_equations else 0.0
    score += proxy2_score
    info["proxy2_equations"] = proxy2_score

    # ── Proxy 3: Answer grounded in shown work (0.3) ─────────────
    # The final answer should appear in the reasoning steps
    # This checks that the answer isn't pulled from nowhere
    extracted_answer = extract_answer(response)
    if extracted_answer:
        # Check if the answer number appears in the think block
        answer_normalized = re.sub(r"[,\s\$]", "", extracted_answer)
        think_normalized = re.sub(r"[,\s\$]", "", think_content)
        answer_in_reasoning = answer_normalized in think_normalized
        proxy3_score = 0.3 if answer_in_reasoning else 0.0
    else:
        proxy3_score = 0.0
        answer_in_reasoning = False

    score += proxy3_score
    info["proxy3_answer_grounded"] = proxy3_score
    info["answer_in_reasoning"] = answer_in_reasoning

    # Scale to max_reward
    # Max possible raw score is 0.7 which equals max_reward
    final_reward = score  # Already on 0.0 to 0.7 scale
    info["process_score"] = final_reward

    return final_reward, info


# ============================================================
# COMPOSITE REWARD FUNCTION
# ============================================================
# This is what GRPO calls at each step.
# Combines all active reward components.
# Logs each component separately to W&B for debugging.

def compute_reward(
    response: str,
    ground_truth: str,
    tokenizer,
    cfg: dict,
) -> tuple[float, dict]:
    """
    Compute composite reward for a single rollout.

    Args:
        response: Model-generated text
        ground_truth: Correct answer string
        tokenizer: For length computation
        cfg: CFG dict — controls which rewards are active

    Returns:
        total_reward: float
        components: dict with individual reward values for logging
    """
    components = {}
    total = 0.0

    # 1. Correctness — always on
    corr_reward, corr_info = correctness_reward_fn(
        response, ground_truth, cfg["correctness_reward"]
    )
    total += corr_reward
    components["correctness"] = corr_reward
    components["correct"] = corr_info["correct"]
    components["parseable"] = corr_info["parseable"]

    # 2. Format — controlled by flag
    if cfg["use_format_reward"]:
        fmt_reward, fmt_info = format_reward_fn(response, cfg["format_reward"])
        total += fmt_reward
        components["format"] = fmt_reward
        components["format_ok"] = fmt_info["format_ok"]
    else:
        components["format"] = 0.0

    # 3. Length penalty — controlled by flag
    if cfg["use_length_penalty"]:
        len_reward, len_info = length_penalty_fn(
            response, tokenizer,
            cfg["length_min_tokens"],
            cfg["length_max_tokens"],
            cfg["length_penalty_coeff"],
        )
        total += len_reward
        components["length_penalty"] = len_reward
        components["token_length"] = len_info["token_length"]
    else:
        components["length_penalty"] = 0.0

    # 4. Process reward — ablation only
    if cfg["use_process_reward"]:
        proc_reward, proc_info = process_reward_fn(
            response, ground_truth, cfg["process_reward_max"]
        )
        total += proc_reward
        components["process"] = proc_reward
        components["process_step_markers"] = proc_info.get("proxy1_step_markers", 0)
        components["process_equations"] = proc_info.get("proxy2_equations", 0)
        components["process_grounded"] = proc_info.get("proxy3_answer_grounded", 0)
    else:
        components["process"] = 0.0

    components["total"] = total
    return total, components




print(f"Loading SFT checkpoint from: {CFG['sft_model_path']}")
print("This checkpoint becomes:")
print("  1. Your trainable policy model (weights will update)")
print("  2. Your frozen reference model (GRPOTrainer handles this internally)")
print()

tokenizer = AutoTokenizer.from_pretrained(CFG["sft_model_path"])
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    CFG["sft_model_path"],
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

model.gradient_checkpointing_enable()

total_params = sum(p.numel() for p in model.parameters())
vram_used = torch.cuda.memory_allocated() / 1e9
vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9

print(f"Model loaded: {total_params/1e9:.2f}B parameters")
print(f"VRAM used: {vram_used:.1f}GB / {vram_total:.1f}GB")
print()
print("Note: GRPOTrainer will load a second copy of this model as frozen reference.")
print(f"Total VRAM needed for both copies: ~{vram_used * 2:.1f}GB + optimizer states")




def build_grpo_dataset(examples: list, tokenizer) -> Dataset:
    """
    Build the GRPO training dataset.
    Each example has:
      - prompt: formatted question (model generates from here)
      - ground_truth: correct answer (used by reward functions, not model)
    """
    formatted = []
    for ex in examples:
        # Apply chat template to format the prompt
        # add_generation_prompt=True because we want the model to generate from here
        messages = [{"role": "user", "content": ex["question"]}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        formatted.append({
            "prompt": prompt,
            "ground_truth": ex["ground_truth"],
        })
    return Dataset.from_list(formatted)


grpo_dataset = build_grpo_dataset(grpo_examples, tokenizer)

print(f"GRPO dataset built: {len(grpo_dataset)} examples")
print()
print("Sample prompt (what model sees):")
print(grpo_dataset[0]["prompt"][:300])
print("...")
print(f"Ground truth: {grpo_dataset[0]['ground_truth']}")





# ── Reward function wrapper for GRPOTrainer ────────────────────────────────
# GRPOTrainer expects: reward_fn(prompts, completions, **kwargs) -> List[float]
# We wrap compute_reward to match this interface and log components to W&B.

# Accumulator for logging reward components averaged over logging interval
reward_log_buffer = defaultdict(list)
step_counter = [0]  # Using list for mutability in closure

def reward_function(prompts: list, completions: list, **kwargs) -> list[float]:
    """
    GRPOTrainer reward function interface.

    Args:
        prompts: List of prompt strings (questions)
        completions: List of model-generated completion strings
        kwargs: Additional data passed through — ground_truth is here

    Returns:
        List of reward floats, one per completion
    """
    ground_truths = kwargs.get("ground_truth", [""] * len(completions))
    rewards = []

    for completion, gt in zip(completions, ground_truths):
        reward, components = compute_reward(
            response=completion,
            ground_truth=gt,
            tokenizer=tokenizer,
            cfg=CFG,
        )
        rewards.append(reward)

        # Buffer component values for logging
        for key, val in components.items():
            if isinstance(val, (int, float)):
                reward_log_buffer[f"reward/{key}"].append(float(val))

    # Log to W&B every logging_steps
    step_counter[0] += 1
    if step_counter[0] % CFG["logging_steps"] == 0:
        log_dict = {
            k: sum(v) / len(v)
            for k, v in reward_log_buffer.items()
            if v
        }
        if wandb.run:
            wandb.log(log_dict, step=step_counter[0])
        reward_log_buffer.clear()

    return rewards


# ── GRPO Training Configuration ────────────────────────────────────────────
grpo_config = GRPOConfig(
    output_dir=CFG["output_dir"],

    # Total training steps
    max_steps=CFG["max_steps"],

    # Batch sizes
    per_device_train_batch_size=CFG["per_device_train_batch_size"],

    # Group size G — rollouts per problem
    num_generations=CFG["num_generations"],

    # Learning rate — smaller than SFT for stable RL updates
    learning_rate=CFG["learning_rate"],

    # KL coefficient — how much to penalize drift from reference model
    beta=CFG["kl_coeff"],

    # Generation settings for rollouts
    max_completion_length=CFG["max_new_tokens"],
    temperature=CFG["temperature"],

    # Precision
    bf16=True,
    fp16=False,

    # Gradient clipping — tighter than SFT for stable RL
    max_grad_norm=CFG["max_grad_norm"],

    # Logging
    logging_steps=CFG["logging_steps"],
    report_to="wandb",

    # Saving
    save_steps=CFG["save_steps"],
    save_total_limit=CFG["save_total_limit"],

    # Reproducibility
    seed=CFG["seed"],
)


# ── Initialize GRPOTrainer ─────────────────────────────────────────────────
# GRPOTrainer internally:
#   1. Makes a frozen copy of your model as the reference model
#   2. Generates num_generations rollouts per prompt
#   3. Calls reward_function to score each rollout
#   4. Computes group-relative advantages
#   5. Computes policy gradient loss + KL penalty
#   6. Updates policy model weights

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=grpo_config,
    train_dataset=grpo_dataset,
    reward_funcs=reward_function,
)

# ── Print training plan ────────────────────────────────────────────────────
print("GRPO Training plan:")
print(f"  Max steps: {CFG['max_steps']}")
print(f"  Problems per batch: {CFG['per_device_train_batch_size']}")
print(f"  Rollouts per problem (G): {CFG['num_generations']}")
print(f"  Total rollouts per step: {CFG['per_device_train_batch_size'] * CFG['num_generations']}")
print(f"  KL coefficient: {CFG['kl_coeff']}")
print(f"  Max new tokens per rollout: {CFG['max_new_tokens']}")
print()
print("What to watch in W&B:")
print("  reward/total       — should trend upward over training")
print("  reward/correctness — correctness rate, key learning signal")
print("  reward/format      — should stabilize quickly after SFT cold start")
print("  reward/token_length — should increase then plateau (model learning to reason more)")
print("  train/kl           — should increase gradually, watch for sudden spikes")
print()
print("Starting GRPO training...")
print("Monitor at: https://wandb.ai")
print()

# ── Train ──────────────────────────────────────────────────────────────────
trainer.train()

print("GRPO training complete.")

final_save_path = os.path.join(CFG["output_dir"], "final")
os.makedirs(final_save_path, exist_ok=True)

print(f"Saving GRPO checkpoint to: {final_save_path}")
trainer.save_model(final_save_path)
tokenizer.save_pretrained(final_save_path)

print("Saved.")
print()
print("For benchmarking, use this path in evaluate_model.sh:")
print(f'  --model_path "{final_save_path}"')
print(f'  --stage_name "{CFG["run_name"]}"')

wandb.finish()
print("\nGRPO pipeline complete. Run evaluate_model.sh to benchmark this checkpoint.")



print("Sampling 5 GRPO model outputs for inspection...")
print()

gsm8k_test = load_dataset("openai/gsm8k", "main", split="test")
test_samples = gsm8k_test.select(range(5))

model.eval()

for i, sample in enumerate(test_samples):
    question = sample["question"]
    gt_answer = sample["answer"].split("####")[-1].strip()

    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    extracted = extract_answer(generated)
    correct = answers_are_equal(extracted, gt_answer) if extracted else False
    token_len = len(tokenizer.encode(generated))

    print(f"Example {i+1} | GT: {gt_answer} | Extracted: {extracted} | {'✓ Correct' if correct else '✗ Wrong'} | Tokens: {token_len}")
    print(f"Output preview: {generated[:300]}")
    print()