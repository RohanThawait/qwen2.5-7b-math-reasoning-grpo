import re
import math
from typing import Optional


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


# ============================================================
# UNIT TESTS — run these before connecting to training
# ============================================================

print("Running reward function unit tests...")
print()

test_cases = [
    {
        "name": "Correct answer with good format",
        "response": "<think>\nJohn has 5 apples. He gets 3 more. 5 + 3 = 8.\nTherefore John has 8 apples.\n</think>\n\nThe answer is: 8",
        "ground_truth": "8",
        "expected_correct": True,
        "expected_format": True,
    },
    {
        "name": "Wrong answer with good format",
        "response": "<think>\nJohn has 5 apples. 5 + 3 = 7.\nTherefore 7 apples.\n</think>\n\nThe answer is: 7",
        "ground_truth": "8",
        "expected_correct": False,
        "expected_format": True,
    },
    {
        "name": "Correct answer no think tags",
        "response": "John has 8 apples. The answer is: 8",
        "ground_truth": "8",
        "expected_correct": True,
        "expected_format": False,
    },
    {
        "name": "Answer with comma formatting (1,000)",
        "response": "<think>\nCalculation: 500 * 2 = 1000\n</think>\n\nThe answer is: 1,000",
        "ground_truth": "1000",
        "expected_correct": True,
        "expected_format": True,
    },
    {
        "name": "Empty response",
        "response": "",
        "ground_truth": "8",
        "expected_correct": False,
        "expected_format": False,
    },
]

all_passed = True
for tc in test_cases:
    corr, corr_info = correctness_reward_fn(tc["response"], tc["ground_truth"])
    fmt, fmt_info = format_reward_fn(tc["response"])

    correct_ok = corr_info["correct"] == tc["expected_correct"]
    format_ok = fmt_info["format_ok"] == tc["expected_format"]
    passed = correct_ok and format_ok

    if not passed:
        all_passed = False

    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status} | {tc['name']}")
    if not passed:
        print(f"       Correct: expected={tc['expected_correct']}, got={corr_info['correct']}")
        print(f"       Format:  expected={tc['expected_format']}, got={fmt_info['format_ok']}")
        print(f"       Extracted answer: {corr_info.get('extracted_answer')}")

print()
if all_passed:
    print("All reward function tests passed. Safe to proceed with training.")
else:
    print("SOME TESTS FAILED. Fix reward functions before running training.")
    print("Broken reward functions corrupt your entire GRPO run.")