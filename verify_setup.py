"""
verify_setup.py
===============
Run this BEFORE any benchmarking to verify your lm-evaluation-harness
installation is correctly configured and your baseline numbers match
published Qwen-2.5-7B-Instruct numbers.

This catches misconfigurations early — wrong dtype, wrong task names,
wrong metric keys — before you waste H100 time on a broken pipeline.

Usage:
    python verify_setup.py
    python verify_setup.py --model_path Qwen/Qwen2.5-7B-Instruct --quick

What it checks:
    1. lm-evaluation-harness is installed and importable
    2. Required task names exist in your installation
    3. GPU is available and has sufficient VRAM
    4. (Optional) Runs GSM8K on 50 examples and checks your number is
       in a reasonable range vs published Qwen-2.5-7B-Instruct scores
"""

import sys
import argparse
import subprocess
import json
import tempfile
from pathlib import Path


# =============================================================================
# Published reference scores for Qwen-2.5-7B-Instruct
# Source: Qwen2.5 technical report and HuggingFace model card
# Use these to sanity-check your evaluation setup before committing to
# full benchmark runs.
#
# IMPORTANT: These are approximate. Your numbers will differ slightly
# due to sampling randomness and lm-eval version differences.
# A difference of ±3% is acceptable. Larger than that — investigate.
# =============================================================================

PUBLISHED_REFERENCE = {
    "gsm8k": {
        "expected_approx": 85.0,       # ~85% for Qwen2.5-7B-Instruct on GSM8K 8-shot
        "acceptable_range": (80.0, 90.0),
        "note": "Qwen2.5-7B-Instruct published ~85% on GSM8K 8-shot",
    },
    "arc_challenge": {
        "expected_approx": 62.0,       # ~62% normalized accuracy
        "acceptable_range": (57.0, 67.0),
        "note": "Qwen2.5-7B-Instruct published ~62% on ARC-Challenge 25-shot",
    },
}

# Required task names — verify these exist in your lm-eval installation
REQUIRED_TASKS = ["gsm8k_cot", "arc_challenge", "hendrycks_math", "tmlu_driving_rule"]
FALLBACK_MATH_TASKS = ["math_500", "math"]


def check_lm_eval_installed() -> bool:
    """Check if lm-evaluation-harness is installed."""
    print("\n[1/4] Checking lm-evaluation-harness installation...")
    try:
        result = subprocess.run(
            ["lm_eval", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            print("  ✓ lm_eval command found")
            return True
        else:
            print("  ✗ lm_eval command failed")
            print(f"    stderr: {result.stderr[:300]}")
            return False
    except FileNotFoundError:
        print("  ✗ lm_eval command not found")
        print("  Fix: git clone https://github.com/EleutherAI/lm-evaluation-harness")
        print("       cd lm-evaluation-harness && pip install -e .")
        return False
    except subprocess.TimeoutExpired:
        print("  ✗ lm_eval --help timed out")
        return False

def check_available_tasks() -> dict:
    """
    Check which required tasks are available in this lm-eval installation.
    Returns dict of task_name -> available (bool).
    """
    print("\n[2/4] Checking available benchmark tasks...")

    result = subprocess.run(
        ["lm_eval", "ls"],  # ✅ updated CLI
        capture_output=True,
        text=True,
        timeout=60,
    )

    available_output = result.stdout + result.stderr
    task_status = {}

    for task in REQUIRED_TASKS:
        if task in available_output:
            print(f"  ✓ {task} — available")
            task_status[task] = True
        else:
            print(f"  ✗ {task} — NOT found")
            task_status[task] = False

    # Special check for MATH task — try fallbacks
    if not task_status.get("hendrycks_math"):
        print("    Checking fallback MATH tasks...")
        for fallback in FALLBACK_MATH_TASKS:
            if fallback in available_output:
                print(f"  ✓ {fallback} — available (use this instead of hendrycks_math)")
                print(f"    Update BENCHMARK_CONFIG in aggregate_results.py to use '{fallback}'")
                print(f"    Update evaluate_model.sh: change hendrycks_math to {fallback}")
                task_status["math_fallback"] = fallback
                break
        else:
            print("  ✗ No MATH task found. Check your lm-eval version.")

    return task_status



def check_gpu() -> bool:
    """Check GPU availability and VRAM."""
    print("\n[3/4] Checking GPU setup...")
    try:
        import torch

        if not torch.cuda.is_available():
            print("  ✗ No CUDA GPU detected — benchmarking will be very slow on CPU")
            print("    This is unexpected on your H100 server. Check your environment.")
            return False

        gpu_count = torch.cuda.device_count()
        print(f"  ✓ {gpu_count} GPU(s) detected")

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / 1e9
            print(f"  ✓ GPU {i}: {props.name} — {vram_gb:.1f} GB VRAM")

            if vram_gb < 20:
                print(f"    WARNING: Less than 20GB VRAM. 7B model in BF16 needs ~14GB.")
                print(f"    You may need to reduce batch_size in evaluate_model.sh.")
            elif vram_gb >= 70:
                print(f"    H100 confirmed. You can use batch_size=32 for faster evaluation.")

        return True

    except ImportError:
        print("  ✗ PyTorch not importable. Check your Python environment.")
        return False


def run_quick_verification(model_path: str) -> bool:
    """
    Run GSM8K on a small subset (50 examples) and verify the number is
    in the acceptable range for Qwen-2.5-7B-Instruct.

    This catches: wrong dtype, wrong model loading, wrong task config.
    """
    print(f"\n[4/4] Running quick verification on 50 GSM8K examples...")
    print(f"      Model: {model_path}")
    print(f"      Expected: ~{PUBLISHED_REFERENCE['gsm8k']['expected_approx']}% (±5% acceptable)")
    print(f"      This takes ~2-3 minutes on H100...")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "verify_output.json"

        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path},dtype=bfloat16",
            "--tasks", "gsm8k",
            "--num_fewshot", "8",
            "--limit", "50",            # Only 50 examples for speed
            "--batch_size", "8",
            "--output_path", str(output_file),
        ]

        print(f"      Running: {' '.join(cmd)}\n")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            print("  ✗ Evaluation failed")
            print(f"    stdout: {result.stdout[-500:]}")
            print(f"    stderr: {result.stderr[-500:]}")
            return False

        if not output_file.exists():
            print("  ✗ Output file not created — check lm-eval output_path behavior")
            # lm-eval sometimes appends to a directory — check for nested files
            nested = list(Path(tmpdir).rglob("*.json"))
            if nested:
                print(f"    Found nested JSON: {nested}")
                output_file = nested[0]
            else:
                return False

        with open(output_file, "r") as f:
            data = json.load(f)

        # Extract GSM8K score
        results = data.get("results", {})
        gsm8k_data = results.get("gsm8k", {})

        # Try multiple metric keys
        score = None
        for metric_key in ["exact_match,flexible-extract", "exact_match,strict-match", "acc,none"]:
            if metric_key in gsm8k_data:
                score = gsm8k_data[metric_key] * 100
                print(f"  Metric used: {metric_key}")
                break

        if score is None:
            print(f"  ✗ Could not find score in results. Available keys: {list(gsm8k_data.keys())}")
            print("    Update BENCHMARK_CONFIG in aggregate_results.py with the correct metric key.")
            print(f"    Raw results: {json.dumps(gsm8k_data, indent=2)}")
            return False

        ref = PUBLISHED_REFERENCE["gsm8k"]
        low, high = ref["acceptable_range"]

        # Scale acceptable range for 50-example subset (more variance expected)
        # Use wider range: ±10% for small sample
        subset_low = ref["expected_approx"] - 10
        subset_high = ref["expected_approx"] + 10

        print(f"\n  Result on 50 examples: {score:.1f}%")
        print(f"  Published reference:   ~{ref['expected_approx']}%")
        print(f"  Acceptable range (50-example subset): {subset_low:.0f}%–{subset_high:.0f}%")

        if subset_low <= score <= subset_high:
            print(f"  ✓ Score is in acceptable range — evaluation setup looks correct")
            print(f"  ✓ Safe to run full benchmarks with evaluate_model.sh")
            return True
        else:
            print(f"  ✗ Score is outside acceptable range")
            print(f"    Possible causes:")
            print(f"    - Wrong model path (not Qwen2.5-7B-Instruct)")
            print(f"    - Wrong dtype (use bfloat16)")
            print(f"    - Wrong num_fewshot (should be 8 for GSM8K)")
            print(f"    - lm-eval version mismatch changing task format")
            print(f"    Do NOT proceed with full benchmarking until this is resolved.")
            return False


def print_next_steps(task_status: dict):
    """Print the exact commands to run next."""
    math_task = "hendrycks_math"
    if not task_status.get("hendrycks_math") and "math_fallback" in task_status:
        math_task = task_status["math_fallback"]

    print("\n" + "=" * 70)
    print("  NEXT STEPS — Run benchmarks in this order:")
    print("=" * 70)
    print()
    print("  # Step 1: Benchmark instruct baseline (before any training)")
    print("  bash evaluate_model.sh \\")
    print("    --model_path Qwen/Qwen2.5-7B-Instruct \\")
    print("    --stage_name instruct_baseline")
    print()
    print("  # Step 2: After SFT — benchmark SFT checkpoint")
    print("  bash evaluate_model.sh \\")
    print("    --model_path ./checkpoints/sft \\")
    print("    --stage_name sft_checkpoint")
    print()
    print("  # Step 3: After GRPO — benchmark final checkpoint")
    print("  bash evaluate_model.sh \\")
    print("    --model_path ./checkpoints/grpo_final \\")
    print("    --stage_name grpo_final")
    print()
    print("  # Step 4: After ablation runs (optional)")
    print("  bash evaluate_model.sh \\")
    print("    --model_path ./checkpoints/grpo_no_format \\")
    print("    --stage_name ablation_no_format")
    print()
    print("  # Step 5: Aggregate all results into comparison table")
    print("  python aggregate_results.py --eval_dir ./evaluation")
    print()

    if math_task != "hendrycks_math":
        print(f"  NOTE: Replace 'hendrycks_math' with '{math_task}' in evaluate_model.sh")
        print()

    print("=" * 70)


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify lm-evaluation-harness setup before benchmarking."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model path for quick verification run. Default: Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip the 50-example verification run (checks installation only)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  LM-EVALUATION-HARNESS SETUP VERIFICATION")
    print("  Run this before any benchmarking.")
    print("=" * 70)

    all_passed = True

    # Check 1: Installation
    if not check_lm_eval_installed():
        print("\n✗ Setup failed at step 1. Fix lm-eval installation before continuing.")
        sys.exit(1)
    
    # Check 2: Tasks
    task_status = check_available_tasks()
    critical_tasks = ["gsm8k", "arc_challenge"]
    for task in critical_tasks:
        if not task_status.get(task):
            all_passed = False

    # Check 3: GPU
    if not check_gpu():
        all_passed = False

    # Check 4: Quick run (optional)
    if not args.quick:
        if not run_quick_verification(args.model_path):
            all_passed = False
    else:
        print("\n[4/4] Skipping quick verification run (--quick flag set)")

    # Final verdict
    print("\n" + "=" * 70)
    if all_passed:
        print("  ✓ ALL CHECKS PASSED — safe to proceed with benchmarking")
    else:
        print("  ✗ SOME CHECKS FAILED — resolve issues above before benchmarking")
        print("    Running benchmarks with a broken setup wastes H100 time.")
    print("=" * 70)

    # Always print next steps
    print_next_steps(task_status)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
