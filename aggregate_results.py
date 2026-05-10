"""
aggregate_results.py
====================
Reads lm-evaluation-harness JSON output files across all evaluated model
stages and produces a clean comparison table with delta analysis.

Usage:
    python aggregate_results.py --eval_dir ./evaluation
    python aggregate_results.py --eval_dir ./evaluation --output results_summary.csv

Output:
    - Prints a formatted comparison table to terminal
    - Saves results_summary.csv for use in your model card / README
    - Saves results_summary.json for programmatic use
    - Saves sample_analysis.txt with qualitative failure/success cases

Expected directory structure:
    evaluation/
        instruct_baseline/
            gsm8k_results.json
            math_results.json
            arc_results.json
        sft_checkpoint/
            gsm8k_results.json
            math_results.json
            arc_results.json
        grpo_final/
            gsm8k_results.json
            math_results.json
            arc_results.json
        ablation_no_format/          (optional)
            gsm8k_results.json
            ...
"""

import json
import os
import csv
import argparse
from pathlib import Path


# =============================================================================
# Configuration — metric keys to extract from lm-eval JSON output
# These are the exact keys lm-evaluation-harness writes into results JSON.
# If your version writes different keys, the script will tell you what's
# available so you can update these.
# =============================================================================

BENCHMARK_CONFIG = {
    "gsm8k": {
        "file": "gsm8k_results.json",
        "display_name": "GSM8K (8-shot)",
        # flexible-extract is more forgiving of formatting — use this over strict-match
        "primary_metric": "exact_match,flexible-extract",
        "fallback_metrics": [
            "exact_match,strict-match",
            "acc,none",
            "acc_norm,none",
        ],
    },
    "math": {
    "file": "math_results.json",
    "display_name": "MATH/hendrycks_math500 (4-shot)",
    "primary_metric": "exact_match,none",   # ← change from flexible-extract to none
    "fallback_metrics": [
            "exact_match,flexible-extract",
            "acc,none",
        ],
    },
    "arc_challenge": {
        "file": "arc_results.json",
        "display_name": "ARC-Challenge (25-shot)",
        # ARC uses normalized accuracy as the standard metric
        "primary_metric": "acc_norm,none",
        "fallback_metrics": [
            "acc,none",
            "exact_match,flexible-extract",
        ],
    },
}

# Stage display order — controls column order in output table.
# Add or remove stages here to match what you actually evaluated.
STAGE_ORDER = [
    "instruct_baseline",
    "sft_checkpoint",
    "grpo_final",
    "ablation_no_format",
    "ablation_with_process_reward",
]

# Which stage is your baseline for delta computation
BASELINE_STAGE = "instruct_baseline"

# Which stage is your SFT stage for GRPO-specific delta
SFT_STAGE = "sft_checkpoint"


# =============================================================================
# Core extraction logic
# =============================================================================

def extract_metric(results_dict: dict, benchmark_key: str) -> float | None:
    """
    Extract the primary metric from a benchmark results dict.
    Falls back through fallback_metrics if primary is not found.
    Returns None if nothing is found — caller handles missing data.
    """
    config = BENCHMARK_CONFIG[benchmark_key]
    task_results = results_dict.get("results", {})

    # lm-eval can nest results under the task name or subtask names
    # Try direct task name first, then look for any key containing the task
    candidate_task_keys = []
    for key in task_results.keys():
        if benchmark_key in key.lower() or any(
            alias in key.lower()
            for alias in ["gsm8k", "hendrycks_math", "arc_challenge", "math"]
            if alias in benchmark_key or alias == benchmark_key
        ):
            candidate_task_keys.append(key)

    if not candidate_task_keys:
        # Nothing matched — print available keys to help debugging
        print(f"    WARNING: No task matching '{benchmark_key}' found in results.")
        print(f"    Available task keys: {list(task_results.keys())}")
        return None

    # For tasks like hendrycks_math that have subtasks, aggregate across subtasks
    scores = []
    for task_key in candidate_task_keys:
        task_data = task_results[task_key]

        # Try primary metric first
        primary = config["primary_metric"]
        if primary in task_data:
            scores.append(task_data[primary])
            continue

        # Try fallback metrics
        found = False
        for fallback in config["fallback_metrics"]:
            if fallback in task_data:
                scores.append(task_data[fallback])
                found = True
                break

        if not found:
            print(f"    WARNING: No known metric found for task '{task_key}'.")
            print(f"    Available metrics: {list(task_data.keys())}")

    if not scores:
        return None

    # If multiple subtasks (e.g. hendrycks_math has algebra, geometry, etc.)
    # return the mean across subtasks
    return sum(scores) / len(scores)


def load_stage_results(stage_dir: Path) -> dict:
    """
    Load all benchmark results for a single model stage directory.
    Returns dict of benchmark_key -> score (float, 0-100 scale) or None.
    """
    scores = {}
    for benchmark_key, config in BENCHMARK_CONFIG.items():
        # lm-eval appends timestamps to filenames — find by prefix
        file_stem = config["file"].replace(".json", "")
        matches = sorted(stage_dir.glob(f"{file_stem}*.json"))
        if not matches:
            print(f"  MISSING: {stage_dir}/{file_stem}*.json — skipping this benchmark for this stage.")
            scores[benchmark_key] = None
            continue
        result_file = matches[-1]  # use most recent if multiple exist

        try:
            with open(result_file, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  ERROR: Could not parse {result_file}: {e}")
            scores[benchmark_key] = None
            continue

        raw_score = extract_metric(data, benchmark_key)

        if raw_score is None:
            scores[benchmark_key] = None
        else:
            # lm-eval returns scores as 0.0-1.0 — convert to percentage
            scores[benchmark_key] = round(raw_score * 100, 2)

    return scores


def load_all_stages(eval_dir: Path) -> dict:
    """
    Scan eval_dir for stage subdirectories and load results for each.
    Respects STAGE_ORDER for consistent column ordering.
    """
    # Find all subdirectories that look like stage directories
    found_stages = {
        d.name: d
        for d in eval_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    }

    if not found_stages:
        raise FileNotFoundError(f"No stage directories found in {eval_dir}")

    # Order stages: known stages first (in STAGE_ORDER), then any extras alphabetically
    ordered_stage_names = [s for s in STAGE_ORDER if s in found_stages]
    extra_stages = sorted(s for s in found_stages if s not in STAGE_ORDER)
    ordered_stage_names.extend(extra_stages)

    print(f"\nFound {len(found_stages)} stage(s): {ordered_stage_names}")

    all_results = {}
    for stage_name in ordered_stage_names:
        stage_dir = found_stages[stage_name]
        print(f"\nLoading results for stage: {stage_name}")
        all_results[stage_name] = load_stage_results(stage_dir)

    return all_results


# =============================================================================
# Delta computation
# =============================================================================

def compute_deltas(all_results: dict) -> dict:
    """
    For each stage, compute:
    - delta_vs_baseline: change from BASELINE_STAGE
    - delta_vs_sft: change from SFT_STAGE (only for GRPO and ablation stages)
    """
    deltas = {}
    baseline_scores = all_results.get(BASELINE_STAGE, {})
    sft_scores = all_results.get(SFT_STAGE, {})

    for stage_name, scores in all_results.items():
        deltas[stage_name] = {}
        for benchmark_key in BENCHMARK_CONFIG:
            score = scores.get(benchmark_key)
            baseline = baseline_scores.get(benchmark_key)
            sft = sft_scores.get(benchmark_key)

            delta_vs_baseline = None
            delta_vs_sft = None

            if score is not None and baseline is not None and stage_name != BASELINE_STAGE:
                delta_vs_baseline = round(score - baseline, 2)

            if score is not None and sft is not None and stage_name not in [BASELINE_STAGE, SFT_STAGE]:
                delta_vs_sft = round(score - sft, 2)

            deltas[stage_name][benchmark_key] = {
                "delta_vs_baseline": delta_vs_baseline,
                "delta_vs_sft": delta_vs_sft,
            }

    return deltas


# =============================================================================
# Output formatting
# =============================================================================

def format_delta(delta: float | None) -> str:
    """Format a delta value with sign and color hint."""
    if delta is None:
        return "   —  "
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.2f}%"


def print_comparison_table(all_results: dict, deltas: dict):
    """Print a formatted comparison table to terminal."""

    stages = list(all_results.keys())
    benchmarks = list(BENCHMARK_CONFIG.keys())

    col_width = 18
    stage_col_width = 30

    print("\n")
    print("=" * 90)
    print("  BENCHMARK COMPARISON TABLE")
    print("=" * 90)

    # Header row
    header = f"{'Benchmark':<{stage_col_width}}"
    for stage in stages:
        # Shorten long stage names for display
        display = stage.replace("_", " ").title()
        if len(display) > col_width - 2:
            display = display[:col_width - 5] + "..."
        header += f"{display:>{col_width}}"
    print(header)
    print("-" * 90)

    # Score rows
    for benchmark_key in benchmarks:
        config = BENCHMARK_CONFIG[benchmark_key]
        row = f"{config['display_name']:<{stage_col_width}}"
        for stage in stages:
            score = all_results[stage].get(benchmark_key)
            cell = f"{score:.2f}%" if score is not None else "N/A"
            row += f"{cell:>{col_width}}"
        print(row)

    print("-" * 90)

    # Delta rows — vs baseline
    if BASELINE_STAGE in stages:
        print(f"\n  Δ vs {BASELINE_STAGE.replace('_', ' ').title()}")
        print("-" * 90)
        for benchmark_key in benchmarks:
            config = BENCHMARK_CONFIG[benchmark_key]
            row = f"{config['display_name']:<{stage_col_width}}"
            for stage in stages:
                delta_info = deltas.get(stage, {}).get(benchmark_key, {})
                delta = delta_info.get("delta_vs_baseline")
                cell = format_delta(delta) if stage != BASELINE_STAGE else "  base  "
                row += f"{cell:>{col_width}}"
            print(row)

    # Delta rows — vs SFT (for GRPO and ablation stages only)
    grpo_stages = [s for s in stages if s not in [BASELINE_STAGE, SFT_STAGE]]
    if SFT_STAGE in stages and grpo_stages:
        print(f"\n  Δ vs {SFT_STAGE.replace('_', ' ').title()} (GRPO contribution)")
        print("-" * 90)
        for benchmark_key in benchmarks:
            config = BENCHMARK_CONFIG[benchmark_key]
            row = f"{config['display_name']:<{stage_col_width}}"
            for stage in stages:
                delta_info = deltas.get(stage, {}).get(benchmark_key, {})
                delta = delta_info.get("delta_vs_sft")
                if stage in [BASELINE_STAGE]:
                    cell = "        "
                elif stage == SFT_STAGE:
                    cell = "  base  "
                else:
                    cell = format_delta(delta)
                row += f"{cell:>{col_width}}"
            print(row)

    print("=" * 90)
    print()


def save_csv(all_results: dict, deltas: dict, output_path: Path):
    """Save results as CSV for use in README / model card."""

    stages = list(all_results.keys())
    rows = []

    for benchmark_key, config in BENCHMARK_CONFIG.items():
        row = {"Benchmark": config["display_name"]}
        for stage in stages:
            score = all_results[stage].get(benchmark_key)
            row[stage] = f"{score:.2f}%" if score is not None else "N/A"

            # Delta vs baseline
            delta_info = deltas.get(stage, {}).get(benchmark_key, {})
            delta_b = delta_info.get("delta_vs_baseline")
            row[f"{stage}_delta_vs_baseline"] = format_delta(delta_b) if delta_b is not None else "—"

            # Delta vs SFT
            delta_s = delta_info.get("delta_vs_sft")
            row[f"{stage}_delta_vs_sft"] = format_delta(delta_s) if delta_s is not None else "—"

        rows.append(row)

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV saved to: {output_path}")


def save_json(all_results: dict, deltas: dict, output_path: Path):
    """Save full results as JSON for programmatic use."""
    output = {
        "stages": list(all_results.keys()),
        "benchmarks": {k: v["display_name"] for k, v in BENCHMARK_CONFIG.items()},
        "scores": all_results,
        "deltas": deltas,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"JSON saved to: {output_path}")


def print_key_findings(all_results: dict, deltas: dict):
    """
    Print a plain-English summary of key findings.
    This is the basis for your model card analysis section.
    """
    stages = list(all_results.keys())

    print("\n" + "=" * 90)
    print("  KEY FINDINGS SUMMARY")
    print("  Use these as the basis for your model card analysis section.")
    print("=" * 90)

    for benchmark_key, config in BENCHMARK_CONFIG.items():
        print(f"\n  {config['display_name']}")
        print(f"  {'─' * 50}")

        for stage in stages:
            score = all_results[stage].get(benchmark_key)
            delta_info = deltas.get(stage, {}).get(benchmark_key, {})
            delta_b = delta_info.get("delta_vs_baseline")
            delta_s = delta_info.get("delta_vs_sft")

            score_str = f"{score:.2f}%" if score is not None else "N/A"
            delta_b_str = f" ({format_delta(delta_b)} vs baseline)" if delta_b is not None else ""
            delta_s_str = f" ({format_delta(delta_s)} vs SFT)" if delta_s is not None else ""

            print(f"    {stage:<35} {score_str}{delta_b_str}{delta_s_str}")

    print("\n  INTERPRETATION GUIDE")
    print("  ─────────────────────")
    print("  SFT delta large, GRPO delta small  → SFT format injection drove gains; RL had limited effect")
    print("  SFT delta small, GRPO delta large  → RL genuinely improved reasoning beyond imitation")
    print("  Both deltas large                  → Both stages contributed — strong result")
    print("  MATH gains > GSM8K gains           → Model learned transferable reasoning (unlikely but great)")
    print("  ARC-Challenge improved             → Math RL transferred to general reasoning (noteworthy)")
    print("  ARC-Challenge flat or negative     → Gains are math-specific (honest, expected finding)")
    print("=" * 90)
    print()


# =============================================================================
# Sample analysis — qualitative inspection of log_samples
# =============================================================================

def analyze_samples(eval_dir: Path, output_path: Path):
    """
    Load log_samples from GSM8K results for each stage and find interesting cases:
    - Cases where instruct wrong, GRPO right (clearest RL wins)
    - Cases where all stages wrong (model limits)
    - Cases where instruct right but GRPO wrong (regressions)

    Only runs if log_samples files are present.
    """
    stages_with_samples = {}

    for stage_dir in eval_dir.iterdir():
        if not stage_dir.is_dir():
            continue
        gsm8k_file = stage_dir / "gsm8k_results.json"
        if not gsm8k_file.exists():
            continue

        with open(gsm8k_file, "r") as f:
            data = json.load(f)

        # log_samples are stored in the samples key
        samples = data.get("samples", {})
        if not samples:
            continue

        # samples is a dict keyed by task name
        for task_key, task_samples in samples.items():
            if "gsm8k" in task_key.lower():
                stages_with_samples[stage_dir.name] = task_samples
                break

    if len(stages_with_samples) < 2:
        print("\n  Sample analysis skipped — need log_samples from at least 2 stages.")
        print("  (log_samples are saved when you use --log_samples flag in lm_eval)")
        return

    # Find cases where instruct was wrong but grpo was right
    instruct_samples = stages_with_samples.get(BASELINE_STAGE, [])
    grpo_samples = stages_with_samples.get("grpo_final", [])

    if not instruct_samples or not grpo_samples:
        print("\n  Sample analysis: instruct_baseline or grpo_final samples not found.")
        return

    # Build lookup by doc_id for alignment
    def build_lookup(samples):
        return {s.get("doc_id", i): s for i, s in enumerate(samples)}

    instruct_lookup = build_lookup(instruct_samples)
    grpo_lookup = build_lookup(grpo_samples)

    grpo_wins = []      # instruct wrong, grpo right
    regressions = []    # instruct right, grpo wrong
    both_wrong = []     # both wrong

    common_ids = set(instruct_lookup.keys()) & set(grpo_lookup.keys())

    for doc_id in common_ids:
        inst = instruct_lookup[doc_id]
        grpo = grpo_lookup[doc_id]

        inst_correct = inst.get("exact_match", inst.get("acc", 0)) == 1
        grpo_correct = grpo.get("exact_match", grpo.get("acc", 0)) == 1

        if not inst_correct and grpo_correct:
            grpo_wins.append({"doc_id": doc_id, "instruct": inst, "grpo": grpo})
        elif inst_correct and not grpo_correct:
            regressions.append({"doc_id": doc_id, "instruct": inst, "grpo": grpo})
        elif not inst_correct and not grpo_correct:
            both_wrong.append({"doc_id": doc_id, "instruct": inst, "grpo": grpo})

    # Write analysis to file
    with open(output_path, "w") as f:
        f.write("QUALITATIVE SAMPLE ANALYSIS — GSM8K\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total compared examples: {len(common_ids)}\n")
        f.write(f"GRPO wins (instruct wrong → GRPO right): {len(grpo_wins)}\n")
        f.write(f"Regressions (instruct right → GRPO wrong): {len(regressions)}\n")
        f.write(f"Both wrong: {len(both_wrong)}\n\n")

        # Write top 10 GRPO wins
        f.write("─" * 70 + "\n")
        f.write("TOP GRPO WINS (clearest evidence of RL learning)\n")
        f.write("─" * 70 + "\n\n")
        for i, case in enumerate(grpo_wins[:10]):
            f.write(f"Case {i+1} (doc_id: {case['doc_id']})\n")
            doc = case["instruct"].get("doc", {})
            f.write(f"  Question: {str(doc.get('question', 'N/A'))[:200]}\n")
            f.write(f"  Instruct output: {str(case['instruct'].get('filtered_resps', 'N/A'))[:300]}\n")
            f.write(f"  GRPO output:     {str(case['grpo'].get('filtered_resps', 'N/A'))[:300]}\n")
            f.write(f"  Ground truth:    {str(doc.get('answer', 'N/A'))[:100]}\n\n")

        # Write top 5 regressions
        f.write("─" * 70 + "\n")
        f.write("REGRESSIONS (instruct right → GRPO wrong — model limits)\n")
        f.write("─" * 70 + "\n\n")
        for i, case in enumerate(regressions[:5]):
            f.write(f"Case {i+1} (doc_id: {case['doc_id']})\n")
            doc = case["instruct"].get("doc", {})
            f.write(f"  Question: {str(doc.get('question', 'N/A'))[:200]}\n")
            f.write(f"  Instruct output: {str(case['instruct'].get('filtered_resps', 'N/A'))[:300]}\n")
            f.write(f"  GRPO output:     {str(case['grpo'].get('filtered_resps', 'N/A'))[:300]}\n")
            f.write(f"  Ground truth:    {str(doc.get('answer', 'N/A'))[:100]}\n\n")

    print(f"Sample analysis saved to: {output_path}")
    print(f"  GRPO wins: {len(grpo_wins)} | Regressions: {len(regressions)} | Both wrong: {len(both_wrong)}")


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate lm-evaluation-harness results across model stages."
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="./evaluation",
        help="Root directory containing stage subdirectories. Default: ./evaluation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Base name for output files (without extension). Default: {eval_dir}/results_summary",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")

    output_base = Path(args.output) if args.output else eval_dir / "results_summary"

    # Load all stage results
    all_results = load_all_stages(eval_dir)

    if not all_results:
        print("No results found. Check your evaluation directory structure.")
        return

    # Compute deltas
    deltas = compute_deltas(all_results)

    # Print comparison table
    print_comparison_table(all_results, deltas)

    # Print key findings interpretation
    print_key_findings(all_results, deltas)

    # Save outputs
    save_csv(all_results, deltas, Path(str(output_base) + ".csv"))
    save_json(all_results, deltas, Path(str(output_base) + ".json"))

    # Qualitative sample analysis (runs only if log_samples present)
    analyze_samples(eval_dir, Path(str(output_base) + "_sample_analysis.txt"))

    print("\nAll done. Files saved:")
    print(f"  {output_base}.csv")
    print(f"  {output_base}.json")
    print(f"  {output_base}_sample_analysis.txt")
    print("\nNext steps:")
    print("  1. Review the comparison table above")
    print("  2. Read sample_analysis.txt for qualitative evidence")
    print("  3. Use results_summary.csv in your README and HuggingFace model card")


if __name__ == "__main__":
    main()
