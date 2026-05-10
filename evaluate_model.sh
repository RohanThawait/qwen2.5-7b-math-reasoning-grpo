#!/bin/bash
# =============================================================================
# evaluate_model.sh
# Run lm-evaluation-harness on a single model checkpoint across all benchmarks.
#
# Usage:
#   bash evaluate_model.sh --model_path PATH --stage_name NAME [--output_dir DIR] [--batch_size N]
#
# Examples:
#   bash evaluate_model.sh --model_path Qwen/Qwen2.5-7B-Instruct --stage_name instruct_baseline
#   bash evaluate_model.sh --model_path ./checkpoints/sft --stage_name sft_checkpoint
#   bash evaluate_model.sh --model_path ./checkpoints/grpo_final --stage_name grpo_final
#
# Outputs:
#   evaluation/{stage_name}/gsm8k_results.json
#   evaluation/{stage_name}/math_results.json
#   evaluation/{stage_name}/arc_results.json
#   evaluation/{stage_name}/eval.log
# =============================================================================

set -e  # Exit immediately if any command fails

# ── Defaults ──────────────────────────────────────────────────────────────────
OUTPUT_BASE="./evaluation"
BATCH_SIZE=16
DTYPE="bfloat16"

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path)   MODEL_PATH="$2";   shift 2 ;;
    --stage_name)   STAGE_NAME="$2";   shift 2 ;;
    --output_dir)   OUTPUT_BASE="$2";  shift 2 ;;
    --batch_size)   BATCH_SIZE="$2";   shift 2 ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: bash evaluate_model.sh --model_path PATH --stage_name NAME [--output_dir DIR] [--batch_size N]"
      exit 1
      ;;
  esac
done

# ── Validate required args ─────────────────────────────────────────────────────
if [[ -z "$MODEL_PATH" ]]; then
  echo "ERROR: --model_path is required."
  exit 1
fi

if [[ -z "$STAGE_NAME" ]]; then
  echo "ERROR: --stage_name is required. Use a descriptive name like 'instruct_baseline', 'sft_checkpoint', 'grpo_final'."
  exit 1
fi

# ── Setup output directory ─────────────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_BASE}/${STAGE_NAME}"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="${OUTPUT_DIR}/eval.log"

echo "============================================================" | tee -a "$LOG_FILE"
echo "  LM-Evaluation-Harness Benchmarking Pipeline"               | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "  Model path  : $MODEL_PATH"                                  | tee -a "$LOG_FILE"
echo "  Stage name  : $STAGE_NAME"                                  | tee -a "$LOG_FILE"
echo "  Output dir  : $OUTPUT_DIR"                                  | tee -a "$LOG_FILE"
echo "  Batch size  : $BATCH_SIZE"                                  | tee -a "$LOG_FILE"
echo "  Dtype       : $DTYPE"                                       | tee -a "$LOG_FILE"
echo "  Timestamp   : $(date)"                                      | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"


# =============================================================================
# BENCHMARK 1 — GSM8K (8-shot)
# Primary benchmark. In-domain for your training. Expect largest gains here.
# Standard: 8-shot, flexible-extract metric.
# =============================================================================
echo "" | tee -a "$LOG_FILE"
echo "[1/3] Running GSM8K (8-shot)..." | tee -a "$LOG_FILE"
echo "      Expected time: ~10-15 min on H100 for 7B model" | tee -a "$LOG_FILE"

lm_eval \
  --model hf \
  --model_args pretrained="${MODEL_PATH}",dtype="${DTYPE}" \
  --tasks gsm8k \
  --num_fewshot 8 \
  --batch_size "${BATCH_SIZE}" \
  --output_path "${OUTPUT_DIR}/gsm8k_results.json" \
  --log_samples \
  2>&1 | tee -a "$LOG_FILE"

echo "[1/3] GSM8K complete." | tee -a "$LOG_FILE"


# =============================================================================
# BENCHMARK 2 — MATH / hendrycks_math (4-shot)
# Hard in-domain benchmark. Competition math across 7 subjects, 5 difficulty
# levels. Tests whether GRPO improved generalizable reasoning, not just
# GSM8K pattern matching.
# Standard: 4-shot. Note: this task takes significantly longer than GSM8K.
#
# If hendrycks_math is not available in your lm-eval version, try:
#   --tasks math_500   (faster, 500 examples)
# =============================================================================
echo "" | tee -a "$LOG_FILE"
echo "[2/3] Running MATH / hendrycks_math (4-shot)..." | tee -a "$LOG_FILE"
echo "      Expected time: ~30-45 min on H100 for 7B model (large task)" | tee -a "$LOG_FILE"
echo "      If this fails, replace hendrycks_math with math_500 in the script." | tee -a "$LOG_FILE"

lm_eval \
  --model hf \
  --model_args pretrained="${MODEL_PATH}",dtype="${DTYPE}" \
  --tasks hendrycks_math500 \
  --num_fewshot 4 \
  --batch_size "${BATCH_SIZE}" \
  --output_path "${OUTPUT_DIR}/math_results.json" \
  --log_samples \
  2>&1 | tee -a "$LOG_FILE"

echo "[2/3] MATH complete." | tee -a "$LOG_FILE"


# =============================================================================
# BENCHMARK 3 — ARC-Challenge (25-shot)
# Out-of-domain benchmark. Abstract reasoning science questions.
# Tests whether math RL training transfers to a different reasoning domain.
# Standard: 25-shot, normalized accuracy metric.
# =============================================================================
echo "" | tee -a "$LOG_FILE"
echo "[3/3] Running ARC-Challenge (25-shot)..." | tee -a "$LOG_FILE"
echo "      Expected time: ~5-10 min on H100 for 7B model" | tee -a "$LOG_FILE"

lm_eval \
  --model hf \
  --model_args pretrained="${MODEL_PATH}",dtype="${DTYPE}" \
  --tasks arc_challenge \
  --num_fewshot 25 \
  --batch_size "${BATCH_SIZE}" \
  --output_path "${OUTPUT_DIR}/arc_results.json" \
  --log_samples \
  2>&1 | tee -a "$LOG_FILE"

echo "[3/3] ARC-Challenge complete." | tee -a "$LOG_FILE"


# ── Summary ────────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "  All benchmarks complete for stage: $STAGE_NAME"             | tee -a "$LOG_FILE"
echo "  Results saved to: $OUTPUT_DIR"                              | tee -a "$LOG_FILE"
echo "  Finished at: $(date)"                                       | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Next step: after running all stages, run:"                    | tee -a "$LOG_FILE"
echo "  python aggregate_results.py --eval_dir ${OUTPUT_BASE}"      | tee -a "$LOG_FILE"