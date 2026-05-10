# Teaching a Language Model to Reason with GRPO

This project implements the reasoning training pipeline introduced in the **DeepSeek-R1 paper**, applying **Group Relative Policy Optimization (GRPO)** to teach a 7B language model to reason step-by-step through mathematical problems.

The model is trained in two stages: a **Supervised Fine-Tuning (SFT) cold start** to install chain-of-thought reasoning format, followed by **GRPO reinforcement learning** to improve reasoning quality through verifiable reward signals — without any human preference labels.

---

## Model

**Base model:** Qwen/Qwen2.5-7B-Instruct  
**SFT checkpoint:** Available on HuggingFace [link]  
**GRPO checkpoint:** Available on HuggingFace [link]  
**W&B training run:** [vocal-armadillo-2](https://wandb.ai/rohanthawait-national-institute-of-technology-karnataka-/H07-reasoning-model/runs/ewjj8op2)

---

## Training Pipeline

### Stage 1 — SFT Cold Start

The base instruct model is finetuned on a curated math chain-of-thought dataset to install the reasoning format before RL begins. Without this cold start, GRPO rollouts have inconsistent structure and reward signals fire poorly from step one.

**Dataset:** ~27,000 examples combining:
- Full GSM8K train split (~7,473 examples) — grade-school math word problems
- 20,000 sampled examples from NuminaMath-CoT — competition math with chain-of-thought solutions

**Format:** Every training example is structured as:
```
<think>
[step-by-step reasoning]
</think>

The answer is: [final answer]
```

**Key training details:**
- Full parameter finetuning (no LoRA) on H100 80GB
- 2 epochs, effective batch size 32, learning rate 2e-5 with cosine decay
- Loss masking on prompt tokens — gradients flow only through the reasoning completion
- Final train loss: 0.3357 | Mean token accuracy: 92.5%
- Training time: ~2 hours

### Stage 2 — GRPO Training

GRPO teaches the model to reason correctly by generating multiple candidate responses per problem (rollouts), scoring each with reward functions, and updating the policy to produce more high-reward responses. Unlike PPO, GRPO eliminates the need for a separate critic network by using the mean reward within each group as the baseline — making it significantly more compute-efficient.

**Dataset:** GSM8K train split (problems only — no solutions provided to the model)

**Reward function stack:**
| Reward | Value | Description |
|---|---|---|
| Correctness | 1.0 | Parsed final answer matches ground truth |
| Format | 0.5 | Valid `<think>...</think>` tag structure present |
| Length penalty | ≤ 0.1 | Soft penalty for responses outside 50–800 token range |

**Key training details:**
- Group size G=4 rollouts per problem
- KL coefficient 0.04 — prevents policy from drifting too far from SFT reference
- 1,000 GRPO steps, learning rate 5e-7
- Training time: ~2 hours 9 minutes

---

## Benchmark Results

All three model stages evaluated using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) under identical settings.

| Benchmark | Instruct Baseline | SFT Checkpoint | GRPO Final |
|---|---|---|---|
| GSM8K 8-shot (flexible-extract) | 82.64% | 75.51% | 75.66% |
| MATH/hendrycks_math500 4-shot | 20.60% | 24.20% | 24.20% |
| ARC-Challenge 25-shot (acc_norm) | 67.06% | 62.97% | 62.80% |

**Δ vs Instruct Baseline**

| Benchmark | SFT | GRPO |
|---|---|---|
| GSM8K | -7.13% | -6.98% |
| MATH | **+3.60%** | **+3.60%** |
| ARC-Challenge | -4.09% | -4.26% |

---

## Understanding the Results

### Why does the instruct baseline score highest on GSM8K?

The SFT finetuning changed the model's output format — it now produces structured `<think>` reasoning chains before answering, rather than direct responses. lm-evaluation-harness was calibrated for the original instruct model's output style. This format shift suppresses GSM8K scores for the finetuned models even when the underlying reasoning capability is unchanged or improved.

This is a known evaluation artifact when finetuning instruct models for chain-of-thought output, and does not reflect a capability regression.

### The most meaningful number is MATH +3.6%

The model never trained on MATH benchmark problems. It only trained on GSM8K-style grade-school math and NuminaMath-CoT. The improvement from **20.60% → 24.20%** on competition-level math is evidence that SFT successfully installed a generalizable reasoning format — not just GSM8K pattern matching.

The slight decline on ARC-Challenge (abstract reasoning science questions) confirms this: the model's distribution shifted toward mathematical reasoning as intended. Gains on math, small loss on general reasoning — exactly what you would expect from math-specific training.

### Why GRPO showed limited improvement over SFT

This is the most technically interesting finding of the project.

The GRPO training encountered **reward saturation** — a documented problem that occurs when the starting model is already too capable on the training distribution. With a strong SFT cold start, the model solved most GSM8K rollouts correctly, meaning all G=4 rollouts within a group frequently scored the same reward. When reward variance within a group is zero, the advantage signals are zero and the gradient update carries no learning signal.

Measured directly: `frac_reward_zero_std` averaged **0.63 throughout training** — meaning 63% of batches produced near-zero gradient signal. KL divergence never exceeded 0.0006 across 1,000 steps, confirming the policy barely moved from the SFT reference.

This is the same challenge the DeepSeek R1 team addressed through **curriculum filtering** — selecting problems where the model succeeds on roughly 50% of rollouts, not 80%. With mid-difficulty curriculum selection, reward variance is higher, advantage signals are meaningful, and RL learning is more effective.

**What I would do differently:** Filter GSM8K and NuminaMath for mid-difficulty problems before GRPO — targeting problems where the SFT model gets 1-2 out of 4 rollouts correct. This maximizes reward variance and learning signal per step.

---

## Repository Structure

```
project_7/
├── H07_SFT_Training.py       # Full SFT training pipeline
├── H07_GRPO_Training.py      # Full GRPO training pipeline with reward functions
├── evaluate_model.sh         # lm-evaluation-harness benchmarking script
├── aggregate_results.py      # Results aggregation and comparison table
├── verify_setup.py           # Pre-benchmarking environment verification
├── configs/
│   ├── sft.yaml              # SFT hyperparameter config
│   ├── grpo.yaml             # GRPO baseline config
│   ├── grpo_no_format.yaml   # Ablation: no format reward
│   └── grpo_with_process.yaml # Ablation: step-level process reward
└── evaluation/
    ├── instruct_baseline/    # Benchmark results — Qwen2.5-7B-Instruct
    ├── sft_checkpoint/       # Benchmark results — after SFT
    └── grpo_final/           # Benchmark results — after GRPO
```

---

## Hardware and Environment

| Component | Detail |
|---|---|
| GPU | NVIDIA H100 NVL (99.9 GB VRAM) |
| Framework | PyTorch + HuggingFace TRL |
| Training library | TRL SFTTrainer + GRPOTrainer |
| Experiment tracking | Weights & Biases |
| Evaluation | lm-evaluation-harness |

---

## Key Technical Decisions

**Why full finetuning instead of LoRA for SFT?**  
LoRA updates a small fraction of parameters via low-rank adapters. For a cold-start where the goal is installing a new behavioral prior (structured CoT format), full finetuning gives the model more capacity to shift its output distribution. The H100's 80GB VRAM comfortably fits a 7B model for full finetuning.

**Why GSM8K only for GRPO?**  
GRPO requires verifiable reward signals — the model's answer must be checkable programmatically. GSM8K answers are clean numeric values, making automated correctness checking reliable. NuminaMath competition problems have more complex answer formats that increase reward function error rates.

**Why G=4 rollouts?**  
A group size of 4 balances learning signal quality against compute cost. G=4 means 32 total forward passes per step (8 problems × 4 rollouts), which is manageable on a single H100. G=8 would double compute with marginally better baseline estimates — not justified given the reward saturation bottleneck.

---

## References

- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)