# Agent Pipelines Documentation for DeepSeek-R1

This document describes all agent pipelines, workflows, and data flows in the DeepSeek-R1 system. Each pipeline includes ASCII diagrams showing the sequence of operations, prompts used, and data transformations.

## Table of Contents
1. [DeepSeek-R1-Zero Pipeline](#deepseek-r1-zero-pipeline)
2. [DeepSeek-R1 Full Training Pipeline](#deepseek-r1-full-training-pipeline)
3. [Distillation Pipeline](#distillation-pipeline)
4. [Inference Pipelines](#inference-pipelines)

---

## DeepSeek-R1-Zero Pipeline

### Overview
DeepSeek-R1-Zero is the first model trained via pure reinforcement learning without any supervised fine-tuning (SFT) as a preliminary step. This approach demonstrates that reasoning capabilities can emerge through RL alone.

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DeepSeek-R1-Zero Training Pipeline                │
└─────────────────────────────────────────────────────────────────────┘

Step 1: Base Model
┌──────────────────┐
│ DeepSeek-V3-Base │
│   (671B params)  │
│  (37B activated) │
└────────┬─────────┘
         │
         ▼
Step 2: Apply Training Template
┌─────────────────────────────────────────────────────────────────┐
│ Template:                                                        │
│ "A conversation between User and Assistant. The user asks a     │
│  question, and the Assistant solves it. The assistant first     │
│  thinks about the reasoning process in the mind and then        │
│  provides the user with the answer. The reasoning process and   │
│  answer are enclosed within <think> </think> and <answer>       │
│  </answer> tags..."                                             │
│                                                                  │
│ Input: {prompt} = Reasoning question (math, code, logic)       │
└────────┬────────────────────────────────────────────────────────┘
         │
         ▼
Step 3: RL Training Loop (Thousands of steps)
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  ┌──────────────┐                                               │
│  │   Prompts    │ (Math, Code, STEM questions)                 │
│  └──────┬───────┘                                               │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────┐                                       │
│  │  Sample G outputs    │ (G=group size, typically 4-64)       │
│  │  from policy πθ_old  │                                       │
│  └──────┬───────────────┘                                       │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────┐                                       │
│  │  Reward Calculation  │                                       │
│  │  ┌─────────────────┐ │                                       │
│  │  │ Accuracy Reward │ │ (Rule-based: correct/incorrect)     │
│  │  └─────────────────┘ │                                       │
│  │  ┌─────────────────┐ │                                       │
│  │  │ Format Reward   │ │ (Enforces <think>/<answer> tags)    │
│  │  └─────────────────┘ │                                       │
│  └──────┬───────────────┘                                       │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────┐                                       │
│  │  GRPO Optimization   │                                       │
│  │  ┌─────────────────┐ │                                       │
│  │  │ Advantage Ai =  │ │                                       │
│  │  │ (ri - mean(r))  │ │                                       │
│  │  │ / std(r)        │ │                                       │
│  │  └─────────────────┘ │                                       │
│  │  Update policy πθ    │                                       │
│  └──────┬───────────────┘                                       │
│         │                                                        │
│         └────────┐                                              │
│                  │                                              │
│         ┌────────▼──────────┐                                   │
│         │ Convergence?      │                                   │
│         │ (~8000 steps)     │                                   │
│         └────────┬──────────┘                                   │
│                  │ No → Loop back                               │
│                  │ Yes ↓                                        │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
Step 4: DeepSeek-R1-Zero Model
┌──────────────────────────────────────────────────────────────────┐
│ Performance:                                                      │
│ - AIME 2024: 71.0% (pass@1), 86.7% (cons@64)                    │
│ - MATH-500: 95.9% (pass@1)                                       │
│ - GPQA Diamond: 73.3%                                            │
│                                                                   │
│ Emergent Behaviors:                                              │
│ - Self-verification and reflection                               │
│ - Long chain-of-thought reasoning (hundreds to thousands tokens) │
│ - "Aha moments" - spontaneous reevaluation                       │
│                                                                   │
│ Limitations:                                                      │
│ - Poor readability                                               │
│ - Language mixing                                                │
│ - Occasional endless repetition                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. **Algorithm: GRPO (Group Relative Policy Optimization)**

Mathematical formulation:
```
J_GRPO(θ) = E[q ~ P(Q), {oi}^G_i=1 ~ πθ_old(O|q)]
            [1/G Σ min(π_θ(oi|q)/π_θ_old(oi|q) * Ai,
                       clip(π_θ(oi|q)/π_θ_old(oi|q), 1-ε, 1+ε) * Ai)]
            - β * D_KL(πθ || πref)

Where:
- Ai = (ri - mean({r1, r2, ..., rG})) / std({r1, r2, ..., rG})
- ε = clipping parameter (prevents large policy updates)
- β = KL divergence coefficient
```

**Why GRPO?**
- No critic model needed (saves compute)
- Estimates baseline from group scores
- More efficient than PPO for large-scale training

#### 2. **Reward System**

```
┌─────────────────────────────────────────────────────────────┐
│                      Reward Components                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Accuracy Reward (Rule-Based):                              │
│  ┌──────────────────────────────────────────┐              │
│  │ • Math: Extract answer from \boxed{}     │              │
│  │   Compare with ground truth              │              │
│  │                                           │              │
│  │ • Code: Run test cases in compiler       │              │
│  │   Check pass/fail on predefined tests    │              │
│  │                                           │              │
│  │ • Logic: Verify final conclusion         │              │
│  └──────────────────────────────────────────┘              │
│  Output: reward ∈ {0, 1} (binary)                          │
│                                                              │
│  Format Reward:                                             │
│  ┌──────────────────────────────────────────┐              │
│  │ • Check presence of <think> tag          │              │
│  │ • Check presence of </think> tag         │              │
│  │ • Check presence of <answer> tag         │              │
│  │ • Check presence of </answer> tag        │              │
│  │ • Verify proper nesting                  │              │
│  └──────────────────────────────────────────┘              │
│  Output: reward ∈ {0, 1} (binary)                          │
│                                                              │
│  Final Reward: r = accuracy_reward + format_reward          │
└─────────────────────────────────────────────────────────────┘
```

**Why No Neural Reward Model?**
- Avoids reward hacking in large-scale RL
- No need to retrain reward model
- Simpler training pipeline
- More stable training

#### 3. **Training Data Distribution**

```
Training Prompts:
├── Mathematics
│   ├── AIME problems
│   ├── MATH dataset
│   └── Competition math
│
├── Code
│   ├── LeetCode problems
│   ├── Codeforces problems
│   └── Algorithm challenges
│
└── STEM/Logic
    ├── GPQA questions
    ├── Science problems
    └── Logical reasoning
```

#### 4. **Self-Evolution Process**

```
Training Progress Timeline:
═══════════════════════════════════════════════════════════

Step 0-2000: Basic Reasoning Emergence
┌──────────────────────────────────────────────┐
│ • Model learns to use <think>/<answer> tags  │
│ • Simple step-by-step reasoning develops     │
│ • Average response length: ~1000-2000 tokens │
│ • AIME accuracy: 15.6% → 35%                 │
└──────────────────────────────────────────────┘

Step 2000-4000: Reflection and Verification
┌──────────────────────────────────────────────┐
│ • Self-verification emerges spontaneously    │
│ • Model checks own work                      │
│ • Average response length: ~2000-4000 tokens │
│ • AIME accuracy: 35% → 55%                   │
└──────────────────────────────────────────────┘

Step 4000-6000: "Aha Moments"
┌──────────────────────────────────────────────┐
│ • Model learns to reevaluate approaches      │
│ • Phrases like "Wait, wait..." appear        │
│ • Explores alternative solutions             │
│ • Average response length: ~4000-6000 tokens │
│ • AIME accuracy: 55% → 68%                   │
└──────────────────────────────────────────────┘

Step 6000-8000+: Advanced Reasoning
┌──────────────────────────────────────────────┐
│ • Long chain-of-thought reasoning            │
│ • Multiple verification passes               │
│ • Average response length: ~6000-10000 tokens│
│ • AIME accuracy: 68% → 71.0%                 │
└──────────────────────────────────────────────┘
```

### Data Flow

```
Input Flow:
Question → Template → Model → Raw Output → Reward

Example:
─────────────────────────────────────────────────────
Input Question: "Solve: √(a - √(a+x)) = x for a>1"

↓ [Apply Template]

Prompt: "A conversation between User and Assistant...
         User: Solve: √(a - √(a+x)) = x for a>1
         Assistant:"

↓ [Model Generation]

Output: "<think>
         To solve √(a - √(a+x)) = x, square both sides...
         [detailed reasoning process]
         ...
         Wait, let me verify this approach...
         [verification]
         </think>
         <answer>
         The sum of real solutions is a² - a
         </answer>"

↓ [Reward Calculation]

Accuracy Reward: Check if "a² - a" is correct → 1.0
Format Reward: Check tags present and valid → 1.0
Final Reward: 2.0

↓ [GRPO Update]

Policy updated to increase probability of this response
─────────────────────────────────────────────────────
```

### Performance Metrics

| Benchmark | Initial (Base) | After RL | Improvement |
|-----------|---------------|----------|-------------|
| AIME 2024 (pass@1) | 15.6% | 71.0% | +55.4% |
| AIME 2024 (cons@64) | - | 86.7% | - |
| MATH-500 | - | 95.9% | - |
| GPQA Diamond | - | 73.3% | - |
| LiveCodeBench | - | 50.0% | - |
| Codeforces Rating | - | 1444 | - |

### Configuration

```yaml
Model:
  base: DeepSeek-V3-Base
  architecture: MoE
  total_params: 671B
  activated_params: 37B

Training:
  algorithm: GRPO
  group_size: 4-64 (depends on task)
  training_steps: ~8000
  temperature: 0.6 (for evaluation)
  max_output_length: 32768 tokens

Rewards:
  accuracy: rule_based
  format: rule_based
  neural_rm: false

Prompts:
  template: training_template (see PROMPTS_DOCUMENTATION.md)
  system_prompt: none
```

---
