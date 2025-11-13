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

## DeepSeek-R1 Full Training Pipeline

### Overview
DeepSeek-R1 improves upon R1-Zero by incorporating cold-start data and a multi-stage training pipeline. This approach addresses readability issues, reduces language mixing, and further enhances reasoning performance to match OpenAI-o1-1217.

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   DeepSeek-R1 Multi-Stage Training Pipeline              │
└─────────────────────────────────────────────────────────────────────────┘

                            ┌──────────────────┐
                            │ DeepSeek-V3-Base │
                            │   (671B params)  │
                            └────────┬─────────┘
                                     │
    ╔════════════════════════════════╧═════════════════════════════════╗
    ║                        STAGE 1: Cold Start                       ║
    ╚════════════════════════════════╤═════════════════════════════════╝
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │  Collect Cold-Start Data       │
                    │  (~thousands of samples)       │
                    ├────────────────────────────────┤
                    │ Methods:                       │
                    │ • Few-shot prompting with      │
                    │   long CoT examples            │
                    │ • Direct prompting for         │
                    │   detailed reasoning           │
                    │ • Curated R1-Zero outputs      │
                    │ • Human annotation refinement  │
                    │                                │
                    │ Format:                        │
                    │ |special_token|                │
                    │ <reasoning_process>            │
                    │ |special_token|                │
                    │ <summary>                      │
                    └────────┬───────────────────────┘
                             │
                             ▼
                    ┌────────────────────────────────┐
                    │  Supervised Fine-Tuning (SFT)  │
                    │  on Cold-Start Data            │
                    │                                │
                    │ Advantages:                    │
                    │ • Better readability           │
                    │ • Markdown formatting          │
                    │ • Summary at end               │
                    │ • No language mixing           │
                    └────────┬───────────────────────┘
                             │
    ╔════════════════════════╧═════════════════════════════════════════╗
    ║              STAGE 2: Reasoning-Oriented RL                      ║
    ╚════════════════════════╤═════════════════════════════════════════╝
                             │
                             ▼
          ┌──────────────────────────────────────────┐
          │  RL Training on Reasoning Tasks          │
          │  (Math, Code, Science, Logic)            │
          ├──────────────────────────────────────────┤
          │ Algorithm: GRPO (same as R1-Zero)        │
          │                                           │
          │ Rewards:                                  │
          │ • Accuracy Reward (rule-based)           │
          │ • Language Consistency Reward            │
          │   (proportion of target language)        │
          │                                           │
          │ Combined Reward:                          │
          │   r = accuracy + lang_consistency        │
          │                                           │
          │ Note: Language consistency causes        │
          │ slight performance drop but improves     │
          │ human preference alignment               │
          └──────────────┬───────────────────────────┘
                         │
                         ▼
          ┌──────────────────────────────────────────┐
          │  Continue Until Convergence              │
          │  on Reasoning Benchmarks                 │
          └──────────────┬───────────────────────────┘
                         │
    ╔════════════════════╧══════════════════════════════════════════════╗
    ║          STAGE 3: Rejection Sampling & SFT                        ║
    ╚════════════════════╤══════════════════════════════════════════════╝
                         │
                         ▼
       ┌─────────────────────────────────────────────────────┐
       │  Collect SFT Data via Rejection Sampling            │
       ├─────────────────────────────────────────────────────┤
       │                                                      │
       │  Reasoning Data (~600k samples):                    │
       │  ┌────────────────────────────────────────┐        │
       │  │ 1. Curate reasoning prompts             │        │
       │  │ 2. Sample multiple responses from RL    │        │
       │  │    checkpoint                           │        │
       │  │ 3. Verify correctness:                  │        │
       │  │    • Rule-based for clear answers       │        │
       │  │    • Generative RM (DeepSeek-V3) for   │        │
       │  │      ambiguous cases                    │        │
       │  │ 4. Filter:                              │        │
       │  │    • Remove language-mixed CoT          │        │
       │  │    • Remove long paragraphs             │        │
       │  │    • Remove messy code blocks           │        │
       │  │ 5. Keep only correct responses          │        │
       │  └────────────────────────────────────────┘        │
       │                                                      │
       │  Non-Reasoning Data (~200k samples):                │
       │  ┌────────────────────────────────────────┐        │
       │  │ • Writing                               │        │
       │  │ • Factual QA                            │        │
       │  │ • Self-cognition                        │        │
       │  │ • Translation                           │        │
       │  │ • Role-playing                          │        │
       │  │                                          │        │
       │  │ Sources:                                │        │
       │  │ • DeepSeek-V3 SFT dataset              │        │
       │  │ • DeepSeek-V3 generated CoT            │        │
       │  │   (for some tasks)                      │        │
       │  │                                          │        │
       │  │ Note: Simple queries like "hello"      │        │
       │  │ don't include CoT                       │        │
       │  └────────────────────────────────────────┘        │
       └──────────────────┬──────────────────────────────────┘
                          │
                          ▼
       ┌─────────────────────────────────────────────────────┐
       │  Supervised Fine-Tuning (2 epochs)                  │
       │  on Combined Dataset (~800k samples)                │
       │                                                      │
       │  Dataset:                                           │
       │  • 600k reasoning samples                           │
       │  • 200k non-reasoning samples                       │
       │                                                      │
       │  Base Model: DeepSeek-V3-Base (fresh start)        │
       └──────────────────┬──────────────────────────────────┘
                          │
    ╔═════════════════════╧═══════════════════════════════════════════╗
    ║            STAGE 4: RL for All Scenarios                        ║
    ╚═════════════════════╤═══════════════════════════════════════════╝
                          │
                          ▼
       ┌─────────────────────────────────────────────────────────────┐
       │  Final RL Training on All Task Types                        │
       ├─────────────────────────────────────────────────────────────┤
       │                                                              │
       │  Reasoning Tasks:                                           │
       │  ┌───────────────────────────────────────┐                 │
       │  │ • Math, Code, Science, Logic           │                 │
       │  │ • Rewards: Rule-based (accuracy)       │                 │
       │  │ • Same as Stage 2                      │                 │
       │  └───────────────────────────────────────┘                 │
       │                                                              │
       │  General Tasks:                                             │
       │  ┌───────────────────────────────────────┐                 │
       │  │ • Writing, QA, Dialogue, etc.          │                 │
       │  │ • Rewards: Neural reward models        │                 │
       │  │                                         │                 │
       │  │ Helpfulness RM:                        │                 │
       │  │   • Evaluates ONLY final summary       │                 │
       │  │   • Focuses on utility and relevance   │                 │
       │  │   • Minimizes reasoning interference   │                 │
       │  │                                         │                 │
       │  │ Harmlessness RM:                       │                 │
       │  │   • Evaluates ENTIRE response          │                 │
       │  │     (reasoning + summary)              │                 │
       │  │   • Identifies risks, biases, harm     │                 │
       │  └───────────────────────────────────────┘                 │
       │                                                              │
       │  Combined Training:                                         │
       │  • Mixed prompt distribution                                │
       │  • Multi-reward signals                                     │
       │  • Continuous until convergence                             │
       └──────────────────┬──────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────────┐
              │   DeepSeek-R1 Final       │
              │   (Production Model)      │
              │                           │
              │ Performance:              │
              │ • AIME 2024: 79.8%        │
              │ • MATH-500: 97.3%         │
              │ • GPQA: 71.5%             │
              │ • AlpacaEval: 87.6%       │
              │ • ArenaHard: 92.3%        │
              │                           │
              │ Qualities:                │
              │ • Strong reasoning        │
              │ • Good readability        │
              │ • No language mixing      │
              │ • General capabilities    │
              └───────────────────────────┘
```

### Stage Details

#### Stage 1: Cold Start

**Purpose**: Provide initial structured reasoning examples to stabilize early RL training

**Data Collection Methods**:
```
Method 1: Few-Shot Prompting
─────────────────────────────
Prompt: "Here's an example of detailed reasoning:
         <example with long CoT>
         Now solve this problem: {question}"

→ Generates structured long-form reasoning

Method 2: Direct Prompting
───────────────────────────
Prompt: "Generate a detailed answer with reflection
         and verification for: {question}"

→ Produces reasoning with self-checks

Method 3: Curated R1-Zero Outputs
──────────────────────────────────
1. Generate responses with R1-Zero
2. Filter for readability
3. Human annotation for refinement

→ Leverages R1-Zero capabilities with quality control

Method 4: Human Annotation
───────────────────────────
Expert annotators write reasoning examples

→ Highest quality but most expensive
```

**Output Format**:
```
|special_token|
<reasoning_process>
Step 1: Understanding the problem...
Step 2: Formulating approach...
[detailed reasoning]
Step N: Verification...
|special_token|
<summary>
The answer is X because [brief explanation].
```

**Advantages over R1-Zero**:
- Better readability with markdown formatting
- Summary section for user-friendly output
- No language mixing
- Cleaner structure

#### Stage 2: Reasoning-Oriented RL

**Differences from R1-Zero**:

```
┌─────────────────────┬──────────────────┬───────────────────────┐
│     Aspect          │   R1-Zero        │       R1 Stage 2      │
├─────────────────────┼──────────────────┼───────────────────────┤
│ Starting Point      │ Base Model       │ Cold-Start SFT Model  │
│ Initial Behavior    │ Random/chaotic   │ Structured reasoning  │
│ Convergence Speed   │ Slower           │ Faster                │
│ Readability         │ Poor             │ Good                  │
│ Language Mixing     │ Frequent         │ Minimal               │
│ Rewards             │ Accuracy+Format  │ Accuracy+Language     │
│                     │                  │ Consistency           │
└─────────────────────┴──────────────────┴───────────────────────┘
```

**Language Consistency Reward**:
```
reward_lang = count(target_language_words) / count(total_words)

Example:
CoT: "首先我们分析 the problem, then we can..."
Target: Chinese
Reward: 0.4 (40% Chinese words)

CoT: "首先我们分析问题，然后我们可以..."
Target: Chinese
Reward: 1.0 (100% Chinese words)
```

**Trade-off**: Slight performance decrease (~1-2%) but better human preference

#### Stage 3: Rejection Sampling & SFT

**Rejection Sampling Process**:

```
For each prompt:
┌────────────────────────────────────────────────────────────┐
│ 1. Sample N responses (N = 4 to 64)                       │
│    from RL checkpoint                                      │
│                                                             │
│ 2. Verify each response:                                   │
│    ┌────────────────────────────────────────────┐         │
│    │ if has_clear_answer_format(response):      │         │
│    │     use rule_based_verification()          │         │
│    │ else:                                       │         │
│    │     use generative_rm(response, gt)        │         │
│    │         # DeepSeek-V3 judges correctness   │         │
│    └────────────────────────────────────────────┘         │
│                                                             │
│ 3. Filter responses:                                       │
│    Remove if:                                              │
│    • has_language_mixing()                                 │
│    • has_long_paragraph_without_breaks()                   │
│    • has_messy_code_blocks()                               │
│                                                             │
│ 4. Keep only correct, clean responses                     │
│                                                             │
│ 5. If no valid response, try with different temperature   │
└────────────────────────────────────────────────────────────┘

Result: ~600k high-quality reasoning samples
```

**Dataset Composition**:

```
Total: ~800k samples
├── 600k Reasoning
│   ├── Math: 200k
│   ├── Code: 250k
│   ├── Science: 100k
│   └── Logic: 50k
│
└── 200k Non-Reasoning
    ├── Writing: 70k
    ├── Factual QA: 50k
    ├── Self-cognition: 20k
    ├── Translation: 30k
    └── General: 30k
```

**Training Configuration**:
- Base Model: DeepSeek-V3-Base (fresh start, not from RL checkpoint)
- Epochs: 2
- Learning Rate: Typical SFT learning rate
- Batch Size: Large (distributed training)

#### Stage 4: RL for All Scenarios

**Multi-Reward System**:

```
┌──────────────────────────────────────────────────────────────┐
│                      Reward Routing                          │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  if task_type == "reasoning":                                │
│      ┌─────────────────────────────────────────┐            │
│      │ Use Rule-Based Rewards                  │            │
│      │ • Accuracy (math, code, logic)          │            │
│      │ • Language consistency                  │            │
│      └─────────────────────────────────────────┘            │
│                                                               │
│  else:  # general tasks                                      │
│      ┌─────────────────────────────────────────┐            │
│      │ Use Neural Reward Models                │            │
│      │                                          │            │
│      │ Helpfulness RM:                         │            │
│      │   Input: Final summary only             │            │
│      │   Output: Helpfulness score             │            │
│      │                                          │            │
│      │ Harmlessness RM:                        │            │
│      │   Input: Entire response                │            │
│      │   Output: Safety score                  │            │
│      └─────────────────────────────────────────┘            │
│                                                               │
│  Final Reward: Weighted combination                          │
└──────────────────────────────────────────────────────────────┘
```

**Why Separate Helpfulness and Harmlessness?**

- **Helpfulness RM** on summary only:
  - Avoids penalizing complex reasoning
  - Focuses on final utility to user
  - Reasoning process can be messy but correct

- **Harmlessness RM** on entire response:
  - Catches harmful reasoning steps
  - Identifies biases in thought process
  - Ensures safety throughout generation

### Comparison: R1-Zero vs R1

```
┌────────────────────────┬──────────────────┬──────────────────────┐
│      Metric            │    R1-Zero       │        R1            │
├────────────────────────┼──────────────────┼──────────────────────┤
│ Training Approach      │ Pure RL          │ SFT + RL + SFT + RL  │
│ Cold-Start Data        │ None             │ Thousands            │
│ Training Stages        │ 1                │ 4                    │
│ AIME 2024 (pass@1)     │ 71.0%            │ 79.8%                │
│ MATH-500               │ 95.9%            │ 97.3%                │
│ GPQA Diamond           │ 73.3%            │ 71.5%                │
│ AlpacaEval 2.0         │ N/A              │ 87.6%                │
│ ArenaHard              │ N/A              │ 92.3%                │
│ Readability            │ Poor             │ Good                 │
│ Language Mixing        │ Frequent         │ Rare                 │
│ General Capabilities   │ Limited          │ Strong               │
│ Training Time          │ Shorter          │ Longer               │
│ Training Complexity    │ Simple           │ Complex              │
└────────────────────────┴──────────────────┴──────────────────────┘
```

### Data Flow Example

```
Complete Flow for Math Question:
═══════════════════════════════════════════════════════════════

Input: "What is the sum of solutions to √(a - √(a+x)) = x?"

Stage 1 (Cold Start SFT):
────────────────────────────
→ Model learns structured format
→ Output has |special_token|, reasoning, summary

Stage 2 (Reasoning RL):
────────────────────────
→ GRPO training on math problems
→ Accuracy reward: Check if answer correct
→ Language reward: Ensure consistent language
→ Model improves: 60% → 75% accuracy

Stage 3 (Rejection Sampling):
──────────────────────────────
→ Generate 64 responses to this question
→ Verify each: "a² - a" is correct answer
→ Filter: Remove mixed language responses
→ Keep: 12 correct, clean responses
→ Add to training set

Stage 3 (SFT on Samples):
──────────────────────────
→ Train on 600k reasoning + 200k general
→ Model learns from diverse high-quality samples
→ Maintains reasoning + gains general capabilities

Stage 4 (Final RL):
───────────────────
→ RL on reasoning: Rule-based rewards
→ RL on general: Neural reward models
→ Model balances all capabilities
→ Final accuracy: 85%+ on similar problems

Final Model Output:
───────────────────
<think>
Let me solve √(a - √(a+x)) = x step by step.

Step 1: Square both sides
a - √(a+x) = x²

Step 2: Isolate the inner square root
√(a+x) = a - x²

Step 3: Square again
a + x = (a - x²)²
a + x = a² - 2ax² + x⁴

Step 4: Rearrange
x⁴ - 2ax² - x + (a² - a) = 0

[Continues with solving and verification...]

Therefore, the sum of solutions is a² - a.
</think>

**Answer**: The sum of the real solutions is **a² - a**.
```

### Configuration

```yaml
Model:
  base: DeepSeek-V3-Base
  architecture: MoE
  total_params: 671B
  activated_params: 37B

Stage 1 - Cold Start:
  data_size: thousands
  training: SFT
  format: "|special_token|<reasoning>|special_token|<summary>"

Stage 2 - Reasoning RL:
  algorithm: GRPO
  rewards:
    - accuracy (rule-based)
    - language_consistency
  training_steps: until_convergence

Stage 3 - Rejection Sampling & SFT:
  sampling:
    responses_per_prompt: 4-64
    verification: rule_based + generative_rm
    filtering: language_mix, formatting
  dataset:
    reasoning: 600k
    non_reasoning: 200k
  training:
    epochs: 2
    base: DeepSeek-V3-Base (fresh)

Stage 4 - Final RL:
  algorithm: GRPO
  rewards:
    reasoning: rule_based
    general:
      helpfulness_rm: summary_only
      harmlessness_rm: full_response
  training_steps: until_convergence

Inference:
  temperature: 0.6
  max_length: 32768
  system_prompt: none
```

---
