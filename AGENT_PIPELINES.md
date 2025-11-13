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

## Distillation Pipeline

### Overview
The distillation pipeline transfers reasoning capabilities from the large DeepSeek-R1 model (671B params) to smaller dense models (1.5B-70B params). This approach is more efficient than training small models with RL and produces better results.

### Pipeline Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                    Distillation Pipeline                          │
└───────────────────────────────────────────────────────────────────┘

Step 1: Teacher Model
┌──────────────────────────────────┐
│      DeepSeek-R1 (Teacher)       │
│      • 671B total params         │
│      • 37B activated params      │
│      • Strong reasoning (79.8% AIME) │
│      • General capabilities      │
└────────┬─────────────────────────┘
         │
         ▼
Step 2: Generate Training Data
┌─────────────────────────────────────────────────────────────────┐
│  Use R1 Checkpoint from Stage 3 (Rejection Sampling)            │
│  (Same 800k dataset as R1 Stage 3 SFT)                         │
│                                                                  │
│  Dataset Composition:                                           │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ Reasoning: ~600k samples                                    ││
│  │ ├── Math: 200k                                              ││
│  │ ├── Code: 250k                                              ││
│  │ ├── Science: 100k                                           ││
│  │ └── Logic: 50k                                              ││
│  │                                                              ││
│  │ Non-Reasoning: ~200k samples                                ││
│  │ ├── Writing: 70k                                            ││
│  │ ├── Factual QA: 50k                                         ││
│  │ ├── Self-cognition: 20k                                     ││
│  │ ├── Translation: 30k                                        ││
│  │ └── General: 30k                                            ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  Key: These are R1-generated reasoning traces                   │
│  - Long chain-of-thought                                        │
│  - Verification steps                                           │
│  - Structured format                                            │
└────────┬────────────────────────────────────────────────────────┘
         │
         ▼
Step 3: Select Base Models
┌─────────────────────────────────────────────────────────────────┐
│  Open-Source Dense Models                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Qwen Series:                                             │   │
│  │ • Qwen2.5-Math-1.5B                                      │   │
│  │ • Qwen2.5-Math-7B                                        │   │
│  │ • Qwen2.5-14B                                            │   │
│  │ • Qwen2.5-32B                                            │   │
│  │                                                           │   │
│  │ Llama Series:                                            │   │
│  │ • Llama-3.1-8B-Base                                      │   │
│  │ • Llama-3.3-70B-Instruct                                 │   │
│  │   (chosen for better reasoning than 3.1)                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Note: Configs and tokenizers slightly modified                │
└────────┬────────────────────────────────────────────────────────┘
         │
         ▼
Step 4: Supervised Fine-Tuning (No RL)
┌─────────────────────────────────────────────────────────────────┐
│  For each base model:                                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. Load base model                                       │   │
│  │ 2. Fine-tune on 800k R1-generated samples               │   │
│  │ 3. Train until convergence                               │   │
│  │ 4. Evaluate on benchmarks                                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Training Details:                                              │
│  • Method: Supervised fine-tuning only (no RL)                 │
│  • Objective: Mimic R1's reasoning patterns                    │
│  • Loss: Standard language modeling loss                       │
│  • No additional RL even though it could improve results       │
│    (left for community exploration)                            │
└────────┬────────────────────────────────────────────────────────┘
         │
         ▼
Step 5: Distilled Models
┌─────────────────────────────────────────────────────────────────┐
│  DeepSeek-R1-Distill Family                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Model               │ Base           │ AIME  │ MATH-500 │   │
│  ├────────────────────┼────────────────┼───────┼──────────┤   │
│  │ R1-Distill-Qwen-1.5B│ Qwen2.5-Math  │ 28.9% │  83.9%   │   │
│  │ R1-Distill-Qwen-7B  │ Qwen2.5-Math  │ 55.5% │  92.8%   │   │
│  │ R1-Distill-Llama-8B │ Llama-3.1     │ 50.4% │  89.1%   │   │
│  │ R1-Distill-Qwen-14B │ Qwen2.5       │ 69.7% │  93.9%   │   │
│  │ R1-Distill-Qwen-32B │ Qwen2.5       │ 72.6% │  94.3%   │   │
│  │ R1-Distill-Llama-70B│ Llama-3.3     │ 70.0% │  94.5%   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Key Achievements:                                              │
│  • 7B model beats GPT-4o on math tasks                         │
│  • 14B model surpasses QwQ-32B-Preview                         │
│  • 32B and 70B models set new records for dense models         │
│  • 32B comparable to o1-mini on most benchmarks                │
└─────────────────────────────────────────────────────────────────┘
```

### Distillation vs RL on Small Models

**Experiment**: Training Qwen-32B with RL vs Distillation

```
┌──────────────────────────────────────────────────────────────────┐
│              Distillation vs RL Comparison (Qwen-32B)            │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Approach 1: RL on Qwen-32B-Base (R1-Zero-Qwen-32B)             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Method:                                                     │ │
│  │ • Large-scale RL on Qwen2.5-32B-Base                       │ │
│  │ • Same GRPO algorithm as R1-Zero                           │ │
│  │ • Math, code, STEM data                                    │ │
│  │ • 10,000+ training steps                                   │ │
│  │                                                             │ │
│  │ Results:                                                    │ │
│  │ • AIME 2024: 47.0% (pass@1), 60.0% (cons@64)              │ │
│  │ • MATH-500: 91.6%                                          │ │
│  │ • GPQA: 55.0%                                              │ │
│  │ • LiveCodeBench: 40.2%                                     │ │
│  │                                                             │ │
│  │ Performance: Similar to QwQ-32B-Preview                    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  Approach 2: Distillation from R1                                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Method:                                                     │ │
│  │ • SFT on 800k R1-generated samples                         │ │
│  │ • No RL stage                                              │ │
│  │ • Much less compute than RL                                │ │
│  │                                                             │ │
│  │ Results:                                                    │ │
│  │ • AIME 2024: 72.6% (pass@1), 83.3% (cons@64)              │ │
│  │ • MATH-500: 94.3%                                          │ │
│  │ • GPQA: 62.1%                                              │ │
│  │ • LiveCodeBench: 57.2%                                     │ │
│  │                                                             │ │
│  │ Performance: Significantly better (+25% AIME, +3% MATH)    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  Conclusion:                                                      │
│  • Distillation >> RL for small models                          │
│  • Reasoning patterns from larger models are crucial            │
│  • Small models can't discover complex patterns via RL alone    │
│  • Distillation is more compute-efficient                       │
└──────────────────────────────────────────────────────────────────┘
```

### Data Flow Example

```
Example: Distilling to Qwen-7B
═══════════════════════════════════════════════════════════════

Step 1: Prompt from Dataset
────────────────────────────────────
Question: "Solve for x: 2^x = 16"

Step 2: R1 Generated Response (Teacher)
────────────────────────────────────────────
<think>
To solve 2^x = 16, I need to express 16 as a power of 2.

I know that:
• 2^1 = 2
• 2^2 = 4
• 2^3 = 8
• 2^4 = 16

So 16 = 2^4.

Therefore, 2^x = 2^4, which means x = 4.

Let me verify: 2^4 = 2 × 2 × 2 × 2 = 16 ✓
</think>

**Answer**: x = 4

Step 3: Training Qwen-7B (Student)
───────────────────────────────────────
Input to student: "Solve for x: 2^x = 16"
Target output: [R1's complete response above]

Training objective: Minimize cross-entropy loss between
student output and teacher output

Step 4: After Training
───────────────────────
Student (Qwen-7B) can now generate similar reasoning:

Input: "Solve for x: 2^x = 16"
Output:
<think>
To solve this, I'll express 16 as a power of 2.
16 = 2 × 8 = 2 × 2 × 4 = 2 × 2 × 2 × 2 = 2^4
So x = 4.
</think>
**Answer**: x = 4

→ Student learned to mimic teacher's reasoning pattern!
```

### Why Distillation Works

```
┌──────────────────────────────────────────────────────────────────┐
│                   Key Insights on Distillation                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. Pattern Transfer                                             │
│     ┌────────────────────────────────────────────────────┐      │
│     │ Large models (671B) can discover complex           │      │
│     │ reasoning patterns through RL:                     │      │
│     │ • Self-verification                                │      │
│     │ • Multiple solution approaches                     │      │
│     │ • Error detection and correction                   │      │
│     │                                                     │      │
│     │ Small models (7B-32B) struggle to find these       │      │
│     │ patterns via RL, but can learn them through        │      │
│     │ imitation (SFT)                                    │      │
│     └────────────────────────────────────────────────────┘      │
│                                                                   │
│  2. Efficiency                                                   │
│     ┌────────────────────────────────────────────────────┐      │
│     │ RL Training: Thousands of steps × expensive        │      │
│     │              environment interactions              │      │
│     │                                                     │      │
│     │ Distillation: Standard SFT on fixed dataset       │      │
│     │               Much faster and cheaper              │      │
│     └────────────────────────────────────────────────────┘      │
│                                                                   │
│  3. Quality                                                      │
│     ┌────────────────────────────────────────────────────┐      │
│     │ R1's reasoning traces are high-quality:            │      │
│     │ • Correct answers (filtered via rejection)         │      │
│     │ • Clean formatting                                 │      │
│     │ • No language mixing                               │      │
│     │ • Comprehensive reasoning                          │      │
│     │                                                     │      │
│     │ Better than what small model RL could produce      │      │
│     └────────────────────────────────────────────────────┘      │
│                                                                   │
│  4. Generalization                                               │
│     ┌────────────────────────────────────────────────────┐      │
│     │ 800k diverse samples cover:                        │      │
│     │ • Multiple reasoning types                         │      │
│     │ • Various difficulty levels                        │      │
│     │ • Different output formats                         │      │
│     │                                                     │      │
│     │ Student learns general reasoning, not just         │      │
│     │ specific problem-solving                           │      │
│     └────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────┘
```

### Future Improvements

From the paper: "Incorporating RL could substantially boost model performance."

```
Potential Enhanced Pipeline:
────────────────────────────────────────────────

Current: Base Model → Distill SFT → Distilled Model

Future: Base Model → Distill SFT → RL → Better Distilled Model
                                     ↑
                                     │
                                Uses reasoning
                                patterns from
                                distillation as
                                starting point

Expected Benefits:
• Further performance gains
• Better adaptation to specific tasks
• Continued improvement beyond teacher
```

### Performance Summary

```
┌────────────────────────────────────────────────────────────────────┐
│        DeepSeek-R1-Distill Models vs Baselines                     │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Benchmark: AIME 2024 (pass@1)                                     │
│  ┌───────────────────────────────────────────────────────────┐    │
│  │ GPT-4o-0513:                9.3%                           │    │
│  │ Claude-3.5-Sonnet-1022:    16.0%                          │    │
│  │ ───────────────────────────────────────                   │    │
│  │ R1-Distill-Qwen-1.5B:      28.9%  ← Beats GPT-4o!        │    │
│  │ R1-Distill-Llama-8B:       50.4%                          │    │
│  │ R1-Distill-Qwen-7B:        55.5%  ← Beats GPT-4o by 6×!  │    │
│  │ OpenAI-o1-mini:            63.6%                          │    │
│  │ R1-Distill-Qwen-14B:       69.7%  ← Beats o1-mini!       │    │
│  │ R1-Distill-Llama-70B:      70.0%                          │    │
│  │ R1-Distill-Qwen-32B:       72.6%  ← New SOTA for dense!  │    │
│  │ OpenAI-o1-1217:            79.2%                          │    │
│  │ DeepSeek-R1:               79.8%                          │    │
│  └───────────────────────────────────────────────────────────┘    │
│                                                                     │
│  Key Takeaway: Even 7B distilled model significantly outperforms   │
│  much larger general-purpose models on reasoning tasks             │
└────────────────────────────────────────────────────────────────────┘
```

### Configuration

```yaml
Teacher Model:
  model: DeepSeek-R1 (Stage 3 checkpoint)
  data: 800k high-quality samples
  composition:
    reasoning: 600k
    non_reasoning: 200k

Student Models:
  qwen_series:
    - Qwen2.5-Math-1.5B
    - Qwen2.5-Math-7B
    - Qwen2.5-14B
    - Qwen2.5-32B
  llama_series:
    - Llama-3.1-8B-Base
    - Llama-3.3-70B-Instruct

Training:
  method: Supervised Fine-Tuning only
  no_rl: true  # Left for community to explore
  loss: Cross-entropy (language modeling)
  config_modifications: slight (tokenizer, configs)

Evaluation:
  temperature: 0.6
  sampling: pass@1 and cons@64
  benchmarks:
    - AIME 2024
    - MATH-500
    - GPQA Diamond
    - LiveCodeBench
    - Codeforces
```

---

## Inference Pipelines

### Web/App Inference Pipeline

```
User Query Flow in Official DeepSeek Web/App:
══════════════════════════════════════════════════════════════

Input: User types question in web interface

         │
         ▼
┌─────────────────────────┐
│  Query Classification   │
│  - Regular query?       │
│  - File upload?         │
│  - Web search needed?   │
└────────┬────────────────┘
         │
         ├─→ Regular Query
         │   └→ No special template, direct to model
         │
         ├─→ File Upload
         │   │
         │   ▼
         │   ┌────────────────────────────────────────┐
         │   │ Apply File Upload Template:            │
         │   │ [file name]: {name}                    │
         │   │ [file content begin]                   │
         │   │ {content}                              │
         │   │ [file content end]                     │
         │   │ {user_question}                        │
         │   └────────┬───────────────────────────────┘
         │            │
         │            └→ Send to model
         │
         └─→ Web Search
             │
             ▼
             ┌─────────────────────────────────────────┐
             │ 1. Perform web search                   │
             │ 2. Fetch top N results                  │
             │ 3. Format as:                           │
             │    [webpage 1 begin]...[webpage 1 end]  │
             │    [webpage 2 begin]...[webpage 2 end]  │
             │ 4. Detect query language                │
             │ 5. Apply appropriate template:          │
             │    - Chinese: search_answer_zh_template │
             │    - English: search_answer_en_template │
             └────────┬────────────────────────────────┘
                      │
                      └→ Send to model

         │ (All paths converge)
         ▼
┌──────────────────────────────────────────────────────────┐
│  DeepSeek-R1 Model Inference                            │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Configuration:                                      │ │
│  │ • Temperature: 0.6                                  │ │
│  │ • Max length: 32,768 tokens                        │ │
│  │ • No system prompt                                  │ │
│  │ • Enforce start with "<think>\n"                   │ │
│  │   (for consistent reasoning)                       │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  Model generates:                                       │
│  <think>                                                │
│  [reasoning process...]                                 │
│  </think>                                               │
│  [summary/answer]                                       │
└────────┬─────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│  Response Post-Processing                                │
│  • Extract summary (user sees this primarily)           │
│  • Format markdown                                       │
│  • Add citations (if web search)                        │
│  • Thinking process available in expandable section     │
└────────┬─────────────────────────────────────────────────┘
         │
         ▼
    Display to User
```

### API Inference Pipeline

```
API Request Flow:
═══════════════════════════════════════════════════════════

Client Request:
POST /v1/chat/completions
{
  "model": "deepseek-reasoner",
  "messages": [
    {"role": "user", "content": "Solve: 2x + 5 = 13"}
  ],
  "temperature": 0.6,
  "max_tokens": 32768
}

         │
         ▼
┌──────────────────────────────────────────────────────────┐
│  API Gateway                                             │
│  • Authenticate                                          │
│  • Rate limit                                            │
│  • Route to inference cluster                           │
└────────┬─────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│  Inference Engine                                        │
│  • Load model shard on GPU cluster                      │
│  • Apply temperature, top-p, etc.                       │
│  • Generate tokens autoregressively                     │
│  • Stop at max_tokens or EOS                            │
└────────┬─────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│  Response Formatting                                     │
│  • Stream or complete response                          │
│  • Include reasoning_content field (optional)           │
│  • Standard OpenAI-compatible format                    │
└────────┬─────────────────────────────────────────────────┘
         │
         ▼
API Response:
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "deepseek-reasoner",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "**Answer**: x = 4",
      "reasoning_content": "<think>...</think>"
    }
  }],
  "usage": {...}
}
```

### Evaluation Pipeline

```
Benchmark Evaluation Flow:
═══════════════════════════════════════════════════════════

┌──────────────────────────┐
│  Benchmark Dataset       │
│  (AIME, MATH, GPQA, ...) │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│  For each question:                                      │
│  ┌────────────────────────────────────────────────────┐ │
│  │ 1. Generate N responses (N = 4 to 64)             │ │
│  │    with temperature = 0.6, top_p = 0.95           │ │
│  │                                                     │ │
│  │ 2. For each response:                              │ │
│  │    • Extract answer                                │ │
│  │    • Verify correctness                            │ │
│  │                                                     │ │
│  │ 3. Calculate metrics:                              │ │
│  │    pass@1 = (1/N) * Σ(correctness_i)              │ │
│  │    cons@64 = majority_vote(all_responses)          │ │
│  └────────────────────────────────────────────────────┘ │
└────────┬─────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│  Aggregate Results                                       │
│  • Average across all questions                         │
│  • Report pass@1 and cons@64                            │
│  • Compare against baselines                            │
└──────────────────────────────────────────────────────────┘

Why pass@1 with temperature > 0?
────────────────────────────────────────────────────────────
Greedy decoding (temp=0) causes:
• High repetition rates
• Significant variability across checkpoints
• Unstable performance

Sampling with temp=0.6:
• More reliable estimates
• Averages out randomness
• Better represents model capability
```

---

## Summary

This document describes three main pipelines and inference flows for DeepSeek-R1:

### Training Pipelines

1. **DeepSeek-R1-Zero**: Pure RL approach
   - Single-stage training
   - No supervised data
   - Demonstrates emergence of reasoning through RL
   - Achieves 71% on AIME 2024

2. **DeepSeek-R1**: Multi-stage approach
   - Stage 1: Cold Start SFT
   - Stage 2: Reasoning-oriented RL
   - Stage 3: Rejection Sampling & SFT
   - Stage 4: RL for all scenarios
   - Achieves 79.8% on AIME 2024 (matches o1-1217)

3. **Distillation**: Transfer to small models
   - Uses R1-generated data (800k samples)
   - SFT only, no RL
   - Highly effective: 7B beats GPT-4o, 32B rivals o1-mini
   - More efficient than training small models with RL

### Inference Flows

4. **Web/App**: User-facing interface with special templates
5. **API**: OpenAI-compatible endpoint
6. **Evaluation**: Benchmark testing with pass@k metrics

### Key Insights

- **RL alone works**: R1-Zero proves reasoning emerges from pure RL
- **SFT+RL better**: R1's multi-stage approach improves performance and quality
- **Distillation superior for small models**: Beats RL-training small models directly
- **No system prompts**: All prompts are user-level instructions
- **Temperature matters**: 0.6 prevents repetition while maintaining quality
- **Patterns transfer**: Large model reasoning can be distilled to small models

---

**Document Version**: 1.0
**Last Updated**: 2025-11-13
**Based on**: DeepSeek-R1 Paper (arXiv:2501.12948) and Official README
