# AI Prompts Documentation for DeepSeek-R1

This document contains all AI prompts used in the DeepSeek-R1 application, grouped by themes with detailed descriptions.

## Table of Contents
1. [Training Templates](#training-templates)
2. [User Interaction Templates](#user-interaction-templates)
3. [Web Search Templates](#web-search-templates)

---

## Training Templates

### 1. DeepSeek-R1-Zero Training Template

**Purpose**: This template is used during the reinforcement learning training phase of DeepSeek-R1-Zero. It guides the base model to structure its output with a thinking process followed by the final answer. This template intentionally avoids content-specific biases to observe the model's natural progression during RL.

**Use Case**:
- Training DeepSeek-R1-Zero model with RL
- Applied directly to base model without SFT
- Enforces structured output format with reasoning and answer separation

**Key Features**:
- Separates reasoning process from final answer
- Uses special tags `<think>` and `<answer>` for structure
- Minimal constraints to allow natural model evolution
- No content-specific biases (e.g., reflective reasoning, specific problem-solving strategies)

**Template**:
```
A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. User: {prompt}. Assistant:
```

**Variables**:
- `{prompt}` - The specific reasoning question or task to be solved

**Output Format**:
```
<think>
[Model's reasoning process here]
</think>
<answer>
[Final answer here]
</answer>
```

**Source**: DeepSeek-R1 Paper, Section 2.2.3 (Training Template), Table 1

**Related Components**:
- Reward Modeling: Accuracy rewards + Format rewards
- RL Algorithm: GRPO (Group Relative Policy Optimization)
- Training Phase: Pure RL without supervised fine-tuning

---
