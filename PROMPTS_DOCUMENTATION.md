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

## User Interaction Templates

### 2. File Upload Template

**Purpose**: This template is used in the official DeepSeek web/app when users upload files. It structures the file content along with the user's question to provide better context for the model to analyze and respond to queries about uploaded documents.

**Use Case**:
- Processing user-uploaded files in web/app interface
- Analyzing documents, code files, or any text-based content
- Providing context-aware responses based on file content

**Key Features**:
- Clear file name and content separation using markers
- Structured format with begin/end markers for file content
- Combines file content with user question in single prompt
- Works with various file types (text, code, documents, etc.)

**Template**:
```
[file name]: {file_name}
[file content begin]
{file_content}
[file content end]
{question}
```

**Variables**:
- `{file_name}` - Name of the uploaded file
- `{file_content}` - Complete content of the uploaded file
- `{question}` - User's question or instruction about the file

**Example Usage**:
```
[file name]: example.py
[file content begin]
def calculate_sum(a, b):
    return a + b

print(calculate_sum(5, 3))
[file content end]
Explain what this code does and suggest improvements.
```

**Source**: README.md, Section 6 (Official Prompts - File Upload)

**Configuration**:
- Temperature: 0.6 (as used in official web/app)
- No system prompt is used
- All instructions contained within user prompt

---

## Web Search Templates

### 3. Web Search Template (Chinese)

**Purpose**: This template is used when performing web searches for Chinese language queries in the official DeepSeek web/app. It provides search results to the model along with instructions on how to cite sources, format responses, and handle different types of queries (listing, creative writing, objective Q&A).

**Use Case**:
- Web search integration for Chinese queries
- RAG (Retrieval-Augmented Generation) for Chinese content
- Providing sourced, verified answers with citations

**Key Features**:
- Structured search results with webpage markers
- Citation format: `[citation:X]` where X is webpage index
- Instructions for different query types (listing, creative, objective)
- Language consistency enforcement
- Current date context for time-sensitive queries
- Guidelines for response structure and readability

**Template**:
```
# 以下内容是基于用户发送的消息的搜索结果:
{search_results}
在我给你的搜索结果中，每个结果都是[webpage X begin]...[webpage X end]格式的，X代表每篇文章的数字索引。请在适当的情况下在句子末尾引用上下文。请按照引用编号[citation:X]的格式在答案中对应部分引用上下文。如果一句话源自多个上下文，请列出所有相关的引用编号，例如[citation:3][citation:5]，切记不要将引用集中在最后返回引用编号，而是在答案对应部分列出。
在回答时，请注意以下几点：
- 今天是{cur_date}。
- 并非搜索结果的所有内容都与用户的问题密切相关，你需要结合问题，对搜索结果进行甄别、筛选。
- 对于列举类的问题（如列举所有航班信息），尽量将答案控制在10个要点以内，并告诉用户可以查看搜索来源、获得完整信息。优先提供信息完整、最相关的列举项；如非必要，不要主动告诉用户搜索结果未提供的内容。
- 对于创作类的问题（如写论文），请务必在正文的段落中引用对应的参考编号，例如[citation:3][citation:5]，不能只在文章末尾引用。你需要解读并概括用户的题目要求，选择合适的格式，充分利用搜索结果并抽取重要信息，生成符合用户要求、极具思想深度、富有创造力与专业性的答案。你的创作篇幅需要尽可能延长，对于每一个要点的论述要推测用户的意图，给出尽可能多角度的回答要点，且务必信息量大、论述详尽。
- 如果回答很长，请尽量结构化、分段落总结。如果需要分点作答，尽量控制在5个点以内，并合并相关的内容。
- 对于客观类的问答，如果问题的答案非常简短，可以适当补充一到两句相关信息，以丰富内容。
- 你需要根据用户要求和回答内容选择合适、美观的回答格式，确保可读性强。
- 你的回答应该综合多个相关网页来回答，不能重复引用一个网页。
- 除非用户要求，否则你回答的语言需要和用户提问的语言保持一致。

# 用户消息为：
{question}
```

**Variables**:
- `{search_results}` - Formatted search results with webpage markers
- `{cur_date}` - Current date for time-sensitive context
- `{question}` - User's original search query

**Search Results Format**:
```
[webpage 1 begin]
Content from first webpage...
[webpage 1 end]

[webpage 2 begin]
Content from second webpage...
[webpage 2 end]
```

**Citation Format**:
- Single source: `[citation:3]`
- Multiple sources: `[citation:3][citation:5]`
- Citations should be inline, not clustered at the end

**Response Guidelines**:
1. **Listing queries**: Limit to 10 key points, inform user to check sources for complete info
2. **Creative tasks** (e.g., essays): Cite sources within paragraphs, provide in-depth analysis
3. **Objective Q&A**: Add 1-2 related sentences if answer is very brief
4. **Structure**: Use paragraphs and sections for long responses (max 5 points)
5. **Multi-source**: Synthesize information from multiple webpages

**Source**: README.md, Section 6 (Official Prompts - Web Search Chinese)

**Configuration**:
- Temperature: 0.6
- No system prompt
- Language: Chinese

---

### 4. Web Search Template (English)

**Purpose**: This template is used when performing web searches for English language queries in the official DeepSeek web/app. It provides the same functionality as the Chinese version but adapted for English language queries and responses.

**Use Case**:
- Web search integration for English queries
- RAG (Retrieval-Augmented Generation) for English content
- Providing sourced, verified answers with citations

**Key Features**:
- Structured search results with webpage markers
- Citation format: `[citation:X]` where X is webpage index
- Instructions for different query types (listing, creative, objective)
- Language consistency enforcement
- Current date context for time-sensitive queries
- Guidelines for response structure and readability

**Template**:
```
# The following contents are the search results related to the user's message:
{search_results}
In the search results I provide to you, each result is formatted as [webpage X begin]...[webpage X end], where X represents the numerical index of each article. Please cite the context at the end of the relevant sentence when appropriate. Use the citation format [citation:X] in the corresponding part of your answer. If a sentence is derived from multiple contexts, list all relevant citation numbers, such as [citation:3][citation:5]. Be sure not to cluster all citations at the end; instead, include them in the corresponding parts of the answer.
When responding, please keep the following points in mind:
- Today is {cur_date}.
- Not all content in the search results is closely related to the user's question. You need to evaluate and filter the search results based on the question.
- For listing-type questions (e.g., listing all flight information), try to limit the answer to 10 key points and inform the user that they can refer to the search sources for complete information. Prioritize providing the most complete and relevant items in the list. Avoid mentioning content not provided in the search results unless necessary.
- For creative tasks (e.g., writing an essay), ensure that references are cited within the body of the text, such as [citation:3][citation:5], rather than only at the end of the text. You need to interpret and summarize the user's requirements, choose an appropriate format, fully utilize the search results, extract key information, and generate an answer that is insightful, creative, and professional. Extend the length of your response as much as possible, addressing each point in detail and from multiple perspectives, ensuring the content is rich and thorough.
- If the response is lengthy, structure it well and summarize it in paragraphs. If a point-by-point format is needed, try to limit it to 5 points and merge related content.
- For objective Q&A, if the answer is very brief, you may add one or two related sentences to enrich the content.
- Choose an appropriate and visually appealing format for your response based on the user's requirements and the content of the answer, ensuring strong readability.
- Your answer should synthesize information from multiple relevant webpages and avoid repeatedly citing the same webpage.
- Unless the user requests otherwise, your response should be in the same language as the user's question.

# The user's message is:
{question}
```

**Variables**:
- `{search_results}` - Formatted search results with webpage markers
- `{cur_date}` - Current date for time-sensitive context
- `{question}` - User's original search query

**Search Results Format**:
```
[webpage 1 begin]
Content from first webpage...
[webpage 1 end]

[webpage 2 begin]
Content from second webpage...
[webpage 2 end]
```

**Citation Format**:
- Single source: `[citation:3]`
- Multiple sources: `[citation:3][citation:5]`
- Citations should be inline, not clustered at the end

**Response Guidelines**:
1. **Listing queries**: Limit to 10 key points, inform user to check sources for complete info
2. **Creative tasks** (e.g., essays): Cite sources within paragraphs, provide in-depth analysis
3. **Objective Q&A**: Add 1-2 related sentences if answer is very brief
4. **Structure**: Use paragraphs and sections for long responses (max 5 points)
5. **Multi-source**: Synthesize information from multiple webpages

**Source**: README.md, Section 6 (Official Prompts - Web Search English)

**Configuration**:
- Temperature: 0.6
- No system prompt
- Language: English

---

## Summary

This documentation covers all official prompts used in the DeepSeek-R1 system:

1. **Training Template**: Used for RL training of DeepSeek-R1-Zero, enforcing structured reasoning output
2. **File Upload Template**: Enables file analysis in web/app interface with structured content presentation
3. **Web Search Templates**: Two language-specific templates (Chinese/English) for RAG-based search integration

### Key Design Principles

- **No System Prompts**: All DeepSeek-R1 prompts avoid system prompts; instructions are included in user prompts
- **Temperature**: Official configuration uses 0.6 temperature to prevent repetitions and maintain coherence
- **Structured Output**: Templates enforce clear structure with markers (tags, sections, citations)
- **Language Consistency**: Responses match query language unless explicitly requested otherwise
- **Citation Standards**: Web search templates require inline citations, not clustered at end

### Usage Recommendations

From official documentation:
1. Set temperature between 0.5-0.7 (0.6 recommended)
2. Avoid adding system prompts - use user prompts only
3. For math problems, include: "Please reason step by step, and put your final answer within \boxed{}."
4. For consistent reasoning, enforce model to start with "\<think\>\n" at the beginning of output
5. Conduct multiple tests and average results for benchmarking

---

**Document Version**: 1.0
**Last Updated**: 2025-11-13
**Based on**: DeepSeek-R1 Paper (arXiv:2501.12948) and Official README
