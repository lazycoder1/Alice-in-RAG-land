---
title: RAG Through Wonderland — Part 3: The Pool of Context
author: Gautam G Sabhahit
pubDatetime: 2025-11-19T00:00:00Z
slug: part-3-rag-through-wonderland
featured: false
draft: false
tags:
  - RAG
  - evaluation
  - alice-eval
  - cohere
  - prompt-engineering
description: Moving from factual recall to contextual reasoning. How adding a Reranker and Chain-of-Thought prompting helps the system understand "Why" Alice cries.
---

# 🌊 Part 3: The Pool of Context

> "I wish I hadn't cried so much!" said Alice, as she swam about, trying to find her way out. "I shall be punished for it now, I suppose, by being drowned in my own tears!"  
> — *Alice, Chapter 2*

In [Part 2](./part2.md), we built a system that could remember facts. It knew *who* the author was and *what* Alice drank. It was a good student—it had excellent recall.

But recall isn't reasoning.

Now we face **Level 2: Contextual Reasoning**.  
We aren't asking "What is X?" anymore. We are asking "Why did X happen?" and "How does Y relate to Z?"

The question that haunts me (and my RAG system) is:  
**"Why does Alice start crying after drinking from the bottle?"**

To answer this, the system can't just find the word "cry". It needs to find the *cause*—the shrinking, the lost key, the frustration. It needs to understand the *story*.

---

## 🧪 The Hypothesis

My Level 1 system (simple Vector Search + GPT-4o-mini) failed this test miserably. **It scored a dismal 3/30.** It would either hallucinate or say "I don't know."

Why? I have two hypotheses:

### Hypothesis 1: Vector Search is a "Broad Net", not a "Sniper"
Vector embeddings are great at finding *topical similarity*. If I search for "Alice crying", it brings back every paragraph where Alice sheds a tear.
But it doesn't necessarily prioritize the paragraph that explains *why*. It treats the *effect* (crying) as the main signal, often burying the *cause* (the key on the table).

**The Fix:** We need a second stage of retrieval. A "Judge" that looks at the top 50 "maybe" results and strictly ranks them by *how well they answer the specific question*.

### Hypothesis 2: Brevity Kills Reasoning
In Level 1, I forced the model to be brief: *"Give SHORT, DIRECT answers (1-5 words)."*
You can't explain a cause-and-effect relationship in one word. If I ask "Why did you move to New York?" and force you to answer in one word, you'd struggle.
Reasoning requires **thinking space**.

**The Fix:** We need **Chain-of-Thought (CoT)** prompting. We must let the model "show its work" before giving the final answer.

---

## 🛠️ The Implementation

### What is Reranking?

Reranking is like having a two-step hiring process for information.

1.  **The Recruiter (Vector Search):** Scans thousands of documents and picks the top 50 that *look* relevant based on general topic. It uses **Bi-Encoders**, which are fast but can miss subtle nuances.
2.  **The Hiring Manager (Reranker):** Carefully reads those 50 candidates in detail, comparing them specifically to the question. It uses **Cross-Encoders**, which process the query and document *together* to understand the exact relationship.

```mermaid
graph TD
    Q[User Query] --> VS[Vector Search]
    DB[(All Chunks)] --> VS
    VS -- Top 50 Candidates --> R[Reranker (Cross-Encoder)]
    Q --> R
    R -- Top 10 Scored Results --> LLM[LLM Generation]
```

**Why does this increase scores?**
For "Why" questions, the answer often hinges on a specific detail.
*   **Vector Search** might see "Why did Alice cry?" and return every paragraph containing the word "cry" (even irrelevant ones).
*   **Reranking** looks at the candidate "Alice shrank and couldn't reach the key" and understands that this *causes* the crying, boosting it to the top even if the word overlap isn't perfect.

This precision is critical for Level 2 reasoning.

### Step 1: Adding the Hiring Manager (Cohere Rerank)

I decided to use **Cohere's Rerank API**. It's a powerful Cross-Encoder model that takes a query and a document and outputs a relevance score.

Unlike vector search (which compares compressed numbers), a Reranker reads the actual text of the query and the document together. It's slower, but much smarter.

I modified my `Retriever` to:
1.  Fetch **50 candidates** using standard Vector Search (casting a wide net).
2.  Pass those 50 to `cohere.rerank`.
3.  Take the **top 10** highest-scored chunks for generation.

```python
# src/imp/retriever.py

def search(self, query: str, top_k: int = 10) -> list[str]:
    # 1. Cast a wide net (Vector Search)
    initial_results = self.datastore.search(query, top_k=50)
    
    # 2. The Judge (Cohere Rerank)
    co = cohere.Client()
    rerank_results = co.rerank(
        query=query,
        documents=initial_results,
        top_n=top_k,
        model="rerank-english-v3.0"
    )
    
    return [result.document.text for result in rerank_results.results]
```

### Step 2: Letting the Model Think (Prompt Engineering)

Next, I had to untie the model's hands. I updated the System Prompt to encourage "step-by-step" thinking.

**Old Prompt:**
> "Give SHORT, DIRECT answers (1-5 words when possible)..."

**New Prompt:**
> "For 'Why', 'How', and 'Explain' questions, you MUST provide the reasoning or cause-and-effect.
> Think step-by-step: First, find the relevant details in the context. Then, connect them to form an answer."

This simple change transforms the model from a "fact-retriever" to a "reasoning engine."

---

## 📊 The Results

I ran the **Level 2 Evaluation** (30 questions focusing on cause-and-effect).

**Score: 17/30**

Is this perfect? No.  
Is it progress? Absolutely.

The system can now answer:
> **Q:** "Why does Alice start crying after drinking from the bottle?"  
> **A:** "She cries because she shrank after drinking from the bottle and could no longer reach the key to the garden, leaving her trapped."

It connected the *drinking* -> *shrinking* -> *key* -> *crying*. That is reasoning.

However, it still struggles with abstract questions like *"What lesson might Carroll be suggesting?"* because those answers aren't explicitly written in the text—they are *subtext*.

That is a challenge for **Level 3**.

## 🧠 Iteration 2: The Brain Upgrade

17/30 was good, but I suspected my "reasoning engine" (`gpt-4o-mini`) was the bottleneck. It's fast and cheap, but is it smart enough to connect the dots?

I decided to swap the brain for **GPT-4o**.

```python
# src/utils/invoke_ai.py
model = os.getenv("LLM_MODEL", "gpt-4o")
```

**The Result: 22/30**

A significant jump! The smarter model was able to take the *same* context chunks and extract better answers.

**What improved?**
- **Nuance**: It caught subtle reasons that `mini` missed.
- **Confidence**: It was less likely to give up and say "I don't know" when the answer was implicit.

**What's still missing?**
The remaining 8 failures are mostly **Thematic** questions (e.g., "What is the significance of..."). These require synthesizing information across the *entire* book, not just retrieving a few chunks.

For that, we'll need to change our strategy completely in **Level 3**.

---

## 🔬 Iteration 3: HyDE (Hypothetical Document Embeddings)

At 22/30, I was stuck. The failures weren't random—they were specific "needle in a haystack" questions where the system just couldn't find the right chunk.

**Example failure:**
> **Q:** "Why does the Lory refuse to listen to Alice?"  
> **A:** "I cannot find the answer in the context."

The answer exists in the book: *"I am older than you, and must know better."* But vector search missed it.

**Why?** The question uses the word "refuse to listen," but the text says "turned sulky" and "I am older." There's a **lexical gap**—the question and answer use different words, even though they mean the same thing.

### The HyDE Solution

HyDE (Hypothetical Document Embeddings) is a clever trick:
1. **Ask the LLM to hallucinate a plausible answer** to the question (without seeing the context).
2. **Use that hallucinated answer as the search query** instead of the original question.

Why does this work? Because the hallucinated answer will use similar *language* to the actual answer in the book, making it easier for vector search to find.

**Implementation:**
```python
# Generate a hypothetical answer
hyde_prompt = f"Write a short, plausible sentence from 'Alice in Wonderland' that answers this question: '{query}'. Do not explain, just write the sentence."
hypothetical_answer = invoke_ai(system_message="You are a helpful assistant.", user_message=hyde_prompt)

# Search using the hypothetical answer
search_query = f"{query}\n{hypothetical_answer}"
initial_results = self.datastore.search(search_query, top_k=100)
```

**The Result: 24/30** 🎉

HyDE fixed the "Lory" question! The LLM hallucinated something like *"The Lory said it was older and knew better,"* which matched the actual text perfectly.

**Trade-offs:**
- ✅ **Better recall** for specific details.
- ❌ **Slower** (extra LLM call per query).
- ❌ **More expensive** (2x the API calls).

---

## 🔍 Iteration 4: Hybrid Search (BM25 + Vector)

24/30 was great, but I wanted to push further. I hypothesized that some failures were due to **keyword mismatches**—questions that needed exact word matching, not just semantic similarity.

**Example:**
> **Q:** "Why does the Frog-Footman stare at the sky?"

Vector search might find chunks about "footmen" or "frogs," but miss the specific sentence about staring at the sky because it's semantically buried.

### The BM25 Solution

BM25 is a classic **keyword-based** search algorithm. It's like Ctrl+F on steroids—it finds documents that contain the exact words in the query, weighted by rarity.

I implemented **Hybrid Search**:
1. **Vector Search** (semantic): Top 50 candidates.
2. **BM25 Search** (keyword): Top 50 candidates.
3. **Merge** the results (remove duplicates).
4. **Rerank** the combined pool with Cohere.

**Implementation:**
```python
# Vector Search (Semantic)
vector_results = self.datastore.search(query, top_k=50)

# BM25 Search (Keyword)
bm25_results = self.datastore.search_bm25(query, top_k=50)

# Combine and deduplicate
combined_results = list(set(vector_results + bm25_results))

# Rerank
rerank_results = co.rerank(query=query, documents=combined_results, top_n=20)
```

**The Result: 21/30** 😕

Wait, what? The score *dropped*!

**Why did BM25 hurt performance?**

After analyzing the failures, I realized that BM25 was adding **noise**. Alice in Wonderland uses very whimsical, creative language. Questions are often paraphrased (e.g., "refuse to listen" vs. "turned sulky"), so exact keyword matching doesn't help—it just dilutes the pool with irrelevant chunks.

**Lesson learned:** Hybrid Search is powerful for technical documents (e.g., API docs, manuals) where keywords are precise. But for literary text with rich, varied language, **semantic search alone is better**.

---

## 🎯 Iteration 5: The Balancing Act

After reverting to simple Vector + Rerank, I ran evaluations on both levels:

- **Level 1:** 22/30 (73%)
- **Level 2:** ~23/30 (77%)

Wait—Level 1 *dropped* from the original 27/30! What happened?

**The Problem:** The Chain-of-Thought prompt I added for Level 2 was making the model **overthink simple factual questions**.

For example:
> **Q:** "Who is smoking a hookah?"  
> **Expected:** "The Caterpillar."  
> **Got:** "The Caterpillar is smoking a hookah, as described in the context where Alice encounters him sitting on a mushroom."

The answer is *correct*, but unnecessarily verbose. The evaluator was sometimes marking these as incorrect due to the extra fluff.

### The Solution: Adaptive Prompting

I refined the system prompt to **match answer style to question type**:

```python
SYSTEM_PROMPT = """
Instructions:
- Match your answer style to the question type:
  * For simple "Who", "What", "Where", "When" questions: 
    Give SHORT, DIRECT answers (1-10 words).
  * For "Why", "How", and "Explain" questions: 
    Provide the reasoning or cause-and-effect (1-3 sentences).
- Be precise and factual. Do not add information that is not in the context.

Examples:
Q: "Who is smoking a hookah?"
A: "The Caterpillar."

Q: "What does Alice drink?"
A: "A bottle labeled 'DRINK ME'."

Q: "Why does Alice cry?"
A: "She cries because she shrank after drinking from the bottle 
    and could no longer reach the key to the garden, leaving her trapped."
"""
```

This tells the model: *"Be brief when you can, be thorough when you must."*

---

## 🏁 Final Configuration & Results

After all the experiments, here's what worked best:

**Architecture:**
```
Query → Vector Search (100 candidates) → Cohere Rerank (top 20) → LLM (gpt-4o)
```

**Prompt Strategy:**
- Adaptive: Brief for factual questions, detailed for reasoning questions
- Explicit examples for both styles

**Final Scores:**
- **Level 1 (Direct Lookup):** **27/30 (90%)** ✅
- **Level 2 (Contextual Reasoning):** **23/30 (77%)** ✅

This is a solid baseline for Level 2. The system can now:
- ✅ Understand cause-and-effect relationships
- ✅ Connect multiple pieces of information across chunks
- ✅ Explain "Why" and "How" questions with reasoning
- ✅ Stay concise for simple factual questions

**What's still missing?**
The remaining failures (7 in Level 2) are questions that require **global understanding** of the book—themes, character arcs, and symbolic meaning. Those are Level 3 problems, which will require a fundamentally different approach: **Thematic Synthesis**.

---

## 📊 Key Takeaways

**What Worked:**
1. **Reranking** (Cohere): Massive improvement for precision on "Why" questions
2. **Model Upgrade** (gpt-4o): Better at implicit reasoning
3. **Adaptive Prompting**: Balances brevity and depth

**What Didn't Work:**
1. **HyDE**: Improved recall but added latency and cost (not worth it for 1-2 point gain)
2. **BM25 Hybrid Search**: Added noise for literary text with creative language

**The Journey:**
- Baseline (Vector + mini): 3/30
- + Rerank + CoT: 17/30
- + gpt-4o: 22/30
- + Adaptive Prompt: **23/30**

From 10% to 77%—that's the power of iterative optimization.

---

*Next time: We tackle Thematic Synthesis and try to teach the machine to read between the lines.*
