---
title: RAG Through Wonderland — Part 2: Finding the Rabbit Hole (Level 1)
author: Gautam G Sabhahit
pubDatetime: 2025-11-17T00:00:00Z
slug: part-2-rag-through-wonderland
featured: false
draft: false
tags:
  - RAG
  - chunking
  - embeddings
  - evaluation
  - alice-eval
ogImage: ../../assets/images/posts/rag-through-wonderland/part-2/cover.png
description: Building and optimizing a RAG system for Level 1 factual recall—from 12/30 to 27/30 through iterative improvements in chunking, retrieval, and evaluation.
canonicalURL: https://example.org/rag-through-wonderland-part-2
---

*Part 2 of the "RAG Through Wonderland" Series*

> "Curiouser and curiouser!"  
> — *Alice, Chapter 2*

---

## 🎯 The Challenge: Can You Remember Wonderland?

Level 1 is deceptively simple: **Answer 30 factual recall questions about Alice's Adventures in Wonderland.**

Questions like:
- "Who is the author of Alice's Adventures in Wonderland?"
- "What does the Caterpillar smoke?"
- "What time is it always at the Mad Tea Party?"

No reasoning. No interpretation. Just pure *retrieval accuracy*.

If a RAG system can't recall basic facts, there's no point chasing deeper mysteries.

**Target Score: 30/30 (100%)**  
**Actual Score: Well... let's just say the rabbit hole was deeper than expected.**

---

## 🏗️ The Initial Setup

### Architecture

I built a classic RAG pipeline with five components:

```
┌─────────────┐
│   Indexer   │ -> Chunks documents, creates DataItems
└──────┬──────┘
       ↓
┌─────────────┐
│  Datastore  │ -> Embeds chunks, stores in LanceDB
└──────┬──────┘
       ↓
┌─────────────┐
│  Retriever  │ -> Searches for relevant chunks
└──────┬──────┘
       ↓
┌─────────────┐
│  Response   │ -> Generates answer with GPT-4o-mini
│  Generator  │
└──────┬──────┘
       ↓
┌─────────────┐
│  Evaluator  │ -> Judges correctness with GPT-4o-mini
└─────────────┘
```

### Initial Configuration

- **Document Processing:** Docling's `DocumentConverter` + `HybridChunker`
- **Chunk Size:** 256 tokens (default)
- **Embeddings:** OpenAI `text-embedding-3-small` (1536 dims)
- **Vector Store:** LanceDB
- **Retrieval:** Top-k = 5 chunks
- **Generation:** GPT-4o-mini
- **Evaluation:** GPT-4o-mini as judge

Simple, clean, textbook RAG.

---

## 🚨 The First Run: A Disappointing Start

```bash
✅ Indexed 42 chunks from 1 documents
✨ Total Score: 12/30
```

**40% accuracy.** With a functioning pipeline.

The basic setup was working — documents indexed, embeddings generated, chunks retrieved — but the results were abysmal. Only 12 out of 30 simple factual questions answered correctly.

### What Was Wrong?

The default configuration was too conservative:
- **256-token chunks** were too small, fragmenting context
- **top_k=5** retrieved too few chunks, missing relevant information  
- **Prompts** weren't optimized for short, direct answers

**Lesson 1: Default configurations are starting points, not solutions.**  
Even a "working" RAG pipeline can fail spectacularly if not tuned for the task.

---

## 🔍 The Investigation: Why Are We Failing?

I added a `--limit` flag to test individual questions and inspect retrieved chunks:

```bash
python main.py evaluate -n 1  # Test first question only
python main.py evaluate --start 25 -n 1  # Test question 25
```

This revealed three types of failures:

### 1. **Retrieval Failures** (Wrong chunks retrieved)
- Query: "What does the Duchess throw at Alice?"
- Retrieved: Mentions of "frying-pan" being thrown
- Missed: The chunk where she throws "the baby"

### 2. **Evaluation Question Issues** (Ambiguous or incorrect expected answers)
- Q29: "Who is the executioner at the Queen's court?"
- Expected: "The executioner" ← Circular!
- The book never names them, just calls them "the executioner"

### 3. **LLM Non-determinism** (Scores varied 22-24/30 across runs)
- Same configuration produced different results
- GPT-4o-mini's temperature introduces randomness

---

## 🔧 The Optimization Journey

### Iteration 1: Increase Chunk Size

**Hypothesis:** Many questions require understanding full sentences or paragraphs. 256-token chunks might be fragmenting critical context, causing the retriever to miss complete answers.

**Experiment:** Increased chunk size to 1024 tokens to preserve more context per chunk:

```python
# Before
self.chunker = HybridChunker(max_tokens=256)

# After  
self.chunker = HybridChunker(max_tokens=1024)
```

**Critical:** Had to re-index after changing chunk size. (At one point I forgot this step and the database was empty - a debugging reminder that pipeline changes require re-indexing!)

**Result: 12/30 -> 22/30** (+10 points!) 🚀

**Why it worked:** Larger chunks preserved complete thoughts and relationships. Questions like "What does the Caterpillar smoke?" now retrieved full scenes instead of sentence fragments. This was the single biggest improvement—confirming that context window size is critical for factual recall.

---

### Iteration 2: Increase top_k Retrieval

**Hypothesis:** With only 5 chunks retrieved, relevant information might be ranked 6th, 7th, or lower. Increasing retrieval breadth should improve the chance of capturing the right context.

**Experiment:** Gradually increased retrieval:

```python
# Evolution
top_k = 5   ->  10  ->  15  ->  20
```

**Result: 22/30 -> 24/30** (+2 points)

**Why it worked:** More chunks meant better coverage of the document. Questions where the answer was semantically similar to multiple passages now had a higher chance of retrieving the correct one. Diminishing returns appeared after 20 chunks.

---

### Iteration 3: Switch to Ollama Embeddings

**Hypothesis:** OpenAI embeddings are optimized for general text, but perhaps a different embedding model would better capture the literary style and semantic relationships in a 19th-century novel. Also, local embeddings = no API costs.

**Experiment:** Switched to Ollama's `mxbai-embed-large` (1024 dims):

```python
# datastore.py
EMBEDDING_MODEL = "mxbai-embed-large"  # 1024 dims

def get_vector(self, content: str) -> List[float]:
    response = ollama.embed(
        model=self.EMBEDDING_MODEL,
        input=content
    )
    return response["embeddings"][0]
```

**Result: 24/30 -> 24/30** (no change)

**Why it worked (or didn't):** Performance was identical to OpenAI embeddings. Ollama's `mxbai-embed-large` captured the semantic relationships just as well for factual recall. The hypothesis that domain-specific performance would differ was wrong—but I gained free, local embeddings with no loss in quality.

**Key insight:** Embedding model choice matters less than expected for simple factual recall. Always benchmark before paying for premium services.

---

### Iteration 4: Refine LLM Prompts

**Hypothesis:** The LLM was generating verbose, explanatory answers when the questions demanded short, factual responses. Better prompts could guide it to extract precise information without over-explaining.

**Experiment:** Rewrote prompts to be laser-focused on short, direct answers:

#### Response Generator Prompt:
```python
SYSTEM_PROMPT = """
You are answering factual questions about Alice's Adventures in Wonderland.

Instructions:
- Give SHORT, DIRECT answers (1-5 words when possible)
- Extract the answer even if mentioned briefly or indirectly
- Look for synonyms, descriptions, or related mentions
- Be confident - if context contains relevant info, extract it
- ONLY say "I cannot find the answer" if context truly has NO relevant info

Examples:
Q: "What does Alice drink?" -> A: "A bottle labeled DRINK ME"
Q: "Who smokes a hookah?" -> A: "The Caterpillar"
"""
```

#### Evaluator Prompt:
```python
SYSTEM_PROMPT = """
You are evaluating answers about Alice's Adventures in Wonderland.

Instructions:
- Check if RESPONSE contains same CORE INFORMATION as expected answer
- Be FLEXIBLE with wording - focus on factual correctness, not exact phrasing
- Accept equivalent answers (e.g., "White Rabbit" = "a White Rabbit")
- Accept longer answers if they include the key fact

Examples of CORRECT matches:
- Expected: "Lewis Carroll" | Response: "The author is Lewis Carroll" -> TRUE
- Expected: "DRINK ME" | Response: "A bottle labeled 'DRINK ME'" -> TRUE
"""
```

**Result: 24/30 -> 25/30** (+1 point)

**Why it worked:** The new prompts emphasized extracting exact information without elaboration. Instructions like "Give SHORT, DIRECT answers (1-5 words)" and concrete examples helped the LLM match the expected answer format. The evaluator prompt became more flexible, accepting equivalent phrasings instead of demanding exact matches.

---

### Iteration 5: Fix Bad Evaluation Questions

**Hypothesis:** Some "failures" weren't retrieval or generation issues—they were bad test questions. If the expected answers were wrong or questions were poorly worded, the system was being penalized for correct responses.

**Investigation:** Manually reviewed failing questions and discovered three fundamentally flawed ones:

**Q16: "What animal's baby turns into a pig?"**
- Expected: "The Duchess"
- Issue: The Duchess is not an animal—the question is semantically incorrect
- Fixed to: "Whose baby turns into a pig?" 
- The system's original response "The Duchess's baby" was more accurate than the expected answer!

**Q29: "Who is the executioner at the Queen's court?"**
- Expected: "The executioner" ← Circular reference
- Fixed to: "Who does the King want to execute the Cheshire Cat?"

**Q30: "What does Alice call the court proceedings?"**
- Expected: "Nonsense"
- Actual quote: "Stuff and nonsense!"
- Fixed expected answer to match the exact text

**Result: 25/30 -> 28/30** (+3 points) 🎉

**Why it worked:** These weren't system improvements—they were corrections to the ground truth. The system was already performing better than the test indicated. This highlighted a critical lesson: **bad evaluation data creates false negatives**. Always validate your test dataset!

---

### Iteration 6: Attempted Overlapping Chunks (Failed)

**Hypothesis:** Some answers might span chunk boundaries. Creating overlapping "bridge" chunks (last 30% of chunk N + first 30% of chunk N+1) would capture context that gets split during chunking.

**Experiment:** Generated overlapping chunks between consecutive chunks:

```python
# Create overlap: last 30% of chunk N + first 30% of chunk N+1
overlap_text = " ".join(
    current_words[-overlap_size:] + next_words[:overlap_size]
)
```

**Result: 42 chunks -> 83 chunks**  
**Score: 28/30 -> 23/30** (-5 points!) ❌

**Why it failed:** The overlapping chunks added noise without adding value. The semantic search retrieved multiple similar overlapping chunks for the same query, diluting the retrieval quality. The hypothesis that boundary-spanning was a major issue was wrong—1024-token chunks were already large enough to capture complete thoughts.

**Reverted.** Sometimes less is more. Clean, non-overlapping chunks outperformed redundant overlap.

---

## 📊 Final Configuration

After 6 major iterations, here's what worked:

| Component | Configuration | Rationale |
|-----------|--------------|-----------|
| **Chunk Size** | 1024 tokens | Preserves more context without fragmentation |
| **Overlap** | None | Added noise, decreased performance |
| **Embeddings** | Ollama `mxbai-embed-large` (1024d) | Outperformed OpenAI for this dataset |
| **Vector Store** | LanceDB | Fast, simple, local |
| **Retrieval** | top_k = 20 | Casts wide net for factual questions |
| **Generation** | GPT-4o-mini | Strong, cost-effective |
| **Evaluation** | GPT-4o-mini as judge | Flexible semantic matching |

---

## 🎯 Final Results

```bash
✨ Total Score: 28/30 (93.3%)
```

**Consistent range: 27-28/30 (90-93%)** across multiple runs.

### Remaining Failures (2 questions)

**Q21:** "Who are the three gardeners painting the roses?"  
- Response: "Five, Seven, and Two" (their names!)
- Expected: "Playing cards"
- *More specific but not matching expected abstraction level*

**Q25:** "What does the Duchess throw at Alice?"  
- Response: "A frying-pan"
- Expected: "The baby"
- *Multiple things thrown in that scene; retrieval found wrong one*

These are **edge cases** that represent the limits of pure semantic search. Potential approaches to explore:
1. **Hybrid search (BM25 + vector)** — hypothesis: exact keyword matching might catch "the baby" vs "frying-pan"
2. **Reranking** — hypothesis: scoring chunks by relevance might surface the right context
3. **Better question design** — make queries more specific to guide retrieval

*(Note: These remain untested hypotheses for future iterations)*

---

## 🧠 Key Lessons from Level 1

### 1. **Retrieval is the Bottleneck**
The vast majority of failures were retrieval issues, not generation issues. When I inspected failing questions, the wrong chunks were being retrieved—not that the LLM was misinterpreting the right chunks. If the right context isn't retrieved, even GPT-4 can't save you.

### 2. **Chunk Size Matters More Than You Think**
256 tokens -> 1024 tokens was a **+10 point** improvement (12/30 -> 22/30)—the single biggest win. Too small = fragmented context. Too large = noisy retrieval. Finding the right balance is critical.

### 3. **Local Embeddings Can Compete**
Ollama's `mxbai-embed-large` (free, local) performed just as well as OpenAI's `text-embedding-3-small` (paid, cloud) on this task. **Always benchmark.** More expensive doesn't guarantee better results.

### 4. **Evaluation Quality Shapes System Quality**
Bad questions = misleading metrics. Fixing Q16, Q29, and Q30 improved the score from 25/30 to 28/30—not by improving the system, but by fixing incorrect expected answers. Some "failures" were actually the system being MORE correct than the evaluation dataset. Always validate your ground truth!

### 5. **More Data ≠ Better Performance**
Overlapping chunks (83 chunks) performed worse than clean chunks (42 chunks). Signal-to-noise ratio matters.

### 6. **LLM Non-determinism is Real**
Score variance of ±2 points across runs is normal. Test multiple times before drawing conclusions.

---

## 🔮 What's Next: Level 2 Preview

Level 1 tested **factual recall** — can you find and reproduce exact information?

**Level 2 will test contextual reasoning** — can you understand *why* things happened?

Questions like:
- "Why did Alice start crying after drinking from the bottle?"
- "What caused the tea party to become stuck in time?"
- "How did Alice offend the Mouse?"

These require:
- **Multi-chunk retrieval** (cause + effect across passages)
- **Temporal reasoning** (understanding sequence)
- **Narrative coherence** (connecting events logically)

Potential approaches to explore:
- Query expansion for better retrieval
- Reranking to prioritize causal relationships
- Graph-based retrieval for event chains

The real question: Will Level 1's optimizations be enough, or will contextual reasoning demand fundamentally new techniques?

---

## 📈 Progress Tracker

| Metric | Initial | Final | Target |
|--------|---------|-------|--------|
| **Score** | 12/30 (40%) | 28/30 (93.3%) | 30/30 (100%) |
| **Chunks** | 0 -> 42 | 42 | - |
| **Chunk Size** | 256 tokens | 1024 tokens | - |
| **top_k** | 5 | 20 | - |
| **Embeddings** | OpenAI | Ollama | - |

**Status: Level 1 - 93% Complete** ✅

28/30 is excellent for a basic RAG system! The remaining 2 points represent edge cases where:
- Q21: The system gives more specific information (card names vs. "playing cards")
- Q25: Multiple objects thrown in scene, retrieval found the wrong one

These would likely require hybrid search (BM25 + vector) or reranking to solve consistently.

---

> "Begin at the beginning, and go on till you come to the end: then stop."  
> — *The King, Chapter 12*

Level 1 complete. Time to chase the next rabbit hole.

---

*Next in the series: **Part 3: The Pool of Context (Level 2)** — where retrieval becomes narrative comprehension.*

---

## 📦 Resources

**Code & Evaluation Dataset:**  
[GitHub Repository - alice-in-wonderland](https://github.com/lazycoder1/Alice-in-RAG-land)  
**Branch:** [`lvl-one`](https://github.com/lazycoder1/Alice-in-RAG-land/tree/lvl-one)

This branch contains:
- Complete Level 1 implementation
- 30-question factual recall evaluation dataset (with corrected ground truth)
- All optimization iterations documented in commit history
- Configuration for reproducible 28/30 (93%) performance

