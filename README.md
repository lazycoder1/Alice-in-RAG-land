# 🕳️ AliceEval: A Journey to Master RAG through Wonderland

*A progressive learning framework for understanding how Retrieval-Augmented Generation systems evolve from simple recall to deep reasoning.*

---

## 🌱 Project Overview

**AliceEval** is a personal learning project built around *Lewis Carroll’s* **“Alice’s Adventures in Wonderland.”**  
The goal is to progressively build and evaluate a Retrieval-Augmented Generation (RAG) system — starting from basic factual recall and advancing to abstract, multi-domain reasoning.

Each level of AliceEval introduces a new category of questions and reasoning depth.  
As I move through the levels, I’ll iteratively improve the system until it consistently scores **10/10 on each stage**, developing an intuitive and practical understanding of how retrieval systems learn, fail, and evolve.

---

## 🎯 Why “Alice in Wonderland”?

Carroll’s classic offers the perfect testbed for RAG exploration:
- It’s **public domain** and compact enough for quick iteration.
- It contains **facts, logic puzzles, and symbolic themes** ideal for testing different reasoning modes.
- Its **dreamlike logic and recurring motifs** naturally expose how well models handle ambiguity, inconsistency, and abstraction.

By grounding all experiments in one coherent narrative, I can isolate retrieval and reasoning performance without dataset bias or domain shift.

---

## 🧩 The 5 Levels of AliceEval

Each level represents a distinct cognitive and retrieval challenge — a step deeper into Wonderland.  
The focus is on *building and improving*, not just answering correctly.

| Level | Focus Area | Example Questions | What I’ll Be Building / Learning |
|-------|-------------|------------------|----------------------------------|
| **1 – Factual Recall** | Retrieve exact answers from the book | “What does Alice drink to shrink?” | Build a basic vector RAG with embeddings + retriever |
| **2 – Contextual Reasoning** | Understand short cause-effect and story details | “Why did Alice cry after shrinking?” | Experiment with chunk size, reranking, and query rephrasing |
| **3 – Thematic Synthesis** | Connect events and ideas across chapters | “How do Alice’s size changes reflect emotional growth?” | Implement summarization, context stitching, and hierarchical retrieval |
| **4 – Relational Reasoning** | Multi-hop and relationship-based understanding | “How do the Queen, Rabbit, and Cat represent different types of control?” | Explore Graph RAG, entity linking, and multi-step reasoning |
| **5 – External Knowledge Integration** | Combine story facts with real-world or literary context | “How does Victorian culture shape Carroll’s satire of authority?” | Add external retrieval sources, cross-corpus RAG, and agentic orchestration |

---

## 🧠 What This Framework Tests

AliceEval measures the growth of a RAG system along **three cognitive dimensions**:

1. **Retrieval Depth**  
   How well can the system find relevant fragments as the conceptual gap between question and answer widens?

2. **Reasoning Fidelity**  
   Does the model synthesize retrieved information truthfully — or hallucinate when context is incomplete?

3. **Knowledge Integration**  
   Can the system combine multiple knowledge sources (book + background) into a cohesive, well-grounded response?

---

## 🧭 Learning Philosophy

Rather than a static benchmark, **AliceEval** is a *guided learning journey*.  
Each level mirrors the way human understanding deepens — from remembering facts → connecting contexts → abstracting meaning → integrating external knowledge.

At every stage, I’ll:
- Evaluate the system’s performance on that level’s questions,  
- Diagnose what limits retrieval or reasoning,  
- Introduce one new improvement (e.g., reranker, summarization chain, graph reasoning),  
- Re-test until I hit a **10/10 score** for that level.

> 💡 This iterative loop — *evaluate → improve → re-evaluate* — is the core of the project.

---

## 🧩 What Success Looks Like

- **Level 1:** The model answers literal, text-based questions flawlessly.  
- **Level 2:** It captures short causal and contextual links.  
- **Level 3:** It summarizes and connects symbolic ideas across the book.  
- **Level 4:** It reasons over relationships and multi-hop chains.  
- **Level 5:** It contextualizes Carroll’s work using external sources.

When all levels reach 10/10, the system should demonstrate a complete retrieval-to-reasoning pipeline — capable of moving from surface understanding to interpretive synthesis.

---

## 📘 Learning Log (WIP)

| Iteration | Level | Key Improvement | Score (/10) | Notes |
|------------|--------|----------------|--------------|-------|
| 1 | 1 | Basic embedding + Chroma | 6/10 | Retrieved chunks correctly but answer formatting inconsistent |
| 2 | 2 | Added MMR retriever | 7.5/10 | Improved multi-sentence recall |
| 3 | 3 | Introduced Map-Reduce summarization | – | Upcoming |
| ... | ... | ... | ... | ... |

---

## 🧩 Why This Matters

Most RAG tutorials focus on *implementation* — not *understanding*.  
**AliceEval** is designed to slow things down, break reasoning into levels, and show exactly *where* retrieval pipelines fail and *why* they improve.

By progressing through the five stages, the project aims to uncover:
- How retrieval noise affects reasoning quality  
- When to use summarization vs graph connections  
- How external corpora improve interpretability  
- And what “understanding” truly means in a retrieval-augmented system

---

> 🕰️ “It’s no use going back to yesterday, because I was a different person then.”  
> — *Alice, Chapter 10*

---

*This project is for learning, exploration, and insight — not competition. Each step down the rabbit hole brings the system closer to understanding not just Wonderland, but how reasoning itself can evolve.*
