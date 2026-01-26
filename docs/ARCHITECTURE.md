# Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HYBRID RAG SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────┐                                                        │
│   │   User Query    │                                                        │
│   └────────┬────────┘                                                        │
│            │                                                                 │
│            ▼                                                                 │
│   ┌─────────────────────────────────────────────────────┐                   │
│   │              HYBRID RETRIEVER                        │                   │
│   │  ┌──────────────────┐    ┌───────────────────┐      │                   │
│   │  │  Dense Retrieval │    │  Sparse Retrieval │      │                   │
│   │  │  ───────────────  │    │  ────────────────  │      │                   │
│   │  │  • all-MiniLM    │    │  • BM25 Algorithm │      │                   │
│   │  │  • FAISS Index   │    │  • Tokenization   │      │                   │
│   │  │  • Cosine Sim    │    │  • TF-IDF weights │      │                   │
│   │  └────────┬─────────┘    └─────────┬─────────┘      │                   │
│   │           │                        │                 │                   │
│   │           └──────────┬─────────────┘                 │                   │
│   │                      ▼                               │                   │
│   │           ┌─────────────────────┐                    │                   │
│   │           │   RRF Fusion        │                    │                   │
│   │           │   k=60              │                    │                   │
│   │           │   ───────────────── │                    │                   │
│   │           │   RRF(d) = Σ 1/(k+r)│                    │                   │
│   │           └──────────┬──────────┘                    │                   │
│   └──────────────────────┼───────────────────────────────┘                   │
│                          │                                                   │
│                          ▼                                                   │
│   ┌─────────────────────────────────────────────────────┐                   │
│   │           RESPONSE GENERATOR                         │                   │
│   │           ──────────────────                         │                   │
│   │           • Flan-T5-base LLM                         │                   │
│   │           • Context Truncation (450 tokens)          │                   │
│   │           • Answer Generation                        │                   │
│   └────────────────────────┬────────────────────────────┘                   │
│                            │                                                 │
│                            ▼                                                 │
│   ┌────────────────────────────────────────────────────┐                    │
│   │                GENERATED ANSWER                     │                    │
│   │                + Source URLs                        │                    │
│   │                + Retrieval Scores                   │                    │
│   └────────────────────────────────────────────────────┘                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Wikipedia (500 URLs)                                                       │
│        │                                                                     │
│        ├── 200 Fixed URLs (diverse categories)                              │
│        └── 300 Random URLs (per indexing run)                               │
│              │                                                               │
│              ▼                                                               │
│   ┌────────────────────┐                                                    │
│   │   Text Extraction  │                                                    │
│   │   (wikipedia-api)  │                                                    │
│   └─────────┬──────────┘                                                    │
│             ▼                                                                │
│   ┌────────────────────┐                                                    │
│   │     Chunking       │                                                    │
│   │  200-400 tokens    │                                                    │
│   │  50-token overlap  │                                                    │
│   └─────────┬──────────┘                                                    │
│             ▼                                                                │
│   ┌────────────────────┐                                                    │
│   │  Index Building    │                                                    │
│   │  • FAISS (dense)   │                                                    │
│   │  • BM25 (sparse)   │                                                    │
│   └────────────────────┘                                                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        EVALUATION PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   100 Q&A Pairs                                                             │
│   ├── Factual (40%)                                                         │
│   ├── Comparative (20%)                                                     │
│   ├── Inferential (25%)                                                     │
│   └── Multi-hop (15%)                                                       │
│              │                                                               │
│              ▼                                                               │
│   ┌─────────────────────────────────────────────────────┐                   │
│   │              METRICS                                 │                   │
│   │  ┌─────────────┐ ┌─────────────┐ ┌───────────────┐  │                   │
│   │  │     MRR     │ │ Faithfulness│ │Context Prec.  │  │                   │
│   │  │  (URL-level)│ │(LLM-Judge)  │ │(Ranking Qual.)│  │                   │
│   │  └─────────────┘ └─────────────┘ └───────────────┘  │                   │
│   └─────────────────────────────────────────────────────┘                   │
│              │                                                               │
│              ▼                                                               │
│   ┌────────────────────┐    ┌────────────────────┐                          │
│   │   Error Analysis   │    │   HTML Reports     │                          │
│   │   & Ablation Study │    │   & Visualizations │                          │
│   └────────────────────┘    └────────────────────┘                          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

| Component | Technology | Purpose |
|-----------|------------|---------|
| Embeddings | all-MiniLM-L6-v2 | 384-dim semantic vectors |
| Vector Index | FAISS (IndexFlatIP) | Cosine similarity search |
| Sparse Search | BM25Okapi | Keyword-based retrieval |
| Fusion | RRF (k=60) | Combines rankings |
| LLM | Flan-T5-base | Answer generation |
| UI | Streamlit | Interactive interface |
| Deployment | Docker | Containerized setup |

## Data Flow

1. **Query Input** → Streamlit UI
2. **Dense Encoding** → all-MiniLM-L6-v2 → FAISS search
3. **Sparse Search** → BM25 tokenization → score calculation
4. **RRF Fusion** → Combine rankings with k=60
5. **Context Selection** → Top-N chunks (truncated to 450 tokens)
6. **Generation** → Flan-T5 produces answer
7. **Response** → Answer + sources + scores
