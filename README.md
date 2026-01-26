# Hybrid RAG System

A Docker-based Hybrid Retrieval-Augmented Generation system combining dense vector retrieval (FAISS), sparse keyword retrieval (BM25), and Reciprocal Rank Fusion (RRF) to answer questions from 500 Wikipedia articles.

## 🚀 Quick Start (Docker)

```bash
# Clone the repository
git clone <repository-url>
cd CI_Assignment_2

# Build and run with Docker
docker-compose up --build

# Access the Streamlit UI
open http://localhost:8501
```

## 📋 Features

| Feature | Description |
|---------|-------------|
| **Hybrid Retrieval** | Combines dense semantic search (FAISS) with sparse BM25 |
| **RRF Fusion** | Reciprocal Rank Fusion for optimal result merging |
| **Flan-T5 Generation** | Open-source LLM for answer generation |
| **Diverse Q&A Evaluation** | 100 questions (factual, comparative, inferential, multi-hop) |
| **Innovative Metrics** | MRR, Faithfulness (LLM-as-Judge), Context Precision |

## 🏗️ Project Structure

```
CI_Assignment_2/
├── src/
│   ├── data_collection.py     # Wikipedia URL collection & text extraction
│   ├── preprocessing.py       # Text chunking (200-400 tokens, 50 overlap)
│   ├── dense_retrieval.py     # Sentence embeddings + FAISS
│   ├── sparse_retrieval.py    # BM25 implementation
│   ├── hybrid_retrieval.py    # RRF fusion
│   ├── response_generation.py # Flan-T5 answer generation
│   └── evaluation/
│       ├── question_generator.py
│       ├── metrics.py         # MRR, Faithfulness, Context Precision
│       ├── evaluation_pipeline.py
│       └── report_generator.py
├── app.py                     # Streamlit UI
├── main.py                    # Pipeline orchestrator
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── data/
│   └── fixed_urls.json        # 200 fixed Wikipedia URLs
└── README.md
```

## 🔧 Installation (Local)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## 📖 Usage

### Build Index

```bash
# Build complete index (200 fixed + 300 random URLs)
python main.py --build-index --generate-questions

# Check status
python main.py --status
```

### Run Evaluation

```bash
# Run full evaluation pipeline
python main.py --evaluate

# With ablation study
python main.py --evaluate --ablation
```

### Start UI

```bash
# Local
streamlit run app.py

# Docker
docker-compose up
```

## 📊 Evaluation Metrics

### Mandatory Metric: MRR (Mean Reciprocal Rank)
- **URL-level evaluation**: Measures rank of first correct source URL
- `MRR = average(1/rank)` across all queries

### Custom Metric 1: Faithfulness Score (LLM-as-Judge)
- **Justification**: Detects hallucinations by checking if answers are grounded in context
- **Calculation**: Extract claims → verify each against context → `score = supported/total`
- **Interpretation**: 1.0 = fully grounded, <0.7 = reliability concerns

### Custom Metric 2: Context Precision
- **Justification**: Evaluates retrieval ranking quality beyond simple recall
- **Calculation**: Weighted precision favoring higher-ranked relevant documents
- **Interpretation**: 1.0 = perfect ranking, lower = relevant docs buried

## 📝 Fixed Wikipedia URLs

The 200 fixed URLs are stored in `data/fixed_urls.json` covering diverse topics:
- Science, Technology, History, Geography, Arts
- Philosophy, Literature, Mathematics, Biology, Physics
- Chemistry, Medicine, Economics, Politics, Sports
- Music, Film, Architecture, Psychology, Sociology

## 🐳 Docker Commands

```bash
# Build image
docker-compose build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Run evaluation inside container
docker-compose run app python main.py --evaluate
```

## 📈 Sample Results

| Metric | Score |
|--------|-------|
| MRR | ~0.72 |
| Hit Rate | ~85% |
| Faithfulness | ~0.78 |
| Context Precision | ~0.68 |
| Mean Response Time | ~250ms |

## 🔍 Technology Stack

- **Embeddings**: all-MiniLM-L6-v2 (sentence-transformers)
- **Vector Search**: FAISS
- **Sparse Search**: BM25 (rank-bm25)
- **LLM**: Flan-T5-base (transformers)
- **UI**: Streamlit
- **Visualization**: Plotly

## 📄 License

MIT License - Educational Project
