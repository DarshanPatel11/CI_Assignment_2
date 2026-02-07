# Hybrid RAG System

A Docker-based Hybrid Retrieval-Augmented Generation system combining dense vector retrieval (FAISS), sparse keyword retrieval (BM25), and Reciprocal Rank Fusion (RRF) to answer questions from 500 Wikipedia articles.

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/DarshanPatel11/CI_Assignment_2
cd CI_Assignment_2

# Build and run with Docker
docker-compose up --build

# Access the Streamlit UI
open http://localhost:8501
```

### Option 2: Local Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

# Build index and generate questions
python main.py --build-index --generate-questions

# Start the UI
streamlit run app.py
```

## 📋 Features

| Feature | Description |
|---------|-------------|
| **Hybrid Retrieval** | Combines dense semantic search (FAISS) with sparse BM25 |
| **RRF Fusion** | Reciprocal Rank Fusion (k=60) for optimal result merging |
| **Flan-T5 Generation** | Open-source LLM for answer generation |
| **100 Q&A Evaluation** | Factual, comparative, inferential, multi-hop questions |
| **Innovative Metrics** | MRR, Faithfulness (LLM-as-Judge), Context Precision |

## 🏗️ Project Structure

```
CI_Assignment_2/
├── src/
│   ├── data_collection.py      # Wikipedia URL collection & text extraction
│   ├── preprocessing.py        # Text chunking (200-400 tokens, 50 overlap)
│   ├── dense_retrieval.py      # Sentence embeddings + FAISS
│   ├── sparse_retrieval.py     # BM25 implementation
│   ├── hybrid_retrieval.py     # RRF fusion
│   ├── response_generation.py  # Flan-T5 answer generation
│   ├── evaluation/
│   │   ├── question_generator.py
│   │   ├── metrics.py          # MRR, Faithfulness, Context Precision
│   │   ├── evaluation_pipeline.py
│   │   ├── report_generator.py
│   │   └── visualizations.py
│   └── creative_evaluation/
│       ├── adversarial.py      # Adversarial testing
│       ├── ablation.py         # Ablation studies
│       ├── llm_judge.py        # LLM-as-Judge evaluation
│       └── confidence.py       # Confidence calibration
├── app.py                      # Streamlit UI
├── main.py                     # Pipeline orchestrator
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── data/
│   ├── fixed_urls.json         # 200 fixed Wikipedia URLs
│   ├── corpus/                 # Extracted articles and chunks
│   ├── indices/                # FAISS and BM25 indices
│   └── evaluation/
│       ├── questions.json      # 100 Q&A evaluation pairs
│       └── results/            # Evaluation outputs
└── report/
    ├── REPORT.md               # Final comprehensive report
    ├── ARCHITECTURE.md         # Architecture diagram
    └── screenshots/            # UI screenshots
```

## 📖 Usage

### Build Index

```bash
# Build complete index (200 fixed + 300 random URLs)
python main.py --build-index --generate-questions

# Check system status
python main.py --status
```

### Run Evaluation

```bash
# Standard evaluation (100 questions)
python main.py --evaluate

# With ablation study
python main.py --evaluate --ablation

# Full innovative evaluation
python main.py --evaluate --innovative --ablation --num-questions 100
```

### Start UI

```bash
# Local
streamlit run app.py

# Docker
docker-compose up
```

## 📊 Evaluation Results

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **MRR (URL-level)** | **0.913** | Excellent source retrieval |
| **Hit Rate** | **95%** | Very high recall |
| **Faithfulness** | 0.32 | Room for improvement |
| **Context Precision** | 0.68 | Good ranking quality |
| **Mean Response Time** | ~250ms | Fast response |

### Metrics Description

#### MRR (Mean Reciprocal Rank) - Mandatory

- **Purpose**: Measures rank of first correct source URL
- **Calculation**: `MRR = average(1/rank)` across all queries
- **Interpretation**: 1.0 = perfect, >0.9 = excellent

#### Faithfulness Score (LLM-as-Judge) - Custom 1

- **Justification**: Detects hallucinations by checking if answers are grounded in context
- **Calculation**: Extract claims → verify each against context → `score = supported/total`
- **Interpretation**: 1.0 = fully grounded, <0.7 = reliability concerns

#### Context Precision - Custom 2

- **Justification**: Evaluates retrieval ranking quality beyond simple recall
- **Calculation**: Weighted precision favoring higher-ranked relevant documents
- **Interpretation**: 1.0 = perfect ranking, lower = relevant docs buried

## 📝 Fixed Wikipedia URLs

The 200 fixed Wikipedia URLs are stored in [`data/fixed_urls.json`](data/fixed_urls.json).

Topics covered include:
- Science, Technology, History, Geography, Arts
- Philosophy, Literature, Mathematics, Biology, Physics
- Chemistry, Medicine, Economics, Politics, Sports
- Music, Film, Architecture, Psychology, Sociology

## 🐳 Docker Commands

```bash
# Build image
docker-compose build

# Run in foreground
docker-compose up

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Run evaluation in Docker
docker-compose run --rm app python main.py --evaluate --num-questions 100

# Full innovative evaluation in Docker
docker-compose run --rm app python main.py --evaluate --innovative --ablation
```

## 🔍 Technology Stack

| Component | Technology |
|-----------|------------|
| **Embeddings** | all-MiniLM-L6-v2 (sentence-transformers) |
| **Vector Search** | FAISS (faiss-cpu) |
| **Sparse Search** | BM25 (rank-bm25) |
| **LLM** | Flan-T5-base (transformers) |
| **UI** | Streamlit |
| **Visualization** | Plotly |
| **Containerization** | Docker |

## 📦 Dependencies

All required libraries are listed in `requirements.txt`:

- sentence-transformers>=2.2.2
- faiss-cpu>=1.7.4
- rank-bm25>=0.2.2
- transformers>=4.35.0
- wikipedia-api>=0.6.0
- beautifulsoup4>=4.12.0
- nltk>=3.8.1
- rouge-score>=0.1.2
- bert-score>=0.3.13
- scikit-learn>=1.3.0
- streamlit>=1.28.0
- plotly>=5.18.0

## 📄 Report

The comprehensive final report is available at [`report/REPORT.md`](report/REPORT.md) and includes:

- Architecture diagram
- Evaluation results with tables
- Metric justifications and interpretations
- Ablation study results
- Error analysis
- 6 UI screenshots

## 📄 License

MIT License - Educational Project
