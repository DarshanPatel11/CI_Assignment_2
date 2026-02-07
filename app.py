"""
Hybrid RAG System - Streamlit Interface

Features:
- Query input
- Generated answer display
- Retrieved chunks with sources
- Dense/sparse/RRF scores
- Response time metrics
"""

import streamlit as st
import time
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Hybrid RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .source-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
    }
    .score-dense { background: #e3f2fd; color: #1565c0; }
    .score-sparse { background: #fff3e0; color: #ef6c00; }
    .score-rrf { background: #e8f5e9; color: #2e7d32; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rag_system():
    """Load and cache the RAG system."""
    from src.hybrid_retrieval import HybridRetriever
    from src.response_generation import RAGPipeline, ResponseGenerator

    data_dir = "data"
    index_name = "hybrid"

    # Check if indices exist
    indices_dir = Path(data_dir) / "indices"
    if not (indices_dir / f"{index_name}_dense.faiss").exists():
        return None, "Indices not found. Please run the indexing pipeline first."

    # Load retriever
    retriever = HybridRetriever(data_dir=data_dir)
    retriever.load_indices(index_name)

    # Load generator
    generator = ResponseGenerator()

    # Create RAG pipeline
    rag = RAGPipeline(retriever, generator)

    return rag, None


def display_retrieval_results(results: dict):
    """Display retrieval results with scores."""

    hybrid_results = results.get('retrieval', {}).get('hybrid_results', [])
    dense_results = results.get('retrieval', {}).get('dense_results', [])
    sparse_results = results.get('retrieval', {}).get('sparse_results', [])

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Hybrid (RRF)", "🧠 Dense", "📝 Sparse (BM25)"])

    with tab1:
        st.markdown("### Hybrid Results (Reciprocal Rank Fusion)")
        for i, r in enumerate(hybrid_results, 1):
            with st.expander(f"**{i}. {r.get('title', 'Unknown')}** — RRF Score: {r.get('score', 0):.4f}"):
                st.markdown(f"**URL:** [{r.get('url', '')}]({r.get('url', '')})")
                st.markdown(f"**Chunk ID:** `{r.get('chunk_id', '')}`")

    with tab2:
        st.markdown("### Dense Retrieval Results")
        for i, r in enumerate(dense_results, 1):
            st.markdown(f"{i}. `{r.get('chunk_id', '')}` — Similarity: **{r.get('score', 0):.4f}**")

    with tab3:
        st.markdown("### Sparse (BM25) Results")
        for i, r in enumerate(sparse_results, 1):
            st.markdown(f"{i}. `{r.get('chunk_id', '')}` — BM25 Score: **{r.get('score', 0):.4f}**")


def display_timing_chart(timing: dict):
    """Display timing breakdown chart."""

    data = {
        'Stage': ['Retrieval', 'Generation', 'Total'],
        'Time (ms)': [
            timing.get('retrieval_ms', 0),
            timing.get('generation_ms', 0),
            timing.get('total_ms', 0)
        ]
    }

    fig = go.Figure(data=[
        go.Bar(
            x=data['Stage'],
            y=data['Time (ms)'],
            marker_color=['#667eea', '#764ba2', '#43a047']
        )
    ])

    fig.update_layout(
        title="Response Time Breakdown",
        xaxis_title="Stage",
        yaxis_title="Time (ms)",
        height=300
    )

    st.plotly_chart(fig, use_container_width=True)


def main():
    # Handle pending query from sample buttons (must be done before widget creation)
    if 'pending_query' in st.session_state:
        st.session_state['_default_query'] = st.session_state.pending_query
        del st.session_state.pending_query

    # Header
    st.markdown('<h1 class="main-header">🔍 Hybrid RAG System</h1>', unsafe_allow_html=True)
    st.markdown("Ask questions about Wikipedia articles using hybrid dense + sparse retrieval")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        top_k = st.slider("Number of chunks to retrieve", 3, 20, 5)

        st.divider()

        st.header("📊 System Status")

        # Load RAG system
        with st.spinner("Loading RAG system..."):
            rag, error = load_rag_system()

        if error:
            st.error(error)
            st.info("Run the following command to build indices:")
            st.code("python main.py --build-index")
            return
        else:
            st.success("✅ RAG System Ready")

            # Show corpus statistics
            corpus_info_path = Path("data/corpus/chunks.json")
            if corpus_info_path.exists():
                with open(corpus_info_path) as f:
                    corpus_info = json.load(f)
                st.metric("Total Chunks", corpus_info.get("total_chunks", 0))
                st.metric("Total Articles", corpus_info.get("total_articles", 0))

        st.divider()

        st.header("📖 About")
        st.markdown("""
        This system combines:
        - **Dense Retrieval**: Semantic embeddings with FAISS
        - **Sparse Retrieval**: BM25 keyword matching
        - **RRF Fusion**: Reciprocal Rank Fusion
        - **Flan-T5**: Answer generation
        """)

    # Main content
    if rag is None:
        st.warning("Please build the indices first using the pipeline.")
        return

    # Query input
    st.markdown("### 💬 Ask a Question")

    # Get default value if set by sample button
    default_query = st.session_state.pop('_default_query', '')

    query = st.text_input(
        "Enter your question:",
        value=default_query,
        placeholder="e.g., What is machine learning?",
        key="query_input"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)

    # Process query
    if search_button and query:
        with st.spinner("Searching and generating answer..."):
            start_time = time.time()

            try:
                result = rag.answer(query, top_k=top_k, include_details=True)
                total_time = (time.time() - start_time) * 1000

                # Display answer
                st.markdown("---")
                st.markdown("### 📝 Answer")
                st.info(result.get('answer', 'No answer generated'))

                # Metrics row
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("⏱️ Total Time", f"{total_time:.0f} ms")
                with col2:
                    st.metric("🔍 Retrieval", f"{result['timing']['retrieval_ms']:.0f} ms")
                with col3:
                    st.metric("🤖 Generation", f"{result['timing']['generation_ms']:.0f} ms")
                with col4:
                    st.metric("📄 Sources", len(result.get('sources', [])))

                st.markdown("---")

                # Two-column layout for results
                left_col, right_col = st.columns([3, 2])

                with left_col:
                    st.markdown("### 📚 Sources Used")
                    for i, source in enumerate(result.get('sources', []), 1):
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>{i}. {source.get('title', 'Unknown')}</strong><br>
                            <a href="{source.get('url', '#')}" target="_blank">{source.get('url', 'No URL')}</a>
                        </div>
                        """, unsafe_allow_html=True)

                with right_col:
                    display_timing_chart(result.get('timing', {}))

                # Detailed retrieval results
                st.markdown("---")
                st.markdown("### 🔎 Retrieval Details")
                display_retrieval_results(result)

                # Context used (expandable)
                with st.expander("📄 Context Used for Generation"):
                    st.text(result.get('context_used', 'No context'))

            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                st.exception(e)

    elif search_button and not query:
        st.warning("Please enter a question.")

    # Sample questions
    st.markdown("---")
    st.markdown("### 💡 Sample Questions")

    sample_questions = [
        "What is the Python programming language?",
        "Who invented the telephone?",
        "What are the main features of machine learning?",
        "Explain the theory of relativity",
        "What is photosynthesis?"
    ]

    cols = st.columns(len(sample_questions))
    for i, (col, q) in enumerate(zip(cols, sample_questions)):
        with col:
            if st.button(q[:30] + "...", key=f"sample_{i}"):
                st.session_state.pending_query = q
                st.rerun()


if __name__ == "__main__":
    main()
