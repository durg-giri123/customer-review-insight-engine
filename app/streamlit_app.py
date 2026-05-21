
import os
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Review Insight Engine",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Title and Description
# -----------------------------
st.title("📊 Customer Review Insight Engine")
st.markdown(
    "AI-powered business insights from customer reviews using "
    "Semantic Search + RAG + Groq LLM."
)

# -----------------------------
# Load API Key from Streamlit Secrets
# -----------------------------
groq_api_key = st.secrets["GROQ_API_KEY"]

# Initialize Groq client (OpenAI-compatible)
client = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1"
)

# -----------------------------
# Load Data and Models
# -----------------------------
@st.cache_resource
def load_resources():
    # Project root = parent of app/
    project_path = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(project_path, "data", "negative_reviews.csv")

    # Load dataset
    df = pd.read_csv(data_path)

    # Use first 5000 negative reviews
    documents = (
        df["reviewText"]
        .dropna()
        .astype(str)
        .head(5000)
        .tolist()
    )

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate embeddings
    embeddings = model.encode(
        documents,
        show_progress_bar=False
    )

    return documents, model, embeddings

documents, model, document_embeddings = load_resources()

# -----------------------------
# Semantic Search Function
# -----------------------------
def retrieve_reviews(query, n_results=5):
    query_embedding = model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        document_embeddings
    )[0]

    top_indices = similarities.argsort()[-n_results:][::-1]

    return [documents[i] for i in top_indices]

# -----------------------------
# RAG Function
# -----------------------------
def ask_business_analyst(question, n_results=5):
    # Retrieve relevant reviews
    reviews = retrieve_reviews(question, n_results)

    # Build context
    context = "\n\n".join(
        [f"Review {i+1}:\n{doc}" for i, doc in enumerate(reviews)]
    )

    # Prompt
    prompt = f"""
You are a senior business analyst.

Analyze the following customer reviews and answer the question.

Question:
{question}

Customer Reviews:
{context}

Provide your response in the following format:

1. Executive Summary
2. Key Complaint Themes
3. Business Recommendations
4. Priority Actions
"""

    # Generate response using Groq
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("📌 Example Questions")

example_questions = [
    "What are the top customer complaints?",
    "Why are customers unhappy?",
    "What should the company improve first?",
    "Summarize the main product defects.",
    "What installation issues are most common?"
]

for q in example_questions:
    st.sidebar.write("• " + q)

# -----------------------------
# Main Input Area
# -----------------------------
default_question = (
    "What are the top customer complaints and "
    "what should the company improve first?"
)

question = st.text_area(
    "Ask a business question:",
    value=default_question,
    height=120
)

# -----------------------------
# Analyze Button
# -----------------------------
if st.button("🚀 Analyze Reviews", use_container_width=True):
    with st.spinner("Analyzing customer reviews..."):
        answer = ask_business_analyst(question)

    st.success("Analysis completed successfully! 🎉")

    st.markdown("## 📊 AI Business Analysis")
    st.markdown(answer)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "Built by Durgesh Giri using NLP, Sentence Transformers, "
    "Semantic Search, RAG, and Groq LLM."
)
