
import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------
app = FastAPI(
    title="Customer Review Insight Engine API",
    version="1.0.0"
)

# --------------------------------------------------
# Request Model
# --------------------------------------------------
class QueryRequest(BaseModel):
    question: str
    n_results: int = 5

# --------------------------------------------------
# Load API Key
# --------------------------------------------------
groq_api_key = os.getenv("GROQ_API_KEY")

client = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1"
)

# --------------------------------------------------
# Load Data and Model on Startup
# --------------------------------------------------
print("Loading resources...")

project_path = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(project_path, "data", "negative_reviews.csv")

df = pd.read_csv(data_path)

documents = (
    df["reviewText"]
    .dropna()
    .astype(str)
    .head(5000)
    .tolist()
)

model = SentenceTransformer("all-MiniLM-L6-v2")
document_embeddings = model.encode(
    documents,
    show_progress_bar=False
)

print("Resources loaded successfully!")

# --------------------------------------------------
# Retrieval Function
# --------------------------------------------------
def retrieve_reviews(query, n_results=5):
    query_embedding = model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        document_embeddings
    )[0]

    top_indices = similarities.argsort()[-n_results:][::-1]

    return [documents[i] for i in top_indices]

# --------------------------------------------------
# RAG Function
# --------------------------------------------------
def generate_analysis(question, n_results=5):
    reviews = retrieve_reviews(question, n_results)

    context = "\n\n".join(
        [f"Review {i+1}:\n{doc}" for i, doc in enumerate(reviews)]
    )

    prompt = f"""
You are a senior business analyst.

Analyze the following customer reviews and answer the question.

Question:
{question}

Customer Reviews:
{context}

Provide:
1. Executive Summary
2. Key Complaint Themes
3. Business Recommendations
4. Priority Actions
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content

# --------------------------------------------------
# Health Check Endpoint
# --------------------------------------------------
@app.get("/")
def home():
    return {
        "message": "Customer Review Insight Engine API is running!"
    }

# --------------------------------------------------
# Analysis Endpoint
# --------------------------------------------------
@app.post("/analyze")
def analyze(request: QueryRequest):
    answer = generate_analysis(
        request.question,
        request.n_results
    )

    return {
        "question": request.question,
        "analysis": answer
    }
