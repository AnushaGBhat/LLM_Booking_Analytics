from fastapi import FastAPI, HTTPException
import pandas as pd
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import uvicorn
from utils.preprocess import preprocess_data
from utils.vector_search import build_faiss_index, load_chroma_db

# Load and preprocess dataset
DATASET_PATH = "hotel_bookings.csv"
df = preprocess_data(DATASET_PATH)

# Precompute insights
daily_cancellations = df[df['is_canceled'] == 1].groupby('reservation_status_date').size().to_dict()

# Initialize FastAPI application
app = FastAPI(title="AI Booking Analytics", description="API for Booking Analytics & AI-powered Search", version="1.0")

# Load Sentence Transformer Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS Vector Storage
index, metadata = build_faiss_index(df, model)

# ChromaDB for Text Search
collection = load_chroma_db(metadata)

# API Request Model
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

# API Routes
@app.get("/")
def home():
    return {"message": "AI Booking Analytics API is Running 🚀"}

@app.get("/analytics/cancellations/{date}")
def get_cancellations(date: str):
    try:
        date = pd.to_datetime(date).date()
        count = daily_cancellations.get(date, 0)
        return {"date": str(date), "cancellations": int(count)}
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Please use YYYY-MM-DD.")

@app.post("/ask")
def ask_ai(request: QueryRequest):
    query_vector = model.encode(request.query).reshape(1, -1)
    distances, indices = index.search(query_vector, request.top_k)
    results = [metadata[i] for i in indices[0] if i < len(metadata)]
    return {"query": request.query, "results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
