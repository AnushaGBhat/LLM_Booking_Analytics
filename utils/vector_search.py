import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

def build_faiss_index(df, model):
    text_data = df['hotel'].astype(str) + " " + df['customer_type'].astype(str) + " " + df['reservation_status'].astype(str)
    embeddings = model.encode(text_data.to_list())

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype=np.float32))

    return index, text_data.to_list()

def load_chroma_db(metadata):
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    collection = chroma_client.get_or_create_collection(name="bookings")

    for i, text in enumerate(metadata):
        collection.add(ids=[str(i)], documents=[text])

    return collection
