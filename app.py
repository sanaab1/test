import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv

# Load your API key from the local .env file
load_dotenv()
client = OpenAI()

DATA_FILE = "data.txt"

def load_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}. Make sure data.txt is in your project folder.")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

print("Loading data.txt...")
source_text = load_file(DATA_FILE)

def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

chunks = chunk_text(source_text)
print(f"Created {len(chunks)} chunks from local file.")

# Build the TF-IDF search index
vectorizer = TfidfVectorizer()
embeddings = vectorizer.fit_transform(chunks)
print("TF-IDF vector index built successfully.")

def retrieve_chunks(query, top_k=3):
    query_embedding = vectorizer.transform([query])
    similarities = cosine_similarity(query_embedding, embeddings)
    top_k_indices = np.argsort(similarities[0])[-top_k:][::-1]
    return [chunks[i] for i in top_k_indices]

def generate_answer(query):
    retrieved = retrieve_chunks(query)
    context = "\n\n".join(retrieved)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the following context to answer the user query."
        },
        {
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {query}"
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    print("\n--- RAG Assistant Ready ---")
    while True:
        user_query = input("\nAsk a question about your notes (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        if not user_query.strip():
            continue
        
        answer = generate_answer(user_query)
        print(f"\nFinal Answer:\n{answer}")
