import os
import re
import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from local .env file
load_dotenv()

app = Flask(__name__, static_folder=".")
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

# Build TF-IDF search index on startup
vectorizer = TfidfVectorizer()
embeddings = vectorizer.fit_transform(chunks)
print("TF-IDF vector index built successfully.")

def retrieve_chunks(query, top_k=3):
    query_embedding = vectorizer.transform([query])
    similarities = cosine_similarity(query_embedding, embeddings)
    top_k_indices = np.argsort(similarities[0])[-top_k:][::-1]
    return [chunks[i] for i in top_k_indices]

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question")
    transcript = data.get("transcript", "")
    
    if not question:
        return jsonify({"type": "rhetorical", "answer": "No question provided."})
        
    # 1. Use OpenAI to classify if the utterance is a genuine question or rhetorical
    classifier_prompt = f"""Analyze this utterance from a lecture: "{question}"
Recent transcript context:
{transcript[-1500:]}

Determine if this is a genuine question directed at getting an answer or a rhetorical/transitional statement.
Return JSON format: {{"type": "genuine"}} or {{"type": "rhetorical"}}."""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": classifier_prompt}],
            response_format={"type": "json_object"}
        )
        result = json.loads(completion.choices[0].message.content)
        
        if result.get("type") != "genuine":
            return jsonify({"type": "rhetorical"})
            
        # 2. Retrieve relevant context from your text file and generate the answer
        retrieved = retrieve_chunks(question)
        context = "\n\n".join(retrieved)

        answer_prompt = f"""You are an AI teaching assistant. Answer the student's question using ONLY the provided lecture context from notes. If the answer is not in the context, say so concisely.

Context:
{context}

Question: {question}
Answer:"""

        ans_completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": answer_prompt}]
        )
        
        answer_text = ans_completion.choices[0].message.content
        return jsonify({"type": "genuine", "answer": answer_text})
        
    except Exception as e:
        print("Error:", e)
        return jsonify({"type": "error", "answer": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
