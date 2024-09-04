from flask import Flask, render_template, request, jsonify
import os
import textwrap
import numpy as np
import pandas as pd
import PyPDF2
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Set up Google Gemini API
API_KEY = 'AIzaSyCrbeU4QGZYHXR2AIAfeiko5AN8NCerQ24'  # Replace with your actual API key
genai.configure(api_key=API_KEY)

# Helper functions (same as before)
def embed_content(title, text, model='models/embedding-001', task_type='retrieval_document'):
    response = genai.embed_content(model=model, content=text, task_type=task_type, title=title)
    return response["embedding"]

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def create_chunks(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def find_top_chunks(query, dataframe, top_n=3, model='models/embedding-001'):
    query_response = genai.embed_content(model=model, content=query, task_type='retrieval_query')
    query_embedding = query_response["embedding"]

    document_embeddings = np.stack(dataframe['Embeddings'])
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]

    return dataframe.iloc[top_indices]['Text'].tolist()

def make_prompt(query, relevant_passages):
    passages = " ".join(relevant_passages)
    escaped = passages.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = textwrap.dedent(f"""\
    You are a helpful and informative bot that answers questions using text from the reference passages included below. \
    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
    strike a friendly and conversational tone. \
    If the passage is irrelevant to the answer, you may ignore it.
    QUESTION: '{query}'
    PASSAGES: '{escaped}'

    ANSWER:
    """)
    return prompt

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.pdf'):
        document_text = extract_text_from_pdf(file)
        chunks = create_chunks(document_text)

        df = pd.DataFrame(chunks, columns=['Text'])
        df['Title'] = ['Chunk ' + str(i+1) for i in range(len(chunks))]
        df['Embeddings'] = df.apply(lambda row: embed_content(row['Title'], row['Text']), axis=1)

        request.environ['df'] = df  # Store dataframe in request context for subsequent use

        return jsonify({"message": "File processed successfully"}), 200
    else:
        return jsonify({"error": "Invalid file type. Please upload a PDF."}), 400

@app.route('/ask', methods=['POST'])
def ask():
    query = request.json.get('query')
    df = request.environ.get('df')

    if not query or df is None:
        return jsonify({"error": "Invalid query or PDF not uploaded yet"}), 400

    top_passages = find_top_chunks(query, df, top_n=3)
    prompt = make_prompt(query, top_passages)
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    answer = model.generate_content(prompt)

    return jsonify({"answer": answer.text}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
