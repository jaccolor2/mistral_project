from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mistralai import Mistral
import numpy as np
import faiss
import os
import time
import pickle
from PyPDF2 import PdfReader
import secrets

# Initialize FastAPI app
app = FastAPI()

# Initialize Mistral client
def load_secrets(file_path):
    secrets = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split('=', 1)
            secrets[key] = value
    return secrets

# Path to the secrets.txt file
secrets_file_path = 'secrets.txt'

# Load the secrets
secrets = load_secrets(secrets_file_path)

# Access the API_KEY
api_key = secrets.get('API_KEY')
client = Mistral(api_key=api_key)

# Define the request model
class ChatRequest(BaseModel):
    theme: str
    user_input: str

# Function to extract text from PDFs
def extract_text_from_pdfs(directory):
    text = ""
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            reader = PdfReader(filepath)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    return text

# Function to get text embeddings
def get_text_embedding(inputs):
    embeddings_batch_response = client.embeddings.create(
        model="mistral-embed",
        inputs=inputs
    )
    time.sleep(1.8)
    return [data.embedding for data in embeddings_batch_response.data]

# Function to split text into batches
def split_into_batches(chunks, max_batch_size):
    batches = []
    current_batch = []
    current_size = 0

    for chunk in chunks:
        chunk_size = len(chunk)
        if current_size + chunk_size > max_batch_size:
            batches.append(current_batch)
            current_batch = []
            current_size = 0
        current_batch.append(chunk)
        current_size += chunk_size

    if current_batch:
        batches.append(current_batch)

    return batches

# Function to retrieve information based on a query
def retrieve_information(theme, question, chunks, index):
    question_embeddings = np.array([get_text_embedding([question])[0]])
    D, I = index.search(question_embeddings, k=2)  # distance, index
    retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
    return retrieved_chunk

# Function to run the Mistral model
def run_mistral(user_message, conversation_history, model="mistral-large-latest"):
    messages = [
        {
            "role": "user", "content": user_message
        }
    ]
    for message in conversation_history:
        messages.append({"role": message["role"], "content": message["content"]})
    chat_response = client.chat.complete(
        model=model,
        messages=messages
    )
    return chat_response.choices[0].message.content

# Load or generate text embeddings for each theme
def load_or_generate_embeddings(theme):
    embeddings_file = f'{theme}_embeddings.pkl'
    chunks_file = f'{theme}_chunks.pkl'
    if os.path.exists(f'./embeddings/{embeddings_file}') and os.path.exists(f'./chunks/{chunks_file}'):
        with open(f'embeddings/{embeddings_file}', 'rb') as f:
            text_embeddings = pickle.load(f)
        with open(f'chunks/{chunks_file}', 'rb') as f:
            chunks = pickle.load(f)
    else:
        text = extract_text_from_pdfs("./pdfs/")
        chunk_size = 2048
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        max_batch_size = 16080
        batches = split_into_batches(chunks, max_batch_size)

        start_time = time.time()
        text_embeddings = []
        for batch in batches:
            batch_embeddings = get_text_embedding(batch)
            text_embeddings.extend(batch_embeddings)
        text_embeddings = np.array(text_embeddings)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken to generate text_embeddings for {theme}: {elapsed_time} seconds")

        with open(f'embeddings/{embeddings_file}', 'wb') as f:
            pickle.dump(text_embeddings, f)
        with open(f'chunks/{chunks_file}', 'wb') as f:
            pickle.dump(chunks, f)

    d = text_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(text_embeddings)

    return chunks, index

# Define the chat endpoint
@app.post("/chat/")
async def chat(request: ChatRequest):
    theme = request.theme
    user_input = request.user_input

    # Load or generate embeddings for the specified theme
    chunks, index = load_or_generate_embeddings(theme)

    # Retrieve relevant information for the user input
    retrieved_chunk = retrieve_information(theme, user_input, chunks, index)
    context_prompt = f"""
    Context information is below.
    ---------------------
    {retrieved_chunk}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {user_input}
    Answer:
    """

    conversation_history = []
    conversation_history.append({"role": "user", "content": context_prompt})
    response = run_mistral(context_prompt, conversation_history)
    conversation_history.append({"role": "assistant", "content": response})

    return {"response": response}

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
