from mistralai import Mistral
import requests
import numpy as np
import faiss
import os
import time
import subprocess
import webbrowser
from PyPDF2 import PdfReader
import pickle

api_key = "3TdKEjpWomNBvZUC5CH6M8Jr8qSHIncJ"
client = Mistral(api_key=api_key)

def extract_text_from_pdfs(directory):
    text = ""
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            reader = PdfReader(filepath)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    return text

def get_text_embedding(inputs):
    embeddings_batch_response = client.embeddings.create(
        model="mistral-embed",
        inputs=inputs
    )
    time.sleep(1.8)
    return [data.embedding for data in embeddings_batch_response.data]

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

def retrieve_information(question, chunks, index):
    question_embeddings = np.array([get_text_embedding([question])[0]])
    D, I = index.search(question_embeddings, k=2)  # distance, index
    retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
    return retrieved_chunk

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

# Extract text from all PDFs in the current directory
text = extract_text_from_pdfs(".")

chunk_size = 2048
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

max_batch_size = 16080
batches = split_into_batches(chunks, max_batch_size)

# Check if embeddings are already saved
embeddings_file = 'text_embeddings.pkl'
if os.path.exists(embeddings_file):
    with open(embeddings_file, 'rb') as f:
        text_embeddings = pickle.load(f)
else:
    # Measure the time taken to generate text_embeddings
    start_time = time.time()

    text_embeddings = []
    for batch in batches:
        batch_embeddings = get_text_embedding(batch)
        text_embeddings.extend(batch_embeddings)

    text_embeddings = np.array(text_embeddings)

    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Time taken to generate text_embeddings: {elapsed_time} seconds")

    # Save the embeddings to a file
    with open(embeddings_file, 'wb') as f:
        pickle.dump(text_embeddings, f)

d = text_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(text_embeddings)

# Chatbot system with context
conversation_history = []
print("Chatbot is ready. Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    # Retrieve relevant information for the user input
    retrieved_chunk = retrieve_information(user_input, chunks, index)
    context_prompt = f"""
    Context information is below.
    ---------------------
    {retrieved_chunk}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {user_input}
    Answer:
    """

    conversation_history.append({"role": "user", "content": context_prompt})
    response = run_mistral(context_prompt, conversation_history)
    conversation_history.append({"role": "assistant", "content": response})
    print(f"Bot: {response}")
