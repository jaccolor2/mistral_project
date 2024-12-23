# Chatbot Application

This project consists of a FastAPI backend and a Django frontend that interact with each other to provide a chatbot service. The FastAPI backend handles the retrieval of information from PDFs and generates responses using a Mistral AI model. The Django frontend provides a web interface for users to interact with the chatbot.

## IMPORTANT

For now, the only 'theme' accepted is "pokemon". It has to be exactly this string.

Also, in directories "chunks" and "embeddings" are provided the chunks of text and embeddings created during the 1st launch. If you want to create them again, delete the files. Take into account that the embedding process can take around 3 minutes for this specific set of texts. It is due to the size of the text, but also the api requests limitations.

## Prerequisites

- Python 3.7+
- pip
- Virtualenv (optional but recommended)

## Setup

### 1. Clone the Repository

```sh
git clone https://github.com/jaccolor2/mistral_project.git
cd mistral_project
```

### 2. Create and Activate a Virtual Environment (Optional)

```sh
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies

Navigate to the project directory and install the required dependencies:

```sh
pip install -r requirements.txt
```

### 4. Set API KEY

#### 4.1 Create secrets.txt

Create a `secrets.txt` file in your project directory.

#### 4.2 Set the API_KEY

Add your Mistral API key:

```plaintext
API_KEY=your_mistral_api_key_here
``` 

### 5. Run the FastAPI Backend

Ensure your FastAPI server is configured to run on port `8000`:

```sh
py -m main
```

Verify that the FastAPI server is running by accessing `http://127.0.0.1:8000/docs` in your web browser. You should see the Swagger UI.

### 6. Run the Django Frontend

Go in the 'chatbot_projet' folder.

```sh
cd chatbot_project
```

Ensure that the Django server is running on a different port, for example, `8001`:

```sh
python manage.py runserver 127.0.0.1:8001
```
or
```sh
py -m manage runserver 127.0.0.1:8001
```

### 7. Access the Website

Open your web browser and navigate to `http://127.0.0.1:8001`. You should see the web interface where you can input a theme and a query, and the response from the FastAPI backend will be displayed on the web page.

### 8. Example prompt

![Example prompt](pictures/example_prompt.png)

### Additional Notes

- Ensure that both the FastAPI and Django servers are running simultaneously.
- Verify that the FastAPI server is accessible at `http://127.0.0.1:8000` and the Django server is accessible at `http://127.0.0.1:8001`.
- Check the logs of both servers for any error messages or warnings.

By following these steps, you should be able to set up and run the entire application, including both the FastAPI backend and the Django frontend.
