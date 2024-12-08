# Chatbot Application

This project consists of a FastAPI backend and a Django frontend that interact with each other to provide a chatbot service. The FastAPI backend handles the retrieval of information from PDFs and generates responses using a Mistral AI model. The Django frontend provides a web interface for users to interact with the chatbot.

## Prerequisites

- Python 3.7+
- pip
- Virtualenv (optional but recommended)

## Setup

### 1. Clone the Repository

```sh
git clone <repository_url>
cd chatbot_project
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

### 4. Set Up FastAPI Backend

#### 4.1. Run the FastAPI Server

Ensure your FastAPI server is configured to run on port `8000`:

```sh
py -m main
```

Verify that the FastAPI server is running by accessing `http://127.0.0.1:8000/docs` in your web browser. You should see the Swagger UI.

### 5. Set Up Django Frontend

#### 5.1. Run the Django Server

Ensure that the Django server is running on a different port, for example, `8001`:

```sh
py -m manage runserver 127.0.0.1:8001
```

### 6. Access the Website

Open your web browser and navigate to `http://127.0.0.1:8001`. You should see the web interface where you can input a theme and a query, and the response from the FastAPI backend will be displayed on the web page.

### Additional Notes

- Ensure that both the FastAPI and Django servers are running simultaneously.
- Verify that the FastAPI server is accessible at `http://127.0.0.1:8000` and the Django server is accessible at `http://127.0.0.1:8001`.
- Check the logs of both servers for any error messages or warnings.

By following these steps, you should be able to set up and run the entire application, including both the FastAPI backend and the Django frontend.