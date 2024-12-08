from django.shortcuts import render
from django.http import JsonResponse
import requests
import logging
from urllib.parse import quote


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def chatbot_view(request):
    if request.method == 'POST':
        theme = quote(str(request.POST.get('theme')))
        user_input = quote(str(request.POST.get('userInput')))

        try:
            url = f'http://127.0.0.1:8000/chat/'
            headers = {
                'Content-Type': 'application/json'
            }
            json = {
                'theme':theme,
                'user_input':user_input
            }
            response = requests.post(url, json=json, headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()
            return render(request, 'index.html', {'response': data.get('response')})
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return render(request, 'index.html', {'response': 'Error: Failed to connect to the API.'})
        except ValueError as e:
            logger.error(f"JSON decode error: {e}")
            return render(request, 'index.html', {'response': 'Error: Invalid response from the API.'})

    return render(request, 'index.html')
