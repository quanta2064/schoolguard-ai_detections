import os
import requests

BACKEND_URL = os.getenv(
    "BACKEND_INCIDENT_URL",
    "http://127.0.0.1:8000/incidents"
)

def send_incident(incident: dict) -> None:
    try:
        response = requests.post(
            BACKEND_URL,
            json=incident,
            timeout=5
        )
        response.raise_for_status()
        print("Incident sent to FastAPI successfully.")
    except requests.RequestException as error:
        print(f"Could not send incident to FastAPI: {error}")