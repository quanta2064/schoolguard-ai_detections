from alert_client import send_incident
import cv2
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

VIDEO_PATH = "videos/fire_test.mp4"
CAMERA_NAME = "Camera 3"
MODEL_ID = "fire-and-smoke-segmentation/11"
FRAME_INTERVAL = 10          # Check one frame every 10 frames
CONFIDENCE_THRESHOLD = 0.50
REQUIRED_DETECTIONS = 2      # Avoid an alert from one frame

Path("evidence").mkdir(exist_ok=True)
Path("temp_frames").mkdir(exist_ok=True)

load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise RuntimeError("ROBOFLOW_API_KEY is missing from .env")

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)

video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    raise RuntimeError(f"Could not open {VIDEO_PATH}")

frame_number = 0
confirmed_detections = 0
alert_sent = False

while True:
    success, frame = video.read()

    if not success:
        break

    frame_number += 1
    display_frame = frame.copy()

    if frame_number % FRAME_INTERVAL == 0:
        temp_path = "temp_frames/current_frame.jpg"
        cv2.imwrite(temp_path, frame)

        try:
            result = client.infer(temp_path, model_id=MODEL_ID)
            predictions = result.get("predictions", [])

            fire_or_smoke_found = False

            for prediction in predictions:
                label = prediction.get("class", "").lower()
                confidence = float(prediction.get("confidence", 0))

                if label in ["fire", "smoke"] and confidence >= CONFIDENCE_THRESHOLD:
                    fire_or_smoke_found = True

                    x = int(prediction["x"])
                    y = int(prediction["y"])
                    w = int(prediction["width"])
                    h = int(prediction["height"])

                    x1, y1 = x - w // 2, y - h // 2
                    x2, y2 = x + w // 2, y + h // 2

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(
                        display_frame,
                        f"{label.upper()} {confidence:.0%}",
                        (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

            if fire_or_smoke_found:
                confirmed_detections += 1
            else:
                confirmed_detections = 0

            if confirmed_detections >= REQUIRED_DETECTIONS and not alert_sent:
                alert_sent = True

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                evidence_path = f"evidence/fire_{timestamp}.jpg"
                cv2.imwrite(evidence_path, display_frame)

                incident = {
                    "camera": CAMERA_NAME,
                    "incident": "Fire / Smoke Detected",
                    "confidence": 0.90,
                    "severity": "Critical",
                    "timestamp": datetime.now().isoformat(),
                    "evidence_path": evidence_path,
                    "status": "new"
                }

                print(json.dumps(incident, indent=2))
                send_incident(incident)

        except Exception as error:
            print(f"Roboflow inference error: {error}")
    if alert_sent:
        cv2.rectangle(display_frame,(10, 10),(display_frame.shape[1] - 10, 55),(0, 0, 255),-1)
        cv2.putText(
            display_frame,
            "CRITICAL: SMOKE",
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    cv2.imshow("SentinelEdu - Fire & Smoke Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()