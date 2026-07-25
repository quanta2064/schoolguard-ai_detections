from alert_client import send_incident
import cv2
import json
from datetime import datetime
from pathlib import Path

import mediapipe as mp
import torch

torch.backends.mkldnn.enabled = False
torch.set_num_threads(1)

from ultralytics import YOLO

VIDEO_PATH = "videos/fall_test.mp4"
OUTPUT_PATH = "evidence/fall_detected.mp4"
CAMERA_NAME = "Camera 2"
FALL_CONFIRMATION_SECONDS = 0.3

Path("evidence").mkdir(exist_ok=True)

model = YOLO("yolo11n.pt")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    raise RuntimeError(f"Could not open {VIDEO_PATH}")

fps = video.get(cv2.CAP_PROP_FPS) or 25
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

down_start_times = {}
alerted_track_ids = set()
frame_number = 0

while True:
    success, frame = video.read()

    if not success:
        break

    frame_number += 1
    current_time = frame_number / fps

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=0.45,
        imgsz=640,
        device="cpu",
        verbose=False
    )

    annotated_frame = frame.copy()
    boxes = results[0].boxes

    if boxes.id is not None:
        track_ids = boxes.id.int().cpu().tolist()
        coordinates = boxes.xyxy.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()

        for track_id, (x1, y1, x2, y2), confidence in zip(
            track_ids, coordinates, confidences
        ):
            person_width = x2 - x1
            person_height = y2 - y1

            # Keep coordinates inside the video frame.
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            person_crop = frame[y1:y2, x1:x2]

            # MediaPipe confirms a human pose is visible.
            has_pose = False
            if person_crop.size > 0:
                rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                pose_result = pose.process(rgb_crop)
                has_pose = pose_result.pose_landmarks is not None

            # A wide bounding box suggests the person is lying horizontally.
            aspect_ratio= person_width/max(person_height,1)
            is_horizontal = aspect_ratio > 0.55
            possible_fall = is_horizontal

            if possible_fall:
                if track_id not in down_start_times:
                    down_start_times[track_id] = current_time

                down_duration = current_time - down_start_times[track_id]

                if down_duration >= FALL_CONFIRMATION_SECONDS:
                    label = f"Person ID {track_id} | ratio: {aspect_ratio:.2f}"
                    color = (0, 0, 255)

                    if track_id not in alerted_track_ids:
                        alerted_track_ids.add(track_id)

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        evidence_path = (
                            f"evidence/fall_{timestamp}_person_{track_id}.jpg"
                        )
                        cv2.imwrite(evidence_path, annotated_frame)

                        incident = {
                            "camera": CAMERA_NAME,
                            "incident": "Possible Fall / Medical Emergency",
                            "confidence": round(float(confidence), 2),
                            "severity": "High",
                            "track_id": track_id,
                            "timestamp": datetime.now().isoformat(),
                            "evidence_path": evidence_path,
                            "status": "new"
                        }

                        print(json.dumps(incident, indent=2))
                        send_incident(incident)
                else:
                    label = f"Possible fall: {down_duration:.1f}s"
                    color = (0, 165, 255)
            else:
                down_start_times.pop(track_id, None)
                label = f"Person ID {track_id}"
                color = (0, 255, 0)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated_frame,
                label,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    cv2.putText(
        annotated_frame,
        "Fall Detection: Active",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    writer.write(annotated_frame)
    cv2.imshow("SentinelEdu - Fall Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
writer.release()
pose.close()
cv2.destroyAllWindows()

print(f"Done. Saved video to: {OUTPUT_PATH}")
