import cv2
import json
from datetime import datetime
from itertools import combinations
from pathlib import Path

import torch

torch.backends.mkldnn.enabled = False
torch.set_num_threads(1)

from ultralytics import YOLO
from alert_client import send_incident

VIDEO_PATH = "videos/unsafe_interaction_test.mp4"
OUTPUT_PATH = "evidence/unsafe_interaction_detected.mp4"
CAMERA_NAME = "Camera 4"


MOTION_THRESHOLD = 0
CONFIRMATION_SECONDS = 0.3
CLOSE_DISTANCE_FACTOR = 4.0

Path("evidence").mkdir(exist_ok=True)

model = YOLO("yolo11n.pt")
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

previous_centers = {}
interaction_start_times = {}
alerted_pairs = set()
frame_number = 0

while True:
    success, frame = video.read()

    if not success:
        break

    frame_number += 1
    current_time = frame_number / fps
    annotated_frame = frame.copy()

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

    boxes = results[0].boxes
    people = {}

    if boxes.id is not None:
        track_ids = boxes.id.int().cpu().tolist()
        coordinates = boxes.xyxy.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()

        for track_id, (x1, y1, x2, y2), confidence in zip(
            track_ids, coordinates, confidences
        ):
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            person_width = x2 - x1

            previous = previous_centers.get(track_id, (center_x, center_y))
            movement = (
                (center_x - previous[0]) ** 2 +
                (center_y - previous[1]) ** 2
            ) ** 0.5

            people[track_id] = {
                "box": (x1, y1, x2, y2),
                "center": (center_x, center_y),
                "width": person_width,
                "movement": movement,
                "confidence": confidence
            }

            previous_centers[track_id] = (center_x, center_y)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated_frame,
                f"Person ID {track_id}",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    for first_id, second_id in combinations(people.keys(), 2):
        first = people[first_id]
        second = people[second_id]

        distance = (
            (first["center"][0] - second["center"][0]) ** 2 +
            (first["center"][1] - second["center"][1]) ** 2
        ) ** 0.5

        close_distance = (
            (first["width"] + second["width"]) / 2
        ) * CLOSE_DISTANCE_FACTOR

        are_close = distance < close_distance
        rapid_movement = (
            first["movement"] + second["movement"]
        ) > MOTION_THRESHOLD

        pair_id = tuple(sorted((first_id, second_id)))
        suspected_interaction = are_close 

        if suspected_interaction:
            if pair_id not in interaction_start_times:
                interaction_start_times[pair_id] = current_time

            duration = current_time - interaction_start_times[pair_id]

            if duration >= CONFIRMATION_SECONDS:
                x1 = min(first["box"][0], second["box"][0])
                y1 = min(first["box"][1], second["box"][1])
                x2 = max(first["box"][2], second["box"][2])
                y2 = max(first["box"][3], second["box"][3])

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    annotated_frame,
                    "SUSPECTED UNSAFE INTERACTION",
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2
                )

                if pair_id not in alerted_pairs:
                    alerted_pairs.add(pair_id)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    evidence_path = f"evidence/interaction_{timestamp}.jpg"
                    cv2.imwrite(evidence_path, annotated_frame)

                    incident = {
                        "camera": CAMERA_NAME,
                        "incident": "Suspected Unsafe Interaction",
                        "confidence": round(
                            float((first["confidence"] + second["confidence"]) / 2),
                            2
                        ),
                        "severity": "High",
                        "track_ids": list(pair_id),
                        "timestamp": datetime.now().isoformat(),
                        "evidence_path": evidence_path,
                        "status": "new"
                    }

                    print(json.dumps(incident, indent=2))
                    send_incident(incident)
        else:
            interaction_start_times.pop(pair_id, None)

    cv2.putText(
        annotated_frame,
        "Unsafe Interaction Detection: Active",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    writer.write(annotated_frame)
    cv2.imshow("SentinelEdu - Unsafe Interaction Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
writer.release()
cv2.destroyAllWindows()

print(f"Done. Saved video to: {OUTPUT_PATH}")