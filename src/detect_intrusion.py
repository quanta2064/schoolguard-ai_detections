import cv2
import json
from datetime import datetime
from pathlib import Path

import torch

torch.backends.mkldnn.enabled = False
torch.set_num_threads(1)

from ultralytics import YOLO

VIDEO_PATH = "videos/test.mp4"
OUTPUT_PATH = "evidence/intrusion_detected.mp4"
CAMERA_NAME = "Camera 1"

Path("evidence").mkdir(exist_ok=True)

model = YOLO("yolo11n.pt")
video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    raise RuntimeError(f"Could not open {VIDEO_PATH}")

fps = video.get(cv2.CAP_PROP_FPS) or 25
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Adjust these values after seeing your video.
ZONE_X1 = int(width * 0.35)
ZONE_Y1 = int(height * 0.25)
ZONE_X2 = int(width * 0.65)
ZONE_Y2 = int(height * 0.85)

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

alerted_track_ids = set()

while True:
    success, frame = video.read()

    if not success:
        break

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

    # Draw the restricted-area zone.
    cv2.rectangle(
        annotated_frame,
        (ZONE_X1, ZONE_Y1),
        (ZONE_X2, ZONE_Y2),
        (0, 0, 255),
        2
    )
    cv2.putText(
        annotated_frame,
        "RESTRICTED AREA",
        (ZONE_X1, ZONE_Y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    boxes = results[0].boxes
    intruder_count = 0

    if boxes.id is not None:
        track_ids = boxes.id.int().cpu().tolist()
        coordinates = boxes.xyxy.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()

        for track_id, (x1, y1, x2, y2), confidence in zip(
            track_ids, coordinates, confidences
        ):
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            is_inside_zone = (
                ZONE_X1 <= center_x <= ZONE_X2
                and ZONE_Y1 <= center_y <= ZONE_Y2
            )

            if is_inside_zone:
                intruder_count += 1
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    annotated_frame,
                    f"ID {track_id} - INTRUDER",
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2)
                cv2.putText(
                    annotated_frame,
                    "INTRUSION ALERT",
                    (x1, max(30, y1 - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                if track_id not in alerted_track_ids:
                    alerted_track_ids.add(track_id)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    evidence_path = f"evidence/intrusion_{timestamp}_person_{track_id}.jpg"
                    cv2.imwrite(evidence_path, annotated_frame)

                    incident = {
                        "camera": CAMERA_NAME,
                        "incident": "Restricted Area Intrusion",
                        "confidence": round(float(confidence), 2),
                        "severity": "Medium",
                        "track_id": track_id,
                        "timestamp": datetime.now().isoformat(),
                        "evidence_path": evidence_path,
                        "status": "new"
                    }

                    print(json.dumps(incident, indent=2))

    if intruder_count > 0:
        cv2.rectangle(annotated_frame, (10, 10), (650, 65), (0, 0, 255), -1)
        cv2.putText(
            annotated_frame,
            f"ALERT: {intruder_count} PERSON(S) IN RESTRICTED AREA",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    writer.write(annotated_frame)
    cv2.imshow("SentinelEdu - Restricted Area Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
writer.release()
cv2.destroyAllWindows()

print(f"Done. Saved video to: {OUTPUT_PATH}")
