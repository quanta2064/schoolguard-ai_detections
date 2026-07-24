import cv2
import torch

torch.backends.mkldnn.enabled = False
torch.set_num_threads(1)
from ultralytics import YOLO

VIDEO_PATH = "videos/test.mp4"
OUTPUT_PATH = "evidence/people_detected.mp4"

# Downloads the small pre-trained YOLO model automatically on first run.
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

while True:
    success, frame = video.read()

    if not success:
        break

    # Class 0 = person in the standard COCO model.
    results = model(frame, classes=[0], conf=0.45, imgsz=640, device="cpu", verbose=False)

    annotated_frame = results[0].plot()
    writer.write(annotated_frame)

    cv2.imshow("SentinelEdu - YOLO Person Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
writer.release()
cv2.destroyAllWindows()

print(f"Done. Saved detection video to: {OUTPUT_PATH}")