import cv2

video = cv2.VideoCapture("videos/test.mp4")

if not video.isOpened():
    raise RuntimeError("Could not open videos/test.mp4")

while True:
    success, frame = video.read()

    if not success:
        break

    cv2.imshow("SentinelEdu – Video Test", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()