import cv2

cap = cv2.VideoCapture(
    "dataset/traffic.mp4"
)

ret, frame = cap.read()

if not ret:
    print("Could not read video")
    cap.release()
    raise SystemExit

cv2.imwrite(
    "signal_test_frame.jpg",
    frame
)

print("Frame saved:")
print("signal_test_frame.jpg")

cap.release()