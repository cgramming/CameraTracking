import cv2
import numpy as np

# Use webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Yellow color range (adjust if needed)
    lower_yellow = np.array([20, 120, 120])
    upper_yellow = np.array([200, 255, 255])

    # Mask for yellow object
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Find contours of yellow regions
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:  # ignore small noise
            x, y, w, h = cv2.boundingRect(cnt)

            # Draw rectangle around the yellow ball
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Yellow Mask", mask)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()