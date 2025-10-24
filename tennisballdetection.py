import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow("HSV Trackbars")

cv2.createTrackbar("Lower H", "HSV Trackbars", 26, 179, nothing)
cv2.createTrackbar("Lower S", "HSV Trackbars", 40, 255, nothing)
cv2.createTrackbar("Lower V", "HSV Trackbars", 80, 255, nothing)

cv2.createTrackbar("Upper H", "HSV Trackbars", 78, 179, nothing)
cv2.createTrackbar("Upper S", "HSV Trackbars", 255, 255, nothing)
cv2.createTrackbar("Upper V", "HSV Trackbars", 255, 255, nothing)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv2.resize(frame, (640, 480))
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    lh = cv2.getTrackbarPos("Lower H", "HSV Trackbars")
    ls = cv2.getTrackbarPos("Lower S", "HSV Trackbars")
    lv = cv2.getTrackbarPos("Lower V", "HSV Trackbars")

    uh = cv2.getTrackbarPos("Upper H", "HSV Trackbars")
    us = cv2.getTrackbarPos("Upper S", "HSV Trackbars")
    uv = cv2.getTrackbarPos("Upper V", "HSV Trackbars")

    lower_yellow = np.array([lh, ls, lv])
    upper_yellow = np.array([uh, us, uv])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 500:
            continue

        ((x, y), radius) = cv2.minEnclosingCircle(cnt)

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    cv2.imshow("Yellow Tennis Ball Detection", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
