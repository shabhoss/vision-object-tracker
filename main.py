"""
File: main.py
Author: Shabir Hossain
Date: 2025-11-16
Description: This module contains functions for detecting an object 
with YOLOv8 and uses a Kalman Filter to track its motion smoothly, 
even when the detection flickers.
License: MIT License
"""

import cv2
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

model = YOLO("yolov8n.pt")  # "n" = nano (fastest, good for laptop)

# INIT KALMAN FILTER
kf = KalmanFilter(dim_x=4, dim_z=2)

# State transition matrix (predict next state)
kf.F = np.array([[1, 0, 1, 0],
                 [0, 1, 0, 1],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])

# Measurement matrix (we can measure x and y)
kf.H = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0]])

kf.P *= 10   # initial uncertainty
kf.R *= 5    # measurement noise
kf.Q *= 0.1  # process noise

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]  # detect objects in the frame

    # If YOLO detects anything
    if len(results.boxes) > 0:
        box = results.boxes[0]  # just pick the first object
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        measurement = np.array([[cx], [cy]])

        kf.update(measurement)  # update Kalman filter with detection
        
    kf.predict()
    x, y = int(kf.x[0]), int(kf.x[1])

    # YOLO detection box (blue)
    if len(results.boxes) > 0:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Kalman filtered center (green)
    cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

    cv2.imshow("Object Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


