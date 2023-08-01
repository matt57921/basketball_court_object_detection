import torch
import cv2
import numpy as np
from pathlib import Path



# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load the image or video
#image_path = "data/kobe_final_game.png"
# For video, you can use cv2.VideoCapture

# Perform object detection
#results = model(image_path)

# Draw bounding boxes around the detected players
#results.show()

#For video, loop through each frame and apply object detection
cap = cv2.VideoCapture("data/warriors_cavs_2018_game1.mp4")


# Check if the video is opened successfully
if not cap.isOpened():
    print("Error opening video file.")
    exit()

# Loop through each frame in the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the frame (use your detection code here)
    detected_players = detect_players(frame)

    # Draw bounding boxes around the detected players (use your drawing code here)
    for player in detected_players:
        x, y, w, h = player
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


    # Display the frame with bounding boxes
    cv2.imshow("Basketball Player Detection", frame)

    # Check for the "Escape" key (ASCII value 27) to exit the video playback
    if cv2.waitKey(1) & 0xFF == 27:
        break

#Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()




