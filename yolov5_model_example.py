# Import necessary libraries
import torch               # PyTorch library for deep learning
import cv2                 # OpenCV library for computer vision
import numpy as np         # NumPy library for numerical operations
from pathlib import Path  # Pathlib library for working with file paths
import tensorflow as tf    # TensorFlow library for machine learning

# Load YOLOv5 model from Ultralytics hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open a video file for reading
cap = cv2.VideoCapture("data/warriors_cavs_2018_game1.mp4")

# Print the frames per second (FPS) of the video
print(cap.get(cv2.CAP_PROP_FPS))

# Check if the video file was opened successfully
if cap.isOpened() == False:
    print('Cannot open file or video stream')

# Loop through each frame of the video
while True:
    # Read the next frame from the video
    ret, frame = cap.read()
    
    # If there are no more frames, exit the loop
    if not ret:
        break;

    # Perform object detection on the current frame using the YOLOv5 model
    results = model(frame)
    print(results.pandas().xyxy[0])

    # Render (draw) bounding boxes around the detected objects on the frame
    results.render()

    # Display the frame with bounding boxes in a window named 'Frame'
    cv2.imshow('Frame', frame)

    # Wait for a key press and check if 'q' was pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()