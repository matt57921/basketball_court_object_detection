import torch
import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf

# Load the YOLOv5 model (this is a simplified example, actual implementation might vary)
model = tf.keras.models.load_model('path_to_yolov5_model')

# Load and preprocess an image
image = tf.io.read_file('path_to_image.jpg')
image = tf.image.decode_jpeg(image, channels=3)
image = tf.image.resize(image, (416, 416))  # Resize to the input size of the model
image = image / 255.0  # Normalize the pixel values

# Run inference
predictions = model.predict(tf.expand_dims(image, 0))

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load the image or video
#image_path = "data/kobe_last_game.png"
# For video, you can use cv2.VideoCapture

# Perform object detection
#results = model(image_path)

# Draw bounding boxes around the detected players
#results.show()

#For video, loop through each frame and apply object detection
cap = cv2.VideoCapture("data/warriors_cavs_2018_game1.mp4")
print(cap.get(cv2.CAP_PROP_FPS))

if cap.isOpened() == False:
    print('Cannot open file or video stream')

while True:
    ret, frame = cap.read()
    if not ret:
        break;

    # Perform object detection
    results = model(frame)

    # Draw bounding boxes around the detected players
    results.render()

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()



#Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()




