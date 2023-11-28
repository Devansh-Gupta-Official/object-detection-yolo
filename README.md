# YOLOv5 Object Detection
This repository contains code for object detection using the YOLOv5 model from the Ultralytics YOLOv5 GitHub repository.

## Installation
To get started, clone the YOLOv5 repository and install the required dependencies:
```
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

## Usage
### Single Image Detection
Load Model
```
import torch

# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")
```

Perform Inference on Single Image
```
# Image path
img = "img.jpeg"

# Inference
results = model(img)

# Print results
results.print()
```
**Replace "img.jpeg" with the desired image file (such as img2.jpg, img3.png, etc.) to perform object detection on different images.**

### Real-Time Video Detection
```
import cv2 as cv
import numpy as np

# Open video capture
capture = cv.VideoCapture(0)
capture.set(3, 640)
capture.set(4, 480)
capture.set(10, 70)

while capture.isOpened():
    ret, frame = capture.read()

    # Inference
    results = model(frame)
    
    cv.imshow('Video', np.squeeze(results.render()))

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
```
**For video detection, replace capture = cv.VideoCapture(0) with the path to your video file (e.g., capture = cv.VideoCapture('vid.mp4')) to perform real-time object detection on the provided video.**


## Results
