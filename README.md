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
### Single Image Detection
### **img3.jpg**
![image](https://github.com/Devansh-Gupta-Official/object-detection-yolo/assets/100591612/a74c6f08-224c-44d3-bcf3-d1d24d4f4f6e)

### **img.jpg**
![image](https://github.com/Devansh-Gupta-Official/object-detection-yolo/assets/100591612/2b14c456-db5b-4506-9a0f-8dbe55f719c3)

### **img2.jpg**
![image](https://github.com/Devansh-Gupta-Official/object-detection-yolo/assets/100591612/f775a483-f4e3-43e5-a0c7-2b3b89d9057d)

### **vid.mp4**
![image](https://github.com/Devansh-Gupta-Official/object-detection-yolo/assets/100591612/ebe8ce4c-3f66-4529-97ea-d1625d2d9d96)
![image](https://github.com/Devansh-Gupta-Official/object-detection-yolo/assets/100591612/b39fe20f-059b-453f-a421-e6138cf5a9ad)

### Real Time Detections
![image](https://github.com/Devansh-Gupta-Official/object-detection-yolo/assets/100591612/2b954366-58c0-4010-883f-2669c4bb1559)
![image](https://github.com/Devansh-Gupta-Official/object-detection-yolo/assets/100591612/81150bdc-aeef-4519-a239-5fea36f1bdee)

