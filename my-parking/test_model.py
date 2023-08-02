# Import the YOLO class from the ultralytics library
from ultralytics import YOLO

# Create a YOLO object and load the pre-trained model from the specified path
model = YOLO("../runs/detect/train6/weights/best.pt")

# Perform object detection on a video specified by the "source" parameter
# Display the video with bounding boxes around detected objects
model(source="../videos/car-out/IMG_0640.MOV", conf=0.3, iou=0.5, show=True)
