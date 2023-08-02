# Import required libraries
from ultralytics import YOLO
import multiprocessing

# Function to ensure correct operation when using multiprocessing
if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Create a YOLO object and load the pre-trained model from "yolov8n.pt" file
    model = YOLO("yolov8n.pt")

    # Set the device to "mps" (MPS: Multi-Process Service) for better GPU utilization
    model.to("mps")

    # Train the model using the specified data configuration, for 3 epochs
    # Use images of size 640x640 for training
    # Utilize the "mps" device for training, with 4 worker processes and a batch size of 4
    model.train(data="config.yaml", epochs=3, imgsz=640, device="mps", workers=4, batch=4)

    # Perform validation on the trained model and collect metrics
    metrics = model.val()

    # Perform object detection on a single image specified by the file path
    # Use image size 640x640 for detection, and "mps" device with 4 workers and a batch size of 4
    results = model('dataset/valid/images/frame_111.jpg', imgsz=640, device="mps", workers=4, batch=4)

    # Export the trained model to ONNX format and save it to the path returned by the export function
    path = model.export(format='onnx')
