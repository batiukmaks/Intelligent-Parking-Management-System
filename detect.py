import argparse
from ultralytics import YOLO
import utils
import torch

def perform_object_detection(source, weights, device, conf=0.3, iou=0.5, show=True):
    # Create a YOLO object and load the pre-trained model from the specified path
    model = YOLO(weights)

    # Set the device for inference
    model.to(device)

    # Perform object detection on a video specified by the "source" parameter
    # Display the video with bounding boxes around detected objects
    model(source=source, conf=conf, iou=iou, show=show)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform object detection on a video using YOLO.")

    # Add arguments with default values
    parser.add_argument("--source", type=str, default='default-dataset/default-video.MOV', help="Path to the video for object detection.")
    parser.add_argument("--weights", type=str, default='example-results/train/weights/best.pt', help="Path to the YOLO pre-trained model weights.")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold for object detection.")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="IOU (Intersection over Union) threshold for object detection.")
    parser.add_argument("--show", default=True, help="Show the video with bounding boxes around detected objects.")
    parser.add_argument("--use_coral", action="store_true", help="Use Coral USB Accelerator for inference if available.")
    parser.add_argument("--device", type=str, default=utils.get_default_device(),
                        help="Device for inference (e.g., 'cuda', 'cpu', 'mps', 'edge').")

    args = parser.parse_args()

    # Determine the device for inference based on Coral availability and user choice
    if args.use_coral and utils.is_coral_available():
        args.device = "edge"

    # Call the perform_object_detection function with the parsed arguments
    perform_object_detection(
        source=args.source,
        weights=args.weights,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        show=args.show
    )
