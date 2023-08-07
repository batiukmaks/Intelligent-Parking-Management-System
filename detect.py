import argparse
from ultralytics import YOLO


def perform_object_detection(source, weights, conf=0.3, iou=0.5, show=True):
    # Create a YOLO object and load the pre-trained model from the specified path
    model = YOLO(weights)

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
    parser.add_argument("--show", default=True,
                        help="Show the video with bounding boxes around detected objects.")

    args = parser.parse_args()

    # Call the perform_object_detection function with the parsed arguments
    perform_object_detection(
        source=args.source,
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        show=args.show
    )
