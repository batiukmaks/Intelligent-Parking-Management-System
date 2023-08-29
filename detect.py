import argparse
from ultralytics import YOLO
import utils
import torch
import cv2


def perform_object_detection(source, weights):
    model = YOLO(weights)
    cap = cv2.VideoCapture(source)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("ok")
                break

        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform object detection on a video using YOLO.")

    # Add arguments with default values
    parser.add_argument("--source", type=str, default='default-dataset/default-video.MOV', help="Path to the video for object detection.")
    parser.add_argument("--weights", type=str, default='example-results/train/weights/best.pt', help="Path to the YOLO pre-trained model weights.")

    args = parser.parse_args()

    # Call the perform_object_detection function with the parsed arguments
    perform_object_detection(
        source=args.source,
        weights=args.weights,
    )
