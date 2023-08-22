# Import required libraries
import argparse
import tensorflow as tf  # Import TensorFlow
from ultralytics import YOLO
import multiprocessing
import utils
import torch


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train_model(epochs, device, workers, batch, validation_image, output_directory,
                model_path='yolov8n.pt', data='default-config.yaml'):
    # Create a YOLO object and load the pre-trained model
    model = YOLO(model_path)

    # Set the device for better performance
    model.to(device)

    # Train the model using the specified data configuration, for the specified number of epochs
    # Use images of size 640x640 for training
    # Utilize the device specified for training, with the specified worker processes and batch size
    model.train(data=data, epochs=epochs, imgsz=640, device=device, workers=workers, batch=batch,
                project=output_directory, name="train", exist_ok=True)

    # Perform validation on the trained model and collect metrics
    model.val()

    # Perform object detection on a single image specified by the file path
    # Use image size 640x640 for detection
    if validation_image:
        model(validation_image, imgsz=640, device=device, workers=workers, batch=batch,
              project=output_directory, name="valid")

    # Export the trained model to ONNX format and save it
    model.export(format='onnx')


# Function to ensure correct operation when using multiprocessing
if __name__ == '__main__':
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Train a YOLO object detection model.")

    # Add arguments with default values
    parser.add_argument("--model_path", type=str, default="yolov8n.pt",
                        help="Path to the pre-trained model.")
    parser.add_argument("--data", type=str, default="default-config.yaml",
                        help="Data configuration for training.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for training.")
    parser.add_argument("--device", type=str, default=utils.get_default_device(),
                        help="Device for training (e.g., 'cuda', 'cpu', 'mps', 'edge').")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of worker processes for training.")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size for training.")
    parser.add_argument("--validation_image", type=str,
                        help="Path to the validation image for object detection.")
    parser.add_argument("--output_directory", type=str, default="output",
                        help="Directory to save the trained model and results.")
    parser.add_argument("--use_coral", action="store_true",
                        help="Use Coral USB Accelerator for inference if available.")

    args = parser.parse_args()

    substituted_data = utils.substitute_env_variables(args.data)

    if args.use_coral and utils.is_coral_available():
        args.device = "edge"
        # Load the EdgeTPU delegate
        edgetpu_delegate = tf.lite.experimental.load_delegate('libedgetpu.so.1')
        # Create the interpreter with EdgeTPU delegate
        interpreter = tf.lite.Interpreter(model_path=args.model_path)
        interpreter.add_delegate(edgetpu_delegate)

    # Call the train_model function with the parsed arguments
    train_model(model_path=args.model_path, data=substituted_data, epochs=args.epochs, device=args.device,
                workers=args.workers, batch=args.batch, validation_image=args.validation_image,
                output_directory=args.output_directory)

    if args.device == "edge":
        interpreter.remove_delegate()
