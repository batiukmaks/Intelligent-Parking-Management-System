# Intelligent Parking Management System

Welcome to the Intelligent Parking Management System repository! This repository contains code for training and performing object detection using YOLO (You Only Look Once) model with the Ultralytics library. The system is designed to manage parking areas by detecting and tracking vehicles and barriers.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Object Detection](#object-detection)
- [Configuration](#configuration)
- [Data annotating (Bonus)](#data-annotating-bonus)

## Getting Started

### Prerequisites

- Python >= 3.6
- CUDA-compatible GPU (for accelerated training)

### Installation

1. Clone the repository:
```
git clone https://github.com/batiukmaks/Intelligent-Parking-Management-System.git
cd Intelligent-Parking-Management-System
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

Before you begin using the Intelligent Parking Management System, make sure to configure the `default-config.yaml` file to match your dataset and requirements. This configuration file contains essential paths and class information necessary for training and object detection.

### Training
Before training, you have to configure the environment and the [config file](#configuration) if you want to use your own data. 

In console, use the following command:
```
export ROOT_DIRECTORY=$(pwd) 
```

To train the YOLO model, use the following command, adjusting parameters as needed:
```
python train.py --epochs 10 --device cuda --batch 8 --validation_image path/to/validation/image.jpg
```

- `--model_path`: Path to the pre-trained model.
- `--data`: Data configuration for training.
- `--epochs`: Number of epochs for training.
- `--device`: Device for training (e.g., 'cuda', 'cpu', 'mps').
- `--batch`: Batch size for training.
- `--validation_image`: Path to a validation image for object detection during training.
- `--use_coral`: Use Coral USB Accelerator for inference if available.

### Object Detection

For object detection on videos, use the following command, adjusting parameters as needed:
```
python detect.py --source path/to/video.mp4 --weights path/to/weights.pt --conf 0.5 --iou 0.5
```

- `--source`: Path to the video for object detection.
- `--weights`: Path to the YOLO pre-trained model weights.
- `--conf`: Confidence threshold for object detection.
- `--iou`: IOU (Intersection over Union) threshold for object detection.
- `--show`: Show the video with bounding boxes around detected objects (True or False).
- `--use_coral`: Use Coral USB Accelerator for inference if available.
- `--device`: Device for inference (e.g., 'cuda', 'cpu', 'mps', 'edge').

Remember that the accuracy and performance of the system heavily depend on the quality of your dataset and the configuration settings in `default-config.yaml`. Ensure the configuration accurately reflects your dataset structure and class labels for successful training and accurate object detection.


## Configuration

The `default-config.yaml` file is a YAML configuration file that plays a crucial role in setting up and customizing the behavior of your Intelligent Parking Management System. This file contains various paths, settings, and class information required for training and object detection.

Here's a breakdown of the key elements within the configuration file:

_!!! Note, that all the paths have to be absolute._
- `path`: This is the base path to the directory where your dataset is located. It is used as a reference point for other dataset-related paths.

- `train`, `test`, `val`: These are subdirectories under the `path` that respectively contain training, testing, and validation data. These paths help the system locate and organize the dataset for different phases of the workflow.

- `nc`: This specifies the number of classes in your dataset. In your case, it's set to 5, indicating that there are five distinct classes (object types) that the system is designed to detect.

- `names`: This is a list of class names corresponding to the different object classes in your dataset. Each class name is a human-readable label associated with an object type. In your case, the class names are "barrier up," "car parked," "car in," "barrier down," and "car out."

These configuration settings are crucial for correctly interpreting and processing your dataset during training and object detection. By adjusting these values, you can adapt the system to different datasets and scenarios. Make sure to keep this configuration file up to date as you work with various datasets and classes.

Feel free to customize this configuration to match your specific dataset structure and class labels.

If you have more specific questions or need further assistance, please let me know!

## Data annotating (Bonus)
### Splitting Video into Frames

As a bonus, you can use the `split_video.py` script located in the "data-annotating" folder to split a video into individual frames for further annotation or analysis. This can be useful for preparing your dataset.

To split a video, run the following command:
```
python data-annotating/split_video.py --video_path path/to/video.mp4 --output_directory path/to/output --frame_interval 3
```

- `--video_path`: Path to the video file to be split.
- `--output_directory`: Path to the output directory for saving frames.
- `--frame_interval`: Interval between frames to be extracted (default is 3).

