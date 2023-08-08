# Import required libraries
import cv2
import os
import argparse

def split_video_to_frames(video_path, output_dir, frame_interval=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_capture = cv2.VideoCapture(video_path)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in the video: {frame_count}")

    frame_number = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_number % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{frame_number}.jpg")
            cv2.imwrite(frame_filename, frame)

        frame_number += 1

    video_capture.release()
    print(f"Extracted {frame_number} frames to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a video into frames.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video file to be split.")
    parser.add_argument("--output_directory", type=str, required=True, help="Path to the output directory for saving frames.")
    parser.add_argument("--frame_interval", type=int, default=3, help="Interval between frames to be extracted.")

    args = parser.parse_args()

    video_path = args.video_path
    output_directory = args.output_directory
    frame_interval = args.frame_interval

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_subdir = os.path.join(output_directory, os.path.splitext(os.path.basename(video_path))[0])
    split_video_to_frames(video_path, output_subdir, frame_interval)
