import cv2
import os

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

        # Save every nth frame, where n is frame_interval
        if frame_number % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{frame_number}.jpg")
            cv2.imwrite(frame_filename, frame)

        frame_number += 1

    video_capture.release()
    print(f"Extracted {frame_number} frames to {output_dir}")

if __name__ == "__main__":
    video_directory = "/home/batiukmaks/Documents/Intelligent-Parking-Management-System/my-parking/videos/car-parked"
    output_directory = "/home/batiukmaks/Documents/Intelligent-Parking-Management-System/my-parking/images/raw-data/car-parked"
    frame_interval = 3  # Change this to extract frames at different intervals if needed

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for video_file in os.listdir(video_directory):
        if video_file.endswith(".MOV"):  # Assuming you have only mp4 videos in the directory
            video_path = os.path.join(video_directory, video_file)
            output_subdir = os.path.join(output_directory, os.path.splitext(video_file)[0])
            split_video_to_frames(video_path, output_subdir, frame_interval)
