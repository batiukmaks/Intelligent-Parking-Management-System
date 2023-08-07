# Import required libraries
import cv2
import os

# Function to split a video into frames and save them as images
def split_video_to_frames(video_path, output_dir, frame_interval=1):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file for reading
    video_capture = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in the video: {frame_count}")

    frame_number = 0
    while True:
        # Read the next frame from the video
        ret, frame = video_capture.read()
        if not ret:
            break

        # Save every nth frame, where n is the frame_interval
        if frame_number % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{frame_number}.jpg")
            cv2.imwrite(frame_filename, frame)

        frame_number += 1

    # Release the video capture object
    video_capture.release()
    print(f"Extracted {frame_number} frames to {output_dir}")

if __name__ == "__main__":
    # Set the directories and frame_interval
    video_directory = "/home/batiukmaks/Documents/Intelligent-Parking-Management-System/my-parking/videos/car-parked"
    output_directory = "/home/batiukmaks/Documents/Intelligent-Parking-Management-System/my-parking/annotated-data/raw-data/car-parked"
    frame_interval = 3  # Change this to extract frames at different intervals if needed

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Loop through all video files in the specified video_directory
    for video_file in os.listdir(video_directory):
        # Check if the file has the ".MOV" extension
        if video_file.endswith(".MOV"):
            # Create the full path of the video file
            video_path = os.path.join(video_directory, video_file)

            # Create a subdirectory for the frames corresponding to each video
            output_subdir = os.path.join(output_directory, os.path.splitext(video_file)[0])

            # Call the split_video_to_frames function to extract frames from the video
            split_video_to_frames(video_path, output_subdir, frame_interval)
